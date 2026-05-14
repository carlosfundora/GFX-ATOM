# SPDX-License-Identifier: Apache-2.0
"""WebSocket handler for streaming text input TTS.

Accepts text incrementally via WebSocket, buffers and splits at sentence
boundaries, and generates audio per sentence using the TTS engine.

Protocol:
    Client -> Server:
        {"type": "session.config", ...}   # Session config (sent once first)
        {"type": "input.text", "text": "..."} # Text chunks
        {"type": "input.done"}            # End of input

    Server -> Client:
        {"type": "audio.start", "sentence_index": 0, "sentence_text": "...", "format": "wav"}
        <binary frame: audio bytes>
        {"type": "audio.done", "sentence_index": 0}
        {"type": "session.done", "total_sentences": N}
        {"type": "error", "message": "..."}

Ported from vLLM-Omni's serving_speech_stream.py.
"""

import asyncio
import json
import logging

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from atom.audio.protocol import AudioSpeechRequest, StreamingSpeechSessionConfig
from atom.audio.text_splitter import SPLIT_CLAUSE, SPLIT_SENTENCE, SentenceSplitter
from atom.audio.utils import audio_to_bytes

logger = logging.getLogger("atom.audio.stream")

_DEFAULT_IDLE_TIMEOUT = 30.0  # seconds
_DEFAULT_CONFIG_TIMEOUT = 10.0  # seconds
_MAX_CONFIG_MESSAGE_SIZE = 4 * 1024 * 1024  # allow large ref_audio payloads
_MAX_INPUT_TEXT_MESSAGE_SIZE = 128 * 1024


class StreamingSpeechHandler:
    """Handles WebSocket sessions for streaming text-input TTS.

    Each WebSocket connection is an independent session. Text arrives
    incrementally, is split at sentence boundaries, and audio is generated
    per sentence using the SpeechServing pipeline.
    """

    def __init__(
        self,
        speech_service: "SpeechServing",
        idle_timeout: float = _DEFAULT_IDLE_TIMEOUT,
        config_timeout: float = _DEFAULT_CONFIG_TIMEOUT,
    ) -> None:
        from atom.entrypoints.openai.serving_speech import SpeechServing
        self._speech_service: SpeechServing = speech_service
        self._idle_timeout = idle_timeout
        self._config_timeout = config_timeout

    async def handle_session(self, websocket: WebSocket) -> None:
        """Main session loop for a single WebSocket connection."""
        await websocket.accept()

        try:
            # 1. Wait for session.config
            config = await self._receive_config(websocket)
            if config is None:
                return

            boundary_re = SPLIT_CLAUSE if config.split_granularity == "clause" else SPLIT_SENTENCE
            splitter = SentenceSplitter(boundary_re=boundary_re)
            sentence_index = 0

            # 2. Receive text chunks until input.done
            while True:
                try:
                    raw = await asyncio.wait_for(
                        websocket.receive_text(),
                        timeout=self._idle_timeout,
                    )
                except asyncio.TimeoutError:
                    await self._send_error(websocket, "Idle timeout exceeded")
                    return

                if len(raw) > _MAX_INPUT_TEXT_MESSAGE_SIZE:
                    await self._send_error(
                        websocket,
                        f"Message exceeds max size ({_MAX_INPUT_TEXT_MESSAGE_SIZE} bytes)",
                    )
                    return

                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    await self._send_error(websocket, "Invalid JSON")
                    return

                msg_type = msg.get("type")

                if msg_type == "input.text":
                    text = msg.get("text", "")
                    if not text:
                        continue

                    try:
                        sentences = splitter.add_text(text)
                    except ValueError as e:
                        await self._send_error(websocket, str(e))
                        return

                    for sentence in sentences:
                        await self._generate_and_send(
                            websocket, config, sentence, sentence_index,
                        )
                        sentence_index += 1

                elif msg_type == "input.done":
                    # Flush remaining text
                    remaining = splitter.flush()
                    if remaining:
                        await self._generate_and_send(
                            websocket, config, remaining, sentence_index,
                        )
                        sentence_index += 1

                    # Send session.done
                    await websocket.send_json({
                        "type": "session.done",
                        "total_sentences": sentence_index,
                    })
                    return

                else:
                    await self._send_error(
                        websocket,
                        f"Unknown message type: {msg_type}",
                    )
                    return

        except WebSocketDisconnect:
            logger.debug("WebSocket disconnected")
        except Exception as e:
            logger.error("WebSocket session error: %s", e, exc_info=True)
            try:
                await self._send_error(websocket, f"Internal error: {e}")
            except Exception:
                pass

    async def _receive_config(
        self, websocket: WebSocket,
    ) -> StreamingSpeechSessionConfig | None:
        """Wait for and parse the session.config message."""
        try:
            raw = await asyncio.wait_for(
                websocket.receive_text(),
                timeout=self._config_timeout,
            )
        except asyncio.TimeoutError:
            await self._send_error(websocket, "Config timeout exceeded")
            return None

        if len(raw) > _MAX_CONFIG_MESSAGE_SIZE:
            await self._send_error(
                websocket,
                f"Config message exceeds max size ({_MAX_CONFIG_MESSAGE_SIZE} bytes)",
            )
            return None

        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            await self._send_error(websocket, "Invalid JSON in config")
            return None

        if msg.get("type") != "session.config":
            await self._send_error(
                websocket,
                f"Expected session.config, got {msg.get('type')}",
            )
            return None

        try:
            config = StreamingSpeechSessionConfig(**{
                k: v for k, v in msg.items() if k != "type"
            })
        except ValidationError as e:
            await self._send_error(websocket, f"Invalid config: {e}")
            return None

        return config

    async def _generate_and_send(
        self,
        websocket: WebSocket,
        config: StreamingSpeechSessionConfig,
        sentence: str,
        sentence_index: int,
    ) -> None:
        """Generate audio for a sentence and send it over WebSocket."""
        fmt = config.response_format

        # Send audio.start
        await websocket.send_json({
            "type": "audio.start",
            "sentence_index": sentence_index,
            "sentence_text": sentence,
            "format": fmt,
        })

        # Build a speech request from the config
        request = AudioSpeechRequest(
            input=sentence,
            model=config.model,
            voice=config.voice,
            response_format=fmt,
            speed=config.speed,
            ref_audio=config.ref_audio,
            ref_text=config.ref_text,
            max_new_tokens=config.max_new_tokens,
            stream=False,  # We handle streaming at the WebSocket level
        )

        try:
            _, engine = self._speech_service._resolve_engine(request.model)
            self._speech_service._apply_uploaded_speaker(request)
            wav, metrics = self._speech_service._run_engine(engine, request)
            sample_rate = self._speech_service._get_sample_rate(engine)

            if config.stream_audio and fmt == "pcm":
                # Stream raw PCM chunks
                pcm = (wav * 32767).clip(-32768, 32767).astype(np.int16)
                chunk_size = sample_rate  # 1 second
                for start in range(0, len(pcm), chunk_size):
                    chunk = pcm[start : start + chunk_size].tobytes()
                    await websocket.send_bytes(chunk)
            else:
                audio_bytes, _ = audio_to_bytes(wav, sample_rate, fmt)
                await websocket.send_bytes(audio_bytes)
        except Exception as e:
            logger.error("Generation failed for sentence %d: %s", sentence_index, e)
            await self._send_error(
                websocket,
                f"Generation failed for sentence {sentence_index}: {e}",
            )
            return

        # Send audio.done
        await websocket.send_json({
            "type": "audio.done",
            "sentence_index": sentence_index,
        })

    async def _send_error(self, websocket: WebSocket, message: str) -> None:
        """Send an error message over WebSocket."""
        try:
            await websocket.send_json({
                "type": "error",
                "message": message,
            })
        except Exception:
            pass
