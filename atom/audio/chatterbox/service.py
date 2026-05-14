# SPDX-License-Identifier: Apache-2.0
"""Chatterbox TTS service — manages the non-LLM components on CPU.

The speech_encoder, embed_tokens, and conditional_decoder (vocoder) run on CPU
via ONNX Runtime. Only the autoregressive language model runs on ATOM's GPU.

Supports two model variants:
  - Standard: onnx-community/chatterbox-ONNX (LlamaForCausalLM, 30 layers)
  - Turbo: ResembleAI/chatterbox-turbo-ONNX (GPT2LMHeadModel, 24 layers, 1-step decoder)
"""

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

logger = logging.getLogger("atom.audio.chatterbox")

SAMPLE_RATE = 24000
START_SPEECH_TOKEN = 6561
STOP_SPEECH_TOKEN = 6562


class ChatterboxService:
    """Manages Chatterbox TTS preprocessing and postprocessing on CPU.

    This class follows the ColBERT service pattern: it loads independently
    from the main ATOM engine and provides prepare/decode methods that
    sandwich the GPU-accelerated autoregressive generation.
    """

    def __init__(
        self,
        model_dir: str,
        variant: str = "standard",
        onnx_variant: str = "fp16",
        num_threads: int = 4,
    ):
        self.model_dir = Path(model_dir)
        self.variant = variant  # "standard" or "turbo"
        self.onnx_variant = onnx_variant
        self.onnx_dir = self.model_dir / "onnx"
        self.num_threads = num_threads

        self._speech_encoder = None
        self._embed_tokens = None
        self._cond_decoder = None
        self._tokenizer = None
        self._default_voice = None

        # Architecture params (set during load)
        self.num_hidden_layers: int = 0
        self.num_heads: int = 0
        self.head_dim: int = 0
        self.backbone_arch: str = ""  # "llama" or "gpt2"

    def load(self) -> None:
        """Load all ONNX sessions and tokenizer."""
        import onnxruntime
        from transformers import AutoTokenizer

        t0 = time.time()
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = self.num_threads
        opts.intra_op_num_threads = self.num_threads

        logger.info("Loading Chatterbox %s ONNX components from %s", self.variant, self.model_dir)

        self._speech_encoder = onnxruntime.InferenceSession(
            str(self.onnx_dir / "speech_encoder.onnx"), opts,
            providers=["CPUExecutionProvider"],
        )
        self._embed_tokens = onnxruntime.InferenceSession(
            str(self.onnx_dir / "embed_tokens.onnx"), opts,
            providers=["CPUExecutionProvider"],
        )

        lm_name = f"language_model_{self.onnx_variant}" if self.onnx_variant != "fp32" else "language_model"
        lm_path = self.onnx_dir / f"{lm_name}.onnx"
        if not lm_path.exists():
            # Turbo may only have fp16
            lm_path = self.onnx_dir / "language_model_fp16.onnx"
        # We don't load the ONNX LM — ATOM handles that on GPU.
        # But we inspect it briefly to determine architecture params.
        lm_session = onnxruntime.InferenceSession(
            str(lm_path), opts, providers=["CPUExecutionProvider"],
        )
        kv_inputs = [i for i in lm_session.get_inputs() if "past_key_values" in i.name]
        self.num_hidden_layers = len(kv_inputs) // 2
        # Infer head_dim and num_heads from KV shape
        if kv_inputs:
            shape = kv_inputs[0].shape  # [batch, num_heads, seq, head_dim]
            self.num_heads = shape[1] if isinstance(shape[1], int) else 16
            self.head_dim = shape[3] if isinstance(shape[3], int) else 64
        del lm_session  # Free ONNX LM — ATOM runs this on GPU

        self._cond_decoder = onnxruntime.InferenceSession(
            str(self.onnx_dir / "conditional_decoder.onnx"), opts,
            providers=["CPUExecutionProvider"],
        )

        self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))

        # Detect architecture
        embed_input_names = [i.name for i in self._embed_tokens.get_inputs()]
        if "exaggeration" in embed_input_names:
            self.backbone_arch = "llama"
        else:
            self.backbone_arch = "gpt2"

        # Load default voice
        voice_path = self.model_dir / "default_voice.wav"
        if not voice_path.exists():
            # Turbo doesn't ship default_voice.wav — check parent dirs
            for candidate in [
                self.model_dir.parent.parent / "default_voice.wav",
                Path("/home/local/Projects/models/huggingface/models--onnx-community--chatterbox-ONNX") / "snapshots",
            ]:
                if candidate.exists():
                    # Find first .wav
                    for f in candidate.rglob("default_voice.wav"):
                        voice_path = f
                        break

        if voice_path.exists():
            audio_values, sr = sf.read(str(voice_path), dtype="float32")
            if sr != SAMPLE_RATE:
                ratio = SAMPLE_RATE / sr
                new_len = int(len(audio_values) * ratio)
                audio_values = np.interp(
                    np.linspace(0, len(audio_values) - 1, new_len),
                    np.arange(len(audio_values)),
                    audio_values,
                ).astype(np.float32)
            self._default_voice = audio_values[np.newaxis, :].astype(np.float32)
            logger.info("Loaded default voice: %.1fs", len(audio_values) / SAMPLE_RATE)

        elapsed = time.time() - t0
        logger.info(
            "Chatterbox %s loaded in %.1fs (arch=%s, layers=%d, heads=%d, head_dim=%d)",
            self.variant, elapsed, self.backbone_arch,
            self.num_hidden_layers, self.num_heads, self.head_dim,
        )

    def encode_reference(
        self,
        audio_path: Optional[str] = None,
        audio_array: Optional[np.ndarray] = None,
    ) -> dict:
        """Run speech encoder on reference audio.

        Returns dict with: cond_emb, prompt_token, ref_x_vector, prompt_feat
        """
        if audio_array is not None:
            audio_values = audio_array
            if audio_values.ndim == 1:
                audio_values = audio_values[np.newaxis, :]
        elif audio_path is not None:
            audio_values, sr = sf.read(audio_path, dtype="float32")
            if sr != SAMPLE_RATE:
                ratio = SAMPLE_RATE / sr
                new_len = int(len(audio_values) * ratio)
                audio_values = np.interp(
                    np.linspace(0, len(audio_values) - 1, new_len),
                    np.arange(len(audio_values)),
                    audio_values,
                ).astype(np.float32)
            audio_values = audio_values[np.newaxis, :].astype(np.float32)
        else:
            if self._default_voice is None:
                raise ValueError("No reference audio provided and no default voice loaded")
            audio_values = self._default_voice

        cond_emb, prompt_token, ref_x_vector, prompt_feat = self._speech_encoder.run(
            None, {"audio_values": audio_values.astype(np.float32)}
        )
        return {
            "cond_emb": cond_emb,
            "prompt_token": prompt_token,
            "ref_x_vector": ref_x_vector,
            "prompt_feat": prompt_feat,
        }

    def prepare_inputs(
        self,
        text: str,
        ref_data: dict,
        exaggeration: float = 0.5,
    ) -> dict:
        """Tokenize text, run embed_tokens, prepend conditioning embedding.

        Returns dict with: inputs_embeds, input_ids, attention_mask
        """
        input_ids = self._tokenizer(text, return_tensors="np")["input_ids"].astype(np.int64)

        # Build embed_tokens inputs
        embed_input_names = [i.name for i in self._embed_tokens.get_inputs()]
        ort_embed_inputs = {"input_ids": input_ids}

        if "position_ids" in embed_input_names:
            position_ids = np.where(
                input_ids >= START_SPEECH_TOKEN, 0,
                np.arange(input_ids.shape[1])[np.newaxis, :] - 1
            )
            ort_embed_inputs["position_ids"] = position_ids

        if "exaggeration" in embed_input_names:
            ort_embed_inputs["exaggeration"] = np.array([exaggeration], dtype=np.float32)

        inputs_embeds = self._embed_tokens.run(None, ort_embed_inputs)[0]

        # Prepend conditioning embedding from speech encoder
        cond_emb = ref_data["cond_emb"]
        full_embeds = np.concatenate((cond_emb, inputs_embeds), axis=1).astype(np.float32)

        return {
            "inputs_embeds": full_embeds,
            "input_ids": input_ids,
            "seq_len": full_embeds.shape[1],
        }

    def embed_single_token(self, token_id: np.ndarray) -> np.ndarray:
        """Embed a single token for autoregressive generation steps."""
        embed_input_names = [i.name for i in self._embed_tokens.get_inputs()]
        ort_inputs = {"input_ids": token_id}
        if "position_ids" in embed_input_names:
            ort_inputs["position_ids"] = np.zeros_like(token_id)
        if "exaggeration" in embed_input_names:
            ort_inputs["exaggeration"] = np.array([0.5], dtype=np.float32)
        return self._embed_tokens.run(None, ort_inputs)[0]

    def decode_speech(
        self,
        speech_tokens: np.ndarray,
        ref_data: dict,
    ) -> np.ndarray:
        """Run conditional decoder (vocoder) to produce waveform.

        Args:
            speech_tokens: Generated speech tokens (excluding start/stop tokens,
                           concatenated with prompt_token).
            ref_data: Output from encode_reference().

        Returns:
            1-D float32 numpy array of audio samples at 24kHz.
        """
        # Concatenate prompt tokens with generated speech tokens
        full_tokens = np.concatenate(
            [ref_data["prompt_token"], speech_tokens], axis=1
        )

        wav = self._cond_decoder.run(None, {
            "speech_tokens": full_tokens,
            "speaker_embeddings": ref_data["ref_x_vector"],
            "speaker_features": ref_data["prompt_feat"],
        })[0]

        return np.squeeze(wav, axis=0)

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE
