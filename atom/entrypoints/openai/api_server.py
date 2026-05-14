# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
ATOM OpenAI-compatible API Server.

FastAPI-based server implementing OpenAI-compatible endpoints for chat
completions and text completions, with reasoning content separation for
thinking models (Kimi-K2, DeepSeek-R1, Qwen3, etc.).

Usage:
    python -m atom.entrypoints.openai_server --model <model> [options]
"""

import argparse
import asyncio
import json
import logging
import os
import time
import uuid
from asyncio import AbstractEventLoop
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import uvicorn
from atom.model_engine.arg_utils import EngineArgs
from atom.model_engine.request import RequestOutput
from atom.retrieval import ColbertService, is_colbert_model_spec
from atom.retrieval.colbert import DEFAULT_MANIFEST_ROOT
from atom.sampling_params import SamplingParams
from fastapi import FastAPI, HTTPException, Request, UploadFile, WebSocket
from fastapi.responses import JSONResponse, StreamingResponse
from transformers import AutoTokenizer

from .protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    EmbeddingObject,
    EmbeddingRequest,
    EmbeddingResponse,
    ModelCard,
    ModelList,
    RerankRequest,
    RerankResponse,
    RerankResult,
)
from .serving_chat import (
    build_chat_response,
    build_chat_response_multi,
    stream_chat_response,
    stream_chat_response_fanout,
)
from .serving_completion import (
    build_completion_response,
    build_completion_response_multi,
    stream_completion_response,
    stream_completion_response_fanout,
)
from atom.audio.protocol import (
    AudioSpeechRequest,
    BatchSpeechRequest,
    VoiceDeleteResponse,
    VoiceUploadResponse,
)

# Configure logging
logger = logging.getLogger("atom")

# Constants
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = int(
    os.environ.get(
        "ATOM_OPENAI_SERVER_PORT",
        os.environ.get("PORT_EMBED_POOL", "9299"),
    )
)


# ============================================================================
# Global State
# ============================================================================

engine = None
tokenizer: Optional[AutoTokenizer] = None
retrieval_service: Optional[ColbertService] = None
speech_serving = None  # SpeechServing instance (if TTS enabled)
lfm25_audio_client = None  # LFM2.5-Audio client for STT / speech-to-speech
model_name: str = ""
allowed_model_names: set[str] = set()
embedding_mode: bool = False
embedding_pooling: str = "last"
embedding_default_dimensions: Optional[int] = None
embedding_allowed_dimensions: set[int] = set()
default_chat_template_kwargs: Dict[str, Any] = {}
_stream_queues: Dict[str, asyncio.Queue] = {}
_seq_id_to_request_id: Dict[int, str] = {}
_stream_loops: Dict[str, AbstractEventLoop] = {}
_request_start_times: Dict[str, float] = {}
_request_logger: Optional[logging.Logger] = None


# ============================================================================
# Request/Response Logging
# ============================================================================


def _log_request_event(event_type: str, request_id: str, data: Any) -> None:
    """Write a JSONL entry to the request log file (if enabled)."""
    if _request_logger is None:
        return
    entry = {
        "timestamp": time.time(),
        "request_id": request_id,
        "type": event_type,
        "data": data,
    }
    _request_logger.info(json.dumps(entry, default=str))


async def _logged_stream(
    gen: AsyncGenerator[str, None], request_id: str
) -> AsyncGenerator[str, None]:
    """Wrap a streaming generator to log each SSE chunk."""
    async for chunk in gen:
        if _request_logger is not None and chunk.startswith("data: "):
            payload = chunk[6:].strip()
            if payload != "[DONE]":
                _log_request_event("stream_chunk", request_id, json.loads(payload))
            else:
                _log_request_event("stream_done", request_id, None)
        yield chunk


# ============================================================================
# Engine Interface
# ============================================================================


def _build_sampling_params(
    temperature: float,
    max_tokens: int,
    stop_strings: Optional[List[str]],
    ignore_eos: bool,
    top_k: int = -1,
    top_p: float = 1.0,
    n: int = 1,
) -> SamplingParams:
    return SamplingParams(
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        max_tokens=max_tokens,
        stop_strings=stop_strings,
        ignore_eos=ignore_eos,
        n=n,
    )


def _coerce_n(requested_n: Optional[int], temperature: Optional[float]) -> int:
    """Return an effective ``n`` for a request.

    * ``None``/``<1`` coerce to ``1`` (matches OpenAI default).
    * ``n > 1`` combined with greedy sampling (``temperature <= 0``) is
      collapsed to ``1`` because all siblings would produce identical
      outputs — other runtimes (vLLM, TGI) silently do the same, and it
      avoids wasting KV cache on duplicate decodes.
    """
    n = requested_n if requested_n is not None else 1
    try:
        n = int(n)
    except (TypeError, ValueError):
        n = 1
    if n < 1:
        n = 1
    if n > 1 and (temperature is None or temperature <= 0.0):
        logger.info(
            "n=%s requested with temperature=%s; collapsing to n=1 because "
            "greedy sampling would produce identical siblings.",
            n,
            temperature,
        )
        n = 1
    return n


def _send_stream_chunk_direct(
    request_output: RequestOutput,
    request_id: str,
    stream_queue: asyncio.Queue,
    loop: AbstractEventLoop,
) -> None:
    """Send stream chunk directly to the queue."""
    global tokenizer

    new_text = tokenizer.decode(request_output.output_tokens, skip_special_tokens=True)
    started_at = _request_start_times.get(request_id)
    chunk_data = {
        "text": new_text,
        "token_ids": request_output.output_tokens,
        "finished": request_output.finished,
        "finish_reason": request_output.finish_reason,
        "finished_at": time.time(),
        "started_at": started_at,
    }
    if getattr(request_output, "kv_transfer_params_output", None):
        chunk_data["kv_transfer_params"] = request_output.kv_transfer_params_output
    loop.call_soon_threadsafe(stream_queue.put_nowait, chunk_data)


def _send_stream_chunk_tagged(
    request_output: RequestOutput,
    sibling_index: int,
    stream_queue: asyncio.Queue,
    loop: AbstractEventLoop,
) -> None:
    """Variant of :func:`_send_stream_chunk_direct` for fan-out siblings.

    Pushes ``(sibling_index, chunk_data)`` tuples onto a single shared
    queue so the merge-stream consumer in :mod:`serving_chat` /
    :mod:`serving_completion` can demultiplex by index.
    """
    global tokenizer

    new_text = tokenizer.decode(request_output.output_tokens, skip_special_tokens=True)
    chunk_data = {
        "text": new_text,
        "token_ids": request_output.output_tokens,
        "finished": request_output.finished,
        "finish_reason": request_output.finish_reason,
    }
    if getattr(request_output, "kv_transfer_params_output", None):
        chunk_data["kv_transfer_params"] = request_output.kv_transfer_params_output
    loop.call_soon_threadsafe(stream_queue.put_nowait, (sibling_index, chunk_data))


async def generate_async(
    prompt: str,
    sampling_params: SamplingParams,
    request_id: str,
    kv_transfer_params: Optional[Dict[str, Any]] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Generate text asynchronously for non-streaming requests."""
    global engine, tokenizer

    token_queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    started_at = time.time()
    first_token_at: Optional[float] = None
    last_token_at: Optional[float] = None
    all_token_ids: List[int] = []
    finish_reason: Optional[str] = None
    seq = None
    kv_transfer_output_meta_info = None

    def completion_callback(request_output: RequestOutput):
        nonlocal kv_transfer_output_meta_info
        kv_transfer_output_meta_info = getattr(
            request_output, "kv_transfer_params_output", None
        )
        now = time.time()
        loop.call_soon_threadsafe(
            token_queue.put_nowait,
            {
                "token_ids": request_output.output_tokens,
                "finished": request_output.finished,
                "finish_reason": request_output.finish_reason,
                "ts": now,
            },
        )

    def do_preprocess():
        return engine.io_processor.preprocess(
            prompt,
            sampling_params,
            stream_callback=completion_callback,
            kv_transfer_params=kv_transfer_params,
        )

    seq = await loop.run_in_executor(None, do_preprocess)
    engine.core_mgr.add_request([seq])

    while True:
        item = await token_queue.get()
        token_ids = item.get("token_ids") or []
        if token_ids:
            if first_token_at is None:
                first_token_at = item.get("ts", time.time())
            last_token_at = item.get("ts", time.time())
            all_token_ids.extend(token_ids)
        if item.get("finished", False):
            finish_reason = item.get("finish_reason")
            break

    text = tokenizer.decode(all_token_ids, skip_special_tokens=True)
    num_tokens_input = (
        seq.num_prompt_tokens if seq is not None else len(tokenizer.encode(prompt))
    )
    num_tokens_output = len(all_token_ids)
    finished_at = time.time()
    latency = finished_at - started_at
    ttft = (first_token_at - started_at) if first_token_at is not None else 0.0
    tpot = (
        (last_token_at - first_token_at) / (num_tokens_output - 1)
        if first_token_at is not None
        and last_token_at is not None
        and num_tokens_output > 1
        else 0.0
    )

    response = {
        "text": text,
        "token_ids": all_token_ids,
        "finish_reason": finish_reason,
        "num_tokens_input": num_tokens_input,
        "num_tokens_output": num_tokens_output,
        "ttft": ttft,
        "tpot": tpot,
        "latency": latency,
    }
    if kv_transfer_output_meta_info is not None:
        response["kv_transfer_output_meta_info"] = kv_transfer_output_meta_info
    yield response


async def generate_async_fanout(
    prompt: str,
    sampling_params: SamplingParams,
    request_id: str,
    kv_transfer_params: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Non-streaming n>1 path: fan out N siblings and await all of them.

    Returns a list of per-sibling output dicts in the same shape as
    :func:`generate_async` yields for n==1, so response builders can treat
    each entry the same way.
    """
    global engine, tokenizer

    n = int(sampling_params.n)
    assert n >= 1

    shared_queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    started_at = time.time()
    per_tokens: List[List[int]] = [[] for _ in range(n)]
    per_first_token_at: List[Optional[float]] = [None] * n
    per_last_token_at: List[Optional[float]] = [None] * n
    per_finish_reason: List[Optional[str]] = [None] * n
    finished = [False] * n

    def make_callback(idx: int):
        def _cb(request_output: RequestOutput) -> None:
            now = time.time()
            loop.call_soon_threadsafe(
                shared_queue.put_nowait,
                (
                    idx,
                    {
                        "token_ids": request_output.output_tokens,
                        "finished": request_output.finished,
                        "finish_reason": request_output.finish_reason,
                        "ts": now,
                    },
                ),
            )

        return _cb

    stream_callbacks = [make_callback(i) for i in range(n)]

    def do_preprocess():
        return engine.io_processor.preprocess_fanout(
            prompt,
            sampling_params,
            stream_callbacks=stream_callbacks,
            kv_transfer_params=kv_transfer_params,
            parent_request_id=request_id,
        )

    seqs = await loop.run_in_executor(None, do_preprocess)
    engine.core_mgr.add_request(seqs)
    num_tokens_input = seqs[0].num_prompt_tokens

    while not all(finished):
        idx, item = await shared_queue.get()
        if finished[idx]:
            continue
        tokens = item.get("token_ids") or []
        if tokens:
            if per_first_token_at[idx] is None:
                per_first_token_at[idx] = item.get("ts", time.time())
            per_last_token_at[idx] = item.get("ts", time.time())
            per_tokens[idx].extend(tokens)
        if item.get("finished", False):
            per_finish_reason[idx] = item.get("finish_reason")
            finished[idx] = True

    finished_at = time.time()
    outputs: List[Dict[str, Any]] = []
    for i in range(n):
        num_tokens_output = len(per_tokens[i])
        ttft = (
            per_first_token_at[i] - started_at
            if per_first_token_at[i] is not None
            else 0.0
        )
        tpot = (
            (per_last_token_at[i] - per_first_token_at[i]) / (num_tokens_output - 1)
            if per_first_token_at[i] is not None
            and per_last_token_at[i] is not None
            and num_tokens_output > 1
            else 0.0
        )
        outputs.append(
            {
                "text": tokenizer.decode(per_tokens[i], skip_special_tokens=True),
                "token_ids": per_tokens[i],
                "finish_reason": per_finish_reason[i],
                "num_tokens_input": num_tokens_input,
                "num_tokens_output": num_tokens_output,
                "ttft": ttft,
                "tpot": tpot,
                "latency": finished_at - started_at,
            }
        )
    return outputs


def validate_model(requested_model: Optional[str]) -> None:
    """Validate that the requested model matches the server's model."""
    if requested_model is None:
        return
    if requested_model != model_name and requested_model not in allowed_model_names:
        raise HTTPException(
            status_code=400,
            detail=f"Requested model '{requested_model}' does not match "
            f"server model '{model_name}'",
        )


def _ensure_endpoint_supported(endpoint: str) -> None:
    if retrieval_service is not None and endpoint not in {"embeddings", "rerank"}:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Model '{model_name}' is a retrieval-only ColBERT model and "
                f"does not support '{endpoint}'."
            ),
        )
    if retrieval_service is None and endpoint == "embeddings" and not embedding_mode:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Model '{model_name}' does not support '{endpoint}'. "
                "Start ATOM with a ColBERT retrieval model to use these routes."
            ),
        )
    if retrieval_service is None and endpoint == "rerank":
        raise HTTPException(
            status_code=400,
            detail=(
                f"Model '{model_name}' does not support '{endpoint}'. "
                "Start ATOM with a ColBERT retrieval model to use this route."
            ),
        )


def _normalize_embedding_inputs(input_text: str | list[str]) -> list[str]:
    if isinstance(input_text, str):
        return [input_text]
    return list(input_text)


async def setup_streaming_request(
    prompt: str,
    sampling_params: SamplingParams,
    request_id: str,
    kv_transfer_params: Optional[Dict[str, Any]] = None,
) -> Tuple[int, asyncio.Queue]:
    """Set up a streaming request with the engine."""
    global engine, _stream_queues, _seq_id_to_request_id
    global _stream_loops, _request_start_times

    stream_queue: asyncio.Queue = asyncio.Queue()
    stream_loop = asyncio.get_running_loop()
    _stream_queues[request_id] = stream_queue
    _stream_loops[request_id] = stream_loop
    _request_start_times[request_id] = time.time()

    def stream_callback(request_output: RequestOutput) -> None:
        _send_stream_chunk_direct(request_output, request_id, stream_queue, stream_loop)

    executor_loop = asyncio.get_event_loop()

    def do_preprocess():
        seq = engine.io_processor.preprocess(
            prompt,
            sampling_params,
            stream_callback=stream_callback,
            kv_transfer_params=kv_transfer_params,
        )
        _seq_id_to_request_id[seq.id] = request_id
        return seq

    seq = await executor_loop.run_in_executor(None, do_preprocess)
    seq_id = seq.id

    logger.info(f"API: Created request_id={request_id}, seq_id={seq_id}")
    engine.core_mgr.add_request([seq])

    return seq_id, stream_queue


def cleanup_streaming_request(request_id: str, seq_id: int) -> None:
    """Clean up resources for a streaming request.

    Safe to call multiple times for the same ``request_id`` with different
    ``seq_id`` values (as happens in fan-out cleanup): the per-request
    dicts use ``dict.pop(..., None)`` so repeated removal is a no-op.
    """
    global engine, _stream_queues, _seq_id_to_request_id
    global _stream_loops, _request_start_times

    _stream_queues.pop(request_id, None)
    _seq_id_to_request_id.pop(seq_id, None)
    _stream_loops.pop(request_id, None)
    _request_start_times.pop(request_id, None)
    engine.io_processor.requests.pop(seq_id, None)


async def setup_streaming_request_fanout(
    prompt: str,
    sampling_params: SamplingParams,
    request_id: str,
    kv_transfer_params: Optional[Dict[str, Any]] = None,
) -> Tuple[List[int], asyncio.Queue]:
    """Fan-out variant of :func:`setup_streaming_request`.

    Creates ``sampling_params.n`` sibling sequences sharing one output
    queue. Every callback pushes ``(sibling_index, chunk_data)`` tuples so
    the merge-stream consumer can rewrite ``choices[0].index`` correctly.
    """
    global engine, _stream_queues, _seq_id_to_request_id
    global _stream_loops, _request_start_times

    n = int(sampling_params.n)
    assert n >= 1

    shared_queue: asyncio.Queue = asyncio.Queue()
    stream_loop = asyncio.get_running_loop()
    _stream_queues[request_id] = shared_queue
    _stream_loops[request_id] = stream_loop
    _request_start_times[request_id] = time.time()

    def make_callback(idx: int):
        def _cb(request_output: RequestOutput) -> None:
            _send_stream_chunk_tagged(request_output, idx, shared_queue, stream_loop)

        return _cb

    stream_callbacks = [make_callback(i) for i in range(n)]

    executor_loop = asyncio.get_event_loop()

    def do_preprocess():
        seqs = engine.io_processor.preprocess_fanout(
            prompt,
            sampling_params,
            stream_callbacks=stream_callbacks,
            kv_transfer_params=kv_transfer_params,
            parent_request_id=request_id,
        )
        for seq in seqs:
            _seq_id_to_request_id[seq.id] = request_id
        return seqs

    seqs = await executor_loop.run_in_executor(None, do_preprocess)
    seq_ids = [seq.id for seq in seqs]
    logger.info(
        f"API: Created fan-out request_id={request_id}, n={n}, seq_ids={seq_ids}"
    )
    engine.core_mgr.add_request(seqs)
    return seq_ids, shared_queue


# ============================================================================
# FastAPI Application
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    logger.info("Server started successfully and ready to accept requests")
    yield
    logger.info("Server shutting down, releasing resources...")
    if engine is not None:
        engine.close()


app = FastAPI(title="ATOM OpenAI API Server", lifespan=lifespan)


# ---- Error handlers ----


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "message": str(exc),
                "type": "invalid_request_error",
                "code": 400,
            }
        },
    )


@app.exception_handler(Exception)
async def general_error_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": str(exc),
                "type": "internal_server_error",
                "code": 500,
            }
        },
    )


# ---- Endpoints ----


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Handle chat completion requests (OpenAI-compatible)."""
    global engine, tokenizer, model_name

    _ensure_endpoint_supported("chat/completions")
    validate_model(request.model)

    try:
        messages = request.get_messages()

        merged_kwargs = dict(default_chat_template_kwargs)
        if request.chat_template_kwargs:
            merged_kwargs.update(request.chat_template_kwargs)
        merged_kwargs["tokenize"] = False
        merged_kwargs["add_generation_prompt"] = True
        # Pass tools so the chat template can inject tool declarations
        if request.tools:
            merged_kwargs["tools"] = request.tools

        prompt = tokenizer.apply_chat_template(
            [msg.to_template_dict() for msg in messages],
            **merged_kwargs,
        )

        effective_n = _coerce_n(request.n, request.temperature)
        sampling_params = _build_sampling_params(
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stop_strings=request.stop,
            ignore_eos=request.ignore_eos,
            top_k=request.top_k,
            top_p=request.top_p,
            n=effective_n,
        )

        request_id = f"chatcmpl-{uuid.uuid4().hex}"

        _log_request_event("request", request_id, request.model_dump())

        # Streaming
        if request.stream:
            if effective_n > 1:
                seq_ids, stream_queue = await setup_streaming_request_fanout(
                    prompt, sampling_params, request_id
                )
                gen = stream_chat_response_fanout(
                    request_id,
                    model_name,
                    prompt,
                    stream_queue,
                    seq_ids,
                    tokenizer,
                    cleanup_streaming_request,
                )
            else:
                seq_id, stream_queue = await setup_streaming_request(
                    prompt, sampling_params, request_id
                )
                gen = stream_chat_response(
                    request_id,
                    model_name,
                    prompt,
                    stream_queue,
                    seq_id,
                    tokenizer,
                    cleanup_streaming_request,
                )
            return StreamingResponse(
                _logged_stream(gen, request_id),
                media_type="text/event-stream",
            )

        # Non-streaming
        if effective_n > 1:
            outputs = await generate_async_fanout(prompt, sampling_params, request_id)
            if not outputs:
                raise RuntimeError("No output generated")
            resp = build_chat_response_multi(request_id, model_name, outputs)
        else:
            final_output = None
            async for output in generate_async(prompt, sampling_params, request_id):
                final_output = output
            if final_output is None:
                raise RuntimeError("No output generated")
            resp = build_chat_response(
                request_id, model_name, final_output["text"], final_output
            )
        _log_request_event("response", request_id, resp.model_dump())
        return resp

    except ValueError as e:
        logger.error(f"Validation error in chat_completions: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in chat_completions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    """Handle text completion requests (OpenAI-compatible)."""
    global engine, tokenizer, model_name

    _ensure_endpoint_supported("completions")
    validate_model(request.model)

    try:
        effective_n = _coerce_n(request.n, request.temperature)
        sampling_params = _build_sampling_params(
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stop_strings=request.stop,
            ignore_eos=request.ignore_eos,
            top_k=request.top_k,
            top_p=request.top_p,
            n=effective_n,
        )

        request_id = f"cmpl-{uuid.uuid4().hex}"

        _log_request_event("request", request_id, request.model_dump())

        # Streaming
        if request.stream:
            if effective_n > 1:
                seq_ids, stream_queue = await setup_streaming_request_fanout(
                    request.prompt,
                    sampling_params,
                    request_id,
                    kv_transfer_params=request.kv_transfer_params,
                )
                gen = stream_completion_response_fanout(
                    request_id,
                    model_name,
                    request.prompt,
                    stream_queue,
                    seq_ids,
                    tokenizer,
                    cleanup_streaming_request,
                )
            else:
                seq_id, stream_queue = await setup_streaming_request(
                    request.prompt,
                    sampling_params,
                    request_id,
                    kv_transfer_params=request.kv_transfer_params,
                )
                gen = stream_completion_response(
                    request_id,
                    model_name,
                    request.prompt,
                    stream_queue,
                    seq_id,
                    tokenizer,
                    cleanup_streaming_request,
                )
            return StreamingResponse(
                _logged_stream(gen, request_id),
                media_type="text/event-stream",
            )

        # Non-streaming
        if effective_n > 1:
            outputs = await generate_async_fanout(
                request.prompt,
                sampling_params,
                request_id,
                kv_transfer_params=request.kv_transfer_params,
            )
            if not outputs:
                raise RuntimeError("No output generated")
            resp = build_completion_response_multi(request_id, model_name, outputs)
        else:
            final_output = None
            async for output in generate_async(
                request.prompt,
                sampling_params,
                request_id,
                kv_transfer_params=request.kv_transfer_params,
            ):
                final_output = output

            if final_output is None:
                raise RuntimeError("No output generated")

            resp = build_completion_response(request_id, model_name, final_output)
        _log_request_event("response", request_id, resp.model_dump())
        return resp

    except ValueError as e:
        logger.error(f"Validation error in completions: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in completions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/embeddings")
async def embeddings(request: EmbeddingRequest):
    """Handle embedding requests."""
    global retrieval_service, model_name, engine

    _ensure_endpoint_supported("embeddings")
    validate_model(request.model)

    texts = _normalize_embedding_inputs(request.input)
    if not texts:
        raise HTTPException(status_code=400, detail="input must not be empty")

    try:
        if retrieval_service is not None:
            embeddings_out, tokens_evaluated = retrieval_service.embed_texts(texts)
            response_model = model_name
        else:
            dimensions = request.dimensions or embedding_default_dimensions
            if dimensions is not None:
                if dimensions <= 0:
                    raise ValueError("dimensions must be positive")
                if (
                    embedding_allowed_dimensions
                    and dimensions not in embedding_allowed_dimensions
                ):
                    raise ValueError(
                        f"dimensions={dimensions} is not supported; allowed dimensions: "
                        f"{sorted(embedding_allowed_dimensions)}"
                    )
            embeddings_out, tokens_evaluated = engine.embed(
                texts,
                pooling=embedding_pooling,
                dimensions=dimensions,
            )
            response_model = model_name
        data = [
            EmbeddingObject(index=index, embedding=embedding)
            for index, embedding in enumerate(embeddings_out)
        ]
        return EmbeddingResponse(
            model=response_model,
            data=data,
            usage={
                "prompt_tokens": tokens_evaluated,
                "total_tokens": tokens_evaluated,
            },
        )
    except ValueError as e:
        logger.error(f"Validation error in embeddings: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in embeddings: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/rerank")
async def rerank(request: RerankRequest):
    """Handle reranking requests for ColBERT retrieval."""
    global retrieval_service, model_name

    _ensure_endpoint_supported("rerank")
    validate_model(request.model)

    if retrieval_service is None:
        raise HTTPException(
            status_code=400,
            detail="No retrieval backend is configured for rerank",
        )

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")
    if not request.documents:
        raise HTTPException(status_code=400, detail="documents must not be empty")

    try:
        results_raw, tokens_evaluated = retrieval_service.rerank(
            request.query,
            request.documents,
            top_n=request.top_k,
        )
        results = [
            RerankResult(
                index=item["index"],
                score=item["score"],
                document=item["document"] if request.return_documents else None,
                meta_info=None,
            )
            for item in results_raw
        ]
        return RerankResponse(
            model=model_name,
            id=request.rid,
            results=results,
            usage={
                "prompt_tokens": tokens_evaluated,
                "total_tokens": tokens_evaluated,
            },
            tokens_evaluated=tokens_evaluated,
        )
    except ValueError as e:
        logger.error(f"Validation error in rerank: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in rerank: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models")
async def list_models():
    """List available models."""
    global model_name, allowed_model_names
    models = sorted(allowed_model_names or {model_name})
    return ModelList(data=[ModelCard(id=item) for item in models])


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


# ============================================================================
# Speech API (OpenAI /v1/audio/speech compatible)
# ============================================================================

_TTS_NOT_ENABLED = (
    "Speech API not enabled. Start with --tts-model, --fish-speech-model, "
    "--voxcpm2-model, or --lfm25-audio."
)


@app.post("/v1/audio/speech")
async def create_speech(request: AudioSpeechRequest):
    """Generate speech audio from text."""
    if speech_serving is None:
        raise HTTPException(status_code=404, detail=_TTS_NOT_ENABLED)
    return await speech_serving.create_speech(request)


@app.post("/v1/audio/transcriptions")
async def create_audio_transcription(
    file: UploadFile,
    model: str | None = None,
    language: str = "auto",
    response_format: str = "json",
):
    """Transcribe audio through the optional LFM2.5-Audio bridge."""
    if lfm25_audio_client is None:
        raise HTTPException(status_code=404, detail="LFM2.5-Audio STT is not enabled. Start with --lfm25-audio.")
    wav_bytes = await file.read()
    try:
        text = await asyncio.to_thread(
            lfm25_audio_client.transcribe_wav_bytes,
            wav_bytes,
            language=language,
        )
    except Exception as exc:
        logger.error("LFM2.5-Audio transcription failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}")
    if response_format == "text":
        return StreamingResponse(iter([text]), media_type="text/plain")
    return {"text": text, "model": model or "lfm2.5-audio"}


@app.post("/v1/audio/transcribe")
async def create_audio_transcribe_alias(
    file: UploadFile,
    model: str | None = None,
    language: str = "auto",
    response_format: str = "json",
):
    """DEMERZEL-compatible alias for `/v1/audio/transcriptions`."""
    return await create_audio_transcription(file, model, language, response_format)


@app.post("/v1/audio/speech/batch")
async def create_speech_batch(request: BatchSpeechRequest):
    """Batch speech synthesis with per-item overrides."""
    if speech_serving is None:
        raise HTTPException(status_code=404, detail=_TTS_NOT_ENABLED)
    result = await speech_serving.create_speech_batch(request)
    return result.model_dump()


@app.get("/v1/audio/voices")
async def list_voices():
    """List available TTS voices (built-in + uploaded + engine-specific)."""
    if speech_serving is None:
        raise HTTPException(status_code=404, detail=_TTS_NOT_ENABLED)
    result = await speech_serving.list_voices()
    return result.model_dump()


@app.post("/v1/audio/voices")
async def upload_voice(
    audio_file: UploadFile,
    consent: str = "agree",
    name: str = "uploaded",
    ref_text: str | None = None,
    speaker_description: str | None = None,
):
    """Upload a reference voice sample for voice cloning."""
    if speech_serving is None:
        raise HTTPException(status_code=404, detail=_TTS_NOT_ENABLED)
    try:
        result = await speech_serving.upload_voice(
            audio_file=audio_file,
            consent=consent,
            name=name,
            ref_text=ref_text,
            speaker_description=speaker_description,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/v1/audio/voices/{name}")
async def delete_voice(name: str):
    """Delete an uploaded voice sample."""
    if speech_serving is None:
        raise HTTPException(status_code=404, detail=_TTS_NOT_ENABLED)
    deleted = await speech_serving.delete_voice(name)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Voice '{name}' not found")
    return VoiceDeleteResponse(name=name).model_dump()


@app.websocket("/v1/audio/speech/stream")
async def speech_stream_ws(websocket: WebSocket):
    """WebSocket endpoint for streaming text-input TTS.

    Protocol:
        Client → {"type": "session.config", ...}
        Client → {"type": "input.text", "text": "..."}
        Client → {"type": "input.done"}
        Server → {"type": "audio.start", ...} + binary audio + {"type": "audio.done", ...}
        Server → {"type": "session.done", "total_sentences": N}
    """
    if speech_serving is None:
        await websocket.close(code=1008, reason=_TTS_NOT_ENABLED)
        return
    from atom.entrypoints.openai.serving_speech_stream import StreamingSpeechHandler
    handler = StreamingSpeechHandler(speech_serving)
    await handler.handle(websocket)


@app.post("/start_profile")
async def start_profile():
    """Start profiling the engine."""
    global engine
    try:
        engine.start_profile()
        return {"status": "success", "message": "Profiling started"}
    except Exception as e:
        logger.error(f"Failed to start profiling: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to start profiling: {str(e)}"
        )


@app.post("/stop_profile")
async def stop_profile():
    """Stop profiling the engine."""
    global engine
    try:
        engine.stop_profile()
        return {
            "status": "success",
            "message": "Profiling stopped. Trace files generated.",
        }
    except Exception as e:
        logger.error(f"Failed to stop profiling: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to stop profiling: {str(e)}"
        )


# ============================================================================
# Main Entry Point
# ============================================================================


def main():
    """Main entry point for the server."""
    global engine, tokenizer, retrieval_service, model_name, allowed_model_names
    global default_chat_template_kwargs, _request_logger
    global embedding_mode, embedding_pooling, embedding_default_dimensions
    global embedding_allowed_dimensions

    parser = argparse.ArgumentParser(description="ATOM OpenAI API Server")
    EngineArgs.add_cli_args(parser)
    parser.add_argument("--host", type=str, default=DEFAULT_HOST, help="Server host")
    parser.add_argument(
        "--server-port",
        type=int,
        default=DEFAULT_PORT,
        help="Server port (note: --port is used for internal engine communication)",
    )
    parser.add_argument(
        "--default-chat-template-kwargs",
        type=str,
        default=None,
        help=(
            "Default kwargs for chat template rendering (JSON string). "
            "Merged with per-request chat_template_kwargs (request wins). "
            "Example: '{\"enable_thinking\": false}'"
        ),
    )
    parser.add_argument(
        "--request-log",
        type=str,
        default=None,
        help="Path to JSONL file for logging all API requests and responses (debug)",
    )
    parser.add_argument(
        "--colbert-manifest-root",
        type=str,
        default=str(DEFAULT_MANIFEST_ROOT),
        help="Root directory containing local model manifests for ColBERT models.",
    )
    parser.add_argument(
        "--colbert-device",
        type=str,
        default=None,
        help="Optional torch device override for ColBERT inference (default: cpu).",
    )
    parser.add_argument(
        "--embedding-mode",
        action="store_true",
        help="Serve /v1/embeddings from ATOM transformer hidden states.",
    )
    parser.add_argument(
        "--embedding-pooling",
        choices=["last", "mean"],
        default="last",
        help="Pooling mode for ATOM dense embeddings.",
    )
    parser.add_argument(
        "--embedding-default-dimensions",
        type=int,
        default=None,
        help="Default dense embedding dimensions when requests omit dimensions.",
    )
    parser.add_argument(
        "--embedding-dimensions",
        type=str,
        default="",
        help="Comma-separated list of allowed dense Matryoshka dimensions.",
    )
    parser.add_argument(
        "--model-alias",
        action="append",
        default=[],
        help="Additional accepted model id. Can be supplied multiple times.",
    )
    # TTS / Speech API arguments
    parser.add_argument(
        "--tts-model",
        type=str,
        default=None,
        help="Path to Chatterbox ONNX model directory to enable /v1/audio/speech.",
    )
    parser.add_argument(
        "--tts-backbone",
        type=str,
        default=None,
        help="Path to HF backbone model for GPU-accelerated TTS. If omitted, uses ONNX CPU.",
    )
    parser.add_argument(
        "--tts-variant",
        choices=["standard", "turbo"],
        default="standard",
        help="Chatterbox variant: standard (Llama 0.5B) or turbo (GPT2 350M).",
    )
    parser.add_argument(
        "--tts-device",
        type=str,
        default="cuda:0",
        help="GPU device for TTS backbone model.",
    )
    parser.add_argument(
        "--tts-dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help="Model dtype for TTS backbone.",
    )
    parser.add_argument(
        "--tts-onnx-variant",
        choices=["fp32", "fp16", "q4", "q4f16", "q8"],
        default="fp16",
        help="ONNX component precision for Chatterbox CPU sessions.",
    )
    parser.add_argument(
        "--tts-onnx-threads",
        type=int,
        default=None,
        help="ONNX Runtime intra-op threads for Chatterbox CPU components. Defaults to physical cores.",
    )
    parser.add_argument(
        "--tts-backend",
        choices=["auto", "onnx", "transformer", "atom_vllm"],
        default=os.environ.get("ATOM_CHATTERBOX_BACKEND", "auto"),
        help="Preferred Chatterbox backend for /v1/audio/speech.",
    )
    parser.add_argument(
        "--tts-vllm-source",
        type=str,
        default=os.environ.get(
            "ATOM_CHATTERBOX_VLLM_SOURCE",
            "/home/local/ai/engines/chatterbox-vllm",
        ),
        help="Path to the chatterbox-vllm source checkout.",
    )
    parser.add_argument(
        "--tts-vllm-max-model-len",
        type=int,
        default=1000,
        help="Maximum T3 sequence length for atom_vllm.",
    )
    parser.add_argument(
        "--tts-vllm-max-batch-size",
        type=int,
        default=10,
        help="Maximum prompt batch size for atom_vllm.",
    )
    parser.add_argument(
        "--tts-vllm-gpu-memory-utilization",
        type=float,
        default=None,
        help="vLLM GPU memory utilization override for atom_vllm.",
    )
    parser.add_argument(
        "--tts-vllm-compile",
        action="store_true",
        help="Allow vLLM compilation for atom_vllm instead of enforcing eager execution.",
    )
    parser.add_argument(
        "--tts-default-voice",
        type=str,
        default=os.environ.get(
            "ATOM_CHATTERBOX_DEFAULT_VOICE",
            "/home/local/ai/projects/DEMERZEL/servers/pipecat-voice/data/kokoro-refs/af_bella.wav",
        ),
        help="Default Chatterbox reference voice. Defaults to DEMERZEL af_bella.",
    )
    parser.add_argument(
        "--fish-speech-model",
        type=str,
        default=None,
        help="Path to Fish Speech S2 Pro model directory (fishaudio/s2-pro). "
        "Enables Fish Speech engine for /v1/audio/speech.",
    )
    parser.add_argument(
        "--voxcpm2-model",
        type=str,
        default=None,
        help="Path to VoxCPM2 model directory (openbmb/VoxCPM2). "
        "Enables VoxCPM2 engine for /v1/audio/speech.",
    )
    parser.add_argument(
        "--lfm25-audio",
        action="store_true",
        help="Enable the LFM2.5-Audio llama.cpp bridge for TTS and STT.",
    )
    parser.add_argument(
        "--lfm25-audio-url",
        type=str,
        default=os.environ.get("ATOM_LFM25_AUDIO_URL", "http://127.0.0.1:30008/v1"),
        help="Base URL for a running llama-liquid-audio-server.",
    )
    args = parser.parse_args()

    if args.request_log:
        _request_logger = logging.getLogger("atom.request_log")
        _request_logger.setLevel(logging.INFO)
        _request_logger.propagate = False
        fh = logging.FileHandler(args.request_log, mode="a")
        fh.setFormatter(logging.Formatter("%(message)s"))
        _request_logger.addHandler(fh)
        logger.info(f"Request logging enabled: {args.request_log}")

    if args.default_chat_template_kwargs:
        default_chat_template_kwargs = json.loads(args.default_chat_template_kwargs)
        logger.info(f"Default chat template kwargs: {default_chat_template_kwargs}")

    colbert_manifest_root = Path(args.colbert_manifest_root)
    if not args.embedding_mode and is_colbert_model_spec(
        args.model, manifest_root=colbert_manifest_root
    ):
        logger.info(f"Initializing ColBERT retrieval backend with model {args.model}...")
        retrieval_service = ColbertService.from_model(
            args.model,
            manifest_root=colbert_manifest_root,
            device=args.colbert_device or "cpu",
        )
        model_name = retrieval_service.model_id
        allowed_model_names = {
            args.model,
            model_name,
            str(retrieval_service.descriptor.weights_path),
        }
        tokenizer = None
        engine = None
    else:
        from atom.model_engine.llm_engine import _load_tokenizer

        embedding_mode = bool(args.embedding_mode)
        embedding_pooling = args.embedding_pooling
        embedding_default_dimensions = args.embedding_default_dimensions
        embedding_allowed_dimensions = {
            int(item.strip())
            for item in args.embedding_dimensions.split(",")
            if item.strip()
        }

        logger.info(f"Loading tokenizer from {args.model}...")
        tokenizer = _load_tokenizer(args.model, args.trust_remote_code)
        model_name = args.model
        allowed_model_names = {model_name, *args.model_alias}

        logger.info(f"Initializing engine with model {args.model}...")
        engine_args = EngineArgs.from_cli_args(args)
        engine = engine_args.create_engine(tokenizer=tokenizer)

    # Initialize TTS engines (multi-model support)
    global speech_serving
    _any_tts = args.tts_model or args.fish_speech_model or args.voxcpm2_model or args.lfm25_audio
    if _any_tts:
        from atom.entrypoints.openai.serving_speech import SpeechServing

        speech_serving = SpeechServing()
        _tts_engines_loaded = []

        # Chatterbox TTS
        if args.tts_model:
            from atom.audio.chatterbox.engine import ChatterboxEngine

            logger.info(
                "Initializing Chatterbox TTS (%s) from %s...",
                args.tts_variant, args.tts_model,
            )
            chatterbox_engine = ChatterboxEngine(
                model_dir=args.tts_model,
                backbone_dir=args.tts_backbone,
                variant=args.tts_variant,
                onnx_variant=args.tts_onnx_variant,
                device=args.tts_device,
                dtype=args.tts_dtype,
                num_threads=args.tts_onnx_threads,
            )
            chatterbox_engine.load()
            speech_serving.register_engine("chatterbox_fallback", chatterbox_engine)
            speech_serving.register_engine("chatterbox_onnx", chatterbox_engine)
            if args.tts_backbone:
                speech_serving.register_engine("chatterbox_transformer", chatterbox_engine)

            default_chatterbox_engine = chatterbox_engine
            if args.tts_backend in ("auto", "atom_vllm"):
                from atom.audio.chatterbox.vllm_backend import ChatterboxAtomVllmEngine

                atom_vllm_engine = ChatterboxAtomVllmEngine(
                    model_dir=args.tts_model,
                    variant=args.tts_variant,
                    device=args.tts_device,
                    source_dir=args.tts_vllm_source,
                    max_model_len=args.tts_vllm_max_model_len,
                    max_batch_size=args.tts_vllm_max_batch_size,
                    gpu_memory_utilization=args.tts_vllm_gpu_memory_utilization,
                    enforce_eager=not args.tts_vllm_compile,
                    default_voice_path=args.tts_default_voice,
                    fallback_engine=chatterbox_engine,
                )
                atom_vllm_engine.load()
                speech_serving.register_engine("atom_vllm", atom_vllm_engine)
                speech_serving.register_engine("chatterbox_atom_vllm", atom_vllm_engine)
                if (
                    args.tts_backend == "atom_vllm"
                    or atom_vllm_engine.unavailable_reason is None
                ):
                    default_chatterbox_engine = atom_vllm_engine

            speech_serving.register_engine(
                "chatterbox",
                default_chatterbox_engine,
                default=True,
            )
            _tts_engines_loaded.append(f"chatterbox:{args.tts_backend}")

        # Fish Speech S2 Pro
        if args.fish_speech_model:
            from atom.models.fish_speech.engine import FishSpeechEngine

            logger.info(
                "Initializing Fish Speech S2 Pro from %s...",
                args.fish_speech_model,
            )
            fish_engine = FishSpeechEngine(
                model_dir=args.fish_speech_model,
                device=args.tts_device,
                dtype=args.tts_dtype,
            )
            fish_engine.load()
            speech_serving.register_engine(
                "fish_speech", fish_engine,
                default=not _tts_engines_loaded,
            )
            _tts_engines_loaded.append("fish_speech")

        # VoxCPM2
        if args.voxcpm2_model:
            from atom.models.voxcpm2.engine import VoxCPM2Engine

            logger.info(
                "Initializing VoxCPM2 from %s...",
                args.voxcpm2_model,
            )
            voxcpm2_engine = VoxCPM2Engine(
                model_dir=args.voxcpm2_model,
                device=args.tts_device,
                dtype=args.tts_dtype,
            )
            voxcpm2_engine.load()
            speech_serving.register_engine(
                "voxcpm2", voxcpm2_engine,
                default=not _tts_engines_loaded,
            )
            _tts_engines_loaded.append("voxcpm2")

        # LFM2.5-Audio llama.cpp bridge
        if args.lfm25_audio:
            global lfm25_audio_client
            from atom.audio.lfm25_audio import LFM25AudioClient, LFM25AudioEngine

            lfm25_audio_client = LFM25AudioClient(base_url=args.lfm25_audio_url)
            lfm25_engine = LFM25AudioEngine(lfm25_audio_client)
            speech_serving.register_engine(
                "lfm25_audio",
                lfm25_engine,
                default=not _tts_engines_loaded,
            )
            speech_serving.register_engine("lfm2.5-audio", lfm25_engine)
            _tts_engines_loaded.append("lfm25_audio")

        logger.info(
            "Speech API enabled at /v1/audio/speech with engines: %s",
            ", ".join(_tts_engines_loaded),
        )

    import signal

    def _sigint_handler(signum, frame):
        logger.info("Received SIGINT, shutting down engine...")
        if engine is not None:
            engine.close()
        import psutil

        try:
            current = psutil.Process()
            children = current.children(recursive=True)
            psutil.wait_procs(children, timeout=2)
            alive = [c for c in children if c.is_running()]
            for c in alive:
                c.kill()
        except psutil.NoSuchProcess:
            pass
        logger.info("Engine shutdown complete.")
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _sigint_handler)

    logger.info(f"Starting server on {args.host}:{args.server_port}...")
    uvicorn.run(app, host=args.host, port=args.server_port)


if __name__ == "__main__":
    main()
