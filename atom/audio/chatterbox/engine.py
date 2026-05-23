# SPDX-License-Identifier: Apache-2.0
"""Chatterbox TTS engine — GPU-accelerated autoregressive speech token generation.

Loads the Chatterbox backbone (Llama 0.5B or GPT2 350M) directly on GPU.
Speech encoder, embed_tokens, and vocoder run on CPU via ONNX Runtime.

This engine is standalone — it does not go through ATOM's batched scheduler
because Chatterbox requires inputs_embeds (pre-computed embeddings from the
speech encoder + text embedder). Future work can integrate with ATOM's engine
by adding inputs_embeds support to the request pipeline.
"""

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

try:
    import rs_codec

    _HAS_RS_CODEC = True
except ImportError:
    _HAS_RS_CODEC = False

try:
    import onnxruntime
    _HAS_ONNXRUNTIME = True
except ImportError:
    _HAS_ONNXRUNTIME = False

from atom.audio.chatterbox.onnx_artifacts import resolve_component_path
from atom.audio.chatterbox.service import (
    SAMPLE_RATE,
    START_SPEECH_TOKEN,
    STOP_SPEECH_TOKEN,
    ChatterboxService,
)
from atom.audio.runtime import create_cpu_inference_session

logger = logging.getLogger("atom.audio.chatterbox")


class RepetitionPenaltyProcessor:
    def __init__(self, penalty: float):
        self.penalty = penalty

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if input_ids.shape[0] == 1:
            ids = input_ids[0]
            score = scores[0, ids]
            torch.where(score < 0, score * self.penalty, score / self.penalty, out=score)
            scores[0, ids] = score
            return scores

        score = torch.gather(scores, 1, input_ids)
        torch.where(score < 0, score * self.penalty, score / self.penalty, out=score)
        scores.scatter_(1, input_ids, score)
        return scores


class ChatterboxEngine:
    """Chatterbox TTS engine with GPU-accelerated language model.

    Pipeline:
        1. encode_reference() — CPU ONNX speech encoder
        2. prepare_inputs() — CPU ONNX embed_tokens + tokenizer
        3. generate() — GPU autoregressive speech token generation
        4. decode_speech() — CPU ONNX conditional decoder (vocoder)
    """

    def __init__(
        self,
        model_dir: str,
        backbone_dir: Optional[str] = None,
        variant: str = "standard",
        onnx_variant: str = "fp16",
        device: str = "cuda:0",
        dtype: str = "float16",
        num_threads: Optional[int] = None,
    ):
        """
        Args:
            model_dir: Path to ONNX model directory (e.g., chatterbox-ONNX snapshot).
            backbone_dir: Path to HF backbone model for GPU loading. If None,
                          uses the ONNX language model on CPU instead.
            variant: "standard" (Llama backbone) or "turbo" (GPT2 backbone).
        onnx_variant: ONNX precision variant (fp16, fp32, q4f16, q8).
            device: GPU device string.
            dtype: Model dtype (float16, bfloat16, float32).
        """
        self.model_dir = Path(model_dir)
        self.backbone_dir = Path(backbone_dir) if backbone_dir else None
        self.variant = variant
        self.device = torch.device(device)
        self.dtype = getattr(torch, dtype)

        # CPU service for ONNX components
        self.service = ChatterboxService(
            model_dir=model_dir,
            variant=variant,
            onnx_variant=onnx_variant,
            num_threads=num_threads,
        )

        # GPU model (loaded in load())
        self._model = None
        self._use_gpu_backbone = backbone_dir is not None

    def load(self) -> None:
        """Load all components."""
        self.service.load()

        if self._use_gpu_backbone:
            self._load_gpu_backbone()
        else:
            self._load_onnx_lm()

    def _load_gpu_backbone(self) -> None:
        """Load the backbone model on GPU using HuggingFace transformers."""
        from transformers import AutoModelForCausalLM, AutoConfig

        t0 = time.time()
        config = AutoConfig.from_pretrained(str(self.backbone_dir))
        arch = config.architectures[0] if config.architectures else "Unknown"
        logger.info(
            "Loading Chatterbox %s backbone (%s) from %s on %s (%s)",
            self.variant,
            arch,
            self.backbone_dir,
            self.device,
            self.dtype,
        )

        model = AutoModelForCausalLM.from_pretrained(
            str(self.backbone_dir),
            torch_dtype=self.dtype,
            device_map=str(self.device),
        )
        model.eval()
        self._model = model

        elapsed = time.time() - t0
        params_m = sum(p.numel() for p in model.parameters()) / 1e6
        logger.info(
            "Backbone loaded: %.0fM params, %.1fs, VRAM=%.0fMB",
            params_m,
            elapsed,
            torch.cuda.memory_allocated(self.device) / 1e6,
        )

    def _load_onnx_lm(self) -> None:
        """Fallback: load ONNX language model on CPU."""
        lm_path = resolve_component_path(
            self.service.onnx_dir,
            "language_model",
            self.service.onnx_variant,
        )

        logger.info("Loading ONNX language model (CPU fallback): %s", lm_path)
        self._model = create_cpu_inference_session(
            str(lm_path),
            num_threads=self.service.num_threads,
        )
        self._use_gpu_backbone = False

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        ref_audio_path: Optional[str] = None,
        ref_audio_array: Optional[np.ndarray] = None,
        exaggeration: float = 0.5,
        max_tokens: int = 512,
        repetition_penalty: float = 1.2,
        temperature: float = 1.0,
        seed: Optional[int] = None,
    ) -> tuple[np.ndarray, dict]:
        """Generate speech audio from text.

        Returns:
            (wav_array, metrics_dict) — 1-D float32 audio at 24kHz and timing metrics.
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        metrics = {}

        # 1. Encode reference voice (CPU)
        t0 = time.time()
        ref_data = self.service.get_reference_data(
            audio_path=ref_audio_path,
            audio_array=ref_audio_array,
        )
        metrics["encode_sec"] = time.time() - t0

        # 2. Prepare inputs (CPU)
        t1 = time.time()
        prep = self.service.prepare_inputs(text, ref_data, exaggeration=exaggeration)
        metrics["prepare_sec"] = time.time() - t1

        # 3. Generate speech tokens (GPU or CPU)
        t2 = time.time()
        if self._use_gpu_backbone:
            speech_tokens = self._generate_gpu(
                prep,
                max_tokens,
                repetition_penalty,
                temperature,
            )
        else:
            speech_tokens = self._generate_onnx_cpu(
                prep,
                ref_data,
                max_tokens,
                repetition_penalty,
                exaggeration,
            )
        metrics["generate_sec"] = time.time() - t2
        metrics["num_tokens"] = speech_tokens.shape[1]
        metrics["tok_per_sec"] = metrics["num_tokens"] / max(
            metrics["generate_sec"], 0.001
        )

        # 4. Decode to audio (CPU)
        t3 = time.time()
        wav = self.service.decode_speech(speech_tokens, ref_data)
        metrics["decode_sec"] = time.time() - t3

        # 5. Apply AGC and Soft Compression (CPU via Rust)
        t4 = time.time()
        if _HAS_RS_CODEC:
            # target_rms=-18dBFS ~ 0.125.
            # Apply soft compression first to tame peaks
            wav, _ = rs_codec.soft_compressor(wav, 0.5, 4.0, 0.01, 0.1, 1.0)
            # Apply AGC to level out everything to -18dBFS
            wav, _ = rs_codec.agc_kernel(wav, 0.125, 0.01, 0.1, 10.0, 2400, 1.0)
        metrics["postprocess_sec"] = time.time() - t4

        metrics["audio_duration"] = len(wav) / SAMPLE_RATE
        metrics["total_sec"] = sum(
            metrics[k]
            for k in [
                "encode_sec",
                "prepare_sec",
                "generate_sec",
                "decode_sec",
                "postprocess_sec",
            ]
        )
        metrics["rtf"] = metrics["total_sec"] / max(metrics["audio_duration"], 0.001)

        return wav, metrics

    def _generate_gpu(
        self,
        prep: dict,
        max_tokens: int,
        repetition_penalty: float,
        temperature: float,
    ) -> np.ndarray:
        """Autoregressive generation on GPU using HuggingFace model."""
        inputs_embeds = torch.as_tensor(prep["inputs_embeds"], device=self.device, dtype=self.dtype)

        rep_penalty = RepetitionPenaltyProcessor(repetition_penalty)


        generate_tokens = torch.zeros((1, max_tokens + 1), dtype=torch.long, device=self.device)
        generate_tokens[0, 0] = START_SPEECH_TOKEN
        gen_idx = 1

        past_key_values = None

        next_token = None
        # Hoist embedder resolution outside the loop
        embedder = self._model.get_input_embeddings()
        for i in range(max_tokens):
            if i == 0:
                outputs = self._model(
                    inputs_embeds=inputs_embeds,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            else:
                # Embed single token
                token_embeds = embedder(next_token)
                outputs = self._model(
                    inputs_embeds=token_embeds,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]

            # Apply repetition penalty
            logits = rep_penalty(generate_tokens[:, :gen_idx], logits)

            # Sample or argmax
            if temperature == 0.0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                # Apply top-k filtering
                top_k = 50
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            generate_tokens[0, gen_idx] = next_token.item()
            gen_idx += 1

            if (next_token.flatten() == STOP_SPEECH_TOKEN).all():
                break

        # Strip start/stop tokens
        tokens = generate_tokens[:, 1:gen_idx]
        if tokens[0, -1] == STOP_SPEECH_TOKEN:
            tokens = tokens[:, :-1]

        return tokens.cpu().numpy()

    def _generate_onnx_cpu(
        self,
        prep: dict,
        ref_data: dict,
        max_tokens: int,
        repetition_penalty: float,
        exaggeration: float,
    ) -> np.ndarray:
        """Autoregressive generation on CPU using ONNX Runtime (fallback)."""
        llm = self._model
        if _HAS_ONNXRUNTIME:
            assert isinstance(llm, onnxruntime.InferenceSession)

        inputs_embeds = prep["inputs_embeds"]
        num_layers = self.service.num_hidden_layers
        num_heads = self.service.num_heads
        head_dim = self.service.head_dim

        # Detect KV dtype
        kv_inputs = [i for i in llm.get_inputs() if "past_key_values" in i.name]
        kv_dtype = (
            np.float16 if kv_inputs and "float16" in kv_inputs[0].type else np.float32
        )

        # Check if LLM needs position_ids
        llm_input_names = {i.name for i in llm.get_inputs()}
        needs_position_ids = "position_ids" in llm_input_names

        def rep_penalty_fn(ids, scores):
            return self._np_rep_penalty(ids, scores, repetition_penalty)

        batch_size = 1
        seq_len = inputs_embeds.shape[1]

        generate_tokens = np.zeros((batch_size, 1 + max_tokens), dtype=np.int64)
        generate_tokens[0, 0] = START_SPEECH_TOKEN
        gen_idx = 1

        attention_mask = np.ones((batch_size, seq_len + max_tokens), dtype=np.int64)

        past_key_values = {
            f"past_key_values.{layer}.{kv}": np.zeros(
                [batch_size, num_heads, 0, head_dim], dtype=kv_dtype
            )
            for layer in range(num_layers)
            for kv in ("key", "value")
        }

        next_token = None
        seq_len = inputs_embeds.shape[1]

        if needs_position_ids:
            pos_ids_full = np.arange(seq_len + max_tokens, dtype=np.int64)[np.newaxis, :]

        for i in range(max_tokens):
            if i == 0:
                cur_embeds = inputs_embeds
            else:
                cur_embeds = self.service.embed_single_token(
                    next_token,
                    exaggeration=exaggeration,
                )

            current_seq_len = seq_len + i
            cur_attention_mask = attention_mask[:, :current_seq_len]

            llm_inputs = {
                "inputs_embeds": cur_embeds,
                "attention_mask": cur_attention_mask,
                **past_key_values,
            }
            if needs_position_ids:
                if i == 0:
                    llm_inputs["position_ids"] = pos_ids_full[:, :seq_len]
                else:
                    llm_inputs["position_ids"] = pos_ids_full[:, current_seq_len - 1:current_seq_len]

            logits, *present_kvs = llm.run(None, llm_inputs)
            logits = logits[:, -1, :]

            cur_gen_tokens = generate_tokens[:, :gen_idx]
            logits = rep_penalty_fn(cur_gen_tokens, logits)

            next_token = np.argmax(logits, axis=-1, keepdims=True).astype(np.int64)
            generate_tokens[:, gen_idx] = next_token[:, 0]
            gen_idx += 1

            if (next_token.flatten() == STOP_SPEECH_TOKEN).all():
                break

            for j, key in enumerate(past_key_values):
                past_key_values[key] = present_kvs[j]

        # Strip start/stop
        tokens = generate_tokens[:, 1:gen_idx]
        if tokens[0, -1] == STOP_SPEECH_TOKEN:
            tokens = tokens[:, :-1]

        return tokens

    @staticmethod
    def _np_rep_penalty(input_ids, scores, penalty):
        if _HAS_RS_CODEC:
            rs_codec.np_rep_penalty(scores, input_ids, penalty)
            return scores
        if input_ids.shape[0] == 1:
            ids = input_ids[0]
            s = scores[0, ids]
            mask = s < 0
            s[mask] = (s[mask] * penalty).astype(scores.dtype)
            s[~mask] = (s[~mask] / penalty).astype(scores.dtype)
            scores[0, ids] = s
            return scores

        score = np.take_along_axis(scores, input_ids, axis=1)
        mask = score < 0
        score[mask] = (score[mask] * penalty).astype(scores.dtype)
        score[~mask] = (score[~mask] / penalty).astype(scores.dtype)
        np.put_along_axis(scores, input_ids, score, axis=1)
        return scores
