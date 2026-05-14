# SPDX-License-Identifier: Apache-2.0
"""Optional Chatterbox-vLLM backend for ATOM speech serving.

This adapter keeps the Chatterbox-vLLM fork as a source reference and runtime
dependency instead of vendoring it into gfxATOM.  When the vLLM runtime is not
installed, construction can still fall back to the existing Chatterbox engine.
"""

from __future__ import annotations

import contextlib
import importlib.util
import logging
import os
import shutil
import sys
import time
import types
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from atom.audio.chatterbox.kernel_candidates import (
    allow_experimental_chatterbox_gemm,
    rdna2_runtime_detected,
)

DEFAULT_CHATTERBOX_VLLM_SOURCE = Path("/home/local/ai/engines/chatterbox-vllm")
DEFAULT_US_FEMALE_VOICE = Path(
    "/home/local/ai/projects/DEMERZEL/servers/pipecat-voice/data/kokoro-refs/af_bella.wav"
)
SAMPLE_RATE = 24000

logger = logging.getLogger("atom.audio.chatterbox.vllm")


class ChatterboxVllmBackendUnavailable(RuntimeError):
    """Raised when the optional Chatterbox-vLLM runtime cannot be loaded."""


@contextlib.contextmanager
def _pushd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _source_src_dir(source_dir: str | Path | None) -> Path:
    root = Path(source_dir or DEFAULT_CHATTERBOX_VLLM_SOURCE).expanduser()
    return root / "src" if (root / "src").is_dir() else root


def _ensure_source_on_path(source_dir: str | Path | None) -> Path:
    src = _source_src_dir(source_dir)
    if not src.exists():
        raise ChatterboxVllmBackendUnavailable(
            f"Chatterbox-vLLM source path does not exist: {src}"
        )
    os.environ["ATOM_CHATTERBOX_VLLM_SOURCE"] = str(src)
    pythonpath = os.environ.get("PYTHONPATH", "")
    pythonpath_parts = [part for part in pythonpath.split(os.pathsep) if part]
    if str(src) not in pythonpath_parts:
        os.environ["PYTHONPATH"] = os.pathsep.join([str(src), *pythonpath_parts])
    _apply_rdna2_stability_env()
    src_str = str(src)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
    return src


def _apply_rdna2_stability_env() -> None:
    """Keep Chatterbox-vLLM on conservative ROCm kernels unless opted in."""
    if not rdna2_runtime_detected():
        return
    if allow_experimental_chatterbox_gemm():
        logger.warning(
            "ATOM_CHATTERBOX_EXPERIMENTAL_GEMM is enabled; donor GEMM kernels "
            "must have passed a local Chatterbox-shape microbench."
        )
        return
    os.environ.setdefault("VLLM_ROCM_USE_AITER_TRITON_GEMM", "False")
    os.environ.setdefault("VLLM_ROCM_USE_SKINNY_GEMM", "False")
    os.environ.setdefault("VLLM_USE_TRITON_FLASH_ATTN", "False")


def _install_vllm_compat_shims() -> None:
    """Bridge small vLLM internal API moves used by the donor T3 model."""
    if "vllm.model_executor.sampling_metadata" not in sys.modules:
        try:
            from vllm.v1.sample.metadata import SamplingMetadata
        except Exception:
            pass
        else:
            module = types.ModuleType("vllm.model_executor.sampling_metadata")
            module.SamplingMetadata = SamplingMetadata
            sys.modules[module.__name__] = module
    try:
        import vllm.inputs as vllm_inputs
        import vllm.multimodal.inputs as mm_inputs
        import vllm.multimodal.parse as mm_parse
        import vllm.multimodal.processing as mm_processing
    except Exception:
        return

    if not hasattr(mm_inputs, "MultiModalKwargs"):
        class MultiModalKwargs(mm_inputs.MultiModalKwargsItems):
            @staticmethod
            def from_items(items: list[Any]):
                grouped: dict[str, list[Any]] = {}
                for item in items:
                    for key in item.keys():
                        grouped.setdefault(key, []).append(item)
                return mm_inputs.MultiModalKwargsItems(grouped)

        mm_inputs.MultiModalKwargs = MultiModalKwargs

    for name in (
        "MultiModalFieldConfig",
        "PlaceholderRange",
    ):
        if not hasattr(mm_processing, name) and hasattr(mm_inputs, name):
            setattr(mm_processing, name, getattr(mm_inputs, name))
    if not hasattr(mm_processing, "MultiModalDataItems"):
        mm_processing.MultiModalDataItems = mm_parse.MultiModalDataItems
    if not hasattr(mm_processing, "MultiModalDataDict"):
        mm_processing.MultiModalDataDict = vllm_inputs.MultiModalDataDict
    if not hasattr(mm_processing, "MultiModalInputs"):
        mm_processing.MultiModalInputs = vllm_inputs.MultiModalInput
    if "vllm.multimodal.profiling" not in sys.modules:
        profiling = types.ModuleType("vllm.multimodal.profiling")
        profiling.BaseDummyInputsBuilder = mm_processing.BaseDummyInputsBuilder
        sys.modules[profiling.__name__] = profiling
    if "vllm.transformers_utils.tokenizer_base" not in sys.modules:
        try:
            from vllm.tokenizers import TokenizerRegistry
        except Exception:
            pass
        else:
            tokenizer_base = types.ModuleType("vllm.transformers_utils.tokenizer_base")
            tokenizer_base.TokenizerRegistry = TokenizerRegistry
            sys.modules[tokenizer_base.__name__] = tokenizer_base


def _load_chatterbox_tts_class(source_dir: str | Path | None):
    """Load donor ChatterboxTTS with ATOM-owned vLLM API compatibility."""
    _ensure_source_on_path(source_dir)
    _install_vllm_compat_shims()

    import chatterbox_vllm.tts as donor_tts
    from chatterbox_vllm.models.t3.entokenizer import EnTokenizer
    from chatterbox_vllm.models.t3.mtltokenizer import MTLTokenizer
    import chatterbox_vllm.models.t3.t3 as donor_t3
    from vllm import LLM as VllmLLM
    from vllm.renderers.registry import RENDERER_REGISTRY
    from vllm.tokenizers import TokenizerRegistry

    # Current vLLM normalizes tokenizer_mode to lowercase after EngineArgs
    # validation, while the donor registers mixed-case tokenizer modes.
    TokenizerRegistry.register(
        "entokenizer", "chatterbox_vllm.models.t3.entokenizer", "EnTokenizer"
    )
    TokenizerRegistry.register(
        "mtltokenizer", "chatterbox_vllm.models.t3.mtltokenizer", "MTLTokenizer"
    )
    RENDERER_REGISTRY.register("entokenizer", "vllm.renderers.hf", "HfRenderer")
    RENDERER_REGISTRY.register("mtltokenizer", "vllm.renderers.hf", "HfRenderer")
    for tokenizer_cls in (EnTokenizer, MTLTokenizer):
        if not hasattr(tokenizer_cls, "max_chars_per_token"):
            tokenizer_cls.max_chars_per_token = 8
        if getattr(tokenizer_cls.from_pretrained, "_atom_accepts_model_arg", False):
            continue
        original_from_pretrained = tokenizer_cls.from_pretrained

        def from_pretrained_compat(
            cls, *args: Any, _original=original_from_pretrained, **kwargs: Any
        ):
            return _original(**kwargs)

        from_pretrained_compat._atom_accepts_model_arg = True  # type: ignore[attr-defined]
        tokenizer_cls.from_pretrained = classmethod(from_pretrained_compat)

    dummy_cls = donor_t3.T3MultiModalDummyInputsBuilder
    info_cls = donor_t3.T3ProcessingInfo
    if not getattr(info_cls, "_atom_uses_t3_data_parser", False):

        def get_data_parser(self):
            return donor_t3.T3MultiModalDataParser()

        info_cls.get_data_parser = get_data_parser
        info_cls._atom_uses_t3_data_parser = True

    if not getattr(dummy_cls.get_dummy_mm_data, "_atom_accepts_mm_options", False):
        original_get_dummy_mm_data = dummy_cls.get_dummy_mm_data

        def get_dummy_mm_data_compat(
            self,
            seq_len: int,
            mm_counts: Any,
            mm_options: Any | None = None,
        ):
            return original_get_dummy_mm_data(self, seq_len, mm_counts)

        get_dummy_mm_data_compat._atom_accepts_mm_options = True  # type: ignore[attr-defined]
        dummy_cls.get_dummy_mm_data = get_dummy_mm_data_compat

    processor_cls = donor_t3.T3MultiModalProcessor
    model_cls = donor_t3.T3VllmModel
    if not getattr(model_cls, "_atom_embed_multimodal_compat", False):

        def embed_multimodal(self, **kwargs: Any):
            conditionals = kwargs.get("conditionals")
            if conditionals is None:
                return None
            if donor_t3.torch.is_tensor(conditionals):
                if conditionals.ndim == 3:
                    return [item for item in conditionals]
                return [conditionals]

            embeddings = []
            for item in conditionals:
                if isinstance(item, (list, tuple)):
                    embeddings.extend(item)
                else:
                    embeddings.append(item)
            return embeddings

        model_cls.embed_multimodal = embed_multimodal
        model_cls._atom_embed_multimodal_compat = True

    if not getattr(model_cls.compute_logits, "_atom_optional_sampling_metadata", False):

        def compute_logits_compat(self, hidden_states, sampling_metadata: Any | None = None):
            cond_hidden_states, uncond_hidden_states = hidden_states.split(
                [self.dim, self.dim], dim=1
            )
            cond_logits = self.logits_processor(self.speech_head, cond_hidden_states)
            uncond_logits = self.logits_processor(self.speech_head, uncond_hidden_states)
            logits = cond_logits + self.cfg_scale * (cond_logits - uncond_logits)
            return donor_t3.torch.cat(
                [
                    donor_t3.torch.zeros(
                        logits.shape[0],
                        donor_t3.SPEECH_TOKEN_OFFSET,
                        device=logits.device,
                    ).fill_(float("-inf")),
                    logits,
                ],
                dim=1,
            )

        compute_logits_compat._atom_optional_sampling_metadata = True  # type: ignore[attr-defined]
        model_cls.compute_logits = compute_logits_compat

    if not getattr(model_cls, "_atom_embed_input_ids_compat", False):

        def embed_input_ids(self, input_ids, multimodal_embeddings: Any | None = None, **_: Any):
            return self.get_input_embeddings(
                input_ids,
                multimodal_embeddings=multimodal_embeddings,
            )

        model_cls.embed_input_ids = embed_input_ids
        model_cls._atom_embed_input_ids_compat = True

    if not getattr(processor_cls.apply, "_atom_accepts_processor_inputs", False):

        def apply_compat(self, inputs, timing_ctx: Any | None = None):
            prompt = inputs.prompt
            tokenizer = self.info.get_tokenizer()
            if isinstance(prompt, str):
                prompt_ids = tokenizer(prompt, return_tensors="pt")[
                    "input_ids"
                ][0].tolist()
            else:
                prompt_ids = list(prompt)

            conditionals_items = inputs.mm_data_items["conditionals"]
            conditionals = getattr(conditionals_items, "data", conditionals_items)
            if hasattr(conditionals, "get"):
                conditionals = conditionals.get(0)
            if not isinstance(conditionals, list):
                conditionals = [conditionals]
            assert conditionals, "Conditionals are required for prefill"
            cond = conditionals[0]

            final_prompt_ids = [
                donor_t3.PREFILL_COND_START_TOKEN,
                *([prompt_ids[0]] * (donor_t3.CONDITIONING_SIZE - 2)),
                donor_t3.PREFILL_COND_END_TOKEN,
                *prompt_ids,
                donor_t3.PREFILL_END_TOKEN,
            ]
            new_conditionals = donor_t3.torch.cat(
                [
                    cond,
                    donor_t3.create_triangular_matrix(
                        len(prompt_ids), cond.shape[1]
                    ).to(cond.device),
                    donor_t3.torch.zeros(1, cond.shape[1]).to(cond.device),
                ],
                dim=0,
            )
            assert len(new_conditionals) == len(final_prompt_ids)

            from vllm.multimodal.inputs import (
                MultiModalBatchedField,
                MultiModalKwargsItem,
                MultiModalKwargsItems,
            )

            conditionals_elems = MultiModalBatchedField().build_elems(
                modality="conditionals",
                key="conditionals",
                data=[new_conditionals],
            )
            new_mm_kwargs = MultiModalKwargsItems(
                {
                    "conditionals": [
                        MultiModalKwargsItem({"conditionals": conditionals_elems[0]})
                    ]
                }
            )
            from vllm.inputs import mm_input

            return mm_input(
                prompt_token_ids=final_prompt_ids,
                prompt=prompt if isinstance(prompt, str) else None,
                mm_kwargs=new_mm_kwargs,
                mm_hashes={"conditionals": [str(time.time_ns())]},
                mm_placeholders={
                    "conditionals": [
                        donor_t3.PlaceholderRange(
                            offset=0,
                            length=len(final_prompt_ids),
                            is_embed=None,
                        )
                    ]
                },
            )

        apply_compat._atom_accepts_processor_inputs = True  # type: ignore[attr-defined]
        processor_cls.apply = apply_compat

    current_llm = getattr(donor_tts, "LLM")
    if not getattr(current_llm, "_atom_task_runner_compat", False):

        class CompatLLM(VllmLLM):
            _atom_task_runner_compat = True

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                task = kwargs.pop("task", None)
                if task is not None and "runner" not in kwargs:
                    kwargs["runner"] = task
                if kwargs.get("tokenizer_mode") == "custom":
                    tokenizer_name = kwargs.get("tokenizer")
                    if tokenizer_name in {"EnTokenizer", "MtlTokenizer"}:
                        kwargs["tokenizer_mode"] = tokenizer_name
                super().__init__(*args, **kwargs)

        donor_tts.LLM = CompatLLM

    return donor_tts.ChatterboxTTS


def is_atom_vllm_available(
    source_dir: str | Path | None = None,
) -> tuple[bool, str | None]:
    """Return whether the optional Chatterbox-vLLM path is importable."""
    if importlib.util.find_spec("vllm") is None:
        return False, "Python package 'vllm' is not importable in this environment"

    try:
        _load_chatterbox_tts_class(source_dir)
    except Exception as exc:
        return False, str(exc)
    return True, None


def _copy_tokenizer_config(source_root: Path, work_dir: Path, variant: str) -> None:
    model_dir_name = "t3-model-multilingual" if variant == "multilingual" else "t3-model"
    src_config_dir = source_root / model_dir_name
    dst_config_dir = work_dir / model_dir_name
    if not src_config_dir.is_dir():
        raise ChatterboxVllmBackendUnavailable(
            f"Missing Chatterbox-vLLM config directory: {src_config_dir}"
        )
    shutil.copytree(src_config_dir, dst_config_dir, dirs_exist_ok=True)


def _find_t3_weights(checkpoint_dir: Path, variant: str) -> Path | None:
    names = (
        ("t3_mtl23ls_v2.safetensors", "t3_cfg.safetensors")
        if variant == "multilingual"
        else ("t3_cfg.safetensors", "t3_mtl23ls_v2.safetensors")
    )
    for name in names:
        direct = checkpoint_dir / name
        if direct.exists():
            return direct
    for name in names:
        matches = sorted(checkpoint_dir.rglob(name))
        if matches:
            return matches[0]
    return None


def _prepare_runtime_workdir(
    checkpoint_dir: str | Path,
    source_dir: str | Path | None,
    work_dir: str | Path | None,
    variant: str,
) -> Path:
    source_root = _source_src_dir(source_dir).parent
    ckpt_dir = Path(checkpoint_dir).expanduser()
    runtime_dir = Path(
        work_dir or Path.home() / ".cache" / "atom" / "chatterbox-vllm"
    ).expanduser()
    runtime_dir.mkdir(parents=True, exist_ok=True)
    _copy_tokenizer_config(source_root, runtime_dir, variant)

    weights = _find_t3_weights(ckpt_dir, variant)
    if weights is None:
        logger.warning("Could not locate T3 safetensors under %s", ckpt_dir)
        return runtime_dir

    model_dir_name = "t3-model-multilingual" if variant == "multilingual" else "t3-model"
    model_link = runtime_dir / model_dir_name / "model.safetensors"
    if model_link.exists() or model_link.is_symlink():
        if model_link.resolve() == weights.resolve():
            return runtime_dir
        model_link.unlink()
    model_link.symlink_to(weights)
    return runtime_dir


def _split_text(text: str, chunk_chars: int | None) -> list[str]:
    if not chunk_chars or len(text) <= chunk_chars:
        return [text]

    chunks: list[str] = []
    remaining = text.strip()
    while remaining:
        if len(remaining) <= chunk_chars:
            chunks.append(remaining)
            break
        window = remaining[:chunk_chars]
        split_at = max(window.rfind(". "), window.rfind("? "), window.rfind("! "), window.rfind("; "))
        if split_at < max(80, chunk_chars // 2):
            split_at = window.rfind(" ")
        if split_at < 1:
            split_at = chunk_chars
        chunk = remaining[: split_at + 1].strip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[split_at + 1 :].strip()
    return chunks or [text]


def _as_float32_array(audio: Any) -> np.ndarray:
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    return arr


class ChatterboxAtomVllmEngine:
    """Chatterbox-vLLM backed TTS engine with existing-engine fallback."""

    def __init__(
        self,
        model_dir: str,
        *,
        variant: str = "standard",
        device: str = "cuda",
        source_dir: str | Path | None = None,
        work_dir: str | Path | None = None,
        max_model_len: int = 1000,
        max_batch_size: int = 10,
        gpu_memory_utilization: float | None = None,
        enforce_eager: bool = True,
        t3_dtype: str = "float16",
        s3gen_use_fp16: bool = False,
        cfg_weight: float = 0.5,
        default_voice_path: str | Path | None = DEFAULT_US_FEMALE_VOICE,
        fallback_engine: Any | None = None,
    ) -> None:
        self.model_dir = Path(model_dir).expanduser()
        self.variant = variant
        self.device = device
        self.source_dir = Path(source_dir or DEFAULT_CHATTERBOX_VLLM_SOURCE).expanduser()
        self.work_dir = Path(work_dir).expanduser() if work_dir else None
        self.max_model_len = max_model_len
        self.max_batch_size = max_batch_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.enforce_eager = enforce_eager
        self.t3_dtype = t3_dtype
        self.s3gen_use_fp16 = s3gen_use_fp16
        self.cfg_weight = cfg_weight
        self.default_voice_path = (
            Path(default_voice_path).expanduser() if default_voice_path else None
        )
        self.fallback_engine = fallback_engine
        self._model: Any | None = None
        self._unavailable_reason: str | None = None

    def load(self) -> None:
        """Load Chatterbox-vLLM or defer to fallback when unavailable."""
        _apply_rdna2_stability_env()
        ok, reason = is_atom_vllm_available(self.source_dir)
        if not ok:
            self._unavailable_reason = reason
            if self.fallback_engine is not None:
                if getattr(self.fallback_engine, "_model", None) is None:
                    self.fallback_engine.load()
                logger.warning("atom_vllm unavailable, using fallback: %s", reason)
                return
            raise ChatterboxVllmBackendUnavailable(reason or "atom_vllm unavailable")

        ChatterboxTTS = _load_chatterbox_tts_class(self.source_dir)

        runtime_dir = _prepare_runtime_workdir(
            self.model_dir,
            self.source_dir,
            self.work_dir,
            "multilingual" if self.variant == "multilingual" else "standard",
        )
        if rdna2_runtime_detected():
            self.t3_dtype = "float16"
            self.enforce_eager = True

        kwargs: dict[str, Any] = {}
        if self.gpu_memory_utilization is not None:
            kwargs["gpu_memory_utilization"] = self.gpu_memory_utilization
        if self.t3_dtype:
            kwargs["dtype"] = self.t3_dtype
        os.environ["CHATTERBOX_CFG_SCALE"] = str(self.cfg_weight)

        with _pushd(runtime_dir):
            self._model = ChatterboxTTS.from_local(
                str(self.model_dir),
                target_device=self.device,
                max_model_len=self.max_model_len,
                compile=not self.enforce_eager,
                max_batch_size=self.max_batch_size,
                variant=(
                    "multilingual" if self.variant == "multilingual" else "english"
                ),
                s3gen_use_fp16=self.s3gen_use_fp16,
                **kwargs,
            )
        logger.info("Loaded Chatterbox atom_vllm backend from %s", self.model_dir)

    @property
    def sample_rate(self) -> int:
        return int(getattr(self._model, "sr", SAMPLE_RATE))

    @property
    def model_type(self) -> str:
        return "chatterbox_atom_vllm"

    @property
    def unavailable_reason(self) -> str | None:
        return self._unavailable_reason

    def list_voices(self) -> list[str]:
        voices = ["default", "us_female", "af_bella"]
        if self.fallback_engine is not None and hasattr(self.fallback_engine, "list_voices"):
            voices.extend(str(v) for v in self.fallback_engine.list_voices())
        return sorted(set(voices))

    def generate(
        self,
        text: str,
        ref_audio_path: str | None = None,
        ref_audio_array: np.ndarray | None = None,
        *,
        exaggeration: float = 0.5,
        max_tokens: int = 1000,
        repetition_penalty: float = 2.0,
        temperature: float = 0.8,
        top_p: float = 1.0,
        min_p: float = 0.05,
        diffusion_steps: int = 10,
        batch_size: int | None = None,
        chunk_chars: int | None = None,
        cfg_weight: float | None = None,
        language: str | None = None,
        language_id: str | None = None,
        seed: int | None = None,
        **_: Any,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if self._model is None:
            if self.fallback_engine is None:
                raise ChatterboxVllmBackendUnavailable(
                    self._unavailable_reason or "atom_vllm backend is not loaded"
                )
            wav, metrics = self.fallback_engine.generate(
                text=text,
                ref_audio_path=ref_audio_path,
                ref_audio_array=ref_audio_array,
                exaggeration=exaggeration,
                max_tokens=max_tokens,
                repetition_penalty=repetition_penalty,
                temperature=temperature,
                seed=seed,
            )
            metrics = dict(metrics)
            metrics["backend"] = "fallback"
            metrics["fallback_reason"] = self._unavailable_reason
            return wav, metrics

        if cfg_weight is not None:
            os.environ["CHATTERBOX_CFG_SCALE"] = str(cfg_weight)

        reference = ref_audio_path
        if reference in {"default", "us_female", "af_bella"}:
            reference = None
        if reference is None and ref_audio_array is None and self.default_voice_path:
            if self.default_voice_path.exists():
                reference = str(self.default_voice_path)

        chunks = _split_text(text, chunk_chars)
        effective_batch = batch_size or min(self.max_batch_size, len(chunks))
        lang = language_id or language or "en"

        t0 = time.time()
        audio_chunks: list[np.ndarray] = []
        try:
            for start in range(0, len(chunks), max(1, effective_batch)):
                prompts = chunks[start : start + max(1, effective_batch)]
                outputs = self._model.generate(
                    prompts,
                    audio_prompt_path=reference,
                    language_id=lang,
                    exaggeration=exaggeration,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    min_p=min_p,
                    repetition_penalty=repetition_penalty,
                    diffusion_steps=diffusion_steps,
                )
                if not isinstance(outputs, Iterable) or isinstance(outputs, np.ndarray):
                    outputs = [outputs]
                audio_chunks.extend(_as_float32_array(item) for item in outputs)
        except Exception as exc:
            if self.fallback_engine is None:
                raise
            reason = f"atom_vllm generation failed: {exc}"
            logger.warning("%s; using fallback backend", reason, exc_info=True)
            wav, metrics = self.fallback_engine.generate(
                text=text,
                ref_audio_path=ref_audio_path,
                ref_audio_array=ref_audio_array,
                exaggeration=exaggeration,
                max_tokens=max_tokens,
                repetition_penalty=repetition_penalty,
                temperature=temperature,
                seed=seed,
            )
            metrics = dict(metrics)
            metrics["backend"] = "fallback"
            metrics["fallback_reason"] = reason
            metrics["requested_backend"] = "atom_vllm"
            return wav, metrics

        wav = np.concatenate(audio_chunks) if audio_chunks else np.zeros(0, dtype=np.float32)
        total_sec = time.time() - t0
        duration = len(wav) / max(self.sample_rate, 1)
        metrics = {
            "backend": "atom_vllm",
            "requested_backend": "atom_vllm",
            "chunk_count": len(chunks),
            "batch_size": effective_batch,
            "generate_sec": total_sec,
            "total_sec": total_sec,
            "audio_duration": duration,
            "rtf": total_sec / max(duration, 0.001),
            "num_tokens": 0,
            "tok_per_sec": 0.0,
            "diffusion_steps": diffusion_steps,
            "cfg_weight": cfg_weight if cfg_weight is not None else self.cfg_weight,
        }
        return wav, metrics

    def generate_batch(
        self,
        texts: list[str],
        **kwargs: Any,
    ) -> list[tuple[np.ndarray, dict[str, Any]]]:
        return [self.generate(text, **kwargs) for text in texts]
