import logging
import os
import sys
from pathlib import Path
from typing import Optional

import torch
from atom.plugin.prepare import _set_framework_backbone
from atom.utils import envs
from atom.plugin.vllm.mla_patch import patch_vllm_mla_attention

logger = logging.getLogger("atom")

# this flag is used to enable the vllm plugin mode
disable_vllm_plugin = envs.ATOM_DISABLE_VLLM_PLUGIN
disable_vllm_plugin_attention = envs.ATOM_DISABLE_VLLM_PLUGIN_ATTENTION

# those 2 models are covering most of dense and moe models
ATOM_CAUSAL_LM_MODEL_WRAPPER = "atom.plugin.vllm.model_wrapper:ATOMForCausalLM"
ATOM_MOE_CAUSAL_LM_MODEL_WRAPPER = "atom.plugin.vllm.model_wrapper:ATOMMoEForCausalLM"

# when register new model to vllm, add here
# Keys is from hf config arch name
_VLLM_MODEL_REGISTRY_OVERRIDES: dict[str, str] = {
    "LlamaForCausalLM": ATOM_CAUSAL_LM_MODEL_WRAPPER,
    "Qwen3ForCausalLM": ATOM_CAUSAL_LM_MODEL_WRAPPER,
    "Qwen2ForCausalLM": ATOM_CAUSAL_LM_MODEL_WRAPPER,
    "Qwen3MoeForCausalLM": ATOM_MOE_CAUSAL_LM_MODEL_WRAPPER,
    "GptOssForCausalLM": ATOM_MOE_CAUSAL_LM_MODEL_WRAPPER,
    "DeepseekV3ForCausalLM": ATOM_MOE_CAUSAL_LM_MODEL_WRAPPER,
    "DeepseekV32ForCausalLM": ATOM_MOE_CAUSAL_LM_MODEL_WRAPPER,
    "Glm4MoeForCausalLM": ATOM_MOE_CAUSAL_LM_MODEL_WRAPPER,
    "GlmMoeDsaForCausalLM": ATOM_MOE_CAUSAL_LM_MODEL_WRAPPER,
    "Qwen3NextForCausalLM": "atom.models.qwen3_next:Qwen3NextForCausalLMVllm",
    "Qwen3_5ForConditionalGeneration": "atom.models.qwen3_5:Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForConditionalGeneration": "atom.models.qwen3_5:Qwen3_5MoeForConditionalGeneration",
    "KimiK25ForConditionalGeneration": "atom.plugin.vllm.models.kimi_k25:KimiK25ForConditionalGeneration",
    "MiniMaxM2ForCausalLM": ATOM_MOE_CAUSAL_LM_MODEL_WRAPPER,
}


def _set_plugin_mode() -> None:
    _set_framework_backbone("vllm")


def register_platform() -> Optional[str]:

    if disable_vllm_plugin:
        # return None instead of error because the flag can be used to
        # run pure vllm mode without ATOM plugin
        logger.info("Disable ATOM OOT plugin platforms")
        return None

    # Do not call _set_plugin_mode() here. SGLang (and other stacks) discover
    # vllm.platform_plugins and would set atom's backbone to "vllm" before
    # importing SGLang plugin modules — then atom.models.qwen3_5's ``if is_vllm():``
    # branch runs and requires vllm.model_executor.models.qwen3_5, which may be
    # absent. Backbone is set in register_model() for real vLLM runs.

    # return the ATOM platform to vllm
    return "atom.plugin.vllm.platform.ATOMPlatform"


def _patch_vllm_attention_process_weights_after_loading(attention) -> None:
    orig = attention.process_weights_after_loading

    if getattr(orig, "_atom_default_act_dtype_patched", False):
        return

    try:
        import inspect

        sig = inspect.signature(orig)
        act_dtype_param = sig.parameters.get("act_dtype")
        if (
            act_dtype_param is not None
            and act_dtype_param.default is not inspect._empty
        ):
            return
    except Exception:
        pass

    import functools

    @functools.wraps(orig)
    def wrapped(self, act_dtype: "torch.dtype" = torch.bfloat16):
        return orig(self, act_dtype)

    setattr(wrapped, "_atom_default_act_dtype_patched", True)
    attention.process_weights_after_loading = wrapped


def _register_chatterbox_vllm_model() -> bool:
    source = os.environ.get("ATOM_CHATTERBOX_VLLM_SOURCE")
    if not source:
        return False

    src = Path(source).expanduser()
    if not src.exists():
        logger.warning("Chatterbox-vLLM source path does not exist: %s", src)
        return False
    src_str = str(src)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)

    try:
        from atom.audio.chatterbox.vllm_backend import _load_chatterbox_tts_class

        _load_chatterbox_tts_class(src)
        import vllm.model_executor.models.registry as vllm_model_registry

        vllm_model_registry.ModelRegistry.register_model(
            "ChatterboxT3",
            "chatterbox_vllm.models.t3.t3:T3VllmModel",
        )
        vllm_model_registry._try_load_model_cls.cache_clear()
        vllm_model_registry._try_inspect_model_cls.cache_clear()
    except Exception as exc:
        logger.warning("Unable to register ChatterboxT3 for vLLM worker: %s", exc)
        return False

    logger.info("Registered ChatterboxT3 donor model for vLLM")
    return True


def register_model() -> None:
    chatterbox_registered = _register_chatterbox_vllm_model()
    if disable_vllm_plugin:
        logger.info("Disable ATOM model register")
        return

    _set_plugin_mode()

    import vllm.model_executor.models.registry as vllm_model_registry

    any_updated = False
    for arch, qual in _VLLM_MODEL_REGISTRY_OVERRIDES.items():
        module_name, class_name = qual.split(":", 1)
        existing = vllm_model_registry.ModelRegistry.models.get(arch)
        if existing is not None:
            # If already overridden to the same target, skip re-registering.
            if (
                getattr(existing, "module_name", None) == module_name
                and getattr(existing, "class_name", None) == class_name
            ):
                continue

        logger.info(f"Register model {arch} to vLLM with {qual}")
        vllm_model_registry.ModelRegistry.register_model(arch, qual)
        any_updated = True

    # clear lru cache
    if any_updated or chatterbox_registered:
        vllm_model_registry._try_load_model_cls.cache_clear()
        vllm_model_registry._try_inspect_model_cls.cache_clear()

    patch_vllm_mla_attention()
    # patch attention process weights after loading
    # to avoid the specific handle in ATOM loader
    try:
        from vllm.attention.layer import Attention, MLAAttention
    except ImportError:
        from vllm.model_executor.layers.attention import Attention, MLAAttention

    _patch_vllm_attention_process_weights_after_loading(Attention)
    _patch_vllm_attention_process_weights_after_loading(MLAAttention)

    # Patch vLLM graph_capture to also enter aiter's ca_comm.capture(),
    # avoiding hipMemcpyAsync in fused_allreduce_rmsnorm when model uses aiter collectives
    from atom.plugin.vllm.graph_capture_patch import apply_graph_capture_patch

    apply_graph_capture_patch()
