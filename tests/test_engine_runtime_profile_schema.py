import importlib.util
import re
from pathlib import Path


def _load_python_profile_class():
    source = (
        Path(__file__).resolve().parents[1] / "python" / "engine_runtime_profile.py"
    )
    spec = importlib.util.spec_from_file_location("engine_runtime_profile", source)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.EngineRuntimeProfile


def _extract_rust_profile_fields() -> list[str]:
    source = (
        Path(__file__).resolve().parents[1]
        / "crates"
        / "rs_atom_engine_profile"
        / "src"
        / "lib.rs"
    )
    text = source.read_text()
    struct_match = re.search(
        r"pub struct EngineRuntimeProfile\s*\{(?P<body>.*?)\n\}",
        text,
        flags=re.DOTALL,
    )
    assert struct_match is not None, "EngineRuntimeProfile struct not found in Rust source"
    body = struct_match.group("body")
    return re.findall(r"pub\s+([a-zA-Z0-9_]+)\s*:", body)


def test_python_and_rust_engine_runtime_profile_fields_match():
    engine_runtime_profile = _load_python_profile_class()
    python_fields = list(engine_runtime_profile.__dataclass_fields__.keys())
    rust_fields = _extract_rust_profile_fields()

    assert python_fields == rust_fields


def test_python_runtime_profile_adaptive_recommendation_helper():
    engine_runtime_profile = _load_python_profile_class()
    profile = engine_runtime_profile().with_adaptive_recommendation(
        {"family": "ratequant", "score": 0.82, "reason": "high_prefix_reuse_high_kv_hit"}
    )
    payload = profile.to_dict()
    assert payload["adaptive_recommendation"] is not None
    assert payload["adaptive_recommendation"]["family"] == "ratequant"


def test_python_runtime_profile_fp8_kv_cache_helper():
    engine_runtime_profile = _load_python_profile_class()
    profile = engine_runtime_profile().with_fp8_kv_cache_state(
        supports_fp8_kv_cache=True,
        supports_fp8_kv_per_tensor_scales=True,
        supports_fp8_kv_per_head_scales=False,
        supports_kv_scale_calibration=True,
        supports_quantized_attention_fusion=False,
    )
    payload = profile.to_dict()
    assert payload["supports_fp8_kv_cache"] is True
    assert payload["supports_fp8_kv_per_tensor_scales"] is True
    assert payload["supports_fp8_kv_per_head_scales"] is False
    assert payload["supports_kv_scale_calibration"] is True
    assert payload["supports_quantized_attention_fusion"] is False


def test_python_runtime_profile_model_routing_helper():
    engine_runtime_profile = _load_python_profile_class()
    profile = engine_runtime_profile().with_model_routing_state(
        supports_model_hot_swap=True,
        supports_model_aliases=True,
        supports_model_groups=False,
        supports_ttl_unload=True,
        supports_request_filters=False,
        supports_config_reload=True,
        supports_direct_model_passthrough=False,
    )
    payload = profile.to_dict()
    assert payload["supports_model_hot_swap"] is True
    assert payload["supports_model_aliases"] is True
    assert payload["supports_ttl_unload"] is True
    assert payload["supports_config_reload"] is True
    assert payload["supports_direct_model_passthrough"] is False


def test_python_runtime_profile_model_packing_helper():
    engine_runtime_profile = _load_python_profile_class()
    profile = engine_runtime_profile().with_model_packing_state(
        supports_dynamic_model_loading=True,
        supports_dynamic_model_unloading=True,
        supports_multi_model_packing=True,
        supports_multi_gpu_distribution=True,
        supports_kvcached_memory_sharing=True,
        supports_model_sleep_mode=True,
        supports_model_move_operations=False,
        supports_gpu_memory_telemetry=True,
    )
    payload = profile.to_dict()
    assert payload["supports_dynamic_model_loading"] is True
    assert payload["supports_multi_model_packing"] is True
    assert payload["supports_kvcached_memory_sharing"] is True
    assert payload["supports_model_move_operations"] is False
    assert payload["supports_gpu_memory_telemetry"] is True


def test_python_runtime_profile_layer_offloading_helper():
    engine_runtime_profile = _load_python_profile_class()
    profile = engine_runtime_profile().with_layer_offloading_state(True)
    payload = profile.to_dict()
    assert payload["supports_layer_offloading"] is True


def test_python_runtime_profile_compact_runtime_helper():
    engine_runtime_profile = _load_python_profile_class()
    profile = engine_runtime_profile().with_compact_runtime_state(
        supports_cpu_only_runtime=True,
        supports_download_on_first_use=True,
        supports_ollama_style_cli=True,
        supports_openai_compatible_server=True,
        supports_progressive_kv_compression=True,
        supports_full_document_mode=False,
    )
    payload = profile.to_dict()
    assert payload["supports_cpu_only_runtime"] is True
    assert payload["supports_download_on_first_use"] is True
    assert payload["supports_ollama_style_cli"] is True
    assert payload["supports_openai_compatible_server"] is True
    assert payload["supports_progressive_kv_compression"] is True
    assert payload["supports_full_document_mode"] is False


def test_python_runtime_profile_storage_orchestration_helper():
    engine_runtime_profile = _load_python_profile_class()
    profile = engine_runtime_profile().with_storage_orchestration_state(
        supports_distributed_memory_pooling=True,
        supports_dynamic_multilevel_caching=True,
        supports_global_metadata_management=True,
        supports_capacity_management=True,
        supports_prefix_matching=True,
        supports_sliding_window_matching=False,
        supports_kv_matching=True,
        supports_two_phase_write=True,
        supports_async_eviction=True,
        supports_trace_replay_optimization=False,
    )
    payload = profile.to_dict()
    assert payload["supports_distributed_memory_pooling"] is True
    assert payload["supports_dynamic_multilevel_caching"] is True
    assert payload["supports_global_metadata_management"] is True
    assert payload["supports_capacity_management"] is True
    assert payload["supports_prefix_matching"] is True
    assert payload["supports_kv_matching"] is True
    assert payload["supports_two_phase_write"] is True
    assert payload["supports_async_eviction"] is True


def test_python_runtime_profile_quantization_pipeline_helper():
    engine_runtime_profile = _load_python_profile_class()
    profile = engine_runtime_profile().with_quantization_pipeline_state(
        supports_model_free_ptq=True,
        supports_compressed_tensors_format=True,
        supports_weight_quantization_pipeline=True,
        supports_activation_quantization_pipeline=True,
        supports_kv_cache_quantization_pipeline=True,
        supports_attention_quantization_pipeline=False,
        supports_disk_offloading_quantization=True,
        supports_distributed_calibration=True,
    )
    payload = profile.to_dict()
    assert payload["supports_model_free_ptq"] is True
    assert payload["supports_compressed_tensors_format"] is True
    assert payload["supports_weight_quantization_pipeline"] is True
    assert payload["supports_activation_quantization_pipeline"] is True
    assert payload["supports_kv_cache_quantization_pipeline"] is True
    assert payload["supports_disk_offloading_quantization"] is True
    assert payload["supports_distributed_calibration"] is True


def test_python_runtime_profile_serving_benchmark_helper():
    engine_runtime_profile = _load_python_profile_class()
    profile = engine_runtime_profile().with_serving_benchmark_state(
        supports_rust_native_gpu_serving=True,
        supports_pure_jax_tpu_serving=True,
        supports_cross_device_benchmarking=True,
        supports_single_graph_capture=True,
        supports_graph_shape_bucketing=True,
        supports_graph_validation_mode=True,
        supports_graph_conditional_nodes=False,
        supports_graph_nested_capture=False,
        supports_fp8_channelscale_epilogue=True,
        supports_cached_ttft_reporting=True,
        supports_peak_throughput_reporting=True,
        supports_perplexity_reporting=False,
        supports_streaming_api_server=True,
    )
    payload = profile.to_dict()
    assert payload["supports_rust_native_gpu_serving"] is True
    assert payload["supports_pure_jax_tpu_serving"] is True
    assert payload["supports_cross_device_benchmarking"] is True
    assert payload["supports_single_graph_capture"] is True
    assert payload["supports_graph_shape_bucketing"] is True
    assert payload["supports_graph_validation_mode"] is True
    assert payload["supports_graph_conditional_nodes"] is False
    assert payload["supports_graph_nested_capture"] is False
    assert payload["supports_fp8_channelscale_epilogue"] is True
    assert payload["supports_cached_ttft_reporting"] is True
    assert payload["supports_peak_throughput_reporting"] is True
    assert payload["supports_perplexity_reporting"] is False
    assert payload["supports_streaming_api_server"] is True


def test_python_runtime_profile_gfxgraph_bridge_helper():
    engine_runtime_profile = _load_python_profile_class()
    profile = engine_runtime_profile().with_gfxgraph_bridge_state(
        supports_graph_shape_bucketing=True,
        supports_graph_validation_mode=True,
        supports_graph_conditional_nodes=True,
        supports_graph_nested_capture=False,
    )
    payload = profile.to_dict()
    assert payload["supports_graph_shape_bucketing"] is True
    assert payload["supports_graph_validation_mode"] is True
    assert payload["supports_graph_conditional_nodes"] is True
    assert payload["supports_graph_nested_capture"] is False


def test_python_runtime_profile_vllm_runtime_family_helper():
    engine_runtime_profile = _load_python_profile_class()
    profile = engine_runtime_profile().with_vllm_runtime_family_state(
        supports_multimodal_serving=True,
        supports_omni_modality=True,
        supports_hardware_plugin_interface=False,
    )
    payload = profile.to_dict()
    assert payload["supports_multimodal_serving"] is True
    assert payload["supports_omni_modality"] is True
    assert payload["supports_hardware_plugin_interface"] is False


def test_python_runtime_profile_kv_quantization_helpers():
    engine_runtime_profile = _load_python_profile_class()
    profile = engine_runtime_profile(
        supports_fp8_kv_cache=True,
        supports_turboquant_kv=True,
        supports_rotorquant_kv=True,
        supports_atom_kv_quant=True,
        supports_kv_cache_quantization_pipeline=True,
    )
    assert profile.supports_kv_quantization_mode("turboquant") is True
    assert profile.supports_kv_quantization_mode("rotorquant") is True
    assert profile.supports_kv_quantization_mode("fp8_e4m3") is True
    assert "rotorquant" in profile.supported_kv_quantization_modes()


def test_python_runtime_profile_warmup_initialization_helper():
    engine_runtime_profile = _load_python_profile_class()
    profile = engine_runtime_profile().with_warmup_initialization_state(
        supports_kv_connector_warmup=True,
        supports_prefill_warmup_batch=True,
        supports_model_load_warmup=False,
        warmup_kv_connector_initialized=True,
        warmup_prefill_batch_executed=True,
    )
    payload = profile.to_dict()
    assert payload["supports_kv_connector_warmup"] is True
    assert payload["supports_prefill_warmup_batch"] is True
    assert payload["supports_model_load_warmup"] is False
    assert payload["warmup_kv_connector_initialized"] is True
    assert payload["warmup_prefill_batch_executed"] is True
