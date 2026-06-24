from dataclasses import asdict, dataclass, field, replace


@dataclass(frozen=True)
class EngineRuntimeProfile:
    supports_atom_backend: bool = True
    supports_atom_attention: bool = True
    supports_atom_kv_quant: bool = False
    supports_atom_rocm_telemetry: bool = True
    supports_atom_fallback: bool = True
    supports_automatic_prefix_caching: bool = True
    supports_radix_cache: bool = True
    supports_lmcache_connector: bool = False
    supports_kv_events: bool = True
    supports_fp8_kv_cache: bool = False
    supports_fp8_kv_per_tensor_scales: bool = False
    supports_fp8_kv_per_head_scales: bool = False
    supports_kv_scale_calibration: bool = False
    supports_quantized_attention_fusion: bool = False
    supports_turboquant_kv: bool = False
    supports_rotorquant_kv: bool = False
    supports_eagle3: bool = False
    supports_medusa: bool = False
    supports_ngram_speculation: bool = False
    supports_phantom: bool = False
    supports_phantom_x: bool = False
    supports_disaggregated_prefill: bool = False
    supports_disaggregated_decode: bool = False
    supports_prefix_aware_attention: bool = False
    supports_content_addressed_cache: bool = False
    supports_position_independent_cache: bool = False
    supports_model_hot_swap: bool = False
    supports_model_aliases: bool = False
    supports_model_groups: bool = False
    supports_ttl_unload: bool = False
    supports_request_filters: bool = False
    supports_config_reload: bool = False
    supports_direct_model_passthrough: bool = False
    supports_dynamic_model_loading: bool = False
    supports_dynamic_model_unloading: bool = False
    supports_multi_model_packing: bool = False
    supports_multi_gpu_distribution: bool = False
    supports_kvcached_memory_sharing: bool = False
    supports_model_sleep_mode: bool = False
    supports_model_move_operations: bool = False
    supports_layer_offloading: bool = False
    supports_gpu_memory_telemetry: bool = False
    supports_cpu_only_runtime: bool = False
    supports_download_on_first_use: bool = False
    supports_ollama_style_cli: bool = False
    supports_openai_compatible_server: bool = False
    supports_progressive_kv_compression: bool = False
    supports_full_document_mode: bool = False
    supports_distributed_memory_pooling: bool = True
    supports_dynamic_multilevel_caching: bool = True
    supports_global_metadata_management: bool = False
    supports_capacity_management: bool = False
    supports_prefix_matching: bool = False
    supports_sliding_window_matching: bool = False
    supports_kv_matching: bool = True
    supports_two_phase_write: bool = False
    supports_async_eviction: bool = True
    supports_trace_replay_optimization: bool = False
    supports_model_free_ptq: bool = False
    supports_compressed_tensors_format: bool = False
    supports_weight_quantization_pipeline: bool = False
    supports_activation_quantization_pipeline: bool = False
    supports_kv_cache_quantization_pipeline: bool = False
    supports_attention_quantization_pipeline: bool = False
    supports_disk_offloading_quantization: bool = False
    supports_distributed_calibration: bool = False
    supports_rust_native_gpu_serving: bool = False
    supports_pure_jax_tpu_serving: bool = False
    supports_cross_device_benchmarking: bool = False
    supports_single_graph_capture: bool = False
    supports_multimodal_serving: bool = False
    supports_omni_modality: bool = False
    supports_hardware_plugin_interface: bool = False
    supports_graph_shape_bucketing: bool = False
    supports_graph_validation_mode: bool = False
    supports_graph_conditional_nodes: bool = False
    supports_graph_nested_capture: bool = False
    supports_fp8_channelscale_epilogue: bool = False
    supports_cached_ttft_reporting: bool = False
    supports_peak_throughput_reporting: bool = False
    supports_perplexity_reporting: bool = False
    supports_streaming_api_server: bool = False
    supports_kv_connector_warmup: bool = False
    supports_prefill_warmup_batch: bool = False
    supports_model_load_warmup: bool = False
    storage_tiers_supported: list[str] = field(default_factory=lambda: ["Hbm", "CpuRam"])
    delegate_backend: str | None = None
    placeholder: str | None = None
    radix_cache_kind: str | None = None
    radix_total_tokens: int | None = None
    radix_protected_tokens: int | None = None
    radix_evictable_tokens: int | None = None
    radix_page_size: int | None = None
    storage_backend: str | None = None
    storage_supports_stats: bool | None = None
    storage_supports_zero_copy: bool | None = None
    storage_layout_mode: str | None = None
    storage_transfer_mem_type: str | None = None
    warmup_kv_connector_initialized: bool | None = None
    warmup_prefill_batch_executed: bool | None = None
    attention_kernel_capabilities: dict[str, object] | None = None
    amd_kv_kernel_profile: dict[str, object] | None = None
    adaptive_recommendation: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def with_delegate_backend(self, delegate_backend: str | None) -> "EngineRuntimeProfile":
        return replace(self, delegate_backend=delegate_backend)

    def with_placeholder(self, placeholder: str | None) -> "EngineRuntimeProfile":
        return replace(self, placeholder=placeholder)

    def with_radix_cache_state(
        self,
        radix_cache_kind: str | None,
        radix_total_tokens: int | None,
        radix_protected_tokens: int | None,
        radix_evictable_tokens: int | None,
        radix_page_size: int | None,
    ) -> "EngineRuntimeProfile":
        return replace(
            self,
            radix_cache_kind=radix_cache_kind,
            radix_total_tokens=radix_total_tokens,
            radix_protected_tokens=radix_protected_tokens,
            radix_evictable_tokens=radix_evictable_tokens,
            radix_page_size=radix_page_size,
        )

    def with_fp8_kv_cache_state(
        self,
        *,
        supports_fp8_kv_cache: bool,
        supports_fp8_kv_per_tensor_scales: bool,
        supports_fp8_kv_per_head_scales: bool,
        supports_kv_scale_calibration: bool,
        supports_quantized_attention_fusion: bool,
    ) -> "EngineRuntimeProfile":
        return replace(
            self,
            supports_fp8_kv_cache=supports_fp8_kv_cache,
            supports_fp8_kv_per_tensor_scales=supports_fp8_kv_per_tensor_scales,
            supports_fp8_kv_per_head_scales=supports_fp8_kv_per_head_scales,
            supports_kv_scale_calibration=supports_kv_scale_calibration,
            supports_quantized_attention_fusion=supports_quantized_attention_fusion,
        )

    def with_storage_backend_state(
        self,
        storage_backend: str | None,
        storage_supports_stats: bool | None,
        storage_supports_zero_copy: bool | None,
        storage_layout_mode: str | None,
        storage_transfer_mem_type: str | None = None,
    ) -> "EngineRuntimeProfile":
        return replace(
            self,
            storage_backend=storage_backend,
            storage_supports_stats=storage_supports_stats,
            storage_supports_zero_copy=storage_supports_zero_copy,
            storage_layout_mode=storage_layout_mode,
            storage_transfer_mem_type=storage_transfer_mem_type,
        )

    def with_adaptive_recommendation(
        self, adaptive_recommendation: dict[str, object] | None
    ) -> "EngineRuntimeProfile":
        return replace(self, adaptive_recommendation=adaptive_recommendation)

    def with_model_routing_state(
        self,
        *,
        supports_model_hot_swap: bool,
        supports_model_aliases: bool,
        supports_model_groups: bool,
        supports_ttl_unload: bool,
        supports_request_filters: bool,
        supports_config_reload: bool,
        supports_direct_model_passthrough: bool,
    ) -> "EngineRuntimeProfile":
        return replace(
            self,
            supports_model_hot_swap=supports_model_hot_swap,
            supports_model_aliases=supports_model_aliases,
            supports_model_groups=supports_model_groups,
            supports_ttl_unload=supports_ttl_unload,
            supports_request_filters=supports_request_filters,
            supports_config_reload=supports_config_reload,
            supports_direct_model_passthrough=supports_direct_model_passthrough,
        )

    def with_model_packing_state(
        self,
        *,
        supports_dynamic_model_loading: bool,
        supports_dynamic_model_unloading: bool,
        supports_multi_model_packing: bool,
        supports_multi_gpu_distribution: bool,
        supports_kvcached_memory_sharing: bool,
        supports_model_sleep_mode: bool,
        supports_model_move_operations: bool,
        supports_gpu_memory_telemetry: bool,
    ) -> "EngineRuntimeProfile":
        return replace(
            self,
            supports_dynamic_model_loading=supports_dynamic_model_loading,
            supports_dynamic_model_unloading=supports_dynamic_model_unloading,
            supports_multi_model_packing=supports_multi_model_packing,
            supports_multi_gpu_distribution=supports_multi_gpu_distribution,
            supports_kvcached_memory_sharing=supports_kvcached_memory_sharing,
            supports_model_sleep_mode=supports_model_sleep_mode,
            supports_model_move_operations=supports_model_move_operations,
            supports_gpu_memory_telemetry=supports_gpu_memory_telemetry,
        )

    def with_layer_offloading_state(
        self, supports_layer_offloading: bool
    ) -> "EngineRuntimeProfile":
        return replace(
            self,
            supports_layer_offloading=supports_layer_offloading,
        )

    def with_gfxgraph_bridge_state(
        self,
        *,
        supports_graph_shape_bucketing: bool,
        supports_graph_validation_mode: bool,
        supports_graph_conditional_nodes: bool,
        supports_graph_nested_capture: bool,
    ) -> "EngineRuntimeProfile":
        return replace(
            self,
            supports_graph_shape_bucketing=supports_graph_shape_bucketing,
            supports_graph_validation_mode=supports_graph_validation_mode,
            supports_graph_conditional_nodes=supports_graph_conditional_nodes,
            supports_graph_nested_capture=supports_graph_nested_capture,
        )

    def with_vllm_runtime_family_state(
        self,
        *,
        supports_multimodal_serving: bool,
        supports_omni_modality: bool,
        supports_hardware_plugin_interface: bool,
    ) -> "EngineRuntimeProfile":
        return replace(
            self,
            supports_multimodal_serving=supports_multimodal_serving,
            supports_omni_modality=supports_omni_modality,
            supports_hardware_plugin_interface=supports_hardware_plugin_interface,
        )

    def supports_kv_quantization_mode(self, mode: str) -> bool:
        normalized = mode.strip().lower()
        if normalized == "none":
            return True
        if normalized in {"fp8_e4m3", "fp8_e5m2"}:
            return self.supports_fp8_kv_cache
        if normalized == "turboquant":
            return self.supports_turboquant_kv
        if normalized == "rotorquant":
            return self.supports_rotorquant_kv
        if normalized in {"int8", "int4"}:
            return self.supports_atom_kv_quant or self.supports_kv_cache_quantization_pipeline
        if normalized == "engine_native":
            return True
        return False

    def supported_kv_quantization_modes(self) -> list[str]:
        modes: list[str] = []
        if self.supports_fp8_kv_cache:
            modes.extend(["fp8_e4m3", "fp8_e5m2"])
        if self.supports_turboquant_kv:
            modes.append("turboquant")
        if self.supports_rotorquant_kv:
            modes.append("rotorquant")
        if self.supports_atom_kv_quant or self.supports_kv_cache_quantization_pipeline:
            modes.extend(["int8", "int4"])
        return modes

    def with_compact_runtime_state(
        self,
        *,
        supports_cpu_only_runtime: bool,
        supports_download_on_first_use: bool,
        supports_ollama_style_cli: bool,
        supports_openai_compatible_server: bool,
        supports_progressive_kv_compression: bool,
        supports_full_document_mode: bool,
    ) -> "EngineRuntimeProfile":
        return replace(
            self,
            supports_cpu_only_runtime=supports_cpu_only_runtime,
            supports_download_on_first_use=supports_download_on_first_use,
            supports_ollama_style_cli=supports_ollama_style_cli,
            supports_openai_compatible_server=supports_openai_compatible_server,
            supports_progressive_kv_compression=supports_progressive_kv_compression,
            supports_full_document_mode=supports_full_document_mode,
        )

    def with_storage_orchestration_state(
        self,
        *,
        supports_distributed_memory_pooling: bool,
        supports_dynamic_multilevel_caching: bool,
        supports_global_metadata_management: bool,
        supports_capacity_management: bool,
        supports_prefix_matching: bool,
        supports_sliding_window_matching: bool,
        supports_kv_matching: bool,
        supports_two_phase_write: bool,
        supports_async_eviction: bool,
        supports_trace_replay_optimization: bool,
    ) -> "EngineRuntimeProfile":
        return replace(
            self,
            supports_distributed_memory_pooling=supports_distributed_memory_pooling,
            supports_dynamic_multilevel_caching=supports_dynamic_multilevel_caching,
            supports_global_metadata_management=supports_global_metadata_management,
            supports_capacity_management=supports_capacity_management,
            supports_prefix_matching=supports_prefix_matching,
            supports_sliding_window_matching=supports_sliding_window_matching,
            supports_kv_matching=supports_kv_matching,
            supports_two_phase_write=supports_two_phase_write,
            supports_async_eviction=supports_async_eviction,
            supports_trace_replay_optimization=supports_trace_replay_optimization,
        )

    def with_quantization_pipeline_state(
        self,
        *,
        supports_model_free_ptq: bool,
        supports_compressed_tensors_format: bool,
        supports_weight_quantization_pipeline: bool,
        supports_activation_quantization_pipeline: bool,
        supports_kv_cache_quantization_pipeline: bool,
        supports_attention_quantization_pipeline: bool,
        supports_disk_offloading_quantization: bool,
        supports_distributed_calibration: bool,
    ) -> "EngineRuntimeProfile":
        return replace(
            self,
            supports_model_free_ptq=supports_model_free_ptq,
            supports_compressed_tensors_format=supports_compressed_tensors_format,
            supports_weight_quantization_pipeline=supports_weight_quantization_pipeline,
            supports_activation_quantization_pipeline=supports_activation_quantization_pipeline,
            supports_kv_cache_quantization_pipeline=supports_kv_cache_quantization_pipeline,
            supports_attention_quantization_pipeline=supports_attention_quantization_pipeline,
            supports_disk_offloading_quantization=supports_disk_offloading_quantization,
            supports_distributed_calibration=supports_distributed_calibration,
        )

    def with_serving_benchmark_state(
        self,
        *,
        supports_rust_native_gpu_serving: bool,
        supports_pure_jax_tpu_serving: bool,
        supports_cross_device_benchmarking: bool,
        supports_single_graph_capture: bool,
        supports_graph_shape_bucketing: bool,
        supports_graph_validation_mode: bool,
        supports_graph_conditional_nodes: bool,
        supports_graph_nested_capture: bool,
        supports_fp8_channelscale_epilogue: bool,
        supports_cached_ttft_reporting: bool,
        supports_peak_throughput_reporting: bool,
        supports_perplexity_reporting: bool,
        supports_streaming_api_server: bool,
    ) -> "EngineRuntimeProfile":
        return replace(
            self,
            supports_rust_native_gpu_serving=supports_rust_native_gpu_serving,
            supports_pure_jax_tpu_serving=supports_pure_jax_tpu_serving,
            supports_cross_device_benchmarking=supports_cross_device_benchmarking,
            supports_single_graph_capture=supports_single_graph_capture,
            supports_graph_shape_bucketing=supports_graph_shape_bucketing,
            supports_graph_validation_mode=supports_graph_validation_mode,
            supports_graph_conditional_nodes=supports_graph_conditional_nodes,
            supports_graph_nested_capture=supports_graph_nested_capture,
            supports_fp8_channelscale_epilogue=supports_fp8_channelscale_epilogue,
            supports_cached_ttft_reporting=supports_cached_ttft_reporting,
            supports_peak_throughput_reporting=supports_peak_throughput_reporting,
            supports_perplexity_reporting=supports_perplexity_reporting,
            supports_streaming_api_server=supports_streaming_api_server,
        )

    def with_warmup_initialization_state(
        self,
        *,
        supports_kv_connector_warmup: bool,
        supports_prefill_warmup_batch: bool,
        supports_model_load_warmup: bool,
        warmup_kv_connector_initialized: bool | None = None,
        warmup_prefill_batch_executed: bool | None = None,
    ) -> "EngineRuntimeProfile":
        return replace(
            self,
            supports_kv_connector_warmup=supports_kv_connector_warmup,
            supports_prefill_warmup_batch=supports_prefill_warmup_batch,
            supports_model_load_warmup=supports_model_load_warmup,
            warmup_kv_connector_initialized=warmup_kv_connector_initialized,
            warmup_prefill_batch_executed=warmup_prefill_batch_executed,
        )
