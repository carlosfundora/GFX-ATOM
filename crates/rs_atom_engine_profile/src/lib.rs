use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum KvStorageTier {
    Hbm,
    CpuRam,
    Lmcache,
    Nvme,
    ObjectStore,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum KvQuantizationMode {
    None,
    Fp8E4M3,
    Fp8E5M2,
    Int8,
    Int4,
    TurboQuant,
    RotorQuant,
    EngineNative,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum KvCompressionMode {
    None,
    CacheGen,
    ZstdSerde,
    LossyPack,
    SparseEviction,
    HybridEvictQuantize,
    EngineNative,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AttentionKernelCapabilities {
    pub supports_prefix_aware_attention: bool,
    pub supports_multi_tile_decode: bool,
    pub supports_shared_prefix_query_packing: bool,
    pub supports_kv_splitting: bool,
    pub supports_online_softmax_merge: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AmdKvKernelProfile {
    pub gfx_arch: String,
    pub wavefront_size: u32,
    pub lds_staging_enabled: bool,
    pub split_k_enabled: bool,
    pub paged_kv_traversal_enabled: bool,
    pub branch_packing_enabled: bool,
    pub hsa_queue_priority: String,
    pub svm_page_migration_enabled: bool,
    pub rocprofiler_enabled: bool,
    pub rocm_smi_polling_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct KvDecodeTelemetryBundle {
    pub request_id: String,
    pub session_id: String,
    pub model_id: String,
    pub engine_id: String,
    pub shape_key: String,
    pub prompt_tokens: u32,
    pub decode_tokens: u32,
    pub prefix_reuse_ratio: f32,
    pub kv_hit_rate: f32,
    pub kv_miss_count: u64,
    pub branch_entropy_mean: Option<f32>,
    pub branch_entropy_max: Option<f32>,
    pub mismatch_depth_mean: Option<f32>,
    pub mismatch_depth_std: Option<f32>,
    pub early_exit_ratio: Option<f32>,
    pub acceptance_rate: Option<f32>,
    pub token_surprisal_mean: Option<f32>,
    pub repeated_token_count: Option<u32>,
    pub calibration_drift_score: Option<f32>,
    pub gpu_memory_used_mb: Option<u64>,
    pub memory_bandwidth_gbps: Option<f32>,
    pub kernel_launch_overhead_ms: Option<f32>,
    pub decode_tps: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct KvObjectRecord {
    pub kv_object_id: String,
    pub cache_key: String,
    pub content_hash: String,
    pub prefix_hash: String,
    pub model_id: String,
    pub engine_id: String,
    pub tokenizer_id: String,
    pub tokenizer_revision: String,
    pub token_start: u32,
    pub token_count: u32,
    pub layer_start: u32,
    pub layer_count: u32,
    pub dtype: String,
    pub compression_mode: KvCompressionMode,
    pub quantization_mode: KvQuantizationMode,
    pub storage_tier: KvStorageTier,
    pub location_uri: String,
    pub reusable_scope: String,
    pub pin_state: String,
    pub eviction_priority: f32,
    pub future_use_score: f32,
    pub metadata_registry: String,
    pub control_plane_store: String,
    pub tensor_store: String,
    pub allow_raw_kv_in_surrealdb: bool,
    pub allow_raw_kv_in_valkey: bool,
    pub created_at_rfc3339: String,
    pub last_used_at_rfc3339: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PrecomputedContextAsset {
    pub asset_id: String,
    pub asset_type: String,
    pub profile_id: String,
    pub content_hash: String,
    pub token_count: u32,
    pub pinned: bool,
    pub storage_tiers_supported: Vec<KvStorageTier>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PrecomputedKvAsset {
    pub kv_asset_id: String,
    pub context_asset_id: String,
    pub kv_object_id: String,
    pub quantization_mode: KvQuantizationMode,
    pub compression_mode: KvCompressionMode,
    pub calibration_status: String,
    pub calibration_dataset_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct KvTransferPlan {
    pub plan_id: String,
    pub source_tier: KvStorageTier,
    pub destination_tier: KvStorageTier,
    pub kv_object_ids: Vec<String>,
    pub disaggregated_prefill_decode: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct KvPrefetchPlan {
    pub plan_id: String,
    pub upcoming_agent_step_ids: Vec<String>,
    pub prefetch_kv_object_ids: Vec<String>,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct KvEvictionDecision {
    pub decision_id: String,
    pub evict_kv_object_ids: Vec<String>,
    pub preserve_kv_object_ids: Vec<String>,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AgentStepGraph {
    pub graph_id: String,
    pub active_step_id: String,
    pub next_likely_step_ids: Vec<String>,
    pub required_kv_object_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelRuntimeProfile {
    pub supports_automatic_prefix_caching: bool,
    pub supports_radix_cache: bool,
    pub supports_lmcache_connector: bool,
    pub supports_kv_events: bool,
    pub supports_fp8_kv_cache: bool,
    pub supports_turboquant_kv: bool,
    pub supports_rotorquant_kv: bool,
    pub supports_eagle3: bool,
    pub supports_medusa: bool,
    pub supports_ngram_speculation: bool,
    pub supports_phantom: bool,
    pub supports_phantom_x: bool,
    pub supports_disaggregated_prefill: bool,
    pub supports_disaggregated_decode: bool,
    pub supports_prefix_aware_attention: bool,
    pub supports_content_addressed_cache: bool,
    pub supports_position_independent_cache: bool,
    pub storage_tiers_supported: Vec<KvStorageTier>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EngineRuntimeProfile {
    pub supports_atom_backend: bool,
    pub supports_atom_attention: bool,
    pub supports_atom_kv_quant: bool,
    pub supports_atom_rocm_telemetry: bool,
    pub supports_atom_fallback: bool,
    pub supports_automatic_prefix_caching: bool,
    pub supports_radix_cache: bool,
    pub supports_lmcache_connector: bool,
    pub supports_kv_events: bool,
    pub supports_fp8_kv_cache: bool,
    pub supports_fp8_kv_per_tensor_scales: bool,
    pub supports_fp8_kv_per_head_scales: bool,
    pub supports_kv_scale_calibration: bool,
    pub supports_quantized_attention_fusion: bool,
    pub supports_turboquant_kv: bool,
    pub supports_rotorquant_kv: bool,
    pub supports_eagle3: bool,
    pub supports_medusa: bool,
    pub supports_ngram_speculation: bool,
    pub supports_phantom: bool,
    pub supports_phantom_x: bool,
    pub supports_disaggregated_prefill: bool,
    pub supports_disaggregated_decode: bool,
    pub supports_prefix_aware_attention: bool,
    pub supports_content_addressed_cache: bool,
    pub supports_position_independent_cache: bool,
    pub supports_model_hot_swap: bool,
    pub supports_model_aliases: bool,
    pub supports_model_groups: bool,
    pub supports_ttl_unload: bool,
    pub supports_request_filters: bool,
    pub supports_config_reload: bool,
    pub supports_direct_model_passthrough: bool,
    pub supports_dynamic_model_loading: bool,
    pub supports_dynamic_model_unloading: bool,
    pub supports_multi_model_packing: bool,
    pub supports_multi_gpu_distribution: bool,
    pub supports_kvcached_memory_sharing: bool,
    pub supports_model_sleep_mode: bool,
    pub supports_model_move_operations: bool,
    pub supports_layer_offloading: bool,
    pub supports_gpu_memory_telemetry: bool,
    pub supports_cpu_only_runtime: bool,
    pub supports_download_on_first_use: bool,
    pub supports_ollama_style_cli: bool,
    pub supports_openai_compatible_server: bool,
    pub supports_progressive_kv_compression: bool,
    pub supports_full_document_mode: bool,
    pub supports_distributed_memory_pooling: bool,
    pub supports_dynamic_multilevel_caching: bool,
    pub supports_global_metadata_management: bool,
    pub supports_capacity_management: bool,
    pub supports_prefix_matching: bool,
    pub supports_sliding_window_matching: bool,
    pub supports_kv_matching: bool,
    pub supports_two_phase_write: bool,
    pub supports_async_eviction: bool,
    pub supports_trace_replay_optimization: bool,
    pub supports_model_free_ptq: bool,
    pub supports_compressed_tensors_format: bool,
    pub supports_weight_quantization_pipeline: bool,
    pub supports_activation_quantization_pipeline: bool,
    pub supports_kv_cache_quantization_pipeline: bool,
    pub supports_attention_quantization_pipeline: bool,
    pub supports_disk_offloading_quantization: bool,
    pub supports_distributed_calibration: bool,
    pub supports_rust_native_gpu_serving: bool,
    pub supports_pure_jax_tpu_serving: bool,
    pub supports_cross_device_benchmarking: bool,
    pub supports_single_graph_capture: bool,
    pub supports_multimodal_serving: bool,
    pub supports_omni_modality: bool,
    pub supports_hardware_plugin_interface: bool,
    pub supports_graph_shape_bucketing: bool,
    pub supports_graph_validation_mode: bool,
    pub supports_graph_conditional_nodes: bool,
    pub supports_graph_nested_capture: bool,
    pub supports_fp8_channelscale_epilogue: bool,
    pub supports_cached_ttft_reporting: bool,
    pub supports_peak_throughput_reporting: bool,
    pub supports_perplexity_reporting: bool,
    pub supports_streaming_api_server: bool,
    pub supports_kv_connector_warmup: bool,
    pub supports_prefill_warmup_batch: bool,
    pub supports_model_load_warmup: bool,
    pub storage_tiers_supported: Vec<KvStorageTier>,
    pub delegate_backend: Option<String>,
    pub placeholder: Option<String>,
    pub radix_cache_kind: Option<String>,
    pub radix_total_tokens: Option<u64>,
    pub radix_protected_tokens: Option<u64>,
    pub radix_evictable_tokens: Option<u64>,
    pub radix_page_size: Option<u32>,
    pub storage_backend: Option<String>,
    pub storage_supports_stats: Option<bool>,
    pub storage_supports_zero_copy: Option<bool>,
    pub storage_layout_mode: Option<String>,
    pub storage_transfer_mem_type: Option<String>,
    pub warmup_kv_connector_initialized: Option<bool>,
    pub warmup_prefill_batch_executed: Option<bool>,
    pub attention_kernel_capabilities: Option<AttentionKernelCapabilities>,
    pub amd_kv_kernel_profile: Option<AmdKvKernelProfile>,
    pub adaptive_recommendation: Option<String>,
}

impl Default for ModelRuntimeProfile {
    fn default() -> Self {
        Self {
            supports_automatic_prefix_caching: true,
            supports_radix_cache: true,
            supports_lmcache_connector: false,
            supports_kv_events: true,
            supports_fp8_kv_cache: false,
            supports_turboquant_kv: false,
            supports_rotorquant_kv: false,
            supports_eagle3: false,
            supports_medusa: false,
            supports_ngram_speculation: false,
            supports_phantom: false,
            supports_phantom_x: false,
            supports_disaggregated_prefill: false,
            supports_disaggregated_decode: false,
            supports_prefix_aware_attention: false,
            supports_content_addressed_cache: false,
            supports_position_independent_cache: false,
            storage_tiers_supported: vec![KvStorageTier::Hbm, KvStorageTier::CpuRam],
        }
    }
}

impl Default for EngineRuntimeProfile {
    fn default() -> Self {
        Self {
            supports_atom_backend: true,
            supports_atom_attention: true,
            supports_atom_kv_quant: false,
            supports_atom_rocm_telemetry: true,
            supports_atom_fallback: true,
            supports_automatic_prefix_caching: true,
            supports_radix_cache: true,
            supports_lmcache_connector: false,
            supports_kv_events: true,
            supports_fp8_kv_cache: false,
            supports_fp8_kv_per_tensor_scales: false,
            supports_fp8_kv_per_head_scales: false,
            supports_kv_scale_calibration: false,
            supports_quantized_attention_fusion: false,
            supports_turboquant_kv: false,
            supports_rotorquant_kv: false,
            supports_eagle3: false,
            supports_medusa: false,
            supports_ngram_speculation: false,
            supports_phantom: false,
            supports_phantom_x: false,
            supports_disaggregated_prefill: false,
            supports_disaggregated_decode: false,
            supports_prefix_aware_attention: false,
            supports_content_addressed_cache: false,
            supports_position_independent_cache: false,
            supports_model_hot_swap: false,
            supports_model_aliases: false,
            supports_model_groups: false,
            supports_ttl_unload: false,
            supports_request_filters: false,
            supports_config_reload: false,
            supports_direct_model_passthrough: false,
            supports_dynamic_model_loading: false,
            supports_dynamic_model_unloading: false,
            supports_multi_model_packing: false,
            supports_multi_gpu_distribution: false,
            supports_kvcached_memory_sharing: false,
            supports_model_sleep_mode: false,
            supports_model_move_operations: false,
            supports_layer_offloading: false,
            supports_gpu_memory_telemetry: false,
            supports_cpu_only_runtime: false,
            supports_download_on_first_use: false,
            supports_ollama_style_cli: false,
            supports_openai_compatible_server: false,
            supports_progressive_kv_compression: false,
            supports_full_document_mode: false,
            supports_distributed_memory_pooling: true,
            supports_dynamic_multilevel_caching: true,
            supports_global_metadata_management: false,
            supports_capacity_management: false,
            supports_prefix_matching: false,
            supports_sliding_window_matching: false,
            supports_kv_matching: true,
            supports_two_phase_write: false,
            supports_async_eviction: true,
            supports_trace_replay_optimization: false,
            supports_model_free_ptq: false,
            supports_compressed_tensors_format: false,
            supports_weight_quantization_pipeline: false,
            supports_activation_quantization_pipeline: false,
            supports_kv_cache_quantization_pipeline: false,
            supports_attention_quantization_pipeline: false,
            supports_disk_offloading_quantization: false,
            supports_distributed_calibration: false,
            supports_rust_native_gpu_serving: false,
            supports_pure_jax_tpu_serving: false,
            supports_cross_device_benchmarking: false,
            supports_single_graph_capture: false,
            supports_multimodal_serving: false,
            supports_omni_modality: false,
            supports_hardware_plugin_interface: false,
            supports_graph_shape_bucketing: false,
            supports_graph_validation_mode: false,
            supports_graph_conditional_nodes: false,
            supports_graph_nested_capture: false,
            supports_fp8_channelscale_epilogue: false,
            supports_cached_ttft_reporting: false,
            supports_peak_throughput_reporting: false,
            supports_perplexity_reporting: false,
            supports_streaming_api_server: false,
            supports_kv_connector_warmup: false,
            supports_prefill_warmup_batch: false,
            supports_model_load_warmup: false,
            storage_tiers_supported: vec![KvStorageTier::Hbm, KvStorageTier::CpuRam],
            delegate_backend: None,
            placeholder: None,
            radix_cache_kind: None,
            radix_total_tokens: None,
            radix_protected_tokens: None,
            radix_evictable_tokens: None,
            radix_page_size: None,
            storage_backend: None,
            storage_supports_stats: None,
            storage_supports_zero_copy: None,
            storage_layout_mode: None,
            storage_transfer_mem_type: None,
            warmup_kv_connector_initialized: None,
            warmup_prefill_batch_executed: None,
            attention_kernel_capabilities: None,
            amd_kv_kernel_profile: None,
            adaptive_recommendation: None,
        }
    }
}

impl EngineRuntimeProfile {
    pub fn with_delegate_backend(mut self, delegate_backend: impl Into<String>) -> Self {
        self.delegate_backend = Some(delegate_backend.into());
        self
    }

    pub fn with_placeholder(mut self, placeholder: impl Into<String>) -> Self {
        self.placeholder = Some(placeholder.into());
        self
    }

    pub fn with_adaptive_recommendation(
        mut self,
        adaptive_recommendation: impl Into<String>,
    ) -> Self {
        self.adaptive_recommendation = Some(adaptive_recommendation.into());
        self
    }

    pub fn with_radix_cache_state(
        mut self,
        radix_cache_kind: impl Into<String>,
        radix_total_tokens: u64,
        radix_protected_tokens: u64,
        radix_evictable_tokens: u64,
        radix_page_size: u32,
    ) -> Self {
        self.radix_cache_kind = Some(radix_cache_kind.into());
        self.radix_total_tokens = Some(radix_total_tokens);
        self.radix_protected_tokens = Some(radix_protected_tokens);
        self.radix_evictable_tokens = Some(radix_evictable_tokens);
        self.radix_page_size = Some(radix_page_size);
        self
    }

    pub fn with_fp8_kv_cache_state(
        mut self,
        supports_fp8_kv_cache: bool,
        supports_fp8_kv_per_tensor_scales: bool,
        supports_fp8_kv_per_head_scales: bool,
        supports_kv_scale_calibration: bool,
        supports_quantized_attention_fusion: bool,
    ) -> Self {
        self.supports_fp8_kv_cache = supports_fp8_kv_cache;
        self.supports_fp8_kv_per_tensor_scales = supports_fp8_kv_per_tensor_scales;
        self.supports_fp8_kv_per_head_scales = supports_fp8_kv_per_head_scales;
        self.supports_kv_scale_calibration = supports_kv_scale_calibration;
        self.supports_quantized_attention_fusion = supports_quantized_attention_fusion;
        self
    }

    pub fn with_storage_backend_state(
        mut self,
        storage_backend: impl Into<String>,
        storage_supports_stats: bool,
        storage_supports_zero_copy: bool,
        storage_layout_mode: impl Into<String>,
        storage_transfer_mem_type: Option<String>,
    ) -> Self {
        self.storage_backend = Some(storage_backend.into());
        self.storage_supports_stats = Some(storage_supports_stats);
        self.storage_supports_zero_copy = Some(storage_supports_zero_copy);
        self.storage_layout_mode = Some(storage_layout_mode.into());
        self.storage_transfer_mem_type = storage_transfer_mem_type;
        self
    }

    pub fn with_model_routing_state(
        mut self,
        supports_model_hot_swap: bool,
        supports_model_aliases: bool,
        supports_model_groups: bool,
        supports_ttl_unload: bool,
        supports_request_filters: bool,
        supports_config_reload: bool,
        supports_direct_model_passthrough: bool,
    ) -> Self {
        self.supports_model_hot_swap = supports_model_hot_swap;
        self.supports_model_aliases = supports_model_aliases;
        self.supports_model_groups = supports_model_groups;
        self.supports_ttl_unload = supports_ttl_unload;
        self.supports_request_filters = supports_request_filters;
        self.supports_config_reload = supports_config_reload;
        self.supports_direct_model_passthrough = supports_direct_model_passthrough;
        self
    }

    pub fn with_model_packing_state(
        mut self,
        supports_dynamic_model_loading: bool,
        supports_dynamic_model_unloading: bool,
        supports_multi_model_packing: bool,
        supports_multi_gpu_distribution: bool,
        supports_kvcached_memory_sharing: bool,
        supports_model_sleep_mode: bool,
        supports_model_move_operations: bool,
        supports_gpu_memory_telemetry: bool,
    ) -> Self {
        self.supports_dynamic_model_loading = supports_dynamic_model_loading;
        self.supports_dynamic_model_unloading = supports_dynamic_model_unloading;
        self.supports_multi_model_packing = supports_multi_model_packing;
        self.supports_multi_gpu_distribution = supports_multi_gpu_distribution;
        self.supports_kvcached_memory_sharing = supports_kvcached_memory_sharing;
        self.supports_model_sleep_mode = supports_model_sleep_mode;
        self.supports_model_move_operations = supports_model_move_operations;
        self.supports_gpu_memory_telemetry = supports_gpu_memory_telemetry;
        self
    }

    pub fn with_layer_offloading_state(mut self, supports_layer_offloading: bool) -> Self {
        self.supports_layer_offloading = supports_layer_offloading;
        self
    }

    pub fn with_compact_runtime_state(
        mut self,
        supports_cpu_only_runtime: bool,
        supports_download_on_first_use: bool,
        supports_ollama_style_cli: bool,
        supports_openai_compatible_server: bool,
        supports_progressive_kv_compression: bool,
        supports_full_document_mode: bool,
    ) -> Self {
        self.supports_cpu_only_runtime = supports_cpu_only_runtime;
        self.supports_download_on_first_use = supports_download_on_first_use;
        self.supports_ollama_style_cli = supports_ollama_style_cli;
        self.supports_openai_compatible_server = supports_openai_compatible_server;
        self.supports_progressive_kv_compression = supports_progressive_kv_compression;
        self.supports_full_document_mode = supports_full_document_mode;
        self
    }

    pub fn with_storage_orchestration_state(
        mut self,
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
    ) -> Self {
        self.supports_distributed_memory_pooling = supports_distributed_memory_pooling;
        self.supports_dynamic_multilevel_caching = supports_dynamic_multilevel_caching;
        self.supports_global_metadata_management = supports_global_metadata_management;
        self.supports_capacity_management = supports_capacity_management;
        self.supports_prefix_matching = supports_prefix_matching;
        self.supports_sliding_window_matching = supports_sliding_window_matching;
        self.supports_kv_matching = supports_kv_matching;
        self.supports_two_phase_write = supports_two_phase_write;
        self.supports_async_eviction = supports_async_eviction;
        self.supports_trace_replay_optimization = supports_trace_replay_optimization;
        self
    }

    pub fn with_quantization_pipeline_state(
        mut self,
        supports_model_free_ptq: bool,
        supports_compressed_tensors_format: bool,
        supports_weight_quantization_pipeline: bool,
        supports_activation_quantization_pipeline: bool,
        supports_kv_cache_quantization_pipeline: bool,
        supports_attention_quantization_pipeline: bool,
        supports_disk_offloading_quantization: bool,
        supports_distributed_calibration: bool,
    ) -> Self {
        self.supports_model_free_ptq = supports_model_free_ptq;
        self.supports_compressed_tensors_format = supports_compressed_tensors_format;
        self.supports_weight_quantization_pipeline = supports_weight_quantization_pipeline;
        self.supports_activation_quantization_pipeline = supports_activation_quantization_pipeline;
        self.supports_kv_cache_quantization_pipeline = supports_kv_cache_quantization_pipeline;
        self.supports_attention_quantization_pipeline = supports_attention_quantization_pipeline;
        self.supports_disk_offloading_quantization = supports_disk_offloading_quantization;
        self.supports_distributed_calibration = supports_distributed_calibration;
        self
    }

    pub fn with_serving_benchmark_state(
        mut self,
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
    ) -> Self {
        self.supports_rust_native_gpu_serving = supports_rust_native_gpu_serving;
        self.supports_pure_jax_tpu_serving = supports_pure_jax_tpu_serving;
        self.supports_cross_device_benchmarking = supports_cross_device_benchmarking;
        self.supports_single_graph_capture = supports_single_graph_capture;
        self.supports_graph_shape_bucketing = supports_graph_shape_bucketing;
        self.supports_graph_validation_mode = supports_graph_validation_mode;
        self.supports_graph_conditional_nodes = supports_graph_conditional_nodes;
        self.supports_graph_nested_capture = supports_graph_nested_capture;
        self.supports_fp8_channelscale_epilogue = supports_fp8_channelscale_epilogue;
        self.supports_cached_ttft_reporting = supports_cached_ttft_reporting;
        self.supports_peak_throughput_reporting = supports_peak_throughput_reporting;
        self.supports_perplexity_reporting = supports_perplexity_reporting;
        self.supports_streaming_api_server = supports_streaming_api_server;
        self
    }

    pub fn with_gfxgraph_bridge_state(
        mut self,
        supports_graph_shape_bucketing: bool,
        supports_graph_validation_mode: bool,
        supports_graph_conditional_nodes: bool,
        supports_graph_nested_capture: bool,
    ) -> Self {
        self.supports_graph_shape_bucketing = supports_graph_shape_bucketing;
        self.supports_graph_validation_mode = supports_graph_validation_mode;
        self.supports_graph_conditional_nodes = supports_graph_conditional_nodes;
        self.supports_graph_nested_capture = supports_graph_nested_capture;
        self
    }

    pub fn with_vllm_runtime_family_state(
        mut self,
        supports_multimodal_serving: bool,
        supports_omni_modality: bool,
        supports_hardware_plugin_interface: bool,
    ) -> Self {
        self.supports_multimodal_serving = supports_multimodal_serving;
        self.supports_omni_modality = supports_omni_modality;
        self.supports_hardware_plugin_interface = supports_hardware_plugin_interface;
        self
    }

    pub fn supports_kv_quantization_mode(&self, mode: &KvQuantizationMode) -> bool {
        match mode {
            KvQuantizationMode::None => true,
            KvQuantizationMode::Fp8E4M3 | KvQuantizationMode::Fp8E5M2 => {
                self.supports_fp8_kv_cache
            }
            KvQuantizationMode::TurboQuant => self.supports_turboquant_kv,
            KvQuantizationMode::RotorQuant => self.supports_rotorquant_kv,
            KvQuantizationMode::Int8 | KvQuantizationMode::Int4 => {
                self.supports_atom_kv_quant || self.supports_kv_cache_quantization_pipeline
            }
            KvQuantizationMode::EngineNative => true,
            KvQuantizationMode::Custom(_) => false,
        }
    }

    pub fn supported_kv_quantization_modes(&self) -> Vec<KvQuantizationMode> {
        let mut modes = Vec::new();
        if self.supports_fp8_kv_cache {
            modes.push(KvQuantizationMode::Fp8E4M3);
            modes.push(KvQuantizationMode::Fp8E5M2);
        }
        if self.supports_turboquant_kv {
            modes.push(KvQuantizationMode::TurboQuant);
        }
        if self.supports_rotorquant_kv {
            modes.push(KvQuantizationMode::RotorQuant);
        }
        if self.supports_atom_kv_quant || self.supports_kv_cache_quantization_pipeline {
            modes.push(KvQuantizationMode::Int8);
            modes.push(KvQuantizationMode::Int4);
        }
        modes
    }

    pub fn with_warmup_initialization_state(
        mut self,
        kv_connector_initialized: bool,
        prefill_batch_executed: bool,
    ) -> Self {
        self.warmup_kv_connector_initialized = Some(kv_connector_initialized);
        self.warmup_prefill_batch_executed = Some(prefill_batch_executed);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_profile_has_expected_flags() {
        let profile = EngineRuntimeProfile::default();
        assert!(profile.supports_atom_backend);
        assert!(profile.supports_atom_attention);
        assert!(profile.supports_automatic_prefix_caching);
        assert!(profile.supports_radix_cache);
        assert!(profile.supports_kv_events);
        assert!(!profile.supports_fp8_kv_cache);
        assert!(!profile.supports_fp8_kv_per_tensor_scales);
        assert!(!profile.supports_fp8_kv_per_head_scales);
        assert!(!profile.supports_kv_scale_calibration);
        assert!(!profile.supports_quantized_attention_fusion);
        assert!(!profile.supports_model_hot_swap);
        assert!(!profile.supports_model_aliases);
        assert!(!profile.supports_model_groups);
        assert!(!profile.supports_ttl_unload);
        assert!(!profile.supports_request_filters);
        assert!(!profile.supports_config_reload);
        assert!(!profile.supports_direct_model_passthrough);
        assert!(!profile.supports_dynamic_model_loading);
        assert!(!profile.supports_dynamic_model_unloading);
        assert!(!profile.supports_multi_model_packing);
        assert!(!profile.supports_multi_gpu_distribution);
        assert!(!profile.supports_kvcached_memory_sharing);
        assert!(!profile.supports_model_sleep_mode);
        assert!(!profile.supports_model_move_operations);
        assert!(!profile.supports_layer_offloading);
        assert!(!profile.supports_gpu_memory_telemetry);
        assert!(!profile.supports_cpu_only_runtime);
        assert!(!profile.supports_download_on_first_use);
        assert!(!profile.supports_ollama_style_cli);
        assert!(!profile.supports_openai_compatible_server);
        assert!(!profile.supports_progressive_kv_compression);
        assert!(!profile.supports_full_document_mode);
        assert!(!profile.supports_multimodal_serving);
        assert!(!profile.supports_omni_modality);
        assert!(!profile.supports_hardware_plugin_interface);
        assert!(!profile.supports_graph_shape_bucketing);
        assert!(!profile.supports_graph_validation_mode);
        assert!(!profile.supports_graph_conditional_nodes);
        assert!(!profile.supports_graph_nested_capture);
        assert_eq!(
            profile.storage_tiers_supported,
            vec![KvStorageTier::Hbm, KvStorageTier::CpuRam]
        );
        assert_eq!(profile.delegate_backend, None);
        assert_eq!(profile.placeholder, None);
        assert_eq!(profile.radix_cache_kind, None);
        assert_eq!(profile.storage_backend, None);
    }

    #[test]
    fn delegate_backend_helper_sets_backend() {
        let profile = EngineRuntimeProfile::default().with_delegate_backend("aiter");
        assert_eq!(profile.delegate_backend.as_deref(), Some("aiter"));
    }

    #[test]
    fn placeholder_helper_sets_placeholder() {
        let profile = EngineRuntimeProfile::default().with_placeholder("runtime_profile");
        assert_eq!(profile.placeholder.as_deref(), Some("runtime_profile"));
    }

    #[test]
    fn radix_cache_state_helper_sets_cache_state() {
        let profile = EngineRuntimeProfile::default().with_radix_cache_state(
            "radix", 1024, 256, 768, 16,
        );
        assert_eq!(profile.radix_cache_kind.as_deref(), Some("radix"));
        assert_eq!(profile.radix_total_tokens, Some(1024));
        assert_eq!(profile.radix_protected_tokens, Some(256));
        assert_eq!(profile.radix_evictable_tokens, Some(768));
        assert_eq!(profile.radix_page_size, Some(16));
    }

    #[test]
    fn fp8_kv_cache_state_helper_sets_fp8_kv_capabilities() {
        let profile = EngineRuntimeProfile::default().with_fp8_kv_cache_state(
            true, true, false, true, false,
        );
        assert!(profile.supports_fp8_kv_cache);
        assert!(profile.supports_fp8_kv_per_tensor_scales);
        assert!(!profile.supports_fp8_kv_per_head_scales);
        assert!(profile.supports_kv_scale_calibration);
        assert!(!profile.supports_quantized_attention_fusion);
    }

    #[test]
    fn storage_backend_state_helper_sets_storage_state() {
        let profile = EngineRuntimeProfile::default().with_storage_backend_state(
            "mooncake",
            true,
            true,
            "page_first_direct",
            Some("FILE".to_string()),
        );
        assert_eq!(profile.storage_backend.as_deref(), Some("mooncake"));
        assert_eq!(profile.storage_supports_stats, Some(true));
        assert_eq!(profile.storage_supports_zero_copy, Some(true));
        assert_eq!(
            profile.storage_layout_mode.as_deref(),
            Some("page_first_direct")
        );
        assert_eq!(profile.storage_transfer_mem_type.as_deref(), Some("FILE"));
    }

    #[test]
    fn model_routing_state_helper_sets_routing_capabilities() {
        let profile = EngineRuntimeProfile::default().with_model_routing_state(
            true, true, false, true, false, true, false,
        );
        assert!(profile.supports_model_hot_swap);
        assert!(profile.supports_model_aliases);
        assert!(!profile.supports_model_groups);
        assert!(profile.supports_ttl_unload);
        assert!(!profile.supports_request_filters);
        assert!(profile.supports_config_reload);
        assert!(!profile.supports_direct_model_passthrough);
    }

    #[test]
    fn model_packing_state_helper_sets_packing_capabilities() {
        let profile = EngineRuntimeProfile::default().with_model_packing_state(
            true, true, true, true, true, true, false, true,
        );
        assert!(profile.supports_dynamic_model_loading);
        assert!(profile.supports_dynamic_model_unloading);
        assert!(profile.supports_multi_model_packing);
        assert!(profile.supports_multi_gpu_distribution);
        assert!(profile.supports_kvcached_memory_sharing);
        assert!(profile.supports_model_sleep_mode);
        assert!(!profile.supports_model_move_operations);
        assert!(profile.supports_gpu_memory_telemetry);
    }

    #[test]
    fn layer_offloading_state_helper_sets_layer_offloading() {
        let profile = EngineRuntimeProfile::default().with_layer_offloading_state(true);
        assert!(profile.supports_layer_offloading);
    }

    #[test]
    fn compact_runtime_state_helper_sets_compact_runtime_capabilities() {
        let profile = EngineRuntimeProfile::default().with_compact_runtime_state(
            true, true, true, true, true, false,
        );
        assert!(profile.supports_cpu_only_runtime);
        assert!(profile.supports_download_on_first_use);
        assert!(profile.supports_ollama_style_cli);
        assert!(profile.supports_openai_compatible_server);
        assert!(profile.supports_progressive_kv_compression);
        assert!(!profile.supports_full_document_mode);
    }

    #[test]
    fn storage_orchestration_state_helper_sets_storage_orchestration_capabilities() {
        let profile = EngineRuntimeProfile::default().with_storage_orchestration_state(
            true, true, true, true, true, false, true, true, true, false,
        );
        assert!(profile.supports_distributed_memory_pooling);
        assert!(profile.supports_dynamic_multilevel_caching);
        assert!(profile.supports_global_metadata_management);
        assert!(profile.supports_capacity_management);
        assert!(profile.supports_prefix_matching);
        assert!(!profile.supports_sliding_window_matching);
        assert!(profile.supports_kv_matching);
        assert!(profile.supports_two_phase_write);
        assert!(profile.supports_async_eviction);
        assert!(!profile.supports_trace_replay_optimization);
    }

    #[test]
    fn quantization_pipeline_state_helper_sets_quantization_pipeline_capabilities() {
        let profile = EngineRuntimeProfile::default().with_quantization_pipeline_state(
            true, true, true, true, true, false, true, true,
        );
        assert!(profile.supports_model_free_ptq);
        assert!(profile.supports_compressed_tensors_format);
        assert!(profile.supports_weight_quantization_pipeline);
        assert!(profile.supports_activation_quantization_pipeline);
        assert!(profile.supports_kv_cache_quantization_pipeline);
        assert!(!profile.supports_attention_quantization_pipeline);
        assert!(profile.supports_disk_offloading_quantization);
        assert!(profile.supports_distributed_calibration);
    }

    #[test]
    fn serving_benchmark_state_helper_sets_serving_benchmark_capabilities() {
        let profile = EngineRuntimeProfile::default().with_serving_benchmark_state(
            true, true, true, true, true, true, true, true, true, true, true, false, true,
        );
        assert!(profile.supports_rust_native_gpu_serving);
        assert!(profile.supports_pure_jax_tpu_serving);
        assert!(profile.supports_cross_device_benchmarking);
        assert!(profile.supports_single_graph_capture);
        assert!(profile.supports_graph_shape_bucketing);
        assert!(profile.supports_graph_validation_mode);
        assert!(profile.supports_graph_conditional_nodes);
        assert!(profile.supports_graph_nested_capture);
        assert!(profile.supports_fp8_channelscale_epilogue);
        assert!(profile.supports_cached_ttft_reporting);
        assert!(profile.supports_peak_throughput_reporting);
        assert!(!profile.supports_perplexity_reporting);
        assert!(profile.supports_streaming_api_server);
    }

    #[test]
    fn gfxgraph_bridge_state_helper_sets_graph_bridge_capabilities() {
        let profile = EngineRuntimeProfile::default().with_gfxgraph_bridge_state(
            true, true, true, false,
        );
        assert!(profile.supports_graph_shape_bucketing);
        assert!(profile.supports_graph_validation_mode);
        assert!(profile.supports_graph_conditional_nodes);
        assert!(!profile.supports_graph_nested_capture);
    }

    #[test]
    fn vllm_runtime_family_state_helper_sets_multimodal_and_plugin_capabilities() {
        let profile = EngineRuntimeProfile::default().with_vllm_runtime_family_state(
            true, true, false,
        );
        assert!(profile.supports_multimodal_serving);
        assert!(profile.supports_omni_modality);
        assert!(!profile.supports_hardware_plugin_interface);
    }

    #[test]
    fn kv_quantization_mode_helpers_cover_turbo_and_rotor() {
        let profile = EngineRuntimeProfile {
            supports_atom_kv_quant: true,
            supports_turboquant_kv: true,
            supports_rotorquant_kv: true,
            supports_kv_cache_quantization_pipeline: true,
            ..EngineRuntimeProfile::default()
        };
        assert!(profile.supports_kv_quantization_mode(&KvQuantizationMode::TurboQuant));
        assert!(profile.supports_kv_quantization_mode(&KvQuantizationMode::RotorQuant));
        assert!(profile.supported_kv_quantization_modes().contains(&KvQuantizationMode::Int8));
        assert!(profile.supported_kv_quantization_modes().contains(&KvQuantizationMode::Int4));
    }

    #[test]
    fn warmup_initialization_helper_sets_warmup_state() {
        let profile = EngineRuntimeProfile::default()
            .with_warmup_initialization_state(true, true);
        assert_eq!(profile.warmup_kv_connector_initialized, Some(true));
        assert_eq!(profile.warmup_prefill_batch_executed, Some(true));
    }

    #[test]
    fn kv_object_contract_disallows_raw_tensor_writes_to_metadata_stores() {
        let record = KvObjectRecord {
            kv_object_id: "obj-1".into(),
            cache_key: "k".into(),
            content_hash: "h".into(),
            prefix_hash: "p".into(),
            model_id: "m".into(),
            engine_id: "e".into(),
            tokenizer_id: "tok".into(),
            tokenizer_revision: "rev".into(),
            token_start: 0,
            token_count: 1,
            layer_start: 0,
            layer_count: 1,
            dtype: "fp16".into(),
            compression_mode: KvCompressionMode::None,
            quantization_mode: KvQuantizationMode::None,
            storage_tier: KvStorageTier::Hbm,
            location_uri: "engine://kv/obj-1".into(),
            reusable_scope: "persona".into(),
            pin_state: "pinned".into(),
            eviction_priority: 1.0,
            future_use_score: 1.0,
            metadata_registry: "surrealdb".into(),
            control_plane_store: "valkey".into(),
            tensor_store: "engine_native".into(),
            allow_raw_kv_in_surrealdb: false,
            allow_raw_kv_in_valkey: false,
            created_at_rfc3339: "2026-01-01T00:00:00Z".into(),
            last_used_at_rfc3339: "2026-01-01T00:00:00Z".into(),
        };
        assert!(!record.allow_raw_kv_in_surrealdb);
        assert!(!record.allow_raw_kv_in_valkey);
        assert_eq!(record.metadata_registry, "surrealdb");
        assert_eq!(record.control_plane_store, "valkey");
    }
}
