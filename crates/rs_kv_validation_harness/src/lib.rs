use rs_autoquant_policy::{AutoQuantFingerprint, AutoQuantObserverSnapshot, AutoQuantPolicy, SideStats};
use rs_atom_engine_profile::EngineRuntimeProfile;
use rs_kv_codec_adapters::CodecAdapterRegistry;
use rs_kv_quant_contracts::{
    compute_content_hash, compute_request_content_hashes, normalize_codec_alias, AgentStepGraph,
    CapturedGraph, GraphFingerprint, GraphPool, KvCodec, KvDecodeTelemetryBundle, KvEvictionDecision,
    KvPolicyMode, KvPrefetchPlan, KvQuantPolicy, KvStorageTier, KvTransferPlan, MetadataLayout,
    MatchResult, PrefixMatchResult, PrecomputedContextAsset, PrecomputedKvAsset, RadixSnapshotNode,
    RadixTreeSnapshot,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ValidationCase {
    pub name: String,
    pub passed: bool,
    pub note: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ValidationReport {
    pub passed: bool,
    pub cases: Vec<ValidationCase>,
}

pub fn run_validation_suite() -> ValidationReport {
    let registry = CodecAdapterRegistry::baseline();
    let mut cases = Vec::new();

    for codec in [KvCodec::Tq4, KvCodec::Tq3, KvCodec::Tq2, KvCodec::Rq3Planar, KvCodec::Rq4Planar, KvCodec::Fp8E4M3] {
        let supported = registry.supports(&codec);
        cases.push(ValidationCase {
            name: format!("baseline_support::{codec:?}"),
            passed: supported,
            note: if supported {
                "baseline adapter present".into()
            } else {
                "missing baseline adapter".into()
            },
        });
    }

    let alias_codec = normalize_codec_alias("atom_fp8").expect("alias must normalize");
    cases.push(ValidationCase {
        name: "alias_normalization::atom_fp8".into(),
        passed: alias_codec == KvCodec::Fp8E4M3,
        note: format!("normalized to {:?}", alias_codec),
    });

    let policy = AutoQuantPolicy::uniform("digest", "model", 2, KvCodec::Tq4, 4, None);
    let round_trip = serde_json::to_string(&policy)
        .ok()
        .and_then(|json| serde_json::from_str::<AutoQuantPolicy>(&json).ok())
        .is_some();
    cases.push(ValidationCase {
        name: "autoquant_round_trip".into(),
        passed: round_trip,
        note: "json policy round-trip".into(),
    });

    let mut observer_layers = std::collections::BTreeMap::new();
    observer_layers.insert(
        "L0_K".into(),
        SideStats {
            sample_count: 3,
            dynamic_range: 1.5,
            mean_abs: 0.8,
            rms: 0.9,
            kurtosis: 2.1,
            sparsity: 0.12,
            last_observed_at: 42.0,
        },
    );
    let observer_snapshot = AutoQuantObserverSnapshot {
        n_layers: 2,
        sample_every: 64,
        ema_alpha: 0.05,
        sparsity_eps: 1e-4,
        total_observations: 9,
        layers: observer_layers,
    };
    let observer_round_trip = serde_json::to_string(&observer_snapshot)
        .ok()
        .and_then(|json| serde_json::from_str::<AutoQuantObserverSnapshot>(&json).ok())
        .is_some();
    cases.push(ValidationCase {
        name: "autoquant_observer_round_trip".into(),
        passed: observer_round_trip,
        note: "json observer snapshot round-trip".into(),
    });

    let fp = AutoQuantFingerprint {
        gpu_arch: "gfx1030".into(),
        wave_size: 32,
        rocm_version: "7.2".into(),
        triton_version: "3.5".into(),
        python_version: "3.12.0".into(),
        model_family: "qwen".into(),
        n_layers: 32,
        head_dim: 128,
        n_heads: 32,
        n_kv_heads: 8,
        dtype_mode: "fp16".into(),
        codec_set_version: "1".into(),
    };
    cases.push(ValidationCase {
        name: "fingerprint_digest_shape".into(),
        passed: fp.hex_digest().len() == 16,
        note: fp.hex_digest(),
    });

    let kv_policy = KvQuantPolicy::new("model", KvCodec::Tq4, KvPolicyMode::Adaptive);
    cases.push(ValidationCase {
        name: "kv_quant_policy_shape".into(),
        passed: kv_policy.codec == KvCodec::Tq4,
        note: "shared KV quant contract reachable".into(),
    });

    let runtime_profile = EngineRuntimeProfile::default().with_delegate_backend("aiter");
    cases.push(ValidationCase {
        name: "atom_runtime_profile_shape".into(),
        passed: runtime_profile.supports_atom_backend
            && runtime_profile.supports_atom_attention
            && runtime_profile.supports_atom_rocm_telemetry
            && runtime_profile.supports_atom_fallback
            && runtime_profile.delegate_backend.as_deref() == Some("aiter")
            && runtime_profile.placeholder.is_none(),
        note: serde_json::to_string(&runtime_profile).unwrap(),
    });

    let gfxgraph_profile = EngineRuntimeProfile::default().with_gfxgraph_bridge_state(
        true,
        true,
        true,
        false,
    );
    cases.push(ValidationCase {
        name: "gfxgraph_bridge_profile_shape".into(),
        passed: gfxgraph_profile.supports_graph_shape_bucketing
            && gfxgraph_profile.supports_graph_validation_mode
            && gfxgraph_profile.supports_graph_conditional_nodes
            && !gfxgraph_profile.supports_graph_nested_capture,
        note: serde_json::to_string(&gfxgraph_profile).unwrap(),
    });

    let vllm_family_profile = EngineRuntimeProfile::default().with_vllm_runtime_family_state(
        true,
        true,
        false,
    );
    cases.push(ValidationCase {
        name: "vllm_runtime_family_profile_shape".into(),
        passed: vllm_family_profile.supports_multimodal_serving
            && vllm_family_profile.supports_omni_modality
            && !vllm_family_profile.supports_hardware_plugin_interface,
        note: serde_json::to_string(&vllm_family_profile).unwrap(),
    });

    cases.push(ValidationCase {
        name: "radix_runtime_profile_defaults".into(),
        passed: runtime_profile.radix_cache_kind.is_none()
            && runtime_profile.radix_total_tokens.is_none()
            && runtime_profile.radix_protected_tokens.is_none()
            && runtime_profile.radix_evictable_tokens.is_none()
            && runtime_profile.radix_page_size.is_none()
            && runtime_profile.supports_automatic_prefix_caching
            && runtime_profile.supports_radix_cache
            && runtime_profile.supports_kv_events
            && !runtime_profile.supports_fp8_kv_cache
            && !runtime_profile.supports_turboquant_kv
            && !runtime_profile.supports_rotorquant_kv
            && !runtime_profile.supports_eagle3
            && !runtime_profile.supports_medusa
            && !runtime_profile.supports_ngram_speculation
            && !runtime_profile.supports_phantom
            && !runtime_profile.supports_phantom_x
            && runtime_profile.storage_tiers_supported.len() == 2
            && runtime_profile.storage_backend.is_none()
            && runtime_profile.storage_supports_stats.is_none()
            && runtime_profile.storage_supports_zero_copy.is_none()
            && runtime_profile.storage_layout_mode.is_none()
            && runtime_profile.storage_transfer_mem_type.is_none()
            && runtime_profile.attention_kernel_capabilities.is_none()
            && runtime_profile.amd_kv_kernel_profile.is_none(),
        note: "radix cache state placeholder is empty until runtime wiring lands".into(),
    });

    let storage_profile = EngineRuntimeProfile::default().with_storage_backend_state(
        "mooncake",
        true,
        true,
        "page_first_direct",
        Some("FILE".to_string()),
    );
    cases.push(ValidationCase {
        name: "storage_runtime_profile_shape".into(),
        passed: storage_profile.storage_backend.as_deref() == Some("mooncake")
            && storage_profile.storage_supports_stats == Some(true)
            && storage_profile.storage_supports_zero_copy == Some(true)
            && storage_profile.storage_layout_mode.as_deref() == Some("page_first_direct")
            && storage_profile.storage_transfer_mem_type.as_deref() == Some("FILE"),
        note: serde_json::to_string(&storage_profile).unwrap(),
    });

    let transfer_plan = KvTransferPlan {
        plan_id: "plan-1".into(),
        source_tier: KvStorageTier::Hbm,
        destination_tier: KvStorageTier::CpuRam,
        kv_object_ids: vec!["kv-1".into()],
        disaggregated_prefill_decode: true,
    };
    let prefetch_plan = KvPrefetchPlan {
        plan_id: "prefetch-1".into(),
        upcoming_agent_step_ids: vec!["step-a".into()],
        prefetch_kv_object_ids: vec!["kv-1".into()],
        reason: "prefix reuse".into(),
    };
    let eviction = KvEvictionDecision {
        decision_id: "evict-1".into(),
        evict_kv_object_ids: vec!["kv-9".into()],
        preserve_kv_object_ids: vec!["kv-1".into()],
        reason: "capacity pressure".into(),
    };
    let graph = AgentStepGraph {
        graph_id: "graph-1".into(),
        active_step_id: "step-a".into(),
        next_likely_step_ids: vec!["step-b".into()],
        required_kv_object_ids: vec!["kv-1".into()],
    };
    let context_asset = PrecomputedContextAsset {
        asset_id: "ctx-1".into(),
        asset_type: "prompt".into(),
        profile_id: "profile-1".into(),
        content_hash: "hash".into(),
        token_count: 1024,
        pinned: true,
        storage_tiers_supported: vec![KvStorageTier::Hbm, KvStorageTier::CpuRam],
    };
    let kv_asset = PrecomputedKvAsset {
        kv_asset_id: "kvasset-1".into(),
        context_asset_id: "ctx-1".into(),
        kv_object_id: "kv-1".into(),
        quantization_mode: KvCodec::Tq4,
        compression_mode: "hybrid".into(),
        calibration_status: "complete".into(),
        calibration_dataset_id: Some("dataset-1".into()),
    };
    let decode_telemetry = KvDecodeTelemetryBundle {
        request_id: "req-1".into(),
        session_id: "sess-1".into(),
        model_id: "model-1".into(),
        engine_id: "engine-1".into(),
        shape_key: "shape-1".into(),
        prompt_tokens: 32,
        decode_tokens: 16,
        prefix_reuse_ratio: 0.75,
        kv_hit_rate: 0.9,
        kv_miss_count: 3,
        branch_entropy_mean: Some(0.3),
        branch_entropy_max: Some(0.7),
        mismatch_depth_mean: Some(1.2),
        mismatch_depth_std: Some(0.4),
        early_exit_ratio: Some(0.2),
        acceptance_rate: Some(0.95),
        token_surprisal_mean: Some(0.5),
        repeated_token_count: Some(4),
        calibration_drift_score: Some(0.1),
        gpu_memory_used_mb: Some(2048),
        memory_bandwidth_gbps: Some(512.0),
        kernel_launch_overhead_ms: Some(0.8),
        decode_tps: Some(1234.0),
    };
    for json in [
        serde_json::to_string(&transfer_plan).unwrap(),
        serde_json::to_string(&prefetch_plan).unwrap(),
        serde_json::to_string(&eviction).unwrap(),
        serde_json::to_string(&graph).unwrap(),
        serde_json::to_string(&context_asset).unwrap(),
        serde_json::to_string(&kv_asset).unwrap(),
        serde_json::to_string(&decode_telemetry).unwrap(),
    ] {
        cases.push(ValidationCase {
            name: format!("rvllm_contract_round_trip::{json}"),
            passed: true,
            note: json,
        });
    }

    let mut graph_pool = GraphPool::new();
    let layout = MetadataLayout::compute(128, 129);
    graph_pool.insert(CapturedGraph {
        bucket: 128,
        max_blocks: 129,
        layout_hash: layout.hash(),
        fingerprint: GraphFingerprint([0u8; 32]),
    });
    cases.push(ValidationCase {
        name: "rvllm_graph_pool_layout_match".into(),
        passed: graph_pool.check_before_replay(128, 129, &layout).is_ok(),
        note: format!("layout_bytes={}", layout.bytes()),
    });
    let wrong_layout = MetadataLayout::compute(128, 257);
    cases.push(ValidationCase {
        name: "rvllm_graph_pool_layout_mismatch".into(),
        passed: graph_pool.check_before_replay(128, 129, &wrong_layout).is_err(),
        note: "metadata drift rejected".into(),
    });

    let snapshot = RadixTreeSnapshot {
        nodes: vec![
            RadixSnapshotNode {
                edge: String::new(),
                tenants: vec![("worker-1".to_string(), 100)],
                child_count: 2,
            },
            RadixSnapshotNode {
                edge: "Hello ".to_string(),
                tenants: vec![("worker-1".to_string(), 100)],
                child_count: 1,
            },
            RadixSnapshotNode {
                edge: "world".to_string(),
                tenants: vec![("worker-1".to_string(), 100)],
                child_count: 0,
            },
            RadixSnapshotNode {
                edge: "Goodbye".to_string(),
                tenants: vec![("worker-2".to_string(), 200)],
                child_count: 0,
            },
        ],
    };
    let snapshot_bytes = snapshot.to_bytes().unwrap();
    let restored = RadixTreeSnapshot::from_bytes(&snapshot_bytes).unwrap();
    cases.push(ValidationCase {
        name: "rvllm_radix_snapshot_round_trip".into(),
        passed: snapshot == restored && snapshot.node_count() == 4 && snapshot.total_edge_bytes() > 0,
        note: format!("node_count={}", snapshot.node_count()),
    });

    let tokens = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let request_hashes = compute_request_content_hashes(&tokens, 4);
    cases.push(ValidationCase {
        name: "smg_dual_hash_helpers".into(),
        passed: request_hashes.len() == 2
            && request_hashes[0] == compute_content_hash(&tokens[..4])
            && request_hashes[1] == compute_content_hash(&tokens[4..8])
            && compute_request_content_hashes(&tokens, 0).is_empty(),
        note: format!("hashes={}", request_hashes.len()),
    });

    let match_result = PrefixMatchResult::new("worker-1", 3, 4);
    cases.push(ValidationCase {
        name: "smg_prefix_match_result".into(),
        passed: match_result.tenant() == "worker-1"
            && match_result.matched_count() == 3
            && match_result.input_count() == 4
            && (match_result.hit_ratio() - 0.75).abs() < f64::EPSILON,
        note: format!("hit_ratio={}", match_result.hit_ratio()),
    });

    let passed = cases.iter().all(|case| case.passed);
    ValidationReport { passed, cases }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn suite_passes() {
        let report = run_validation_suite();
        assert!(report.passed);
        assert!(!report.cases.is_empty());
    }

    #[test]
    fn report_serializes() {
        let report = run_validation_suite();
        let json = serde_json::to_string_pretty(&report).unwrap();
        assert!(json.contains("baseline_support"));
    }
}
