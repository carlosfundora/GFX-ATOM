use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use thiserror::Error;

pub const XXH3_SEED: u64 = 1337;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ContentHash(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SequenceHash(pub u64);

pub fn compute_content_hash(token_ids: &[u32]) -> ContentHash {
    let mut bytes = Vec::with_capacity(token_ids.len() * core::mem::size_of::<u32>());
    for &token in token_ids {
        bytes.extend_from_slice(&token.to_le_bytes());
    }
    ContentHash(xxhash_rust::xxh3::xxh3_64_with_seed(&bytes, XXH3_SEED))
}

pub fn compute_request_content_hashes(tokens: &[u32], block_size: usize) -> Vec<ContentHash> {
    if block_size == 0 {
        return Vec::new();
    }
    tokens
        .chunks(block_size)
        .filter(|chunk| chunk.len() == block_size)
        .map(compute_content_hash)
        .collect()
}

pub type TenantId = String;

pub trait MatchResult {
    fn tenant(&self) -> &TenantId;
    fn matched_count(&self) -> usize;
    fn input_count(&self) -> usize;

    fn hit_ratio(&self) -> f64 {
        let input = self.input_count();
        if input == 0 {
            0.0
        } else {
            self.matched_count() as f64 / input as f64
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrefixMatchResult {
    pub tenant: TenantId,
    pub matched_count: usize,
    pub input_count: usize,
}

impl PrefixMatchResult {
    pub fn new(tenant: impl Into<String>, matched_count: usize, input_count: usize) -> Self {
        Self {
            tenant: tenant.into(),
            matched_count,
            input_count,
        }
    }
}

impl MatchResult for PrefixMatchResult {
    fn tenant(&self) -> &TenantId {
        &self.tenant
    }

    fn matched_count(&self) -> usize {
        self.matched_count
    }

    fn input_count(&self) -> usize {
        self.input_count
    }
}

/// Composite key for a positional KV block (position + content hash).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PositionalIndexKey {
    pub position: usize,
    pub content_hash: ContentHash,
}

/// Full positional index record: key + sequence hash + owning worker.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PositionalIndexEntry {
    pub key: PositionalIndexKey,
    pub sequence_hash: SequenceHash,
    pub worker_id: u32,
}

/// Typed errors for positional index operations.
#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum PositionalIndexError {
    #[error("worker {0} is not tracked in the positional index")]
    WorkerNotTracked(u32),
    #[error("parent block at position {0} not found for worker {1}")]
    ParentBlockNotFound(usize, u32),
}

pub type PositionalIndexResult<T> = Result<T, PositionalIndexError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvCodec {
    Auto,
    Bf16,
    Fp8E4M3,
    Fp8E5M2,
    Int8,
    Tq1,
    Tq4,
    Tq3,
    Tq2,
    Tq8,
    Rq3Planar,
    Rq4Planar,
    Rq3Iso,
    Rq4Iso,
}

impl KvCodec {
    pub fn is_turboquant(&self) -> bool {
        matches!(
            self,
            KvCodec::Tq1 | KvCodec::Tq2 | KvCodec::Tq3 | KvCodec::Tq4 | KvCodec::Tq8
        )
    }

    pub fn is_rotorquant(&self) -> bool {
        matches!(
            self,
            KvCodec::Rq3Planar | KvCodec::Rq4Planar | KvCodec::Rq3Iso | KvCodec::Rq4Iso
        )
    }

    pub fn is_rotorquant_planar(&self) -> bool {
        matches!(self, KvCodec::Rq3Planar | KvCodec::Rq4Planar)
    }

    pub fn is_rotorquant_iso(&self) -> bool {
        matches!(self, KvCodec::Rq3Iso | KvCodec::Rq4Iso)
    }

    pub fn bit_width(&self) -> Option<u8> {
        match self {
            KvCodec::Tq1 => Some(1),
            KvCodec::Tq2 => Some(2),
            KvCodec::Tq3 => Some(3),
            KvCodec::Tq4 => Some(4),
            KvCodec::Tq8 => Some(8),
            KvCodec::Rq3Planar | KvCodec::Rq3Iso => Some(3),
            KvCodec::Rq4Planar | KvCodec::Rq4Iso => Some(4),
            KvCodec::Fp8E4M3 | KvCodec::Fp8E5M2 | KvCodec::Int8 => Some(8),
            KvCodec::Auto | KvCodec::Bf16 => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvPolicyMode {
    Static,
    Adaptive,
    Learned,
    Fallback,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvStorageTier {
    Hbm,
    CpuRam,
    Lmcache,
    Nvme,
    ObjectStore,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UniversalKvStage {
    HotRotor,
    WarmRotorPolar,
    ColdTurboResidual,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UniversalKvBlockHeaderV1 {
    pub block_size: u16,
    pub bit_width: u8,
    pub rotor_id: u8,
    pub codec: KvCodec,
    pub stage: UniversalKvStage,
    pub scale: f32,
    pub flags: u8,
    pub origin_model_tag: u16,
}

impl UniversalKvBlockHeaderV1 {
    pub const FLAG_PINNED: u8 = 0b0000_0001;
    pub const FLAG_TURBO_RESIDUAL: u8 = 0b0000_0010;

    pub fn has_flag(&self, flag: u8) -> bool {
        self.flags & flag != 0
    }

    pub fn should_apply_turbo_residual(&self) -> bool {
        self.stage == UniversalKvStage::ColdTurboResidual
            || self.has_flag(Self::FLAG_TURBO_RESIDUAL)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct UniversalKvPlacementPolicy {
    pub hot_importance_threshold: f32,
    pub warm_importance_threshold: f32,
    pub warm_age_steps: u32,
    pub cold_age_steps: u32,
    pub gpu_high_watermark: f32,
}

impl Default for UniversalKvPlacementPolicy {
    fn default() -> Self {
        Self {
            hot_importance_threshold: 0.80,
            warm_importance_threshold: 0.35,
            warm_age_steps: 128,
            cold_age_steps: 512,
            gpu_high_watermark: 0.90,
        }
    }
}

impl UniversalKvPlacementPolicy {
    pub fn select_stage(
        &self,
        importance: f32,
        age_steps: u32,
        gpu_utilization_pct: f32,
    ) -> UniversalKvStage {
        let importance = importance.clamp(0.0, 1.0);
        let gpu_utilization_pct = gpu_utilization_pct.clamp(0.0, 1.0);

        if importance >= self.hot_importance_threshold
            && age_steps <= self.warm_age_steps
            && gpu_utilization_pct <= self.gpu_high_watermark
        {
            return UniversalKvStage::HotRotor;
        }

        if importance >= self.warm_importance_threshold && age_steps <= self.cold_age_steps {
            return UniversalKvStage::WarmRotorPolar;
        }

        UniversalKvStage::ColdTurboResidual
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvQuantPolicy {
    pub model_id: String,
    pub codec: KvCodec,
    pub mode: KvPolicyMode,
    pub layer_id: Option<u32>,
    pub stage_id: Option<String>,
    pub note: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvTransferPlan {
    pub plan_id: String,
    pub source_tier: KvStorageTier,
    pub destination_tier: KvStorageTier,
    pub kv_object_ids: Vec<String>,
    pub disaggregated_prefill_decode: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvPrefetchPlan {
    pub plan_id: String,
    pub upcoming_agent_step_ids: Vec<String>,
    pub prefetch_kv_object_ids: Vec<String>,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvEvictionDecision {
    pub decision_id: String,
    pub evict_kv_object_ids: Vec<String>,
    pub preserve_kv_object_ids: Vec<String>,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentStepGraph {
    pub graph_id: String,
    pub active_step_id: String,
    pub next_likely_step_ids: Vec<String>,
    pub required_kv_object_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrecomputedContextAsset {
    pub asset_id: String,
    pub asset_type: String,
    pub profile_id: String,
    pub content_hash: String,
    pub token_count: u32,
    pub pinned: bool,
    pub storage_tiers_supported: Vec<KvStorageTier>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrecomputedKvAsset {
    pub kv_asset_id: String,
    pub context_asset_id: String,
    pub kv_object_id: String,
    pub quantization_mode: KvCodec,
    pub compression_mode: String,
    pub calibration_status: String,
    pub calibration_dataset_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct KvQuantTelemetry {
    pub prefix_reuse_ratio: Option<u64>,
    pub kv_hit_rate: Option<u64>,
    pub kv_used_bytes: Option<u64>,
    pub kv_capacity_bytes: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MetaLayoutHash(pub [u8; 32]);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MetadataLayout {
    pub bucket: u32,
    pub max_blocks: u32,
    pub token_ids_off: u32,
    pub positions_off: u32,
    pub context_lens_off: u32,
    pub block_tables_off: u32,
    pub slot_mapping_off: u32,
    pub seq_start_pos_off: u32,
    pub total_elements: u32,
}

impl MetadataLayout {
    pub fn compute(bucket: u32, max_blocks: u32) -> Self {
        let token_ids_off = 0u32;
        let positions_off = token_ids_off + bucket;
        let context_lens_off = positions_off + bucket;
        let block_tables_off = context_lens_off + bucket;
        let slot_mapping_off = block_tables_off + bucket * max_blocks;
        let seq_start_pos_off = slot_mapping_off + bucket;
        let total_elements = seq_start_pos_off + bucket + 1;
        Self {
            bucket,
            max_blocks,
            token_ids_off,
            positions_off,
            context_lens_off,
            block_tables_off,
            slot_mapping_off,
            seq_start_pos_off,
            total_elements,
        }
    }

    pub fn hash(&self) -> MetaLayoutHash {
        use sha2::{Digest, Sha256};

        let mut h = Sha256::new();
        h.update(self.bucket.to_le_bytes());
        h.update(self.max_blocks.to_le_bytes());
        h.update(self.token_ids_off.to_le_bytes());
        h.update(self.positions_off.to_le_bytes());
        h.update(self.context_lens_off.to_le_bytes());
        h.update(self.block_tables_off.to_le_bytes());
        h.update(self.slot_mapping_off.to_le_bytes());
        h.update(self.seq_start_pos_off.to_le_bytes());
        h.update(self.total_elements.to_le_bytes());
        let digest = h.finalize();
        let mut out = [0u8; 32];
        out.copy_from_slice(&digest);
        MetaLayoutHash(out)
    }

    pub fn bytes(&self) -> usize {
        (self.total_elements as usize) * core::mem::size_of::<i32>()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GraphFingerprint(pub [u8; 32]);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapturedGraph {
    pub bucket: u32,
    pub max_blocks: u32,
    pub layout_hash: MetaLayoutHash,
    pub fingerprint: GraphFingerprint,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum GraphError {
    #[error("graph bucket missing: padded_batch={padded_batch}")]
    BucketMissing { padded_batch: u32 },
    #[error("capture metadata mismatch: captured={captured:?} replay={replay:?}")]
    CaptureMetadataMismatch {
        captured: MetaLayoutHash,
        replay: MetaLayoutHash,
    },
}

pub type GraphResult<T> = Result<T, GraphError>;

#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GraphPool {
    graphs: BTreeMap<(u32, u32), CapturedGraph>,
}

impl GraphPool {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, graph: CapturedGraph) {
        self.graphs.insert((graph.bucket, graph.max_blocks), graph);
    }

    pub fn get(&self, bucket: u32, max_blocks: u32) -> Option<&CapturedGraph> {
        self.graphs.get(&(bucket, max_blocks))
    }

    pub fn len(&self) -> usize {
        self.graphs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.graphs.is_empty()
    }

    pub fn check_before_replay(
        &self,
        bucket: u32,
        max_blocks: u32,
        current: &MetadataLayout,
    ) -> GraphResult<&CapturedGraph> {
        let graph = self.get(bucket, max_blocks).ok_or(GraphError::BucketMissing {
            padded_batch: bucket,
        })?;
        let replay = current.hash();
        if graph.layout_hash != replay {
            return Err(GraphError::CaptureMetadataMismatch {
                captured: graph.layout_hash,
                replay,
            });
        }
        Ok(graph)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RadixTreeSnapshot {
    pub nodes: Vec<RadixSnapshotNode>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RadixSnapshotNode {
    pub edge: String,
    pub tenants: Vec<(String, u64)>,
    pub child_count: u32,
}

impl RadixTreeSnapshot {
    pub fn empty() -> Self {
        Self { nodes: Vec::new() }
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, Box<bincode::ErrorKind>> {
        bincode::serialize(self)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Box<bincode::ErrorKind>> {
        bincode::deserialize(bytes)
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn total_edge_bytes(&self) -> usize {
        self.nodes.iter().map(|node| node.edge.len()).sum()
    }
}

#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum KvCodecError {
    #[error("unsupported kv codec alias: {0}")]
    UnsupportedAlias(String),
    #[error("FP8 KV cache dimension {0} is not aligned to 16-byte boundary (required for {1})")]
    Fp8DimensionMisaligned(usize, String),
    #[error("KV codec constraint validation failed: {0}")]
    ConstraintViolation(String),
}

pub fn normalize_codec_alias(alias: &str) -> Result<KvCodec, KvCodecError> {
    match alias.to_ascii_lowercase().as_str() {
        "auto" => Ok(KvCodec::Auto),
        "bf16" | "bfloat16" => Ok(KvCodec::Bf16),
        "fp8_e4m3" | "atom_fp8" => Ok(KvCodec::Fp8E4M3),
        "fp8_e5m2" => Ok(KvCodec::Fp8E5M2),
        "int8" => Ok(KvCodec::Int8),
        "tq1" => Ok(KvCodec::Tq1),
        "tq4" => Ok(KvCodec::Tq4),
        "tq3" => Ok(KvCodec::Tq3),
        "tq2" => Ok(KvCodec::Tq2),
        "tq8" => Ok(KvCodec::Tq8),
        "rq3" | "rq3_planar" => Ok(KvCodec::Rq3Planar),
        "rq4" | "rq4_planar" => Ok(KvCodec::Rq4Planar),
        "rq3_iso" => Ok(KvCodec::Rq3Iso),
        "rq4_iso" => Ok(KvCodec::Rq4Iso),
        other => Err(KvCodecError::UnsupportedAlias(other.to_string())),
    }
}

/// Validates that the given head dimension is aligned to 16 bytes for FP8 KV cache.
/// FP8 KV caches require 16-byte aligned dimensions for efficient vectorized access.
/// Formula: `aligned_dim = ((head_dim + 15) // 16) * 16`
pub fn validate_fp8_kv_dimension(head_dim: usize, model_name: &str) -> Result<usize, KvCodecError> {
    let aligned = ((head_dim + 15) / 16) * 16;
    if head_dim % 16 != 0 {
        return Err(KvCodecError::Fp8DimensionMisaligned(
            head_dim,
            format!("{} (aligned to {})", model_name, aligned),
        ));
    }
    Ok(head_dim)
}

/// Suggests the proper aligned dimension for FP8 KV cache.
/// This is useful for model implementations that compute `head_dim + offset` (e.g., DeepSeek v2).
pub fn align_dimension_to_16(dimension: usize) -> usize {
    ((dimension + 15) / 16) * 16
}

impl KvQuantPolicy {
    pub fn new(model_id: impl Into<String>, codec: KvCodec, mode: KvPolicyMode) -> Self {
        Self {
            model_id: model_id.into(),
            codec,
            mode,
            layer_id: None,
            stage_id: None,
            note: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aliases_normalize() {
        assert_eq!(normalize_codec_alias("atom_fp8").unwrap(), KvCodec::Fp8E4M3);
        assert_eq!(normalize_codec_alias("rq3").unwrap(), KvCodec::Rq3Planar);
    }

    #[test]
    fn policy_serializes() {
        let p = KvQuantPolicy::new("m", KvCodec::Tq4, KvPolicyMode::Adaptive);
        let s = serde_json::to_string(&p).unwrap();
        assert!(s.contains("tq4"));
    }

    #[test]
    fn codec_family_helpers_cover_turbo_and_rotor() {
        assert!(KvCodec::Tq2.is_turboquant());
        assert!(KvCodec::Rq3Planar.is_rotorquant_planar());
        assert!(KvCodec::Rq4Iso.is_rotorquant_iso());
        assert_eq!(KvCodec::Rq4Iso.bit_width(), Some(4));
    }

    #[test]
    fn rvllm_kv_contracts_round_trip() {
        let transfer = KvTransferPlan {
            plan_id: "plan-1".into(),
            source_tier: KvStorageTier::Hbm,
            destination_tier: KvStorageTier::CpuRam,
            kv_object_ids: vec!["kv-1".into(), "kv-2".into()],
            disaggregated_prefill_decode: true,
        };
        let prefetch = KvPrefetchPlan {
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
        let ctx = PrecomputedContextAsset {
            asset_id: "ctx-1".into(),
            asset_type: "prompt".into(),
            profile_id: "profile-1".into(),
            content_hash: "hash".into(),
            token_count: 1024,
            pinned: true,
            storage_tiers_supported: vec![KvStorageTier::Hbm, KvStorageTier::CpuRam],
        };
        let kv = PrecomputedKvAsset {
            kv_asset_id: "kvasset-1".into(),
            context_asset_id: "ctx-1".into(),
            kv_object_id: "kv-1".into(),
            quantization_mode: KvCodec::Tq4,
            compression_mode: "hybrid".into(),
            calibration_status: "complete".into(),
            calibration_dataset_id: Some("dataset-1".into()),
        };
        let telemetry = KvDecodeTelemetryBundle {
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
        let round_trips = [
            serde_json::to_string(&transfer)
                .and_then(|json| serde_json::from_str::<KvTransferPlan>(&json).map(|value| value == transfer))
                .unwrap(),
            serde_json::to_string(&prefetch)
                .and_then(|json| serde_json::from_str::<KvPrefetchPlan>(&json).map(|value| value == prefetch))
                .unwrap(),
            serde_json::to_string(&eviction)
                .and_then(|json| serde_json::from_str::<KvEvictionDecision>(&json).map(|value| value == eviction))
                .unwrap(),
            serde_json::to_string(&graph)
                .and_then(|json| serde_json::from_str::<AgentStepGraph>(&json).map(|value| value == graph))
                .unwrap(),
            serde_json::to_string(&ctx)
                .and_then(|json| serde_json::from_str::<PrecomputedContextAsset>(&json).map(|value| value == ctx))
                .unwrap(),
            serde_json::to_string(&kv)
                .and_then(|json| serde_json::from_str::<PrecomputedKvAsset>(&json).map(|value| value == kv))
                .unwrap(),
            serde_json::to_string(&telemetry)
                .and_then(|json| serde_json::from_str::<KvDecodeTelemetryBundle>(&json).map(|value| value == telemetry))
                .unwrap(),
        ];
        assert!(round_trips.into_iter().all(|passed| passed));
    }

    #[test]
    fn rvllm_graph_contracts_enforce_layout_hashes() {
        let mut pool = GraphPool::new();
        let layout = MetadataLayout::compute(128, 129);
        pool.insert(CapturedGraph {
            bucket: 128,
            max_blocks: 129,
            layout_hash: layout.hash(),
            fingerprint: GraphFingerprint([0u8; 32]),
        });

        assert!(pool.check_before_replay(128, 129, &layout).is_ok());

        let wrong = MetadataLayout::compute(128, 257);
        let err = pool.check_before_replay(128, 129, &wrong).unwrap_err();
        assert!(matches!(
            err,
            GraphError::CaptureMetadataMismatch { .. }
        ));

        let missing = pool
            .check_before_replay(1, 8, &MetadataLayout::compute(1, 8))
            .unwrap_err();
        assert!(matches!(missing, GraphError::BucketMissing { .. }));
    }

    #[test]
    fn radix_snapshot_round_trip() {
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
        let bytes = snapshot.to_bytes().unwrap();
        let restored = RadixTreeSnapshot::from_bytes(&bytes).unwrap();
        assert_eq!(snapshot, restored);
        assert_eq!(snapshot.node_count(), 4);
        assert!(snapshot.total_edge_bytes() > 0);
    }

    #[test]
    fn dual_hash_helpers_are_deterministic() {
        let tokens = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let hash_a = compute_content_hash(&tokens[..4]);
        let hash_b = compute_content_hash(&tokens[..4]);
        assert_eq!(hash_a, hash_b);

        let request_hashes = compute_request_content_hashes(&tokens, 4);
        assert_eq!(request_hashes.len(), 2);
        assert_eq!(request_hashes[0], compute_content_hash(&tokens[..4]));
        assert_eq!(request_hashes[1], compute_content_hash(&tokens[4..8]));
        assert!(compute_request_content_hashes(&tokens, 0).is_empty());
    }

    #[test]
    fn prefix_match_result_reports_hit_ratio() {
        let result = PrefixMatchResult::new("worker-1", 3, 4);
        assert_eq!(result.tenant(), "worker-1");
        assert_eq!(result.matched_count(), 3);
        assert_eq!(result.input_count(), 4);
        assert!((result.hit_ratio() - 0.75).abs() < f64::EPSILON);
        let empty = PrefixMatchResult::new("worker-2", 0, 0);
        assert_eq!(empty.hit_ratio(), 0.0);
    }

    #[test]
    fn positional_index_contract_round_trips() {
        let key = PositionalIndexKey {
            position: 3,
            content_hash: ContentHash(0xdead_beef),
        };
        let entry = PositionalIndexEntry {
            key,
            sequence_hash: SequenceHash(0xfeed_cafe),
            worker_id: 7,
        };
        let json = serde_json::to_string(&entry).expect("serialize");
        let back: PositionalIndexEntry = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(entry, back);
        assert_eq!(back.key.position, 3);
        assert_eq!(back.worker_id, 7);

        let e1 = PositionalIndexError::WorkerNotTracked(99);
        assert!(e1.to_string().contains("99"));
        let e2 = PositionalIndexError::ParentBlockNotFound(5, 2);
        assert!(e2.to_string().contains("5"));
        assert!(e2.to_string().contains("2"));

        let ok: PositionalIndexResult<u32> = Ok(42);
        assert_eq!(ok.unwrap(), 42);
        let err: PositionalIndexResult<u32> = Err(PositionalIndexError::WorkerNotTracked(1));
        assert!(err.is_err());
    }

    #[test]
    fn fp8_kv_dimension_alignment_validation() {
        // Aligned dimensions should pass
        assert_eq!(validate_fp8_kv_dimension(16, "model").unwrap(), 16);
        assert_eq!(validate_fp8_kv_dimension(32, "model").unwrap(), 32);
        assert_eq!(validate_fp8_kv_dimension(64, "model").unwrap(), 64);
        assert_eq!(validate_fp8_kv_dimension(128, "model").unwrap(), 128);

        // Misaligned dimensions should fail with proper error
        let err = validate_fp8_kv_dimension(20, "deepseek_v2");
        assert!(err.is_err());
        let msg = err.unwrap_err().to_string();
        assert!(msg.contains("20"));
        assert!(msg.contains("aligned to 32"));

        let err = validate_fp8_kv_dimension(100, "test_model");
        assert!(err.is_err());
        let msg = err.unwrap_err().to_string();
        assert!(msg.contains("100"));
        assert!(msg.contains("aligned to 112"));
    }

    #[test]
    fn align_dimension_to_16_helper() {
        // DeepSeek v2 case: head_dim (128) + 4 offset = 132, should align to 144
        assert_eq!(align_dimension_to_16(132), 144);

        // Already aligned
        assert_eq!(align_dimension_to_16(16), 16);
        assert_eq!(align_dimension_to_16(64), 64);
        assert_eq!(align_dimension_to_16(128), 128);

        // Various unaligned cases
        assert_eq!(align_dimension_to_16(1), 16);
        assert_eq!(align_dimension_to_16(17), 32);
        assert_eq!(align_dimension_to_16(100), 112);
        assert_eq!(align_dimension_to_16(255), 256);
    }

    #[test]
    fn universal_kv_stage_policy_prefers_hot_for_recent_high_importance() {
        let policy = UniversalKvPlacementPolicy::default();
        let stage = policy.select_stage(0.95, 8, 0.40);
        assert_eq!(stage, UniversalKvStage::HotRotor);
    }

    #[test]
    fn universal_kv_stage_policy_demotes_under_pressure() {
        let policy = UniversalKvPlacementPolicy::default();
        let stage = policy.select_stage(0.95, 8, 0.98);
        assert_eq!(stage, UniversalKvStage::WarmRotorPolar);
        let cold = policy.select_stage(0.10, 900, 0.98);
        assert_eq!(cold, UniversalKvStage::ColdTurboResidual);
    }

    #[test]
    fn universal_kv_block_header_round_trip_and_flags() {
        let header = UniversalKvBlockHeaderV1 {
            block_size: 16,
            bit_width: 3,
            rotor_id: 7,
            codec: KvCodec::Rq3Planar,
            stage: UniversalKvStage::WarmRotorPolar,
            scale: 0.125,
            flags: UniversalKvBlockHeaderV1::FLAG_TURBO_RESIDUAL,
            origin_model_tag: 42,
        };
        let json = serde_json::to_string(&header).unwrap();
        let restored: UniversalKvBlockHeaderV1 = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.block_size, 16);
        assert_eq!(restored.bit_width, 3);
        assert!(restored.should_apply_turbo_residual());
    }
}
