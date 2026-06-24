//! Stage-1 HIP scaffold for Universal KV compression + reshape.
//!
//! This module is intentionally non-invasive: it does not alter production
//! dispatch, and only records guardrails + discovered kernel integration points
//! for future wiring.

/// Static descriptor of a reuse-first kernel asset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DiscoveredKernelAsset {
    pub id: &'static str,
    pub source_path: &'static str,
    pub role: &'static str,
    pub integration_point: &'static str,
}

/// Guardrails for enabling Stage-1 HIP kernel execution on RDNA2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Stage1HipGuardrails {
    pub target_gfx: &'static str,
    pub require_wave32: bool,
    pub require_hip_runtime: bool,
}

impl Default for Stage1HipGuardrails {
    fn default() -> Self {
        Self {
            target_gfx: "gfx1030",
            require_wave32: true,
            require_hip_runtime: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GuardrailDecision {
    Ready,
    Reject { reason: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stage1DispatchPlan {
    pub rotor_hot_asset_id: &'static str,
    pub turbo_residual_asset_id: &'static str,
    pub reshape_asset_id: &'static str,
    pub notes: &'static str,
}

/// Reuse inventory for Stage-1 compression + reshape wiring.
///
/// INTEGRATION_POINT(rotor-hot):
///   lift Planar/Iso block layout and pack/unpack behavior from these kernels.
///
/// INTEGRATION_POINT(turbo-cold):
///   feed residual path through existing TurboQuant packing/rotation assets.
///
/// INTEGRATION_POINT(reshape):
///   align Stage-1 tensor layout transforms with RDNA2 decode kernels.
pub const STAGE1_KERNEL_ASSETS: &[DiscoveredKernelAsset] = &[
    DiscoveredKernelAsset {
        id: "inventory-kv-dedupe-map",
        source_path: "gfxATOM-Rust/inventory/kv-dedupe-map.json",
        role: "Canonical source map for kernel reuse and dedupe ownership",
        integration_point: "broker_kernel_registry",
    },
    DiscoveredKernelAsset {
        id: "build-tq3-kv-cache-core",
        source_path: "/home/local/ai/build/kernels/tq3_0-kv-cache/ggml/src/ggml-quants.c",
        role: "TurboQuant/TQ block packing and quant-dequant reference path",
        integration_point: "stage1_turbo_residual_pack",
    },
    DiscoveredKernelAsset {
        id: "build-llama-tq3-cuda-path",
        source_path: "/home/local/ai/build/kernels/llama-cpp-tq3-kvcache/ggml-cuda/mmvq.cu",
        role: "Fused CUDA MMVQ path useful for HIP parity planning",
        integration_point: "stage1_turbo_fused_dot",
    },
    DiscoveredKernelAsset {
        id: "donor-llama-planar-quant",
        source_path: "donors/llama.cpp-1-bit-turbo/ggml/src/ggml-planar-quant.c",
        role: "Rotor PlanarQuant block layout and sign/magnitude packing",
        integration_point: "stage1_rotor_planar_pack",
    },
    DiscoveredKernelAsset {
        id: "donor-llama-iso-quant",
        source_path: "donors/llama.cpp-1-bit-turbo/ggml/src/ggml-iso-quant.c",
        role: "Rotor IsoQuant block layout and dequant layout parity",
        integration_point: "stage1_rotor_iso_pack",
    },
    DiscoveredKernelAsset {
        id: "donor-sglang-rdna2-decode",
        source_path:
            "donors/sglang-1-bit-turbo/python/sglang/srt/layers/attention/triton_ops/rdna2/decode_attention.py",
        role: "RDNA2 decode path and Wave32-oriented block sizing assumptions",
        integration_point: "stage1_reshape_to_decode_layout",
    },
    DiscoveredKernelAsset {
        id: "donor-sglang-rotor-engine",
        source_path:
            "donors/sglang-1-bit-turbo/python/sglang/srt/layers/quantization/rotorquant_engine.py",
        role: "RotorQuant normalization/rotation pipeline for Stage-1 encode semantics",
        integration_point: "stage1_rotor_encode_contract",
    },
    DiscoveredKernelAsset {
        id: "donor-sglang-turbo-kv",
        source_path:
            "donors/sglang-1-bit-turbo/python/sglang/srt/layers/quantization/turboquant_kv.py",
        role: "TurboQuant latent+residual split contract for warm/cold tiers",
        integration_point: "stage1_turbo_residual_contract",
    },
];

pub fn stage1_kernel_assets() -> &'static [DiscoveredKernelAsset] {
    STAGE1_KERNEL_ASSETS
}

pub fn evaluate_guardrails(
    guardrails: Stage1HipGuardrails,
    detected_arch: &str,
    wavefront_size: u32,
    hip_runtime_present: bool,
) -> GuardrailDecision {
    if detected_arch != guardrails.target_gfx {
        return GuardrailDecision::Reject {
            reason: format!(
                "stage1 hip scaffold requires {}, got {}",
                guardrails.target_gfx, detected_arch
            ),
        };
    }
    if guardrails.require_wave32 && wavefront_size != 32 {
        return GuardrailDecision::Reject {
            reason: format!("stage1 hip scaffold requires wave32, got wave{}", wavefront_size),
        };
    }
    if guardrails.require_hip_runtime && !hip_runtime_present {
        return GuardrailDecision::Reject {
            reason: "stage1 hip scaffold requires hip runtime".to_string(),
        };
    }
    GuardrailDecision::Ready
}

pub fn stage1_scaffold_plan() -> Stage1DispatchPlan {
    Stage1DispatchPlan {
        rotor_hot_asset_id: "donor-llama-planar-quant",
        turbo_residual_asset_id: "build-tq3-kv-cache-core",
        reshape_asset_id: "donor-sglang-rdna2-decode",
        notes: "Scaffold-only mapping: no production dispatch wired yet.",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stage1_assets_cover_required_sources() {
        let assets = stage1_kernel_assets();
        assert!(assets.iter().any(|a| a.source_path.contains("inventory/")));
        assert!(assets.iter().any(|a| a.source_path.contains("/home/local/ai/build/")));
        assert!(assets.iter().any(|a| a.source_path.contains("donors/llama.cpp-1-bit-turbo")));
        assert!(assets.iter().any(|a| a.source_path.contains("donors/sglang-1-bit-turbo")));
    }

    #[test]
    fn guardrails_accept_gfx1030_wave32_hip() {
        let decision = evaluate_guardrails(Stage1HipGuardrails::default(), "gfx1030", 32, true);
        assert_eq!(decision, GuardrailDecision::Ready);
    }

    #[test]
    fn guardrails_reject_non_wave32() {
        let decision = evaluate_guardrails(Stage1HipGuardrails::default(), "gfx1030", 64, true);
        assert!(matches!(decision, GuardrailDecision::Reject { .. }));
    }
}
