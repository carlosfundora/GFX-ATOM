//! TurboQuant KV Cache Codec for gfxATOM
//!
//! Provides two-stage vector compression (PolarQuant + QJL) for extreme KV cache compression.
//! Achieves 8-16x compression with tunable accuracy floors.

use serde::{Deserialize, Serialize};
use std::fmt;

pub mod error;
pub mod polar;
pub mod qjl;
pub mod turbo;
pub mod simd;

#[cfg(feature = "python")]
pub mod ffi;

pub use error::{Error, Result};
pub use polar::{PolarCode, PolarQuantizer};
pub use qjl::{QjlQuantizer, QjlSketch};
pub use turbo::{TurboCode, TurboQuantizer};

/// Supported bit-width modes for TurboQuant KV compression
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BitWidth {
    /// 1-bit per value: aggressive compression, ~20% loss
    Bit1 = 1,
    /// 2-bit per value: standard KV cache, ~10% loss
    Bit2 = 2,
    /// 3-bit per value: high-quality KV, ~5% loss
    Bit3 = 3,
    /// 4-bit per value: premium compression, ~3% loss
    Bit4 = 4,
    /// 8-bit per value: near-lossless, ~0.5% loss
    Bit8 = 8,
}

impl BitWidth {
    /// Human-readable name for this bit width
    pub fn name(&self) -> &'static str {
        match self {
            BitWidth::Bit1 => "1-bit-turbo",
            BitWidth::Bit2 => "2-bit-turbo",
            BitWidth::Bit3 => "3-bit-turbo",
            BitWidth::Bit4 => "4-bit-turbo",
            BitWidth::Bit8 => "8-bit-turbo",
        }
    }

    /// Maximum accuracy loss (relative) for this mode
    pub fn max_loss(&self) -> f32 {
        match self {
            BitWidth::Bit1 => 0.20,
            BitWidth::Bit2 => 0.10,
            BitWidth::Bit3 => 0.05,
            BitWidth::Bit4 => 0.03,
            BitWidth::Bit8 => 0.005,
        }
    }

    /// Typical use case for this mode
    pub fn use_case(&self) -> &'static str {
        match self {
            BitWidth::Bit1 => "Aggressive compression (batch inference)",
            BitWidth::Bit2 => "Standard KV cache (production)",
            BitWidth::Bit3 => "High-quality KV (latency-sensitive)",
            BitWidth::Bit4 => "Premium compression (accuracy critical)",
            BitWidth::Bit8 => "Reference/verification (near-lossless)",
        }
    }

    /// Bytes used per dimension (approximate)
    pub fn bytes_per_dim(&self) -> usize {
        match self {
            BitWidth::Bit1 => 1, // 8 values per byte
            BitWidth::Bit2 => 1, // 4 values per byte
            BitWidth::Bit3 => 1, // ~2.67 values per byte
            BitWidth::Bit4 => 2, // 2 values per byte
            BitWidth::Bit8 => 1, // 1 value per byte
        }
    }
}

impl fmt::Display for BitWidth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Codec registry entry for TurboQuant modes (non-serialized)
#[derive(Debug, Clone)]
pub struct CodecInfo {
    pub name: String,
    pub bit_width: BitWidth,
    pub accuracy_floor: f32,
    pub compression_ratio: f32,
    pub use_case: String,
    pub flags: Vec<String>,
}

impl CodecInfo {
    /// Create info for a given bit width
    pub fn for_mode(bit_width: BitWidth) -> Self {
        let compression_ratio = match bit_width {
            BitWidth::Bit1 => 16.0,
            BitWidth::Bit2 => 8.0,
            BitWidth::Bit3 => 5.33,
            BitWidth::Bit4 => 4.0,
            BitWidth::Bit8 => 2.0,
        };

        let flags = match bit_width {
            BitWidth::Bit1 => vec!["experimental".to_string(), "aggressive".to_string()],
            BitWidth::Bit2 => vec!["production".to_string(), "recommended".to_string()],
            BitWidth::Bit3 => vec!["production".to_string(), "high_quality".to_string()],
            BitWidth::Bit4 => vec!["production".to_string(), "premium".to_string()],
            BitWidth::Bit8 => vec!["reference".to_string(), "verification".to_string()],
        };

        Self {
            name: format!("turboquant_{}", bit_width.name()),
            bit_width,
            accuracy_floor: bit_width.max_loss(),
            compression_ratio,
            use_case: bit_width.use_case().to_string(),
            flags,
        }
    }

    /// All available codec modes
    pub fn all_modes() -> Vec<Self> {
        vec![
            Self::for_mode(BitWidth::Bit1),
            Self::for_mode(BitWidth::Bit2),
            Self::for_mode(BitWidth::Bit3),
            Self::for_mode(BitWidth::Bit4),
            Self::for_mode(BitWidth::Bit8),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bit_width_properties() {
        assert_eq!(BitWidth::Bit1.name(), "1-bit-turbo");
        assert_eq!(BitWidth::Bit2.max_loss(), 0.10);
        assert_eq!(BitWidth::Bit8.max_loss(), 0.005);
    }

    #[test]
    fn test_codec_registry() {
        let all = CodecInfo::all_modes();
        assert_eq!(all.len(), 5);
        assert!(all.iter().all(|c| c.bit_width == BitWidth::Bit1
            || c.bit_width == BitWidth::Bit2
            || c.bit_width == BitWidth::Bit3
            || c.bit_width == BitWidth::Bit4
            || c.bit_width == BitWidth::Bit8));
    }
}
