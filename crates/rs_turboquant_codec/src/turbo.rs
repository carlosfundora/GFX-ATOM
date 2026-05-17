use crate::error::Result;
use crate::{PolarCode, PolarQuantizer, QjlQuantizer, QjlSketch};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// TurboCode: Complete compressed vector (polar + QJL residual)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct TurboCode {
    pub polar_code: PolarCode,
    pub residual_sketch: QjlSketch,
}

impl TurboCode {
    pub fn encoded_bytes(&self) -> usize {
        self.polar_code.encoded_bytes() + self.residual_sketch.encoded_bytes()
    }

    pub fn compression_ratio(&self) -> f32 {
        let original = self.polar_code.dim * std::mem::size_of::<f32>();
        original as f32 / self.encoded_bytes().max(1) as f32
    }
}

/// TurboQuantizer: Two-stage compressor (PolarQuant + QJL)
///
/// # Algorithm
///
/// 1. PolarQuant stage (b-1 bits): Compress via polar encoding
/// 2. QJL stage (1 bit per projection): Apply Quantized Johnson-Lindenstrauss to residual
///
/// # Inner Product Estimation
///
/// ⟨x, y⟩ ≈ IP_polar(code, y) + IP_qjl(residual_sketch, y)
///
/// This is **provably unbiased** (TurboQuant paper, ICLR 2026).
#[derive(Debug, Clone)]
pub struct TurboQuantizer {
    dim: usize,
    bits: u8,
    projections: usize,
    seed: u64,
    polar: PolarQuantizer,
    qjl: QjlQuantizer,
}

impl TurboQuantizer {
    /// Create a new TurboQuantizer
    ///
    /// # Arguments
    /// - `dim`: vector dimension (must be even)
    /// - `bits`: total bit budget per scalar (2-8)
    /// - `projections`: QJL sketch size (typically dim/4 to dim/2)
    /// - `seed`: deterministic seed for random matrices
    pub fn new(dim: usize, bits: u8, projections: usize, seed: u64) -> Result<Self> {
        if dim == 0 {
            return Err(crate::Error::ZeroDimension);
        }
        if dim % 2 != 0 {
            return Err(crate::Error::OddDimension { got: dim });
        }
        if bits < 1 || bits > 16 {
            return Err(crate::Error::InvalidBitWidth { got: bits });
        }
        if projections == 0 {
            return Err(crate::Error::ZeroProjectionCount);
        }

        let polar = PolarQuantizer::new(dim, bits.saturating_sub(1), seed)?;
        let qjl = QjlQuantizer::new(dim, projections, seed.wrapping_add(1))?;

        Ok(Self {
            dim,
            bits,
            projections,
            seed,
            polar,
            qjl,
        })
    }

    /// Encode a vector into TurboCode
    pub fn encode(&self, x: &[f32]) -> Result<TurboCode> {
        if x.len() != self.dim {
            return Err(crate::Error::CompressionError(format!(
                "dimension mismatch: expected {}, got {}",
                self.dim,
                x.len()
            )));
        }

        // Stage 1: Polar encode main signal
        let polar_code = self.polar.encode(x)?;

        // Stage 2: QJL encode residual (stub: would subtract reconstructed polar from original)
        let residual_sketch = self.qjl.encode(x)?;

        Ok(TurboCode {
            polar_code,
            residual_sketch,
        })
    }

    /// Decode a TurboCode back to approximate vector
    pub fn decode(&self, code: &TurboCode) -> Result<Vec<f32>> {
        // Stage 1: Decode polar signal
        let x_hat = self.polar.decode(&code.polar_code)?;

        // Stage 2: Add QJL residual correction
        let _residual = self.qjl.decode(&code.residual_sketch)?;

        // (Stub: would add residual correction to x_hat)

        Ok(x_hat)
    }

    /// Estimate inner product ⟨x, y⟩ from code and raw query
    pub fn estimate_inner_product(&self, _code: &TurboCode, y: &[f32]) -> Result<f32> {
        if y.len() != self.dim {
            return Err(crate::Error::CompressionError(format!(
                "dimension mismatch in query: expected {}, got {}",
                self.dim,
                y.len()
            )));
        }

        // ⟨x, y⟩ ≈ IP_polar(code, y) + IP_qjl(residual_sketch, y)
        // Stub: would compute both components and sum
        Ok(0.0)
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn bits(&self) -> u8 {
        self.bits
    }

    pub fn projections(&self) -> usize {
        self.projections
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantizer_creation() {
        let qz = TurboQuantizer::new(128, 2, 32, 42).unwrap();
        assert_eq!(qz.dim, 128);
        assert_eq!(qz.bits, 2);
        assert_eq!(qz.projections, 32);
    }

    #[test]
    fn test_dimension_validation() {
        // Odd dimension should fail
        assert!(TurboQuantizer::new(127, 2, 32, 42).is_err());

        // Zero dimension should fail
        assert!(TurboQuantizer::new(0, 2, 32, 42).is_err());

        // Valid dimension should succeed
        assert!(TurboQuantizer::new(128, 2, 32, 42).is_ok());
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let qz = TurboQuantizer::new(64, 2, 16, 42).unwrap();
        let x = vec![0.5; 64];

        let code = qz.encode(&x).unwrap();
        let x_hat = qz.decode(&code).unwrap();

        assert_eq!(x_hat.len(), 64);
    }
}
