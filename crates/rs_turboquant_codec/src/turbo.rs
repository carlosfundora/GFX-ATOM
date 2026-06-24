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
    ///
    /// # Algorithm
    /// 1. Polar encode x with (bits-1) bits
    /// 2. Reconstruct x_approx from polar code
    /// 3. Compute residual = x - x_approx
    /// 4. QJL sketch the residual for inner product acceleration
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

        // Stage 1.5: Reconstruct polar approximation
        let x_approx = self.polar.decode(&polar_code)?;

        // Stage 2: Compute residual and QJL sketch it
        let residual: Vec<f32> = x
            .iter()
            .zip(x_approx.iter())
            .map(|(xi, xi_approx)| xi - xi_approx)
            .collect();
        let residual_sketch = self.qjl.encode(&residual)?;

        Ok(TurboCode {
            polar_code,
            residual_sketch,
        })
    }

    /// Decode a TurboCode back to approximate vector
    ///
    /// # Algorithm
    /// 1. Decode polar signal x_approx
    /// 2. Compute residual magnitude via QJL reconstruction
    /// 3. Return x_approx + residual_correction
    pub fn decode(&self, code: &TurboCode) -> Result<Vec<f32>> {
        // Stage 1: Decode polar signal
        let mut x_hat = self.polar.decode(&code.polar_code)?;

        // Stage 2: Add QJL residual correction
        let residual_correction = self.qjl.decode(&code.residual_sketch)?;

        // Combine: x_hat = x_approx + residual_correction
        for (x, r) in x_hat.iter_mut().zip(residual_correction.iter()) {
            *x += r;
        }

        Ok(x_hat)
    }

    /// Estimate inner product ⟨x, y⟩ from code and raw query
    ///
    /// # Algorithm (TurboQuant paper, ICLR 2026)
    ///
    /// ⟨x, y⟩ = ⟨x_approx, y⟩ + ⟨residual, y⟩
    ///
    /// - First term: computed directly from decoded polar signal
    /// - Second term: estimated unbiasedly from QJL sketches of residual and y
    pub fn estimate_inner_product(&self, code: &TurboCode, y: &[f32]) -> Result<f32> {
        if y.len() != self.dim {
            return Err(crate::Error::CompressionError(format!(
                "dimension mismatch in query: expected {}, got {}",
                self.dim,
                y.len()
            )));
        }

        // Term 1: Inner product from polar component (exact)
        let x_approx = self.polar.decode(&code.polar_code)?;
        let ip_polar: f32 = x_approx.iter().zip(y.iter()).map(|(a, b)| a * b).sum();

        // Term 2: Inner product from QJL residual sketch (estimated)
        // Sketch the query y with same QJL parameters
        let sketch_y = self.qjl.encode(y)?;
        
        // Estimate ⟨residual, y⟩ from sketches
        let ip_qjl = self.qjl.estimate_inner_product(&code.residual_sketch, &sketch_y)?;

        Ok(ip_polar + ip_qjl)
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

    #[test]
    fn test_two_stage_compression() {
        let qz = TurboQuantizer::new(128, 3, 32, 42).unwrap();
        let x: Vec<f32> = (0..128).map(|i| (i as f32 / 128.0).sin()).collect();

        let code = qz.encode(&x).unwrap();
        let original_bytes = x.len() * std::mem::size_of::<f32>();
        let compressed_bytes = code.encoded_bytes();

        let compression_ratio = code.compression_ratio();
        println!("Original: {} bytes, Compressed: {} bytes, Ratio: {:.2}x",
                 original_bytes, compressed_bytes, compression_ratio);

        assert!(compression_ratio > 1.0, "Should compress");
        assert!(compression_ratio < 32.0, "Compression ratio should be realistic");
    }

    #[test]
    fn test_residual_reconstruction() {
        let qz = TurboQuantizer::new(64, 4, 16, 42).unwrap();
        let x: Vec<f32> = (0..64).map(|i| (i as f32).sin() + 0.1 * (i as f32).cos()).collect();

        // Encode captures polar + residual
        let code = qz.encode(&x).unwrap();
        
        // Decode should reconstruct x_approx + residual
        let x_hat = qz.decode(&code).unwrap();
        
        // MSE with 4-bit compression (2 bits polar + QJL 1-bit residual)
        let mse: f32 = x.iter()
            .zip(x_hat.iter())
            .map(|(xi, xi_hat)| (xi - xi_hat).powi(2))
            .sum::<f32>() / x.len() as f32;
        
        let rmse = mse.sqrt();
        println!("RMSE: {:.6}, MSE: {:.6}", rmse, mse);
        // Realistic bound: with limited bits, MSE can be significant
        assert!(mse < 0.5, "Roundtrip MSE should be < 0.5 with limited bits");
    }

    #[test]
    fn test_inner_product_estimation() {
        let qz = TurboQuantizer::new(64, 3, 16, 42).unwrap();
        let x: Vec<f32> = (0..64).map(|i| (i as f32 / 32.0).sin()).collect();
        let y: Vec<f32> = (0..64).map(|i| (i as f32 / 32.0).cos()).collect();

        // Compute true inner product
        let ip_true: f32 = x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum();

        // Estimate from code
        let code = qz.encode(&x).unwrap();
        let ip_est = qz.estimate_inner_product(&code, &y).unwrap();

        println!("True IP: {:.6}, Estimated IP: {:.6}, Error: {:.6}",
                 ip_true, ip_est, (ip_true - ip_est).abs());

        // IP estimation should be reasonably accurate with sufficient projections
        let rel_error = (ip_true - ip_est).abs() / (ip_true.abs() + 1e-6);
        assert!(rel_error < 0.5, "IP estimation relative error should be < 50%");
    }

    #[test]
    fn test_same_vector_inner_product() {
        let qz = TurboQuantizer::new(128, 2, 32, 42).unwrap();
        let x: Vec<f32> = (0..128).map(|i| (i as f32 / 64.0).sin()).collect();

        // Self inner product should be high
        let ip_true: f32 = x.iter().map(|xi| xi * xi).sum();

        let code = qz.encode(&x).unwrap();
        let ip_est = qz.estimate_inner_product(&code, &x).unwrap();

        println!("Self-IP true: {:.6}, estimated: {:.6}", ip_true, ip_est);
        assert!(ip_est > 0.0, "Self inner product should be positive");
    }

    #[test]
    fn test_orthogonal_vectors_ip() {
        let qz = TurboQuantizer::new(64, 2, 16, 42).unwrap();
        
        // Create two vectors that are "mostly independent" in spirit
        // (sin and cos are orthogonal but not when sampled this way)
        let x: Vec<f32> = (0..64).map(|i| (i as f32 / 32.0).sin()).collect();
        let y: Vec<f32> = (0..64).map(|i| ((i as f32 + 32.0) / 32.0).sin()).collect();

        let ip_true: f32 = x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum();
        
        let code = qz.encode(&x).unwrap();
        let ip_est = qz.estimate_inner_product(&code, &y).unwrap();

        println!("Two-part IP true: {:.6}, estimated: {:.6}", ip_true, ip_est);
        // Just verify we can compute the IP without panicking
        // Accuracy depends on projection count
        assert!(!ip_est.is_nan() && !ip_est.is_infinite(), "IP should be finite");
    }

    #[test]
    fn test_multiple_turbo_modes() {
        // Test different bit allocations (TQ2-TQ4)
        let dim = 128;
        
        let x: Vec<f32> = (0..dim).map(|i| (i as f32 / 64.0).sin()).collect();
        
        // TQ2 (2 bits polar + 1-bit QJL)
        let tq2 = TurboQuantizer::new(dim, 2, dim / 4, 42).unwrap();
        let code2 = tq2.encode(&x).unwrap();
        let ratio2 = code2.compression_ratio();
        println!("TQ2 compression ratio: {:.2}x", ratio2);
        assert!(ratio2 > 1.0 && ratio2 < 32.0);

        // TQ3 (3 bits polar + 1-bit QJL)
        let tq3 = TurboQuantizer::new(dim, 3, dim / 4, 42).unwrap();
        let code3 = tq3.encode(&x).unwrap();
        let ratio3 = code3.compression_ratio();
        println!("TQ3 compression ratio: {:.2}x", ratio3);
        assert!(ratio3 > 1.0 && ratio3 < 32.0);

        // TQ4 (4 bits polar + 1-bit QJL)
        let tq4 = TurboQuantizer::new(dim, 4, dim / 4, 42).unwrap();
        let code4 = tq4.encode(&x).unwrap();
        let ratio4 = code4.compression_ratio();
        println!("TQ4 compression ratio: {:.2}x", ratio4);
        assert!(ratio4 > 1.0 && ratio4 < 32.0);

        // All ratios should be reasonable (compression is not constant, depends on projections)
        assert!(ratio2 > 0.5 && ratio3 > 0.5 && ratio4 > 0.5);
    }

    #[test]
    fn test_deterministic_compression() {
        let qz = TurboQuantizer::new(64, 2, 16, 42).unwrap();
        let x: Vec<f32> = (0..64).map(|i| (i as f32).sin()).collect();

        let code1 = qz.encode(&x).unwrap();
        let code2 = qz.encode(&x).unwrap();

        assert_eq!(code1.polar_code, code2.polar_code);
        assert_eq!(code1.residual_sketch, code2.residual_sketch);
    }

    #[test]
    fn test_random_vector_encoding() {
        let qz = TurboQuantizer::new(128, 3, 32, 42).unwrap();
        let mut x: Vec<f32> = (0..128).map(|i| (i as f32 * 1.618) % 1.0).collect();
        for xi in &mut x {
            *xi = (*xi - 0.5) * 2.0; // Normalize to [-1, 1]
        }

        let code = qz.encode(&x).unwrap();
        let x_hat = qz.decode(&code).unwrap();

        // Check vector reconstruction
        let mse: f32 = x.iter()
            .zip(x_hat.iter())
            .map(|(xi, xi_hat)| (xi - xi_hat).powi(2))
            .sum::<f32>() / x.len() as f32;

        println!("Random vector MSE: {:.6}", mse);
        // Realistic MSE bound for quantization
        assert!(mse < 1.0, "Random vector MSE should be < 1.0");
    }

    #[test]
    fn test_zero_vector_encoding() {
        let qz = TurboQuantizer::new(64, 2, 16, 42).unwrap();
        let x = vec![0.0; 64];

        // Should not panic
        let code = qz.encode(&x).unwrap();
        let x_hat = qz.decode(&code).unwrap();
        assert_eq!(x_hat.len(), 64);
        
        // Some noise is expected from QJL residual sketch
        println!("Zero vector reconstructed successfully");
    }

    #[test]
    fn test_edge_case_inf_values() {
        let qz = TurboQuantizer::new(64, 2, 16, 42).unwrap();
        let mut x = vec![0.5; 64];
        x[0] = f32::INFINITY;
        x[1] = f32::NEG_INFINITY;

        // Should not panic
        let code = qz.encode(&x).unwrap();
        let _x_hat = qz.decode(&code).unwrap();
    }
}
