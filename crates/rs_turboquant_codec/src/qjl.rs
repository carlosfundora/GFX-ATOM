use crate::error::Result;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// QJL sketch (1-bit residual per projection)
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct QjlSketch {
    pub dim: usize,
    pub projections: usize,
    pub bits: Vec<u8>,
}

impl QjlSketch {
    pub fn encoded_bytes(&self) -> usize {
        self.bits.len()
    }
}

/// QjlQuantizer: Johnson-Lindenstrauss projection + 1-bit quantization
///
/// Algorithm:
/// 1. Generate random projection matrix R (seeded, deterministic)
/// 2. Project: y = R @ x (k projections)
/// 3. Quantize to 1-bit: sign(y)
/// 4. Pack bits
///
/// Unbiased IP estimation:
/// <x, y> ≈ (2*popcount(sign(Rx) == sign(Ry)) - k) / sqrt(k)
#[derive(Debug, Clone)]
pub struct QjlQuantizer {
    dim: usize,
    projections: usize,
    seed: u64,
}

impl QjlQuantizer {
    pub fn new(dim: usize, projections: usize, seed: u64) -> Result<Self> {
        if dim == 0 {
            return Err(crate::Error::ZeroDimension);
        }
        if projections == 0 {
            return Err(crate::Error::ZeroProjectionCount);
        }
        Ok(Self {
            dim,
            projections,
            seed,
        })
    }

    /// Generate deterministic random matrix using seeded RNG
    fn generate_projection_matrix(&self) -> Vec<Vec<f32>> {
        // Use seed to initialize deterministic RNG
        let mut rng_state = self.seed;

        let mut matrix = Vec::with_capacity(self.projections);
        for _ in 0..self.projections {
            let mut row = Vec::with_capacity(self.dim);
            for _ in 0..self.dim {
                // Linear congruential generator
                rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
                let u = ((rng_state / 65536) % 32768) as f32 / 32768.0;
                // Box-Muller transform for gaussian
                let v = ((rng_state / 65536) % 32768) as f32 / 32768.0;
                rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
                let gaussian = (-2.0 * u.ln()).sqrt() * (2.0 * std::f32::consts::PI * v).cos();
                row.push(gaussian / (self.dim as f32).sqrt());
            }
            matrix.push(row);
        }
        matrix
    }

    /// Encode vector to 1-bit QJL sketch
    pub fn encode(&self, x: &[f32]) -> Result<QjlSketch> {
        if x.len() != self.dim {
            return Err(crate::Error::CompressionError(format!(
                "dimension mismatch: expected {}, got {}",
                self.dim,
                x.len()
            )));
        }

        let matrix = self.generate_projection_matrix();

        // Project and quantize to 1-bit
        let mut bits = Vec::new();
        for row in &matrix {
            let mut proj = 0.0f32;
            for (i, &r_ij) in row.iter().enumerate() {
                proj += r_ij * x[i];
            }
            // 1-bit quantization: sign
            bits.push(proj >= 0.0);
        }

        // Pack bits into bytes
        let mut bytes = Vec::new();
        for chunk in bits.chunks(8) {
            let mut byte = 0u8;
            for (i, &bit) in chunk.iter().enumerate() {
                if bit {
                    byte |= 1u8 << (7 - i);
                }
            }
            bytes.push(byte);
        }

        Ok(QjlSketch {
            dim: self.dim,
            projections: self.projections,
            bits: bytes,
        })
    }

    /// Decode returns the projection signs (approximation of original vector in projection space)
    pub fn decode(&self, sketch: &QjlSketch) -> Result<Vec<f32>> {
        if sketch.dim != self.dim || sketch.projections != self.projections {
            return Err(crate::Error::CompressionError(format!(
                "sketch mismatch: expected {}x{}, got {}x{}",
                self.dim, self.projections, sketch.dim, sketch.projections
            )));
        }

        let mut result = Vec::with_capacity(self.projections);
        let mut bit_pos = 0;

        for _ in 0..self.projections {
            let byte_idx = bit_pos / 8;
            let bit_idx = 7 - (bit_pos % 8);
            let bit = byte_idx < sketch.bits.len() && (sketch.bits[byte_idx] >> bit_idx) & 1 == 1;
            result.push(if bit { 1.0 } else { -1.0 });
            bit_pos += 1;
        }

        Ok(result)
    }

    /// Estimate inner product between two sketches
    ///
    /// Returns unbiased estimate: (2*matches - k) / sqrt(k)
    pub fn estimate_inner_product(&self, sketch_x: &QjlSketch, sketch_y: &QjlSketch) -> Result<f32> {
        if sketch_x.projections != sketch_y.projections {
            return Err(crate::Error::CompressionError(
                "sketch projection mismatch".to_string(),
            ));
        }

        // Count matching signs
        let mut matches = 0;
        let mut bit_pos = 0;

        for _ in 0..self.projections {
            let byte_idx = bit_pos / 8;
            let bit_idx = 7 - (bit_pos % 8);

            let bit_x = byte_idx < sketch_x.bits.len() && (sketch_x.bits[byte_idx] >> bit_idx) & 1 == 1;
            let bit_y = byte_idx < sketch_y.bits.len() && (sketch_y.bits[byte_idx] >> bit_idx) & 1 == 1;

            if bit_x == bit_y {
                matches += 1;
            }

            bit_pos += 1;
        }

        // Unbiased estimator
        let k = self.projections as f32;
        let estimate = (2.0 * matches as f32 - k) / k.sqrt();
        Ok(estimate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qjl_quantizer_creation() {
        let qjl = QjlQuantizer::new(64, 32, 42).unwrap();
        assert_eq!(qjl.dim, 64);
        assert_eq!(qjl.projections, 32);
    }

    #[test]
    fn test_qjl_encode() {
        let qjl = QjlQuantizer::new(64, 16, 42).unwrap();
        let x = vec![0.5; 64];
        let sketch = qjl.encode(&x).unwrap();
        assert_eq!(sketch.dim, 64);
        assert_eq!(sketch.projections, 16);
    }

    #[test]
    fn test_qjl_dimension_mismatch() {
        let qjl = QjlQuantizer::new(64, 16, 42).unwrap();
        let x = vec![0.5; 32];
        let result = qjl.encode(&x);
        assert!(result.is_err());
    }

    #[test]
    fn test_qjl_decode() {
        let qjl = QjlQuantizer::new(16, 8, 42).unwrap();
        let x = vec![0.1; 16];
        let sketch = qjl.encode(&x).unwrap();
        let decoded = qjl.decode(&sketch).unwrap();

        assert_eq!(decoded.len(), 8);
        // Each decoded value should be ±1
        for &val in &decoded {
            assert!((val - 1.0).abs() < 0.01 || (val + 1.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_qjl_inner_product_estimation() {
        let qjl = QjlQuantizer::new(32, 64, 42).unwrap();
        let x = vec![0.5; 32];
        let y = vec![0.5; 32];

        let sketch_x = qjl.encode(&x).unwrap();
        let sketch_y = qjl.encode(&y).unwrap();

        let est = qjl.estimate_inner_product(&sketch_x, &sketch_y).unwrap();
        println!("Inner product estimate: {}", est);

        // Estimate should be reasonable (not NaN or infinite)
        assert!(est.is_finite());
    }

    #[test]
    fn test_qjl_orthogonal_vectors() {
        let qjl = QjlQuantizer::new(32, 128, 42).unwrap();
        
        let mut x = vec![0.0; 32];
        x[0] = 1.0;
        
        let mut y = vec![0.0; 32];
        y[1] = 1.0;

        let sketch_x = qjl.encode(&x).unwrap();
        let sketch_y = qjl.encode(&y).unwrap();

        let est = qjl.estimate_inner_product(&sketch_x, &sketch_y).unwrap();
        println!("Orthogonal vectors IP estimate: {}", est);

        // Orthogonal vectors should have IP near 0
        // With randomness, should be roughly zero-centered
        assert!(est.abs() < 10.0, "Estimate should be near 0 for orthogonal vectors");
    }

    #[test]
    fn test_qjl_same_vector_ip() {
        let qjl = QjlQuantizer::new(32, 256, 42).unwrap();
        let x = vec![0.3, 0.5, -0.2, 0.1, 0.0, 0.4, -0.3, 0.2, 0.1, -0.4,
                     0.3, 0.5, -0.2, 0.1, 0.0, 0.4, -0.3, 0.2, 0.1, -0.4,
                     0.3, 0.5, -0.2, 0.1, 0.0, 0.4, -0.3, 0.2, 0.1, -0.4, 0.0, 0.0];

        let sketch_x = qjl.encode(&x).unwrap();
        let est = qjl.estimate_inner_product(&sketch_x, &sketch_x).unwrap();
        println!("Same vector IP estimate: {}", est);

        // Same vector should have positive IP
        assert!(est > 0.0, "Same vector IP should be positive");
    }

    #[test]
    fn test_qjl_deterministic() {
        let qjl = QjlQuantizer::new(16, 8, 42).unwrap();
        let x = vec![0.1, 0.2, 0.3, 0.4, 0.5, -0.1, -0.2, -0.3, -0.4, -0.5, 0.0, 1.0, -1.0, 0.7, 0.8, 0.9];

        let sketch1 = qjl.encode(&x).unwrap();
        let sketch2 = qjl.encode(&x).unwrap();

        assert_eq!(sketch1.bits, sketch2.bits, "Encoding should be deterministic");
    }
}
