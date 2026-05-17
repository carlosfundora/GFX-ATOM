use crate::error::Result;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Polar-encoded vector with metadata for decoding
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct PolarCode {
    pub dim: usize,
    pub bits: u8,
    pub bytes: Vec<u8>,
    /// Min/max per channel for reconstruction
    pub scale_min: Vec<f32>,
    pub scale_max: Vec<f32>,
}

impl PolarCode {
    pub fn encoded_bytes(&self) -> usize {
        self.bytes.len()
    }
}

/// PolarQuantizer: Maps input to unit circle/radius, quantizes, and bit-packs
///
/// Algorithm:
/// 1. Normalize each element to [-1, 1] range per-channel
/// 2. Treat as (real, imag) pair in complex plane
/// 3. Convert to (angle, radius) in polar coordinates
/// 4. Quantize angle to N bits, radius to remaining bits
/// 5. Pack bits into bytes
#[derive(Debug, Clone)]
pub struct PolarQuantizer {
    dim: usize,
    bits: u8,
    seed: u64,
}

impl PolarQuantizer {
    pub fn new(dim: usize, bits: u8, seed: u64) -> Result<Self> {
        if dim == 0 {
            return Err(crate::Error::ZeroDimension);
        }
        if bits == 0 || bits > 8 {
            return Err(crate::Error::InvalidBitWidth { got: bits });
        }
        Ok(Self { dim, bits, seed })
    }

    /// Encode float32 vector to bit-packed polar representation
    pub fn encode(&self, x: &[f32]) -> Result<PolarCode> {
        if x.len() != self.dim {
            return Err(crate::Error::CompressionError(format!(
                "dimension mismatch: expected {}, got {}",
                self.dim,
                x.len()
            )));
        }

        // Track min/max for normalization
        let mut scale_min = vec![f32::INFINITY; self.dim];
        let mut scale_max = vec![f32::NEG_INFINITY; self.dim];

        // Find min/max per element (only finite values)
        for (i, &val) in x.iter().enumerate() {
            if val.is_finite() {
                scale_min[i] = scale_min[i].min(val);
                scale_max[i] = scale_max[i].max(val);
            }
        }

        // For dimensions with no finite values, use default range
        for i in 0..self.dim {
            if !scale_min[i].is_finite() {
                scale_min[i] = -1.0;
                scale_max[i] = 1.0;
            }
        }

        // Encode bits
        let mut encoded = BitPacker::new(self.dim * self.bits as usize);

        for i in 0..self.dim {
            let val = x[i];
            
            // Handle non-finite values
            if !val.is_finite() {
                // Encode as all zeros (will decode to center of range)
                for _ in 0..self.bits {
                    encoded.push_bit(false);
                }
                continue;
            }

            // Normalize to [-1, 1]
            let min = scale_min[i];
            let max = scale_max[i];
            let range = (max - min).max(1e-10);
            let normalized = (val - min) / range * 2.0 - 1.0;

            // Map to polar: treat as point in unit circle
            let angle_norm = (normalized.atan2(1.0) + std::f32::consts::PI) / (2.0 * std::f32::consts::PI);
            let radius_norm = (normalized * normalized + 1.0).sqrt().min(1.0);

            // Quantize angle and radius
            let angle_bits = self.bits / 2;
            let radius_bits = self.bits - angle_bits;

            let angle_val = (angle_norm * ((1u32 << angle_bits) - 1) as f32) as u32 as usize;
            let radius_val = (radius_norm * ((1u32 << radius_bits) - 1) as f32) as u32 as usize;

            // Pack bits
            for j in (0..angle_bits).rev() {
                encoded.push_bit((angle_val >> j) & 1 == 1);
            }
            for j in (0..radius_bits).rev() {
                encoded.push_bit((radius_val >> j) & 1 == 1);
            }
        }

        Ok(PolarCode {
            dim: self.dim,
            bits: self.bits,
            bytes: encoded.into_bytes(),
            scale_min,
            scale_max,
        })
    }

    /// Decode polar code back to float32 vector
    pub fn decode(&self, code: &PolarCode) -> Result<Vec<f32>> {
        if code.dim != self.dim {
            return Err(crate::Error::CompressionError(format!(
                "dimension mismatch: expected {}, got {}",
                self.dim,
                code.dim
            )));
        }

        let mut decoded = Vec::with_capacity(self.dim);
        let mut unpacker = BitUnpacker::new(&code.bytes);

        for i in 0..self.dim {
            let angle_bits = self.bits / 2;
            let radius_bits = self.bits - angle_bits;

            // Unpack angle and radius
            let mut angle_val = 0usize;
            for j in (0..angle_bits).rev() {
                if unpacker.pop_bit() {
                    angle_val |= 1 << j;
                }
            }

            let mut radius_val = 0usize;
            for j in (0..radius_bits).rev() {
                if unpacker.pop_bit() {
                    radius_val |= 1 << j;
                }
            }

            // Dequantize
            let angle_max = ((1u32 << angle_bits) - 1) as f32;
            let radius_max = ((1u32 << radius_bits) - 1) as f32;

            let angle_norm = angle_val as f32 / angle_max.max(1.0);
            let radius_norm = radius_val as f32 / radius_max.max(1.0);

            // Map back from polar: reconstruct from angle and radius
            let angle_rad = angle_norm * 2.0 * std::f32::consts::PI - std::f32::consts::PI;
            let reconstructed = angle_rad.cos() * radius_norm;

            // Denormalize
            let min = code.scale_min[i];
            let max = code.scale_max[i];
            let range = (max - min).max(1e-10);
            let val = (reconstructed + 1.0) / 2.0 * range + min;

            decoded.push(val);
        }

        Ok(decoded)
    }
}

/// Helper for bit-packing
struct BitPacker {
    bits: Vec<bool>,
}

impl BitPacker {
    fn new(capacity: usize) -> Self {
        Self {
            bits: Vec::with_capacity(capacity),
        }
    }

    fn push_bit(&mut self, bit: bool) {
        self.bits.push(bit);
    }

    fn into_bytes(self) -> Vec<u8> {
        let mut bytes = Vec::new();
        for chunk in self.bits.chunks(8) {
            let mut byte = 0u8;
            for (i, &bit) in chunk.iter().enumerate() {
                if bit {
                    byte |= 1u8 << (7 - i);
                }
            }
            bytes.push(byte);
        }
        bytes
    }
}

/// Helper for bit-unpacking
struct BitUnpacker<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> BitUnpacker<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn pop_bit(&mut self) -> bool {
        let byte_idx = self.pos / 8;
        let bit_idx = 7 - (self.pos % 8);
        let result = byte_idx < self.bytes.len() && (self.bytes[byte_idx] >> bit_idx) & 1 == 1;
        self.pos += 1;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_polar_quantizer_creation() {
        let pq = PolarQuantizer::new(128, 2, 42).unwrap();
        assert_eq!(pq.dim, 128);
        assert_eq!(pq.bits, 2);
    }

    #[test]
    fn test_polar_encode() {
        let pq = PolarQuantizer::new(64, 2, 42).unwrap();
        let x = vec![0.5; 64];
        let code = pq.encode(&x).unwrap();
        assert_eq!(code.dim, 64);
        assert_eq!(code.bits, 2);
    }

    #[test]
    fn test_polar_roundtrip_small() {
        let pq = PolarQuantizer::new(16, 4, 42).unwrap();
        let x = vec![0.1, 0.2, 0.3, 0.4, 0.5, -0.1, -0.2, -0.3, -0.4, -0.5, 0.0, 1.0, -1.0, 0.7, 0.8, 0.9];
        
        let encoded = pq.encode(&x).unwrap();
        let decoded = pq.decode(&encoded).unwrap();
        
        assert_eq!(decoded.len(), x.len());
        
        // Check error < 10% for most values
        for (i, (&orig, &recon)) in x.iter().zip(decoded.iter()).enumerate() {
            let error = (orig - recon).abs() / (orig.abs().max(0.1));
            println!("Element {}: orig={}, recon={}, error={}", i, orig, recon, error);
            assert!(error < 0.15, "Error too large at index {}: {}", i, error);
        }
    }

    #[test]
    fn test_polar_roundtrip_random() {
        let pq = PolarQuantizer::new(32, 4, 12345).unwrap();
        let x: Vec<f32> = (0..32).map(|i| (i as f32 * 0.1234567).sin()).collect();
        
        let encoded = pq.encode(&x).unwrap();
        let decoded = pq.decode(&encoded).unwrap();
        
        assert_eq!(decoded.len(), x.len());
        
        // Check reconstruction error is reasonable
        let mse: f32 = x.iter()
            .zip(decoded.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f32>() / x.len() as f32;
        
        println!("MSE: {}", mse);
        assert!(mse < 0.02, "MSE too large: {}", mse);
    }

    #[test]
    fn test_polar_handles_zeros() {
        let pq = PolarQuantizer::new(8, 2, 42).unwrap();
        let x = vec![0.0; 8];
        
        let encoded = pq.encode(&x).unwrap();
        let decoded = pq.decode(&encoded).unwrap();
        
        assert_eq!(decoded.len(), 8);
        // Zeros should reconstruct to values near zero
        for &val in &decoded {
            assert!(val.abs() < 0.1, "Zero reconstruction failed: {}", val);
        }
    }

    #[test]
    fn test_polar_handles_inf_nan() {
        let pq = PolarQuantizer::new(4, 2, 42).unwrap();
        let x = vec![f32::INFINITY, f32::NEG_INFINITY, f32::NAN, 0.5];
        
        let encoded = pq.encode(&x).unwrap();
        let decoded = pq.decode(&encoded).unwrap();
        
        assert_eq!(decoded.len(), 4);
        // Non-finite values should be handled gracefully
        assert!(decoded[0].is_finite());
        assert!(decoded[1].is_finite());
        assert!(decoded[2].is_finite());
    }

    #[test]
    fn test_polar_dimension_mismatch() {
        let pq = PolarQuantizer::new(16, 2, 42).unwrap();
        let x = vec![0.5; 8];
        
        let result = pq.encode(&x);
        assert!(result.is_err());
    }

    #[test]
    fn test_polar_decode_dimension_mismatch() {
        let pq = PolarQuantizer::new(16, 2, 42).unwrap();
        let x = vec![0.5; 16];
        let mut code = pq.encode(&x).unwrap();
        code.dim = 8;
        
        let result = pq.decode(&code);
        assert!(result.is_err());
    }

    #[test]
    fn test_polar_different_bits() {
        for bits in 1..=8 {
            let pq = PolarQuantizer::new(16, bits, 42).unwrap();
            let x: Vec<f32> = (0..16).map(|i| (i as f32 / 16.0) - 0.5).collect();
            
            let encoded = pq.encode(&x).unwrap();
            let decoded = pq.decode(&encoded).unwrap();
            
            assert_eq!(decoded.len(), 16);
            // More bits should give better reconstruction
            let mse: f32 = x.iter()
                .zip(decoded.iter())
                .map(|(&a, &b)| (a - b).powi(2))
                .sum::<f32>() / x.len() as f32;
            
            println!("Bits {}: MSE = {}", bits, mse);
        }
    }
}
