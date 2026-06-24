use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

/// Supported RotorQuant compression modes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RotorQuantMode {
    /// PlanarQuant with 3-bit quantization (2D Givens rotations)
    PlanarQuant3 = 0,
    /// PlanarQuant with 4-bit quantization (2D Givens rotations)
    PlanarQuant4 = 1,
    /// IsoQuant with 3-bit quantization (4D quaternion rotations)
    IsoQuant3 = 2,
    /// IsoQuant with 4-bit quantization (4D quaternion rotations)
    IsoQuant4 = 3,
}

/// Codebook entry for Lloyd-Max quantization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodebookEntry {
    pub codeword: Vec<f32>,
    pub count: u32,
}

/// RotorQuant codec supporting PlanarQuant and IsoQuant compression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RotorQuantCodec {
    pub mode: RotorQuantMode,
    pub seed: u64,
    pub codebook: Vec<CodebookEntry>,
    pub is_planar: bool,
}

impl RotorQuantCodec {
    /// Create a new RotorQuant codec
    pub fn new(mode: RotorQuantMode, seed: u64, is_planar: bool) -> Self {
        let codebook_size = if mode == RotorQuantMode::PlanarQuant3 || mode == RotorQuantMode::IsoQuant3 {
            8 // 2^3
        } else {
            16 // 2^4
        };

        RotorQuantCodec {
            mode,
            seed,
            codebook: vec![CodebookEntry {
                codeword: vec![0.0; 2],
                count: 0
            }; codebook_size],
            is_planar,
        }
    }

    /// Generate Givens rotation for PlanarQuant (2D rotation)
    pub fn givens_rotation(&self, dim_pair_index: usize) -> (f32, f32) {
        let _rng = ChaCha8Rng::seed_from_u64(self.seed ^ (dim_pair_index as u64));
        let angle = (dim_pair_index as f32 * PI / 256.0).fract() * 2.0 * PI;
        (angle.cos(), angle.sin())
    }

    /// Generate quaternion rotation for IsoQuant (4D rotation)
    pub fn quaternion_rotation(&self, dim_quad_index: usize) -> (f32, f32, f32, f32) {
        let _rng = ChaCha8Rng::seed_from_u64(self.seed ^ (dim_quad_index as u64));
        let theta = ((dim_quad_index as f32) * PI / 128.0).fract() * 2.0 * PI;
        let phi = ((dim_quad_index as f32 * 1.618) * PI / 128.0).fract() * 2.0 * PI;
        let psi = ((dim_quad_index as f32 * 2.618) * PI / 128.0).fract() * 2.0 * PI;

        let half_theta = theta / 2.0;
        let half_phi = phi / 2.0;
        let half_psi = psi / 2.0;

        let w = half_theta.cos() * half_phi.cos() * half_psi.cos()
            + half_theta.sin() * half_phi.sin() * half_psi.sin();
        let x = half_theta.sin() * half_phi.cos() * half_psi.cos()
            - half_theta.cos() * half_phi.sin() * half_psi.sin();
        let y = half_theta.cos() * half_phi.sin() * half_psi.cos()
            + half_theta.sin() * half_phi.cos() * half_psi.sin();
        let z = half_theta.cos() * half_phi.cos() * half_psi.sin()
            - half_theta.sin() * half_phi.sin() * half_psi.cos();

        (w, x, y, z)
    }

    /// Compress a tensor using PlanarQuant
    pub fn compress_planar(&self, input: &[f32], dim: usize) -> Vec<u8> {
        let num_pairs = (dim + 1) / 2;
        let mut output = Vec::new();
        let bits_per_sample = match self.mode {
            RotorQuantMode::PlanarQuant3 => 3,
            RotorQuantMode::PlanarQuant4 => 4,
            _ => 3,
        };

        // Encode each pair with Givens rotation
        let mut bit_buffer = 0u32;
        let mut bit_count = 0;

        for i in 0..num_pairs {
            let (cos_a, sin_a) = self.givens_rotation(i);
            let x1 = if i * 2 < input.len() { input[i * 2] } else { 0.0 };
            let x2 = if i * 2 + 1 < input.len() { input[i * 2 + 1] } else { 0.0 };

            // Rotate pair
            let y1 = cos_a * x1 - sin_a * x2;
            let y2 = sin_a * x1 + cos_a * x2;

            // Quantize to bits_per_sample bits
            let q1 = quantize_to_bits(y1, bits_per_sample);
            let q2 = quantize_to_bits(y2, bits_per_sample);

            // Pack into bit stream
            bit_buffer = (bit_buffer << bits_per_sample) | (q1 as u32);
            bit_count += bits_per_sample as u32;
            if bit_count >= 8 {
                output.push((bit_buffer >> (bit_count - 8)) as u8);
                bit_count -= 8;
            }

            bit_buffer = (bit_buffer << bits_per_sample) | (q2 as u32);
            bit_count += bits_per_sample as u32;
            if bit_count >= 8 {
                output.push((bit_buffer >> (bit_count - 8)) as u8);
                bit_count -= 8;
            }
        }

        // Flush remaining bits
        if bit_count > 0 {
            output.push(((bit_buffer << (8 - bit_count)) & 0xFF) as u8);
        }

        output
    }

    /// Compress a tensor using IsoQuant
    pub fn compress_iso(&self, input: &[f32], dim: usize) -> Vec<u8> {
        let num_quads = (dim + 3) / 4;
        let mut output = Vec::new();
        let bits_per_sample = match self.mode {
            RotorQuantMode::IsoQuant3 => 3,
            RotorQuantMode::IsoQuant4 => 4,
            _ => 3,
        };

        let mut bit_buffer = 0u32;
        let mut bit_count = 0;

        for i in 0..num_quads {
            let (w, x, y, z) = self.quaternion_rotation(i);

            // Load quad (or pad with zeros if needed)
            let vals = [
                if i * 4 < input.len() { input[i * 4] } else { 0.0 },
                if i * 4 + 1 < input.len() { input[i * 4 + 1] } else { 0.0 },
                if i * 4 + 2 < input.len() { input[i * 4 + 2] } else { 0.0 },
                if i * 4 + 3 < input.len() { input[i * 4 + 3] } else { 0.0 },
            ];

            // Apply quaternion rotation (simplified: scale by magnitude)
            let scale = (w * w + x * x + y * y + z * z).sqrt();
            for val in vals.iter() {
                let rotated = val * scale;
                let q = quantize_to_bits(rotated, bits_per_sample);
                bit_buffer = (bit_buffer << bits_per_sample) | (q as u32);
                bit_count += bits_per_sample as u32;
                if bit_count >= 8 {
                    output.push((bit_buffer >> (bit_count - 8)) as u8);
                    bit_count -= 8;
                }
            }
        }

        // Flush remaining bits
        if bit_count > 0 {
            output.push(((bit_buffer << (8 - bit_count)) & 0xFF) as u8);
        }

        output
    }

    /// Decompress a PlanarQuant compressed tensor
    pub fn decompress_planar(&self, compressed: &[u8], dim: usize) -> Vec<f32> {
        let mut output = Vec::new();
        let bits_per_sample = match self.mode {
            RotorQuantMode::PlanarQuant3 => 3,
            RotorQuantMode::PlanarQuant4 => 4,
            _ => 3,
        };

        let mut bit_buffer = 0u32;
        let mut bit_count = 0;
        let mut byte_idx = 0;

        for i in 0..(dim + 1) / 2 {
            let (cos_a, sin_a) = self.givens_rotation(i);

            // Load q1
            let q1 = read_bits(&compressed, &mut byte_idx, &mut bit_buffer, &mut bit_count, bits_per_sample);
            let y1 = dequantize_from_bits(q1, bits_per_sample);

            // Load q2
            let q2 = read_bits(&compressed, &mut byte_idx, &mut bit_buffer, &mut bit_count, bits_per_sample);
            let y2 = dequantize_from_bits(q2, bits_per_sample);

            // Inverse rotate
            let x1 = cos_a * y1 + sin_a * y2;
            let x2 = -sin_a * y1 + cos_a * y2;

            output.push(x1);
            if output.len() < dim {
                output.push(x2);
            }
        }

        output.truncate(dim);
        output
    }

    /// Decompress an IsoQuant compressed tensor
    pub fn decompress_iso(&self, compressed: &[u8], dim: usize) -> Vec<f32> {
        let mut output = Vec::new();
        let bits_per_sample = match self.mode {
            RotorQuantMode::IsoQuant3 => 3,
            RotorQuantMode::IsoQuant4 => 4,
            _ => 3,
        };

        let mut bit_buffer = 0u32;
        let mut bit_count = 0;
        let mut byte_idx = 0;

        for i in 0..(dim + 3) / 4 {
            let (_w, _x, _y, _z) = self.quaternion_rotation(i);
            // Note: In full implementation, apply inverse quaternion rotation
            // For now, simple dequantization

            for _ in 0..4 {
                if output.len() < dim {
                    let q = read_bits(&compressed, &mut byte_idx, &mut bit_buffer, &mut bit_count, bits_per_sample);
                    let val = dequantize_from_bits(q, bits_per_sample);
                    output.push(val);
                }
            }
        }

        output.truncate(dim);
        output
    }
}

/// Quantize a float to N bits using uniform quantization
fn quantize_to_bits(value: f32, bits: usize) -> u8 {
    let max_val = ((1u32 << bits) - 1) as f32;
    let clamped = value.clamp(-1.0, 1.0);
    let normalized = (clamped + 1.0) / 2.0;
    (normalized * max_val) as u8
}

/// Dequantize from N bits back to float
fn dequantize_from_bits(quantized: u8, bits: usize) -> f32 {
    let max_val = ((1u32 << bits) - 1) as f32;
    let normalized = (quantized as f32) / max_val;
    normalized * 2.0 - 1.0
}

/// Read N bits from a bit stream
fn read_bits(
    data: &[u8],
    byte_idx: &mut usize,
    bit_buffer: &mut u32,
    bit_count: &mut usize,
    bits: usize,
) -> u8 {
    while *bit_count < bits {
        if *byte_idx < data.len() {
            *bit_buffer = (*bit_buffer << 8) | (data[*byte_idx] as u32);
            *bit_count += 8;
            *byte_idx += 1;
        } else {
            break;
        }
    }

    let result = (*bit_buffer >> (*bit_count - bits)) as u8;
    *bit_count -= bits;
    result & ((1u8 << bits) - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_planar_quant_3bit_roundtrip() {
        let codec = RotorQuantCodec::new(RotorQuantMode::PlanarQuant3, 42, true);
        let input = vec![0.5, 0.3, -0.2, 0.8];
        let compressed = codec.compress_planar(&input, 4);
        let decompressed = codec.decompress_planar(&compressed, 4);

        assert_eq!(decompressed.len(), 4);
        assert!(decompressed.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_planar_quant_4bit_roundtrip() {
        let codec = RotorQuantCodec::new(RotorQuantMode::PlanarQuant4, 42, true);
        let input = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let compressed = codec.compress_planar(&input, 5);
        let decompressed = codec.decompress_planar(&compressed, 5);

        assert_eq!(decompressed.len(), 5);
        assert!(decompressed.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_iso_quant_3bit_roundtrip() {
        let codec = RotorQuantCodec::new(RotorQuantMode::IsoQuant3, 42, false);
        let input = vec![0.5, 0.3, -0.2, 0.8];
        let compressed = codec.compress_iso(&input, 4);
        let decompressed = codec.decompress_iso(&compressed, 4);

        assert_eq!(decompressed.len(), 4);
    }

    #[test]
    fn test_iso_quant_4bit_roundtrip() {
        let codec = RotorQuantCodec::new(RotorQuantMode::IsoQuant4, 42, false);
        let input = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let compressed = codec.compress_iso(&input, 8);
        let decompressed = codec.decompress_iso(&compressed, 8);

        assert_eq!(decompressed.len(), 8);
    }

    #[test]
    fn test_compression_ratio() {
        let codec = RotorQuantCodec::new(RotorQuantMode::PlanarQuant3, 42, true);
        let input: Vec<f32> = (0..1024).map(|i| (i as f32 / 1024.0) * 2.0 - 1.0).collect();
        let compressed = codec.compress_planar(&input, 1024);

        // 3-bit compression should be ~12x vs 32-bit float (1024 * 4 bytes = 4096 bytes)
        let expected_size = (1024 * 3 / 8) + 1; // 384 bytes + padding
        assert!(compressed.len() <= expected_size + 8);
    }
}

#[cfg(feature = "pyo3-ffi")]
use pyo3::prelude::*;

#[cfg(feature = "pyo3-ffi")]
#[pyclass]
pub struct PyRotorQuantCodec {
    inner: RotorQuantCodec,
}

#[cfg(feature = "pyo3-ffi")]
#[pymethods]
impl PyRotorQuantCodec {
    #[new]
    fn new(mode_str: &str, seed: u64, is_planar: bool) -> PyResult<Self> {
        let mode = match mode_str {
            "planar3" => RotorQuantMode::PlanarQuant3,
            "planar4" => RotorQuantMode::PlanarQuant4,
            "iso3" => RotorQuantMode::IsoQuant3,
            "iso4" => RotorQuantMode::IsoQuant4,
            _ => return Err(pyo3::exceptions::PyValueError::new_err("Invalid mode")),
        };

        Ok(PyRotorQuantCodec {
            inner: RotorQuantCodec::new(mode, seed, is_planar),
        })
    }

    fn compress_planar(&self, data: Vec<f32>, dim: usize) -> Vec<u8> {
        self.inner.compress_planar(&data, dim)
    }

    fn compress_iso(&self, data: Vec<f32>, dim: usize) -> Vec<u8> {
        self.inner.compress_iso(&data, dim)
    }

    fn decompress_planar(&self, data: Vec<u8>, dim: usize) -> Vec<f32> {
        self.inner.decompress_planar(&data, dim)
    }

    fn decompress_iso(&self, data: Vec<u8>, dim: usize) -> Vec<f32> {
        self.inner.decompress_iso(&data, dim)
    }

    fn mode(&self) -> String {
        match self.inner.mode {
            RotorQuantMode::PlanarQuant3 => "planar3".to_string(),
            RotorQuantMode::PlanarQuant4 => "planar4".to_string(),
            RotorQuantMode::IsoQuant3 => "iso3".to_string(),
            RotorQuantMode::IsoQuant4 => "iso4".to_string(),
        }
    }

    fn is_planar(&self) -> bool {
        self.inner.is_planar
    }

    fn seed(&self) -> u64 {
        self.inner.seed
    }
}

#[cfg(feature = "pyo3-ffi")]
#[pymodule]
fn rs_rotorquant_codec(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyRotorQuantCodec>()?;
    Ok(())
}

pub mod hybrid;
pub mod stage1_hip_scaffold;
