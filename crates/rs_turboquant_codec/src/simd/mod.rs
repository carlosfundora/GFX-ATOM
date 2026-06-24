//! SIMD acceleration layer for TurboQuantizer
//!
//! Provides platform-specific and fallback implementations for:
//! - Polar coordinate transformation (sin, cos, sqrt)
//! - Bit packing/unpacking
//! - Vector operations (normalize, inner product)
//!
//! # Dispatch Strategy
//!
//! ```text
//! Runtime CPU detection
//!   ├─ AVX2 (x86_64): 4x float32 at a time
//!   ├─ NEON (ARM64): 4x float32 at a time
//!   └─ Scalar (all): 1x float32 per iteration (portable)
//! ```

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
pub mod avx2;

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub mod neon;

pub mod scalar;
///
/// Normalizes a vector to [-1, 1] range by min-max scaling.
/// Returns (normalized_vec, min_val, max_val)
#[inline]
pub fn normalize_vector(x: &[f32]) -> (Vec<f32>, f32, f32) {
    scalar::normalize_vector(x)
}

/// SIMD-accelerated angle/radius computation
///
/// Given normalized vector in [-1, 1], compute polar coordinates:
/// - angle = atan2(x, 1.0)
/// - radius = sqrt(x^2 + 1.0)
pub fn compute_polar_coords(normalized: &[f32]) -> (Vec<f32>, Vec<f32>) {
    scalar::compute_polar_coords(normalized)
}

/// SIMD-accelerated polar reconstruction
///
/// Given angle and radius in [-1, 1], reconstruct original:
/// reconstructed = angle.cos() * radius_norm
pub fn reconstruct_from_polar(angles: &[f32], radii: &[f32]) -> Vec<f32> {
    scalar::reconstruct_from_polar(angles, radii)
}

/// SIMD-accelerated denormalization
///
/// Reverses normalization: denormalized = normalized * (max - min) + min
pub fn denormalize_vector(normalized: &[f32], min_val: f32, max_val: f32) -> Vec<f32> {
    scalar::denormalize_vector(normalized, min_val, max_val)
}

/// SIMD-accelerated inner product
///
/// Computes sum of element-wise products
#[inline]
pub fn inner_product(x: &[f32], y: &[f32]) -> f32 {
    scalar::inner_product(x, y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_denormalize_roundtrip() {
        let x = vec![0.1, 0.5, -0.3, 0.8, -0.2];
        let (normalized, min_val, max_val) = normalize_vector(&x);
        let x_restored = denormalize_vector(&normalized, min_val, max_val);

        for (xi, xi_restored) in x.iter().zip(x_restored.iter()) {
            assert!((xi - xi_restored).abs() < 1e-5);
        }
    }

    #[test]
    fn test_inner_product() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 3.0, 4.0, 5.0];
        let ip = inner_product(&x, &y);
        assert_eq!(ip, 40.0);
    }
}
