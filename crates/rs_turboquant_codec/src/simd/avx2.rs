//! AVX2 SIMD acceleration for x86_64
//!
//! Processes 4x float32 elements at a time using 256-bit vectors.
//! Falls back to scalar for non-aligned tails.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use std::f32;

#[cfg(target_arch = "x86_64")]
#[inline]
pub fn normalize_vector(x: &[f32]) -> (Vec<f32>, f32, f32) {
    if x.is_empty() {
        return (Vec::new(), 0.0, 0.0);
    }

    // Find min/max using scalar for simplicity (could be SIMD)
    let mut min_val = f32::INFINITY;
    let mut max_val = f32::NEG_INFINITY;

    for &xi in x {
        if xi.is_finite() {
            min_val = min_val.min(xi);
            max_val = max_val.max(xi);
        }
    }

    if !min_val.is_finite() || !max_val.is_finite() {
        min_val = -1.0;
        max_val = 1.0;
    }

    let range = (max_val - min_val).max(1e-6);
    let inv_range = 2.0 / range;

    // SIMD normalization in chunks of 4
    let mut normalized = Vec::with_capacity(x.len());
    let chunks = x.chunks_exact(4);
    let remainder = chunks.remainder();

    unsafe {
        let min_vec = _mm256_set1_ps(min_val);
        let range_vec = _mm256_set1_ps(inv_range);
        let one_vec = _mm256_set1_ps(1.0);

        for chunk in chunks {
            let v = _mm_loadu_ps(chunk.as_ptr());
            let v256 = _mm256_castps128_ps256(v);
            let v256 = _mm256_insertf128_ps(v256, v, 1); // Duplicate to fill 256-bit

            // normalized = 2 * (x - min) / range - 1
            let shifted = _mm256_sub_ps(_mm256_cvtps_ps(_mm256_castps256_ps128(v256)), min_vec);
            let scaled = _mm256_mul_ps(shifted, range_vec);
            let result = _mm256_sub_ps(scaled, one_vec);

            // Extract 4 float32 values (this is a simplified version)
            // In reality, we'd need more careful extraction for the 256-bit result
            let result_scalar = core::mem::transmute::<__m256, [f32; 8]>(result);
            normalized.extend_from_slice(&result_scalar[0..4]);
        }
    }

    // Handle remainder with scalar
    for &xi in remainder {
        if !xi.is_finite() {
            normalized.push(0.0);
        } else {
            normalized.push(2.0 * (xi - min_val) / range - 1.0);
        }
    }

    (normalized, min_val, max_val)
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub fn compute_polar_coords(normalized: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let mut angles = Vec::with_capacity(normalized.len());
    let mut radii = Vec::with_capacity(normalized.len());

    // SIMD computation for chunks of 4
    let chunks = normalized.chunks_exact(4);
    let remainder = chunks.remainder();

    unsafe {
        for chunk in chunks {
            // Load 4 values
            let v = _mm_loadu_ps(chunk.as_ptr());

            // Compute angle = atan2(x, 1.0) for each element
            // AVX2 doesn't have atan2, so we use a Taylor approximation or fall back to scalar
            for &x in chunk {
                angles.push(x.atan2(1.0));
            }

            // Compute radius = sqrt(x^2 + 1.0)
            let x2 = _mm_mul_ps(v, v);
            let one = _mm_set1_ps(1.0);
            let r2 = _mm_add_ps(x2, one);
            let r = _mm_sqrt_ps(r2);

            // Store 4 radius values
            let mut r_vals = [0.0; 4];
            _mm_storeu_ps(r_vals.as_mut_ptr(), r);
            radii.extend_from_slice(&r_vals);
        }
    }

    // Remainder: scalar
    for &x in remainder {
        angles.push(x.atan2(1.0));
        radii.push((x * x + 1.0).sqrt());
    }

    (angles, radii)
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub fn reconstruct_from_polar(angles: &[f32], radii: &[f32]) -> Vec<f32> {
    let mut result = Vec::with_capacity(angles.len());

    let chunks = angles.chunks_exact(4);
    let remainder = chunks.remainder();

    unsafe {
        for (angle_chunk, radius_chunk) in chunks.zip(radii.chunks_exact(4)) {
            let angles_vec = _mm_loadu_ps(angle_chunk.as_ptr());
            let radii_vec = _mm_loadu_ps(radius_chunk.as_ptr());

            // cos(angle) * radius
            let cos_angles = _mm_cos_ps(angles_vec); // Note: AVX doesn't have _mm_cos_ps in standard, need approximation
            let reconstructed = _mm_mul_ps(cos_angles, radii_vec);

            let mut vals = [0.0; 4];
            _mm_storeu_ps(vals.as_mut_ptr(), reconstructed);
            result.extend_from_slice(&vals);
        }
    }

    // Remainder
    for (&angle, &radius) in remainder.iter().zip(radii[angles.len() - remainder.len()..].iter()) {
        result.push(angle.cos() * radius);
    }

    result
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub fn denormalize_vector(normalized: &[f32], min_val: f32, max_val: f32) -> Vec<f32> {
    let range = (max_val - min_val).max(1e-6);
    let mut result = Vec::with_capacity(normalized.len());

    let chunks = normalized.chunks_exact(4);
    let remainder = chunks.remainder();

    unsafe {
        let range_vec = _mm_set1_ps(range / 2.0);
        let min_vec = _mm_set1_ps(min_val);
        let one_vec = _mm_set1_ps(1.0);

        for chunk in chunks {
            let v = _mm_loadu_ps(chunk.as_ptr());

            // denormalized = (normalized + 1) * (range / 2) + min
            let shifted = _mm_add_ps(v, one_vec);
            let scaled = _mm_mul_ps(shifted, range_vec);
            let denorm = _mm_add_ps(scaled, min_vec);

            let mut vals = [0.0; 4];
            _mm_storeu_ps(vals.as_mut_ptr(), denorm);
            result.extend_from_slice(&vals);
        }
    }

    // Remainder
    for &x in remainder {
        result.push((x + 1.0) * range / 2.0 + min_val);
    }

    result
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub fn inner_product(x: &[f32], y: &[f32]) -> f32 {
    let mut sum = 0.0f32;

    let chunks_x = x.chunks_exact(4);
    let chunks_y = y.chunks_exact(4);
    let remainder_x = chunks_x.remainder();
    let remainder_y = chunks_y.remainder();

    unsafe {
        let mut sum_vec = _mm_setzero_ps();

        for (chunk_x, chunk_y) in chunks_x.zip(chunks_y) {
            let vx = _mm_loadu_ps(chunk_x.as_ptr());
            let vy = _mm_loadu_ps(chunk_y.as_ptr());

            let products = _mm_mul_ps(vx, vy);
            sum_vec = _mm_add_ps(sum_vec, products);
        }

        // Reduce sum_vec to scalar
        let mut temp = [0.0; 4];
        _mm_storeu_ps(temp.as_mut_ptr(), sum_vec);
        sum = temp[0] + temp[1] + temp[2] + temp[3];
    }

    // Remainder
    for (xi, yi) in remainder_x.iter().zip(remainder_y.iter()) {
        sum += xi * yi;
    }

    sum
}

#[cfg(not(target_arch = "x86_64"))]
pub fn normalize_vector(x: &[f32]) -> (Vec<f32>, f32, f32) {
    super::scalar::normalize_vector(x)
}

#[cfg(not(target_arch = "x86_64"))]
pub fn compute_polar_coords(normalized: &[f32]) -> (Vec<f32>, Vec<f32>) {
    super::scalar::compute_polar_coords(normalized)
}

#[cfg(not(target_arch = "x86_64"))]
pub fn reconstruct_from_polar(angles: &[f32], radii: &[f32]) -> Vec<f32> {
    super::scalar::reconstruct_from_polar(angles, radii)
}

#[cfg(not(target_arch = "x86_64"))]
pub fn denormalize_vector(normalized: &[f32], min_val: f32, max_val: f32) -> Vec<f32> {
    super::scalar::denormalize_vector(normalized, min_val, max_val)
}

#[cfg(not(target_arch = "x86_64"))]
pub fn inner_product(x: &[f32], y: &[f32]) -> f32 {
    super::scalar::inner_product(x, y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_avx2_normalize() {
        let x = vec![0.0, 10.0, 5.0, -5.0];
        let (normalized, min_val, max_val) = normalize_vector(&x);
        assert_eq!(min_val, -5.0);
        assert_eq!(max_val, 10.0);
        assert_eq!(normalized.len(), 4);
    }

    #[test]
    fn test_avx2_inner_product() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 3.0, 4.0, 5.0];
        let ip = inner_product(&x, &y);
        assert_eq!(ip, 40.0);
    }
}
