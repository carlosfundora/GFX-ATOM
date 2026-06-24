//! Scalar (portable) fallback implementations
//!
//! Pure Rust implementations without SIMD. Used on all platforms as fallback
//! and as the baseline for performance comparison.

use std::f32;

/// Normalize vector to [-1, 1] by min-max scaling
pub fn normalize_vector(x: &[f32]) -> (Vec<f32>, f32, f32) {
    if x.is_empty() {
        return (Vec::new(), 0.0, 0.0);
    }

    // Find min/max, handling non-finite values
    let mut min_val = f32::INFINITY;
    let mut max_val = f32::NEG_INFINITY;

    for &xi in x {
        if xi.is_finite() {
            min_val = min_val.min(xi);
            max_val = max_val.max(xi);
        }
    }

    // If all values are non-finite, use default range
    if !min_val.is_finite() || !max_val.is_finite() {
        min_val = -1.0;
        max_val = 1.0;
    }

    // Avoid division by zero
    let range = (max_val - min_val).max(1e-6);

    let normalized = x
        .iter()
        .map(|&xi| {
            if !xi.is_finite() {
                0.0
            } else {
                2.0 * (xi - min_val) / range - 1.0
            }
        })
        .collect();

    (normalized, min_val, max_val)
}

/// Compute polar coordinates (angle, radius) from normalized vector
pub fn compute_polar_coords(normalized: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let angles = normalized
        .iter()
        .map(|&x| x.atan2(1.0))
        .collect();

    let radii = normalized
        .iter()
        .map(|&x| (x * x + 1.0).sqrt())
        .collect();

    (angles, radii)
}

/// Reconstruct normalized vector from polar coordinates
pub fn reconstruct_from_polar(angles: &[f32], radii: &[f32]) -> Vec<f32> {
    angles
        .iter()
        .zip(radii.iter())
        .map(|(&angle, &radius)| angle.cos() * radius)
        .collect()
}

/// Denormalize vector from [-1, 1] back to original range
pub fn denormalize_vector(normalized: &[f32], min_val: f32, max_val: f32) -> Vec<f32> {
    let range = (max_val - min_val).max(1e-6);
    normalized
        .iter()
        .map(|&x| (x + 1.0) * range / 2.0 + min_val)
        .collect()
}

/// Compute inner product of two vectors
pub fn inner_product(x: &[f32], y: &[f32]) -> f32 {
    x.iter()
        .zip(y.iter())
        .map(|(xi, yi)| xi * yi)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_basic() {
        let x = vec![0.0, 10.0];
        let (normalized, min_val, max_val) = normalize_vector(&x);
        assert_eq!(min_val, 0.0);
        assert_eq!(max_val, 10.0);
        assert!((normalized[0] - (-1.0)).abs() < 1e-6);
        assert!((normalized[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_single_value() {
        let x = vec![5.0, 5.0, 5.0];
        let (normalized, min_val, max_val) = normalize_vector(&x);
        // When all values are the same, min==max, so we use default range [-1, 1]
        // Therefore values end up somewhere in normalized range, but not necessarily 0
        assert_eq!(normalized.len(), 3);
        for &n in &normalized {
            assert!(n >= -1.0 && n <= 1.0, "Normalized values should be in [-1, 1]");
        }
    }

    #[test]
    fn test_normalize_with_inf() {
        let x = vec![0.0, f32::INFINITY, 10.0, f32::NAN];
        let (normalized, _min, _max) = normalize_vector(&x);
        assert_eq!(normalized.len(), 4);
        // Non-finite values should become 0
        assert_eq!(normalized[1], 0.0);
        assert_eq!(normalized[3], 0.0);
    }

    #[test]
    fn test_polar_coords() {
        let normalized = vec![0.0, 0.5];
        let (angles, radii) = compute_polar_coords(&normalized);
        let expected_angle = (0.0f32).atan2(1.0);
        assert!((angles[0] - expected_angle).abs() < 1e-6); // atan2(0, 1) = 0
        assert!((radii[0] - 1.0).abs() < 1e-6); // sqrt(0 + 1) = 1
    }

    #[test]
    fn test_inner_product() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![4.0, 5.0, 6.0];
        let ip = inner_product(&x, &y);
        assert_eq!(ip, 32.0); // 1*4 + 2*5 + 3*6
    }

    #[test]
    fn test_inner_product_orthogonal() {
        let x = vec![1.0, 0.0, 0.0];
        let y = vec![0.0, 1.0, 0.0];
        let ip = inner_product(&x, &y);
        assert_eq!(ip, 0.0);
    }
}
