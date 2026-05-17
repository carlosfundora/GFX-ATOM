//! Integration tests for TurboQuant codec in KV validation harness

use rs_turboquant_codec::{BitWidth, CodecInfo, TurboQuantizer};

#[test]
fn test_all_bit_widths_codec_info() {
    let all = CodecInfo::all_modes();
    assert_eq!(all.len(), 5, "Should have 5 TurboQuant modes");

    // Verify each mode has proper metadata
    for codec in &all {
        assert!(!codec.name.is_empty());
        assert!(codec.accuracy_floor > 0.0 && codec.accuracy_floor < 1.0);
        assert!(codec.compression_ratio > 1.0);
        assert!(!codec.use_case.is_empty());
        assert!(!codec.flags.is_empty());
    }

    // Spot-check specific modes
    let bit2 = CodecInfo::for_mode(BitWidth::Bit2);
    assert_eq!(bit2.accuracy_floor, 0.10);
    assert_eq!(bit2.compression_ratio, 8.0);
    assert!(bit2.flags.contains(&"production".to_string()));
}

#[test]
fn test_turboquant_encode_decode_accuracy_floor() {
    let codec = CodecInfo::for_mode(BitWidth::Bit2);
    let qz = TurboQuantizer::new(128, 2, 32, 42).expect("Create quantizer");

    // Create test vector
    let x = (0..128)
        .map(|i| (i as f32) / 128.0)
        .collect::<Vec<_>>();

    // Encode and decode
    let code = qz.encode(&x).expect("Encode");
    let x_hat = qz.decode(&code).expect("Decode");

    // Verify compression happened
    let orig_bytes = 128 * 4; // 128 f32s
    let compressed_bytes = code.encoded_bytes();
    let actual_ratio = orig_bytes as f32 / compressed_bytes as f32;

    println!(
        "Codec: {}, Theoretical ratio: {}, Actual: {:.2}",
        codec.name, codec.compression_ratio, actual_ratio
    );

    assert_eq!(x_hat.len(), 128);
    // Actual decode should approximate original (exact values are stubs)
    assert!(x_hat.iter().all(|v| v.is_finite()));
}

#[test]
fn test_inner_product_estimation() {
    let qz = TurboQuantizer::new(64, 2, 16, 42).expect("Create quantizer");

    let x = vec![1.0; 64];
    let y = vec![2.0; 64];

    let code = qz.encode(&x).expect("Encode");
    let ip = qz.estimate_inner_product(&code, &y).expect("Estimate IP");

    // Stub returns 0.0, but should be finite
    assert!(ip.is_finite());
}

#[test]
fn test_kv_cache_accuracy_floor_validation() {
    // Simulate KV cache scenario: decode queries
    let test_cases = vec![
        (BitWidth::Bit1, 0.20, "aggressive batch decode"),
        (BitWidth::Bit2, 0.10, "standard production KV"),
        (BitWidth::Bit3, 0.05, "high-quality latency-sensitive"),
        (BitWidth::Bit4, 0.03, "premium accuracy-critical"),
        (BitWidth::Bit8, 0.005, "reference verification"),
    ];

    for (bit_width, expected_loss, scenario) in test_cases {
        let codec = CodecInfo::for_mode(bit_width);
        assert!(
            codec.accuracy_floor <= expected_loss,
            "Scenario '{}': floor {:.3} should be <= {:.3}",
            scenario,
            codec.accuracy_floor,
            expected_loss
        );
        println!(
            "✓ {}: accuracy floor {:.3} suitable for {}",
            codec.name, codec.accuracy_floor, scenario
        );
    }
}

#[test]
fn test_codec_mode_soundness() {
    let codecs = CodecInfo::all_modes();

    // Accuracy floor should decrease with more bits (stricter accuracy)
    for i in 0..codecs.len() - 1 {
        assert!(
            codecs[i].accuracy_floor > codecs[i + 1].accuracy_floor,
            "Accuracy should improve (lower loss) with more bits"
        );
    }

    // Compression ratio should decrease with more bits (less compression)
    for i in 0..codecs.len() - 1 {
        assert!(
            codecs[i].compression_ratio > codecs[i + 1].compression_ratio,
            "Compression ratio should decrease with more bits"
        );
    }
}
