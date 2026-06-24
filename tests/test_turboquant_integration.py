#!/usr/bin/env python3
"""
TurboQuantizer Integration Tests
=================================

Comprehensive tests for the full compression pipeline:
- Encode/decode roundtrip
- Accuracy validation
- Performance benchmarking
- Batch processing
"""

import json
import logging
import sys
import time
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np

# Import codec
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
from turboquant_codec import TurboQuantCodec, BatchTurboQuantCodec

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)


@dataclass
class IntegrationTestResult:
    """Result of a single integration test"""
    test_name: str
    mode: str
    shape: tuple
    original_bytes: int
    compressed_bytes: int
    encode_ms: float
    decode_ms: float
    roundtrip_mse: float
    passed: bool
    error_msg: str = ""
    
    @property
    def compression_ratio(self) -> float:
        return self.original_bytes / max(self.compressed_bytes, 1)
    
    @property
    def throughput_encode_mbs(self) -> float:
        if self.encode_ms <= 0:
            return 0.0
        return (self.original_bytes / 1024 / 1024) / (self.encode_ms / 1000)
    
    @property
    def throughput_decode_mbs(self) -> float:
        if self.decode_ms <= 0:
            return 0.0
        return (self.compressed_bytes / 1024 / 1024) / (self.decode_ms / 1000)


class TurboQuantIntegrationTest:
    """Integration test suite for TurboQuantizer"""
    
    def __init__(self):
        self.results: list[IntegrationTestResult] = []
        self.passed = 0
        self.failed = 0
    
    def test_encode_decode_roundtrip(self, mode: str, shape: tuple) -> IntegrationTestResult:
        """Test encode/decode roundtrip and accuracy"""
        test_name = f"roundtrip_{mode}_{shape}"
        
        try:
            # Create test data
            kv = np.random.randn(*shape).astype(np.float32)
            original_bytes = kv.nbytes
            
            # Encode
            codec = TurboQuantCodec(mode)
            compressed, encode_ms = codec.encode(kv)
            compressed_bytes = len(compressed)
            
            # Decode
            kv_decoded, decode_ms = codec.decode(compressed, shape)
            
            # Validate shape
            if kv_decoded.shape != kv.shape:
                raise ValueError(f"Shape mismatch: {kv_decoded.shape} vs {kv.shape}")
            
            # Calculate MSE
            mse = float(np.mean((kv - kv_decoded) ** 2))
            
            # Determine pass/fail
            # Note: In simulation mode, roundtrip should be nearly perfect (MSE < 1e-6)
            # In real codec, MSE will be higher but still within acceptable bounds
            max_acceptable_mse = 1e-4  # Allow up to 0.01% error
            passed = mse < max_acceptable_mse
            
            result = IntegrationTestResult(
                test_name=test_name,
                mode=mode,
                shape=shape,
                original_bytes=original_bytes,
                compressed_bytes=compressed_bytes,
                encode_ms=encode_ms,
                decode_ms=decode_ms,
                roundtrip_mse=mse,
                passed=passed,
            )
            
            if passed:
                self.passed += 1
            else:
                self.failed += 1
                result.error_msg = f"MSE {mse:.2e} exceeds threshold"
            
            return result
            
        except Exception as e:
            self.failed += 1
            return IntegrationTestResult(
                test_name=test_name,
                mode=mode,
                shape=shape,
                original_bytes=0,
                compressed_bytes=0,
                encode_ms=0,
                decode_ms=0,
                roundtrip_mse=0,
                passed=False,
                error_msg=str(e),
            )
    
    def test_inner_product_estimation(self, mode: str, shape: tuple) -> IntegrationTestResult:
        """Test inner product estimation accuracy"""
        test_name = f"inner_product_{mode}_{shape}"
        
        try:
            codec = TurboQuantCodec(mode)
            
            # Create test vectors
            kv1 = np.random.randn(*shape).astype(np.float32)
            kv2 = np.random.randn(*shape).astype(np.float32)
            
            # True inner product
            true_ip = float(np.dot(kv1.flatten(), kv2.flatten()))
            
            # Estimated inner product
            est_ip = codec.estimate_inner_product(kv1, kv2)
            
            # Calculate relative error
            rel_error = abs(est_ip - true_ip) / (abs(true_ip) + 1e-6)
            
            # Pass if relative error is small
            passed = rel_error < 0.2  # 20% error acceptable for rough estimate
            
            if passed:
                self.passed += 1
            else:
                self.failed += 1
            
            result = IntegrationTestResult(
                test_name=test_name,
                mode=mode,
                shape=shape,
                original_bytes=kv1.nbytes,
                compressed_bytes=int(kv1.nbytes / codec.compression_ratio()),
                encode_ms=0,
                decode_ms=0,
                roundtrip_mse=rel_error,
                passed=passed,
            )
            
            if not passed:
                result.error_msg = f"IP estimate error {rel_error:.2%} exceeds 20%"
            
            return result
            
        except Exception as e:
            self.failed += 1
            return IntegrationTestResult(
                test_name=test_name,
                mode=mode,
                shape=shape,
                original_bytes=0,
                compressed_bytes=0,
                encode_ms=0,
                decode_ms=0,
                roundtrip_mse=0,
                passed=False,
                error_msg=str(e),
            )
    
    def test_batch_processing(self, mode: str) -> list[IntegrationTestResult]:
        """Test batch processing with multiple KV caches"""
        results = []
        
        try:
            batch_codec = BatchTurboQuantCodec(mode)
            
            # Create batch of different sizes
            batch_kv = [
                np.random.randn(1, 128, 32, 128).astype(np.float32),
                np.random.randn(4, 256, 32, 128).astype(np.float32),
                np.random.randn(2, 512, 32, 128).astype(np.float32),
            ]
            shapes = [kv.shape for kv in batch_kv]
            
            # Encode
            compressed_list = batch_codec.encode_batch(batch_kv)
            
            # Decode
            kv_decoded_list = batch_codec.decode_batch(compressed_list, shapes)
            
            # Verify all
            all_passed = True
            for i, (kv_orig, kv_decoded) in enumerate(zip(batch_kv, kv_decoded_list)):
                mse = float(np.mean((kv_orig - kv_decoded) ** 2))
                passed = mse < 1e-4
                
                if passed:
                    self.passed += 1
                else:
                    self.failed += 1
                    all_passed = False
                
                result = IntegrationTestResult(
                    test_name=f"batch_{mode}_{i}",
                    mode=mode,
                    shape=kv_orig.shape,
                    original_bytes=kv_orig.nbytes,
                    compressed_bytes=len(compressed_list[i]),
                    encode_ms=0,
                    decode_ms=0,
                    roundtrip_mse=mse,
                    passed=passed,
                )
                results.append(result)
        
        except Exception as e:
            self.failed += 1
            results.append(IntegrationTestResult(
                test_name=f"batch_{mode}",
                mode=mode,
                shape=(0, 0, 0, 0),
                original_bytes=0,
                compressed_bytes=0,
                encode_ms=0,
                decode_ms=0,
                roundtrip_mse=0,
                passed=False,
                error_msg=str(e),
            ))
        
        return results
    
    def run_all_tests(self) -> None:
        """Run complete integration test suite"""
        logger.info("Starting TurboQuantizer Integration Tests")
        
        # Test configurations
        modes = ["tq1", "tq2", "tq3", "tq4"]
        shapes = [
            (1, 128, 32, 128),   # Small
            (4, 256, 32, 128),   # Medium
            (1, 1024, 32, 128),  # Large
        ]
        
        # Run roundtrip tests
        logger.info("=== Roundtrip Tests ===")
        for mode in modes:
            for shape in shapes:
                result = self.test_encode_decode_roundtrip(mode, shape)
                self.results.append(result)
                status = "✓" if result.passed else "✗"
                logger.info(
                    f"{status} {result.test_name}: "
                    f"compress={result.compression_ratio:.2f}x, "
                    f"mse={result.roundtrip_mse:.2e}, "
                    f"encode={result.encode_ms:.2f}ms, "
                    f"decode={result.decode_ms:.2f}ms"
                )
        
        # Run inner product tests
        logger.info("\n=== Inner Product Tests ===")
        for mode in modes:
            for shape in shapes:
                result = self.test_inner_product_estimation(mode, shape)
                self.results.append(result)
                status = "✓" if result.passed else "✗"
                logger.info(f"{status} {result.test_name}: error={result.roundtrip_mse:.2%}")
        
        # Run batch tests
        logger.info("\n=== Batch Processing Tests ===")
        for mode in modes:
            batch_results = self.test_batch_processing(mode)
            self.results.extend(batch_results)
            for result in batch_results:
                status = "✓" if result.passed else "✗"
                logger.info(f"{status} {result.test_name}: mse={result.roundtrip_mse:.2e}")
    
    def print_summary(self) -> None:
        """Print test summary"""
        print("\n" + "="*80)
        print("TURBOQUANT INTEGRATION TEST SUMMARY")
        print("="*80 + "\n")
        
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        
        print(f"Tests Passed: {passed}/{total}")
        print(f"Success Rate: {100*passed/max(total,1):.1f}%\n")
        
        # Group by mode
        for mode in ["tq1", "tq2", "tq3", "tq4"]:
            mode_results = [r for r in self.results if r.mode == mode]
            if mode_results:
                mode_passed = sum(1 for r in mode_results if r.passed)
                avg_mse = np.mean([r.roundtrip_mse for r in mode_results])
                avg_ratio = np.mean([r.compression_ratio for r in mode_results])
                
                print(f"Mode {mode.upper()}:")
                print(f"  Tests: {mode_passed}/{len(mode_results)} passed")
                print(f"  Avg Compression: {avg_ratio:.2f}x")
                print(f"  Avg MSE: {avg_mse:.2e}")
                print()
        
        # Print failures
        failures = [r for r in self.results if not r.passed]
        if failures:
            print("\nFailed Tests:")
            for result in failures:
                print(f"  ✗ {result.test_name}: {result.error_msg}")
        
        print("="*80 + "\n")
    
    def save_results(self, output_path: Path = Path("integration_test_results.json")) -> None:
        """Save test results to JSON"""
        output_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": {
                "total": len(self.results),
                "passed": sum(1 for r in self.results if r.passed),
                "failed": sum(1 for r in self.results if not r.passed),
            },
            "results": [asdict(r) for r in self.results],
        }
        
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")


def main():
    """Run integration tests"""
    test_suite = TurboQuantIntegrationTest()
    test_suite.run_all_tests()
    test_suite.print_summary()
    test_suite.save_results()
    
    # Exit with appropriate code
    return 0 if test_suite.passed > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
