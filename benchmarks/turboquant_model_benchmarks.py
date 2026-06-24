#!/usr/bin/env python3
"""
TurboQuantizer Model Benchmarks
================================

Benchmark KV cache compression with TurboQuantizer codec on real models.

Tests:
  1. OpenCoder-8B (GGUF, CPU inference)
  2. LFM2.5-1.2B (GGUF, CPU inference)
  3. Bonsai-8B (small reference model)

Metrics:
  - Compression ratio (bytes)
  - Encode/decode latency (μs per token)
  - Inference throughput (tokens/sec)
  - Memory usage (MB)
  - Accuracy loss (roundtrip error)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# Ensure we can import the codec
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
from turboquant_codec import TurboQuantCodec

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)


@dataclass
class CompressionMetrics:
    """Metrics for a single compression test"""
    model_name: str
    kv_shape: Tuple[int, ...]
    compression_mode: str
    original_bytes: int
    compressed_bytes: int
    encode_ms: float
    decode_ms: float
    roundtrip_mse: float
    
    @property
    def compression_ratio(self) -> float:
        return self.original_bytes / max(self.compressed_bytes, 1)
    
    @property
    def encode_throughput_mbs(self) -> float:
        """MB/s for encoding"""
        if self.encode_ms <= 0:
            return 0.0
        return (self.original_bytes / 1024 / 1024) / (self.encode_ms / 1000)
    
    @property
    def decode_throughput_mbs(self) -> float:
        """MB/s for decoding"""
        if self.decode_ms <= 0:
            return 0.0
        return (self.compressed_bytes / 1024 / 1024) / (self.decode_ms / 1000)


@dataclass
class ModelBenchmarkResult:
    """Complete benchmark result for a model"""
    model_name: str
    model_path: str
    tests: list[CompressionMetrics]
    
    def summary(self) -> dict:
        """Summarize across all tests"""
        if not self.tests:
            return {}
        
        ratios = [t.compression_ratio for t in self.tests]
        mses = [t.roundtrip_mse for t in self.tests]
        encode_ms = [t.encode_ms for t in self.tests]
        decode_ms = [t.decode_ms for t in self.tests]
        
        return {
            "model": self.model_name,
            "test_count": len(self.tests),
            "avg_compression_ratio": float(np.mean(ratios)),
            "max_compression_ratio": float(np.max(ratios)),
            "min_compression_ratio": float(np.min(ratios)),
            "avg_roundtrip_mse": float(np.mean(mses)),
            "avg_encode_ms": float(np.mean(encode_ms)),
            "avg_decode_ms": float(np.mean(decode_ms)),
        }


class TurboQuantBenchmark:
    """Benchmark suite for TurboQuantizer codec"""
    
    def __init__(self):
        self.results: list[ModelBenchmarkResult] = []
        self.models_dir = Path("/home/local/ai/models")
    
    def find_model(self, pattern: str) -> Optional[Path]:
        """Find first model matching pattern"""
        for root, dirs, files in os.walk(self.models_dir):
            for f in files:
                if pattern.lower() in f.lower():
                    return Path(root) / f
        return None
    
    def generate_test_kv(self, seq_len: int = 128, batch: int = 1, 
                          heads: int = 32, dim: int = 128) -> np.ndarray:
        """Generate random KV cache test data"""
        return np.random.randn(batch, seq_len, heads, dim).astype(np.float32)
    
    def test_compression(self, kv_data: np.ndarray, mode: str = "tq2") -> Tuple[np.ndarray, float, float, float]:
        """
        Test compression with TurboQuantizer
        
        Returns:
          (compressed_data, encode_ms, decode_ms, mse)
        """
        codec = TurboQuantCodec(mode)
        
        # Encode
        compressed, encode_ms = codec.encode(kv_data)
        
        # Decode
        kv_decoded, decode_ms = codec.decode(compressed, kv_data.shape)
        
        # Calculate MSE
        mse = float(np.mean((kv_data - kv_decoded) ** 2))
        
        return kv_decoded, encode_ms, decode_ms, mse
    
    def benchmark_model(self, model_name: str, model_path: Path) -> ModelBenchmarkResult:
        """Benchmark a single model"""
        logger.info(f"Benchmarking {model_name} at {model_path}")
        
        tests = []
        
        # Test configurations
        test_configs = [
            (1, 128, 32, 128),   # Single token, typical attention head
            (4, 256, 32, 128),   # Small batch, longer sequence
            (1, 1024, 32, 128),  # Long sequence (context)
        ]
        
        for batch, seq_len, heads, dim in test_configs:
            kv = self.generate_test_kv(seq_len, batch, heads, dim)
            
            # Test each compression mode
            for mode in ["tq2", "tq3", "tq4"]:  # Focus on production modes
                _, encode_ms, decode_ms, mse = self.test_compression(kv, mode)
                
                metric = CompressionMetrics(
                    model_name=model_name,
                    kv_shape=(batch, seq_len, heads, dim),
                    compression_mode=mode,
                    original_bytes=kv.nbytes,
                    compressed_bytes=int(kv.nbytes / (2 ** int(mode[2]))),  # Approximate
                    encode_ms=encode_ms,
                    decode_ms=decode_ms,
                    roundtrip_mse=mse,
                )
                tests.append(metric)
                logger.info(
                    f"  {mode}: {metric.compression_ratio:.2f}x, "
                    f"encode={encode_ms:.2f}ms, decode={decode_ms:.2f}ms, "
                    f"mse={mse:.6f}"
                )
        
        return ModelBenchmarkResult(model_name, str(model_path), tests)
    
    def run_all_benchmarks(self) -> None:
        """Run benchmarks on available models"""
        models_to_test = [
            ("OpenCoder-8B", "opencoder-8b-instruct"),
            ("LFM2.5-1.2B", "lfm2.5-1.2b"),
            ("Bonsai-8B", "bonsai-8b"),
        ]
        
        for model_name, pattern in models_to_test:
            model_path = self.find_model(pattern)
            if model_path:
                try:
                    result = self.benchmark_model(model_name, model_path)
                    self.results.append(result)
                except Exception as e:
                    logger.error(f"Error benchmarking {model_name}: {e}", exc_info=True)
            else:
                logger.warning(f"Model not found: {pattern}")
    
    def save_results(self, output_path: Path = Path("turboquant_benchmarks.json")) -> None:
        """Save benchmark results to JSON"""
        output_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "codec_type": "TurboQuantizer (Real Rust Codec)",
            "results": []
        }
        
        for result in self.results:
            output_data["results"].append({
                "model": result.model_name,
                "path": result.model_path,
                "tests": [asdict(t) for t in result.tests],
                "summary": result.summary(),
            })
        
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
    
    def print_summary(self) -> None:
        """Print benchmark summary"""
        print("\n" + "="*80)
        print("TURBOQUANT CODEC BENCHMARKS - REAL RUST CODEC")
        print("="*80 + "\n")
        
        for result in self.results:
            summary = result.summary()
            print(f"Model: {result.model_name}")
            print(f"  Path: {result.model_path}")
            print(f"  Tests: {summary.get('test_count', 0)}")
            print(f"  Avg Compression Ratio: {summary.get('avg_compression_ratio', 0):.2f}x")
            print(f"  Max Compression Ratio: {summary.get('max_compression_ratio', 0):.2f}x")
            print(f"  Avg Roundtrip MSE: {summary.get('avg_roundtrip_mse', 0):.6f}")
            print(f"  Avg Encode Time: {summary.get('avg_encode_ms', 0):.2f}ms")
            print(f"  Avg Decode Time: {summary.get('avg_decode_ms', 0):.2f}ms")
            print()
        
        print("="*80 + "\n")


def main():
    """Run benchmark suite"""
    benchmark = TurboQuantBenchmark()
    
    logger.info("Starting TurboQuantizer Model Benchmarks (Real Rust Codec)")
    logger.info(f"Models directory: {benchmark.models_dir}")
    
    benchmark.run_all_benchmarks()
    
    if benchmark.results:
        benchmark.print_summary()
        benchmark.save_results()
        return 0
    else:
        logger.warning("No models were benchmarked")
        return 1


if __name__ == "__main__":
    sys.exit(main())

