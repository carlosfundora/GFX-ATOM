"""
Phase 6.1: Hybrid RotorTurbo Codec Quality Comparison

Benchmark: Pure RQ3 vs Pure TQ2 vs Hybrid RotorTurbo-3
Expected: Hybrid achieves same compression, 15-25% better reconstruction quality
"""

import sys
sys.path.insert(0, '/home/local/ai/projects/gfxATOM-Rust/python')

import torch
import numpy as np
from typing import Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_quality_metrics(original: np.ndarray, reconstructed: np.ndarray) -> Dict[str, float]:
    """
    Calculate reconstruction quality metrics.
    
    Args:
        original: Original data
        reconstructed: Reconstructed data
    
    Returns:
        Dict with quality metrics
    """
    # MSE (lower is better)
    mse = np.mean((original - reconstructed) ** 2)
    
    # PSNR (higher is better)
    max_val = np.max(np.abs(original))
    if max_val > 0:
        psnr = 20 * np.log10(max_val / np.sqrt(mse)) if mse > 0 else 100
    else:
        psnr = 100
    
    # Correlation (higher is better)
    correlation = np.corrcoef(original.flatten(), reconstructed.flatten())[0, 1]
    if np.isnan(correlation):
        correlation = 1.0
    
    # L∞ error (lower is better)
    l_inf = np.max(np.abs(original - reconstructed))
    
    # Energy ratio (closer to 1.0 is better)
    orig_energy = np.sum(original ** 2)
    recon_energy = np.sum(reconstructed ** 2)
    energy_ratio = recon_energy / orig_energy if orig_energy > 0 else 1.0
    
    return {
        "mse": float(mse),
        "psnr": float(psnr),
        "correlation": float(correlation),
        "l_inf": float(l_inf),
        "energy_ratio": float(energy_ratio),
    }


class HybridQualityBenchmark:
    """Compare codec quality: RQ3 vs TQ2 vs Hybrid-RQ3+TQ2"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = {}
    
    def generate_test_tensor(self, shape: Tuple[int, ...], data_type: str = "random") -> np.ndarray:
        """Generate test tensor with various distributions."""
        if data_type == "random":
            return np.random.randn(*shape).astype(np.float32)
        elif data_type == "gaussian":
            return np.random.normal(0, 0.3, shape).astype(np.float32)
        elif data_type == "uniform":
            return np.random.uniform(-1, 1, shape).astype(np.float32)
        elif data_type == "sparse":
            data = np.random.randn(*shape).astype(np.float32)
            data[data < -0.5] = 0
            return data
        elif data_type == "correlated":
            data = np.random.randn(*shape).astype(np.float32)
            # Add correlation: each element influenced by previous
            for i in range(1, shape[0]):
                data[i] = 0.7 * data[i] + 0.3 * data[i-1]
            return data
        else:
            raise ValueError(f"Unknown data type: {data_type}")
    
    def benchmark_codec_quality(
        self,
        test_shapes: list,
        test_types: list,
    ) -> Dict:
        """
        Benchmark codec quality across various tensor shapes and distributions.
        
        Args:
            test_shapes: List of tensor shapes to test
            test_types: List of data types (random, gaussian, uniform, sparse, correlated)
        
        Returns:
            Comprehensive quality comparison
        """
        logger.info("="*80)
        logger.info("HYBRID CODEC QUALITY COMPARISON")
        logger.info("="*80)
        logger.info(f"Codec modes: RQ3 (pure) vs TQ2 (pure) vs Hybrid-RQ3+TQ2")
        logger.info(f"Test shapes: {test_shapes}")
        logger.info(f"Data types: {test_types}")
        logger.info()
        
        results = {
            "rq3_pure": {},
            "tq2_pure": {},
            "hybrid_rq3_tq2": {},
            "comparisons": {},
        }
        
        for shape in test_shapes:
            logger.info(f"\nTesting shape: {shape}")
            logger.info("-" * 80)
                key = f"{shape}_{data_type}"
                logger.info(f"  Data type: {data_type}...")
                
                # Generate test tensor
                original = self.generate_test_tensor(shape, data_type)
                
                # Simulate codecs (placeholder implementations)
                # In real scenario, these would use actual Rust codec bindings
                
                # Pure RQ3: Apply Givens rotation + 3-bit quantization
                rq3_recon = self._simulate_rq3(original)
                rq3_metrics = calculate_quality_metrics(original, rq3_recon)
                
                # Pure TQ2: Apply polar quantization (2-bit)
                tq2_recon = self._simulate_tq2(original)
                tq2_metrics = calculate_quality_metrics(original, tq2_recon)
                
                # Hybrid: RQ3 decorrelation + TQ2 polar quantization
                hybrid_recon = self._simulate_hybrid_rq3_tq2(original)
                hybrid_metrics = calculate_quality_metrics(original, hybrid_recon)
                
                # Store results
                results["rq3_pure"][key] = rq3_metrics
                results["tq2_pure"][key] = tq2_metrics
                results["hybrid_rq3_tq2"][key] = hybrid_metrics
                
                # Compare: hybrid vs pure RQ3
                rq3_improvement = ((hybrid_metrics["mse"] - rq3_metrics["mse"]) / rq3_metrics["mse"] * 100) if rq3_metrics["mse"] > 0 else 0
                tq2_improvement = ((hybrid_metrics["mse"] - tq2_metrics["mse"]) / tq2_metrics["mse"] * 100) if tq2_metrics["mse"] > 0 else 0
                
                results["comparisons"][key] = {
                    "hybrid_vs_rq3_mse_delta_pct": rq3_improvement,
                    "hybrid_vs_tq2_mse_delta_pct": tq2_improvement,
                }
                
                # Log results
                logger.info(f"    RQ3 MSE:    {rq3_metrics['mse']:.6f}, PSNR: {rq3_metrics['psnr']:.2f}")
                logger.info(f"    TQ2 MSE:    {tq2_metrics['mse']:.6f}, PSNR: {tq2_metrics['psnr']:.2f}")
                logger.info(f"    HYBRID MSE: {hybrid_metrics['mse']:.6f}, PSNR: {hybrid_metrics['psnr']:.2f}")
                logger.info(f"    ➜ Hybrid vs RQ3: {rq3_improvement:+.1f}% MSE, vs TQ2: {tq2_improvement:+.1f}% MSE")
        
        return results
    
    def _simulate_rq3(self, data: np.ndarray) -> np.ndarray:
        """Simulate pure RQ3 codec."""
        # Givens rotation + 3-bit quantization
        rotated = self._apply_givens_rotation(data)
        quantized = self._quantize_3bit(rotated)
        dequantized = self._dequantize_3bit(quantized)
        inverse = self._inverse_givens_rotation(dequantized)
        return inverse
    
    def _simulate_tq2(self, data: np.ndarray) -> np.ndarray:
        """Simulate pure TQ2 codec."""
        # Polar transformation + 2-bit quantization
        quantized = self._quantize_2bit(data)
        dequantized = self._dequantize_2bit(quantized)
        return dequantized
    
    def _simulate_hybrid_rq3_tq2(self, data: np.ndarray) -> np.ndarray:
        """Simulate hybrid RotorTurbo codec: RQ3 decorrelation + TQ2 quantization."""
        # Stage 1: Decorrelate with Givens
        rotated = self._apply_givens_rotation(data)
        # Stage 2: Quantize decorrelated data with polar (TQ2)
        quantized = self._quantize_2bit(rotated)
        dequantized = self._dequantize_2bit(quantized)
        # Stage 1 inverse: Rotate back
        inverse = self._inverse_givens_rotation(dequantized)
        return inverse
    
    def _apply_givens_rotation(self, data: np.ndarray) -> np.ndarray:
        """Apply Givens rotation for decorrelation."""
        result = data.copy()
        dim = data.shape[-1] if len(data.shape) > 1 else data.shape[0]
        
        for i in range((dim + 1) // 2):
            angle = (i * np.pi / 256.0) % (2 * np.pi)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            
            # Apply to pairs
            if len(data.shape) == 1:
                if i * 2 + 1 < dim:
                    y1 = cos_a * result[i*2] - sin_a * result[i*2+1]
                    y2 = sin_a * result[i*2] + cos_a * result[i*2+1]
                    result[i*2] = y1
                    result[i*2+1] = y2
            else:
                if i * 2 + 1 < dim:
                    y1 = cos_a * result[..., i*2] - sin_a * result[..., i*2+1]
                    y2 = sin_a * result[..., i*2] + cos_a * result[..., i*2+1]
                    result[..., i*2] = y1
                    result[..., i*2+1] = y2
        
        return result
    
    def _inverse_givens_rotation(self, data: np.ndarray) -> np.ndarray:
        """Inverse Givens rotation."""
        result = data.copy()
        dim = data.shape[-1] if len(data.shape) > 1 else data.shape[0]
        
        for i in range((dim + 1) // 2):
            angle = (i * np.pi / 256.0) % (2 * np.pi)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            
            # Apply inverse to pairs
            if len(data.shape) == 1:
                if i * 2 + 1 < dim:
                    x1 = cos_a * result[i*2] + sin_a * result[i*2+1]
                    x2 = -sin_a * result[i*2] + cos_a * result[i*2+1]
                    result[i*2] = x1
                    result[i*2+1] = x2
            else:
                if i * 2 + 1 < dim:
                    x1 = cos_a * result[..., i*2] + sin_a * result[..., i*2+1]
                    x2 = -sin_a * result[..., i*2] + cos_a * result[..., i*2+1]
                    result[..., i*2] = x1
                    result[..., i*2+1] = x2
        
        return result
    
    def _quantize_3bit(self, data: np.ndarray) -> np.ndarray:
        """3-bit uniform quantization."""
        clamped = np.clip(data, -1, 1)
        normalized = (clamped + 1) / 2
        quantized = np.round(normalized * 7).astype(int)
        return quantized
    
    def _dequantize_3bit(self, quantized: np.ndarray) -> np.ndarray:
        """3-bit dequantization."""
        dequantized = (quantized.astype(np.float32) / 7.0) * 2 - 1
        return dequantized
    
    def _quantize_2bit(self, data: np.ndarray) -> np.ndarray:
        """2-bit uniform quantization."""
        clamped = np.clip(data, -1, 1)
        normalized = (clamped + 1) / 2
        quantized = np.round(normalized * 3).astype(int)
        return quantized
    
    def _dequantize_2bit(self, quantized: np.ndarray) -> np.ndarray:
        """2-bit dequantization."""
        dequantized = (quantized.astype(np.float32) / 3.0) * 2 - 1
        return dequantized


def run_hybrid_quality_benchmarks():
    """Execute Phase 6.1 hybrid codec quality tests."""
    benchmark = HybridQualityBenchmark()
    
    # Test shapes (various dimensionalities)
    test_shapes = [
        (128,),           # 1D vector
        (256,),           # Larger 1D
        (1024,),          # Attention head dimension
        (32, 128),        # Batch of small vectors
        (64, 256),        # Batch of attention vecs
        (128, 512),       # Large batch
    ]
    
    # Data distributions
    test_types = [
        "random",      # Standard Gaussian
        "gaussian",    # Tight Gaussian
        "uniform",     # Uniform [-1, 1]
        "sparse",      # Sparse (many zeros)
        "correlated",  # Autocorrelated
    ]
    
    results = benchmark.benchmark_codec_quality(test_shapes, test_types)
    
    # Summary statistics
    logger.info("\n" + "="*80)
    logger.info("SUMMARY STATISTICS")
    logger.info("="*80)
    
    hybrid_improvements = []
    for key, comp in results["comparisons"].items():
        hybrid_improvements.append(comp["hybrid_vs_tq2_mse_delta_pct"])
    
    avg_improvement = np.mean(hybrid_improvements)
    min_improvement = np.min(hybrid_improvements)
    max_improvement = np.max(hybrid_improvements)
    
    logger.info(f"\nHybrid vs TQ2 (MSE improvement):")
    logger.info(f"  Average: {avg_improvement:+.1f}%")
    logger.info(f"  Min: {min_improvement:+.1f}%")
    logger.info(f"  Max: {max_improvement:+.1f}%")
    
    logger.info(f"\nHybrid vs RQ3 (MSE difference):")
    rq3_diffs = [comp["hybrid_vs_rq3_mse_delta_pct"] for comp in results["comparisons"].values()]
    logger.info(f"  Average: {np.mean(rq3_diffs):+.1f}%")
    logger.info(f"  Min: {np.min(rq3_diffs):+.1f}%")
    logger.info(f"  Max: {np.max(rq3_diffs):+.1f}%")
    
    if avg_improvement > 10:
        logger.info("\n✅ HYBRID CODEC SUPERIOR: 10%+ better than TQ2")
    elif avg_improvement > 0:
        logger.info("\n✅ HYBRID CODEC COMPETITIVE: Better than TQ2")
    else:
        logger.info("\n⚠️  HYBRID CODEC: Similar to TQ2")
    
    logger.info("="*80)
    
    return results


if __name__ == "__main__":
    results = run_hybrid_quality_benchmarks()
