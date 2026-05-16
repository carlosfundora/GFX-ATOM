import time
import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from unittest.mock import MagicMock

def patch_modules():
    # Mock hard-to-load modules for basic pure CPU benchmarking logic testing
    import sys
    sys.modules['torch'] = MagicMock()
    sys.modules['aiter'] = MagicMock()
    sys.modules['torch.nn.functional'] = MagicMock()

try:
    from atom.audio.chatterbox.engine import ChatterboxEngine
except ImportError:
    patch_modules()
    from atom.audio.chatterbox.engine import ChatterboxEngine

def main():
    parser = argparse.ArgumentParser(description="Benchmark Chatterbox TTS Engine Latency")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to Chatterbox ONNX snapshot")
    parser.add_argument("--backbone_dir", type=str, required=False, help="Path to backbone (GPU)", default=None)
    parser.add_argument("--text", type=str, default="This is a quick test of the TTS streaming latency. We need to ensure that the response is incredibly fast.", help="Text to generate")
    parser.add_argument("--runs", type=int, default=3, help="Number of times to run")
    args = parser.parse_args()

    print(f"Loading engine with model_dir={args.model_dir}...")
    engine = ChatterboxEngine(
        model_dir=args.model_dir,
        backbone_dir=args.backbone_dir,
        device="cpu",
        dtype="float32",
    )

    t0 = time.time()
    try:
        engine.load()
    except Exception as e:
        print(f"Engine load failed (expected if mock used or paths invalid): {e}")
        return

    print(f"Engine loaded in {time.time() - t0:.2f}s")

    # Warmup
    print("Running warmup pass...")
    try:
        engine.generate("Warmup text.")
    except Exception as e:
         pass # Might fail in mock

    for i in range(args.runs):
        print(f"\n--- Run {i+1} ---")
        t_start = time.time()
        wav, metrics = engine.generate(args.text)
        t_total = time.time() - t_start

        print("Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
        print(f"Total outside time: {t_total:.4f}s")
        print(f"RTF (external): {t_total / max(metrics.get('audio_duration', 0.001), 0.001):.4f}")

if __name__ == "__main__":
    main()
