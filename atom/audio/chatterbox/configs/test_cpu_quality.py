#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Chatterbox CPU Quality Verification Test (Optimized).

Generates a 5-sentence female voice clip using the Standard ONNX model
on CPU with optimized ONNX RT settings (intra=12, inter=1, ORT_ENABLE_ALL),
then plays it over system speakers via aplay.
"""

import sys
import time
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[4]  # gfxatom/
sys.path.insert(0, str(project_root))

from atom.audio.chatterbox.engine import ChatterboxEngine
from atom.audio.utils import audio_to_bytes

# ─── Model Paths (from chatterbox_standard_cpu.yaml) ─────────────
MODEL_DIR = (
    "/home/local/Projects/models/huggingface/"
    "models--onnx-community--chatterbox-ONNX/"
    "snapshots/3cab09af388d3f02bba43443fce88c1f4525ac43"
)

# ─── Test Text: 5-sentence female voice passage ──────────────────
TEST_TEXT = (
    "The morning light filtered through the tall windows, casting golden "
    "patterns across the polished wooden floor. She paused at the doorway, "
    "taking in the quiet beauty of the empty library. Rows upon rows of "
    "leather-bound books lined the shelves, each one holding a world of its "
    "own. A faint scent of lavender drifted from the garden outside, mingling "
    "with the musty fragrance of old paper. She smiled softly and stepped "
    "inside, ready to lose herself in a new story."
)


def main() -> None:
    import subprocess

    print("=" * 60)
    print(" Chatterbox Standard — CPU Quality Test (Optimized)")
    print("=" * 60)
    print(f"\nModel: {MODEL_DIR}")
    print(f"Device: CPU (ONNX Runtime)")
    print(f"ONNX RT: intra_op=12, inter_op=1, ORT_ENABLE_ALL")
    print(f"Text: {len(TEST_TEXT)} chars, ~5 sentences")
    print()

    # Initialize engine (CPU mode — no backbone_dir)
    # num_threads=12 matches physical cores on Ryzen 9 3900X
    engine = ChatterboxEngine(
        model_dir=MODEL_DIR,
        backbone_dir=None,  # CPU-only: use ONNX language model
        device="cpu",
        dtype="float16",
        num_threads=12,  # Physical cores — optimized from 4
    )

    print("Loading model...")
    t0 = time.time()
    engine.load()
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s")
    print(f"  Architecture: {engine.service.backbone_arch}")
    print(f"  Layers: {engine.service.num_hidden_layers}")
    print(f"  Heads: {engine.service.num_heads}")
    print(f"  Head dim: {engine.service.head_dim}")
    print()

    # Generate speech
    print(f"Generating speech for:\n  \"{TEST_TEXT[:80]}...\"")
    print()

    wav, metrics = engine.generate(
        TEST_TEXT,
        max_tokens=512,
        repetition_penalty=1.2,
        temperature=0.0,  # Greedy argmax for CPU
        exaggeration=0.5,
    )

    # Report metrics
    print("─" * 40)
    print("Generation Metrics:")
    for key, val in metrics.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.3f}")
        else:
            print(f"  {key}: {val}")
    print("─" * 40)

    # Save to WAV
    wav_bytes, _ = audio_to_bytes(wav, engine.service.sample_rate, response_format="wav")
    out_file = Path(__file__).parent / "test_cpu_quality.wav"
    with open(out_file, "wb") as f:
        f.write(wav_bytes)
    print(f"\nSaved: {out_file} ({len(wav_bytes)} bytes)")

    # Play over speakers
    print("\n🔊 Playing over system speakers...")
    result = subprocess.run(["aplay", str(out_file)], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  aplay error: {result.stderr.strip()}")
    else:
        print("  ✓ Playback complete")

    print("\n✅ CPU quality test finished.")


if __name__ == "__main__":
    main()
