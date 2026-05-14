# SPDX-License-Identifier: Apache-2.0
"""Run local Chatterbox backend trials and play results on the line-out sink.

This script assumes the caller is using an existing gfxATOM environment. It
does not install packages or create environments.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import soundfile as sf

from atom.audio.chatterbox.engine import ChatterboxEngine
from atom.audio.chatterbox.vllm_backend import (
    DEFAULT_CHATTERBOX_VLLM_SOURCE,
    DEFAULT_US_FEMALE_VOICE,
    ChatterboxAtomVllmEngine,
)

LINE_OUT_SINK = "alsa_output.pci-0000_0f_00.4.analog-stereo"

TRIALS = [
    {"backend": "fallback", "diffusion_steps": 10, "batch_size": 1, "chunk_chars": 240, "cfg_weight": 0.5},
    {"backend": "atom_vllm", "diffusion_steps": 10, "batch_size": 4, "chunk_chars": 160, "cfg_weight": 0.5},
    {"backend": "atom_vllm", "diffusion_steps": 5, "batch_size": 8, "chunk_chars": 240, "cfg_weight": 0.5},
    {"backend": "atom_vllm", "diffusion_steps": 3, "batch_size": 12, "chunk_chars": 320, "cfg_weight": 0.3},
    {"backend": "atom_vllm", "diffusion_steps": 2, "batch_size": 15, "chunk_chars": 400, "cfg_weight": 0.0},
]


def play_wav(path: Path, sink: str) -> None:
    if shutil.which("paplay"):
        subprocess.run(["paplay", "--device", sink, str(path)], check=False)
    elif shutil.which("pw-play"):
        subprocess.run(["pw-play", "--target", sink, str(path)], check=False)
    else:
        subprocess.run(["aplay", str(path)], check=False)


def build_engine(args: argparse.Namespace):
    fallback = ChatterboxEngine(
        model_dir=args.model_dir,
        backbone_dir=args.backbone_dir,
        variant=args.variant,
        onnx_variant=args.onnx_variant,
        device=args.device,
        dtype=args.dtype,
        num_threads=args.onnx_threads,
    )
    engine = ChatterboxAtomVllmEngine(
        model_dir=args.model_dir,
        variant=args.variant,
        device=args.device,
        source_dir=args.vllm_source,
        max_batch_size=args.max_batch_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=not args.vllm_compile,
        default_voice_path=args.voice,
        fallback_engine=fallback,
    )
    engine.load()
    return engine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--backbone-dir")
    parser.add_argument("--variant", default="standard", choices=["standard", "turbo"])
    parser.add_argument("--onnx-variant", default="fp16", choices=["fp32", "fp16", "q4", "q4f16", "q8"])
    parser.add_argument("--onnx-threads", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--vllm-source", default=str(DEFAULT_CHATTERBOX_VLLM_SOURCE))
    parser.add_argument("--voice", default=str(DEFAULT_US_FEMALE_VOICE))
    parser.add_argument("--sink", default=LINE_OUT_SINK)
    parser.add_argument("--max-batch-size", type=int, default=15)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--vllm-compile", action="store_true")
    parser.add_argument("--text", default="Hello. This is Bella on the ATOM Chatterbox tuning path. The goal is a clear, smooth, low latency streaming voice.")
    parser.add_argument("--out-dir", default="/tmp/atom-chatterbox-trials")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    engine = build_engine(args)
    results = []

    for index, trial in enumerate(TRIALS, start=1):
        t0 = time.time()
        wav, metrics = engine.generate(
            args.text,
            ref_audio_path=args.voice,
            max_tokens=1000,
            repetition_penalty=2.0,
            temperature=0.8,
            top_p=1.0,
            min_p=0.05,
            **trial,
        )
        metrics["wall_sec"] = time.time() - t0
        wav_path = out_dir / f"trial-{index:02d}-{trial['backend']}.wav"
        sf.write(str(wav_path), wav, engine.sample_rate)
        print(
            json.dumps(
                {
                    "trial": index,
                    "wav": str(wav_path),
                    "backend": metrics.get("backend", trial["backend"]),
                    "tokens_per_second": metrics.get("tok_per_sec", 0.0),
                    "audio_duration": metrics.get("audio_duration", 0.0),
                    "rtf": metrics.get("rtf", 0.0),
                    "wall_sec": metrics["wall_sec"],
                    "diffusion_steps": trial["diffusion_steps"],
                    "batch_size": trial["batch_size"],
                    "chunk_chars": trial["chunk_chars"],
                    "cfg_weight": trial["cfg_weight"],
                    "sink": args.sink,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        play_wav(wav_path, args.sink)
        results.append(metrics)

    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
