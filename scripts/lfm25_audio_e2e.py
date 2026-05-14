#!/usr/bin/env python3
"""Run LFM2.5-Audio STT/TTS/interleaved proof trials.

The script uses the local llama.cpp-audio-max binary and existing GGUF files.
It does not install dependencies.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
import soundfile as sf

from atom.audio.lfm25_audio import (
    DEFAULT_PORT,
    LFM25AudioClient,
    build_lfm25_server_command,
    lfm25_runtime_env,
    resolve_lfm25_audio_paths,
)


LINE_OUT_SINK = "alsa_output.pci-0000_0f_00.4.analog-stereo"


def wait_ready(base_url: str, timeout: float = 180.0) -> None:
    deadline = time.time() + timeout
    last_error = ""
    root_url = base_url[:-3] if base_url.endswith("/v1") else base_url
    while time.time() < deadline:
        try:
            resp = httpx.get(f"{root_url}/health", timeout=2.0)
            if resp.status_code < 500:
                return
        except Exception as exc:
            last_error = str(exc)
        time.sleep(1.0)
    raise RuntimeError(f"LFM2.5-Audio server did not become ready: {last_error}")


def play_wav(path: Path, sink: str) -> None:
    env = dict(**os.environ)
    env.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    cmd = ["paplay", "--device", sink, str(path)]
    if subprocess.run(cmd, check=False, env=env).returncode != 0:
        subprocess.run(["aplay", str(path)], check=False, env=env)


def run_trial(args: argparse.Namespace, precision: str) -> dict:
    paths = resolve_lfm25_audio_paths(args.model_dir, precision=precision)
    port = args.port
    base_url = f"http://127.0.0.1:{port}/v1"
    command = build_lfm25_server_command(
        paths,
        port=port,
        n_gpu_layers=args.gpu_layers,
        threads=args.threads,
        flash_attn=args.flash_attn,
    )
    log_path = args.out_dir / f"lfm25-{precision}.log"
    metrics: dict = {"precision": precision, "model": str(paths.model), "log": str(log_path)}

    process = None
    try:
        if not args.use_existing:
            log = log_path.open("w")
            process = subprocess.Popen(
                command,
                env=lfm25_runtime_env(),
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
        wait_ready(base_url, timeout=args.ready_timeout)
        client = LFM25AudioClient(base_url=base_url, timeout=args.request_timeout)

        t0 = time.time()
        tts = client.text_to_speech(args.prompt, max_tokens=args.max_tokens)
        metrics["tts_sec"] = time.time() - t0
        tts_wav = args.out_dir / f"lfm25-{precision}-tts.wav"
        tts_wav.write_bytes(tts["wav_bytes"])
        metrics["tts_audio_chunks"] = tts["audio_chunks"]
        metrics["tts_wav"] = str(tts_wav)
        if args.play:
            play_wav(tts_wav, args.sink)

        t0 = time.time()
        transcript = client.transcribe_wav_bytes(tts["wav_bytes"], max_tokens=256)
        metrics["stt_sec"] = time.time() - t0
        metrics["transcript"] = transcript

        t0 = time.time()
        s2s = client.speech_to_speech(tts["wav_bytes"], max_tokens=args.max_tokens)
        metrics["s2s_sec"] = time.time() - t0
        s2s_wav = args.out_dir / f"lfm25-{precision}-s2s.wav"
        s2s_wav.write_bytes(s2s["wav_bytes"])
        metrics["s2s_text"] = s2s["text"]
        metrics["s2s_audio_chunks"] = s2s["audio_chunks"]
        metrics["s2s_wav"] = str(s2s_wav)
        if args.play:
            play_wav(s2s_wav, args.sink)

        for key, wav_path in (("tts_duration", tts_wav), ("s2s_duration", s2s_wav)):
            audio, sr = sf.read(wav_path)
            metrics[key] = len(audio) / max(sr, 1)
        return metrics
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--precision", choices=["q8", "f16", "both"], default="both")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--gpu-layers", default=0)
    parser.add_argument("--flash-attn", choices=["on", "off", "auto"], default="off")
    parser.add_argument("--use-existing", action="store_true")
    parser.add_argument("--play", action="store_true")
    parser.add_argument("--sink", default=LINE_OUT_SINK)
    parser.add_argument("--ready-timeout", type=float, default=180.0)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--max-tokens", type=int, default=320)
    parser.add_argument(
        "--prompt",
        default="This is an ATOM LFM two point five audio proof using the US female voice.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path(tempfile.gettempdir()) / "atom-lfm25-audio")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    precisions = ["q8", "f16"] if args.precision == "both" else [args.precision]
    results = []
    for precision in precisions:
        try:
            result = run_trial(args, precision)
            results.append(result)
            print(json.dumps(result, indent=2), flush=True)
        except FileNotFoundError as exc:
            print(f"SKIP {precision}: {exc}", file=sys.stderr, flush=True)
        except Exception as exc:
            print(f"FAIL {precision}: {exc}", file=sys.stderr, flush=True)
            results.append({"precision": precision, "error": str(exc)})

    metrics_path = args.out_dir / "lfm25-audio-e2e-metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2))
    print(f"metrics: {metrics_path}")
    return 0 if results and not all("error" in item for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
