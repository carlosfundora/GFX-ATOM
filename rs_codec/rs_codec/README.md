# `rs_codec`

A high-performance Rust implementation for the Demerzel audio DSP kernels, integrated natively using PyO3.

## Purpose

The `AudioCodec` pipeline inside `src/audio/processing/codec.py` historically relied on Python Fallbacks, `scipy.signal.lfilter`, and Numba JIT compiled implementations to operate soft noise gates, presence boosts, soft compressors, automatic gain controls (AGC), and frequency shelving.

While Numba provided a substantial speedup over pure Python, the overhead of the JIT compilation cache, LLVM dependency chain, and unpredictable warmup times generated difficulties.

`rs_codec` offloads these critical hot-loops into static, pure Rust functions exposed directly as a python C extension.

## Features

- **Soft Compressor:** Implements peak-following threshold compression `soft_compressor`.
- **AGC Kernel:** Applies continuous dynamic digital gain targeting -18 dBFS (`agc_kernel`).
- **IIR 1-Pole Kernel:** Accelerates naive array smoothing / presence shelving filtering (`iir_1pole_kernel`).
- **Highpass Kernel:** A fast 1st order DC-removal filter (`highpass_kernel`).
- **Noise Gate Kernel:** Performs moving RMS-window evaluation over frames and applies a noise-floor gating algorithm avoiding dynamic temporary array allocations (`noise_gate_kernel`).

## Integration Points

The Rust crate is configured to be automatically resolved into a native extension module when built through Python via `pyproject.toml`'s `maturin` hooks.

To manually inspect or modify:
1. `maturin develop --release` inside the workspace or repo root.
2. The python bindings look for the `rs_codec` namespace via `import rs_codec`.
3. See `src/audio/processing/codec.py` block prefixed by `_HAS_RUST` checking.
