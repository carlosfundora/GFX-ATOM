# SPDX-License-Identifier: Apache-2.0
"""Fish Speech S2 Pro — Dual-AR TTS with DAC codec.

Architecture:
  Slow AR (Qwen3-based, 36 layers, 2560 hidden) → semantic tokens
  Fast AR (4-layer residual predictor, 10 RVQ codebooks) → residual codes
  DAC decoder → 44.1 kHz waveform

Ported from vLLM-Omni's fish_speech module.
"""
