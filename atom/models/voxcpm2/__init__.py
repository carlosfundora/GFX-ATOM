# SPDX-License-Identifier: Apache-2.0
"""VoxCPM2 — TTS + voice cloning with built-in speaker presets.

Architecture:
  MiniCPM4 base_lm (28 layers, PagedAttention + fp32 RoPE)
  → FSQ → MiniCPM4 residual_lm (8 layers, no RoPE)
  → LocDiT (CFM solver) → AudioVAE → 48kHz waveform

Supports zero-shot TTS, voice cloning via ref_audio, and voice continuation
via ref_audio + ref_text (prompt-based cloning).

Ported from vLLM-Omni's voxcpm2 module.
"""
