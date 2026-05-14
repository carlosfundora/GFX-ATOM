# SPDX-License-Identifier: Apache-2.0
"""Fish Speech S2 Pro Slow AR model — Qwen3-based backbone.

This is the main text-to-semantic-token transformer (36 layers, 2560 hidden).
Uses Qwen3 architecture with codebook embeddings and semantic logit masking.

The model uses interleaved (GPT-J style) RoPE, multi-codebook embedding for
voice cloning, and a nested Fast AR predictor for residual codebook codes.

Weight remapping transforms Fish Speech HF checkpoint names to Qwen3 format:
  model.layers.N.attention.wqkv → model.layers.N.self_attn.{q,k,v}_proj
  model.layers.N.feed_forward.{w1,w2,w3} → model.layers.N.mlp.{gate,down,up}_proj

Ported from vLLM-Omni's fish_speech_slow_ar.py.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn

from atom.models.fish_speech.configuration_fish_speech import (
    FishSpeechConfig,
    FishSpeechFastARConfig,
    FishSpeechSlowARConfig,
)

logger = logging.getLogger("atom.fish_speech")


# ---------------------------------------------------------------------------
#  Weight remapping: Fish Speech HF → Qwen3 format
# ---------------------------------------------------------------------------


def _remap_fish_speech_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
    n_head: int,
    n_local_heads: int,
    head_dim: int,
    fast_n_head: int,
    fast_n_local_heads: int,
    fast_head_dim: int,
) -> Iterable[tuple[str, torch.Tensor]]:
    """Transform Fish Speech HF weight names to Qwen3-compatible format."""

    def _split_wqkv(tensor: torch.Tensor, nh: int, nkv: int, hd: int):
        q_size = nh * hd
        kv_size = nkv * hd
        q, k, v = tensor.split([q_size, kv_size, kv_size], dim=0)
        return q, k, v

    for name, tensor in weights:
        # Skip rotary embedding inverse frequencies
        if "rotary_emb.inv_freq" in name:
            continue

        # --- Slow AR (text_model / model) ---
        if name.startswith("model."):
            inner = name[len("model."):]

            # Fused QKV → separate q/k/v
            if ".attention.wqkv.weight" in inner:
                layer_part = inner.split(".attention.wqkv.weight")[0]
                q, k, v = _split_wqkv(tensor, n_head, n_local_heads, head_dim)
                yield f"model.{layer_part}.self_attn.q_proj.weight", q
                yield f"model.{layer_part}.self_attn.k_proj.weight", k
                yield f"model.{layer_part}.self_attn.v_proj.weight", v
                continue

            # Attention output projection
            inner = inner.replace(".attention.wo.", ".self_attn.o_proj.")

            # Q/K norms
            inner = inner.replace(".attention.q_norm.", ".self_attn.q_norm.")
            inner = inner.replace(".attention.k_norm.", ".self_attn.k_norm.")

            # Attention norms → standard names
            inner = inner.replace(".attention_norm.", ".input_layernorm.")
            inner = inner.replace(".ffn_norm.", ".post_attention_layernorm.")

            # FFN: w1=gate, w2=down, w3=up
            inner = inner.replace(".feed_forward.w1.", ".mlp.gate_proj.")
            inner = inner.replace(".feed_forward.w2.", ".mlp.down_proj.")
            inner = inner.replace(".feed_forward.w3.", ".mlp.up_proj.")

            # Top-level norm
            inner = inner.replace("norm.", "norm.")

            yield f"model.{inner}", tensor
            continue

        # --- Fast AR (audio_decoder) ---
        if name.startswith("audio_decoder."):
            inner = name[len("audio_decoder."):]

            # Fused QKV
            if ".attention.wqkv.weight" in inner:
                layer_part = inner.split(".attention.wqkv.weight")[0]
                q, k, v = _split_wqkv(tensor, fast_n_head, fast_n_local_heads, fast_head_dim)
                yield f"fast_ar.model.{layer_part}.self_attn.q_proj.weight", q
                yield f"fast_ar.model.{layer_part}.self_attn.k_proj.weight", k
                yield f"fast_ar.model.{layer_part}.self_attn.v_proj.weight", v
                continue

            inner = inner.replace(".attention.wo.", ".self_attn.o_proj.")
            inner = inner.replace(".attention.q_norm.", ".self_attn.q_norm.")
            inner = inner.replace(".attention.k_norm.", ".self_attn.k_norm.")
            inner = inner.replace(".attention_norm.", ".input_layernorm.")
            inner = inner.replace(".ffn_norm.", ".post_attention_layernorm.")
            inner = inner.replace(".feed_forward.w1.", ".mlp.gate_proj.")
            inner = inner.replace(".feed_forward.w2.", ".mlp.down_proj.")
            inner = inner.replace(".feed_forward.w3.", ".mlp.up_proj.")

            # Fast AR specific
            inner = inner.replace("fast_embeddings.", "fast_embeddings.")
            inner = inner.replace("fast_output.", "fast_output.")
            inner = inner.replace("fast_project_in.", "fast_project_in.")

            # Norm → fast_norm
            if inner == "norm.weight":
                yield "fast_ar.fast_norm.weight", tensor
                continue

            yield f"fast_ar.{inner}", tensor
            continue

        # --- Top-level weights ---
        # codebook_embeddings, lm_head, embed_tokens
        yield name, tensor
