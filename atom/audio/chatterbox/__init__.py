# SPDX-License-Identifier: Apache-2.0
"""Chatterbox TTS integration for ATOM.

Supports both standard Chatterbox (LlamaForCausalLM backbone, 0.5B) and
Chatterbox Turbo (GPT2LMHeadModel backbone, 350M, 1-step distilled decoder).

Architecture:
    Reference Audio → [speech_encoder (CPU/ONNX)] → cond_emb, prompt_tokens, speaker_emb
    Text → [tokenizer + embed_tokens (CPU/ONNX)] → inputs_embeds
    concat(cond_emb, inputs_embeds) → [ATOM LLM (GPU)] → speech_tokens
    speech_tokens → [conditional_decoder/vocoder (CPU/ONNX)] → 24kHz WAV
"""
