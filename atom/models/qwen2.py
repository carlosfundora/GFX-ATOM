# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Adapted from ATOM's Qwen3 model and SGLang/vLLM Qwen2 implementations.

from typing import Any, Iterable

import torch
from aiter.dist.parallel_state import get_tp_group
from aiter.rotary_embedding import get_rope
from atom.config import Config
from atom.model_loader.loader import WeightsMapper, load_model_in_plugin_mode
from atom.model_ops.activation import SiluAndMul
from atom.model_ops.base_attention import Attention
from atom.model_ops.embed_head import ParallelLMHead, VocabParallelEmbedding
from atom.model_ops.layernorm import RMSNorm
from atom.model_ops.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from atom.models.utils import maybe_prefix
from atom.utils.decorators import support_torch_compile
from torch import nn
from transformers import Qwen2Config


class Qwen2Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 32768,
        head_dim: int | None = None,
        rope_theta: float = 1000000,
        rope_scaling: tuple | None = None,
        kv_cache_dtype: str = "fp16",
        layer_num: int = 0,
        atom_config: Config = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        tp_size = get_tp_group().world_size
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=True,
            quant_config=atom_config.quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=atom_config.quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            scale=self.scaling,
            num_kv_heads=self.num_kv_heads,
            kv_cache_dtype=kv_cache_dtype,
            layer_num=layer_num,
            use_mla=False,
            rotary_emb=self.rotary_emb,
            config=atom_config,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        **model_kwargs: dict[str, Any] | None,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = torch.split(qkv, [self.q_size, self.kv_size, self.kv_size], dim=-1)
        output = self.attn(q, k, v, positions, **model_kwargs)
        return self.o_proj(output)


class Qwen2MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        return self.down_proj(x)


class Qwen2DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen2Config,
        atom_config: Config,
        layer_num: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__()
        kv_cache_dtype = atom_config.kv_cache_dtype
        self.self_attn = Qwen2Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=getattr(config, "max_position_embeddings", 32768),
            head_dim=getattr(config, "head_dim", None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
            kv_cache_dtype=kv_cache_dtype,
            layer_num=layer_num,
            atom_config=atom_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = Qwen2MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=atom_config.quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        **model_kwargs: dict[str, Any] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions, hidden_states=hidden_states, **model_kwargs
        )
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


@support_torch_compile(dynamic_arg_dims={"input_ids": 0, "positions": -1})
class Qwen2Model(nn.Module):
    def __init__(self, *, atom_config: Config, prefix: str = "") -> None:
        super().__init__()
        hf_config = atom_config.hf_config
        self.embed_tokens = VocabParallelEmbedding(
            hf_config.vocab_size, hf_config.hidden_size
        )
        self.layers = nn.ModuleList(
            [
                Qwen2DecoderLayer(
                    config=hf_config,
                    atom_config=atom_config,
                    layer_num=layer_num,
                    prefix=f"{prefix}.layers.{layer_num}",
                )
                for layer_num in range(hf_config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(hf_config.hidden_size, eps=hf_config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        **model_kwargs: dict[str, Any],
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
                **model_kwargs,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen2ForCausalLM(nn.Module):
    weights_mapper = WeightsMapper(
        orig_to_new_prefix={
            "embed_tokens.": "model.embed_tokens.",
            "layers.": "model.layers.",
            "norm.": "model.norm.",
        },
    )

    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config: Any, prefix: str = "") -> None:
        super().__init__()
        self.atom_config = config
        self.hf_config = self.atom_config.hf_config
        self.model = Qwen2Model(
            atom_config=self.atom_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            num_embeddings=self.hf_config.vocab_size,
            embedding_dim=self.hf_config.hidden_size,
            bias=False,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if self.hf_config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
        **model_kwargs: dict[str, Any],
    ) -> torch.Tensor:
        return self.model(input_ids=input_ids, positions=positions, **model_kwargs)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return load_model_in_plugin_mode(
            model=self,
            config=self.atom_config,
            weights_mapper=self.weights_mapper,
        )
