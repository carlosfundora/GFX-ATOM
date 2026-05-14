# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/gpt2/modeling_gpt2.py
# Copyright 2023 The vLLM team.
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only GPT-2 model compatible with HuggingFace weights."""

from collections.abc import Iterable
from itertools import islice
from typing import Optional

import torch
import torch.nn.functional as F
from aiter.dist.parallel_state import get_pp_group, get_tensor_model_parallel_world_size
from atom.config import Config, QuantizationConfig
from atom.model_ops.base_attention import Attention
from atom.model_ops.embed_head import ParallelLMHead, VocabParallelEmbedding
from atom.model_ops.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from atom.model_loader.loader import default_weight_loader
from atom.models.utils import (
    IntermediateTensors,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from atom.utils.decorators import support_torch_compile
from torch import nn
from transformers import GPT2Config


# GPT-2 activation functions
_ACT_FN_MAP = {
    "gelu": F.gelu,
    "gelu_new": lambda x: F.gelu(x, approximate="tanh"),
    "relu": F.relu,
    "tanh": torch.tanh,
}


class GPT2Attention(nn.Module):
    def __init__(
        self,
        config: GPT2Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        total_num_heads = config.num_attention_heads
        tensor_model_parallel_world_size = get_tensor_model_parallel_world_size()
        assert total_num_heads % tensor_model_parallel_world_size == 0
        self.num_heads = total_num_heads // tensor_model_parallel_world_size
        self.head_dim = self.hidden_size // total_num_heads
        self.scale = self.head_dim**-0.5

        self.c_attn = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            total_num_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.c_attn",
        )
        self.c_proj = RowParallelLinear(
            self.hidden_size,
            self.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.c_proj",
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            scale=self.scale,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.c_attn(hidden_states)
        q, k, v = qkv.chunk(chunks=3, dim=-1)
        attn_output = self.attn(q, k, v)
        attn_output, _ = self.c_proj(attn_output)
        return attn_output


class GPT2MLP(nn.Module):
    def __init__(
        self,
        intermediate_size: int,
        config: GPT2Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        hidden_size = config.hidden_size
        self.c_fc = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.c_fc",
        )
        self.c_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.c_proj",
        )
        act_fn_name = getattr(config, "activation_function", "gelu_new")
        self.act = _ACT_FN_MAP.get(act_fn_name, F.gelu)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states, _ = self.c_proj(hidden_states)
        return hidden_states


class GPT2Block(nn.Module):
    def __init__(
        self,
        config: GPT2Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(
            config, quant_config, prefix=f"{prefix}.attn"
        )
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(inner_dim, config, quant_config, prefix=f"{prefix}.mlp")

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output = self.attn(hidden_states=hidden_states)
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = residual + feed_forward_hidden_states
        return hidden_states


@support_torch_compile
class GPT2Model(nn.Module):
    def __init__(self, atom_config: Config, prefix: str = ""):
        super().__init__()

        config = atom_config.hf_config
        quant_config = atom_config.quant_config

        self.config = config
        assert not config.add_cross_attention
        assert not config.scale_attn_by_inverse_layer_idx
        assert not config.reorder_and_upcast_attn
        self.embed_dim = config.hidden_size
        self.wte = VocabParallelEmbedding(
            config.vocab_size,
            self.embed_dim,
            prefix=f"{prefix}.wte",
        )
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)
        self.start_layer, self.end_layer, self.h = make_layers(
            config.num_hidden_layers,
            lambda prefix: GPT2Block(config, quant_config, prefix=prefix),
            prefix=f"{prefix}.h",
        )
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], config.n_embd
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if get_pp_group().is_first_rank:
            if inputs_embeds is None:
                inputs_embeds = self.wte(input_ids)
            position_embeds = self.wpe(positions)
            hidden_states = inputs_embeds + position_embeds
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        for layer in islice(self.h, self.start_layer, self.end_layer):
            hidden_states = layer(hidden_states)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})

        hidden_states = self.ln_f(hidden_states)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if ".attn.bias" in name or ".attn.masked_bias" in name:
                continue
            if name not in params_dict:
                continue
            param = params_dict[name]
            # HF GPT-2 uses Conv1D instead of Linear — transpose weights.
            for conv1d_weight_name in ["c_attn", "c_proj", "c_fc"]:
                if conv1d_weight_name not in name:
                    continue
                if not name.endswith(".weight"):
                    continue
                loaded_weight = loaded_weight.t()
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class GPT2LMHeadModel(nn.Module):
    packed_modules_mapping = {
        "c_attn": ("c_attn",),
    }

    def __init__(self, atom_config: Config, prefix: str = ""):
        super().__init__()
        config = atom_config.hf_config
        quant_config = atom_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.transformer = GPT2Model(
            atom_config=atom_config, prefix=maybe_prefix(prefix, "transformer")
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                self.config.vocab_size,
                self.config.hidden_size,
                prefix=f"{prefix}.lm_head",
            )
            if self.config.tie_word_embeddings:
                self.lm_head.weight = self.transformer.wte.weight

        self.make_empty_intermediate_tensors = (
            self.transformer.make_empty_intermediate_tensors
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.transformer(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        weights = _add_transformer_prefix(weights)
        return self.transformer.load_weights(weights)


def _add_transformer_prefix(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterable[tuple[str, torch.Tensor]]:
    for name, tensor in weights:
        if not name.startswith("transformer.") and not name.startswith("lm_head"):
            name = "transformer." + name
        yield name, tensor
