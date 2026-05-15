# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM and SGLang projects
#
# LFM2 support adapted from the ROCm vLLM LFM2 model and the local SGLang
# LFM2 donor implementation for gfxATOM's native model interface.

from typing import Any, Iterable

import torch
import torch.nn.functional as F
from aiter.dist.parallel_state import get_pp_group, get_tp_group
from aiter.rotary_embedding import get_rope
from atom.config import Config
from atom.model_ops.activation import SiluAndMul
from atom.model_ops.embed_head import ParallelLMHead, VocabParallelEmbedding
from atom.model_ops.layernorm import RMSNorm
from atom.model_ops.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from atom.models.utils import (
    IntermediateTensors,
    PPMissingLayer,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from torch import nn
from transformers import Lfm2Config


def _lfm2_ff_dim(config: Lfm2Config) -> int:
    ff_dim = getattr(config, "block_ff_dim", None) or config.intermediate_size
    if getattr(config, "block_auto_adjust_ff_dim", False):
        ff_dim = int(2 * ff_dim / 3)
        multiplier = getattr(config, "block_ffn_dim_multiplier", None)
        if multiplier is not None:
            ff_dim = int(multiplier * ff_dim)
        multiple_of = getattr(config, "block_multiple_of", 256)
        ff_dim = multiple_of * ((ff_dim + multiple_of - 1) // multiple_of)
    return ff_dim


class Lfm2MLP(nn.Module):
    def __init__(
        self,
        config: Lfm2Config,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        ff_dim = _lfm2_ff_dim(config)
        hidden_size = getattr(config, "block_dim", config.hidden_size)
        self.w1 = MergedColumnParallelLinear(
            hidden_size,
            [ff_dim] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.w1",
        )
        self.w2 = RowParallelLinear(
            ff_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.w2",
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.w1(x)
        x = self.act_fn(x)
        return self.w2(x)


class Lfm2Attention(nn.Module):
    def __init__(
        self,
        config: Lfm2Config,
        atom_config: Config,
        layer_num: int,
        prefix: str = "",
    ) -> None:
        super().__init__()
        tp_size = get_tp_group().world_size
        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = getattr(config, "head_dim", None) or (
            self.hidden_size // self.total_num_heads
        )
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=atom_config.quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.out_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            quant_config=atom_config.quant_config,
            prefix=f"{prefix}.out_proj",
        )

        rope_parameters = getattr(config, "rope_parameters", None) or {
            "rope_theta": getattr(config, "rope_theta", 10000.0),
            "rope_type": "default",
        }
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=getattr(config, "max_position_embeddings", 8192),
            base=rope_parameters.get("rope_theta", 10000.0),
            rope_scaling=rope_parameters,
        )
        self.q_layernorm = RMSNorm(self.head_dim, eps=config.norm_eps)
        self.k_layernorm = RMSNorm(self.head_dim, eps=config.norm_eps)
        assert self.num_heads % self.num_kv_heads == 0
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.register_buffer("_k_cache", torch.empty(0), persistent=False)
        self.register_buffer("_v_cache", torch.empty(0), persistent=False)

    def _repeat_kv(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.num_kv_groups == 1:
            return tensor
        return tensor.repeat_interleave(self.num_kv_groups, dim=1)

    def _attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_new_sequence: bool,
    ) -> torch.Tensor:
        if is_new_sequence:
            self._k_cache = k.detach()
            self._v_cache = v.detach()
            k_all = k
            v_all = v
        else:
            if self._k_cache.numel() == 0 or self._k_cache.device != k.device:
                self._k_cache = k.detach()
                self._v_cache = v.detach()
            else:
                self._k_cache = torch.cat([self._k_cache.to(dtype=k.dtype), k.detach()], dim=0)
                self._v_cache = torch.cat([self._v_cache.to(dtype=v.dtype), v.detach()], dim=0)
            k_all = self._k_cache.to(dtype=k.dtype)
            v_all = self._v_cache.to(dtype=v.dtype)

        k_all = self._repeat_kv(k_all)
        v_all = self._repeat_kv(v_all)
        scores = torch.einsum("thd,shd->hts", q, k_all) * self.scaling
        if is_new_sequence and q.shape[0] > 1:
            mask = torch.triu(
                torch.ones(q.shape[0], k_all.shape[0], device=q.device, dtype=torch.bool),
                diagonal=1,
            )
            scores = scores.masked_fill(mask.unsqueeze(0), torch.finfo(scores.dtype).min)
        probs = torch.softmax(scores.float(), dim=-1).to(dtype=q.dtype)
        output = torch.einsum("hts,shd->thd", probs, v_all)
        return output.reshape(q.shape[0], self.num_heads * self.head_dim)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        **model_kwargs: dict[str, Any],
    ) -> torch.Tensor:
        n_tokens = hidden_states.shape[0]
        qkv = self.qkv_proj(hidden_states)
        q, k, v = torch.split(qkv, [self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = self.q_layernorm(q.view(n_tokens, self.num_heads, self.head_dim)).view(
            n_tokens, self.num_heads * self.head_dim
        )
        k = self.k_layernorm(k.view(n_tokens, self.num_kv_heads, self.head_dim)).view(
            n_tokens, self.num_kv_heads * self.head_dim
        )
        q = q.view(n_tokens, self.num_heads, self.head_dim)
        k = k.view(n_tokens, self.num_kv_heads, self.head_dim)
        v = v.view(n_tokens, self.num_kv_heads, self.head_dim)
        q, k = self.rotary_emb(positions, q, k)
        is_new_sequence = positions.numel() != 1 or int(positions[0].item()) == 0
        output = self._attention(q, k, v, is_new_sequence=is_new_sequence)
        return self.out_proj(output)


class Lfm2ShortConv(nn.Module):
    """Depthwise short convolution for LFM2.

    gfxATOM does not yet expose a scheduler-owned per-request cache for LFM2's
    conv layers. This module keeps the conv state locally, which is correct for
    single-sequence serving and lets the LFM2 text model run on a gfxATOM
    instance while the full cache integration is isolated for a later change.
    """

    def __init__(
        self,
        config: Lfm2Config,
        atom_config: Config,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.kernel_size = int(config.conv_L_cache)
        self.hidden_size = config.hidden_size
        tp_size = get_tp_group().world_size
        assert self.hidden_size % tp_size == 0
        self.hidden_size_per_partition = self.hidden_size // tp_size
        bias = bool(getattr(config, "conv_bias", False))

        self.in_proj = MergedColumnParallelLinear(
            self.hidden_size,
            [self.hidden_size] * 3,
            bias=bias,
            quant_config=atom_config.quant_config,
            prefix=f"{prefix}.in_proj",
        )
        self.conv = nn.Conv1d(
            self.hidden_size_per_partition,
            self.hidden_size_per_partition,
            self.kernel_size,
            groups=self.hidden_size_per_partition,
            bias=bias,
        )
        self.out_proj = RowParallelLinear(
            self.hidden_size,
            self.hidden_size,
            bias=bias,
            quant_config=atom_config.quant_config,
            prefix=f"{prefix}.out_proj",
        )
        self.register_buffer("_conv_state", torch.empty(0), persistent=False)

    def _zero_state(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            self.kernel_size - 1,
            self.hidden_size_per_partition,
            dtype=x.dtype,
            device=x.device,
        )

    def _prefill_conv(self, x: torch.Tensor) -> torch.Tensor:
        x_t = x.transpose(0, 1).unsqueeze(0)
        padded = F.pad(x_t, (self.kernel_size - 1, 0))
        out = F.conv1d(padded, self.conv.weight, self.conv.bias, groups=x.shape[-1])
        if x.shape[0] >= self.kernel_size - 1:
            self._conv_state = x[-(self.kernel_size - 1) :].detach()
        else:
            state = self._zero_state(x)
            state[-x.shape[0] :] = x.detach()
            self._conv_state = state
        return out.squeeze(0).transpose(0, 1)

    def _decode_conv(self, x: torch.Tensor) -> torch.Tensor:
        if self._conv_state.numel() == 0 or self._conv_state.device != x.device:
            self._conv_state = self._zero_state(x)
        window = torch.cat([self._conv_state.to(dtype=x.dtype), x], dim=0)
        weight = self.conv.weight.squeeze(1).to(dtype=x.dtype)
        out = (window.transpose(0, 1) * weight).sum(dim=1, keepdim=True).transpose(0, 1)
        if self.conv.bias is not None:
            out = out + self.conv.bias.to(dtype=out.dtype)
        self._conv_state = window[-(self.kernel_size - 1) :].detach()
        return out

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        projected = self.in_proj(hidden_states)
        b_gate, c_gate, x = projected.chunk(3, dim=-1)
        x = b_gate * x
        is_new_sequence = positions.numel() != 1 or int(positions[0].item()) == 0
        conv_out = self._prefill_conv(x) if is_new_sequence else self._decode_conv(x)
        return self.out_proj(c_gate * conv_out)


class Lfm2DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Lfm2Config,
        atom_config: Config,
        layer_num: int,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_type = config.layer_types[layer_num]
        if self.layer_type == "full_attention":
            self.self_attn = Lfm2Attention(
                config=config,
                atom_config=atom_config,
                layer_num=layer_num,
                prefix=f"{prefix}.self_attn",
            )
        else:
            self.conv = Lfm2ShortConv(
                config=config,
                atom_config=atom_config,
                prefix=f"{prefix}.conv",
            )
        self.feed_forward = Lfm2MLP(
            config=config,
            quant_config=atom_config.quant_config,
            prefix=f"{prefix}.feed_forward",
        )
        self.operator_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        **model_kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.operator_norm(hidden_states)
        else:
            hidden_states, residual = self.operator_norm(hidden_states, residual)

        if self.layer_type == "full_attention":
            hidden_states = self.self_attn(
                positions=positions, hidden_states=hidden_states, **model_kwargs
            )
        else:
            hidden_states = self.conv(
                positions=positions, hidden_states=hidden_states
            )

        hidden_states, residual = self.ffn_norm(hidden_states, residual)
        hidden_states = self.feed_forward(hidden_states)
        return hidden_states, residual


class Lfm2Model(nn.Module):
    def __init__(self, *, atom_config: Config, prefix: str = "") -> None:
        super().__init__()
        config = atom_config.hf_config
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )

        def get_layer(prefix: str, layer_num: int):
            return Lfm2DecoderLayer(
                config=config,
                atom_config=atom_config,
                layer_num=layer_num,
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )
        if get_pp_group().is_last_rank:
            self.embedding_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        else:
            self.embedding_norm = PPMissingLayer()

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **model_kwargs: dict[str, Any],
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            hidden_states = (
                inputs_embeds
                if inputs_embeds is not None
                else self.embed_tokens(input_ids)
            )
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in self.layers[self.start_layer : self.end_layer]:
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
                **model_kwargs,
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.embedding_norm(hidden_states, residual)
        return hidden_states


class Lfm2ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "w1": ("w1", 0),
        "w3": ("w1", 1),
    }

    def __init__(self, config: Config, prefix: str = "") -> None:
        super().__init__()
        self.atom_config = config
        self.hf_config = config.hf_config
        self.model = Lfm2Model(
            atom_config=self.atom_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                self.hf_config.vocab_size,
                self.hf_config.hidden_size,
                bias=False,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if getattr(self.hf_config, "tie_word_embeddings", True):
                self.lm_head.weight = self.model.embed_tokens.weight
        else:
            self.lm_head = PPMissingLayer()
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **model_kwargs: dict[str, Any],
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **model_kwargs,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        raise NotImplementedError("gfxATOM loads native LFM2 weights via load_model().")
