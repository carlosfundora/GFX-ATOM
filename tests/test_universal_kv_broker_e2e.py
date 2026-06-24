from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from sglang_backend_adapter import TieredKvCacheAdapter  # noqa: E402


def _target_shape(model_id: str) -> tuple[int, int]:
    # Minimal registry for cross-model materialization contract checks.
    registry = {
        "lfm2.5-thinking": (8, 16),
        "qwen3.5-instruct": (8, 16),
        "opencoder-8b": (8, 16),
    }
    return registry[model_id]


def test_cross_model_materialization_contract():
    adapter = TieredKvCacheAdapter(
        gpu_capacity_mb=1,
        ram_capacity_mb=16,
        dimension=64,
        num_heads=8,
    )

    seq_len = 32
    head_count, head_dim = _target_shape("lfm2.5-thinking")
    k_cache = torch.randn(seq_len, head_count * head_dim, dtype=torch.float32)
    v_cache = torch.randn(seq_len, head_count * head_dim, dtype=torch.float32)

    block_id = adapter.allocate_kv_block(
        request_id="lfm-prefill",
        layer_idx=0,
        k_cache=k_cache,
        v_cache=v_cache,
        importance_score=0.95,
    )
    materialized = adapter.get_kv_block(block_id)

    # Broker is model-agnostic; the consumer can reshape for target model contract.
    qwen_heads, qwen_head_dim = _target_shape("qwen3.5-instruct")
    reshaped = materialized[:, : qwen_heads * qwen_head_dim].reshape(
        seq_len, qwen_heads, qwen_head_dim
    )

    assert reshaped.shape == (seq_len, qwen_heads, qwen_head_dim)
    assert torch.isfinite(reshaped).all()


def test_multi_model_pool_forces_spill_and_restore():
    adapter = TieredKvCacheAdapter(
        gpu_capacity_mb=1,
        ram_capacity_mb=8,
        dimension=64,
        num_heads=8,
    )

    block_ids: list[int] = []
    for idx, model_id in enumerate(
        ["lfm2.5-thinking", "qwen3.5-instruct", "opencoder-8b"] * 6
    ):
        heads, dim = _target_shape(model_id)
        k_cache = torch.randn(48, heads * dim, dtype=torch.float32)
        v_cache = torch.randn(48, heads * dim, dtype=torch.float32)
        block_ids.append(
            adapter.allocate_kv_block(
                request_id=f"{model_id}-{idx}",
                layer_idx=idx % 4,
                k_cache=k_cache,
                v_cache=v_cache,
                importance_score=0.2 if idx % 2 else 0.9,
            )
        )

    stats = adapter.get_cache_stats()
    assert stats["ram_tier"]["blocks"] > 0

    recovered = adapter.get_kv_block(block_ids[-1])
    assert recovered.shape[0] == 48
    assert torch.isfinite(recovered).all()
