from pathlib import Path

import torch

from atom.retrieval.colbert import (
    DEFAULT_MANIFEST_ROOT,
    is_colbert_model_spec,
    maxsim_score,
    mean_pool_embeddings,
    resolve_colbert_descriptor,
)


def test_resolve_colbert_descriptor_from_registry():
    descriptor = resolve_colbert_descriptor("lfm2-colbert-350m", DEFAULT_MANIFEST_ROOT)

    assert descriptor.model_id == "lfm2-colbert-350m"
    assert descriptor.display_name == "LiquidAI/LFM2-ColBERT-350M"
    assert descriptor.weights_path == Path(
        "/home/local/ai/models/registry/LiquidAI/LFM2-ColBERT-350M"
    )
    assert descriptor.architecture == "Lfm2Model"
    assert descriptor.preferred_engine == "LLAMA_CPP"
    assert descriptor.query_prefix == "[Q] "
    assert descriptor.document_prefix == "[D] "
    assert descriptor.query_length == 32
    assert descriptor.document_length == 512
    assert descriptor.embedding_size == 128


def test_is_colbert_model_spec_accepts_manifest_id_and_path():
    weights_path = Path("/home/local/ai/models/registry/LiquidAI/LFM2-ColBERT-350M")

    assert is_colbert_model_spec("lfm2-colbert-350m", DEFAULT_MANIFEST_ROOT)
    assert is_colbert_model_spec(str(weights_path), DEFAULT_MANIFEST_ROOT)


def test_mean_pool_embeddings_and_maxsim_score():
    token_embeddings = torch.tensor(
        [[2.0, 0.0], [0.0, 2.0], [10.0, 10.0]],
        dtype=torch.float32,
    )
    token_mask = torch.tensor([True, True, False])
    pooled = mean_pool_embeddings(token_embeddings, token_mask)
    expected = torch.tensor([1.0, 1.0], dtype=torch.float32)
    expected = expected / torch.linalg.norm(expected)

    assert torch.allclose(pooled, expected, atol=1e-6)

    query_tokens = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    document_tokens = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        dtype=torch.float32,
    )
    assert maxsim_score(query_tokens, document_tokens) == 2.0
