from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger("atom")

DEFAULT_MANIFEST_ROOT = Path(
    os.environ.get(
        "ATOM_MODEL_MANIFEST_ROOT",
        "/home/local/ai/projects/ENCOM/reg/model/manifests",
    )
)
DEFAULT_WEIGHTS_ROOT = Path(
    os.environ.get(
        "ATOM_MODEL_WEIGHTS_ROOT",
        "/home/local/ai/models/registry",
    )
)
DEFAULT_DEVICE = os.environ.get("ATOM_COLBERT_DEVICE", "cpu")


class ColbertConfigError(ValueError):
    """Raised when a ColBERT model cannot be resolved."""


@dataclass(frozen=True)
class ColbertDescriptor:
    """Resolved metadata for a ColBERT-capable model."""

    model_id: str
    display_name: str
    manifest_dir: Path
    weights_path: Path
    architecture: str
    preferred_engine: str | None
    roles: tuple[str, ...]
    query_prefix: str
    document_prefix: str
    query_length: int
    document_length: int
    hidden_size: int
    embedding_size: int


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _manifest_matches(config: dict[str, Any], model_spec: str) -> bool:
    model_id = str(config.get("id", ""))
    model_name = str(config.get("name", ""))
    return model_spec in {model_id, model_name, model_name.rsplit("/", 1)[-1]}


def _resolve_manifest_dir(
    model_spec: str, manifest_root: Path = DEFAULT_MANIFEST_ROOT
) -> Path | None:
    if not manifest_root.exists():
        return None
    for config_path in manifest_root.rglob("config.json"):
        try:
            config = _read_json(config_path)
        except json.JSONDecodeError:
            continue
        if _manifest_matches(config, model_spec):
            return config_path.parent
    return None


def _resolve_weights_path(
    model_spec: str, manifest_dir: Path | None, config: dict[str, Any], runtime: dict[str, Any]
) -> Path:
    runtime_path = runtime.get("paths", {}).get("weights")
    if runtime_path:
        return Path(runtime_path)

    if manifest_dir is not None:
        local_path = config.get("paths", {}).get("local")
        if local_path:
            candidate = Path(local_path)
            if candidate.is_absolute():
                return candidate
            return DEFAULT_WEIGHTS_ROOT / candidate

    candidate = Path(model_spec)
    if candidate.is_dir():
        return candidate

    raise ColbertConfigError(f"Unable to resolve weights path for {model_spec!r}")


def resolve_colbert_descriptor(
    model_spec: str, manifest_root: Path = DEFAULT_MANIFEST_ROOT
) -> ColbertDescriptor:
    """Resolve a model spec or local path into a ColBERT descriptor."""

    candidate = Path(model_spec)
    manifest_dir: Path | None = None
    config: dict[str, Any]
    runtime: dict[str, Any]
    metadata: dict[str, Any]

    if candidate.is_dir():
        manifest_dir = candidate
    else:
        manifest_dir = _resolve_manifest_dir(model_spec, manifest_root)

    if manifest_dir is None:
        raise ColbertConfigError(
            f"Model {model_spec!r} is not registered as a local ColBERT manifest"
        )

    config = _read_json(manifest_dir / "config.json")
    runtime = _read_json(manifest_dir / "runtime.json") if (manifest_dir / "runtime.json").exists() else {}
    metadata = _read_json(manifest_dir / "metadata.json") if (manifest_dir / "metadata.json").exists() else {}

    weights_path = _resolve_weights_path(model_spec, manifest_dir, config, runtime)
    if not weights_path.exists():
        raise ColbertConfigError(f"Weights path does not exist: {weights_path}")

    sentence_cfg_path = weights_path / "config_sentence_transformers.json"
    if not sentence_cfg_path.exists():
        legacy_sentence_cfg_path = weights_path / "sentence_bert_config.json"
        if not legacy_sentence_cfg_path.exists():
            raise ColbertConfigError(
                f"Missing sentence-transformers config at {sentence_cfg_path}"
            )
        sentence_cfg_path = legacy_sentence_cfg_path

    sentence_cfg = _read_json(sentence_cfg_path)
    architecture = str(metadata.get("specs", {}).get("architecture", config.get("architecture", "Lfm2Model")))
    preferred_engine = runtime.get("preferred_engine")
    roles = tuple(str(role) for role in config.get("roles", []))
    display_name = str(metadata.get("name") or config.get("id") or model_spec)
    hidden_size = int(metadata.get("specs", {}).get("hidden_size", 0) or 1024)
    embedding_size = int(_read_json(weights_path / "1_Dense" / "config.json").get("out_features", 128))

    return ColbertDescriptor(
        model_id=str(config.get("id", model_spec)),
        display_name=display_name,
        manifest_dir=manifest_dir,
        weights_path=weights_path,
        architecture=architecture,
        preferred_engine=preferred_engine,
        roles=roles,
        query_prefix=str(sentence_cfg.get("query_prefix", "")),
        document_prefix=str(sentence_cfg.get("document_prefix", "")),
        query_length=int(sentence_cfg.get("query_length", 32)),
        document_length=int(sentence_cfg.get("document_length", 512)),
        hidden_size=hidden_size,
        embedding_size=embedding_size,
    )


def is_colbert_model_spec(
    model_spec: str, manifest_root: Path = DEFAULT_MANIFEST_ROOT
) -> bool:
    try:
        resolve_colbert_descriptor(model_spec, manifest_root=manifest_root)
        return True
    except ColbertConfigError:
        candidate = Path(model_spec)
        return candidate.is_dir() and (candidate / "sentence_bert_config.json").exists()


def mean_pool_embeddings(
    token_embeddings: torch.Tensor, token_mask: torch.Tensor
) -> torch.Tensor:
    """Mean-pool token embeddings across the unmasked tokens."""

    if token_embeddings.ndim != 2:
        raise ValueError("token_embeddings must be 2D")
    if token_mask.ndim != 1:
        raise ValueError("token_mask must be 1D")
    if token_embeddings.shape[0] != token_mask.shape[0]:
        raise ValueError("token_embeddings and token_mask must match")

    mask = token_mask.to(dtype=token_embeddings.dtype).unsqueeze(-1)
    masked = token_embeddings * mask
    denom = mask.sum().clamp_min(1.0)
    pooled = masked.sum(dim=0) / denom
    return F.normalize(pooled, p=2, dim=0)


def maxsim_score(query_tokens: torch.Tensor, document_tokens: torch.Tensor) -> float:
    """Compute ColBERT MaxSim using L2-normalized token embeddings."""

    if query_tokens.ndim != 2 or document_tokens.ndim != 2:
        raise ValueError("query_tokens and document_tokens must both be 2D")
    if query_tokens.shape[1] != document_tokens.shape[1]:
        raise ValueError("query_tokens and document_tokens must have the same width")
    if query_tokens.shape[0] == 0 or document_tokens.shape[0] == 0:
        raise ValueError("query_tokens and document_tokens must be non-empty")

    similarity = query_tokens @ document_tokens.T
    return float(similarity.max(dim=1).values.sum().item())


class ColbertService:
    """CPU-first ColBERT runtime backed by the local HF model files."""

    def __init__(self, descriptor: ColbertDescriptor, device: str = DEFAULT_DEVICE):
        self.descriptor = descriptor
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ColbertConfigError("Requested CUDA device for ColBERT, but CUDA is unavailable")

        logger.info(
            "Loading ColBERT model %s from %s",
            descriptor.display_name,
            descriptor.weights_path,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(descriptor.weights_path),
            local_files_only=True,
            trust_remote_code=False,
        )
        self.model = AutoModel.from_pretrained(
            str(descriptor.weights_path),
            local_files_only=True,
            trust_remote_code=False,
        ).to(self.device)
        self.model.eval()

        dense_state = load_file(
            str(descriptor.weights_path / "1_Dense" / "model.safetensors"),
            device="cpu",
        )
        dense_weight = dense_state["linear.weight"]
        self.projection = torch.nn.Linear(
            dense_weight.shape[1], dense_weight.shape[0], bias=False
        ).to(self.device)
        with torch.no_grad():
            self.projection.weight.copy_(dense_weight.to(self.device))
        self.projection.eval()

    @classmethod
    def from_model(
        cls,
        model_spec: str,
        manifest_root: Path = DEFAULT_MANIFEST_ROOT,
        device: str = DEFAULT_DEVICE,
    ) -> "ColbertService":
        return cls(resolve_colbert_descriptor(model_spec, manifest_root=manifest_root), device=device)

    @property
    def model_id(self) -> str:
        return self.descriptor.display_name

    def _encode_batch(
        self, texts: Sequence[str], *, role: str
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        if role == "query":
            prefix = self.descriptor.query_prefix
            max_length = self.descriptor.query_length
        elif role == "document":
            prefix = self.descriptor.document_prefix
            max_length = self.descriptor.document_length
        else:
            raise ValueError(f"Unsupported ColBERT role: {role}")

        prepared = [f"{prefix}{text}" for text in texts]
        encoded = self.tokenizer(
            prepared,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            return_special_tokens_mask=True,
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = self.model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
            )

        token_embeddings = self.projection(outputs.last_hidden_state)
        token_embeddings = F.normalize(token_embeddings, p=2, dim=-1)
        token_mask = encoded["attention_mask"].bool() & ~encoded["special_tokens_mask"].bool()
        tokens_evaluated = int(token_mask.sum().item())
        return token_embeddings, token_mask, tokens_evaluated

    def embed_texts(self, texts: Sequence[str]) -> tuple[list[list[float]], int]:
        if not texts:
            return [], 0

        token_embeddings, token_mask, tokens_evaluated = self._encode_batch(
            texts, role="document"
        )
        embeddings: list[list[float]] = []
        for row, mask in zip(token_embeddings, token_mask, strict=True):
            selected = row[mask]
            if selected.numel() == 0:
                raise ColbertConfigError("Encountered an empty embedding after masking")
            embeddings.append(
                F.normalize(selected.mean(dim=0), p=2, dim=0).tolist()
            )
        return embeddings, tokens_evaluated

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_n: int | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        if not documents:
            return [], 0

        query_tokens, query_mask, query_tokens_evaluated = self._encode_batch(
            [query], role="query"
        )
        document_tokens, document_mask, document_tokens_evaluated = self._encode_batch(
            documents, role="document"
        )

        q_tokens = query_tokens[0][query_mask[0]]
        if q_tokens.numel() == 0:
            raise ColbertConfigError("Query produced no usable tokens")

        scored: list[tuple[int, float, str]] = []
        for index, (document, row, mask) in enumerate(
            zip(documents, document_tokens, document_mask, strict=True)
        ):
            d_tokens = row[mask]
            if d_tokens.numel() == 0:
                raise ColbertConfigError(f"Document at index {index} produced no usable tokens")
            score = maxsim_score(q_tokens, d_tokens)
            scored.append((index, score, document))

        scored.sort(key=lambda item: (-item[1], item[0]))
        if top_n is not None:
            if top_n < 1:
                raise ColbertConfigError("top_n must be >= 1")
            scored = scored[: min(top_n, len(scored))]

        results = [
            {"index": index, "score": score, "document": document}
            for index, score, document in scored
        ]
        tokens_evaluated = query_tokens_evaluated + document_tokens_evaluated
        return results, tokens_evaluated
