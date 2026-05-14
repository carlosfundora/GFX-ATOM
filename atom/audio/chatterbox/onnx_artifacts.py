# SPDX-License-Identifier: Apache-2.0
"""ONNX artifact helpers for Chatterbox sidecar models.

The Chatterbox Hugging Face cache uses symlink-heavy snapshots and ONNX external
data files. These helpers keep source snapshots immutable and write optimized
artifacts to a separate model root.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

logger = logging.getLogger("atom.audio.chatterbox")

DEFAULT_Q8_OP_TYPES = ("MatMul", "Gemm", "Conv")
Q8_COMPONENTS = ("speech_encoder", "embed_tokens", "conditional_decoder", "language_model")


@dataclass(frozen=True)
class QuantizedComponent:
    component: str
    source_path: Path
    output_path: Path
    quantized: bool


def component_filename(component: str, variant: str = "fp16") -> str:
    normalized = variant.lower()
    if component == "language_model":
        if normalized in {"", "fp32"}:
            return "language_model.onnx"
        return f"language_model_{normalized}.onnx"
    if normalized == "q8":
        return f"{component}_q8.onnx"
    return f"{component}.onnx"


def component_candidates(onnx_dir: Path, component: str, variant: str = "fp16") -> list[Path]:
    normalized = variant.lower()
    candidates = [onnx_dir / component_filename(component, normalized)]
    if normalized == "q8":
        if component == "language_model":
            candidates.extend(
                [
                    onnx_dir / "language_model_fp16.onnx",
                    onnx_dir / "language_model.onnx",
                ]
            )
        else:
            candidates.append(onnx_dir / f"{component}.onnx")
    elif component == "language_model" and normalized != "fp32":
        candidates.append(onnx_dir / "language_model.onnx")
    elif component == "language_model" and normalized == "fp32":
        candidates.append(onnx_dir / "language_model_fp16.onnx")
    return candidates


def resolve_component_path(
    onnx_dir: Path,
    component: str,
    variant: str = "fp16",
    *,
    allow_fallback: bool = True,
) -> Path:
    candidates = component_candidates(onnx_dir, component, variant)
    for index, candidate in enumerate(candidates):
        if candidate.exists():
            if index > 0 and variant.lower() == "q8":
                logger.warning(
                    "Q8 Chatterbox component %s not found in %s; falling back to %s",
                    component,
                    onnx_dir,
                    candidate.name,
                )
            return candidate
        if not allow_fallback:
            break
    raise FileNotFoundError(
        f"No ONNX component for {component!r} variant {variant!r}; checked "
        + ", ".join(str(path) for path in candidates)
    )


def require_onnx_tooling():
    try:
        import onnx  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "The Chatterbox Q8 converter requires the optional "
            "`onnx` package. Add it to the gfxatom environment with uv-managed "
            "dependencies before running conversion."
        ) from exc


def check_onnx_model(model_path: Path) -> None:
    """Run ONNX checker by path so external data models are handled correctly."""
    require_onnx_tooling()
    import onnx

    onnx.checker.check_model(str(model_path))


def infer_shapes_to_path(model_path: Path, output_path: Path) -> Path:
    """Run ONNX shape inference without mutating the source model."""
    require_onnx_tooling()
    import onnx

    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.shape_inference.infer_shapes_path(str(model_path), str(output_path))
    return output_path


def quantize_dynamic_q8(
    source_path: Path,
    output_path: Path,
    *,
    op_types: Sequence[str] = DEFAULT_Q8_OP_TYPES,
    per_channel: bool = True,
    reduce_range: bool = True,
    use_external_data_format: bool = True,
    overwrite: bool = False,
) -> QuantizedComponent:
    """Create a dynamic QInt8 ONNX sidecar using ONNX Runtime quantization."""
    require_onnx_tooling()
    if output_path.exists() and not overwrite:
        return QuantizedComponent(
            component=output_path.stem.removesuffix("_q8"),
            source_path=source_path,
            output_path=output_path,
            quantized=False,
        )

    from onnxruntime.quantization import QuantType, quantize_dynamic

    output_path.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        model_input=source_path,
        model_output=output_path,
        op_types_to_quantize=list(op_types),
        per_channel=per_channel,
        reduce_range=reduce_range,
        weight_type=QuantType.QInt8,
        use_external_data_format=use_external_data_format,
    )
    return QuantizedComponent(
        component=output_path.stem.removesuffix("_q8"),
        source_path=source_path,
        output_path=output_path,
        quantized=True,
    )


def _copy_snapshot_metadata(source_model_dir: Path, output_model_dir: Path) -> None:
    output_model_dir.mkdir(parents=True, exist_ok=True)
    for child in source_model_dir.iterdir():
        if child.name == "onnx":
            continue
        target = output_model_dir / child.name
        if target.exists():
            continue
        if child.is_dir():
            shutil.copytree(child, target, symlinks=False)
        elif child.is_file() or child.is_symlink():
            shutil.copy2(child.resolve(), target)


def quantize_chatterbox_q8_sidecar(
    source_model_dir: Path,
    output_model_dir: Path,
    *,
    components: Iterable[str] = Q8_COMPONENTS,
    overwrite: bool = False,
) -> list[QuantizedComponent]:
    """Write a Q8 Chatterbox model root without modifying the source snapshot."""
    source_model_dir = Path(source_model_dir)
    output_model_dir = Path(output_model_dir)
    source_onnx_dir = source_model_dir / "onnx"
    output_onnx_dir = output_model_dir / "onnx"

    _copy_snapshot_metadata(source_model_dir, output_model_dir)
    output_onnx_dir.mkdir(parents=True, exist_ok=True)

    results: list[QuantizedComponent] = []
    for component in components:
        source_path = resolve_component_path(
            source_onnx_dir,
            component,
            "fp32" if component == "language_model" else "fp16",
        )
        output_path = output_onnx_dir / component_filename(component, "q8")
        results.append(
            quantize_dynamic_q8(
                source_path,
                output_path,
                overwrite=overwrite,
            )
        )
    return results
