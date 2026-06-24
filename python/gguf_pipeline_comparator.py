from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Mapping


_TEXT_SUFFIXES = {".py", ".rs", ".c", ".h", ".cpp", ".cc", ".md", ".toml", ".yaml", ".yml"}
_SKIP_DIRS = {".git", "target", "build", ".venv", "__pycache__", "node_modules"}

_GGUF_RE = re.compile(r"\bgguf\b", re.IGNORECASE)
_GGUF_LOADER_RE = re.compile(
    r"gguf[_\-\s]?(reader|loader|load|init|context)|load_gguf|GGUFReader",
    re.IGNORECASE,
)
_KV_RE = re.compile(
    r"kv[_\-\s]?cache|req_to_token|token_to_kv|paged[_\-\s]?kv|prefix[_\-\s]?cache",
    re.IGNORECASE,
)
_QUANT_MODE_RE = re.compile(r"\b(?:tq|rq)\d+\b", re.IGNORECASE)


@dataclass(frozen=True)
class EngineGgufSignals:
    engine: str
    repo_path: str
    gguf_mentions: int
    gguf_loader_mentions: int
    kv_cache_mentions: int
    rust_gguf_mentions: int
    quant_mode_mentions: int
    quant_modes: tuple[str, ...]

    @property
    def score(self) -> int:
        return (
            self.gguf_loader_mentions * 5
            + self.kv_cache_mentions * 3
            + self.quant_mode_mentions * 2
            + self.rust_gguf_mentions * 2
            + self.gguf_mentions
        )

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["score"] = self.score
        return payload


@dataclass(frozen=True)
class AssimilationStep:
    priority: int
    step_id: str
    source_engine: str
    target_module: str
    reason: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class GgufComparisonReport:
    engines: dict[str, EngineGgufSignals]
    ranking: list[str]
    assimilation_steps: list[AssimilationStep]

    def to_dict(self) -> dict[str, object]:
        return {
            "engines": {name: signal.to_dict() for name, signal in self.engines.items()},
            "ranking": self.ranking,
            "assimilation_steps": [step.to_dict() for step in self.assimilation_steps],
        }


def _iter_source_files(repo_path: Path) -> list[Path]:
    files: list[Path] = []
    for path in repo_path.rglob("*"):
        if not path.is_file():
            continue
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        if path.suffix.lower() in _TEXT_SUFFIXES:
            files.append(path)
    return files


def scan_engine(engine: str, repo_path: Path) -> EngineGgufSignals:
    gguf_mentions = 0
    gguf_loader_mentions = 0
    kv_cache_mentions = 0
    rust_gguf_mentions = 0
    quant_mode_mentions = 0
    quant_modes: set[str] = set()

    for source_path in _iter_source_files(repo_path):
        text = source_path.read_text(encoding="utf-8", errors="ignore")
        gguf_mentions += len(_GGUF_RE.findall(text))
        gguf_loader_mentions += len(_GGUF_LOADER_RE.findall(text))
        kv_cache_mentions += len(_KV_RE.findall(text))

        mode_hits = [hit.lower() for hit in _QUANT_MODE_RE.findall(text)]
        quant_mode_mentions += len(mode_hits)
        quant_modes.update(mode_hits)

        if source_path.suffix.lower() == ".rs":
            rust_gguf_mentions += len(_GGUF_RE.findall(text))

    return EngineGgufSignals(
        engine=engine,
        repo_path=str(repo_path),
        gguf_mentions=gguf_mentions,
        gguf_loader_mentions=gguf_loader_mentions,
        kv_cache_mentions=kv_cache_mentions,
        rust_gguf_mentions=rust_gguf_mentions,
        quant_mode_mentions=quant_mode_mentions,
        quant_modes=tuple(sorted(quant_modes)),
    )


def synthesize_atom_framework(engine_signals: Mapping[str, EngineGgufSignals]) -> list[AssimilationStep]:
    steps: list[AssimilationStep] = []
    atom = engine_signals["atom"]
    sglang = engine_signals["sglang"]
    llama = engine_signals["llama.cpp"]

    if llama.gguf_loader_mentions > atom.gguf_loader_mentions:
        steps.append(
            AssimilationStep(
                priority=1,
                step_id="atom-gguf-loader-core",
                source_engine="llama.cpp",
                target_module="rs_gguf_loader_core",
                reason="llama.cpp currently exhibits stronger GGUF loader/parser signal density than ATOM",
            )
        )

    if sglang.kv_cache_mentions > atom.kv_cache_mentions:
        steps.append(
            AssimilationStep(
                priority=2,
                step_id="atom-kv-runtime-bridge",
                source_engine="sglang",
                target_module="rs_kv_runtime_bridge",
                reason="SGLang currently exposes richer KV routing/scheduler/cache signal coverage than ATOM",
            )
        )

    if len(sglang.quant_modes) > len(atom.quant_modes):
        steps.append(
            AssimilationStep(
                priority=3,
                step_id="atom-quant-mode-router",
                source_engine="sglang",
                target_module="rs_quant_mode_router",
                reason="SGLang advertises broader quant mode vocabulary that should be normalized in ATOM",
            )
        )

    steps.append(
        AssimilationStep(
            priority=4,
            step_id="atom-backend-cutover-gates",
            source_engine="atom",
            target_module="rs_backend_cutover_gates",
            reason="Add capability-gated cutover boundaries so ATOM can replace SGLang backend safely",
        )
    )

    return sorted(steps, key=lambda step: step.priority)


def compare_engines(engine_paths: Mapping[str, Path]) -> GgufComparisonReport:
    required = {"sglang", "atom", "llama.cpp"}
    missing = required.difference(engine_paths.keys())
    if missing:
        raise ValueError(f"Missing required engines: {sorted(missing)}")

    signals = {
        name: scan_engine(name, path)
        for name, path in engine_paths.items()
    }
    ranking = [name for name, _ in sorted(signals.items(), key=lambda item: item[1].score, reverse=True)]
    steps = synthesize_atom_framework(signals)
    return GgufComparisonReport(engines=signals, ranking=ranking, assimilation_steps=steps)


def default_engine_paths() -> dict[str, Path]:
    return {
        "sglang": Path("/home/local/ai/projects/donors/sglang-1-bit-turbo"),
        "atom": Path("/home/local/ai/build/wip/gfxATOM-Rust"),
        "llama.cpp": Path("/home/local/ai/projects/donors/llama.cpp-1-bit-turbo"),
    }


def write_report_json(report: GgufComparisonReport, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
