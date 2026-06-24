from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from gguf_pipeline_comparator import compare_engines, scan_engine  # noqa: E402


def _write(path: Path, rel: str, content: str) -> None:
    target = path / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def test_scan_engine_detects_loader_quant_and_kv_signals(tmp_path: Path):
    repo = tmp_path / "sglang"
    _write(
        repo,
        "python/server_args.py",
        "kv_cache_dtype = 'tq2'\n# GGUFReader enabled\ngguf loader\nreq_to_token = True\n",
    )
    _write(repo, "python/memory_pool.py", "token_to_kv = {}\nkv cache\n")

    signal = scan_engine("sglang", repo)

    assert signal.gguf_mentions > 0
    assert signal.gguf_loader_mentions > 0
    assert signal.kv_cache_mentions > 0
    assert "tq2" in signal.quant_modes


def test_compare_engines_synthesizes_atom_assimilation_steps(tmp_path: Path):
    sglang = tmp_path / "sglang"
    atom = tmp_path / "atom"
    llama = tmp_path / "llama"

    _write(sglang, "runtime.py", "kv_cache_dtype='tq2'\nkv cache\nreq_to_token\ntoken_to_kv\n")
    _write(atom, "python/loader.py", "minimal loader\n")
    _write(llama, "ggml.c", "GGUFReader\nload_gguf\n")

    report = compare_engines({"sglang": sglang, "atom": atom, "llama.cpp": llama})
    step_ids = [step.step_id for step in report.assimilation_steps]

    assert "atom-gguf-loader-core" in step_ids
    assert "atom-kv-runtime-bridge" in step_ids
    assert "atom-backend-cutover-gates" in step_ids
