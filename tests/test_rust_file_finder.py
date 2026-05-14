import builtins
import os
from pathlib import Path

import pytest

from atom.utils import file_finder


def _relative_files(paths: list[str], root: Path) -> set[str]:
    return {Path(path).relative_to(root).as_posix() for path in paths}


def _write_tree(root: Path) -> None:
    (root / "nested").mkdir(parents=True)
    (root / "nested" / "model.py").write_text("x = 1\n", encoding="utf-8")
    (root / ".hidden.json").write_text("{}", encoding="utf-8")

    external = root / "external"
    external.mkdir()
    (external / "linked_config.json").write_text("{}", encoding="utf-8")
    os.symlink(external, root / "linked_external", target_is_directory=True)


def test_atom_rust_find_files_recurses_and_follows_symlinks(tmp_path: Path) -> None:
    atom_rust = pytest.importorskip("atom_rust")
    root = tmp_path / "root"
    root.mkdir()
    _write_tree(root)

    found = _relative_files(atom_rust.find_files(str(root)), root)

    assert "nested/model.py" in found
    assert ".hidden.json" in found
    assert "linked_external/linked_config.json" in found


def test_file_finder_python_fallback_on_import_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    _write_tree(root)

    real_import = builtins.__import__

    def fail_atom_rust_import(name, *args, **kwargs):
        if name == "atom_rust":
            raise ImportError("forced fallback")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_atom_rust_import)

    found = _relative_files(file_finder.find_files(root), root)

    assert "nested/model.py" in found
    assert ".hidden.json" in found
    assert "linked_external/linked_config.json" in found


def test_file_finder_python_fallback_on_rust_runtime_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    _write_tree(root)

    class BrokenAtomRust:
        @staticmethod
        def find_files(_root: str) -> list[str]:
            raise OSError("forced runtime failure")

    real_import = builtins.__import__

    def import_broken_atom_rust(name, *args, **kwargs):
        if name == "atom_rust":
            return BrokenAtomRust
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_broken_atom_rust)

    found = _relative_files(file_finder.find_files(root), root)

    assert "nested/model.py" in found
    assert ".hidden.json" in found
    assert "linked_external/linked_config.json" in found


def test_find_files_handles_huggingface_snapshot_shape(tmp_path: Path) -> None:
    atom_rust = pytest.importorskip("atom_rust")
    cache_root = tmp_path / "huggingface"
    snapshot = (
        cache_root
        / "models--ResembleAI--chatterbox-turbo-ONNX"
        / "snapshots"
        / "d21799bd0354adb85e348b8a0442a8405110a2cf"
    )
    blob_dir = cache_root / "models--ResembleAI--chatterbox-turbo-ONNX" / "blobs"
    onnx_dir = snapshot / "onnx"
    blob_dir.mkdir(parents=True)
    onnx_dir.mkdir(parents=True)

    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    model_blob = blob_dir / "model-blob"
    model_blob.write_text("onnx", encoding="utf-8")
    os.symlink(model_blob, onnx_dir / "language_model_fp16.onnx")

    found = _relative_files(atom_rust.find_files(str(snapshot)), snapshot)

    assert "config.json" in found
    assert "onnx/language_model_fp16.onnx" in found
