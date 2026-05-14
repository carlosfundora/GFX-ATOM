# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path
from unittest.mock import MagicMock

# Stub out missing/heavy dependencies before importing the module under test.
# This allows the tests to run in environments without GPU dependencies.
for mod in [
    "numpy",
    "soundfile",
    "fastapi",
    "fastapi.responses",
    "pydantic",
    "torch",
    "safetensors",
    "safetensors.torch",
    "atom.audio.protocol",
    "atom.audio.utils",
]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

# Handle atom/__init__.py dependencies
for mod in [
    "atom.model_engine.llm_engine",
    "atom.sampling_params",
    "atom.plugin",
]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

from atom.entrypoints.openai.serving_speech import _validate_path_within_directory


class TestValidatePathWithinDirectory:
    """Unit tests for the _validate_path_within_directory security helper."""

    def test_path_within_directory(self, tmp_path: Path):
        directory = tmp_path / "data"
        directory.mkdir()
        file_path = directory / "voice.wav"
        assert _validate_path_within_directory(file_path, directory) is True

    def test_path_is_directory_itself(self, tmp_path: Path):
        assert _validate_path_within_directory(tmp_path, tmp_path) is True

    def test_path_traversal_attempt(self, tmp_path: Path):
        directory = tmp_path / "safe"
        directory.mkdir()
        # Path("/tmp/safe/../unsafe.txt").resolve() -> Path("/tmp/unsafe.txt")
        file_path = directory / ".." / "unsafe.txt"
        assert _validate_path_within_directory(file_path, directory) is False

    def test_path_outside_directory(self, tmp_path: Path):
        directory = tmp_path / "safe"
        directory.mkdir()
        other_dir = tmp_path / "other"
        other_dir.mkdir()
        file_path = other_dir / "voice.wav"
        assert _validate_path_within_directory(file_path, directory) is False

    def test_similar_prefix_directory(self, tmp_path: Path):
        # Ensure it doesn't just do a string prefix match
        directory = tmp_path / "base"
        directory.mkdir()
        other_dir = tmp_path / "base_extended"
        other_dir.mkdir()
        file_path = other_dir / "voice.wav"
        assert _validate_path_within_directory(file_path, directory) is False

    def test_nested_directory(self, tmp_path: Path):
        directory = tmp_path / "base"
        directory.mkdir()
        subdir = directory / "subdir" / "nested"
        subdir.mkdir(parents=True)
        file_path = subdir / "voice.wav"
        assert _validate_path_within_directory(file_path, directory) is True

    def test_exception_handling(self):
        # Passing None or invalid types should be caught by the try-except
        assert _validate_path_within_directory(None, Path("/tmp")) is False # type: ignore
        assert _validate_path_within_directory(Path("/tmp"), None) is False # type: ignore
