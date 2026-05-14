# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import logging
import os
from os import PathLike

logger = logging.getLogger("atom")


def _python_find_files(root: str) -> list[str]:
    files: list[str] = []
    for dirpath, _, filenames in os.walk(root, followlinks=True):
        for filename in filenames:
            files.append(os.path.join(dirpath, filename))
    return files


def find_files(root: str | PathLike[str]) -> list[str]:
    """Return recursive file paths, using Rust when the extension is available."""
    root_str = os.fspath(root)
    try:
        import atom_rust

        rust_find_files = getattr(atom_rust, "find_files")
    except (ImportError, AttributeError):
        return _python_find_files(root_str)

    try:
        return list(rust_find_files(root_str))
    except Exception as exc:
        logger.warning("Failed to use Rust find_files: %s. Falling back to os.walk.", exc)
        return _python_find_files(root_str)
