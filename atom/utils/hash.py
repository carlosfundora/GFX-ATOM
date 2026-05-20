# SPDX-License-Identifier: Apache-2.0

import hashlib

def stable_hash(content: str | bytes) -> str:
    """Universal stable hash wrapper, backed by fast rust xxhash if available."""
    try:
        import atom_rust
        if isinstance(content, bytes):
            return atom_rust.compute_bytes_hash(content)
        return atom_rust.compute_string_hash(str(content))
    except (ImportError, AttributeError):
        # Fallback to python hashlib (MD5 for speed/compatibility)
        if isinstance(content, str):
            content = content.encode("utf-8")
        return hashlib.md5(content, usedforsecurity=False).hexdigest()
