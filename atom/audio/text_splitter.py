# SPDX-License-Identifier: Apache-2.0
"""Multi-language sentence boundary detector for streaming TTS input.

Buffers incoming text and splits at sentence boundaries (English and CJK),
yielding complete sentences for audio generation.

Ported from vLLM-Omni's text_splitter.py.
"""

import logging

try:
    import rs_codec
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False

logger = logging.getLogger("atom.audio")

class SentenceSplitter:
    """Incremental sentence splitter for streaming text input.

    Buffers text and yields complete sentences when boundaries are detected.
    Designed for TTS pipelines where text arrives incrementally.
    Powered by rs_codec fast Rust implementation.
    """

    def __init__(self, min_sentence_length: int = 2, boundary_re=None) -> None:
        if boundary_re is not None:
            logger.warning("SentenceSplitter: boundary_re is ignored when using Rust backend.")
            
        if not _HAS_RUST:
            raise RuntimeError("rs_codec is required for SentenceSplitter but is not installed.")
            
        self._splitter = rs_codec.SentenceSplitter(min_sentence_length=min_sentence_length)

    @property
    def buffer(self) -> str:
        return self._splitter.buffer

    def add_text(self, text: str) -> list[str]:
        if not text:
            return []
        return self._splitter.add_text(text)

    def flush(self) -> str | None:
        return self._splitter.flush()

