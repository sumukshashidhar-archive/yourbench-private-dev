"""Utilities for splitting text into token-based chunks."""

from __future__ import annotations
import logging
from dataclasses import dataclass
from collections.abc import Callable

import tiktoken


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ChunkSettings:
    """Configuration for :func:`split_into_token_chunks`."""

    chunk_tokens: int = 1024
    overlap: int = 100
    encoding_name: str = "cl100k_base"
    preprocess: Callable[[str], str] | None = None

    def __post_init__(self) -> None:  # pragma: no cover - simple validation
        if self.chunk_tokens <= 0:
            msg = "chunk_tokens must be positive"
            logger.error(msg)
            raise ValueError(msg)
        if self.overlap < 0:
            msg = "overlap cannot be negative"
            logger.error(msg)
            raise ValueError(msg)
        if self.overlap >= self.chunk_tokens:
            msg = "overlap must be smaller than chunk_tokens"
            logger.error(msg)
            raise ValueError(msg)


def split_into_token_chunks(text: str, *, settings: ChunkSettings | None = None) -> list[str]:
    """Return ``text`` split into token-based chunks."""

    settings = settings or ChunkSettings()

    if settings.preprocess:
        try:
            text = settings.preprocess(text)
        except Exception as exc:  # pragma: no cover - user-defined
            logger.error("Preprocessing failed: %s", exc)
            raise

    try:
        enc = tiktoken.get_encoding(settings.encoding_name)
    except Exception as exc:  # pragma: no cover - external library
        logger.error("Invalid encoding '%s': %s", settings.encoding_name, exc)
        raise

    tokens = enc.encode(text)
    logger.debug("Tokenized text into %d tokens", len(tokens))

    stride = settings.chunk_tokens - settings.overlap
    chunk_indices = range(0, len(tokens), stride)
    return [enc.decode(tokens[i : i + settings.chunk_tokens]) for i in chunk_indices]
