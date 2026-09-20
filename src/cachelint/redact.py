"""Redaction helpers for traces that leave the machine.

A trace holds full prompts. Before it goes into a CI artifact or a bug report,
strip what does not need to be there:

* :func:`strip_media` drops base64 payloads (images, PDFs) and keeps their size,
  so prefix diffs still work on text and the file stays small.
* :func:`hash_text` replaces every text with a hash placeholder padded to the
  original length. Character counts, token estimates and size-based findings
  stay correct; the report can still tell *which block* changed and how big it
  was, but not the bytes, so offsets and excerpts are meaningless. Use it when
  the content itself is confidential.
"""

from __future__ import annotations

import copy
import hashlib
from typing import Any, cast

_BASE64_MIN = 256


def _walk(value: Any, fn: Any) -> Any:
    if isinstance(value, dict):
        return {k: _walk(fn(k, v), fn) for k, v in value.items()}
    if isinstance(value, list):
        return [_walk(item, fn) for item in value]
    return value


def strip_media(body: dict[str, Any]) -> dict[str, Any]:
    """Replace long base64 ``data`` strings with a size placeholder."""

    def fn(key: str, value: Any) -> Any:
        if key == "data" and isinstance(value, str) and len(value) >= _BASE64_MIN:
            return f"<{len(value)} base64 chars redacted>"
        return value

    return cast(dict[str, Any], _walk(copy.deepcopy(body), fn))


def _placeholder(value: str) -> str:
    """``sha256:<hex>`` padded (or cut) to the exact length of ``value``."""
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    tag = f"sha256:{digest[:12]}"
    if len(value) <= len(tag):
        return tag[: len(value)]
    return tag.ljust(len(value))


def hash_text(body: dict[str, Any]) -> dict[str, Any]:
    """Replace text content with a same-length ``sha256:<hex>`` placeholder; structure is kept."""
    keys = {"text", "content", "input_text", "output_text", "instructions", "system", "description"}

    def fn(key: str, value: Any) -> Any:
        if key in keys and isinstance(value, str):
            return _placeholder(value)
        if key == "data" and isinstance(value, str) and len(value) >= _BASE64_MIN:
            return f"<{len(value)} base64 chars redacted>"
        return value

    return cast(dict[str, Any], _walk(copy.deepcopy(body), fn))


__all__ = ["hash_text", "strip_media"]
