"""Redaction helpers for traces that leave the machine.

A trace holds full prompts. Before it goes into a CI artifact or a bug report,
strip what does not need to be there:

* :func:`strip_media` drops base64 payloads (images, PDFs) and keeps their size,
  so prefix diffs still work on text and the file stays small.
* :func:`hash_text` replaces every text with its SHA-256 digest repeated or cut
  to the original length in characters. Character counts, token estimates and
  size-based findings stay correct, different texts stay different, and the
  report can still tell *which block* changed and how big it was, but not the
  bytes, so offsets and excerpts are meaningless. Use it when the content itself
  is confidential.
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
    """The SHA-256 hex digest of ``value``, repeated or cut to its exact character length.

    No readable prefix: two different short texts differ from their first
    characters on, so redacted openers still separate sessions and a break in a
    short block is still a break. Length is preserved in *characters*, not
    bytes, because token estimates count characters.
    """
    if not value:
        return value
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    reps = len(value) // len(digest) + 1
    return (digest * reps)[: len(value)]


def hash_text(body: dict[str, Any]) -> dict[str, Any]:
    """Replace text content with a same-length hex placeholder; structure is kept."""
    keys = {"text", "content", "input_text", "output_text", "instructions", "system", "description"}

    def fn(key: str, value: Any) -> Any:
        if key in keys and isinstance(value, str):
            return _placeholder(value)
        if key == "data" and isinstance(value, str) and len(value) >= _BASE64_MIN:
            return f"<{len(value)} base64 chars redacted>"
        return value

    return cast(dict[str, Any], _walk(copy.deepcopy(body), fn))


__all__ = ["hash_text", "strip_media"]
