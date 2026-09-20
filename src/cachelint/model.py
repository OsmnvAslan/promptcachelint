"""Core data types shared by every module."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

SegmentKind = Literal["tool", "system", "message"]


@dataclass(frozen=True, slots=True)
class Segment:
    """One unit of the cacheable prefix, in the order the provider renders it.

    ``text`` is the block's own text (or a canonical JSON rendering of a
    non-text block) *without* cache markers, so offsets reported against it
    are offsets into the block the caller wrote. ``role`` and ``block`` carry
    the message role and block type separately; two segments are the same
    prefix element iff :attr:`key` is equal.
    """

    path: str
    kind: SegmentKind
    text: str
    role: str | None = None
    block: str | None = None
    breakpoint: bool = False
    ttl: str | None = None

    @property
    def chars(self) -> int:
        return len(self.text)

    @property
    def key(self) -> tuple[str, str | None, str | None, str]:
        return (self.kind, self.role, self.block, self.text)

    def same(self, other: Segment) -> bool:
        return self.key == other.key


@dataclass(frozen=True, slots=True)
class Usage:
    """Provider-reported token accounting for one request, normalized.

    ``input_tokens`` is the *uncached* remainder only; the full prompt size is
    ``input_tokens + cache_read + cache_write``.
    """

    input_tokens: int
    cache_read: int = 0
    cache_write: int = 0
    output_tokens: int | None = None

    @property
    def prompt_tokens(self) -> int:
        return self.input_tokens + self.cache_read + self.cache_write

    @property
    def hit_ratio(self) -> float:
        total = self.prompt_tokens
        return self.cache_read / total if total else 0.0


Redactor = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass(slots=True)
class Record:
    """One captured request (and, when available, its response usage)."""

    provider: str
    body: dict[str, Any]
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    at: float = field(default_factory=time.time)
    model: str | None = None
    usage: Usage | None = None
    session_id: str | None = None
    #: True when the id came from the caller (``cachelint.session`` / ``session_id=``),
    #: False when the recorder assigned it by prefix similarity.
    session_explicit: bool = False
    stream: bool = False
    url: str | None = None
    response_id: str | None = None

    def to_dict(self, redact: Redactor | None = None) -> dict[str, Any]:
        d = asdict(self)
        if redact is not None:
            d["body"] = redact(self.body)
        if self.usage is None:
            d["usage"] = None
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Record:
        usage = d.get("usage")
        return cls(
            provider=d["provider"],
            body=d["body"],
            id=d.get("id") or uuid.uuid4().hex,
            at=float(d.get("at") or 0.0),
            model=d.get("model"),
            usage=Usage(**usage) if usage else None,
            session_id=d.get("session_id"),
            session_explicit=bool(d.get("session_explicit", False)),
            stream=bool(d.get("stream", False)),
            url=d.get("url"),
            response_id=d.get("response_id"),
        )


CHARS_PER_TOKEN = 4


def estimate_tokens_from_chars(chars: int) -> int:
    """Rough token estimate (~4 chars per token). Always labelled as an estimate."""
    return (chars + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN if chars > 0 else 0


def estimate_tokens(text: str) -> int:
    return estimate_tokens_from_chars(len(text))
