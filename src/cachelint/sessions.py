"""Grouping requests into sessions.

Two modes: explicit (``with cachelint.session("id"):`` or ``Record.session_id``)
and automatic. Automatic grouping is a heuristic and the report says so.

The automatic rule: a request continues a session when

* its **first message block** is byte-identical to the session's first message
  block (the *anchor*; unrelated conversations that share a system prompt
  differ here), and
* the tools/system head is the same or *nearly* the same (a timestamp or a
  version bump in the system prompt must keep the request in its session,
  otherwise the break it causes could never be seen).

Sessions are indexed by anchor, so assignment is O(1) in the number of open
sessions rather than a scan.
"""

from __future__ import annotations

import contextvars
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

from cachelint.model import Segment

_current: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "cachelint_session", default=None
)


@contextmanager
def session(session_id: str | None = None) -> Iterator[str]:
    """Tag every request recorded inside the block with ``session_id``."""
    sid = session_id or uuid.uuid4().hex
    token = _current.set(sid)
    try:
        yield sid
    finally:
        _current.reset(token)


def current_session() -> str | None:
    return _current.get()


# A head segment (tool/system) still "belongs" when most of its bytes match.
NEAR_MATCH = 0.6
MAX_OPEN_SESSIONS = 4096

AnchorKey = tuple[str, tuple[str, str | None, str | None, str] | None]


@dataclass(slots=True)
class _Open:
    id: str
    provider: str
    segments: list[Segment]
    last_at: float
    explicit: bool
    anchor: AnchorKey


def _split(segments: list[Segment]) -> tuple[list[Segment], list[Segment]]:
    head = [s for s in segments if s.kind != "message"]
    body = [s for s in segments if s.kind == "message"]
    return head, body


def _anchor(provider: str, segments: list[Segment]) -> AnchorKey:
    _, body = _split(segments)
    return (provider, body[0].key if body else None)


def _common_prefix(x: str, y: str) -> int:
    n = min(len(x), len(y))
    i = 0
    while i < n and x[i] == y[i]:
        i += 1
    return i


def _near(x: Segment, y: Segment) -> bool:
    if x.kind != y.kind or x.block != y.block:
        return False
    if x.text == y.text:
        return True
    longest = max(len(x.text), len(y.text)) or 1
    return _common_prefix(x.text, y.text) / longest >= NEAR_MATCH


def continues(prev: list[Segment], new: list[Segment]) -> bool:
    """True if ``new`` looks like the next request of the conversation ``prev`` came from."""
    prev_head, prev_body = _split(prev)
    new_head, new_body = _split(new)

    if prev_body and (not new_body or not prev_body[0].same(new_body[0])):
        return False

    if prev_head:
        # Align heads pairwise; tolerate near matches and a differing tail (tool added).
        matched = 0
        for x, y in zip(prev_head, new_head, strict=False):
            if not _near(x, y):
                break
            matched += 1
        if matched < max(1, (len(prev_head) + 1) // 2):
            return False
    return True


@dataclass(slots=True)
class SessionIndex:
    """Assigns a session id to each incoming request."""

    max_gap_seconds: float = 6 * 3600
    _open: dict[str, _Open] = field(default_factory=dict)
    _by_anchor: dict[AnchorKey, list[str]] = field(default_factory=dict)
    _order: list[str] = field(default_factory=list)

    def assign(
        self,
        provider: str,
        segments: list[Segment],
        at: float,
        explicit: str | None = None,
    ) -> str:
        if explicit is not None:
            self._touch(explicit, provider, segments, at, explicit=True)
            return explicit

        anchor = _anchor(provider, segments)
        for sid in reversed(self._by_anchor.get(anchor, [])):
            s = self._open[sid]
            if s.explicit or at - s.last_at > self.max_gap_seconds:
                continue
            if continues(s.segments, segments):
                self._touch(sid, provider, segments, at, explicit=False)
                return sid

        sid = uuid.uuid4().hex
        self._touch(sid, provider, segments, at, explicit=False)
        return sid

    def is_explicit(self, session_id: str) -> bool:
        s = self._open.get(session_id)
        return bool(s and s.explicit)

    def _touch(
        self, sid: str, provider: str, segments: list[Segment], at: float, *, explicit: bool
    ) -> None:
        anchor = _anchor(provider, segments)
        s = self._open.get(sid)
        if s is None:
            s = _Open(sid, provider, segments, at, explicit, anchor)
            self._open[sid] = s
            self._order.append(sid)
            self._by_anchor.setdefault(anchor, []).append(sid)
            self._evict()
            return
        if s.anchor != anchor:
            self._by_anchor[s.anchor].remove(sid)
            self._by_anchor.setdefault(anchor, []).append(sid)
            s.anchor = anchor
        s.segments = segments
        s.last_at = at

    def _evict(self) -> None:
        while len(self._order) > MAX_OPEN_SESSIONS:
            sid = self._order.pop(0)
            s = self._open.pop(sid)
            self._by_anchor[s.anchor].remove(sid)
