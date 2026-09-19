"""Grouping requests into sessions.

Two modes: explicit (``with cachelint.session("id"):`` or ``Record.session_id``)
and automatic, where a request joins the most recent session whose prefix it
continues. Automatic grouping is a heuristic and says so in the report.
"""

from __future__ import annotations

import contextvars
import difflib
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


@dataclass(slots=True)
class _Open:
    id: str
    provider: str
    segments: list[Segment]
    last_at: float
    explicit: bool


@dataclass(slots=True)
class SessionIndex:
    """Assigns a session id to each incoming request."""

    max_gap_seconds: float = 6 * 3600
    min_shared_fraction: float = 0.5
    _open: list[_Open] = field(default_factory=list)

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

        best: _Open | None = None
        best_score = 0.0
        for s in self._open:
            if s.provider != provider or s.explicit or at - s.last_at > self.max_gap_seconds:
                continue
            score = _similarity(s.segments, segments)
            if score <= 0 or score < max(1.0, len(s.segments) * self.min_shared_fraction):
                continue
            if score > best_score:
                best, best_score = s, score

        sid = best.id if best else uuid.uuid4().hex
        self._touch(sid, provider, segments, at, explicit=False)
        return sid

    def _touch(
        self, sid: str, provider: str, segments: list[Segment], at: float, *, explicit: bool
    ) -> None:
        for s in self._open:
            if s.id == sid:
                s.segments = segments
                s.last_at = at
                return
        self._open.append(_Open(sid, provider, segments, at, explicit))
        self._open = self._open[-256:]


# A modified segment still "belongs" to the same session when most of its bytes
# match: that is exactly the case of a prefix broken by a timestamp or a version
# bump, and the whole point is to keep such requests together so the break is seen.
_NEAR_MATCH = 0.6


def _similarity(a: list[Segment], b: list[Segment]) -> float:
    """How much of ``a`` reappears in ``b``: exact segments count 1, near matches too."""
    keys_a: list[tuple[str, str]] = [(x.kind, x.text) for x in a]
    keys_b: list[tuple[str, str]] = [(y.kind, y.text) for y in b]
    matcher = difflib.SequenceMatcher(None, keys_a, keys_b, autojunk=False)
    matched_a: set[int] = set()
    for block in matcher.get_matching_blocks():
        matched_a.update(range(block.a, block.a + block.size))
    score = float(len(matched_a))

    # One near match for each unmatched segment of ``a`` that has a same-kind
    # segment in ``b`` sharing most of its leading bytes.
    unmatched_b = [j for j, _ in enumerate(b) if j not in _matched_b(matcher)]
    for i, x in enumerate(a):
        if i in matched_a:
            continue
        for j in unmatched_b:
            y = b[j]
            if y.kind != x.kind:
                continue
            longest = max(len(x.text), len(y.text)) or 1
            if _common_prefix(x.text, y.text) / longest >= _NEAR_MATCH:
                score += 1.0
                unmatched_b.remove(j)
                break
    return score


def _matched_b(matcher: difflib.SequenceMatcher[tuple[str, str]]) -> set[int]:
    out: set[int] = set()
    for block in matcher.get_matching_blocks():
        out.update(range(block.b, block.b + block.size))
    return out


def _common_prefix(x: str, y: str) -> int:
    n = min(len(x), len(y))
    i = 0
    while i < n and x[i] == y[i]:
        i += 1
    return i
