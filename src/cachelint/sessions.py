"""Grouping requests into sessions.

Two modes: explicit (``with cachelint.session("id"):`` or ``Record.session_id``)
and automatic. Automatic grouping is a heuristic and the report says so.

The automatic rule: a request continues a session when its **first message
block** is byte-identical to the session's (the *anchor*) and either

* the previous message body is a prefix of the new one (the conversation
  grew: that is the signature of a continuation whatever happened to the
  tools or the system prompt), or
* the body is repeated exactly (a retry).

Anything else with the same opener is another conversation: two users who
both start with "hi" stay apart from their second turn on.

Two more shapes are recognised so that the rewrite they cause is reported
rather than hidden:

* **sliding-window history**: no session anchors on the new first block, but
  one has it among its first ``ANCHOR_BLOCKS`` message blocks (the oldest
  turns were dropped);
* **edited history**: same opener, head near, body not shorter, and most of
  the old blocks still present in order (old tool results truncated, a turn
  summarised).

Sessions are indexed by hashes of their first message blocks, so assignment is
O(1) in the number of open sessions.
"""

from __future__ import annotations

import contextvars
import hashlib
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
#: How many leading message blocks are indexed: the depth at which a sliding
#: history window is still recognised (one agent exchange is four blocks).
ANCHOR_BLOCKS = 16
#: Share of the previous history that must reappear, in order, for a request
#: with the same opener to count as that history *edited* rather than another
#: conversation.
EDITED_MIN_SHARE = 0.6


@dataclass(slots=True)
class _Open:
    id: str
    provider: str
    segments: list[Segment]
    last_at: float
    explicit: bool
    anchors: list[int]  # hashes of the first ANCHOR_BLOCKS message blocks

    @property
    def anchor0(self) -> int | None:
        return self.anchors[0] if self.anchors else None


def split(segments: list[Segment]) -> tuple[list[Segment], list[Segment]]:
    """(tools+system head, message body)."""
    head = [s for s in segments if s.kind != "message"]
    body = [s for s in segments if s.kind == "message"]
    return head, body


def _hash(provider: str, seg: Segment) -> int:
    """Deterministic across processes, unlike ``hash()``, so an index could be persisted."""
    h = hashlib.blake2b(digest_size=16)
    for part in (provider, seg.kind, seg.role or "", seg.block or "", seg.text):
        h.update(part.encode("utf-8"))
        h.update(b"\0")
    return int.from_bytes(h.digest(), "big")


def _anchors(provider: str, segments: list[Segment]) -> list[int]:
    _, body = split(segments)
    return [_hash(provider, s) for s in body[:ANCHOR_BLOCKS]]


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


def head_near(prev_head: list[Segment], new_head: list[Segment]) -> bool:
    """At least half of the previous head reappears, pairwise, allowing near matches."""
    if not prev_head:
        return True
    matched = 0
    for x, y in zip(prev_head, new_head, strict=False):
        if not _near(x, y):
            break
        matched += 1
    return matched >= max(1, (len(prev_head) + 1) // 2)


def continues(prev: list[Segment], new: list[Segment]) -> bool:
    """True if ``new`` is the next request of the conversation ``prev`` came from."""
    prev_head, prev_body = split(prev)
    new_head, new_body = split(new)

    if prev_body:
        if not new_body or len(new_body) < len(prev_body):
            return False
        # Grew or repeated: a continuation whatever happened to the head.
        return all(p.same(n) for p, n in zip(prev_body, new_body, strict=False))
    return head_near(prev_head, new_head)


def truncated(prev: list[Segment], new: list[Segment]) -> bool:
    """True if ``new`` looks like ``prev`` with its oldest message blocks dropped."""
    prev_head, prev_body = split(prev)
    new_head, new_body = split(new)
    if not new_body or len(prev_body) < 2:
        return False
    first = new_body[0]
    for k in range(1, min(len(prev_body), ANCHOR_BLOCKS)):
        if prev_body[k].same(first):
            return head_near(prev_head, new_head)
    return False


def edited(prev: list[Segment], new: list[Segment]) -> bool:
    """True if ``new`` is ``prev``'s conversation with some history blocks rewritten.

    Same opener, head near, body not shorter, and most of the old blocks still
    present in order (old tool results truncated, a turn summarised). Two
    conversations that merely share an opener diverge on every later block and
    fail the share test.
    """
    prev_head, prev_body = split(prev)
    new_head, new_body = split(new)
    if len(prev_body) < 2 or len(new_body) < len(prev_body):
        return False
    if not prev_body[0].same(new_body[0]) or not head_near(prev_head, new_head):
        return False
    j = 0
    matched = 0
    for p in prev_body:
        k = j
        while k < len(new_body) and not p.same(new_body[k]):
            k += 1
        if k < len(new_body):  # found later on: keep it, skip what was rewritten
            matched += 1
            j = k + 1
    return matched / len(prev_body) >= EDITED_MIN_SHARE


@dataclass(slots=True)
class SessionIndex:
    """Assigns a session id to each incoming request."""

    max_gap_seconds: float = 6 * 3600
    _open: dict[str, _Open] = field(default_factory=dict)
    _by_anchor: dict[int, list[str]] = field(default_factory=dict)
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

        anchors = _anchors(provider, segments)
        first = anchors[0] if anchors else None
        candidates = list(reversed(self._by_anchor.get(first, []))) if first is not None else []

        for sid in candidates:
            s = self._open[sid]
            if s.explicit or at - s.last_at > self.max_gap_seconds or s.anchor0 != first:
                continue
            if continues(s.segments, segments):
                self._touch(sid, provider, segments, at, explicit=False)
                return sid

        for sid in candidates:
            s = self._open[sid]
            if s.explicit or at - s.last_at > self.max_gap_seconds or s.anchor0 == first:
                continue
            if truncated(s.segments, segments):
                self._touch(sid, provider, segments, at, explicit=False)
                return sid

        for sid in candidates:
            s = self._open[sid]
            if s.explicit or at - s.last_at > self.max_gap_seconds or s.anchor0 != first:
                continue
            if edited(s.segments, segments):
                self._touch(sid, provider, segments, at, explicit=False)
                return sid

        if first is None:
            # No messages at all: continue the latest head-only session with a near head.
            for sid in reversed(self._order):
                s = self._open[sid]
                if (
                    not s.explicit
                    and s.provider == provider
                    and not s.anchors
                    and at - s.last_at <= self.max_gap_seconds
                    and continues(s.segments, segments)
                ):
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
        anchors = _anchors(provider, segments)
        s = self._open.get(sid)
        if s is None:
            s = _Open(sid, provider, segments, at, explicit, anchors)
            self._open[sid] = s
            self._order.append(sid)
            self._index(sid, anchors)
            self._evict()
            return
        if s.anchors != anchors:
            self._unindex(sid, s.anchors)
            self._index(sid, anchors)
            s.anchors = anchors
        s.segments = segments
        s.last_at = at

    def _index(self, sid: str, anchors: list[int]) -> None:
        for h in set(anchors):
            self._by_anchor.setdefault(h, []).append(sid)

    def _unindex(self, sid: str, anchors: list[int]) -> None:
        for h in set(anchors):
            ids = self._by_anchor.get(h)
            if ids and sid in ids:
                ids.remove(sid)
                if not ids:
                    del self._by_anchor[h]

    def _evict(self) -> None:
        while len(self._order) > MAX_OPEN_SESSIONS:
            sid = self._order.pop(0)
            s = self._open.pop(sid)
            self._unindex(sid, s.anchors)
