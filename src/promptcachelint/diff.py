"""Prefix diff between two consecutive requests of one session."""

from __future__ import annotations

from dataclasses import dataclass

from promptcachelint.model import Segment, estimate_tokens_from_chars

EXCERPT = 60


@dataclass(frozen=True, slots=True)
class PrefixDiff:
    """Where the new request stops matching the previous one.

    ``broken`` is True only when the divergence lies *inside* the part of the
    previous request that the provider could have served from cache. Extra
    segments appended after that part are the normal growth of a conversation.
    ``offset`` is a character offset into the block's own text.
    """

    common_segments: int
    prev_cacheable: int
    broken: bool
    segment_index: int | None = None
    path: str | None = None
    kind: str | None = None
    role: str | None = None
    offset: int | None = None
    before: str = ""
    after: str = ""
    lost_chars: int = 0

    @property
    def lost_tokens_estimate(self) -> int:
        return estimate_tokens_from_chars(self.lost_chars)


def _common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _excerpt(text: str, offset: int) -> str:
    start = max(0, offset - EXCERPT // 3)
    end = min(len(text), offset + EXCERPT)
    piece = text[start:end]
    if start > 0:
        piece = "…" + piece
    if end < len(text):
        piece = piece + "…"
    return piece


def _lost_chars(prev: list[Segment], i: int, offset: int, prev_cacheable: int) -> int:
    """Characters that were cached last time and must be re-billed now.

    With explicit breakpoints (Anthropic) reads can only land *at* a marker, so
    everything after the last intact marker before the break is lost, even the
    part of the broken segment before ``offset``. Without markers (automatic
    prefix caching) the loss starts at the byte that changed.
    """
    if any(seg.breakpoint for seg in prev):
        start = 0
        for k in range(i):
            if prev[k].breakpoint:
                start = k + 1
        return sum(seg.chars for seg in prev[start:prev_cacheable])
    return sum(seg.chars for seg in prev[i:prev_cacheable]) - offset


def diff_prefix(prev: list[Segment], new: list[Segment], prev_cacheable: int) -> PrefixDiff:
    """Compare ``new`` against ``prev``; ``prev_cacheable`` = cacheable segment count of prev."""
    n = min(len(prev), len(new))
    i = 0
    while i < n and prev[i].same(new[i]):
        i += 1

    if i >= prev_cacheable:
        return PrefixDiff(common_segments=i, prev_cacheable=prev_cacheable, broken=False)

    if i >= len(new):
        # The new request is shorter than the cached prefix: everything from here is gone.
        return PrefixDiff(
            common_segments=i,
            prev_cacheable=prev_cacheable,
            broken=True,
            segment_index=i,
            path=prev[i].path,
            kind=prev[i].kind,
            role=prev[i].role,
            offset=0,
            before=_excerpt(prev[i].text, 0),
            after="",
            lost_chars=_lost_chars(prev, i, 0, prev_cacheable),
        )

    a, b = prev[i], new[i]
    aligned = a.kind == b.kind and a.role == b.role and a.block == b.block
    offset = _common_prefix_len(a.text, b.text) if aligned else 0
    return PrefixDiff(
        common_segments=i,
        prev_cacheable=prev_cacheable,
        broken=True,
        segment_index=i,
        path=b.path if aligned else a.path,
        kind=a.kind,
        role=a.role,
        offset=offset,
        before=_excerpt(a.text, offset),
        after=_excerpt(b.text, offset),
        lost_chars=_lost_chars(prev, i, offset, prev_cacheable),
    )
