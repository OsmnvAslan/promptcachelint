"""Turn records into a report: sessions, diffs, findings, totals.

:class:`Analyzer` is incremental (one record at a time) so the live watcher
and the offline report share exactly the same rules; :func:`analyze` runs it
over a whole trace.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from typing import Any

from cachelint import findings as F
from cachelint.detectors import structure, volatile_content
from cachelint.diff import PrefixDiff, diff_prefix
from cachelint.model import Record, Segment, Usage, estimate_tokens_from_chars
from cachelint.providers import get_provider
from cachelint.providers.anthropic import SCOPE_KEYS as ANTHROPIC_SCOPE
from cachelint.providers.base import Provider
from cachelint.providers.openai import SCOPE_KEYS as OPENAI_SCOPE
from cachelint.sessions import SessionIndex

LOOKBACK_POSITIONS = 20


@dataclass(slots=True)
class RequestReport:
    record: Record
    index: int
    segments: list[Segment]
    cacheable: int
    diff: PrefixDiff | None
    findings: list[F.Finding]

    @property
    def usage(self) -> Usage | None:
        return self.record.usage

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.record.id,
            "index": self.index,
            "at": self.record.at,
            "model": self.record.model,
            "segments": len(self.segments),
            "cacheable": self.cacheable,
            "usage": None
            if self.usage is None
            else {
                "input_tokens": self.usage.input_tokens,
                "cache_read": self.usage.cache_read,
                "cache_write": self.usage.cache_write,
                "output_tokens": self.usage.output_tokens,
            },
            "diff": None
            if self.diff is None
            else {
                "broken": self.diff.broken,
                "common_segments": self.diff.common_segments,
                "path": self.diff.path,
                "offset": self.diff.offset,
                "lost_tokens_estimate": self.diff.lost_tokens_estimate,
            },
            "findings": [f.to_dict() for f in self.findings],
        }


@dataclass(slots=True)
class SessionReport:
    id: str
    provider: str
    explicit: bool
    requests: list[RequestReport] = field(default_factory=list)
    #: Requests seen, including ones not kept when history is off.
    count: int = 0
    last_at: float = 0.0

    @property
    def findings(self) -> list[F.Finding]:
        return [f for r in self.requests for f in r.findings]

    @property
    def totals(self) -> Totals:
        return Totals.of(self.requests)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "provider": self.provider,
            "explicit": self.explicit,
            "totals": self.totals.to_dict(),
            "requests": [r.to_dict() for r in self.requests],
        }


@dataclass(frozen=True, slots=True)
class Totals:
    requests: int
    with_usage: int
    input_tokens: int
    cache_read: int
    cache_write: int
    breaks: int
    lost_tokens_estimate: int

    @property
    def prompt_tokens(self) -> int:
        return self.input_tokens + self.cache_read + self.cache_write

    @property
    def hit_ratio(self) -> float:
        return self.cache_read / self.prompt_tokens if self.prompt_tokens else 0.0

    @classmethod
    def of(cls, requests: Iterable[RequestReport]) -> Totals:
        reqs = list(requests)
        usages = [r.usage for r in reqs if r.usage is not None]
        return cls(
            requests=len(reqs),
            with_usage=len(usages),
            input_tokens=sum(u.input_tokens for u in usages),
            cache_read=sum(u.cache_read for u in usages),
            cache_write=sum(u.cache_write for u in usages),
            breaks=sum(1 for r in reqs if r.diff is not None and r.diff.broken),
            lost_tokens_estimate=sum(
                r.diff.lost_tokens_estimate for r in reqs if r.diff is not None and r.diff.broken
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "requests": self.requests,
            "with_usage": self.with_usage,
            "input_tokens": self.input_tokens,
            "cache_read": self.cache_read,
            "cache_write": self.cache_write,
            "prompt_tokens": self.prompt_tokens,
            "hit_ratio": round(self.hit_ratio, 4),
            "breaks": self.breaks,
            "lost_tokens_estimate": self.lost_tokens_estimate,
        }


@dataclass(slots=True)
class Report:
    sessions: list[SessionReport] = field(default_factory=list)

    @property
    def findings(self) -> list[F.Finding]:
        return [f for s in self.sessions for f in s.findings]

    @property
    def totals(self) -> Totals:
        return Totals.of(r for s in self.sessions for r in s.requests)

    @property
    def auto_grouped(self) -> bool:
        return any(not s.explicit for s in self.sessions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "totals": self.totals.to_dict(),
            "auto_grouped": self.auto_grouped,
            "sessions": [s.to_dict() for s in self.sessions],
        }

    def to_text(self) -> str:
        from cachelint.render import render_text

        return render_text(self)


MAX_TRACKED_SESSIONS = 4096
MAX_TRACKED_HEADS = 4096


class Analyzer:
    """Incremental analysis: feed records in time order, read the report any time.

    With ``history=True`` (the offline report) every request report is kept.
    With ``history=False`` (live mode) only the last report per session is
    kept for diffing, and sessions idle for longer than the index's
    ``max_gap_seconds`` are dropped, so memory is bounded by the number of
    *active* conversations, not by the number of requests ever seen.
    """

    def __init__(self, index: SessionIndex | None = None, *, history: bool = True) -> None:
        self.index = index or SessionIndex()
        self.history = history
        self.sessions: dict[str, SessionReport] = {}
        self._last: dict[str, RequestReport] = {}
        self._static_seen: dict[str, set[tuple[str, str | None]]] = {}
        # Last single-turn request per (provider, scope, tools+system head), across sessions.
        self._last_by_head: dict[tuple[Any, ...], RequestReport] = {}
        self._steps = 0

    def step(self, record: Record) -> RequestReport:
        provider = get_provider(record.provider)
        segments = provider.segments(record.body)
        cacheable = provider.cacheable_segments(record.body, segments)
        sid = self.index.assign(record.provider, segments, record.at, explicit=record.session_id)
        session = self.sessions.get(sid)
        if session is None:
            session = SessionReport(
                id=sid, provider=record.provider, explicit=record.session_explicit
            )
            self.sessions[sid] = session

        findings = self._static(sid, provider, record, segments, cacheable)
        prev = self._last.get(sid)
        diff = None
        if prev is not None:
            diff = diff_prefix(prev.segments, segments, prev.cacheable)
            diff = _soften_for_automatic_caching(provider, prev, record, diff)
            findings += session_findings(provider, prev, record, segments, cacheable, diff)

        report = RequestReport(record, session.count, segments, cacheable, diff, findings)
        if prev is None:
            findings += self._head_findings(provider, report)
        session.count += 1
        session.last_at = max(session.last_at, record.at)
        if self.history:
            session.requests.append(report)
        self._last[sid] = report

        self._steps += 1
        if not self.history and self._steps % 64 == 0:
            self._evict(record.at)
        return report

    def _evict(self, now: float) -> None:
        """Live mode: forget sessions idle past the gap window, and cap the rest."""
        gap = self.index.max_gap_seconds
        stale = {sid for sid, s in self.sessions.items() if now - s.last_at > gap}
        overflow = len(self.sessions) - len(stale) - MAX_TRACKED_SESSIONS
        if overflow > 0:
            live = sorted((s.last_at, sid) for sid, s in self.sessions.items() if sid not in stale)
            stale.update(sid for _, sid in live[:overflow])
        for sid in stale:
            self.sessions.pop(sid, None)
            self._last.pop(sid, None)
            self._static_seen.pop(sid, None)
        if len(self._last_by_head) > MAX_TRACKED_HEADS:
            for key in list(self._last_by_head)[: len(self._last_by_head) // 2]:
                del self._last_by_head[key]

    def _head_findings(self, provider: Provider, report: RequestReport) -> list[F.Finding]:
        """Cross-session checks for requests that share tools+system.

        Single-turn requests with different questions never share a session,
        but they share a head. Two shapes are diagnosable from the usage alone:

        * head unmarked, marker after the per-request part: every request
          writes, none reads (CL014);
        * head marked and identical, yet nothing is read: an unexplained miss
          on the head (CL005).
        """
        head = [s for s in report.segments if s.kind != "message"]
        if not head or not provider.explicit_markers:
            return []
        # Caches are model-scoped and thinking/effort changes invalidate them too,
        # so requests under a different scope are a different cache.
        scope = tuple(sorted(provider.scope(report.record.body).items()))
        key = (provider.name, scope, tuple(s.key for s in head))
        prev = self._last_by_head.get(key)
        self._last_by_head[key] = report
        u = report.usage
        pu = prev.usage if prev is not None else None
        if prev is None or u is None or pu is None:
            return []
        if not (
            u.cache_write > 0 and u.cache_read == 0 and pu.cache_write > 0 and pu.cache_read == 0
        ):
            return []
        gap = report.record.at - prev.record.at
        if gap > provider.ttl_seconds(prev.segments):
            return []
        if head[-1].breakpoint:
            expected = estimate_tokens_from_chars(sum(s.chars for s in head))
            if expected < provider.min_prefix_tokens(report.record.model):
                return []
            return [
                F.Finding(
                    code=F.UNEXPLAINED_MISS,
                    severity="warning",
                    message=(
                        f"tools+system are marked and identical to a previous request "
                        f"(~{expected} tokens, estimate) but cache_read is 0"
                    ),
                    hint=(
                        "The marked head should have been read. Suspects outside the payload: "
                        "a different workspace/API key, thinking/effort differences, or the "
                        "previous write never landed. On Anthropic, cache diagnostics can confirm."
                    ),
                    data={"expected_tokens_estimate": expected},
                )
            ]
        if report.cacheable > len(head):
            return [
                F.Finding(
                    code=F.WRITE_WITHOUT_READ,
                    severity="warning",
                    message=(
                        "requests with the same tools+system write cache but never read it back"
                    ),
                    hint=(
                        "The cache marker (explicit, or the automatic one on the last block) "
                        "lands after per-request content, so every request writes a distinct "
                        "entry. Add a marker on the last system block so the shared head is read."
                    ),
                )
            ]
        return []

    def _static(
        self,
        sid: str,
        provider: Provider,
        record: Record,
        segments: list[Segment],
        cacheable: int,
    ) -> list[F.Finding]:
        """Static findings, each reported once per session (same code and path)."""
        static = structure(provider, record.body, segments, cacheable)
        static += volatile_content(segments, cacheable)
        seen = self._static_seen.setdefault(sid, set())
        out: list[F.Finding] = []
        for f in static:
            key = (f.code, f.path)
            if key not in seen:
                seen.add(key)
                out.append(f)
        return out

    def report(self) -> Report:
        return Report(sessions=list(self.sessions.values()))


def analyze(records: Iterable[Record], *, index: SessionIndex | None = None) -> Report:
    """Group records into sessions, diff consecutive requests, collect findings."""
    analyzer = Analyzer(index)
    for record in sorted(records, key=lambda r: r.at):
        analyzer.step(record)
    return analyzer.report()


def _soften_for_automatic_caching(
    provider: Provider, prev: RequestReport, record: Record, diff: PrefixDiff
) -> PrefixDiff:
    """On providers without markers, a divergence in the tail is not a break.

    OpenAI serves whatever prefix still matches, in 128-token steps. A changed
    last message loses less than one step, and if the provider reports a read
    covering (nearly) the whole previous prefix, nothing was lost at all. A
    *partial* read that stays stable from request to request is not a pass:
    that is exactly what a recurring mid-prefix break looks like.
    """
    if provider.explicit_markers or not diff.broken:
        return diff
    if diff.lost_tokens_estimate < provider.cache_granularity_tokens:
        return replace(diff, broken=False)
    if record.usage is not None:
        expected = estimate_tokens_from_chars(sum(s.chars for s in prev.segments[: prev.cacheable]))
        if record.usage.cache_read >= expected - provider.cache_granularity_tokens > 0:
            return replace(diff, broken=False)
    return diff


def _window_slid(prev: list[Segment], new: list[Segment], diff: PrefixDiff) -> bool:
    """The break is at the first message block and that block now holds a later prev block."""
    prev_body = [s for s in prev if s.kind == "message"]
    new_body = [s for s in new if s.kind == "message"]
    if not prev_body or not new_body or diff.kind != "message":
        return False
    first_index = next(i for i, s in enumerate(prev) if s.kind == "message")
    if diff.segment_index != first_index:
        return False
    return any(p.same(new_body[0]) for p in prev_body[1:])


def _history_edited(prev: list[Segment], new: list[Segment], diff: PrefixDiff) -> bool:
    """The break is inside the message body, the opener matches and the body did not shrink."""
    prev_body = [s for s in prev if s.kind == "message"]
    new_body = [s for s in new if s.kind == "message"]
    if diff.kind != "message" or len(prev_body) < 2 or len(new_body) < len(prev_body):
        return False
    return prev_body[0].same(new_body[0])


def positions(segments: Iterable[Segment]) -> int:
    """Cache positions in a run of segments.

    Consecutive ``tool_use`` blocks count as one position, and so do
    consecutive ``tool_result`` blocks (Anthropic's lookback rule).
    """
    count = 0
    run: str | None = None
    for seg in segments:
        block = seg.block if seg.block in ("tool_use", "tool_result") else None
        if block is not None and block == run:
            continue
        run = block
        count += 1
    return count


def session_findings(
    provider: Provider,
    prev: RequestReport,
    record: Record,
    segments: list[Segment],
    cacheable: int,
    diff: PrefixDiff,
) -> list[F.Finding]:
    """Findings that need the previous request of the same session."""
    out: list[F.Finding] = []
    hints = ANTHROPIC_SCOPE if provider.name == "anthropic" else OPENAI_SCOPE

    prev_scope = provider.scope(prev.record.body)
    scope = provider.scope(record.body)
    for key in sorted(set(prev_scope) | set(scope)):
        if prev_scope.get(key) != scope.get(key):
            out.append(
                F.Finding(
                    code=F.SCOPE_CHANGED,
                    severity="error" if key == "model" else "warning",
                    message=f"{key} changed: {prev_scope.get(key)} -> {scope.get(key)}",
                    hint=hints.get(key, "Pin this parameter per route."),
                    data={"key": key, "before": prev_scope.get(key), "after": scope.get(key)},
                )
            )

    prev_tools = [s.text for s in prev.segments if s.kind == "tool"]
    tools = [s.text for s in segments if s.kind == "tool"]
    if diff.broken:
        if prev_tools != tools:
            out.append(
                F.Finding(
                    code=F.TOOLS_CHANGED,
                    severity="error",
                    message=(
                        f"tool definitions changed ({len(prev_tools)} -> {len(tools)} tools, "
                        f"first difference at {diff.path})"
                    ),
                    hint=(
                        "Tools render first; any add/remove/reorder invalidates the whole cache. "
                        "Keep the tool list fixed and sorted, or use mid-conversation tool "
                        "changes where the model supports them."
                    ),
                    path=diff.path,
                    offset=diff.offset,
                    excerpt=diff.after,
                    data={"before": diff.before, "after": diff.after},
                )
            )
        else:
            slid = _window_slid(prev.segments, segments, diff)
            edited = not slid and _history_edited(prev.segments, segments, diff)
            out.append(
                F.Finding(
                    code=F.PREFIX_BROKEN,
                    severity="error",
                    message=(
                        f"prefix diverged at {diff.path} offset {diff.offset}; "
                        f"~{diff.lost_tokens_estimate} previously cached tokens re-billed "
                        "(estimate)" + ("; the history window slid" if slid else "")
                    ),
                    hint=(
                        "The history window slid: the oldest turns were dropped, so the "
                        "message prefix is rewritten on every request and nothing after the "
                        "system prompt can be read. Keep history append-only (compact or "
                        "summarise only at a breakpoint you control), or accept that only the "
                        "head is cached."
                        if slid
                        else "A block inside the history was rewritten (typically an old tool "
                        "result truncated or a turn summarised). Every request after the "
                        "edit re-bills the whole tail. Edit history only at a breakpoint you "
                        "control, or leave old blocks as they were."
                        if edited
                        else "Whatever changed here must be moved after the last breakpoint "
                        "or frozen."
                    ),
                    path=diff.path,
                    offset=diff.offset,
                    excerpt=diff.after,
                    data={
                        "before": diff.before,
                        "after": diff.after,
                        "role": diff.role,
                        "common_segments": diff.common_segments,
                        "window_slid": slid,
                        "history_edited": edited,
                    },
                )
            )

    if provider.explicit_markers and cacheable < prev.cacheable and not diff.broken:
        out.append(
            F.Finding(
                code=F.BREAKPOINT_MOVED_BACK,
                severity="warning",
                message=(
                    f"last breakpoint moved from segment {prev.cacheable} to {cacheable}; "
                    "the tail written last time will not be read"
                ),
                hint="Keep the breakpoint on the last block of the newest turn as history grows.",
            )
        )

    appended = positions(segments[diff.common_segments :])
    if provider.explicit_markers and not diff.broken and appended > LOOKBACK_POSITIONS:
        out.append(
            F.Finding(
                code=F.LOOKBACK_EXCEEDED,
                severity="warning",
                message=(
                    f"{appended} positions appended since the previous request; the "
                    f"breakpoint lookback window is {LOOKBACK_POSITIONS}"
                ),
                hint="Place an intermediate breakpoint every ~15 positions in long turns.",
                data={"appended_positions": appended},
            )
        )

    gap = record.at - prev.record.at
    ttl = provider.ttl_seconds(prev.segments)
    if gap > ttl and prev.cacheable > 0:
        out.append(
            F.Finding(
                code=F.TTL_GAP,
                severity="info",
                message=f"{gap:.0f}s since the previous request; cache TTL is about {ttl}s",
                hint="Expect a write, not a read. Use ttl: '1h' for bursty traffic, or pre-warm."
                if provider.explicit_markers
                else "Entries live 5-10 minutes; expect the prefix to be recomputed.",
                data={"gap_seconds": round(gap, 1), "ttl_seconds": ttl},
            )
        )

    usage = record.usage
    tail_only = not diff.broken or diff.kind == "message"
    if (
        tail_only
        and provider.explicit_markers
        and usage is not None
        and prev.usage is not None
        and prev.cacheable > 0
        and usage.cache_write > 0
        and usage.cache_read == 0
        and prev.usage.cache_write > 0
        and prev.usage.cache_read == 0
    ):
        out.append(
            F.Finding(
                code=F.WRITE_WITHOUT_READ,
                severity="warning",
                message="consecutive requests write cache but never read it back",
                hint=(
                    "The breakpoint sits after per-request content, so every request writes "
                    "a distinct entry. Put the marker at the end of the shared part."
                ),
            )
        )

    if usage is not None and not diff.broken and prev.cacheable > 0 and gap <= ttl:
        expected_chars = sum(s.chars for s in prev.segments[: prev.cacheable])
        expected = estimate_tokens_from_chars(expected_chars)
        scope_changed = any(f.code == F.SCOPE_CHANGED for f in out)
        if (
            usage.cache_read == 0
            and expected >= provider.min_prefix_tokens(record.model)
            and not scope_changed
        ):
            out.append(
                F.Finding(
                    code=F.UNEXPLAINED_MISS,
                    severity="info" if provider.best_effort else "warning",
                    message=(
                        f"prefix matched the previous request (~{expected} tokens, "
                        "estimate) but cache_read is 0"
                    ),
                    hint=(
                        "The cache is best-effort; occasional misses on an intact prefix "
                        "are expected."
                        if provider.best_effort
                        else "The payload looks identical; suspects are outside it: a different "
                        "workspace/API key, a server-side invalidator (thinking blocks "
                        "stripped on older models), or the previous write never landed "
                        "(prefix below minimum). On Anthropic, cache diagnostics can confirm."
                    ),
                    data={"expected_tokens_estimate": expected},
                )
            )
    return out
