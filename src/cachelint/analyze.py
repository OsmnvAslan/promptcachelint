"""Turn a list of records into a report: sessions, diffs, findings, totals."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from cachelint import findings as F
from cachelint.detectors import structure, volatile_content
from cachelint.diff import PrefixDiff, diff_prefix
from cachelint.model import Record, Segment, Usage, estimate_tokens
from cachelint.providers import get_provider
from cachelint.providers.base import Provider
from cachelint.sessions import SessionIndex

TTL_SECONDS = {"5m": 5 * 60, "1h": 60 * 60}
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
    requests: list[RequestReport] = field(default_factory=list)

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

    def to_dict(self) -> dict[str, Any]:
        return {"totals": self.totals.to_dict(), "sessions": [s.to_dict() for s in self.sessions]}

    def to_text(self) -> str:
        from cachelint.render import render_text

        return render_text(self)


def analyze(records: Iterable[Record], *, index: SessionIndex | None = None) -> Report:
    """Group records into sessions, diff consecutive requests, collect findings."""
    index = index or SessionIndex()
    sessions: dict[str, SessionReport] = {}
    last: dict[str, RequestReport] = {}

    for record in sorted(records, key=lambda r: r.at):
        provider = get_provider(record.provider)
        segments = provider.segments(record.body)
        cacheable = provider.cacheable_segments(record.body, segments)
        sid = index.assign(record.provider, segments, record.at, explicit=record.session_id)
        session = sessions.setdefault(sid, SessionReport(id=sid, provider=record.provider))

        prev = last.get(sid)
        diff = None
        findings: list[F.Finding] = structure(provider, record.body, segments, cacheable)
        findings += volatile_content(segments, cacheable)
        if prev is not None:
            diff = diff_prefix(prev.segments, segments, prev.cacheable)
            findings += _session_findings(provider, prev, record, segments, cacheable, diff)

        report = RequestReport(record, len(session.requests), segments, cacheable, diff, findings)
        session.requests.append(report)
        last[sid] = report

    return Report(sessions=list(sessions.values()))


def _session_findings(
    provider: Provider,
    prev: RequestReport,
    record: Record,
    segments: list[Segment],
    cacheable: int,
    diff: PrefixDiff,
) -> list[F.Finding]:
    out: list[F.Finding] = []

    # Top-level scope changes (model, thinking, effort, cache key).
    prev_scope = provider.scope(prev.record.body)
    scope = provider.scope(record.body)
    for key in sorted(set(prev_scope) | set(scope)):
        if prev_scope.get(key) != scope.get(key):
            out.append(
                F.Finding(
                    code=F.SCOPE_CHANGED,
                    severity="error" if key == "model" else "warning",
                    message=f"{key} changed: {prev_scope.get(key)} -> {scope.get(key)}",
                    hint=(
                        "Caches are model-scoped; a model switch rebuilds everything."
                        if key == "model"
                        else "Thinking/effort/cache-key changes invalidate the messages cache. "
                        "Pin them per route."
                    ),
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
            out.append(
                F.Finding(
                    code=F.PREFIX_BROKEN,
                    severity="error",
                    message=(
                        f"prefix diverged at {diff.path} offset {diff.offset}; "
                        f"~{diff.lost_tokens_estimate} previously cached tokens re-billed "
                        "(estimate)"
                    ),
                    hint="Whatever changed here must be moved after the last breakpoint or frozen.",
                    path=diff.path,
                    offset=diff.offset,
                    excerpt=diff.after,
                    data={
                        "before": diff.before,
                        "after": diff.after,
                        "common_segments": diff.common_segments,
                    },
                )
            )

    if provider.name == "anthropic" and cacheable < prev.cacheable and not diff.broken:
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

    appended = len(segments) - diff.common_segments
    if provider.name == "anthropic" and not diff.broken and appended > LOOKBACK_POSITIONS:
        out.append(
            F.Finding(
                code=F.LOOKBACK_EXCEEDED,
                severity="warning",
                message=(
                    f"{appended} blocks appended since the previous request; the breakpoint "
                    f"lookback window is {LOOKBACK_POSITIONS} positions"
                ),
                hint="Place an intermediate breakpoint every ~15 positions in long turns.",
                data={"appended": appended},
            )
        )

    gap = record.at - prev.record.at
    ttl = _ttl_seconds(prev.segments)
    if gap > ttl and prev.cacheable > 0:
        out.append(
            F.Finding(
                code=F.TTL_GAP,
                severity="info",
                message=f"{gap:.0f}s since the previous request; cache TTL is {ttl}s",
                hint="Expect a write, not a read. Use ttl: '1h' for bursty traffic, or pre-warm.",
                data={"gap_seconds": round(gap, 1), "ttl_seconds": ttl},
            )
        )

    usage = record.usage
    if (
        usage is not None
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
        expected = estimate_tokens("x" * expected_chars) if expected_chars else 0
        if usage.cache_read == 0 and expected >= provider.min_prefix_tokens(record.model):
            scope_changed = any(f.code == F.SCOPE_CHANGED for f in out)
            if not scope_changed:
                out.append(
                    F.Finding(
                        code=F.UNEXPLAINED_MISS,
                        severity="warning",
                        message=(
                            f"prefix matched the previous request (~{expected} tokens, "
                            "estimate) but cache_read is 0"
                        ),
                        hint=(
                            "The payload looks identical; suspects are outside it: a different "
                            "workspace/API key, a server-side invalidator (thinking blocks "
                            "stripped on older models), or the previous write never landed "
                            "(prefix below minimum). On Anthropic, cache diagnostics can confirm."
                        ),
                        data={"expected_tokens_estimate": expected},
                    )
                )
    return out


def _ttl_seconds(segments: list[Segment]) -> int:
    ttl = "5m"
    for seg in segments:
        if seg.breakpoint and seg.ttl:
            ttl = seg.ttl
    return TTL_SECONDS.get(ttl, TTL_SECONDS["5m"])
