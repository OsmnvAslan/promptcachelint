"""Live mode: analyze each request as it is recorded and log problems."""

from __future__ import annotations

import logging
from typing import Any

from cachelint import findings as F
from cachelint.analyze import RequestReport
from cachelint.detectors import structure, volatile_content
from cachelint.diff import diff_prefix
from cachelint.model import Record
from cachelint.providers import get_provider

_LEVEL = {"error": logging.WARNING, "warning": logging.WARNING, "info": logging.INFO}


class LogWatcher:
    """A :class:`Recorder` sink that logs findings the moment they appear.

    Keeps the last request per session so it can diff without re-reading the
    whole trace. Static findings (``CL01x``) are logged once per session to
    avoid repeating the same warning on every turn.
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.log = logger or logging.getLogger("cachelint")
        self._last: dict[str, RequestReport] = {}
        self._static_seen: dict[str, set[str]] = {}

    def __call__(self, record: Record) -> None:
        provider = get_provider(record.provider)
        segments = provider.segments(record.body)
        cacheable = provider.cacheable_segments(record.body, segments)
        sid = record.session_id or "?"

        findings = structure(provider, record.body, segments, cacheable) + volatile_content(
            segments, cacheable
        )
        seen = self._static_seen.setdefault(sid, set())
        findings = [f for f in findings if f.code not in seen]
        seen.update(f.code for f in findings)

        prev = self._last.get(sid)
        diff = None
        if prev is not None:
            from cachelint.analyze import _session_findings

            diff = diff_prefix(prev.segments, segments, prev.cacheable)
            findings += _session_findings(provider, prev, record, segments, cacheable, diff)

        report = RequestReport(
            record, prev.index + 1 if prev else 0, segments, cacheable, diff, findings
        )
        self._last[sid] = report

        for f in findings:
            extra: dict[str, Any] = {"cachelint": f.to_dict(), "session": sid}
            where = f" at {f.path}+{f.offset}" if f.path else ""
            self.log.log(
                _LEVEL[f.severity],
                "cachelint %s %s%s%s",
                f.code,
                f.message,
                where,
                f" | {f.hint}" if f.hint else "",
                extra=extra,
            )
        if record.usage is not None and not any(f.code == F.PREFIX_BROKEN for f in findings):
            u = record.usage
            self.log.debug(
                "cachelint session=%s read=%d write=%d uncached=%d hit=%.1f%%",
                sid[:12],
                u.cache_read,
                u.cache_write,
                u.input_tokens,
                u.hit_ratio * 100,
            )
