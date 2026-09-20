"""Live mode: analyze each request as it is recorded and log problems."""

from __future__ import annotations

import logging
from typing import Any

from promptcachelint import findings as F
from promptcachelint.analyze import Analyzer
from promptcachelint.model import Record

_LEVEL = {"error": logging.WARNING, "warning": logging.WARNING, "info": logging.INFO}


class LogWatcher:
    """A :class:`Recorder` sink that logs findings the moment they appear.

    Runs the same :class:`~promptcachelint.analyze.Analyzer` as the offline report,
    so a warning in the log and a line in ``promptcachelint report`` never disagree,
    but without history: only the last request per session is kept, and idle
    sessions are dropped, so memory stays proportional to active conversations.
    Session ids come with the record (the recorder assigned them).
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.log = logger or logging.getLogger("promptcachelint")
        self.analyzer = Analyzer(history=False)

    def __call__(self, record: Record) -> None:
        report = self.analyzer.step(record)
        sid = record.session_id or "?"
        for f in report.findings:
            extra: dict[str, Any] = {"promptcachelint": f.to_dict(), "session": sid}
            where = f" at {f.path}+{f.offset}" if f.path else ""
            self.log.log(
                _LEVEL[f.severity],
                "promptcachelint %s %s%s%s",
                f.code,
                f.message,
                where,
                f" | {f.hint}" if f.hint else "",
                extra=extra,
            )
        u = record.usage
        if u is not None and not any(f.code == F.PREFIX_BROKEN for f in report.findings):
            self.log.debug(
                "promptcachelint session=%s read=%d write=%d uncached=%d hit=%.1f%%",
                sid[:12],
                u.cache_read,
                u.cache_write,
                u.input_tokens,
                u.hit_ratio * 100,
            )
