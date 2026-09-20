"""Recording requests: explicit API, JSONL sink, and loading traces back."""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

from promptcachelint.model import Record, Redactor, Usage
from promptcachelint.providers import get_provider
from promptcachelint.sessions import SessionIndex, current_session

Sink = Callable[[Record], None]


class JsonlSink:
    """Append one JSON object per record to a file (or any text stream).

    ``redact`` transforms the request body before it is written; see
    :mod:`promptcachelint.redact` for ``strip_media`` and ``hash_text``.
    """

    def __init__(self, target: str | Path | TextIO, *, redact: Redactor | None = None) -> None:
        self._file: TextIO | None = None
        self._path: Path | None = None
        self._redact = redact
        self._lock = threading.Lock()
        if isinstance(target, str | Path):
            self._path = Path(target)
        else:
            self._file = target

    def __call__(self, record: Record) -> None:
        line = json.dumps(record.to_dict(self._redact), ensure_ascii=False)
        with self._lock:
            if self._file is not None:
                self._file.write(line + "\n")
                self._file.flush()
            else:
                assert self._path is not None
                with self._path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")


class Recorder:
    """Collects records, assigns sessions, fans out to sinks and watchers.

    The last ``keep`` records stay in memory (``recorder.records``; default
    10 000, ``None`` for unlimited, ``0`` for a pure live setup that only feeds
    sinks) and every record goes to the sinks. Session ids come from
    ``promptcachelint.session(...)`` when active, otherwise from the automatic prefix
    grouping. Safe to share between threads.

    Recording runs on the caller's thread (for the transport: right after the
    response body is consumed). The cost is one pass over the request body
    plus an O(1) session lookup; for very large prompts on a latency-critical
    path, record from a worker thread instead.
    """

    def __init__(
        self, *sinks: Sink, index: SessionIndex | None = None, keep: int | None = 10_000
    ) -> None:
        self.records: deque[Record] = deque(maxlen=keep)
        self._sinks: list[Sink] = list(sinks)
        self.index = index or SessionIndex()
        self._lock = threading.Lock()

    def add_sink(self, sink: Sink) -> None:
        self._sinks.append(sink)

    def record(
        self,
        provider: str,
        body: dict[str, Any],
        *,
        usage: Usage | None = None,
        response: dict[str, Any] | None = None,
        sse_events: list[dict[str, Any]] | None = None,
        session_id: str | None = None,
        stream: bool = False,
        url: str | None = None,
        at: float | None = None,
        response_id: str | None = None,
        note: str | None = None,
    ) -> Record:
        """Record one request. Pass either ``usage`` or the raw ``response``/``sse_events``."""
        p = get_provider(provider)
        if usage is None and response is not None:
            usage = p.usage(response)
            if response_id is None and isinstance(response.get("id"), str):
                response_id = response["id"]
        if usage is None and sse_events:
            usage = p.usage_from_sse(sse_events)
        rec = Record(
            provider=provider,
            body=body,
            at=at if at is not None else time.time(),
            model=body.get("model") if isinstance(body.get("model"), str) else None,
            usage=usage,
            stream=stream,
            url=url,
            response_id=response_id,
            note=note,
        )
        explicit = session_id or current_session()
        segments = p.segments(body)
        rec.session_explicit = explicit is not None
        with self._lock:
            rec.session_id = self.index.assign(provider, segments, rec.at, explicit=explicit)
            if self.records.maxlen != 0:
                self.records.append(rec)
            for sink in self._sinks:
                sink(rec)
        return rec


@dataclass(slots=True)
class Trace:
    """Records read from a JSONL file plus the lines that could not be read."""

    records: list[Record] = field(default_factory=list)
    skipped: list[tuple[int, str]] = field(default_factory=list)  # (line number, reason)


def read_trace(source: str | Path | Iterable[str]) -> Trace:
    """Read a trace written by :class:`JsonlSink`, skipping lines that cannot be parsed.

    Traces are appended to and a process may die mid-line, so a broken last
    line is normal; an unknown provider or a record without a body is skipped
    too. Skipped lines are reported, never fatal.
    """
    from promptcachelint.providers import provider_names

    lines: Iterator[str]
    if isinstance(source, str | Path):
        lines = iter(Path(source).read_text(encoding="utf-8").splitlines())
    else:
        lines = iter(source)
    trace = Trace()
    known = set(provider_names())
    for number, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            trace.skipped.append((number, f"invalid JSON: {exc.msg}"))
            continue
        if not isinstance(data, dict) or not isinstance(data.get("body"), dict):
            trace.skipped.append((number, "no request body"))
            continue
        if data.get("provider") not in known:
            trace.skipped.append((number, f"unknown provider {data.get('provider')!r}"))
            continue
        try:
            trace.records.append(Record.from_dict(data))
        except (KeyError, TypeError, ValueError) as exc:
            trace.skipped.append((number, f"malformed record: {exc}"))
    return trace


def load_jsonl(source: str | Path | Iterable[str], *, strict: bool = False) -> list[Record]:
    """Records from a JSONL trace. With ``strict`` a bad line raises ``ValueError``."""
    trace = read_trace(source)
    if strict and trace.skipped:
        number, reason = trace.skipped[0]
        raise ValueError(f"line {number}: {reason}")
    return trace.records
