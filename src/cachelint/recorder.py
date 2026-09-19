"""Recording requests: explicit API, JSONL sink, and loading traces back."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any, TextIO

from cachelint.model import Record, Usage
from cachelint.providers import get_provider
from cachelint.sessions import SessionIndex, current_session

Sink = Callable[[Record], None]


class JsonlSink:
    """Append one JSON object per record to a file (or any text stream)."""

    def __init__(self, target: str | Path | TextIO) -> None:
        self._file: TextIO | None = None
        self._path: Path | None = None
        if isinstance(target, str | Path):
            self._path = Path(target)
        else:
            self._file = target

    def __call__(self, record: Record) -> None:
        line = json.dumps(record.to_dict(), ensure_ascii=False)
        if self._file is not None:
            self._file.write(line + "\n")
            self._file.flush()
        else:
            assert self._path is not None
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")


class Recorder:
    """Collects records, assigns sessions, fans out to sinks and watchers.

    Records are kept in memory (``recorder.records``) and, optionally, written
    to sinks. Session ids come from ``cachelint.session(...)`` when active,
    otherwise from the automatic prefix grouping.
    """

    def __init__(self, *sinks: Sink, index: SessionIndex | None = None) -> None:
        self.records: list[Record] = []
        self._sinks: list[Sink] = list(sinks)
        self.index = index or SessionIndex()

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
        )
        explicit = session_id or current_session()
        rec.session_id = self.index.assign(provider, p.segments(body), rec.at, explicit=explicit)
        self.records.append(rec)
        for sink in self._sinks:
            sink(rec)
        return rec


def load_jsonl(source: str | Path | Iterable[str]) -> list[Record]:
    """Read records written by :class:`JsonlSink`."""
    lines: Iterator[str]
    if isinstance(source, str | Path):
        lines = iter(Path(source).read_text(encoding="utf-8").splitlines())
    else:
        lines = iter(source)
    out: list[Record] = []
    for line in lines:
        line = line.strip()
        if line:
            out.append(Record.from_dict(json.loads(line)))
    return out
