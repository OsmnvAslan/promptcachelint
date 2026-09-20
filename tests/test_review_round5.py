"""Regression tests for the fifth external review."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx2 as hx
import pytest

from promptcachelint import LogWatcher, Recorder, analyze, load_jsonl, read_trace
from promptcachelint.analyze import Analyzer
from promptcachelint.cli import main
from promptcachelint.redact import hash_text
from promptcachelint.transport import client
from tests.conftest import anthropic_body, rec

# 1. hash_text: short texts stay distinct


def test_hash_text_short_texts_do_not_collide() -> None:
    out = {t: hash_text({"text": t})["text"] for t in ["hi", "ok", "yes", "no!", "hello", "thank"]}
    assert all(len(v) == len(k) for k, v in out.items())
    assert len(set(out.values())) == len(out)
    assert not any(v.startswith("sha") for v in out.values())
    assert hash_text({"text": ""})["text"] == ""
    long = hash_text({"text": "x" * 200})["text"]
    assert len(long) == 200 and len(set(long)) > 1


def test_hash_text_keeps_sessions_apart_and_breaks_visible() -> None:
    a = anthropic_body(user="hi", mark_last=True)
    b = anthropic_body(user="ok", mark_last=True)
    plain = analyze([rec(a, 1.0), rec(b, 2.0)])
    hashed = analyze([rec(hash_text(a), 1.0), rec(hash_text(b), 2.0)])
    assert len(plain.sessions) == len(hashed.sessions) == 2

    h1 = anthropic_body(
        history=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        user="z",
        mark_last=True,
    )
    h2 = anthropic_body(
        history=[{"role": "user", "content": [{"type": "text", "text": "ok"}]}],
        user="z",
        mark_last=True,
    )
    plain = analyze([rec(h1, 1.0, session_id="s"), rec(h2, 2.0, session_id="s")])
    hashed = analyze(
        [rec(hash_text(h1), 1.0, session_id="s"), rec(hash_text(h2), 2.0, session_id="s")]
    )
    assert plain.totals.breaks == hashed.totals.breaks == 1


# 2. memory: live mode is bounded


def test_recorder_keep_bounds_records() -> None:
    r = Recorder(keep=3)
    for i in range(10):
        r.record("anthropic", anthropic_body(user=f"u{i}"))
    assert [x.body["messages"][0]["content"][0]["text"] for x in r.records] == ["u7", "u8", "u9"]
    none = Recorder(keep=0)
    none.record("anthropic", anthropic_body())
    assert len(none.records) == 0
    unlimited = Recorder(keep=None)
    for i in range(20_001 if False else 5):
        unlimited.record("anthropic", anthropic_body(user=f"u{i}"))
    assert len(unlimited.records) == 5


def test_live_watcher_keeps_one_report_per_active_session_and_drops_idle_ones() -> None:
    watcher = LogWatcher()
    r = Recorder(watcher, keep=0)
    t = 0.0
    for i in range(4000):
        r.record("anthropic", anthropic_body(user=f"u{i}", mark_last=True), at=t)
        t += 1.0
    kept = sum(len(s.requests) for s in watcher.analyzer.sessions.values())
    assert kept == 0  # no history in live mode
    assert len(watcher.analyzer._last) <= 4000

    # A long-running conversation keeps only its last report.
    convo = Recorder(watcher, keep=0)
    history: list[dict[str, Any]] = []
    for i in range(50):
        convo.record(
            "anthropic", anthropic_body(history=history, user=f"q{i}", mark_last=True), at=t
        )
        history += [
            {"role": "user", "content": [{"type": "text", "text": f"q{i}"}]},
            {"role": "assistant", "content": "a"},
        ]
        t += 1.0
    sid = convo.records[-1].session_id if convo.records else None
    assert sid is None  # keep=0
    assert all(s.count >= 1 and len(s.requests) == 0 for s in watcher.analyzer.sessions.values())

    # Idle sessions are evicted once past the gap window.
    far = t + 7 * 3600
    for i in range(70):
        convo.record("anthropic", anthropic_body(user=f"late{i}", mark_last=True), at=far + i)
    assert len(watcher.analyzer.sessions) <= 70 + 64
    assert len(watcher.analyzer._last) == len(watcher.analyzer.sessions)


def test_offline_analyzer_still_keeps_history() -> None:
    a = Analyzer()
    for i in range(3):
        a.step(rec(anthropic_body(user=f"u{i}"), float(i)))
    assert sum(len(s.requests) for s in a.sessions.values()) == 3


# 3. broken trace lines


def test_read_trace_skips_bad_lines_and_counts_them(tmp_path: Path) -> None:
    good = json.dumps({"provider": "anthropic", "body": anthropic_body(), "at": 1})
    lines = [
        good,
        json.dumps({"provider": "martian", "body": anthropic_body(), "at": 2}),
        json.dumps({"provider": "anthropic", "at": 3}),
        '{"provider": "anth',
    ]
    path = tmp_path / "t.jsonl"
    path.write_text("\n".join(lines) + "\n")
    trace = read_trace(path)
    assert len(trace.records) == 1
    assert [n for n, _ in trace.skipped] == [2, 3, 4]
    assert "unknown provider" in trace.skipped[0][1]
    assert "no request body" in trace.skipped[1][1]
    assert "invalid JSON" in trace.skipped[2][1]
    assert len(load_jsonl(path)) == 1
    with pytest.raises(ValueError):
        load_jsonl(path, strict=True)


def test_cli_reports_skipped_lines_and_exit_codes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "t.jsonl"
    path.write_text(
        json.dumps({"provider": "anthropic", "body": anthropic_body(), "at": 1})
        + '\n{"provider": "anth'
    )
    assert main(["report", str(path)]) == 0
    out = capsys.readouterr().out
    assert "1 line(s) skipped" in out and "invalid JSON" in out
    assert main(["report", str(path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["skipped_lines"][0]["line"] == 2

    path.write_text('{"provider": "anth\n{"x":')
    assert main(["report", str(path)]) == 2
    assert "no records" in capsys.readouterr().err


# minor: aborted stream is recorded with a note


def test_unconsumed_stream_is_recorded_with_a_note() -> None:
    def handler(request: hx.Request) -> hx.Response:
        return hx.Response(
            200, headers={"content-type": "text/event-stream"}, content=b"data: {}\n\n"
        )

    r = Recorder()
    with (
        client(r, transport=hx.MockTransport(handler)) as c,
        c.stream(
            "POST",
            "https://api.anthropic.com/v1/messages",
            json=anthropic_body() | {"stream": True},
        ) as resp,
    ):
        resp.close()  # never read a byte
    assert len(r.records) == 1
    assert r.records[0].usage is None and r.records[0].stream
    assert r.records[0].note and "never consumed" in r.records[0].note
    assert "never consumed" in analyze(r.records).to_text()
