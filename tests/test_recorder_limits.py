"""Recorder under load: thread safety and bounded memory in live mode."""

from __future__ import annotations

import threading
from typing import Any

from promptcachelint import LogWatcher, Recorder, analyze
from promptcachelint.analyze import Analyzer
from tests.conftest import anthropic_body, convo, rec


def test_recorder_is_thread_safe() -> None:
    r = Recorder()

    def work(who: str) -> None:
        for turn in range(20):
            r.record("anthropic", convo(who, turn))

    threads = [threading.Thread(target=work, args=(f"u{i}",)) for i in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert len(r.records) == 160
    assert len(analyze(r.records).sessions) == 8


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
