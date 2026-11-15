"""How requests are grouped into conversations when no session id is given."""

from __future__ import annotations

import time
from typing import Any

from promptcachelint import LogWatcher, Recorder, analyze
from promptcachelint import findings as F
from promptcachelint.providers.anthropic import AnthropicProvider
from promptcachelint.sessions import SessionIndex, _hash, continues, edited
from tests.conftest import LONG, anthropic_body, convo, outcome, rec, tool

P = AnthropicProvider()


def test_interleaved_conversations_with_a_shared_system_prompt_stay_apart() -> None:
    r = Recorder()
    t = 0.0
    for turn in range(3):
        for who in ("alice", "bob"):
            r.record("anthropic", convo(who, turn), at=t)
            t += 1
    report = analyze(r.records)
    assert len(report.sessions) == 2
    assert report.totals.breaks == 0
    assert report.auto_grouped
    assert "grouped automatically" in report.to_text()


def test_fifty_parallel_dialogs_group_correctly_and_fast() -> None:
    r = Recorder()
    t = 0.0
    for turn in range(40):
        for d in range(50):
            r.record("anthropic", convo(f"d{d}", turn), at=t)
            t += 1
    t0 = time.perf_counter()
    report = analyze(r.records)
    elapsed = time.perf_counter() - t0
    assert len(report.sessions) == 50
    assert report.totals.breaks == 0
    assert elapsed < 10.0  # generous: CI runners are slow; the count is the real assertion


def test_a_timestamp_in_the_system_prompt_keeps_the_conversation_together() -> None:
    a = convo("x", 0, system=LONG + " Now: 2026-09-20T10:15:00Z")
    b = convo("x", 1, system=LONG + " Now: 2026-09-20T10:16:00Z")
    report = analyze([rec(a, 1.0), rec(b, 2.0)])
    assert len(report.sessions) == 1
    assert F.PREFIX_BROKEN in {f.code for f in report.findings}


def test_tool_added_mid_conversation_stays_in_session_and_is_reported() -> None:
    a = convo("x", 0, tools=[tool("search")])
    b = convo("x", 1, tools=[tool("search"), tool("calc")])
    report = analyze([rec(a, 1.0), rec(b, 2.0)])
    assert len(report.sessions) == 1
    assert F.TOOLS_CHANGED in {f.code for f in report.findings}


def test_tool_reorder_mid_conversation_is_reported() -> None:
    a = convo("x", 0, tools=[tool("search"), tool("calc")])
    b = convo("x", 1, tools=[tool("calc"), tool("search")])
    sessions, _, found = outcome([rec(a, 1.0), rec(b, 2.0)])
    assert sessions == 1 and F.TOOLS_CHANGED in found


def test_full_system_rewrite_with_growing_history_stays_in_session() -> None:
    a = convo("x", 0, system="A" * 6000)
    b = convo("x", 1, system="B" * 6000)
    sessions, breaks, found = outcome([rec(a, 1.0), rec(b, 2.0)])
    assert sessions == 1 and breaks == 1 and F.PREFIX_BROKEN in found


def test_explicit_sessions_are_never_merged() -> None:
    report = analyze(
        [rec(convo("x", 0), 1.0, session_id="s1"), rec(convo("x", 1), 2.0, session_id="s2")]
    )
    assert {s.id for s in report.sessions} == {"s1", "s2"}
    assert not report.auto_grouped


def test_two_users_saying_hi_stay_apart() -> None:
    r = Recorder()
    t = 0.0
    for turn in range(3):
        for who in ("A", "B"):
            history: list[dict[str, Any]] = []
            for i in range(turn):
                text = "hi" if i == 0 else f"{who} q{i}"
                history.append({"role": "user", "content": [{"type": "text", "text": text}]})
                history.append({"role": "assistant", "content": f"{who} a{i}"})
            user = "hi" if turn == 0 else f"{who} q{turn}"
            r.record("anthropic", anthropic_body(history=history, user=user, mark_last=True), at=t)
            t += 1
    report = analyze(r.records)
    assert report.totals.breaks == 0
    assert len(report.sessions) == 2


def test_two_users_saying_hi_still_stay_apart_with_the_edited_rule() -> None:
    r = Recorder()
    t = 0.0
    for turn in range(4):
        for who in ("A", "B"):
            history: list[dict[str, Any]] = []
            for i in range(turn):
                text = "hi" if i == 0 else f"{who} q{i}"
                history.append({"role": "user", "content": [{"type": "text", "text": text}]})
                history.append({"role": "assistant", "content": f"{who} a{i}"})
            user = "hi" if turn == 0 else f"{who} q{turn}"
            r.record("anthropic", anthropic_body(history=history, user=user, mark_last=True), at=t)
            t += 1
    report = analyze(r.records)
    assert len(report.sessions) == 2 and report.totals.breaks == 0


def test_canned_greeting_after_hi_does_not_merge_conversations() -> None:
    r = Recorder()
    t = 0.0
    for turn in range(1, 4):
        for who in ("A", "B"):
            history: list[dict[str, Any]] = [
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
                {"role": "assistant", "content": "Hello! How can I help you today?"},
            ]
            for i in range(1, turn):
                history.append(
                    {"role": "user", "content": [{"type": "text", "text": f"{who} q{i}"}]}
                )
                history.append({"role": "assistant", "content": f"{who} a{i}"})
            r.record(
                "anthropic",
                anthropic_body(history=history, user=f"{who} q{turn}", mark_last=True),
                at=t,
            )
            t += 1
    report = analyze(r.records)
    assert len(report.sessions) == 2 and report.totals.breaks == 0


def test_shorter_or_equal_different_body_is_not_a_continuation() -> None:
    long_ = P.segments(convo("x", 2))
    short = P.segments(convo("x", 1))
    assert not continues(long_, short)
    other = P.segments(
        anthropic_body(
            history=[{"role": "user", "content": "x q0"}, {"role": "assistant", "content": "zzz"}],
            user="x q1",
            mark_last=True,
        )
    )
    assert not continues(P.segments(convo("x", 1)), other)
    assert continues(short, short)  # exact repeat (retry)


def test_edited_rule_rejects_a_different_conversation_with_the_same_opener() -> None:
    a = P.segments(
        anthropic_body(
            history=[
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
                {"role": "assistant", "content": "A a0"},
                {"role": "user", "content": [{"type": "text", "text": "A q1"}]},
                {"role": "assistant", "content": "A a1"},
            ],
            user="A q2",
            mark_last=True,
        )
    )
    b = P.segments(
        anthropic_body(
            history=[
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
                {"role": "assistant", "content": "B a0"},
                {"role": "user", "content": [{"type": "text", "text": "B q1"}]},
                {"role": "assistant", "content": "B a1"},
            ],
            user="B q2",
            mark_last=True,
        )
    )
    assert not edited(a, b)


def test_log_watcher_needs_no_shared_index() -> None:
    r = Recorder(LogWatcher())
    r.record("anthropic", convo("x", 0))
    r.record("anthropic", convo("x", 1))
    assert r.records[0].session_id == r.records[1].session_id


def test_anchor_index_is_hashed_and_cleaned() -> None:
    idx = SessionIndex()
    segs = P.segments(convo("x", 0))
    sid = idx.assign("anthropic", segs, 1.0)
    assert all(isinstance(k, int) for k in idx._by_anchor)
    idx.assign("anthropic", P.segments(convo("x", 1)), 2.0)
    assert sum(sid in ids for ids in idx._by_anchor.values()) >= 1


def test_anchor_hash_is_deterministic() -> None:
    seg = P.segments(anthropic_body())[1]
    assert _hash("anthropic", seg) == _hash("anthropic", seg)
    assert _hash("anthropic", seg) != _hash("openai", seg)
    idx = SessionIndex()
    idx.assign("anthropic", P.segments(anthropic_body()), 1.0)
    assert all(k > 0 for k in idx._by_anchor)
