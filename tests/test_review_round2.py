"""Regression tests for the second external review."""

from __future__ import annotations

from typing import Any

from cachelint import LogWatcher, Recorder, analyze
from cachelint import findings as F
from cachelint.model import Record, Usage
from cachelint.providers.anthropic import AnthropicProvider
from cachelint.sessions import SessionIndex, continues, truncated
from tests.conftest import LONG, anthropic_body, rec, tool

P = AnthropicProvider()


def convo(who: str, turn: int, **kw: Any) -> dict[str, Any]:
    history: list[dict[str, Any]] = []
    for i in range(turn):
        history.append({"role": "user", "content": [{"type": "text", "text": f"{who} q{i}"}]})
        history.append({"role": "assistant", "content": f"a{i}"})
    return anthropic_body(history=history, user=f"{who} q{turn}", mark_last=True, **kw)


def codes(records: list[Record]) -> tuple[int, int, set[str]]:
    report = analyze(records)
    return len(report.sessions), report.totals.breaks, {f.code for f in report.findings}


# A. tool reorder keeps the session and reports CL002


def test_tool_reorder_mid_conversation_is_reported() -> None:
    a = convo("x", 0, tools=[tool("search"), tool("calc")])
    b = convo("x", 1, tools=[tool("calc"), tool("search")])
    sessions, _, found = codes([rec(a, 1.0), rec(b, 2.0)])
    assert sessions == 1 and F.TOOLS_CHANGED in found


def test_full_system_rewrite_with_growing_history_stays_in_session() -> None:
    a = convo("x", 0, system="A" * 6000)
    b = convo("x", 1, system="B" * 6000)
    sessions, breaks, found = codes([rec(a, 1.0), rec(b, 2.0)])
    assert sessions == 1 and breaks == 1 and F.PREFIX_BROKEN in found


# B. identical openers do not merge from the second turn on


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


# C. OpenAI recurring mid-prefix break is reported every time


def test_openai_recurring_partial_read_is_reported_each_time() -> None:
    records = []
    for i in range(5):
        body = {
            "model": "gpt-5",
            "messages": [
                {"role": "system", "content": "S" * 5000 + f" now=2026-09-20T10:0{i}" + "T" * 1000},
                {"role": "user", "content": "hi"},
            ],
        }
        records.append(
            rec(body, float(i), provider="openai", usage=Usage(300, 0 if i == 0 else 1280, 0))
        )
    sessions, breaks, found = codes(records)
    assert sessions == 1
    assert breaks == 4 and F.PREFIX_BROKEN in found


def test_openai_full_read_softens_only_when_the_whole_prefix_was_read() -> None:
    """Same explicit session, divergence in the tail: softened iff cache_read covers prev."""
    sysm = {"role": "system", "content": "S" * 6000}  # ~1500 tokens

    def body(q: str) -> dict[str, Any]:
        return {"model": "gpt-5", "messages": [sysm, {"role": "user", "content": q}]}

    def run(read: int) -> int:
        records = [
            rec(body("a" * 600), 0.0, provider="openai", usage=Usage(1650, 0, 0), session_id="s"),
            rec(body("b" * 600), 1.0, provider="openai", usage=Usage(150, read, 0), session_id="s"),
        ]
        return analyze(records).totals.breaks

    assert run(1536) == 0  # read >= ~1650 - 128: the tail alone changed
    assert run(1024) == 1  # a stable partial read is a real break


# D. CL014 vs CL005 depending on whether the head is marked


def test_marked_head_with_no_reads_is_an_unexplained_miss_not_placement() -> None:
    records = [
        rec(anthropic_body(user=f"unique {i}", mark_last=True), float(i), usage=Usage(0, 0, 1700))
        for i in range(2)
    ]
    found = analyze(records).findings
    assert [f.code for f in found] == [F.UNEXPLAINED_MISS]
    assert "marked" in found[0].message


def test_unmarked_head_with_tail_marker_is_write_without_read() -> None:
    records = [
        rec(
            anthropic_body(user=f"unique {i}", mark_system=False, mark_last=True),
            float(i),
            usage=Usage(0, 0, 1700),
        )
        for i in range(2)
    ]
    found = analyze(records).findings
    assert F.WRITE_WITHOUT_READ in {f.code for f in found}
    assert F.UNEXPLAINED_MISS not in {f.code for f in found}


# E. sliding-window history


def sliding(i: int, window: int = 4) -> dict[str, Any]:
    msgs: list[dict[str, Any]] = []
    for k in range(i + 1):
        msgs.append({"role": "user", "content": [{"type": "text", "text": f"q{k}"}]})
        if k < i:
            msgs.append({"role": "assistant", "content": f"a{k}"})
    kept = msgs[-window:]
    kept[-1] = {
        "role": "user",
        "content": [{"type": "text", "text": f"q{i}", "cache_control": {"type": "ephemeral"}}],
    }
    return {
        "model": "claude-opus-5",
        "max_tokens": 10,
        "system": [{"type": "text", "text": LONG, "cache_control": {"type": "ephemeral"}}],
        "messages": kept,
    }


def test_sliding_window_is_one_session_with_window_breaks() -> None:
    records = [rec(sliding(i), float(i)) for i in range(6)]
    report = analyze(records)
    assert len(report.sessions) == 1
    slid = [f for f in report.findings if f.code == F.PREFIX_BROKEN and f.data.get("window_slid")]
    assert len(slid) >= 3
    assert "window" in slid[0].hint.lower()
    assert slid[0].path == "messages[0].content[0]"


def test_truncated_detector() -> None:
    prev = P.segments(sliding(3))
    new = P.segments(sliding(4))
    assert truncated(prev, new) and not continues(prev, new)


# minor


def test_ttl_uses_the_longest_marker() -> None:
    body = anthropic_body(ttl="1h", mark_last=True)
    body["messages"][0]["content"][0]["cache_control"] = {"type": "ephemeral"}  # 5m on the tail
    assert P.ttl_seconds(P.segments(body)) == 3600


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
