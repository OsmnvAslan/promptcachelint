"""Regression tests for the third external review."""

from __future__ import annotations

from typing import Any

from cachelint import Recorder, analyze
from cachelint import findings as F
from cachelint.model import Usage
from cachelint.providers.anthropic import AnthropicProvider
from cachelint.sessions import ANCHOR_BLOCKS, SessionIndex, _hash, edited
from tests.conftest import LONG, anthropic_body, rec

P = AnthropicProvider()


def exchange(i: int) -> list[dict[str, Any]]:
    """One agent turn: question, tool_use, tool_result, answer (four blocks)."""
    return [
        {"role": "user", "content": [{"type": "text", "text": f"q{i}"}]},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": f"t{i}", "name": "f", "input": {}}],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": f"t{i}", "content": "r"}],
        },
        {"role": "assistant", "content": f"a{i}"},
    ]


def windowed(i: int, window_blocks: int = 8) -> dict[str, Any]:
    msgs: list[dict[str, Any]] = []
    for k in range(i + 1):
        msgs += exchange(k)
    kept = msgs[-window_blocks:]
    kept.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"next{i}", "cache_control": {"type": "ephemeral"}}
            ],
        }
    )
    return {
        "model": "claude-opus-5",
        "max_tokens": 10,
        "system": [{"type": "text", "text": LONG, "cache_control": {"type": "ephemeral"}}],
        "messages": kept,
    }


# 1. sliding window dropping a whole agent exchange per request


def test_window_dropping_four_blocks_is_recognised() -> None:
    assert ANCHOR_BLOCKS >= 16
    report = analyze([rec(windowed(i), float(i)) for i in range(4)])
    assert len(report.sessions) == 1
    breaks = [f for f in report.findings if f.code == F.PREFIX_BROKEN]
    assert len(breaks) == 3  # every request after the first rewrites the prefix
    # Request 1 still holds exchange 0 (its trailing question was replaced: an edit);
    # requests 2 and 3 drop a whole four-block exchange each: the window slid.
    assert [f.data.get("window_slid") for f in breaks] == [False, True, True]


# 2. cross-session head check is scoped by model / thinking


def test_marked_head_across_models_is_not_an_unexplained_miss() -> None:
    records = [
        rec(
            anthropic_body(user=f"u{i}", mark_last=True, model=m), float(i), usage=Usage(0, 0, 1700)
        )
        for i, m in enumerate(["claude-opus-5", "claude-sonnet-5"])
    ]
    assert F.UNEXPLAINED_MISS not in {f.code for f in analyze(records).findings}

    thinking = [
        rec(anthropic_body(user=f"u{i}", mark_last=True) | extra, float(i), usage=Usage(0, 0, 1700))
        for i, extra in enumerate([{}, {"thinking": {"type": "adaptive"}}])
    ]
    assert F.UNEXPLAINED_MISS not in {f.code for f in analyze(thinking).findings}

    same = [
        rec(anthropic_body(user=f"u{i}", mark_last=True), float(i), usage=Usage(0, 0, 1700))
        for i in range(2)
    ]
    assert F.UNEXPLAINED_MISS in {f.code for f in analyze(same).findings}


# 3. edited history (old tool results truncated) is followed and reported


def truncated_history(i: int) -> dict[str, Any]:
    history: list[dict[str, Any]] = []
    for k in range(i):
        result = "[truncated]" if k < i - 1 else "R" * 500
        history += [
            {"role": "user", "content": [{"type": "text", "text": f"q{k}"}]},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": f"t{k}", "name": "f", "input": {}}],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": f"t{k}", "content": result}],
            },
            {"role": "assistant", "content": f"a{k}"},
        ]
    return anthropic_body(history=history, user=f"q{i}", mark_last=True)


def test_truncated_tool_results_are_one_session_with_edit_breaks() -> None:
    report = analyze([rec(truncated_history(i), float(i)) for i in range(4)])
    assert len(report.sessions) == 1
    edits = [
        f for f in report.findings if f.code == F.PREFIX_BROKEN and f.data.get("history_edited")
    ]
    assert len(edits) == 2  # requests 2 and 3 rewrite the previous tool_result
    assert "rewritten" in edits[0].hint
    assert edits[0].path.startswith("messages[2]")


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


# minor


def test_anchor_hash_is_deterministic() -> None:
    seg = P.segments(anthropic_body())[1]
    assert _hash("anthropic", seg) == _hash("anthropic", seg)
    assert _hash("anthropic", seg) != _hash("openai", seg)
    idx = SessionIndex()
    idx.assign("anthropic", P.segments(anthropic_body()), 1.0)
    assert all(k > 0 for k in idx._by_anchor)


# round 4


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


def test_hash_text_keeps_lengths_so_estimates_and_cl011_stay_right() -> None:
    from cachelint.redact import hash_text

    a, b = anthropic_body(system=LONG + "v1"), anthropic_body(system=LONG + "v2")
    plain = analyze([rec(a, 1.0), rec(b, 2.0)])
    hashed = analyze([rec(hash_text(a), 1.0), rec(hash_text(b), 2.0)])
    assert hashed.totals.lost_tokens_estimate == plain.totals.lost_tokens_estimate
    assert {f.code for f in hashed.findings} == {f.code for f in plain.findings}
    assert len(hash_text(a)["system"][0]["text"]) == len(a["system"][0]["text"])
    assert len(hash_text({"text": "hi"})["text"]) == 2
