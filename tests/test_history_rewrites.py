"""Sliding-window and edited histories: one session, each rewrite reported as a break."""

from __future__ import annotations

from typing import Any

from promptcachelint import analyze
from promptcachelint import findings as F
from promptcachelint.providers.anthropic import AnthropicProvider
from promptcachelint.sessions import ANCHOR_BLOCKS, continues, truncated
from tests.conftest import LONG, anthropic_body, rec

P = AnthropicProvider()


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


def test_window_dropping_four_blocks_is_recognised() -> None:
    assert ANCHOR_BLOCKS >= 16
    report = analyze([rec(windowed(i), float(i)) for i in range(4)])
    assert len(report.sessions) == 1
    breaks = [f for f in report.findings if f.code == F.PREFIX_BROKEN]
    assert len(breaks) == 3  # every request after the first rewrites the prefix
    # Request 1 still holds exchange 0 (its trailing question was replaced: an edit);
    # requests 2 and 3 drop a whole four-block exchange each: the window slid.
    assert [f.data.get("window_slid") for f in breaks] == [False, True, True]


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
