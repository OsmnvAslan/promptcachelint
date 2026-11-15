"""Anthropic specifics: block offsets, lookback positions, breakpoint slots, scope, TTL."""

from __future__ import annotations

from typing import Any

from promptcachelint import analyze
from promptcachelint import findings as F
from promptcachelint.analyze import positions
from promptcachelint.detectors import lint_request
from promptcachelint.diff import diff_prefix
from promptcachelint.model import Segment
from promptcachelint.providers.anthropic import AnthropicProvider
from tests.conftest import anthropic_body, convo, rec, tool

P = AnthropicProvider()


def test_message_offsets_are_relative_to_the_block_text() -> None:
    a = anthropic_body(
        history=[{"role": "user", "content": [{"type": "text", "text": "abcXdef"}]}],
        user="z",
        mark_last=True,
    )
    b = anthropic_body(
        history=[{"role": "user", "content": [{"type": "text", "text": "abcYdef"}]}],
        user="z",
        mark_last=True,
    )
    sa, sb = P.segments(a), P.segments(b)
    d = diff_prefix(sa, sb, P.cacheable_segments(a, sa))
    assert d.broken and d.offset == 3 and d.role == "user"
    assert sa[1].text == "abcXdef" and sa[1].role == "user"


def test_same_text_different_role_is_a_different_segment() -> None:
    a = Segment(path="m", kind="message", text="hi", role="user")
    b = Segment(path="m", kind="message", text="hi", role="assistant")
    assert not a.same(b)


def test_positions_collapse_tool_runs() -> None:
    segs = [Segment(path=str(i), kind="message", text="x", block="tool_use") for i in range(25)]
    segs += [Segment(path=str(i), kind="message", text="y", block="tool_result") for i in range(25)]
    assert positions(segs) == 2
    segs.append(Segment(path="t", kind="message", text="z", block="text"))
    assert positions(segs) == 3


def test_parallel_tool_calls_do_not_trigger_lookback_warning() -> None:
    a = convo("x", 0)
    history: list[dict[str, Any]] = [
        {"role": "user", "content": [{"type": "text", "text": "x q0"}]}
    ]
    history.append(
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": f"t{i}", "name": "f", "input": {}} for i in range(25)
            ],
        }
    )
    history.append(
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": f"t{i}", "content": "ok"} for i in range(25)
            ],
        }
    )
    b = anthropic_body(history=history, user="x q1", mark_last=True)
    report = analyze([rec(a, 1.0), rec(b, 2.0)])
    assert F.LOOKBACK_EXCEEDED not in {f.code for f in report.findings}


def test_top_level_cache_control_counts_as_a_slot() -> None:
    marked = [tool(n) | {"cache_control": {"type": "ephemeral"}} for n in "abc"]
    body = anthropic_body(tools=marked, mark_system=True, top_level=True)
    assert F.TOO_MANY_BREAKPOINTS in {f.code for f in lint_request(P, body)}


def test_scope_sees_tool_choice_and_speed() -> None:
    a = convo("x", 0) | {"tool_choice": {"type": "auto"}}
    b = convo("x", 1) | {"tool_choice": {"type": "any"}, "speed": "fast"}
    report = analyze([rec(a, 1.0), rec(b, 2.0)])
    keys = sorted(f.data["key"] for f in report.findings if f.code == F.SCOPE_CHANGED)
    assert keys == ["speed", "tool_choice"]


def test_ttl_uses_the_longest_marker() -> None:
    body = anthropic_body(ttl="1h", mark_last=True)
    body["messages"][0]["content"][0]["cache_control"] = {"type": "ephemeral"}  # 5m on the tail
    assert P.ttl_seconds(P.segments(body)) == 3600
