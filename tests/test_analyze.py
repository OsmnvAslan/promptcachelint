from __future__ import annotations

from typing import Any

from promptcachelint import findings as F
from promptcachelint.analyze import analyze
from promptcachelint.model import Usage
from promptcachelint.sessions import SessionIndex
from tests.conftest import LONG, anthropic_body, rec, tool


def turn(n: int, **kw: Any) -> dict[str, Any]:
    """A growing conversation: n completed exchanges plus one new user turn."""
    history: list[dict[str, Any]] = []
    for i in range(n):
        history.append({"role": "user", "content": [{"type": "text", "text": f"q{i}"}]})
        history.append({"role": "assistant", "content": f"a{i}"})
    return anthropic_body(history=history, user=f"q{n}", mark_last=True, **kw)


def codes(report_findings: list[F.Finding]) -> set[str]:
    return {f.code for f in report_findings}


def test_clean_conversation_has_no_findings_and_sums_usage() -> None:
    records = [
        rec(turn(0), 1.0, usage=Usage(10, 0, 1600)),
        rec(turn(1), 2.0, usage=Usage(10, 1600, 20)),
        rec(turn(2), 3.0, usage=Usage(10, 1620, 20)),
    ]
    report = analyze(records)
    assert len(report.sessions) == 1
    assert codes(report.findings) == set()
    t = report.totals
    assert t.requests == 3 and t.breaks == 0
    assert t.cache_read == 3220 and t.cache_write == 1640 and t.input_tokens == 30
    assert 0.65 < t.hit_ratio < 0.66


def test_timestamp_in_system_breaks_prefix_and_is_flagged() -> None:
    a = turn(0, system=LONG + " Now: 2026-09-20T10:15:00Z")
    b = turn(1, system=LONG + " Now: 2026-09-20T10:16:00Z")
    report = analyze([rec(a, 1.0, usage=Usage(5, 0, 1600)), rec(b, 2.0, usage=Usage(1600, 0, 0))])
    found = codes(report.findings)
    assert F.PREFIX_BROKEN in found and F.VOLATILE_CONTENT in found
    assert F.WRITE_WITHOUT_READ not in found
    assert sum(f.code == F.VOLATILE_CONTENT for f in report.findings) == 1  # once per session
    broken = next(f for f in report.findings if f.code == F.PREFIX_BROKEN)
    assert broken.path == "system[0]"
    assert broken.offset == len(LONG) + len(" Now: 2026-09-20T10:1")
    assert report.totals.breaks == 1
    assert report.totals.lost_tokens_estimate > 1000
    text = report.to_text()
    assert "prefix broken at system[0]" in text and "CL012" in text


def test_tool_change_is_its_own_finding() -> None:
    a = turn(0, tools=[tool("search")])
    b = turn(1, tools=[tool("search"), tool("calc")])
    report = analyze([rec(a, 1.0), rec(b, 2.0)])
    found = codes(report.findings)
    assert F.TOOLS_CHANGED in found and F.PREFIX_BROKEN not in found


def test_scope_changes() -> None:
    a = turn(0)
    b = turn(1, model="claude-sonnet-5")
    c = turn(2, model="claude-sonnet-5") | {"thinking": {"type": "adaptive"}}
    report = analyze([rec(a, 1.0), rec(b, 2.0), rec(c, 3.0)])
    scope = [f for f in report.findings if f.code == F.SCOPE_CHANGED]
    assert [f.data["key"] for f in scope] == ["model", "thinking"]
    assert scope[0].severity == "error" and scope[1].severity == "warning"


def test_unexplained_miss_and_ttl_gap() -> None:
    a = turn(0)
    b = turn(1)
    miss = analyze([rec(a, 1.0, usage=Usage(5, 0, 1600)), rec(b, 2.0, usage=Usage(1650, 0, 0))])
    assert F.UNEXPLAINED_MISS in codes(miss.findings)
    assert F.PREFIX_BROKEN not in codes(miss.findings)

    gap = analyze(
        [rec(a, 1.0, usage=Usage(5, 0, 1600)), rec(b, 1.0 + 3600, usage=Usage(5, 0, 1650))]
    )
    found = codes(gap.findings)
    assert F.TTL_GAP in found and F.UNEXPLAINED_MISS not in found


def test_one_hour_ttl_extends_the_gap_window() -> None:
    a = turn(0, ttl="1h")
    b = turn(1, ttl="1h")
    report = analyze([rec(a, 1.0), rec(b, 1.0 + 1800)])
    assert F.TTL_GAP not in codes(report.findings)


def test_breakpoint_moved_back_and_write_without_read() -> None:
    a = turn(0)  # marker on system + last turn
    b = anthropic_body(
        history=[
            {"role": "user", "content": [{"type": "text", "text": "q0"}]},
            {"role": "assistant", "content": "a0"},
        ],
        user="q1",
        mark_last=False,
    )
    report = analyze([rec(a, 1.0), rec(b, 2.0)])
    assert F.BREAKPOINT_MOVED_BACK in codes(report.findings)

    # Per-request tail marked: writes every time, never reads.
    w1 = anthropic_body(user="unique 1", mark_system=False, mark_last=True)
    w2 = anthropic_body(user="unique 2", mark_system=False, mark_last=True)
    report = analyze([rec(w1, 1.0, usage=Usage(0, 0, 1700)), rec(w2, 2.0, usage=Usage(0, 0, 1700))])
    assert F.WRITE_WITHOUT_READ in codes(report.findings)


def test_lookback_exceeded() -> None:
    a = turn(0)
    big_history = [{"role": "user", "content": [{"type": "text", "text": "q0"}]}] + [
        {"role": "assistant", "content": f"step {i}"} for i in range(25)
    ]
    b = anthropic_body(history=big_history, user="q1", mark_last=True)
    report = analyze([rec(a, 1.0), rec(b, 2.0)])
    assert F.LOOKBACK_EXCEEDED in codes(report.findings)


def test_explicit_sessions_are_respected_and_auto_grouping_separates_unrelated() -> None:
    a = turn(0)
    b = turn(1)
    other = anthropic_body(system="completely different " * 100, user="x")
    report = analyze([rec(a, 1.0), rec(b, 2.0), rec(other, 3.0)])
    assert len(report.sessions) == 2
    assert report.auto_grouped

    report = analyze([rec(a, 1.0, session_id="s1"), rec(b, 2.0, session_id="s2")])
    assert {s.id for s in report.sessions} == {"s1", "s2"}
    assert all(r.diff is None for s in report.sessions for r in s.requests)


def test_custom_index_and_json_output() -> None:
    report = analyze([rec(turn(0), 1.0), rec(turn(1), 2.0)], index=SessionIndex())
    d = report.to_dict()
    assert d["totals"]["requests"] == 2
    assert d["sessions"][0]["requests"][1]["diff"] == {
        "broken": False,
        "common_segments": 2,
        "path": None,
        "offset": None,
        "lost_tokens_estimate": 0,
    }
