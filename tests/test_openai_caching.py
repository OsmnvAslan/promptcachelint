"""OpenAI automatic caching: what counts as a break when there are no markers."""

from __future__ import annotations

from typing import Any

from promptcachelint import Recorder, analyze
from promptcachelint import findings as F
from promptcachelint.model import Usage
from tests.conftest import outcome, rec


def test_openai_shared_prefix_varying_question_is_not_a_break() -> None:
    r = Recorder()
    sysm = {"role": "system", "content": "S" * 6000}
    for i, q in enumerate(["what is 1+1", "what is 2+2", "what is 3+3"]):
        r.record(
            "openai",
            {"model": "gpt-5", "messages": [sysm, {"role": "user", "content": q}]},
            usage=Usage(20, 0 if i == 0 else 1536, 0),
            at=float(i),
        )
    report = analyze(r.records)
    assert report.totals.breaks == 0
    assert F.PREFIX_BROKEN not in {f.code for f in report.findings}


def test_openai_real_prefix_change_is_still_a_break() -> None:
    r = Recorder()
    for i, s in enumerate(["S" * 6000, "S" * 5000 + "T" * 1000]):
        r.record(
            "openai",
            {
                "model": "gpt-5",
                "messages": [{"role": "system", "content": s}, {"role": "user", "content": "hi"}],
            },
            usage=Usage(1500, 0, 0),
            at=float(i),
        )
    report = analyze(r.records)
    assert report.totals.breaks == 1


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
    sessions, breaks, found = outcome(records)
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


def test_openai_unexplained_miss_is_info_and_ttl_is_ten_minutes() -> None:
    body = {
        "model": "gpt-5",
        "messages": [{"role": "system", "content": "S" * 6000}, {"role": "user", "content": "hi"}],
    }
    report = analyze(
        [
            rec(body, 0.0, provider="openai", usage=Usage(1500, 0, 0)),
            rec(body, 400.0, provider="openai", usage=Usage(1500, 0, 0)),
        ]
    )
    miss = [f for f in report.findings if f.code == F.UNEXPLAINED_MISS]
    assert miss and miss[0].severity == "info"
    assert F.TTL_GAP not in {f.code for f in report.findings}
