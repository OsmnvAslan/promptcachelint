"""A tools+system head shared across conversations: CL005 when marked, CL014 when not."""

from __future__ import annotations

from promptcachelint import analyze
from promptcachelint import findings as F
from promptcachelint.model import Usage
from tests.conftest import anthropic_body, rec


def test_marked_head_with_no_reads_is_an_unexplained_miss_not_placement() -> None:
    records = [
        rec(anthropic_body(user=f"unique {i}", mark_last=True), float(i), usage=Usage(0, 0, 1700))
        for i in range(2)
    ]
    found = analyze(records).findings
    assert [f.code for f in found] == [F.UNEXPLAINED_MISS]
    assert "marked" in found[0].message


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
