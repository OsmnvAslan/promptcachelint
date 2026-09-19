from __future__ import annotations

from cachelint import findings as F
from cachelint.detectors import lint_request, volatile_content
from cachelint.diff import diff_prefix
from cachelint.model import Segment
from cachelint.providers.anthropic import AnthropicProvider
from tests.conftest import anthropic_body, tool

P = AnthropicProvider()


def segs(body: dict) -> tuple[list[Segment], int]:  # type: ignore[type-arg]
    s = P.segments(body)
    return s, P.cacheable_segments(body, s)


class TestDiff:
    def test_appending_a_turn_is_not_a_break(self) -> None:
        prev, cacheable = segs(anthropic_body(mark_last=True))
        new, _ = segs(
            anthropic_body(
                history=[
                    {"role": "user", "content": [{"type": "text", "text": "hello"}]},
                    {"role": "assistant", "content": "hi!"},
                ],
                user="next",
                mark_last=True,
            )
        )
        d = diff_prefix(prev, new, cacheable)
        assert not d.broken
        assert d.common_segments == cacheable == 2

    def test_change_inside_prefix_reports_block_and_offset(self) -> None:
        prev, cacheable = segs(anthropic_body(system="Rules v1. Be nice." * 100))
        new, _ = segs(anthropic_body(system="Rules v2. Be nice." * 100))
        d = diff_prefix(prev, new, cacheable)
        assert d.broken
        assert d.path == "system[0]" and d.kind == "system"
        assert d.offset == 7
        assert d.before.startswith("Rules v1") and d.after.startswith("Rules v2")
        assert d.lost_tokens_estimate > 400

    def test_tool_change_breaks_at_position_zero(self) -> None:
        prev, cacheable = segs(anthropic_body(tools=[tool("a"), tool("b")]))
        new, _ = segs(anthropic_body(tools=[tool("b"), tool("a")]))
        d = diff_prefix(prev, new, cacheable)
        assert d.broken and d.segment_index == 0 and d.kind == "tool"

    def test_shorter_request_than_cached_prefix(self) -> None:
        prev, cacheable = segs(anthropic_body(mark_last=True))
        d = diff_prefix(prev, prev[:1], cacheable)
        assert d.broken and d.segment_index == 1 and d.after == ""

    def test_no_cacheable_prefix_never_breaks(self) -> None:
        prev, cacheable = segs(anthropic_body(mark_system=False))
        assert cacheable == 0
        new, _ = segs(anthropic_body(mark_system=False, system="other"))
        assert not diff_prefix(prev, new, cacheable).broken


class TestDetectors:
    def test_volatile_patterns(self) -> None:
        cases = {
            "iso-datetime": "Now: 2026-09-20T10:15:00Z.",
            "uuid": "session 123e4567-e89b-12d3-a456-426614174000",
            "hex-id": "trace 9f1c2b3a4d5e6f708192a3b4c5d6e7f8",
            "date-phrase": "Today is Saturday, September 20.",
            "rfc-date": "Sat, 20 Sep 2026 10:15:00 GMT",
        }
        for name, text in cases.items():
            s = [Segment(path="system[0]", kind="system", text=text, breakpoint=True)]
            found = volatile_content(s, 1)
            assert found and found[0].data["pattern"] == name, name
            assert found[0].code == F.VOLATILE_CONTENT

    def test_volatile_outside_prefix_is_ignored(self) -> None:
        body = anthropic_body(user="It is 2026-09-20T10:15:00Z, what's up?")
        assert not [f for f in lint_request(P, body) if f.code == F.VOLATILE_CONTENT]

    def test_no_breakpoint_and_too_short(self) -> None:
        codes = {f.code for f in lint_request(P, anthropic_body(mark_system=False))}
        assert F.NO_BREAKPOINT in codes
        short = anthropic_body(system="tiny system prompt")
        codes = {f.code for f in lint_request(P, short)}
        assert F.PREFIX_TOO_SHORT in codes and F.NO_BREAKPOINT not in codes
        assert not lint_request(P, anthropic_body())

    def test_too_many_breakpoints(self) -> None:
        marked = [tool(n) | {"cache_control": {"type": "ephemeral"}} for n in "abcd"]
        body = anthropic_body(tools=marked, mark_system=True)
        assert F.TOO_MANY_BREAKPOINTS in {f.code for f in lint_request(P, body)}
