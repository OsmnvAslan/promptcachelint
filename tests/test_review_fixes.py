"""Regression tests for the findings of the first external review."""

from __future__ import annotations

import gzip
import threading
import time
from typing import Any

import httpx2 as hx

from cachelint import Recorder, analyze
from cachelint import findings as F
from cachelint.analyze import positions
from cachelint.diff import diff_prefix
from cachelint.model import Segment, Usage
from cachelint.providers.anthropic import AnthropicProvider
from cachelint.redact import hash_text, strip_media
from cachelint.transport import client, parse_sse
from tests.conftest import LONG, anthropic_body, rec, tool

P = AnthropicProvider()
ANTHROPIC = "https://api.anthropic.com/v1/messages"


def convo(who: str, turn: int, **kw: Any) -> dict[str, Any]:
    history: list[dict[str, Any]] = []
    for i in range(turn):
        history.append({"role": "user", "content": [{"type": "text", "text": f"{who} q{i}"}]})
        history.append({"role": "assistant", "content": f"a{i}"})
    return anthropic_body(history=history, user=f"{who} q{turn}", mark_last=True, **kw)


# ---- 1. session grouping -------------------------------------------------


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


def test_explicit_sessions_are_never_merged() -> None:
    report = analyze(
        [rec(convo("x", 0), 1.0, session_id="s1"), rec(convo("x", 1), 2.0, session_id="s2")]
    )
    assert {s.id for s in report.sessions} == {"s1", "s2"}
    assert not report.auto_grouped


# ---- 2. gzip -------------------------------------------------------------


def test_transport_decodes_compressed_responses() -> None:
    payload = b'{"id":"msg_gz","usage":{"input_tokens":5,"cache_creation_input_tokens":1600}}'

    def handler(request: hx.Request) -> hx.Response:
        return hx.Response(
            200,
            headers={"content-type": "application/json", "content-encoding": "gzip"},
            content=gzip.compress(payload),
        )

    r = Recorder()
    with client(r, transport=hx.MockTransport(handler)) as c:
        assert c.post(ANTHROPIC, json=anthropic_body()).json()["id"] == "msg_gz"
    assert len(r.records) == 1
    assert r.records[0].usage is not None and r.records[0].usage.cache_write == 1600
    assert r.records[0].response_id == "msg_gz"


def test_parse_sse_accepts_crlf() -> None:
    data = b'data: {"type":"a"}\r\n\r\ndata: {"type":"b"}\r\n\r\n'
    assert [e["type"] for e in parse_sse(data)] == ["a", "b"]


# ---- 3. offsets ----------------------------------------------------------


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


# ---- 4. OpenAI automatic caching -----------------------------------------


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


# ---- important -----------------------------------------------------------


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
    from cachelint.detectors import lint_request

    assert F.TOO_MANY_BREAKPOINTS in {f.code for f in lint_request(P, body)}


def test_scope_sees_tool_choice_and_speed() -> None:
    a = convo("x", 0) | {"tool_choice": {"type": "auto"}}
    b = convo("x", 1) | {"tool_choice": {"type": "any"}, "speed": "fast"}
    report = analyze([rec(a, 1.0), rec(b, 2.0)])
    keys = sorted(f.data["key"] for f in report.findings if f.code == F.SCOPE_CHANGED)
    assert keys == ["speed", "tool_choice"]


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


def test_redaction_helpers() -> None:
    body = anthropic_body()
    body["messages"][0]["content"].append(
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": "A" * 1000},
        }
    )
    stripped = strip_media(body)
    assert "redacted" in stripped["messages"][0]["content"][1]["source"]["data"]
    assert stripped["system"][0]["text"] == body["system"][0]["text"]
    assert body["messages"][0]["content"][1]["source"]["data"] == "A" * 1000  # original untouched

    hashed = hash_text(body)
    assert hashed["system"][0]["text"].startswith("sha256:")
    short = hashed["messages"][0]["content"][0]["text"]  # "hello": same length, still hidden
    assert len(short) == 5 and short != "hello"
    assert hashed["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_jsonl_sink_redacts(tmp_path: Any) -> None:
    from cachelint import JsonlSink, load_jsonl

    path = tmp_path / "t.jsonl"
    r = Recorder(JsonlSink(path, redact=hash_text))
    r.record("anthropic", anthropic_body())
    assert load_jsonl(path)[0].body["system"][0]["text"].startswith("sha256:")
