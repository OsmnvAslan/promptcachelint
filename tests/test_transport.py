from __future__ import annotations

import json
from typing import Any

import httpx2 as hx
import pytest

from cachelint import Recorder
from cachelint.transport import async_client, client, parse_sse, wrap_transport
from tests.conftest import anthropic_body

ANTHROPIC = "https://api.anthropic.com/v1/messages"
OPENAI = "https://api.openai.com/v1/chat/completions"

SSE = (
    b'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_1","usage":'
    b'{"input_tokens":4,"cache_read_input_tokens":1600,"cache_creation_input_tokens":0}}}\n\n'
    b'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"text":"hi"}}\n\n'
    b'event: message_delta\ndata: {"type":"message_delta","usage":{"output_tokens":7}}\n\n'
    b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
)


def handler(request: hx.Request) -> hx.Response:
    body = json.loads(request.content)
    if body.get("stream"):
        return hx.Response(200, headers={"content-type": "text/event-stream"}, content=SSE)
    if body.get("fail"):
        return hx.Response(429, json={"error": "rate"})
    if "messages" in body and "max_tokens" not in body:  # openai chat
        return hx.Response(
            200,
            json={
                "id": "chatcmpl",
                "usage": {"prompt_tokens": 1100, "prompt_tokens_details": {"cached_tokens": 1024}},
            },
        )
    return hx.Response(
        200,
        json={
            "id": "msg_2",
            "usage": {"input_tokens": 5, "cache_creation_input_tokens": 1600, "output_tokens": 2},
        },
    )


def test_parse_sse() -> None:
    events = parse_sse(SSE + b"data: [DONE]\n\n")
    assert [e["type"] for e in events] == [
        "message_start",
        "content_block_delta",
        "message_delta",
        "message_stop",
    ]


def test_sync_client_records_json_and_sse_and_passes_through() -> None:
    recorder = Recorder()
    with client(recorder, transport=hx.MockTransport(handler)) as c:
        r1 = c.post(ANTHROPIC, json=anthropic_body())
        assert r1.json()["id"] == "msg_2"

        with c.stream("POST", ANTHROPIC, json=anthropic_body() | {"stream": True}) as r2:
            chunks = list(r2.iter_bytes())
        assert b"".join(chunks) == SSE

        r3 = c.post(OPENAI, json={"model": "gpt-5", "messages": [{"role": "user", "content": "x"}]})
        assert r3.status_code == 200
        c.post(ANTHROPIC, json=anthropic_body() | {"fail": True})
        c.post("https://example.com/other", json={"x": 1})

    recs = recorder.records
    assert [r.provider for r in recs] == ["anthropic", "anthropic", "openai"]
    assert recs[0].usage is not None and recs[0].usage.cache_write == 1600
    assert recs[0].response_id == "msg_2" and recs[0].url == ANTHROPIC
    assert recs[1].stream and recs[1].usage is not None
    assert recs[1].usage.cache_read == 1600 and recs[1].usage.output_tokens == 7
    assert recs[2].usage is not None and recs[2].usage.cache_read == 1024
    assert recs[0].session_id == recs[1].session_id


def test_partially_consumed_stream_is_still_recorded_on_close() -> None:
    recorder = Recorder()
    with (
        client(recorder, transport=hx.MockTransport(handler)) as c,
        c.stream("POST", ANTHROPIC, json=anthropic_body() | {"stream": True}) as r,
    ):
        next(r.iter_bytes(chunk_size=64))
    assert len(recorder.records) == 1
    assert recorder.records[0].usage is not None


@pytest.mark.asyncio
async def test_async_client() -> None:
    recorder = Recorder()
    async with async_client(recorder, transport=hx.MockTransport(handler)) as c:
        r1 = await c.post(ANTHROPIC, json=anthropic_body())
        assert r1.json()["id"] == "msg_2"
        async with c.stream("POST", ANTHROPIC, json=anthropic_body() | {"stream": True}) as r2:
            data = b"".join([chunk async for chunk in r2.aiter_bytes()])
        assert data == SSE
    assert len(recorder.records) == 2
    assert recorder.records[1].usage is not None and recorder.records[1].usage.cache_read == 1600


def test_wrap_transport_picks_sync_or_async() -> None:
    recorder = Recorder()
    assert (
        type(wrap_transport(hx.MockTransport(handler), recorder)).__name__ == "RecordingTransport"
    )

    class AsyncOnly(hx.AsyncBaseTransport):
        async def handle_async_request(
            self, request: hx.Request
        ) -> hx.Response:  # pragma: no cover
            return hx.Response(200)

    assert type(wrap_transport(AsyncOnly(), recorder)).__name__ == "AsyncRecordingTransport"
    forced = wrap_transport(hx.MockTransport(handler), recorder, use_async=True)
    assert type(forced).__name__ == "AsyncRecordingTransport"


def test_unparseable_bodies_never_break_the_request() -> None:
    recorder = Recorder()

    def bad(request: hx.Request) -> hx.Response:
        return hx.Response(200, content=b"not json", headers={"content-type": "application/json"})

    with client(recorder, transport=hx.MockTransport(bad)) as c:
        assert c.post(ANTHROPIC, content=b"also not json").text == "not json"
        assert c.post(ANTHROPIC, json=anthropic_body()).text == "not json"
    assert recorder.records == []


def test_sdk_style_usage_matches_recorder_sessions() -> None:
    """Two growing turns through the transport land in one session with an intact prefix."""
    from cachelint import analyze

    recorder = Recorder()
    history: list[dict[str, Any]] = []
    with client(recorder, transport=hx.MockTransport(handler)) as c:
        for q in ("q0", "q1"):
            body = anthropic_body(history=history, user=q, mark_last=True)
            c.post(ANTHROPIC, json=body)
            history += [
                {"role": "user", "content": [{"type": "text", "text": q}]},
                {"role": "assistant", "content": "a"},
            ]
    report = analyze(recorder.records)
    assert len(report.sessions) == 1 and report.totals.breaks == 0
