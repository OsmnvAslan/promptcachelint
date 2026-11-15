"""Transport edge cases: compressed bodies, CRLF event streams, streams nobody read."""

from __future__ import annotations

import gzip

import httpx2 as hx

from promptcachelint import Recorder, analyze
from promptcachelint.transport import client, parse_sse
from tests.conftest import anthropic_body

ANTHROPIC = "https://api.anthropic.com/v1/messages"


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


def test_unconsumed_stream_is_recorded_with_a_note() -> None:
    def handler(request: hx.Request) -> hx.Response:
        return hx.Response(
            200, headers={"content-type": "text/event-stream"}, content=b"data: {}\n\n"
        )

    r = Recorder()
    with (
        client(r, transport=hx.MockTransport(handler)) as c,
        c.stream(
            "POST",
            "https://api.anthropic.com/v1/messages",
            json=anthropic_body() | {"stream": True},
        ) as resp,
    ):
        resp.close()  # never read a byte
    assert len(r.records) == 1
    assert r.records[0].usage is None and r.records[0].stream
    assert r.records[0].note and "never consumed" in r.records[0].note
    assert "never consumed" in analyze(r.records).to_text()
