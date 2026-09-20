"""A tool-using agent loop with promptcachelint watching the cache.

Run with ANTHROPIC_API_KEY set:  uv run --with anthropic examples/anthropic_agent.py

The system prompt deliberately interpolates the current time so the second turn
breaks the prefix; promptcachelint logs CL012 on the first request and CL001 on the
second, then the final report shows where the bytes diverged.
"""

from __future__ import annotations

import datetime as dt
import logging

import anthropic
import httpx2

import promptcachelint
from promptcachelint.redact import strip_media
from promptcachelint.transport import wrap_transport

logging.basicConfig(level=logging.INFO, format="%(message)s")

recorder = promptcachelint.Recorder(
    promptcachelint.LogWatcher(),
    promptcachelint.JsonlSink("trace.jsonl", redact=strip_media),
)
claude = anthropic.Anthropic(
    http_client=anthropic.DefaultHttpxClient(
        transport=wrap_transport(httpx2.HTTPTransport(), recorder)
    )
)

TOOLS = [
    {
        "name": "get_weather",
        "description": "Weather for a city.",
        "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
    }
]


def system_prompt() -> list[dict[str, object]]:
    rules = "You are a travel assistant. Answer briefly. " * 60
    now = dt.datetime.now(dt.UTC).isoformat(timespec="minutes")  # <- the bug
    return [
        {
            "type": "text",
            "text": f"{rules}\nCurrent time: {now}",
            "cache_control": {"type": "ephemeral"},
        }
    ]


messages: list[dict[str, object]] = []
with promptcachelint.session("demo"):
    for question in ("Weather in Lisbon?", "And in Porto?"):
        messages.append({"role": "user", "content": question})
        response = claude.messages.create(
            model="claude-opus-5",
            max_tokens=256,
            system=system_prompt(),
            tools=TOOLS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

print()
print(promptcachelint.analyze(recorder.records).to_text())
