"""The same six-turn tool-using conversation, run twice against the real API.

Run A ("before") interpolates the current time into the system prompt, the
most common way to break a prompt cache. Run B ("after") moves that line into
the latest user turn. Both runs are recorded; the script prints both reports
and the difference in what the provider actually billed.

    export ANTHROPIC_API_KEY=...
    uv run --with anthropic examples/before_after.py

Writes trace-before.jsonl and trace-after.jsonl next to the script. Costs a
few cents: twelve small requests with a ~2k-token system prompt.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

import anthropic
import httpx2

import promptcachelint
from promptcachelint.redact import strip_media
from promptcachelint.transport import wrap_transport

MODEL = "claude-opus-5"
HERE = Path(__file__).resolve().parent

RULES = (
    "You are a travel assistant for a small agency. Answer briefly and concretely. "
    "Prefer the get_weather tool over guessing. When a city is ambiguous, ask which one. "
    "Never invent prices. Use metric units. Keep answers under three sentences. "
) * 40  # ~2k tokens: comfortably above the cacheable minimum

TOOLS: list[dict[str, Any]] = [
    {
        "name": "get_weather",
        "description": "Current weather for a city.",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }
]

QUESTIONS = [
    "What's the weather in Lisbon right now?",
    "And in Porto?",
    "Which of the two is better for a walk today?",
    "Should I pack an umbrella for Porto tomorrow?",
    "Give me one sentence I can send to my client about Lisbon.",
    "Thanks. Summarise the two cities in one line each.",
]


def fake_weather(city: str) -> str:
    table = {"lisbon": "24 °C, sunny, light breeze", "porto": "19 °C, overcast, 40% rain"}
    return table.get(city.lower(), "no data")


def run(label: str, *, time_in_system: bool) -> promptcachelint.Report:
    recorder = promptcachelint.Recorder(
        promptcachelint.JsonlSink(HERE / f"trace-{label}.jsonl", redact=strip_media)
    )
    client = anthropic.Anthropic(
        http_client=anthropic.DefaultHttpxClient(
            transport=wrap_transport(httpx2.HTTPTransport(), recorder)
        )
    )
    messages: list[dict[str, Any]] = []

    with promptcachelint.session(label):
        for question in QUESTIONS:
            now = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
            system_text = RULES + (f"\nCurrent time: {now}" if time_in_system else "")
            user_text = question if time_in_system else f"{question}\n(Current time: {now})"
            messages.append({"role": "user", "content": [{"type": "text", "text": user_text}]})

            while True:
                response = client.messages.create(
                    model=MODEL,
                    max_tokens=300,
                    system=[
                        {
                            "type": "text",
                            "text": system_text,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    tools=TOOLS,
                    messages=_mark_last(messages),
                )
                messages.append({"role": "assistant", "content": _plain(response.content)})
                if response.stop_reason != "tool_use":
                    break
                results = [
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": fake_weather(block.input.get("city", "")),
                    }
                    for block in response.content
                    if block.type == "tool_use"
                ]
                messages.append({"role": "user", "content": results})

    return promptcachelint.analyze(recorder.records)


def _plain(content: Any) -> list[dict[str, Any]]:
    """SDK blocks -> plain dicts so the request body is JSON-serialisable and stable."""
    return [json.loads(block.model_dump_json(exclude_none=True)) for block in content]


def _mark_last(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy of the history with cache_control on the last block of the last message."""
    out = json.loads(json.dumps(messages))
    last = out[-1]["content"]
    if isinstance(last, list) and last:
        last[-1]["cache_control"] = {"type": "ephemeral"}
    return out


def main() -> int:
    if not anthropic.Anthropic().api_key:
        print("ANTHROPIC_API_KEY is not set", file=sys.stderr)
        return 2
    before = run("before", time_in_system=True)
    after = run("after", time_in_system=False)

    for label, report in (
        ("BEFORE: time in the system prompt", before),
        ("AFTER: time in the last user turn", after),
    ):
        print("=" * 80)
        print(label)
        print("=" * 80)
        print(report.to_text())
        print()

    b, a = before.totals, after.totals
    print("=" * 80)
    print(f"{'':<28}{'before':>12}{'after':>12}")
    print(f"{'requests':<28}{b.requests:>12}{a.requests:>12}")
    print(f"{'prompt tokens billed':<28}{b.prompt_tokens:>12}{a.prompt_tokens:>12}")
    print(f"{'  read from cache':<28}{b.cache_read:>12}{a.cache_read:>12}")
    print(f"{'  written to cache':<28}{b.cache_write:>12}{a.cache_write:>12}")
    print(f"{'  uncached':<28}{b.input_tokens:>12}{a.input_tokens:>12}")
    print(f"{'hit ratio':<28}{b.hit_ratio:>11.1%}{a.hit_ratio:>12.1%}")
    print(f"{'prefix breaks':<28}{b.breaks:>12}{a.breaks:>12}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
