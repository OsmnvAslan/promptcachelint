from __future__ import annotations

from typing import Any

import pytest

from cachelint.model import Record, Usage

LONG = "You are a meticulous assistant. " * 200  # ~6.4k chars ≈ 1.6k tokens


def anthropic_body(
    *,
    system: str = LONG,
    tools: list[dict[str, Any]] | None = None,
    history: list[dict[str, Any]] | None = None,
    user: str = "hello",
    mark_system: bool = True,
    mark_last: bool = False,
    top_level: bool = False,
    model: str = "claude-opus-5",
    ttl: str | None = None,
) -> dict[str, Any]:
    cc: dict[str, Any] = {"type": "ephemeral"}
    if ttl:
        cc["ttl"] = ttl
    sys_block: dict[str, Any] = {"type": "text", "text": system}
    if mark_system:
        sys_block["cache_control"] = dict(cc)
    last: dict[str, Any] = {"type": "text", "text": user}
    if mark_last:
        last["cache_control"] = dict(cc)
    body: dict[str, Any] = {
        "model": model,
        "max_tokens": 1024,
        "system": [sys_block],
        "messages": [*(history or []), {"role": "user", "content": [last]}],
    }
    if tools is not None:
        body["tools"] = tools
    if top_level:
        body["cache_control"] = {"type": "ephemeral"}
    return body


def tool(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "description": f"{name} tool",
        "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
    }


def rec(
    body: dict[str, Any],
    at: float,
    *,
    provider: str = "anthropic",
    usage: Usage | None = None,
    session_id: str | None = None,
) -> Record:
    return Record(
        provider=provider,
        body=body,
        at=at,
        model=body.get("model"),
        usage=usage,
        session_id=session_id,
        session_explicit=session_id is not None,
    )


@pytest.fixture
def long_system() -> str:
    return LONG
