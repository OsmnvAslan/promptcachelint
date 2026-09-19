"""Anthropic Messages API.

Render order is ``tools -> system -> messages``. Explicit ``cache_control``
markers on content blocks (max 4) or a top-level ``cache_control`` (automatic
placement on the last cacheable block) define where reads can land. The cache
key is the exact bytes up to each marker, so the marker itself is stripped
before segments are compared.
"""

from __future__ import annotations

import re
from typing import Any

from cachelint.model import Segment, Usage
from cachelint.providers.base import canonical

_URL = re.compile(r"/v1/messages(?:\?|$|/count_tokens)")

# Minimum cacheable prefix per model family. Shorter prefixes silently do not cache.
_MIN_TOKENS: list[tuple[re.Pattern[str], int]] = [
    (re.compile(r"claude-(opus-5|fable-5|mythos-5)"), 512),
    (re.compile(r"claude-(opus-4-6|opus-4-5|haiku-4-5)"), 4096),
    (re.compile(r"claude-opus-4-7|claude-haiku-3-5|claude-3-5-haiku"), 2048),
]
_DEFAULT_MIN_TOKENS = 1024

MAX_BREAKPOINTS = 4


def _strip_cache_control(block: Any) -> Any:
    if isinstance(block, dict) and "cache_control" in block:
        return {k: v for k, v in block.items() if k != "cache_control"}
    return block


def _marker(block: Any) -> tuple[bool, str | None]:
    if isinstance(block, dict):
        cc = block.get("cache_control")
        if isinstance(cc, dict):
            ttl = cc.get("ttl")
            return True, str(ttl) if ttl is not None else "5m"
    return False, None


class AnthropicProvider:
    name = "anthropic"

    def matches(self, url: str) -> bool:
        return "anthropic" in url and bool(_URL.search(url))

    def segments(self, body: dict[str, Any]) -> list[Segment]:
        out: list[Segment] = []

        for i, tool in enumerate(body.get("tools") or []):
            bp, ttl = _marker(tool)
            out.append(
                Segment(
                    path=f"tools[{i}]",
                    kind="tool",
                    text=canonical(_strip_cache_control(tool)),
                    breakpoint=bp,
                    ttl=ttl,
                )
            )

        system = body.get("system")
        if isinstance(system, str):
            out.append(Segment(path="system", kind="system", text=system))
        elif isinstance(system, list):
            for i, block in enumerate(system):
                bp, ttl = _marker(block)
                out.append(
                    Segment(
                        path=f"system[{i}]",
                        kind="system",
                        text=_block_text(block),
                        breakpoint=bp,
                        ttl=ttl,
                    )
                )

        for m, message in enumerate(body.get("messages") or []):
            role = message.get("role", "?") if isinstance(message, dict) else "?"
            content = message.get("content") if isinstance(message, dict) else None
            if isinstance(content, str):
                out.append(
                    Segment(
                        path=f"messages[{m}]",
                        kind="message",
                        text=f"{role}: {content}",
                    )
                )
            elif isinstance(content, list):
                for c, block in enumerate(content):
                    bp, ttl = _marker(block)
                    out.append(
                        Segment(
                            path=f"messages[{m}].content[{c}]",
                            kind="message",
                            text=f"{role}: {_block_text(block)}",
                            breakpoint=bp,
                            ttl=ttl,
                        )
                    )
        return out

    def cacheable_segments(self, body: dict[str, Any], segments: list[Segment]) -> int:
        last = 0
        for i, seg in enumerate(segments):
            if seg.breakpoint:
                last = i + 1
        top_level = isinstance(body.get("cache_control"), dict)
        if top_level:
            # Automatic caching: marker lands on the last cacheable block.
            return max(last, len(segments))
        return last

    def usage(self, response: dict[str, Any]) -> Usage | None:
        u = response.get("usage")
        return _usage(u) if isinstance(u, dict) else None

    def usage_from_sse(self, events: list[dict[str, Any]]) -> Usage | None:
        start: dict[str, Any] | None = None
        output: int | None = None
        for ev in events:
            t = ev.get("type")
            if t == "message_start":
                msg = ev.get("message")
                if isinstance(msg, dict) and isinstance(msg.get("usage"), dict):
                    start = msg["usage"]
            elif t == "message_delta":
                u = ev.get("usage")
                if isinstance(u, dict) and isinstance(u.get("output_tokens"), int):
                    output = u["output_tokens"]
        if start is None:
            return None
        base = _usage(start)
        if base is None:
            return None
        return Usage(base.input_tokens, base.cache_read, base.cache_write, output)

    def min_prefix_tokens(self, model: str | None) -> int:
        if model:
            for pattern, minimum in _MIN_TOKENS:
                if pattern.search(model):
                    return minimum
        return _DEFAULT_MIN_TOKENS

    def scope(self, body: dict[str, Any]) -> dict[str, str]:
        out: dict[str, str] = {}
        for key in ("model", "thinking"):
            if key in body:
                out[key] = canonical(body[key])
        effort = (body.get("output_config") or {}).get("effort")
        if effort is not None:
            out["output_config.effort"] = canonical(effort)
        return out


def _block_text(block: Any) -> str:
    if (
        isinstance(block, dict)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
    ):
        return str(block["text"])
    return canonical(_strip_cache_control(block))


def _usage(u: dict[str, Any]) -> Usage | None:
    if not isinstance(u.get("input_tokens"), int):
        return None
    return Usage(
        input_tokens=u["input_tokens"],
        cache_read=int(u.get("cache_read_input_tokens") or 0),
        cache_write=int(u.get("cache_creation_input_tokens") or 0),
        output_tokens=u.get("output_tokens") if isinstance(u.get("output_tokens"), int) else None,
    )
