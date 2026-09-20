"""Anthropic Messages API.

Render order is ``tools -> system -> messages``. Explicit ``cache_control``
markers on content blocks (max 4 slots, the top-level automatic marker takes
one) define where reads can land. The cache key is the exact bytes up to each
marker, so the marker itself is stripped before segments are compared.
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
TTL_SECONDS = {"5m": 5 * 60, "1h": 60 * 60}

# Top-level parameters and what their change invalidates (from the API's
# invalidation hierarchy): tools+system+messages, or messages only.
SCOPE_KEYS: dict[str, str] = {
    "model": "Caches are model-scoped; a model switch rebuilds everything.",
    "thinking": "Thinking changes invalidate the messages cache (and more on some models). "
    "Pin it per route.",
    "output_config.effort": "Effort changes invalidate the messages cache. Pin it per route, "
    "or change it via a mid-conversation system message where supported.",
    "tool_choice": "tool_choice changes invalidate the messages cache (tools+system survive).",
    "speed": "Toggling speed invalidates the system and messages caches.",
}


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


def _block(block: Any) -> tuple[str, str | None]:
    """(text, block type) for one content block."""
    if isinstance(block, dict):
        kind = block.get("type")
        if kind == "text" and isinstance(block.get("text"), str):
            return str(block["text"]), "text"
        return canonical(_strip_cache_control(block)), str(kind) if kind else None
    return canonical(block), None


class AnthropicProvider:
    name = "anthropic"
    explicit_markers = True
    cache_granularity_tokens = 0
    best_effort = False

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
            out.append(Segment(path="system", kind="system", text=system, block="text"))
        elif isinstance(system, list):
            for i, item in enumerate(system):
                bp, ttl = _marker(item)
                text, block = _block(item)
                out.append(
                    Segment(
                        path=f"system[{i}]",
                        kind="system",
                        text=text,
                        block=block,
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
                        text=content,
                        role=str(role),
                        block="text",
                    )
                )
            elif isinstance(content, list):
                for c, item in enumerate(content):
                    bp, ttl = _marker(item)
                    text, block = _block(item)
                    out.append(
                        Segment(
                            path=f"messages[{m}].content[{c}]",
                            kind="message",
                            text=text,
                            role=str(role),
                            block=block,
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
        if isinstance(body.get("cache_control"), dict):
            # Automatic caching: marker lands on the last cacheable block.
            return max(last, len(segments))
        return last

    def marker_slots(self, body: dict[str, Any], segments: list[Segment]) -> int:
        explicit = sum(1 for seg in segments if seg.breakpoint)
        automatic = 1 if isinstance(body.get("cache_control"), dict) else 0
        return explicit + automatic

    def ttl_seconds(self, segments: list[Segment]) -> int:
        """Longest TTL among the markers: a 1h entry on the head stays readable
        after the 5m entry on the last turn expired."""
        ttls = [
            TTL_SECONDS.get(seg.ttl or "5m", TTL_SECONDS["5m"])
            for seg in segments
            if seg.breakpoint
        ]
        return max(ttls, default=TTL_SECONDS["5m"])

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
        for key in ("model", "thinking", "tool_choice", "speed"):
            if key in body:
                out[key] = canonical(body[key])
        effort = (body.get("output_config") or {}).get("effort")
        if effort is not None:
            out["output_config.effort"] = canonical(effort)
        return out


def _usage(u: dict[str, Any]) -> Usage | None:
    if not isinstance(u.get("input_tokens"), int):
        return None
    return Usage(
        input_tokens=u["input_tokens"],
        cache_read=int(u.get("cache_read_input_tokens") or 0),
        cache_write=int(u.get("cache_creation_input_tokens") or 0),
        output_tokens=u.get("output_tokens") if isinstance(u.get("output_tokens"), int) else None,
    )
