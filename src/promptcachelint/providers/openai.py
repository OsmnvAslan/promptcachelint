"""OpenAI Chat Completions and Responses APIs.

Caching is automatic: the server reuses the longest previously seen prefix
(in 128-token steps, from 1024 tokens). There are no markers, so every
segment is potentially cacheable and the only things that matter are prefix
stability and ``prompt_cache_key`` routing. The cache is documented as
best-effort and entries live 5-10 minutes (longer off-peak).
"""

from __future__ import annotations

import re
from typing import Any

from promptcachelint.model import Segment, SegmentKind, Usage
from promptcachelint.providers.base import canonical

_URL = re.compile(r"/v1/(chat/completions|responses)(?:\?|$)")

SCOPE_KEYS: dict[str, str] = {
    "model": "Caches are model-scoped; a model switch rebuilds everything.",
    "prompt_cache_key": "prompt_cache_key routes to a cache shard; changing it per request "
    "defeats sharing.",
}


def _part(part: Any) -> tuple[str, str | None]:
    if isinstance(part, dict):
        kind = part.get("type")
        if kind in ("text", "input_text", "output_text") and isinstance(part.get("text"), str):
            return str(part["text"]), str(kind)
        return canonical(part), str(kind) if kind else None
    return canonical(part), None


class OpenAIProvider:
    name = "openai"
    explicit_markers = False
    cache_granularity_tokens = 128
    best_effort = True

    def matches(self, url: str) -> bool:
        return "openai" in url and bool(_URL.search(url))

    def segments(self, body: dict[str, Any]) -> list[Segment]:
        out: list[Segment] = []
        for i, tool in enumerate(body.get("tools") or []):
            out.append(Segment(path=f"tools[{i}]", kind="tool", text=canonical(tool)))

        if "input" in body or "instructions" in body:
            return out + self._responses_segments(body)
        return out + self._chat_segments(body)

    @staticmethod
    def _chat_segments(body: dict[str, Any]) -> list[Segment]:
        out: list[Segment] = []
        for m, message in enumerate(body.get("messages") or []):
            role = str(message.get("role", "?")) if isinstance(message, dict) else "?"
            content = message.get("content") if isinstance(message, dict) else None
            kind: SegmentKind = "system" if role in ("system", "developer") else "message"
            seg_role = None if kind == "system" else role
            if isinstance(content, list):
                for c, part in enumerate(content):
                    text, block = _part(part)
                    out.append(
                        Segment(
                            path=f"messages[{m}].content[{c}]",
                            kind=kind,
                            text=text,
                            role=seg_role,
                            block=block,
                        )
                    )
                continue
            rest = {k: v for k, v in message.items() if k not in ("role", "content")}
            if isinstance(content, str) and not rest:
                out.append(
                    Segment(
                        path=f"messages[{m}]", kind=kind, text=content, role=seg_role, block="text"
                    )
                )
            else:
                # tool_calls / tool results / refusals: one structured block
                payload = {k: v for k, v in message.items() if k != "role"}
                out.append(
                    Segment(
                        path=f"messages[{m}]",
                        kind=kind,
                        text=canonical(payload),
                        role=seg_role,
                        block="tool_calls" if "tool_calls" in rest else "structured",
                    )
                )
        return out

    @staticmethod
    def _responses_segments(body: dict[str, Any]) -> list[Segment]:
        out: list[Segment] = []
        instructions = body.get("instructions")
        if isinstance(instructions, str):
            out.append(Segment(path="instructions", kind="system", text=instructions, block="text"))
        items = body.get("input")
        if isinstance(items, str):
            out.append(Segment(path="input", kind="message", text=items, role="user", block="text"))
        elif isinstance(items, list):
            for i, item in enumerate(items):
                if not isinstance(item, dict):
                    out.append(Segment(path=f"input[{i}]", kind="message", text=canonical(item)))
                    continue
                role = item.get("role")
                content = item.get("content")
                if isinstance(content, list):
                    for c, part in enumerate(content):
                        text, block = _part(part)
                        out.append(
                            Segment(
                                path=f"input[{i}].content[{c}]",
                                kind="message",
                                text=text,
                                role=str(role) if role else None,
                                block=block,
                            )
                        )
                elif isinstance(content, str):
                    out.append(
                        Segment(
                            path=f"input[{i}]",
                            kind="message",
                            text=content,
                            role=str(role) if role else None,
                            block="text",
                        )
                    )
                else:
                    out.append(
                        Segment(
                            path=f"input[{i}]",
                            kind="message",
                            text=canonical(item),
                            role=str(role) if role else None,
                            block=str(item.get("type")) if item.get("type") else None,
                        )
                    )
        return out

    def cacheable_segments(self, body: dict[str, Any], segments: list[Segment]) -> int:
        return len(segments)

    def marker_slots(self, body: dict[str, Any], segments: list[Segment]) -> int:
        return 0

    def ttl_seconds(self, segments: list[Segment]) -> int:
        return 10 * 60

    def usage(self, response: dict[str, Any]) -> Usage | None:
        u = response.get("usage")
        return _usage(u) if isinstance(u, dict) else None

    def usage_from_sse(self, events: list[dict[str, Any]]) -> Usage | None:
        for ev in reversed(events):
            # Chat completions: a final chunk with `usage` (needs stream_options.include_usage).
            u = ev.get("usage")
            if isinstance(u, dict):
                return _usage(u)
            # Responses API: `response.completed` carries the full response.
            resp = ev.get("response")
            if isinstance(resp, dict) and isinstance(resp.get("usage"), dict):
                return _usage(resp["usage"])
        return None

    def min_prefix_tokens(self, model: str | None) -> int:
        return 1024

    def scope(self, body: dict[str, Any]) -> dict[str, str]:
        out: dict[str, str] = {}
        for key in ("model", "prompt_cache_key"):
            if key in body:
                out[key] = canonical(body[key])
        return out


def _usage(u: dict[str, Any]) -> Usage | None:
    # Chat completions: prompt_tokens (includes cached) + prompt_tokens_details.cached_tokens.
    # Responses: input_tokens (includes cached) + input_tokens_details.cached_tokens.
    if isinstance(u.get("prompt_tokens"), int):
        total = u["prompt_tokens"]
        details = u.get("prompt_tokens_details") or {}
        output = u.get("completion_tokens")
    elif isinstance(u.get("input_tokens"), int):
        total = u["input_tokens"]
        details = u.get("input_tokens_details") or {}
        output = u.get("output_tokens")
    else:
        return None
    cached = int(details.get("cached_tokens") or 0) if isinstance(details, dict) else 0
    return Usage(
        input_tokens=max(0, total - cached),
        cache_read=cached,
        cache_write=0,
        output_tokens=output if isinstance(output, int) else None,
    )
