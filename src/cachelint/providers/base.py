"""Provider protocol: everything cachelint knows about one API lives behind it."""

from __future__ import annotations

import json
from typing import Any, Protocol, runtime_checkable

from cachelint.model import Segment, Usage


def canonical(value: Any) -> str:
    """Deterministic JSON for structured blocks (key order is irrelevant on the wire)."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


@runtime_checkable
class Provider(Protocol):
    """Adapter for one provider's wire format.

    Only this object knows how a request body renders into a cache prefix and
    how the response reports cache usage. When a provider changes its API,
    this is the one place to update.
    """

    name: str

    def matches(self, url: str) -> bool:
        """True if requests to ``url`` belong to this provider."""
        ...

    def segments(self, body: dict[str, Any]) -> list[Segment]:
        """The prefix in render order, cache markers stripped from ``text``."""
        ...

    def cacheable_segments(self, body: dict[str, Any], segments: list[Segment]) -> int:
        """How many leading segments the provider would try to serve from cache."""
        ...

    def usage(self, response: dict[str, Any]) -> Usage | None:
        """Normalized usage from a non-streaming response body."""
        ...

    def usage_from_sse(self, events: list[dict[str, Any]]) -> Usage | None:
        """Normalized usage from parsed SSE event payloads."""
        ...

    def min_prefix_tokens(self, model: str | None) -> int:
        """Smallest prefix the provider will cache for ``model``."""
        ...

    def scope(self, body: dict[str, Any]) -> dict[str, str]:
        """Top-level parameters whose change invalidates part of the cache.

        Returns ``{name: canonical_value}``; ``name`` is used in findings.
        """
        ...
