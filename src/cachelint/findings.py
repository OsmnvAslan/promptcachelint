"""Finding codes, one per thing that can go wrong with a prompt cache."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Severity = Literal["error", "warning", "info"]


@dataclass(frozen=True, slots=True)
class Finding:
    code: str
    severity: Severity
    message: str
    hint: str = ""
    path: str | None = None
    offset: int | None = None
    excerpt: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "hint": self.hint,
            "path": self.path,
            "offset": self.offset,
            "excerpt": self.excerpt,
            "data": self.data,
        }


# Session-level (need the previous request)
PREFIX_BROKEN = "CL001"
TOOLS_CHANGED = "CL002"
SCOPE_CHANGED = "CL003"
BREAKPOINT_MOVED_BACK = "CL004"
UNEXPLAINED_MISS = "CL005"
TTL_GAP = "CL006"
LOOKBACK_EXCEEDED = "CL007"
# Single-request (static)
NO_BREAKPOINT = "CL010"
PREFIX_TOO_SHORT = "CL011"
VOLATILE_CONTENT = "CL012"
TOO_MANY_BREAKPOINTS = "CL013"
WRITE_WITHOUT_READ = "CL014"

DESCRIPTIONS: dict[str, str] = {
    PREFIX_BROKEN: "content inside the previously cached prefix changed (confirmed by diff)",
    TOOLS_CHANGED: "tool definitions changed between requests (invalidates everything)",
    SCOPE_CHANGED: "a top-level parameter changed (model / thinking / effort / cache key)",
    BREAKPOINT_MOVED_BACK: "the last cache breakpoint moved earlier than in the previous request",
    UNEXPLAINED_MISS: "prefix was intact but the provider reported no cache read",
    TTL_GAP: "time since the previous request exceeded the cache TTL",
    LOOKBACK_EXCEEDED: "more than 20 positions appended in one turn (breakpoint lookback window)",
    NO_BREAKPOINT: "no cache_control marker anywhere; nothing can be read from cache",
    PREFIX_TOO_SHORT: "cacheable prefix is below the model's minimum (estimate)",
    VOLATILE_CONTENT: "looks volatile (timestamp / UUID / random id) inside the cacheable "
    "prefix (suspicion; CL001 confirms)",
    TOO_MANY_BREAKPOINTS: "more than 4 cache_control markers in one request",
    WRITE_WITHOUT_READ: "every request writes cache but reads nothing back",
}
