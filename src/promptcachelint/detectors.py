"""Static detectors: things visible in a single request, no history needed."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from promptcachelint import findings as F
from promptcachelint.model import Segment, estimate_tokens_from_chars
from promptcachelint.providers.anthropic import MAX_BREAKPOINTS
from promptcachelint.providers.base import Provider


@dataclass(frozen=True, slots=True)
class Pattern:
    name: str
    regex: re.Pattern[str]
    hint: str


VOLATILE_PATTERNS: tuple[Pattern, ...] = (
    Pattern(
        "iso-datetime",
        re.compile(
            r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?"
        ),
        "Timestamps change every request. Move them after the last breakpoint or drop them.",
    ),
    Pattern(
        "uuid",
        re.compile(
            r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
        ),
        "A per-request or per-session id inside the prefix. Move it after the last breakpoint.",
    ),
    Pattern(
        "hex-id",
        re.compile(r"\b[0-9a-f]{24,64}\b"),
        "Looks like a random hex id. If it varies per request, it invalidates the cache.",
    ),
    Pattern(
        "date-phrase",
        re.compile(
            r"(?i)\b(today(?:'s date)? is|current (?:date|time)(?: is)?|the date is|now is)"
            r"\b[^\n]{0,40}"
        ),
        "An interpolated date in the system prompt. Put it in the last user turn instead.",
    ),
    Pattern(
        "rfc-date",
        re.compile(
            r"\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun), \d{1,2} "
            r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d{4}"
        ),
        "Timestamps change every request. Move them after the last breakpoint or drop them.",
    ),
)


def volatile_content(segments: list[Segment], cacheable: int) -> list[F.Finding]:
    """Flag volatile-looking values inside the cacheable prefix."""
    out: list[F.Finding] = []
    for seg in segments[:cacheable]:
        for pat in VOLATILE_PATTERNS:
            m = pat.regex.search(seg.text)
            if not m:
                continue
            out.append(
                F.Finding(
                    code=F.VOLATILE_CONTENT,
                    severity="warning",
                    message=f"{pat.name} inside the cacheable prefix",
                    hint=pat.hint,
                    path=seg.path,
                    offset=m.start(),
                    excerpt=m.group(0)[:80],
                    data={"pattern": pat.name},
                )
            )
            break  # one finding per segment is enough
    return out


def structure(
    provider: Provider, body: dict[str, Any], segments: list[Segment], cacheable: int
) -> list[F.Finding]:
    """Provider structure checks: markers present, prefix long enough, marker count."""
    out: list[F.Finding] = []
    model = body.get("model") if isinstance(body.get("model"), str) else None
    minimum = provider.min_prefix_tokens(model)
    prefix_tokens = estimate_tokens_from_chars(sum(seg.chars for seg in segments[:cacheable]))
    total_tokens = estimate_tokens_from_chars(sum(seg.chars for seg in segments))

    if provider.explicit_markers:
        markers = provider.marker_slots(body, segments)
        if markers > MAX_BREAKPOINTS:
            out.append(
                F.Finding(
                    code=F.TOO_MANY_BREAKPOINTS,
                    severity="error",
                    message=(
                        f"{markers} cache_control slots used (explicit markers plus the "
                        f"top-level automatic one); the API allows {MAX_BREAKPOINTS}"
                    ),
                    hint="One per stability boundary: tools+system, shared context, last turn.",
                    data={"markers": markers},
                )
            )
        if cacheable == 0:
            if total_tokens >= minimum:
                out.append(
                    F.Finding(
                        code=F.NO_BREAKPOINT,
                        severity="error",
                        message=(
                            f"no cache_control anywhere; ~{total_tokens} tokens are re-billed "
                            "at full price on every request"
                        ),
                        hint=(
                            "Add cache_control to the last system block (caches tools+system) "
                            "or pass a top-level cache_control for automatic placement."
                        ),
                        data={"estimated_tokens": total_tokens, "minimum": minimum},
                    )
                )
            return out

    if 0 < prefix_tokens < minimum:
        out.append(
            F.Finding(
                code=F.PREFIX_TOO_SHORT,
                severity="warning",
                message=(
                    f"cacheable prefix is ~{prefix_tokens} tokens; {model or 'this model'} "
                    f"caches from {minimum} (estimate, ~4 chars/token)"
                ),
                hint="Below the minimum the marker is accepted but nothing is written or read.",
                data={"estimated_tokens": prefix_tokens, "minimum": minimum},
            )
        )
    return out


def lint_request(provider: Provider, body: dict[str, Any]) -> list[F.Finding]:
    """All single-request findings for one body."""
    segments = provider.segments(body)
    cacheable = provider.cacheable_segments(body, segments)
    return structure(provider, body, segments, cacheable) + volatile_content(segments, cacheable)
