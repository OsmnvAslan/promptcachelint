"""Plain-text rendering of a report."""

from __future__ import annotations

from typing import TYPE_CHECKING

from promptcachelint import findings as F

if TYPE_CHECKING:
    from promptcachelint.analyze import Report, RequestReport


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _fmt_request(r: RequestReport) -> list[str]:
    u = r.usage
    usage = (
        f"in={u.input_tokens} read={u.cache_read} write={u.cache_write}"
        if u is not None
        else f"usage: n/a ({r.record.note})"
        if r.record.note
        else "usage: n/a"
    )
    head = (
        f"  #{r.index}  {r.record.model or '?'}  "
        f"segments={len(r.segments)} cacheable={r.cacheable}  {usage}"
    )
    lines = [head]
    if r.diff is not None and r.diff.broken:
        who = f" ({r.diff.role})" if r.diff.role else ""
        lines.append(
            f"      ✗ prefix broken at {r.diff.path}{who} +{r.diff.offset} "
            f"(~{r.diff.lost_tokens_estimate} tokens lost, estimate)"
        )
        lines.append(f"        was: {r.diff.before!r}")
        lines.append(f"        now: {r.diff.after!r}")
    for f in r.findings:
        if f.code == F.PREFIX_BROKEN and r.diff is not None and r.diff.broken:
            continue  # already shown above
        mark = {"error": "✗", "warning": "!", "info": "·"}[f.severity]
        where = f" at {f.path}" if f.path and f.code not in (F.PREFIX_BROKEN,) else ""
        lines.append(f"      {mark} {f.code} {f.message}{where}")
        if f.hint:
            lines.append(f"        → {f.hint}")
    return lines


def render_text(report: Report) -> str:
    lines: list[str] = []
    t = report.totals
    lines.append(
        f"promptcachelint: {t.requests} requests in {len(report.sessions)} session(s); "
        f"prompt tokens {t.prompt_tokens} (read {t.cache_read}, write {t.cache_write}, "
        f"uncached {t.input_tokens}); hit ratio {_pct(t.hit_ratio)}; "
        f"{t.breaks} prefix break(s), ~{t.lost_tokens_estimate} tokens lost (estimate)"
    )
    if t.with_usage < t.requests:
        lines.append(f"  ({t.requests - t.with_usage} request(s) without usage data)")
    if report.auto_grouped:
        lines.append(
            "  (sessions grouped automatically by first message; use promptcachelint.session(id) "
            "for exact grouping)"
        )
    for s in report.sessions:
        st = s.totals
        lines.append("")
        lines.append(
            f"session {s.id[:12]} [{s.provider}] "
            f"{st.requests} req, hit ratio {_pct(st.hit_ratio)}, {st.breaks} break(s)"
        )
        for r in s.requests:
            lines.extend(_fmt_request(r))
    codes = sorted({f.code for f in report.findings})
    if codes:
        lines.append("")
        lines.append("codes: " + "; ".join(f"{c} {F.DESCRIPTIONS.get(c, '')}" for c in codes))
    return "\n".join(lines)
