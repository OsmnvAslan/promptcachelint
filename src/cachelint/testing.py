"""pytest helpers: fail a test when the cache prefix is not stable."""

from __future__ import annotations

from collections.abc import Iterable

from cachelint import findings as F
from cachelint.analyze import Report, analyze
from cachelint.model import Record

DEFAULT_FAIL_ON: frozenset[str] = frozenset(
    {F.PREFIX_BROKEN, F.TOOLS_CHANGED, F.NO_BREAKPOINT, F.TOO_MANY_BREAKPOINTS}
)


def assert_cache_stable(
    records: Iterable[Record] | Report,
    *,
    fail_on: Iterable[str] = DEFAULT_FAIL_ON,
) -> Report:
    """Raise ``AssertionError`` with the rendered report if any failing code appears.

    Returns the report so a test can make further assertions on totals.
    """
    report = records if isinstance(records, Report) else analyze(records)
    codes = set(fail_on)
    bad = [f for f in report.findings if f.code in codes]
    if bad:
        raise AssertionError(
            f"cachelint: {len(bad)} cache problem(s) "
            f"({', '.join(sorted({f.code for f in bad}))})\n\n{report.to_text()}"
        )
    return report
