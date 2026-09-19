"""cachelint: explain why your LLM prompt cache missed.

Record requests (explicitly, or through the httpx transport), then ask why
``cache_read`` was zero: the report names the exact block and offset where the
prefix diverged, flags silent invalidators, and totals the hit ratio.
"""

from cachelint import findings
from cachelint.analyze import Report, RequestReport, SessionReport, Totals, analyze
from cachelint.detectors import lint_request
from cachelint.diff import PrefixDiff, diff_prefix
from cachelint.findings import Finding
from cachelint.live import LogWatcher
from cachelint.model import Record, Segment, Usage, estimate_tokens
from cachelint.providers import detect_provider, get_provider, register_provider
from cachelint.recorder import JsonlSink, Recorder, load_jsonl
from cachelint.sessions import SessionIndex, current_session, session
from cachelint.testing import assert_cache_stable

__version__ = "0.1.0"

__all__ = [
    "Finding",
    "JsonlSink",
    "LogWatcher",
    "PrefixDiff",
    "Record",
    "Recorder",
    "Report",
    "RequestReport",
    "Segment",
    "SessionIndex",
    "SessionReport",
    "Totals",
    "Usage",
    "__version__",
    "analyze",
    "assert_cache_stable",
    "current_session",
    "detect_provider",
    "diff_prefix",
    "estimate_tokens",
    "findings",
    "get_provider",
    "lint_request",
    "load_jsonl",
    "register_provider",
    "session",
]
