"""promptcachelint: explain why your LLM prompt cache missed.

Record requests (explicitly, or through the httpx2 transport), then ask why
``cache_read`` was zero: the report names the exact block and offset where the
prefix diverged, flags silent invalidators, and totals the hit ratio.
"""

from importlib.metadata import PackageNotFoundError, version

from promptcachelint import findings, redact
from promptcachelint.analyze import Analyzer, Report, RequestReport, SessionReport, Totals, analyze
from promptcachelint.detectors import lint_request
from promptcachelint.diff import PrefixDiff, diff_prefix
from promptcachelint.findings import Finding
from promptcachelint.live import LogWatcher
from promptcachelint.model import (
    Record,
    Segment,
    Usage,
    estimate_tokens,
    estimate_tokens_from_chars,
)
from promptcachelint.providers import detect_provider, get_provider, register_provider
from promptcachelint.recorder import JsonlSink, Recorder, Trace, load_jsonl, read_trace
from promptcachelint.sessions import SessionIndex, current_session, session
from promptcachelint.testing import assert_cache_stable

try:
    __version__ = version("promptcachelint")
except PackageNotFoundError:  # pragma: no cover - source checkout without install
    __version__ = "0.0.0"

__all__ = [
    "Analyzer",
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
    "Trace",
    "Usage",
    "__version__",
    "analyze",
    "assert_cache_stable",
    "current_session",
    "detect_provider",
    "diff_prefix",
    "estimate_tokens",
    "estimate_tokens_from_chars",
    "findings",
    "get_provider",
    "lint_request",
    "load_jsonl",
    "read_trace",
    "redact",
    "register_provider",
    "session",
]
