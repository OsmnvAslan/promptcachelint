"""cachelint: explain why your LLM prompt cache missed.

Record requests (explicitly, or through the httpx2 transport), then ask why
``cache_read`` was zero: the report names the exact block and offset where the
prefix diverged, flags silent invalidators, and totals the hit ratio.
"""

from importlib.metadata import PackageNotFoundError, version

from cachelint import findings, redact
from cachelint.analyze import Analyzer, Report, RequestReport, SessionReport, Totals, analyze
from cachelint.detectors import lint_request
from cachelint.diff import PrefixDiff, diff_prefix
from cachelint.findings import Finding
from cachelint.live import LogWatcher
from cachelint.model import Record, Segment, Usage, estimate_tokens, estimate_tokens_from_chars
from cachelint.providers import detect_provider, get_provider, register_provider
from cachelint.recorder import JsonlSink, Recorder, load_jsonl
from cachelint.sessions import SessionIndex, current_session, session
from cachelint.testing import assert_cache_stable

try:
    __version__ = version("cachelint")
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
    "redact",
    "register_provider",
    "session",
]
