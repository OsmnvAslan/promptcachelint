from __future__ import annotations

import io
import json
import logging
from pathlib import Path

import pytest

from cachelint import (
    JsonlSink,
    LogWatcher,
    Recorder,
    analyze,
    assert_cache_stable,
    current_session,
    load_jsonl,
    session,
)
from cachelint.cli import main
from tests.conftest import LONG, anthropic_body


def test_recorder_assigns_sessions_and_parses_responses() -> None:
    r = Recorder()
    a = r.record(
        "anthropic",
        anthropic_body(mark_last=True),
        response={"id": "msg_1", "usage": {"input_tokens": 5, "cache_creation_input_tokens": 1600}},
    )
    b = r.record(
        "anthropic",
        anthropic_body(
            history=[
                {"role": "user", "content": [{"type": "text", "text": "hello"}]},
                {"role": "assistant", "content": "hi"},
            ],
            user="more",
            mark_last=True,
        ),
        sse_events=[
            {
                "type": "message_start",
                "message": {"usage": {"input_tokens": 2, "cache_read_input_tokens": 1600}},
            }
        ],
        stream=True,
    )
    assert a.session_id == b.session_id
    assert a.response_id == "msg_1" and a.usage is not None and a.usage.cache_write == 1600
    assert b.usage is not None and b.usage.cache_read == 1600 and b.stream


def test_explicit_session_context() -> None:
    r = Recorder()
    assert current_session() is None
    with session("abc") as sid:
        assert sid == "abc" and current_session() == "abc"
        x = r.record("anthropic", anthropic_body())
    with session() as sid2:
        y = r.record("anthropic", anthropic_body())
    assert x.session_id == "abc" and y.session_id == sid2 != "abc"
    assert current_session() is None


def test_jsonl_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    r = Recorder(JsonlSink(path))
    r.record("anthropic", anthropic_body(), response={"usage": {"input_tokens": 1}})
    r.record("openai", {"model": "gpt-5", "messages": [{"role": "user", "content": "x"}]})
    loaded = load_jsonl(path)
    assert [x.provider for x in loaded] == ["anthropic", "openai"]
    assert loaded[0].usage is not None and loaded[0].usage.input_tokens == 1
    assert loaded[0].to_dict() == r.records[0].to_dict()

    buf = io.StringIO()
    Recorder(JsonlSink(buf)).record("openai", {"messages": []})
    assert json.loads(buf.getvalue())["provider"] == "openai"


def test_assert_cache_stable() -> None:
    good = Recorder()
    good.record("anthropic", anthropic_body(mark_last=True))
    report = assert_cache_stable(good.records)
    assert report.totals.requests == 1

    bad = Recorder()
    bad.record("anthropic", anthropic_body(system=LONG + "v1"))
    bad.record("anthropic", anthropic_body(system=LONG + "v2"))
    with pytest.raises(AssertionError) as e:
        assert_cache_stable(bad.records)
    assert "CL001" in str(e.value) and "prefix broken" in str(e.value)
    assert_cache_stable(analyze(bad.records), fail_on=["CL013"])


def test_log_watcher_logs_breaks_once_and_static_once(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger="cachelint")
    r = Recorder(LogWatcher())
    with session("s"):
        r.record("anthropic", anthropic_body(system="short " + "2026-09-20T10:00:00Z"))
        r.record("anthropic", anthropic_body(system="short " + "2026-09-20T10:01:00Z"))
        r.record("anthropic", anthropic_body(system="short " + "2026-09-20T10:02:00Z"))
    msgs = [rec.getMessage() for rec in caplog.records]
    assert sum("CL001" in m for m in msgs) == 2
    assert sum("CL012" in m for m in msgs) == 1  # static finding reported once per session
    assert sum("CL011" in m for m in msgs) == 1


def test_cli_report_and_lint(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "trace.jsonl"
    r = Recorder(JsonlSink(path))
    r.record(
        "anthropic", anthropic_body(system=LONG + "v1"), response={"usage": {"input_tokens": 1}}
    )
    r.record(
        "anthropic", anthropic_body(system=LONG + "v2"), response={"usage": {"input_tokens": 1}}
    )

    assert main(["report", str(path)]) == 0
    out = capsys.readouterr().out
    assert "1 prefix break(s)" in out and "CL001" in out

    assert main(["report", str(path), "--fail-on", "CL001"]) == 1
    capsys.readouterr()
    assert main(["report", str(path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["totals"]["breaks"] == 1

    req = tmp_path / "req.json"
    req.write_text(json.dumps(anthropic_body(mark_system=False)))
    assert main(["lint", str(req), "--provider", "anthropic"]) == 1
    assert "CL010" in capsys.readouterr().out
    req.write_text(json.dumps(anthropic_body()))
    assert main(["lint", str(req), "--provider", "anthropic"]) == 0
    assert capsys.readouterr().out.strip() == "cachelint: no findings"
    req.write_text(json.dumps(anthropic_body(system="tiny", mark_system=False)))
    assert main(["lint", str(req), "--provider", "anthropic"]) == 0
    assert "below the" in capsys.readouterr().out
    assert main(["codes"]) == 0
    assert "CL001" in capsys.readouterr().out
