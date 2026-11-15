"""Reading a trace file that contains broken lines."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from promptcachelint import load_jsonl, read_trace
from promptcachelint.cli import main
from tests.conftest import anthropic_body


def test_read_trace_skips_bad_lines_and_counts_them(tmp_path: Path) -> None:
    good = json.dumps({"provider": "anthropic", "body": anthropic_body(), "at": 1})
    lines = [
        good,
        json.dumps({"provider": "martian", "body": anthropic_body(), "at": 2}),
        json.dumps({"provider": "anthropic", "at": 3}),
        '{"provider": "anth',
    ]
    path = tmp_path / "t.jsonl"
    path.write_text("\n".join(lines) + "\n")
    trace = read_trace(path)
    assert len(trace.records) == 1
    assert [n for n, _ in trace.skipped] == [2, 3, 4]
    assert "unknown provider" in trace.skipped[0][1]
    assert "no request body" in trace.skipped[1][1]
    assert "invalid JSON" in trace.skipped[2][1]
    assert len(load_jsonl(path)) == 1
    with pytest.raises(ValueError):
        load_jsonl(path, strict=True)


def test_cli_reports_skipped_lines_and_exit_codes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "t.jsonl"
    path.write_text(
        json.dumps({"provider": "anthropic", "body": anthropic_body(), "at": 1})
        + '\n{"provider": "anth'
    )
    assert main(["report", str(path)]) == 0
    out = capsys.readouterr().out
    assert "1 line(s) skipped" in out and "invalid JSON" in out
    assert main(["report", str(path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["skipped_lines"][0]["line"] == 2

    path.write_text('{"provider": "anth\n{"x":')
    assert main(["report", str(path)]) == 2
    assert "no records" in capsys.readouterr().err
