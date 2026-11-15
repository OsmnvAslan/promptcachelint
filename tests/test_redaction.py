"""Redaction helpers keep traces analysable: lengths, distinctness and markers survive."""

from __future__ import annotations

from typing import Any

from promptcachelint import JsonlSink, Recorder, analyze, load_jsonl
from promptcachelint.redact import hash_text, strip_media
from tests.conftest import LONG, anthropic_body, rec


def test_redaction_helpers() -> None:
    body = anthropic_body()
    body["messages"][0]["content"].append(
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": "A" * 1000},
        }
    )
    stripped = strip_media(body)
    assert "redacted" in stripped["messages"][0]["content"][1]["source"]["data"]
    assert stripped["system"][0]["text"] == body["system"][0]["text"]
    assert body["messages"][0]["content"][1]["source"]["data"] == "A" * 1000  # original untouched

    hashed = hash_text(body)
    assert len(hashed["system"][0]["text"]) == len(body["system"][0]["text"])
    assert hashed["system"][0]["text"] != body["system"][0]["text"]
    short = hashed["messages"][0]["content"][0]["text"]  # "hello": same length, still hidden
    assert len(short) == 5 and short != "hello"
    assert hashed["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_jsonl_sink_redacts(tmp_path: Any) -> None:
    path = tmp_path / "t.jsonl"
    r = Recorder(JsonlSink(path, redact=hash_text))
    r.record("anthropic", anthropic_body())
    stored = load_jsonl(path)[0].body["system"][0]["text"]
    assert stored != LONG and len(stored) == len(LONG)


def test_hash_text_keeps_lengths_so_estimates_and_cl011_stay_right() -> None:
    a, b = anthropic_body(system=LONG + "v1"), anthropic_body(system=LONG + "v2")
    plain = analyze([rec(a, 1.0), rec(b, 2.0)])
    hashed = analyze([rec(hash_text(a), 1.0), rec(hash_text(b), 2.0)])
    assert hashed.totals.lost_tokens_estimate == plain.totals.lost_tokens_estimate
    assert {f.code for f in hashed.findings} == {f.code for f in plain.findings}
    assert len(hash_text(a)["system"][0]["text"]) == len(a["system"][0]["text"])
    assert len(hash_text({"text": "hi"})["text"]) == 2


def test_hash_text_short_texts_do_not_collide() -> None:
    out = {t: hash_text({"text": t})["text"] for t in ["hi", "ok", "yes", "no!", "hello", "thank"]}
    assert all(len(v) == len(k) for k, v in out.items())
    assert len(set(out.values())) == len(out)
    assert not any(v.startswith("sha") for v in out.values())
    assert hash_text({"text": ""})["text"] == ""
    long = hash_text({"text": "x" * 200})["text"]
    assert len(long) == 200 and len(set(long)) > 1


def test_hash_text_keeps_sessions_apart_and_breaks_visible() -> None:
    a = anthropic_body(user="hi", mark_last=True)
    b = anthropic_body(user="ok", mark_last=True)
    plain = analyze([rec(a, 1.0), rec(b, 2.0)])
    hashed = analyze([rec(hash_text(a), 1.0), rec(hash_text(b), 2.0)])
    assert len(plain.sessions) == len(hashed.sessions) == 2

    h1 = anthropic_body(
        history=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        user="z",
        mark_last=True,
    )
    h2 = anthropic_body(
        history=[{"role": "user", "content": [{"type": "text", "text": "ok"}]}],
        user="z",
        mark_last=True,
    )
    plain = analyze([rec(h1, 1.0, session_id="s"), rec(h2, 2.0, session_id="s")])
    hashed = analyze(
        [rec(hash_text(h1), 1.0, session_id="s"), rec(hash_text(h2), 2.0, session_id="s")]
    )
    assert plain.totals.breaks == hashed.totals.breaks == 1
