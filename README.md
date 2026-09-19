# cachelint

Explain why your LLM prompt cache missed.

Prompt caching on Anthropic and OpenAI is a prefix match: one changed byte
anywhere before the cache marker and everything after it is billed again at
full price. The providers tell you *that* it happened (`cache_read_input_tokens: 0`)
but not *where*. `cachelint` records your requests, diffs each one against the
previous request of the same conversation, and names the block, the offset and
the bytes that broke the prefix, plus the usual suspects it can see without
history: timestamps in the system prompt, a missing marker, a prefix below
the model's minimum.

```
cachelint: 3 requests in 1 session(s); prompt tokens 4880 (read 0, write 4806, uncached 74);
hit ratio 0.0%; 2 prefix break(s), ~3096 tokens lost (estimate)

session 87a72ac21002 [anthropic] 3 req, hit ratio 0.0%, 2 break(s)
  #0  claude-opus-5  segments=2 cacheable=1  in=24 read=0 write=1601
      ! CL012 iso-datetime inside the cacheable prefix at system[0]
        → Timestamps change every request. Move them after the last breakpoint or drop them.
  #1  claude-opus-5  segments=4 cacheable=1  in=25 read=0 write=1602
      ✗ prefix broken at system[0] +6190 (~1548 tokens lost, estimate)
        was: '…ime: 2026-09-20T10:15'
        now: '…ime: 2026-09-20T10:16'
  #2  claude-opus-5  segments=6 cacheable=1  in=25 read=0 write=1603
      ✗ prefix broken at system[0] +6190 (~1548 tokens lost, estimate)
        was: '…ime: 2026-09-20T10:16'
        now: '…ime: 2026-09-20T10:17'
```

Zero dependencies. Python 3.11+. Anthropic Messages API, OpenAI Chat Completions
and Responses.

## Install

```bash
pip install cachelint            # core: explicit recording, analysis, CLI
pip install "cachelint[httpx2]"  # + transport that records SDK traffic automatically
```

## Quick start

The SDKs accept an `http_client`. Give them one that records:

```python
import anthropic
import cachelint
from cachelint.transport import client

recorder = cachelint.Recorder(
    cachelint.LogWatcher(),               # warn in the log the moment a prefix breaks
    cachelint.JsonlSink("trace.jsonl"),   # keep a trace for the offline report
)
claude = anthropic.Anthropic(http_client=client(recorder))

# ... run your agent as usual ...

print(cachelint.analyze(recorder.records).to_text())
```

Then, or on a trace from production:

```bash
cachelint report trace.jsonl              # human-readable
cachelint report trace.jsonl --json       # machine-readable
cachelint report trace.jsonl --fail-on CL001,CL002   # non-zero exit for CI
cachelint lint request.json --provider anthropic     # static checks on one body
cachelint codes
```

No SDK, or a gateway of your own? Record explicitly:

```python
recorder.record("anthropic", body, response=response_json)          # non-streaming
recorder.record("anthropic", body, sse_events=events, stream=True)  # streaming
```

## What it reports

| Code | Meaning |
|---|---|
| `CL001` | Content inside the previously cached prefix changed. Block, offset, before/after excerpt, estimated tokens lost. |
| `CL002` | Tool definitions changed between requests (tools render first: everything is invalidated). |
| `CL003` | A top-level parameter changed: `model`, `thinking`, `output_config.effort`, `prompt_cache_key`. |
| `CL004` | The last breakpoint moved earlier than in the previous request. |
| `CL005` | Prefix was byte-identical and within TTL, but the provider read nothing. Suspects outside the payload. |
| `CL006` | The gap since the previous request exceeds the cache TTL (5 min, or 1 h with `ttl: "1h"`). |
| `CL007` | More than 20 blocks appended in one turn; Anthropic's breakpoint lookback window is 20 positions. |
| `CL010` | No `cache_control` anywhere on a prompt large enough to cache. |
| `CL011` | Cacheable prefix below the model's minimum (512 to 4096 tokens depending on model). Estimate. |
| `CL012` | Timestamp / UUID / random id / "today is" inside the cacheable prefix. |
| `CL013` | More than 4 `cache_control` markers. |
| `CL014` | Consecutive requests write cache and never read it: the marker sits after per-request content. |

Bytes are exact; tokens are estimates. The place a prefix broke is found by
comparing the rendered request bytes, so it is precise. What that costs is
reported from the provider's own `usage` fields; only "tokens that could have
been read" is estimated (~4 characters per token) and is labelled as such.

## How it works

1. **Prefix model per provider.** Anthropic renders `tools → system → messages`
   and reads only at `cache_control` markers (explicit, or the top-level automatic
   one). OpenAI caches the longest previously seen prefix automatically. Each
   request is turned into an ordered list of segments with markers stripped from
   the compared text (the moving marker is not an invalidator).
2. **Sessions.** Requests are grouped by explicit id (`with cachelint.session("id"):`
   or `Record.session_id`) or, by default, automatically: a request joins the
   recent session whose segments it mostly repeats. A segment that changed by a
   few bytes still counts as the same segment; keeping such requests together is
   the whole point.
3. **Diff.** The new request is compared segment by segment with the previous one.
   A divergence *after* the previous cacheable prefix is normal conversation
   growth; a divergence *inside* it is a break, reported at the first differing
   character.
4. **Accounting.** `usage` is normalized to `input_tokens` (uncached remainder),
   `cache_read`, `cache_write`; hit ratio is `cache_read / (all three)`.

## In tests

```python
from cachelint.testing import assert_cache_stable

def test_agent_keeps_its_cache(recorder):
    run_agent(http_client=client(recorder))
    assert_cache_stable(recorder.records)   # fails with the rendered report on CL001/002/010/013
```

## Custom gateways

Register an adapter for a proxy that speaks a known dialect on another host:

```python
from cachelint.providers import AnthropicProvider, register_provider

class Gateway(AnthropicProvider):
    name = "gateway"
    def matches(self, url: str) -> bool:
        return "llm.internal" in url

register_provider(Gateway())
```

## Not in scope (yet)

Rewriting requests or placing markers for you, Gemini, semantic caches, a proxy
server. The diagnosis has to be right before the fix can be automatic.

## License

MIT
