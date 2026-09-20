# cachelint

Explain why your LLM prompt cache missed.

Prompt caching on Anthropic and OpenAI is a prefix match: one changed byte
anywhere before the cache marker and everything after it is billed again at
full price. The providers tell you *that* it happened (`cache_read_input_tokens: 0`)
but not *where*. `cachelint` records your requests, diffs each one against the
previous request of the same conversation, and names the block, the offset and
the characters that broke the prefix, plus the usual suspects it can see without
history: timestamps in the system prompt, a missing marker, a prefix below
the model's minimum.

```
cachelint: 3 requests in 1 session(s); prompt tokens 4880 (read 0, write 4806, uncached 74);
hit ratio 0.0%; 2 prefix break(s), ~3096 tokens lost (estimate)
  (sessions grouped automatically by first message; use cachelint.session(id) for exact grouping)

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

The SDKs accept an `http_client`. Keep their own client class (it carries the
SDK's connection limits and timeouts) and wrap only the transport:

```python
import anthropic, httpx2
import cachelint
from cachelint.transport import wrap_transport

recorder = cachelint.Recorder(
    cachelint.LogWatcher(),               # warn in the log the moment a prefix breaks
    cachelint.JsonlSink("trace.jsonl"),   # keep a trace for the offline report
)
claude = anthropic.Anthropic(
    http_client=anthropic.DefaultHttpxClient(
        transport=wrap_transport(httpx2.HTTPTransport(), recorder)
    )
)
# openai: openai.OpenAI(http_client=openai.DefaultHttpxClient(transport=wrap_transport(...)))
# async:  anthropic.DefaultAsyncHttpxClient(transport=wrap_transport(httpx2.AsyncHTTPTransport(), recorder))

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
| `CL001` | Content inside the previously cached prefix changed. Block, offset, before/after excerpt, estimated tokens lost. **Confirmed** by the diff. |
| `CL002` | Tool definitions changed between requests (tools render first: everything is invalidated). |
| `CL003` | A top-level parameter changed: `model`, `thinking`, `output_config.effort`, `tool_choice`, `speed`, `prompt_cache_key`. |
| `CL004` | The last breakpoint moved earlier than in the previous request. |
| `CL005` | Prefix (or a marked tools+system head shared across requests) was identical and within TTL, but the provider read nothing. Suspects outside the payload. Informational on OpenAI, whose cache is best-effort. |
| `CL006` | The gap since the previous request exceeds the cache TTL (Anthropic: the longest marker's, 5 min or 1 h; OpenAI about 10 min). |
| `CL007` | More than 20 positions appended in one turn; Anthropic's breakpoint lookback window is 20 (a run of parallel `tool_use` blocks is one position, as is a run of `tool_result` blocks). |
| `CL010` | No `cache_control` anywhere on a prompt large enough to cache. |
| `CL011` | Cacheable prefix below the model's minimum (512 to 4096 tokens depending on model). Estimate. |
| `CL012` | Timestamp / UUID / random-looking id / "today is" inside the cacheable prefix. A **suspicion** from patterns; `CL001` is the confirmation. |
| `CL013` | More than 4 `cache_control` slots (explicit markers plus the top-level automatic one). |
| `CL014` | Requests with the same, unmarked tools+system write cache and never read it: the only marker sits after per-request content. |

## What is exact and what is estimated

The diff runs on a canonical rendering of the request: each tool, system block
and message block becomes one segment; text blocks keep their text as is,
structured blocks are serialized as JSON with sorted keys, and cache markers are
stripped (the marker moving forward is not an invalidator). Offsets are
character offsets into that block's own text, so they point at the character
you wrote. Key order inside JSON blocks is *not* compared, because the provider
parses the JSON before hashing.

Costs come from the provider's own `usage` fields and are exact. Only "tokens
that could have been read" is an estimate (about 4 characters per token) and is
labelled as such everywhere.

## Sessions

Consecutive requests are diffed within a *session*. Two modes:

* **Explicit**: `with cachelint.session("conv-42"):` around the calls, or
  `session_id=` on `recorder.record`. Reliable; use it in production.
* **Automatic** (default): a request continues the session whose **first message
  block** it repeats byte for byte, when its message history is the previous
  history plus new turns (or an exact retry). Growth of the history is the
  signature of a continuation, so a tool reorder or a rewritten system prompt
  mid-conversation stays in the session and is reported. Interleaved
  conversations that share a system prompt stay apart, and so do two users who
  both open with "hi" from their second turn on. A **sliding history window**
  (oldest turns dropped) is recognised and reported as a prefix rewrite. What
  automatic mode cannot follow is a history *edited* in the middle; use explicit
  sessions for that. The report says when grouping was automatic.

## Privacy and cost

A trace contains full prompts. Before it leaves the machine, pass a redactor to
the sink:

```python
from cachelint.redact import strip_media, hash_text
cachelint.JsonlSink("trace.jsonl", redact=strip_media)  # drop base64 images/PDFs, keep text
cachelint.JsonlSink("trace.jsonl", redact=hash_text)    # keep structure and sizes only
```

With `hash_text` the report still says *which block* changed and by how much,
but not the bytes.

Recording runs on the caller's thread right after the response body is consumed:
one pass over the request body plus an O(1) session lookup. `Recorder` is
thread-safe. On a latency-critical async path with very large prompts, record
from a worker instead of the transport.

## In tests

```python
from cachelint.testing import assert_cache_stable

def test_agent_keeps_its_cache(recorder):
    run_agent(http_client=...)
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
