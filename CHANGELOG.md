# Changelog

## 0.1.0

Initial release.

- Prefix models for Anthropic Messages (explicit and top-level `cache_control`)
  and OpenAI Chat Completions / Responses (automatic prefix caching).
- Segment-level prefix diff with block path, character offset into the block's
  own text, before/after excerpts and an estimate of tokens lost.
- Findings CL001–CL007 (session) and CL010–CL014 (single request).
- Sessions: explicit `session()` context, or automatic grouping anchored on the
  first message block; a continuation is a grown or repeated history; sliding
  history windows and edited histories (truncated tool results) are
  recognised and reported with their own hints.
- `Recorder` (thread-safe, bounded by `keep`) with JSONL sink and redactors
  (`strip_media`, length-preserving `hash_text`), `LogWatcher` live mode
  sharing the offline analyzer without history (memory bounded by active
  sessions), `assert_cache_stable` for tests, `read_trace` that skips and
  counts unreadable lines.
- `cachelint.transport`: recording httpx2 transport (sync and async), SSE aware,
  decodes compressed responses.
- CLI: `report`, `lint`, `codes`.
