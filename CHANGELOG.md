# Changelog

## 0.1.0

Initial release.

- Prefix models for Anthropic Messages (explicit and top-level `cache_control`)
  and OpenAI Chat Completions / Responses (automatic prefix caching).
- Segment-level prefix diff with block path, character offset and excerpts.
- Findings CL001–CL007 (session) and CL010–CL014 (single request).
- Automatic session grouping with near-match tolerance; explicit `session()` context.
- `Recorder` with JSONL sink, `LogWatcher` live mode, `assert_cache_stable` for tests.
- `cachelint.transport`: recording httpx2/httpx transport (sync and async, SSE aware).
- CLI: `report`, `lint`, `codes`.
