"""httpx / httpx2 transport that records provider requests as they go by.

The Anthropic and OpenAI SDKs both accept an ``http_client``; hand them one
built here and every Messages / Chat Completions / Responses call is recorded
with its usage, streaming or not, without touching SDK internals::

    import cachelint
    from cachelint.transport import client

    recorder = cachelint.Recorder(cachelint.LogWatcher())
    anthropic_client = anthropic.Anthropic(http_client=client(recorder))

``httpx2`` (what current SDKs depend on) is preferred; ``httpx`` is used when
that is what is installed. Nothing here ever raises into the caller: a body
that cannot be parsed is simply not recorded.
"""

from __future__ import annotations

import json
import logging
import types
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

from cachelint.providers import detect_provider
from cachelint.providers.base import Provider
from cachelint.recorder import Recorder

log = logging.getLogger("cachelint.transport")


def _load_httpx() -> types.ModuleType:
    try:
        import httpx2

        return httpx2
    except ImportError:  # pragma: no cover - depends on the environment
        try:
            import httpx

            return httpx
        except ImportError as exc:
            raise ImportError(
                "cachelint.transport needs httpx2 or httpx: pip install 'cachelint[httpx2]'"
            ) from exc


hx = _load_httpx()


# ----------------------------------------------------------------- parsing


def parse_sse(data: bytes) -> list[dict[str, Any]]:
    """JSON payloads of the ``data:`` lines of an SSE body (non-JSON lines skipped)."""
    events: list[dict[str, Any]] = []
    for raw_block in data.decode("utf-8", errors="replace").split("\n\n"):
        payload = "\n".join(
            line[5:].lstrip() for line in raw_block.splitlines() if line.startswith("data:")
        )
        if not payload:
            continue
        try:
            obj = json.loads(payload)
        except ValueError:
            continue
        if isinstance(obj, dict):
            events.append(obj)
    return events


def _request_body(content: bytes) -> dict[str, Any] | None:
    if not content:
        return None
    try:
        body = json.loads(content)
    except ValueError:
        return None
    return body if isinstance(body, dict) else None


def _record(
    recorder: Recorder,
    provider: Provider,
    body: dict[str, Any],
    url: str,
    status: int,
    content_type: str,
    data: bytes,
) -> None:
    if status < 200 or status >= 300:
        return  # errors never touch the cache
    try:
        if "text/event-stream" in content_type:
            recorder.record(provider.name, body, sse_events=parse_sse(data), stream=True, url=url)
        else:
            response = json.loads(data) if data else {}
            recorder.record(
                provider.name,
                body,
                response=response if isinstance(response, dict) else None,
                stream=bool(body.get("stream")),
                url=url,
            )
    except Exception:  # never break the caller's request
        log.debug("cachelint: could not record %s", url, exc_info=True)


# ----------------------------------------------------------------- streams


class _TeeStream(hx.SyncByteStream):  # type: ignore[name-defined]
    def __init__(self, inner: Any, on_done: Callable[[bytes], None]) -> None:
        self._inner = inner
        self._on_done = on_done
        self._chunks: list[bytes] = []
        self._done = False

    def __iter__(self) -> Iterator[bytes]:
        for chunk in self._inner:
            self._chunks.append(chunk)
            yield chunk
        self._finish()

    def close(self) -> None:
        try:
            self._inner.close()
        finally:
            self._finish()

    def _finish(self) -> None:
        if self._done:
            return
        self._done = True
        self._on_done(b"".join(self._chunks))


class _AsyncTeeStream(hx.AsyncByteStream):  # type: ignore[name-defined]
    def __init__(self, inner: Any, on_done: Callable[[bytes], None]) -> None:
        self._inner = inner
        self._on_done = on_done
        self._chunks: list[bytes] = []
        self._done = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for chunk in self._inner:
            self._chunks.append(chunk)
            yield chunk
        self._finish()

    async def aclose(self) -> None:
        try:
            await self._inner.aclose()
        finally:
            self._finish()

    def _finish(self) -> None:
        if self._done:
            return
        self._done = True
        self._on_done(b"".join(self._chunks))


# -------------------------------------------------------------- transports


class RecordingTransport(hx.BaseTransport):  # type: ignore[name-defined]
    """Sync transport wrapper. Requests to unknown URLs pass through untouched."""

    def __init__(self, inner: Any, recorder: Recorder) -> None:
        self._inner = inner
        self.recorder = recorder

    def handle_request(self, request: Any) -> Any:
        url = str(request.url)
        provider = detect_provider(url)
        if provider is None:
            return self._inner.handle_request(request)

        request.read()
        body = _request_body(request.content)
        response = self._inner.handle_request(request)
        if body is None:
            return response

        content_type = response.headers.get("content-type", "")
        status = response.status_code

        def done(data: bytes) -> None:
            _record(self.recorder, provider, body, url, status, content_type, data)

        return hx.Response(
            status_code=status,
            headers=response.headers,
            stream=_TeeStream(response.stream, done),
            request=request,
            extensions=response.extensions,
        )

    def close(self) -> None:
        self._inner.close()


class AsyncRecordingTransport(hx.AsyncBaseTransport):  # type: ignore[name-defined]
    """Async transport wrapper. Requests to unknown URLs pass through untouched."""

    def __init__(self, inner: Any, recorder: Recorder) -> None:
        self._inner = inner
        self.recorder = recorder

    async def handle_async_request(self, request: Any) -> Any:
        url = str(request.url)
        provider = detect_provider(url)
        if provider is None:
            return await self._inner.handle_async_request(request)

        await request.aread()
        body = _request_body(request.content)
        response = await self._inner.handle_async_request(request)
        if body is None:
            return response

        content_type = response.headers.get("content-type", "")
        status = response.status_code

        def done(data: bytes) -> None:
            _record(self.recorder, provider, body, url, status, content_type, data)

        return hx.Response(
            status_code=status,
            headers=response.headers,
            stream=_AsyncTeeStream(response.stream, done),
            request=request,
            extensions=response.extensions,
        )

    async def aclose(self) -> None:
        await self._inner.aclose()


# ----------------------------------------------------------------- helpers


def wrap_transport(inner: Any, recorder: Recorder, *, use_async: bool | None = None) -> Any:
    """Wrap an existing transport.

    Sync when it implements ``handle_request``, async when it only implements
    ``handle_async_request``; ``use_async`` forces the choice (``MockTransport``
    implements both).
    """
    if use_async is None:
        use_async = not isinstance(inner, hx.BaseTransport)
    if use_async:
        return AsyncRecordingTransport(inner, recorder)
    return RecordingTransport(inner, recorder)


def client(recorder: Recorder, *, transport: Any = None, **client_kwargs: Any) -> Any:
    """An ``httpx2.Client`` (or ``httpx.Client``) that records provider traffic.

    Pass it to an SDK as ``http_client=``. ``transport`` defaults to the
    library's ``HTTPTransport``; extra keyword arguments go to the client.
    """
    inner = transport if transport is not None else hx.HTTPTransport()
    return hx.Client(transport=RecordingTransport(inner, recorder), **client_kwargs)


def async_client(recorder: Recorder, *, transport: Any = None, **client_kwargs: Any) -> Any:
    """Async counterpart of :func:`client`."""
    inner = transport if transport is not None else hx.AsyncHTTPTransport()
    return hx.AsyncClient(transport=AsyncRecordingTransport(inner, recorder), **client_kwargs)


__all__ = [
    "AsyncRecordingTransport",
    "RecordingTransport",
    "async_client",
    "client",
    "parse_sse",
    "wrap_transport",
]
