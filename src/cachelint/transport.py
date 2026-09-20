"""httpx2 transport that records provider requests as they go by.

The Anthropic and OpenAI SDKs both accept an ``http_client`` built on
``httpx2``; wrap its transport and every Messages / Chat Completions /
Responses call is recorded with its usage, streaming or not, without touching
SDK internals::

    import anthropic, httpx2
    import cachelint
    from cachelint.transport import wrap_transport

    recorder = cachelint.Recorder(cachelint.LogWatcher())
    claude = anthropic.Anthropic(
        http_client=anthropic.DefaultHttpxClient(
            transport=wrap_transport(httpx2.HTTPTransport(), recorder)
        )
    )

Nothing here ever raises into the caller: a body that cannot be parsed is
simply not recorded (at DEBUG level in the ``cachelint.transport`` logger).
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

import httpx2 as hx

from cachelint.providers import detect_provider
from cachelint.providers.base import Provider
from cachelint.recorder import Recorder

log = logging.getLogger("cachelint.transport")


# ----------------------------------------------------------------- parsing


def parse_sse(data: bytes) -> list[dict[str, Any]]:
    """JSON payloads of the ``data:`` lines of an SSE body (non-JSON lines skipped)."""
    events: list[dict[str, Any]] = []
    text = data.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    for raw_block in text.split("\n\n"):
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


def _decode(status: int, headers: Any, data: bytes) -> bytes:
    """Apply the response's content-encoding (gzip, br, ...) to the raw wire bytes."""
    if not headers.get("content-encoding"):
        return data
    return bytes(hx.Response(status, headers=headers, content=data).read())


def _record(
    recorder: Recorder,
    provider: Provider,
    body: dict[str, Any],
    url: str,
    status: int,
    headers: Any,
    data: bytes,
) -> None:
    if status < 200 or status >= 300:
        return  # errors never touch the cache
    try:
        data = _decode(status, headers, data)
        content_type = headers.get("content-type", "")
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


class _TeeStream(hx.SyncByteStream):
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


class _AsyncTeeStream(hx.AsyncByteStream):
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


class RecordingTransport(hx.BaseTransport):
    """Sync transport wrapper. Requests to unknown URLs pass through untouched."""

    def __init__(self, inner: hx.BaseTransport, recorder: Recorder) -> None:
        self._inner = inner
        self.recorder = recorder

    def handle_request(self, request: hx.Request) -> hx.Response:
        url = str(request.url)
        provider = detect_provider(url)
        if provider is None:
            return self._inner.handle_request(request)

        request.read()
        body = _request_body(request.content)
        response = self._inner.handle_request(request)
        if body is None:
            return response

        headers = response.headers
        status = response.status_code

        def done(data: bytes) -> None:
            _record(self.recorder, provider, body, url, status, headers, data)

        return hx.Response(
            status_code=status,
            headers=headers,
            stream=_TeeStream(response.stream, done),
            request=request,
            extensions=response.extensions,
        )

    def close(self) -> None:
        self._inner.close()


class AsyncRecordingTransport(hx.AsyncBaseTransport):
    """Async transport wrapper. Requests to unknown URLs pass through untouched."""

    def __init__(self, inner: hx.AsyncBaseTransport, recorder: Recorder) -> None:
        self._inner = inner
        self.recorder = recorder

    async def handle_async_request(self, request: hx.Request) -> hx.Response:
        url = str(request.url)
        provider = detect_provider(url)
        if provider is None:
            return await self._inner.handle_async_request(request)

        await request.aread()
        body = _request_body(request.content)
        response = await self._inner.handle_async_request(request)
        if body is None:
            return response

        headers = response.headers
        status = response.status_code

        def done(data: bytes) -> None:
            _record(self.recorder, provider, body, url, status, headers, data)

        return hx.Response(
            status_code=status,
            headers=headers,
            stream=_AsyncTeeStream(response.stream, done),
            request=request,
            extensions=response.extensions,
        )

    async def aclose(self) -> None:
        await self._inner.aclose()


# ----------------------------------------------------------------- helpers


def wrap_transport(
    inner: hx.BaseTransport | hx.AsyncBaseTransport,
    recorder: Recorder,
    *,
    use_async: bool | None = None,
) -> RecordingTransport | AsyncRecordingTransport:
    """Wrap an existing transport.

    Sync when it implements ``handle_request``, async when it only implements
    ``handle_async_request``; ``use_async`` forces the choice (``MockTransport``
    implements both).
    """
    if use_async is None:
        use_async = not isinstance(inner, hx.BaseTransport)
    if use_async:
        return AsyncRecordingTransport(inner, recorder)  # type: ignore[arg-type]
    return RecordingTransport(inner, recorder)  # type: ignore[arg-type]


def client(
    recorder: Recorder, *, transport: hx.BaseTransport | None = None, **client_kwargs: Any
) -> hx.Client:
    """A plain ``httpx2.Client`` that records provider traffic.

    For SDKs prefer their own client class with :func:`wrap_transport` (see the
    module docstring) so the SDK's connection limits and timeouts are kept.
    """
    inner = transport if transport is not None else hx.HTTPTransport()
    return hx.Client(transport=RecordingTransport(inner, recorder), **client_kwargs)


def async_client(
    recorder: Recorder, *, transport: hx.AsyncBaseTransport | None = None, **client_kwargs: Any
) -> hx.AsyncClient:
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
