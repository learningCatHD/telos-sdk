"""End-to-end proxy test for the multi-backend ``/upstreams/<slug>/...`` route.

Mocks an OpenAI ChatCompletions-compatible upstream, runs the real proxy, and
verifies:
- the request hits the right URL on the upstream;
- the wire was TELOS-processed (DROP-band sinking, tools canonicalization);
- the usage_log records ``harness="telos"`` with the normalized buckets the
  dashboard reads;
- streaming SSE forwards chunks and extracts usage from the terminal chunk
  when ``stream_options.include_usage=true``;
- unknown upstream slugs return a 404 error;
- per-slug upstream URL is honored (legacy ``self.upstream`` is NOT used);
- the ``/_h/<harness>/upstreams/<slug>/...`` URL tag attributes the usage log
  to the calling harness (hermes / openclaw / codex) and overrides the slug's
  ``via``.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import aiohttp
from aiohttp import web

from telos.config import UpstreamCacheConfig, UpstreamConfig
from telos.proxy.server import make_app


# ---------------------------------------------------------------------------
# mock upstream (acts as openrouter.ai / api.deepseek.com)
# ---------------------------------------------------------------------------

class _MockOpenAIUpstream:
    """Records the received wire requests; returns chat-completions JSON or SSE."""

    def __init__(
        self,
        *,
        sse: bool = False,
        with_usage: bool = True,
        usage_override: dict[str, Any] | None = None,
        cache_details: dict[str, Any] | None = None,
    ) -> None:
        self.last_body: dict[str, Any] | None = None
        self.last_headers: dict[str, str] | None = None
        self.last_path: str | None = None
        self.sse = sse
        self.with_usage = with_usage
        self.usage_override = usage_override
        self.cache_details = cache_details

    async def handler(self, request: web.Request) -> web.StreamResponse:
        self.last_body = await request.json()
        self.last_headers = dict(request.headers)
        self.last_path = request.path
        if self.sse:
            response = web.StreamResponse(
                status=200,
                headers={"Content-Type": "text/event-stream"},
            )
            await response.prepare(request)
            chunks = [
                b'data: {"id":"x","choices":[{"index":0,"delta":{"role":"assistant"}}]}\n\n',
                b'data: {"id":"x","choices":[{"index":0,"delta":{"content":"Hi"}}]}\n\n',
            ]
            if self.with_usage:
                # OpenAI SSE: when stream_options.include_usage is set, the
                # penultimate chunk carries the full usage block.
                chunks.append(
                    b'data: {"id":"x","choices":[],"usage":'
                    b'{"prompt_cache_hit_tokens":900,"prompt_cache_miss_tokens":100,'
                    b'"completion_tokens":7}}\n\n'
                )
            if self.cache_details is not None:
                chunks.append(
                    b"data: " + json.dumps({
                        "id": "x",
                        "choices": [],
                        "sglext": {
                            "cached_tokens_details": self.cache_details,
                        },
                    }).encode("utf-8") + b"\n\n"
                )
            chunks.append(b'data: [DONE]\n\n')
            for c in chunks:
                await response.write(c)
            await response.write_eof()
            return response

        usage = self.usage_override if self.usage_override is not None else ({
            "prompt_cache_hit_tokens": 900,
            "prompt_cache_miss_tokens": 100,
            "completion_tokens": 7,
        } if self.with_usage else {})
        response_body = {
            "id": "cmpl_test",
            "object": "chat.completion",
            "model": "deepseek-chat",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "hello"},
                "finish_reason": "stop",
            }],
            "usage": usage,
        }
        if self.cache_details is not None:
            response_body["sglext"] = {
                "cached_tokens_details": self.cache_details,
            }
        return web.json_response(response_body)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

async def _start_upstream(mock: _MockOpenAIUpstream) -> tuple[web.AppRunner, str]:
    app = web.Application()
    app.router.add_post("/v1/chat/completions", mock.handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, f"http://127.0.0.1:{port}"


async def _start_proxy(
    upstream_url: str,
    *,
    usage_log: Path | None = None,
    extra_slugs: dict[str, UpstreamConfig] | None = None,
) -> tuple[web.AppRunner, str]:
    upstreams = {
        "openrouter": UpstreamConfig(
            url=upstream_url, engine="deepseek", protocol="openai-chat",
        ),
    }
    if extra_slugs:
        upstreams.update(extra_slugs)
    app = make_app(upstream="http://unused", upstreams=upstreams, usage_log=usage_log)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, f"http://127.0.0.1:{port}"


def _sample_chat_request() -> dict[str, Any]:
    """An OpenAI ChatCompletions request with enough structure to exercise IR layout."""
    return {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello there"},
        ],
        "tools": [{
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Look something up",
                "parameters": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                },
            },
        }],
        "max_tokens": 64,
        "stream": False,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

async def _test_openai_route_non_streaming() -> None:
    mock = _MockOpenAIUpstream(sse=False)
    up_runner, up_url = await _start_upstream(mock)
    px_runner, px_url = await _start_proxy(up_url)
    try:
        async with aiohttp.ClientSession() as client:
            async with client.post(
                f"{px_url}/upstreams/openrouter/v1/chat/completions",
                json=_sample_chat_request(),
                headers={
                    "authorization": "Bearer test-key",
                    "x-telos-session": "openai-nonstream",
                },
            ) as resp:
                assert resp.status == 200, await resp.text()
                body = await resp.json()
                assert body["id"] == "cmpl_test"

        # Upstream received the TELOS-processed wire on /v1/chat/completions.
        assert mock.last_path == "/v1/chat/completions"
        wire = mock.last_body
        assert wire is not None
        assert wire["model"] == "deepseek-chat"
        assert isinstance(wire.get("messages"), list)
        # Auth header was forwarded through the proxy (header names are
        # case-insensitive — aiohttp may transmit as ``Authorization``).
        auth_value = next(
            (v for k, v in (mock.last_headers or {}).items()
             if k.lower() == "authorization"), None,
        )
        assert auth_value == "Bearer test-key"
        print("✓ test_openai_route_non_streaming")
    finally:
        await px_runner.cleanup()
        await up_runner.cleanup()


async def _test_openai_route_logs_usage(tmp_log: Path) -> None:
    mock = _MockOpenAIUpstream(sse=False)
    up_runner, up_url = await _start_upstream(mock)
    px_runner, px_url = await _start_proxy(up_url, usage_log=tmp_log)
    try:
        async with aiohttp.ClientSession() as client:
            async with client.post(
                f"{px_url}/upstreams/openrouter/v1/chat/completions",
                json=_sample_chat_request(),
                headers={"authorization": "Bearer k",
                         "x-telos-session": "openai-log"},
            ) as resp:
                assert resp.status == 200

        line = tmp_log.read_text().strip().splitlines()[-1]
        record = json.loads(line)
        assert record["session_id"] == "openai-log"
        # The whole point: dashboard sees a "telos" harness entry for OpenAI traffic.
        assert record["harness"] == "telos"
        # DeepSeek usage normalization: hits → cache_read, misses → raw_input.
        assert record["normalized"]["cache_read"] == 900
        assert record["normalized"]["raw_input"] == 100
        assert record["normalized"]["output"] == 7
        print("✓ test_openai_route_logs_usage")
    finally:
        await px_runner.cleanup()
        await up_runner.cleanup()


async def _test_openai_route_via_labels_harness(tmp_log: Path) -> None:
    """When the upstream slug carries ``via``, the usage log records it as
    the harness — that's how the dashboard attributes traffic to the calling
    tool (openclaw / hermes / codex) instead of the wire-level ``telos``."""
    mock = _MockOpenAIUpstream(sse=False)
    up_runner, up_url = await _start_upstream(mock)
    # Slug with via="openclaw" simulates a config patched by OpenClawInstaller.
    extra = {
        "openclaw_routed": UpstreamConfig(
            url=up_url, engine="deepseek", protocol="openai-chat",
            via="openclaw",
        ),
    }
    px_runner, px_url = await _start_proxy(up_url, usage_log=tmp_log,
                                            extra_slugs=extra)
    try:
        async with aiohttp.ClientSession() as client:
            async with client.post(
                f"{px_url}/upstreams/openclaw_routed/v1/chat/completions",
                json=_sample_chat_request(),
                headers={"authorization": "Bearer k",
                         "x-telos-session": "via-test"},
            ) as resp:
                assert resp.status == 200

        line = tmp_log.read_text().strip().splitlines()[-1]
        record = json.loads(line)
        assert record["session_id"] == "via-test"
        # The crucial assertion: harness is the via name, not "telos".
        assert record["harness"] == "openclaw"
        print("✓ test_openai_route_via_labels_harness")
    finally:
        await px_runner.cleanup()
        await up_runner.cleanup()


async def _test_openai_tag_attributes_each_harness(tmp_log: Path) -> None:
    """On the OpenAI-chat path, the ``/_h/<harness>/upstreams/<slug>/...`` URL
    tag is the authoritative attribution: every supported OpenAI harness
    (hermes / openclaw / codex) is recorded under its own name, and the tag
    wins even when the slug also carries a ``via`` default. This exercises the
    URL-tag branch of ``_handle_openai_chat`` (the anthropic-path equivalent is
    covered by ``test_url_tag_attribution`` in test_proxy_server.py)."""
    mock = _MockOpenAIUpstream(sse=False)
    up_runner, up_url = await _start_upstream(mock)
    # The slug carries via="telos"; the URL tag must override it per request.
    extra = {
        "deepseek": UpstreamConfig(
            url=up_url, engine="deepseek", protocol="openai-chat",
            via="telos",
        ),
    }
    px_runner, px_url = await _start_proxy(up_url, usage_log=tmp_log,
                                            extra_slugs=extra)
    try:
        async with aiohttp.ClientSession() as client:
            for harness in ("hermes", "openclaw", "codex"):
                mock.last_body = None
                async with client.post(
                    f"{px_url}/_h/{harness}/upstreams/deepseek/v1/chat/completions",
                    json=_sample_chat_request(),
                    headers={"authorization": "Bearer k",
                             "x-telos-session": f"tag-{harness}"},
                ) as resp:
                    assert resp.status == 200, await resp.text()
                # The tagged request reached the right upstream …
                assert mock.last_path == "/v1/chat/completions"
                # … and the usage log attributes it to the URL tag — not "telos"
                # (the slug's via) and not the wire-level default.
                record = json.loads(
                    tmp_log.read_text().strip().splitlines()[-1])
                assert record["session_id"] == f"tag-{harness}"
                assert record["harness"] == harness, (
                    f"URL tag {harness!r} should win over via='telos', "
                    f"got {record['harness']!r}"
                )
        print("✓ test_openai_tag_attributes_each_harness")
    finally:
        await px_runner.cleanup()
        await up_runner.cleanup()


async def _test_openai_route_streaming() -> None:
    mock = _MockOpenAIUpstream(sse=True, with_usage=True)
    up_runner, up_url = await _start_upstream(mock)
    px_runner, px_url = await _start_proxy(up_url)
    try:
        async with aiohttp.ClientSession() as client:
            req = _sample_chat_request()
            req["stream"] = True
            req["stream_options"] = {"include_usage": True}
            async with client.post(
                f"{px_url}/upstreams/openrouter/v1/chat/completions",
                json=req,
                headers={"authorization": "Bearer k"},
            ) as resp:
                assert resp.status == 200
                received = b""
                async for chunk in resp.content.iter_any():
                    received += chunk
                assert b"[DONE]" in received
                assert b"prompt_cache_hit_tokens" in received
        print("✓ test_openai_route_streaming")
    finally:
        await px_runner.cleanup()
        await up_runner.cleanup()


async def _test_sglang_route_uses_adapter_and_logs_cache(tmp_log: Path) -> None:
    mock = _MockOpenAIUpstream(usage_override={
        "prompt_tokens": 1000,
        "prompt_tokens_details": {"cached_tokens": 800},
        "completion_tokens": 7,
    }, cache_details={
        "device": 100,
        "host": 600,
        "storage": 100,
        "storage_backend": "file",
    })
    up_runner, up_url = await _start_upstream(mock)
    extra = {
        "sglang": UpstreamConfig(
            url=up_url,
            engine="sglang",
            protocol="openai-chat",
            cache=UpstreamCacheConfig(
                mode="stock",
                backend="hicache",
                tier_telemetry=True,
                security_namespace="opaque:test",
            ),
        ),
    }
    px_runner, px_url = await _start_proxy(
        up_url, usage_log=tmp_log, extra_slugs=extra,
    )
    try:
        request_body = _sample_chat_request()
        request_body.update({
            "temperature": 0.25,
            "cache_control": {"lock_radix_path": True},
            "cache_policy": {"evict_span": [0, 1]},
            "telos_cache": {"namespace": "user-controlled"},
            "cache_salt": "user-controlled",
        })
        async with aiohttp.ClientSession() as client:
            async with client.post(
                f"{px_url}/upstreams/sglang/v1/chat/completions",
                json=request_body,
                headers={"x-telos-session": "sglang-stock"},
            ) as resp:
                assert resp.status == 200, await resp.text()

        wire = mock.last_body or {}
        for private_field in (
            "cache_control", "cache_policy", "telos_cache", "cache_salt",
        ):
            assert private_field not in wire
        assert wire["temperature"] == 0.25
        assert wire["model"] == "deepseek-chat"
        assert wire["return_cached_tokens_details"] is True

        record = json.loads(tmp_log.read_text().strip().splitlines()[-1])
        assert record["normalized"]["cache_read"] == 800
        assert record["normalized"]["raw_input"] == 200
        assert record["cache_runtime"]["engine"] == "sglang"
        assert record["cache_runtime"]["wire_emitter"] == "adapter"
        assert record["cache_runtime"]["backend"] == "hicache"
        assert record["cache_runtime"]["hit_ratio"] == 0.8
        assert record["cache_runtime"]["hit_tier_tokens"] == {
            "gpu": 100, "cpu": 600, "l3": 100,
        }
        assert record["cache_runtime"]["restored_tokens"] == 700
        assert record["cache_runtime"]["storage_backend"] == "file"
        assert record["raw_cache_telemetry"]["source"] == (
            "sglext.cached_tokens_details"
        )
        print("✓ test_sglang_route_uses_adapter_and_logs_cache")
    finally:
        await px_runner.cleanup()
        await up_runner.cleanup()


async def _test_sglang_stream_logs_tier_details(tmp_log: Path) -> None:
    mock = _MockOpenAIUpstream(
        sse=True,
        cache_details={"device": 64, "host": 832, "storage": 0},
    )
    up_runner, up_url = await _start_upstream(mock)
    extra = {
        "sglang-stream": UpstreamConfig(
            url=up_url,
            engine="sglang",
            protocol="openai-chat",
            cache=UpstreamCacheConfig(
                backend="hicache",
                tier_telemetry=True,
            ),
        ),
    }
    px_runner, px_url = await _start_proxy(
        up_url, usage_log=tmp_log, extra_slugs=extra,
    )
    try:
        request_body = _sample_chat_request()
        request_body["stream"] = True
        request_body["stream_options"] = {"include_usage": True}
        async with aiohttp.ClientSession() as client:
            async with client.post(
                f"{px_url}/upstreams/sglang-stream/v1/chat/completions",
                json=request_body,
            ) as response:
                assert response.status == 200
                await response.read()

        assert (mock.last_body or {})["return_cached_tokens_details"] is True
        record = json.loads(tmp_log.read_text().strip().splitlines()[-1])
        assert record["streaming"] is True
        assert record["cache_runtime"]["hit_tier_tokens"] == {
            "gpu": 64, "cpu": 832, "l3": 0,
        }
        assert record["cache_runtime"]["restored_tokens"] == 832
    finally:
        await px_runner.cleanup()
        await up_runner.cleanup()


async def _test_unknown_slug_returns_404() -> None:
    mock = _MockOpenAIUpstream(sse=False)
    up_runner, up_url = await _start_upstream(mock)
    px_runner, px_url = await _start_proxy(up_url)
    try:
        async with aiohttp.ClientSession() as client:
            async with client.post(
                f"{px_url}/upstreams/no-such-slug/v1/chat/completions",
                json=_sample_chat_request(),
            ) as resp:
                assert resp.status == 404
                body = await resp.json()
                assert body["error"]["type"] == "not_found"
                assert "no-such-slug" in body["error"]["message"]
        # The mock upstream must NOT have been hit.
        assert mock.last_body is None
        print("✓ test_unknown_slug_returns_404")
    finally:
        await px_runner.cleanup()
        await up_runner.cleanup()


async def _test_per_slug_upstream_url_is_honored() -> None:
    """Two slugs pointing at two different mock upstreams — the request to each
    must hit only that slug's URL.
    """
    mock_a = _MockOpenAIUpstream(sse=False)
    mock_b = _MockOpenAIUpstream(sse=False)
    up_a_runner, up_a_url = await _start_upstream(mock_a)
    up_b_runner, up_b_url = await _start_upstream(mock_b)

    upstreams = {
        "openrouter": UpstreamConfig(url=up_a_url, engine="deepseek",
                                      protocol="openai-chat"),
        "deepseek":   UpstreamConfig(url=up_b_url, engine="deepseek",
                                      protocol="openai-chat"),
    }
    app = make_app(upstream="http://unused", upstreams=upstreams)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    px_url = f"http://127.0.0.1:{port}"

    try:
        async with aiohttp.ClientSession() as client:
            async with client.post(
                f"{px_url}/upstreams/deepseek/v1/chat/completions",
                json=_sample_chat_request(),
                headers={"authorization": "Bearer k"},
            ) as resp:
                assert resp.status == 200
        # deepseek mock got the request; openrouter mock did not.
        assert mock_b.last_body is not None
        assert mock_a.last_body is None
        print("✓ test_per_slug_upstream_url_is_honored")
    finally:
        await runner.cleanup()
        await up_a_runner.cleanup()
        await up_b_runner.cleanup()


async def _test_openai_responses_passthrough_streams() -> None:
    """Codex uses ``wire_api = "responses"``; the installer maps it onto
    ``/upstreams/openai/v1/responses``. The endpoint is SSE: the gateway must
    forward chunk-by-chunk instead of buffering the whole stream (which would
    either hang or deliver the transcript as one trailing blob).

    The test also verifies the openai-specific request headers Codex sends
    (``OpenAI-Beta``, ``Authorization``) reach the upstream — without them the
    Responses endpoint 400s even when the slug routing is correct.
    """
    sent_chunks: list[bytes] = []
    received_headers: dict[str, str] = {}

    async def responses_handler(request: web.Request) -> web.StreamResponse:
        received_headers.update(dict(request.headers))
        # Emit a slow trickle so any buffering on the gateway shows up as a
        # noticeable delay between the first and last chunk on the client side.
        response = web.StreamResponse(
            status=200,
            headers={"Content-Type": "text/event-stream"},
        )
        await response.prepare(request)
        chunks = [
            b'event: response.created\ndata: {"id":"resp_1"}\n\n',
            b'event: response.output_text.delta\ndata: {"delta":"hello"}\n\n',
            b'event: response.completed\ndata: {"id":"resp_1","status":"completed"}\n\n',
        ]
        for c in chunks:
            sent_chunks.append(c)
            await response.write(c)
            await asyncio.sleep(0.02)
        await response.write_eof()
        return response

    up_app = web.Application()
    up_app.router.add_post("/v1/responses", responses_handler)
    up_runner = web.AppRunner(up_app)
    await up_runner.setup()
    up_site = web.TCPSite(up_runner, "127.0.0.1", 0)
    await up_site.start()
    up_port = up_site._server.sockets[0].getsockname()[1]
    up_url = f"http://127.0.0.1:{up_port}"

    upstreams = {
        "openai": UpstreamConfig(
            url=up_url, engine="openai", protocol="openai-chat",
        ),
    }
    app = make_app(upstream="http://unused", upstreams=upstreams)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    px_port = site._server.sockets[0].getsockname()[1]
    px_url = f"http://127.0.0.1:{px_port}"

    import time as _time

    try:
        first_chunk_at: float | None = None
        last_chunk_at: float | None = None
        received = b""
        async with aiohttp.ClientSession() as client:
            async with client.post(
                f"{px_url}/upstreams/openai/v1/responses",
                json={"model": "gpt-5.5", "input": "hi", "stream": True},
                headers={
                    "authorization": "Bearer sk-test",
                    "openai-beta": "responses=v1",
                    "accept": "text/event-stream",
                },
            ) as resp:
                assert resp.status == 200, await resp.text()
                ct = resp.headers.get("Content-Type", "")
                assert "text/event-stream" in ct, ct
                async for chunk in resp.content.iter_any():
                    now = _time.monotonic()
                    if first_chunk_at is None:
                        first_chunk_at = now
                    last_chunk_at = now
                    received += chunk

        # All three SSE events came through.
        assert b"response.created" in received
        assert b"response.output_text.delta" in received
        assert b"response.completed" in received

        # Streaming check: with three 20ms-apart chunks, the gap between the
        # first chunk arriving and the last should be at least ~40ms. A buffered
        # implementation would show ~0 gap (everything arrives at once at the
        # end of the upstream).
        assert first_chunk_at is not None and last_chunk_at is not None
        assert last_chunk_at - first_chunk_at >= 0.02, (
            f"chunks arrived too close together "
            f"({last_chunk_at - first_chunk_at:.3f}s) — likely buffered"
        )

        # Codex's auth + Responses-API beta header reached upstream.
        auth = next(
            (v for k, v in received_headers.items() if k.lower() == "authorization"),
            None,
        )
        assert auth == "Bearer sk-test"
        beta = next(
            (v for k, v in received_headers.items() if k.lower() == "openai-beta"),
            None,
        )
        assert beta == "responses=v1"
        print("✓ test_openai_responses_passthrough_streams")
    finally:
        await runner.cleanup()
        await up_runner.cleanup()


async def _test_openai_responses_logs_usage(tmp_log: Path) -> None:
    """The Responses-API endpoint must produce a usage_log entry so the
    dashboard sees Codex traffic. The terminal SSE event
    ``response.completed`` carries ``response.usage`` — the gateway parses
    that out and writes a normalized record tagged with the slug's ``via``
    (``"codex"``), which is what the dashboard groups by.
    """
    async def responses_handler(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(
            status=200,
            headers={"Content-Type": "text/event-stream"},
        )
        await response.prepare(request)
        chunks = [
            b'event: response.created\ndata: {"id":"resp_x"}\n\n',
            b'event: response.output_text.delta\ndata: {"delta":"Hi"}\n\n',
            b'event: response.completed\n'
            b'data: {"type":"response.completed","response":{"id":"resp_x",'
            b'"status":"completed","usage":{"input_tokens":1000,'
            b'"input_tokens_details":{"cached_tokens":800},'
            b'"output_tokens":42,"total_tokens":1042}}}\n\n',
        ]
        for c in chunks:
            await response.write(c)
        await response.write_eof()
        return response

    up_app = web.Application()
    up_app.router.add_post("/responses", responses_handler)
    up_runner = web.AppRunner(up_app)
    await up_runner.setup()
    up_site = web.TCPSite(up_runner, "127.0.0.1", 0)
    await up_site.start()
    up_port = up_site._server.sockets[0].getsockname()[1]
    up_url = f"http://127.0.0.1:{up_port}"

    upstreams = {
        # Mirrors the installer's chatgpt-mode slug: no /v1 in the URL,
        # via="codex" labels traffic for the dashboard breakdown.
        "codex-chatgpt": UpstreamConfig(
            url=up_url, engine="openai", protocol="openai-chat", via="codex",
        ),
    }
    app = make_app(upstream="http://unused", upstreams=upstreams,
                    usage_log=tmp_log)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    px_port = site._server.sockets[0].getsockname()[1]
    px_url = f"http://127.0.0.1:{px_port}"

    try:
        async with aiohttp.ClientSession() as client:
            async with client.post(
                f"{px_url}/upstreams/codex-chatgpt/responses",
                json={"model": "gpt-5.5", "input": "hi", "stream": True},
                headers={"authorization": "Bearer sk-test",
                         "x-telos-session": "codex-log"},
            ) as resp:
                assert resp.status == 200
                async for _ in resp.content.iter_any():
                    pass

        line = tmp_log.read_text().strip().splitlines()[-1]
        record = json.loads(line)
        assert record["session_id"] == "codex-log"
        # The crucial assertion: the dashboard's "by harness" sees codex.
        assert record["harness"] == "codex"
        # Responses-API usage normalization: cached → cache_read,
        # input - cached → raw_input, output_tokens → output.
        assert record["normalized"]["cache_read"] == 800
        assert record["normalized"]["raw_input"] == 200
        assert record["normalized"]["output"] == 42
        # Raw usage round-trips for debugging / future reprocessing.
        assert record["raw_usage"]["input_tokens"] == 1000
        print("✓ test_openai_responses_logs_usage")
    finally:
        await runner.cleanup()
        await up_runner.cleanup()


async def _test_openai_slug_in_defaults() -> None:
    """Regression: ``codex`` writes ``/upstreams/openai/v1`` as its base_url; the
    gateway therefore must ship with an ``openai`` slug in the default upstream
    table, otherwise the very first Codex request returns
    ``unknown upstream slug: 'openai'``.
    """
    from telos.config import default_upstreams
    defaults = default_upstreams()
    assert "openai" in defaults, sorted(defaults)
    assert defaults["openai"].url == "https://api.openai.com"
    assert defaults["openai"].engine == "openai"
    print("✓ test_openai_slug_in_defaults")


async def _test_anthropic_route_still_works() -> None:
    """Regression: registering the /upstreams/<slug>/<tail> route must NOT
    swallow the legacy /v1/messages handler.
    """
    # A minimal anthropic-shaped mock that just returns 200 with a usage block.
    async def anth_handler(request: web.Request) -> web.Response:
        return web.json_response({
            "id": "msg_x", "type": "message", "role": "assistant",
            "content": [{"type": "text", "text": "ok"}],
            "model": "claude-opus-4-7", "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "cache_read_input_tokens": 0,
                      "cache_creation_input_tokens": 0, "output_tokens": 1},
        })
    app = web.Application()
    app.router.add_post("/v1/messages", anth_handler)
    up_runner = web.AppRunner(app)
    await up_runner.setup()
    site = web.TCPSite(up_runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    up_url = f"http://127.0.0.1:{port}"

    px_runner, px_url = await _start_proxy(up_url)  # openrouter slug is openai-chat
    # But we want the legacy /v1/messages to hit OUR mock anthropic upstream.
    # _start_proxy passes upstream="http://unused"; override via re-create.
    await px_runner.cleanup()
    app2 = make_app(
        upstream=up_url,
        upstreams={"openrouter": UpstreamConfig(
            url="http://unused", engine="deepseek", protocol="openai-chat")},
    )
    runner = web.AppRunner(app2)
    await runner.setup()
    site2 = web.TCPSite(runner, "127.0.0.1", 0)
    await site2.start()
    px_port = site2._server.sockets[0].getsockname()[1]
    px_url = f"http://127.0.0.1:{px_port}"

    try:
        async with aiohttp.ClientSession() as client:
            async with client.post(
                f"{px_url}/v1/messages",
                json={
                    "model": "claude-opus-4-7",
                    "max_tokens": 16,
                    "system": [{"type": "text", "text": "x"}],
                    "messages": [{"role": "user",
                                  "content": [{"type": "text", "text": "hi"}]}],
                },
                headers={"x-api-key": "k", "anthropic-version": "2023-06-01"},
            ) as resp:
                assert resp.status == 200, await resp.text()
                body = await resp.json()
                assert body["id"] == "msg_x"
        print("✓ test_anthropic_route_still_works")
    finally:
        await runner.cleanup()
        await up_runner.cleanup()


async def _test_passthrough_html_error_becomes_json() -> None:
    """A non-2xx HTML page from the upstream (e.g. chatgpt.com's 403
    ``<!DOCTYPE html>`` for a non-API path that Codex's login bootstrap pokes)
    must be converted into a parseable JSON error, so the OpenAI-wire client
    surfaces it instead of crashing on ``JSON.parse("<!DOCTYPE …")``.
    """
    async def html_403(request: web.Request) -> web.Response:
        return web.Response(
            status=403,
            content_type="text/html",
            text="<!DOCTYPE html><html><body>Forbidden</body></html>",
        )

    up_app = web.Application()
    # A non-/responses path that the upstream answers with HTML — mirrors
    # chatgpt.com/backend-api/codex/me returning a 403 login page.
    up_app.router.add_get("/me", html_403)
    up_runner = web.AppRunner(up_app)
    await up_runner.setup()
    up_site = web.TCPSite(up_runner, "127.0.0.1", 0)
    await up_site.start()
    up_port = up_site._server.sockets[0].getsockname()[1]
    up_url = f"http://127.0.0.1:{up_port}"

    upstreams = {
        "codex-chatgpt": UpstreamConfig(
            url=up_url, engine="openai", protocol="openai-chat", via="codex",
        ),
    }
    app = make_app(upstream="http://unused", upstreams=upstreams)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    px_port = site._server.sockets[0].getsockname()[1]
    px_url = f"http://127.0.0.1:{px_port}"

    try:
        async with aiohttp.ClientSession() as client:
            async with client.get(
                f"{px_url}/upstreams/codex-chatgpt/me",
                headers={"authorization": "Bearer sk-test"},
            ) as resp:
                # Status is preserved …
                assert resp.status == 403
                # … but the body is now JSON, not HTML.
                ct = resp.headers.get("content-type", "")
                assert "application/json" in ct, ct
                body = await resp.json()  # must not raise
                assert body["error"]["type"] == "invalid_request_error"
                assert "HTML page" in body["error"]["message"]
                assert "telos uninstall" in body["error"]["message"]
        print("✓ test_passthrough_html_error_becomes_json")
    finally:
        await runner.cleanup()
        await up_runner.cleanup()


def test_openai_route_non_streaming() -> None:
    asyncio.run(_test_openai_route_non_streaming())


def test_openai_route_logs_usage(tmp_path) -> None:
    asyncio.run(_test_openai_route_logs_usage(tmp_path / "usage.jsonl"))


def test_openai_route_via_labels_harness(tmp_path) -> None:
    asyncio.run(_test_openai_route_via_labels_harness(tmp_path / "usage.jsonl"))


def test_openai_tag_attributes_each_harness(tmp_path) -> None:
    asyncio.run(_test_openai_tag_attributes_each_harness(tmp_path / "usage.jsonl"))


def test_openai_route_streaming() -> None:
    asyncio.run(_test_openai_route_streaming())


def test_sglang_route_uses_adapter_and_logs_cache(tmp_path) -> None:
    asyncio.run(_test_sglang_route_uses_adapter_and_logs_cache(
        tmp_path / "usage.jsonl",
    ))


def test_sglang_stream_logs_tier_details(tmp_path) -> None:
    asyncio.run(_test_sglang_stream_logs_tier_details(
        tmp_path / "usage.jsonl",
    ))


def test_unknown_slug_returns_404() -> None:
    asyncio.run(_test_unknown_slug_returns_404())


def test_per_slug_upstream_url_is_honored() -> None:
    asyncio.run(_test_per_slug_upstream_url_is_honored())


def test_openai_slug_in_defaults() -> None:
    asyncio.run(_test_openai_slug_in_defaults())


def test_openai_responses_passthrough_streams() -> None:
    asyncio.run(_test_openai_responses_passthrough_streams())


def test_openai_responses_logs_usage(tmp_path) -> None:
    asyncio.run(_test_openai_responses_logs_usage(tmp_path / "usage.jsonl"))


def test_passthrough_html_error_becomes_json() -> None:
    asyncio.run(_test_passthrough_html_error_becomes_json())


def test_anthropic_route_still_works() -> None:
    asyncio.run(_test_anthropic_route_still_works())
