"""aiohttp reverse-proxy server.

Listens on ``POST /v1/messages``: runs the TELOS pipeline, forwards to Anthropic (or a
custom upstream), and supports SSE streaming responses. Other paths
(``/v1/messages/batches``, ``/v1/models``, etc.) are passed through as-is, without TELOS
rewriting.

Zero-intrusion integration:

::

    # start the proxy
    python -m telos.proxy --port 7171

    # any Anthropic-SDK client:
    export ANTHROPIC_BASE_URL=http://localhost:7171
    claude  # or your own agent

Design:

- ``ProxyApp`` holds a shared ``aiohttp.ClientSession``, reusing keep-alive connections.
- Streaming path (``stream=true``): writes to downstream while reading upstream content;
  at the same time, parses SSE events on a side channel, extracting usage from
  ``message_start`` / ``message_delta``.
- Non-streaming path: full read → forward; extracts usage from the JSON.
- Errors must be returned per the Anthropic wire schema (``{"type": "error", "error": {...}}``),
  otherwise the client SDK gets a structured exception.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, TYPE_CHECKING

import aiohttp
from aiohttp import web

from telos import load_engine
from telos.bridge import BridgeSessionState
from telos.corpus import record_call
from telos.output_filter import MODE_LABELS, TelosMode, apply_filter, build_filter
from telos.proxy.inspector import (
    SessionInspector as _SessionInspector,
    SessionInspectorEntry as _SessionInspectorEntry,
    ToolStat as _ToolStat,
    entry_to_json as _inspector_entry_to_json,
)
from telos.proxy.pipeline import (
    _RESERVED_CACHE_FIELDS,
    PipelineResult,
    process_anthropic_request,
    process_openai_request,
)
from telos.scripts.telos_anthropic_transport import _detect_harness_signal

if TYPE_CHECKING:
    from telos.config import UpstreamConfig


_DEFAULT_UPSTREAM = "https://api.anthropic.com"

# Harness tags accepted by the ``/_h/<harness>/`` URL prefix mechanism.
# Installers emit these in the base_url they patch into the agent's config so
# attribution is fixed at the source, no detection guess-work involved.
_VALID_HARNESS_TAGS = frozenset({"claude-code", "hermes", "openclaw", "codex"})

# Headers preserved when forwarding to upstream (auth / protocol version / Anthropic private beta).
# Host / Content-Length are computed by aiohttp itself; do not pass them through from the client.
# The OpenAI-side entries (``openai-*``, ``chatgpt-*``, ``session_id``, ``originator``) are
# needed by Codex's Responses API: ``codex`` attaches ``OpenAI-Beta: responses=v1``, an
# ``OpenAI-Organization`` / ``OpenAI-Project`` for billing routing, and proprietary
# ``ChatGPT-Account-Id`` / ``session_id`` / ``originator`` headers that the Responses
# endpoint relies on for conversation state. Dropping them caused the OpenAI server to
# 400 even when the slug routing was correct.
_FORWARD_HEADER_WHITELIST = (
    "x-api-key",
    "authorization",
    "anthropic-version",
    "anthropic-beta",
    "anthropic-dangerous-direct-browser-access",
    "user-agent",
    "openai-organization",
    "openai-project",
    "openai-beta",
    "chatgpt-account-id",
    "session_id",
    "originator",
)

_log = logging.getLogger("telos.proxy")

# Bounded backoff retry for transient failures during the upstream connection phase
# (TLS handshake / connect). These errors all occur before the request body reaches
# upstream, so retrying does not double-bill / double-process.
_MAX_CONNECT_RETRIES = 3
_CONNECT_BACKOFF_BASE = 0.5   # seconds; exponential backoff 0.5 / 1.0 / 2.0


def _wire_tool_result_first(req: Mapping[str, Any]) -> bool:
    """Verify that the tool_result blocks of every user message in the request body come before non-tool_result blocks.

    Anthropic requires the tool_result of a user message to be physically first; a violation
    is rejected by upstream with a 400. The proxy uses this as a safety net before sending --
    if TELOS rewriting produces an illegal wire, it falls back to passthrough.
    """
    messages = req.get("messages")
    if not isinstance(messages, list):
        return True
    for msg in messages:
        if not isinstance(msg, Mapping) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        seen_non_tool_result = False
        for blk in content:
            if not isinstance(blk, Mapping):
                continue
            if blk.get("type") == "tool_result":
                if seen_non_tool_result:
                    return False
            else:
                seen_non_tool_result = True
    return True


# ---------------------------------------------------------------------------
# Stable session-id derivation
# ---------------------------------------------------------------------------

def _client_identity(headers: Mapping[str, str]) -> str:
    """Get a stable string from the headers that distinguishes different callers.

    Prefers ``x-api-key``; then ``authorization`` (with the ``Bearer`` prefix stripped);
    if neither is present, returns an empty string (multiple anonymous clients will share
    a session-id -- good enough for single-machine development).
    """
    api_key = headers.get("x-api-key")
    if api_key:
        return api_key
    auth = headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return auth.strip()


def _derive_session_id(raw: Mapping[str, Any], headers: Mapping[str, str]) -> str:
    """Derive a stable session-id from "client identity + conversation seed".

    Under the Anthropic /v1/messages protocol, the part of a "conversation" that does not
    change across turns is: ``system`` / ``tools`` / ``messages[0]`` (i.e. the first user
    message of the conversation). Each turn only appends new content to the tail of
    ``messages[]``. So the hash of these four items uniquely identifies a conversation.

    Returns something like ``"telos-<16-hex-bytes>"``, convenient for grep.
    """
    seed = {
        "client": _client_identity(headers),
        "system": raw.get("system") or [],
        "tools": raw.get("tools") or [],
        # messages[0] is usually the first user message of the conversation;
        # even if the list is empty, a None placeholder is used to keep the hash computable.
        "msg0": (raw.get("messages") or [None])[0],
    }
    body = json.dumps(seed, sort_keys=True, ensure_ascii=False,
                       default=str).encode("utf-8")
    digest = hashlib.blake2b(body, digest_size=8).hexdigest()
    return f"telos-{digest}"


# ---------------------------------------------------------------------------
# Session registry: keyed by session_id, with an LRU cap to avoid memory blowup on long runs
# ---------------------------------------------------------------------------

_DEFAULT_MAX_SESSIONS = 10_000


class _SessionRegistry:
    """A bounded-LRU wrapper around ``dict[session_id, BridgeSessionState]``.

    Under aiohttp's single event loop, all accesses happen on the same thread, so no lock
    is needed. Concurrent requests for the same session see the accumulation sequentially
    -- good enough. If true concurrency is needed in the future, just add an
    ``asyncio.Lock`` per entry.
    """

    def __init__(self, max_size: int = _DEFAULT_MAX_SESSIONS) -> None:
        self._max = max_size
        self._sessions: OrderedDict[str, BridgeSessionState] = OrderedDict()

    def get_or_create(self, session_id: str) -> BridgeSessionState:
        if session_id in self._sessions:
            self._sessions.move_to_end(session_id)
            return self._sessions[session_id]
        state = BridgeSessionState()
        self._sessions[session_id] = state
        if len(self._sessions) > self._max:
            evicted, _ = self._sessions.popitem(last=False)
            _log.info("session LRU evicted: %s (size=%d)", evicted, self._max)
        return state

    def __len__(self) -> int:
        return len(self._sessions)


# ---------------------------------------------------------------------------
# usage normalization (same schema as telos_anthropic_transport._normalize_usage)
# ---------------------------------------------------------------------------

def _normalize_usage(u: dict[str, Any]) -> dict[str, int]:
    """Normalization into 4 buckets; keeps ``cache_creation.ephemeral_{5m,1h}_input_tokens``
    in raw_usage, which the dashboard reads to bill correctly at the 5m / 1h prices."""
    return {
        "raw_input": int(u.get("input_tokens", 0) or 0),
        "cache_read": int(u.get("cache_read_input_tokens", 0) or 0),
        "cache_write": int(u.get("cache_creation_input_tokens", 0) or 0),
        "output": int(u.get("output_tokens", 0) or 0),
    }


def _normalize_openai_usage(u: Mapping[str, Any]) -> dict[str, int]:
    """OpenAI / DeepSeek / OpenRouter usage → the same 4-bucket schema.

    Two competing field conventions across upstreams:
    - DeepSeek (passed through by OpenRouter):
        ``prompt_cache_hit_tokens`` / ``prompt_cache_miss_tokens``
    - OpenAI native:
        ``prompt_tokens`` + ``prompt_tokens_details.cached_tokens``
    Neither side bills for cache-write separately on the OpenAI ecosystem, so
    that bucket is always 0. ``output`` comes from ``completion_tokens``.
    """
    if not u:
        return {"raw_input": 0, "cache_read": 0, "cache_write": 0, "output": 0}
    if "prompt_cache_hit_tokens" in u or "prompt_cache_miss_tokens" in u:
        hit = int(u.get("prompt_cache_hit_tokens") or 0)
        miss = int(u.get("prompt_cache_miss_tokens") or 0)
        return {
            "raw_input": miss,
            "cache_read": hit,
            "cache_write": 0,
            "output": int(u.get("completion_tokens") or 0),
        }
    pt = int(u.get("prompt_tokens") or 0)
    details_obj = u.get("prompt_tokens_details") or {}
    details = details_obj if isinstance(details_obj, Mapping) else {}
    cached = int(details.get("cached_tokens", 0) or u.get("cached_tokens", 0))
    return {
        "raw_input": max(pt - cached, 0),
        "cache_read": cached,
        "cache_write": 0,
        "output": int(u.get("completion_tokens") or 0),
    }


def _normalize_responses_usage(u: Mapping[str, Any]) -> dict[str, int]:
    """OpenAI Responses API usage → the same 4-bucket schema.

    Responses API uses ``input_tokens`` plus an optional
    ``input_tokens_details.cached_tokens`` (Codex-style traffic through
    chatgpt.com keeps the same shape). There is no cache-write concept on
    this endpoint, so that bucket is always 0. ``output_tokens`` covers both
    visible output and reasoning tokens — the dashboard pays per these.
    """
    if not u:
        return {"raw_input": 0, "cache_read": 0, "cache_write": 0, "output": 0}
    it = int(u.get("input_tokens") or 0)
    cached = int((u.get("input_tokens_details") or {}).get("cached_tokens", 0))
    return {
        "raw_input": max(it - cached, 0),
        "cache_read": cached,
        "cache_write": 0,
        "output": int(u.get("output_tokens") or 0),
    }


def _anthropic_error(status: int, err_type: str, message: str) -> web.Response:
    return web.json_response(
        {"type": "error", "error": {"type": err_type, "message": message}},
        status=status,
    )


def _openai_error(status: int, err_type: str, message: str) -> web.Response:
    """OpenAI-shaped error envelope, for clients on the OpenAI wire (Codex's
    ``wire_api = "responses"`` passthrough). Mirrors the ``{"error": {...}}``
    body chatgpt.com / api.openai.com return so the client parses it natively.
    """
    return web.json_response(
        {"error": {"message": message, "type": err_type, "code": err_type}},
        status=status,
    )


def _looks_like_html(content_type: str, body: bytes) -> bool:
    """True when a response body is an HTML page rather than JSON.

    Checks the declared content-type first, then sniffs the leading bytes
    (upstreams/CDNs sometimes mislabel HTML error pages as ``text/plain`` or
    omit the header). Only the first non-whitespace bytes matter.
    """
    if "text/html" in content_type.lower():
        return True
    head = body[:512].lstrip().lower()
    return head.startswith(b"<!doctype") or head.startswith(b"<html")


# ---------------------------------------------------------------------------
# Raw message summary (the developer page shows the conversation as it was before TELOS rewriting)
# ---------------------------------------------------------------------------

_RAW_MSG_PREVIEW_CHARS = 240


def _summarize_raw_block(blk: Mapping[str, Any]) -> dict[str, Any]:
    """A single content block → a summary dict (type / chars / preview / extra).

    text → the text itself; tool_use → the input JSON (extra=tool name);
    tool_result → the content (extra=tool_use_id); everything else → the whole block JSON.
    The preview is truncated to ``_RAW_MSG_PREVIEW_CHARS``.
    """
    btype = str(blk.get("type", "?"))
    extra = ""
    if btype == "text":
        preview = str(blk.get("text") or "")
    elif btype == "tool_use":
        preview = json.dumps(blk.get("input") or {}, ensure_ascii=False,
                             default=str)
        extra = str(blk.get("name") or "")
    elif btype == "tool_result":
        content = blk.get("content")
        preview = (content if isinstance(content, str)
                   else json.dumps(content, ensure_ascii=False, default=str))
        extra = str(blk.get("tool_use_id") or "")
    else:
        preview = json.dumps(blk, ensure_ascii=False, default=str)
    chars = len(preview)
    return {
        "type": btype,
        "chars": chars,
        "preview": preview[:_RAW_MSG_PREVIEW_CHARS],
        "truncated": chars > _RAW_MSG_PREVIEW_CHARS,
        "extra": extra,
    }


def _summarize_raw_messages(raw: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Extract a summary of each message from the raw ``/v1/messages`` request body.

    Returns ``[{"role", "blocks": [<block summary>...]}, ...]``. When content is a string,
    it is normalized into a single text block. Never raises on error -- the developer page
    must always be renderable.
    """
    messages = raw.get("messages")
    if not isinstance(messages, list):
        return []
    out: list[dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, Mapping):
            continue
        role = str(msg.get("role", "?"))
        content = msg.get("content")
        blocks: list[dict[str, Any]] = []
        if isinstance(content, str):
            blocks.append(_summarize_raw_block({"type": "text", "text": content}))
        elif isinstance(content, list):
            for blk in content:
                if isinstance(blk, Mapping):
                    blocks.append(_summarize_raw_block(blk))
                else:
                    s = str(blk)
                    blocks.append({
                        "type": "?", "chars": len(s),
                        "preview": s[:_RAW_MSG_PREVIEW_CHARS],
                        "truncated": len(s) > _RAW_MSG_PREVIEW_CHARS,
                        "extra": "",
                    })
        out.append({"role": role, "blocks": blocks})
    return out


def _summarize_openai_messages(raw: Mapping[str, Any]) -> list[dict[str, Any]]:
    """OpenAI ChatCompletions counterpart of ``_summarize_raw_messages``.

    OpenAI's ``content`` is usually a flat string; ``tool_calls`` is attached
    on the assistant message; ``role=tool`` carries ``tool_call_id``. We
    surface enough for the developer panel to render without needing the IR.
    """
    messages = raw.get("messages")
    if not isinstance(messages, list):
        return []
    out: list[dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, Mapping):
            continue
        role = str(msg.get("role", "?"))
        content = msg.get("content")
        blocks: list[dict[str, Any]] = []
        if isinstance(content, str) and content:
            blocks.append({
                "type": "text",
                "chars": len(content),
                "preview": content[:_RAW_MSG_PREVIEW_CHARS],
                "truncated": len(content) > _RAW_MSG_PREVIEW_CHARS,
                "extra": "",
            })
        elif isinstance(content, list):
            for blk in content:
                if isinstance(blk, Mapping):
                    blocks.append(_summarize_raw_block(blk))
        for tc in msg.get("tool_calls") or []:
            if isinstance(tc, Mapping):
                fn = (tc.get("function") or {}).get("name", "?")
                blocks.append({
                    "type": "tool_call",
                    "chars": 0, "preview": "", "truncated": False,
                    "extra": f"name={fn}",
                })
        if role == "tool":
            blocks.append({
                "type": "tool_result",
                "chars": 0, "preview": "", "truncated": False,
                "extra": f"id={msg.get('tool_call_id', '')}",
            })
        out.append({"role": role, "blocks": blocks})
    return out


# ---------------------------------------------------------------------------
# ProxyApp: holds the shared session + the request handlers
# ---------------------------------------------------------------------------

class ProxyApp:
    def __init__(
        self,
        *,
        upstream: str = _DEFAULT_UPSTREAM,
        upstreams: Mapping[str, "UpstreamConfig"] | None = None,
        usage_log: Path | None = None,
        harness_override: str | None = None,
        request_timeout: float = 600.0,
        strict: bool = False,
        max_sessions: int = _DEFAULT_MAX_SESSIONS,
        dashboard_refresh: int = 5,
        mode: TelosMode | None = None,
        corpus_dir: Path | None = None,
        record: bool = True,
    ):
        self.upstream = upstream.rstrip("/")
        # Named upstream table for the multi-backend ``/upstreams/<slug>/...``
        # path. Empty by default (legacy ``/v1/messages`` -> ``self.upstream``
        # still works); callers pass in ``cfg.upstreams`` to enable.
        self.upstreams: dict[str, "UpstreamConfig"] = dict(upstreams or {})
        self.usage_log = usage_log
        self.harness_override = harness_override
        self.request_timeout = request_timeout
        # Process-level default switch; can be overridden by a single request's X-Telos-Mode
        # header (and the first request's value is made sticky to that session).
        self.mode = mode or TelosMode()
        # Tool-result filter: uses rtk if the rtk binary is available, otherwise a pure
        # Python fallback. Constructed once, reused across all sessions (stateless).
        self._filter = build_filter()
        # Session recording: enabled by default, records each call's raw request into
        # corpus_dir for `telos replay` to replay. Records the request only, not the
        # response. Can be turned off with --no-record.
        self._record = record
        self._corpus_dir = corpus_dir if record else None
        # strict=False (default): on TELOS failure, pass through to upstream as-is,
        # guaranteeing the optimization layer never breaks correctness (the same
        # "rewrite fails → original command" principle as RTK).
        # strict=True: for testing / debugging, a TELOS failure returns 500 directly.
        self.strict = strict
        # The meta-refresh interval (seconds) of /__telos/dashboard; 0 = disable auto-refresh.
        self.dashboard_refresh = dashboard_refresh

        if usage_log is not None:
            usage_log.parent.mkdir(parents=True, exist_ok=True)

        self._session: aiohttp.ClientSession | None = None
        self._call_count = 0
        self._pipeline_failures = 0
        self._registry = _SessionRegistry(max_size=max_sessions)
        self._inspector = _SessionInspector(max_size=max_sessions)
        # per-client harness memory: key = _client_identity(headers). Once a client is
        # confidently identified by any request (HTTP headers / rich content), it is
        # remembered -- subsequent tool-less auxiliary requests it sends (Haiku title
        # generation / topic detection, etc.) have no harness features at all, and inherit
        # via this memory rather than being misclassified as openclaw. One entry per API
        # key, tiny in size.
        self._client_harness: dict[str, str] = {}

    # ------------------------------------------------------------------
    # harness resolution
    # ------------------------------------------------------------------

    def _resolve_harness(self, request: web.Request,
                         raw: Mapping[str, Any]) -> str:
        """Decide which harness to use to parse this request.

        Priority:
          1. URL tag stamped by ``handle_tagged_route`` (``/_h/<harness>/``)
             — the authoritative source for installs from 2026-05+. Bypasses
             all detection.
          2. Explicit ``--harness`` override (proxy CLI flag or upstream ``via``
             temporarily injected by the slug route).
          3. A confident signal from this request (header / envelope tags /
             thinking block / tool fingerprint). Pinned to per-client memory.
          4. This client's prior confident memory.
          5. Fallback ``openclaw`` (Anthropic-compatible catch-all).

        Steps 3-5 are legacy fallback for un-tagged installs; in a fully
        migrated fleet only step 1 fires.
        """
        tag = request.get("_telos_harness_tag")
        if tag:
            return tag
        if self.harness_override:
            return self.harness_override
        client = _client_identity(request.headers)
        signal = _detect_harness_signal(raw, request.headers)
        if signal is not None:
            if self._client_harness.get(client) != signal:
                self._client_harness[client] = signal
                _log.info("harness pinned: client=%s -> %s (User-Agent=%r)",
                          client or "(anon)", signal,
                          request.headers.get("user-agent", ""))
            return signal
        # This request has no confident signal -- inherit the harness this client was
        # confidently identified as before.
        remembered = self._client_harness.get(client)
        if remembered is not None:
            return remembered
        return "openclaw"


    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def _session_get(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(
                limit=64,
                ttl_dns_cache=300,
                keepalive_timeout=30.0,
                # Mitigates the jitter from aiohttp SSL transports not being released
                # promptly on macOS.
                enable_cleanup_closed=True,
            )
            self._session = aiohttp.ClientSession(
                connector=connector,
                # sock_connect: a stuck connection fails fast into a retry rather than
                # exhausting total.
                timeout=aiohttp.ClientTimeout(
                    total=self.request_timeout, sock_connect=15.0,
                ),
                # auto_decompress=True: the proxy needs to understand the response body
                # (parse usage, peek SSE), so it must get the decompressed plaintext. The
                # response does not forward Content-Encoding, so this cannot be turned off
                # -- otherwise the client would get compressed bytes with no declared encoding.
                auto_decompress=True,
            )
        return self._session

    async def _post_upstream(
        self,
        session: aiohttp.ClientSession,
        url: str,
        body_bytes: bytes,
        headers: dict[str, str],
        call_index: int,
    ) -> aiohttp.ClientResponse:
        """POST to upstream, with bounded backoff retry for transient failures during the "connection phase".

        Only retries two classes of errors **that occur before a response is produced**:

        - ``ClientConnectorError``: the connection was never established (TLS handshake /
          connect failed).
        - ``ServerDisconnectedError``: upstream disconnected before returning any response,
          usually a dead reused keep-alive connection.

        In both failure classes, not a single byte of the request body has reached
        upstream yet, so retrying is 100% safe -- no double-billing, no double-processing
        by upstream. Other ``ClientError`` / timeouts are not retried (the request may have
        been delivered) and are re-raised to ``handle_messages`` as before to return a 502.
        """
        last_exc: Exception | None = None
        for attempt in range(_MAX_CONNECT_RETRIES + 1):
            try:
                return await session.post(url, data=body_bytes, headers=headers)
            except (aiohttp.ClientConnectorError,
                    aiohttp.ServerDisconnectedError) as e:
                last_exc = e
                if attempt >= _MAX_CONNECT_RETRIES:
                    break
                backoff = _CONNECT_BACKOFF_BASE * (2 ** attempt)
                _log.warning(
                    "upstream connect failed (call=%d, attempt %d/%d): %s "
                    "— retrying in %.1fs", call_index, attempt + 1,
                    _MAX_CONNECT_RETRIES + 1, e, backoff)
                await asyncio.sleep(backoff)
        assert last_exc is not None
        raise last_exc

    async def on_shutdown(self, app: web.Application) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()

    def _forward_headers(self, request: web.Request) -> dict[str, str]:
        out: dict[str, str] = {"content-type": "application/json"}
        for h in _FORWARD_HEADER_WHITELIST:
            v = request.headers.get(h)
            if v is not None:
                out[h] = v
        accept = request.headers.get("accept")
        if accept:
            out["accept"] = accept
        return out

    def _resolve_mode(
        self, request: web.Request, session_state: BridgeSessionState,
    ) -> TelosMode:
        """Decide the mode for this request.

        Priority: ``X-Telos-Mode`` header > the session's locked sticky_mode > the proxy
        process default. The first request carrying the header makes the value sticky to
        that session, guaranteeing the same session never changes gears throughout a
        comparison experiment.
        """
        header = request.headers.get("x-telos-mode")
        if header:
            mode = TelosMode.from_label(header)
            if session_state.sticky_mode is None:
                session_state.sticky_mode = mode.label
            return mode
        if session_state.sticky_mode is not None:
            return TelosMode.from_label(session_state.sticky_mode)
        return self.mode

    def set_mode(self, mode: TelosMode) -> None:
        """Hot-swap the process-level default mode.

        Under the single-threaded asyncio event loop, the assignment to ``self.mode`` is
        atomic, and the next call to ``_resolve_mode`` takes effect immediately -- no
        gateway restart needed. Old sessions that have already locked sticky_mode are
        unaffected (this is the intended semantics of sticky).
        """
        old = self.mode
        self.mode = mode
        _log.info("default mode hot-swap: %s → %s (telos=%s rtk=%s)",
                  old.label, mode.label, mode.telos, mode.rtk)

    def _resolve_compare_group(
        self, request: web.Request, session_state: BridgeSessionState,
    ) -> str | None:
        """The comparison-experiment group label (``X-Telos-Compare-Group`` header), sticky."""
        header = request.headers.get("x-telos-compare-group")
        if header:
            if session_state.compare_group is None:
                session_state.compare_group = header
            return header
        return session_state.compare_group

    # ------------------------------------------------------------------
    # POST /v1/messages -- the main handler
    # ------------------------------------------------------------------

    async def handle_messages(self, request: web.Request) -> web.StreamResponse:
        self._call_count += 1
        call_index = self._call_count

        try:
            raw = await request.json()
        except web.HTTPRequestEntityTooLarge as e:  # noqa: BLE001
            # The request body exceeds client_max_size: distinguished from a JSON syntax
            # error, returns 413.
            _log.warning("request body too large (call=%d): %s", call_index, e)
            return _anthropic_error(413, "request_too_large", str(e))
        except Exception as e:  # noqa: BLE001
            return _anthropic_error(400, "invalid_request_error", f"Invalid JSON: {e}")

        session_id = (
            request.headers.get("x-telos-session")
            or (raw.get("metadata") or {}).get("user_id")
            or _derive_session_id(raw, request.headers)
        )
        session_state = self._registry.get_or_create(session_id)

        # ---- 0a. Record the session corpus (raw request, before RTK / TELOS rewriting) ----
        if self._corpus_dir is not None:
            try:
                record_call(self._corpus_dir, session_id, call_index, raw)
            except Exception:  # noqa: BLE001 — a recording failure must never affect the proxy
                _log.exception("corpus record failed (call=%d)", call_index)

        # ---- 0. Resolve this request's mode / compare_group (sticky per session) ----
        mode = self._resolve_mode(request, session_state)
        compare_group = self._resolve_compare_group(request, session_state)

        # ---- 0b. RTK tool-result filtering (only mode.rtk) ----
        effective_raw: Mapping[str, Any] = raw
        filter_reduction: dict[str, Any] = {}
        if mode.rtk:
            try:
                effective_raw, fstats = apply_filter(raw, self._filter)
                filter_reduction = fstats.as_dict()
            except Exception:  # noqa: BLE001 — the filter layer never breaks the request
                _log.exception("RTK filter failed (call=%d) — using raw request",
                               call_index)
                effective_raw = raw

        # ---- 1. TELOS pipeline (only mode.telos) ----
        if mode.telos:
            # The harness is resolved at the proxy layer: HTTP headers / per-client memory
            # all live at this layer, and content detection cannot see them. The resolved
            # result is passed explicitly to the pipeline, overriding the pipeline's
            # internal content detection.
            resolved_harness = self._resolve_harness(request, raw)
            try:
                result = process_anthropic_request(
                    effective_raw,
                    session_id=session_id,
                    session_state=session_state,
                    harness_name=resolved_harness,
                )
            except Exception as e:  # noqa: BLE001
                self._pipeline_failures += 1
                # Log a full traceback on the first failure; subsequent ones log a single
                # short line to reduce log noise.
                if self._pipeline_failures == 1:
                    _log.exception("TELOS pipeline failed (call=%d) — falling back to "
                                   "passthrough. Further failures will log a single line.",
                                   call_index)
                else:
                    _log.warning("TELOS pipeline failed (call=%d, total=%d): %s",
                                 call_index, self._pipeline_failures, e)
                if self.strict:
                    return _anthropic_error(500, "api_error",
                                            f"TELOS pipeline failed: {e}")
                # Graceful degradation: use the (filtered) raw as the wire, build an empty
                # result for passthrough.
                result = PipelineResult(
                    wire=dict(effective_raw),
                    harness="passthrough",
                    plan_slots=[],
                    routing_key=None,
                    model=raw.get("model", ""),
                )
        else:
            # TELOS off: do not add cache markers, the (filtered) raw is used directly as the wire.
            result = PipelineResult(
                wire=dict(effective_raw),
                harness="rtk-only" if mode.rtk else "passthrough",
                plan_slots=[],
                routing_key=None,
                model=raw.get("model", ""),
            )

        # The proxy layer backfills the switch metadata into result, for _log_usage to persist.
        result.mode = mode.label
        result.compare_group = compare_group
        result.tool_output_reduction = filter_reduction
        # The message summary of the raw (pre-TELOS-rewrite) request, for the developer page.
        result.raw_messages = _summarize_raw_messages(raw)

        # Safety check before sending: if TELOS rewriting produces a structurally illegal
        # wire (tool_result not first, etc.), it must not hit upstream and get a 400. Fall
        # back to passthrough -- the un-rewritten effective_raw is structurally valid.
        # Consistent with the strict=False "the optimization layer never breaks correctness".
        if not _wire_tool_result_first(result.wire):
            if _wire_tool_result_first(effective_raw):
                _log.warning(
                    "TELOS wire failed tool_result-order check (call=%d) — "
                    "falling back to passthrough", call_index)
                result.wire = dict(effective_raw)
                result.harness = "passthrough"
            else:
                _log.warning(
                    "tool_result-order issue not introduced by TELOS "
                    "(call=%d) — forwarding request as-is", call_index)

        # Attribution overlay: keep the dashboard's "breakdown by harness"
        # pointed at the *calling agent* (hermes / openclaw / …) even when the
        # optimization layer degraded to passthrough or rtk-only. Mode info
        # lives in result.mode; this field is the caller identity. Mirrors the
        # overlay in _handle_openai_chat / _handle_openai_responses.
        if result.harness in ("passthrough", "rtk-only"):
            result.harness = self._resolve_harness(request, raw)

        is_streaming = bool(raw.get("stream", False))
        url = f"{self.upstream}/v1/messages"
        headers = self._forward_headers(request)
        body_bytes = json.dumps(result.wire).encode("utf-8")

        session = await self._session_get()
        t0 = time.time()

        try:
            upstream = await self._post_upstream(
                session, url, body_bytes, headers, call_index)
        except aiohttp.ClientError as e:
            _log.error("Upstream connection failed after retries (call=%d): %s",
                       call_index, e)
            return _anthropic_error(502, "api_error", f"Upstream error: {e}")

        if is_streaming:
            return await self._stream_response(
                request, upstream, session_id, result, session_state,
                call_index, t0,
            )
        return await self._buffered_response(
            upstream, session_id, result, session_state, call_index, t0,
        )

    # ------------------------------------------------------------------
    # Non-streaming path
    # ------------------------------------------------------------------

    async def _buffered_response(
        self,
        upstream: aiohttp.ClientResponse,
        session_id: str,
        result: PipelineResult,
        session_state: BridgeSessionState,
        call_index: int,
        t0: float,
    ) -> web.Response:
        try:
            body = await upstream.read()
            status = upstream.status
            ct = upstream.headers.get("content-type", "application/json")
        finally:
            upstream.release()

        usage: dict[str, Any] = {}
        if status == 200:
            try:
                parsed = json.loads(body.decode("utf-8"))
                usage = parsed.get("usage") or {}
            except Exception:  # noqa: BLE001
                pass
        elif status >= 400:
            # Record the upstream error body -- the body of a 4xx/5xx usually states the
            # cause directly (e.g. messages.N: tool_result must be first), and is the
            # first-hand clue for troubleshooting.
            _log.warning("upstream %d (call=%d): %s", status, call_index,
                         body[:2000].decode("utf-8", "replace"))

        self._accumulate_into_state(session_state, usage)
        latency_s = time.time() - t0
        self._log_usage(
            session_id, result, usage, session_state,
            latency_s=latency_s,
            streaming=False,
            status=status,
            call_index=call_index,
        )
        self._update_inspector(session_id, result, usage, call_index, latency_s)
        return web.Response(body=body, status=status, headers={"Content-Type": ct})

    # ------------------------------------------------------------------
    # Streaming path (SSE)
    # ------------------------------------------------------------------

    async def _stream_response(
        self,
        request: web.Request,
        upstream: aiohttp.ClientResponse,
        session_id: str,
        result: PipelineResult,
        session_state: BridgeSessionState,
        call_index: int,
        t0: float,
    ) -> web.StreamResponse:
        status = upstream.status

        # An error response will not be SSE: read the full body and return it at once.
        if status != 200:
            try:
                body = await upstream.read()
                ct = upstream.headers.get("content-type", "application/json")
            finally:
                upstream.release()
            if status >= 400:
                _log.warning("upstream %d (call=%d): %s", status, call_index,
                             body[:2000].decode("utf-8", "replace"))
            self._log_usage(
                session_id, result, {}, session_state,
                latency_s=time.time() - t0,
                streaming=True,
                status=status,
                call_index=call_index,
            )
            return web.Response(body=body, status=status, headers={"Content-Type": ct})

        # 200: start streaming forwarding.
        downstream = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": upstream.headers.get(
                    "content-type", "text/event-stream"
                ),
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
        try:
            await downstream.prepare(request)
        except (ConnectionResetError, asyncio.CancelledError) as e:
            # The downstream client disconnected before we wrote back the response headers
            # -- common and harmless (user interruption, client timeout). Not an error:
            # silently release the upstream connection and wrap up, to avoid an escaping
            # exception printing a noisy traceback and to avoid an upstream connection leak.
            _log.info("downstream disconnected before stream start (call=%d): %s",
                      call_index, e)
            upstream.release()
            return downstream

        usage_aggregate: dict[str, Any] = {}
        sse_buf = b""

        try:
            async for chunk in upstream.content.iter_any():
                # Forward immediately, without waiting for a complete SSE event
                try:
                    await downstream.write(chunk)
                except (ConnectionResetError, asyncio.CancelledError):
                    _log.info("downstream disconnected mid-stream (call=%d)", call_index)
                    break

                # Side channel: accumulate and parse complete SSE blocks to extract usage
                sse_buf += chunk
                while b"\n\n" in sse_buf:
                    block, sse_buf = sse_buf.split(b"\n\n", 1)
                    self._peek_sse_block(block, usage_aggregate)
        except aiohttp.ClientPayloadError:
            _log.warning("upstream closed connection mid-stream (call=%d)", call_index)

        try:
            await downstream.write_eof()
        except Exception:
            pass
        finally:
            upstream.release()

        self._accumulate_into_state(session_state, usage_aggregate)
        latency_s = time.time() - t0
        self._log_usage(
            session_id, result, usage_aggregate, session_state,
            latency_s=latency_s,
            streaming=True,
            status=200,
            call_index=call_index,
        )
        self._update_inspector(session_id, result, usage_aggregate, call_index, latency_s)
        return downstream

    # ------------------------------------------------------------------
    # SSE parsing (side channel)
    # ------------------------------------------------------------------

    def _peek_sse_block(self, block: bytes, usage: dict[str, Any]) -> None:
        """Extract usage fields from an SSE event block.

        Anthropic SSE protocol: ``message_start`` carries the input/cache fields,
        ``message_delta`` carries the cumulative output_tokens. Errors are silently
        swallowed and never affect the proxy.
        """
        event: str | None = None
        data_raw: bytes | None = None
        for line in block.split(b"\n"):
            if line.startswith(b"event:"):
                event = line[6:].strip().decode("ascii", "ignore")
            elif line.startswith(b"data:"):
                data_raw = line[5:].strip()
        if data_raw is None:
            return
        try:
            data = json.loads(data_raw.decode("utf-8"))
        except Exception:  # noqa: BLE001
            return

        if event == "message_start":
            u = (data.get("message") or {}).get("usage") or {}
            for k in (
                "input_tokens",
                "cache_read_input_tokens",
                "cache_creation_input_tokens",
                "output_tokens",
            ):
                if k in u:
                    usage[k] = int(u[k])
            # Key: carry over the 5m / 1h split as-is, for the dashboard to do precise billing
            cc = u.get("cache_creation")
            if isinstance(cc, dict):
                usage["cache_creation"] = {
                    "ephemeral_5m_input_tokens":
                        int(cc.get("ephemeral_5m_input_tokens", 0) or 0),
                    "ephemeral_1h_input_tokens":
                        int(cc.get("ephemeral_1h_input_tokens", 0) or 0),
                }
        elif event == "message_delta":
            u = data.get("usage") or {}
            if "output_tokens" in u:
                usage["output_tokens"] = int(u["output_tokens"])

    # ------------------------------------------------------------------
    # GET /__telos/dashboard —— live savings dashboard
    # ------------------------------------------------------------------

    async def handle_dashboard(self, request: web.Request) -> web.Response:
        """Read the usage_log live → aggregate → render HTML.

        Each request re-reads the whole file + re-renders. The usage_log will not be large
        (on the order of tens of KB/day, one or two hundred bytes per jsonl line), so there
        is no caching or incremental processing.
        """
        # Lazy import, to avoid a hard dependency on the dashboard module at proxy cold start.
        from telos.scripts.build_savings_dashboard import render_from_usage_log

        try:
            body = render_from_usage_log(
                self.usage_log,
                refresh_seconds=self.dashboard_refresh,
            )
        except Exception as e:  # noqa: BLE001
            _log.exception("dashboard render failed")
            return web.Response(
                text=f"<pre>dashboard render failed: {e}</pre>",
                content_type="text/html", status=500,
            )
        return web.Response(text=body, content_type="text/html")

    # ------------------------------------------------------------------
    # GET/POST /__telos/control/mode -- hot-update the default mode (loopback only)
    # ------------------------------------------------------------------

    @staticmethod
    def _is_loopback(request: web.Request) -> bool:
        """The control endpoint only accepts local-origin requests, preventing a remote from changing gateway behavior."""
        remote = request.remote or ""
        return remote in ("127.0.0.1", "::1", "::ffff:127.0.0.1")

    async def handle_control_mode(self, request: web.Request) -> web.Response:
        """GET reads the current default mode; POST {"mode": "<label>"} hot-swaps the default mode."""
        if not self._is_loopback(request):
            return web.json_response(
                {"error": "control endpoint is loopback-only"}, status=403)

        if request.method == "GET":
            return web.json_response({
                "mode": self.mode.label,
                "telos": self.mode.telos,
                "rtk": self.mode.rtk,
            })

        try:
            body = await request.json()
        except Exception as e:  # noqa: BLE001
            return web.json_response(
                {"error": f"invalid JSON: {e}"}, status=400)
        label = (body or {}).get("mode")
        if label not in MODE_LABELS:
            return web.json_response(
                {"error": f"unknown mode {label!r}; expected one of "
                          f"{', '.join(MODE_LABELS)}"},
                status=400)
        self.set_mode(TelosMode.from_label(label))
        return web.json_response({
            "mode": self.mode.label,
            "telos": self.mode.telos,
            "rtk": self.mode.rtk,
        })

    # ------------------------------------------------------------------
    # POST /__telos/control/reset -- clear the usage_log → zero the dashboard
    # ------------------------------------------------------------------

    async def handle_control_reset(self, request: web.Request) -> web.Response:
        """Zero the savings dashboard by clearing the usage_log (loopback-only).

        The current log is rotated to a timestamped ``.bak`` sibling (so the
        data is recoverable) and a fresh empty log takes its place; the next
        dashboard refresh then shows the welcome / empty state. Pass body
        ``{"keep_backup": false}`` to delete the old log outright instead.
        """
        if not self._is_loopback(request):
            return web.json_response(
                {"error": "control endpoint is loopback-only"}, status=403)

        if self.usage_log is None:
            return web.json_response(
                {"error": "no usage_log configured; nothing to reset"},
                status=400)

        keep_backup = True
        if request.can_read_body:
            try:
                body = await request.json()
                keep_backup = bool((body or {}).get("keep_backup", True))
            except Exception:  # noqa: BLE001
                pass  # empty / non-JSON body → defaults

        log = self.usage_log
        if not log.exists() or log.stat().st_size == 0:
            return web.json_response(
                {"status": "already empty", "usage_log": str(log),
                 "lines_cleared": 0, "backup": None})

        try:
            lines = sum(1 for _ in log.open()) if log.exists() else 0
            backup: str | None = None
            if keep_backup:
                stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
                backup_path = log.with_name(f"{log.name}.{stamp}.bak")
                log.replace(backup_path)          # atomic rename
                backup = str(backup_path)
            else:
                log.unlink()
            log.touch()                            # fresh empty log for new writes
        except Exception as e:  # noqa: BLE001
            _log.exception("usage log reset failed")
            return web.json_response(
                {"error": f"reset failed: {e}"}, status=500)

        _log.info("usage log reset: %d line(s) cleared%s",
                  lines, f", backed up → {backup}" if backup else "")
        return web.json_response({
            "status": "reset",
            "usage_log": str(log),
            "lines_cleared": lines,
            "backup": backup,
        })

    # ------------------------------------------------------------------
    # GET /__telos/developer -- developer-facing live session structure / tool-call statistics
    # ------------------------------------------------------------------

    async def handle_developer(self, request: web.Request) -> web.Response:
        """Render in real time the IR structure, bp regions, and tool calls of all sessions currently in memory.

        If the query carries ``?session=<id>``, render only the details of that one
        session; otherwise render the overview (session list) + the session details of the
        most recent call.
        """
        from telos.scripts.build_developer_page import render_developer

        try:
            body = render_developer(
                self._inspector,
                self._registry,
                focus_session=request.query.get("session"),
                refresh_seconds=self.dashboard_refresh,
                tab=request.query.get("tab", "overview"),
            )
        except Exception as e:  # noqa: BLE001
            _log.exception("developer page render failed")
            return web.Response(
                text=f"<pre>developer page render failed: {e}</pre>",
                content_type="text/html", status=500,
            )
        return web.Response(text=body, content_type="text/html")

    # ------------------------------------------------------------------
    # GET /__telos/developer.json -- the same data, machine-readable
    # ------------------------------------------------------------------

    async def handle_developer_json(self, request: web.Request) -> web.Response:
        """JSON view: for scripts / third-party tools to read the session state."""
        sid = request.query.get("session")
        if sid:
            entry = self._inspector.get(sid)
            if entry is None:
                return web.json_response(
                    {"error": "unknown session", "session_id": sid}, status=404)
            return web.json_response(_inspector_entry_to_json(entry))

        return web.json_response({
            "session_count": len(self._inspector),
            "sessions": [
                {
                    "session_id": sid,
                    "last_seen": e.last_seen,
                    "calls": len(e.calls),
                    "model": e.last_model,
                    "harness": e.last_harness,
                    "tools_seen": sorted(e.tools_stat.keys()),
                }
                for sid, e in self._inspector.items()
            ],
        })

    # ------------------------------------------------------------------
    # Internal: push state to the inspector on each response received
    # ------------------------------------------------------------------

    def _update_inspector(
        self,
        session_id: str,
        result: "PipelineResult",  # noqa: F821 — cross-module forward reference
        usage: Mapping[str, Any],
        call_index: int,
        latency_s: float,
    ) -> None:
        try:
            entry = self._inspector.touch(session_id)
            entry.record(
                call_index=call_index,
                layout=dict(result.ir_layout),
                plan_slots=list(result.plan_slots),
                tool_uses=list(result.tool_uses),
                tool_results=list(result.tool_results),
                raw_messages=list(result.raw_messages),
                usage_norm=_normalize_usage(dict(usage)),
                usage_raw=dict(usage),
                latency_s=latency_s,
                model=result.model,
                harness=result.harness,
            )
        except Exception:  # noqa: BLE001
            _log.exception("inspector record failed (call=%d)", call_index)

    # ------------------------------------------------------------------
    # Transparent passthrough: all paths other than /v1/messages
    # ------------------------------------------------------------------

    async def handle_passthrough(self, request: web.Request) -> web.StreamResponse:
        url = f"{self.upstream}{request.rel_url}"
        headers = self._forward_headers(request)
        body = await request.read()
        session = await self._session_get()

        try:
            upstream = await session.request(
                request.method, url, headers=headers, data=body,
            )
        except aiohttp.ClientError as e:
            return _anthropic_error(502, "api_error", f"Upstream error: {e}")

        try:
            body_bytes = await upstream.read()
            status = upstream.status
            ct = upstream.headers.get("content-type", "application/octet-stream")
        finally:
            upstream.release()

        return web.Response(body=body_bytes, status=status, headers={"Content-Type": ct})

    # ------------------------------------------------------------------
    # /_h/{harness}/{tail:.*}  -- URL-tagged route (attribution at injection time)
    # ------------------------------------------------------------------

    async def handle_tagged_route(self, request: web.Request) -> web.StreamResponse:
        """Strip the ``/_h/<harness>/`` URL tag, stamp it on the request, dispatch.

        Installers prepend ``/_h/<harness>/`` to the agent's base_url so the
        proxy gets the calling agent's identity directly from the URL rather
        than guessing via headers/content/sticky-memory. This handler peels
        the prefix and forwards the underlying request to the normal handler
        (``handle_messages`` for ``v1/messages``, ``handle_upstream_route``
        for ``upstreams/<slug>/<...>``).

        Downstream handlers read ``request["_telos_harness_tag"]`` and use it
        as the authoritative ``result.harness`` (overriding all other detection).
        Requests without this prefix flow through the legacy detection chain
        for backward compatibility with un-tagged installs.
        """
        tag = request.match_info["harness"]
        tail = request.match_info["tail"].lstrip("/")

        if tag not in _VALID_HARNESS_TAGS:
            return _anthropic_error(
                404, "not_found",
                f"unknown harness tag: {tag!r}. Known: {sorted(_VALID_HARNESS_TAGS)}",
            )
        request["_telos_harness_tag"] = tag

        # Dispatch by tail. The set is intentionally narrow: tagged URLs are
        # generated by installers, never typed by humans, so we only honor the
        # paths installers actually produce.
        if tail == "v1/messages" and request.method == "POST":
            return await self.handle_messages(request)
        if tail.startswith("upstreams/"):
            rest = tail[len("upstreams/"):]
            slug, _, sub_tail = rest.partition("/")
            if not slug:
                return _anthropic_error(
                    404, "not_found", "missing slug in tagged route")
            return await self.handle_upstream_route(
                request, slug=slug, tail=sub_tail,
            )
        return _anthropic_error(
            404, "not_found",
            f"tagged route does not support tail: {tail!r}",
        )

    # ------------------------------------------------------------------
    # POST /upstreams/{slug}/{tail:.*}  -- multi-backend route
    # ------------------------------------------------------------------

    async def handle_upstream_route(
        self,
        request: web.Request,
        *,
        slug: str | None = None,
        tail: str | None = None,
    ) -> web.StreamResponse:
        """Dispatch ``/upstreams/<slug>/<...>`` requests.

        - ``<slug>`` must be registered in ``self.upstreams``; otherwise 404.
        - If the upstream is an ``openai-chat`` protocol and the tail is
          ``v1/chat/completions``, run the TELOS OpenAI pipeline and forward.
        - If the upstream is an ``anthropic-messages`` protocol and the tail
          is ``v1/messages``, run the existing anthropic pipeline (via the
          slug's upstream rather than ``self.upstream``).
        - Anything else under the same slug is passthrough to ``<url>/<tail>``
          (so users can hit ``/v1/models``, ``/v1/embeddings``, etc.).

        ``slug`` / ``tail`` may be passed explicitly when re-dispatched from
        the ``/_h/<harness>/upstreams/...`` tag-route handler; otherwise they
        come from the router's match_info.
        """
        if slug is None:
            slug = request.match_info["slug"]
        if tail is None:
            tail = request.match_info["tail"]

        upstream_cfg = self.upstreams.get(slug)
        if upstream_cfg is None:
            return _anthropic_error(
                404, "not_found",
                f"unknown upstream slug: {slug!r}. "
                f"Known: {sorted(self.upstreams)}",
            )

        # Normalize tail (no leading slash).
        tail = tail.lstrip("/")
        _log.info("upstream_route: slug=%r tail=%r method=%s", slug, tail, request.method)

        if (upstream_cfg.protocol == "openai-chat"
                and tail.endswith("chat/completions")
                and request.method == "POST"):
            return await self._handle_openai_chat(
                request, slug, upstream_cfg, tail,
            )
        if (upstream_cfg.protocol == "openai-chat"
                and tail.endswith("responses")
                and request.method == "POST"):
            # Codex's wire_api="responses" path. Treat as a TELOS-blind
            # passthrough (the Responses payload shape isn't ChatCompletions),
            # but parse the terminal SSE event for usage so the dashboard sees
            # Codex traffic instead of a silent tunnel.
            return await self._handle_openai_responses(
                request, slug, upstream_cfg, tail,
            )
        if (upstream_cfg.protocol == "anthropic-messages"
                and tail.endswith("v1/messages")
                and request.method == "POST"):
            # The anthropic pipeline already runs through handle_messages;
            # temporarily swap self.upstream so the forward target is this
            # slug's url, then delegate. ``upstream.via`` (when set) also
            # overrides harness detection for the duration of the call so the
            # dashboard attributes traffic to the calling agent (e.g.
            # "openclaw") rather than the wire-level "hermes" / "openclaw"
            # auto-detect default.
            saved_upstream = self.upstream
            saved_harness = self.harness_override
            self.upstream = upstream_cfg.url.rstrip("/")
            if upstream_cfg.via:
                self.harness_override = upstream_cfg.via
            try:
                return await self.handle_messages(request)
            finally:
                self.upstream = saved_upstream
                self.harness_override = saved_harness
        # Default: transparent passthrough to <slug.url>/<tail>
        return await self._passthrough_to_upstream(
            request, upstream_cfg.url.rstrip("/"), tail,
        )

    async def _passthrough_to_upstream(
        self,
        request: web.Request,
        upstream_url: str,
        tail: str,
    ) -> web.StreamResponse:
        url = f"{upstream_url}/{tail}"
        if request.query_string:
            url = f"{url}?{request.query_string}"
        headers = self._forward_headers(request)
        body = await request.read()
        session = await self._session_get()
        try:
            upstream = await session.request(
                request.method, url, headers=headers, data=body,
            )
        except aiohttp.ClientError as e:
            return _anthropic_error(502, "api_error", f"Upstream error: {e}")

        status = upstream.status
        ct = upstream.headers.get("content-type", "application/octet-stream")

        # The OpenAI Responses API (used by Codex when ``wire_api = "responses"``)
        # returns SSE on success: buffering the whole body would either hang for
        # the lifetime of the stream or, worse, deliver the entire transcript as
        # one blob at the end. Detect SSE on the content-type and forward
        # chunk-by-chunk; otherwise buffer (cheap, and JSON tools expect a
        # single response body).
        if status == 200 and "text/event-stream" in ct.lower():
            return await self._stream_passthrough(request, upstream, ct)

        try:
            body_bytes = await upstream.read()
        finally:
            upstream.release()

        # Guard against relaying an HTML error page to a JSON client. When a
        # request hits a path the upstream answers with HTML (e.g. Codex's
        # login/bootstrap probing the bare base_url or ``/me`` while the
        # ``model_provider = telos`` override is active, which chatgpt.com's
        # codex backend answers with a 403 ``<!DOCTYPE html>`` page), the
        # client does ``JSON.parse(html)`` and dies with
        # ``Unexpected token '<', "<!DOCTYPE "... is not valid JSON``. Swap the
        # HTML for a parseable JSON error so the client surfaces something
        # actionable instead of crashing on the first byte.
        if status >= 400 and _looks_like_html(ct, body_bytes):
            return _openai_error(
                status, "invalid_request_error",
                f"Upstream returned an HTML page (status {status}) instead of "
                f"JSON for {request.method} {tail!r}. This is usually an "
                f"auth/login path that is not supported through the gateway — "
                f"re-login with the telos provider removed "
                f"(`telos uninstall --harness codex`, sign in, then "
                f"`telos init codex`).",
            )
        return web.Response(body=body_bytes, status=status,
                            headers={"Content-Type": ct})

    async def _stream_passthrough(
        self,
        request: web.Request,
        upstream: aiohttp.ClientResponse,
        content_type: str,
    ) -> web.StreamResponse:
        """Forward an SSE / chunked upstream response to the client without buffering.

        Mirrors the chunk-forwarding loop from ``_stream_response`` (the
        anthropic-messages path), but does not attempt any usage parsing — the
        passthrough is a transparent tunnel for endpoints TELOS does not
        understand (notably ``/v1/responses``).
        """
        downstream = web.StreamResponse(
            status=upstream.status,
            headers={
                "Content-Type": content_type,
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
        try:
            await downstream.prepare(request)
        except (ConnectionResetError, asyncio.CancelledError):
            upstream.release()
            return downstream

        try:
            async for chunk in upstream.content.iter_any():
                try:
                    await downstream.write(chunk)
                except (ConnectionResetError, asyncio.CancelledError):
                    break
        except aiohttp.ClientPayloadError:
            pass
        finally:
            upstream.release()
        try:
            await downstream.write_eof()
        except Exception:  # noqa: BLE001
            pass
        return downstream

    async def _handle_openai_chat(
        self,
        request: web.Request,
        slug: str,
        upstream_cfg: "UpstreamConfig",
        tail: str,
    ) -> web.StreamResponse:
        """``POST /upstreams/<slug>/<tail>`` (tail ends in ``chat/completions``) —
        TELOS-process, forward verbatim to ``<upstream.url>/<tail>``, log usage.

        ``tail`` is whatever path the client put after the slug prefix. This is
        passed through unchanged: if the client sent ``v1/chat/completions``,
        the forward goes to ``<url>/v1/chat/completions``; if it sent
        ``chat/completions``, the forward goes to ``<url>/chat/completions``.
        The slug's ``url`` should be the OpenAI-SDK-style ``base_url`` (with or
        without ``/v1``) that matches the path convention the client appends.
        """
        self._call_count += 1
        call_index = self._call_count

        try:
            raw = await request.json()
        except web.HTTPRequestEntityTooLarge as e:  # noqa: BLE001
            _log.warning("request body too large (call=%d): %s", call_index, e)
            return _anthropic_error(413, "request_too_large", str(e))
        except Exception as e:  # noqa: BLE001
            return _anthropic_error(400, "invalid_request_error",
                                    f"Invalid JSON: {e}")
        if not isinstance(raw, dict):
            return _anthropic_error(
                400, "invalid_request_error", "JSON body must be an object",
            )
        # Cache policy is gateway-owned. Strip client attempts even when the
        # request later runs in passthrough optimization mode.
        raw = {
            key: value for key, value in raw.items()
            if key not in _RESERVED_CACHE_FIELDS
        }

        session_id = (
            request.headers.get("x-telos-session")
            or _derive_session_id(raw, request.headers)
        )
        session_state = self._registry.get_or_create(session_id)

        # Corpus recording (raw, pre-rewrite).
        if self._corpus_dir is not None:
            try:
                record_call(self._corpus_dir, session_id, call_index, raw)
            except Exception:  # noqa: BLE001
                _log.exception("corpus record failed (call=%d)", call_index)

        mode = self._resolve_mode(request, session_state)
        compare_group = self._resolve_compare_group(request, session_state)

        # ---- TELOS pipeline (mode.telos only; rtk is anthropic-shaped, skip) ----
        if mode.telos:
            try:
                result = process_openai_request(
                    raw,
                    session_id=session_id,
                    session_state=session_state,
                    engine_name=upstream_cfg.engine,
                    cache_mode=upstream_cfg.cache.mode,
                    cache_backend=upstream_cfg.cache.backend,
                    cache_tier_telemetry=upstream_cfg.cache.tier_telemetry,
                )
            except Exception as e:  # noqa: BLE001
                self._pipeline_failures += 1
                if self._pipeline_failures == 1:
                    _log.exception("TELOS openai pipeline failed (call=%d) — "
                                   "falling back to passthrough", call_index)
                else:
                    _log.warning("TELOS openai pipeline failed (call=%d): %s",
                                 call_index, e)
                if self.strict:
                    return _anthropic_error(500, "api_error",
                                            f"TELOS pipeline failed: {e}")
                result = PipelineResult(
                    wire=dict(raw),
                    harness="passthrough",
                    plan_slots=[],
                    routing_key=None,
                    model=raw.get("model", ""),
                    engine=upstream_cfg.engine,
                    wire_protocol="openai-chat",
                    wire_emitter="passthrough",
                    cache_mode=upstream_cfg.cache.mode,
                    cache_backend=upstream_cfg.cache.backend,
                )
        else:
            result = PipelineResult(
                wire=dict(raw),
                harness="passthrough",
                plan_slots=[],
                routing_key=None,
                model=raw.get("model", ""),
                engine=upstream_cfg.engine,
                wire_protocol="openai-chat",
                wire_emitter="passthrough",
                cache_mode=upstream_cfg.cache.mode,
                cache_backend=upstream_cfg.cache.backend,
            )

        # Attribution: label the usage-log entry with the calling agent's identity.
        # Priority: URL tag (``/_h/<harness>/`` stamped by handle_tagged_route) >
        #           slug's via field (set at install time) >
        #           HTTP header detection (covers un-tagged un-via'd installs).
        # The `!= "passthrough"` guard is intentionally dropped: the via field
        # identifies the *caller*, not the optimization mode — even passthrough or
        # pipeline-failure entries should be attributed to the right agent so the
        # dashboard's "breakdown by harness" stays accurate. This also aligns with
        # the anthropic-messages path in handle_upstream_route, which always applies via.
        tag = request.get("_telos_harness_tag")
        if tag:
            result.harness = tag
        elif upstream_cfg.via:
            result.harness = upstream_cfg.via
        else:
            _client = _client_identity(request.headers)
            _detected = _detect_harness_signal(raw, request.headers)
            if _detected is not None:
                result.harness = _detected
                if self._client_harness.get(_client) != _detected:
                    self._client_harness[_client] = _detected
                    _log.info("harness pinned (openai-chat): client=%s -> %s (User-Agent=%r)",
                              _client or "(anon)", _detected,
                              request.headers.get("user-agent", ""))
            elif (remembered := self._client_harness.get(_client)):
                result.harness = remembered

        result.mode = mode.label
        result.compare_group = compare_group
        # raw_messages summary: OpenAI shape is flat strings most of the time,
        # so just record role + content length, mirroring the anthropic-side summary.
        result.raw_messages = _summarize_openai_messages(raw)

        is_streaming = bool(raw.get("stream", False))
        url = f"{upstream_cfg.url.rstrip('/')}/{tail}"
        if request.query_string:
            url = f"{url}?{request.query_string}"
        headers = self._forward_headers(request)
        body_bytes = json.dumps(result.wire).encode("utf-8")
        # Force JSON content-type; some clients send chunked.
        headers["content-type"] = "application/json"

        session = await self._session_get()
        t0 = time.time()
        try:
            upstream = await self._post_upstream(
                session, url, body_bytes, headers, call_index)
        except aiohttp.ClientError as e:
            _log.error("Upstream connection failed (call=%d): %s",
                       call_index, e)
            return _anthropic_error(502, "api_error", f"Upstream error: {e}")

        if is_streaming:
            return await self._stream_openai_response(
                request, upstream, session_id, result, session_state,
                call_index, t0,
            )
        return await self._buffered_openai_response(
            upstream, session_id, result, session_state, call_index, t0,
        )

    async def _buffered_openai_response(
        self,
        upstream: aiohttp.ClientResponse,
        session_id: str,
        result: PipelineResult,
        session_state: BridgeSessionState,
        call_index: int,
        t0: float,
    ) -> web.Response:
        """Non-streaming chat completions response: read body, extract usage, log."""
        try:
            body = await upstream.read()
            status = upstream.status
            ct = upstream.headers.get("content-type", "application/json")
        finally:
            upstream.release()

        usage: dict[str, Any] = {}
        cache_telemetry: Mapping[str, Any] = {}
        if status == 200:
            try:
                parsed = json.loads(body.decode("utf-8"))
                usage = parsed.get("usage") or {}
                if isinstance(parsed, Mapping):
                    cache_telemetry = load_engine(
                        result.engine,
                    ).parse_cache_telemetry(parsed)
            except Exception:  # noqa: BLE001
                pass
        elif status >= 400:
            _log.warning("upstream %d (call=%d): %s", status, call_index,
                         body[:2000].decode("utf-8", "replace"))

        latency_s = time.time() - t0
        self._log_openai_usage(
            session_id, result, usage, session_state,
            latency_s=latency_s,
            streaming=False,
            status=status,
            call_index=call_index,
            cache_telemetry=cache_telemetry,
        )
        # ``_update_inspector`` reads usage via the anthropic normalizer; for
        # OpenAI traffic the dashboard's openai-side metrics are owned by
        # ``_log_openai_usage``, so we skip inspector here in Phase 1.
        return web.Response(body=body, status=status,
                            headers={"Content-Type": ct})

    async def _stream_openai_response(
        self,
        request: web.Request,
        upstream: aiohttp.ClientResponse,
        session_id: str,
        result: PipelineResult,
        session_state: BridgeSessionState,
        call_index: int,
        t0: float,
    ) -> web.StreamResponse:
        """OpenAI ChatCompletions SSE: byte-forward chunks; extract usage from
        the trailing chunk if ``stream_options.include_usage`` was set.
        """
        status = upstream.status
        if status != 200:
            try:
                body = await upstream.read()
                ct = upstream.headers.get("content-type", "application/json")
            finally:
                upstream.release()
            if status >= 400:
                _log.warning("upstream %d (call=%d): %s", status, call_index,
                             body[:2000].decode("utf-8", "replace"))
            self._log_openai_usage(
                session_id, result, {}, session_state,
                latency_s=time.time() - t0,
                streaming=True,
                status=status,
                call_index=call_index,
            )
            return web.Response(body=body, status=status,
                                headers={"Content-Type": ct})

        downstream = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": upstream.headers.get(
                    "content-type", "text/event-stream"),
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
        try:
            await downstream.prepare(request)
        except (ConnectionResetError, asyncio.CancelledError) as e:
            _log.info("downstream disconnected before stream start (call=%d): %s",
                      call_index, e)
            upstream.release()
            return downstream

        usage_aggregate: dict[str, Any] = {}
        cache_details: dict[str, Any] = {}
        sse_buf = b""
        try:
            async for chunk in upstream.content.iter_any():
                try:
                    await downstream.write(chunk)
                except (ConnectionResetError, asyncio.CancelledError):
                    _log.info("downstream disconnected mid-stream (call=%d)",
                              call_index)
                    break
                sse_buf += chunk
                while b"\n\n" in sse_buf:
                    block, sse_buf = sse_buf.split(b"\n\n", 1)
                    self._peek_openai_sse_block(
                        block, usage_aggregate, cache_details,
                    )
        except aiohttp.ClientPayloadError:
            _log.warning("upstream closed connection mid-stream (call=%d)",
                         call_index)

        try:
            await downstream.write_eof()
        except Exception:
            pass
        finally:
            upstream.release()

        latency_s = time.time() - t0
        cache_telemetry: Mapping[str, Any] = {}
        if cache_details:
            try:
                cache_telemetry = load_engine(
                    result.engine,
                ).parse_cache_telemetry({
                    "sglext": {"cached_tokens_details": cache_details},
                })
            except Exception:  # noqa: BLE001
                pass
        self._log_openai_usage(
            session_id, result, usage_aggregate, session_state,
            latency_s=latency_s,
            streaming=True,
            status=200,
            call_index=call_index,
            cache_telemetry=cache_telemetry,
        )
        return downstream

    def _peek_openai_sse_block(
        self,
        block: bytes,
        usage: dict[str, Any],
        cache_details: dict[str, Any] | None = None,
    ) -> None:
        """OpenAI SSE chunk format: ``data: {...}\\n\\n``. The terminal chunk
        when ``stream_options.include_usage=true`` carries the full ``usage``.
        Silently swallow errors — proxying must never break on bad chunks.
        """
        for line in block.split(b"\n"):
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]" or not payload:
                continue
            try:
                data = json.loads(payload.decode("utf-8"))
            except Exception:  # noqa: BLE001
                continue
            u = data.get("usage")
            if isinstance(u, dict):
                usage.update(u)
            extension = data.get("sglext")
            if cache_details is not None and isinstance(extension, Mapping):
                details = extension.get("cached_tokens_details")
                if isinstance(details, Mapping):
                    cache_details.update(details)

    def _log_openai_usage(
        self,
        session_id: str,
        result: PipelineResult,
        usage: Mapping[str, Any],
        session_state: BridgeSessionState,
        *,
        latency_s: float,
        streaming: bool,
        status: int,
        call_index: int,
        cache_telemetry: Mapping[str, Any] | None = None,
    ) -> None:
        """Append one OpenAI-side usage line to the usage log (dashboard input)."""
        if self.usage_log is None:
            return
        try:
            normalized = _normalize_openai_usage(usage)
            prompt_tokens = normalized["raw_input"] + normalized["cache_read"]
            cache_runtime = {
                "engine": result.engine,
                "mode": result.cache_mode,
                "backend": result.cache_backend,
                "wire_emitter": result.wire_emitter,
                "prompt_tokens": prompt_tokens,
                "cached_tokens": normalized["cache_read"],
                "computed_prefill_tokens": normalized["raw_input"],
                "hit_ratio": (
                    round(normalized["cache_read"] / prompt_tokens, 6)
                    if prompt_tokens else 0.0
                ),
            }
            if cache_telemetry:
                tier_tokens = cache_telemetry.get("hit_tier_tokens")
                if isinstance(tier_tokens, Mapping):
                    normalized_tiers = {
                        name: int(tier_tokens.get(name, 0) or 0)
                        for name in ("gpu", "cpu", "l3")
                    }
                    cache_runtime["hit_tier_tokens"] = normalized_tiers
                    cache_runtime["restored_tokens"] = (
                        normalized_tiers["cpu"] + normalized_tiers["l3"]
                    )
                cache_runtime["storage_backend"] = cache_telemetry.get(
                    "storage_backend",
                )
            with self.usage_log.open("a") as f:
                f.write(json.dumps({
                    "ts": time.time(),
                    "session_id": session_id,
                    "call_index": call_index,
                    "model": result.model,
                    "harness": result.harness,
                    "mode": result.mode,
                    "compare_group": result.compare_group,
                    "tool_output_reduction": {},
                    "n_slots": len(result.plan_slots),
                    "slots": result.plan_slots,
                    "latency_s": round(latency_s, 3),
                    "streaming": streaming,
                    "status": status,
                    "raw_usage": dict(usage),
                    "raw_cache_telemetry": dict(cache_telemetry or {}),
                    "normalized": normalized,
                    "cache_runtime": cache_runtime,
                    "cumulative": {
                        "cache_creation":
                            session_state.stats.cumulative_cache_creation,
                        "real_requests_since_refresh":
                            session_state.stats.real_requests_since_refresh,
                        "refpool_slugs": sorted(session_state.refpool.slugs),
                    },
                }, ensure_ascii=False) + "\n")
        except Exception:  # noqa: BLE001
            _log.exception("usage log write failed")

    # ------------------------------------------------------------------
    # POST .../<tail ending in "responses"> — OpenAI Responses API
    # (used by Codex with wire_api="responses"). No TELOS rewriting; we
    # passthrough + parse usage from the terminal SSE event so codex traffic
    # is visible on the dashboard.
    # ------------------------------------------------------------------

    async def _handle_openai_responses(
        self,
        request: web.Request,
        slug: str,
        upstream_cfg: "UpstreamConfig",
        tail: str,
    ) -> web.StreamResponse:
        self._call_count += 1
        call_index = self._call_count

        try:
            raw = await request.json()
        except web.HTTPRequestEntityTooLarge as e:  # noqa: BLE001
            _log.warning("request body too large (call=%d): %s", call_index, e)
            return _anthropic_error(413, "request_too_large", str(e))
        except Exception as e:  # noqa: BLE001
            return _anthropic_error(400, "invalid_request_error",
                                    f"Invalid JSON: {e}")

        session_id = (
            request.headers.get("x-telos-session")
            or _derive_session_id(raw, request.headers)
        )
        session_state = self._registry.get_or_create(session_id)

        if self._corpus_dir is not None:
            try:
                record_call(self._corpus_dir, session_id, call_index, raw)
            except Exception:  # noqa: BLE001
                _log.exception("corpus record failed (call=%d)", call_index)

        mode = self._resolve_mode(request, session_state)
        compare_group = self._resolve_compare_group(request, session_state)

        # No TELOS pipeline: the Responses payload shape (``input`` array)
        # is structurally different from ChatCompletions, and rewriting it
        # would require a separate engine adapter. For now we just observe.
        result = PipelineResult(
            wire=dict(raw),
            harness="passthrough",
            plan_slots=[],
            routing_key=None,
            model=raw.get("model", ""),
        )
        # Tag wins; else via (install time); else header / per-client memory.
        tag = request.get("_telos_harness_tag")
        if tag:
            result.harness = tag
        elif upstream_cfg.via:
            result.harness = upstream_cfg.via
        else:
            _client = _client_identity(request.headers)
            _detected = _detect_harness_signal(raw, request.headers)
            if _detected is not None:
                result.harness = _detected
                self._client_harness[_client] = _detected
            elif (remembered := self._client_harness.get(_client)):
                result.harness = remembered
        result.mode = mode.label
        result.compare_group = compare_group

        is_streaming = bool(raw.get("stream", False))
        url = f"{upstream_cfg.url.rstrip('/')}/{tail}"
        if request.query_string:
            url = f"{url}?{request.query_string}"
        headers = self._forward_headers(request)
        body_bytes = json.dumps(result.wire).encode("utf-8")
        headers["content-type"] = "application/json"

        session = await self._session_get()
        t0 = time.time()
        try:
            upstream = await self._post_upstream(
                session, url, body_bytes, headers, call_index)
        except aiohttp.ClientError as e:
            _log.error("Upstream connection failed (call=%d): %s",
                       call_index, e)
            return _anthropic_error(502, "api_error", f"Upstream error: {e}")

        if is_streaming:
            return await self._stream_responses_response(
                request, upstream, session_id, result, session_state,
                call_index, t0,
            )
        return await self._buffered_responses_response(
            upstream, session_id, result, session_state, call_index, t0,
        )

    async def _buffered_responses_response(
        self,
        upstream: aiohttp.ClientResponse,
        session_id: str,
        result: PipelineResult,
        session_state: BridgeSessionState,
        call_index: int,
        t0: float,
    ) -> web.Response:
        try:
            body = await upstream.read()
            status = upstream.status
            ct = upstream.headers.get("content-type", "application/json")
        finally:
            upstream.release()

        usage: dict[str, Any] = {}
        if status == 200:
            try:
                parsed = json.loads(body.decode("utf-8"))
                # Non-streamed Responses puts usage on the top-level object.
                u = parsed.get("usage")
                if isinstance(u, dict):
                    usage = u
            except Exception:  # noqa: BLE001
                pass
        elif status >= 400:
            _log.warning("upstream %d (call=%d): %s", status, call_index,
                         body[:2000].decode("utf-8", "replace"))

        latency_s = time.time() - t0
        self._log_responses_usage(
            session_id, result, usage, session_state,
            latency_s=latency_s, streaming=False, status=status,
            call_index=call_index,
        )
        return web.Response(body=body, status=status,
                            headers={"Content-Type": ct})

    async def _stream_responses_response(
        self,
        request: web.Request,
        upstream: aiohttp.ClientResponse,
        session_id: str,
        result: PipelineResult,
        session_state: BridgeSessionState,
        call_index: int,
        t0: float,
    ) -> web.StreamResponse:
        status = upstream.status
        if status != 200:
            try:
                body = await upstream.read()
                ct = upstream.headers.get("content-type", "application/json")
            finally:
                upstream.release()
            if status >= 400:
                _log.warning("upstream %d (call=%d): %s", status, call_index,
                             body[:2000].decode("utf-8", "replace"))
            self._log_responses_usage(
                session_id, result, {}, session_state,
                latency_s=time.time() - t0, streaming=True, status=status,
                call_index=call_index,
            )
            return web.Response(body=body, status=status,
                                headers={"Content-Type": ct})

        downstream = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": upstream.headers.get(
                    "content-type", "text/event-stream"),
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
        try:
            await downstream.prepare(request)
        except (ConnectionResetError, asyncio.CancelledError) as e:
            _log.info("downstream disconnected before stream start (call=%d): %s",
                      call_index, e)
            upstream.release()
            return downstream

        usage_aggregate: dict[str, Any] = {}
        sse_buf = b""
        try:
            async for chunk in upstream.content.iter_any():
                try:
                    await downstream.write(chunk)
                except (ConnectionResetError, asyncio.CancelledError):
                    _log.info("downstream disconnected mid-stream (call=%d)",
                              call_index)
                    break
                sse_buf += chunk
                while b"\n\n" in sse_buf:
                    block, sse_buf = sse_buf.split(b"\n\n", 1)
                    self._peek_responses_sse_block(block, usage_aggregate)
        except aiohttp.ClientPayloadError:
            _log.warning("upstream closed connection mid-stream (call=%d)",
                         call_index)

        try:
            await downstream.write_eof()
        except Exception:
            pass
        finally:
            upstream.release()

        latency_s = time.time() - t0
        self._log_responses_usage(
            session_id, result, usage_aggregate, session_state,
            latency_s=latency_s, streaming=True, status=200,
            call_index=call_index,
        )
        return downstream

    def _peek_responses_sse_block(
        self, block: bytes, usage: dict[str, Any],
    ) -> None:
        """Responses API SSE: usage is carried on the ``response.completed``
        event under ``data.response.usage``. Silently swallow errors.
        """
        event: str | None = None
        data_raw: bytes | None = None
        for line in block.split(b"\n"):
            if line.startswith(b"event:"):
                event = line[6:].strip().decode("ascii", "ignore")
            elif line.startswith(b"data:"):
                data_raw = line[5:].strip()
        if event != "response.completed" or data_raw is None:
            return
        if data_raw == b"[DONE]" or not data_raw:
            return
        try:
            data = json.loads(data_raw.decode("utf-8"))
        except Exception:  # noqa: BLE001
            return
        resp = data.get("response")
        if not isinstance(resp, dict):
            return
        u = resp.get("usage")
        if isinstance(u, dict):
            usage.update(u)

    def _log_responses_usage(
        self,
        session_id: str,
        result: PipelineResult,
        usage: Mapping[str, Any],
        session_state: BridgeSessionState,
        *,
        latency_s: float,
        streaming: bool,
        status: int,
        call_index: int,
    ) -> None:
        """Append one Responses-API usage line to usage_log.

        Same record shape as ``_log_openai_usage`` so the dashboard's existing
        aggregator picks it up; only ``normalized`` uses the Responses-side
        field names (``input_tokens`` + ``input_tokens_details.cached_tokens``).
        """
        if self.usage_log is None:
            return
        try:
            with self.usage_log.open("a") as f:
                f.write(json.dumps({
                    "ts": time.time(),
                    "session_id": session_id,
                    "call_index": call_index,
                    "model": result.model,
                    "harness": result.harness,
                    "mode": result.mode,
                    "compare_group": result.compare_group,
                    "tool_output_reduction": {},
                    "n_slots": len(result.plan_slots),
                    "slots": result.plan_slots,
                    "latency_s": round(latency_s, 3),
                    "streaming": streaming,
                    "status": status,
                    "raw_usage": dict(usage),
                    "normalized": _normalize_responses_usage(usage),
                    "cumulative": {
                        "cache_creation":
                            session_state.stats.cumulative_cache_creation,
                        "real_requests_since_refresh":
                            session_state.stats.real_requests_since_refresh,
                        "refpool_slugs": sorted(session_state.refpool.slugs),
                    },
                }, ensure_ascii=False) + "\n")
        except Exception:  # noqa: BLE001
            _log.exception("usage log write failed")

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _accumulate_into_state(
        self, session_state: BridgeSessionState, usage: Mapping[str, Any],
    ) -> None:
        """Accumulate the cache_creation tokens of the upstream response into session_state.

        This is the equivalent of TELOS's ``bridge.absorb_usage`` on the proxy path.
        Without this step, the R8 trigger condition can never reach the
        ``cumulative_cache_creation`` threshold.
        """
        cache_write = int(usage.get("cache_creation_input_tokens", 0) or 0)
        if cache_write:
            session_state.stats.cumulative_cache_creation += cache_write

    def _log_usage(
        self,
        session_id: str,
        result: PipelineResult,
        usage: dict[str, Any],
        session_state: BridgeSessionState,
        *,
        latency_s: float,
        streaming: bool,
        status: int,
        call_index: int,
    ) -> None:
        if self.usage_log is None:
            return
        try:
            with self.usage_log.open("a") as f:
                f.write(json.dumps({
                    "ts": time.time(),
                    "session_id": session_id,
                    "call_index": call_index,
                    "model": result.model,
                    "harness": result.harness,
                    "mode": result.mode,
                    "compare_group": result.compare_group,
                    "tool_output_reduction": result.tool_output_reduction,
                    "n_slots": len(result.plan_slots),
                    "slots": result.plan_slots,
                    "latency_s": round(latency_s, 3),
                    "streaming": streaming,
                    "status": status,
                    "raw_usage": usage,
                    "normalized": _normalize_usage(usage),
                    "cumulative": {
                        "cache_creation": session_state.stats.cumulative_cache_creation,
                        "real_requests_since_refresh":
                            session_state.stats.real_requests_since_refresh,
                        "refpool_slugs": sorted(session_state.refpool.slugs),
                    },
                }, ensure_ascii=False) + "\n")
        except Exception:  # noqa: BLE001
            _log.exception("usage log write failed")


# ---------------------------------------------------------------------------
# Application construction
# ---------------------------------------------------------------------------

@web.middleware
async def _access_log_middleware(request: web.Request, handler):  # type: ignore[type-arg]
    ua = request.headers.get("User-Agent", "")
    _log.info("ACCESS %s %s  ua=%r", request.method, request.path_qs, ua)
    return await handler(request)


# aiohttp >= 3.9 deprecates bare-string application keys (NotAppKeyWarning).
# Use a typed AppKey so the stored ProxyApp handle stays accessible without the
# warning. ProxyApp is defined above; the key is module-level for reuse.
PROXY_APP_KEY = web.AppKey("proxy", ProxyApp)


def make_app(
    *,
    upstream: str = _DEFAULT_UPSTREAM,
    upstreams: Mapping[str, "UpstreamConfig"] | None = None,
    usage_log: Path | None = None,
    harness_override: str | None = None,
    strict: bool = False,
    dashboard_refresh: int = 5,
    mode: TelosMode | None = None,
    corpus_dir: Path | None = None,
    record: bool = True,
) -> web.Application:
    """Construct a complete aiohttp application. Reusable for tests / ASGI embedding."""
    proxy = ProxyApp(
        upstream=upstream,
        upstreams=upstreams,
        usage_log=usage_log,
        harness_override=harness_override,
        strict=strict,
        dashboard_refresh=dashboard_refresh,
        mode=mode,
        corpus_dir=corpus_dir,
        record=record,
    )
    # client_max_size: aiohttp defaults to only 1 MiB. Harnesses like Claude Code, in a
    # long conversation, produce a single request body (full messages history + system +
    # tools) that far exceeds 1 MiB, triggering HTTPRequestEntityTooLarge → wrapped by
    # handle_messages' except into "400 Invalid JSON: Request Entity Too Large". Setting it
    # to 1 GiB is effectively unlimited.
    app = web.Application(client_max_size=1024 ** 3, middlewares=[_access_log_middleware])
    # Tagged route: /_h/<harness>/<tail>. Strips the tag and dispatches the
    # underlying request. Registered first so it takes priority over the bare
    # /v1/messages and /upstreams routes.
    app.router.add_route(
        "*", "/_h/{harness}/{tail:.*}", proxy.handle_tagged_route,
    )
    app.router.add_post("/v1/messages", proxy.handle_messages)
    # Multi-backend route: /upstreams/<slug>/<tail>. Registered before the
    # catch-all passthrough so the slug dispatch wins.
    app.router.add_route(
        "*", "/upstreams/{slug}/{tail:.*}", proxy.handle_upstream_route,
    )
    # Must be registered before the catch-all passthrough, otherwise it gets swallowed.
    app.router.add_get("/__telos/dashboard", proxy.handle_dashboard)
    app.router.add_route("GET", "/__telos/control/mode", proxy.handle_control_mode)
    app.router.add_route("POST", "/__telos/control/mode", proxy.handle_control_mode)
    app.router.add_post("/__telos/control/reset", proxy.handle_control_reset)
    app.router.add_get("/__telos/developer", proxy.handle_developer)
    app.router.add_get("/__telos/developer.json", proxy.handle_developer_json)
    app.router.add_route("*", "/{tail:.*}", proxy.handle_passthrough)
    app.on_shutdown.append(proxy.on_shutdown)
    app[PROXY_APP_KEY] = proxy
    return app


def run(
    *,
    host: str = "127.0.0.1",
    port: int = 7171,
    upstream: str = _DEFAULT_UPSTREAM,
    upstreams: Mapping[str, "UpstreamConfig"] | None = None,
    usage_log: Path | None = None,
    harness_override: str | None = None,
    strict: bool = False,
    dashboard_refresh: int = 5,
    mode: TelosMode | None = None,
    corpus_dir: Path | None = None,
    record: bool = True,
) -> None:
    """Blocking startup (for the CLI entry point)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    mode = mode or TelosMode()
    app = make_app(
        upstream=upstream,
        upstreams=upstreams,
        usage_log=usage_log,
        harness_override=harness_override,
        strict=strict,
        dashboard_refresh=dashboard_refresh,
        mode=mode,
        corpus_dir=corpus_dir,
        record=record,
    )
    _log.info("TELOS gateway listening on http://%s:%d → %s", host, port, upstream)
    _log.info("default mode  → %s (telos=%s rtk=%s); a single request can override with X-Telos-Mode",
              mode.label, mode.telos, mode.rtk)
    if record and corpus_dir is not None:
        _log.info("session corpus → %s (records raw requests for telos replay; --no-record to disable)",
                  corpus_dir)
    if usage_log:
        _log.info("usage log    → %s", usage_log)
        _log.info("dashboard    → http://%s:%d/__telos/dashboard"
                  " (refresh=%ds)", host, port, dashboard_refresh)
    else:
        _log.info("dashboard    → http://%s:%d/__telos/dashboard"
                  " (no usage_log; will show empty state)", host, port)
    _log.info("developer    → http://%s:%d/__telos/developer"
              " (live session inspector; JSON at /__telos/developer.json)",
              host, port)
    if strict:
        _log.info("strict mode ON — a TELOS failure returns 500 (no degradation to passthrough)")
    web.run_app(app, host=host, port=port, print=None, access_log=None)
