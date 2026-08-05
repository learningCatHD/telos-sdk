"""TELOS processing pipeline -- pure functions, producing wire requests from raw inputs.

Two protocol-specific entry points share the same parse → bridge → canonicalize
core and the same ``PipelineResult``:

- ``process_anthropic_request`` for ``/v1/messages``-shaped input
  (openclaw / hermes harnesses, anthropic engine).
- ``process_openai_request``    for ``/v1/chat/completions``-shaped input
  (telos harness parses OpenAI shape; engine is configurable, defaults to
  ``deepseek`` for OpenRouter-style upstreams).

This split mirrors the existing two transports (``TelosAnthropicTransport`` and
``TelosOpenAITransport`` in ``telos.scripts``); the proxy uses these so the
wire produced by an in-process SDK call and a request through the gateway are
identical.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from telos import Bridge, load_engine, load_harness
from telos.bridge import BridgeSessionState, _canonicalize_ir
from telos.engine.base import UnsupportedWireProtocolError
from telos.engine.openai_chat import render_chat_completions
from telos.ir import Band, TelosIR, assert_ir_invariants
from telos.scripts.telos_anthropic_transport import _detect_harness


# Fields passed through to the upstream that do not take part in the TELOS pipeline.
_PASSTHROUGH_FIELDS = (
    "max_tokens", "temperature", "top_p", "stream", "stop_sequences",
    "tool_choice", "thinking", "metadata", "service_tier", "top_k",
)

# Fields passed through on the OpenAI ChatCompletions path.
_OPENAI_PASSTHROUGH_FIELDS = (
    "temperature", "top_p", "max_tokens", "stream", "stop",
    "tool_choice", "response_format", "frequency_penalty", "presence_penalty",
    "seed", "stream_options", "logit_bias", "logprobs", "top_logprobs", "n",
    "user",
)

# Never trust client-supplied cache-control fields. Stock adapters do not emit
# these, and future extension fields must be derived from authenticated gateway
# state after capability negotiation.
_RESERVED_CACHE_FIELDS = frozenset({
    "telos_cache", "cache_control", "cache_policy", "cache_salt",
})


@dataclass
class PipelineResult:
    """The output of the TELOS pipeline.

    Attributes:
        wire:        a request body that can be sent directly to
                     ``api.anthropic.com/v1/messages``.
        harness:     the harness name actually used (auto-detected or explicitly passed).
        plan_slots:  the list of slot names of the ``EmitPlan`` (for diagnostics).
        routing_key: always ``None`` for Anthropic; the field is kept to align with the
                     generic schema.
        model:       the model field in the request (passed through, used by the dashboard
                     for cost computation).
        cumulative_cache_creation: cache_write tokens accumulated across turns (from
                                   session_state). 0 on the first call of a new session.
        real_requests_since_refresh: the real-request count since the last refresh.
        ir_layout:   a snapshot of the IR structure for the developer panel (segment × band
                     char counts / block counts, plus the role/kind/band list of each
                     message). The full IR does not enter the wire dict, to avoid log bloat.
        tool_uses:   the list of tool_use entries initiated by the assistant in this
                     request (name + the char length of the argument body), used for the
                     tool-call statistics of the developer panel.
        tool_results: the tool_result blocks in the user segment of this request
                     (tool_use_id + the content char length).
    """

    wire: dict[str, Any]
    harness: str
    plan_slots: list[str]
    routing_key: str | None
    model: str = ""
    engine: str = ""
    wire_protocol: str = ""
    wire_emitter: str = ""
    cache_mode: str = "stock"
    cache_backend: str = "auto"
    cumulative_cache_creation: int = 0
    real_requests_since_refresh: int = 0
    ir_layout: dict[str, Any] = field(default_factory=dict)
    tool_uses: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    # ↓ fields backfilled by the proxy layer after the pipeline runs (see proxy/server.py).
    # Comparison experiments need to slice the usage_log by (mode, compare_group), so they
    # are put into the result and persisted together. The pipeline itself does not set them.
    mode: str = "telos"
    compare_group: str | None = None
    tool_output_reduction: dict[str, Any] = field(default_factory=dict)
    # A summary of each message in the raw (pre-TELOS-rewrite) request, for display on the
    # developer page. Backfilled by the proxy layer in handle_messages; the pipeline itself
    # does not set it.
    raw_messages: list[dict[str, Any]] = field(default_factory=list)


def process_anthropic_request(
    raw: Mapping[str, Any],
    *,
    session_id: str,
    session_state: BridgeSessionState | None = None,
    harness_name: str | None = None,
    engine_name: str = "anthropic",
) -> PipelineResult:
    """Run the TELOS pipeline once, returning the processed wire request + diagnostic info.

    Args:
        raw:           the raw ``/v1/messages`` request body (dict).
        session_id:    the TELOS session identifier, used for the IR.session_id field
                       inside the Bridge.
        session_state: the cross-turn persisted Bridge state. **When passed, the ref-pool /
                       R8 counter accumulate across calls**; without it, each turn is
                       independent (behavior degrades to the early version).
        harness_name:  ``"openclaw"`` / ``"hermes"`` / ``None`` (auto-detect).
        engine_name:   defaults to ``"anthropic"``.

    Returns:
        ``PipelineResult``. Note that ``wire`` has already been through
        ``_canonicalize_ir`` (tools sorting, payload key sorting) and can be forwarded directly.
    """
    if harness_name:
        name = harness_name
    elif session_state is not None and session_state.sticky_harness:
        name = session_state.sticky_harness
    else:
        name = _detect_harness(raw)
        if session_state is not None:
            session_state.sticky_harness = name
    harness = load_harness(name)
    engine = load_engine(engine_name)

    ir = harness.parse(
        raw,
        session_id=session_id,
        engine=engine_name,
        model=raw.get("model", ""),
    )
    bridge = Bridge(ir, engine, session_state=session_state)

    # bridge.emit_with_plan() internally runs canonicalize → plan_marks → emit,
    # doing the cache_control marking and the tool canonical sorting together.
    wire_dict, plan = bridge.emit_with_plan()
    wire: dict[str, Any] = dict(wire_dict)

    # Pass through the caller's original non-TELOS fields
    for k in _PASSTHROUGH_FIELDS:
        if k in raw and raw[k] is not None:
            wire[k] = raw[k]

    state = bridge.session_state
    snapshot = bridge.snapshot_ir()
    layout = _summarize_ir_layout(snapshot)
    tool_uses, tool_results = _extract_tool_calls(snapshot)
    return PipelineResult(
        wire=wire,
        harness=name,
        plan_slots=[s.name for s in plan.slots],
        routing_key=plan.routing_key,
        model=raw.get("model", ""),
        engine=engine_name,
        wire_protocol="anthropic-messages",
        wire_emitter="adapter",
        cumulative_cache_creation=state.stats.cumulative_cache_creation,
        real_requests_since_refresh=state.stats.real_requests_since_refresh,
        ir_layout=layout,
        tool_uses=tool_uses,
        tool_results=tool_results,
    )


# ---------------------------------------------------------------------------
# OpenAI ChatCompletions pipeline
# ---------------------------------------------------------------------------

def process_openai_request(
    raw: Mapping[str, Any],
    *,
    session_id: str,
    session_state: BridgeSessionState | None = None,
    engine_name: str = "deepseek",
    cache_mode: str = "stock",
    cache_backend: str = "auto",
    cache_tier_telemetry: bool = False,
) -> "PipelineResult":
    """Run the TELOS pipeline once for an OpenAI ChatCompletions request.

    Args:
        raw:           the raw ``/v1/chat/completions`` request body (dict).
        session_id:    the TELOS session identifier (used inside the Bridge).
        session_state: cross-turn persisted Bridge state. **When passed, the
                       ref-pool / R8 counter accumulate across calls**.
        engine_name:   the engine adapter used by ``Bridge.mark`` (for routing
                       key derivation and ``parse_usage`` later). Defaults to
                       ``"deepseek"`` because OpenRouter / DeepSeek expose the
                       most informative cache fields; ``"openai"`` also works.

    Returns:
        ``PipelineResult`` whose ``wire`` is a ChatCompletions-shaped dict
        ready to POST to ``<upstream>/v1/chat/completions``. ``harness`` is
        always ``"telos"`` (this is what the dashboard expects for OpenAI
        traffic; the ``telos`` harness is the one that parses OpenAI shape).
    """
    harness = load_harness("telos")
    engine = load_engine(engine_name)
    sanitized_raw = {
        key: value for key, value in raw.items()
        if key not in _RESERVED_CACHE_FIELDS
    }
    model = str(sanitized_raw.get("model", ""))

    ir = harness.parse(
        sanitized_raw,
        session_id=session_id,
        engine=engine_name,
        model=model,
    )
    bridge = Bridge(ir, engine, session_state=session_state)

    ir2 = _canonicalize_ir(bridge.snapshot_ir())
    assert_ir_invariants(ir2)
    plan = engine.plan_marks(ir2)
    try:
        wire = dict(engine.emit_for_protocol(
            ir2, plan, protocol="openai-chat",
        ))
        wire_emitter = "adapter"
    except UnsupportedWireProtocolError:
        wire = render_chat_completions(ir2, model=model)
        wire_emitter = "generic"

    # Adapter-owned public telemetry fields are injected after rendering and
    # before caller sampling fields. Clients cannot spoof these fields because
    # they are not part of the passthrough allowlist.
    wire.update(engine.cache_telemetry_request_fields(
        tier_details=cache_tier_telemetry and cache_mode != "off",
    ))

    # Pass through the caller's non-TELOS fields (temperature, stream, etc.).
    for k in _OPENAI_PASSTHROUGH_FIELDS:
        if k in sanitized_raw and sanitized_raw[k] is not None:
            wire[k] = sanitized_raw[k]

    # Mirror Bridge.emit_with_plan: this protocol-specific path performs its
    # own canonicalize/plan/emit sequence, so update R8 visibility here.
    state = bridge.session_state
    state.stats.real_requests_since_refresh += 1

    snapshot = bridge.snapshot_ir()
    layout = _summarize_ir_layout(snapshot)
    tool_uses, tool_results = _extract_tool_calls(snapshot)
    return PipelineResult(
        wire=wire,
        harness="telos",
        plan_slots=[s.name for s in plan.slots],
        routing_key=plan.routing_key,
        model=model,
        engine=engine_name,
        wire_protocol="openai-chat",
        wire_emitter=wire_emitter,
        cache_mode=cache_mode,
        cache_backend=cache_backend,
        cumulative_cache_creation=state.stats.cumulative_cache_creation,
        real_requests_since_refresh=state.stats.real_requests_since_refresh,
        ir_layout=layout,
        tool_uses=tool_uses,
        tool_results=tool_results,
    )


# ---------------------------------------------------------------------------
# IR summary: region byte counts + per-message band sequence for the developer panel
# ---------------------------------------------------------------------------

_BANDS = ("pin", "fold", "drop")


def _payload_size(payload: Any) -> int:
    """Estimate the character volume of a payload (used for the "prompt regions" display).

    text uses len(); dict / list uses the char count of json serialization (roughly the
    same order of magnitude as the wire). Any exception degrades to ``len(str(payload))``,
    and never raises (the developer panel must always be renderable).
    """
    if isinstance(payload, str):
        return len(payload)
    try:
        import json as _json
        return len(_json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                default=str))
    except Exception:  # noqa: BLE001
        return len(str(payload))


def _summarize_ir_layout(ir: TelosIR) -> dict[str, Any]:
    """Return ``{segment: {pin/fold/drop: {blocks, chars}}, messages: [...]}``.

    - segment ∈ {tools, system, messages}
    - each message separately records (role, blocks: [(band, kind, chars, id)])
    so the developer panel can trace the increase/decrease of the fold region per message.
    """
    out: dict[str, Any] = {
        "session_id": ir.session_id,
        "engine": ir.hints.engine,
        "model": ir.hints.model,
        "segments": {seg: {b: {"blocks": 0, "chars": 0} for b in _BANDS}
                     for seg in ("tools", "system", "messages")},
        "messages": [],
        "ref_pool": [],
    }
    # tools
    for blk in ir.tools:
        s = out["segments"]["tools"][blk.band.value]
        s["blocks"] += 1
        s["chars"] += _payload_size(blk.payload)
    # system
    for blk in ir.system:
        s = out["segments"]["system"][blk.band.value]
        s["blocks"] += 1
        s["chars"] += _payload_size(blk.payload)
    # messages (also aggregated into the segments.messages.* buckets & per-message detail)
    for mi, msg in enumerate(ir.messages):
        detail = {"index": mi, "role": msg.role, "blocks": []}
        for blk in msg.blocks:
            chars = _payload_size(blk.payload)
            s = out["segments"]["messages"][blk.band.value]
            s["blocks"] += 1
            s["chars"] += chars
            detail["blocks"].append({
                "id": blk.id,
                "band": blk.band.value,
                "kind": blk.kind,
                "chars": chars,
                "source_tag": blk.source_tag,
                "ref_slug": blk.ref_slug,
            })
        out["messages"].append(detail)
    # ref-pool: list the slug + the current payload char count
    for slug, blk in ir.ref_pool.items():
        out["ref_pool"].append({
            "slug": slug,
            "band": blk.band.value,
            "chars": _payload_size(blk.payload),
        })
    return out


def _extract_tool_calls(ir: TelosIR) -> tuple[list[dict[str, Any]],
                                                list[dict[str, Any]]]:
    """Extract (tool_uses, tool_results) from the IR.

    tool_use comes from assistant messages; tool_result comes from user messages. Each
    record contains name / args_chars / result_chars / tool_use_id (if any), for
    SessionInspector to accumulate statistics.
    """
    uses: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for mi, msg in enumerate(ir.messages):
        for blk in msg.blocks:
            if blk.kind == "tool_use" and isinstance(blk.payload, Mapping):
                p = blk.payload
                args = p.get("input") or p.get("arguments") or {}
                uses.append({
                    "message_index": mi,
                    "id": p.get("id") or blk.id,
                    "name": p.get("name") or "?",
                    "args_chars": _payload_size(args),
                })
            elif blk.kind == "tool_result" and isinstance(blk.payload, Mapping):
                p = blk.payload
                results.append({
                    "message_index": mi,
                    "tool_use_id": p.get("tool_use_id", ""),
                    "result_chars": _payload_size(p.get("content", "")),
                })
    return uses, results
