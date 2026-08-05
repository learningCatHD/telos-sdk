"""TelosOpenAITransport: wires an OpenAI-shaped chat.completions client through telos.

mini_swe_runner (the telos-vendored hermes) calls:

    self.client.chat.completions.create(model=..., messages=[...], tools=[...])

This transport implements the same interface, but internally routes through the
``telos`` harness → TELOS Bridge → canonicalize / band-reorder → converts back to
the chat-completions shape → actually sends it to OpenRouter's
``/v1/chat/completions``. When the response comes back, usage is normalized with
the ``deepseek`` adapter (DeepSeek V3+ on OpenRouter has usage fields
``prompt_cache_hit_tokens / prompt_cache_miss_tokens``) and written to the jsonl log.

Design notes:

- **Do not break tool_calls structure**: role=assistant's ``tool_calls`` and
  role=tool's ``tool_call_id`` must be re-attached to the wire per the OpenAI
  protocol; they cannot be inlined directly into text the way the DeepSeek
  adapter does — otherwise the agent loop cannot get the tool results.
- **Apply a minimal subset of the TELOS policy**: the DROP band
  (``<environment_info>`` / ``Current time:`` etc.) sinks to the tail of each
  user message's text; tool definitions are canonicalized (key ordering); the
  rest follows the §5 order. These are the two rules most directly tied to cache-hit gains.
- **usage records both raw and normalized**: raw is kept for diagnostics, while
  normalized aligns directly with ``compute-metrics.py`` per the ``UsageReport`` schema.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from os.path import commonprefix
from pathlib import Path
from typing import Any, Mapping

from telos import Band, Bridge, load_engine, load_harness
from telos.bridge import BridgeSessionState, _canonicalize_ir
from telos.engine.base import UnsupportedWireProtocolError
from telos.engine.openai_chat import render_chat_completions


# ---------------------------------------------------------------------------
# IR -> OpenAI ChatCompletions wire (preserves tool_calls / role=tool structure)
# ---------------------------------------------------------------------------

def _ir_to_chat_completions(ir, *, model: str) -> dict[str, Any]:
    """Backward-compatible import path for the shared renderer."""
    return render_chat_completions(ir, model=model)


# ---------------------------------------------------------------------------
# Usage normalization: compatible with DeepSeek-style and OpenAI-style fields
# ---------------------------------------------------------------------------

def _normalize_usage(response_usage: Mapping[str, Any]) -> dict[str, int]:
    if not response_usage:
        return {"raw_input": 0, "cache_read": 0, "cache_write": 0, "output": 0}
    # DeepSeek (passed through by OpenRouter)
    if "prompt_cache_hit_tokens" in response_usage or "prompt_cache_miss_tokens" in response_usage:
        hit = int(response_usage.get("prompt_cache_hit_tokens") or 0)
        miss = int(response_usage.get("prompt_cache_miss_tokens") or 0)
        return {
            "raw_input": miss,
            "cache_read": hit,
            "cache_write": 0,
            "output": int(response_usage.get("completion_tokens") or 0),
        }
    # OpenAI / others: ``cached_tokens`` is in the prompt_tokens sub-field or at the top level
    pt = int(response_usage.get("prompt_tokens") or 0)
    cached = int(
        (response_usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        or response_usage.get("cached_tokens", 0)
    )
    return {
        "raw_input": max(pt - cached, 0),
        "cache_read": cached,
        "cache_write": 0,
        "output": int(response_usage.get("completion_tokens") or 0),
    }


# ---------------------------------------------------------------------------
# Prompt-construction trace helpers
# ---------------------------------------------------------------------------

def _msg_text(m: Mapping[str, Any]) -> str:
    c = m.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts = []
        for p in c:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, Mapping):
                parts.append(str(p.get("text") or p.get("content") or ""))
        return "\n".join(parts)
    return "" if c is None else str(c)


def _summarize_messages(msgs: list[Mapping[str, Any]]) -> dict[str, Any]:
    by_role: dict[str, dict[str, int]] = {}
    for m in msgs:
        role = str(m.get("role", "?"))
        slot = by_role.setdefault(role, {"count": 0, "chars": 0, "tool_calls": 0})
        slot["count"] += 1
        slot["chars"] += len(_msg_text(m))
        if m.get("tool_calls"):
            slot["tool_calls"] += len(m["tool_calls"])  # type: ignore[arg-type]
    return {
        "n_messages": len(msgs),
        "total_chars": sum(s["chars"] for s in by_role.values()),
        "by_role": by_role,
    }


def _summarize_ir(ir) -> dict[str, Any]:
    """Per-band, per-segment block stats. PIN+FOLD+DROP visibility."""
    def _bucket():
        return {b.name: {"blocks": 0, "chars": 0} for b in Band}

    def _add(buck, blocks):
        for b in blocks:
            slot = buck[b.band.name]
            slot["blocks"] += 1
            try:
                slot["chars"] += len(json.dumps(b.payload, ensure_ascii=False)
                                      if not isinstance(b.payload, str)
                                      else b.payload)
            except Exception:  # noqa: BLE001
                slot["chars"] += len(str(b.payload))

    tools = _bucket(); _add(tools, ir.tools)
    system = _bucket(); _add(system, ir.system)
    msgs_band = _bucket()
    msg_kinds: dict[str, int] = {}
    for m in ir.messages:
        _add(msgs_band, m.blocks)
        for b in m.blocks:
            msg_kinds[b.kind] = msg_kinds.get(b.kind, 0) + 1
    return {
        "n_tools": len(ir.tools),
        "n_system_blocks": len(ir.system),
        "n_messages": len(ir.messages),
        "bands": {"tools": tools, "system": system, "messages": msgs_band},
        "msg_block_kinds": msg_kinds,
    }


def _flatten_regions(ir_summary: Mapping[str, Any]) -> dict[str, Any]:
    """Reduce per-band/per-segment summary into flat numbers convenient for the
    dashboard: total chars per segment, total chars per band, grand total.
    """
    bands = ir_summary.get("bands") or {}
    regions: dict[str, Any] = {}
    band_totals = {b.name: 0 for b in Band}
    grand = 0
    for seg in ("tools", "system", "messages"):
        seg_buck = bands.get(seg) or {}
        seg_entry = {b.name: int((seg_buck.get(b.name) or {}).get("chars", 0))
                     for b in Band}
        seg_entry["total"] = sum(seg_entry.values())
        regions[seg] = seg_entry
        for b in Band:
            band_totals[b.name] += seg_entry[b.name]
        grand += seg_entry["total"]
    return {"by_segment": regions, "by_band": band_totals, "total": grand}


def _summarize_plan(plan) -> dict[str, Any]:
    return {
        "routing_key": plan.routing_key,
        "n_slots": len(plan.slots),
        "slots": [
            {
                "name": s.name,
                "segment": s.segment,
                "index": s.index,
                "message_index": s.message_index,
                "ttl_class": s.ttl_class,
            }
            for s in plan.slots
        ],
        "extras": dict(plan.extras) if plan.extras else {},
    }


def _wire_text(wire: Mapping[str, Any]) -> str:
    """Concatenate role-tagged messages — used to measure cross-call prefix match."""
    parts = []
    for m in wire.get("messages", []):
        parts.append(f"[{m.get('role', '?')}]\n{_msg_text(m)}")
        if m.get("tool_calls"):
            parts.append(json.dumps(m["tool_calls"], ensure_ascii=False, sort_keys=True))
    if wire.get("tools"):
        parts.append(json.dumps(wire["tools"], ensure_ascii=False, sort_keys=True))
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Transport (duck interface: mini_swe_runner only uses .chat.completions.create)
# ---------------------------------------------------------------------------

class TelosOpenAITransport:
    """A client with an OpenAI-shaped duck interface, internally routing through TELOS.

    Args:
        base_url: the underlying real endpoint, e.g. ``https://openrouter.ai/api/v1``.
        api_key:  passed explicitly when not readable from an envvar.
        session_id: reuse Bridge stats within the same session.
        usage_log: path that appends one jsonl line per call; ``None`` means don't write.
        engine_name: name of the engine adapter (``deepseek`` for OpenRouter+DS).
        harness_name: defaults to ``telos``.
    """

    def __init__(
        self,
        *,
        base_url: str = "https://openrouter.ai/api/v1",
        api_key: str | None = None,
        session_id: str = "telos-session",
        usage_log: str | None = None,
        prompt_trace_log: str | None = None,
        engine_name: str = "deepseek",
        harness_name: str = "telos",
        session_state: BridgeSessionState | None = None,
    ):
        from openai import OpenAI  # deferred import

        self.base_url = base_url
        self._inner = OpenAI(
            base_url=base_url,
            api_key=api_key or os.environ.get("OPENROUTER_API_KEY", ""),
        )
        self._harness = load_harness(harness_name)
        self._engine = load_engine(engine_name)
        self._engine_name = engine_name
        self._session_id = session_id
        self._usage_log = Path(usage_log) if usage_log else None
        self._trace_log = Path(prompt_trace_log) if prompt_trace_log else None
        self._call_count = 0
        self._prev_wire_text: str = ""
        self._prev_regions: dict[str, Any] | None = None

        # Bridge cross-turn state: one transport instance = one session.
        self._session_state = (
            session_state if session_state is not None else BridgeSessionState()
        )

        # duck interface
        self.chat = _ChatNS(self)

    @property
    def session_state(self) -> BridgeSessionState:
        return self._session_state

    # ------------------------------------------------------------------
    # Internal: execute one create
    # ------------------------------------------------------------------
    def _do_create(self, kwargs: dict[str, Any]):
        self._call_count += 1
        model = kwargs.get("model", "")
        # ---- 0. snapshot of the caller's raw input ----
        in_msgs = list(kwargs.get("messages") or [])
        in_tools = list(kwargs.get("tools") or [])
        input_summary = _summarize_messages(in_msgs)
        input_summary["n_tools"] = len(in_tools)

        # 1. parse → IR
        ir = self._harness.parse(
            kwargs,
            session_id=self._session_id,
            engine=self._engine_name,
            model=model,
        )
        ir_in_summary = _summarize_ir(ir)

        # 2. Bridge: pass session_state so the R8 counters / cache_creation accumulate across turns
        bridge = Bridge(ir, self._engine, session_state=self._session_state)

        # 3. Canonicalize (tools ordering, payload key ordering) — cannot use the
        # ir snapshot directly; it must run through _canonicalize_ir to guarantee
        # the multi-round prefix bytes are stable. This is the key fix on the
        # OpenAI path: previously ir2 was fed straight to the wire builder, skipping this step.
        ir2 = _canonicalize_ir(bridge.snapshot_ir())
        plan = self._engine.plan_marks(ir2)
        plan_summary = _summarize_plan(plan)
        ir_out_summary = _summarize_ir(ir2)
        regions = _flatten_regions(ir_out_summary)
        # Growth process: char changes relative to the previous call (by segment & band)
        if self._prev_regions is None:
            region_deltas: dict[str, Any] = {"first_call": True}
        else:
            prev = self._prev_regions
            region_deltas = {
                "first_call": False,
                "by_segment": {
                    seg: regions["by_segment"][seg]["total"]
                         - prev["by_segment"][seg]["total"]
                    for seg in ("tools", "system", "messages")
                },
                "by_band": {
                    b.name: regions["by_band"][b.name] - prev["by_band"][b.name]
                    for b in Band
                },
                "total": regions["total"] - prev["total"],
            }
        try:
            wire = dict(self._engine.emit_for_protocol(
                ir2, plan, protocol="openai-chat",
            ))
        except UnsupportedWireProtocolError:
            wire = _ir_to_chat_completions(ir2, model=model)
        # 4. pass through some fields that telos does not care about
        for k in ("temperature", "top_p", "max_tokens", "stream",
                  "timeout", "tool_choice", "response_format"):
            if k in kwargs and kwargs[k] is not None:
                wire[k] = kwargs[k]

        wire_summary = _summarize_messages(wire.get("messages", []))
        wire_summary["n_tools"] = len(wire.get("tools") or [])
        # Cross-call prefix stability: the strongest leading indicator of cache hits
        wire_text = _wire_text(wire)
        prefix_match_chars = len(commonprefix([self._prev_wire_text, wire_text])) \
            if self._prev_wire_text else 0

        # The real request is about to be sent — equivalent to the +1 at the end
        # of bridge.emit_with_plan. This path renders explicitly for the
        # Chat Completions protocol instead of calling Bridge.emit_with_plan.
        self._session_state.stats.real_requests_since_refresh += 1

        # 5. actually send the request
        t0 = time.time()
        response = self._inner.chat.completions.create(**wire)
        dt = time.time() - t0

        # 6. usage normalization + cross-turn accumulation
        usage_obj = getattr(response, "usage", None)
        usage_dict = usage_obj.model_dump() if usage_obj is not None else {}
        normalized = _normalize_usage(usage_dict)
        inp_total = normalized["raw_input"] + normalized["cache_read"]
        cache_share = (normalized["cache_read"] / inp_total) if inp_total else 0.0

        # bridge.absorb_usage: extracts cache_write via engine.parse_usage and
        # accumulates it into session_state. DeepSeek/OpenAI's cache_write is
        # usually 0, but the call form is aligned with the anthropic transport,
        # keeping R8 visibility.
        try:
            bridge.absorb_usage({"usage": usage_dict})
        except Exception:  # noqa: BLE001
            pass

        if self._usage_log is not None:
            self._usage_log.parent.mkdir(parents=True, exist_ok=True)
            with self._usage_log.open("a") as f:
                f.write(json.dumps({
                    "ts": time.time(),
                    "session_id": self._session_id,
                    "call_index": self._call_count,
                    "model": model,
                    "harness": self._harness.__class__.__name__.lower().replace("plugin", ""),
                    "latency_s": round(dt, 3),
                    "routing_key": plan.routing_key,
                    "raw_usage": usage_dict,
                    "normalized": normalized,
                    "cumulative": {
                        "cache_creation":
                            self._session_state.stats.cumulative_cache_creation,
                        "real_requests_since_refresh":
                            self._session_state.stats.real_requests_since_refresh,
                        "refpool_slugs": sorted(self._session_state.refpool.slugs),
                    },
                }, ensure_ascii=False) + "\n")

        if self._trace_log is not None:
            self._trace_log.parent.mkdir(parents=True, exist_ok=True)
            with self._trace_log.open("a") as f:
                f.write(json.dumps({
                    "session_id": self._session_id,
                    "call_index": self._call_count,
                    "model": model,
                    "latency_s": round(dt, 3),
                    "input": input_summary,
                    "ir_after_parse": ir_in_summary,
                    "ir_after_canonicalize": ir_out_summary,
                    "regions": regions,
                    "region_deltas": region_deltas,
                    "plan": plan_summary,
                    "breakpoints": plan_summary["slots"],
                    "wire": wire_summary,
                    "prefix": {
                        "prev_wire_chars": len(self._prev_wire_text),
                        "this_wire_chars": len(wire_text),
                        "common_prefix_chars": prefix_match_chars,
                        "prefix_stability": (
                            prefix_match_chars / len(self._prev_wire_text)
                            if self._prev_wire_text else None
                        ),
                    },
                    "cache": {
                        "raw_input": normalized["raw_input"],
                        "cache_read": normalized["cache_read"],
                        "output": normalized["output"],
                        "input_total": inp_total,
                        "cache_share": round(cache_share, 4),
                    },
                    "cumulative": {
                        "cache_creation":
                            self._session_state.stats.cumulative_cache_creation,
                        "real_requests_since_refresh":
                            self._session_state.stats.real_requests_since_refresh,
                        "refpool_slugs": sorted(self._session_state.refpool.slugs),
                    },
                }, ensure_ascii=False) + "\n")

        self._prev_wire_text = wire_text
        self._prev_regions = regions
        return response


class _ChatNS:
    def __init__(self, t: TelosOpenAITransport):
        self.completions = _CompletionsNS(t)


class _CompletionsNS:
    def __init__(self, t: TelosOpenAITransport):
        self._t = t

    def create(self, **kwargs):
        return self._t._do_create(kwargs)


class VanillaOpenAITransport:
    """Plain OpenAI client with the same duck interface and usage-log schema as
    ``TelosOpenAITransport`` — but without TELOS in the path. Used for A/B
    comparisons (``--no-telos`` runs).
    """

    def __init__(
        self,
        *,
        base_url: str = "https://openrouter.ai/api/v1",
        api_key: str | None = None,
        session_id: str = "vanilla-session",
        usage_log: str | None = None,
        engine_name: str = "deepseek",
        harness_name: str = "vanilla",
    ):
        from openai import OpenAI

        self._inner = OpenAI(
            base_url=base_url,
            api_key=api_key or os.environ.get("OPENROUTER_API_KEY", ""),
        )
        self._session_id = session_id
        self._usage_log = Path(usage_log) if usage_log else None
        self._engine_name = engine_name
        self._harness_name = harness_name
        self._call_count = 0
        self.chat = _VanillaChatNS(self)

    def _do_create(self, kwargs: dict[str, Any]):
        self._call_count += 1
        model = kwargs.get("model", "")
        t0 = time.time()
        response = self._inner.chat.completions.create(**kwargs)
        dt = time.time() - t0

        usage_obj = getattr(response, "usage", None)
        usage_dict = usage_obj.model_dump() if usage_obj is not None else {}
        normalized = _normalize_usage(usage_dict)

        if self._usage_log is not None:
            self._usage_log.parent.mkdir(parents=True, exist_ok=True)
            with self._usage_log.open("a") as f:
                f.write(json.dumps({
                    "ts": time.time(),
                    "session_id": self._session_id,
                    "call_index": self._call_count,
                    "model": model,
                    "harness": self._harness_name,
                    "latency_s": round(dt, 3),
                    "raw_usage": usage_dict,
                    "normalized": normalized,
                }, ensure_ascii=False) + "\n")
        return response


class _VanillaChatNS:
    def __init__(self, t: VanillaOpenAITransport):
        self.completions = _VanillaCompletionsNS(t)


class _VanillaCompletionsNS:
    def __init__(self, t: VanillaOpenAITransport):
        self._t = t

    def create(self, **kwargs):
        return self._t._do_create(kwargs)
