"""Bridge: the policy core of TELOS. Five primitives + one canonicalize.

```
upstream agent → harness.parse() → IR
                                    │
                                    ▼
                          ┌────────────────────┐
                          │     Bridge          │
                          │   place / pin /     │
                          │   mark  / fold /    │
                          │   refresh           │
                          └─────────┬──────────┘
                                    │ IR (after rewrite)
                                    ▼
                          engine.emit() → wire request
                          engine.parse_usage() → UsageReport
```

The Bridge is **stateful** (one instance per session):
- maintains the ref-pool (a slug is frozen once registered)
- maintains "the number of real requests since the last mark", used for ``refresh``'s
  adaptive gating (fixes R8)
- maintains a cumulative ``cache_creation`` counter and hints the upstream to ``Fold``
  once a threshold is reached

The Bridge does **not** track any engine-private state (breakpoint slot numbers, TTL
slots, etc.) -- those are recomputed on demand by the engine adapter during ``plan_marks``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, replace
from typing import Any, Mapping

from telos.engine.base import (
    BidirectionalEngineAdapter,
    EmitPlan,
    EngineAdapter,
    ProbeResult,
)
from telos.ir import (
    Band,
    TelosBlock,
    TelosIR,
    TelosInvariantError,
    TelosMessage,
    UsageReport,
    assert_band_order,
    assert_ir_invariants,
)
from telos.refpool import RefPool


# ---------------------------------------------------------------------------
# Bridge state persisted across turns
# ---------------------------------------------------------------------------

@dataclass
class _SessionStats:
    """Tracks the number of real requests since the last refresh (used for refresh adaptive gating)."""

    real_requests_since_refresh: int = 0
    cumulative_cache_creation: int = 0
    last_refresh_at: float = field(default_factory=time.monotonic)


@dataclass
class BridgeSessionState:
    """All cross-turn state of a single conversation session.

    Design intent: the upstream (proxy / SDK transport) can hold one
    ``BridgeSessionState`` per session_id and pass it in when constructing a new
    ``Bridge`` each turn. This way:

    - a ref-pool slug is registered once and shared across the whole session (fold state
      is preserved across turns)
    - the request count for R8 adaptive refresh can truly accumulate
    - the cumulative cache_creation counter can truly accumulate and trigger the upstream
      fold hint

    When omitted (``None``), the Bridge creates one itself, degrading the behavior to
    "independent per turn" -- exactly equivalent to before, guaranteeing existing callers
    are not broken.
    """

    refpool: RefPool = field(default_factory=RefPool)
    stats: _SessionStats = field(default_factory=_SessionStats)
    # Once a session's harness is identified (hermes / openclaw), subsequent requests in
    # the same session reuse it directly, avoiding a re-probe on every call that would
    # cause "openclaw <-> hermes flip-flopping -> inconsistent source_tag prefix ->
    # ref-pool slug mismatch". Explicitly passing ``harness_name`` always overrides this field.
    sticky_harness: str | None = None
    # Likewise: the mode (none/telos/rtk/both) a session sees the first time is locked,
    # and subsequent requests in the same session reuse it, to avoid a session switching
    # gears mid-way in a comparison experiment. The proxy config default can still be
    # overridden once by the first request's X-Telos-Mode header.
    sticky_mode: str | None = None
    # Comparison-experiment group label (X-Telos-Compare-Group header). Multiple sessions
    # under the same compare_group with different modes are shown side by side on the dashboard.
    compare_group: str | None = None


REFRESH_THRESHOLD = 11  # Janus §6.3.1: at least 11 real requests per renewal window to break even


# ---------------------------------------------------------------------------
# Canonicalization (fixes R5: cross-engine generic, must be done uniformly before emit)
# ---------------------------------------------------------------------------

# Array keys with "set semantics" in JSON-Schema: order-insensitive, byte-stable once sorted.
# Treated as sets only within the schema subtree of a tool_def; the payload of tool_use /
# tool_result is user data and must never be touched (a field that happens to be named
# ``required`` in a payload must not be silently reordered).
#
# Keys deliberately *not* sorted (kept conservative, in sync with Janus tools.ts):
#   - ``enum``        : order is often used as a tie-break preference
#   - ``examples``    : documentation examples may be intentionally ordered
#   - ``anyOf`` / ``oneOf`` / ``allOf`` : the spec says unordered, but prompts commonly
#                       rely on "prefer first matching schema" semantics
# Exposed as a module-level name so special harnesses can monkey-patch it after import.
_SCHEMA_SET_ARRAY_KEYS: frozenset[str] = frozenset({"required"})

# Tool source ordering: builtin first, MCP next, user last; untagged sorts last (safe default).
_TOOL_SOURCE_RANK: Mapping[str, int] = {"builtin": 0, "mcp": 1, "user": 2}
_TOOL_SOURCE_DEFAULT_RANK = 3


def _canonicalize_payload(payload: Any) -> Any:
    """Sort keys for dict-like payloads; return other types unchanged.

    The Anthropic docs explicitly note that Swift / Go JSON serialization randomizes key
    order, which invalidates the cache. DeepSeek's prefix is exact-match, OpenAI's prefix
    is a hash -- all engines are affected. So this belongs in the bridge, not the adapter.

    This function does *not* touch set-semantics arrays (such as ``required``) -- that
    only happens in ``_canonicalize_schema`` and only on the schema subtree of a tool_def.
    """
    if isinstance(payload, dict):
        return {k: _canonicalize_payload(payload[k]) for k in sorted(payload.keys())}
    if isinstance(payload, list):
        return [_canonicalize_payload(x) for x in payload]
    return payload


def _canonicalize_schema(node: Any, *, parent_key: str | None = None) -> Any:
    """Canonicalization dedicated to JSON-Schema subtrees: dict key sorting + set-semantics array sorting.

    The only difference from ``_canonicalize_payload`` is: when a list's parent key belongs
    to ``_SCHEMA_SET_ARRAY_KEYS``, it is sorted as strings rather than keeping the original
    order. This way ``required: ["b","a"]`` and ``["a","b"]`` are byte-identical, preventing
    implicit reordering like ``list(set(...))`` from breaking the prefix cache.
    """
    if isinstance(node, dict):
        return {
            k: _canonicalize_schema(node[k], parent_key=k)
            for k in sorted(node.keys())
        }
    if isinstance(node, list):
        if parent_key in _SCHEMA_SET_ARRAY_KEYS:
            sorted_items = sorted(node, key=lambda x: str(x))
            return [_canonicalize_schema(x) for x in sorted_items]
        return [_canonicalize_schema(x) for x in node]
    return node


def _canonicalize_tool_def(payload: Any) -> Any:
    """Canonicalize a tool_def payload: recognizes both the Anthropic and OpenAI schema fields.

    - Anthropic shape: ``{"name", "description", "input_schema": {...}}``
    - OpenAI   shape: ``{"type": "function", "function": {"name", "description",
                          "parameters": {...}}}``

    The schema subtree (``input_schema`` / ``parameters``) goes through
    ``_canonicalize_schema``, the other fields through plain ``_canonicalize_payload``.
    An unrecognized shape degrades to a whole ``_canonicalize_payload`` -- equivalent to
    before the change.
    """
    if not isinstance(payload, dict):
        return _canonicalize_payload(payload)

    # Anthropic shape
    if "input_schema" in payload:
        out: dict[str, Any] = {}
        for k in sorted(payload.keys()):
            v = payload[k]
            out[k] = _canonicalize_schema(v) if k == "input_schema" else _canonicalize_payload(v)
        return out

    # OpenAI function-tool shape
    fn = payload.get("function")
    if isinstance(fn, dict) and ("parameters" in fn or "name" in fn):
        out = {}
        for k in sorted(payload.keys()):
            v = payload[k]
            if k == "function":
                inner: dict[str, Any] = {}
                for fk in sorted(v.keys()):
                    fv = v[fk]
                    inner[fk] = _canonicalize_schema(fv) if fk == "parameters" else _canonicalize_payload(fv)
                out[k] = inner
            else:
                out[k] = _canonicalize_payload(v)
        return out

    return _canonicalize_payload(payload)


def _canonicalize_block(blk: TelosBlock) -> TelosBlock:
    """Canonicalize a single block: field sorting for tool_def / tool_use / tool_result.

    ``tool_def`` goes through the schema-aware path (additionally canonicalizes set arrays
    such as ``required``); ``tool_use`` / ``tool_result`` only do dict key sorting -- their
    payload is user data, so array order must not be changed.
    """
    if blk.kind == "tool_def":
        return replace(blk, payload=_canonicalize_tool_def(blk.payload))
    if blk.kind in ("tool_use", "tool_result"):
        return replace(blk, payload=_canonicalize_payload(blk.payload))
    return blk


def _tool_name(blk: TelosBlock) -> str:
    """Extract the tool name from a tool_def block (covers both the Anthropic and OpenAI shapes)."""
    p = blk.payload
    if isinstance(p, dict):
        name = p.get("name")
        if isinstance(name, str):
            return name
        fn = p.get("function")
        if isinstance(fn, dict):
            name = fn.get("name")
            if isinstance(name, str):
                return name
    return blk.id


def _tool_sort_key(blk: TelosBlock) -> tuple[int, str, str]:
    """The stable sort key for the tool array: ``(source_rank, mcp_server, name)``.

    - ``source_rank`` : builtin(0) → mcp(1) → user(2) → untagged(3)
    - ``mcp_server``  : when both are MCP, sort stably by server name to avoid a
                        multi-server startup race interleaving inserts between two servers
                        and breaking the prefix
    - ``name``        : lexicographic order within a group

    All PIN tools are still PIN after sorting, so §5 band order is not broken
    (``assert_band_order`` re-checks at emit time). ``extra["source"]`` and
    ``extra["mcp_server"]`` are harness-convention tags -- when missing, they degrade to
    the last position (safe default).
    """
    extra = blk.extra or {}
    src = extra.get("source") if isinstance(extra, Mapping) else None
    if isinstance(src, str):
        rank = _TOOL_SOURCE_RANK.get(src, _TOOL_SOURCE_DEFAULT_RANK)
    else:
        rank = _TOOL_SOURCE_DEFAULT_RANK
    server = extra.get("mcp_server") if isinstance(extra, Mapping) else None
    return (rank, str(server or ""), _tool_name(blk))


def _canonicalize_ir(ir: TelosIR) -> TelosIR:
    # tools: canonicalize each block internally → stable-sort the whole array
    # (still all PIN, so §5 is not broken)
    canon_tools = sorted(
        (_canonicalize_block(b) for b in ir.tools),
        key=_tool_sort_key,
    )
    new_tools = tuple(canon_tools)

    new_system = tuple(_canonicalize_block(b) for b in ir.system)
    new_messages = tuple(
        TelosMessage(role=m.role, blocks=tuple(_canonicalize_block(b) for b in m.blocks))
        for m in ir.messages
    )
    return replace(ir, tools=new_tools, system=new_system, messages=new_messages)


# ---------------------------------------------------------------------------
# Bridge body
# ---------------------------------------------------------------------------

class Bridge:
    """One instance per session. Not thread-safe (a single session is usually processed sequentially).

    Cross-turn state is externalized: passing in ``session_state`` lets the ref-pool +
    R8 counter accumulate across calls; without it, each Bridge holds its own state
    (equivalent to early-version behavior).
    """

    def __init__(
        self,
        ir: TelosIR,
        engine: EngineAdapter,
        *,
        session_state: BridgeSessionState | None = None,
    ):
        self._ir = ir
        self._engine = engine
        self._state = session_state if session_state is not None else BridgeSessionState()
        # Sync the ref_pool from the current IR into state.refpool.
        # Use register_or_skip: from the second call on, an already-registered slug (which
        # may have been folded into a placeholder) is not overwritten back by a new round's
        # full payload.
        for slug, blk in ir.ref_pool.items():
            self._state.refpool.register_or_skip(slug, blk)
        # The initial IR also goes through the §5 check, so harness plugins can't cut corners
        assert_ir_invariants(self._ir)

    @property
    def session_state(self) -> BridgeSessionState:
        """Expose the externalized state. The upstream may read ``cumulative_cache_creation`` etc. for diagnostics."""
        return self._state

    # Backward compatibility: old code may read these fields directly. Keep the property read path.
    @property
    def _refpool(self) -> RefPool:  # type: ignore[override]
        return self._state.refpool

    @property
    def _stats(self) -> _SessionStats:  # type: ignore[override]
        return self._state.stats

    # ------------------------------------------------------------------
    # The five primitives
    # ------------------------------------------------------------------

    def place(self, segment: str, blocks: tuple[TelosBlock, ...]) -> "Bridge":
        """**Place**: replace all blocks of a segment (``"tools"`` / ``"system"`` / ``"messages"``),
        and immediately re-run the §5 check.

        Place is the explicit "accept new IR" action, called by the harness whenever a new turn arrives.
        """
        if segment == "tools":
            assert_band_order(blocks, "tools")
            if any(b.band is not Band.PIN for b in blocks):
                raise TelosInvariantError("tools blocks must all be band=PIN")
            self._ir = replace(self._ir, tools=blocks)
        elif segment == "system":
            assert_band_order(blocks, "system")
            self._ir = replace(self._ir, system=blocks)
        else:
            raise ValueError(f"Unknown segment for place(): {segment!r}")
        return self

    def append_message(self, msg: TelosMessage) -> "Bridge":
        """A message-specific shortcut for **Place**: append a new message.

        Every append checks the §5 order inside the message -- this is the key to fixing
        Janus C6: a user message's envelope must be cut into a ``DROP`` sub-block.
        """
        assert_band_order(msg.blocks, f"new message (role={msg.role})")
        self._ir = replace(self._ir, messages=self._ir.messages + (msg,))
        return self

    def pin(self, slug: str, payload: str, *, source_tag: str | None = None) -> "Bridge":
        """**Pin**: register a ref-pool entry; the slug is frozen immediately.

        Note that Pin registers a foldable entry with ``band=FOLD`` (seemingly at odds
        with the primitive name "Pin", but Pin here means "fix this large chunk of content
        in the ref-pool and give it a stable pointer", not "band=PIN").
        """
        blk = TelosBlock(
            id=f"ref:{slug}",
            band=Band.FOLD,
            kind="text",
            payload=payload,
            ref_slug=slug,
            source_tag=source_tag or "ref-pool/registered",
        )
        self._refpool.register(slug, blk)
        # Render the ref-pool at the tail of the system segment (§4 keeps all large content here)
        self._sync_refpool_into_system()
        return self

    def mark(self) -> EmitPlan:
        """**Mark**: let the engine adapter decide the cache anchors for this emit.

        The bridge has no knowledge of engine-private concepts like cache_control /
        prompt_cache_key, and delegates the decision entirely to the adapter.
        """
        return self._engine.plan_marks(self._ir)

    def fold(
        self,
        *,
        slugs: tuple[str, ...] = (),
        message_range: tuple[int, int] | None = None,
        summary: str = "<folded prior turns>",
    ) -> "Bridge":
        """**Fold**: fold ref-pool entries, or fold a span of history messages into a summary.

        Fixes R4: after a Fold, every Mark slot that lands after the fold region must be
        re-planned by the next ``mark()`` -- the bridge does not cache the plan, so this
        holds naturally.

        Parameters:
            slugs: list of ref-pool slugs to fold (only the payload is replaced, the slug stays)
            message_range: ``(start, end)`` half-open interval; replaces this span of history
                messages with a single ``band=FOLD`` summary message
            summary: the placeholder text used when folding messages
        """
        for slug in slugs:
            self._refpool.fold(slug)
        self._sync_refpool_into_system()

        if message_range is not None:
            start, end = message_range
            if not (0 <= start < end <= len(self._ir.messages)):
                raise TelosInvariantError(
                    f"Invalid message_range {message_range!r} for "
                    f"{len(self._ir.messages)} messages"
                )
            placeholder = TelosMessage(
                role="user",
                blocks=(
                    TelosBlock(
                        id=f"folded:{start}-{end}",
                        band=Band.FOLD,
                        kind="text",
                        payload=summary,
                        source_tag="bridge/fold-history",
                    ),
                ),
            )
            new_msgs = (
                self._ir.messages[:start]
                + (placeholder,)
                + self._ir.messages[end:]
            )
            self._ir = replace(self._ir, messages=new_msgs)
        return self

    def refresh(self, plan: EmitPlan) -> bool:
        """**Refresh**: trigger the engine's keep-alive; a no-op if the engine doesn't support it.

        Fixes R8: adaptive gating -- if the number of real requests in the window is below
        the threshold, skip the renewal and let the cache expire naturally. This avoids a
        low-activity session where the renewal cost > the benefit.
        """
        if not self._engine.capabilities.prewarmable:
            return False
        if self._stats.real_requests_since_refresh < REFRESH_THRESHOLD:
            return False
        self._engine.refresh(self._ir, plan)
        self._stats.last_refresh_at = time.monotonic()
        self._stats.real_requests_since_refresh = 0
        return True

    # ------------------------------------------------------------------
    # emit / flow-back: the bridge is also responsible for normalizing the usage the engine returns
    # ------------------------------------------------------------------

    def emit(self) -> Mapping[str, Any]:
        """Canonicalize → validate → delegate to engine.emit() to produce the wire request."""
        wire, _ = self.emit_with_plan()
        return wire

    def emit_with_plan(self) -> tuple[Mapping[str, Any], EmitPlan]:
        """The two-value return version of ``emit()``: get both the wire and the EmitPlan used this time.

        Use this when the proxy / transport wants to record plan diagnostics (slot names,
        routing_key, etc.), to avoid them re-running canonicalize + plan_marks themselves.
        """
        canon = _canonicalize_ir(self._ir)
        # One more full §5 check before rendering (there are many entry points that modify
        # the IR, this is the last line of defense)
        assert_ir_invariants(canon)
        # ref-pool lint: scan all [ref:...] references inside text blocks
        self._state.refpool.lint_blocks(canon.system, "system")
        for i, m in enumerate(canon.messages):
            self._state.refpool.lint_blocks(m.blocks, f"messages[{i}]")
        plan = self._engine.plan_marks(canon)
        wire = self._engine.emit(canon, plan)
        self._state.stats.real_requests_since_refresh += 1
        return wire, plan

    def absorb_usage(self, raw_response: Mapping[str, Any]) -> UsageReport:
        """Parse the engine response and update the cumulative cache_creation counter."""
        report = self._engine.parse_usage(raw_response)
        self._stats.cumulative_cache_creation += report.cache_write
        return report

    # ------------------------------------------------------------------
    # Diagnostics / debugging
    # ------------------------------------------------------------------

    @property
    def cumulative_cache_creation(self) -> int:
        return self._stats.cumulative_cache_creation

    # ------------------------------------------------------------------
    # Bidirectional operations (only capability-verified extension adapters;
    # stock SGLang and closed APIs remain no-ops)
    # ------------------------------------------------------------------

    @property
    def is_bidirectional(self) -> bool:
        return isinstance(self._engine, BidirectionalEngineAdapter)

    def probe_cache(self) -> ProbeResult:
        """**Probe**: ask the server "is the prefix still in the cache?"

        Adapters without a verified lookup extension return ``hit=False``.
        The bridge uses this result to decide whether to skip an imminent ``refresh``,
        saving one RTT.
        """
        if not isinstance(self._engine, BidirectionalEngineAdapter):
            return ProbeResult(hit=False)
        plan = self._engine.plan_marks(self._ir)
        return self._engine.probe(self._ir, plan)

    def cooperative_fold(
        self,
        *,
        slugs: tuple[str, ...] = (),
        message_range: tuple[int, int] | None = None,
        summary: str = "<folded prior turns>",
    ) -> Mapping[str, Any]:
        """**Cooperative Fold**: client-side fold + server-side evict-span / fork-and-replace.

        Unlike a plain ``fold()``: this method not only modifies the IR but also returns a
        ``cache_control`` / ``cache_policy`` fragment for the caller to merge into the next
        emit's plan extras. Physical eviction/fork behavior is available only
        when a concrete extension adapter declares and implements that contract.

        Calling this method on a closed-source API is equivalent to ``fold()`` + returning ``{}``.
        """
        # First do the client-side IR rewrite (same as a plain fold)
        self.fold(slugs=slugs, message_range=message_range, summary=summary)
        if not isinstance(self._engine, BidirectionalEngineAdapter):
            return {}

        caps = self._engine.capabilities
        # Prefer fork_and_replace when the extension explicitly advertises it.
        if caps.fork_and_replace and message_range is not None:
            plan = self._engine.plan_marks(self._ir)
            path_hash = plan.extras.get("path_hash") or plan.routing_key or ""
            return self._engine.fork_and_replace(
                self._ir,
                path_hash=path_hash,
                replace_suffix={"text": summary},
            )
        # Next best: evict_span (vLLM's main path)
        if caps.span_eviction and message_range is not None:
            start, end = message_range
            return self._engine.evict_span(self._ir, start, end)
        return {}

    def emit_with_extras(self, extras: Mapping[str, Any]) -> Mapping[str, Any]:
        """The extended version of ``emit()``: lets the caller merge the cache_control fragment
        returned by a bidirectional operation into plan.extras.

        Typical usage:

            ctrl = bridge.cooperative_fold(message_range=(2, 8), summary="…")
            wire = bridge.emit_with_extras(ctrl)
        """
        canon = _canonicalize_ir(self._ir)
        assert_ir_invariants(canon)
        self._refpool.lint_blocks(canon.system, "system")
        for i, m in enumerate(canon.messages):
            self._refpool.lint_blocks(m.blocks, f"messages[{i}]")
        plan = self._engine.plan_marks(canon)
        merged = EmitPlan(
            slots=plan.slots,
            routing_key=plan.routing_key,
            extras={**dict(plan.extras), **dict(extras)},
        )
        wire = self._engine.emit(canon, merged)
        self._stats.real_requests_since_refresh += 1
        return wire

    def snapshot_ir(self) -> TelosIR:
        """Return a snapshot of the current IR (for serialization / testing)."""
        return self._ir

    def dump_layout(self) -> str:
        """Print the band distribution of the current IR; for debugging."""
        lines: list[str] = [f"-- session {self._ir.session_id} --"]

        def fmt(blocks: tuple[TelosBlock, ...]) -> str:
            return " | ".join(f"{b.band.value}:{b.id}" for b in blocks)

        lines.append(f"tools  : {fmt(self._ir.tools)}")
        lines.append(f"system : {fmt(self._ir.system)}")
        for i, m in enumerate(self._ir.messages):
            lines.append(f"msg[{i}] {m.role:9s}: {fmt(m.blocks)}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal: sync the ref-pool into the system segment (pin* → fold(ref-pool)*)
    # ------------------------------------------------------------------

    def _sync_refpool_into_system(self) -> None:
        # Take all non-ref-pool-sourced blocks in system (keep harness-injected system pin / drop)
        non_pool = tuple(b for b in self._ir.system if b.ref_slug is None)
        # Recombine: keep the existing pins → add the ref-pool fold → keep the existing drops
        pins = tuple(b for b in non_pool if b.band is Band.PIN)
        drops = tuple(b for b in non_pool if b.band is Band.DROP)
        # Render the ref-pool in lexicographic order (guarantees byte stability across emits)
        pool_blocks = self._refpool.render_blocks()
        new_system = pins + pool_blocks + drops
        assert_band_order(new_system, "system (after refpool sync)")
        self._ir = replace(
            self._ir,
            system=new_system,
            ref_pool=self._refpool.to_mapping(),
        )
