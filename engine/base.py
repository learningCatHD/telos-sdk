"""Engine adapter abstract base class and capability matrix.

Every adapter implements this interface; the bridge always works against
the interface and never branches on the engine name.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

from telos.ir import TelosIR, UsageReport


# ---------------------------------------------------------------------------
# Capability matrix
# ---------------------------------------------------------------------------

FeatureSupport = Literal[
    "unsupported", "native", "config_only", "extension", "emulated",
]
WireProtocol = Literal[
    "anthropic-messages", "openai-chat", "openai-responses",
]


@dataclass(frozen=True)
class CacheCapabilities:
    """Feature-level support exposed by an engine's public cache contract.

    ``config_only`` means the runtime provides the feature through server
    configuration, not through a per-request field. ``extension`` is reserved
    for a capability-negotiated Telos extension and must not be sent to a
    stock server.
    """

    prefix_reuse: FeatureSupport = "unsupported"
    cache_report: FeatureSupport = "unsupported"
    tier_report: FeatureSupport = "unsupported"
    hierarchical_storage: FeatureSupport = "unsupported"
    request_namespace: FeatureSupport = "unsupported"
    cache_manifest: FeatureSupport = "unsupported"
    boundary_resolution: FeatureSupport = "unsupported"
    retention_hint: FeatureSupport = "unsupported"
    prefetch_hint: FeatureSupport = "unsupported"
    explicit_evict: FeatureSupport = "unsupported"


@dataclass(frozen=True)
class EngineCapabilities:
    """Declares which cache control primitives an engine supports.

    The bridge uses these booleans to decide whether to call the
    corresponding adapter method; an adapter must *never* silently turn
    an unsupported operation into a no-op — it must explicitly declare
    ``False``.
    """

    explicit_breakpoints: bool       #: Anthropic only
    ttl_control: Literal["none", "presets", "seconds"]
    prewarmable: bool                #: ``max_tokens:0``-style keep-alive
    routing_key: bool                #: OpenAI ``prompt_cache_key``
    retention_policy: Literal["fixed", "configurable"]
    max_breakpoints: int             #: 0 = no explicit BP
    thinking_preserved_across_non_tool_result: bool = False
    """Fix R6: True only for Opus 4.5+/Sonnet 4.6+; False for all earlier models and Haiku."""

    # —— Deprecated active-control flags. Stock adapters keep these False; a
    # verified patched-runtime adapter may opt in explicitly. ——
    cache_probe: bool = False        #: client can read server-side cache hit status
    span_eviction: bool = False      #: client can explicitly release a span of KV blocks
    fork_and_replace: bool = False   #: SGLang radix fork: replace the tail while keeping the prefix
    tier_hint: bool = False          #: HiCache three-tier (GPU/CPU/disk) explicit hint
    pin_unpin: bool = False          #: explicit pin / unpin to prevent LRU eviction
    # Feature-level truth source. The booleans above remain for compatibility
    # with existing adapters and are deprecated for new cache integrations.
    cache: CacheCapabilities = field(default_factory=CacheCapabilities)


# ---------------------------------------------------------------------------
# Mark slot abstraction —— the bridge only sees a list of slots, it does not
# know what cache_control looks like
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MarkSlot:
    """A single cache anchor's position + desired TTL.

    "Position" is a logical pointer: ``segment`` ∈ {``"tools"``, ``"system"``,
    ``"message"``} + ``index`` indicates which block within that segment. At
    ``emit`` time the adapter translates it into engine-private fields
    (Anthropic's ``cache_control``, an OpenAI ``prompt_cache_key``-derived
    hash, etc.).
    """

    name: str                        #: diagnostic: BP-T / BP-S / BP-R / BP-X / BP-mid
    segment: Literal["tools", "system", "message"]
    index: int                       #: block index within the segment; the message segment also needs message_index
    message_index: int | None = None
    ttl_class: Literal["short", "long", "none"] = "long"


@dataclass(frozen=True)
class EmitPlan:
    """Return value of ``Mark()``; the engine-private emit decision."""

    slots: tuple[MarkSlot, ...] = ()
    routing_key: str | None = None
    extras: Mapping[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Adapter base class
# ---------------------------------------------------------------------------

class EngineAdapter(ABC):
    """Three methods + one property — the entire interface of an engine adapter."""

    @property
    @abstractmethod
    def capabilities(self) -> EngineCapabilities: ...

    @abstractmethod
    def plan_marks(self, ir: TelosIR) -> EmitPlan:
        """Decide where to place the anchors for this emit, based on the IR."""

    @abstractmethod
    def emit(self, ir: TelosIR, plan: EmitPlan) -> Mapping[str, Any]:
        """Translate IR + plan into a wire request (dict form; the caller POSTs it itself)."""

    @abstractmethod
    def parse_usage(self, response: Mapping[str, Any]) -> UsageReport:
        """Extract usage from the engine response, normalized into a ``UsageReport``."""

    def cache_telemetry_request_fields(
        self,
        *,
        tier_details: bool = False,
    ) -> Mapping[str, Any]:
        """Return adapter-owned opt-in fields for public cache telemetry.

        The default is deliberately empty. A stock adapter may override this
        only for a documented public request field; private cache controls
        belong behind a separately negotiated extension.
        """
        return {}

    def parse_cache_telemetry(
        self,
        response: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Normalize engine-specific cache-tier telemetry.

        Missing or malformed telemetry returns an empty mapping. Callers must
        not infer a physical cache tier from the aggregate cached-token count.
        """
        return {}

    def emit_for_protocol(
        self,
        ir: TelosIR,
        plan: EmitPlan,
        *,
        protocol: WireProtocol,
    ) -> Mapping[str, Any]:
        """Emit for an explicitly selected transport protocol.

        Adapters opt in protocol by protocol. This prevents an OpenAI
        Responses body from accidentally being sent to Chat Completions.
        """
        raise UnsupportedWireProtocolError(
            f"{type(self).__name__} does not emit protocol {protocol!r}"
        )

    def refresh(self, ir: TelosIR, plan: EmitPlan) -> None:
        """Optional: issue a keep-alive request; no-op by default."""
        return None


class UnsupportedWireProtocolError(ValueError):
    """Raised when an adapter cannot render the requested wire protocol."""


# ---------------------------------------------------------------------------
# Bidirectional extension mixin; the bridge uses isinstance to detect it.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProbeResult:
    """Result of a server-side prefix-cache hit query."""

    hit: bool
    cached_token_count: int = 0
    tier: Literal["gpu", "cpu", "disk", "none"] = "none"


class BidirectionalEngineAdapter(EngineAdapter):
    """Optional active-control plane for a verified runtime extension.

    An adapter implementing this abstract class must set the corresponding
    capability bits to True in ``capabilities``; the bridge runs an
    ``isinstance`` check before calling, and closed APIs do not implement
    this class, so the bridge will not call them by mistake.
    """

    def probe(self, ir: TelosIR, plan: EmitPlan) -> ProbeResult:
        """Ask the server: "Do you still have this prefix cached?"

        Returns a miss by default; concrete adapters override it. When it
        returns ``hit=True`` the bridge can skip the ``refresh`` request it
        was about to issue, saving one RTT.
        """
        return ProbeResult(hit=False)

    def evict_span(self, ir: TelosIR, start_block: int, end_block: int) -> Mapping[str, Any]:
        """Explicitly evict a span of KV blocks; returns the ``cache_policy``
        fragment to carry along with the next emit.

        The bridge calls this during a ``Fold``: the server releases the KV
        of the old span, and the next request only recomputes the much
        shorter summary tail."""
        return {}

    def fork_and_replace(
        self,
        ir: TelosIR,
        path_hash: str,
        replace_suffix: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Request a radix fork + suffix replacement from an extension.

        Effect: keep the prefix KV corresponding to ``path_hash`` unchanged,
        and replace the span after it with ``replace_suffix`` (typically a
        short summary). A concrete adapter must document the exact server
        contract and recomputation semantics before enabling the capability.
        """
        return {}
