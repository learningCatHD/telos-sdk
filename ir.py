"""TELOS IR: the one data structure passed between the three layers.

Design principles
-----------------
1. **Immutable** -- every dataclass is frozen; the bridge's "modify" actions return a new
   IR rather than mutating bytes on the original object. This way, even if an upstream agent
   sends the same IR to multiple bridges, there is no shared-state write race.
2. **Narrow fields** -- the fewer the fields, the harder they are to misuse. Any
   engine-specific knob is kept out of the IR; the engine adapter decides it at emit time.
3. **Three-color, all-or-nothing** -- every block must land in exactly one of
   ``pin / fold / drop``; there is no gray "both cacheable and non-cacheable" state.

Invariants (re-checked by the bridge before and after every primitive call, see ``bridge.py``)
----------------------------------------------------------------------------------------------
**§5 order invariant**: within each segment (``tools`` / ``system`` / each ``message``),
blocks must be physically ordered ``pin* → fold* → drop*``. Exception: ``tool_result``
blocks in a ``message`` segment always come first (``tool_result* → pin* → fold* → drop*``)
-- Anthropic requires the tool_result of a user message to be physically first, and this hard
protocol constraint takes priority over band ordering. The ``tools`` / ``system`` segments
contain no tool_result and are unaffected.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Iterable, Literal, Mapping


# ---------------------------------------------------------------------------
# Band (three-color band)
# ---------------------------------------------------------------------------

class Band(str, Enum):
    """The cache lifecycle category of each block.

    - ``PIN``: long-lived, stable segment. Tool definitions, system prompt, the user's
      current question. Requests 1h TTL by default (Anthropic) / 24h retention (OpenAI).
    - ``FOLD``: cacheable but discardable on compaction. Assistant answers, tool_result,
      large ref-pool documents. Requests 5m TTL by default.
    - ``DROP``: never enters the cache hash. Timestamps, cwd, git status,
      ``<system-reminder>`` envelopes. Must appear at the very end of its segment.
    """

    PIN = "pin"
    FOLD = "fold"
    DROP = "drop"


_BAND_RANK: Mapping[Band, int] = {Band.PIN: 0, Band.FOLD: 1, Band.DROP: 2}


# ---------------------------------------------------------------------------
# Block / Message / IR
# ---------------------------------------------------------------------------

BlockKind = Literal[
    "text",
    "tool_def",
    "tool_use",
    "tool_result",
    "image",
    "thinking",
]

EngineName = Literal["anthropic", "openai", "deepseek", "vllm", "sglang"]
ReuseScope = Literal["request", "session", "project", "tenant"]
RetentionClass = Literal["hot", "warm", "cold", "drop"]
Sensitivity = Literal["private", "project", "tenant"]


@dataclass(frozen=True)
class LogicalPosition:
    """Engine-neutral block position; token boundaries are resolved downstream."""

    segment: Literal["tools", "system", "message"]
    index: int
    message_index: int | None = None


@dataclass(frozen=True)
class CacheBoundary:
    """A logical cache boundary and its advisory retention class."""

    name: str
    end: LogicalPosition
    band: Band
    retention: RetentionClass
    expected_reuses: int = 0


@dataclass(frozen=True)
class CacheIntent:
    """Advisory cache policy for a capability-negotiated engine extension.

    Stock adapters deliberately do not serialize this object. ``namespace``
    must come from trusted gateway identity, never from user prompt text.
    """

    schema_version: int
    namespace: str
    reuse_scope: ReuseScope
    sensitivity: Sensitivity
    boundaries: tuple[CacheBoundary, ...]
    next_use_distance: int | None = None
    branch_parent: str | None = None


@dataclass(frozen=True)
class TelosBlock:
    """The smallest unit of content in TELOS.

    A block is equivalent to "a cache-computation granule the engine treats as
    indivisible". In most cases this maps one-to-one with an Anthropic content block
    or an OpenAI message-piece.
    """

    id: str                       #: stable identifier within a session (for diagnostics / references)
    band: Band
    kind: BlockKind
    payload: Any                  #: engine-agnostic content; translated by the adapter at emit time
    ref_slug: str | None = None   #: if non-empty, this block comes from the ref-pool (see §4)
    source_tag: str | None = None #: diagnostic field: which harness rule assigned it to this band
    extra: Mapping[str, Any] = field(default_factory=dict)
    """Holds stable side information the engine may need, e.g. an image's ``detail`` field.

    Any field that *must* enter the cache hash should be written into ``extra`` and
    serialized by the adapter at emit time; the harness must never inject such a field
    at emit time, otherwise the byte stability guaranteed by §5 is broken.
    """


@dataclass(frozen=True)
class TelosMessage:
    """A single conversation message (user / assistant).

    Corresponds to one OpenAI / Anthropic message, but its internal blocks **must**
    be ordered per §5: ``tool_result*`` first (required by the Anthropic protocol),
    followed by ``pin* → fold* → drop*``. The most common case is a user message
    forced into ``(tool_result: previous turn's tool output) + (pin: user question) +
    (fold: history echo / [ref:...] references) + (drop: harness envelope)``.
    """

    role: Literal["system", "user", "assistant"]
    blocks: tuple[TelosBlock, ...]


@dataclass(frozen=True)
class TelosIR:
    """The one transport object passed between harness → bridge → engine."""

    session_id: str
    tools: tuple[TelosBlock, ...]                  #: all band=PIN (schema does not change)
    system: tuple[TelosBlock, ...]                 #: pin* → fold* (the fold part contains the ref-pool) → drop*
    messages: tuple[TelosMessage, ...]
    ref_pool: Mapping[str, TelosBlock]             #: slug → block
    hints: "TelosHints" = field(default_factory=lambda: TelosHints())
    cache_intent: CacheIntent | None = None


@dataclass(frozen=True)
class TelosHints:
    """Non-binding metadata the engine adapter uses to make plan decisions."""

    engine: EngineName = "anthropic"
    model: str = ""
    expected_turns: int = 0       #: the harness's estimate of total turns; affects the mid-rolling anchor toggle


# ---------------------------------------------------------------------------
# Usage report flowing back after engine output (§9)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UsageReport:
    """Unified usage return format.

    Aligned with the ``raw_input / cache_read / cache_write`` triple schema of
    ``benchmark/scripts/compute-metrics.py`` -- any engine's raw ``usage`` fields must be
    normalized to these three numbers, otherwise the north-star metric cannot be computed.
    """

    raw_input: int                 #: input tokens that neither hit nor wrote the cache
    cache_read: int                #: read from the cache
    cache_write: int               #: new tokens written to the cache by this request
    output: int
    raw: Mapping[str, Any] = field(default_factory=dict)  #: raw usage fields, kept for diagnostics


# ---------------------------------------------------------------------------
# §5 order invariant checks
# ---------------------------------------------------------------------------

class TelosInvariantError(ValueError):
    """Raised when any invariant of §5 / canonicalization / ref-pool registration is broken."""


def _order_key(blk: TelosBlock) -> tuple[int, int]:
    """The physical sort key for a block within a message.

    ``tool_result`` blocks must be physically first -- this is a hard Anthropic protocol
    constraint (a user message's tool_result must come before all other content), taking
    priority over the band cache-lifecycle ordering. The rest of the blocks follow
    ``pin* → fold* → drop*``.
    """
    return (0 if blk.kind == "tool_result" else 1, _BAND_RANK[blk.band])


def enforce_band_order(blocks: Iterable[TelosBlock]) -> tuple[TelosBlock, ...]:
    """Stably sort blocks into ``tool_result* → pin* → fold* → drop*``.

    When a harness assembles a message it often iterates in ``content[]`` source order,
    with each content item expanding into its own (PIN, FOLD*, DROP*) subsequence; a plain
    ``extend`` interleaves bands (PIN, DROP, PIN, DROP, ...) and violates §5. The harness
    should call this function once at the message level as a safety net, then freeze the
    result into ``TelosMessage.blocks``.

    ``tool_result`` blocks are always sorted first regardless of band -- Anthropic requires
    a user message's tool_result to be physically first, and placing it after text gets the
    request rejected with a 400.

    Stability guaranteed: the incoming order is preserved within each group -- this matters
    for the readable semantics of "question A before question B" and must not jump around.
    """
    return tuple(sorted(blocks, key=_order_key))


def assert_band_order(blocks: tuple[TelosBlock, ...], where: str) -> None:
    """Assert that blocks satisfy the strict ``tool_result* → pin* → fold* → drop*`` order.

    Complexity is O(n), a single scan; the bridge runs it before and after every primitive
    call, so the cost is negligible -- this is the "safety gate" of the whole protocol.

    ``tool_result`` blocks are treated as rank ``-1``: they must come before all
    non-tool_result blocks (required by the Anthropic protocol), and appearing after text is
    flagged as a violation. The ``tools`` / ``system`` segments contain no tool_result, so
    the check behaves exactly as before.
    """
    last_rank = -2
    for blk in blocks:
        rank = -1 if blk.kind == "tool_result" else _BAND_RANK[blk.band]
        if rank < last_rank:
            raise TelosInvariantError(
                f"Band order violated in {where}: block {blk.id!r} "
                f"(kind={blk.kind!r}, band={blk.band.value!r}) appears after a "
                f"higher-rank block. Required order is "
                f"tool_result* -> pin* -> fold* -> drop*."
            )
        last_rank = rank


def assert_ir_invariants(ir: TelosIR) -> None:
    """Run a full §5 check over the entire IR."""
    assert_band_order(ir.tools, "tools")
    if any(b.band is not Band.PIN for b in ir.tools):
        raise TelosInvariantError("All blocks in `tools` must have band=PIN")
    assert_band_order(ir.system, "system")
    for i, msg in enumerate(ir.messages):
        assert_band_order(msg.blocks, f"messages[{i}] (role={msg.role})")
    if ir.cache_intent is not None:
        _assert_cache_intent(ir, ir.cache_intent)


def _assert_cache_intent(ir: TelosIR, intent: CacheIntent) -> None:
    if intent.schema_version != 1:
        raise TelosInvariantError(
            f"Unsupported CacheIntent schema_version={intent.schema_version!r}"
        )
    namespace = intent.namespace
    if (not namespace or len(namespace) > 256
            or any(ord(ch) < 33 or ord(ch) == 127 for ch in namespace)):
        raise TelosInvariantError(
            "CacheIntent namespace must be a non-empty opaque token without whitespace/control characters"
        )
    if intent.reuse_scope not in {"request", "session", "project", "tenant"}:
        raise TelosInvariantError(f"Invalid CacheIntent reuse_scope={intent.reuse_scope!r}")
    if intent.sensitivity not in {"private", "project", "tenant"}:
        raise TelosInvariantError(f"Invalid CacheIntent sensitivity={intent.sensitivity!r}")
    if intent.next_use_distance is not None and intent.next_use_distance < 0:
        raise TelosInvariantError("CacheIntent next_use_distance must be non-negative")

    seen_names: set[str] = set()
    for boundary in intent.boundaries:
        if not boundary.name or boundary.name in seen_names:
            raise TelosInvariantError(
                f"CacheIntent boundary names must be non-empty and unique: {boundary.name!r}"
            )
        seen_names.add(boundary.name)
        if boundary.expected_reuses < 0:
            raise TelosInvariantError(
                f"CacheIntent boundary {boundary.name!r} has negative expected_reuses"
            )
        target = _block_at_position(ir, boundary.end)
        if target.band is not boundary.band:
            raise TelosInvariantError(
                f"CacheIntent boundary {boundary.name!r} declares band={boundary.band.value!r} "
                f"but points to band={target.band.value!r}"
            )


def _block_at_position(ir: TelosIR, pos: LogicalPosition) -> TelosBlock:
    if pos.index < 0:
        raise TelosInvariantError("CacheIntent boundary index must be non-negative")
    if pos.segment == "tools":
        if pos.message_index is not None:
            raise TelosInvariantError("tools boundary cannot have message_index")
        blocks = ir.tools
    elif pos.segment == "system":
        if pos.message_index is not None:
            raise TelosInvariantError("system boundary cannot have message_index")
        blocks = ir.system
    elif pos.segment == "message":
        if pos.message_index is None:
            raise TelosInvariantError("message boundary requires message_index")
        if not 0 <= pos.message_index < len(ir.messages):
            raise TelosInvariantError(
                f"CacheIntent message_index {pos.message_index} is out of range"
            )
        blocks = ir.messages[pos.message_index].blocks
    else:
        raise TelosInvariantError(f"Invalid CacheIntent segment={pos.segment!r}")
    if pos.index >= len(blocks):
        raise TelosInvariantError(
            f"CacheIntent {pos.segment} index {pos.index} is out of range"
        )
    return blocks[pos.index]


# ---------------------------------------------------------------------------
# Convenience constructors (commonly used by harness plugins, so kept in the IR module itself)
# ---------------------------------------------------------------------------

def with_messages(ir: TelosIR, messages: tuple[TelosMessage, ...]) -> TelosIR:
    """Return a new IR with ``messages`` replaced; convenient for functional-style bridge operations."""
    return replace(ir, messages=messages)


def with_ref_pool(ir: TelosIR, ref_pool: Mapping[str, TelosBlock]) -> TelosIR:
    return replace(ir, ref_pool=dict(ref_pool))
