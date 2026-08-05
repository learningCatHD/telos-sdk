# TELOS — Architecture Design and Code Implementation Reference

> This is the authoritative architecture document for the `telos-sdk` repository (the Python `telos` package). It covers the design philosophy,
> the three-layer structure, the core data structures, the code implementation of each module, the two integration paths, the RTK filtering layer,
> recording/replay comparison, and all invariants and extension points.
>
> - Want to get started → [playbook.md](playbook.md), [User-guide.md](User-guide.md)
> - Want the protocol spec → [2026-05-06-telos-protocol.md](2026-05-06-telos-protocol.md)
> - Want the local KV control-plane proposal → [telos-sglang-hicache-lmcache-design.md](telos-sglang-hicache-lmcache-design.md)
> - Want the change history → [../CHANGELOG.md](../CHANGELOG.md)
>
> Last updated: 2026-08-05

---

## Table of Contents

1. [Design Philosophy](#1-design-philosophy)
2. [Repository Map](#2-repository-map)
3. [Three-Layer Architecture Overview](#3-three-layer-architecture-overview)
4. [Core Data Structure — TelosIR](#4-core-data-structure--telosir)
5. [Bridge — Policy Core](#5-bridge--policy-core)
6. [Harness Layer](#6-harness-layer)
7. [Engine Layer](#7-engine-layer)
8. [ref-pool](#8-ref-pool)
9. [Integration Paths A / B](#9-integration-paths-a--b)
10. [RTK Output Filtering Layer](#10-rtk-output-filtering-layer)
11. [Recording and Replay Comparison](#11-recording-and-replay-comparison)
12. [Observability](#12-observability)
13. [Invariants and Design Constraints](#13-invariants-and-design-constraints)
14. [End-to-End Data Flow](#14-end-to-end-data-flow)
15. [Extension Points](#15-extension-points)

---

## 1. Design Philosophy

### 1.1 The Problem to Solve

The KV cache of LLM inference can retain the computed results of "recurring prefixes"; on a hit,
the input token price drops to ~10% (Anthropic). But an agent's multi-turn conversations are by default **not**
cache-friendly: in each round's request, the slightest jitter in the concatenation order of the system prompt, tool definitions, and conversation history (JSON keys out of order, tool array order changing, timestamps mixed into the prefix)
changes the prefix hash, and the entire cache is invalidated.

The whole value of TELOS is one sentence: **stabilize the parts that are genuinely stable, so they keep hitting the KV
cache.** All other complexity is cut away.

### 1.2 The "Stone Tablet" Metaphor

TELOS = **S**table prefix · **T**iered bands · **E**phemeral tail ·
**L**ayered adapters · **A**nchored marks.

The inscription on the base of a stone tablet (the durable prefix) is carved once and used for a lifetime; the inscriptions added on top over time
(new content each round) can be erased and rewritten at any time, but never touch the base. The entire value of the KV cache is to
keep the base preserved.

### 1.3 Three Core Design Decisions

1. **The three-color band (Band) is a clean cut.** Every content block must land in one of `PIN` / `FOLD` / `DROP`;
   there is no gray "both cacheable and non-cacheable" state.
2. **The ordering invariant is the only hard constraint.** Within each segment, blocks must be physically arranged as
   `pin* → fold* → drop*` (protocol §5). A violation immediately raises `TelosInvariantError`.
   Everything else is a soft recommendation.
3. **The three layers only pass values downward, never reference backward.** The harness does not know the engine, the engine does not know
   the harness; the IR in the middle is the only contract. Cross-request state is only allowed to exist in the
   `BridgeSessionState` held by the Bridge.

### 1.4 The Orthogonal Second Optimization Line — RTK

What TELOS stabilizes is the **request prefix** (system / tools / conversation prefix). But each round, the agent also
appends large chunks of tool output (bash / pytest / docker logs) to the tail of the conversation. TELOS absorbed
the ideas of [rtk-ai/rtk](https://github.com/rtk-ai/rtk) and added an orthogonal
**RTK output filtering** layer: before a request enters the TELOS pipeline, it compresses away the large repetitive
output in `tool_result`. The two lines are independently controlled by the `TelosMode` four-state switch (see §10).

---

## 2. Repository Map

```
telos-sdk/                         (Python package name = telos; pyproject maps the root directory to telos)
│
├── ir.py                  Core data structures: Band / TelosBlock / TelosMessage / TelosIR / UsageReport
├── bridge.py              Policy core: 5 primitives + canonicalize + BridgeSessionState
├── refpool.py             ref-pool: the "pointer table" for large content, slug freezing
├── registry.py            Factory that loads harness / engine by name
├── cli.py                 `telos` unified CLI: proxy / init / dashboard / replay
├── corpus.py              Session corpus: records raw requests for replay
│
├── harness/               Layer 1: upstream agent request → TelosIR
│   ├── base.py            HarnessPlugin ABC
│   ├── _user_split.py     user text envelope splitting (PIN/FOLD/DROP)
│   ├── openclaw.py        OpenClaw (Anthropic /v1/messages shape)
│   ├── hermes.py          Hermes / Claude Code (Anthropic shape + different envelope)
│   └── telos.py           Telos (OpenAI ChatCompletions shape)
│
├── engine/                Layer 3: TelosIR → each engine's wire request
│   ├── base.py            EngineAdapter / BidirectionalEngineAdapter / EngineCapabilities
│   ├── anthropic.py       AnthropicAdapter (the only one with explicit BP support)
│   ├── openai.py          OpenAIAdapter (layout + routing key)
│   ├── deepseek.py        DeepSeekAdapter (zero control plane)
│   ├── vllm.py            VLLMAdapter (bidirectional)
│   ├── openai_chat.py     Shared stock Chat Completions renderer
│   └── sglang.py          SGLangAdapter (stock RadixAttention/HiCache-safe)
│
├── output_filter/         RTK-style tool result filtering layer (orthogonal to TELOS)
│   ├── mode.py            TelosMode four-state switch
│   ├── filters.py         ToolResultFilter / RtkFilter / FallbackFilter / CompositeFilter
│   └── preprocess.py      apply_filter: rewrites tool_result in the raw request
│
├── proxy/                 Integration path B: HTTP reverse proxy
│   ├── server.py          aiohttp reverse proxy (SSE-aware)
│   ├── pipeline.py        process_anthropic_request: parse→bridge→emit pure function
│   ├── inspector.py       SessionInspector: in-memory diagnostic snapshot store
│   └── __main__.py        `python -m telos.proxy` entry point
│
├── deploy/sglang/         Stock SGLang + HiCache launch/config templates
├── scripts/               Live SGLang A/B and HiCache restore contract runners
│
├── replay/                Recording → replay comparison engine
│   ├── __init__.py        replay_session engine
│   └── __main__.py        `telos replay` CLI
│
├── init/                  Integration path B installers
│   ├── base.py            AgentInstaller ABC + InstallResult
│   ├── claude_code.py     patch ~/.claude/settings.json
│   ├── generic.py         prints export instructions
│   └── __main__.py        `python -m telos.init` entry point
│
├── scripts/
│   ├── telos_anthropic_transport.py   Integration path A (Anthropic shape) + _detect_harness
│   ├── telos_transport.py             Integration path A (OpenAI shape)
│   ├── build_savings_dashboard.py     Live savings dashboard
│   ├── build_developer_page.py        Developer inspector page
│   └── show_prompt_trace.py           Terminal pretty-printer for prompt_trace.jsonl
│
├── tests/                 Test suite (77 test functions)
└── docs/                  Design and usage documentation
```

---

## 3. Three-Layer Architecture Overview

```
Upstream agent (Claude Code / OpenClaw / Hermes / in-house)
    │  Raw request (Anthropic /v1/messages or OpenAI ChatCompletions)
    ▼
┌─────────────────────────────────────────────────────────┐
│ Layer 1 · HARNESS (stateless pure function)               │
│   harness.parse(raw) → TelosIR                           │
│   Responsibilities: envelope splitting, large docs into ref-pool, content banding │
└──────────────────────────┬────────────────────────────────┘
                           ▼  TelosIR
┌─────────────────────────────────────────────────────────┐
│ Layer 2 · BRIDGE (stateful, one per session)              │
│   5 primitives: place / pin / mark / fold / refresh       │
│   canonicalize (key sorting, tool sorting) + §5 invariant check │
│   holds BridgeSessionState (ref-pool, R8 counter)         │
└──────────────────────────┬────────────────────────────────┘
                           ▼  TelosIR (rewritten) + EmitPlan
┌─────────────────────────────────────────────────────────┐
│ Layer 3 · ENGINE (stateless, capability-aware)            │
│   plan_marks(ir) → EmitPlan                              │
│   emit(ir, plan) → wire request                          │
│   parse_usage(response) → UsageReport                    │
└──────────────────────────┬────────────────────────────────┘
                           ▼  Engine-native wire request
Real LLM service (Anthropic / OpenAI / DeepSeek / vLLM / SGLang)
```

**Core invariant**: Cross-request state can only exist in Layer 2. Both the harness and the engine are pure
functions / stateless objects; the same input always produces the same result.

`registry.py` provides `load_harness(name)` / `load_engine(name)`, so the bridge
does not directly import any concrete implementation — this is how "the three layers only pass values downward" is realized at the code level.

---

## 4. Core Data Structure — TelosIR

Defined in [ir.py](../ir.py). All dataclasses are **frozen (immutable)**: the bridge's
"modifications" return a new IR rather than mutating bytes on the original object, avoiding write races on shared state.

### 4.1 Band — The Three-Color Band

```python
class Band(str, Enum):
    PIN  = "pin"    # Long-lived stable segment: tool definitions, system prompt, the user's current question
    FOLD = "fold"   # Cacheable but droppable on compact: assistant replies, tool_result, ref-pool large docs
    DROP = "drop"   # Never enters the cache hash: timestamp, cwd, git status, envelope; must be at the segment end
```

The sorting weights are `_BAND_RANK = {PIN: 0, FOLD: 1, DROP: 2}`.

### 4.2 TelosBlock — The Smallest Content Unit

| Field | Type | Description |
|---|---|---|
| `id` | `str` | Stable identifier within the session (for diagnostics / referencing) |
| `band` | `Band` | Three-color band membership |
| `kind` | `BlockKind` | `text` / `tool_def` / `tool_use` / `tool_result` / `image` / `thinking` |
| `payload` | `Any` | Engine-agnostic content; translated by the adapter at emit time |
| `ref_slug` | `str \| None` | Non-null means this block comes from the ref-pool |
| `source_tag` | `str \| None` | Diagnostic field: which harness rule banded it |
| `extra` | `Mapping` | Stable side information needed by the engine (e.g. the image's `detail`, a tool's `source`/`mcp_server`) |

> **Key constraint**: Any field that must enter the cache hash must be written into `extra` and serialized by the adapter
> together at emit time. The harness must never inject a field at emit time, otherwise byte stability
> is broken.

### 4.3 TelosMessage / TelosIR / TelosHints

```python
@dataclass(frozen=True)
class TelosMessage:
    role: Literal["system", "user", "assistant"]
    blocks: tuple[TelosBlock, ...]      # Must satisfy §5: pin* → fold* → drop*

@dataclass(frozen=True)
class TelosIR:
    session_id: str
    tools:    tuple[TelosBlock, ...]    # All band=PIN
    system:   tuple[TelosBlock, ...]    # pin* → fold*(incl. ref-pool) → drop*
    messages: tuple[TelosMessage, ...]
    ref_pool: Mapping[str, TelosBlock]  # slug → block
    hints:    TelosHints

@dataclass(frozen=True)
class TelosHints:
    engine: Literal["anthropic", "openai", "deepseek"] = "anthropic"
    model:  str = ""
    expected_turns: int = 0             # Affects whether the mid-rolling anchor is enabled
```

### 4.4 UsageReport — Normalized Usage

The raw `usage` fields of all engines are normalized to these four numbers, otherwise cost metrics cannot be computed:

```python
@dataclass(frozen=True)
class UsageReport:
    raw_input:   int   # Input tokens that were neither hit nor written to cache
    cache_read:  int   # Read from cache
    cache_write: int   # New tokens written to cache by this request
    output:      int
    raw: Mapping       # Raw usage fields, kept for diagnostics
```

### 4.5 §5 Validation Functions

- `enforce_band_order(blocks)` — stably sorts into `pin* → fold* → drop*`
  (preserves insertion order within the same band). Called as a fallback when the harness assembles a message.
- `assert_band_order(blocks, where)` — O(n) single-pass scan; raises `TelosInvariantError`
  as soon as a rank regression is found.
- `assert_ir_invariants(ir)` — runs full validation over the entire IR; additionally requires that `tools`
  are all `band=PIN`.

---

## 5. Bridge — Policy Core

Defined in [bridge.py](../bridge.py). One instance per session, **stateful**.

### 5.1 The Five Primitives

| Primitive | Method | Effect | Protocol Section |
|---|---|---|---|
| **Place** | `place(segment, blocks)` / `append_message(msg)` | Replace all blocks of a segment / append a message, immediately running the §5 check | §6.1 |
| **Pin** | `pin(slug, payload)` | Register a ref-pool entry; the slug is frozen immediately | §6.2 |
| **Mark** | `mark()` | Delegate to the engine adapter to decide the cache anchor positions for this emit (returns an `EmitPlan`) | §6.3 |
| **Fold** | `fold(slugs=, message_range=, summary=)` | Fold a ref-pool entry (swap payload only, not slug), or fold a stretch of history messages into a summary | §6.4 |
| **Refresh** | `refresh(plan)` | Trigger engine keep-alive; adaptive gating (see §5.4) | §6.5 |

> Note: `pin()` registers a foldable entry with `band=FOLD`. The primitive name "Pin" refers to
> "fixing this large content in the ref-pool and giving it a stable pointer", **not** `band=PIN`.

### 5.2 The emit Flow

```python
def emit_with_plan(self) -> tuple[wire, EmitPlan]:
    canon = _canonicalize_ir(self._ir)        # ① Canonicalize
    assert_ir_invariants(canon)               # ② Full §5 check (last line of defense)
    refpool.lint_blocks(...)                  # ③ Scan [ref:slug] references, fail-fast on unregistered ones
    plan = engine.plan_marks(canon)           # ④ Engine decides the anchor positions
    wire = engine.emit(canon, plan)           # ⑤ Engine produces the wire
    stats.real_requests_since_refresh += 1
    return wire, plan
```

`emit()` is the single-return version of `emit_with_plan()`. **Callers must go through `emit*`, not
call `engine.emit(ir, plan)` directly** — otherwise `_canonicalize_ir` is skipped.

### 5.3 Canonicalization (fixes R5)

Common across engines, must be done uniformly before emit. Root cause: the JSON
serialization of Swift / Go randomizes key order, causing prefix hash drift and cache invalidation.

- `_canonicalize_payload` — sorts dict keys lexicographically (recursive).
- `_canonicalize_schema` — dedicated to JSON-Schema subtrees: besides key sorting, it also sorts
  **set-semantics arrays**. Currently only `required` (`_SCHEMA_SET_ARRAY_KEYS`).
  Deliberately **does not** sort `enum` / `examples` / `anyOf` / `oneOf` / `allOf` (order is semantic).
- `_canonicalize_tool_def` — recognizes the two tool shapes, Anthropic (`input_schema`) and OpenAI
  (`function.parameters`); the schema subtree goes through schema canonicalization.
- `_tool_sort_key` — the stable sort key for the tool array, `(source_rank, mcp_server, name)`.
  `source_rank`: builtin(0) → mcp(1) → user(2) → unmarked(3). Guarantees that a multi-MCP
  server startup race does not interleave insertions between two servers and break the prefix.

> The payload of `tool_use` / `tool_result` is **user data**; only key sorting is done,
> array order is never touched (a field that happens to be named `required` in a payload must not be silently reordered).

### 5.4 BridgeSessionState — Cross-Turn State

```python
@dataclass
class BridgeSessionState:
    refpool: RefPool                    # slug registry, persists across turns once frozen
    stats: _SessionStats                # cumulative_cache_creation + real_requests_since_refresh
    sticky_harness: str | None = None   # The harness identified in the first turn, locked and reused
    sticky_mode: str | None = None      # The mode of the first turn (none/telos/rtk/both), locked
    compare_group: str | None = None    # Comparison-experiment grouping label
```

The upstream (proxy / SDK transport) holds one copy per session_id and passes it in when constructing a new `Bridge`
each round. If not passed, the Bridge news up its own one, degrading to "independent per round".

`REFRESH_THRESHOLD = 11` (Janus §6.3.1): **R8 adaptive gating** — if the number of
real requests within the renewal window is below the threshold, refresh is skipped, letting the cache expire naturally and avoiding a renewal
cost > benefit for low-activity sessions.

### 5.5 Bidirectional Operations (extension adapters only)

`is_bidirectional` = `isinstance(engine, BidirectionalEngineAdapter)`.

| Bridge Method | Stock/closed adapter | Capability-verified extension adapter |
|---|---|---|
| `probe_cache()` | Returns `ProbeResult(hit=False)` | May issue a verified lookup asking "is the prefix still in cache" |
| `cooperative_fold(...)` | Equivalent to `fold()` + returns `{}` | Client-side fold + returns a negotiated `evict_span` / `fork_and_replace` fragment |
| `emit_with_extras(extras)` | Merges the fragments into `plan.extras` then emits | Same as left |

Stock SGLang does not implement this interface. Its Fold path changes the suffix and lets RadixAttention
naturally reuse only the token-identical prefix before the branch; the compacted suffix still requires Prefill.

---

## 6. Harness Layer

Layer 1. Each harness plugin is a **pure stateless function**: `parse(raw) → TelosIR`.
The signature of `HarnessPlugin.parse` in the base class [harness/base.py](../harness/base.py):

```python
def parse(self, raw_request, *, session_id, engine, model="", expected_turns=0) -> TelosIR
```

### 6.1 Banding Rules (common to the three harnesses)

| Upstream content | Band | Description |
|---|---|---|
| `tools[]` | PIN | `kind=tool_def` |
| `system` text (≤ 2048 chars) | PIN | |
| `system` text (> 2048) / `<file>` block | FOLD (into ref-pool) + PIN reference stub | Threshold `_REFPOOL_THRESHOLD = 2048` |
| user text | → `split_user_text` (PIN/FOLD/DROP) | See §6.2 |
| user `tool_result` | FOLD | |
| assistant text / `tool_use` / `thinking` | FOLD | |

After each message is assembled, it is passed once through `enforce_band_order` (openclaw / hermes call it explicitly;
telos does not call it because its construction order is naturally compliant).

### 6.2 `_user_split.split_user_text`

Splits one user text into PIN (the actual question) + FOLD (history echo) + DROP (envelope).
The regex set:

- **DROP** (the envelope that changes each round): `<environment_info>` / `<system-reminder>`
  / `<command-message>` / `<command-name>` / `Current time:...`
- **FOLD** (explicitly wrapped history echo): `<prev>...</prev>`
- **PIN**: whatever remains after stripping the above is the user's question.

The returned tuple is already in §5 order and can be put directly into a `TelosMessage`.

### 6.3 Differences Among the Three Harnesses

| | OpenClaw | Hermes | Telos |
|---|---|---|---|
| Wire shape | Anthropic `/v1/messages` | Anthropic `/v1/messages` | OpenAI ChatCompletions |
| Identity markers | Default / fallback | `<system-reminder>`, `<command-message>`, thinking blocks, the Claude Code tool set | Standalone transport (OpenAI shape) |
| Tool classification | `_classify_anthropic_tool` (type prefix + server field + metadata) | Reuses `_classify_anthropic_tool` | `_classify_openai_tool` (metadata only) |
| `<file>` into ref-pool | No (the whole oversized system item goes into the pool, slug `system-doc-{i}`) | Yes (`<file path=...>`, dotted slug) | Yes (dash slug) |
| `thinking` block | Not handled | FOLD `kind=thinking` | Not applicable |
| `tool_result` source | Embedded in the user message | Embedded in the user message | Standalone `role=tool` → wrapped into a user message |
| system extraction | The `system` field | The `system` field | The consecutive `role=system` at the start of `messages[]` |
| `source_tag` prefix | `openclaw/*` | `hermes/*` | `telos/*` |

Tool classification writes `source` (builtin / mcp / user) into `TelosBlock.extra`, feeding it to
the bridge's `_tool_sort_key` to protect PIN prefix stability.

### 6.4 `_detect_harness` (in [scripts/telos_anthropic_transport.py](../scripts/telos_anthropic_transport.py))

For Anthropic-shaped traffic, picks between `hermes` / `openclaw`. Detection order
(returns on the first hit):

0. **HTTP header fingerprint** — `User-Agent` containing `claude-cli` or `x-app: cli` → `hermes`.
   Only the proxy path has request headers. This is a per-client signal that content detection cannot obtain
   yet is reliable for every request: the auxiliary requests Claude Code sends with Haiku (title generation / topic detection)
   have neither tools nor markers, so 1–3 below all miss; only the header fingerprint can correctly identify them.
1. **Envelope markers** — run paired
   open/close regexes (`system-reminder` / `command-message` / `command-name`) in the system text **and in every user text block**.
   Paired matching avoids a misclassification where the user is merely discussing tags in prose. Hit → `hermes`.
2. **thinking block** — any assistant message containing a `thinking` content block → `hermes`.
3. **Tool fingerprint** — the intersection of tool names with `{Bash, Edit, Read, Write, Grep, Glob,
   TodoWrite, Task, WebFetch, WebSearch, NotebookEdit}` ≥ 3 → `hermes`.
   Catches the first-turn request (before the reminder is injected).
4. Fallback → `openclaw`.

`_detect_harness_signal` returns a **confident signal** (a hit on 0–3) or `None` (will only fall back).
The proxy uses it for **per-client harness memory** (`ProxyApp._client_harness`, keyed by
`_client_identity`): once a client is confidently identified by any request, it is remembered; its subsequent
signal-less requests (such as tool-less auxiliary requests) inherit it directly and are not misclassified as `openclaw`.

The SDK transport path has no HTTP headers; the detection result is sticky within the session (written into
`BridgeSessionState.sticky_harness`), avoiding re-probing per call which would cause the harness to
flip and the prefix to destabilize.

---

## 7. Engine Layer

Layer 3. Base class [engine/base.py](../engine/base.py). The bridge programs only against the abstract interface
and never branches by engine name.

### 7.1 Interface Contract

`EngineAdapter` (ABC) has four members:
- `capabilities` (property) → `EngineCapabilities`
- `plan_marks(ir) → EmitPlan` — decides the anchor positions
- `emit(ir, plan) → wire dict`
- `parse_usage(response) → UsageReport`
- `refresh(ir, plan)` — optional keep-alive; the base class defaults to a no-op

`MarkSlot` (an engine-agnostic logical cache anchor): `name` / `segment` / `index` /
`message_index` / `ttl_class`. The bridge only sees the slot list and never knows what
`cache_control` looks like.

`BidirectionalEngineAdapter` additionally adds a read path + explicit server-side state mutation:
`probe` / `evict_span` / `fork_and_replace`. Closed-source APIs do not inherit it; the bridge
relies on `isinstance` to guarantee it never calls them by mistake.

### 7.2 Capability Matrix

> The SGLang column is the stock-safe default. vLLM remains a prototype private-extension adapter;
> its private `cache_policy` contract must not be used against an unpatched stock runtime. The
> [local KV control-plane design](telos-sglang-hicache-lmcache-design.md) defines
> `native` / `config_only` / `extension` / `unsupported` levels.

| Field | Anthropic | OpenAI | DeepSeek | vLLM | SGLang |
|---|:---:|:---:|:---:|:---:|:---:|
| `explicit_breakpoints` | ✓ | ✗ | ✗ | ✓ | ✗ |
| `max_breakpoints` | **4** | 0 | 0 | 2 | 0 |
| `ttl_control` | presets(5m/1h) | presets(in-memory/24h) | none | none | none |
| `prewarmable` | ✓(`max_tokens:0`) | ✗ | ✗ | ✓(`max_tokens:1`) | ✗ |
| `routing_key` | ✗ | ✓(`prompt_cache_key`) | ✗ | ✓(`cache_salt`) | ✗ |
| `cache_probe` | ✗ | ✗ | ✗ | ✓ | ✗ |
| `span_eviction` | ✗ | ✗ | ✗ | ✓ | ✗ |
| `fork_and_replace` | ✗ | ✗ | ✗ | ✗ | ✗ |
| `tier_hint` | ✗ | ✗ | ✗ | ✗ | ✗ |
| Bidirectional class | No | No | No | Yes (prototype) | No |
| `cache.prefix_reuse` | — | — | — | — | native |
| `cache.cache_report` | — | — | — | — | config_only |
| `cache.tier_report` | — | — | — | — | native |
| `cache.hierarchical_storage` | — | — | — | — | config_only |

### 7.3 Each Engine's emit Strategy

- **Anthropic** ([anthropic.py](../engine/anthropic.py)) — the only one with explicit
  breakpoint support. `plan_marks` produces at most 4 candidate slots: **BP-T** (end of the tools segment),
  **BP-S** (the last PIN of system), **BP-R** (the last FOLD of system =
  end of the ref-pool), **BP-X** (the last non-DROP block of the most recent message, 5m rolling),
  **BP-mid** (when messages ≥ 19, adds an anchor at `len-19`, fixes R2). If more than 4,
  truncates by the R7 priority `BP-T < BP-S < BP-R < BP-mid < BP-X`. At emit time, `cache_control` is attached to the block
  at the slot's landing point (5m `{"type":"ephemeral"}` /
  1h adds `"ttl":"1h"`); DROP blocks are not attached and must come after all BPs. Constants
  `_LOOKBACK=20`, `_MID_ANCHOR_STRIDE=19`.
- **OpenAI** ([openai.py](../engine/openai.py)) — no explicit BP. `plan_marks`
  only produces a `routing_key` (`telos-<sha256[:16]>`, hashing tools + PIN system + ref-pool
  keys) + retention. emit arranges blocks as non-DROP→DROP so OpenAI's
  automatic prefix matching hits a stable head; writes `prompt_cache_key` / `prompt_cache_retention`.
  `cache_write` is always 0 (OpenAI's implicit cache write is not billed separately). `KEY_RPM_SOFT_CAP=12`.
- **DeepSeek** ([deepseek.py](../engine/deepseek.py)) — zero control plane. The disk
  context cache is always on. `plan_marks` returns an empty `EmitPlan()`. emit relies only on arranging
  blocks as non-DROP→DROP so the exact-match prefix hits.
- **vLLM** ([vllm.py](../engine/vllm.py)) — prototype bidirectional adapter. emit writes
  `cache_salt` plus the private-extension field `cache_policy`
  (`pin_prefix_until_block` / `evict_span`); stock deployments must not receive the private field.
- **SGLang** ([sglang.py](../engine/sglang.py)) — stock OpenAI Chat Completions adapter.
  It emits no private cache fields. RadixAttention provides automatic prefix reuse;
  `--enable-cache-report` and HiCache/LMCache are server-startup configuration. When an
  upstream explicitly enables `tier_telemetry`, the adapter adds SGLang's public
  `return_cached_tokens_details` request field and normalizes response tiers
  `device/host/storage` to `gpu/cpu/l3`; missing fields remain unknown.

### 7.4 usage Parsing

| Engine | cache_read field | cache_write field |
|---|---|---|
| Anthropic | `cache_read_input_tokens` | `cache_creation_input_tokens` |
| OpenAI | `prompt_tokens_details.cached_tokens` | always 0 |
| DeepSeek | `prompt_cache_hit_tokens` | always 0 (write cost folded into the miss price) |
| vLLM | `cached_tokens` (version-dependent) | always 0 |
| SGLang | `prompt_tokens_details.cached_tokens` with `--enable-cache-report`; legacy fallback may read `cached_tokens` | always 0 |

For SGLang HiCache, `cache_runtime.backend` records the configured deployment label and
optional `cache_runtime.hit_tier_tokens`, `restored_tokens`, and `storage_backend` come
only from `sglext.cached_tokens_details`. Telos does not infer a tier from aggregate
`cached_tokens`.

---

## 8. ref-pool

Defined in [refpool.py](../refpool.py). The "pointer table" for all large chunks of content.

- **A slug is frozen once registered.** `register` validates the slug regex `^[A-Za-z0-9_\-./]+$`,
  the block must be `band=FOLD`, and `ref_slug` must equal the slug. Duplicate registration raises an error.
- **`register_or_skip`** — idempotent registration. Used when sharing a RefPool across turns: the harness
  produces a ref_pool with the full payload every round, and this method prevents the second round from overwriting an entry that the first round
  already `fold`ed into a placeholder back to the full content.
- **`fold(slug, summary=)`** — swaps an entry for a short placeholder. **The slug is not touched**, and the bytes of
  every `[ref:slug]` reference point in the text stay unchanged → subsequent BPs can still hit. This is how "references fold
  naturally" is realized.
- **`render_blocks`** — renders in lexicographic slug order, guaranteeing byte stability across emits.
- **`lint_text` / `lint_blocks`** — before emit, scan all `[ref:slug]`; on finding
  an unregistered slug, immediately `fail-fast`.

The bridge's `_sync_refpool_into_system` renders the ref-pool into the system segment:
`pin* + ref-pool fold* + drop*`.

---

## 9. Integration Paths A / B

The two paths are **functionally equivalent** (same TELOS pipeline, same state accumulation, same `cache_control`
injection); the only differences are the process boundary / error handling / streaming.

### 9.1 Path A — SDK Transport (in-process)

Swap the agent's LLM client for the TELOS transport; the duck interface is identical.

- **`TelosAnthropicTransport`** ([scripts/telos_anthropic_transport.py](../scripts/telos_anthropic_transport.py)) —
  wraps `anthropic.Anthropic`, exposes `.messages.create(...)`. The `_do_create`
  flow: snapshot the inputs → pick the harness (explicit > sticky > auto-detect) → `harness.parse`
  → `Bridge(...).emit_with_plan()` → pass through non-TELOS fields → send the real request →
  `bridge.absorb_usage` accumulation → write `usage_log` / `prompt_trace_log`.
- **`TelosOpenAITransport`** ([scripts/telos_transport.py](../scripts/telos_transport.py)) —
  wraps `openai.OpenAI`, exposes `.chat.completions.create(...)`. Goes through the `telos`
  harness, using a custom `_ir_to_chat_completions` to produce the wire (preserving OpenAI's
  `tool_calls` / `role=tool` structure rather than inlining it into text).

A transport instance = one session, internally holding a `BridgeSessionState`.

### 9.2 Path B — HTTP Reverse Proxy (out-of-process)

The agent sets `ANTHROPIC_BASE_URL=http://127.0.0.1:7171`, with zero code changes.

- **`proxy/server.py`** — aiohttp reverse proxy. `POST /v1/messages` is forwarded after passing through
  the TELOS pipeline; SSE streaming is supported (side-channel parsing of `message_start` /
  `message_delta` to extract usage); other paths are transparently passed through. It embeds
  `/__telos/dashboard`, `/__telos/developer`, `/__telos/developer.json`.
  Default non-strict: on TELOS failure it degrades to passthrough (`--strict` changes it to return 500).
- **`proxy/pipeline.py`** — `process_anthropic_request(raw, ...)` is a pure function,
  splitting out parse → bridge → emit, shared by the proxy and the transport, eliminating wire drift.
- **session-id derivation** priority: `x-telos-session` header → `metadata.user_id`
  → `blake2b(api_key + system + tools + messages[0])` → `telos-<16hex>`.
- **`_SessionRegistry`** — an OrderedDict LRU (default cap 10000), holding a
  `BridgeSessionState` per session_id.

### 9.3 Installers (Path B)

[init/](../init/). The `AgentInstaller` ABC requires **idempotency** + `uninstall` to restore exactly.

- **`ClaudeCodeInstaller`** — writes `ANTHROPIC_BASE_URL` into the `env` field of
  `~/.claude/settings.json`. The first patch backs up to `.telos.bak`; keeps the user's original value in
  `__telos_previous_base_url`; the marker key is `__telos_installed`. Atomic write
  (`.tmp` + `os.replace`). Does not touch the npm package, does not touch PATH, survives `npm update`.
- **`GenericInstaller`** — only prints the `export ANTHROPIC_BASE_URL=...` instruction.

---

## 10. RTK Output Filtering Layer

[output_filter/](../output_filter/). **Orthogonal** to the TELOS pipeline: TELOS stabilizes the request
prefix to get the KV cache, while this layer shrinks the tool result tail to reduce the new tokens added each round.

### 10.1 TelosMode — The Four-State Switch

```python
@dataclass(frozen=True)
class TelosMode:
    telos: bool = True    # Run the TELOS pipeline (cache_control / ref-pool)
    rtk:   bool = False   # Run RTK tool result filtering
```

| label | telos | rtk | Meaning |
|---|:---:|:---:|---|
| `none` | ✗ | ✗ | Pure passthrough, the proxy does not change a single byte |
| `telos` | ✓ | ✗ | TELOS prefix caching only (proxy default) |
| `rtk` | ✗ | ✓ | RTK tool filtering only, no cache markers applied |
| `both` | ✓ | ✓ | Both enabled |

An unknown / empty value degrades to the default `telos` (preserving the historical behavior from before the switch was introduced).

### 10.2 Filters

- **`RtkFilter`** — shells out to the `rtk` binary (`rtk filter --command <cmd>`
  reading stdin). A conventional call form; any failure degrades to passthrough.
- **`FallbackFilter`** — a dependency-free pure-Python filter: consecutive repeated lines folded into
  `<line> (×N)`, head/tail truncation, pytest summary preserved. Guarantees the switch still takes effect when rtk is not installed.
- **`CompositeFilter`** — rtk first; if it saves no bytes, fall back to fallback.
- `build_filter()` — rtk available → `Composite(rtk, fallback)`, otherwise a pure
  `FallbackFilter`.
- Thresholds: output shorter than 600 chars is not filtered; after dedup, output over 4000 chars goes through head/tail truncation.

### 10.3 apply_filter

`preprocess.apply_filter(raw, flt) → (new_raw, FilterStats)` — a pure function that
deep-copies raw, rewrites the text content of all `tool_result` (supporting both the str and block-list
content forms), and looks up the command hint from the `tool_use` of the previous assistant message.
`FilterStats` records `original_chars` / `filtered_chars` / `blocks_filtered`
/ `by_rule`.

### 10.4 proxy Wiring

- The `--mode` CLI switch + the `X-Telos-Mode` header (sticky to the session on the first request).
- The `X-Telos-Compare-Group` header → comparison-experiment grouping.
- `mode.rtk` on → `apply_filter` before entering TELOS; `mode.telos` off → skip the
  pipeline and go to passthrough.
- usage_log gains the `mode` / `compare_group` / `tool_output_reduction` fields.

---

## 11. Recording and Replay Comparison

### 11.1 corpus — The Session Corpus

[corpus.py](../corpus.py). By default the proxy records the **raw request** of every call to
`~/.telos/corpus/<session>.jsonl` (records only requests, not responses — Anthropic is stateless,
the Nth-turn request already contains all the content of the previous N-1 turns). `--no-record` turns it off, `--corpus-dir`
changes the directory. Functions: `record_call` / `load_session` / `list_sessions`.

### 11.2 replay — Controlled Replay Comparison

[replay/](../replay/). `replay_session(turns, mode, ...)` replays a real session under
a given mode: a byte-identical turn sequence → RTK filtering (if `mode.rtk`) → the TELOS
pipeline (if `mode.telos`) → sent upstream with `max_tokens=1` → only the usage is taken.

- **Why `max_tokens=1`**: only prefill / cache billing is measured; output generation is deliberately neutered.
- **Cache isolation**: by default a unique prefix
  `[telos-replay ns=<session>/<mode>]` is injected at the very front of the system segment for each mode, so Anthropic-side caches are independent of each other,
  preventing an earlier-replayed mode from warming the cache for a later one to free-ride on.
- The result is appended to usage_log, with `compare_group` = the original session id and `replay: true`.

CLI: `telos replay --list` / `telos replay --session <id> --modes ...`.
See [replay-comparison.md](replay-comparison.md) for the principle and boundaries.

### 11.3 replay vs Dual Session

| | Cost | Controlled variables | Suitable claim |
|---|---|---|---|
| replay | 1 real session + cheap prefill | Good (turns pinned) | "For a given workload, the token bill drops by X" |
| Dual session | N×K full sessions | Poor (trajectory forks) | "Using TELOS, the agent is cheaper overall" |

---

## 12. Observability

### 12.1 usage_log

Shared by the proxy and the SDK transport. One jsonl line per call. Key fields: `session_id`
/ `call_index` / `harness` / `mode` / `compare_group` / `tool_output_reduction`
/ `normalized` (4 fields) / `raw_usage` / `cumulative` (`cache_creation` /
`real_requests_since_refresh` / `refpool_slugs`).

### 12.2 Savings Dashboard ([build_savings_dashboard.py](../scripts/build_savings_dashboard.py))

`telos dashboard` or the proxy-embedded `/__telos/dashboard`. Aggregates usage_log into
"how many tokens / how many dollars saved". Includes the 2026 price table (with the cache_write 5m/1h split).
The live dashboard intentionally stays focused on savings totals, timeline, and breakdowns by harness / model / session; replay and showcase comparison records keep their `compare_group` metadata outside this page.

### 12.3 Developer Page ([build_developer_page.py](../scripts/build_developer_page.py))

The proxy-embedded `/__telos/developer`. Renders the IR
structure of all sessions **currently in memory**, the PIN/FOLD/DROP character distribution of prompt regions, a recent-calls table, a per-message band
view, and tool-call statistics. The data source is the `SessionInspector` of `proxy/inspector.py`
(an OrderedDict LRU, keeping the most recent `INSPECTOR_HISTORY=25` calls per session).

### 12.4 prompt_trace + show_prompt_trace

The SDK transport additionally writes `prompt_trace_log` (IR layout snapshot, plan details, prefix overlap
across calls). `scripts/show_prompt_trace.py` pretty-prints it in the terminal.

---

## 13. Invariants and Design Constraints

| ID | Constraint | Realization |
|---|---|---|
| §5 | Within each segment, `pin* → fold* → drop*` | `assert_band_order`, validated once before and once after emit |
| I3 | A ref-pool slug is frozen once registered | `RefPool.register` raises on duplicate registration |
| §4 | A `[ref:slug]` reference must be findable in the ref-pool | `lint_blocks` fail-fast before emit |
| R2 | Long conversations need a mid-rolling anchor | Anthropic `BP-mid` (messages ≥ 19) |
| R5 | Cross-language JSON key disorder breaks the cache | `_canonicalize_*` done uniformly in the bridge |
| R6 | A thinking block cannot have cache_control attached directly | harness bands it FOLD, the engine does not attach it on emit |
| R7 | When BPs exceed 4, truncate by priority | Anthropic `BP-T<BP-S<BP-R<BP-mid<BP-X` |
| R8 | Renewing a low-activity session loses money | `refresh` adaptive gating, threshold `REFRESH_THRESHOLD=11` |

---

## 14. End-to-End Data Flow

Taking path B (proxy) + `mode=both` as an example:

```
1. The agent sends POST /v1/messages to the proxy
2. The proxy derives the session_id and fetches the BridgeSessionState
3. [Recording] record_call writes the raw request into the corpus
4. Parse the mode (header > sticky > process default) + compare_group
5. [RTK] mode.rtk → apply_filter shortens tool_result
6. [TELOS] mode.telos → process_anthropic_request:
     a. _detect_harness → pick the harness (sticky)
     b. harness.parse → TelosIR
     c. Bridge(ir, engine, session_state).emit_with_plan():
        - _canonicalize_ir (key sorting, tool sorting)
        - assert_ir_invariants (§5 check)
        - refpool.lint_blocks (reference check)
        - engine.plan_marks → EmitPlan (BP anchor positions)
        - engine.emit → wire (cache_control attached)
     d. Pass through non-TELOS fields
7. The proxy forwards the wire to the real Anthropic
8. On receiving the response: side-channel parse of usage
9. The bridge accumulates cache_creation; writes usage_log + inspector
10. The response is returned to the agent as-is
```

---

## 15. Extension Points

| What you want to do | What to change |
|---|---|
| Add a new agent installer | In [init/](../init/) add `<name>.py` implementing `AgentInstaller`, register it in `init.INSTALLERS` |
| Add a new harness | In [harness/](../harness/) add a plugin, register it in [registry.py](../registry.py) |
| Add a new engine adapter | In [engine/](../engine/) add an `EngineAdapter` / `BidirectionalEngineAdapter` subclass, register it in the registry |
| Add a new tool filtering rule | The `FallbackFilter` of [output_filter/filters.py](../output_filter/filters.py), or let `RtkFilter` go through the rtk binary |
| Add a `/v1/chat/completions` proxy path | In [proxy/server.py](../proxy/server.py) add a route, reusing the same OpenAI pipeline. Codex's `/v1/responses` path is currently routed through the same upstream mechanism as passthrough. |
| Persist session state | `BridgeSessionState` is an ordinary dataclass, serializable to JSON; change `_SessionRegistry` to use external storage |
| Adjust the canonical sorting | The bridge's `_SCHEMA_SET_ARRAY_KEYS`, `_TOOL_SOURCE_RANK` are module-level names, monkey-patchable |
