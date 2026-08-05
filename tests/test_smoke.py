"""Minimal self-check of the TELOS protocol: run the requests from the demo, verify invariants, Mark slots, usage parsing.

How to run:
    python -m telos.tests.test_smoke
"""

from __future__ import annotations

import sys
from pathlib import Path

# allow running directly via python -m telos.tests.test_smoke
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from telos import Band, Bridge, TelosInvariantError  # noqa: E402
from telos import load_engine, load_harness          # noqa: E402
from telos.demo import RAW_REQUEST                   # noqa: E402
from telos.ir import assert_ir_invariants            # noqa: E402


def test_harness_band_split() -> None:
    """An OpenClaw user message must be split into (PIN, DROP); the envelope must not contaminate the PIN."""
    harness = load_harness("openclaw")
    ir = harness.parse(RAW_REQUEST, session_id="t1", engine="anthropic",
                       model="claude-opus-4-7")
    user_msg = ir.messages[0]
    bands = [b.band for b in user_msg.blocks]
    assert Band.PIN in bands, f"user message missing PIN block: {bands}"
    assert Band.DROP in bands, f"user message missing DROP block: {bands}"
    assert bands.index(Band.PIN) < bands.index(Band.DROP), \
        f"PIN must precede DROP, got {bands}"
    # no PIN block's payload may contain an envelope pattern
    for blk in user_msg.blocks:
        if blk.band is Band.PIN:
            assert "<environment_info>" not in str(blk.payload)
            assert "Current time:" not in str(blk.payload)
    print("✓ test_harness_band_split")


def test_refpool_lifts_large_doc() -> None:
    """A large block of system text must be moved into the ref-pool, leaving a [ref:...] reference."""
    harness = load_harness("openclaw")
    ir = harness.parse(RAW_REQUEST, session_id="t2", engine="anthropic")
    assert ir.ref_pool, "expected ref-pool entries for large system doc"
    # the ref reference should appear in the system segment
    found_ref = any("[ref:" in str(b.payload) for b in ir.system)
    assert found_ref, "system should contain a [ref:...] pointer"
    print("✓ test_refpool_lifts_large_doc")


def test_anthropic_mark_slots() -> None:
    """The Anthropic adapter should produce ≤4 slots, and 1h must come before 5m."""
    harness = load_harness("openclaw")
    engine = load_engine("anthropic")
    ir = harness.parse(RAW_REQUEST, session_id="t3", engine="anthropic",
                       model="claude-opus-4-7")
    bridge = Bridge(ir, engine)
    plan = bridge.mark()
    assert len(plan.slots) <= 4
    # a long-TTL slot must appear before a short-TTL one (in segment order)
    seg_order = {"tools": 0, "system": 1, "message": 2}
    last_ttl = "long"
    for s in sorted(plan.slots, key=lambda s: (seg_order[s.segment], s.index)):
        if last_ttl == "long" and s.ttl_class == "short":
            last_ttl = "short"
        elif last_ttl == "short":
            assert s.ttl_class == "short", \
                f"1h slot {s.name} appears after a 5m slot — violates Anthropic ordering"
    print("✓ test_anthropic_mark_slots")


def test_emit_round_trip_three_engines() -> None:
    """All five engines can emit the same IR into their respective wire requests without raising."""
    harness = load_harness("openclaw")
    for name, model in (
        ("anthropic", "claude-opus-4-7"),
        ("openai",    "gpt-5.1"),
        ("deepseek",  "deepseek-chat"),
        ("vllm",      "Qwen/Qwen3-32B"),
        ("sglang",    "deepseek-ai/DeepSeek-V3"),
    ):
        ir = harness.parse(RAW_REQUEST, session_id=f"t4-{name}", engine=name, model=model)
        bridge = Bridge(ir, load_engine(name))
        wire = bridge.emit()
        assert wire["model"]
        assert "messages" in wire or "input" in wire
    print("✓ test_emit_round_trip_three_engines")


def test_bidirectional_only_on_verified_extensions() -> None:
    """Stock SGLang must not claim an unimplemented active control plane."""
    harness = load_harness("openclaw")
    bidi = {"vllm"}
    for name in ("anthropic", "openai", "deepseek", "vllm", "sglang"):
        ir = harness.parse(RAW_REQUEST, session_id=f"b-{name}", engine=name)
        bridge = Bridge(ir, load_engine(name))
        assert bridge.is_bidirectional == (name in bidi), \
            f"[{name}] is_bidirectional should be {name in bidi}"
        # probe can always be called; closed-source engines just return hit=False (no-op)
        probe = bridge.probe_cache()
        if name not in bidi:
            assert probe.hit is False
    print("✓ test_bidirectional_only_on_verified_extensions")


def test_sglang_stock_fold_emits_no_private_control() -> None:
    """Stock SGLang folds client-side and relies on natural prefix reuse."""
    harness = load_harness("openclaw")
    ir = harness.parse(RAW_REQUEST, session_id="cf-sgl", engine="sglang",
                       model="deepseek-ai/DeepSeek-V3")
    bridge = Bridge(ir, load_engine("sglang"))
    ctrl = bridge.cooperative_fold(message_range=(1, 3), summary="<folded>")
    assert ctrl == {}
    wire = bridge.emit_with_extras(ctrl)
    assert "cache_control" not in wire
    assert "telos_cache" not in wire
    print("✓ test_sglang_stock_fold_emits_no_private_control")


def test_sglang_capabilities_are_truthful() -> None:
    caps = load_engine("sglang").capabilities
    assert caps.cache.prefix_reuse == "native"
    assert caps.cache.cache_report == "config_only"
    assert caps.cache.hierarchical_storage == "config_only"
    assert caps.span_eviction is False
    assert caps.fork_and_replace is False
    assert caps.pin_unpin is False
    print("✓ test_sglang_capabilities_are_truthful")


def test_vllm_pin_cache_policy() -> None:
    """vLLM's pin_until must appear in the cache_policy field of the wire body."""
    harness = load_harness("openclaw")
    ir = harness.parse(RAW_REQUEST, session_id="cf-vllm", engine="vllm",
                       model="Qwen/Qwen3-32B")
    bridge = Bridge(ir, load_engine("vllm"))
    wire = bridge.emit()
    policy = wire.get("cache_policy", {})
    assert "pin_prefix_until_block" in policy, \
        f"vLLM wire missing pin_prefix_until_block: {policy}"
    assert "cache_salt" in wire
    print("✓ test_vllm_pin_cache_policy")


def test_band_order_violation_caught() -> None:
    """Manually construct an IR that violates §5 and confirm the bridge rejects it."""
    from telos.ir import TelosBlock, TelosIR, TelosMessage, TelosHints

    bad = TelosIR(
        session_id="bad",
        tools=(),
        system=(),
        messages=(TelosMessage(role="user", blocks=(
            TelosBlock(id="d", band=Band.DROP, kind="text", payload="x"),
            TelosBlock(id="p", band=Band.PIN,  kind="text", payload="y"),  # PIN after DROP → violation
        )),),
        ref_pool={},
        hints=TelosHints(engine="anthropic"),
    )
    try:
        assert_ir_invariants(bad)
    except TelosInvariantError:
        print("✓ test_band_order_violation_caught")
        return
    raise AssertionError("expected TelosInvariantError")


def test_usage_normalization() -> None:
    """The usage of all three engines can be normalized to (raw_input, cache_read, cache_write)."""
    harness = load_harness("openclaw")
    cases = [
        ("anthropic", {"usage": {"input_tokens": 80, "cache_read_input_tokens": 1000,
                                  "cache_creation_input_tokens": 50, "output_tokens": 10}},
         (80, 1000, 50)),
        ("openai",    {"usage": {"prompt_tokens": 1100,
                                  "prompt_tokens_details": {"cached_tokens": 1000},
                                  "completion_tokens": 10}},
         (100, 1000, 0)),
        ("deepseek",  {"usage": {"prompt_cache_hit_tokens": 1000,
                                  "prompt_cache_miss_tokens": 100,
                                  "completion_tokens": 10}},
         (100, 1000, 0)),
        ("vllm",      {"usage": {"prompt_tokens": 1100, "cached_tokens": 1000,
                                  "completion_tokens": 10}},
         (100, 1000, 0)),
        ("sglang",    {"usage": {"prompt_tokens": 1100,
                                  "prompt_tokens_details": {"cached_tokens": 1000},
                                  "completion_tokens": 10}},
         (100, 1000, 0)),
    ]
    for name, response, expected in cases:
        ir = harness.parse(RAW_REQUEST, session_id=f"u-{name}", engine=name)
        bridge = Bridge(ir, load_engine(name))
        report = bridge.absorb_usage(response)
        got = (report.raw_input, report.cache_read, report.cache_write)
        assert got == expected, f"[{name}] expected {expected}, got {got}"
    print("✓ test_usage_normalization")


def main() -> None:
    test_harness_band_split()
    test_refpool_lifts_large_doc()
    test_anthropic_mark_slots()
    test_emit_round_trip_three_engines()
    test_bidirectional_only_on_verified_extensions()
    test_sglang_stock_fold_emits_no_private_control()
    test_sglang_capabilities_are_truthful()
    test_vllm_pin_cache_policy()
    test_band_order_violation_caught()
    test_usage_normalization()
    print("\nall smoke tests passed.")


if __name__ == "__main__":
    main()
