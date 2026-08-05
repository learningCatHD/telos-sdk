"""CacheIntent validation tests."""

from __future__ import annotations

from dataclasses import replace

import pytest

from telos.ir import (
    Band,
    CacheBoundary,
    CacheIntent,
    LogicalPosition,
    TelosBlock,
    TelosHints,
    TelosIR,
    TelosInvariantError,
    TelosMessage,
    assert_ir_invariants,
)


def _ir() -> TelosIR:
    return TelosIR(
        session_id="intent-test",
        tools=(TelosBlock("tool", Band.PIN, "tool_def", {"name": "lookup"}),),
        system=(TelosBlock("system", Band.PIN, "text", "stable"),),
        messages=(TelosMessage(
            role="user",
            blocks=(TelosBlock("question", Band.PIN, "text", "hello"),),
        ),),
        ref_pool={},
        hints=TelosHints(engine="sglang", model="test-model"),
    )


def _intent(**changes) -> CacheIntent:
    base = CacheIntent(
        schema_version=1,
        namespace="opaque:tenant/project",
        reuse_scope="project",
        sensitivity="project",
        boundaries=(CacheBoundary(
            name="pin_end",
            end=LogicalPosition("system", 0),
            band=Band.PIN,
            retention="hot",
            expected_reuses=3,
        ),),
        next_use_distance=1,
    )
    return replace(base, **changes)


def test_valid_cache_intent() -> None:
    assert_ir_invariants(replace(_ir(), cache_intent=_intent()))


@pytest.mark.parametrize("namespace", ["", "has whitespace", "bad\nvalue"])
def test_cache_intent_rejects_invalid_namespace(namespace: str) -> None:
    with pytest.raises(TelosInvariantError):
        assert_ir_invariants(replace(
            _ir(), cache_intent=_intent(namespace=namespace),
        ))


def test_cache_intent_rejects_out_of_range_boundary() -> None:
    boundary = CacheBoundary(
        name="missing",
        end=LogicalPosition("message", 2, message_index=0),
        band=Band.PIN,
        retention="hot",
    )
    with pytest.raises(TelosInvariantError):
        assert_ir_invariants(replace(
            _ir(), cache_intent=_intent(boundaries=(boundary,)),
        ))


def test_cache_intent_rejects_negative_distance_and_reuses() -> None:
    with pytest.raises(TelosInvariantError):
        assert_ir_invariants(replace(
            _ir(), cache_intent=_intent(next_use_distance=-1),
        ))
    boundary = replace(_intent().boundaries[0], expected_reuses=-1)
    with pytest.raises(TelosInvariantError):
        assert_ir_invariants(replace(
            _ir(), cache_intent=_intent(boundaries=(boundary,)),
        ))
