"""Stock SGLang adapter contract tests."""

from __future__ import annotations

from telos import load_engine, load_harness
from telos.bridge import _canonicalize_ir
from telos.proxy.pipeline import process_openai_request


def _request() -> dict:
    return {
        "model": "test-model",
        "messages": [
            {"role": "system", "content": "stable system"},
            {"role": "user", "content": "first"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "result"},
            {"role": "user", "content": "continue"},
        ],
        "tools": [{
            "type": "function",
            "function": {
                "name": "lookup",
                "parameters": {"type": "object", "properties": {}},
            },
        }],
    }


def _ir():
    return _canonicalize_ir(load_harness("telos").parse(
        _request(), session_id="sglang", engine="sglang", model="test-model",
    ))


def test_stock_wire_has_no_private_cache_fields_and_preserves_tools() -> None:
    adapter = load_engine("sglang")
    ir = _ir()
    plan = adapter.plan_marks(ir)
    wire = adapter.emit_for_protocol(ir, plan, protocol="openai-chat")
    assert "cache_control" not in wire
    assert "cache_policy" not in wire
    assert "telos_cache" not in wire
    assert "cache_salt" not in wire
    assert wire["model"] == "test-model"
    assert wire["tools"] == [ir.tools[0].payload]
    assistant = next(m for m in wire["messages"] if m["role"] == "assistant")
    tool_result = next(m for m in wire["messages"] if m["role"] == "tool")
    assert assistant["tool_calls"][0]["id"] == "call_1"
    assert tool_result["tool_call_id"] == "call_1"


def test_sglang_usage_prefers_nested_cached_tokens() -> None:
    report = load_engine("sglang").parse_usage({
        "usage": {
            "prompt_tokens": 1200,
            "prompt_tokens_details": {"cached_tokens": 900},
            "cached_tokens": 10,
            "completion_tokens": 20,
        },
    })
    assert report.raw_input == 300
    assert report.cache_read == 900
    assert report.cache_write == 0
    assert report.output == 20


def test_sglang_usage_accepts_legacy_top_level_cached_tokens() -> None:
    report = load_engine("sglang").parse_usage({
        "usage": {
            "prompt_tokens": 1200,
            "cached_tokens": 800,
            "completion_tokens": 20,
        },
    })
    assert report.raw_input == 400
    assert report.cache_read == 800


def test_sglang_tier_telemetry_is_explicitly_opted_in() -> None:
    adapter = load_engine("sglang")
    assert adapter.cache_telemetry_request_fields() == {}
    assert adapter.cache_telemetry_request_fields(
        tier_details=True,
    ) == {"return_cached_tokens_details": True}

    enabled = process_openai_request(
        _request(),
        session_id="tier-on",
        engine_name="sglang",
        cache_backend="hicache",
        cache_tier_telemetry=True,
    )
    assert enabled.wire["return_cached_tokens_details"] is True
    assert enabled.cache_backend == "hicache"

    disabled = process_openai_request(
        _request(),
        session_id="tier-off",
        engine_name="sglang",
        cache_mode="off",
        cache_tier_telemetry=True,
    )
    assert "return_cached_tokens_details" not in disabled.wire


def test_sglang_parses_hicache_tier_details() -> None:
    telemetry = load_engine("sglang").parse_cache_telemetry({
        "sglext": {
            "cached_tokens_details": {
                "device": 128,
                "host": 768,
                "storage": 256,
                "storage_backend": "file",
            },
        },
    })
    assert telemetry == {
        "source": "sglext.cached_tokens_details",
        "hit_tier_tokens": {"gpu": 128, "cpu": 768, "l3": 256},
        "storage_backend": "file",
    }


def test_sglang_tier_details_fail_open_when_missing_or_malformed() -> None:
    adapter = load_engine("sglang")
    assert adapter.parse_cache_telemetry({}) == {}
    assert adapter.parse_cache_telemetry({"sglext": {}}) == {}
    assert adapter.parse_cache_telemetry({
        "sglext": {"cached_tokens_details": {"device": "bad"}},
    }) == {}
