"""Pure tests for the live SGLang HiCache restore contract evaluator."""

from __future__ import annotations

from telos.scripts.contract_sglang_hicache import (
    evaluate_restore,
    parse_tier_details,
    parse_usage,
)


def _response(*, device: int, host: int, storage: int) -> dict:
    return {
        "usage": {
            "prompt_tokens": 1000,
            "prompt_tokens_details": {"cached_tokens": 900},
            "completion_tokens": 1,
        },
        "sglext": {
            "cached_tokens_details": {
                "device": device,
                "host": host,
                "storage": storage,
                "storage_backend": "file" if storage else None,
            },
        },
    }


def test_parse_contract_response() -> None:
    response = _response(device=100, host=600, storage=200)
    assert parse_usage(response) == {
        "prompt_tokens": 1000,
        "cached_tokens": 900,
        "computed_prefill_tokens": 100,
        "hit_ratio": 0.9,
    }
    assert parse_tier_details(response) == {
        "device": 100,
        "host": 600,
        "storage": 200,
        "storage_backend": "file",
    }


def test_host_restore_passes() -> None:
    verdict = evaluate_restore(
        _response(device=100, host=800, storage=0),
        expected_tier="host",
        min_cached_ratio=0.8,
    )
    assert verdict["passed"] is True
    assert verdict["restored_tokens"] == 800


def test_contract_rejects_gpu_only_hit() -> None:
    verdict = evaluate_restore(
        _response(device=900, host=0, storage=0),
        expected_tier="any_non_device",
        min_cached_ratio=0.8,
    )
    assert verdict["passed"] is False
    assert any("host or storage" in item for item in verdict["failures"])


def test_contract_rejects_missing_tier_details_and_low_hit_ratio() -> None:
    verdict = evaluate_restore(
        {
            "usage": {
                "prompt_tokens": 1000,
                "prompt_tokens_details": {"cached_tokens": 100},
            },
        },
        expected_tier="host",
        min_cached_ratio=0.8,
    )
    assert verdict["passed"] is False
    assert len(verdict["failures"]) == 2


def test_malformed_tier_details_fail_open_to_unknown() -> None:
    assert parse_tier_details({
        "sglext": {
            "cached_tokens_details": {
                "device": "not-an-int",
                "host": 1,
            },
        },
    }) == {}
