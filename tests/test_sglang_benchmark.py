"""Pure helper tests for the real SGLang benchmark CLI."""

from __future__ import annotations

from telos.scripts.benchmark_sglang_cache import Sample, _summary, _usage_metrics


def test_usage_metrics_reads_sglang_nested_shape() -> None:
    assert _usage_metrics({
        "prompt_tokens": 1000,
        "prompt_tokens_details": {"cached_tokens": 800},
    }) == (1000, 800, 200, 0.8)


def test_summary_excludes_cache_fill_round() -> None:
    samples = [
        Sample("telos", 0, 1000, 0, 1000, 0.0, 10, 20, 30),
        Sample("telos", 1, 1000, 800, 200, 0.8, 5, 8, 10),
        Sample("telos", 2, 1000, 900, 100, 0.9, 4, 7, 9),
    ]
    summary = _summary(samples)
    assert summary["measured_rounds"] == 2
    assert summary["median_cached_tokens"] == 850
    assert summary["median_computed_prefill_tokens"] == 150
    assert summary["median_cache_hit_ratio"] == 0.85
