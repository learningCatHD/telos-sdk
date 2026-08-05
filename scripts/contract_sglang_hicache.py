"""Verify that a prefix evicted from SGLang L1 is restored from HiCache.

This is a live contract test, not a synthetic performance claim. It warms an
exact target prompt, creates enough unrelated KV pressure to evict the target
from GPU, then sends the exact target again with SGLang's public tier-detail
response enabled. Success requires both a high aggregate prefix-cache hit and
an explicit host/storage hit reported by ``sglext.cached_tokens_details``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any, Mapping

import aiohttp


def parse_usage(response: Mapping[str, Any]) -> dict[str, Any]:
    """Extract aggregate cache usage from a Chat Completions response."""
    usage_obj = response.get("usage")
    usage = usage_obj if isinstance(usage_obj, Mapping) else {}
    details_obj = usage.get("prompt_tokens_details")
    details = details_obj if isinstance(details_obj, Mapping) else {}
    prompt = _nonnegative_int(usage.get("prompt_tokens"))
    cached = _nonnegative_int(
        details.get("cached_tokens", usage.get("cached_tokens")),
    )
    return {
        "prompt_tokens": prompt,
        "cached_tokens": cached,
        "computed_prefill_tokens": max(0, prompt - cached),
        "hit_ratio": round(cached / prompt, 6) if prompt else 0.0,
    }


def parse_tier_details(response: Mapping[str, Any]) -> dict[str, Any]:
    """Extract current SGLang tier details without guessing missing fields."""
    extension = response.get("sglext")
    if not isinstance(extension, Mapping):
        return {}
    details = extension.get("cached_tokens_details")
    if not isinstance(details, Mapping):
        return {}
    device = _strict_nonnegative_int(details.get("device", 0))
    host = _strict_nonnegative_int(details.get("host", 0))
    storage = _strict_nonnegative_int(details.get("storage", 0))
    if device is None or host is None or storage is None:
        return {}
    backend = details.get("storage_backend")
    return {
        "device": device,
        "host": host,
        "storage": storage,
        "storage_backend": str(backend) if backend is not None else None,
    }


def evaluate_restore(
    response: Mapping[str, Any],
    *,
    expected_tier: str,
    min_cached_ratio: float,
) -> dict[str, Any]:
    """Return a machine-readable contract verdict for the restore request."""
    usage = parse_usage(response)
    tiers = parse_tier_details(response)
    failures: list[str] = []
    if not tiers:
        failures.append(
            "SGLang returned no valid sglext.cached_tokens_details; verify the "
            "server release supports return_cached_tokens_details and that the "
            "request field is enabled"
        )
    if usage["prompt_tokens"] <= 0:
        failures.append("response usage.prompt_tokens is missing or zero")
    elif usage["hit_ratio"] < min_cached_ratio:
        failures.append(
            f"cached-token hit ratio {usage['hit_ratio']:.6f} is below "
            f"required {min_cached_ratio:.6f}"
        )

    if tiers:
        if expected_tier == "host" and tiers["host"] <= 0:
            failures.append("HiCache restore did not report any host-tier tokens")
        elif expected_tier == "storage" and tiers["storage"] <= 0:
            failures.append("HiCache restore did not report any storage-tier tokens")
        elif expected_tier == "any_non_device" and (
            tiers["host"] + tiers["storage"] <= 0
        ):
            failures.append("restore did not report host or storage-tier tokens")

    return {
        "passed": not failures,
        "expected_tier": expected_tier,
        "min_cached_ratio": min_cached_ratio,
        "usage": usage,
        "tier_details": tiers,
        "restored_tokens": (
            tiers.get("host", 0) + tiers.get("storage", 0)
            if tiers else 0
        ),
        "failures": failures,
    }


def _nonnegative_int(value: Any) -> int:
    parsed = _strict_nonnegative_int(value)
    return parsed if parsed is not None else 0


def _strict_nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value or 0)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _target_messages(stable_payload: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "Telos HiCache restore contract target v1.",
        },
        {
            "role": "user",
            "content": stable_payload + "\nReply with one word: OK",
        },
    ]


def _pressure_messages(index: int, pressure_payload: str) -> list[dict[str, str]]:
    # The unique marker is the first semantic token, preventing the pressure
    # requests from sharing a long radix prefix with the target or each other.
    return [
        {
            "role": "system",
            "content": f"unique-pressure-{index:06d}",
        },
        {
            "role": "user",
            "content": pressure_payload + f"\npressure-round={index}",
        },
    ]


async def _post_completion(
    session: aiohttp.ClientSession,
    *,
    url: str,
    model: str,
    messages: list[dict[str, str]],
    headers: Mapping[str, str],
) -> tuple[dict[str, Any], float]:
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 1,
        "stream": False,
        "return_cached_tokens_details": True,
    }
    started = time.perf_counter()
    async with session.post(url, json=body, headers=headers) as response:
        payload_text = await response.text()
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        if response.status >= 400:
            raise RuntimeError(
                f"completion failed: HTTP {response.status}: {payload_text[:1000]}"
            )
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError("completion returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("completion response must be a JSON object")
        return payload, elapsed_ms


async def _flush_cache(
    session: aiohttp.ClientSession,
    *,
    url: str | None,
    admin_api_key: str,
) -> None:
    if not url:
        return
    headers = {}
    if admin_api_key:
        headers["authorization"] = f"Bearer {admin_api_key}"
    async with session.post(url, headers=headers) as response:
        body = await response.text()
        if response.status >= 400:
            raise RuntimeError(
                f"cache flush failed: HTTP {response.status}: {body[:1000]}"
            )


async def run_contract(args: argparse.Namespace) -> dict[str, Any]:
    seed = "Stable Telos HiCache contract context. Preserve every exact token.\n"
    stable_payload = (seed * (args.prefix_chars // len(seed) + 1))[
        :args.prefix_chars
    ]
    pressure_seed = "Unrelated KV pressure payload with deterministic length.\n"
    pressure_payload = (pressure_seed * (
        args.prefix_chars // len(pressure_seed) + 1
    ))[:args.prefix_chars]

    headers = {"content-type": "application/json"}
    if args.api_key:
        headers["authorization"] = f"Bearer {args.api_key}"
    timeout = aiohttp.ClientTimeout(total=args.timeout_seconds)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        await _flush_cache(
            session,
            url=args.flush_url,
            admin_api_key=args.admin_api_key,
        )
        warm_response, warm_ms = await _post_completion(
            session,
            url=args.url,
            model=args.model,
            messages=_target_messages(stable_payload),
            headers=headers,
        )
        await asyncio.sleep(args.settle_seconds)

        pressure_samples: list[dict[str, Any]] = []
        for index in range(args.pressure_rounds):
            pressure_response, elapsed_ms = await _post_completion(
                session,
                url=args.url,
                model=args.model,
                messages=_pressure_messages(index, pressure_payload),
                headers=headers,
            )
            pressure_samples.append({
                "round": index,
                "latency_ms": elapsed_ms,
                "usage": parse_usage(pressure_response),
                "tier_details": parse_tier_details(pressure_response),
            })
        await asyncio.sleep(args.settle_seconds)

        restore_response, restore_ms = await _post_completion(
            session,
            url=args.url,
            model=args.model,
            messages=_target_messages(stable_payload),
            headers=headers,
        )

    verdict = evaluate_restore(
        restore_response,
        expected_tier=args.expected_tier,
        min_cached_ratio=args.min_cached_ratio,
    )
    return {
        "schema": "telos.sglang-hicache-contract/v1",
        "model": args.model,
        "endpoint": args.url,
        "parameters": {
            "prefix_chars": args.prefix_chars,
            "pressure_rounds": args.pressure_rounds,
            "settle_seconds": args.settle_seconds,
            "expected_tier": args.expected_tier,
            "min_cached_ratio": args.min_cached_ratio,
            "cache_flushed": bool(args.flush_url),
        },
        "warm": {
            "latency_ms": warm_ms,
            "usage": parse_usage(warm_response),
            "tier_details": parse_tier_details(warm_response),
        },
        "pressure": pressure_samples,
        "restore": {
            "latency_ms": restore_ms,
            **verdict,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:30000/v1/chat/completions",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--flush-url",
        default="http://127.0.0.1:30000/flush_cache",
        help="empty string disables the initial cache flush",
    )
    parser.add_argument("--api-key", default="")
    parser.add_argument("--admin-api-key", default="")
    parser.add_argument("--prefix-chars", type=int, default=64_000)
    parser.add_argument("--pressure-rounds", type=int, default=16)
    parser.add_argument("--settle-seconds", type=float, default=1.0)
    parser.add_argument(
        "--expected-tier",
        choices=("host", "storage", "any_non_device"),
        default="host",
    )
    parser.add_argument("--min-cached-ratio", type=float, default=0.8)
    parser.add_argument("--timeout-seconds", type=float, default=300)
    parser.add_argument("--output")
    parser.add_argument(
        "--no-assert",
        action="store_true",
        help="always exit zero while retaining the verdict in JSON",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.prefix_chars < 1:
        raise SystemExit("--prefix-chars must be positive")
    if args.pressure_rounds < 1:
        raise SystemExit("--pressure-rounds must be positive")
    if not 0 <= args.min_cached_ratio <= 1:
        raise SystemExit("--min-cached-ratio must be between 0 and 1")
    result = asyncio.run(run_contract(args))
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if not args.no_assert and not result["restore"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
