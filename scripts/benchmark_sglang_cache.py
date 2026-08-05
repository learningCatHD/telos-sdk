"""Measure stock SGLang prefix reuse directly and through the Telos gateway.

The workload places a changing ``Current time:`` envelope before a large,
stable user payload. Direct Chat Completions loses the user-payload prefix at
that change; Telos moves the envelope to the DROP tail, so the stable payload
can remain part of the reusable prefix.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import aiohttp


@dataclass(frozen=True)
class Sample:
    endpoint: str
    round: int
    prompt_tokens: int
    cached_tokens: int
    computed_prefill_tokens: int
    cache_hit_ratio: float
    first_event_ms: float
    first_token_ms: float | None
    e2e_ms: float


def _usage_metrics(usage: Mapping[str, Any]) -> tuple[int, int, int, float]:
    prompt = int(usage.get("prompt_tokens", 0) or 0)
    details_obj = usage.get("prompt_tokens_details") or {}
    details = details_obj if isinstance(details_obj, Mapping) else {}
    cached = int(details.get("cached_tokens", 0) or usage.get("cached_tokens", 0))
    computed = max(prompt - cached, 0)
    ratio = cached / prompt if prompt else 0.0
    return prompt, cached, computed, ratio


def _event_has_token(payload: Mapping[str, Any]) -> bool:
    for choice in payload.get("choices") or []:
        if not isinstance(choice, Mapping):
            continue
        delta = choice.get("delta") or {}
        if not isinstance(delta, Mapping):
            continue
        if delta.get("content") or delta.get("reasoning_content"):
            return True
        if delta.get("tool_calls"):
            return True
    return False


async def _flush(session: aiohttp.ClientSession, url: str | None) -> None:
    if not url:
        return
    async with session.post(url) as response:
        if response.status >= 400:
            body = await response.text()
            raise RuntimeError(f"cache flush failed: HTTP {response.status}: {body[:500]}")


async def _one_request(
    session: aiohttp.ClientSession,
    *,
    endpoint_name: str,
    url: str,
    model: str,
    stable_payload: str,
    round_index: int,
    headers: Mapping[str, str],
) -> Sample:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a precise cache benchmark assistant."},
            {
                "role": "user",
                "content": (
                    f"Current time: benchmark-round-{round_index}\n"
                    f"{stable_payload}\n"
                    "Reply with exactly OK."
                ),
            },
        ],
        "temperature": 0,
        "max_tokens": 4,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    started = time.perf_counter()
    first_event_at: float | None = None
    first_token_at: float | None = None
    usage: dict[str, Any] = {}
    buffer = b""
    async with session.post(url, json=body, headers=headers) as response:
        if response.status >= 400:
            text = await response.text()
            raise RuntimeError(
                f"{endpoint_name} round {round_index}: HTTP {response.status}: {text[:1000]}"
            )
        async for chunk in response.content.iter_any():
            now = time.perf_counter()
            if first_event_at is None:
                first_event_at = now
            buffer += chunk
            while b"\n\n" in buffer:
                block, buffer = buffer.split(b"\n\n", 1)
                for line in block.splitlines():
                    if not line.startswith(b"data:"):
                        continue
                    raw = line[5:].strip()
                    if not raw or raw == b"[DONE]":
                        continue
                    try:
                        event = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if first_token_at is None and _event_has_token(event):
                        first_token_at = now
                    event_usage = event.get("usage")
                    if isinstance(event_usage, Mapping):
                        usage.update(event_usage)
    finished = time.perf_counter()
    prompt, cached, computed, ratio = _usage_metrics(usage)
    first_event = first_event_at or finished
    return Sample(
        endpoint=endpoint_name,
        round=round_index,
        prompt_tokens=prompt,
        cached_tokens=cached,
        computed_prefill_tokens=computed,
        cache_hit_ratio=round(ratio, 6),
        first_event_ms=round((first_event - started) * 1000, 3),
        first_token_ms=(
            round((first_token_at - started) * 1000, 3)
            if first_token_at is not None else None
        ),
        e2e_ms=round((finished - started) * 1000, 3),
    )


def _summary(samples: list[Sample]) -> dict[str, Any]:
    # The first request fills the cache; subsequent requests measure reuse.
    measured = samples[1:] if len(samples) > 1 else samples
    if not measured:
        return {}
    token_latencies = [s.first_token_ms for s in measured if s.first_token_ms is not None]
    return {
        "rounds": len(samples),
        "measured_rounds": len(measured),
        "median_cached_tokens": statistics.median(s.cached_tokens for s in measured),
        "median_computed_prefill_tokens": statistics.median(
            s.computed_prefill_tokens for s in measured
        ),
        "median_cache_hit_ratio": round(
            statistics.median(s.cache_hit_ratio for s in measured), 6,
        ),
        "median_first_token_ms": (
            round(statistics.median(token_latencies), 3) if token_latencies else None
        ),
        "median_e2e_ms": round(statistics.median(s.e2e_ms for s in measured), 3),
    }


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    if args.prefix_file:
        stable_payload = Path(args.prefix_file).read_text(encoding="utf-8")
    else:
        seed = "Stable project context: deterministic tools, rules, and reference data.\n"
        stable_payload = (seed * (args.stable_chars // len(seed) + 1))[:args.stable_chars]

    headers = {
        "content-type": "application/json",
        "x-telos-session": "sglang-cache-benchmark",
    }
    if args.api_key:
        headers["authorization"] = f"Bearer {args.api_key}"
    endpoints = [
        ("direct", args.direct_url),
        ("telos", args.telos_url),
    ]
    endpoints = [(name, url) for name, url in endpoints if url]
    if not endpoints:
        raise SystemExit("provide --direct-url and/or --telos-url")

    all_samples: dict[str, list[Sample]] = {}
    timeout = aiohttp.ClientTimeout(total=args.timeout_seconds)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for endpoint_name, url in endpoints:
            await _flush(session, args.flush_url)
            samples: list[Sample] = []
            for round_index in range(args.rounds):
                samples.append(await _one_request(
                    session,
                    endpoint_name=endpoint_name,
                    url=url,
                    model=args.model,
                    stable_payload=stable_payload,
                    round_index=round_index,
                    headers=headers,
                ))
            all_samples[endpoint_name] = samples

    return {
        "schema": "telos.sglang-cache-benchmark/v1",
        "model": args.model,
        "stable_chars": len(stable_payload),
        "samples": {
            name: [asdict(sample) for sample in samples]
            for name, samples in all_samples.items()
        },
        "summary": {
            name: _summary(samples) for name, samples in all_samples.items()
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--direct-url", help="SGLang /v1/chat/completions URL")
    parser.add_argument("--telos-url", help="Telos local-sglang chat/completions URL")
    parser.add_argument("--flush-url", help="Optional SGLang /flush_cache URL")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--stable-chars", type=int, default=64_000)
    parser.add_argument("--prefix-file")
    parser.add_argument("--timeout-seconds", type=float, default=300)
    parser.add_argument("--output", help="Optional JSON output path")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.rounds < 2:
        raise SystemExit("--rounds must be at least 2 (one fill + one reuse)")
    result = asyncio.run(_run(args))
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
