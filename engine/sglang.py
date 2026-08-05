"""Stock-compatible SGLang adapter.

SGLang's public OpenAI-compatible endpoint relies on automatic RadixAttention
prefix matching. HiCache and LMCache are selected at server startup; they are
not request-level ``cache_control`` fields. This adapter therefore emits only
standard Chat Completions fields and declares active controls unsupported
until a separate capability-negotiated extension is implemented.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from telos.engine.base import (
    CacheCapabilities,
    EmitPlan,
    EngineAdapter,
    EngineCapabilities,
    UnsupportedWireProtocolError,
    WireProtocol,
)
from telos.engine.openai_chat import render_chat_completions
from telos.ir import Band, TelosIR, UsageReport


class SGLangAdapter(EngineAdapter):
    """SGLang OpenAI-compatible adapter in safe stock mode."""

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            explicit_breakpoints=False,
            ttl_control="none",
            prewarmable=False,
            routing_key=False,
            retention_policy="fixed",
            max_breakpoints=0,
            cache=CacheCapabilities(
                prefix_reuse="native",
                cache_report="config_only",
                tier_report="native",
                hierarchical_storage="config_only",
                request_namespace="unsupported",
                cache_manifest="unsupported",
                boundary_resolution="unsupported",
                retention_hint="unsupported",
                prefetch_hint="unsupported",
                explicit_evict="unsupported",
            ),
        )

    def plan_marks(self, ir: TelosIR) -> EmitPlan:
        """Return diagnostics only; stock SGLang has no explicit mark slots."""
        return EmitPlan(
            extras={"prefix_digest": self._prefix_digest(ir)},
        )

    def emit(self, ir: TelosIR, plan: EmitPlan) -> Mapping[str, Any]:
        """Emit a stock OpenAI Chat Completions body.

        ``plan`` remains useful for diagnostics, but none of its logical marks
        are serialized because stock SGLang has no public per-span control
        contract. RadixAttention finds the longest matching token prefix
        automatically.
        """
        return render_chat_completions(
            ir,
            model=ir.hints.model or "sglang-served",
        )

    def emit_for_protocol(
        self,
        ir: TelosIR,
        plan: EmitPlan,
        *,
        protocol: WireProtocol,
    ) -> Mapping[str, Any]:
        if protocol != "openai-chat":
            raise UnsupportedWireProtocolError(
                f"SGLangAdapter does not emit protocol {protocol!r}"
            )
        return self.emit(ir, plan)

    def parse_usage(self, response: Mapping[str, Any]) -> UsageReport:
        usage_obj = response.get("usage", {})
        usage = usage_obj if isinstance(usage_obj, Mapping) else {}
        details_obj = usage.get("prompt_tokens_details", {})
        details = details_obj if isinstance(details_obj, Mapping) else {}
        prompt = int(usage.get("prompt_tokens", 0) or 0)
        # Official SGLang --enable-cache-report shape first; retain the legacy
        # top-level form for old pinned images and existing telemetry fixtures.
        cached = int(details.get("cached_tokens", usage.get("cached_tokens", 0)) or 0)
        return UsageReport(
            raw_input=max(0, prompt - cached),
            cache_read=cached,
            cache_write=0,
            output=int(usage.get("completion_tokens", 0) or 0),
            raw=usage,
        )

    def cache_telemetry_request_fields(
        self,
        *,
        tier_details: bool = False,
    ) -> Mapping[str, Any]:
        """Opt in to SGLang's public per-tier cache telemetry extension."""
        if not tier_details:
            return {}
        return {"return_cached_tokens_details": True}

    def parse_cache_telemetry(
        self,
        response: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Parse current SGLang ``sglext.cached_tokens_details`` safely.

        SGLang reports device, host, and optional external-storage token
        counts. The Telos schema names those physical tiers gpu/cpu/l3 while
        retaining the reported storage backend. Older SGLang versions simply
        produce no telemetry, which is represented as an empty mapping.
        """
        extension = response.get("sglext")
        if not isinstance(extension, Mapping):
            return {}
        details = extension.get("cached_tokens_details")
        if not isinstance(details, Mapping):
            return {}

        def _token_count(name: str) -> int | None:
            value = details.get(name)
            if value is None:
                return 0
            if isinstance(value, bool):
                return None
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                return None
            return parsed if parsed >= 0 else None

        device = _token_count("device")
        host = _token_count("host")
        storage = _token_count("storage")
        if device is None or host is None or storage is None:
            return {}

        backend = details.get("storage_backend")
        if backend is not None and not isinstance(backend, str):
            backend = str(backend)
        return {
            "source": "sglext.cached_tokens_details",
            "hit_tier_tokens": {
                "gpu": device,
                "cpu": host,
                "l3": storage,
            },
            "storage_backend": backend or None,
        }

    def _prefix_digest(self, ir: TelosIR) -> str:
        """Diagnostic identity of the stable tool + system PIN prefix."""
        digest = hashlib.sha256()
        for block in ir.tools:
            digest.update(json.dumps(block.payload, sort_keys=True).encode())
        for block in ir.system:
            if block.band is Band.PIN:
                digest.update(str(block.payload).encode())
        for slug in sorted(ir.ref_pool):
            digest.update(slug.encode())
        return f"sha256:{digest.hexdigest()}"
