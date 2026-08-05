"""``~/.telos/config.json`` —— telos global user config.

A lightweight JSON config file that records:

- ``mode``                 default optimization mode (none/telos/rtk/both)
- ``gateway``              gateway listen host / port / usage_log
- ``favorite_harness``     the harness the bare ``telos`` command enters by default
- ``harness_executables``  harness name → custom executable name (overrides the
  default guess)
- ``upstreams``            slug → {url, engine, path}; the gateway forwards
  ``/upstreams/<slug>/<...>`` requests to ``url``, using ``engine`` to drive the
  TELOS pipeline. Defaults cover Anthropic, OpenRouter, DeepSeek, OpenAI, and a
  stock-safe local SGLang target at ``http://127.0.0.1:30000``; user-added
  entries are preserved verbatim.

Design principles:
- Missing file → all defaults, no error (zero-config first use).
- Bad JSON → raise ``RuntimeError`` with a fix hint (do not silently swallow user data).
- Unknown keys are preserved round-trip (forward compatibility: an old telos
  does not drop fields written by a newer version).
- Atomic write (``.tmp`` + ``os.replace``), same as ``init/claude_code._atomic_write``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_SCHEMA_VERSION = 1

DEFAULT_GATEWAY_HOST = "127.0.0.1"
DEFAULT_GATEWAY_PORT = 7171
DEFAULT_MODE = "telos"


def telos_home() -> Path:
    """The ``~/.telos`` directory (can be overridden by the ``TELOS_HOME`` environment variable, handy for testing)."""
    env = os.environ.get("TELOS_HOME")
    return Path(env) if env else Path.home() / ".telos"


def config_path() -> Path:
    return telos_home() / "config.json"


def default_usage_log() -> Path:
    return telos_home() / "usage.jsonl"


@dataclass
class GatewayConfig:
    host: str = DEFAULT_GATEWAY_HOST
    port: int = DEFAULT_GATEWAY_PORT
    usage_log: str = ""  # empty string → use default_usage_log()

    def resolved_usage_log(self) -> Path:
        return Path(self.usage_log).expanduser() if self.usage_log else default_usage_log()

    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass(frozen=True)
class UpstreamCacheConfig:
    """Cache behavior for one upstream.

    Phase 1 implements ``off`` and stock automatic prefix reuse. Active
    capability negotiation remains a later extension, so unsupported mode
    strings safely normalize to ``stock`` while loading older/future configs.
    """

    mode: str = "stock"  # "off" | "stock"
    backend: str = "auto"  # "auto" | "radix" | "hicache" | "lmcache"
    tier_telemetry: bool = False
    security_namespace: str = ""
    allow_cross_session: bool = False
    capability_ttl_seconds: int = 60


@dataclass(frozen=True)
class UpstreamConfig:
    """One named forward target.

    Attributes:
        url:    upstream base URL (no trailing slash).
        engine: ``EngineAdapter`` name used by the TELOS pipeline (``anthropic``
                / ``openai`` / ``deepseek`` / ``vllm`` / ``sglang``). Drives both
                emit and ``parse_usage`` of responses.
        protocol: wire protocol the upstream expects on its request endpoint:
                ``anthropic-messages``  → POST ``/v1/messages``
                ``openai-chat``         → POST ``/v1/chat/completions``
                The gateway dispatches incoming ``/upstreams/<slug>/...`` requests
                onto the matching pipeline based on this.
        via:    optional harness identity attached to this slug at install time
                (e.g. ``"openclaw"``, ``"hermes"``). When set, the gateway
                labels usage-log entries for this upstream with this name so
                the dashboard's "breakdown by harness" attributes traffic to
                the calling tool rather than the wire-level harness
                (``"telos"`` for OpenAI-shape traffic). Empty / missing →
                gateway falls back to its content-detection default.
    """

    url: str
    engine: str
    protocol: str  # "anthropic-messages" | "openai-chat"
    via: str = ""
    cache: UpstreamCacheConfig = field(default_factory=UpstreamCacheConfig)


_DEFAULT_UPSTREAMS: dict[str, UpstreamConfig] = {
    "anthropic": UpstreamConfig(
        url="https://api.anthropic.com",
        engine="anthropic",
        protocol="anthropic-messages",
    ),
    "openrouter": UpstreamConfig(
        # No ``/v1`` suffix: the gateway forwards the inbound tail verbatim,
        # so the client supplies the version segment (``/v1/chat/completions``).
        # This matches the anthropic / deepseek defaults and avoids a double-/v1.
        url="https://openrouter.ai/api",
        engine="deepseek",            # DS-style usage fields pass through OpenRouter
        protocol="openai-chat",
    ),
    "deepseek": UpstreamConfig(
        url="https://api.deepseek.com",
        engine="deepseek",
        protocol="openai-chat",
    ),
    "openai": UpstreamConfig(
        # Used by Codex's custom provider profile. Today the gateway optimizes
        # OpenAI ChatCompletions traffic and transparently passes Responses API
        # traffic through this same upstream route.
        url="https://api.openai.com",
        engine="openai",
        protocol="openai-chat",
    ),
    "local-sglang": UpstreamConfig(
        url="http://127.0.0.1:30000",
        engine="sglang",
        protocol="openai-chat",
        cache=UpstreamCacheConfig(
            mode="stock",
            backend="radix",
            tier_telemetry=False,
            security_namespace="local-dev",
            allow_cross_session=False,
        ),
    ),
}


def default_upstreams() -> dict[str, UpstreamConfig]:
    """Fresh copy of the built-in upstreams table."""
    return dict(_DEFAULT_UPSTREAMS)


_VALID_PROTOCOLS = ("anthropic-messages", "openai-chat")
_VALID_CACHE_MODES = ("off", "stock")
_VALID_CACHE_BACKENDS = ("auto", "radix", "hicache", "lmcache")


def _parse_upstream_cache(raw: Any) -> UpstreamCacheConfig:
    if not isinstance(raw, dict):
        return UpstreamCacheConfig()
    mode = raw.get("mode", "stock")
    if mode not in _VALID_CACHE_MODES:
        mode = "stock"
    backend = raw.get("backend", "auto")
    if backend not in _VALID_CACHE_BACKENDS:
        backend = "auto"
    namespace = raw.get("security_namespace", "")
    ttl = raw.get("capability_ttl_seconds", 60)
    try:
        ttl_seconds = max(1, int(ttl))
    except (TypeError, ValueError):
        ttl_seconds = 60
    return UpstreamCacheConfig(
        mode=str(mode),
        backend=str(backend),
        tier_telemetry=raw.get("tier_telemetry") is True,
        security_namespace=str(namespace or ""),
        allow_cross_session=raw.get("allow_cross_session") is True,
        capability_ttl_seconds=ttl_seconds,
    )


def _parse_upstreams(raw: Any) -> dict[str, UpstreamConfig]:
    """Merge user-supplied upstreams over the defaults.

    Tolerates a missing / malformed section by falling back to defaults: this
    matches the "never block on user data" principle of the rest of the file.
    """
    out = default_upstreams()
    if not isinstance(raw, dict):
        return out
    for slug, entry in raw.items():
        if not isinstance(slug, str) or not isinstance(entry, dict):
            continue
        url = entry.get("url")
        engine = entry.get("engine")
        protocol = entry.get("protocol")
        via = entry.get("via") or ""
        if not isinstance(url, str) or not isinstance(engine, str):
            continue
        if protocol not in _VALID_PROTOCOLS:
            # Backfill from the default of the same slug if any; otherwise skip.
            if slug in _DEFAULT_UPSTREAMS:
                protocol = _DEFAULT_UPSTREAMS[slug].protocol
            else:
                continue
        out[slug] = UpstreamConfig(
            url=url.rstrip("/"),
            engine=engine,
            protocol=protocol,
            via=str(via),
            cache=_parse_upstream_cache(entry.get("cache")),
        )
    return out


def _serialize_upstreams(upstreams: dict[str, UpstreamConfig]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for slug, u in upstreams.items():
        entry: dict[str, Any] = {
            "url": u.url,
            "engine": u.engine,
            "protocol": u.protocol,
        }
        # Only persist ``via`` when non-empty so old configs / defaults stay
        # tidy on round-trip.
        if u.via:
            entry["via"] = u.via
        if u.cache != UpstreamCacheConfig():
            entry["cache"] = {
                "mode": u.cache.mode,
                "backend": u.cache.backend,
                "tier_telemetry": u.cache.tier_telemetry,
                "security_namespace": u.cache.security_namespace,
                "allow_cross_session": u.cache.allow_cross_session,
                "capability_ttl_seconds": u.cache.capability_ttl_seconds,
            }
        out[slug] = entry
    return out


@dataclass
class TelosConfig:
    """telos global config. ``_extra`` preserves unknown keys so they are not lost on write-back."""

    mode: str = DEFAULT_MODE
    gateway: GatewayConfig = field(default_factory=GatewayConfig)
    favorite_harness: str | None = None
    harness_executables: dict[str, str] = field(default_factory=dict)
    upstreams: dict[str, UpstreamConfig] = field(default_factory=default_upstreams)
    _extra: dict[str, Any] = field(default_factory=dict, repr=False)

    def anthropic_upstream_url(self) -> str:
        """The URL the legacy ``/v1/messages`` route forwards to.

        Read from ``upstreams.anthropic.url``; falls back to
        ``https://api.anthropic.com`` if the user removed that entry.
        """
        anth = self.upstreams.get("anthropic")
        return anth.url if anth is not None else "https://api.anthropic.com"

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = dict(self._extra)
        data["_schema"] = _SCHEMA_VERSION
        data["mode"] = self.mode
        data["gateway"] = {
            "host": self.gateway.host,
            "port": self.gateway.port,
            "usage_log": self.gateway.usage_log,
        }
        data["favorite_harness"] = self.favorite_harness
        data["harness_executables"] = dict(self.harness_executables)
        data["upstreams"] = _serialize_upstreams(self.upstreams)
        return data


_KNOWN_KEYS = {"_schema", "mode", "gateway", "favorite_harness",
               "harness_executables", "upstreams"}


def load_config() -> TelosConfig:
    """Read ``~/.telos/config.json``; returns all defaults if missing."""
    path = config_path()
    if not path.exists():
        return TelosConfig()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"{path} is not valid JSON ({e}). Please fix it manually, or delete the file and let telos regenerate it."
        ) from e
    if not isinstance(data, dict):
        raise RuntimeError(f"{path} top level must be a JSON object, but is actually {type(data).__name__}")

    gw_raw = data.get("gateway") or {}
    if not isinstance(gw_raw, dict):
        gw_raw = {}
    gateway = GatewayConfig(
        host=str(gw_raw.get("host", DEFAULT_GATEWAY_HOST)),
        port=int(gw_raw.get("port", DEFAULT_GATEWAY_PORT)),
        usage_log=str(gw_raw.get("usage_log", "") or ""),
    )
    execs_raw = data.get("harness_executables") or {}
    harness_executables = (
        {str(k): str(v) for k, v in execs_raw.items()}
        if isinstance(execs_raw, dict) else {}
    )
    fav = data.get("favorite_harness")
    upstreams = _parse_upstreams(data.get("upstreams"))
    extra = {k: v for k, v in data.items() if k not in _KNOWN_KEYS}
    return TelosConfig(
        mode=str(data.get("mode", DEFAULT_MODE)),
        gateway=gateway,
        favorite_harness=str(fav) if fav else None,
        harness_executables=harness_executables,
        upstreams=upstreams,
        _extra=extra,
    )


def save_config(cfg: TelosConfig) -> Path:
    """Atomically write back ``~/.telos/config.json``."""
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)
    return path


def update_config(**fields: Any) -> TelosConfig:
    """A convenient wrapper for load → modify fields → save.

    Supported fields: ``mode`` / ``favorite_harness``; for ``gateway`` subfields use
    ``gateway_host`` / ``gateway_port`` / ``gateway_usage_log``.
    """
    cfg = load_config()
    if "mode" in fields:
        cfg.mode = str(fields["mode"])
    if "favorite_harness" in fields:
        fav = fields["favorite_harness"]
        cfg.favorite_harness = str(fav) if fav else None
    if "gateway_host" in fields:
        cfg.gateway.host = str(fields["gateway_host"])
    if "gateway_port" in fields:
        cfg.gateway.port = int(fields["gateway_port"])
    if "gateway_usage_log" in fields:
        cfg.gateway.usage_log = str(fields["gateway_usage_log"] or "")
    save_config(cfg)
    return cfg


def revert_upstreams_owned_by(via: str) -> tuple[Path | None, list[str]]:
    """Undo upstream-table entries owned by a given installer.

    Each installer (hermes / openclaw / codex) tags the slugs it patches with
    ``via=<name>`` at install time. On uninstall we walk the table:

    - slug present in :data:`_DEFAULT_UPSTREAMS` → reset it to the built-in
      default (e.g. ``openrouter`` / ``deepseek`` / ``openai`` were defaults
      that the installer only added ``via`` to);
    - slug not a default → remove it entirely (e.g. ``codex-chatgpt`` is
      installer-created and has no canonical default to fall back to).

    Returns ``(saved_path, change_descriptions)``. If nothing matched, returns
    ``(None, [])`` and ``~/.telos/config.json`` is not rewritten.
    """
    if not via:
        return None, []
    cfg = load_config()
    changes: list[str] = []
    for slug, entry in list(cfg.upstreams.items()):
        if entry.via != via:
            continue
        default = _DEFAULT_UPSTREAMS.get(slug)
        if default is not None:
            if entry == default:
                continue
            cfg.upstreams[slug] = default
            changes.append(f"reset upstream {slug!r} to default (cleared via={via!r})")
        else:
            del cfg.upstreams[slug]
            changes.append(f"removed installer-added upstream {slug!r}")
    if not changes:
        return None, []
    return save_config(cfg), changes
