"""``telos.config`` tests: round-trip / defaults / bad JSON / unknown-key preservation."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import telos.config as cfgmod


def _tmp_home() -> Path:
    return Path(tempfile.mkdtemp(prefix="telos-cfg-"))


def _with_home(home: Path):
    os.environ["TELOS_HOME"] = str(home)


def test_load_missing_returns_defaults() -> None:
    _with_home(_tmp_home())
    c = cfgmod.load_config()
    assert c.mode == "telos"
    assert c.gateway.port == 7171
    assert c.favorite_harness is None
    assert c.upstreams["openai"].url == "https://api.openai.com"
    assert c.upstreams["openai"].engine == "openai"
    assert c.upstreams["local-sglang"].url == "http://127.0.0.1:30000"
    assert c.upstreams["local-sglang"].cache.mode == "stock"
    assert c.upstreams["local-sglang"].cache.backend == "radix"
    assert c.upstreams["local-sglang"].cache.tier_telemetry is False
    print("✓ test_load_missing_returns_defaults")


def test_save_load_round_trip() -> None:
    _with_home(_tmp_home())
    c = cfgmod.load_config()
    c.mode = "both"
    c.gateway.port = 9999
    c.favorite_harness = "codex"
    c.harness_executables = {"openclaw": "openclaw-beta"}
    cfgmod.save_config(c)

    c2 = cfgmod.load_config()
    assert c2.mode == "both"
    assert c2.gateway.port == 9999
    assert c2.favorite_harness == "codex"
    assert c2.harness_executables["openclaw"] == "openclaw-beta"
    print("✓ test_save_load_round_trip")


def test_update_config() -> None:
    _with_home(_tmp_home())
    cfgmod.update_config(mode="rtk", gateway_port=8080)
    c = cfgmod.load_config()
    assert c.mode == "rtk"
    assert c.gateway.port == 8080
    print("✓ test_update_config")


def test_unknown_keys_preserved() -> None:
    home = _tmp_home()
    _with_home(home)
    path = home / "config.json"
    path.write_text(json.dumps({"mode": "both", "future_field": {"x": 1}}))
    c = cfgmod.load_config()
    cfgmod.save_config(c)
    reloaded = json.loads(path.read_text())
    assert reloaded["future_field"] == {"x": 1}
    print("✓ test_unknown_keys_preserved")


def test_bad_json_raises() -> None:
    home = _tmp_home()
    _with_home(home)
    (home / "config.json").write_text("{not json")
    try:
        cfgmod.load_config()
    except RuntimeError as e:
        assert "JSON" in str(e)
        print("✓ test_bad_json_raises")
        return
    raise AssertionError("expected RuntimeError")


def test_upstream_cache_config_round_trip() -> None:
    _with_home(_tmp_home())
    c = cfgmod.load_config()
    c.upstreams["sglang-hicache"] = cfgmod.UpstreamConfig(
        url="http://127.0.0.1:31000",
        engine="sglang",
        protocol="openai-chat",
        cache=cfgmod.UpstreamCacheConfig(
            mode="stock",
            backend="hicache",
            tier_telemetry=True,
            security_namespace="opaque:local-dev",
            allow_cross_session=False,
            capability_ttl_seconds=90,
        ),
    )
    cfgmod.save_config(c)
    loaded = cfgmod.load_config().upstreams["sglang-hicache"]
    assert loaded.cache.security_namespace == "opaque:local-dev"
    assert loaded.cache.backend == "hicache"
    assert loaded.cache.tier_telemetry is True
    assert loaded.cache.allow_cross_session is False
    assert loaded.cache.capability_ttl_seconds == 90


def test_unknown_cache_mode_fails_safe_to_stock() -> None:
    home = _tmp_home()
    _with_home(home)
    (home / "config.json").write_text(json.dumps({
        "upstreams": {
            "future": {
                "url": "http://127.0.0.1:32000",
                "engine": "sglang",
                "protocol": "openai-chat",
                "cache": {"mode": "future-active-control"},
            },
        },
    }))
    assert cfgmod.load_config().upstreams["future"].cache.mode == "stock"


def test_unknown_cache_backend_fails_safe_to_auto() -> None:
    home = _tmp_home()
    _with_home(home)
    (home / "config.json").write_text(json.dumps({
        "upstreams": {
            "future": {
                "url": "http://127.0.0.1:32000",
                "engine": "sglang",
                "protocol": "openai-chat",
                "cache": {
                    "backend": "unknown-private-plugin",
                    "tier_telemetry": True,
                },
            },
        },
    }))
    cache = cfgmod.load_config().upstreams["future"].cache
    assert cache.backend == "auto"
    assert cache.tier_telemetry is True


def test_revert_upstreams_owned_by() -> None:
    """The installer-uninstall path uses this helper to undo per-installer
    edits in ~/.telos/config.json. Three cases must hold:

    1. A *default* slug that an installer tagged with ``via`` is reset to its
       canonical default (URL/engine kept, ``via`` cleared).
    2. A *non-default* slug the installer added is removed entirely.
    3. Slugs not owned by this installer are left untouched.
    """
    _with_home(_tmp_home())
    c = cfgmod.load_config()
    # (1) Default slug an installer has tagged.
    c.upstreams["openrouter"] = cfgmod.UpstreamConfig(
        url="https://openrouter.ai/api", engine="deepseek",
        protocol="openai-chat", via="hermes",
    )
    # (2) Installer-added slug not in defaults.
    c.upstreams["codex-chatgpt"] = cfgmod.UpstreamConfig(
        url="https://chatgpt.com/backend-api/codex", engine="openai",
        protocol="openai-chat", via="codex",
    )
    # (3) Slug owned by someone else.
    c.upstreams["deepseek"] = cfgmod.UpstreamConfig(
        url="https://api.deepseek.com", engine="deepseek",
        protocol="openai-chat", via="openclaw",
    )
    cfgmod.save_config(c)

    saved, changes = cfgmod.revert_upstreams_owned_by("hermes")
    assert saved is not None
    assert any("openrouter" in s for s in changes), changes

    c2 = cfgmod.load_config()
    # Default slug bounced back to the canonical default (no via).
    assert c2.upstreams["openrouter"].via == ""
    assert c2.upstreams["openrouter"].url == "https://openrouter.ai/api"
    # Other-owner slug untouched.
    assert c2.upstreams["deepseek"].via == "openclaw"
    # Codex slug still there (different owner).
    assert "codex-chatgpt" in c2.upstreams

    # Now uninstall codex; the codex-chatgpt slug disappears entirely.
    saved, changes = cfgmod.revert_upstreams_owned_by("codex")
    assert saved is not None
    c3 = cfgmod.load_config()
    assert "codex-chatgpt" not in c3.upstreams

    # Calling again is a no-op (idempotent).
    saved, changes = cfgmod.revert_upstreams_owned_by("codex")
    assert saved is None and changes == []
    print("✓ test_revert_upstreams_owned_by")


def main() -> None:
    test_load_missing_returns_defaults()
    test_save_load_round_trip()
    test_update_config()
    test_unknown_keys_preserved()
    test_bad_json_raises()
    test_upstream_cache_config_round_trip()
    test_unknown_cache_mode_fails_safe_to_stock()
    test_unknown_cache_backend_fails_safe_to_auto()
    test_revert_upstreams_owned_by()
    print("\nall config tests passed.")


if __name__ == "__main__":
    main()
