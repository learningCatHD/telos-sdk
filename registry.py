"""Plugin / adapter registry (load by name, avoiding hard top-level module dependencies).

Both harnesses and engines are instantiated here, ensuring the bridge does not
directly import any concrete implementation —— this is the code-level realization
of "the three tiers only pass values downward, never reference upward".
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from telos.harness.base import HarnessPlugin
    from telos.engine.base import EngineAdapter


# Canonical harness name → human-facing display name shown on dashboards/reports.
# These four are the user-visible identities; each owns its own installer
# (init/<name>.py) and its own parser plugin (harness/<name>.py).
_HARNESS_DISPLAY_NAMES: dict[str, str] = {
    "claude-code": "Claude Code",
    "hermes":      "Hermes",
    "openclaw":    "OpenClaw",
    "codex":       "Codex",
    "telos":       "Telos",   # internal — not user-installable
}


def harness_display_name(name: str) -> str:
    """Map a harness name to its dashboard display name.

    Unknown names (``passthrough`` / ``rtk-only`` / ``?``) are returned as-is
    so the dashboard's "Breakdown by harness" can surface anonymous traffic
    without a lookup miss.
    """
    if not name:
        return name
    return _HARNESS_DISPLAY_NAMES.get(name, name)


def load_harness(name: str) -> "HarnessPlugin":
    """Load a harness plugin by name.

    Supported: ``claude-code``, ``hermes``, ``openclaw``, ``codex``, ``telos``.

    Each name maps one-to-one to ``harness/<name with hyphens → underscores>.py``.
    Aliases were removed in 2026-05; callers must pass the canonical name.
    """
    if name == "claude-code":
        from telos.harness.claude_code import ClaudeCodePlugin
        return ClaudeCodePlugin()
    if name == "hermes":
        from telos.harness.hermes import HermesPlugin
        return HermesPlugin()
    if name == "openclaw":
        from telos.harness.openclaw import OpenClawPlugin
        return OpenClawPlugin()
    if name == "codex":
        from telos.harness.codex import CodexPlugin
        return CodexPlugin()
    if name == "telos":
        from telos.harness.telos import TelosPlugin
        return TelosPlugin()
    raise ValueError(f"Unknown harness plugin: {name!r}")


def load_engine(name: str) -> "EngineAdapter":
    """Load an engine adapter by name.

    Supported:
    - Closed-source APIs: ``anthropic``, ``openai``, ``deepseek``
    - Open-source inference: ``vllm``, ``sglang``. Only adapters backed by a
      verified control-plane extension implement bidirectional operations;
      stock SGLang relies on automatic RadixAttention prefix reuse.
    """
    if name == "anthropic":
        from telos.engine.anthropic import AnthropicAdapter
        return AnthropicAdapter()
    if name == "openai":
        from telos.engine.openai import OpenAIAdapter
        return OpenAIAdapter()
    if name == "deepseek":
        from telos.engine.deepseek import DeepSeekAdapter
        return DeepSeekAdapter()
    if name == "vllm":
        from telos.engine.vllm import VLLMAdapter
        return VLLMAdapter()
    if name == "sglang":
        from telos.engine.sglang import SGLangAdapter
        return SGLangAdapter()
    raise ValueError(f"Unknown engine adapter: {name!r}")
