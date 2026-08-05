"""Telos harness plugin.

Input contract: a ``raw_request`` in **OpenAI ChatCompletions** shape:

::

    {
      "model": "...",
      "messages": [
        {"role": "system",    "content": "..."},
        {"role": "user",      "content": "..."},
        {"role": "assistant", "content": "...", "tool_calls": [...]},
        {"role": "tool",      "content": "...", "tool_call_id": "..."},
      ],
      "tools": [{"type": "function", "function": {...}}, ...],
    }

This is the format emitted jointly by telos's vendored Hermes
``mini_swe_runner`` and the upper chat loop (via
``openai.OpenAI().chat.completions.create``).

Differences from ``hermes.py`` (which parses the Anthropic ``/v1/messages`` shape):

- ``messages[].content`` is usually a string, not a content-block array;
- a sub-call forms a separate message via ``role="tool"``, instead of being
  a ``tool_result`` block embedded in a user message;
- ``tools[].function.parameters`` instead of ``tools[].input_schema``.

ref-pool triggering likewise relies on ``<file path="...">...</file>`` blocks
(>2KB) — a Telos agent will not necessarily stuff such an envelope into a
SWE-bench prompt, but compatibility is kept.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping

from telos.harness._user_split import split_user_text
from telos.harness.base import HarnessPlugin
from telos.ir import (
    Band,
    TelosBlock,
    TelosHints,
    TelosIR,
    TelosMessage,
)


_REFPOOL_THRESHOLD = 2048
_FILE_BLOCK_RE = re.compile(r'<file path="([^"]+)">(.*?)</file>', re.DOTALL)


def _classify_openai_tool(t: Mapping[str, Any]) -> tuple[str, str | None]:
    """Classify an OpenAI function-tool: relies solely on upstream metadata.

    OpenAI's ``tools`` array has no native builtin/mcp distinction; if a
    harness wants to distinguish further, it must shape ``metadata.source``
    (optionally plus ``metadata.mcp_server``) for each tool in the
    ``raw_request`` it sends up. Otherwise it defaults to ``user``.
    """
    meta = t.get("metadata") if isinstance(t, Mapping) else None
    if isinstance(meta, Mapping):
        src = meta.get("source")
        server = meta.get("mcp_server")
        if isinstance(src, str):
            return src, server if isinstance(server, str) else None
    return "user", None


def _slug_from_path(path: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", path).strip("-").lower() or "file"


def _coerce_content_to_text(content: Any) -> str:
    """An OpenAI message ``content`` may be a str or list[{type,text}]; normalize to a string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                else:
                    parts.append(json.dumps(item, ensure_ascii=False, sort_keys=True))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


class TelosPlugin(HarnessPlugin):
    def parse(
        self,
        raw_request: Mapping[str, Any],
        *,
        session_id: str,
        engine: str,
        model: str = "",
        expected_turns: int = 0,
    ) -> TelosIR:
        ref_pool: dict[str, TelosBlock] = {}

        # ---- tools (OpenAI function-tool schema) ----
        def _build_tool(i: int, t: Mapping[str, Any]) -> TelosBlock:
            source, mcp_server = _classify_openai_tool(t)
            extra: dict[str, Any] = {"source": source}
            if mcp_server:
                extra["mcp_server"] = mcp_server
            return TelosBlock(
                id=f"tool:{(t.get('function') or {}).get('name', i)}",
                band=Band.PIN,
                kind="tool_def",
                payload=t,
                source_tag="telos/tools",
                extra=extra,
            )

        tools = tuple(
            _build_tool(i, t)
            for i, t in enumerate(raw_request.get("tools") or [])
        )

        # ---- under the OpenAI shape, system + messages are mixed into messages[].
        # Split: the leading run of role=system entries goes into the system segment; the rest into messages.
        raw_messages = list(raw_request.get("messages") or [])
        system_blocks: list[TelosBlock] = []
        cursor = 0
        for cursor, msg in enumerate(raw_messages):
            if msg.get("role") != "system":
                break
            text = _coerce_content_to_text(msg.get("content"))
            stripped = text
            for m in _FILE_BLOCK_RE.finditer(text):
                path, body = m.group(1), m.group(2)
                if len(body) > _REFPOOL_THRESHOLD:
                    slug = _slug_from_path(path)
                    if slug not in ref_pool:
                        ref_pool[slug] = TelosBlock(
                            id=f"ref:{slug}",
                            band=Band.FOLD,
                            kind="text",
                            payload=body,
                            ref_slug=slug,
                            source_tag="telos/file-block",
                        )
                    stripped = stripped.replace(m.group(0), f"[ref:{slug}]")
            system_blocks.append(TelosBlock(
                id=f"system/{len(system_blocks)}",
                band=Band.PIN,
                kind="text",
                payload=stripped,
                source_tag="telos/system",
            ))
        else:
            # all system; the cursor lands at the end, advance it manually
            cursor = len(raw_messages)

        # ---- the remaining messages (user / assistant / tool) ----
        messages: list[TelosMessage] = []
        for mi, msg in enumerate(raw_messages[cursor:], start=cursor):
            role = msg.get("role")
            text_content = _coerce_content_to_text(msg.get("content"))

            if role == "user":
                blocks = list(split_user_text(text_content, base_id=f"msg{mi}"))
                messages.append(TelosMessage(role="user", blocks=tuple(blocks)))

            elif role == "assistant":
                blocks: list[TelosBlock] = []
                if text_content:
                    blocks.append(TelosBlock(
                        id=f"msg{mi}/at",
                        band=Band.FOLD,
                        kind="text",
                        payload=text_content,
                        source_tag="telos/assistant-text",
                    ))
                # Preserve ``reasoning_content`` (DeepSeek / OpenAI o-series
                # thinking-mode field). Reasoning models REQUIRE the previous
                # turn's reasoning_content to be echoed back on the next call,
                # so this block must round-trip verbatim — drop it and the
                # upstream returns "reasoning_content in the thinking mode
                # must be passed back to the API" (HTTP 400).
                reasoning = msg.get("reasoning_content")
                if isinstance(reasoning, str) and reasoning:
                    blocks.append(TelosBlock(
                        id=f"msg{mi}/ar",
                        band=Band.FOLD,
                        kind="reasoning",
                        payload=reasoning,
                        source_tag="telos/assistant-reasoning",
                    ))
                for ti, tc in enumerate(msg.get("tool_calls") or []):
                    blocks.append(TelosBlock(
                        id=f"msg{mi}/au{ti}",
                        band=Band.FOLD,
                        kind="tool_use",
                        payload=tc,
                        source_tag="telos/assistant-tool-use",
                    ))
                if not blocks:
                    # empty assistant message: skip, to avoid an empty message in the IR
                    continue
                messages.append(TelosMessage(role="assistant", blocks=tuple(blocks)))

            elif role == "tool":
                # OpenAI: a standalone role=tool message. In the TELOS protocol a
                # tool_result must be attached inside a user message (aligned with
                # Anthropic), so we wrap it into a user message.
                payload = {
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id", ""),
                    "content": text_content,
                }
                blocks = (TelosBlock(
                    id=f"msg{mi}/tr",
                    band=Band.FOLD,
                    kind="tool_result",
                    payload=payload,
                    source_tag="telos/tool-result",
                ),)
                messages.append(TelosMessage(role="user", blocks=blocks))

            elif role == "system":
                # a system message appearing mid-stream (uncommon): insert it as a PIN text into the user segment.
                blocks = (TelosBlock(
                    id=f"msg{mi}/sys",
                    band=Band.PIN,
                    kind="text",
                    payload=text_content,
                    source_tag="telos/inline-system",
                ),)
                messages.append(TelosMessage(role="user", blocks=blocks))

        return TelosIR(
            session_id=session_id,
            tools=tools,
            system=tuple(system_blocks),
            messages=tuple(messages),
            ref_pool=ref_pool,
            hints=TelosHints(
                engine=engine,  # type: ignore[arg-type]
                model=model or raw_request.get("model", ""),
                expected_turns=expected_turns,
            ),
        )
