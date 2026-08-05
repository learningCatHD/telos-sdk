"""Shared OpenAI Chat Completions renderer for Telos engine adapters."""

from __future__ import annotations

from typing import Any

from telos.ir import Band, TelosIR


def render_chat_completions(ir: TelosIR, *, model: str) -> dict[str, Any]:
    """Render canonical IR without losing assistant tool calls or tool results."""
    system_blocks = sorted(
        ir.system,
        key=lambda block: 0 if block.band is not Band.DROP else 1,
    )
    system_text = "\n\n".join(str(block.payload) for block in system_blocks)

    wire_messages: list[dict[str, Any]] = []
    if system_text.strip():
        wire_messages.append({"role": "system", "content": system_text})

    for message in ir.messages:
        ordered = sorted(
            message.blocks,
            key=lambda block: 0 if block.band is not Band.DROP else 1,
        )
        if message.role == "user":
            for block in ordered:
                if block.kind != "tool_result":
                    continue
                payload = block.payload or {}
                wire_messages.append({
                    "role": "tool",
                    "tool_call_id": payload.get("tool_use_id", ""),
                    "content": str(payload.get("content", "")),
                })
            text_parts = [str(block.payload) for block in ordered if block.kind == "text"]
            joined = "\n".join(part for part in text_parts if part)
            if joined.strip():
                wire_messages.append({"role": "user", "content": joined})
        elif message.role == "assistant":
            text_parts = [str(block.payload) for block in ordered if block.kind == "text"]
            tool_calls = [block.payload for block in ordered if block.kind == "tool_use"]
            reasoning_parts = [
                str(block.payload) for block in ordered if block.kind == "reasoning"
            ]
            entry: dict[str, Any] = {
                "role": "assistant",
                "content": "\n".join(text_parts) if text_parts else None,
            }
            if tool_calls:
                entry["tool_calls"] = list(tool_calls)
            if reasoning_parts:
                entry["reasoning_content"] = "\n".join(reasoning_parts)
            wire_messages.append(entry)

    wire: dict[str, Any] = {"model": model, "messages": wire_messages}
    if ir.tools:
        wire["tools"] = [block.payload for block in ir.tools]
    return wire
