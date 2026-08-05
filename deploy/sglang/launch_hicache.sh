#!/usr/bin/env bash
set -euo pipefail

# Stock SGLang + HiCache launcher for Telos Phase 2.
# Required: MODEL_PATH=/path/or/huggingface-id

: "${MODEL_PATH:?MODEL_PATH is required}"

SGLANG_HOST="${SGLANG_HOST:-0.0.0.0}"
SGLANG_PORT="${SGLANG_PORT:-30000}"
TP_SIZE="${TP_SIZE:-1}"
HICACHE_RATIO="${HICACHE_RATIO:-2}"
HICACHE_WRITE_POLICY="${HICACHE_WRITE_POLICY:-write_through}"
HICACHE_IO_BACKEND="${HICACHE_IO_BACKEND:-kernel}"
HICACHE_MEM_LAYOUT="${HICACHE_MEM_LAYOUT:-layer_first}"

case "$HICACHE_WRITE_POLICY" in
  write_back|write_through|write_through_selective) ;;
  *)
    echo "invalid HICACHE_WRITE_POLICY: $HICACHE_WRITE_POLICY" >&2
    exit 2
    ;;
esac

case "$HICACHE_MEM_LAYOUT" in
  layer_first|page_first) ;;
  *)
    echo "invalid HICACHE_MEM_LAYOUT: $HICACHE_MEM_LAYOUT" >&2
    exit 2
    ;;
esac

args=(
  python -m sglang.launch_server
  --model-path "$MODEL_PATH"
  --host "$SGLANG_HOST"
  --port "$SGLANG_PORT"
  --tp-size "$TP_SIZE"
  --enable-hierarchical-cache
  --hicache-write-policy "$HICACHE_WRITE_POLICY"
  --hicache-io-backend "$HICACHE_IO_BACKEND"
  --hicache-mem-layout "$HICACHE_MEM_LAYOUT"
  --enable-cache-report
  --enable-metrics
)

# hicache-size and hicache-ratio are alternative capacity controls.
if [[ -n "${HICACHE_SIZE_GB:-}" ]]; then
  args+=(--hicache-size "$HICACHE_SIZE_GB")
else
  args+=(--hicache-ratio "$HICACHE_RATIO")
fi

if [[ -n "${HICACHE_STORAGE_BACKEND:-}" ]]; then
  args+=(--hicache-storage-backend "$HICACHE_STORAGE_BACKEND")
fi
if [[ -n "${SGLANG_API_KEY:-}" ]]; then
  args+=(--api-key "$SGLANG_API_KEY")
fi
if [[ -n "${SGLANG_ADMIN_API_KEY:-}" ]]; then
  args+=(--admin-api-key "$SGLANG_ADMIN_API_KEY")
fi

# Extra pinned-version flags may be appended by the operator, for example a
# storage-backend-specific JSON config. They remain visible in process logs.
exec "${args[@]}" "$@"
