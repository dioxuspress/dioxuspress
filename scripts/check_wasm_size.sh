#!/usr/bin/env bash
# E5.4: Frontend WebAssembly bundle size guardrail.
#
# Compiles `app` under `--profile wasm-release` (`opt-level = "z"`, `lto = "fat"`)
# for both `basic` (`--no-default-features --features web`) and `full` (default `web` +
# `payments`) tiers, and verifies raw `.wasm` and `gzip -9` byte sizes stay within budget.
#
# Override any budget via environment variables (used for CI and red/green gate self-tests):
#   MAX_BASIC_WASM_RAW_BYTES (default: 6800000)
#   MAX_BASIC_WASM_GZ_BYTES  (default: 1950000)
#   MAX_FULL_WASM_RAW_BYTES  (default: 9200000)
#   MAX_FULL_WASM_GZ_BYTES   (default: 2550000)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="${CARGO_TARGET_DIR:-$REPO_ROOT/target}"

MAX_BASIC_WASM_RAW_BYTES="${MAX_BASIC_WASM_RAW_BYTES:-6800000}"
MAX_BASIC_WASM_GZ_BYTES="${MAX_BASIC_WASM_GZ_BYTES:-1950000}"
MAX_FULL_WASM_RAW_BYTES="${MAX_FULL_WASM_RAW_BYTES:-9200000}"
MAX_FULL_WASM_GZ_BYTES="${MAX_FULL_WASM_GZ_BYTES:-2550000}"

SKIP_BUILD="${SKIP_WASM_BUILD:-0}"
FAILED=0

check_tier_wasm() {
  local tier="$1"
  local cargo_flags="$2"
  local max_raw="$3"
  local max_gz="$4"

  if [ "$SKIP_BUILD" != "1" ]; then
    echo "==> Building wasm-release ($tier): cargo build --profile wasm-release -p app --target wasm32-unknown-unknown $cargo_flags"
    # shellcheck disable=SC2086
    (cd "$REPO_ROOT" && CARGO_TARGET_DIR="$TARGET_DIR" cargo build --profile wasm-release -p app --target wasm32-unknown-unknown $cargo_flags)
  fi

  local wasm_path="$TARGET_DIR/wasm32-unknown-unknown/wasm-release/app.wasm"
  if [ ! -f "$wasm_path" ]; then
    echo "FAIL [$tier]: missing $wasm_path" >&2
    FAILED=1
    return
  fi

  local raw_bytes gz_bytes br_info
  raw_bytes="$(wc -c < "$wasm_path" | tr -d ' ')"
  gz_bytes="$(gzip -9 -c "$wasm_path" | wc -c | tr -d ' ')"
  if command -v brotli >/dev/null 2>&1; then
    br_info="$(brotli -q 11 -c "$wasm_path" | wc -c | tr -d ' ') B"
  else
    br_info="n/a (brotli CLI not installed)"
  fi

  printf '[wasm-size] %-6s raw=%s B (limit %s B) | gzip=%s B (limit %s B) | brotli=%s\n' \
    "$tier" "$raw_bytes" "$max_raw" "$gz_bytes" "$max_gz" "$br_info"

  if [ "$raw_bytes" -gt "$max_raw" ]; then
    echo "FAIL [$tier]: raw .wasm size ${raw_bytes} B exceeds budget ${max_raw} B" >&2
    FAILED=1
  fi
  if [ "$gz_bytes" -gt "$max_gz" ]; then
    echo "FAIL [$tier]: gzipped .wasm.gz size ${gz_bytes} B exceeds budget ${max_gz} B" >&2
    FAILED=1
  fi
}

check_tier_wasm "basic" "--no-default-features --features web" "$MAX_BASIC_WASM_RAW_BYTES" "$MAX_BASIC_WASM_GZ_BYTES"
check_tier_wasm "full" "" "$MAX_FULL_WASM_RAW_BYTES" "$MAX_FULL_WASM_GZ_BYTES"

if [ "$FAILED" -ne 0 ]; then
  echo "wasm bundle size check FAILED" >&2
  exit 1
fi
echo "wasm bundle size check ok"
