#!/usr/bin/env bash
# Reproducible SSR, ETag/304, memory (RSS), cold-start, and Tantivy search benchmark.
#
# Usage:
#   ./scripts/bench_ssr.sh                 # Benchmark both basic and accounts tiers
#   ./scripts/bench_ssr.sh basic           # Benchmark only basic tier
#   ./scripts/bench_ssr.sh accounts        # Benchmark only accounts tier
#   BENCH_JSON=/tmp/bench.json ./scripts/bench_ssr.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="${CARGO_TARGET_DIR:-$REPO_ROOT/target}"
TIERS=("${@:-basic accounts}")
# shellcheck disable=SC2206
TIERS=(${TIERS[*]})
BENCH_JSON="${BENCH_JSON:-}"

tier_cargo_flags() {
  case "$1" in
    basic) echo "--no-default-features --features server" ;;
    accounts) echo "--no-default-features --features server,accounts" ;;
    full) echo "--no-default-features --features server,payments" ;;
    *)
      echo "unknown tier: $1 (expected: basic | accounts | full)" >&2
      exit 2
      ;;
  esac
}

WORK_BASE="$(mktemp -d "${TMPDIR:-/tmp}/dp-bench.XXXXXX")"
SERVER_PID=""
cleanup() {
  if [ -n "$SERVER_PID" ]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  rm -rf "$WORK_BASE"
}
trap cleanup EXIT

RESULTS_DIR="$WORK_BASE/results"
mkdir -p "$RESULTS_DIR"

for idx in "${!TIERS[@]}"; do
  tier="${TIERS[$idx]}"
  flags="$(tier_cargo_flags "$tier")"
  echo "================================================================"
  echo "[bench] Building release server binary for tier: $tier ($flags)"
  echo "================================================================"
  # shellcheck disable=SC2086
  (cd "$REPO_ROOT" && CARGO_TARGET_DIR="$TARGET_DIR" cargo build --release -p app $flags)

  BIN_PATH="$TARGET_DIR/release/app"
  mkdir -p "$TARGET_DIR/release/public"
  BIN_BYTES="$(python3 -c 'import os,sys; print(os.path.getsize(sys.argv[1]))' "$BIN_PATH")"

  RUN_DIR="$WORK_BASE/run-$tier"
  IDX_DIR="$WORK_BASE/idx-$tier"
  mkdir -p "$RUN_DIR" "$IDX_DIR"
  ln -s "$REPO_ROOT/assets" "$RUN_DIR/assets"

  PORT=$((19120 + idx))
  START_NS="$(python3 -c 'import time; print(time.perf_counter_ns())')"

  (
    cd "$RUN_DIR"
    exec env -i \
      PATH="$PATH" \
      HOME="${HOME:-/tmp}" \
      IP="127.0.0.1" \
      PORT="$PORT" \
      RUST_LOG="warn" \
      CONTENT_DIR="" \
      RATE_LIMIT_DISABLED="1" \
      SEARCH_INDEX_DIR="$IDX_DIR" \
      BASE_URL="http://127.0.0.1:$PORT" \
      JWT_SECRET="bench-jwt-secret-key-for-reproducible-local-test-only-64b" \
      "$BIN_PATH" >"$RUN_DIR/server.log" 2>&1
  ) &
  SERVER_PID=$!

  READY_NS="$(
    python3 - "$PORT" "$SERVER_PID" <<'PY'
import http.client
import os
import sys
import time

port = int(sys.argv[1])
pid = int(sys.argv[2])
deadline = time.perf_counter() + 20.0
while time.perf_counter() < deadline:
  try:
    os.kill(pid, 0)
  except OSError:
    sys.exit("server process exited prematurely")
  try:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=0.5)
    conn.request("GET", "/healthz")
    resp = conn.getresponse()
    resp.read()
    conn.close()
    if resp.status == 200:
      print(time.perf_counter_ns())
      sys.exit(0)
  except OSError:
    time.sleep(0.005)
sys.exit("timed out waiting for /healthz")
PY
  )"

  COLD_START_MS="$(python3 -c "print(round(($READY_NS - $START_NS) / 1e6, 2))")"
  # Allow background startup incremental reindex (~50ms on built-in assets) to settle before idle RSS snapshot
  sleep 0.4
  IDLE_RSS_KB="$(ps -o rss= -p "$SERVER_PID" | tr -d ' ')"

  python3 - "$PORT" "$tier" "$BIN_BYTES" "$COLD_START_MS" "$IDLE_RSS_KB" "$SERVER_PID" "$RESULTS_DIR/$tier.json" <<'PY'
import concurrent.futures
import http.client
import json
import subprocess
import sys
import threading
import time

port = int(sys.argv[1])
tier = sys.argv[2]
bin_bytes = int(sys.argv[3])
cold_start_ms = float(sys.argv[4])
idle_rss_kb = int(sys.argv[5])
server_pid = sys.argv[6]
out_json = sys.argv[7]

def percentile(sorted_vals, pct):
  if not sorted_vals:
    return 0.0
  k = (len(sorted_vals) - 1) * (pct / 100.0)
  f = int(k)
  c = min(f + 1, len(sorted_vals) - 1)
  if f == c:
    return sorted_vals[f]
  return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)

def send_req(conn, method, path, body, hdrs):
  try:
    conn.request(method, path, body=body, headers=hdrs)
    resp = conn.getresponse()
    data = resp.read()
    return conn, resp, data
  except (http.client.HTTPException, OSError):
    conn.close()
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5.0)
    conn.request(method, path, body=body, headers=hdrs)
    resp = conn.getresponse()
    data = resp.read()
    return conn, resp, data

def bench_serial(method, path, body=None, headers=None, n=200, warmup=20):
  hdrs = dict(headers or {})
  conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5.0)
  last_status = 0
  last_headers = {}
  last_bytes = 0
  samples = []
  for i in range(warmup + n):
    t0 = time.perf_counter_ns()
    conn, resp, data = send_req(conn, method, path, body, hdrs)
    dt_ms = (time.perf_counter_ns() - t0) / 1e6
    if i == 0:
      last_status = resp.status
      last_headers = {k.lower(): v for k, v in resp.getheaders()}
      last_bytes = len(data)
    if i >= warmup:
      samples.append(dt_ms)
  conn.close()
  samples.sort()
  return {
    "status": last_status,
    "bytes": last_bytes,
    "etag": last_headers.get("etag"),
    "cache_control": last_headers.get("cache-control"),
    "p50_ms": round(percentile(samples, 50), 3),
    "p95_ms": round(percentile(samples, 95), 3),
    "p99_ms": round(percentile(samples, 99), 3),
  }

def bench_concurrent(method, path, body=None, headers=None, concurrency=16, total=1600):
  hdrs = dict(headers or {})
  per_worker = total // concurrency
  tls = threading.local()

  def worker(_):
    conn = getattr(tls, "conn", None)
    if conn is None:
      conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5.0)
      tls.conn = conn
    lat = []
    non_ok = 0
    for _ in range(per_worker):
      t0 = time.perf_counter_ns()
      conn, resp, _ = send_req(conn, method, path, body, hdrs)
      tls.conn = conn
      lat.append((time.perf_counter_ns() - t0) / 1e6)
      if resp.status not in (200, 304):
        non_ok += 1
    conn.close()
    return lat, non_ok

  t_start = time.perf_counter()
  all_lat = []
  errors = 0
  with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
    for lat, non_ok in pool.map(worker, range(concurrency)):
      all_lat.extend(lat)
      errors += non_ok
  elapsed = time.perf_counter() - t_start
  all_lat.sort()
  actual_total = per_worker * concurrency
  return {
    "qps": round(actual_total / elapsed, 1),
    "p50_ms": round(percentile(all_lat, 50), 3),
    "p99_ms": round(percentile(all_lat, 99), 3),
    "errors": errors,
  }

cases = [
  ("home", "GET", "/", None, {}),
  ("quickstart_zh", "GET", "/docs/guide/quickstart", None, {}),
  ("quickstart_en", "GET", "/docs/guide/quickstart?lang=en", None, {}),
  ("mdx_showcase", "GET", "/docs/content/mdx", None, {}),
  ("sitemap_xml", "GET", "/sitemap.xml", None, {}),
  ("feed_xml", "GET", "/feed.xml", None, {}),
]

endpoints = {}
for name, method, path, body, hdrs in cases:
  serial = bench_serial(method, path, body=body, headers=hdrs)
  conc = bench_concurrent(method, path, body=body, headers=hdrs)
  endpoints[name] = {"path": path, "serial": serial, "concurrent_c16": conc}

# Conditional GET check using ETag from quickstart_zh if present (or synthetic probe if ETag not yet supported)
probe_etag = endpoints["quickstart_zh"]["serial"]["etag"] or 'W/"probe-before-etag"'
cond_hdrs = {"If-None-Match": probe_etag}
cond_serial = bench_serial("GET", "/docs/guide/quickstart", headers=cond_hdrs)
cond_conc = bench_concurrent("GET", "/docs/guide/quickstart", headers=cond_hdrs)
endpoints["quickstart_conditional_304"] = {
  "path": "/docs/guide/quickstart (If-None-Match)",
  "serial": cond_serial,
  "concurrent_c16": cond_conc,
}

# Tantivy full-text search RPC
search_body = json.dumps({"q": "DioxusPress", "kind": None, "limit": 10}).encode("utf-8")
search_hdrs = {"Content-Type": "application/json"}
search_serial = bench_serial("POST", "/api/search/query", body=search_body, headers=search_hdrs)
search_conc = bench_concurrent("POST", "/api/search/query", body=search_body, headers=search_hdrs)
endpoints["search_query"] = {
  "path": "POST /api/search/query",
  "serial": search_serial,
  "concurrent_c16": search_conc,
}

rss_after_kb = int(
  subprocess.check_output(["ps", "-o", "rss=", "-p", server_pid]).decode().strip()
)

report = {
  "tier": tier,
  "binary_bytes": bin_bytes,
  "binary_mib": round(bin_bytes / (1024 * 1024), 2),
  "cold_start_ms": cold_start_ms,
  "idle_rss_mib": round(idle_rss_kb / 1024.0, 2),
  "post_load_rss_mib": round(rss_after_kb / 1024.0, 2),
  "endpoints": endpoints,
}

with open(out_json, "w", encoding="utf-8") as f:
  json.dump(report, f, indent=2)

print(
  f"\nTier `{tier}`: binary={report['binary_mib']} MiB | "
  f"cold_start={report['cold_start_ms']} ms | "
  f"idle_rss={report['idle_rss_mib']} MiB | "
  f"post_load_rss={report['post_load_rss_mib']} MiB"
)
print(f"{'Endpoint':<28} {'Status':<7} {'P50(ms)':<9} {'P99(ms)':<9} {'c=16 QPS':<11} {'c=16 P99(ms)':<12}")
print("-" * 80)
for key, item in endpoints.items():
  s = item["serial"]
  c = item["concurrent_c16"]
  print(
    f"{key:<28} {s['status']:<7} {s['p50_ms']:<9.3f} {s['p99_ms']:<9.3f} "
    f"{c['qps']:<11.1f} {c['p99_ms']:<12.3f}"
  )
PY

  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
  SERVER_PID=""
done

if [ -n "$BENCH_JSON" ]; then
  python3 - "$RESULTS_DIR" "$BENCH_JSON" <<'PY'
import glob
import json
import os
import sys

res_dir, out_path = sys.argv[1], sys.argv[2]
combined = {}
for p in sorted(glob.glob(os.path.join(res_dir, "*.json"))):
  with open(p, "r", encoding="utf-8") as f:
    data = json.load(f)
  combined[data["tier"]] = data
with open(out_path, "w", encoding="utf-8") as f:
  json.dump(combined, f, indent=2)
print(f"\nSaved JSON benchmark summary to {out_path}")
PY
fi
