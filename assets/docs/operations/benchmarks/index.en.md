---
title: "Verified Benchmarks & Size Guardrails: Memory, SSR Latency, Tantivy Search & WASM Budgets"
sidebar_label: Benchmarks & Size Guards
sidebar_position: 6
description: Reproducible DioxusPress performance numbers covering memory footprint (RSS), binary/container sizes, SSR & ETag 304 throughput, Tantivy search latency, and CI WASM size budgets.
keywords: [benchmarks, memory footprint RSS, SSR latency, ETag 304, Tantivy QPS, WASM bundle size]
---

# Verified Benchmarks & CI Size Guardrails

DioxusPress follows a strict **"measure first, optimize by verified impact, and guard in CI"** discipline. Every metric below comes directly from built-in scripts (`./scripts/bench_ssr.sh` and `./scripts/check_wasm_size.sh`) that you can reproduce locally in minutes.

---

## 1. Binary, Container Image & Memory Footprint

Measured across our multi-site production stack (`debian:trixie-slim` runtime images):

| Component / Tier | Binary / Image Size | Cold Start to `/healthz 200` | Idle Memory (RSS) | Typical Workload |
| :--- | :--- | :--- | :--- | :--- |
| **`dioxuspress-gateway`** (Pingora TLS SNI proxy) | `143 MB` image | `< 50 ms` | **`6.4 MiB`** | Multi-domain TLS termination, HTTP/2, and `301` host redirects |
| **`basic` Zero-DB Tier** (`--no-default-features --features server`) | `34.2 MiB` bin / `269 MB` image | **`547 ms`** | **`91.7 MiB`** (includes Jieba dictionary + Tantivy mmap) | Open-source docs & blogs (e.g. `dioxuspress.dev`) |
| **`accounts` / `full` Tier** | `47.4 MiB` bin / `319 MB` image | **`549 ms`** | **`95.6 MiB`** (~`459 MiB` live RSS serving 2,200+ indexed docs & courses) | Fullstack community & paid course portals (e.g. `rustineverything.app`) |

---

## 2. First-Paint SSR, Conditional `ETag` (`304`), and Tantivy Search Latency

Measured on Apple Silicon loopback (`RUST_LOG=warn`, `--release`, `./scripts/bench_ssr.sh`, `N=200` serial samples and `c=16` concurrent load with `N=1,600` requests):

| Endpoint (`basic` Tier) | Status | P50 Latency | P99 Latency | `c=16` Throughput | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Home Page SSR (`GET /`) | `200 OK` | **`2.79 ms`** | `3.57 ms` | **`1,832 QPS`** | Full layout, navbar, feature grid, and SEO metadata |
| Docs Page SSR (`GET /docs/guide/quickstart`) | `200 OK` | **`4.04 ms`** | `5.05 ms` | **`1,236 QPS`** | Live Markdown parse + 3-level sidebar + breadcrumbs + VirtualDOM SSR |
| English Companion SSR (`GET /docs/guide/quickstart?lang=en`) | `200 OK` | **`4.15 ms`** | `6.67 ms` | **`1,219 QPS`** | `index.en.md` companion resolution + `hreflang` alternates |
| **Anonymous Conditional Revalidation (`If-None-Match`)** | **`304 Not Modified`** | **`0.096 ms`** | **`0.151 ms`** | **`13,636 QPS`** | **Skips Markdown & VirtualDOM SSR completely (41x faster P50, 10.3x QPS)** |
| Atom Feed (`GET /feed.xml`) | `200 OK` | **`0.182 ms`** | `0.310 ms` | **`11,785 QPS`** | Served from `DirListingCache` |
| **Tantivy Full-Text Search (`POST /api/search/query`)** | `200 OK` | **`0.314 ms`** | **`0.425 ms`** | **`9,215 QPS`** | BM25 ranking + bilingual Jieba tokenization + highlighted snippets |

:::tip Strict Privacy Isolation Against Cache Cross-Contamination
`conditional_cache_mw` enforces a fail-closed cookie allowlist: weak `ETag` caching and `304` short-circuiting apply **only** to anonymous `GET`/`HEAD` requests carrying no `Authorization` header and at most display-only cookies (`site_lang`, `site_theme`). Any request carrying `session` or `oauth_pkce` cookies — or targeting `/admin`, `/me`, `/forum`, `/topics`, `/course`, or `/api` — receives `Cache-Control: private, no-store` with `ETag` stripped.
:::

---

## 3. Zero-Downtime `CONTENT_DIR` Hot Reload (`SIGHUP`)

When external content is mounted at `CONTENT_DIR`, running `docker kill --signal=HUP dioxuspress-app` refreshes content in-place without restarting the container or dropping active TCP connections. Verified under 8-worker continuous load (`1,782` concurrent requests):
- **Error rate during reload**: **`0` non-200 responses**;
- **SSR prose propagation**: **`3.96 ms`**;
- **Background incremental Tantivy reindex**: **`272 ms`** (offloaded via `spawn_blocking` behind an async writer mutex while search readers remain 100% lock-free).

---

## 4. Frontend WebAssembly Bundle Size & CI Guardrail

Root `Cargo.toml` defines `[profile.wasm-release]` (`opt-level = "z"`, `lto = "fat"`, `codegen-units = 1`, `strip = "debuginfo"`), and GitHub Actions CI enforces upper bounds on every pull request via `./scripts/check_wasm_size.sh`:

| Client Build Tier | Raw `.wasm` (pre-`wasm-opt`) | `gzip -9` Transfer Size | `brotli -q 11` Transfer Size | CI Gzip Ceiling |
| :--- | :--- | :--- | :--- | :--- |
| **`basic` (Docs & Blog Hydration)** | `5.52 MiB` (`5,790,105 B`) | `1.54 MiB` | **`1.04 MiB`** | `1.86 MiB` (`1,950,000 B`) |
| **`full` (Community, Courses & Admin)** | `7.59 MiB` (`7,958,284 B`) | `2.07 MiB` | **`1.35 MiB`** (~`1.19 MiB` after `dx bundle` `wasm-opt`) | `2.43 MiB` (`2,550,000 B`) |

---

## 5. Reproduce Locally

```bash
# 1. Run full SSR, 304 conditional cache, RSS, and Tantivy benchmark across basic & accounts tiers
./scripts/bench_ssr.sh

# 2. Run the CI WebAssembly bundle size guardrail
./scripts/check_wasm_size.sh
```
