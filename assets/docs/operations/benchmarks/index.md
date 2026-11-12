---
title: 实测性能看板与体积守卫：内存占用、SSR 延迟、Tantivy 检索与 WASM 门禁
sidebar_label: 实测性能看板与守卫
sidebar_position: 6
description: DioxusPress 可复现的官方性能基准：涵盖零数据库与全栈档位内存占用、二进制/容器体积、SSR 与 ETag 304 吞吐、Tantivy 检索延迟及 CI WASM 包体积守卫。
keywords: [性能基准, 内存占用 RSS, SSR 延迟, ETag 304, Tantivy QPS, WASM 包体积]
---

# 实测性能看板与 CI 体积守卫

DioxusPress 遵循「**先测量、后优化、门禁可复现**」的工程准则：所有公开性能指标均由仓库内置脚本（`./scripts/bench_ssr.sh` 与 `./scripts/check_wasm_size.sh`）实测产生，任何人可在本地一键复现。

---

## 1. 二进制、容器镜像与内存占用（Footprint）

在单台云服务器部署多站点（Pingora SNI 网关 + 零数据库文档站 + 全栈门户）时，各组件实测资源开销如下：

| 组件 / 编译档位 | 产物 / 镜像体积 | 冷启动就绪耗时（`/healthz 200`） | 空闲内存（Idle RSS） | 适用形态 |
| :--- | :--- | :--- | :--- | :--- |
| **`dioxuspress-gateway`**（Pingora TLS SNI 网关） | 镜像 `143 MB` | `< 50 ms` | **`6.4 MiB`** | 多域名自动 TLS 卸载、HTTP/2 与 `301` 路由 |
| **`basic` 零数据库档**（`--no-default-features --features server`） | 二进制 `34.2 MiB` / 镜像 `269 MB` | **`547 ms`** | **`91.7 MiB`**（含内嵌结巴词典与 Tantivy 索引映射） | 开源库文档站、个人技术博客（如 `dioxuspress.dev`） |
| **`accounts` / `full` 全栈档** | 二进制 `47.4 MiB` / 镜像 `319 MB` | **`549 ms`** | **`95.6 MiB`**（2,200+ 篇长文档生产门户常驻约 `459 MiB`） | OAuth 社区、划线标注、付费课程门户（如 `rustineverything.app`） |

---

## 2. 首屏 SSR、匿名条件缓存（`ETag` / `304`）与全文检索延迟

测试环境：Apple Silicon 本地回环（`RUST_LOG=warn`，`--release` 优化构建，`./scripts/bench_ssr.sh`，单连接 `N=200` 采样与 `c=16` 并发压测 `N=1,600`）：

| 请求场景（`basic` 档位） | HTTP 状态 | P50 延迟 | P99 延迟 | `c=16` 并发吞吐 | 说明 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 首页完整 SSR（`GET /`） | `200 OK` | **`2.79 ms`** | `3.57 ms` | **`1,832 QPS`** | 含布局、导航、特性栅格与 SEO 头完整服务端渲染 |
| 中文长文档 SSR（`GET /docs/guide/quickstart`） | `200 OK` | **`4.04 ms`** | `5.05 ms` | **`1,236 QPS`** | 实时 Markdown 解析 + 三级侧栏 + 面包屑 + VirtualDOM 渲染 |
| 英文伴生文档 SSR（`GET /docs/guide/quickstart?lang=en`） | `200 OK` | **`4.15 ms`** | `6.67 ms` | **`1,219 QPS`** | 双语伴生 `index.en.md` 解析 + `hreflang` 注入 |
| **匿名条件复验（`If-None-Match` 命中）** | **`304 Not Modified`** | **`0.096 ms`** | **`0.151 ms`** | **`13,636 QPS`** | **跳过 Markdown 与 VirtualDOM SSR，延迟降低 41 倍，吞吐提升 10.3 倍** |
| Atom 订阅源（`GET /feed.xml`） | `200 OK` | **`0.182 ms`** | `0.310 ms` | **`11,785 QPS`** | `DirListingCache` 命中直出 XML |
| **Tantivy 中英全文检索（`POST /api/search/query`）** | `200 OK` | **`0.314 ms`** | **`0.425 ms`** | **`9,215 QPS`** | BM25 排序 + 结巴中文分词 + 高亮摘要提取 |

:::tip 登录态与私密路由零串缓存红线
`conditional_cache_mw` 采用严格白名单策略：只有无 `Authorization` 头且仅含展示偏好 Cookie（`site_lang`、`site_theme`）的纯匿名 `GET`/`HEAD` 请求才会启用弱 `ETag` 与 `304` 短路；任何携带 `session` / `oauth_pkce` Cookie 的请求，或访问 `/admin`、`/me`、`/forum`、`/topics`、`/course`、`/api` 的请求，一律强制返回 `Cache-Control: private, no-store` 并剥离 `ETag`。
:::

---

## 3. 零停机增量内容热刷新（`SIGHUP`）

挂载外部 `CONTENT_DIR` 时，发送 `docker kill --signal=HUP dioxuspress-app` 即可在不重启容器、不断开任何活跃 TCP 连接的前提下完成热更新。8 线程持续并发请求压测（共 `1,782` 次请求）实测表现：
- **并发错误率**：**`0` 个非 200 响应**（零闪断）；
- **SSR 页面新内容生效耗时**：**`3.96 ms`**；
- **Tantivy 搜索索引后台增量同步耗时**：**`272 ms`**（基于 `spawn_blocking` 与异步写锁串行化，读取端全程无锁）。

---

## 4. 前端 WebAssembly 包体积与 CI 门禁

根 `Cargo.toml` 针对前端水合产物配置了独立的 `[profile.wasm-release]`（`opt-level = "z"`、`lto = "fat"`、`codegen-units = 1`、`strip = "debuginfo"`），并在 GitHub Actions CI 中通过 `./scripts/check_wasm_size.sh` 实施硬性阈值守卫：

| 前端编译档位 | `.wasm` 原始体积（`wasm-opt` 前） | `gzip -9` 传输体积 | `brotli -q 11` 传输体积 | CI Gzip 门禁上限 |
| :--- | :--- | :--- | :--- | :--- |
| **`basic`（纯文档/博客水合端）** | `5.52 MiB`（`5,790,105 B`） | `1.54 MiB` | **`1.04 MiB`** | `1.86 MiB`（`1,950,000 B`） |
| **`full`（含社区/课程/管理台完整水合端）** | `7.59 MiB`（`7,958,284 B`） | `2.07 MiB` | **`1.35 MiB`**（经 `dx bundle` `wasm-opt` 后约 `1.19 MiB`） | `2.43 MiB`（`2,550,000 B`） |

---

## 5. 本地一键复现命令

```bash
# 1. 复现 basic 与 accounts 双档位 SSR、304 与 Tantivy 检索基准（可指定 BENCH_JSON 导出报告）
./scripts/bench_ssr.sh

# 2. 检查前端 WASM 产物包体积门禁
./scripts/check_wasm_size.sh
```
