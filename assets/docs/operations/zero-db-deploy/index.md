---
title: 零数据库轻量部署指南：一行命令运行多架构 DioxusPress 文档与博客镜像
sidebar_label: 零数据库轻量部署
sidebar_position: 1
description: 使用官方预构建双架构镜像 ghcr.io/dioxuspress/dioxuspress-basic 挂载 Markdown 目录秒级启动文档与博客站，无需数据库或 Rust 编译环境。
keywords: [零数据库部署, dioxuspress-basic, 多架构镜像, 单容器部署, Docusaurus 替代部署]
---

# 零数据库轻量部署指南

如果你使用 DioxusPress 搭建类似 Docusaurus 的技术文档站、产品官网或纯静态内容博客，**既不需要安装 Rust 编译工具链，也不需要部署 PostgreSQL 或配置任何密钥环境变量**。

官方发布的 `ghcr.io/dioxuspress/dioxuspress-basic` 原生支持 `linux/amd64`（x86_64 服务器）与 `linux/arm64`（Apple Silicon Mac / ARM 云主机）双架构。

---

## 1. 零编译直接启动官方镜像（推荐）

只需准备好包含 `site.json`、`docs/` 或 `posts/` 的本地目录（可通过 `dioxuspress init my-site` 一键生成），挂载至容器内的 `/app/content:ro` 即可在 2 秒内启动：

```bash
# 本地预览（未设置 BASE_URL 时自动回退至 http://localhost:8080，挂载 /app/content 自动识别）
docker run --rm -p 8080:8080 \
  -v ./my-site:/app/content:ro \
  ghcr.io/dioxuspress/dioxuspress-basic:latest
```

在生产服务器上后台常驻运行时，显式传入公开域名 `BASE_URL`（用于生成准确的 `<link rel="canonical">`、`og:url` 与 `/sitemap.xml`）并持久化全文索引卷：

```bash
docker run -d \
  --name dioxuspress-docs \
  --restart unless-stopped \
  -p 127.0.0.1:8081:8080 \
  -e BASE_URL=https://docs.example.com \
  -e SEARCH_INDEX_DIR=/app/data/search-index \
  -v /srv/my-site:/app/content:ro \
  -v dioxuspress-search:/app/data/search-index \
  ghcr.io/dioxuspress/dioxuspress-basic:latest
```

:::tip 零编译品牌与配置热覆盖
当 `/app/content` 下存在 `site.json` 与 `images/` 时，容器会自动覆盖默认站名、布局（`docs` / `classic` / `minimal`）、10 套内置主题、导航、Hero 特性卡与站点图标，并在 SSR 与 WASM 水合双端保持严格一致。
:::

---

## 2. 从源码自行构建 `basic` 档位镜像（可选）

如果你fork了引擎仓库并注册了自定义 MDX 组件，可使用仓库根目录的 `Dockerfile` 本地构建：

```bash
docker build \
  --build-arg APP_FEATURES="--no-default-features" \
  --build-arg INCLUDE_PAPERS_INGEST=0 \
  -t dioxuspress-basic:local .
```

该镜像基于 `debian:trixie-slim` 并以非特权用户 `app` 运行，内部包含 SSR 服务端二进制与 Brotli/Gzip 预优化的 WASM 前端产物。

---

## 3. 验证运行状态与接入 HTTPS 网关

```bash
# 健康检查（basic 档位下 db 字段返回 "disabled" 属于预期正常状态）
curl -fsS http://127.0.0.1:8081/healthz

# 验证自动生成的站点地图与 Atom 订阅源
curl -fsS http://127.0.0.1:8081/sitemap.xml | head -n 10
```

最后在宿主机的反向代理（内置 Pingora 网关 `ghcr.io/dioxuspress/dioxuspress-gateway` / Caddy / Nginx）中将你的域名转发至 `127.0.0.1:8081` 即可正式上线。
