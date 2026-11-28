---
title: "Zero-Database Deployment Guide: Run Multi-Arch DioxusPress Docs & Blog Containers in One Command"
sidebar_label: Zero-DB Deployment
sidebar_position: 1
description: Deploy documentation and blog sites in seconds with the prebuilt multi-arch ghcr.io/dioxuspress/dioxuspress-basic image — no PostgreSQL or Rust compiler required.
keywords: [zero database deploy, dioxuspress-basic, multi-arch Docker, single container, Docusaurus alternative]
---

# Zero-Database Deployment Guide

For documentation centers, project landing pages, and technical blogs, **you need neither a Rust compiler nor PostgreSQL or secret tokens on your server**.

The official `ghcr.io/dioxuspress/dioxuspress-basic` image is published for both `linux/amd64` (x86_64 servers) and `linux/arm64` (Apple Silicon & ARM cloud instances).

---

## 1. Run the Prebuilt Image Directly (Recommended)

Mount any directory containing `site.json`, `docs/`, or `posts/` (scaffolded via `dioxuspress init my-site`) read-only at `/app/content:ro`:

```bash
# Local preview (BASE_URL defaults to http://localhost:8080 and /app/content is auto-detected)
docker run --rm -p 8080:8080 \
  -v ./my-site:/app/content:ro \
  ghcr.io/dioxuspress/dioxuspress-basic:latest
```

In production, set your public `BASE_URL` (used for `<link rel="canonical">`, `og:url`, and `/sitemap.xml`) and attach a named volume so Tantivy's incremental index persists across updates:

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

:::tip Zero-Compile Branding & Theme Overrides
Whenever `site.json` and `images/` exist inside `/app/content`, the container overrides the default site name, layout (`docs` / `classic` / `minimal`), all 10 dual-mode themes, topbar navigation, hero features, and favicons consistently across both SSR and WASM hydration.
:::

---

## 2. Building a Custom `basic` Image from Source (Optional)

If you forked the workspace to register custom Rust MDX components, build locally with the root `Dockerfile`:

```bash
docker build \
  --build-arg APP_FEATURES="--no-default-features" \
  --build-arg INCLUDE_PAPERS_INGEST=0 \
  -t dioxuspress-basic:local .
```

The runtime stage runs on `debian:trixie-slim` as an unprivileged `app` user, bundling the SSR server binary alongside pre-compressed WASM assets.

---

## 3. Health Checks & HTTPS Reverse Proxy

```bash
# Readiness check ("db": "disabled" is expected in basic tier)
curl -fsS http://127.0.0.1:8081/healthz

# Verify dynamic sitemap and Atom feed
curl -fsS http://127.0.0.1:8081/sitemap.xml | head -n 10
```

Point your TLS reverse proxy (`ghcr.io/dioxuspress/dioxuspress-gateway` Pingora SNI gateway, Caddy, or Nginx) to `127.0.0.1:8081` to go live over HTTPS.
