---
title: "Production Deployment, Full-Text Search & Module Architecture Overview"
sidebar_label: Operations & Architecture
sidebar_position: 4
description: Deploy zero-database containers or full PostgreSQL communities, configure multi-domain SNI TLS gateways, tune Tantivy search, and build custom modules.
keywords: [Docker deployment, zero database ops, Pingora reverse proxy, Tantivy search, module development]
---

# Operations, Search & Architecture

DioxusPress follows a "tiny footprint for small sites, resilient architecture for large portals" philosophy — whether hosting several zero-database docs sites on an entry-level $5 VPS or running a full PostgreSQL community portal with private content mounts.

## Choosing Your Deployment Topology

| Target Scenario | Recommended Build Tier | Typical Server Footprint | Guide |
| :--- | :--- | :--- | :--- |
| Open-source library docs, personal blog, engineering handbook | `basic` (`--no-default-features`) | **0.5 vCPU / 256 MB RAM** (Zero DB) | [Zero-Database Deployment](/docs/operations/zero-db-deploy) |
| Community portal with OAuth, forum, courses & `/admin` | Default full tier or `course` | **>= 1 vCPU / 1 GB RAM** (App + PostgreSQL) | [Fullstack Production Ops](/docs/operations/production-deploy) |

## Volume Chapters

- **[Zero-Database Deployment](/docs/operations/zero-db-deploy)**: Launch a multi-arch docs container behind HTTPS in under a minute.
- **[Fullstack Production Deployment & Backups](/docs/operations/production-deploy)**: Docker Compose orchestration, read-only content mounts, `/healthz` probes, and database backups.
- **[Tantivy Search & Technical SEO](/docs/operations/search-and-seo)**: Incremental `manifest.json` indexing, sitemaps, Schema.org JSON-LD, and Google Search Console setup.
- **[Engine Architecture & Custom Modules](/docs/operations/module-development)**: Add new cohesive domain crates with zero cross-module coupling.
- **[Environment Variables & Pingora Gateway Reference](/docs/operations/env-reference)**: Complete `.env`, OAuth2, payment webhook, and multi-domain SNI TLS dictionary.
- **[Verified Benchmarks & Size Guardrails](/docs/operations/benchmarks)**: Reproducible RSS memory, SSR / `304` throughput, Tantivy latency, and CI WASM bundle budgets.
