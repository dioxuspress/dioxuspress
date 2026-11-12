---
title: "Official Starter Templates Guide: Comparing docs, blog, community & portal Presets"
sidebar_label: Starter Templates
sidebar_position: 6
description: Compare DioxusPress's four official starter templates (open-source docs, minimal blog, community hub, and fullstack portal) and upgrade seamlessly without rewriting Markdown.
keywords: [Starter Templates, dioxuspress init, docs template, blog template, community template]
---

# Official Starter Templates Guide

DioxusPress maintains four production-ready starter templates under [`templates/`](https://github.com/dioxuspress/dioxuspress/tree/main/templates), embedded at compile time into `dioxuspress init` as a **Single Source of Truth**. You can also preview all four templates in light and dark mode on the interactive **[Templates Showcase (`/templates`)](/templates)**.

## Preset Comparison Matrix

| Preset | CLI Flag | Shell Layout | Default Theme | Image & Dependencies | Included Out of the Box |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Open-Source Docs** | `--template docs` | `docs` | `emerald` | `dioxuspress-basic` (Zero DB) | Release banner, hero feature grid, `guide` + `api` 3-level sidebar, GitHub Actions CI lint workflow |
| **Minimal Tech Blog** | `--template blog` | `minimal` | `coffee` | `dioxuspress-basic` (Zero DB) | Warm parchment palette, server-rendered MathML, code tabs, `authors.yml`, prominent `/feed.xml` |
| **Tech Community** | `--template community` | `classic` | `ocean` | `dioxuspress-app` + PostgreSQL 16 | GitHub/Google/Discord OAuth2 PKCE setup, `/topics` forum, inline annotations, URL blocklist |
| **Knowledge Portal** | `--template portal` | `classic` | `corporate` | `dioxuspress-app` + PostgreSQL 16 | Free-preview & paid courses (`courses/`), showcase (`cases/`), embedded `<PodcastCard />`, optional LLM moderation |

## Scaffold in One Command

```bash
# Example: create a zero-database documentation site (site_name is derived from the folder name)
dioxuspress init my-project-docs --template docs
cd my-project-docs
dioxuspress check
docker compose up -d
```

## Upgrading from Zero-DB to a Fullstack Community

All four presets share the **exact same content directory contract** (`site.json` + `docs/` + `posts/` + `images/`). If you launch with `docs` or `blog` today and later want comments, discussions, or paid courses, **zero Markdown files need to be migrated**:

1. Switch the image in `docker-compose.yml` from `ghcr.io/dioxuspress/dioxuspress-basic:latest` to `ghcr.io/dioxuspress/dioxuspress-app:latest` and add `postgres:16-alpine` (see `templates/community/docker-compose.yml`);
2. Create `.env` with `DATABASE_URL`, `JWT_SECRET`, and your OAuth client credentials;
3. Set `auth.enabled` and `modules.forum.enabled` (or `modules.course.enabled`) to `true` in `site.json` and restart — SeaORM runs schema migrations automatically.
