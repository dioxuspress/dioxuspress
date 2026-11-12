---
title: "Fullstack Production Deployment & Operations: Docker Compose, Content Mounts & Backups"
sidebar_label: Fullstack Production Ops
sidebar_position: 2
description: Operate PostgreSQL-backed DioxusPress portals with decoupled private content mounts, /healthz probes, and backup workflows.
keywords: [production deployment, docker-compose, CONTENT_DIR mount, PostgreSQL backup, healthz]
---

# Fullstack Production Deployment & Operations

For portals running OAuth sign-in, community forums, course entitlements, and payment webhooks, orchestrate the application and PostgreSQL 16 with `docker-compose.yml` + `docker-compose.prod.yml`.

## 1. Standard Startup Flow

```bash
# 1. Prepare production environment variables
cp .env.example .env
# Required: BASE_URL, JWT_SECRET (>= 32 chars), POSTGRES_PASSWORD

# 2. Launch application + PostgreSQL 16 (SeaORM migrations run automatically)
docker compose up -d

# 3. Inspect logs and verify health probe
docker compose logs -f app
curl -fsS http://127.0.0.1:8080/healthz
```

## 2. Decoupling Open-Source Engine & Private Content (Best Practice)

To pull upstream DioxusPress releases cleanly with zero merge conflicts, keep your site's real documentation, paid courses, blog posts, and brand assets in a separate Git repository:

```text
/srv/my-site-content/
├── site.json          # Site-specific branding, layout, theme, and module configuration
├── images/            # Custom logos, favicons, and Open Graph cards (served first at /images/*)
├── docs/              # Documentation volumes (supports optional index.en.md companions)
├── posts/             # Blog posts and authors.yml
└── courses/           # Free-preview and paid courses
```

Mount `/srv/my-site-content` read-only at `/app/content` (`CONTENT_DIR=/app/content`):
- **Brand image overlay**: Requests to `/images/*` check `$CONTENT_DIR/images/` first and gracefully fall back to built-in `assets/images/`;
- **Fast content releases**: Publishing new docs or course lessons only requires syncing the content repo on the host — no Rust recompilation needed.

## 3. Persistent Volumes & Backup Checklist

Back up three stateful paths in production:
1. **PostgreSQL data volume**: Run scheduled `pg_dump -Fc` snapshots to off-site object storage;
2. **User uploads (`/app/assets/uploads`)**: Stores images uploaded in forum posts and comments;
3. **Search index (`/app/data/search-index`)**: Rebuilds automatically from Markdown and PostgreSQL if empty, while keeping a persistent volume makes incremental checks millisecond-fast.
