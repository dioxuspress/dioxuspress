---
title: "Complete Frontmatter & YAML Metadata Reference: Docs, Blog, Courses, Showcase & Podcast"
sidebar_label: Frontmatter Reference
sidebar_position: 6
description: Authoritative YAML Frontmatter and metadata dictionary for all five file-driven DioxusPress modules (docs, blog, course, cases, podcast).
keywords: [Frontmatter, DocMeta, authors.yml, course.yaml, case.yaml, episode.yaml]
---

# Frontmatter & Metadata Reference

All file-driven modules in DioxusPress use standard YAML frontmatter (`--- ... ---`) or directory-level `.yaml` manifests. Use this page as a quick lookup table while authoring content.

---

## 1. Docs Module (`assets/docs/**/index.{md,mdx}` & `index.en.{md,mdx}`)

Maps to `module_docs::server::DocMeta`:

```yaml
---
title: "Tokio Scheduler Internals: Work-Stealing Queues & Cooperative Budgeting"
sidebar_label: Scheduler Internals
sidebar_position: 4
description: Deep dive into Tokio's multi-threaded run queues, global injection queue, and task coop budget.
keywords: [Tokio, Work-Stealing, async scheduler, Rust concurrency]
image: /images/og-tokio.jpg
sort_children: asc
source:
  crate: tokio
  version: "1.49.0"
---
```

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `title` | `string` | Directory slug | Full page headline (`<title>`, `og:title`, and search index; subpages without volume name are auto-qualified as `Title · Volume`) |
| `sidebar_label` | `string` | Falls back to `title` | Compact label used in the left sidebar tree and previous/next navigation cards |
| `sidebar_position` | `integer` | Sorted after numbered siblings | Sort weight within the current folder (smaller numbers appear first; ties break alphabetically by slug) |
| `description` | `string` | First body paragraph | SEO `<meta name="description">` and volume preview summary |
| `keywords` | `string[]` | `[]` | Injected into `<meta name="keywords">` and JSON-LD `TechArticle` |
| `image` | `string` | `site.json::og_image` | Per-page Open Graph / Twitter image override |
| `sort_children` | `"asc" \| "desc"` | `"asc"` | Set on a parent folder's `index.md` to reverse direct child ordering (`"desc"` pins newest issues first) |
| `source` | `{ crate, version }` | `null` | Renders a `docs.rs` version pill above the article and emits `SoftwareSourceCode` schema metadata |

---

## 2. Blog Module (`assets/posts/<slug>/index.{md,mdx}` & `authors.yml`)

### Post Frontmatter

```yaml
---
title: "Why We Built a Fullstack Knowledge Engine on Dioxus 0.7"
description: "Comparing static site generators against isomorphic Rust SSR + WASM hydration."
date: "2026-11-12"
updated: "2026-11-15"
authors: [yuxuetr]
tags: [Rust, Dioxus, Fullstack]
image: cover.webp
---
```

| Field | Type | Requirement | Description |
| :--- | :--- | :--- | :--- |
| `title` | `string` | **Required** | Post headline |
| `description` | `string` | Recommended | Post card excerpt, Atom `<summary>`, and meta description |
| `date` | `string` | **Required** | Publication date (`YYYY-MM-DD`) used for reverse-chronological sorting |
| `updated` | `string` | Optional | Last revision date (`YYYY-MM-DD`); takes precedence for `/sitemap.xml` `<lastmod>` |
| `authors` | `string \| string[]` | Recommended | Author key(s) defined in `assets/posts/authors.yml` |
| `tags` | `string[]` | Recommended | Topic tags aggregated on `/blog` |
| `image` | `string` | Optional | Social card image (co-located filename like `cover.webp` or `/images/...`) |

### Global Authors Registry (`assets/posts/authors.yml`)

```yaml
yuxuetr:
  name: yuxuetr
  title: Core Maintainer
  url: https://github.com/yuxuetr
  image_url: /images/logo-mobile.png
```

---

## 3. Courses Module (`assets/courses/<course_slug>/`)

Uses a three-tier manifest hierarchy: **`course.yaml` → `_chapter.yaml` → lesson `index.{md,mdx}`**.

### 3.1 `course.yaml`

```yaml
title: Async Rust Runtime in Action
description: Build a work-stealing Future executor from scratch to production tuning.
cover: /images/courses/async-rust.webp
order: 1
level: Advanced
tags: [rust, async, tokio]
access_tier: paid
price: 9900
currency: CNY
price_usd: 1499
```

| Field | Type | Description |
| :--- | :--- | :--- |
| `title` / `description` | `string` | Course title and public summary |
| `cover` | `string` | Course cover image path |
| `order` | `integer` | Display order on `/course` (ascending) |
| `level` | `string` | Difficulty badge (e.g. `"Beginner"`, `"Intermediate"`, `"Advanced"`) |
| `tags` | `string[]` | Topic tags |
| `access_tier` | `"free" \| "paid" \| "pro"` | `free`: open to all; `paid`: single-course purchase or Pro membership; `pro`: Pro members only |
| `price` / `currency` | `integer` / `string` | Price in minor units (e.g. `9900` = `¥99.00`), default `"CNY"` |
| `price_usd` | `integer` | Optional Stripe USD price in cents (e.g. `1499` = `$14.99`) |

### 3.2 `<chapter_dir>/_chapter.yaml`

```yaml
title: "Chapter 1: Future State Machines & Waker Contracts"
description: How rustc lowers async fn into zero-allocation state machines.
order: 1
```

### 3.3 Lesson `<chapter_dir>/<lesson_dir>/index.{md,mdx}`

```yaml
---
title: "1.1 Writing a Minimal Pin & Poll Executor"
description: Complete a working async task wakeup loop in 50 lines of Rust.
sidebar_position: 1
duration: "12:30"
preview: true
video_url: ""
audio_url: ""
---
```

- **`preview: true`**: Marks the lesson as a **free public preview** even inside a `paid` or `pro` course; lessons with `preview: false` (or omitted) are strictly protected on the server.

---

## 4. Showcase Module (`assets/cases/<slug>/case.yaml` + `README.md`)

```yaml
name: Dioxus Fullstack Template
description: Isomorphic Dioxus 0.7 starter with SSR, Server Functions, and Tailwind v4.
category: fullstack
tags: [dioxus, fullstack, wasm]
repo: https://github.com/dioxuspress/dioxuspress
website: https://dioxuspress.dev
author: DioxusPress Team
author_url: https://github.com/dioxuspress
language: Rust
stars: 1200
favorite: true
date_added: 2026-11-12
```

---

## 5. Podcast Module (`assets/podcasts/<slug>/episode.yaml`)

```yaml
id: 1
title: Production Lessons from Running Fullstack Rust on the Web
description: Compile times, WASM binary budgets, and single-node concurrency numbers.
duration: "24:15"
date: 2026-11-12
audio_url: /audio/ep01-fullstack-rust.m4a
guest: yuxuetr
tags: [Rust, WebAssembly, Architecture]
```

- Embed any episode inside `.mdx` pages with `<PodcastCard id="1" />`.
