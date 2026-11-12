---
title: "Git-Driven Content Authoring Engine: Docs Trees, Rich MDX, Blogs & Structured Courses"
sidebar_label: Content Authoring
sidebar_position: 2
description: Master DioxusPress's filesystem-first Markdown and MDX content pipeline across docs, blogs, courses, showcases, and podcasts.
keywords: [Git authoring, MDX rendering, documentation tree, blog engine, course platform]
---

# Content Authoring Overview

In DioxusPress, every structured piece of knowledge lives as **Git-tracked plain-text Markdown / MDX** as a Single Source of Truth — stored in `assets/` by default or mounted read-only via `CONTENT_DIR=/app/content` in production so **open-source engine code** and **private content repositories** remain completely decoupled.

## Content Formats at a Glance

| Module ID | Default Directory | Route Prefix | Highlights |
| :--- | :--- | :--- | :--- |
| **`docs`** | `assets/docs/` | `/docs/*` | 3-level sidebar tree, `sidebar_position` ordering, right-rail TOC, breadcrumbs, bilingual `index.en.md` companions, volume-qualified SEO titles |
| **`blog`** | `assets/posts/` | `/blog`, `/blog/:slug` | Rich MDX components, multi-author profiles (`authors.yml`), tag filtering, automatic `/feed.xml` |
| **`course`** | `assets/courses/` | `/course/*` | Course → Chapter → Lesson outline, free preview flag, server-enforced single-course & Pro entitlements |
| **`cases`** | `assets/cases/` | `/case`, `/case/:slug` | Curated project showcase, tech-stack tag filter, direct repository & live site links |
| **`podcast`** | `assets/podcasts/` | `/podcast` | Built-in web audio player, episode show notes, guest metadata, embeddable `<PodcastCard />` |

## Volume Chapters

- **[Docs Module Specification](/docs/content/docs-module)**: Directory layout, bilingual companion files, Frontmatter fields, and long-tail SEO title qualification.
- **[MDX Syntax & Components](/docs/content/mdx)**: Code tabs, server-side MathML equations, Mermaid diagrams, 7 admonition types, and media embeds.
- **[Blog & Multi-Author System](/docs/content/blog)**: Publishing posts, co-located cover images, and `authors.yml`.
- **[Structured Courses Module](/docs/content/courses)**: Authoring free preview lessons and paywalled chapters.
- **[Showcase & Podcast Modules](/docs/content/cases-and-podcast)**: Building a "Built with..." gallery and audio podcast channel.
- **[Frontmatter Reference](/docs/content/frontmatter-reference)**: Complete YAML field dictionary across all five content modules.
