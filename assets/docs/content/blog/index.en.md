---
title: "Blog Module Guide: Post Directories, Multi-Author authors.yml & Atom/RSS Feeds"
sidebar_label: Blog & Authors
sidebar_position: 3
description: Organize blog posts under assets/posts with co-located images, multi-author profiles in authors.yml, tag filtering, and automatic /feed.xml generation.
keywords: [blog module, authors.yml, Atom feed, tag filtering, MDX blog]
---

# Blog & Multi-Author System

Each blog post lives in its own directory under `assets/posts/<slug>/`, keeping `index.mdx` (or `index.md`) alongside co-located diagrams and cover images.

## Directory Layout

```text
assets/posts/
├── authors.yml                 # Global author directory (avatars, titles, profile URLs)
├── welcome/
│   ├── index.mdx               # Served at /blog/welcome
│   └── cover.webp              # Referenced via relative path ![Cover](cover.webp)
└── release-0-7/
    └── index.md
```

## `authors.yml` Format

```yaml
yuxuetr:
  name: yuxuetr
  title: DioxusPress Creator
  url: https://github.com/yuxuetr
  image_url: /images/logo-mobile.png
```

Reference one or multiple author keys in post Frontmatter:

```yaml
---
title: "Introducing DioxusPress: Rebuilding Content & Community Platforms in Pure Rust"
description: "Why we moved from static site generators to isomorphic Dioxus 0.7 SSR + WASM."
date: "2026-11-12"
authors: [yuxuetr]
tags: [Rust, Dioxus, Architecture]
---
```

## Built-In Aggregation Features

- **Instant Tag Filtering**: `/blog` aggregates all post tags ordered by frequency with pagination;
- **Syndication Feed**: `/feed.xml` serves a standards-compliant Atom feed (automatically disabled if `modules.blog.enabled` is `false`);
- **Community Integration**: When compiled with `accounts`, every post automatically mounts `CommentBox` and a linked forum discussion panel at the footer.
