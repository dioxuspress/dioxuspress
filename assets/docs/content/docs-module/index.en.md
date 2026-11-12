---
title: "Docs Module Specification: 3-Level Trees, Bilingual Companions, Breadcrumbs & SEO Titles"
sidebar_label: Docs Module Spec
sidebar_position: 1
description: Organize multi-volume documentation under assets/docs with sidebar_position, sort_children, index.en.md bilingual files, and automatic SEO title qualification.
keywords: [documentation tree, sidebar_position, sidebar_label, index.en.md, breadcrumbs, docs SEO]
---

# Docs Module Specification

`module-docs` delivers a Docusaurus-class three-level documentation system engineered for multi-volume technical manuals and clean search-engine indexing.

## Directory Convention

```text
assets/docs/
├── guide/                      # Top-level volume: /docs/guide
│   ├── index.md                # Default / Chinese volume overview (optional; group-only if omitted)
│   ├── index.en.md             # Optional English companion (/docs/guide?lang=en)
│   ├── quickstart/
│   │   ├── index.md            # Chapter: /docs/guide/quickstart
│   │   └── index.en.md         # English companion: /docs/guide/quickstart?lang=en
│   └── advanced/
│       └── ssr/
│           └── index.md        # Section (level 3): /docs/guide/advanced/ssr
└── changelog/
    ├── index.md                # Set sort_children: desc to pin latest releases on top
    ├── v0-7-0/index.md
    └── v0-7-1/index.md
```

- Directories starting with `_` or `.` (such as `_drafts`) are skipped by both sidebar scanning and Tantivy indexing.
- Both `index.md` and `index.mdx` are supported, alongside optional `index.en.md` / `index.en.mdx` companions: when a visitor selects English (`?lang=en` or `site_lang=en` cookie), `module-docs` serves the `.en.{md,mdx}` file with localized sidebar, breadcrumbs, and `hreflang` alternates — smoothly falling back to `index.{md,mdx}` whenever a page has not been translated yet.

## Frontmatter Metadata

```yaml
---
title: "Axum Common Pitfalls: Handler Trait Bounds, Route Collisions & Layer Ordering"
sidebar_label: Common Pitfalls
sidebar_position: 10
description: Troubleshoot frequent compile-time Handler errors and body-extractor ordering mistakes in Axum.
keywords: [axum, FromRequest, Handler trait, troubleshooting]
sort_children: asc
source:
  crate: axum
  version: "0.8.4"
---
```

### Practical Design Details

1. **Decoupled `title` vs. `sidebar_label`**: Left navigation trees and previous/next cards display the concise `sidebar_label` (`Common Pitfalls`), while `<title>`, `og:title`, and JSON-LD `TechArticle.headline` use the descriptive `title`.
2. **Automatic Volume Qualification**: When a subpage sets a short `title` (e.g. `Routing & Extractors`), the server automatically qualifies SEO tags and search index titles as `Routing & Extractors · Axum`, eliminating duplicate titles across multi-crate handbooks.
3. **`source` Crate Badge**: Setting `source.crate` and `source.version` renders a pill linking to `https://docs.rs/<crate>/<version>` at the top of the article and attaches a Schema.org `SoftwareSourceCode` `about` node.
4. **Dead-Link-Free Breadcrumbs & Edit Links**: Group directories without an `index.md` render as plain text in breadcrumbs (never producing 404 links), and setting `site.json::docs_edit_url_base` appends a direct GitHub edit link at the bottom of every doc page.
