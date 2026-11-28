---
title: "Embedded Tantivy Full-Text Search & Technical SEO: Sitemaps, JSON-LD & Search Console"
sidebar_label: Search & SEO
sidebar_position: 3
description: Inside DioxusPress's embedded Tantivy incremental indexer, crawler-ready SSR metadata, hreflang alternates, and Google Search Console setup.
keywords: [Tantivy search, technical SEO, sitemap.xml, JSON-LD, hreflang, Google Search Console]
---

# Embedded Tantivy Search & Technical SEO

DioxusPress builds instant full-text search and search-engine discoverability directly into the Rust binary — without external SaaS dependencies like Algolia.

## 1. Embedded Tantivy Full-Text Search (`⌘K`)

- **In-Process Pure Rust**: `module-search` embeds Tantivy directly inside the Axum server with bilingual CJK + English tokenization, boosted title scoring, kind filters (`doc` / `blog` / `case` / `topic`), and highlighted context snippets.
- **Incremental `manifest.json` Diffing**: File `mtime` stamps are compared against `/app/data/search-index/manifest.json` so only added or modified Markdown files are upserted and deleted files are pruned. Even across **2,500+ documentation pages**, incremental refreshes finish in milliseconds and cold full builds take ~6 seconds.
- **Versioned Schema Auto-Upgrade**: Bumping `IndexManifest::CURRENT_VERSION` automatically triggers a clean one-time re-upsert when indexing rules evolve.

## 2. End-to-End Technical SEO

1. **Complete SSR `<head>` on First Paint**: Every route emits accurate `<title>`, `<meta name="description">`, `<link rel="canonical">`, `hreflang` alternates (on paired bilingual docs and papers), and Open Graph / Twitter `summary_large_image` tags before any client JS runs.
2. **Schema.org JSON-LD**:
   - Homepage emits `WebSite`;
   - Documentation pages emit `TechArticle` (with `inLanguage` matching the resolved prose language and optional `SoftwareSourceCode` `about` metadata);
   - Blog posts emit `Article` with publication/revision timestamps and author attribution.
3. **Dynamic `/sitemap.xml` & Canonical Slash Redirects**:
   - `/sitemap.xml` reflects only compiled and enabled modules (plus custom internal `site.json::navigation` routes and `?lang=en` URLs for paired bilingual docs), served with a 1-hour `public` cache header;
   - `trailing_slash_mw` permanently redirects `/docs/guide/` (`308`) to `/docs/guide` to prevent duplicate-URL dilution.

## 3. Google Search Console Checklist

After pointing your domain to DioxusPress:
1. Verify domain ownership in Google Search Console via DNS `TXT` record;
2. Submit `https://<your-domain>/sitemap.xml` under **Sitemaps**;
3. Pair concise `sidebar_label` strings with descriptive long-tail `title` frontmatter to capture high-intent developer queries.
