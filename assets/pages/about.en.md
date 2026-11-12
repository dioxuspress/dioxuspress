---
title: About DioxusPress
description: Open-source fullstack Rust documentation, content, and community engine built on Dioxus 0.7 and Axum.
---

# About DioxusPress

**DioxusPress** is an open-source, pure-Rust fullstack knowledge publishing and community engine built on [Dioxus 0.7](https://dioxuslabs.com) (server-side rendering SSR + client-side WebAssembly hydration) and the [`dioxus-shadcn`](https://github.com/dioxuspress/dioxus-ui) component library.

## Design Philosophy

1. **One Codebase, Elastic Tiers**: Compile with `--no-default-features` (`basic` tier) to run a fast, zero-database Docusaurus-style documentation site with embedded Tantivy full-text search; enable default features to unlock OAuth2 authentication, community forums, comments, inline text annotations, LLM content moderation, and paid courses.
2. **Files as Content, Git as Workflow**: All documentation, blog posts, courses, and showcase entries live as standard Markdown / MDX in version control, with read-only external repository mounting (`CONTENT_DIR`) and zero-downtime `SIGHUP` hot reload.
3. **Hardened Security & SEO by Default**: Strict `'unsafe-eval'`-free CSP, first-paint SSR Open Graph and JSON-LD structured data, volume-qualified chapter titles, conditional `ETag` / `304` caching, and dynamic bilingual sitemaps.

## Production Showcases

- **Official Website**: [dioxuspress.dev](https://dioxuspress.dev)
- **[Rust in Everything (`rustineverything.app`)](https://rustineverything.app)**: Serves 38 volumes and 2,100+ deep-dive Rust ecosystem chapters, structured AI engineering courses, paper digests, and community discussions — powered end-to-end by DioxusPress.
- **Source Repository**: [github.com/dioxuspress/dioxuspress](https://github.com/dioxuspress/dioxuspress) (MIT License).
