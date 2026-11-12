---
title: "Engine Architecture & Custom Module Development: High Cohesion, Zero Sibling Coupling"
sidebar_label: Custom Modules
sidebar_position: 4
description: Add new domain crates to DioxusPress following strict dependency boundaries, automatic ModuleGate routing, search indexing, and sitemap integration.
keywords: [module development, ModuleSpec, Dioxus architecture, composition root, Rust workspace]
---

# Engine Architecture & Custom Module Development

DioxusPress avoids speculative plugin loaders and single-implementation trait factories. Every domain module is an ordinary **Rust crate** wired cleanly through explicit data contracts at the `crates/app` composition root.

## Layering & Dependency Rules

```mermaid
flowchart BT
  Core["crates/core: SiteConfig / i18n / Auth / ModuleEngine / DB Pool"]
  Widgets["crates/widgets: Markdown & MDX / SEO Meta / ReadingLayout / Browser APIs"]
  Modules["crates/modules/*: docs / blog / cases / forum / course / search ..."]
  App["crates/app: Route Enum / 3 Shell Layouts / Cross-Module Slot Composition"]

  Core --> Widgets
  Widgets --> Modules
  Modules --> App
```

- **Zero Sibling Coupling**: Crates under `crates/modules/*` **never depend on one another**. For example, `module-docs` knows nothing about `module-forum` (discussions) or `module-course` (annotations); instead it accepts a `footer: Element` slot assembled by `crates/app` according to active Cargo features.

## Adding a Domain Module in Four Steps

1. **Create the Crate**: Under `crates/modules/<your-mod>/`, separate `#[cfg(feature = "server")]` data loaders from isomorphic Dioxus `#[component]` views.
2. **Register a `ModuleSpec`**: Add the module ID, route prefixes, and optional `static_path` to `builtin_module_specs()` in `crates/core/src/engines/module.rs` — wiring `site.json::modules.<id>.enabled` and `/sitemap.xml` automatically.
3. **Mount Routes Behind `ModuleGate`**: Add variants to `Route` in `crates/app/src/routes/mod.rs` wrapped in `ModuleGate { id: "<your-mod>", ... }` so runtime-disabled modules cleanly return `404`.
4. **(Optional) Index into Tantivy**: Add a collector in `crates/modules/search/src/indexer.rs` to surface entries inside the `⌘K` search dialog.
