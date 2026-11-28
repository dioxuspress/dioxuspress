---
title: "Tailwind CSS v4 & dioxus-shadcn Component Styling Workflow"
sidebar_label: Tailwind CSS Workflow
sidebar_position: 5
description: Learn how DioxusPress scans RSX classes across workspace crates and compiles minified Tailwind CSS v4 stylesheets.
keywords: [Tailwind CSS v4, dioxus-shadcn, tw-sources, RSX styling]
---

# Tailwind CSS v4 Workflow

DioxusPress uses **Tailwind CSS v4** paired with our semantic component library [`dioxus-shadcn`](https://github.com/dioxuspress/dioxus-ui), converging colors onto semantic CSS variables (`--background`, `--foreground`, `--primary`, `--muted`, etc.).

## Workspace `@source` Auto-Discovery

Because RSX templates span multiple Rust crates (`crates/modules/*`, `crates/widgets`) plus `dioxus-shadcn` in Cargo's cache, `scripts/tw-sources.mjs` automatically generates `@source` directives before running the Tailwind compiler:

1. **Entrypoint**: `crates/app/tailwind-input.css`
2. **Production Build**:
   ```bash
   cd crates/app
   npm install
   npm run build
   ```
3. **Committed Artifact**: The minified stylesheet is written to `assets/tailwind.css` and tracked in Git — contributors and users building from Rust source **do not need Node.js installed** unless they add new utility classes.

## Watch Mode During UI Development

When editing Tailwind utility classes inside `.rs` files, run watch mode in a second terminal:

```bash
cd crates/app
npm run dev
```

Saving any `.rs` file rebuilds `assets/tailwind.css` incrementally in ~`100ms`, pairing cleanly with `dx serve` hot reload.
