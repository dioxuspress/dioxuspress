# dioxuspress-cli

Lightweight official CLI for [DioxusPress](https://dioxuspress.dev) — scaffold zero-database documentation sites or fullstack Rust communities (`dioxuspress init`) and statically verify `site.json`, YAML frontmatter, MDX tags, fenced code blocks, and internal links (`dioxuspress check`).

## Installation

```bash
cargo install dioxuspress-cli
```

## Quick Start

```bash
# 1. Scaffold a new site (presets: docs | blog | community | portal)
dioxuspress init my-docs --template docs
cd my-docs

# 2. Validate site.json, frontmatter, fenced blocks, and internal links offline
dioxuspress check

# 3. Launch with the prebuilt multi-arch Docker image
docker compose up -d
# Open http://localhost:8080
```

## Commands

- `dioxuspress init <dir> [--template docs|blog|community|portal] [--force]` — Generate a ready-to-run DioxusPress content repository with IDE JSON Schema validation and `docker-compose.yml`.
- `dioxuspress check [dir]` — Millisecond offline static checker for `site.json` keys/themes, Markdown/MDX frontmatter, unclosed code fences/admonitions, and broken internal links (ideal for GitHub Actions CI).

## Links

- Official Docs: <https://dioxuspress.dev>
- Starter Templates Preview: <https://dioxuspress.dev/templates>
- UI Component Library: <https://ui.dioxuspress.dev>
- Source Code: <https://github.com/dioxuspress/dioxuspress>
