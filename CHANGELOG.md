# Changelog

All notable changes to **DioxusPress** are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
commits follow [Conventional Commits](https://www.conventionalcommits.org/).

## [0.3.0] - 2026-11-28

Version **0.3.0** establishes **DioxusPress** as a pure, composable open-source
full-stack publishing and documentation framework built on Dioxus 0.7 (SSR +
WASM hydration), Axum, SeaORM/PostgreSQL, embedded Tantivy search, Pingora SNI
gateway, and `dioxus-shadcn 0.7.0`.

### Added
- **Composable Library & Server Extension Hooks (`dioxuspress_app`)**:
  - Exposed `dioxuspress_app` (`crates/app`) as a reusable library crate with
    `AppShell`, `run_server(App, ServerExtensions)`, `run_client(App)`, and
    `register_core_components_and_sources(...)` so downstream site repositories
    can compose custom vertical modules without forking the framework.
  - Added `ServerExtensions` (`extra_migrations`, `SeoExtensions`,
    `extra_hot_reload`, `extra_etag_stamp`, `extra_router`),
    `register_extra_module_specs`, `conditional_cache::register_extra_route_rules`,
    and `SiteNavbar { extra_nav_links, ecosystems, footer_boards }` hooks.
- **Docusaurus-Style Docs Template & Official Site (`dioxuspress.dev`)**:
  - `active_layout = "docs"` / `"classic"` / `"minimal"` shells with
    announcement banner, hero with copyable install command, feature grid,
    multi-column footer, bilingual (`zh` / `en`) official documentation, and
    JSON Schema (`assets/schemas/site.schema.json`).
  - Starter templates (`docs`, `blog`, `community`, `portal`) and
    `dioxuspress-cli` (`0.3.0`) for zero-compile content scaffolding (`init`)
    and offline static checks (`check`).
- **10 WCAG-AA Dual-Mode Themes & `dioxus-shadcn 0.7.0`**:
  - Paired light (`:root`) and dark (`.dark`) presets (`ocean`, `emerald`,
    `nord`, `corporate`, `winter`, `dracula`, `sunset`, `coffee`, `catppuccin`,
    `silk`), `available_themes` allowlist, and zero-`'unsafe-eval'` strict CSP
    compatibility.
- **MDX & Publishing Pipeline**:
  - Interactive zoomable dual-theme **SVG Diagram Viewer**, Prism syntax
    highlighting (40 languages) with `:::code-tabs`, Mermaid dark-mode palettes,
    `<VocabCard />`, KaTeX/Typst math rendering, and `CONTENT_DIR` live overlay.
- **Four Cargo Feature Tiers & Multi-Arch Images**:
  - `basic` (zero-DB), `accounts`, `course`, and `full` feature tiers.
  - Pingora multi-domain SNI TLS gateway (`crates/gateway`) and automated
    `linux/amd64` + `linux/arm64` GHCR builds for `dioxuspress-basic`,
    `dioxuspress-app`, and `dioxuspress-gateway`.

### Changed
- Purified the public framework workspace to 7 core generic content & community
  modules (`docs`, `blog`, `podcast`, `cases`, `forum`, `course`, `about`) plus
  generic infrastructure (`search`, `comments`, `uploads`, `moderation`,
  `admin`, `payment`, `board`, `llm`).
- Made `board` optional in `crates/app` (enabled only under `server`), reducing
  WASM client bundle size across all four tiers.
- Upgraded `dioxus-shadcn` to **`0.7.0`** and aligned all workspace crates and
  `dioxuspress-cli` to **`0.3.0`**.
