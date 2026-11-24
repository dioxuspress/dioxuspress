# Changelog

All notable changes to **DioxusPress** are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
commits follow [Conventional Commits](https://www.conventionalcommits.org/).

## [Unreleased]

## [0.2.0] - 2026-11-24

Version **0.2.0** marks the first major public milestone for the `dioxuspress`
organization: a complete zero-database Docusaurus-style documentation engine,
10 WCAG-AA dual-mode themes, multi-arch GHCR images, interactive AI/WASM
visualizers, and `dioxus-shadcn 0.6.8` (100 components).

### Added
- **Docusaurus-Style Docs Template & Official Site (`dioxuspress.dev`)**:
  - `active_layout = "docs"` shell with announcement banner, centered hero with
    copyable install command, feature grid, multi-column footer, and bilingual
    (`zh` / `en`) official documentation under `assets/docs/`.
  - JSON Schema (`assets/schemas/site.schema.json`) and complete `site.json`
    configuration reference.
  - Starter template gallery (`/templates`) and `dioxuspress-cli` (`check`,
    `new`, `build` helpers) for zero-compile content deployments.
- **10 WCAG-AA Dual-Mode Themes**:
  - Paired light (`:root`) and dark (`.dark`) presets (`ocean`, `emerald`,
    `nord`, `corporate`, `winter`, `dracula`, `sunset`, `coffee`, `catppuccin`,
    `silk`), `available_themes` allowlist, and interactive `<ThemeShowcase />`.
- **Interactive AI & WASM Atlas Modules**:
  - **Model Architecture & Safetensors Visualizer (`/models`)**: curated Model
    Atlas with online persistence, collapsible architecture graphs, and support
    for LLM, Diffusers image/video pipelines, audio, hybrid linear SSM, and
    embedding models.
  - **AI Components Atlas (`/ai/components`)**: 8 pure-Rust WASM operator labs
    (attention, sequence, audio, routing, and more).
  - **AI Math Foundations (`/ai/math`)**: interactive proof lab and WASM
    canvases covering spectral decomposition, optimizer trajectories, and RL
    precision.
- **MDX & Publishing Enhancements**:
  - Interactive zoomable dual-theme **SVG Diagram Viewer** and frontmatter
    `image` OpenGraph SEO metadata support.
  - Expanded **Prism syntax highlighting bundle to 40 languages** (including
    `typst`, `c`, `cpp`, `msl`, `cuda`, `wgsl`, `sql`, `diff`) with `:::code-tabs`.
  - `<VocabCard />` MDX widget for interactive terminology cards.
  - Full `CONTENT_DIR` overlay for `/case` showcase entries and detail cover OG
    metadata.
- **Multi-Arch Docker & Gateway**:
  - Pingora SNI gateway (`crates/gateway`) supporting multi-domain TLS
    termination and `HOST_REDIRECTS`.
  - GitHub Actions multi-arch (`linux/amd64` + `linux/arm64`) GHCR release
    workflow for `dioxuspress-basic`, `dioxuspress-app`, and
    `dioxuspress-gateway`.

### Changed
- Upgraded `dioxus-shadcn` to **`0.6.8`** (100 public components, zero
  `'unsafe-eval'` CSP compatibility).
- Synchronized active `zh` / `en` language across sibling ecosystem links
  (`dioxuspress.dev`, `ui.dioxuspress.dev`, `rustineverything.app`).
- Pre-fetched Cargo dependencies into the shared BuildKit cache in `Dockerfile`
  prior to `dx bundle`.
- Gated `migration` strictly on the `accounts` feature tier and unified
  `pulldown-latex` in workspace dependencies.

### Fixed
- Resolved RustSec advisories by updating `lettre`, `cookie`, `url`, and
  `quinn-udp` (`0.5.16`).
- Replaced `/topics/new` robots disallow with `noindex` and auth-gated actions.
- Scoped `unreachable_code` lint on `Routable` derives for Rust 1.100+.
- Hardened HTTP/2 session cookie handling, SSR theme CSS hydration, Mermaid
  dark-mode palettes, and bilingual About / Docs pages.

## [0.1.0] - 2026-10-30

Initial open-source release of DioxusPress built on Dioxus 0.7 fullstack (SSR +
WASM hydration), Axum, SeaORM/PostgreSQL, embedded Tantivy search, four Cargo
feature tiers (`basic`, `accounts`, `course`, `full`), and `dioxus-shadcn`.
