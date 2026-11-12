# DioxusPress

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/images/logo-dark.png" />
    <source media="(prefers-color-scheme: light)" srcset="assets/images/logo-light.png" />
    <img src="assets/images/logo-light.png" alt="DioxusPress" height="64" />
  </picture>
</p>

<p align="center">
  <strong>English</strong> · <a href="README.zh-CN.md">简体中文</a>
</p>

An open-source, pure-Rust fullstack documentation, content publishing, and community engine built on [Dioxus 0.7](https://dioxuslabs.com). Run it as a **zero-database `basic` tier** single-container alternative to Docusaurus / VitePress, or opt into OAuth login, forums, paragraph annotations, paid courses, and online payments via Cargo features.

- **Official Website & Docs**: [`dioxuspress.dev`](https://dioxuspress.dev) (`active_layout = "docs"`, running zero-database in a single container)
- **UI Component Showcase**: [`ui.dioxuspress.dev`](https://ui.dioxuspress.dev) ([`dioxus-shadcn`](https://github.com/dioxuspress/dioxus-ui) — 84 Tailwind-styled Dioxus 0.7 components)
- **Flagship Fullstack Portal**: [`rustineverything.app`](https://rustineverything.app) (`active_layout = "classic"`, serving 38 volumes / 2,100+ pages of docs, courses, and community)

Customize [`assets/site.json`](assets/site.json) (`site_name`, `active_layout` — `docs` / `classic` / `minimal`, `theme`, `hero`, `features`, and `modules`) to launch your own documentation site or fullstack community in minutes.

## Tech Stack

- **Fullstack Rust**: Dioxus 0.7 fullstack (SSR + WebAssembly hydration) + Axum, SeaORM + PostgreSQL (optional in `basic` tier).
- **UI Components**: [`dioxus-shadcn`](https://github.com/dioxuspress/dioxus-ui) 0.6 (`shadcn/ui` implemented in pure Dioxus, [live showcase](https://ui.dioxuspress.dev)). Semantic design tokens serve as the single source of color across the entire site.
- **Styling & Themes**: Tailwind CSS v4 + zero-flash dark mode; built-in visitor-switchable themes (`ocean`, `sunset`, `catppuccin`) and bilingual i18n (`zh` / `en`).
- **Security Baseline**: Strict Content-Security-Policy (no `'unsafe-eval'`), allowlist sanitization for user-generated content, OAuth2 + PKCE, rate limiting, and explicit confirmation on destructive admin actions.
- **Full-Text Search**: Embedded [Tantivy](https://github.com/quickwit-oss/tantivy) search engine with Chinese and English tokenization (`⌘K` instant search).

## Features

- **Git-First Content** (filesystem as the single source of truth under `assets/` or `CONTENT_DIR`):
  - **Documentation**: 3-level hierarchical docs volumes with breadcrumbs, prev/next navigation, On-This-Page TOC, and "Edit this page on GitHub".
  - **Blog**: MDX authoring (YAML frontmatter, GFM, KaTeX math, Mermaid diagrams, syntax highlighting, and interactive Rust components).
  - **Courses**: Course → Chapter → Lesson hierarchy with free/paid lessons, Pro membership, and online checkout (`payments` feature).
  - **Showcase, Podcast & Ecosystem Hubs**: Dedicated sections for Showcase (`/case`), Podcast, Embedded, AI, WebAssembly, Web3, CLI, and Papers.
- **Community & Engagement**: Forum (topics & replies, one-click discussion threads from articles), comments, paragraph-level inline annotations, and image uploads.
- **Moderation Pipeline**: Domain blocklist + optional LLM content moderation with a human review queue.
- **Authentication**: GitHub / Google / Discord / X (OAuth2 + PKCE) with signed JWT sessions.
- **Admin Console** (`/admin`, admin-only): Dashboard overview, users & roles, comments, forum topics, moderation queue & policy settings, and course entitlements.

Themes, auth providers, and moderation policies compile directly into a single native Rust binary — no runtime dynamic plugin overhead.

## Feature Tiers

A single codebase compiles into four progressive capability tiers via Cargo features (see [`docs/FEATURE_TIERS.md`](docs/FEATURE_TIERS.md)):

| Tier | Cargo Flags | Included Capabilities | Runtime Requirements |
| --- | --- | --- | --- |
| **Basic** | `--no-default-features` | Docs, Blog, Podcast, Showcase, Ecosystem hubs, Tantivy Search, About, Layouts / Themes / Dark Mode / i18n | Only `BASE_URL` (**zero database**) |
| **Accounts** | `--no-default-features --features accounts` | + OAuth Login, Forum, Comments, Uploads, Admin Console | + PostgreSQL, `JWT_SECRET` |
| **Course** | `--no-default-features --features course` | + Courses, Inline Annotations, Entitlements Admin, Pro Membership | Same as Accounts |
| **Full (Default)** | *(default)* | + Online Payments (WeChat Pay / Alipay) | + Payment merchant keys |

Run a specific tier locally:

```bash
# Zero-database documentation & blog mode (Basic tier)
cd crates/app && dx serve --web --no-default-features

# Community portal mode (Accounts tier)
cd crates/app && dx serve --web --no-default-features --features accounts
```

> Pass `--web` explicitly whenever `--no-default-features` is used so `dx` knows which target platform to serve.

## Quick Start

### Option A: Zero-Database Docs & Blog (`basic` tier)

```bash
# 1. Clone the repository & install Dioxus CLI
git clone https://github.com/dioxuspress/dioxuspress.git
cd dioxuspress
cargo install dioxus-cli

# 2. Build Tailwind CSS (first run or after adding utility classes)
cd crates/app && npm install && npm run build

# 3. Serve locally (no PostgreSQL needed)
BASE_URL=http://127.0.0.1:8080 dx serve --web --no-default-features
```

### Option B: Fullstack Community & Courses (with PostgreSQL)

```bash
# 1. Configure site metadata (assets/site.json) and environment variables
cp .env.example .env

# 2. Start PostgreSQL (migrations run automatically on startup)
docker compose up -d postgres

# 3. Build Tailwind CSS and start the dev server
cd crates/app && npm install && npm run build
dx serve
```

Open the exact `BASE_URL` configured in `.env` (e.g., `http://127.0.0.1:8080`). Visiting via `localhost` when `BASE_URL` uses `127.0.0.1` will cause OAuth callback cookies not to match.

**First Admin User**: Sign in once via OAuth, run `set -a; source .env; set +a; scripts/promote_admin.sh <username>` from the repository root (requires `psql`), then sign out and sign back in to receive an `admin` JWT session. See [`docs/ADMIN_SPEC.md`](docs/ADMIN_SPEC.md).

## Verified Performance & Footprint

Reproduce every metric locally with [`./scripts/bench_ssr.sh`](scripts/bench_ssr.sh) and [`./scripts/check_wasm_size.sh`](scripts/check_wasm_size.sh) (see [`dioxuspress.dev/docs/operations/benchmarks?lang=en`](https://dioxuspress.dev/docs/operations/benchmarks?lang=en)):

| Metric | `basic` Tier (Zero-DB Docs & Blog) | `accounts` / `full` Tier | Notes |
| --- | --- | --- | --- |
| **Release Binary / Image** | `34.2 MiB` / `269 MB` | `47.4 MiB` / `319 MB` | Pingora SNI gateway image is `143 MB` (`6.4 MiB` live RSS) |
| **Cold Start (`/healthz 200`)** | `547 ms` | `549 ms` | Ready to serve HTTP before background incremental index pass |
| **Idle Memory (RSS)** | `91.7 MiB` | `95.6 MiB` | Includes embedded Jieba dictionary + Tantivy mmap |
| **First-Paint SSR (`GET /` / `/docs/*`)** | `2.79 ms` / `4.04 ms` P50 (`1,832` / `1,236 QPS`) | `2.90 ms` / `4.02 ms` P50 (`1,764` / `1,234 QPS`) | Full Markdown parse + 3-level sidebar + VirtualDOM SSR |
| **Conditional `ETag` (`304`)** | **`0.096 ms` P50 (`13,636 QPS`)** | **`0.093 ms` P50 (`13,462 QPS`)** | Anonymous-only; auth/dynamic routes strictly emit `private, no-store` |
| **Tantivy Search (`POST /api/search/query`)** | **`0.314 ms` P50 (`9,215 QPS`)** | **`0.317 ms` P50 (`9,326 QPS`)** | Bilingual BM25 + snippet highlighting; `SIGHUP` hot-reloads with `0` dropped requests |
| **Frontend WASM (`brotli -q 11`)** | **`1.04 MiB`** (`1.54 MiB` gzip) | **`1.35 MiB`** (`2.07 MiB` gzip) | Guarded on every PR in CI by `scripts/check_wasm_size.sh` |

## Pre-Commit Verification

```bash
cargo fmt --all -- --check
cargo clippy --workspace --features server --all-targets -- -D warnings
cargo test --features server --workspace -- --test-threads=1
cargo check -p app --target wasm32-unknown-unknown --features web
scripts/check_tiers.sh      # Verifies all 4 tiers × (server + web) compile cleanly
scripts/check_wasm_size.sh  # Verifies wasm-release bundle sizes stay under budget
```

## Tailwind CSS Build Pipeline

Tailwind CSS v4 inputs and Node scripts live under `crates/app/`:

- **Input**: `crates/app/tailwind-input.css` (includes `@import "tailwindcss"` and auto-generated `@source` directives scanning `dioxus-shadcn` component sources)
- **Output**: `crates/app/assets/tailwind.css` (gitignored, built locally/in Docker) → automatically synced to `assets/tailwind.css`
- **Watch mode**: `cd crates/app && npm run dev`
- **One-shot build**: `cd crates/app && npm run build`

See [`docs/TAILWIND_GUIDE.md`](docs/TAILWIND_GUIDE.md) for details.

## Architecture & Engineering Docs

- [Live Documentation (`dioxuspress.dev/docs`)](https://dioxuspress.dev/docs)
- [Developer Guide](docs/DEVELOPER.md) / [Architecture Assessment](docs/ARCHITECTURE_ASSESSMENT.md)
- [Engines](docs/ENGINES_SPEC.md) / [Module System](docs/MODULE_SPEC.md) / [Feature Tiers](docs/FEATURE_TIERS.md)
- [`dioxus-shadcn` Migration](docs/DIOXUS_UI_MIGRATION.md) / [UI Library Feedback](docs/DIOXUS_UI_FEEDBACK.md)
- [MDX Content](docs/MDX_SPEC.md) / [Blog](docs/BLOG_SPEC.md) / [Docs Module](docs/DOCS_MODULE_SPEC.md) / [Courses](docs/COURSE_SPEC.md) / [Payments](docs/PAYMENT_SPEC.md)
- [Forum](docs/FORUM_SPEC.md) / [Comments](docs/COMMENTS_SPEC.md) / [Annotations](docs/ANNOTATION_SPEC.md) / [Uploads](docs/UPLOADS_SPEC.md)
- [Moderation](docs/MODERATION_SPEC.md) / [Admin Console](docs/ADMIN_SPEC.md)
- [Authentication](docs/AUTH_SPEC.md) / [Auth Setup Guide](docs/AUTH_GUIDE.md) / [Sessions](docs/SESSION_SPEC.md)
- [Themes](docs/THEME_SPEC.md) / [Search](docs/SEARCH_SPEC.md) / [SEO](docs/SEO_SPEC.md)
- [Deployment Guide](docs/DEPLOY_GUIDE.md) / [Operations](docs/OPERATIONS.md) / [Security Remediation](docs/SECURITY_REMEDIATION.md)

## License

[MIT](LICENSE)
