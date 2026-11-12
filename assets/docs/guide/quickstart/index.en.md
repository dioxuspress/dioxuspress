---
title: "5-Minute Quickstart: Zero-Compile Docker Setup, CLI Scaffolding & Source Development"
sidebar_label: 5-Minute Quickstart
sidebar_position: 1
description: Launch a documentation site or blog in under 50 seconds with the dioxuspress CLI and prebuilt multi-arch Docker image, or clone the engine for fullstack Rust development.
keywords: [DioxusPress install, dioxuspress init, dioxuspress check, zero database setup, Docker docs]
---

# 5-Minute Quickstart

DioxusPress offers three clear onboarding paths depending on what you are building:

1. **Zero-Compile 50-Second Setup (Recommended for Authors)**: No Rust toolchain required on the server — scaffold a content folder and run the official prebuilt container.
2. **Local Engine Development (Zero-Database Mode)**: Clone the Rust workspace to customize UI components, Tailwind styles, or MDX widgets.
3. **Fullstack Community Portal (with PostgreSQL)**: Enable OAuth2 sign-in, discussion forums, inline paragraph annotations, and paid courses.

---

## Path 1: Zero-Compile 50-Second Setup (Recommended)

When your goal is writing Markdown documentation or technical blog posts, pair the lightweight `dioxuspress` CLI with our multi-arch image `ghcr.io/dioxuspress/dioxuspress-basic` (`linux/amd64` + Apple Silicon `linux/arm64`):

### 1. Install the Lightweight CLI (Optional — compiles in ~6 seconds)

```bash
cargo install --git https://github.com/dioxuspress/dioxuspress dioxuspress-cli
```

:::tip Pure Docker Without Installing Rust
If Rust is not installed locally, you can skip the CLI altogether, create a folder containing `docs/index.md`, and run:
`docker run --rm -p 8080:8080 -v ./my-site:/app/content:ro ghcr.io/dioxuspress/dioxuspress-basic:latest`
:::

### 2. Scaffold & Validate Your Site

`dioxuspress init` ships with four scenario-tailored presets (`docs`, `blog`, `community`, and `portal` — preview them side by side in the [Templates Showcase](/templates)):

```bash
# Generate site.json (with IDE JSON Schema autocomplete), starter docs, and docker-compose.yml
dioxuspress init my-docs --template docs
cd my-docs

# Millisecond offline check for config typos, YAML frontmatter, code fences, and broken internal links
dioxuspress check
```

### 3. Start the Container

```bash
docker compose up -d
```

Open `http://localhost:8080` in your browser to explore the Docusaurus-style landing page, three-level sidebar navigation, dual-mode theme switcher, and `⌘K` bilingual full-text search.

---

## Path 2: Local Engine Development (Zero-Database Mode)

When modifying Rust RSX components or Tailwind CSS styles, use the `dx` CLI with hot reload:

### Prerequisites

- **Stable Rust toolchain** + WASM target:
  ```bash
  rustup update stable
  rustup target add wasm32-unknown-unknown
  ```
- **Dioxus CLI (`dx`)**:
  ```bash
  cargo install dioxus-cli
  ```

### Run the `basic` Tier Dev Server

```bash
git clone https://github.com/dioxuspress/dioxuspress.git
cd dioxuspress/crates/app
BASE_URL=http://127.0.0.1:8080 dx serve --web --no-default-features
```

:::tip
Whenever `--no-default-features` is passed to `dx serve`, explicitly include `--web` so `dx` selects the fullstack web target.
:::

---

## Path 3: Fullstack Community Mode (Auth, Forum & Courses)

To test OAuth2 sign-in, comments, forum topics, inline annotations, or the `/admin` dashboard locally, start PostgreSQL and run with default features:

```bash
# 1. Copy the environment template at the repo root and set JWT_SECRET (e.g. openssl rand -hex 32) and DATABASE_URL
cp .env.example .env

# 2. Start PostgreSQL 16 (SeaORM migrations run automatically on app startup)
docker compose up -d postgres

# 3. Launch the fullstack dev server
cd crates/app
dx serve
```

:::warning Avoid Mixing `localhost` and `127.0.0.1` During OAuth Testing
When testing GitHub, Google, Discord, or X OAuth locally, always open the exact hostname configured in `BASE_URL` (for example `http://127.0.0.1:8080`). Mixing `localhost` and `127.0.0.1` prevents the browser from sending the PKCE cookie on callback.
:::

## Next Steps

- Read the [`site.json` Configuration Reference](/docs/guide/site-config) to customize branding, navigation, and themes with IDE autocomplete;
- Follow the [Zero-Database Deployment Guide](/docs/operations/zero-db-deploy) to ship your content directory to any Linux VPS in seconds;
- Add `dioxuspress check` to GitHub Actions so every content PR catches broken links and syntax errors before merge.
