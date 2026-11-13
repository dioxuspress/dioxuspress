---
title: "site.json Complete Configuration Reference: JSON Schema, Branding, Layouts & Module Switches"
sidebar_label: site.json Reference
sidebar_position: 3
description: Authoritative field dictionary for DioxusPress site.json, covering IDE JSON Schema autocomplete, startup diagnostics, dual-mode logos, hero sections, multi-column footers, and moderation rules.
keywords: [site.json, JSON Schema, site configuration, dual-mode logo, announcement bar, footer columns, module switches]
---

# `site.json` Configuration Reference

`assets/site.json` (or `$CONTENT_DIR/site.json` when mounting an external content repository) is the **non-secret site configuration file** for DioxusPress. All sensitive credentials (database URLs, JWT secrets, OAuth client secrets, payment keys, and LLM API keys) are injected strictly through environment variables — see the [Environment & Gateway Reference](/docs/operations/env-reference).

---

## 1. IDE Autocomplete & Startup Diagnostics

### Real-Time JSON Schema Validation

DioxusPress hosts an official JSON Schema generated directly from the `SiteConfig` Rust struct. Add `"$schema"` at the top of `site.json` to get **hover documentation, enum dropdowns, and inline type squiggles** in VS Code, Zed, and RustRover:

```json
{
  "$schema": "https://dioxuspress.dev/schemas/site.schema.json",
  "site_name": "My Docs",
  "active_layout": "docs",
  "theme": "ocean"
}
```

:::tip Offline Schema & Keeping Custom Forks in Sync
The schema is also served locally by every instance at `/schemas/site.schema.json` (`assets/schemas/site.schema.json`). If you add fields to `SiteConfig` in `crates/core/src/settings.rs`, regenerate the schema and pass the drift test with:

```bash
UPDATE_SCHEMA=1 cargo test --features server -p app-core site_schema_json_is_up_to_date
```
:::

### Non-Blocking Startup Diagnostics

To prevent handwritten JSON typos (such as writing `layout` instead of `active_layout`, or `hero.titel` instead of `hero.title`) from being silently ignored by `serde` defaults, the server runs `diagnose_site_json` at boot and logs structured `WARN` messages without blocking startup:

- **Unknown field detection** across the root object and all nested blocks (`announcement`, `hero`, `features[*]`, `footer`, `annotations`, `auth`, `modules.*`, `moderation`);
- **Enum validation** for `active_layout`, `default_language`, `theme`, `available_themes`, and `auth.providers`;
- **Asset existence checks** ensuring `/images/*` paths referenced by `logo`, `favicon`, and `og_image` exist under `$CONTENT_DIR/images` or `assets/images`.

---

## 2. Core Identity, Layout & Theme

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `$schema` | `string` | — | JSON Schema URL; set to `"https://dioxuspress.dev/schemas/site.schema.json"` |
| `site_name` | `string` | `"DioxusPress"` | Site name appended to `<title>` tags, used as navbar text fallback and copyright owner |
| `site_description` | `string` | `""` | Default homepage `<meta name="description">` and Atom/RSS `<subtitle>` |
| `author` | `string` | `""` | Default site author attribution |
| `default_language` | `"zh" \| "en"` | `"zh"` | Fallback UI and docs language when neither `?lang=` nor the `site_lang` cookie is present |
| `active_layout` | `"classic" \| "docs" \| "minimal"` | `"classic"` | Global shell layout:<br>• `"docs"`: Docusaurus-style open-source docs shell (announcement bar + project hero + multi-column footer)<br>• `"classic"`: Multi-section knowledge portal<br>• `"minimal"`: Distraction-free reading header with no global footer |
| `theme` | `string` | `"ocean"` | Default dual-mode color theme (`"ocean"`, `"emerald"`, `"nord"`, `"corporate"`, `"winter"`, `"dracula"`, `"sunset"`, `"coffee"`, `"catppuccin"`, `"silk"`); see [Layouts & Themes](/docs/guide/layouts-and-themes) |
| `available_themes` | `string[]` | `[]` | Ordered allowlist of theme IDs exposed in the topbar `ThemePicker` (`[]` exposes all 10; a single-item array such as `["ocean"]` locks the brand palette and hides the picker) |

---

## 3. Branding Assets, PWA Icons & Social Cards

Every `/images/*` path supports **overlay resolution**: when `CONTENT_DIR` is mounted, `$CONTENT_DIR/images/*` takes precedence and automatically falls back to built-in `assets/images/*`.

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `favicon` | `string` | `"/images/favicon.ico"` | Primary browser tab icon (`.ico` or `.svg`) |
| `favicon_16` | `string \| null` | `null` | Optional 16×16 PNG icon |
| `favicon_32` | `string \| null` | `null` | Optional 32×32 PNG icon |
| `apple_touch_icon` | `string \| null` | `null` | Optional iOS home-screen icon (180×180 PNG recommended) |
| `manifest` | `string \| null` | `null` | Optional Web App Manifest path |
| `logo` | `string \| null` | `null` | Light-mode (or universal) navbar logo; `null` renders text `site_name` |
| `logo_dark` | `string \| null` | `null` | Dark-mode navbar logo swapped via pure CSS (`dark:hidden` / `hidden dark:block`) with zero hydration flash |
| `logo_mobile` | `string \| null` | `null` | Compact square icon shown on narrow viewports |
| `logo_wordmark` | `boolean` | `false` | Set `true` if `logo` already embeds the site name text so the navbar does not render duplicate text beside it |
| `og_image` | `string` | `"/images/og-default.jpg"` | Default Open Graph / Twitter `summary_large_image` card (1200×630 recommended); overridable per page via Frontmatter `image` |
| `twitter_handle` | `string \| null` | `null` | Optional X / Twitter handle (e.g. `"@dioxuspress"`) injected into `<meta name="twitter:site">` |
| `docs_edit_url_base` | `string \| null` | `null` | Base URL for the "Edit this page" link on doc pages (e.g. `"https://github.com/org/repo/edit/main/assets/docs"`) |
| `submit_case_url` | `string \| null` | `null` | External "Submit Showcase" button URL on `/case` (`null` hides the button) |

---

## 4. Announcement Bar (`announcement`), Hero (`hero`) & Features (`features`)

### `announcement` (Dismissible Top Banner)

| Field | Type | Description |
| :--- | :--- | :--- |
| `announcement.id` | `string` | Version identifier stored in `localStorage["dp_ann_<id>"]` on dismiss; bump `id` to show a new release banner to returning visitors |
| `announcement.text` | `string` | Default / Chinese banner text (empty string disables the banner) |
| `announcement.text_en` | `string \| null` | English banner text (falls back to `text`) |
| `announcement.href` | `string \| null` | Optional internal route or external URL |

### `hero` (Homepage Lead Section)

Unset text fields fall back to `hero.*` keys in `assets/i18n/{zh,en}.ftl`:

| Field | Type | Description |
| :--- | :--- | :--- |
| `hero.badge` / `hero.badge_en` | `string \| null` | Pill badge above the headline (rendered in `active_layout = "docs"`) |
| `hero.title` / `hero.title_en` | `string \| null` | Primary hero headline |
| `hero.subtitle` / `hero.subtitle_en` | `string \| null` | Supporting lead paragraph |
| `hero.install_command` | `string \| null` | One-click copyable terminal command box (`active_layout = "docs"`) |
| `hero.primary_cta_text` / `primary_cta_text_en` | `string \| null` | Primary CTA button label |
| `hero.primary_cta_href` | `string \| null` | Primary CTA target route or URL |
| `hero.secondary_cta_text` / `secondary_cta_text_en` | `string \| null` | Secondary CTA button label |
| `hero.secondary_cta_href` | `string \| null` | Secondary CTA target route or URL |

### `features` (3-Column Feature Grid in `docs` Layout)

```json
"features": [
  {
    "icon": "bolt",
    "title": "全栈 Rust SSR + WASM 水合",
    "title_en": "Fullstack Rust SSR & Hydration",
    "description": "首屏直出完整语义化 HTML，水合后享受无刷新秒开导航。",
    "description_en": "Instant crawler-friendly HTML first paint with smooth SPA navigation.",
    "href": "/docs/guide"
  }
]
```

- Built-in vector `icon` names: `"bolt"`, `"layers"`, `"book"`, `"search"`, `"palette"`, `"shield"` (any other short string or emoji is rendered centered as text).

---

## 5. Footer (`footer`) & Custom Topbar Navigation (`navigation`)

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `footer.tagline` / `tagline_en` | `string \| null` | `null` | Footer brand tagline (falls back to i18n `footer.tagline`) |
| `footer.copyright` | `string \| null` | `null` | Copyright holder text (falls back to `site_name`) |
| `footer.icp` | `string \| null` | `null` | Optional regulatory filing badge text |
| `footer.icp_url` | `string \| null` | `"https://beian.miit.gov.cn/"` | Regulatory link URL |
| `footer.social` | `array` | `[]` | Array of `{ "label": "GitHub", "url": "https://..." }`; the first GitHub URL also populates the top-right GitHub icon in `docs` layout |
| `footer.columns` | `array` | `[]` | Multi-column link groups with `title`, `title_en`, and `links: [{ "label", "label_en", "href" }]` |
| `navigation` | `array` | `[]` | Custom topbar links in `docs` layout: `{ "key": "nav.templates or literal label", "route": "/templates", "new_tab": false }` (`http(s)://` opens in a new tab by default; set `new_tab: false` for same-tab sibling domains; internal routes are automatically added to `/sitemap.xml`) |

---

## 6. Modules, Auth, Annotations & Moderation

### `modules` (Runtime Module Toggles)

Disabling any compiled module in `site.json` removes it simultaneously from **topbar navigation, `ModuleGate` routes, `⌘K` Tantivy search, `/sitemap.xml`, and `/feed.xml`**:

```json
"modules": {
  "docs": { "enabled": true },
  "blog": { "enabled": true },
  "cases": { "enabled": false },
  "podcast": { "enabled": false },
  "forum": { "enabled": false },
  "course": { "enabled": false },
  "papers": { "enabled": false },
  "embedded": { "enabled": false },
  "ai": { "enabled": false },
  "web3": { "enabled": false },
  "wasm": { "enabled": false },
  "cli": { "enabled": false },
  "about": { "enabled": true }
}
```

:::info Compile-Time Tiers vs. Runtime Switches
`site.json::modules` filters within the modules compiled into your binary. In the zero-database `basic` tier, `forum`, `course`, and `papers` are not compiled at all. See [Feature Tiers](/docs/guide/feature-tiers).
:::

### `auth` (OAuth2 Providers — requires `accounts` tier)

```json
"auth": {
  "enabled": true,
  "providers": ["github", "google", "discord", "twitter"]
}
```

- Buttons appear in the sign-in modal in the declared order **only when the corresponding `*_CLIENT_ID` and `*_CLIENT_SECRET` environment variables are non-empty**.

### `annotations` (Inline Paragraph Highlights — requires `accounts` tier)

```json
"annotations": {
  "course": true,
  "doc": true,
  "blog": false
}
```

### `moderation` (URL Blocklist & Fail-Safe LLM Review — requires `accounts` tier)

Disabled by default (`enabled: false`). When enabled, guards comments, forum topics/replies, and paragraph notes:

```json
"moderation": {
  "enabled": true,
  "url_blocklist": ["scam.example", "*.phishing.example"],
  "llm_review": true,
  "on_llm_failure": "review",
  "thresholds": {
    "block_above": 0.9,
    "flag_above": 0.5
  }
}
```

- `on_llm_failure` accepts only `"review"` (default: queue for manual admin approval on LLM timeout/error) or `"reject"` (reject submission). By security design (`SEC-12`), **there is no fail-open option**, preventing attackers from bypassing moderation by exhausting API quotas.
