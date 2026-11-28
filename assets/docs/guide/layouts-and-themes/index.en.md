---
title: "Built-In Shell Layouts (Docs / Classic / Minimal) & 10 Dual-Mode Themes Live Preview"
sidebar_label: Layouts & Themes
sidebar_position: 4
description: Compare DioxusPress's three full-site layouts, interactively preview all 10 WCAG AA dual-mode themes powered by dioxus-ui, and configure available_themes allowlists.
keywords: [Docusaurus layout, DocsShell, ClassicShell, dark mode, dioxus-shadcn, theme preview, available_themes]
---

# Layouts & Themes

DioxusPress cleanly decouples **page shell structure (`active_layout`)** from **semantic color tokens (`theme`)**. Without touching a single Markdown file or Rust component, you can switch between three full-site shells and 10 WCAG AA contrast-verified light/dark themes in `site.json`.

---

## 1. Three Built-In Shell Layouts (`active_layout`)

### `"docs"` — Docusaurus-Style Open-Source Documentation Shell
Purpose-built for SDK manuals, open-source documentation, and engineering handbooks:
- **Left-aligned clean topbar**: Brand logo paired with flat navigation links (`Docs / Templates / Blog / Showcase / About`), plus a right-side `⌘K` search pill, GitHub icon, swatch `ThemePicker`, language switcher, and dark-mode toggle.
- **Dismissible release banner**: Driven by `site.json::announcement`, persisted per `announcement.id`.
- **Dedicated open-source landing page (`DocsLanding`)**: Centered hero headline, one-click terminal install command, 3-column `features` grid, and auto-fetched documentation volume cards.
- **Structured multi-column footer**: Configured declaratively via `site.json::footer.columns`.

### `"classic"` — Multi-Section Knowledge Portal & Community Shell
Designed for large multi-domain portals such as [Rust in Everything](https://rustineverything.app):
- **Ecosystem Mega Menus**: Groups vertical topic boards (Embedded, AI, WASM, Web3, CLI, Daily Papers) alongside learning resources.
- **Portal Homepage**: Showcases featured community projects, structured courses, and a two-column feed of latest blog posts and active forum discussions.

### `"minimal"` — Distraction-Free Personal Blog Shell
- Compact 48px translucent glass header retaining only the brand mark, search trigger, and appearance controls, with the global footer hidden for long-form essays.

---

## 2. Live Preview: 10 Built-In Dual-Mode Themes

DioxusPress shares the shadcn/ui v4 semantic CSS token architecture with [`dioxus-ui` (`dioxus-shadcn`)](https://ui.dioxuspress.dev). Each of the 10 built-in presets pairs a tuned **light palette (`:root`)** with a **dark palette (`.dark`)**, enforced by automated unit tests to meet **WCAG AA contrast (`>= 4.5:1`)** on body copy and primary actions.

A tiny CSP-safe inline boot script (`DARK_MODE_BOOT_SCRIPT`, zero `'unsafe-eval'`) applies `html.dark` before first paint based on `localStorage.theme` or `prefers-color-scheme`, eliminating white flashes during SSR hydration.

Click **Live Try** on any card below to switch the entire site immediately (toggle the sun/moon icon in the topbar to inspect both polarities), then click **Copy JSON** for your `site.json`:

<ThemeShowcase />

### Built-In Theme Quick Reference

| Theme ID | Recommended Positioning | Light Palette (`:root`) | Dark Palette (`.dark`) |
| :--- | :--- | :--- | :--- |
| `"ocean"` *(default)* | Open-source engineering & modern SaaS docs | Crisp white + slate ink + electric cyan (`#0e7490`) | Zinc night (`#09090b`) + cyan accent (`#22d3ee`) |
| `"emerald"` | Developer SDKs, API references & DevOps | Pure white + pine slate + emerald (`#047857`) | Deep pine charcoal (`#0b1310`) + mint (`#34d399`) |
| `"nord"` | Low-glare handbooks & long reading sessions | Snow mist (`#f8fafc`) + frost blue (`#3b6ea5`) | Polar slate (`#242933`) + ice cyan (`#88c0d0`) |
| `"corporate"` | Enterprise portals, specs & standards | Pure white + GitHub Primer blue (`#0969da`) | GitHub Dark (`#0d1117`) + sky blue (`#58a6ff`) |
| `"winter"` | Compilers, systems programming & deep-dive blogs | Crisp white + indigo (`#4338ca`) | Tokyo Night (`#16161e`) + periwinkle (`#7aa2f7`) |
| `"dracula"` | CLI tools, creative coding & hacker projects | Soft lavender white (`#fcfbfe`) + royal violet (`#6d28d9`) | Dracula (`#1e1f29`) + neon purple (`#bd93f9`) |
| `"sunset"` | Course portals, product launches & communities | Warm cream (`#fffaf3`) + amber (`#d97706`) | Warm espresso (`#1c0d05`) + coral orange (`#fb923c`) |
| `"coffee"` | Independent blogs, newsletters & essays | Warm parchment (`#fbf8f3`) + roast brown (`#854d0e`) | Dark espresso (`#181310`) + latte gold (`#ddb892`) |
| `"catppuccin"` | Soothing pastel digital gardens | Catppuccin Latte (`#eff1f5`) + lavender blue | Catppuccin Macchiato (`#24273a`) + pastel blue |
| `"silk"` | Design systems, humanities & editorial sites | Ivory silk (`#faf8f6`) + rose crimson (`#be123c`) | Rosé Pine (`#191724`) + rose gold (`#ebbcba`) |

---

## 3. Controlling `theme` and `available_themes` in `site.json`

By default (`available_themes` omitted or `[]`), the topbar `ThemePicker` exposes all 10 built-in themes. You can tailor this behavior per site:

### Scenario A: Curate a 3-Theme Allowlist

```json
{
  "$schema": "https://dioxuspress.dev/schemas/site.schema.json",
  "active_layout": "docs",
  "theme": "emerald",
  "available_themes": ["emerald", "ocean", "nord"]
}
```

- The dropdown preserves the exact order in `available_themes` with dual light/dark color swatches;
- If `theme` is omitted from `available_themes`, DioxusPress automatically prepends it;
- Stale visitor `site_theme` cookies outside the allowlist safely fall back to `theme`.

### Scenario B: Lock a Single Brand Theme & Hide the Picker

To enforce a single brand palette across all visitors (while keeping the light/dark mode toggle), pass a single-element `available_themes` array:

```json
{
  "theme": "corporate",
  "available_themes": ["corporate"]
}
```

When `<= 1` theme is enabled, `ThemePicker` renders zero DOM nodes.
