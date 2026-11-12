---
title: "Four Compile-Time Feature Tiers (Cargo Features): From Static Docs to Paid Course Communities"
sidebar_label: Feature Tiers
sidebar_position: 2
description: Understand how DioxusPress prunes modules at compile time across basic, accounts, course, and full payments tiers.
keywords: [Cargo features, feature tiers, zero database, conditional compilation, DioxusPress architecture]
---

# Feature Tiers (Cargo Features)

DioxusPress organizes each domain into an isolated workspace crate and wires them at the `crates/app` composition root via Cargo features. Disabled capabilities are not merely hidden in the UI — their backend API handlers, SeaORM entities, and database migrations are completely omitted from the compiled binary.

## Tier Capabilities & Runtime Matrix

| Tier | Build Flags (`crates/app`) | Included Capabilities | Required Env Vars |
| :--- | :--- | :--- | :--- |
| **`basic`** (Docs / Blog) | `--no-default-features` | Docs, Blog, Cases, Podcast, Topic Boards, Tantivy Search, About, Themes/Dark Mode/i18n, Sitemap/Atom Feed | Optional `BASE_URL` (**No Database**) |
| **`accounts`** (Community) | `--no-default-features --features accounts` | `basic` + OAuth2 PKCE Sign-In, Forum Topics, Post Comments, Image Uploads, LLM Moderation, `/admin` Console | + `DATABASE_URL`, `JWT_SECRET` |
| **`course`** (Courses & Pro) | `--no-default-features --features course` | `accounts` + Structured Courses, Free Previews / Server Paywall, Paragraph Annotations, Manual Entitlements | Same as `accounts` |
| **`full`** (Default) | *(default — no extra flags)* | `course` + WeChat Pay / Alipay / Stripe Checkout, Signed Webhook Fulfillment, Orders & TipJar | Same + Payment Merchant Keys |

## Two-Layer Control: Compile-Time Pruning × Runtime Switches

DioxusPress separates module control into two complementary layers:

1. **Compile-Time Physical Boundary (Cargo Features)**: At startup, `register_compiled_modules` records only the crates linked into the binary. Uncompiled routes degrade cleanly to a standard `404` via `absent_pages!`, leaving zero unused attack surface.
2. **Per-Site Runtime Toggles (`site.json::modules`)**: Even when running a full binary, any compiled module can be toggled off in `site.json` (for example keeping only `docs` and `blog`). Top navigation, `ModuleGate` route guards, `⌘K` search indexing, `/sitemap.xml`, and `/feed.xml` update in lockstep:

```json
{
  "modules": {
    "docs": { "enabled": true },
    "blog": { "enabled": true },
    "cases": { "enabled": true },
    "podcast": { "enabled": false },
    "embedded": { "enabled": false }
  }
}
```

## Pre-Merge 8-Way Tier Verification

If you customize framework components or routes, run the built-in matrix script before opening a PR to verify all 4 feature tiers × (`server` + `web` WASM) = 8 targets with zero warnings:

```bash
./scripts/check_tiers.sh
```
