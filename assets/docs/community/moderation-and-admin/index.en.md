---
title: "Content Moderation Pipeline (URL Blocklist + LLM) & /admin Console"
sidebar_label: Moderation & Admin
sidebar_position: 4
description: Block spam links deterministically, score UGC with OpenAI/Anthropic-compatible LLMs, and manage review queues live in /admin.
keywords: [content moderation, URL blocklist, LLM safety, admin dashboard, spam prevention]
---

# Content Moderation Pipeline & `/admin` Console

Public comment boxes and forums are prime targets for SEO spam injection. DioxusPress ships a two-stage `ModerationPipeline` paired with a built-in `/admin` console.

## Two-Stage Moderation Pipeline

1. **Stage 1: Deterministic URL Blocklist (Zero Latency)**
   Matches exact domains (`"scam.example"`) and wildcard subdomains (`"*.phishing.example"`). Matching submissions are blocked immediately without spending a single LLM token.
2. **Stage 2: Optional LLM Semantic Review**
   Compatible with OpenAI and Anthropic APIs (`OPENAI_LLM_API_KEY` / `OPENAI_LLM_BASE_URL` / `OPENAI_LLM_MODEL`). Submissions route by risk score:
   - `< flag_above` (default `0.5`): Published immediately;
   - `flag_above .. block_above` (`0.5 .. 0.9`): Held in the **Manual Review Queue**, indexed into search only after an admin approves;
   - `>= block_above` (default `0.9`): Rejected immediately.

:::important
**Fail-Safe by Type System**: When LLM review is enabled and the upstream API times out or errors, `on_llm_failure` only permits `"review"` (hold for admin approval, default) or `"reject"`. Attackers cannot bypass moderation by exhausting your LLM quota.
:::

## `/admin` Console Capabilities

Administrators visiting `/admin` can manage day-to-day operations from one place:
- **Overview Dashboard**: User totals, pending moderation count, and recent forum/comment activity;
- **Moderation Queue (`/admin/moderation`)**: One-click Approve or Reject for flagged items;
- **Live Moderation Settings (`/admin/moderation/settings`)**: Toggle rules, adjust score thresholds, and edit URL blocklists in the browser — persisted back to `site.json` and hot-reloaded in memory;
- **Users, Topics, Comments & Entitlements**: Role management, content cleanup, and manual course/Pro grants.
