---
title: "Community Engagement, Content Governance & Monetization Overview"
sidebar_label: Accounts & Community
sidebar_position: 3
description: Overview of DioxusPress's OAuth2 PKCE authentication, forum topics, inline paragraph annotations, fail-safe LLM moderation, and native payments.
keywords: [OAuth2 sign-in, Rust forum engine, paragraph annotations, LLM moderation, online payments]
---

# Community Engagement & Governance Overview

Once your documentation site or technical blog grows an active readership, there is no need to migrate stacks or bolt on third-party comment widgets. Enable the `accounts`, `course`, or `payments` Cargo features and connect PostgreSQL to unlock built-in community interaction and monetization.

## Core Design Principles

1. **Security Built In from Day One**:
   - Full **OAuth2 + PKCE + State** validation across all identity providers;
   - `HttpOnly + SameSite=Lax + Secure` JWT cookies, with logout strictly enforcing `POST` and same-origin headers against cross-site logout CSRF;
   - Strict CSP without `'unsafe-eval'` plus per-IP token-bucket rate limiting on `/api/*`.
2. **Discussions Anchored to Content**: Forum threads live both on `/topics` and inside contextual `DiscussionPanel` footers on individual doc pages and showcase items, turning reader questions into searchable community Q&A.
3. **Fail-Safe Content Moderation**: Built-in URL blocklists combine with optional OpenAI/Anthropic-compatible LLM scoring. Whenever an upstream LLM times out or fails, submissions default to manual review or rejection — never fail-open.

## Volume Chapters

- **[OAuth2 Sign-In & Session Security](/docs/community/auth)**
- **[Forum Topics & Post Comments](/docs/community/forum-and-comments)**
- **[Inline Paragraph Highlights & Notes](/docs/community/annotations)**
- **[Moderation Pipeline & `/admin` Console](/docs/community/moderation-and-admin)**
- **[Payments & TipJar Integration](/docs/community/payments)**
