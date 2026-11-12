---
title: "Environment Variables & Multi-Domain Gateway Reference: .env, OAuth, Payments & Pingora SNI"
sidebar_label: Env & Gateway Reference
sidebar_position: 5
description: Complete dictionary of DioxusPress server environment variables (.env) and Pingora SNI TLS gateway settings across all feature tiers.
keywords: [environment variables, .env, BASE_URL, JWT_SECRET, OAuth PKCE, Pingora gateway, SNI_CERTS]
---

# Environment Variables & Gateway Reference

DioxusPress follows 12-Factor App boundaries: **public branding and structure belong in [`site.json`](/docs/guide/site-config), while all secrets, database URLs, and host topology settings are injected via environment variables (or `.env`)**.

---

## 1. Core Runtime & Directory Variables

| Variable | Tiers | Requirement | Default / Example | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| `BASE_URL` | All | Required in `accounts`+; optional in `basic` | `http://localhost:8080` (`basic` fallback) | Public origin without trailing slash; used for `/sitemap.xml`, `/feed.xml`, canonical tags, and OAuth redirect URIs |
| `CONTENT_DIR` | All | Optional | Auto-detected when `/app/content` is non-empty | External read-only content directory overriding `site.json`, `docs/`, `posts/`, `courses/`, and `images/` |
| `SEARCH_INDEX_DIR` | All | Optional | `./data/search-index` | Persistent directory for Tantivy segments and `manifest.json` |
| `RUST_LOG` | All | Optional | `info` | `tracing-subscriber` filter directive (e.g. `info,app_core=debug,sqlx=warn`) |

:::warning Match Hostnames Exactly for Local OAuth
When OAuth sign-in is enabled, open the exact hostname configured in `BASE_URL` (e.g. `http://127.0.0.1:8080`). Visiting `http://localhost:8080` while `BASE_URL` uses `127.0.0.1` prevents the browser from returning the `oauth_pkce` cookie on callback.
:::

---

## 2. Database, Session Crypto & Migration Guards (`accounts` Tier & Up)

Ignored completely in the zero-database `basic` tier (`--no-default-features`):

| Variable | Requirement | Example / Generator | Purpose |
| :--- | :--- | :--- | :--- |
| `JWT_SECRET` | **Required in `accounts`+** | `openssl rand -hex 32` | Signs and verifies `HttpOnly` session JWTs (>= 32 chars; startup panics if left as `.env.example` placeholder) |
| `DATABASE_URL` | Recommended | `postgres://rie:pass@127.0.0.1:5432/rie` | PostgreSQL 16 connection URL; runs SeaORM migrations on startup |
| `DATA_ENCRYPTION_KEY` | Recommended | `openssl rand -hex 32` | Dedicated AES-256-GCM key (encrypts `oauth_pkce` cookies) decoupled from `JWT_SECRET` for independent rotation |
| `STRICT_MIGRATION` | Optional | `0` (`1` in prod Compose) | Set `1` to exit immediately if DB connection or migration fails |

---

## 3. Rate Limiting & Security Headers (Optional Across All Tiers)

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `RATE_LIMIT_API_RPS` / `RATE_LIMIT_API_BURST` | `20` / `60` | Per-IP token bucket for general `/api/*` Server Functions (`429 Too Many Requests` when exceeded) |
| `RATE_LIMIT_SENSITIVE_RPS` / `RATE_LIMIT_SENSITIVE_BURST` | `5` / `15` | Stricter per-IP bucket for `/api/auth/*` and `/api/pay/*` |
| `CSP_POLICY` | Built-in strict policy (no `'unsafe-eval'`) | Custom override for `Content-Security-Policy` |
| `RATE_LIMIT_DISABLED` / `SECURITY_HEADERS_DISABLED` | Unset | Local debugging only — **never set in production** |

---

## 4. OAuth2 Provider Credentials (`accounts` Tier)

Declare provider order in `site.json::auth.providers` and supply credentials for active providers (unconfigured providers hide automatically):

| Provider | Client ID | Client Secret | Redirect URI |
| :--- | :--- | :--- | :--- |
| **GitHub** | `GITHUB_CLIENT_ID` | `GITHUB_CLIENT_SECRET` | `{BASE_URL}/api/auth/github/callback` |
| **Google** | `GOOGLE_CLIENT_ID` | `GOOGLE_CLIENT_SECRET` | `{BASE_URL}/api/auth/google/callback` |
| **Discord** | `DISCORD_CLIENT_ID` | `DISCORD_CLIENT_SECRET` | `{BASE_URL}/api/auth/discord/callback` |
| **X (Twitter)** | `TWITTER_CLIENT_ID` | `TWITTER_CLIENT_SECRET` | `{BASE_URL}/api/auth/twitter/callback` (PKCE + Basic Auth) |

---

## 5. LLM Moderation & Payment Gateways

### LLM Review (`site.json::moderation.llm_review = true`)

- **OpenAI-compatible APIs (OpenAI / DeepSeek / Qwen / Ollama)**: `OPENAI_LLM_API_KEY`, `OPENAI_LLM_BASE_URL`, `OPENAI_LLM_MODEL`
- **Anthropic API**: `ANTHROPIC_LLM_API_KEY`, `ANTHROPIC_LLM_BASE_URL`, `ANTHROPIC_LLM_MODEL`

### Payment Gateways (`payments` Tier)

Shared webhook base `PAY_NOTIFY_BASE=https://your-domain.com`:

- **Alipay**: `ALIPAY_APP_ID`, `ALIPAY_APP_PRIVATE_KEY`, `ALIPAY_PUBLIC_KEY`, `ALIPAY_GATEWAY` (`{PAY_NOTIFY_BASE}/api/pay/alipay/notify`)
- **WeChat Pay v3**: `WECHAT_MCHID`, `WECHAT_APP_ID`, `WECHAT_API_V3_KEY`, `WECHAT_MCH_PRIVATE_KEY`, `WECHAT_MCH_SERIAL_NO`, `WECHAT_PLATFORM_PUBLIC_KEY` (`{PAY_NOTIFY_BASE}/api/pay/wechat/notify`)
- **Stripe**: `STRIPE_SECRET_KEY`, `STRIPE_WEBHOOK_SECRET` (`{PAY_NOTIFY_BASE}/api/pay/stripe/notify`)
- **Background Reconciliation**: `PAY_RECONCILE_INTERVAL_SECS` (default `300`, min `60`), `PAY_RECONCILE_MIN_AGE_SECS` (`300`), `PAY_RECONCILE_CLOSE_AFTER_SECS` (`7200`)

---

## 6. Pingora Multi-Domain SNI Gateway (`crates/gateway`)

Listens on `:80` (301 redirect to HTTPS) and `:443`, hosting multiple domains and upstream containers on a single machine:

| Variable | Example | Purpose |
| :--- | :--- | :--- |
| `TLS_CERT_PATH` / `TLS_KEY_PATH` | `/certs/default/fullchain.pem` | Fallback TLS certificate chain and private key |
| `SNI_CERTS` | `dioxuspress.dev=/certs/dp/fullchain.pem:/certs/dp/privkey.pem` | Comma/semicolon-separated per-domain SNI certificates |
| `UPSTREAM_ADDR` | `127.0.0.1:8080` | Default upstream address |
| `HOST_UPSTREAMS` | `dioxuspress.dev=127.0.0.1:8081,rustineverything.app=127.0.0.1:8080` | Routes requests by `Host` header to separate local DioxusPress containers |
| `HOST_REDIRECTS` | `www.dioxuspress.dev=https://dioxuspress.dev` | Domain-level `301 Moved Permanently` redirects preserving path and query string |
