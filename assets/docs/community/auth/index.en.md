---
title: "OAuth2 + PKCE Authentication: GitHub, Google, Discord & X (Twitter)"
sidebar_label: OAuth2 Sign-In
sidebar_position: 1
description: Configure site.json::auth and environment secrets for passwordless OAuth2 PKCE login, JWT sessions, and promoting your first admin account.
keywords: [OAuth2 PKCE, GitHub login, Google login, JWT session, admin promotion]
---

# OAuth2 Sign-In & Session Configuration

DioxusPress natively supports **GitHub**, **Google**, **Discord**, and **X (Twitter)** OAuth2 with PKCE — eliminating password database liabilities and email verification boilerplate.

## 1. Declare Provider Order in `site.json`

```json
{
  "auth": {
    "enabled": true,
    "providers": ["github", "google", "discord", "twitter"]
  }
}
```

Any provider whose client credentials are left unset in `.env` is automatically hidden from the sign-in dialog.

## 2. Inject Credentials via `.env`

For a GitHub OAuth App, register `<BASE_URL>/api/auth/github/callback` as the authorization callback URL:

```dotenv
BASE_URL=https://your-domain.com
JWT_SECRET=<generate a >= 32-char secret with openssl rand -hex 32>

GITHUB_CLIENT_ID=Iv1.xxxxxxxxxxxxxxxx
GITHUB_CLIENT_SECRET=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Google, Discord, and X follow the same pattern via `GOOGLE_CLIENT_*`, `DISCORD_CLIENT_*`, and `TWITTER_CLIENT_*` (see `.env.example` and the [Environment Reference](/docs/operations/env-reference)).

## 3. Promoting the First Administrator

First-time OAuth users default to the `member` role. Sign in once, then promote your account using the helper script:

```bash
set -a; source .env; set +a
./scripts/promote_admin.sh <your_username>
```

Sign out and sign back in once so your refreshed JWT carries the `admin` role, revealing the **Admin Console (`/admin`)** entry in your user menu.
