---
title: 环境变量与多域名网关配置字典：.env 全参数、OAuth、支付与 Pingora SNI
sidebar_label: 环境变量与网关字典
sidebar_position: 5
description: DioxusPress 服务端环境变量（.env）与 Pingora TLS 网关完整参数参考，涵盖按档位必填项、限流阈值、LLM 审核及多站点 SNI 路由。
keywords: [环境变量, .env, BASE_URL, JWT_SECRET, OAuth PKCE, Pingora 网关, SNI_CERTS]
---

# 环境变量与多域名网关配置字典

DioxusPress 严格遵循十二要素应用（12-Factor App）规范：**站点外观与公开结构写入 [`site.json`](/docs/guide/site-config)，所有密钥、连接串与宿主拓扑通过环境变量（或仓库根目录 `.env`）注入**。

---

## 1. 核心运行与目录变量

| 变量名 | 生效档位 | 必填性 | 默认值 / 示例 | 作用说明 |
| :--- | :--- | :--- | :--- | :--- |
| `BASE_URL` | 全部档位 | `accounts` 必填；`basic` 可选 | `http://localhost:8080`（`basic` 默认） | 站点对外根 URL（含协议、无尾斜杠）。用于生成 `/sitemap.xml`、`/feed.xml` Canonical 地址与 OAuth 回调地址；`basic` 档位本地预览未设时自动回退至 `http://localhost:{PORT}`，生产环境建议显式配置公网域名 |
| `CONTENT_DIR` | 全部档位 | 可选 | 容器挂载非空 `/app/content` 时自动识别 | 外部内容目录（如 `/app/content`）。设置或挂载后，`site.json`、`docs/`、`posts/`、`courses/` 与优先覆盖的 `images/` 均从该目录读取 |
| `SEARCH_INDEX_DIR` | 全部档位 | 可选 | `./data/search-index` | Tantivy 全文索引与增量清单持久化目录；容器部署建议挂载独立 Volume 以加速重启校验 |
| `RUST_LOG` | 全部档位 | 可选 | `info` | `tracing-subscriber` 日志过滤表达式（如 `info,app_core=debug,sqlx=warn`） |

:::warning 本地开发 `localhost` 与 `127.0.0.1` 混用陷阱
配置了 OAuth 登录时，浏览器访问地址的 Host **必须与 `BASE_URL` 完全一致**。若 `BASE_URL=http://127.0.0.1:8080` 而你在浏览器打开 `http://localhost:8080`，OAuth 回调回跳到 `127.0.0.1` 时将无法读取 `localhost` 域下的 `oauth_pkce` Cookie，导致授权校验失败。
:::

---

## 2. 数据库、会话加密与安全门禁（`accounts` 及以上档位）

零数据库的 `basic` 档位（`--no-default-features`）完全不读取本节密钥变量；启用 `accounts` / `course` / 默认完整档位时生效：

| 变量名 | 必填性 | 默认值 / 生成命令 | 作用说明 |
| :--- | :--- | :--- | :--- |
| `JWT_SECRET` | **`accounts` 必填** | `openssl rand -hex 32` | 签发与校验用户会话 JWT 的主密钥（至少 32 字符，禁止保留 `.env.example` 占位符，否则启动直接 `panic` 拒绝裸奔） |
| `DATABASE_URL` | 推荐必填 | `postgres://rie:pass@127.0.0.1:5432/rie` | PostgreSQL 16 连接串；启动成功后自动执行 SeaORM 表结构迁移 |
| `DATA_ENCRYPTION_KEY` | 推荐配置 | `openssl rand -hex 32` | 独立 AES-256-GCM 数据加密密钥（加密 `oauth_pkce` Cookie 等），与 `JWT_SECRET` 解耦以便独立轮换；未设时回退由 `JWT_SECRET` 派生并打印告警 |
| `STRICT_MIGRATION` | 可选 | `0`（生产 Compose 默认 `1`） | 设为 `1` 时，若数据库连接或 Schema 迁移失败则直接终止进程退出，防止带着残缺表结构接收写请求 |

---

## 3. 应用层限流与安全响应头（全部档位可选）

| 变量名 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `RATE_LIMIT_API_RPS` / `RATE_LIMIT_API_BURST` | `20` / `60` | 常规 `/api/*` Server Function 每 IP 令牌桶速率与突发上限（超限返回 `429 Too Many Requests`） |
| `RATE_LIMIT_SENSITIVE_RPS` / `RATE_LIMIT_SENSITIVE_BURST` | `5` / `15` | 敏感端点（`/api/auth/*`、`/api/pay/*`）更严格的独立令牌桶配额 |
| `CSP_POLICY` | 内置无 `'unsafe-eval'` 策略 | 自定义覆盖 `Content-Security-Policy` 响应头（为空时使用内置严格策略） |
| `RATE_LIMIT_DISABLED` / `SECURITY_HEADERS_DISABLED` | 未设置 | 仅限本地排障临时设为 `1`，**生产环境严禁开启** |

---

## 4. OAuth 第三方登录凭据（`accounts` 档位可选）

在 `site.json` 的 `auth.providers` 中声明顺序后，按需填写对应平台的凭据（未填凭据的平台会自动在登录弹窗中隐藏）：

| 登录提供方 | Client ID 变量 | Client Secret 变量 | OAuth 回调 URL（填至开发者后台） |
| :--- | :--- | :--- | :--- |
| **GitHub** | `GITHUB_CLIENT_ID` | `GITHUB_CLIENT_SECRET` | `{BASE_URL}/api/auth/github/callback` |
| **Google** | `GOOGLE_CLIENT_ID` | `GOOGLE_CLIENT_SECRET` | `{BASE_URL}/api/auth/google/callback` |
| **Discord** | `DISCORD_CLIENT_ID` | `DISCORD_CLIENT_SECRET` | `{BASE_URL}/api/auth/discord/callback` |
| **X (Twitter)** | `TWITTER_CLIENT_ID` | `TWITTER_CLIENT_SECRET` | `{BASE_URL}/api/auth/twitter/callback`（自动启用 PKCE + Basic Auth） |

---

## 5. LLM 自动内容风控与在线支付网关

### LLM 接口配置（配合 `site.json::moderation.llm_review = true`）

- **OpenAI 兼容协议（支持 OpenAI / DeepSeek / Qwen / Moonshot / Ollama）**：
  - `OPENAI_LLM_API_KEY`、`OPENAI_LLM_BASE_URL`（如 `https://api.deepseek.com/v1`）、`OPENAI_LLM_MODEL`（如 `deepseek-chat`）
- **Anthropic 协议**：
  - `ANTHROPIC_LLM_API_KEY`、`ANTHROPIC_LLM_BASE_URL`、`ANTHROPIC_LLM_MODEL`

### 在线支付网关（需 `payments` 档位）

共用回调根域名变量 `PAY_NOTIFY_BASE=https://your-domain.com`，三套通道可按需独立启用任意组合：

- **支付宝当面付 / 电脑网站支付**：`ALIPAY_APP_ID`、`ALIPAY_APP_PRIVATE_KEY`、`ALIPAY_PUBLIC_KEY`、`ALIPAY_GATEWAY`（回调：`{PAY_NOTIFY_BASE}/api/pay/alipay/notify`）
- **微信支付 v3（公钥模式）**：`WECHAT_MCHID`、`WECHAT_APP_ID`、`WECHAT_API_V3_KEY`（32 字节）、`WECHAT_MCH_PRIVATE_KEY`、`WECHAT_MCH_SERIAL_NO`、`WECHAT_PLATFORM_PUBLIC_KEY`（回调：`{PAY_NOTIFY_BASE}/api/pay/wechat/notify`）
- **Stripe 国际支付**：`STRIPE_SECRET_KEY`、`STRIPE_WEBHOOK_SECRET`（回调：`{PAY_NOTIFY_BASE}/api/pay/stripe/notify`）
- **掉单自动对账任务**：`PAY_RECONCILE_INTERVAL_SECS`（默认 `300`，下限 `60`）、`PAY_RECONCILE_MIN_AGE_SECS`（默认 `300`）、`PAY_RECONCILE_CLOSE_AFTER_SECS`（默认 `7200`）

---

## 6. Pingora 多站点 SNI 网关（`crates/gateway`）

生产环境使用内置 Pingora 网关监听 `:80`（自动 301 跳转 HTTPS）与 `:443`，支持在**单台宿主机同端口承载多个独立域名与后端容器**：

| 变量名 | 示例格式 | 说明 |
| :--- | :--- | :--- |
| `TLS_CERT_PATH` / `TLS_KEY_PATH` | `/certs/default/fullchain.pem` | 默认兜底 TLS 证书链与私钥路径 |
| `SNI_CERTS` | `dioxuspress.dev=/certs/dp/fullchain.pem:/certs/dp/privkey.pem` | 多域名 SNI 证书映射（多项以英文逗号 `,` 或分号 `;` 分隔，自动按客户端握手域名选择对应证书） |
| `UPSTREAM_ADDR` | `127.0.0.1:8080` | 默认上游应用地址 |
| `HOST_UPSTREAMS` | `dioxuspress.dev=127.0.0.1:8081,rustineverything.app=127.0.0.1:8080` | 按请求 `Host` 头将流量分发至本机不同端口的 DioxusPress 实例 |
| `HOST_REDIRECTS` | `www.dioxuspress.dev=https://dioxuspress.dev` | 域名级 `301 Moved Permanently` 规范化重定向（自动保留原始 Path 与 Query） |
