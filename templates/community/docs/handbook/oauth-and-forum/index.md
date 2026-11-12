---
title: OAuth 登录、论坛与段落标注配置指南
sidebar_label: OAuth 与论坛配置
sidebar_position: 1
description: 配置 OAuth2 PKCE 回调地址、开启文档/博客划线批注并设置首位社区管理员。
keywords: [OAuth2, PKCE, 论坛, 段落标注, Moderation]
---

# OAuth 登录、论坛与段落标注配置指南

`community` 预设基于 `dioxuspress-app` + `postgres:16-alpine` 双容器编排，启动时会自动执行全部 SeaORM 数据库迁移。

## 1. 配置 OAuth2 第三方登录

复制 `.env.example` 为 `.env`，并在对应开发者后台注册回调地址：

| 登录渠道 | 环境变量 | 授权回调地址（Callback URL） |
| :--- | :--- | :--- |
| GitHub | `GITHUB_CLIENT_ID` / `GITHUB_CLIENT_SECRET` | `{BASE_URL}/api/auth/github/callback` |
| Google | `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` | `{BASE_URL}/api/auth/google/callback` |
| Discord | `DISCORD_CLIENT_ID` / `DISCORD_CLIENT_SECRET` | `{BASE_URL}/api/auth/discord/callback` |

:::warning 避免混用 `127.0.0.1` 与 `localhost`
OAuth2 PKCE 状态存储在 `SameSite=Lax` Cookie 中。若 `.env` 配置 `BASE_URL=http://127.0.0.1:8080`，本地浏览器也请统一通过 `http://127.0.0.1:8080` 访问，否则回调时浏览器不会携带 PKCE Cookie。
:::

## 2. 段落划线标注与话题讨论

本站已在 `site.json` 中开启：

- **段落划线标注（`annotations`）**：登录用户在任意文档或博客正文中选中文字，即可添加个人笔记或公开批注（在 `/me/annotations` 集中管理）。
- **话题讨论区（`/topics`）**：支持按标签筛选、Markdown 发帖与文档底部关联讨论面板。
- **反垃圾链接拦截（`moderation`）**：内置域名黑名单同步拦截恶意广告链接；如需接入大模型自动风控，可在 `.env` 中配置 `OPENAI_LLM_API_KEY` 并将 `moderation.llm_review` 设为 `true`。

## 3. 提升首位管理员账号

完成首次 OAuth 登录后，在宿主机执行以下命令将你的账号设为 `admin`（解锁 `/admin` 审核与治理后台）：

```bash
docker compose exec postgres psql -U dioxuspress -d dioxuspress \
  -c "UPDATE users SET role = 'admin' WHERE id = 1;"
```
