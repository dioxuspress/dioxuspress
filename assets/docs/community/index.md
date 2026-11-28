---
title: 社区互动、内容治理与商业化闭环总览
sidebar_label: 账号与社区治理
sidebar_position: 3
description: 了解 DioxusPress 的 OAuth2 认证、论坛话题、段落划线标注、LLM 自动内容审核与在线支付体系。
keywords: [OAuth2 登录, Rust 论坛系统, 段落标注, LLM 内容审核, 在线支付]
---

# 社区互动与治理总览

当你的文档站或技术博客积累起稳定读者后，无需迁移技术栈或外挂第三方评论插件，只需启用 `accounts` / `course` / `payments` 特性并连接 PostgreSQL，即可在站内原生开启完整的用户互动与商业化闭环。

## 核心设计原则

1. **安全基线内建，而非事后打补丁**：
   - OAuth2 全流程强制启用 **PKCE + State 双重校验**；
   - JWT 会话采用 `HttpOnly + SameSite=Lax + Secure` Cookie 承载，登出接口强制校验 `POST` 与同源请求头（防御跨站登出 CSRF）；
   - 全站默认下发无 `'unsafe-eval'` 的严格 CSP 头，API 路由内置按客户端 IP 的令牌桶限流。
2. **讨论与内容深度绑定**：论坛话题不仅能独立发帖，还能直接关联到任意一篇文档或博客底部（`DiscussionPanel`），把分散的读者提问沉淀为可检索的结构化问答。
3. **默认零信任的内容审核**：内置域名黑名单拦截与可选 OpenAI / Anthropic 兼容 LLM 自动审核流水线；当 LLM 服务超时或异常时，强制执行「转人工复核或拒绝（Fail-Safe）」，绝不因审核接口故障而裸奔放行垃圾外链。

## 本卷章节导航

- **[OAuth2 登录与会话安全](/docs/community/auth)**
- **[论坛话题与文章评论](/docs/community/forum-and-comments)**
- **[段落划线高亮与笔记标注](/docs/community/annotations)**
- **[LLM 内容审核与管理后台](/docs/community/moderation-and-admin)**
- **[微信 / 支付宝支付与打赏](/docs/community/payments)**
