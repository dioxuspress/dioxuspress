---
title: 内容资产与付费课程管线详解
sidebar_label: 内容与课程管线
sidebar_position: 1
description: 配置 course.yaml 定价、试读课节鉴权、案例卡片、播客嵌入与 LLM 自动风控。
keywords: [实战课程, 案例库, 播客, LLM 审核, Stripe]
---

# 内容资产与付费课程管线详解

`portal` 预设展示了 DioxusPress 五大内容模块的完整协作结构：

```text
.
├── courses/fullstack-rust/      # 课程元数据 + 章节 _chapter.yaml + 试读/付费课节
├── cases/dioxuspress-starter/   # 案例库卡片 case.yaml + 详细介绍 README.md
├── podcasts/ep01-architecture/  # 播客节目元数据 episode.yaml（支持 MDX 内嵌）
├── docs/architecture/           # 结构化技术文档卷
└── posts/                       # 支持 .mdx 交互组件的专栏文章
```

## 1. 试读课节与服务端严密鉴权

在 `courses/fullstack-rust/` 中：

- [1.1 同构 SSR 与客户端水合原理](/course/fullstack-rust/01-foundations/01-ssr-hydration) 标记了 `preview: true`，未登录访客与搜索引擎均可完整阅读；
- `02-production-ops` 标记了 `preview: false`，未解锁权益时服务端不会下发任何受保护正文或附件字节。
- 管理员可在 `/admin/entitlements` 手动授予学员课程权益，或在 `.env` 中配置微信 Native 支付 / Stripe Checkout 解锁全自动购买闭环。

## 2. 开启大模型 UGC 自动审核

当社区评论、论坛与划线批注流量增长后，只需在 `.env` 填入兼容 OpenAI 协议的 `OPENAI_LLM_API_KEY`，并将 `site.json` 中的 `"moderation.llm_review"` 设为 `true`，即可对高风险提交自动拦截或转入 `/admin/moderation` 人工复核队列。
