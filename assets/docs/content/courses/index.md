---
title: 实战课程模块：三级课纲编排、免费试读与会员权益门禁
sidebar_label: 实战课程模块
sidebar_position: 4
description: 使用 course 特性构建体系化专栏与付费课程，实现服务端严密鉴权的免费试读与单课/Pro 会员解锁。
keywords: [知识付费, 课程模块, 付费墙鉴权, Pro 会员, 课节编排]
---

# 实战课程模块

当以 `course` 或默认完整特性编译时，DioxusPress 提供完整的「**课程（Course）→ 章节（Chapter）→ 课节（Lesson）**」三级在线学习与付费解锁体验。

## 课程目录结构

```text
assets/courses/
└── rust-fullstack-in-action/
    ├── course.yaml             # 课程级元数据：标题、定价、难度、简介、讲师
    ├── 01-foundations/
    │   ├── chapter.yaml        # 章节标题与排序
    │   ├── 01-setup.mdx        # 免费试读课节（free: true）
    │   └── 02-ssr-hydration.mdx# 付费课节（free: false，服务端严格拦截未授权读取）
    └── 02-production/
        └── ...
```

## 服务端安全门禁（非前端遮罩）

与纯静态站点只能靠前端 CSS 隐藏正文不同，DioxusPress 的课节内容读取完全在 Axum 服务端校验用户会话与数据库权益表（`user_course_entitlements` / Pro 会员有效期）：

- **免费课节（`free: true`）**：未登录访客与搜索引擎爬虫均可完整阅读，承担自然流量获客入口；
- **付费课节（`free: false`）**：未解锁用户调用接口时，服务端仅返回课程大纲元信息与购买引导，**绝不下发受保护的 Markdown 原文或课程静态附件**。

管理员既可在 `/admin/entitlements` 后台手动为学员开通单课或 Pro 会员（`course` 档位），也可开启 `payments` 档位接入微信支付与支付宝实现自动回调发货。
