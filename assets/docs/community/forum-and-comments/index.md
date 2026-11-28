---
title: 论坛话题与文章评论：从独立讨论区到文档内嵌问答面板
sidebar_label: 论坛与评论系统
sidebar_position: 2
description: 使用 module-forum 与 module-comments 实现标签话题广场、Markdown 安全回帖与页面级关联讨论。
keywords: [论坛模块, 文章评论, 关联讨论, UGC 白名单清洗, Tantivy 话题检索]
---

# 论坛话题与文章评论

启用 `accounts` 档位后，站点同时获得轻量级 **文章评论（`module-comments`）** 与结构化 **社区论坛（`module-forum`）**。

## 两种互动形态的定位分工

| 维度 | 文章评论（`CommentBox`） | 社区论坛（`Topics` + `DiscussionPanel`） |
| :--- | :--- | :--- |
| **挂载位置** | 博客文章末尾 | 独立 `/topics` 广场 + 文档/案例底部「💬 关联讨论」面板 |
| **信息结构** | 围绕单篇文章的扁平即时留言 | 独立标题、标签分类、多轮楼层回复、独立永久 URL（`/topics/:id`） |
| **搜索收录** | 不单独进全站搜索结果页 | 审核通过的公开话题自动增量同步至 Tantivy 全站搜索 |

## 从文档页一键发起关联讨论

阅读任意文档章节或实战案例时，点击底部的「发起讨论」按钮跳转到 `/topics/new?kind=doc&path=...`，发布后该话题既会出现在 `/topics` 社区广场，也会自动聚合在对应文档页底部，帮助后续遇到相同问题的读者直接查阅解决方案。

## 严格的 UGC 渲染隔离

所有评论与论坛回帖均通过 `widgets::parse_untrusted_markdown` 渲染：
- 支持代码高亮、列表、粗斜体与安全外链；
- 原生 HTML 标签一律转义为纯文本，图片仅允许引用站内 `/uploads/` 受控上传目录；
- 所有一级标题 `#` 自动降级为 `##`，保持宿主页面标题层级语义完整。
