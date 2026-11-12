---
title: Git 驱动的内容创作引擎：文档树、MDX 富排版、博客与实战课程
sidebar_label: 内容创作引擎
sidebar_position: 2
description: 全面掌握 DioxusPress 以文件系统为单一事实来源（Single Source of Truth）的多形态内容管线。
keywords: [Git 写作, MDX 渲染, 三级文档树, 博客系统, 课程系统]
---

# 内容创作引擎总览

在 DioxusPress 中，所有结构化知识资产均以 **Git 跟踪的纯文本 Markdown / MDX 文件**作为单一事实来源（Single Source of Truth），默认存放在 `assets/` 下，生产环境亦可通过环境变量 `CONTENT_DIR=/app/content` 只读挂载外部私有内容仓库，实现**「开源引擎代码」与「私有知识资产」彻底解耦、秒级热更新**。

## 内容形态一览

| 模块 ID | 默认目录 | 路由前缀 | 核心特性 |
| :--- | :--- | :--- | :--- |
| **`docs`** | `assets/docs/` | `/docs/*` | 三级侧栏树、`sidebar_position` 排序、右侧 TOC、面包屑、子章节 SEO 标题自动限定 |
| **`blog`** | `assets/posts/` | `/blog`、`/blog/:slug` | MDX 富组件、多作者署名（`authors.yml`）、标签过滤、自动 `/feed.xml` |
| **`course`** | `assets/courses/` | `/course/*` | 课程 → 章节 → 课节三级大纲、免费试读开关、会员与单课权益校验 |
| **`cases`** | `assets/cases/` | `/case`、`/case/:slug` | 精选项目橱窗、技术栈标签筛选、外链仓库与站点直达 |
| **`podcast`** | `assets/podcasts/` | `/podcast` | 在线音频播放器、单集时间轴与嘉宾介绍 |

## 本卷章节导航

- **[文档模块详解](/docs/content/docs-module)**：目录组织、Frontmatter 字段与长尾 SEO 机制。
- **[MDX 语法与内置组件](/docs/content/mdx)**：代码标签组、数学公式、Mermaid 图表、7 种提示框与视频嵌入。
- **[博客与作者系统](/docs/content/blog)**：发布文章、配置多作者头像与社交资料。
- **[实战课程模块](/docs/content/courses)**：编排免费与付费课节。
- **[案例展示与播客](/docs/content/cases-and-podcast)**：搭建开源项目展示墙与技术播客专栏。
