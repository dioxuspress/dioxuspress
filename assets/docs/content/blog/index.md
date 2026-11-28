---
title: 博客模块指南：目录规范、authors.yml 多作者署名与 RSS 订阅
sidebar_label: 博客与多作者
sidebar_position: 3
description: 在 assets/posts 下组织独立文章目录、随文图片资源、多作者信息卡片与全站 RSS 2.0 订阅输出。
keywords: [博客模块, authors.yml, RSS feed, 标签筛选, MDX 博客]
---

# 博客与多作者系统

每篇博客文章在 `assets/posts/<slug>/` 下拥有独立子目录，方便将正文 `index.mdx`（或 `index.md`）与配图放在一起就近管理。

## 目录组织结构

```text
assets/posts/
├── authors.yml                 # 全局作者名录（集中维护头像、头衔与主页链接）
├── welcome/
│   ├── index.mdx               # 访问路径：/blog/welcome
│   └── cover.webp              # 文章内直接写 ![封面](cover.webp) 即可相对路径引用
└── release-0-7/
    └── index.md
```

## `authors.yml` 配置格式

```yaml
yuxuetr:
  name: yuxuetr
  title: DioxusPress Creator
  url: https://github.com/yuxuetr
  image_url: /images/logo-mobile.png
```

在文章 Frontmatter 中只需引用作者键名（支持单作者字符串或合著数组）：

```yaml
---
title: "发布 DioxusPress：用纯 Rust 重构内容与社区平台"
description: "分享我们为什么从静态生成器转向 Dioxus 0.7 全栈 SSR + WASM 架构。"
date: "2026-11-12"
authors: [yuxuetr]
tags: [Rust, Dioxus, Architecture]
---
```

## 自动生成的聚合特性

- **按标签即时过滤**：`/blog` 列表页自动提取全部文章标签并按频次排序，支持展开/收起低频标签与分页浏览；
- **RSS 订阅源**：服务端自动在 `/feed.xml` 生成符合规范的 RSS 2.0 XML（遵循模块开关，关闭博客时自动停止暴露）；
- **关联社区互动**：当启用了 `accounts` 档位时，每篇博客底部会自动挂载评论区（`CommentBox`）与「围绕本文发起论坛讨论」面板。
