---
title: 全模块 Frontmatter 与元数据字典：文档、博客、课程、案例与播客字段总表
sidebar_label: Frontmatter 字段字典
sidebar_position: 6
description: 汇总 DioxusPress 五大内容模块（docs、blog、course、cases、podcast）的 YAML Frontmatter 与目录元数据字段规范。
keywords: [Frontmatter, DocMeta, authors.yml, course.yaml, case.yaml, episode.yaml]
---

# 全模块 Frontmatter 与元数据字典

DioxusPress 所有文件型内容模块均采用标准 YAML 元数据头（`--- ... ---`）或目录级 `.yaml` 描述文件。本页汇总五大模块的全部有效字段与默认值，方便写作时随时查阅。

---

## 1. 文档模块（`assets/docs/**/index.{md,mdx}`）

对应服务端结构体 `module_docs::server::DocMeta`：

```yaml
---
title: Tokio 调度器源码剖析：Work-Stealing 任务队列与协作式让出
sidebar_label: 调度器源码剖析
sidebar_position: 4
description: 深入解析 Tokio 多线程运行时的本地队列、全局注入队列与饥饿防护机制。
keywords: [Tokio, Work-Stealing, 异步调度器, Rust 并发]
image: /images/og-tokio.jpg
sort_children: asc
source:
  crate: tokio
  version: "1.49.0"
---
```

| 字段名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `title` | `string` | 目录名 | 页面完整标题（用于 `<h1>`、`<title>`、`og:title` 与搜索索引；子章节未含卷名时服务端自动追加 `· <卷名>`） |
| `sidebar_label` | `string` | 回退 `title` | 左侧文档树与底部「上一篇 / 下一篇」导航卡片显示的紧凑短标题 |
| `sidebar_position` | `integer` | 未设排在显式编号之后 | 同级目录内的排序权重（默认越小越靠前；同权重按 slug 字典序稳定排序） |
| `description` | `string` | `""` | 页面 SEO `<meta name="description">` 与分卷卡片摘要 |
| `keywords` | `string[]` | `[]` | 注入 `<meta name="keywords">` 与 JSON-LD `TechArticle` |
| `image` | `string` | 回退 `site.json::og_image` | 单篇文档专属社交分享大图路径 |
| `sort_children` | `"asc" \| "desc"` | `"asc"` | 在父目录 `index.md` 中设置，控制其直接子章节的排列方向（周刊 / 更新日志设为 `"desc"` 可让最新一期置顶） |
| `source` | `{ crate, version }` | `null` | 标注本文所基于的 Rust Crate 及版本号，正文顶部自动渲染 `docs.rs` 直达胶囊 |

---

## 2. 博客模块（`assets/posts/<slug>/index.{md,mdx}` 与 `authors.yml`）

### 文章 Frontmatter

```yaml
---
title: "为什么我们用 Dioxus 0.7 构建全栈知识引擎"
description: "对比静态站点生成器与同构 Rust SSR + WASM 架构在首屏延迟与交互演进上的取舍。"
date: "2026-11-12"
updated: "2026-11-15"
authors: [yuxuetr]
tags: [Rust, Dioxus, Fullstack]
image: cover.webp
---
```

| 字段名 | 类型 | 必填建议 | 说明 |
| :--- | :--- | :--- | :--- |
| `title` | `string` | **必填** | 文章标题 |
| `description` | `string` | 推荐 | 列表页摘要、RSS `<description>` 与 SEO 描述 |
| `date` | `string` | **必填** | 发布日期（`YYYY-MM-DD`），列表页与 `/feed.xml` 按此倒序排列 |
| `updated` | `string` | 可选 | 最后修订日期（`YYYY-MM-DD`），优先作为 `/sitemap.xml` 的 `<lastmod>` |
| `authors` | `string \| string[]` | 推荐 | 关联 `assets/posts/authors.yml` 中的作者键名（支持单字符串或数组合著） |
| `tags` | `string[]` | 推荐 | 文章分类标签，自动聚合至 `/blog` 顶部标签筛选器 |
| `image` | `string` | 可选 | 社交分享图（支持文章目录内相对文件名如 `cover.webp` 或 `/images/...`） |

### 全局作者表（`assets/posts/authors.yml`）

```yaml
yuxuetr:
  name: yuxuetr
  title: Core Maintainer
  url: https://github.com/yuxuetr
  image_url: /images/logo-mobile.png
```

---

## 3. 实战课程模块（`assets/courses/<course_slug>/`）

采用三级文件配置：**课程级 `course.yaml` → 章节级 `_chapter.yaml` → 课节级 `index.{md,mdx}`**。

### 3.1 课程配置 `course.yaml`

```yaml
title: Rust 异步运行时实战
description: 从零手写 Future 执行器到高并发生产调优。
cover: /images/courses/async-rust.webp
order: 1
level: 进阶
tags: [rust, async, tokio]
access_tier: paid
price: 9900
currency: CNY
price_usd: 1499
```

| 字段名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `title` / `description` | `string` | 课程标题与简介（进入公开搜索索引） |
| `cover` | `string` | 课程封面图路径 |
| `order` | `integer` | `/course` 列表页排序权重（越小越靠前） |
| `level` | `string` | 难度标签（如 `"入门"`、`"进阶"`、`"专家"`） |
| `tags` | `string[]` | 技术标签列表 |
| `access_tier` | `"free" \| "paid" \| "pro"` | 权益门槛：`free` 全免费；`paid` 需单课购买或 Pro 会员；`pro` 仅限 Pro 会员 |
| `price` / `currency` | `integer` / `string` | 人民币定价，**单位为「分」**（如 `9900` 表示 `¥99.00`），默认 `"CNY"` |
| `price_usd` | `integer` | 可选海外 Stripe 美元定价，**单位为「美分」**（如 `1499` 表示 `$14.99`） |

### 3.2 章节配置 `<chapter_dir>/_chapter.yaml`

```yaml
title: 第一章：Future 状态机与 Waker 唤醒机制
description: 理解编译器如何将 async fn 降级为零分配状态机。
order: 1
```

### 3.3 课节 Frontmatter `<chapter_dir>/<lesson_dir>/index.{md,mdx}`

```yaml
---
title: 1.1 手写最小 Pin 与 Poll 执行器
description: 50 行代码跑通自定义异步任务唤醒闭环。
sidebar_position: 1
duration: "12:30"
preview: true
video_url: ""
audio_url: ""
---
```

- **`preview: true`**：将该课节标记为**免费试读**——即使所属课程为 `access_tier: paid`，未登录访客与搜索引擎也可完整阅读本节；未设或 `preview: false` 的付费课节由服务端严格鉴权拦截（连同同目录下的音视频与代码附件一并受保护）。

---

## 4. 案例展示库（`assets/cases/<slug>/case.yaml` + `README.md`）

每个案例目录包含一份结构化属性 `case.yaml` 与详细介绍正文 `README.md`：

```yaml
name: Dioxus Fullstack Template
description: Dioxus 0.7 全栈同构模板，涵盖 SSR、Server Function 与 Tailwind v4。
category: fullstack
tags: [dioxus, fullstack, wasm]
repo: https://github.com/dioxuspress/dioxuspress
website: https://dioxuspress.dev
author: DioxusPress Team
author_url: https://github.com/dioxuspress
language: Rust
stars: 1200
favorite: true
date_added: 2026-11-12
```

---

## 5. 播客模块（`assets/podcasts/<slug>/episode.yaml`）

```yaml
id: 1
title: 深入聊聊 Rust 全栈 Web 在生产环境的真实体感
description: 从编译耗时、WASM 体积优化到单机并发成本的实测复盘。
duration: "24:15"
date: 2026-11-12
audio_url: /audio/ep01-fullstack-rust.m4a
guest: yuxuetr
tags: [Rust, WebAssembly, 架构复盘]
```

- 在任意博客或文档的 `.mdx` 正文中，可通过 `<PodcastCard id="1" />` 按 `id` 直接嵌入该期节目的交互式音频播放卡片。
