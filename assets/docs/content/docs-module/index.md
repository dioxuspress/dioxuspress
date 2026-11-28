---
title: 文档模块规范：三级目录树、侧栏排序、面包屑与 SEO 标题自动优化
sidebar_label: 文档模块规范
sidebar_position: 1
description: 详解 assets/docs 三级目录结构、Frontmatter 字段、sort_children 倒序周刊模式以及跨卷 SEO 标题防重机制。
keywords: [文档目录树, sidebar_position, sidebar_label, 面包屑导航, 文档 SEO]
---

# 文档模块规范

`module-docs` 提供开箱即用的类 Docusaurus 三级文档系统，兼顾大型多卷技术手册的侧栏紧凑性与搜索引擎长尾词友好度。

## 目录结构约定

```text
assets/docs/
├── guide/                      # 一级分卷：/docs/guide
│   ├── index.md                # 分卷首页（可选；若无则仅作侧栏折叠分组）
│   ├── quickstart/
│   │   └── index.md            # 二级章节：/docs/guide/quickstart
│   └── advanced/
│       └── ssr/
│           └── index.md        # 三级小节：/docs/guide/advanced/ssr
└── changelog/
    ├── index.md                # 设置 sort_children: desc 让最新期排最前
    ├── v0-7-0/index.md
    └── v0-7-1/index.md
```

- 以 `_` 或 `.` 开头的目录（如 `_drafts`）在构建目录树与搜索索引时会自动忽略。
- 同时支持 `index.md` 与 `index.mdx`（优先匹配 `.mdx`）。

## Frontmatter 元数据字段

```yaml
---
title: Axum 常见坑与报错排查：Handler 不满足、路由冲突与中间件顺序
sidebar_label: 常见坑
sidebar_position: 10
description: 汇总 Axum 开发中最常遇到的编译期 trait 报错与运行期请求体消费陷阱。
keywords: [axum, FromRequest, Handler trait, 常见报错]
sort_children: asc
source:
  crate: axum
  version: "0.8.4"
---
```

### 几个实用设计细节

1. **`title` 与 `sidebar_label` 读写分离**：左侧目录树与上下篇卡片优先显示简短的 `sidebar_label`（如 `常见坑`），而页面 HTML `<title>`、`og:title` 与 JSON-LD `TechArticle.headline` 使用信息量完整的 `title`。
2. **子章节自动补齐所属卷名**：若子章节只写了简短标题（如 `title: 路由与提取器`），服务端在输出 SEO 元信息与写入 Tantivy 搜索索引时会自动补齐为 `路由与提取器 · Axum`，彻底消除多卷同名章节在 Google 收录与 `⌘K` 搜索中的重复与歧义。
3. **`source` 版本徽章**：配置 `source.crate` 与 `source.version` 后，正文顶部会自动渲染指向 `https://docs.rs/<crate>/<version>` 的版本胶囊，并在 Schema.org 结构化数据中注入 `SoftwareSourceCode` 关联。
4. **安全面包屑与编辑链接**：详情页顶部自动生成 `文档 / 分卷 / 章节` 面包屑（纯分组无正文目录自动降级为纯文本，绝不产生 404 死链）；配置 `site.json::docs_edit_url_base` 后底部自动附带 GitHub 编辑入口。
