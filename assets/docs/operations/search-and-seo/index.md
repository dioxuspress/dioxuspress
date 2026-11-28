---
title: Tantivy 全文检索与全站 SEO 工程规范：Sitemap、JSON-LD 与 GSC 指南
sidebar_label: 全文检索与 SEO
sidebar_position: 3
description: 深入了解 DioxusPress 的嵌入式 Tantivy 增量索引机制、零 JS 爬虫友好 SSR 元信息与 Google Search Console 实践。
keywords: [Tantivy 全文搜索, SEO 优化, sitemap.xml, JSON-LD 结构化数据, Google Search Console]
---

# Tantivy 全文检索与全站 SEO

DioxusPress 将「站内秒级全文搜索」与「站外搜索引擎高权重收录」作为核心基础能力内建在框架底层，无需接入 Algolia 等外部收费服务。

## 1. 嵌入式 Tantivy 全站搜索（`⌘K`）

- **纯 Rust 零外部进程**：`module-search` 直接在应用进程内集成 Tantivy 搜索引擎，支持中英文混合分词、标题高权重加权、按内容类型过滤（文档 / 博客 / 案例 / 论坛）以及关键词上下文高亮摘要。
- **基于 `manifest.json` 的增量刷新**：应用启动时扫描内容文件的 `mtime` 并对比 `/app/data/search-index/manifest.json` 清单，仅对新增或修改过的文件执行 `upsert`，已删除文件自动清理。即便拥有 **2,500+ 篇长文档**，日常重启增量校验也仅需毫秒级，全量重建仅需约 6 秒。
- **版本化 Schema 自动迁移**：当索引标题生成策略升级时，框架通过递增 `IndexManifest::CURRENT_VERSION` 自动触发一次性存量重写入，无需人工手动删盘重建。

## 2. 搜索引擎优化（SEO）全链路保障

1. **SSR 首屏完整 `<head>` 输出**：每张页面在服务端渲染阶段即输出准确的 `<title>`、`<meta name="description">`、`<link rel="canonical">`、`hreflang` 多语言互指链接以及完整的 Open Graph / Twitter 大图卡片标签。
2. **Schema.org JSON-LD 结构化数据**：
   - 首页自动注入 `WebSite` 结构；
   - 文档页自动注入 `TechArticle`（配置了 `source` 时额外携带关联 Rust Crate 的 `SoftwareSourceCode` 节点）；
   - 博客文章自动注入 `BlogPosting`（含发布时间与作者信息）。
3. **动态 `/sitemap.xml` 与尾部斜杠规范化**：
   - `/sitemap.xml` 严格跟随已编译且启用的模块动态生成，绝不输出已关闭板块的 404 链接，并附带 `1 小时` 公共缓存头减轻爬虫抓取压力；
   - 全局 `trailing_slash_mw` 中间件将 `/docs/guide/` 永久重定向（`308`）至规范路径 `/docs/guide`，杜绝重复URL权重分散。

## 3. 上线后接入 Google Search Console 建议

新域名或二级域名部署上线后：
1. 前往 Google Search Console 使用 **网域（Domain）资源** TXT 记录完成所有权验证；
2. 在「站点地图」菜单提交 `https://<你的域名>/sitemap.xml`；
3. 利用文档模块 `sidebar_label`（短侧栏名）+ `title`（含具体版本号、报错关键词或核心概念的长尾标题）双轨机制，持续承接精准开发者搜索流量。
