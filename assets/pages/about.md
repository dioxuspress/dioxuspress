---
title: 关于 DioxusPress
description: 基于 Dioxus 0.7 与 Axum 构建的开源全栈 Rust 文档、内容与社区引擎。
---

# 关于 DioxusPress

**DioxusPress** 是一个开源的纯 Rust 全栈知识出版与社区引擎，基于 [Dioxus 0.7](https://dioxuslabs.com)（服务端渲染 SSR + 客户端 WebAssembly 水合）与 [`dioxus-shadcn`](https://github.com/dioxuspress/dioxus-ui) 组件库构建。

## 我们的设计理念

1. **一套代码，弹性伸缩**：需要轻量技术文档站时，以 `--no-default-features`（`basic` 档位）编译即可获得零数据库依赖、内置 Tantivy 全文检索的类 Docusaurus 极速站点；需要用户体系与商业化闭环时，开启默认特性即可拥有 OAuth2 登录、论坛、评论、段落划线标注、LLM 内容审核与付费课程系统。
2. **文件即内容，Git 即工作流**：所有文档、博客、课程与案例均以标准 Markdown / MDX 存储在版本库中，支持外部私有内容仓库只读挂载热更新。
3. **默认工程安全与 SEO 友好**：严格无 `'unsafe-eval'` CSP 策略、SSR 首屏完整 Open Graph / JSON-LD 结构化数据输出、子章节标题自动卷名限定以及动态 Sitemap。

## 线上标杆案例

- **官方网站**：[dioxuspress.dev](https://dioxuspress.dev)
- **[Rust in Everything (`rustineverything.app`)](https://rustineverything.app)**：承载 38 卷 2,100+ 篇 Rust 生态源码解析文档、体系化 AI 实战课程、论文速递与技术社区，完全由 DioxusPress 驱动运行。
- **源码仓库**：[github.com/dioxuspress/dioxuspress](https://github.com/dioxuspress/dioxuspress)（MIT 开源协议）。
