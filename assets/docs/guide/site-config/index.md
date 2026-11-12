---
title: site.json 全字段配置手册：品牌标识、公告条、首页特性与页脚导航
sidebar_label: site.json 配置手册
sidebar_position: 3
description: 逐项说明 assets/site.json 的全部配置字段，涵盖明暗双态 Logo、PWA 清单、类 Docusaurus 首页与模块开关。
keywords: [site.json, 站点配置, Logo 配置, 公告栏, 多列页脚]
---

# `site.json` 全字段配置手册

`assets/site.json` 是 DioxusPress 的非敏感站点级配置中心（敏感凭据如数据库连接串、JWT 密钥、OAuth 密钥一律通过 `.env` 环境变量注入，绝不写入 `site.json`）。

## 完整配置示例（类 Docusaurus 文档站模式）

```json
{
  "site_name": "DioxusPress",
  "site_description": "基于 Dioxus 0.7 的纯 Rust 全栈文档、内容与社区引擎",
  "author": "yuxuetr",
  "default_language": "zh",
  "theme": "ocean",
  "active_layout": "docs",
  "favicon": "/images/favicon.ico",
  "favicon_16": "/images/favicon-16.png",
  "favicon_32": "/images/favicon-32.png",
  "apple_touch_icon": "/images/apple-touch-icon.png",
  "manifest": "/images/site.webmanifest",
  "logo": "/images/logo-light.png",
  "logo_dark": "/images/logo-dark.png",
  "logo_mobile": "/images/logo-mobile.png",
  "logo_wordmark": true,
  "og_image": "/images/og-default.jpg",
  "docs_edit_url_base": "https://github.com/dioxuspress/dioxuspress/edit/main/assets/docs",
  "announcement": {
    "id": "v07-docs-layout",
    "text": "🎉 DioxusPress 现已内置类 Docusaurus 文档布局与零数据库部署模式",
    "text_en": "🎉 DioxusPress now ships with a Docusaurus-style docs layout & zero-DB mode",
    "href": "/docs/guide"
  },
  "hero": {
    "badge": "Dioxus 0.7 · Fullstack Rust",
    "badge_en": "Dioxus 0.7 · Fullstack Rust",
    "title": "用纯 Rust 构建极速文档与知识社区",
    "title_en": "Build Blazing-Fast Docs & Communities in Pure Rust",
    "subtitle": "SSR 首屏直出 + WASM 无刷新导航 + Tantivy 全文检索，单套代码从零数据库文档站平滑扩展至全栈付费社区。",
    "subtitle_en": "SSR + WASM hydration + Tantivy search. Scale from a zero-database docs site to a fullstack community.",
    "install_command": "git clone https://github.com/dioxuspress/dioxuspress.git",
    "primary_cta_text": "5 分钟快速开始",
    "primary_cta_text_en": "Get Started",
    "primary_cta_href": "/docs/guide/quickstart",
    "secondary_cta_text": "GitHub 源码",
    "secondary_cta_text_en": "GitHub",
    "secondary_cta_href": "https://github.com/dioxuspress/dioxuspress"
  }
}
```

## 关键字段速查表

| 字段路径 | 类型 | 默认值与行为说明 |
| :--- | :--- | :--- |
| `active_layout` | `string` | `"docs"`（类 Docusaurus 文档站）、`"classic"`（双生态综合门户）或 `"minimal"`（极简无页脚阅读条） |
| `theme` | `string` | 默认配色主题：`"ocean"`（青蓝）、`"sunset"`（暖橙）或 `"catppuccin"`（柔和紫）；访客可在顶栏自由切换 |
| `default_language` | `string` | `"zh"` 或 `"en"`；支持 `?lang=en` URL 参数与 Cookie 覆盖 |
| `logo` / `logo_dark` | `string` | 顶栏浅色 / 深色模式 Logo 路径；采用纯 CSS `dark:hidden` 双图切换，SSR 首屏零闪烁 |
| `logo_wordmark` | `bool` | 设为 `true` 表示横版 Logo 图片内已包含文字站名，顶栏不再重复拼接纯文本标题 |
| `docs_edit_url_base` | `string` | 非空时在每篇文档正文底部自动拼出「在 GitHub 上编辑此页」外链（自动区分 `index.md` 与 `index.mdx`） |
| `announcement` | `object` | 顶部可关闭横幅；访客点击关闭后按 `dp_ann_{id}` 记入 `localStorage`，更新 `id` 即可向全员展示新公告 |
| `features` | `array` | `active_layout = "docs"` 时首页展示的 3 列特性卡片，每项支持 `icon`（`bolt`/`book`/`layers`/`search`/`shield`/`palette`）、中英标题描述及可选跳转 `href` |
| `footer.columns` | `array` | 多列结构化页脚分组，支持混合配置站内路由（如 `"/docs/guide"`）与外部链接 |
