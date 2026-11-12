---
title: site.json 全字段配置参考手册：JSON Schema 补全、品牌标识、布局与模块开关
sidebar_label: site.json 配置手册
sidebar_position: 3
description: DioxusPress assets/site.json 权威字段字典，涵盖 IDE JSON Schema 自动补全、启动期拼写体检、明暗双态 Logo、Hero、多列页脚及审核配置。
keywords: [site.json, JSON Schema, 站点配置, Logo 配置, 公告栏, 多列页脚, 模块开关]
---

# `site.json` 全字段配置参考手册

`assets/site.json`（或双仓分离模式下的 `$CONTENT_DIR/site.json`）是 DioxusPress 的**非敏感站点级配置中心**。所有敏感凭据（数据库连接串、JWT 密钥、OAuth 密钥、支付私钥与 LLM API Key）一律通过环境变量注入，参见 [环境变量与网关配置字典](/docs/operations/env-reference)。

---

## 1. 开启 IDE 自动补全与启动期配置体检

### JSON Schema 实时补全与校验

DioxusPress 官方托管了与 `SiteConfig` Rust 结构体严格同步的 JSON Schema。只需在 `site.json` 第一行声明 `"$schema"`，VS Code、Zed、RustRover 等编辑器即可自动提供**中文字段悬停提示、枚举值下拉补全与类型错误红线告警**：

```json
{
  "$schema": "https://dioxuspress.dev/schemas/site.schema.json",
  "site_name": "My Docs",
  "active_layout": "docs",
  "theme": "ocean"
}
```

:::tip 离线环境与二次开发同步
该文件同时随仓库内置于 `/schemas/site.schema.json`（对应源码 `assets/schemas/site.schema.json`）。若你在二次开发中扩展了 `crates/core/src/settings.rs` 的 `SiteConfig` 字段，运行以下命令即可自动重新生成 Schema 并通过一致性守卫单测：

```bash
UPDATE_SCHEMA=1 cargo test --features server -p app-core site_schema_json_is_up_to_date
```
:::

### 启动期非阻塞配置体检（Startup Diagnostics）

为了防止手写 JSON 时因字段名拼写错误（例如误将 `active_layout` 写成 `layout`，或将 `hero.title` 写成 `hero.titel`）被默认值静默吞没，服务端启动时会自动运行 `diagnose_site_json` 体检并以 `WARN` 日志提示，同时不阻塞正常启动：

- **未知字段检测**：递归检查根对象及 `announcement`、`hero`、`features[*]`、`footer`、`annotations`、`auth`、`modules.*`、`moderation` 等全部嵌套块；
- **枚举合法性检查**：校验 `active_layout`、`default_language`、`theme` 与 `auth.providers`；
- **静态资源存在性校验**：检查 `logo`、`favicon`、`og_image` 等 `/images/*` 引用在 `assets/images` 或 `$CONTENT_DIR/images` 下是否真实存在。

---

## 2. 基础元信息、布局与主题

| 字段名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `$schema` | `string` | — | JSON Schema 地址，推荐设为 `"https://dioxuspress.dev/schemas/site.schema.json"` |
| `site_name` | `string` | `"DioxusPress"` | 站点名称；用于 HTML `<title>` 后缀、未配图片 Logo 时的顶栏标题及默认版权名 |
| `site_description` | `string` | `""` | 站点简介；注入首页默认 `<meta name="description">` 与 RSS `<description>` |
| `author` | `string` | `""` | 默认站点作者署名 |
| `default_language` | `"zh" \| "en"` | `"zh"` | 默认界面语言；优先级低于 URL `?lang=` 参数与访客 `site_lang` Cookie |
| `active_layout` | `"classic" \| "docs" \| "minimal"` | `"classic"` | 全局壳层布局：<br>• `"docs"`：类 Docusaurus 文档站（公告条 + 开源项目首页 + 多列页脚）<br>• `"classic"`：综合知识门户（多板块聚合首页）<br>• `"minimal"`：极简沉浸阅读条（隐藏页脚，适合个人博客） |
| `theme` | `string` | `"ocean"` | 站点默认内置配色 ID（当前内置 `"ocean"`、`"sunset"`、`"catppuccin"`）；访客可在顶栏切换并以 `site_theme` Cookie 记住偏好 |

---

## 3. 品牌视觉、PWA 图标与 SEO 分享图

所有 `/images/*` 路径均支持**双仓叠加**：当设置了 `CONTENT_DIR` 时，优先命中 `$CONTENT_DIR/images/*`，未命中自动回退到框架自带 `assets/images/*`。

| 字段名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `favicon` | `string` | `"/images/favicon.ico"` | 主浏览器标签页图标路径（`.ico` 或 `.svg`） |
| `favicon_16` | `string \| null` | `null` | 可选 16×16 PNG 图标（如 `"/images/favicon-16.png"`） |
| `favicon_32` | `string \| null` | `null` | 可选 32×32 PNG 图标（如 `"/images/favicon-32.png"`） |
| `apple_touch_icon` | `string \| null` | `null` | 可选 iOS 主屏幕图标（推荐 180×180 PNG，如 `"/images/apple-touch-icon.png"`） |
| `manifest` | `string \| null` | `null` | 可选 Web App Manifest 路径（如 `"/images/site.webmanifest"`） |
| `logo` | `string \| null` | `null` | 顶栏浅色模式（或通用）品牌 Logo；留空则仅显示纯文字 `site_name` |
| `logo_dark` | `string \| null` | `null` | 顶栏深色模式专属 Logo；配置后通过纯 CSS `dark:hidden` / `hidden dark:block` 零闪烁切换 |
| `logo_mobile` | `string \| null` | `null` | 窄屏移动端紧凑图标（适合横版带字 Logo 在窄屏下自动收缩为方形图形标） |
| `logo_wordmark` | `boolean` | `false` | 若 `logo` 图片内部已包含文字站名，设为 `true` 避免顶栏在图片旁重复渲染文字标题 |
| `og_image` | `string` | `"/images/og-default.jpg"` | 全站默认 Open Graph / Twitter 大图卡片（推荐 1200×630）；单篇文章可通过 Frontmatter `image` 覆盖 |
| `twitter_handle` | `string \| null` | `null` | 可选 X / Twitter 账号（如 `"@yuxuetr"`），非空时注入 `<meta name="twitter:site">` |
| `docs_edit_url_base` | `string \| null` | `null` | 文档页底部「在 GitHub 上编辑此页」前缀（如 `"https://github.com/org/repo/edit/main/assets/docs"`），自动拼接相对源文件路径 |
| `submit_case_url` | `string \| null` | `null` | `/case` 案例墙右上角「提交案例」按钮外链（留空则隐藏按钮） |

---

## 4. 顶部公告条（`announcement`）与首页配置（`hero` / `features`）

### `announcement`（可关闭顶部通知条）

| 字段路径 | 类型 | 说明 |
| :--- | :--- | :--- |
| `announcement.id` | `string` | 公告唯一版本号；访客点击关闭后写入 `localStorage["dp_ann_<id>"]`，发布新活动只需修改 `id` 即可重新向所有访客展示 |
| `announcement.text` | `string` | 中文（或默认）公告文案；为空字符串时不渲染公告条 |
| `announcement.text_en` | `string \| null` | 英文界面下的公告文案（未配时回退 `text`） |
| `announcement.href` | `string \| null` | 可选点击跳转链接（支持站内路由如 `"/docs/guide/quickstart"` 或外部 URL） |

### `hero`（首页首屏主视觉）

未配置的文案字段会自动回退到 `assets/i18n/{zh,en}.ftl` 中的 `hero.*` 词条：

| 字段路径 | 类型 | 说明 |
| :--- | :--- | :--- |
| `hero.badge` / `hero.badge_en` | `string \| null` | 主标题上方的圆角胶囊标签（在 `active_layout = "docs"` 下展示） |
| `hero.title` / `hero.title_en` | `string \| null` | 首页主标题 |
| `hero.subtitle` / `hero.subtitle_en` | `string \| null` | 首页副标题说明 |
| `hero.install_command` | `string \| null` | 一键复制终端安装/克隆命令框（仅 `active_layout = "docs"` 渲染） |
| `hero.primary_cta_text` / `primary_cta_text_en` | `string \| null` | 主行动按钮文案 |
| `hero.primary_cta_href` | `string \| null` | 主行动按钮链接（留空不渲染主按钮） |
| `hero.secondary_cta_text` / `secondary_cta_text_en` | `string \| null` | 次行动按钮文案 |
| `hero.secondary_cta_href` | `string \| null` | 次行动按钮链接（留空不渲染次按钮） |

### `features`（文档站首页 3 列特性卡片数组）

当 `active_layout = "docs"` 且 `features` 非空时，首页自动渲染响应式 3 列特性栅格：

```json
"features": [
  {
    "icon": "bolt",
    "title": "全栈 Rust SSR + WASM 水合",
    "title_en": "Fullstack Rust SSR & Hydration",
    "description": "首屏直出完整语义化 HTML，水合后享受无刷新秒开导航。",
    "description_en": "Instant crawler-friendly HTML first paint with smooth SPA navigation.",
    "href": "/docs/guide"
  }
]
```

- `icon` 内置矢量图标关键字：`"bolt"`（闪电）、`"layers"`（分层）、`"book"`（文档）、`"search"`（检索）、`"palette"`（调色盘）、`"shield"`（安全盾）；传入其它短字符串（如 Emoji）则直接原样居中展示。

---

## 5. 页脚（`footer`）与多列导航

| 字段路径 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `footer.tagline` / `tagline_en` | `string \| null` | `null` | 页脚品牌口号（留空回退 i18n `footer.tagline`） |
| `footer.copyright` | `string \| null` | `null` | 版权主体文本（留空回退 `site_name`） |
| `footer.icp` | `string \| null` | `null` | 中国大陆 ICP 备案号；非空时自动在底部渲染合规备案外链 |
| `footer.icp_url` | `string \| null` | `"https://beian.miit.gov.cn/"` | 备案点击跳转地址 |
| `footer.social` | `array` | `[]` | `{ "label": "GitHub", "url": "https://..." }` 数组；首个 GitHub 链接会自动提取到文档站顶栏右上角图标 |
| `footer.columns` | `array` | `[]` | 结构化多列链接分组，每列含 `title`、`title_en` 与 `links: [{ "label", "label_en", "href" }]` |

---

## 6. 模块开关、登录、段落标注与内容审核

### `modules`（按需启停业务板块）

你可以随时在 `site.json` 中关闭任何已编译进二进制的内置模块——关闭后会自动从**顶栏导航、路由门禁（`ModuleGate`）、`⌘K` 搜索索引、`/sitemap.xml` 与 `/feed.xml`** 中同步移除：

```json
"modules": {
  "docs": { "enabled": true },
  "blog": { "enabled": true },
  "cases": { "enabled": false },
  "podcast": { "enabled": false },
  "forum": { "enabled": false },
  "course": { "enabled": false },
  "papers": { "enabled": false },
  "embedded": { "enabled": false },
  "ai": { "enabled": false },
  "web3": { "enabled": false },
  "wasm": { "enabled": false },
  "cli": { "enabled": false },
  "about": { "enabled": true }
}
```

:::info 编译期档位与运行期开关的关系
`site.json::modules` 只能在**当前编译档位已包含的模块集合内**做运行期裁剪。例如在零数据库的 `basic` 档位下，`forum`、`course`、`papers` 本身未参与编译，无需额外配置即不存在。详见 [四档功能分层](/docs/guide/feature-tiers)。
:::

### `auth`（OAuth 第三方登录，需 `accounts` 档位）

```json
"auth": {
  "enabled": true,
  "providers": ["github", "google", "discord", "twitter"]
}
```

- 登录弹窗按 `providers` 数组顺序展示，且**仅当对应环境变量（如 `GITHUB_CLIENT_ID` + `GITHUB_CLIENT_SECRET`）非空时**才会实际渲染该按钮。

### `annotations`（段落划线标注，需 `accounts` 档位）

```json
"annotations": {
  "course": true,
  "doc": true,
  "blog": false
}
```

### `moderation`（UGC 域名黑名单与 LLM 自动风控，需 `accounts` 档位）

默认 `enabled: false`（零外部调用开销）。开启后作用于评论、论坛发帖与段落标注提交路径：

```json
"moderation": {
  "enabled": true,
  "url_blocklist": ["scam.example", "*.phishing.example"],
  "llm_review": true,
  "on_llm_failure": "review",
  "thresholds": {
    "block_above": 0.9,
    "flag_above": 0.5
  }
}
```

- `on_llm_failure`：仅允许 `"review"`（默认，LLM 超时或异常时转入后台人工待审队列）或 `"reject"`（直接拒绝提交）；出于安全基线设计（SEC-12），**不提供失败自动放行选项**，防止攻击者通过打满配额绕过风控。
