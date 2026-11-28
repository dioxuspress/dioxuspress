# 界面语言的首屏渲染与英文 URL（V10，2026-11-05）

## 现状与问题

- 界面语言只在客户端定：`init_i18n` 一律给 `Language::Zh`，hydration 之后的 `use_effect`
  读 `site_lang` cookie 再切。SSR 永远出中文。
  - 选了英文的访客：每次打开页面先看到中文，wasm 加载完再跳成英文。
  - 搜索引擎看到的永远是中文，没有能被收录的英文页面。
- 站内的英文内容其实只有**论文速递**：75 篇论文都有 `en` 解读（`paper_summaries.lang = en`）。
  博客 164 篇、文档约 2000 页都只有中文，给它们开英文 URL 只会得到「英文导航 + 中文正文」的重复页。

## 决定

1. **首屏语言在服务端就定下来**，SSR 与 hydration 用同一个值，不再闪：
   `?lang=en|zh` 查询参数 > `site_lang` cookie > 中文。服务端从当前请求
   （`FullstackContext`）读，客户端从 `location.search` / `document.cookie` 读，规则同一个纯函数。
2. **按真实双语内容开放英文 URL**（论文页 + 带 `index.en.{md,mdx}` 伴生文件的文档页 + 带 `about.en.{md,mdx}` 的 About 页）：
   - 论文页：`/papers?lang=en`、`/papers/<id>?lang=en`、`/papers/topic/<slug>?lang=en`；
   - 文档页（Phase E5.1）：当某文档目录同时存在 `index.{md,mdx}` 与 `index.en.{md,mdx}` 时（如 `dioxuspress.dev` 官方文档全集），自动开放 `/docs/<path>?lang=en` 与 `/docs?lang=en`，英文界面优先读取 `index.en.{md,mdx}` 并本地化侧栏/面包屑/上下篇标题，缺失时平滑回退中文原文件（且将 `<article lang="zh-CN">` 保持为实际正文语言）。
   - About 页：当生效的 `pages/` 目录同时存在 `about.{md,mdx}` 与 `about.en.{md,mdx}` 时，自动开放 `/about?lang=en` 并在语言切换时响应式加载对应正文；若外部挂载 `$CONTENT_DIR/pages/about.md` 未提供英文伴生文件，则严格在该挂载目录内回退中文原稿，不会串读镜像内置的 `assets/pages/about.en.md`。
   - 这些真实双语页面输出 `<link rel="alternate" hreflang="zh-CN|en">`，canonical 指向各自语言的 URL，且 `/sitemap.xml` 自动收录对应的 `?lang=en` 条目。
   - 不输出 `x-default`：它的 href 与 zh-CN 相同，而 `document::Link` 按 href+rel 去重
     （dioxus-document 0.7.9 `should_insert_link`），第二条会被丢掉。x-default 对 Google 是可选项，
     两种语言、默认中文的情况下不影响收录。
   - 单语页面（如 `rustineverything.app` 上无 `.en.md` 伴生文件的中文文档与博客）带 `?lang=en` 也能用（界面英文），但不宣告 hreflang、不进英文 sitemap，canonical 仍是无参数的中文 URL，不会形成重复收录。
3. 用查询参数而不是 `/en/...` 前缀：路由枚举不用整套复制，现有链接都不变。

## 不做（当前版本）

- 博客文章的英文 URL：目前博客文章以单语为主。**重估条件**：博客出现成对 `index.en.md` 超过 20 篇。
- 按 `Accept-Language` 自动选语言：同一 URL 按请求头出不同语言，对缓存和收录都不友好。

## 验收

- `curl -H 'Cookie: site_lang=en' /papers` 的 HTML 是英文界面；无 JS 打开与有 JS 打开的语言一致。
- `/papers/<id>?lang=en` 的 SSR HTML 含英文解读、`hreflang` 两条（zh-CN、en）、canonical 带 `?lang=en`。
- 切语言、SPA 导航后无 hydration 报错。
