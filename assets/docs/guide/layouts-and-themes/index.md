---
title: 三套内置布局（Docs / Classic / Minimal）与多主题配色系统
sidebar_label: 布局与主题定制
sidebar_position: 4
description: 对比 DioxusPress 内置的三种整站布局外壳与基于 dioxus-shadcn 语义令牌的明暗双态主题引擎。
keywords: [Docusaurus 模板, DocsShell, ClassicShell, 暗色模式, dioxus-shadcn]
---

# 布局与主题定制

DioxusPress 在应用外壳层内置了三套完整的布局模板，并通过 `dioxus-shadcn` 语义化 CSS 变量统一驱动明暗模式与多配色方案。

## 三套内置布局（`active_layout`）

### 1. `"docs"` —— 类 Docusaurus 开源文档模板
专为开源库文档、技术手册、工程团队知识库设计：
- **清爽左对齐顶栏**：Logo 紧邻扁平化的「文档 / 博客 / 案例 / 关于」入口，右侧配备 `⌘K` 搜索胶囊、GitHub 仓库图标、主题配色器、中英切换与明暗切换按钮。
- **可关闭版本公告条**：支持顶部渐变横幅提示新版本发布。
- **开源项目专属首页（`DocsLanding`）**：居中大字标题、终端安装命令一键复制框、`site.json::features` 驱动的三列特性栅格以及文档分卷速览卡片。
- **多列结构化页脚**：由 `site.json::footer.columns` 自由编排多列内外链。

### 2. `"classic"` —— 综合技术门户与社区模板
专为多板块聚合门户（如官方线上站 [Rust in Everything](https://rustineverything.app)）设计：
- **双生态 Mega 下拉菜单**：支持按应用领域与学习资源分栏聚合嵌入式、AI、WASM、Web3、CLI、论文速递等垂直板块。
- **门户级聚合首页**：依次编排旗舰案例橱窗、体系化付费课程展示与「最新博客 + 论坛热帖」双栏社区动态。

### 3. `"minimal"` —— 沉浸式阅读与个人写作模板
- 48px 超窄半透明毛玻璃顶栏，仅保留紧凑品牌徽标、搜索与外观切换按钮，无页脚干扰，适合纯粹的个人长文笔记站。

---

## 零闪烁明暗模式与三套内置主题

DioxusPress 在服务端 `<head>` 中内联了极轻量的 `DARK_MODE_BOOT_SCRIPT`（纯静态脚本，严格遵守无 `'unsafe-eval'` 的 CSP 策略），在浏览器首次绘制前即根据 `localStorage.theme` 或系统 `prefers-color-scheme` 挂载 `html.dark` 类名，彻底杜绝 SSR 水合时的白屏闪光。

内置三套可由访客实时切换并持久化至 Cookie 的配色主题：
- **`ocean`**（默认）：石墨（Zinc）中性底色 + 电光青蓝主色，清爽利落的现代工程风；
- **`sunset`**：暖调琥珀橙主色，醒目富有活力；
- **`catppuccin`**：柔和低饱和紫粉调，适合长时间夜间深度阅读。
