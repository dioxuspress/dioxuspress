---
title: 三套内置布局（Docs / Classic / Minimal）与 10 套双模主题色卡实时试穿
sidebar_label: 布局与主题定制
sidebar_position: 4
description: 对比 DioxusPress 内置的三种整站布局外壳，实时试穿全部 10 套打通 dioxus-ui 的明暗双极性配色主题，并了解 available_themes 白名单定制。
keywords: [Docusaurus 模板, DocsShell, ClassicShell, 暗色模式, dioxus-shadcn, 主题预览, available_themes]
---

# 布局与主题定制

DioxusPress 将**页面骨架（Layout）**与**配色语义变量（Theme）**彻底正交解耦：你可以在不改动任何一篇 Markdown 或业务代码的前提下，仅通过 `site.json` 切换三套整站外壳与 10 套经过 WCAG AA 对比度校验的双模配色方案。

---

## 1. 三套内置布局外壳（`active_layout`）

### `"docs"` —— 类 Docusaurus 开源文档模板
专为开源库文档、SDK 手册、工程团队知识库设计：
- **清爽左对齐顶栏**：Logo 紧邻扁平化的「文档 / 博客 / 案例 / 关于」入口，右侧配备 `⌘K` 全文搜索胶囊、GitHub 仓库图标、色卡主题切换器、中英切换与明暗切换按钮。
- **可关闭版本公告条**：由 `site.json::announcement` 驱动，支持按 `id` 持久化关闭状态。
- **开源项目专属首页（`DocsLanding`）**：居中大字标题、终端安装命令一键复制框、`site.json::features` 驱动的三列特性栅格以及文档分卷速览卡片。
- **多列结构化页脚**：由 `site.json::footer.columns` 自由编排多列内外链。

### `"classic"` —— 综合技术门户与社区模板
专为多板块聚合门户（如线上旗舰门户 [Rust in Everything](https://rustineverything.app)）设计：
- **双生态 Mega 下拉菜单**：支持按应用领域与学习资源分栏聚合嵌入式、AI、WASM、Web3、CLI、论文速递等垂直板块。
- **门户级聚合首页**：依次编排旗舰案例橱窗、体系化付费课程展示与「最新博客 + 论坛热帖」双栏社区动态。

### `"minimal"` —— 沉浸式阅读与个人写作模板
- 48px 超窄半透明毛玻璃顶栏，仅保留紧凑品牌徽标、搜索与外观切换按钮，隐藏全局页脚，适合纯粹的个人长文笔记站。

---

## 2. 10 套内置双模主题实时试穿

DioxusPress 与官方组件库底座 [`dioxus-ui`（`dioxus-shadcn`）](https://ui.dioxuspress.dev) 共用统一的 shadcn/ui v4 语义化 CSS 变量体系，并将精选调色盘合成为**亮色（`:root`）+ 暗色（`.dark`）成对调优**的 10 套内置双模主题，所有正文与主按钮对比度均严格通过 **WCAG AA（≥ 4.5:1）** 自动化单测守卫。

服务端 `<head>` 内联了零 `'unsafe-eval'` 的极小静态脚本 `DARK_MODE_BOOT_SCRIPT`，在首帧绘制前即根据 `localStorage.theme` 或系统 `prefers-color-scheme` 挂载 `html.dark`，彻底消除 SSR 水合时的白屏闪光。

点击下方任意卡片的 **「实时试穿」** 即可全站立即切换主题（你可以同时点击右上角太阳/月亮按钮体验该主题的亮暗双态表现），满意后点击 **「复制配置」** 粘贴至 `site.json`：

<ThemeShowcase />

### 内置主题速查表

| 主题 ID | 推荐站点定位 | 亮色基调（`:root`） | 暗色基调（`.dark`） |
| :--- | :--- | :--- | :--- |
| `"ocean"`（默认） | 通用开源工程与现代 SaaS 文档 | 纯白底 + 石墨灰 + 电光青蓝（`#0e7490`） | 极夜锌黑（`#09090b`）+ 青色高光（`#22d3ee`） |
| `"emerald"` | 开发者 SDK、API 手册、运维文档 | 纯白底 + 冷松灰 + 翡翠绿（`#047857`） | 深松炭黑（`#0b1310`）+ 薄荷翠绿（`#34d399`） |
| `"nord"` | 低眩光极客知识库、长期阅读手册 | 霜雪灰白（`#f8fafc`）+ 北欧冰蓝（`#3b6ea5`） | 极夜岩石灰（`#242933`）+ 极光冰青（`#88c0d0`） |
| `"corporate"` | 企业技术门户、标准规范与开源社区 | 纯白底 + GitHub Primer 工程蓝（`#0969da`） | GitHub Dark（`#0d1117`）+ 明亮天蓝（`#58a6ff`） |
| `"winter"` | 编程语言、编译器与硬核架构博客 | 纯白底 + 凛冬靛蓝（`#4338ca`） | Tokyo Night 深蓝夜色（`#16161e`）+ 矢车菊蓝（`#7aa2f7`） |
| `"dracula"` | 创意工具、终端 CLI 与极客项目主页 | 薰衣草微紫白（`#fcfbfe`）+ 皇家紫（`#6d28d9`） | 经典 Dracula 暗紫（`#1e1f29`）+ 魅紫（`#bd93f9`） |
| `"sunset"` | 课程门户、产品发布与高亲和力社区 | 暖奶油白（`#fffaf3`）+ 落日琥珀橙（`#d97706`） | 暖褐深炭（`#1c0d05`）+ 珊瑚暖橙（`#fb923c`） |
| `"coffee"` | 独立博客、专栏连载与深度长文阅读 | 复古羊皮纸（`#fbf8f3`）+ 焦糖棕（`#854d0e`） | 浓缩深咖（`#181310`）+ 拿铁奶金（`#ddb892`） |
| `"catppuccin"` | 全天候柔和粉彩视觉、个人数字花园 | Catppuccin Latte（`#eff1f5`）+ 矢车菊蓝 | Catppuccin Macchiato（`#24273a`）+ 柔和粉蓝 |
| `"silk"` | 设计系统文档、人文随笔与杂志感排版 | 丝绸象牙白（`#faf8f6`）+ 玫瑰深红（`#be123c`） | Rose Pine 玫瑰松针暗夜（`#191724`）+ 雾粉金（`#ebbcba`） |

---

## 3. 在 `site.json` 中配置默认主题与可选白名单（`available_themes`）

默认情况下（不配置 `available_themes` 或留空数组 `[]`），顶栏主题菜单会向访客开放全部 10 套内置主题。你可以通过 `theme` 与 `available_themes` 精细控制站点外观策略：

### 场景 A：指定默认主题并开放精选的 3 套供访客切换

```json
{
  "$schema": "https://dioxuspress.dev/schemas/site.schema.json",
  "active_layout": "docs",
  "theme": "emerald",
  "available_themes": ["emerald", "ocean", "nord"]
}
```

- 菜单将严格按 `available_themes` 中声明的顺序展示，并带有亮/暗双色圆点预览；
- 若 `available_themes` 漏写了 `theme` 指定的默认主题，系统会自动将其补在列表首位；
- 若访客此前残留的 `site_theme` Cookie 不在白名单内，服务端会自动安全回退至站点默认 `theme`。

### 场景 B：锁定品牌唯一主题并隐藏切换菜单

如果你希望全站严格统一品牌色（仍保留亮色/暗色切换按钮，但隐藏多主题下拉菜单），只需将 `available_themes` 设为仅包含默认主题的单元素数组：

```json
{
  "theme": "corporate",
  "available_themes": ["corporate"]
}
```

当开放主题数 `<= 1` 时，顶栏 `ThemePicker` 会自动隐藏，零多余 DOM 渲染。
