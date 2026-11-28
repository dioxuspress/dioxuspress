# Theme Spec

> 站点外观由「主题」（颜色 token）与「布局」（页面骨架）描述，二者正交。
> 主题内置在宿主代码里（R2，2026-10-08 起不再是 WASM 插件；Phase E2 与 `dioxus-ui` 语义调色盘打通扩展至 10 套双模预设）；默认值与可选白名单来自 `site.json`，访客可用 cookie 覆盖。

## 1. 总体架构

```text
site.json "theme": "ocean"              访客 cookie site_theme=emerald（可选）
site.json "available_themes": [...]                │
          │                                        │
          └───────────────────┬────────────────────┘
                              ▼
   engines::theme::resolve_theme_for_site(default, available, cookie)
   （crates/core/src/engines/theme.rs；只认 THEMES 表内且在白名单中的 id）
                              │
                              ▼
   server fn get_aggregated_theme_css → Theme.css（include_str! 的 .css 文件）
                              │
                              ▼
   crates/app/src/main.rs：<style id="wasm-theme-style">{css}</style>
```

## 2. 内置双模主题矩阵（10 套）

`crates/core/src/engines/theme.rs` 的 `THEMES` 表，顺序即切换菜单默认顺序，第一项是回退主题。每套均在 `:root` 与 `.dark` 提供完整成对语义变量，并附带色卡预览元数据（`light_bg` / `light_primary` / `dark_bg` / `dark_primary`）：

| id | CSS 文件 | 风格定位 |
| --- | --- | --- |
| `ocean` | `crates/core/src/engines/themes/ocean.css` | 默认：石墨（zinc）中性色 + 电光青蓝主色 |
| `emerald` | `crates/core/src/engines/themes/emerald.css` | 清爽翡翠绿 + 深松炭黑暗色，适合开发者与 API 文档站 |
| `nord` | `crates/core/src/engines/themes/nord.css` | 北欧极地冰蓝 + 霜雪灰 / 极夜岩灰，低眩光耐看 |
| `corporate` | `crates/core/src/engines/themes/corporate.css` | GitHub Primer 风格皇家工程蓝，企业与开源门户 |
| `winter` | `crates/core/src/engines/themes/winter.css` | 凛冬靛蓝 + Tokyo Night 深蓝暗色 |
| `dracula` | `crates/core/src/engines/themes/dracula.css` | 薰衣草浅白 + 经典 Dracula 暗紫 |
| `sunset` | `crates/core/src/engines/themes/sunset.css` | 暖米底色 + 落日琥珀橙 |
| `coffee` | `crates/core/src/engines/themes/coffee.css` | 羊皮纸暖底 + 浓缩深咖暗色，适合独立博客与长文阅读 |
| `catppuccin` | `crates/core/src/engines/themes/catppuccin.css` | Catppuccin Latte / Macchiato 柔和粉彩 |
| `silk` | `crates/core/src/engines/themes/silk.css` | 丝绸象牙白 + Rose Pine 玫瑰松针暗夜 |

CSS 在编译期嵌入（`include_str!`），属于作者写的受信内容，原样输出。

## 3. 白名单过滤（`available_themes`）与访客 Cookie 覆盖

- `resolve_available_themes(site_default, available)`：
  - `available` 为空时返回 `THEMES` 全表；
  - `available` 非空时按配置顺序过滤合法内置 id 并去重，若漏写了 `site_default` 则自动补在首位；
  - 若只开放 1 套主题（如 `"theme": "ocean", "available_themes": ["ocean"]`），前端 `ThemePicker` 因列表长度 `<= 1` 自动隐藏。
- ThemePicker 或 `<ThemeShowcase />` 选中主题后，`/api/theme/set` 写：

```text
Cookie 名:  site_theme
值:         当前站点开放的内置主题 id（如 emerald）
属性:       Path=/; Max-Age=31536000; SameSite=Lax（非 HttpOnly：前端也会写同一 cookie）
生产 HTTPS: 自动追加 Secure
```

`resolve_theme_for_site` 的规则：cookie 是表内且在当前站点开放白名单内的 id → 用它；否则记 warn 后忽略，回退到 `site.json` 的 `theme`；后者也不在表内时用 `THEMES[0]`。清空 cookie 等价于回到 `site.json` 默认。

## 4. ThemePicker 与 ThemeShowcase 组件

- `crates/app/src/components/theme_picker.rs`：
  - `list_available_themes` 返回当前站点开放的主题列表（含亮/暗色卡色值与中英简介），`is_active` 标出当前生效项；
  - 下拉菜单为每项渲染亮/暗主色双圆点预览并支持限高滚动（`max-h-72 overflow-y-auto`）；
  - 共享辅助函数 `apply_user_theme(id, version)` 负责 `set_user_theme` + `document.cookie` 双写并递增 `ThemeVersion` Signal。
- `crates/app/src/components/theme_showcase.rs`：
  - 在 `crates/app/src/main.rs` 注册为 `<ThemeShowcase />` MDX 组件；
  - 在文档正文中渲染响应式双模色带卡片矩阵，支持一键全站实时试穿与 `"theme": "<id>"` 配置片段复制。

## 5. 新增主题

1. 在 `crates/core/src/engines/themes/` 新建 `<id>.css`，按 §8「Token 契约」在 `:root` 与 `.dark` 各给一套完整的 token；
2. 在 `THEMES` 表加一项 `Theme { id, label, description, description_en, light_bg, light_primary, dark_bg, dark_primary, css: include_str!("themes/<id>.css") }`；
3. 运行 `UPDATE_SCHEMA=1 cargo test --features server -p app-core`：
   - `theme_ids_are_unique_and_every_theme_sets_light_and_dark_tokens` 自动校验 id 唯一、亮暗 token 齐全及正文/主按钮满足 WCAG AA `>= 4.5:1` 对比度；
   - `site_schema_json_is_up_to_date` 自动同步 `assets/schemas/site.schema.json` 的枚举候选。

不要引用外部资源（字体、图片 URL）：CSP 的 `font-src` / `img-src` 不放行外部主机，主题对所有访客全局生效。

## 6. 布局 (Layout)

布局是「壳」（shell）级别的页面骨架，与主题（颜色变量）正交。`SiteConfig.active_layout` 控制当前布局，默认 `"classic"`：

| 布局 | 文件 | 形态 |
| --- | --- | --- |
| `docs` | `crates/app/src/components/layouts/docs.rs::DocsShell` | 类 Docusaurus 公告条 + 清爽左对齐顶栏 + `DocsLanding` + 多列页脚 |
| `classic` | `crates/app/src/components/layouts/classic.rs::ClassicShell` | Navbar + 双生态下拉主导航 + 工具区 + Footer |
| `minimal` | `crates/app/src/components/layouts/minimal.rs::MinimalShell` | 紧凑顶部条 (无主导航 / 无 Footer)，写作 / 阅读优先 |

## 7. 与 site.json 的集成

```jsonc
{
  "$schema": "https://dioxuspress.dev/schemas/site.schema.json",
  "theme": "ocean",                         // 内置主题 id，缺省为 "ocean"
  "available_themes": ["ocean", "emerald"], // 可选白名单，缺省 [] 开放全部 10 套
  "active_layout": "docs"                   // "docs" | "classic" | "minimal"
}
```

## 8. Token 契约

主题输出 [shadcn/ui](https://ui.shadcn.com/docs/theming) 语义 token，站点组件（`dioxus-shadcn`）与手写类名都从这些变量取色。默认值定义在 `crates/app/tailwind-input.css`，主题 CSS 以 `<style id="wasm-theme-style">` 在其后注入，同名变量覆盖默认值。

| token | 用途 |
| --- | --- |
| `--background` / `--foreground` | 页面底色与正文 |
| `--card` / `--card-foreground` | 卡片、面板 |
| `--popover` / `--popover-foreground` | 弹层、下拉、对话框 |
| `--primary` / `--primary-foreground` | 主按钮、强调、当前导航项 |
| `--secondary` / `--secondary-foreground` | 次按钮、弱化标签底色 |
| `--muted` / `--muted-foreground` | 弱化底色与次要文字 |
| `--accent` / `--accent-foreground` | 悬停、选中项 |
| `--border` / `--input` / `--ring` | 边框、输入框边框、焦点环 |

主题应在 `:root` 与 `.dark` 各给一套完整的值（`.dark` 加在 `<html>` 上）。`--destructive` / `--success` / `--warning` / `--info`、`--chart-*`、`--sidebar-*` 可按需覆盖或沿用默认值。

**旧变量（兼容）**：`--color-primary` / `--color-bg` / `--color-surface` / `--color-text` / `--color-text-muted` / `--color-border` 在 `assets/css/main.css` 中定义为上表 token 的别名（`--color-bg: var(--background)` 等）。它们只为存量的 `var(--color-*)` 手写类名保留；只改别名不会让组件跟随，主题请输出语义 token。
