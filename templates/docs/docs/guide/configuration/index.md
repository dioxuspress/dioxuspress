---
title: 站点与富排版配置：主题色卡、代码标签组与提示框
sidebar_label: 站点与排版配置
sidebar_position: 2
description: 演示 site.json 自动补全、多语言代码切换标签组与结构化提示框语法。
keywords: [site.json, 代码标签组, Admonitions]
---

# 站点与富排版配置

`site.json` 顶层已声明 `"$schema"`，在 VS Code 或 Zed 中编辑时可获得完整的字段悬停说明与主题枚举补全。

## 多语言代码切换标签组（`:::code-tabs`）

在同一个代码块内对比不同安装方式或编程语言：

:::code-tabs
```bash [Cargo CLI]
cargo install --git https://github.com/dioxuspress/dioxuspress.git dioxuspress-cli
dioxuspress check
```
```bash [Docker Compose]
docker compose up -d
curl -I http://localhost:8080
```
:::

## 常用配置速查

| 字段 | 当前预设值 | 作用说明 |
| :--- | :--- | :--- |
| `active_layout` | `"docs"` | 启用类 Docusaurus 左对齐顶栏、公告横幅与多列页脚 |
| `theme` | `"emerald"` | 默认采用清翠绿文档配色，自动适配亮色与 `.dark` 暗色模式 |
| `available_themes` | 5 套精选色卡 | 控制顶栏调色盘向访客开放的配色列表 |

:::important 持续集成门禁
仓库内置的 `.github/workflows/check.yml` 会在每次提交 Pull Request 时自动运行 `dioxuspress check`，拦截断链与未闭合代码块。详见 [生产环境部署](/docs/guide/advanced/deployment)。
:::
