---
title: Tailwind CSS v4 与 dioxus-shadcn 组件样式构建工作流
sidebar_label: Tailwind CSS 工作流
sidebar_position: 5
description: 了解 DioxusPress 如何自动扫描全工作区 Rust RSX 类名并编译压缩 Tailwind CSS v4 样式表。
keywords: [Tailwind CSS v4, dioxus-shadcn, tw-sources, RSX 样式构建]
---

# Tailwind CSS v4 工作流

DioxusPress 使用 **Tailwind CSS v4** 搭配自研组件库 [`dioxus-shadcn`](https://github.com/dioxuspress/dioxus-ui)，所有色彩均收敛至 `--background`、`--foreground`、`--primary`、`--muted` 等语义化设计令牌。

## 自动源扫描机制

由于 Rust 工作区包含多个模块 Crate（`crates/modules/*`、`crates/widgets`）以及外部 Cargo 缓存中的 `dioxus-shadcn` 组件源码，仓库提供了 `scripts/tw-sources.mjs` 脚本自动生成 `@source` 声明：

1. **输入文件**：`crates/app/tailwind-input.css`
2. **构建命令**：
   ```bash
   cd crates/app
   npm install
   npm run build
   ```
3. **产物同步**：编译生成的压缩样式会同步输出至 `assets/tailwind.css` 并纳入版本控制，因此普通使用者克隆仓库后**无需安装 Node.js 也能直接运行或构建 Rust 二进制**。

## 开发期实时监听

在修改页面 RSX 中的 Tailwind 类名时，可在独立终端开启 Watch 模式：

```bash
cd crates/app
npm run dev
```

每次保存 `.rs` 文件后，增量样式表会在约 `100ms` 内重编完成，配合 `dx serve` 热重载即时呈现。
