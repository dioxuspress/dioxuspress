---
title: 5 分钟快速开始：本地运行 DioxusPress 零数据库文档站与全栈模式
sidebar_label: 5 分钟快速开始
sidebar_position: 1
description: 安装 dioxus-cli，使用 basic 无数据库档位在 1 分钟内启动文档站，或通过 Docker Compose 启动全栈模式。
keywords: [DioxusPress 安装, dx serve, 零数据库启动, Dioxus 全栈开发]
---

# 5 分钟快速开始

DioxusPress 支持两种典型本地启动路径：**零数据库文档站模式（最快，无需 Docker 与 PostgreSQL）** 与 **完整全栈社区模式**。

## 前置工具

- **Rust 工具链**：建议使用最新稳定版（`rustup update stable`），并添加 WASM 编译目标：
  ```bash
  rustup target add wasm32-unknown-unknown
  ```
- **Dioxus CLI (`dx`)**：用于全栈 SSR + WASM 热重载打包：
  ```bash
  cargo install dioxus-cli
  ```
- **Node.js 20+**（仅在修改了 Tailwind CSS 类名重新构建样式时需要；仓库已附带预编译产物，直接运行可跳过）。

---

## 方式一：零数据库文档站模式（推荐尝鲜）

如果你只想搭建类似 Docusaurus 的技术文档、个人博客或项目展示站，使用 `basic` 档位（`--no-default-features`）即可，**无需配置数据库或任何密钥**：

```bash
# 1. 克隆仓库
git clone https://github.com/dioxuspress/dioxuspress.git
cd dioxuspress

# 2. 仅需指定站点基础 URL，直接以 basic 档位启动
cd crates/app
BASE_URL=http://127.0.0.1:8080 dx serve --web --no-default-features
```

编译完成后，浏览器访问 `http://127.0.0.1:8080` 即可看到完整的类 Docusaurus 首页、三级文档树、博客与 `⌘K` 全站搜索。

:::tip
当不使用默认 Cargo features 时，务必显式传入 `--web`，以便 `dx` CLI 识别全栈 Web 平台目标。
:::

---

## 方式二：完整全栈社区模式（含登录、论坛、课程）

需要体验 OAuth 登录、评论、论坛、段落划线标注或管理后台时，启动 PostgreSQL 并使用默认特性集：

```bash
# 1. 在仓库根目录复制环境变量模板
cp .env.example .env

# 2. 编辑 .env，至少填写 JWT_SECRET（32 字节以上随机串）与 DATABASE_URL
#    可用 openssl rand -hex 32 生成密钥

# 3. 启动本地 PostgreSQL（应用启动时会自动执行 SeaORM 迁移建表）
docker compose up -d postgres

# 4. 启动完整全栈开发服
cd crates/app
dx serve
```

:::warning
若配置了 GitHub / Google 等 OAuth 登录，请务必使用 `.env` 中 `BASE_URL` 完全一致的主机名（如 `http://127.0.0.1:8080`）访问浏览器；混用 `localhost` 与 `127.0.0.1` 会导致回调时 PKCE Cookie 域名不匹配。
:::

## 下一步

- 打开 `assets/site.json` 修改 `site_name`、`hero` 与 `features`，刷新页面查看效果；
- 在 `assets/docs/` 下新建一个目录与 `index.md`，体验自动侧栏挂载与全文索引收录。
