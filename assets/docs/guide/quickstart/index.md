---
title: 5 分钟快速开始：零编译 Docker 建站、CLI 脚手架与源码开发
sidebar_label: 5 分钟快速开始
sidebar_position: 1
description: 使用 dioxuspress CLI 与官方预构建零数据库镜像在 50 秒内启动文档或博客站，或克隆源码运行全栈模式。
keywords: [DioxusPress 安装, dioxuspress init, dioxuspress check, 零数据库启动, Docker 建站]
---

# 5 分钟快速开始

DioxusPress 提供三条清晰的上手路径：

1. **零编译 50 秒建站（推荐大多数内容创作者）**：无需安装 Rust 编译链，一行命令初始化内容目录并通过官方预构建容器启动。
2. **克隆引擎源码本地开发（零数据库模式）**：适合定制 UI 组件、样式或模块扩展。
3. **完整全栈社区模式（含 PostgreSQL）**：适合启用 OAuth 登录、话题论坛、段落划线标注与实战课程。

---

## 方式一：零编译 50 秒建站（推荐）

只想专注编写 Markdown 文档与技术博客时，使用轻量 CLI `dioxuspress` 搭配官方双架构镜像 `ghcr.io/dioxuspress/dioxuspress-basic`（同时支持 Apple Silicon `arm64` 与 Linux `amd64`）：

### 1. 安装官方轻量 CLI（可选，编译仅需约 6 秒）

```bash
cargo install --git https://github.com/dioxuspress/dioxuspress dioxuspress-cli
```

:::tip 纯 Docker 免安装运行
若本机未安装 Rust，也可以跳过 CLI，直接新建含 `docs/index.md` 的目录并执行：
`docker run --rm -p 8080:8080 -v ./my-site:/app/content:ro ghcr.io/dioxuspress/dioxuspress-basic:latest`
:::

### 2. 初始化内容仓库并校验

`dioxuspress init` 内置 4 套场景预设（`docs` 文档站、`blog` 极简博客、`community` 社区、`portal` 知识门户；可在 [模板中心](/templates) 在线预览对比）：

```bash
# 生成带 JSON Schema 自动补全的 site.json、示例文档与 docker-compose.yml
dioxuspress init my-docs --template docs
cd my-docs

# 毫秒级静态检查配置拼写、Frontmatter、围栏代码块与站内死链
dioxuspress check
```

### 3. 启动容器并预览

```bash
docker compose up -d
```

打开浏览器访问 `http://localhost:8080`，即可看到完整的类 Docusaurus 首页、三级文档侧栏、双模主题切换与 `⌘K` 中英文全文检索。

---

## 方式二：克隆引擎源码本地开发（零数据库模式）

当你需要修改 Rust 组件或 Tailwind CSS 样式时，使用 `dx` CLI 本地热重载：

### 前置工具

- **Rust 稳定版工具链**与 WASM 编译目标：
  ```bash
  rustup update stable
  rustup target add wasm32-unknown-unknown
  ```
- **Dioxus CLI (`dx`)**：
  ```bash
  cargo install dioxus-cli
  ```

### 启动 `basic` 无数据库开发服

```bash
git clone https://github.com/dioxuspress/dioxuspress.git
cd dioxuspress/crates/app
BASE_URL=http://127.0.0.1:8080 dx serve --web --no-default-features
```

:::tip
当传入 `--no-default-features` 时，请务必显式附带 `--web`，以便 `dx` 识别全栈 Web 平台目标。
:::

---

## 方式三：完整全栈社区模式（含登录、论坛、课程）

需要本地体验 OAuth 登录、评论、论坛、段落划线标注或管理后台时，启动 PostgreSQL 并使用默认特性集：

```bash
# 1. 在仓库根目录复制环境变量模板，填写 JWT_SECRET（可用 openssl rand -hex 32 生成）与 DATABASE_URL
cp .env.example .env

# 2. 启动本地 PostgreSQL（应用启动时会自动执行 SeaORM 迁移建表）
docker compose up -d postgres

# 3. 启动完整全栈开发服
cd crates/app
dx serve
```

:::warning 本地开发 `localhost` 与 `127.0.0.1` 混用陷阱
若配置了 GitHub / Google 等 OAuth 登录，请务必使用 `.env` 中 `BASE_URL` 完全一致的主机名（如 `http://127.0.0.1:8080`）访问浏览器；混用 `localhost` 与 `127.0.0.1` 会导致回调时 PKCE Cookie 域名不匹配。
:::

## 下一步

- 阅读 [site.json 完整配置参考](/docs/guide/site-config)，借助编辑器自动补全定制品牌、导航与双模主题；
- 阅读 [零数据库部署指南](/docs/operations/zero-db-deploy)，了解如何将内容目录秒级部署到任意云服务器；
- 在 CI 中加入 `dioxuspress check`，让每次文档 PR 自动拦截死链与语法错误。
