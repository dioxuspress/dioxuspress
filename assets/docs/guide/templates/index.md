---
title: 官方 Starter 模板选用指南：docs、blog、community 与 portal 四套预设对比
sidebar_label: Starter 模板选用
sidebar_position: 6
description: 对比 DioxusPress 四套场景化 Starter 预设（开源文档、独立博客、技术社区、知识门户）的布局、配色、目录结构与无缝升级路径。
keywords: [Starter Templates, dioxuspress init, 文档站模板, 独立博客模板, 知识社区模板]
---

# 官方 Starter 模板选用指南

DioxusPress 在引擎仓库 [`templates/`](https://github.com/dioxuspress/dioxuspress/tree/main/templates) 维护了四套官方精调预设，并通过编译期内嵌与 `dioxuspress init` 保持**唯一真实源同步**。你也可以前往交互式 **[模板中心（`/templates`）](/templates)** 同屏预览四套模板在桌面亮色与暗色下的视觉表现。

## 四套预设速查表

| 预设名 | 初始化参数 | 默认布局 | 默认主题 | 容器镜像与依赖 | 核心预置内容 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **开源文档站** | `--template docs` | `docs` | `emerald` | `dioxuspress-basic`（零数据库） | 顶部公告条、首页特性卡、`guide` + `api` 双卷三级侧栏、GitHub Actions 自动体检流 |
| **极简技术博客** | `--template blog` | `minimal` | `coffee` | `dioxuspress-basic`（零数据库） | 暖羊皮纸护眼配色、服务端 MathML 公式、代码对比标签组、`authors.yml`、突出 `/feed.xml` |
| **技术社区站** | `--template community` | `classic` | `ocean` | `dioxuspress-app` + PostgreSQL 16 | GitHub/Google/Discord OAuth2 PKCE 指引、话题论坛（`/topics`）、段落划线标注、域名黑名单 |
| **全栈知识门户** | `--template portal` | `classic` | `corporate` | `dioxuspress-app` + PostgreSQL 16 | 试读与付费课程（`courses/`）、精选案例库（`cases/`）、MDX 内嵌播客卡片、可选 LLM 自动审核 |

## 一行命令初始化

```bash
# 示例：创建一套零数据库的开源文档站（站点名将自动根据目录名生成）
dioxuspress init my-project-docs --template docs
cd my-project-docs
dioxuspress check
docker compose up -d
```

## 从零数据库平滑升级到全栈社区

四套预设遵循**完全一致的内容目录契约**（`site.json` + `docs/` + `posts/` + `images/`）。当你最初使用 `docs` 或 `blog` 零数据库起步，后续希望为读者开启评论、论坛或付费专栏时，**无需迁移或改写任何一篇 Markdown**：

1. 将 `docker-compose.yml` 中的镜像从 `ghcr.io/dioxuspress/dioxuspress-basic:latest` 切换为 `ghcr.io/dioxuspress/dioxuspress-app:latest` 并挂载 `postgres:16-alpine` 服务（可直接参考 `templates/community/docker-compose.yml`）；
2. 新增 `.env` 配置 `DATABASE_URL`、`JWT_SECRET` 与所需的 OAuth Client 凭据；
3. 在 `site.json` 中将 `auth.enabled` 与 `modules.forum.enabled`（或 `modules.course.enabled`）设为 `true` 并重启容器，SeaORM 会自动建表生效。
