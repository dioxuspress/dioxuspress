---
title: 快速开始：1 分钟启动文档站
sidebar_label: 快速开始
sidebar_position: 1
description: 无需安装 Rust 编译工具链，使用官方预构建镜像在本地运行并校验 DioxusPress Docs Starter。
keywords: [快速开始, Docker, dioxuspress check]
---

# 快速开始

当前仓库采用**内容与引擎分离**架构：所有 Markdown 文档、发布博客与 `site.json` 均以纯文本维护，由官方零数据库镜像 `ghcr.io/dioxuspress/dioxuspress-basic` 直接挂载渲染。

## 目录结构一览

```text
.
├── .github/workflows/check.yml  # GitHub Actions 内容与死链自动体检
├── docker-compose.yml           # 零数据库单容器编排
├── site.json                    # 品牌、布局、主题白名单与首页配置
├── docs/                        # 多卷三级文档树（guide/ 与 api/）
├── posts/                       # 版本发布公告与技术博客
└── images/                      # 自定义 Logo、Favicon 与配图
```

## 本地启动与静态校验

```bash
# 1. 毫秒级检查 site.json 字段拼写、代码块闭合与站内链接
dioxuspress check

# 2. 启动零数据库预览容器
docker compose up -d
```

:::tip 接下来做什么？
尝试修改 [站点与排版配置](/docs/guide/configuration) 或查看第一篇发布公告 [v0.1 正式发布](/blog/v0-1-release)。
:::
