---
title: 案例展示库（Showcase）与播客专栏模块配置指南
sidebar_label: 案例库与播客
sidebar_position: 5
description: 配置开源项目/客户案例展示墙（/case）与内置音频播放器的技术播客模块（/podcast）。
keywords: [Showcase 案例墙, 技术播客, 音频播放器, 项目展示]
---

# 案例库与播客模块

除了文档、博客与课程，DioxusPress 在零数据库的 `basic` 档位下还内置了两个极具表现力的内容模块：**案例展示库（`cases`）** 与 **音频播客（`podcast`）**。

## 案例展示库（`assets/cases/`）

非常适合用来展示「使用你框架构建的项目（Built with ...）」或精选生态项目：

```text
assets/cases/
└── dioxus-fullstack-template/
    ├── meta.json               # 名称、一句话简介、分类、仓库链接、线上网址、featured 标记
    └── README.md               # 案例详细介绍正文
```

- 在 `site.json` 中设置 `"submit_case_url": "https://github.com/<org>/<repo>/issues/new?template=add-case.yml"`，案例列表页右上角会自动出现「提交案例」按钮，引导社区贡献；留空则自动隐藏。

## 播客专栏（`assets/podcasts/`）

在 `assets/podcasts/` 下按单集存放元数据与节目文稿（Shownotes），配合 `assets/audio/`（或外部对象存储音频直链），即可生成带内嵌音频进度控制、嘉宾标签与分集检索的 `/podcast` 频道页，且支持在任意博客或文档中通过 `<PodcastCard id="..." />` 直接嵌入单集收听卡片。
