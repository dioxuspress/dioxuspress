---
title: 四档功能分层（Cargo Features）：从静态文档站到付费课程社区按需编译
sidebar_label: 四档功能分层
sidebar_position: 2
description: 详解 DioxusPress 的 basic、accounts、course 与 payments 四档编译期特性裁剪机制与运行时依赖。
keywords: [Cargo features, 功能分层, 零数据库, 按需编译, DioxusPress 架构]
---

# 四档功能分层（Cargo Features）

DioxusPress 采用「每个业务领域独立 Crate + 组合根按 Feature 装配」的工程设计。不需要的功能不仅在界面上隐藏，连对应的后端 API 路由与数据库迁移逻辑都不会编译进最终产物。

## 四档能力与依赖矩阵

| 档位名称 | 编译参数（`crates/app`） | 包含能力 | 运行时必需环境变量 |
| :--- | :--- | :--- | :--- |
| **`basic`**（文档/博客站） | `--no-default-features` | 文档、博客、案例、播客、专题板块、Tantivy 搜索、About、主题/明暗/双语、Sitemap/RSS | 仅 `BASE_URL`（**无需数据库**） |
| **`accounts`**（互动社区） | `--no-default-features --features accounts` | `basic` + OAuth2 登录、论坛话题、文章评论、图片上传、LLM 内容审核、`/admin` 后台 | + `DATABASE_URL`、`JWT_SECRET` |
| **`course`**（知识付费） | `--no-default-features --features course` | `accounts` + 实战课程目录、免费试读/付费墙、段落划线标注、会员权益手动发放 | 同上 |
| **`full`**（默认完整版） | 默认（无需额外参数） | `course` + 微信支付 / 支付宝在线下单、异步回调自动发货、订单中心与文章打赏 | 同上 + 支付商户密钥 |

## 双重开关机制：编译期裁剪 × 运行时启停

DioxusPress 将模块控制拆为两层，互不冲突：

1. **编译期物理边界（Cargo Features）**：应用启动时通过 `register_compiled_modules` 登记当前二进制实际链接的模块集合；未编译的路由自动由 `absent_pages!` 宏降级为标准 `404` 页面，不存在未授权接口暴露面。
2. **运行时按站裁剪（`site.json::modules`）**：即使编译了完整镜像，也可以在 `site.json` 中关闭当前站点不需要的板块（例如只保留 `docs` 与 `blog`），导航栏、`ModuleGate`、全站搜索与 `sitemap.xml` 会同步生效：

```json
{
  "modules": {
    "docs": { "enabled": true },
    "blog": { "enabled": true },
    "cases": { "enabled": true },
    "podcast": { "enabled": false },
    "embedded": { "enabled": false }
  }
}
```

## 提交前四档门禁检查

如果你对框架组件或路由做了二次开发，合并前可运行仓库内置的四档交叉编译门禁脚本，一次性验证 4 档 ×（`server` + `web` WASM）共 8 种组合零警告通过：

```bash
./scripts/check_tiers.sh
```
