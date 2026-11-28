---
title: 生产部署、全文检索与模块扩展开发总览
sidebar_label: 部署与架构扩展
sidebar_position: 4
description: 涵盖零数据库单容器部署、Docker Compose 生产运维、HTTPS 网关配置、Tantivy 搜索与自定义模块开发。
keywords: [Docker 部署, 零数据库运维, Pingora 反向代理, Tantivy 搜索, 模块开发]
---

# 部署、搜索与架构扩展

DioxusPress 的生产部署遵循「小站点极简、大站点稳健」的原则：无论是每月几美元的入门云主机跑多个零数据库文档站，还是带 PostgreSQL 与双仓库内容热挂载的全功能生产集群，都有清晰的标准化方案。

## 部署形态选型建议

| 你的目标场景 | 推荐构建档位 | 服务器资源需求 | 推荐阅读 |
| :--- | :--- | :--- | :--- |
| 开源库官方文档、个人博客、团队手册 | `basic`（`--no-default-features`） | **0.5 CPU / 256 MB RAM**（无数据库） | [零数据库轻量部署](/docs/operations/zero-db-deploy) |
| 含登录、评论、论坛、课程与后台的社区站 | 默认完整档位或 `course` | **≥ 1 CPU / 1 GB RAM**（App + PostgreSQL） | [全栈生产部署与运维](/docs/operations/production-deploy) |

## 本卷章节导航

- **[零数据库轻量部署](/docs/operations/zero-db-deploy)**：单 Docker 容器 + 反向代理跑起极速文档站。
- **[全栈生产部署与备份运维](/docs/operations/production-deploy)**：Docker Compose、外部私有内容挂载、健康检查与数据库备份。
- **[Tantivy 搜索与全站 SEO](/docs/operations/search-and-seo)**：增量索引清单、GSC 站点地图提交与结构化数据。
- **[引擎架构与新模块开发](/docs/operations/module-development)**：如何按高内聚低耦合规范为 DioxusPress 新增一个业务模块。
- **[环境变量与网关配置字典](/docs/operations/env-reference)**：`.env` 全量运行参数、OAuth / 支付密钥与 Pingora 多域名 SNI 网关参考。
- **[实测性能看板与体积守卫](/docs/operations/benchmarks)**：空闲内存、SSR / `304` 延迟吞吐、Tantivy 检索速度与前端 WASM 体积 CI 门禁。
