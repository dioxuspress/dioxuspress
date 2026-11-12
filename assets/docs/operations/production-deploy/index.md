---
title: 全栈生产环境部署与运维：Docker Compose、私有内容挂载与数据备份
sidebar_label: 全栈生产部署与运维
sidebar_position: 2
description: 部署带 PostgreSQL 的完整社区版 DioxusPress，实现引擎与私有内容双仓库分离、健康探针监控与日常备份恢复。
keywords: [生产部署, docker-compose, CONTENT_DIR 挂载, 数据库备份, healthz]
---

# 全栈生产部署与日常运维

对于开启了用户登录、社区论坛、课程权益与在线支付的生产站点，推荐使用仓库内置的 `docker-compose.yml` + `docker-compose.prod.yml` 编排运行。

## 1. 标准启动流程

```bash
# 1. 准备生产环境变量
cp .env.example .env
# 必填：BASE_URL、JWT_SECRET（>=32 字符）、POSTGRES_PASSWORD

# 2. 启动应用与 PostgreSQL 16（自动完成数据库表结构迁移）
docker compose up -d

# 3. 检查服务日志与健康状态
docker compose logs -f app
curl -fsS http://127.0.0.1:8080/healthz
```

## 2. 开源框架与私有内容双仓库分离（推荐最佳实践）

为了让你随时 `git pull` 升级上游 DioxusPress 框架而不产生内容合并冲突，推荐将你站点的真实文档、付费课程、博客与专属品牌图片独立存放在一个私有 Git 仓库中：

```text
/srv/my-site-content/
├── site.json          # 站点专属配置（构建前覆盖框架默认 site.json）
├── images/            # 站点专属 Logo、Favicon、OG 分享图（运行时自动优先命中）
├── docs/              # 全部文档分卷
├── posts/             # 全部博客文章与 authors.yml
└── courses/           # 私有付费课程
```

在容器启动时将 `/srv/my-site-content` 只读挂载到 `/app/content`（设置 `CONTENT_DIR=/app/content`）：
- **静态品牌图覆盖**：请求 `/images/*` 时，服务端优先返回 `$CONTENT_DIR/images/` 下的同名图片，不存在时平滑回退到框架内置图标；
- **纯内容热发布**：日常新增文档或修改课程只需在宿主机同步私有内容仓库并重启容器，数秒内生效，无需重新跑 Cargo 编译。

## 3. 数据持久化卷与定时备份

生产环境中需要重点保护三类持久化数据：
1. **PostgreSQL 数据卷**：建议每日通过 `pg_dump -Fc` 导出压缩快照并同步至异地对象存储；
2. **用户上传图片卷（`/app/assets/uploads`）**：保存论坛发帖与评论中的用户上传配图；
3. **搜索索引卷（`/app/data/search-index`）**：可随时由源文件与数据库自动重建，挂载持久化卷可将容器重启时的索引校验缩短至毫秒级。
