---
title: 零数据库轻量部署指南：单容器运行高性能 DioxusPress 文档与博客站
sidebar_label: 零数据库轻量部署
sidebar_position: 1
description: 使用 APP_FEATURES="--no-default-features" 构建无数据库依赖的轻量 Docker 镜像，配合网关绑定独立域名或二级域名。
keywords: [零数据库部署, basic 档位, 单容器部署, Docusaurus 替代部署, Docker 镜像]
---

# 零数据库轻量部署指南

如果你使用 DioxusPress 搭建类似 Docusaurus 的技术文档站、产品官网或纯静态内容博客，**完全不需要部署 PostgreSQL，也不需要配置 `JWT_SECRET` 或任何 OAuth 密钥**。

## 1. 构建 `basic` 档位轻量镜像

仓库根目录的 `Dockerfile` 原生支持 `APP_FEATURES` 构建参数：

```bash
docker build   --build-arg APP_FEATURES="--no-default-features"   -t dioxuspress:basic .
```

该镜像基于 `debian:trixie-slim` 并以非特权用户 `app` 运行，内部已包含 SSR 服务端二进制、预压缩 WASM 前端产物以及 `assets/` 内容资源。

## 2. 一条命令启动容器

只需传入公开访问的 `BASE_URL`（用于生成准确的 `<link rel="canonical">`、`og:url` 与 `sitemap.xml`）：

```bash
docker run -d   --name dioxuspress-docs   --restart unless-stopped   -p 127.0.0.1:8081:8080   -e BASE_URL=https://docs.example.com   -v dioxuspress-search:/app/data/search-index   dioxuspress:basic
```

:::tip
如果你希望在不重新编译镜像的情况下随时修改文档 Markdown，只需将宿主机上的内容目录只读挂载进去并指定 `-v /srv/my-docs:/app/content:ro -e CONTENT_DIR=/app/content`，更新文章后 `docker restart dioxuspress-docs` 即可在 1 秒内完成增量索引刷新。
:::

## 3. 验证运行状态

```bash
# 健康检查（basic 档位下 db 字段返回 "disabled" 属于预期正常状态）
curl -fsS http://127.0.0.1:8081/healthz

# 验证自动生成的站点地图与 RSS 源
curl -fsS http://127.0.0.1:8081/sitemap.xml | head -n 10
```

最后在宿主机的反向代理（Pingora / Caddy / Nginx）中将你的域名（或二级域名）转发至 `127.0.0.1:8081` 并开启自动 HTTPS 证书签发即可正式上线。
