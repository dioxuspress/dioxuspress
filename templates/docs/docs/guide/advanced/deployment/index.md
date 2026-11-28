---
title: 生产环境部署：只读挂载与 HTTPS 反向代理
sidebar_label: 生产环境部署
sidebar_position: 1
description: 在云服务器上以只读卷挂载运行 dioxuspress-basic 容器，并开启持久化全文索引。
keywords: [生产部署, Docker, HTTPS]
---

# 生产环境部署

在生产服务器上，只需将 `BASE_URL` 改为你的正式 HTTPS 域名即可生成准确的 Canonical 链接、`/sitemap.xml` 与 `/feed.xml`：

```yaml
services:
  site:
    image: ghcr.io/dioxuspress/dioxuspress-basic:latest
    ports:
      - "127.0.0.1:8080:8080"
    environment:
      BASE_URL: https://docs.example.com
      SEARCH_INDEX_DIR: /app/data/search-index
    volumes:
      - ./:/app/content:ro
      - search-index:/app/data/search-index
    restart: unless-stopped

volumes:
  search-index:
```

接下来可以继续阅读第二卷：[API 与 CLI 参考](/docs/api)。
