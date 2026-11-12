# DioxusPress Community Starter

基于 [DioxusPress](https://dioxuspress.dev)（`community` 技术社区与讨论站预设）创建的内容仓库。

## 特性亮点

- **知识库 + 论坛一体化**：文档、博客、话题讨论区（`/topics`）与文章评论同库联动。
- **OAuth2 PKCE 安全登录**：原生支持 GitHub、Google 与 Discord 登录，未配置密钥的渠道自动隐藏。
- **段落划线批注与内容治理**：支持文档与博客正文划词批注，内置恶意域名黑名单与 `/admin` 治理后台。

## 快速启动

```bash
# 1. 复制环境变量并替换随机密钥
cp .env.example .env

# 2. 离线检查配置与站内链接
dioxuspress check

# 3. 启动 PostgreSQL 16 与社区主容器
docker compose up -d
```

打开浏览器访问 <http://127.0.0.1:8080>。
