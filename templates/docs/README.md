# DioxusPress Docs Starter

基于 [DioxusPress](https://dioxuspress.dev)（`docs` 零数据库开源文档预设）构建的技术文档站内容仓库。

## 特性亮点

- **零数据库单容器**：直接挂载至 `ghcr.io/dioxuspress/dioxuspress-basic:latest`，空闲内存仅 ~30 MB。
- **多卷三级目录树**：内置 `docs/guide` 与 `docs/api` 两套分卷及三级嵌套侧栏示例。
- **双模主题与全文搜索**：默认启用 `emerald` 技术文档配色与嵌入式 Tantivy 中英文毫秒级检索。
- **CI 自动防腐烂**：附带 `.github/workflows/check.yml`，自动校验 `site.json`、代码块闭合与站内死链。

## 快速启动

```bash
# 1. 本地离线静态体检（可选）
dioxuspress check

# 2. 启动预览服务
docker compose up -d
```

打开浏览器访问 <http://localhost:8080>。
