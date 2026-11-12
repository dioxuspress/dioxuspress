# DioxusPress Portal Starter

基于 [DioxusPress](https://dioxuspress.dev)（`portal` 全栈知识与商业化门户预设）创建的内容仓库，对齐 [Rust in Everything](https://rustineverything.app) 生产最佳实践。

## 特性亮点

- **五大内容模块全开**：预置结构化文档（`docs/`）、MDX 专栏（`posts/`）、试读与付费课程（`courses/`）、精选案例库（`cases/`）与技术播客（`podcasts/`）。
- **商业化与互动闭环**：支持免费试看课节与服务端严密付费守卫、全站段落划线标注、话题论坛及可选 Stripe / 微信支付接入。
- **企业级稳重视觉**：默认启用 `corporate` 双模商务配色，开放 6 套 WCAG AA 色卡切换。

## 快速启动

```bash
cp .env.example .env
dioxuspress check
docker compose up -d
```

打开浏览器访问 <http://127.0.0.1:8080>。
