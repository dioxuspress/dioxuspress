# DioxusPress Starter Templates

本目录存放 DioxusPress 官方维护的四套场景化建站预设，同时作为 `dioxuspress init <dir> --template <preset>` 的编译期唯一真实数据源（Single Source of Truth）：

| 模板 ID | 目录 | 适用场景 | 推荐镜像 / 数据依赖 | 默认布局与主题 |
| :--- | :--- | :--- | :--- | :--- |
| **`docs`** | [`./docs`](./docs) | 开源库文档、API 手册、团队内部 Wiki | `dioxuspress-basic`（零数据库） | `docs` + `emerald` |
| **`blog`** | [`./blog`](./blog) | 独立技术博客、数学/系统专栏、数字花园 | `dioxuspress-basic`（零数据库） | `minimal` + `coffee` |
| **`community`** | [`./community`](./community) | 开源社区、SIG 讨论组、共创知识库 | `dioxuspress-app` + PostgreSQL 16 | `classic` + `ocean` |
| **`portal`** | [`./portal`](./portal) | 商业化付费课程、案例展示与综合知识门户 | `dioxuspress-app` + PostgreSQL 16 | `classic` + `corporate` |

## 一键初始化新站点

```bash
# 任选一套预设生成独立内容仓库（自动将站点名替换为目录名）
dioxuspress init my-site --template docs
cd my-site
dioxuspress check
docker compose up -d
```
