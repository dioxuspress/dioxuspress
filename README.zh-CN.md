# DioxusPress

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/images/logo-dark.png" />
    <source media="(prefers-color-scheme: light)" srcset="assets/images/logo-light.png" />
    <img src="assets/images/logo-light.png" alt="DioxusPress" height="64" />
  </picture>
</p>

<p align="center">
  <a href="README.md">English</a> · <strong>简体中文</strong>
</p>

基于 [Dioxus 0.7](https://dioxuslabs.com) 的纯 Rust 全栈文档、内容与社区引擎：既能以**零数据库 `basic` 档位**充当 Docusaurus / VitePress 的高性能替代方案，也能按需开启登录、论坛、划线标注、付费课程与在线支付。

- **官方网站与文档中心**：[`dioxuspress.dev`](https://dioxuspress.dev)（`active_layout = "docs"`，零数据库单容器运行）
- **UI 组件库在线预览**：[`ui.dioxuspress.dev`](https://ui.dioxuspress.dev)（[`dioxus-shadcn`](https://github.com/dioxuspress/dioxus-ui) — 84 个 Dioxus 0.7 组件）
- **旗舰全栈社区实例**：[Rust in Everything (`rustineverything.app`)](https://rustineverything.app)（`active_layout = "classic"`，承载 38 卷 2,100+ 篇文档、实战课程与社区）

开发者只需修改 [`assets/site.json`](assets/site.json) 中的 `site_name`、`active_layout`（`docs` / `classic` / `minimal`）、`theme` 与 `features` 等字段，即可快速定制并部署自己的文档站或全栈社区。

## 技术栈

- **全栈 Rust**：Dioxus 0.7 fullstack（SSR + 水合）+ Axum，SeaORM + PostgreSQL（`basic` 档位免数据库）。
- **UI 组件**：[`dioxus-shadcn`](https://github.com/dioxuspress/dioxus-ui) 0.6（自研，shadcn/ui 的 Dioxus 实现，[预览站](https://ui.dioxuspress.dev)）。
  全站界面基于它构建，shadcn token 是唯一的配色来源，石墨（zinc）中性色 + 青色主色。
- **样式**：Tailwind CSS v4 + 深色模式；内置主题 ocean / sunset / catppuccin，访客可切换。
- **安全**：严格 CSP（无 `'unsafe-eval'`），用户内容经白名单清洗，删除类管理操作先确认。
- **全站搜索**：Tantivy 嵌入式搜索引擎，支持中英文分词。

## 功能

- **内容**（文件系统为单一来源，作者通过 git 写作，`assets/` 下）：
  - 博客：MDX（frontmatter、GFM、数学公式、Mermaid、代码高亮、嵌入组件）。
  - 文档：三级文档树，支持面包屑导航、大纲锚点与 GitHub 编辑直达。
  - 课程：课程 → 章节 → 课节，支持免费 / 付费课节、Pro 会员与在线支付（`payments` 特性）。
  - 案例展示、播客，以及 Rust 生态 / AI 生态下的嵌入式、AI、WebAssembly、Web3、CLI、论文板块。
- **互动**：论坛（话题 / 回复，可从文章发起讨论）、评论、段落标注、图片上传。
- **内容审核**：链接黑名单 + 可选 LLM 审核，可疑内容进入人工复核队列。
- **登录**：GitHub / Google / Discord / X（OAuth2 + PKCE），JWT 会话。
- **管理后台**（`/admin`，仅 admin 可见）：概览、用户与角色、评论、话题、审核队列、审核设置、课程权益。

主题、登录方式、审核都是编译进宿主的普通 Rust 代码，不加载运行时插件。

## 功能分层

同一套代码按 Cargo feature 编译出四档站点（设计见 [docs/FEATURE_TIERS.md](docs/FEATURE_TIERS.md)）：

| 档位 | feature 参数 | 内容 | 需要 |
| --- | --- | --- | --- |
| basic | `--no-default-features` | 博客、文档、播客、案例、专题、搜索、About、布局 / 主题 / 明暗 / 中英 | 只要 `BASE_URL`，不用数据库 |
| 账号 | `--no-default-features --features accounts` | + 登录、论坛、评论、上传、管理后台 | + Postgres、`JWT_SECRET` |
| 课程 | `--no-default-features --features course` | + 课程、标注、权益后台、Pro 会员 | 同上 |
| 完整（默认） | 无 | + 在线支付（微信 / 支付宝） | + 商户密钥 |

本地运行某一档：`cd crates/app && dx serve --web --no-default-features --features accounts`（不用默认 feature 时要加 `--web`，否则 dx 判断不出平台）。

## 快速开始

```bash
# 0. 克隆仓库
git clone https://github.com/dioxuspress/dioxuspress.git
cd dioxuspress

# 1. 安装 Dioxus CLI
cargo install dioxus-cli

# 2. 配置站点名称与模块（按需修改 assets/site.json 的 site_name、active_layout、theme 等）
#    配置环境变量（DATABASE_URL、JWT_SECRET、BASE_URL、OAuth 凭据等）
cp .env.example .env

# 3. 启动 PostgreSQL（应用启动时自动执行数据库迁移；basic 档位可跳过）
docker compose up -d postgres

# 4. 安装并编译 Tailwind CSS（首次，或新增了类名之后）
cd crates/app && npm install && npm run build

# 5. 启动开发服务器（在 crates/app 目录下）
dx serve
```

浏览器访问 `.env` 中 `BASE_URL` 对应的地址（如 `http://127.0.0.1:8080`）。
用 `localhost` 访问会因 OAuth 回调的 cookie 不匹配而登录失败。

首个管理员：用 OAuth 登录一次后，在仓库根目录运行 `set -a; source .env; set +a; scripts/promote_admin.sh <昵称>`（需要 `psql`），
再退出并重新登录，新签发的会话才带 admin 角色。详见 [管理后台](docs/ADMIN_SPEC.md)。

## 实测性能与资源占用

所有数据均可通过仓库内置脚本 [`./scripts/bench_ssr.sh`](scripts/bench_ssr.sh) 与 [`./scripts/check_wasm_size.sh`](scripts/check_wasm_size.sh) 本地复现（详见在线文档 [`dioxuspress.dev/docs/operations/benchmarks`](https://dioxuspress.dev/docs/operations/benchmarks)）：

| 指标 | `basic` 零数据库档 | `accounts` / `full` 全栈档 | 说明 |
| --- | --- | --- | --- |
| **服务端二进制 / 容器镜像** | `34.2 MiB` / `269 MB` | `47.4 MiB` / `319 MB` | Pingora 多域名 SNI 网关镜像 `143 MB`（常驻内存仅 `6.4 MiB`） |
| **冷启动就绪耗时（`/healthz 200`）** | `547 ms` | `549 ms` | 毫秒级对外响应，Tantivy 差量校验异步执行 |
| **空闲内存占用（Idle RSS）** | `91.7 MiB` | `95.6 MiB` | 含内嵌结巴中文分词词典与 Tantivy 内存映射索引 |
| **首屏完整 SSR（`/` / `/docs/*`）** | `2.79 ms` / `4.04 ms` P50（`1,832` / `1,236 QPS`） | `2.90 ms` / `4.02 ms` P50（`1,764` / `1,234 QPS`） | 实时 Markdown 解析 + 三级侧栏 + VirtualDOM 渲染 |
| **匿名条件缓存（`ETag` → `304`）** | **`0.096 ms` P50（`13,636 QPS`）** | **`0.093 ms` P50（`13,462 QPS`）** | 仅限纯匿名只读请求；登录态与私密路径严格 `private, no-store` |
| **Tantivy 全文检索（`/api/search/query`）** | **`0.314 ms` P50（`9,215 QPS`）** | **`0.317 ms` P50（`9,326 QPS`）** | 中英 BM25 + 摘要提取；支持 `SIGHUP` 零停机增量热刷新 |
| **前端 WASM 传输体积（`brotli -q 11`）** | **`1.04 MiB`**（`1.54 MiB` gzip） | **`1.35 MiB`**（`2.07 MiB` gzip） | CI 每次 PR 执行 `scripts/check_wasm_size.sh` 硬性体积门禁 |

## 提交前检查

```bash
cargo fmt --all -- --check
cargo clippy --workspace --features server --all-targets -- -D warnings
cargo test --features server --workspace -- --test-threads=1
cargo check -p app --target wasm32-unknown-unknown --features web
scripts/check_tiers.sh      # 四档 × server / web 都要编译通过且无 warning
scripts/check_wasm_size.sh  # 校验 wasm-release 前端包体积未超预算
```

## Tailwind CSS 构建

Tailwind 源文件和 npm 工具链位于 `crates/app/` 下：

- **源文件**：`crates/app/tailwind-input.css`（含 `@import "tailwindcss"` 和 `@source` 指令，也扫描 dioxus-shadcn 组件里的类名）
- **输出**：`crates/app/assets/tailwind.css`（不入库，需本地构建）→ 自动反向同步到 `assets/tailwind.css`
- **开发 Watch**：`cd crates/app && npm run dev`
- **一次性构建**：`cd crates/app && npm run build`

详见 [Tailwind CSS 指南](docs/TAILWIND_GUIDE.md)。

## 文档

- [在线官方文档 (`dioxuspress.dev/docs`)](https://dioxuspress.dev/docs)
- [开发者指南](docs/DEVELOPER.md)
- [架构评估总结报告](docs/ARCHITECTURE_ASSESSMENT.md)
- [引擎](docs/ENGINES_SPEC.md) / [模块](docs/MODULE_SPEC.md)
- [dioxus-shadcn 迁移](docs/DIOXUS_UI_MIGRATION.md) / [组件库反馈](docs/DIOXUS_UI_FEEDBACK.md)
- [MDX 内容](docs/MDX_SPEC.md) / [博客](docs/BLOG_SPEC.md) / [文档模块](docs/DOCS_MODULE_SPEC.md) / [课程](docs/COURSE_SPEC.md) / [支付](docs/PAYMENT_SPEC.md)
- [论坛](docs/FORUM_SPEC.md) / [评论](docs/COMMENTS_SPEC.md) / [标注](docs/ANNOTATION_SPEC.md) / [上传](docs/UPLOADS_SPEC.md)
- [内容审核](docs/MODERATION_SPEC.md) / [管理后台](docs/ADMIN_SPEC.md)
- [认证系统](docs/AUTH_SPEC.md) / [认证配置](docs/AUTH_GUIDE.md) / [会话](docs/SESSION_SPEC.md)
- [主题](docs/THEME_SPEC.md) / [搜索](docs/SEARCH_SPEC.md) / [SEO](docs/SEO_SPEC.md)
- [部署](docs/DEPLOY_GUIDE.md) / [运维](docs/OPERATIONS.md) / [安全整改记录](docs/SECURITY_REMEDIATION.md)

## 开源协议

[MIT](LICENSE)
