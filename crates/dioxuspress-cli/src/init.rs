use std::fs;
use std::path::Path;

const SCHEMA_URL: &str = "https://dioxuspress.dev/schemas/site.schema.json";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Template {
  Docs,
  Blog,
  Community,
  Portal,
}

impl Template {
  pub fn parse(raw: &str) -> Result<Self, String> {
    match raw.trim().to_ascii_lowercase().as_str() {
      "docs" => Ok(Self::Docs),
      "blog" | "minimal" => Ok(Self::Blog),
      "community" => Ok(Self::Community),
      "portal" => Ok(Self::Portal),
      other => {
        Err(format!("unknown template '{other}' (expected one of: docs, blog, community, portal)"))
      }
    }
  }

  pub const fn id(self) -> &'static str {
    match self {
      Self::Docs => "docs",
      Self::Blog => "blog",
      Self::Community => "community",
      Self::Portal => "portal",
    }
  }

  pub const fn uses_database(self) -> bool {
    matches!(self, Self::Community | Self::Portal)
  }
}

pub fn init_site(target_dir: &Path, template: Template, force: bool) -> Result<(), String> {
  let site_json_file = target_dir.join("site.json");
  if site_json_file.exists() && !force {
    return Err(format!("{} already exists (pass --force to overwrite)", site_json_file.display()));
  }

  let raw_name = target_dir
    .file_name()
    .and_then(|s| s.to_str())
    .filter(|s| !s.is_empty() && *s != ".")
    .unwrap_or("My DioxusPress Site");
  let site_name = humanize_site_name(raw_name);

  fs::create_dir_all(target_dir.join("docs/guide/getting-started"))
    .map_err(|e| format!("failed to create docs directory: {e}"))?;
  fs::create_dir_all(target_dir.join("posts/hello-dioxuspress"))
    .map_err(|e| format!("failed to create posts directory: {e}"))?;
  fs::create_dir_all(target_dir.join("images"))
    .map_err(|e| format!("failed to create images directory: {e}"))?;

  write_text(&site_json_file, &build_site_json(&site_name, template)?)?;
  write_text(&target_dir.join("docs/index.md"), &docs_root_md(&site_name))?;
  write_text(&target_dir.join("docs/guide/index.md"), &docs_guide_index_md(&site_name))?;
  write_text(
    &target_dir.join("docs/guide/getting-started/index.md"),
    &docs_getting_started_md(&site_name),
  )?;
  write_text(&target_dir.join("posts/hello-dioxuspress/index.md"), &sample_post_md(&site_name))?;
  write_text(&target_dir.join("images/.gitkeep"), "")?;
  write_text(&target_dir.join("docker-compose.yml"), &docker_compose_yml(template))?;
  write_text(&target_dir.join(".gitignore"), gitignore_txt())?;
  write_text(&target_dir.join("README.md"), &readme_md(&site_name, template))?;

  if template.uses_database() {
    write_text(&target_dir.join(".env.example"), env_example_txt())?;
  }

  Ok(())
}

fn write_text(path: &Path, content: &str) -> Result<(), String> {
  fs::write(path, content).map_err(|e| format!("failed to write {}: {e}", path.display()))
}

fn humanize_site_name(slug: &str) -> String {
  let words: Vec<String> = slug
    .split(['-', '_'])
    .filter(|w| !w.is_empty())
    .map(|w| {
      let mut chars = w.chars();
      match chars.next() {
        Some(first) => format!("{}{}", first.to_uppercase(), chars.as_str()),
        None => String::new(),
      }
    })
    .collect();
  if words.is_empty() {
    "My DioxusPress Site".to_string()
  } else {
    words.join(" ")
  }
}

fn build_site_json(site_name: &str, template: Template) -> Result<String, String> {
  let value = match template {
    Template::Docs => serde_json::json!({
      "$schema": SCHEMA_URL,
      "site_name": site_name,
      "site_description": format!("{site_name} — 基于 DioxusPress 构建的极速技术文档站"),
      "default_language": "zh",
      "active_layout": "docs",
      "theme": "emerald",
      "available_themes": ["emerald", "ocean", "nord", "corporate", "dracula"],
      "hero": {
        "badge": "Zero-DB · Full-Text Search · Dual-Mode Themes",
        "title": site_name,
        "subtitle": "编辑 Markdown 即可实时生成三级文档侧栏、代码高亮、Mermaid 架构图与毫秒级全文检索。",
        "primary_cta_text": "阅读文档",
        "primary_cta_href": "/docs/guide/getting-started",
        "secondary_cta_text": "浏览博客",
        "secondary_cta_href": "/blog"
      },
      "features": [
        {
          "icon": "book",
          "title": "目录即文档树",
          "description": "按文件夹层级自动生成三级侧栏、面包屑与上下篇导航，零手工路由配置。",
          "href": "/docs/guide/getting-started"
        },
        {
          "icon": "search",
          "title": "毫秒级中英检索",
          "description": "内置 Tantivy + Jieba 分词引擎，Cmd+K 随时唤起全站搜索。",
          "href": "/docs"
        },
        {
          "icon": "palette",
          "title": "双模主题色卡",
          "description": "全部内置主题均通过 WCAG AA 对比度校验，支持亮暗双态无闪烁切换。"
        }
      ],
      "modules": {
        "docs": { "enabled": true },
        "blog": { "enabled": true },
        "cases": { "enabled": false },
        "podcast": { "enabled": false },
        "embedded": { "enabled": false },
        "ai": { "enabled": false },
        "web3": { "enabled": false },
        "wasm": { "enabled": false },
        "cli": { "enabled": false },
        "about": { "enabled": false }
      },
      "footer": {
        "tagline": format!("{site_name} · Powered by DioxusPress"),
        "copyright": site_name
      }
    }),
    Template::Blog => serde_json::json!({
      "$schema": SCHEMA_URL,
      "site_name": site_name,
      "site_description": format!("{site_name} — 专注深度写作与长文阅读的独立博客"),
      "default_language": "zh",
      "active_layout": "minimal",
      "theme": "coffee",
      "available_themes": ["coffee", "silk", "nord", "ocean"],
      "hero": {
        "title": site_name,
        "subtitle": "极简阅读顶栏与暖调护眼配色，内置 LaTeX 公式、代码对比标签页与 Atom 订阅源。"
      },
      "modules": {
        "blog": { "enabled": true },
        "docs": { "enabled": true },
        "cases": { "enabled": false },
        "podcast": { "enabled": false },
        "embedded": { "enabled": false },
        "ai": { "enabled": false },
        "web3": { "enabled": false },
        "wasm": { "enabled": false },
        "cli": { "enabled": false },
        "about": { "enabled": false }
      },
      "footer": {
        "copyright": site_name
      }
    }),
    Template::Community => serde_json::json!({
      "$schema": SCHEMA_URL,
      "site_name": site_name,
      "site_description": format!("{site_name} — 集成文档、博客、OAuth 登录与话题讨论的技术社区"),
      "default_language": "zh",
      "active_layout": "classic",
      "theme": "ocean",
      "hero": {
        "title": site_name,
        "subtitle": "一个二进制同时承载知识库、专栏文章、论坛交流与段落划线标注。"
      },
      "auth": {
        "enabled": true,
        "providers": ["github", "google"]
      },
      "annotations": {
        "doc": true,
        "blog": true,
        "course": true
      },
      "modules": {
        "docs": { "enabled": true },
        "blog": { "enabled": true },
        "forum": { "enabled": true },
        "cases": { "enabled": false },
        "podcast": { "enabled": false },
        "embedded": { "enabled": false },
        "ai": { "enabled": false },
        "web3": { "enabled": false },
        "wasm": { "enabled": false },
        "cli": { "enabled": false }
      },
      "footer": {
        "tagline": format!("{site_name} 技术交流社区"),
        "copyright": site_name
      }
    }),
    Template::Portal => serde_json::json!({
      "$schema": SCHEMA_URL,
      "site_name": site_name,
      "site_description": format!("{site_name} — 文档、博客、实战课程与生态案例一站式知识门户"),
      "default_language": "zh",
      "active_layout": "classic",
      "theme": "corporate",
      "hero": {
        "title": site_name,
        "subtitle": "汇聚结构化文档卷、实战课程体系、开源案例库与技术讨论。"
      },
      "auth": {
        "enabled": true,
        "providers": ["github", "google", "discord"]
      },
      "annotations": {
        "doc": true,
        "blog": true,
        "course": true
      },
      "moderation": {
        "enabled": false,
        "llm_review": false,
        "on_llm_failure": "review"
      },
      "footer": {
        "tagline": format!("{site_name} 全栈知识门户"),
        "copyright": site_name
      }
    }),
  };

  let mut out = serde_json::to_string_pretty(&value)
    .map_err(|e| format!("failed to serialize site.json: {e}"))?;
  out.push('\n');
  Ok(out)
}

fn docs_root_md(site_name: &str) -> String {
  format!(
    "# {site_name} 文档中心\n\n欢迎来到 **{site_name}**！请从左侧目录选择章节开始阅读，或直接前往 [快速开始](/docs/guide/getting-started)。\n"
  )
}

fn docs_guide_index_md(site_name: &str) -> String {
  format!(
    "---\ntitle: 使用指南\nsidebar_label: 使用指南\nsidebar_position: 1\ndescription: {site_name} 核心概念、目录结构与日常写作指南。\n---\n\n# 使用指南\n\n本卷介绍如何组织 **{site_name}** 的文档与配置。先阅读 [快速开始](/docs/guide/getting-started) 了解基本工作流。\n"
  )
}

fn docs_getting_started_md(site_name: &str) -> String {
  format!(
    "---\ntitle: 快速开始\nsidebar_label: 快速开始\nsidebar_position: 1\ndescription: 在 1 分钟内预览、编辑并校验 {site_name} 的 Markdown 内容。\nkeywords: [DioxusPress, 快速开始, Markdown]\n---\n\n# 快速开始\n\n恭喜你初始化了 **{site_name}**！当前目录中的所有内容都与引擎代码解耦。\n\n## 目录约定\n\n- `site.json`：站点名称、布局、主题色卡与导航模块开关（支持 VS Code / Zed 自动补全）\n- `docs/`：多卷三级技术文档（自动生成左侧树与上下篇链接）\n- `posts/`：博客文章（每篇一个子目录，内含 `index.md` 或 `index.mdx`）\n- `images/`：自定义 Logo、Favicon 与配图（通过 `/images/<文件名>` 引用）\n\n:::tip 提交前静态体检\n在发布或推送 Git 前运行 `dioxuspress check`，可在毫秒级检查字段拼写、代码块闭合与站内死链。\n:::\n\n也可以查看示例文章：[你好，DioxusPress](/blog/hello-dioxuspress)。\n"
  )
}

fn sample_post_md(site_name: &str) -> String {
  format!(
    "---\ntitle: 你好，{site_name}\ndescription: 第一篇由 dioxuspress init 生成的示例文章，支持代码高亮、提示框与全文检索。\ndate: '2026-01-01'\ntags:\n  - 公告\n  - DioxusPress\n---\n\n# 你好，{site_name}\n\n这是你的第一篇博客文章！编辑 `posts/hello-dioxuspress/index.md` 即可修改正文，或前往 [文档快速开始](/docs/guide/getting-started) 了解更多排版语法。\n\n```rust\nfn main() {{\n  println!(\"Hello from {site_name}!\");\n}}\n```\n"
  )
}

fn docker_compose_yml(template: Template) -> String {
  if !template.uses_database() {
    "services:\n  site:\n    image: ghcr.io/dioxuspress/dioxuspress-basic:latest\n    ports:\n      - \"8080:8080\"\n    environment:\n      BASE_URL: http://localhost:8080\n      SEARCH_INDEX_DIR: /app/data/search-index\n    volumes:\n      - ./:/app/content:ro\n      - search-index:/app/data/search-index\n    restart: unless-stopped\n\nvolumes:\n  search-index:\n".to_string()
  } else {
    "services:\n  postgres:\n    image: postgres:16-alpine\n    environment:\n      POSTGRES_USER: dioxuspress\n      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-change_me_in_env}\n      POSTGRES_DB: dioxuspress\n    volumes:\n      - pg-data:/var/lib/postgresql/data\n    healthcheck:\n      test: [\"CMD-SHELL\", \"pg_isready -U dioxuspress\"]\n      interval: 5s\n      timeout: 3s\n      retries: 10\n    restart: unless-stopped\n\n  site:\n    image: ghcr.io/dioxuspress/dioxuspress-app:latest\n    depends_on:\n      postgres:\n        condition: service_healthy\n    ports:\n      - \"8080:8080\"\n    env_file:\n      - .env\n    environment:\n      SEARCH_INDEX_DIR: /app/data/search-index\n    volumes:\n      - ./:/app/content:ro\n      - search-index:/app/data/search-index\n    restart: unless-stopped\n\nvolumes:\n  pg-data:\n  search-index:\n".to_string()
  }
}

fn env_example_txt() -> &'static str {
  "# 复制为 .env 并替换以下密钥（可用 `openssl rand -hex 32` 生成）\nBASE_URL=http://127.0.0.1:8080\nPOSTGRES_PASSWORD=please_change_this_db_password\nDATABASE_URL=postgres://dioxuspress:please_change_this_db_password@postgres:5432/dioxuspress\nJWT_SECRET=please_replace_with_32_byte_random_hex_secret_value\n\n# 可选：GitHub OAuth 登录凭据（回调地址：{BASE_URL}/api/auth/github/callback）\nGITHUB_CLIENT_ID=\nGITHUB_CLIENT_SECRET=\n"
}

fn gitignore_txt() -> &'static str {
  ".DS_Store\n.env\ndata/\n"
}

fn readme_md(site_name: &str, template: Template) -> String {
  let preset = template.id();
  if !template.uses_database() {
    format!(
      "# {site_name}\n\n基于 [DioxusPress](https://dioxuspress.dev)（`{preset}` 零数据库预设）创建的内容仓库。\n\n## 本地启动（无需安装 Rust 编译环境）\n\n```bash\n# 1. 静态检查配置与 Markdown 链接\ndioxuspress check\n\n# 2. 启动官方预构建容器\ndocker compose up -d\n```\n\n打开浏览器访问 <http://localhost:8080>。\n"
    )
  } else {
    format!(
      "# {site_name}\n\n基于 [DioxusPress](https://dioxuspress.dev)（`{preset}` 全栈社区预设）创建的内容仓库。\n\n## 本地启动\n\n```bash\n# 1. 复制并配置密钥环境变量\ncp .env.example .env\n\n# 2. 静态检查配置与 Markdown 链接\ndioxuspress check\n\n# 3. 启动 PostgreSQL 与 DioxusPress 容器\ndocker compose up -d\n```\n\n打开浏览器访问 <http://127.0.0.1:8080>。\n"
    )
  }
}
