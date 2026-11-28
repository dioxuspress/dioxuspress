//! Phase E4.2：`TemplatesPage` — 官方 Starter 模板中心展示与一键选用页（`/templates`）。
//!
//! 展示四套场景化预设（`docs` / `blog` / `community` / `portal`）：
//! - 真实桌面浏览器亮色 / 暗色双极性高保真界面缩略预览（零外部大图负载、Retina 矢量清晰）；
//! - 适用场景、默认布局、内置主题色卡与一键全站实时试穿；
//! - 零数据库单容器（`dioxuspress-basic`）与全栈社区（`dioxuspress-app` + PostgreSQL 16）资源占用对比；
//! - 一键复制 `dioxuspress init` + `docker compose up -d` 启动命令。

use app_core::engines::theme::THEMES;
use app_core::i18n::Language;
use dioxus::prelude::*;
use dioxus::router::Link;

use crate::components::theme_picker::{apply_user_theme, use_theme_version};
use crate::components::view::Container;
use crate::i18n::use_i18n;

struct PresetSpec {
  id: &'static str,
  sample_dir: &'static str,
  name_zh: &'static str,
  name_en: &'static str,
  tier_badge_zh: &'static str,
  tier_badge_en: &'static str,
  uses_db: bool,
  layout_id: &'static str,
  theme_id: &'static str,
  summary_zh: &'static str,
  summary_en: &'static str,
  audience_zh: &'static str,
  audience_en: &'static str,
  highlights_zh: &'static [&'static str],
  highlights_en: &'static [&'static str],
}

const PRESETS: &[PresetSpec] = &[
  PresetSpec {
    id: "docs",
    sample_dir: "my-docs",
    name_zh: "开源项目文档站",
    name_en: "Open-Source Docs Hub",
    tier_badge_zh: "零数据库 · basic",
    tier_badge_en: "Zero-DB · basic",
    uses_db: false,
    layout_id: "docs",
    theme_id: "emerald",
    summary_zh:
      "类 Docusaurus 左对齐顶栏 + 可关闭发布横幅 + 多卷三级折叠侧栏 + 嵌入式 Tantivy 毫秒级搜索，内置 GitHub Actions 死链与语法自动体检。",
    summary_en:
      "Docusaurus-style left-aligned navbar, dismissible release banner, multi-volume 3-level sidebar, embedded Tantivy search, and GitHub Actions CI check.",
    audience_zh: "开源库文档、SDK / API 参考手册、团队内部技术 Wiki",
    audience_en: "Open-source library docs, SDK/API references, internal engineering wikis",
    highlights_zh: &[
      "多卷（guide/ + api/）与三级嵌套侧栏示例",
      "附带 .github/workflows/check.yml 自动门禁",
      "单容器 ~30 MB 内存，无需任何数据库或密钥",
    ],
    highlights_en: &[
      "Multi-volume (guide/ + api/) & 3-level sidebar tree",
      "Ships with .github/workflows/check.yml CI workflow",
      "Single container (~30 MB RSS), zero database or secrets",
    ],
  },
  PresetSpec {
    id: "blog",
    sample_dir: "my-blog",
    name_zh: "极简技术博客与数字花园",
    name_en: "Minimal Tech Blog & Digital Garden",
    tier_badge_zh: "零数据库 · basic",
    tier_badge_en: "Zero-DB · basic",
    uses_db: false,
    layout_id: "minimal",
    theme_id: "coffee",
    summary_zh:
      "收起复杂导航噪音，采用暖羊皮纸 / 护眼深咖双模配色，原生支持服务端 LaTeX → MathML 公式推导、多语言代码对比标签组与标准 Atom 订阅源。",
    summary_en:
      "Distraction-free minimal chrome with warm parchment/coffee colors, server-rendered MathML formulas, multi-language code tabs, and native Atom RSS.",
    audience_zh: "独立开发者长文博客、算法/系统工程专栏、常青卡片笔记",
    audience_en: "Indie developer essays, systems/math series, evergreen digital gardens",
    highlights_zh: &[
      "零客户端 JS 负担的服务端原生 MathML 数学公式",
      "突出展示 /feed.xml Atom 订阅源与 authors.yml 署名",
      "专栏长文（posts/）与常青笔记（docs/）双轨并存",
    ],
    highlights_en: &[
      "Zero-JS server-rendered LaTeX → MathML equations",
      "First-class /feed.xml Atom feed & authors.yml profiles",
      "Chronological essays (posts/) + evergreen notes (docs/)",
    ],
  },
  PresetSpec {
    id: "community",
    sample_dir: "my-community",
    name_zh: "技术社区与共创讨论站",
    name_en: "Developer Community & Forum",
    tier_badge_zh: "全栈社区 · PostgreSQL 16",
    tier_badge_en: "Fullstack · PostgreSQL 16",
    uses_db: true,
    layout_id: "classic",
    theme_id: "ocean",
    summary_zh:
      "一套部署编排串联 PostgreSQL 16 与社区应用：知识库、专栏文章、OAuth2 PKCE 安全登录、话题讨论区（/topics）与正文段落划线批注开箱即用。",
    summary_en:
      "Compose stack pairing PostgreSQL 16 with DioxusPress: docs, blog, OAuth2 PKCE sign-in, topic forum (/topics), and inline paragraph annotations.",
    audience_zh: "开源社区官网、技术 SIG 讨论组、团队共创知识库",
    audience_en: "Open-source communities, SIG discussion hubs, collaborative knowledge bases",
    highlights_zh: &[
      "GitHub / Google / Discord OAuth2 PKCE 登录指引",
      "文档与博客正文划词批注 + 底部关联讨论面板",
      "内置恶意链接域名黑名单与 /admin 审核管理后台",
    ],
    highlights_en: &[
      "GitHub / Google / Discord OAuth2 PKCE setup handbook",
      "Inline text annotations + linked forum threads on docs/blog",
      "Built-in URL blocklist & /admin moderation dashboard",
    ],
  },
  PresetSpec {
    id: "portal",
    sample_dir: "my-portal",
    name_zh: "全栈知识与付费专栏门户",
    name_en: "Pro Knowledge & Course Portal",
    tier_badge_zh: "商业闭环 · PostgreSQL 16",
    tier_badge_en: "Commercial · PostgreSQL 16",
    uses_db: true,
    layout_id: "classic",
    theme_id: "corporate",
    summary_zh:
      "对齐 Rust in Everything 生产架构：完整启用结构化文档、免费试读/付费锁定实战课程、精选案例库、MDX 内嵌播客卡片、论坛与可选 LLM 风控。",
    summary_en:
      "Production blueprint aligned with Rust in Everything: docs, preview & paid courses, case showcase, MDX podcast cards, forum, and optional LLM moderation.",
    audience_zh: "商业化技术专栏、在线实战训练营、大型生态门户",
    audience_en: "Commercial course platforms, technical bootcamps, flagship ecosystem portals",
    highlights_zh: &[
      "完整示范 course.yaml 定价、试读章节与服务端鉴权守卫",
      "预置 cases/ 精选案例库与 podcasts/ 音频卡片内嵌",
      "支持接入 Stripe / 微信支付及兼容 OpenAI 的自动审核",
    ],
    highlights_en: &[
      "Full course.yaml pricing, free preview & server entitlement guard",
      "Pre-wired cases/ showcase & <PodcastCard /> MDX embedding",
      "Ready for Stripe / WeChat Pay & OpenAI-compatible moderation",
    ],
  },
];

fn lookup_theme_colors(id: &str) -> (&'static str, &'static str, &'static str, &'static str) {
  THEMES
    .iter()
    .find(|t| t.id == id)
    .map(|t| (t.light_bg, t.light_primary, t.dark_bg, t.dark_primary))
    .unwrap_or(("#ffffff", "#0284c7", "#020617", "#38bdf8"))
}

fn init_command_for(preset: &PresetSpec) -> String {
  if preset.uses_db {
    format!(
      "dioxuspress init {} --template {} && cd {} && cp .env.example .env && docker compose up -d",
      preset.sample_dir, preset.id, preset.sample_dir
    )
  } else {
    format!(
      "dioxuspress init {} --template {} && cd {} && docker compose up -d",
      preset.sample_dir, preset.id, preset.sample_dir
    )
  }
}

/// `/templates` 路由页面主体。
#[component]
pub fn TemplatesPage() -> Element {
  let lang = use_i18n();
  let theme_version = use_theme_version();
  let mut preview_dark = use_signal(|| false);
  let mut copied_preset = use_signal::<Option<String>>(|| None);

  use_effect(move || {
    preview_dark.set(widgets::browser::dark_mode_preference());
  });

  let is_en = matches!(lang(), Language::En);
  let page_title = if is_en {
    "Starter Templates — Launch Docs, Blogs, Communities & Portals in 50 Seconds"
  } else {
    "官方 Starter 模板中心 — 50 秒初始化文档站、独立博客与全栈社区"
  };
  let page_desc = if is_en {
    "Explore 4 official DioxusPress starter templates (docs, blog, community, portal) with live light/dark previews, resource comparison, and one-command Docker setup."
  } else {
    "浏览 DioxusPress 四套官方场景化预设（docs、blog、community、portal）：支持亮暗双模桌面预览、零数据库与 PostgreSQL 资源占用对比及一行命令启动。"
  };

  rsx! {
    widgets::PageMeta {
      title: page_title.to_string(),
      description: page_desc.to_string(),
      path: "/templates".to_string(),
    }

    div { class: "py-12 sm:py-16",
      Container {
        // ── Hero 标题与亮暗预览切换栏 ──
        div { class: "flex flex-col lg:flex-row lg:items-end justify-between gap-6 pb-10 border-b border-slate-200 dark:border-slate-800",
          div { class: "max-w-3xl",
            span { class: "inline-flex items-center gap-1.5 rounded-full bg-primary/10 px-3 py-1 text-xs font-semibold text-primary mb-4",
              if is_en {
                "4 Official Presets · Single Source of Truth with CLI"
              } else {
                "4 套官方精调预设 · 与 dioxuspress init 同源同步"
              }
            }
            h1 { class: "text-3xl sm:text-4xl font-extrabold tracking-tight text-slate-900 dark:text-white",
              if is_en {
                "Pick a Starter Template for Your Scenario"
              } else {
                "按场景选择开箱即用的 Starter 模板"
              }
            }
            p { class: "mt-3 text-base sm:text-lg text-slate-600 dark:text-slate-300 leading-relaxed",
              if is_en {
                "Every preset ships with a schema-validated site.json, curated dual-mode palette, sample content, and Docker Compose stack — start with zero database and upgrade to fullstack anytime without rewriting Markdown."
              } else {
                "每套模板均内置通过 Schema 校验的 site.json、精选双模主题、真实示例内容与 Docker Compose 编排。你可以从零数据库极简起步，未来随时无缝升级至全栈社区而无需重写任何 Markdown。"
              }
            }
          }

          // 桌面缩略图亮色 / 暗色实时预览切换开关
          div { class: "inline-flex items-center self-start lg:self-auto rounded-lg border border-slate-200 dark:border-slate-800 bg-slate-100/80 dark:bg-slate-900 p-1 text-xs font-semibold shrink-0",
            button {
              r#type: "button",
              class: if !preview_dark() {
                "px-3 py-1.5 rounded-md bg-white text-slate-900 shadow-xs cursor-pointer transition-all"
              } else {
                "px-3 py-1.5 rounded-md text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white cursor-pointer transition-all"
              },
              onclick: move |_| preview_dark.set(false),
              if is_en { "☀️ Light Desktop" } else { "☀️ 桌面亮色预览" }
            }
            button {
              r#type: "button",
              class: if preview_dark() {
                "px-3 py-1.5 rounded-md bg-slate-800 text-white shadow-xs cursor-pointer transition-all"
              } else {
                "px-3 py-1.5 rounded-md text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white cursor-pointer transition-all"
              },
              onclick: move |_| preview_dark.set(true),
              if is_en { "🌙 Dark Desktop" } else { "🌙 桌面暗色预览" }
            }
          }
        }

        // ── 四套预设 2×2 卡片矩阵 ──
        div { class: "mt-10 grid grid-cols-1 lg:grid-cols-2 gap-8",
          for preset in PRESETS.iter() {
            {
              let preset_id = preset.id;
              let theme_for_try = preset.theme_id.to_string();
              let cmd = init_command_for(preset);
              let cmd_for_copy = cmd.clone();
              let is_copied = copied_preset().as_deref() == Some(preset_id);
              let name = if is_en { preset.name_en } else { preset.name_zh };
              let tier_badge = if is_en { preset.tier_badge_en } else { preset.tier_badge_zh };
              let summary = if is_en { preset.summary_en } else { preset.summary_zh };
              let audience = if is_en { preset.audience_en } else { preset.audience_zh };
              let highlights = if is_en { preset.highlights_en } else { preset.highlights_zh };
              let gh_href = format!("https://github.com/dioxuspress/dioxuspress/tree/main/templates/{preset_id}");
              let tier_pill_class = if preset.uses_db {
                "bg-amber-500/12 text-amber-700 dark:text-amber-300 border border-amber-500/25"
              } else {
                "bg-emerald-500/12 text-emerald-700 dark:text-emerald-300 border border-emerald-500/25"
              };

              rsx! {
                article {
                  key: "{preset_id}",
                  class: "flex flex-col rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/60 overflow-hidden shadow-xs hover:border-primary/50 transition-colors",

                  // 真实配色桌面浏览器窗口缩略图
                  TemplateBrowserMockup {
                    preset_id: preset.id,
                    theme_id: preset.theme_id,
                    dark: preview_dark(),
                  }

                  // 正文说明与操作区
                  div { class: "p-6 flex-1 flex flex-col justify-between gap-5",
                    div { class: "space-y-3",
                      div { class: "flex flex-wrap items-center justify-between gap-2",
                        div { class: "flex items-center gap-2",
                          h2 { class: "text-xl font-bold text-slate-900 dark:text-white",
                            "{name}"
                          }
                          code { class: "px-2 py-0.5 rounded bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 font-mono text-xs",
                            "--template {preset_id}"
                          }
                        }
                        span { class: "inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-semibold {tier_pill_class}",
                          "{tier_badge}"
                        }
                      }

                      p { class: "text-sm text-slate-600 dark:text-slate-300 leading-relaxed",
                        "{summary}"
                      }

                      // 布局 + 默认主题徽章条
                      div { class: "flex flex-wrap items-center gap-2 pt-1 text-xs",
                        span { class: "inline-flex items-center gap-1 rounded-md bg-slate-100 dark:bg-slate-800/90 px-2 py-1 text-slate-700 dark:text-slate-300",
                          span { class: "text-slate-400", "layout:" }
                          code { class: "font-mono font-semibold", "\"{preset.layout_id}\"" }
                        }
                        span { class: "inline-flex items-center gap-1.5 rounded-md bg-slate-100 dark:bg-slate-800/90 px-2 py-1 text-slate-700 dark:text-slate-300",
                          span { class: "text-slate-400", "theme:" }
                          code { class: "font-mono font-semibold", "\"{preset.theme_id}\"" }
                        }
                        span { class: "text-slate-500 dark:text-slate-400",
                          if is_en { "Best for: {audience}" } else { "适用：{audience}" }
                        }
                      }

                      ul { class: "space-y-1.5 pt-1 text-xs sm:text-sm text-slate-600 dark:text-slate-300",
                        for item in highlights.iter() {
                          li { class: "flex items-start gap-2",
                            span { class: "text-primary font-bold mt-0.5", "✓" }
                            span { "{item}" }
                          }
                        }
                      }
                    }

                    // 终端一键初始化命令 + 试穿配色 & 源码目录按钮
                    div { class: "space-y-3 pt-2 border-t border-slate-100 dark:border-slate-800/80",
                      div { class: "flex items-center justify-between gap-2 rounded-xl bg-slate-950 text-slate-100 px-3.5 py-2.5 font-mono text-xs overflow-x-auto",
                        code { class: "whitespace-nowrap select-all", "{cmd}" }
                        button {
                          r#type: "button",
                          class: "shrink-0 ml-2 rounded-md bg-white/10 hover:bg-white/20 px-2.5 py-1 text-[11px] font-sans font-semibold text-white transition-colors cursor-pointer",
                          onclick: move |_| {
                            let id_owned = preset_id.to_string();
                            let text = cmd_for_copy.clone();
                            spawn(async move {
                              if widgets::browser::copy_text(&text).await {
                                copied_preset.set(Some(id_owned));
                              }
                            });
                          },
                          if is_copied {
                            if is_en { "✓ Copied" } else { "✓ 已复制" }
                          } else {
                            if is_en { "Copy" } else { "复制命令" }
                          }
                        }
                      }

                      div { class: "flex flex-wrap items-center justify-between gap-2 text-xs",
                        button {
                          r#type: "button",
                          class: "inline-flex items-center gap-1.5 rounded-lg border border-slate-200 dark:border-slate-700 hover:border-primary/60 px-3 py-1.5 font-semibold text-slate-700 dark:text-slate-200 hover:text-primary transition-colors cursor-pointer",
                          onclick: move |_| {
                            apply_user_theme(theme_for_try.clone(), theme_version);
                          },
                          if is_en {
                            "🎨 Try \"{preset.theme_id}\" theme on this site"
                          } else {
                            "🎨 在本站实时试穿 \"{preset.theme_id}\" 配色"
                          }
                        }

                        a {
                          href: "{gh_href}",
                          target: "_blank",
                          rel: "noopener noreferrer",
                          class: "inline-flex items-center gap-1 font-semibold text-primary hover:underline",
                          if is_en {
                            "Browse templates/{preset_id} ↗"
                          } else {
                            "查看 templates/{preset_id} 目录源码 ↗"
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }

        // ── 资源占用与架构选型对比表 ──
        section { class: "mt-16",
          h2 { class: "text-2xl font-extrabold tracking-tight text-slate-900 dark:text-white",
            if is_en {
              "Resource Footprint & Architecture Comparison"
            } else {
              "资源占用与数据依赖对比（零数据库 vs PostgreSQL）"
            }
          }
          p { class: "mt-2 text-sm text-slate-600 dark:text-slate-400",
            if is_en {
              "All four presets share the exact same Markdown & site.json contract — switching from Zero-DB to Fullstack requires only changing the Docker Compose image."
            } else {
              "四套预设共享完全一致的 Markdown 目录约定与 site.json 配置契约——从零数据库升级到全栈社区只需切换 Compose 镜像。"
            }
          }

          div { class: "mt-5 overflow-x-auto rounded-xl border border-slate-200 dark:border-slate-800",
            table { class: "w-full text-left border-collapse text-xs sm:text-sm",
              thead { class: "bg-slate-50 dark:bg-slate-900 text-slate-700 dark:text-slate-200 border-b border-slate-200 dark:border-slate-800",
                tr {
                  th { class: "py-3 px-4 font-semibold", if is_en { "Dimension" } else { "对比维度" } }
                  th { class: "py-3 px-4 font-semibold", "docs" }
                  th { class: "py-3 px-4 font-semibold", "blog" }
                  th { class: "py-3 px-4 font-semibold", "community" }
                  th { class: "py-3 px-4 font-semibold", "portal" }
                }
              }
              tbody { class: "divide-y divide-slate-200 dark:divide-slate-800 text-slate-600 dark:text-slate-300",
                tr {
                  td { class: "py-3 px-4 font-medium text-slate-900 dark:text-white", if is_en { "Container Image" } else { "官方容器镜像" } }
                  td { class: "py-3 px-4 font-mono text-xs", "dioxuspress-basic" }
                  td { class: "py-3 px-4 font-mono text-xs", "dioxuspress-basic" }
                  td { class: "py-3 px-4 font-mono text-xs", "dioxuspress-app" }
                  td { class: "py-3 px-4 font-mono text-xs", "dioxuspress-app" }
                }
                tr {
                  td { class: "py-3 px-4 font-medium text-slate-900 dark:text-white", if is_en { "Database Required" } else { "外部数据库依赖" } }
                  td { class: "py-3 px-4 text-emerald-600 dark:text-emerald-400 font-semibold", if is_en { "None (Zero-DB)" } else { "无（纯文件只读挂载）" } }
                  td { class: "py-3 px-4 text-emerald-600 dark:text-emerald-400 font-semibold", if is_en { "None (Zero-DB)" } else { "无（纯文件只读挂载）" } }
                  td { class: "py-3 px-4", "PostgreSQL 16" }
                  td { class: "py-3 px-4", "PostgreSQL 16" }
                }
                tr {
                  td { class: "py-3 px-4 font-medium text-slate-900 dark:text-white", if is_en { "Typical Idle RAM" } else { "典型空闲内存占用" } }
                  td { class: "py-3 px-4 font-mono text-xs", "~28–35 MB" }
                  td { class: "py-3 px-4 font-mono text-xs", "~25–32 MB" }
                  td { class: "py-3 px-4 font-mono text-xs", "~85 MB (incl. PG)" }
                  td { class: "py-3 px-4 font-mono text-xs", "~95 MB (incl. PG)" }
                }
                tr {
                  td { class: "py-3 px-4 font-medium text-slate-900 dark:text-white", if is_en { "Embedded Search (⌘K)" } else { "嵌入式全文搜索 (⌘K)" } }
                  td { class: "py-3 px-4", "✓ Tantivy" }
                  td { class: "py-3 px-4", "✓ Tantivy" }
                  td { class: "py-3 px-4", "✓ Tantivy" }
                  td { class: "py-3 px-4", "✓ Tantivy" }
                }
                tr {
                  td { class: "py-3 px-4 font-medium text-slate-900 dark:text-white", if is_en { "OAuth2 / Forum / Annotations" } else { "OAuth 登录 / 论坛 / 划线批注" } }
                  td { class: "py-3 px-4 text-slate-400", "—" }
                  td { class: "py-3 px-4 text-slate-400", "—" }
                  td { class: "py-3 px-4 text-emerald-600 dark:text-emerald-400 font-semibold", "✓" }
                  td { class: "py-3 px-4 text-emerald-600 dark:text-emerald-400 font-semibold", "✓" }
                }
                tr {
                  td { class: "py-3 px-4 font-medium text-slate-900 dark:text-white", if is_en { "Paid Courses / Cases / Podcast" } else { "付费课程 / 案例库 / 播客电台" } }
                  td { class: "py-3 px-4 text-slate-400", "—" }
                  td { class: "py-3 px-4 text-slate-400", "—" }
                  td { class: "py-3 px-4 text-slate-400", "—" }
                  td { class: "py-3 px-4 text-emerald-600 dark:text-emerald-400 font-semibold", "✓" }
                }
              }
            }
          }

          div { class: "mt-6 flex flex-wrap items-center justify-between gap-4 rounded-xl bg-slate-50 dark:bg-slate-900/80 border border-slate-200 dark:border-slate-800 p-5 text-sm",
            div { class: "text-slate-700 dark:text-slate-300",
              if is_en {
                "Don't have the CLI installed yet? Install dioxuspress-cli in ~6 seconds or jump straight to the 5-minute Quickstart."
              } else {
                "尚未安装 dioxuspress CLI？仅需约 6 秒即可编译安装，或直接查阅 5 分钟快速开始指南。"
              }
            }
            div { class: "flex flex-wrap items-center gap-3",
              Link {
                to: "/docs/guide/quickstart",
                class: "inline-flex items-center rounded-lg bg-primary px-4 py-2 text-xs font-semibold text-primary-foreground hover:opacity-90 transition-opacity",
                if is_en { "5-Minute Quickstart →" } else { "5 分钟快速开始 →" }
              }
              Link {
                to: "/docs/guide/layouts-and-themes",
                class: "inline-flex items-center rounded-lg border border-slate-300 dark:border-slate-700 px-4 py-2 text-xs font-semibold text-slate-700 dark:text-slate-200 hover:border-primary/60 transition-colors",
                if is_en { "Explore All 10 Themes" } else { "预览全部 10 套双模主题" }
              }
            }
          }
        }
      }
    }
  }
}

/// 真实主题色值驱动的桌面端浏览器布局骨架预览。
#[component]
fn TemplateBrowserMockup(preset_id: &'static str, theme_id: &'static str, dark: bool) -> Element {
  let (light_bg, light_primary, dark_bg, dark_primary) = lookup_theme_colors(theme_id);
  let bg = if dark { dark_bg } else { light_bg };
  let primary = if dark { dark_primary } else { light_primary };
  let fg = if dark { "#f1f5f9" } else { "#0f172a" };
  let muted_fg = if dark { "#94a3b8" } else { "#64748b" };
  let surface = if dark { "rgba(255,255,255,0.05)" } else { "rgba(15,23,42,0.04)" };
  let border = if dark { "rgba(255,255,255,0.10)" } else { "rgba(15,23,42,0.10)" };
  let chrome_bg = if dark { "#0f172a" } else { "#f1f5f9" };
  let mode_label = if dark { "dark" } else { "light" };

  rsx! {
    div {
      class: "border-b border-slate-200 dark:border-slate-800 select-none",
      style: "background-color: {bg}; color: {fg};",
      "aria-hidden": "true",

      // 桌面浏览器窗口控制栏
      div {
        class: "flex items-center justify-between px-3.5 py-2 text-[10px] font-mono",
        style: "background-color: {chrome_bg}; border-bottom: 1px solid {border}; color: {muted_fg};",
        div { class: "flex items-center gap-1.5",
          span { class: "w-2.5 h-2.5 rounded-full bg-rose-400/80" }
          span { class: "w-2.5 h-2.5 rounded-full bg-amber-400/80" }
          span { class: "w-2.5 h-2.5 rounded-full bg-emerald-400/80" }
        }
        span { class: "px-2.5 py-0.5 rounded", style: "background-color: {surface};",
          "localhost:8080 · {preset_id} ({theme_id})"
        }
        span { "{mode_label}" }
      }

      // 各预设高还原度布局画布（固定高度 h-48 保证 2×2 卡片严整对齐）
      match preset_id {
        "docs" => rsx! {
          div { class: "h-48 flex flex-col text-[10px]",
            // 顶部可关闭发布公告条
            div {
              class: "px-3 py-1 text-center text-[9px] font-semibold truncate",
              style: "background-color: {primary}; color: {bg};",
              "🎉 v0.1 Released — Multi-volume 3-level docs + Tantivy search →"
            }
            // 左对齐 Docs 顶栏
            div {
              class: "px-3 py-1.5 flex items-center justify-between",
              style: "border-bottom: 1px solid {border};",
              div { class: "flex items-center gap-3",
                span { class: "font-bold", style: "color: {primary};", "◆ Docs" }
                span { class: "font-semibold", "Guide" }
                span { style: "color: {muted_fg};", "API" }
                span { style: "color: {muted_fg};", "Blog" }
              }
              span { class: "px-1.5 py-0.5 rounded font-mono text-[9px]", style: "background-color: {surface}; color: {muted_fg};", "⌘K" }
            }
            // 三栏主体：左侧树 + 中间正文代码标签 + 右侧 TOC
            div { class: "flex-1 grid grid-cols-12 overflow-hidden",
              div { class: "col-span-3 p-2.5 space-y-1.5", style: "border-right: 1px solid {border};",
                div { class: "font-semibold text-[9px]", style: "color: {muted_fg};", "GUIDE" }
                div { class: "px-1.5 py-0.5 rounded font-semibold", style: "background-color: {surface}; color: {primary};", "Quickstart" }
                div { class: "px-1.5 text-[9px]", style: "color: {muted_fg};", "Configuration" }
                div { class: "px-1.5 text-[9px]", style: "color: {muted_fg};", "▾ Advanced" }
                div { class: "pl-3 text-[9px]", style: "color: {muted_fg};", "Deployment" }
              }
              div { class: "col-span-6 p-3 space-y-2",
                div { class: "text-[9px] font-mono", style: "color: {primary};", "Guide / Quickstart" }
                div { class: "font-bold text-xs", "Getting Started in 50s" }
                div { class: "h-2 rounded w-5/6", style: "background-color: {surface};" }
                div { class: "rounded p-2 font-mono text-[9px] space-y-1", style: "background-color: {surface}; border: 1px solid {border};",
                  div { class: "flex gap-2 pb-1", style: "border-bottom: 1px solid {border};",
                    span { style: "color: {primary};", "[Docker]" }
                    span { style: "color: {muted_fg};", "[CLI]" }
                  }
                  div { "docker compose up -d" }
                }
              }
              div { class: "col-span-3 p-2.5 space-y-1 text-[9px]", style: "border-left: 1px solid {border}; color: {muted_fg};",
                div { class: "font-semibold", style: "color: {fg};", "On this page" }
                div { style: "color: {primary};", "• Directory Tree" }
                div { "• Code Tabs" }
                div { "• CI Check" }
              }
            }
          }
        },
        "blog" => rsx! {
          div { class: "h-48 flex flex-col text-[10px]",
            // Minimal 极简顶栏
            div {
              class: "px-5 py-2 flex items-center justify-between",
              style: "border-bottom: 1px solid {border};",
              span { class: "font-serif font-bold tracking-tight", "Indie Systems Notes" }
              div { class: "flex items-center gap-2.5 text-[9px]",
                span { "Essays" }
                span { style: "color: {muted_fg};", "Notes" }
                span {
                  class: "px-1.5 py-0.5 rounded font-mono font-semibold",
                  style: "background-color: {primary}; color: {bg};",
                  "RSS /feed.xml"
                }
              }
            }
            // 居中宽裕留白长文阅读流 + MathML 公式块
            div { class: "flex-1 px-8 py-3 flex flex-col justify-between max-w-md mx-auto w-full",
              div { class: "space-y-1",
                div { class: "text-[9px] font-mono", style: "color: {muted_fg};", "2026-11-12 · Systems Series #2 · @author" }
                div { class: "font-bold text-xs leading-snug", "Understanding P99 Tail Latency via Little's Law" }
                div { class: "h-1.5 rounded w-full", style: "background-color: {surface};" }
              }
              div {
                class: "rounded-lg py-2 px-3 text-center font-serif text-xs italic",
                style: "background-color: {surface}; border: 1px solid {border}; color: {primary};",
                "E[T] = 1 / (μ · (1 − ρ))   ·   L = λW"
              }
              div { class: "flex items-center justify-between text-[9px]", style: "color: {muted_fg};",
                span { "#rust  #queueing-theory" }
                span { style: "color: {primary};", "Next in series →" }
              }
            }
          }
        },
        "community" => rsx! {
          div { class: "h-48 flex flex-col text-[10px]",
            // Classic 社区顶栏 + OAuth 登录按钮
            div {
              class: "px-3.5 py-1.5 flex items-center justify-between",
              style: "border-bottom: 1px solid {border};",
              div { class: "flex items-center gap-3",
                span { class: "font-bold", style: "color: {primary};", "● Community" }
                span { style: "color: {muted_fg};", "Docs" }
                span { style: "color: {muted_fg};", "Blog" }
                span { class: "font-semibold", "Forum" }
              }
              span {
                class: "px-2 py-0.5 rounded-full text-[9px] font-semibold",
                style: "background-color: {primary}; color: {bg};",
                "GitHub Sign In"
              }
            }
            // 上半：正文段落划线批注气泡；下半：关联论坛热帖列表
            div { class: "flex-1 p-3 grid grid-cols-12 gap-2.5 overflow-hidden",
              div { class: "col-span-7 space-y-2 pr-2", style: "border-right: 1px solid {border};",
                div { class: "font-bold text-[11px]", "Forum & Inline Annotations" }
                div {
                  class: "p-1.5 rounded text-[9px]",
                  style: "background-color: {surface}; border-left: 2px solid {primary};",
                  "✎ Highlighted note: PKCE uses SameSite=Lax cookies..."
                }
                div { class: "space-y-1.5 pt-0.5",
                  div { class: "flex items-center justify-between p-1.5 rounded", style: "background-color: {surface};",
                    span { class: "truncate font-medium", "💬 RFC: Async plugin hooks?" }
                    span { class: "font-mono text-[9px]", style: "color: {primary};", "12 replies" }
                  }
                }
              }
              div { class: "col-span-5 space-y-1.5 text-[9px]",
                div { class: "font-semibold", "Active Tags" }
                div { class: "flex flex-wrap gap-1",
                  span { class: "px-1.5 py-0.5 rounded", style: "background-color: {surface}; color: {primary};", "#oauth2" }
                  span { class: "px-1.5 py-0.5 rounded", style: "background-color: {surface};", "#wasm" }
                  span { class: "px-1.5 py-0.5 rounded", style: "background-color: {surface};", "#moderation" }
                }
                div { class: "pt-1", style: "color: {muted_fg};", "✓ URL blocklist active" }
                div { style: "color: {muted_fg};", "✓ /admin queue ready" }
              }
            }
          }
        },
        _ => rsx! {
          div { class: "h-48 flex flex-col text-[10px]",
            // Portal 企业级全栈门户顶栏
            div {
              class: "px-3.5 py-1.5 flex items-center justify-between",
              style: "border-bottom: 1px solid {border};",
              div { class: "flex items-center gap-2.5",
                span { class: "font-bold", style: "color: {primary};", "■ Portal" }
                span { style: "color: {muted_fg};", "Docs" }
                span { class: "font-semibold", "Courses" }
                span { style: "color: {muted_fg};", "Cases" }
                span { style: "color: {muted_fg};", "Podcast" }
              }
              span { class: "px-1.5 py-0.5 rounded font-mono text-[9px]", style: "background-color: {surface}; color: {primary};", "PRO" }
            }
            // 课程试读/锁定目录 + 播客音频条 + 案例卡
            div { class: "flex-1 p-3 grid grid-cols-12 gap-2.5 overflow-hidden",
              div { class: "col-span-7 space-y-1.5",
                div { class: "flex items-center justify-between",
                  span { class: "font-bold text-[11px]", "Fullstack Rust Course" }
                  span { class: "font-mono font-semibold text-[9px]", style: "color: {primary};", "¥99 / $14.99" }
                }
                div { class: "p-1.5 rounded flex items-center justify-between text-[9px]", style: "background-color: {surface}; border: 1px solid {border};",
                  span { "1.1 SSR & Hydration" }
                  span { class: "px-1.5 rounded font-semibold", style: "background-color: {primary}; color: {bg};", "FREE PREVIEW" }
                }
                div { class: "p-1.5 rounded flex items-center justify-between text-[9px]", style: "background-color: {surface}; color: {muted_fg};",
                  span { "1.2 Production Security" }
                  span { "🔒 Locked" }
                }
              }
              div { class: "col-span-5 flex flex-col justify-between",
                div { class: "p-2 rounded space-y-1 text-[9px]", style: "background-color: {surface}; border: 1px solid {border};",
                  div { class: "font-semibold", "🎙️ EP01 Podcast" }
                  div { class: "h-1 rounded-full overflow-hidden", style: "background-color: {border};",
                    div { class: "h-full w-2/3", style: "background-color: {primary};" }
                  }
                  div { class: "font-mono text-[8px]", style: "color: {muted_fg};", "▶ 12:14 / 18:40" }
                }
                div { class: "p-1.5 rounded text-[9px] flex items-center justify-between", style: "background-color: {surface};",
                  span { class: "truncate", "★ Case Showcase" }
                  span { style: "color: {primary};", "→" }
                }
              }
            }
          }
        },
      }
    }
  }
}
