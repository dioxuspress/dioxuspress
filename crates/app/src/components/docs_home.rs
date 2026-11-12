//! 类 Docusaurus 开源文档站首页（`active_layout = "docs"`）。
//!
//! 结构：
//! 1. 居中排版 Hero：顶部胶囊标 + 大标题 + 副标题 + 主次行动按钮 + 一键复制终端安装命令框
//! 2. 特性卡片栅格：由 `site.json::features` 纯配置驱动的 3 列网格
//! 3. 文档分卷速览：服务端预取 `list_doc_volumes`，直达各核心分卷与首章

use dioxus::prelude::*;
use dioxus::router::Link;
use dioxus_shadcn::{badge_class, card_class, BadgeVariant};
use module_docs::server::{list_doc_volumes, DocVolume, CARD_CHAPTERS};

use crate::components::home_sections::CommunityFeed;
use crate::components::view::Container;
use crate::i18n::{t, use_i18n};
use crate::routes::Route;

fn is_external_href(href: &str) -> bool {
  href.starts_with("http://") || href.starts_with("https://") || href.starts_with("mailto:")
}

fn feature_icon(name: &str) -> Element {
  match name.trim() {
    "bolt" => rsx! {
      svg { class: "w-5 h-5", fill: "none", stroke: "currentColor", view_box: "0 0 24 24",
        path { stroke_linecap: "round", stroke_linejoin: "round", stroke_width: "2", d: "M13 10V3L4 14h7v7l9-11h-7z" }
      }
    },
    "book" => rsx! {
      svg { class: "w-5 h-5", fill: "none", stroke: "currentColor", view_box: "0 0 24 24",
        path { stroke_linecap: "round", stroke_linejoin: "round", stroke_width: "2", d: "M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" }
      }
    },
    "layers" => rsx! {
      svg { class: "w-5 h-5", fill: "none", stroke: "currentColor", view_box: "0 0 24 24",
        path { stroke_linecap: "round", stroke_linejoin: "round", stroke_width: "2", d: "M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" }
      }
    },
    "search" => rsx! {
      svg { class: "w-5 h-5", fill: "none", stroke: "currentColor", view_box: "0 0 24 24",
        path { stroke_linecap: "round", stroke_linejoin: "round", stroke_width: "2", d: "M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" }
      }
    },
    "shield" => rsx! {
      svg { class: "w-5 h-5", fill: "none", stroke: "currentColor", view_box: "0 0 24 24",
        path { stroke_linecap: "round", stroke_linejoin: "round", stroke_width: "2", d: "M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" }
      }
    },
    "palette" => rsx! {
      svg { class: "w-5 h-5", fill: "none", stroke: "currentColor", view_box: "0 0 24 24",
        path { stroke_linecap: "round", stroke_linejoin: "round", stroke_width: "2", d: "M7 21a4 4 0 01-4-4V5a2 2 0 012-2h4a2 2 0 012 2v12a4 4 0 01-4 4zm0 0h12a2 2 0 002-2v-4a2 2 0 00-2-2h-2.343M11 7.343l1.657-1.657a2 2 0 012.828 0l2.829 2.829a2 2 0 010 2.828l-8.486 8.485M7 17h.01" }
      }
    },
    other if !other.is_empty() => rsx! { span { class: "text-base font-bold", "{other}" } },
    _ => rsx! {
      svg { class: "w-5 h-5", fill: "none", stroke: "currentColor", view_box: "0 0 24 24",
        path { stroke_linecap: "round", stroke_linejoin: "round", stroke_width: "2", d: "M5 13l4 4L19 7" }
      }
    },
  }
}

#[component]
fn DocsVolumePreview() -> Element {
  let lang = use_i18n();
  let vols_res = use_server_future(|| async move { list_doc_volumes().await.unwrap_or_default() })?;
  let Some(vols) = vols_res() else {
    return rsx! {};
  };
  if vols.is_empty() {
    return rsx! {};
  }

  rsx! {
    section { class: "py-16 border-t border-slate-200/70 dark:border-slate-800/80 bg-slate-50/50 dark:bg-slate-900/30",
      Container {
        div { class: "flex flex-col sm:flex-row sm:items-end justify-between gap-4 mb-10",
          div {
            h2 { class: "text-2xl sm:text-3xl font-extrabold tracking-tight text-slate-900 dark:text-white",
              "{t(lang(), \"home.docs_volumes.title\")}"
            }
            p { class: "mt-2 text-slate-600 dark:text-slate-400",
              "{t(lang(), \"home.docs_volumes.subtitle\")}"
            }
          }
          Link {
            to: Route::Docs {},
            class: "text-sm font-semibold text-[var(--color-primary)] hover:underline shrink-0",
            "{t(lang(), \"home.docs_volumes.all\")} →"
          }
        }

        div { class: "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5",
          for vol in vols.iter().take(8) {
            {render_volume_card(vol, lang())}
          }
        }
      }
    }
  }
}

fn render_volume_card(vol: &DocVolume, lang: app_core::i18n::Language) -> Element {
  let node = &vol.node;
  let href = if node.has_content {
    format!("/docs/{}", node.path)
  } else if let Some(first_child) = node.children.first() {
    format!("/docs/{}", first_child.path)
  } else {
    "/docs".to_string()
  };
  let chapters: Vec<String> =
    node.children.iter().take(CARD_CHAPTERS.min(4)).map(|c| c.title.clone()).collect();

  rsx! {
    Link {
      key: "{node.path}",
      to: href,
      class: card_class("group flex flex-col justify-between p-5 rounded-xl hover:border-primary/50 hover:shadow-md transition-all bg-white dark:bg-slate-950"),
      div {
        div { class: "flex items-center justify-between gap-2 mb-2.5",
          h3 { class: "font-bold text-base text-slate-900 dark:text-white group-hover:text-primary transition-colors",
            "{node.title}"
          }
          span { class: badge_class(BadgeVariant::Secondary, "rounded-full px-2 py-0.5 text-xs font-normal text-muted-foreground shrink-0"),
            "{vol.pages} {t(lang, \"docs.count_unit\")}"
          }
        }
        if !node.description.is_empty() {
          p { class: "text-xs text-slate-500 dark:text-slate-400 line-clamp-2 mb-3 leading-relaxed",
            "{node.description}"
          }
        }
      }
      if !chapters.is_empty() {
        div { class: "pt-3 border-t border-slate-100 dark:border-slate-800/80 flex flex-wrap gap-1.5",
          for ch in chapters {
            span { class: "text-[11px] px-2 py-0.5 rounded bg-slate-100 dark:bg-slate-900 text-slate-600 dark:text-slate-400",
              "{ch}"
            }
          }
        }
      }
    }
  }
}

/// 类 Docusaurus 开源项目首页主体。
#[component]
pub fn DocsLanding(enabled: Vec<String>) -> Element {
  let lang = use_i18n();
  let site_cfg = app_core::settings::SiteConfig::embedded();
  let mut copied = use_signal(|| false);

  let on = |id: &str| enabled.iter().any(|e| e == id);
  let on_docs = on("docs");
  let on_blog = on("blog");
  let on_forum = on("forum");

  let primary_cta: Option<(String, String)> =
    site_cfg.hero_primary_cta(lang()).map(|(txt, url)| (txt.to_string(), url.to_string())).or_else(
      || {
        if on_docs {
          Some((t(lang(), "hero.btn.docs"), "/docs".to_string()))
        } else {
          None
        }
      },
    );

  let secondary_cta: Option<(String, String)> = site_cfg
    .hero_secondary_cta(lang())
    .map(|(txt, url)| (txt.to_string(), url.to_string()))
    .or_else(|| site_cfg.github_repo_url().map(|u| ("GitHub".to_string(), u.to_string())));

  let install_cmd = site_cfg.hero_install_command().map(str::to_string);

  rsx! {
    // 1. 居中开源项目 Hero
    section { class: "relative overflow-hidden py-20 sm:py-28 bg-gradient-to-b from-cyan-50/60 via-white to-white dark:from-slate-950 dark:via-slate-950 dark:to-slate-900 border-b border-slate-200/70 dark:border-slate-800/80",
      // 柔和背景光斑
      div {
        "aria-hidden": "true",
        class: "pointer-events-none absolute inset-x-0 -top-24 flex justify-center overflow-hidden",
        div { class: "w-[42rem] h-[18rem] rounded-full bg-cyan-500/10 dark:bg-cyan-400/10 blur-3xl" }
      }

      Container {
        div { class: "relative max-w-4xl mx-auto text-center",
          if let Some(badge) = site_cfg.hero_badge(lang()) {
            div { class: "mb-6 inline-flex items-center gap-2 rounded-full border border-cyan-500/30 bg-cyan-500/10 px-3.5 py-1 text-xs font-semibold text-cyan-700 dark:text-cyan-300",
              span { class: "h-2 w-2 rounded-full bg-cyan-500 dark:bg-cyan-400" }
              "{badge}"
            }
          }

          h1 { class: "text-4xl sm:text-5xl lg:text-6xl font-extrabold tracking-tight text-slate-900 dark:text-white leading-tight",
            "{site_cfg.hero_title(lang())}"
          }

          p { class: "mt-6 max-w-2xl mx-auto text-base sm:text-lg md:text-xl text-slate-600 dark:text-slate-300 leading-relaxed",
            "{site_cfg.hero_subtitle(lang())}"
          }

          if primary_cta.is_some() || secondary_cta.is_some() {
            div { class: "mt-9 flex flex-wrap items-center justify-center gap-3.5",
              if let Some((txt, url)) = primary_cta {
                if is_external_href(&url) {
                  a {
                    href: "{url}",
                    target: "_blank",
                    rel: "noopener noreferrer",
                    class: "inline-flex items-center justify-center gap-2 rounded-lg btn-flow px-6 py-3 text-sm font-semibold shadow-sm transition-all",
                    "{txt}"
                    span { "aria-hidden": "true", "→" }
                  }
                } else {
                  Link {
                    to: url,
                    class: "inline-flex items-center justify-center gap-2 rounded-lg btn-flow px-6 py-3 text-sm font-semibold shadow-sm transition-all",
                    "{txt}"
                    span { "aria-hidden": "true", "→" }
                  }
                }
              }
              if let Some((txt, url)) = secondary_cta {
                if is_external_href(&url) {
                  a {
                    href: "{url}",
                    target: "_blank",
                    rel: "noopener noreferrer",
                    class: "inline-flex items-center justify-center gap-2 rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 px-6 py-3 text-sm font-semibold text-slate-800 dark:text-slate-100 hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors",
                    "{txt}"
                  }
                } else {
                  Link {
                    to: url,
                    class: "inline-flex items-center justify-center gap-2 rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 px-6 py-3 text-sm font-semibold text-slate-800 dark:text-slate-100 hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors",
                    "{txt}"
                  }
                }
              }
            }
          }

          if let Some(cmd) = install_cmd {
            div { class: "mt-8 max-w-xl mx-auto",
              div { class: "flex items-center justify-between gap-3 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-900 dark:bg-slate-900/90 px-4 py-2.5 text-left font-mono text-xs sm:text-sm text-slate-200 shadow-sm",
                div { class: "flex items-center gap-2.5 min-w-0 overflow-x-auto",
                  span { class: "text-cyan-400 select-none shrink-0", "$" }
                  code { class: "whitespace-nowrap", "{cmd}" }
                }
                button {
                  r#type: "button",
                  title: "{t(lang(), \"hero.copy_cmd\")}",
                  "aria-label": "{t(lang(), \"hero.copy_cmd\")}",
                  class: "shrink-0 inline-flex items-center gap-1 rounded-md px-2 py-1 text-xs text-slate-300 hover:text-white hover:bg-white/10 transition-colors",
                  onclick: {
                    let cmd_to_copy = cmd.clone();
                    move |_| {
                      let text = cmd_to_copy.clone();
                      spawn(async move {
                        if widgets::browser::copy_text(&text).await {
                          copied.set(true);
                        }
                      });
                    }
                  },
                  if copied() {
                    span { class: "text-emerald-400 font-sans font-medium", "✓ {t(lang(), \"hero.copied\")}" }
                  } else {
                    svg { class: "w-4 h-4", fill: "none", stroke: "currentColor", view_box: "0 0 24 24",
                      path { stroke_linecap: "round", stroke_linejoin: "round", stroke_width: "2", d: "M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }

    // 2. 特性卡片栅格（Docusaurus 经典三列布局）
    if !site_cfg.features.is_empty() {
      section { class: "py-16 sm:py-20 bg-white dark:bg-slate-950",
        Container {
          div { class: "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6",
            for (idx, item) in site_cfg.features.iter().enumerate() {
              if let Some(href) = item.href_or_none() {
                if is_external_href(href) {
                  a {
                    key: "feat-{idx}",
                    href: "{href}",
                    target: "_blank",
                    rel: "noopener noreferrer",
                    class: card_class("group p-6 rounded-2xl hover:border-primary/50 hover:shadow-md transition-all"),
                    {render_feature_body(item, lang())}
                  }
                } else {
                  Link {
                    key: "feat-{idx}",
                    to: href.to_string(),
                    class: card_class("group p-6 rounded-2xl hover:border-primary/50 hover:shadow-md transition-all"),
                    {render_feature_body(item, lang())}
                  }
                }
              } else {
                div {
                  key: "feat-{idx}",
                  class: card_class("p-6 rounded-2xl"),
                  {render_feature_body(item, lang())}
                }
              }
            }
          }
        }
      }
    }

    // 3. 文档分卷速览
    if on_docs {
      SuspenseBoundary { fallback: |_| VNode::empty(),
        DocsVolumePreview {}
      }
    }

    // 4. 最新博客 / 社区动态（若启用）
    if on_blog || on_forum {
      CommunityFeed { blog: on_blog, forum: on_forum }
    }
  }
}

fn render_feature_body(
  item: &app_core::settings::FeatureItem,
  lang: app_core::i18n::Language,
) -> Element {
  rsx! {
    div { class: "mb-4 inline-flex h-10 w-10 items-center justify-center rounded-xl bg-primary/10 text-primary",
      {feature_icon(&item.icon)}
    }
    h3 { class: "text-lg font-bold text-slate-900 dark:text-white group-hover:text-primary transition-colors",
      "{item.title_for(lang)}"
    }
    p { class: "mt-2 text-sm text-slate-600 dark:text-slate-400 leading-relaxed",
      "{item.description_for(lang)}"
    }
  }
}
