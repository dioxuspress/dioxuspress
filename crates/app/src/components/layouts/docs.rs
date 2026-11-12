//! 类 Docusaurus 文档站布局（`active_layout = "docs"`）。
//!
//! 特点：
//! - 顶部可选可关闭公告横幅（`site.json::announcement`，状态记入 `localStorage`）
//! - 左对齐清爽顶栏：Logo + 文档 / 博客 / 案例等扁平导航（无复杂生态下拉菜单）
//! - 右侧工具栏：`⌘K` 搜索胶囊、GitHub 仓库图标、主题配色、语言切换、暗色模式
//! - 底部多列结构化页脚（支持 `site.json::footer.columns` 自定义分组）

use dioxus::prelude::*;
use dioxus::router::{Link, Outlet};
use dioxus_shadcn::{
  Button, ButtonSize, ButtonVariant, Sheet, SheetClose, SheetContent, SheetOverlay, SheetSide,
  SheetTitle,
};

use crate::components::lang_picker::LangPicker;
use crate::components::module_gate::use_enabled_modules;
use crate::components::theme_picker::ThemePicker;
use crate::components::user_menu::UserMenu;
use crate::components::view::Container;
use crate::i18n::{t, use_i18n};
use crate::routes::Route;
use module_search::search::SearchButton;

fn is_external_or_raw_href(href: &str) -> bool {
  href.starts_with("http://")
    || href.starts_with("https://")
    || href.starts_with("mailto:")
    || href == "/feed.xml"
    || href == "/sitemap.xml"
}

/// 顶部可关闭公告横幅。SSR 默认渲染可见，水合后读取 `localStorage` 判断是否已被用户关闭。
#[component]
fn AnnouncementBar() -> Element {
  let lang = use_i18n();
  let site_cfg = app_core::settings::SiteConfig::embedded();
  let Some(ref ann) = site_cfg.announcement else {
    return rsx! {};
  };
  let Some(text) = ann.text_for(lang()) else {
    return rsx! {};
  };
  let storage_key = ann.storage_key();
  let href = ann.href_or_none().map(str::to_string);
  let mut dismissed = use_signal(|| false);

  let key_for_effect = storage_key.clone();
  use_effect(move || {
    if widgets::browser::storage_get(&key_for_effect).as_deref() == Some("1") {
      dismissed.set(true);
    }
  });

  if dismissed() {
    return rsx! {};
  }

  rsx! {
    div { class: "relative bg-gradient-to-r from-cyan-600 via-sky-600 to-blue-600 text-white text-xs sm:text-sm py-2 px-10 text-center font-medium",
      if let Some(url) = href {
        if is_external_or_raw_href(&url) {
          a {
            href: "{url}",
            target: "_blank",
            rel: "noopener noreferrer",
            class: "hover:underline underline-offset-4 inline-flex items-center gap-1",
            "{text}"
            span { "aria-hidden": "true", "→" }
          }
        } else {
          Link {
            to: url,
            class: "hover:underline underline-offset-4 inline-flex items-center gap-1",
            "{text}"
            span { "aria-hidden": "true", "→" }
          }
        }
      } else {
        span { "{text}" }
      }
      button {
        r#type: "button",
        class: "absolute right-2.5 top-1/2 -translate-y-1/2 p-1 rounded hover:bg-white/15 transition-colors",
        "aria-label": "{t(lang(), \"nav.dismiss_announcement\")}",
        onclick: move |_| {
          dismissed.set(true);
          widgets::browser::storage_set(&storage_key, "1");
        },
        svg { class: "w-3.5 h-3.5", fill: "none", stroke: "currentColor", view_box: "0 0 24 24",
          path { stroke_linecap: "round", stroke_linejoin: "round", stroke_width: "2", d: "M6 18L18 6M6 6l12 12" }
        }
      }
    }
  }
}

/// 类 Docusaurus 文档站外壳：公告条 + 左对齐导航顶栏 + `Outlet::<Route>` + 多列页脚。
#[component]
pub fn DocsShell() -> Element {
  let route = use_route::<Route>();
  let lang = use_i18n();
  let site_cfg = app_core::settings::SiteConfig::embedded();
  let mut is_dark = use_signal(|| false);
  let mut show_auth_modal = crate::use_auth_modal();
  let session_user = crate::use_session_user();
  let mut show_mobile_menu = use_signal(|| false);

  let enabled = use_enabled_modules();
  let on = |id: &str| enabled.iter().any(|s| s == id);
  let on_docs = on("docs");
  let on_blog = on("blog");
  let on_cases = on("cases");
  let on_course = on("course");
  let on_forum = on("forum");
  let on_about = on("about");

  let nav_item_class = move |target: Route| {
    let is_active = match (&route, &target) {
      (Route::DocPage { .. }, Route::Docs {}) => true,
      (Route::Blog { .. }, Route::BlogIndex {}) => true,
      (Route::CaseDetail { .. }, Route::Cases {}) => true,
      (Route::CourseDetail { .. }, Route::Courses {}) => true,
      (Route::Lesson { .. }, Route::Courses {}) => true,
      (Route::TopicsByTag { .. }, Route::TopicsIndex {}) => true,
      (Route::TopicDetail { .. }, Route::TopicsIndex {}) => true,
      (Route::TopicsNew {}, Route::TopicsIndex {}) => true,
      (cur, tgt) => cur == tgt,
    };
    if is_active {
      "px-2.5 text-sm font-semibold text-[var(--color-primary)] h-14 inline-flex items-center border-b-2 border-[var(--color-primary)]"
    } else {
      "px-2.5 text-sm font-medium text-slate-600 hover:text-slate-900 dark:text-slate-300 dark:hover:text-white transition-colors h-14 inline-flex items-center"
    }
  };

  use_effect(move || {
    let dark = widgets::browser::dark_mode_preference();
    widgets::browser::set_class(true, "dark", dark);
    is_dark.set(dark);
  });

  let toggle_dark = move |_| {
    let new_val = !is_dark();
    is_dark.set(new_val);
    widgets::browser::set_dark_mode(new_val);
  };

  rsx! {
    div { class: "min-h-screen flex flex-col bg-white dark:bg-slate-950",
      AnnouncementBar {}

      header { class: "sticky top-0 z-50 border-b border-slate-200/80 bg-white/90 backdrop-blur dark:bg-slate-950/85 dark:border-slate-800",
        div { class: "max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-14 flex items-center justify-between gap-4",
          // 左侧：移动端菜单按钮 + 品牌标识 + 左对齐扁平导航（Docusaurus 经典布局）
          div { class: "flex items-center gap-2 md:gap-6 min-w-0",
            Button {
              r#type: "button",
              variant: ButtonVariant::Ghost,
              size: ButtonSize::Icon,
              class: "md:hidden -ml-1 text-muted-foreground",
              onclick: move |_| show_mobile_menu.set(!show_mobile_menu()),
              "aria-label": "{t(lang(), \"nav.menu\")}",
              "aria-expanded": show_mobile_menu().to_string(),
              svg { class: "w-5 h-5", fill: "none", stroke: "currentColor", view_box: "0 0 24 24",
                path { stroke_linecap: "round", stroke_linejoin: "round", stroke_width: "2", d: "M4 6h16M4 12h16M4 18h16" }
              }
            }

            Link {
              to: Route::Home {},
              class: "font-extrabold tracking-tight text-flow text-lg inline-flex items-center gap-2 shrink-0",
              super::BrandMark {}
            }

            nav { class: "hidden md:flex items-center gap-1",
              if on_docs {
                Link { to: Route::Docs {}, class: nav_item_class(Route::Docs {}), "{t(lang(), \"nav.docs\")}" }
              }
              if on_blog {
                Link { to: Route::BlogIndex {}, class: nav_item_class(Route::BlogIndex {}), "{t(lang(), \"nav.blog\")}" }
              }
              if on_cases {
                Link { to: Route::Cases {}, class: nav_item_class(Route::Cases {}), "{t(lang(), \"nav.cases\")}" }
              }
              if on_course {
                Link { to: Route::Courses {}, class: nav_item_class(Route::Courses {}), "{t(lang(), \"nav.course\")}" }
              }
              if on_forum {
                Link { to: Route::TopicsIndex {}, class: nav_item_class(Route::TopicsIndex {}), "{t(lang(), \"nav.forum\")}" }
              }
              if on_about {
                Link { to: Route::About {}, class: nav_item_class(Route::About {}), "{t(lang(), \"footer.about\")}" }
              }
            }
          }

          // 右侧：搜索胶囊 + GitHub + 主题 / 语言 / 暗色 / 可选登录
          div { class: "flex items-center gap-1.5 sm:gap-2 shrink-0",
            SearchButton {}

            if let Some(gh_url) = site_cfg.github_repo_url() {
              a {
                href: "{gh_url}",
                target: "_blank",
                rel: "noopener noreferrer",
                title: "{t(lang(), \"nav.github\")}",
                "aria-label": "{t(lang(), \"nav.github\")}",
                class: "inline-flex items-center justify-center h-8 w-8 rounded-md text-slate-600 hover:text-slate-900 hover:bg-slate-100 dark:text-slate-300 dark:hover:text-white dark:hover:bg-slate-800 transition-colors",
                svg { class: "w-4.5 h-4.5", fill: "currentColor", view_box: "0 0 24 24",
                  path {
                    fill_rule: "evenodd",
                    clip_rule: "evenodd",
                    d: "M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.53 1.032 1.53 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z",
                  }
                }
              }
            }

            div { class: "hidden sm:inline-flex items-center gap-1.5",
              ThemePicker {}
              LangPicker {}
            }

            Button {
              r#type: "button",
              variant: ButtonVariant::Ghost,
              size: ButtonSize::Icon,
              class: "text-muted-foreground",
              onclick: toggle_dark,
              "aria-label": "{t(lang(), \"nav.toggle_dark\")}",
              "aria-pressed": is_dark().to_string(),
              if is_dark() {
                svg { class: "w-4 h-4", fill: "none", stroke: "currentColor", view_box: "0 0 24 24",
                  path { stroke_linecap: "round", stroke_linejoin: "round", stroke_width: "2", d: "M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" }
                }
              } else {
                svg { class: "w-4 h-4", fill: "none", stroke: "currentColor", view_box: "0 0 24 24",
                  path { stroke_linecap: "round", stroke_linejoin: "round", stroke_width: "2", d: "M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" }
                }
              }
            }

            if let Some(ref u) = session_user() {
              UserMenu { user: u.clone(), compact: true, show_my_topics: on_forum }
            } else if cfg!(feature = "accounts") && site_cfg.auth.enabled {
              Button {
                r#type: "button",
                variant: ButtonVariant::Ghost,
                size: ButtonSize::Sm,
                class: "text-xs",
                onclick: move |_| show_auth_modal.set(true),
                "{t(lang(), \"auth.sign_in\")}"
              }
            }
          }
        }
      }

      // 移动端侧滑抽屉
      Sheet { open: show_mobile_menu(), on_open_change: move |v| show_mobile_menu.set(v),
        SheetOverlay { class: "md:hidden" }
        SheetContent { side: SheetSide::Left, class: "md:hidden w-72 overflow-y-auto",
          div { class: "flex items-center justify-between pb-3 border-b border-border",
            SheetTitle { "{t(lang(), \"nav.menu\")}" }
            SheetClose {
              svg { class: "w-4 h-4", fill: "none", stroke: "currentColor", view_box: "0 0 24 24",
                path { stroke_linecap: "round", stroke_linejoin: "round", stroke_width: "2", d: "M6 18L18 6M6 6l12 12" }
              }
            }
          }
          nav { class: "pt-3 flex flex-col gap-1 text-sm font-medium",
            if on_docs {
              Link { to: Route::Docs {}, class: "px-2.5 py-2 rounded-md hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-700 dark:text-slate-200", onclick: move |_| show_mobile_menu.set(false), "{t(lang(), \"nav.docs\")}" }
            }
            if on_blog {
              Link { to: Route::BlogIndex {}, class: "px-2.5 py-2 rounded-md hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-700 dark:text-slate-200", onclick: move |_| show_mobile_menu.set(false), "{t(lang(), \"nav.blog\")}" }
            }
            if on_cases {
              Link { to: Route::Cases {}, class: "px-2.5 py-2 rounded-md hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-700 dark:text-slate-200", onclick: move |_| show_mobile_menu.set(false), "{t(lang(), \"nav.cases\")}" }
            }
            if on_course {
              Link { to: Route::Courses {}, class: "px-2.5 py-2 rounded-md hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-700 dark:text-slate-200", onclick: move |_| show_mobile_menu.set(false), "{t(lang(), \"nav.course\")}" }
            }
            if on_forum {
              Link { to: Route::TopicsIndex {}, class: "px-2.5 py-2 rounded-md hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-700 dark:text-slate-200", onclick: move |_| show_mobile_menu.set(false), "{t(lang(), \"nav.forum\")}" }
            }
            if on_about {
              Link { to: Route::About {}, class: "px-2.5 py-2 rounded-md hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-700 dark:text-slate-200", onclick: move |_| show_mobile_menu.set(false), "{t(lang(), \"footer.about\")}" }
            }
            div { class: "sm:hidden mt-3 pt-3 border-t border-slate-200/70 dark:border-slate-800 flex items-center gap-2",
              ThemePicker {}
              LangPicker {}
            }
          }
        }
      }

      main { class: "flex-1",
        Outlet::<Route> {}
      }

      footer { class: "border-t border-slate-200 dark:border-slate-800 bg-slate-50/70 dark:bg-slate-950 shrink-0",
        Container {
          div { class: "py-12 grid grid-cols-2 md:grid-cols-5 gap-8 text-sm",
            div { class: "col-span-2 md:col-span-2",
              Link {
                to: Route::Home {},
                class: "font-extrabold text-flow text-base inline-flex items-center gap-2",
                super::BrandMark {}
              }
              p { class: "mt-3 max-w-sm text-slate-500 dark:text-slate-400 leading-relaxed",
                "{site_cfg.footer_tagline(lang())}"
              }
            }

            if !site_cfg.footer.columns.is_empty() {
              for (col_idx, col) in site_cfg.footer.columns.iter().enumerate() {
                div { key: "col-{col_idx}",
                  p { class: "text-xs font-semibold uppercase tracking-wider text-slate-900 dark:text-slate-200 mb-3",
                    "{col.title_for(lang())}"
                  }
                  div { class: "flex flex-col gap-2 text-slate-600 dark:text-slate-400",
                    for (lnk_idx, lnk) in col.links.iter().enumerate() {
                      if is_external_or_raw_href(&lnk.href) {
                        a {
                          key: "lnk-{col_idx}-{lnk_idx}",
                          href: "{lnk.href}",
                          target: if lnk.href.starts_with("http") { "_blank" },
                          rel: if lnk.href.starts_with("http") { "noopener noreferrer" },
                          class: "hover:text-[var(--color-primary)] transition-colors inline-flex items-center gap-1",
                          "{lnk.label_for(lang())}"
                        }
                      } else {
                        Link {
                          key: "lnk-{col_idx}-{lnk_idx}",
                          to: lnk.href.clone(),
                          class: "hover:text-[var(--color-primary)] transition-colors",
                          "{lnk.label_for(lang())}"
                        }
                      }
                    }
                  }
                }
              }
            } else {
              div {
                p { class: "text-xs font-semibold uppercase tracking-wider text-slate-900 dark:text-slate-200 mb-3",
                  "{t(lang(), \"footer.col.content\")}"
                }
                div { class: "flex flex-col gap-2 text-slate-600 dark:text-slate-400",
                  if on_docs {
                    Link { to: Route::Docs {}, class: "hover:text-[var(--color-primary)] transition-colors", "{t(lang(), \"nav.docs\")}" }
                  }
                  if on_blog {
                    Link { to: Route::BlogIndex {}, class: "hover:text-[var(--color-primary)] transition-colors", "{t(lang(), \"nav.blog\")}" }
                  }
                  if on_cases {
                    Link { to: Route::Cases {}, class: "hover:text-[var(--color-primary)] transition-colors", "{t(lang(), \"nav.cases\")}" }
                  }
                }
              }
              div {
                p { class: "text-xs font-semibold uppercase tracking-wider text-slate-900 dark:text-slate-200 mb-3",
                  "{t(lang(), \"footer.col.community\")}"
                }
                div { class: "flex flex-col gap-2 text-slate-600 dark:text-slate-400",
                  a { href: "/feed.xml", class: "hover:text-[var(--color-primary)] transition-colors", "{t(lang(), \"footer.rss\")}" }
                  if on_about {
                    Link { to: Route::About {}, class: "hover:text-[var(--color-primary)] transition-colors", "{t(lang(), \"footer.about\")}" }
                  }
                  for item in site_cfg.footer.social.iter().filter(|s| !s.label.trim().is_empty() && !s.url.trim().is_empty()) {
                    a {
                      key: "{item.label}-{item.url}",
                      href: "{item.url}",
                      target: "_blank",
                      rel: "noopener noreferrer",
                      class: "hover:text-[var(--color-primary)] transition-colors",
                      "{item.label}"
                    }
                  }
                }
              }
            }
          }

          div { class: "py-5 border-t border-slate-200/70 dark:border-slate-900 text-xs text-slate-400 flex flex-wrap items-center justify-between gap-2",
            span { "© {site_cfg.footer_copyright()}" }
            if let Some((icp, icp_url)) = site_cfg.footer_icp() {
              a {
                href: "{icp_url}",
                target: "_blank",
                rel: "noopener noreferrer",
                class: "hover:text-slate-600 dark:hover:text-slate-300 transition-colors",
                "{icp}"
              }
            }
          }
        }
      }
    }
  }
}
