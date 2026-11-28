//! 板块落地页与文章详情页。站内导航用 `Link { to: String }`，不依赖 app 的 `Route`。
//!
//! 重构 B3：数据经 `use_server_future` 服务端预取并随 SSR HTML 下发，
//! 客户端 hydration 直接拿到内容。子主题筛选 / 搜索是客户端 signal 交互，
//! 置于 `SuspenseBoundary` 内的子组件。

use dioxus::prelude::*;
use dioxus_shadcn::{
  badge_class, card_class, BadgeVariant, Empty, EmptyDescription, Input, Spinner, SpinnerSize,
};
use widgets::{get_seo_base_url, inject_seo, parse_mdx, LoadFailed, Markdown, NotFound, PageMeta};

use app_core::i18n::{t, Language};

use crate::server::{get_board_article, get_board_hub, list_board_articles};
use crate::{matches_query, normalize_tag, ArticleSummary, Board, BoardHubItem, BoardHubResources};

/// 读取全局语言信号（缺省回退 Zh）。方案 A：板块文案随该信号切换。
fn current_lang() -> Language {
  try_consume_context::<Signal<Language>>().map(|s| s()).unwrap_or_default()
}

/// 子主题筛选 chip：选中为主色实心，否则为次要底色。
fn chip_class(active: bool) -> String {
  if active {
    badge_class(BadgeVariant::Default, "rounded-full px-3 py-1.5 text-sm")
  } else {
    badge_class(
      BadgeVariant::Secondary,
      "rounded-full px-3 py-1.5 text-sm font-medium hover:bg-accent",
    )
  }
}

#[component]
pub fn BoardIndex(board: Board) -> Element {
  let lang = current_lang();
  let label = t(lang, &format!("{}.label", board.id));
  let tagline = t(lang, &format!("{}.tagline", board.id));
  rsx! {
      PageMeta { title: label.clone(), description: tagline.clone(), path: board.route.to_string() }
      section { class: "py-12 bg-white dark:bg-slate-950",
          div { class: "max-w-6xl mx-auto px-4 sm:px-6",
              div { class: "mb-10",
                  h1 { class: "text-3xl sm:text-4xl font-extrabold tracking-tight text-slate-900 dark:text-white", "{label}" }
                  p { class: "mt-3 text-lg text-slate-500 dark:text-slate-400 max-w-2xl", "{tagline}" }
              }
              // 文章列表经 use_server_future 服务端预取；SuspenseBoundary 在未就绪时渲染 spinner。
              SuspenseBoundary {
                  fallback: |_| rsx! {
                      div { class: "flex items-center justify-center py-20",
                          Spinner { size: SpinnerSize::Lg, class: "border-t-primary" }
                      }
                  },
                  BoardIndexList { board }
              }
          }
      }
  }
}

#[component]
fn BoardIndexList(board: Board) -> Element {
  let lang = current_lang();
  let id = board.id;
  let lang_param = match lang {
    Language::En => "en",
    Language::Zh => "zh",
  };
  let articles_res = use_server_future(move || async move {
    list_board_articles(id.to_string()).await.unwrap_or_default()
  })?;
  let hub_res = use_server_future(use_reactive!(|lang_param| async move {
    get_board_hub(id.to_string(), Some(lang_param.to_string())).await.unwrap_or_default()
  }))?;
  let articles: Vec<ArticleSummary> = articles_res().unwrap_or_default();
  let hub: BoardHubResources = hub_res().unwrap_or_default();

  let mut active_subtopic = use_signal(String::new);
  let mut query = use_signal(String::new);

  let subtopics = board.subtopics_with_articles(&articles);
  let q = query();
  let sub = active_subtopic();
  let filtered: Vec<ArticleSummary> = articles
    .iter()
    .filter(|a| sub.is_empty() || a.subtopic == sub)
    .filter(|a| matches_query(&a.title, &a.description, &a.tags, &q))
    .cloned()
    .collect();

  let has_hub = !hub.is_empty();

  rsx! {
      if has_hub {
          BoardHubOverview { lang, hub: hub.clone() }
      }

      div { class: if has_hub { "pt-8 border-t border-border" } else { "" },
          if has_hub {
              h2 { class: "text-xl font-bold text-slate-900 dark:text-white mb-5",
                  "{t(lang, \"board.hub.articles\")}"
              }
          }

          div { class: "mb-6",
              Input {
                  r#type: "search",
                  class: "max-w-md",
                  placeholder: t(lang, "board.search"),
                  "aria-label": t(lang, "board.search"),
                  value: query(),
                  on_value_change: move |v: String| query.set(v),
              }
          }

          div { class: "flex flex-wrap gap-2 mb-8",
              button {
                  r#type: "button",
                  "aria-pressed": sub.is_empty().to_string(),
                  class: chip_class(sub.is_empty()),
                  onclick: move |_| active_subtopic.set(String::new()),
                  "{t(lang, \"board.all\")}"
              }
              for s in subtopics.iter() {
                  {
                      let slug = s.to_string();
                      let is_active = sub == *s;
                      let chip = t(lang, &format!("{}.sub.{}.label", id, s));
                      let blurb = t(lang, &format!("{}.sub.{}.blurb", id, s));
                      rsx! {
                          button {
                              r#type: "button",
                              "aria-pressed": is_active.to_string(),
                              class: chip_class(is_active),
                              title: "{blurb}",
                              onclick: move |_| active_subtopic.set(slug.clone()),
                              "{chip}"
                          }
                      }
                  }
              }
          }

          div { class: "grid grid-cols-1 lg:grid-cols-3 gap-8",
              div { class: "lg:col-span-2",
                  if filtered.is_empty() {
                      Empty { class: "py-16",
                          EmptyDescription { "{t(lang, \"board.empty\")}" }
                      }
                  } else {
                      div { class: "space-y-4",
                          for a in filtered.iter() {
                              ArticleCard { key: "{a.slug}", board, article: a.clone() }
                          }
                      }
                  }
              }

              aside {
                  h2 { class: "text-sm font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400 mb-4", "{t(lang, \"board.featured\")}" }
                  div { class: "space-y-3",
                      for c in board.featured_crates.iter() {
                          {
                              let blurb = t(lang, &format!("{}.crate.{}.blurb", id, normalize_tag(c.name)));
                              rsx! {
                                  a {
                                      href: "{c.url}",
                                      target: "_blank",
                                      rel: "noopener noreferrer",
                                      class: "block p-3 rounded-lg border border-border hover:border-primary/50 transition-colors",
                                      div { class: "font-mono text-sm font-bold text-slate-900 dark:text-white", "{c.name}" }
                                      div { class: "text-xs text-slate-500 dark:text-slate-400 mt-0.5", "{blurb}" }
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

#[component]
fn BoardHubOverview(lang: Language, hub: BoardHubResources) -> Element {
  rsx! {
      div { class: "space-y-10 mb-12",
          // 领域资源统计条（仅显示非空分类）
          div { class: "flex flex-wrap items-center gap-2 text-xs",
              if !hub.docs.is_empty() {
                  span { class: badge_class(BadgeVariant::Secondary, "px-2.5 py-1 font-medium"),
                      "{t(lang, \"board.hub.docs\")} · {hub.docs.len()}"
                  }
              }
              if !hub.courses.is_empty() {
                  span { class: badge_class(BadgeVariant::Secondary, "px-2.5 py-1 font-medium"),
                      "{t(lang, \"board.hub.courses\")} · {hub.courses.len()}"
                  }
              }
              if !hub.cases.is_empty() {
                  span { class: badge_class(BadgeVariant::Secondary, "px-2.5 py-1 font-medium"),
                      "{t(lang, \"board.hub.cases\")} · {hub.cases.len()}"
                  }
              }
              if !hub.posts.is_empty() {
                  span { class: badge_class(BadgeVariant::Secondary, "px-2.5 py-1 font-medium"),
                      "{t(lang, \"board.hub.posts\")} · {hub.posts.len()}"
                  }
              }
          }

          if !hub.docs.is_empty() {
              HubResourceSection {
                  title: t(lang, "board.hub.docs"),
                  view_all_label: t(lang, "board.hub.view_all"),
                  view_all_href: "/docs".to_string(),
                  items: hub.docs.clone(),
              }
          }
          if !hub.courses.is_empty() {
              HubResourceSection {
                  title: t(lang, "board.hub.courses"),
                  view_all_label: t(lang, "board.hub.view_all"),
                  view_all_href: "/course".to_string(),
                  items: hub.courses.clone(),
              }
          }
          if !hub.cases.is_empty() {
              HubResourceSection {
                  title: t(lang, "board.hub.cases"),
                  view_all_label: t(lang, "board.hub.view_all"),
                  view_all_href: "/case".to_string(),
                  items: hub.cases.clone(),
              }
          }
          if !hub.posts.is_empty() {
              HubResourceSection {
                  title: t(lang, "board.hub.posts"),
                  view_all_label: t(lang, "board.hub.view_all"),
                  view_all_href: "/blog".to_string(),
                  items: hub.posts.clone(),
              }
          }
      }
  }
}

#[component]
fn HubResourceSection(
  title: String,
  view_all_label: String,
  view_all_href: String,
  items: Vec<BoardHubItem>,
) -> Element {
  rsx! {
      section { class: "space-y-4",
          div { class: "flex items-center justify-between gap-4",
              h2 { class: "text-xl font-bold text-slate-900 dark:text-white", "{title}" }
              Link {
                  to: view_all_href,
                  class: "text-sm font-medium text-primary hover:underline shrink-0",
                  "{view_all_label}"
              }
          }
          div { class: "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4",
              for item in items.iter() {
                  Link {
                      key: "{item.href}",
                      to: item.href.clone(),
                      class: card_class("group flex flex-col justify-between p-4 rounded-xl shadow-none hover:shadow-md hover:border-primary/50 transition-all"),
                      div {
                          div { class: "flex items-start justify-between gap-2",
                              h3 { class: "text-base font-bold text-slate-900 dark:text-white group-hover:text-primary transition-colors line-clamp-1",
                                  "{item.title}"
                              }
                              if !item.meta.is_empty() {
                                  span { class: badge_class(BadgeVariant::Secondary, "shrink-0 text-[11px] font-medium"),
                                      "{item.meta}"
                                  }
                              }
                          }
                          if !item.description.is_empty() {
                              p { class: "mt-1.5 text-sm text-slate-600 dark:text-slate-400 line-clamp-2",
                                  "{item.description}"
                              }
                          }
                      }
                      if !item.tags.is_empty() {
                          div { class: "mt-3 flex flex-wrap gap-1.5",
                              for tag in item.tags.iter().take(4) {
                                  span { class: badge_class(BadgeVariant::Secondary, "font-normal text-[11px] text-muted-foreground"),
                                      "#{tag}"
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

#[component]
fn ArticleCard(board: Board, article: ArticleSummary) -> Element {
  let lang = current_lang();
  let href = format!("{}/{}", board.route, article.slug);
  let sub = if board.subtopics.contains(&article.subtopic.as_str()) {
    t(lang, &format!("{}.sub.{}.label", board.id, article.subtopic))
  } else {
    String::new()
  };
  rsx! {
      Link {
          to: href,
          class: card_class("block p-5 rounded-xl shadow-none hover:shadow-md hover:border-primary/50 transition-all"),
          div { class: "flex items-center gap-2 mb-2 text-xs",
              if !sub.is_empty() {
                  span { class: badge_class(BadgeVariant::Secondary, "font-medium text-primary"), "{sub}" }
              }
              span { class: "text-slate-400", "{article.date}" }
          }
          h3 { class: "text-lg font-bold text-slate-900 dark:text-white", "{article.title}" }
          p { class: "mt-1 text-sm text-slate-600 dark:text-slate-400", "{article.description}" }
          if !article.tags.is_empty() {
              div { class: "mt-3 flex flex-wrap gap-1.5",
                  for tag in article.tags.iter() {
                      span { class: badge_class(BadgeVariant::Secondary, "font-normal text-muted-foreground"), "#{tag}" }
                  }
              }
          }
      }
  }
}

#[component]
pub fn BoardArticle(board: Board, slug: String) -> Element {
  let lang = current_lang();
  let back =
    format!("{} {}", t(lang, "board.back_prefix"), t(lang, &format!("{}.label", board.id)));
  rsx! {
      section { class: "py-12 bg-white dark:bg-slate-950",
          div { class: "max-w-4xl mx-auto px-4 sm:px-6",
              Link {
                  to: board.route,
                  class: "inline-flex items-center gap-1 text-sm text-primary hover:underline mb-8",
                  "{back}"
              }
              div { class: "text-slate-700 dark:text-slate-200",
                  // 正文经 use_server_future 服务端预取（随 SSR HTML 下发）。
                  SuspenseBoundary {
                      fallback: |_| rsx! {
                          div { class: "flex items-center justify-center py-20",
                              Spinner { size: SpinnerSize::Lg, class: "border-t-primary" }
                          }
                      },
                      BoardArticleContent { board, slug: slug.clone() }
                  }
              }
          }
      }
  }
}

/// 文章正文：`use_reactive!` 让路由参数 slug 变化时重取。
#[component]
fn BoardArticleContent(board: Board, slug: String) -> Element {
  let lang = current_lang();
  let id = board.id;
  let content_res = use_server_future(use_reactive!(|slug| async move {
    get_board_article(id.to_string(), slug).await
  }))?;
  let base_url_res =
    use_server_future(|| async move { get_seo_base_url().await.unwrap_or_default() })?;
  let base_url = base_url_res().unwrap_or_default();
  match content_res() {
    Some(Ok(Some(content))) => {
      let (meta, _body) = parse_mdx(&content);
      let path = format!("{}/{}", board.route, slug);
      let date = meta.date.clone().unwrap_or_default();
      rsx! {
          {inject_seo(&meta, &path, &base_url)}
          if !date.is_empty() || !meta.tags.is_empty() {
              div { class: "flex flex-wrap items-center gap-2 mb-4 text-xs",
                  if !date.is_empty() {
                      time { class: "text-slate-400", datetime: "{date}", "{date}" }
                  }
                  for tag in meta.tags.iter() {
                      span { class: badge_class(BadgeVariant::Secondary, "font-normal text-muted-foreground"), "#{tag}" }
                  }
              }
          }
          Markdown { content: content.clone(), blog_id: slug.clone() }
      }
    }
    Some(Ok(None)) => rsx! { NotFound { message: t(lang, "board.not_found") } },
    Some(Err(_)) => rsx! { LoadFailed { message: t(lang, "common.load_failed") } },
    None => rsx! {
        div { class: "flex items-center justify-center py-20",
            Spinner { size: SpinnerSize::Lg, class: "border-t-primary" }
        }
    },
  }
}
