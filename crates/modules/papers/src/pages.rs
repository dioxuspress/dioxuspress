use app_core::i18n::Language;
use dioxus::prelude::*;
use dioxus_shadcn::{Empty, EmptyDescription};

use crate::digest::{MindNode, PaperDigest};
use crate::server::{
  get_paper, list_papers_page, total_pages, PaperCard, PaperDetail, PaperPage, DEFAULT_PAGE_SIZE,
};
use crate::topics::{self, Topic, TOPICS};

fn use_language_signal() -> Signal<Language> {
  try_use_context::<Signal<Language>>().unwrap_or_else(|| Signal::new(Language::Zh))
}

/// 解读按界面语言取；存的语言代码与这里一致。
fn lang_code(lang: Language) -> &'static str {
  match lang {
    Language::En => "en",
    Language::Zh => "zh",
  }
}

/// `/papers`：按 HF 日期分组或热度排序分页列出已生成解读的论文。
#[component]
pub fn PapersIndexPage() -> Element {
  let lang = use_language_signal()();
  let title = lang.pick("论文速递", "Papers");
  let subtitle = lang.pick(
    "每天从 Hugging Face Daily Papers 选题，读 arXiv 全文，整理带出处的要点与思维导图（AI 生成）。",
    "Daily picks from Hugging Face Daily Papers, read in full from arXiv, with cited key points and a mind map (AI-generated).",
  );
  rsx! {
      widgets::PageMeta {
          title: title.to_string(),
          description: subtitle.to_string(),
          path: "/papers".to_string(),
          bilingual: Some(lang == Language::En),
      }
      section { class: "py-12 min-h-screen bg-white dark:bg-slate-950",
          div { class: "mx-auto max-w-6xl px-4 sm:px-6 lg:px-8",
              div { class: "flex flex-wrap items-center justify-between gap-3",
                  h1 { class: "text-3xl md:text-4xl font-extrabold text-slate-900 dark:text-white", "{title}" }
                  RssFeedLink {}
              }
              p { class: "mt-3 text-slate-600 dark:text-slate-400", "{subtitle}" }
              SuspenseBoundary { fallback: |_| rsx! {}, PaperList { topic: String::new() } }
          }
      }
  }
}

/// `/papers/topic/:slug`：一个主题下的论文。
#[component]
pub fn PaperTopicPage(slug: String) -> Element {
  let lang = use_language_signal()();
  let Some(topic) = topics::find(&slug) else {
    return rsx! { widgets::NotFound { message: lang.pick("没有这个主题", "Topic not found").to_string() } };
  };
  let name = topic_name(topic, lang);
  let about = lang.pick(topic.about_zh, topic.about_en);
  let title = format!("{name} · {}", lang.pick("论文速递", "Papers"));
  rsx! {
      widgets::PageMeta {
          title,
          description: about.to_string(),
          path: format!("/papers/topic/{}", topic.slug),
          bilingual: Some(lang == Language::En),
      }
      section { class: "py-12 min-h-screen bg-white dark:bg-slate-950",
          div { class: "mx-auto max-w-6xl px-4 sm:px-6 lg:px-8",
              a { href: "/papers", class: "text-sm text-slate-500 hover:underline", {lang.pick("← 论文速递", "← Papers")} }
              div { class: "mt-4 flex flex-wrap items-center justify-between gap-3",
                  h1 { class: "text-3xl md:text-4xl font-extrabold text-slate-900 dark:text-white", "{name}" }
                  RssFeedLink {}
              }
              p { class: "mt-3 text-slate-600 dark:text-slate-400", "{about}" }
              SuspenseBoundary { fallback: |_| rsx! {}, PaperList { topic: topic.slug.to_string() } }
          }
      }
  }
}

#[component]
fn RssFeedLink() -> Element {
  let lang = use_language_signal()();
  rsx! {
      a {
          href: "/papers/feed.xml",
          target: "_blank",
          rel: "noopener noreferrer",
          title: lang.pick("订阅论文 Atom RSS", "Subscribe to Papers Atom Feed"),
          class: "inline-flex items-center gap-1.5 rounded-full border border-slate-200 dark:border-slate-700 px-3 py-1 text-xs font-medium text-slate-600 dark:text-slate-300 hover:border-slate-400 dark:hover:border-slate-500 transition-colors",
          svg { class: "w-3.5 h-3.5 text-amber-500", fill: "none", stroke: "currentColor", view_box: "0 0 24 24", stroke_width: "2",
              path { stroke_linecap: "round", stroke_linejoin: "round", d: "M6 18a2 2 0 100-4 2 2 0 000 4zM4 11a9 9 0 019 9M4 4a16 16 0 0116 16" }
          }
          "RSS"
      }
  }
}

fn topic_name(t: &Topic, lang: Language) -> &'static str {
  match lang {
    Language::Zh => t.zh,
    Language::En => t.en,
  }
}

/// 桌面端左侧粘性垂直主题菜单 + 移动端顶部单行横向可滚动胶囊条。
/// `current` 是当前主题页的 slug，总览页为空（高亮「全部」）。
#[component]
fn TopicNav(current: String, all_total: u64, topic_counts: Vec<(String, u64)>) -> Element {
  let lang = use_language_signal()();
  let is_all = current.is_empty();
  let count_for =
    |slug: &str| topic_counts.iter().find(|(s, _)| s == slug).map(|(_, n)| *n).unwrap_or(0);
  let sidebar_link_class = |active: bool| {
    if active {
      "flex items-center justify-between gap-2 rounded-lg px-2.5 py-2 text-sm font-semibold bg-slate-900 text-white dark:bg-slate-100 dark:text-slate-900 transition-colors"
    } else {
      "flex items-center justify-between gap-2 rounded-lg px-2.5 py-2 text-sm text-slate-700 dark:text-slate-300 hover:bg-slate-200/70 dark:hover:bg-slate-800/80 hover:text-slate-900 dark:hover:text-white transition-colors"
    }
  };
  let sidebar_count_class = |active: bool| {
    if active {
      "text-xs opacity-75 tabular-nums shrink-0"
    } else {
      "text-xs text-slate-400 dark:text-slate-500 tabular-nums shrink-0"
    }
  };

  rsx! {
      // 移动端：顶部单行横向可滚动胶囊，避免折成多行挤占首屏
      nav {
          class: "flex lg:hidden items-center gap-2 overflow-x-auto pb-2 -mx-1 px-1",
          aria_label: lang.pick("主题", "Topics"),
          a {
              class: if is_all { "pt-chip pt-chip-on" } else { "pt-chip" },
              href: "/papers",
              aria_current: if is_all { Some("page") } else { None },
              {lang.pick("全部", "All")}
              if all_total > 0 {
                  span { class: "ml-1.5 text-xs opacity-65 tabular-nums", "{all_total}" }
              }
          }
          for t in TOPICS {
              {
                  let cnt = count_for(t.slug);
                  let active = t.slug == current;
                  rsx! {
                      a {
                          key: "m-{t.slug}",
                          class: if active { "pt-chip pt-chip-on" } else { "pt-chip" },
                          href: "/papers/topic/{t.slug}",
                          aria_current: if active { Some("page") } else { None },
                          {topic_name(t, lang)}
                          if all_total > 0 {
                              span { class: "ml-1.5 text-xs opacity-65 tabular-nums", "{cnt}" }
                          }
                      }
                  }
              }
          }
      }

      // 桌面端：左侧粘性垂直导航列表
      nav {
          class: "hidden lg:block rounded-xl border border-slate-200/80 dark:border-slate-800 bg-slate-50/60 dark:bg-slate-900/40 p-3",
          aria_label: lang.pick("主题分类", "Topics"),
          div { class: "px-2.5 pb-2 text-xs font-semibold tracking-wide text-slate-500 dark:text-slate-400",
              {lang.pick("主题分类", "Topics")}
          }
          ul { class: "space-y-0.5",
              li {
                  a {
                      class: sidebar_link_class(is_all),
                      href: "/papers",
                      aria_current: if is_all { Some("page") } else { None },
                      span { class: "truncate", {lang.pick("全部", "All")} }
                      if all_total > 0 {
                          span { class: sidebar_count_class(is_all), "{all_total}" }
                      }
                  }
              }
              for t in TOPICS {
                  {
                      let cnt = count_for(t.slug);
                      let active = t.slug == current;
                      rsx! {
                          li { key: "s-{t.slug}",
                              a {
                                  class: sidebar_link_class(active),
                                  href: "/papers/topic/{t.slug}",
                                  aria_current: if active { Some("page") } else { None },
                                  span { class: "truncate", {topic_name(t, lang)} }
                                  if all_total > 0 {
                                      span { class: sidebar_count_class(active), "{cnt}" }
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

/// 论文的主题标签，点了去主题页。
#[component]
fn TopicTags(slugs: Vec<String>) -> Element {
  let lang = use_language_signal()();
  rsx! {
      for t in slugs.iter().filter_map(|s| topics::find(s)) {
          a { key: "{t.slug}", class: "pt-tag", href: "/papers/topic/{t.slug}", {topic_name(t, lang)} }
      }
  }
}

/// 论文列表单项（同时用于按日期分组列表和热度排行列表）。
#[component]
fn PaperRow(p: PaperCard, show_date: bool) -> Element {
  let lang = use_language_signal()();
  rsx! {
      li { class: "py-4",
          div { class: "flex items-baseline gap-3",
              a {
                  href: "/papers/{p.arxiv_id}",
                  class: "flex-1 min-w-0 font-semibold text-slate-900 dark:text-slate-100 hover:underline break-words",
                  "{p.title}"
              }
              div { class: "shrink-0 flex items-center gap-2 text-xs text-slate-500 tabular-nums",
                  if p.has_code {
                      span {
                          class: "inline-flex items-center rounded-full bg-emerald-50 dark:bg-emerald-950/50 border border-emerald-200 dark:border-emerald-800/70 px-2 py-0.5 text-[11px] font-medium text-emerald-700 dark:text-emerald-300",
                          {lang.pick("Code", "Code")}
                      }
                  }
                  if show_date {
                      span { "{p.date}" }
                  }
                  span { "▲ {p.hf_upvotes}" }
              }
          }
          if !p.tldr.is_empty() {
              p { class: "mt-1 text-sm text-slate-600 dark:text-slate-400", "{p.tldr}" }
          }
          if !p.topics.is_empty() {
              div { class: "mt-2 flex flex-wrap gap-1.5", TopicTags { slugs: p.topics.clone() } }
          }
      }
  }
}

#[component]
fn PaperList(topic: String) -> Element {
  let lang_signal = use_language_signal();
  let lang = lang_signal();
  let code_now = lang_code(lang).to_string();

  let first = use_server_future(use_reactive!(|topic| {
    let code = lang_code(lang_signal()).to_string();
    async move { list_papers_page(code, topic, "latest".to_string(), false, 1, DEFAULT_PAGE_SIZE).await }
  }))?;

  let mut sort = use_signal(|| "latest".to_string());
  let mut code_only = use_signal(|| false);
  let mut loading = use_signal(|| false);
  let mut load_err = use_signal(|| false);
  // 连同 `(topic, lang)` 一起缓存，切主题或切语言时自然回退到 SSR/新语言首屏
  let mut loaded = use_signal::<Option<((String, String), PaperPage)>>(|| None);

  let cache_key = (topic.clone(), code_now.clone());
  let topic_for_load = topic.clone();
  let load = move |next_sort: String, next_code: bool, next_page: u64| {
    sort.set(next_sort.clone());
    code_only.set(next_code);
    let t = topic_for_load.clone();
    let code = lang_code(lang_signal()).to_string();
    let key = (t.clone(), code.clone());
    spawn(async move {
      loading.set(true);
      load_err.set(false);
      match list_papers_page(code, t, next_sort, next_code, next_page, DEFAULT_PAGE_SIZE).await {
        Ok(p) => loaded.set(Some((key, p))),
        Err(_) => load_err.set(true),
      }
      loading.set(false);
    });
  };

  let active_loaded = loaded().filter(|(k, _)| *k == cache_key).map(|(_, p)| p);
  let page_data = match active_loaded {
    Some(p) => p,
    None => match first() {
      Some(Ok(p)) => p,
      Some(Err(_)) => {
        return rsx! { widgets::LoadFailed { message: lang.pick("加载失败", "Failed to load").to_string() } }
      }
      None => PaperPage::default(),
    },
  };

  let cur_sort = sort();
  let cur_code = code_only();
  let cur_page = page_data.page.max(1);
  let pages_total = total_pages(page_data.total, page_data.page_size);
  let sort_btn_class = |active: bool| {
    if active {
      "rounded-lg bg-slate-900 dark:bg-slate-100 text-white dark:text-slate-900 px-3 py-1.5 text-xs font-semibold transition-colors cursor-pointer"
    } else {
      "rounded-lg border border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-300 hover:border-slate-400 dark:hover:border-slate-500 px-3 py-1.5 text-xs font-medium transition-colors cursor-pointer"
    }
  };
  let code_btn_class = if cur_code {
    "rounded-lg bg-emerald-600 text-white px-3 py-1.5 text-xs font-semibold transition-colors cursor-pointer"
  } else {
    "rounded-lg border border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-300 hover:border-emerald-500 hover:text-emerald-600 dark:hover:text-emerald-400 px-3 py-1.5 text-xs font-medium transition-colors cursor-pointer"
  };

  let mut load_latest = load.clone();
  let mut load_week = load.clone();
  let mut load_all = load.clone();
  let mut load_code = load.clone();
  let mut load_prev = load.clone();
  let mut load_next = load;

  let count_summary = if lang == Language::En {
    format!("{} papers", page_data.total)
  } else {
    format!("共 {} 篇", page_data.total)
  };
  let page_indicator = if lang == Language::En {
    format!("Page {} / {}", cur_page, pages_total.max(1))
  } else {
    format!("第 {} / {} 页", cur_page, pages_total.max(1))
  };

  rsx! {
      div { class: "mt-8 grid grid-cols-1 lg:grid-cols-[15rem_minmax(0,1fr)] gap-8 items-start",
          aside { class: "min-w-0 lg:sticky lg:top-20",
              TopicNav {
                  current: topic.clone(),
                  all_total: page_data.all_total,
                  topic_counts: page_data.topic_counts.clone(),
              }
          }
          div { class: "min-w-0",
              // 工具栏：排序切换 + 仅看有代码 + 命中篇数
              div { class: "mb-8 flex flex-wrap items-center justify-between gap-3 pb-4 border-b border-slate-200 dark:border-slate-800",
                  div { class: "flex flex-wrap items-center gap-2",
                      button {
                          r#type: "button",
                          "aria-pressed": (cur_sort == "latest").to_string(),
                          class: sort_btn_class(cur_sort == "latest"),
                          onclick: move |_| load_latest("latest".to_string(), code_only(), 1),
                          {lang.pick("最新", "Latest")}
                      }
                      button {
                          r#type: "button",
                          "aria-pressed": (cur_sort == "top_week").to_string(),
                          class: sort_btn_class(cur_sort == "top_week"),
                          onclick: move |_| load_week("top_week".to_string(), code_only(), 1),
                          {lang.pick("近 7 天最热", "Top 7d")}
                      }
                      button {
                          r#type: "button",
                          "aria-pressed": (cur_sort == "top_all").to_string(),
                          class: sort_btn_class(cur_sort == "top_all"),
                          onclick: move |_| load_all("top_all".to_string(), code_only(), 1),
                          {lang.pick("历史最热", "Top all")}
                      }
                      button {
                          r#type: "button",
                          "aria-pressed": cur_code.to_string(),
                          class: code_btn_class,
                          onclick: move |_| load_code(sort(), !code_only(), 1),
                          {lang.pick("仅看有代码", "Code only")}
                      }
                  }
                  span { class: "text-xs text-slate-500 dark:text-slate-400 tabular-nums",
                      {if loading() { lang.pick("加载中…", "Loading…").to_string() } else { count_summary }}
                  }
              }
              if load_err() {
                  p { class: "mb-6 text-sm text-rose-600 dark:text-rose-400",
                      {lang.pick("刷新列表失败，请重试。", "Failed to refresh list. Please try again.")}
                  }
              }
              if page_data.items.is_empty() {
                  Empty { class: "rounded-2xl py-20",
                      EmptyDescription { {lang.pick("当前筛选条件下还没有论文", "No papers match this filter")} }
                  }
              } else if cur_sort == "latest" {
                  for (date, day) in group_by_date(page_data.items) {
                      section { key: "{date}", class: "mb-10",
                          h2 { class: "text-sm font-semibold text-slate-500 dark:text-slate-400 mb-3", "{date}" }
                          ul { class: "divide-y divide-slate-200 dark:divide-slate-800",
                              for p in day {
                                  PaperRow { key: "{p.arxiv_id}", p, show_date: false }
                              }
                          }
                      }
                  }
              } else {
                  ul { class: "divide-y divide-slate-200 dark:divide-slate-800 mb-10",
                      for p in page_data.items {
                          PaperRow { key: "{p.arxiv_id}", p, show_date: true }
                      }
                  }
              }
              if pages_total > 1 {
                  nav {
                      class: "mt-8 pt-6 border-t border-slate-200 dark:border-slate-800 flex items-center justify-between gap-4",
                      aria_label: lang.pick("分页导航", "Pagination"),
                      button {
                          r#type: "button",
                          disabled: cur_page <= 1 || loading(),
                          class: "rounded-lg border border-slate-200 dark:border-slate-700 px-3.5 py-1.5 text-sm font-medium text-slate-700 dark:text-slate-200 hover:border-slate-400 disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer transition-colors",
                          onclick: move |_| {
                              if cur_page > 1 {
                                  load_prev(sort(), code_only(), cur_page - 1);
                              }
                          },
                          {lang.pick("← 上一页", "← Prev")}
                      }
                      span { class: "text-xs text-slate-500 dark:text-slate-400 tabular-nums", "{page_indicator}" }
                      button {
                          r#type: "button",
                          disabled: !page_data.has_more || loading(),
                          class: "rounded-lg border border-slate-200 dark:border-slate-700 px-3.5 py-1.5 text-sm font-medium text-slate-700 dark:text-slate-200 hover:border-slate-400 disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer transition-colors",
                          onclick: move |_| {
                              if page_data.has_more {
                                  load_next(sort(), code_only(), cur_page + 1);
                              }
                          },
                          {lang.pick("下一页 →", "Next →")}
                      }
                  }
              }
          }
      }
  }
}

/// 按日期分组；输入已按日期倒序，组的顺序随之保留。
fn group_by_date(papers: Vec<PaperCard>) -> Vec<(String, Vec<PaperCard>)> {
  let mut groups: Vec<(String, Vec<PaperCard>)> = Vec::new();
  for p in papers {
    match groups.last_mut() {
      Some((date, day)) if *date == p.date => day.push(p),
      _ => groups.push((p.date.clone(), vec![p])),
    }
  }
  groups
}

/// 拼装论文的 BibTeX 引用文本（纯 Rust，无需外部依赖）。
pub fn format_bibtex(paper: &PaperDetail) -> String {
  let key_suffix: String =
    paper.arxiv_id.chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).collect();
  let year: String = paper
    .published
    .chars()
    .take(4)
    .collect::<String>()
    .chars()
    .filter(|c| c.is_ascii_digit())
    .collect();
  let year = if year.len() == 4 { year } else { paper.date.chars().take(4).collect() };
  let escape = |s: &str| s.replace('{', "\\{").replace('}', "\\}");
  let authors = escape(&paper.authors.join(" and "));
  let title = escape(&paper.title);
  format!(
    "@article{{arxiv_{key_suffix},\n  title={{{title}}},\n  author={{{authors}}},\n  year={{{year}}},\n  eprint={{{id}}},\n  archivePrefix={{arXiv}},\n  url={{https://arxiv.org/abs/{id}}}\n}}",
    id = paper.arxiv_id
  )
}

/// `/papers/:arxiv_id`：单篇解读。`footer` 插槽由组合根 app 注入论坛讨论面板。
#[component]
pub fn PaperDetailPage(
  arxiv_id: String,
  #[props(default = VNode::empty())] footer: Element,
) -> Element {
  rsx! {
      section { class: "py-12 min-h-screen bg-white dark:bg-slate-950",
          div { class: "mx-auto max-w-3xl px-4 sm:px-6 lg:px-8",
              SuspenseBoundary {
                  fallback: |_| rsx! {},
                  PaperDetailLoaded { arxiv_id, footer: footer.clone() }
              }
          }
      }
  }
}

#[component]
fn PaperDetailLoaded(arxiv_id: String, footer: Element) -> Element {
  let lang_signal = use_language_signal();
  let lang = lang_signal();
  let res = use_server_future(use_reactive!(|arxiv_id| {
    let code = lang_code(lang_signal()).to_string();
    async move { get_paper(arxiv_id, code).await }
  }))?;
  match res() {
    // 已经在 SuspenseBoundary 里，用 PageMetaTags（PageMeta 自带的那层会让 hydration 错位）
    Some(Ok(Some(paper))) => rsx! {
        widgets::PageMetaTags {
            title: display_title(&paper, lang).to_string(),
            description: paper.digest.tldr.clone(),
            path: format!("/papers/{}", paper.arxiv_id),
            schema: Some(paper_schema(&paper, lang)),
            bilingual: Some(lang == Language::En),
        }
        PaperBody { paper, wanted: lang_code(lang).to_string(), footer: footer.clone() }
    },
    Some(Err(_)) => {
      rsx! { widgets::LoadFailed { message: lang.pick("加载失败", "Failed to load").to_string() } }
    }
    _ => {
      rsx! { widgets::NotFound { message: lang.pick("没有这篇论文", "Paper not found").to_string() } }
    }
  }
}

/// 中文界面有中文标题时用中文，否则原文标题。
fn display_title(paper: &PaperDetail, lang: Language) -> &str {
  match (&paper.title_zh, lang) {
    (Some(zh), Language::Zh) => zh,
    _ => &paper.title,
  }
}

/// 页面的 JSON-LD：本页是一篇解读（`Article`），讲的是 arXiv 上那篇论文（`about`）。
fn paper_schema(paper: &PaperDetail, lang: Language) -> serde_json::Value {
  let id = &paper.arxiv_id;
  let authors: Vec<serde_json::Value> =
    paper.authors.iter().map(|a| serde_json::json!({ "@type": "Person", "name": a })).collect();
  let keywords: Vec<&str> =
    paper.topics.iter().filter_map(|s| topics::find(s)).map(|t| topic_name(t, lang)).collect();
  let mut about = serde_json::json!({
    "@type": "ScholarlyArticle",
    "name": paper.title,
    "url": format!("https://arxiv.org/abs/{id}"),
    "identifier": format!("arXiv:{id}"),
    "author": authors,
    "datePublished": paper.published,
  });
  if let Some(zh) = &paper.title_zh {
    about["alternateName"] = zh.as_str().into();
  }
  serde_json::json!({
    "@type": "Article",
    "headline": display_title(paper, lang),
    "description": paper.digest.tldr,
    "inLanguage": match lang { Language::Zh => "zh-CN", Language::En => "en" },
    "datePublished": paper.date,
    "keywords": keywords,
    "about": about,
    "isPartOf": { "@type": "CollectionPage", "name": lang.pick("论文速递", "Papers") },
  })
}

#[component]
fn PaperBody(paper: PaperDetail, wanted: String, footer: Element) -> Element {
  let lang = use_language_signal()();
  let mut bibtex_copied = use_signal(|| false);
  let bibtex = format_bibtex(&paper);
  let d: &PaperDigest = &paper.digest;
  let id = &paper.arxiv_id;
  let version = paper.arxiv_version.map(|v| format!("v{v}")).unwrap_or_default();
  let authors = match paper.authors.len() {
    0 => String::new(),
    n if n > 6 => format!("{} et al.", paper.authors[..6].join(", ")),
    _ => paper.authors.join(", "),
  };
  let section_label = |s: &str| format!("§{}", crate::digest::clean_section(s));
  let link = "underline decoration-slate-300 hover:decoration-slate-500";
  rsx! {
      a { href: "/papers", class: "text-sm text-slate-500 hover:underline", {lang.pick("← 论文速递", "← Papers")} }
      h1 { class: "mt-4 text-2xl md:text-3xl font-extrabold text-slate-900 dark:text-white break-words", "{display_title(&paper, lang)}" }
      // 原文标题：用 hidden 而不是 if。切换语言时 if 块在 hydration 后被增删，
      // dioxus 0.7.9 报 `cannot reclaim ElementId`（2026-11-04 本地二分确认）。
      p {
          class: "mt-2 text-slate-600 dark:text-slate-400 break-words",
          lang: "en",
          hidden: display_title(&paper, lang) == paper.title,
          "{paper.title}"
      }
      if !paper.topics.is_empty() {
          div { class: "mt-3 flex flex-wrap gap-1.5", TopicTags { slugs: paper.topics.clone() } }
      }
      if !authors.is_empty() {
          p { class: "mt-2 text-sm text-slate-500 break-words", "{authors}" }
      }
      div { class: "mt-3 flex flex-wrap items-center gap-x-4 gap-y-2 text-sm text-slate-600 dark:text-slate-400",
          span { "{paper.date}" }
          span { "▲ {paper.hf_upvotes}" }
          a { class: link, href: "https://arxiv.org/abs/{id}{version}", target: "_blank", rel: "noopener noreferrer", "arXiv {id}{version}" }
          a { class: link, href: "https://huggingface.co/papers/{id}", target: "_blank", rel: "noopener noreferrer", "Hugging Face" }
          if let Some(repo) = paper.github_repo.as_ref().filter(|r| r.starts_with("https://")) {
              a { class: link, href: "{repo}", target: "_blank", rel: "noopener noreferrer", {lang.pick("代码", "Code")} }
          }
          button {
              r#type: "button",
              class: "inline-flex items-center rounded-md border border-slate-200 dark:border-slate-700 px-2.5 py-0.5 text-xs font-medium text-slate-600 dark:text-slate-300 hover:border-slate-400 dark:hover:border-slate-500 cursor-pointer transition-colors",
              onclick: move |_| {
                  let text = bibtex.clone();
                  spawn(async move {
                      if widgets::browser::copy_text(&text).await {
                          bibtex_copied.set(true);
                          widgets::browser::sleep_ms(2000).await;
                          bibtex_copied.set(false);
                      }
                  });
              },
              {if bibtex_copied() { lang.pick("已复制 BibTeX", "BibTeX copied") } else { lang.pick("复制 BibTeX", "Copy BibTeX") }}
          }
      }
      div { class: "mt-6 rounded-lg border border-amber-300/60 bg-amber-50 dark:bg-amber-950/30 dark:border-amber-800/60 px-4 py-3 text-sm text-amber-900 dark:text-amber-200",
          p {
              {lang.pick("本页由 AI 根据论文全文生成（", "AI-generated from the full paper (")}
              "{paper.model}"
              {lang.pick("），可能有误，请以原文为准。", "). It may contain mistakes; check the original.")}
          }
          if !paper.has_html {
              p { class: "mt-1", {lang.pick("这篇论文没有 HTML 全文，解读只依据摘要。", "No HTML full text was available; this digest is based on the abstract only.")} }
          }
          if paper.lang != wanted {
              p { class: "mt-1", {lang.pick("暂无中文版，下面显示英文。", "No version in this language yet; showing English.")} }
          }
      }

      DigestSection { title: lang.pick("一句话", "TL;DR").to_string(),
          p { class: "text-lg text-slate-800 dark:text-slate-200", "{d.tldr}" }
      }
      DigestSection { title: lang.pick("要解决的问题", "Problem").to_string(),
          p { "{d.problem}" }
      }
      DigestSection { title: lang.pick("方法", "Method").to_string(),
          p { "{d.method}" }
      }
      DigestSection { title: lang.pick("实验结果", "Results").to_string(),
          ul { class: "list-disc pl-5 space-y-2",
              for (i, r) in d.results.iter().enumerate() {
                  li { key: "{i}",
                      "{r.claim} "
                      span { class: "text-xs text-slate-500", "{section_label(&r.section)}" }
                      for n in r.numbers.iter().filter(|n| paper.unverified_numbers.contains(n)) {
                          span { class: "ml-2 text-xs text-amber-700 dark:text-amber-400",
                              "{n} "
                              {lang.pick("（原文中未核实）", "(not found in the paper)")}
                          }
                      }
                  }
              }
          }
      }
      DigestSection { title: lang.pick("局限", "Limitations").to_string(),
          ul { class: "list-disc pl-5 space-y-1",
              for (i, l) in d.limitations.iter().enumerate() {
                  li { key: "{i}", "{l}" }
              }
          }
      }
      DigestSection { title: lang.pick("要点", "Key points").to_string(),
          ul { class: "list-disc pl-5 space-y-2",
              for (i, k) in d.key_points.iter().enumerate() {
                  li { key: "{i}",
                      "{k.text} "
                      span { class: "text-xs text-slate-500", "{section_label(&k.section)}" }
                  }
              }
          }
      }
      DigestSection { title: lang.pick("思维导图", "Mind map").to_string(),
          div { class: "mm overflow-x-auto", {mind_node(&d.mindmap, 0, "")} }
      }
      if !paper.related.is_empty() {
          DigestSection { title: lang.pick("相关论文", "Related papers").to_string(),
              div { class: "grid grid-cols-1 sm:grid-cols-2 gap-4",
                  for rel in paper.related.iter() {
                      a {
                          key: "{rel.arxiv_id}",
                          href: "/papers/{rel.arxiv_id}",
                          class: "group block rounded-xl border border-slate-200 dark:border-slate-800 p-4 hover:border-slate-400 dark:hover:border-slate-600 transition-colors",
                          div { class: "flex items-center justify-between gap-2 text-xs text-slate-500 dark:text-slate-400 tabular-nums mb-1.5",
                              span { "{rel.date}" }
                              div { class: "flex items-center gap-2",
                                  if rel.has_code {
                                      span { class: "rounded-full bg-emerald-50 dark:bg-emerald-950/50 border border-emerald-200 dark:border-emerald-800/70 px-1.5 py-0.5 text-[10px] font-medium text-emerald-700 dark:text-emerald-300", "Code" }
                                  }
                                  span { "▲ {rel.hf_upvotes}" }
                              }
                          }
                          h3 { class: "text-sm font-semibold text-slate-900 dark:text-slate-100 group-hover:underline line-clamp-2", "{rel.title}" }
                          if !rel.tldr.is_empty() {
                              p { class: "mt-1.5 text-xs text-slate-600 dark:text-slate-400 line-clamp-2", "{rel.tldr}" }
                          }
                      }
                  }
              }
          }
      }
      div { class: "mt-12", {footer} }
  }
}

/// 一级分支的颜色；子节点沿用所在分支的颜色。导图每层最多 6 个子节点（`MINDMAP_MAX_CHILDREN`）。
const BRANCH_COLORS: [&str; 6] = ["#0d9488", "#2563eb", "#7c3aed", "#d97706", "#db2777", "#65a30d"];

/// 思维导图的一个节点及其子树。样式在 `main.css` 的 `.mm` 段：
/// 宽屏是从左往右的树，窄屏退成缩进大纲。
fn mind_node(node: &MindNode, depth: usize, color: &str) -> Element {
  // 二级及以下：有子节点的是小标题，叶子一律是标签，不管树有几层。
  let level = match depth {
    0 => "mm-root",
    1 => "mm-d1",
    _ if node.children.is_empty() => "mm-leaf",
    _ => "mm-d2",
  };
  let style = if color.is_empty() { String::new() } else { format!("--c: {color}") };
  rsx! {
      div { class: "mm-node {level}", style: "{style}",
          span { class: "mm-label", "{node.label}" }
          if !node.children.is_empty() {
              div { class: "mm-kids",
                  for (i, child) in node.children.iter().enumerate() {
                      {
                          let c = if depth == 0 { BRANCH_COLORS[i % BRANCH_COLORS.len()] } else { color };
                          mind_node(child, depth + 1, c)
                      }
                  }
              }
          }
      }
  }
}

#[component]
fn DigestSection(title: String, children: Element) -> Element {
  rsx! {
      section { class: "mt-10",
          h2 { class: "text-lg font-bold text-slate-900 dark:text-white mb-3", "{title}" }
          div { class: "text-slate-700 dark:text-slate-300 leading-relaxed break-words", {children} }
      }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn card(id: &str, date: &str) -> PaperCard {
    PaperCard {
      arxiv_id: id.into(),
      title: id.into(),
      date: date.into(),
      hf_upvotes: 0,
      tldr: String::new(),
      topics: Vec::new(),
      has_code: false,
      has_html: true,
    }
  }

  #[test]
  fn groups_consecutive_dates_in_order() {
    let groups = group_by_date(vec![
      card("a", "2026-10-28"),
      card("b", "2026-10-28"),
      card("c", "2026-10-27"),
    ]);
    let shape: Vec<(String, usize)> = groups.iter().map(|(d, v)| (d.clone(), v.len())).collect();
    assert_eq!(shape, vec![("2026-10-28".into(), 2), ("2026-10-27".into(), 1)]);
  }

  #[test]
  fn formats_bibtex_with_sanitized_key_and_authors() {
    let detail = PaperDetail {
      arxiv_id: "2611.15789".into(),
      title: "Fast {Attention} for Agents".into(),
      title_zh: Some("智能体快速注意力".into()),
      topics: vec!["agents".into()],
      published: "2026-11-18".into(),
      authors: vec!["Alice Smith".into(), "Bob Lee".into()],
      date: "2026-11-19".into(),
      hf_upvotes: 86,
      github_repo: None,
      has_html: true,
      arxiv_version: Some(1),
      lang: "zh".into(),
      model: "deepseek-chat".into(),
      digest: PaperDigest {
        tldr: "t".into(),
        problem: "p".into(),
        method: "m".into(),
        results: vec![],
        limitations: vec![],
        key_points: vec![],
        mindmap: MindNode { label: "r".into(), children: vec![] },
      },
      unverified_numbers: vec![],
      related: vec![],
    };
    let bib = format_bibtex(&detail);
    assert!(bib.starts_with("@article{arxiv_2611_15789,"), "{bib}");
    assert!(bib.contains("title={Fast \\{Attention\\} for Agents}"), "{bib}");
    assert!(bib.contains("author={Alice Smith and Bob Lee}"), "{bib}");
    assert!(bib.contains("year={2026}"), "{bib}");
    assert!(bib.contains("eprint={2611.15789}"), "{bib}");
    assert!(bib.contains("url={https://arxiv.org/abs/2611.15789}"), "{bib}");
  }
}
