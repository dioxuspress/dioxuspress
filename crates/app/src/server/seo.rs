//! S7（风险 R8）：SEO 路由（sitemap.xml / feed.xml / robots.txt）。
//!
//! 从 `main.rs` 拆出。此前 sitemap 与 feed 各自复制了一份
//! `collect_board!` 宏与条目收集逻辑——新增内容板块时漏改一处就会造成
//! 「sitemap 有、feed 无」的不一致。现统一为 [`collect_content_entries`]，
//! 两个端点共用同一收集函数。

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use axum::routing::get;
use axum::Router;
use widgets::ContentEntry;

pub type ExtraStaticPathsFn = fn(&[String]) -> Vec<String>;
pub type ExtraDetailPathsFn =
  fn(Vec<String>) -> Pin<Box<dyn Future<Output = Vec<(String, Option<String>)>> + Send>>;
pub type ExtraContentEntriesFn =
  fn(Vec<String>) -> Pin<Box<dyn Future<Output = Vec<ContentEntry>> + Send>>;

/// 宿主站点的 SEO 扩展回调（供私有站点壳向 `/sitemap.xml` 与 `/feed.xml` 追加专属条目）。
#[derive(Clone, Copy, Default)]
pub struct SeoExtensions {
  pub extra_static_paths: Option<ExtraStaticPathsFn>,
  pub extra_detail_paths: Option<ExtraDetailPathsFn>,
  pub extra_content_entries: Option<ExtraContentEntriesFn>,
}

/// 收集全部启用模块的内容条目（博客正文；宿主站点自定义板块通过 `SeoExtensions::extra_content_entries` 注入）。
async fn collect_content_entries(is_on: impl Fn(&str) -> bool) -> Vec<ContentEntry> {
  let mut entries: Vec<ContentEntry> = Vec::new();

  if is_on("blog") {
    let posts = module_blog::server::list_blog_posts().await.unwrap_or_default();
    entries.extend(posts.into_iter().map(|p| ContentEntry {
      url_path: format!("/blog/{}", p.slug),
      title: p.title,
      description: p.description,
      date: if p.updated.is_empty() { p.date } else { p.updated },
      tags: p.tags,
    }));
  }

  entries
}

/// 详情页（文档 / 案例 / 课程详情与公开课节）：只进 sitemap，不进 feed。
/// 第二项是 lastmod（文档取文件 mtime），没有就空。需要购买的课节不收录。
async fn collect_detail_paths(is_on: impl Fn(&str) -> bool) -> Vec<(String, Option<String>)> {
  let mut paths = Vec::new();
  if is_on("docs") {
    let tree = module_docs::server::list_doc_tree().await.unwrap_or_default();
    paths.extend(doc_paths(&tree).into_iter().map(|p| {
      let lastmod = module_docs::server::doc_lastmod(&p);
      (format!("/docs/{p}"), lastmod)
    }));
    let bi_paths = bilingual_doc_paths(&tree);
    if !bi_paths.is_empty() {
      paths.push(("/docs?lang=en".to_string(), None));
      paths.extend(bi_paths.into_iter().map(|p| {
        let lastmod = module_docs::server::doc_lastmod_for_lang(&p, app_core::i18n::Language::En);
        (widgets::lang_path(&format!("/docs/{p}"), true), lastmod)
      }));
    }
  }
  if is_on("cases") {
    let cases = module_cases::server::list_cases(None, None, None).await.unwrap_or_default();
    paths.extend(cases.into_iter().map(|c| (format!("/case/{}", c.slug), None)));
  }
  #[cfg(feature = "course")]
  if is_on("course") {
    let courses = module_course::server::list_courses().await.unwrap_or_default();
    for c in courses {
      paths.push((format!("/course/{}", c.slug), None));
      // G6：访客能读的课节（免费课程全部、付费课程的试看课节）
      if let Some(course) = module_course::server::read_course(&c.slug) {
        paths.extend(course.open_lesson_paths().into_iter().map(|p| (p, None)));
      }
    }
  }
  paths
}

/// 文档树里有正文的节点 → 相对 docs 根的路径（深度优先，父在子前）。
fn doc_paths(nodes: &[module_docs::server::DocTreeNode]) -> Vec<String> {
  let mut out = Vec::new();
  for n in nodes {
    if n.has_content && !n.path.is_empty() {
      out.push(n.path.clone());
    }
    out.extend(doc_paths(&n.children));
  }
  out
}

/// 同时拥有中英文正文的文档节点路径（用于生成 `/docs/<path>?lang=en` sitemap 条目）。
fn bilingual_doc_paths(nodes: &[module_docs::server::DocTreeNode]) -> Vec<String> {
  let mut out = Vec::new();
  for n in nodes {
    if n.has_content && n.has_bilingual && !n.path.is_empty() {
      out.push(n.path.clone());
    }
    out.extend(bilingual_doc_paths(&n.children));
  }
  out
}

fn xml_response(content_type: &str, cache_control: &str, body: String) -> axum::response::Response {
  axum::response::Response::builder()
    .header("content-type", content_type)
    .header("cache-control", cache_control)
    .body(axum::body::Body::from(body))
    .unwrap_or_else(|_| axum::response::Response::new(axum::body::Body::empty()))
}

async fn sitemap_handler(base: String, ext: Arc<SeoExtensions>) -> axum::response::Response {
  use widgets::build_sitemap_xml;

  // Phase 3.4：按模块开关过滤静态路径与内容条目。
  let module_engine = app_core::engines::module::default_module_engine();
  let enabled = module_engine.enabled_ids();
  let is_on = |id: &str| enabled.iter().any(|s| s == id);

  let mut entries = collect_content_entries(is_on).await;
  if let Some(extra_fn) = ext.extra_content_entries {
    entries.extend(extra_fn(enabled.clone()).await);
  }

  // 静态路径：首页恒收录；其它模块从 ModuleSpec.static_path 自动获取。
  // Phase 8.7：加新模块只动 ModuleSpec.static_path 即可在 sitemap 出现。
  let mut static_paths: Vec<String> = vec!["/".to_string()];
  for spec in module_engine.enabled_modules() {
    if let Some(p) = spec.static_path.as_deref() {
      static_paths.push(p.to_string());
    }
  }
  if is_on("about") && crate::server::about_page_is_bilingual() {
    static_paths.push("/about?lang=en".to_string());
  }
  // 站点自定义顶栏站内导航（如 dioxuspress.dev 的 /templates）自动纳入 sitemap。
  let site_cfg = app_core::settings::SiteConfig::active();
  for nav_item in &site_cfg.navigation {
    let r = nav_item.route.trim();
    if r.starts_with('/')
      && r != "/feed.xml"
      && r != "/sitemap.xml"
      && !static_paths.iter().any(|p| p == r)
    {
      static_paths.push(r.to_string());
    }
  }
  if let Some(extra_static) = ext.extra_static_paths {
    for p in extra_static(&enabled) {
      if !static_paths.iter().any(|s| s == &p) {
        static_paths.push(p);
      }
    }
  }
  let mut detail_paths = collect_detail_paths(is_on).await;
  if let Some(extra_details) = ext.extra_detail_paths {
    detail_paths.extend(extra_details(enabled.clone()).await);
  }
  for (path, lastmod) in detail_paths {
    match lastmod {
      Some(date) => entries.push(ContentEntry {
        url_path: path,
        title: String::new(),
        description: String::new(),
        date,
        tags: Vec::new(),
      }),
      None => static_paths.push(path),
    }
  }
  let static_paths_ref: Vec<&str> = static_paths.iter().map(String::as_str).collect();

  let xml = build_sitemap_xml(&entries, &static_paths_ref, &base);
  // Phase 8.5：sitemap 1 小时 Cache-Control，让爬虫与 CDN 不至于
  // 每次请求都打满 list_* hot path。`public` 表示中间代理也能缓存。
  xml_response("application/xml; charset=utf-8", "public, max-age=3600", xml)
}

async fn feed_handler(base: String, ext: Arc<SeoExtensions>) -> axum::response::Response {
  use widgets::build_atom_feed_with_author;

  // Phase 3.4：blog 关闭时输出空 feed，但保留站点元信息。
  let module_engine = app_core::engines::module::default_module_engine();
  let enabled = module_engine.enabled_ids();
  let is_on = |id: &str| enabled.iter().any(|s| s == id);

  let mut entries = collect_content_entries(is_on).await;
  if let Some(extra_fn) = ext.extra_content_entries {
    entries.extend(extra_fn(enabled.clone()).await);
  }

  // 全站按日期降序，取最近 50 篇。
  entries.sort_by(|a, b| b.date.cmp(&a.date));
  entries.truncate(50);

  // 取站点元信息：优先读挂载的 CONTENT_DIR/site.json，否则回退内置 assets/site.json。S10：mtime 缓存读取。
  let cfg = app_core::settings::SiteConfig::load_cached(
    app_core::utils::site_json_path().to_str().unwrap_or_default(),
  )
  .unwrap_or_default();
  let xml = build_atom_feed_with_author(
    &entries,
    cfg.site_name_or_default(),
    &cfg.site_description,
    &cfg.author,
    &base,
  );
  // Phase 8.5：同 sitemap，feed 1 小时 Cache-Control 缓解 RSS reader 轮询压力。
  xml_response("application/atom+xml; charset=utf-8", "public, max-age=3600", xml)
}

async fn robots_handler(base: String) -> axum::response::Response {
  let body = widgets::build_robots_txt(&base);
  // robots.txt 几乎不变，6h 缓存。
  xml_response("text/plain; charset=utf-8", "public, max-age=21600", body)
}

/// 挂载 SEO 路由。`base_url` 为站点对外 URL（无尾斜杠）。
pub fn mount(router: Router, base_url: &str) -> Router {
  mount_with_extensions(router, base_url, SeoExtensions::default())
}

/// 挂载 SEO 路由并附带宿主站点扩展回调。
pub fn mount_with_extensions(router: Router, base_url: &str, ext: SeoExtensions) -> Router {
  let ext = Arc::new(ext);
  let sitemap_base = base_url.to_string();
  let sitemap_ext = ext.clone();
  let feed_base = base_url.to_string();
  let feed_ext = ext;
  let robots_base = base_url.to_string();
  router
    .route(
      "/sitemap.xml",
      get(move || {
        let base = sitemap_base.clone();
        let ext = sitemap_ext.clone();
        async move { sitemap_handler(base, ext).await }
      }),
    )
    .route(
      "/feed.xml",
      get(move || {
        let base = feed_base.clone();
        let ext = feed_ext.clone();
        async move { feed_handler(base, ext).await }
      }),
    )
    .route(
      "/robots.txt",
      get(move || {
        let base = robots_base.clone();
        async move { robots_handler(base).await }
      }),
    )
}

/// 带尾斜杠的页面地址对应的无斜杠地址（G7）：`/blog/` 与 `/blog` 返回同一页，
/// 爬虫会当成两个 URL。根路径和 `/api/*` 不处理；去掉斜杠后以 `//` 开头的
/// 不跳（浏览器会把 `//host` 当成别的站点，即开放重定向）。
fn slash_redirect_target(path: &str, query: Option<&str>) -> Option<String> {
  if path == "/" || !path.ends_with('/') || path.starts_with("/api/") {
    return None;
  }
  let trimmed = path.trim_end_matches('/');
  if trimmed.is_empty() || trimmed.starts_with("//") {
    return None;
  }
  Some(match query {
    Some(q) => format!("{trimmed}?{q}"),
    None => trimmed.to_string(),
  })
}

pub type RedirectResolverFn = fn(&str) -> Option<String>;

fn extra_redirect_resolvers_slot() -> &'static std::sync::RwLock<Vec<RedirectResolverFn>> {
  static RESOLVERS: std::sync::OnceLock<std::sync::RwLock<Vec<RedirectResolverFn>>> =
    std::sync::OnceLock::new();
  RESOLVERS.get_or_init(|| std::sync::RwLock::new(Vec::new()))
}

/// 注册宿主站点自定义永久跳转规则（如从旧静态博客迁移的 301 重定向，参见 [`docusaurus_legacy_blog_redirect`]）。
pub fn register_extra_redirect_resolver(resolver: RedirectResolverFn) {
  if let Ok(mut guard) = extra_redirect_resolvers_slot().write() {
    guard.push(resolver);
  }
}

/// 可选内置工具：Docusaurus 旧博客地址到 DioxusPress 的 301 映射规则。
/// 公开框架不再无条件硬编码执行，需从 Docusaurus 迁移的宿主站点可在 `ServerExtensions::extra_redirect` 中按需挂载。
pub fn docusaurus_legacy_blog_redirect(path: &str) -> Option<String> {
  let trimmed = path.trim_end_matches('/');
  let english = trimmed.strip_prefix("/en").filter(|p| p.starts_with("/blog"));
  let rest = match english.unwrap_or(trimmed).strip_prefix("/blog") {
    // `/blog` 本身不跳（尾斜杠交给 slash_redirect_target）
    Some("") => return english.map(|_| "/blog".to_string()),
    Some(rest) => rest.strip_prefix('/')?,
    None => return None,
  };
  let segments: Vec<&str> = rest.split('/').collect();
  match segments.as_slice() {
    [y, m, d, slug]
      if [(y, 4), (m, 2), (d, 2)]
        .iter()
        .all(|(s, n)| s.len() == *n && s.bytes().all(|b| b.is_ascii_digit())) =>
    {
      let slug = slug.replace('.', "-");
      let safe = !slug.is_empty()
        && slug.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-');
      safe.then(|| format!("/blog/{slug}"))
    }
    ["tags", ..] | ["page", ..] | ["archive"] | ["authors", ..] => Some("/blog".to_string()),
    _ => None,
  }
}

fn extra_redirect_target(path: &str) -> Option<String> {
  extra_redirect_resolvers_slot().read().ok()?.iter().find_map(|f| f(path))
}

fn matches_seg(path: &str, prefix: &str) -> bool {
  path == prefix
    || (path.len() > prefix.len()
      && path.starts_with(prefix)
      && path.as_bytes().get(prefix.len()) == Some(&b'/'))
}

/// 私密后台、个人中心、发帖表单与登录/支付跳转端点不应进入搜索引擎索引：
/// 通过 `X-Robots-Tag: noindex, nofollow` 明确告知爬虫不收录，避免依赖 `robots.txt`
/// 屏蔽站内已链接的交互页而在 Search Console 触发「已被 robots.txt 屏蔽」告警。
fn should_noindex_path(path: &str) -> bool {
  path == "/topics/new"
    || ["/admin", "/me", "/api/auth", "/pay", "/orders"].iter().any(|p| matches_seg(path, p))
}

/// GET / HEAD 的旧博客地址与尾斜杠地址永久跳转到新地址；对私密/交互表单路径下发 `X-Robots-Tag: noindex, nofollow`。
pub async fn trailing_slash_mw(
  req: axum::extract::Request,
  next: axum::middleware::Next,
) -> axum::response::Response {
  use axum::http::{HeaderName, HeaderValue, Method};
  use axum::response::{IntoResponse, Redirect};
  let path = req.uri().path();
  if matches!(*req.method(), Method::GET | Method::HEAD) {
    let target =
      extra_redirect_target(path).or_else(|| slash_redirect_target(path, req.uri().query()));
    if let Some(to) = target {
      return Redirect::permanent(&to).into_response();
    }
  }
  let noindex = should_noindex_path(path);
  let mut resp = next.run(req).await;
  if noindex {
    resp.headers_mut().insert(
      HeaderName::from_static("x-robots-tag"),
      HeaderValue::from_static("noindex, nofollow"),
    );
  }
  resp
}

#[cfg(test)]
mod tests {
  use super::*;
  use module_docs::server::DocTreeNode;

  fn node(path: &str, has_content: bool, children: Vec<DocTreeNode>) -> DocTreeNode {
    DocTreeNode {
      slug: path.rsplit('/').next().unwrap_or_default().to_string(),
      title: path.to_string(),
      path: path.to_string(),
      has_content,
      has_bilingual: false,
      description: String::new(),
      children,
    }
  }

  /// G7：尾斜杠 301 到无斜杠地址；根路径、API 与会变成 `//host` 的路径不跳。
  #[test]
  fn trailing_slash_redirect_targets() {
    assert_eq!(slash_redirect_target("/blog/", None).as_deref(), Some("/blog"));
    assert_eq!(
      slash_redirect_target("/docs/axum//", Some("a=1")).as_deref(),
      Some("/docs/axum?a=1")
    );
    assert_eq!(slash_redirect_target("/", None), None);
    assert_eq!(slash_redirect_target("/blog", None), None);
    assert_eq!(slash_redirect_target("/api/orders/", None), None);
    assert_eq!(slash_redirect_target("//evil.com/", None), None, "不能变成开放重定向");
    assert_eq!(slash_redirect_target("///", None), None);
  }

  /// M5：旧博客地址跳到新地址；新站自己的地址不跳。
  #[test]
  fn legacy_blog_redirect_targets() {
    let t = |p: &str| docusaurus_legacy_blog_redirect(p);
    assert_eq!(t("/blog/2024/01/09/rust-ownership").as_deref(), Some("/blog/rust-ownership"));
    assert_eq!(t("/blog/2024/10/17/rust-1.82/").as_deref(), Some("/blog/rust-1-82"));
    assert_eq!(t("/en/blog/2024/01/09/rust-ownership").as_deref(), Some("/blog/rust-ownership"));
    for old in [
      "/blog/tags/rust",
      "/blog/tags",
      "/blog/page/3",
      "/blog/archive",
      "/blog/authors",
      "/en/blog",
      "/en/blog/",
    ] {
      assert_eq!(t(old).as_deref(), Some("/blog"), "{old}");
    }
    for keep in [
      "/blog",
      "/blog/rust-ownership",
      "/blog/2024/01/09",
      "/blog/24/01/09/x",
      "/blog/2024/01/09/x%2Fy",
      "/blog/2024/01/09/中文",
      "/en/docs",
      "/docs/blog/tags",
      "/blogs/tags",
    ] {
      assert_eq!(t(keep), None, "{keep}");
    }
  }

  #[test]
  fn noindex_paths_match_only_private_and_form_routes() {
    for private_path in [
      "/topics/new",
      "/admin",
      "/admin/users",
      "/admin/moderation/settings",
      "/me",
      "/me/topics",
      "/me/orders",
      "/me/annotations",
      "/api/auth/login/github",
      "/api/auth/callback/google",
      "/pay/checkout",
      "/orders/123",
    ] {
      assert!(should_noindex_path(private_path), "expected noindex for {private_path}");
    }

    for public_path in [
      "/",
      "/topics",
      "/topics/42",
      "/topics/tag/rust",
      "/docs",
      "/docs/admin-guide",
      "/blog/me-and-rust",
      "/course/model-finetuning",
      "/models/deepseek-v3",
    ] {
      assert!(!should_noindex_path(public_path), "unexpected noindex for {public_path}");
    }
  }

  #[tokio::test]
  async fn trailing_slash_mw_sets_x_robots_tag_on_noindex_routes_only() {
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    let app = Router::new()
      .route("/topics", get(|| async { "topics" }))
      .route("/topics/new", get(|| async { "new topic" }))
      .route("/admin/users", get(|| async { "admin users" }))
      .layer(axum::middleware::from_fn(trailing_slash_mw));

    let resp_new = app
      .clone()
      .oneshot(
        Request::builder()
          .uri("/topics/new?ref_kind=doc&ref_path=tokio")
          .body(Body::empty())
          .unwrap(),
      )
      .await
      .unwrap();
    assert_eq!(resp_new.status(), StatusCode::OK);
    assert_eq!(
      resp_new.headers().get("x-robots-tag").and_then(|v| v.to_str().ok()),
      Some("noindex, nofollow")
    );

    let resp_admin = app
      .clone()
      .oneshot(Request::builder().uri("/admin/users").body(Body::empty()).unwrap())
      .await
      .unwrap();
    assert_eq!(
      resp_admin.headers().get("x-robots-tag").and_then(|v| v.to_str().ok()),
      Some("noindex, nofollow")
    );

    let resp_public =
      app.oneshot(Request::builder().uri("/topics").body(Body::empty()).unwrap()).await.unwrap();
    assert!(resp_public.headers().get("x-robots-tag").is_none());
  }

  #[test]
  fn doc_paths_keeps_only_nodes_with_content() {
    let tree = vec![
      node(
        "basics",
        true,
        vec![node("basics/ownership", true, vec![]), node("basics/wip", false, vec![])],
      ),
      node("empty-dir", false, vec![node("empty-dir/leaf", true, vec![])]),
    ];
    assert_eq!(doc_paths(&tree), vec!["basics", "basics/ownership", "empty-dir/leaf"]);
  }

  /// 验收（O4）：sitemap 的详情页条目 = 各来源计数之和，用真实 assets 独立计数比对。
  #[tokio::test]
  async fn detail_paths_cover_every_doc_case_and_course() {
    let root = app_core::utils::get_asset_root();
    let count_dirs = |sub: &str| {
      std::fs::read_dir(root.join(sub))
        .map(|rd| {
          rd.flatten()
            .filter(|e| {
              e.path().is_dir() && !e.file_name().to_string_lossy().starts_with(['_', '.'])
            })
            .count()
        })
        .unwrap_or(0)
    };
    let tree = module_docs::server::list_doc_tree().await.unwrap_or_default();
    let docs = doc_paths(&tree).len();
    let en_docs = bilingual_doc_paths(&tree).len();
    let en_index_entry = usize::from(en_docs > 0);
    let index_files = walk_count(&app_core::utils::content_dir("docs"));
    assert!(docs > 0);
    assert_eq!(docs, index_files, "文档树漏掉了磁盘上的文档（深度上限 / _meta.json 过滤？）");

    let paths = collect_detail_paths(|_| true).await;
    let courses = if cfg!(feature = "course") { count_dirs("courses") } else { 0 };
    let open_lessons = paths
      .iter()
      .filter(|(p, _)| p.starts_with("/course/") && p.matches('/').count() == 4)
      .count();
    // 文档详情路径（含 ?lang=en 伴生页）都带 lastmod（文件 mtime）
    assert!(paths.iter().filter(|(p, _)| p.starts_with("/docs/")).all(|(_, m)| m.is_some()));
    assert_eq!(
      paths.len(),
      docs + en_docs + en_index_entry + count_dirs("cases") + courses + open_lessons
    );
    // G6：示例付费课程只有第一课是试看。
    if cfg!(feature = "course") {
      let has = |p: &str| paths.iter().any(|(x, _)| x == p);
      assert!(has("/course/rust-basics/01-fundamentals/01-what-is-rust"));
      assert!(!has("/course/rust-basics/02-ownership/01-borrow-rules"), "付费课节不收录");
    }
    assert!(collect_detail_paths(|_| false).await.is_empty());
  }

  /// 非根目录下 index.md(x) 的数量（根 index 是 /docs 本身，已在静态路径里）。
  fn walk_count(dir: &std::path::Path) -> usize {
    let Ok(rd) = std::fs::read_dir(dir) else { return 0 };
    rd.flatten()
      .map(|e| {
        let p = e.path();
        if p.is_dir() {
          let own = ["index.md", "index.mdx"].iter().any(|f| p.join(f).is_file()) as usize;
          own + walk_count(&p)
        } else {
          0
        }
      })
      .sum()
  }
}
