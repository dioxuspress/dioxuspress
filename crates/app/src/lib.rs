// S9（风险 R12）：生产代码禁 unwrap/expect（workspace lints）；测试代码豁免。
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

use dioxus::prelude::*;
use dioxus::router::Router;

pub mod components;
pub mod i18n;
pub mod routes;
pub mod server;

use crate::components::theme_picker::use_theme_version_provider;
use crate::i18n::init_i18n;
use crate::routes::Route;
use crate::server::get_aggregated_theme_css;
use app_core::session::SessionUser;
#[cfg(all(feature = "accounts", not(feature = "server")))]
use migration as _;
use module_search::search::{use_search_open_provider, SearchModal};

/// Static assets used by the application.
// Dioxus 0.7 默认在 crate root 的 assets 目录下寻找
const MAIN_CSS: Asset = asset!("/assets/css/main.css");
const TAILWIND_CSS: Asset = asset!("/assets/tailwind.css");
const MATH_CSS: Asset = asset!("/assets/css/math.css");
const PRISM_CSS: Asset = asset!("/assets/css/prism-tomorrow.min.css");
/// Prism core + 语言包合成的一个文件（带哈希、长缓存），由引导脚本在页面出现代码块时才加载
const PRISM_JS: Asset = asset!("/assets/js/prism.bundle.min.js");

/// 全局一次性引导脚本：初始化 mermaid，对**整个文档**开 MutationObserver，
/// 一旦 DOM 中出现未处理的 `<code class="language-*">` 或 `.mermaid` 块就
/// 自动跑 Prism / mermaid。SPA 导航换入新 Markdown 内容时同样能触发。
const GLOBAL_REHIGHLIGHT_BOOT_SCRIPT: &str = r#"
(function(){
  var seen = window.__rie_seen || (window.__rie_seen = new WeakMap());
  function tryHighlight(root){
    var nodes = (root || document).querySelectorAll('code[class*="language-"]');
    if(nodes.length === 0) return;
    if(!window.Prism || typeof window.Prism.highlightElement !== 'function'){ loadPrism(); return; }
    for(var i = 0; i < nodes.length; i++){
      var src = nodes[i].textContent;
      if(seen.get(nodes[i]) === src) continue;
      try { window.Prism.highlightElement(nodes[i]); } catch(e) {}
      seen.set(nodes[i], nodes[i].textContent);
    }
  }
  function tryMermaid(root){
    var all = (root || document).querySelectorAll('pre.mermaid');
    var hits = [];
    for(var i = 0; i < all.length; i++){
      if(all[i].querySelector('svg')) continue;
      var src = all[i].textContent;
      if(seen.get(all[i]) === src) continue;
      seen.set(all[i], src);
      all[i].removeAttribute('data-processed');
      all[i].setAttribute('data-rie-theme', window.__rie_mermaid_theme || '');
      hits.push(all[i]);
    }
    if(hits.length === 0) return;
    if(!window.mermaid || typeof window.mermaid.run !== 'function'){
      for(var j = 0; j < hits.length; j++) seen.delete(hits[j]);
      loadMermaid();
      return;
    }
    try { Promise.resolve(window.mermaid.run({nodes: hits})).then(function(){ keepReadable(hits); }, function(){}); } catch(e) {}
  }
  function keepReadable(nodes){
    for(var i = 0; i < nodes.length; i++){
      var svg = nodes[i].querySelector('svg');
      var vb = svg && svg.viewBox && svg.viewBox.baseVal;
      if(vb && vb.width > 0) svg.style.minWidth = Math.min(vb.width, 560) + 'px';
    }
  }
  function loadPrism(){
    if(window.__rie_prism_loading || !window.__rie_prism_src) return;
    window.__rie_prism_loading = true;
    window.Prism = window.Prism || {manual: true};
    var s = document.createElement('script');
    s.src = window.__rie_prism_src;
    s.onload = sweep;
    document.head.appendChild(s);
  }
  function loadMermaid(){
    if(window.__rie_mermaid_loading) return;
    window.__rie_mermaid_loading = true;
    var s = document.createElement('script');
    s.src = '/js/mermaid.min.js';
    s.onload = sweep;
    document.head.appendChild(s);
  }
  function mermaidTheme(){
    return document.documentElement.classList.contains('dark') ? 'dark' : 'default';
  }
  var MINDMAP_FILLS = {
    'default': ['#BAE6FD','#BBF7D0','#DDD6FE','#FDE68A','#FECACA','#99F6E4','#FBCFE8','#E7E5E4'],
    'dark': ['#075985','#166534','#5B21B6','#92400E','#991B1B','#115E59','#9D174D','#57534E']
  };
  function themeVariables(theme){
    var fills = MINDMAP_FILLS[theme], label = theme === 'dark' ? '#E7E5E4' : '#292524';
    var vars = {git0: '#C2410C', gitBranchLabel0: '#FFFFFF'};
    for(var i = 0; i < 12; i++){
      vars['cScale' + i] = fills[i % fills.length];
      vars['cScaleLabel' + i] = label;
    }
    return vars;
  }
  function bootMermaid(){
    if(!window.mermaid || typeof window.mermaid.initialize !== 'function') return;
    var theme = mermaidTheme();
    if(window.__rie_mermaid_theme === theme) return;
    window.__rie_mermaid_theme = theme;
    try {
      window.mermaid.initialize({startOnLoad: false, theme: theme, themeVariables: themeVariables(theme)});
    } catch(e) {}
  }
  function rethemeMermaid(){
    var theme = mermaidTheme();
    var all = document.querySelectorAll('pre.mermaid');
    for(var i = 0; i < all.length; i++){
      var src = seen.get(all[i]);
      if(src === undefined || all[i].getAttribute('data-rie-theme') === theme) continue;
      all[i].textContent = src;
      seen.delete(all[i]);
    }
    sweep();
  }
  function sweep(){
    bootMermaid();
    tryHighlight(document);
    tryMermaid(document);
  }
  if(document.readyState === 'loading'){
    document.addEventListener('DOMContentLoaded', sweep);
  } else {
    sweep();
  }
  if(typeof MutationObserver === 'function' && !window.__rie_obs){
    window.__rie_obs = new MutationObserver(function(){
      if(window.__rie_obs_q) return;
      window.__rie_obs_q = true;
      setTimeout(function(){
        window.__rie_obs_q = false;
        sweep();
      }, 30);
    });
    window.__rie_obs.observe(document.body || document.documentElement, {childList: true, subtree: true});
  }
  if(typeof MutationObserver === 'function' && !window.__rie_theme_obs){
    window.__rie_theme_obs = new MutationObserver(rethemeMermaid);
    window.__rie_theme_obs.observe(document.documentElement, {attributes: true, attributeFilter: ['class']});
  }
})();
"#;

/// 见 HeadAssets：`?lang=` 优先、其次 `site_lang` cookie、最后 `site.json::default_language`，
/// 英文时把 `<html lang>` 改成 `en`。
fn lang_boot_script() -> String {
  let default_lang = match app_core::settings::SiteConfig::active().default_language_enum() {
    app_core::i18n::Language::En => "en",
    app_core::i18n::Language::Zh => "zh",
  };
  format!(
    "(function(){{try{{var d='{default_lang}';var q=new URLSearchParams(location.search).get('lang');\
if(q==='en'||q==='zh'){{document.cookie='site_lang='+q+'; path=/; max-age=31536000; samesite=lax';}}\
var c=(document.cookie.match(/(?:^|; )site_lang=([^;]*)/)||[])[1];\
var l=(q==='en'||q==='zh')?q:((c==='en'||c==='zh')?c:d);\
document.documentElement.lang=(l==='en')?'en':'zh-CN';}}catch(e){{}}}})();"
  )
}

/// 当前主题的 CSS。首屏经 `use_server_future` 随 SSR 下发；切换主题时重拉替换。
#[component]
fn ThemeStyle() -> Element {
  let version = crate::components::theme_picker::use_theme_version();
  let first =
    use_server_future(|| async move { get_aggregated_theme_css().await.unwrap_or_default() })?;
  let mut switched = use_signal::<Option<String>>(|| None);
  use_effect(move || {
    if version() == 0 {
      return;
    }
    spawn(async move {
      match get_aggregated_theme_css().await {
        Ok(css) => switched.set(Some(css)),
        Err(e) => tracing::warn!(error = ?e, "frontend: failed to fetch theme"),
      }
    });
  });
  let css = switched().unwrap_or_else(|| first().unwrap_or_default());
  rsx! {
      style { id: "wasm-theme-style", dangerous_inner_html: "{css}" }
  }
}

/// 见 HeadAssets：首次绘制前按 `localStorage.theme`（没有则跟随系统）加 `dark` class。
const DARK_MODE_BOOT_SCRIPT: &str = "(function(){try{var t=localStorage.getItem('theme');\
var d=t?t==='dark':matchMedia('(prefers-color-scheme: dark)').matches;\
if(d)document.documentElement.classList.add('dark');}catch(e){}})();";

/// 核心框架默认编译进来的模块 id（不含宿主站点通过 `extra_compiled_modules` 追加的私有模块）。
pub fn core_compiled_modules() -> Vec<&'static str> {
  let mut ids = vec!["blog", "podcast", "cases", "docs", "about"];
  if cfg!(feature = "accounts") {
    ids.push("forum");
  }
  if cfg!(feature = "course") {
    ids.push("course");
  }
  ids
}

/// 注册 DioxusPress 核心编译模块、内置 MDX 嵌入组件与服务端全文检索源。
///
/// 供默认 `app` 二进制与外部宿主站点壳共同复用；宿主站点可通过 `extra_compiled_modules`
/// 追加自己的模块 id（如 `"ai"`、`"papers"`、`"embedded"` 等）。
pub fn register_core_components_and_sources(extra_compiled_modules: &[&'static str]) {
  let mut ids = core_compiled_modules();
  for &id in extra_compiled_modules {
    if !ids.contains(&id) {
      ids.push(id);
    }
  }
  app_core::engines::module::register_compiled_modules(ids);

  // 启动期一次性注册 MDX 嵌入组件（双端调用确保 SSR + hydration registry 一致）。
  widgets::register_default_components();
  module_podcast::register_components();
  components::theme_showcase::register_components();

  // 把 cases / course 注册为搜索索引的外部文档来源（IoC）。
  #[cfg(feature = "server")]
  app_core::engines::doc_source::register_doc_source(|| {
    module_cases::server::scan_cases()
      .into_iter()
      .map(|case| {
        let readme = case.readme_md.unwrap_or_default();
        let body = format!(
          "{} {} {}",
          case.description,
          case.tags.join(" "),
          module_search::text::truncate_chars(&readme, 4000)
        );
        app_core::engines::doc_source::ExternalIndexedDoc {
          kind: "case".to_string(),
          ref_id: case.slug.clone(),
          title: case.name,
          body,
          url: format!("/case/{}", case.slug),
          created_at: case.date_added,
        }
      })
      .collect()
  });

  #[cfg(all(feature = "server", feature = "course"))]
  app_core::engines::doc_source::register_doc_source(|| {
    module_course::server::scan_courses(&module_course::server::get_courses_root())
      .into_iter()
      .map(|c| app_core::engines::doc_source::ExternalIndexedDoc {
        kind: "course".to_string(),
        ref_id: c.slug.clone(),
        title: c.title,
        body: format!("{} {}", c.description, c.tags.join(" ")),
        url: format!("/course/{}", c.slug),
        created_at: String::new(),
      })
      .collect()
  });
}

#[cfg(all(feature = "server", feature = "accounts"))]
pub type ExtraMigrationsFn = fn() -> Vec<Box<dyn migration::MigrationTrait>>;

#[cfg(feature = "server")]
pub type ExtraRouterFn = fn(axum::Router, &str) -> axum::Router;

/// 宿主站点服务端扩展钩子（最小集合，供私有站点壳向框架服务端注入扩展逻辑）。
#[cfg(feature = "server")]
#[derive(Default)]
pub struct ServerExtensions {
  /// 额外的 SeaORM 数据库迁移（与框架内置 `migration::Migrator` 合并运行）。
  #[cfg(feature = "accounts")]
  pub extra_migrations: Option<ExtraMigrationsFn>,
  /// 额外的 SEO 条目（sitemap / feed）。
  pub seo: crate::server::seo::SeoExtensions,
  /// 额外的零停机热刷新缓存失效回调（`SIGHUP` / `CONTENT_DIR` 变更时触发）。
  pub extra_hot_reload: Option<fn()>,
  /// 额外的匿名路由 ETag 版本戳回调。
  pub extra_etag_stamp: Option<fn(&str) -> Option<u128>>,
  /// 额外的自定义 Axum 路由挂载（如 `/papers/feed.xml`），自动受全站中间件保护。
  pub extra_router: Option<ExtraRouterFn>,
}

#[cfg(all(feature = "server", feature = "accounts"))]
static EXTRA_MIGRATIONS: std::sync::OnceLock<ExtraMigrationsFn> = std::sync::OnceLock::new();

#[cfg(all(feature = "server", feature = "accounts"))]
struct CombinedMigrator;

#[cfg(all(feature = "server", feature = "accounts"))]
#[migration::async_trait::async_trait]
impl migration::MigratorTrait for CombinedMigrator {
  fn migrations() -> Vec<Box<dyn migration::MigrationTrait>> {
    let mut list = migration::Migrator::migrations();
    if let Some(extra) = EXTRA_MIGRATIONS.get() {
      list.extend(extra());
    }
    list
  }
}

/// 启动 DioxusPress 全栈服务端（Axum + 全站安全/缓存/限流/热刷新/迁移管线）。
#[cfg(feature = "server")]
pub fn run_server(app: fn() -> Element, ext: ServerExtensions) {
  if let Some(cb) = ext.extra_hot_reload {
    crate::server::hot_reload::register_extra_hot_reload(cb);
  }
  if let Some(cb) = ext.extra_etag_stamp {
    crate::server::conditional_cache::register_extra_etag_stamp(cb);
  }
  #[cfg(feature = "accounts")]
  if let Some(cb) = ext.extra_migrations {
    let _ = EXTRA_MIGRATIONS.set(cb);
  }

  let seo_ext = ext.seo;
  let extra_router = ext.extra_router;

  dioxus::serve(move || async move {
    use axum::routing::get;

    dotenvy::dotenv().ok();

    let _ = tracing_subscriber::fmt()
      .with_env_filter(
        tracing_subscriber::EnvFilter::try_from_default_env()
          .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
      )
      .with_target(true)
      .try_init();

    #[cfg(feature = "accounts")]
    let _ = app_core::session::get_jwt_secret();

    let base_url = resolve_startup_base_url(
      std::env::var("BASE_URL").ok().as_deref(),
      std::env::var("PORT").ok().as_deref(),
    );
    #[cfg(feature = "accounts")]
    let cookie_is_secure = base_url.starts_with("https://");

    #[cfg(feature = "accounts")]
    if let Ok(dek) = std::env::var("DATA_ENCRYPTION_KEY") {
      if !dek.is_empty() {
        app_core::session::assert_not_placeholder("DATA_ENCRYPTION_KEY", &dek);
      }
    }

    #[cfg(feature = "accounts")]
    {
      use crate::server::health::{set_startup_health, MigrationStatus, StartupHealth};
      let strict_migration = std::env::var("STRICT_MIGRATION").map(|v| v == "1").unwrap_or(false);

      if let Ok(db_url) = std::env::var("DATABASE_URL") {
        app_core::session::assert_not_placeholder("DATABASE_URL", &db_url);
        match app_core::db::init_pool(&db_url).await {
          Err(e) => {
            tracing::error!(error = %e, "startup: DB pool init failed (will retry on demand)");
            if strict_migration {
              panic!("STRICT_MIGRATION=1：DB 连接失败，拒绝启动: {}", e);
            }
            set_startup_health(StartupHealth {
              db_configured: true,
              db_connected: false,
              migrations: MigrationStatus::Skipped,
            });
          }
          Ok(()) => {
            tracing::info!("startup: DB pool initialized");
            let mig_status = match app_core::db::get_or_init_pool().await {
              Err(e) => {
                tracing::error!(error = %e, "startup: cannot get pool for migration");
                if strict_migration {
                  panic!("STRICT_MIGRATION=1：迁移前取连接池失败，拒绝启动: {}", e);
                }
                MigrationStatus::Skipped
              }
              Ok(db) => {
                use migration::MigratorTrait;
                match CombinedMigrator::up(&db, None).await {
                  Ok(()) => {
                    tracing::info!("startup: schema migrations applied");
                    MigrationStatus::Applied
                  }
                  Err(e) => {
                    tracing::error!(error = %e, "startup: migration failed");
                    if strict_migration {
                      panic!("STRICT_MIGRATION=1：schema 迁移失败，拒绝启动: {}", e);
                    }
                    MigrationStatus::Failed
                  }
                }
              }
            };
            set_startup_health(StartupHealth {
              db_configured: true,
              db_connected: true,
              migrations: mig_status,
            });
          }
        }
      } else {
        tracing::warn!(
          "startup: DATABASE_URL not set; DB-backed features will error on first call"
        );
        set_startup_health(StartupHealth {
          db_configured: false,
          db_connected: false,
          migrations: MigrationStatus::Skipped,
        });
      }
    }

    #[cfg(not(feature = "accounts"))]
    {
      use crate::server::health::{set_startup_health, MigrationStatus, StartupHealth};
      tracing::info!("startup: built without the accounts feature; no database");
      set_startup_health(StartupHealth {
        db_configured: false,
        db_connected: false,
        migrations: MigrationStatus::Skipped,
      });
    }

    tokio::spawn(async {
      let mut ticker = tokio::time::interval(std::time::Duration::from_secs(600));
      loop {
        ticker.tick().await;
        match module_search::engine::reindex_incremental().await {
          Ok(r) => tracing::info!(
            upserts = r.upserts,
            deletes = r.deletes,
            elapsed_ms = r.elapsed_ms,
            "search: index refreshed"
          ),
          Err(e) => tracing::warn!(error = %e, "search: index refresh failed"),
        }
      }
    });

    crate::server::hot_reload::spawn_hot_reload_tasks();

    #[cfg(feature = "payments")]
    module_course::module_payment::host::log_config_status();
    #[cfg(feature = "payments")]
    if std::env::var("PAY_RECONCILE_DISABLED").ok().as_deref() != Some("1") {
      let interval_secs = std::env::var("PAY_RECONCILE_INTERVAL_SECS")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(300)
        .max(60);
      tokio::spawn(async move {
        let mut ticker = tokio::time::interval(std::time::Duration::from_secs(interval_secs));
        ticker.tick().await;
        loop {
          ticker.tick().await;
          let _ = module_course::server::reconcile_pending_orders_once().await;
        }
      });
      tracing::info!(interval_secs, "startup: payment reconcile task scheduled");
    }

    static ASSETS_ROOT: std::sync::OnceLock<&'static str> = std::sync::OnceLock::new();
    let assets_root: &'static str = ASSETS_ROOT.get_or_init(|| {
      Box::leak(app_core::utils::get_asset_root().to_string_lossy().into_owned().into_boxed_str())
    });
    let content_dir_opt = app_core::utils::content_root();
    app_core::settings::log_startup_diagnostics(
      std::path::Path::new(assets_root),
      content_dir_opt.as_deref(),
    );

    let router = dioxus::server::router(app).route("/healthz", get(crate::server::health::healthz));
    #[cfg(feature = "accounts")]
    let router = crate::server::auth_routes::mount(router, cookie_is_secure);
    let router = crate::server::static_assets::mount(router, assets_root);
    let router = crate::server::pay_routes::mount(router);
    let router = crate::server::seo::mount_with_extensions(router, &base_url, seo_ext);
    let router =
      if let Some(mount_extra) = extra_router { mount_extra(router, &base_url) } else { router };

    let router = router.layer(axum::middleware::from_fn(crate::server::body_limit::body_limit_mw));
    let router = router.layer(axum::middleware::from_fn(crate::server::csrf::csrf_protect_mw));
    let router = router.layer(axum::middleware::from_fn(crate::server::rate_limit::rate_limit_mw));
    let router = router.layer(axum::middleware::from_fn(crate::server::seo::trailing_slash_mw));
    let router = router
      .layer(axum::middleware::from_fn(crate::server::conditional_cache::conditional_cache_mw));
    let router = router.layer(tower_http::compression::CompressionLayer::new());
    let router =
      router.layer(axum::middleware::from_fn(crate::server::security::security_headers_mw));

    Ok(router)
  });
}

/// 启动客户端 WASM 运行时：先从 SSR 注入的 `<meta name="dp-site-config">` 水合站点配置，再挂载 `app`。
#[cfg(not(feature = "server"))]
pub fn run_client(app: fn() -> Element) {
  if let Some(raw) = widgets::browser::meta_content("dp-site-config") {
    if let Some(cfg) = app_core::settings::SiteConfig::from_json(&raw) {
      widgets::update_site_seo_meta(
        cfg.site_name_or_default(),
        cfg.og_image_or_default(),
        cfg.twitter_handle_or_none(),
        cfg.default_language_enum() == app_core::i18n::Language::En,
      );
      app_core::settings::SiteConfig::init_hydrated(cfg);
    }
  }
  dioxus::launch(app);
}

/// DioxusPress 默认开箱二进制入口（供 `crates/app/src/main.rs` 调用）。
pub fn run_default_app() {
  register_core_components_and_sources(&[]);

  #[cfg(feature = "server")]
  run_server(App, ServerExtensions::default());

  #[cfg(not(feature = "server"))]
  run_client(App);
}

#[cfg(feature = "server")]
fn resolve_startup_base_url(raw_base_url: Option<&str>, raw_port: Option<&str>) -> String {
  #[cfg(feature = "accounts")]
  {
    let _ = raw_port;
    // S9 豁免：带登录档位缺失 BASE_URL 有意 fail-fast，防止 OAuth 回调误跳 localhost。
    #[allow(clippy::expect_used)]
    let base_url = raw_base_url
      .map(str::trim)
      .filter(|s| !s.is_empty())
      .expect("BASE_URL 未配置，请在环境变量或 .env 中设置 BASE_URL")
      .to_string();
    app_core::session::assert_not_placeholder("BASE_URL", &base_url);
    base_url
  }
  #[cfg(not(feature = "accounts"))]
  {
    match raw_base_url.map(str::trim).filter(|s| !s.is_empty()) {
      Some(url) => {
        app_core::session::assert_not_placeholder("BASE_URL", url);
        url.to_string()
      }
      None => {
        let port = raw_port.map(str::trim).filter(|s| !s.is_empty()).unwrap_or("8080");
        let fallback = format!("http://localhost:{port}");
        tracing::info!(
          base_url = %fallback,
          "startup: BASE_URL not set in basic mode, defaulting to local address (set BASE_URL in production for accurate sitemap/canonical URLs)"
        );
        fallback
      }
    }
  }
}

/// 全局登录模态框状态
pub fn use_auth_modal() -> Signal<bool> {
  use_context::<Signal<bool>>()
}

/// 全局用户会话状态
pub fn use_session_user() -> Signal<Option<SessionUser>> {
  use_context::<Signal<Option<SessionUser>>>()
}

/// 静态 head 资源（图标 / 全局 CSS / 语法高亮脚本；mermaid 按需加载）。
#[component]
fn HeadAssets(#[props(default)] tailwind_css: Option<Asset>) -> Element {
  let prism_src = PRISM_JS.to_string();
  let lang_script = lang_boot_script();
  let site_cfg = app_core::settings::SiteConfig::active();
  let site_cfg_json = site_cfg.to_json();
  let favicon_href = site_cfg.favicon_or_default();
  let favicon_32_href = site_cfg.favicon_32_url();
  let favicon_16_href = site_cfg.favicon_16_url();
  let apple_touch_href = site_cfg.apple_touch_icon_url();
  let manifest_href = site_cfg.manifest_url();
  let tw_css = tailwind_css.unwrap_or(TAILWIND_CSS);
  rsx! {
      document::Meta { name: "dp-site-config", content: "{site_cfg_json}" }
      document::Script { {lang_script} }
      document::Script { {DARK_MODE_BOOT_SCRIPT} }
      document::Link { rel: "icon", href: "{favicon_href}" }
      if let Some(href) = favicon_32_href {
          document::Link { rel: "icon", r#type: "image/png", sizes: "32x32", href: "{href}" }
      }
      if let Some(href) = favicon_16_href {
          document::Link { rel: "icon", r#type: "image/png", sizes: "16x16", href: "{href}" }
      }
      if let Some(href) = apple_touch_href {
          document::Link { rel: "apple-touch-icon", sizes: "180x180", href: "{href}" }
      }
      if let Some(href) = manifest_href {
          document::Link { rel: "manifest", href: "{href}" }
      }
      document::Link { rel: "alternate", r#type: "application/atom+xml", title: widgets::site_name(), href: "/feed.xml" }
      document::Link { rel: "stylesheet", href: MAIN_CSS }
      document::Link { rel: "stylesheet", href: tw_css }

      document::Style { "
        body {{ 
          background-color: var(--color-bg, white); 
          color: var(--color-text, #0f172a);
          transition: background-color 0.3s ease, color 0.3s ease;
        }}
        .dark body {{ 
          background-color: var(--color-bg, #020617); 
          color: var(--color-text, #f8fafc);
        }}
        .prose-comment .prose {{ font-size: 0.875rem; }}
        .prose-comment .prose h1 {{ font-size: 1.1em; margin: 0.4em 0; line-height: 1.3; }}
        .prose-comment .prose h2 {{ font-size: 1em; margin: 0.3em 0; line-height: 1.3; }}
        .prose-comment .prose h3 {{ font-size: 0.95em; margin: 0.2em 0; line-height: 1.3; }}
        .prose-comment .prose p {{ margin: 0.3em 0; line-height: 1.5; }}
        .prose-comment .prose img {{ max-height: 200px; border-radius: 0.5rem; margin: 0.5em 0; }}
      " }

      document::Style { "
        math {{ font-size: 1.1em; }}
        .math-display math {{ font-size: 1.4em; }}
        .prose code::before, .prose code::after {{ content: none !important; }}
        .prose :where(h2, h3)[id] {{ scroll-margin-top: 5rem; }}
      " }

      document::Link { rel: "stylesheet", href: MATH_CSS }
      document::Link { rel: "stylesheet", href: PRISM_CSS }
      document::Script { "window.__rie_prism_src='{prism_src}';" }
      document::Script { {GLOBAL_REHIGHLIGHT_BOOT_SCRIPT} }
      document::Script { src: "/js/annotations.js", defer: true }
  }
}

/// 登录弹窗（accounts 层）；basic 层没有登录入口，渲染为空。
fn auth_modal(show: Signal<bool>) -> Element {
  #[cfg(feature = "accounts")]
  {
    rsx! { crate::components::auth_modal::AuthModal { show } }
  }
  #[cfg(not(feature = "accounts"))]
  {
    let _ = show;
    rsx! {}
  }
}

/// 通用根外壳组件：提供全局 SEO / i18n / 主题 / 搜索 / 登录态上下文，供宿主站点包裹自己的 `Router::<SiteRoute>`。
#[component]
pub fn AppShell(#[props(default)] tailwind_css: Option<Asset>, children: Element) -> Element {
  let site_cfg = app_core::settings::SiteConfig::active();
  widgets::update_site_seo_meta(
    site_cfg.site_name_or_default(),
    site_cfg.og_image_or_default(),
    site_cfg.twitter_handle_or_none(),
    site_cfg.default_language_enum() == app_core::i18n::Language::En,
  );
  init_i18n();

  let show_auth = use_signal(|| false);
  use_context_provider(|| show_auth);
  let _ = use_search_open_provider();
  let _ = use_theme_version_provider();

  let user: Signal<Option<SessionUser>> = use_signal(|| None);
  use_context_provider(|| user);

  #[cfg(feature = "accounts")]
  {
    let mut user_signal = user;
    use_effect(move || {
      spawn(async move {
        if let Ok(Some(u)) = crate::server::get_current_user().await {
          user_signal.set(Some(u));
        }
      });
    });
  }

  rsx! {
      HeadAssets { tailwind_css }
      SuspenseBoundary { fallback: |_| VNode::empty(), ThemeStyle {} }
      {children}
      {auth_modal(show_auth)}
      SearchModal {}
  }
}

#[component]
fn App() -> Element {
  rsx! {
      AppShell {
          Router::<Route> {}
      }
  }
}

#[cfg(feature = "server")]
fn format_stars(stars: i64) -> String {
  if stars >= 1000 {
    format!("★ {:.1}k", stars as f64 / 1000.0)
  } else if stars > 0 {
    format!("★ {stars}")
  } else {
    String::new()
  }
}

/// 通用的领域聚合 Hub 资源构建器，供任意 `board::Board` 列表复用（文档卷 / 课程 / 案例 / 博客）。
#[cfg(feature = "server")]
pub fn build_board_hub_for(
  boards: &[board::Board],
  board_id: &str,
  lang: app_core::i18n::Language,
) -> board::BoardHubResources {
  const MAX_HUB_ITEMS: usize = 6;
  let Some(b) = boards.iter().find(|b| b.id == board_id) else {
    return board::BoardHubResources::default();
  };
  let engine = app_core::engines::module::default_module_engine();
  let mut resources = board::BoardHubResources::default();

  if engine.is_enabled("docs") {
    let mut matched_docs: Vec<(usize, board::BoardHubItem)> =
      module_docs::server::cached_volumes_for_lang(lang)
        .into_iter()
        .filter_map(|vol| {
          let rank = b.doc_slug_rank(&vol.node.slug)?;
          let meta = if vol.pages > 0 {
            format!("{} {}", vol.pages, lang.pick("页", "pages"))
          } else {
            String::new()
          };
          Some((
            rank,
            board::BoardHubItem {
              title: vol.node.title,
              description: vol.node.description,
              href: format!("/docs/{}", vol.node.path),
              meta,
              tags: Vec::new(),
            },
          ))
        })
        .collect();
    matched_docs.sort_by_key(|(rank, _)| *rank);
    resources.docs = matched_docs.into_iter().take(MAX_HUB_ITEMS).map(|(_, item)| item).collect();
  }

  #[cfg(feature = "course")]
  if engine.is_enabled("course") {
    let mut courses =
      module_course::server::scan_courses(&module_course::server::get_courses_root());
    courses.sort_by(|a, b| a.order.cmp(&b.order).then_with(|| a.slug.cmp(&b.slug)));
    resources.courses = courses
      .into_iter()
      .filter(|c| b.matches_resource(&c.tags, None, &c.slug))
      .take(MAX_HUB_ITEMS)
      .map(|c| {
        let lessons: usize = c.chapters.iter().map(|ch| ch.lessons.len()).sum();
        let meta = if lessons > 0 {
          format!("{lessons} {}", lang.pick("课节", "lessons"))
        } else {
          String::new()
        };
        board::BoardHubItem {
          title: c.title,
          description: c.description,
          href: format!("/course/{}", c.slug),
          meta,
          tags: c.tags,
        }
      })
      .collect();
  }

  if engine.is_enabled("cases") {
    let mut cases = module_cases::server::scan_cases();
    cases.sort_by(|a, b| {
      b.favorite
        .cmp(&a.favorite)
        .then_with(|| b.stars.cmp(&a.stars))
        .then_with(|| a.name.cmp(&b.name))
    });
    resources.cases = cases
      .into_iter()
      .filter(|c| b.matches_resource(&c.tags, Some(&c.category), &c.slug))
      .take(MAX_HUB_ITEMS)
      .map(|c| board::BoardHubItem {
        title: c.name,
        description: c.description,
        href: format!("/case/{}", c.slug),
        meta: format_stars(c.stars),
        tags: c.tags,
      })
      .collect();
  }

  if engine.is_enabled("blog") {
    let posts = module_blog::server::cached_blog_posts();
    resources.posts = posts
      .iter()
      .filter(|p| b.matches_resource(&p.tags, None, &p.slug))
      .take(MAX_HUB_ITEMS)
      .map(|p| board::BoardHubItem {
        title: p.title.clone(),
        description: p.description.clone(),
        href: format!("/blog/{}", p.slug),
        meta: p.date.clone(),
        tags: p.tags.clone(),
      })
      .collect();
  }

  resources
}

#[cfg(all(test, feature = "server"))]
mod startup_tests {
  use super::{build_board_hub_for, core_compiled_modules, resolve_startup_base_url};

  #[test]
  fn core_compiled_modules_matches_tier_features() {
    let ids = core_compiled_modules();
    for id in ["blog", "podcast", "cases", "docs", "about"] {
      assert!(ids.contains(&id), "missing core module {id}");
    }
    assert_eq!(ids.contains(&"forum"), cfg!(feature = "accounts"));
    assert_eq!(ids.contains(&"course"), cfg!(feature = "course"));
    assert!(build_board_hub_for(&[], "nonexistent", app_core::i18n::Language::Zh).is_empty());
  }

  #[test]
  fn explicit_base_url_is_preserved_and_trimmed() {
    let url = resolve_startup_base_url(Some(" https://docs.example.com "), Some("9000"));
    assert_eq!(url, "https://docs.example.com");
  }

  #[cfg(not(feature = "accounts"))]
  #[test]
  fn basic_mode_defaults_base_url_when_unset() {
    assert_eq!(resolve_startup_base_url(None, None), "http://localhost:8080");
    assert_eq!(resolve_startup_base_url(Some("   "), Some("3000")), "http://localhost:3000");
  }

  #[cfg(feature = "accounts")]
  #[test]
  #[should_panic(expected = "BASE_URL 未配置")]
  fn accounts_mode_requires_explicit_base_url() {
    let _ = resolve_startup_base_url(None, None);
  }
}
