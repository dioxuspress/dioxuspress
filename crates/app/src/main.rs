// S9（风险 R12）：生产代码禁 unwrap/expect（workspace lints）；测试代码豁免。
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

use dioxus::prelude::*;
use dioxus::router::Router;

mod components;
mod i18n;
mod routes;
mod server;
mod taxonomy;

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
///
/// 设计要点：
/// - **只挂载一次**（在 App 根的 document::Script 里），避免 mdx 组件级别
///   的 Script 引起 dioxus-document props-change 警告且无法重跑。
/// - prism 和 mermaid 都不在 head 里：首次见到代码块 / `.mermaid` 块时才注入
///   `<script>`，加载完再 sweep（G3，mermaid 3MB；prism 40KB，列表页用不到）。
///   prism 的地址带哈希，由 App 在脚本前写进 `window.__rie_prism_src`。
/// - 代码块 / mermaid 块的内容是 `dangerous_inner_html`（见 widgets mdx.rs），
///   Dioxus 不跟踪其子节点，这里改写才不会弄坏 hydration。
/// - 是否处理过按源码比较（WeakMap），SPA 导航复用节点换内容时会重新处理。
const GLOBAL_REHIGHLIGHT_BOOT_SCRIPT: &str = r#"
(function(){
  // 节点 → 上次按哪份源码处理过。块的内容由 Dioxus 以 innerHTML 设置，SPA 导航可能
  // 复用同一节点换内容，所以按源码比较而不是打一次性标记。
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
      // 渲染是异步的，进行中的节点还没有 svg，别重复提交
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
  // mermaid 的 svg 是 width:100% 加 max-width:自身宽度，手机上大图会被压到字看不清。
  // 给一个最小宽度（自身宽度和 560px 取小），容器 overflow-x-auto，窄屏上横向滑动；
  // 正文列比 560px 宽，桌面不受影响，小图也不会被放大。
  function keepReadable(nodes){
    for(var i = 0; i < nodes.length; i++){
      var svg = nodes[i].querySelector('svg');
      var vb = svg && svg.viewBox && svg.viewBox.baseVal;
      if(vb && vb.width > 0) svg.style.minWidth = Math.min(vb.width, 560) + 'px';
    }
  }
  // 按需加载 prism：只请求一次，加载完再扫一遍。manual 让它不自己扫全页，统一由 sweep 处理。
  function loadPrism(){
    if(window.__rie_prism_loading || !window.__rie_prism_src) return;
    window.__rie_prism_loading = true;
    window.Prism = window.Prism || {manual: true};
    var s = document.createElement('script');
    s.src = window.__rie_prism_src;
    s.onload = sweep;
    document.head.appendChild(s);
  }
  // 按需加载 mermaid：只请求一次，加载完再扫一遍。
  function loadMermaid(){
    if(window.__rie_mermaid_loading) return;
    window.__rie_mermaid_loading = true;
    var s = document.createElement('script');
    s.src = '/js/mermaid.min.js';
    s.onload = sweep;
    document.head.appendChild(s);
  }
  // mermaid 的主题跟 <html class="dark"> 走。主题是 initialize 时定的、画进 svg 里的，
  // 换主题要重新 initialize，再把已画好的图还原成源码重画。
  function mermaidTheme(){
    return document.documentElement.classList.contains('dark') ? 'dark' : 'default';
  }
  // 思维导图按分支取 cScale0..11 上色（默认是彩虹色，暗色主题里还有一支接近底色），
  // 换成站点的低饱和色：浅色用 Tailwind 200 档配深字，暗色用 800 档配浅字；根节点用 Rust 橙。
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
  // 按每块提交时记下的主题判断，不看全局主题：DOM 观察器的 sweep 可能先一步
  // 用新主题 initialize 过了。
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
  // 初始扫一遍（处理首屏 / hydration 后已存在的内容）
  if(document.readyState === 'loading'){
    document.addEventListener('DOMContentLoaded', sweep);
  } else {
    sweep();
  }
  // 观察后续 DOM 插入（SPA 路由切换换入新 markdown 时触发）
  if(typeof MutationObserver === 'function' && !window.__rie_obs){
    window.__rie_obs = new MutationObserver(function(){
      // 节流：合并同一 tick 的多次插入
      if(window.__rie_obs_q) return;
      window.__rie_obs_q = true;
      setTimeout(function(){
        window.__rie_obs_q = false;
        sweep();
      }, 30);
    });
    window.__rie_obs.observe(document.body || document.documentElement, {childList: true, subtree: true});
  }
  // 切换暗色模式只改 <html> 的 class
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

/// 当前主题的 CSS。以前用 `use_resource` 在 hydration 之后才拉，SSR 的 HTML 里没有
/// 主题变量，每次打开页面都先按默认色绘制、再跳成主题色（背景灰 → 白、主色变色）。
/// 现在首屏经 `use_server_future` 随 SSR 下发；用户切换主题（`ThemeVersion` 变化）时
/// 直接调 server fn 拿新 CSS 替换，不重新挂起。
///
/// 注意：**不能**用 `document::Style`（变更 props 会触发警告且不更新 DOM）；用普通
/// `style` + `dangerous_inner_html`：CSS 落在 <body> 仍全局生效，且 vdom diff 正常更新内容。
#[component]
fn ThemeStyle() -> Element {
  let version = crate::components::theme_picker::use_theme_version();
  let first =
    use_server_future(|| async move { get_aggregated_theme_css().await.unwrap_or_default() })?;
  let mut switched = use_signal::<Option<String>>(|| None);
  use_effect(move || {
    // 首次挂载时 version 为 0，用 SSR 的那份；之后每次切换重取
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

/// 本次编译带进来的模块 id（docs/FEATURE_TIERS.md DL2）。导航 / ModuleGate / sitemap /
/// 搜索只认这个集合；site.json 的模块开关只能在其中再关。
fn compiled_modules() -> Vec<&'static str> {
  let mut ids =
    vec!["blog", "podcast", "cases", "docs", "embedded", "ai", "web3", "wasm", "cli", "about"];
  if cfg!(feature = "accounts") {
    ids.push("forum");
  }
  if cfg!(feature = "course") {
    ids.push("course");
  }
  if cfg!(feature = "papers") {
    ids.push("papers");
  }
  ids
}

fn main() {
  app_core::engines::module::register_compiled_modules(compiled_modules());
  // Phase 2.1 / 2.2: 启动期一次性注册 MDX 嵌入组件。
  // 在 server 和 client 两边都调用，用于 SSR + hydration 双边 registry 一致。
  // 1) widgets 内置 9 个默认组件 (YouTube / Bilibili / 5 色 / Underline / Strikethrough)
  widgets::register_default_components();
  // 2) 各业务模块与宿主应用注册自身提供的 MDX 组件 (PodcastCard / ThemeShowcase ...)
  module_podcast::register_components();
  components::theme_showcase::register_components();

  // 3) A3 解耦：把 cases 注册为搜索索引的外部文档来源（IoC）。
  //    使 module-search 不再编译期依赖 module-cases —— 数据由组合根 app 在此注入。
  //    仅 server 端需要（scan_cases 读磁盘 + 索引只在服务端构建）。
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

  // 论文解读在数据库里，异步取。
  #[cfg(all(feature = "server", feature = "papers"))]
  app_core::engines::doc_source::register_async_doc_source(module_papers::server::search_docs);

  // O10：课程同样经注册表进索引。只收标题 / 简介 / 标签——课时正文是付费内容，不进公开搜索。
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

  // Server: customize the Axum router to serve blog post static assets
  #[cfg(feature = "server")]
  dioxus::serve(|| async move {
    use axum::routing::get;

    // 加载 .env 环境变量
    dotenvy::dotenv().ok();

    // Phase 7.5：初始化 tracing 订阅。日志级别由 `RUST_LOG` 控制
    // （默认 `info`），格式为人可读 + 含 target。
    //
    // 用 `try_init`（而非 `init`）：在 `dx serve` 等开发运行时下，
    // dioxus-devtools 可能已安装全局 subscriber，`init` 会 panic
    // （"a global default trace dispatcher has already been set"）。
    // 已存在时静默跳过，生产 `dx bundle` 产物里没有冲突 subscriber，正常生效。
    let _ = tracing_subscriber::fmt()
      .with_env_filter(
        tracing_subscriber::EnvFilter::try_from_default_env()
          .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
      )
      .with_target(true)
      .try_init();

    // 安全门禁：启动时必须提供关键环境变量，避免 fallback 到不安全默认值
    // 1) JWT_SECRET 必须配置且非 placeholder（panic on missing / 命中模板片段）。
    //    只有带登录的构建（accounts 层）需要；basic 层不签发也不认 JWT。
    #[cfg(feature = "accounts")]
    let _ = app_core::session::get_jwt_secret();
    // 2) BASE_URL 解析：
    //    - accounts 及以上档位：涉及 OAuth 回调与 Secure 会话 Cookie，缺失即 fail-fast；
    //    - basic 零数据库档位：支持 `docker run -p 8080:8080 -v ./my-site:/app/content:ro`
    //      零环境变量开箱即跑，未设时自动回退到 `http://localhost:{PORT}` 并输出提示日志。
    let base_url = resolve_startup_base_url(
      std::env::var("BASE_URL").ok().as_deref(),
      std::env::var("PORT").ok().as_deref(),
    );
    #[cfg(feature = "accounts")]
    let cookie_is_secure = base_url.starts_with("https://");
    // S5（风险 R2）：独立数据加密密钥。可选；配了就不能是占位值。
    // 未配置时 crypto 模块会回退到 JWT_SECRET 派生并 warn。
    #[cfg(feature = "accounts")]
    if let Ok(dek) = std::env::var("DATA_ENCRYPTION_KEY") {
      if !dek.is_empty() {
        app_core::session::assert_not_placeholder("DATA_ENCRYPTION_KEY", &dek);
      }
    }

    // 3) 数据库只在 accounts 层使用（登录 / 论坛 / 评论 / 课程）。
    #[cfg(feature = "accounts")]
    {
      // 3) 提前初始化数据库连接池，后续 server fn 都走共享连接。
      //    DATABASE_URL 也按 placeholder 校验，避免「忘了改 docker-compose 密码」
      //    导致 sqlx 拒绝认证的难以排查的失败。
      //    连接失败仅在日志提示，不阻塞启动，以保证静态页面仍可访问。
      //    成功连上数据库后再跑 sea-orm-migration（Phase 7.1）。
      // S3（风险 R3）：启动结果记入健康快照（/healthz 暴露）；
      // `STRICT_MIGRATION=1` 时迁移/连接失败 fail-fast，避免带着不一致
      // schema 静默接受写请求（生产推荐开启）。
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
            // 自动迁移：用同一连接池跑 sea-orm-migration。
            // 默认失败仅日志 + /healthz 报 degraded；STRICT_MIGRATION=1 时退出。
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
                match migration::Migrator::up(&db, None).await {
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
    // basic 层：不连库（/healthz 报 db=disabled，属正常）。
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
    // 搜索索引在卷里跨部署保留，且论坛发帖不写索引：启动时增量刷新一次（新镜像带来的
    // 文章 / 文档 / 案例），之后每 10 分钟一次（新论坛主题）。按 mtime 差分，开销小（Y6）。
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

    // E5.3：启动 SIGHUP 零停机热刷新与 CONTENT_DIR 节流变更探测（避免纯内容发布重启容器）。
    crate::server::hot_reload::spawn_hot_reload_tasks();

    // M5e：支付对账定时任务（payments feature）。周期扫描滞留 pending 单，
    // 向网关查单回填发货（回调丢失兜底）/ 超窗关单；网关未配置时每轮
    // 空跑（零成本）。PAY_RECONCILE_DISABLED=1 关闭。
    #[cfg(feature = "payments")]
    module_course::module_payment::host::log_config_status();
    #[cfg(feature = "payments")]
    if std::env::var("PAY_RECONCILE_DISABLED").ok().as_deref() != Some("1") {
      let interval_secs = std::env::var("PAY_RECONCILE_INTERVAL_SECS")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(300)
        .max(60); // 下限 60s，防误配频繁打网关
      tokio::spawn(async move {
        let mut ticker = tokio::time::interval(std::time::Duration::from_secs(interval_secs));
        // interval 的首个 tick 立即就绪：先吃掉，避免启动即打网关。
        ticker.tick().await;
        loop {
          ticker.tick().await;
          // 回填/关单明细已在 target=pay_audit 日志留痕。
          let _ = module_course::server::reconcile_pending_orders_once().await;
        }
      });
      tracing::info!(interval_secs, "startup: payment reconcile task scheduled");
    }

    // Phase 8.8：用 OnceLock 取代 Box::leak，避免 `dx serve` 反复重启的开发态泄漏
    // 累积。`OnceLock<&'static str>` 在首次写入时把 String leak 成 &'static，
    // 后续运行直接读 cell（无重复 leak、无可观察的语义差异）。
    static ASSETS_ROOT: std::sync::OnceLock<&'static str> = std::sync::OnceLock::new();
    let assets_root: &'static str = ASSETS_ROOT.get_or_init(|| {
      Box::leak(app_core::utils::get_asset_root().to_string_lossy().into_owned().into_boxed_str())
    });
    let content_dir_opt = app_core::utils::content_root();
    app_core::settings::log_startup_diagnostics(
      std::path::Path::new(assets_root),
      content_dir_opt.as_deref(),
    );

    // S7（风险 R8）：router 组装拆到 server 子模块，main.rs 只留引导。
    // auth / pay / 静态资源 / SEO 各自一个 mount 函数，行为与拆分前一致。
    let router = dioxus::server::router(App)
      // S3：健康检查（ok/degraded），供 LB / 监控 / 部署脚本判活。
      .route("/healthz", get(crate::server::health::healthz));
    #[cfg(feature = "accounts")]
    let router = crate::server::auth_routes::mount(router, cookie_is_secure);
    let router = crate::server::static_assets::mount(router, assets_root);
    let router = crate::server::pay_routes::mount(router);
    // Phase 2.4：公开 SEO 路由（sitemap / feed / robots），条目收集逻辑
    // 已统一到 server::seo::collect_content_entries（消除双份宏重复）。
    let router = crate::server::seo::mount(router, &base_url);

    // 请求体超过上限时，dioxus 读 server fn 请求体会 panic（500）：/api/upload 放宽到 8 MiB，
    // 声明的 Content-Length 超限的请求直接回 413。
    let router = router.layer(axum::middleware::from_fn(crate::server::body_limit::body_limit_mw));

    // S2（风险 R4）：应用层 per-IP 限流，仅作用于 /api/*（auth / pay 更严）。
    // gateway 限流仍是第一道防线；这里是 app 裸跑时的兑底。
    let router = router.layer(axum::middleware::from_fn(crate::server::rate_limit::rate_limit_mw));

    // G7：`/blog/` 这类尾斜杠地址 301 到 `/blog`，避免重复 URL。
    let router = router.layer(axum::middleware::from_fn(crate::server::seo::trailing_slash_mw));

    // E5.2：匿名只读页面弱 ETag / If-None-Match (304) 短路与私密路由 private, no-store 防护。
    // 挂在 CompressionLayer 内侧，使 304 命中直接跳过 SSR 与流式压缩，同时 200 仍正常经外层 br/gzip 压缩。
    let router = router
      .layer(axum::middleware::from_fn(crate::server::conditional_cache::conditional_cache_mw));

    // G4：按 Accept-Encoding 压缩响应（br / gzip）。HTML、CSS、JS、wasm、
    // sitemap / feed 都受益；图片和流式 SSE 由默认判定跳过，已带
    // Content-Encoding 的响应不重复压缩。
    let router = router.layer(tower_http::compression::CompressionLayer::new());

    // S1（风险 R6）：全站安全响应头（CSP / nosniff / Referrer-Policy / X-Frame-Options）。
    // 挂在最外层，覆盖页面、server fn、静态资源与上面全部自定义路由（含 429 响应）。
    let router =
      router.layer(axum::middleware::from_fn(crate::server::security::security_headers_mw));

    Ok(router)
  });

  // Client: 从 SSR 注入的 `<meta name="dp-site-config">` 水合运行期站点配置（使零编译镜像挂载
  // `/app/content/site.json` 时双端一致），随后启动 Dioxus 客户端运行时。
  #[cfg(not(feature = "server"))]
  {
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
    dioxus::launch(App);
  }
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
///
/// 这些节点内容固定、与任何响应式状态无关。单独封成无 props 的组件后会被
/// Dioxus memoize：仅首次渲染挂载一次，App 因主题 / 语言切换 bump 信号重渲时
/// **不会**重渲它们，从而消除 dioxus-document 的
/// "Changing the props of `Style {}` / `Script {}` is not supported" 警告。
#[component]
fn HeadAssets() -> Element {
  // asset 路径只含 [A-Za-z0-9./_-]，可直接放进单引号字符串
  let prism_src = PRISM_JS.to_string();
  let lang_script = lang_boot_script();
  let site_cfg = app_core::settings::SiteConfig::active();
  let site_cfg_json = site_cfg.to_json();
  let favicon_href = site_cfg.favicon_or_default();
  let favicon_32_href = site_cfg.favicon_32_url();
  let favicon_16_href = site_cfg.favicon_16_url();
  let apple_touch_href = site_cfg.apple_touch_icon_url();
  let manifest_href = site_cfg.manifest_url();
  rsx! {
      document::Meta { name: "dp-site-config", content: "{site_cfg_json}" }
      // <html lang> 跟首屏语言走（index.html 写死 zh-CN）；规则同 i18n::pick_language
      document::Script { {lang_script} }
      // 暗色偏好在首次绘制前就加到 <html> 上（同步内联脚本，规则同
      // widgets::browser::dark_mode_preference）。以前只在 wasm 加载后的 use_effect 里设，
      // 暗色用户每次打开页面先看到一屏白。
      document::Script { {DARK_MODE_BOOT_SCRIPT} }
      // Head links：统一走 `/images/*`（支持 `$CONTENT_DIR/images` 私有品牌覆盖与 `Last-Modified` 复验）
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
      // feed 自动发现（O4）：/feed.xml 是 Atom。
      document::Link { rel: "alternate", r#type: "application/atom+xml", title: widgets::site_name(), href: "/feed.xml" }
      document::Link { rel: "stylesheet", href: MAIN_CSS }
      document::Link { rel: "stylesheet", href: TAILWIND_CSS }

      // Global Fixed Styles (Static)
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

      // mdx 内容的全局静态样式（math / 评论 prose 缩排）。
      document::Style { "
        math {{ font-size: 1.1em; }}
        .math-display math {{ font-size: 1.4em; }}
        .prose code::before, .prose code::after {{ content: none !important; }}
        .prose :where(h2, h3)[id] {{ scroll-margin-top: 5rem; }}
      " }

      // pulldown-latex math fonts & styles
      document::Link { rel: "stylesheet", href: MATH_CSS }

      // PrismJS 样式常驻（很小）；脚本由下面的引导脚本按需加载。
      document::Link { rel: "stylesheet", href: PRISM_CSS }

      // 全局 rehighlight / mermaid 触发器。**只挂载一次**，靠 MutationObserver
      // 自动捕获后续插入到 DOM 的代码块 / mermaid 块（含 SPA 路由切换换入的内容）。
      // prism / mermaid 都由它在页面首次出现对应块时再加载。
      document::Script { "window.__rie_prism_src='{prism_src}';" }
      document::Script { {GLOBAL_REHIGHLIGHT_BOOT_SCRIPT} }

      // 运行时标注运行时（PR-D）；晚于 wasm 加载时由 `__rieAnnoPending` 补上
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

#[component]
fn App() -> Element {
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

  // 搜索 modal 全局状态(Cmd+K 快捷键 + 导航栏按钮共享)
  let _ = use_search_open_provider();

  // Phase 3.1：ThemeVersion context。Navbar 的 ThemePicker 切换后 bump 该 Signal，
  // ThemeStyle 据此重取聚合 CSS。
  let _ = use_theme_version_provider();

  // 全局用户会话
  let user: Signal<Option<SessionUser>> = use_signal(|| None);
  use_context_provider(|| user);

  // 加载当前用户。
  //
  // 重构 B6 评估：有意保留客户端加载（use_effect + server fn），**不** 迁移到
  // use_server_future。理由：`get_current_user` 是每请求的登录态（读 session
  // cookie）；若随 SSR 预渲染会把个性化用户信息烘进页面 HTML，与 CDN /
  // 代理层的公共缓存（sitemap/feed 已用 public Cache-Control）相冲，且有信息泄露风险。
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
      // 静态 head 资源（图标 / CSS / 脚本）。抽到独立 memoized 组件，避免随 App
      // 重渲（主题 / 语言切换会 bump 信号触发 App 重渲）反复重挂 document::Style /
      // Script，进而触发 dioxus-document "Changing the props … is not supported" 警告。
      HeadAssets {}

      // 当前主题的 CSS 随 SSR 输出（自带一层只包住它的 SuspenseBoundary，挂起不波及整站）。
      SuspenseBoundary { fallback: |_| VNode::empty(), ThemeStyle {} }

      // Main router entry
      Router::<Route> {}

      // Auth modal (rendered at root level to avoid stacking context issues)
      {auth_modal(show_auth)}

      // 全局搜索模态框(在根挂一次,任意页面都可 Cmd+K 拉起)
      SearchModal {}
  }
}

#[cfg(all(test, feature = "server"))]
mod startup_tests {
  use super::resolve_startup_base_url;

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
