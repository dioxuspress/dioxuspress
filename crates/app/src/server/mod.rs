#[cfg(all(feature = "server", feature = "accounts"))]
pub mod auth_routes;
#[cfg(feature = "server")]
pub mod body_limit;
#[cfg(feature = "server")]
pub mod conditional_cache;
#[cfg(feature = "server")]
pub mod health;
#[cfg(feature = "server")]
pub mod hot_reload;
#[cfg(feature = "server")]
pub mod pay_routes;
#[cfg(feature = "server")]
pub mod rate_limit;
#[cfg(feature = "server")]
pub mod security;
#[cfg(feature = "server")]
pub mod seo;
#[cfg(feature = "server")]
pub mod static_assets;

#[cfg(feature = "accounts")]
use app_core::session::SessionUser;
#[cfg(feature = "server")]
use app_core::settings::SiteConfig;
#[cfg(feature = "server")]
use app_core::utils::{content_dir, get_asset_root, site_json_path};
use dioxus::fullstack::{post, ServerFnError};
use dioxus::prelude::*;
use serde::{Deserialize, Serialize};

// ========== 辅助：从 FullstackContext 读取 Cookie 中的用户 ==========

/// server-only: 从当前请求上下文的 Cookie 中解析 SessionUser
#[cfg(all(feature = "server", feature = "accounts"))]
fn current_session_user() -> Option<SessionUser> {
  app_core::session::current_session_user()
}

/// server-only: 读当前请求 Cookie 头中指定 key 的值（Phase 3.1 主题覆盖用）。
/// 使用 `get_all("cookie")` 兼容 HTTP/2 多条 `cookie` 头（RFC 9113 §8.2.3）。
#[cfg(feature = "server")]
fn read_request_cookie(name: &str) -> Option<String> {
  use dioxus::fullstack::FullstackContext;
  let ctx = FullstackContext::current()?;
  let parts = ctx.parts_mut();
  let found = parts
    .headers
    .get_all("cookie")
    .iter()
    .filter_map(|v| v.to_str().ok())
    .flat_map(|raw| raw.split(';'))
    .find_map(|pair| {
      let (k, v) = pair.trim().split_once('=')?;
      (k == name && !v.is_empty()).then(|| v.to_string())
    });
  drop(parts);
  found
}

/// Cookie 名（Phase 3.1）：存储用户选择的主题 id。
#[cfg(feature = "server")]
pub const THEME_COOKIE_NAME: &str = "site_theme";

// ========== 主题 ==========

/// 主题切换菜单及预览卡片的一项。前端 `ThemePicker` 与 `ThemeShowcase` 复用。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ThemeInfo {
  /// 内置主题 id，也是 cookie 存储的值。
  pub id: String,
  pub label: String,
  /// 风格简述（中文）。
  pub description: String,
  /// 风格简述（英文）。
  pub description_en: String,
  /// 亮色底色色值。
  pub light_bg: String,
  /// 亮色主色色值。
  pub light_primary: String,
  /// 暗色底色色值。
  pub dark_bg: String,
  /// 暗色主色色值。
  pub dark_primary: String,
  /// 是否是本次请求生效的主题。
  pub is_active: bool,
}

#[post("/api/theme/aggregated-css")]
pub async fn get_aggregated_theme_css() -> Result<String, ServerFnError> {
  #[cfg(feature = "server")]
  {
    use app_core::engines::theme::resolve_theme_for_site;

    // S10：热路径（每次页面渲染拉主题 CSS）——mtime 缓存读取。
    let config =
      SiteConfig::load_cached(site_json_path().to_str().unwrap_or_default()).unwrap_or_default();
    let cookie = read_request_cookie(THEME_COOKIE_NAME);
    Ok(
      resolve_theme_for_site(&config.theme, &config.available_themes, cookie.as_deref())
        .css
        .to_string(),
    )
  }
  #[cfg(not(feature = "server"))]
  {
    Ok("".to_string())
  }
}

/// 主题切换菜单的选项：按 `site.json::available_themes` 过滤（留空返回全部内置主题），标出当前生效项。
#[post("/api/theme/list")]
pub async fn list_available_themes() -> Result<Vec<ThemeInfo>, ServerFnError> {
  #[cfg(feature = "server")]
  {
    use app_core::engines::theme::{resolve_available_themes, resolve_theme_for_site};

    let config =
      SiteConfig::load_cached(site_json_path().to_str().unwrap_or_default()).unwrap_or_default();
    let cookie = read_request_cookie(THEME_COOKIE_NAME);
    let active =
      resolve_theme_for_site(&config.theme, &config.available_themes, cookie.as_deref()).id;
    let allowed = resolve_available_themes(&config.theme, &config.available_themes);
    Ok(
      allowed
        .into_iter()
        .map(|t| ThemeInfo {
          id: t.id.to_string(),
          label: t.label.to_string(),
          description: t.description.to_string(),
          description_en: t.description_en.to_string(),
          light_bg: t.light_bg.to_string(),
          light_primary: t.light_primary.to_string(),
          dark_bg: t.dark_bg.to_string(),
          dark_primary: t.dark_primary.to_string(),
          is_active: t.id == active,
        })
        .collect(),
    )
  }
  #[cfg(not(feature = "server"))]
  {
    Ok(vec![])
  }
}

/// 设置访客主题 cookie（覆盖 site.json 的默认主题）。
///
/// 传入空字符串表示“重置”（删除 cookie）；其他值必须是当前站点开放的内置主题 id（SEC-11）。
/// 写 `Set-Cookie: site_theme=...; Path=/; Max-Age=31536000; SameSite=Lax`。
/// 生产环境（`BASE_URL` 以 https 开头）额外附加 `Secure`。
#[post("/api/theme/set")]
pub async fn set_user_theme(id: String) -> Result<(), ServerFnError> {
  #[cfg(feature = "server")]
  {
    use app_core::engines::theme::resolve_available_themes;
    use dioxus::fullstack::FullstackContext;

    let config =
      SiteConfig::load_cached(site_json_path().to_str().unwrap_or_default()).unwrap_or_default();
    let allowed = resolve_available_themes(&config.theme, &config.available_themes);

    let trimmed = id.trim();
    let cookie_value = match trimmed {
      "" => "",
      _ => allowed.into_iter().find(|t| t.id == trimmed).map(|t| t.id).ok_or_else(|| {
        ServerFnError::new(format!("主题不存在或未开放: {}", trimmed.escape_debug()))
      })?,
    };

    let secure_flag = std::env::var("BASE_URL")
      .ok()
      .filter(|u| u.starts_with("https://"))
      .map(|_| "; Secure")
      .unwrap_or("");

    // 主题名是非敏感展示偏好，不使用 HttpOnly；前端也会用 document.cookie 兜底写入，
    // 确保随后的主题 CSS 重新请求能立即携带新 cookie，刷新后也能保持选择。
    let header_value = if cookie_value.is_empty() {
      format!("{}=; Path=/; Max-Age=0; SameSite=Lax{}", THEME_COOKIE_NAME, secure_flag)
    } else {
      format!(
        "{}={}; Path=/; Max-Age=31536000; SameSite=Lax{}",
        THEME_COOKIE_NAME, cookie_value, secure_flag
      )
    };

    let parsed: axum::http::HeaderValue =
      header_value.parse().map_err(|e| ServerFnError::new(format!("不合法 Cookie 头: {}", e)))?;
    if let Some(ctx) = FullstackContext::current() {
      ctx.add_response_header(axum::http::header::SET_COOKIE, parsed);
    }
    Ok(())
  }
  #[cfg(not(feature = "server"))]
  {
    let _ = id;
    Ok(())
  }
}

// ========== Auth 辅助 (server-only) ==========

#[cfg(feature = "accounts")]
#[cfg(feature = "server")]
fn build_auth_service() -> (app_core::auth::AuthService, SiteConfig) {
  use app_core::auth::{AuthConfig, AuthService};

  // BASE_URL 未配置时 panic，避免生产环境误用 localhost
  // S9 豁免：与启动门禁同策略，缺失即 panic 是有意的 fail-fast。
  #[allow(clippy::expect_used)]
  let base_url =
    std::env::var("BASE_URL").expect("BASE_URL 未配置，请在环境变量或 .env 中设置 BASE_URL");
  let config = AuthConfig { base_url };
  let site_path = site_json_path();
  // S10：mtime 缓存读取；返回需要 owned，clone 一次（配置体积小）。
  let site_config = site_path
    .to_str()
    .and_then(|p| SiteConfig::load_cached(p).ok())
    .map(|cfg| (*cfg).clone())
    .unwrap_or_default();
  let auth_service = AuthService::new(config);
  (auth_service, site_config)
}

/// URL 里的 provider 段 → site.json 启用的内置 provider。
#[cfg(feature = "accounts")]
#[cfg(feature = "server")]
fn resolve_provider(
  site_config: &SiteConfig,
  provider: &str,
) -> Result<app_core::auth::Provider, app_core::error::AppError> {
  app_core::auth::enabled_providers(site_config)
    .find(|p| p.id() == provider)
    .ok_or_else(|| format!("未启用的登录方式: {}", provider).into())
}

// ========== Auth 端点 ==========

#[cfg(feature = "accounts")]
#[post("/api/auth/providers")]
pub async fn get_auth_providers() -> Result<Vec<app_core::AuthProviderDisplay>, ServerFnError> {
  #[cfg(feature = "server")]
  {
    let (auth_service, site_config) = build_auth_service();
    Ok(auth_service.list_available_providers(&site_config))
  }
  #[cfg(not(feature = "server"))]
  {
    Ok(vec![])
  }
}

/// Phase 7.2：准备一次 OAuth 登录，返回 (跳转 URL, 加密后的 cookie value)。
/// 仅供 axum 路由 `/api/auth/login/{provider}` 使用：路由把 cookie value
/// 套上 [`app_core::auth::build_pkce_set_cookie`] 后下发，再 302 到 url。
#[cfg(feature = "accounts")]
#[cfg(feature = "server")]
pub async fn prepare_login_for_provider(
  provider: String,
) -> Result<(String, String), app_core::error::AppError> {
  let (auth_service, site_config) = build_auth_service();
  let provider = resolve_provider(&site_config, &provider)?;
  let (url, payload) = auth_service.prepare_login(provider)?;
  let cookie_value = payload.encode()?;
  Ok((url, cookie_value))
}

/// 内部 auth callback — 仅 server 端调用，返回 (welcome_message, jwt_token)。
/// Phase 7.2：state / verifier 来自浏览器加密 cookie（不再有进程内 HashMap）。
///
/// Phase 7.5 debug 习惯：`#[tracing::instrument]` 自动记录入口 + 退出 +
/// elapsed；`code` skip 以免 OAuth code 进日志，但 code_len 入字段供
/// 联调判定；`provider` / 解密后 cookie 内的 `pkce_cookie.provider`
/// 都已脱敏。
#[cfg(feature = "accounts")]
#[cfg(feature = "server")]
#[tracing::instrument(
  name = "server::auth_callback",
  level = "info",
  skip(code, pkce_cookie),
  fields(
    provider = %provider,
    code_len = code.len(),
    state_match = received_state == pkce_cookie.state,
  ),
  err
)]
pub async fn auth_callback_internal(
  code: String,
  provider: String,
  received_state: String,
  pkce_cookie: app_core::auth::PkceCookiePayload,
) -> Result<(String, String), app_core::error::AppError> {
  use app_core::db::get_or_init_pool;
  use app_core::session::create_jwt;

  let (auth_service, site_config) = build_auth_service();
  let provider = resolve_provider(&site_config, &provider)?;

  tracing::debug!(
    provider = %provider.id(),
    code_len = code.len(),
    state_len = received_state.len(),
    "auth callback received"
  );

  let db = get_or_init_pool().await?;

  let user =
    auth_service.handle_callback(&db, provider, code, &received_state, pkce_cookie).await?;

  let jwt_token = create_jwt(&user)?;
  tracing::info!(user = %user.nickname, "auth callback login success");
  Ok((format!("欢迎回来, {}!", user.nickname), jwt_token))
}

/// 获取当前登录用户 — 前端调用
#[cfg(feature = "accounts")]
#[post("/api/auth/me")]
pub async fn get_current_user() -> Result<Option<SessionUser>, ServerFnError> {
  #[cfg(feature = "server")]
  {
    Ok(current_session_user())
  }
  #[cfg(not(feature = "server"))]
  {
    Ok(None)
  }
}

// ========== 模块开关（Phase 3.4） ==========

/// Phase 3.4：返回当前启用的模块 id 列表。
///
/// `site.json::modules.<id>.enabled = false` 的模块不会出现在结果中。
/// 默认（未在 site.json 显式声明）模块按 [`default_module_specs`] 全开。
///
/// 用例：
/// - 前端 Navbar 决定是否渲染 blog / podcast / cases / forum 等链接
/// - 各页面（`ModuleGate`）根据列表判定是否显示“该模块已停用”占位
#[post("/api/modules/enabled")]
pub async fn enabled_module_ids() -> Result<Vec<String>, ServerFnError> {
  #[cfg(feature = "server")]
  {
    let engine = app_core::engines::module::default_module_engine();
    Ok(engine.enabled_ids())
  }
  #[cfg(not(feature = "server"))]
  {
    // 客户端 fallback：编译进来的模块全开，避免 hydration 闪烁。
    Ok(app_core::engines::module::default_module_specs().into_iter().map(|s| s.id).collect())
  }
}

/// Phase 3.4：单模块开关查询。便于页面级 ModuleGate 调用。
#[post("/api/modules/is-enabled")]
pub async fn is_module_enabled(id: String) -> Result<bool, ServerFnError> {
  #[cfg(feature = "server")]
  {
    let engine = app_core::engines::module::default_module_engine();
    Ok(engine.is_enabled(&id))
  }
  #[cfg(not(feature = "server"))]
  {
    let _ = id;
    Ok(true)
  }
}

// ========== About ==========

/// About 页服务端返回载荷（含双语伴生标记，用于输出 `hreflang` 与按语言区分的 `canonical`）。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AboutPageResponse {
  pub content: String,
  #[serde(default)]
  pub has_bilingual: bool,
}

#[cfg(feature = "server")]
const ABOUT_ZH_CANDIDATES: &[&str] = &["about.mdx", "about.md"];
#[cfg(feature = "server")]
const ABOUT_EN_CANDIDATES: &[&str] = &["about.en.mdx", "about.en.md"];

#[cfg(feature = "server")]
fn find_first_file(dir: &std::path::Path, names: &[&str]) -> Option<std::path::PathBuf> {
  names.iter().map(|n| dir.join(n)).find(|p| p.is_file())
}

/// 选取生效的 `pages/` 目录：若挂载目录（`$CONTENT_DIR/pages`）已自定义任何 `about*.{md,mdx}`，
/// 则严格限定在该挂载目录内回退（防止挂载单语 `about.md` 的站点在英文模式下误回退到内置 DioxusPress 英文介绍）。
#[cfg(feature = "server")]
pub(crate) fn select_about_pages_dir(
  content_pages: &std::path::Path,
  asset_pages: &std::path::Path,
) -> std::path::PathBuf {
  if find_first_file(content_pages, ABOUT_ZH_CANDIDATES).is_some()
    || find_first_file(content_pages, ABOUT_EN_CANDIDATES).is_some()
  {
    content_pages.to_path_buf()
  } else {
    asset_pages.to_path_buf()
  }
}

#[cfg(feature = "server")]
pub(crate) fn active_about_pages_dir() -> std::path::PathBuf {
  select_about_pages_dir(&content_dir("pages"), &get_asset_root().join("pages"))
}

/// 在给定 `pages/` 目录中按目标语言解析 `about` 源文件及是否具备双语伴生文件。
#[cfg(feature = "server")]
pub(crate) fn resolve_about_file_in_dir(
  pages_dir: &std::path::Path,
  lang: app_core::i18n::Language,
) -> Option<(std::path::PathBuf, bool)> {
  let zh = find_first_file(pages_dir, ABOUT_ZH_CANDIDATES);
  let en = find_first_file(pages_dir, ABOUT_EN_CANDIDATES);
  let has_bilingual = zh.is_some() && en.is_some();
  let chosen = match lang {
    app_core::i18n::Language::En => en.or(zh),
    app_core::i18n::Language::Zh => zh.or(en),
  }?;
  Some((chosen, has_bilingual))
}

/// 当前站点的 `/about` 是否同时拥有中英双语源文件（供 `sitemap.xml` 收录 `/about?lang=en` 使用）。
#[cfg(feature = "server")]
pub fn about_page_is_bilingual() -> bool {
  let dir = active_about_pages_dir();
  find_first_file(&dir, ABOUT_ZH_CANDIDATES).is_some()
    && find_first_file(&dir, ABOUT_EN_CANDIDATES).is_some()
}

/// About 页正文（优先读 `$CONTENT_DIR/pages/about[.en].{mdx,md}`，未提供时回退 `assets/pages/`）。
#[post("/api/pages/about")]
pub async fn get_about_page(
  lang: Option<String>,
) -> Result<Option<AboutPageResponse>, ServerFnError> {
  #[cfg(feature = "server")]
  {
    let target_lang = match lang.as_deref() {
      Some("en") => app_core::i18n::Language::En,
      Some("zh") => app_core::i18n::Language::Zh,
      _ => SiteConfig::active().default_language_enum(),
    };
    let pages_dir = active_about_pages_dir();
    let Some((target, has_bilingual)) = resolve_about_file_in_dir(&pages_dir, target_lang) else {
      return Ok(None);
    };
    match std::fs::read_to_string(target) {
      Ok(content) => Ok(Some(AboutPageResponse { content, has_bilingual })),
      Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
      Err(e) => {
        tracing::warn!(error = %e, "about: read failed");
        Err(ServerFnError::new("读取 About 页失败"))
      }
    }
  }
  #[cfg(not(feature = "server"))]
  {
    let _ = lang;
    Ok(None)
  }
}

// ========== 布局 ==========

/// Phase 3.3：返回 `site.json::active_layout`（空字符串回退到 `"classic"`）。
#[post("/api/layout/active")]
pub async fn get_active_layout() -> Result<String, ServerFnError> {
  #[cfg(feature = "server")]
  {
    // S10：mtime 缓存读取（每次布局渲染都会调）。
    let cfg =
      SiteConfig::load_cached(site_json_path().to_str().unwrap_or_default()).unwrap_or_default();
    Ok(cfg.active_layout_or_default().to_string())
  }
  #[cfg(not(feature = "server"))]
  {
    Ok("classic".to_string())
  }
}

#[cfg(all(test, feature = "server"))]
mod tests {
  use super::{resolve_about_file_in_dir, select_about_pages_dir};
  use app_core::i18n::Language;

  #[test]
  fn about_bilingual_resolves_english_companion_and_falls_back_cleanly() {
    let tmp = std::env::temp_dir().join(format!("dp-about-i18n-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    let asset_pages = tmp.join("assets_pages");
    let custom_pages = tmp.join("custom_pages");
    std::fs::create_dir_all(&asset_pages).unwrap();
    std::fs::create_dir_all(&custom_pages).unwrap();

    std::fs::write(asset_pages.join("about.md"), "# 关于").unwrap();
    std::fs::write(asset_pages.join("about.en.md"), "# About").unwrap();

    // 1. 未挂载自定义 about 时使用内置双语目录
    assert_eq!(select_about_pages_dir(&custom_pages, &asset_pages), asset_pages);
    let (en_file, bi1) = resolve_about_file_in_dir(&asset_pages, Language::En).unwrap();
    assert_eq!(en_file, asset_pages.join("about.en.md"));
    assert!(bi1);
    let (zh_file, bi2) = resolve_about_file_in_dir(&asset_pages, Language::Zh).unwrap();
    assert_eq!(zh_file, asset_pages.join("about.md"));
    assert!(bi2);

    // 2. 挂载目录仅提供单语 about.md 时，英文请求必须回退到该挂载目录内的中文版，不得串读内置英文版
    std::fs::write(custom_pages.join("about.md"), "# 我的站点").unwrap();
    let chosen_dir = select_about_pages_dir(&custom_pages, &asset_pages);
    assert_eq!(chosen_dir, custom_pages);
    let (fallback, bi3) = resolve_about_file_in_dir(&chosen_dir, Language::En).unwrap();
    assert_eq!(fallback, custom_pages.join("about.md"));
    assert!(!bi3);

    let _ = std::fs::remove_dir_all(&tmp);
  }
}
