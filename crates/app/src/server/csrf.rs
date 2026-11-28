//! 全局 `/api/*` CSRF 防护中间件（[M2]）。
//!
//! 为什么要额外做应用层 CSRF 校验（即使 session cookie 已带 `SameSite=Lax`）：
//! 1. `dioxus-fullstack 0.7.9` 反序列化 server fn 请求时直接读原始 `Bytes` 解析 JSON，
//!    不强制检查 `Content-Type: application/json`，导致攻击页可通过 `text/plain` 表单
//!    或 `fetch(..., { mode: "no-cors" })` 绕过 CORS 预检发送有效 JSON 载荷；
//! 2. `SameSite=Lax` 不防护同主域兄弟子域（`same-site`）发起的跨源请求。
//!
//! 校验规则（作用于所有非豁免 `/api/*` 的 `POST`/`PUT`/`PATCH`/`DELETE`）：
//! - **豁免**：第三方支付服务端异步回调（`/api/pay/*/notify`），其本身受密码学验签保护；
//! - **`Sec-Fetch-Site`**：存在时仅允许 `same-origin` 或 `none`（拒绝 `cross-site` 与 `same-site`）；
//! - **`Origin`**：存在时（含 `null`）必须与当前请求 `Host`（或 `X-Forwarded-Host` / `BASE_URL`）同主机；
//! - **`Content-Type`**：除 `/api/auth/logout` 外，拒绝 CORS 简单表单类型
//!   （`text/plain`、`application/x-www-form-urlencoded`、`multipart/form-data`），回 `415`。

use axum::extract::Request;
use axum::http::header::{CONTENT_TYPE, HOST, ORIGIN};
use axum::http::{HeaderMap, Method, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};

/// 服务端到服务端的异步回调（密码学验签保护，不带浏览器 Origin/Sec-Fetch-Site）。
fn is_webhook_exempt(path: &str) -> bool {
  path.starts_with("/api/pay/") && path.ends_with("/notify")
}

/// 归一化 `host[:port]`（转小写、去末尾点；保留非默认端口的完整 authority 比较由调用方决定）。
fn normalize_authority(raw: &str) -> Option<String> {
  let s = raw.trim().trim_end_matches('/').to_ascii_lowercase();
  if s.is_empty() {
    return None;
  }
  Some(s)
}

/// 从 `Origin` 头（如 `https://example.com:8443`）提取 `host[:port]`。
fn extract_origin_authority(origin: &str) -> Option<String> {
  let trimmed = origin.trim();
  let rest = trimmed.strip_prefix("https://").or_else(|| trimmed.strip_prefix("http://"))?;
  if rest.contains('/') || rest.contains('?') || rest.contains('#') || rest.contains('@') {
    return None;
  }
  normalize_authority(rest)
}

fn expected_host_authority(headers: &HeaderMap) -> Option<String> {
  if let Some(h) = headers.get("x-forwarded-host").and_then(|v| v.to_str().ok()) {
    if let Some(first) = h.split(',').next().and_then(normalize_authority) {
      return Some(first);
    }
  }
  if let Some(h) = headers.get(HOST).and_then(|v| v.to_str().ok()) {
    if let Some(a) = normalize_authority(h) {
      return Some(a);
    }
  }
  std::env::var("BASE_URL").ok().and_then(|u| extract_origin_authority(u.trim_end_matches('/')))
}

fn is_cors_simple_content_type(ct: &str) -> bool {
  let essence = ct.split(';').next().unwrap_or("").trim().to_ascii_lowercase();
  matches!(
    essence.as_str(),
    "text/plain" | "application/x-www-form-urlencoded" | "multipart/form-data"
  )
}

/// 纯校验函数（便于单测覆盖所有绕过向量）。
pub fn validate_csrf_request(
  method: &Method,
  path: &str,
  headers: &HeaderMap,
) -> Result<(), (StatusCode, &'static str)> {
  if !path.starts_with("/api/") || is_webhook_exempt(path) {
    return Ok(());
  }
  if !matches!(*method, Method::POST | Method::PUT | Method::PATCH | Method::DELETE) {
    return Ok(());
  }

  // 1. Sec-Fetch-Site：现代浏览器必定附带；拒绝 cross-site 与 same-site（兄弟子域）
  if let Some(raw) = headers.get("sec-fetch-site").and_then(|v| v.to_str().ok()) {
    let site = raw.trim().to_ascii_lowercase();
    if site != "same-origin" && site != "none" {
      return Err((StatusCode::FORBIDDEN, "CSRF 校验未通过：禁止跨源请求"));
    }
  }

  // 2. Origin：若存在则必须严格匹配当前 Host（拒绝 `Origin: null` 与异域）
  if let Some(raw_origin) = headers.get(ORIGIN).and_then(|v| v.to_str().ok()) {
    let Some(origin_auth) = extract_origin_authority(raw_origin) else {
      return Err((StatusCode::FORBIDDEN, "CSRF 校验未通过：Origin 非法"));
    };
    let Some(expected_auth) = expected_host_authority(headers) else {
      return Err((StatusCode::FORBIDDEN, "CSRF 校验未通过：缺少 Host"));
    };
    if origin_auth != expected_auth {
      return Err((StatusCode::FORBIDDEN, "CSRF 校验未通过：Origin 不匹配"));
    }
  }

  // 3. Content-Type：阻断 `text/plain` 等免预检 CORS 简单请求投递 JSON 给 dioxus-fullstack
  if path != "/api/auth/logout" {
    if let Some(ct) = headers.get(CONTENT_TYPE).and_then(|v| v.to_str().ok()) {
      if is_cors_simple_content_type(ct) {
        return Err((StatusCode::UNSUPPORTED_MEDIA_TYPE, "不支持的请求媒体类型"));
      }
    }
  }

  Ok(())
}

/// Axum CSRF 防护中间件。
pub async fn csrf_protect_mw(req: Request, next: Next) -> Response {
  if let Err((status, msg)) = validate_csrf_request(req.method(), req.uri().path(), req.headers()) {
    tracing::warn!(
      target: "csrf",
      method = %req.method(),
      path = %req.uri().path(),
      status = status.as_u16(),
      reason = msg,
      "blocked suspicious API request"
    );
    return (status, msg).into_response();
  }
  next.run(req).await
}

#[cfg(test)]
mod tests {
  use super::*;
  use axum::http::HeaderValue;

  fn headers(pairs: &[(&str, &str)]) -> HeaderMap {
    let mut map = HeaderMap::new();
    for (k, v) in pairs {
      map.insert(
        axum::http::HeaderName::from_bytes(k.as_bytes()).expect("header name"),
        HeaderValue::from_str(v).expect("header val"),
      );
    }
    map
  }

  #[test]
  fn allows_same_origin_json_api_post() {
    let h = headers(&[
      ("host", "rustineverything.app"),
      ("origin", "https://rustineverything.app"),
      ("sec-fetch-site", "same-origin"),
      ("content-type", "application/json"),
    ]);
    assert!(validate_csrf_request(&Method::POST, "/api/comments/post", &h).is_ok());
  }

  #[test]
  fn rejects_cross_site_and_same_site_subdomain() {
    for site in ["cross-site", "same-site"] {
      let h = headers(&[
        ("host", "rustineverything.app"),
        ("sec-fetch-site", site),
        ("content-type", "application/json"),
      ]);
      let err = validate_csrf_request(&Method::POST, "/api/comments/post", &h).unwrap_err();
      assert_eq!(err.0, StatusCode::FORBIDDEN, "site={site}");
    }
  }

  #[test]
  fn rejects_origin_mismatch_and_null_origin() {
    for origin in ["https://evil.example", "https://sub.rustineverything.app", "null"] {
      let h = headers(&[
        ("host", "rustineverything.app"),
        ("origin", origin),
        ("content-type", "application/json"),
      ]);
      let err = validate_csrf_request(&Method::POST, "/api/orders/create", &h).unwrap_err();
      assert_eq!(err.0, StatusCode::FORBIDDEN, "origin={origin}");
    }
  }

  #[test]
  fn rejects_cors_simple_content_types_on_rpc_endpoints() {
    for ct in [
      "text/plain",
      "text/plain;charset=UTF-8",
      "application/x-www-form-urlencoded",
      "multipart/form-data; boundary=abc",
    ] {
      let h = headers(&[
        ("host", "rustineverything.app"),
        ("origin", "https://rustineverything.app"),
        ("sec-fetch-site", "same-origin"),
        ("content-type", ct),
      ]);
      let err = validate_csrf_request(&Method::POST, "/api/comments/post", &h).unwrap_err();
      assert_eq!(err.0, StatusCode::UNSUPPORTED_MEDIA_TYPE, "ct={ct}");
    }
  }

  #[test]
  fn allows_same_origin_form_post_on_logout_and_exempts_payment_webhooks() {
    let logout_h = headers(&[
      ("host", "rustineverything.app"),
      ("origin", "https://rustineverything.app"),
      ("sec-fetch-site", "same-origin"),
      ("content-type", "application/x-www-form-urlencoded"),
    ]);
    assert!(validate_csrf_request(&Method::POST, "/api/auth/logout", &logout_h).is_ok());

    let alipay_h = headers(&[("content-type", "application/x-www-form-urlencoded")]);
    assert!(validate_csrf_request(&Method::POST, "/api/pay/alipay/notify", &alipay_h).is_ok());
    assert!(validate_csrf_request(&Method::POST, "/api/pay/wechat/notify", &alipay_h).is_ok());
    assert!(validate_csrf_request(&Method::POST, "/api/pay/stripe/notify", &alipay_h).is_ok());
  }

  #[test]
  fn ignores_get_and_non_api_paths() {
    let h = headers(&[("sec-fetch-site", "cross-site"), ("origin", "https://evil.example")]);
    assert!(validate_csrf_request(&Method::GET, "/api/comments/list", &h).is_ok());
    assert!(validate_csrf_request(&Method::POST, "/blog/hello", &h).is_ok());
  }
}
