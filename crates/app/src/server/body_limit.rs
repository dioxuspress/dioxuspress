//! 请求体大小上限：axum 默认 2 MiB，只给图片上传放宽。
//!
//! server fn 的请求体由 dioxus-fullstack 0.7.9 用 `Bytes::from_request(..).unwrap()`
//! 读取，超过上限时直接 panic、回 500，而不是 413。`/api/upload` 允许 5 MB 图片，
//! 经 base64 再包进 JSON 后远超 2 MiB，所以这条路径的上限要跟上传上限对齐。
//!
//! 本中间件同时防御两类超限请求：
//! 1. 带 `Content-Length` 头且超限 → 立即返回 413，零内存分配；
//! 2. 缺 `Content-Length`（如 `Transfer-Encoding: chunked`）→ 先用 `axum::body::to_bytes`
//!    流式限长读入，超限返回 413，杜绝请求进入 dioxus-fullstack 触发 `.unwrap()` panic。

use axum::body::Body;
use axum::extract::{DefaultBodyLimit, Request};
use axum::http::header::CONTENT_LENGTH;
use axum::http::{HeaderValue, Method, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use tower::{Layer, ServiceExt};

/// 图片上传的 server fn 路径（`module_uploads::server::upload_image`）。
const UPLOAD_PATH: &str = "/api/upload";

/// 5 MB 图片经 base64（×4/3）约 6.7 MB，再加 JSON 外壳；取 8 MiB。
const UPLOAD_BODY_LIMIT: usize = 8 * 1024 * 1024;

/// axum `DefaultBodyLimit` 的默认值。
const DEFAULT_BODY_LIMIT: usize = 2 * 1024 * 1024;

/// 请求体上限：`/api/upload` 放宽到 8 MiB，其余路径保持 axum 默认的 2 MiB。
pub async fn body_limit_mw(req: Request, next: Next) -> Response {
  let is_upload = req.uri().path() == UPLOAD_PATH;
  let limit = if is_upload { UPLOAD_BODY_LIMIT } else { DEFAULT_BODY_LIMIT };
  let declared = req
    .headers()
    .get(CONTENT_LENGTH)
    .and_then(|v| v.to_str().ok())
    .and_then(|v| v.parse::<usize>().ok());
  if declared.is_some_and(|len| len > limit) {
    return (StatusCode::PAYLOAD_TOO_LARGE, "请求体过大").into_response();
  }

  // 对无 Content-Length 的写请求（如 Transfer-Encoding: chunked）执行流式限长校验，
  // 防止绕过头检查后在 dioxus-fullstack 内部 `Bytes::from_request().unwrap()` panic。
  let req = if declared.is_none()
    && matches!(req.method(), &Method::POST | &Method::PUT | &Method::PATCH | &Method::DELETE)
  {
    let (mut parts, body) = req.into_parts();
    match axum::body::to_bytes(body, limit).await {
      Ok(bytes) => {
        if let Ok(v) = HeaderValue::from_str(&bytes.len().to_string()) {
          parts.headers.insert(CONTENT_LENGTH, v);
        }
        Request::from_parts(parts, Body::from(bytes))
      }
      Err(_) => return (StatusCode::PAYLOAD_TOO_LARGE, "请求体过大").into_response(),
    }
  } else {
    req
  };

  if !is_upload {
    return next.run(req).await;
  }
  match DefaultBodyLimit::max(UPLOAD_BODY_LIMIT).layer(next).oneshot(req).await {
    Ok(res) => res,
    Err(never) => match never {},
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use axum::body::{Body, Bytes};
  use axum::http::StatusCode;
  use axum::routing::post;
  use axum::Router;

  fn app() -> Router {
    let echo_len = |body: Bytes| async move { body.len().to_string() };
    Router::new()
      .route(UPLOAD_PATH, post(echo_len))
      .route("/api/other", post(echo_len))
      .layer(axum::middleware::from_fn(body_limit_mw))
  }

  async fn post_bytes(path: &str, len: usize) -> StatusCode {
    let req = Request::post(path).body(Body::from(vec![b'a'; len])).unwrap();
    app().oneshot(req).await.unwrap().status()
  }

  #[tokio::test]
  async fn upload_accepts_bodies_above_the_default_limit() {
    assert_eq!(post_bytes(UPLOAD_PATH, 3 * 1024 * 1024).await, StatusCode::OK);
    assert_eq!(post_bytes(UPLOAD_PATH, UPLOAD_BODY_LIMIT + 1).await, StatusCode::PAYLOAD_TOO_LARGE);
  }

  #[tokio::test]
  async fn other_paths_keep_the_default_limit() {
    assert_eq!(post_bytes("/api/other", 3 * 1024 * 1024).await, StatusCode::PAYLOAD_TOO_LARGE);
    assert_eq!(post_bytes("/api/other", 1024).await, StatusCode::OK);
  }

  /// dioxus 的 server fn 读请求体失败时 unwrap panic（500）；声明的长度超限时，
  /// 中间件要在请求到达处理函数之前就回 413。
  #[tokio::test]
  async fn declared_length_over_limit_is_rejected_before_the_handler() {
    let app = Router::new()
      .route("/api/ignores_body", post(|| async { "handled" }))
      .route(UPLOAD_PATH, post(|| async { "handled" }))
      .layer(axum::middleware::from_fn(body_limit_mw));
    for (path, len) in
      [("/api/ignores_body", 3 * 1024 * 1024), (UPLOAD_PATH, UPLOAD_BODY_LIMIT + 1)]
    {
      let req = Request::post(path)
        .header(axum::http::header::CONTENT_LENGTH, len)
        .body(Body::from(vec![b'a'; len]))
        .unwrap();
      let status = app.clone().oneshot(req).await.unwrap().status();
      assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE, "{path}");
    }
  }

  /// [M5] 缺少 Content-Length（chunked 传输）且超限的请求也必须在进入 handler 前回 413。
  #[tokio::test]
  async fn chunked_body_without_content_length_over_limit_returns_413_before_handler() {
    let app = Router::new()
      .route(
        "/api/chunked",
        post(|| async { panic!("handler must not run for oversized chunked body") }),
      )
      .layer(axum::middleware::from_fn(body_limit_mw));
    let req =
      Request::post("/api/chunked").body(Body::from(vec![b'x'; DEFAULT_BODY_LIMIT + 1])).unwrap();
    let status = app.oneshot(req).await.unwrap().status();
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
  }
}
