//! E5.2：匿名只读页面弱 `ETag` / `If-None-Match`（`304 Not Modified`）条件缓存与私密路由防串缓存守卫。
//!
//! ## 设计边界与安全红线
//! 1. **私密 / 动态路径强隔离**：`/api`、`/admin`、`/me`、`/forum`、`/topics`、`/course`、
//!    `/papers`、`/pay`、`/orders`、`/healthz` 一律下发 `Cache-Control: private, no-store`
//!    并剥离 `ETag`，绝不读写公共表。
//! 2. **登录态 / 凭据请求强隔离（Fail-Closed）**：凡携带 `Authorization` 头，或 `Cookie` 头中
//!    出现除纯展示偏好（`site_lang`、`site_theme`）以外的**任何** Cookie（含 `session`、
//!    `oauth_pkce`、`__Host-*` 等，包括 HTTP/2 多条同名 `cookie` 头），一律按登录态处理：
//!    下发 `Cache-Control: private, no-store`，绝不触碰公共缓存。
//! 3. **匿名只读快路径**：匿名 `GET` / `HEAD` 访问公开内容页（`/`、`/docs/*`、`/blog/*`、
//!    `/templates`、`/sitemap.xml`、`/feed.xml` 等）时：
//!    - 仅缓存元信息 `(epoch, source_stamp, etag, cache_control)`（表上限 4096 条，不存 HTML 正文，
//!      内存占用 < 512 KiB）；
//!    - 命中 `If-None-Match` 且内容纪元与源文件时间戳未变时，直接短路返回 `304 Not Modified`，
//!      完全跳过 Markdown 解析与 Dioxus VirtualDOM SSR；
//!    - 响应追加 `Vary: Cookie`，确保语言（`site_lang`）与主题（`site_theme`）切换互不串扰。

use std::collections::{HashMap, VecDeque};
use std::path::Path;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{OnceLock, RwLock};
use std::task::{Context, Poll};
use std::time::UNIX_EPOCH;

use axum::body::{Body, Bytes, HttpBody};
use axum::extract::Request;
use axum::http::{header, HeaderMap, HeaderValue, Method, StatusCode};
use axum::middleware::Next;
use axum::response::Response;

/// 单表最大缓存键数量（仅存元数据，约数十字节/条；超限直接清空重建，防随机 query 撑大内存）。
const MAX_ETAG_ENTRIES: usize = 4096;
/// 允许进入条件缓存计算的最大响应体积（4 MiB）。
const MAX_CACHEABLE_BODY_BYTES: usize = 4 * 1024 * 1024;
/// 允许进入缓存键的最大 URI（path + query）字节长度。
const MAX_URI_KEY_LEN: usize = 512;

const PRIVATE_NO_STORE: &str = "private, no-store";
const PUBLIC_REVALIDATE: &str = "public, max-age=0, must-revalidate";

static CONTENT_EPOCH: AtomicU64 = AtomicU64::new(0);

/// 当前内容纪元号（`SIGHUP` 热刷新或管理端变更配置时原子递增）。
pub fn current_content_epoch() -> u64 {
  CONTENT_EPOCH.load(Ordering::Acquire)
}

/// 原子递增内容纪元并清空已缓存的 `ETag` 元数据表（供热刷新与测试调用）。
pub fn bump_content_epoch() -> u64 {
  let next = CONTENT_EPOCH.fetch_add(1, Ordering::AcqRel) + 1;
  if let Ok(mut guard) = etag_table().write() {
    guard.clear();
  }
  next
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct CacheKey {
  uri: String,
  lang: String,
  theme: String,
}

#[derive(Debug, Clone)]
struct CachedETag {
  epoch: u64,
  source_stamp: u128,
  etag: HeaderValue,
  cache_control: HeaderValue,
}

fn etag_table() -> &'static RwLock<HashMap<CacheKey, CachedETag>> {
  static TABLE: OnceLock<RwLock<HashMap<CacheKey, CachedETag>>> = OnceLock::new();
  TABLE.get_or_init(|| RwLock::new(HashMap::new()))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RouteClass {
  /// 已由 `ServeDir` 或构建产物路由托管的静态文件，原样透传。
  StaticAsset,
  /// 涉及会话、权限、交互或后台管理的动态路径，强制 `private, no-store`。
  PrivateDynamic,
  /// 公开只读 SSR 页面与 XML/文本索引，对纯匿名访客启用弱 `ETag` 条件缓存。
  AnonymousReadOnly,
}

fn matches_seg(path: &str, prefix: &str) -> bool {
  path == prefix
    || (path.len() > prefix.len()
      && path.starts_with(prefix)
      && path.as_bytes().get(prefix.len()) == Some(&b'/'))
}

fn extra_static_paths_slot() -> &'static RwLock<Vec<&'static str>> {
  static EXTRA_STATIC_PATHS: OnceLock<RwLock<Vec<&'static str>>> = OnceLock::new();
  EXTRA_STATIC_PATHS.get_or_init(|| RwLock::new(Vec::new()))
}

fn extra_private_prefixes_slot() -> &'static RwLock<Vec<&'static str>> {
  static EXTRA_PRIVATE_PREFIXES: OnceLock<RwLock<Vec<&'static str>>> = OnceLock::new();
  EXTRA_PRIVATE_PREFIXES.get_or_init(|| RwLock::new(Vec::new()))
}

/// 注册宿主站点的自定义条件缓存分类规则（支持多模块叠加追加，不再受单槽 OnceLock 限制）。
pub fn register_extra_route_rules(
  static_paths: &'static [&'static str],
  private_prefixes: &'static [&'static str],
) {
  if let Ok(mut guard) = extra_static_paths_slot().write() {
    for &p in static_paths {
      if !guard.contains(&p) {
        guard.push(p);
      }
    }
  }
  if let Ok(mut guard) = extra_private_prefixes_slot().write() {
    for &p in private_prefixes {
      if !guard.contains(&p) {
        guard.push(p);
      }
    }
  }
}

fn classify_path(path: &str) -> RouteClass {
  if extra_static_paths_slot().read().ok().is_some_and(|list| list.contains(&path)) {
    return RouteClass::StaticAsset;
  }

  for static_prefix in [
    "/assets",
    "/images",
    "/schemas",
    "/posts",
    "/docs-files",
    "/js",
    "/uploads",
    "/audio",
    "/podcasts",
    "/cases",
    "/courses",
  ] {
    if matches_seg(path, static_prefix) {
      return RouteClass::StaticAsset;
    }
  }

  for dynamic_prefix in
    ["/api", "/admin", "/me", "/forum", "/topics", "/course", "/pay", "/orders", "/healthz"]
  {
    if matches_seg(path, dynamic_prefix) {
      return RouteClass::PrivateDynamic;
    }
  }

  if extra_private_prefixes_slot()
    .read()
    .ok()
    .is_some_and(|list| list.iter().any(|prefix| matches_seg(path, prefix)))
  {
    return RouteClass::PrivateDynamic;
  }

  RouteClass::AnonymousReadOnly
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Anonymity {
  Anonymous { lang: String, theme: String },
  AuthenticatedOrSensitive,
}

/// 严格白名单解析请求匿名性：除 `site_lang` / `site_theme` 外出现任何 Cookie 或 `Authorization` 头即视为非匿名。
fn inspect_anonymity(headers: &HeaderMap) -> Anonymity {
  if headers.contains_key(header::AUTHORIZATION) {
    return Anonymity::AuthenticatedOrSensitive;
  }

  let mut lang = String::new();
  let mut theme = String::new();

  for raw_cookie in headers.get_all(header::COOKIE) {
    let Ok(cookie_str) = raw_cookie.to_str() else {
      return Anonymity::AuthenticatedOrSensitive;
    };
    for part in cookie_str.split(';') {
      let trimmed = part.trim();
      if trimmed.is_empty() {
        continue;
      }
      let Some((raw_k, raw_v)) = trimmed.split_once('=') else {
        return Anonymity::AuthenticatedOrSensitive;
      };
      let key = raw_k.trim();
      let val = raw_v.trim();
      match key {
        "site_lang" => {
          if val.len() > 16 || !val.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-') {
            return Anonymity::AuthenticatedOrSensitive;
          }
          lang = val.to_ascii_lowercase();
        }
        "site_theme" => {
          if val.len() > 32
            || !val.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
          {
            return Anonymity::AuthenticatedOrSensitive;
          }
          theme = val.to_ascii_lowercase();
        }
        _ => return Anonymity::AuthenticatedOrSensitive,
      }
    }
  }

  Anonymity::Anonymous { lang, theme }
}

/// RFC 9110 §13.1.2 弱比较：剥离可选 `W/` 前缀后比对引号包裹的实体标签，支持 `*` 与逗号分隔列表。
pub fn if_none_match_matches(inm_header: &str, current_etag: &str) -> bool {
  let target = strip_weak_prefix(current_etag.trim());
  if !target.starts_with('"') || !target.ends_with('"') || target.len() < 2 {
    return false;
  }
  for candidate in inm_header.split(',') {
    let c = candidate.trim();
    if c == "*" {
      return true;
    }
    if strip_weak_prefix(c) == target {
      return true;
    }
  }
  false
}

fn strip_weak_prefix(tag: &str) -> &str {
  tag.strip_prefix("W/").or_else(|| tag.strip_prefix("w/")).unwrap_or(tag)
}

fn build_stamp() -> u64 {
  static STAMP: OnceLock<u64> = OnceLock::new();
  *STAMP.get_or_init(|| {
    let mut h = fnv1a64(env!("CARGO_PKG_VERSION").as_bytes());
    if let Ok(exe) = std::env::current_exe() {
      if let Ok(meta) = std::fs::metadata(exe) {
        h = fnv1a64_extend(h, &meta.len().to_le_bytes());
        if let Ok(mtime) = meta.modified() {
          if let Ok(dur) = mtime.duration_since(UNIX_EPOCH) {
            h = fnv1a64_extend(h, &dur.as_nanos().to_le_bytes());
          }
        }
      }
    }
    h
  })
}

fn mtime_nanos(path: &Path) -> u128 {
  std::fs::metadata(path)
    .ok()
    .and_then(|m| m.modified().ok())
    .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
    .map(|d| d.as_nanos())
    .unwrap_or(0)
}

fn max_mtime_in(candidates: impl IntoIterator<Item = std::path::PathBuf>) -> u128 {
  candidates.into_iter().map(|p| mtime_nanos(&p)).max().unwrap_or(0)
}

pub type EtagStampFn = fn(&str) -> Option<u128>;

fn extra_etag_stamps_slot() -> &'static RwLock<Vec<EtagStampFn>> {
  static EXTRA_ETAG_STAMP: OnceLock<RwLock<Vec<EtagStampFn>>> = OnceLock::new();
  EXTRA_ETAG_STAMP.get_or_init(|| RwLock::new(Vec::new()))
}

/// 注册宿主站点的自定义路由 ETag 版本戳计算函数（支持多模块叠加追加，命中首个 `Some` 即生效）。
pub fn register_extra_etag_stamp(callback: EtagStampFn) {
  if let Ok(mut guard) = extra_etag_stamps_slot().write() {
    guard.push(callback);
  }
}

/// 轻量内容源指纹（仅数次 VFS `stat`，微秒级）：确保本地改 Markdown 或 `site.json` 无需等信号即刻换 `ETag`。
fn probe_source_stamp(path: &str) -> u128 {
  let site_stamp = mtime_nanos(&app_core::utils::site_json_path());
  if let Ok(guard) = extra_etag_stamps_slot().read() {
    if let Some(extra) = guard.iter().find_map(|f| f(path)) {
      return site_stamp.wrapping_mul(6364136223846793005) ^ extra;
    }
  }
  let docs_root = app_core::utils::content_dir("docs");
  let posts_root = app_core::utils::content_dir("posts");
  let cases_root = app_core::utils::content_dir("cases");
  let podcasts_root = app_core::utils::content_dir("podcasts");

  let route_stamp = if path == "/docs" || path.starts_with("/docs/") {
    let dir_stamp = mtime_nanos(&docs_root) ^ mtime_nanos(&docs_root.join("_meta.json"));
    let rel = path.strip_prefix("/docs/").unwrap_or("");
    if rel.contains("..") {
      dir_stamp
    } else {
      let node_dir = if rel.is_empty() { docs_root } else { docs_root.join(rel) };
      let leaf_stamp = max_mtime_in([
        node_dir.join("index.md"),
        node_dir.join("index.mdx"),
        node_dir.join("index.en.md"),
        node_dir.join("index.en.mdx"),
        node_dir.join("_meta.json"),
        node_dir.join("_meta.en.json"),
      ]);
      dir_stamp.wrapping_mul(1_000_003) ^ leaf_stamp
    }
  } else if path == "/blog" || path.starts_with("/blog/") {
    let dir_stamp = mtime_nanos(&posts_root);
    if let Some(slug) = path.strip_prefix("/blog/").filter(|s| !s.is_empty() && !s.contains("..")) {
      let leaf_stamp = max_mtime_in([
        posts_root.join(slug).join("index.md"),
        posts_root.join(slug).join("index.mdx"),
        posts_root.join(format!("{slug}.md")),
        posts_root.join(format!("{slug}.mdx")),
      ]);
      dir_stamp.wrapping_mul(1_000_003) ^ leaf_stamp
    } else {
      dir_stamp
    }
  } else if path == "/case" || path.starts_with("/case/") {
    let dir_stamp = mtime_nanos(&cases_root);
    if let Some(slug) = path.strip_prefix("/case/").filter(|s| app_core::utils::is_safe_slug(s)) {
      let leaf_stamp = max_mtime_in([
        cases_root.join(slug).join("case.yaml"),
        cases_root.join(slug).join("README.md"),
      ]);
      dir_stamp.wrapping_mul(1_000_003) ^ leaf_stamp
    } else {
      dir_stamp
    }
  } else if path == "/podcast" || path.starts_with("/podcast/") {
    mtime_nanos(&podcasts_root)
  } else if path == "/about" {
    let pages_dir = crate::server::active_about_pages_dir();
    let dir_stamp = mtime_nanos(&pages_dir);
    let leaf_stamp = max_mtime_in([
      pages_dir.join("about.md"),
      pages_dir.join("about.mdx"),
      pages_dir.join("about.en.md"),
      pages_dir.join("about.en.mdx"),
    ]);
    dir_stamp.wrapping_mul(1_000_003) ^ leaf_stamp
  } else {
    mtime_nanos(&docs_root).wrapping_mul(1_000_003)
      ^ mtime_nanos(&posts_root).wrapping_mul(1_000_033)
      ^ mtime_nanos(&cases_root).wrapping_mul(1_000_037)
      ^ mtime_nanos(&podcasts_root)
  };

  site_stamp.wrapping_mul(6364136223846793005) ^ route_stamp
}

fn fnv1a64(bytes: &[u8]) -> u64 {
  fnv1a64_extend(0xcbf29ce484222325, bytes)
}

fn fnv1a64_extend(mut hash: u64, bytes: &[u8]) -> u64 {
  for &b in bytes {
    hash ^= u64::from(b);
    hash = hash.wrapping_mul(0x100000001b3);
  }
  hash
}

fn compute_weak_etag(
  epoch: u64,
  source_stamp: u128,
  key: &CacheKey,
  body: &[u8],
) -> Option<HeaderValue> {
  let mut h = build_stamp();
  h = fnv1a64_extend(h, &epoch.to_le_bytes());
  h = fnv1a64_extend(h, &source_stamp.to_le_bytes());
  h = fnv1a64_extend(h, key.lang.as_bytes());
  h = fnv1a64_extend(h, &[0xff]);
  h = fnv1a64_extend(h, key.theme.as_bytes());
  h = fnv1a64_extend(h, &[0xff]);
  h = fnv1a64_extend(h, body);
  let tag = format!("W/\"{:016x}-{:x}\"", h, body.len());
  HeaderValue::from_str(&tag).ok()
}

fn ensure_vary_cookie(headers: &mut HeaderMap) {
  let already_has_cookie = headers
    .get_all(header::VARY)
    .iter()
    .filter_map(|v| v.to_str().ok())
    .flat_map(|s| s.split(','))
    .any(|part| part.trim().eq_ignore_ascii_case("cookie") || part.trim() == "*");
  if !already_has_cookie {
    headers.append(header::VARY, HeaderValue::from_static("Cookie"));
  }
}

fn enforce_private_no_store(mut resp: Response) -> Response {
  resp.headers_mut().insert(header::CACHE_CONTROL, HeaderValue::from_static(PRIVATE_NO_STORE));
  resp.headers_mut().remove(header::ETAG);
  resp
}

fn not_modified_response(etag: HeaderValue, cache_control: HeaderValue) -> Response {
  let mut resp = Response::new(Body::empty());
  *resp.status_mut() = StatusCode::NOT_MODIFIED;
  resp.headers_mut().insert(header::ETAG, etag);
  resp.headers_mut().insert(header::CACHE_CONTROL, cache_control);
  ensure_vary_cookie(resp.headers_mut());
  resp
}

/// 响应头显式要求禁止共享缓存，或携带 `Set-Cookie` 时绝不生成公共 ETag。
fn response_forbids_public_cache(headers: &HeaderMap) -> bool {
  if headers.contains_key(header::SET_COOKIE) {
    return true;
  }
  headers.get_all(header::CACHE_CONTROL).iter().filter_map(|v| v.to_str().ok()).any(|v| {
    let lc = v.to_ascii_lowercase();
    lc.contains("no-store") || lc.contains("private")
  })
}

/// 当流式响应超过 `MAX_CACHEABLE_BODY_BYTES` 时，把已读帧与剩余未读流无损拼接透传，
/// 彻底杜绝超过 4 MiB 被静默截断为空 200 OK 的数据损坏。
struct ReplayBody {
  buffered: VecDeque<Bytes>,
  rest: Body,
}

impl HttpBody for ReplayBody {
  type Data = Bytes;
  type Error = axum::Error;

  fn poll_frame(
    self: Pin<&mut Self>,
    cx: &mut Context<'_>,
  ) -> Poll<Option<Result<http_body::Frame<Self::Data>, Self::Error>>> {
    let this = self.get_mut();
    if let Some(chunk) = this.buffered.pop_front() {
      return Poll::Ready(Some(Ok(http_body::Frame::data(chunk))));
    }
    Pin::new(&mut this.rest).poll_frame(cx)
  }
}

enum CollectResult {
  Complete(Bytes),
  Overflow(Body),
}

async fn collect_up_to_limit(mut body: Body, limit: usize) -> CollectResult {
  if body.size_hint().lower() > limit as u64 {
    return CollectResult::Overflow(body);
  }
  let mut chunks: VecDeque<Bytes> = VecDeque::new();
  let mut total = 0usize;
  loop {
    let next = std::future::poll_fn(|cx| Pin::new(&mut body).poll_frame(cx)).await;
    match next {
      Some(Ok(frame)) => {
        if let Ok(data) = frame.into_data() {
          if data.is_empty() {
            continue;
          }
          total = total.saturating_add(data.len());
          chunks.push_back(data);
          if total > limit {
            return CollectResult::Overflow(Body::new(ReplayBody { buffered: chunks, rest: body }));
          }
        }
      }
      Some(Err(_)) => {
        return CollectResult::Overflow(Body::new(ReplayBody { buffered: chunks, rest: body }));
      }
      None => {
        if chunks.len() == 1 {
          if let Some(only) = chunks.pop_front() {
            return CollectResult::Complete(only);
          }
        }
        let mut out = Vec::with_capacity(total);
        for c in chunks {
          out.extend_from_slice(&c);
        }
        return CollectResult::Complete(Bytes::from(out));
      }
    }
  }
}

/// Axum 条件缓存中间件：挂在 `CompressionLayer` 内侧，使 `304` 短路在 SSR 与压缩之前完成。
pub async fn conditional_cache_mw(req: Request, next: Next) -> Response {
  let path = req.uri().path();
  match classify_path(path) {
    RouteClass::StaticAsset => return next.run(req).await,
    RouteClass::PrivateDynamic => {
      return enforce_private_no_store(next.run(req).await);
    }
    RouteClass::AnonymousReadOnly => {}
  }

  let method = req.method().clone();
  if method != Method::GET && method != Method::HEAD {
    return enforce_private_no_store(next.run(req).await);
  }

  let Anonymity::Anonymous { lang, theme } = inspect_anonymity(req.headers()) else {
    return enforce_private_no_store(next.run(req).await);
  };

  let uri_str =
    req.uri().path_and_query().map(|pq| pq.as_str()).unwrap_or_else(|| req.uri().path());
  if uri_str.len() > MAX_URI_KEY_LEN {
    let mut resp = next.run(req).await;
    ensure_vary_cookie(resp.headers_mut());
    return resp;
  }

  let key = CacheKey { uri: uri_str.to_string(), lang, theme };
  let epoch = current_content_epoch();
  let source_stamp = probe_source_stamp(path);
  let inm =
    req.headers().get(header::IF_NONE_MATCH).and_then(|v| v.to_str().ok()).map(str::to_owned);

  if let Some(ref inm_val) = inm {
    if let Ok(guard) = etag_table().read() {
      if let Some(entry) = guard.get(&key) {
        if entry.epoch == epoch && entry.source_stamp == source_stamp {
          if let Ok(etag_str) = entry.etag.to_str() {
            if if_none_match_matches(inm_val, etag_str) {
              return not_modified_response(entry.etag.clone(), entry.cache_control.clone());
            }
          }
        }
      }
    }
  }

  let resp = next.run(req).await;
  if resp.status() != StatusCode::OK {
    return resp;
  }
  if response_forbids_public_cache(resp.headers()) {
    return enforce_private_no_store(resp);
  }
  let declared_oversize = resp
    .headers()
    .get(header::CONTENT_LENGTH)
    .and_then(|v| v.to_str().ok())
    .and_then(|v| v.parse::<usize>().ok())
    .is_some_and(|n| n > MAX_CACHEABLE_BODY_BYTES);
  if declared_oversize {
    let mut resp = resp;
    ensure_vary_cookie(resp.headers_mut());
    return resp;
  }

  let (mut parts, body) = resp.into_parts();
  let bytes = match collect_up_to_limit(body, MAX_CACHEABLE_BODY_BYTES).await {
    CollectResult::Complete(b) => b,
    CollectResult::Overflow(passthrough) => {
      ensure_vary_cookie(&mut parts.headers);
      let out = if method == Method::HEAD { Body::empty() } else { passthrough };
      return Response::from_parts(parts, out);
    }
  };

  let cache_control = parts
    .headers
    .get(header::CACHE_CONTROL)
    .cloned()
    .unwrap_or_else(|| HeaderValue::from_static(PUBLIC_REVALIDATE));
  parts.headers.insert(header::CACHE_CONTROL, cache_control.clone());
  ensure_vary_cookie(&mut parts.headers);

  if let Some(etag) = compute_weak_etag(epoch, source_stamp, &key, &bytes) {
    if let Ok(mut guard) = etag_table().write() {
      if guard.len() >= MAX_ETAG_ENTRIES && !guard.contains_key(&key) {
        guard.clear();
      }
      guard.insert(
        key,
        CachedETag {
          epoch,
          source_stamp,
          etag: etag.clone(),
          cache_control: cache_control.clone(),
        },
      );
    }

    if let Some(ref inm_val) = inm {
      if let Ok(etag_str) = etag.to_str() {
        if if_none_match_matches(inm_val, etag_str) {
          return not_modified_response(etag, cache_control);
        }
      }
    }

    parts.headers.insert(header::ETAG, etag);
  }

  let out_body = if method == Method::HEAD { Body::empty() } else { Body::from(bytes) };
  Response::from_parts(parts, out_body)
}

#[cfg(test)]
mod tests {
  use std::sync::atomic::{AtomicUsize, Ordering};
  use std::sync::Arc;

  use axum::routing::get;
  use axum::Router;
  use tower::ServiceExt;

  use super::*;

  async fn send(
    app: &Router,
    uri: &str,
    cookie: Option<&str>,
    inm: Option<&str>,
  ) -> axum::response::Response {
    let mut builder = Request::builder().uri(uri);
    if let Some(c) = cookie {
      builder = builder.header(header::COOKIE, c);
    }
    if let Some(tag) = inm {
      builder = builder.header(header::IF_NONE_MATCH, tag);
    }
    app.clone().oneshot(builder.body(Body::empty()).expect("req")).await.expect("resp")
  }

  #[test]
  fn rfc9110_weak_if_none_match_comparison() {
    assert!(if_none_match_matches(r#"W/"abc-123""#, r#"W/"abc-123""#));
    assert!(if_none_match_matches(r#""abc-123""#, r#"W/"abc-123""#));
    assert!(if_none_match_matches(r#"W/"other", W/"abc-123""#, r#"W/"abc-123""#));
    assert!(if_none_match_matches("*", r#"W/"abc-123""#));
    assert!(!if_none_match_matches(r#"W/"different""#, r#"W/"abc-123""#));
    assert!(!if_none_match_matches("", r#"W/"abc-123""#));
  }

  #[test]
  fn route_classification_separates_ssr_static_and_private() {
    assert_eq!(classify_path("/"), RouteClass::AnonymousReadOnly);
    assert_eq!(classify_path("/docs/guide/quickstart"), RouteClass::AnonymousReadOnly);
    assert_eq!(classify_path("/blog/welcome"), RouteClass::AnonymousReadOnly);
    assert_eq!(classify_path("/case/demo"), RouteClass::AnonymousReadOnly);
    assert_eq!(classify_path("/podcast"), RouteClass::AnonymousReadOnly);
    assert_eq!(classify_path("/sitemap.xml"), RouteClass::AnonymousReadOnly);
    assert_eq!(classify_path("/feed.xml"), RouteClass::AnonymousReadOnly);

    assert_eq!(classify_path("/docs-files/guide/pic.png"), RouteClass::StaticAsset);
    assert_eq!(classify_path("/posts/welcome/cover.jpg"), RouteClass::StaticAsset);
    assert_eq!(classify_path("/cases/demo/shot.png"), RouteClass::StaticAsset);
    assert_eq!(classify_path("/podcasts/ep1.mp3"), RouteClass::StaticAsset);
    assert_eq!(classify_path("/courses/rust/01/index.md"), RouteClass::StaticAsset);

    assert_eq!(classify_path("/admin"), RouteClass::PrivateDynamic);
    assert_eq!(classify_path("/admin/users"), RouteClass::PrivateDynamic);
    assert_eq!(classify_path("/me"), RouteClass::PrivateDynamic);
    assert_eq!(classify_path("/forum"), RouteClass::PrivateDynamic);
    assert_eq!(classify_path("/topics/42"), RouteClass::PrivateDynamic);
    assert_eq!(classify_path("/course/rust-basics"), RouteClass::PrivateDynamic);
    assert_eq!(classify_path("/api/docs/content"), RouteClass::PrivateDynamic);
  }

  #[tokio::test]
  async fn anonymous_repeat_get_hits_304_and_skips_ssr_handler() {
    bump_content_epoch();
    let renders = Arc::new(AtomicUsize::new(0));
    let renders_clone = renders.clone();

    let app = Router::new()
      .route(
        "/docs/guide/quickstart",
        get(move || {
          let r = renders_clone.clone();
          async move {
            let n = r.fetch_add(1, Ordering::SeqCst) + 1;
            format!("<html>render #{n}</html>")
          }
        }),
      )
      .layer(axum::middleware::from_fn(conditional_cache_mw));

    let r1 = send(&app, "/docs/guide/quickstart", None, None).await;
    assert_eq!(r1.status(), StatusCode::OK);
    let etag =
      r1.headers().get(header::ETAG).expect("etag present").to_str().expect("str").to_string();
    assert!(etag.starts_with("W/\""));
    assert_eq!(
      r1.headers().get(header::CACHE_CONTROL).and_then(|v| v.to_str().ok()),
      Some(PUBLIC_REVALIDATE)
    );
    assert!(
      r1.headers()
        .get_all(header::VARY)
        .iter()
        .filter_map(|v| v.to_str().ok())
        .any(|v| v.contains("Cookie")),
      "Vary must include Cookie"
    );
    let b1 = axum::body::to_bytes(r1.into_body(), 4096).await.expect("body");
    assert_eq!(&b1[..], b"<html>render #1</html>");
    assert_eq!(renders.load(Ordering::SeqCst), 1);

    // Repeat GET with matching If-None-Match MUST return 304 without invoking SSR handler again
    let r2 = send(&app, "/docs/guide/quickstart", None, Some(&etag)).await;
    assert_eq!(r2.status(), StatusCode::NOT_MODIFIED);
    assert_eq!(renders.load(Ordering::SeqCst), 1, "SSR handler must be skipped on 304 hit");
    let b2 = axum::body::to_bytes(r2.into_body(), 4096).await.expect("empty");
    assert!(b2.is_empty());

    // Different site_lang cookie MUST NOT hit previous ETag
    let r_en = send(&app, "/docs/guide/quickstart", Some("site_lang=en"), Some(&etag)).await;
    assert_eq!(r_en.status(), StatusCode::OK);
    let etag_en =
      r_en.headers().get(header::ETAG).expect("etag_en").to_str().expect("str").to_string();
    assert_ne!(etag, etag_en, "English cookie representation must get a distinct ETag");
    assert_eq!(renders.load(Ordering::SeqCst), 2);

    // Different site_theme cookie MUST NOT hit previous ETag
    let r_theme =
      send(&app, "/docs/guide/quickstart", Some("site_theme=sunset"), Some(&etag)).await;
    assert_eq!(r_theme.status(), StatusCode::OK);
    let etag_theme =
      r_theme.headers().get(header::ETAG).expect("etag_theme").to_str().expect("str").to_string();
    assert_ne!(etag, etag_theme, "Different theme representation must get a distinct ETag");
    assert_eq!(renders.load(Ordering::SeqCst), 3);

    // Bumping content epoch (hot reload) invalidates cached ETag immediately
    bump_content_epoch();
    let r3 = send(&app, "/docs/guide/quickstart", None, Some(&etag)).await;
    assert_eq!(r3.status(), StatusCode::OK);
    assert_eq!(renders.load(Ordering::SeqCst), 4);
  }

  #[tokio::test]
  async fn session_cookies_and_dynamic_routes_enforce_private_no_store_and_never_cache() {
    bump_content_epoch();
    let public_renders = Arc::new(AtomicUsize::new(0));
    let pr = public_renders.clone();

    let app = Router::new()
      .route(
        "/docs/guide/quickstart",
        get(move |headers: HeaderMap| {
          let pr = pr.clone();
          async move {
            pr.fetch_add(1, Ordering::SeqCst);
            let has_sess = headers
              .get(header::COOKIE)
              .and_then(|v| v.to_str().ok())
              .is_some_and(|c| c.contains("session="));
            if has_sess {
              "<html>Hello Alice (logged-in)</html>"
            } else {
              "<html>Hello Anonymous</html>"
            }
          }
        }),
      )
      .route("/admin", get(|| async { "admin panel" }))
      .route("/me", get(|| async { "profile" }))
      .route("/forum", get(|| async { "forum" }))
      .route("/course/rust-basics", get(|| async { "course" }))
      .route("/api/test", get(|| async { "{\"ok\":true}" }))
      .layer(axum::middleware::from_fn(conditional_cache_mw));

    // 1. Authenticated user visits /docs/guide/quickstart first: must get private, no-store, no ETag,
    //    and MUST NOT poison the anonymous cache
    let auth_resp =
      send(&app, "/docs/guide/quickstart", Some("site_lang=en; session=jwt-secret-alice"), None)
        .await;
    assert_eq!(auth_resp.status(), StatusCode::OK);
    assert_eq!(
      auth_resp.headers().get(header::CACHE_CONTROL).and_then(|v| v.to_str().ok()),
      Some("private, no-store")
    );
    assert!(
      auth_resp.headers().get(header::ETAG).is_none(),
      "logged-in response must not emit ETag"
    );
    let auth_body = axum::body::to_bytes(auth_resp.into_body(), 4096).await.expect("body");
    assert_eq!(&auth_body[..], b"<html>Hello Alice (logged-in)</html>");

    // 2. Subsequent anonymous visitor gets public anonymous HTML, never Alice's HTML
    let anon_resp = send(&app, "/docs/guide/quickstart", None, None).await;
    let anon_etag =
      anon_resp.headers().get(header::ETAG).expect("anon etag").to_str().expect("str").to_string();
    let anon_body = axum::body::to_bytes(anon_resp.into_body(), 4096).await.expect("body");
    assert_eq!(&anon_body[..], b"<html>Hello Anonymous</html>");

    // 3. Authenticated user sending anonymous ETag in If-None-Match STILL bypasses 304 and gets private, no-store
    let auth_recheck =
      send(&app, "/docs/guide/quickstart", Some("session=jwt-secret-alice"), Some(&anon_etag))
        .await;
    assert_eq!(auth_recheck.status(), StatusCode::OK);
    assert_eq!(
      auth_recheck.headers().get(header::CACHE_CONTROL).and_then(|v| v.to_str().ok()),
      Some("private, no-store")
    );
    let recheck_body = axum::body::to_bytes(auth_recheck.into_body(), 4096).await.expect("body");
    assert_eq!(&recheck_body[..], b"<html>Hello Alice (logged-in)</html>");

    // 4. Dynamic routes (/admin, /me, /forum, /course/*, /api/*) ALWAYS get private, no-store even without cookies
    for dyn_path in ["/admin", "/me", "/forum", "/course/rust-basics", "/api/test"] {
      let r = send(&app, dyn_path, None, None).await;
      assert_eq!(
        r.headers().get(header::CACHE_CONTROL).and_then(|v| v.to_str().ok()),
        Some("private, no-store"),
        "{dyn_path} must emit Cache-Control: private, no-store"
      );
      assert!(r.headers().get(header::ETAG).is_none(), "{dyn_path} must not emit ETag");
    }
  }

  #[tokio::test]
  async fn oversized_body_streams_intact_and_set_cookie_skips_public_cache() {
    bump_content_epoch();
    let payload_len = MAX_CACHEABLE_BODY_BYTES + 1024;
    let app = Router::new()
      .route("/sitemap.xml", get(move || async move { vec![b'x'; payload_len] }))
      .route(
        "/set-cookie-page",
        get(|| async {
          ([(header::SET_COOKIE, HeaderValue::from_static("site_theme=ocean; Path=/"))], "hello")
        }),
      )
      .layer(axum::middleware::from_fn(conditional_cache_mw));

    // >4 MiB 响应不得退化为空 200 OK，必须完整透传正文
    let large = send(&app, "/sitemap.xml", None, None).await;
    assert_eq!(large.status(), StatusCode::OK);
    assert!(large.headers().get(header::ETAG).is_none());
    let body = axum::body::to_bytes(large.into_body(), payload_len + 64).await.expect("full body");
    assert_eq!(body.len(), payload_len);

    // 含 Set-Cookie 的响应必须跳过公共 ETag 并标记 private, no-store
    let cookie_resp = send(&app, "/set-cookie-page", None, None).await;
    assert_eq!(cookie_resp.status(), StatusCode::OK);
    assert!(cookie_resp.headers().get(header::ETAG).is_none());
    assert_eq!(
      cookie_resp.headers().get(header::CACHE_CONTROL).and_then(|v| v.to_str().ok()),
      Some("private, no-store")
    );
  }
}
