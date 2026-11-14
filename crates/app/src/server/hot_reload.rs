//! E5.3：挂载内容目录（`CONTENT_DIR`）零停机增量热刷新。
//!
//! 消除纯内容发布时 `docker restart dioxuspress-app` 带来的连接闪断与冷启动重扫：
//! 1. **即时信号触发（Unix `SIGHUP`）**：执行 `docker kill --signal=HUP dioxuspress-app`
//!    或 `kill -HUP <pid>` 时，进程不断开任何现有 TCP 连接，毫秒级原子刷新 `SiteConfig`、
//!    `ModuleEngine`、全部 `DirListingCache`、`ETag` 内容纪元，并在后台线程池增量更新 Tantivy 检索索引。
//! 2. **节流目录指纹兜底（`CONTENT_DIR`）**：当挂载外部内容目录时，默认每 5 秒（可通过
//!    `CONTENT_WATCH_INTERVAL_SECS` 调整，`0` 关闭）轻量检查 `.git/HEAD`、`site.json` 与顶级子目录
//!    mtime；即便手工 `git pull` / `rsync` 未发信号，也能在数秒内自动热同步。

use std::path::Path;
use std::time::{Duration, Instant, UNIX_EPOCH};

use app_core::i18n::Language;
use app_core::settings::SiteConfig;

/// 同步清空全部内存配置与目录列表缓存，并推进 `ETag` 纪元。
/// 纯内存原子/短锁操作（微秒级），可在任何异步上下文直接调用。
pub fn invalidate_all_in_memory_caches() -> u64 {
  SiteConfig::invalidate_cache();
  let cfg = SiteConfig::active();
  widgets::update_site_seo_meta(
    cfg.site_name_or_default(),
    cfg.og_image_or_default(),
    cfg.twitter_handle_or_none(),
    cfg.default_language_enum() == Language::En,
  );
  app_core::engines::module::invalidate_default_module_engine();
  module_docs::server::invalidate_doc_caches();
  module_blog::server::invalidate_blog_cache();
  module_ai::atlas::invalidate_models_cache();
  board::server::invalidate_board_cache();
  crate::server::conditional_cache::bump_content_epoch()
}

/// 执行一次完整的零停机内容热刷新：先原子失效全部内存/ETag 缓存，再增量同步 Tantivy 索引。
pub async fn reload_content_now(reason: &'static str) -> u64 {
  let started = Instant::now();
  let epoch = invalidate_all_in_memory_caches();
  let (search_upserts, search_deletes) = match module_search::engine::reindex_incremental().await {
    Ok(r) => (r.upserts, r.deletes),
    Err(err) => {
      tracing::warn!(reason, epoch, error = %err, "hot_reload: incremental search reindex failed");
      (0, 0)
    }
  };
  let elapsed_ms = started.elapsed().as_millis().min(u64::MAX as u128) as u64;
  tracing::info!(
    reason,
    epoch,
    search_upserts,
    search_deletes,
    elapsed_ms,
    "hot_reload: content and search index refreshed with zero downtime"
  );
  epoch
}

fn mtime_ns(path: &Path) -> u128 {
  std::fs::metadata(path)
    .ok()
    .and_then(|m| m.modified().ok())
    .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
    .map(|d| d.as_nanos())
    .unwrap_or(0)
}

fn fnv1a(bytes: &[u8]) -> u64 {
  let mut h: u64 = 0xcbf29ce484222325;
  for &b in bytes {
    h ^= u64::from(b);
    h = h.wrapping_mul(0x100000001b3);
  }
  h
}

/// 计算挂载内容根目录的轻量指纹：
/// 覆盖 `.git/HEAD`（及指向的 ref 文件，捕获 `git push` / `git pull`）、`site.json` 与顶级内容目录 mtime。
pub fn compute_content_root_stamp(root: &Path) -> u128 {
  let mut git_sig: u64 = 0;
  let git_head = root.join(".git/HEAD");
  if let Ok(head_raw) = std::fs::read_to_string(&git_head) {
    let trimmed = head_raw.trim();
    git_sig = fnv1a(trimmed.as_bytes());
    if let Some(rel_ref) = trimmed.strip_prefix("ref: ") {
      let ref_path = root.join(".git").join(rel_ref.trim());
      if let Ok(commit_sha) = std::fs::read_to_string(&ref_path) {
        git_sig ^= fnv1a(commit_sha.trim().as_bytes()).rotate_left(13);
      } else {
        git_sig ^= (mtime_ns(&ref_path) as u64).rotate_left(17);
      }
    }
  }

  let mut mtimes: u128 = mtime_ns(&root.join("site.json"));
  for sub in ["docs", "posts", "courses", "cases", "models", "podcasts", "topics", "images"] {
    mtimes = mtimes.wrapping_mul(1_000_003) ^ mtime_ns(&root.join(sub));
  }
  (u128::from(git_sig) << 64) ^ mtimes
}

/// 启动后台零停机热刷新守护协程（Unix `SIGHUP` + 可选 `CONTENT_DIR` 节流轮询）。
pub fn spawn_hot_reload_tasks() {
  #[cfg(unix)]
  tokio::spawn(async move {
    use tokio::signal::unix::{signal, SignalKind};
    let mut hup = match signal(SignalKind::hangup()) {
      Ok(sig) => sig,
      Err(err) => {
        tracing::warn!(error = %err, "hot_reload: failed to register SIGHUP listener");
        return;
      }
    };
    tracing::info!("startup: SIGHUP zero-downtime content reload listener armed");
    while hup.recv().await.is_some() {
      let _ = reload_content_now("sighup").await;
    }
  });

  let watch_secs = std::env::var("CONTENT_WATCH_INTERVAL_SECS")
    .ok()
    .and_then(|v| v.trim().parse::<u64>().ok())
    .unwrap_or(5);

  if watch_secs > 0 {
    if let Some(root) = app_core::utils::content_root() {
      tokio::spawn(async move {
        let mut last_stamp = compute_content_root_stamp(&root);
        let mut ticker = tokio::time::interval(Duration::from_secs(watch_secs.max(1)));
        ticker.tick().await;
        loop {
          ticker.tick().await;
          let next_stamp = compute_content_root_stamp(&root);
          if next_stamp != last_stamp {
            last_stamp = next_stamp;
            let _ = reload_content_now("content_watch").await;
          }
        }
      });
    }
  }
}

#[cfg(test)]
mod tests {
  use std::sync::atomic::{AtomicUsize, Ordering};
  use std::sync::Arc;

  use axum::body::Body;
  use axum::http::{header, Request, StatusCode};
  use axum::routing::get;
  use axum::Router;
  use tower::ServiceExt;

  use super::*;

  #[test]
  fn content_root_stamp_detects_git_ref_and_site_json_updates() {
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::create_dir_all(tmp.path().join(".git/refs/heads")).expect("git dirs");
    std::fs::create_dir_all(tmp.path().join("docs")).expect("docs dir");
    std::fs::write(tmp.path().join(".git/HEAD"), "ref: refs/heads/main\n").expect("HEAD");
    std::fs::write(tmp.path().join(".git/refs/heads/main"), "1111111111111111\n").expect("ref1");
    std::fs::write(tmp.path().join("site.json"), r#"{"site_name":"V1"}"#).expect("site1");

    let s1 = compute_content_root_stamp(tmp.path());

    // Simulate git push updateInstead advancing branch SHA even within the same filesystem second
    std::fs::write(tmp.path().join(".git/refs/heads/main"), "2222222222222222\n").expect("ref2");
    let s2 = compute_content_root_stamp(tmp.path());
    assert_ne!(s1, s2, "advancing git commit SHA must immediately change content root stamp");
  }

  #[tokio::test]
  async fn concurrent_requests_experience_zero_errors_during_hot_reload_invalidation() {
    let version = Arc::new(AtomicUsize::new(1));
    let v_route = version.clone();

    let app = Router::new()
      .route(
        "/docs/guide/quickstart",
        get(move || {
          let v = v_route.load(Ordering::Acquire);
          async move { format!("<html>doc-v{v}</html>") }
        }),
      )
      .layer(axum::middleware::from_fn(crate::server::conditional_cache::conditional_cache_mw));

    // Warm up v1 cache
    let r1 = app
      .clone()
      .oneshot(Request::builder().uri("/docs/guide/quickstart").body(Body::empty()).expect("req"))
      .await
      .expect("resp");
    assert_eq!(r1.status(), StatusCode::OK);
    let etag_v1 = r1.headers().get(header::ETAG).expect("etag").to_str().expect("str").to_string();

    // Hammer the router concurrently across 8 tasks while flipping content to v2 + hot-reloading mid-flight
    let mut handles = Vec::new();
    for worker_id in 0..8usize {
      let router = app.clone();
      let etag = etag_v1.clone();
      handles.push(tokio::spawn(async move {
        let mut non_ok = 0usize;
        for iter in 0..60usize {
          let mut req = Request::builder().uri("/docs/guide/quickstart");
          if (worker_id + iter) % 2 == 0 {
            req = req.header(header::IF_NONE_MATCH, &etag);
          }
          let resp =
            router.clone().oneshot(req.body(Body::empty()).expect("req")).await.expect("ok");
          if resp.status() != StatusCode::OK && resp.status() != StatusCode::NOT_MODIFIED {
            non_ok += 1;
          }
          tokio::task::yield_now().await;
        }
        non_ok
      }));
      if worker_id == 3 {
        version.store(2, Ordering::Release);
        invalidate_all_in_memory_caches();
      }
    }

    let mut total_errors = 0usize;
    for h in handles {
      total_errors += h.await.expect("join");
    }
    assert_eq!(total_errors, 0, "zero non-200/304 responses allowed during hot reload");

    // Verify stale ETag no longer returns 304 and fresh v2 content is served immediately
    let post_reload = app
      .oneshot(
        Request::builder()
          .uri("/docs/guide/quickstart")
          .header(header::IF_NONE_MATCH, &etag_v1)
          .body(Body::empty())
          .expect("req"),
      )
      .await
      .expect("resp");
    assert_eq!(post_reload.status(), StatusCode::OK);
    let body = axum::body::to_bytes(post_reload.into_body(), 4096).await.expect("bytes");
    assert_eq!(&body[..], b"<html>doc-v2</html>");
  }
}
