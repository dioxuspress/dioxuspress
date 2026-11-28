//! S2（风险 R4）：应用层 per-IP 限流中间件。
//!
//! gateway（Pingora + governor）是第一道防线，但它是**可选**部署组件；
//! app 裸跑公网时写接口（评论 / 话题 / 上传）与昂贵接口（搜索 / moderation）
//! 需要自带基础防护。本模块提供轻量 token-bucket 限流：
//!
//! - **作用范围**：仅 `/api/*`（server fn、auth、pay 回调都在该前缀下）。
//!   SSR 页面与静态资源不限流——页面加载会并发拉多个资源，限流误伤大。
//! - **分级**：`/api/auth/*`、`/api/pay/*` 与下单（`/api/orders/create`、
//!   `/api/orders/tip`）用更严的 sensitive 桶（防爆破 / 防回调滥用 / 防刷单）；
//!   其余 `/api/*` 用宽松的默认桶。
//! - **键**：默认从受信任反代头取真实客户端 IP（优先单跳覆写的 `x-real-ip`，其次
//!   `x-forwarded-for` 最右非受信跳，防客户端伪造最左 XFF）；设置
//!   `TRUST_PROXY_HEADERS=0` 可彻底禁用代理头（回退 `ConnectInfo<SocketAddr>` / `"global"`）。
//!
//! ## 运维开关（env）
//! - `RATE_LIMIT_DISABLED=1`：整体禁用。
//! - `TRUST_PROXY_HEADERS`：`0`/`false` 禁用代理头解析；默认 `1`（配合单跳反代/网关）。
//! - `TRUST_PROXY_HOPS`：外层可信反代跳数（默认 `1`，取 XFF 倒数第 N 个 IP）。
//! - `RATE_LIMIT_API_RPS` / `RATE_LIMIT_API_BURST`：默认桶（默认 20 rps / 60）。
//! - `RATE_LIMIT_SENSITIVE_RPS` / `RATE_LIMIT_SENSITIVE_BURST`：敏感桶
//!   （默认 5 rps / 15）。
//!
//! 实现为手写 token bucket（无新增依赖）；桶表有容量上限 + 过期剪枝 + 最老桶淘汰，
//! 防止恶意伪造海量 IP 撑爆内存或折叠正常用户到单一 `__overflow__` 桶。

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use axum::extract::{ConnectInfo, Request};
use axum::http::{HeaderValue, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};

/// 桶表条目数上限：超过即触发过期剪枝，仍满时淘汰最老 10% 条目（避免全局 `__overflow__` DoS）。
const MAX_TRACKED_KEYS: usize = 50_000;
/// 剪枝阈值：超过该时长无活动的桶视为过期。
const STALE_AFTER: Duration = Duration::from_secs(600);

/// 单 IP 的 token bucket。
struct Bucket {
  tokens: f64,
  last: Instant,
}

/// 一组按 key 分桶的限流器（纯逻辑，可注入时钟单测）。
pub struct TokenBucketMap {
  rps: f64,
  burst: f64,
  max_keys: usize,
  inner: Mutex<HashMap<String, Bucket>>,
}

impl TokenBucketMap {
  pub fn new(rps: f64, burst: f64) -> Self {
    Self::with_max_keys(rps, burst, MAX_TRACKED_KEYS)
  }

  fn with_max_keys(rps: f64, burst: f64, max_keys: usize) -> Self {
    // 防御非法配置：rps/burst 至少为 1，避免除零或永远拒绝。
    Self {
      rps: rps.max(0.01),
      burst: burst.max(1.0),
      max_keys: max_keys.max(16),
      inner: Mutex::new(HashMap::new()),
    }
  }

  /// 尝试取 1 个 token。`true` = 放行。
  pub fn try_acquire(&self, key: &str) -> bool {
    self.try_acquire_at(key, Instant::now())
  }

  /// 时钟可注入版本（单测用）。
  fn try_acquire_at(&self, key: &str, now: Instant) -> bool {
    // 锁中毒时直接取内层数据继续（限流器状态轻微失真优于整体拒绝服务）。
    let mut map = match self.inner.lock() {
      Ok(g) => g,
      Err(poisoned) => poisoned.into_inner(),
    };

    // 容量防御：先剪过期，仍满时淘汰最久未活跃的 10% 桶，保障新正常访客拥有独立桶。
    if !map.contains_key(key) && map.len() >= self.max_keys {
      map.retain(|_, b| now.duration_since(b.last) < STALE_AFTER);
      if map.len() >= self.max_keys {
        let evict_count = (self.max_keys / 10).max(1);
        let mut ages: Vec<(Instant, String)> =
          map.iter().map(|(k, b)| (b.last, k.clone())).collect();
        ages.sort_by_key(|(last, _)| *last);
        for (_, old_key) in ages.into_iter().take(evict_count) {
          map.remove(&old_key);
        }
      }
    }

    self.acquire_in(&mut map, key, now)
  }

  fn acquire_in(&self, map: &mut HashMap<String, Bucket>, key: &str, now: Instant) -> bool {
    let bucket = map.entry(key.to_string()).or_insert(Bucket { tokens: self.burst, last: now });
    let dt = now.duration_since(bucket.last).as_secs_f64();
    bucket.tokens = (bucket.tokens + dt * self.rps).min(self.burst);
    bucket.last = now;
    if bucket.tokens >= 1.0 {
      bucket.tokens -= 1.0;
      true
    } else {
      false
    }
  }
}

/// 两级限流器：默认 `/api/*` 桶 + 敏感（auth/pay）桶。
pub struct RateLimiters {
  api: TokenBucketMap,
  sensitive: TokenBucketMap,
}

fn read_env_f64(key: &str, default: f64) -> f64 {
  std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

impl RateLimiters {
  pub fn from_env() -> Self {
    Self {
      api: TokenBucketMap::new(
        read_env_f64("RATE_LIMIT_API_RPS", 20.0),
        read_env_f64("RATE_LIMIT_API_BURST", 60.0),
      ),
      sensitive: TokenBucketMap::new(
        read_env_f64("RATE_LIMIT_SENSITIVE_RPS", 5.0),
        read_env_f64("RATE_LIMIT_SENSITIVE_BURST", 15.0),
      ),
    }
  }

  pub fn check(&self, sensitive: bool, key: &str) -> bool {
    if sensitive {
      self.sensitive.try_acquire(key)
    } else {
      self.api.try_acquire(key)
    }
  }
}

/// 路径是否在限流范围内；返回 `Some(is_sensitive)`，`None` = 不限流。
pub fn classify_path(path: &str) -> Option<bool> {
  if !path.starts_with("/api/") {
    return None;
  }
  // 下单（H4）每次写一行订单并调用一次网关，与登录 / 回调同用严格桶。
  const SENSITIVE: [&str; 4] = ["/api/auth/", "/api/pay/", "/api/orders/create", "/api/orders/tip"];
  Some(SENSITIVE.iter().any(|p| path.starts_with(p)))
}

/// 代理头信任策略（[H2]）：
/// - `TRUST_PROXY_HEADERS=0|false|none`：彻底忽略代理头，只读 `ConnectInfo<SocketAddr>` 或 `"global"`；
/// - 默认（`TRUST_PROXY_HEADERS=1`）：优先读单跳反代覆写的 `X-Real-IP`，其次取
///   `X-Forwarded-For` 右侧倒数第 `TRUST_PROXY_HOPS`（默认 `1`）个 IP，避免客户端伪造最左 XFF。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProxyTrustConfig {
  pub trust_headers: bool,
  pub trusted_hops: usize,
}

impl ProxyTrustConfig {
  pub fn from_env() -> Self {
    let trust_headers = match std::env::var("TRUST_PROXY_HEADERS") {
      Ok(v) => !matches!(v.trim().to_ascii_lowercase().as_str(), "0" | "false" | "none" | "off"),
      Err(_) => true,
    };
    let trusted_hops = std::env::var("TRUST_PROXY_HOPS")
      .ok()
      .and_then(|v| v.trim().parse::<usize>().ok())
      .filter(|&n| n > 0)
      .unwrap_or(1);
    Self { trust_headers, trusted_hops }
  }
}

/// 从请求头 / 连接信息解析限流键（可注入策略，便于单测）。
pub fn client_key_with_config(req: &Request, cfg: ProxyTrustConfig) -> String {
  if cfg.trust_headers {
    if let Some(real) = req.headers().get("x-real-ip").and_then(|v| v.to_str().ok()) {
      let ip = real.trim();
      if !ip.is_empty() {
        return ip.to_string();
      }
    }
    if let Some(xff) = req.headers().get("x-forwarded-for").and_then(|v| v.to_str().ok()) {
      let hops: Vec<&str> = xff.split(',').map(str::trim).filter(|s| !s.is_empty()).collect();
      if !hops.is_empty() {
        // 标准反代（Nginx / Cloudflare / ALB）把真实客户端 IP 追加到 XFF 右侧；
        // 取倒数第 `trusted_hops` 跳（单跳反代即最右一项），防止客户端伪造最左项绕过。
        let idx = hops.len().saturating_sub(cfg.trusted_hops.max(1));
        return hops[idx].to_string();
      }
    }
  }
  if let Some(ConnectInfo(addr)) = req.extensions().get::<ConnectInfo<SocketAddr>>() {
    return addr.ip().to_string();
  }
  "global".to_string()
}

/// 从请求解析限流键（读取全局环境配置）。
pub fn client_key(req: &Request) -> String {
  static PROXY_CFG: std::sync::OnceLock<ProxyTrustConfig> = std::sync::OnceLock::new();
  client_key_with_config(req, *PROXY_CFG.get_or_init(ProxyTrustConfig::from_env))
}

/// 中间件本体。超限返回 `429 Too Many Requests` + `Retry-After: 1`。
pub async fn rate_limit_mw(req: Request, next: Next) -> Response {
  static STATE: std::sync::OnceLock<Option<RateLimiters>> = std::sync::OnceLock::new();
  let limiters = STATE.get_or_init(|| {
    if std::env::var("RATE_LIMIT_DISABLED").map(|v| v == "1").unwrap_or(false) {
      tracing::warn!("rate_limit: disabled via RATE_LIMIT_DISABLED=1");
      None
    } else {
      Some(RateLimiters::from_env())
    }
  });

  let Some(limiters) = limiters else {
    return next.run(req).await;
  };
  let Some(sensitive) = classify_path(req.uri().path()) else {
    return next.run(req).await;
  };

  let key = client_key(&req);
  if !limiters.check(sensitive, &key) {
    tracing::warn!(
      target: "rate_limit",
      key = %key,
      path = %req.uri().path(),
      sensitive,
      "rate limit exceeded"
    );
    let mut resp = (StatusCode::TOO_MANY_REQUESTS, "请求过于频繁，请稍后重试").into_response();
    resp.headers_mut().insert("retry-after", HeaderValue::from_static("1"));
    return resp;
  }
  next.run(req).await
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn burst_allows_then_blocks() {
    let map = TokenBucketMap::new(1.0, 3.0);
    let now = Instant::now();
    assert!(map.try_acquire_at("ip1", now));
    assert!(map.try_acquire_at("ip1", now));
    assert!(map.try_acquire_at("ip1", now));
    assert!(!map.try_acquire_at("ip1", now), "burst 用尽后同刻应拒绝");
  }

  #[test]
  fn refill_restores_tokens() {
    let map = TokenBucketMap::new(2.0, 2.0);
    let t0 = Instant::now();
    assert!(map.try_acquire_at("ip1", t0));
    assert!(map.try_acquire_at("ip1", t0));
    assert!(!map.try_acquire_at("ip1", t0));
    // 1 秒后按 2 rps 回填 2 个 token
    let t1 = t0 + Duration::from_secs(1);
    assert!(map.try_acquire_at("ip1", t1));
    assert!(map.try_acquire_at("ip1", t1));
    assert!(!map.try_acquire_at("ip1", t1));
  }

  #[test]
  fn keys_are_isolated() {
    let map = TokenBucketMap::new(1.0, 1.0);
    let now = Instant::now();
    assert!(map.try_acquire_at("a", now));
    assert!(!map.try_acquire_at("a", now));
    assert!(map.try_acquire_at("b", now), "不同 key 桶互相独立");
  }

  #[test]
  fn classify_scopes_only_api() {
    assert_eq!(classify_path("/blog/hello"), None);
    assert_eq!(classify_path("/assets/tailwind.css"), None);
    assert_eq!(classify_path("/api/comments/list"), Some(false));
    assert_eq!(classify_path("/api/auth/login/github"), Some(true));
    assert_eq!(classify_path("/api/pay/alipay/notify"), Some(true));
    // H4：下单每次写一行订单并调用一次网关。
    assert_eq!(classify_path("/api/orders/create"), Some(true));
    assert_eq!(classify_path("/api/orders/tip"), Some(true));
    assert_eq!(classify_path("/api/orders/mine"), Some(false));
  }

  #[test]
  fn client_key_prefers_real_ip_and_rightmost_xff_to_prevent_spoofing() {
    let cfg = ProxyTrustConfig { trust_headers: true, trusted_hops: 1 };
    let req_with_real = Request::builder()
      .uri("/api/x")
      .header("x-forwarded-for", "1.2.3.4, 203.0.113.7")
      .header("x-real-ip", "198.51.100.2")
      .body(axum::body::Body::empty())
      .expect("test request");
    assert_eq!(client_key_with_config(&req_with_real, cfg), "198.51.100.2");

    // 仅 XFF 时：客户端伪造最左 1.2.3.4，反代在右侧追加真实 IP 203.0.113.7
    let req_xff_only = Request::builder()
      .uri("/api/x")
      .header("x-forwarded-for", "1.2.3.4, 203.0.113.7")
      .body(axum::body::Body::empty())
      .expect("test request");
    assert_eq!(client_key_with_config(&req_xff_only, cfg), "203.0.113.7");

    // 禁用代理头时忽略 XFF/X-Real-IP
    let no_proxy = ProxyTrustConfig { trust_headers: false, trusted_hops: 1 };
    assert_eq!(client_key_with_config(&req_with_real, no_proxy), "global");
  }

  #[test]
  fn client_key_falls_back_to_connect_info_then_global() {
    let no_proxy = ProxyTrustConfig { trust_headers: false, trusted_hops: 1 };
    let mut req = Request::builder()
      .uri("/api/x")
      .header("x-forwarded-for", "1.2.3.4")
      .body(axum::body::Body::empty())
      .expect("test request");
    req.extensions_mut().insert(ConnectInfo("203.0.113.9:43210".parse::<SocketAddr>().unwrap()));
    assert_eq!(client_key_with_config(&req, no_proxy), "203.0.113.9");

    let bare =
      Request::builder().uri("/api/x").body(axum::body::Body::empty()).expect("test request");
    assert_eq!(client_key_with_config(&bare, no_proxy), "global");
  }

  #[test]
  fn full_table_evicts_oldest_instead_of_collapsing_into_global_overflow_bucket() {
    let map = TokenBucketMap::with_max_keys(1.0, 1.0, 20);
    let t0 = Instant::now();
    // 攻击者用 20 个不同伪造 IP 填满桶表
    for i in 0..20 {
      assert!(map.try_acquire_at(&format!("attacker-{i}"), t0 + Duration::from_millis(i)));
    }
    // 随后到来的多个正常访客不应被折叠到同一个 __overflow__ 桶导致连坐 429
    let t1 = t0 + Duration::from_secs(1);
    assert!(map.try_acquire_at("legit-a", t1));
    assert!(map.try_acquire_at("legit-b", t1));
    assert!(map.try_acquire_at("legit-c", t1));
    // 桶表规模依然严格受控
    let guard = map.inner.lock().unwrap();
    assert!(guard.len() <= 20);
  }

  #[test]
  fn invalid_config_clamped() {
    // rps/burst 非法值不 panic、不永久拒绝
    let map = TokenBucketMap::new(0.0, 0.0);
    assert!(map.try_acquire_at("x", Instant::now()), "clamp 后至少允许 1 个 burst");
  }
}
