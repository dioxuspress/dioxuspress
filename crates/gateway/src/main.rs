//! Pingora-based TLS terminator + reverse proxy for the rustineverything.app
//! deployment. Listens on :443 (TLS) and :80 (HTTP→HTTPS 301), proxies all
//! traffic to the app on `127.0.0.1:8080`.
//!
//! Run:
//!   TLS_CERT_PATH=/data/cert/rustineverything.app.cert \
//!   TLS_KEY_PATH=/data/cert/rustineverything.app.key \
//!     ./rie-gateway
//!
//! 证书只在启动时读：换证书后重启网关（`scripts/cert_refresh.sh` 默认
//! `docker restart rie-gateway`）。不要发 SIGHUP：pingora 不处理它，进程直接退出。
//! SIGTERM 是平滑停机，宽限期见 `main` 里的 `ServerConf`。
//!
//! ## Phase 8.3 安全 / 性能强化
//! - 注入 HSTS；应用未下发时补 X-Content-Type-Options / X-Frame-Options / Referrer-Policy。
//!   CSP 只由应用设置，gateway 原样转发（SEC-10）
//! - `X-Forwarded-For` 改 `insert`（覆盖客户端伪造），同时 strip RFC 7239 `Forwarded`
//! - 限流只在 app 里做（`crates/app/src/server/rate_limit.rs`，按 XFF 分桶、只限 `/api/*`）。
//!   gateway 原先也按 IP 限流，但把静态资源算在内：一次冷加载约 57 个请求，
//!   读配额 60/min 刷新一次就会 429，`/api/auth/me` 还被算进 10/min 的写配额

use std::collections::HashMap;
use std::time::Duration;

use async_trait::async_trait;
use pingora_core::apps::HttpServerOptions;
use pingora_core::listeners::TlsAccept;
use pingora_core::protocols::tls::TlsRef;
use pingora_core::server::configuration::{Opt, ServerConf};
use pingora_core::server::Server;
use pingora_core::tls::ext::{ssl_add_chain_cert, ssl_use_certificate, ssl_use_private_key};
use pingora_core::tls::pkey::{PKey, Private};
use pingora_core::tls::ssl::NameType;
use pingora_core::tls::x509::X509;
use pingora_core::upstreams::peer::HttpPeer;
use pingora_core::Result;
use pingora_http::{RequestHeader, ResponseHeader};
use pingora_proxy::{http_proxy_service, ProxyHttp, Session};

/// 反代上游 = 本机 app 容器；HTTP/1.1 + keep-alive。
const UPSTREAM_ADDR: &str = "127.0.0.1:8080";
const UPSTREAM_SNI: &str = ""; // 上游 plain HTTP，不需要 SNI

const HTTPS_BIND: &str = "0.0.0.0:443";
const HTTP_BIND: &str = "0.0.0.0:80";

const HSTS_VALUE: &str = "max-age=31536000; includeSubDomains";

/// 下游连接两个请求之间最多空闲多久（秒）：HTTP/1 keep-alive 与没有在途流的 h2 连接。
const DOWNSTREAM_IDLE_SECS: u64 = 60;

/// pingora 读完请求头后把 HTTP/1.1 keep-alive 设成无限（`Some(0)`），
/// 下一个请求之前的那次读就没有超时；换回有限值。`None`（客户端要求 close）不动。
fn cap_keepalive(session: &mut Session) {
  if session.get_keepalive() == Some(0) {
    session.set_keepalive(Some(DOWNSTREAM_IDLE_SECS));
  }
}

struct AppGateway {
  default_upstream: String,
  host_upstreams: HashMap<String, String>,
  host_redirects: HashMap<String, String>,
}

impl AppGateway {
  fn from_env() -> Self {
    let default_upstream = std::env::var("UPSTREAM_ADDR")
      .ok()
      .map(|s| s.trim().to_string())
      .filter(|s| !s.is_empty())
      .unwrap_or_else(|| UPSTREAM_ADDR.to_string());
    let host_upstreams =
      std::env::var("HOST_UPSTREAMS").ok().map(|s| parse_host_upstreams(&s)).unwrap_or_default();
    let host_redirects =
      std::env::var("HOST_REDIRECTS").ok().map(|s| parse_host_redirects(&s)).unwrap_or_default();
    Self { default_upstream, host_upstreams, host_redirects }
  }

  fn select_upstream(&self, req: &RequestHeader) -> &str {
    if !self.host_upstreams.is_empty() {
      if let Some(host) = extract_request_host(req) {
        if let Some(addr) = self.host_upstreams.get(&host) {
          return addr.as_str();
        }
      }
    }
    &self.default_upstream
  }

  fn select_redirect_location(&self, req: &RequestHeader) -> Option<String> {
    if self.host_redirects.is_empty() {
      return None;
    }
    let host = extract_request_host(req)?;
    let target_base = self.host_redirects.get(&host)?;
    let path = req.uri.path_and_query().map(|p| p.as_str()).unwrap_or("/");
    if path.starts_with('/') {
      Some(format!("{target_base}{path}"))
    } else {
      Some(format!("{target_base}/{path}"))
    }
  }
}

/// 解析 `HOST_REDIRECTS` 环境变量（格式：`old.example.com=https://new.dev,www.new.dev=https://new.dev`）。
fn parse_host_redirects(raw: &str) -> HashMap<String, String> {
  let mut map = HashMap::new();
  for pair in raw.split(',') {
    let Some((host, target)) = pair.trim().split_once('=') else {
      continue;
    };
    let Some(host) = normalize_host(host) else {
      continue;
    };
    let target = target.trim().trim_end_matches('/');
    if target.starts_with("https://") || target.starts_with("http://") {
      map.insert(host, target.to_string());
    }
  }
  map
}

struct CertBundle {
  leaf: X509,
  intermediates: Vec<X509>,
  key: PKey<Private>,
}

impl CertBundle {
  fn from_pem(cert_pem: &[u8], key_pem: &[u8]) -> std::result::Result<Self, String> {
    let mut chain =
      X509::stack_from_pem(cert_pem).map_err(|e| format!("parse cert chain PEM failed: {e}"))?;
    if chain.is_empty() {
      return Err("certificate PEM contains no X509 certificates".to_string());
    }
    let leaf = chain.remove(0);
    let key = PKey::private_key_from_pem(key_pem)
      .map_err(|e| format!("parse private key PEM failed: {e}"))?;
    Ok(Self { leaf, intermediates: chain, key })
  }

  fn load(cert_path: &str, key_path: &str) -> std::result::Result<Self, String> {
    let cert_pem =
      std::fs::read(cert_path).map_err(|e| format!("read cert {cert_path} failed: {e}"))?;
    let key_pem =
      std::fs::read(key_path).map_err(|e| format!("read key {key_path} failed: {e}"))?;
    Self::from_pem(&cert_pem, &key_pem)
  }
}

/// 解析 `SNI_CERTS` 环境变量（格式：`domain1=/certs/d1/fullchain.pem:/certs/d1/privkey.pem,...`）。
fn parse_sni_cert_specs(raw: &str) -> Vec<(String, String, String)> {
  let mut specs = Vec::new();
  for entry in raw.split(',') {
    let Some((domain, paths)) = entry.trim().split_once('=') else {
      continue;
    };
    let Some(domain) = normalize_host(domain.trim_start_matches("*.")) else {
      continue;
    };
    let Some((cert_path, key_path)) = paths.trim().split_once(':') else {
      continue;
    };
    let cert_path = cert_path.trim();
    let key_path = key_path.trim();
    if !cert_path.is_empty() && !key_path.is_empty() {
      specs.push((domain, cert_path.to_string(), key_path.to_string()));
    }
  }
  // 更具体的子域名优先匹配（如 api.dioxuspress.dev 优先于 dioxuspress.dev）
  specs.sort_by_key(|a| std::cmp::Reverse(a.0.len()));
  specs
}

fn host_matches_domain(sni_host: &str, domain: &str) -> bool {
  sni_host == domain || sni_host.strip_suffix(domain).is_some_and(|prefix| prefix.ends_with('.'))
}

struct SniCertResolver {
  default_bundle: CertBundle,
  routes: Vec<(String, CertBundle)>,
}

impl SniCertResolver {
  fn load(
    default_cert: &str,
    default_key: &str,
    sni_raw: &str,
  ) -> std::result::Result<Option<Self>, String> {
    let specs = parse_sni_cert_specs(sni_raw);
    if specs.is_empty() {
      return Ok(None);
    }
    let default_bundle = CertBundle::load(default_cert, default_key)?;
    let mut routes = Vec::with_capacity(specs.len());
    for (domain, cert_path, key_path) in specs {
      let bundle = CertBundle::load(&cert_path, &key_path)?;
      routes.push((domain, bundle));
    }
    Ok(Some(Self { default_bundle, routes }))
  }

  fn select_bundle(&self, sni_raw: Option<&str>) -> &CertBundle {
    if let Some(host) = sni_raw.and_then(normalize_host) {
      for (domain, bundle) in &self.routes {
        if host_matches_domain(&host, domain) {
          return bundle;
        }
      }
    }
    &self.default_bundle
  }
}

#[async_trait]
impl TlsAccept for SniCertResolver {
  async fn certificate_callback(&self, ssl: &mut TlsRef) {
    let bundle = self.select_bundle(ssl.servername(NameType::HOST_NAME));
    if let Err(e) = ssl_use_certificate(ssl, &bundle.leaf) {
      log::error!("SNI ssl_use_certificate failed: {e}");
      return;
    }
    for inter in &bundle.intermediates {
      if let Err(e) = ssl_add_chain_cert(ssl, inter) {
        log::error!("SNI ssl_add_chain_cert failed: {e}");
        return;
      }
    }
    if let Err(e) = ssl_use_private_key(ssl, &bundle.key) {
      log::error!("SNI ssl_use_private_key failed: {e}");
    }
  }
}

/// 解析 `HOST_UPSTREAMS` 环境变量（格式：`host1=127.0.0.1:8081,host2=127.0.0.1:8082`）。
fn parse_host_upstreams(raw: &str) -> HashMap<String, String> {
  let mut map = HashMap::new();
  for pair in raw.split(',') {
    let Some((host, addr)) = pair.trim().split_once('=') else {
      continue;
    };
    let Some(host) = normalize_host(host) else {
      continue;
    };
    let addr = addr.trim();
    if !addr.is_empty() {
      map.insert(host, addr.to_string());
    }
  }
  map
}

/// 归一化域名：去端口、去末尾 `.`、转小写 ASCII。
fn normalize_host(raw: &str) -> Option<String> {
  let trimmed = raw.trim();
  if trimmed.is_empty() {
    return None;
  }
  let host_only = if trimmed.starts_with('[') {
    let end = trimmed.find(']')?;
    &trimmed[..=end]
  } else {
    trimmed.split(':').next().unwrap_or("")
  };
  let normalized = host_only.trim().trim_end_matches('.').to_ascii_lowercase();
  if normalized.is_empty() {
    None
  } else {
    Some(normalized)
  }
}

/// 提取请求域名：优先读 HTTP/2 `:authority`（`req.uri.host()`），回退 HTTP/1.1 `Host` 头。
fn extract_request_host(req: &RequestHeader) -> Option<String> {
  if let Some(uri_host) = req.uri.host() {
    if let Some(h) = normalize_host(uri_host) {
      return Some(h);
    }
  }
  let header_host = req.headers.get("host").and_then(|v| v.to_str().ok())?;
  normalize_host(header_host)
}

#[async_trait]
impl ProxyHttp for AppGateway {
  type CTX = ();
  fn new_ctx(&self) -> Self::CTX {}

  async fn early_request_filter(&self, session: &mut Session, _ctx: &mut ()) -> Result<()> {
    cap_keepalive(session);
    Ok(())
  }

  /// 域名级 301 重定向（如将旧演示二级域名或 `www.` 统一归并到主域名）。
  async fn request_filter(&self, session: &mut Session, _ctx: &mut ()) -> Result<bool> {
    if let Some(location) = self.select_redirect_location(session.req_header()) {
      let mut resp = ResponseHeader::build(301, None)
        .map_err(|_| pingora_core::Error::new_str("failed to build 301 response header"))?;
      resp.insert_header("Location", location).ok();
      resp.insert_header("Strict-Transport-Security", HSTS_VALUE).ok();
      resp.insert_header("Server", "rie-gateway").ok();
      resp.insert_header("Content-Length", "0").ok();
      session.write_response_header_ref(&resp, true).await.ok();
      return Ok(true);
    }
    Ok(false)
  }

  /// 选定上游：优先匹配 `HOST_UPSTREAMS` 子域名路由，否则走默认 `default_upstream`。
  async fn upstream_peer(&self, session: &mut Session, _ctx: &mut ()) -> Result<Box<HttpPeer>> {
    let addr = self.select_upstream(session.req_header());
    Ok(Box::new(HttpPeer::new(addr, false, UPSTREAM_SNI.into())))
  }

  /// 上游请求改写：补 X-Forwarded-* 让 app 端 cookie/日志看到真实客户端 + scheme。
  ///
  /// Phase 8.3 修正：
  /// - `X-Forwarded-For` 由 `append` 改 `insert`：客户端伪造的同名头会被覆盖而非保留，
  ///   避免日志 / app 限流按伪造 IP 分桶
  /// - 同时移除 RFC 7239 `Forwarded`（不同标准的等价头，攻击者可用来绕过 XFF 校验）
  async fn upstream_request_filter(
    &self,
    session: &mut Session,
    req: &mut RequestHeader,
    _ctx: &mut (),
  ) -> Result<()> {
    req.insert_header("X-Forwarded-Proto", "https").ok();
    if !req.headers.contains_key("host") {
      if let Some(authority) = req.uri.authority().map(|a| a.as_str().to_string()) {
        req.insert_header("Host", authority).ok();
      }
    }
    // 清掉可能的伪造 Forwarded（RFC 7239 变体）
    req.remove_header("Forwarded");

    if let Some(ip) = session.client_addr().and_then(forwarded_ip) {
      req.insert_header("X-Real-IP", ip.clone()).ok();
      // insert 覆盖客户端送上来的伪造 XFF；如未来真有可信外层代理，
      // 在该 filter 里改回 append + 白名单校验来源 IP
      req.insert_header("X-Forwarded-For", ip).ok();
    } else {
      // 拿不到客户端地址也要清掉伪造 XFF
      req.remove_header("X-Forwarded-For");
    }
    coalesce_cookie_headers(req);
    Ok(())
  }

  /// 响应改写：补安全头，见 [`apply_security_headers`]。
  async fn response_filter(
    &self,
    _session: &mut Session,
    upstream_response: &mut ResponseHeader,
    _ctx: &mut (),
  ) -> Result<()> {
    apply_security_headers(upstream_response);
    Ok(())
  }
}

/// 写给 app 的客户端地址只要 IP：带端口时每条 TCP 连接都不一样，app 按它分桶限流就等于没限。
fn forwarded_ip(addr: &pingora_core::protocols::l4::socket::SocketAddr) -> Option<String> {
  Some(addr.as_inet()?.ip().to_string())
}

/// RFC 9113 §8.2.3：HTTP/2 允许浏览器把多个 cookie 拆成多条 `cookie` 头（提升 HPACK 压缩率）；
/// 转为 HTTP/1.1 发往上游前必须按 `; ` 合并为单条 `Cookie` 头，否则上游 `headers.get("cookie")`
/// 只读到第一条（如 `site_lang` / `site_theme`），丢掉后面的 `session` cookie。
fn coalesce_cookie_headers(req: &mut RequestHeader) {
  let mut iter = req.headers.get_all("cookie").into_iter();
  let Some(first) = iter.next() else { return };
  let Some(second) = iter.next() else { return };
  let mut joined = String::new();
  for v in [first, second].into_iter().chain(iter) {
    if let Ok(s) = v.to_str() {
      let s = s.trim();
      if !s.is_empty() {
        if !joined.is_empty() {
          joined.push_str("; ");
        }
        joined.push_str(s);
      }
    }
  }
  if !joined.is_empty() {
    req.insert_header("Cookie", joined).ok();
  }
}

/// 安全响应头：
/// - HSTS 与 `Server` 由 gateway 负责（TLS 在这里终止），总是覆盖。
/// - 通用头（nosniff / X-Frame-Options / Referrer-Policy）应用也会下发；
///   只在缺失时补，保留应用按路由给出的值。
/// - **CSP 不碰**（SEC-10）：唯一定义在 `crates/app/src/server/security.rs`，
///   gateway 覆盖会丢掉 `'wasm-unsafe-eval'` 与内联启动脚本许可，hydration 全挂。
fn apply_security_headers(resp: &mut ResponseHeader) {
  resp.insert_header("Strict-Transport-Security", HSTS_VALUE).ok();
  // Server header 减少版本指纹
  resp.insert_header("Server", "rie-gateway").ok();
  for (name, value) in [
    ("X-Content-Type-Options", "nosniff"),
    ("X-Frame-Options", "DENY"),
    ("Referrer-Policy", "strict-origin-when-cross-origin"),
  ] {
    if !resp.headers.contains_key(name) {
      resp.insert_header(name, value).ok();
    }
  }
}

/// 80 端口专用：所有请求 301 到同 host 的 https 版本。
struct HttpToHttps;

#[async_trait]
impl ProxyHttp for HttpToHttps {
  type CTX = ();
  fn new_ctx(&self) -> Self::CTX {}

  async fn request_filter(&self, session: &mut Session, _ctx: &mut ()) -> Result<bool> {
    cap_keepalive(session);
    let req = session.req_header();
    let host = req.headers.get("host").and_then(|v| v.to_str().ok()).unwrap_or("");
    let path = req.uri.path_and_query().map(|p| p.as_str()).unwrap_or("/");
    let location = format!("https://{}{}", host, path);

    let mut resp = ResponseHeader::build(301, None).unwrap();
    resp.append_header("Location", location).ok();
    resp.append_header("Content-Length", "0").ok();
    // end_of_stream=true：301 无 body，回写头即结束
    session.write_response_header_ref(&resp, true).await.ok();
    Ok(true) // short-circuit；不再走 upstream_peer
  }

  async fn upstream_peer(&self, _session: &mut Session, _ctx: &mut ()) -> Result<Box<HttpPeer>> {
    // 永远走不到这里：request_filter 已 Ok(true) 短路。
    Err(pingora_core::Error::new_str("unreachable: redirect short-circuit returned"))
  }
}

fn main() {
  env_logger::init();

  // 默认宽限期 300 秒，docker 10 秒后就 SIGKILL；在那之前自己退出
  let conf = ServerConf {
    grace_period_seconds: Some(5),
    graceful_shutdown_timeout_seconds: Some(1),
    ..Default::default()
  };
  let mut server = Server::new_with_opt_and_conf(None::<Opt>, conf);
  server.bootstrap();

  // 443: TLS 终止 + 反代到 app（支持 SNI 多域名证书、HOST_REDIRECTS 301 与 HOST_UPSTREAMS 路由）
  let gateway = AppGateway::from_env();
  let default_upstream_log = gateway.default_upstream.clone();
  let host_routes_log = gateway.host_upstreams.len();
  let host_redirects_log = gateway.host_redirects.len();
  let mut proxy = http_proxy_service(&server.configuration, gateway);
  let Ok(cert) = std::env::var("TLS_CERT_PATH") else {
    eprintln!("rie-gateway: TLS_CERT_PATH 未配置（指向默认 fullchain.pem 路径）");
    std::process::exit(1);
  };
  let Ok(key) = std::env::var("TLS_KEY_PATH") else {
    eprintln!("rie-gateway: TLS_KEY_PATH 未配置（指向默认 privkey.pem 路径）");
    std::process::exit(1);
  };
  let sni_raw = std::env::var("SNI_CERTS").unwrap_or_default();
  let sni_resolver = match SniCertResolver::load(&cert, &key, &sni_raw) {
    Ok(r) => r,
    Err(e) => {
      eprintln!("rie-gateway: 加载 SNI_CERTS 失败：{e}");
      std::process::exit(1);
    }
  };
  let sni_certs_log = sni_resolver.as_ref().map_or(0, |r| r.routes.len());
  let tls_res = match sni_resolver {
    Some(resolver) => pingora_core::listeners::tls::TlsSettings::with_callbacks(Box::new(resolver)),
    None => pingora_core::listeners::tls::TlsSettings::intermediate(&cert, &key),
  };
  let mut tls = match tls_res {
    Ok(t) => t,
    Err(e) => {
      eprintln!("rie-gateway: 初始化 TLS 失败：{e}");
      std::process::exit(1);
    }
  };
  tls.enable_h2();
  proxy.add_tls_with_settings(HTTPS_BIND, None, tls);
  let mut opts = HttpServerOptions::default();
  opts.h2_idle_timeout = Some(Duration::from_secs(DOWNSTREAM_IDLE_SECS));
  let Some(app) = proxy.app_logic_mut() else {
    unreachable!("http_proxy_service always holds its HttpProxy app");
  };
  app.server_options = Some(opts);
  server.add_service(proxy);

  // 80: HTTP→HTTPS 301
  let mut redirect = http_proxy_service(&server.configuration, HttpToHttps);
  redirect.add_tcp(HTTP_BIND);
  server.add_service(redirect);

  log::info!(
    "rie-gateway: HTTPS={} (default_tls={}, sni_certs={}), HTTP={} → 301 https; default_upstream={}, host_routes={}, host_redirects={}",
    HTTPS_BIND,
    cert,
    sni_certs_log,
    HTTP_BIND,
    default_upstream_log,
    host_routes_log,
    host_redirects_log
  );
  server.run_forever();
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn forwarded_ip_drops_port() {
    let v4 =
      pingora_core::protocols::l4::socket::SocketAddr::Inet("203.0.113.7:51234".parse().unwrap());
    assert_eq!(forwarded_ip(&v4).as_deref(), Some("203.0.113.7"));
    let v6 =
      pingora_core::protocols::l4::socket::SocketAddr::Inet("[2001:db8::1]:443".parse().unwrap());
    assert_eq!(forwarded_ip(&v6).as_deref(), Some("2001:db8::1"));
  }

  fn header<'a>(resp: &'a ResponseHeader, name: &str) -> Option<&'a str> {
    resp.headers.get(name).and_then(|v| v.to_str().ok())
  }

  /// SEC-10：CSP 只由应用决定，gateway 原样转发（不覆盖、不补）。
  #[test]
  fn csp_from_app_passes_through_untouched() {
    let app_csp = "default-src 'self'; script-src 'self' 'wasm-unsafe-eval'";
    let mut resp = ResponseHeader::build(200, None).unwrap();
    resp.insert_header("Content-Security-Policy", app_csp).unwrap();
    apply_security_headers(&mut resp);
    assert_eq!(header(&resp, "content-security-policy"), Some(app_csp));
    assert_eq!(resp.headers.get_all("content-security-policy").iter().count(), 1);

    let mut bare = ResponseHeader::build(200, None).unwrap();
    apply_security_headers(&mut bare);
    assert_eq!(header(&bare, "content-security-policy"), None);
  }

  /// 应用已下发的通用安全头保留应用的值；缺失时 gateway 补默认值。HSTS 由 gateway 负责。
  #[test]
  fn generic_headers_fill_only_when_missing() {
    let mut resp = ResponseHeader::build(200, None).unwrap();
    resp.insert_header("X-Frame-Options", "SAMEORIGIN").unwrap();
    apply_security_headers(&mut resp);
    assert_eq!(header(&resp, "x-frame-options"), Some("SAMEORIGIN"));
    assert_eq!(header(&resp, "x-content-type-options"), Some("nosniff"));
    assert_eq!(header(&resp, "referrer-policy"), Some("strict-origin-when-cross-origin"));
    assert_eq!(header(&resp, "strict-transport-security"), Some(HSTS_VALUE));
  }

  /// RFC 9113 §8.2.3：多条 HTTP/2 `cookie` 头必须合并为单条 `; ` 分隔的 `Cookie` 头。
  #[test]
  fn coalesces_multiple_h2_cookie_headers() {
    let mut req = RequestHeader::build("POST", b"/api/auth/me", None).unwrap();
    req.append_header("cookie", "site_lang=zh").unwrap();
    req.append_header("cookie", "theme=ocean").unwrap();
    req.append_header("cookie", "session=jwt_token").unwrap();

    coalesce_cookie_headers(&mut req);

    assert_eq!(req.headers.get_all("cookie").iter().count(), 1);
    assert_eq!(
      req.headers.get("cookie").and_then(|v| v.to_str().ok()),
      Some("site_lang=zh; theme=ocean; session=jwt_token")
    );
  }

  #[test]
  fn host_upstreams_routes_h1_host_and_h2_authority_and_falls_back() {
    let gw = AppGateway {
      default_upstream: "127.0.0.1:8080".to_string(),
      host_upstreams: parse_host_upstreams(
        " dioxuspress.dev = 127.0.0.1:8081 , docs.example.com.=127.0.0.1:8082 , malformed ",
      ),
      host_redirects: parse_host_redirects(
        " dioxuspress.rustineverything.app = https://dioxuspress.dev/ , www.dioxuspress.dev=https://dioxuspress.dev ",
      ),
    };
    assert_eq!(gw.host_upstreams.len(), 2);
    assert_eq!(gw.host_redirects.len(), 2);

    // HTTP/1.1 Host header with port & uppercase
    let mut h1 = RequestHeader::build("GET", b"/docs/guide/intro", None).unwrap();
    h1.insert_header("Host", "DioxusPress.DEV:443").unwrap();
    assert_eq!(gw.select_upstream(&h1), "127.0.0.1:8081");
    assert_eq!(gw.select_redirect_location(&h1), None);

    // 301 redirect preserves path and query
    let mut old_sub = RequestHeader::build("GET", b"/docs/guide/quickstart?lang=en", None).unwrap();
    old_sub.insert_header("Host", "dioxuspress.rustineverything.app").unwrap();
    assert_eq!(
      gw.select_redirect_location(&old_sub).as_deref(),
      Some("https://dioxuspress.dev/docs/guide/quickstart?lang=en")
    );

    // HTTP/2 :authority stored in req.uri without Host header
    let mut h2 = RequestHeader::build("GET", b"/docs", None).unwrap();
    h2.set_uri("https://DOCS.example.com:443/docs".parse().unwrap());
    assert_eq!(gw.select_upstream(&h2), "127.0.0.1:8082");

    // Unmatched host falls back to default_upstream
    let mut main_req = RequestHeader::build("GET", b"/", None).unwrap();
    main_req.insert_header("Host", "rustineverything.app").unwrap();
    assert_eq!(gw.select_upstream(&main_req), "127.0.0.1:8080");
  }

  #[test]
  fn sni_domain_matching_handles_exact_and_wildcard_subdomains_safely() {
    let specs = parse_sni_cert_specs(
      " *.dioxuspress.dev = /certs/dp/full.pem:/certs/dp/key.pem , api.dioxuspress.dev=/certs/api/full.pem:/certs/api/key.pem ",
    );
    assert_eq!(specs.len(), 2);
    // Longer domain sorted first
    assert_eq!(specs[0].0, "api.dioxuspress.dev");
    assert_eq!(specs[1].0, "dioxuspress.dev");

    assert!(host_matches_domain("dioxuspress.dev", "dioxuspress.dev"));
    assert!(host_matches_domain("www.dioxuspress.dev", "dioxuspress.dev"));
    // Suffix collision without dot boundary must NOT match
    assert!(!host_matches_domain("fakedioxuspress.dev", "dioxuspress.dev"));
    assert!(!host_matches_domain("rustineverything.app", "dioxuspress.dev"));
  }
}
