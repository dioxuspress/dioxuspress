//! 服务端模型元数据安全拉取器（支持 HuggingFace / HF-Mirror / ModelScope 双源与 HTTP Range 头部采样）。
//!
//! 安全边界（S1 / SSRF 防护）：
//! - 前端仅传入结构化 `(repo, source, revision)`，禁止直接传入任意外部 URL 请求；
//! - 若用户粘贴了完整的 HuggingFace 或 ModelScope 浏览器地址，解析器仅从中提取 `owner/name` 并校验字符集；
//! - `owner`、`name`、`revision` 仅允许 `[A-Za-z0-9._-]` 且禁止 `.` / `..`；
//! - 自定义重定向策略仅允许跳转至受信任的模型托管与对象存储 CDN 域名（`https` only）；
//! - 流式按块读取响应并强制字节上限，单文件或分片 `.safetensors` 仅经 `Range` 拉取头部 JSON，绝不下载权重体。

use dioxus::prelude::*;
use serde::{Deserialize, Serialize};

use crate::arch::ModelBlueprint;

/// 支持的模型托管源。
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ModelHubSource {
  /// 自动选择：优先 HuggingFace，受限或超时自动回退 HF-Mirror 与 ModelScope
  Auto,
  HuggingFace,
  ModelScope,
}

impl ModelHubSource {
  pub fn parse_str(raw: &str) -> Self {
    match raw.trim().to_ascii_lowercase().as_str() {
      "hf" | "huggingface" => Self::HuggingFace,
      "ms" | "modelscope" => Self::ModelScope,
      _ => Self::Auto,
    }
  }

  pub fn as_str(&self) -> &'static str {
    match self {
      Self::Auto => "auto",
      Self::HuggingFace => "hf",
      Self::ModelScope => "modelscope",
    }
  }
}

/// 校验单个路径段（owner / repo_name / revision）：非空、长度 ≤ 96、仅含安全字符且不含 `..`。
pub fn is_safe_hub_segment(seg: &str) -> bool {
  !seg.is_empty()
    && seg.len() <= 96
    && seg != "."
    && seg != ".."
    && !seg.contains("..")
    && seg.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
}

/// 规范化用户输入的仓库标识（兼容粘贴完整 HuggingFace / ModelScope URL 或 `owner/name`）。
///
/// 返回 `(source, canonical_repo_id, normalized_revision)`。
pub fn normalize_repo_request(
  raw_repo: &str,
  raw_source: &str,
  raw_revision: &str,
) -> Result<(ModelHubSource, String, Option<String>), String> {
  let mut s = raw_repo.trim();
  if s.is_empty() {
    return Err("请输入模型仓库名（如 Qwen/Qwen3-8B 或 deepseek-ai/DeepSeek-V3）".to_string());
  }

  let mut inferred_source = ModelHubSource::parse_str(raw_source);

  // 兼容用户直接粘贴完整网页链接：
  // - https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json
  // - https://hf-mirror.com/Qwen/Qwen3-8B
  // - https://modelscope.cn/models/deepseek-ai/DeepSeek-V3/summary
  // - https://www.modelscope.cn/models/Qwen/Qwen3-30B-A3B
  if let Some(rest) = s.strip_prefix("https://").or_else(|| s.strip_prefix("http://")) {
    let (host, path) = rest.split_once('/').unwrap_or((rest, ""));
    let host_lc = host.to_ascii_lowercase();
    if host_lc.ends_with("huggingface.co") || host_lc.ends_with("hf-mirror.com") {
      if inferred_source == ModelHubSource::Auto {
        inferred_source = ModelHubSource::HuggingFace;
      }
      s = path;
    } else if host_lc.ends_with("modelscope.cn") {
      if inferred_source == ModelHubSource::Auto {
        inferred_source = ModelHubSource::ModelScope;
      }
      s = path.strip_prefix("models/").unwrap_or(path);
    } else {
      return Err("仅支持 HuggingFace 或 ModelScope 模型仓库链接".to_string());
    }
  }

  let clean_path = s.split(['?', '#']).next().unwrap_or(s).trim_matches('/');
  let mut parts = clean_path.split('/');
  let (Some(owner), Some(name)) = (parts.next(), parts.next()) else {
    return Err("仓库格式应为 owner/model（例如 Qwen/Qwen3-8B）".to_string());
  };

  if !is_safe_hub_segment(owner) || !is_safe_hub_segment(name) {
    return Err("仓库名包含非法字符，仅允许字母、数字、`.`、`_`、`-`".to_string());
  }

  let rev_trimmed = raw_revision.trim();
  let rev = if rev_trimmed.is_empty() {
    None
  } else if is_safe_hub_segment(rev_trimmed) {
    Some(rev_trimmed.to_string())
  } else {
    return Err("分支或版本 (revision) 包含非法字符".to_string());
  };

  Ok((inferred_source, format!("{owner}/{name}"), rev))
}

#[cfg(feature = "server")]
mod imp {
  use std::collections::HashMap;
  use std::sync::{Mutex, OnceLock};
  use std::time::{Duration, Instant};

  use crate::arch::{build_model_blueprint, load_builtin_preset, ModelBlueprint};
  use crate::safetensors_meta::{
    parse_safetensors_header_bytes, parse_safetensors_header_len, parse_safetensors_index_json,
    select_representative_shards, summarize_index_and_headers, summarize_single_header,
    RawTensorMeta, SafetensorsSummary, MAX_SAFETENSORS_HEADER_BYTES,
  };

  use super::{normalize_repo_request, ModelHubSource};

  /// `config.json` / `tokenizer_config.json` 单文件上限（2 MiB）。
  const MAX_CONFIG_BYTES: usize = 2 * 1024 * 1024;
  /// `model.safetensors.index.json` 单文件上限（12 MiB，兼容 671B~1T MoE 模型的数万条分片映射）。
  const MAX_INDEX_BYTES: usize = 12 * 1024 * 1024;
  /// 首次 Range 探测读取窗口（64 KiB，95% 以上的 Safetensors Header 一次往返即可读完）。
  const INITIAL_HEADER_PROBE_BYTES: usize = 64 * 1024;
  /// 缓存容量与过期时间。
  const CACHE_CAPACITY: usize = 64;
  const CACHE_TTL: Duration = Duration::from_secs(30 * 60);

  #[derive(Clone)]
  struct CacheEntry {
    inserted_at: Instant,
    blueprint: ModelBlueprint,
  }

  fn blueprint_cache() -> &'static Mutex<HashMap<String, CacheEntry>> {
    static CACHE: OnceLock<Mutex<HashMap<String, CacheEntry>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
  }

  /// 判断重定向目标主机是否属于受信任的 HuggingFace / ModelScope 及其底层 LFS/OSS CDN。
  pub fn is_trusted_redirect_host(host: &str) -> bool {
    let h = host.to_ascii_lowercase();
    const TRUSTED_SUFFIXES: &[&str] = &[
      "huggingface.co",
      "hf.co",
      "hf-mirror.com",
      "modelscope.cn",
      "aliyuncs.com",
      "cloudfront.net",
      "amazonaws.com",
    ];
    TRUSTED_SUFFIXES.iter().any(|suf| h == *suf || h.ends_with(&format!(".{suf}")))
  }

  fn shared_hub_client() -> Result<&'static reqwest::Client, String> {
    static CLIENT: OnceLock<Result<reqwest::Client, String>> = OnceLock::new();
    CLIENT
      .get_or_init(|| {
        let policy = reqwest::redirect::Policy::custom(|attempt| {
          if attempt.previous().len() >= 6 {
            return attempt.error("重定向次数过多");
          }
          let bad_host = {
            let url = attempt.url();
            if url.scheme() != "https" {
              return attempt.error("仅允许 HTTPS 重定向");
            }
            match url.host_str() {
              None => return attempt.error("重定向缺少主机名"),
              Some(host) if !is_trusted_redirect_host(host) => Some(host.to_string()),
              Some(_) => None,
            }
          };
          if let Some(host) = bad_host {
            return attempt.error(format!("拒绝跳转到非受信主机: {host}"));
          }
          attempt.follow()
        });

        reqwest::Client::builder()
          .user_agent("DioxusPress-ModelVisualizer/0.1 (+https://rustineverything.app)")
          .connect_timeout(Duration::from_secs(8))
          .timeout(Duration::from_secs(20))
          .redirect(policy)
          .build()
          .map_err(|e| format!("初始化 HTTP 客户端失败: {e}"))
      })
      .as_ref()
      .map_err(Clone::clone)
  }

  /// 单个远端端点的文件 URL 组装配置。
  #[derive(Debug, Clone)]
  pub struct HubEndpoint {
    pub label: &'static str,
    /// 形如 `https://huggingface.co/{repo}/resolve/{rev}`
    pub base_resolve_url: String,
    pub bearer_token: Option<String>,
  }

  impl HubEndpoint {
    pub fn file_url(&self, filename: &str) -> String {
      format!("{}/{}", self.base_resolve_url.trim_end_matches('/'), filename)
    }
  }

  /// 按用户指定的源组装候选上游列表（支持主站受限/超时自动回退）。
  fn build_candidate_endpoints(
    source: ModelHubSource,
    repo_id: &str,
    revision: Option<&str>,
  ) -> Vec<HubEndpoint> {
    let hf_rev = revision.unwrap_or("main");
    let ms_rev = revision.unwrap_or("master");
    let hf_token =
      std::env::var("HF_TOKEN").ok().map(|s| s.trim().to_string()).filter(|s| !s.is_empty());
    let ms_token = std::env::var("MODELSCOPE_API_TOKEN")
      .ok()
      .map(|s| s.trim().to_string())
      .filter(|s| !s.is_empty());

    let hf_main = HubEndpoint {
      label: "HuggingFace",
      base_resolve_url: format!("https://huggingface.co/{repo_id}/resolve/{hf_rev}"),
      bearer_token: hf_token.clone(),
    };
    let hf_mirror = HubEndpoint {
      label: "HF-Mirror",
      base_resolve_url: format!("https://hf-mirror.com/{repo_id}/resolve/{hf_rev}"),
      bearer_token: hf_token,
    };
    let ms_master = HubEndpoint {
      label: "ModelScope",
      base_resolve_url: format!("https://modelscope.cn/models/{repo_id}/resolve/{ms_rev}"),
      bearer_token: ms_token.clone(),
    };
    let ms_main = HubEndpoint {
      label: "ModelScope",
      base_resolve_url: format!("https://modelscope.cn/models/{repo_id}/resolve/main"),
      bearer_token: ms_token,
    };

    match source {
      ModelHubSource::HuggingFace => vec![hf_main, hf_mirror],
      ModelHubSource::ModelScope => {
        if revision.is_some() {
          vec![ms_master]
        } else {
          vec![ms_master, ms_main]
        }
      }
      ModelHubSource::Auto => {
        let mut list = vec![hf_main, ms_master, hf_mirror];
        if revision.is_none() {
          list.push(ms_main);
        }
        list
      }
    }
  }

  /// 带缓存拉取并构建 [`ModelBlueprint`]。
  pub async fn resolve_model_blueprint(
    raw_repo: &str,
    raw_source: &str,
    raw_revision: &str,
  ) -> Result<ModelBlueprint, String> {
    let (source, repo_id, revision) = normalize_repo_request(raw_repo, raw_source, raw_revision)?;
    let cache_key = format!(
      "{}:{}:{}",
      source.as_str(),
      repo_id.to_ascii_lowercase(),
      revision.as_deref().unwrap_or("-")
    );

    if let Ok(mut guard) = blueprint_cache().lock() {
      if let Some(hit) = guard.get(&cache_key) {
        if hit.inserted_at.elapsed() < CACHE_TTL {
          return Ok(hit.blueprint.clone());
        }
        guard.remove(&cache_key);
      }
    }

    let client = shared_hub_client()?;
    let endpoints = build_candidate_endpoints(source, &repo_id, revision.as_deref());
    let mut errors: Vec<String> = Vec::new();

    for ep in &endpoints {
      match fetch_from_endpoint_with_client(client, ep, &repo_id).await {
        Ok(bp) => {
          if let Ok(mut guard) = blueprint_cache().lock() {
            if guard.len() >= CACHE_CAPACITY {
              guard.retain(|_, v| v.inserted_at.elapsed() < CACHE_TTL);
              if guard.len() >= CACHE_CAPACITY {
                guard.clear();
              }
            }
            guard
              .insert(cache_key, CacheEntry { inserted_at: Instant::now(), blueprint: bp.clone() });
          }
          return Ok(bp);
        }
        Err(err) => {
          errors.push(format!("{}: {err}", ep.label));
        }
      }
    }

    // 若门禁模型（如 meta-llama/*）在线未授权，而恰好命中内置高精度快照，则优雅回退
    if revision.is_none() {
      if let Some(preset) = load_builtin_preset(&repo_id) {
        return Ok(preset);
      }
    }

    Err(format!(
      "未能从上游获取 `{repo_id}` 的架构配置（{}）。若是受限模型（Gated Repo），可切换 ModelScope 源或直接粘贴本地 config.json。",
      errors.join("；")
    ))
  }

  /// 从单个 [`HubEndpoint`] 拉取 L1 配置与 L2 Safetensors 元数据（注入自定义 `Client` 便于本地集成测试）。
  pub async fn fetch_from_endpoint_with_client(
    client: &reqwest::Client,
    endpoint: &HubEndpoint,
    repo_id: &str,
  ) -> Result<ModelBlueprint, String> {
    let config_url = endpoint.file_url("config.json");
    let gen_url = endpoint.file_url("generation_config.json");
    let tok_url = endpoint.file_url("tokenizer_config.json");
    let idx_url = endpoint.file_url("model.safetensors.index.json");

    let (cfg_res, gen_res, tok_res, idx_res) = tokio::join!(
      fetch_utf8_capped(client, &config_url, endpoint.bearer_token.as_deref(), MAX_CONFIG_BYTES),
      fetch_utf8_capped(client, &gen_url, endpoint.bearer_token.as_deref(), MAX_CONFIG_BYTES),
      fetch_utf8_capped(client, &tok_url, endpoint.bearer_token.as_deref(), MAX_CONFIG_BYTES),
      fetch_utf8_capped(client, &idx_url, endpoint.bearer_token.as_deref(), MAX_INDEX_BYTES),
    );

    let config_json = cfg_res?;
    let generation_json = gen_res.ok();
    let tokenizer_json = tok_res.ok();

    // L2：若存在多分片 `model.safetensors.index.json`，挑最多 3 个代表性分片并发做 Range 采样；
    // 否则尝试单文件 `model.safetensors` 的 Range Header 读取。
    let safetensors: Option<SafetensorsSummary> = match idx_res {
      Ok(raw_idx) => match parse_safetensors_index_json(&raw_idx) {
        Ok((total_size, weight_map)) => {
          let sample_shards = select_representative_shards(&weight_map);
          let sampled = sample_shard_headers(client, endpoint, &sample_shards).await;
          Some(summarize_index_and_headers(total_size, &weight_map, &sampled))
        }
        Err(e) => {
          tracing::warn!(repo = %repo_id, error = %e, "ai visualizer: failed to parse safetensors index");
          None
        }
      },
      Err(_) => {
        let single_url = endpoint.file_url("model.safetensors");
        match fetch_safetensors_header_range(
          client,
          &single_url,
          endpoint.bearer_token.as_deref(),
          Some("model.safetensors"),
        )
        .await
        {
          Ok(tensors) if !tensors.is_empty() => Some(summarize_single_header(tensors)),
          _ => None,
        }
      }
    };

    build_model_blueprint(
      repo_id,
      endpoint.label,
      &config_json,
      generation_json.as_deref(),
      tokenizer_json.as_deref(),
      safetensors,
    )
  }

  async fn sample_shard_headers(
    client: &reqwest::Client,
    endpoint: &HubEndpoint,
    shards: &[String],
  ) -> Vec<RawTensorMeta> {
    let mut out = Vec::new();
    let futs = shards.iter().map(|shard| {
      let url = endpoint.file_url(shard);
      let token = endpoint.bearer_token.as_deref();
      async move { fetch_safetensors_header_range(client, &url, token, Some(shard)).await }
    });

    // 最多 3 个分片，顺序或小并发收集
    for fut in futs {
      if let Ok(mut items) = fut.await {
        out.append(&mut items);
      }
    }
    out
  }

  /// 流式读取 UTF-8 文本响应，严格限制最大字节数 `max_bytes`。
  async fn fetch_utf8_capped(
    client: &reqwest::Client,
    url: &str,
    bearer_token: Option<&str>,
    max_bytes: usize,
  ) -> Result<String, String> {
    let mut req = client.get(url);
    if let Some(tok) = bearer_token {
      req = req.bearer_auth(tok);
    }
    let mut resp = req.send().await.map_err(|e| format!("请求失败 ({e})"))?;
    let status = resp.status();
    if !status.is_success() {
      return Err(format!("HTTP {}", status.as_u16()));
    }

    let mut buf = Vec::new();
    while let Some(chunk) = resp.chunk().await.map_err(|e| format!("读取响应流中断 ({e})"))?
    {
      if buf.len().saturating_add(chunk.len()) > max_bytes {
        return Err(format!("文件体积超过解析上限 ({max_bytes} B)"));
      }
      buf.extend_from_slice(&chunk);
    }

    String::from_utf8(buf).map_err(|_| "响应内容不是合法 UTF-8 JSON".to_string())
  }

  /// 使用 HTTP `Range` 仅拉取 `.safetensors` 文件的 JSON Header（零权重下载）。
  ///
  /// 即使遇到忽略 `Range` 直接返回 `200 OK` 的服务端，也会在读够所需头部字节后立即主动断开流。
  pub async fn fetch_safetensors_header_range(
    client: &reqwest::Client,
    url: &str,
    bearer_token: Option<&str>,
    shard_name: Option<&str>,
  ) -> Result<Vec<RawTensorMeta>, String> {
    let probe_end = INITIAL_HEADER_PROBE_BYTES - 1;
    let first_buf =
      fetch_byte_window(client, url, bearer_token, 0, probe_end, INITIAL_HEADER_PROBE_BYTES)
        .await?;

    let header_len = parse_safetensors_header_len(&first_buf)?;
    let total_needed = 8usize.saturating_add(header_len);
    if total_needed <= first_buf.len() {
      return parse_safetensors_header_bytes(&first_buf[..total_needed], shard_name);
    }

    if header_len > MAX_SAFETENSORS_HEADER_BYTES {
      return Err("Safetensors Header 超出允许上限".to_string());
    }

    // 首包 64 KiB 未装下完整 Header 时，精确补齐剩余 `[first_buf.len() .. total_needed - 1]` 区间
    let rem_start = first_buf.len();
    let rem_end = total_needed - 1;
    let rem_len = total_needed - rem_start;
    let rest = fetch_byte_window(client, url, bearer_token, rem_start, rem_end, rem_len).await?;

    let mut combined = Vec::with_capacity(total_needed);
    combined.extend_from_slice(&first_buf);
    combined.extend_from_slice(&rest);
    parse_safetensors_header_bytes(&combined, shard_name)
  }

  async fn fetch_byte_window(
    client: &reqwest::Client,
    url: &str,
    bearer_token: Option<&str>,
    start: usize,
    end_inclusive: usize,
    max_collect: usize,
  ) -> Result<Vec<u8>, String> {
    let mut req =
      client.get(url).header(reqwest::header::RANGE, format!("bytes={start}-{end_inclusive}"));
    if let Some(tok) = bearer_token {
      req = req.bearer_auth(tok);
    }
    let mut resp = req.send().await.map_err(|e| format!("Range 请求失败 ({e})"))?;
    let status = resp.status();
    if !status.is_success() {
      return Err(format!("Range HTTP {}", status.as_u16()));
    }
    // 若上游忽略了 Range 头并在第二次偏移请求中返回 200（从头重发），仅当 start == 0 时可安全截取前缀
    if start > 0 && status != reqwest::StatusCode::PARTIAL_CONTENT {
      return Err("上游不支持分段 Range 续读".to_string());
    }

    let mut out = Vec::with_capacity(max_collect.min(65536));
    while let Some(chunk) = resp.chunk().await.map_err(|e| format!("读取分片头部失败 ({e})"))?
    {
      let remaining = max_collect.saturating_sub(out.len());
      if remaining == 0 {
        break;
      }
      let take = remaining.min(chunk.len());
      out.extend_from_slice(&chunk[..take]);
      if out.len() >= max_collect {
        break;
      }
    }
    Ok(out)
  }
}

/// Dioxus Server Function：按 HuggingFace / ModelScope 仓库标识实时解析模型架构与 Safetensors 元数据。
#[server]
pub async fn fetch_model_blueprint(
  repo: String,
  source: String,
  revision: String,
) -> Result<ModelBlueprint, ServerFnError> {
  imp::resolve_model_blueprint(&repo, &source, &revision).await.map_err(ServerFnError::new)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
  use super::*;

  #[test]
  fn normalizes_plain_repos_and_full_urls() {
    let (src, repo, rev) = normalize_repo_request("Qwen/Qwen3-8B", "auto", "").unwrap();
    assert_eq!(src, ModelHubSource::Auto);
    assert_eq!(repo, "Qwen/Qwen3-8B");
    assert_eq!(rev, None);

    let (src2, repo2, _) = normalize_repo_request(
      "https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json",
      "auto",
      "",
    )
    .unwrap();
    assert_eq!(src2, ModelHubSource::HuggingFace);
    assert_eq!(repo2, "deepseek-ai/DeepSeek-V3");

    let (src3, repo3, _) = normalize_repo_request(
      "https://www.modelscope.cn/models/Qwen/Qwen3-30B-A3B/summary",
      "auto",
      "v1.0",
    )
    .unwrap();
    assert_eq!(src3, ModelHubSource::ModelScope);
    assert_eq!(repo3, "Qwen/Qwen3-30B-A3B");
  }

  #[test]
  fn rejects_ssrf_and_path_traversal_inputs() {
    assert!(normalize_repo_request("http://127.0.0.1:8080/admin/secret", "auto", "").is_err());
    assert!(normalize_repo_request("https://evil.example.com/a/b", "auto", "").is_err());
    assert!(normalize_repo_request("../etc/passwd", "auto", "").is_err());
    assert!(normalize_repo_request("Qwen/..", "hf", "").is_err());
    assert!(normalize_repo_request("Qwen/Qwen3-8B", "hf", "../main").is_err());
  }

  #[cfg(feature = "server")]
  #[tokio::test]
  async fn fetches_config_and_range_safetensors_header_from_local_server() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    let header_json = br#"{"model.embed_tokens.weight":{"dtype":"BF16","shape":[32000,2048],"data_offsets":[0,131072000]},"model.layers.0.self_attn.q_norm.weight":{"dtype":"BF16","shape":[128],"data_offsets":[131072000,131072256]}}"#;
    let mut st_bytes = Vec::from((header_json.len() as u64).to_le_bytes());
    st_bytes.extend_from_slice(header_json);
    st_bytes.extend_from_slice(&vec![0xAAu8; 4096]); // simulated weight body

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let server_task = tokio::spawn(async move {
      loop {
        let Ok((mut sock, _)) = listener.accept().await else {
          break;
        };
        let st_clone = st_bytes.clone();
        tokio::spawn(async move {
          let mut req_buf = [0u8; 2048];
          let n = sock.read(&mut req_buf).await.unwrap_or(0);
          let req_str = String::from_utf8_lossy(&req_buf[..n]);
          let first_line = req_str.lines().next().unwrap_or("");
          let path = first_line.split_whitespace().nth(1).unwrap_or("");

          let (status, body): (&str, Vec<u8>) = if path.ends_with("/config.json") {
            (
              "200 OK",
              br#"{"architectures":["Qwen3ForCausalLM"],"model_type":"qwen3","vocab_size":32000,"hidden_size":2048,"intermediate_size":6144,"num_hidden_layers":24,"num_attention_heads":16,"num_key_value_heads":4,"head_dim":128}"#.to_vec(),
            )
          } else if path.ends_with("/model.safetensors") {
            ("206 Partial Content", st_clone)
          } else {
            ("404 Not Found", b"not found".to_vec())
          };

          let resp_head = format!(
            "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
          );
          let _ = sock.write_all(resp_head.as_bytes()).await;
          let _ = sock.write_all(&body).await;
        });
      }
    });

    // AGENTS.md Rust HTTP Testing Guidelines: MUST use `.no_proxy()` for loopback test server!
    let client = reqwest::Client::builder().no_proxy().build().expect("test client");
    let endpoint = imp::HubEndpoint {
      label: "TestHub",
      base_resolve_url: format!("http://{addr}/demo/tiny-qwen3/resolve/main"),
      bearer_token: None,
    };

    let bp = imp::fetch_from_endpoint_with_client(&client, &endpoint, "demo/tiny-qwen3")
      .await
      .expect("fetch blueprint");

    assert_eq!(bp.hidden_size, 2048);
    assert_eq!(bp.num_hidden_layers, 24);
    let st = bp.safetensors.expect("safetensors summary parsed via Range");
    assert_eq!(st.total_tensor_count, 2);
    assert!(st.has_qk_norm);

    server_task.abort();
  }
}
