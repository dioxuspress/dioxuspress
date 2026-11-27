//! 开源模型架构确定性巡检与诊断证据包生成器（供 `models-audit` 每日定时任务及 `model-arch-reviewer` Skill 复用）。
//!
//! 核心保障（杜绝静默传播错误知识）：
//! 1. **存量巡检安全闸门**：重跑已收录模型时，若新校验结果退化为 `Partial` / `Unverified`，
//!    **绝不覆盖线上已通过核验的 `blueprint_json`**，仅写入 `model_audit_queue` 报警；
//! 2. **热门新模型自动分流**：拉取 HuggingFace Trending 开源模型，过滤纯量化转码仓（GGUF/AWQ/EXL2 等），
//!    🟢 全绿进入 `verified_candidate`，🟡/🔴 一律拦截进 `pending_review` 并附上 `modeling_*.py` 源码直链与完整诊断证据包；
//! 3. **零幻觉证据包**：所有数值、Shape 冲突、L2 未建模张量族与 `config.json` 未知字段全部来自确定性 Rust 校验器。

use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::arch::{format_count, ModelBlueprint, VerificationReport, VerificationStatus};
use crate::atlas_store::config_fingerprint_hex;
use crate::server::resolve_model_blueprint_server;

/// 单条未建模 L2 张量证据摘要（可序列化供队列落库与评审 Skill 读取）。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct UnmodeledTensorEvidence {
  pub pattern: String,
  pub shape: Option<Vec<u64>>,
  pub repeat_count: usize,
  pub total_params: u64,
}

/// 确定性诊断证据包：记录单个模型完整的 L1×L2 对账事实与仓库 `modeling_*.py` 源码直链。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AuditEvidence {
  pub repo_id: String,
  pub source: String,
  pub modality: String,
  pub family_badge: String,
  pub architectures: Vec<String>,
  pub model_type: String,
  pub total_params_human: String,
  pub active_params_human: String,
  pub num_layers: u64,
  pub hidden_size: u64,
  pub l2_source_kind: Option<String>,
  pub l2_estimated_params_human: Option<String>,
  pub l2_total_bytes_gb: Option<String>,
  pub config_fingerprint: String,
  pub verification: VerificationReport,
  pub unmodeled_l2_tensors: Vec<UnmodeledTensorEvidence>,
  pub modeling_files: Vec<String>,
}

/// 构造确定性诊断证据包（纯函数，无网络 IO）。
pub fn build_audit_evidence(bp: &ModelBlueprint, modeling_files: Vec<String>) -> AuditEvidence {
  let verification = bp.verification_report().unwrap_or_else(|| VerificationReport {
    status: VerificationStatus::Partial,
    headline: "未生成核验报告".to_string(),
    checks: Vec::new(),
    issues: Vec::new(),
    circuit_broken: false,
  });
  let (l2_source_kind, l2_estimated_params_human, l2_total_bytes_gb, unmodeled_l2_tensors) =
    match &bp.safetensors {
      Some(st) => {
        let est = st.l2_estimated_params.map(format_count);
        let gb = st.total_bytes.map(|b| format!("{:.2}GB", (b as f64) / 1e9));
        let unmodeled = st
          .l2_unmodeled_other
          .iter()
          .map(|u| UnmodeledTensorEvidence {
            pattern: u.pattern.clone(),
            shape: u.shape.clone(),
            repeat_count: u.repeat_count,
            total_params: u.total_params,
          })
          .collect();
        (Some(st.source_kind.clone()), est, gb, unmodeled)
      }
      None => (None, None, None, Vec::new()),
    };

  AuditEvidence {
    repo_id: bp.repo_id.clone(),
    source: bp.source_label.clone(),
    modality: bp.modality_badge_label().to_string(),
    family_badge: bp.family_badge_label().to_string(),
    architectures: bp.architectures.clone(),
    model_type: bp.model_type.clone(),
    total_params_human: format_count(bp.params.total_params),
    active_params_human: format_count(bp.params.active_params),
    num_layers: bp.num_hidden_layers,
    hidden_size: bp.hidden_size,
    l2_source_kind,
    l2_estimated_params_human,
    l2_total_bytes_gb,
    config_fingerprint: config_fingerprint_hex(&bp.raw_config_pretty),
    verification,
    unmodeled_l2_tensors,
    modeling_files,
  }
}

/// 根据核验状态映射队列默认流转状态：🟢 进候选发布，🟡/🔴 一律拦截进待人工/Skill 复核。
pub fn queue_status_for(status: VerificationStatus) -> &'static str {
  match status {
    VerificationStatus::Verified | VerificationStatus::CuratedVerified => "verified_candidate",
    VerificationStatus::Partial | VerificationStatus::Unverified => "pending_review",
  }
}

/// 过滤纯量化转码/LoRA 衍生仓（如 GGUF / AWQ / GPTQ / EXL2 / MLX / LoRA），聚焦上游原生架构仓。
pub fn is_original_arch_candidate(repo_id: &str) -> bool {
  let lc = repo_id.to_ascii_lowercase();
  let skip_tokens = [
    "-gguf",
    "_gguf",
    "-awq",
    "-gptq",
    "-exl2",
    "-mlx",
    "-bnb-",
    "-4bit",
    "-8bit",
    "lora",
    "adapter",
    "quantized",
    "bartowski/",
    "unsloth/",
    "lmstudio-community/",
    "mlx-community/",
    "thebloke/",
  ];
  !skip_tokens.iter().any(|t| lc.contains(t))
}

fn audit_http_client() -> Result<reqwest::Client, String> {
  reqwest::Client::builder()
    .user_agent("DioxusPress-ModelsAudit/0.1 (+https://rustineverything.app)")
    .connect_timeout(Duration::from_secs(10))
    .timeout(Duration::from_secs(30))
    .build()
    .map_err(|e| format!("初始化审计 HTTP 客户端失败: {e}"))
}

/// 探测远端模型仓库中的 `modeling_*.py` 与 `configuration_*.py` 文件，返回可直接审阅的源码直链。
pub async fn probe_modeling_files(repo_id: &str, source_label: &str) -> Vec<String> {
  let Ok(client) = audit_http_client() else {
    return Vec::new();
  };
  let is_ms = source_label.to_ascii_lowercase().contains("modelscope");
  if is_ms {
    let url = format!("https://modelscope.cn/api/v1/models/{repo_id}/repo/files?Revision=master");
    let Ok(resp) = client.get(&url).send().await else {
      return Vec::new();
    };
    let Ok(val) = resp.json::<Value>().await else {
      return Vec::new();
    };
    let mut urls = Vec::new();
    if let Some(files) = val.pointer("/Data/Files").and_then(Value::as_array) {
      for f in files {
        if let Some(name) = f.get("Name").and_then(Value::as_str) {
          if name.ends_with(".py")
            && (name.starts_with("modeling_") || name.starts_with("configuration_"))
          {
            urls.push(format!("https://modelscope.cn/models/{repo_id}/resolve/master/{name}"));
          }
        }
      }
    }
    urls.sort();
    urls.truncate(8);
    return urls;
  }

  let url = format!("https://huggingface.co/api/models/{repo_id}/tree/main");
  let mut req = client.get(&url);
  if let Ok(tok) = std::env::var("HF_TOKEN") {
    let t = tok.trim();
    if !t.is_empty() {
      req = req.bearer_auth(t);
    }
  }
  let Ok(resp) = req.send().await else {
    return Vec::new();
  };
  let Ok(arr) = resp.json::<Vec<Value>>().await else {
    return Vec::new();
  };
  let mut urls = Vec::new();
  for item in arr {
    if let Some(path) = item.get("path").and_then(Value::as_str) {
      let base = path.rsplit('/').next().unwrap_or(path);
      if base.ends_with(".py")
        && (base.starts_with("modeling_") || base.starts_with("configuration_"))
      {
        urls.push(format!("https://huggingface.co/{repo_id}/blob/main/{path}"));
      }
    }
  }
  urls.sort();
  urls.truncate(8);
  urls
}

/// 从 HuggingFace Trending API 抓取近期热门开源模型候选列表（已过滤 GGUF/纯量化转码仓）。
pub async fn fetch_trending_open_models(limit: usize) -> Result<Vec<String>, String> {
  let client = audit_http_client()?;
  let fetch_n = (limit.saturating_mul(3)).clamp(15, 80);
  let url =
    format!("https://huggingface.co/api/models?sort=trendingScore&direction=-1&limit={fetch_n}");
  let mut req = client.get(&url);
  if let Ok(tok) = std::env::var("HF_TOKEN") {
    let t = tok.trim();
    if !t.is_empty() {
      req = req.bearer_auth(t);
    }
  }
  let resp = req.send().await.map_err(|e| format!("拉取 HF Trending 失败: {e}"))?;
  if !resp.status().is_success() {
    return Err(format!("HF Trending API 返回 HTTP {}", resp.status().as_u16()));
  }
  let arr: Vec<Value> =
    resp.json().await.map_err(|e| format!("解析 HF Trending JSON 失败: {e}"))?;
  let mut out = Vec::new();
  for item in arr {
    let gated = item
      .get("gated")
      .map(|v| v.as_bool().unwrap_or(false) || v.as_str().is_some())
      .unwrap_or(false);
    if gated {
      continue;
    }
    let Some(id) = item.get("id").and_then(Value::as_str) else {
      continue;
    };
    if !id.contains('/') || !is_original_arch_candidate(id) {
      continue;
    }
    out.push(id.to_string());
    if out.len() >= limit {
      break;
    }
  }
  Ok(out)
}

/// 对单个上游模型执行实时 L1×L2 硬校验并产出完整诊断证据包（零数据库依赖）。
pub async fn check_single_repo(
  repo: &str,
  source: &str,
  revision: &str,
) -> Result<(ModelBlueprint, AuditEvidence), String> {
  let bp = resolve_model_blueprint_server(repo, source, revision).await?;
  let modeling_files = probe_modeling_files(&bp.repo_id, &bp.source_label).await;
  let evidence = build_audit_evidence(&bp, modeling_files);
  Ok((bp, evidence))
}

/// 将诊断证据包幂等写入 `model_audit_queue` 表（按 `repo_id` upsert）。
pub async fn upsert_audit_queue_row(
  db: &sea_orm::DatabaseConnection,
  source: &str,
  evidence: &AuditEvidence,
) -> Result<(), String> {
  use app_core::entities::model_audit_queue;
  use sea_orm::sea_query::OnConflict;
  use sea_orm::{EntityTrait, Set};

  let now = chrono::Utc::now().fixed_offset();
  let queue_status = queue_status_for(evidence.verification.status).to_string();
  let evidence_json =
    serde_json::to_string(evidence).map_err(|e| format!("序列化 AuditEvidence 失败: {e}"))?;
  let modeling_files_json = serde_json::to_string(&evidence.modeling_files)
    .map_err(|e| format!("序列化 modeling_files 失败: {e}"))?;

  let active = model_audit_queue::ActiveModel {
    id: sea_orm::ActiveValue::NotSet,
    repo_id: Set(evidence.repo_id.clone()),
    source: Set(source.to_string()),
    queue_status: Set(queue_status),
    verification_status: Set(evidence.verification.status.as_str().to_string()),
    headline: Set(evidence.verification.headline.clone()),
    evidence_json: Set(evidence_json),
    modeling_files_json: Set(modeling_files_json),
    discovered_at: Set(now),
    updated_at: Set(now),
  };

  model_audit_queue::Entity::insert(active)
    .on_conflict(
      OnConflict::column(model_audit_queue::Column::RepoId)
        .update_columns([
          model_audit_queue::Column::Source,
          model_audit_queue::Column::QueueStatus,
          model_audit_queue::Column::VerificationStatus,
          model_audit_queue::Column::Headline,
          model_audit_queue::Column::EvidenceJson,
          model_audit_queue::Column::ModelingFilesJson,
          model_audit_queue::Column::UpdatedAt,
        ])
        .to_owned(),
    )
    .exec(db)
    .await
    .map_err(|e| format!("写入 model_audit_queue 失败: {e}"))?;

  Ok(())
}

/// 每日巡检汇总统计。
#[derive(Debug, Default, Serialize)]
pub struct AuditRunSummary {
  pub checked: usize,
  pub verified: usize,
  pub partial: usize,
  pub unverified: usize,
  pub errors: usize,
}

/// 巡检数据库中全部已收录的 `curated_models`：
/// - 若重跑通过 🟢 `Verified`：安全刷新 `verified_at` / `verification_report_json` / `upstream_fingerprint`；
/// - 若退化为 🟡/🔴：**绝不覆盖线上 `blueprint_json`**，仅写入 `model_audit_queue` (`pending_review`) 报警！
pub async fn audit_curated_models(
  db: &sea_orm::DatabaseConnection,
  dry_run: bool,
) -> Result<AuditRunSummary, String> {
  use app_core::entities::curated_model;
  use sea_orm::{ActiveModelTrait, EntityTrait, Set};

  let rows = curated_model::Entity::find()
    .all(db)
    .await
    .map_err(|e| format!("读取 curated_models 失败: {e}"))?;

  let mut stats = AuditRunSummary::default();
  for row in rows {
    let repo = row.hf_repo.trim().to_string();
    if repo.is_empty() || !repo.contains('/') {
      continue;
    }
    stats.checked += 1;
    let src = if row.ms_repo.as_deref().is_some_and(|s| !s.trim().is_empty()) && repo.is_empty() {
      "modelscope"
    } else {
      "auto"
    };
    match check_single_repo(&repo, src, "").await {
      Ok((_bp, evidence)) => match evidence.verification.status {
        VerificationStatus::Verified | VerificationStatus::CuratedVerified => {
          stats.verified += 1;
          tracing::info!(
            slug = %row.slug,
            repo = %repo,
            status = %evidence.verification.status.as_str(),
            "models-audit: curated model verified"
          );
          if !dry_run {
            let now = chrono::Utc::now().fixed_offset();
            let rep_json = serde_json::to_string(&evidence.verification).ok();
            let mut am: curated_model::ActiveModel = row.into();
            am.verification_status = Set(evidence.verification.status.as_str().to_string());
            am.verification_report_json = Set(rep_json);
            am.verified_at = Set(Some(now));
            am.upstream_fingerprint = Set(Some(evidence.config_fingerprint.clone()));
            am.updated_at = Set(now);
            if let Err(e) = am.update(db).await {
              tracing::warn!(repo = %repo, error = %e, "models-audit: update curated_model failed");
            }
          }
        }
        VerificationStatus::Partial => {
          stats.partial += 1;
          tracing::warn!(
            slug = %row.slug,
            repo = %repo,
            headline = %evidence.verification.headline,
            "models-audit: curated model downgraded to partial; preserving live blueprint and queueing review"
          );
          if !dry_run {
            if let Err(e) = upsert_audit_queue_row(db, src, &evidence).await {
              tracing::warn!(repo = %repo, error = %e, "models-audit: queue insert failed");
            }
          }
        }
        VerificationStatus::Unverified => {
          stats.unverified += 1;
          tracing::error!(
            slug = %row.slug,
            repo = %repo,
            headline = %evidence.verification.headline,
            "models-audit: curated model triggered circuit breaker; preserving live blueprint and queueing review"
          );
          if !dry_run {
            if let Err(e) = upsert_audit_queue_row(db, src, &evidence).await {
              tracing::warn!(repo = %repo, error = %e, "models-audit: queue insert failed");
            }
          }
        }
      },
      Err(err) => {
        stats.errors += 1;
        tracing::warn!(slug = %row.slug, repo = %repo, error = %err, "models-audit: fetch failed");
      }
    }
  }
  Ok(stats)
}

/// 发现并预审 HuggingFace Trending 开源新模型，按校验等级自动分流进 `model_audit_queue`。
pub async fn discover_and_audit_trending(
  db: Option<&sea_orm::DatabaseConnection>,
  limit: usize,
  dry_run: bool,
) -> Result<(AuditRunSummary, Vec<AuditEvidence>), String> {
  let repos = fetch_trending_open_models(limit).await?;
  let mut stats = AuditRunSummary::default();
  let mut evidences = Vec::new();

  for repo in repos {
    stats.checked += 1;
    match check_single_repo(&repo, "hf", "").await {
      Ok((_bp, evidence)) => {
        match evidence.verification.status {
          VerificationStatus::Verified | VerificationStatus::CuratedVerified => {
            stats.verified += 1;
          }
          VerificationStatus::Partial => stats.partial += 1,
          VerificationStatus::Unverified => stats.unverified += 1,
        }
        if !dry_run {
          if let Some(conn) = db {
            if let Err(e) = upsert_audit_queue_row(conn, "hf", &evidence).await {
              tracing::warn!(repo = %repo, error = %e, "models-audit: upsert trending row failed");
            }
          }
        }
        evidences.push(evidence);
      }
      Err(err) => {
        stats.errors += 1;
        tracing::info!(repo = %repo, error = %err, "models-audit: skipped non-standard or unavailable trending repo");
      }
    }
  }
  Ok((stats, evidences))
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::arch::load_builtin_preset;

  #[test]
  fn filters_quantized_derivatives_and_builds_deterministic_evidence() {
    assert!(is_original_arch_candidate("deepseek-ai/DeepSeek-V4-Pro"));
    assert!(is_original_arch_candidate("Qwen/Qwen3-30B-A3B"));
    assert!(!is_original_arch_candidate("bartowski/Qwen3-8B-GGUF"));
    assert!(!is_original_arch_candidate("mlx-community/DeepSeek-V3-4bit"));
    assert!(!is_original_arch_candidate("org/Model-AWQ"));

    let bp = load_builtin_preset("deepseek-v3").expect("builtin preset");
    let ev = build_audit_evidence(
      &bp,
      vec!["https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/modeling_deepseek.py".into()],
    );
    assert_eq!(ev.verification.status, VerificationStatus::CuratedVerified);
    assert_eq!(queue_status_for(ev.verification.status), "verified_candidate");
    assert_eq!(queue_status_for(VerificationStatus::Partial), "pending_review");
    assert_eq!(queue_status_for(VerificationStatus::Unverified), "pending_review");
    assert_eq!(ev.config_fingerprint.len(), 16);
    assert_eq!(ev.modeling_files.len(), 1);
  }
}
