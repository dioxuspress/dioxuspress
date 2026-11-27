//! 模型架构图鉴在线收录存储层（PostgreSQL 持久化 + 静态目录双源合并 + 自动亮点推导 + LLM 润色）。
//!
//! 设计边界：
//! - 零数据库档位（`basic` / `dioxuspress.dev`）下 `app_core::db::pool::pool()` 返回 `None`，自动无感退化为纯磁盘模式；
//! - 全功能档位（`accounts` / `full`）下，数据库 `curated_models` 与 `<CONTENT_DIR>/models` 按 `slug` 合并（同 `slug` 时数据库优先覆盖，标记 `RecordOrigin::Override`）；
//! - 内存缓存由单调递增代数 `ATLAS_DB_STAMP` 控制，管理员增删改或 `SIGHUP` 时即时失效并同步刷新匿名页面弱 ETag。

#[cfg(any(feature = "server", test))]
use std::collections::HashMap;
#[cfg(feature = "server")]
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(feature = "server")]
use std::sync::OnceLock;

use app_core::i18n::Language;
#[cfg(feature = "accounts")]
use dioxus::prelude::*;
use serde::{Deserialize, Serialize};

#[cfg(any(feature = "server", feature = "accounts", test))]
use crate::arch::format_count;
use crate::arch::{ArchFamily, ModelBlueprint};
#[cfg(any(feature = "server", test))]
use crate::arch::{AttentionSpec, FfnSpec, OutputHeadSpec, PrecisionPreset};
#[cfg(any(feature = "server", test))]
use crate::atlas::CuratedModelDetail;
#[cfg(any(feature = "server", feature = "accounts", test))]
#[cfg(feature = "server")]
use crate::atlas::CuratedModelSummary;

/// 模型所属主模态分类（支撑 `/models` 顶部模态筛选 Tab）。
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum ModelModality {
  #[default]
  Llm,
  Vlm,
  Audio,
  Diffusion,
  Embedding,
  Detection,
  Segmentation,
  Classic,
}

impl ModelModality {
  pub const ALL: &'static [Self] = &[
    Self::Llm,
    Self::Vlm,
    Self::Audio,
    Self::Diffusion,
    Self::Embedding,
    Self::Detection,
    Self::Segmentation,
    Self::Classic,
  ];

  pub fn as_str(self) -> &'static str {
    match self {
      Self::Llm => "llm",
      Self::Vlm => "vlm",
      Self::Audio => "audio",
      Self::Diffusion => "diffusion",
      Self::Embedding => "embedding",
      Self::Detection => "detection",
      Self::Segmentation => "segmentation",
      Self::Classic => "classic",
    }
  }

  pub fn parse_str(raw: &str) -> Option<Self> {
    match raw.trim().to_ascii_lowercase().as_str() {
      "llm" | "text" | "causal_lm" => Some(Self::Llm),
      "vlm" | "multimodal" | "vision-language" => Some(Self::Vlm),
      "audio" | "asr" | "tts" | "speech" | "omni" => Some(Self::Audio),
      "diffusion" | "image" | "video" | "dit" | "mmdit" => Some(Self::Diffusion),
      "embedding" | "reranker" | "encoder" => Some(Self::Embedding),
      "detection" | "det" | "yolo" | "object-detection" => Some(Self::Detection),
      "segmentation" | "seg" | "unet" | "sam" | "medical" => Some(Self::Segmentation),
      "classic" | "base" | "foundation" | "cnn" | "rnn" | "nlp" => Some(Self::Classic),
      _ => None,
    }
  }

  pub fn label(self, lang: Language) -> &'static str {
    match self {
      Self::Llm => lang.pick("LLM 语言模型", "LLM"),
      Self::Vlm => lang.pick("VLM 视觉多模态", "VLM"),
      Self::Audio => lang.pick("Audio 语音 (ASR/TTS)", "Audio (ASR/TTS)"),
      Self::Diffusion => lang.pick("Diffusion 图像/视频", "Diffusion"),
      Self::Embedding => lang.pick("Embedding 向量/重排", "Embedding"),
      Self::Detection => lang.pick("目标检测 (YOLO/DETR)", "Detection"),
      Self::Segmentation => lang.pick("图像分割 / 医学 (SAM/U-Net)", "Segmentation"),
      Self::Classic => lang.pick("经典网络与基石", "Classic & Base"),
    }
  }

  pub fn short_badge(self) -> &'static str {
    match self {
      Self::Llm => "LLM",
      Self::Vlm => "VLM",
      Self::Audio => "Audio",
      Self::Diffusion => "Diffusion",
      Self::Embedding => "Embedding",
      Self::Detection => "Detection",
      Self::Segmentation => "Segmentation",
      Self::Classic => "Classic",
    }
  }

  /// 从已解析的 `ModelBlueprint` 自动推断模态（当 YAML 或表单未显式指定时兜底）。
  pub fn infer_from_blueprint(bp: &ModelBlueprint) -> Self {
    match bp.family {
      ArchFamily::AudioAsr | ArchFamily::SpeechTtsOmni => return Self::Audio,
      ArchFamily::EncoderEmbedding => return Self::Embedding,
      ArchFamily::DiffusionPipeline => return Self::Diffusion,
      ArchFamily::DeclarativeGraph => return Self::Classic,
      _ => {}
    }
    if bp.diffusion.is_some() {
      return Self::Diffusion;
    }
    if bp.vision.is_some() {
      return Self::Vlm;
    }
    if bp.audio.is_some() {
      return Self::Audio;
    }
    let joined = format!("{} {}", bp.architectures.join(" "), bp.model_type).to_ascii_lowercase();
    if joined.contains("whisper")
      || joined.contains("audio")
      || joined.contains("speech")
      || joined.contains("tts")
      || joined.contains("omni")
      || joined.contains("cosyvoice")
      || joined.contains("kokoro")
    {
      Self::Audio
    } else if joined.contains("flux")
      || joined.contains("diffusion")
      || joined.contains("unet")
      || joined.contains("dit")
      || joined.contains("wan")
      || joined.contains("hunyuanvideo")
    {
      Self::Diffusion
    } else if joined.contains("embedding")
      || joined.contains("rerank")
      || (joined.contains("bert") && !joined.contains("albert_causal"))
      || joined.contains("xlm-roberta")
    {
      Self::Embedding
    } else {
      Self::Llm
    }
  }
}

/// 图鉴条目数据来源标记（用于管理员界面区分静态内置 vs 在线数据库收录）。
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum RecordOrigin {
  #[default]
  Builtin,
  Database,
  Override,
}

/// 管理员新增或编辑图鉴模型的完整输入载荷（前后端共享，仅 `accounts` 档位编译进 WASM 以保持 `basic` 零膨胀）。
#[cfg(any(feature = "server", feature = "accounts", test))]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CuratedModelUpsertInput {
  pub slug: String,
  pub name: String,
  pub vendor: String,
  pub modality: ModelModality,
  pub series: String,
  pub release_date: String,
  pub hf_repo: String,
  pub ms_repo: Option<String>,
  pub featured: bool,
  pub order: i32,
  pub summary: String,
  pub summary_en: String,
  pub tags: Vec<String>,
  pub highlights: Vec<String>,
  pub highlights_en: Vec<String>,
  pub readme_md: Option<String>,
  #[serde(with = "crate::arch::blueprint_wire")]
  pub blueprint: ModelBlueprint,
}

/// AI 润色返回的双语摘要与核心亮点结构。
#[cfg(any(feature = "server", feature = "accounts", test))]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CuratedModelAiPolish {
  pub summary: String,
  pub summary_en: String,
  pub highlights: Vec<String>,
  pub highlights_en: Vec<String>,
}

/// 校验图鉴模型 `slug`：仅允许小写字母、数字、`-`、`_`，长度 `2..=96`。
#[cfg(any(feature = "server", feature = "accounts", test))]
pub fn is_safe_model_slug(slug: &str) -> bool {
  let len = slug.len();
  (2..=96).contains(&len)
    && slug.as_bytes().first().is_some_and(|b| b.is_ascii_lowercase() || b.is_ascii_digit())
    && slug.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || matches!(c, '-' | '_'))
}

/// 从 `owner/model` 推导整洁的 URL `slug`（如 `Qwen/Qwen2.5-VL-7B-Instruct` → `qwen2-5-vl-7b`）。
#[cfg(any(feature = "server", feature = "accounts", test))]
pub fn slugify_repo_id(repo_id: &str) -> String {
  let raw_name = repo_id.split('/').nth(1).unwrap_or(repo_id).trim();
  let mut s = String::with_capacity(raw_name.len());
  let mut prev_dash = false;
  for ch in raw_name.chars() {
    if ch.is_ascii_alphanumeric() {
      s.push(ch.to_ascii_lowercase());
      prev_dash = false;
    } else if !prev_dash && !s.is_empty() {
      s.push('-');
      prev_dash = true;
    }
  }
  while s.ends_with('-') {
    s.pop();
  }
  for suffix in ["-instruct", "-chat", "-hf"] {
    if s.len() > suffix.len() + 4 && s.ends_with(suffix) {
      s.truncate(s.len() - suffix.len());
      break;
    }
  }
  if s.len() < 2 {
    "custom-model".to_string()
  } else if s.len() > 80 {
    s.truncate(80);
    s.trim_end_matches('-').to_string()
  } else {
    s
  }
}

/// 根据仓库组织名推断人类可读厂商名称。
#[cfg(any(feature = "server", feature = "accounts", test))]
pub fn infer_vendor_from_repo(repo_id: &str) -> String {
  let owner = repo_id.split('/').next().unwrap_or("").trim();
  match owner.to_ascii_lowercase().as_str() {
    "deepseek-ai" => "DeepSeek".to_string(),
    "qwen" => "Alibaba Qwen".to_string(),
    "wan-ai" => "Alibaba Wan".to_string(),
    "meta-llama" | "facebook" => "Meta".to_string(),
    "google" => "Google DeepMind".to_string(),
    "mistralai" => "Mistral AI".to_string(),
    "thudm" | "zai-org" | "zhipuai" => "Zhipu AI (THUDM)".to_string(),
    "openai" => "OpenAI".to_string(),
    "baai" => "BAAI".to_string(),
    "01-ai" => "01.AI".to_string(),
    "internlm" | "opengvlab" => "Shanghai AI Lab".to_string(),
    "black-forest-labs" => "Black Forest Labs".to_string(),
    "stabilityai" => "Stability AI".to_string(),
    "tencent" => "Tencent Hunyuan".to_string(),
    "microsoft" => "Microsoft".to_string(),
    "cohereforai" | "cohere" => "Cohere".to_string(),
    "state-spaces" => "State Spaces (Mamba)".to_string(),
    "ai21labs" => "AI21 Labs".to_string(),
    "blinkdl" | "rwkv" => "RWKV".to_string(),
    "minimaxai" => "MiniMax".to_string(),
    "" | "custom" => "Community".to_string(),
    _ => owner.to_string(),
  }
}

/// 标准化系列分类标识（保留显式声明的合法演进系列如 `yolo`/`unet`/`sam`/`classic`，缺省按仓库自动推断）。
#[cfg(any(feature = "server", feature = "accounts", test))]
pub fn normalize_series_id(raw: &str, repo_id: &str) -> String {
  let s = raw.trim().to_ascii_lowercase();
  if !s.is_empty()
    && s.len() <= 32
    && s.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || matches!(c, '-' | '_'))
  {
    return s;
  }
  let r = repo_id.to_ascii_lowercase();
  if r.contains("deepseek") {
    "deepseek".to_string()
  } else if r.contains("qwen") {
    "qwen".to_string()
  } else if r.contains("llama") {
    "llama".to_string()
  } else if r.contains("yolo") {
    "yolo".to_string()
  } else if r.contains("unet") {
    "unet".to_string()
  } else if r.contains("sam") {
    "sam".to_string()
  } else {
    "other".to_string()
  }
}

/// 由元数据字段与已解析的 `ModelBlueprint` 统一构建 `CuratedModelSummary`（单点真值计算）。
#[cfg(any(feature = "server", test))]
#[allow(clippy::too_many_arguments)]
pub fn build_curated_summary(
  slug: String,
  name: String,
  vendor: String,
  modality: Option<ModelModality>,
  series: String,
  release_date: String,
  hf_repo: String,
  ms_repo: Option<String>,
  featured: bool,
  order: i32,
  summary: String,
  summary_en: String,
  tags: Vec<String>,
  highlights: Vec<String>,
  highlights_en: Vec<String>,
  blueprint: &ModelBlueprint,
  origin: RecordOrigin,
) -> CuratedModelSummary {
  let resolved_modality =
    modality.unwrap_or_else(|| ModelModality::infer_from_blueprint(blueprint));
  let main_seg = blueprint.segments.last();
  let kv_compression_ratio = main_seg.map(|s| s.attention.kv_compression_ratio()).unwrap_or(1.0);
  let attention_short =
    main_seg.map(|s| s.attention.short_label()).unwrap_or_else(|| "Attention".to_string());
  let ffn_short = main_seg.map(|s| s.ffn.short_label()).unwrap_or_else(|| "FFN".to_string());
  let series_norm = normalize_series_id(&series, &hf_repo);
  let has_mla = matches!(main_seg.map(|s| &s.attention), Some(AttentionSpec::DeepSeekMla { .. }));
  let has_hybrid_attn = main_seg.is_some_and(|s| s.schedule_note.is_some())
    || series_norm == "ssm"
    || tags.iter().any(|t| {
      matches!(
        t.to_ascii_lowercase().as_str(),
        "ssm" | "mamba" | "mamba-2" | "rwkv" | "linear-attention" | "swa"
      )
    });
  let eval_ctx = 32_768u64.min(blueprint.max_position_embeddings.max(4096));
  let fp8_vram_bytes = blueprint.estimate_vram(PrecisionPreset::Fp8, eval_ctx, 1).total_bytes;

  CuratedModelSummary {
    slug,
    name,
    vendor,
    modality: resolved_modality,
    origin,
    series: series_norm,
    release_date,
    hf_repo,
    ms_repo: ms_repo.map(|s| s.trim().to_string()).filter(|s| !s.is_empty()),
    featured,
    order,
    summary,
    summary_en,
    tags,
    highlights,
    highlights_en,
    family_badge: blueprint.family_badge_label().to_string(),
    total_params: blueprint.params.total_params,
    active_params: blueprint.params.active_params,
    num_layers: blueprint.num_hidden_layers,
    hidden_size: blueprint.hidden_size,
    max_context: blueprint.max_position_embeddings,
    kv_compression_ratio,
    attention_short,
    ffn_short,
    has_moe: blueprint.has_moe(),
    has_vision: blueprint.vision.is_some(),
    has_mla,
    has_hybrid_attn,
    fp8_vram_bytes,
    is_declarative: blueprint.is_declarative_graph(),
    series_order: 0,
    version_label: String::new(),
    delta_from_prev: String::new(),
  }
}

/// 从 `ModelBlueprint` 零延迟自动合成高质量双语图鉴草稿（服务端与测试使用完整双语模板）。
#[cfg(any(feature = "server", test))]
pub fn draft_from_blueprint(bp: &ModelBlueprint) -> CuratedModelUpsertInput {
  let repo_id = if bp.repo_id.trim().is_empty() || bp.repo_id == "custom/pasted-config" {
    format!("custom/{}", bp.model_type)
  } else {
    bp.repo_id.trim().to_string()
  };

  let slug = slugify_repo_id(&repo_id);
  let raw_title = repo_id.split('/').nth(1).unwrap_or(&slug);
  let total_str = format_count(bp.params.total_params);
  let active_str = format_count(bp.params.active_params);
  let is_sparse = bp.params.active_params < bp.params.total_params;

  let name = if raw_title.chars().any(|c| c.is_ascii_digit())
    && raw_title.to_ascii_lowercase().contains('b')
  {
    raw_title.to_string()
  } else {
    format!("{raw_title} ({total_str})")
  };

  let vendor = infer_vendor_from_repo(&repo_id);
  let modality = ModelModality::infer_from_blueprint(bp);
  let series = normalize_series_id("", &repo_id);

  let main_seg = bp.segments.last();
  let attn_short = main_seg.map(|s| s.attention.short_label()).unwrap_or_else(|| "GQA".to_string());
  let ffn_short = main_seg.map(|s| s.ffn.short_label()).unwrap_or_else(|| "SwiGLU".to_string());
  let kv_ratio = main_seg.map(|s| s.attention.kv_compression_ratio()).unwrap_or(1.0);
  let ctx_k = bp.max_position_embeddings / 1024;

  // 自动标签提取
  let mut tags = Vec::new();
  if modality != ModelModality::Llm {
    tags.push(modality.as_str().to_string());
  }
  if let Some(ref diff) = bp.diffusion {
    tags.push(if diff.denoiser_kind.contains("UNet") { "unet" } else { "dit" }.to_string());
    tags.push(if diff.is_video { "video-gen" } else { "image-gen" }.to_string());
    if diff.scheduler_summary.to_ascii_lowercase().contains("flow") {
      tags.push("flow-matching".to_string());
    }

    let media_zh = if diff.is_video { "视频" } else { "图像" };
    let media_en = if diff.is_video { "video" } else { "image" };
    let conds: Vec<String> = diff
      .components
      .iter()
      .filter(|c| c.role == "conditioner")
      .map(|c| format!("{} ({})", c.class_name, format_count(c.estimated_params)))
      .collect();
    let cond_str = if conds.is_empty() { "Text Encoder".to_string() } else { conds.join(" + ") };
    let vae_ratio = if diff.is_video && diff.temporal_compression > 1 {
      format!(
        "{}×{}×{} 时空压缩",
        diff.temporal_compression, diff.spatial_compression, diff.spatial_compression
      )
    } else {
      format!("{}×{} 空间压缩", diff.spatial_compression, diff.spatial_compression)
    };

    let summary = format!(
      "{vendor} 开源多组件{media_zh}生成管线（{}）：全管线常驻参数 {total_str}（去噪主干单步激活 {active_str}），采用 {} 层 × {} 维 {} 与 {}。",
      diff.pipeline_class, bp.num_hidden_layers, bp.hidden_size, diff.denoiser_kind, diff.scheduler_summary
    );
    let summary_en = format!(
      "{vendor} multi-component {media_en} generation pipeline ({}): {total_str} total pipeline params ({active_str} active denoiser), featuring a {}L × {}d {} with {}.",
      diff.pipeline_class, bp.num_hidden_layers, bp.hidden_size, diff.denoiser_kind, diff.scheduler_summary
    );
    let highlights = vec![
      format!(
        "多编码器语义条件与采样调度：{cond_str} 提供语义引导，配合 {} 在潜空间 `{}` 上迭代去噪。",
        diff.scheduler_summary, diff.default_latent_shape
      ),
      format!(
        "去噪主干拓扑：{}（{} 层 × {} 维，{}，块化 {}），单步前向激活 {active_str} 参数且持久 KV Cache 为 0 B。",
        diff.denoiser_kind, bp.num_hidden_layers, bp.hidden_size, attn_short, diff.patch_summary
      ),
      format!(
        "潜空间编解码器（VAE）：{} 通道潜表征，{}，仅在迭代去噪结束后执行一次像素空间解码。",
        diff.latent_channels, vae_ratio
      ),
    ];
    let highlights_en = vec![
      format!(
        "Conditioning & Scheduler: {cond_str} paired with {} over latent `{}`.",
        diff.scheduler_summary, diff.default_latent_shape
      ),
      format!(
        "Denoiser Backbone: {} ({}L × {}d, {}, {}), activating {active_str} per step with 0 B KV cache.",
        diff.denoiser_kind, bp.num_hidden_layers, bp.hidden_size, attn_short, diff.patch_summary
      ),
      format!(
        "Latent VAE Codec: {}-channel latent space ({}x spatial, {}x temporal compression).",
        diff.latent_channels, diff.spatial_compression, diff.temporal_compression
      ),
    ];

    return CuratedModelUpsertInput {
      slug,
      name,
      vendor,
      modality,
      series,
      release_date: String::new(),
      hf_repo: repo_id,
      ms_repo: None,
      featured: false,
      order: 50,
      summary,
      summary_en,
      tags,
      highlights,
      highlights_en,
      readme_md: None,
      blueprint: bp.clone(),
    };
  }
  if bp.has_moe() {
    tags.push("moe".to_string());
  } else {
    tags.push("dense".to_string());
  }
  match main_seg.map(|s| &s.attention) {
    Some(AttentionSpec::DeepSeekMla { .. }) => tags.push("mla".to_string()),
    Some(AttentionSpec::LinearRecurrent { .. }) => {
      tags.push("ssm".to_string());
      tags.push("linear-attn".to_string());
    }
    Some(AttentionSpec::StandardGqa {
      qk_norm,
      nope_layer_interval,
      sliding_window_pattern,
      is_causal,
      has_cross_attn,
      ..
    }) => {
      tags.push(if *is_causal { "gqa" } else { "bidirectional" }.to_string());
      if *qk_norm {
        tags.push("qk-norm".to_string());
      }
      if *has_cross_attn {
        tags.push("cross-attn".to_string());
      }
      if nope_layer_interval.is_some() {
        tags.push("irope".to_string());
      }
      if sliding_window_pattern.is_some() {
        tags.push("hybrid-swa".to_string());
      }
    }
    None => {}
  }
  if bp.has_linear_recurrent() && !tags.iter().any(|t| t == "ssm") {
    tags.push("ssm".to_string());
  }
  if bp.num_nextn_predict_layers > 0 {
    tags.push("mtp".to_string());
  }
  if bp.quantization_summary.as_deref().is_some_and(|q| q.to_ascii_lowercase().contains("fp8")) {
    tags.push("fp8".to_string());
  }

  let param_desc_zh = if is_sparse {
    format!("总参数 {total_str}（单 Token 激活 {active_str}）稀疏架构")
  } else {
    format!("{total_str} 稠密参数架构")
  };
  let param_desc_en = if is_sparse {
    format!("{total_str} total / {active_str} active sparse architecture")
  } else {
    format!("{total_str} dense architecture")
  };

  let mm_prefix_zh = match (&bp.vision, &bp.audio) {
    (Some(v), Some(a)) => format!(
      "融合 {} 层视觉塔与 {} 层声学塔（{} Mel @ {}kHz）的全模态 ",
      v.num_hidden_layers,
      a.num_layers,
      a.num_mel_bins,
      a.sampling_rate_hz / 1000
    ),
    (Some(v), None) => {
      format!("融合 {} 层视觉编码塔（{}d）与 ", v.num_hidden_layers, v.hidden_size)
    }
    (None, Some(a)) => format!(
      "配备 {} 层声学编码器（{} Mel @ {}kHz）与 ",
      a.num_layers,
      a.num_mel_bins,
      a.sampling_rate_hz / 1000
    ),
    (None, None) => String::new(),
  };
  let mm_prefix_en = match (&bp.vision, &bp.audio) {
    (Some(v), Some(a)) => format!(
      "Integrates {}L vision + {}L audio ({} Mel) towers with a ",
      v.num_hidden_layers, a.num_layers, a.num_mel_bins
    ),
    (Some(v), None) => {
      format!("Integrates a {}L vision encoder ({}d) with a ", v.num_hidden_layers, v.hidden_size)
    }
    (None, Some(a)) => format!(
      "Pairs a {}L audio encoder ({} Mel @ {}kHz) with a ",
      a.num_layers,
      a.num_mel_bins,
      a.sampling_rate_hz / 1000
    ),
    (None, None) => "Features a ".to_string(),
  };

  let ctx_label_zh = if ctx_k >= 1 {
    format!("{ctx_k}K 上下文")
  } else {
    format!("{} 序列窗口", bp.max_position_embeddings)
  };
  let ctx_label_en = if ctx_k >= 1 {
    format!("{ctx_k}K max context")
  } else {
    format!("{}-token window", bp.max_position_embeddings)
  };

  let summary = format!(
    "{vendor} 开源模型，采用{mm_prefix_zh}{param_desc_zh}：{} 层 × {} 维主干，搭载 {attn_short} 与 {ffn_short}，支持 {ctx_label_zh}。",
    bp.num_hidden_layers, bp.hidden_size
  );
  let summary_en = format!(
    "{mm_prefix_en}{param_desc_en} by {vendor}: {}L × {}d backbone with {attn_short}, {ffn_short}, and {ctx_label_en}.",
    bp.num_hidden_layers, bp.hidden_size
  );

  let mut highlights = Vec::new();
  let mut highlights_en = Vec::new();

  // 亮点 1：注意力 / 线性状态空间机制与推理缓存效率
  if let Some(seg) = main_seg {
    match &seg.attention {
      AttentionSpec::DeepSeekMla { kv_lora_rank, q_lora_rank, qk_rope_head_dim, .. } => {
        let q_note = match q_lora_rank {
          Some(qr) => format!("Q 低秩压缩 (rank={qr}) + "),
          None => "直投射 Query + ".to_string(),
        };
        highlights.push(format!(
          "MLA 潜空间注意力：{q_note}KV 联合低秩压缩 (kv_lora_rank={kv_lora_rank}, rope={qk_rope_head_dim})，KV 显存相比标准 MHA 压缩 {kv_ratio:.1}×。"
        ));
        highlights_en.push(format!(
          "Multi-head Latent Attention: Compresses KV cache to {kv_lora_rank}d latent + {qk_rope_head_dim}d RoPE ({kv_ratio:.1}x smaller than MHA)."
        ));
      }
      AttentionSpec::LinearRecurrent {
        kind, num_heads, head_dim, state_dim, conv_kernel, ..
      } => {
        highlights.push(format!(
          "线性状态空间混合层：{kind}（{num_heads}H×{head_dim}d, state={state_dim}d, conv={conv_kernel}），推理期仅驻留 O(1) 恒定循环状态，随上下文增长零膨胀。"
        ));
        highlights_en.push(format!(
          "Linear / SSM Sequence Mixer: {kind} ({num_heads}H×{head_dim}d, state={state_dim}d, conv={conv_kernel}) with constant O(1) recurrent state across sequence lengths."
        ));
      }
      AttentionSpec::StandardGqa {
        num_q_heads,
        num_kv_heads,
        head_dim,
        qk_norm,
        is_causal,
        has_cross_attn,
        ..
      } => {
        if !*is_causal {
          highlights.push(format!(
            "双向全注意力编码器：{num_q_heads} 头（head_dim={head_dim}）无因果掩码并行交互，单次前向编码持久 KV Cache 为 0 B。"
          ));
          highlights_en.push(format!(
            "Bidirectional Full Attention: {num_q_heads} heads (head_dim={head_dim}) encode tokens in parallel with 0 B persistent KV cache."
          ));
        } else {
          let norm_zh = if *qk_norm {
            "，配每头 QK-RMSNorm"
          } else if *has_cross_attn {
            "，并附带编码器-解码器 Cross-Attention"
          } else {
            ""
          };
          let norm_en = if *qk_norm {
            " with per-head QK-RMSNorm"
          } else if *has_cross_attn {
            " plus Encoder-Decoder Cross-Attention"
          } else {
            ""
          };
          highlights.push(format!(
            "注意力子层配置：{num_q_heads} 个 Query 头对 {num_kv_heads} 个 KV 头（head_dim={head_dim}），实现 {kv_ratio:.1}× KV Cache 压缩{norm_zh}。"
          ));
          highlights_en.push(format!(
            "Attention Topology: {num_q_heads} Q heads to {num_kv_heads} KV heads (head_dim={head_dim}, {kv_ratio:.1}x KV compression){norm_en}."
          ));
        }
      }
    }

    // 亮点 2：FFN / MoE 与异构层段
    match &seg.ffn {
      FfnSpec::SparseMoe {
        num_routed_experts,
        top_k,
        moe_intermediate_size,
        num_shared_experts,
        ..
      } => {
        let seg_note = if bp.segments.len() > 1 {
          format!("异构 {} 段层栈，主干每层", bp.segments.len())
        } else {
          format!("全栈统一 {} 层稀疏路由，每层", bp.num_hidden_layers)
        };
        highlights.push(format!(
          "{seg_note}由 {num_shared_experts} 个共享专家 + {num_routed_experts} 个路由专家选 Top-{top_k}（单专家维 {moe_intermediate_size}）组成，单 Token 仅激活 {active_str} 参数。"
        ));
        highlights_en.push(format!(
          "Sparse MoE FFN: {num_shared_experts} shared + Top-{top_k} of {num_routed_experts} routed experts (I={moe_intermediate_size}), activating {active_str} of {total_str} params per token."
        ));
      }
      FfnSpec::DenseGated { intermediate_size: 0, .. } => {
        highlights.push(format!(
          "内融合门控 SSM 架构：全栈 {} 层将通道扩展与门控非线性融合进状态空间算子内部，省去独立 MLP 子层。",
          bp.num_hidden_layers
        ));
        highlights_en.push(format!(
          "Fused Gated SSM Block: Integrates channel expansion and SiLU gating directly inside {} SSM layers without separate MLP blocks.",
          bp.num_hidden_layers
        ));
      }
      FfnSpec::DenseGated { intermediate_size, act_fn, fused_gate_up, is_gated, .. } => {
        let ratio = (*intermediate_size as f64) / (bp.hidden_size.max(1) as f64);
        let kind_zh = if *is_gated { "稠密门控前馈层" } else { "标准双矩阵前馈层" };
        let kind_en = if *is_gated { "Dense Gated FFN" } else { "Standard 2-Layer MLP" };
        let fused_note = if *fused_gate_up { "（单矩阵融合投影）" } else { "" };
        highlights.push(format!(
          "{kind_zh}：{act_fn} 激活函数，中间维 {intermediate_size}（约为隐藏维 {} 的 {ratio:.2}×）{fused_note}。",
          bp.hidden_size
        ));
        highlights_en.push(format!(
          "{kind_en}: {act_fn} activation with intermediate_size={intermediate_size} ({ratio:.2}x expansion over D={}).",
          bp.hidden_size
        ));
      }
    }
  }

  // 亮点 3：多模态声学/视觉塔、特殊输出头或交替层调度
  if let OutputHeadSpec::SpeechSynthesizer {
    ref talker_summary,
    ref codec_summary,
    sampling_rate_hz,
    estimated_params,
    ..
  } = bp.output_head
  {
    let t_zh = talker_summary.as_deref().unwrap_or("声学离散码本头");
    highlights.push(format!(
      "端到端语音合成管线：{t_zh} → {codec_summary}（附加约 {} 参数），直接输出 {} kHz 高保真波形。",
      format_count(estimated_params),
      sampling_rate_hz / 1000
    ));
    highlights_en.push(format!(
      "Speech Synthesis Pipeline: {t_zh} -> {codec_summary} (~{} params) generating {} kHz audio waveforms.",
      format_count(estimated_params),
      sampling_rate_hz / 1000
    ));
  } else if let Some(ref a) = bp.audio {
    highlights.push(format!(
      "语音声学编码塔：{} Mel @ {} kHz 前端接 {}（{} 层 × {} 维，约 {} 参数）。",
      a.num_mel_bins,
      a.sampling_rate_hz / 1000,
      a.model_type,
      a.num_layers,
      a.hidden_size,
      format_count(a.estimated_params)
    ));
    highlights_en.push(format!(
      "Audio Encoder Tower: {} Mel @ {} kHz frontend into {} ({}L × {}d, ~{} params).",
      a.num_mel_bins,
      a.sampling_rate_hz / 1000,
      a.model_type,
      a.num_layers,
      a.hidden_size,
      format_count(a.estimated_params)
    ));
  } else if let Some(ref v) = bp.vision {
    highlights.push(format!(
      "原生多模态视觉支路：内置 {}（{} 层 × {} 维），参数约 {}，经多模态投影接入语言主干。",
      v.model_type,
      v.num_hidden_layers,
      v.hidden_size,
      format_count(v.estimated_params)
    ));
    highlights_en.push(format!(
      "Vision Tower: Embeds {} ({}L × {}d, ~{} params) projected into the language token stream.",
      v.model_type,
      v.num_hidden_layers,
      v.hidden_size,
      format_count(v.estimated_params)
    ));
  } else if let OutputHeadSpec::EmbeddingPool { ref pooling_mode, output_dim, normalize } =
    bp.output_head
  {
    let norm_zh = if normalize { " + L2 单位球归一化" } else { "" };
    highlights.push(format!(
      "句向量池化输出头：采用 {pooling_mode} 池化{norm_zh}产出 {output_dim} 维稠密语义表征，词表规模 {}。",
      bp.vocab_size
    ));
    highlights_en.push(format!(
      "Embedding Pooling Head: {pooling_mode} pooling{} producing {output_dim}d vectors (vocab_size={}).",
      if normalize { " + L2 normalization" } else { "" },
      bp.vocab_size
    ));
  } else if let Some(note) = main_seg.and_then(|s| s.schedule_note.as_deref()) {
    highlights.push(note.to_string());
    highlights_en.push(format!(
      "Hybrid Layer Schedule across {} layers with vocab_size={} and {ctx_label_en}.",
      bp.num_hidden_layers, bp.vocab_size
    ));
  } else {
    let tie_zh = if bp.tie_word_embeddings {
      "输入输出共享词表权重"
    } else {
      "独立 Embedding 与 LM Head"
    };
    let tie_en = if bp.tie_word_embeddings {
      "tied input/output embeddings"
    } else {
      "untied Embedding and LM Head"
    };
    highlights.push(format!(
      "词表与长上下文：词表规模 {}（{tie_zh}），位置编码 {}，最大上下文窗口 {} Tokens。",
      bp.vocab_size, bp.rope_summary, bp.max_position_embeddings
    ));
    highlights_en.push(format!(
      "Vocabulary & Context: vocab_size={} ({tie_en}), {}, max_position_embeddings={}.",
      bp.vocab_size, bp.rope_summary, bp.max_position_embeddings
    ));
  }

  if bp.family == ArchFamily::SafetensorsTree {
    highlights.push("由 Safetensors L2 二进制分片头零权重采样直接还原张量拓扑。".to_string());
    highlights_en.push(
      "Reconstructed directly from L2 Safetensors shard headers without downloading weights."
        .to_string(),
    );
  }

  CuratedModelUpsertInput {
    slug,
    name,
    vendor,
    modality,
    series,
    release_date: String::new(),
    hf_repo: repo_id,
    ms_repo: None,
    featured: false,
    order: 50,
    summary,
    summary_en,
    tags,
    highlights,
    highlights_en,
    readme_md: None,
    blueprint: bp.clone(),
  }
}

/// 客户端 WASM 轻量版草稿初始化（完整多模态双语文案由服务端 `admin_fetch_model_draft` / `admin_ai_polish_model_draft` 生成，
/// 避免将 450+ 行管理员文案模板打入全站公共 WASM 包）。
#[cfg(all(feature = "accounts", not(feature = "server"), not(test)))]
pub fn draft_from_blueprint(bp: &ModelBlueprint) -> CuratedModelUpsertInput {
  let repo_id = if bp.repo_id.trim().is_empty() || bp.repo_id == "custom/pasted-config" {
    format!("custom/{}", bp.model_type)
  } else {
    bp.repo_id.trim().to_string()
  };
  let slug = slugify_repo_id(&repo_id);
  let raw_title = repo_id.split('/').nth(1).unwrap_or(&slug);
  let total_str = format_count(bp.params.total_params);
  let active_str = format_count(bp.params.active_params);
  let vendor = infer_vendor_from_repo(&repo_id);
  let modality = ModelModality::infer_from_blueprint(bp);
  let series = normalize_series_id("", &repo_id);
  let name = if raw_title.chars().any(|c| c.is_ascii_digit())
    && raw_title.to_ascii_lowercase().contains('b')
  {
    raw_title.to_string()
  } else {
    format!("{raw_title} ({total_str})")
  };
  let mut tags = Vec::new();
  if modality != ModelModality::Llm {
    tags.push(modality.as_str().to_string());
  }
  tags.push(if bp.has_moe() { "moe" } else { "dense" }.to_string());
  let summary = format!(
    "{vendor} 开源模型（{total_str} 总参 / {active_str} 激活）：{} 层 × {} 维 {} 架构。",
    bp.num_hidden_layers,
    bp.hidden_size,
    bp.family_badge_label()
  );
  let summary_en = format!(
    "{vendor} open model ({total_str} total / {active_str} active): {}L × {}d {} architecture.",
    bp.num_hidden_layers,
    bp.hidden_size,
    bp.family_badge_label()
  );
  CuratedModelUpsertInput {
    slug,
    name,
    vendor,
    modality,
    series,
    release_date: String::new(),
    hf_repo: repo_id,
    ms_repo: None,
    featured: false,
    order: 50,
    summary,
    summary_en,
    tags,
    highlights: vec![format!(
      "{}L × {}d · {}",
      bp.num_hidden_layers, bp.hidden_size, bp.rope_summary
    )],
    highlights_en: vec![format!(
      "{}L × {}d · {}",
      bp.num_hidden_layers, bp.hidden_size, bp.rope_summary
    )],
    readme_md: None,
    blueprint: bp.clone(),
  }
}

/// 纯函数：合并静态目录模型列表与数据库模型列表（同 `slug` 时数据库记录优先覆盖并标记 `Override`）。
#[cfg(any(feature = "server", test))]
pub fn merge_builtin_and_db_models(
  builtin: &[CuratedModelDetail],
  db_items: Vec<CuratedModelDetail>,
) -> Vec<CuratedModelDetail> {
  if db_items.is_empty() {
    return builtin.to_vec();
  }

  let mut by_slug: HashMap<String, CuratedModelDetail> =
    builtin.iter().map(|d| (d.meta.slug.clone(), d.clone())).collect();

  for mut db_entry in db_items {
    let existed = by_slug.contains_key(&db_entry.meta.slug);
    db_entry.meta.origin = if existed { RecordOrigin::Override } else { RecordOrigin::Database };
    by_slug.insert(db_entry.meta.slug.clone(), db_entry);
  }

  let mut merged: Vec<CuratedModelDetail> = by_slug.into_values().collect();
  merged
    .sort_by(|a, b| a.meta.order.cmp(&b.meta.order).then_with(|| a.meta.slug.cmp(&b.meta.slug)));
  merged
}

// ─────────────────────────────────────────────────────────────────────────────
// 服务端数据库缓存、双源合并读取与后台写接口
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "server")]
static ATLAS_DB_STAMP: AtomicU64 = AtomicU64::new(1);

#[cfg(feature = "server")]
type DbCacheCell = tokio::sync::RwLock<Option<(u64, Vec<CuratedModelDetail>)>>;

#[cfg(feature = "server")]
fn db_cache_cell() -> &'static DbCacheCell {
  static CELL: OnceLock<DbCacheCell> = OnceLock::new();
  CELL.get_or_init(|| tokio::sync::RwLock::new(None))
}

/// 当前数据库模型图鉴变更代数（供 `conditional_cache.rs` 混入弱 ETag 指纹，保证写后立即刷新）。
#[cfg(feature = "server")]
pub fn atlas_db_stamp() -> u64 {
  ATLAS_DB_STAMP.load(Ordering::Relaxed)
}

/// 使数据库模型图鉴内存缓存立即失效（增删改及 `SIGHUP` 时调用）。
#[cfg(feature = "server")]
pub fn invalidate_db_models_cache() {
  ATLAS_DB_STAMP.fetch_add(1, Ordering::SeqCst);
}

#[cfg(feature = "server")]
pub fn config_fingerprint_hex(raw: &str) -> String {
  // 确定性 FNV-1a 64-bit 指纹（零外部哈希依赖，用于检测上游 config.json 内容是否变动）
  let mut h: u64 = 0xcbf2_9ce4_8422_2325;
  for &b in raw.as_bytes() {
    h ^= u64::from(b);
    h = h.wrapping_mul(0x0000_0100_0000_01b3);
  }
  format!("{h:016x}")
}

#[cfg(feature = "server")]
fn row_to_detail(
  row: app_core::entities::curated_model::Model,
) -> Result<CuratedModelDetail, String> {
  let mut blueprint: ModelBlueprint = serde_json::from_str(&row.blueprint_json)
    .map_err(|e| format!("解析数据库模型 `{}` 的 blueprint_json 失败: {e}", row.slug))?;
  if blueprint.verification_wire.is_none() {
    if let Some(ref rep_json) = row.verification_report_json {
      if let Ok(rep) = serde_json::from_str::<crate::arch::VerificationReport>(rep_json) {
        blueprint.verification_wire = Some(rep.encode_wire());
      }
    }
  }
  let tags: Vec<String> = serde_json::from_str(&row.tags_json).unwrap_or_default();
  let highlights: Vec<String> = serde_json::from_str(&row.highlights_json).unwrap_or_default();
  let highlights_en: Vec<String> =
    serde_json::from_str(&row.highlights_en_json).unwrap_or_default();
  let modality = ModelModality::parse_str(&row.modality);

  let meta = build_curated_summary(
    row.slug,
    row.name,
    row.vendor,
    modality,
    row.series,
    row.release_date,
    row.hf_repo,
    row.ms_repo,
    row.featured,
    row.sort_order,
    row.summary,
    row.summary_en,
    tags,
    highlights,
    highlights_en,
    &blueprint,
    RecordOrigin::Database,
  );
  let readme_md = row.readme_md.filter(|s| !s.trim().is_empty());
  Ok(CuratedModelDetail { meta, blueprint, readme_md })
}

#[cfg(feature = "server")]
async fn load_db_curated_models() -> Vec<CuratedModelDetail> {
  use app_core::entities::curated_model;
  use sea_orm::{EntityTrait, QueryOrder};

  let Some(db) = app_core::db::pool::pool() else {
    return Vec::new();
  };

  let cur_stamp = atlas_db_stamp();
  if let Some((cached_stamp, ref list)) = *db_cache_cell().read().await {
    if cached_stamp == cur_stamp {
      return list.clone();
    }
  }

  let rows = match curated_model::Entity::find()
    .order_by_asc(curated_model::Column::SortOrder)
    .order_by_asc(curated_model::Column::Slug)
    .all(&db)
    .await
  {
    Ok(r) => r,
    Err(err) => {
      tracing::warn!(error = %err, "atlas_store: failed to query curated_models (falling back to builtin)");
      return Vec::new();
    }
  };

  let mut out = Vec::with_capacity(rows.len());
  for row in rows {
    match row_to_detail(row) {
      Ok(d) => out.push(d),
      Err(err) => {
        tracing::warn!(error = %err, "atlas_store: skipping malformed curated_models row")
      }
    }
  }

  let mut guard = db_cache_cell().write().await;
  *guard = Some((cur_stamp, out.clone()));
  out
}

/// 加载合并后的全部指定模型详情（静态 `models/` + PostgreSQL `curated_models`）。
#[cfg(feature = "server")]
pub async fn load_merged_curated_models_full() -> Vec<CuratedModelDetail> {
  let builtin = crate::atlas::load_curated_models_full();
  let db_items = load_db_curated_models().await;
  merge_builtin_and_db_models(&builtin, db_items)
}

/// 加载合并后的全部指定模型摘要列表（供页面、Sitemap 与搜索索引共用）。
#[cfg(feature = "server")]
pub async fn list_curated_models_merged() -> Vec<CuratedModelSummary> {
  load_merged_curated_models_full().await.into_iter().map(|d| d.meta).collect()
}

/// 供全文检索异步来源（`register_async_doc_source`）收集全部静态+数据库模型条目。
#[cfg(feature = "server")]
pub async fn search_docs() -> Vec<app_core::engines::doc_source::ExternalIndexedDoc> {
  let enabled = app_core::engines::module::default_module_engine().enabled_ids();
  if !enabled.iter().any(|s| s == "ai") {
    return Vec::new();
  }
  list_curated_models_merged()
    .await
    .into_iter()
    .map(|m| app_core::engines::doc_source::ExternalIndexedDoc {
      kind: "model".to_string(),
      ref_id: m.slug.clone(),
      title: format!("{} ({})", m.name, m.family_badge),
      body: format!(
        "{} {} {} {} {} {} {}",
        m.vendor,
        m.modality.short_badge(),
        m.hf_repo,
        m.summary,
        m.summary_en,
        m.tags.join(" "),
        m.highlights.join(" ")
      ),
      url: format!("/models/{}", m.slug),
      created_at: m.release_date,
    })
    .collect()
}

/// 校验并清理管理员提交的保存参数。
#[cfg(any(feature = "server", feature = "accounts", test))]
pub fn validate_upsert_input(
  mut input: CuratedModelUpsertInput,
) -> Result<CuratedModelUpsertInput, String> {
  input.slug = input.slug.trim().to_ascii_lowercase();
  if !is_safe_model_slug(&input.slug) {
    return Err("模型 Slug 仅允许小写字母、数字、连字符与下划线（长度 2–96）".to_string());
  }
  input.name = input.name.trim().to_string();
  if input.name.is_empty() || input.name.len() > 160 {
    return Err("模型名称不能为空且长度不超过 160 字符".to_string());
  }
  input.vendor = input.vendor.trim().to_string();
  if input.vendor.is_empty() {
    input.vendor = infer_vendor_from_repo(&input.hf_repo);
  }
  input.hf_repo = input.hf_repo.trim().to_string();
  if input.hf_repo.is_empty() || input.hf_repo.len() > 190 {
    return Err("仓库标识（owner/model）不能为空".to_string());
  }
  input.ms_repo = input.ms_repo.map(|s| s.trim().to_string()).filter(|s| !s.is_empty());
  input.release_date = input.release_date.trim().chars().take(32).collect();
  input.series = normalize_series_id(&input.series, &input.hf_repo);
  input.summary = input.summary.trim().to_string();
  if input.summary.is_empty() {
    return Err("请填写模型中文架构简介".to_string());
  }
  input.summary_en = input.summary_en.trim().to_string();
  input.tags = input
    .tags
    .into_iter()
    .map(|t| t.trim().trim_start_matches('#').to_ascii_lowercase())
    .filter(|t| !t.is_empty())
    .take(12)
    .collect();
  input.highlights = input
    .highlights
    .into_iter()
    .map(|h| h.trim().to_string())
    .filter(|h| !h.is_empty())
    .take(6)
    .collect();
  input.highlights_en = input
    .highlights_en
    .into_iter()
    .map(|h| h.trim().to_string())
    .filter(|h| !h.is_empty())
    .take(6)
    .collect();
  input.readme_md = input.readme_md.map(|s| s.trim().to_string()).filter(|s| !s.is_empty());
  if input.blueprint.params.total_params == 0 && input.blueprint.segments.is_empty() {
    return Err("模型架构蓝图为空，请先解析有效配置".to_string());
  }
  Ok(input)
}

#[cfg(feature = "server")]
fn extract_json_object_slice(raw: &str) -> Option<&str> {
  let trimmed = raw.trim();
  let start = trimmed.find('{')?;
  let end = trimmed.rfind('}')?;
  (end > start).then_some(&trimmed[start..=end])
}

/// 基于真实架构事实调用服务端配置的 LLM 生成双语架构导读（服务端纯函数，零幻觉数字）。
#[cfg(feature = "server")]
pub async fn polish_draft_via_llm(
  draft: &CuratedModelUpsertInput,
) -> Result<CuratedModelAiPolish, String> {
  use llm::LlmMessage;

  let Some(client) = llm::default_client_from_env() else {
    return Err(
      "服务端未配置 OPENAI_LLM_* 或 ANTHROPIC_LLM_*，当前已使用本地规则精算草稿".to_string(),
    );
  };

  let bp = &draft.blueprint;
  let main_seg = bp.segments.last();
  let facts = serde_json::json!({
    "repo_id": draft.hf_repo,
    "vendor": draft.vendor,
    "modality": draft.modality.as_str(),
    "family": bp.family_badge_label(),
    "total_params": format_count(bp.params.total_params),
    "active_params": format_count(bp.params.active_params),
    "num_layers": bp.num_hidden_layers,
    "hidden_size": bp.hidden_size,
    "vocab_size": bp.vocab_size,
    "tie_word_embeddings": bp.tie_word_embeddings,
    "max_position_embeddings": bp.max_position_embeddings,
    "rope": bp.rope_summary,
    "attention": main_seg.map(|s| s.attention.short_label()),
    "kv_compression_ratio": main_seg.map(|s| format!("{:.1}x", s.attention.kv_compression_ratio())),
    "ffn": main_seg.map(|s| s.ffn.short_label()),
    "segment_count": bp.segments.len(),
    "schedule_note": main_seg.and_then(|s| s.schedule_note.clone()),
    "vision": bp.vision,
    "audio": bp.audio,
    "output_head": bp.output_head,
    "mtp_layers": bp.num_nextn_predict_layers,
    "quantization": bp.quantization_summary,
  });

  let sys = "你是资深大模型系统架构师。请严格依据用户提供的模型架构 JSON 事实，生成用于「模型架构图鉴」的中英双语技术简介与 3 条硬核架构亮点。\
严禁编造 JSON 中未出现的数字；必须只输出合法 JSON 对象，格式为：\
{\"summary\":\"一句精炼中文架构定位（50-90字）\",\"summary_en\":\"Concise English architectural summary (1 sentence)\",\"highlights\":[\"中文亮点1\",\"中文亮点2\",\"中文亮点3\"],\"highlights_en\":[\"English highlight 1\",\"English highlight 2\",\"English highlight 3\"]}";

  let user_prompt = format!(
    "请为以下模型生成双语架构导读 JSON：\n{}",
    serde_json::to_string_pretty(&facts).unwrap_or_default()
  );

  let reply = client
    .chat(vec![LlmMessage::system(sys), LlmMessage::user(user_prompt)])
    .await
    .map_err(|e| format!("LLM 生成失败: {e}"))?;

  let json_slice = extract_json_object_slice(&reply)
    .ok_or_else(|| "LLM 返回内容未包含合法 JSON 对象".to_string())?;
  let parsed: CuratedModelAiPolish =
    serde_json::from_str(json_slice).map_err(|e| format!("解析 LLM JSON 响应失败: {e}"))?;

  if parsed.summary.trim().is_empty() || parsed.highlights.is_empty() {
    return Err("LLM 返回的摘要或亮点为空".to_string());
  }
  Ok(parsed)
}

// ─────────────────────────────────────────────────────────────────────────────
// 管理员专属 Server Functions（拉取草稿 / AI 润色 / 保存发布 / 删除，仅 `accounts` 档位启用）
// ─────────────────────────────────────────────────────────────────────────────

/// 管理员按 `owner/model` 从 HuggingFace / ModelScope 拉取蓝图并自动生成图鉴草稿（JSON 字符串，避免 CBOR 重复单态化 `ModelBlueprint`）。
#[cfg(feature = "accounts")]
#[post("/api/ai/models/fetch-draft")]
pub async fn admin_fetch_model_draft(
  repo: String,
  source: String,
  revision: String,
) -> Result<String, ServerFnError> {
  #[cfg(feature = "server")]
  {
    let _ = app_core::session::require_admin().await?;
    let bp = crate::server::resolve_model_blueprint_server(&repo, &source, &revision)
      .await
      .map_err(ServerFnError::new)?;
    let mut draft = draft_from_blueprint(&bp);
    // 若该 slug 已存在，沿用既有排序/精选/双语详情，仅刷新蓝图或允许管理员在此基础上编辑
    let existing = load_merged_curated_models_full()
      .await
      .into_iter()
      .find(|d| d.meta.slug == draft.slug || d.meta.hf_repo.eq_ignore_ascii_case(&bp.repo_id));
    if let Some(ex) = existing {
      draft.slug = ex.meta.slug;
      draft.name = ex.meta.name;
      draft.vendor = ex.meta.vendor;
      draft.modality = ex.meta.modality;
      draft.series = ex.meta.series;
      draft.release_date = ex.meta.release_date;
      draft.ms_repo = ex.meta.ms_repo;
      draft.featured = ex.meta.featured;
      draft.order = ex.meta.order;
      draft.summary = ex.meta.summary;
      draft.summary_en = ex.meta.summary_en;
      draft.tags = ex.meta.tags;
      draft.highlights = ex.meta.highlights;
      draft.highlights_en = ex.meta.highlights_en;
      draft.readme_md = ex.readme_md;
    }
    serde_json::to_string(&draft).map_err(|e| ServerFnError::new(e.to_string()))
  }
  #[cfg(not(feature = "server"))]
  {
    let _ = (repo, source, revision);
    Err(ServerFnError::new("server only"))
  }
}

/// 管理员调用服务端配置的 LLM（OpenAI/Anthropic 兼容协议），严格基于真实架构数值润色双语简介与核心亮点。
#[cfg(feature = "accounts")]
#[post("/api/ai/models/ai-polish")]
pub async fn admin_ai_polish_model_draft(draft_json: String) -> Result<String, ServerFnError> {
  #[cfg(feature = "server")]
  {
    let _ = app_core::session::require_admin().await?;
    let draft: CuratedModelUpsertInput = serde_json::from_slice(draft_json.as_bytes())
      .map_err(|e| ServerFnError::new(e.to_string()))?;
    let polished = polish_draft_via_llm(&draft).await.map_err(ServerFnError::new)?;
    serde_json::to_string(&polished).map_err(|e| ServerFnError::new(e.to_string()))
  }
  #[cfg(not(feature = "server"))]
  {
    let _ = draft_json;
    Err(ServerFnError::new("server only"))
  }
}

/// 管理员保存（新增或更新）指定模型至 PostgreSQL `curated_models` 表并即时刷新全站缓存，返回保存成功的 `slug`。
#[cfg(feature = "accounts")]
#[post("/api/ai/models/save")]
pub async fn admin_save_curated_model(input_json: String) -> Result<String, ServerFnError> {
  #[cfg(feature = "server")]
  {
    use app_core::entities::curated_model;
    use sea_orm::sea_query::OnConflict;
    use sea_orm::{EntityTrait, Set};

    let admin = app_core::session::require_admin().await?;
    let input: CuratedModelUpsertInput = serde_json::from_slice(input_json.as_bytes())
      .map_err(|e| ServerFnError::new(e.to_string()))?;
    let clean = validate_upsert_input(input).map_err(ServerFnError::new)?;
    let db = app_core::db::get_or_init_pool()
      .await
      .map_err(|e| ServerFnError::new(format!("数据库连接不可用: {e}")))?;

    let tags_json =
      serde_json::to_string(&clean.tags).map_err(|e| ServerFnError::new(e.to_string()))?;
    let highlights_json =
      serde_json::to_string(&clean.highlights).map_err(|e| ServerFnError::new(e.to_string()))?;
    let highlights_en_json =
      serde_json::to_string(&clean.highlights_en).map_err(|e| ServerFnError::new(e.to_string()))?;
    let blueprint_json =
      serde_json::to_string(&clean.blueprint).map_err(|e| ServerFnError::new(e.to_string()))?;
    let ver_report = clean.blueprint.verification_report();
    let verification_status = ver_report
      .as_ref()
      .map(|v| v.status.as_str().to_string())
      .unwrap_or_else(|| "partial".to_string());
    let verification_report_json = ver_report.as_ref().and_then(|v| serde_json::to_string(v).ok());
    let upstream_fingerprint = Some(config_fingerprint_hex(&clean.blueprint.raw_config_pretty));
    let now = chrono::Utc::now().fixed_offset();
    let verified_at =
      matches!(verification_status.as_str(), "verified" | "curated_verified").then_some(now);

    let active = curated_model::ActiveModel {
      slug: Set(clean.slug.clone()),
      name: Set(clean.name.clone()),
      vendor: Set(clean.vendor.clone()),
      modality: Set(clean.modality.as_str().to_string()),
      series: Set(clean.series.clone()),
      release_date: Set(clean.release_date.clone()),
      hf_repo: Set(clean.hf_repo.clone()),
      ms_repo: Set(clean.ms_repo.clone()),
      featured: Set(clean.featured),
      sort_order: Set(clean.order),
      summary: Set(clean.summary.clone()),
      summary_en: Set(clean.summary_en.clone()),
      tags_json: Set(tags_json),
      highlights_json: Set(highlights_json),
      highlights_en_json: Set(highlights_en_json),
      readme_md: Set(clean.readme_md.clone()),
      blueprint_json: Set(blueprint_json),
      verification_status: Set(verification_status),
      verification_report_json: Set(verification_report_json),
      verified_at: Set(verified_at),
      upstream_fingerprint: Set(upstream_fingerprint),
      created_by: Set(Some(admin.id)),
      created_at: Set(now),
      updated_at: Set(now),
    };

    curated_model::Entity::insert(active)
      .on_conflict(
        OnConflict::column(curated_model::Column::Slug)
          .update_columns([
            curated_model::Column::Name,
            curated_model::Column::Vendor,
            curated_model::Column::Modality,
            curated_model::Column::Series,
            curated_model::Column::ReleaseDate,
            curated_model::Column::HfRepo,
            curated_model::Column::MsRepo,
            curated_model::Column::Featured,
            curated_model::Column::SortOrder,
            curated_model::Column::Summary,
            curated_model::Column::SummaryEn,
            curated_model::Column::TagsJson,
            curated_model::Column::HighlightsJson,
            curated_model::Column::HighlightsEnJson,
            curated_model::Column::ReadmeMd,
            curated_model::Column::BlueprintJson,
            curated_model::Column::VerificationStatus,
            curated_model::Column::VerificationReportJson,
            curated_model::Column::VerifiedAt,
            curated_model::Column::UpstreamFingerprint,
            curated_model::Column::UpdatedAt,
          ])
          .to_owned(),
      )
      .exec(&db)
      .await
      .map_err(|e| ServerFnError::new(format!("写入 curated_models 失败: {e}")))?;

    invalidate_db_models_cache();

    Ok(clean.slug)
  }
  #[cfg(not(feature = "server"))]
  {
    let _ = input_json;
    Err(ServerFnError::new("server only"))
  }
}

/// 管理员删除数据库中的指定模型记录（若存在同名内置静态模型，删除后自动恢复为静态默认版本）。
#[cfg(feature = "accounts")]
#[post("/api/ai/models/delete")]
pub async fn admin_delete_curated_model(slug: String) -> Result<bool, ServerFnError> {
  #[cfg(feature = "server")]
  {
    use app_core::entities::curated_model;
    use sea_orm::EntityTrait;

    let _ = app_core::session::require_admin().await?;
    let clean = slug.trim().to_ascii_lowercase();
    if !is_safe_model_slug(&clean) {
      return Err(ServerFnError::new("非法模型 Slug".to_string()));
    }
    let db = app_core::db::get_or_init_pool()
      .await
      .map_err(|e| ServerFnError::new(format!("数据库连接不可用: {e}")))?;

    let res = curated_model::Entity::delete_by_id(&clean)
      .exec(&db)
      .await
      .map_err(|e| ServerFnError::new(format!("删除模型记录失败: {e}")))?;

    invalidate_db_models_cache();
    Ok(res.rows_affected > 0)
  }
  #[cfg(not(feature = "server"))]
  {
    let _ = slug;
    Err(ServerFnError::new("server only"))
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::arch::load_builtin_preset;

  #[test]
  fn slugifies_repos_and_validates_slugs() {
    assert_eq!(slugify_repo_id("Qwen/Qwen2.5-VL-7B-Instruct"), "qwen2-5-vl-7b");
    assert_eq!(slugify_repo_id("deepseek-ai/DeepSeek-V3"), "deepseek-v3");
    assert_eq!(slugify_repo_id("black-forest-labs/FLUX.1-dev"), "flux-1-dev");
    assert!(is_safe_model_slug("qwen2-5-vl-7b"));
    assert!(!is_safe_model_slug("../etc"));
    assert!(!is_safe_model_slug("UpperCase"));
  }

  #[test]
  fn auto_drafts_bilingual_metadata_from_blueprint() {
    let bp = load_builtin_preset("deepseek-v3").expect("deepseek-v3 preset");
    let draft = draft_from_blueprint(&bp);
    assert_eq!(draft.slug, "deepseek-v3");
    assert_eq!(draft.vendor, "DeepSeek");
    assert_eq!(draft.modality, ModelModality::Llm);
    assert_eq!(draft.series, "deepseek");
    assert!(draft.tags.contains(&"mla".to_string()));
    assert!(draft.tags.contains(&"moe".to_string()));
    assert_eq!(draft.highlights.len(), 3);
    assert_eq!(draft.highlights_en.len(), 3);
    assert!(validate_upsert_input(draft).is_ok());
  }

  #[test]
  fn merges_builtin_and_database_entries_with_override_precedence() {
    let bp_ds = load_builtin_preset("deepseek-v3").expect("deepseek-v3");
    let bp_qwen = load_builtin_preset("qwen3-30b-a3b").expect("qwen3");

    let d1_draft = draft_from_blueprint(&bp_ds);
    let builtin_entry = CuratedModelDetail {
      meta: build_curated_summary(
        "deepseek-v3".to_string(),
        "DeepSeek-V3 Builtin".to_string(),
        d1_draft.vendor.clone(),
        Some(ModelModality::Llm),
        "deepseek".to_string(),
        "2024-12".to_string(),
        d1_draft.hf_repo.clone(),
        None,
        true,
        10,
        "builtin summary".to_string(),
        String::new(),
        vec!["mla".to_string()],
        vec![],
        vec![],
        &bp_ds,
        RecordOrigin::Builtin,
      ),
      blueprint: bp_ds.clone(),
      readme_md: None,
    };

    let override_entry = CuratedModelDetail {
      meta: build_curated_summary(
        "deepseek-v3".to_string(),
        "DeepSeek-V3 Customized".to_string(),
        "DeepSeek".to_string(),
        Some(ModelModality::Llm),
        "deepseek".to_string(),
        "2024-12".to_string(),
        "deepseek-ai/DeepSeek-V3".to_string(),
        None,
        true,
        5,
        "db override summary".to_string(),
        String::new(),
        vec!["mla".to_string()],
        vec![],
        vec![],
        &bp_ds,
        RecordOrigin::Database,
      ),
      blueprint: bp_ds,
      readme_md: Some("# Notes".to_string()),
    };

    let new_db_entry = CuratedModelDetail {
      meta: build_curated_summary(
        "qwen3-30b-custom".to_string(),
        "Qwen3-30B-Custom".to_string(),
        "Alibaba Qwen".to_string(),
        Some(ModelModality::Llm),
        "qwen".to_string(),
        "2025-04".to_string(),
        "Qwen/Qwen3-30B-A3B".to_string(),
        None,
        false,
        20,
        "new db entry".to_string(),
        String::new(),
        vec!["moe".to_string()],
        vec![],
        vec![],
        &bp_qwen,
        RecordOrigin::Database,
      ),
      blueprint: bp_qwen,
      readme_md: None,
    };

    let merged = merge_builtin_and_db_models(&[builtin_entry], vec![override_entry, new_db_entry]);
    assert_eq!(merged.len(), 2);
    assert_eq!(merged[0].meta.slug, "deepseek-v3");
    assert_eq!(merged[0].meta.name, "DeepSeek-V3 Customized");
    assert_eq!(merged[0].meta.origin, RecordOrigin::Override);
    assert_eq!(merged[1].meta.slug, "qwen3-30b-custom");
    assert_eq!(merged[1].meta.origin, RecordOrigin::Database);
  }

  #[test]
  fn auto_drafts_audio_ssm_and_embedding_modalities() {
    use crate::arch::{build_model_blueprint_with_aux, AuxConfigFiles};

    let whisper_bp = build_model_blueprint_with_aux(
      "openai/whisper-large-v3",
      "HuggingFace",
      r#"{"architectures":["WhisperForConditionalGeneration"],"model_type":"whisper","d_model":1280,"encoder_layers":32,"encoder_attention_heads":20,"encoder_ffn_dim":5120,"decoder_layers":32,"decoder_attention_heads":20,"decoder_ffn_dim":5120,"num_mel_bins":128,"max_source_positions":1500,"max_target_positions":448,"vocab_size":51866}"#,
      AuxConfigFiles::default(),
      None,
    )
    .expect("whisper");
    let w_draft = draft_from_blueprint(&whisper_bp);
    assert_eq!(w_draft.modality, ModelModality::Audio);
    assert!(w_draft.tags.contains(&"audio".to_string()));
    assert!(w_draft.tags.contains(&"cross-attn".to_string()));
    assert!(validate_upsert_input(w_draft).is_ok());

    let mamba_bp = build_model_blueprint_with_aux(
      "state-spaces/mamba-2.8b-hf",
      "HuggingFace",
      r#"{"architectures":["MambaForCausalLM"],"model_type":"mamba","d_model":2560,"n_layer":64,"state_size":16,"conv_kernel":4,"expand":2,"vocab_size":50280}"#,
      AuxConfigFiles::default(),
      None,
    )
    .expect("mamba");
    let m_draft = draft_from_blueprint(&mamba_bp);
    assert_eq!(m_draft.modality, ModelModality::Llm);
    assert!(m_draft.tags.contains(&"ssm".to_string()));
    assert!(validate_upsert_input(m_draft).is_ok());

    let bge_bp = build_model_blueprint_with_aux(
      "BAAI/bge-m3",
      "HuggingFace",
      r#"{"architectures":["XLMRobertaModel"],"model_type":"xlm-roberta","hidden_size":1024,"num_hidden_layers":24,"num_attention_heads":16,"intermediate_size":4096,"max_position_embeddings":8194,"vocab_size":250002}"#,
      AuxConfigFiles {
        pooling_json: Some(r#"{"word_embedding_dimension":1024,"pooling_mode_cls_token":true}"#),
        ..Default::default()
      },
      None,
    )
    .expect("bge-m3");
    let b_draft = draft_from_blueprint(&bge_bp);
    assert_eq!(b_draft.modality, ModelModality::Embedding);
    assert!(b_draft.tags.contains(&"embedding".to_string()));
    assert!(b_draft.tags.contains(&"bidirectional".to_string()));
    assert!(validate_upsert_input(b_draft).is_ok());

    let flux_bp = crate::arch::build_diffusion_pipeline_blueprint(
      "black-forest-labs/FLUX.1-dev",
      "HuggingFace",
      r#"{"_class_name":"FluxPipeline","text_encoder":["transformers","CLIPTextModel"],"text_encoder_2":["transformers","T5EncoderModel"],"transformer":["diffusers","FluxTransformer2DModel"],"vae":["diffusers","AutoencoderKL"],"scheduler":["diffusers","FlowMatchEulerDiscreteScheduler"]}"#,
      &[(
        "transformer".to_string(),
        r#"{"_class_name":"FluxTransformer2DModel","num_layers":19,"num_single_layers":38,"num_attention_heads":24,"attention_head_dim":128,"in_channels":64,"patch_size":1}"#.to_string(),
      )],
      None,
    )
    .expect("flux");
    let f_draft = draft_from_blueprint(&flux_bp);
    assert_eq!(f_draft.modality, ModelModality::Diffusion);
    assert!(f_draft.tags.contains(&"diffusion".to_string()));
    assert!(f_draft.tags.contains(&"dit".to_string()));
    assert!(validate_upsert_input(f_draft).is_ok());
  }
}
