//! AI 与大模型数学基础动态推导图鉴（`/ai/math` 全景导航页 + `/ai/math/:slug` 分步色标推导实验室 + 双向算子/模型反查）。
//!
//! 数据源合并规则（零数据库依赖，天然兼容 `basic` 档位与 `SIGHUP` 秒级热刷新）：
//! 1. 先加载框架内置标杆推导词条 `assets/ai-math/<slug>/index.md`；
//! 2. 若挂载了外部内容仓库 `<CONTENT_DIR>/ai-math/<slug>/index.md`，自动合并覆盖同名 `slug` 并追加全量 48 篇扩展词条。

#[cfg(feature = "server")]
use std::collections::BTreeMap;
#[cfg(feature = "server")]
use std::path::{Path, PathBuf};
#[cfg(feature = "server")]
use std::sync::Arc;

use app_core::i18n::Language;
use dioxus::prelude::*;
use dioxus_shadcn::{badge_class, card_class, BadgeVariant, Input, Spinner, SpinnerSize};
use serde::{Deserialize, Serialize};
use widgets::{Markdown, OnThisPage, PageMeta};

use crate::arch::{InspectorFocus, ModelBlueprint};
#[cfg(feature = "server")]
use crate::atlas::CuratedModelSummary;
use crate::component_labs::current_lang;
use crate::components_atlas::RelatedModelLink;
use crate::math_labs::{MathLabKind, MathLabWorkbench};

/// 9 大 AI 与大模型底层数学支柱。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum MathPillar {
  #[default]
  LinearGeometry,
  MatrixCalculusAutograd,
  ProbabilityStats,
  InformationVariational,
  SignalSpectralGraph,
  OptimizationScaling,
  LossMetricDistill,
  DiffusionSdeFlow,
  RlAlignmentPrecision,
}

impl MathPillar {
  pub const ALL: &'static [Self] = &[
    Self::LinearGeometry,
    Self::MatrixCalculusAutograd,
    Self::ProbabilityStats,
    Self::InformationVariational,
    Self::SignalSpectralGraph,
    Self::OptimizationScaling,
    Self::LossMetricDistill,
    Self::DiffusionSdeFlow,
    Self::RlAlignmentPrecision,
  ];

  pub fn parse_str(raw: &str) -> Self {
    match raw.trim().to_ascii_lowercase().as_str() {
      "linear" | "linear_geometry" | "geometry" | "pillar_01" => Self::LinearGeometry,
      "calculus" | "autograd" | "matrix_calculus_autograd" | "pillar_02" => {
        Self::MatrixCalculusAutograd
      }
      "prob" | "probability" | "probability_stats" | "stats" | "pillar_03" => {
        Self::ProbabilityStats
      }
      "info" | "information" | "information_variational" | "variational" | "pillar_04" => {
        Self::InformationVariational
      }
      "spectral" | "signal" | "signal_spectral_graph" | "fourier" | "graph" | "pillar_05" => {
        Self::SignalSpectralGraph
      }
      "opt" | "optimization" | "optimization_scaling" | "scaling" | "pillar_06" => {
        Self::OptimizationScaling
      }
      "loss" | "losses" | "loss_metric_distill" | "distill" | "metric" | "pillar_07" => {
        Self::LossMetricDistill
      }
      "sde" | "diffusion" | "diffusion_sde_flow" | "flow" | "ot" | "pillar_08" => {
        Self::DiffusionSdeFlow
      }
      "rl" | "alignment" | "precision" | "rl_alignment_precision" | "pillar_09" => {
        Self::RlAlignmentPrecision
      }
      _ => Self::LinearGeometry,
    }
  }

  pub fn label(self, lang: Language) -> &'static str {
    match self {
      Self::LinearGeometry => {
        lang.pick("01 · 线性代数与高维几何", "01 · Linear Algebra & Geometry")
      }
      Self::MatrixCalculusAutograd => {
        lang.pick("02 · 矩阵求导与自动微分", "02 · Matrix Calculus & Autograd")
      }
      Self::ProbabilityStats => {
        lang.pick("03 · 概率统计与信号传播", "03 · Probability & Signal Prop")
      }
      Self::InformationVariational => {
        lang.pick("04 · 信息论·散度与变分", "04 · Info Theory & Variational")
      }
      Self::SignalSpectralGraph => {
        lang.pick("05 · 傅里叶·时频与图谱", "05 · Spectral & Graph Theory")
      }
      Self::OptimizationScaling => {
        lang.pick("06 · 非凸优化与缩放定律", "06 · Optimization & Scaling")
      }
      Self::LossMetricDistill => {
        lang.pick("07 · 损失·度量与蒸馏几何", "07 · Losses, Metrics & Distill")
      }
      Self::DiffusionSdeFlow => {
        lang.pick("08 · 扩散 SDE 与最优传输流", "08 · Diffusion SDE & Flow")
      }
      Self::RlAlignmentPrecision => {
        lang.pick("09 · 强化学习·对齐与浮点算术", "09 · RL, Alignment & Precision")
      }
    }
  }

  pub fn short_badge(self, lang: Language) -> &'static str {
    match self {
      Self::LinearGeometry => lang.pick("线性高维几何", "Linear/SVD"),
      Self::MatrixCalculusAutograd => lang.pick("矩阵微分/VJP", "Autograd/VJP"),
      Self::ProbabilityStats => lang.pick("概率/方差守恒", "Probability"),
      Self::InformationVariational => lang.pick("信息散度/ELBO", "KL & ELBO"),
      Self::SignalSpectralGraph => lang.pick("傅里叶/图谱", "Spectral/GNN"),
      Self::OptimizationScaling => lang.pick("优化器/缩放律", "Optim/Scaling"),
      Self::LossMetricDistill => lang.pick("损失/偏好几何", "Loss/Distill"),
      Self::DiffusionSdeFlow => lang.pick("扩散SDE/流匹配", "SDE & Flow"),
      Self::RlAlignmentPrecision => lang.pick("RL对齐/IEEE754", "RL & Float"),
    }
  }
}

/// 数学推导卡片摘要元数据（精简跨端 CBOR 载荷）。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MathSummary {
  pub slug: String,
  pub title: String,
  pub title_en: String,
  pub pillar: MathPillar,
  /// `"cnn_audio"` | `"llm"` | `"diffusion"` | `"rl_align"` | `"quant_opt"`
  pub domains: Vec<String>,
  pub lab_kind: MathLabKind,
  pub lab_mode: String,
  pub theorem_short: String,
  pub proof_steps_badge: String,
  pub summary: String,
  pub summary_en: String,
  pub tags: Vec<String>,
}

impl MathSummary {
  pub fn localized_title(&self, lang: Language) -> &str {
    match lang {
      Language::En if !self.title_en.trim().is_empty() => &self.title_en,
      _ => &self.title,
    }
  }

  pub fn localized_summary(&self, lang: Language) -> &str {
    match lang {
      Language::En if !self.summary_en.trim().is_empty() => &self.summary_en,
      _ => &self.summary,
    }
  }
}

/// 同支柱兄弟定理轻量导航项。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SiblingMathLink {
  pub slug: String,
  pub title: String,
  pub title_en: String,
}

impl SiblingMathLink {
  pub fn localized_title(&self, lang: Language) -> &str {
    match lang {
      Language::En if !self.title_en.trim().is_empty() => &self.title_en,
      _ => &self.title,
    }
  }
}

/// 该数学定理直接支撑的下游算子轻量卡片。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RelatedComponentLink {
  pub slug: String,
  pub title: String,
  pub title_en: String,
  pub family_badge: String,
  pub formula_short: String,
}

impl RelatedComponentLink {
  pub fn localized_title(&self, lang: Language) -> &str {
    match lang {
      Language::En if !self.title_en.trim().is_empty() => &self.title_en,
      _ => &self.title,
    }
  }
}

/// 单个数学推导专题详情页完整载荷。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MathDetail {
  pub meta: MathSummary,
  pub body_md: String,
  pub related_components: Vec<RelatedComponentLink>,
  pub related_models: Vec<RelatedModelLink>,
  pub siblings: Vec<SiblingMathLink>,
}

/// 根据数学专题 `slug` 确定性推断 `<MathLab id="..." />` 的默认画布内核与交互子模式（覆盖全部 48 个核心词条）。
pub fn infer_math_lab_by_slug(slug: &str, mode_override: &str) -> (MathLabKind, String) {
  let s = slug.trim().to_ascii_lowercase();
  let (kind, default_mode) = if s.contains("fourier")
    || s.contains("stft")
    || s.contains("spectral")
    || s.contains("laplacian")
  {
    let m = if s.contains("laplacian") {
      "graph_laplacian"
    } else if s.contains("stft") || s.contains("mel") {
      "stft_uncertainty"
    } else if s.contains("bias") || s.contains("ntk") {
      "spectral_bias"
    } else {
      "dft_winding"
    };
    (MathLabKind::SpectralAndGraph, m)
  } else if s.contains("smoothness")
    || s.contains("momentum")
    || s.contains("adam")
    || s.contains("muon")
    || s.contains("shampoo")
    || s.contains("lagrange")
    || s.contains("bregman")
    || s.contains("scaling")
    || s.contains("chinchilla")
    || s.contains("ddpm")
    || s.contains("score")
    || s.contains("langevin")
    || s.contains("ito")
    || s.contains("fokker")
    || s.contains("cfm")
    || s.contains("rectified")
  {
    let m = if s.contains("cfm") || s.contains("rectified") || s.contains("optimal-transport") {
      "rectified_flow"
    } else if s.contains("ito") || s.contains("sde") || s.contains("score") || s.contains("ddpm") {
      "sde_ode"
    } else if s.contains("muon") || s.contains("shampoo") {
      "muon_ns"
    } else {
      "adamw_canyon"
    };
    (MathLabKind::OptimizerAndFlow, m)
  } else if s.contains("bellman")
    || s.contains("mdp")
    || s.contains("policy-gradient")
    || s.contains("gae")
    || s.contains("ppo")
    || s.contains("trpo")
    || s.contains("dpo")
    || s.contains("grpo")
    || s.contains("rlvr")
    || s.contains("ieee754")
    || s.contains("bf16")
    || s.contains("fp8")
  {
    let m = if s.contains("ieee754") || s.contains("bf16") || s.contains("fp8") {
      "ieee754_bits"
    } else if s.contains("dpo") || s.contains("grpo") || s.contains("rlvr") {
      "dpo_grpo"
    } else if s.contains("bellman") || s.contains("mdp") || s.contains("gae") {
      "bellman_gae"
    } else {
      "ppo_clip"
    };
    (MathLabKind::RlAndBitPrecision, m)
  } else if s.contains("vjp")
    || s.contains("jvp")
    || s.contains("gradient")
    || s.contains("jacobian")
    || s.contains("hessian")
    || s.contains("trace-trick")
    || s.contains("checkpointing")
    || s.contains("ste")
    || s.contains("mse")
    || s.contains("huber")
    || s.contains("focal")
    || s.contains("arcface")
    || s.contains("bradley-terry")
    || s.contains("distillation")
  {
    let m = if s.contains("checkpoint") || s.contains("ste") {
      "checkpoint_ste"
    } else if s.contains("mse") || s.contains("huber") || s.contains("focal") {
      "robust_focal"
    } else if s.contains("distill") || s.contains("bradley") || s.contains("arcface") {
      "distill_bt"
    } else {
      "vjp_graph"
    };
    (MathLabKind::AutogradAndLoss, m)
  } else if s.contains("kl")
    || s.contains("entropy")
    || s.contains("wasserstein")
    || s.contains("infonce")
    || s.contains("jensen")
    || s.contains("elbo")
    || s.contains("bayes")
    || s.contains("gumbel")
    || s.contains("clt")
    || s.contains("kaiming")
    || s.contains("xavier")
    || s.contains("importance-sampling")
    || s.contains("calibration")
  {
    let m = if s.contains("wasserstein") || s.contains("js") || s.contains("infonce") {
      "wasserstein_js"
    } else if s.contains("jensen") || s.contains("elbo") || s.contains("gumbel") {
      "elbo_jensen"
    } else if s.contains("bayes")
      || s.contains("kaiming")
      || s.contains("xavier")
      || s.contains("clt")
      || s.contains("calibration")
      || s.contains("importance")
    {
      "bayes_init_cal"
    } else {
      "kl_modes_k3"
    };
    (MathLabKind::DistributionDivergence, m)
  } else {
    let m = if s.contains("norms") || s.contains("vectors") {
      "lp_norm"
    } else if s.contains("householder") || s.contains("so2") || s.contains("orthogonal") {
      "householder"
    } else if s.contains("high-dim") || s.contains("jl-lemma") {
      "high_dim"
    } else {
      "svd"
    };
    (MathLabKind::LinearTransform2d, m)
  };

  let m = if mode_override.trim().is_empty() {
    default_mode.to_string()
  } else {
    mode_override.to_string()
  };
  (kind, m)
}

/// 为 `/models/:slug` 与 `/ai/visualizer` 算子检视面板推荐对应的底层数学公理推导直通链接（仅指向框架内置标杆 slug 以兼容 `basic` 档位）。
pub fn recommended_math_links(
  bp: &ModelBlueprint,
  focus: InspectorFocus,
  lang: Language,
) -> Vec<(&'static str, &'static str, &'static str)> {
  match focus {
    InspectorFocus::Overview => recommended_math_links(bp, bp.primary_operator_focus(), lang),
    InspectorFocus::Attention => vec![(
      "/ai/math/svd-eckart-young-pseudoinverse",
      "SVD & Low-Rank Proof",
      lang.pick(
        "SVD 奇异值几何与 Eckart-Young 最优低秩定理（MLA / LoRA 地基）",
        "SVD Geometry & Eckart-Young Low-Rank Theorem",
      ),
    )],
    InspectorFocus::Vision => vec![(
      "/ai/math/continuity-equation-cfm-theorem",
      "CFM & Rectified Flow Proof",
      lang.pick(
        "连续性方程与条件流匹配（CFM）梯度等价性核心定理（FLUX / Wan 地基）",
        "Continuity Equation & Conditional Flow Matching Equivalence Proof",
      ),
    )],
    InspectorFocus::Audio => vec![(
      "/ai/math/fourier-convolution-aliasing",
      "DFT & Aliasing Proof",
      lang.pick(
        "复平面傅里叶缠绕正交性、卷积定理与奈奎斯特混叠证明",
        "Complex DFT Winding, Convolution Theorem & Nyquist Aliasing Proof",
      ),
    )],
    InspectorFocus::FfnOrMoe => vec![(
      "/ai/math/ieee754-bf16-fp8-numerical-math",
      "IEEE-754 & BF16/FP8 Proof",
      lang.pick(
        "IEEE-754 位域解剖、BF16 梯度吞没与 FP32 主权重必选定理",
        "IEEE-754 Anatomy, BF16 Swamping & FP32 Master Weights Proof",
      ),
    )],
    InspectorFocus::NormAndResidual => vec![(
      "/ai/math/computational-graph-vjp-vs-jvp",
      "Autograd VJP Proof",
      lang.pick(
        "反向模式自动微分 VJP 伴随链复杂度与梯度驻留定理",
        "Reverse-Mode Autograd VJP & Activation Memory Proof",
      ),
    )],
    InspectorFocus::Embedding | InspectorFocus::LmHead | InspectorFocus::MtpHead => vec![(
      "/ai/math/forward-reverse-kl-and-k3",
      "KL & k3 Estimator Proof",
      lang.pick(
        "Forward vs Reverse KL 寻峰坍缩与 Schulman k₃ 无偏低方差证明",
        "Forward vs Reverse KL & Unbiased k3 Estimator Proof",
      ),
    )],
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 服务端目录扫描与双源合并（`assets/ai-math` + `<CONTENT_DIR>/ai-math`）
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "server")]
#[derive(Debug, Clone, Deserialize, Default)]
struct RawMathFrontmatter {
  #[serde(default)]
  slug: String,
  #[serde(default)]
  title: String,
  #[serde(default)]
  title_en: String,
  #[serde(default)]
  pillar: String,
  #[serde(default)]
  domains: Vec<String>,
  #[serde(default = "default_order")]
  order: i32,
  #[serde(default)]
  lab_kind: String,
  #[serde(default)]
  lab_mode: String,
  #[serde(default, alias = "formula_short")]
  theorem_short: String,
  #[serde(default)]
  proof_steps_badge: String,
  #[serde(default, alias = "description")]
  summary: String,
  #[serde(default)]
  summary_en: String,
  #[serde(default)]
  tags: Vec<String>,
  #[serde(default)]
  component_slugs: Vec<String>,
  #[serde(default)]
  model_keywords: Vec<String>,
}

#[cfg(feature = "server")]
fn default_order() -> i32 {
  100
}

#[cfg(feature = "server")]
#[derive(Debug, Clone)]
pub struct StoredMathEntry {
  pub meta: MathSummary,
  pub order: i32,
  pub component_slugs: Vec<String>,
  pub model_keywords: Vec<String>,
  pub body_md: String,
}

#[cfg(feature = "server")]
static MATH_CACHE: app_core::utils::DirListingCache<Vec<StoredMathEntry>> =
  app_core::utils::DirListingCache::new();

#[cfg(feature = "server")]
pub fn invalidate_math_cache() {
  MATH_CACHE.invalidate();
}

#[cfg(feature = "server")]
fn parse_math_dir(dir: &Path, slug_fallback: &str) -> Result<StoredMathEntry, String> {
  let md_path = if dir.join("index.mdx").is_file() {
    dir.join("index.mdx")
  } else if dir.join("index.md").is_file() {
    dir.join("index.md")
  } else {
    return Err(format!("{} 缺少 index.md", dir.display()));
  };

  let raw = std::fs::read_to_string(&md_path)
    .map_err(|e| format!("读取 {} 失败: {e}", md_path.display()))?;
  let (fm_yaml, body) = widgets::split_frontmatter(&raw)
    .ok_or_else(|| format!("{} 缺少闭合的 YAML frontmatter", md_path.display()))?;
  let fm: RawMathFrontmatter = serde_yaml::from_str(fm_yaml)
    .map_err(|e| format!("解析 {} YAML 失败: {e}", md_path.display()))?;

  let slug =
    if fm.slug.trim().is_empty() { slug_fallback.to_string() } else { fm.slug.trim().to_string() };
  let (inferred_kind, inferred_mode) = infer_math_lab_by_slug(&slug, &fm.lab_mode);
  let lab_kind = if fm.lab_kind.trim().is_empty() {
    inferred_kind
  } else {
    MathLabKind::parse_str(&fm.lab_kind)
  };
  let domains = if fm.domains.is_empty() { vec!["llm".to_string()] } else { fm.domains };

  let meta = MathSummary {
    slug: slug.clone(),
    title: if fm.title.trim().is_empty() { slug } else { fm.title },
    title_en: fm.title_en,
    pillar: MathPillar::parse_str(&fm.pillar),
    domains,
    lab_kind,
    lab_mode: inferred_mode,
    theorem_short: fm.theorem_short,
    proof_steps_badge: fm.proof_steps_badge,
    summary: fm.summary,
    summary_en: fm.summary_en,
    tags: fm.tags,
  };

  Ok(StoredMathEntry {
    meta,
    order: fm.order,
    component_slugs: fm.component_slugs,
    model_keywords: fm.model_keywords,
    body_md: body.trim_start().to_string(),
  })
}

#[cfg(feature = "server")]
fn scan_math_dir_into(root: &Path, merged: &mut BTreeMap<String, StoredMathEntry>) {
  let Ok(rd) = std::fs::read_dir(root) else { return };
  let mut paths: Vec<PathBuf> = rd.flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect();
  paths.sort();
  for path in paths {
    let Some(name) = path.file_name().and_then(|s| s.to_str()) else { continue };
    if name.starts_with('.') || name.starts_with('_') || !app_core::utils::is_safe_slug(name) {
      continue;
    }
    match parse_math_dir(&path, name) {
      Ok(entry) => {
        merged.insert(entry.meta.slug.clone(), entry);
      }
      Err(err) => {
        tracing::warn!(dir = %path.display(), error = %err, "skip invalid ai-math entry");
      }
    }
  }
}

/// 加载合并后的全部数学推导词条（内置 `assets/ai-math` + 外部 `<CONTENT_DIR>/ai-math`）。
#[cfg(feature = "server")]
pub fn load_all_math_entries() -> Arc<Vec<StoredMathEntry>> {
  let builtin_root = app_core::utils::get_asset_root().join("ai-math");
  let content_root = app_core::utils::content_dir("ai-math");
  let is_md = |p: &Path| {
    p.file_name().and_then(|n| n.to_str()).is_some_and(|n| matches!(n, "index.md" | "index.mdx"))
  };
  let mut fp = app_core::utils::fingerprint_for_dir(&builtin_root, is_md);
  if content_root != builtin_root {
    fp.extend(app_core::utils::fingerprint_for_dir(&content_root, is_md));
  }

  MATH_CACHE.get_or_rebuild(fp, || {
    let mut map = BTreeMap::new();
    scan_math_dir_into(&builtin_root, &mut map);
    if content_root != builtin_root {
      scan_math_dir_into(&content_root, &mut map);
    }
    let mut items: Vec<StoredMathEntry> = map.into_values().collect();
    items.sort_by(|a, b| a.order.cmp(&b.order).then_with(|| a.meta.slug.cmp(&b.meta.slug)));
    items
  })
}

/// 为嵌入式 Tantivy 全文检索提供全部数学推导专题文档。
#[cfg(feature = "server")]
pub fn search_docs() -> Vec<app_core::engines::doc_source::ExternalIndexedDoc> {
  load_all_math_entries()
    .iter()
    .map(|entry| {
      let body = format!(
        "{} {} {} {} {}",
        entry.meta.summary,
        entry.meta.summary_en,
        entry.meta.tags.join(" "),
        entry.meta.theorem_short,
        entry.body_md.chars().take(4000).collect::<String>(),
      );
      app_core::engines::doc_source::ExternalIndexedDoc {
        kind: "ai".to_string(),
        ref_id: format!("math-{}", entry.meta.slug),
        title: entry.meta.title.clone(),
        body,
        url: format!("/ai/math/{}", entry.meta.slug),
        created_at: String::new(),
      }
    })
    .collect()
}

#[cfg(feature = "server")]
fn model_matches_math(model: &CuratedModelSummary, math: &StoredMathEntry) -> bool {
  let hay = format!(
    "{} {} {} {} {} {} {}",
    model.slug,
    model.name,
    model.series,
    model.family_badge,
    model.attention_short,
    model.ffn_short,
    model.tags.join(" ")
  )
  .to_ascii_lowercase();

  math.model_keywords.iter().any(|kw| {
    let k = kw.trim().to_ascii_lowercase();
    !k.is_empty() && hay.contains(&k)
  })
}

#[get("/api/ai/math")]
pub async fn list_ai_math_entries() -> Result<Vec<MathSummary>, ServerFnError> {
  Ok(load_all_math_entries().iter().map(|e| e.meta.clone()).collect())
}

#[get("/api/ai/math/:slug")]
pub async fn get_ai_math_detail(slug: String) -> Result<Option<MathDetail>, ServerFnError> {
  let clean = slug.trim();
  if !app_core::utils::is_safe_slug(clean) {
    return Ok(None);
  }
  let all = load_all_math_entries();
  let Some(found) = all.iter().find(|e| e.meta.slug == clean) else {
    return Ok(None);
  };

  let siblings: Vec<SiblingMathLink> = all
    .iter()
    .filter(|e| e.meta.pillar == found.meta.pillar)
    .map(|e| SiblingMathLink {
      slug: e.meta.slug.clone(),
      title: e.meta.title.clone(),
      title_en: e.meta.title_en.clone(),
    })
    .collect();

  let all_comps = crate::components_atlas::load_all_components();
  let mut related_components: Vec<RelatedComponentLink> = Vec::new();
  for target_slug in &found.component_slugs {
    if let Some(c) = all_comps.iter().find(|c| c.meta.slug == target_slug.trim()) {
      related_components.push(RelatedComponentLink {
        slug: c.meta.slug.clone(),
        title: c.meta.title.clone(),
        title_en: c.meta.title_en.clone(),
        family_badge: c.meta.family.short_badge(Language::Zh).to_string(),
        formula_short: c.meta.formula_short.clone(),
      });
    }
  }
  if related_components.is_empty() {
    for c in all_comps
      .iter()
      .filter(|c| {
        found.meta.tags.iter().any(|t| c.meta.tags.iter().any(|ct| ct.eq_ignore_ascii_case(t)))
      })
      .take(4)
    {
      related_components.push(RelatedComponentLink {
        slug: c.meta.slug.clone(),
        title: c.meta.title.clone(),
        title_en: c.meta.title_en.clone(),
        family_badge: c.meta.family.short_badge(Language::Zh).to_string(),
        formula_short: c.meta.formula_short.clone(),
      });
    }
  }

  let curated_models = crate::atlas_store::list_curated_models_merged().await;
  let related_models: Vec<RelatedModelLink> = curated_models
    .into_iter()
    .filter(|m| model_matches_math(m, found))
    .take(6)
    .map(|m| RelatedModelLink {
      spec_line: format!("Attn: {} · FFN: {}", m.attention_short, m.ffn_short),
      slug: m.slug,
      name: m.name,
      family_badge: m.family_badge,
    })
    .collect();

  Ok(Some(MathDetail {
    meta: found.meta.clone(),
    body_md: found.body_md.clone(),
    related_components,
    related_models,
    siblings,
  }))
}

// ─────────────────────────────────────────────────────────────────────────────
// `/ai/math`：AI 数学基石全景索引页（9 大支柱 + 推荐推导拓扑条 + 下游领域筛选）
// ─────────────────────────────────────────────────────────────────────────────

#[component]
pub fn AiMathAtlasPage() -> Element {
  let lang = current_lang();
  let title =
    lang.pick("AI 与大模型数学基础动态推导图鉴", "AI & LLM Math Foundations Proof Lab").to_string();
  let desc = lang
    .pick(
      "打破符号墙：从奇异值分解、矩阵微积分 VJP、概率散度与变分 ELBO，到傅里叶图谱、非凸优化 Muon、扩散伊藤 SDE、流匹配 CFM 与强化学习 PPO/DPO/GRPO 及 IEEE-754 浮点算术的零跳步色标同步推导实验室。",
      "Zero-skip color-synced mathematical proofs paired with live pure-Rust WASM geometric, variational, spectral, SDE/Flow, and RL alignment sandboxes.",
    )
    .to_string();

  rsx! {
    PageMeta {
      title: title.clone(),
      description: desc.clone(),
      path: "/ai/math".to_string(),
    }

    section { class: "py-10 sm:py-12 min-h-screen bg-white dark:bg-slate-950",
      div { class: "max-w-7xl mx-auto px-4 sm:px-6",
        SuspenseBoundary {
          fallback: |_| rsx! {
            div { class: "flex items-center justify-center py-24",
              Spinner { size: SpinnerSize::Lg, class: "border-t-primary" }
            }
          },
          AiMathLoaded { title, desc }
        }
      }
    }
  }
}

#[component]
fn AiMathLoaded(title: String, desc: String) -> Element {
  let lang = current_lang();
  let res = use_server_future(list_ai_math_entries)?;
  let items = match res() {
    Some(Ok(list)) => list,
    Some(Err(_)) => {
      return rsx! {
        widgets::LoadFailed {
          message: lang.pick("数学推导图鉴加载失败，请稍后重试", "Failed to load AI Math Proof Lab").to_string(),
        }
      };
    }
    None => Vec::new(),
  };

  let mut pillar_filter: Signal<Option<MathPillar>> = use_signal(|| None);
  let mut domain_filter = use_signal(|| "all".to_string());
  let mut search_q = use_signal(String::new);

  let cur_pil = pillar_filter();
  let cur_dom = domain_filter();
  let q = search_q().trim().to_ascii_lowercase();

  let filtered: Vec<MathSummary> = items
    .iter()
    .filter(|it| cur_pil.is_none_or(|p| it.pillar == p))
    .filter(|it| cur_dom == "all" || it.domains.iter().any(|d| d == &cur_dom))
    .filter(|it| {
      if q.is_empty() {
        true
      } else {
        it.slug.to_ascii_lowercase().contains(&q)
          || it.title.to_ascii_lowercase().contains(&q)
          || it.title_en.to_ascii_lowercase().contains(&q)
          || it.summary.to_ascii_lowercase().contains(&q)
          || it.tags.iter().any(|t| t.to_ascii_lowercase().contains(&q))
      }
    })
    .cloned()
    .collect();

  rsx! {
    div { class: "space-y-7",
      // 英雄头卡 + 三层 AI 知识飞轮快捷互跳
      div { class: "rounded-2xl border border-violet-200/80 dark:border-violet-900/70 bg-linear-to-br from-violet-50/90 via-white to-amber-50/60 dark:from-violet-950/50 dark:via-slate-900 dark:to-slate-950 p-6 sm:p-8",
        div { class: "flex flex-wrap items-center justify-between gap-4",
          div { class: "max-w-3xl",
            div { class: "inline-flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-violet-600 dark:text-violet-400",
              span { class: "px-2.5 py-0.5 rounded-full bg-violet-600 text-white text-[10px]", "Zero-Skip Proof Lab" }
              span { "Color-Synced MathML · Pure Rust WASM Geometry" }
            }
            h1 { class: "mt-2 text-2xl sm:text-3xl font-extrabold tracking-tight text-slate-900 dark:text-white",
              "{title}"
            }
            p { class: "mt-2 text-sm sm:text-base text-slate-600 dark:text-slate-300 leading-relaxed",
              "{desc}"
            }
          }
          div { class: "flex flex-wrap gap-2",
            Link {
              to: "/ai/components",
              class: "px-3.5 py-2 rounded-xl border border-emerald-200 dark:border-emerald-800 bg-white dark:bg-slate-900 text-xs font-bold text-emerald-700 dark:text-emerald-300 hover:border-emerald-500 transition-colors",
              {lang.pick("🔬 组件与算子图鉴 (/ai/components) →", "🔬 Operators Atlas →")}
            }
            Link {
              to: "/models",
              class: "px-3.5 py-2 rounded-xl bg-violet-600 text-white text-xs font-bold hover:bg-violet-700 transition-colors",
              {lang.pick("🏛️ 旗舰模型整机图鉴 (/models) →", "🏛️ Flagship Models →")}
            }
          }
        }

        // 推荐学习依赖拓扑条（三阶递进：基础空间微分 → 分布·频域·优化 → 前沿流匹配与强化对齐）
        div { class: "mt-5 grid grid-cols-1 md:grid-cols-3 gap-2.5 text-xs",
          for (stage_tag, stage_title, stage_desc, target_pil) in [
            (
              "Stage I · 空间与微分地基",
              "01 线性几何 → 02 矩阵自动微分 → 03 概率统计",
              "打通 SVD 低秩逼近、Trace Trick、反向 VJP 与 Kaiming 方差守恒",
              MathPillar::LinearGeometry,
            ),
            (
              "Stage II · 信息、谱与非凸优化",
              "04 散度变分 → 05 傅里叶图谱 → 06/07 优化与损失",
              "掌握 Forward/Reverse KL、ELBO、频谱偏置、AdamW/Muon 与蒸馏温度",
              MathPillar::InformationVariational,
            ),
            (
              "Stage III · 生成流与对齐前沿",
              "08 扩散 SDE 与流匹配 → 09 RL 对齐与浮点位宽",
              "证透 Tweedie、CFM 条件流等价、PPO 截断、DPO/GRPO 与 BF16/FP8",
              MathPillar::RlAlignmentPrecision,
            ),
          ] {
            button {
              key: "{stage_tag}",
              r#type: "button",
              class: "text-left rounded-xl border border-violet-200/70 dark:border-slate-800 bg-white/85 dark:bg-slate-900/80 p-3 hover:border-violet-500 transition-colors cursor-pointer",
              onclick: move |_| pillar_filter.set(Some(target_pil)),
              div { class: "text-[10px] font-mono font-bold uppercase text-violet-600 dark:text-violet-400", "{stage_tag}" }
              div { class: "mt-0.5 font-extrabold text-slate-900 dark:text-white", "{stage_title}" }
              div { class: "mt-0.5 text-[11px] text-slate-500 dark:text-slate-400 line-clamp-1", "{stage_desc}" }
            }
          }
        }

        // 下游领域与搜索栏
        div { class: "mt-5 pt-5 border-t border-violet-200/60 dark:border-slate-800 space-y-3",
          div { class: "flex flex-col sm:flex-row gap-3 items-stretch sm:items-center justify-between",
            div { class: "flex flex-wrap gap-1.5",
              for (d_id, d_zh, d_en) in [
                ("all", "全部下游任务", "All Downstream"),
                ("cnn_audio", "CNN·音频·图谱", "CNN, Audio & Graph"),
                ("llm", "Transformer & LLM", "Transformer & LLM"),
                ("diffusion", "扩散与流生成", "Diffusion & Flow"),
                ("rl_align", "强化学习与对齐", "RL & Alignment"),
                ("quant_opt", "量化与现代优化", "Quant & Optim"),
              ] {
                button {
                  key: "{d_id}",
                  r#type: "button",
                  class: if cur_dom == d_id {
                    "px-3 py-1 rounded-full text-xs font-bold bg-slate-900 text-white dark:bg-white dark:text-slate-900 cursor-pointer"
                  } else {
                    "px-3 py-1 rounded-full text-xs font-medium bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-300 hover:border-violet-400 cursor-pointer"
                  },
                  onclick: move |_| domain_filter.set(d_id.to_string()),
                  "{lang.pick(d_zh, d_en)}"
                }
              }
            }
            div { class: "w-full sm:w-72",
              Input {
                class: "text-xs bg-white dark:bg-slate-950",
                placeholder: lang.pick("搜索定理、符号或算法（如 SVD / VJP / KL / PPO）", "Search theorems (e.g. SVD, VJP, KL, PPO)"),
                "aria-label": "Search AI math foundations",
                value: search_q(),
                on_value_change: move |v: String| search_q.set(v),
              }
            }
          }

          // 9 大数学支柱胶囊筛选栏
          div { class: "flex flex-wrap gap-1.5 pt-1",
            button {
              r#type: "button",
              class: if cur_pil.is_none() {
                "px-2.5 py-1 rounded-lg text-xs font-semibold bg-violet-600 text-white cursor-pointer"
              } else {
                "px-2.5 py-1 rounded-lg text-xs bg-white/80 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-slate-600 dark:text-slate-300 cursor-pointer"
              },
              onclick: move |_| pillar_filter.set(None),
              {lang.pick("全部 9 大支柱", "All 9 Pillars")}
              " ({items.len()})"
            }
            for pil in MathPillar::ALL.iter().copied() {
              {
                let cnt = items.iter().filter(|i| i.pillar == pil).count();
                if cnt > 0 {
                  rsx! {
                    button {
                      key: "{pil.short_badge(Language::En)}",
                      r#type: "button",
                      class: if cur_pil == Some(pil) {
                        "px-2.5 py-1 rounded-lg text-xs font-semibold bg-violet-600 text-white cursor-pointer"
                      } else {
                        "px-2.5 py-1 rounded-lg text-xs bg-white/80 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-slate-600 dark:text-slate-300 hover:border-violet-400 cursor-pointer"
                      },
                      onclick: move |_| pillar_filter.set(Some(pil)),
                      "{pil.label(lang)} ({cnt})"
                    }
                  }
                } else {
                  rsx! {}
                }
              }
            }
          }
        }
      }

      // 定理卡片矩阵
      if filtered.is_empty() {
        div { class: "py-16 text-center text-sm text-slate-500 dark:text-slate-400",
          {lang.pick("未找到匹配筛选条件的数学推导词条", "No matching math proof entries found")}
        }
      } else {
        div { class: "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5",
          for item in filtered {
            MathSummaryCard { key: "{item.slug}", item }
          }
        }
      }
    }
  }
}

#[component]
fn MathSummaryCard(item: MathSummary) -> Element {
  let lang = current_lang();
  let href = format!("/ai/math/{}", item.slug);

  rsx! {
    Link {
      to: href,
      class: "group flex flex-col justify-between rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/70 p-5 hover:border-violet-400 dark:hover:border-violet-500 hover:shadow-md transition-all",
      div {
        div { class: "flex flex-wrap items-center justify-between gap-2 mb-2.5",
          span { class: "px-2 py-0.5 rounded-md bg-violet-500/15 text-violet-700 dark:text-violet-300 text-[11px] font-bold",
            "{item.pillar.short_badge(lang)}"
          }
          span { class: badge_class(BadgeVariant::Secondary, "text-[10px] font-mono"),
            "{item.lab_kind.badge_label(lang)}"
          }
        }

        h2 { class: "text-base sm:text-lg font-extrabold text-slate-900 dark:text-white group-hover:text-violet-600 dark:group-hover:text-violet-400 transition-colors",
          "{item.localized_title(lang)}"
        }

        p { class: "mt-2 text-xs text-slate-600 dark:text-slate-400 line-clamp-3 leading-relaxed",
          "{item.localized_summary(lang)}"
        }

        if !item.theorem_short.is_empty() {
          div { class: "mt-3 rounded-lg bg-slate-50 dark:bg-slate-950/80 border border-slate-200/80 dark:border-slate-800 px-2.5 py-1.5 text-[11px] font-mono text-violet-700 dark:text-violet-300 truncate",
            "{item.theorem_short}"
          }
        }
      }

      div { class: "mt-4 pt-3 border-t border-slate-100 dark:border-slate-800 flex items-center justify-between text-xs",
        div { class: "flex flex-wrap gap-1",
          for tag in item.tags.iter().take(3) {
            span { key: "{tag}", class: "px-1.5 py-0.5 rounded bg-slate-100 dark:bg-slate-800 text-[10px] font-mono text-slate-500 dark:text-slate-400",
              "#{tag}"
            }
          }
        }
        span { class: "font-bold text-violet-600 dark:text-violet-400 group-hover:translate-x-0.5 transition-transform shrink-0",
          {lang.pick("推演色标证明 →", "Open Proof Lab →")}
        }
      }
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// `/ai/math/:slug`：单定理交互几何画布 + 零跳步色标证明链 + 下游算子与模型反查页
// ─────────────────────────────────────────────────────────────────────────────

#[component]
pub fn AiMathDetailPage(slug: String) -> Element {
  rsx! {
    section { class: "py-8 sm:py-10 min-h-screen bg-white dark:bg-slate-950",
      div { class: "max-w-7xl mx-auto px-4 sm:px-6",
        SuspenseBoundary {
          fallback: |_| rsx! {
            div { class: "flex items-center justify-center py-24",
              Spinner { size: SpinnerSize::Lg, class: "border-t-primary" }
            }
          },
          AiMathDetailLoaded { slug }
        }
      }
    }
  }
}

#[component]
fn AiMathDetailLoaded(slug: String) -> Element {
  let lang = current_lang();
  let res = use_server_future(use_reactive!(|slug| async move { get_ai_math_detail(slug).await }))?;

  let detail = match res() {
    Some(Ok(Some(d))) => d,
    Some(Ok(None)) => {
      return rsx! {
        widgets::NotFound {
          message: lang.pick("未找到该数学推导词条", "Math proof entry not found").to_string(),
        }
      };
    }
    Some(Err(_)) => {
      return rsx! {
        widgets::LoadFailed {
          message: lang.pick("加载数学推导详情失败", "Failed to load math proof detail").to_string(),
        }
      };
    }
    None => return rsx! {},
  };

  let toc = widgets::heading_outline(&detail.body_md);
  let title = detail.meta.localized_title(lang).to_string();
  let desc = detail.meta.localized_summary(lang).to_string();

  rsx! {
    PageMeta {
      title: title.clone(),
      description: desc.clone(),
      path: format!("/ai/math/{}", detail.meta.slug),
    }

    div { class: "space-y-6",
      // 面包屑与上层算子图鉴快捷跳转
      div { class: "flex flex-wrap items-center justify-between gap-3 text-xs",
        div { class: "flex flex-wrap items-center gap-2 text-slate-500 dark:text-slate-400",
          Link { to: "/ai", class: "hover:text-violet-600", "AI" }
          span { "/" }
          Link { to: "/ai/math", class: "hover:text-violet-600 font-semibold",
            {lang.pick("AI 数学基石", "Math Foundations")}
          }
          span { "/" }
          span { class: "text-violet-600 dark:text-violet-400 font-bold",
            "{detail.meta.pillar.label(lang)}"
          }
        }
        Link {
          to: "/ai/components",
          class: "font-semibold text-emerald-600 dark:text-emerald-400 hover:underline",
          {lang.pick("查看这些定理支撑的 AI 算子实验室 →", "Browse Downstream Operator Labs →")}
        }
      }

      // 定理头卡 + 色标图例条 + 同支柱兄弟定理切换
      div { class: card_class("p-5 sm:p-6 bg-slate-50/70 dark:bg-slate-900/70"),
        div { class: "flex flex-wrap items-center gap-2 mb-2",
          span { class: "px-2.5 py-0.5 rounded-md bg-violet-600 text-white text-xs font-bold",
            "{detail.meta.pillar.short_badge(lang)}"
          }
          if !detail.meta.proof_steps_badge.is_empty() {
            span { class: badge_class(BadgeVariant::Secondary, "font-mono text-xs"),
              "{detail.meta.proof_steps_badge}"
            }
          }
          for dm in detail.meta.domains.iter() {
            span { key: "{dm}", class: "px-2 py-0.5 rounded bg-slate-200/70 dark:bg-slate-800 text-[11px] font-mono text-slate-600 dark:text-slate-300",
              "{dm}"
            }
          }
        }
        h1 { class: "text-2xl sm:text-3xl font-extrabold tracking-tight text-slate-900 dark:text-white",
          "{title}"
        }
        p { class: "mt-2 text-sm sm:text-base text-slate-600 dark:text-slate-300 leading-relaxed max-w-4xl",
          "{desc}"
        }

        // 四色语义双向锚定图例说明
        div { class: "mt-3.5 flex flex-wrap items-center gap-3 text-[11px] text-slate-600 dark:text-slate-400",
          span { class: "font-bold text-slate-700 dark:text-slate-200",
            {lang.pick("色标锚定约定：", "Color-Sync Legend:")}
          }
          span { class: "inline-flex items-center gap-1 font-mono text-indigo-700 dark:text-indigo-300",
            span { class: "w-2 h-2 rounded-full bg-indigo-600" }
            "靛蓝 #4f46e5 = 输入基/先验/前向原值"
          }
          span { class: "inline-flex items-center gap-1 font-mono text-emerald-700 dark:text-emerald-300",
            span { class: "w-2 h-2 rounded-full bg-emerald-600" }
            "翠绿 #059669 = 主不变量/最优下界/守恒项"
          }
          span { class: "inline-flex items-center gap-1 font-mono text-amber-700 dark:text-amber-300",
            span { class: "w-2 h-2 rounded-full bg-amber-500" }
            "琥珀 #d97706 = 几何缩放/条件数/均值覆盖"
          }
          span { class: "inline-flex items-center gap-1 font-mono text-rose-700 dark:text-rose-300",
            span { class: "w-2 h-2 rounded-full bg-rose-600" }
            "玫红 #e11d48 = 截断残差/伴随梯度/KL 惩罚间隙"
          }
        }

        if detail.siblings.len() > 1 {
          div { class: "mt-4 pt-3 border-t border-slate-200 dark:border-slate-800 flex flex-wrap items-center gap-1.5 text-xs",
            span { class: "text-slate-400 mr-1",
              {lang.pick("同支柱定理切换：", "Pillar Siblings:")}
            }
            for sib in detail.siblings.iter() {
              Link {
                key: "{sib.slug}",
                to: format!("/ai/math/{}", sib.slug),
                class: if sib.slug == detail.meta.slug {
                  "px-2.5 py-1 rounded-md font-bold bg-violet-600 text-white"
                } else {
                  "px-2.5 py-1 rounded-md bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-300 hover:border-violet-400"
                },
                "{sib.localized_title(lang)}"
              }
            }
          }
        }
      }

      // 1. 置顶动态 WASM 色标同步几何/概率推导画布
      MathLabWorkbench {
        key: "{detail.meta.slug}",
        kind: detail.meta.lab_kind,
        initial_mode: detail.meta.lab_mode.clone(),
        embedded: false,
      }

      // 2. 零跳步色标同步证明链与数值验算代码（带右侧粘性目录）
      div { class: "grid grid-cols-1 lg:grid-cols-12 gap-8 items-start pt-2",
        article { class: "lg:col-span-9 min-w-0 prose prose-slate dark:prose-invert max-w-none",
          Markdown {
            content: detail.body_md.clone(),
            blog_id: format!("ai-math-{}", detail.meta.slug),
          }
        }
        aside { class: "hidden lg:block lg:col-span-3 sticky top-20",
          OnThisPage {
            entries: toc,
            title: lang.pick("推导步骤目录", "Proof Outline").to_string(),
          }
        }
      }

      // 3. 反向导航 A：该定理直接支撑的深度学习算子与原语（/ai/components/:slug）
      if !detail.related_components.is_empty() {
        div { class: "pt-6 border-t border-slate-200 dark:border-slate-800",
          h2 { class: "text-lg font-extrabold text-slate-900 dark:text-white mb-3",
            {lang.pick("该数学定理直接支撑的算子与组件（点击进入算子实验室）", "Downstream AI Operators Powered by This Theorem")}
          }
          div { class: "grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3.5",
            for c in detail.related_components.iter() {
              Link {
                key: "{c.slug}",
                to: format!("/ai/components/{}", c.slug),
                class: "group rounded-xl border border-emerald-200/80 dark:border-emerald-900/60 bg-emerald-50/40 dark:bg-slate-900/60 p-3.5 hover:border-emerald-500 transition-all",
                div { class: "flex items-center justify-between gap-2",
                  span { class: "font-bold text-sm text-slate-900 dark:text-white group-hover:text-emerald-600 dark:group-hover:text-emerald-400 line-clamp-1",
                    "{c.localized_title(lang)}"
                  }
                  span { class: "text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-500/15 text-emerald-700 dark:text-emerald-300 shrink-0",
                    "{c.family_badge}"
                  }
                }
                if !c.formula_short.is_empty() {
                  div { class: "mt-1.5 text-[11px] font-mono text-slate-500 dark:text-slate-400 truncate",
                    "{c.formula_short}"
                  }
                }
              }
            }
          }
        }
      }

      // 4. 反向导航 B：采用该数学基石的旗舰大模型（/models/:slug）
      if !detail.related_models.is_empty() {
        div { class: "pt-6 border-t border-slate-200 dark:border-slate-800",
          h2 { class: "text-lg font-extrabold text-slate-900 dark:text-white mb-3",
            {lang.pick("依赖该定理工程落地的代表性旗舰模型", "Flagship Models Grounded in This Theorem")}
          }
          div { class: "grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3.5",
            for m in detail.related_models.iter() {
              Link {
                key: "{m.slug}",
                to: format!("/models/{}", m.slug),
                class: "group rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/70 dark:bg-slate-900/60 p-3.5 hover:border-violet-400 transition-all",
                div { class: "flex items-center justify-between gap-2",
                  span { class: "font-bold text-sm text-slate-900 dark:text-white group-hover:text-violet-600 dark:group-hover:text-violet-400",
                    "{m.name}"
                  }
                  span { class: "text-[10px] font-mono px-2 py-0.5 rounded bg-violet-500/15 text-violet-700 dark:text-violet-300",
                    "{m.family_badge}"
                  }
                }
                div { class: "mt-1.5 text-[11px] font-mono text-slate-500 dark:text-slate-400 truncate",
                  "{m.spec_line}"
                }
              }
            }
          }
        }
      }
    }
  }
}

#[cfg(all(test, feature = "server"))]
mod tests {
  use super::*;

  #[test]
  fn builtin_ai_math_entries_parse_and_cover_all_six_wasm_kernels() {
    let loaded = load_all_math_entries();
    assert!(
      loaded.len() >= 6,
      "expected at least 6 builtin flagship ai-math exemplars covering all 6 M1+M2 kernels, got {}",
      loaded.len()
    );
    for expected_kind in [
      MathLabKind::LinearTransform2d,
      MathLabKind::AutogradAndLoss,
      MathLabKind::DistributionDivergence,
      MathLabKind::SpectralAndGraph,
      MathLabKind::OptimizerAndFlow,
      MathLabKind::RlAndBitPrecision,
    ] {
      assert!(
        loaded.iter().any(|e| e.meta.lab_kind == expected_kind),
        "missing builtin exemplar for math kernel {expected_kind:?}"
      );
    }
    for item in loaded.iter() {
      assert!(!item.meta.slug.is_empty());
      assert!(!item.meta.title.is_empty());
      assert!(!item.body_md.is_empty());
    }
  }
}
