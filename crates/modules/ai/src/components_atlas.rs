//! 深度学习与大模型全谱系组件/算子图鉴（`/ai/components` 索引页 + `/ai/components/:slug` 详情页 + 双向模型挂接）。
//!
//! 数据源合并规则（零数据库依赖，天然兼容 `basic` 档位与 `SIGHUP` 秒级热刷新）：
//! 1. 先加载框架内置标杆词条 `assets/ai-components/<slug>/index.md`；
//! 2. 若挂载了外部内容仓库 `<CONTENT_DIR>/ai-components/<slug>/index.md`，自动合并覆盖同名 `slug` 并追加全量扩展词条。

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

use crate::arch::{InspectorFocus, ModelBlueprint, OutputHeadSpec};
use crate::atlas::CuratedModelSummary;
use crate::component_labs::{ComponentLabKind, ComponentLabWorkbench};

fn current_lang() -> Language {
  try_consume_context::<Signal<Language>>().map(|s| s()).unwrap_or_default()
}

/// 12 大全谱系算子与组件家族。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ComponentFamily {
  LinearDense,
  #[default]
  Activation,
  Convolution,
  PoolingResample,
  NormConditioning,
  ResidualBackbone,
  RecurrentSsm,
  Attention,
  TokenPositional,
  AudioCodec,
  VisionMultimodal,
  MoePeftQuantDecode,
}

impl ComponentFamily {
  pub const ALL: &'static [Self] = &[
    Self::LinearDense,
    Self::Activation,
    Self::Convolution,
    Self::PoolingResample,
    Self::NormConditioning,
    Self::ResidualBackbone,
    Self::RecurrentSsm,
    Self::Attention,
    Self::TokenPositional,
    Self::AudioCodec,
    Self::VisionMultimodal,
    Self::MoePeftQuantDecode,
  ];

  pub fn parse_str(raw: &str) -> Self {
    match raw.trim().to_ascii_lowercase().as_str() {
      "linear" | "linear_dense" | "dense" | "mlp" => Self::LinearDense,
      "activation" | "activations" => Self::Activation,
      "conv" | "convolution" => Self::Convolution,
      "pool" | "pooling" | "pooling_resample" | "resample" => Self::PoolingResample,
      "norm" | "normalization" | "norm_conditioning" | "conditioning" => Self::NormConditioning,
      "residual" | "residual_backbone" | "backbone" | "skip" => Self::ResidualBackbone,
      "rnn" | "recurrent" | "recurrent_ssm" | "ssm" => Self::RecurrentSsm,
      "attn" | "attention" => Self::Attention,
      "pos" | "positional" | "token_positional" | "tokenizer" => Self::TokenPositional,
      "audio" | "audio_codec" | "speech" | "codec" => Self::AudioCodec,
      "vision" | "vision_multimodal" | "multimodal" | "gnn" | "vae" => Self::VisionMultimodal,
      "moe" | "peft" | "quant" | "decode" | "moe_peft_quant_decode" => Self::MoePeftQuantDecode,
      _ => Self::Activation,
    }
  }

  pub fn label(self, lang: Language) -> &'static str {
    match self {
      Self::LinearDense => lang.pick("01 · 线性与基础原语", "01 · Linear & Primitives"),
      Self::Activation => lang.pick("02 · 激活函数族", "02 · Activations"),
      Self::Convolution => lang.pick("03 · 卷积算子族", "03 · Convolutions"),
      Self::PoolingResample => lang.pick("04 · 池化与重采样", "04 · Pooling & Resample"),
      Self::NormConditioning => lang.pick("05 · 归一化与调制", "05 · Norm & Conditioning"),
      Self::ResidualBackbone => lang.pick("06 · 残差与骨干块", "06 · Residual & Backbones"),
      Self::RecurrentSsm => lang.pick("07 · 循环与状态空间", "07 · RNN & Selective SSM"),
      Self::Attention => lang.pick("08 · 注意力谱系", "08 · Attention Mechanisms"),
      Self::TokenPositional => lang.pick("09 · 分词与位置编码", "09 · Token & Positional"),
      Self::AudioCodec => lang.pick("10 · 音频与神经码本", "10 · Audio & Codecs"),
      Self::VisionMultimodal => lang.pick("11 · 视觉·多模态·图", "11 · Vision, VLM & GNN"),
      Self::MoePeftQuantDecode => lang.pick("12 · MoE·微调·量化·解码", "12 · MoE, PEFT & Decode"),
    }
  }

  pub fn short_badge(self, lang: Language) -> &'static str {
    match self {
      Self::LinearDense => lang.pick("线性原语", "Linear"),
      Self::Activation => lang.pick("激活函数", "Activation"),
      Self::Convolution => lang.pick("卷积算子", "Convolution"),
      Self::PoolingResample => lang.pick("池化重排", "Pooling"),
      Self::NormConditioning => lang.pick("归一化/调制", "Norm/AdaLN"),
      Self::ResidualBackbone => lang.pick("残差拓扑", "Residual"),
      Self::RecurrentSsm => lang.pick("循环/SSM", "RNN/SSM"),
      Self::Attention => lang.pick("注意力机制", "Attention"),
      Self::TokenPositional => lang.pick("分词/位置", "Positional"),
      Self::AudioCodec => lang.pick("音频/声码", "Audio/Codec"),
      Self::VisionMultimodal => lang.pick("视觉/多模态", "Vision/VLM"),
      Self::MoePeftQuantDecode => lang.pick("MoE/微调/量化", "MoE/PEFT/Quant"),
    }
  }
}

/// 组件列表卡片摘要元数据。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ComponentSummary {
  pub slug: String,
  pub title: String,
  pub title_en: String,
  pub family: ComponentFamily,
  /// `"classic"` | `"modern"` | `"frontier"`
  pub era: String,
  /// `"vision"` | `"nlp"` | `"audio"` | `"video"` | `"multimodal"` | `"graph"` | `"general"`
  pub modalities: Vec<String>,
  pub order: i32,
  pub featured: bool,
  pub lab_kind: ComponentLabKind,
  pub lab_mode: String,
  pub formula_short: String,
  pub complexity_badge: String,
  pub summary: String,
  pub summary_en: String,
  pub tags: Vec<String>,
  pub model_keywords: Vec<String>,
}

impl ComponentSummary {
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

/// 单个算子/组件详情页完整载荷。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ComponentDetail {
  pub meta: ComponentSummary,
  pub body_md: String,
  pub related_models: Vec<CuratedModelSummary>,
  pub siblings: Vec<ComponentSummary>,
}

/// 根据组件 `slug` 推断 `<ComponentLab id="..." />` 的默认交互内核与模式（纯前端与服务端双端确定性一致）。
pub fn infer_lab_preset_by_slug(slug: &str, mode_override: &str) -> (ComponentLabKind, String) {
  let s = slug.trim().to_ascii_lowercase();
  let (kind, default_mode) = if s.contains("rope") || s.contains("yarn") || s.contains("pos") {
    (ComponentLabKind::Positional, "yarn")
  } else if s.contains("norm") || s.contains("adaln") || s.contains("film") {
    (ComponentLabKind::NormPrecision, "rmsnorm")
  } else if s.contains("conv") || s.contains("pool") || s.contains("roi") || s.contains("shuffle") {
    (ComponentLabKind::GridKernel, "depthwise_separable")
  } else {
    (ComponentLabKind::Activation, "swiglu")
  };
  let m = if mode_override.trim().is_empty() {
    default_mode.to_string()
  } else {
    mode_override.to_string()
  };
  (kind, m)
}

/// 为 `/models/:slug` 与 `/ai/visualizer` 右侧算子放大镜生成上下文关联的动态组件实验室深链。
pub fn recommended_component_links(
  bp: &ModelBlueprint,
  focus: InspectorFocus,
  lang: Language,
) -> Vec<(&'static str, &'static str, &'static str)> {
  let mut out = Vec::new();
  match focus {
    InspectorFocus::Attention => {
      out.push((
        "/ai/components/rope-and-yarn-extrapolation",
        "Positional Lab",
        lang.pick("旋转位置编码 RoPE / YaRN 外推实验台", "RoPE & YaRN Extrapolation Lab"),
      ));
      out.push((
        "/ai/components/rmsnorm-and-adaln-zero",
        "QK-Norm & FP16",
        lang.pick("注意力 QK-Norm 与半精度溢出防线", "QK-Norm & FP16 Overflow Guard"),
      ));
    }
    InspectorFocus::FfnOrMoe => {
      out.push((
        "/ai/components/activations-gelu-silu-swiglu",
        "Activation Lab",
        lang.pick("GELU / SiLU / SwiGLU 门控激活与导数实验台", "GELU, SiLU & SwiGLU Lab"),
      ));
    }
    InspectorFocus::NormAndResidual => {
      out.push((
        "/ai/components/rmsnorm-and-adaln-zero",
        if bp.is_diffusion_pipeline() { "AdaLN-Zero Lab" } else { "RMSNorm Lab" },
        if bp.is_diffusion_pipeline() {
          lang.pick("AdaLN-Zero 零门控调制与归一化实验台", "AdaLN-Zero & GroupNorm Lab")
        } else {
          lang.pick("RMSNorm / LayerNorm 统计切片与精度实验台", "RMSNorm & Precision Lab")
        },
      ));
    }
    InspectorFocus::Vision | InspectorFocus::Audio => {
      out.push((
        "/ai/components/conv-and-depthwise-separable",
        "Conv & Pool Lab",
        lang.pick("卷积、深度可分离卷积与下采样实验台", "Conv, Depthwise & Pooling Lab"),
      ));
      out.push((
        "/ai/components/rope-and-yarn-extrapolation",
        "2D Axial RoPE",
        lang.pick("多模态二维轴向位置编码实验台", "2D Axial & Multimodal RoPE Lab"),
      ));
    }
    InspectorFocus::Embedding | InspectorFocus::LmHead | InspectorFocus::MtpHead => {
      if matches!(bp.output_head, OutputHeadSpec::LatentDecoder { .. }) {
        out.push((
          "/ai/components/conv-and-depthwise-separable",
          "Deconv Checkerboard",
          lang.pick("VAE 解码转置卷积与棋盘伪影实验台", "Transposed Conv Checkerboard Lab"),
        ));
      } else {
        out.push((
          "/ai/components/rope-and-yarn-extrapolation",
          "RoPE Geometry",
          lang.pick("序列位置几何与上下文窗口扩展实验台", "Positional Geometry Lab"),
        ));
      }
    }
  }
  out
}

// ─────────────────────────────────────────────────────────────────────────────
// 服务端目录扫描与双源合并（`assets/ai-components` + `<CONTENT_DIR>/ai-components`）
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "server")]
#[derive(Debug, Clone, Deserialize, Default)]
struct RawComponentFrontmatter {
  #[serde(default)]
  slug: String,
  #[serde(default)]
  title: String,
  #[serde(default)]
  title_en: String,
  #[serde(default)]
  family: String,
  #[serde(default = "default_era")]
  era: String,
  #[serde(default)]
  modalities: Vec<String>,
  #[serde(default = "default_order")]
  order: i32,
  #[serde(default)]
  featured: bool,
  #[serde(default)]
  lab_kind: String,
  #[serde(default)]
  lab_mode: String,
  #[serde(default)]
  formula_short: String,
  #[serde(default)]
  complexity_badge: String,
  #[serde(default, alias = "description")]
  summary: String,
  #[serde(default)]
  summary_en: String,
  #[serde(default)]
  tags: Vec<String>,
  #[serde(default)]
  model_keywords: Vec<String>,
}

#[cfg(feature = "server")]
fn default_era() -> String {
  "modern".to_string()
}

#[cfg(feature = "server")]
fn default_order() -> i32 {
  100
}

#[cfg(feature = "server")]
#[derive(Debug, Clone)]
pub struct StoredComponentEntry {
  pub meta: ComponentSummary,
  pub body_md: String,
}

#[cfg(feature = "server")]
static COMPONENTS_CACHE: app_core::utils::DirListingCache<Vec<StoredComponentEntry>> =
  app_core::utils::DirListingCache::new();

#[cfg(feature = "server")]
pub fn invalidate_components_cache() {
  COMPONENTS_CACHE.invalidate();
}

#[cfg(feature = "server")]
fn parse_component_dir(dir: &Path, slug_fallback: &str) -> Result<StoredComponentEntry, String> {
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
  let fm: RawComponentFrontmatter = serde_yaml::from_str(fm_yaml)
    .map_err(|e| format!("解析 {} YAML 失败: {e}", md_path.display()))?;

  let slug =
    if fm.slug.trim().is_empty() { slug_fallback.to_string() } else { fm.slug.trim().to_string() };
  let (inferred_kind, inferred_mode) = infer_lab_preset_by_slug(&slug, &fm.lab_mode);
  let lab_kind = if fm.lab_kind.trim().is_empty() {
    inferred_kind
  } else {
    ComponentLabKind::parse_str(&fm.lab_kind)
  };
  let modalities =
    if fm.modalities.is_empty() { vec!["general".to_string()] } else { fm.modalities };

  let meta = ComponentSummary {
    slug: slug.clone(),
    title: if fm.title.trim().is_empty() { slug } else { fm.title },
    title_en: fm.title_en,
    family: ComponentFamily::parse_str(&fm.family),
    era: fm.era,
    modalities,
    order: fm.order,
    featured: fm.featured,
    lab_kind,
    lab_mode: inferred_mode,
    formula_short: fm.formula_short,
    complexity_badge: fm.complexity_badge,
    summary: fm.summary,
    summary_en: fm.summary_en,
    tags: fm.tags,
    model_keywords: fm.model_keywords,
  };

  Ok(StoredComponentEntry { meta, body_md: body.trim_start().to_string() })
}

#[cfg(feature = "server")]
fn scan_dir_into(root: &Path, merged: &mut BTreeMap<String, StoredComponentEntry>) {
  let Ok(rd) = std::fs::read_dir(root) else { return };
  let mut paths: Vec<PathBuf> = rd.flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect();
  paths.sort();
  for path in paths {
    let Some(name) = path.file_name().and_then(|s| s.to_str()) else { continue };
    if name.starts_with('.') || name.starts_with('_') || !app_core::utils::is_safe_slug(name) {
      continue;
    }
    match parse_component_dir(&path, name) {
      Ok(entry) => {
        merged.insert(entry.meta.slug.clone(), entry);
      }
      Err(err) => {
        tracing::warn!(dir = %path.display(), error = %err, "skip invalid ai-component entry");
      }
    }
  }
}

/// 加载合并后的全部组件词条（内置 `assets/ai-components` + 外部 `<CONTENT_DIR>/ai-components`）。
#[cfg(feature = "server")]
pub fn load_all_components() -> Arc<Vec<StoredComponentEntry>> {
  let builtin_root = app_core::utils::get_asset_root().join("ai-components");
  let content_root = app_core::utils::content_dir("ai-components");
  let is_md = |p: &Path| {
    p.file_name().and_then(|n| n.to_str()).is_some_and(|n| matches!(n, "index.md" | "index.mdx"))
  };
  let mut fp = app_core::utils::fingerprint_for_dir(&builtin_root, is_md);
  if content_root != builtin_root {
    fp.extend(app_core::utils::fingerprint_for_dir(&content_root, is_md));
  }

  COMPONENTS_CACHE.get_or_rebuild(fp, || {
    let mut map = BTreeMap::new();
    scan_dir_into(&builtin_root, &mut map);
    if content_root != builtin_root {
      scan_dir_into(&content_root, &mut map);
    }
    let mut items: Vec<StoredComponentEntry> = map.into_values().collect();
    items
      .sort_by(|a, b| a.meta.order.cmp(&b.meta.order).then_with(|| a.meta.slug.cmp(&b.meta.slug)));
    items
  })
}

/// 为嵌入式 Tantivy 全文检索提供全部组件词条索引文档。
#[cfg(feature = "server")]
pub fn search_docs() -> Vec<app_core::engines::doc_source::ExternalIndexedDoc> {
  load_all_components()
    .iter()
    .map(|entry| {
      let body = format!(
        "{} {} {} {} {}",
        entry.meta.summary,
        entry.meta.summary_en,
        entry.meta.tags.join(" "),
        entry.meta.formula_short,
        entry.body_md.chars().take(4000).collect::<String>(),
      );
      app_core::engines::doc_source::ExternalIndexedDoc {
        kind: "ai".to_string(),
        ref_id: format!("component-{}", entry.meta.slug),
        title: entry.meta.title.clone(),
        body,
        url: format!("/ai/components/{}", entry.meta.slug),
        created_at: String::new(),
      }
    })
    .collect()
}

#[cfg(feature = "server")]
fn model_matches_component(model: &CuratedModelSummary, comp: &ComponentSummary) -> bool {
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

  if comp.model_keywords.iter().any(|kw| {
    let k = kw.trim().to_ascii_lowercase();
    !k.is_empty() && hay.contains(&k)
  }) {
    return true;
  }

  match comp.family {
    ComponentFamily::Activation => {
      hay.contains("swiglu") || hay.contains("gelu") || hay.contains("silu")
    }
    ComponentFamily::Convolution => {
      model.has_vision
        || matches!(
          model.modality,
          crate::atlas_store::ModelModality::Audio | crate::atlas_store::ModelModality::Diffusion
        )
    }
    ComponentFamily::NormConditioning => true,
    ComponentFamily::TokenPositional => hay.contains("rope") || model.max_context >= 32768,
    _ => false,
  }
}

#[get("/api/ai/components")]
pub async fn list_ai_components() -> Result<Vec<ComponentSummary>, ServerFnError> {
  Ok(load_all_components().iter().map(|e| e.meta.clone()).collect())
}

#[get("/api/ai/components/:slug")]
pub async fn get_ai_component_detail(
  slug: String,
) -> Result<Option<ComponentDetail>, ServerFnError> {
  let clean = slug.trim();
  if !app_core::utils::is_safe_slug(clean) {
    return Ok(None);
  }
  let all = load_all_components();
  let Some(found) = all.iter().find(|e| e.meta.slug == clean) else {
    return Ok(None);
  };
  let siblings: Vec<ComponentSummary> =
    all.iter().filter(|e| e.meta.family == found.meta.family).map(|e| e.meta.clone()).collect();
  let curated_models = crate::atlas_store::list_curated_models_merged().await;
  let related_models: Vec<CuratedModelSummary> = curated_models
    .into_iter()
    .filter(|m| model_matches_component(m, &found.meta))
    .take(6)
    .collect();

  Ok(Some(ComponentDetail {
    meta: found.meta.clone(),
    body_md: found.body_md.clone(),
    related_models,
    siblings,
  }))
}

// ─────────────────────────────────────────────────────────────────────────────
// `/ai/components`：组件与算子百科全景索引页
// ─────────────────────────────────────────────────────────────────────────────

#[component]
pub fn AiComponentsAtlasPage() -> Element {
  let lang = current_lang();
  let title =
    lang.pick("深度学习与大模型组件算子图鉴", "Deep Learning & LLM Components Atlas").to_string();
  let desc = lang
    .pick(
      "从经典感知机、卷积、池化、激活函数与循环网络，到 RoPE、MLA、FlashAttention、AdaLN-Zero、神经语音码本与稀疏 MoE 的动态可视化、数学推导与 Rust/PyTorch 对照图鉴。",
      "Interactive WASM visualizations, mathematical derivations, and PyTorch/Rust implementations across convolutions, pooling, activations, normalization, RNN/SSM, attention, and frontier LLM primitives.",
    )
    .to_string();

  rsx! {
    PageMeta {
      title: title.clone(),
      description: desc.clone(),
      path: "/ai/components".to_string(),
    }

    section { class: "py-10 sm:py-12 min-h-screen bg-white dark:bg-slate-950",
      div { class: "max-w-7xl mx-auto px-4 sm:px-6",
        SuspenseBoundary {
          fallback: |_| rsx! {
            div { class: "flex items-center justify-center py-24",
              Spinner { size: SpinnerSize::Lg, class: "border-t-primary" }
            }
          },
          AiComponentsLoaded { title, desc }
        }
      }
    }
  }
}

#[component]
fn AiComponentsLoaded(title: String, desc: String) -> Element {
  let lang = current_lang();
  let res = use_server_future(list_ai_components)?;
  let items = match res() {
    Some(Ok(list)) => list,
    Some(Err(_)) => {
      return rsx! {
        widgets::LoadFailed {
          message: lang.pick("组件图鉴加载失败，请稍后重试", "Failed to load Components Atlas").to_string(),
        }
      };
    }
    None => Vec::new(),
  };

  let mut family_filter: Signal<Option<ComponentFamily>> = use_signal(|| None);
  let mut modality_filter = use_signal(|| "all".to_string());
  let mut search_q = use_signal(String::new);

  let cur_fam = family_filter();
  let cur_mod = modality_filter();
  let q = search_q().trim().to_ascii_lowercase();

  let filtered: Vec<ComponentSummary> = items
    .iter()
    .filter(|it| cur_fam.is_none_or(|f| it.family == f))
    .filter(|it| cur_mod == "all" || it.modalities.iter().any(|m| m == &cur_mod))
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
      // 顶部英雄标题与三层知识飞轮快捷导航
      div { class: "rounded-2xl border border-indigo-200/80 dark:border-indigo-900/70 bg-linear-to-br from-indigo-50/90 via-white to-sky-50/70 dark:from-indigo-950/50 dark:via-slate-900 dark:to-slate-950 p-6 sm:p-8",
        div { class: "flex flex-wrap items-center justify-between gap-4",
          div { class: "max-w-3xl",
            div { class: "inline-flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-indigo-600 dark:text-indigo-400",
              span { class: "px-2.5 py-0.5 rounded-full bg-indigo-600 text-white text-[10px]", "Operators & Primitives" }
              span { "Pure Rust WASM · Native MathML · PyTorch & Candle/Burn" }
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
              to: "/models",
              class: "px-3.5 py-2 rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 text-xs font-bold text-slate-700 dark:text-slate-200 hover:border-indigo-400 transition-colors",
              {lang.pick("← 宏观整机架构图鉴 (/models)", "← Flagship Model Atlas")}
            }
            Link {
              to: "/ai/visualizer",
              class: "px-3.5 py-2 rounded-xl bg-indigo-600 text-white text-xs font-bold hover:bg-indigo-700 transition-colors",
              {lang.pick("打开仓库实时解析器 ↗", "Live Repo Visualizer ↗")}
            }
          }
        }

        // 搜索与模态过滤栏
        div { class: "mt-6 pt-5 border-t border-indigo-200/60 dark:border-slate-800 space-y-3",
          div { class: "flex flex-col sm:flex-row gap-3 items-stretch sm:items-center justify-between",
            div { class: "flex flex-wrap gap-1.5",
              for (m_id, m_zh, m_en) in [
                ("all", "全部模态", "All Modalities"),
                ("general", "通用基础", "General"),
                ("vision", "视觉 CV", "Vision"),
                ("nlp", "语言 LLM", "NLP / LLM"),
                ("audio", "语音 Audio", "Audio"),
                ("video", "视频/扩散", "Video / Diffusion"),
                ("multimodal", "多模态 VLM", "Multimodal"),
              ] {
                button {
                  key: "{m_id}",
                  r#type: "button",
                  class: if cur_mod == m_id {
                    "px-3 py-1 rounded-full text-xs font-bold bg-slate-900 text-white dark:bg-white dark:text-slate-900 cursor-pointer"
                  } else {
                    "px-3 py-1 rounded-full text-xs font-medium bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-300 hover:border-indigo-400 cursor-pointer"
                  },
                  onclick: move |_| modality_filter.set(m_id.to_string()),
                  "{lang.pick(m_zh, m_en)}"
                }
              }
            }
            div { class: "w-full sm:w-72",
              Input {
                class: "text-xs bg-white dark:bg-slate-950",
                placeholder: lang.pick("搜索算子、缩写或公式（如 SwiGLU / Conv / RoPE）", "Search operators (e.g. SwiGLU, Conv, RoPE)"),
                "aria-label": "Search AI components and operators",
                value: search_q(),
                on_value_change: move |v: String| search_q.set(v),
              }
            }
          }

          // 12 大算子家族胶囊导航
          div { class: "flex flex-wrap gap-1.5 pt-1",
            button {
              r#type: "button",
              class: if cur_fam.is_none() {
                "px-2.5 py-1 rounded-lg text-xs font-semibold bg-indigo-600 text-white cursor-pointer"
              } else {
                "px-2.5 py-1 rounded-lg text-xs bg-white/80 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-slate-600 dark:text-slate-300 cursor-pointer"
              },
              onclick: move |_| family_filter.set(None),
              {lang.pick("全部 12 族", "All 12 Families")}
              " ({items.len()})"
            }
            for fam in ComponentFamily::ALL.iter().copied() {
              {
                let cnt = items.iter().filter(|i| i.family == fam).count();
                if cnt > 0 {
                  rsx! {
                    button {
                      key: "{fam.short_badge(Language::En)}",
                      r#type: "button",
                      class: if cur_fam == Some(fam) {
                        "px-2.5 py-1 rounded-lg text-xs font-semibold bg-indigo-600 text-white cursor-pointer"
                      } else {
                        "px-2.5 py-1 rounded-lg text-xs bg-white/80 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-slate-600 dark:text-slate-300 hover:border-indigo-400 cursor-pointer"
                      },
                      onclick: move |_| family_filter.set(Some(fam)),
                      "{fam.label(lang)} ({cnt})"
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

      // 算子卡片矩阵
      if filtered.is_empty() {
        div { class: "py-16 text-center text-sm text-slate-500 dark:text-slate-400",
          {lang.pick("未找到匹配筛选条件的算子词条", "No matching operator entries found")}
        }
      } else {
        div { class: "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5",
          for item in filtered {
            ComponentSummaryCard { key: "{item.slug}", item }
          }
        }
      }
    }
  }
}

#[component]
fn ComponentSummaryCard(item: ComponentSummary) -> Element {
  let lang = current_lang();
  let href = format!("/ai/components/{}", item.slug);

  rsx! {
    Link {
      to: href,
      class: "group flex flex-col justify-between rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/70 p-5 hover:border-indigo-400 dark:hover:border-indigo-500 hover:shadow-md transition-all",
      div {
        div { class: "flex flex-wrap items-center justify-between gap-2 mb-2.5",
          span { class: "px-2 py-0.5 rounded-md bg-indigo-500/15 text-indigo-700 dark:text-indigo-300 text-[11px] font-bold",
            "{item.family.short_badge(lang)}"
          }
          span { class: badge_class(BadgeVariant::Secondary, "text-[10px] font-mono"),
            "{item.lab_kind.badge_label(lang)}"
          }
        }

        h2 { class: "text-base sm:text-lg font-extrabold text-slate-900 dark:text-white group-hover:text-indigo-600 dark:group-hover:text-indigo-400 transition-colors",
          "{item.localized_title(lang)}"
        }

        p { class: "mt-2 text-xs text-slate-600 dark:text-slate-400 line-clamp-3 leading-relaxed",
          "{item.localized_summary(lang)}"
        }

        if !item.formula_short.is_empty() {
          div { class: "mt-3 rounded-lg bg-slate-50 dark:bg-slate-950/80 border border-slate-200/80 dark:border-slate-800 px-2.5 py-1.5 text-[11px] font-mono text-indigo-700 dark:text-indigo-300 truncate",
            "{item.formula_short}"
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
        span { class: "font-bold text-indigo-600 dark:text-indigo-400 group-hover:translate-x-0.5 transition-transform shrink-0",
          {lang.pick("启动动态实验 →", "Open Live Lab →")}
        }
      }
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// `/ai/components/:slug`：单算子交互工作台 + 公式推导 + 对照实现 + 关联模型详情页
// ─────────────────────────────────────────────────────────────────────────────

#[component]
pub fn AiComponentDetailPage(slug: String) -> Element {
  rsx! {
    section { class: "py-8 sm:py-10 min-h-screen bg-white dark:bg-slate-950",
      div { class: "max-w-7xl mx-auto px-4 sm:px-6",
        SuspenseBoundary {
          fallback: |_| rsx! {
            div { class: "flex items-center justify-center py-24",
              Spinner { size: SpinnerSize::Lg, class: "border-t-primary" }
            }
          },
          AiComponentDetailLoaded { slug }
        }
      }
    }
  }
}

#[component]
fn AiComponentDetailLoaded(slug: String) -> Element {
  let lang = current_lang();
  let res =
    use_server_future(use_reactive!(|slug| async move { get_ai_component_detail(slug).await }))?;

  let detail = match res() {
    Some(Ok(Some(d))) => d,
    Some(Ok(None)) => {
      return rsx! {
        widgets::NotFound {
          message: lang.pick("未找到该算子或组件词条", "Component entry not found").to_string(),
        }
      };
    }
    Some(Err(_)) => {
      return rsx! {
        widgets::LoadFailed {
          message: lang.pick("加载组件详情失败", "Failed to load component detail").to_string(),
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
      path: format!("/ai/components/{}", detail.meta.slug),
    }

    div { class: "space-y-6",
      // 面包屑与家族返回条
      div { class: "flex flex-wrap items-center justify-between gap-3 text-xs",
        div { class: "flex flex-wrap items-center gap-2 text-slate-500 dark:text-slate-400",
          Link { to: "/ai", class: "hover:text-indigo-600", "AI" }
          span { "/" }
          Link { to: "/ai/components", class: "hover:text-indigo-600 font-semibold",
            {lang.pick("组件与算子图鉴", "Components Atlas")}
          }
          span { "/" }
          span { class: "text-indigo-600 dark:text-indigo-400 font-bold",
            "{detail.meta.family.label(lang)}"
          }
        }
        Link {
          to: "/models",
          class: "font-semibold text-indigo-600 dark:text-indigo-400 hover:underline",
          {lang.pick("查看使用这些组件的旗舰模型图鉴 →", "Browse Flagship Models →")}
        }
      }

      // 词条头卡
      div { class: card_class("p-5 sm:p-6 bg-slate-50/70 dark:bg-slate-900/70"),
        div { class: "flex flex-wrap items-center gap-2 mb-2",
          span { class: "px-2.5 py-0.5 rounded-md bg-indigo-600 text-white text-xs font-bold",
            "{detail.meta.family.short_badge(lang)}"
          }
          if !detail.meta.complexity_badge.is_empty() {
            span { class: badge_class(BadgeVariant::Secondary, "font-mono text-xs"),
              "{detail.meta.complexity_badge}"
            }
          }
          for m in detail.meta.modalities.iter() {
            span { key: "{m}", class: "px-2 py-0.5 rounded bg-slate-200/70 dark:bg-slate-800 text-[11px] font-mono text-slate-600 dark:text-slate-300",
              "{m}"
            }
          }
        }
        h1 { class: "text-2xl sm:text-3xl font-extrabold tracking-tight text-slate-900 dark:text-white",
          "{title}"
        }
        p { class: "mt-2 text-sm sm:text-base text-slate-600 dark:text-slate-300 leading-relaxed max-w-4xl",
          "{desc}"
        }
        if !detail.siblings.is_empty() && detail.siblings.len() > 1 {
          div { class: "mt-4 pt-3 border-t border-slate-200 dark:border-slate-800 flex flex-wrap items-center gap-1.5 text-xs",
            span { class: "text-slate-400 mr-1",
              {lang.pick("同族算子切换：", "Siblings:")}
            }
            for sib in detail.siblings.iter() {
              Link {
                key: "{sib.slug}",
                to: format!("/ai/components/{}", sib.slug),
                class: if sib.slug == detail.meta.slug {
                  "px-2.5 py-1 rounded-md font-bold bg-indigo-600 text-white"
                } else {
                  "px-2.5 py-1 rounded-md bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-300 hover:border-indigo-400"
                },
                "{sib.localized_title(lang)}"
              }
            }
          }
        }
      }

      // 1. 置顶动态 WASM 算子交互实验台
      ComponentLabWorkbench {
        key: "{detail.meta.slug}",
        kind: detail.meta.lab_kind,
        initial_mode: detail.meta.lab_mode.clone(),
        embedded: false,
      }

      // 2. 深度数学原理、结构拆解与 PyTorch/Rust 对照代码（带右侧粘性目录）
      div { class: "grid grid-cols-1 lg:grid-cols-12 gap-8 items-start pt-2",
        article { class: "lg:col-span-9 min-w-0 prose prose-slate dark:prose-invert max-w-none",
          Markdown {
            content: detail.body_md.clone(),
            blog_id: format!("ai-component-{}", detail.meta.slug),
          }
        }
        aside { class: "hidden lg:block lg:col-span-3 sticky top-20",
          OnThisPage {
            entries: toc,
            title: lang.pick("本页目录", "On this page").to_string(),
          }
        }
      }

      // 3. 反向关联：采用该算子的已收录旗舰大模型
      if !detail.related_models.is_empty() {
        div { class: "pt-6 border-t border-slate-200 dark:border-slate-800",
          h2 { class: "text-lg font-extrabold text-slate-900 dark:text-white mb-3",
            {lang.pick("采用该组件/原语的代表性旗舰模型（点击查看整机拓扑）", "Flagship Models Using This Primitive")}
          }
          div { class: "grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3.5",
            for m in detail.related_models.iter() {
              Link {
                key: "{m.slug}",
                to: format!("/models/{}", m.slug),
                class: "group rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/70 dark:bg-slate-900/60 p-3.5 hover:border-indigo-400 transition-all",
                div { class: "flex items-center justify-between gap-2",
                  span { class: "font-bold text-sm text-slate-900 dark:text-white group-hover:text-indigo-600 dark:group-hover:text-indigo-400",
                    "{m.name}"
                  }
                  span { class: "text-[10px] font-mono px-2 py-0.5 rounded bg-indigo-500/15 text-indigo-700 dark:text-indigo-300",
                    "{m.family_badge}"
                  }
                }
                div { class: "mt-1.5 text-[11px] font-mono text-slate-500 dark:text-slate-400 truncate",
                  "Attn: {m.attention_short} · FFN: {m.ffn_short}"
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
  fn builtin_ai_components_parse_and_load_without_errors() {
    let loaded = load_all_components();
    assert!(
      loaded.len() >= 4,
      "expected at least 4 builtin flagship ai-components exemplars, got {}",
      loaded.len()
    );
    for item in loaded.iter() {
      assert!(!item.meta.slug.is_empty());
      assert!(!item.meta.title.is_empty());
      assert!(!item.body_md.is_empty());
    }
  }
}
