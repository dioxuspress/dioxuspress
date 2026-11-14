//! 指定模型架构图鉴（`/models` 索引页 + `/models/:slug` 详情页 + 管理员在线收录弹窗 + 静态目录扫描器）。
//!
//! 数据源合并优先级：
//! 1. PostgreSQL 在线收录表 `curated_models`（管理员通过 `/models` 或 `/ai/visualizer` 直发保存，支持覆盖同名静态条目）；
//! 2. 外部挂载内容库 `<CONTENT_DIR>/models/<slug>/`（如 `rustineverything-content/models/`，支持 `SIGHUP` 秒级热重载）；
//! 3. 框架内置示例库 `assets/models/<slug>/`。

#[cfg(feature = "server")]
use std::path::{Path, PathBuf};
#[cfg(feature = "server")]
use std::sync::Arc;

use app_core::i18n::Language;
use dioxus::prelude::*;
#[cfg(not(feature = "accounts"))]
use dioxus_shadcn::ButtonVariant;
use dioxus_shadcn::{
  badge_class, button_class, card_class, BadgeVariant, Button, ButtonSize, Input, Spinner,
  SpinnerSize, UiDensity,
};
#[cfg(feature = "accounts")]
use dioxus_shadcn::{ButtonVariant, Textarea};
use serde::{Deserialize, Serialize};
use widgets::{Markdown, PageMeta};

use crate::arch::{format_bytes_iec, format_count, ModelBlueprint};
#[cfg(feature = "accounts")]
use crate::atlas_store::{
  admin_ai_polish_model_draft, admin_delete_curated_model, admin_fetch_model_draft,
  admin_save_curated_model, CuratedModelUpsertInput,
};
use crate::atlas_store::{ModelModality, RecordOrigin};
#[cfg(feature = "accounts")]
use crate::server::ModelHubSource;
#[cfg(feature = "accounts")]
use crate::visualizer::clean_server_err;
use crate::visualizer::ModelVisualizerWorkbench;

#[cfg(feature = "server")]
use crate::arch::{build_model_blueprint, load_builtin_preset};
#[cfg(feature = "server")]
use crate::safetensors_meta::{parse_safetensors_index_json, summarize_index_and_headers};

fn current_lang() -> Language {
  try_consume_context::<Signal<Language>>().map(|s| s()).unwrap_or_default()
}

/// 读取全局会话上下文判断当前用户是否为管理员。
/// SSR 与水合首帧均为 `None -> false`，水合后由顶层 `App` 会话 `use_effect` 无缝点亮管理入口。
pub(crate) fn current_user_is_admin() -> bool {
  #[cfg(feature = "accounts")]
  {
    try_consume_context::<Signal<Option<app_core::session::SessionUser>>>()
      .and_then(|s| s())
      .is_some_and(|u| u.is_admin())
  }
  #[cfg(not(feature = "accounts"))]
  {
    false
  }
}

/// 图鉴列表卡片与侧边快速切换所需的轻量级模型摘要。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CuratedModelSummary {
  pub slug: String,
  pub name: String,
  pub vendor: String,
  #[serde(default)]
  pub modality: ModelModality,
  #[serde(default)]
  pub origin: RecordOrigin,
  /// 系列归属：`"deepseek"` | `"qwen"` | `"llama"` | `"other"`
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

  // 以下字段由 `config.json` / `ModelBlueprint` 实时精算得出，杜绝人工维护偏差
  pub family_badge: String,
  pub total_params: u64,
  pub active_params: u64,
  pub num_layers: u64,
  pub hidden_size: u64,
  pub max_context: u64,
  pub kv_compression_ratio: f64,
  pub attention_short: String,
  pub ffn_short: String,
  pub has_moe: bool,
  pub has_vision: bool,
  pub has_mla: bool,
  pub has_hybrid_attn: bool,
  pub fp8_vram_bytes: f64,
}

impl CuratedModelSummary {
  pub fn localized_summary(&self, lang: Language) -> &str {
    match lang {
      Language::En if !self.summary_en.trim().is_empty() => &self.summary_en,
      _ => &self.summary,
    }
  }

  pub fn localized_highlights(&self, lang: Language) -> &[String] {
    match lang {
      Language::En if !self.highlights_en.is_empty() => &self.highlights_en,
      _ => &self.highlights,
    }
  }
}

/// 单个指定模型的完整详情数据包（含 `ModelBlueprint` 与可选深度解读 Markdown）。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CuratedModelDetail {
  pub meta: CuratedModelSummary,
  pub blueprint: ModelBlueprint,
  pub readme_md: Option<String>,
}

#[cfg(feature = "accounts")]
fn detail_to_upsert_input(detail: &CuratedModelDetail) -> CuratedModelUpsertInput {
  CuratedModelUpsertInput {
    slug: detail.meta.slug.clone(),
    name: detail.meta.name.clone(),
    vendor: detail.meta.vendor.clone(),
    modality: detail.meta.modality,
    series: detail.meta.series.clone(),
    release_date: detail.meta.release_date.clone(),
    hf_repo: detail.meta.hf_repo.clone(),
    ms_repo: detail.meta.ms_repo.clone(),
    featured: detail.meta.featured,
    order: detail.meta.order,
    summary: detail.meta.summary.clone(),
    summary_en: detail.meta.summary_en.clone(),
    tags: detail.meta.tags.clone(),
    highlights: detail.meta.highlights.clone(),
    highlights_en: detail.meta.highlights_en.clone(),
    readme_md: detail.readme_md.clone(),
    blueprint: detail.blueprint.clone(),
  }
}

/// `models/<slug>/model.yaml` 磁盘反序列化结构（服务端与 CLI 校验共用）。
#[cfg(feature = "server")]
#[derive(Debug, Clone, Deserialize, Default)]
pub struct RawModelYaml {
  #[serde(default)]
  pub slug: String,
  #[serde(default)]
  pub name: String,
  #[serde(default)]
  pub vendor: String,
  #[serde(default)]
  pub modality: String,
  #[serde(default)]
  pub series: String,
  #[serde(default)]
  pub release_date: String,
  #[serde(default)]
  pub hf_repo: String,
  #[serde(default)]
  pub ms_repo: Option<String>,
  #[serde(default)]
  pub featured: bool,
  #[serde(default = "default_order")]
  pub order: i32,
  #[serde(default)]
  pub summary: String,
  #[serde(default)]
  pub summary_en: String,
  #[serde(default)]
  pub tags: Vec<String>,
  #[serde(default)]
  pub highlights: Vec<String>,
  #[serde(default)]
  pub highlights_en: Vec<String>,
}

#[cfg(feature = "server")]
fn default_order() -> i32 {
  100
}

// ─────────────────────────────────────────────────────────────────────────────
// 服务端目录扫描与 mtime 缓存（支持 SIGHUP 内容热重载 + 数据库合并）
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "server")]
static MODELS_CACHE: app_core::utils::DirListingCache<Vec<CuratedModelDetail>> =
  app_core::utils::DirListingCache::new();

/// 显式清空模型图鉴内存缓存（同时失效静态目录缓存与数据库合并缓存，供 `hot_reload` 调用）。
#[cfg(feature = "server")]
pub fn invalidate_models_cache() {
  MODELS_CACHE.invalidate();
  crate::atlas_store::invalidate_db_models_cache();
}

/// 解析当前生效的 `models/` 根目录：优先外部挂载 `<CONTENT_DIR>/models`（若存在且非空），否则回退 `assets/models`。
#[cfg(feature = "server")]
pub fn resolve_models_root() -> PathBuf {
  let primary = app_core::utils::content_dir("models");
  let has_entries =
    primary.is_dir() && std::fs::read_dir(&primary).ok().and_then(|mut it| it.next()).is_some();
  if has_entries {
    primary
  } else {
    app_core::utils::get_asset_root().join("models")
  }
}

/// 从指定 `models_root` 目录解析单个模型子目录（供运行时与 CLI 校验复用）。
#[cfg(feature = "server")]
pub fn parse_model_dir(dir: &Path, slug_fallback: &str) -> Result<CuratedModelDetail, String> {
  let yaml_path = if dir.join("model.yaml").is_file() {
    dir.join("model.yaml")
  } else if dir.join("model.yml").is_file() {
    dir.join("model.yml")
  } else {
    return Err(format!("{} 缺少 model.yaml", dir.display()));
  };

  let cfg_path = dir.join("config.json");
  if !cfg_path.is_file() {
    return Err(format!("{} 缺少 config.json", dir.display()));
  }

  let yaml_raw = std::fs::read_to_string(&yaml_path)
    .map_err(|e| format!("读取 {} 失败: {e}", yaml_path.display()))?;
  let raw_meta: RawModelYaml = serde_yaml::from_str(&yaml_raw)
    .map_err(|e| format!("解析 {} YAML 失败: {e}", yaml_path.display()))?;

  let slug = if raw_meta.slug.trim().is_empty() {
    slug_fallback.to_string()
  } else {
    raw_meta.slug.trim().to_string()
  };

  let cfg_raw = std::fs::read_to_string(&cfg_path)
    .map_err(|e| format!("读取 {} 失败: {e}", cfg_path.display()))?;
  let gen_raw = std::fs::read_to_string(dir.join("generation_config.json")).ok();
  let tok_raw = std::fs::read_to_string(dir.join("tokenizer_config.json")).ok();

  // 若目录下附带 safetensors.index.json 则优先解析，否则尝试复用内置快照的 L2 张量表
  let idx_raw = std::fs::read_to_string(dir.join("model.safetensors.index.json"))
    .or_else(|_| std::fs::read_to_string(dir.join("safetensors.index.json")))
    .ok();
  let safetensors = if let Some(ref idx_s) = idx_raw {
    let (tot, wm) = parse_safetensors_index_json(idx_s)?;
    Some(summarize_index_and_headers(tot, &wm, &[]))
  } else {
    load_builtin_preset(&slug)
      .or_else(|| load_builtin_preset(&raw_meta.hf_repo))
      .and_then(|bp| bp.safetensors)
  };

  let repo_id = if raw_meta.hf_repo.trim().is_empty() {
    slug.clone()
  } else {
    raw_meta.hf_repo.trim().to_string()
  };

  let blueprint = build_model_blueprint(
    &repo_id,
    "Curated Model Atlas (Official Config)",
    &cfg_raw,
    gen_raw.as_deref(),
    tok_raw.as_deref(),
    safetensors,
  )?;

  let readme_md = ["README.md", "readme.md", "index.md"]
    .iter()
    .map(|name| dir.join(name))
    .find(|p| p.is_file())
    .and_then(|p| std::fs::read_to_string(p).ok())
    .filter(|s| !s.trim().is_empty());

  let name = if raw_meta.name.trim().is_empty() { slug.clone() } else { raw_meta.name };
  let vendor = if raw_meta.vendor.trim().is_empty() {
    crate::atlas_store::infer_vendor_from_repo(&repo_id)
  } else {
    raw_meta.vendor
  };
  let explicit_modality = ModelModality::parse_str(&raw_meta.modality);

  let meta = crate::atlas_store::build_curated_summary(
    slug,
    name,
    vendor,
    explicit_modality,
    raw_meta.series,
    raw_meta.release_date,
    repo_id,
    raw_meta.ms_repo,
    raw_meta.featured,
    raw_meta.order,
    raw_meta.summary,
    raw_meta.summary_en,
    raw_meta.tags,
    raw_meta.highlights,
    raw_meta.highlights_en,
    &blueprint,
    RecordOrigin::Builtin,
  );

  Ok(CuratedModelDetail { meta, blueprint, readme_md })
}

/// 加载全部静态目录内置模型详情（带目录指纹缓存）。
#[cfg(feature = "server")]
pub fn load_curated_models_full() -> Arc<Vec<CuratedModelDetail>> {
  let root = resolve_models_root();
  let fp = app_core::utils::fingerprint_for_dir(&root, |p| {
    p.file_name().and_then(|n| n.to_str()).is_some_and(|n| {
      matches!(
        n,
        "model.yaml"
          | "model.yml"
          | "config.json"
          | "README.md"
          | "readme.md"
          | "model.safetensors.index.json"
          | "safetensors.index.json"
      )
    })
  });

  MODELS_CACHE.get_or_rebuild(fp, || {
    let mut items = Vec::new();
    let Ok(entries) = std::fs::read_dir(&root) else {
      return items;
    };
    for entry in entries.flatten() {
      let path = entry.path();
      if !path.is_dir() {
        continue;
      }
      let Some(dir_name) = path.file_name().and_then(|n| n.to_str()) else {
        continue;
      };
      if dir_name.starts_with('.') || dir_name.starts_with('_') {
        continue;
      }
      match parse_model_dir(&path, dir_name) {
        Ok(detail) => items.push(detail),
        Err(err) => {
          tracing::warn!(dir = %path.display(), error = %err, "skip invalid curated model entry");
        }
      }
    }
    items
      .sort_by(|a, b| a.meta.order.cmp(&b.meta.order).then_with(|| a.meta.slug.cmp(&b.meta.slug)));
    items
  })
}

/// 同步枚举磁盘内置模型摘要（用于离线校验与单元测试）。
#[cfg(feature = "server")]
pub fn scan_curated_models() -> Vec<CuratedModelSummary> {
  load_curated_models_full().iter().map(|d| d.meta.clone()).collect()
}

/// 获取合并后的全部指定模型卡片列表（静态 `models/` + PostgreSQL 在线收录）。
#[get("/api/ai/models")]
pub async fn list_curated_models() -> Result<Vec<CuratedModelSummary>, ServerFnError> {
  Ok(crate::atlas_store::list_curated_models_merged().await)
}

/// 按 `slug` 获取单个指定模型的完整架构蓝图与解读详情（数据库优先覆盖同名静态条目）。
#[get("/api/ai/models/:slug")]
pub async fn get_curated_model_detail(
  slug: String,
) -> Result<Option<CuratedModelDetail>, ServerFnError> {
  let clean = slug.trim();
  if clean.is_empty() || clean.contains("..") || clean.contains('/') || clean.contains('\\') {
    return Ok(None);
  }
  let all = crate::atlas_store::load_merged_curated_models_full().await;
  Ok(all.into_iter().find(|d| d.meta.slug == clean))
}

// ─────────────────────────────────────────────────────────────────────────────
// `/models`：指定模型架构图鉴总览页
// ─────────────────────────────────────────────────────────────────────────────

#[component]
pub fn ModelsAtlasPage() -> Element {
  let lang = current_lang();
  let title = lang.pick("主流模型架构图鉴", "Flagship Model Architecture Atlas").to_string();
  let desc = lang
    .pick(
      "涵盖 LLM、VLM 视觉多模态、Audio 语音（ASR/TTS）、Diffusion 图像视频生成与 Embedding 向量模型的官方架构精算与多层级可折叠框图。",
      "Interactive, multi-scale collapsible architecture diagrams across LLM, VLM, Audio, Diffusion, and Embedding models derived from official configs.",
    )
    .to_string();

  rsx! {
    PageMeta {
      title: title.clone(),
      description: desc.clone(),
      path: "/models".to_string(),
    }

    section { class: "py-10 sm:py-12 min-h-screen bg-white dark:bg-slate-950",
      div { class: "max-w-7xl mx-auto px-4 sm:px-6",
        SuspenseBoundary {
          fallback: |_| rsx! {
            div { class: "flex items-center justify-center py-24",
              Spinner { size: SpinnerSize::Lg, class: "border-t-primary" }
            }
          },
          ModelsAtlasLoaded { title, desc }
        }
      }
    }
  }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ArchTraitFilter {
  All,
  Mla,
  Moe,
  Dense,
  Multimodal,
  HybridAttn,
}

#[component]
fn ModelsAtlasLoaded(title: String, desc: String) -> Element {
  let lang = current_lang();
  let is_admin = current_user_is_admin();
  let models_res = use_server_future(list_curated_models)?;
  #[cfg(feature = "accounts")]
  let mut live_models_override: Signal<Option<Vec<CuratedModelSummary>>> = use_signal(|| None);
  #[cfg(not(feature = "accounts"))]
  let live_models_override: Signal<Option<Vec<CuratedModelSummary>>> = use_signal(|| None);

  let models = if let Some(list) = live_models_override() {
    list
  } else {
    match models_res() {
      Some(Ok(list)) => list,
      Some(Err(_)) => {
        return rsx! {
          widgets::LoadFailed {
            message: lang.pick("模型图鉴加载失败，请稍后重试", "Failed to load Model Atlas").to_string(),
          }
        };
      }
      None => Vec::new(),
    }
  };

  let mut modality_filter: Signal<Option<ModelModality>> = use_signal(|| None);
  let mut series_filter = use_signal(|| "all".to_string());
  let mut trait_filter = use_signal(|| ArchTraitFilter::All);
  let mut search_q = use_signal(String::new);
  let mut custom_repo = use_signal(String::new);
  #[cfg(feature = "accounts")]
  let mut show_add_modal = use_signal(|| false);
  #[cfg(feature = "accounts")]
  let mut saved_slug_banner: Signal<Option<String>> = use_signal(|| None);
  let nav = use_navigator();

  #[cfg(feature = "accounts")]
  let admin_quick_add_el = if is_admin {
    rsx! {
      div { class: "pt-2 border-t border-slate-100 dark:border-slate-800 flex items-center justify-between gap-2",
        span { class: "text-[11px] text-slate-500 dark:text-slate-400",
          {lang.pick("管理员直发：按仓库拉取并永久入库", "Admin: Fetch & save to Atlas DB")}
        }
        Button {
          variant: ButtonVariant::Outline,
          size: ButtonSize::Sm,
          onclick: move |_| show_add_modal.set(true),
          {lang.pick("＋ 收录模型", "+ Add Model")}
        }
      }
    }
  } else {
    rsx! {}
  };
  #[cfg(not(feature = "accounts"))]
  let admin_quick_add_el = rsx! {};

  #[cfg(feature = "accounts")]
  let admin_saved_banner_el = if let Some(saved_slug) = saved_slug_banner() {
    rsx! {
      div { class: "mt-4 rounded-xl border border-emerald-200 dark:border-emerald-900/70 bg-emerald-50 dark:bg-emerald-950/40 px-4 py-2.5 text-xs text-emerald-800 dark:text-emerald-300 flex flex-wrap items-center justify-between gap-2",
        span {
          {lang.pick("已成功收录模型至图鉴：", "Saved to Model Atlas: ")}
          span { class: "font-mono font-bold", "/models/{saved_slug}" }
        }
        Link {
          to: format!("/models/{saved_slug}"),
          class: "font-bold underline hover:opacity-80",
          {lang.pick("打开模型架构详情页 →", "View Model Page →")}
        }
      }
    }
  } else {
    rsx! {}
  };
  #[cfg(not(feature = "accounts"))]
  let admin_saved_banner_el = rsx! {};

  #[cfg(feature = "accounts")]
  let admin_add_modal_el = if show_add_modal() {
    rsx! {
      SaveToAtlasModal {
        initial_draft: None,
        initial_repo_hint: custom_repo().trim().to_string(),
        on_close: move |_| show_add_modal.set(false),
        on_saved: move |slug: String| {
          show_add_modal.set(false);
          saved_slug_banner.set(Some(slug));
          spawn(async move {
            if let Ok(fresh) = list_curated_models().await {
              live_models_override.set(Some(fresh));
            }
          });
        },
      }
    }
  } else {
    rsx! {}
  };
  #[cfg(not(feature = "accounts"))]
  let admin_add_modal_el = rsx! {};

  let count_modality = |m_opt: Option<ModelModality>| -> usize {
    match m_opt {
      None => models.len(),
      Some(target) => models.iter().filter(|m| m.modality == target).count(),
    }
  };

  let cur_modality = modality_filter();
  let count_series = |s: &str| -> usize {
    models
      .iter()
      .filter(|m| cur_modality.is_none_or(|md| m.modality == md))
      .filter(|m| s == "all" || m.series == s)
      .count()
  };

  let q_norm = search_q().trim().to_ascii_lowercase();
  let cur_series = series_filter();
  let cur_trait = trait_filter();

  let filtered: Vec<CuratedModelSummary> = models
    .iter()
    .filter(|m| cur_modality.is_none_or(|md| m.modality == md))
    .filter(|m| cur_series == "all" || m.series == cur_series)
    .filter(|m| match cur_trait {
      ArchTraitFilter::All => true,
      ArchTraitFilter::Mla => m.has_mla,
      ArchTraitFilter::Moe => m.has_moe,
      ArchTraitFilter::Dense => !m.has_moe,
      ArchTraitFilter::Multimodal => m.has_vision || m.modality != ModelModality::Llm,
      ArchTraitFilter::HybridAttn => m.has_hybrid_attn,
    })
    .filter(|m| {
      if q_norm.is_empty() {
        return true;
      }
      m.name.to_ascii_lowercase().contains(&q_norm)
        || m.hf_repo.to_ascii_lowercase().contains(&q_norm)
        || m.vendor.to_ascii_lowercase().contains(&q_norm)
        || m.family_badge.to_ascii_lowercase().contains(&q_norm)
        || m.modality.short_badge().to_ascii_lowercase().contains(&q_norm)
        || m.summary.to_ascii_lowercase().contains(&q_norm)
        || m.summary_en.to_ascii_lowercase().contains(&q_norm)
        || m.tags.iter().any(|t| t.to_ascii_lowercase().contains(&q_norm))
    })
    .cloned()
    .collect();

  let modality_tabs: [(Option<ModelModality>, &str); 6] = [
    (None, lang.pick("全部模态", "All Modalities")),
    (Some(ModelModality::Llm), ModelModality::Llm.label(lang)),
    (Some(ModelModality::Vlm), ModelModality::Vlm.label(lang)),
    (Some(ModelModality::Audio), ModelModality::Audio.label(lang)),
    (Some(ModelModality::Diffusion), ModelModality::Diffusion.label(lang)),
    (Some(ModelModality::Embedding), ModelModality::Embedding.label(lang)),
  ];

  let series_tabs = [
    ("all", lang.pick("全部家族", "All Families")),
    ("deepseek", lang.pick("DeepSeek 系列", "DeepSeek")),
    ("qwen", lang.pick("Qwen 通义系列", "Qwen")),
    ("llama", lang.pick("Llama 系列", "Llama")),
    ("other", lang.pick("主流开源 (Gemma/GLM/FLUX...)", "Open Frontier")),
  ];

  let trait_chips = [
    (ArchTraitFilter::All, lang.pick("全部架构特性", "All Traits")),
    (ArchTraitFilter::Mla, "MLA 潜空间压缩"),
    (ArchTraitFilter::Moe, "Sparse MoE 混合专家"),
    (ArchTraitFilter::Dense, "Dense 稠密主干"),
    (ArchTraitFilter::Multimodal, "Multimodal 多模态"),
    (ArchTraitFilter::HybridAttn, "SWA / iRoPE / SSM 混合"),
  ];

  rsx! {
    div { class: "space-y-8",
      // 1. 顶部主视觉区 + 任意 HF/ModelScope 模型实时沙盒直达栏（管理员额外可见直发收录按钮）
      div { class: "rounded-3xl border border-slate-200 dark:border-slate-800 bg-linear-to-br from-slate-50 via-white to-indigo-50/60 dark:from-slate-900 dark:via-slate-950 dark:to-indigo-950/30 p-6 sm:p-8 shadow-xs",
        div { class: "flex flex-col lg:flex-row lg:items-end justify-between gap-6",
          div { class: "max-w-3xl",
            div { class: "inline-flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-indigo-600 dark:text-indigo-400 mb-2.5",
              span { class: "px-2.5 py-0.5 rounded-full bg-indigo-600 text-white text-[10px]",
                "Model Atlas"
              }
              span {
                {lang.pick(
                  "官方 Config 精算 · 多层级收起/展开拓扑图",
                  "Verified Configs · Multi-Scale Collapsible Graphs"
                )}
              }
            }
            h1 { class: "text-2xl sm:text-4xl font-extrabold tracking-tight text-slate-900 dark:text-white",
              "{title}"
            }
            p { class: "mt-2.5 text-sm sm:text-base text-slate-600 dark:text-slate-400 leading-relaxed",
              "{desc}"
            }
          }

          // 实时拉取任意仓库快捷入口 + 管理员在线收录入口
          div { class: "w-full lg:w-96 shrink-0 rounded-2xl border border-indigo-200/80 dark:border-indigo-900/60 bg-white/90 dark:bg-slate-900/90 p-3.5 space-y-2.5",
            div { class: "flex items-center justify-between text-xs font-semibold text-slate-700 dark:text-slate-200",
              span { {lang.pick("想看图鉴之外的任意仓库？", "Inspect any custom repo?")} }
              Link {
                to: "/ai/visualizer",
                class: "text-indigo-600 dark:text-indigo-400 hover:underline",
                {lang.pick("打开实时沙盒 →", "Live Sandbox →")}
              }
            }
            div {
              class: "flex items-center gap-1.5",
              onkeydown: move |evt: KeyboardEvent| {
                if evt.key() == Key::Enter {
                  let r = custom_repo().trim().to_string();
                  if !r.is_empty() {
                    let _ = nav.push(format!("/ai/visualizer?repo={r}"));
                  } else {
                    let _ = nav.push("/ai/visualizer");
                  }
                }
              },
              Input {
                class: "h-9 text-xs font-mono bg-slate-50 dark:bg-slate-950",
                placeholder: "输入 owner/model 回车解析...",
                "aria-label": "Custom HuggingFace or ModelScope repository",
                value: custom_repo(),
                on_value_change: move |v: String| custom_repo.set(v),
              }
              Button {
                size: ButtonSize::Sm,
                onclick: move |_| {
                  let r = custom_repo().trim().to_string();
                  if !r.is_empty() {
                    let _ = nav.push(format!("/ai/visualizer?repo={r}"));
                  } else {
                    let _ = nav.push("/ai/visualizer");
                  }
                },
                {lang.pick("解析", "Go")}
              }
            }
            {admin_quick_add_el}
          }
        }

        {admin_saved_banner_el}
      }

      {admin_add_modal_el}

      // 2. 多维筛选控制台：五大模态 Tab + 家族胶囊 + 架构特性胶囊 + 搜索框
      div { class: "space-y-3",
        // 2A. 模态一级导航 Tab
        div { class: "flex flex-wrap items-center gap-1.5 border-b border-slate-200 dark:border-slate-800 pb-3",
          for (m_opt, m_lbl) in modality_tabs {
            {
              let cnt = count_modality(m_opt);
              let active = modality_filter() == m_opt;
              let cls = if active {
                "px-3.5 py-1.5 rounded-xl text-xs font-bold bg-indigo-600 text-white shadow-xs cursor-pointer transition-all"
              } else {
                "px-3.5 py-1.5 rounded-xl text-xs font-semibold bg-slate-100 dark:bg-slate-900 text-slate-600 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-800 cursor-pointer transition-all"
              };
              let key_s = m_opt.map(|m| m.as_str()).unwrap_or("all");
              rsx! {
                button {
                  key: "mod-{key_s}",
                  r#type: "button",
                  class: cls,
                  onclick: move |_| modality_filter.set(m_opt),
                  span { "{m_lbl}" }
                  span { class: if active { "ml-1.5 opacity-85 font-mono" } else { "ml-1.5 text-slate-400 font-mono" },
                    "({cnt})"
                  }
                }
              }
            }
          }
        }

        // 2B. 系列页签 + 搜索框
        div { class: "flex flex-col md:flex-row md:items-center justify-between gap-3",
          div { class: "flex flex-wrap gap-1.5",
            for (s_id, s_lbl) in series_tabs {
              {
                let cnt = count_series(s_id);
                let active = series_filter() == s_id;
                let cls = if active {
                  "px-3 py-1.5 rounded-xl text-xs font-bold bg-slate-900 text-white dark:bg-white dark:text-slate-900 shadow-xs cursor-pointer transition-all"
                } else {
                  "px-3 py-1.5 rounded-xl text-xs font-semibold bg-slate-100 dark:bg-slate-900 text-slate-600 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-800 cursor-pointer transition-all"
                };
                rsx! {
                  button {
                    key: "{s_id}",
                    r#type: "button",
                    class: cls,
                    onclick: move |_| series_filter.set(s_id.to_string()),
                    span { "{s_lbl}" }
                    span { class: if active { "ml-1.5 opacity-80 font-mono" } else { "ml-1.5 text-slate-400 font-mono" },
                      "({cnt})"
                    }
                  }
                }
              }
            }
          }

          div { class: "w-full md:w-64",
            Input {
              r#type: "search",
              class: "h-9 text-xs",
              placeholder: lang.pick("搜索模型名、模态、架构、标签...", "Filter models, modality, tags..."),
              "aria-label": "Filter curated models",
              value: search_q(),
              on_value_change: move |v: String| search_q.set(v),
            }
          }
        }

        // 2C. 架构机制快速过滤条
        div { class: "flex flex-wrap items-center gap-1.5",
          for (tf, tf_lbl) in trait_chips {
            {
              let active = trait_filter() == tf;
              let cls = if active {
                "px-2.5 py-1 rounded-lg text-[11px] font-semibold bg-indigo-600 text-white cursor-pointer"
              } else {
                "px-2.5 py-1 rounded-lg text-[11px] font-medium border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 text-slate-600 dark:text-slate-400 hover:border-indigo-400 cursor-pointer"
              };
              rsx! {
                button {
                  key: "{tf_lbl}",
                  r#type: "button",
                  class: cls,
                  onclick: move |_| trait_filter.set(tf),
                  "{tf_lbl}"
                }
              }
            }
          }
        }
      }

      // 3. 模型规格对比卡片网格
      if filtered.is_empty() {
        div { class: card_class("p-12 text-center text-sm text-slate-500 dark:text-slate-400"),
          {lang.pick("没有匹配当前筛选条件的模型，试试切换「全部模态 / 全部家族」或清空搜索词。", "No models match the current filters.")}
        }
      } else {
        div { class: "grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-5",
          for m in filtered {
            CuratedModelCard { key: "{m.slug}", model: m, show_origin: is_admin }
          }
        }
      }
    }
  }
}

fn series_accent_badge(series: &str) -> (&'static str, &'static str) {
  match series {
    "deepseek" => (
      "DeepSeek",
      "bg-blue-500/10 text-blue-700 dark:text-blue-300 border-blue-200 dark:border-blue-800",
    ),
    "qwen" => (
      "Qwen",
      "bg-violet-500/10 text-violet-700 dark:text-violet-300 border-violet-200 dark:border-violet-800",
    ),
    "llama" => (
      "Meta Llama",
      "bg-sky-500/10 text-sky-700 dark:text-sky-300 border-sky-200 dark:border-sky-800",
    ),
    _ => (
      "Open Frontier",
      "bg-emerald-500/10 text-emerald-700 dark:text-emerald-300 border-emerald-200 dark:border-emerald-800",
    ),
  }
}

fn modality_badge_style(modality: ModelModality) -> &'static str {
  match modality {
    ModelModality::Llm => "bg-indigo-500/15 text-indigo-700 dark:text-indigo-300",
    ModelModality::Vlm => "bg-fuchsia-500/15 text-fuchsia-700 dark:text-fuchsia-300",
    ModelModality::Audio => "bg-cyan-500/15 text-cyan-700 dark:text-cyan-300",
    ModelModality::Diffusion => "bg-rose-500/15 text-rose-700 dark:text-rose-300",
    ModelModality::Embedding => "bg-teal-500/15 text-teal-700 dark:text-teal-300",
  }
}

#[component]
fn CuratedModelCard(model: CuratedModelSummary, show_origin: bool) -> Element {
  let lang = current_lang();
  let (_, series_cls) = series_accent_badge(&model.series);
  let mod_cls = modality_badge_style(model.modality);
  let is_sparse = model.active_params < model.total_params;
  let detail_href = format!("/models/{}", model.slug);

  rsx! {
    Link {
      to: detail_href,
      class: "group flex flex-col justify-between rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/70 p-5 hover:border-indigo-400 dark:hover:border-indigo-500 hover:shadow-md transition-all",
      div {
        // 厂商徽章 + 模态徽章 + 家族标签
        div { class: "flex flex-wrap items-center justify-between gap-2 mb-3",
          span { class: "px-2 py-0.5 rounded-md border text-[11px] font-bold {series_cls}",
            "{model.vendor}"
          }
          div { class: "flex items-center gap-1.5",
            if show_origin && model.origin != RecordOrigin::Builtin {
              span { class: "px-1.5 py-0.5 rounded bg-amber-500/15 text-amber-700 dark:text-amber-300 text-[10px] font-mono font-semibold",
                if model.origin == RecordOrigin::Override { "DB·Override" } else { "DB" }
              }
            }
            span { class: "px-2 py-0.5 rounded-md text-[10px] font-semibold {mod_cls}",
              "{model.modality.short_badge()}"
            }
            span { class: badge_class(BadgeVariant::Secondary, "text-[10px] font-mono"),
              "{model.family_badge}"
            }
          }
        }

        // 模型标题与 HF Repo
        h2 { class: "text-lg font-extrabold text-slate-900 dark:text-white group-hover:text-indigo-600 dark:group-hover:text-indigo-400 transition-colors",
          "{model.name}"
        }
        div { class: "mt-0.5 text-[11px] font-mono text-slate-400 dark:text-slate-500 truncate",
          "{model.hf_repo}"
          if !model.release_date.is_empty() {
            " · {model.release_date}"
          }
        }

        p { class: "mt-2.5 text-xs text-slate-600 dark:text-slate-400 line-clamp-2 leading-relaxed",
          "{model.localized_summary(lang)}"
        }

        // 4 宫格硬核指标矩阵
        div { class: "mt-3.5 grid grid-cols-2 gap-2 rounded-xl bg-slate-50 dark:bg-slate-950/70 border border-slate-200/70 dark:border-slate-800/80 p-2.5 font-mono",
          div {
            div { class: "text-[10px] text-slate-400",
              {lang.pick("总参数 / 激活", "Total / Active")}
            }
            div { class: "text-xs font-bold text-slate-900 dark:text-white tabular-nums",
              if is_sparse {
                "{format_count(model.total_params)} / "
                span { class: "text-amber-600 dark:text-amber-400",
                  "{format_count(model.active_params)}"
                }
              } else {
                "{format_count(model.total_params)} Dense"
              }
            }
          }
          div {
            div { class: "text-[10px] text-slate-400",
              {lang.pick("层栈 × 隐藏维", "Layers × Hidden")}
            }
            div { class: "text-xs font-bold text-slate-900 dark:text-white tabular-nums",
              "{model.num_layers}L × {model.hidden_size}d"
            }
          }
          div {
            div { class: "text-[10px] text-slate-400",
              {lang.pick("KV 压缩效率", "KV Compression")}
            }
            div { class: "text-xs font-bold text-indigo-600 dark:text-indigo-400 tabular-nums",
              "{model.kv_compression_ratio:.1}× vs MHA"
            }
          }
          div {
            div { class: "text-[10px] text-slate-400",
              {lang.pick("FP8 预估显存", "Est. FP8 VRAM")}
            }
            div { class: "text-xs font-bold text-slate-800 dark:text-slate-200 tabular-nums",
              "{format_bytes_iec(model.fp8_vram_bytes)}"
            }
          }
        }

        // 注意力与前馈简明签名
        div { class: "mt-3 space-y-1 text-[11px] font-mono",
          div { class: "truncate text-indigo-700 dark:text-indigo-300 bg-indigo-50/70 dark:bg-indigo-950/40 px-2 py-0.5 rounded",
            "Attn: {model.attention_short}"
          }
          div { class: "truncate text-amber-800 dark:text-amber-300 bg-amber-50/70 dark:bg-amber-950/40 px-2 py-0.5 rounded",
            "FFN:  {model.ffn_short}"
          }
        }
      }

      // 卡片底栏
      div { class: "mt-4 pt-3 border-t border-slate-100 dark:border-slate-800/80 flex items-center justify-between text-xs",
        div { class: "flex flex-wrap gap-1",
          for tag in model.tags.iter().take(3) {
            span { key: "{tag}", class: "px-1.5 py-0.5 rounded bg-slate-100 dark:bg-slate-800 text-[10px] font-mono text-slate-500 dark:text-slate-400",
              "#{tag}"
            }
          }
        }
        span { class: "font-bold text-indigo-600 dark:text-indigo-400 group-hover:translate-x-0.5 transition-transform shrink-0",
          {lang.pick("展开架构图 →", "Inspect Graph →")}
        }
      }
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// `/models/:slug`：指定模型可折叠架构详情页
// ─────────────────────────────────────────────────────────────────────────────

#[component]
pub fn ModelDetailPage(slug: String) -> Element {
  rsx! {
    section { class: "py-8 sm:py-10 min-h-screen bg-white dark:bg-slate-950",
      div { class: "max-w-7xl mx-auto px-4 sm:px-6",
        SuspenseBoundary {
          fallback: |_| rsx! {
            div { class: "flex items-center justify-center py-24",
              Spinner { size: SpinnerSize::Lg, class: "border-t-primary" }
            }
          },
          ModelDetailLoaded { slug }
        }
      }
    }
  }
}

#[component]
fn ModelDetailLoaded(slug: String) -> Element {
  let lang = current_lang();
  #[cfg(feature = "accounts")]
  let is_admin = current_user_is_admin();
  #[cfg(feature = "accounts")]
  let nav = use_navigator();

  let detail_res =
    use_server_future(use_reactive!(|slug| async move { get_curated_model_detail(slug).await }))?;
  let all_res = use_server_future(list_curated_models)?;

  #[cfg(feature = "accounts")]
  let mut live_detail_override: Signal<Option<(String, CuratedModelDetail)>> = use_signal(|| None);
  #[cfg(not(feature = "accounts"))]
  let live_detail_override: Signal<Option<(String, CuratedModelDetail)>> = use_signal(|| None);
  #[cfg(feature = "accounts")]
  let mut live_all_override: Signal<Option<Vec<CuratedModelSummary>>> = use_signal(|| None);
  #[cfg(not(feature = "accounts"))]
  let live_all_override: Signal<Option<Vec<CuratedModelSummary>>> = use_signal(|| None);
  #[cfg(feature = "accounts")]
  let mut edit_draft: Signal<Option<CuratedModelUpsertInput>> = use_signal(|| None);
  #[cfg(feature = "accounts")]
  let mut refreshing_upstream = use_signal(|| false);
  #[cfg(feature = "accounts")]
  let mut confirm_delete = use_signal(|| false);
  #[cfg(feature = "accounts")]
  let mut deleting = use_signal(|| false);
  #[cfg(feature = "accounts")]
  let mut admin_notice: Signal<Option<String>> = use_signal(|| None);

  let all_models =
    live_all_override().unwrap_or_else(|| all_res().and_then(Result::ok).unwrap_or_default());

  let detail = if let Some((ref ov_slug, ref ov_d)) = live_detail_override() {
    if ov_slug == &slug {
      ov_d.clone()
    } else {
      match detail_res() {
        Some(Ok(Some(d))) => d,
        Some(Err(_)) => {
          return rsx! {
            widgets::LoadFailed {
              message: lang.pick("加载模型架构详情失败", "Failed to load model architecture").to_string(),
            }
          };
        }
        _ => return render_model_not_found(lang),
      }
    }
  } else {
    match detail_res() {
      Some(Ok(Some(d))) => d,
      Some(Err(_)) => {
        return rsx! {
          widgets::LoadFailed {
            message: lang.pick("加载模型架构详情失败", "Failed to load model architecture").to_string(),
          }
        };
      }
      _ => return render_model_not_found(lang),
    }
  };

  let meta = &detail.meta;
  let highlights = meta.localized_highlights(lang);
  let hf_url = format!("https://huggingface.co/{}", meta.hf_repo);
  let ms_url = meta.ms_repo.as_ref().map(|r| format!("https://modelscope.cn/models/{r}"));
  let live_sandbox_href = format!("/ai/visualizer?repo={}", meta.hf_repo);

  #[cfg(feature = "accounts")]
  let admin_detail_toolbar = if is_admin {
    let detail_for_edit = detail.clone();
    let repo_for_refresh = meta.hf_repo.clone();
    let slug_for_delete = meta.slug.clone();
    let origin_for_delete = meta.origin;
    rsx! {
      div { class: "rounded-xl border border-amber-200/80 dark:border-amber-900/60 bg-amber-50/60 dark:bg-amber-950/25 px-3.5 py-2.5 flex flex-wrap items-center justify-between gap-2 text-xs",
        div { class: "flex flex-wrap items-center gap-2",
          span { class: "px-2 py-0.5 rounded bg-amber-500/20 text-amber-800 dark:text-amber-300 font-mono font-bold text-[11px]",
            match meta.origin {
              RecordOrigin::Builtin => lang.pick("数据源：静态目录 models/", "Origin: Static FS"),
              RecordOrigin::Database => lang.pick("数据源：PostgreSQL 在线收录", "Origin: PostgreSQL"),
              RecordOrigin::Override => lang.pick("数据源：PostgreSQL（已覆盖同名静态）", "Origin: DB Override"),
            }
          }
          if let Some(msg) = admin_notice() {
            span { class: "text-emerald-700 dark:text-emerald-300 font-medium", "{msg}" }
          }
        }
        div { class: "flex flex-wrap items-center gap-2",
          Button {
            variant: ButtonVariant::Outline,
            size: ButtonSize::Sm,
            onclick: move |_| {
              confirm_delete.set(false);
              edit_draft.set(Some(detail_to_upsert_input(&detail_for_edit)));
            },
            {lang.pick("编辑图鉴信息", "Edit Metadata")}
          }
          Button {
            variant: ButtonVariant::Outline,
            size: ButtonSize::Sm,
            disabled: refreshing_upstream(),
            onclick: move |_| {
              let r = repo_for_refresh.clone();
              refreshing_upstream.set(true);
              admin_notice.set(None);
              spawn(async move {
                match admin_fetch_model_draft(r, "auto".to_string(), String::new()).await {
                  Ok(fresh_draft) => edit_draft.set(Some(fresh_draft)),
                  Err(e) => admin_notice.set(Some(clean_server_err(&e.to_string()))),
                }
                refreshing_upstream.set(false);
              });
            },
            if refreshing_upstream() {
              {lang.pick("正在拉取上游配置...", "Refreshing...")}
            } else {
              {lang.pick("从上游刷新蓝图", "Refresh Blueprint")}
            }
          }
          if meta.origin != RecordOrigin::Builtin {
            if confirm_delete() {
              Button {
                variant: ButtonVariant::Destructive,
                size: ButtonSize::Sm,
                disabled: deleting(),
                onclick: move |_| {
                  let s = slug_for_delete.clone();
                  deleting.set(true);
                  spawn(async move {
                    match admin_delete_curated_model(s.clone()).await {
                      Ok(_) => {
                        confirm_delete.set(false);
                        if origin_for_delete == RecordOrigin::Override {
                          if let Ok(Some(restored)) = get_curated_model_detail(s.clone()).await {
                            live_detail_override.set(Some((s, restored)));
                          }
                          if let Ok(list) = list_curated_models().await {
                            live_all_override.set(Some(list));
                          }
                          admin_notice.set(Some("已恢复为静态目录默认版本".to_string()));
                        } else {
                          let _ = nav.push("/models");
                        }
                      }
                      Err(e) => admin_notice.set(Some(clean_server_err(&e.to_string()))),
                    }
                    deleting.set(false);
                  });
                },
                if origin_for_delete == RecordOrigin::Override {
                  {lang.pick("确认恢复静态默认", "Confirm Restore")}
                } else {
                  {lang.pick("确认永久删除", "Confirm Delete")}
                }
              }
              Button {
                variant: ButtonVariant::Outline,
                size: ButtonSize::Sm,
                onclick: move |_| confirm_delete.set(false),
                {lang.pick("取消", "Cancel")}
              }
            } else {
              Button {
                variant: ButtonVariant::Outline,
                size: ButtonSize::Sm,
                onclick: move |_| confirm_delete.set(true),
                if meta.origin == RecordOrigin::Override {
                  {lang.pick("恢复静态默认", "Restore Builtin")}
                } else {
                  {lang.pick("删除收录", "Delete")}
                }
              }
            }
          }
        }
      }
    }
  } else {
    rsx! {}
  };
  #[cfg(not(feature = "accounts"))]
  let admin_detail_toolbar = rsx! {};

  #[cfg(feature = "accounts")]
  let admin_edit_modal_el = if let Some(draft) = edit_draft() {
    let cur_slug = slug.clone();
    rsx! {
      SaveToAtlasModal {
        initial_draft: Some(draft),
        on_close: move |_| edit_draft.set(None),
        on_saved: move |saved_slug: String| {
          edit_draft.set(None);
          let c_slug = cur_slug.clone();
          spawn(async move {
            if let Ok(list) = list_curated_models().await {
              live_all_override.set(Some(list));
            }
            if saved_slug == c_slug {
              if let Ok(Some(fresh_d)) = get_curated_model_detail(saved_slug.clone()).await {
                live_detail_override.set(Some((saved_slug, fresh_d)));
              }
              admin_notice.set(Some("已保存并即时更新图鉴页面".to_string()));
            } else {
              let _ = nav.push(format!("/models/{saved_slug}"));
            }
          });
        },
      }
    }
  } else {
    rsx! {}
  };
  #[cfg(not(feature = "accounts"))]
  let admin_edit_modal_el = rsx! {};

  rsx! {
    PageMeta {
      title: format!("{} · {}", meta.name, lang.pick("模型架构图鉴", "Model Architecture")),
      description: meta.localized_summary(lang).to_string(),
      path: format!("/models/{}", meta.slug),
    }

    div { class: "space-y-6",
      // 1. 顶部面包屑 + 指定模型横向一键切换导航条
      div { class: "flex flex-col gap-3",
        div { class: "flex flex-wrap items-center justify-between gap-2 text-xs",
          div { class: "inline-flex items-center gap-2 font-semibold text-indigo-600 dark:text-indigo-400",
            Link { to: "/models", class: "hover:underline",
              {lang.pick("← 返回模型架构图鉴", "← Back to Model Atlas")}
            }
            span { class: "text-slate-300 dark:text-slate-700", "/" }
            span { class: "text-slate-600 dark:text-slate-300", "{meta.vendor}" }
            span { class: "text-slate-300 dark:text-slate-700", "/" }
            span { class: "font-mono text-slate-900 dark:text-white", "{meta.name}" }
          }

          div { class: "flex flex-wrap items-center gap-2",
            a {
              href: "{hf_url}",
              target: "_blank",
              rel: "noopener noreferrer",
              class: "px-2.5 py-1 rounded-lg border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-900 text-slate-700 dark:text-slate-300 hover:border-indigo-400 font-mono text-[11px]",
              "HuggingFace ↗"
            }
            if let Some(ms) = ms_url {
              a {
                href: "{ms}",
                target: "_blank",
                rel: "noopener noreferrer",
                class: "px-2.5 py-1 rounded-lg border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-900 text-slate-700 dark:text-slate-300 hover:border-indigo-400 font-mono text-[11px]",
                "ModelScope ↗"
              }
            }
            Link {
              to: live_sandbox_href,
              class: "px-2.5 py-1 rounded-lg bg-indigo-50 dark:bg-indigo-950/60 border border-indigo-200 dark:border-indigo-800 text-indigo-700 dark:text-indigo-300 hover:bg-indigo-100 text-[11px] font-semibold",
              {lang.pick("实时拉取远端 Safetensors ↗", "Fetch Live Safetensors ↗")}
            }
          }
        }

        {admin_detail_toolbar}

        // 快速对比切换条：同页面一键切换其它指定旗舰模型观察架构差异
        if all_models.len() > 1 {
          div { class: "flex items-center gap-1.5 overflow-x-auto pb-1",
            span { class: "text-[11px] font-semibold text-slate-400 shrink-0 mr-1",
              {lang.pick("快速切换模型：", "Switch Model:")}
            }
            for item in all_models.iter() {
              {
                let is_cur = item.slug == meta.slug;
                let cls = if is_cur {
                  "shrink-0 px-2.5 py-1 rounded-lg text-xs font-bold bg-indigo-600 text-white shadow-xs"
                } else {
                  "shrink-0 px-2.5 py-1 rounded-lg text-xs font-medium bg-slate-100 dark:bg-slate-900 text-slate-600 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-800 transition-colors"
                };
                rsx! {
                  Link {
                    key: "{item.slug}",
                    to: format!("/models/{}", item.slug),
                    class: cls,
                    "{item.name}"
                  }
                }
              }
            }
          }
        }
      }

      {admin_edit_modal_el}

      // 2. 模型核心架构导读与工程看点卡
      div { class: card_class("p-5 sm:p-6 bg-linear-to-r from-indigo-50/60 via-white to-slate-50/70 dark:from-indigo-950/30 dark:via-slate-900 dark:to-slate-900"),
        div { class: "flex flex-wrap items-baseline justify-between gap-2",
          div { class: "flex flex-wrap items-center gap-2",
            h1 { class: "text-xl sm:text-2xl font-extrabold text-slate-900 dark:text-white",
              "{meta.name}"
            }
            span { class: format!("px-2 py-0.5 rounded-md text-[11px] font-semibold {}", modality_badge_style(meta.modality)),
              "{meta.modality.label(lang)}"
            }
          }
          span { class: "text-xs font-mono text-slate-500 dark:text-slate-400",
            {lang.pick(
              "提示：左侧拓扑图支持「紧凑总览 / 标准观察 / 全展开+算子」及单个模块独立收起与展开",
              "Tip: Collapse or expand any layer segment, vision branch, or operator table on the left graph"
            )}
          }
        }
        p { class: "mt-2 text-sm text-slate-700 dark:text-slate-300 leading-relaxed",
          "{meta.localized_summary(lang)}"
        }

        if !highlights.is_empty() {
          div { class: "mt-4 grid grid-cols-1 md:grid-cols-3 gap-2.5",
            for (idx, point) in highlights.iter().enumerate() {
              div {
                key: "{idx}",
                class: "rounded-xl border border-slate-200/90 dark:border-slate-800 bg-white/90 dark:bg-slate-950/60 px-3.5 py-2.5 text-xs text-slate-700 dark:text-slate-300 leading-relaxed",
                span { class: "font-mono font-bold text-indigo-600 dark:text-indigo-400 mr-1.5",
                  "0{idx + 1}."
                }
                "{point}"
              }
            }
          }
        }
      }

      // 3. 完整交互式架构可视化工作台（带多层级收起/展开拓扑图 + 算子放大镜 + 显存精算器）
      ModelVisualizerWorkbench {
        key: "atlas-{meta.slug}-{detail.blueprint.source_label}",
        blueprint: detail.blueprint.clone(),
        initial_focus: None,
        embedded: false,
      }

      // 4. 可选深度架构解读文档（若 `models/<slug>/README.md` 或数据库 `readme_md` 存在）
      if let Some(md) = detail.readme_md {
        div { class: card_class("p-6 sm:p-8 bg-white dark:bg-slate-900/60"),
          h2 { class: "text-lg font-extrabold text-slate-900 dark:text-white mb-4 pb-2 border-b border-slate-200 dark:border-slate-800",
            {lang.pick("架构深度拆解笔记", "Architecture Deep-Dive Notes")}
          }
          Markdown {
            blog_id: format!("model:{}", meta.slug),
            content: md,
          }
        }
      }
    }
  }
}

fn render_model_not_found(lang: Language) -> Element {
  rsx! {
    div { class: "py-12 space-y-6 text-center",
      widgets::NotFound {
        message: lang.pick("未找到该指定模型图鉴条目", "Curated model not found").to_string(),
      }
      Link {
        to: "/models",
        class: button_class(ButtonVariant::Primary, ButtonSize::Md, UiDensity::Comfortable, "font-semibold"),
        {lang.pick("返回模型图鉴", "Back to Model Atlas")}
      }
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 管理员通用组件：`SaveToAtlasModal`（按 owner/model 拉取 + 自动草稿 + AI 润色 + 持久化）
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "accounts")]
#[component]
pub fn SaveToAtlasModal(
  initial_draft: Option<CuratedModelUpsertInput>,
  #[props(default)] initial_repo_hint: String,
  on_close: EventHandler<()>,
  on_saved: EventHandler<String>,
) -> Element {
  let lang = current_lang();

  let init_repo =
    initial_draft.as_ref().map(|d| d.hf_repo.clone()).unwrap_or_else(|| initial_repo_hint.clone());
  let mut repo_query = use_signal(|| init_repo);
  let mut hub_source = use_signal(|| ModelHubSource::Auto);
  let mut rev_query = use_signal(String::new);

  let mut draft_bp = use_signal(|| initial_draft.as_ref().map(|d| d.blueprint.clone()));
  let mut slug = use_signal(|| initial_draft.as_ref().map(|d| d.slug.clone()).unwrap_or_default());
  let mut name = use_signal(|| initial_draft.as_ref().map(|d| d.name.clone()).unwrap_or_default());
  let mut vendor =
    use_signal(|| initial_draft.as_ref().map(|d| d.vendor.clone()).unwrap_or_default());
  let mut modality =
    use_signal(|| initial_draft.as_ref().map(|d| d.modality).unwrap_or(ModelModality::Llm));
  let mut series = use_signal(|| {
    initial_draft.as_ref().map(|d| d.series.clone()).unwrap_or_else(|| "other".to_string())
  });
  let mut release_date =
    use_signal(|| initial_draft.as_ref().map(|d| d.release_date.clone()).unwrap_or_default());
  let mut hf_repo =
    use_signal(|| initial_draft.as_ref().map(|d| d.hf_repo.clone()).unwrap_or_default());
  let mut ms_repo =
    use_signal(|| initial_draft.as_ref().and_then(|d| d.ms_repo.clone()).unwrap_or_default());
  let mut order_str = use_signal(|| {
    initial_draft.as_ref().map(|d| d.order.to_string()).unwrap_or_else(|| "50".to_string())
  });
  let mut featured = use_signal(|| initial_draft.as_ref().is_some_and(|d| d.featured));
  let mut tags_str =
    use_signal(|| initial_draft.as_ref().map(|d| d.tags.join(", ")).unwrap_or_default());
  let mut summary =
    use_signal(|| initial_draft.as_ref().map(|d| d.summary.clone()).unwrap_or_default());
  let mut summary_en =
    use_signal(|| initial_draft.as_ref().map(|d| d.summary_en.clone()).unwrap_or_default());
  let mut highlights_str =
    use_signal(|| initial_draft.as_ref().map(|d| d.highlights.join("\n")).unwrap_or_default());
  let mut highlights_en_str =
    use_signal(|| initial_draft.as_ref().map(|d| d.highlights_en.join("\n")).unwrap_or_default());
  let mut readme_md =
    use_signal(|| initial_draft.as_ref().and_then(|d| d.readme_md.clone()).unwrap_or_default());

  let mut fetching = use_signal(|| false);
  let mut polishing = use_signal(|| false);
  let mut saving = use_signal(|| false);
  let mut err_msg: Signal<Option<String>> = use_signal(|| None);
  let mut info_msg: Signal<Option<String>> = use_signal(|| None);

  let mut do_fetch_draft = move || {
    let r = repo_query().trim().to_string();
    if r.is_empty() {
      err_msg.set(Some("请输入 HuggingFace 或 ModelScope 的 owner/model".to_string()));
      return;
    }
    let src = hub_source().as_str().to_string();
    let rev = rev_query().trim().to_string();
    fetching.set(true);
    err_msg.set(None);
    info_msg.set(None);
    spawn(async move {
      match admin_fetch_model_draft(r, src, rev).await {
        Ok(d) => {
          slug.set(d.slug);
          name.set(d.name);
          vendor.set(d.vendor);
          modality.set(d.modality);
          series.set(d.series);
          release_date.set(d.release_date);
          hf_repo.set(d.hf_repo);
          ms_repo.set(d.ms_repo.unwrap_or_default());
          order_str.set(d.order.to_string());
          featured.set(d.featured);
          tags_str.set(d.tags.join(", "));
          summary.set(d.summary);
          summary_en.set(d.summary_en);
          highlights_str.set(d.highlights.join("\n"));
          highlights_en_str.set(d.highlights_en.join("\n"));
          if let Some(md) = d.readme_md {
            readme_md.set(md);
          }
          draft_bp.set(Some(d.blueprint));
          info_msg.set(Some(
            "已从远端解析模型蓝图并生成双语摘要与亮点，核对后点击「保存并发布」即可上线。"
              .to_string(),
          ));
        }
        Err(e) => err_msg.set(Some(clean_server_err(&e.to_string()))),
      }
      fetching.set(false);
    });
  };

  let do_ai_polish = move |_| {
    let Some(bp) = draft_bp() else {
      err_msg.set(Some("请先拉取或加载模型架构蓝图".to_string()));
      return;
    };
    let req_draft = CuratedModelUpsertInput {
      slug: slug(),
      name: name(),
      vendor: vendor(),
      modality: modality(),
      series: series(),
      release_date: release_date(),
      hf_repo: hf_repo(),
      ms_repo: Some(ms_repo()).filter(|s| !s.trim().is_empty()),
      featured: featured(),
      order: order_str().trim().parse::<i32>().unwrap_or(50),
      summary: summary(),
      summary_en: summary_en(),
      tags: Vec::new(),
      highlights: Vec::new(),
      highlights_en: Vec::new(),
      readme_md: None,
      blueprint: bp,
    };
    polishing.set(true);
    err_msg.set(None);
    info_msg.set(None);
    spawn(async move {
      match admin_ai_polish_model_draft(req_draft).await {
        Ok(polished) => {
          summary.set(polished.summary);
          summary_en.set(polished.summary_en);
          highlights_str.set(polished.highlights.join("\n"));
          highlights_en_str.set(polished.highlights_en.join("\n"));
          info_msg.set(Some("AI 已根据蓝图精确参数完成中英双语摘要与看点润色。".to_string()));
        }
        Err(e) => err_msg.set(Some(clean_server_err(&e.to_string()))),
      }
      polishing.set(false);
    });
  };

  let do_save = move |_| {
    let Some(bp) = draft_bp() else {
      err_msg.set(Some("请先通过上方输入 owner/model 拉取模型蓝图后再保存".to_string()));
      return;
    };
    let parsed_order = order_str().trim().parse::<i32>().unwrap_or(50);
    let input = CuratedModelUpsertInput {
      slug: slug(),
      name: name(),
      vendor: vendor(),
      modality: modality(),
      series: series(),
      release_date: release_date(),
      hf_repo: hf_repo(),
      ms_repo: Some(ms_repo()).filter(|s| !s.trim().is_empty()),
      featured: featured(),
      order: parsed_order,
      summary: summary(),
      summary_en: summary_en(),
      tags: tags_str().split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect(),
      highlights: highlights_str()
        .lines()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect(),
      highlights_en: highlights_en_str()
        .lines()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect(),
      readme_md: Some(readme_md()).filter(|s| !s.trim().is_empty()),
      blueprint: bp,
    };

    saving.set(true);
    err_msg.set(None);
    spawn(async move {
      match admin_save_curated_model(input).await {
        Ok(saved_summary) => on_saved.call(saved_summary.slug),
        Err(e) => err_msg.set(Some(clean_server_err(&e.to_string()))),
      }
      saving.set(false);
    });
  };

  rsx! {
    div {
      class: "fixed inset-0 z-50 flex items-center justify-center bg-slate-950/60 backdrop-blur-xs p-3 sm:p-6",
      role: "dialog",
      "aria-modal": "true",
      onkeydown: move |evt: KeyboardEvent| {
        if evt.key() == Key::Escape {
          on_close.call(());
        }
      },
      div { class: "w-full max-w-4xl max-h-[92vh] overflow-y-auto rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-2xl p-5 sm:p-6 space-y-5",
        // 标题栏
        div { class: "flex items-start justify-between gap-4 pb-3 border-b border-slate-200 dark:border-slate-800",
          div {
            h2 { class: "text-lg font-extrabold text-slate-900 dark:text-white",
              {lang.pick("收录 / 编辑指定模型架构图鉴", "Save Model to Curated Atlas")}
            }
            p { class: "mt-0.5 text-xs text-slate-500 dark:text-slate-400",
              {lang.pick(
                "直接持久化至 PostgreSQL 并与静态 models/ 目录无缝合并，保存后全站即时生效。",
                "Persisted in PostgreSQL and merged live with static models/ entries."
              )}
            }
          }
          button {
            r#type: "button",
            class: "px-2.5 py-1 rounded-lg text-xs font-semibold text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800 cursor-pointer",
            onclick: move |_| on_close.call(()),
            "✕"
          }
        }

        // Step 1：按 owner/model 拉取或刷新蓝图栏
        div { class: "rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/80 dark:bg-slate-950/60 p-3.5 space-y-2.5",
          div { class: "text-xs font-bold text-slate-700 dark:text-slate-200",
            {lang.pick("1. 按 owner/model 拉取远端架构配置（自动推导参数、模态与双语看点）", "1. Fetch Blueprint by owner/model")}
          }
          div { class: "flex flex-col sm:flex-row items-stretch sm:items-center gap-2",
            div { class: "inline-flex rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-0.5 shrink-0",
              for (src_opt, label) in [
                (ModelHubSource::Auto, "Auto"),
                (ModelHubSource::HuggingFace, "HF"),
                (ModelHubSource::ModelScope, "ModelScope"),
              ] {
                {
                  let active = hub_source() == src_opt;
                  let cls = if active {
                    "px-2.5 py-1 rounded-md text-xs font-semibold bg-slate-900 text-white dark:bg-slate-100 dark:text-slate-900 cursor-pointer"
                  } else {
                    "px-2.5 py-1 rounded-md text-xs font-medium text-slate-600 dark:text-slate-300 cursor-pointer"
                  };
                  rsx! {
                    button {
                      key: "{label}",
                      r#type: "button",
                      class: cls,
                      onclick: move |_| hub_source.set(src_opt),
                      "{label}"
                    }
                  }
                }
              }
            }
            div {
              class: "flex-1",
              onkeydown: move |evt: KeyboardEvent| {
                if evt.key() == Key::Enter {
                  do_fetch_draft();
                }
              },
              Input {
                class: "h-9 text-xs font-mono bg-white dark:bg-slate-950",
                placeholder: "如 Qwen/Qwen2.5-VL-7B-Instruct 或 deepseek-ai/DeepSeek-R1",
                value: repo_query(),
                on_value_change: move |v: String| repo_query.set(v),
              }
            }
            div { class: "w-full sm:w-28 shrink-0",
              Input {
                class: "h-9 text-xs font-mono bg-white dark:bg-slate-950",
                placeholder: "分支(可选)",
                value: rev_query(),
                on_value_change: move |v: String| rev_query.set(v),
              }
            }
            Button {
              size: ButtonSize::Sm,
              disabled: fetching(),
              onclick: move |_| do_fetch_draft(),
              if fetching() {
                Spinner { size: SpinnerSize::Sm, class: "mr-1 border-t-current" }
                {lang.pick("解析中...", "Fetching...")}
              } else {
                {lang.pick("拉取并生成草稿", "Fetch Draft")}
              }
            }
          }

          if let Some(ref bp) = draft_bp() {
            div { class: "flex flex-wrap items-center justify-between gap-2 pt-1 text-[11px] font-mono text-indigo-700 dark:text-indigo-300",
              span {
                "✓ 已就绪蓝图：{bp.repo_id} · {bp.family.badge_label()} · {format_count(bp.params.total_params)} ({bp.num_hidden_layers}L × {bp.hidden_size}d)"
              }
              Button {
                variant: ButtonVariant::Outline,
                size: ButtonSize::Sm,
                disabled: polishing(),
                onclick: do_ai_polish,
                if polishing() {
                  Spinner { size: SpinnerSize::Sm, class: "mr-1 border-t-current" }
                  {lang.pick("AI 润色中...", "Polishing...")}
                } else {
                  {lang.pick("✨ AI 润色双语亮点", "✨ AI Polish")}
                }
              }
            }
          }
        }

        if let Some(err) = err_msg() {
          div { class: "rounded-xl border border-rose-200 dark:border-rose-900 bg-rose-50 dark:bg-rose-950/40 px-3.5 py-2.5 text-xs text-rose-700 dark:text-rose-300",
            "{err}"
          }
        }
        if let Some(info) = info_msg() {
          div { class: "rounded-xl border border-emerald-200 dark:border-emerald-900 bg-emerald-50 dark:bg-emerald-950/40 px-3.5 py-2.5 text-xs text-emerald-700 dark:text-emerald-300",
            "{info}"
          }
        }

        // Step 2：核心元数据表单
        div { class: "grid grid-cols-1 sm:grid-cols-3 gap-3",
          div {
            label { class: "block text-xs font-semibold text-slate-700 dark:text-slate-300 mb-1",
              "URL Slug（唯一主键）"
            }
            Input {
              class: "h-9 text-xs font-mono",
              placeholder: "qwen2-5-vl-7b",
              value: slug(),
              on_value_change: move |v: String| slug.set(v),
            }
          }
          div {
            label { class: "block text-xs font-semibold text-slate-700 dark:text-slate-300 mb-1",
              "展示名称 (Name)"
            }
            Input {
              class: "h-9 text-xs",
              placeholder: "Qwen2.5-VL-7B",
              value: name(),
              on_value_change: move |v: String| name.set(v),
            }
          }
          div {
            label { class: "block text-xs font-semibold text-slate-700 dark:text-slate-300 mb-1",
              "研发厂商 (Vendor)"
            }
            Input {
              class: "h-9 text-xs",
              placeholder: "Alibaba Qwen",
              value: vendor(),
              on_value_change: move |v: String| vendor.set(v),
            }
          }
        }

        // 模态选择 + 系列选择
        div { class: "grid grid-cols-1 md:grid-cols-2 gap-3",
          div {
            label { class: "block text-xs font-semibold text-slate-700 dark:text-slate-300 mb-1",
              "所属模态 (Modality)"
            }
            div { class: "flex flex-wrap gap-1.5",
              for m_opt in [
                ModelModality::Llm,
                ModelModality::Vlm,
                ModelModality::Audio,
                ModelModality::Diffusion,
                ModelModality::Embedding,
              ] {
                {
                  let active = modality() == m_opt;
                  let cls = if active {
                    "px-2.5 py-1 rounded-lg text-xs font-bold bg-indigo-600 text-white cursor-pointer"
                  } else {
                    "px-2.5 py-1 rounded-lg text-xs font-medium bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 cursor-pointer"
                  };
                  rsx! {
                    button {
                      key: "{m_opt.as_str()}",
                      r#type: "button",
                      class: cls,
                      onclick: move |_| modality.set(m_opt),
                      "{m_opt.short_badge()}"
                    }
                  }
                }
              }
            }
          }

          div {
            label { class: "block text-xs font-semibold text-slate-700 dark:text-slate-300 mb-1",
              "系列归属 (Series)"
            }
            div { class: "flex flex-wrap gap-1.5",
              for (s_id, s_lbl) in [
                ("deepseek", "DeepSeek"),
                ("qwen", "Qwen"),
                ("llama", "Llama"),
                ("other", "Other / Open"),
              ] {
                {
                  let active = series() == s_id;
                  let cls = if active {
                    "px-2.5 py-1 rounded-lg text-xs font-bold bg-slate-900 text-white dark:bg-white dark:text-slate-900 cursor-pointer"
                  } else {
                    "px-2.5 py-1 rounded-lg text-xs font-medium bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 cursor-pointer"
                  };
                  rsx! {
                    button {
                      key: "{s_id}",
                      r#type: "button",
                      class: cls,
                      onclick: move |_| series.set(s_id.to_string()),
                      "{s_lbl}"
                    }
                  }
                }
              }
            }
          }
        }

        // 仓库地址、发布日期、排序权重、标签
        div { class: "grid grid-cols-1 sm:grid-cols-4 gap-3",
          div { class: "sm:col-span-2",
            label { class: "block text-xs font-semibold text-slate-700 dark:text-slate-300 mb-1",
              "HuggingFace Repo (owner/model)"
            }
            Input {
              class: "h-9 text-xs font-mono",
              value: hf_repo(),
              on_value_change: move |v: String| hf_repo.set(v),
            }
          }
          div {
            label { class: "block text-xs font-semibold text-slate-700 dark:text-slate-300 mb-1",
              "ModelScope Repo (可选)"
            }
            Input {
              class: "h-9 text-xs font-mono",
              placeholder: "Qwen/Qwen2.5-VL-7B-Instruct",
              value: ms_repo(),
              on_value_change: move |v: String| ms_repo.set(v),
            }
          }
          div {
            label { class: "block text-xs font-semibold text-slate-700 dark:text-slate-300 mb-1",
              "发布月份 & 排序权重"
            }
            div { class: "flex gap-1.5",
              Input {
                class: "h-9 text-xs font-mono flex-1",
                placeholder: "2025-05",
                value: release_date(),
                on_value_change: move |v: String| release_date.set(v),
              }
              Input {
                class: "h-9 text-xs font-mono w-16",
                placeholder: "50",
                value: order_str(),
                on_value_change: move |v: String| order_str.set(v),
              }
            }
          }
        }

        div { class: "flex flex-col sm:flex-row sm:items-center gap-3",
          div { class: "flex-1",
            label { class: "block text-xs font-semibold text-slate-700 dark:text-slate-300 mb-1",
              "技术标签（逗号分隔）"
            }
            Input {
              class: "h-9 text-xs font-mono",
              placeholder: "vlm, gqa, qk-norm, moe",
              value: tags_str(),
              on_value_change: move |v: String| tags_str.set(v),
            }
          }
          label { class: "inline-flex items-center gap-2 text-xs font-semibold text-slate-700 dark:text-slate-300 pt-4 cursor-pointer",
            input {
              r#type: "checkbox",
              checked: featured(),
              onchange: move |evt| featured.set(evt.checked()),
            }
            "设为精选模型 (Featured)"
          }
        }

        // 双语摘要与双语技术亮点
        div { class: "grid grid-cols-1 lg:grid-cols-2 gap-3",
          div {
            label { class: "block text-xs font-semibold text-slate-700 dark:text-slate-300 mb-1",
              "中文架构简介 (Summary ZH)"
            }
            Textarea {
              class: "h-20 text-xs",
              value: summary(),
              on_value_change: move |v: String| summary.set(v),
            }
          }
          div {
            label { class: "block text-xs font-semibold text-slate-700 dark:text-slate-300 mb-1",
              "英文架构简介 (Summary EN)"
            }
            Textarea {
              class: "h-20 text-xs",
              value: summary_en(),
              on_value_change: move |v: String| summary_en.set(v),
            }
          }
          div {
            label { class: "block text-xs font-semibold text-slate-700 dark:text-slate-300 mb-1",
              "核心架构看点 ZH（每行一条，建议 3 条）"
            }
            Textarea {
              class: "h-28 text-xs",
              value: highlights_str(),
              on_value_change: move |v: String| highlights_str.set(v),
            }
          }
          div {
            label { class: "block text-xs font-semibold text-slate-700 dark:text-slate-300 mb-1",
              "核心架构看点 EN（每行一条）"
            }
            Textarea {
              class: "h-28 text-xs",
              value: highlights_en_str(),
              on_value_change: move |v: String| highlights_en_str.set(v),
            }
          }
        }

        // 可选深度拆解笔记 Markdown
        div {
          label { class: "block text-xs font-semibold text-slate-700 dark:text-slate-300 mb-1",
            "架构深度拆解笔记 Markdown（可选，渲染在详情页底部）"
          }
          Textarea {
            class: "h-24 text-xs font-mono",
            placeholder: "## 核心设计拆解\n可在此撰写公式、算子对比或部署建议...",
            value: readme_md(),
            on_value_change: move |v: String| readme_md.set(v),
          }
        }

        // 底部保存按钮栏
        div { class: "flex items-center justify-end gap-2.5 pt-3 border-t border-slate-200 dark:border-slate-800",
          Button {
            variant: ButtonVariant::Outline,
            size: ButtonSize::Sm,
            onclick: move |_| on_close.call(()),
            {lang.pick("取消", "Cancel")}
          }
          Button {
            size: ButtonSize::Sm,
            disabled: saving() || draft_bp().is_none(),
            onclick: do_save,
            if saving() {
              Spinner { size: SpinnerSize::Sm, class: "mr-1.5 border-t-current" }
              {lang.pick("正在写入数据库...", "Saving...")}
            } else {
              {lang.pick("保存并发布到图鉴", "Save & Publish")}
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
  fn scans_builtin_starter_models_and_sorts_by_order() {
    invalidate_models_cache();
    let list = scan_curated_models();
    assert!(list.len() >= 2, "expected at least 2 starter models in assets/models");
    assert_eq!(list[0].slug, "deepseek-v3");
    assert_eq!(list[0].modality, ModelModality::Llm);
    assert_eq!(list[0].origin, RecordOrigin::Builtin);
    assert!(list[0].has_mla);
    assert!(list[0].has_moe);
    assert!(list.iter().any(|m| m.slug == "qwen3-30b-a3b"));
  }
}
