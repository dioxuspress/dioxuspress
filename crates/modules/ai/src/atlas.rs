//! 指定模型架构图鉴（`/models` 索引页 + `/models/:slug` 详情页 + 内容目录扫描器）。
//!
//! 数据源目录优先级：
//! 1. 外部挂载内容库 `<CONTENT_DIR>/models/<slug>/`（如 `rustineverything-content/models/`，支持 `SIGHUP` 秒级热重载）；
//! 2. 框架内置示例库 `assets/models/<slug>/`。
//!
//! 每个模型子目录包含：
//! - `model.yaml`：模型名称、厂商、系列分类、仓库地址与核心架构看点；
//! - `config.json`：官方原始 HuggingFace / ModelScope 架构配置（由 `arch::build_model_blueprint` 精确推导）；
//! - `README.md`（可选）：深度架构剖析笔记。

#[cfg(feature = "server")]
use std::path::{Path, PathBuf};
#[cfg(feature = "server")]
use std::sync::Arc;

use app_core::i18n::Language;
use dioxus::prelude::*;
use dioxus_shadcn::{
  badge_class, button_class, card_class, BadgeVariant, Button, ButtonSize, ButtonVariant, Input,
  Spinner, SpinnerSize, UiDensity,
};
use serde::{Deserialize, Serialize};
use widgets::{Markdown, PageMeta};

use crate::arch::{format_bytes_iec, format_count, ModelBlueprint};
use crate::visualizer::ModelVisualizerWorkbench;

#[cfg(feature = "server")]
use crate::arch::{build_model_blueprint, load_builtin_preset, PrecisionPreset};
#[cfg(feature = "server")]
use crate::safetensors_meta::{parse_safetensors_index_json, summarize_index_and_headers};

fn current_lang() -> Language {
  try_consume_context::<Signal<Language>>().map(|s| s()).unwrap_or_default()
}

/// 图鉴列表卡片与侧边快速切换所需的轻量级模型摘要。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CuratedModelSummary {
  pub slug: String,
  pub name: String,
  pub vendor: String,
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

  // 以下字段由 `config.json` 实时精算得出，杜绝人工维护偏差
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
// 服务端目录扫描与 mtime 缓存（支持 SIGHUP 内容热重载）
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "server")]
static MODELS_CACHE: app_core::utils::DirListingCache<Vec<CuratedModelDetail>> =
  app_core::utils::DirListingCache::new();

/// 显式清空模型图鉴内存缓存（供 `hot_reload` 收到 `SIGHUP` 时调用）。
#[cfg(feature = "server")]
pub fn invalidate_models_cache() {
  MODELS_CACHE.invalidate();
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

/// 从指定 `models_root` 目录解析单个模型子目录（供运行时与 `dioxuspress check` CLI 复用）。
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

  let main_seg = blueprint.segments.last();
  let kv_compression_ratio = main_seg.map(|s| s.attention.kv_compression_ratio()).unwrap_or(1.0);
  let attention_short =
    main_seg.map(|s| s.attention.short_label()).unwrap_or_else(|| "Attention".to_string());
  let ffn_short = main_seg.map(|s| s.ffn.short_label()).unwrap_or_else(|| "FFN".to_string());
  let has_mla =
    matches!(main_seg.map(|s| &s.attention), Some(crate::arch::AttentionSpec::DeepSeekMla { .. }));
  let has_hybrid_attn = main_seg.is_some_and(|s| s.schedule_note.is_some());
  let eval_ctx = 32_768u64.min(blueprint.max_position_embeddings.max(4096));
  let fp8_vram_bytes = blueprint.estimate_vram(PrecisionPreset::Fp8, eval_ctx, 1).total_bytes;

  let series = normalize_series(&raw_meta.series, &repo_id);
  let name = if raw_meta.name.trim().is_empty() { slug.clone() } else { raw_meta.name };
  let vendor = if raw_meta.vendor.trim().is_empty() {
    repo_id.split('/').next().unwrap_or("OpenSource").to_string()
  } else {
    raw_meta.vendor
  };

  let meta = CuratedModelSummary {
    slug,
    name,
    vendor,
    series,
    release_date: raw_meta.release_date,
    hf_repo: repo_id,
    ms_repo: raw_meta.ms_repo.filter(|s| !s.trim().is_empty()),
    featured: raw_meta.featured,
    order: raw_meta.order,
    summary: raw_meta.summary,
    summary_en: raw_meta.summary_en,
    tags: raw_meta.tags,
    highlights: raw_meta.highlights,
    highlights_en: raw_meta.highlights_en,
    family_badge: blueprint.family.badge_label().to_string(),
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
  };

  Ok(CuratedModelDetail { meta, blueprint, readme_md })
}

#[cfg(feature = "server")]
fn normalize_series(raw: &str, repo_id: &str) -> String {
  let s = raw.trim().to_ascii_lowercase();
  if matches!(s.as_str(), "deepseek" | "qwen" | "llama" | "other") {
    return s;
  }
  let r = repo_id.to_ascii_lowercase();
  if r.contains("deepseek") {
    "deepseek".to_string()
  } else if r.contains("qwen") {
    "qwen".to_string()
  } else if r.contains("llama") {
    "llama".to_string()
  } else {
    "other".to_string()
  }
}

/// 加载全部指定模型详情（带目录指纹缓存）。
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

/// 供全文检索索引（`doc_source`）与站点地图（`seo.rs`）同步枚举全部已收录模型摘要。
#[cfg(feature = "server")]
pub fn scan_curated_models() -> Vec<CuratedModelSummary> {
  load_curated_models_full().iter().map(|d| d.meta.clone()).collect()
}

/// 获取全部指定模型卡片列表（SSR + 客户端筛选共用）。
#[get("/api/ai/models")]
pub async fn list_curated_models() -> Result<Vec<CuratedModelSummary>, ServerFnError> {
  Ok(scan_curated_models())
}

/// 按 `slug` 获取单个指定模型的完整架构蓝图与解读详情。
#[get("/api/ai/models/:slug")]
pub async fn get_curated_model_detail(
  slug: String,
) -> Result<Option<CuratedModelDetail>, ServerFnError> {
  let clean = slug.trim();
  if clean.is_empty() || clean.contains("..") || clean.contains('/') || clean.contains('\\') {
    return Ok(None);
  }
  let all = load_curated_models_full();
  Ok(all.iter().find(|d| d.meta.slug == clean).cloned())
}

// ─────────────────────────────────────────────────────────────────────────────
// `/models`：指定模型架构图鉴总览页
// ─────────────────────────────────────────────────────────────────────────────

#[component]
pub fn ModelsAtlasPage() -> Element {
  let lang = current_lang();
  let title = lang.pick("主流大模型架构图鉴", "Flagship LLM Architecture Atlas").to_string();
  let desc = lang
    .pick(
      "精选 DeepSeek、Qwen3、Llama 4、Gemma 3、Mixtral、GLM-4 等主流开源旗舰模型，支持多尺度收起与展开观察 MLA、稀疏 MoE、iRoPE、混合滑动窗口及视觉编码支路。",
      "Interactive, multi-scale collapsible architecture diagrams for DeepSeek, Qwen3, Llama 4, Gemma 3, Mixtral, and GLM-4 derived from official configs.",
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
  let models_res = use_server_future(list_curated_models)?;
  let models = match models_res() {
    Some(Ok(list)) => list,
    Some(Err(_)) => {
      return rsx! {
        widgets::LoadFailed {
          message: lang.pick("模型图鉴加载失败，请稍后重试", "Failed to load Model Atlas").to_string(),
        }
      };
    }
    None => Vec::new(),
  };

  let mut series_filter = use_signal(|| "all".to_string());
  let mut trait_filter = use_signal(|| ArchTraitFilter::All);
  let mut search_q = use_signal(String::new);
  let mut custom_repo = use_signal(String::new);
  let nav = use_navigator();

  let count_series = |s: &str| -> usize {
    if s == "all" {
      models.len()
    } else {
      models.iter().filter(|m| m.series == s).count()
    }
  };

  let q_norm = search_q().trim().to_ascii_lowercase();
  let cur_series = series_filter();
  let cur_trait = trait_filter();

  let filtered: Vec<CuratedModelSummary> = models
    .iter()
    .filter(|m| cur_series == "all" || m.series == cur_series)
    .filter(|m| match cur_trait {
      ArchTraitFilter::All => true,
      ArchTraitFilter::Mla => m.has_mla,
      ArchTraitFilter::Moe => m.has_moe,
      ArchTraitFilter::Dense => !m.has_moe,
      ArchTraitFilter::Multimodal => m.has_vision,
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
        || m.summary.to_ascii_lowercase().contains(&q_norm)
        || m.summary_en.to_ascii_lowercase().contains(&q_norm)
        || m.tags.iter().any(|t| t.to_ascii_lowercase().contains(&q_norm))
    })
    .cloned()
    .collect();

  let series_tabs = [
    ("all", lang.pick("全部旗舰", "All Models")),
    ("deepseek", lang.pick("DeepSeek 系列", "DeepSeek")),
    ("qwen", lang.pick("Qwen 通义系列", "Qwen")),
    ("llama", lang.pick("Llama 系列", "Llama")),
    ("other", lang.pick("主流开源 (Mixtral/Gemma/GLM)", "Mixtral · Gemma · GLM")),
  ];

  let trait_chips = [
    (ArchTraitFilter::All, lang.pick("全部架构特性", "All Traits")),
    (ArchTraitFilter::Mla, "MLA 潜空间压缩"),
    (ArchTraitFilter::Moe, "Sparse MoE 混合专家"),
    (ArchTraitFilter::Dense, "Dense 稠密主干"),
    (ArchTraitFilter::Multimodal, "Multimodal 视觉多模态"),
    (ArchTraitFilter::HybridAttn, "SWA / iRoPE 交替层"),
  ];

  rsx! {
    div { class: "space-y-8",
      // 1. 顶部主视觉区 + 任意 HF/ModelScope 模型实时沙盒直达栏
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

          // 实时拉取任意仓库快捷入口（跳转 `/ai/visualizer?repo=...`）
          div { class: "w-full lg:w-96 shrink-0 rounded-2xl border border-indigo-200/80 dark:border-indigo-900/60 bg-white/90 dark:bg-slate-900/90 p-3.5",
            div { class: "flex items-center justify-between text-xs font-semibold text-slate-700 dark:text-slate-200 mb-2",
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
          }
        }
      }

      // 2. 多维筛选控制台：系列页签 + 架构特性胶囊 + 搜索框
      div { class: "space-y-3",
        div { class: "flex flex-col md:flex-row md:items-center justify-between gap-3",
          div { class: "flex flex-wrap gap-1.5",
            for (s_id, s_lbl) in series_tabs {
              {
                let cnt = count_series(s_id);
                let active = series_filter() == s_id;
                let cls = if active {
                  "px-3.5 py-1.5 rounded-xl text-xs font-bold bg-slate-900 text-white dark:bg-white dark:text-slate-900 shadow-xs cursor-pointer transition-all"
                } else {
                  "px-3.5 py-1.5 rounded-xl text-xs font-semibold bg-slate-100 dark:bg-slate-900 text-slate-600 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-800 cursor-pointer transition-all"
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
              placeholder: lang.pick("搜索模型名、架构、标签...", "Filter models, arch, tags..."),
              "aria-label": "Filter curated models",
              value: search_q(),
              on_value_change: move |v: String| search_q.set(v),
            }
          }
        }

        // 架构机制快速过滤条
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
          {lang.pick("没有匹配当前筛选条件的模型，试试切换「全部旗舰」或清空搜索词。", "No models match the current filters.")}
        }
      } else {
        div { class: "grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-5",
          for m in filtered {
            CuratedModelCard { key: "{m.slug}", model: m }
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

#[component]
fn CuratedModelCard(model: CuratedModelSummary) -> Element {
  let lang = current_lang();
  let (_, series_cls) = series_accent_badge(&model.series);
  let is_sparse = model.active_params < model.total_params;
  let detail_href = format!("/models/{}", model.slug);

  rsx! {
    Link {
      to: detail_href,
      class: "group flex flex-col justify-between rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/70 p-5 hover:border-indigo-400 dark:hover:border-indigo-500 hover:shadow-md transition-all",
      div {
        // 厂商徽章 + 家族标签
        div { class: "flex flex-wrap items-center justify-between gap-2 mb-3",
          span { class: "px-2 py-0.5 rounded-md border text-[11px] font-bold {series_cls}",
            "{model.vendor}"
          }
          div { class: "flex items-center gap-1.5",
            if model.has_vision {
              span { class: "px-2 py-0.5 rounded-md bg-fuchsia-500/15 text-fuchsia-700 dark:text-fuchsia-300 text-[10px] font-semibold",
                "Multimodal"
              }
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
  let detail_res =
    use_server_future(use_reactive!(|slug| async move { get_curated_model_detail(slug).await }))?;
  let all_res = use_server_future(list_curated_models)?;

  let all_models = all_res().and_then(Result::ok).unwrap_or_default();

  let detail = match detail_res() {
    Some(Ok(Some(d))) => d,
    Some(Err(_)) => {
      return rsx! {
        widgets::LoadFailed {
          message: lang.pick("加载模型架构详情失败", "Failed to load model architecture").to_string(),
        }
      };
    }
    _ => {
      return rsx! {
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
      };
    }
  };

  let meta = &detail.meta;
  let highlights = meta.localized_highlights(lang);
  let hf_url = format!("https://huggingface.co/{}", meta.hf_repo);
  let ms_url = meta.ms_repo.as_ref().map(|r| format!("https://modelscope.cn/models/{r}"));
  let live_sandbox_href = format!("/ai/visualizer?repo={}", meta.hf_repo);

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

      // 2. 模型核心架构导读与工程看点卡
      div { class: card_class("p-5 sm:p-6 bg-linear-to-r from-indigo-50/60 via-white to-slate-50/70 dark:from-indigo-950/30 dark:via-slate-900 dark:to-slate-900"),
        div { class: "flex flex-wrap items-baseline justify-between gap-2",
          h1 { class: "text-xl sm:text-2xl font-extrabold text-slate-900 dark:text-white",
            "{meta.name}"
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
        key: "atlas-{meta.slug}",
        blueprint: detail.blueprint.clone(),
        initial_focus: None,
        embedded: false,
      }

      // 4. 可选深度架构解读文档（若 `models/<slug>/README.md` 存在）
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

#[cfg(all(test, feature = "server"))]
mod tests {
  use super::*;

  #[test]
  fn scans_builtin_starter_models_and_sorts_by_order() {
    invalidate_models_cache();
    let list = scan_curated_models();
    assert!(list.len() >= 2, "expected at least 2 starter models in assets/models");
    assert_eq!(list[0].slug, "deepseek-v3");
    assert!(list[0].has_mla);
    assert!(list[0].has_moe);
    assert!(list.iter().any(|m| m.slug == "qwen3-30b-a3b"));
  }
}
