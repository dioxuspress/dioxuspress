//! 传统机器学习交互式算法图鉴（`/ai/ml` 全景导航与对比表 + `/ai/ml/:slug` 2D 交互实验台与 Rust 生态对照）。
//!
//! 数据源合并规则（零数据库依赖，天然兼容 `SIGHUP` 热刷新）：
//! 1. 先加载框架内置标杆词条 `assets/ai-ml/<slug>/index.md`；
//! 2. 若挂载外部内容仓库 `<CONTENT_DIR>/ai-ml/<slug>/index.md`，自动覆盖同名 `slug` 并追加全部扩展词条。
//!
//! WASM 体积守恒策略：服务端接口返回统一 JSON 字符串，避免在 WASM 中额外单态化嵌套 CBOR 状态机。

#[cfg(feature = "server")]
use std::collections::BTreeMap;
#[cfg(feature = "server")]
use std::path::{Path, PathBuf};
#[cfg(feature = "server")]
use std::sync::Arc;

use app_core::i18n::Language;
use dioxus::prelude::*;
use dioxus_shadcn::{badge_class, card_class, BadgeVariant, Input, Spinner, SpinnerSize};
#[cfg(feature = "server")]
use serde::Deserialize;
use widgets::{Markdown, OnThisPage, PageMeta};

use crate::component_labs::current_lang;
#[cfg(feature = "server")]
use crate::ml_labs::infer_ml_lab_by_slug;
use crate::ml_labs::{MlLabKind, MlLabWorkbench};

/// 传统机器学习算法四大族系。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MlFamily {
  #[default]
  TreeEnsemble,
  LinearKernel,
  ClusterDim,
  InstanceProb,
}

impl MlFamily {
  #[cfg(feature = "server")]
  pub fn as_str(self) -> &'static str {
    match self {
      Self::TreeEnsemble => "tree_ensemble",
      Self::LinearKernel => "linear_kernel",
      Self::ClusterDim => "cluster_dim",
      Self::InstanceProb => "instance_prob",
    }
  }

  pub fn parse_str(raw: &str) -> Self {
    match raw.trim().to_ascii_lowercase().as_str() {
      "tree" | "tree_ensemble" | "ensemble" | "boosting" => Self::TreeEnsemble,
      "linear" | "kernel" | "linear_kernel" | "svm" => Self::LinearKernel,
      "cluster" | "dim" | "cluster_dim" | "unsupervised" | "pca" => Self::ClusterDim,
      "instance" | "prob" | "instance_prob" | "knn" | "bayes" => Self::InstanceProb,
      _ => Self::TreeEnsemble,
    }
  }

  pub fn label(self, lang: Language) -> &'static str {
    match self {
      Self::TreeEnsemble => lang.pick("树模型与集成学习", "Trees & Ensembles"),
      Self::LinearKernel => lang.pick("线性模型与核方法", "Linear & Kernel Methods"),
      Self::ClusterDim => lang.pick("聚类与降维投影", "Clustering & Dim Reduction"),
      Self::InstanceProb => lang.pick("实例近邻与概率模型", "Instance & Probabilistic"),
    }
  }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MlSummary {
  pub slug: String,
  pub title: String,
  pub title_en: String,
  pub family: MlFamily,
  pub lab_kind: String,
  pub lab_mode: String,
  pub time_complexity: String,
  pub space_complexity: String,
  pub needs_scaling: bool,
  pub interpretability: String,
  pub rust_crates: String,
  pub formula_short: String,
  pub summary: String,
  pub summary_en: String,
  pub tags: Vec<String>,
  pub order: i32,
}

impl MlSummary {
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MlSiblingLink {
  pub slug: String,
  pub title: String,
  pub title_en: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MlDetail {
  pub meta: MlSummary,
  pub body_md: String,
  pub related_models: Vec<String>,
  pub related_math: Vec<String>,
  pub siblings: Vec<MlSiblingLink>,
}

const FSEP: char = '\x1f';
const RSEP: char = '\x1e';

#[cfg(feature = "server")]
fn push_clean(out: &mut String, s: &str) {
  for ch in s.chars() {
    if ch == FSEP || ch == RSEP {
      out.push(' ');
    } else {
      out.push(ch);
    }
  }
}

#[cfg(feature = "server")]
fn encode_summary_row(out: &mut String, m: &MlSummary) {
  let tags = m.tags.join(",");
  let scaling = if m.needs_scaling { "1" } else { "0" };
  let order_s = m.order.to_string();
  let fields = [
    m.slug.as_str(),
    m.title.as_str(),
    m.title_en.as_str(),
    m.family.as_str(),
    m.lab_kind.as_str(),
    m.lab_mode.as_str(),
    m.time_complexity.as_str(),
    m.space_complexity.as_str(),
    scaling,
    m.interpretability.as_str(),
    m.rust_crates.as_str(),
    m.formula_short.as_str(),
    m.summary.as_str(),
    m.summary_en.as_str(),
    tags.as_str(),
    order_s.as_str(),
  ];
  for (i, f) in fields.iter().enumerate() {
    if i > 0 {
      out.push(FSEP);
    }
    push_clean(out, f);
  }
}

#[inline(never)]
fn decode_summary_row(line: &str) -> Option<MlSummary> {
  let mut it = line.split(FSEP);
  let mut next = || it.next().unwrap_or("").to_string();
  let slug = next();
  if slug.is_empty() {
    return None;
  }
  let title = next();
  let title_en = next();
  let family = MlFamily::parse_str(&next());
  let lab_kind = next();
  let lab_mode = next();
  let time_complexity = next();
  let space_complexity = next();
  let needs_scaling = next() == "1";
  let interpretability = next();
  let rust_crates = next();
  let formula_short = next();
  let summary = next();
  let summary_en = next();
  let tags_raw = next();
  let tags = tags_raw.split(',').filter(|s| !s.is_empty()).map(str::to_string).collect();
  let order = next().parse::<i32>().unwrap_or(100);
  Some(MlSummary {
    slug,
    title,
    title_en,
    family,
    lab_kind,
    lab_mode,
    time_complexity,
    space_complexity,
    needs_scaling,
    interpretability,
    rust_crates,
    formula_short,
    summary,
    summary_en,
    tags,
    order,
  })
}

#[inline(never)]
fn decode_summary_list(wire: &str) -> Vec<MlSummary> {
  wire.split(RSEP).filter_map(decode_summary_row).collect()
}

#[inline(never)]
fn decode_detail_wire(wire: &str) -> Option<MlDetail> {
  let mut sections = wire.split(RSEP);
  let meta = decode_summary_row(sections.next()?)?;
  let rel_models = sections
    .next()
    .unwrap_or("")
    .split(',')
    .filter(|s| !s.is_empty())
    .map(str::to_string)
    .collect();
  let rel_math = sections
    .next()
    .unwrap_or("")
    .split(',')
    .filter(|s| !s.is_empty())
    .map(str::to_string)
    .collect();
  let siblings = sections
    .next()
    .unwrap_or("")
    .split('\n')
    .filter_map(|line| {
      let mut it = line.split(FSEP);
      let slug = it.next()?.to_string();
      if slug.is_empty() {
        return None;
      }
      Some(MlSiblingLink {
        slug,
        title: it.next().unwrap_or("").to_string(),
        title_en: it.next().unwrap_or("").to_string(),
      })
    })
    .collect();
  let body_md = sections.next().unwrap_or("").to_string();
  Some(MlDetail { meta, body_md, related_models: rel_models, related_math: rel_math, siblings })
}

// ─────────────────────────────────────────────────────────────────────────────
// 服务端：零数据库双源扫描与内存缓存（支持 SIGHUP 毫秒级热刷新）
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "server")]
#[derive(Debug, Clone)]
pub struct StoredMlEntry {
  pub meta: MlSummary,
  pub body_md: String,
  pub related_models: Vec<String>,
  pub related_math: Vec<String>,
}

#[cfg(feature = "server")]
#[derive(Debug, Deserialize, Default)]
struct MlFrontmatter {
  #[serde(default)]
  slug: String,
  #[serde(default)]
  title: String,
  #[serde(default)]
  title_en: String,
  #[serde(default)]
  family: String,
  #[serde(default)]
  lab_kind: String,
  #[serde(default)]
  lab_mode: String,
  #[serde(default)]
  time_complexity: String,
  #[serde(default)]
  space_complexity: String,
  #[serde(default)]
  needs_scaling: bool,
  #[serde(default)]
  interpretability: String,
  #[serde(default)]
  rust_crates: String,
  #[serde(default)]
  formula_short: String,
  #[serde(default)]
  summary: String,
  #[serde(default)]
  summary_en: String,
  #[serde(default)]
  tags: Vec<String>,
  #[serde(default)]
  related_models: Vec<String>,
  #[serde(default)]
  related_math: Vec<String>,
  #[serde(default = "default_ml_order")]
  order: i32,
}

#[cfg(feature = "server")]
fn default_ml_order() -> i32 {
  100
}

#[cfg(feature = "server")]
fn ml_cache() -> &'static std::sync::RwLock<Option<Arc<Vec<StoredMlEntry>>>> {
  static CACHE: std::sync::OnceLock<std::sync::RwLock<Option<Arc<Vec<StoredMlEntry>>>>> =
    std::sync::OnceLock::new();
  CACHE.get_or_init(|| std::sync::RwLock::new(None))
}

#[cfg(feature = "server")]
pub fn invalidate_ml_cache() {
  if let Ok(mut guard) = ml_cache().write() {
    *guard = None;
  }
}

#[cfg(feature = "server")]
fn builtin_ml_dir() -> PathBuf {
  let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
  for cand in [
    cwd.join("assets/ai-ml"),
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../assets/ai-ml"),
  ] {
    if cand.is_dir() {
      return cand;
    }
  }
  cwd.join("assets/ai-ml")
}

#[cfg(feature = "server")]
fn external_ml_dir() -> Option<PathBuf> {
  app_core::utils::content_root().map(|r| r.join("ai-ml")).filter(|p| p.is_dir())
}

#[cfg(feature = "server")]
fn parse_ml_markdown(dir_slug: &str, raw: &str) -> Option<StoredMlEntry> {
  let (fm_str, body_str) = widgets::split_frontmatter(raw)?;
  let fm: MlFrontmatter = serde_yaml::from_str(fm_str).ok()?;
  let slug =
    if fm.slug.trim().is_empty() { dir_slug.to_string() } else { fm.slug.trim().to_string() };
  if !app_core::utils::is_safe_slug(&slug) {
    return None;
  }
  let (kind, mode) = infer_ml_lab_by_slug(&slug, &fm.lab_kind, &fm.lab_mode);
  let title = if fm.title.trim().is_empty() { slug.clone() } else { fm.title.trim().to_string() };
  let interp = if fm.interpretability.trim().is_empty() {
    "medium".to_string()
  } else {
    fm.interpretability.trim().to_ascii_lowercase()
  };

  Some(StoredMlEntry {
    meta: MlSummary {
      slug,
      title,
      title_en: fm.title_en.trim().to_string(),
      family: MlFamily::parse_str(&fm.family),
      lab_kind: kind.as_str().to_string(),
      lab_mode: mode,
      time_complexity: fm.time_complexity.trim().to_string(),
      space_complexity: fm.space_complexity.trim().to_string(),
      needs_scaling: fm.needs_scaling,
      interpretability: interp,
      rust_crates: fm.rust_crates.trim().to_string(),
      formula_short: fm.formula_short.trim().to_string(),
      summary: fm.summary.trim().to_string(),
      summary_en: fm.summary_en.trim().to_string(),
      tags: fm.tags,
      order: fm.order,
    },
    body_md: body_str.trim().to_string(),
    related_models: fm.related_models,
    related_math: fm.related_math,
  })
}

#[cfg(feature = "server")]
fn scan_ml_dir_into(root: &Path, out: &mut BTreeMap<String, StoredMlEntry>) {
  let Ok(entries) = std::fs::read_dir(root) else {
    return;
  };
  for ent in entries.flatten() {
    let p = ent.path();
    if !p.is_dir() {
      continue;
    }
    let Some(dir_name) = p.file_name().and_then(|s| s.to_str()) else {
      continue;
    };
    let md_path = p.join("index.md");
    let Ok(raw) = std::fs::read_to_string(&md_path) else {
      continue;
    };
    if let Some(parsed) = parse_ml_markdown(dir_name, &raw) {
      out.insert(parsed.meta.slug.clone(), parsed);
    }
  }
}

#[cfg(feature = "server")]
pub fn load_all_ml_entries() -> Arc<Vec<StoredMlEntry>> {
  if let Ok(guard) = ml_cache().read() {
    if let Some(cached) = guard.as_ref() {
      return Arc::clone(cached);
    }
  }
  let mut merged = BTreeMap::new();
  scan_ml_dir_into(&builtin_ml_dir(), &mut merged);
  if let Some(ext) = external_ml_dir() {
    scan_ml_dir_into(&ext, &mut merged);
  }
  let mut list: Vec<StoredMlEntry> = merged.into_values().collect();
  list.sort_by(|a, b| a.meta.order.cmp(&b.meta.order).then_with(|| a.meta.slug.cmp(&b.meta.slug)));
  let arc = Arc::new(list);
  if let Ok(mut guard) = ml_cache().write() {
    *guard = Some(Arc::clone(&arc));
  }
  arc
}

/// 注册进全站 Tantivy 检索索引。
#[cfg(feature = "server")]
pub fn search_docs() -> Vec<app_core::engines::doc_source::ExternalIndexedDoc> {
  load_all_ml_entries()
    .iter()
    .map(|e| app_core::engines::doc_source::ExternalIndexedDoc {
      kind: "ai".to_string(),
      ref_id: format!("ml-{}", e.meta.slug),
      title: e.meta.title.clone(),
      body: format!(
        "{} {} {} {}",
        e.meta.summary, e.meta.formula_short, e.meta.rust_crates, e.body_md
      ),
      url: format!("/ai/ml/{}", e.meta.slug),
      created_at: String::new(),
    })
    .collect()
}

#[get("/api/ai/ml/list")]
pub async fn list_ai_ml_wire() -> Result<String, ServerFnError> {
  let all = load_all_ml_entries();
  let mut out = String::with_capacity(all.len() * 256);
  for (idx, e) in all.iter().enumerate() {
    if idx > 0 {
      out.push(RSEP);
    }
    encode_summary_row(&mut out, &e.meta);
  }
  Ok(out)
}

#[get("/api/ai/ml/detail/:slug")]
pub async fn get_ai_ml_detail_wire(slug: String) -> Result<String, ServerFnError> {
  let clean = slug.trim();
  if !app_core::utils::is_safe_slug(clean) {
    return Ok(String::new());
  }
  let all = load_all_ml_entries();
  let Some(found) = all.iter().find(|e| e.meta.slug == clean) else {
    return Ok(String::new());
  };
  let mut out = String::with_capacity(found.body_md.len() + 512);
  encode_summary_row(&mut out, &found.meta);
  out.push(RSEP);
  push_clean(&mut out, &found.related_models.join(","));
  out.push(RSEP);
  push_clean(&mut out, &found.related_math.join(","));
  out.push(RSEP);
  let mut first_sib = true;
  for e in all.iter().filter(|e| e.meta.family == found.meta.family) {
    if !first_sib {
      out.push('\n');
    }
    first_sib = false;
    push_clean(&mut out, &e.meta.slug);
    out.push(FSEP);
    push_clean(&mut out, &e.meta.title);
    out.push(FSEP);
    push_clean(&mut out, &e.meta.title_en);
  }
  out.push(RSEP);
  out.push_str(&found.body_md);
  Ok(out)
}

fn interp_label(raw: &str, lang: Language) -> &'static str {
  match raw {
    "high" => lang.pick("高（白盒可解释）", "High (White-box)"),
    "low" => lang.pick("低（黑盒核/集成）", "Low (Black-box)"),
    _ => lang.pick("中（特征重要性/边界）", "Medium"),
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// `/ai/ml`：传统机器学习全景图鉴与工程选型对比页
// ─────────────────────────────────────────────────────────────────────────────

#[inline(never)]
fn join_path(prefix: &str, slug: &str) -> String {
  let mut s = String::with_capacity(prefix.len() + slug.len());
  s.push_str(prefix);
  s.push_str(slug);
  s
}

#[inline(never)]
fn ml_page_shell(inner: Element) -> Element {
  rsx! {
    section { class: "py-8 sm:py-10 min-h-screen bg-white dark:bg-slate-950",
      div { class: "max-w-7xl mx-auto px-4 sm:px-6",
        SuspenseBoundary {
          fallback: |_| rsx! {
            div { class: "flex items-center justify-center py-24",
              Spinner { size: SpinnerSize::Lg, class: "border-t-primary" }
            }
          },
          {inner}
        }
      }
    }
  }
}

#[component]
pub fn AiMlAtlasPage() -> Element {
  let lang = current_lang();
  let title = lang
    .pick("传统机器学习交互式算法图鉴", "Classic Machine Learning Interactive Atlas")
    .to_string();
  let desc = lang
    .pick(
      "决策树、随机森林、GBDT、XGBoost、线性/岭回归、逻辑回归、SVM、K-Means、PCA 与 KNN 的纯 Rust WASM 2D 交互画布、数学目标推导、复杂度选型表与 linfa/smartcore 代码对照。",
      "Interactive 2D Rust WASM visualizations, objective derivations, complexity comparisons, and linfa/smartcore code across trees, boosting, SVM, regression, clustering, and PCA.",
    )
    .to_string();

  rsx! {
    PageMeta {
      title: title.clone(),
      description: desc.clone(),
      path: "/ai/ml".to_string(),
    }
    {ml_page_shell(rsx! { AiMlLoaded { title, desc } })}
  }
}

#[component]
fn AiMlLoaded(title: String, desc: String) -> Element {
  let lang = current_lang();
  let res = use_server_future(list_ai_ml_wire)?;
  let items: Vec<MlSummary> = match res() {
    Some(Ok(raw)) => decode_summary_list(&raw),
    Some(Err(_)) => {
      return rsx! {
        widgets::LoadFailed {
          message: lang.pick("传统机器学习图鉴加载失败，请稍后重试", "Failed to load Classic ML Atlas").to_string(),
        }
      };
    }
    None => Vec::new(),
  };

  let mut family_filter: Signal<Option<MlFamily>> = use_signal(|| None);
  let mut search_q = use_signal(String::new);
  let cur_fam = family_filter();
  let q = search_q().trim().to_ascii_lowercase();

  let filtered: Vec<MlSummary> = items
    .iter()
    .filter(|it| cur_fam.is_none_or(|f| it.family == f))
    .filter(|it| {
      q.is_empty()
        || it.slug.to_ascii_lowercase().contains(&q)
        || it.title.to_ascii_lowercase().contains(&q)
        || it.title_en.to_ascii_lowercase().contains(&q)
        || it.rust_crates.to_ascii_lowercase().contains(&q)
        || it.tags.iter().any(|t| t.to_ascii_lowercase().contains(&q))
    })
    .cloned()
    .collect();

  rsx! {
    div { class: "space-y-7",
      // Hero + 跨频道跳转
      div { class: "rounded-2xl border border-teal-200/80 dark:border-teal-900/70 bg-linear-to-br from-teal-50/90 via-white to-sky-50/70 dark:from-teal-950/45 dark:via-slate-900 dark:to-slate-950 p-6 sm:p-8",
        div { class: "flex flex-wrap items-center justify-between gap-4",
          div { class: "max-w-3xl",
            div { class: "inline-flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-teal-700 dark:text-teal-400",
              span { class: "px-2.5 py-0.5 rounded-full bg-teal-600 text-white text-[10px]", "Classic ML Atlas" }
              span { "Trees · Boosting · SVM · Regression · K-Means · PCA · KNN" }
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
              class: "px-3.5 py-2 rounded-xl border border-indigo-200 dark:border-indigo-800 bg-white dark:bg-slate-900 text-xs font-bold text-indigo-700 dark:text-indigo-300 hover:border-indigo-500 transition-colors",
              {lang.pick("🏛️ 神经网络与大模型图鉴 (/models) →", "🏛️ Neural & LLM Atlas →")}
            }
            Link {
              to: "/ai/math",
              class: "px-3.5 py-2 rounded-xl bg-teal-600 text-white text-xs font-bold hover:bg-teal-700 transition-colors",
              {lang.pick("📐 AI 数学推导实验室 (/ai/math) →", "📐 Math Proof Lab →")}
            }
          }
        }

        // 筛选 + 搜索
        div { class: "mt-5 pt-5 border-t border-teal-200/60 dark:border-slate-800 flex flex-col sm:flex-row gap-3 items-stretch sm:items-center justify-between",
          div { class: "flex flex-wrap gap-1.5",
            for (opt_fam, btn_lbl) in [
              (None, lang.pick("全部算法", "All")),
              (Some(MlFamily::TreeEnsemble), MlFamily::TreeEnsemble.label(lang)),
              (Some(MlFamily::LinearKernel), MlFamily::LinearKernel.label(lang)),
              (Some(MlFamily::ClusterDim), MlFamily::ClusterDim.label(lang)),
              (Some(MlFamily::InstanceProb), MlFamily::InstanceProb.label(lang)),
            ] {
              {
                let btn_cls = if cur_fam == opt_fam {
                  "px-3 py-1 rounded-full text-xs font-bold bg-teal-600 text-white cursor-pointer"
                } else {
                  "px-3 py-1 rounded-full text-xs font-medium bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-300 cursor-pointer"
                };
                rsx! {
                  button {
                    key: "{btn_lbl}",
                    r#type: "button",
                    class: "{btn_cls}",
                    onclick: move |_| family_filter.set(opt_fam),
                    "{btn_lbl}"
                  }
                }
              }
            }
          }
          div { class: "w-full sm:w-64",
            Input {
              class: "text-xs bg-white dark:bg-slate-950",
              placeholder: lang.pick("搜索算法或 Crate（如 XGBoost / SVM / linfa）", "Search algorithms or crates"),
              "aria-label": "Search classic ML algorithms",
              value: search_q(),
              on_value_change: move |v: String| search_q.set(v),
            }
          }
        }
      }

      // 卡片矩阵
      div { class: "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5",
        for item in filtered.iter() {
          {
            let href = format!("/ai/ml/{}", item.slug);
            let scale_badge = if item.needs_scaling {
              lang.pick("需归一化", "Needs Scaling")
            } else {
              lang.pick("尺度不变", "Scale-Invariant")
            };
            let scale_cls = if item.needs_scaling {
              "bg-amber-500/15 text-amber-700 dark:text-amber-300"
            } else {
              "bg-emerald-500/15 text-emerald-700 dark:text-emerald-300"
            };
            rsx! {
              Link {
                key: "{item.slug}",
                to: href,
                class: "group flex flex-col justify-between rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/70 p-5 hover:border-teal-400 dark:hover:border-teal-500 hover:shadow-md transition-all",
                div {
                  div { class: "flex flex-wrap items-center justify-between gap-2 mb-2.5",
                    span { class: "px-2 py-0.5 rounded-md bg-teal-500/15 text-teal-700 dark:text-teal-300 text-[11px] font-bold",
                      "{item.family.label(lang)}"
                    }
                    span { class: "px-2 py-0.5 rounded-md text-[10px] font-bold {scale_cls}",
                      "{scale_badge}"
                    }
                  }
                  h2 { class: "text-base sm:text-lg font-extrabold text-slate-900 dark:text-white group-hover:text-teal-600 dark:group-hover:text-teal-400 transition-colors",
                    "{item.localized_title(lang)}"
                  }
                  p { class: "mt-2 text-xs text-slate-600 dark:text-slate-400 line-clamp-3 leading-relaxed",
                    "{item.localized_summary(lang)}"
                  }
                  if !item.formula_short.is_empty() {
                    div { class: "mt-3 rounded-lg bg-slate-50 dark:bg-slate-950/80 border border-slate-200/80 dark:border-slate-800 px-2.5 py-1.5 text-[11px] font-mono text-teal-700 dark:text-teal-300 truncate",
                      "{item.formula_short}"
                    }
                  }
                }
                div { class: "mt-4 pt-3 border-t border-slate-100 dark:border-slate-800 flex items-center justify-between text-[11px] font-mono text-slate-500 dark:text-slate-400",
                  span { class: "truncate max-w-[60%]", "{item.time_complexity}" }
                  span { class: "font-bold text-teal-600 dark:text-teal-400 shrink-0",
                    {lang.pick("进入 2D 画布 →", "Open Lab →")}
                  }
                }
              }
            }
          }
        }
      }

      // 横向工程选型速查表
      if !items.is_empty() {
        div { class: card_class("p-5 overflow-x-auto"),
          h3 { class: "text-sm font-extrabold text-slate-900 dark:text-white mb-3",
            {lang.pick("传统机器学习工程选型速查矩阵", "Classic ML Engineering Selection Matrix")}
          }
          table { class: "w-full text-left text-xs border-collapse min-w-[680px]",
            thead { class: "border-b border-slate-200 dark:border-slate-800 text-slate-500 dark:text-slate-400",
              tr {
                th { class: "py-2 pr-3", {lang.pick("算法", "Algorithm")} }
                th { class: "py-2 pr-3", {lang.pick("训练复杂度", "Train Time")} }
                th { class: "py-2 pr-3", {lang.pick("归一化", "Scaling")} }
                th { class: "py-2 pr-3", {lang.pick("可解释性", "Interpretability")} }
                th { class: "py-2", "Rust Crate" }
              }
            }
            tbody { class: "divide-y divide-slate-100 dark:divide-slate-800/70",
              for row in items.iter() {
                tr { key: "{row.slug}",
                  td { class: "py-2 pr-3 font-bold",
                    Link { to: format!("/ai/ml/{}", row.slug), class: "text-teal-700 dark:text-teal-300 hover:underline",
                      "{row.localized_title(lang)}"
                    }
                  }
                  td { class: "py-2 pr-3 font-mono text-slate-600 dark:text-slate-300", "{row.time_complexity}" }
                  td { class: "py-2 pr-3",
                    if row.needs_scaling { "是 / Yes" } else { "否 / No" }
                  }
                  td { class: "py-2 pr-3 text-slate-600 dark:text-slate-300", "{interp_label(&row.interpretability, lang)}" }
                  td { class: "py-2 font-mono text-indigo-600 dark:text-indigo-400", "{row.rust_crates}" }
                }
              }
            }
          }
        }
      }
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// `/ai/ml/:slug`：单个传统机器学习算法详情页 + 2D 实时实验台
// ─────────────────────────────────────────────────────────────────────────────

#[component]
pub fn AiMlDetailPage(slug: String) -> Element {
  ml_page_shell(rsx! { AiMlDetailLoaded { slug } })
}

#[component]
fn AiMlDetailLoaded(slug: String) -> Element {
  let lang = current_lang();
  let res =
    use_server_future(use_reactive!(|slug| async move { get_ai_ml_detail_wire(slug).await }))?;
  let detail: MlDetail = match res() {
    Some(Ok(raw)) if !raw.is_empty() => match decode_detail_wire(&raw) {
      Some(d) => d,
      None => {
        return rsx! {
          widgets::LoadFailed {
            message: lang.pick("算法词条数据解析失败", "Failed to decode ML entry").to_string(),
          }
        };
      }
    },
    Some(Ok(_)) => {
      return rsx! {
        widgets::NotFound {
          message: lang.pick("未找到该传统机器学习算法词条", "Classic ML entry not found").to_string(),
        }
      };
    }
    Some(Err(_)) => {
      return rsx! {
        widgets::LoadFailed {
          message: lang.pick("加载传统机器学习详情失败", "Failed to load Classic ML detail").to_string(),
        }
      };
    }
    None => return rsx! {},
  };

  let toc = widgets::heading_outline(&detail.body_md);
  let title = detail.meta.localized_title(lang).to_string();
  let desc = detail.meta.localized_summary(lang).to_string();
  let lab_kind = MlLabKind::parse_str(&detail.meta.lab_kind);
  let scale_txt = if detail.meta.needs_scaling {
    lang.pick("强依赖特征标准化", "Requires Feature Scaling")
  } else {
    lang.pick("对单调缩放天然不敏感", "Scale-Invariant")
  };

  rsx! {
    PageMeta {
      title: format!("{title} · 传统机器学习图鉴"),
      description: desc.clone(),
      path: format!("/ai/ml/{}", detail.meta.slug),
    }

    div { class: "space-y-6",
      // 面包屑与头卡
      div { class: "space-y-3",
        div { class: "flex flex-wrap items-center gap-2 text-xs text-slate-500 dark:text-slate-400",
          Link { to: "/ai", class: "hover:text-teal-600", "AI Hub" }
          span { "/" }
          Link { to: "/ai/ml", class: "hover:text-teal-600", {lang.pick("传统机器学习图鉴", "Classic ML Atlas")} }
          span { "/" }
          span { class: "font-mono text-slate-700 dark:text-slate-200", "{detail.meta.slug}" }
        }
        div { class: "flex flex-wrap items-center gap-2",
          span { class: badge_class(BadgeVariant::Secondary, "text-[11px] font-bold"),
            "{detail.meta.family.label(lang)}"
          }
          if !detail.meta.rust_crates.is_empty() {
            span { class: "px-2 py-0.5 rounded-md bg-indigo-500/15 text-indigo-700 dark:text-indigo-300 text-[11px] font-mono font-bold",
              "Rust: {detail.meta.rust_crates}"
            }
          }
        }
        h1 { class: "text-2xl sm:text-3xl font-extrabold tracking-tight text-slate-900 dark:text-white",
          "{title}"
        }
        p { class: "text-sm sm:text-base text-slate-600 dark:text-slate-300 max-w-4xl leading-relaxed",
          "{desc}"
        }
      }

      // 4 格工程元数据速览条（单模板循环收敛 WASM 体积）
      div { class: "grid grid-cols-2 lg:grid-cols-4 gap-3 text-xs",
        for (lbl, val, val_cls) in [
          (lang.pick("训练时间复杂度", "Training Complexity"), detail.meta.time_complexity.as_str(), "font-mono text-slate-900 dark:text-white"),
          (lang.pick("空间 / 推理复杂度", "Space / Inference"), detail.meta.space_complexity.as_str(), "font-mono text-slate-900 dark:text-white"),
          (lang.pick("特征缩放敏感性", "Feature Scaling"), scale_txt, "text-teal-700 dark:text-teal-300"),
          (lang.pick("可解释性评级", "Interpretability"), interp_label(&detail.meta.interpretability, lang), "text-indigo-700 dark:text-indigo-300"),
        ] {
          div { key: "{lbl}", class: card_class("p-3.5"),
            div { class: "text-[11px] text-slate-400", "{lbl}" }
            div { class: "mt-1 font-bold {val_cls}", "{val}" }
          }
        }
      }

      // 顶部 2D 纯 Rust WASM 交互实验台
      MlLabWorkbench { kind: lab_kind, initial_mode: detail.meta.lab_mode.clone() }

      // 正文 + 右侧大纲与关联导航
      div { class: "grid grid-cols-1 lg:grid-cols-[minmax(0,1fr)_260px] gap-8 items-start",
        article { class: "min-w-0 prose prose-slate dark:prose-invert max-w-none",
          Markdown {
            content: detail.body_md.clone(),
            blog_id: format!("ai-ml-{}", detail.meta.slug),
          }
        }
        aside { class: "space-y-4 lg:sticky lg:top-20",
          if !toc.is_empty() {
            OnThisPage {
              entries: toc,
              title: lang.pick("算法目录", "Outline").to_string(),
            }
          }
          if !detail.siblings.is_empty() {
            div { class: card_class("p-4 space-y-2"),
              h3 { class: "text-xs font-extrabold uppercase tracking-wider text-slate-500",
                {lang.pick("同族算法切换", "Same-Family Algorithms")}
              }
              div { class: "flex flex-col gap-1.5 text-xs",
                for s in detail.siblings.iter() {
                  {
                    let is_cur = s.slug == detail.meta.slug;
                    let cls = if is_cur {
                      "font-bold text-teal-600 dark:text-teal-400"
                    } else {
                      "text-slate-600 dark:text-slate-300 hover:text-teal-600"
                    };
                    let lbl = match lang {
                      Language::En if !s.title_en.trim().is_empty() => s.title_en.as_str(),
                      _ => s.title.as_str(),
                    };
                    rsx! {
                      Link { key: "{s.slug}", to: format!("/ai/ml/{}", s.slug), class: "{cls}", "{lbl}" }
                    }
                  }
                }
              }
            }
          }
          if !detail.related_models.is_empty() || !detail.related_math.is_empty() {
            {
              let mut links: Vec<(String, &'static str)> = Vec::new();
              for m in &detail.related_models {
                links.push((join_path("/models/", m), "bg-indigo-500/15 text-indigo-700 dark:text-indigo-300"));
              }
              for t in &detail.related_math {
                links.push((join_path("/ai/math/", t), "bg-violet-500/15 text-violet-700 dark:text-violet-300"));
              }
              rsx! {
                div { class: card_class("p-4 space-y-2.5 text-xs"),
                  h3 { class: "text-xs font-extrabold uppercase tracking-wider text-slate-500",
                    {lang.pick("跨频道深度关联", "Cross-Channel Links")}
                  }
                  div { class: "flex flex-wrap gap-1.5",
                    for (href, cls) in links {
                      Link {
                        key: "{href}",
                        to: href.clone(),
                        class: "px-2 py-0.5 rounded font-mono text-[11px] hover:underline {cls}",
                        "{href}"
                      }
                    }
                  }
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
  fn builtin_ai_ml_entries_parse_and_cover_starter_labs() {
    let entries = load_all_ml_entries();
    assert!(
      entries.len() >= 2,
      "expected >= 2 builtin starter ai-ml entries, got {}",
      entries.len()
    );
    assert!(entries.iter().any(|e| e.meta.slug == "decision-tree"));
    assert!(entries.iter().any(|e| e.meta.slug == "svm"));
    for e in entries.iter() {
      assert!(!e.meta.title.is_empty());
      assert!(!e.meta.time_complexity.is_empty());
      assert!(!e.body_md.is_empty());
    }
  }
}
