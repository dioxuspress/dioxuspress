//! LLM 模型架构可视化交互工作台：同时支撑 `/ai/visualizer` 独立页面与 `<ModelArch />` MDX 嵌入组件。

use std::collections::HashMap;

use app_core::i18n::Language;
use dioxus::prelude::*;
use dioxus_shadcn::{
  badge_class, card_class, BadgeVariant, Button, ButtonSize, ButtonVariant, Input, Spinner,
  SpinnerSize, Textarea,
};
use widgets::PageMeta;

use crate::arch::{
  format_bytes_iec, format_count, load_builtin_preset, recommend_gpu_tier, InspectorFocus,
  ModelBlueprint, OutputHeadSpec, PrecisionPreset, BUILTIN_PRESETS,
};
#[cfg(feature = "accounts")]
use crate::atlas::{current_user_is_admin, SaveToAtlasModal};
#[cfg(feature = "accounts")]
use crate::atlas_store::{draft_from_blueprint, CuratedModelUpsertInput};
use crate::config_guide::{
  extract_annotated_fields_with_docs, get_config_guide_json_cached, get_or_parse_config_guide,
  group_annotated_fields, render_highlighted_json, render_inline_markup, AnnotatedConfigField,
  ConfigFieldDocDto, ConfigFileKind, FieldGroup,
};
use crate::diagram::{MacroArchitectureFlow, MicroSubgraphDiagram};
use crate::safetensors_meta::TensorPatternEntry;
use crate::server::{
  fetch_model_blueprint, normalize_repo_request, parse_custom_model_config, ModelHubSource,
};
use crate::tensor_graph::TensorFlowGraphView;

fn current_lang() -> Language {
  try_consume_context::<Signal<Language>>().map(|s| s()).unwrap_or_default()
}

/// `/ai/visualizer` 独立全屏交互工具页。
///
/// SSR 首屏渲染默认旗舰快照（`deepseek-v3`），水合后自动检测 URL 查询参数
/// `?preset=...` 或 `?repo=...&source=...&rev=...` 并按需触发远端实时拉取，确保 SSR/水合 100% 一致。
#[component]
pub fn AiVisualizerPage() -> Element {
  let lang = current_lang();
  let title =
    lang.pick("多模态模型架构可视化器", "Multimodal Model Architecture Visualizer").to_string();
  let desc = lang
    .pick(
      "通过 HuggingFace 或 ModelScope 的 config.json / model_index.json 与 Safetensors 元数据，精确解析并可视化 LLM、VLM、ASR/TTS 语音、图像/视频扩散 (DiT/UNet)、线性注意力 SSM 及 Embedding 架构拓扑与推理显存。",
      "Visualize LLM, VLM, ASR/TTS Audio, Image/Video Diffusion (MMDiT/3D DiT/UNet), Hybrid Linear/SSM, and Embedding architectures, Safetensors weights, and VRAM from HuggingFace or ModelScope.",
    )
    .to_string();

  let default_bp = load_builtin_preset("deepseek-v3");
  let mut blueprint = use_signal(|| default_bp);
  let mut repo_input = use_signal(|| "deepseek-ai/DeepSeek-V3".to_string());
  let mut source_sel = use_signal(|| ModelHubSource::Auto);
  let mut rev_input = use_signal(String::new);
  let mut loading = use_signal(|| false);
  let mut error_msg: Signal<Option<String>> = use_signal(|| None);
  let mut show_json_drawer = use_signal(|| false);
  let mut raw_cfg_input = use_signal(String::new);
  let mut raw_idx_input = use_signal(String::new);
  #[cfg(feature = "accounts")]
  let mut save_modal_draft: Signal<Option<CuratedModelUpsertInput>> = use_signal(|| None);
  #[cfg(feature = "accounts")]
  let mut saved_slug_banner: Signal<Option<String>> = use_signal(|| None);

  #[cfg(feature = "accounts")]
  let admin_save_btn = if current_user_is_admin() {
    if let Some(bp_cur) = blueprint() {
      rsx! {
        Button {
          size: ButtonSize::Sm,
          onclick: move |_| {
            let mut draft = draft_from_blueprint(&bp_cur);
            let r_trim = repo_input().trim().to_string();
            if draft.hf_repo.starts_with("custom/") && r_trim.contains('/') {
              draft.hf_repo = r_trim;
            }
            save_modal_draft.set(Some(draft));
          },
          {lang.pick("＋ 收录到模型图鉴", "+ Save to Model Atlas")}
        }
      }
    } else {
      rsx! {}
    }
  } else {
    rsx! {}
  };
  #[cfg(not(feature = "accounts"))]
  let admin_save_btn = rsx! {};

  #[cfg(feature = "accounts")]
  let admin_saved_banner = if let Some(saved_slug) = saved_slug_banner() {
    rsx! {
      div { class: "mt-3 rounded-lg border border-emerald-200 dark:border-emerald-900/70 bg-emerald-50 dark:bg-emerald-950/40 px-3.5 py-2.5 text-xs text-emerald-800 dark:text-emerald-300 flex flex-wrap items-center justify-between gap-2",
        span {
          {lang.pick("已成功保存并发布至模型架构图鉴：", "Published to Model Atlas: ")}
          span { class: "font-mono font-bold", "/models/{saved_slug}" }
        }
        Link {
          to: format!("/models/{saved_slug}"),
          class: "font-bold underline hover:opacity-80",
          {lang.pick("前往查看图鉴详情页 →", "Open Curated Page →")}
        }
      }
    }
  } else {
    rsx! {}
  };
  #[cfg(not(feature = "accounts"))]
  let admin_saved_banner = rsx! {};

  #[cfg(feature = "accounts")]
  let admin_modal_el = if let Some(draft) = save_modal_draft() {
    rsx! {
      SaveToAtlasModal {
        initial_draft: Some(draft),
        on_close: move |_| save_modal_draft.set(None),
        on_saved: move |slug: String| {
          save_modal_draft.set(None);
          saved_slug_banner.set(Some(slug));
        },
      }
    }
  } else {
    rsx! {}
  };
  #[cfg(not(feature = "accounts"))]
  let admin_modal_el = rsx! {};

  // 客户端水合后解析 URL 查询参数（例如从文章嵌入卡片点击「打开完整工作台 ↗」跳转过来）
  use_effect(move || {
    if let Some(p) = widgets::browser::query_param("preset") {
      if let Some(bp) = load_builtin_preset(&p) {
        repo_input.set(bp.repo_id.clone());
        blueprint.set(Some(bp));
        return;
      }
    }
    if let Some(r) = widgets::browser::query_param("repo") {
      let src_q = widgets::browser::query_param("source").unwrap_or_else(|| "auto".to_string());
      let rev_q = widgets::browser::query_param("rev").unwrap_or_default();
      repo_input.set(r.clone());
      source_sel.set(ModelHubSource::parse_str(&src_q));
      rev_input.set(rev_q.clone());

      if rev_q.is_empty() && src_q == "preset" {
        if let Some(bp) = load_builtin_preset(&r) {
          blueprint.set(Some(bp));
          return;
        }
      }

      loading.set(true);
      error_msg.set(None);
      spawn(async move {
        match fetch_model_blueprint(r, src_q, rev_q).await {
          Ok(bp) => blueprint.set(Some(bp)),
          Err(e) => error_msg.set(Some(clean_server_err(&e.to_string()))),
        }
        loading.set(false);
      });
    }
  });

  let mut trigger_live_fetch = move || {
    let raw_r = repo_input();
    let src = source_sel().as_str().to_string();
    let rev = rev_input();
    if let Err(msg) = normalize_repo_request(&raw_r, &src, &rev) {
      error_msg.set(Some(msg));
      return;
    }
    loading.set(true);
    error_msg.set(None);
    spawn(async move {
      match fetch_model_blueprint(raw_r, src, rev).await {
        Ok(bp) => blueprint.set(Some(bp)),
        Err(e) => error_msg.set(Some(clean_server_err(&e.to_string()))),
      }
      loading.set(false);
    });
  };

  let mut parse_custom_json = move || {
    let cfg_str = raw_cfg_input();
    if cfg_str.trim().is_empty() {
      error_msg.set(Some("请先粘贴 config.json 内容".to_string()));
      return;
    }
    let idx_str = raw_idx_input();
    loading.set(true);
    error_msg.set(None);
    spawn(async move {
      match parse_custom_model_config(cfg_str, idx_str).await {
        Ok(bp) => {
          error_msg.set(None);
          blueprint.set(Some(bp));
          show_json_drawer.set(false);
        }
        Err(e) => error_msg.set(Some(clean_server_err(&e.to_string()))),
      }
      loading.set(false);
    });
  };

  rsx! {
    PageMeta {
      title: title.clone(),
      description: desc.clone(),
      path: "/ai/visualizer".to_string(),
    }

    section { class: "py-10 bg-white dark:bg-slate-950 min-h-screen",
      div { class: "max-w-7xl mx-auto px-4 sm:px-6",
        // 页面头部
        div { class: "mb-6 flex flex-col lg:flex-row lg:items-end lg:justify-between gap-4",
          div {
            div { class: "inline-flex flex-wrap items-center gap-2 text-xs font-semibold text-indigo-600 dark:text-indigo-400 mb-2",
              Link { to: "/models", class: "hover:underline",
                {lang.pick("← 模型架构图鉴", "← Model Atlas")}
              }
              span { class: "text-slate-300 dark:text-slate-700", "·" }
              Link { to: "/ai", class: "hover:underline text-slate-500 dark:text-slate-400",
                {lang.pick("AI 工程板块", "Rust AI")}
              }
              span { class: "text-slate-300 dark:text-slate-700", "/" }
              span { "L1 Config + L2 Safetensors Live Sandbox" }
            }
            h1 { class: "text-2xl sm:text-3xl font-extrabold tracking-tight text-slate-900 dark:text-white",
              "{title}"
            }
            p { class: "mt-1.5 text-sm text-slate-500 dark:text-slate-400 max-w-3xl",
              "{desc}"
            }
          }

          div { class: "flex flex-wrap items-center gap-2",
            {admin_save_btn}
            Button {
              variant: ButtonVariant::Outline,
              size: ButtonSize::Sm,
              onclick: move |_| show_json_drawer.set(!show_json_drawer()),
              if show_json_drawer() {
                {lang.pick("收起 JSON 粘贴板", "Hide JSON Input")}
              } else {
                {lang.pick("粘贴本地 config.json", "Paste config.json")}
              }
            }
          }
        }

        // 顶部控制台：快速预设 + HuggingFace / ModelScope 实时解析栏
        div { class: card_class("p-4 mb-6 bg-slate-50/70 dark:bg-slate-900/60"),
          // 1) 一键预设基准模型
          div { class: "flex flex-wrap items-center gap-2 mb-3.5",
            span { class: "text-xs font-semibold text-slate-500 dark:text-slate-400 mr-1",
              {lang.pick("经典架构基准：", "Flagship Presets:")}
            }
            for p in BUILTIN_PRESETS.iter() {
              {
                let pid = p.id;
                let prepo = p.repo_id;
                let ptitle = p.title;
                let ptag = p.tag;
                let is_cur = blueprint()
                  .as_ref()
                  .is_some_and(|b| b.repo_id.eq_ignore_ascii_case(prepo));
                let cls = if is_cur {
                  "px-3 py-1.5 rounded-lg text-xs font-semibold bg-indigo-600 text-white shadow-xs transition-all cursor-pointer"
                } else {
                  "px-3 py-1.5 rounded-lg text-xs font-medium bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-200 hover:border-indigo-400 cursor-pointer"
                };
                rsx! {
                  button {
                    key: "{pid}",
                    r#type: "button",
                    class: cls,
                    onclick: move |_| {
                      if let Some(bp) = load_builtin_preset(pid) {
                        repo_input.set(prepo.to_string());
                        error_msg.set(None);
                        blueprint.set(Some(bp));
                      }
                    },
                    span { "{ptitle}" }
                    span { class: if is_cur { "ml-1.5 text-[10px] text-indigo-100 font-mono" } else { "ml-1.5 text-[10px] text-slate-400 font-mono" },
                      "· {ptag}"
                    }
                  }
                }
              }
            }
          }

          // 2) 在线仓库拉取输入行
          div { class: "flex flex-col sm:flex-row items-stretch sm:items-center gap-2",
            // 托管源选择按钮组
            div { class: "inline-flex rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-0.5 shrink-0",
              for (src_opt, label) in [
                (ModelHubSource::Auto, "Auto"),
                (ModelHubSource::HuggingFace, "HuggingFace"),
                (ModelHubSource::ModelScope, "ModelScope"),
              ] {
                {
                  let active = source_sel() == src_opt;
                  let btn_c = if active {
                    "px-2.5 py-1 rounded-md text-xs font-semibold bg-slate-900 text-white dark:bg-slate-100 dark:text-slate-900 cursor-pointer"
                  } else {
                    "px-2.5 py-1 rounded-md text-xs font-medium text-slate-600 dark:text-slate-300 hover:text-slate-900 dark:hover:text-white cursor-pointer"
                  };
                  rsx! {
                    button {
                      key: "{label}",
                      r#type: "button",
                      class: btn_c,
                      onclick: move |_| source_sel.set(src_opt),
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
                  trigger_live_fetch();
                }
              },
              Input {
                class: "font-mono text-sm bg-white dark:bg-slate-950",
                placeholder: "输入 owner/model 或粘贴 HF / ModelScope 链接（如 Qwen/Qwen2.5-72B-Instruct）",
                "aria-label": "Model repository ID or URL",
                value: repo_input(),
                on_value_change: move |v: String| repo_input.set(v),
              }
            }

            div { class: "w-full sm:w-32 shrink-0",
              Input {
                class: "font-mono text-xs bg-white dark:bg-slate-950",
                placeholder: "分支 (默认)",
                "aria-label": "Git revision or branch",
                value: rev_input(),
                on_value_change: move |v: String| rev_input.set(v),
              }
            }

            Button {
              class: "shrink-0 whitespace-nowrap",
              loading: loading(),
              onclick: move |_| trigger_live_fetch(),
              {lang.pick("实时拉取架构", "Inspect Live")}
            }
          }

          // 3) 可选本地 JSON 粘贴抽屉
          if show_json_drawer() {
            div { class: "mt-4 pt-4 border-t border-slate-200 dark:border-slate-800 grid grid-cols-1 lg:grid-cols-2 gap-3",
              div {
                label { class: "block text-xs font-semibold text-slate-700 dark:text-slate-300 mb-1",
                  "1. config.json 或 model_index.json（必填，纯前端 WASM 即时解析）"
                }
                Textarea {
                  class: "h-36 font-mono text-xs bg-white dark:bg-slate-950",
                  placeholder: "{{\n  \"architectures\": [\"Qwen3ForCausalLM\"],\n  \"hidden_size\": 4096\n}}",
                  value: raw_cfg_input(),
                  on_value_change: move |v: String| raw_cfg_input.set(v),
                }
              }
              div {
                label { class: "block text-xs font-semibold text-slate-700 dark:text-slate-300 mb-1",
                  "2. model.safetensors.index.json（可选，补充真实分片与权重映射）"
                }
                Textarea {
                  class: "h-36 font-mono text-xs bg-white dark:bg-slate-950",
                  placeholder: "{{\n  \"metadata\": {{ \"total_size\": 16380000000 }},\n  \"weight_map\": {{}}\n}}",
                  value: raw_idx_input(),
                  on_value_change: move |v: String| raw_idx_input.set(v),
                }
              }
              div { class: "lg:col-span-2 flex justify-end gap-2",
                Button {
                  size: ButtonSize::Sm,
                  loading: loading(),
                  onclick: move |_| parse_custom_json(),
                  {lang.pick("解析粘贴的 JSON", "Render Pasted JSON")}
                }
              }
            }
          }

          if let Some(err) = error_msg() {
            div { class: "mt-3 rounded-lg border border-rose-200 dark:border-rose-900/70 bg-rose-50 dark:bg-rose-950/40 px-3.5 py-2.5 text-xs text-rose-700 dark:text-rose-300",
              "{err}"
            }
          }
          {admin_saved_banner}
        }

        {admin_modal_el}

        // 主可视化工作台
        if let Some(bp) = blueprint() {
          ModelVisualizerWorkbench {
            key: "{bp.repo_id}-{bp.source_label}",
            blueprint: bp,
            initial_focus: None,
            embedded: false,
          }
        }
      }
    }
  }
}

/// MDX `<ModelArch ... />` 嵌入组件属性渲染器。
///
/// 支持用法：
/// - `<ModelArch preset="deepseek-v3" focus="attention" />`
/// - `<ModelArch repo="Qwen/Qwen3-8B" source="hf" />`
/// - `<ModelArch repo="deepseek-ai/DeepSeek-V3" source="modelscope" focus="moe" />`
pub struct ModelArchComponent;

impl widgets::MdxComponent for ModelArchComponent {
  fn name(&self) -> &'static str {
    "ModelArch"
  }

  fn render(&self, attrs: &HashMap<String, String>) -> Element {
    let preset = attrs.get("preset").cloned().unwrap_or_default();
    let repo = attrs.get("repo").cloned().unwrap_or_default();
    let source = attrs.get("source").cloned().unwrap_or_else(|| "auto".to_string());
    let revision = attrs.get("revision").cloned().unwrap_or_default();
    let focus = attrs.get("focus").map(|f| InspectorFocus::from_attr(f));

    if preset.trim().is_empty() && repo.trim().is_empty() {
      return rsx! {
        div { class: "not-prose my-8 rounded-xl border border-amber-200 dark:border-amber-800 bg-amber-50 dark:bg-amber-950/40 p-4 text-sm text-amber-800 dark:text-amber-300",
          "ModelArch 组件需指定 preset（如 deepseek-v3）或 repo（如 Qwen/Qwen3-8B）。"
        }
      };
    }

    rsx! {
      ModelArchEmbed { preset, repo, source, revision, focus }
    }
  }
}

#[component]
fn ModelArchEmbed(
  preset: String,
  repo: String,
  source: String,
  revision: String,
  focus: Option<InspectorFocus>,
) -> Element {
  let initial_bp = if !preset.trim().is_empty() {
    load_builtin_preset(&preset)
  } else if revision.trim().is_empty() {
    load_builtin_preset(&repo)
  } else {
    None
  };

  let needs_remote = initial_bp.is_none() && !repo.trim().is_empty();
  let mut bp_sig = use_signal(|| initial_bp);
  let mut err_sig: Signal<Option<String>> = use_signal(|| None);
  let mut loading_sig = use_signal(|| needs_remote);

  let repo_for_eff = repo.clone();
  let src_for_eff = source.clone();
  let rev_for_eff = revision.clone();
  use_effect(move || {
    if bp_sig().is_some() || repo_for_eff.trim().is_empty() {
      return;
    }
    let r = repo_for_eff.clone();
    let s = src_for_eff.clone();
    let rv = rev_for_eff.clone();
    loading_sig.set(true);
    spawn(async move {
      match fetch_model_blueprint(r, s, rv).await {
        Ok(b) => bp_sig.set(Some(b)),
        Err(e) => err_sig.set(Some(clean_server_err(&e.to_string()))),
      }
      loading_sig.set(false);
    });
  });

  rsx! {
    div { class: "not-prose my-8",
      if let Some(bp) = bp_sig() {
        ModelVisualizerWorkbench {
          blueprint: bp,
          initial_focus: focus,
          embedded: true,
        }
      } else if loading_sig() {
        div { class: card_class("p-8 flex items-center justify-center gap-3 text-sm text-slate-500 dark:text-slate-400"),
          Spinner { size: SpinnerSize::Md, class: "border-t-primary" }
          span { "正在从 {source} 解析 `{repo}` 架构配置与 Safetensors 元数据..." }
        }
      } else if let Some(err) = err_sig() {
        div { class: "rounded-xl border border-rose-200 dark:border-rose-900 bg-rose-50 dark:bg-rose-950/40 p-4 text-sm text-rose-700 dark:text-rose-300",
          "{err}"
        }
      }
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 共享核心可视化工作台（页面模式 + MDX 嵌入模式共用）
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DetailTab {
  VramSimulator,
  SafetensorsTree,
  HyperparamsAndJson,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WorkbenchViewMode {
  LayeredWorkbench,
  TensorFlowDag,
}

#[component]
pub fn ModelVisualizerWorkbench(
  blueprint: ModelBlueprint,
  initial_focus: Option<InspectorFocus>,
  embedded: bool,
  #[props(default = false)] hide_identity_header: bool,
) -> Element {
  let default_layer = blueprint.segments.last().map(|s| s.start_layer).unwrap_or(0);
  let mut selected_layer = use_signal(|| default_layer);
  let mut focus = use_signal(|| initial_focus.unwrap_or_else(|| blueprint.default_focus()));
  let mut view_mode = use_signal(|| WorkbenchViewMode::LayeredWorkbench);
  let mut detail_tab = use_signal(|| DetailTab::VramSimulator);
  let initial_compact = initial_focus.is_none();

  // VRAM 模拟器交互状态
  let default_prec = if blueprint.quantization_summary.as_deref().is_some_and(|q| q.contains("FP8"))
  {
    PrecisionPreset::Fp8
  } else {
    PrecisionPreset::Bf16
  };
  let mut precision = use_signal(|| default_prec);
  let mut ctx_len = use_signal(|| 32_768u64.min(blueprint.max_position_embeddings.max(4096)));
  let mut batch_size = use_signal(|| 1u64);
  let mut weight_stage_filter = use_signal(|| "all".to_string());

  let active_pct = if blueprint.params.total_params > 0 {
    (blueprint.params.active_params as f64) * 100.0 / (blueprint.params.total_params as f64)
  } else {
    100.0
  };
  let is_sparse = blueprint.params.active_params < blueprint.params.total_params;

  let cur_segment = blueprint.segment_for_layer(selected_layer()).cloned();
  let kv_ratio = cur_segment.as_ref().map(|s| s.attention.kv_compression_ratio()).unwrap_or(1.0);

  let open_href = format!("/ai/visualizer?repo={}", blueprint.repo_id);

  rsx! {
    div { class: if embedded {
      card_class("p-4 sm:p-6 bg-white dark:bg-slate-950 shadow-sm")
    } else {
      "space-y-5".to_string()
    },
      // 1. 顶部模型身份栏 + 核心工程指标卡 + 参数分布进度条
      div { class: if embedded { "pb-4 mb-5 border-b border-slate-200 dark:border-slate-800" } else { card_class("p-4 sm:p-5 bg-white dark:bg-slate-900/80") },
        div { class: "flex flex-wrap items-start justify-between gap-3",
          div {
            if !hide_identity_header {
              div { class: "flex flex-wrap items-center gap-2",
                h2 { class: "text-lg sm:text-xl font-extrabold text-slate-900 dark:text-white font-mono",
                  "{blueprint.repo_id}"
                }
                span { class: badge_class(BadgeVariant::Default, "text-xs"),
                  "{blueprint.family.badge_label()}"
                }
                span { class: badge_class(BadgeVariant::Secondary, "text-xs font-mono"),
                  "{blueprint.torch_dtype}"
                }
                if let Some(ref q) = blueprint.quantization_summary {
                  span { class: "px-2 py-0.5 rounded-md bg-amber-500/15 text-amber-700 dark:text-amber-300 text-xs font-mono font-semibold",
                    "{q}"
                  }
                }
              }
            } else {
              div { class: "flex flex-wrap items-center gap-2",
                span { class: badge_class(BadgeVariant::Default, "text-xs"),
                  "{blueprint.family.badge_label()}"
                }
                span { class: badge_class(BadgeVariant::Secondary, "text-xs font-mono"),
                  "{blueprint.torch_dtype}"
                }
                if let Some(ref q) = blueprint.quantization_summary {
                  span { class: "px-2 py-0.5 rounded-md bg-amber-500/15 text-amber-700 dark:text-amber-300 text-xs font-mono font-semibold",
                    "{q}"
                  }
                }
                span { class: "text-xs text-slate-500 dark:text-slate-400 font-mono",
                  "{blueprint.repo_id}"
                }
              }
            }
            div { class: "mt-1 text-xs text-slate-500 dark:text-slate-400 flex flex-wrap items-center gap-x-3 gap-y-1 font-mono",
              span { "Source: {blueprint.source_label}" }
              if let Some(arch0) = blueprint.architectures.first() {
                span { "· Class: {arch0}" }
              }
              if let Some(ref st) = blueprint.safetensors {
                span { "· L2: {st.source_kind}" }
              }
            }
            if let Some(ref mm) = blueprint.multimodal_note {
              div { class: "mt-2 text-xs text-indigo-600 dark:text-indigo-400 font-medium",
                "ℹ {mm}"
              }
            }
          }

          if embedded {
            Link {
              to: open_href,
              class: "text-xs font-semibold text-indigo-600 dark:text-indigo-400 hover:underline shrink-0",
              "打开完整架构工作台 ↗"
            }
          }
        }

        // 4 格核心工程指标卡
        div { class: "mt-3.5 grid grid-cols-2 lg:grid-cols-4 gap-3",
          div { class: "rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/70 dark:bg-slate-950/60 p-3",
            div { class: "text-xs font-medium text-slate-500 dark:text-slate-400",
              if blueprint.is_diffusion_pipeline() {
                "全管线总参 / 单步去噪激活"
              } else {
                "总参数量 / 单 Token 激活"
              }
            }
            div { class: "mt-1 text-base sm:text-lg font-extrabold text-slate-900 dark:text-white font-mono tabular-nums",
              if is_sparse {
                "{format_count(blueprint.params.total_params)} · "
                span { class: "text-amber-600 dark:text-amber-400",
                  "{format_count(blueprint.params.active_params)} ({active_pct:.1}%)"
                }
              } else {
                "{format_count(blueprint.params.total_params)} (100% Dense)"
              }
            }
          }

          div { class: "rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/70 dark:bg-slate-950/60 p-3",
            div { class: "text-xs font-medium text-slate-500 dark:text-slate-400",
              "主干维度 (Layers × Hidden)"
            }
            div { class: "mt-1 text-base sm:text-lg font-extrabold text-slate-900 dark:text-white font-mono tabular-nums",
              "{blueprint.num_hidden_layers}L × {blueprint.hidden_size}d"
            }
            div { class: "text-xs text-slate-500 dark:text-slate-400 font-mono truncate",
              if let Some(ref diff) = blueprint.diffusion {
                "{diff.components.len()} 组件 · {diff.latent_channels}ch Latent"
              } else {
                "Vocab: {blueprint.vocab_size}"
              }
            }
          }

          div { class: "rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/70 dark:bg-slate-950/60 p-3",
            div { class: "text-xs font-medium text-slate-500 dark:text-slate-400",
              if blueprint.is_diffusion_pipeline() {
                "去噪采样与缓存"
              } else if blueprint.is_single_pass_encoder() {
                "推理状态缓存"
              } else if blueprint.has_linear_recurrent() {
                "循环状态 / KV 效率"
              } else {
                "KV Cache 压缩效率"
              }
            }
            div { class: "mt-1 text-base sm:text-lg font-extrabold text-indigo-600 dark:text-indigo-400 font-mono tabular-nums",
              if blueprint.is_diffusion_pipeline() || blueprint.is_single_pass_encoder() {
                "0 B KV Cache"
              } else if kv_ratio >= 500.0 {
                "O(1) State"
              } else {
                "{kv_ratio:.1}× vs MHA"
              }
            }
            div { class: "text-xs text-slate-500 dark:text-slate-400 font-mono truncate",
              if let Some(ref diff) = blueprint.diffusion {
                "{diff.scheduler_summary}"
              } else if blueprint.is_single_pass_encoder() {
                "单次并行编码 (无自回归缓存)"
              } else if blueprint.has_linear_recurrent() {
                "循环状态随序列长度 S 零增长"
              } else if let Some(ref seg) = cur_segment {
                "{seg.attention.cached_elements_per_token()} elems/tok/layer"
              }
            }
          }

          div { class: "rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/70 dark:bg-slate-950/60 p-3",
            div { class: "text-xs font-medium text-slate-500 dark:text-slate-400",
              if blueprint.is_diffusion_pipeline() {
                "潜空间序列 & 位置编码"
              } else {
                "上下文窗口 & 位置编码"
              }
            }
            div { class: "mt-1 text-base sm:text-lg font-extrabold text-slate-900 dark:text-white font-mono tabular-nums",
              if blueprint.is_diffusion_pipeline() {
                "{blueprint.max_position_embeddings / 1024}K latents"
              } else {
                "{blueprint.max_position_embeddings / 1024}K ctx"
              }
            }
            div { class: "text-xs text-slate-500 dark:text-slate-400 font-mono truncate",
              if let Some(ref diff) = blueprint.diffusion {
                "{diff.patch_summary} · {blueprint.rope_summary}"
              } else {
                "{blueprint.rope_summary}"
              }
            }
          }
        }

        // 参数占比彩色堆叠条
        ParamStackedBar {
          params: blueprint.params.clone(),
          is_diffusion: blueprint.is_diffusion_pipeline(),
        }
      }

      // 1.5 顶层视图模式切换：分层联动工作台 vs 全网张量维流图（色块 + 维度流 DAG）
      div { class: "flex flex-wrap items-center justify-between gap-2.5 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/70 dark:bg-slate-900/50 px-3 py-2",
        div { class: "inline-flex items-center gap-1.5",
          button {
            r#type: "button",
            class: if view_mode() == WorkbenchViewMode::LayeredWorkbench {
              "px-3 py-1.5 rounded-lg text-xs font-bold bg-slate-900 text-white dark:bg-white dark:text-slate-900 shadow-2xs cursor-pointer"
            } else {
              "px-3 py-1.5 rounded-lg text-xs font-semibold text-slate-600 dark:text-slate-300 hover:bg-white dark:hover:bg-slate-800 cursor-pointer"
            },
            onclick: move |_| view_mode.set(WorkbenchViewMode::LayeredWorkbench),
            "分层联动工作台"
          }
          button {
            r#type: "button",
            class: if view_mode() == WorkbenchViewMode::TensorFlowDag {
              "inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-bold bg-indigo-600 text-white shadow-2xs cursor-pointer"
            } else {
              "inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold text-slate-700 dark:text-slate-200 hover:bg-white dark:hover:bg-slate-800 cursor-pointer"
            },
            onclick: move |_| view_mode.set(WorkbenchViewMode::TensorFlowDag),
            span { "全网张量维流图 (DAG)" }
            span { class: if view_mode() == WorkbenchViewMode::TensorFlowDag {
              "px-1.5 py-0.5 rounded bg-white/20 text-[10px] font-mono"
            } else {
              "px-1.5 py-0.5 rounded bg-indigo-100 dark:bg-indigo-950/70 text-indigo-700 dark:text-indigo-300 text-[10px] font-mono"
            },
              "色块维度流"
            }
          }
        }
        span { class: "text-xs text-slate-500 dark:text-slate-400",
          if view_mode() == WorkbenchViewMode::LayeredWorkbench {
            "默认展示端到端总体架构，点左侧层段/模块在右侧钻取算子子图、Safetensors 权重与显存精算。"
          } else {
            "网格蓝图 + 算子实色块 + 框内张量维度 + 右上角图例；支持切换 B×S 实数维度、对照 Inception-v3 / ResNet-50 及导出高清 SVG。"
          }
        }
      }

      if view_mode() == WorkbenchViewMode::TensorFlowDag {
        TensorFlowGraphView {
          blueprint: blueprint.clone(),
          on_drill_to_workbench: move |(layer_opt, target_focus)| {
            if let Some(l) = layer_opt {
              selected_layer.set(l);
            }
            focus.set(target_focus);
            view_mode.set(WorkbenchViewMode::LayeredWorkbench);
          },
        }
      } else {
      // 2. 双栏联动工作区：左「宏观分层拓扑流 (桌面吸顶)」+ 右「总体架构 / 算子级内部子图 & 工程检视面板」
      div { class: "grid grid-cols-1 lg:grid-cols-12 gap-5 items-start",
        // 左栏：宏观架构流
        div { class: format!("lg:col-span-5 lg:sticky lg:top-20 {}", card_class("p-4 bg-white dark:bg-slate-900/60")),
          div { class: "flex items-center justify-between mb-3",
            span { class: "text-xs font-bold uppercase tracking-wider text-slate-600 dark:text-slate-300",
              "Macro Forward Pass (点击任意模块钻取细节)"
            }
          }
          MacroArchitectureFlow {
            blueprint: blueprint.clone(),
            selected_layer: selected_layer(),
            focus: focus(),
            initial_compact,
            on_select_layer: move |l| selected_layer.set(l),
            on_select_focus: move |f| focus.set(f),
          }
        }

        // 右栏：总体架构总览 / 微观子图放大镜 + 真实张量核验 + 三合一深度面板
        div { class: "lg:col-span-7 space-y-5",
          // 2A. 总体架构 / 算子级内部子图卡片
          div { class: card_class("p-4 sm:p-5 bg-white dark:bg-slate-900/60"),
            div { class: "flex flex-wrap items-center justify-between gap-2.5 pb-3 mb-4 border-b border-slate-200 dark:border-slate-800",
              div { class: "min-w-0",
                if focus() == InspectorFocus::Overview {
                  span { class: "text-xs font-bold uppercase tracking-wider text-indigo-600 dark:text-indigo-400",
                    "Architecture Overview · 全模型端到端总体架构"
                  }
                } else if focus().is_layer_scoped() {
                  div { class: "flex flex-wrap items-center gap-1.5",
                    span { class: "text-xs font-bold uppercase tracking-wider text-indigo-600 dark:text-indigo-400",
                      "Operator Subgraph · Layer #{selected_layer()}"
                    }
                    if let Some(ref seg) = cur_segment {
                      span { class: "px-2 py-0.5 rounded-md bg-slate-100 dark:bg-slate-800 text-xs font-medium text-slate-700 dark:text-slate-300",
                        "{seg.range_label()} · {seg.title}"
                      }
                    }
                  }
                } else {
                  span { class: "text-xs font-bold uppercase tracking-wider text-indigo-600 dark:text-indigo-400",
                    "Operator Subgraph · Global Module · {focus_short_label(focus())}"
                  }
                }
              }
              // 快捷切换：先总体架构，后各细粒度算子模块
              div { class: "flex flex-wrap items-center gap-1",
                button {
                  r#type: "button",
                  class: if focus() == InspectorFocus::Overview {
                    "px-2.5 py-1 rounded-md text-xs font-bold bg-slate-900 text-white dark:bg-white dark:text-slate-900 shadow-2xs cursor-pointer"
                  } else {
                    "px-2.5 py-1 rounded-md text-xs font-semibold border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 text-slate-700 dark:text-slate-200 hover:border-indigo-400 cursor-pointer"
                  },
                  onclick: move |_| focus.set(InspectorFocus::Overview),
                  "总体架构"
                }
                span { class: "mx-0.5 text-slate-300 dark:text-slate-700 select-none", "|" }
                if blueprint.vision.is_some() {
                  button {
                    r#type: "button",
                    class: if focus() == InspectorFocus::Vision {
                      "px-2.5 py-1 rounded-md text-xs font-semibold bg-fuchsia-600 text-white cursor-pointer"
                    } else {
                      "px-2.5 py-1 rounded-md text-xs font-medium bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700 cursor-pointer"
                    },
                    onclick: move |_| focus.set(InspectorFocus::Vision),
                    "Vision"
                  }
                }
                if blueprint.audio.is_some() {
                  button {
                    r#type: "button",
                    class: if focus() == InspectorFocus::Audio {
                      "px-2.5 py-1 rounded-md text-xs font-semibold bg-cyan-600 text-white cursor-pointer"
                    } else {
                      "px-2.5 py-1 rounded-md text-xs font-medium bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700 cursor-pointer"
                    },
                    onclick: move |_| focus.set(InspectorFocus::Audio),
                    "Audio"
                  }
                }
                for (f_opt, f_label) in [
                  (InspectorFocus::Embedding, if blueprint.is_diffusion_pipeline() { "Conditioner" } else { "Embed" }),
                  (InspectorFocus::Attention, if blueprint.has_linear_recurrent() { "Attn / SSM" } else { "Attention" }),
                  (InspectorFocus::FfnOrMoe, if blueprint.has_moe() { "FFN / MoE" } else { "MLP / FFN" }),
                  (InspectorFocus::NormAndResidual, if blueprint.is_diffusion_pipeline() { "AdaLN" } else { "Norm" }),
                  (
                    InspectorFocus::LmHead,
                    match blueprint.output_head {
                      OutputHeadSpec::CausalLmHead => "LM Head",
                      OutputHeadSpec::EmbeddingPool { .. } => "Pooling",
                      OutputHeadSpec::SequenceScorer { .. } => "Scorer",
                      OutputHeadSpec::SpeechSynthesizer { .. } => "Vocoder",
                      OutputHeadSpec::LatentDecoder { .. } => "VAE Decoder",
                    },
                  ),
                ] {
                  {
                    let is_f = focus() == f_opt;
                    let c = if is_f {
                      "px-2.5 py-1 rounded-md text-xs font-semibold bg-indigo-600 text-white cursor-pointer"
                    } else {
                      "px-2.5 py-1 rounded-md text-xs font-medium bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700 cursor-pointer"
                    };
                    rsx! {
                      button {
                        key: "{f_label}",
                        r#type: "button",
                        class: c,
                        onclick: move |_| focus.set(f_opt),
                        "{f_label}"
                      }
                    }
                  }
                }
                if blueprint.num_nextn_predict_layers > 0 {
                  button {
                    r#type: "button",
                    class: if focus() == InspectorFocus::MtpHead {
                      "px-2.5 py-1 rounded-md text-xs font-semibold bg-purple-600 text-white cursor-pointer"
                    } else {
                      "px-2.5 py-1 rounded-md text-xs font-medium bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 cursor-pointer"
                    },
                    onclick: move |_| focus.set(InspectorFocus::MtpHead),
                    "MTP"
                  }
                }
              }
            }

            MicroSubgraphDiagram {
              blueprint: blueprint.clone(),
              selected_layer: selected_layer(),
              focus: focus(),
              on_select_layer: move |l| selected_layer.set(l),
              on_select_focus: move |f| focus.set(f),
            }

            // 跨层段共用同一 Attention 时的智能导引条：避免在异构模型（如 DeepSeek-V3）切段后误以为右侧无变化
            if focus() == InspectorFocus::Attention && blueprint.has_uniform_attention() {
              if let Some(ref seg) = cur_segment {
                div { class: "mt-3 rounded-lg border border-amber-200/80 dark:border-amber-900/60 bg-amber-50/70 dark:bg-amber-950/30 px-3 py-2 flex flex-wrap items-center justify-between gap-2 text-xs text-amber-900 dark:text-amber-200",
                  span {
                    "当前观测 "
                    strong { class: "font-mono", "{seg.range_label()}" }
                    "（{seg.title}）：全部层段共用 {seg.attention.short_label()} 注意力，本段异构特征位于 FFN 子层。"
                  }
                  button {
                    r#type: "button",
                    class: "shrink-0 px-2.5 py-1 rounded-md bg-amber-600 text-white font-semibold hover:bg-amber-700 cursor-pointer transition-colors",
                    onclick: move |_| focus.set(InspectorFocus::FfnOrMoe),
                    "查看本段 FFN / MoE →"
                  }
                }
              }
            }

            // 上下文关联算子动态原理实验室直通车（与 /ai/components/:slug 双向咬合）
            {
              let lang_here = current_lang();
              let comp_links =
                crate::components_atlas::recommended_component_links(&blueprint, focus(), lang_here);
              let math_links =
                crate::math_atlas::recommended_math_links(&blueprint, focus(), lang_here);
              if !comp_links.is_empty() || !math_links.is_empty() {
                rsx! {
                  div { class: "mt-3.5 pt-3 border-t border-slate-100 dark:border-slate-800/80 space-y-2 text-[13px]",
                    if !comp_links.is_empty() {
                      div { class: "flex flex-wrap items-center gap-2",
                        span { class: "text-xs font-bold text-emerald-700 dark:text-emerald-400 shrink-0",
                          {lang_here.pick("🔬 深入算子动态原理：", "🔬 Interactive Operator Lab:")}
                        }
                        for (href, badge_txt, link_title) in comp_links {
                          Link {
                            key: "{href}-{badge_txt}",
                            to: href,
                            class: "inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg border border-emerald-200 dark:border-emerald-900/70 bg-emerald-50/70 dark:bg-emerald-950/30 text-slate-800 dark:text-slate-200 hover:border-emerald-500 transition-colors",
                            span { class: "px-1.5 py-0.5 rounded bg-emerald-600 text-white text-[11px] font-mono font-bold",
                              "{badge_txt}"
                            }
                            span { class: "font-medium", "{link_title} →" }
                          }
                        }
                      }
                    }
                    if !math_links.is_empty() {
                      div { class: "flex flex-wrap items-center gap-2",
                        span { class: "text-xs font-bold text-violet-700 dark:text-violet-400 shrink-0",
                          {lang_here.pick("📐 底层数学定理推导：", "📐 Mathematical Proof Lab:")}
                        }
                        for (href, badge_txt, link_title) in math_links {
                          Link {
                            key: "{href}-{badge_txt}",
                            to: href,
                            class: "inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg border border-violet-200 dark:border-violet-900/70 bg-violet-50/70 dark:bg-violet-950/30 text-slate-800 dark:text-slate-200 hover:border-violet-500 transition-colors",
                            span { class: "px-1.5 py-0.5 rounded bg-violet-600 text-white text-[11px] font-mono font-bold",
                              "{badge_txt}"
                            }
                            span { class: "font-medium", "{link_title} →" }
                          }
                        }
                      }
                    }
                  }
                }
              } else {
                rsx! {}
              }
            }

            // 当前选中子模块关联的 L2 Safetensors 真实权重对照表（总览态给轻量直达摘要，钻取态精准匹配当前层段张量）
            if focus() == InspectorFocus::Overview {
              if let Some(ref st) = blueprint.safetensors {
                div { class: "mt-4 pt-3 border-t border-slate-200 dark:border-slate-800 flex flex-wrap items-center justify-between gap-2 text-xs text-slate-600 dark:text-slate-300",
                  span { class: "font-mono",
                    "L2 Safetensors 核验：{st.patterns.len()} 类权重模式 · {st.shard_count} 分片 · {st.total_tensor_count} 个原始张量"
                  }
                  button {
                    r#type: "button",
                    class: "font-semibold text-indigo-600 dark:text-indigo-400 hover:underline cursor-pointer",
                    onclick: move |_| detail_tab.set(DetailTab::SafetensorsTree),
                    "在下方展开全量权重命名空间 ↓"
                  }
                }
              }
            } else {
              {
                let matched = blueprint.matching_tensors_for_layer(focus(), selected_layer());
                if !matched.is_empty() {
                  rsx! {
                    div { class: "mt-4 pt-3 border-t border-slate-200 dark:border-slate-800",
                      div { class: "text-[13px] font-semibold text-slate-700 dark:text-slate-300 mb-2 flex items-center justify-between",
                        span { "匹配该子模块的真实 Safetensors 权重张量 ({matched.len()} 项)" }
                        span { class: "text-xs font-mono text-slate-400", "Shape = [out, in]" }
                      }
                      TensorPatternTable { items: matched.into_iter().cloned().collect() }
                    }
                  }
                } else {
                  rsx! {}
                }
              }
            }
          }

          // 2B. 工程推演与底层数据标签页（显存模拟器 / 全量权重树 / 超参 & JSON）
          div { class: card_class("p-4 sm:p-5 bg-white dark:bg-slate-900/60"),
            div { class: "flex flex-wrap gap-1.5 pb-3 mb-4 border-b border-slate-200 dark:border-slate-800",
              for (tab_id, tab_title) in [
                (DetailTab::VramSimulator, "推理显存 & KV Cache 精算器"),
                (DetailTab::SafetensorsTree, "Safetensors 权重命名空间"),
                (DetailTab::HyperparamsAndJson, "完整超参 & config.json"),
              ] {
                {
                  let active = detail_tab() == tab_id;
                  let cls = if active {
                    "px-3 py-1.5 rounded-lg text-xs font-semibold bg-slate-900 text-white dark:bg-white dark:text-slate-900 cursor-pointer"
                  } else {
                    "px-3 py-1.5 rounded-lg text-xs font-medium text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800 cursor-pointer"
                  };
                  rsx! {
                    button {
                      key: "{tab_title}",
                      r#type: "button",
                      class: cls,
                      onclick: move |_| detail_tab.set(tab_id),
                      "{tab_title}"
                    }
                  }
                }
              }
            }

            match detail_tab() {
              DetailTab::VramSimulator => {
                let est = blueprint.estimate_vram(precision(), ctx_len(), batch_size());
                let cache_total = est.kv_cache_bytes + est.state_cache_bytes;
                let saved_kv = (est.mha_baseline_kv_bytes - cache_total).max(0.0);
                let w_pct = (est.weights_bytes / est.total_bytes.max(1.0) * 100.0).clamp(2.0, 96.0);
                let kv_pct = if cache_total <= 0.0 {
                  0.0
                } else {
                  (cache_total / est.total_bytes.max(1.0) * 100.0).clamp(1.0, 96.0)
                };
                rsx! {
                  div { class: "space-y-4",
                    // 控制拨盘：精度 / 上下文长度 / Batch Size
                    div { class: "grid grid-cols-1 sm:grid-cols-3 gap-3",
                      div {
                        div { class: "text-[11px] font-semibold text-slate-500 dark:text-slate-400 mb-1.5",
                          "部署量化精度"
                        }
                        div { class: "flex flex-wrap gap-1",
                          for p in PrecisionPreset::ALL {
                            button {
                              key: "{p.label()}",
                              r#type: "button",
                              class: if precision() == p {
                                "px-2 py-1 rounded text-[11px] font-semibold bg-indigo-600 text-white cursor-pointer"
                              } else {
                                "px-2 py-1 rounded text-[11px] bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 cursor-pointer"
                              },
                              onclick: move |_| precision.set(p),
                              "{p.label()}"
                            }
                          }
                        }
                      }

                      div {
                        div { class: "text-[11px] font-semibold text-slate-500 dark:text-slate-400 mb-1.5",
                          if blueprint.is_diffusion_pipeline() {
                            "潜空间序列 (Latent Tokens)"
                          } else {
                            "上下文长度 (Context)"
                          }
                        }
                        div { class: "flex flex-wrap gap-1",
                          for (c_val, c_lbl) in [(4096u64, "4K"), (16384, "16K"), (32768, "32K"), (65536, "64K"), (131072, "128K")] {
                            button {
                              key: "{c_lbl}",
                              r#type: "button",
                              class: if ctx_len() == c_val {
                                "px-2 py-1 rounded text-[11px] font-mono font-semibold bg-indigo-600 text-white cursor-pointer"
                              } else {
                                "px-2 py-1 rounded text-[11px] font-mono bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 cursor-pointer"
                              },
                              onclick: move |_| ctx_len.set(c_val),
                              "{c_lbl}"
                            }
                          }
                        }
                      }

                      div {
                        div { class: "text-[11px] font-semibold text-slate-500 dark:text-slate-400 mb-1.5",
                          "并发序列数 (Batch)"
                        }
                        div { class: "flex flex-wrap gap-1",
                          for b_val in [1u64, 4, 16, 32] {
                            button {
                              key: "{b_val}",
                              r#type: "button",
                              class: if batch_size() == b_val {
                                "px-2.5 py-1 rounded text-[11px] font-mono font-semibold bg-indigo-600 text-white cursor-pointer"
                              } else {
                                "px-2.5 py-1 rounded text-[11px] font-mono bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 cursor-pointer"
                              },
                              onclick: move |_| batch_size.set(b_val),
                              "B={b_val}"
                            }
                          }
                        }
                      }
                    }

                    // 显存总量与分项占比条
                    div { class: "rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/80 dark:bg-slate-950/60 p-4",
                      div { class: "flex flex-wrap items-baseline justify-between gap-2 mb-2",
                        div {
                          span { class: "text-xs text-slate-500 dark:text-slate-400 mr-2", "预估推理总显存：" }
                          span { class: "text-xl font-extrabold font-mono text-slate-900 dark:text-white",
                            "{format_bytes_iec(est.total_bytes)}"
                          }
                        }
                        span { class: "text-xs font-semibold text-indigo-600 dark:text-indigo-400",
                          "硬件建议：{recommend_gpu_tier(est.total_bytes)}"
                        }
                      }

                      div { class: "h-3 w-full rounded-full overflow-hidden flex bg-slate-200 dark:bg-slate-800 my-2.5",
                        div {
                          class: "bg-indigo-500 h-full transition-all",
                          style: "width: {w_pct:.1}%",
                          title: "Model Weights",
                        }
                        div {
                          class: "bg-amber-500 h-full transition-all",
                          style: "width: {kv_pct:.1}%",
                          title: "KV / Recurrent State Cache",
                        }
                        div {
                          class: "bg-slate-400 dark:bg-slate-600 h-full flex-1",
                          title: "Runtime Workspace & Activations",
                        }
                      }

                      div { class: "grid grid-cols-1 sm:grid-cols-3 gap-2 text-xs font-mono pt-1",
                        div { class: "flex items-center gap-2",
                          span { class: "w-2.5 h-2.5 rounded-xs bg-indigo-500 shrink-0" }
                          span { class: "text-slate-600 dark:text-slate-300",
                            "模型权重: {format_bytes_iec(est.weights_bytes)}"
                          }
                        }
                        div { class: "flex items-center gap-2",
                          span { class: "w-2.5 h-2.5 rounded-xs bg-amber-500 shrink-0" }
                          span { class: "text-slate-600 dark:text-slate-300",
                            if est.state_cache_bytes > 0.0 && est.kv_cache_bytes > 0.0 {
                              "KV+O(1)状态: {format_bytes_iec(est.kv_cache_bytes)}+{format_bytes_iec(est.state_cache_bytes)}"
                            } else if est.state_cache_bytes > 0.0 {
                              "O(1)循环状态: {format_bytes_iec(est.state_cache_bytes)}"
                            } else {
                              "KV Cache: {format_bytes_iec(est.kv_cache_bytes)}"
                            }
                          }
                        }
                        div { class: "flex items-center gap-2",
                          span { class: "w-2.5 h-2.5 rounded-xs bg-slate-400 shrink-0" }
                          span { class: "text-slate-600 dark:text-slate-300",
                            "激活/工作区: {format_bytes_iec(est.activation_overhead_bytes)}"
                          }
                        }
                      }

                      if blueprint.is_diffusion_pipeline() {
                        div { class: "mt-3 pt-2.5 border-t border-slate-200 dark:border-slate-800 text-xs text-emerald-700 dark:text-emerald-400 font-medium",
                          "★ 扩散管线特性：迭代去噪无自回归 KV Cache（恒为 0 B）；激活显存随潜空间分辨率序列长度（如 1024px 图像≈4K、480p 视频≈32K、720p 视频≈64K）及 VAE 解码峰值缓冲增长。"
                        }
                      } else if blueprint.is_single_pass_encoder() {
                        div { class: "mt-3 pt-2.5 border-t border-slate-200 dark:border-slate-800 text-xs text-emerald-700 dark:text-emerald-400 font-medium",
                          "★ 编码器特性：单次双向并行前向即可产出句向量/重排打分，无需自回归驻留 KV Cache（恒为 0 B）。"
                        }
                      } else if saved_kv > 1024.0 * 1024.0 {
                        div { class: "mt-3 pt-2.5 border-t border-slate-200 dark:border-slate-800 text-xs text-emerald-700 dark:text-emerald-400 font-medium",
                          "★ 架构收益：在当前 B={batch_size()}、Ctx={ctx_len()} 配置下，相比标准多头注意力 (MHA 需 {format_bytes_iec(est.mha_baseline_kv_bytes)}) 节省了 {format_bytes_iec(saved_kv)} 缓存显存！"
                        }
                      }
                    }
                  }
                }
              }

              DetailTab::SafetensorsTree => {
                if let Some(ref st) = blueprint.safetensors {
                  let cur_filter = weight_stage_filter();
                  let filtered_pats: Vec<TensorPatternEntry> = st
                    .patterns
                    .iter()
                    .filter(|p| cur_filter == "all" || p.stage == cur_filter)
                    .cloned()
                    .collect();

                  rsx! {
                    div { class: "space-y-3",
                      div { class: "flex flex-wrap items-center justify-between gap-2 text-xs text-slate-600 dark:text-slate-300 font-mono",
                        div { class: "flex flex-wrap items-center gap-2",
                          span { "分片数: {st.shard_count}" }
                          span { "· 原始张量数: {st.total_tensor_count}" }
                          if let Some(bytes) = st.total_bytes {
                            span { "· 磁盘总计: {format_bytes_iec(bytes as f64)}" }
                          }
                        }
                        div { class: "flex flex-wrap gap-1.5",
                          for (dt, cnt) in st.dtype_histogram.iter() {
                            span { key: "{dt}", class: "px-2 py-0.5 rounded bg-slate-100 dark:bg-slate-800 text-[11px]",
                              "{dt}: {cnt}"
                            }
                          }
                        }
                      }

                      div { class: "flex flex-wrap gap-1",
                        for (st_key, st_lbl) in [
                          ("all", "全部阶段"),
                          ("vision", "Vision"),
                          ("audio", "Audio"),
                          ("embedding", "Embedding"),
                          ("attention", "Attn / SSM"),
                          ("ffn", "MLP / FFN"),
                          ("moe", "Sparse MoE"),
                          ("norm", "Norm"),
                          ("lm_head", "Output Head"),
                        ] {
                          if st_key == "all" || st.patterns.iter().any(|p| p.stage == st_key) {
                            button {
                              key: "{st_key}",
                              r#type: "button",
                              class: if weight_stage_filter() == st_key {
                                "px-2 py-1 rounded text-[11px] font-semibold bg-indigo-600 text-white cursor-pointer"
                              } else {
                                "px-2 py-1 rounded text-[11px] bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 cursor-pointer"
                              },
                              onclick: move |_| weight_stage_filter.set(st_key.to_string()),
                              "{st_lbl}"
                            }
                          }
                        }
                      }

                      TensorPatternTable { items: filtered_pats }
                    }
                  }
                } else {
                  rsx! {
                    div { class: "py-6 text-center text-xs text-slate-500 dark:text-slate-400",
                      "该仓库未提供公开的 model.safetensors.index.json 或单文件 Safetensors Header（例如仅含 GGUF/PyTorch bin 权重或粘贴了单 config.json）。"
                    }
                  }
                }
              }

              DetailTab::HyperparamsAndJson => {
                let lang_here = current_lang();
                rsx! {
                  ConfigAnatomyPanel {
                    blueprint: blueprint.clone(),
                    focus,
                    lang: lang_here,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConfigViewMode {
  Annotated,
  RawJson,
}

fn focus_short_label(f: InspectorFocus) -> &'static str {
  match f {
    InspectorFocus::Overview => "总体架构",
    InspectorFocus::Vision => "Vision",
    InspectorFocus::Audio => "Audio",
    InspectorFocus::Embedding => "Embed",
    InspectorFocus::Attention => "Attention",
    InspectorFocus::FfnOrMoe => "FFN / MoE",
    InspectorFocus::NormAndResidual => "Norm",
    InspectorFocus::LmHead => "Output Head",
    InspectorFocus::MtpHead => "MTP",
  }
}

/// 「完整超参 & config.json」逐字段交互解剖与拓扑图联动面板（按需加载服务端词典，避免客户端 WASM 膨胀）。
#[component]
fn ConfigAnatomyPanel(
  blueprint: ModelBlueprint,
  focus: Signal<InspectorFocus>,
  lang: Language,
) -> Element {
  rsx! {
    SuspenseBoundary {
      fallback: |_| rsx! {
        div { class: "flex items-center justify-center py-8",
          Spinner { size: SpinnerSize::Md, class: "border-t-primary" }
        }
      },
      ConfigAnatomyLoaded { blueprint, focus, lang }
    }
  }
}

#[component]
fn ConfigAnatomyLoaded(
  blueprint: ModelBlueprint,
  focus: Signal<InspectorFocus>,
  lang: Language,
) -> Element {
  let docs_res = use_server_future(get_config_guide_json_cached)?;
  let view_mode = use_signal(|| ConfigViewMode::Annotated);
  let group_filter: Signal<Option<FieldGroup>> = use_signal(|| None);
  let copied_json = use_signal(|| false);
  let docs: &[ConfigFieldDocDto] = match docs_res() {
    Some(Ok(raw)) => &get_or_parse_config_guide(&raw).field_docs,
    _ => &[],
  };

  render_config_anatomy_body(&blueprint, focus, lang, docs, view_mode, group_filter, copied_json)
}

fn render_config_anatomy_body(
  blueprint: &ModelBlueprint,
  focus: Signal<InspectorFocus>,
  lang: Language,
  docs: &[ConfigFieldDocDto],
  mut view_mode: Signal<ConfigViewMode>,
  mut group_filter: Signal<Option<FieldGroup>>,
  mut copied_json: Signal<bool>,
) -> Element {
  let annotated_fields = extract_annotated_fields_with_docs(
    &blueprint.raw_config_pretty,
    ConfigFileKind::ModelConfig,
    docs,
  );
  let grouped_fields = group_annotated_fields(&annotated_fields);
  let known_count = annotated_fields.iter().filter(|f| f.doc.is_some()).count();
  let total_count = annotated_fields.len();

  let aux_annotated: Vec<(String, Vec<AnnotatedConfigField>)> = blueprint
    .aux_configs
    .iter()
    .map(|snap| {
      let fk = ConfigFileKind::from_filename(&snap.filename);
      (snap.filename.clone(), extract_annotated_fields_with_docs(&snap.raw_json_pretty, fk, docs))
    })
    .filter(|(_, items)| !items.is_empty())
    .collect();

  let raw_for_copy = blueprint.raw_config_pretty.clone();
  let sel_group = group_filter();

  rsx! {
    div { class: "space-y-4",
      // 顶部子视图切换 + 一键复制 JSON + 全景指南入口
      div { class: "flex flex-wrap items-center justify-between gap-2",
        div { class: "inline-flex rounded-lg border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950 p-0.5",
          button {
            r#type: "button",
            class: if view_mode() == ConfigViewMode::Annotated {
              "px-2.5 py-1 rounded-md text-xs font-semibold bg-indigo-600 text-white cursor-pointer"
            } else {
              "px-2.5 py-1 rounded-md text-xs font-medium text-slate-600 dark:text-slate-300 hover:text-slate-900 dark:hover:text-white cursor-pointer"
            },
            onclick: move |_| view_mode.set(ConfigViewMode::Annotated),
            {lang.pick("逐字段架构解剖", "Annotated Fields")}
            span { class: "ml-1 opacity-80 font-mono text-[10px]", "({known_count}/{total_count})" }
          }
          button {
            r#type: "button",
            class: if view_mode() == ConfigViewMode::RawJson {
              "px-2.5 py-1 rounded-md text-xs font-semibold bg-indigo-600 text-white cursor-pointer"
            } else {
              "px-2.5 py-1 rounded-md text-xs font-medium text-slate-600 dark:text-slate-300 hover:text-slate-900 dark:hover:text-white cursor-pointer"
            },
            onclick: move |_| view_mode.set(ConfigViewMode::RawJson),
            {lang.pick("原始 JSON 原文", "Raw JSON")}
          }
        }

        div { class: "flex flex-wrap items-center gap-2 text-xs",
          button {
            r#type: "button",
            class: "px-2.5 py-1 rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 text-slate-700 dark:text-slate-200 hover:border-indigo-400 font-medium cursor-pointer transition-colors",
            onclick: move |_| {
              let text = raw_for_copy.clone();
              spawn(async move {
                if widgets::browser::copy_text(&text).await {
                  copied_json.set(true);
                }
              });
            },
            if copied_json() {
              {lang.pick("✓ 已复制 JSON", "✓ Copied JSON")}
            } else {
              {lang.pick("复制原始 JSON", "Copy Raw JSON")}
            }
          }
          Link {
            to: "/models/configs",
            class: "inline-flex items-center gap-1 px-2.5 py-1 rounded-lg border border-indigo-200 dark:border-indigo-900/70 bg-indigo-50/70 dark:bg-indigo-950/30 text-indigo-700 dark:text-indigo-300 font-semibold hover:border-indigo-400 transition-colors",
            {lang.pick("📘 配置文件全景指南 ↗", "📘 Full Config Guide ↗")}
          }
        }
      }

      // 6 格核心速览条
      div { class: "grid grid-cols-2 sm:grid-cols-3 gap-2 text-xs font-mono",
        HyperKv { k: "model_type", v: blueprint.model_type.clone() }
        HyperKv { k: "hidden_act", v: blueprint.hidden_act.clone() }
        HyperKv { k: "tie_word_embeddings", v: blueprint.tie_word_embeddings.to_string() }
        HyperKv {
          k: "rms_norm_eps",
          v: blueprint.rms_norm_eps.map(|e| format!("{e:.0e}")).unwrap_or_else(|| "-".to_string()),
        }
        HyperKv {
          k: "tokenizer_class",
          v: blueprint.tokenizer_class.clone().unwrap_or_else(|| "-".to_string()),
        }
        HyperKv {
          k: "sampling (T / top_p)",
          v: match (blueprint.default_temperature, blueprint.default_top_p) {
            (Some(t), Some(p)) => format!("T={t}, p={p}"),
            _ => "-".to_string(),
          },
        }
      }

      match view_mode() {
        ConfigViewMode::Annotated => rsx! {
          div { class: "space-y-4",
            // 语义分组筛选条
            if grouped_fields.len() > 1 {
              div { class: "flex flex-wrap items-center gap-1.5",
                button {
                  r#type: "button",
                  class: if sel_group.is_none() {
                    "px-2 py-1 rounded-md text-[11px] font-semibold bg-slate-900 text-white dark:bg-white dark:text-slate-900 cursor-pointer"
                  } else {
                    "px-2 py-1 rounded-md text-[11px] bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 cursor-pointer"
                  },
                  onclick: move |_| group_filter.set(None),
                  {lang.pick("全部分组", "All Groups")}
                  span { class: "ml-1 opacity-75 font-mono", "({total_count})" }
                }
                for (grp, items) in grouped_fields.iter() {
                  {
                    let g = *grp;
                    let cnt = items.len();
                    let active = sel_group == Some(g);
                    let cls = if active {
                      "px-2 py-1 rounded-md text-[11px] font-semibold bg-slate-900 text-white dark:bg-white dark:text-slate-900 cursor-pointer"
                    } else {
                      "px-2 py-1 rounded-md text-[11px] bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700 cursor-pointer"
                    };
                    rsx! {
                      button {
                        key: "cfg-grp-{g.id()}",
                        r#type: "button",
                        class: cls,
                        onclick: move |_| group_filter.set(Some(g)),
                        "{g.label(lang)}"
                        span { class: "ml-1 opacity-75 font-mono", "({cnt})" }
                      }
                    }
                  }
                }
              }
            }

            // 按语义分组渲染主配置字段
            div { class: "space-y-3 max-h-[34rem] overflow-auto pr-1",
              for (grp, items) in grouped_fields.iter() {
                if sel_group.is_none_or(|g| g == *grp) {
                  div {
                    key: "section-{grp.id()}",
                    class: "rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-950/40 overflow-hidden",
                    div { class: "px-3 py-2 border-b border-slate-200 dark:border-slate-800 bg-slate-100/70 dark:bg-slate-900/70 flex items-center justify-between text-xs",
                      span { class: "font-bold text-slate-800 dark:text-slate-200",
                        "{grp.label(lang)}"
                      }
                      span { class: "font-mono text-[11px] text-slate-500 dark:text-slate-400",
                        "{items.len()} fields"
                      }
                    }
                    div { class: "divide-y divide-slate-200/70 dark:divide-slate-800/70",
                      for field in items.iter() {
                        {render_annotated_field_row(
                          format!(
                            "{}-{}",
                            field.source_slot.as_deref().unwrap_or(""),
                            field.path
                          ),
                          field,
                          focus,
                          lang,
                        )}
                      }
                    }
                  }
                }
              }
            }

            // 附属配置文件摘要解剖（generation_config.json / tokenizer_config.json / preprocessor_config.json / 1_Pooling/config.json）
            if !aux_annotated.is_empty() {
              div { class: "pt-3 border-t border-slate-200 dark:border-slate-800 space-y-3",
                div { class: "text-xs font-bold text-slate-700 dark:text-slate-200 flex items-center justify-between",
                  span {
                    {lang.pick(
                      "附属配置文件解析（generation / tokenizer / preprocessor / pooling）",
                      "Auxiliary Config Files (generation / tokenizer / preprocessor / pooling)",
                    )}
                  }
                  span { class: "font-mono text-[11px] text-slate-400", "{aux_annotated.len()} files" }
                }
                for (fname, items) in aux_annotated.iter() {
                  div {
                    key: "aux-{fname}",
                    class: "rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/40 dark:bg-slate-950/40 overflow-hidden",
                    div { class: "px-3 py-1.5 border-b border-slate-200 dark:border-slate-800 bg-slate-100/70 dark:bg-slate-900/70 flex items-center justify-between text-xs font-mono",
                      span { class: "font-bold text-indigo-700 dark:text-indigo-300", "{fname}" }
                      span { class: "text-[11px] text-slate-500 dark:text-slate-400", "{items.len()} fields" }
                    }
                    div { class: "divide-y divide-slate-200/70 dark:divide-slate-800/70 max-h-56 overflow-auto",
                      for field in items.iter() {
                        {render_annotated_field_row(
                          format!("aux-{fname}-{}", field.path),
                          field,
                          focus,
                          lang,
                        )}
                      }
                    }
                  }
                }
              }
            }
          }
        },
        ConfigViewMode::RawJson => rsx! {
          div { class: "space-y-3",
            pre { class: "max-h-96 overflow-auto rounded-xl bg-slate-950 text-slate-100 p-3.5 text-xs font-mono leading-relaxed border border-slate-800",
              {render_highlighted_json(&blueprint.raw_config_pretty)}
            }
            for snap in blueprint.aux_configs.iter() {
              div {
                key: "raw-aux-{snap.filename}",
                class: "rounded-xl border border-slate-800 bg-slate-950 overflow-hidden",
                div { class: "px-3 py-1.5 border-b border-slate-800 text-[11px] font-mono text-indigo-300",
                  "{snap.filename}"
                }
                pre { class: "max-h-52 overflow-auto p-3 text-xs font-mono text-slate-200 leading-relaxed",
                  {render_highlighted_json(&snap.raw_json_pretty)}
                }
              }
            }
          }
        },
      }
    }
  }
}

fn render_annotated_field_row(
  row_key: String,
  field: &AnnotatedConfigField,
  mut focus: Signal<InspectorFocus>,
  lang: Language,
) -> Element {
  let active_focus = field.focus.is_some_and(|target| focus() == target);

  rsx! {
    div {
      key: "{row_key}",
      class: "px-3 py-2.5 text-xs space-y-1.5 hover:bg-white/70 dark:hover:bg-slate-900/50 transition-colors",
      div { class: "flex flex-wrap items-center justify-between gap-2",
        div { class: "flex flex-wrap items-center gap-1.5 min-w-0",
          if let Some(ref slot) = field.source_slot {
            span { class: "px-1.5 py-0.5 rounded bg-slate-200/80 dark:bg-slate-800 text-[10px] font-mono text-slate-600 dark:text-slate-300 shrink-0",
              "{slot}"
            }
          }
          code { class: "font-mono font-bold text-indigo-700 dark:text-indigo-300 break-all",
            "{field.path}"
          }
          span { class: "text-slate-400", "=" }
          code { class: "px-1.5 py-0.5 rounded bg-slate-200/70 dark:bg-slate-800 font-mono font-semibold text-slate-900 dark:text-slate-100 break-all",
            "{field.display_value}"
          }
        }

        if let Some(target_focus) = field.focus {
          button {
            r#type: "button",
            class: if active_focus {
              "shrink-0 px-2 py-0.5 rounded-md text-[11px] font-semibold bg-indigo-600 text-white cursor-pointer"
            } else {
              "shrink-0 px-2 py-0.5 rounded-md text-[11px] font-semibold border border-indigo-200 dark:border-indigo-900/70 bg-indigo-50/70 dark:bg-indigo-950/30 text-indigo-700 dark:text-indigo-300 hover:border-indigo-400 cursor-pointer transition-colors"
            },
            onclick: move |_| focus.set(target_focus),
            if active_focus {
              {lang.pick("已聚焦 · ", "Focused · ")}
              "{focus_short_label(target_focus)}"
            } else {
              {lang.pick("定位到架构图 ↗ ", "Locate in Graph ↗ ")}
              "{focus_short_label(target_focus)}"
            }
          }
        }
      }

      if let Some(ref doc) = field.doc {
        p { class: "text-slate-600 dark:text-slate-400 leading-relaxed",
          {render_inline_markup(doc.summary(lang))}
        }
        if doc.formula_impact.is_some() || doc.lab_link.is_some() {
          div { class: "flex flex-wrap items-center justify-between gap-2 pt-0.5",
            if let Some(ref formula) = doc.formula_impact {
              span { class: "font-mono text-[11px] text-indigo-700 dark:text-indigo-300",
                "ƒ {formula}"
              }
            }
            if let Some((ref href, ref label)) = doc.lab_link {
              Link {
                to: href.clone(),
                class: "text-[11px] font-semibold text-emerald-600 dark:text-emerald-400 hover:underline",
                "{label} ↗"
              }
            }
          }
        }
      } else {
        div { class: "text-[11px] text-slate-400",
          {lang.pick("模型扩展字段（原文保留，未在通用词典中固定建模）", "Model-specific extension field (preserved verbatim)")}
        }
      }
    }
  }
}

#[component]
fn HyperKv(k: String, v: String) -> Element {
  rsx! {
    div { class: "rounded-lg border border-slate-200 dark:border-slate-800 bg-slate-50/60 dark:bg-slate-950/50 px-2.5 py-2",
      div { class: "text-[10px] text-slate-400", "{k}" }
      div { class: "text-xs font-semibold text-slate-800 dark:text-slate-100 truncate", "{v}" }
    }
  }
}

#[component]
fn ParamStackedBar(params: crate::arch::ParamBreakdown, is_diffusion: bool) -> Element {
  let tot = params.total_params.max(1) as f64;
  let slices = [
    (if is_diffusion { "VAE Codec" } else { "Vision" }, params.vision_encoder, "bg-fuchsia-500"),
    ("Audio / Speech", params.audio_and_speech, "bg-cyan-500"),
    (if is_diffusion { "Text Encoders" } else { "Embedding" }, params.embedding, "bg-emerald-500"),
    ("Attention / SSM", params.attention, "bg-indigo-500"),
    ("Dense FFN", params.dense_ffn, "bg-sky-500"),
    ("MoE Routed", params.moe_routed, "bg-amber-500"),
    ("MoE Shared", params.moe_shared, "bg-teal-500"),
    ("Output Head", params.lm_head, "bg-rose-500"),
  ];

  rsx! {
    div { class: "mt-4",
      div { class: "h-2.5 w-full rounded-full overflow-hidden flex bg-slate-100 dark:bg-slate-800",
        for (name, val, color) in slices.iter() {
          if *val > 0 {
            {
              let pct = (*val as f64) * 100.0 / tot;
              rsx! {
                div {
                  key: "{name}",
                  class: "{color} h-full",
                  style: "width: {pct:.2}%",
                  title: "{name}: {format_count(*val)} ({pct:.1}%)",
                }
              }
            }
          }
        }
      }
      div { class: "mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs font-mono text-slate-600 dark:text-slate-400",
        for (name, val, color) in slices.iter() {
          if *val > 0 {
            {
              let pct = (*val as f64) * 100.0 / tot;
              rsx! {
                span { key: "{name}", class: "inline-flex items-center gap-1.5",
                  span { class: "w-2 h-2 rounded-xs {color}" }
                  "{name}: {format_count(*val)} ({pct:.1}%)"
                }
              }
            }
          }
        }
      }
    }
  }
}

#[component]
fn TensorPatternTable(items: Vec<TensorPatternEntry>) -> Element {
  rsx! {
    div { class: "max-h-72 overflow-auto rounded-lg border border-slate-200 dark:border-slate-800",
      table { class: "w-full text-left border-collapse text-[13px] font-mono",
        thead { class: "bg-slate-100 dark:bg-slate-800/90 text-xs text-slate-600 dark:text-slate-300 sticky top-0",
          tr {
            th { class: "py-2 px-3 font-semibold whitespace-nowrap", "Tensor Pattern" }
            th { class: "py-2 px-2.5 font-semibold whitespace-nowrap", "Shape" }
            th { class: "py-2 px-2.5 font-semibold whitespace-nowrap", "Dtype" }
            th { class: "py-2 px-2.5 font-semibold text-right whitespace-nowrap", "Count" }
            th { class: "py-2 px-3 font-semibold text-right whitespace-nowrap", "Params" }
          }
        }
        tbody { class: "divide-y divide-slate-200 dark:divide-slate-800 bg-white dark:bg-slate-950/60",
          for item in items {
            {
              let shape_str = item
                .shape
                .as_ref()
                .map(|s| {
                  format!(
                    "[{}]",
                    s.iter().map(u64::to_string).collect::<Vec<_>>().join(", ")
                  )
                })
                .unwrap_or_else(|| "—".to_string());
              let dt_str = item.dtype.clone().unwrap_or_else(|| "—".to_string());
              let tot_str = item
                .total_elements()
                .map(format_count)
                .unwrap_or_else(|| "—".to_string());
              rsx! {
                tr {
                  key: "{item.pattern}",
                  class: "hover:bg-slate-50 dark:hover:bg-slate-900/70",
                  td { class: "py-2 px-3 text-slate-800 dark:text-slate-200 whitespace-nowrap",
                    title: item.example_shard.clone().unwrap_or_default(),
                    "{item.pattern}"
                  }
                  td { class: "py-2 px-2.5 text-indigo-600 dark:text-indigo-400 whitespace-nowrap",
                    "{shape_str}"
                  }
                  td { class: "py-2 px-2.5 text-slate-600 dark:text-slate-400 whitespace-nowrap",
                    "{dt_str}"
                  }
                  td { class: "py-2 px-2.5 text-right text-slate-500 tabular-nums whitespace-nowrap",
                    "×{item.repeat_count}"
                  }
                  td { class: "py-2 px-3 text-right text-slate-700 dark:text-slate-300 tabular-nums whitespace-nowrap",
                    "{tot_str}"
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

pub(crate) fn clean_server_err(raw: &str) -> String {
  raw.strip_prefix("error running server function: ").unwrap_or(raw).to_string()
}

#[cfg(test)]
mod tests {
  use super::*;
  use widgets::MdxComponent;

  #[test]
  fn ssr_renders_mdx_model_arch_preset_with_mla_and_moe_details() {
    #[component]
    fn TestHost() -> Element {
      let mut attrs = HashMap::new();
      attrs.insert("preset".to_string(), "deepseek-v3".to_string());
      attrs.insert("focus".to_string(), "attention".to_string());
      ModelArchComponent.render(&attrs)
    }

    let mut vdom = VirtualDom::new(TestHost);
    vdom.rebuild_in_place();
    let html = dioxus_ssr::render(&vdom);
    assert!(html.contains("deepseek-ai/DeepSeek-V3"));
    assert!(html.contains("DeepSeek MLA + MoE"));
    assert!(html.contains("kv_a_proj_with_mqa"));
    assert!(html.contains("671.03B"));
  }

  #[test]
  fn ssr_renders_missing_attr_warning_gracefully() {
    #[component]
    fn EmptyHost() -> Element {
      let attrs = HashMap::new();
      ModelArchComponent.render(&attrs)
    }

    let mut vdom = VirtualDom::new(EmptyHost);
    vdom.rebuild_in_place();
    let html = dioxus_ssr::render(&vdom);
    assert!(html.contains("ModelArch 组件需指定"));
  }

  #[test]
  fn ssr_renders_audio_ssm_and_embedding_workbenches() {
    use crate::arch::{build_model_blueprint_with_aux, AuxConfigFiles};

    #[component]
    fn MultimodalHost() -> Element {
      let whisper = build_model_blueprint_with_aux(
        "openai/whisper-large-v3",
        "HuggingFace",
        r#"{"architectures":["WhisperForConditionalGeneration"],"model_type":"whisper","d_model":1280,"encoder_layers":32,"encoder_attention_heads":20,"encoder_ffn_dim":5120,"decoder_layers":32,"decoder_attention_heads":20,"decoder_ffn_dim":5120,"num_mel_bins":128,"max_source_positions":1500,"max_target_positions":448,"vocab_size":51866}"#,
        AuxConfigFiles::default(),
        None,
      )
      .expect("whisper");
      let mamba = build_model_blueprint_with_aux(
        "state-spaces/mamba-2.8b-hf",
        "HuggingFace",
        r#"{"architectures":["MambaForCausalLM"],"model_type":"mamba","d_model":2560,"n_layer":64,"state_size":16,"conv_kernel":4,"expand":2,"vocab_size":50280}"#,
        AuxConfigFiles::default(),
        None,
      )
      .expect("mamba");
      let bge = build_model_blueprint_with_aux(
        "BAAI/bge-m3",
        "HuggingFace",
        r#"{"architectures":["XLMRobertaModel"],"model_type":"xlm-roberta","hidden_size":1024,"num_hidden_layers":24,"num_attention_heads":16,"intermediate_size":4096,"max_position_embeddings":8194,"vocab_size":250002}"#,
        AuxConfigFiles {
          pooling_json: Some(r#"{"word_embedding_dimension":1024,"pooling_mode_cls_token":true}"#),
          ..Default::default()
        },
        None,
      )
      .expect("bge");

      rsx! {
        div {
          ModelVisualizerWorkbench {
            blueprint: whisper,
            initial_focus: Some(InspectorFocus::Audio),
            embedded: true,
          }
          ModelVisualizerWorkbench {
            blueprint: mamba,
            initial_focus: Some(InspectorFocus::Attention),
            embedded: true,
          }
          ModelVisualizerWorkbench {
            blueprint: bge,
            initial_focus: Some(InspectorFocus::LmHead),
            embedded: true,
          }
        }
      }
    }

    let mut vdom = VirtualDom::new(MultimodalHost);
    vdom.rebuild_in_place();
    let html = dioxus_ssr::render(&vdom);
    assert!(html.contains("Audio Tower"));
    assert!(html.contains("Conv1D Acoustic Stem"));
    assert!(html.contains("O(1) Recurrent State"));
    assert!(html.contains("0 B KV Cache"));
    assert!(html.contains("Sentence / Document Embedding Pooling Head"));
  }

  #[test]
  fn ssr_renders_diffusion_flux_and_wan_pipeline_workbenches() {
    use crate::arch::build_diffusion_pipeline_blueprint;

    #[component]
    fn DiffusionHost() -> Element {
      let flux = build_diffusion_pipeline_blueprint(
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

      let wan = build_diffusion_pipeline_blueprint(
        "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        "HuggingFace",
        r#"{"_class_name":"WanPipeline","text_encoder":["transformers","UMT5EncoderModel"],"transformer":["diffusers","WanTransformer3DModel"],"vae":["diffusers","AutoencoderKLWan"],"scheduler":["diffusers","UniPCMultistepScheduler"]}"#,
        &[(
          "transformer".to_string(),
          r#"{"_class_name":"WanTransformer3DModel","num_layers":40,"num_attention_heads":40,"attention_head_dim":128,"ffn_dim":13824,"in_channels":16,"patch_size":[1,2,2]}"#.to_string(),
        )],
        None,
      )
      .expect("wan");

      rsx! {
        div {
          ModelVisualizerWorkbench {
            blueprint: flux,
            initial_focus: Some(InspectorFocus::Embedding),
            embedded: true,
          }
          ModelVisualizerWorkbench {
            blueprint: wan,
            initial_focus: Some(InspectorFocus::LmHead),
            embedded: true,
          }
        }
      }
    }

    let mut vdom = VirtualDom::new(DiffusionHost);
    vdom.rebuild_in_place();
    let html = dioxus_ssr::render(&vdom);
    assert!(html.contains("Prompt Conditioners"));
    assert!(html.contains("Multi-Encoder Prompt Conditioning"));
    assert!(html.contains("Latent Unpatchify"));
    assert!(html.contains("AutoencoderKLWan"));
    assert!(html.contains("VAE Decoder"));
  }

  #[cfg(feature = "server")]
  #[test]
  fn ssr_renders_annotated_config_anatomy_and_aux_files() {
    use crate::arch::{build_model_blueprint_with_aux, AuxConfigFiles};
    use crate::config_guide::{
      build_config_field_docs, build_config_guide_payload, render_config_guide_content,
    };

    #[derive(Routable, Clone, PartialEq)]
    enum TestRoute {
      #[route("/")]
      Home {},
    }

    #[component]
    fn Home() -> Element {
      let bp = build_model_blueprint_with_aux(
        "deepseek-ai/DeepSeek-V3",
        "Test",
        &load_builtin_preset("deepseek-v3").expect("deepseek-v3").raw_config_pretty,
        AuxConfigFiles {
          generation_json: Some(
            r#"{"do_sample":true,"temperature":0.6,"top_p":0.95,"eos_token_id":[1,2]}"#,
          ),
          tokenizer_json: Some(r#"{"tokenizer_class":"LlamaTokenizerFast","add_bos_token":true}"#),
          preprocessor_json: None,
          pooling_json: None,
        },
        None,
      )
      .expect("blueprint with aux");
      let focus = use_signal(|| InspectorFocus::Attention);
      let view_mode = use_signal(|| ConfigViewMode::Annotated);
      let group_filter: Signal<Option<FieldGroup>> = use_signal(|| None);
      let copied_json = use_signal(|| false);
      let active_modality = use_signal(|| "llm".to_string());
      let active_file_spec = use_signal(|| ConfigFileKind::ModelConfig);
      let copied_snippet = use_signal(|| None);
      let filter_category = use_signal(|| None);
      let filter_file: Signal<Option<ConfigFileKind>> = use_signal(|| None);
      let filter_group: Signal<Option<FieldGroup>> = use_signal(|| None);
      let search_q = use_signal(String::new);
      let docs = build_config_field_docs();
      let payload = build_config_guide_payload();

      rsx! {
        div {
          {render_config_anatomy_body(
            &bp,
            focus,
            Language::Zh,
            &docs,
            view_mode,
            group_filter,
            copied_json,
          )}
          {render_config_guide_content(
            &payload,
            "test desc",
            active_modality,
            active_file_spec,
            copied_snippet,
            filter_category,
            filter_file,
            filter_group,
            search_q,
          )}
        }
      }
    }

    #[component]
    fn RouterHost() -> Element {
      rsx! { Router::<TestRoute> {} }
    }

    let mut vdom = VirtualDom::new(RouterHost);
    vdom.rebuild_in_place();
    let html = dioxus_ssr::render(&vdom);
    assert!(html.contains("逐字段架构解剖"));
    assert!(html.contains("/models/configs"));
    assert!(html.contains("kv_lora_rank"));
    assert!(html.contains("n_routed_experts"));
    assert!(html.contains("定位到架构图"));
    assert!(html.contains("generation_config.json"));
    assert!(html.contains("tokenizer_config.json"));
    assert!(html.contains("大模型配置文件全景解剖指南"));
    assert!(html.contains("模型架构与管线"));
    assert!(html.contains("输入与推理控制"));
    assert!(html.contains("权重存储与分片"));
    assert!(html.contains("微调与适配器"));
    assert!(html.contains("adapter_config.json"));
    // 行内反引号代码已渲染为 <code>，不再泄漏原始反引号
    assert!(html.contains("AutoConfig.from_pretrained</code>"));
    assert!(!html.contains("`AutoConfig.from_pretrained`"));
    // JSON 词法高亮生效（键名、字符串值与注释均带独立语义色）
    assert!(html.contains("text-sky-300 font-medium"));
    assert!(html.contains("text-emerald-300"));
    assert!(html.contains("text-slate-500 italic"));
    // 四大职责引导词采用与卡片一致的同色系加粗文字
    assert!(html.contains("font-semibold text-indigo-600 dark:text-indigo-400"));
    assert!(html.contains("font-semibold text-emerald-600 dark:text-emerald-400"));
    assert!(html.contains("font-semibold text-amber-600 dark:text-amber-400"));
    assert!(html.contains("font-semibold text-violet-600 dark:text-violet-400"));
    assert!(html.contains("text-[13px] font-mono"));
  }

  #[test]
  fn ssr_defaults_to_overview_first_and_distinguishes_segments_and_global_modules() {
    let bp = load_builtin_preset("deepseek-v3").expect("deepseek-v3 preset");

    // 1. 异构分段行为验证：DeepSeek-V3 全部层段共用 MLA，点击层段时优先聚焦差异所在的 FFN / MoE
    assert_eq!(bp.default_focus(), InspectorFocus::Overview);
    assert!(bp.has_uniform_attention());
    assert_eq!(bp.recommended_focus_for_segment(0), InspectorFocus::FfnOrMoe);
    assert_eq!(bp.recommended_focus_for_segment(1), InspectorFocus::FfnOrMoe);
    assert!(bp.matching_tensors_for_layer(InspectorFocus::Overview, 0).is_empty());

    let dense_tensors = bp.matching_tensors_for_layer(InspectorFocus::FfnOrMoe, 0);
    let moe_tensors = bp.matching_tensors_for_layer(InspectorFocus::FfnOrMoe, 3);
    assert!(!dense_tensors.is_empty());
    assert!(!moe_tensors.is_empty());
    assert!(dense_tensors.iter().all(|t| t.stage == "ffn"));
    assert!(moe_tensors.iter().all(|t| t.stage == "moe"));

    // 2. 默认渲染：先总体架构（紧凑总览 + 全模型端到端 SVG + 异构段对照），左栏吸顶，隐藏重复 h2
    #[derive(Routable, Clone, PartialEq)]
    enum OverviewRoute {
      #[route("/")]
      Home {},
    }

    #[component]
    fn Home() -> Element {
      let ds = load_builtin_preset("deepseek-v3").expect("deepseek-v3");
      let ds_lm = ds.clone();
      let ds_attn = ds.clone();
      rsx! {
        div {
          ModelVisualizerWorkbench {
            blueprint: ds,
            initial_focus: None,
            embedded: false,
            hide_identity_header: true,
          }
          ModelVisualizerWorkbench {
            blueprint: ds_lm,
            initial_focus: Some(InspectorFocus::LmHead),
            embedded: true,
          }
          ModelVisualizerWorkbench {
            blueprint: ds_attn,
            initial_focus: Some(InspectorFocus::Attention),
            embedded: true,
          }
        }
      }
    }

    #[component]
    fn Host() -> Element {
      rsx! { Router::<OverviewRoute> {} }
    }

    let mut vdom = VirtualDom::new(Host);
    vdom.rebuild_in_place();
    let html = dioxus_ssr::render(&vdom);

    assert!(html.contains("Architecture Overview · 全模型端到端总体架构"));
    assert!(html.contains("总体架构"));
    assert!(html.contains("异构层段架构对照 (2 段)"));
    assert!(html.contains("钻取 Dense FFN →"));
    assert!(html.contains("钻取 Sparse MoE →"));
    assert!(html.contains("展开层块"));
    assert!(html.contains("展开输出层"));
    assert!(html.contains("lg:sticky lg:top-20"));
    assert!(html.contains("Operator Subgraph · Global Module · Output Head"));
    assert!(html.contains("全部层段共用"));
    assert!(html.contains("查看本段 FFN / MoE →"));
    // TensorPatternTable 不再使用 break-all 折断权重名
    assert!(html.contains("text-slate-800 dark:text-slate-200 whitespace-nowrap"));
    // 顶层视图模式切换栏包含「分层联动工作台」与「全网张量维流图 (DAG)」
    assert!(html.contains("分层联动工作台"));
    assert!(html.contains("全网张量维流图 (DAG)"));
  }

  #[test]
  fn ssr_renders_tensor_flow_dag_view_with_legend_and_export() {
    #[component]
    fn DagHost() -> Element {
      let ds = load_builtin_preset("deepseek-v3").expect("deepseek-v3");
      rsx! {
        TensorFlowGraphView {
          blueprint: ds,
          on_drill_to_workbench: move |_| {},
        }
      }
    }

    let mut vdom = VirtualDom::new(DagHost);
    vdom.rebuild_in_place();
    let html = dioxus_ssr::render(&vdom);

    assert!(html.contains("算子色块图例 (Legend)"));
    assert!(html.contains("Inception-v3 (经典 CNN)"));
    assert!(html.contains("ResNet-50 (残差 CNN)"));
    assert!(html.contains("导出高清 SVG ↓"));
    assert!(html.contains("data:image/svg+xml;charset=utf-8,"));
    assert!(html.contains("B×S×7168"));
    assert!(html.contains("B×S×576"));
    assert!(html.contains("切回分层工作台查看该算子 →"));
  }
}
