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
  build_model_blueprint, format_bytes_iec, format_count, load_builtin_preset, recommend_gpu_tier,
  InspectorFocus, ModelBlueprint, PrecisionPreset, BUILTIN_PRESETS,
};
use crate::diagram::{MacroArchitectureFlow, MicroSubgraphDiagram};
use crate::safetensors_meta::{
  parse_safetensors_index_json, summarize_index_and_headers, TensorPatternEntry,
};
use crate::server::{fetch_model_blueprint, normalize_repo_request, ModelHubSource};

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
  let title = lang.pick("LLM 模型架构可视化器", "LLM Architecture Visualizer").to_string();
  let desc = lang
    .pick(
      "通过 HuggingFace 或 ModelScope 的 config.json 与 Safetensors 元数据，精确解析并可视化 Llama、Qwen、DeepSeek (MLA/MoE) 内部架构、张量维度与 KV Cache 显存占用。",
      "Visualize Llama, Qwen, and DeepSeek (MLA/MoE) architectures, tensor shapes, Safetensors weights, and KV cache VRAM directly from HuggingFace or ModelScope metadata.",
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
    let st_opt = if idx_str.trim().is_empty() {
      None
    } else {
      match parse_safetensors_index_json(&idx_str) {
        Ok((tot, wm)) => Some(summarize_index_and_headers(tot, &wm, &[])),
        Err(e) => {
          error_msg.set(Some(e));
          return;
        }
      }
    };
    match build_model_blueprint(
      "custom/pasted-config",
      "Custom Local JSON",
      &cfg_str,
      None,
      None,
      st_opt,
    ) {
      Ok(bp) => {
        error_msg.set(None);
        blueprint.set(Some(bp));
        show_json_drawer.set(false);
      }
      Err(e) => error_msg.set(Some(e)),
    }
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

          div { class: "flex items-center gap-2",
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
              disabled: loading(),
              onclick: move |_| trigger_live_fetch(),
              if loading() {
                Spinner { size: SpinnerSize::Sm, class: "mr-1.5 border-t-current" }
                {lang.pick("拉取解析中...", "Fetching...")}
              } else {
                {lang.pick("实时拉取架构", "Inspect Live")}
              }
            }
          }

          // 3) 可选本地 JSON 粘贴抽屉
          if show_json_drawer() {
            div { class: "mt-4 pt-4 border-t border-slate-200 dark:border-slate-800 grid grid-cols-1 lg:grid-cols-2 gap-3",
              div {
                label { class: "block text-xs font-semibold text-slate-700 dark:text-slate-300 mb-1",
                  "1. config.json（必填，纯前端 WASM 即时解析）"
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
        }

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

#[component]
pub fn ModelVisualizerWorkbench(
  blueprint: ModelBlueprint,
  initial_focus: Option<InspectorFocus>,
  embedded: bool,
) -> Element {
  let default_layer = blueprint.segments.last().map(|s| s.start_layer).unwrap_or(0);
  let mut selected_layer = use_signal(|| default_layer);
  let mut focus = use_signal(|| initial_focus.unwrap_or_else(|| blueprint.default_focus()));
  let mut detail_tab = use_signal(|| DetailTab::VramSimulator);

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

  let kv_ratio = blueprint
    .segment_for_layer(selected_layer())
    .map(|s| s.attention.kv_compression_ratio())
    .unwrap_or(1.0);

  let open_href = format!("/ai/visualizer?repo={}", blueprint.repo_id);

  rsx! {
    div { class: if embedded {
      card_class("p-4 sm:p-6 bg-white dark:bg-slate-950 shadow-sm")
    } else {
      "space-y-6".to_string()
    },
      // 1. 顶部模型身份栏 + 参数分布进度条
      div { class: if embedded { "pb-4 mb-5 border-b border-slate-200 dark:border-slate-800" } else { card_class("p-5 bg-white dark:bg-slate-900/80") },
        div { class: "flex flex-wrap items-start justify-between gap-3",
          div {
            div { class: "flex flex-wrap items-center gap-2",
              h2 { class: "text-lg sm:text-xl font-extrabold text-slate-900 dark:text-white font-mono",
                "{blueprint.repo_id}"
              }
              span { class: badge_class(BadgeVariant::Default, "text-[11px]"),
                "{blueprint.family.badge_label()}"
              }
              span { class: badge_class(BadgeVariant::Secondary, "text-[11px] font-mono"),
                "{blueprint.torch_dtype}"
              }
              if let Some(ref q) = blueprint.quantization_summary {
                span { class: "px-2 py-0.5 rounded-md bg-amber-500/15 text-amber-700 dark:text-amber-300 text-[11px] font-mono font-semibold",
                  "{q}"
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
        div { class: "mt-4 grid grid-cols-2 lg:grid-cols-4 gap-3",
          div { class: "rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/70 dark:bg-slate-950/60 p-3",
            div { class: "text-[11px] font-medium text-slate-500 dark:text-slate-400",
              "总参数量 / 单 Token 激活"
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
            div { class: "text-[11px] font-medium text-slate-500 dark:text-slate-400",
              "主干维度 (Layers × Hidden)"
            }
            div { class: "mt-1 text-base sm:text-lg font-extrabold text-slate-900 dark:text-white font-mono tabular-nums",
              "{blueprint.num_hidden_layers}L × {blueprint.hidden_size}d"
            }
            div { class: "text-[11px] text-slate-500 dark:text-slate-400 font-mono",
              "Vocab: {blueprint.vocab_size}"
            }
          }

          div { class: "rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/70 dark:bg-slate-950/60 p-3",
            div { class: "text-[11px] font-medium text-slate-500 dark:text-slate-400",
              "KV Cache 压缩效率"
            }
            div { class: "mt-1 text-base sm:text-lg font-extrabold text-indigo-600 dark:text-indigo-400 font-mono tabular-nums",
              "{kv_ratio:.1}× vs MHA"
            }
            div { class: "text-[11px] text-slate-500 dark:text-slate-400 font-mono truncate",
              if let Some(seg) = blueprint.segment_for_layer(selected_layer()) {
                "{seg.attention.cached_elements_per_token()} elems/tok/layer"
              }
            }
          }

          div { class: "rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/70 dark:bg-slate-950/60 p-3",
            div { class: "text-[11px] font-medium text-slate-500 dark:text-slate-400",
              "上下文窗口 & 位置编码"
            }
            div { class: "mt-1 text-base sm:text-lg font-extrabold text-slate-900 dark:text-white font-mono tabular-nums",
              "{blueprint.max_position_embeddings / 1024}K ctx"
            }
            div { class: "text-[11px] text-slate-500 dark:text-slate-400 font-mono truncate",
              "{blueprint.rope_summary}"
            }
          }
        }

        // 参数占比彩色堆叠条
        ParamStackedBar { params: blueprint.params.clone() }
      }

      // 2. 双栏联动工作区：左「宏观分层拓扑流」+ 右「算子级内部子图 & 工程检视面板」
      div { class: "grid grid-cols-1 lg:grid-cols-12 gap-5 items-start",
        // 左栏：宏观架构流
        div { class: format!("lg:col-span-5 {}", card_class("p-4 bg-white dark:bg-slate-900/60")),
          div { class: "flex items-center justify-between mb-3",
            span { class: "text-xs font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400",
              "Macro Forward Pass (点击任意模块钻取细节)"
            }
          }
          MacroArchitectureFlow {
            blueprint: blueprint.clone(),
            selected_layer: selected_layer(),
            focus: focus(),
            on_select_layer: move |l| selected_layer.set(l),
            on_select_focus: move |f| focus.set(f),
          }
        }

        // 右栏：微观子图放大镜 + 真实张量核验 + 三合一深度面板
        div { class: "lg:col-span-7 space-y-5",
          // 2A. 算子级内部子图卡片
          div { class: card_class("p-4 sm:p-5 bg-white dark:bg-slate-900/60"),
            div { class: "flex flex-wrap items-center justify-between gap-2 pb-3 mb-4 border-b border-slate-200 dark:border-slate-800",
              div {
                span { class: "text-xs font-bold uppercase tracking-wider text-indigo-600 dark:text-indigo-400",
                  "Operator Subgraph · Layer #{selected_layer()}"
                }
              }
              // 快捷切换子模块焦点按钮
              div { class: "flex flex-wrap gap-1",
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
                for (f_opt, f_label) in [
                  (InspectorFocus::Embedding, "Embed"),
                  (InspectorFocus::Attention, "Attention"),
                  (InspectorFocus::FfnOrMoe, if blueprint.has_moe() { "FFN / MoE" } else { "SwiGLU FFN" }),
                  (InspectorFocus::NormAndResidual, "RMSNorm"),
                  (InspectorFocus::LmHead, "LM Head"),
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
            }

            // 当前选中子模块关联的 L2 Safetensors 真实权重对照表
            {
              let matched = blueprint.matching_tensors(focus());
              if !matched.is_empty() {
                rsx! {
                  div { class: "mt-4 pt-3 border-t border-slate-200 dark:border-slate-800",
                    div { class: "text-xs font-semibold text-slate-700 dark:text-slate-300 mb-2 flex items-center justify-between",
                      span { "匹配该子模块的真实 Safetensors 权重张量 ({matched.len()} 项)" }
                      span { class: "text-[11px] font-mono text-slate-400", "Shape = [out, in]" }
                    }
                    TensorPatternTable { items: matched.into_iter().cloned().collect() }
                  }
                }
              } else {
                rsx! {}
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
                let saved_kv = (est.mha_baseline_kv_bytes - est.kv_cache_bytes).max(0.0);
                let w_pct = (est.weights_bytes / est.total_bytes.max(1.0) * 100.0).clamp(2.0, 96.0);
                let kv_pct = (est.kv_cache_bytes / est.total_bytes.max(1.0) * 100.0).clamp(1.0, 96.0);
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
                          "上下文长度 (Context)"
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
                          title: "KV Cache",
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
                            "KV Cache: {format_bytes_iec(est.kv_cache_bytes)}"
                          }
                        }
                        div { class: "flex items-center gap-2",
                          span { class: "w-2.5 h-2.5 rounded-xs bg-slate-400 shrink-0" }
                          span { class: "text-slate-600 dark:text-slate-300",
                            "激活/工作区: {format_bytes_iec(est.activation_overhead_bytes)}"
                          }
                        }
                      }

                      if saved_kv > 1024.0 * 1024.0 {
                        div { class: "mt-3 pt-2.5 border-t border-slate-200 dark:border-slate-800 text-xs text-emerald-700 dark:text-emerald-400 font-medium",
                          "★ 架构收益：在当前 B={batch_size()}、Ctx={ctx_len()} 配置下，相比标准多头注意力 (MHA 需 {format_bytes_iec(est.mha_baseline_kv_bytes)}) 节省了 {format_bytes_iec(saved_kv)} KV 显存！"
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
                          ("embedding", "Embedding"),
                          ("attention", "Attention"),
                          ("ffn", "Dense FFN"),
                          ("moe", "Sparse MoE"),
                          ("norm", "RMSNorm"),
                          ("lm_head", "LM Head"),
                        ] {
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
                rsx! {
                  div { class: "space-y-3",
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

                    pre { class: "max-h-72 overflow-auto rounded-xl bg-slate-950 text-slate-100 p-3.5 text-xs font-mono leading-relaxed border border-slate-800",
                      "{blueprint.raw_config_pretty}"
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
fn ParamStackedBar(params: crate::arch::ParamBreakdown) -> Element {
  let tot = params.total_params.max(1) as f64;
  let slices = [
    ("Vision", params.vision_encoder, "bg-fuchsia-500"),
    ("Embedding", params.embedding, "bg-emerald-500"),
    ("Attention", params.attention, "bg-indigo-500"),
    ("Dense FFN", params.dense_ffn, "bg-sky-500"),
    ("MoE Routed", params.moe_routed, "bg-amber-500"),
    ("MoE Shared", params.moe_shared, "bg-teal-500"),
    ("LM Head", params.lm_head, "bg-rose-500"),
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
      div { class: "mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[11px] font-mono text-slate-600 dark:text-slate-400",
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
    div { class: "max-h-64 overflow-auto rounded-lg border border-slate-200 dark:border-slate-800",
      table { class: "w-full text-left border-collapse text-xs font-mono",
        thead { class: "bg-slate-100 dark:bg-slate-800/90 text-[11px] text-slate-600 dark:text-slate-300 sticky top-0",
          tr {
            th { class: "py-2 px-2.5 font-semibold", "Tensor Pattern" }
            th { class: "py-2 px-2 font-semibold", "Shape" }
            th { class: "py-2 px-2 font-semibold", "Dtype" }
            th { class: "py-2 px-2 font-semibold text-right", "Count" }
            th { class: "py-2 px-2.5 font-semibold text-right", "Params" }
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
                  td { class: "py-1.5 px-2.5 text-slate-800 dark:text-slate-200 break-all",
                    title: item.example_shard.clone().unwrap_or_default(),
                    "{item.pattern}"
                  }
                  td { class: "py-1.5 px-2 text-indigo-600 dark:text-indigo-400 whitespace-nowrap",
                    "{shape_str}"
                  }
                  td { class: "py-1.5 px-2 text-slate-600 dark:text-slate-400 whitespace-nowrap",
                    "{dt_str}"
                  }
                  td { class: "py-1.5 px-2 text-right text-slate-500 tabular-nums whitespace-nowrap",
                    "×{item.repeat_count}"
                  }
                  td { class: "py-1.5 px-2.5 text-right text-slate-700 dark:text-slate-300 tabular-nums whitespace-nowrap",
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

fn clean_server_err(raw: &str) -> String {
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
}
