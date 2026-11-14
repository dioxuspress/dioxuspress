//! 纯 Rust 声明式架构拓扑图与算子级内部子图（SVG）。
//!
//! 特点：
//! - 零外部 JS 图表运行时依赖，SSR 首屏直出，水合零抖动，严格兼容无 `'unsafe-eval'` CSP；
//! - 所有配色采用 Tailwind 亮/暗双模语义类名，切换暗色主题即时响应；
//! - 左侧宏观流图（`MacroArchitectureFlow`）与右侧算子级放大镜（`MicroSubgraphDiagram`）点击联动。

use dioxus::prelude::*;

use crate::arch::{
  format_count, AttentionSpec, FfnSpec, InspectorFocus, LayerSegment, ModelBlueprint,
};

/// 左侧宏观纵向架构拓扑图（含残差旁路、异构层段切换与点击高亮）。
#[component]
pub fn MacroArchitectureFlow(
  blueprint: ModelBlueprint,
  selected_layer: u64,
  focus: InspectorFocus,
  on_select_layer: EventHandler<u64>,
  on_select_focus: EventHandler<InspectorFocus>,
) -> Element {
  let active_seg = blueprint.segment_for_layer(selected_layer).cloned();
  let d = blueprint.hidden_size;
  let v = blueprint.vocab_size;

  let card_cls = |target: InspectorFocus| {
    let base = "w-full text-left rounded-xl border p-3 transition-all cursor-pointer ";
    if focus == target {
      format!(
        "{base}border-indigo-500 bg-indigo-50/80 dark:bg-indigo-950/50 ring-2 ring-indigo-500/30 shadow-sm"
      )
    } else {
      format!(
        "{base}border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 hover:border-indigo-400 dark:hover:border-indigo-600"
      )
    }
  };

  rsx! {
    div { class: "flex flex-col items-center select-none",
      // 0. 输入 Token 序列胶囊
      div { class: "px-3.5 py-1 rounded-full bg-slate-100 dark:bg-slate-800 border border-slate-300 dark:border-slate-700 text-xs font-mono text-slate-600 dark:text-slate-300",
        "input_ids : int64 [B, S]"
      }
      VerticalArrow { label: "" }

      // 1. Token Embedding 节点
      button {
        r#type: "button",
        class: card_cls(InspectorFocus::Embedding),
        onclick: move |_| on_select_focus.call(InspectorFocus::Embedding),
        div { class: "flex items-center justify-between gap-2",
          span { class: "text-xs font-bold uppercase tracking-wider text-emerald-600 dark:text-emerald-400",
            "Token Embedding"
          }
          span { class: "text-[11px] font-mono text-slate-500 dark:text-slate-400",
            "{format_count(blueprint.params.embedding)}"
          }
        }
        div { class: "mt-1 text-sm font-semibold text-slate-900 dark:text-white font-mono",
          "embed_tokens [{v}, {d}]"
        }
      }

      VerticalArrow { label: &format!("h_0 : [B, S, {d}]") }

      // 2. 异构层段选择器（当模型如 DeepSeek-V3 同时具有前置 Dense 层与后续 MoE 层时尤为直观）
      if blueprint.segments.len() > 1 {
        div { class: "w-full mb-2 flex flex-wrap gap-1.5 justify-center",
          for seg in blueprint.segments.iter() {
            {
              let start = seg.start_layer;
              let is_in_seg = selected_layer >= seg.start_layer && selected_layer <= seg.end_layer;
              let label = seg.range_label();
              let pill_cls = if is_in_seg {
                "px-2.5 py-1 rounded-lg text-xs font-semibold bg-indigo-600 text-white shadow-xs cursor-pointer"
              } else {
                "px-2.5 py-1 rounded-lg text-xs font-medium bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700 cursor-pointer"
              };
              rsx! {
                button {
                  key: "{start}",
                  r#type: "button",
                  class: pill_cls,
                  onclick: move |_| on_select_layer.call(start),
                  "{label}"
                }
              }
            }
          }
        }
      }

      // 3. Transformer 主干层块（带双残差回路可视化容器）
      if let Some(seg) = active_seg {
        div { class: "w-full rounded-2xl border-2 border-dashed border-slate-300 dark:border-slate-700 bg-slate-50/70 dark:bg-slate-900/40 p-3.5",
          // 层头与层号步进器
          div { class: "flex items-center justify-between gap-2 mb-3 pb-2 border-b border-slate-200 dark:border-slate-800",
            div {
              div { class: "text-xs font-bold text-slate-800 dark:text-slate-100",
                "{seg.title}"
              }
              div { class: "text-[11px] font-mono text-slate-500 dark:text-slate-400",
                "{seg.range_label()} · 当前观测 Layer #{selected_layer}"
              }
            }
            if blueprint.num_hidden_layers > 1 {
              div { class: "flex items-center gap-1",
                button {
                  r#type: "button",
                  class: "w-6 h-6 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800 text-xs font-bold text-slate-700 dark:text-slate-200 disabled:opacity-40 cursor-pointer",
                  disabled: selected_layer == 0,
                  title: "上一层",
                  onclick: move |_| on_select_layer.call(selected_layer.saturating_sub(1)),
                  "−"
                }
                span { class: "px-1.5 text-xs font-mono font-semibold text-indigo-600 dark:text-indigo-400 tabular-nums",
                  "{selected_layer}/{blueprint.num_hidden_layers.saturating_sub(1)}"
                }
                button {
                  r#type: "button",
                  class: "w-6 h-6 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800 text-xs font-bold text-slate-700 dark:text-slate-200 disabled:opacity-40 cursor-pointer",
                  disabled: selected_layer + 1 >= blueprint.num_hidden_layers,
                  title: "下一层",
                  onclick: move |_| {
                    let next = (selected_layer + 1).min(blueprint.num_hidden_layers.saturating_sub(1));
                    on_select_layer.call(next);
                  },
                  "+"
                }
              }
            }
          }

          // 第一残差子块：Pre-Attn Norm + Self-Attention / MLA
          div { class: "relative pl-5 border-l-2 border-indigo-400/70 dark:border-indigo-500/60 space-y-2 py-1",
            span { class: "absolute -left-[9px] top-0 w-4 h-4 rounded-full bg-indigo-100 dark:bg-indigo-950 border border-indigo-400 text-[9px] flex items-center justify-center font-mono text-indigo-700 dark:text-indigo-300",
              "x"
            }
            span { class: "text-[10px] font-mono text-indigo-500 dark:text-indigo-400 block -mt-1",
              "Residual Skip #1"
            }

            button {
              r#type: "button",
              class: card_cls(InspectorFocus::NormAndResidual),
              onclick: move |_| on_select_focus.call(InspectorFocus::NormAndResidual),
              div { class: "flex items-center justify-between text-xs",
                span { class: "font-semibold text-slate-700 dark:text-slate-200 font-mono",
                  "input_layernorm (RMSNorm)"
                }
                span { class: "font-mono text-[11px] text-slate-400", "[{d}]" }
              }
            }

            button {
              r#type: "button",
              class: card_cls(InspectorFocus::Attention),
              onclick: move |_| on_select_focus.call(InspectorFocus::Attention),
              div { class: "flex items-center justify-between gap-2",
                span { class: "text-xs font-bold uppercase tracking-wider text-indigo-600 dark:text-indigo-400",
                  "Self-Attention"
                }
                span { class: "px-1.5 py-0.5 rounded bg-indigo-500/10 text-indigo-700 dark:text-indigo-300 text-[10px] font-mono font-semibold",
                  "KV ÷{seg.attention.kv_compression_ratio():.1}×"
                }
              }
              div { class: "mt-1 text-xs font-semibold text-slate-900 dark:text-white font-mono",
                "{seg.attention.short_label()}"
              }
              div { class: "mt-0.5 text-[11px] text-slate-500 dark:text-slate-400 font-mono",
                "{blueprint.rope_summary} · {format_count(seg.attention.params_per_layer(d))}/layer"
              }
            }

            div { class: "flex items-center gap-2 pt-0.5",
              span { class: "inline-flex items-center justify-center w-5 h-5 rounded-full bg-indigo-600 text-white text-xs font-bold shadow-xs",
                "+"
              }
              span { class: "text-[11px] font-mono text-slate-500 dark:text-slate-400",
                "h_mid = x + SelfAttn(RMSNorm(x)) : [B, S, {d}]"
              }
            }
          }

          div { class: "h-3 w-0.5 bg-slate-300 dark:bg-slate-700 mx-auto my-1" }

          // 第二残差子块：Post-Attn Norm + Dense FFN / Sparse MoE
          div { class: "relative pl-5 border-l-2 border-amber-400/80 dark:border-amber-500/60 space-y-2 py-1",
            span { class: "absolute -left-[9px] top-0 w-4 h-4 rounded-full bg-amber-100 dark:bg-amber-950 border border-amber-400 text-[9px] flex items-center justify-center font-mono text-amber-700 dark:text-amber-300",
              "h"
            }
            span { class: "text-[10px] font-mono text-amber-600 dark:text-amber-400 block -mt-1",
              "Residual Skip #2"
            }

            button {
              r#type: "button",
              class: card_cls(InspectorFocus::NormAndResidual),
              onclick: move |_| on_select_focus.call(InspectorFocus::NormAndResidual),
              div { class: "flex items-center justify-between text-xs",
                span { class: "font-semibold text-slate-700 dark:text-slate-200 font-mono",
                  "post_attention_layernorm (RMSNorm)"
                }
                span { class: "font-mono text-[11px] text-slate-400", "[{d}]" }
              }
            }

            {
              let (ffn_tot, ffn_act) = seg.ffn.params_per_layer(d);
              let is_moe_seg = matches!(seg.ffn, FfnSpec::SparseMoe { .. });
              rsx! {
                button {
                  r#type: "button",
                  class: card_cls(InspectorFocus::FfnOrMoe),
                  onclick: move |_| on_select_focus.call(InspectorFocus::FfnOrMoe),
                  div { class: "flex items-center justify-between gap-2",
                    span { class: if is_moe_seg {
                      "text-xs font-bold uppercase tracking-wider text-amber-600 dark:text-amber-400"
                    } else {
                      "text-xs font-bold uppercase tracking-wider text-sky-600 dark:text-sky-400"
                    },
                      if is_moe_seg { "Sparse MoE Feed-Forward" } else { "Gated MLP (SwiGLU)" }
                    }
                    span { class: "text-[11px] font-mono text-slate-500 dark:text-slate-400",
                      if ffn_tot != ffn_act {
                        "{format_count(ffn_act)} act / {format_count(ffn_tot)} tot"
                      } else {
                        "{format_count(ffn_tot)}/layer"
                      }
                    }
                  }
                  div { class: "mt-1 text-xs font-semibold text-slate-900 dark:text-white font-mono",
                    "{seg.ffn.short_label()}"
                  }
                }
              }
            }

            div { class: "flex items-center gap-2 pt-0.5",
              span { class: "inline-flex items-center justify-center w-5 h-5 rounded-full bg-amber-500 text-white text-xs font-bold shadow-xs",
                "+"
              }
              span { class: "text-[11px] font-mono text-slate-500 dark:text-slate-400",
                "h_out = h_mid + FFN(RMSNorm(h_mid)) : [B, S, {d}]"
              }
            }
          }
        }
      }

      // 4. 可选 MTP 多 Token 预测头（DeepSeek-V3 特色模块）
      if blueprint.num_nextn_predict_layers > 0 {
        VerticalArrow { label: "MTP Auxiliary Branch" }
        button {
          r#type: "button",
          class: card_cls(InspectorFocus::MtpHead),
          onclick: move |_| on_select_focus.call(InspectorFocus::MtpHead),
          div { class: "flex items-center justify-between gap-2",
            span { class: "text-xs font-bold uppercase tracking-wider text-purple-600 dark:text-purple-400",
              "Multi-Token Prediction (MTP ×{blueprint.num_nextn_predict_layers})"
            }
            span { class: "text-[11px] font-mono text-slate-500 dark:text-slate-400",
              "+{format_count(blueprint.params.mtp_head)}"
            }
          }
          div { class: "mt-1 text-xs font-mono text-slate-700 dark:text-slate-300",
            "eh_proj [2×{d} → {d}] + Transformer Block (Speculative Decoding)"
          }
        }
      }

      VerticalArrow { label: &format!("h_final : [B, S, {d}]") }

      // 5. Final RMSNorm
      button {
        r#type: "button",
        class: card_cls(InspectorFocus::NormAndResidual),
        onclick: move |_| on_select_focus.call(InspectorFocus::NormAndResidual),
        div { class: "flex items-center justify-between text-xs",
          span { class: "font-semibold text-slate-800 dark:text-slate-100 font-mono",
            "model.norm (Final RMSNorm)"
          }
          span { class: "font-mono text-slate-500 dark:text-slate-400", "[{d}]" }
        }
      }

      VerticalArrow { label: "" }

      // 6. LM Head 输出投影
      button {
        r#type: "button",
        class: card_cls(InspectorFocus::LmHead),
        onclick: move |_| on_select_focus.call(InspectorFocus::LmHead),
        div { class: "flex items-center justify-between gap-2",
          span { class: "text-xs font-bold uppercase tracking-wider text-rose-600 dark:text-rose-400",
            "Output Projection (LM Head)"
          }
          span { class: "text-[11px] font-mono text-slate-500 dark:text-slate-400",
            if blueprint.tie_word_embeddings {
              "0 (Tied w/ Embed)"
            } else {
              "{format_count(blueprint.params.lm_head)}"
            }
          }
        }
        div { class: "mt-1 text-sm font-semibold text-slate-900 dark:text-white font-mono",
          "lm_head [{d} → {v}]"
        }
      }

      VerticalArrow { label: "" }
      div { class: "px-3.5 py-1 rounded-full bg-rose-50 dark:bg-rose-950/50 border border-rose-300 dark:border-rose-800 text-xs font-mono font-semibold text-rose-700 dark:text-rose-300",
        "logits : float [B, S, {v}]"
      }
    }
  }
}

#[component]
fn VerticalArrow(label: String) -> Element {
  rsx! {
    div { class: "flex flex-col items-center my-1",
      div { class: "h-3 w-0.5 bg-slate-300 dark:bg-slate-700" }
      if !label.is_empty() {
        span { class: "text-[10px] font-mono text-slate-400 dark:text-slate-500 my-0.5",
          "{label}"
        }
        div { class: "h-2 w-0.5 bg-slate-300 dark:bg-slate-700" }
      }
      div { class: "w-0 h-0 border-l-4 border-r-4 border-t-[6px] border-l-transparent border-r-transparent border-t-slate-400 dark:border-t-slate-600" }
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 右侧算子级内部子图（SVG）
// ─────────────────────────────────────────────────────────────────────────────

/// 根据用户点击选中的 [`InspectorFocus`] 与当前激活层段，绘制精确带维度标注的内部算子拓扑 SVG。
#[component]
pub fn MicroSubgraphDiagram(
  blueprint: ModelBlueprint,
  selected_layer: u64,
  focus: InspectorFocus,
) -> Element {
  let seg_opt: Option<LayerSegment> = blueprint.segment_for_layer(selected_layer).cloned();
  let d = blueprint.hidden_size;
  let v = blueprint.vocab_size;

  match focus {
    InspectorFocus::Attention => match seg_opt.map(|s| s.attention) {
      Some(AttentionSpec::StandardGqa {
        num_q_heads,
        num_kv_heads,
        head_dim,
        qkv_bias,
        o_bias,
        qk_norm,
        sliding_window,
      }) => rsx! {
        GqaSubgraphSvg {
          hidden_size: d,
          num_q_heads,
          num_kv_heads,
          head_dim,
          qkv_bias,
          o_bias,
          qk_norm,
          sliding_window,
          rope_label: blueprint.rope_summary.clone(),
        }
      },
      Some(AttentionSpec::DeepSeekMla {
        num_heads,
        q_lora_rank,
        kv_lora_rank,
        qk_nope_head_dim,
        qk_rope_head_dim,
        v_head_dim,
      }) => rsx! {
        MlaSubgraphSvg {
          hidden_size: d,
          num_heads,
          q_lora_rank,
          kv_lora_rank,
          qk_nope_head_dim,
          qk_rope_head_dim,
          v_head_dim,
        }
      },
      None => rsx! { EmptyDiagramNote {} },
    },
    InspectorFocus::FfnOrMoe => match seg_opt.map(|s| s.ffn) {
      Some(FfnSpec::DenseGated { intermediate_size, act_fn, bias }) => rsx! {
        DenseSwiGluSvg { hidden_size: d, intermediate_size, act_fn, bias }
      },
      Some(FfnSpec::SparseMoe {
        num_routed_experts,
        top_k,
        moe_intermediate_size,
        num_shared_experts,
        shared_intermediate_size,
        router_groups,
        scoring_func,
        act_fn,
        ..
      }) => rsx! {
        SparseMoeSvg {
          hidden_size: d,
          num_routed_experts,
          top_k,
          moe_intermediate_size,
          num_shared_experts,
          shared_intermediate_size,
          router_groups,
          scoring_func,
          act_fn,
        }
      },
      None => rsx! { EmptyDiagramNote {} },
    },
    InspectorFocus::Embedding => rsx! {
      EndpointSubgraphSvg {
        mode: "embedding".to_string(),
        vocab_size: v,
        hidden_size: d,
        tied: blueprint.tie_word_embeddings,
        eps: blueprint.rms_norm_eps.unwrap_or(1e-6),
      }
    },
    InspectorFocus::LmHead => rsx! {
      EndpointSubgraphSvg {
        mode: "lm_head".to_string(),
        vocab_size: v,
        hidden_size: d,
        tied: blueprint.tie_word_embeddings,
        eps: blueprint.rms_norm_eps.unwrap_or(1e-6),
      }
    },
    InspectorFocus::NormAndResidual => rsx! {
      EndpointSubgraphSvg {
        mode: "rmsnorm".to_string(),
        vocab_size: v,
        hidden_size: d,
        tied: blueprint.tie_word_embeddings,
        eps: blueprint.rms_norm_eps.unwrap_or(1e-6),
      }
    },
    InspectorFocus::MtpHead => rsx! {
      EndpointSubgraphSvg {
        mode: "mtp".to_string(),
        vocab_size: v,
        hidden_size: d,
        tied: blueprint.tie_word_embeddings,
        eps: blueprint.rms_norm_eps.unwrap_or(1e-6),
      }
    },
  }
}

#[component]
fn EmptyDiagramNote() -> Element {
  rsx! {
    div { class: "p-6 text-center text-sm text-slate-500 dark:text-slate-400",
      "当前模型未包含标准 Transformer 层参数，请通过下方 Safetensors 权重树检视真实张量分布。"
    }
  }
}

/// 标准 MHA / GQA / MQA + 可选 QK-Norm (Qwen3) 算子级精确流图。
#[component]
fn GqaSubgraphSvg(
  hidden_size: u64,
  num_q_heads: u64,
  num_kv_heads: u64,
  head_dim: u64,
  qkv_bias: bool,
  o_bias: bool,
  qk_norm: bool,
  sliding_window: Option<u64>,
  rope_label: String,
) -> Element {
  let q_out = num_q_heads * head_dim;
  let kv_out = num_kv_heads * head_dim;
  let group_size = (num_q_heads / num_kv_heads.max(1)).max(1);
  let bias_tag = if qkv_bias { "+bias" } else { "no bias" };
  let o_bias_tag = if o_bias { "+bias" } else { "no bias" };
  let mask_tag = match sliding_window {
    Some(w) => format!("Causal SWA (W={w})"),
    None => "Causal Mask".to_string(),
  };

  rsx! {
    div { class: "w-full overflow-x-auto",
      svg {
        view_box: "0 0 680 350",
        class: "w-full min-w-[540px] h-auto font-mono",
        role: "img",
        "aria-label": "Grouped Query Attention internal operator graph",

        // 顶部输入张量
        rect { x: "230", y: "10", width: "220", height: "30", rx: "8", class: "fill-slate-100 dark:fill-slate-800 stroke-slate-300 dark:stroke-slate-600", stroke_width: "1.5" }
        text { x: "340", y: "29", text_anchor: "middle", font_size: "11", class: "fill-slate-800 dark:fill-slate-100 font-semibold",
          "x_norm [B, S, {hidden_size}]"
        }

        // 三路分发连线
        path { d: "M 340 40 L 340 54 M 115 54 L 565 54 M 115 54 L 115 68 M 340 54 L 340 68 M 565 54 L 565 68", class: "stroke-slate-400 dark:stroke-slate-500", stroke_width: "1.5", fill: "none" }

        // Q 投影框
        rect { x: "25", y: "68", width: "180", height: "44", rx: "8", class: "fill-indigo-50 dark:fill-indigo-950/70 stroke-indigo-500", stroke_width: "1.5" }
        text { x: "115", y: "86", text_anchor: "middle", font_size: "11", class: "fill-indigo-700 dark:fill-indigo-300 font-bold",
          "q_proj ({bias_tag})"
        }
        text { x: "115", y: "103", text_anchor: "middle", font_size: "10", class: "fill-slate-600 dark:fill-slate-300",
          "[{hidden_size} → {q_out}] ({num_q_heads}×{head_dim})"
        }

        // K 投影框
        rect { x: "250", y: "68", width: "180", height: "44", rx: "8", class: "fill-sky-50 dark:fill-sky-950/70 stroke-sky-500", stroke_width: "1.5" }
        text { x: "340", y: "86", text_anchor: "middle", font_size: "11", class: "fill-sky-700 dark:fill-sky-300 font-bold",
          "k_proj ({bias_tag})"
        }
        text { x: "340", y: "103", text_anchor: "middle", font_size: "10", class: "fill-slate-600 dark:fill-slate-300",
          "[{hidden_size} → {kv_out}] ({num_kv_heads}×{head_dim})"
        }

        // V 投影框
        rect { x: "475", y: "68", width: "180", height: "44", rx: "8", class: "fill-teal-50 dark:fill-teal-950/70 stroke-teal-500", stroke_width: "1.5" }
        text { x: "565", y: "86", text_anchor: "middle", font_size: "11", class: "fill-teal-700 dark:fill-teal-300 font-bold",
          "v_proj ({bias_tag})"
        }
        text { x: "565", y: "103", text_anchor: "middle", font_size: "10", class: "fill-slate-600 dark:fill-slate-300",
          "[{hidden_size} → {kv_out}] ({num_kv_heads}×{head_dim})"
        }

        // QK-Norm + RoPE 阶段
        line { x1: "115", y1: "112", x2: "115", y2: "130", class: "stroke-indigo-400", stroke_width: "1.5" }
        line { x1: "340", y1: "112", x2: "340", y2: "130", class: "stroke-sky-400", stroke_width: "1.5" }
        line { x1: "565", y1: "112", x2: "565", y2: "182", class: "stroke-teal-400", stroke_width: "1.5" }

        rect { x: "25", y: "130", width: "180", height: "38", rx: "8", class: if qk_norm { "fill-emerald-50 dark:fill-emerald-950/60 stroke-emerald-500" } else { "fill-slate-50 dark:fill-slate-800/80 stroke-slate-400" }, stroke_width: "1.5" }
        text { x: "115", y: "146", text_anchor: "middle", font_size: "10.5", class: "fill-slate-800 dark:fill-slate-100 font-semibold",
          if qk_norm { "q_norm [{head_dim}] + RoPE" } else { "Reshape + RoPE" }
        }
        text { x: "115", y: "160", text_anchor: "middle", font_size: "9.5", class: "fill-slate-500 dark:fill-slate-400",
          "{rope_label}"
        }

        rect { x: "250", y: "130", width: "180", height: "38", rx: "8", class: if qk_norm { "fill-emerald-50 dark:fill-emerald-950/60 stroke-emerald-500" } else { "fill-slate-50 dark:fill-slate-800/80 stroke-slate-400" }, stroke_width: "1.5" }
        text { x: "340", y: "146", text_anchor: "middle", font_size: "10.5", class: "fill-slate-800 dark:fill-slate-100 font-semibold",
          if qk_norm { "k_norm [{head_dim}] + RoPE" } else { "Reshape + RoPE" }
        }
        text { x: "340", y: "160", text_anchor: "middle", font_size: "9.5", class: "fill-slate-500 dark:fill-slate-400",
          "Q:KV Group = {group_size}:1"
        }

        // KV Cache 驻留槽
        line { x1: "340", y1: "168", x2: "340", y2: "182", class: "stroke-sky-400", stroke_width: "1.5" }
        rect { x: "250", y: "182", width: "405", height: "34", rx: "8", class: "fill-amber-50 dark:fill-amber-950/50 stroke-amber-500", stroke_dasharray: "5 3", stroke_width: "1.5" }
        text { x: "452", y: "203", text_anchor: "middle", font_size: "10.5", class: "fill-amber-800 dark:fill-amber-300 font-bold",
          "★ KV Cache (K & V): 2 × [B, {num_kv_heads}, S, {head_dim}] ({2 * kv_out} elems/tok/layer)"
        }

        // 汇入 SDPA 核心算子
        path { d: "M 115 168 L 115 245 L 200 245 M 452 216 L 452 245 L 480 245", class: "stroke-slate-400 dark:stroke-slate-500", stroke_width: "1.5", fill: "none" }
        line { x1: "452", y1: "216", x2: "340", y2: "230", class: "stroke-amber-500", stroke_width: "1.5" }

        rect { x: "160", y: "230", width: "360", height: "42", rx: "10", class: "fill-indigo-600 dark:fill-indigo-600 stroke-indigo-400", stroke_width: "1.5" }
        text { x: "340", y: "248", text_anchor: "middle", font_size: "11", class: "fill-white font-bold",
          "Scaled Dot-Product Attention (1/√{head_dim} · {mask_tag})"
        }
        text { x: "340", y: "263", text_anchor: "middle", font_size: "10", class: "fill-indigo-100",
          "Softmax(Q · Kᵀ) · V → Concat [B, S, {q_out}]"
        }

        // o_proj 输出
        line { x1: "340", y1: "272", x2: "340", y2: "292", class: "stroke-indigo-400", stroke_width: "1.5" }
        rect { x: "210", y: "292", width: "260", height: "42", rx: "8", class: "fill-slate-100 dark:fill-slate-800 stroke-slate-400 dark:stroke-slate-600", stroke_width: "1.5" }
        text { x: "340", y: "310", text_anchor: "middle", font_size: "11", class: "fill-slate-900 dark:fill-white font-bold",
          "o_proj [{q_out} → {hidden_size}] ({o_bias_tag})"
        }
        text { x: "340", y: "325", text_anchor: "middle", font_size: "10", class: "fill-slate-500 dark:fill-slate-400",
          "Output: [B, S, {hidden_size}] → Residual Add (+)"
        }
      }
    }
  }
}

/// DeepSeek-V2 / V3 / R1 专属 Multi-head Latent Attention (MLA) 潜空间压缩与解耦 RoPE 子图。
#[component]
fn MlaSubgraphSvg(
  hidden_size: u64,
  num_heads: u64,
  q_lora_rank: Option<u64>,
  kv_lora_rank: u64,
  qk_nope_head_dim: u64,
  qk_rope_head_dim: u64,
  v_head_dim: u64,
) -> Element {
  let qk_dim = qk_nope_head_dim + qk_rope_head_dim;
  let q_total = num_heads * qk_dim;
  let kv_a_out = kv_lora_rank + qk_rope_head_dim;
  let kv_b_out = num_heads * (qk_nope_head_dim + v_head_dim);
  let o_in = num_heads * v_head_dim;
  let mha_elems = 2 * num_heads * v_head_dim;
  let ratio = (mha_elems as f64) / (kv_a_out.max(1) as f64);

  rsx! {
    div { class: "w-full overflow-x-auto",
      svg {
        view_box: "0 0 680 390",
        class: "w-full min-w-[560px] h-auto font-mono",
        role: "img",
        "aria-label": "DeepSeek Multi-head Latent Attention internal operator graph",

        // 输入
        rect { x: "230", y: "8", width: "220", height: "28", rx: "8", class: "fill-slate-100 dark:fill-slate-800 stroke-slate-300 dark:stroke-slate-600", stroke_width: "1.5" }
        text { x: "340", y: "26", text_anchor: "middle", font_size: "11", class: "fill-slate-800 dark:fill-slate-100 font-semibold",
          "x_norm [B, S, {hidden_size}]"
        }

        path { d: "M 340 36 L 340 48 M 165 48 L 505 48 M 165 48 L 165 62 M 505 48 L 505 62", class: "stroke-slate-400 dark:stroke-slate-500", stroke_width: "1.5", fill: "none" }

        // ── 左支路：Query LoRA 压缩 + 解耦 RoPE ──
        if let Some(q_rank) = q_lora_rank {
          rect { x: "35", y: "62", width: "260", height: "42", rx: "8", class: "fill-indigo-50 dark:fill-indigo-950/70 stroke-indigo-500", stroke_width: "1.5" }
          text { x: "165", y: "80", text_anchor: "middle", font_size: "10.5", class: "fill-indigo-700 dark:fill-indigo-300 font-bold",
            "q_a_proj [{hidden_size} → {q_rank}] + RMSNorm"
          }
          text { x: "165", y: "95", text_anchor: "middle", font_size: "9.5", class: "fill-slate-500 dark:fill-slate-400",
            "Query Latent c_Q : [B, S, {q_rank}]"
          }
          line { x1: "165", y1: "104", x2: "165", y2: "120", class: "stroke-indigo-400", stroke_width: "1.5" }
          rect { x: "35", y: "120", width: "260", height: "40", rx: "8", class: "fill-indigo-50 dark:fill-indigo-950/70 stroke-indigo-500", stroke_width: "1.5" }
          text { x: "165", y: "137", text_anchor: "middle", font_size: "10.5", class: "fill-indigo-700 dark:fill-indigo-300 font-bold",
            "q_b_proj [{q_rank} → {q_total}] ({num_heads}×{qk_dim})"
          }
          text { x: "165", y: "151", text_anchor: "middle", font_size: "9.5", class: "fill-slate-600 dark:fill-slate-300",
            "Split → q_nope[{qk_nope_head_dim}] + RoPE(q_pe[{qk_rope_head_dim}])"
          }
        } else {
          rect { x: "35", y: "85", width: "260", height: "54", rx: "8", class: "fill-indigo-50 dark:fill-indigo-950/70 stroke-indigo-500", stroke_width: "1.5" }
          text { x: "165", y: "107", text_anchor: "middle", font_size: "11", class: "fill-indigo-700 dark:fill-indigo-300 font-bold",
            "q_proj [{hidden_size} → {q_total}]"
          }
          text { x: "165", y: "125", text_anchor: "middle", font_size: "9.5", class: "fill-slate-600 dark:fill-slate-300",
            "Split → q_nope[{qk_nope_head_dim}] + RoPE(q_pe[{qk_rope_head_dim}])"
          }
        }

        // ── 右支路：KV 联合潜空间压缩（MLA 省显存核心！）──
        rect { x: "355", y: "62", width: "300", height: "42", rx: "8", class: "fill-emerald-50 dark:fill-emerald-950/70 stroke-emerald-500", stroke_width: "1.5" }
        text { x: "505", y: "80", text_anchor: "middle", font_size: "10.5", class: "fill-emerald-700 dark:fill-emerald-300 font-bold",
          "kv_a_proj_with_mqa [{hidden_size} → {kv_a_out}]"
        }
        text { x: "505", y: "95", text_anchor: "middle", font_size: "9.5", class: "fill-slate-600 dark:fill-slate-300",
          "Split: c_KV [{kv_lora_rank}] + Shared k_pe [{qk_rope_head_dim}]"
        }

        // 唯一进入 KV Cache 的压缩潜向量高亮框
        line { x1: "505", y1: "104", x2: "505", y2: "120", class: "stroke-amber-500", stroke_width: "1.5" }
        rect { x: "355", y: "120", width: "300", height: "50", rx: "8", class: "fill-amber-50 dark:fill-amber-950/60 stroke-amber-500", stroke_dasharray: "5 3", stroke_width: "1.8" }
        text { x: "505", y: "139", text_anchor: "middle", font_size: "10.5", class: "fill-amber-800 dark:fill-amber-300 font-bold",
          "★ Latent KV Cache: ONLY {kv_a_out} dims/tok!"
        }
        text { x: "505", y: "156", text_anchor: "middle", font_size: "9.5", class: "fill-amber-700 dark:fill-amber-400",
          "RMSNorm(c_KV[{kv_lora_rank}]) + RoPE(k_pe[{qk_rope_head_dim}]) · {ratio:.1}× vs MHA"
        }

        // kv_b_proj 上投影还原（或推理期矩阵吸收）
        line { x1: "505", y1: "170", x2: "505", y2: "188", class: "stroke-emerald-500", stroke_width: "1.5" }
        rect { x: "355", y: "188", width: "300", height: "44", rx: "8", class: "fill-emerald-50 dark:fill-emerald-950/70 stroke-emerald-500", stroke_width: "1.5" }
        text { x: "505", y: "206", text_anchor: "middle", font_size: "10.5", class: "fill-emerald-700 dark:fill-emerald-300 font-bold",
          "kv_b_proj [{kv_lora_rank} → {kv_b_out}]"
        }
        text { x: "505", y: "221", text_anchor: "middle", font_size: "9.5", class: "fill-slate-600 dark:fill-slate-300",
          "Unpack {num_heads}H: k_nope[{qk_nope_head_dim}] + v[{v_head_dim}] (Absorbable)"
        }

        // 汇聚至 MLA Attention 核心
        path { d: "M 165 160 L 165 252 L 240 252 M 505 232 L 505 252 L 440 252", class: "stroke-slate-400 dark:stroke-slate-500", stroke_width: "1.5", fill: "none" }

        rect { x: "145", y: "252", width: "390", height: "46", rx: "10", class: "fill-indigo-600 stroke-indigo-400", stroke_width: "1.5" }
        text { x: "340", y: "271", text_anchor: "middle", font_size: "11", class: "fill-white font-bold",
          "MLA Attention ({num_heads} Heads, Scale 1/√{qk_dim} + YaRN mscale)"
        }
        text { x: "340", y: "287", text_anchor: "middle", font_size: "10", class: "fill-indigo-100",
          "Score = q_nope·k_nopeᵀ + q_pe·k_peᵀ → Weighted V [B, S, {o_in}]"
        }

        // o_proj
        line { x1: "340", y1: "298", x2: "340", y2: "318", class: "stroke-indigo-400", stroke_width: "1.5" }
        rect { x: "205", y: "318", width: "270", height: "44", rx: "8", class: "fill-slate-100 dark:fill-slate-800 stroke-slate-400 dark:stroke-slate-600", stroke_width: "1.5" }
        text { x: "340", y: "337", text_anchor: "middle", font_size: "11", class: "fill-slate-900 dark:fill-white font-bold",
          "o_proj [{o_in} → {hidden_size}]"
        }
        text { x: "340", y: "352", text_anchor: "middle", font_size: "10", class: "fill-slate-500 dark:fill-slate-400",
          "Output: [B, S, {hidden_size}] → Residual Add (+)"
        }
      }
    }
  }
}

/// 稠密门控前馈网络（SwiGLU / GeGLU）内部算子流图。
#[component]
fn DenseSwiGluSvg(hidden_size: u64, intermediate_size: u64, act_fn: String, bias: bool) -> Element {
  let bias_note = if bias { "+bias" } else { "no bias" };
  let ratio = (intermediate_size as f64) / (hidden_size.max(1) as f64);

  rsx! {
    div { class: "w-full overflow-x-auto",
      svg {
        view_box: "0 0 680 280",
        class: "w-full min-w-[500px] h-auto font-mono",
        role: "img",
        "aria-label": "SwiGLU gated MLP operator graph",

        rect { x: "230", y: "10", width: "220", height: "28", rx: "8", class: "fill-slate-100 dark:fill-slate-800 stroke-slate-300 dark:stroke-slate-600", stroke_width: "1.5" }
        text { x: "340", y: "28", text_anchor: "middle", font_size: "11", class: "fill-slate-800 dark:fill-slate-100 font-semibold",
          "h_norm [B, S, {hidden_size}]"
        }

        path { d: "M 340 38 L 340 52 M 190 52 L 490 52 M 190 52 L 190 68 M 490 52 L 490 68", class: "stroke-slate-400 dark:stroke-slate-500", stroke_width: "1.5", fill: "none" }

        // Gate 分支
        rect { x: "75", y: "68", width: "230", height: "42", rx: "8", class: "fill-sky-50 dark:fill-sky-950/70 stroke-sky-500", stroke_width: "1.5" }
        text { x: "190", y: "86", text_anchor: "middle", font_size: "11", class: "fill-sky-700 dark:fill-sky-300 font-bold",
          "gate_proj ({bias_note})"
        }
        text { x: "190", y: "101", text_anchor: "middle", font_size: "10", class: "fill-slate-600 dark:fill-slate-300",
          "[{hidden_size} → {intermediate_size}] ({ratio:.2}× D)"
        }

        line { x1: "190", y1: "110", x2: "190", y2: "128", class: "stroke-sky-400", stroke_width: "1.5" }
        rect { x: "115", y: "128", width: "150", height: "30", rx: "8", class: "fill-amber-50 dark:fill-amber-950/60 stroke-amber-500", stroke_width: "1.5" }
        text { x: "190", y: "147", text_anchor: "middle", font_size: "10.5", class: "fill-amber-700 dark:fill-amber-300 font-bold",
          "Activation: {act_fn}(x)"
        }

        // Up 分支
        rect { x: "375", y: "68", width: "230", height: "42", rx: "8", class: "fill-teal-50 dark:fill-teal-950/70 stroke-teal-500", stroke_width: "1.5" }
        text { x: "490", y: "86", text_anchor: "middle", font_size: "11", class: "fill-teal-700 dark:fill-teal-300 font-bold",
          "up_proj ({bias_note})"
        }
        text { x: "490", y: "101", text_anchor: "middle", font_size: "10", class: "fill-slate-600 dark:fill-slate-300",
          "[{hidden_size} → {intermediate_size}]"
        }

        // 逐元素相乘 ⊙
        path { d: "M 190 158 L 190 185 L 318 185 M 490 110 L 490 185 L 362 185", class: "stroke-slate-400 dark:stroke-slate-500", stroke_width: "1.5", fill: "none" }
        circle { cx: "340", cy: "185", r: "18", class: "fill-indigo-600 stroke-indigo-400", stroke_width: "1.5" }
        text { x: "340", y: "190", text_anchor: "middle", font_size: "14", class: "fill-white font-bold", "⊙" }

        // down_proj
        line { x1: "340", y1: "203", x2: "340", y2: "222", class: "stroke-indigo-400", stroke_width: "1.5" }
        rect { x: "205", y: "222", width: "270", height: "42", rx: "8", class: "fill-slate-100 dark:fill-slate-800 stroke-slate-400 dark:stroke-slate-600", stroke_width: "1.5" }
        text { x: "340", y: "240", text_anchor: "middle", font_size: "11", class: "fill-slate-900 dark:fill-white font-bold",
          "down_proj [{intermediate_size} → {hidden_size}]"
        }
        text { x: "340", y: "255", text_anchor: "middle", font_size: "10", class: "fill-slate-500 dark:fill-slate-400",
          "Output: [B, S, {hidden_size}]"
        }
      }
    }
  }
}

/// 稀疏混合专家（Sparse MoE：支持分组路由 Top-K + 常驻共享专家双通道）算子流图。
#[component]
fn SparseMoeSvg(
  hidden_size: u64,
  num_routed_experts: u64,
  top_k: u64,
  moe_intermediate_size: u64,
  num_shared_experts: u64,
  shared_intermediate_size: u64,
  router_groups: Option<(u64, u64)>,
  scoring_func: String,
  act_fn: String,
) -> Element {
  let idle_experts = num_routed_experts.saturating_sub(top_k);
  let group_str = match router_groups {
    Some((ng, tg)) => format!("{scoring_func} · {ng} groups→Top-{tg}"),
    None => format!("{scoring_func} Top-{top_k}"),
  };
  let has_shared = num_shared_experts > 0 && shared_intermediate_size > 0;

  rsx! {
    div { class: "w-full overflow-x-auto",
      svg {
        view_box: "0 0 680 350",
        class: "w-full min-w-[540px] h-auto font-mono",
        role: "img",
        "aria-label": "Sparse Mixture-of-Experts internal operator graph",

        rect { x: "230", y: "8", width: "220", height: "28", rx: "8", class: "fill-slate-100 dark:fill-slate-800 stroke-slate-300 dark:stroke-slate-600", stroke_width: "1.5" }
        text { x: "340", y: "26", text_anchor: "middle", font_size: "11", class: "fill-slate-800 dark:fill-slate-100 font-semibold",
          "h_norm [B, S, {hidden_size}]"
        }

        if has_shared {
          path { d: "M 340 36 L 340 50 M 220 50 L 545 50 M 220 50 L 220 66 M 545 50 L 545 142", class: "stroke-slate-400 dark:stroke-slate-500", stroke_width: "1.5", fill: "none" }
        } else {
          line { x1: "340", y1: "36", x2: "220", y2: "66", class: "stroke-slate-400 dark:stroke-slate-500", stroke_width: "1.5" }
        }

        // Router Gate
        rect { x: "55", y: "66", width: "330", height: "46", rx: "8", class: "fill-amber-50 dark:fill-amber-950/70 stroke-amber-500", stroke_width: "1.5" }
        text { x: "220", y: "85", text_anchor: "middle", font_size: "11", class: "fill-amber-800 dark:fill-amber-300 font-bold",
          "Router Gate: mlp.gate [{hidden_size} → {num_routed_experts}]"
        }
        text { x: "220", y: "101", text_anchor: "middle", font_size: "9.5", class: "fill-slate-600 dark:fill-slate-300",
          "{group_str} → Select {top_k} of {num_routed_experts} experts/token"
        }

        // 分叉至「激活专家」与「休眠专家」
        path { d: "M 220 112 L 220 126 M 125 126 L 315 126 M 125 126 L 125 142 M 315 126 L 315 142", class: "stroke-amber-500", stroke_width: "1.5", fill: "none" }

        // Active Top-K Experts
        rect { x: "25", y: "142", width: "195", height: "84", rx: "10", class: "fill-amber-100/80 dark:fill-amber-950/80 stroke-amber-500", stroke_width: "2" }
        text { x: "122", y: "162", text_anchor: "middle", font_size: "11", class: "fill-amber-800 dark:fill-amber-300 font-bold",
          "★ Active Top-{top_k} Experts"
        }
        text { x: "122", y: "180", text_anchor: "middle", font_size: "9.5", class: "fill-slate-700 dark:fill-slate-200",
          "gate/up [{hidden_size}→{moe_intermediate_size}]"
        }
        text { x: "122", y: "195", text_anchor: "middle", font_size: "9.5", class: "fill-slate-700 dark:fill-slate-200",
          "{act_fn} ⊙ → down [{moe_intermediate_size}→{hidden_size}]"
        }
        text { x: "122", y: "213", text_anchor: "middle", font_size: "9.5", class: "fill-amber-700 dark:fill-amber-400 font-semibold",
          "Computed per token"
        }

        // Idle Experts (Skipped FLOPs, Still in VRAM!)
        rect { x: "235", y: "142", width: "170", height: "84", rx: "10", class: "fill-slate-100/70 dark:fill-slate-900/70 stroke-slate-300 dark:stroke-slate-700", stroke_dasharray: "4 3", stroke_width: "1.5" }
        text { x: "320", y: "165", text_anchor: "middle", font_size: "10.5", class: "fill-slate-500 dark:fill-slate-400 font-semibold",
          "Idle {idle_experts} Experts"
        }
        text { x: "320", y: "184", text_anchor: "middle", font_size: "9.5", class: "fill-slate-400 dark:fill-slate-500",
          "0 FLOPs for this token"
        }
        text { x: "320", y: "202", text_anchor: "middle", font_size: "9.5", class: "fill-slate-400 dark:fill-slate-500",
          "(Resident in GPU VRAM)"
        }

        // Always-On Shared Expert Lane (if present)
        if has_shared {
          rect { x: "425", y: "142", width: "235", height: "84", rx: "10", class: "fill-emerald-50 dark:fill-emerald-950/70 stroke-emerald-500", stroke_width: "1.8" }
          text { x: "542", y: "162", text_anchor: "middle", font_size: "11", class: "fill-emerald-700 dark:fill-emerald-300 font-bold",
            "Always-On Shared ×{num_shared_experts}"
          }
          text { x: "542", y: "180", text_anchor: "middle", font_size: "9.5", class: "fill-slate-700 dark:fill-slate-200",
            "gate/up [{hidden_size}→{shared_intermediate_size}]"
          }
          text { x: "542", y: "195", text_anchor: "middle", font_size: "9.5", class: "fill-slate-700 dark:fill-slate-200",
            "down [{shared_intermediate_size}→{hidden_size}]"
          }
          text { x: "542", y: "213", text_anchor: "middle", font_size: "9.5", class: "fill-emerald-700 dark:fill-emerald-400 font-semibold",
            "Bypass router (100% tokens)"
          }
        }

        // 加权求和汇总输出
        if has_shared {
          path { d: "M 122 226 L 122 268 L 210 268 M 542 226 L 542 268 L 470 268", class: "stroke-slate-400 dark:stroke-slate-500", stroke_width: "1.5", fill: "none" }
        } else {
          path { d: "M 122 226 L 122 268 L 210 268", class: "stroke-amber-500", stroke_width: "1.5", fill: "none" }
        }

        rect { x: "185", y: "252", width: "310", height: "46", rx: "10", class: "fill-indigo-600 stroke-indigo-400", stroke_width: "1.5" }
        text { x: "340", y: "272", text_anchor: "middle", font_size: "11", class: "fill-white font-bold",
          if has_shared {
            "Σ (w_i · RoutedExpert_i(x)) + SharedExpert(x)"
          } else {
            "Σ_{{i ∈ TopK}} w_i · RoutedExpert_i(x)"
          }
        }
        text { x: "340", y: "288", text_anchor: "middle", font_size: "10", class: "fill-indigo-100",
          "moe_out : [B, S, {hidden_size}] → Residual Add (+)"
        }
      }
    }
  }
}

/// Embedding / RMSNorm / LM Head / MTP 端点算子流图。
#[component]
fn EndpointSubgraphSvg(
  mode: String,
  vocab_size: u64,
  hidden_size: u64,
  tied: bool,
  eps: f64,
) -> Element {
  let (title, box1, box2, foot) = match mode.as_str() {
    "embedding" => (
      "Token Embedding Lookup",
      format!("Table Weight W_emb : [{vocab_size}, {hidden_size}]"),
      format!("Gather(W_emb, input_ids) → h_0 : [B, S, {hidden_size}]"),
      if tied {
        "Weight Tied: 与末端 lm_head 共享同一份 [V, D] 参数矩阵，节省一倍词表显存".to_string()
      } else {
        "Untied Embeddings: 与末端 lm_head 独立存储参数".to_string()
      },
    ),
    "lm_head" => (
      "LM Head Vocabulary Projection",
      if tied {
        format!("Shared Weight W_embᵀ : [{hidden_size}, {vocab_size}] (Tied)")
      } else {
        format!("Linear Weight lm_head : [{vocab_size}, {hidden_size}]")
      },
      format!("MatMul(h_final, Wᵀ) → logits : [B, S, {vocab_size}]"),
      "推理期 Prefill 阶段通常仅对序列最后一个位置 [-1] 计算 Logits 以节省显存带宽".to_string(),
    ),
    "mtp" => (
      "Multi-Token Prediction (MTP Speculative Module)",
      format!("Concat(RMSNorm(h_t), RMSNorm(Embed(t_{{k+1}}))) → [B, S, {}]", 2 * hidden_size),
      format!("eh_proj [{} → {hidden_size}] → Transformer Block → Shared LM Head", 2 * hidden_size),
      "训练期增强多步语义规划；推理期可直接充当原生投机采样 (Speculative Decoding) 草稿头"
        .to_string(),
    ),
    _ => (
      "Pre-Norm RMSNorm & Residual Stream",
      format!("RMS(x) = sqrt(mean(x²) + {eps:.0e}),  γ : [{hidden_size}]"),
      format!("y = (x / RMS(x)) ⊙ γ → [B, S, {hidden_size}]"),
      "相比 LayerNorm 省略均值中心化与 β 偏置，减少约 30% 归一化访存开销并稳定深层梯度".to_string(),
    ),
  };

  rsx! {
    div { class: "w-full overflow-x-auto",
      svg {
        view_box: "0 0 680 220",
        class: "w-full min-w-[480px] h-auto font-mono",
        role: "img",
        "aria-label": "{title}",

        text { x: "340", y: "26", text_anchor: "middle", font_size: "12", class: "fill-indigo-600 dark:fill-indigo-400 font-bold",
          "{title}"
        }
        rect { x: "90", y: "44", width: "500", height: "42", rx: "8", class: "fill-slate-100 dark:fill-slate-800 stroke-slate-400 dark:stroke-slate-600", stroke_width: "1.5" }
        text { x: "340", y: "69", text_anchor: "middle", font_size: "11", class: "fill-slate-900 dark:fill-white font-semibold",
          "{box1}"
        }
        line { x1: "340", y1: "86", x2: "340", y2: "112", class: "stroke-indigo-500", stroke_width: "1.5" }
        rect { x: "90", y: "112", width: "500", height: "44", rx: "8", class: "fill-indigo-50 dark:fill-indigo-950/70 stroke-indigo-500", stroke_width: "1.5" }
        text { x: "340", y: "138", text_anchor: "middle", font_size: "11", class: "fill-indigo-800 dark:fill-indigo-200 font-bold",
          "{box2}"
        }
        text { x: "340", y: "188", text_anchor: "middle", font_size: "10.5", class: "fill-slate-500 dark:text-slate-400",
          "{foot}"
        }
      }
    }
  }
}
