//! 纯 Rust WASM 深度学习与大模型算子动态交互实验室（M2 后四大核心数值内核）。
//!
//! 涵盖：
//! 5. [`SequenceStateLab`]：RNN BPTT 谱半径相变与梯度裁剪、LSTM 四门细胞公路、Gated DeltaNet 矩阵擦写 vs 朴素线性串扰、Mamba 选择性步长 ZOH 离散化；
//! 6. [`AttentionMatrixLab`]：SDPA $\sqrt{d_k}$ 方差与雅可比坍缩、MHA/MQA/GQA 与 DeepSeek MLA 4 步矩阵吸收、FlashAttention SRAM 在线重缩放逐位校验、滑窗汇聚点与 PagedAttention 页表；
//! 7. [`AudioSpectralAndCodecLab`]：加窗 STFT 时频权衡与非均匀三角 Mel 滤波器组、CTC 空白符前向动态规划网格、RVQ 多级残差向量量化逼近；
//! 8. [`RoutingAndSamplingLab`]：Sparse MoE 无辅助损失动态偏置均衡仿真、Temperature/Top-K/Top-P/Min-P 采样器、无损投机解码拒绝采样恒等式、扩散 CFG 流形外推与分辨率 `time_shift`。

use std::f64::consts::PI;

use dioxus::prelude::*;

use crate::component_labs::{
  lab_metric_card, push_svg_pt, sigmoid, softplus, svg_grid_cell, svg_vbar,
};

/// 跨 M2 四大实验台共享的模式切换按钮（收敛重复 class 字符串与 VirtualDOM 模板）。
#[inline(never)]
pub(crate) fn mode_btn(label: &str, active: bool, onclick: EventHandler<MouseEvent>) -> Element {
  let cls = if active {
    "px-2.5 py-1.5 rounded-lg text-xs font-semibold bg-indigo-600 text-white cursor-pointer"
  } else {
    "px-2.5 py-1.5 rounded-lg text-xs font-medium bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700 cursor-pointer"
  };
  rsx! {
    button {
      key: "{label}",
      r#type: "button",
      class: "{cls}",
      onclick: move |ev| onclick.call(ev),
      "{label}"
    }
  }
}

/// 根据数值热度映射统一的蓝—靛—琥珀—玫红网格配色类。
#[inline(never)]
fn heat_cell_cls(norm_val: f64, highlighted: bool) -> &'static str {
  if highlighted {
    "fill-amber-200/90 dark:fill-amber-900/70 stroke-amber-500"
  } else if norm_val > 0.72 {
    "fill-indigo-600 text-white stroke-indigo-700"
  } else if norm_val > 0.42 {
    "fill-indigo-300/80 dark:fill-indigo-800/70 stroke-indigo-400 dark:stroke-indigo-700"
  } else if norm_val > 0.16 {
    "fill-indigo-100/80 dark:fill-indigo-950/70 stroke-slate-300 dark:stroke-slate-700"
  } else if norm_val < -0.25 {
    "fill-rose-200/80 dark:fill-rose-950/70 stroke-rose-400 dark:stroke-rose-800"
  } else {
    "fill-slate-50 dark:fill-slate-900 stroke-slate-200 dark:stroke-slate-800"
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 内核 5：SequenceStateLab（RNN BPTT / LSTM 四门 / Gated DeltaNet 擦写 / Mamba 选择性 SSM）
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SeqStateMode {
  ElmanBptt,
  LstmHighway,
  GatedDeltaNet,
  SelectiveMamba,
}

impl SeqStateMode {
  fn parse_mode(raw: &str) -> Self {
    match raw.trim().to_ascii_lowercase().as_str() {
      "rnn" | "elman" | "bptt" | "clip" => Self::ElmanBptt,
      "lstm" | "gru" | "xlstm" | "gates" => Self::LstmHighway,
      "deltanet" | "gated_deltanet" | "linear_attn" | "rwkv" | "retnet" => Self::GatedDeltaNet,
      "mamba" | "mamba2" | "ssm" | "s4" | "ssd" => Self::SelectiveMamba,
      _ => Self::GatedDeltaNet,
    }
  }

  fn label(self) -> &'static str {
    match self {
      Self::ElmanBptt => "朴素 RNN 与 BPTT 谱半径裁剪",
      Self::LstmHighway => "LSTM 四门与加法细胞梯度公路",
      Self::GatedDeltaNet => "Gated DeltaNet 矩阵擦写 vs 线性累加",
      Self::SelectiveMamba => "选择性 Mamba (Δ_t ZOH 离散状态)",
    }
  }
}

/// 朴素二维旋转循环网络 $W_{hh} = \rho R(\theta)$ 前向与 16 步 BPTT 反向梯度范数仿真（支持全局 L2 范数等比例裁剪）。
pub(crate) fn simulate_rnn_bptt(
  w_hh: f64,
  clip_max_norm: Option<f64>,
) -> ([f64; 16], [f64; 16], f64) {
  let (c_th, s_th) = (0.72_f64.cos(), 0.72_f64.sin());
  let mut h0_hist = [0.0_f64; 16];
  let mut h1_hist = [0.0_f64; 16];
  let (mut p0, mut p1) = (0.04_f64, 0.02_f64);

  for t in 0..16 {
    let x_t = 0.025 * ((t as f64) * 0.65).cos();
    let u0 = w_hh * (c_th * p0 - s_th * p1) + x_t;
    let u1 = w_hh * (s_th * p0 + c_th * p1);
    p0 = u0.tanh();
    p1 = u1.tanh();
    h0_hist[t] = p0;
    h1_hist[t] = p1;
  }

  let mut grads = [0.0_f64; 16];
  let (mut g0, mut g1) = (1.0_f64, 0.0_f64);
  grads[15] = 1.0;
  for rev in 0..15 {
    let t = 14 - rev;
    let d0 = (1.0 - h0_hist[t + 1] * h0_hist[t + 1]) * g0;
    let d1 = (1.0 - h1_hist[t + 1] * h1_hist[t + 1]) * g1;
    // 反传通过 W_hh^T = w_hh · R(-θ)
    g0 = w_hh * (c_th * d0 + s_th * d1);
    g1 = w_hh * (-s_th * d0 + c_th * d1);
    grads[t] = (g0 * g0 + g1 * g1).sqrt();
  }

  let raw_norm = grads.iter().map(|g| g * g).sum::<f64>().sqrt();
  if let Some(c) = clip_max_norm {
    if raw_norm > c && raw_norm > 1e-12 {
      let scale = c / raw_norm;
      for g in &mut grads {
        *g *= scale;
      }
    }
  }
  (h0_hist, grads, raw_norm)
}

const LSTM_TOKENS: [&str; 12] =
  ["The", "keys", "(", "near", "the", "old", "wooden", "cabin", ")", "are", "rusty", "."];

/// 模拟主谓一致长跨度从句中 LSTM 四门（f_t, i_t, c_t）及从动词回溯到主语的细胞梯度存活率。
pub(crate) fn simulate_lstm_clause(forget_bias: f64) -> [(f64, f64, f64, f64); 12] {
  let mut out = [(0.0_f64, 0.0, 0.0, 0.0); 12];
  let mut c_prev = 0.0_f64;
  for (t, slot) in out.iter_mut().enumerate() {
    let (f_logit, i_logit, g_cand) = match t {
      1 => (1.8 + forget_bias, 2.5, 0.96), // "keys" 复数特征写入细胞
      2..=8 => (1.7 + forget_bias, -2.9, -0.25), // 括号从句：高遗忘门保值、关闭输入门防干扰
      9 => (1.5 + forget_bias, -2.2, 0.10), // "are" 读取复数细胞状态
      11 => (-2.8 + 0.35 * forget_bias, -1.8, 0.0), // "." 句末主动遗忘清零
      _ => (0.9 + forget_bias, -1.6, 0.15),
    };
    let f_t = sigmoid(f_logit);
    let i_t = sigmoid(i_logit);
    let c_t = f_t * c_prev + i_t * g_cand;
    *slot = (f_t, i_t, c_t, 1.0);
    c_prev = c_t;
  }
  // 计算从 t=9 ("are") 沿加法细胞公路 ∂c_9 / ∂c_t = ∏_{k=t+1}^9 f_k 的反向梯度存活系数
  let mut running = 1.0_f64;
  out[9].3 = 1.0;
  for rev in 0..9 {
    let t = 8 - rev;
    running *= out[t + 1].0;
    out[t].3 = running;
  }
  out[10].3 = 0.0;
  out[11].3 = 0.0;
  out
}

struct KvWriteOp {
  tag: &'static str,
  slot_id: usize,
  key: [f64; 4],
  val: [f64; 4],
}

const DELTA_OPS: [KvWriteOp; 8] = [
  KvWriteOp {
    tag: "t1: 写入 K_A → V1",
    slot_id: 0,
    key: [1.0, 0.0, 0.0, 0.0],
    val: [0.9, 0.7, 0.4, -0.2],
  },
  KvWriteOp {
    tag: "t2: 写入 K_B → V_b",
    slot_id: 1,
    key: [0.0, 1.0, 0.0, 0.0],
    val: [0.2, 0.8, -0.5, 0.4],
  },
  KvWriteOp {
    tag: "t3: 写入 K_C → V_c",
    slot_id: 2,
    key: [0.0, 0.0, 1.0, 0.0],
    val: [-0.5, 0.2, 0.9, -0.3],
  },
  KvWriteOp {
    tag: "t4: 斜交干扰键 K_AB",
    slot_id: 4,
    key: [0.6, 0.8, 0.0, 0.0],
    val: [0.5, 0.5, -0.2, -0.4],
  },
  KvWriteOp {
    tag: "t5: 覆写 K_A → V2!",
    slot_id: 0,
    key: [1.0, 0.0, 0.0, 0.0],
    val: [0.8, 0.5, 0.6, 0.5],
  },
  KvWriteOp {
    tag: "t6: 写入 K_D → V_d",
    slot_id: 3,
    key: [0.0, 0.0, 0.0, 1.0],
    val: [0.1, -0.4, -0.6, 0.9],
  },
  KvWriteOp {
    tag: "t7: 覆写 K_B → V_b2!",
    slot_id: 1,
    key: [0.0, 1.0, 0.0, 0.0],
    val: [-0.7, -0.3, 0.8, 0.2],
  },
  KvWriteOp {
    tag: "t8: 二次覆写 K_A → V3!",
    slot_id: 0,
    key: [1.0, 0.0, 0.0, 0.0],
    val: [-0.7, 0.6, -0.8, -0.3],
  },
];

/// 对比朴素线性注意力累加 $S_t = \alpha S_{t-1} + v_t k_t^\top$ 与 Gated DeltaNet
/// $S_t = \alpha S_{t-1}(I - \beta k_t k_t^\top) + \beta v_t k_t^\top$ 在键值覆写后的检索误差。
pub(crate) fn simulate_deltanet_vs_linear(
  alpha: f64,
  beta: f64,
  upto_step: usize,
) -> ([[f64; 4]; 4], [[f64; 4]; 4], f64, f64) {
  let limit = upto_step.min(DELTA_OPS.len() - 1);
  let mut s_lin = [[0.0_f64; 4]; 4];
  let mut s_delta = [[0.0_f64; 4]; 4];
  let mut latest_va = DELTA_OPS[0].val;

  for op in DELTA_OPS.iter().take(limit + 1) {
    if op.slot_id == 0 {
      latest_va = op.val;
    }
    // 1. 朴素线性累加：旧值永不擦除，重复键直接叠加串扰
    for (r, row) in s_lin.iter_mut().enumerate() {
      for (c, cell) in row.iter_mut().enumerate() {
        *cell = alpha * (*cell) + op.val[r] * op.key[c];
      }
    }
    // 2. Gated DeltaNet：先从 α·S 检索旧预测 r = (α·S)·k，再写入差值 β·(v - r)·k^T
    let mut retrieved = [0.0_f64; 4];
    for (r, ret_r) in retrieved.iter_mut().enumerate() {
      let mut acc = 0.0;
      for (c, &kc) in op.key.iter().enumerate() {
        acc += alpha * s_delta[r][c] * kc;
      }
      *ret_r = acc;
    }
    for (r, row) in s_delta.iter_mut().enumerate() {
      let diff_r = beta * (op.val[r] - retrieved[r]);
      for (c, cell) in row.iter_mut().enumerate() {
        *cell = alpha * (*cell) + diff_r * op.key[c];
      }
    }
  }

  let k_a = DELTA_OPS[0].key;
  let l2_err = |mat: &[[f64; 4]; 4]| -> f64 {
    let mut sum_sq = 0.0;
    for (r, &target) in latest_va.iter().enumerate() {
      let pred: f64 = mat[r].iter().zip(k_a.iter()).map(|(m, k)| m * k).sum();
      let d = pred - target;
      sum_sq += d * d;
    }
    sum_sq.sqrt()
  };

  (s_lin, s_delta, l2_err(&s_lin), l2_err(&s_delta))
}

const MAMBA_STREAM: [(&str, f64, f64); 12] = [
  ("<bos>", 0.15, 0.0),
  ("Order", 0.72, 0.6),
  ("#409", 0.96, 1.0),
  ("uh", 0.04, -0.8),
  ("um", 0.03, 0.7),
  ("wait", 0.06, -0.6),
  ("ship", 0.68, 0.5),
  ("Tokyo", 0.95, -0.9),
  ("hmm", 0.04, 0.75),
  ("...", 0.02, -0.65),
  ("now", 0.35, 0.3),
  ("<eos>", 0.12, 0.0),
];

/// 选择性状态空间（Mamba）零阶保持 ZOH 离散化仿真：对比固定步长 S4 与输入依赖步长 $\Delta_t(x_t)$。
pub(crate) fn simulate_selective_mamba(selectivity: f64) -> [(f64, f64, f64); 12] {
  let sel = selectivity.clamp(0.0, 1.0);
  let mut out = [(0.0_f64, 0.0, 0.0); 12];
  let mut h = 0.0_f64;
  let fixed_delta = 0.34_f64;

  for (t, &(_, salience, signal)) in MAMBA_STREAM.iter().enumerate() {
    let dyn_delta = softplus(-3.6 + 5.6 * salience);
    let delta_t = (1.0 - sel) * fixed_delta + sel * dyn_delta;
    // A = -1 时的精确 ZOH 离散化：Ā_t = exp(-Δ_t), B̄_t = 1 - exp(-Δ_t)
    let a_bar = (-delta_t).exp();
    let b_bar = 1.0 - a_bar;
    h = a_bar * h + b_bar * signal;
    out[t] = (delta_t, a_bar, h);
  }
  out
}

#[component]
pub fn SequenceStateLab(initial_mode: String) -> Element {
  let mut mode = use_signal(move || SeqStateMode::parse_mode(&initial_mode));
  let mut step_t = use_signal(|| 4_usize);
  let mut playing = use_signal(|| false);
  // 各子模式核心物理参数
  let mut rnn_whh = use_signal(|| 1.28_f64);
  let mut rnn_clip = use_signal(|| false);
  let mut lstm_fbias = use_signal(|| 1.2_f64);
  let mut delta_alpha = use_signal(|| 0.98_f64);
  let mut delta_beta = use_signal(|| 1.0_f64);
  let mut mamba_sel = use_signal(|| 1.0_f64);

  use_effect(move || {
    if !playing() {
      return;
    }
    spawn(async move {
      while playing() {
        widgets::browser::sleep_ms(520).await;
        if !playing() {
          break;
        }
        step_t.with_mut(|v| *v = v.wrapping_add(1));
      }
    });
  });

  let cur_mode = mode();

  rsx! {
    div { class: "space-y-4",
      div { class: "flex flex-wrap gap-1.5",
        for m in [
          SeqStateMode::GatedDeltaNet,
          SeqStateMode::SelectiveMamba,
          SeqStateMode::LstmHighway,
          SeqStateMode::ElmanBptt,
        ] {
          {mode_btn(m.label(), cur_mode == m, EventHandler::new(move |_| {
            playing.set(false);
            mode.set(m);
            step_t.set(if m == SeqStateMode::GatedDeltaNet { 4 } else { 0 });
          }))}
        }
      }

      match cur_mode {
        SeqStateMode::ElmanBptt => {
          let w = rnn_whh();
          let clipped = rnn_clip();
          let (h_seq, grads, raw_norm) = simulate_rnn_bptt(w, if clipped { Some(2.0) } else { None });
          let max_g = grads.iter().copied().fold(1.0_f64, |a, b| a.max(b.abs()));
          let clipped_norm = grads.iter().map(|g| g * g).sum::<f64>().sqrt();

          rsx! {
            div { class: "space-y-3",
              div { class: "grid grid-cols-1 md:grid-cols-3 gap-3 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/70 dark:bg-slate-950/60 p-3 text-xs font-mono",
                div {
                  div { class: "mb-1 font-semibold", "循环谱半径近似 w_hh = {w:.2}" }
                  input {
                    r#type: "range", min: "0.45", max: "1.45", step: "0.02", value: "{w}",
                    class: "w-full accent-indigo-600 cursor-pointer",
                    oninput: move |e| { if let Ok(v) = e.value().parse::<f64>() { rnn_whh.set(v); } },
                  }
                }
                div { class: "flex items-center gap-2",
                  button {
                    r#type: "button",
                    class: if clipped {
                      "px-3 py-1.5 rounded-lg bg-emerald-600 text-white font-semibold cursor-pointer"
                    } else {
                      "px-3 py-1.5 rounded-lg bg-slate-200 dark:bg-slate-800 font-semibold cursor-pointer"
                    },
                    onclick: move |_| rnn_clip.set(!clipped),
                    if clipped { "✓ 已启用全局范数裁剪 (c = 2.0)" } else { "启用全局梯度范数裁剪 (Clip=2.0)" }
                  }
                }
                div { class: "flex items-center text-[11px] text-slate-600 dark:text-slate-300",
                  "判据：|w_hh·(1-h²)| < 1 指数消失；> 1 指数爆炸，全局缩放保持梯度方向余弦 = 1.0"
                }
              }

              div { class: "w-full overflow-x-auto rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950 p-3",
                svg {
                  view_box: "0 0 680 215",
                  class: "w-full min-w-[520px] h-auto font-mono select-none",
                  role: "img",
                  "aria-label": "BPTT gradient norm across 16 recurrent steps",
                  text { x: "340", y: "18", text_anchor: "middle", font_size: "10.5", class: "fill-slate-700 dark:fill-slate-200 font-bold",
                    "BPTT 反向传播各时间步梯度幅值 |∂L_15 / ∂h_t|（右端 t=15 为损失起点）"
                  }
                  line { x1: "35", y1: "182", x2: "655", y2: "182", class: "stroke-slate-300 dark:stroke-slate-700", stroke_width: "1" }
                  for (t, &g) in grads.iter().enumerate() {
                    {
                      let x = 42.0 + (t as f64) * 38.0;
                      let h_px = (g.abs() / max_g) * 138.0;
                      let cls = if raw_norm > 5.0 && !clipped {
                        "fill-rose-500"
                      } else if g.abs() < 0.02 {
                        "fill-slate-300 dark:fill-slate-700"
                      } else {
                        "fill-indigo-600 dark:fill-indigo-400"
                      };
                      let top_lbl = format!("{:.2}", g.abs());
                      let bot_lbl = format!("t{t}");
                      let key_id = format!("rnn-g-{t}");
                      svg_vbar(&key_id, x, 182.0, 24.0, h_px, cls, &top_lbl, &bot_lbl)
                    }
                  }
                }
              }

              div { class: "grid grid-cols-2 sm:grid-cols-4 gap-2.5 font-mono text-xs",
                {lab_metric_card("首步回传梯度 |∂L/∂h₀|", &format!("{:.2e}", grads[0].abs()), if grads[0].abs() < 1e-3 || grads[0].abs() > 10.0 { "text-rose-600 dark:text-rose-400" } else { "text-indigo-600 dark:text-indigo-400" })}
                {lab_metric_card("裁剪前全局范数 ‖g‖₂", &format!("{raw_norm:.3}"), "text-amber-600 dark:text-amber-400")}
                {lab_metric_card("当前有效范数 ‖g_eff‖₂", &format!("{clipped_norm:.3}"), "text-emerald-600 dark:text-emerald-400")}
                {lab_metric_card("末步隐状态 h₁₅", &format!("{:+.4}", h_seq[15]), "text-slate-800 dark:text-slate-100")}
              }
            }
          }
        }

        SeqStateMode::LstmHighway => {
          let fb = lstm_fbias();
          let trace = simulate_lstm_clause(fb);

          rsx! {
            div { class: "space-y-3",
              div { class: "grid grid-cols-1 md:grid-cols-2 gap-3 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/70 dark:bg-slate-950/60 p-3 text-xs font-mono",
                div {
                  div { class: "mb-1 font-semibold", "遗忘门初始化偏置 b_f = {fb:+.2}（Jozefowicz 推荐 +1.0 ~ +2.0）" }
                  input {
                    r#type: "range", min: "-1.5", max: "2.5", step: "0.1", value: "{fb}",
                    class: "w-full accent-indigo-600 cursor-pointer",
                    oninput: move |e| { if let Ok(v) = e.value().parse::<f64>() { lstm_fbias.set(v); } },
                  }
                }
                div { class: "flex items-center text-[11px] text-slate-600 dark:text-slate-300",
                  "c_t = f_t ⊙ c_{{t-1}} + i_t ⊙ c̃_t：当括号从句内 f_t ≈ 1、i_t ≈ 0 时，\"keys\" 的复数特征跨越 7 个干扰词无损直达谓语 \"are\"！"
                }
              }

              div { class: "w-full overflow-x-auto rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950 p-3",
                svg {
                  view_box: "0 0 680 215",
                  class: "w-full min-w-[540px] h-auto font-mono select-none",
                  role: "img",
                  "aria-label": "LSTM forget gate and additive cell highway across parenthetical clause",
                  text { x: "340", y: "18", text_anchor: "middle", font_size: "10.5", class: "fill-slate-700 dark:fill-slate-200 font-bold",
                    "上行：遗忘门 f_t · 中行：细胞记忆 c_t（锁住复数特征）· 下行：从 \"are\" 回传的细胞公路梯度"
                  }
                  for (t, &(f_t, _, c_t, grad_h)) in trace.iter().enumerate() {
                    {
                      let x = 48.0 + (t as f64) * 51.5;
                      let f_cls = heat_cell_cls(f_t * 0.85, t == 1 || t == 9);
                      let c_cls = heat_cell_cls(c_t.clamp(-1.0, 1.0), false);
                      let g_cls = heat_cell_cls(grad_h * 0.9, t == 1);
                      rsx! {
                        g { key: "lstm-col-{t}",
                          text { x: "{x + 22.0}", y: "42", text_anchor: "middle", font_size: "9.5", class: "fill-slate-800 dark:fill-slate-100 font-bold", "{LSTM_TOKENS[t]}" }
                          {svg_grid_cell(&format!("lf-{t}"), x, 50.0, 44.0, 34.0, f_cls, "1.2", &format!("{f_t:.2}"), "fill-slate-900 dark:fill-white font-semibold")}
                          {svg_grid_cell(&format!("lc-{t}"), x, 94.0, 44.0, 34.0, c_cls, "1.2", &format!("{c_t:+.2}"), "fill-slate-900 dark:fill-white font-semibold")}
                          {svg_grid_cell(&format!("lg-{t}"), x, 138.0, 44.0, 34.0, g_cls, "1.2", &format!("{grad_h:.2}"), "fill-slate-900 dark:fill-white font-semibold")}
                        }
                      }
                    }
                  }
                  text { x: "8", y: "71", font_size: "9", class: "fill-slate-500 font-bold", "f_t" }
                  text { x: "8", y: "115", font_size: "9", class: "fill-indigo-600 dark:fill-indigo-400 font-bold", "c_t" }
                  text { x: "4", y: "159", font_size: "8.5", class: "fill-amber-600 dark:fill-amber-400 font-bold", "∂c₉/∂c" }
                  text { x: "340", y: "198", text_anchor: "middle", font_size: "9.5", class: "fill-slate-500",
                    "把 b_f 拉低到 -1.5 观察细胞记忆在右括号前迅速归零（解释为何早期无偏置 LSTM 训练初期学不会长依赖）"
                  }
                }
              }

              div { class: "grid grid-cols-2 sm:grid-cols-4 gap-2.5 font-mono text-xs",
                {lab_metric_card("主语写入后细胞 c₁ (\"keys\")", &format!("{:+.3}", trace[1].2), "text-indigo-600 dark:text-indigo-400")}
                {lab_metric_card("谓语读取时细胞 c₉ (\"are\")", &format!("{:+.3}", trace[9].2), "text-emerald-600 dark:text-emerald-400")}
                {lab_metric_card("跨 8 步梯度存活 ∂c₉/∂c₁", &format!("{:.1}%", trace[1].3 * 100.0), if trace[1].3 < 0.15 { "text-rose-600" } else { "text-amber-600 dark:text-amber-400" })}
                {lab_metric_card("句号清零后残余 c₁₁ (\".\")", &format!("{:+.3}", trace[11].2), "text-slate-700 dark:text-slate-300")}
              }
            }
          }
        }

        SeqStateMode::GatedDeltaNet => {
          let cur_step = step_t() % DELTA_OPS.len();
          let alpha = delta_alpha();
          let beta = delta_beta();
          let (s_lin, s_delta, err_lin, err_delta) = simulate_deltanet_vs_linear(alpha, beta, cur_step);

          rsx! {
            div { class: "space-y-3",
              div { class: "grid grid-cols-1 md:grid-cols-3 gap-3 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/70 dark:bg-slate-950/60 p-3 text-xs font-mono",
                div {
                  div { class: "mb-1 font-semibold", "衰减门 α_t = {alpha:.2}" }
                  input {
                    r#type: "range", min: "0.75", max: "1.00", step: "0.01", value: "{alpha}",
                    class: "w-full accent-indigo-600 cursor-pointer",
                    oninput: move |e| { if let Ok(v) = e.value().parse::<f64>() { delta_alpha.set(v); } },
                  }
                }
                div {
                  div { class: "mb-1 font-semibold", "Delta 擦写率 β_t = {beta:.2}" }
                  input {
                    r#type: "range", min: "0.10", max: "1.00", step: "0.05", value: "{beta}",
                    class: "w-full accent-emerald-600 cursor-pointer",
                    oninput: move |e| { if let Ok(v) = e.value().parse::<f64>() { delta_beta.set(v); } },
                  }
                }
                div { class: "flex items-center justify-between gap-1.5",
                  button {
                    r#type: "button",
                    class: "px-2.5 py-1.5 rounded-lg bg-indigo-600 text-white font-semibold cursor-pointer",
                    onclick: move |_| playing.set(!playing()),
                    if playing() { "⏸ 暂停推演" } else { "▶ 播放键值覆写" }
                  }
                  span { class: "text-[11px] font-bold text-amber-600 dark:text-amber-400", "{DELTA_OPS[cur_step].tag}" }
                }
              }

              div { class: "flex flex-wrap gap-1",
                for (idx, op) in DELTA_OPS.iter().enumerate() {
                  button {
                    key: "dop-{idx}",
                    r#type: "button",
                    class: if cur_step == idx {
                      "px-2 py-1 rounded text-[11px] font-mono font-bold bg-amber-500 text-white cursor-pointer"
                    } else {
                      "px-2 py-1 rounded text-[11px] font-mono bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 cursor-pointer"
                    },
                    onclick: move |_| { playing.set(false); step_t.set(idx); },
                    "{op.tag}"
                  }
                }
              }

              div { class: "w-full overflow-x-auto rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950 p-3",
                svg {
                  view_box: "0 0 680 220",
                  class: "w-full min-w-[540px] h-auto font-mono select-none",
                  role: "img",
                  "aria-label": "Naive linear attention crosstalk vs Gated DeltaNet Householder erase-and-write state matrix",
                  text { x: "165", y: "20", text_anchor: "middle", font_size: "10.5", class: "fill-rose-600 dark:fill-rose-400 font-bold",
                    "朴素线性注意力 S_lin ∈ ℝ⁴ˣ⁴（旧值叠加强串扰）"
                  }
                  text { x: "510", y: "20", text_anchor: "middle", font_size: "10.5", class: "fill-emerald-600 dark:fill-emerald-400 font-bold",
                    "Gated DeltaNet S_delta ∈ ℝ⁴ˣ⁴（Householder 精准擦写）"
                  }
                  for r in 0..4 {
                    for c in 0..4 {
                      {
                        let v_l = s_lin[r][c];
                        let v_d = s_delta[r][c];
                        let xl = 65.0 + (c as f64) * 50.0;
                        let xd = 410.0 + (c as f64) * 50.0;
                        let y = 34.0 + (r as f64) * 40.0;
                        rsx! {
                          g { key: "dm-{r}-{c}",
                            {svg_grid_cell(&format!("sl-{r}-{c}"), xl, y, 46.0, 35.0, heat_cell_cls(v_l * 0.55, c == 0), "1.2", &format!("{v_l:+.2}"), "fill-slate-900 dark:fill-white font-semibold")}
                            {svg_grid_cell(&format!("sd-{r}-{c}"), xd, y, 46.0, 35.0, heat_cell_cls(v_d * 0.7, c == 0), "1.2", &format!("{v_d:+.2}"), "fill-slate-900 dark:fill-white font-semibold")}
                          }
                        }
                      }
                    }
                  }
                  text { x: "340", y: "210", text_anchor: "middle", font_size: "9.5", class: "fill-slate-500",
                    "高亮第 0 列对应键 K_A：点击 t5 / t8 覆写 K_A 后，右侧 Delta 规则精确替换为新值，左侧旧值混叠失真"
                  }
                }
              }

              div { class: "grid grid-cols-2 sm:grid-cols-4 gap-2.5 font-mono text-xs",
                {lab_metric_card("线性累加检索误差 ‖S_lin·K_A - V_A‖₂", &format!("{err_lin:.4}"), "text-rose-600 dark:text-rose-400")}
                {lab_metric_card("DeltaNet 检索误差 ‖S_Δ·K_A - V_A‖₂", &format!("{err_delta:.4}"), "text-emerald-600 dark:text-emerald-400")}
                {lab_metric_card("单层推理常驻状态尺寸", "O(d_v · d_k) 恒定", "text-indigo-600 dark:text-indigo-400")}
                {lab_metric_card("随序列长度 S 显存增长", "0 Bytes / Token", "text-amber-600 dark:text-amber-400")}
              }
            }
          }
        }

        SeqStateMode::SelectiveMamba => {
          let sel = mamba_sel();
          let m_trace = simulate_selective_mamba(sel);

          rsx! {
            div { class: "space-y-3",
              div { class: "grid grid-cols-1 md:grid-cols-2 gap-3 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/70 dark:bg-slate-950/60 p-3 text-xs font-mono",
                div {
                  div { class: "mb-1 font-semibold", "输入选择性强度 Selectivity = {sel:.2}（0 = 传统时不变 S4，1 = Mamba）" }
                  input {
                    r#type: "range", min: "0.0", max: "1.0", step: "0.05", value: "{sel}",
                    class: "w-full accent-indigo-600 cursor-pointer",
                    oninput: move |e| { if let Ok(v) = e.value().parse::<f64>() { mamba_sel.set(v); } },
                  }
                }
                div { class: "flex items-center text-[11px] text-slate-600 dark:text-slate-300",
                  "遇到无意义语气词（uh/um/hmm）时 Δ_t → 0 ⇒ Ā_t = exp(-Δ_t) → 1，自动冻结状态屏蔽噪声；遇到实体词（#409 / Tokyo）时放大 Δ_t 刷新记忆！"
                }
              }

              div { class: "w-full overflow-x-auto rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950 p-3",
                svg {
                  view_box: "0 0 680 215",
                  class: "w-full min-w-[540px] h-auto font-mono select-none",
                  role: "img",
                  "aria-label": "Selective Mamba discretization step size delta and state retention",
                  text { x: "340", y: "18", text_anchor: "middle", font_size: "10.5", class: "fill-slate-700 dark:fill-slate-200 font-bold",
                    "各词元动态离散步长 Δ_t(x_t) 柱状图（靛蓝=关键实体采样刷新，灰=语气噪声自动跳过）"
                  }
                  line { x1: "35", y1: "176", x2: "655", y2: "176", class: "stroke-slate-300 dark:stroke-slate-700", stroke_width: "1" }
                  for (t, &(dt, _, _)) in m_trace.iter().enumerate() {
                    {
                      let x = 42.0 + (t as f64) * 51.0;
                      let h_px = (dt / 2.2).min(1.0) * 132.0;
                      let cls = if MAMBA_STREAM[t].1 > 0.5 {
                        "fill-indigo-600 dark:fill-indigo-400"
                      } else {
                        "fill-slate-300 dark:fill-slate-700"
                      };
                      svg_vbar(&format!("mb-{t}"), x, 176.0, 34.0, h_px, cls, &format!("{dt:.2}"), MAMBA_STREAM[t].0)
                    }
                  }
                }
              }

              div { class: "grid grid-cols-2 sm:grid-cols-4 gap-2.5 font-mono text-xs",
                {lab_metric_card("实体 \"#409\" 步长 Δ₂ / 保持率 Ā₂", &format!("{:.2} / {:.2}", m_trace[2].0, m_trace[2].1), "text-indigo-600 dark:text-indigo-400")}
                {lab_metric_card("噪声 \"uh\" 步长 Δ₃ / 保持率 Ā₃", &format!("{:.2} / {:.3}", m_trace[3].0, m_trace[3].1), "text-emerald-600 dark:text-emerald-400")}
                {lab_metric_card("历经 3 个语气词后状态 h₅", &format!("{:+.3}", m_trace[5].2), "text-amber-600 dark:text-amber-400")}
                {lab_metric_card("序列末状态 h₁₁", &format!("{:+.3}", m_trace[11].2), "text-slate-800 dark:text-slate-100")}
              }
            }
          }
        }
      }
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 内核 6：AttentionMatrixLab（SDPA 缩放 / MHA-GQA-MLA 吸收 / FlashAttention / SWA+Paged）
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AttnLabMode {
  SdpaScale,
  GqaAndMla,
  FlashTiling,
  SwaAndPaged,
}

impl AttnLabMode {
  fn parse_mode(raw: &str) -> Self {
    match raw.trim().to_ascii_lowercase().as_str() {
      "sdpa" | "scaled_dot" | "additive" => Self::SdpaScale,
      "gqa" | "mqa" | "mha" | "mla" | "kvcache" => Self::GqaAndMla,
      "flash" | "flash_attention" | "online_softmax" => Self::FlashTiling,
      "paged" | "swa" | "sink" | "sparse" | "mmdit" | "ring" => Self::SwaAndPaged,
      _ => Self::FlashTiling,
    }
  }

  fn label(self) -> &'static str {
    match self {
      Self::SdpaScale => "SDPA √d_k 缩放与雅可比坍缩",
      Self::GqaAndMla => "MHA/GQA 与 MLA 矩阵吸收",
      Self::FlashTiling => "FlashAttention 在线重缩放",
      Self::SwaAndPaged => "SWA 汇聚点 & PagedAttention 页表",
    }
  }
}

const BASE_UNIT_DOTS: [[f64; 6]; 6] = [
  [0.9, -0.4, 0.2, -0.7, 0.5, 0.1],
  [-0.3, 1.2, -0.6, 0.4, -0.2, 0.3],
  [0.4, -0.5, 1.3, -0.8, 0.6, -0.4],
  [-0.7, 0.3, -0.2, 1.1, 0.5, -0.6],
  [0.2, -0.8, 0.7, -0.3, 1.4, -0.1],
  [0.5, 0.1, -0.9, 0.4, -0.2, 1.25],
];

/// 计算 $6\times 6$ 注意力分布、平均信息熵以及 Softmax 雅可比矩阵 $J = \operatorname{diag}(p) - pp^\top$ 的平均 Frobenius 范数。
pub(crate) fn compute_sdpa_matrix(d_k: usize, scaled: bool) -> ([[f64; 6]; 6], f64, f64, f64) {
  let std_unscaled = (d_k.max(1) as f64).sqrt();
  let eff_scale = if scaled { 1.0 } else { std_unscaled };
  let mut probs = [[0.0_f64; 6]; 6];
  let mut entropy_sum = 0.0;
  let mut jac_frob_sum = 0.0;

  for (r, base_row) in BASE_UNIT_DOTS.iter().enumerate() {
    let mut logits = [0.0_f64; 6];
    let mut max_l = f64::NEG_INFINITY;
    for (c, &u) in base_row.iter().enumerate() {
      let z = u * eff_scale;
      logits[c] = z;
      if z > max_l {
        max_l = z;
      }
    }
    let mut denom = 0.0;
    for (c, &z) in logits.iter().enumerate() {
      let e = (z - max_l).exp();
      probs[r][c] = e;
      denom += e;
    }
    let mut p2 = 0.0;
    let mut p3 = 0.0;
    for p in &mut probs[r] {
      *p /= denom;
      if *p > 1e-15 {
        entropy_sum -= (*p) * p.ln();
      }
      p2 += (*p) * (*p);
      p3 += (*p) * (*p) * (*p);
    }
    jac_frob_sum += (p2 - 2.0 * p3 + p2 * p2).max(0.0).sqrt();
  }
  (probs, eff_scale, entropy_sum / 6.0, jac_frob_sum / 6.0)
}

#[derive(Debug, Clone)]
pub(crate) struct FlashTileStep {
  pub m_old: f64,
  pub m_new: f64,
  pub alpha_rescale: f64,
  pub l_new: f64,
  pub out_flash: [f64; 4],
  pub out_full_ref: [f64; 4],
  pub max_abs_diff: f64,
}

const FLASH_SCORES_8: [f64; 8] = [-0.6, 1.2, 3.4, 0.5, 2.1, 5.6, 1.8, 4.2];
const FLASH_VALUES_8X4: [[f64; 4]; 8] = [
  [0.4, -0.2, 0.8, 0.1],
  [-0.5, 0.9, 0.3, -0.4],
  [1.1, 0.2, -0.6, 0.7],
  [0.0, -0.7, 0.5, 0.3],
  [-0.3, 0.6, 0.9, -0.8],
  [0.8, -0.4, 0.2, 1.2],
  [0.3, 0.5, -0.4, 0.2],
  [-0.6, 0.7, 0.4, -0.5],
];

/// 运行 FlashAttention-2 在线重缩放（Online Softmax，$B_c = 2$，共 4 块 SRAM Tiling）并与全量 Softmax 核对误差。
pub(crate) fn run_flash_online_softmax(upto_tile: usize) -> FlashTileStep {
  let max_tile = upto_tile.min(3);
  let mut m_run = f64::NEG_INFINITY;
  let mut l_run = 0.0_f64;
  let mut o_acc = [0.0_f64; 4];
  let mut last_m_old = f64::NEG_INFINITY;
  let mut last_alpha = 1.0_f64;

  for tile_j in 0..(max_tile + 1) {
    let idx0 = tile_j * 2;
    let s0 = FLASH_SCORES_8[idx0];
    let s1 = FLASH_SCORES_8[idx0 + 1];
    let tile_max = s0.max(s1);
    let m_new = m_run.max(tile_max);
    let alpha = if m_run.is_finite() { (m_run - m_new).exp() } else { 0.0 };
    let p0 = (s0 - m_new).exp();
    let p1 = (s1 - m_new).exp();

    for (d, acc_d) in o_acc.iter_mut().enumerate() {
      *acc_d =
        (*acc_d) * alpha + p0 * FLASH_VALUES_8X4[idx0][d] + p1 * FLASH_VALUES_8X4[idx0 + 1][d];
    }
    last_m_old = m_run;
    last_alpha = alpha;
    l_run = l_run * alpha + p0 + p1;
    m_run = m_new;
  }

  let mut out_flash = [0.0_f64; 4];
  for (d, slot) in out_flash.iter_mut().enumerate() {
    *slot = o_acc[d] / l_run.max(1e-15);
  }

  // 全矩阵标准 Softmax 基准
  let full_max = FLASH_SCORES_8.iter().copied().fold(f64::NEG_INFINITY, f64::max);
  let mut full_l = 0.0_f64;
  let mut out_ref = [0.0_f64; 4];
  for (i, &s) in FLASH_SCORES_8.iter().enumerate() {
    let w = (s - full_max).exp();
    full_l += w;
    for (d, r_d) in out_ref.iter_mut().enumerate() {
      *r_d += w * FLASH_VALUES_8X4[i][d];
    }
  }
  let mut max_diff = 0.0_f64;
  for (d, r_d) in out_ref.iter_mut().enumerate() {
    *r_d /= full_l;
    max_diff = max_diff.max((out_flash[d] - *r_d).abs());
  }

  FlashTileStep {
    m_old: last_m_old,
    m_new: m_run,
    alpha_rescale: last_alpha,
    l_new: l_run,
    out_flash,
    out_full_ref: out_ref,
    max_abs_diff: max_diff,
  }
}

#[component]
pub fn AttentionMatrixLab(initial_mode: String) -> Element {
  let mut mode = use_signal(move || AttnLabMode::parse_mode(&initial_mode));
  let mut dk = use_signal(|| 128_usize);
  let mut use_scale = use_signal(|| true);
  let mut arch_idx = use_signal(|| 4_usize); // 0=MHA,1=GQA8,2=GQA4,3=MQA,4=MLA
  let mut mla_step = use_signal(|| 2_usize);
  let mut flash_tile = use_signal(|| 3_usize);
  let mut swa_win = use_signal(|| 3_usize);
  let mut keep_sink = use_signal(|| true);

  let cur_mode = mode();

  rsx! {
    div { class: "space-y-4",
      div { class: "flex flex-wrap gap-1.5",
        for m in [
          AttnLabMode::FlashTiling,
          AttnLabMode::GqaAndMla,
          AttnLabMode::SdpaScale,
          AttnLabMode::SwaAndPaged,
        ] {
          {mode_btn(m.label(), cur_mode == m, EventHandler::new(move |_| mode.set(m)))}
        }
      }

      match cur_mode {
        AttnLabMode::SdpaScale => {
          let cur_dk = dk();
          let scaled = use_scale();
          let (probs, eff_std, mean_ent, jac_norm) = compute_sdpa_matrix(cur_dk, scaled);

          rsx! {
            div { class: "space-y-3",
              div { class: "grid grid-cols-1 md:grid-cols-3 gap-3 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/70 dark:bg-slate-950/60 p-3 text-xs font-mono",
                div {
                  div { class: "mb-1 font-semibold", "注意力头维度 d_k = {cur_dk}" }
                  input {
                    r#type: "range", min: "4", max: "256", step: "4", value: "{cur_dk}",
                    class: "w-full accent-indigo-600 cursor-pointer",
                    oninput: move |e| { if let Ok(v) = e.value().parse::<usize>() { dk.set(v); } },
                  }
                }
                div { class: "flex items-center",
                  button {
                    r#type: "button",
                    class: if scaled {
                      "px-3 py-1.5 rounded-lg bg-indigo-600 text-white font-semibold cursor-pointer"
                    } else {
                      "px-3 py-1.5 rounded-lg bg-rose-600 text-white font-semibold cursor-pointer"
                    },
                    onclick: move |_| use_scale.set(!scaled),
                    if scaled { "✓ 已除以 √d_k（方差归一 = 1.0）" } else { "✗ 未缩放 qᵀk（点击体验梯度死锁！）" }
                  }
                }
                div { class: "flex items-center text-[11px] text-slate-600 dark:text-slate-300",
                  "关闭 ÷√d_k 后，d_k=128 时点积标准差膨胀 11.3 倍，Softmax 坍缩成 One-Hot 尖刺，雅可比范数逼近 0！"
                }
              }

              div { class: "w-full overflow-x-auto rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950 p-3",
                svg {
                  view_box: "0 0 680 230",
                  class: "w-full min-w-[500px] h-auto font-mono select-none",
                  role: "img",
                  "aria-label": "Scaled vs unscaled dot-product attention probability matrix",
                  text { x: "340", y: "18", text_anchor: "middle", font_size: "10.5", class: "fill-slate-700 dark:fill-slate-200 font-bold",
                    "Softmax 注意力权重矩阵 P ∈ ℝ⁶ˣ⁶（深靛蓝 = 概率质量向单点尖刺坍缩）"
                  }
                  for (r, p_row) in probs.iter().enumerate() {
                    for (c, &p) in p_row.iter().enumerate() {
                      {
                        let x = 190.0 + (c as f64) * 50.0;
                        let y = 28.0 + (r as f64) * 32.0;
                        svg_grid_cell(
                          &format!("sdpa-{r}-{c}"),
                          x,
                          y,
                          46.0,
                          28.0,
                          heat_cell_cls(p, false),
                          "1.0",
                          &format!("{p:.2}"),
                          if p > 0.72 { "fill-white font-bold" } else { "fill-slate-800 dark:fill-slate-100" },
                        )
                      }
                    }
                  }
                }
              }

              div { class: "grid grid-cols-2 sm:grid-cols-4 gap-2.5 font-mono text-xs",
                {lab_metric_card("Logits 有效标准差 σ(z)", &format!("{eff_std:.2}"), "text-indigo-600 dark:text-indigo-400")}
                {lab_metric_card("行均注意力熵 H(p)", &format!("{mean_ent:.3} nats"), if mean_ent < 0.15 { "text-rose-600 dark:text-rose-400" } else { "text-emerald-600 dark:text-emerald-400" })}
                {lab_metric_card("Softmax 雅可比范数 ‖J‖_F", &format!("{jac_norm:.4}"), if jac_norm < 0.03 { "text-rose-600 dark:text-rose-400" } else { "text-amber-600 dark:text-amber-400" })}
                {lab_metric_card("最大单格权重 max(P)", &format!("{:.4}", probs[2][2]), "text-slate-800 dark:text-slate-100")}
              }
            }
          }
        }

        AttnLabMode::GqaAndMla => {
          let a_idx = arch_idx();
          let step_m = mla_step();
          // 以 DeepSeek-V3 级别 H=128, d_h=128, BF16 (2B) 核算单层每 Token KV Cache 字节数
          let (arch_name, kv_groups, kv_bytes, compress_x, mla_note) = match a_idx {
            0 => ("MHA (128Q : 128KV)", 8_usize, 65_536_u64, 1.0_f64, "每个查询头独占一份 K, V，长文本解码完全卡死在 HBM 显存带宽"),
            1 => ("GQA-8 (128Q : 8KV)", 4, 4_096, 16.0, "每 16 个查询头共享 1 组 KV，解码算术强度提升 16 倍（Llama-3 / Qwen3 主力）"),
            2 => ("GQA-4 (128Q : 4KV)", 2, 2_048, 32.0, "更高分组共享比，兼顾超大 Batch 吞吐与多头表达力"),
            3 => ("MQA (128Q : 1KV)", 1, 512, 128.0, "全头共享单组 KV，容量极省但超大规模模型表征质量略损"),
            _ => ("DeepSeek MLA (c_KV=512 + k_R=64)", 1, 1_152, 56.89, "缓存低秩潜向量 c_t^KV，推理期把解压矩阵 W^UK 吸收进查询侧，兼具 MHA 表达力与近 MQA 显存！"),
          };

          let step_desc = match step_m {
            0 => "Step 0（朴素解压反例）：若推理时显式算出 K_t = c_t^KV · W^UK ∈ ℝ^{128×128}，中间激活瞬间膨胀 32 倍撑爆 SRAM！",
            1 => "Step 1（结合律左吸 W^UK）：q_t^N · (c_t^KV · W_i^UK)ᵀ = (q_t^N · (W_i^UK)ᵀ) · (c_t^KV)ᵀ —— 查询向量直接投影至 512 维潜空间做 MQA！",
            2 => "Step 2（结合律右吸 W^UV）：∑ α_{t,j} (c_j^KV · W_i^UV) · W_i^O = (∑ α_{t,j} c_j^KV) · (W_i^UV · W_i^O) —— 先在 512 维加权求和，最后一步融合输出！",
            _ => "Step 3（解耦 RoPE 拼接）：因旋转矩阵 R_{m-n} 夹在中间不可交换，单独切出 64 维共享 k_t^R 走 RoPE，总缓存仅 (512 + 64)×2 = 1,152 B/token！",
          };

          rsx! {
            div { class: "space-y-3",
              div { class: "flex flex-wrap gap-1.5",
                for (idx, lbl) in [
                  (0_usize, "MHA (1:1)"),
                  (1, "GQA-8"),
                  (2, "GQA-4"),
                  (3, "MQA"),
                  (4, "DeepSeek MLA"),
                ] {
                  {mode_btn(lbl, a_idx == idx, EventHandler::new(move |_| arch_idx.set(idx)))}
                }
              }

              if a_idx == 4 {
                div { class: "flex flex-wrap gap-1.5 rounded-xl border border-indigo-200 dark:border-indigo-900/60 bg-indigo-50/60 dark:bg-indigo-950/30 p-2.5 text-xs",
                  for (st, st_lbl) in [
                    (0_usize, "① 朴素解压反例"),
                    (1, "② 左吸 W^UK 入 Q"),
                    (2, "③ 右吸 W^UV 入 W^O"),
                    (3, "④ 解耦 k_t^R 拼接"),
                  ] {
                    button {
                      key: "mla-st-{st}",
                      r#type: "button",
                      class: if step_m == st {
                        "px-2.5 py-1 rounded bg-amber-600 text-white font-bold cursor-pointer"
                      } else {
                        "px-2.5 py-1 rounded bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-200 cursor-pointer"
                      },
                      onclick: move |_| mla_step.set(st),
                      "{st_lbl}"
                    }
                  }
                  div { class: "w-full mt-1 text-slate-700 dark:text-slate-200 font-mono text-[11px]", "{step_desc}" }
                }
              }

              div { class: "w-full overflow-x-auto rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950 p-3",
                svg {
                  view_box: "0 0 680 185",
                  class: "w-full min-w-[520px] h-auto font-mono select-none",
                  role: "img",
                  "aria-label": "Query heads to KV heads or MLA latent vector routing",
                  text { x: "340", y: "18", text_anchor: "middle", font_size: "10.5", class: "fill-slate-700 dark:fill-slate-200 font-bold",
                    "{arch_name} · 头路由拓扑示意（上排 8 组 Query 头 → 下排常驻缓存槽）"
                  }
                  for q in 0..8_usize {
                    {
                      let qx = 65.0 + (q as f64) * 72.0;
                      let kv_slot = (q * kv_groups) / 8;
                      let kx = if a_idx == 4 {
                        318.0
                      } else {
                        65.0 + ((kv_slot as f64 + 0.5) * (8.0 / kv_groups as f64) - 0.5) * 72.0
                      };
                      rsx! {
                        g { key: "q-wire-{q}",
                          line { x1: "{qx + 22.0}", y1: "64", x2: "{kx + 22.0}", y2: "126", class: "stroke-indigo-400/75 dark:stroke-indigo-500/70", stroke_width: "1.6" }
                          {svg_grid_cell(&format!("qh-{q}"), qx, 34.0, 44.0, 28.0, "fill-indigo-600 stroke-indigo-700", "1", &format!("Q{q}"), "fill-white font-bold")}
                        }
                      }
                    }
                  }
                  if a_idx == 4 {
                    {svg_grid_cell("mla-ckv", 240.0, 126.0, 145.0, 34.0, "fill-emerald-600 stroke-emerald-700", "1.5", "c_t^KV (512d)", "fill-white font-bold")}
                    {svg_grid_cell("mla-kr", 395.0, 126.0, 110.0, 34.0, "fill-amber-500 stroke-amber-600", "1.5", "k_t^R (64d RoPE)", "fill-white font-bold")}
                  } else {
                    for kv in 0..kv_groups {
                      {
                        let kx = 65.0 + ((kv as f64 + 0.5) * (8.0 / kv_groups as f64) - 0.5) * 72.0;
                        svg_grid_cell(&format!("kvh-{kv}"), kx, 126.0, 48.0, 32.0, "fill-emerald-600 stroke-emerald-700", "1.2", &format!("KV{kv}"), "fill-white font-bold")
                      }
                    }
                  }
                  text { x: "340", y: "176", text_anchor: "middle", font_size: "9.5", class: "fill-slate-500", "{mla_note}" }
                }
              }

              div { class: "grid grid-cols-2 sm:grid-cols-4 gap-2.5 font-mono text-xs",
                {lab_metric_card("单层单 Token KV 缓存", &format!("{kv_bytes} Bytes"), "text-indigo-600 dark:text-indigo-400")}
                {lab_metric_card("相比 MHA 显存压缩比", &format!("{compress_x:.2}×"), "text-emerald-600 dark:text-emerald-400")}
                {lab_metric_card("60 层 × 32K 上下文总 KV", &format!("{:.2} GiB", (kv_bytes as f64 * 60.0 * 32768.0) / (1024.0 * 1024.0 * 1024.0)), "text-amber-600 dark:text-amber-400")}
                {lab_metric_card("解码算术强度倍率", &format!("{compress_x:.1}× FLOPs/B"), "text-slate-800 dark:text-slate-100")}
              }
            }
          }
        }

        AttnLabMode::FlashTiling => {
          let cur_tile = flash_tile().min(3);
          let trace = run_flash_online_softmax(cur_tile);

          rsx! {
            div { class: "space-y-3",
              div { class: "flex flex-wrap items-center justify-between gap-2 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/80 dark:bg-slate-950/60 p-3 text-xs font-mono",
                div { class: "flex flex-wrap items-center gap-1.5",
                  span { class: "font-semibold text-slate-600 dark:text-slate-300", "SRAM 分块加载步进 (B_c = 2)：" }
                  for tj in 0..4_usize {
                    button {
                      key: "ft-{tj}",
                      r#type: "button",
                      class: if cur_tile == tj {
                        "px-2.5 py-1 rounded bg-indigo-600 text-white font-bold cursor-pointer"
                      } else {
                        "px-2.5 py-1 rounded bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 cursor-pointer"
                      },
                      onclick: move |_| flash_tile.set(tj),
                      "Tile #{tj} (K[{tj * 2},{tj * 2 + 1}])"
                    }
                  }
                }
                span { class: "text-[11px] text-emerald-600 dark:text-emerald-400 font-bold",
                  "零 HBM 读写 N×N 中间矩阵！"
                }
              }

              div { class: "w-full overflow-x-auto rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950 p-3",
                svg {
                  view_box: "0 0 680 190",
                  class: "w-full min-w-[540px] h-auto font-mono select-none",
                  role: "img",
                  "aria-label": "FlashAttention SRAM block tiling and online softmax rescaling",
                  text { x: "340", y: "18", text_anchor: "middle", font_size: "10.5", class: "fill-slate-700 dark:fill-slate-200 font-bold",
                    "HBM 分数流 S[0..8] 分 4 批调入片上 SRAM（琥珀高亮 = 当前驻留块，绿 = 已归约块）"
                  }
                  for (i, &sc) in FLASH_SCORES_8.iter().enumerate() {
                    {
                      let tj = i / 2;
                      let x = 44.0 + (i as f64) * 75.0;
                      let cls = if tj == cur_tile {
                        "fill-amber-200 dark:fill-amber-900/80 stroke-amber-500"
                      } else if tj < cur_tile {
                        "fill-emerald-100 dark:fill-emerald-950/70 stroke-emerald-500"
                      } else {
                        "fill-slate-100 dark:fill-slate-900 stroke-slate-300 dark:stroke-slate-700"
                      };
                      svg_grid_cell(&format!("fs-{i}"), x, 36.0, 66.0, 42.0, cls, "1.6", &format!("s{i}={sc:+.1}"), "fill-slate-900 dark:fill-white font-bold")
                    }
                  }
                  text { x: "340", y: "106", text_anchor: "middle", font_size: "10", class: "fill-indigo-600 dark:fill-indigo-400 font-bold",
                    "在线补偿更新：Õ ← exp(m_old - m_new)·Õ + exp(S_tile - m_new)·V_tile（当前重缩放因子 α = {trace.alpha_rescale:.4}）"
                  }
                  for d in 0..4_usize {
                    {
                      let x = 95.0 + (d as f64) * 125.0;
                      svg_grid_cell(
                        &format!("fo-{d}"),
                        x,
                        122.0,
                        112.0,
                        42.0,
                        "fill-indigo-50 dark:fill-indigo-950/60 stroke-indigo-400",
                        "1.3",
                        &format!("O[{d}]={:+.4}", trace.out_flash[d]),
                        "fill-indigo-900 dark:fill-indigo-200 font-bold",
                      )
                    }
                  }
                  text { x: "340", y: "180", text_anchor: "middle", font_size: "9.5", class: "fill-slate-500",
                    "标准全矩阵基准 O_ref = [{trace.out_full_ref[0]:+.4}, {trace.out_full_ref[1]:+.4}, {trace.out_full_ref[2]:+.4}, {trace.out_full_ref[3]:+.4}]（走完 Tile #3 后逐元素完全重合！）"
                  }
                }
              }

              div { class: "grid grid-cols-2 sm:grid-cols-4 gap-2.5 font-mono text-xs",
                {lab_metric_card("最大值跳变 m_old → m_new", &format!("{:.1} → {:.1}", trace.m_old, trace.m_new), "text-indigo-600 dark:text-indigo-400")}
                {lab_metric_card("历史累加器重缩放 α", &format!("{:.5}", trace.alpha_rescale), "text-amber-600 dark:text-amber-400")}
                {lab_metric_card("在线分母 l_new", &format!("{:.4}", trace.l_new), "text-slate-800 dark:text-slate-100")}
                {lab_metric_card("与全量 Softmax 最大残差", &format!("{:.2e}", trace.max_abs_diff), if cur_tile == 3 { "text-emerald-600 dark:text-emerald-400" } else { "text-slate-500" })}
              }
            }
          }
        }

        AttnLabMode::SwaAndPaged => {
          let w = swa_win();
          let sink = keep_sink();
          // 物理块映射表：Req A 与 Req B 共享前 2 块 System Prompt (Physical #3, #1)
          const PHYS_MAP_A: [usize; 4] = [3, 1, 6, 0];
          const PHYS_MAP_B: [usize; 4] = [3, 1, 4, 7];

          rsx! {
            div { class: "space-y-3",
              div { class: "grid grid-cols-1 md:grid-cols-3 gap-3 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/70 dark:bg-slate-950/60 p-3 text-xs font-mono",
                div {
                  div { class: "mb-1 font-semibold", "滑窗宽度 W = {w}" }
                  input {
                    r#type: "range", min: "2", max: "5", step: "1", value: "{w}",
                    class: "w-full accent-indigo-600 cursor-pointer",
                    oninput: move |e| { if let Ok(v) = e.value().parse::<usize>() { swa_win.set(v); } },
                  }
                }
                div { class: "flex items-center",
                  button {
                    r#type: "button",
                    class: if sink {
                      "px-3 py-1.5 rounded-lg bg-emerald-600 text-white font-semibold cursor-pointer"
                    } else {
                      "px-3 py-1.5 rounded-lg bg-rose-600 text-white font-semibold cursor-pointer"
                    },
                    onclick: move |_| keep_sink.set(!sink),
                    if sink { "✓ 保留第 0 列 Attention Sink" } else { "✗ 裁掉第 0 列（流式滑窗崩盘）" }
                  }
                }
                div { class: "flex items-center text-[11px] text-slate-600 dark:text-slate-300",
                  "右图：PagedAttention 将连续逻辑块映射至非连续物理 KV 页，系统提示词块（#3, #1）引用计数 Ref=2 零拷贝共享"
                }
              }

              div { class: "w-full overflow-x-auto rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950 p-3",
                svg {
                  view_box: "0 0 680 215",
                  class: "w-full min-w-[540px] h-auto font-mono select-none",
                  role: "img",
                  "aria-label": "StreamingLLM attention sink mask and PagedAttention block table",
                  text { x: "155", y: "18", text_anchor: "middle", font_size: "10.5", class: "fill-slate-700 dark:fill-slate-200 font-bold",
                    "左：SWA + Attention Sink 因果掩码 (7×7)"
                  }
                  for r in 0..7_usize {
                    for c in 0..7_usize {
                      {
                        let active = c <= r && ((r.saturating_sub(c) < w) || (sink && c == 0));
                        let is_sink_col = sink && c == 0 && r >= w;
                        let cls = if is_sink_col {
                          "fill-emerald-500 stroke-emerald-600"
                        } else if active {
                          "fill-indigo-500 stroke-indigo-600"
                        } else {
                          "fill-slate-100 dark:fill-slate-900 stroke-slate-200 dark:stroke-slate-800"
                        };
                        svg_grid_cell(
                          &format!("swa-{r}-{c}"),
                          38.0 + (c as f64) * 33.0,
                          28.0 + (r as f64) * 25.0,
                          30.0,
                          22.0,
                          cls,
                          "1",
                          if is_sink_col { "S" } else if active { "1" } else { "·" },
                          if active { "fill-white font-bold" } else { "fill-slate-400" },
                        )
                      }
                    }
                  }

                  text { x: "495", y: "18", text_anchor: "middle", font_size: "10.5", class: "fill-slate-700 dark:fill-slate-200 font-bold",
                    "右：PagedAttention 页表（绿 = CoW 共享前缀页）"
                  }
                  for b in 0..4_usize {
                    {
                      let x = 350.0 + (b as f64) * 74.0;
                      let shared = b < 2;
                      let cls_a = if shared { "fill-emerald-600 stroke-emerald-700" } else { "fill-indigo-600 stroke-indigo-700" };
                      let cls_b = if shared { "fill-emerald-600 stroke-emerald-700" } else { "fill-violet-600 stroke-violet-700" };
                      rsx! {
                        g { key: "pg-{b}",
                          {svg_grid_cell(&format!("pga-{b}"), x, 42.0, 66.0, 34.0, cls_a, "1.2", &format!("A[{b}]→P{}", PHYS_MAP_A[b]), "fill-white font-bold")}
                          {svg_grid_cell(&format!("pgb-{b}"), x, 92.0, 66.0, 34.0, cls_b, "1.2", &format!("B[{b}]→P{}", PHYS_MAP_B[b]), "fill-white font-bold")}
                        }
                      }
                    }
                  }
                  for p in 0..8_usize {
                    {
                      let x = 335.0 + (p as f64) * 41.0;
                      let ref_cnt = usize::from(PHYS_MAP_A.contains(&p)) + usize::from(PHYS_MAP_B.contains(&p));
                      let cls = match ref_cnt {
                        2 => "fill-emerald-100 dark:fill-emerald-950 stroke-emerald-500",
                        1 => "fill-indigo-100 dark:fill-indigo-950 stroke-indigo-400",
                        _ => "fill-slate-50 dark:fill-slate-900 stroke-slate-300 dark:stroke-slate-800",
                      };
                      svg_grid_cell(&format!("phys-{p}"), x, 152.0, 37.0, 36.0, cls, "1.2", &format!("P{p}:{ref_cnt}"), "fill-slate-800 dark:fill-slate-100 font-bold")
                    }
                  }
                  text { x: "495", y: "204", text_anchor: "middle", font_size: "9", class: "fill-slate-500",
                    "物理显存池 P0..P7（冒号后为引用计数 RefCount，P1/P3 共享省去 50% 前缀显存）"
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

// ─────────────────────────────────────────────────────────────────────────────
// 内核 7：AudioSpectralAndCodecLab（STFT+Mel 滤波器 / CTC 前向网格 / RVQ 多级残差码本）
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AudioLabMode {
  StftMel,
  CtcLattice,
  RvqCodebook,
}

impl AudioLabMode {
  fn parse_mode(raw: &str) -> Self {
    match raw.trim().to_ascii_lowercase().as_str() {
      "stft" | "mel" | "mfcc" | "specaugment" | "conformer" => Self::StftMel,
      "ctc" | "rnnt" | "transducer" => Self::CtcLattice,
      "rvq" | "vq" | "fsq" | "vocoder" | "codec" => Self::RvqCodebook,
      _ => Self::StftMel,
    }
  }

  fn label(self) -> &'static str {
    match self {
      Self::StftMel => "加窗 STFT 时频权衡 & Mel 滤波器组",
      Self::CtcLattice => "CTC 空白符前向动态规划网格",
      Self::RvqCodebook => "RVQ 多级残差向量量化逼近",
    }
  }
}

/// 纯 Rust 合成双谐波元音 + 第 54 采样点爆破辅音瞬态脉冲，并计算 Hann 加窗 STFT 与三角 Mel 能量谱。
pub(crate) fn compute_stft_and_mel(
  win_len: usize,
  spec_augment: bool,
) -> ([f64; 128], [[f64; 10]; 10], [f64; 8]) {
  let mut wave = [0.0_f64; 128];
  for (n, sample) in wave.iter_mut().enumerate() {
    let t = (n as f64) / 128.0;
    let harmonic = 0.55 * (2.0 * PI * 9.0 * t).sin() + 0.40 * (2.0 * PI * 23.0 * t).sin();
    let click = if (52..56).contains(&n) { 1.35 } else { 0.0 };
    *sample = harmonic + click;
  }

  let w_len = win_len.clamp(12, 64);
  let mut spec = [[0.0_f64; 10]; 10]; // [frame 0..10][freq_bin 0..10]
  let mut max_mag = 1e-9_f64;

  for (fr, spec_col) in spec.iter_mut().enumerate() {
    let center = 12 + fr * 10;
    for (k_bin, mag_slot) in spec_col.iter_mut().enumerate() {
      let freq_idx = (k_bin + 1) * 3;
      let mut re = 0.0;
      let mut im = 0.0;
      for m in 0..w_len {
        let sig_idx = (center + m).saturating_sub(w_len / 2).min(127);
        let hann = 0.5 * (1.0 - (2.0 * PI * (m as f64) / ((w_len - 1).max(1) as f64)).cos());
        let x = wave[sig_idx] * hann;
        let ang = 2.0 * PI * (freq_idx as f64) * (m as f64) / 64.0;
        re += x * ang.cos();
        im -= x * ang.sin();
      }
      let mag = (re * re + im * im).sqrt() / (w_len as f64).sqrt();
      *mag_slot = mag;
      max_mag = max_mag.max(mag);
    }
  }

  for (fr, col) in spec.iter_mut().enumerate() {
    for (fb, cell) in col.iter_mut().enumerate() {
      if spec_augment && ((4..6).contains(&fr) || fb == 3) {
        *cell = 0.0;
      } else {
        *cell /= max_mag;
      }
    }
  }

  // 非均匀 Mel 尺度积分到 8 个梅尔通道（低频窄、高频宽）
  let mut mel_bars = [0.0_f64; 8];
  for (m_idx, bar) in mel_bars.iter_mut().enumerate() {
    let f_lo = ((m_idx as f64 / 8.0).powf(1.55) * 9.0) as usize;
    let f_hi = (((m_idx + 1) as f64 / 8.0).powf(1.55) * 9.0).ceil() as usize;
    let mut acc = 0.0;
    let mut cnt = 0.0;
    for col in &spec {
      for &val in col.iter().take(f_hi.min(9) + 1).skip(f_lo.min(9)) {
        acc += val;
        cnt += 1.0;
      }
    }
    *bar = if cnt > 0.0 { acc / cnt } else { 0.0 };
  }

  (wave, spec, mel_bars)
}

/// CTC 前向动态规划 $\alpha_{t, s}$：支持切换普通单词 `"cat"` 与含重复字母的 `"aa"`（验证重复字符间不可跳过 `<blank>`）。
pub(crate) fn compute_ctc_forward_lattice(
  repeat_word: bool,
  blank_prob: f64,
) -> (Vec<&'static str>, [[f64; 7]; 8], bool, f64) {
  let labels: Vec<&'static str> = if repeat_word {
    vec!["ϵ", "a", "ϵ", "a", "ϵ"]
  } else {
    vec!["ϵ", "c", "ϵ", "a", "ϵ", "t", "ϵ"]
  };
  let s_len = labels.len();
  let p_blank = blank_prob.clamp(0.10, 0.80);
  let p_char = (1.0 - p_blank) / 2.0;

  let mut alpha = [[0.0_f64; 7]; 8];
  alpha[0][0] = p_blank;
  alpha[0][1] = p_char;

  let can_skip_1_to_3 = s_len > 3 && labels[3] != "ϵ" && labels[3] != labels[1];

  for t in 1..8 {
    for s in 0..s_len {
      let emit = if labels[s] == "ϵ" { p_blank } else { p_char };
      let mut prev_sum = alpha[t - 1][s];
      if s >= 1 {
        prev_sum += alpha[t - 1][s - 1];
      }
      if s >= 2 && labels[s] != "ϵ" && labels[s] != labels[s - 2] {
        prev_sum += alpha[t - 1][s - 2];
      }
      alpha[t][s] = emit * prev_sum;
    }
  }

  let total_prob = alpha[7][s_len - 1] + alpha[7][s_len - 2];
  (labels, alpha, can_skip_1_to_3, total_prob)
}

/// 多级残差向量量化（RVQ，$Q = 1\dots 6$，含零向量锚点与双环旋转质心）：返回逐级累积逼近轨迹与单调非增 MSE。
pub(crate) fn compute_rvq_stages(zx: f64, zy: f64, max_q: usize) -> (Vec<(f64, f64)>, [f64; 6]) {
  let q_depth = max_q.clamp(1, 6);
  let mut rx = zx;
  let mut ry = zy;
  let mut ax = 0.0_f64;
  let mut ay = 0.0_f64;
  let mut path = Vec::with_capacity(q_depth + 1);
  path.push((0.0, 0.0));
  let mut mses = [0.0_f64; 6];

  for (q, mse_slot) in mses.iter_mut().enumerate() {
    let radius = 1.15 * 0.54_f64.powi(q as i32);
    let rot = (q as f64) * 0.68;
    // 以零向量 (0, 0) 为保底候选，严格保证每级残差范数单调不增
    let mut best_cx = 0.0;
    let mut best_cy = 0.0;
    let mut best_d2 = rx * rx + ry * ry;
    for ring_scale in [0.48_f64, 1.0] {
      let r_ring = radius * ring_scale;
      for k in 0..8 {
        let ang = rot + (k as f64) * (2.0 * PI / 8.0);
        let cx = r_ring * ang.cos();
        let cy = r_ring * ang.sin();
        let d2 = (rx - cx) * (rx - cx) + (ry - cy) * (ry - cy);
        if d2 < best_d2 {
          best_d2 = d2;
          best_cx = cx;
          best_cy = cy;
        }
      }
    }
    rx -= best_cx;
    ry -= best_cy;
    ax += best_cx;
    ay += best_cy;
    if q < q_depth {
      path.push((ax, ay));
    }
    *mse_slot = rx * rx + ry * ry;
  }
  (path, mses)
}

#[component]
pub fn AudioSpectralAndCodecLab(initial_mode: String) -> Element {
  let mut mode = use_signal(move || AudioLabMode::parse_mode(&initial_mode));
  let mut win_len = use_signal(|| 32_usize);
  let mut spec_aug = use_signal(|| false);
  let mut ctc_repeat = use_signal(|| false);
  let mut ctc_blank = use_signal(|| 0.35_f64);
  let mut rvq_q = use_signal(|| 4_usize);
  let mut target_x = use_signal(|| 1.25_f64);
  let mut target_y = use_signal(|| -0.85_f64);

  let cur_mode = mode();

  rsx! {
    div { class: "space-y-4",
      div { class: "flex flex-wrap gap-1.5",
        for m in [AudioLabMode::StftMel, AudioLabMode::CtcLattice, AudioLabMode::RvqCodebook] {
          {mode_btn(m.label(), cur_mode == m, EventHandler::new(move |_| mode.set(m)))}
        }
      }

      match cur_mode {
        AudioLabMode::StftMel => {
          let wl = win_len();
          let aug = spec_aug();
          let (wave, spec, mel_bars) = compute_stft_and_mel(wl, aug);
          let mut wave_pts = String::with_capacity(1400);
          for (i, &w) in wave.iter().enumerate() {
            let sx = 28.0 + (i as f64 / 127.0) * 260.0;
            let sy = 68.0 - (w.clamp(-2.2, 2.2) / 2.2) * 34.0;
            push_svg_pt(&mut wave_pts, i == 0, sx, sy);
          }

          rsx! {
            div { class: "space-y-3",
              div { class: "flex flex-wrap items-center justify-between gap-3 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/80 dark:bg-slate-950/60 p-3 text-xs font-mono",
                div { class: "flex items-center gap-1.5",
                  span { class: "font-semibold", "Hann 窗长 N_win：" }
                  for w_opt in [16_usize, 32, 64] {
                    button {
                      key: "wl-{w_opt}",
                      r#type: "button",
                      class: if wl == w_opt {
                        "px-2.5 py-1 rounded bg-indigo-600 text-white font-bold cursor-pointer"
                      } else {
                        "px-2.5 py-1 rounded bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 cursor-pointer"
                      },
                      onclick: move |_| win_len.set(w_opt),
                      if w_opt == 16 { "16 (窄窗定瞬态)" } else if w_opt == 32 { "32 (均衡)" } else { "64 (宽窗辨谐波)" }
                    }
                  }
                }
                button {
                  r#type: "button",
                  class: if aug {
                    "px-3 py-1 rounded bg-amber-600 text-white font-semibold cursor-pointer"
                  } else {
                    "px-3 py-1 rounded bg-slate-200 dark:bg-slate-800 font-semibold cursor-pointer"
                  },
                  onclick: move |_| spec_aug.set(!aug),
                  if aug { "✓ SpecAugment 时频遮挡开启" } else { "开启 SpecAugment 时频遮挡" }
                }
              }

              div { class: "w-full overflow-x-auto rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950 p-3",
                svg {
                  view_box: "0 0 680 225",
                  class: "w-full min-w-[540px] h-auto font-mono select-none",
                  role: "img",
                  "aria-label": "Waveform, STFT spectrogram, and Mel filterbank energy",
                  text { x: "158", y: "18", text_anchor: "middle", font_size: "10", class: "fill-slate-700 dark:fill-slate-200 font-bold",
                    "左上：合成语音波形（双谐波 + n=54 爆破瞬态）"
                  }
                  rect { x: "24", y: "26", width: "268", height: "84", rx: "5", class: "fill-slate-50 dark:fill-slate-900 stroke-slate-200 dark:stroke-slate-800", stroke_width: "1" }
                  polyline { points: "{wave_pts}", fill: "none", class: "stroke-indigo-600 dark:stroke-indigo-400", stroke_width: "1.6" }

                  text { x: "158", y: "128", text_anchor: "middle", font_size: "10", class: "fill-slate-700 dark:fill-slate-200 font-bold",
                    "左下：8 通道非均匀 Mel 滤波器积分能量"
                  }
                  for (m_i, &mb) in mel_bars.iter().enumerate() {
                    {svg_vbar(&format!("mel-{m_i}"), 32.0 + (m_i as f64) * 32.0, 206.0, 22.0, mb * 62.0, "fill-emerald-500", &format!("{mb:.2}"), &format!("M{m_i}"))}
                  }

                  text { x: "490", y: "18", text_anchor: "middle", font_size: "10.5", class: "fill-slate-700 dark:fill-slate-200 font-bold",
                    "右：STFT 时频语谱图 (横轴帧 t=0..9，纵轴低频→高频)"
                  }
                  for (fr, col) in spec.iter().enumerate() {
                    for (fb, &mag) in col.iter().enumerate() {
                      {
                        let x = 330.0 + (fr as f64) * 32.0;
                        let y = 186.0 - (fb as f64) * 17.5;
                        svg_grid_cell(&format!("sp-{fr}-{fb}"), x, y, 29.0, 15.5, heat_cell_cls(mag, false), "0.8", "", "")
                      }
                    }
                  }
                }
              }

              div { class: "rounded-lg bg-indigo-50/70 dark:bg-indigo-950/30 border border-indigo-200/70 dark:border-indigo-900/60 px-3.5 py-2 text-xs text-slate-700 dark:text-slate-300",
                "Gabor 不确定性原理 Δt · Δf ≥ 1/(4π)：切换 N_win=16 可看清第 4 帧竖直亮线（爆破辅音时刻），切换 N_win=64 则水平双谐波条纹清晰分离但瞬态被横向抹宽！"
              }
            }
          }
        }

        AudioLabMode::CtcLattice => {
          let rep = ctc_repeat();
          let bp = ctc_blank();
          let (labels, alpha, can_skip, total_p) = compute_ctc_forward_lattice(rep, bp);
          let s_len = labels.len();

          rsx! {
            div { class: "space-y-3",
              div { class: "grid grid-cols-1 md:grid-cols-3 gap-3 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/70 dark:bg-slate-950/60 p-3 text-xs font-mono",
                div { class: "flex items-center gap-1.5",
                  button {
                    r#type: "button",
                    class: if !rep { "px-2.5 py-1.5 rounded bg-indigo-600 text-white font-bold cursor-pointer" } else { "px-2.5 py-1.5 rounded bg-slate-200 dark:bg-slate-800 cursor-pointer" },
                    onclick: move |_| ctc_repeat.set(false),
                    "目标词 \"cat\" (异字母可跳 ϵ)"
                  }
                  button {
                    r#type: "button",
                    class: if rep { "px-2.5 py-1.5 rounded bg-amber-600 text-white font-bold cursor-pointer" } else { "px-2.5 py-1.5 rounded bg-slate-200 dark:bg-slate-800 cursor-pointer" },
                    onclick: move |_| ctc_repeat.set(true),
                    "重复词 \"aa\" (必经中间 ϵ!)"
                  }
                }
                div {
                  div { class: "mb-1 font-semibold", "每帧空白符发射概率 P(ϵ) = {bp:.2}" }
                  input {
                    r#type: "range", min: "0.15", max: "0.75", step: "0.05", value: "{bp}",
                    class: "w-full accent-indigo-600 cursor-pointer",
                    oninput: move |e| { if let Ok(v) = e.value().parse::<f64>() { ctc_blank.set(v); } },
                  }
                }
                div { class: "flex items-center text-[11px] text-slate-600 dark:text-slate-300",
                  if can_skip {
                    "s=1('c') → s=3('a') 允许跨过空白符直接跳变（l_s ≠ l_{{s-2}}）"
                  } else {
                    "禁跳规则生效：s=1('a') 严禁跳到 s=3('a')，否则折叠后变成单 'a'！"
                  }
                }
              }

              div { class: "w-full overflow-x-auto rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950 p-3",
                svg {
                  view_box: "0 0 680 225",
                  class: "w-full min-w-[540px] h-auto font-mono select-none",
                  role: "img",
                  "aria-label": "CTC forward dynamic programming alpha lattice",
                  for (s, lbl) in labels.iter().enumerate() {
                    text { x: "32", y: "{44.0 + (s as f64) * 25.0}", text_anchor: "middle", font_size: "10.5", class: "fill-indigo-600 dark:fill-indigo-400 font-bold", "{lbl}" }
                  }
                  for t in 0..8_usize {
                    text { x: "{86.0 + (t as f64) * 72.0}", y: "20", text_anchor: "middle", font_size: "9.5", class: "fill-slate-500 font-bold", "t={t}" }
                    for s in 0..s_len {
                      {
                        let a_val = alpha[t][s];
                        let reachable = a_val > 1e-12;
                        let score_norm = if reachable { (1.0 + (a_val.ln() / 12.0)).clamp(0.18, 0.95) } else { 0.0 };
                        let txt = if reachable { format!("{:.0e}", a_val) } else { "0".to_string() };
                        svg_grid_cell(
                          &format!("ctc-{t}-{s}"),
                          56.0 + (t as f64) * 72.0,
                          28.0 + (s as f64) * 25.0,
                          60.0,
                          21.0,
                          heat_cell_cell_ctc(reachable, s >= s_len - 2 && t == 7, score_norm),
                          "1",
                          &txt,
                          if reachable { "fill-slate-900 dark:fill-white font-semibold" } else { "fill-slate-400" },
                        )
                      }
                    }
                  }
                }
              }

              div { class: "grid grid-cols-2 sm:grid-cols-4 gap-2.5 font-mono text-xs",
                {lab_metric_card("扩展标签序列长度 |Y'|", &format!("2|Y|+1 = {s_len}"), "text-indigo-600 dark:text-indigo-400")}
                {lab_metric_card("跨空白跳变 l₁ → l₃", if can_skip { "✓ 合法放行" } else { "✗ 强制拦截 (防折叠)" }, if can_skip { "text-emerald-600" } else { "text-amber-600" })}
                {lab_metric_card("末帧合法终点 α_{7,|Y'|-1}", &format!("{:.2e}", alpha[7][s_len - 1]), "text-slate-800 dark:text-slate-100")}
                {lab_metric_card("总边际似然 P(Y | X)", &format!("{total_p:.3e}"), "text-emerald-600 dark:text-emerald-400")}
              }
            }
          }
        }

        AudioLabMode::RvqCodebook => {
          let q_cnt = rvq_q();
          let zx = target_x();
          let zy = target_y();
          let (path, mses) = compute_rvq_stages(zx, zy, q_cnt);
          let map_px = |v: f64| -> f64 { 175.0 + (v / 2.0) * 115.0 };
          let map_py = |v: f64| -> f64 { 118.0 - (v / 2.0) * 82.0 };
          let mut chain_pts = String::with_capacity(256);
          for (i, &(px, py)) in path.iter().enumerate() {
            push_svg_pt(&mut chain_pts, i == 0, map_px(px), map_py(py));
          }
          let last_pt = path.last().copied().unwrap_or((0.0, 0.0));

          rsx! {
            div { class: "space-y-3",
              div { class: "grid grid-cols-1 md:grid-cols-3 gap-3 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/70 dark:bg-slate-950/60 p-3 text-xs font-mono",
                div {
                  div { class: "mb-1 font-semibold", "启用残差码本层数 Q = {q_cnt} / 6" }
                  input {
                    r#type: "range", min: "1", max: "6", step: "1", value: "{q_cnt}",
                    class: "w-full accent-indigo-600 cursor-pointer",
                    oninput: move |e| { if let Ok(v) = e.value().parse::<usize>() { rvq_q.set(v); } },
                  }
                }
                div {
                  div { class: "mb-1 font-semibold", "连续潜向量 z_x = {zx:+.2}" }
                  input {
                    r#type: "range", min: "-1.8", max: "1.8", step: "0.05", value: "{zx}",
                    class: "w-full accent-emerald-600 cursor-pointer",
                    oninput: move |e| { if let Ok(v) = e.value().parse::<f64>() { target_x.set(v); } },
                  }
                }
                div {
                  div { class: "mb-1 font-semibold", "连续潜向量 z_y = {zy:+.2}" }
                  input {
                    r#type: "range", min: "-1.8", max: "1.8", step: "0.05", value: "{zy}",
                    class: "w-full accent-amber-600 cursor-pointer",
                    oninput: move |e| { if let Ok(v) = e.value().parse::<f64>() { target_y.set(v); } },
                  }
                }
              }

              div { class: "w-full overflow-x-auto rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950 p-3",
                svg {
                  view_box: "0 0 680 220",
                  class: "w-full min-w-[540px] h-auto font-mono select-none",
                  role: "img",
                  "aria-label": "Residual Vector Quantization coarse-to-fine vector sum and MSE decay",
                  text { x: "175", y: "18", text_anchor: "middle", font_size: "10.5", class: "fill-slate-700 dark:fill-slate-200 font-bold",
                    "左：2D 潜空间逐级残差向量逼近 ẑ = ∑ e_q"
                  }
                  rect { x: "45", y: "28", width: "260", height: "180", rx: "6", class: "fill-slate-50 dark:fill-slate-900 stroke-slate-200 dark:stroke-slate-800", stroke_width: "1" }
                  line { x1: "45", y1: "118", x2: "305", y2: "118", class: "stroke-slate-200 dark:stroke-slate-800", stroke_width: "1" }
                  line { x1: "175", y1: "28", x2: "175", y2: "208", class: "stroke-slate-200 dark:stroke-slate-800", stroke_width: "1" }
                  polyline { points: "{chain_pts}", fill: "none", class: "stroke-indigo-600 dark:stroke-indigo-400", stroke_width: "2.4" }
                  circle { cx: "{map_px(zx):.1}", cy: "{map_py(zy):.1}", r: "5.5", class: "fill-rose-500" }
                  circle { cx: "{map_px(last_pt.0):.1}", cy: "{map_py(last_pt.1):.1}", r: "4.5", class: "fill-emerald-500 stroke-white dark:stroke-slate-950", stroke_width: "1.5" }

                  text { x: "500", y: "18", text_anchor: "middle", font_size: "10.5", class: "fill-slate-700 dark:fill-slate-200 font-bold",
                    "右：第 1..6 级残差量化误差 ‖z - ẑ_q‖₂² 指数衰减"
                  }
                  line { x1: "355", y1: "190", x2: "645", y2: "190", class: "stroke-slate-300 dark:stroke-slate-700", stroke_width: "1" }
                  for (q_i, &mse) in mses.iter().enumerate() {
                    {
                      let h_px = (mse / (mses[0].max(0.25) * 1.2)).min(1.0) * 140.0;
                      let cls = if q_i < q_cnt { "fill-indigo-600 dark:fill-indigo-400" } else { "fill-slate-300 dark:fill-slate-700" };
                      svg_vbar(&format!("rvq-m-{q_i}"), 368.0 + (q_i as f64) * 45.0, 190.0, 28.0, h_px, cls, &format!("{mse:.3}"), &format!("Q{}", q_i + 1))
                    }
                  }
                }
              }

              div { class: "grid grid-cols-2 sm:grid-cols-4 gap-2.5 font-mono text-xs",
                {lab_metric_card("首层粗量化误差 Q1 MSE", &format!("{:.4}", mses[0]), "text-amber-600 dark:text-amber-400")}
                {lab_metric_card(&format!("当前 Q{q_cnt} 残差 MSE"), &format!("{:.5}", mses[q_cnt - 1]), "text-emerald-600 dark:text-emerald-400")}
                {lab_metric_card("等效码本组合容量 K^Q", &format!("8^{q_cnt} = {}", 8_u64.pow(q_cnt as u32)), "text-indigo-600 dark:text-indigo-400")}
                {lab_metric_card("每帧传输码率 (K=8)", &format!("{} bits/frame", q_cnt * 3), "text-slate-800 dark:text-slate-100")}
              }
            }
          }
        }
      }
    }
  }
}

#[inline(never)]
fn heat_cell_cell_ctc(reachable: bool, is_final: bool, score: f64) -> &'static str {
  if is_final && reachable {
    "fill-emerald-200 dark:fill-emerald-900/80 stroke-emerald-500"
  } else if reachable {
    heat_cell_cls(score * 0.65, false)
  } else {
    "fill-slate-50 dark:fill-slate-900/60 stroke-slate-200 dark:stroke-slate-800"
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 内核 8：RoutingAndSamplingLab（Aux-Loss-Free MoE / Samplers / 投机解码 / CFG 外推）
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RoutingLabMode {
  MoeBalance,
  TokenSamplers,
  SpeculativeDecode,
  DiffusionCfg,
}

impl RoutingLabMode {
  fn parse_mode(raw: &str) -> Self {
    match raw.trim().to_ascii_lowercase().as_str() {
      "moe" | "aux_free" | "deepseek_moe" | "routing" => Self::MoeBalance,
      "sampling" | "samplers" | "top_p" | "min_p" | "temperature" => Self::TokenSamplers,
      "speculative" | "mtp" | "eagle" => Self::SpeculativeDecode,
      "cfg" | "diffusion_cfg" | "scheduler" | "time_shift" | "lora" | "peft" => Self::DiffusionCfg,
      _ => Self::MoeBalance,
    }
  }

  fn label(self) -> &'static str {
    match self {
      Self::MoeBalance => "Sparse MoE 无辅助损失动态偏置均衡",
      Self::TokenSamplers => "解码采样器 (Temp / Top-K / Top-P / Min-P)",
      Self::SpeculativeDecode => "无损投机采样 (修正拒绝采样恒等式)",
      Self::DiffusionCfg => "扩散 CFG 流形外推 & 分辨率 Time-Shift",
    }
  }
}

const NATURAL_EXPERT_PRIOR: [f64; 8] = [1.55, 1.15, 0.25, -0.10, -0.35, -0.55, -0.75, -0.95];

/// 仿真 8 专家 Top-2 路由在偏斜流量下的负载分布：
/// 返回 `(无均衡塌缩负载, 传统 AuxLoss 负载, DeepSeek 无辅助损失动态偏置负载, 最终动态偏置 b_i, 传统门控权重语义失真度)`。
pub(crate) fn simulate_moe_routing(
  batches: usize,
  bias_step_gamma: f64,
) -> ([f64; 8], [f64; 8], [f64; 8], [f64; 8], f64) {
  let n_batches = batches.clamp(1, 80);
  let mut collapse_drift = [0.0_f64; 8];
  let mut aux_penalty = [0.0_f64; 8];
  let mut dyn_bias = [0.0_f64; 8];
  let mut load_none = [0.0_f64; 8];
  let mut load_aux = [0.0_f64; 8];
  let mut load_free = [0.0_f64; 8];
  let win_start = n_batches.saturating_sub(8);
  let win_len = (n_batches - win_start).max(1) as f64;

  for b in 0..n_batches {
    let mut cnt_none = [0.0_f64; 8];
    let mut cnt_aux = [0.0_f64; 8];
    let mut cnt_free = [0.0_f64; 8];
    for tok in 0..32_usize {
      let mut s_none = [0.0_f64; 8];
      let mut s_aux = [0.0_f64; 8];
      let mut s_free = [0.0_f64; 8];
      for e in 0..8 {
        let wave = 0.65 * (((b * 7 + tok * 3 + e * 5) as f64) * 0.37).sin();
        let affinity = sigmoid(NATURAL_EXPERT_PRIOR[e] + wave);
        s_none[e] = affinity + collapse_drift[e];
        s_aux[e] = affinity - aux_penalty[e];
        // DeepSeek-V3 核心设计：动态偏置 b_e 仅加在 Top-K 选通分数上，绝不污染后续乘进残差流的门控权重！
        s_free[e] = affinity + dyn_bias[e];
      }
      for idx in top2_indices(&s_none) {
        cnt_none[idx] += 1.0;
      }
      for idx in top2_indices(&s_aux) {
        cnt_aux[idx] += 1.0;
      }
      for idx in top2_indices(&s_free) {
        cnt_free[idx] += 1.0;
      }
    }
    for e in 0..8 {
      let bn = cnt_none[e] / 64.0;
      let ba = cnt_aux[e] / 64.0;
      let bf = cnt_free[e] / 64.0;
      if b >= win_start {
        load_none[e] += bn / win_len;
        load_aux[e] += ba / win_len;
        load_free[e] += bf / win_len;
      }
      // 赢者通吃正反馈：被频繁选中的专家获得更多梯度更新，表征更强
      collapse_drift[e] += 0.012 * (bn - 0.125);
      aux_penalty[e] += 0.055 * (ba - 0.125);
      let diff = 0.125 - bf;
      let sign = if diff.abs() < 1e-4 {
        0.0
      } else if diff > 0.0 {
        1.0
      } else {
        -1.0
      };
      dyn_bias[e] += bias_step_gamma * sign;
    }
  }

  let aux_gate_distortion = aux_penalty.iter().map(|p| p.abs()).sum::<f64>() / 8.0;
  (load_none, load_aux, load_free, dyn_bias, aux_gate_distortion)
}

fn top2_indices(scores: &[f64; 8]) -> [usize; 2] {
  let mut best0 = 0_usize;
  let mut best1 = 1_usize;
  if scores[best1] > scores[best0] {
    std::mem::swap(&mut best0, &mut best1);
  }
  for (i, &v) in scores.iter().enumerate().skip(2) {
    if v > scores[best0] {
      best1 = best0;
      best0 = i;
    } else if v > scores[best1] {
      best1 = i;
    }
  }
  [best0, best1]
}

const CODE_LOGITS: [(&str, f64); 10] = [
  ("Ok(", 6.8),
  ("Err(", 3.9),
  ("return", 2.8),
  ("let", 1.9),
  ("match", 1.2),
  ("todo!", 0.4),
  ("unsafe", -0.2),
  ("banana", -0.8),
  ("???", -1.3),
  ("<glitch>", -1.9),
];

const CREATIVE_LOGITS: [(&str, f64); 10] = [
  ("星河", 3.2),
  ("微风", 2.95),
  ("篝火", 2.7),
  ("潮汐", 2.45),
  ("暮色", 2.15),
  ("远山", 1.85),
  ("松涛", 1.45),
  ("白鹭", 1.05),
  ("乱码#", -0.9),
  ("NULL", -1.6),
];

/// 计算 Temperature + Top-K + Top-P + Min-P 联合截断后的归一化采样分布。
pub(crate) fn compute_token_samplers(
  creative_mode: bool,
  temp: f64,
  top_k: usize,
  top_p: f64,
  min_p: f64,
) -> (Vec<(&'static str, f64, f64, bool)>, usize) {
  let src = if creative_mode { &CREATIVE_LOGITS } else { &CODE_LOGITS };
  let t_safe = temp.clamp(0.05, 3.0);
  let max_l = src[0].1 / t_safe;
  let mut raw_p = [0.0_f64; 10];
  let mut denom = 0.0;
  for (i, &(_, z)) in src.iter().enumerate() {
    let e = ((z / t_safe) - max_l).exp();
    raw_p[i] = e;
    denom += e;
  }
  for p in &mut raw_p {
    *p /= denom;
  }

  let p_max = raw_p[0];
  let k_lim = top_k.clamp(1, 10);
  let mut cum = 0.0;
  let mut keep = [false; 10];
  let mut kept_sum = 0.0;
  let mut kept_cnt = 0_usize;

  for i in 0..10 {
    let pass_k = i < k_lim;
    let pass_top_p = i == 0 || cum < top_p;
    let pass_min_p = i == 0 || raw_p[i] >= min_p * p_max;
    cum += raw_p[i];
    if pass_k && pass_top_p && pass_min_p {
      keep[i] = true;
      kept_sum += raw_p[i];
      kept_cnt += 1;
    }
  }

  let mut out = Vec::with_capacity(10);
  for i in 0..10 {
    let final_p = if keep[i] { raw_p[i] / kept_sum.max(1e-15) } else { 0.0 };
    out.push((src[i].0, raw_p[i], final_p, keep[i]));
  }
  (out, kept_cnt)
}

/// 验证投机解码（Speculative Decoding）修正拒绝采样恒等式：
/// $P_{\text{out}}(x) = q(x)\min(1, p(x)/q(x)) + (1 - \alpha)\frac{\max(0, p(x)-q(x))}{1-\alpha} \equiv p(x)$。
pub(crate) fn verify_speculative_decoding(
  draft_skew: f64,
  gamma_draft_len: usize,
) -> ([f64; 6], [f64; 6], [f64; 6], f64, f64, f64) {
  let target_p = [0.46, 0.24, 0.14, 0.09, 0.05, 0.02];
  let skew_raw = [0.20, 0.36, 0.18, 0.12, 0.09, 0.05];
  let s = draft_skew.clamp(0.0, 1.0);
  let mut draft_q = [0.0_f64; 6];
  let mut q_sum = 0.0;
  for i in 0..6 {
    draft_q[i] = (1.0 - s) * target_p[i] + s * skew_raw[i];
    q_sum += draft_q[i];
  }
  for q in &mut draft_q {
    *q /= q_sum;
  }

  let accept_alpha: f64 = target_p.iter().zip(draft_q.iter()).map(|(p, q)| p.min(*q)).sum();
  let rem_mass = (1.0 - accept_alpha).max(0.0);
  let mut out_marginal = [0.0_f64; 6];
  let mut max_err = 0.0_f64;

  for i in 0..6 {
    let beta_i = (target_p[i] / draft_q[i].max(1e-15)).min(1.0);
    let bonus_i =
      if rem_mass > 1e-14 { (target_p[i] - draft_q[i]).max(0.0) / rem_mass } else { 0.0 };
    out_marginal[i] = draft_q[i] * beta_i + rem_mass * bonus_i;
    max_err = max_err.max((out_marginal[i] - target_p[i]).abs());
  }

  let g = gamma_draft_len.clamp(1, 8) as i32;
  let expected_tokens = if (1.0 - accept_alpha).abs() < 1e-9 {
    (g + 1) as f64
  } else {
    (1.0 - accept_alpha.powi(g + 1)) / (1.0 - accept_alpha)
  };
  (target_p, draft_q, out_marginal, accept_alpha, max_err, expected_tokens)
}

#[component]
pub fn RoutingAndSamplingLab(initial_mode: String) -> Element {
  let mut mode = use_signal(move || RoutingLabMode::parse_mode(&initial_mode));
  let mut moe_batches = use_signal(|| 60_usize);
  let mut moe_gamma = use_signal(|| 0.015_f64);
  let mut creative = use_signal(|| false);
  let mut samp_temp = use_signal(|| 1.1_f64);
  let mut samp_topk = use_signal(|| 8_usize);
  let mut samp_topp = use_signal(|| 0.95_f64);
  let mut samp_minp = use_signal(|| 0.08_f64);
  let mut spec_skew = use_signal(|| 0.65_f64);
  let mut spec_gamma = use_signal(|| 5_usize);
  let mut cfg_w = use_signal(|| 5.5_f64);
  let mut cfg_rescale = use_signal(|| 0.0_f64);
  let mut flow_shift = use_signal(|| 3.0_f64);

  let cur_mode = mode();

  rsx! {
    div { class: "space-y-4",
      div { class: "flex flex-wrap gap-1.5",
        for m in [
          RoutingLabMode::MoeBalance,
          RoutingLabMode::TokenSamplers,
          RoutingLabMode::SpeculativeDecode,
          RoutingLabMode::DiffusionCfg,
        ] {
          {mode_btn(m.label(), cur_mode == m, EventHandler::new(move |_| mode.set(m)))}
        }
      }

      match cur_mode {
        RoutingLabMode::MoeBalance => {
          let b_cnt = moe_batches();
          let gm = moe_gamma();
          let (l_none, _, l_free, biases, aux_distort) = simulate_moe_routing(b_cnt, gm);
          let max_none = l_none.iter().copied().fold(0.0_f64, f64::max);
          let max_free = l_free.iter().copied().fold(0.0_f64, f64::max);

          rsx! {
            div { class: "space-y-3",
              div { class: "grid grid-cols-1 md:grid-cols-3 gap-3 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/70 dark:bg-slate-950/60 p-3 text-xs font-mono",
                div {
                  div { class: "mb-1 font-semibold", "训练演化步数 Batch = {b_cnt}" }
                  input {
                    r#type: "range", min: "1", max: "80", step: "1", value: "{b_cnt}",
                    class: "w-full accent-indigo-600 cursor-pointer",
                    oninput: move |e| { if let Ok(v) = e.value().parse::<usize>() { moe_batches.set(v); } },
                  }
                }
                div {
                  div { class: "mb-1 font-semibold", "动态偏置更新步长 γ = {gm:.3}" }
                  input {
                    r#type: "range", min: "0.000", max: "0.080", step: "0.005", value: "{gm}",
                    class: "w-full accent-emerald-600 cursor-pointer",
                    oninput: move |e| { if let Ok(v) = e.value().parse::<f64>() { moe_gamma.set(v); } },
                  }
                }
                div { class: "flex items-center text-[11px] text-slate-600 dark:text-slate-300",
                  "DeepSeek-V3 无辅助损失均衡：选通用 argtopK(s_i + b_i)，加权输出仍用纯净 s_i，彻底根除 Aux Loss 对主任务梯度的干扰！"
                }
              }

              div { class: "w-full overflow-x-auto rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950 p-3",
                svg {
                  view_box: "0 0 680 215",
                  class: "w-full min-w-[540px] h-auto font-mono select-none",
                  role: "img",
                  "aria-label": "Sparse MoE expert load collapse vs DeepSeek Aux-Loss-Free dynamic bias balancing",
                  text { x: "340", y: "18", text_anchor: "middle", font_size: "10.5", class: "fill-slate-700 dark:fill-slate-200 font-bold",
                    "8 个路由专家流量占比对比（玫红 = 无均衡坍缩 · 翠绿 = 无辅助损失动态偏置 b_i · 虚线 = 理想均值 12.5%）"
                  }
                  line { x1: "40", y1: "138", x2: "650", y2: "138", class: "stroke-amber-500/80", stroke_dasharray: "4 3", stroke_width: "1.3" }
                  line { x1: "40", y1: "180", x2: "650", y2: "180", class: "stroke-slate-300 dark:stroke-slate-700", stroke_width: "1" }
                  for e in 0..8_usize {
                    {
                      let gx = 52.0 + (e as f64) * 74.0;
                      let h_none = (l_none[e] / 0.45).min(1.0) * 148.0;
                      let h_free = (l_free[e] / 0.45).min(1.0) * 148.0;
                      rsx! {
                        g { key: "moe-e-{e}",
                          {svg_vbar(&format!("mn-{e}"), gx, 180.0, 22.0, h_none, "fill-rose-500/85", &format!("{:.0}%", l_none[e] * 100.0), "")}
                          {svg_vbar(&format!("mf-{e}"), gx + 25.0, 180.0, 22.0, h_free, "fill-emerald-500", &format!("{:.0}%", l_free[e] * 100.0), &format!("E{e}({:+.2})", biases[e]))}
                        }
                      }
                    }
                  }
                }
              }

              div { class: "grid grid-cols-2 sm:grid-cols-4 gap-2.5 font-mono text-xs",
                {lab_metric_card("无均衡最拥堵专家负载", &format!("{:.1}% (理想 12.5%)", max_none * 100.0), "text-rose-600 dark:text-rose-400")}
                {lab_metric_card("动态偏置最高专家负载", &format!("{:.1}%", max_free * 100.0), "text-emerald-600 dark:text-emerald-400")}
                {lab_metric_card("传统 AuxLoss 门控失真", &format!("{aux_distort:.3}"), "text-amber-600 dark:text-amber-400")}
                {lab_metric_card("Aux-Loss-Free 干扰梯度", "0.000 (纯非梯度调节)", "text-indigo-600 dark:text-indigo-400")}
              }
            }
          }
        }

        RoutingLabMode::TokenSamplers => {
          let is_cr = creative();
          let t = samp_temp();
          let tk = samp_topk();
          let tp = samp_topp();
          let mp = samp_minp();
          let (bars, kept_cnt) = compute_token_samplers(is_cr, t, tk, tp, mp);

          rsx! {
            div { class: "space-y-3",
              div { class: "flex flex-wrap gap-2 items-center text-xs font-mono",
                button {
                  r#type: "button",
                  class: if !is_cr { "px-2.5 py-1 rounded bg-indigo-600 text-white font-bold cursor-pointer" } else { "px-2.5 py-1 rounded bg-slate-200 dark:bg-slate-800 cursor-pointer" },
                  onclick: move |_| creative.set(false),
                  "场景 A：高确定性代码补全峰值"
                }
                button {
                  r#type: "button",
                  class: if is_cr { "px-2.5 py-1 rounded bg-indigo-600 text-white font-bold cursor-pointer" } else { "px-2.5 py-1 rounded bg-slate-200 dark:bg-slate-800 cursor-pointer" },
                  onclick: move |_| creative.set(true),
                  "场景 B：开放式创意写作平坦分布"
                }
              }

              div { class: "grid grid-cols-2 md:grid-cols-4 gap-3 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/70 dark:bg-slate-950/60 p-3 text-xs font-mono",
                div {
                  div { class: "mb-1 font-semibold", "Temp T = {t:.2}" }
                  input { r#type: "range", min: "0.2", max: "2.0", step: "0.1", value: "{t}", class: "w-full accent-indigo-600 cursor-pointer", oninput: move |e| { if let Ok(v) = e.value().parse::<f64>() { samp_temp.set(v); } } }
                }
                div {
                  div { class: "mb-1 font-semibold", "Top-K = {tk}" }
                  input { r#type: "range", min: "1", max: "10", step: "1", value: "{tk}", class: "w-full accent-indigo-600 cursor-pointer", oninput: move |e| { if let Ok(v) = e.value().parse::<usize>() { samp_topk.set(v); } } }
                }
                div {
                  div { class: "mb-1 font-semibold", "Top-P = {tp:.2}" }
                  input { r#type: "range", min: "0.30", max: "1.00", step: "0.05", value: "{tp}", class: "w-full accent-amber-600 cursor-pointer", oninput: move |e| { if let Ok(v) = e.value().parse::<f64>() { samp_topp.set(v); } } }
                }
                div {
                  div { class: "mb-1 font-semibold text-emerald-700 dark:text-emerald-400", "Min-P = {mp:.2}" }
                  input { r#type: "range", min: "0.00", max: "0.35", step: "0.02", value: "{mp}", class: "w-full accent-emerald-600 cursor-pointer", oninput: move |e| { if let Ok(v) = e.value().parse::<f64>() { samp_minp.set(v); } } }
                }
              }

              div { class: "w-full overflow-x-auto rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950 p-3",
                svg {
                  view_box: "0 0 680 210",
                  class: "w-full min-w-[540px] h-auto font-mono select-none",
                  role: "img",
                  "aria-label": "Autoregressive token sampling probabilities under Temperature, Top-K, Top-P, and Min-P",
                  line { x1: "35", y1: "178", x2: "655", y2: "178", class: "stroke-slate-300 dark:stroke-slate-700", stroke_width: "1" }
                  for (i, &(tok, raw_p, fin_p, alive)) in bars.iter().enumerate() {
                    {
                      let x = 44.0 + (i as f64) * 60.5;
                      let h_show = (if alive { fin_p } else { raw_p }).min(1.0) * 142.0;
                      let cls = if alive { "fill-indigo-600 dark:fill-indigo-400" } else { "fill-rose-300/60 dark:fill-rose-950/70" };
                      let top_txt = if alive { format!("{:.1}%", fin_p * 100.0) } else { "截断".to_string() };
                      svg_vbar(&format!("spb-{i}"), x, 178.0, 38.0, h_show, cls, &top_txt, tok)
                    }
                  }
                }
              }

              div { class: "grid grid-cols-2 sm:grid-cols-4 gap-2.5 font-mono text-xs",
                {lab_metric_card("存活候选词数量", &format!("{kept_cnt} / 10"), "text-indigo-600 dark:text-indigo-400")}
                {lab_metric_card("首选词原始概率 p_max", &format!("{:.1}%", bars[0].1 * 100.0), "text-slate-800 dark:text-slate-100")}
                {lab_metric_card("Min-P 动态截断门槛", &format!("p ≥ {:.2}%", bars[0].1 * mp * 100.0), "text-emerald-600 dark:text-emerald-400")}
                {lab_metric_card("重归一化首选词概率", &format!("{:.1}%", bars[0].2 * 100.0), "text-amber-600 dark:text-amber-400")}
              }
            }
          }
        }

        RoutingLabMode::SpeculativeDecode => {
          let skew = spec_skew();
          let gm = spec_gamma();
          let (p_tar, q_drf, p_out, alpha, max_err, exp_toks) = verify_speculative_decoding(skew, gm);

          rsx! {
            div { class: "space-y-3",
              div { class: "grid grid-cols-1 md:grid-cols-3 gap-3 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/70 dark:bg-slate-950/60 p-3 text-xs font-mono",
                div {
                  div { class: "mb-1 font-semibold", "草稿模型分布偏移度 δ = {skew:.2}" }
                  input { r#type: "range", min: "0.0", max: "1.0", step: "0.05", value: "{skew}", class: "w-full accent-indigo-600 cursor-pointer", oninput: move |e| { if let Ok(v) = e.value().parse::<f64>() { spec_skew.set(v); } } }
                }
                div {
                  div { class: "mb-1 font-semibold", "单步连猜长度 γ = {gm} tokens" }
                  input { r#type: "range", min: "1", max: "8", step: "1", value: "{gm}", class: "w-full accent-emerald-600 cursor-pointer", oninput: move |e| { if let Ok(v) = e.value().parse::<usize>() { spec_gamma.set(v); } } }
                }
                div { class: "flex items-center text-[11px] text-slate-600 dark:text-slate-300",
                  "不管小模型猜得多歪（δ=1.0），接受概率 min(1, p/q) 配合残差重采样 (p-q)_+ 保证最终输出与大模型 100% 逐位同分布！"
                }
              }

              div { class: "w-full overflow-x-auto rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950 p-3",
                svg {
                  view_box: "0 0 680 205",
                  class: "w-full min-w-[540px] h-auto font-mono select-none",
                  role: "img",
                  "aria-label": "Speculative decoding target vs draft vs exact output distribution",
                  text { x: "340", y: "18", text_anchor: "middle", font_size: "10.5", class: "fill-slate-700 dark:fill-slate-200 font-bold",
                    "三柱对照：靛蓝 = 目标大模型 p(x) · 琥珀 = 草稿小模型 q(x) · 翠绿 = 投机校验输出 P_spec(x)"
                  }
                  line { x1: "40", y1: "174", x2: "645", y2: "174", class: "stroke-slate-300 dark:stroke-slate-700", stroke_width: "1" }
                  for i in 0..6_usize {
                    {
                      let gx = 55.0 + (i as f64) * 98.0;
                      rsx! {
                        g { key: "sp-g-{i}",
                          {svg_vbar(&format!("spt-{i}"), gx, 174.0, 20.0, p_tar[i] * 260.0, "fill-indigo-600", &format!("{:.2}", p_tar[i]), "")}
                          {svg_vbar(&format!("spq-{i}"), gx + 22.0, 174.0, 20.0, q_drf[i] * 260.0, "fill-amber-500", &format!("{:.2}", q_drf[i]), &format!("w{i}"))}
                          {svg_vbar(&format!("spo-{i}"), gx + 44.0, 174.0, 20.0, p_out[i] * 260.0, "fill-emerald-500", &format!("{:.2}", p_out[i]), "")}
                        }
                      }
                    }
                  }
                }
              }

              div { class: "grid grid-cols-2 sm:grid-cols-4 gap-2.5 font-mono text-xs",
                {lab_metric_card("单步平均接受率 α = 1 - ½‖p-q‖₁", &format!("{:.1}%", alpha * 100.0), "text-amber-600 dark:text-amber-400")}
                {lab_metric_card("大模型单次前向期望产出", &format!("{exp_toks:.2} tokens"), "text-indigo-600 dark:text-indigo-400")}
                {lab_metric_card("无损校验误差 max|P_spec - p|", &format!("{max_err:.2e}"), "text-emerald-600 dark:text-emerald-400")}
                {lab_metric_card("首位拒绝白赚残差 Token", "+1 Bonus Token", "text-slate-800 dark:text-slate-100")}
              }
            }
          }
        }

        RoutingLabMode::DiffusionCfg => {
          let w = cfg_w();
          let phi = cfg_rescale();
          let m_shift = flow_shift();
          // 无条件速度 v_0 = (0.55, 0.35), 条件速度 v_c = (0.82, 0.62)
          let (v0x, v0y) = (0.55_f64, 0.35_f64);
          let (vcx, vcy) = (0.82_f64, 0.62_f64);
          let raw_gx = v0x + w * (vcx - v0x);
          let raw_gy = v0y + w * (vcy - v0y);
          let norm_c = (vcx * vcx + vcy * vcy).sqrt();
          let norm_g = (raw_gx * raw_gx + raw_gy * raw_gy).sqrt().max(1e-9);
          let scale_g = phi * (norm_c / norm_g) + (1.0 - phi);
          let (gx, gy) = (raw_gx * scale_g, raw_gy * scale_g);
          let overshoot = ((gx * gx + gy * gy).sqrt() - 1.25).max(0.0);

          let mut shift_pts = String::with_capacity(600);
          let mut lin_pts = String::with_capacity(128);
          push_svg_pt(&mut lin_pts, true, 370.0, 185.0);
          push_svg_pt(&mut lin_pts, false, 640.0, 35.0);
          for st in 0..41_usize {
            let t = (st as f64) / 40.0;
            let t_shifted = (m_shift * t) / (1.0 + (m_shift - 1.0) * t);
            push_svg_pt(&mut shift_pts, st == 0, 370.0 + t * 270.0, 185.0 - t_shifted * 150.0);
          }

          rsx! {
            div { class: "space-y-3",
              div { class: "grid grid-cols-1 md:grid-cols-3 gap-3 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/70 dark:bg-slate-950/60 p-3 text-xs font-mono",
                div {
                  div { class: "mb-1 font-semibold", "引导强度 CFG Scale w = {w:.1}" }
                  input { r#type: "range", min: "1.0", max: "11.0", step: "0.5", value: "{w}", class: "w-full accent-indigo-600 cursor-pointer", oninput: move |e| { if let Ok(v) = e.value().parse::<f64>() { cfg_w.set(v); } } }
                }
                div {
                  div { class: "mb-1 font-semibold", "CFG Rescale φ = {phi:.2}（防过曝饱和）" }
                  input { r#type: "range", min: "0.0", max: "1.0", step: "0.1", value: "{phi}", class: "w-full accent-emerald-600 cursor-pointer", oninput: move |e| { if let Ok(v) = e.value().parse::<f64>() { cfg_rescale.set(v); } } }
                }
                div {
                  div { class: "mb-1 font-semibold", "高分流匹配偏移 m = {m_shift:.1} (FLUX/Wan)" }
                  input { r#type: "range", min: "1.0", max: "6.0", step: "0.5", value: "{m_shift}", class: "w-full accent-amber-600 cursor-pointer", oninput: move |e| { if let Ok(v) = e.value().parse::<f64>() { flow_shift.set(v); } } }
                }
              }

              div { class: "w-full overflow-x-auto rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950 p-3",
                svg {
                  view_box: "0 0 680 215",
                  class: "w-full min-w-[540px] h-auto font-mono select-none",
                  role: "img",
                  "aria-label": "Classifier-Free Guidance vector extrapolation and Rectified Flow resolution time shift",
                  text { x: "175", y: "18", text_anchor: "middle", font_size: "10.5", class: "fill-slate-700 dark:fill-slate-200 font-bold",
                    "左：CFG 向量外推 v_∅ + w(v_c - v_∅) 与自然流形边界"
                  }
                  circle { cx: "65", cy: "185", r: "145", fill: "none", class: "stroke-emerald-500/50", stroke_dasharray: "5 4", stroke_width: "2" }
                  line { x1: "65", y1: "185", x2: "{65.0 + v0x * 95.0:.1}", y2: "{185.0 - v0y * 95.0:.1}", class: "stroke-slate-400", stroke_width: "2" }
                  line { x1: "65", y1: "185", x2: "{65.0 + vcx * 95.0:.1}", y2: "{185.0 - vcy * 95.0:.1}", class: "stroke-indigo-500", stroke_width: "2.2" }
                  line { x1: "65", y1: "185", x2: "{(65.0 + gx * 68.0).min(325.0):.1}", y2: "{(185.0 - gy * 68.0).max(28.0):.1}", class: if overshoot > 0.4 { "stroke-rose-500" } else { "stroke-emerald-500" }, stroke_width: "2.8" }
                  circle { cx: "{(65.0 + gx * 68.0).min(325.0):.1}", cy: "{(185.0 - gy * 68.0).max(28.0):.1}", r: "5", class: if overshoot > 0.4 { "fill-rose-500" } else { "fill-emerald-500" } }

                  text { x: "505", y: "18", text_anchor: "middle", font_size: "10.5", class: "fill-slate-700 dark:fill-slate-200 font-bold",
                    "右：分辨率噪声时间偏移 t' = m·t / (1 + (m-1)t)"
                  }
                  rect { x: "370", y: "35", width: "270", height: "150", rx: "5", class: "fill-slate-50 dark:fill-slate-900 stroke-slate-200 dark:stroke-slate-800", stroke_width: "1" }
                  polyline { points: "{lin_pts}", fill: "none", class: "stroke-slate-400", stroke_dasharray: "4 3", stroke_width: "1.4" }
                  polyline { points: "{shift_pts}", fill: "none", class: "stroke-amber-500", stroke_width: "2.5" }
                }
              }

              div { class: "grid grid-cols-2 sm:grid-cols-4 gap-2.5 font-mono text-xs",
                {lab_metric_card("引导合成速度模长 ‖v_cfg‖", &format!("{:.3}", (gx * gx + gy * gy).sqrt()), "text-indigo-600 dark:text-indigo-400")}
                {lab_metric_card("流形越界过曝量 Overshoot", &format!("{overshoot:.3}"), if overshoot > 0.4 { "text-rose-600 dark:text-rose-400" } else { "text-emerald-600 dark:text-emerald-400" })}
                {lab_metric_card("中点 t=0.5 偏移后噪声步 t'", &format!("{:.3}", (m_shift * 0.5) / (1.0 + (m_shift - 1.0) * 0.5)), "text-amber-600 dark:text-amber-400")}
                {lab_metric_card("双次前向开销 NFE", "2× Denoiser / step", "text-slate-800 dark:text-slate-100")}
              }
            }
          }
        }
      }
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn rnn_bptt_vanishing_exploding_and_direction_preserving_clipping() {
    let (_, g_vanish, _) = simulate_rnn_bptt(0.55, None);
    assert!(g_vanish[0].abs() < 1e-3, "spectral radius < 1 must vanish across 15 steps");

    let (_, g_raw, raw_norm) = simulate_rnn_bptt(1.42, None);
    let (_, g_clip, _) = simulate_rnn_bptt(1.42, Some(2.0));
    assert!(raw_norm > 5.0, "large recurrent weight must explode without clipping");
    let clip_norm = g_clip.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!((clip_norm - 2.0).abs() < 1e-9, "global norm clipping must clamp norm to 2.0");
    // 验证方向余弦严格等于 1.0
    let dot: f64 = g_raw.iter().zip(g_clip.iter()).map(|(a, b)| a * b).sum();
    let cos = dot / (raw_norm * clip_norm);
    assert!(
      (cos - 1.0).abs() < 1e-12,
      "global norm clipping must preserve exact gradient direction"
    );
  }

  #[test]
  fn gated_deltanet_erases_old_value_on_key_overwrite_unlike_linear_attention() {
    // 走完第 7 步（二次覆写 K_A -> V3），α=1.0, β=1.0 时 Delta 规则应几乎无损检索最新 V3
    let (_, _, err_lin, err_delta) = simulate_deltanet_vs_linear(1.0, 1.0, 7);
    assert!(
      err_lin > 1.2,
      "naive linear accumulation must suffer severe crosstalk on key overwrite"
    );
    assert!(err_delta < 1e-12, "DeltaNet with α=1, β=1 on unit key overwrite must retrieve exact latest value, got {err_delta}");
  }

  #[test]
  fn selective_mamba_shields_state_from_filler_tokens() {
    let lti = simulate_selective_mamba(0.0);
    let sel = simulate_selective_mamba(1.0);
    // t=2 写入 "#409" 后历经 t=3,4,5 三个语气噪声词 ("uh", "um", "wait")：选择性 Mamba 的 Ā_3 接近 1.0
    assert!(sel[3].1 > 0.96, "selective Mamba must keep A_bar close to 1 on filler tokens");
    assert!(
      sel[5].2.abs() > lti[5].2.abs(),
      "selective Mamba must preserve entity state better than fixed-step LTI"
    );
  }

  #[test]
  fn sdpa_unscaled_jacobian_collapses_at_large_dk_while_scaled_stays_healthy() {
    let (_, _, ent_scaled, jac_scaled) = compute_sdpa_matrix(256, true);
    let (_, _, ent_unscaled, jac_unscaled) = compute_sdpa_matrix(256, false);
    assert!(ent_scaled > 1.2 && jac_scaled > 0.15);
    assert!(
      ent_unscaled < 0.05 && jac_unscaled < 0.01,
      "unscaled d_k=256 must saturate softmax and kill Jacobian"
    );
  }

  #[test]
  fn flash_attention_online_softmax_matches_full_softmax_to_machine_precision() {
    let final_step = run_flash_online_softmax(3);
    assert!(
      final_step.max_abs_diff < 1e-14,
      "Online Softmax tiling must match full materialized Softmax to IEEE-754 precision, diff={}",
      final_step.max_abs_diff
    );
  }

  #[test]
  fn ctc_lattice_forbids_skip_blank_on_repeated_character_only() {
    let (_, _, can_skip_cat, p_cat) = compute_ctc_forward_lattice(false, 0.35);
    let (_, _, can_skip_aa, p_aa) = compute_ctc_forward_lattice(true, 0.35);
    assert!(can_skip_cat, "distinct adjacent chars 'c' -> 'a' may skip intermediate blank");
    assert!(!can_skip_aa, "identical adjacent chars 'a' -> 'a' must NOT skip intermediate blank");
    assert!(p_cat > 0.0 && p_aa > 0.0);
  }

  #[test]
  fn rvq_residual_error_monotonically_decreases_with_codebook_depth() {
    let (_, mses) = compute_rvq_stages(1.3, -0.9, 6);
    for q in 1..6 {
      assert!(mses[q] <= mses[q - 1] + 1e-12, "RVQ stage {q} MSE must not increase");
    }
    assert!(mses[5] < mses[0] * 0.05, "6-stage RVQ must reduce quantization MSE by >20x");
  }

  #[test]
  fn aux_loss_free_moe_balances_skewed_experts_without_gate_distortion() {
    let (l_none, _, l_free, _, _) = simulate_moe_routing(70, 0.015);
    let max_none = l_none.iter().copied().fold(0.0_f64, f64::max);
    let max_free = l_free.iter().copied().fold(0.0_f64, f64::max);
    assert!(max_none > 0.30, "unbalanced MoE must exhibit expert load concentration");
    assert!(
      max_free < 0.18,
      "Aux-Loss-Free dynamic bias must balance skewed experts near 12.5%, got {max_free}"
    );
  }

  #[test]
  fn speculative_decoding_rejection_plus_residual_is_strictly_lossless() {
    for step in 0..=10 {
      let skew = (step as f64) * 0.1;
      let (_, _, _, alpha, max_err, exp_toks) = verify_speculative_decoding(skew, 5);
      assert!(alpha > 0.0 && alpha <= 1.000_001);
      assert!(exp_toks >= 1.0);
      assert!(
        max_err < 1e-14,
        "speculative decoding marginal distribution must match target p(x) identically at skew={skew}, err={max_err}"
      );
    }
  }
}
