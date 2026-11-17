//! 纯 Rust WASM 深度学习与大模型算子动态交互实验室（M1 四大核心数值内核 + `<ComponentLab />` MDX 组件）。
//!
//! 设计准则：
//! - **真算，不写空洞动画 DSL**：所有曲线、卷积感受野、转置卷积重叠热力图、半精度平方溢出与 RoPE/YaRN
//!   复平面旋转均由纯 Rust `f64` 实时精确计算，SSR 首屏直出默认帧 SVG，水合零抖动；
//! - **严格兼容无 `'unsafe-eval'` CSP**：自动步进仅复用 `widgets::browser::sleep_ms`，零内联脚本求值；
//! - **全站三处复用**：既驱动 `/ai/components/:slug` 顶部交互台，也支持在课程/文档/博客中通过
//!   `<ComponentLab kind="grid_kernel" mode="depthwise_separable" />` 或 `<ComponentLab id="..." />` 嵌入。

use std::collections::HashMap;
use std::f64::consts::PI;

use app_core::i18n::Language;
use dioxus::prelude::*;
use dioxus_shadcn::card_class;
use serde::{Deserialize, Serialize};

fn current_lang() -> Language {
  try_consume_context::<Signal<Language>>().map(|s| s()).unwrap_or_default()
}

/// 八大可复用算子交互内核分类（M1 完整交付前 4 套高频空间/数值/几何内核，M2 补齐后 4 套序列/注意力/音频/路由内核）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ComponentLabKind {
  #[default]
  Activation,
  GridKernel,
  NormPrecision,
  Positional,
  SequenceState,
  AttentionMatrix,
  AudioSpectral,
  RoutingSampling,
}

impl ComponentLabKind {
  pub fn parse_str(raw: &str) -> Self {
    match raw.trim().to_ascii_lowercase().as_str() {
      "activation" | "activations" | "act" => Self::Activation,
      "grid_kernel" | "conv" | "convolution" | "pooling" | "pool" => Self::GridKernel,
      "norm_precision" | "norm" | "normalization" | "adaln" | "quant" => Self::NormPrecision,
      "positional" | "rope" | "pos" | "yarn" => Self::Positional,
      "sequence_state" | "rnn" | "lstm" | "ssm" | "mamba" => Self::SequenceState,
      "attention_matrix" | "attention" | "attn" | "mla" | "flash" => Self::AttentionMatrix,
      "audio_spectral" | "audio" | "stft" | "mel" | "rvq" => Self::AudioSpectral,
      "routing_sampling" | "moe" | "sampling" | "speculative" => Self::RoutingSampling,
      _ => Self::Activation,
    }
  }

  pub fn badge_label(self, lang: Language) -> &'static str {
    match self {
      Self::Activation => lang.pick("激活与门控曲线引擎", "Activation & Gating Lab"),
      Self::GridKernel => lang.pick("离散网格卷积/池化步进引擎", "Grid Conv & Pool Lab"),
      Self::NormPrecision => lang.pick("三维归一化与浮点溢出引擎", "Norm & Precision Lab"),
      Self::Positional => lang.pick("复平面旋转与外推几何引擎", "Rotary & Extrapolation Lab"),
      Self::SequenceState => lang.pick("序列状态记忆引擎", "Sequence State Lab"),
      Self::AttentionMatrix => lang.pick("注意力矩阵与分块引擎", "Attention Matrix Lab"),
      Self::AudioSpectral => lang.pick("时频谱与神经码本引擎", "Audio Spectral Lab"),
      Self::RoutingSampling => lang.pick("稀疏路由与采样引擎", "Routing & Sampling Lab"),
    }
  }
}

/// 顶层交互实验台分发组件。
#[component]
pub fn ComponentLabWorkbench(
  kind: ComponentLabKind,
  #[props(default)] initial_mode: String,
  #[props(default = false)] embedded: bool,
) -> Element {
  let lang = current_lang();
  let outer_cls = if embedded {
    format!("not-prose my-6 {}", card_class("p-4 sm:p-5 bg-white dark:bg-slate-900/80"))
  } else {
    card_class("p-4 sm:p-6 bg-white dark:bg-slate-900/70")
  };

  rsx! {
    div { class: "{outer_cls}",
      div { class: "flex flex-wrap items-center justify-between gap-2 pb-3 mb-4 border-b border-slate-200 dark:border-slate-800",
        div { class: "flex items-center gap-2",
          span { class: "px-2 py-0.5 rounded-full bg-indigo-600 text-white text-[10px] font-bold uppercase tracking-wider",
            "Interactive WASM Lab"
          }
          span { class: "text-xs font-bold text-slate-800 dark:text-slate-100",
            "{kind.badge_label(lang)}"
          }
        }
        span { class: "text-[11px] font-mono text-slate-400 dark:text-slate-500",
          {lang.pick("纯 Rust 实时浮点计算 · 零外部脚本", "Pure Rust f64 Kernel · SSR Ready")}
        }
      }

      match kind {
        ComponentLabKind::Activation => rsx! { ActivationCurveLab { initial_mode } },
        ComponentLabKind::GridKernel => rsx! { GridSlidingKernelLab { initial_mode } },
        ComponentLabKind::NormPrecision => rsx! { NormAndPrecisionLab { initial_mode } },
        ComponentLabKind::Positional => rsx! { PositionalGeometryLab { initial_mode } },
        ComponentLabKind::SequenceState
        | ComponentLabKind::AttentionMatrix
        | ComponentLabKind::AudioSpectral
        | ComponentLabKind::RoutingSampling => rsx! {
          PositionalGeometryLab { initial_mode }
        },
      }
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 内核 1：ActivationCurveLab（涵盖 17 种经典与现代激活函数、导数衰减与门控切片）
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ActivationFn {
  Sigmoid,
  Tanh,
  HardSigmoid,
  ReLU,
  LeakyReLU,
  ReLU6,
  SquaredReLU,
  Elu,
  Selu,
  Softplus,
  GELUExact,
  GELUTanh,
  SiLU,
  HardSwish,
  Mish,
  SwiGLUSlice,
  Snake,
}

impl ActivationFn {
  const ALL: &'static [Self] = &[
    Self::ReLU,
    Self::LeakyReLU,
    Self::GELUExact,
    Self::GELUTanh,
    Self::SiLU,
    Self::SwiGLUSlice,
    Self::Sigmoid,
    Self::Tanh,
    Self::Elu,
    Self::Selu,
    Self::Softplus,
    Self::Mish,
    Self::ReLU6,
    Self::HardSigmoid,
    Self::HardSwish,
    Self::SquaredReLU,
    Self::Snake,
  ];

  fn parse_mode(raw: &str) -> Self {
    match raw.trim().to_ascii_lowercase().as_str() {
      "sigmoid" => Self::Sigmoid,
      "tanh" => Self::Tanh,
      "hard_sigmoid" | "hardsigmoid" => Self::HardSigmoid,
      "relu" => Self::ReLU,
      "leaky_relu" | "leakyrelu" | "prelu" => Self::LeakyReLU,
      "relu6" => Self::ReLU6,
      "relu2" | "squared_relu" => Self::SquaredReLU,
      "elu" => Self::Elu,
      "selu" => Self::Selu,
      "softplus" => Self::Softplus,
      "gelu" | "gelu_exact" => Self::GELUExact,
      "gelu_tanh" => Self::GELUTanh,
      "silu" | "swish" => Self::SiLU,
      "hard_swish" | "hardswish" => Self::HardSwish,
      "mish" => Self::Mish,
      "swiglu" | "geglu" | "glu" => Self::SwiGLUSlice,
      "snake" | "siren" | "periodic" => Self::Snake,
      _ => Self::SwiGLUSlice,
    }
  }

  fn label(self) -> &'static str {
    match self {
      Self::Sigmoid => "Sigmoid",
      Self::Tanh => "Tanh",
      Self::HardSigmoid => "HardSigmoid",
      Self::ReLU => "ReLU",
      Self::LeakyReLU => "LeakyReLU(α)",
      Self::ReLU6 => "ReLU6",
      Self::SquaredReLU => "ReLU²",
      Self::Elu => "ELU(α)",
      Self::Selu => "SELU",
      Self::Softplus => "Softplus(β)",
      Self::GELUExact => "GELU (Exact)",
      Self::GELUTanh => "GELU (Tanh)",
      Self::SiLU => "SiLU / Swish(β)",
      Self::HardSwish => "HardSwish",
      Self::Mish => "Mish",
      Self::SwiGLUSlice => "SwiGLU (Gate)",
      Self::Snake => "Snake(a)",
    }
  }

  fn formula_note(self) -> &'static str {
    match self {
      Self::Sigmoid => "σ(x) = 1 / (1 + e⁻ˣ)，最大导数仅 0.25，深层连乘必然梯度消失",
      Self::Tanh => "tanh(x) = (eˣ - e⁻ˣ)/(eˣ + e⁻ˣ)，零中心输出但两端仍指数饱和",
      Self::HardSigmoid => "clamp((x + 3)/6, 0, 1)，分段线性免指数指令，端侧量化友好",
      Self::ReLU => "max(0, x)，正半轴导数恒为 1，但 x < 0 导数恒为 0（Dead ReLU 死区）",
      Self::LeakyReLU => "x ≥ 0 ? x : α·x，为负半轴保留非零斜率 α 防止神经元永久死亡",
      Self::ReLU6 => "clamp(x, 0, 6)，限制上界为 6 以适配移动端 INT8 定点动态范围",
      Self::SquaredReLU => "max(0, x)²，二阶超稀疏激活（Primer / Nemotron 采用）",
      Self::Elu => "x ≥ 0 ? x : α(eˣ - 1)，负半轴软饱和使激活均值接近 0",
      Self::Selu => "λ·ELU(x, α_selu)，不动点定理保障全连接层均值 0、方差 1 自归一化",
      Self::Softplus => "(1/β)·ln(1 + e^(βx))，ReLU 的解析光滑近似，导数恰为 Sigmoid",
      Self::GELUExact => "0.5·x·(1 + erf(x / √2))，按输入幅值分位数做高斯随机门控",
      Self::GELUTanh => "0.5·x·(1 + tanh(√(2/π)·(x + 0.044715x³)))，与精确解最大残差约 4.73e-4",
      Self::SiLU => "x · σ(βx)，平滑非单调负波谷（x ≈ -1.28 处微负）救活负边缘梯度",
      Self::HardSwish => "x · clamp((x + 3)/6, 0, 1)，MobileNetV3 零超越函数近似",
      Self::Mish => "x · tanh(ln(1 + eˣ))，高阶连续可微视觉骨干激活",
      Self::SwiGLUSlice => "SiLU(x) ⊙ (α·x + 0.6)，双线性门控乘积赋予局部二次表达力（LLM 标配）",
      Self::Snake => "x + sin²(a·x)/a，兼具单调趋势与周期高频振荡（BigVGAN 神经声码器核心）",
    }
  }

  fn eval(self, x: f64, param: f64) -> f64 {
    let p = param.max(0.05);
    match self {
      Self::Sigmoid => sigmoid(x),
      Self::Tanh => x.tanh(),
      Self::HardSigmoid => ((x + 3.0) / 6.0).clamp(0.0, 1.0),
      Self::ReLU => x.max(0.0),
      Self::LeakyReLU => {
        if x >= 0.0 {
          x
        } else {
          0.1 * p * x
        }
      }
      Self::ReLU6 => x.clamp(0.0, 6.0),
      Self::SquaredReLU => {
        let r = x.max(0.0);
        r * r
      }
      Self::Elu => {
        if x >= 0.0 {
          x
        } else {
          p * (x.exp() - 1.0)
        }
      }
      Self::Selu => {
        const LAMBDA: f64 = 1.050_700_987_355_480_5;
        const ALPHA: f64 = 1.673_263_242_354_377_2;
        if x >= 0.0 {
          LAMBDA * x
        } else {
          LAMBDA * ALPHA * (x.exp() - 1.0)
        }
      }
      Self::Softplus => softplus(p * x) / p,
      Self::GELUExact => 0.5 * x * (1.0 + erf_approx(x * std::f64::consts::FRAC_1_SQRT_2)),
      Self::GELUTanh => {
        let k = (2.0 / PI).sqrt();
        0.5 * x * (1.0 + (k * (x + 0.044_715 * x * x * x)).tanh())
      }
      Self::SiLU => x * sigmoid(p * x),
      Self::HardSwish => x * ((x + 3.0) / 6.0).clamp(0.0, 1.0),
      Self::Mish => x * softplus(x).tanh(),
      Self::SwiGLUSlice => (x * sigmoid(x)) * (0.55 * p * x + 0.6),
      Self::Snake => {
        let freq = (1.5 * p).max(0.2);
        let s = (freq * x).sin();
        x + (s * s) / freq
      }
    }
  }

  fn deriv(self, x: f64, param: f64) -> f64 {
    let h = 1e-5;
    (self.eval(x + h, param) - self.eval(x - h, param)) / (2.0 * h)
  }
}

fn sigmoid(x: f64) -> f64 {
  if x >= 0.0 {
    1.0 / (1.0 + (-x).exp())
  } else {
    let e = x.exp();
    e / (1.0 + e)
  }
}

fn softplus(x: f64) -> f64 {
  x.max(0.0) + (1.0 + (-x.abs()).exp()).ln()
}

/// Abramowitz & Stegun 7.1.26 高精度误差函数近似（最大绝对误差 < 1.5e-7）。
fn erf_approx(x: f64) -> f64 {
  let sign = if x < 0.0 { -1.0 } else { 1.0 };
  let ax = x.abs();
  let t = 1.0 / (1.0 + 0.327_591_1 * ax);
  let poly = t
    * (0.254_829_592
      + t * (-0.284_496_736 + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
  sign * (1.0 - poly * (-ax * ax).exp())
}

#[component]
fn ActivationCurveLab(initial_mode: String) -> Element {
  let init_primary = ActivationFn::parse_mode(&initial_mode);
  let init_cmp = if init_primary == ActivationFn::GELUExact {
    Some(ActivationFn::GELUTanh)
  } else if init_primary == ActivationFn::ReLU {
    Some(ActivationFn::SiLU)
  } else {
    Some(ActivationFn::ReLU)
  };

  let mut primary = use_signal(move || init_primary);
  let mut compare: Signal<Option<ActivationFn>> = use_signal(move || init_cmp);
  let mut probe_x = use_signal(|| -0.8_f64);
  let mut shape_param = use_signal(|| 1.0_f64);
  let mut show_deriv = use_signal(|| true);

  let p_fn = primary();
  let cmp_fn = compare();
  let x0 = probe_x();
  let alpha = shape_param();

  // 坐标映射：x ∈ [-5, 5] -> [55, 655] (width 600), y ∈ [-2.0, 5.0] -> [245, 25] (height 220)
  let map_x = |x: f64| -> f64 { 55.0 + ((x + 5.0) / 10.0) * 600.0 };
  let map_y = |y: f64| -> f64 { 245.0 - ((y.clamp(-2.0, 5.0) + 2.0) / 7.0) * 220.0 };

  let mut pts_primary = String::with_capacity(2048);
  let mut pts_deriv = String::with_capacity(2048);
  let mut pts_cmp = String::with_capacity(2048);
  let mut dead_spans: Vec<(f64, f64)> = Vec::new();
  let mut in_dead: Option<f64> = None;
  let mut max_cmp_diff = 0.0_f64;

  for step in 0..=160 {
    let x = -5.0 + (step as f64) * (10.0 / 160.0);
    let y = p_fn.eval(x, alpha);
    let dy = p_fn.deriv(x, alpha);
    let sx = map_x(x);
    let sy = map_y(y);
    let sdy = map_y(dy);
    if step > 0 {
      pts_primary.push(' ');
      pts_deriv.push(' ');
    }
    pts_primary.push_str(&format!("{sx:.1},{sy:.1}"));
    pts_deriv.push_str(&format!("{sx:.1},{sdy:.1}"));

    if let Some(cf) = cmp_fn {
      let cy = cf.eval(x, alpha);
      max_cmp_diff = max_cmp_diff.max((y - cy).abs());
      if step > 0 {
        pts_cmp.push(' ');
      }
      pts_cmp.push_str(&format!("{sx:.1},{:.1}", map_y(cy)));
    }

    if dy.abs() < 0.05 {
      if in_dead.is_none() {
        in_dead = Some(sx);
      }
    } else if let Some(start_sx) = in_dead.take() {
      dead_spans.push((start_sx, sx));
    }
  }
  if let Some(start_sx) = in_dead {
    dead_spans.push((start_sx, map_x(5.0)));
  }

  let y0 = p_fn.eval(x0, alpha);
  let dy0 = p_fn.deriv(x0, alpha);
  let ten_layer_gain = dy0.abs().powi(10);
  let px0 = map_x(x0);
  let py0 = map_y(y0);
  // 局部切线 x0 ± 0.9
  let tx1 = (x0 - 0.9).max(-5.0);
  let tx2 = (x0 + 0.9).min(5.0);
  let ty1 = y0 + dy0 * (tx1 - x0);
  let ty2 = y0 + dy0 * (tx2 - x0);
  let zero_x = map_x(0.0);
  let zero_y = map_y(0.0);

  rsx! {
    div { class: "space-y-4",
      // 1. 主激活函数选择器
      div {
        div { class: "text-[11px] font-semibold text-slate-500 dark:text-slate-400 mb-1.5",
          "1. 选择主激活函数 f(x)（覆盖饱和族 / ReLU 族 / 自门控 / 门控双线性 / 音频周期激活）"
        }
        div { class: "flex flex-wrap gap-1.5",
          for item in ActivationFn::ALL.iter().copied() {
            button {
              key: "{item.label()}",
              r#type: "button",
              class: if p_fn == item {
                "px-2.5 py-1 rounded-md text-xs font-semibold bg-indigo-600 text-white cursor-pointer"
              } else {
                "px-2.5 py-1 rounded-md text-xs font-medium bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700 cursor-pointer"
              },
              onclick: move |_| primary.set(item),
              "{item.label()}"
            }
          }
        }
      }

      // 2. 对比叠加 + 滑块控制栏
      div { class: "grid grid-cols-1 md:grid-cols-3 gap-3 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/70 dark:bg-slate-950/60 p-3",
        div {
          div { class: "flex items-center justify-between text-xs font-mono mb-1",
            span { class: "text-slate-600 dark:text-slate-300 font-semibold", "探针位置 x₀ = {x0:+.2}" }
            button {
              r#type: "button",
              class: "text-[11px] text-indigo-600 dark:text-indigo-400 underline cursor-pointer",
              onclick: move |_| show_deriv.set(!show_deriv()),
              if show_deriv() { "隐藏导数 f'(x)" } else { "显示导数 f'(x)" }
            }
          }
          input {
            r#type: "range",
            min: "-4.5",
            max: "4.5",
            step: "0.1",
            value: "{x0}",
            class: "w-full accent-indigo-600 cursor-pointer",
            oninput: move |e| {
              if let Ok(v) = e.value().parse::<f64>() {
                probe_x.set(v);
              }
            },
          }
        }

        div {
          div { class: "text-xs font-mono text-slate-600 dark:text-slate-300 font-semibold mb-1",
            "形态/频率参数 α·β·a = {alpha:.2}"
          }
          input {
            r#type: "range",
            min: "0.2",
            max: "3.2",
            step: "0.1",
            value: "{alpha}",
            class: "w-full accent-amber-500 cursor-pointer",
            oninput: move |e| {
              if let Ok(v) = e.value().parse::<f64>() {
                shape_param.set(v);
              }
            },
          }
        }

        div {
          div { class: "text-[11px] font-semibold text-slate-600 dark:text-slate-300 mb-1",
            "同屏对比基准 g(x)"
          }
          div { class: "flex flex-wrap gap-1",
            for (opt, lbl) in [
              (None, "无"),
              (Some(ActivationFn::ReLU), "ReLU"),
              (Some(ActivationFn::Sigmoid), "Sigmoid"),
              (Some(ActivationFn::GELUTanh), "GELU-Tanh"),
              (Some(ActivationFn::SiLU), "SiLU"),
            ] {
              button {
                key: "{lbl}",
                r#type: "button",
                class: if cmp_fn == opt {
                  "px-2 py-0.5 rounded text-[11px] font-semibold bg-emerald-600 text-white cursor-pointer"
                } else {
                  "px-2 py-0.5 rounded text-[11px] bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-300 cursor-pointer"
                },
                onclick: move |_| compare.set(opt),
                "{lbl}"
              }
            }
          }
        }
      }

      // 3. 响应式双曲线 + 死区着色 + 实时切线 SVG 画布
      div { class: "w-full overflow-x-auto rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950 p-2",
        svg {
          view_box: "0 0 680 270",
          class: "w-full min-w-[500px] h-auto font-mono select-none",
          role: "img",
          "aria-label": "Interactive activation function and derivative curve plot",

          // 死区 / 饱和带（|f'(x)| < 0.05）淡红色警示背景
          for (idx, (sx1, sx2)) in dead_spans.iter().copied().enumerate() {
            if sx2 > sx1 + 1.5 {
              rect {
                key: "dead-{idx}",
                x: "{sx1:.1}",
                y: "25",
                width: "{(sx2 - sx1):.1}",
                height: "220",
                class: "fill-rose-500/10 dark:fill-rose-500/15",
              }
            }
          }

          // 背景网格线
          for gx in [-4.0, -2.0, 2.0, 4.0] {
            line {
              key: "gx-{gx}",
              x1: "{map_x(gx):.1}",
              y1: "25",
              x2: "{map_x(gx):.1}",
              y2: "245",
              class: "stroke-slate-200/80 dark:stroke-slate-800",
              stroke_dasharray: "3 3",
              stroke_width: "1",
            }
          }
          for gy in [-1.0, 1.0, 2.0, 3.0, 4.0] {
            line {
              key: "gy-{gy}",
              x1: "55",
              y1: "{map_y(gy):.1}",
              x2: "655",
              y2: "{map_y(gy):.1}",
              class: "stroke-slate-200/80 dark:stroke-slate-800",
              stroke_dasharray: "3 3",
              stroke_width: "1",
            }
          }

          // 主坐标轴 x=0, y=0
          line { x1: "55", y1: "{zero_y:.1}", x2: "655", y2: "{zero_y:.1}", class: "stroke-slate-400 dark:stroke-slate-600", stroke_width: "1.4" }
          line { x1: "{zero_x:.1}", y1: "25", x2: "{zero_x:.1}", y2: "245", class: "stroke-slate-400 dark:stroke-slate-600", stroke_width: "1.4" }
          text { x: "648", y: "{zero_y - 6.0:.1}", text_anchor: "end", font_size: "10", class: "fill-slate-500", "x" }
          text { x: "{zero_x + 6.0:.1}", y: "36", font_size: "10", class: "fill-slate-500", "f(x)" }

          // 对比激活曲线 g(x)
          if cmp_fn.is_some() {
            polyline {
              points: "{pts_cmp}",
              fill: "none",
              class: "stroke-emerald-500/85",
              stroke_width: "2",
              stroke_dasharray: "6 3",
            }
          }

          // 一阶导数曲线 f'(x)
          if show_deriv() {
            polyline {
              points: "{pts_deriv}",
              fill: "none",
              class: "stroke-amber-500",
              stroke_width: "1.8",
              stroke_dasharray: "3 3",
            }
          }

          // 主激活函数曲线 f(x)
          polyline {
            points: "{pts_primary}",
            fill: "none",
            class: "stroke-indigo-600 dark:stroke-indigo-400",
            stroke_width: "2.8",
          }

          // 探针局部切线与交点
          line {
            x1: "{map_x(tx1):.1}",
            y1: "{map_y(ty1):.1}",
            x2: "{map_x(tx2):.1}",
            y2: "{map_y(ty2):.1}",
            class: "stroke-rose-500",
            stroke_width: "2",
          }
          line {
            x1: "{px0:.1}",
            y1: "{zero_y:.1}",
            x2: "{px0:.1}",
            y2: "{py0:.1}",
            class: "stroke-indigo-400/70",
            stroke_dasharray: "2 2",
            stroke_width: "1.2",
          }
          circle {
            cx: "{px0:.1}",
            cy: "{py0:.1}",
            r: "5.5",
            class: "fill-rose-500 stroke-white dark:stroke-slate-950",
            stroke_width: "2",
          }

          // 图例说明
          text { x: "66", y: "40", font_size: "10.5", class: "fill-indigo-600 dark:fill-indigo-400 font-bold",
            "— {p_fn.label()}"
          }
          if show_deriv() {
            text { x: "66", y: "56", font_size: "10", class: "fill-amber-600 dark:fill-amber-400 font-semibold",
              "-- 导数 f'(x) · 粉底区间为 |f'(x)| < 0.05 死区/饱和区"
            }
          }
        }
      }

      // 4. 实时数值仪表盘与性质解读
      div { class: "grid grid-cols-2 sm:grid-cols-4 gap-2.5 font-mono text-xs",
        div { class: "rounded-lg border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-900 p-2.5",
          div { class: "text-[10px] text-slate-400", "前向激活值 f(x₀)" }
          div { class: "mt-0.5 font-bold text-slate-900 dark:text-white tabular-nums", "{y0:+.5}" }
        }
        div { class: "rounded-lg border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-900 p-2.5",
          div { class: "text-[10px] text-slate-400", "局部反向雅可比 f'(x₀)" }
          div {
            class: if dy0.abs() < 0.05 {
              "mt-0.5 font-bold text-rose-600 dark:text-rose-400 tabular-nums"
            } else {
              "mt-0.5 font-bold text-amber-600 dark:text-amber-400 tabular-nums"
            },
            "{dy0:+.5}"
          }
        }
        div { class: "rounded-lg border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-900 p-2.5",
          div { class: "text-[10px] text-slate-400", "10 层连乘存活率 |f'|¹⁰" }
          div {
            class: if ten_layer_gain < 1e-4 {
              "mt-0.5 font-bold text-rose-600 dark:text-rose-400 tabular-nums"
            } else {
              "mt-0.5 font-bold text-indigo-600 dark:text-indigo-400 tabular-nums"
            },
            "{ten_layer_gain:.2e}"
          }
        }
        div { class: "rounded-lg border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-900 p-2.5",
          div { class: "text-[10px] text-slate-400",
            if let Some(cf) = cmp_fn { "与 {cf.label()} 最大残差" } else { "区间死区段数" }
          }
          div { class: "mt-0.5 font-bold text-emerald-600 dark:text-emerald-400 tabular-nums",
            if cmp_fn.is_some() { "{max_cmp_diff:.2e}" } else { "{dead_spans.len()} 段" }
          }
        }
      }

      div { class: "rounded-lg bg-indigo-50/70 dark:bg-indigo-950/30 border border-indigo-200/70 dark:border-indigo-900/60 px-3.5 py-2 text-xs text-slate-700 dark:text-slate-300",
        span { class: "font-bold text-indigo-700 dark:text-indigo-300 mr-1.5", "物理机理：" }
        "{p_fn.formula_note()}"
      }
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 内核 2：GridSlidingKernelLab（标准/深度可分离/空洞/转置棋盘伪影/池化梯度/RoIAlign/PixelUnshuffle）
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GridKernelMode {
  StandardConv,
  DepthwiseSeparable,
  TransposedCheckerboard,
  PoolGradRouting,
  RoIAlignSubpixel,
}

impl GridKernelMode {
  fn parse_mode(raw: &str) -> Self {
    match raw.trim().to_ascii_lowercase().as_str() {
      "standard" | "conv" | "conv2d" | "dilated" | "atrous" => Self::StandardConv,
      "depthwise_separable" | "depthwise" | "separable" | "grouped" => Self::DepthwiseSeparable,
      "transposed" | "deconv" | "checkerboard" | "pixel_shuffle" => Self::TransposedCheckerboard,
      "pool" | "pooling" | "maxpool" | "avgpool" | "softpool" => Self::PoolGradRouting,
      "roialign" | "roi_align" | "roi" => Self::RoIAlignSubpixel,
      _ => Self::DepthwiseSeparable,
    }
  }

  fn label(self) -> &'static str {
    match self {
      Self::StandardConv => "标准 / 空洞卷积 (Conv & Dilated)",
      Self::DepthwiseSeparable => "深度可分离卷积 (DW + 1×1 PW)",
      Self::TransposedCheckerboard => "转置卷积棋盘伪影 (Deconv)",
      Self::PoolGradRouting => "池化与反向梯度路由 (Max/Avg/Soft)",
      Self::RoIAlignSubpixel => "RoI Pooling vs RoIAlign",
    }
  }
}

const SAMPLE_6X6: [[f64; 6]; 6] = [
  [0.0, 0.2, 1.0, 1.0, 0.2, 0.0],
  [0.1, 0.9, 1.5, 1.2, 0.3, 0.0],
  [0.2, 1.4, 2.0, 0.8, 0.1, 0.0],
  [0.0, 0.8, 1.6, 1.9, 0.7, 0.1],
  [0.0, 0.2, 0.7, 1.4, 1.3, 0.4],
  [0.0, 0.0, 0.1, 0.5, 0.8, 0.2],
];

const KERNEL_EDGE_3X3: [[f64; 3]; 3] = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]];

#[component]
fn GridSlidingKernelLab(initial_mode: String) -> Element {
  let mut mode = use_signal(move || GridKernelMode::parse_mode(&initial_mode));
  let mut stride = use_signal(|| 1_usize);
  let mut padding = use_signal(|| 0_usize);
  let mut dilation = use_signal(|| 1_usize);
  let mut step_idx = use_signal(|| 0_usize);
  let mut playing = use_signal(|| false);
  // 深度可分离 / 转置卷积 / 池化 / RoIAlign 专属参数
  let mut c_in = use_signal(|| 64_u64);
  let mut c_out = use_signal(|| 128_u64);
  let mut deconv_k = use_signal(|| 3_usize);
  let mut pool_kind = use_signal(|| 0_u8); // 0=Max, 1=Avg, 2=SoftPool
  let mut roi_offset = use_signal(|| 0.35_f64);

  // 客户端自动步进循环（SSR 不触发 use_effect）
  use_effect(move || {
    if !playing() {
      return;
    }
    spawn(async move {
      while playing() {
        widgets::browser::sleep_ms(620).await;
        if !playing() {
          break;
        }
        step_idx.with_mut(|v| *v = v.wrapping_add(1));
      }
    });
  });

  let cur_mode = mode();
  let s = stride().max(1);
  let p = padding();
  let d = dilation().max(1);

  // 计算标准/空洞卷积输出网格尺寸（基于 6×6 输入与 3×3 卷积核）
  let padded_n = 6 + 2 * p;
  let eff_k = d * 2 + 1;
  let out_n = if padded_n >= eff_k { (padded_n - eff_k) / s + 1 } else { 1 };
  let total_positions = (out_n * out_n).max(1);
  let active_pos = step_idx() % total_positions;
  let out_r = active_pos / out_n;
  let out_c = active_pos % out_n;

  // 读取带零填充的输入特征值
  let get_padded = |pr: usize, pc: usize| -> (f64, bool) {
    if pr < p || pc < p || pr >= p + 6 || pc >= p + 6 {
      (0.0, true)
    } else {
      (SAMPLE_6X6[pr - p][pc - p], false)
    }
  };

  rsx! {
    div { class: "space-y-4",
      // 模式切换栏
      div { class: "flex flex-wrap gap-1.5",
        for m in [
          GridKernelMode::StandardConv,
          GridKernelMode::DepthwiseSeparable,
          GridKernelMode::TransposedCheckerboard,
          GridKernelMode::PoolGradRouting,
          GridKernelMode::RoIAlignSubpixel,
        ] {
          button {
            key: "{m.label()}",
            r#type: "button",
            class: if cur_mode == m {
              "px-2.5 py-1.5 rounded-lg text-xs font-semibold bg-indigo-600 text-white cursor-pointer"
            } else {
              "px-2.5 py-1.5 rounded-lg text-xs font-medium bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700 cursor-pointer"
            },
            onclick: move |_| {
              playing.set(false);
              mode.set(m);
              step_idx.set(0);
            },
            "{m.label()}"
          }
        }
      }

      match cur_mode {
        GridKernelMode::StandardConv => {
          // 收集当前输出格 (out_r, out_c) 命中的 9 个采样坐标及乘累加值
          let mut hits: HashMap<(usize, usize), f64> = HashMap::new();
          let mut mac_sum = 0.0_f64;
          for (kr, k_row) in KERNEL_EDGE_3X3.iter().enumerate() {
            for (kc, &w) in k_row.iter().enumerate() {
              let pr = out_r * s + kr * d;
              let pc = out_c * s + kc * d;
              let (val, _) = get_padded(pr, pc);
              hits.insert((pr, pc), w);
              mac_sum += val * w;
            }
          }

          rsx! {
            div { class: "space-y-3",
              div { class: "flex flex-wrap items-center justify-between gap-2 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/80 dark:bg-slate-950/60 p-3 text-xs",
                div { class: "flex flex-wrap items-center gap-3",
                  div { class: "flex items-center gap-1",
                    span { class: "text-slate-500", "Stride s:" }
                    for sv in [1_usize, 2] {
                      button {
                        key: "s-{sv}",
                        r#type: "button",
                        class: if s == sv { "px-2 py-0.5 rounded bg-indigo-600 text-white font-mono" } else { "px-2 py-0.5 rounded bg-white dark:bg-slate-800 font-mono" },
                        onclick: move |_| { stride.set(sv); step_idx.set(0); },
                        "{sv}"
                      }
                    }
                  }
                  div { class: "flex items-center gap-1",
                    span { class: "text-slate-500", "Padding p:" }
                    for pv in [0_usize, 1] {
                      button {
                        key: "p-{pv}",
                        r#type: "button",
                        class: if p == pv { "px-2 py-0.5 rounded bg-indigo-600 text-white font-mono" } else { "px-2 py-0.5 rounded bg-white dark:bg-slate-800 font-mono" },
                        onclick: move |_| { padding.set(pv); step_idx.set(0); },
                        "{pv}"
                      }
                    }
                  }
                  div { class: "flex items-center gap-1",
                    span { class: "text-slate-500", "Dilation d:" }
                    for dv in [1_usize, 2] {
                      button {
                        key: "d-{dv}",
                        r#type: "button",
                        class: if d == dv { "px-2 py-0.5 rounded bg-amber-600 text-white font-mono" } else { "px-2 py-0.5 rounded bg-white dark:bg-slate-800 font-mono" },
                        onclick: move |_| { dilation.set(dv); step_idx.set(0); },
                        "{dv}"
                      }
                    }
                  }
                }

                // 步进播放控制条
                div { class: "flex items-center gap-1.5 font-mono",
                  button {
                    r#type: "button",
                    class: "px-2.5 py-1 rounded bg-slate-200 dark:bg-slate-800 hover:bg-slate-300 cursor-pointer",
                    onclick: move |_| {
                      playing.set(false);
                      step_idx.set(active_pos.checked_sub(1).unwrap_or(total_positions - 1));
                    },
                    "◀"
                  }
                  button {
                    r#type: "button",
                    class: "px-3 py-1 rounded bg-indigo-600 text-white font-semibold cursor-pointer",
                    onclick: move |_| playing.set(!playing()),
                    if playing() { "⏸ 暂停" } else { "▶ 自动滑窗" }
                  }
                  button {
                    r#type: "button",
                    class: "px-2.5 py-1 rounded bg-slate-200 dark:bg-slate-800 hover:bg-slate-300 cursor-pointer",
                    onclick: move |_| {
                      playing.set(false);
                      step_idx.set((active_pos + 1) % total_positions);
                    },
                    "▶"
                  }
                }
              }

              // 输入网格 → 3×3 Sobel 卷积核 → 输出特征网格（点击任意输出格可锁定观察）
              div { class: "w-full overflow-x-auto rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950 p-3",
                svg {
                  view_box: "0 0 680 250",
                  class: "w-full min-w-[520px] h-auto font-mono select-none",
                  role: "img",
                  "aria-label": "2D sliding convolution receptive field simulator",

                  text { x: "115", y: "20", text_anchor: "middle", font_size: "11", class: "fill-slate-700 dark:fill-slate-200 font-bold",
                    "Input [{padded_n}×{padded_n}] (含 pad={p})"
                  }
                  for pr in 0..padded_n {
                    for pc in 0..padded_n {
                      {
                        let cell_sz = if padded_n >= 8 { 24.0 } else { 28.0 };
                        let x = 20.0 + (pc as f64) * cell_sz;
                        let y = 30.0 + (pr as f64) * cell_sz;
                        let (val, is_pad) = get_padded(pr, pc);
                        let hit_w = hits.get(&(pr, pc)).copied();
                        let rect_cls = if hit_w.is_some() {
                          "fill-indigo-500/25 dark:fill-indigo-500/40 stroke-indigo-600 dark:stroke-indigo-400"
                        } else if is_pad {
                          "fill-slate-100/70 dark:fill-slate-900/50 stroke-slate-300 dark:stroke-slate-800"
                        } else {
                          "fill-slate-50 dark:fill-slate-900 stroke-slate-300 dark:stroke-slate-700"
                        };
                        rsx! {
                          g { key: "in-{pr}-{pc}",
                            rect {
                              x: "{x:.1}",
                              y: "{y:.1}",
                              width: "{cell_sz - 2.0:.1}",
                              height: "{cell_sz - 2.0:.1}",
                              rx: "3",
                              class: "{rect_cls}",
                              stroke_width: if hit_w.is_some() { "2" } else { "1" },
                            }
                            text {
                              x: "{x + (cell_sz - 2.0) * 0.5:.1}",
                              y: "{y + (cell_sz - 2.0) * 0.62:.1}",
                              text_anchor: "middle",
                              font_size: "9",
                              class: if is_pad { "fill-slate-400" } else { "fill-slate-800 dark:fill-slate-100" },
                              "{val:.1}"
                            }
                          }
                        }
                      }
                    }
                  }

                  // 中间 3×3 卷积核展示
                  text { x: "335", y: "52", text_anchor: "middle", font_size: "11", class: "fill-amber-600 dark:fill-amber-400 font-bold",
                    "Kernel [3×3] (eff {eff_k}×{eff_k})"
                  }
                  for kr in 0..3 {
                    for kc in 0..3 {
                      {
                        let kx = 290.0 + (kc as f64) * 30.0;
                        let ky = 64.0 + (kr as f64) * 30.0;
                        let w = KERNEL_EDGE_3X3[kr][kc];
                        rsx! {
                          g { key: "k-{kr}-{kc}",
                            rect { x: "{kx}", y: "{ky}", width: "28", height: "28", rx: "4", class: "fill-amber-50 dark:fill-amber-950/60 stroke-amber-500", stroke_width: "1.5" }
                            text { x: "{kx + 14.0}", y: "{ky + 18.0}", text_anchor: "middle", font_size: "10", class: "fill-amber-800 dark:fill-amber-200 font-bold", "{w:+.0}" }
                          }
                        }
                      }
                    }
                  }
                  text { x: "335", y: "178", text_anchor: "middle", font_size: "10", class: "fill-slate-500",
                    "Σ(x · w) = {mac_sum:+.2}"
                  }

                  // 右侧输出网格（支持点击切换观察位置）
                  text { x: "545", y: "20", text_anchor: "middle", font_size: "11", class: "fill-emerald-700 dark:fill-emerald-400 font-bold",
                    "Output [{out_n}×{out_n}] (点击格子定位)"
                  }
                  for r in 0..out_n {
                    for c in 0..out_n {
                      {
                        let mut cell_sum = 0.0_f64;
                        for (kr, k_row) in KERNEL_EDGE_3X3.iter().enumerate() {
                          for (kc, &w) in k_row.iter().enumerate() {
                            let (v, _) = get_padded(r * s + kr * d, c * s + kc * d);
                            cell_sum += v * w;
                          }
                        }
                        let ox = 465.0 + (c as f64) * 32.0;
                        let oy = 36.0 + (r as f64) * 32.0;
                        let is_sel = r == out_r && c == out_c;
                        let idx_target = r * out_n + c;
                        rsx! {
                          g {
                            key: "out-{r}-{c}",
                            class: "cursor-pointer",
                            onclick: move |_| {
                              playing.set(false);
                              step_idx.set(idx_target);
                            },
                            rect {
                              x: "{ox}",
                              y: "{oy}",
                              width: "30",
                              height: "30",
                              rx: "5",
                              class: if is_sel {
                                "fill-emerald-500 stroke-emerald-700 dark:stroke-emerald-300"
                              } else {
                                "fill-emerald-50 dark:fill-emerald-950/40 stroke-emerald-500/60"
                              },
                              stroke_width: if is_sel { "2.2" } else { "1.2" },
                            }
                            text {
                              x: "{ox + 15.0}",
                              y: "{oy + 19.0}",
                              text_anchor: "middle",
                              font_size: "9.5",
                              class: if is_sel { "fill-white font-bold" } else { "fill-slate-800 dark:fill-slate-200" },
                              "{cell_sum:+.1}"
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }

              div { class: "rounded-lg bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 px-3.5 py-2 text-xs font-mono text-slate-700 dark:text-slate-300",
                "输出尺寸公式：⌊(6 + 2×{p} - {d}×(3-1) - 1)/{s} + 1⌋ = {out_n}×{out_n} · "
                if d > 1 {
                  span { class: "text-amber-600 dark:text-amber-400 font-semibold",
                    "★ Dilation={d} 时有效感受野扩至 {eff_k}×{eff_k}（零新增参数），但相邻采样点之间出现空洞盲区（Gridding Effect）！"
                  }
                } else {
                  "点击右侧任意输出单元格可逆向点亮其精确输入感受野。"
                }
              }
            }
          }
        }

        GridKernelMode::DepthwiseSeparable => {
          let cin = c_in();
          let cout = c_out();
          let k = 3_u64;
          let std_params = cin * cout * k * k;
          let dw_params = cin * k * k;
          let pw_params = cin * cout;
          let sep_params = dw_params + pw_params;
          let speedup = (std_params as f64) / (sep_params.max(1) as f64);
          let bar_pct = ((sep_params as f64 / std_params.max(1) as f64) * 100.0).clamp(3.0, 100.0);

          rsx! {
            div { class: "space-y-4",
              div { class: "grid grid-cols-1 sm:grid-cols-2 gap-3 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/70 dark:bg-slate-950/60 p-3 text-xs font-mono",
                div {
                  div { class: "mb-1 font-semibold", "输入通道数 C_in = {cin}" }
                  input {
                    r#type: "range", min: "16", max: "512", step: "16", value: "{cin}",
                    class: "w-full accent-indigo-600 cursor-pointer",
                    oninput: move |e| { if let Ok(v) = e.value().parse::<u64>() { c_in.set(v); } },
                  }
                }
                div {
                  div { class: "mb-1 font-semibold", "输出通道数 C_out = {cout}" }
                  input {
                    r#type: "range", min: "16", max: "512", step: "16", value: "{cout}",
                    class: "w-full accent-emerald-600 cursor-pointer",
                    oninput: move |e| { if let Ok(v) = e.value().parse::<u64>() { c_out.set(v); } },
                  }
                }
              }

              div { class: "w-full overflow-x-auto rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950 p-3",
                svg {
                  view_box: "0 0 680 195",
                  class: "w-full min-w-[500px] h-auto font-mono",
                  role: "img",
                  "aria-label": "Standard convolution vs Depthwise Separable decomposition",

                  // 上排：标准耦合卷积
                  rect { x: "20", y: "18", width: "640", height: "64", rx: "10", class: "fill-rose-50/70 dark:fill-rose-950/30 stroke-rose-400", stroke_width: "1.5" }
                  text { x: "35", y: "42", font_size: "11", class: "fill-rose-700 dark:fill-rose-300 font-bold",
                    "标准 3×3 卷积 (空间滤波与跨通道耦合在单步完成)"
                  }
                  text { x: "35", y: "64", font_size: "10.5", class: "fill-slate-700 dark:fill-slate-300",
                    "Params = C_in({cin}) × C_out({cout}) × 3×3 = {std_params} 参数 (100% 基准)"
                  }

                  // 下排：Depthwise + Pointwise 解耦两步
                  rect { x: "20", y: "102", width: "295", height: "76", rx: "10", class: "fill-sky-50 dark:fill-sky-950/50 stroke-sky-500", stroke_width: "1.6" }
                  text { x: "167", y: "126", text_anchor: "middle", font_size: "11", class: "fill-sky-700 dark:fill-sky-300 font-bold",
                    "① Depthwise 3×3 (groups = {cin})"
                  }
                  text { x: "167", y: "146", text_anchor: "middle", font_size: "10", class: "fill-slate-600 dark:fill-slate-300",
                    "每通道独立 3×3 空间滤波：{dw_params} 参数"
                  }
                  text { x: "167", y: "164", text_anchor: "middle", font_size: "9.5", class: "fill-sky-600 dark:fill-sky-400",
                    "只看空间邻域，不跨通道混合"
                  }

                  line { x1: "315", y1: "140", x2: "355", y2: "140", class: "stroke-indigo-500", stroke_width: "2" }
                  text { x: "335", y: "133", text_anchor: "middle", font_size: "11", class: "fill-indigo-500 font-bold", "→" }

                  rect { x: "355", y: "102", width: "305", height: "76", rx: "10", class: "fill-emerald-50 dark:fill-emerald-950/50 stroke-emerald-500", stroke_width: "1.6" }
                  text { x: "507", y: "126", text_anchor: "middle", font_size: "11", class: "fill-emerald-700 dark:fill-emerald-300 font-bold",
                    "② Pointwise 1×1 Conv [{cin} → {cout}]"
                  }
                  text { x: "507", y: "146", text_anchor: "middle", font_size: "10", class: "fill-slate-600 dark:fill-slate-300",
                    "逐像素跨通道线性重组：{pw_params} 参数"
                  }
                  text { x: "507", y: "164", text_anchor: "middle", font_size: "9.5", class: "fill-emerald-700 dark:fill-emerald-400 font-bold",
                    "总计 {sep_params} 参数 · 算力直降 {speedup:.2}×！"
                  }
                }
              }

              div { class: "rounded-lg border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-900 p-3 font-mono text-xs",
                div { class: "flex justify-between mb-1.5",
                  span { "Depthwise Separable 参数/MACs 仅占标准卷积的 {bar_pct:.1}%" }
                  span { class: "font-bold text-emerald-600 dark:text-emerald-400", "压缩比 {speedup:.2}× (理论极限 k²=9×)" }
                }
                div { class: "h-2.5 w-full rounded-full bg-rose-200 dark:bg-rose-950 overflow-hidden",
                  div { class: "h-full bg-emerald-500 transition-all", style: "width: {bar_pct:.1}%" }
                }
              }
            }
          }
        }

        GridKernelMode::TransposedCheckerboard => {
          // 4×4 全 1 输入经 stride=2、kernel=k (3 或 4) 的转置卷积每个输出像素被叠加的次数热力图
          let k = deconv_k();
          let s_up = 2_usize;
          let in_sz = 4_usize;
          let out_sz = (in_sz - 1) * s_up + k;
          let mut counts = vec![vec![0_u32; out_sz]; out_sz];
          for ir in 0..in_sz {
            for ic in 0..in_sz {
              for kr in 0..k {
                for kc in 0..k {
                  counts[ir * s_up + kr][ic * s_up + kc] += 1;
                }
              }
            }
          }
          // 检验内部中心区域是否均匀
          let center_a = counts[2][2];
          let center_b = counts[2][3];
          let has_checker = center_a != center_b;

          rsx! {
            div { class: "space-y-3",
              div { class: "flex flex-wrap items-center justify-between gap-2 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/80 dark:bg-slate-950/60 p-3 text-xs",
                span { class: "font-semibold text-slate-700 dark:text-slate-200",
                  "固定上采样步幅 Stride = 2，切换转置卷积核大小 k 观察像素重叠频次："
                }
                div { class: "flex gap-2 font-mono",
                  button {
                    r#type: "button",
                    class: if k == 3 { "px-3 py-1 rounded-md bg-rose-600 text-white font-bold cursor-pointer" } else { "px-3 py-1 rounded-md bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 cursor-pointer" },
                    onclick: move |_| deconv_k.set(3),
                    "k = 3 (不可被 s=2 整除 → 棋盘伪影!)"
                  }
                  button {
                    r#type: "button",
                    class: if k == 4 { "px-3 py-1 rounded-md bg-emerald-600 text-white font-bold cursor-pointer" } else { "px-3 py-1 rounded-md bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 cursor-pointer" },
                    onclick: move |_| deconv_k.set(4),
                    "k = 4 (可被 s=2 整除 → 内部均匀!)"
                  }
                }
              }

              div { class: "w-full overflow-x-auto rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950 p-3",
                svg {
                  view_box: "0 0 680 235",
                  class: "w-full min-w-[480px] h-auto font-mono",
                  role: "img",
                  "aria-label": "Transposed convolution kernel overlap count heatmap",

                  text { x: "340", y: "20", text_anchor: "middle", font_size: "11", class: "fill-slate-800 dark:fill-slate-200 font-bold",
                    "输出像素累加覆盖次数热力图 [{out_sz}×{out_sz}]（全 1 输入 × 全 1 核）"
                  }
                  for r in 0..out_sz {
                    for c in 0..out_sz {
                      {
                        let cnt = counts[r][c];
                        let ox = 205.0 + (c as f64) * 28.0;
                        let oy = 32.0 + (r as f64) * 24.0;
                        let cls = match cnt {
                          1 => "fill-slate-100 dark:fill-slate-900 stroke-slate-300 dark:stroke-slate-700",
                          2 => "fill-sky-200/80 dark:fill-sky-900/70 stroke-sky-500",
                          4 => "fill-emerald-400/80 dark:fill-emerald-700 stroke-emerald-600",
                          _ => "fill-rose-500/85 stroke-rose-700",
                        };
                        rsx! {
                          g { key: "dc-{r}-{c}",
                            rect { x: "{ox}", y: "{oy}", width: "26", height: "22", rx: "3", class: "{cls}", stroke_width: "1" }
                            text { x: "{ox + 13.0}", y: "{oy + 15.0}", text_anchor: "middle", font_size: "10", class: "fill-slate-900 dark:fill-white font-bold", "{cnt}×" }
                          }
                        }
                      }
                    }
                  }
                }
              }

              div {
                class: if has_checker {
                  "rounded-lg border border-rose-300 dark:border-rose-900 bg-rose-50 dark:bg-rose-950/40 px-3.5 py-2 text-xs text-rose-800 dark:text-rose-200"
                } else {
                  "rounded-lg border border-emerald-300 dark:border-emerald-900 bg-emerald-50 dark:bg-emerald-950/40 px-3.5 py-2 text-xs text-emerald-800 dark:text-emerald-200"
                },
                if has_checker {
                  "⚠️ 根因诊断：k=3, s=2 时内部相邻像素交替被叠加 1×、2×、4× 次！即使输入与权重完全均匀，解码图像也会天然印上高频明暗网格（GAN / VAE 解码器常见棋盘格伪影）。"
                } else {
                  "✓ 消除棋盘格：当 k=4 是 s=2 的整数倍（或改用 Nearest Upsample + 3×3 Conv / PixelShuffle）时，内部所有像素均匀接收 4× 次核权重贡献，彻底消除周期性伪影。"
                }
              }
            }
          }
        }

        GridKernelMode::PoolGradRouting => {
          // 演示 2×2 窗口 [1.2, 3.4; 0.5, 2.1] 在上游梯度 dL/dy = 1.00 时的前向输出与反向梯度路由
          let patch = [[1.2_f64, 3.4], [0.5, 2.1]];
          let pk = pool_kind();
          let (fwd_val, grads, desc) = match pk {
            0 => (
              3.4_f64,
              [[0.0_f64, 1.0], [0.0, 0.0]],
              "MaxPool：仅最大值像素 (3.4) 独占 100% 上游梯度，其余 3 个位置梯度严格为 0（保留强边缘但梯度稀疏）。",
            ),
            1 => (
              (1.2 + 3.4 + 0.5 + 2.1) * 0.25,
              [[0.25, 0.25], [0.25, 0.25]],
              "AvgPool：不论输入幅值高低，4 个像素绝对均分 1/k² = 0.25 梯度（平滑背景但冲淡显著峰值）。",
            ),
            _ => {
              let m = 3.4_f64;
              let exps = [
                [(1.2 - m).exp(), (3.4 - m).exp()],
                [(0.5 - m).exp(), (2.1 - m).exp()],
              ];
              let sum_e = exps[0][0] + exps[0][1] + exps[1][0] + exps[1][1];
              let w = [
                [exps[0][0] / sum_e, exps[0][1] / sum_e],
                [exps[1][0] / sum_e, exps[1][1] / sum_e],
              ];
              let y = w[0][0] * 1.2 + w[0][1] * 3.4 + w[1][0] * 0.5 + w[1][1] * 2.1;
              (
                y,
                w,
                "SoftPool：按指数权重 w_i = e^(x_i)/Σe^(x_j) 加权汇聚，兼顾峰值凸显且所有输入像素均获得平滑非零梯度。",
              )
            }
          };

          rsx! {
            div { class: "space-y-3",
              div { class: "flex flex-wrap gap-2",
                for (id, lbl) in [(0_u8, "MaxPool 2×2"), (1, "AvgPool 2×2"), (2, "SoftPool 2×2 (指数软池化)")] {
                  button {
                    key: "{lbl}",
                    r#type: "button",
                    class: if pk == id { "px-3 py-1 rounded-md text-xs font-bold bg-indigo-600 text-white cursor-pointer" } else { "px-3 py-1 rounded-md text-xs bg-slate-100 dark:bg-slate-800 cursor-pointer" },
                    onclick: move |_| pool_kind.set(id),
                    "{lbl}"
                  }
                }
              }

              div { class: "grid grid-cols-1 sm:grid-cols-3 gap-3 font-mono text-xs items-center rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950 p-4",
                div {
                  div { class: "text-[11px] font-bold text-slate-500 mb-2", "前向输入窗口 X [2×2]" }
                  div { class: "grid grid-cols-2 gap-2 max-w-[160px]",
                    for r in 0..2 {
                      for c in 0..2 {
                        div {
                          key: "px-{r}-{c}",
                          class: "rounded-lg border border-slate-300 dark:border-slate-700 bg-slate-50 dark:bg-slate-900 p-2.5 text-center font-bold",
                          "{patch[r][c]:.2}"
                        }
                      }
                    }
                  }
                }
                div { class: "text-center space-y-1",
                  div { class: "text-[11px] text-slate-400", "前向汇聚输出 y" }
                  div { class: "text-2xl font-extrabold text-indigo-600 dark:text-indigo-400", "{fwd_val:.3}" }
                  div { class: "text-[11px] text-rose-500 pt-1", "← 反向传入 ∂L/∂y = 1.00" }
                }
                div {
                  div { class: "text-[11px] font-bold text-rose-600 dark:text-rose-400 mb-2", "反向输入梯度路由 ∂y/∂X" }
                  div { class: "grid grid-cols-2 gap-2 max-w-[160px]",
                    for r in 0..2 {
                      for c in 0..2 {
                        div {
                          key: "pg-{r}-{c}",
                          class: if grads[r][c] > 0.4 {
                            "rounded-lg border-2 border-rose-500 bg-rose-50 dark:bg-rose-950/60 p-2.5 text-center font-extrabold text-rose-700 dark:text-rose-300"
                          } else if grads[r][c] > 0.0 {
                            "rounded-lg border border-amber-400 bg-amber-50/70 dark:bg-amber-950/40 p-2.5 text-center font-semibold text-amber-800 dark:text-amber-200"
                          } else {
                            "rounded-lg border border-slate-200 dark:border-slate-800 bg-slate-100/60 dark:bg-slate-900/40 p-2.5 text-center text-slate-400"
                          },
                          "{grads[r][c]:.2}"
                        }
                      }
                    }
                  }
                }
              }

              div { class: "rounded-lg bg-indigo-50/70 dark:bg-indigo-950/30 border border-indigo-200/70 dark:border-indigo-900/60 px-3.5 py-2 text-xs text-slate-700 dark:text-slate-300",
                "{desc}"
              }
            }
          }
        }

        GridKernelMode::RoIAlignSubpixel => {
          // 4 邻域特征值：[1,1]=1.0, [1,2]=4.0, [2,1]=2.0, [2,2]=8.0
          let f00 = 1.0_f64;
          let f01 = 4.0_f64;
          let f10 = 2.0_f64;
          let f11 = 8.0_f64;
          let delta = roi_offset();
          // RoI Pooling 粗暴 floor 取整恒读 f00=1.0（直到跨越 1.0 发生阶跃跳变）
          let roi_pool_val = if delta < 0.999 { f00 } else { f11 };
          // RoIAlign 亚像素双线性插值（dx = delta, dy = 0.6 * delta）
          let dx = delta;
          let dy = (0.6 * delta).clamp(0.0, 1.0);
          let w00 = (1.0 - dx) * (1.0 - dy);
          let w01 = dx * (1.0 - dy);
          let w10 = (1.0 - dx) * dy;
          let w11 = dx * dy;
          let roi_align_val = w00 * f00 + w01 * f01 + w10 * f10 + w11 * f11;
          let quantization_err = (roi_align_val - roi_pool_val).abs();

          rsx! {
            div { class: "space-y-3 font-mono text-xs",
              div { class: "rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/80 dark:bg-slate-950/60 p-3",
                div { class: "mb-1 font-semibold",
                  "拖动候选框浮点亚像素偏移量 (Δx = {dx:.2}, Δy = {dy:.2})："
                }
                input {
                  r#type: "range", min: "0.0", max: "0.95", step: "0.05", value: "{delta}",
                  class: "w-full accent-indigo-600 cursor-pointer",
                  oninput: move |e| { if let Ok(v) = e.value().parse::<f64>() { roi_offset.set(v); } },
                }
              }

              div { class: "grid grid-cols-1 sm:grid-cols-3 gap-3",
                div { class: "rounded-xl border border-rose-200 dark:border-rose-900/70 bg-rose-50/50 dark:bg-rose-950/30 p-3",
                  div { class: "text-[11px] text-rose-700 dark:text-rose-300 font-bold", "RoI Pooling (floor 硬取整)" }
                  div { class: "mt-1 text-xl font-extrabold text-rose-600 dark:text-rose-400", "{roi_pool_val:.3}" }
                  div { class: "mt-1 text-[11px] text-slate-500", "框移动 0.95 格输出仍卡死不变，坐标梯度为 0" }
                }
                div { class: "rounded-xl border border-emerald-200 dark:border-emerald-900/70 bg-emerald-50/50 dark:bg-emerald-950/30 p-3",
                  div { class: "text-[11px] text-emerald-700 dark:text-emerald-300 font-bold", "RoIAlign (四邻域双线性插值)" }
                  div { class: "mt-1 text-xl font-extrabold text-emerald-600 dark:text-emerald-400", "{roi_align_val:.3}" }
                  div { class: "mt-1 text-[11px] text-slate-500", "权重: [{w00:.2}, {w01:.2}, {w10:.2}, {w11:.2}]" }
                }
                div { class: "rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-900 p-3",
                  div { class: "text-[11px] text-slate-500 font-bold", "取整特征错位误差" }
                  div { class: "mt-1 text-xl font-extrabold text-amber-600 dark:text-amber-400", "{quantization_err:.3}" }
                  div { class: "mt-1 text-[11px] text-slate-500", "在 stride=32 特征图上 0.8 格偏移 = 原图 25.6px 错位！" }
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
// 内核 3：NormAndPrecisionLab（[N, C, S] 归一化切片、FP16 平方溢出、AdaLN-Zero）
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NormLabMode {
  BatchNorm,
  LayerNorm,
  InstanceNorm,
  GroupNorm,
  RmsNorm,
  AdaLnZero,
}

impl NormLabMode {
  fn parse_mode(raw: &str) -> Self {
    match raw.trim().to_ascii_lowercase().as_str() {
      "batchnorm" | "bn" => Self::BatchNorm,
      "layernorm" | "ln" => Self::LayerNorm,
      "instancenorm" | "in" | "adain" => Self::InstanceNorm,
      "groupnorm" | "gn" => Self::GroupNorm,
      "adaln" | "adaln_zero" | "adaln-zero" | "film" => Self::AdaLnZero,
      _ => Self::RmsNorm,
    }
  }

  fn label(self) -> &'static str {
    match self {
      Self::BatchNorm => "BatchNorm [跨 N,S]",
      Self::LayerNorm => "LayerNorm [跨 C]",
      Self::InstanceNorm => "InstanceNorm [跨 S]",
      Self::GroupNorm => "GroupNorm (G=2)",
      Self::RmsNorm => "RMSNorm (无去均值)",
      Self::AdaLnZero => "AdaLN-Zero (DiT)",
    }
  }

  /// 判断张量单元格 (n, c, s) 是否与探针焦点 (0, 0, 0) 处于同一个统计归约集合。
  fn in_same_group(self, n: usize, c: usize, s: usize) -> bool {
    match self {
      Self::BatchNorm => c == 0,
      Self::LayerNorm | Self::RmsNorm | Self::AdaLnZero => n == 0 && s == 0,
      Self::InstanceNorm => n == 0 && c == 0,
      Self::GroupNorm => n == 0 && (c / 2 == 0),
    }
  }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FloatFmt {
  Fp32,
  Fp16,
  Bf16,
}

impl FloatFmt {
  /// 模拟半精度数值量化与动态范围上溢（FP16 最大有限值 65504.0，BF16 7-bit 尾数步长约 1/128）。
  fn quantize(self, val: f64) -> f64 {
    match self {
      Self::Fp32 => val,
      Self::Fp16 => {
        if val.abs() > 65_504.0 {
          f64::INFINITY * val.signum()
        } else if val.abs() < 6.0e-8 {
          0.0
        } else {
          // 10-bit 尾数舍入模拟
          let exp = val.abs().log2().floor();
          let step = 2.0_f64.powf(exp - 10.0);
          (val / step).round() * step
        }
      }
      Self::Bf16 => {
        if val.abs() == 0.0 {
          0.0
        } else {
          // 7-bit 尾数（动态范围与 FP32 同达 3.4e38，不会在 65504 溢出）
          let exp = val.abs().log2().floor();
          let step = 2.0_f64.powf(exp - 7.0);
          (val / step).round() * step
        }
      }
    }
  }
}

#[component]
fn NormAndPrecisionLab(initial_mode: String) -> Element {
  let mut mode = use_signal(move || NormLabMode::parse_mode(&initial_mode));
  let mut outlier = use_signal(|| 3.5_f64);
  let mut precision = use_signal(|| FloatFmt::Fp32);
  let mut adaln_alpha = use_signal(|| 0.0_f64);
  let adaln_gamma = 0.2_f64;

  let cur_mode = mode();
  let spike = outlier();
  let fmt = precision();

  // 构造确定性 [N=2, C=4, S=4] 激活张量，(0, 0, 0) 注入可变离群值 spike
  let cell_val = |n: usize, c: usize, s: usize| -> f64 {
    if n == 0 && c == 0 && s == 0 {
      spike
    } else {
      let raw = ((n * 7 + c * 3 + s * 5) as f64 * 0.37).sin() * 1.4 + (c as f64 - 1.5) * 0.4;
      (raw * 100.0).round() / 100.0
    }
  };

  // 收集与 (0, 0, 0) 同归约组的所有元素，并按当前精度模拟归约计算
  let mut group_vals = Vec::new();
  for n in 0..2 {
    for c in 0..4 {
      for s in 0..4 {
        if cur_mode.in_same_group(n, c, s) {
          group_vals.push(fmt.quantize(cell_val(n, c, s)));
        }
      }
    }
  }

  let count = group_vals.len().max(1) as f64;
  let mean = if cur_mode == NormLabMode::RmsNorm {
    0.0
  } else {
    fmt.quantize(group_vals.iter().sum::<f64>() / count)
  };

  let mut sq_sum = 0.0_f64;
  for &v in &group_vals {
    let centered = fmt.quantize(v - mean);
    let sq = fmt.quantize(centered * centered);
    sq_sum = fmt.quantize(sq_sum + sq);
  }
  let denom = fmt.quantize((fmt.quantize(sq_sum / count) + 1e-5).sqrt());
  let overflowed = !sq_sum.is_finite() || !denom.is_finite() || denom == 0.0;

  // 计算 token (n=0, s=0) 四个通道归一化前后的值
  let raw_ch: [f64; 4] =
    [cell_val(0, 0, 0), cell_val(0, 1, 0), cell_val(0, 2, 0), cell_val(0, 3, 0)];
  let mut norm_ch = [0.0_f64; 4];
  for ch in 0..4 {
    // 为每个通道各自收集其归约组
    let mut gv = Vec::new();
    for n in 0..2 {
      for c in 0..4 {
        for s in 0..4 {
          let same = match cur_mode {
            NormLabMode::BatchNorm => c == ch,
            NormLabMode::LayerNorm | NormLabMode::RmsNorm | NormLabMode::AdaLnZero => {
              n == 0 && s == 0
            }
            NormLabMode::InstanceNorm => n == 0 && c == ch,
            NormLabMode::GroupNorm => n == 0 && (c / 2 == ch / 2),
          };
          if same {
            gv.push(fmt.quantize(cell_val(n, c, s)));
          }
        }
      }
    }
    let cnt = gv.len().max(1) as f64;
    let mu = if cur_mode == NormLabMode::RmsNorm {
      0.0
    } else {
      fmt.quantize(gv.iter().sum::<f64>() / cnt)
    };
    let mut ss = 0.0_f64;
    for &v in &gv {
      let d = fmt.quantize(v - mu);
      ss = fmt.quantize(ss + fmt.quantize(d * d));
    }
    let std_v = fmt.quantize((fmt.quantize(ss / cnt) + 1e-5).sqrt());
    let normed = if !std_v.is_finite() || std_v == 0.0 {
      f64::NAN
    } else {
      (fmt.quantize(raw_ch[ch]) - mu) / std_v
    };
    norm_ch[ch] = if cur_mode == NormLabMode::AdaLnZero {
      let modulated = (1.0 + adaln_gamma) * normed;
      // 模拟残差门控输出 h' = x + α · SubLayer(modulated)
      raw_ch[ch] + adaln_alpha() * modulated.tanh()
    } else {
      normed
    };
  }

  rsx! {
    div { class: "space-y-4",
      div { class: "flex flex-wrap gap-1.5",
        for m in [
          NormLabMode::RmsNorm,
          NormLabMode::LayerNorm,
          NormLabMode::BatchNorm,
          NormLabMode::GroupNorm,
          NormLabMode::InstanceNorm,
          NormLabMode::AdaLnZero,
        ] {
          button {
            key: "{m.label()}",
            r#type: "button",
            class: if cur_mode == m {
              "px-2.5 py-1.5 rounded-lg text-xs font-semibold bg-indigo-600 text-white cursor-pointer"
            } else {
              "px-2.5 py-1.5 rounded-lg text-xs font-medium bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 cursor-pointer"
            },
            onclick: move |_| mode.set(m),
            "{m.label()}"
          }
        }
      }

      // 控制滑块：离群值注入 + 累加浮点精度切换 + AdaLN 门控参数
      div { class: "grid grid-cols-1 md:grid-cols-3 gap-3 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/70 dark:bg-slate-950/60 p-3 text-xs font-mono",
        div {
          div { class: "mb-1 font-semibold",
            "注入激活离群值 x[0,0,0] = {spike:.1}"
          }
          input {
            r#type: "range", min: "1.0", max: "300.0", step: "5.0", value: "{spike}",
            class: "w-full accent-rose-500 cursor-pointer",
            oninput: move |e| { if let Ok(v) = e.value().parse::<f64>() { outlier.set(v); } },
          }
          div { class: "text-[10px] text-slate-400 mt-0.5", "拉至 ≥260 触发 FP16 x² > 65504 溢出！" }
        }

        div {
          div { class: "mb-1 font-semibold", "归约累加浮点精度" }
          div { class: "flex gap-1.5",
            for (f, lbl) in [(FloatFmt::Fp32, "FP32 安全"), (FloatFmt::Fp16, "纯 FP16"), (FloatFmt::Bf16, "BF16")] {
              button {
                key: "{lbl}",
                r#type: "button",
                class: if fmt == f {
                  "px-2.5 py-1 rounded bg-slate-900 text-white dark:bg-white dark:text-slate-900 font-bold cursor-pointer"
                } else {
                  "px-2.5 py-1 rounded bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 cursor-pointer"
                },
                onclick: move |_| precision.set(f),
                "{lbl}"
              }
            }
          }
        }

        if cur_mode == NormLabMode::AdaLnZero {
          div {
            div { class: "mb-1 font-semibold text-indigo-600 dark:text-indigo-400",
              "AdaLN-Zero 残差门 α = {adaln_alpha():.2}"
            }
            input {
              r#type: "range", min: "0.0", max: "1.5", step: "0.05", value: "{adaln_alpha()}",
              class: "w-full accent-indigo-600 cursor-pointer",
              oninput: move |e| { if let Ok(v) = e.value().parse::<f64>() { adaln_alpha.set(v); } },
            }
            div { class: "text-[10px] text-emerald-600 dark:text-emerald-400",
              if adaln_alpha() == 0.0 { "α=0：严格恒等映射 h' = x！" } else { "α>0：逐步注入条件调制分支" }
            }
          }
        } else {
          div { class: "flex flex-col justify-center",
            div { class: "text-[11px] text-slate-500", "当前归约切片大小：{group_vals.len()} 个元素" }
            div { class: "font-bold text-slate-800 dark:text-slate-100 mt-0.5",
              if overflowed {
                span { class: "text-rose-600 dark:text-rose-400", "⚠️ 平方和 Inf → 输出 NaN！" }
              } else {
                "μ = {mean:+.2} · RMS/σ = {denom:.2}"
              }
            }
          }
        }
      }

      // [N=2, C=4, S=4] 张量切片可视化 SVG（靛蓝色高亮表示与 [0,0,0] 共享同一均值/方差统计量）
      div { class: "w-full overflow-x-auto rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950 p-3",
        svg {
          view_box: "0 0 680 205",
          class: "w-full min-w-[520px] h-auto font-mono select-none",
          role: "img",
          "aria-label": "3D tensor normalization slice and channel output comparison",

          for n in 0..2_usize {
            {
              let base_x = 24.0 + (n as f64) * 185.0;
              rsx! {
                g { key: "batch-{n}",
                  text { x: "{base_x + 70.0}", y: "20", text_anchor: "middle", font_size: "10.5", class: "fill-slate-700 dark:fill-slate-300 font-bold",
                    "Sample N={n} [C=4 × S=4]"
                  }
                  for c in 0..4_usize {
                    for s in 0..4_usize {
                      {
                        let cx = base_x + (s as f64) * 35.0;
                        let cy = 32.0 + (c as f64) * 34.0;
                        let active = cur_mode.in_same_group(n, c, s);
                        let is_spike = n == 0 && c == 0 && s == 0;
                        let cls = if is_spike {
                          "fill-rose-500 stroke-rose-700"
                        } else if active {
                          "fill-indigo-500/25 dark:fill-indigo-500/45 stroke-indigo-600 dark:stroke-indigo-400"
                        } else {
                          "fill-slate-100 dark:fill-slate-900 stroke-slate-300 dark:stroke-slate-800"
                        };
                        rsx! {
                          g { key: "cell-{n}-{c}-{s}",
                            rect { x: "{cx}", y: "{cy}", width: "31", height: "28", rx: "4", class: "{cls}", stroke_width: if active { "1.8" } else { "1" } }
                            text {
                              x: "{cx + 15.5}",
                              y: "{cy + 18.0}",
                              text_anchor: "middle",
                              font_size: "8.5",
                              class: if is_spike { "fill-white font-bold" } else { "fill-slate-700 dark:fill-slate-200" },
                              "{cell_val(n, c, s):.1}"
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

          text { x: "200", y: "188", text_anchor: "middle", font_size: "9.5", class: "fill-indigo-600 dark:fill-indigo-400 font-semibold",
            "■ 紫框高亮单元 = 与红色离群点 x[0,0,0] 一起求均值/均方根的统计归约域"
          }

          // 右侧：Token (0,0) 四通道输出对比
          text { x: "535", y: "20", text_anchor: "middle", font_size: "10.5", class: "fill-slate-800 dark:fill-slate-200 font-bold",
            if cur_mode == NormLabMode::AdaLnZero { "Token(0,0) 残差流输出 h'" } else { "Token(0,0) 归一化后 4 通道输出" }
          }
          for ch in 0..4_usize {
            {
              let y = 36.0 + (ch as f64) * 36.0;
              let nv = norm_ch[ch];
              let is_nan = !nv.is_finite();
              let bar_w = if is_nan { 160.0 } else { (nv.abs().min(4.0) / 4.0) * 140.0 };
              rsx! {
                g { key: "nout-{ch}",
                  text { x: "405", y: "{y + 16.0}", font_size: "10", class: "fill-slate-500", "C{ch}" }
                  rect {
                    x: "430",
                    y: "{y + 2.0}",
                    width: "{bar_w.max(4.0):.1}",
                    height: "20",
                    rx: "4",
                    class: if is_nan { "fill-rose-600" } else if nv >= 0.0 { "fill-emerald-500" } else { "fill-sky-500" },
                  }
                  text {
                    x: "585",
                    y: "{y + 16.0}",
                    font_size: "10",
                    class: if is_nan { "fill-rose-600 font-bold" } else { "fill-slate-800 dark:fill-slate-100 font-bold" },
                    if is_nan { "NaN (FP16 Overflow)" } else { "{nv:+.3}" }
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
// 内核 4：PositionalGeometryLab（1D RoPE 内积平移不变性、PI / NTK / YaRN 外推、2D Axial RoPE）
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PosLabMode {
  Rope1d,
  PosInterp,
  NtkAware,
  Yarn,
  Axial2d,
}

impl PosLabMode {
  fn parse_mode(raw: &str) -> Self {
    match raw.trim().to_ascii_lowercase().as_str() {
      "rope" | "rope1d" | "1d" => Self::Rope1d,
      "pi" | "linear" | "pos_interp" => Self::PosInterp,
      "ntk" | "ntk_aware" => Self::NtkAware,
      "axial" | "2d" | "mrope" | "axial2d" => Self::Axial2d,
      _ => Self::Yarn,
    }
  }

  fn label(self) -> &'static str {
    match self {
      Self::Rope1d => "标准 1D RoPE",
      Self::PosInterp => "Position Interp (m/s)",
      Self::NtkAware => "NTK-Aware 基底扩展",
      Self::Yarn => "YaRN 分频斜坡 + √t",
      Self::Axial2d => "2D Axial RoPE (图像 H×W)",
    }
  }
}

/// 计算第 pair_idx 组二维子空间（共 8 组，d=16）在不同外推策略下的有效角频率与 YaRN 斜坡权重 γ ∈ [0,1]。
fn rotary_freq_and_gamma(mode: PosLabMode, pair_idx: usize, scale: f64) -> (f64, f64, f64) {
  let d = 16.0_f64;
  let base = 10_000.0_f64;
  let exp = (2 * pair_idx) as f64 / d;
  let theta_0 = base.powf(-exp);
  let s = scale.max(1.0);

  match mode {
    PosLabMode::Rope1d | PosLabMode::Axial2d => (theta_0, 1.0, 1.0),
    PosLabMode::PosInterp => (theta_0 / s, 0.0, 1.0),
    PosLabMode::NtkAware => {
      let scaled_base = base * s.powf(d / (d - 2.0));
      (scaled_base.powf(-exp), 0.5, 1.0)
    }
    PosLabMode::Yarn => {
      // 按波长 λ = 2π/θ_0 与训练窗口 L_train=32 的比例划分：高频 γ=1 不插值，低频 γ=0 内插
      let wavelength = 2.0 * PI / theta_0;
      let r = 32.0 / wavelength;
      let alpha_b = 1.0;
      let beta_b = 8.0;
      let gamma = ((r - alpha_b) / (beta_b - alpha_b)).clamp(0.0, 1.0);
      let theta_yarn = (1.0 - gamma) * (theta_0 / s) + gamma * theta_0;
      let attn_temp_sqrt = 0.1 * s.ln() + 1.0;
      (theta_yarn, gamma, attn_temp_sqrt)
    }
  }
}

#[component]
fn PositionalGeometryLab(initial_mode: String) -> Element {
  let mut mode = use_signal(move || PosLabMode::parse_mode(&initial_mode));
  let mut pos_m = use_signal(|| 6_i32);
  let mut pos_n = use_signal(|| 18_i32);
  let mut scale = use_signal(|| 4.0_f64);
  let mut img_row = use_signal(|| 3_i32);
  let mut img_col = use_signal(|| 5_i32);

  let cur_mode = mode();
  let m = pos_m();
  let n = pos_n();
  let delta = m - n;
  let s = scale();

  // 计算当前 (m, n) 在 8 个二维旋转子空间上的精确旋转内积 Σ cos((m-n)·θ_j) · temp
  let score_at_delta = |dist: f64, m_sel: PosLabMode, sc: f64| -> f64 {
    let mut acc = 0.0;
    for j in 0..8 {
      let (th, _, temp) = rotary_freq_and_gamma(m_sel, j, sc);
      acc += (dist * th).cos() * temp;
    }
    acc / 8.0
  };

  let cur_score = score_at_delta(delta as f64, cur_mode, s);
  let shifted_score = score_at_delta(((m + 15) - (n + 15)) as f64, cur_mode, s);

  // 右侧衰减曲线坐标：Δ ∈ [0, 64] -> x ∈ [355, 655], score ∈ [-0.5, 1.15] -> y ∈ [225, 40]
  let mut curve_base = String::with_capacity(1024);
  let mut curve_cur = String::with_capacity(1024);
  for d_step in 0..=64 {
    let x = 355.0 + (d_step as f64 / 64.0) * 300.0;
    let y_base =
      225.0 - ((score_at_delta(d_step as f64, PosLabMode::Rope1d, 1.0) + 0.5) / 1.65) * 185.0;
    let y_cur = 225.0 - ((score_at_delta(d_step as f64, cur_mode, s) + 0.5) / 1.65) * 185.0;
    if d_step > 0 {
      curve_base.push(' ');
      curve_cur.push(' ');
    }
    curve_base.push_str(&format!("{x:.1},{y_base:.1}"));
    curve_cur.push_str(&format!("{x:.1},{y_cur:.1}"));
  }
  let abs_delta = delta.unsigned_abs().min(64) as f64;
  let marker_x = 355.0 + (abs_delta / 64.0) * 300.0;
  let marker_y = 225.0 - ((cur_score + 0.5) / 1.65) * 185.0;

  // 选取 4 个代表性频率通道展示复平面表盘
  let dial_pairs: [usize; 4] = [0, 2, 4, 7];

  rsx! {
    div { class: "space-y-4",
      div { class: "flex flex-wrap gap-1.5",
        for m_opt in [
          PosLabMode::Rope1d,
          PosLabMode::PosInterp,
          PosLabMode::NtkAware,
          PosLabMode::Yarn,
          PosLabMode::Axial2d,
        ] {
          button {
            key: "{m_opt.label()}",
            r#type: "button",
            class: if cur_mode == m_opt {
              "px-2.5 py-1.5 rounded-lg text-xs font-semibold bg-indigo-600 text-white cursor-pointer"
            } else {
              "px-2.5 py-1.5 rounded-lg text-xs font-medium bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 cursor-pointer"
            },
            onclick: move |_| mode.set(m_opt),
            "{m_opt.label()}"
          }
        }
      }

      // 控制面板
      div { class: "grid grid-cols-1 md:grid-cols-3 gap-3 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/70 dark:bg-slate-950/60 p-3 text-xs font-mono",
        if cur_mode == PosLabMode::Axial2d {
          div {
            div { class: "mb-1 font-semibold", "图像 Patch 行坐标 y = {img_row()}" }
            input {
              r#type: "range", min: "0", max: "15", step: "1", value: "{img_row()}",
              class: "w-full accent-fuchsia-600 cursor-pointer",
              oninput: move |e| { if let Ok(v) = e.value().parse::<i32>() { img_row.set(v); pos_m.set(v); } },
            }
          }
          div {
            div { class: "mb-1 font-semibold", "图像 Patch 列坐标 x = {img_col()}" }
            input {
              r#type: "range", min: "0", max: "15", step: "1", value: "{img_col()}",
              class: "w-full accent-cyan-600 cursor-pointer",
              oninput: move |e| { if let Ok(v) = e.value().parse::<i32>() { img_col.set(v); pos_n.set(v); } },
            }
          }
          div { class: "flex flex-col justify-center text-[11px] text-slate-600 dark:text-slate-300",
            "2D Axial RoPE：前一半通道旋转 y·θ_j（垂直轴），后一半通道旋转 x·θ_j（水平轴），天然支持任意长宽比！"
          }
        } else {
          div {
            div { class: "flex justify-between mb-1 font-semibold",
              span { "Query 位置 m = {m}" }
              button {
                r#type: "button",
                class: "text-[11px] text-indigo-600 dark:text-indigo-400 underline cursor-pointer",
                onclick: move |_| {
                  pos_m.set((m + 4).min(60));
                  pos_n.set((n + 4).min(64));
                },
                "整体平移 +4 (验不变性)"
              }
            }
            input {
              r#type: "range", min: "0", max: "64", step: "1", value: "{m}",
              class: "w-full accent-indigo-600 cursor-pointer",
              oninput: move |e| { if let Ok(v) = e.value().parse::<i32>() { pos_m.set(v); } },
            }
          }
          div {
            div { class: "mb-1 font-semibold", "Key 位置 n = {n} (相对距离 Δ = {delta})" }
            input {
              r#type: "range", min: "0", max: "64", step: "1", value: "{n}",
              class: "w-full accent-emerald-600 cursor-pointer",
              oninput: move |e| { if let Ok(v) = e.value().parse::<i32>() { pos_n.set(v); } },
            }
          }
          div {
            div { class: "mb-1 font-semibold", "外推窗口倍率 s = {s:.0}×" }
            div { class: "flex gap-1.5",
              for sv in [1.0_f64, 2.0, 4.0, 8.0] {
                button {
                  key: "sc-{sv}",
                  r#type: "button",
                  class: if (s - sv).abs() < 0.01 {
                    "px-2.5 py-0.5 rounded bg-amber-600 text-white font-bold cursor-pointer"
                  } else {
                    "px-2.5 py-0.5 rounded bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 cursor-pointer"
                  },
                  onclick: move |_| scale.set(sv),
                  "{sv:.0}×"
                }
              }
            }
          }
        }
      }

      // 左：4 组复平面多频旋转圆盘 | 右：相对距离内积衰减包络
      div { class: "w-full overflow-x-auto rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950 p-3",
        svg {
          view_box: "0 0 680 255",
          class: "w-full min-w-[540px] h-auto font-mono select-none",
          role: "img",
          "aria-label": "Rotary position embedding complex plane dials and relative distance decay curve",

          text { x: "165", y: "18", text_anchor: "middle", font_size: "10.5", class: "fill-slate-700 dark:fill-slate-200 font-bold",
            if cur_mode == PosLabMode::Axial2d {
              "复平面轴向分解表盘（紫=行 y 轴, 绿=列 x 轴）"
            } else {
              "复平面频率旋转表盘（紫=R_m·q, 绿=R_n·k）"
            }
          }

          for (slot, pair_j) in dial_pairs.iter().copied().enumerate() {
            {
              let col = slot % 2;
              let row = slot / 2;
              let cx = 85.0 + (col as f64) * 155.0;
              let cy = 78.0 + (row as f64) * 104.0;
              let r = 34.0;
              let (theta, gamma, _) = rotary_freq_and_gamma(cur_mode, pair_j, s);
              let ang_m = (m as f64) * theta;
              let ang_n = (n as f64) * theta;
              let qx = cx + r * ang_m.cos();
              let qy = cy - r * ang_m.sin();
              let kx = cx + r * ang_n.cos();
              let ky = cy - r * ang_n.sin();
              let freq_tag = match slot {
                0 => "高频 Pair#0",
                1 => "中高频 Pair#2",
                2 => "中低频 Pair#4",
                _ => "超低频 Pair#7",
              };

              rsx! {
                g { key: "dial-{pair_j}",
                  circle { cx: "{cx}", cy: "{cy}", r: "{r}", fill: "none", class: "stroke-slate-300 dark:stroke-slate-700", stroke_width: "1.4" }
                  line { x1: "{cx - r}", y1: "{cy}", x2: "{cx + r}", y2: "{cy}", class: "stroke-slate-200 dark:stroke-slate-800", stroke_width: "1" }
                  line { x1: "{cx}", y1: "{cy - r}", x2: "{cx}", y2: "{cy + r}", class: "stroke-slate-200 dark:stroke-slate-800", stroke_width: "1" }
                  // R_m q 指针
                  line { x1: "{cx}", y1: "{cy}", x2: "{qx:.1}", y2: "{qy:.1}", class: "stroke-indigo-600 dark:stroke-indigo-400", stroke_width: "2.4" }
                  circle { cx: "{qx:.1}", cy: "{qy:.1}", r: "3.5", class: "fill-indigo-600 dark:fill-indigo-400" }
                  // R_n k 指针
                  line { x1: "{cx}", y1: "{cy}", x2: "{kx:.1}", y2: "{ky:.1}", class: "stroke-emerald-500", stroke_width: "2.2" }
                  circle { cx: "{kx:.1}", cy: "{ky:.1}", r: "3.5", class: "fill-emerald-500" }

                  text { x: "{cx}", y: "{cy + r + 13.0}", text_anchor: "middle", font_size: "9", class: "fill-slate-700 dark:fill-slate-300 font-semibold",
                    "{freq_tag} (γ={gamma:.1})"
                  }
                }
              }
            }
          }

          // 右侧：相对距离内积衰减波形
          text { x: "505", y: "18", text_anchor: "middle", font_size: "10.5", class: "fill-slate-700 dark:fill-slate-200 font-bold",
            "相对距离内积响应 ⟨R_m q, R_n k⟩ vs |m - n|"
          }
          rect { x: "355", y: "32", width: "300", height: "198", rx: "6", class: "fill-slate-50/70 dark:fill-slate-900/50 stroke-slate-200 dark:stroke-slate-800", stroke_width: "1" }
          polyline {
            points: "{curve_base}",
            fill: "none",
            class: "stroke-slate-400/70",
            stroke_dasharray: "4 3",
            stroke_width: "1.5",
          }
          polyline {
            points: "{curve_cur}",
            fill: "none",
            class: "stroke-indigo-600 dark:stroke-indigo-400",
            stroke_width: "2.3",
          }
          circle {
            cx: "{marker_x:.1}",
            cy: "{marker_y:.1}",
            r: "5",
            class: "fill-rose-500 stroke-white dark:stroke-slate-950",
            stroke_width: "1.8",
          }
          text { x: "362", y: "48", font_size: "9.5", class: "fill-indigo-600 dark:fill-indigo-400 font-bold", "— 当前策略 ({s:.0}×)" }
          text { x: "362", y: "62", font_size: "9.5", class: "fill-slate-400", "-- 原始 1× RoPE 基准" }
          text { x: "505", y: "245", text_anchor: "middle", font_size: "9.5", class: "fill-slate-500",
            "相对距离 |m - n| ∈ [0, 64] · 当前 ⟨R_{m}q, R_{n}k⟩ = {cur_score:+.4} (平移+15后 = {shifted_score:+.4})"
          }
        }
      }
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// MDX `<ComponentLab ... />` 全局嵌入组件适配器
// ─────────────────────────────────────────────────────────────────────────────

/// 支持在任意 Markdown / 课程 / 博客中写入：
/// - `<ComponentLab kind="grid_kernel" mode="depthwise_separable" />`
/// - `<ComponentLab kind="activation" mode="swiglu" />`
/// - `<ComponentLab id="rope-and-yarn-extrapolation" />`
pub struct ComponentLabMdx;

impl widgets::MdxComponent for ComponentLabMdx {
  fn name(&self) -> &'static str {
    "ComponentLab"
  }

  fn render(&self, attrs: &HashMap<String, String>) -> Element {
    let id = attrs.get("id").map(|s| s.trim()).unwrap_or_default();
    let explicit_kind = attrs.get("kind").map(|s| s.as_str()).unwrap_or_default();
    let explicit_mode = attrs.get("mode").cloned().unwrap_or_default();

    let (kind, initial_mode) = if !explicit_kind.is_empty() {
      (ComponentLabKind::parse_str(explicit_kind), explicit_mode)
    } else {
      crate::components_atlas::infer_lab_preset_by_slug(id, &explicit_mode)
    };

    rsx! {
      ComponentLabWorkbench {
        kind,
        initial_mode,
        embedded: true,
      }
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn activation_math_and_gelu_approx_bounds_are_exact() {
    assert!((ActivationFn::Sigmoid.eval(0.0, 1.0) - 0.5).abs() < 1e-12);
    assert!((ActivationFn::Sigmoid.deriv(0.0, 1.0) - 0.25).abs() < 1e-8);
    assert_eq!(ActivationFn::ReLU.eval(-3.2, 1.0), 0.0);
    // 验证精确 GELU 与 Tanh 近似在 [-5, 5] 上的最大残差 ||GELU_exact - GELU_tanh||_∞ ≈ 4.73e-4
    let mut max_err = 0.0_f64;
    for i in -100..=100 {
      let x = (i as f64) * 0.05;
      let diff = (ActivationFn::GELUExact.eval(x, 1.0) - ActivationFn::GELUTanh.eval(x, 1.0)).abs();
      max_err = max_err.max(diff);
    }
    assert!(max_err > 4.5e-4 && max_err < 5.0e-4, "unexpected GELU approx diff: {max_err}");
  }

  #[test]
  fn fp16_square_overflow_triggers_above_256_while_bf16_stays_finite() {
    let x = 260.0_f64;
    let sq_fp16 = FloatFmt::Fp16.quantize(x * x);
    let sq_bf16 = FloatFmt::Bf16.quantize(x * x);
    assert!(!sq_fp16.is_finite(), "260^2 = 67600 > 65504 must overflow FP16");
    assert!(sq_bf16.is_finite(), "BF16 shares FP32 exponent range and must stay finite");
  }

  #[test]
  fn rope_relative_dot_product_is_strictly_shift_invariant() {
    let mut s1 = 0.0;
    let mut s2 = 0.0;
    for j in 0..8 {
      let (th, _, _) = rotary_freq_and_gamma(PosLabMode::Rope1d, j, 1.0);
      // 模拟 q = [1, 0], k = [1, 0] 经 R_m 与 R_n 旋转后的内积 cos((m - n) * th)
      let (m1, n1) = (5.0_f64, 19.0_f64);
      let (m2, n2) = (42.0_f64, 56.0_f64);
      let r1 = (m1 * th).cos() * (n1 * th).cos() + (m1 * th).sin() * (n1 * th).sin();
      let r2 = (m2 * th).cos() * (n2 * th).cos() + (m2 * th).sin() * (n2 * th).sin();
      s1 += r1;
      s2 += r2;
    }
    assert!((s1 - s2).abs() < 1e-12, "RoPE inner product must depend only on m - n");
  }

  #[test]
  fn ssr_renders_all_four_m1_interactive_labs_deterministically() {
    #[component]
    fn AllLabsHost() -> Element {
      rsx! {
        div {
          ComponentLabWorkbench {
            kind: ComponentLabKind::Activation,
            initial_mode: "swiglu".to_string(),
            embedded: true,
          }
          ComponentLabWorkbench {
            kind: ComponentLabKind::GridKernel,
            initial_mode: "depthwise_separable".to_string(),
            embedded: true,
          }
          ComponentLabWorkbench {
            kind: ComponentLabKind::NormPrecision,
            initial_mode: "rmsnorm".to_string(),
            embedded: true,
          }
          ComponentLabWorkbench {
            kind: ComponentLabKind::Positional,
            initial_mode: "yarn".to_string(),
            embedded: true,
          }
        }
      }
    }

    let mut vdom = VirtualDom::new(AllLabsHost);
    vdom.rebuild_in_place();
    let html = dioxus_ssr::render(&vdom);
    assert!(html.contains("SwiGLU"));
    assert!(html.contains("Depthwise"));
    assert!(html.contains("RMSNorm"));
    assert!(html.contains("YaRN"));
  }
}
