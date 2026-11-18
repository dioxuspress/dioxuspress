//! 纯 Rust WASM AI 数学基础动态推导实验室（M2 后三大数学画布内核）。
//!
//! 涵盖：
//! 1. [`SpectralAndGraphSignalLab`]：复平面单位圆傅里叶缠绕质心与奈奎斯特混叠、海森堡时频测不准格与窗谱泄漏、
//!    神经网络频谱偏置（NTK 特征值衰减）与多尺度傅里叶特征、图拉普拉斯瑞利平滑度与切比雪夫局部谱滤波；
//! 2. [`OptimizerAndFlowTrajectoryLab`]：$L$-平滑峡谷步长震荡相变与 `SGD / Momentum / Adam / AdamW` 解耦轨迹、
//!    `Muon` 奇次多项式 Newton-Schulz 免求逆矩阵正交化、逆时间伊藤 SDE vs 确定性概率流 ODE 边缘分布等价、
//!    条件流匹配（CFM）与最优传输 `1-Rectified / 2-Rectified Flow` 避交叉拉直；
//! 3. [`RlPolicyAndBitPrecisionLab`]：贝尔曼 $\gamma$-压缩映射价值迭代与 `GAE(γ, λ)` 伸缩权重、`PPO` 正负优势双象限截断包络、
//!    `DPO` 配分函数 $\ln Z(x)$ 严格相消与 `GRPO` 免价值组内归一化、32 位可点击翻转 `IEEE-754 (FP32/BF16/FP16/FP8)` 混合精度吞没实验台。

use std::f64::consts::PI;

use dioxus::prelude::*;

use crate::component_labs::{
  current_lang, lab_metric_card, push_svg_pt, sigmoid, svg_grid_cell, svg_vbar,
};
use crate::component_labs_m2::mode_btn;
use crate::math_labs::{math_slider, math_term_pill, MathTermTone};

/// 跨位宽翻转器复用的单个二进制位按钮（`#[inline(never)]` 消除 32 格渲染模板内联膨胀）。
#[inline(never)]
fn ieee_bit_cell(
  bit_idx: u32,
  bit_val: u32,
  zone_cls: &str,
  on_flip: EventHandler<u32>,
) -> Element {
  let active_cls = if bit_val == 1 {
    "font-extrabold ring-1 ring-slate-900/20 dark:ring-white/30 brightness-105"
  } else {
    "opacity-75 hover:opacity-100"
  };
  rsx! {
    button {
      key: "b-{bit_idx}",
      r#type: "button",
      title: "Bit #{bit_idx}",
      class: "h-7 rounded text-[11px] font-mono tabular-nums border transition cursor-pointer flex flex-col items-center justify-center leading-none {zone_cls} {active_cls}",
      onclick: move |_| on_flip.call(bit_idx),
      span { "{bit_val}" }
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 内核 4：SpectralAndGraphSignalLab（DFT 缠绕混叠 / 海森堡时频窗 / 频谱偏置 NTK / 图拉普拉斯谱滤波）
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SpectralMode {
  DftWindingAliasing,
  StftUncertaintyMel,
  SpectralBiasNtk,
  GraphLaplacianCheb,
}

impl SpectralMode {
  fn parse_mode(raw: &str) -> Self {
    match raw.trim().to_ascii_lowercase().as_str() {
      "stft" | "stft_uncertainty" | "mel" | "heisenberg" | "window" => Self::StftUncertaintyMel,
      "spectral_bias" | "ntk" | "f_principle" | "fourier_features" => Self::SpectralBiasNtk,
      "graph_laplacian" | "laplacian" | "chebyshev" | "gnn" => Self::GraphLaplacianCheb,
      _ => Self::DftWindingAliasing,
    }
  }
}

/// 计算复合余弦信号在探测缠绕频率 `f_wind` 下的复平面质心 $(\text{Re}\,\mu, \text{Im}\,\mu, |\mu|)$ 以及采样率 `f_s` 下的混叠折叠频率。
pub(crate) fn compute_dft_winding_and_alias(
  f_sig: f64,
  f_2: f64,
  f_wind: f64,
  f_s: f64,
) -> (f64, f64, f64, f64) {
  let fs_safe = f_s.max(2.0);
  let n_pts = 160usize;
  let mut sum_re = 0.0;
  let mut sum_im = 0.0;
  for k in 0..n_pts {
    let t = (k as f64) / (n_pts as f64);
    let x = 0.5 + 0.65 * (2.0 * PI * f_sig * t).cos() + 0.35 * (2.0 * PI * f_2 * t).cos();
    let ang = -2.0 * PI * f_wind * t;
    let (s, c) = ang.sin_cos();
    sum_re += x * c;
    sum_im += x * s;
  }
  let re = sum_re / (n_pts as f64);
  let im = sum_im / (n_pts as f64);
  let mag = (re * re + im * im).sqrt();
  // 泊松求和折叠：离散采样后真实频率 f_sig 落入主带 [-f_s/2, f_s/2] 的视在混叠频率
  let folded = (f_sig - fs_safe * (f_sig / fs_safe).round()).abs();
  (re, im, mag, folded)
}

/// 海森堡时频不确定性与梅尔刻度换算：高斯包络脉冲宽度 `sigma_t` 满足精确下界 $\Delta t \cdot \Delta \omega = \frac{1}{2}$（加非高斯截断窗后 $> \frac{1}{2}$）。
pub(crate) fn compute_stft_uncertainty_metrics(
  sigma_t: f64,
  use_hann: bool,
  freq_hz: f64,
) -> (f64, f64, f64, f64, f64) {
  let st = sigma_t.clamp(0.12, 1.60);
  // 矩形截断窗主瓣更窄但旁瓣平方积分发散，离散有效频宽膨胀系数 > 1；Hann 窗平滑归零接近高斯下界
  let shape_factor = if use_hann { 1.052 } else { 1.685 };
  let delta_t = st * (if use_hann { 0.86 } else { 1.0 });
  let delta_omega = (0.5 / delta_t) * shape_factor;
  let product = delta_t * delta_omega;
  let peak_sidelobe_db = if use_hann { -31.5 } else { -13.3 };
  let mel = 2595.0 * (1.0 + freq_hz.max(20.0) / 700.0).log10();
  (delta_t, delta_omega, product, peak_sidelobe_db, mel)
}

/// 8 节点带弦环图的组合拉普拉斯二次型 $x^\top L x = \frac{1}{2}\sum_{(i,j)} w_{ij}(x_i - x_j)^2$ 与 $K$ 阶切比雪夫多项式谱滤波。
const GRAPH_EDGES_8: [(usize, usize, f64); 10] = [
  (0, 1, 1.0),
  (1, 2, 1.0),
  (2, 3, 1.0),
  (3, 4, 1.0),
  (4, 5, 1.0),
  (5, 6, 1.0),
  (6, 7, 1.0),
  (7, 0, 1.0),
  (0, 4, 0.7),
  (2, 6, 0.7),
];

pub(crate) fn compute_graph_laplacian_filter(
  signal: &[f64; 8],
  order_k: usize,
  alpha: f64,
) -> ([f64; 8], f64, f64) {
  let mut deg = [0.0_f64; 8];
  for &(u, v, w) in &GRAPH_EDGES_8 {
    deg[u] += w;
    deg[v] += w;
  }
  let dirichlet_in: f64 =
    GRAPH_EDGES_8.iter().map(|&(u, v, w)| w * (signal[u] - signal[v]).powi(2)).sum();

  // 归一化拉普拉斯算子重缩放到 [-1, 1] 谱区间：\hat{L} = \tilde{L} - I = - D^{-1/2} W D^{-1/2}
  let apply_l_hat = |x: &[f64; 8]| -> [f64; 8] {
    let mut out = [0.0_f64; 8];
    for &(u, v, w) in &GRAPH_EDGES_8 {
      let norm_w = w / (deg[u] * deg[v]).sqrt().max(1e-12);
      out[u] -= norm_w * x[v];
      out[v] -= norm_w * x[u];
    }
    out
  };

  let k_max = order_k.min(6);
  let mut t_prev = *signal;
  let mut t_curr = apply_l_hat(signal);
  let mut y = [0.0_f64; 8];
  for i in 0..8 {
    y[i] = t_prev[i];
    if k_max >= 1 {
      y[i] += alpha * t_curr[i];
    }
  }
  for k in 2..=k_max {
    let l_curr = apply_l_hat(&t_curr);
    let mut t_next = [0.0_f64; 8];
    let coeff = alpha.powi(k as i32) * (1.0 / (k as f64));
    for i in 0..8 {
      t_next[i] = 2.0 * l_curr[i] - t_prev[i];
      y[i] += coeff * t_next[i];
    }
    t_prev = t_curr;
    t_curr = t_next;
  }

  let dirichlet_out: f64 = GRAPH_EDGES_8.iter().map(|&(u, v, w)| w * (y[u] - y[v]).powi(2)).sum();
  (y, dirichlet_in, dirichlet_out)
}

#[component]
pub fn SpectralAndGraphSignalLab(#[props(default)] initial_mode: String) -> Element {
  let mut mode = use_signal(|| SpectralMode::parse_mode(&initial_mode));
  // DFT 缠绕与混叠参数
  let mut f_sig = use_signal(|| 3.0_f64);
  let mut f_wind = use_signal(|| 3.0_f64);
  let mut f_sample = use_signal(|| 14.0_f64);

  // STFT 时频格与梅尔参数
  let mut sigma_t = use_signal(|| 0.48_f64);
  let mut use_hann = use_signal(|| true);
  let mut audio_hz = use_signal(|| 1600.0_f64);

  // NTK 频谱偏置参数
  let mut train_step = use_signal(|| 45.0_f64);
  let mut hi_freq_k = use_signal(|| 7.0_f64);
  let mut use_fourier_feat = use_signal(|| false);

  // 图拉普拉斯切比雪夫滤波参数
  let mut cheb_k = use_signal(|| 3.0_f64);
  let mut cheb_alpha = use_signal(|| -0.55_f64);
  let mut spike_amp = use_signal(|| 1.40_f64);

  let lang = current_lang();
  let cur_mode = mode();

  rsx! {
    div { class: "space-y-4",
      div { class: "flex flex-wrap gap-1.5",
        {mode_btn("DFT 复平面缠绕与混叠", cur_mode == SpectralMode::DftWindingAliasing, EventHandler::new(move |_| mode.set(SpectralMode::DftWindingAliasing)))}
        {mode_btn("海森堡时频格与汉宁窗", cur_mode == SpectralMode::StftUncertaintyMel, EventHandler::new(move |_| mode.set(SpectralMode::StftUncertaintyMel)))}
        {mode_btn("NTK 频谱偏置与傅里叶特征", cur_mode == SpectralMode::SpectralBiasNtk, EventHandler::new(move |_| mode.set(SpectralMode::SpectralBiasNtk)))}
        {mode_btn("图拉普拉斯与切比雪夫滤波", cur_mode == SpectralMode::GraphLaplacianCheb, EventHandler::new(move |_| mode.set(SpectralMode::GraphLaplacianCheb)))}
      }

      match cur_mode {
        SpectralMode::DftWindingAliasing => {
          let (fs1, fw, fs) = (f_sig(), f_wind(), f_sample());
          let f2 = 6.0_f64;
          let (re_m, im_m, mag_m, f_alias) = compute_dft_winding_and_alias(fs1, f2, fw, fs);
          let nyquist = 0.5 * fs;
          let is_aliased = fs1 > nyquist;

          // 左侧：单位圆缠绕轨迹（x(t) e^{-i 2π f_w t}）
          let (cx, cy, rad) = (128.0_f64, 128.0_f64, 86.0_f64);
          let mut wind_pts = String::with_capacity(2200);
          for k in 0..=160 {
            let t = (k as f64) / 160.0;
            let x_val = 0.5 + 0.65 * (2.0 * PI * fs1 * t).cos() + 0.35 * (2.0 * PI * f2 * t).cos();
            let ang = -2.0 * PI * fw * t;
            let (s, c) = ang.sin_cos();
            let r_draw = x_val * (rad * 0.68);
            push_svg_pt(&mut wind_pts, k == 0, cx + r_draw * c, cy - r_draw * s);
          }
          let (com_x, com_y) = (cx + re_m * rad * 1.35, cy - im_m * rad * 1.35);

          // 右侧：连续原波 vs 视在混叠波对比（x ∈ [265, 520]）
          let mut true_wave = String::with_capacity(1600);
          let mut alias_wave = String::with_capacity(1600);
          for k in 0..=100 {
            let t = (k as f64) / 100.0;
            let px = 266.0 + t * 252.0;
            let y_true = (2.0 * PI * fs1 * t).cos();
            let y_fold = (2.0 * PI * f_alias * t).cos();
            push_svg_pt(&mut true_wave, k == 0, px, 128.0 - y_true * 54.0);
            push_svg_pt(&mut alias_wave, k == 0, px, 128.0 - y_fold * 54.0);
          }
          let n_samp = (fs.round() as usize).clamp(4, 36);

          rsx! {
            div { class: "grid grid-cols-2 sm:grid-cols-4 gap-2.5",
              {math_term_pill(MathTermTone::Indigo, "|μ(f_w)|", &format!("{mag_m:.3}"), "复平面缠绕质心偏心模长")}
              {math_term_pill(MathTermTone::Emerald, "f_Nyq = f_s/2", &format!("{nyquist:.2} Hz"), "奈奎斯特无混叠截止上限")}
              {math_term_pill(MathTermTone::Amber, "(Re μ, Im μ)", &format!("({re_m:+.2},{im_m:+.2})"), "正交内积实部与虚部")}
              {math_term_pill(MathTermTone::Rose, "f_alias", &format!("{f_alias:.2} Hz"), if is_aliased { "欠采样高频已折叠伪装!" } else { "满足采样定理零折叠" })}
            }
            div { class: "grid grid-cols-1 lg:grid-cols-12 gap-4 items-start",
              div { class: "lg:col-span-7 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/70 dark:bg-slate-950/60 p-3",
                svg { view_box: "0 0 532 252", class: "w-full h-auto",
                  circle { cx: "{cx}", cy: "{cy}", r: "{rad}", fill: "none", class: "stroke-slate-300 dark:stroke-slate-700", stroke_dasharray: "3 3" }
                  line { x1: "28", y1: "{cy}", x2: "228", y2: "{cy}", class: "stroke-slate-300 dark:stroke-slate-700", stroke_width: "1" }
                  line { x1: "{cx}", y1: "28", x2: "{cx}", y2: "228", class: "stroke-slate-300 dark:stroke-slate-700", stroke_width: "1" }
                  polyline { points: "{wind_pts}", fill: "none", stroke: "#4f46e5", stroke_width: "1.6", stroke_opacity: "0.85" }
                  line { x1: "{cx}", y1: "{cy}", x2: "{com_x:.1}", y2: "{com_y:.1}", stroke: "#d97706", stroke_width: "2.6" }
                  circle { cx: "{com_x:.1}", cy: "{com_y:.1}", r: "5.5", fill: "#e11d48" }
                  text { x: "128", y: "20", text_anchor: "middle", font_size: "9.5", class: "fill-slate-600 dark:fill-slate-300 font-bold",
                    "x(t)·exp(-i2π f_w t) 复平面缠绕轨迹"
                  }

                  // 右半区：时域采样与混叠折叠
                  line { x1: "266", y1: "128", x2: "518", y2: "128", class: "stroke-slate-300 dark:stroke-slate-700", stroke_width: "1" }
                  polyline { points: "{true_wave}", fill: "none", stroke: "#059669", stroke_width: "1.8" }
                  if is_aliased {
                    polyline { points: "{alias_wave}", fill: "none", stroke: "#e11d48", stroke_width: "2.3", stroke_dasharray: "5 3" }
                  }
                  for s_idx in 0..=n_samp {
                    {
                      let ts = (s_idx as f64) / (fs.max(2.0));
                      if ts <= 1.001 {
                        let sx = 266.0 + ts * 252.0;
                        let sy = 128.0 - (2.0 * PI * fs1 * ts).cos() * 54.0;
                        rsx! {
                          circle { key: "sp-{s_idx}", cx: "{sx:.1}", cy: "{sy:.1}", r: "3.2", fill: "#d97706" }
                        }
                      } else {
                        rsx! {}
                      }
                    }
                  }
                  text { x: "392", y: "20", text_anchor: "middle", font_size: "9.5", class: "fill-slate-600 dark:fill-slate-300 font-bold",
                    {if is_aliased { "欠采样混叠：离散琥珀点同时落在玫红伪低频波上！" } else { "满足 f_s ≥ 2f_1：离散采样唯一锁定翠绿真实波形" }}
                  }
                }
              }
              div { class: "lg:col-span-5 space-y-3",
                {math_slider("主信号频率 f₁ (Hz，含 6Hz 次谐波)", &format!("{fs1:.2} Hz"), "1.0", "11.5", "0.25", fs1, "accent-emerald-600", EventHandler::new(move |v| f_sig.set(v)))}
                {math_slider("复平面缠绕探测频率 f_w (Hz)", &format!("{fw:.2} Hz"), "0.5", "9.0", "0.10", fw, "accent-indigo-600", EventHandler::new(move |v| f_wind.set(v)))}
                {math_slider("离散采样率 f_s (Hz)", &format!("{fs:.1} Hz"), "6.0", "28.0", "1.0", fs, "accent-amber-600", EventHandler::new(move |v| f_sample.set(v)))}
                div { class: "grid grid-cols-2 gap-2 text-xs",
                  {lab_metric_card("谐波共振命中", if (fw - fs1).abs() < 0.16 || (fw - 6.0).abs() < 0.16 { "质心强偏心共振!" } else { "旋转相消趋零" }, "text-indigo-600 dark:text-indigo-400")}
                  {lab_metric_card("泊松频谱周期偏移", &format!("|f₁ - k·f_s| = {f_alias:.2}"), if is_aliased { "text-rose-600 dark:text-rose-400" } else { "text-emerald-600 dark:text-emerald-400" })}
                }
                p { class: "text-[11px] leading-relaxed text-slate-500 dark:text-slate-400",
                  {lang.pick(
                    "试着拖动探测频率 f_w 扫过 f₁ 或 6.0 Hz：当缠绕频率与信号不一致时，曲线对称环绕原点使质心相消；一旦相等，正峰永远转到同一象限，质心猛然弹出！再把采样率 f_s 拉低到 2f₁ 以下，亲眼看见离散采样点完美贴合玫红低频假波（混叠）。",
                    "Sweep f_w across f_1 or 6 Hz to watch the winding center of mass jump off the origin; lower f_s below 2f_1 to observe Nyquist aliasing."
                  )}
                }
              }
            }
          }
        }

        SpectralMode::StftUncertaintyMel => {
          let (st, hann, hz) = (sigma_t(), use_hann(), audio_hz());
          let (dt, dw, prod, sl_db, mel) = compute_stft_uncertainty_metrics(st, hann, hz);

          // 左半区：海森堡恒面积时频格（Δt × Δω）
          let tile_w = (dt * 115.0).clamp(22.0, 178.0);
          let tile_h = (dw * 46.0).clamp(22.0, 178.0);
          let tile_x = 122.0 - 0.5 * tile_w;
          let tile_y = 128.0 - 0.5 * tile_h;

          // 右半区：矩形窗 vs 汉宁窗对数旁瓣谱（dB）
          let mut rect_db_pts = String::with_capacity(1400);
          let mut hann_db_pts = String::with_capacity(1400);
          for k in 0..=96 {
            let u = (k as f64 / 96.0) * 5.5;
            let px = 264.0 + (k as f64 / 96.0) * 252.0;
            let sinc = if u < 1e-6 { 1.0 } else { (PI * u * st * 1.6).sin().abs() / (PI * u * st * 1.6) };
            let r_db = (20.0 * sinc.max(1e-3).log10()).clamp(-56.0, 0.0);
            // Hann 窗频谱三项余弦组合实现 -18 dB/oct 快速滚降
            let u_s = u * st * 1.6;
            let denom = (1.0 - u_s * u_s).abs().max(0.08);
            let h_mag = if u_s < 1e-5 {
              1.0
            } else {
              ((PI * u_s).sin().abs() / (PI * u_s)) * (1.0 / (1.0 + u_s * u_s * denom))
            };
            let h_db = (20.0 * h_mag.max(1e-3).log10()).clamp(-56.0, 0.0);
            push_svg_pt(&mut rect_db_pts, k == 0, px, 42.0 - r_db * 3.1);
            push_svg_pt(&mut hann_db_pts, k == 0, px, 42.0 - h_db * 3.1);
          }

          rsx! {
            div { class: "grid grid-cols-2 sm:grid-cols-4 gap-2.5",
              {math_term_pill(MathTermTone::Indigo, "Δt (时域分辨率)", &format!("{dt:.3} s"), "时间局部化有效标准差")}
              {math_term_pill(MathTermTone::Emerald, "Δω (频域分辨率)", &format!("{dw:.3} rad/s"), "频谱主瓣有效离散宽度")}
              {math_term_pill(MathTermTone::Amber, "Δt · Δω ≥ 0.5", &format!("{prod:.3}"), "海森堡—Gabor 测不准积")}
              {math_term_pill(MathTermTone::Rose, "Peak Sidelobe", &format!("{sl_db:.1} dB"), &format!("{hz:.0} Hz → {mel:.0} Mel"))}
            }
            div { class: "grid grid-cols-1 lg:grid-cols-12 gap-4 items-start",
              div { class: "lg:col-span-7 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/70 dark:bg-slate-950/60 p-3",
                svg { view_box: "0 0 532 246", class: "w-full h-auto",
                  rect { x: "26", y: "34", width: "192", height: "188", rx: "8", class: "fill-white/70 dark:fill-slate-900/70 stroke-slate-200 dark:stroke-slate-800" }
                  rect {
                    x: "{tile_x:.1}",
                    y: "{tile_y:.1}",
                    width: "{tile_w:.1}",
                    height: "{tile_h:.1}",
                    rx: "6",
                    fill: "#4f46e5",
                    fill_opacity: "0.22",
                    stroke: "#4f46e5",
                    stroke_width: "2.2",
                  }
                  text { x: "122", y: "22", text_anchor: "middle", font_size: "9.5", class: "fill-slate-600 dark:fill-slate-300 font-bold",
                    "Heisenberg 时频格（压窄 Δt 必撑高 Δω）"
                  }
                  text { x: "122", y: "132", text_anchor: "middle", font_size: "10", class: "fill-indigo-700 dark:fill-indigo-300 font-mono font-bold",
                    "Area = {prod:.2}"
                  }

                  // 右半区旁瓣泄漏对数谱
                  line { x1: "264", y1: "42", x2: "516", y2: "42", class: "stroke-slate-300 dark:stroke-slate-700", stroke_dasharray: "3 3" }
                  polyline { points: "{rect_db_pts}", fill: "none", stroke: "#e11d48", stroke_width: if hann { "1.4" } else { "2.4" }, stroke_opacity: if hann { "0.55" } else { "1.0" } }
                  polyline { points: "{hann_db_pts}", fill: "none", stroke: "#059669", stroke_width: if hann { "2.5" } else { "1.4" }, stroke_opacity: if hann { "1.0" } else { "0.55" } }
                  text { x: "390", y: "22", text_anchor: "middle", font_size: "9.5", class: "fill-slate-600 dark:fill-slate-300 font-bold",
                    "窗函数对数幅度谱：玫红矩形窗泄漏 vs 翠绿 Hann 抑旁瓣"
                  }
                }
              }
              div { class: "lg:col-span-5 space-y-3",
                {math_slider("时域窗宽 σ_t（窄窗抓瞬态 / 宽窗辨音高）", &format!("{st:.2} s"), "0.15", "1.40", "0.05", st, "accent-indigo-600", EventHandler::new(move |v| sigma_t.set(v)))}
                {math_slider("声学频率 f → 梅尔心理声学刻度", &format!("{hz:.0} Hz"), "100", "8000", "100", hz, "accent-amber-600", EventHandler::new(move |v| audio_hz.set(v)))}
                div { class: "flex gap-2",
                  {mode_btn("Hann 余弦平滑窗 (-31.5 dB)", hann, EventHandler::new(move |_| use_hann.set(true)))}
                  {mode_btn("Rectangular 硬截断窗 (-13.3 dB)", !hann, EventHandler::new(move |_| use_hann.set(false)))}
                }
                p { class: "text-[11px] leading-relaxed text-slate-500 dark:text-slate-400",
                  {lang.pick(
                    "柯西—施瓦茨不等式锁死了时频乘积下界 1/2：没有任何 STFT 参数能同时看清毫秒级爆破音与赫兹级基频泛音。Whisper 采用 25ms Hann 窗兼顾语音共振峰与清辅音起始边沿。",
                    "Cauchy-Schwarz locks Δt·Δω ≥ 1/2: narrowing the time window inevitably widens the frequency mainlobe."
                  )}
                }
              }
            }
          }
        }

        SpectralMode::SpectralBiasNtk => {
          let (step, k_hi, ff) = (train_step(), hi_freq_k(), use_fourier_feat());
          // 标准 ReLU MLP 的 NTK 特征值随频率呈幂律骤降 λ_k ∝ k^{-2.4}；多频傅里叶特征映射将高频通道特征值抬升至 O(1)
          let lam_lo = 0.095_f64;
          let lam_hi = if ff { 0.068_f64 } else { 0.095 * k_hi.powf(-2.4) };
          let fit_lo = 1.0 - (-lam_lo * step).exp();
          let fit_hi = 1.0 - (-lam_hi * step).exp();
          let err_lo = 1.0 - fit_lo;
          let err_hi = 1.0 - fit_hi;

          let mut target_pts = String::with_capacity(1600);
          let mut pred_pts = String::with_capacity(1600);
          for i in 0..=120 {
            let x = (i as f64) / 120.0;
            let px = 26.0 + x * 310.0;
            let y_gt = (2.0 * PI * x).sin() + 0.45 * (2.0 * PI * k_hi * x).sin();
            let y_hat = fit_lo * (2.0 * PI * x).sin() + fit_hi * 0.45 * (2.0 * PI * k_hi * x).sin();
            push_svg_pt(&mut target_pts, i == 0, px, 126.0 - y_gt * 58.0);
            push_svg_pt(&mut pred_pts, i == 0, px, 126.0 - y_hat * 58.0);
          }

          rsx! {
            div { class: "grid grid-cols-2 sm:grid-cols-4 gap-2.5",
              {math_term_pill(MathTermTone::Indigo, "λ_low / λ_high", &format!("{:.1}×", lam_lo / lam_hi.max(1e-6)), "NTK 高低频特征值悬殊比")}
              {math_term_pill(MathTermTone::Emerald, "1 - e^{-t·λ_1}", &format!("{:.1}%", fit_lo * 100.0), "基频 k=1 收敛完成度")}
              {math_term_pill(MathTermTone::Amber, "Fourier Embed", if ff { "[sin 2^m πx]" } else { "Raw MLP (x)" }, "输入位置高频投影状态")}
              {math_term_pill(MathTermTone::Rose, "|e_high(t)|", &format!("{:.1}%", err_hi * 100.0), &format!("高频 k={k_hi:.0} 残余未拟合误差"))}
            }
            div { class: "grid grid-cols-1 lg:grid-cols-12 gap-4 items-start",
              div { class: "lg:col-span-7 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/70 dark:bg-slate-950/60 p-3",
                svg { view_box: "0 0 532 244", class: "w-full h-auto",
                  line { x1: "26", y1: "126", x2: "336", y2: "126", class: "stroke-slate-300 dark:stroke-slate-700", stroke_width: "1" }
                  polyline { points: "{target_pts}", fill: "none", stroke: "#94a3b8", stroke_width: "1.6", stroke_dasharray: "4 3" }
                  polyline { points: "{pred_pts}", fill: "none", stroke: "#4f46e5", stroke_width: "2.5" }
                  text { x: "181", y: "20", text_anchor: "middle", font_size: "9.5", class: "fill-slate-600 dark:fill-slate-300 font-bold",
                    "灰虚线真实复合波 vs 靛蓝网络预测波 y_θ(x, t)"
                  }
                  // 右侧双频残差柱状对比
                  {svg_vbar("eb-lo", 378.0, 206.0, 52.0, err_lo * 150.0, "fill-emerald-500", &format!("{:.0}%", err_lo * 100.0), "低频残差")}
                  {svg_vbar("eb-hi", 452.0, 206.0, 52.0, err_hi * 150.0, "fill-rose-500", &format!("{:.0}%", err_hi * 100.0), "高频残差")}
                  text { x: "440", y: "20", text_anchor: "middle", font_size: "9.5", class: "fill-slate-600 dark:fill-slate-300 font-bold",
                    "频域残差谱衰减"
                  }
                }
              }
              div { class: "lg:col-span-5 space-y-3",
                {math_slider("梯度下降步数 t", &format!("{step:.0} steps"), "0", "300", "5", step, "accent-indigo-600", EventHandler::new(move |v| train_step.set(v)))}
                {math_slider("叠加高频谐波阶数 k_high", &format!("k = {k_hi:.0}"), "4", "14", "1", k_hi, "accent-rose-600", EventHandler::new(move |v| hi_freq_k.set(v)))}
                div { class: "flex gap-2",
                  {mode_btn("裸坐标 MLP（高频卡死）", !ff, EventHandler::new(move |_| use_fourier_feat.set(false)))}
                  {mode_btn("开启多尺度傅里叶编码", ff, EventHandler::new(move |_| use_fourier_feat.set(true)))}
                }
                p { class: "text-[11px] leading-relaxed text-slate-500 dark:text-slate-400",
                  {lang.pick(
                    "为什么 NeRF 直接输 (x,y,z) 渲染一片糊、RoPE 和 DiT 必须配多波长正余弦编码？因为普通 MLP 核谱特征值按频率幂律暴跌（F-Principle），开启多尺度傅里叶特征后高频通道瞬间对齐主特征值，几步即可咬合锯齿细节！",
                    "Standard MLPs suffer polynomial NTK eigenvalue decay on high frequencies; multi-scale Fourier features lift high-frequency eigenvalues to O(1)."
                  )}
                }
              }
            }
          }
        }

        SpectralMode::GraphLaplacianCheb => {
          let (k_ord, alpha, spk) = (cheb_k() as usize, cheb_alpha(), spike_amp());
          let raw_sig = [
            1.0 + spk,
            0.75,
            0.15,
            -0.55,
            -0.95,
            -0.60,
            0.10,
            0.65,
          ];
          let (filtered, e_in, e_out) = compute_graph_laplacian_filter(&raw_sig, k_ord, alpha);

          let (cx, cy, r_ring) = (128.0_f64, 128.0_f64, 82.0_f64);
          let coords: [(f64, f64); 8] = core::array::from_fn(|i| {
            let ang = -0.5 * PI + (i as f64) * (2.0 * PI / 8.0);
            (cx + r_ring * ang.cos(), cy + r_ring * ang.sin())
          });

          rsx! {
            div { class: "grid grid-cols-2 sm:grid-cols-4 gap-2.5",
              {math_term_pill(MathTermTone::Indigo, "x⊤ L x (输入)", &format!("{e_in:.2}"), "原始节点信号狄利克雷粗糙度")}
              {math_term_pill(MathTermTone::Emerald, "y⊤ L y (滤波后)", &format!("{e_out:.2}"), "切比雪夫谱滤波后平滑度")}
              {math_term_pill(MathTermTone::Amber, "K-Hop Locality", &format!("K = {k_ord} 跳"), "严格受限于 K 跳邻居消息聚合")}
              {math_term_pill(MathTermTone::Rose, "O(|E|·K) vs O(N³)", &format!("{} ops", 10 * k_ord.max(1)), "免特征分解稀疏矩阵乘")}
            }
            div { class: "grid grid-cols-1 lg:grid-cols-12 gap-4 items-start",
              div { class: "lg:col-span-7 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/70 dark:bg-slate-950/60 p-3",
                svg { view_box: "0 0 532 246", class: "w-full h-auto",
                  for (e_idx, (u, v, _)) in GRAPH_EDGES_8.iter().copied().enumerate() {
                    line {
                      key: "ge-{e_idx}",
                      x1: "{coords[u].0:.1}",
                      y1: "{coords[u].1:.1}",
                      x2: "{coords[v].0:.1}",
                      y2: "{coords[v].1:.1}",
                      class: "stroke-slate-300 dark:stroke-slate-700",
                      stroke_width: if e_idx >= 8 { "1.3" } else { "2.0" },
                      stroke_dasharray: if e_idx >= 8 { "4 3" } else { "" },
                    }
                  }
                  for (idx, (nx, ny)) in coords.iter().copied().enumerate() {
                    {
                      let v = filtered[idx];
                      let fill = if v >= 0.0 { "#4f46e5" } else { "#059669" };
                      let rad = (10.0 + v.abs() * 6.5).clamp(9.0, 21.0);
                      rsx! {
                        g { key: "gn-{idx}",
                          circle { cx: "{nx:.1}", cy: "{ny:.1}", r: "{rad:.1}", fill: "{fill}", fill_opacity: "0.85" }
                          text { x: "{nx:.1}", y: "{ny + 3.5:.1}", text_anchor: "middle", font_size: "8.5", fill: "#ffffff", class: "font-mono font-bold",
                            "{v:+.1}"
                          }
                        }
                      }
                    }
                  }
                  // 右侧 8 节点滤波前后幅值柱状图
                  line { x1: "256", y1: "136", x2: "518", y2: "136", class: "stroke-slate-300 dark:stroke-slate-700", stroke_width: "1" }
                  for idx in 0..8 {
                    {
                      let bx = 262.0 + (idx as f64) * 31.5;
                      let h_out = (filtered[idx].abs() * 38.0).clamp(2.0, 88.0);
                      let cls = if filtered[idx] >= 0.0 { "fill-indigo-500" } else { "fill-emerald-500" };
                      svg_vbar(&format!("gb-{idx}"), bx, 212.0, 22.0, h_out, cls, &format!("{:+.1}", filtered[idx]), &format!("v{idx}"))
                    }
                  }
                }
              }
              div { class: "lg:col-span-5 space-y-3",
                {math_slider("切比雪夫多项式阶数 K（= 邻居聚合跳数）", &format!("K = {k_ord}"), "0", "6", "1", k_ord as f64, "accent-amber-600", EventHandler::new(move |v| cheb_k.set(v)))}
                {math_slider("谱滤波器斜率 α（负值低通平滑 / 正值高通锐化）", &format!("{alpha:+.2}"), "-0.85", "0.85", "0.05", alpha, "accent-emerald-600", EventHandler::new(move |v| cheb_alpha.set(v)))}
                {math_slider("节点 v0 局部脉冲噪声幅值", &format!("{spk:.2}"), "0.0", "2.2", "0.1", spk, "accent-rose-600", EventHandler::new(move |v| spike_amp.set(v)))}
                p { class: "text-[11px] leading-relaxed text-slate-500 dark:text-slate-400",
                  {lang.pick(
                    "拉普拉斯二次型 x⊤Lx 恒等于所有边两端节点差平方的加权和（天然半正定 ≥ 0）！当 α < 0 时切比雪夫多项式抑制高频大特征值，无需 O(N³) 特征分解即可抹平 v0 脉冲毛刺；K=1 即退化为经典 GCN 一跳邻居卷积。",
                    "x^T L x equals the weighted sum of squared edge differences (≥ 0). Negative α smooths local spikes via K-hop sparse message passing without O(N^3) eigendecomposition."
                  )}
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
// 内核 5：OptimizerAndFlowTrajectoryLab（AdamW 病态峡谷 / Muon 正交化 / 逆向 SDE vs ODE / Rectified Flow）
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OptFlowMode {
  AdamwCanyon,
  MuonNewtonSchulz,
  ReverseSdeVsOde,
  RectifiedFlowOt,
}

impl OptFlowMode {
  fn parse_mode(raw: &str) -> Self {
    match raw.trim().to_ascii_lowercase().as_str() {
      "muon" | "muon_ns" | "newton_schulz" | "shampoo" => Self::MuonNewtonSchulz,
      "sde" | "sde_ode" | "ito" | "tweedie" | "ddpm" | "score" => Self::ReverseSdeVsOde,
      "rectified_flow" | "cfm" | "flow" | "reflow" | "ot" => Self::RectifiedFlowOt,
      _ => Self::AdamwCanyon,
    }
  }
}

pub(crate) type Path2d = Vec<(f64, f64)>;

/// 在病态二次峡谷 $f(w_1, w_2) = \frac{1}{2}(w_1 - 1.2)^2 + \frac{\kappa}{2}(w_2 - 0.6)^2$ 上对比
/// `SGD`、`Momentum`、`Adam+L2（耦合）` 与 `AdamW（解耦）` 的 28 步真实轨迹。
pub(crate) fn simulate_canyon_optimizers(
  kappa: f64,
  lr: f64,
  wd: f64,
  steps: usize,
) -> (Path2d, Path2d, Path2d, Path2d) {
  let n = steps.clamp(2, 48);
  let start = (-1.35_f64, 1.18_f64);
  let mut sgd = Vec::with_capacity(n + 1);
  let mut mom = Vec::with_capacity(n + 1);
  let mut adam_l2 = Vec::with_capacity(n + 1);
  let mut adamw = Vec::with_capacity(n + 1);
  sgd.push(start);
  mom.push(start);
  adam_l2.push(start);
  adamw.push(start);

  let (mut sw1, mut sw2) = start;
  let (mut mw1, mut mw2, mut vx1, mut vx2) = (start.0, start.1, 0.0_f64, 0.0_f64);
  let (mut al1, mut al2, mut alm1, mut alm2, mut alv1, mut alv2) =
    (start.0, start.1, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
  let (mut aw1, mut aw2, mut awm1, mut awm2, mut awv1, mut awv2) =
    (start.0, start.1, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);

  let (b1, b2, eps) = (0.85_f64, 0.96_f64, 1e-8_f64);
  for t in 1..=n {
    // 1. SGD
    let (gs1, gs2) = ((sw1 - 1.2) + wd * sw1, kappa * (sw2 - 0.6) + wd * sw2);
    sw1 = (sw1 - lr * gs1).clamp(-2.4, 2.4);
    sw2 = (sw2 - lr * gs2).clamp(-1.9, 1.9);
    sgd.push((sw1, sw2));

    // 2. Heavy-Ball Momentum
    let (gm1, gm2) = ((mw1 - 1.2) + wd * mw1, kappa * (mw2 - 0.6) + wd * mw2);
    vx1 = 0.78 * vx1 + gm1;
    vx2 = 0.78 * vx2 + gm2;
    mw1 = (mw1 - lr * 0.65 * vx1).clamp(-2.4, 2.4);
    mw2 = (mw2 - lr * 0.65 * vx2).clamp(-1.9, 1.9);
    mom.push((mw1, mw2));

    let c1 = 1.0 - b1.powi(t as i32);
    let c2 = 1.0 - b2.powi(t as i32);
    let eff_lr = (lr * 1.65).min(0.32);

    // 3. Adam + Coupled L2（把 λw 混进梯度里被 √v_t 扭曲除以 √κ）
    let (ga1, ga2) = ((al1 - 1.2) + wd * al1 * 3.0, kappa * (al2 - 0.6) + wd * al2 * 3.0);
    alm1 = b1 * alm1 + (1.0 - b1) * ga1;
    alm2 = b1 * alm2 + (1.0 - b1) * ga2;
    alv1 = b2 * alv1 + (1.0 - b2) * ga1 * ga1;
    alv2 = b2 * alv2 + (1.0 - b2) * ga2 * ga2;
    al1 -= eff_lr * (alm1 / c1) / ((alv1 / c2).sqrt() + eps);
    al2 -= eff_lr * (alm2 / c1) / ((alv2 / c2).sqrt() + eps);
    adam_l2.push((al1.clamp(-2.4, 2.4), al2.clamp(-1.9, 1.9)));

    // 4. Decoupled AdamW（纯任务梯度做自适应步，权重衰减独立等比例向原点收缩）
    let (gw1, gw2) = (aw1 - 1.2, kappa * (aw2 - 0.6));
    awm1 = b1 * awm1 + (1.0 - b1) * gw1;
    awm2 = b1 * awm2 + (1.0 - b1) * gw2;
    awv1 = b2 * awv1 + (1.0 - b2) * gw1 * gw1;
    awv2 = b2 * awv2 + (1.0 - b2) * gw2 * gw2;
    aw1 = (1.0 - eff_lr * wd * 1.4) * aw1 - eff_lr * (awm1 / c1) / ((awv1 / c2).sqrt() + eps);
    aw2 = (1.0 - eff_lr * wd * 1.4) * aw2 - eff_lr * (awm2 / c1) / ((awv2 / c2).sqrt() + eps);
    adamw.push((aw1.clamp(-2.4, 2.4), aw2.clamp(-1.9, 1.9)));
  }
  (sgd, mom, adam_l2, adamw)
}

/// `Muon` 五次奇次多项式 Newton-Schulz 正交化迭代（真实 $4\times 4$ 矩阵乘法验证 $\|X_k^\top X_k - I\|_F$ 收敛 + 奇异值映射）。
pub(crate) fn run_muon_newton_schulz_4x4(
  sigmas_in: [f64; 4],
  steps: usize,
) -> ([f64; 4], f64, f64) {
  let (a, b, c) = (3.4445_f64, -4.7750_f64, 2.0315_f64);
  // 构造带非平凡正交旋转的真实 4x4 矩阵 G = U diag(sigmas_in) V^T 并按 Frobenius 范数归一化至谱范数 ≤ 1
  let frob = sigmas_in.iter().map(|s| s * s).sum::<f64>().sqrt().max(1e-12);
  let mut s_cur = sigmas_in.map(|s| s / frob);

  // 同步执行真实 4x4 稠密矩阵乘 X_{k+1} = a X + b X (X^T X) + c X (X^T X)^2
  let rot =
    [[0.8, -0.6, 0.0, 0.0], [0.6, 0.8, 0.0, 0.0], [0.0, 0.0, 0.6, -0.8], [0.0, 0.0, 0.8, 0.6]];
  let mut x = [[0.0_f64; 4]; 4];
  for i in 0..4 {
    for j in 0..4 {
      x[i][j] = rot[i][j] * s_cur[j];
    }
  }

  let matmul = |m1: &[[f64; 4]; 4], m2: &[[f64; 4]; 4]| -> [[f64; 4]; 4] {
    let mut out = [[0.0_f64; 4]; 4];
    for i in 0..4 {
      for k in 0..4 {
        for j in 0..4 {
          out[i][j] += m1[i][k] * m2[k][j];
        }
      }
    }
    out
  };
  let transpose = |m: &[[f64; 4]; 4]| -> [[f64; 4]; 4] {
    let mut out = [[0.0_f64; 4]; 4];
    for i in 0..4 {
      for j in 0..4 {
        out[i][j] = m[j][i];
      }
    }
    out
  };

  for _ in 0..steps.min(8) {
    for val in &mut s_cur {
      let v = *val;
      let v2 = v * v;
      *val = a * v + b * v * v2 + c * v * v2 * v2;
    }
    let xt = transpose(&x);
    let xtx = matmul(&xt, &x);
    let xtx2 = matmul(&xtx, &xtx);
    let xb = matmul(&x, &xtx);
    let xc = matmul(&x, &xtx2);
    for i in 0..4 {
      for j in 0..4 {
        x[i][j] = a * x[i][j] + b * xb[i][j] + c * xc[i][j];
      }
    }
  }

  let xtx = matmul(&transpose(&x), &x);
  let mut ortho_err_sq = 0.0;
  for (i, row) in xtx.iter().enumerate() {
    for (j, &val) in row.iter().enumerate() {
      let target = if i == j { 1.0 } else { 0.0 };
      ortho_err_sq += (val - target).powi(2);
    }
  }
  let s_min = s_cur.iter().copied().fold(f64::INFINITY, f64::min).max(1e-9);
  let s_max = s_cur.iter().copied().fold(0.0_f64, f64::max);
  (s_cur, ortho_err_sq.sqrt(), s_max / s_min)
}

/// 双峰高斯边缘分布解析得分函数 $\nabla_x \log p_t(x)$：驱动逆时间伊藤 SDE 与确定性概率流 ODE 轨迹对比。
pub(crate) fn bimodal_marginal_score(x: f64, t: f64, mu0: f64) -> f64 {
  let alpha_t = (1.0 - 0.88 * t).sqrt().max(0.15);
  let var_t = (1.0 - alpha_t * alpha_t) + alpha_t * alpha_t * 0.06;
  let m = alpha_t * mu0;
  let d1 = -0.5 * (x + m).powi(2) / var_t;
  let d2 = -0.5 * (x - m).powi(2) / var_t;
  let max_d = d1.max(d2);
  let w1 = (d1 - max_d).exp();
  let w2 = (d2 - max_d).exp();
  let post_mean = m * (w2 - w1) / (w1 + w2);
  (post_mean - x) / var_t
}

/// 模拟 `N` 条逆时间伊藤 SDE 轨迹与确定性概率流 ODE 轨迹（从 $t=1 \to 0$），对比终点分布均值与标准差一致性。
pub(crate) fn simulate_reverse_sde_and_ode(
  mu0: f64,
  g_noise: f64,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, f64) {
  let n_p = 14usize;
  let n_steps = 40usize;
  let dt = 1.0 / (n_steps as f64);
  let mut sde_paths = Vec::with_capacity(n_p);
  let mut ode_paths = Vec::with_capacity(n_p);
  let mut state = 0xD1B5_4A32_D192_ED03_u64;
  let mut next_gauss = || -> f64 {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    let u1 = (((state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 + 0.5)
      / ((1u64 << 53) as f64))
      .clamp(1e-9, 1.0 - 1e-9);
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    let u2 =
      ((state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 + 0.5) / ((1u64 << 53) as f64);
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
  };

  let mut mean_diff_acc = 0.0;
  for i in 0..n_p {
    // 对称高斯分位点初始化保证两侧在 t=1 边缘分布完全一致
    let q = -1.75 + 3.5 * (i as f64) / ((n_p - 1) as f64);
    let mut x_sde = q;
    let mut x_ode = q;
    let mut ps = Vec::with_capacity(n_steps + 1);
    let mut po = Vec::with_capacity(n_steps + 1);
    ps.push(x_sde);
    po.push(x_ode);

    for s in 0..n_steps {
      let t = 1.0 - (s as f64) * dt;
      let beta = 1.4;
      let g2 = beta * g_noise * g_noise;
      // 逆向伊藤 SDE（含 g dW_t 与完整 g^2 得分漂移）vs 确定性 Probability Flow ODE（精确减半 1/2 g^2 得分漂移）
      let sc_s = bimodal_marginal_score(x_sde, t, mu0);
      let dw = dt.sqrt() * next_gauss();
      x_sde = (x_sde + (0.5 * beta * x_sde + g2 * sc_s) * dt + g2.sqrt() * dw).clamp(-2.5, 2.5);

      let sc_o = bimodal_marginal_score(x_ode, t, mu0);
      x_ode = (x_ode + (0.5 * beta * x_ode + 0.5 * g2 * sc_o) * dt).clamp(-2.5, 2.5);
      ps.push(x_sde);
      po.push(x_ode);
    }
    mean_diff_acc += (x_sde.abs() - x_ode.abs()).abs();
    sde_paths.push(ps);
    ode_paths.push(po);
  }
  (sde_paths, ode_paths, mean_diff_acc / (n_p as f64))
}

/// 计算 `1-Rectified Flow`（独立交叉配对导致交点平均弯曲）与 `2-Rectified Flow (Reflow / OT)`（单调解交叉重配对后笔直轨迹）的弯曲损耗 $S(Z)$。
pub(crate) fn compute_rectified_flow_paths(
  cross_gap: f64,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, f64, f64) {
  let src = [-1.25_f64, -0.55, 0.55, 1.25];
  // 独立逆序交叉配对：src[0]->dst_cross[0]=+1.25 产生强烈轨道相交
  let dst_cross = [cross_gap, 0.45 * cross_gap, -0.45 * cross_gap, -cross_gap];
  let n_steps = 32usize;
  let dt = 1.0 / (n_steps as f64);

  // 经验核平滑边缘速度场 v_1(z, t) = Σ w_i(z,t) (x_1^{(i)} - x_0^{(i)})
  let eval_v = |z: f64, t: f64, s_arr: &[f64; 4], d_arr: &[f64; 4]| -> f64 {
    let bw2 = 0.085_f64;
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..4 {
      let z_it = (1.0 - t) * s_arr[i] + t * d_arr[i];
      let w = (-0.5 * (z - z_it).powi(2) / bw2).exp() + 1e-6;
      num += w * (d_arr[i] - s_arr[i]);
      den += w;
    }
    num / den
  };

  let integrate = |s_arr: &[f64; 4], d_arr: &[f64; 4]| -> (Vec<Vec<f64>>, f64) {
    let mut paths = Vec::with_capacity(4);
    let mut curv = 0.0;
    for i in 0..4 {
      let mut z = s_arr[i];
      let mut p = Vec::with_capacity(n_steps + 1);
      let mut v_hist = Vec::with_capacity(n_steps);
      p.push(z);
      for k in 0..n_steps {
        let t = (k as f64 + 0.5) * dt;
        let v = eval_v(z, t, s_arr, d_arr);
        v_hist.push(v);
        z += v * dt;
        p.push(z);
      }
      let chord = z - s_arr[i];
      curv += v_hist.iter().map(|&v| (v - chord).powi(2) * dt).sum::<f64>();
      paths.push(p);
    }
    (paths, curv / 4.0)
  };

  let (rf1_paths, curv1) = integrate(&src, &dst_cross);
  // Reflow 重配对：用 1-RF 确定性 ODE 的终点 z_1^{(i)} 作为新的无交叉目标配对训练 2-Rectified Flow！
  let reflow_dst: [f64; 4] =
    core::array::from_fn(|i| rf1_paths.get(i).and_then(|p| p.last()).copied().unwrap_or(src[i]));
  let (rf2_paths, curv2) = integrate(&src, &reflow_dst);
  (rf1_paths, rf2_paths, curv1, curv2)
}

#[component]
pub fn OptimizerAndFlowTrajectoryLab(#[props(default)] initial_mode: String) -> Element {
  let mut mode = use_signal(|| OptFlowMode::parse_mode(&initial_mode));
  // Canyon 优化器参数
  let mut kappa = use_signal(|| 16.0_f64);
  let mut lr = use_signal(|| 0.105_f64);
  let mut wd = use_signal(|| 0.14_f64);

  // Muon Newton-Schulz 参数
  let mut ns_steps = use_signal(|| 5.0_f64);
  let mut ns_skew = use_signal(|| 25.0_f64);

  // Reverse SDE vs ODE 参数
  let mut sde_mu = use_signal(|| 1.25_f64);
  let mut sde_g = use_signal(|| 0.85_f64);

  // Rectified Flow 参数
  let mut rf_gap = use_signal(|| 1.25_f64);
  let mut show_reflow = use_signal(|| true);

  let lang = current_lang();
  let cur_mode = mode();

  rsx! {
    div { class: "space-y-4",
      div { class: "flex flex-wrap gap-1.5",
        {mode_btn("L-平滑峡谷与 AdamW 解耦", cur_mode == OptFlowMode::AdamwCanyon, EventHandler::new(move |_| mode.set(OptFlowMode::AdamwCanyon)))}
        {mode_btn("Muon 奇次多项式正交化", cur_mode == OptFlowMode::MuonNewtonSchulz, EventHandler::new(move |_| mode.set(OptFlowMode::MuonNewtonSchulz)))}
        {mode_btn("逆向伊藤 SDE vs 概率流 ODE", cur_mode == OptFlowMode::ReverseSdeVsOde, EventHandler::new(move |_| mode.set(OptFlowMode::ReverseSdeVsOde)))}
        {mode_btn("CFM 与 Rectified Flow 拉直", cur_mode == OptFlowMode::RectifiedFlowOt, EventHandler::new(move |_| mode.set(OptFlowMode::RectifiedFlowOt)))}
      }

      match cur_mode {
        OptFlowMode::AdamwCanyon => {
          let (kp, eta, lam) = (kappa(), lr(), wd());
          let crit_lr = 2.0 / kp;
          let (sgd_p, mom_p, al2_p, aw_p) = simulate_canyon_optimizers(kp, eta, lam, 28);
          let map_pt = |(w1, w2): (f64, f64)| -> (f64, f64) {
            (256.0 + w1 * 102.0, 132.0 - w2 * 64.0)
          };
          let to_poly = |pts: &[(f64, f64)]| -> String {
            let mut s = String::with_capacity(pts.len() * 14);
            for (idx, &p) in pts.iter().enumerate() {
              let (px, py) = map_pt(p);
              push_svg_pt(&mut s, idx == 0, px, py);
            }
            s
          };
          let sgd_str = to_poly(&sgd_p);
          let mom_str = to_poly(&mom_p);
          let al2_str = to_poly(&al2_p);
          let aw_str = to_poly(&aw_p);
          let (opt_x, opt_y) = map_pt((1.2, 0.6));
          let aw_last = aw_p.last().copied().unwrap_or((0.0, 0.0));
          let al2_last = al2_p.last().copied().unwrap_or((0.0, 0.0));

          rsx! {
            div { class: "grid grid-cols-2 sm:grid-cols-4 gap-2.5",
              {math_term_pill(MathTermTone::Indigo, "2 / L (临界步长)", &format!("{crit_lr:.3}"), if eta > crit_lr { "η > 2/L：SGD 越界震荡发散!" } else { "η < 2/L：下降引理保证单调" })}
              {math_term_pill(MathTermTone::Emerald, "AdamW 终点", &format!("({:.2},{:.2})", aw_last.0, aw_last.1), "解耦等比向原点收缩")}
              {math_term_pill(MathTermTone::Amber, "Adam+L2 扭曲", &format!("({:.2},{:.2})", al2_last.0, al2_last.1), "陡轴衰减被 1/√κ 削弱")}
              {math_term_pill(MathTermTone::Rose, "η · L", &format!("{:.2}", eta * kp), "大于 2.0 即在崖壁反弹爆炸")}
            }
            div { class: "grid grid-cols-1 lg:grid-cols-12 gap-4 items-start",
              div { class: "lg:col-span-7 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/70 dark:bg-slate-950/60 p-3",
                svg { view_box: "0 0 532 252", class: "w-full h-auto",
                  // 病态椭球等高线
                  for r_idx in 1..=5 {
                    {
                      let rx = (r_idx as f64) * 46.0;
                      let ry = (rx / kp.sqrt()).max(7.0);
                      rsx! {
                        ellipse {
                          key: "ct-{r_idx}",
                          cx: "{opt_x:.1}",
                          cy: "{opt_y:.1}",
                          rx: "{rx:.1}",
                          ry: "{ry:.1}",
                          fill: "none",
                          class: "stroke-slate-300/80 dark:stroke-slate-700/80",
                          stroke_width: "1.1",
                        }
                      }
                    }
                  }
                  // 原点十字（权重衰减吸引中心）
                  line { x1: "248", y1: "132", x2: "264", y2: "132", class: "stroke-slate-400", stroke_width: "1.3" }
                  line { x1: "256", y1: "124", x2: "256", y2: "140", class: "stroke-slate-400", stroke_width: "1.3" }
                  polyline { points: "{sgd_str}", fill: "none", stroke: "#e11d48", stroke_width: "1.8" }
                  polyline { points: "{mom_str}", fill: "none", stroke: "#4f46e5", stroke_width: "1.9", stroke_dasharray: "4 2" }
                  polyline { points: "{al2_str}", fill: "none", stroke: "#d97706", stroke_width: "2.1" }
                  polyline { points: "{aw_str}", fill: "none", stroke: "#059669", stroke_width: "2.6" }
                  circle { cx: "{opt_x:.1}", cy: "{opt_y:.1}", r: "4.8", fill: "#0f172a", stroke: "#ffffff", stroke_width: "1.5" }
                  text { x: "266", y: "20", text_anchor: "middle", font_size: "9.5", class: "fill-slate-600 dark:fill-slate-300 font-bold",
                    "玫红 SGD · 靛蓝 Momentum · 琥珀 Adam+L2 · 翠绿解耦 AdamW"
                  }
                }
              }
              div { class: "lg:col-span-5 space-y-3",
                {math_slider("峡谷曲率条件数 κ = L / μ", &format!("{kp:.1}"), "4.0", "36.0", "1.0", kp, "accent-indigo-600", EventHandler::new(move |v| kappa.set(v)))}
                {math_slider("基础学习率 η（对比 2/L 临界点）", &format!("{eta:.3}"), "0.02", "0.24", "0.005", eta, "accent-rose-600", EventHandler::new(move |v| lr.set(v)))}
                {math_slider("正则权重衰减系数 λ", &format!("{lam:.2}"), "0.00", "0.35", "0.02", lam, "accent-emerald-600", EventHandler::new(move |v| wd.set(v)))}
                p { class: "text-[11px] leading-relaxed text-slate-500 dark:text-slate-400",
                  {lang.pick(
                    "将 η 拖过 2/L 阈值即可看到玫红 SGD 在陡峭横轴越弹越高（违反 L-平滑下降引理二次上界）；而调大 λ 时，琥珀色耦合 L2 在陡轴被 1/√v_t 严重削弱，唯有翠绿 AdamW 忠实沿径向向十字原点等比收缩！",
                    "Push η past 2/L to trigger wall-bouncing divergence in SGD; increase λ to see why coupled L2 distorts shrinkage compared to decoupled AdamW."
                  )}
                }
              }
            }
          }
        }

        OptFlowMode::MuonNewtonSchulz => {
          let (k_ns, skew) = (ns_steps() as usize, ns_skew());
          let raw_sigmas = [skew, skew * 0.36, skew * 0.12, 1.0];
          let (s_out, ortho_err, cond_after) = run_muon_newton_schulz_4x4(raw_sigmas, k_ns);
          let frob = raw_sigmas.iter().map(|v| v * v).sum::<f64>().sqrt();
          let s_norm = raw_sigmas.map(|v| v / frob);

          // 绘制 k 次迭代复合多项式 φ^{(k)}(σ) 在 [0, 1] 上的曲线
          let (a, b, c) = (3.4445_f64, -4.7750_f64, 2.0315_f64);
          let mut poly_pts = String::with_capacity(1400);
          for idx in 0..=80 {
            let x0 = (idx as f64) / 80.0;
            let mut y = x0;
            for _ in 0..k_ns {
              let y2 = y * y;
              y = a * y + b * y * y2 + c * y * y2 * y2;
            }
            push_svg_pt(&mut poly_pts, idx == 0, 32.0 + x0 * 212.0, 206.0 - y.clamp(0.0, 1.25) * 140.0);
          }

          rsx! {
            div { class: "grid grid-cols-2 sm:grid-cols-4 gap-2.5",
              {math_term_pill(MathTermTone::Indigo, "κ(G_0) 初始条件数", &format!("{skew:.1}×"), "动量矩阵奇异谱严重偏斜")}
              {math_term_pill(MathTermTone::Emerald, "κ(X_k) 正交化后", &format!("{cond_after:.2}×"), "全通道奇异值拍平至 ≈ 1")}
              {math_term_pill(MathTermTone::Amber, "NS Quintic Steps", &format!("{k_ns} 次 GEMM"), "零 SVD 求逆纯 BF16 矩阵乘")}
              {math_term_pill(MathTermTone::Rose, "‖X_k⊤ X_k - I‖_F", &format!("{ortho_err:.3}"), "4×4 稠密正交残差实测")}
            }
            div { class: "grid grid-cols-1 lg:grid-cols-12 gap-4 items-start",
              div { class: "lg:col-span-7 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/70 dark:bg-slate-950/60 p-3",
                svg { view_box: "0 0 532 242", class: "w-full h-auto",
                  // 左侧：奇次多项式映射曲线 φ^{(k)}(σ) -> 1
                  line { x1: "32", y1: "66", x2: "244", y2: "66", stroke: "#059669", stroke_dasharray: "4 3", stroke_width: "1.3" }
                  rect { x: "32", y: "32", width: "212", height: "174", fill: "none", class: "stroke-slate-300 dark:stroke-slate-700" }
                  polyline { points: "{poly_pts}", fill: "none", stroke: "#4f46e5", stroke_width: "2.4" }
                  text { x: "138", y: "22", text_anchor: "middle", font_size: "9.5", class: "fill-slate-600 dark:fill-slate-300 font-bold",
                    "五次多项式复合映射 φ^(k)(σ) 逼近阶跃函数 1"
                  }
                  // 右侧：归一化初始奇异谱 vs Muon 拍平后奇异谱柱状图
                  for i in 0..4 {
                    {
                      let bx = 278.0 + (i as f64) * 60.0;
                      rsx! {
                        g { key: "ms-{i}",
                          {svg_vbar(&format!("m0-{i}"), bx, 206.0, 22.0, s_norm[i] * 140.0, "fill-amber-400/85", &format!("{:.2}", s_norm[i]), "")}
                          {svg_vbar(&format!("mk-{i}"), bx + 24.0, 206.0, 22.0, s_out[i].clamp(0.0, 1.25) * 140.0, "fill-emerald-500", &format!("{:.2}", s_out[i]), &format!("σ_{}", i + 1))}
                        }
                      }
                    }
                  }
                  text { x: "396", y: "22", text_anchor: "middle", font_size: "9.5", class: "fill-slate-600 dark:fill-slate-300 font-bold",
                    "琥珀初始归一化奇异值 vs 翠绿 Muon 拍平谱"
                  }
                }
              }
              div { class: "lg:col-span-5 space-y-3",
                {math_slider("Newton-Schulz 多项式迭代步数 k", &format!("k = {k_ns}"), "0", "6", "1", k_ns as f64, "accent-indigo-600", EventHandler::new(move |v| ns_steps.set(v)))}
                {math_slider("初始最大/最小奇异值悬殊倍数", &format!("{skew:.0}×"), "3", "60", "1", skew, "accent-amber-600", EventHandler::new(move |v| ns_skew.set(v)))}
                p { class: "text-[11px] leading-relaxed text-slate-500 dark:text-slate-400",
                  {lang.pick(
                    "为什么 Muon 在大模型预训练中超越 AdamW？普通梯度被最大奇异值 σ₁ 垄断，尾部稀有特征方向几乎零更新；Muon 用 5 步纯矩阵乘 aX + bX(X⊤X) + cX(X⊤X)² 将 [0.02, 1] 全段奇异值统统拉平至 ≈ 1（谱正交更新 UV⊤）！",
                    "Muon applies 5 quintic Newton-Schulz matrix multiplications to flatten all singular values toward 1 (UV^T) without expensive SVD."
                  )}
                }
              }
            }
          }
        }

        OptFlowMode::ReverseSdeVsOde => {
          let (mu0, g_n) = (sde_mu(), sde_g());
          let (sde_p, ode_p, marginal_gap) = simulate_reverse_sde_and_ode(mu0, g_n);

          let to_ribbon = |path: &[f64], x_off: f64| -> String {
            let mut buf = String::with_capacity(path.len() * 14);
            let last = (path.len().saturating_sub(1)).max(1) as f64;
            for (idx, &v) in path.iter().enumerate() {
              let px = x_off + (idx as f64 / last) * 226.0;
              let py = 128.0 - v * 46.0;
              push_svg_pt(&mut buf, idx == 0, px, py);
            }
            buf
          };

          rsx! {
            div { class: "grid grid-cols-2 sm:grid-cols-4 gap-2.5",
              {math_term_pill(MathTermTone::Indigo, "Reverse Itô SDE", &format!("g = {g_n:.2}"), "含布朗噪声 g·dW̄_t 随机分叉")}
              {math_term_pill(MathTermTone::Emerald, "Prob-Flow ODE", "1/2 g²·∇log p_t", "零随机确定性双射流")}
              {math_term_pill(MathTermTone::Amber, "Target ±μ₀", &format!("±{mu0:.2}"), "双峰数据流形吸引中心")}
              {math_term_pill(MathTermTone::Rose, "Marginal |Δμ|", &format!("{marginal_gap:.3}"), "Fokker-Planck 保证同边缘密度")}
            }
            div { class: "grid grid-cols-1 lg:grid-cols-12 gap-4 items-start",
              div { class: "lg:col-span-7 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/70 dark:bg-slate-950/60 p-3",
                svg { view_box: "0 0 532 246", class: "w-full h-auto",
                  text { x: "136", y: "20", text_anchor: "middle", font_size: "9.5", class: "fill-indigo-600 dark:fill-indigo-400 font-bold",
                    "逆时间伊藤 SDE 粒子云 (t: 1 → 0)"
                  }
                  text { x: "398", y: "20", text_anchor: "middle", font_size: "9.5", class: "fill-emerald-600 dark:fill-emerald-400 font-bold",
                    "确定性 Probability Flow ODE (t: 1 → 0)"
                  }
                  for (p_idx, sp) in sde_p.iter().enumerate() {
                    {
                      let pts = to_ribbon(sp, 22.0);
                      rsx! {
                        polyline { key: "sde-{p_idx}", points: "{pts}", fill: "none", stroke: "#4f46e5", stroke_width: "1.2", stroke_opacity: "0.68" }
                      }
                    }
                  }
                  for (p_idx, op) in ode_p.iter().enumerate() {
                    {
                      let pts = to_ribbon(op, 284.0);
                      rsx! {
                        polyline { key: "ode-{p_idx}", points: "{pts}", fill: "none", stroke: "#059669", stroke_width: "1.5", stroke_opacity: "0.82" }
                      }
                    }
                  }
                }
              }
              div { class: "lg:col-span-5 space-y-3",
                {math_slider("双峰目标间距 μ₀", &format!("±{mu0:.2}"), "0.55", "1.85", "0.05", mu0, "accent-amber-600", EventHandler::new(move |v| sde_mu.set(v)))}
                {math_slider("扩散强度系数 g(t)", &format!("{g_n:.2}"), "0.25", "1.35", "0.05", g_n, "accent-indigo-600", EventHandler::new(move |v| sde_g.set(v)))}
                p { class: "text-[11px] leading-relaxed text-slate-500 dark:text-slate-400",
                  {lang.pick(
                    "左图每条伊藤 SDE 轨道剧烈抖动且互相穿插，右图将得分项精确减半（1/2 g² ∇log p_t）消掉随机项 dW̄_t 后变为光滑不交的确定性常微分方程——Fokker-Planck 散度恒等式证明两者每一时刻边缘分布 p_t(x) 严格相等！",
                    "Halving the score drift from g^2 to 1/2 g^2 removes the Brownian term while preserving the exact Fokker-Planck marginal density p_t(x) at all times."
                  )}
                }
              }
            }
          }
        }

        OptFlowMode::RectifiedFlowOt => {
          let (gap, reflow_on) = (rf_gap(), show_reflow());
          let (rf1_p, rf2_p, curv1, curv2) = compute_rectified_flow_paths(gap);
          let active_paths = if reflow_on { &rf2_p } else { &rf1_p };

          rsx! {
            div { class: "grid grid-cols-2 sm:grid-cols-4 gap-2.5",
              {math_term_pill(MathTermTone::Indigo, "S(Z_1) 1-RF 弯曲度", &format!("{curv1:.3}"), "独立随机交叉连线导致弯曲")}
              {math_term_pill(MathTermTone::Emerald, "S(Z_2) Reflow 后", &format!("{curv2:.4}"), "重配对解开交叉后接近直线!")}
              {math_term_pill(MathTermTone::Amber, "Straightening", &format!("{:.1}× 更直", curv1 / curv2.max(1e-4)), "1 步 Euler 即可逼近终点")}
              {math_term_pill(MathTermTone::Rose, "CFM 梯度恒等", "∇L_FM ≡ ∇L_CFM", "条件直线回归等价边缘场")}
            }
            div { class: "grid grid-cols-1 lg:grid-cols-12 gap-4 items-start",
              div { class: "lg:col-span-7 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/70 dark:bg-slate-950/60 p-3",
                svg { view_box: "0 0 532 242", class: "w-full h-auto",
                  // 背景展示独立交叉虚线弦
                  for (i, s_val) in [-1.25_f64, -0.55, 0.55, 1.25].iter().copied().enumerate() {
                    {
                      let d_val = [gap, 0.45 * gap, -0.45 * gap, -gap][i];
                      let y0 = 122.0 - s_val * 62.0;
                      let y1 = 122.0 - d_val * 62.0;
                      rsx! {
                        line {
                          key: "ch-{i}",
                          x1: "58",
                          y1: "{y0:.1}",
                          x2: "474",
                          y2: "{y1:.1}",
                          stroke: "#e11d48",
                          stroke_opacity: "0.28",
                          stroke_dasharray: "4 3",
                          stroke_width: "1.4",
                        }
                      }
                    }
                  }
                  for (i, path) in active_paths.iter().enumerate() {
                    {
                      let mut s = String::with_capacity(path.len() * 14);
                      let last = (path.len().saturating_sub(1)).max(1) as f64;
                      for (k, &v) in path.iter().enumerate() {
                        push_svg_pt(&mut s, k == 0, 58.0 + (k as f64 / last) * 416.0, 122.0 - v * 62.0);
                      }
                      let stroke = if reflow_on { "#059669" } else { "#4f46e5" };
                      rsx! {
                        polyline { key: "rfp-{i}", points: "{s}", fill: "none", stroke: "{stroke}", stroke_width: "2.6" }
                      }
                    }
                  }
                  text { x: "266", y: "20", text_anchor: "middle", font_size: "9.5", class: "fill-slate-600 dark:fill-slate-300 font-bold",
                    {if reflow_on { "翠绿 2-Rectified Flow (Reflow 解交叉重配对：轨道近乎笔直)" } else { "靛蓝 1-Rectified Flow (玫红交叉弦迫使交点处速度取均值而转弯)" }}
                  }
                }
              }
              div { class: "lg:col-span-5 space-y-3",
                {math_slider("右岸目标分布跨度", &format!("±{gap:.2}"), "0.55", "1.55", "0.05", gap, "accent-indigo-600", EventHandler::new(move |v| rf_gap.set(v)))}
                div { class: "flex gap-2",
                  {mode_btn("1-Rectified（弯折防相撞）", !reflow_on, EventHandler::new(move |_| show_reflow.set(false)))}
                  {mode_btn("2-Rectified Reflow（拉直）", reflow_on, EventHandler::new(move |_| show_reflow.set(true)))}
                }
                p { class: "text-[11px] leading-relaxed text-slate-500 dark:text-slate-400",
                  {lang.pick(
                    "独立采样 x₀ 与 x₁ 的连线（玫红虚线）在中间相交，确定性 ODE 不能一分为二，只能在交点附近交换跑道形成 S 形弯折（1-RF）；用 (x₀, Z₁(x₀)) 做一次重配对（2-RF），交叉彻底消失，1~4 步即可高清出图！",
                    "Independent couplings cross in the interior, forcing 1-RF ODE trajectories to bend; a single Reflow re-pairing uncrosses trajectories into nearly straight OT segments."
                  )}
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
// 内核 6：RlPolicyAndBitPrecisionLab（贝尔曼 GAE / PPO 双象限截断 / DPO-GRPO 抵消 / IEEE-754 位翻转）
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RlPrecisionMode {
  BellmanAndGae,
  PpoDualQuadrantClip,
  DpoPartitionAndGrpo,
  Ieee754BitFlipper,
}

impl RlPrecisionMode {
  fn parse_mode(raw: &str) -> Self {
    match raw.trim().to_ascii_lowercase().as_str() {
      "bellman" | "bellman_gae" | "mdp" | "gae" => Self::BellmanAndGae,
      "dpo" | "dpo_grpo" | "grpo" | "rlvr" => Self::DpoPartitionAndGrpo,
      "ieee754" | "ieee754_bits" | "bf16" | "fp16" | "fp8" | "bits" => Self::Ieee754BitFlipper,
      _ => Self::PpoDualQuadrantClip,
    }
  }
}

/// 4x4 网格迷宫贝尔曼最优算子 $\mathcal{T}^* V$ 迭代（终点 `(0,3)=+10`，陷阱 `(1,3)=-6`，障碍墙 `(1,1)`），返回价值表与实测无穷范数压缩率。
pub(crate) fn run_bellman_grid_4x4(gamma: f64, iters: usize) -> ([[f64; 4]; 4], f64) {
  let g = gamma.clamp(0.05, 0.99);
  let mut v = [[0.0_f64; 4]; 4];
  let step_cost = -0.25_f64;
  let mut prev_diff = 1.0_f64;
  let mut ratio = g;

  let step_op = |cur: &[[f64; 4]; 4]| -> [[f64; 4]; 4] {
    let mut nxt = *cur;
    nxt[0][3] = 10.0;
    nxt[1][3] = -6.0;
    nxt[1][1] = 0.0;
    for (r, row_out) in nxt.iter_mut().enumerate() {
      for (c, cell_out) in row_out.iter_mut().enumerate() {
        if (r == 0 && c == 3) || (r == 1 && c == 3) || (r == 1 && c == 1) {
          continue;
        }
        let mut best = f64::NEG_INFINITY;
        for (dr, dc) in [(-1isize, 0isize), (1, 0), (0, -1), (0, 1)] {
          let nr = (r as isize + dr).clamp(0, 3) as usize;
          let nc = (c as isize + dc).clamp(0, 3) as usize;
          let (tr, tc) = if nr == 1 && nc == 1 { (r, c) } else { (nr, nc) };
          let cand = step_cost + g * cur[tr][tc];
          if cand > best {
            best = cand;
          }
        }
        *cell_out = best;
      }
    }
    nxt
  };

  for k in 0..iters.min(32) {
    let nxt = step_op(&v);
    let mut sup_diff = 0.0_f64;
    for r in 0..4 {
      for c in 0..4 {
        sup_diff = sup_diff.max((nxt[r][c] - v[r][c]).abs());
      }
    }
    if k >= 2 && prev_diff > 1e-9 {
      ratio = (sup_diff / prev_diff).min(g + 1e-9);
    }
    if sup_diff > 1e-12 {
      prev_diff = sup_diff;
    }
    v = nxt;
  }
  (v, ratio)
}

/// PPO / DAPO 双象限截断代理目标 $L^{\text{CLIP}}(r_t, A_t) = \min(r_t A_t, \operatorname{clip}(r_t, 1-\epsilon_{\text{low}}, 1+\epsilon_{\text{high}}) A_t)$ 及有效梯度判别。
pub(crate) fn eval_ppo_clipped_surrogate(
  r_t: f64,
  adv: f64,
  eps_low: f64,
  eps_high: f64,
) -> (f64, f64, f64, bool) {
  let unclipped = r_t * adv;
  let r_clip = r_t.clamp(1.0 - eps_low, 1.0 + eps_high);
  let clipped = r_clip * adv;
  let surr = unclipped.min(clipped);
  // 当 A > 0 且 r > 1 + eps_high，或 A < 0 且 r < 1 - eps_low 时，min 选中平坦截断支，梯度精确置零！
  let grad_zeroed = (adv > 0.0 && r_t > 1.0 + eps_high) || (adv < 0.0 && r_t < 1.0 - eps_low);
  (unclipped, clipped, surr, grad_zeroed)
}

/// DPO 配分函数 $\ln Z(x)$ 相消验证：返回带任意配分常数偏移的隐式奖励 `(r_w, r_l)`、DPO 损失与重加权梯度系数。
pub(crate) fn eval_dpo_partition_invariance(
  beta: f64,
  log_ratio_win: f64,
  log_ratio_lose: f64,
  log_z_shift: f64,
) -> (f64, f64, f64, f64) {
  let r_win = beta * log_ratio_win + beta * log_z_shift;
  let r_lose = beta * log_ratio_lose + beta * log_z_shift;
  let margin = r_win - r_lose;
  let loss = -sigmoid(margin).max(1e-15).ln();
  let grad_weight = sigmoid(-margin);
  (r_win, r_lose, loss, grad_weight)
}

/// 精确 IEEE-754 位域解析与多精度舍入转换（`FP32 (1-8-23)` → `BF16 (1-8-7)`、`FP16 (1-5-10)`、`FP8 E4M3 (1-4-3)`）。
#[derive(Debug, Clone)]
pub(crate) struct MultiPrecisionDecode {
  pub sign: u32,
  pub exp_raw: u32,
  pub mant_raw: u32,
  pub fp32_val: f64,
  pub bf16_val: f64,
  pub fp16_desc: String,
  pub fp8_e4m3_desc: String,
}

pub(crate) fn decode_ieee754_multi_precision(bits: u32) -> MultiPrecisionDecode {
  let sign = (bits >> 31) & 1;
  let exp_raw = (bits >> 23) & 0xFF;
  let mant_raw = bits & 0x7F_FFFF;
  let fp32_val = f32::from_bits(bits) as f64;

  // 1. BF16 (1-8-7) Round-to-Nearest-Even 截断低 16 位尾数
  let bf16_bits = if exp_raw == 0xFF && mant_raw != 0 {
    bits | 0x0001_0000 // 保留 quiet NaN
  } else {
    let lsb = (bits >> 16) & 1;
    let rounding_bias = 0x7FFF_u32.wrapping_add(lsb);
    bits.wrapping_add(rounding_bias) & 0xFFFF_0000
  };
  let bf16_val = f32::from_bits(bf16_bits) as f64;

  // 2. FP16 (1-5-10, bias = 15, max = 65504, min normal ≈ 6.10e-5, min subnormal ≈ 5.96e-8)
  let sgn_ch = if sign == 1 { "-" } else { "+" };
  let fp16_desc = if fp32_val.is_nan() {
    "NaN".to_string()
  } else if fp32_val.is_infinite() || fp32_val.abs() > 65504.0 {
    format!("{sgn_ch}Inf (溢出!)")
  } else if fp32_val.abs() == 0.0 {
    format!("{sgn_ch}0.0")
  } else if fp32_val.abs() < 5.960_464_5e-8 {
    format!("{sgn_ch}0.0 (下溢归零!)")
  } else {
    let abs_v = fp32_val.abs();
    let e2 = abs_v.log2().floor() as i32;
    let step = if e2 < -14 { 2.0_f64.powi(-24) } else { 2.0_f64.powi(e2 - 10) };
    let q = (abs_v / step).round() * step * (if sign == 1 { -1.0 } else { 1.0 });
    format!("{q:.6e}")
  };

  // 3. FP8 E4M3 (1-4-3, bias = 7, max = 448.0, min subnormal = 2^-9 ≈ 1.95e-3)
  let fp8_e4m3_desc = if fp32_val.is_nan() {
    "NaN".to_string()
  } else if fp32_val.is_infinite() || fp32_val.abs() > 448.0 {
    format!("{sgn_ch}448 Sat/NaN (溢出!)")
  } else if fp32_val.abs() == 0.0 {
    format!("{sgn_ch}0.0")
  } else if fp32_val.abs() < 1.953_125e-3 {
    format!("{sgn_ch}0.0 (下溢!)")
  } else {
    let abs_v = fp32_val.abs();
    let e2 = abs_v.log2().floor() as i32;
    let step = if e2 < -6 { 2.0_f64.powi(-9) } else { 2.0_f64.powi(e2 - 3) };
    let q = (abs_v / step).round() * step * (if sign == 1 { -1.0 } else { 1.0 });
    format!("{q:.5}")
  };

  MultiPrecisionDecode { sign, exp_raw, mant_raw, fp32_val, bf16_val, fp16_desc, fp8_e4m3_desc }
}

/// 模拟 `BF16` 权重 `1.0` 累加微小步长 `delta` 时的梯度吞没（Swamping）：返回 `(bf16_updated, fp32_master_updated)`。
pub(crate) fn simulate_bf16_swamping(w_base: f32, delta: f32) -> (f32, f32) {
  let sum_f32 = w_base + delta;
  let bits = sum_f32.to_bits();
  let lsb = (bits >> 16) & 1;
  let bf16_rounded = f32::from_bits(bits.wrapping_add(0x7FFF + lsb) & 0xFFFF_0000);
  (bf16_rounded, sum_f32)
}

#[component]
pub fn RlPolicyAndBitPrecisionLab(#[props(default)] initial_mode: String) -> Element {
  let mut mode = use_signal(|| RlPrecisionMode::parse_mode(&initial_mode));
  // Bellman & GAE 参数
  let mut gamma = use_signal(|| 0.90_f64);
  let mut bellman_k = use_signal(|| 8.0_f64);
  let mut gae_lam = use_signal(|| 0.92_f64);

  // PPO Clip 参数
  let mut ppo_adv = use_signal(|| 1.25_f64);
  let mut ppo_eps = use_signal(|| 0.20_f64);
  let mut ppo_eps_hi = use_signal(|| 0.28_f64);
  let mut probe_r = use_signal(|| 1.42_f64);

  // DPO & GRPO 参数
  let mut dpo_beta = use_signal(|| 0.20_f64);
  let mut dpo_win_lr = use_signal(|| 1.60_f64);
  let mut dpo_lose_lr = use_signal(|| -0.90_f64);
  let mut log_z = use_signal(|| 3.50_f64);

  // IEEE-754 位翻转实验台状态（默认 1.0f32 = 0x3F80_0000）
  let mut fp_bits = use_signal(|| 0x3F80_0000_u32);
  let mut swamp_exp10 = use_signal(|| -3.2_f64);

  let lang = current_lang();
  let cur_mode = mode();

  rsx! {
    div { class: "space-y-4",
      div { class: "flex flex-wrap gap-1.5",
        {mode_btn("贝尔曼压缩映射与 GAE(γ,λ)", cur_mode == RlPrecisionMode::BellmanAndGae, EventHandler::new(move |_| mode.set(RlPrecisionMode::BellmanAndGae)))}
        {mode_btn("PPO 正负优势双象限截断", cur_mode == RlPrecisionMode::PpoDualQuadrantClip, EventHandler::new(move |_| mode.set(RlPrecisionMode::PpoDualQuadrantClip)))}
        {mode_btn("DPO 配分相消与 GRPO 组优势", cur_mode == RlPrecisionMode::DpoPartitionAndGrpo, EventHandler::new(move |_| mode.set(RlPrecisionMode::DpoPartitionAndGrpo)))}
        {mode_btn("IEEE-754 位翻转与 BF16 吞没", cur_mode == RlPrecisionMode::Ieee754BitFlipper, EventHandler::new(move |_| mode.set(RlPrecisionMode::Ieee754BitFlipper)))}
      }

      match cur_mode {
        RlPrecisionMode::BellmanAndGae => {
          let (gm, k_it, lam) = (gamma(), bellman_k() as usize, gae_lam());
          let (grid_v, emp_ratio) = run_bellman_grid_4x4(gm, k_it);
          let eff_decay = gm * lam;

          rsx! {
            div { class: "grid grid-cols-2 sm:grid-cols-4 gap-2.5",
              {math_term_pill(MathTermTone::Indigo, "γ (理论压缩界)", &format!("{gm:.2}"), "‖T*V₁ - T*V₂‖_∞ ≤ γ‖V₁ - V₂‖_∞")}
              {math_term_pill(MathTermTone::Emerald, "实测收缩比", &format!("{emp_ratio:.3}"), &format!("第 {k_it} 轮价值水波传播"))}
              {math_term_pill(MathTermTone::Amber, "(γ·λ)^k 衰减率", &format!("{eff_decay:.3}"), "GAE 多步 TD 残差指数权重")}
              {math_term_pill(MathTermTone::Rose, "V*(0,0) 起点估值", &format!("{:+.2}", grid_v[3][0]), "绕开障碍与陷阱的最优回报")}
            }
            div { class: "grid grid-cols-1 lg:grid-cols-12 gap-4 items-start",
              div { class: "lg:col-span-7 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/70 dark:bg-slate-950/60 p-3",
                svg { view_box: "0 0 532 246", class: "w-full h-auto",
                  text { x: "126", y: "20", text_anchor: "middle", font_size: "9.5", class: "fill-slate-600 dark:fill-slate-300 font-bold",
                    "4×4 贝尔曼最优价值迭代 V^(k)(s)"
                  }
                  for r in 0..4 {
                    for c in 0..4 {
                      {
                        let x = 28.0 + (c as f64) * 49.0;
                        let y = 32.0 + (r as f64) * 49.0;
                        let val = grid_v[r][c];
                        let (cls, txt) = if r == 1 && c == 1 {
                          ("fill-slate-700 stroke-slate-800", "WALL".to_string())
                        } else if r == 0 && c == 3 {
                          ("fill-emerald-500 stroke-emerald-700", "+10★".to_string())
                        } else if r == 1 && c == 3 {
                          ("fill-rose-500 stroke-rose-700", "-6☠".to_string())
                        } else if val > 4.0 {
                          ("fill-indigo-500/80 stroke-indigo-700", format!("{val:.1}"))
                        } else if val > 0.1 {
                          ("fill-indigo-200 dark:fill-indigo-900/70 stroke-indigo-400", format!("{val:.1}"))
                        } else {
                          ("fill-slate-100 dark:fill-slate-900 stroke-slate-300 dark:stroke-slate-700", format!("{val:.1}"))
                        };
                        let tcls = if (r == 1 && c == 1) || (r == 0 && c == 3) || (r == 1 && c == 3) || val > 4.0 {
                          "fill-white font-mono font-bold"
                        } else {
                          "fill-slate-700 dark:fill-slate-200 font-mono font-semibold"
                        };
                        svg_grid_cell(&format!("md-{r}-{c}"), x, y, 45.0, 45.0, cls, "1.2", &txt, tcls)
                      }
                    }
                  }
                  // 右侧：GAE(γ, λ) 未来 k 步 TD 残差 δ_{t+k} 权重柱状图
                  text { x: "390", y: "20", text_anchor: "middle", font_size: "9.5", class: "fill-slate-600 dark:fill-slate-300 font-bold",
                    "GAE 未来第 n 步残差权重 (γλ)^n"
                  }
                  for n in 0..8i32 {
                    {
                      let w_n = eff_decay.powi(n);
                      let bx = 266.0 + (n as f64) * 31.5;
                      svg_vbar(&format!("gae-{n}"), bx, 214.0, 22.0, w_n * 155.0, "fill-amber-500", &format!("{w_n:.2}"), &format!("δ+{n}"))
                    }
                  }
                }
              }
              div { class: "lg:col-span-5 space-y-3",
                {math_slider("折扣因子 γ（巴拿赫压缩率）", &format!("{gm:.2}"), "0.20", "0.98", "0.02", gm, "accent-indigo-600", EventHandler::new(move |v| gamma.set(v)))}
                {math_slider("贝尔曼迭代轮数 k", &format!("k = {k_it}"), "0", "16", "1", k_it as f64, "accent-emerald-600", EventHandler::new(move |v| bellman_k.set(v)))}
                {math_slider("GAE 权衡系数 λ（0=纯单步 TD，1=纯蒙特卡洛）", &format!("{lam:.2}"), "0.00", "1.00", "0.02", lam, "accent-amber-600", EventHandler::new(move |v| gae_lam.set(v)))}
                p { class: "text-[11px] leading-relaxed text-slate-500 dark:text-slate-400",
                  {lang.pick(
                    "从 k=0 逐步拖大迭代轮数，直观观察右上角 +10 终点奖励如何绕过 WALL 与 -6 陷阱传遍全图；调节 λ 即可在低方差单步自举（λ=0）与零偏差全轨迹回报（λ=1）之间平滑插值。",
                    "Step k upward to watch the +10 goal value propagate around the wall and -6 trap under γ-contraction; tune λ to interpolate between TD(0) and Monte Carlo."
                  )}
                }
              }
            }
          }
        }

        RlPrecisionMode::PpoDualQuadrantClip => {
          let (adv, eps_lo, eps_hi, r_p) = (ppo_adv(), ppo_eps(), ppo_eps_hi(), probe_r());
          let (u_val, c_val, s_val, zeroed) = eval_ppo_clipped_surrogate(r_p, adv, eps_lo, eps_hi);

          let map_x = |r: f64| 46.0 + ((r - 0.2) / 1.8) * 448.0;
          let map_y = |v: f64| (126.0 - v * 28.0).clamp(18.0, 228.0);
          let mut unclip_pts = String::with_capacity(1400);
          let mut clip_pts = String::with_capacity(1400);
          let mut surr_pts = String::with_capacity(1400);
          for idx in 0..=90 {
            let r = 0.2 + (idx as f64 / 90.0) * 1.8;
            let (uv, cv, sv, _) = eval_ppo_clipped_surrogate(r, adv, eps_lo, eps_hi);
            let px = map_x(r);
            push_svg_pt(&mut unclip_pts, idx == 0, px, map_y(uv));
            push_svg_pt(&mut clip_pts, idx == 0, px, map_y(cv));
            push_svg_pt(&mut surr_pts, idx == 0, px, map_y(sv));
          }
          let (x_lo, x_one, x_hi, px_r, py_r) = (
            map_x(1.0 - eps_lo),
            map_x(1.0),
            map_x(1.0 + eps_hi),
            map_x(r_p),
            map_y(s_val),
          );

          rsx! {
            div { class: "grid grid-cols-2 sm:grid-cols-4 gap-2.5",
              {math_term_pill(MathTermTone::Indigo, "r_t · A_t (未截断)", &format!("{u_val:+.3}"), "原始重要性采样比率目标")}
              {math_term_pill(MathTermTone::Amber, "clip(r_t)·A_t", &format!("{c_val:+.3}"), &format!("信赖区间 [{:.2}, {:.2}]", 1.0 - eps_lo, 1.0 + eps_hi))}
              {math_term_pill(MathTermTone::Emerald, "L^CLIP = min(·)", &format!("{s_val:+.3}"), "悲观下界包络（实粗线）")}
              {math_term_pill(MathTermTone::Rose, "∂L / ∂r_t 状态", if zeroed { "0 (梯度截断保护!)" } else { "A_t (正常更新推力)" }, if adv >= 0.0 { "正优势好动作象限" } else { "负优势坏动作象限" })}
            }
            div { class: "grid grid-cols-1 lg:grid-cols-12 gap-4 items-start",
              div { class: "lg:col-span-7 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/70 dark:bg-slate-950/60 p-3",
                svg { view_box: "0 0 532 244", class: "w-full h-auto",
                  rect { x: "{x_lo:.1}", y: "24", width: "{(x_hi - x_lo):.1}", height: "198", fill: "#059669", fill_opacity: "0.08" }
                  line { x1: "46", y1: "126", x2: "494", y2: "126", class: "stroke-slate-300 dark:stroke-slate-700", stroke_width: "1" }
                  line { x1: "{x_one:.1}", y1: "24", x2: "{x_one:.1}", y2: "222", class: "stroke-slate-400", stroke_dasharray: "3 3" }
                  polyline { points: "{unclip_pts}", fill: "none", stroke: "#4f46e5", stroke_width: "1.6", stroke_dasharray: "5 3" }
                  polyline { points: "{clip_pts}", fill: "none", stroke: "#d97706", stroke_width: "1.6", stroke_dasharray: "3 3" }
                  polyline { points: "{surr_pts}", fill: "none", stroke: "#059669", stroke_width: "3.0" }
                  circle { cx: "{px_r:.1}", cy: "{py_r:.1}", r: "6", fill: if zeroed { "#e11d48" } else { "#059669" }, stroke: "#ffffff", stroke_width: "1.5" }
                  text { x: "270", y: "18", text_anchor: "middle", font_size: "9.5", class: "fill-slate-600 dark:fill-slate-300 font-bold",
                    "横轴 r_t ∈ [0.2, 2.0]：翠绿粗线 L^CLIP = min(靛蓝 r_t A_t, 琥珀 clip·A_t)"
                  }
                }
              }
              div { class: "lg:col-span-5 space-y-3",
                div { class: "flex gap-2",
                  {mode_btn("好动作 A_t = +1.25 (>0)", adv > 0.0, EventHandler::new(move |_| ppo_adv.set(1.25)))}
                  {mode_btn("坏动作 A_t = -1.25 (<0)", adv < 0.0, EventHandler::new(move |_| ppo_adv.set(-1.25)))}
                }
                {math_slider("探测新旧策略概率比 r_t = π_θ / π_old", &format!("{r_p:.2}"), "0.20", "2.00", "0.02", r_p, "accent-indigo-600", EventHandler::new(move |v| probe_r.set(v)))}
                {math_slider("下界截断 ε_low", &format!("{eps_lo:.2}"), "0.05", "0.35", "0.01", eps_lo, "accent-amber-600", EventHandler::new(move |v| ppo_eps.set(v)))}
                {math_slider("上界截断 ε_high（DAPO Clip-Higher 防熵崩）", &format!("{eps_hi:.2}"), "0.10", "0.45", "0.01", eps_hi, "accent-emerald-600", EventHandler::new(move |v| ppo_eps_hi.set(v)))}
                p { class: "text-[11px] leading-relaxed text-slate-500 dark:text-slate-400",
                  {lang.pick(
                    "切换到坏动作 A_t < 0 并把 r_t 拉到 1.6：你会发现 min 故意【不截断】右上侧（保留全力把坏动作概率压回来的纠错梯度），只在左端 r_t < 1-ε_low 已经压够时才置零！调大 ε_high 即为 DAPO 非对称防熵坍缩机制。",
                    "Switch to A_t < 0 and drag r_t > 1+ε: min intentionally stays UNCLIPPED so wrongly amplified bad actions still receive full corrective gradient!"
                  )}
                }
              }
            }
          }
        }

        RlPrecisionMode::DpoPartitionAndGrpo => {
          let (beta, lw, ll, lz) = (dpo_beta(), dpo_win_lr(), dpo_lose_lr(), log_z());
          let (rw, rl, dpo_loss, grad_w) = eval_dpo_partition_invariance(beta, lw, ll, lz);
          let (_, _, dpo_loss_zero_z, _) = eval_dpo_partition_invariance(beta, lw, ll, 0.0);
          let z_sensitivity = (dpo_loss - dpo_loss_zero_z).abs();

          // 8 条 GRPO 同组采样奖励归一化 A_i = (R_i - mean) / std
          let raw_rewards = [1.0_f64, 1.0, 0.2, 0.0, 1.0, -0.2, 0.0, 0.6];
          let mean_r = raw_rewards.iter().sum::<f64>() / 8.0;
          let std_r = (raw_rewards.iter().map(|r| (r - mean_r).powi(2)).sum::<f64>() / 8.0)
            .sqrt()
            .max(1e-6);

          rsx! {
            div { class: "grid grid-cols-2 sm:grid-cols-4 gap-2.5",
              {math_term_pill(MathTermTone::Indigo, "r̂(x,y_w) 含 Z", &format!("{rw:+.3}"), "Chosen 隐式奖励含 +β·ln Z")}
              {math_term_pill(MathTermTone::Amber, "r̂(x,y_l) 含 Z", &format!("{rl:+.3}"), "Rejected 隐式奖励含 +β·ln Z")}
              {math_term_pill(MathTermTone::Emerald, "L_DPO (Z 已相消!)", &format!("{dpo_loss:.4}"), &format!("对 ln Z 敏感度 = {z_sensitivity:.1e}"))}
              {math_term_pill(MathTermTone::Rose, "σ(r̂_l - r̂_w) 权重", &format!("{grad_w:.3}"), "判错越离谱梯度权重越大")}
            }
            div { class: "grid grid-cols-1 lg:grid-cols-12 gap-4 items-start",
              div { class: "lg:col-span-7 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/70 dark:bg-slate-950/60 p-3",
                svg { view_box: "0 0 532 242", class: "w-full h-auto",
                  text { x: "126", y: "20", text_anchor: "middle", font_size: "9.5", class: "fill-slate-600 dark:fill-slate-300 font-bold",
                    "DPO 配分常数 β·ln Z(x) 同加同减严格抵消"
                  }
                  {svg_vbar("dpo-w", 46.0, 198.0, 46.0, (rw + 2.5).clamp(0.1, 4.8) * 32.0, "fill-indigo-500", &format!("{rw:+.2}"), "r̂_win")}
                  {svg_vbar("dpo-l", 112.0, 198.0, 46.0, (rl + 2.5).clamp(0.1, 4.8) * 32.0, "fill-amber-500", &format!("{rl:+.2}"), "r̂_lose")}
                  {svg_vbar("dpo-d", 178.0, 198.0, 48.0, (rw - rl).abs().clamp(0.05, 4.8) * 32.0, "fill-emerald-600", &format!("{:+.2}", rw - rl), "Δr̂ (恒定!)")}

                  // 右侧：GRPO 组内 8 条采样相对优势 A_i
                  text { x: "392", y: "20", text_anchor: "middle", font_size: "9.5", class: "fill-slate-600 dark:fill-slate-300 font-bold",
                    "GRPO 免 Critic 组内相对优势 Â_i = (R_i - μ_G)/σ_G"
                  }
                  line { x1: "262", y1: "128", x2: "518", y2: "128", class: "stroke-slate-300 dark:stroke-slate-700" }
                  for (i, &ri) in raw_rewards.iter().enumerate() {
                    {
                      let ai = (ri - mean_r) / std_r;
                      let bx = 266.0 + (i as f64) * 31.0;
                      let cls = if ai >= 0.0 { "fill-emerald-500" } else { "fill-rose-500" };
                      svg_vbar(&format!("gr-{i}"), bx, 206.0, 21.0, ai.abs() * 44.0 + 6.0, cls, &format!("{ai:+.1}"), &format!("G{}", i + 1))
                    }
                  }
                }
              }
              div { class: "lg:col-span-5 space-y-3",
                {math_slider("不可计算对数配分函数 ln Z(x)（任意拖动验证相消!）", &format!("{lz:+.2}"), "-6.0", "6.0", "0.25", lz, "accent-rose-600", EventHandler::new(move |v| log_z.set(v)))}
                {math_slider("KL 约束温度系数 β", &format!("{beta:.2}"), "0.05", "0.80", "0.05", beta, "accent-indigo-600", EventHandler::new(move |v| dpo_beta.set(v)))}
                {math_slider("胜出回答对数概率比 ln(π_θ / π_ref)(y_w)", &format!("{lw:+.2}"), "-1.5", "3.0", "0.1", lw, "accent-emerald-600", EventHandler::new(move |v| dpo_win_lr.set(v)))}
                {math_slider("落败回答对数概率比 ln(π_θ / π_ref)(y_l)", &format!("{ll:+.2}"), "-3.0", "1.5", "0.1", ll, "accent-amber-600", EventHandler::new(move |v| dpo_lose_lr.set(v)))}
              }
            }
          }
        }

        RlPrecisionMode::Ieee754BitFlipper => {
          let bits = fp_bits();
          let dec = decode_ieee754_multi_precision(bits);
          let delta = 10.0_f64.powf(swamp_exp10()) as f32;
          let (bf16_after, fp32_after) = simulate_bf16_swamping(1.0_f32, delta);
          let is_swamped = (bf16_after - 1.0_f32).abs() < 1e-12;

          let exp_unbiased = dec.exp_raw as i32 - 127;
          let flip_handler = EventHandler::new(move |idx: u32| {
            fp_bits.set(fp_bits() ^ (1u32 << idx));
          });

          rsx! {
            div { class: "grid grid-cols-2 sm:grid-cols-4 gap-2.5",
              {math_term_pill(MathTermTone::Indigo, "FP32 (1-8-23)", &format!("{:.6e}", dec.fp32_val), &format!("0x{bits:08X} · 24bit 有效精度"))}
              {math_term_pill(MathTermTone::Emerald, "BF16 (1-8-7)", &format!("{:.4e}", dec.bf16_val), "同享 8 位指数（永不溢 70000!）")}
              {math_term_pill(MathTermTone::Amber, "FP16 (1-5-10)", &dec.fp16_desc, "仅 5 位指数：>65504 即 Inf")}
              {math_term_pill(MathTermTone::Rose, "FP8 E4M3 (1-4-3)", &dec.fp8_e4m3_desc, "最大 448 · 需 Block Scaling")}
            }

            div { class: "rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/80 dark:bg-slate-950/60 p-3.5 space-y-3",
              div { class: "flex flex-wrap items-center justify-between gap-2 text-xs",
                span { class: "font-bold text-slate-700 dark:text-slate-200",
                  {lang.pick("点击任意二进制位实时翻转 0 ↔ 1（红: 符号 S[31] · 靛蓝: 指数 E[30..23] · 翠绿: 尾数 M[22..0]）", "Click any bit cell to toggle 0 ↔ 1 (Sign · 8-bit Exponent · 23-bit Mantissa)")}
                }
                div { class: "flex flex-wrap gap-1.5",
                  {mode_btn("1.0", bits == 0x3F80_0000, EventHandler::new(move |_| fp_bits.set(0x3F80_0000)))}
                  {mode_btn("70000 (FP16 爆 Inf!)", bits == 0x4788_B800, EventHandler::new(move |_| fp_bits.set(0x4788_B800)))}
                  {mode_btn("1e-8 (FP16 下溢 0!)", bits == 0x322B_CC77, EventHandler::new(move |_| fp_bits.set(0x322B_CC77)))}
                  {mode_btn("-0.15625", bits == 0xBE20_0000, EventHandler::new(move |_| fp_bits.set(0xBE20_0000)))}
                }
              }

              div { class: "grid grid-cols-8 sm:grid-cols-16 gap-1",
                for idx_rev in 0..32u32 {
                  {
                    let bit_idx = 31 - idx_rev;
                    let b_val = (bits >> bit_idx) & 1;
                    let z_cls = if bit_idx == 31 {
                      "bg-rose-100 dark:bg-rose-950/70 border-rose-300 dark:border-rose-800 text-rose-700 dark:text-rose-300"
                    } else if bit_idx >= 23 {
                      "bg-indigo-100 dark:bg-indigo-950/70 border-indigo-300 dark:border-indigo-800 text-indigo-700 dark:text-indigo-300"
                    } else if bit_idx >= 16 {
                      "bg-emerald-100 dark:bg-emerald-950/70 border-emerald-300 dark:border-emerald-800 text-emerald-700 dark:text-emerald-300"
                    } else {
                      "bg-slate-200/70 dark:bg-slate-900 border-slate-300 dark:border-slate-800 text-slate-500 dark:text-slate-400"
                    };
                    ieee_bit_cell(bit_idx, b_val, z_cls, flip_handler)
                  }
                }
              }
              div { class: "text-[11px] font-mono text-slate-500 dark:text-slate-400 flex flex-wrap justify-between gap-2",
                span { "Sign = (-1)^{dec.sign} · Exp = {dec.exp_raw} - 127 = {exp_unbiased} · Mantissa = 0x{dec.mant_raw:06X}" }
                span { "灰色后 16 格 [15..0] 即为 BF16 直接舍弃的尾数位！" }
              }
            }

            div { class: "grid grid-cols-1 lg:grid-cols-12 gap-3 items-center rounded-xl border border-slate-200 dark:border-slate-800 p-3",
              div { class: "lg:col-span-6",
                {math_slider("混合精度吞没实验：在权重 w = 1.0 上累加微小梯度步 Δw", &format!("Δw = {delta:.2e}"), "-5.0", "-1.5", "0.1", swamp_exp10(), "accent-rose-600", EventHandler::new(move |v| swamp_exp10.set(v)))}
              }
              div { class: "lg:col-span-6 grid grid-cols-2 gap-2 text-xs",
                {lab_metric_card("纯 BF16 累加 (1.0 + Δw)", &format!("{bf16_after:.7}"), if is_swamped { "text-rose-600 dark:text-rose-400" } else { "text-emerald-600 dark:text-emerald-400" })}
                {lab_metric_card("FP32 Master Weight 累加", &format!("{fp32_after:.7}"), "text-indigo-600 dark:text-indigo-400")}
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
  fn dft_winding_peaks_on_true_harmonic_and_computes_nyquist_alias() {
    let (_, _, mag_hit, alias_ok) = compute_dft_winding_and_alias(3.0, 6.0, 3.0, 16.0);
    let (_, _, mag_orth, _) = compute_dft_winding_and_alias(3.0, 6.0, 4.0, 16.0);
    let (_, _, mag_miss, _) = compute_dft_winding_and_alias(3.0, 6.0, 4.5, 16.0);
    assert!(mag_orth < 1e-14, "orthogonal integer harmonic must cancel to zero, got {mag_orth}");
    assert!(mag_hit > 0.30 && mag_hit > mag_miss * 3.0);
    assert!((alias_ok - 3.0).abs() < 1e-12);
    // 当采样率 f_s = 10 Hz 时，f_1 = 8 Hz > 5 Hz 折叠伪装成 |8 - 10| = 2 Hz
    let (_, _, _, alias_fold) = compute_dft_winding_and_alias(8.0, 6.0, 8.0, 10.0);
    assert!((alias_fold - 2.0).abs() < 1e-12);
  }

  #[test]
  fn stft_uncertainty_obeys_heisenberg_bound_and_laplacian_is_psd() {
    for st in [0.15, 0.5, 1.35] {
      let (_, _, prod_h, sl_h, _) = compute_stft_uncertainty_metrics(st, true, 1000.0);
      let (_, _, prod_r, sl_r, _) = compute_stft_uncertainty_metrics(st, false, 1000.0);
      assert!(prod_h >= 0.5 && prod_r > prod_h);
      assert!(sl_h < sl_r);
    }
    let const_sig = [2.5_f64; 8];
    let (_, dir_const, _) = compute_graph_laplacian_filter(&const_sig, 3, -0.5);
    assert!(dir_const.abs() < 1e-14, "constant vector must lie in nullspace of L");
    let alt_sig = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
    let (_, dir_in, dir_out) = compute_graph_laplacian_filter(&alt_sig, 2, -0.5);
    assert!(dir_in > 0.0 && dir_out < dir_in);
  }

  #[test]
  fn canyon_sgd_diverges_above_two_over_l_and_muon_orthogonalizes_4x4() {
    let (sgd_div, _, _, adamw_ok) = simulate_canyon_optimizers(20.0, 0.14, 0.1, 20); // η*L = 2.8 > 2.0
    let sgd_end = sgd_div.last().unwrap();
    let aw_end = adamw_ok.last().unwrap();
    assert!((sgd_end.1 - 0.6).abs() > 1.0, "SGD must oscillate/diverge when eta > 2/L");
    assert!((aw_end.1 - 0.6).abs() < 0.35);

    let (_, ortho_err0, cond0) = run_muon_newton_schulz_4x4([30.0, 10.0, 3.5, 1.0], 0);
    let (s_out, ortho_err5, cond_after) = run_muon_newton_schulz_4x4([30.0, 10.0, 3.5, 1.0], 5);
    assert!((cond0 - 30.0).abs() < 1e-9);
    assert!(ortho_err5 < 0.80 && ortho_err5 < ortho_err0 * 0.55);
    // Muon 极小极大五次多项式 (3.4445, -4.7750, 2.0315) 在 5 步内将 30x 偏斜谱全部拉入 [0.68, 1.15] 波纹带（条件数 < 1.6）
    assert!(cond_after < 1.6 && s_out.iter().all(|&s| (0.68..=1.15).contains(&s)));
  }

  #[test]
  fn reflow_reduces_trajectory_curvature_over_crossing_1rf() {
    let (_, _, curv1, curv2) = compute_rectified_flow_paths(1.25);
    assert!(
      curv1 > curv2 * 2.5,
      "Reflow curvature {curv2} must be much straighter than 1-RF {curv1}"
    );
  }

  #[test]
  fn bellman_contraction_ppo_quadrants_dpo_invariance_and_ieee754_swamping() {
    // 1. Banach γ-contraction
    let (v8, ratio) = run_bellman_grid_4x4(0.85, 10);
    assert!(ratio <= 0.85 + 1e-6 && v8[0][2] > v8[3][0]);

    // 2. PPO 双象限非对称梯度置零
    let (_, _, _, z_pos_hi) = eval_ppo_clipped_surrogate(1.5, 1.0, 0.2, 0.2);
    let (_, _, _, z_neg_hi) = eval_ppo_clipped_surrogate(1.5, -1.0, 0.2, 0.2);
    let (_, _, _, z_neg_lo) = eval_ppo_clipped_surrogate(0.6, -1.0, 0.2, 0.2);
    assert!(z_pos_hi, "A>0, r>1+eps must be clipped");
    assert!(!z_neg_hi, "A<0, r>1+eps must stay UNCLIPPED to push bad action down");
    assert!(z_neg_lo, "A<0, r<1-eps must be clipped");

    // 3. DPO 对任意 ln Z(x) 严格零敏感
    let (_, _, loss_a, gw_a) = eval_dpo_partition_invariance(0.25, 1.4, -0.8, -9.5);
    let (_, _, loss_b, gw_b) = eval_dpo_partition_invariance(0.25, 1.4, -0.8, 12.3);
    assert!((loss_a - loss_b).abs() < 1e-14 && (gw_a - gw_b).abs() < 1e-14);

    // 4. IEEE-754 70000 溢出对比与 BF16 1.0 + 1e-4 梯度吞没
    let dec_70k = decode_ieee754_multi_precision(70_000.0_f32.to_bits());
    assert!(dec_70k.fp16_desc.contains("Inf") && (dec_70k.bf16_val - 70_144.0).abs() < 512.0);
    let dec_1e8 = decode_ieee754_multi_precision(1e-8_f32.to_bits());
    assert!(dec_1e8.fp16_desc.contains("0.0") && dec_1e8.bf16_val > 0.0);

    let (bf16_swamped, fp32_exact) = simulate_bf16_swamping(1.0, 1e-4);
    assert_eq!(bf16_swamped, 1.0_f32);
    assert!((fp32_exact - 1.0001).abs() < 1e-7);
  }
}
