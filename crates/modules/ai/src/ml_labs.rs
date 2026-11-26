//! 纯 Rust WASM 传统机器学习 2D 交互式实验台（4 大数值求解内核 + `<MlLab />` MDX 组件）。
//!
//! 设计准则：
//! - **真算，不写伪动画**：CART Gini 贪心分裂、随机森林多树投票、GBDT/XGBoost 二阶泰勒残差提升、
//!   线性/岭回归高斯消元闭式解、逻辑回归梯度下降、SVM 线性间隔与 RBF 核边界、K-Means Lloyd 迭代、
//!   PCA 2×2 特征值分解与 KNN Voronoi 划分全部由纯 Rust `f64` 实时精确求解；
//! - **极致 WASM 体积收敛**：全部 SVG 图形收敛至 5 个 `#[inline(never)]` 轻量原语 + 单节点
//!   `dangerous_inner_html` 挂载，杜绝 `format!`/`write!` 重复展开，严格兼容无 `'unsafe-eval'` CSP。

use std::collections::HashMap;
use std::f64::consts::PI;
use std::fmt::Write as _;

use app_core::i18n::Language;
use dioxus::prelude::*;
use dioxus_shadcn::card_class;

use crate::component_labs::{current_lang, lab_metric_card, sigmoid};
use crate::component_labs_m2::mode_btn;

/// 传统机器学习四大可复用交互画布内核。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MlLabKind {
  #[default]
  TreeBoost,
  Boundary2d,
  RegressionFit,
  ClusterPca,
}

impl MlLabKind {
  pub fn parse_str(raw: &str) -> Self {
    match raw.trim().to_ascii_lowercase().as_str() {
      "tree_boost" | "tree" | "decision_tree" | "random_forest" | "gbdt" | "xgboost" => {
        Self::TreeBoost
      }
      "boundary_2d" | "boundary" | "logistic" | "svm" | "svm_rbf" | "knn" => Self::Boundary2d,
      "regression_fit" | "regression" | "linear" | "linear_ridge" | "ridge" => Self::RegressionFit,
      "cluster_pca" | "cluster" | "kmeans" | "pca" => Self::ClusterPca,
      _ => Self::TreeBoost,
    }
  }

  pub fn as_str(self) -> &'static str {
    match self {
      Self::TreeBoost => "tree_boost",
      Self::Boundary2d => "boundary_2d",
      Self::RegressionFit => "regression_fit",
      Self::ClusterPca => "cluster_pca",
    }
  }

  pub fn badge_label(self, lang: Language) -> &'static str {
    match self {
      Self::TreeBoost => lang.pick("树分裂与梯度提升交互台", "Tree & Boosting Lab"),
      Self::Boundary2d => lang.pick("2D 分类边界与核间隔画布", "2D Boundary & Margin Lab"),
      Self::RegressionFit => lang.pick("回归拟合与岭正则实验台", "Regression & Ridge Lab"),
      Self::ClusterPca => lang.pick("聚类迭代与主成分投影台", "Clustering & PCA Lab"),
    }
  }
}

/// 根据词条 `slug` 确定性推断默认画布内核与交互模式。
#[inline(never)]
pub fn infer_ml_lab_by_slug(
  slug: &str,
  kind_override: &str,
  mode_override: &str,
) -> (MlLabKind, String) {
  let s = slug.trim().to_ascii_lowercase();
  let (def_kind, def_mode) = if s.contains("xgboost") {
    (MlLabKind::TreeBoost, "xgboost")
  } else if s.contains("gbdt") || s.contains("boost") {
    (MlLabKind::TreeBoost, "gbdt")
  } else if s.contains("random-forest") || s.contains("forest") {
    (MlLabKind::TreeBoost, "random_forest")
  } else if s.contains("tree") {
    (MlLabKind::TreeBoost, "decision_tree")
  } else if s.contains("svm") {
    (MlLabKind::Boundary2d, "svm_rbf")
  } else if s.contains("knn") {
    (MlLabKind::Boundary2d, "knn")
  } else if s.contains("logistic") {
    (MlLabKind::Boundary2d, "logistic")
  } else if s.contains("linear") || s.contains("ridge") {
    (MlLabKind::RegressionFit, "linear_ridge")
  } else if s.contains("pca") {
    (MlLabKind::ClusterPca, "pca")
  } else if s.contains("kmeans") || s.contains("k-means") {
    (MlLabKind::ClusterPca, "kmeans")
  } else {
    (MlLabKind::TreeBoost, "decision_tree")
  };
  let kind =
    if kind_override.trim().is_empty() { def_kind } else { MlLabKind::parse_str(kind_override) };
  let mode = if mode_override.trim().is_empty() {
    def_mode.to_string()
  } else {
    mode_override.trim().to_ascii_lowercase()
  };
  (kind, mode)
}

// ─────────────────────────────────────────────────────────────────────────────
// 5 个极致精简 SVG 非内联原语（把全模块 write! 模板压缩至极小字节）
// ─────────────────────────────────────────────────────────────────────────────

#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn s_rect(
  out: &mut String,
  x: f64,
  y: f64,
  w: f64,
  h: f64,
  rx: f64,
  fill: &str,
  op: f64,
  stroke: &str,
  sw: f64,
) {
  let _ = write!(
    out,
    "<rect x=\"{x:.1}\" y=\"{y:.1}\" width=\"{w:.1}\" height=\"{h:.1}\" rx=\"{rx:.1}\" fill=\"{fill}\" fill-opacity=\"{op:.2}\""
  );
  if !stroke.is_empty() {
    let _ = write!(out, " stroke=\"{stroke}\" stroke-width=\"{sw:.1}\"");
  }
  out.push_str("/>");
}

#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn s_line(out: &mut String, x1: f64, y1: f64, x2: f64, y2: f64, stroke: &str, sw: f64, dash: &str) {
  let _ = write!(
    out,
    "<line x1=\"{x1:.1}\" y1=\"{y1:.1}\" x2=\"{x2:.1}\" y2=\"{y2:.1}\" stroke=\"{stroke}\" stroke-width=\"{sw:.1}\""
  );
  if !dash.is_empty() {
    let _ = write!(out, " stroke-dasharray=\"{dash}\"");
  }
  out.push_str("/>");
}

#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn s_circle(out: &mut String, cx: f64, cy: f64, r: f64, fill: &str, stroke: &str, sw: f64) {
  let _ = write!(
    out,
    "<circle cx=\"{cx:.1}\" cy=\"{cy:.1}\" r=\"{r:.1}\" fill=\"{fill}\" stroke=\"{stroke}\" stroke-width=\"{sw:.1}\"/>"
  );
}

#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn s_text(out: &mut String, x: f64, y: f64, sz: f64, fill: &str, bold: bool, mid: bool, txt: &str) {
  let w = if bold { " font-weight=\"700\"" } else { "" };
  let a = if mid { " text-anchor=\"middle\"" } else { "" };
  let _ = write!(
    out,
    "<text x=\"{x:.1}\" y=\"{y:.1}\" font-size=\"{sz:.1}\" fill=\"{fill}\"{w}{a}>{txt}</text>"
  );
}

#[inline(never)]
fn s_poly(out: &mut String, pts: &[(f64, f64)], stroke: &str, sw: f64, dash: &str) {
  out.push_str("<polyline points=\"");
  for (i, &(x, y)) in pts.iter().enumerate() {
    if i > 0 {
      out.push(' ');
    }
    let _ = write!(out, "{x:.1},{y:.1}");
  }
  let _ = write!(out, "\" fill=\"none\" stroke=\"{stroke}\" stroke-width=\"{sw:.1}\"");
  if !dash.is_empty() {
    let _ = write!(out, " stroke-dasharray=\"{dash}\"");
  }
  out.push_str("/>");
}

#[inline(never)]
fn s_panel(out: &mut String, x: f64, y: f64, w: f64, h: f64) {
  s_rect(out, x, y, w, h, 8.0, "#f8fafc", 1.0, "#cbd5e1", 1.0);
}

#[inline(never)]
fn fmt_n(pre: &str, v: f64, prec: usize, suf: &str) -> String {
  let mut s = String::with_capacity(pre.len() + suf.len() + 12);
  s.push_str(pre);
  let _ = write!(s, "{v:.prec$}");
  s.push_str(suf);
  s
}

#[inline(never)]
fn fmt_i(pre: &str, v: i64, suf: &str) -> String {
  let mut s = String::with_capacity(pre.len() + suf.len() + 10);
  s.push_str(pre);
  let _ = write!(s, "{v}");
  s.push_str(suf);
  s
}

#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn s_bars(
  out: &mut String,
  rx0: f64,
  rw: f64,
  base_y: f64,
  max_h: f64,
  vals: &[f64],
  prefix: char,
  last_col: &str,
) {
  let max_v = vals.first().copied().unwrap_or(1.0).max(0.05);
  let n = vals.len().max(1);
  let bw = ((rw - 28.0) / n as f64).min(28.0);
  let gap = if n > 1 { (rw - 28.0 - bw * n as f64) / (n - 1) as f64 } else { 0.0 };
  let mut p_buf = [0u8; 4];
  let p_str = prefix.encode_utf8(&mut p_buf);
  for (i, &v) in vals.iter().enumerate() {
    let bx = rx0 + 14.0 + i as f64 * (bw + gap);
    let bh = ((v / max_v) * max_h).max(4.0);
    let by = base_y - bh;
    let col = if i + 1 == n { last_col } else { "#6366f1" };
    s_rect(out, bx, by, bw, bh, 3.0, col, 0.85, "", 0.0);
    let cx = bx + bw * 0.5;
    s_text(out, cx, base_y + 12.0, 8.0, "#475569", false, true, &fmt_i(p_str, i as i64, ""));
    s_text(out, cx, (by - 3.0).max(48.0), 7.5, "#64748b", false, true, &fmt_n("", v, 2, ""));
  }
}

/// 统一 2D 分类样例点集（确定性，零随机依赖，SSR 与 WASM 逐像素一致）。
const CLS_LINEAR: &[(f64, f64, u8)] = &[
  (-0.82, -0.58, 0),
  (-0.70, -0.18, 0),
  (-0.58, -0.72, 0),
  (-0.48, 0.14, 0),
  (-0.35, -0.45, 0),
  (-0.28, 0.36, 0),
  (-0.20, -0.12, 0),
  (-0.12, 0.56, 0),
  (-0.62, 0.28, 0),
  (-0.08, -0.68, 0),
  (-0.40, -0.05, 0),
  (-0.76, -0.40, 0),
  (0.18, -0.30, 1),
  (0.25, 0.42, 1),
  (0.36, -0.52, 1),
  (0.42, 0.12, 1),
  (0.55, 0.64, 1),
  (0.62, -0.18, 1),
  (0.74, 0.38, 1),
  (0.82, -0.48, 1),
  (0.12, 0.68, 1),
  (0.48, -0.66, 1),
  (0.66, 0.08, 1),
  (0.30, -0.02, 1),
];

const CLS_MOONS: &[(f64, f64, u8)] = &[
  (-0.82, 0.10, 0),
  (-0.66, 0.48, 0),
  (-0.42, 0.72, 0),
  (-0.14, 0.76, 0),
  (0.12, 0.58, 0),
  (0.32, 0.24, 0),
  (-0.54, 0.32, 0),
  (-0.26, 0.56, 0),
  (0.02, 0.40, 0),
  (-0.74, -0.08, 0),
  (-0.06, 0.18, 0),
  (0.20, 0.05, 0),
  (-0.24, -0.14, 1),
  (-0.06, -0.46, 1),
  (0.20, -0.68, 1),
  (0.48, -0.66, 1),
  (0.70, -0.42, 1),
  (0.84, -0.08, 1),
  (0.04, -0.26, 1),
  (0.34, -0.44, 1),
  (0.60, -0.20, 1),
  (0.76, 0.12, 1),
  (-0.16, -0.34, 1),
  (0.46, -0.18, 1),
];

const CLS_RINGS: &[(f64, f64, u8)] = &[
  (-0.24, 0.12, 1),
  (-0.10, 0.28, 1),
  (0.14, 0.24, 1),
  (0.28, 0.04, 1),
  (0.18, -0.22, 1),
  (-0.06, -0.28, 1),
  (-0.26, -0.14, 1),
  (0.02, 0.04, 1),
  (-0.12, 0.08, 1),
  (0.12, -0.08, 1),
  (0.04, 0.18, 1),
  (-0.14, -0.16, 1),
  (-0.82, 0.18, 0),
  (-0.64, 0.58, 0),
  (-0.18, 0.78, 0),
  (0.36, 0.72, 0),
  (0.74, 0.40, 0),
  (0.84, -0.12, 0),
  (0.64, -0.56, 0),
  (0.22, -0.78, 0),
  (-0.32, -0.74, 0),
  (-0.72, -0.44, 0),
  (-0.54, 0.42, 0),
  (0.58, -0.36, 0),
];

fn dataset_points(ds: u8) -> &'static [(f64, f64, u8)] {
  match ds {
    1 => CLS_MOONS,
    2 => CLS_RINGS,
    _ => CLS_LINEAR,
  }
}

fn lcg_next(state: &mut u64) -> u64 {
  *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
  *state >> 32
}

#[inline(never)]
fn sort_f64_small(v: &mut [f64]) {
  for i in 1..v.len() {
    let mut j = i;
    while j > 0 && v[j - 1] > v[j] {
      v.swap(j - 1, j);
      j -= 1;
    }
  }
}

#[inline(never)]
fn sort_dist_cls_small(v: &mut [(f64, u8)]) {
  for i in 1..v.len() {
    let mut j = i;
    while j > 0 && v[j - 1].0 > v[j].0 {
      v.swap(j - 1, j);
      j -= 1;
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 内核 1：CART 决策树 / 随机森林 / GBDT / XGBoost 真实求解器
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone)]
struct CartNode {
  axis: usize,
  thresh: f64,
  gini: f64,
  prob1: f64,
  left: Option<Box<CartNode>>,
  right: Option<Box<CartNode>>,
}

fn gini_of(n0: usize, n1: usize) -> f64 {
  let tot = (n0 + n1) as f64;
  if tot <= 0.0 {
    return 0.0;
  }
  let p0 = n0 as f64 / tot;
  let p1 = n1 as f64 / tot;
  1.0 - p0 * p0 - p1 * p1
}

#[inline(never)]
fn fit_cart(pts: &[(f64, f64, u8)], max_depth: usize, depth: usize, feat_mask: u8) -> CartNode {
  let count = pts.len();
  let n1 = pts.iter().filter(|p| p.2 == 1).count();
  let gini = gini_of(count.saturating_sub(n1), n1);
  let prob1 = if count == 0 { 0.5 } else { n1 as f64 / count as f64 };
  if depth >= max_depth || count < 4 || gini < 1e-6 {
    return CartNode { axis: 0, thresh: 0.0, gini, prob1, left: None, right: None };
  }

  let (mut best_imp, mut best_axis, mut best_th) = (f64::INFINITY, 0usize, 0.0f64);
  for axis in 0..2usize {
    if (feat_mask & (1 << axis)) == 0 {
      continue;
    }
    let mut vals: Vec<f64> = pts.iter().map(|p| if axis == 0 { p.0 } else { p.1 }).collect();
    sort_f64_small(&mut vals);
    for w in vals.windows(2) {
      if (w[1] - w[0]).abs() < 1e-4 {
        continue;
      }
      let th = 0.5 * (w[0] + w[1]);
      let (mut l0, mut l1, mut r0, mut r1) = (0usize, 0usize, 0usize, 0usize);
      for &(x, y, c) in pts {
        let v = if axis == 0 { x } else { y };
        if v <= th {
          if c == 1 {
            l1 += 1;
          } else {
            l0 += 1;
          }
        } else if c == 1 {
          r1 += 1;
        } else {
          r0 += 1;
        }
      }
      let (lc, rc) = (l0 + l1, r0 + r1);
      if lc == 0 || rc == 0 {
        continue;
      }
      let imp = (lc as f64 * gini_of(l0, l1) + rc as f64 * gini_of(r0, r1)) / count as f64;
      if imp < best_imp {
        best_imp = imp;
        best_axis = axis;
        best_th = th;
      }
    }
  }
  if !best_imp.is_finite() || best_imp >= gini - 1e-6 {
    return CartNode { axis: 0, thresh: 0.0, gini, prob1, left: None, right: None };
  }
  let (lp, rp): (Vec<_>, Vec<_>) =
    pts.iter().copied().partition(|&(x, y, _)| (if best_axis == 0 { x } else { y }) <= best_th);
  CartNode {
    axis: best_axis,
    thresh: best_th,
    gini,
    prob1,
    left: Some(Box::new(fit_cart(&lp, max_depth, depth + 1, 0b11))),
    right: Some(Box::new(fit_cart(&rp, max_depth, depth + 1, 0b11))),
  }
}

fn eval_cart(node: &CartNode, x: f64, y: f64) -> f64 {
  match (&node.left, &node.right) {
    (Some(l), Some(r)) => {
      let v = if node.axis == 0 { x } else { y };
      if v <= node.thresh {
        eval_cart(l, x, y)
      } else {
        eval_cart(r, x, y)
      }
    }
    _ => node.prob1,
  }
}

fn count_leaves(node: &CartNode) -> usize {
  match (&node.left, &node.right) {
    (Some(l), Some(r)) => count_leaves(l) + count_leaves(r),
    _ => 1,
  }
}

#[inline(never)]
fn render_tree_dag(svg: &mut String, node: &CartNode, x0: f64, x1: f64, y: f64, dy: f64, d: usize) {
  let cx = 0.5 * (x0 + x1);
  if let (Some(l), Some(r)) = (&node.left, &node.right) {
    let (lx, rx, ny) = (0.5 * (x0 + cx), 0.5 * (cx + x1), y + dy);
    s_line(svg, cx, y + 14.0, lx, ny, "#94a3b8", 1.3, "");
    s_line(svg, cx, y + 14.0, rx, ny, "#94a3b8", 1.3, "");
    render_tree_dag(svg, l, x0, cx, ny, dy, d + 1);
    render_tree_dag(svg, r, cx, x1, ny, dy, d + 1);
    let w = if d <= 1 { 82.0 } else { 64.0 };
    s_rect(svg, cx - w * 0.5, y - 14.0, w, 28.0, 6.0, "#eef2ff", 1.0, "#4f46e5", 1.2);
    s_text(
      svg,
      cx,
      y - 2.0,
      9.0,
      "#312e81",
      true,
      true,
      &fmt_n(if node.axis == 0 { "x₁ ≤ " } else { "x₂ ≤ " }, node.thresh, 2, ""),
    );
    s_text(svg, cx, y + 9.0, 7.5, "#6366f1", false, true, &fmt_n("G=", node.gini, 2, ""));
  } else {
    let pos = node.prob1 >= 0.5;
    let (fl, st, tc) =
      if pos { ("#ffe4e6", "#e11d48", "#881337") } else { ("#e0e7ff", "#4f46e5", "#312e81") };
    let pct = (if pos { node.prob1 } else { 1.0 - node.prob1 }) * 100.0;
    s_rect(svg, cx - 22.0, y - 12.0, 44.0, 24.0, 5.0, fl, 1.0, st, 1.2);
    s_text(
      svg,
      cx,
      y + 3.0,
      8.5,
      tc,
      true,
      true,
      &fmt_n(if pos { "C1 " } else { "C0 " }, pct, 0, "%"),
    );
  }
}

#[inline(never)]
fn render_2d_field(
  svg: &mut String,
  pts: &[(f64, f64, u8)],
  eval_prob: &dyn Fn(f64, f64) -> f64,
  highlight_sv: Option<&[bool]>,
  query_pt: Option<(f64, f64, f64)>,
) -> f64 {
  let (gx0, gy0, gw, gh) = (18.0_f64, 18.0_f64, 288.0_f64, 244.0_f64);
  let (cols, rows) = (24usize, 18usize);
  let (cw, ch) = (gw / cols as f64, gh / rows as f64);
  s_panel(svg, gx0, gy0, gw, gh);

  for r in 0..rows {
    for c in 0..cols {
      let wx = -1.0 + (c as f64 + 0.5) * 2.0 / cols as f64;
      let wy = 1.0 - (r as f64 + 0.5) * 2.0 / rows as f64;
      let p = eval_prob(wx, wy).clamp(0.0, 1.0);
      let (col, al) = if p >= 0.5 {
        ("#fb7185", 0.08 + (p - 0.5) * 0.62)
      } else {
        ("#6366f1", 0.08 + (0.5 - p) * 0.62)
      };
      s_rect(
        svg,
        gx0 + c as f64 * cw,
        gy0 + r as f64 * ch,
        cw + 0.4,
        ch + 0.4,
        0.0,
        col,
        al,
        "",
        0.0,
      );
    }
  }
  s_line(svg, gx0, gy0 + gh * 0.5, gx0 + gw, gy0 + gh * 0.5, "#94a3b8", 1.0, "3 3");
  s_line(svg, gx0 + gw * 0.5, gy0, gx0 + gw * 0.5, gy0 + gh, "#94a3b8", 1.0, "3 3");

  if let Some((qx, qy, radius)) = query_pt {
    let sx = gx0 + (qx + 1.0) * 0.5 * gw;
    let sy = gy0 + (1.0 - qy) * 0.5 * gh;
    let _ = write!(
      svg,
      "<ellipse cx=\"{sx:.1}\" cy=\"{sy:.1}\" rx=\"{:.1}\" ry=\"{:.1}\" fill=\"#f59e0b\" fill-opacity=\"0.10\" stroke=\"#d97706\" stroke-width=\"1.4\" stroke-dasharray=\"4 3\"/>",
      radius * 0.5 * gw,
      radius * 0.5 * gh
    );
    for &(px, py, _) in pts {
      if ((px - qx).powi(2) + (py - qy).powi(2)).sqrt() <= radius + 1e-6 {
        s_line(
          svg,
          sx,
          sy,
          gx0 + (px + 1.0) * 0.5 * gw,
          gy0 + (1.0 - py) * 0.5 * gh,
          "#d97706",
          1.2,
          "",
        );
      }
    }
    s_circle(svg, sx, sy, 5.5, "#f59e0b", "#0f172a", 1.6);
  }

  let mut correct = 0usize;
  for (idx, &(px, py, cls)) in pts.iter().enumerate() {
    if u8::from(eval_prob(px, py) >= 0.5) == cls {
      correct += 1;
    }
    let sx = gx0 + (px + 1.0) * 0.5 * gw;
    let sy = gy0 + (1.0 - py) * 0.5 * gh;
    if highlight_sv.and_then(|s| s.get(idx).copied()).unwrap_or(false) {
      s_circle(svg, sx, sy, 8.2, "none", "#f59e0b", 2.0);
    }
    s_circle(svg, sx, sy, 4.4, if cls == 1 { "#e11d48" } else { "#4f46e5" }, "#ffffff", 1.3);
  }
  if pts.is_empty() {
    0.0
  } else {
    correct as f64 / pts.len() as f64
  }
}

#[inline(never)]
fn render_boosting_1d(
  is_xgb: bool,
  rounds: usize,
  lr_pct: i32,
  lam_x10: i32,
) -> (String, [String; 4]) {
  let lr = (lr_pct as f64 / 100.0).clamp(0.05, 1.0);
  let lam = (lam_x10 as f64 / 10.0).max(0.0);
  let n = 22usize;
  let mut xs = Vec::with_capacity(n);
  let mut ys = Vec::with_capacity(n);
  for i in 0..n {
    let x = -1.0 + 2.0 * (i as f64) / (n - 1) as f64;
    let noise = (((i * 37 + 11) % 19) as f64 - 9.0) * 0.018;
    xs.push(x);
    ys.push((2.4 * x).sin() * 0.72 + 0.28 * x + noise);
  }
  let mean_y = ys.iter().sum::<f64>() / n as f64;
  let mut pred = vec![mean_y; n];
  let calc_mse =
    |p: &[f64]| p.iter().zip(ys.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>() / n as f64;
  let mut mses = vec![calc_mse(&pred)];
  let mut stumps = Vec::with_capacity(rounds);
  let reg = if is_xgb { lam } else { 0.0 };

  for _ in 0..rounds {
    let total_g: f64 = pred.iter().zip(ys.iter()).map(|(p, y)| p - y).sum();
    let total_h = n as f64;
    let p_score = (total_g * total_g) / (total_h + reg);
    let (mut bg, mut bsp, mut bwl, mut bwr) = (-1.0_f64, 0.0, 0.0, 0.0);
    for k in 1..n {
      let gl: f64 = pred[..k].iter().zip(ys[..k].iter()).map(|(p, y)| p - y).sum();
      let hl = k as f64;
      let (gr, hr) = (total_g - gl, total_h - hl);
      let gain = 0.5 * (gl * gl / (hl + reg) + gr * gr / (hr + reg) - p_score);
      if gain > bg {
        bg = gain;
        bsp = 0.5 * (xs[k - 1] + xs[k]);
        bwl = -gl / (hl + reg);
        bwr = -gr / (hr + reg);
      }
    }
    for i in 0..n {
      pred[i] += lr * if xs[i] <= bsp { bwl } else { bwr };
    }
    stumps.push((bsp, bwl, bwr));
    mses.push(calc_mse(&pred));
  }

  let mut svg = String::with_capacity(3072);
  let (x0, y0, w, h) = (26.0, 22.0, 300.0, 232.0);
  let mx = |x: f64| x0 + (x + 1.0) * 0.5 * w;
  let my = |y: f64| y0 + (1.25 - y) / 2.5 * h;
  s_panel(&mut svg, x0, y0, w, h);

  let ref_p: Vec<(f64, f64)> = (0..=40)
    .map(|i| {
      let x = -1.0 + 2.0 * i as f64 / 40.0;
      (mx(x), my((2.4 * x).sin() * 0.72 + 0.28 * x))
    })
    .collect();
  s_poly(&mut svg, &ref_p, "#94a3b8", 1.4, "4 3");

  for i in 0..n {
    s_line(&mut svg, mx(xs[i]), my(ys[i]), mx(xs[i]), my(pred[i]), "#f59e0b", 1.3, "");
    s_circle(&mut svg, mx(xs[i]), my(ys[i]), 3.4, "#4f46e5", "#ffffff", 1.0);
  }
  let fit_p: Vec<(f64, f64)> = (0..=80)
    .map(|i| {
      let x = -1.0 + 2.0 * i as f64 / 80.0;
      let mut yh = mean_y;
      for &(sp, wl, wr) in &stumps {
        yh += lr * if x <= sp { wl } else { wr };
      }
      (mx(x), my(yh))
    })
    .collect();
  let col = if is_xgb { "#059669" } else { "#e11d48" };
  s_poly(&mut svg, &fit_p, col, 2.4, "");
  s_text(
    &mut svg,
    x0 + 10.0,
    38.0,
    9.5,
    col,
    true,
    false,
    &fmt_i("F_", rounds as i64, "(x) = F₀ + Σ η·f_t(x)"),
  );

  s_panel(&mut svg, 346.0, 22.0, 276.0, 232.0);
  s_text(&mut svg, 358.0, 40.0, 9.5, "#334155", true, false, "串行残差衰减 (MSE vs Round t)");
  s_bars(&mut svg, 346.0, 276.0, 190.0, 105.0, &mses, 't', col);
  let tag = if is_xgb {
    fmt_n("w* = -G/(H+", lam, 1, ") · 二阶泰勒叶权")
  } else {
    "r_i = y_i - F_{t-1}(x_i) · 负梯度残差拟合".to_string()
  };
  s_rect(&mut svg, 358.0, 210.0, 252.0, 32.0, 6.0, "#eef2ff", 1.0, "#818cf8", 1.0);
  s_text(&mut svg, 484.0, 230.0, 9.2, "#312e81", true, true, &tag);

  let init_m = mses.first().copied().unwrap_or(0.0);
  let cur_m = mses.last().copied().unwrap_or(0.0);
  let red = if init_m > 1e-9 { (1.0 - cur_m / init_m) * 100.0 } else { 0.0 };
  let last_w = stumps.last().map(|s| s.1.abs().max(s.2.abs())).unwrap_or(0.0);
  (
    svg,
    [
      fmt_i("", rounds as i64, " 轮弱树"),
      fmt_n("", cur_m, 4, ""),
      fmt_n("-", red, 1, "%"),
      fmt_n("", last_w, 3, ""),
    ],
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// 内核 2：2D 分类边界求解器（Logistic / SVM-Linear & RBF / KNN）
// ─────────────────────────────────────────────────────────────────────────────

fn draw_linear_line(svg: &mut String, w1: f64, w2: f64, rhs: f64, col: &str, sw: f64, dash: &str) {
  let (gx0, gy0, gw, gh) = (18.0_f64, 18.0_f64, 288.0_f64, 244.0_f64);
  let (xa, ya, xb, yb) = if w2.abs() >= w1.abs() {
    (-1.0, ((rhs + w1) / w2).clamp(-1.0, 1.0), 1.0, ((rhs - w1) / w2).clamp(-1.0, 1.0))
  } else {
    (((rhs + w2) / w1).clamp(-1.0, 1.0), -1.0, ((rhs - w2) / w1).clamp(-1.0, 1.0), 1.0)
  };
  s_line(
    svg,
    gx0 + (xa + 1.0) * 0.5 * gw,
    gy0 + (1.0 - ya) * 0.5 * gh,
    gx0 + (xb + 1.0) * 0.5 * gw,
    gy0 + (1.0 - yb) * 0.5 * gh,
    col,
    sw,
    dash,
  );
}

#[inline(never)]
fn render_boundary_2d(mode: &str, ds: u8, p1: i32, p2: i32) -> (String, [String; 4]) {
  let pts = dataset_points(ds);
  let mut svg = String::with_capacity(4096);

  let metrics = if mode == "knn" {
    let k = (p1 as usize).clamp(1, 9);
    let qx = (p2 as f64 - 50.0) / 60.0;
    let qy = 0.08_f64;
    let mut dists: Vec<(f64, u8)> =
      pts.iter().map(|&(px, py, c)| (((px - qx).powi(2) + (py - qy).powi(2)).sqrt(), c)).collect();
    sort_dist_cls_small(&mut dists);
    let radius = dists.get(k.saturating_sub(1)).map(|d| d.0).unwrap_or(0.2);
    let k_pos = dists.iter().take(k).filter(|d| d.1 == 1).count();
    let eval = |wx: f64, wy: f64| -> f64 {
      let mut dd: Vec<(f64, u8)> =
        pts.iter().map(|&(px, py, c)| ((px - wx).powi(2) + (py - wy).powi(2), c)).collect();
      sort_dist_cls_small(&mut dd);
      dd.iter().take(k).filter(|d| d.1 == 1).count() as f64 / k as f64
    };
    let acc = render_2d_field(&mut svg, pts, &eval, None, Some((qx, qy, radius)));
    let k0 = k - k_pos;
    s_panel(&mut svg, 330.0, 18.0, 292.0, 244.0);
    s_text(
      &mut svg,
      346.0,
      42.0,
      10.2,
      "#0f172a",
      true,
      false,
      &fmt_i("KNN 查询点 · K=", k as i64, " 邻居投票"),
    );
    s_rect(
      &mut svg,
      346.0,
      60.0,
      (250.0 * k0 as f64 / k as f64).max(56.0),
      24.0,
      4.0,
      "#4f46e5",
      1.0,
      "",
      0.0,
    );
    s_text(&mut svg, 354.0, 76.0, 9.5, "#ffffff", true, false, &fmt_i("C0: ", k0 as i64, " 票"));
    s_rect(
      &mut svg,
      346.0,
      94.0,
      (250.0 * k_pos as f64 / k as f64).max(56.0),
      24.0,
      4.0,
      "#e11d48",
      1.0,
      "",
      0.0,
    );
    s_text(
      &mut svg,
      354.0,
      110.0,
      9.5,
      "#ffffff",
      true,
      false,
      &fmt_i("C1: ", k_pos as i64, " 票"),
    );
    s_text(
      &mut svg,
      346.0,
      152.0,
      9.2,
      "#334155",
      false,
      false,
      "· K=1：纯 Voronoi 镶嵌，高方差易受噪点干扰",
    );
    s_text(
      &mut svg,
      346.0,
      174.0,
      9.2,
      "#334155",
      false,
      false,
      "· K 增大：边界平滑抗噪，过大将欠拟合",
    );
    s_text(
      &mut svg,
      346.0,
      212.0,
      9.0,
      "#4f46e5",
      true,
      false,
      "ŷ(x) = argmax_c Σ_{i∈N_K(x)} I(y_i = c)",
    );
    [
      fmt_i("K = ", k as i64, ""),
      fmt_n("", acc * 100.0, 1, "%"),
      fmt_n("", radius, 2, ""),
      fmt_i("", k_pos as i64, " 正类"),
    ]
  } else if mode == "logistic" {
    let reg = (100.0 / (p1.max(1) as f64)).clamp(0.02, 5.0);
    let (mut w1, mut w2, mut b) = (0.0_f64, 0.0_f64, 0.0_f64);
    for _ in 0..80 {
      let (mut g1, mut g2, mut gb) = (0.0, 0.0, 0.0);
      for &(x, y, c) in pts {
        let diff = sigmoid(w1 * x + w2 * y + b) - c as f64;
        g1 += diff * x;
        g2 += diff * y;
        gb += diff;
      }
      let inv_n = 1.0 / pts.len() as f64;
      w1 -= 0.45 * (g1 * inv_n + reg * 0.05 * w1);
      w2 -= 0.45 * (g2 * inv_n + reg * 0.05 * w2);
      b -= 0.45 * (gb * inv_n);
    }
    let acc = render_2d_field(&mut svg, pts, &|x, y| sigmoid(w1 * x + w2 * y + b), None, None);
    if w2.abs() > 0.05 || w1.abs() > 0.05 {
      draw_linear_line(&mut svg, w1, w2, -b, "#0f172a", 2.2, "");
    }
    let sig_p: Vec<(f64, f64)> = (0..=40)
      .map(|i| {
        let z = -6.0 + 12.0 * i as f64 / 40.0;
        (350.0 + (z + 6.0) / 12.0 * 252.0, 215.0 - sigmoid(z) * 140.0)
      })
      .collect();
    let bce = pts
      .iter()
      .map(|&(x, y, c)| {
        let p = sigmoid(w1 * x + w2 * y + b).clamp(1e-6, 1.0 - 1e-6);
        if c == 1 {
          -p.ln()
        } else {
          -(1.0 - p).ln()
        }
      })
      .sum::<f64>()
      / pts.len() as f64;
    s_panel(&mut svg, 330.0, 18.0, 292.0, 244.0);
    s_text(&mut svg, 346.0, 42.0, 10.2, "#0f172a", true, false, "Sigmoid 概率映射 p = σ(w·x + b)");
    s_line(&mut svg, 350.0, 145.0, 602.0, 145.0, "#94a3b8", 1.0, "3 3");
    s_line(&mut svg, 476.0, 68.0, 476.0, 220.0, "#94a3b8", 1.0, "3 3");
    s_poly(&mut svg, &sig_p, "#4f46e5", 2.4, "");
    s_text(
      &mut svg,
      346.0,
      244.0,
      9.0,
      "#334155",
      false,
      false,
      &fmt_n("Log-Odds 交叉熵 BCE = ", bce, 3, ""),
    );
    [
      fmt_n("C = ", p1 as f64 / 10.0, 1, ""),
      fmt_n("", acc * 100.0, 1, "%"),
      fmt_n("", bce, 3, ""),
      fmt_n("", (w1 * w1 + w2 * w2).sqrt(), 2, ""),
    ]
  } else {
    let gamma = (p1 as f64 / 10.0).clamp(0.2, 8.0);
    let c_val = (p2 as f64 / 10.0).clamp(0.2, 10.0);
    let n = pts.len();
    let (mut alpha, mut b, use_rbf) = (vec![0.0_f64; n], 0.0_f64, ds != 0);
    let kfn = |ax: f64, ay: f64, bx: f64, by: f64| -> f64 {
      if use_rbf {
        (-gamma * ((ax - bx).powi(2) + (ay - by).powi(2))).exp()
      } else {
        ax * bx + ay * by
      }
    };
    let mut kmat = vec![0.0_f64; n * n];
    for i in 0..n {
      for j in 0..n {
        kmat[i * n + j] = kfn(pts[i].0, pts[i].1, pts[j].0, pts[j].1);
      }
    }
    for _ in 0..60 {
      for i in 0..n {
        let yi = if pts[i].2 == 1 { 1.0 } else { -1.0 };
        let mut f = b;
        for j in 0..n {
          f += alpha[j] * (if pts[j].2 == 1 { 1.0 } else { -1.0 }) * kmat[i * n + j];
        }
        let m = yi * f;
        if m < 1.0 {
          alpha[i] =
            (alpha[i] + 0.35 * (1.0 - m) / (kmat[i * n + i].max(1e-4) + 0.15)).clamp(0.0, c_val);
          b += 0.04 * yi * (1.0 - m);
        } else if alpha[i] > 0.0 {
          alpha[i] = (alpha[i] * 0.96).max(0.0);
        }
      }
    }
    let sv_flags: Vec<bool> = alpha.iter().map(|&a| a > 0.08).collect();
    let sv_count = sv_flags.iter().filter(|&&f| f).count();
    let decision = |wx: f64, wy: f64| -> f64 {
      let mut f = b;
      for j in 0..n {
        if alpha[j] > 1e-4 {
          f +=
            alpha[j] * (if pts[j].2 == 1 { 1.0 } else { -1.0 }) * kfn(pts[j].0, pts[j].1, wx, wy);
        }
      }
      f
    };
    let acc =
      render_2d_field(&mut svg, pts, &|x, y| sigmoid(2.2 * decision(x, y)), Some(&sv_flags), None);
    let mut w_norm = 0.0_f64;
    if !use_rbf {
      let (mut w1, mut w2) = (0.0, 0.0);
      for j in 0..n {
        let yj = if pts[j].2 == 1 { 1.0 } else { -1.0 };
        w1 += alpha[j] * yj * pts[j].0;
        w2 += alpha[j] * yj * pts[j].1;
      }
      w_norm = (w1 * w1 + w2 * w2).sqrt().max(1e-3);
      draw_linear_line(&mut svg, w1, w2, -b, "#0f172a", 2.2, "");
      draw_linear_line(&mut svg, w1, w2, 1.0 - b, "#e11d48", 1.3, "4 3");
      draw_linear_line(&mut svg, w1, w2, -1.0 - b, "#4f46e5", 1.3, "4 3");
    }
    let margin_s =
      if !use_rbf { fmt_n("", 2.0 / w_norm, 2, "") } else { fmt_n("γ=", gamma, 1, "") };
    s_panel(&mut svg, 330.0, 18.0, 292.0, 244.0);
    s_text(
      &mut svg,
      346.0,
      42.0,
      10.2,
      "#0f172a",
      true,
      false,
      if use_rbf {
        "RBF 高斯核 K(x,z) = exp(-γ‖x-z‖²)"
      } else {
        "线性最大间隔 w·x + b = ±1"
      },
    );
    s_text(
      &mut svg,
      346.0,
      74.0,
      9.2,
      "#334155",
      false,
      false,
      &fmt_i("· 琥珀外环为支持向量（共 ", sv_count as i64, " 个）"),
    );
    s_text(
      &mut svg,
      346.0,
      98.0,
      9.2,
      "#334155",
      false,
      false,
      "· 非支持向量 α_i = 0 对超平面无影响",
    );
    s_rect(&mut svg, 346.0, 134.0, 260.0, 44.0, 6.0, "#fffbeb", 1.0, "#f59e0b", 1.0);
    s_text(
      &mut svg,
      476.0,
      154.0,
      9.2,
      "#92400e",
      true,
      true,
      "min ½‖w‖² + C·Σ max(0, 1 - y_i f(x_i))",
    );
    s_text(&mut svg, 476.0, 170.0, 8.2, "#b45309", false, true, "软间隔 Hinge 损失 + 对偶核技巧");
    s_text(
      &mut svg,
      346.0,
      226.0,
      9.0,
      "#4f46e5",
      true,
      false,
      &fmt_n("Support Vectors · C = ", c_val, 1, ""),
    );
    [
      fmt_i("", sv_count as i64, " SV"),
      fmt_n("", acc * 100.0, 1, "%"),
      margin_s,
      fmt_n("C=", c_val, 1, ""),
    ]
  };
  (svg, metrics)
}

// ─────────────────────────────────────────────────────────────────────────────
// 内核 3：线性 / 多项式岭回归闭式高斯消元求解器
// ─────────────────────────────────────────────────────────────────────────────

#[inline(never)]
fn solve_linear_system(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
  let n = b.len();
  for i in 0..n {
    let mut piv = i;
    for r in (i + 1)..n {
      if a[r][i].abs() > a[piv][i].abs() {
        piv = r;
      }
    }
    a.swap(i, piv);
    b.swap(i, piv);
    let diag = a[i][i];
    if diag.abs() < 1e-12 {
      continue;
    }
    for r in (i + 1)..n {
      let f = a[r][i] / diag;
      for c in i..n {
        a[r][c] -= f * a[i][c];
      }
      b[r] -= f * b[i];
    }
  }
  let mut x = vec![0.0_f64; n];
  for i in (0..n).rev() {
    let mut s = b[i];
    for c in (i + 1)..n {
      s -= a[i][c] * x[c];
    }
    if a[i][i].abs() > 1e-12 {
      x[i] = s / a[i][i];
    }
  }
  x
}

#[inline(never)]
fn render_regression_fit(deg: usize, lam_x100: i32) -> (String, [String; 4]) {
  let d = deg.clamp(1, 5);
  let lam = (lam_x100 as f64 / 100.0).max(0.0);
  let train: &[(f64, f64)] = &[
    (-0.92, -0.58),
    (-0.70, -0.82),
    (-0.48, -0.34),
    (-0.25, -0.10),
    (-0.04, 0.24),
    (0.18, 0.68),
    (0.42, 0.72),
    (0.66, 0.45),
    (0.88, 0.14),
    (0.96, 0.36),
  ];
  let val: &[(f64, f64)] =
    &[(-0.80, -0.72), (-0.36, -0.42), (0.08, 0.35), (0.54, 0.62), (0.78, 0.30)];

  let m = d + 1;
  let mut xtx = vec![vec![0.0_f64; m]; m];
  let mut xty = vec![0.0_f64; m];
  for &(x, y) in train {
    let mut phi = vec![1.0_f64; m];
    for k in 1..m {
      phi[k] = phi[k - 1] * x;
    }
    for r in 0..m {
      xty[r] += phi[r] * y;
      for c in 0..m {
        xtx[r][c] += phi[r] * phi[c];
      }
    }
  }
  for (k, row) in xtx.iter_mut().enumerate().take(m).skip(1) {
    row[k] += lam;
  }
  let w = solve_linear_system(xtx, xty);
  let eval = |x: f64| -> f64 {
    let (mut y, mut xp) = (0.0, 1.0);
    for &wk in &w {
      y += wk * xp;
      xp *= x;
    }
    y
  };
  let train_mse =
    train.iter().map(|&(x, y)| (eval(x) - y).powi(2)).sum::<f64>() / train.len() as f64;
  let val_mse = val.iter().map(|&(x, y)| (eval(x) - y).powi(2)).sum::<f64>() / val.len() as f64;
  let w_norm = w.iter().skip(1).map(|v| v * v).sum::<f64>().sqrt();

  let mut svg = String::with_capacity(3072);
  let (x0, y0, pw, ph) = (26.0, 22.0, 344.0, 232.0);
  let mx = |x: f64| x0 + (x + 1.05) / 2.1 * pw;
  let my = |y: f64| y0 + (1.2 - y) / 2.4 * ph;
  s_panel(&mut svg, x0, y0, pw, ph);
  let curve: Vec<(f64, f64)> = (0..=64)
    .map(|i| {
      let x = -1.0 + 2.0 * i as f64 / 64.0;
      (mx(x), my(eval(x).clamp(-1.15, 1.15)))
    })
    .collect();
  s_poly(&mut svg, &curve, "#4f46e5", 2.5, "");
  for &(x, y) in train {
    s_line(&mut svg, mx(x), my(y), mx(x), my(eval(x).clamp(-1.15, 1.15)), "#f59e0b", 1.3, "");
    s_circle(&mut svg, mx(x), my(y), 4.2, "#4f46e5", "#ffffff", 1.2);
  }
  for &(x, y) in val {
    s_rect(&mut svg, mx(x) - 3.5, my(y) - 3.5, 7.0, 7.0, 1.5, "#059669", 1.0, "#ffffff", 1.1);
  }

  s_panel(&mut svg, 386.0, 22.0, 236.0, 232.0);
  s_text(&mut svg, 398.0, 42.0, 10.0, "#0f172a", true, false, "各阶权重 |w_k|（L2 收缩）");
  let max_w = w.iter().map(|v| v.abs()).fold(1.0_f64, f64::max);
  for (k, &wk) in w.iter().enumerate() {
    let by = 58.0 + k as f64 * 28.0;
    let bw = ((wk.abs() / max_w) * 145.0).max(4.0);
    let col = if k == 0 {
      "#64748b"
    } else if wk >= 0.0 {
      "#4f46e5"
    } else {
      "#e11d48"
    };
    s_text(&mut svg, 398.0, by + 11.0, 9.0, "#334155", false, false, &fmt_i("w", k as i64, ""));
    s_rect(&mut svg, 422.0, by, bw, 15.0, 3.0, col, 0.85, "", 0.0);
    s_text(&mut svg, 428.0 + bw, by + 11.0, 8.5, "#475569", false, false, &fmt_n("", wk, 2, ""));
  }
  s_text(
    &mut svg,
    398.0,
    240.0,
    8.5,
    "#059669",
    true,
    false,
    "w* = (XᵀX + λI)⁻¹ Xᵀy (蓝=训练 绿=验证)",
  );
  (
    svg,
    [
      fmt_i("deg = ", d as i64, ""),
      fmt_n("", train_mse, 4, ""),
      fmt_n("", val_mse, 4, ""),
      fmt_n("", w_norm, 2, ""),
    ],
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// 内核 4：K-Means Lloyd 迭代 / PCA 2×2 精确特征分解
// ─────────────────────────────────────────────────────────────────────────────

const CLUSTER_PTS: &[(f64, f64)] = &[
  (-0.68, 0.56),
  (-0.54, 0.68),
  (-0.74, 0.38),
  (-0.44, 0.52),
  (-0.60, 0.42),
  (-0.48, 0.72),
  (0.56, 0.52),
  (0.68, 0.64),
  (0.46, 0.40),
  (0.74, 0.44),
  (0.60, 0.34),
  (0.50, 0.66),
  (-0.48, -0.58),
  (-0.62, -0.44),
  (-0.34, -0.66),
  (-0.56, -0.72),
  (-0.38, -0.46),
  (-0.66, -0.60),
  (0.48, -0.52),
  (0.66, -0.40),
  (0.54, -0.68),
  (0.72, -0.58),
  (0.38, -0.56),
  (0.60, -0.48),
];

#[inline(never)]
fn render_cluster_pca(mode: &str, p1: i32, p2: i32) -> (String, [String; 4]) {
  let mut svg = String::with_capacity(4096);
  if mode == "pca" {
    let angle = p1 as f64;
    let spread = (p2 as f64 / 100.0).clamp(0.08, 0.85);
    let (ca, sa) = ((angle * PI / 180.0).cos(), (angle * PI / 180.0).sin());
    let mut pts = Vec::with_capacity(24);
    for i in 0..24usize {
      let u = -0.82 + 1.64 * (i as f64 / 23.0);
      let v = (((i * 29 + 7) % 17) as f64 / 8.0 - 1.0) * spread;
      pts.push((u * ca - v * sa, u * sa + v * ca));
    }
    let n = pts.len() as f64;
    let mx = pts.iter().map(|p| p.0).sum::<f64>() / n;
    let my = pts.iter().map(|p| p.1).sum::<f64>() / n;
    let (mut cxx, mut cxy, mut cyy) = (0.0, 0.0, 0.0);
    for &(x, y) in &pts {
      let (dx, dy) = (x - mx, y - my);
      cxx += dx * dx;
      cxy += dx * dy;
      cyy += dy * dy;
    }
    cxx /= n;
    cxy /= n;
    cyy /= n;
    let tr = cxx + cyy;
    let disc = ((tr * tr * 0.25) - (cxx * cyy - cxy * cxy)).max(0.0).sqrt();
    let lam1 = tr * 0.5 + disc;
    let lam2 = (tr * 0.5 - disc).max(0.0);
    let (u1x, u1y) = if cxy.abs() > 1e-8 {
      let (vx, vy) = (lam1 - cyy, cxy);
      let nm = (vx * vx + vy * vy).sqrt().max(1e-8);
      (vx / nm, vy / nm)
    } else if cxx >= cyy {
      (1.0, 0.0)
    } else {
      (0.0, 1.0)
    };
    let (u2x, u2y) = (-u1y, u1x);
    let evr = if lam1 + lam2 > 1e-9 { lam1 / (lam1 + lam2) } else { 1.0 };

    let (gx0, gy0, gw, gh) = (22.0, 18.0, 290.0, 244.0);
    let sx = |x: f64| gx0 + (x + 1.1) / 2.2 * gw;
    let sy = |y: f64| gy0 + (1.1 - y) / 2.2 * gh;
    s_panel(&mut svg, gx0, gy0, gw, gh);
    s_line(
      &mut svg,
      sx(mx - u1x * 0.95),
      sy(my - u1y * 0.95),
      sx(mx + u1x * 0.95),
      sy(my + u1y * 0.95),
      "#4f46e5",
      2.2,
      "",
    );
    s_line(
      &mut svg,
      sx(mx - u2x * 0.45),
      sy(my - u2y * 0.45),
      sx(mx + u2x * 0.45),
      sy(my + u2y * 0.45),
      "#059669",
      1.6,
      "4 3",
    );
    for &(x, y) in &pts {
      let t = (x - mx) * u1x + (y - my) * u1y;
      let (px, py) = (mx + t * u1x, my + t * u1y);
      s_line(&mut svg, sx(x), sy(y), sx(px), sy(py), "#f59e0b", 1.1, "2 2");
      s_circle(&mut svg, sx(px), sy(py), 2.6, "#4f46e5", "#4f46e5", 0.5);
      s_circle(&mut svg, sx(x), sy(y), 3.8, "#0f172a", "#ffffff", 1.1);
    }
    s_panel(&mut svg, 330.0, 18.0, 292.0, 244.0);
    s_text(&mut svg, 346.0, 42.0, 10.2, "#0f172a", true, false, "主成分方差解释比例 (Σ u = λ u)");
    s_text(
      &mut svg,
      346.0,
      68.0,
      9.0,
      "#4f46e5",
      true,
      false,
      &fmt_n("PC1 方差解释比: ", evr * 100.0, 1, "%"),
    );
    s_rect(&mut svg, 346.0, 76.0, (240.0 * evr).max(12.0), 20.0, 4.0, "#4f46e5", 1.0, "", 0.0);
    s_text(
      &mut svg,
      346.0,
      118.0,
      9.0,
      "#059669",
      true,
      false,
      &fmt_n("PC2 重构误差比: ", (1.0 - evr) * 100.0, 1, "%"),
    );
    s_rect(
      &mut svg,
      346.0,
      126.0,
      (240.0 * (1.0 - evr)).max(6.0),
      20.0,
      4.0,
      "#059669",
      1.0,
      "",
      0.0,
    );
    s_text(
      &mut svg,
      346.0,
      174.0,
      9.2,
      "#334155",
      false,
      false,
      "· 琥珀虚线 = 降维至 1D 的正交重构残差",
    );
    s_text(&mut svg, 346.0, 224.0, 9.0, "#4f46e5", true, false, "Cov = XᵀX/n = U Λ Uᵀ · z = X u₁");
    return (
      svg,
      [
        fmt_n("", evr * 100.0, 1, "%"),
        fmt_n("", lam1, 3, ""),
        fmt_n("", lam2, 3, ""),
        fmt_i("", angle as i64, "°"),
      ],
    );
  }

  let k = (p1 as usize).clamp(2, 5);
  let steps = (p2 as usize).clamp(0, 6);
  let seeds: &[(f64, f64)] =
    &[(-0.15, 0.15), (0.15, 0.12), (-0.12, -0.18), (0.18, -0.14), (0.0, 0.0)];
  let pal = ["#4f46e5", "#e11d48", "#059669", "#d97706", "#7c3aed"];
  let mut centers: Vec<(f64, f64)> = seeds.iter().take(k).copied().collect();
  let mut history = vec![centers.clone()];
  let assign = |cs: &[(f64, f64)]| -> (Vec<usize>, f64) {
    let mut asgn = Vec::with_capacity(CLUSTER_PTS.len());
    let mut wcss = 0.0;
    for &(x, y) in CLUSTER_PTS {
      let (mut bj, mut bd) = (0usize, f64::INFINITY);
      for (j, &(cx, cy)) in cs.iter().enumerate() {
        let d = (x - cx).powi(2) + (y - cy).powi(2);
        if d < bd {
          bd = d;
          bj = j;
        }
      }
      asgn.push(bj);
      wcss += bd;
    }
    (asgn, wcss)
  };
  let (mut labels, mut wcss) = assign(&centers);
  let mut wcss_hist = vec![wcss];
  for _ in 0..steps {
    let mut sums = vec![(0.0, 0.0, 0usize); k];
    for (i, &(x, y)) in CLUSTER_PTS.iter().enumerate() {
      let c = labels[i];
      sums[c].0 += x;
      sums[c].1 += y;
      sums[c].2 += 1;
    }
    for j in 0..k {
      if sums[j].2 > 0 {
        centers[j] = (sums[j].0 / sums[j].2 as f64, sums[j].1 / sums[j].2 as f64);
      }
    }
    history.push(centers.clone());
    let (nl, nw) = assign(&centers);
    labels = nl;
    wcss = nw;
    wcss_hist.push(wcss);
  }

  let (gx0, gy0, gw, gh) = (18.0, 18.0, 288.0, 244.0);
  let (cols, rows) = (24usize, 18usize);
  let (cw, ch) = (gw / cols as f64, gh / rows as f64);
  s_panel(&mut svg, gx0, gy0, gw, gh);
  for r in 0..rows {
    for c in 0..cols {
      let wx = -1.0 + (c as f64 + 0.5) * 2.0 / cols as f64;
      let wy = 1.0 - (r as f64 + 0.5) * 2.0 / rows as f64;
      let (mut bj, mut bd) = (0usize, f64::INFINITY);
      for (j, &(cx, cy)) in centers.iter().enumerate() {
        let d = (wx - cx).powi(2) + (wy - cy).powi(2);
        if d < bd {
          bd = d;
          bj = j;
        }
      }
      s_rect(
        &mut svg,
        gx0 + c as f64 * cw,
        gy0 + r as f64 * ch,
        cw + 0.4,
        ch + 0.4,
        0.0,
        pal[bj % pal.len()],
        0.16,
        "",
        0.0,
      );
    }
  }
  let sx = |x: f64| gx0 + (x + 1.0) * 0.5 * gw;
  let sy = |y: f64| gy0 + (1.0 - y) * 0.5 * gh;
  for (i, &(x, y)) in CLUSTER_PTS.iter().enumerate() {
    s_circle(&mut svg, sx(x), sy(y), 4.2, pal[labels[i] % pal.len()], "#ffffff", 1.2);
  }
  for j in 0..k {
    let col = pal[j % pal.len()];
    for pair in history.windows(2) {
      s_line(
        &mut svg,
        sx(pair[0][j].0),
        sy(pair[0][j].1),
        sx(pair[1][j].0),
        sy(pair[1][j].1),
        col,
        1.6,
        "3 2",
      );
    }
    let (cx, cy) = centers[j];
    s_rect(&mut svg, sx(cx) - 5.5, sy(cy) - 5.5, 11.0, 11.0, 2.0, col, 1.0, "#0f172a", 1.6);
  }
  s_panel(&mut svg, 330.0, 18.0, 292.0, 244.0);
  s_text(
    &mut svg,
    346.0,
    42.0,
    10.2,
    "#0f172a",
    true,
    false,
    &fmt_i("Lloyd 迭代总惯量 WCSS (K=", k as i64, ")"),
  );
  s_bars(&mut svg, 330.0, 292.0, 205.0, 125.0, &wcss_hist, 's', "#059669");
  s_text(
    &mut svg,
    346.0,
    246.0,
    8.5,
    "#4f46e5",
    true,
    false,
    "J = Σ_k Σ_{x∈C_k} ‖x - μ_k‖² (方块=质心)",
  );
  (
    svg,
    [
      fmt_i("K = ", k as i64, ""),
      fmt_i("Iter ", steps as i64, ""),
      fmt_n("", wcss, 2, ""),
      fmt_i("", CLUSTER_PTS.len() as i64, " pts"),
    ],
  )
}

#[inline(never)]
fn compute_lab_frame(
  kind: MlLabKind,
  cur_mode: &str,
  cur_ds: u8,
  a: i32,
  b: i32,
) -> (String, [&'static str; 4], [String; 4]) {
  match kind {
    MlLabKind::TreeBoost => {
      if cur_mode == "gbdt" || cur_mode == "xgboost" {
        let is_xgb = cur_mode == "xgboost";
        let (s, vals) = render_boosting_1d(
          is_xgb,
          (a as usize).clamp(1, 8),
          if is_xgb { 45 } else { b.clamp(10, 100) },
          if is_xgb { b.clamp(0, 50) } else { 0 },
        );
        (s, ["弱学习器轮数", "当前 MSE", "残差下降", "末轮最大叶权"], vals)
      } else if cur_mode == "random_forest" {
        let (n_trees, depth) = ((a as usize).clamp(1, 7), (b as usize).clamp(1, 4));
        let pts = dataset_points(cur_ds);
        let mut seed = 0x51A7_C0DE_u64;
        let mut forest = Vec::with_capacity(n_trees);
        for t in 0..n_trees {
          let mut sample = Vec::with_capacity(pts.len());
          for _ in 0..pts.len() {
            sample.push(pts[(lcg_next(&mut seed) as usize) % pts.len()]);
          }
          let mask = if t % 3 == 0 {
            0b11
          } else if t % 3 == 1 {
            0b01
          } else {
            0b10
          };
          forest.push(fit_cart(&sample, depth, 0, mask));
        }
        let mut svg = String::with_capacity(4096);
        let acc = render_2d_field(
          &mut svg,
          pts,
          &|x, y| forest.iter().map(|tr| eval_cart(tr, x, y)).sum::<f64>() / n_trees as f64,
          None,
          None,
        );
        s_panel(&mut svg, 330.0, 18.0, 292.0, 244.0);
        s_text(
          &mut svg,
          346.0,
          40.0,
          10.0,
          "#0f172a",
          true,
          false,
          &fmt_i("并行 Bagging · ", n_trees as i64, " 棵子采样树投票"),
        );
        if let Some(ft) = forest.first() {
          render_tree_dag(&mut svg, ft, 340.0, 614.0, 68.0, 48.0, 0);
        }
        let tot_leaves: usize = forest.iter().map(count_leaves).sum();
        (
          svg,
          ["森林树数量", "集成准确率", "单树最大深度", "总叶子节点"],
          [
            fmt_i("", n_trees as i64, " Trees"),
            fmt_n("", acc * 100.0, 1, "%"),
            fmt_i("depth=", depth as i64, ""),
            fmt_i("", tot_leaves as i64, " leaves"),
          ],
        )
      } else {
        let depth = (a as usize).clamp(1, 4);
        let pts = dataset_points(cur_ds);
        let tree = fit_cart(pts, depth, 0, 0b11);
        let mut svg = String::with_capacity(4096);
        let acc = render_2d_field(&mut svg, pts, &|x, y| eval_cart(&tree, x, y), None, None);
        s_panel(&mut svg, 330.0, 18.0, 292.0, 244.0);
        s_text(
          &mut svg,
          346.0,
          38.0,
          9.5,
          "#0f172a",
          true,
          false,
          &fmt_i("实时 CART Gini 分裂树 DAG (depth=", depth as i64, ")"),
        );
        render_tree_dag(&mut svg, &tree, 336.0, 616.0, 64.0, 46.0, 0);
        (
          svg,
          ["最大树深", "分类准确率", "根节点 Gini", "叶子分区数"],
          [
            fmt_i("depth=", depth as i64, ""),
            fmt_n("", acc * 100.0, 1, "%"),
            fmt_n("", tree.gini, 3, ""),
            fmt_i("", count_leaves(&tree) as i64, " leaves"),
          ],
        )
      }
    }
    MlLabKind::Boundary2d => {
      let (s, vals) = render_boundary_2d(cur_mode, cur_ds, a, b);
      let lbls = match cur_mode {
        "knn" => ["邻居数 K", "样本准确率", "覆盖半径 r", "查询点投票"],
        "logistic" => ["正则倒数 C", "分类准确率", "交叉熵 BCE", "权重模长 ‖w‖"],
        _ => ["支持向量数", "分类准确率", "间隔 / 核带宽", "惩罚系数 C"],
      };
      (s, lbls, vals)
    }
    MlLabKind::RegressionFit => {
      let (s, vals) = render_regression_fit(a as usize, b);
      (s, ["多项式阶数", "训练 MSE", "验证 MSE", "权重范数 ‖w‖₂"], vals)
    }
    MlLabKind::ClusterPca => {
      let (s, vals) = render_cluster_pca(cur_mode, a, b);
      let lbls = if cur_mode == "pca" {
        ["PC1 方差占比", "特征值 λ₁", "特征值 λ₂", "点云主轴角"]
      } else {
        ["簇数量 K", "迭代步数", "总惯量 WCSS", "样本总数"]
      };
      (s, lbls, vals)
    }
  }
}

#[inline(never)]
fn ml_slider(
  label: &str,
  min: i32,
  max: i32,
  val: i32,
  val_txt: &str,
  accent: &str,
  mut sig: Signal<i32>,
) -> Element {
  rsx! {
    label { class: "block space-y-1",
      div { class: "flex justify-between text-slate-600 dark:text-slate-300 font-medium",
        span { "{label}" }
        span { class: "font-mono font-bold {accent}", "{val_txt}" }
      }
      input {
        r#type: "range",
        min: "{min}",
        max: "{max}",
        value: "{val}",
        class: "w-full accent-teal-600 cursor-pointer",
        oninput: move |e| {
          if let Ok(v) = e.value().parse::<i32>() {
            sig.set(v);
          }
        },
      }
    }
  }
}

#[component]
pub fn MlLabWorkbench(kind: MlLabKind, #[props(default)] initial_mode: String) -> Element {
  let lang = current_lang();
  let mut mode = use_signal(move || {
    if initial_mode.trim().is_empty() {
      match kind {
        MlLabKind::TreeBoost => "decision_tree".to_string(),
        MlLabKind::Boundary2d => "svm_rbf".to_string(),
        MlLabKind::RegressionFit => "linear_ridge".to_string(),
        MlLabKind::ClusterPca => "kmeans".to_string(),
      }
    } else {
      initial_mode.trim().to_ascii_lowercase()
    }
  });
  let mut dataset = use_signal(|| u8::from(mode() == "svm_rbf"));
  let mut param_a = use_signal(|| match kind {
    MlLabKind::TreeBoost => {
      if mode() == "gbdt" || mode() == "xgboost" {
        5
      } else {
        3
      }
    }
    MlLabKind::Boundary2d => {
      if mode() == "knn" {
        5
      } else {
        20
      }
    }
    MlLabKind::RegressionFit => 3,
    MlLabKind::ClusterPca => {
      if mode() == "pca" {
        35
      } else {
        4
      }
    }
  });
  let mut param_b = use_signal(|| match kind {
    MlLabKind::TreeBoost => {
      if mode() == "xgboost" {
        10
      } else {
        40
      }
    }
    MlLabKind::Boundary2d => {
      if mode() == "knn" {
        55
      } else {
        30
      }
    }
    MlLabKind::RegressionFit => 15,
    MlLabKind::ClusterPca => {
      if mode() == "pca" {
        26
      } else {
        4
      }
    }
  });

  let (cur_mode, cur_ds, a, b) = (mode(), dataset(), param_a(), param_b());
  let (svg_body, m_labels, m_vals) = compute_lab_frame(kind, &cur_mode, cur_ds, a, b);

  let (lbl_a, min_a, max_a, val_a_txt, lbl_b, min_b, max_b, val_b_txt, show_ds) =
    match (kind, cur_mode.as_str()) {
      (MlLabKind::TreeBoost, "decision_tree") => (
        "最大树深 max_depth",
        1,
        4,
        fmt_i("", a as i64, ""),
        "特征维度（全轴最优）",
        1,
        2,
        "x₁ & x₂".to_string(),
        true,
      ),
      (MlLabKind::TreeBoost, "random_forest") => (
        "弱树数量 n_estimators",
        1,
        7,
        fmt_i("", a as i64, ""),
        "单树最大深度",
        1,
        4,
        fmt_i("", b as i64, ""),
        true,
      ),
      (MlLabKind::TreeBoost, "gbdt") => (
        "提升轮数 n_rounds",
        1,
        8,
        fmt_i("", a as i64, ""),
        "学习率 η (%)",
        10,
        100,
        fmt_n("", b as f64 / 100.0, 2, ""),
        false,
      ),
      (MlLabKind::TreeBoost, _) => (
        "提升轮数 n_rounds",
        1,
        8,
        fmt_i("", a as i64, ""),
        "L2 叶权正则 λ",
        0,
        50,
        fmt_n("", b as f64 / 10.0, 1, ""),
        false,
      ),
      (MlLabKind::Boundary2d, "knn") => (
        "邻居数 K",
        1,
        9,
        fmt_i("", a as i64, ""),
        "查询点水平位置 x",
        10,
        90,
        fmt_n("", (b as f64 - 50.0) / 60.0, 2, ""),
        true,
      ),
      (MlLabKind::Boundary2d, "logistic") => (
        "逆正则强度 C (×0.1)",
        2,
        60,
        fmt_n("", a as f64 / 10.0, 1, ""),
        "L2 正则强度",
        1,
        10,
        "L2".to_string(),
        true,
      ),
      (MlLabKind::Boundary2d, _) => (
        "RBF 核带宽 γ (×0.1)",
        2,
        70,
        fmt_n("", a as f64 / 10.0, 1, ""),
        "软间隔惩罚 C (×0.1)",
        2,
        80,
        fmt_n("", b as f64 / 10.0, 1, ""),
        true,
      ),
      (MlLabKind::RegressionFit, _) => (
        "多项式阶数 degree",
        1,
        5,
        fmt_i("", a as i64, ""),
        "岭回归 L2 系数 λ (×0.01)",
        0,
        200,
        fmt_n("", b as f64 / 100.0, 2, ""),
        false,
      ),
      (MlLabKind::ClusterPca, "pca") => (
        "主轴旋转角 θ (°)",
        0,
        165,
        fmt_i("", a as i64, "°"),
        "次轴噪声扩散度",
        8,
        85,
        fmt_n("", b as f64 / 100.0, 2, ""),
        false,
      ),
      (MlLabKind::ClusterPca, _) => (
        "聚类数 K",
        2,
        5,
        fmt_i("", a as i64, ""),
        "Lloyd 迭代步数",
        0,
        6,
        fmt_i("", b as i64, ""),
        false,
      ),
    };

  let mode_tabs: &[(&str, &str)] = match kind {
    MlLabKind::TreeBoost => &[
      ("decision_tree", "CART 决策树"),
      ("random_forest", "随机森林"),
      ("gbdt", "GBDT 残差链"),
      ("xgboost", "XGBoost 二阶正则"),
    ],
    MlLabKind::Boundary2d => {
      &[("logistic", "逻辑回归"), ("svm_rbf", "SVM 间隔/核"), ("knn", "KNN 近邻")]
    }
    MlLabKind::RegressionFit => &[("linear_ridge", "多项式 + 岭回归")],
    MlLabKind::ClusterPca => &[("kmeans", "K-Means 质心迭代"), ("pca", "PCA 主成分投影")],
  };

  rsx! {
    div { class: card_class("p-4 sm:p-5 border-teal-200/80 dark:border-teal-900/70 space-y-4"),
      div { class: "flex flex-wrap items-center justify-between gap-2",
        div { class: "flex items-center gap-2",
          span { class: "px-2 py-0.5 rounded-full bg-teal-600 text-white text-[10px] font-bold uppercase tracking-wider", "Pure Rust WASM 2D Lab" }
          span { class: "text-xs font-bold text-slate-700 dark:text-slate-200", "{kind.badge_label(lang)}" }
        }
        div { class: "flex flex-wrap gap-1.5",
          for &(m_id, m_lbl) in mode_tabs {
            {mode_btn(m_lbl, cur_mode == m_id, EventHandler::new(move |_| {
              mode.set(m_id.to_string());
              let (da, db) = match m_id {
                "decision_tree" => (3, 2),
                "random_forest" => (5, 3),
                "gbdt" => (5, 40),
                "xgboost" => (5, 12),
                "knn" => (5, 55),
                "logistic" => { dataset.set(0); (20, 10) }
                "svm_rbf" => (25, 30),
                "kmeans" => (4, 4),
                "pca" => (35, 24),
                _ => (3, 15),
              };
              param_a.set(da);
              param_b.set(db);
            }))}
          }
        }
      }

      if show_ds {
        div { class: "flex flex-wrap items-center gap-1.5 text-xs",
          span { class: "text-slate-500 dark:text-slate-400 font-semibold mr-1", {lang.pick("二维数据分布：", "2D Dataset:")} }
          for (ds_id, ds_lbl) in [(0u8, "线性可分"), (1u8, "双月牙"), (2u8, "同心圆")] {
            {mode_btn(ds_lbl, cur_ds == ds_id, EventHandler::new(move |_| dataset.set(ds_id)))}
          }
        }
      }

      div { class: "grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs",
        {ml_slider(lbl_a, min_a, max_a, a, &val_a_txt, "text-teal-600 dark:text-teal-400", param_a)}
        {ml_slider(lbl_b, min_b, max_b, b, &val_b_txt, "text-indigo-600 dark:text-indigo-400", param_b)}
      }

      div { class: "w-full overflow-x-auto rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950 p-2",
        svg {
          view_box: "0 0 640 280",
          class: "w-full min-w-[540px] h-auto select-none",
          role: "img",
          "aria-label": "{kind.badge_label(lang)}",
          dangerous_inner_html: "{svg_body}",
        }
      }

      div { class: "grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs",
        {lab_metric_card(m_labels[0], &m_vals[0], "text-teal-600 dark:text-teal-400")}
        {lab_metric_card(m_labels[1], &m_vals[1], "text-indigo-600 dark:text-indigo-400")}
        {lab_metric_card(m_labels[2], &m_vals[2], "text-amber-600 dark:text-amber-400")}
        {lab_metric_card(m_labels[3], &m_vals[3], "text-rose-600 dark:text-rose-400")}
      }
    }
  }
}

/// MDX 嵌入组件：`<MlLab id="svm" />` 或 `<MlLab kind="tree_boost" mode="xgboost" />`。
pub struct MlLabMdx;

impl widgets::MdxComponent for MlLabMdx {
  fn name(&self) -> &'static str {
    "MlLab"
  }

  fn render(&self, props: &HashMap<String, String>) -> Element {
    let id = props.get("id").or_else(|| props.get("slug")).map(String::as_str).unwrap_or("");
    let kind_raw = props.get("kind").map(String::as_str).unwrap_or("");
    let mode_raw = props.get("mode").map(String::as_str).unwrap_or("");
    let (kind, mode) = infer_ml_lab_by_slug(id, kind_raw, mode_raw);
    rsx! {
      div { class: "my-6 not-prose",
        MlLabWorkbench { kind, initial_mode: mode }
      }
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn all_four_ml_kernels_compute_valid_svg_outputs() {
    let (xgb_svg, xgb_m) = render_boosting_1d(true, 5, 45, 10);
    assert!(xgb_svg.contains("<polyline"));
    assert!(xgb_m[0].contains('5'));

    let (svm_svg, svm_m) = render_boundary_2d("svm_rbf", 1, 25, 30);
    assert!(svm_svg.contains("RBF"));
    assert!(svm_m[0].contains("SV"));

    let (reg_svg, reg_m) = render_regression_fit(3, 15);
    assert!(reg_svg.contains("w*"));
    assert_eq!(reg_m[0], "deg = 3");

    let (km_svg, _) = render_cluster_pca("kmeans", 4, 4);
    assert!(km_svg.contains("WCSS"));

    let (pca_svg, pca_m) = render_cluster_pca("pca", 35, 24);
    assert!(pca_svg.contains("PC1"));
    assert!(pca_m[0].ends_with('%'));
  }
}
