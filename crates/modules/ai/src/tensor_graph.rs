//! 全网张量维流图（Tensor Flow DAG）引擎与可视化组件。
//!
//! 采用纯 Rust 确定性整数网格布局（零泛型膨胀、零 `HashMap`/`Cow` 开销、零第三方 JS 依赖、100% 兼容无 `'unsafe-eval'` CSP）：
//! 1. 从任意 [`ModelBlueprint`] 自动推导端到端多分支张量维度流（覆盖 LLM / VLM / ASR / TTS / Diffusion / SSM / Embedding）；
//! 2. 符号维度（`B×S`）与实数维度（`1×1` / `1×512` / `4×4096`）一键切换实时重算；
//! 3. 内置经典 CV 静态表驱动教学对照（Inception-v3 8 色完整图例 + 4 分支拼接、ResNet-50 残差瓶颈）；
//! 4. 点击任意色块查看算子变换公式并一键切回分层工作台对应算子子图；
//! 5. 零 JS 导出独立高清 `.svg` 矢量图。

use dioxus::prelude::*;
use dioxus_shadcn::card_class;
#[cfg(any(feature = "server", test))]
use serde::{Deserialize, Serialize};

use crate::arch::{
  format_count, AttentionSpec, FfnSpec, InspectorFocus, LayerSegment, ModelBlueprint,
  OutputHeadSpec,
};

#[inline(never)]
fn push_u64(out: &mut String, mut n: u64) {
  if n == 0 {
    out.push('0');
    return;
  }
  let mut buf = [0u8; 20];
  let mut i = buf.len();
  while n > 0 {
    i -= 1;
    buf[i] = b'0' + (n % 10) as u8;
    n /= 10;
  }
  if let Ok(s) = std::str::from_utf8(&buf[i..]) {
    out.push_str(s);
  }
}

#[inline(never)]
fn push_i32(out: &mut String, n: i32) {
  if n < 0 {
    out.push('-');
    push_u64(out, n.unsigned_abs() as u64);
  } else {
    push_u64(out, n as u64);
  }
}

/// 算子着色类别（与右上角图例一一对应）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(any(feature = "server", test), derive(Serialize, Deserialize))]
#[cfg_attr(any(feature = "server", test), serde(rename_all = "snake_case"))]
#[repr(u8)]
pub enum OpKind {
  Input = 0,
  Embedding = 1,
  #[default]
  ConvOrLinear = 2,
  AvgPool = 3,
  MaxPool = 4,
  Attention = 5,
  MoeRouter = 6,
  Merge = 7,
  Softmax = 8,
  FcHead = 9,
  Dropout = 10,
}

impl OpKind {
  fn from_u8(v: u8) -> Self {
    match v {
      0 => Self::Input,
      1 => Self::Embedding,
      3 => Self::AvgPool,
      4 => Self::MaxPool,
      5 => Self::Attention,
      6 => Self::MoeRouter,
      7 => Self::Merge,
      8 => Self::Softmax,
      9 => Self::FcHead,
      10 => Self::Dropout,
      _ => Self::ConvOrLinear,
    }
  }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(any(feature = "server", test), derive(Serialize, Deserialize))]
#[cfg_attr(any(feature = "server", test), serde(rename_all = "snake_case"))]
#[repr(u8)]
pub enum LegendStyle {
  LlmOrMultimodal = 0,
  #[default]
  ClassicCnn = 1,
}

impl LegendStyle {
  fn from_u8(v: u8) -> Self {
    if v == 0 {
      Self::LlmOrMultimodal
    } else {
      Self::ClassicCnn
    }
  }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShapePreset {
  Symbolic,
  Decode1,
  Prefill512,
  Batch4K,
}

impl ShapePreset {
  pub fn label(self) -> &'static str {
    match self {
      Self::Symbolic => "符号 (B×S)",
      Self::Decode1 => "Decode (1×1)",
      Self::Prefill512 => "Prefill (1×512)",
      Self::Batch4K => "Batch (4×4096)",
    }
  }

  fn seq_prefix(self) -> &'static str {
    match self {
      Self::Symbolic => "B×S",
      Self::Decode1 => "1×1",
      Self::Prefill512 => "1×512",
      Self::Batch4K => "4×4096",
    }
  }

  fn batch_str(self) -> &'static str {
    match self {
      Self::Symbolic => "B",
      Self::Decode1 | Self::Prefill512 => "1",
      Self::Batch4K => "4",
    }
  }

  fn seq_str(self) -> &'static str {
    match self {
      Self::Symbolic => "S",
      Self::Decode1 => "1",
      Self::Prefill512 => "512",
      Self::Batch4K => "4096",
    }
  }

  #[inline(never)]
  fn fmt_seq(self, dim: u64) -> String {
    let mut s = String::with_capacity(16);
    s.push_str(self.seq_prefix());
    s.push('×');
    push_u64(&mut s, dim);
    s
  }

  #[inline(never)]
  fn fmt_batch(self, suffix: &str) -> String {
    let mut s = String::with_capacity(suffix.len() + 4);
    s.push_str(self.batch_str());
    s.push('×');
    s.push_str(suffix);
    s
  }

  #[inline(never)]
  fn fmt_batch_u64(self, n: u64, suffix: &str) -> String {
    let mut s = String::with_capacity(suffix.len() + 12);
    s.push_str(self.batch_str());
    s.push('×');
    push_u64(&mut s, n);
    s.push_str(suffix);
    s
  }

  #[inline(never)]
  fn fmt_heads(self, heads: u64, head_dim: u64) -> String {
    let mut s = String::with_capacity(20);
    s.push_str(self.batch_str());
    s.push('×');
    push_u64(&mut s, heads);
    s.push('×');
    s.push_str(self.seq_str());
    s.push('×');
    push_u64(&mut s, head_dim);
    s
  }

  #[inline(never)]
  fn fmt_cv(self, h: u64, w: u64, c: u64) -> String {
    let mut s = String::with_capacity(18);
    if h == 0 && w == 0 {
      s.push_str(self.batch_str());
      s.push('×');
      push_u64(&mut s, c);
    } else if matches!(self, Self::Symbolic) {
      push_u64(&mut s, h);
      s.push('*');
      push_u64(&mut s, w);
      s.push('*');
      push_u64(&mut s, c);
    } else {
      s.push_str(self.batch_str());
      s.push('×');
      push_u64(&mut s, h);
      s.push('×');
      push_u64(&mut s, w);
      s.push('×');
      push_u64(&mut s, c);
    }
    s
  }
}

impl OpKind {
  pub fn palette(self) -> (&'static str, &'static str, &'static str) {
    match self {
      Self::Input => ("#f472b6", "#db2777", "#0f172a"),
      Self::Embedding => ("#fb7185", "#e11d48", "#0f172a"),
      Self::ConvOrLinear => ("#f97316", "#c2410c", "#ffffff"),
      Self::AvgPool => ("#0ea5e9", "#0369a1", "#ffffff"),
      Self::MaxPool => ("#16a34a", "#15803d", "#ffffff"),
      Self::Attention => ("#4f46e5", "#3730a3", "#ffffff"),
      Self::MoeRouter => ("#facc15", "#ca8a04", "#0f172a"),
      Self::Merge => ("#dc2626", "#991b1b", "#ffffff"),
      Self::Softmax => ("#22d3ee", "#0891b2", "#0f172a"),
      Self::FcHead => ("#a21caf", "#701a75", "#ffffff"),
      Self::Dropout => ("#1e3a8a", "#172554", "#ffffff"),
    }
  }

  pub fn legend_label(self, style: LegendStyle) -> &'static str {
    match (self, style) {
      (Self::Input, LegendStyle::ClassicCnn) => "输入图像",
      (Self::Input, LegendStyle::LlmOrMultimodal) => "输入张量",
      (Self::Embedding, _) => "Embedding / 切片",
      (Self::ConvOrLinear, LegendStyle::ClassicCnn) => "卷积层",
      (Self::ConvOrLinear, LegendStyle::LlmOrMultimodal) => "线性投影 / 卷积",
      (Self::AvgPool, LegendStyle::ClassicCnn) => "平均池化层",
      (Self::AvgPool, LegendStyle::LlmOrMultimodal) => "潜空间压缩 / 池化",
      (Self::MaxPool, LegendStyle::ClassicCnn) => "最大池化层",
      (Self::MaxPool, LegendStyle::LlmOrMultimodal) => "归一化 / 激活门控",
      (Self::Attention, _) => "注意力 / SSM 混合",
      (Self::MoeRouter, _) => "MoE 路由 / 专家",
      (Self::Merge, LegendStyle::ClassicCnn) => "拼接 / 残差相加",
      (Self::Merge, LegendStyle::LlmOrMultimodal) => "残差相加 / 拼接融合",
      (Self::Softmax, _) => "Softmax / 归一化输出",
      (Self::FcHead, LegendStyle::ClassicCnn) => "全连接层",
      (Self::FcHead, LegendStyle::LlmOrMultimodal) => "输出头 / 解码投影",
      (Self::Dropout, _) => "Dropout",
    }
  }

  const ALL_ORDER: [Self; 11] = [
    Self::Input,
    Self::Embedding,
    Self::AvgPool,
    Self::ConvOrLinear,
    Self::Attention,
    Self::MoeRouter,
    Self::Merge,
    Self::Softmax,
    Self::FcHead,
    Self::MaxPool,
    Self::Dropout,
  ];
}

/// 连线语义类别（用于区分主数据流、层内残差、跨块长跳连与旁路/控制流）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(any(feature = "server", test), derive(Serialize, Deserialize))]
#[cfg_attr(any(feature = "server", test), serde(rename_all = "snake_case"))]
#[repr(u8)]
pub enum EdgeKind {
  /// 主张量数据流（深灰实线）
  #[default]
  Main = 0,
  /// 单层子块内部残差跳连（画在 RepeatGroup 大框内部左轨，靛蓝虚线）
  InnerSkip = 1,
  /// 跨块 / 外部长跳连（画在画布最左侧大框外部轨道，翠绿虚线）
  MacroSkip = 2,
  /// 旁路 / 路由调度 / 辅助预测分支（如 k_pe 旁路、MoE dispatch、MTP，琥珀橙细虚线）
  AuxBypass = 3,
}

impl EdgeKind {
  fn from_u8(v: u8) -> Self {
    match v {
      1 => Self::InnerSkip,
      2 => Self::MacroSkip,
      3 => Self::AuxBypass,
      _ => Self::Main,
    }
  }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorNode {
  pub kind: OpKind,
  pub shape: String,
  pub side_note: String,
  pub op_title: String,
  pub formula: String,
  pub params_hint: String,
  pub col_q: i32,
  pub row: usize,
  pub focus_target: Option<InspectorFocus>,
  pub layer_target: Option<u64>,
  /// 若为 `Some(seg_idx)`，表示该节点是收起态分段汇总卡，点击可直接展开该分段。
  pub toggle_seg: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorEdge {
  pub from: usize,
  pub to: usize,
  pub label: String,
  pub kind: EdgeKind,
  /// 所属 `RepeatGroup` 索引（`u8::MAX` 表示顶层非分组边）。
  pub group_idx: u8,
  /// 多泳道轨道编号（`0` 为默认紧贴轨道，`1..=7` 逐级向外错开 20px，支撑 U-Net / YOLO 多尺度跨层跳连）。
  pub lane: u8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepeatGroup {
  /// 对应模型分段索引（`Some(i)` 支持点击展开/收起；`None` 为静态教学框）。
  pub seg_idx: Option<usize>,
  pub collapsed: bool,
  pub row_start: usize,
  pub row_end: usize,
  /// Attention/Mixer 与 FFN/MoE 子块之间的分隔行（收起或单子块时为 `None`）。
  pub split_after_row: Option<usize>,
  pub min_col_q: i32,
  pub max_col_q: i32,
  pub badge: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorGraph {
  pub file_slug: String,
  pub title: String,
  pub subtitle: String,
  pub legend_style: LegendStyle,
  pub nodes: Vec<TensorNode>,
  pub edges: Vec<TensorEdge>,
  pub groups: Vec<RepeatGroup>,
  pub total_rows: usize,
}

impl TensorGraph {
  #[inline(never)]
  pub fn active_legend(&self) -> Vec<(OpKind, &'static str)> {
    let mut mask: u16 = 0;
    for n in &self.nodes {
      mask |= 1u16 << (n.kind as u8);
    }
    let mut out = Vec::with_capacity(8);
    for k in OpKind::ALL_ORDER {
      if (mask & (1u16 << (k as u8))) != 0 {
        out.push((k, k.legend_label(self.legend_style)));
      }
    }
    out
  }

  #[inline(never)]
  pub fn active_edge_mask(&self) -> u8 {
    let mut mask = 0u8;
    for e in &self.edges {
      mask |= 1u8 << (e.kind as u8);
    }
    mask
  }
}

struct GraphBuilder {
  nodes: Vec<TensorNode>,
  edges: Vec<TensorEdge>,
  groups: Vec<RepeatGroup>,
  focus: Option<InspectorFocus>,
  layer: Option<u64>,
  cur_group: u8,
}

impl GraphBuilder {
  fn new() -> Self {
    Self {
      nodes: Vec::new(),
      edges: Vec::new(),
      groups: Vec::new(),
      focus: None,
      layer: None,
      cur_group: u8::MAX,
    }
  }

  fn set_ctx(&mut self, focus: Option<InspectorFocus>, layer: Option<u64>) {
    self.focus = focus;
    self.layer = layer;
  }

  /// 紧凑无泛型节点构造（复用 `focus`/`layer` 上下文，缩减 WASM 调用参数槽位）。
  #[inline(never)]
  #[allow(clippy::too_many_arguments)]
  fn add(
    &mut self,
    kind: OpKind,
    shape: &str,
    side_note: &str,
    op_title: &'static str,
    formula: &str,
    params_hint: &str,
    col_q: i32,
    row: usize,
  ) -> usize {
    let idx = self.nodes.len();
    self.nodes.push(TensorNode {
      kind,
      shape: shape.to_string(),
      side_note: side_note.to_string(),
      op_title: op_title.to_string(),
      formula: formula.to_string(),
      params_hint: params_hint.to_string(),
      col_q,
      row,
      focus_target: self.focus,
      layer_target: self.layer,
      toggle_seg: None,
    });
    idx
  }

  #[inline(never)]
  #[allow(clippy::too_many_arguments)]
  fn add_norm(
    &mut self,
    prev: usize,
    row: usize,
    d: u64,
    seq_d: &str,
    side: &str,
    title: &'static str,
    hint: &str,
  ) -> usize {
    let saved = self.focus;
    self.focus = Some(InspectorFocus::NormAndResidual);
    let idx =
      self.add(OpKind::MaxPool, seq_d, side, title, &fmt_num("RMSNorm(dim=", d, ")"), hint, 0, row);
    self.focus = saved;
    self.edge(prev, idx);
    idx
  }

  #[inline(never)]
  #[allow(clippy::too_many_arguments)]
  fn add_residual(
    &mut self,
    from: usize,
    skip_from: usize,
    row: usize,
    seq_d: &str,
    skip_label: &'static str,
    title: &'static str,
    formula: &str,
    hint: &str,
  ) -> usize {
    let saved = self.focus;
    self.focus = Some(InspectorFocus::NormAndResidual);
    let idx = self.add(OpKind::Merge, seq_d, "Add (+)", title, formula, hint, 0, row);
    self.focus = saved;
    self.edge(from, idx);
    self.edge_inner_skip(skip_from, idx, skip_label);
    idx
  }

  #[inline(never)]
  #[allow(clippy::too_many_arguments)]
  fn add_proj(
    &mut self,
    kind: OpKind,
    shape: &str,
    side: &str,
    title: &'static str,
    in_d: u64,
    out_d: u64,
    hint: &str,
    col_q: i32,
    row: usize,
  ) -> usize {
    let f = fmt_proj(in_d, out_d);
    self.add(kind, shape, side, title, &f, hint, col_q, row)
  }

  fn edge(&mut self, from: usize, to: usize) {
    self.edges.push(TensorEdge {
      from,
      to,
      label: String::new(),
      kind: EdgeKind::Main,
      group_idx: self.cur_group,
      lane: 0,
    });
  }

  fn edge_inner_skip(&mut self, from: usize, to: usize, label: &'static str) {
    self.edges.push(TensorEdge {
      from,
      to,
      label: label.to_string(),
      kind: EdgeKind::InnerSkip,
      group_idx: self.cur_group,
      lane: 0,
    });
  }

  fn edge_aux(&mut self, from: usize, to: usize, label: &'static str) {
    self.edges.push(TensorEdge {
      from,
      to,
      label: label.to_string(),
      kind: EdgeKind::AuxBypass,
      group_idx: self.cur_group,
      lane: 0,
    });
  }

  #[allow(clippy::too_many_arguments)]
  fn group(
    &mut self,
    seg_idx: Option<usize>,
    collapsed: bool,
    row_start: usize,
    row_end: usize,
    split_after_row: Option<usize>,
    min_col_q: i32,
    max_col_q: i32,
    badge: String,
  ) {
    self.groups.push(RepeatGroup {
      seg_idx,
      collapsed,
      row_start,
      row_end,
      split_after_row,
      min_col_q,
      max_col_q,
      badge,
    });
  }

  fn finish(
    self,
    file_slug: String,
    title: String,
    subtitle: String,
    legend_style: LegendStyle,
  ) -> TensorGraph {
    let total_rows = self.nodes.iter().map(|n| n.row + 1).max().unwrap_or(1);
    TensorGraph {
      file_slug,
      title,
      subtitle,
      legend_style,
      nodes: self.nodes,
      edges: self.edges,
      groups: self.groups,
      total_rows,
    }
  }
}

#[inline(never)]
fn fmt_proj(in_d: u64, out_d: u64) -> String {
  let mut s = String::with_capacity(24);
  s.push_str("Linear [");
  push_u64(&mut s, in_d);
  s.push_str(" → ");
  push_u64(&mut s, out_d);
  s.push(']');
  s
}

#[inline(never)]
fn fmt_pair(a: u64, sep: &str, b: u64) -> String {
  let mut s = String::with_capacity(16);
  push_u64(&mut s, a);
  s.push_str(sep);
  push_u64(&mut s, b);
  s
}

#[inline(never)]
fn fmt_param(prefix: &str, count: u64) -> String {
  let mut s = String::with_capacity(prefix.len() + 12);
  s.push_str(prefix);
  s.push_str(&format_count(count));
  s
}

#[inline(never)]
fn fmt_num(prefix: &str, n: u64, suffix: &str) -> String {
  let mut s = String::with_capacity(prefix.len() + suffix.len() + 10);
  s.push_str(prefix);
  push_u64(&mut s, n);
  s.push_str(suffix);
  s
}

/// 从任意 [`ModelBlueprint`] 推导全网多分支张量维流图（默认全部展开）。
#[inline(never)]
pub fn build_blueprint_tensor_graph(
  bp: &ModelBlueprint,
  preset: ShapePreset,
  compact: bool,
) -> TensorGraph {
  build_blueprint_tensor_graph_with_collapse(bp, preset, compact, 0)
}

/// 从任意 [`ModelBlueprint`] 推导全网多分支张量维流图，支持按位掩码 `collapsed_mask` 折叠指定分段大块。
#[inline(never)]
pub fn build_blueprint_tensor_graph_with_collapse(
  bp: &ModelBlueprint,
  preset: ShapePreset,
  compact: bool,
  collapsed_mask: usize,
) -> TensorGraph {
  let mut g = GraphBuilder::new();
  let d = bp.hidden_size;
  let v = bp.vocab_size;
  let seq_d = preset.fmt_seq(d);
  let seq_v = preset.fmt_seq(v);
  let mut row: usize = 0;

  g.set_ctx(Some(InspectorFocus::Embedding), None);
  let mut cur = if let Some(ref diff) = bp.diffusion {
    let cond_n = diff.components.iter().filter(|c| c.role == "conditioner").count().max(1) as u64;
    let p_in = g.add(
      OpKind::Input,
      &preset.fmt_batch("77"),
      "prompt",
      "Prompt Token Input",
      &fmt_num("Prompt → ", cond_n, " Text Conditioner(s)"),
      &fmt_param("Conditioner 总参: ", bp.params.embedding),
      -3,
      row,
    );
    let z_in = g.add(
      OpKind::Input,
      &truncate_label(&diff.default_latent_shape, 18),
      "z_t noise",
      "Noisy Latent Tensor",
      &diff.default_latent_shape,
      &diff.scheduler_summary,
      3,
      row,
    );
    row += 1;
    let c_out = g.add(
      OpKind::Embedding,
      &seq_d,
      &fmt_num("×", cond_n, " Enc"),
      "Text Conditioners Projection",
      "CLIP / T5 / UMT5 → context_embed",
      "单次前向编码，跨去噪步复用",
      -3,
      row,
    );
    let z_out = g.add(
      OpKind::ConvOrLinear,
      &seq_d,
      &truncate_label(&diff.patch_summary, 11),
      "Latent Patchify / Conv Embed",
      &diff.patch_summary,
      &diff.denoiser_kind,
      3,
      row,
    );
    g.edge(p_in, c_out);
    g.edge(z_in, z_out);
    row += 1;
    g.set_ctx(Some(InspectorFocus::Attention), Some(0));
    let fused = g.add(
      OpKind::Merge,
      &seq_d,
      "Joint/Cross",
      "Condition + Latent Stream Fusion",
      "MMDiT Joint Sequence / Cross-Attn Conditioning",
      &fmt_num("Backbone: ", bp.num_hidden_layers, " Layers"),
      0,
      row,
    );
    g.edge(c_out, fused);
    g.edge(z_out, fused);
    fused
  } else if bp.vision.is_some() || bp.audio.is_some() {
    let mut modal_outs: Vec<usize> = Vec::new();
    let has_both = bp.vision.is_some() && bp.audio.is_some();
    let vis_col = if has_both { -5 } else { -3 };
    let aud_col = if has_both { 0 } else { -3 };
    let txt_col = if has_both { 5 } else { 3 };

    if let Some(ref vis) = bp.vision {
      g.set_ctx(Some(InspectorFocus::Vision), None);
      let img_dim = vis.image_size.unwrap_or(384);
      let patch = vis.patch_size.unwrap_or(14);
      let n_p = (img_dim / patch.max(1)).pow(2);
      let v_in = g.add(
        OpKind::Input,
        &preset.fmt_cv(img_dim, img_dim, 3),
        "RGB",
        "Vision Image Input",
        &fmt_num("RGB Image ", img_dim, "px"),
        &vis.model_type,
        vis_col,
        row,
      );
      let v_enc = g.add(
        OpKind::ConvOrLinear,
        &preset.fmt_batch(&fmt_pair(n_p, "×", vis.hidden_size)),
        &fmt_num("ViT ×", vis.num_hidden_layers, "L"),
        "Vision Patchify + ViT Encoder",
        &fmt_num("Patchify + ", vis.num_hidden_layers, "L ViT"),
        &fmt_param("视觉塔参数: ", vis.estimated_params),
        vis_col,
        row + 1,
      );
      let v_proj = g.add_proj(
        OpKind::AvgPool,
        &preset.fmt_batch(&fmt_num("N_v×", d, "")),
        &fmt_pair(vis.hidden_size, "→", d),
        "Vision Spatial Merger & Projector",
        vis.hidden_size,
        d,
        "对齐至主干隐藏维度",
        vis_col,
        row + 2,
      );
      g.edge(v_in, v_enc);
      g.edge(v_enc, v_proj);
      modal_outs.push(v_proj);
    }

    if let Some(ref aud) = bp.audio {
      g.set_ctx(Some(InspectorFocus::Audio), None);
      let a_in = g.add(
        OpKind::Input,
        &preset.fmt_batch_u64(aud.num_mel_bins, "×T"),
        &fmt_num("", aud.sampling_rate_hz / 1000, "kHz"),
        "Log-Mel Spectrogram Input",
        &fmt_num("", aud.num_mel_bins, " Mel Bins"),
        &aud.model_type,
        aud_col,
        row,
      );
      let a_enc = g.add(
        OpKind::ConvOrLinear,
        &preset.fmt_batch(&fmt_num("T/2×", aud.hidden_size, "")),
        &fmt_num("Conv ×", aud.num_layers, "L"),
        "Audio Conv1d Stem + Encoder",
        &fmt_num("Conv1d(s=2) + ", aud.num_layers, "L Audio Encoder"),
        &fmt_param("音频塔参数: ", aud.estimated_params),
        aud_col,
        row + 1,
      );
      let a_proj = g.add_proj(
        OpKind::AvgPool,
        &preset.fmt_batch(&fmt_num("T_a×", d, "")),
        &fmt_pair(aud.hidden_size, "→", d),
        "Audio Temporal Pool & Projector",
        aud.hidden_size,
        d,
        "映射至语言主干隐空间",
        aud_col,
        row + 2,
      );
      g.edge(a_in, a_enc);
      g.edge(a_enc, a_proj);
      modal_outs.push(a_proj);
    }

    g.set_ctx(Some(InspectorFocus::Embedding), None);
    let t_in = g.add(
      OpKind::Input,
      preset.seq_prefix(),
      "int64",
      "Text input_ids",
      &fmt_param("Vocab = ", v),
      "文本 Token 序列输入",
      txt_col,
      row,
    );
    let t_emb = g.add_proj(
      OpKind::Embedding,
      &seq_d,
      "embed",
      "Token Embedding Lookup",
      v,
      d,
      &fmt_param("Embedding 参数: ", bp.params.embedding),
      txt_col,
      row + 2,
    );
    g.edge(t_in, t_emb);

    row += 3;
    let fused = g.add(
      OpKind::Merge,
      &seq_d,
      "Concat",
      "Multimodal Token Sequence Concat",
      "Concat(Modal Tokens, Text Embeds)",
      "多模态序列统一送入主干层栈",
      0,
      row,
    );
    for m_idx in modal_outs {
      g.edge(m_idx, fused);
    }
    g.edge(t_emb, fused);
    fused
  } else {
    let t_in = g.add(
      OpKind::Input,
      preset.seq_prefix(),
      "input_ids",
      "Token IDs Input",
      &fmt_param("Token IDs · Vocab = ", v),
      &fmt_num("上下文窗口: ", bp.max_position_embeddings / 1024, "K tokens"),
      0,
      row,
    );
    row += 1;
    let t_emb = g.add_proj(
      OpKind::Embedding,
      &seq_d,
      &fmt_pair(v, "→", d),
      "Token Embedding (embed_tokens)",
      v,
      d,
      &fmt_param("Embedding 参数: ", bp.params.embedding),
      0,
      row,
    );
    g.edge(t_in, t_emb);
    t_emb
  };

  for (seg_idx, seg) in bp.segments.iter().enumerate() {
    row += 1;
    let seg_start = row;
    let mut badge = String::with_capacity(48);
    badge.push_str(&seg.range_label());
    badge.push_str(" · ");
    badge.push_str(&seg.title);
    let grp_id = g.groups.len() as u8;
    let is_collapsed = (collapsed_mask & (1usize << seg_idx)) != 0;

    if is_collapsed {
      let attn_p = seg.attention.params_per_layer(d);
      let (ffn_tot, ffn_act) = seg.ffn.params_per_layer(d);
      let layer_n = seg.layer_count();
      let is_moe = matches!(seg.ffn, FfnSpec::SparseMoe { .. });
      let mut side = String::with_capacity(20);
      push_u64(&mut side, layer_n);
      side.push_str("L · 已收起");
      let mut formula = String::with_capacity(64);
      formula.push_str(match &seg.attention {
        AttentionSpec::DeepSeekMla { .. } => "MLA",
        AttentionSpec::StandardGqa { .. } => "GQA / Attention",
        AttentionSpec::LinearRecurrent { .. } => "Linear / SSM",
      });
      formula.push_str(" + ");
      formula.push_str(if is_moe { "Sparse MoE" } else { "Dense FFN" });
      formula.push_str(" × ");
      push_u64(&mut formula, layer_n);
      formula.push_str(" 层（点击标题栏或本色块展开内部维流）");
      let mut hint = fmt_param("本段总参: ", (attn_p + ffn_tot) * layer_n);
      hint.push_str(" · 单层激活: ");
      hint.push_str(&format_count(attn_p + ffn_act));

      g.set_ctx(Some(InspectorFocus::Attention), Some(seg.start_layer));
      let summary_idx = g.add(
        if is_moe { OpKind::MoeRouter } else { OpKind::Attention },
        &seq_d,
        &side,
        "已收起主干分段（点击展开全部算子流）",
        &formula,
        &hint,
        0,
        seg_start,
      );
      if let Some(node) = g.nodes.get_mut(summary_idx) {
        node.toggle_seg = Some(seg_idx);
      }
      g.edge(cur, summary_idx);
      g.group(Some(seg_idx), true, seg_start, seg_start, None, -3, 3, badge);
      cur = summary_idx;
      row = seg_start;
    } else {
      g.cur_group = grp_id;
      let (next_idx, next_row, split_row, min_q, max_q) =
        append_segment_subgraph(&mut g, bp, seg, cur, row, preset, compact, &seq_d);
      g.cur_group = u8::MAX;
      g.group(Some(seg_idx), false, seg_start, next_row, split_row, min_q, max_q, badge);
      cur = next_idx;
      row = next_row;
    }
  }

  row += 1;
  g.set_ctx(Some(InspectorFocus::LmHead), None);
  if !compact {
    cur = g.add_norm(
      cur,
      row,
      d,
      &seq_d,
      "FinalNorm",
      "Final RMSNorm / LayerNorm",
      &fmt_param("参数量: ", d),
    );
    row += 1;
  }

  let has_mtp = bp.num_nextn_predict_layers > 0;
  let head_col = if has_mtp { -3 } else { 0 };

  match &bp.output_head {
    OutputHeadSpec::CausalLmHead => {
      let tied_hint = if bp.tie_word_embeddings {
        "LM Head 与 Embedding 共享权重".to_string()
      } else {
        fmt_param("LM Head 参数: ", bp.params.lm_head)
      };
      let lm = g.add_proj(
        OpKind::FcHead,
        &seq_v,
        &fmt_pair(d, "→", v),
        "Output Projection (lm_head)",
        d,
        v,
        &tied_hint,
        head_col,
        row,
      );
      g.edge(cur, lm);
      let sm = g.add(
        OpKind::Softmax,
        &seq_v,
        "probs",
        "Vocab Softmax / Sampling",
        "Softmax(logits / T) → Next-Token Distribution",
        "Top-p / Top-k 采样输出下一 Token",
        head_col,
        row + 1,
      );
      g.edge(lm, sm);

      if has_mtp {
        g.set_ctx(Some(InspectorFocus::MtpHead), None);
        let mtp = g.add_proj(
          OpKind::FcHead,
          &seq_d,
          &fmt_num("MTP ×", bp.num_nextn_predict_layers, ""),
          "Multi-Token Prediction (MTP)",
          2 * d,
          d,
          &fmt_param("Concat [h_t, Embed(t+1)] → 轻量层 · 额外参数: +", bp.params.mtp_head),
          3,
          row,
        );
        let mtp_out = g.add_proj(
          OpKind::Softmax,
          &seq_v,
          "t+2..k",
          "MTP Shared LM Head Draft",
          d,
          v,
          "复用主干 lm_head 进行多 Token 投机验证",
          3,
          row + 1,
        );
        g.edge_aux(cur, mtp, "MTP");
        g.edge(mtp, mtp_out);
      }
    }
    OutputHeadSpec::EmbeddingPool { pooling_mode, output_dim, normalize } => {
      let out_s = preset.fmt_batch_u64(*output_dim, "");
      let pool = g.add(
        OpKind::AvgPool,
        &out_s,
        &truncate_label(pooling_mode, 10),
        "Sequence Pooling Head",
        pooling_mode,
        "单次前向编码，0 B KV Cache",
        0,
        row,
      );
      g.edge(cur, pool);
      if *normalize {
        let l2 = g.add(
          OpKind::Softmax,
          &out_s,
          "L2 Norm",
          "L2 Vector Normalization",
          "v / ||v||_2 → Unit Embedding",
          "用于余弦相似度向量检索",
          0,
          row + 1,
        );
        g.edge(pool, l2);
      }
    }
    OutputHeadSpec::SequenceScorer { num_labels } => {
      let out_s = preset.fmt_batch_u64(*num_labels, "");
      let cls = g.add_proj(
        OpKind::FcHead,
        &out_s,
        &fmt_pair(d, "→", *num_labels),
        "Cross-Encoder Scorer Head",
        d,
        *num_labels,
        &fmt_param("Head 参数: ", bp.params.lm_head),
        0,
        row,
      );
      let sm = g.add(
        OpKind::Softmax,
        &out_s,
        "score",
        "Relevance Probability Output",
        "Sigmoid / Softmax → Relevance Scores",
        "用于 RAG 重排序打分",
        0,
        row + 1,
      );
      g.edge(cur, cls);
      g.edge(cls, sm);
    }
    OutputHeadSpec::SpeechSynthesizer {
      codec_summary,
      num_codebooks,
      sampling_rate_hz,
      estimated_params,
      ..
    } => {
      let tk = g.add(
        OpKind::MoeRouter,
        &preset.fmt_batch_u64(*num_codebooks, "×T"),
        &fmt_num("", *num_codebooks, " RVQ"),
        "Acoustic Codebook Predictor",
        &fmt_num("Predict ", *num_codebooks, " RVQ Codebooks"),
        &fmt_param("声学后端参数: ", *estimated_params),
        0,
        row,
      );
      let voc = g.add(
        OpKind::FcHead,
        &preset.fmt_batch("T_wav"),
        &fmt_num("", sampling_rate_hz / 1000, "kHz"),
        "Neural Vocoder",
        codec_summary,
        "端到端还原高保真音频波形",
        0,
        row + 1,
      );
      g.edge(cur, tk);
      g.edge(tk, voc);
    }
    OutputHeadSpec::LatentDecoder {
      vae_class,
      latent_channels,
      spatial_compression,
      is_video,
      estimated_params,
      ..
    } => {
      let unp = g.add(
        OpKind::AvgPool,
        &preset.fmt_batch_u64(*latent_channels, "×h×w"),
        "Unpatch",
        "Latent Unpatchify",
        &fmt_num("Unpatchify · ×", *spatial_compression, " Spatial"),
        "还原潜空间网格张量",
        0,
        row,
      );
      let vae = g.add(
        OpKind::FcHead,
        &preset.fmt_batch(if *is_video { "F×3×H×W" } else { "3×H×W" }),
        &truncate_label(vae_class, 11),
        "Latent VAE Decoder",
        vae_class,
        &fmt_param("VAE 参数: ", *estimated_params),
        0,
        row + 1,
      );
      g.edge(cur, unp);
      g.edge(unp, vae);
    }
  }

  let slug = bp.repo_id.replace('/', "-").to_lowercase();
  let mut title = String::with_capacity(bp.repo_id.len() + 24);
  title.push_str(&bp.repo_id);
  title.push_str(" · 全网张量维流图");
  let mut subtitle = String::with_capacity(96);
  subtitle.push_str(bp.family.badge_label());
  subtitle.push_str(" · ");
  push_u64(&mut subtitle, bp.num_hidden_layers);
  subtitle.push_str("L × ");
  push_u64(&mut subtitle, d);
  subtitle.push_str("d · 总参 ");
  subtitle.push_str(&format_count(bp.params.total_params));
  subtitle.push_str(" (激活 ");
  subtitle.push_str(&format_count(bp.params.active_params));
  subtitle.push_str(") · ");
  subtitle.push_str(preset.label());
  g.finish(slug, title, subtitle, LegendStyle::LlmOrMultimodal)
}

#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn append_segment_subgraph(
  g: &mut GraphBuilder,
  bp: &ModelBlueprint,
  seg: &LayerSegment,
  prev: usize,
  start_row: usize,
  preset: ShapePreset,
  compact: bool,
  seq_d: &str,
) -> (usize, usize, Option<usize>, i32, i32) {
  let d = bp.hidden_size;
  let layer = seg.start_layer;
  let mut row = start_row;
  let mut min_q: i32 = -3;
  let mut max_q: i32 = 3;

  g.set_ctx(Some(InspectorFocus::Attention), Some(layer));
  let attn_in = if compact {
    prev
  } else {
    let n = g.add_norm(
      prev,
      row,
      d,
      seq_d,
      "Pre-Norm",
      "input_layernorm (RMSNorm)",
      &fmt_param("Norm 参数: ", d),
    );
    row += 1;
    n
  };

  let attn_params = seg.attention.params_per_layer(d);
  let attn_add = match &seg.attention {
    AttentionSpec::DeepSeekMla {
      num_heads,
      q_lora_rank,
      kv_lora_rank,
      qk_nope_head_dim,
      qk_rope_head_dim,
      v_head_dim,
    } => {
      let qk_dim = qk_nope_head_dim + qk_rope_head_dim;
      let kv_a_dim = kv_lora_rank + qk_rope_head_dim;
      let kv_b_head = qk_nope_head_dim + v_head_dim;
      let o_in = num_heads * v_head_dim;

      let q_leaf = if let Some(qr) = q_lora_rank {
        let qa = g.add_proj(
          OpKind::ConvOrLinear,
          &preset.fmt_seq(*qr),
          "q_a_proj",
          "Query Low-Rank Down Proj (q_a_proj)",
          d,
          *qr,
          "Query 潜空间低秩压缩 + RMSNorm",
          -3,
          row,
        );
        let qb = g.add_proj(
          OpKind::ConvOrLinear,
          &preset.fmt_heads(*num_heads, qk_dim),
          &fmt_num("q_b ", *num_heads, "H"),
          "Query Up Proj + Decoupled RoPE (q_b_proj)",
          *qr,
          num_heads * qk_dim,
          "Split q_nope + Decoupled RoPE q_pe",
          -3,
          row + 1,
        );
        g.edge(attn_in, qa);
        g.edge(qa, qb);
        qb
      } else {
        let qd = g.add_proj(
          OpKind::ConvOrLinear,
          &preset.fmt_heads(*num_heads, qk_dim),
          &fmt_num("q ", *num_heads, "H"),
          "Direct Query Projection",
          d,
          num_heads * qk_dim,
          "无 Query 低秩压缩",
          -3,
          row,
        );
        g.edge(attn_in, qd);
        qd
      };

      let kva = g.add_proj(
        OpKind::AvgPool,
        &preset.fmt_seq(kv_a_dim),
        &fmt_num("KV ", kv_a_dim, "d"),
        "Latent KV Compression (kv_a_proj_with_mqa)",
        d,
        kv_a_dim,
        &fmt_num("推理仅缓存 ", kv_a_dim, " elems/tok"),
        3,
        row,
      );
      let kvb = g.add_proj(
        OpKind::ConvOrLinear,
        &preset.fmt_heads(*num_heads, kv_b_head),
        "kv_b (512→)",
        "Latent KV Up Proj (kv_b_proj, Absorbable)",
        *kv_lora_rank,
        num_heads * kv_b_head,
        "仅对 512d c_KV 上投影；64d k_pe 经 RoPE 旁路直通 MLA",
        3,
        row + 1,
      );
      g.edge(attn_in, kva);
      g.edge(kva, kvb);
      row += 2;

      let mla = g.add(
        OpKind::Attention,
        &preset.fmt_seq(o_in),
        &fmt_num("MLA ", *num_heads, "H"),
        "Multi-Head Latent Attention (MLA)",
        "Softmax((q_nope·k_nopeᵀ + q_pe·k_peᵀ)/√d) · V",
        &fmt_param("本层 Attention 总参: ", attn_params),
        0,
        row,
      );
      g.edge(q_leaf, mla);
      g.edge(kvb, mla);
      g.edge_aux(kva, mla, "k_pe 64d");
      row += 1;

      let op = g.add_proj(
        OpKind::ConvOrLinear,
        seq_d,
        "o_proj",
        "Attention Output Projection (o_proj)",
        o_in,
        d,
        &fmt_param("参数量: ", o_in * d),
        0,
        row,
      );
      g.edge(mla, op);
      row += 1;

      g.add_residual(
        op,
        prev,
        row,
        seq_d,
        "Attn Skip",
        "Residual Skip Add #1 (Attention)",
        "h_mid = h_in + o_proj(MLA(Norm(h_in)))",
        "单层内 Attention 子块残差汇合",
      )
    }
    AttentionSpec::StandardGqa {
      num_q_heads,
      num_kv_heads,
      head_dim,
      qk_norm,
      sliding_window,
      is_causal,
      has_cross_attn,
      ..
    } => {
      min_q = min_q.min(-4);
      max_q = max_q.max(4);
      let q_out = num_q_heads * head_dim;
      let kv_out = num_kv_heads * head_dim;
      let kv_shape = preset.fmt_heads(*num_kv_heads, *head_dim);

      let q = g.add_proj(
        OpKind::ConvOrLinear,
        &preset.fmt_heads(*num_q_heads, *head_dim),
        &fmt_num(if *qk_norm { "q+N " } else { "q " }, *num_q_heads, "H"),
        "Query Projection (q_proj + RoPE)",
        d,
        q_out,
        &fmt_param("参数量: ", d * q_out),
        -4,
        row,
      );
      let k = g.add_proj(
        OpKind::AvgPool,
        &kv_shape,
        &fmt_num(if *qk_norm { "k+N " } else { "k " }, *num_kv_heads, "H"),
        "Key Projection (k_proj → KV Cache)",
        d,
        kv_out,
        &fmt_param("KV elems/tok: ", seg.attention.cached_elements_per_token()),
        0,
        row,
      );
      let v = g.add_proj(
        OpKind::AvgPool,
        &kv_shape,
        &fmt_num("v ", *num_kv_heads, "H"),
        "Value Projection (v_proj → KV Cache)",
        d,
        kv_out,
        "与 K 共同构成推理期 KV Cache",
        4,
        row,
      );
      g.edge(attn_in, q);
      g.edge(attn_in, k);
      g.edge(attn_in, v);
      row += 1;

      let tag = match (sliding_window, is_causal, has_cross_attn) {
        (Some(w), _, _) => fmt_num("SWA(", *w, ")"),
        (None, false, _) => "Bi-Attn".to_string(),
        (None, true, true) => "Self+Cross".to_string(),
        _ => fmt_pair(*num_q_heads, ":", *num_kv_heads),
      };
      let sdpa = g.add(
        OpKind::Attention,
        &preset.fmt_seq(q_out),
        &tag,
        "Scaled Dot-Product Attention (GQA / FlashAttn)",
        "Softmax(Q·Kᵀ/√d_k)·V → Concat Heads",
        &fmt_param("本层 Attention 总参: ", attn_params),
        0,
        row,
      );
      g.edge(q, sdpa);
      g.edge(k, sdpa);
      g.edge(v, sdpa);
      row += 1;

      let op = g.add_proj(
        OpKind::ConvOrLinear,
        seq_d,
        "o_proj",
        "Attention Output Projection (o_proj)",
        q_out,
        d,
        &fmt_param("参数量: ", q_out * d),
        0,
        row,
      );
      g.edge(sdpa, op);
      row += 1;

      g.add_residual(
        op,
        prev,
        row,
        seq_d,
        "Attn Skip",
        "Residual Skip Add #1 (Attention)",
        "h_mid = h_in + o_proj(Attn(Norm(h_in)))",
        "单层内 Attention 子块残差汇合",
      )
    }
    AttentionSpec::LinearRecurrent {
      kind,
      num_heads,
      head_dim,
      state_dim,
      conv_kernel,
      expand_factor,
    } => {
      let d_inner = (num_heads * head_dim).max((*expand_factor).max(1) * d);
      let seq_in = preset.fmt_seq(d_inner);
      let conv = g.add_proj(
        OpKind::ConvOrLinear,
        &seq_in,
        &fmt_num("Conv k=", *conv_kernel, ""),
        "Input Proj + Causal Conv1d",
        d,
        d_inner,
        "O(1) 短局部卷积缓存",
        -3,
        row,
      );
      let ssm = g.add(
        OpKind::Attention,
        &seq_in,
        &fmt_num("state=", *state_dim, "d"),
        "Recurrent State Update (Linear / SSM)",
        kind,
        &fmt_param("固定状态 elems/seq: ", seg.attention.fixed_state_elements_per_seq(d)),
        -3,
        row + 1,
      );
      let gate = g.add_proj(
        OpKind::MaxPool,
        &seq_in,
        "SiLU Gate",
        "Parallel Output Gate",
        d,
        d_inner,
        "逐元素门控调制",
        3,
        row,
      );
      g.edge(attn_in, conv);
      g.edge(conv, ssm);
      g.edge(attn_in, gate);
      row += 2;

      let mul = g.add(
        OpKind::Merge,
        &seq_in,
        "Gate ⊙",
        "Gated Hadamard Product",
        "y = SSM(x) ⊙ SiLU(z)",
        "门控特征融合",
        0,
        row,
      );
      g.edge(ssm, mul);
      g.edge(gate, mul);
      row += 1;

      let op = g.add_proj(
        OpKind::ConvOrLinear,
        seq_d,
        "out_proj",
        "SSM Output Projection (out_proj)",
        d_inner,
        d,
        &fmt_param("本层 Mixer 总参: ", attn_params),
        0,
        row,
      );
      g.edge(mul, op);
      row += 1;

      g.add_residual(
        op,
        prev,
        row,
        seq_d,
        "SSM Skip",
        "Residual Skip Add #1 (SSM Mixer)",
        "h_mid = h_in + out_proj(y)",
        "单层内 SSM 子块残差输出",
      )
    }
  };

  let (ffn_tot, ffn_act) = seg.ffn.params_per_layer(d);
  if matches!(seg.ffn, FfnSpec::DenseGated { intermediate_size: 0, .. }) {
    return (attn_add, row, None, min_q, max_q);
  }
  let split_after_row = Some(row);

  row += 1;
  g.set_ctx(Some(InspectorFocus::FfnOrMoe), Some(layer));
  let ffn_in = if compact {
    attn_add
  } else {
    let n2 = g.add_norm(
      attn_add,
      row,
      d,
      seq_d,
      "Post-Norm",
      "post_attention_layernorm (RMSNorm)",
      "FFN / MoE 子层入口归一化",
    );
    row += 1;
    n2
  };

  let ffn_add = match &seg.ffn {
    FfnSpec::DenseGated { intermediate_size, act_fn, is_gated, .. } => {
      let seq_i = preset.fmt_seq(*intermediate_size);
      let p_up = fmt_param("参数量: ", d * intermediate_size);
      if *is_gated {
        let mut gate_lbl = String::with_capacity(act_fn.len() + 5);
        gate_lbl.push_str("gate+");
        gate_lbl.push_str(act_fn);
        let gate = g.add_proj(
          OpKind::ConvOrLinear,
          &seq_i,
          &gate_lbl,
          "Gate Projection + Activation",
          d,
          *intermediate_size,
          &p_up,
          -3,
          row,
        );
        let up = g.add_proj(
          OpKind::ConvOrLinear,
          &seq_i,
          "up_proj",
          "Up Projection (up_proj)",
          d,
          *intermediate_size,
          &p_up,
          3,
          row,
        );
        g.edge(ffn_in, gate);
        g.edge(ffn_in, up);
        row += 1;

        let mul = g.add(
          OpKind::Merge,
          &seq_i,
          "Mul ⊙",
          "SwiGLU / GeGLU Elementwise Multiply",
          "h_ffn = Act(gate_proj(x)) ⊙ up_proj(x)",
          "门控双支路哈达玛乘积",
          0,
          row,
        );
        g.edge(gate, mul);
        g.edge(up, mul);
        row += 1;

        let down = g.add_proj(
          OpKind::ConvOrLinear,
          seq_d,
          "down_proj",
          "Down Projection (down_proj)",
          *intermediate_size,
          d,
          &fmt_param("本层 Dense FFN 总参: ", ffn_tot),
          0,
          row,
        );
        g.edge(mul, down);
        row += 1;

        g.add_residual(
          down,
          attn_add,
          row,
          seq_d,
          "FFN Skip",
          "Residual Skip Add #2 (FFN Output)",
          "h_out = h_mid + down_proj(h_ffn)",
          "单层内 FFN 子块残差完成",
        )
      } else {
        let fc1 = g.add_proj(
          OpKind::ConvOrLinear,
          &seq_i,
          "fc1+act",
          "MLP Up Projection",
          d,
          *intermediate_size,
          &p_up,
          0,
          row,
        );
        g.edge(ffn_in, fc1);
        row += 1;
        let fc2 = g.add_proj(
          OpKind::ConvOrLinear,
          seq_d,
          "fc2",
          "MLP Down Projection",
          *intermediate_size,
          d,
          &fmt_param("本层 MLP 总参: ", ffn_tot),
          0,
          row,
        );
        g.edge(fc1, fc2);
        row += 1;
        g.add_residual(
          fc2,
          attn_add,
          row,
          seq_d,
          "FFN Skip",
          "Residual Skip Add #2 (MLP Output)",
          "h_out = h_mid + fc2(fc1(Norm(h_mid)))",
          "单层内 MLP 子块残差完成",
        )
      }
    }
    FfnSpec::SparseMoe {
      num_routed_experts,
      top_k,
      moe_intermediate_size,
      num_shared_experts,
      shared_intermediate_size,
      scoring_func,
      ..
    } => {
      let has_shared = *num_shared_experts > 0 && *shared_intermediate_size > 0;
      if has_shared {
        min_q = min_q.min(-4);
        max_q = max_q.max(4);
      }
      let router_col = if has_shared { 0 } else { -3 };
      let expert_col = if has_shared { 4 } else { 3 };

      let shared_leaf = if has_shared {
        let su = g.add_proj(
          OpKind::MoeRouter,
          &preset.fmt_seq(*shared_intermediate_size),
          &fmt_num("Shared ×", *num_shared_experts, ""),
          "Always-On Shared Expert",
          d,
          *shared_intermediate_size,
          "100% Token 直通常驻共享专家",
          -4,
          row,
        );
        let sd = g.add_proj(
          OpKind::ConvOrLinear,
          seq_d,
          "sh_down",
          "Shared Expert Down Proj",
          *shared_intermediate_size,
          d,
          "与路由专家结果叠加",
          -4,
          row + 1,
        );
        g.edge(ffn_in, su);
        g.edge(su, sd);
        Some(sd)
      } else {
        None
      };

      let router = g.add_proj(
        OpKind::MoeRouter,
        &preset.fmt_seq(*num_routed_experts),
        &fmt_pair(d, "→", *num_routed_experts),
        "MoE Router Gate",
        d,
        *num_routed_experts,
        scoring_func,
        router_col,
        row,
      );
      let mut topk_tag = String::with_capacity(14);
      topk_tag.push_str("Top-");
      push_u64(&mut topk_tag, *top_k);
      topk_tag.push('/');
      push_u64(&mut topk_tag, *num_routed_experts);
      let topk = g.add(
        OpKind::Softmax,
        &preset.fmt_seq(*top_k),
        &topk_tag,
        "Router Scoring & Top-K Dispatch",
        &fmt_num("Top-", *top_k, " Active Weights g_i"),
        "仅激活 Top-K 专家，其余 0 FLOPs",
        router_col,
        row + 1,
      );
      g.edge(ffn_in, router);
      g.edge(router, topk);

      let mut eu_shape = String::with_capacity(24);
      push_u64(&mut eu_shape, *top_k);
      eu_shape.push('×');
      eu_shape.push_str(&preset.fmt_seq(*moe_intermediate_size));
      let mut ed_shape = String::with_capacity(24);
      push_u64(&mut ed_shape, *top_k);
      ed_shape.push('×');
      ed_shape.push_str(seq_d);
      let eu = g.add_proj(
        OpKind::MoeRouter,
        &eu_shape,
        &fmt_num("×", *top_k, " Active"),
        "Top-K Routed Experts SwiGLU",
        d,
        *moe_intermediate_size,
        &fmt_param("单层总参: ", ffn_tot),
        expert_col,
        row,
      );
      let ed = g.add_proj(
        OpKind::ConvOrLinear,
        &ed_shape,
        "exp_down",
        "Routed Experts Down Proj",
        *moe_intermediate_size,
        d,
        "按路由权重 g_i 加权汇总",
        expert_col,
        row + 1,
      );
      g.edge(ffn_in, eu);
      g.edge(eu, ed);
      // 水平向右路由调度与门控加权：避免倒指斜线与 BxSxK 混入主干加法流
      g.edge_aux(router, eu, "dispatch");
      g.edge_aux(topk, ed, "× g_i");
      row += 2;

      let add2 = g.add(
        OpKind::Merge,
        seq_d,
        "Sum + Add",
        "MoE Weighted Combine + Residual Add #2",
        if has_shared {
          "h_out = h_mid + Shared(x) + Σ g_i·Expert_i(x)"
        } else {
          "h_out = h_mid + Σ g_i·Expert_i(x)"
        },
        &fmt_param("单层稀疏激活: ", ffn_act),
        0,
        row,
      );
      if let Some(sh) = shared_leaf {
        g.edge(sh, add2);
      }
      g.edge(ed, add2);
      g.edge_inner_skip(attn_add, add2, "MoE Skip");
      add2
    }
  };

  (ffn_add, row, split_after_row, min_q, max_q)
}

// ─────────────────────────────────────────────────────────────────────────────
// 经典 CV 教学对照（静态只读数据表驱动）
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
struct StaticCvNode {
  kind: OpKind,
  h: u64,
  w: u64,
  c: u64,
  note: &'static str,
  title: &'static str,
  formula: &'static str,
  hint: &'static str,
  col_q: i32,
  row: usize,
}

#[cfg(test)]
const INCEPTION_NODES: &[StaticCvNode] = &[
  StaticCvNode {
    kind: OpKind::Input,
    h: 299,
    w: 299,
    c: 3,
    note: "RGB",
    title: "输入图像 (299×299)",
    formula: "ImageNet RGB [299×299×3]",
    hint: "Inception-v3 标准输入",
    col_q: 0,
    row: 0,
  },
  StaticCvNode {
    kind: OpKind::ConvOrLinear,
    h: 149,
    w: 149,
    c: 32,
    note: "3*3 /2",
    title: "Stem Conv1 (3×3, s=2)",
    formula: "⌊(299−3)/2⌋+1 = 149 → 32ch",
    hint: "参数量 864",
    col_q: 0,
    row: 1,
  },
  StaticCvNode {
    kind: OpKind::ConvOrLinear,
    h: 147,
    w: 147,
    c: 32,
    note: "3*3",
    title: "Stem Conv2 (3×3, valid)",
    formula: "149−3+1 = 147 → 32ch",
    hint: "参数量 9.2K",
    col_q: 0,
    row: 2,
  },
  StaticCvNode {
    kind: OpKind::ConvOrLinear,
    h: 147,
    w: 147,
    c: 64,
    note: "3*3 pad",
    title: "Stem Conv3 (3×3, same)",
    formula: "Same Padding 通道 32 → 64",
    hint: "参数量 18.4K",
    col_q: 0,
    row: 3,
  },
  StaticCvNode {
    kind: OpKind::MaxPool,
    h: 73,
    w: 73,
    c: 64,
    note: "3*3 /2",
    title: "Stem MaxPool1 (3×3, s=2)",
    formula: "空间下采样 147 → 73",
    hint: "零参数池化",
    col_q: 0,
    row: 4,
  },
  StaticCvNode {
    kind: OpKind::ConvOrLinear,
    h: 73,
    w: 73,
    c: 80,
    note: "1*1",
    title: "Stem Conv4 (1×1)",
    formula: "点卷积通道扩展 64 → 80",
    hint: "参数量 5.1K",
    col_q: 0,
    row: 5,
  },
  StaticCvNode {
    kind: OpKind::ConvOrLinear,
    h: 71,
    w: 71,
    c: 192,
    note: "3*3",
    title: "Stem Conv5 (3×3, valid)",
    formula: "73−3+1 = 71, 通道 80 → 192",
    hint: "参数量 138.2K",
    col_q: 0,
    row: 6,
  },
  StaticCvNode {
    kind: OpKind::MaxPool,
    h: 35,
    w: 35,
    c: 192,
    note: "3*3 /2",
    title: "Stem MaxPool2 (3×3, s=2)",
    formula: "下采样至 35×35×192，进入 4 分支",
    hint: "Stem 末端特征图",
    col_q: 0,
    row: 7,
  },
  StaticCvNode {
    kind: OpKind::AvgPool,
    h: 35,
    w: 35,
    c: 192,
    note: "3*3",
    title: "Branch 1: AvgPool (3×3)",
    formula: "平滑上下文池化 [35×35×192]",
    hint: "零参数池化支路",
    col_q: -6,
    row: 8,
  },
  StaticCvNode {
    kind: OpKind::ConvOrLinear,
    h: 35,
    w: 35,
    c: 32,
    note: "1*1",
    title: "Branch 1: 1×1 Conv",
    formula: "通道压缩 192 → 32",
    hint: "贡献 32 通道",
    col_q: -6,
    row: 9,
  },
  StaticCvNode {
    kind: OpKind::ConvOrLinear,
    h: 35,
    w: 35,
    c: 64,
    note: "1*1",
    title: "Branch 2: 1×1 Bottleneck",
    formula: "降维 192 → 64",
    hint: "减少双 3×3 计算量",
    col_q: -2,
    row: 8,
  },
  StaticCvNode {
    kind: OpKind::ConvOrLinear,
    h: 35,
    w: 35,
    c: 96,
    note: "3*3",
    title: "Branch 2: First 3×3 Conv",
    formula: "中尺度卷积 64 → 96",
    hint: "参数量 55.3K",
    col_q: -2,
    row: 9,
  },
  StaticCvNode {
    kind: OpKind::ConvOrLinear,
    h: 35,
    w: 35,
    c: 96,
    note: "3*3",
    title: "Branch 2: Second 3×3 Conv",
    formula: "双 3×3 等效 5×5 感受野",
    hint: "贡献 96 通道",
    col_q: -2,
    row: 10,
  },
  StaticCvNode {
    kind: OpKind::ConvOrLinear,
    h: 35,
    w: 35,
    c: 48,
    note: "1*1",
    title: "Branch 3: 1×1 Bottleneck",
    formula: "降维 192 → 48",
    hint: "为 5×5 节省 FLOPs",
    col_q: 2,
    row: 8,
  },
  StaticCvNode {
    kind: OpKind::ConvOrLinear,
    h: 35,
    w: 35,
    c: 64,
    note: "5*5",
    title: "Branch 3: 5×5 Conv",
    formula: "大感受野 5×5 卷积 → 64ch",
    hint: "贡献 64 通道",
    col_q: 2,
    row: 9,
  },
  StaticCvNode {
    kind: OpKind::ConvOrLinear,
    h: 35,
    w: 35,
    c: 64,
    note: "1*1",
    title: "Branch 4: Direct 1×1 Conv",
    formula: "点卷积直连分支 192 → 64",
    hint: "贡献 64 通道",
    col_q: 6,
    row: 9,
  },
  StaticCvNode {
    kind: OpKind::Merge,
    h: 35,
    w: 35,
    c: 256,
    note: "Concat",
    title: "Inception-A Channel Concat",
    formula: "Concat(32 + 96 + 64 + 64) = 256ch",
    hint: "四分支多尺度融合",
    col_q: 0,
    row: 11,
  },
  StaticCvNode {
    kind: OpKind::ConvOrLinear,
    h: 17,
    w: 17,
    c: 768,
    note: "Inc-B ×5",
    title: "Reduction + Inception-B (×5)",
    formula: "1×7 & 7×1 非对称分解卷积 → 768ch",
    hint: "下采样 35 → 17",
    col_q: 0,
    row: 12,
  },
  StaticCvNode {
    kind: OpKind::ConvOrLinear,
    h: 8,
    w: 8,
    c: 2048,
    note: "Inc-C ×3",
    title: "Reduction + Inception-C (×3)",
    formula: "高维多分支扩展 → 8×8×2048",
    hint: "深层语义特征",
    col_q: 0,
    row: 13,
  },
  StaticCvNode {
    kind: OpKind::AvgPool,
    h: 1,
    w: 1,
    c: 2048,
    note: "8*8 GAP",
    title: "Global Average Pooling",
    formula: "8×8 全局平均池化 → 2048d",
    hint: "替代 Flatten 防过拟合",
    col_q: 0,
    row: 14,
  },
  StaticCvNode {
    kind: OpKind::Dropout,
    h: 0,
    w: 0,
    c: 2048,
    note: "p=0.2",
    title: "Dropout (p = 0.2)",
    formula: "训练期随机丢弃 20% 通道",
    hint: "正则化防共适应",
    col_q: 0,
    row: 15,
  },
  StaticCvNode {
    kind: OpKind::FcHead,
    h: 0,
    w: 0,
    c: 1000,
    note: "2048→1K",
    title: "Fully Connected (FC)",
    formula: "Linear [2048 → 1000]",
    hint: "全网总参 23.8M",
    col_q: 0,
    row: 16,
  },
  StaticCvNode {
    kind: OpKind::Softmax,
    h: 0,
    w: 0,
    c: 1000,
    note: "1K cls",
    title: "Softmax Output",
    formula: "Softmax(logits) → 1000 类概率",
    hint: "ImageNet-1K 分类输出",
    col_q: 0,
    row: 17,
  },
];

#[cfg(test)]
const INCEPTION_EDGES: &[(usize, usize, &str, bool)] = &[
  (0, 1, "", false),
  (1, 2, "", false),
  (2, 3, "", false),
  (3, 4, "", false),
  (4, 5, "", false),
  (5, 6, "", false),
  (6, 7, "", false),
  (7, 8, "", false),
  (8, 9, "", false),
  (7, 10, "", false),
  (10, 11, "", false),
  (11, 12, "", false),
  (7, 13, "", false),
  (13, 14, "", false),
  (7, 15, "", false),
  (9, 16, "", false),
  (12, 16, "", false),
  (14, 16, "", false),
  (15, 16, "", false),
  (16, 17, "", false),
  (17, 18, "", false),
  (18, 19, "", false),
  (19, 20, "", false),
  (20, 21, "", false),
  (21, 22, "", false),
];

#[cfg(test)]
const RESNET_NODES: &[StaticCvNode] = &[
  StaticCvNode {
    kind: OpKind::Input,
    h: 224,
    w: 224,
    c: 3,
    note: "RGB",
    title: "输入图像 (224×224)",
    formula: "ImageNet RGB [224×224×3]",
    hint: "ResNet-50 标准输入",
    col_q: 0,
    row: 0,
  },
  StaticCvNode {
    kind: OpKind::ConvOrLinear,
    h: 112,
    w: 112,
    c: 64,
    note: "7*7 /2",
    title: "Stem Conv1 (7×7, s=2)",
    formula: "7×7 卷积 + BN + ReLU → 112×112×64",
    hint: "参数量 9.4K",
    col_q: 0,
    row: 1,
  },
  StaticCvNode {
    kind: OpKind::MaxPool,
    h: 56,
    w: 56,
    c: 64,
    note: "3*3 /2",
    title: "Stem MaxPool (3×3, s=2)",
    formula: "空间下采样至 56×56×64",
    hint: "进入 Bottleneck 残差块",
    col_q: 0,
    row: 2,
  },
  StaticCvNode {
    kind: OpKind::ConvOrLinear,
    h: 56,
    w: 56,
    c: 256,
    note: "1*1 Skip",
    title: "Shortcut Projection",
    formula: "首块 1×1 (64→256)，后续恒等直连",
    hint: "消除深层梯度消失",
    col_q: -3,
    row: 4,
  },
  StaticCvNode {
    kind: OpKind::ConvOrLinear,
    h: 56,
    w: 56,
    c: 64,
    note: "1*1",
    title: "Bottleneck 1×1 Reduce",
    formula: "1×1 压缩通道至 64",
    hint: "降低 3×3 计算量",
    col_q: 3,
    row: 3,
  },
  StaticCvNode {
    kind: OpKind::ConvOrLinear,
    h: 56,
    w: 56,
    c: 64,
    note: "3*3",
    title: "Bottleneck 3×3 Conv",
    formula: "低维 64 通道 3×3 空间卷积",
    hint: "参数量 36.9K",
    col_q: 3,
    row: 4,
  },
  StaticCvNode {
    kind: OpKind::ConvOrLinear,
    h: 56,
    w: 56,
    c: 256,
    note: "1*1 ×4",
    title: "Bottleneck 1×1 Expand",
    formula: "1×1 恢复 4× 通道至 256",
    hint: "与 Shortcut 对齐",
    col_q: 3,
    row: 5,
  },
  StaticCvNode {
    kind: OpKind::Merge,
    h: 56,
    w: 56,
    c: 256,
    note: "Add (+)",
    title: "Residual Add + ReLU",
    formula: "y = ReLU(F(x) + Shortcut(x))",
    hint: "残差恒等映射核心节点",
    col_q: 0,
    row: 6,
  },
  StaticCvNode {
    kind: OpKind::ConvOrLinear,
    h: 28,
    w: 28,
    c: 512,
    note: "Conv3 ×4",
    title: "Stage 3 · Conv3_x (×4)",
    formula: "56→28, 256→512 (4× Bottleneck)",
    hint: "瓶颈宽度 128",
    col_q: 0,
    row: 7,
  },
  StaticCvNode {
    kind: OpKind::ConvOrLinear,
    h: 14,
    w: 14,
    c: 1024,
    note: "Conv4 ×6",
    title: "Stage 4 · Conv4_x (×6)",
    formula: "28→14, 512→1024 (6× Bottleneck)",
    hint: "瓶颈宽度 256",
    col_q: 0,
    row: 8,
  },
  StaticCvNode {
    kind: OpKind::ConvOrLinear,
    h: 7,
    w: 7,
    c: 2048,
    note: "Conv5 ×3",
    title: "Stage 5 · Conv5_x (×3)",
    formula: "14→7, 1024→2048 (3× Bottleneck)",
    hint: "瓶颈宽度 512",
    col_q: 0,
    row: 9,
  },
  StaticCvNode {
    kind: OpKind::AvgPool,
    h: 1,
    w: 1,
    c: 2048,
    note: "7*7 GAP",
    title: "Global Average Pooling",
    formula: "7×7 平均池化 → 2048d",
    hint: "零参数池化",
    col_q: 0,
    row: 10,
  },
  StaticCvNode {
    kind: OpKind::FcHead,
    h: 0,
    w: 0,
    c: 1000,
    note: "2048→1K",
    title: "Fully Connected (FC)",
    formula: "Linear [2048 → 1000]",
    hint: "ResNet-50 总参 25.6M",
    col_q: 0,
    row: 11,
  },
  StaticCvNode {
    kind: OpKind::Softmax,
    h: 0,
    w: 0,
    c: 1000,
    note: "1K cls",
    title: "Softmax Output",
    formula: "Softmax(logits) → 1000 类概率",
    hint: "ImageNet 分类输出",
    col_q: 0,
    row: 12,
  },
];

#[cfg(test)]
const RESNET_EDGES: &[(usize, usize, &str, bool)] = &[
  (0, 1, "", false),
  (1, 2, "", false),
  (2, 3, "Skip", true),
  (2, 4, "", false),
  (4, 5, "", false),
  (5, 6, "", false),
  (3, 7, "", false),
  (6, 7, "", false),
  (7, 8, "", false),
  (8, 9, "", false),
  (9, 10, "", false),
  (10, 11, "", false),
  (11, 12, "", false),
  (12, 13, "", false),
];

#[cfg(test)]
#[inline(never)]
fn build_static_cv_graph(
  slug: &str,
  title: &str,
  subtitle_prefix: &str,
  raw_nodes: &[StaticCvNode],
  raw_edges: &[(usize, usize, &'static str, bool)],
  group_spec: (usize, usize, i32, i32, &str),
  preset: ShapePreset,
) -> TensorGraph {
  let mut g = GraphBuilder::new();
  for n in raw_nodes {
    let sh = preset.fmt_cv(n.h, n.w, n.c);
    g.add(n.kind, &sh, n.note, n.title, n.formula, n.hint, n.col_q, n.row);
  }
  for &(from, to, label, dashed) in raw_edges {
    g.edges.push(TensorEdge {
      from,
      to,
      label: label.to_string(),
      kind: if dashed { EdgeKind::MacroSkip } else { EdgeKind::Main },
      group_idx: u8::MAX,
      lane: 0,
    });
  }
  g.group(
    None,
    false,
    group_spec.0,
    group_spec.1,
    None,
    group_spec.2,
    group_spec.3,
    group_spec.4.to_string(),
  );
  let mut sub = String::with_capacity(subtitle_prefix.len() + 20);
  sub.push_str(subtitle_prefix);
  sub.push_str(" · ");
  sub.push_str(preset.label());
  g.finish(slug.to_string(), title.to_string(), sub, LegendStyle::ClassicCnn)
}

#[cfg(test)]
pub fn build_inception_v3_graph(preset: ShapePreset) -> TensorGraph {
  build_static_cv_graph(
    "inception-v3",
    "Inception-v3 · 经典多分支 CNN 张量维流图",
    "23.8M Params · 4-Branch Inception-A · 8 类经典算子色块",
    INCEPTION_NODES,
    INCEPTION_EDGES,
    (8, 11, -6, 6, "Inception-A 4-Branch Block (×3)"),
    preset,
  )
}

#[cfg(test)]
pub fn build_resnet50_graph(preset: ShapePreset) -> TensorGraph {
  build_static_cv_graph(
    "resnet-50",
    "ResNet-50 · 经典残差瓶颈 CNN 张量维流图",
    "25.6M Params · Bottleneck 1×1→3×3→1×1 + Shortcut",
    RESNET_NODES,
    RESNET_EDGES,
    (3, 6, -3, 3, "Bottleneck Residual Block · Conv2_x (×3)"),
    preset,
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// 声明式外置图谱（`models/<slug>/graph.json` 驱动 YOLO / U-Net / SAM / 经典网络）
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(any(feature = "server", test))]
fn default_scale_one() -> u32 {
  1
}

#[cfg(any(feature = "server", test))]
fn default_shape_mode() -> String {
  "image_2d".to_string()
}

/// 声明式图谱分辨率 / 批次预设（仅服务端解析 `graph.json` 使用，零 WASM 体积开销）。
#[cfg(any(feature = "server", test))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeclarativeShapePreset {
  pub label: String,
  #[serde(default)]
  pub batch: String,
  #[serde(default = "default_scale_one")]
  pub scale_num: u32,
  #[serde(default = "default_scale_one")]
  pub scale_den: u32,
}

/// 声明式图谱可折叠阶段组块（仅服务端解析 `graph.json` 使用）。
#[cfg(any(feature = "server", test))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeclarativeGroup {
  pub badge: String,
  #[serde(default)]
  pub collapsed_shape: String,
  #[serde(default)]
  pub collapsed_note: String,
  #[serde(default)]
  pub collapsed_title: String,
  #[serde(default)]
  pub collapsed_formula: String,
  #[serde(default)]
  pub collapsed_hint: String,
  pub row_start: usize,
  pub row_end: usize,
  #[serde(default)]
  pub split_after_row: Option<usize>,
  pub min_col_q: i32,
  pub max_col_q: i32,
  #[serde(default)]
  pub default_collapsed: bool,
}

/// 声明式图谱节点定义（仅服务端解析 `graph.json` 使用）。
#[cfg(any(feature = "server", test))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeclarativeNode {
  pub id: String,
  #[serde(default)]
  pub kind: OpKind,
  #[serde(default)]
  pub shape: String,
  /// 可选结构化维度：
  /// - `image_2d`: `[h, w, c]`（`h == 0 && w == 0` 表示展平向量 `[c]`）
  /// - `volume_3d`: `[d, h, w, c]`
  /// - `seq`: `[s, d]`
  /// - `vector`: `[d]` 或 `[a, b]`
  #[serde(default)]
  pub dims: Vec<u64>,
  #[serde(default)]
  pub side_note: String,
  pub op_title: String,
  #[serde(default)]
  pub formula: String,
  #[serde(default)]
  pub params_hint: String,
  #[serde(default)]
  pub col_q: i32,
  pub row: usize,
}

/// 声明式图谱语义连线定义（仅服务端解析 `graph.json` 使用）。
#[cfg(any(feature = "server", test))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeclarativeEdge {
  pub from: String,
  pub to: String,
  #[serde(default)]
  pub label: String,
  #[serde(default)]
  pub kind: EdgeKind,
  /// 多泳道轨道号（`0..=7`，用于在 U-Net / YOLO 中平行错开多条跨尺度长跳连）。
  #[serde(default)]
  pub lane: u8,
  #[serde(default)]
  pub group_idx: Option<usize>,
}

/// 外置 `models/<slug>/graph.json` 完整声明式图谱规格（服务端校验并编码为紧凑线协议下发）。
#[cfg(any(feature = "server", test))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeclarativeGraphSpec {
  pub title: String,
  #[serde(default)]
  pub subtitle: String,
  #[serde(default)]
  pub legend_style: LegendStyle,
  /// 形状域：`"image_2d"` | `"volume_3d"` | `"vector"` | `"seq"`
  #[serde(default = "default_shape_mode")]
  pub shape_mode: String,
  #[serde(default)]
  pub presets: Vec<DeclarativeShapePreset>,
  #[serde(default)]
  pub groups: Vec<DeclarativeGroup>,
  pub nodes: Vec<DeclarativeNode>,
  pub edges: Vec<DeclarativeEdge>,
}

#[cfg(any(feature = "server", test))]
fn push_wire_field(out: &mut String, s: &str) {
  out.push('\t');
  for ch in s.chars() {
    if matches!(ch, '\t' | '\n' | '\r') {
      out.push(' ');
    } else {
      out.push(ch);
    }
  }
}

#[inline(never)]
fn parse_i64_ascii(s: &str, fallback: i64) -> i64 {
  let bytes = s.as_bytes();
  if bytes.is_empty() {
    return fallback;
  }
  let (neg, start) = if bytes[0] == b'-' { (true, 1) } else { (false, 0) };
  if start >= bytes.len() {
    return fallback;
  }
  let mut acc = 0i64;
  for &b in &bytes[start..] {
    if !b.is_ascii_digit() {
      return fallback;
    }
    acc = acc.saturating_mul(10).saturating_add((b - b'0') as i64);
  }
  if neg {
    -acc
  } else {
    acc
  }
}

#[cfg(any(feature = "server", test))]
const DEFAULT_PRESETS_IMG: &[(&str, &str, u32, u32)] =
  &[("标准 (B×1×)", "B", 1, 1), ("单图推理 (1×)", "1", 1, 1), ("高分 (1×2×)", "1", 2, 1)];
#[cfg(any(feature = "server", test))]
const DEFAULT_PRESETS_VOL: &[(&str, &str, u32, u32)] =
  &[("符号 (B×1×)", "B", 1, 1), ("单体积 (1×1×)", "1", 1, 1), ("低分 (1×0.5×)", "1", 1, 2)];
#[cfg(any(feature = "server", test))]
const DEFAULT_PRESETS_VEC: &[(&str, &str, u32, u32)] =
  &[("符号 (B×D)", "B", 1, 1), ("单样本 (1×D)", "1", 1, 1), ("小批 (32×D)", "32", 1, 1)];
#[cfg(any(feature = "server", test))]
const DEFAULT_PRESETS_SEQ: &[(&str, &str, u32, u32)] =
  &[("符号 (B×S)", "B", 1, 1), ("单句 (1×1×)", "1", 1, 1), ("长文 (1×2×)", "1", 2, 1)];

#[cfg(any(feature = "server", test))]
impl DeclarativeGraphSpec {
  /// 校验声明式图谱的结构合法性（零 `unwrap`，供服务端加载与离线校验复用）。
  pub fn validate(&self) -> Result<(), String> {
    if self.title.trim().is_empty() {
      return Err("graph.json 缺少 title".to_string());
    }
    if self.nodes.is_empty() {
      return Err("graph.json 至少需要 1 个节点".to_string());
    }
    for (i, n) in self.nodes.iter().enumerate() {
      if n.id.trim().is_empty() {
        return Err(format!("nodes[{i}] 缺少 id"));
      }
      if self.nodes[..i].iter().any(|prev| prev.id == n.id) {
        return Err(format!("nodes[{i}] 存在重复 id: {}", n.id));
      }
    }
    for (i, g) in self.groups.iter().enumerate() {
      if g.row_start > g.row_end {
        return Err(format!("groups[{i}] row_start({}) > row_end({})", g.row_start, g.row_end));
      }
      if g.min_col_q > g.max_col_q {
        return Err(format!("groups[{i}] min_col_q({}) > max_col_q({})", g.min_col_q, g.max_col_q));
      }
    }
    for (i, e) in self.edges.iter().enumerate() {
      if !self.nodes.iter().any(|n| n.id == e.from) {
        return Err(format!("edges[{i}] 引用了不存在的起点 id: {}", e.from));
      }
      if !self.nodes.iter().any(|n| n.id == e.to) {
        return Err(format!("edges[{i}] 引用了不存在的终点 id: {}", e.to));
      }
      if e.lane > 7 {
        return Err(format!("edges[{i}] lane({}) 超出允许范围 0..=7", e.lane));
      }
      if let Some(gi) = e.group_idx {
        if gi >= self.groups.len() {
          return Err(format!("edges[{i}] group_idx({gi}) 超出 groups 长度"));
        }
      }
    }
    Ok(())
  }

  /// 返回生效的分辨率/批次预设列表（未声明时按 `shape_mode` 提供合理默认值）。
  pub fn effective_presets(&self) -> Vec<DeclarativeShapePreset> {
    if !self.presets.is_empty() {
      return self.presets.clone();
    }
    let table = match self.shape_mode.as_str() {
      "volume_3d" => DEFAULT_PRESETS_VOL,
      "vector" => DEFAULT_PRESETS_VEC,
      "seq" => DEFAULT_PRESETS_SEQ,
      _ => DEFAULT_PRESETS_IMG,
    };
    table
      .iter()
      .map(|&(label, batch, scale_num, scale_den)| DeclarativeShapePreset {
        label: label.to_string(),
        batch: batch.to_string(),
        scale_num,
        scale_den,
      })
      .collect()
  }

  /// 计算默认折叠掩码（支持最多 30 个组）。
  pub fn default_collapsed_mask(&self) -> usize {
    let mut mask = 0usize;
    for (gi, g) in self.groups.iter().take(30).enumerate() {
      if g.default_collapsed {
        mask |= 1usize << gi;
      }
    }
    mask
  }

  /// 服务端将声明式图谱编码为预解析索引的 TAB 分隔行协议（客户端零 `serde` / 零中间结构体直出 `TensorGraph`）。
  pub fn encode_wire(&self) -> String {
    let mut out = String::with_capacity(2048);
    out.push('H');
    push_wire_field(&mut out, &self.title);
    push_wire_field(&mut out, &self.subtitle);
    out.push('\t');
    push_u64(&mut out, self.legend_style as u64);
    push_wire_field(&mut out, &self.shape_mode);
    out.push('\t');
    push_u64(&mut out, self.default_collapsed_mask() as u64);
    out.push('\n');

    for p in self.effective_presets() {
      out.push('P');
      push_wire_field(&mut out, &p.label);
      push_wire_field(&mut out, &p.batch);
      out.push('\t');
      push_u64(&mut out, p.scale_num as u64);
      out.push('\t');
      push_u64(&mut out, p.scale_den as u64);
      out.push('\n');
    }

    for g in &self.groups {
      out.push('G');
      push_wire_field(&mut out, &g.badge);
      push_wire_field(&mut out, &g.collapsed_shape);
      push_wire_field(&mut out, &g.collapsed_note);
      push_wire_field(&mut out, &g.collapsed_title);
      push_wire_field(&mut out, &g.collapsed_formula);
      push_wire_field(&mut out, &g.collapsed_hint);
      out.push('\t');
      push_u64(&mut out, g.row_start as u64);
      out.push('\t');
      push_u64(&mut out, g.row_end as u64);
      out.push('\t');
      if let Some(sr) = g.split_after_row {
        push_u64(&mut out, sr as u64);
      }
      out.push('\t');
      push_i32(&mut out, g.min_col_q);
      out.push('\t');
      push_i32(&mut out, g.max_col_q);
      out.push('\n');
    }

    for n in &self.nodes {
      out.push('N');
      out.push('\t');
      push_u64(&mut out, n.kind as u64);
      push_wire_field(&mut out, &n.shape);
      out.push('\t');
      for (i, &d) in n.dims.iter().enumerate() {
        if i > 0 {
          out.push(',');
        }
        push_u64(&mut out, d);
      }
      push_wire_field(&mut out, &n.side_note);
      push_wire_field(&mut out, &n.op_title);
      push_wire_field(&mut out, &n.formula);
      push_wire_field(&mut out, &n.params_hint);
      out.push('\t');
      push_i32(&mut out, n.col_q);
      out.push('\t');
      push_u64(&mut out, n.row as u64);
      out.push('\n');
    }

    for e in &self.edges {
      let (Some(of), Some(ot)) = (
        self.nodes.iter().position(|n| n.id == e.from),
        self.nodes.iter().position(|n| n.id == e.to),
      ) else {
        continue;
      };
      out.push('E');
      out.push('\t');
      push_u64(&mut out, of as u64);
      out.push('\t');
      push_u64(&mut out, ot as u64);
      push_wire_field(&mut out, &e.label);
      out.push('\t');
      push_u64(&mut out, e.kind as u64);
      out.push('\t');
      push_u64(&mut out, e.lane.min(7) as u64);
      out.push('\t');
      if let Some(gi) = e.group_idx {
        push_u64(&mut out, gi as u64);
      }
      out.push('\n');
    }
    out
  }
}

#[inline(never)]
fn scale_u64_ratio(val: u64, num: u32, den: u32) -> u64 {
  if val == 0 {
    return 0;
  }
  ((val.saturating_mul(num.max(1) as u64)) / (den.max(1) as u64)).max(1)
}

#[inline(never)]
fn push_batch_prefix(out: &mut String, batch: &str) {
  if !batch.is_empty() {
    out.push_str(batch);
    out.push('×');
  }
}

#[inline(never)]
fn format_wire_shape(
  shape: &str,
  dims_csv: &str,
  shape_mode: &str,
  batch: &str,
  snum: u32,
  sden: u32,
) -> String {
  if dims_csv.is_empty() {
    if shape.contains("{B}") {
      return shape.replace("{B}", if batch.is_empty() { "B" } else { batch });
    }
    return shape.to_string();
  }
  let mut dims = [0u64; 6];
  let mut dlen = 0usize;
  for part in dims_csv.split(',') {
    if dlen < dims.len() {
      let v = parse_i64_ascii(part, -1);
      if v >= 0 {
        dims[dlen] = v as u64;
        dlen += 1;
      }
    }
  }
  let mut out = String::with_capacity(24);
  push_batch_prefix(&mut out, batch);
  if shape_mode == "image_2d" && dlen == 3 {
    if dims[0] == 0 && dims[1] == 0 {
      push_u64(&mut out, dims[2]);
    } else {
      push_u64(&mut out, scale_u64_ratio(dims[0], snum, sden));
      out.push('×');
      push_u64(&mut out, scale_u64_ratio(dims[1], snum, sden));
      out.push('×');
      push_u64(&mut out, dims[2]);
    }
  } else if shape_mode == "volume_3d" && dlen == 4 {
    for (i, &d) in dims[..4].iter().enumerate() {
      if i > 0 {
        out.push('×');
      }
      push_u64(&mut out, if i < 3 { scale_u64_ratio(d, snum, sden) } else { d });
    }
  } else if shape_mode == "seq" && dlen == 2 {
    push_u64(&mut out, scale_u64_ratio(dims[0], snum, sden));
    out.push('×');
    push_u64(&mut out, dims[1]);
  } else {
    for (i, &d) in dims[..dlen].iter().enumerate() {
      if i > 0 {
        out.push('×');
      }
      push_u64(&mut out, d);
    }
  }
  out
}

#[inline(never)]
fn next_s<'a>(it: &mut std::str::Split<'a, char>) -> &'a str {
  it.next().unwrap_or("")
}

#[inline(never)]
fn next_i(it: &mut std::str::Split<'_, char>, fallback: i64) -> i64 {
  parse_i64_ascii(it.next().unwrap_or(""), fallback)
}

/// 全端零中间结构体：直接从服务端编码的紧凑线协议构建 [`TensorGraph`]、预设标签列表与默认折叠掩码。
#[inline(never)]
pub fn build_declarative_graph_from_wire(
  wire: &str,
  slug: &str,
  preset_idx: usize,
  toggle_mask: usize,
) -> Option<(TensorGraph, Vec<String>, usize)> {
  let mut hdr_line: Option<&str> = None;
  let mut p_lines: Vec<&str> = Vec::new();
  let mut g_lines: Vec<&str> = Vec::new();
  let mut n_lines: Vec<&str> = Vec::new();
  let mut e_lines: Vec<&str> = Vec::new();

  for line in wire.lines() {
    match line.as_bytes().first().copied() {
      Some(b'H') => hdr_line = Some(line),
      Some(b'P') => p_lines.push(line),
      Some(b'G') => g_lines.push(line),
      Some(b'N') => n_lines.push(line),
      Some(b'E') => e_lines.push(line),
      _ => {}
    }
  }
  let hdr = hdr_line?;
  if n_lines.is_empty() {
    return None;
  }

  let mut hit = hdr.split('\t');
  let _ = hit.next();
  let title = next_s(&mut hit);
  let sub_base = next_s(&mut hit);
  let legend_style = LegendStyle::from_u8(next_i(&mut hit, 1) as u8);
  let shape_mode = next_s(&mut hit);
  let default_mask = next_i(&mut hit, 0).max(0) as usize;
  let eff_mask = toggle_mask ^ default_mask;

  let mut preset_labels: Vec<String> = Vec::with_capacity(p_lines.len());
  let (mut p_label, mut p_batch, mut p_num, mut p_den) = ("", "B", 1u32, 1u32);
  for (pi, &pl) in p_lines.iter().enumerate() {
    let mut pit = pl.split('\t');
    let _ = pit.next();
    let lbl = next_s(&mut pit);
    let bat = next_s(&mut pit);
    let sn = next_i(&mut pit, 1).max(1) as u32;
    let sd = next_i(&mut pit, 1).max(1) as u32;
    preset_labels.push(lbl.to_string());
    if pi == preset_idx || (pi == 0 && preset_idx >= p_lines.len()) {
      p_label = lbl;
      p_batch = bat;
      p_num = sn;
      p_den = sd;
    }
  }

  let is_col = |gi: usize| gi < 30 && (eff_mask & (1usize << gi)) != 0;

  let g_cnt = g_lines.len();
  let mut g_bounds: Vec<usize> = Vec::with_capacity(g_cnt * 2);
  for &gl in &g_lines {
    let mut git = gl.split('\t');
    for _ in 0..7 {
      let _ = git.next();
    }
    let rs = next_i(&mut git, 0).max(0) as usize;
    let re = next_i(&mut git, rs as i64).max(0) as usize;
    g_bounds.push(rs);
    g_bounds.push(re);
  }

  let mut node_owner: Vec<usize> = Vec::with_capacity(n_lines.len());
  let mut used_rows: Vec<usize> = Vec::with_capacity(n_lines.len());
  for &nl in &n_lines {
    let row = parse_i64_ascii(nl.split('\t').nth(9).unwrap_or("0"), 0).max(0) as usize;
    let mut owner = usize::MAX;
    for gi in 0..g_cnt {
      if row >= g_bounds[gi * 2] && row <= g_bounds[gi * 2 + 1] {
        owner = gi;
        break;
      }
    }
    node_owner.push(owner);
    let eff_row = if owner < g_cnt && is_col(owner) { g_bounds[owner * 2] } else { row };
    if !used_rows.contains(&eff_row) {
      let pos = used_rows.iter().position(|&r| r > eff_row).unwrap_or(used_rows.len());
      used_rows.insert(pos, eff_row);
    }
  }
  let map_row = |raw: usize| -> usize {
    used_rows.iter().position(|&r| r == raw).unwrap_or(used_rows.len().saturating_sub(1))
  };

  let mut nodes: Vec<TensorNode> = Vec::with_capacity(n_lines.len());
  let mut old_to_new: Vec<usize> = vec![usize::MAX; n_lines.len()];
  let mut col_node: Vec<usize> = vec![usize::MAX; g_cnt];
  let mut groups: Vec<RepeatGroup> = Vec::with_capacity(g_cnt);

  for (gi, &gl) in g_lines.iter().enumerate() {
    let mut git = gl.split('\t');
    let _ = git.next();
    let badge = next_s(&mut git);
    let c_shape_raw = next_s(&mut git);
    let c_note = next_s(&mut git);
    let c_title = next_s(&mut git);
    let c_formula = next_s(&mut git);
    let c_hint = next_s(&mut git);
    let _ = git.next();
    let _ = git.next();
    let sr_raw = next_s(&mut git);
    let min_q = next_i(&mut git, 0) as i32;
    let max_q = next_i(&mut git, 0) as i32;

    let collapsed = is_col(gi);
    let r_start = map_row(g_bounds[gi * 2]);
    let r_end = if collapsed { r_start } else { map_row(g_bounds[gi * 2 + 1]) };
    let split = if collapsed || sr_raw.is_empty() {
      None
    } else {
      let v = parse_i64_ascii(sr_raw, -1);
      if v >= 0 {
        Some(map_row(v as usize))
      } else {
        None
      }
    };
    groups.push(RepeatGroup {
      seg_idx: Some(gi),
      collapsed,
      row_start: r_start,
      row_end: r_end,
      split_after_row: split,
      min_col_q: min_q,
      max_col_q: max_q,
      badge: badge.to_string(),
    });

    if collapsed {
      let new_idx = nodes.len();
      let c_shape = if c_shape_raw.is_empty() {
        "已收起阶段".to_string()
      } else if c_shape_raw.contains("{B}") {
        c_shape_raw.replace("{B}", if p_batch.is_empty() { "B" } else { p_batch })
      } else {
        c_shape_raw.to_string()
      };
      nodes.push(TensorNode {
        kind: OpKind::Attention,
        shape: c_shape,
        side_note: if c_note.is_empty() { "点击展开" } else { c_note }.to_string(),
        op_title: if c_title.is_empty() { badge } else { c_title }.to_string(),
        formula: if c_formula.is_empty() {
          "点击该汇总卡或右上角「展开 ▾」按钮，展开该阶段内部全部张量算子。"
        } else {
          c_formula
        }
        .to_string(),
        params_hint: c_hint.to_string(),
        col_q: (min_q + max_q) / 2,
        row: r_start,
        focus_target: None,
        layer_target: None,
        toggle_seg: Some(gi),
      });
      col_node[gi] = new_idx;
    }
  }

  for (i, &nl) in n_lines.iter().enumerate() {
    let owner = node_owner[i];
    if owner < g_cnt && col_node[owner] != usize::MAX {
      old_to_new[i] = col_node[owner];
      continue;
    }
    let mut nit = nl.split('\t');
    let _ = nit.next();
    let kind = OpKind::from_u8(next_i(&mut nit, 2) as u8);
    let shape_raw = next_s(&mut nit);
    let dims_csv = next_s(&mut nit);
    let side_note = next_s(&mut nit);
    let op_title = next_s(&mut nit);
    let formula = next_s(&mut nit);
    let params_hint = next_s(&mut nit);
    let col_q = next_i(&mut nit, 0) as i32;
    let row = next_i(&mut nit, 0).max(0) as usize;

    old_to_new[i] = nodes.len();
    nodes.push(TensorNode {
      kind,
      shape: format_wire_shape(shape_raw, dims_csv, shape_mode, p_batch, p_num, p_den),
      side_note: side_note.to_string(),
      op_title: op_title.to_string(),
      formula: formula.to_string(),
      params_hint: params_hint.to_string(),
      col_q,
      row: map_row(row),
      focus_target: None,
      layer_target: None,
      toggle_seg: None,
    });
  }

  let mut edges: Vec<TensorEdge> = Vec::with_capacity(e_lines.len());
  for &el in &e_lines {
    let mut eit = el.split('\t');
    let _ = eit.next();
    let of = next_i(&mut eit, -1);
    let ot = next_i(&mut eit, -1);
    let lbl = next_s(&mut eit);
    let kind = EdgeKind::from_u8(next_i(&mut eit, 0) as u8);
    let lane = next_i(&mut eit, 0).clamp(0, 7) as u8;
    let grp_raw = next_s(&mut eit);
    if of < 0 || ot < 0 {
      continue;
    }
    let (of, ot) = (of as usize, ot as usize);
    if of >= old_to_new.len() || ot >= old_to_new.len() {
      continue;
    }
    let (nf, nt) = (old_to_new[of], old_to_new[ot]);
    if nf == usize::MAX || nt == usize::MAX || nf == nt {
      continue;
    }
    if edges
      .iter()
      .any(|prev| prev.from == nf && prev.to == nt && prev.kind == kind && prev.lane == lane)
    {
      continue;
    }
    let resolved_grp = if !grp_raw.is_empty() {
      let gi = parse_i64_ascii(grp_raw, -1);
      if gi >= 0 && (gi as usize) < g_cnt && !groups[gi as usize].collapsed {
        gi as u8
      } else {
        u8::MAX
      }
    } else if node_owner[of] == node_owner[ot]
      && node_owner[of] < g_cnt
      && !is_col(node_owner[of])
      && kind == EdgeKind::InnerSkip
    {
      node_owner[of] as u8
    } else {
      u8::MAX
    };
    edges.push(TensorEdge {
      from: nf,
      to: nt,
      label: lbl.to_string(),
      kind,
      group_idx: resolved_grp,
      lane,
    });
  }

  let total_rows = nodes.iter().map(|n| n.row + 1).max().unwrap_or(1);
  let mut subtitle = String::with_capacity(sub_base.len() + p_label.len() + 6);
  subtitle.push_str(sub_base);
  if !p_label.is_empty() {
    if !subtitle.is_empty() {
      subtitle.push_str(" · ");
    }
    subtitle.push_str(p_label);
  }

  Some((
    TensorGraph {
      file_slug: slug.to_string(),
      title: title.to_string(),
      subtitle,
      legend_style,
      nodes,
      edges,
      groups,
      total_rows,
    },
    preset_labels,
    default_mask,
  ))
}

// ─────────────────────────────────────────────────────────────────────────────
// 确定性整数网格布局 + 统一 SVG 渲染器（页面与高清导出 100% 共享）
// ─────────────────────────────────────────────────────────────────────────────

const NODE_W: i32 = 158;
const NODE_H: i32 = 34;
const COL_Q_STEP: i32 = 59;
const ROW_STEP: i32 = 72;
const TOP_PAD: i32 = 52;
const LEFT_OUTER_PAD: i32 = 42;
const GRP_LEFT_PAD: i32 = 96;
const GRP_RIGHT_PAD: i32 = 162;
const LEGEND_W: i32 = 224;
const LEGEND_GAP: i32 = 28;
const RIGHT_PAD: i32 = 24;
const BOTTOM_PAD: i32 = 36;

#[derive(Clone, Copy)]
struct GroupRect {
  x: i32,
  y: i32,
  w: i32,
  h: i32,
  entry_y: i32,
  inner_rail_x: i32,
  /// `> 0` 表示 Attention/Mixer 与 FFN/MoE 之间的分隔线 Y 坐标；`0` 表示无分隔线。
  split_y: i32,
}

struct LayoutMetrics {
  canvas_w: i32,
  canvas_h: i32,
  graph_origin_x: i32,
  macro_rail_x: i32,
  max_macro_lane: u8,
  max_aux_lane: u8,
  legend_x: i32,
  legend_y: i32,
  coords: Vec<(i32, i32)>,
  group_rects: Vec<GroupRect>,
}

#[inline(never)]
fn compute_layout(graph: &TensorGraph, legend_len: usize, edge_mask: u8) -> LayoutMetrics {
  let mut min_q = 0i32;
  let mut max_q = 0i32;
  for n in &graph.nodes {
    min_q = min_q.min(n.col_q);
    max_q = max_q.max(n.col_q);
  }
  let mut grp_min_q = min_q;
  let mut grp_max_q = max_q;
  for grp in &graph.groups {
    grp_min_q = grp_min_q.min(grp.min_col_q);
    grp_max_q = grp_max_q.max(grp.max_col_q);
  }
  let eff_min_q = min_q.min(grp_min_q);
  let eff_max_q = max_q.max(grp_max_q);

  let mut max_macro_lane = 0u8;
  let mut max_aux_lane = 0u8;
  for e in &graph.edges {
    match e.kind {
      EdgeKind::MacroSkip => max_macro_lane = max_macro_lane.max(e.lane.min(7)),
      EdgeKind::AuxBypass => max_aux_lane = max_aux_lane.max(e.lane.min(7)),
      _ => {}
    }
  }
  let macro_extra_left = (max_macro_lane as i32) * 20;
  let aux_extra_right = (max_aux_lane as i32) * 20;

  let graph_origin_x =
    LEFT_OUTER_PAD + macro_extra_left + GRP_LEFT_PAD - eff_min_q * COL_Q_STEP + NODE_W / 2;
  let grp_x = graph_origin_x + grp_min_q * COL_Q_STEP - NODE_W / 2 - GRP_LEFT_PAD;
  let grp_right = graph_origin_x + grp_max_q * COL_Q_STEP + NODE_W / 2 + GRP_RIGHT_PAD;
  let grp_w = (grp_right - grp_x).max(NODE_W + 140);
  let macro_rail_x = (grp_x - 22).max(16 + macro_extra_left);
  let right_edge = grp_right.max(graph_origin_x + eff_max_q * COL_Q_STEP + NODE_W / 2 + 86);
  let legend_x = right_edge + aux_extra_right + LEGEND_GAP;
  let canvas_w = legend_x + LEGEND_W + RIGHT_PAD;

  // 逐行累积 Y 坐标：为每个大框顶部/底部与 Attention-FFN 分隔区预留呼吸间距，杜绝连线压标题。
  let mut row_y = Vec::with_capacity(graph.total_rows);
  let mut cur_y = TOP_PAD + NODE_H / 2;
  for r in 0..graph.total_rows {
    if r > 0 {
      let mut dy = ROW_STEP;
      for grp in &graph.groups {
        if grp.row_start == r {
          dy += if grp.collapsed { 38 } else { 58 };
        }
        if grp.row_end + 1 == r {
          dy += 24;
        }
        if grp.split_after_row == Some(r - 1) {
          dy += 18;
        }
      }
      cur_y += dy;
    }
    row_y.push(cur_y);
  }

  let mut coords = Vec::with_capacity(graph.nodes.len());
  for n in &graph.nodes {
    let cy = row_y.get(n.row).copied().unwrap_or(TOP_PAD + NODE_H / 2);
    coords.push((graph_origin_x + n.col_q * COL_Q_STEP, cy));
  }

  let mut group_rects = Vec::with_capacity(graph.groups.len());
  for grp in &graph.groups {
    let y_start = row_y.get(grp.row_start).copied().unwrap_or(TOP_PAD + NODE_H / 2);
    let y_end = row_y.get(grp.row_end).copied().unwrap_or(y_start);
    let top_pad = if grp.collapsed { 20 } else { 42 };
    let y = y_start - NODE_H / 2 - top_pad;
    let h = (y_end - y_start) + NODE_H + top_pad + 16;
    let entry_y = y + 20;
    let inner_rail_x = grp_x + 18;
    let split_y = match grp.split_after_row {
      Some(sr) => match (row_y.get(sr), row_y.get(sr + 1)) {
        (Some(&ya), Some(&yb)) => (ya + yb) / 2,
        _ => 0,
      },
      None => 0,
    };
    group_rects.push(GroupRect { x: grp_x, y, w: grp_w, h, entry_y, inner_rail_x, split_y });
  }

  let edge_items = edge_mask.count_ones() as i32;
  let legend_h = 34 + (legend_len as i32) * 28 + 16 + 34 + edge_items * 24 + 12;
  let nodes_h = row_y.last().copied().unwrap_or(TOP_PAD) + NODE_H / 2 + BOTTOM_PAD + 20;
  let canvas_h = nodes_h.max(TOP_PAD + legend_h + BOTTOM_PAD);

  LayoutMetrics {
    canvas_w,
    canvas_h,
    graph_origin_x,
    macro_rail_x,
    max_macro_lane,
    max_aux_lane,
    legend_x,
    legend_y: TOP_PAD,
    coords,
    group_rects,
  }
}

#[inline(never)]
fn truncate_label(s: &str, max_chars: usize) -> String {
  if s.chars().count() <= max_chars {
    s.to_string()
  } else {
    let mut out: String = s.chars().take(max_chars.saturating_sub(1)).collect();
    out.push('…');
    out
  }
}

fn write_xml_escaped(out: &mut String, s: &str) {
  for ch in s.chars() {
    match ch {
      '&' => out.push_str("&amp;"),
      '<' => out.push_str("&lt;"),
      '>' => out.push_str("&gt;"),
      '"' => out.push_str("&quot;"),
      '\'' => out.push_str("&apos;"),
      _ => out.push(ch),
    }
  }
}

struct SvgTheme {
  grid_g: &'static str,
  grp_rect: &'static str,
  grp_pill: &'static str,
  grp_txt: &'static str,
  grp_hint: &'static str,
  grp_split_line: &'static str,
  grp_split_txt: &'static str,
  edge_main_s: &'static str,
  edge_main_f: &'static str,
  edge_main_t: &'static str,
  edge_inner_s: &'static str,
  edge_inner_f: &'static str,
  edge_inner_t: &'static str,
  edge_macro_s: &'static str,
  edge_macro_f: &'static str,
  edge_macro_t: &'static str,
  edge_aux_s: &'static str,
  edge_aux_f: &'static str,
  edge_aux_t: &'static str,
  leg_bg: &'static str,
  leg_title: &'static str,
  leg_line: &'static str,
  leg_lbl: &'static str,
  note_txt: &'static str,
}

const SVG_STANDALONE: SvgTheme = SvgTheme {
  grid_g: "stroke=\"#e2e8f0\"",
  grp_rect: "fill=\"#f8fafc\" fill-opacity=\"0.72\" stroke=\"#94a3b8\"",
  grp_pill: "fill=\"#eef2ff\" stroke=\"#a5b4fc\"",
  grp_txt: "font-weight=\"700\" fill=\"#312e81\"",
  grp_hint: "font-weight=\"600\" fill=\"#6366f1\"",
  grp_split_line: "stroke=\"#cbd5e1\"",
  grp_split_txt: "font-weight=\"600\" fill=\"#64748b\"",
  edge_main_s: "stroke=\"#334155\"",
  edge_main_f: "fill=\"#334155\"",
  edge_main_t: "fill=\"#334155\" font-weight=\"700\"",
  edge_inner_s: "stroke=\"#6366f1\"",
  edge_inner_f: "fill=\"#6366f1\"",
  edge_inner_t: "fill=\"#4f46e5\" font-weight=\"700\"",
  edge_macro_s: "stroke=\"#059669\"",
  edge_macro_f: "fill=\"#059669\"",
  edge_macro_t: "fill=\"#047857\" font-weight=\"700\"",
  edge_aux_s: "stroke=\"#d97706\"",
  edge_aux_f: "fill=\"#d97706\"",
  edge_aux_t: "fill=\"#b45309\" font-weight=\"700\"",
  leg_bg: "fill=\"#fff\" stroke=\"#cbd5e1\"",
  leg_title: "font-weight=\"700\" fill=\"#0f172a\"",
  leg_line: "stroke=\"#e2e8f0\"",
  leg_lbl: "font-weight=\"600\" fill=\"#1e293b\"",
  note_txt: "font-weight=\"600\" fill=\"#334155\"",
};

const SVG_THEMED: SvgTheme = SvgTheme {
  grid_g: "class=\"stroke-slate-200/80 dark:stroke-slate-800/80\"",
  grp_rect:
    "class=\"fill-slate-50/75 dark:fill-slate-900/55 stroke-slate-400 dark:stroke-slate-600\"",
  grp_pill:
    "class=\"fill-indigo-50/95 dark:fill-indigo-950/80 stroke-indigo-300 dark:stroke-indigo-700\"",
  grp_txt: "class=\"fill-indigo-950 dark:fill-indigo-200 font-bold\"",
  grp_hint: "class=\"fill-indigo-600 dark:fill-indigo-400 font-semibold\"",
  grp_split_line: "class=\"stroke-slate-300 dark:stroke-slate-700\"",
  grp_split_txt: "class=\"fill-slate-500 dark:fill-slate-400 font-semibold\"",
  edge_main_s: "class=\"stroke-slate-700 dark:stroke-slate-400\"",
  edge_main_f: "class=\"fill-slate-700 dark:fill-slate-400\"",
  edge_main_t: "class=\"fill-slate-700 dark:fill-slate-300 font-bold\"",
  edge_inner_s: "class=\"stroke-indigo-500 dark:stroke-indigo-400\"",
  edge_inner_f: "class=\"fill-indigo-500 dark:fill-indigo-400\"",
  edge_inner_t: "class=\"fill-indigo-600 dark:fill-indigo-400 font-bold\"",
  edge_macro_s: "class=\"stroke-emerald-600 dark:stroke-emerald-400\"",
  edge_macro_f: "class=\"fill-emerald-600 dark:fill-emerald-400\"",
  edge_macro_t: "class=\"fill-emerald-700 dark:fill-emerald-400 font-bold\"",
  edge_aux_s: "class=\"stroke-amber-600 dark:stroke-amber-400\"",
  edge_aux_f: "class=\"fill-amber-600 dark:fill-amber-400\"",
  edge_aux_t: "class=\"fill-amber-700 dark:fill-amber-400 font-bold\"",
  leg_bg: "class=\"fill-white/95 dark:fill-slate-900/95 stroke-slate-300 dark:stroke-slate-700\"",
  leg_title: "class=\"fill-slate-900 dark:fill-slate-100 font-bold\"",
  leg_line: "class=\"stroke-slate-200 dark:stroke-slate-800\"",
  leg_lbl: "class=\"fill-slate-800 dark:fill-slate-200 font-semibold\"",
  note_txt: "class=\"fill-slate-700 dark:fill-slate-300 font-semibold\"",
};

#[inline(never)]
fn push_attr_i32(out: &mut String, k: &str, v: i32) {
  out.push(' ');
  out.push_str(k);
  out.push_str("=\"");
  push_i32(out, v);
  out.push('"');
}

#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn push_rect(out: &mut String, x: i32, y: i32, w: i32, h: i32, rx: i32, style: &str, extra: &str) {
  out.push_str("<rect");
  push_attr_i32(out, "x", x);
  push_attr_i32(out, "y", y);
  push_attr_i32(out, "width", w);
  push_attr_i32(out, "height", h);
  push_attr_i32(out, "rx", rx);
  if !style.is_empty() {
    out.push(' ');
    out.push_str(style);
  }
  if !extra.is_empty() {
    out.push(' ');
    out.push_str(extra);
  }
  out.push_str("/>");
}

#[inline(never)]
fn push_line(out: &mut String, x1: i32, y1: i32, x2: i32, y2: i32, style: &str, extra: &str) {
  out.push_str("<line");
  push_attr_i32(out, "x1", x1);
  push_attr_i32(out, "y1", y1);
  push_attr_i32(out, "x2", x2);
  push_attr_i32(out, "y2", y2);
  if !style.is_empty() {
    out.push(' ');
    out.push_str(style);
  }
  if !extra.is_empty() {
    out.push(' ');
    out.push_str(extra);
  }
  out.push_str("/>");
}

#[inline(never)]
fn push_circle(out: &mut String, cx: i32, cy: i32, r: i32, tail: &str) {
  out.push_str("<circle");
  push_attr_i32(out, "cx", cx);
  push_attr_i32(out, "cy", cy);
  push_attr_i32(out, "r", r);
  out.push(' ');
  out.push_str(tail);
  out.push_str("/>");
}

#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn push_filled_rect(
  out: &mut String,
  x: i32,
  y: i32,
  w: i32,
  h: i32,
  rx: i32,
  fill: &str,
  stroke: &str,
  extra: &str,
) {
  out.push_str("<rect");
  push_attr_i32(out, "x", x);
  push_attr_i32(out, "y", y);
  push_attr_i32(out, "width", w);
  push_attr_i32(out, "height", h);
  push_attr_i32(out, "rx", rx);
  out.push_str(" fill=\"");
  out.push_str(fill);
  out.push_str("\" stroke=\"");
  out.push_str(stroke);
  out.push('"');
  if !extra.is_empty() {
    out.push(' ');
    out.push_str(extra);
  }
  out.push_str("/>");
}

#[inline(never)]
fn push_cmd_xy(out: &mut String, cmd: &str, x: i32, y: i32) {
  out.push_str(cmd);
  push_i32(out, x);
  out.push(' ');
  push_i32(out, y);
}

#[inline(never)]
fn push_pt(out: &mut String, x: i32, y: i32) {
  push_i32(out, x);
  out.push(',');
  push_i32(out, y);
}

#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn push_text(
  out: &mut String,
  x: i32,
  y: i32,
  size: &str,
  extra: &str,
  style: &str,
  text: &str,
  escape: bool,
) {
  out.push_str("<text");
  push_attr_i32(out, "x", x);
  push_attr_i32(out, "y", y);
  if !extra.is_empty() {
    out.push(' ');
    out.push_str(extra);
  }
  out.push_str(" font-size=\"");
  out.push_str(size);
  out.push_str("\" ");
  out.push_str(style);
  out.push('>');
  if escape {
    write_xml_escaped(out, text);
  } else {
    out.push_str(text);
  }
  out.push_str("</text>");
}

#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn push_arrow_line(
  out: &mut String,
  x1: i32,
  y1: i32,
  x2: i32,
  y2: i32,
  stroke_attr: &str,
  fill_attr: &str,
  dash: &str,
) {
  let dx = (x2 - x1) as f32;
  let dy = (y2 - y1) as f32;
  let len = (dx * dx + dy * dy).sqrt().max(1.0);
  let ux = dx / len;
  let uy = dy / len;
  let bx = (x2 as f32 - ux * 8.5).round() as i32;
  let by = (y2 as f32 - uy * 8.5).round() as i32;
  let px = (-uy * 4.0).round() as i32;
  let py = (ux * 4.0).round() as i32;
  out.push_str("<path d=\"");
  push_cmd_xy(out, "M ", x1, y1);
  push_cmd_xy(out, " L ", bx, by);
  out.push_str("\" fill=\"none\" ");
  out.push_str(stroke_attr);
  out.push_str(" stroke-width=\"1.5\"");
  if !dash.is_empty() {
    out.push_str(" stroke-dasharray=\"");
    out.push_str(dash);
    out.push('"');
  }
  out.push_str("/><polygon points=\"");
  push_pt(out, x2, y2);
  out.push(' ');
  push_pt(out, bx + px, by + py);
  out.push(' ');
  push_pt(out, bx - px, by - py);
  out.push_str("\" ");
  out.push_str(fill_attr);
  out.push_str("/>");
}

#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn push_rail_skip(
  out: &mut String,
  sx: i32,
  sy: i32,
  rail_x: i32,
  ex: i32,
  ey: i32,
  dot_start: bool,
  label: &str,
  stroke_attr: &str,
  fill_attr: &str,
  text_attr: &str,
) {
  if dot_start {
    push_circle(out, sx, sy, 3, fill_attr);
  }
  let dir = if rail_x < ex { 1 } else { -1 };
  let tip_x = ex - dir * 7;
  out.push_str("<path d=\"");
  push_cmd_xy(out, "M ", sx, sy);
  push_cmd_xy(out, " L ", rail_x, sy);
  push_cmd_xy(out, " L ", rail_x, ey);
  push_cmd_xy(out, " L ", tip_x, ey);
  out.push_str("\" fill=\"none\" ");
  out.push_str(stroke_attr);
  out.push_str(" stroke-width=\"1.5\" stroke-dasharray=\"4 3\"/><polygon points=\"");
  push_pt(out, ex, ey);
  out.push(' ');
  push_pt(out, ex - dir * 8, ey - 4);
  out.push(' ');
  push_pt(out, ex - dir * 8, ey + 4);
  out.push_str("\" ");
  out.push_str(fill_attr);
  out.push_str("/>");
  if !label.is_empty() {
    let tx = rail_x + 5;
    push_text(out, tx, (sy + ey) / 2 + 3, "9", "", text_attr, label, true);
  }
}

#[inline(never)]
fn write_group_header_svg(
  out: &mut String,
  grp: &RepeatGroup,
  gr: GroupRect,
  origin_x: i32,
  interactive: bool,
  th: &SvgTheme,
) {
  let pill_y = gr.y - 26;
  // 标题胶囊严格留在中轴主干线左侧，按实际文本像素自适应宽度避免无谓截断。
  let max_left_w = (origin_x - gr.x - 18).max(200);
  let mut title = String::with_capacity(grp.badge.len() + 4);
  let mut used_px = 0i32;
  if grp.seg_idx.is_some() {
    title.push_str(if grp.collapsed { "▸ " } else { "▾ " });
    used_px = 14;
  }
  let max_txt_px = max_left_w - 20;
  for ch in grp.badge.chars() {
    let cw = if ch.is_ascii() { 6 } else { 10 };
    if used_px + cw > max_txt_px {
      title.pop();
      title.push('…');
      used_px = max_txt_px;
      break;
    }
    used_px += cw;
    title.push(ch);
  }
  let pill_w = (used_px + 20).clamp(180, max_left_w);
  push_rect(out, gr.x + 8, pill_y, pill_w, 20, 5, th.grp_pill, "stroke-width=\"1\"");
  push_text(out, gr.x + 16, pill_y + 14, "9.5", "", th.grp_txt, &title, true);
  // 交互态将收起/展开按钮独立放置在大框右上角，避免挤压左半边标题空间。
  if interactive && grp.seg_idx.is_some() {
    let btn_w = 54;
    let btn_x = gr.x + gr.w - btn_w - 8;
    let hint = if grp.collapsed { "展开 ▾" } else { "收起 ▴" };
    push_rect(out, btn_x, pill_y, btn_w, 20, 5, th.grp_pill, "stroke-width=\"1\"");
    push_text(
      out,
      btn_x + btn_w / 2,
      pill_y + 14,
      "9",
      "text-anchor=\"middle\"",
      th.grp_hint,
      hint,
      false,
    );
  }
}

#[inline(never)]
fn write_node_svg(
  out: &mut String,
  n: &TensorNode,
  cx: i32,
  cy: i32,
  selected: bool,
  th: &SvgTheme,
) {
  let (fill, stroke, tc) = n.kind.palette();
  let rx = cx - NODE_W / 2;
  let ry = cy - NODE_H / 2;
  if selected {
    push_rect(
      out,
      rx - 4,
      ry - 4,
      NODE_W + 8,
      NODE_H + 8,
      8,
      "fill=\"none\" class=\"stroke-indigo-600 dark:stroke-amber-400\"",
      "stroke-width=\"2.2\"",
    );
  }
  push_filled_rect(
    out,
    rx,
    ry,
    NODE_W,
    NODE_H,
    5,
    fill,
    stroke,
    if n.toggle_seg.is_some() {
      "stroke-width=\"1.2\" stroke-dasharray=\"5 3\""
    } else {
      "stroke-width=\"1.2\""
    },
  );
  out.push_str("<text");
  push_attr_i32(out, "x", cx);
  push_attr_i32(out, "y", cy + 4);
  out.push_str(" text-anchor=\"middle\" font-size=\"11.5\" font-weight=\"700\" fill=\"");
  out.push_str(tc);
  out.push_str("\">");
  write_xml_escaped(out, &truncate_label(&n.shape, 20));
  out.push_str("</text>");
  if !n.side_note.is_empty() {
    push_text(
      out,
      rx + NODE_W + 8,
      cy + 4,
      "10.5",
      "",
      th.note_txt,
      &truncate_label(&n.side_note, 13),
      true,
    );
  }
}

#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn push_edge_legend_row(
  out: &mut String,
  lx: i32,
  ry: i32,
  s_attr: &str,
  f_attr: &str,
  dash: &str,
  lbl: &str,
  th: &SvgTheme,
) {
  push_line(out, lx, ry, lx + LEGEND_W, ry, th.leg_line, "");
  push_arrow_line(out, lx + 10, ry + 12, lx + 54, ry + 12, s_attr, f_attr, dash);
  push_text(out, lx + 64, ry + 16, "10", "", th.leg_lbl, lbl, false);
}

#[inline(never)]
fn write_static_svg_layers(
  out: &mut String,
  graph: &TensorGraph,
  layout: &LayoutMetrics,
  legend: &[(OpKind, &'static str)],
  edge_mask: u8,
  draw_group_headers: bool,
  th: &SvgTheme,
) {
  out.push_str("<g ");
  out.push_str(th.grid_g);
  out.push_str(" stroke-width=\"0.75\">");
  let mut gx = 32;
  while gx < layout.canvas_w {
    push_line(out, gx, 0, gx, layout.canvas_h, "", "");
    gx += 32;
  }
  let mut gy = 32;
  while gy < layout.canvas_h {
    push_line(out, 0, gy, layout.canvas_w, gy, "", "");
    gy += 32;
  }
  out.push_str("</g>");

  // 1. 绘制大框容器与 Attention/FFN 轻量内部子块分隔线
  for &gr in &layout.group_rects {
    push_rect(
      out,
      gr.x,
      gr.y,
      gr.w,
      gr.h,
      10,
      th.grp_rect,
      "stroke-width=\"1.2\" stroke-dasharray=\"5 4\"",
    );

    if gr.split_y > 0 {
      let sy = gr.split_y;
      push_line(
        out,
        gr.inner_rail_x + 42,
        sy,
        gr.x + gr.w - 10,
        sy,
        th.grp_split_line,
        "stroke-width=\"1\" stroke-dasharray=\"3 3\"",
      );
      let rx = gr.x + gr.w - 12;
      push_text(
        out,
        rx,
        sy - 5,
        "9",
        "text-anchor=\"end\"",
        th.grp_split_txt,
        "① Attention / Mixer",
        false,
      );
      push_text(
        out,
        rx,
        sy + 12,
        "9",
        "text-anchor=\"end\"",
        th.grp_split_txt,
        "② FFN / MoE",
        false,
      );
    }
  }

  // 2. 绘制全部语义连线（含展开大框内部入口锚点 h_in、框内左轨残差、多泳道跨块绿轨、旁路橙线）
  let mut groups_with_entry_skip = 0u64;
  for edge in &graph.edges {
    if edge.kind == EdgeKind::InnerSkip && edge.group_idx != u8::MAX {
      let gi = edge.group_idx as usize;
      if let (Some(grp), Some(fn_n)) = (graph.groups.get(gi), graph.nodes.get(edge.from)) {
        if !grp.collapsed && fn_n.row < grp.row_start {
          groups_with_entry_skip |= 1u64 << (gi & 63);
        }
      }
    }
  }
  let mut entry_drawn_mask = 0u64;
  for edge in &graph.edges {
    if let (Some(&(x1, y1_c)), Some(&(x2, y2_c)), Some(fn_n), Some(tn_n)) = (
      layout.coords.get(edge.from),
      layout.coords.get(edge.to),
      graph.nodes.get(edge.from),
      graph.nodes.get(edge.to),
    ) {
      let (s_attr, f_attr, t_attr) = match edge.kind {
        EdgeKind::Main => (th.edge_main_s, th.edge_main_f, th.edge_main_t),
        EdgeKind::InnerSkip => (th.edge_inner_s, th.edge_inner_f, th.edge_inner_t),
        EdgeKind::MacroSkip => (th.edge_macro_s, th.edge_macro_f, th.edge_macro_t),
        EdgeKind::AuxBypass => (th.edge_aux_s, th.edge_aux_f, th.edge_aux_t),
      };

      let owned_grp = if edge.group_idx != u8::MAX {
        let gi = edge.group_idx as usize;
        match (graph.groups.get(gi), layout.group_rects.get(gi)) {
          (Some(g), Some(&gr)) if !g.collapsed && g.seg_idx.is_some() => Some((gi, g, gr)),
          _ => None,
        }
      } else {
        None
      };

      match edge.kind {
        EdgeKind::InnerSkip => {
          if let Some((_, grp, gr)) = owned_grp {
            let ex = x2 - NODE_W / 2;
            if fn_n.row < grp.row_start {
              // 第 1 条子层残差：从框内入口锚点 h_in 出发，完全闭合在 RepeatGroup 内部！
              push_rail_skip(
                out,
                layout.graph_origin_x,
                gr.entry_y,
                gr.inner_rail_x,
                ex,
                y2_c - 4,
                false,
                &edge.label,
                s_attr,
                f_attr,
                t_attr,
              );
            } else {
              // 第 2 条子层残差：从前一个 Add (+) 左侧偏下圆点出发，与上一条终点错开 9px 永不粘连！
              let sx = x1 - NODE_W / 2;
              push_rail_skip(
                out,
                sx,
                y1_c + 5,
                gr.inner_rail_x,
                ex,
                y2_c,
                true,
                &edge.label,
                s_attr,
                f_attr,
                t_attr,
              );
            }
          } else {
            push_rail_skip(
              out,
              x1 - NODE_W / 2,
              y1_c,
              layout.macro_rail_x,
              x2 - NODE_W / 2,
              y2_c,
              true,
              &edge.label,
              s_attr,
              f_attr,
              t_attr,
            );
          }
        }
        EdgeKind::MacroSkip => {
          if tn_n.row > fn_n.row + 1 {
            let rail_x = layout.macro_rail_x - (edge.lane.min(7) as i32) * 20;
            if layout.max_macro_lane == 0 {
              push_rail_skip(
                out,
                x1 - NODE_W / 2,
                y1_c,
                rail_x,
                x2 - NODE_W / 2,
                y2_c,
                true,
                &edge.label,
                s_attr,
                f_attr,
                t_attr,
              );
            } else {
              // 多泳道并行长跳连（如 U-Net 4 级嵌套 / YOLO P3-P5）：轨道按 lane×20px 平行错开，标签置于起点水平出线上方防撞。
              push_rail_skip(
                out,
                x1 - NODE_W / 2,
                y1_c,
                rail_x,
                x2 - NODE_W / 2,
                y2_c,
                true,
                "",
                s_attr,
                f_attr,
                t_attr,
              );
              if !edge.label.is_empty() {
                push_text(
                  out,
                  x1 - NODE_W / 2 - 8,
                  y1_c - 5,
                  "9",
                  "text-anchor=\"end\"",
                  t_attr,
                  &edge.label,
                  true,
                );
              }
            }
          } else {
            let y1 = y1_c + NODE_H / 2;
            let y2 = y2_c - NODE_H / 2;
            push_arrow_line(out, x1, y1, x2, y2, s_attr, f_attr, "5 3");
            if !edge.label.is_empty() {
              push_text(
                out,
                (x1 + x2) / 2 - 26,
                (y1 + y2) / 2,
                "9.5",
                "",
                t_attr,
                &edge.label,
                true,
              );
            }
          }
        }
        EdgeKind::AuxBypass => {
          if tn_n.row > fn_n.row + 1 {
            // 右侧旁路绕行（如 MLA 的 k_pe 64d 或多泳道右侧特征传递；两端均上偏 13px 避开右侧文字）
            let sx = x1 + NODE_W / 2;
            let ex = x2 + NODE_W / 2;
            let rail_x = sx.max(ex) + 98 + (edge.lane.min(7) as i32) * 20;
            if layout.max_aux_lane == 0 {
              push_rail_skip(
                out,
                sx,
                y1_c - 13,
                rail_x,
                ex,
                y2_c - 13,
                true,
                &edge.label,
                s_attr,
                f_attr,
                t_attr,
              );
            } else {
              push_rail_skip(
                out,
                sx,
                y1_c - 13,
                rail_x,
                ex,
                y2_c - 13,
                true,
                "",
                s_attr,
                f_attr,
                t_attr,
              );
              if !edge.label.is_empty() {
                push_text(out, sx + 6, y1_c - 18, "9", "", t_attr, &edge.label, true);
              }
            }
          } else if tn_n.row <= fn_n.row {
            // 同行水平路由/加权控制边（如 Router -> Experts dispatch、Top-K -> exp_down × g_i；上偏 12px 避开侧注）
            let sx = x1 + NODE_W / 2;
            let sy = y1_c - 12;
            let ex = x2 - NODE_W / 2;
            let ey = y2_c - 12;
            push_arrow_line(out, sx, sy, ex, ey, s_attr, f_attr, "3 3");
            if !edge.label.is_empty() {
              push_text(
                out,
                (sx + ex) / 2,
                sy.min(ey) - 5,
                "9",
                "text-anchor=\"middle\"",
                t_attr,
                &edge.label,
                true,
              );
            }
          } else {
            // 向下旁路分支（如 MTP；标签沿法向偏出斜线，杜绝线段穿字）
            let y1 = y1_c + NODE_H / 2;
            let y2 = y2_c - NODE_H / 2;
            push_arrow_line(out, x1, y1, x2, y2, s_attr, f_attr, "4 3");
            if !edge.label.is_empty() {
              let (lx, ly, anchor) = if x2 > x1 {
                ((x1 + x2) / 2 + 14, (y1 + y2) / 2 - 4, "")
              } else if x2 < x1 {
                ((x1 + x2) / 2 - 14, (y1 + y2) / 2 - 4, "text-anchor=\"end\"")
              } else {
                (x1 + 8, (y1 + y2) / 2 + 3, "")
              };
              push_text(out, lx, ly, "9.5", anchor, t_attr, &edge.label, true);
            }
          }
        }
        EdgeKind::Main => {
          if let Some((gi, grp, gr)) = owned_grp {
            let bit = 1u64 << (gi & 63);
            if (groups_with_entry_skip & bit) != 0
              && fn_n.row < grp.row_start
              && tn_n.row == grp.row_start
            {
              // 主干先垂直进入大框顶部内侧的 h_in 入口锚点，再从 h_in 分叉，杜绝跨框斜线压标题！
              let jx = layout.graph_origin_x;
              let jy = gr.entry_y;
              if (entry_drawn_mask & bit) == 0 {
                entry_drawn_mask |= bit;
                push_line(out, x1, y1_c + NODE_H / 2, jx, jy, s_attr, "stroke-width=\"1.5\"");
                push_circle(out, jx, jy, 3, th.edge_inner_f);
                push_text(out, jx + 8, jy - 4, "9", "", th.edge_inner_t, "h_in", false);
              }
              push_arrow_line(out, jx, jy, x2, y2_c - NODE_H / 2, s_attr, f_attr, "");
              continue;
            }
          }
          let y1 = y1_c + NODE_H / 2;
          let y2 = y2_c - NODE_H / 2;
          push_arrow_line(out, x1, y1, x2, y2, s_attr, f_attr, "");
          if !edge.label.is_empty() {
            push_text(
              out,
              (x1 + x2) / 2 + 8,
              (y1 + y2) / 2 + 3,
              "9.5",
              "",
              t_attr,
              &edge.label,
              true,
            );
          }
        }
      }
    }
  }

  // 2.5 将大框标题胶囊置于连线之上（与交互模式层叠一致），防止多路斜线穿过标题文字
  for (gi, grp) in graph.groups.iter().enumerate() {
    if let Some(&gr) = layout.group_rects.get(gi) {
      if draw_group_headers || grp.seg_idx.is_none() {
        write_group_header_svg(out, grp, gr, layout.graph_origin_x, false, th);
      }
    }
  }

  // 3. 右上角算子色块图例 + 动态连线语义图例
  let lx = layout.legend_x;
  let ly = layout.legend_y;
  let lh = 30 + (legend.len() as i32) * 28;
  push_rect(out, lx, ly, LEGEND_W, lh, 6, th.leg_bg, "stroke-width=\"1.2\"");
  push_text(out, lx + 12, ly + 20, "11", "", th.leg_title, "算子色块图例 (Legend)", false);
  for (i, &(kind, lbl)) in legend.iter().enumerate() {
    let ry = ly + 30 + (i as i32) * 28;
    let (fill, stroke, _) = kind.palette();
    push_line(out, lx, ry, lx + LEGEND_W, ry, th.leg_line, "");
    push_filled_rect(out, lx + 10, ry + 5, 46, 18, 3, fill, stroke, "stroke-width=\"1\"");
    push_text(out, lx + 64, ry + 18, "10.5", "", th.leg_lbl, lbl, false);
  }

  let edge_count = edge_mask.count_ones() as i32;
  if edge_count > 0 {
    let ey0 = ly + lh + 14;
    let eh = 28 + edge_count * 24 + 4;
    push_rect(out, lx, ey0, LEGEND_W, eh, 6, th.leg_bg, "stroke-width=\"1.2\"");
    push_text(out, lx + 12, ey0 + 19, "11", "", th.leg_title, "连线语义图例 (Edges)", false);
    let mut ry = ey0 + 28;
    if (edge_mask & 1) != 0 {
      push_edge_legend_row(out, lx, ry, th.edge_main_s, th.edge_main_f, "", "主张量数据流", th);
      ry += 24;
    }
    if (edge_mask & 2) != 0 {
      push_edge_legend_row(
        out,
        lx,
        ry,
        th.edge_inner_s,
        th.edge_inner_f,
        "4 3",
        "层内残差 (框内 Skip)",
        th,
      );
      ry += 24;
    }
    if (edge_mask & 4) != 0 {
      push_edge_legend_row(
        out,
        lx,
        ry,
        th.edge_macro_s,
        th.edge_macro_f,
        "5 3",
        "跨块长跳连 (外部 Skip)",
        th,
      );
      ry += 24;
    }
    if (edge_mask & 8) != 0 {
      push_edge_legend_row(
        out,
        lx,
        ry,
        th.edge_aux_s,
        th.edge_aux_f,
        "3 3",
        "旁路 / 路由 / MTP",
        th,
      );
    }
  }
}

#[inline(never)]
fn percent_encode_svg(svg: &str) -> String {
  let mut out = String::with_capacity(svg.len() + svg.len() / 4);
  for b in svg.bytes() {
    match b {
      b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => out.push(b as char),
      _ => {
        out.push('%');
        let hex = |v: u8| if v < 10 { (b'0' + v) as char } else { (b'A' + v - 10) as char };
        out.push(hex(b >> 4));
        out.push(hex(b & 0x0f));
      }
    }
  }
  out
}

/// 生成独立可下载的高清 `.svg` 矢量文件源码。
#[inline(never)]
pub fn export_standalone_svg(graph: &TensorGraph) -> String {
  let legend = graph.active_legend();
  let edge_mask = graph.active_edge_mask();
  let layout = compute_layout(graph, legend.len(), edge_mask);
  let mut out = String::with_capacity(12_288);
  out.push_str("<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 ");
  push_i32(&mut out, layout.canvas_w);
  out.push(' ');
  push_i32(&mut out, layout.canvas_h);
  out.push('"');
  push_attr_i32(&mut out, "width", layout.canvas_w);
  push_attr_i32(&mut out, "height", layout.canvas_h);
  out.push_str(
    " style=\"background:#fff;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace\">",
  );
  write_static_svg_layers(&mut out, graph, &layout, &legend, edge_mask, true, &SVG_STANDALONE);
  out.push_str("<text x=\"24\" y=\"28\" font-size=\"14\" font-weight=\"700\" fill=\"#0f172a\">");
  write_xml_escaped(&mut out, &graph.title);
  out.push_str("</text>");
  for (idx, n) in graph.nodes.iter().enumerate() {
    if let Some(&(cx, cy)) = layout.coords.get(idx) {
      write_node_svg(&mut out, n, cx, cy, false, &SVG_STANDALONE);
    }
  }
  out.push_str("</svg>");
  out
}

const SHAPE_PRESETS: &[ShapePreset] =
  &[ShapePreset::Symbolic, ShapePreset::Decode1, ShapePreset::Prefill512, ShapePreset::Batch4K];

/// 交互式「全网张量维流图（Tensor Flow DAG）」组件。
#[component]
pub fn TensorFlowGraphView(
  blueprint: ModelBlueprint,
  on_drill_to_workbench: EventHandler<(Option<u64>, InspectorFocus)>,
) -> Element {
  // 复用全局已单态化的基础类型 Signal（u8 / bool / usize），避免为私有枚举额外膨胀 WASM 体积。
  let mut preset_id: Signal<u8> = use_signal(|| 0u8);
  let mut compact = use_signal(|| false);
  let mut selected_idx: Signal<usize> = use_signal(|| usize::MAX);
  let mut collapsed_mask: Signal<usize> = use_signal(|| 0usize);

  let cur_mask = collapsed_mask();
  let cur_preset =
    SHAPE_PRESETS.get(preset_id() as usize).copied().unwrap_or(ShapePreset::Symbolic);
  let (graph, preset_labels, is_declarative, decl_default_mask) = if let Some((g, p_lbls, d_mask)) =
    blueprint.declarative_graph_wire.as_deref().and_then(|w| {
      build_declarative_graph_from_wire(w, &blueprint.repo_id, preset_id() as usize, cur_mask)
    }) {
    (g, p_lbls, true, d_mask)
  } else {
    (
      build_blueprint_tensor_graph_with_collapse(&blueprint, cur_preset, compact(), cur_mask),
      SHAPE_PRESETS.iter().map(|p| p.label().to_string()).collect(),
      false,
      0,
    )
  };

  let seg_count = graph.groups.len().min(30);
  let all_collapsed_mask = if seg_count == 0 { 0 } else { (1usize << seg_count) - 1 };
  let effective_mask = if is_declarative { cur_mask ^ decl_default_mask } else { cur_mask };
  let all_collapsed = seg_count > 0 && (effective_mask & all_collapsed_mask) == all_collapsed_mask;

  let legend = graph.active_legend();
  let edge_mask = graph.active_edge_mask();
  let layout = compute_layout(&graph, legend.len(), edge_mask);

  let mut static_layers = String::with_capacity(10_240);
  write_static_svg_layers(
    &mut static_layers,
    &graph,
    &layout,
    &legend,
    edge_mask,
    false,
    &SVG_THEMED,
  );

  const SEG_HDR_BASE: usize = 0x10000;
  const SEG_NODE_BASE: usize = 0x20000;
  let raw_sel = selected_idx();
  let active_idx = if raw_sel < graph.nodes.len() {
    raw_sel
  } else {
    graph
      .nodes
      .iter()
      .position(|n| matches!(n.kind, OpKind::Attention | OpKind::Merge))
      .unwrap_or(0)
  };

  // 合并大块折叠标题栏与节点色块为单一 `Vec<(usize, String)>`，零额外泛型与 RSX 循环开销。
  let mut interactive_svgs: Vec<(usize, String)> =
    Vec::with_capacity(graph.groups.len() + graph.nodes.len());
  for (gi, grp) in graph.groups.iter().enumerate() {
    if let (Some(seg_idx), Some(&gr)) = (grp.seg_idx, layout.group_rects.get(gi)) {
      let mut buf = String::with_capacity(256);
      write_group_header_svg(&mut buf, grp, gr, layout.graph_origin_x, true, &SVG_THEMED);
      interactive_svgs.push((SEG_HDR_BASE + seg_idx, buf));
    }
  }
  for (idx, n) in graph.nodes.iter().enumerate() {
    let mut buf = String::with_capacity(256);
    if let Some(&(cx, cy)) = layout.coords.get(idx) {
      write_node_svg(&mut buf, n, cx, cy, idx == active_idx, &SVG_THEMED);
    }
    let code = match n.toggle_seg {
      Some(seg_idx) => SEG_NODE_BASE + seg_idx,
      None => idx,
    };
    interactive_svgs.push((code, buf));
  }

  let standalone_svg = export_standalone_svg(&graph);
  let mut svg_data_uri = String::with_capacity(standalone_svg.len() + 48);
  svg_data_uri.push_str("data:image/svg+xml;charset=utf-8,");
  svg_data_uri.push_str(&percent_encode_svg(&standalone_svg));
  let mut download_filename = String::with_capacity(graph.file_slug.len() + 18);
  download_filename.push_str(&graph.file_slug);
  download_filename.push_str("-tensor-graph.svg");

  rsx! {
    div { class: card_class("p-4 sm:p-5 bg-white dark:bg-slate-900/70 space-y-4"),
      div { class: "flex flex-wrap items-center justify-between gap-3 pb-3.5 border-b border-slate-200 dark:border-slate-800",
        div { class: "flex flex-wrap items-center gap-2",
          span { class: "text-xs font-bold text-indigo-600 dark:text-indigo-400",
            if is_declarative { "全谱系张量维流图" } else { "全网张量维流图" }
          }
          div { class: "inline-flex items-center rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-950 p-0.5",
            for (p_idx, p_lbl) in preset_labels.iter().enumerate() {
              button {
                key: "{p_lbl}",
                r#type: "button",
                class: if preset_id() == p_idx as u8 {
                  "px-2 py-1 rounded-md text-[11px] font-mono font-bold bg-slate-900 text-white dark:bg-white dark:text-slate-900 cursor-pointer"
                } else {
                  "px-2 py-1 rounded-md text-[11px] font-mono font-medium text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white cursor-pointer"
                },
                onclick: move |_| preset_id.set(p_idx as u8),
                "{p_lbl}"
              }
            }
          }
        }

        div { class: "flex flex-wrap items-center gap-2",
          if !is_declarative {
            button {
              r#type: "button",
              class: "px-2.5 py-1.5 rounded-lg text-xs font-semibold border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 text-slate-700 dark:text-slate-200 hover:border-indigo-400 cursor-pointer",
              onclick: move |_| compact.set(!compact()),
              if compact() { "展开 Norm 节点" } else { "紧凑主干流" }
            }
          }

          if seg_count > 0 {
            button {
              r#type: "button",
              class: "px-2.5 py-1.5 rounded-lg text-xs font-semibold border border-indigo-200 dark:border-indigo-800 bg-indigo-50/70 dark:bg-indigo-950/50 text-indigo-700 dark:text-indigo-300 hover:bg-indigo-100 dark:hover:bg-indigo-950/80 cursor-pointer",
              onclick: move |_| {
                let target_eff = if all_collapsed { 0 } else { all_collapsed_mask };
                collapsed_mask.set(if is_declarative {
                  target_eff ^ decl_default_mask
                } else {
                  target_eff
                });
                selected_idx.set(usize::MAX);
              },
              if all_collapsed { "展开全部大块" } else { "收起全部大块" }
            }
          }

          a {
            href: "{svg_data_uri}",
            download: "{download_filename}",
            class: "inline-flex items-center gap-1 px-2.5 py-1.5 rounded-lg text-xs font-semibold border border-emerald-300 dark:border-emerald-800 bg-emerald-50 dark:bg-emerald-950/40 text-emerald-700 dark:text-emerald-300 hover:bg-emerald-100 dark:hover:bg-emerald-950/70 transition-colors",
            "导出高清 SVG ↓"
          }
        }
      }

      div { class: "flex flex-wrap items-baseline justify-between gap-2",
        div {
          h3 { class: "text-sm sm:text-[15px] font-bold text-slate-900 dark:text-white", "{graph.title}" }
          p { class: "text-xs text-slate-500 dark:text-slate-400 mt-0.5 font-mono", "{graph.subtitle}" }
        }
        span { class: "text-xs text-slate-500 dark:text-slate-400",
          "共 {graph.nodes.len()} 个张量算子节点 · 点击大块标题可展开/收起，点击色块查看公式"
        }
      }

      div { class: "w-full overflow-x-auto rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950/90 p-2 sm:p-3",
        svg {
          view_box: "0 0 {layout.canvas_w} {layout.canvas_h}",
          class: "w-full min-w-[780px] h-auto font-mono select-none",
          role: "img",
          "aria-label": "{graph.title}",

          g { dangerous_inner_html: "{static_layers}" }

          for (code, item_svg) in interactive_svgs {
            g {
              key: "{code}",
              class: "cursor-pointer",
              onclick: move |_| {
                if code >= SEG_HDR_BASE {
                  let seg_idx = code & 0xFFFF;
                  collapsed_mask.set(collapsed_mask() ^ (1usize << seg_idx));
                  selected_idx.set(usize::MAX);
                } else {
                  selected_idx.set(code);
                }
              },
              dangerous_inner_html: "{item_svg}",
            }
          }
        }
      }

      if let Some(node) = graph.nodes.get(active_idx) {
        {
          let (fill, _, tc) = node.kind.palette();
          let mut badge_style = String::with_capacity(48);
          badge_style.push_str("background-color:");
          badge_style.push_str(fill);
          badge_style.push_str(";color:");
          badge_style.push_str(tc);
          badge_style.push(';');
          let kind_lbl = node.kind.legend_label(graph.legend_style);
          let drill_focus = node.focus_target;
          let drill_layer = node.layer_target;
          rsx! {
            div { class: "rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/80 dark:bg-slate-950/60 p-3.5 sm:p-4 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3",
              div { class: "space-y-1 min-w-0",
                div { class: "flex flex-wrap items-center gap-2",
                  span {
                    class: "px-2 py-0.5 rounded text-[11px] font-bold shrink-0",
                    style: "{badge_style}",
                    "{kind_lbl}"
                  }
                  span { class: "text-[13px] font-bold text-slate-900 dark:text-white", "{node.op_title}" }
                  span { class: "px-2 py-0.5 rounded bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 text-xs font-mono font-bold text-indigo-600 dark:text-indigo-400", "{node.shape}" }
                  if !node.side_note.is_empty() {
                    span { class: "text-xs font-mono text-slate-500 dark:text-slate-400", "({node.side_note})" }
                  }
                }
                div { class: "text-[13px] font-mono text-slate-700 dark:text-slate-300", "{node.formula}" }
                if !node.params_hint.is_empty() {
                  div { class: "text-xs text-slate-500 dark:text-slate-400", "{node.params_hint}" }
                }
              }

              if let Some(target_focus) = drill_focus {
                button {
                  r#type: "button",
                  class: "shrink-0 px-3 py-1.5 rounded-lg text-xs font-semibold bg-indigo-600 text-white hover:bg-indigo-700 transition-colors cursor-pointer",
                  onclick: move |_| on_drill_to_workbench.call((drill_layer, target_focus)),
                  "进入分层工作台查看该算子 →"
                }
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
  use crate::arch::{load_builtin_preset, BUILTIN_PRESETS};

  #[test]
  fn builds_valid_tensor_graphs_for_all_builtin_presets_and_classic_cnns() {
    for meta in BUILTIN_PRESETS {
      let slug = meta.id;
      let bp = load_builtin_preset(slug).expect("builtin preset");
      for preset in [ShapePreset::Symbolic, ShapePreset::Prefill512] {
        let graph = build_blueprint_tensor_graph(&bp, preset, false);
        assert!(graph.nodes.len() >= 6, "{slug} should have >= 6 nodes");
        assert!(!graph.edges.is_empty(), "{slug} should have edges");
        assert!(!graph.active_legend().is_empty(), "{slug} should have legend items");

        for e in &graph.edges {
          assert!(e.from < graph.nodes.len(), "{slug}: invalid edge from {}", e.from);
          assert!(e.to < graph.nodes.len(), "{slug}: invalid edge to {}", e.to);
        }

        let svg = export_standalone_svg(&graph);
        assert!(svg.starts_with("<svg "));
        assert!(svg.contains("算子色块图例 (Legend)"));
        assert!(svg.contains("连线语义图例 (Edges)"));
      }
    }

    let ds = load_builtin_preset("deepseek-v3").expect("deepseek-v3");
    let ds_g = build_blueprint_tensor_graph(&ds, ShapePreset::Prefill512, false);
    assert_eq!(ds_g.groups.len(), 2);
    // 校验 RepeatGroup 标题不再出现重复 (×3) (×3) / (×58) (×58)
    assert!(!ds_g.groups[0].badge.contains("(×3) (×3)"));
    assert!(!ds_g.groups[1].badge.contains("(×58) (×58)"));
    assert!(ds_g.nodes.iter().any(|n| n.shape.contains("1×512×7168")));
    assert!(ds_g.nodes.iter().any(|n| n.shape.contains("1×512×576")));
    assert!(ds_g.nodes.iter().any(|n| n.side_note.contains("Top-8/256")));
    // 校验框内残差与旁路边分类清晰（k_pe 64d、dispatch、× g_i、MTP 均属于 AuxBypass，残差属于 InnerSkip）
    assert!(ds_g.edges.iter().any(|e| e.kind == EdgeKind::InnerSkip && e.label == "Attn Skip"));
    assert!(ds_g.edges.iter().any(|e| e.kind == EdgeKind::InnerSkip && e.label == "MoE Skip"));
    assert!(ds_g.edges.iter().any(|e| e.kind == EdgeKind::AuxBypass && e.label == "k_pe 64d"));
    assert!(ds_g.edges.iter().any(|e| e.kind == EdgeKind::AuxBypass && e.label == "dispatch"));
    assert!(ds_g.edges.iter().any(|e| e.kind == EdgeKind::AuxBypass && e.label == "× g_i"));
    assert!(ds_g.edges.iter().any(|e| e.kind == EdgeKind::AuxBypass && e.label == "MTP"));
    // 校验折叠单个大块与全部大块行为
    let ds_col_one =
      build_blueprint_tensor_graph_with_collapse(&ds, ShapePreset::Symbolic, true, 0b10);
    assert!(ds_col_one.nodes.len() < ds_g.nodes.len());
    assert!(!ds_col_one.groups[0].collapsed);
    assert!(ds_col_one.groups[1].collapsed);
    assert!(ds_col_one.nodes.iter().any(|n| n.toggle_seg == Some(1)));
    let ds_col_all =
      build_blueprint_tensor_graph_with_collapse(&ds, ShapePreset::Symbolic, true, 0b11);
    assert!(ds_col_all.nodes.len() < ds_col_one.nodes.len());
    assert!(ds_col_all.groups.iter().all(|g| g.collapsed));

    let inc = build_inception_v3_graph(ShapePreset::Symbolic);
    assert_eq!(inc.active_legend().len(), 8);
    assert!(inc.nodes.iter().any(|n| n.shape == "299*299*3"));
    assert!(inc.nodes.iter().any(|n| n.shape == "35*35*192"));
    assert!(inc.nodes.iter().any(|n| n.shape == "35*35*256"));
    let inc_svg = export_standalone_svg(&inc);
    assert!(inc_svg.contains("输入图像"));
    assert!(inc_svg.contains("平均池化层"));
    assert!(inc_svg.contains("最大池化层"));
    assert!(inc_svg.contains("Dropout"));

    let res = build_resnet50_graph(ShapePreset::Symbolic);
    assert!(res.nodes.iter().any(|n| n.shape == "224*224*3"));
    assert!(res.nodes.iter().any(|n| n.shape == "56*56*256"));
    assert!(res.edges.iter().any(|e| e.kind == EdgeKind::MacroSkip));
  }

  #[test]
  fn declarative_graph_supports_multi_lane_skips_multi_input_and_group_collapse() {
    let raw_json = r#"{
      "title": "U-Net · Multi-Lane Skip Demo",
      "subtitle": "31.0M Params · 4-Level Symmetric Encoder-Decoder",
      "legend_style": "classic_cnn",
      "shape_mode": "image_2d",
      "presets": [
        { "label": "标准 (B×512²)", "batch": "B", "scale_num": 1, "scale_den": 1 },
        { "label": "高分 (1×1024²)", "batch": "1", "scale_num": 2, "scale_den": 1 }
      ],
      "groups": [
        {
          "badge": "Encoder · Contracting Path (4 Stages)",
          "collapsed_shape": "{B}×32×32×512",
          "collapsed_note": "点击展开 Encoder",
          "row_start": 1,
          "row_end": 4,
          "min_col_q": -2,
          "max_col_q": 2
        }
      ],
      "nodes": [
        { "id": "img", "kind": "input", "dims": [512, 512, 1], "side_note": "Image", "op_title": "Input Image", "col_q": -2, "row": 0 },
        { "id": "prompt", "kind": "embedding", "shape": "{B}×N×256", "side_note": "Prompt", "op_title": "Prompt Tokens", "col_q": 2, "row": 0 },
        { "id": "e1", "kind": "conv_or_linear", "dims": [512, 512, 64], "op_title": "Enc Stage 1", "col_q": 0, "row": 1 },
        { "id": "e2", "kind": "conv_or_linear", "dims": [256, 256, 128], "op_title": "Enc Stage 2", "col_q": 0, "row": 2 },
        { "id": "e3", "kind": "conv_or_linear", "dims": [128, 128, 256], "op_title": "Enc Stage 3", "col_q": 0, "row": 3 },
        { "id": "e4", "kind": "conv_or_linear", "dims": [64, 64, 512], "op_title": "Enc Stage 4", "col_q": 0, "row": 4 },
        { "id": "bot", "kind": "attention", "dims": [32, 32, 1024], "op_title": "Bottleneck", "col_q": 0, "row": 5 },
        { "id": "d4", "kind": "merge", "dims": [64, 64, 512], "op_title": "Dec Stage 4", "col_q": 0, "row": 6 },
        { "id": "d3", "kind": "merge", "dims": [128, 128, 256], "op_title": "Dec Stage 3", "col_q": 0, "row": 7 },
        { "id": "d2", "kind": "merge", "dims": [256, 256, 128], "op_title": "Dec Stage 2", "col_q": 0, "row": 8 },
        { "id": "d1", "kind": "merge", "dims": [512, 512, 64], "op_title": "Dec Stage 1", "col_q": 0, "row": 9 }
      ],
      "edges": [
        { "from": "img", "to": "e1" },
        { "from": "prompt", "to": "e1", "kind": "aux_bypass", "label": "Cond" },
        { "from": "e1", "to": "e2" },
        { "from": "e2", "to": "e3" },
        { "from": "e3", "to": "e4" },
        { "from": "e4", "to": "bot" },
        { "from": "bot", "to": "d4" },
        { "from": "d4", "to": "d3" },
        { "from": "d3", "to": "d2" },
        { "from": "d2", "to": "d1" },
        { "from": "e4", "to": "d4", "kind": "macro_skip", "label": "Skip 1/8", "lane": 0 },
        { "from": "e3", "to": "d3", "kind": "macro_skip", "label": "Skip 1/4", "lane": 1 },
        { "from": "e2", "to": "d2", "kind": "macro_skip", "label": "Skip 1/2", "lane": 2 },
        { "from": "e1", "to": "d1", "kind": "macro_skip", "label": "Skip 1/1", "lane": 3 }
      ]
    }"#;

    let spec: DeclarativeGraphSpec = serde_json::from_str(raw_json).expect("parse graph.json");
    spec.validate().expect("valid declarative spec");
    let wire = spec.encode_wire();

    let (g_std, p_lbls, d_mask) =
      build_declarative_graph_from_wire(&wire, "unet-demo", 0, 0).expect("build standard");
    assert_eq!(d_mask, 0);
    assert_eq!(p_lbls.len(), 2);
    assert_eq!(g_std.nodes.len(), 11);
    assert!(g_std.nodes.iter().any(|n| n.shape == "B×512×512×64"));
    assert!(g_std.nodes.iter().any(|n| n.shape == "B×N×256"));
    let layout = compute_layout(&g_std, g_std.active_legend().len(), g_std.active_edge_mask());
    assert_eq!(layout.max_macro_lane, 3);
    assert!(layout.macro_rail_x - 3 * 20 >= 16, "outermost lane must stay inside canvas");

    let svg = export_standalone_svg(&g_std);
    for lbl in ["Skip 1/8", "Skip 1/4", "Skip 1/2", "Skip 1/1"] {
      assert!(svg.contains(lbl), "SVG should contain multi-lane label {lbl}");
    }

    // 验证 2× 高分辨率预设动态缩放
    let (g_hd, _, _) =
      build_declarative_graph_from_wire(&wire, "unet-demo", 1, 0).expect("build hd");
    assert!(g_hd.nodes.iter().any(|n| n.shape == "1×1024×1024×64"));

    // 验证组块收起：4 个 Encoder 节点收敛为 1 张汇总卡且总行数紧凑收缩
    let (g_col, _, _) =
      build_declarative_graph_from_wire(&wire, "unet-demo", 0, 0b1).expect("build collapsed");
    assert_eq!(g_col.nodes.len(), 8);
    assert!(g_col.total_rows < g_std.total_rows);
    assert!(g_col.groups[0].collapsed);
    assert!(g_col.nodes.iter().any(|n| n.toggle_seg == Some(0) && n.shape == "B×32×32×512"));

    // 验证非法边拦截
    let mut bad = spec.clone();
    bad.edges.push(DeclarativeEdge {
      from: "ghost".to_string(),
      to: "e1".to_string(),
      label: String::new(),
      kind: EdgeKind::Main,
      lane: 0,
      group_idx: None,
    });
    assert!(bad.validate().is_err());
  }
}
