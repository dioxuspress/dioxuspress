//! 全网张量维流图（Tensor Flow DAG）引擎与可视化组件。
//!
//! 灵感源自经典的「网格表格 + 算子实色块 + 框内输出维度 + 右侧短参数注解 + 右上角图例」网络结构图，
//! 零第三方 JS 依赖、100% 兼容无 `'unsafe-eval'` CSP，支持：
//! 1. 从任意 [`ModelBlueprint`] 自动推导端到端多分支张量维度流（覆盖 LLM / VLM / ASR / TTS / Diffusion / SSM / Embedding）；
//! 2. 符号维度（`B×S`）与具体实数维度（`1×1` / `1×512` / `4×4096`）一键切换实时重算；
//! 3. 内置经典 CV 多分支拓扑对照（Inception-v3 / ResNet-50，完整展示 Conv / AvgPool / MaxPool / Concat / Dropout / FC / Softmax 图例）；
//! 4. 点击任意节点查看精确算子变换公式并一键跳回分层工作台对应算子子图；
//! 5. 纯 Rust 零 JS 导出独立高清 `.svg` 矢量图。

use std::collections::{HashMap, HashSet};

use dioxus::prelude::*;
use dioxus_shadcn::card_class;

use crate::arch::{
  format_count, AttentionSpec, FfnSpec, InspectorFocus, LayerSegment, ModelBlueprint,
  OutputHeadSpec,
};

/// 算子着色类别（与右上角图例一一对应）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpKind {
  /// 输入张量（输入图像 / Token IDs / 梅尔谱 / 潜空间噪声）
  Input,
  /// 词嵌入 / Patchify 切片
  Embedding,
  /// 卷积层 / 线性投影（Conv2d / q_proj / kv_proj / gate_proj / down_proj）
  ConvOrLinear,
  /// 平均池化 / 潜空间压缩 / Pooling
  AvgPool,
  /// 最大池化 / 归一化 / 激活门控（MaxPool / RMSNorm / SiLU）
  MaxPool,
  /// 注意力 / 状态空间混合器（MLA / GQA / DeltaNet / Mamba）
  Attention,
  /// MoE 路由器与稀疏/共享专家
  MoeRouter,
  /// 拼接 / 残差相加 / 门控乘积（Concat / Add / Mul）
  Merge,
  /// Softmax / 概率归一化 / L2 Norm
  Softmax,
  /// 全连接头 / LM Head / VAE Decoder / Vocoder
  FcHead,
  /// Dropout / 正则化
  Dropout,
}

/// 图例命名风格：按当前展示网络自动切换最贴切的中文术语。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LegendStyle {
  LlmOrMultimodal,
  ClassicCnn,
}

/// 张量维度代入模式。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShapePreset {
  /// 符号维度（如 `B×S×7168`、`299×299×3`）
  Symbolic,
  /// 单步自回归解码（`B=1, S=1`）
  Decode1,
  /// 典型 Prefill（`B=1, S=512`）
  Prefill512,
  /// 高吞吐批处理（`B=4, S=4096`）
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

  fn fmt_seq(self, dim: u64) -> String {
    format!("{}×{dim}", self.seq_prefix())
  }

  fn fmt_heads(self, heads: u64, head_dim: u64) -> String {
    format!("{}×{heads}×{}×{head_dim}", self.batch_str(), self.seq_str())
  }

  fn fmt_cv(self, h: u64, w: u64, c: u64) -> String {
    match self {
      Self::Symbolic => format!("{h}*{w}*{c}"),
      Self::Decode1 | Self::Prefill512 => format!("1×{h}×{w}×{c}"),
      Self::Batch4K => format!("4×{h}×{w}×{c}"),
    }
  }
}

/// 画布展示来源：当前模型或内置经典 CV 教学对照。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphSource {
  CurrentModel,
  InceptionV3,
  ResNet50,
}

impl OpKind {
  /// 实色块填充色、描边色与文字颜色（兼顾亮/暗双主题及独立导出 SVG 的高对比度可读性）。
  pub fn palette(self) -> (&'static str, &'static str, &'static str) {
    // (fill_hex, stroke_hex, text_hex)
    match self {
      Self::Input => ("#f472b6", "#db2777", "#0f172a"),
      Self::Embedding => ("#fb7185", "#e11d48", "#0f172a"),
      Self::ConvOrLinear => ("#f97316", "#c2410c", "#ffffff"),
      Self::AvgPool => ("#0ea5e9", "#0369a1", "#ffffff"),
      Self::MaxPool => ("#16a34a", "#15803d", "#ffffff"),
      Self::Attention => ("#4f46e5", "#3730a3", "#ffffff"),
      Self::MoeRouter => ("#d97706", "#b45309", "#ffffff"),
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

/// 单个张量维流节点。
#[derive(Debug, Clone, PartialEq)]
pub struct TensorNode {
  pub id: String,
  pub kind: OpKind,
  /// 方框内主文案（输出张量维度）
  pub shape: String,
  /// 方框右侧短注解（如 `1*1`、`q_a_proj`、`Top-8/256`）
  pub side_note: String,
  /// 点击节点后的详细算子标题
  pub op_title: String,
  /// 详细形状与矩阵变换公式
  pub formula: String,
  /// 参数量或显存要点
  pub params_hint: String,
  /// 逻辑列坐标（0.0 为主干中轴，负数偏左、正数偏右）
  pub col: f64,
  /// 逻辑行号（从上到下递增）
  pub row: usize,
  /// 关联的分层工作台钻取目标
  pub focus_target: Option<InspectorFocus>,
  /// 关联的层索引
  pub layer_target: Option<u64>,
}

/// 节点间有向数据流边。
#[derive(Debug, Clone, PartialEq)]
pub struct TensorEdge {
  pub from_id: String,
  pub to_id: String,
  pub label: String,
  /// `true` 表示残差直连或外侧绕行虚线边
  pub dashed: bool,
}

/// 重复块外框标记（如 `Layers 0–2 (×3)` 或 `Inception-A Block (×3)`）。
#[derive(Debug, Clone, PartialEq)]
pub struct RepeatGroup {
  pub row_start: usize,
  pub row_end: usize,
  pub min_col: f64,
  pub max_col: f64,
  pub badge: String,
  pub subtitle: String,
}

/// 完整的全网张量维流图（数据驱动，与具体渲染后端解耦）。
#[derive(Debug, Clone, PartialEq)]
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
  /// 返回当前图中实际出现过的图例项（保持稳定顺序，不展示无关空项）。
  pub fn active_legend(&self) -> Vec<(OpKind, &'static str)> {
    let present: HashSet<OpKind> = self.nodes.iter().map(|n| n.kind).collect();
    OpKind::ALL_ORDER
      .iter()
      .copied()
      .filter(|k| present.contains(k))
      .map(|k| (k, k.legend_label(self.legend_style)))
      .collect()
  }
}

struct GraphBuilder {
  nodes: Vec<TensorNode>,
  edges: Vec<TensorEdge>,
  groups: Vec<RepeatGroup>,
  seq: usize,
}

impl GraphBuilder {
  fn new() -> Self {
    Self { nodes: Vec::new(), edges: Vec::new(), groups: Vec::new(), seq: 0 }
  }

  #[allow(clippy::too_many_arguments)]
  fn add_node(
    &mut self,
    kind: OpKind,
    shape: String,
    side_note: String,
    op_title: String,
    formula: String,
    params_hint: String,
    col: f64,
    row: usize,
    focus_target: Option<InspectorFocus>,
    layer_target: Option<u64>,
  ) -> String {
    self.seq += 1;
    let id = format!("n{}", self.seq);
    self.nodes.push(TensorNode {
      id: id.clone(),
      kind,
      shape,
      side_note,
      op_title,
      formula,
      params_hint,
      col,
      row,
      focus_target,
      layer_target,
    });
    id
  }

  fn edge(&mut self, from: &str, to: &str) {
    self.edges.push(TensorEdge {
      from_id: from.to_string(),
      to_id: to.to_string(),
      label: String::new(),
      dashed: false,
    });
  }

  fn edge_labeled(&mut self, from: &str, to: &str, label: &str, dashed: bool) {
    self.edges.push(TensorEdge {
      from_id: from.to_string(),
      to_id: to.to_string(),
      label: label.to_string(),
      dashed,
    });
  }

  fn add_group(
    &mut self,
    row_start: usize,
    row_end: usize,
    min_col: f64,
    max_col: f64,
    badge: String,
    subtitle: String,
  ) {
    self.groups.push(RepeatGroup { row_start, row_end, min_col, max_col, badge, subtitle });
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

/// 从任意 [`ModelBlueprint`] 推导全网多分支张量维流图。
pub fn build_blueprint_tensor_graph(
  bp: &ModelBlueprint,
  preset: ShapePreset,
  compact: bool,
) -> TensorGraph {
  let mut g = GraphBuilder::new();
  let d = bp.hidden_size;
  let v = bp.vocab_size;
  let mut row: usize = 0;

  // 1. 输入与多模态前端（双塔并行扇出 → 红色融合节点）
  let mut cur_id = if let Some(ref diff) = bp.diffusion {
    let cond_count = diff.components.iter().filter(|c| c.role == "conditioner").count().max(1);
    let prompt_in = g.add_node(
      OpKind::Input,
      format!("{}×77", preset.batch_str()),
      "prompt".to_string(),
      "Prompt Token Input".to_string(),
      format!("Tokenized text prompt → {cond_count} Text Encoder(s)"),
      format!("Conditioner 总参: {}", format_count(bp.params.embedding)),
      -0.85,
      row,
      Some(InspectorFocus::Embedding),
      None,
    );
    let latent_in = g.add_node(
      OpKind::Input,
      truncate_label(&diff.default_latent_shape, 18),
      "z_t noise".to_string(),
      "Noisy Latent Tensor".to_string(),
      format!("{} ({}ch Latent)", diff.default_latent_shape, diff.latent_channels),
      format!("Scheduler: {}", diff.scheduler_summary),
      0.85,
      row,
      Some(InspectorFocus::Embedding),
      None,
    );
    row += 1;
    let cond_out = g.add_node(
      OpKind::Embedding,
      preset.fmt_seq(d),
      format!("×{cond_count} Enc"),
      "Text Conditioners Projection".to_string(),
      format!("CLIP / T5 / UMT5 → context_embed [{}]", preset.fmt_seq(d)),
      "单次前向编码，跨去噪步复用".to_string(),
      -0.85,
      row,
      Some(InspectorFocus::Embedding),
      None,
    );
    let patch_out = g.add_node(
      OpKind::ConvOrLinear,
      preset.fmt_seq(d),
      truncate_label(&diff.patch_summary, 11),
      "Latent Patchify / Conv Embed".to_string(),
      format!("{} → patch tokens [{}]", diff.patch_summary, preset.fmt_seq(d)),
      format!("Denoiser: {}", diff.denoiser_kind),
      0.85,
      row,
      Some(InspectorFocus::Embedding),
      None,
    );
    g.edge(&prompt_in, &cond_out);
    g.edge(&latent_in, &patch_out);
    row += 1;
    let fused = g.add_node(
      OpKind::Merge,
      preset.fmt_seq(d),
      "Joint/Cross".to_string(),
      "Condition + Latent Stream Fusion".to_string(),
      format!("MMDiT Joint Sequence / Cross-Attn Conditioning → [{}]", preset.fmt_seq(d)),
      format!("Backbone: {}L × {d}d", bp.num_hidden_layers),
      0.0,
      row,
      Some(InspectorFocus::Attention),
      Some(0),
    );
    g.edge(&cond_out, &fused);
    g.edge(&patch_out, &fused);
    fused
  } else if bp.vision.is_some() || bp.audio.is_some() {
    let mut modal_outs: Vec<String> = Vec::new();
    let has_both = bp.vision.is_some() && bp.audio.is_some();
    let vis_col = if has_both { -1.2 } else { -0.85 };
    let aud_col = if has_both { 0.0 } else { -0.85 };
    let txt_col = if has_both { 1.2 } else { 0.85 };

    if let Some(ref vis) = bp.vision {
      let img_dim = vis.image_size.unwrap_or(384);
      let patch = vis.patch_size.unwrap_or(14);
      let n_patches = (img_dim / patch.max(1)).pow(2);
      let v_in = g.add_node(
        OpKind::Input,
        preset.fmt_cv(img_dim, img_dim, 3),
        "RGB".to_string(),
        "Vision Image Input".to_string(),
        format!("RGB Image [{img_dim}×{img_dim}×3]"),
        format!("Vision Tower: {}", vis.model_type),
        vis_col,
        row,
        Some(InspectorFocus::Vision),
        None,
      );
      let v_enc = g.add_node(
        OpKind::ConvOrLinear,
        format!("{}×{n_patches}×{}", preset.batch_str(), vis.hidden_size),
        format!("p={patch}, ×{}L", vis.num_hidden_layers),
        format!("Vision Patchify + ViT ({}L)", vis.num_hidden_layers),
        format!(
          "PatchConv({patch}×{patch}) + {}L ViT({}H) → [{}×{n_patches}×{}]",
          vis.num_hidden_layers,
          vis.num_attention_heads,
          preset.batch_str(),
          vis.hidden_size
        ),
        format!("视觉塔参数: {}", format_count(vis.estimated_params)),
        vis_col,
        row + 1,
        Some(InspectorFocus::Vision),
        None,
      );
      let v_proj = g.add_node(
        OpKind::AvgPool,
        format!("{}×N_v×{d}", preset.batch_str()),
        format!("{}→{d}", vis.hidden_size),
        "Vision Spatial Merger & Projector".to_string(),
        format!("Spatial Pool/Merge + MLP Projector [{} → {d}]", vis.hidden_size),
        "对齐到主干隐藏维度".to_string(),
        vis_col,
        row + 2,
        Some(InspectorFocus::Vision),
        None,
      );
      g.edge(&v_in, &v_enc);
      g.edge(&v_enc, &v_proj);
      modal_outs.push(v_proj);
    }

    if let Some(ref aud) = bp.audio {
      let a_in = g.add_node(
        OpKind::Input,
        format!("{}×{}×T", preset.batch_str(), aud.num_mel_bins),
        format!("{}kHz", aud.sampling_rate_hz / 1000),
        "Log-Mel Spectrogram Input".to_string(),
        format!("{} Mel Bins @ {} Hz", aud.num_mel_bins, aud.sampling_rate_hz),
        format!("Audio Tower: {}", aud.model_type),
        aud_col,
        row,
        Some(InspectorFocus::Audio),
        None,
      );
      let a_enc = g.add_node(
        OpKind::ConvOrLinear,
        format!("{}×T/2×{}", preset.batch_str(), aud.hidden_size),
        format!("Conv1d, ×{}L", aud.num_layers),
        format!("Audio Conv1d Stem + Encoder ({}L)", aud.num_layers),
        format!(
          "Conv1d(s=2) + {}L Audio Transformer({}H) → [{}×T/2×{}]",
          aud.num_layers,
          aud.num_heads,
          preset.batch_str(),
          aud.hidden_size
        ),
        format!("音频塔参数: {}", format_count(aud.estimated_params)),
        aud_col,
        row + 1,
        Some(InspectorFocus::Audio),
        None,
      );
      let a_proj = g.add_node(
        OpKind::AvgPool,
        format!("{}×T_a×{d}", preset.batch_str()),
        format!("{}→{d}", aud.hidden_size),
        "Audio Temporal Pool & Projector".to_string(),
        format!("Adapter Projection [{} → {d}]", aud.hidden_size),
        "映射到语言主干隐空间".to_string(),
        aud_col,
        row + 2,
        Some(InspectorFocus::Audio),
        None,
      );
      g.edge(&a_in, &a_enc);
      g.edge(&a_enc, &a_proj);
      modal_outs.push(a_proj);
    }

    let t_in = g.add_node(
      OpKind::Input,
      preset.seq_prefix().to_string(),
      "int64".to_string(),
      "Text input_ids".to_string(),
      format!("Token IDs [{}], Vocab = {v}", preset.seq_prefix()),
      format!("词表规模: {}", format_count(v)),
      txt_col,
      row,
      Some(InspectorFocus::Embedding),
      None,
    );
    let t_emb = g.add_node(
      OpKind::Embedding,
      preset.fmt_seq(d),
      "embed".to_string(),
      "Token Embedding Lookup".to_string(),
      format!("embed_tokens [{v} × {d}] → [{}]", preset.fmt_seq(d)),
      format!("Embedding 参数: {}", format_count(bp.params.embedding)),
      txt_col,
      row + 2,
      Some(InspectorFocus::Embedding),
      None,
    );
    g.edge(&t_in, &t_emb);

    row += 3;
    let fused = g.add_node(
      OpKind::Merge,
      preset.fmt_seq(d),
      "Concat".to_string(),
      "Multimodal Token Sequence Concat".to_string(),
      format!("Concat(Modal Tokens, Text Embeds) → [{}]", preset.fmt_seq(d)),
      "多模态序列统一送入主干层栈".to_string(),
      0.0,
      row,
      Some(InspectorFocus::Embedding),
      None,
    );
    for m_id in &modal_outs {
      g.edge(m_id, &fused);
    }
    g.edge(&t_emb, &fused);
    fused
  } else {
    let t_in = g.add_node(
      OpKind::Input,
      preset.seq_prefix().to_string(),
      "input_ids".to_string(),
      "Token IDs Input".to_string(),
      format!("int64 [{}], Vocab = {v}", preset.seq_prefix()),
      format!("上下文上限: {}K tokens", bp.max_position_embeddings / 1024),
      0.0,
      row,
      Some(InspectorFocus::Embedding),
      None,
    );
    row += 1;
    let t_emb = g.add_node(
      OpKind::Embedding,
      preset.fmt_seq(d),
      format!("{v}→{d}"),
      "Token Embedding (embed_tokens)".to_string(),
      format!("Lookup [{v}, {d}] → [{}]", preset.fmt_seq(d)),
      format!("Embedding 参数: {}", format_count(bp.params.embedding)),
      0.0,
      row,
      Some(InspectorFocus::Embedding),
      None,
    );
    g.edge(&t_in, &t_emb);
    t_emb
  };

  // 2. 逐个异构层段展开多分支张量流（每个异构段带外框与 ×N 重复标记）
  for seg in &bp.segments {
    row += 1;
    let seg_start_row = row;
    let (next_id, next_row, min_col, max_col) =
      append_segment_subgraph(&mut g, bp, seg, &cur_id, row, preset, compact);
    let badge = format!("{} (×{}) · {}", seg.range_label(), seg.layer_count(), seg.title);
    let subtitle = format!("{}  |  {}", seg.attention.short_label(), seg.ffn.short_label());
    g.add_group(seg_start_row, next_row, min_col, max_col, badge, subtitle);
    cur_id = next_id;
    row = next_row;
  }

  // 3. 末端输出阶段（Final Norm + Output Head + 可选并行 MTP 分支）
  row += 1;
  if !compact {
    let norm_id = g.add_node(
      OpKind::MaxPool,
      preset.fmt_seq(d),
      "FinalNorm".to_string(),
      "Final RMSNorm / LayerNorm".to_string(),
      format!("RMSNorm([{}], eps={:.0e})", preset.fmt_seq(d), bp.rms_norm_eps.unwrap_or(1e-6)),
      format!("参数量: {d}"),
      0.0,
      row,
      Some(InspectorFocus::NormAndResidual),
      None,
    );
    g.edge(&cur_id, &norm_id);
    cur_id = norm_id;
    row += 1;
  }

  let has_mtp = bp.num_nextn_predict_layers > 0;
  let head_col = if has_mtp { -0.65 } else { 0.0 };

  match &bp.output_head {
    OutputHeadSpec::CausalLmHead => {
      let tied_tag = if bp.tie_word_embeddings {
        "Tied Embed".to_string()
      } else {
        format_count(bp.params.lm_head)
      };
      let lm_id = g.add_node(
        OpKind::FcHead,
        preset.fmt_seq(v),
        format!("{d}→{v}"),
        "Output Projection (lm_head)".to_string(),
        format!("Linear [{d} → {v}] → logits [{}]", preset.fmt_seq(v)),
        format!("LM Head 参数: {tied_tag}"),
        head_col,
        row,
        Some(InspectorFocus::LmHead),
        None,
      );
      g.edge(&cur_id, &lm_id);

      let sm_id = g.add_node(
        OpKind::Softmax,
        preset.fmt_seq(v),
        "probs".to_string(),
        "Vocab Softmax / Sampling".to_string(),
        format!("Softmax(logits / T) → next-token probs [{}]", preset.fmt_seq(v)),
        "Top-p / Top-k 采样输出下一 Token".to_string(),
        head_col,
        row + 1,
        Some(InspectorFocus::LmHead),
        None,
      );
      g.edge(&lm_id, &sm_id);

      if has_mtp {
        let mtp_id = g.add_node(
          OpKind::MoeRouter,
          preset.fmt_seq(d),
          format!("MTP ×{}", bp.num_nextn_predict_layers),
          format!("Multi-Token Prediction (×{})", bp.num_nextn_predict_layers),
          format!("eh_proj [2×{d} → {d}] + Speculative Block"),
          format!("MTP 额外参数: +{}", format_count(bp.params.mtp_head)),
          0.95,
          row,
          Some(InspectorFocus::MtpHead),
          None,
        );
        let mtp_out = g.add_node(
          OpKind::Softmax,
          preset.fmt_seq(v),
          "t+2..k".to_string(),
          "MTP Shared LM Head + Speculative Draft".to_string(),
          format!("Shared lm_head [{d} → {v}] → 投机解码草稿 Token"),
          "推理期加速多 Token 并行验证".to_string(),
          0.95,
          row + 1,
          Some(InspectorFocus::MtpHead),
          None,
        );
        g.edge_labeled(&cur_id, &mtp_id, "MTP", true);
        g.edge(&mtp_id, &mtp_out);
      }
    }
    OutputHeadSpec::EmbeddingPool { pooling_mode, output_dim, normalize } => {
      let pool_id = g.add_node(
        OpKind::AvgPool,
        format!("{}×{output_dim}", preset.batch_str()),
        truncate_label(pooling_mode, 10),
        format!("{pooling_mode} Sequence Pooling"),
        format!("{pooling_mode}(h_final) → [{}×{output_dim}]", preset.batch_str()),
        "单次前向编码，0 B KV Cache".to_string(),
        0.0,
        row,
        Some(InspectorFocus::LmHead),
        None,
      );
      g.edge(&cur_id, &pool_id);
      if *normalize {
        let norm_id = g.add_node(
          OpKind::Softmax,
          format!("{}×{output_dim}", preset.batch_str()),
          "L2 Norm".to_string(),
          "L2 Vector Normalization".to_string(),
          format!("v / ||v||_2 → Unit Embedding [{}×{output_dim}]", preset.batch_str()),
          "直接用于余弦相似度检索".to_string(),
          0.0,
          row + 1,
          Some(InspectorFocus::LmHead),
          None,
        );
        g.edge(&pool_id, &norm_id);
      }
    }
    OutputHeadSpec::SequenceScorer { num_labels } => {
      let cls_id = g.add_node(
        OpKind::FcHead,
        format!("{}×{num_labels}", preset.batch_str()),
        format!("{d}→{num_labels}"),
        "Cross-Encoder Classifier / Rerank Head".to_string(),
        format!("Linear [{d} → {num_labels}] → scores [{}×{num_labels}]", preset.batch_str()),
        format!("Head 参数: {}", format_count(bp.params.lm_head)),
        0.0,
        row,
        Some(InspectorFocus::LmHead),
        None,
      );
      let sm_id = g.add_node(
        OpKind::Softmax,
        format!("{}×{num_labels}", preset.batch_str()),
        "score".to_string(),
        "Relevance Probability Normalization".to_string(),
        format!("Sigmoid / Softmax → [{}×{num_labels}]", preset.batch_str()),
        "用于 RAG 重排序打分".to_string(),
        0.0,
        row + 1,
        Some(InspectorFocus::LmHead),
        None,
      );
      g.edge(&cur_id, &cls_id);
      g.edge(&cls_id, &sm_id);
    }
    OutputHeadSpec::SpeechSynthesizer {
      talker_summary,
      codec_summary,
      num_codebooks,
      sampling_rate_hz,
      estimated_params,
    } => {
      let tk_id = g.add_node(
        OpKind::MoeRouter,
        format!("{}×{num_codebooks}×T", preset.batch_str()),
        format!("{num_codebooks} RVQ"),
        talker_summary.clone().unwrap_or_else(|| "Acoustic Codebook Predictor".to_string()),
        format!("Predict {num_codebooks} RVQ Acoustic Tokens"),
        format!("声学后端总参: {}", format_count(*estimated_params)),
        0.0,
        row,
        Some(InspectorFocus::LmHead),
        None,
      );
      let voc_id = g.add_node(
        OpKind::FcHead,
        format!("{}×T_wav", preset.batch_str()),
        format!("{}kHz", sampling_rate_hz / 1000),
        format!("Neural Vocoder ({codec_summary})"),
        format!("{codec_summary} → PCM Waveform @ {} Hz", sampling_rate_hz),
        "端到端还原高保真音频波形".to_string(),
        0.0,
        row + 1,
        Some(InspectorFocus::LmHead),
        None,
      );
      g.edge(&cur_id, &tk_id);
      g.edge(&tk_id, &voc_id);
    }
    OutputHeadSpec::LatentDecoder {
      vae_class,
      latent_channels,
      spatial_compression,
      temporal_compression,
      is_video,
      estimated_params,
    } => {
      let unp_id = g.add_node(
        OpKind::AvgPool,
        format!("{}×{latent_channels}×h×w", preset.batch_str()),
        "Unpatch".to_string(),
        "Latent Unpatchify".to_string(),
        format!("Unpatchify([{}]) → {latent_channels}ch Latent Grid", preset.fmt_seq(d)),
        format!(
          "压缩率: ×{spatial_compression} 空间{}",
          if *is_video && *temporal_compression > 1 {
            format!(" · ×{temporal_compression} 时间")
          } else {
            String::new()
          }
        ),
        0.0,
        row,
        Some(InspectorFocus::LmHead),
        None,
      );
      let out_shape = if *is_video {
        format!("{}×F×3×H×W", preset.batch_str())
      } else {
        format!("{}×3×H×W", preset.batch_str())
      };
      let vae_id = g.add_node(
        OpKind::FcHead,
        out_shape.clone(),
        truncate_label(vae_class, 11),
        format!("VAE Decoder ({vae_class})"),
        format!("{vae_class} Decode → {out_shape}"),
        format!("VAE 解码器参数: {}", format_count(*estimated_params)),
        0.0,
        row + 1,
        Some(InspectorFocus::LmHead),
        None,
      );
      g.edge(&cur_id, &unp_id);
      g.edge(&unp_id, &vae_id);
    }
  }

  let slug = bp.repo_id.replace('/', "-").to_lowercase();
  let subtitle = format!(
    "{} · {}L × {}d · 总参 {} (单 Token 激活 {}) · {}",
    bp.family.badge_label(),
    bp.num_hidden_layers,
    bp.hidden_size,
    format_count(bp.params.total_params),
    format_count(bp.params.active_params),
    preset.label()
  );
  g.finish(slug, format!("{} · 全网张量维流图", bp.repo_id), subtitle, LegendStyle::LlmOrMultimodal)
}

/// 为单个异构层段构建「Attention/SSM 多分支 + FFN/MoE 多分支」张量流子图。
fn append_segment_subgraph(
  g: &mut GraphBuilder,
  bp: &ModelBlueprint,
  seg: &LayerSegment,
  prev_id: &str,
  start_row: usize,
  preset: ShapePreset,
  compact: bool,
) -> (String, usize, f64, f64) {
  let d = bp.hidden_size;
  let layer = seg.start_layer;
  let mut row = start_row;
  let mut min_col: f64 = -0.6;
  let mut max_col: f64 = 0.6;

  // ── 1) Attention / SSM 子层 ──
  let attn_in_id = if compact {
    prev_id.to_string()
  } else {
    let norm_id = g.add_node(
      OpKind::MaxPool,
      preset.fmt_seq(d),
      "Pre-Norm".to_string(),
      format!("input_layernorm (Layer #{layer})"),
      format!("RMSNorm([{}]) → [{}]", preset.fmt_seq(d), preset.fmt_seq(d)),
      format!("{} · 每层 Norm 参数 {}", seg.range_label(), format_count(d)),
      0.0,
      row,
      Some(InspectorFocus::NormAndResidual),
      Some(layer),
    );
    g.edge(prev_id, &norm_id);
    row += 1;
    norm_id
  };

  let attn_params = seg.attention.params_per_layer(d);
  let attn_add_id = match &seg.attention {
    AttentionSpec::DeepSeekMla {
      num_heads,
      q_lora_rank,
      kv_lora_rank,
      qk_nope_head_dim,
      qk_rope_head_dim,
      v_head_dim,
    } => {
      min_col = min_col.min(-0.85);
      max_col = max_col.max(0.85);
      let qk_dim = qk_nope_head_dim + qk_rope_head_dim;
      let kv_a_dim = kv_lora_rank + qk_rope_head_dim;
      let kv_b_head = qk_nope_head_dim + v_head_dim;
      let o_in = num_heads * v_head_dim;

      // 左路 Query 低秩压缩与还原，右路 KV 潜空间压缩（核心 KV Cache 576d！）
      let q_leaf = if let Some(qr) = q_lora_rank {
        let qa = g.add_node(
          OpKind::ConvOrLinear,
          preset.fmt_seq(*qr),
          "q_a_proj".to_string(),
          "Query Low-Rank Down Projection (q_a_proj)".to_string(),
          format!("[{d} → {qr}] + RMSNorm → Query Latent c_Q"),
          format!("压缩比 {:.1}×", (d as f64) / (*qr as f64)),
          -0.75,
          row,
          Some(InspectorFocus::Attention),
          Some(layer),
        );
        let qb = g.add_node(
          OpKind::ConvOrLinear,
          preset.fmt_heads(*num_heads, qk_dim),
          format!("q_b ({num_heads}H)"),
          "Query Up Projection + Decoupled RoPE (q_b_proj)".to_string(),
          format!(
            "[{qr} → {}] → q_nope({qk_nope_head_dim}) ⊕ RoPE(q_pe({qk_rope_head_dim}))",
            num_heads * qk_dim
          ),
          format!("{num_heads} Heads × {qk_dim}d"),
          -0.75,
          row + 1,
          Some(InspectorFocus::Attention),
          Some(layer),
        );
        g.edge(&attn_in_id, &qa);
        g.edge(&qa, &qb);
        qb
      } else {
        let q_direct = g.add_node(
          OpKind::ConvOrLinear,
          preset.fmt_heads(*num_heads, qk_dim),
          format!("q_proj ({num_heads}H)"),
          "Direct Query Projection".to_string(),
          format!("[{d} → {}] ({num_heads}H × {qk_dim}d)", num_heads * qk_dim),
          "无 Query 低秩压缩".to_string(),
          -0.75,
          row,
          Some(InspectorFocus::Attention),
          Some(layer),
        );
        g.edge(&attn_in_id, &q_direct);
        q_direct
      };

      let kva = g.add_node(
        OpKind::AvgPool,
        preset.fmt_seq(kv_a_dim),
        format!("KV {kv_a_dim}d"),
        "Latent KV Compression (kv_a_proj_with_mqa)".to_string(),
        format!("[{d} → {kv_a_dim}] → c_KV({kv_lora_rank}) + Shared k_pe({qk_rope_head_dim})"),
        format!(
          "推理期仅缓存 {kv_a_dim} elems/tok ({:.1}× vs MHA)",
          seg.attention.kv_compression_ratio()
        ),
        0.75,
        row,
        Some(InspectorFocus::Attention),
        Some(layer),
      );
      let kvb = g.add_node(
        OpKind::ConvOrLinear,
        preset.fmt_heads(*num_heads, kv_b_head),
        "kv_b_proj".to_string(),
        "Latent KV Up Projection (kv_b_proj, Matrix Absorbable)".to_string(),
        format!(
          "[{kv_lora_rank} → {}] → k_nope({qk_nope_head_dim}) + v({v_head_dim})",
          num_heads * kv_b_head
        ),
        "推理阶段可与 W_Q / W_O 做矩阵吸收免显式展开".to_string(),
        0.75,
        row + 1,
        Some(InspectorFocus::Attention),
        Some(layer),
      );
      g.edge(&attn_in_id, &kva);
      g.edge(&kva, &kvb);
      row += 2;

      let mla = g.add_node(
        OpKind::Attention,
        preset.fmt_seq(o_in),
        format!("MLA {num_heads}H"),
        "Multi-Head Latent Attention (MLA)".to_string(),
        format!(
          "Softmax((q_nope·k_nopeᵀ + q_pe·k_peᵀ)/√{qk_dim}) · V → [{}]",
          preset.fmt_seq(o_in)
        ),
        format!("本层 Attention 总参: {}", format_count(attn_params)),
        0.0,
        row,
        Some(InspectorFocus::Attention),
        Some(layer),
      );
      g.edge(&q_leaf, &mla);
      g.edge(&kvb, &mla);
      row += 1;

      let o_proj = g.add_node(
        OpKind::ConvOrLinear,
        preset.fmt_seq(d),
        "o_proj".to_string(),
        "Attention Output Projection (o_proj)".to_string(),
        format!("Linear [{o_in} → {d}] → [{}]", preset.fmt_seq(d)),
        format!("投影矩阵参数: {}", format_count(o_in * d)),
        0.0,
        row,
        Some(InspectorFocus::Attention),
        Some(layer),
      );
      g.edge(&mla, &o_proj);
      row += 1;

      let add_id = g.add_node(
        OpKind::Merge,
        preset.fmt_seq(d),
        "Add (+)".to_string(),
        "Residual Skip Add #1 (Attention)".to_string(),
        format!("h_mid = h_in + o_proj(MLA(Norm(h_in))) → [{}]", preset.fmt_seq(d)),
        "保持梯度直通的恒等残差连接".to_string(),
        0.0,
        row,
        Some(InspectorFocus::NormAndResidual),
        Some(layer),
      );
      g.edge(&o_proj, &add_id);
      g.edge_labeled(prev_id, &add_id, "Skip", true);
      add_id
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
      min_col = min_col.min(-1.1);
      max_col = max_col.max(1.1);
      let q_out = num_q_heads * head_dim;
      let kv_out = num_kv_heads * head_dim;
      let q_note =
        if *qk_norm { format!("q+Norm {num_q_heads}H") } else { format!("q_proj {num_q_heads}H") };
      let k_note = if *qk_norm {
        format!("k+Norm {num_kv_heads}H")
      } else {
        format!("k_proj {num_kv_heads}H")
      };

      let q_id = g.add_node(
        OpKind::ConvOrLinear,
        preset.fmt_heads(*num_q_heads, *head_dim),
        q_note,
        "Query Projection (q_proj + RoPE)".to_string(),
        format!(
          "Linear [{d} → {q_out}] → reshape [{num_q_heads}Q × {head_dim}d]{}",
          if *qk_norm { " + QK-Norm" } else { "" }
        ),
        format!("参数量: {}", format_count(d * q_out)),
        -1.0,
        row,
        Some(InspectorFocus::Attention),
        Some(layer),
      );
      let k_id = g.add_node(
        OpKind::AvgPool,
        preset.fmt_heads(*num_kv_heads, *head_dim),
        k_note,
        "Key Projection (k_proj → KV Cache)".to_string(),
        format!("Linear [{d} → {kv_out}] → reshape [{num_kv_heads}KV × {head_dim}d]"),
        format!("KV 压缩比: {:.1}× vs MHA", seg.attention.kv_compression_ratio()),
        0.0,
        row,
        Some(InspectorFocus::Attention),
        Some(layer),
      );
      let v_id = g.add_node(
        OpKind::AvgPool,
        preset.fmt_heads(*num_kv_heads, *head_dim),
        format!("v_proj {num_kv_heads}H"),
        "Value Projection (v_proj → KV Cache)".to_string(),
        format!("Linear [{d} → {kv_out}] → reshape [{num_kv_heads}KV × {head_dim}d]"),
        format!("与 K 共同构成 KV Cache ({} elems/tok)", seg.attention.cached_elements_per_token()),
        1.0,
        row,
        Some(InspectorFocus::Attention),
        Some(layer),
      );
      g.edge(&attn_in_id, &q_id);
      g.edge(&attn_in_id, &k_id);
      g.edge(&attn_in_id, &v_id);
      row += 1;

      let attn_tag = match (sliding_window, is_causal, has_cross_attn) {
        (Some(w), _, _) => format!("SWA({w})"),
        (None, false, _) => "Bi-Attn".to_string(),
        (None, true, true) => "Self+Cross".to_string(),
        _ => format!("{num_q_heads}Q:{num_kv_heads}KV"),
      };
      let sdpa = g.add_node(
        OpKind::Attention,
        preset.fmt_seq(q_out),
        attn_tag,
        "Scaled Dot-Product Attention (FlashAttention / GQA)".to_string(),
        format!("Softmax(Q·Kᵀ/√{head_dim})·V → concat heads [{}]", preset.fmt_seq(q_out)),
        format!("本层 Attention 总参: {}", format_count(attn_params)),
        0.0,
        row,
        Some(InspectorFocus::Attention),
        Some(layer),
      );
      g.edge(&q_id, &sdpa);
      g.edge(&k_id, &sdpa);
      g.edge(&v_id, &sdpa);
      row += 1;

      let o_proj = g.add_node(
        OpKind::ConvOrLinear,
        preset.fmt_seq(d),
        "o_proj".to_string(),
        "Attention Output Projection (o_proj)".to_string(),
        format!("Linear [{q_out} → {d}] → [{}]", preset.fmt_seq(d)),
        format!("参数量: {}", format_count(q_out * d)),
        0.0,
        row,
        Some(InspectorFocus::Attention),
        Some(layer),
      );
      g.edge(&sdpa, &o_proj);
      row += 1;

      let add_id = g.add_node(
        OpKind::Merge,
        preset.fmt_seq(d),
        "Add (+)".to_string(),
        "Residual Skip Add #1 (Attention)".to_string(),
        format!("h_mid = h_in + o_proj(Attn(Norm(h_in))) → [{}]", preset.fmt_seq(d)),
        "保持恒等映射的残差跳连".to_string(),
        0.0,
        row,
        Some(InspectorFocus::NormAndResidual),
        Some(layer),
      );
      g.edge(&o_proj, &add_id);
      g.edge_labeled(prev_id, &add_id, "Skip", true);
      add_id
    }
    AttentionSpec::LinearRecurrent {
      kind,
      num_heads,
      head_dim,
      state_dim,
      conv_kernel,
      expand_factor,
    } => {
      min_col = min_col.min(-0.85);
      max_col = max_col.max(0.85);
      let d_inner = (num_heads * head_dim).max((*expand_factor).max(1) * d);

      let conv_id = g.add_node(
        OpKind::ConvOrLinear,
        preset.fmt_seq(d_inner),
        format!("Conv1d k={conv_kernel}"),
        format!("Linear Input Proj + Causal Depthwise Conv1d (k={conv_kernel})"),
        format!("[{d} → {d_inner}] + Conv1d(k={conv_kernel}) → [{}]", preset.fmt_seq(d_inner)),
        "短局部卷积缓存，O(1) 常数空间".to_string(),
        -0.75,
        row,
        Some(InspectorFocus::Attention),
        Some(layer),
      );
      let ssm_id = g.add_node(
        OpKind::Attention,
        preset.fmt_seq(d_inner),
        format!("state={state_dim}d"),
        format!("{kind} Recurrent State Update"),
        format!(
          "S_t = α_t·S_{{t-1}} + k_t⊗v_t ({num_heads}H×{head_dim}d×{state_dim}d) → [{}]",
          preset.fmt_seq(d_inner)
        ),
        format!(
          "固定状态缓存: {} elems/seq (0 B KV 增长)",
          seg.attention.fixed_state_elements_per_seq(d)
        ),
        -0.75,
        row + 1,
        Some(InspectorFocus::Attention),
        Some(layer),
      );
      let gate_id = g.add_node(
        OpKind::MaxPool,
        preset.fmt_seq(d_inner),
        "SiLU Gate".to_string(),
        "Parallel Multiplicative Output Gate".to_string(),
        format!("SiLU(z_proj([{d} → {d_inner}])) → [{}]", preset.fmt_seq(d_inner)),
        "逐元素调制 SSM 输出通道".to_string(),
        0.75,
        row,
        Some(InspectorFocus::Attention),
        Some(layer),
      );
      g.edge(&attn_in_id, &conv_id);
      g.edge(&conv_id, &ssm_id);
      g.edge(&attn_in_id, &gate_id);
      row += 2;

      let mul_id = g.add_node(
        OpKind::Merge,
        preset.fmt_seq(d_inner),
        "Gate ⊙".to_string(),
        "Gated Hadamard Product".to_string(),
        format!("y = SSM(x) ⊙ SiLU(z) → [{}]", preset.fmt_seq(d_inner)),
        "门控特征选择".to_string(),
        0.0,
        row,
        Some(InspectorFocus::Attention),
        Some(layer),
      );
      g.edge(&ssm_id, &mul_id);
      g.edge(&gate_id, &mul_id);
      row += 1;

      let out_proj = g.add_node(
        OpKind::ConvOrLinear,
        preset.fmt_seq(d),
        "out_proj".to_string(),
        "SSM Output Projection (out_proj)".to_string(),
        format!("Linear [{d_inner} → {d}] → [{}]", preset.fmt_seq(d)),
        format!("本层 Mixer 总参: {}", format_count(attn_params)),
        0.0,
        row,
        Some(InspectorFocus::Attention),
        Some(layer),
      );
      g.edge(&mul_id, &out_proj);
      row += 1;

      let add_id = g.add_node(
        OpKind::Merge,
        preset.fmt_seq(d),
        "Add (+)".to_string(),
        "Residual Skip Add #1 (SSM Mixer)".to_string(),
        format!("h_mid = h_in + out_proj(y) → [{}]", preset.fmt_seq(d)),
        "线性复杂度残差块输出".to_string(),
        0.0,
        row,
        Some(InspectorFocus::NormAndResidual),
        Some(layer),
      );
      g.edge(&out_proj, &add_id);
      g.edge_labeled(prev_id, &add_id, "Skip", true);
      add_id
    }
  };

  // ── 2) FFN / Sparse MoE 子层 ──
  let (ffn_tot, ffn_act) = seg.ffn.params_per_layer(d);
  let skip_ffn = matches!(seg.ffn, FfnSpec::DenseGated { intermediate_size: 0, .. });
  if skip_ffn {
    return (attn_add_id, row, min_col, max_col);
  }

  row += 1;
  let ffn_in_id = if compact {
    attn_add_id.clone()
  } else {
    let norm2 = g.add_node(
      OpKind::MaxPool,
      preset.fmt_seq(d),
      "Post-Norm".to_string(),
      format!("post_attention_layernorm (Layer #{layer})"),
      format!("RMSNorm([{}]) → [{}]", preset.fmt_seq(d), preset.fmt_seq(d)),
      format!("{} FFN 入口归一化", seg.range_label()),
      0.0,
      row,
      Some(InspectorFocus::NormAndResidual),
      Some(layer),
    );
    g.edge(&attn_add_id, &norm2);
    row += 1;
    norm2
  };

  let ffn_add_id = match &seg.ffn {
    FfnSpec::DenseGated { intermediate_size, act_fn, is_gated, .. } => {
      if *is_gated {
        min_col = min_col.min(-0.85);
        max_col = max_col.max(0.85);
        let gate_id = g.add_node(
          OpKind::ConvOrLinear,
          preset.fmt_seq(*intermediate_size),
          format!("gate+{act_fn}"),
          format!("Gate Projection + {act_fn} (gate_proj)"),
          format!(
            "{act_fn}(Linear([{d} → {intermediate_size}])) → [{}]",
            preset.fmt_seq(*intermediate_size)
          ),
          format!("参数量: {}", format_count(d * intermediate_size)),
          -0.75,
          row,
          Some(InspectorFocus::FfnOrMoe),
          Some(layer),
        );
        let up_id = g.add_node(
          OpKind::ConvOrLinear,
          preset.fmt_seq(*intermediate_size),
          "up_proj".to_string(),
          "Up Projection (up_proj)".to_string(),
          format!("Linear [{d} → {intermediate_size}] → [{}]", preset.fmt_seq(*intermediate_size)),
          format!("参数量: {}", format_count(d * intermediate_size)),
          0.75,
          row,
          Some(InspectorFocus::FfnOrMoe),
          Some(layer),
        );
        g.edge(&ffn_in_id, &gate_id);
        g.edge(&ffn_in_id, &up_id);
        row += 1;

        let mul_id = g.add_node(
          OpKind::Merge,
          preset.fmt_seq(*intermediate_size),
          "Mul ⊙".to_string(),
          "SwiGLU / GeGLU Elementwise Multiply".to_string(),
          format!(
            "h_ffn = {act_fn}(gate_proj(x)) ⊙ up_proj(x) → [{}]",
            preset.fmt_seq(*intermediate_size)
          ),
          "双支路门控哈达玛乘积".to_string(),
          0.0,
          row,
          Some(InspectorFocus::FfnOrMoe),
          Some(layer),
        );
        g.edge(&gate_id, &mul_id);
        g.edge(&up_id, &mul_id);
        row += 1;

        let down_id = g.add_node(
          OpKind::ConvOrLinear,
          preset.fmt_seq(d),
          "down_proj".to_string(),
          "Down Projection (down_proj)".to_string(),
          format!("Linear [{intermediate_size} → {d}] → [{}]", preset.fmt_seq(d)),
          format!("本层 Dense FFN 总参: {} (100% 激活)", format_count(ffn_tot)),
          0.0,
          row,
          Some(InspectorFocus::FfnOrMoe),
          Some(layer),
        );
        g.edge(&mul_id, &down_id);
        row += 1;

        let add2 = g.add_node(
          OpKind::Merge,
          preset.fmt_seq(d),
          "Add (+)".to_string(),
          "Residual Skip Add #2 (FFN Output)".to_string(),
          format!("h_out = h_mid + down_proj(h_ffn) → [{}]", preset.fmt_seq(d)),
          format!("完成 {} 单层前向传播", seg.range_label()),
          0.0,
          row,
          Some(InspectorFocus::NormAndResidual),
          Some(layer),
        );
        g.edge(&down_id, &add2);
        g.edge_labeled(&attn_add_id, &add2, "Skip", true);
        add2
      } else {
        let fc1 = g.add_node(
          OpKind::ConvOrLinear,
          preset.fmt_seq(*intermediate_size),
          format!("fc1+{act_fn}"),
          format!("MLP Up Projection + {act_fn}"),
          format!("{act_fn}(Linear([{d} → {intermediate_size}]))"),
          format!("参数量: {}", format_count(d * intermediate_size)),
          0.0,
          row,
          Some(InspectorFocus::FfnOrMoe),
          Some(layer),
        );
        g.edge(&ffn_in_id, &fc1);
        row += 1;
        let fc2 = g.add_node(
          OpKind::ConvOrLinear,
          preset.fmt_seq(d),
          "fc2".to_string(),
          "MLP Down Projection".to_string(),
          format!("Linear [{intermediate_size} → {d}] → [{}]", preset.fmt_seq(d)),
          format!("本层 MLP 总参: {}", format_count(ffn_tot)),
          0.0,
          row,
          Some(InspectorFocus::FfnOrMoe),
          Some(layer),
        );
        g.edge(&fc1, &fc2);
        row += 1;
        let add2 = g.add_node(
          OpKind::Merge,
          preset.fmt_seq(d),
          "Add (+)".to_string(),
          "Residual Skip Add #2 (MLP Output)".to_string(),
          format!("h_out = h_mid + fc2(fc1(Norm(h_mid))) → [{}]", preset.fmt_seq(d)),
          format!("完成 {} 单层前向传播", seg.range_label()),
          0.0,
          row,
          Some(InspectorFocus::NormAndResidual),
          Some(layer),
        );
        g.edge(&fc2, &add2);
        g.edge_labeled(&attn_add_id, &add2, "Skip", true);
        add2
      }
    }
    FfnSpec::SparseMoe {
      num_routed_experts,
      top_k,
      moe_intermediate_size,
      num_shared_experts,
      shared_intermediate_size,
      scoring_func,
      act_fn,
      ..
    } => {
      let has_shared = *num_shared_experts > 0 && *shared_intermediate_size > 0;
      if has_shared {
        min_col = min_col.min(-1.15);
        max_col = max_col.max(1.15);
      } else {
        min_col = min_col.min(-0.85);
        max_col = max_col.max(0.85);
      }

      let shared_col = -1.05;
      let router_col = if has_shared { 0.0 } else { -0.75 };
      let expert_col = if has_shared { 1.05 } else { 0.75 };

      let shared_leaf = if has_shared {
        let sh_up = g.add_node(
          OpKind::MoeRouter,
          preset.fmt_seq(*shared_intermediate_size),
          format!("Shared ×{num_shared_experts}"),
          format!("Always-On Shared Expert (×{num_shared_experts})"),
          format!("SwiGLU({act_fn}) [{d} → {shared_intermediate_size}] · 100% Token 直通"),
          "捕获跨领域通用知识，避免路由冗余".to_string(),
          shared_col,
          row,
          Some(InspectorFocus::FfnOrMoe),
          Some(layer),
        );
        let sh_down = g.add_node(
          OpKind::ConvOrLinear,
          preset.fmt_seq(d),
          "sh_down".to_string(),
          "Shared Expert Down Projection".to_string(),
          format!("Linear [{shared_intermediate_size} → {d}] → [{}]", preset.fmt_seq(d)),
          "与路由专家结果线性叠加".to_string(),
          shared_col,
          row + 1,
          Some(InspectorFocus::FfnOrMoe),
          Some(layer),
        );
        g.edge(&ffn_in_id, &sh_up);
        g.edge(&sh_up, &sh_down);
        Some(sh_down)
      } else {
        None
      };

      let router_id = g.add_node(
        OpKind::MoeRouter,
        preset.fmt_seq(*num_routed_experts),
        format!("gate {d}→{num_routed_experts}"),
        "MoE Router Gate Projection".to_string(),
        format!("Linear [{d} → {num_routed_experts}] 产生每个专家的路由打分"),
        format!("Scoring: {scoring_func} · 总专家 {num_routed_experts}"),
        router_col,
        row,
        Some(InspectorFocus::FfnOrMoe),
        Some(layer),
      );
      let topk_id = g.add_node(
        OpKind::Softmax,
        preset.fmt_seq(*top_k),
        format!("Top-{top_k}/{num_routed_experts}"),
        format!("Router {scoring_func} & Top-{top_k} Dispatch"),
        format!("{scoring_func}(logits) → 仅选中 Top-{top_k} 活跃专家权重 g_i"),
        format!(
          "稀疏激活率: {:.1}%",
          (*top_k as f64) * 100.0 / (*num_routed_experts).max(1) as f64
        ),
        router_col,
        row + 1,
        Some(InspectorFocus::FfnOrMoe),
        Some(layer),
      );
      g.edge(&ffn_in_id, &router_id);
      g.edge(&router_id, &topk_id);

      let exp_up = g.add_node(
        OpKind::MoeRouter,
        format!("{top_k}×{}", preset.fmt_seq(*moe_intermediate_size)),
        format!("×{top_k} Active"),
        format!("Top-{top_k} Routed Experts SwiGLU ({act_fn})"),
        format!(
          "{top_k} 个激活专家并行计算 [{d} → {moe_intermediate_size}]（其余 {} 个休眠 0 FLOPs）",
          num_routed_experts.saturating_sub(*top_k)
        ),
        format!("单层总参 {} / 激活 {}", format_count(ffn_tot), format_count(ffn_act)),
        expert_col,
        row,
        Some(InspectorFocus::FfnOrMoe),
        Some(layer),
      );
      let exp_down = g.add_node(
        OpKind::ConvOrLinear,
        format!("{top_k}×{}", preset.fmt_seq(d)),
        "exp_down".to_string(),
        "Routed Experts Down Projection".to_string(),
        format!("Top-{top_k} × Linear [{moe_intermediate_size} → {d}]"),
        "等待按路由权重 g_i 加权汇总".to_string(),
        expert_col,
        row + 1,
        Some(InspectorFocus::FfnOrMoe),
        Some(layer),
      );
      g.edge(&ffn_in_id, &exp_up);
      g.edge(&exp_up, &exp_down);
      row += 2;

      let add2 = g.add_node(
        OpKind::Merge,
        preset.fmt_seq(d),
        "Sum + Add".to_string(),
        "MoE Weighted Combine + Residual Add #2".to_string(),
        if has_shared {
          format!("h_out = h_mid + Shared(x) + Σ g_i·Expert_i(x) → [{}]", preset.fmt_seq(d))
        } else {
          format!("h_out = h_mid + Σ g_i·Expert_i(x) → [{}]", preset.fmt_seq(d))
        },
        format!("每层稀疏激活仅 {} / 总参 {}", format_count(ffn_act), format_count(ffn_tot)),
        0.0,
        row,
        Some(InspectorFocus::FfnOrMoe),
        Some(layer),
      );
      if let Some(sh) = shared_leaf {
        g.edge(&sh, &add2);
      }
      g.edge(&topk_id, &add2);
      g.edge(&exp_down, &add2);
      g.edge_labeled(&attn_add_id, &add2, "Skip", true);
      add2
    }
  };

  (ffn_add_id, row, min_col, max_col)
}

/// 经典 CV 教学对照 1：**Inception-v3**（精确还原 4 分支并行卷积/池化/拼接 + 8 色完整算子图例）。
pub fn build_inception_v3_graph(preset: ShapePreset) -> TensorGraph {
  let mut g = GraphBuilder::new();

  let n0 = g.add_node(
    OpKind::Input,
    preset.fmt_cv(299, 299, 3),
    "RGB".to_string(),
    "输入图像 (Inception-v3 Standard 299×299)".to_string(),
    "ImageNet RGB Input Tensor [299 × 299 × 3]".to_string(),
    "输入分辨率 299×299，3 通道".to_string(),
    0.0,
    0,
    None,
    None,
  );
  let n1 = g.add_node(
    OpKind::ConvOrLinear,
    preset.fmt_cv(149, 149, 32),
    "3*3 /2".to_string(),
    "Stem Conv1 (3×3, stride=2, valid)".to_string(),
    "⌊(299 − 3)/2⌋ + 1 = 149 → [149×149×32]".to_string(),
    "参数量: 3×3×3×32 = 864".to_string(),
    0.0,
    1,
    None,
    None,
  );
  let n2 = g.add_node(
    OpKind::ConvOrLinear,
    preset.fmt_cv(147, 147, 32),
    "3*3".to_string(),
    "Stem Conv2 (3×3, stride=1, valid)".to_string(),
    "149 − 3 + 1 = 147 → [147×147×32]".to_string(),
    "参数量: 3×3×32×32 = 9.2K".to_string(),
    0.0,
    2,
    None,
    None,
  );
  let n3 = g.add_node(
    OpKind::ConvOrLinear,
    preset.fmt_cv(147, 147, 64),
    "3*3 pad".to_string(),
    "Stem Conv3 (3×3, stride=1, same)".to_string(),
    "Same Padding 保持空间尺寸 147×147，通道升至 64".to_string(),
    "参数量: 3×3×32×64 = 18.4K".to_string(),
    0.0,
    3,
    None,
    None,
  );
  let n4 = g.add_node(
    OpKind::MaxPool,
    preset.fmt_cv(73, 73, 64),
    "3*3 /2".to_string(),
    "Stem MaxPool1 (3×3, stride=2)".to_string(),
    "⌊(147 − 3)/2⌋ + 1 = 73 → [73×73×64]".to_string(),
    "零参数空间下采样 2×".to_string(),
    0.0,
    4,
    None,
    None,
  );
  let n5 = g.add_node(
    OpKind::ConvOrLinear,
    preset.fmt_cv(73, 73, 80),
    "1*1".to_string(),
    "Stem Conv4 (1×1 Pointwise)".to_string(),
    "1×1 卷积通道扩展 64 → 80".to_string(),
    "参数量: 64×80 = 5.1K".to_string(),
    0.0,
    5,
    None,
    None,
  );
  let n6 = g.add_node(
    OpKind::ConvOrLinear,
    preset.fmt_cv(71, 71, 192),
    "3*3".to_string(),
    "Stem Conv5 (3×3, stride=1, valid)".to_string(),
    "73 − 3 + 1 = 71 → [71×71×192]".to_string(),
    "参数量: 3×3×80×192 = 138.2K".to_string(),
    0.0,
    6,
    None,
    None,
  );
  let n7 = g.add_node(
    OpKind::MaxPool,
    preset.fmt_cv(35, 35, 192),
    "3*3 /2".to_string(),
    "Stem MaxPool2 (3×3, stride=2)".to_string(),
    "⌊(71 − 3)/2⌋ + 1 = 35 → [35×35×192]".to_string(),
    "Stem 末端特征图，进入 Inception-A 4 分支".to_string(),
    0.0,
    7,
    None,
    None,
  );

  g.edge(&n0, &n1);
  g.edge(&n1, &n2);
  g.edge(&n2, &n3);
  g.edge(&n3, &n4);
  g.edge(&n4, &n5);
  g.edge(&n5, &n6);
  g.edge(&n6, &n7);

  // Inception-A 经典 4 路并行多尺度分支（完全对齐示例图）
  let b1_pool = g.add_node(
    OpKind::AvgPool,
    preset.fmt_cv(35, 35, 192),
    "3*3".to_string(),
    "Branch 1: AvgPool2d (3×3, stride=1, same)".to_string(),
    "局部平均池化保留平滑上下文 [35×35×192]".to_string(),
    "零参数池化支路".to_string(),
    -1.5,
    8,
    None,
    None,
  );
  let b1_conv = g.add_node(
    OpKind::ConvOrLinear,
    preset.fmt_cv(35, 35, 32),
    "1*1".to_string(),
    "Branch 1: 1×1 Conv Projection".to_string(),
    "通道压缩 192 → 32，贡献 32 通道至 Concat".to_string(),
    "参数量: 192×32 = 6.1K".to_string(),
    -1.5,
    9,
    None,
    None,
  );

  let b2_c1 = g.add_node(
    OpKind::ConvOrLinear,
    preset.fmt_cv(35, 35, 64),
    "1*1".to_string(),
    "Branch 2: 1×1 Bottleneck Conv".to_string(),
    "降维 192 → 64，减少后续双 3×3 计算量".to_string(),
    "参数量: 192×64 = 12.3K".to_string(),
    -0.5,
    8,
    None,
    None,
  );
  let b2_c2 = g.add_node(
    OpKind::ConvOrLinear,
    preset.fmt_cv(35, 35, 96),
    "3*3".to_string(),
    "Branch 2: First 3×3 Conv".to_string(),
    "3×3 卷积提取中尺度特征 [35×35×96]".to_string(),
    "参数量: 9×64×96 = 55.3K".to_string(),
    -0.5,
    9,
    None,
    None,
  );
  let b2_c3 = g.add_node(
    OpKind::ConvOrLinear,
    preset.fmt_cv(35, 35, 96),
    "3*3".to_string(),
    "Branch 2: Second 3×3 Conv (Equivalent 5×5)".to_string(),
    "双层 3×3 串联等效 5×5 感受野，贡献 96 通道".to_string(),
    "参数量: 9×96×96 = 82.9K".to_string(),
    -0.5,
    10,
    None,
    None,
  );

  let b3_c1 = g.add_node(
    OpKind::ConvOrLinear,
    preset.fmt_cv(35, 35, 48),
    "1*1".to_string(),
    "Branch 3: 1×1 Bottleneck Conv".to_string(),
    "降维 192 → 48，为 5×5 大核节省 FLOPs".to_string(),
    "参数量: 192×48 = 9.2K".to_string(),
    0.5,
    8,
    None,
    None,
  );
  let b3_c2 = g.add_node(
    OpKind::ConvOrLinear,
    preset.fmt_cv(35, 35, 64),
    "5*5".to_string(),
    "Branch 3: 5×5 Large-Kernel Conv".to_string(),
    "大感受野 5×5 卷积提取全局纹理，贡献 64 通道".to_string(),
    "参数量: 25×48×64 = 76.8K".to_string(),
    0.5,
    9,
    None,
    None,
  );

  let b4_c1 = g.add_node(
    OpKind::ConvOrLinear,
    preset.fmt_cv(35, 35, 64),
    "1*1".to_string(),
    "Branch 4: Direct 1×1 Conv".to_string(),
    "点卷积直连分支，保留局部精细像素特征，贡献 64 通道".to_string(),
    "参数量: 192×64 = 12.3K".to_string(),
    1.5,
    9,
    None,
    None,
  );

  let concat_a = g.add_node(
    OpKind::Merge,
    preset.fmt_cv(35, 35, 256),
    "Concat".to_string(),
    "Inception-A Channel Concatenation".to_string(),
    "Concat(32 + 96 + 64 + 64) = 256 通道 → [35×35×256]".to_string(),
    "四分支多尺度特征沿通道维拼接".to_string(),
    0.0,
    11,
    None,
    None,
  );

  g.edge(&n7, &b1_pool);
  g.edge(&b1_pool, &b1_conv);
  g.edge(&n7, &b2_c1);
  g.edge(&b2_c1, &b2_c2);
  g.edge(&b2_c2, &b2_c3);
  g.edge(&n7, &b3_c1);
  g.edge(&b3_c1, &b3_c2);
  g.edge(&n7, &b4_c1);
  g.edge(&b1_conv, &concat_a);
  g.edge(&b2_c3, &concat_a);
  g.edge(&b3_c2, &concat_a);
  g.edge(&b4_c1, &concat_a);
  g.add_group(
    8,
    11,
    -1.5,
    1.5,
    "Inception-A 4-Branch Block (×3)".to_string(),
    "AvgPool(32) + 3×3×2(96) + 5×5(64) + 1×1(64) → Concat 256ch".to_string(),
  );

  // 下游 Reduction + Inception-B/C + 分类头（补全 Dropout / FC / Softmax）
  let stage_b = g.add_node(
    OpKind::ConvOrLinear,
    preset.fmt_cv(17, 17, 768),
    "Inc-B ×5".to_string(),
    "Grid Reduction + Inception-B (1×7 & 7×1 Factorized ×5)".to_string(),
    "空间下采样 35×35 → 17×17，非对称分解卷积堆叠 5 组 → 768 通道".to_string(),
    "用 1×7 + 7×1 替代 7×7 降低 33% 计算量".to_string(),
    0.0,
    12,
    None,
    None,
  );
  let stage_c = g.add_node(
    OpKind::ConvOrLinear,
    preset.fmt_cv(8, 8, 2048),
    "Inc-C ×3".to_string(),
    "Grid Reduction + Inception-C (High-Dim Expand ×3)".to_string(),
    "空间下采样 17×17 → 8×8，通道扩展至 2048".to_string(),
    "深层语义特征汇聚".to_string(),
    0.0,
    13,
    None,
    None,
  );
  let gap = g.add_node(
    OpKind::AvgPool,
    preset.fmt_cv(1, 1, 2048),
    "8*8 GAP".to_string(),
    "Global Average Pooling (8×8)".to_string(),
    "8×8 空间全局平均池化 → [1×1×2048] 向量".to_string(),
    "替代大尺寸 Flatten 全连接，大幅防过拟合".to_string(),
    0.0,
    14,
    None,
    None,
  );
  let drop = g.add_node(
    OpKind::Dropout,
    format!("{}×2048", preset.batch_str()),
    "p=0.2".to_string(),
    "Dropout Regularization (p = 0.2)".to_string(),
    "训练期以 20% 概率随机置零通道，推理期恒等直通".to_string(),
    "抑制分类头共适应过拟合".to_string(),
    0.0,
    15,
    None,
    None,
  );
  let fc = g.add_node(
    OpKind::FcHead,
    format!("{}×1000", preset.batch_str()),
    "2048→1000".to_string(),
    "Fully Connected Classifier (FC)".to_string(),
    "Linear [2048 → 1000] → ImageNet-1K Logits".to_string(),
    "全连接头参数: 2.05M（全网总参 23.8M）".to_string(),
    0.0,
    16,
    None,
    None,
  );
  let sm = g.add_node(
    OpKind::Softmax,
    format!("{}×1000", preset.batch_str()),
    "1K cls".to_string(),
    "Softmax Classification Probabilities".to_string(),
    "Softmax(logits) → 1000 类归一化概率分布".to_string(),
    "Top-1 / Top-5 图像分类输出".to_string(),
    0.0,
    17,
    None,
    None,
  );

  g.edge(&concat_a, &stage_b);
  g.edge(&stage_b, &stage_c);
  g.edge(&stage_c, &gap);
  g.edge(&gap, &drop);
  g.edge(&drop, &fc);
  g.edge(&fc, &sm);

  g.finish(
    "inception-v3".to_string(),
    "Inception-v3 · 经典多分支 CNN 张量维流图".to_string(),
    format!(
      "23.8M Params · 4-Branch Multi-Scale Inception-A · 8 类经典算子色块 · {}",
      preset.label()
    ),
    LegendStyle::ClassicCnn,
  )
}

/// 经典 CV 教学对照 2：**ResNet-50**（残差瓶颈 1×1→3×3→1×1 + Shortcut 跳连）。
pub fn build_resnet50_graph(preset: ShapePreset) -> TensorGraph {
  let mut g = GraphBuilder::new();

  let n0 = g.add_node(
    OpKind::Input,
    preset.fmt_cv(224, 224, 3),
    "RGB".to_string(),
    "输入图像 (ImageNet 224×224×3)".to_string(),
    "RGB Input Tensor [224×224×3]".to_string(),
    "ResNet-50 标准输入分辨率".to_string(),
    0.0,
    0,
    None,
    None,
  );
  let n1 = g.add_node(
    OpKind::ConvOrLinear,
    preset.fmt_cv(112, 112, 64),
    "7*7 /2".to_string(),
    "Stem Conv1 (7×7, stride=2) + BN + ReLU".to_string(),
    "7×7 大核步长 2 下采样 → [112×112×64]".to_string(),
    "参数量: 7×7×3×64 = 9.4K".to_string(),
    0.0,
    1,
    None,
    None,
  );
  let n2 = g.add_node(
    OpKind::MaxPool,
    preset.fmt_cv(56, 56, 64),
    "3*3 /2".to_string(),
    "Stem MaxPool (3×3, stride=2)".to_string(),
    "空间下采样 2× → [56×56×64]".to_string(),
    "进入 Conv2_x 残差瓶颈阶段".to_string(),
    0.0,
    2,
    None,
    None,
  );
  g.edge(&n0, &n1);
  g.edge(&n1, &n2);

  // Bottleneck Residual Block (Conv2_x ×3)
  let skip = g.add_node(
    OpKind::ConvOrLinear,
    preset.fmt_cv(56, 56, 256),
    "1*1 Skip".to_string(),
    "Shortcut Projection (1×1 Conv, 仅首块升维)".to_string(),
    "首块 64 → 256 对齐通道，后续 2 块为恒等直连 Identity".to_string(),
    "消除深层网络梯度消失".to_string(),
    -0.85,
    4,
    None,
    None,
  );
  let b1 = g.add_node(
    OpKind::ConvOrLinear,
    preset.fmt_cv(56, 56, 64),
    "1*1".to_string(),
    "Bottleneck Conv 1×1 (通道压缩)".to_string(),
    "1×1 卷积压缩通道至 64，显著降低 3×3 计算开销".to_string(),
    "BN + ReLU 激活".to_string(),
    0.75,
    3,
    None,
    None,
  );
  let b2 = g.add_node(
    OpKind::ConvOrLinear,
    preset.fmt_cv(56, 56, 64),
    "3*3".to_string(),
    "Bottleneck Conv 3×3 (空间特征提取)".to_string(),
    "在低维 64 通道上执行 3×3 空间卷积".to_string(),
    "参数量: 9×64×64 = 36.9K".to_string(),
    0.75,
    4,
    None,
    None,
  );
  let b3 = g.add_node(
    OpKind::ConvOrLinear,
    preset.fmt_cv(56, 56, 256),
    "1*1 ×4".to_string(),
    "Bottleneck Conv 1×1 (4× 通道恢复)".to_string(),
    "1×1 卷积将通道扩回 256 (4× expansion)".to_string(),
    "与 Shortcut 支路通道严格对齐".to_string(),
    0.75,
    5,
    None,
    None,
  );
  let add = g.add_node(
    OpKind::Merge,
    preset.fmt_cv(56, 56, 256),
    "Add (+)".to_string(),
    "Residual Elementwise Add + ReLU".to_string(),
    "y = ReLU(F(x) + Shortcut(x)) → [56×56×256]".to_string(),
    "残差恒等映射核心节点".to_string(),
    0.0,
    6,
    None,
    None,
  );
  g.edge_labeled(&n2, &skip, "Shortcut", true);
  g.edge(&n2, &b1);
  g.edge(&b1, &b2);
  g.edge(&b2, &b3);
  g.edge(&skip, &add);
  g.edge(&b3, &add);
  g.add_group(
    3,
    6,
    -0.85,
    0.75,
    "Bottleneck Residual Block · Conv2_x (×3)".to_string(),
    "1×1(64) → 3×3(64) → 1×1(256) + Shortcut Add".to_string(),
  );

  let s3 = g.add_node(
    OpKind::ConvOrLinear,
    preset.fmt_cv(28, 28, 512),
    "Conv3 ×4".to_string(),
    "Stage 3 · Conv3_x Bottleneck (×4, stride=2)".to_string(),
    "空间 56→28，通道 256→512（4 个 Bottleneck 块）".to_string(),
    "中间层瓶颈宽度 128".to_string(),
    0.0,
    7,
    None,
    None,
  );
  let s4 = g.add_node(
    OpKind::ConvOrLinear,
    preset.fmt_cv(14, 14, 1024),
    "Conv4 ×6".to_string(),
    "Stage 4 · Conv4_x Bottleneck (×6, stride=2)".to_string(),
    "空间 28→14，通道 512→1024（6 个 Bottleneck 块）".to_string(),
    "中间层瓶颈宽度 256".to_string(),
    0.0,
    8,
    None,
    None,
  );
  let s5 = g.add_node(
    OpKind::ConvOrLinear,
    preset.fmt_cv(7, 7, 2048),
    "Conv5 ×3".to_string(),
    "Stage 5 · Conv5_x Bottleneck (×3, stride=2)".to_string(),
    "空间 14→7，通道 1024→2048（3 个 Bottleneck 块）".to_string(),
    "中间层瓶颈宽度 512".to_string(),
    0.0,
    9,
    None,
    None,
  );
  let gap = g.add_node(
    OpKind::AvgPool,
    preset.fmt_cv(1, 1, 2048),
    "7*7 GAP".to_string(),
    "Global Average Pooling (7×7)".to_string(),
    "7×7 空间平均池化 → [1×1×2048]".to_string(),
    "零参数全局池化".to_string(),
    0.0,
    10,
    None,
    None,
  );
  let fc = g.add_node(
    OpKind::FcHead,
    format!("{}×1000", preset.batch_str()),
    "2048→1000".to_string(),
    "Fully Connected Classifier (FC)".to_string(),
    "Linear [2048 → 1000] → Logits".to_string(),
    "ResNet-50 总参 25.6M".to_string(),
    0.0,
    11,
    None,
    None,
  );
  let sm = g.add_node(
    OpKind::Softmax,
    format!("{}×1000", preset.batch_str()),
    "1K cls".to_string(),
    "Softmax Output".to_string(),
    "Softmax(logits) → 1000 类概率分布".to_string(),
    "ImageNet 分类输出".to_string(),
    0.0,
    12,
    None,
    None,
  );
  g.edge(&add, &s3);
  g.edge(&s3, &s4);
  g.edge(&s4, &s5);
  g.edge(&s5, &gap);
  g.edge(&gap, &fc);
  g.edge(&fc, &sm);

  g.finish(
    "resnet-50".to_string(),
    "ResNet-50 · 经典残差瓶颈 CNN 张量维流图".to_string(),
    format!("25.6M Params · Bottleneck 1×1→3×3→1×1 + Shortcut · {}", preset.label()),
    LegendStyle::ClassicCnn,
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// 确定性网格几何布局（主画布与右上角图例数学隔离，0% 重叠）
// ─────────────────────────────────────────────────────────────────────────────

const NODE_W: f64 = 158.0;
const NODE_H: f64 = 34.0;
const COL_STEP: f64 = 236.0;
const ROW_STEP: f64 = 74.0;
const TOP_PAD: f64 = 52.0;
const LEFT_RAIL_PAD: f64 = 68.0;
const LEGEND_W: f64 = 210.0;
const LEGEND_GAP: f64 = 30.0;
const RIGHT_PAD: f64 = 24.0;
const BOTTOM_PAD: f64 = 36.0;

struct LayoutMetrics {
  canvas_w: f64,
  canvas_h: f64,
  graph_origin_x: f64,
  legend_x: f64,
  legend_y: f64,
  coords: HashMap<String, (f64, f64)>,
}

fn compute_layout(graph: &TensorGraph) -> LayoutMetrics {
  let min_col = graph.nodes.iter().map(|n| n.col).fold(0.0_f64, f64::min);
  let max_col = graph.nodes.iter().map(|n| n.col).fold(0.0_f64, f64::max);
  let col_span = (max_col - min_col).max(1.2);
  let graph_inner_w = col_span * COL_STEP + NODE_W + 84.0;
  let graph_origin_x = LEFT_RAIL_PAD - min_col * COL_STEP + (NODE_W / 2.0);
  let legend_x = LEFT_RAIL_PAD + graph_inner_w + LEGEND_GAP;
  let canvas_w = legend_x + LEGEND_W + RIGHT_PAD;

  let legend_count = graph.active_legend().len();
  let legend_h = 34.0 + (legend_count as f64) * 28.0 + 16.0;
  let nodes_h = TOP_PAD + (graph.total_rows as f64) * ROW_STEP + BOTTOM_PAD;
  let canvas_h = nodes_h.max(TOP_PAD + legend_h + BOTTOM_PAD);

  let mut coords = HashMap::with_capacity(graph.nodes.len());
  for n in &graph.nodes {
    let cx = graph_origin_x + n.col * COL_STEP;
    let cy = TOP_PAD + (n.row as f64) * ROW_STEP + (NODE_H / 2.0);
    coords.insert(n.id.clone(), (cx, cy));
  }

  LayoutMetrics { canvas_w, canvas_h, graph_origin_x, legend_x, legend_y: TOP_PAD, coords }
}

/// 生成边的 SVG 路径与末端实心箭头多边形点集（不依赖 `<marker id>`，无跨组件 ID 冲突）。
fn compute_edge_geometry(
  edge: &TensorEdge,
  from: (f64, f64),
  to: (f64, f64),
  from_row: usize,
  to_row: usize,
  skip_rail_x: f64,
) -> (String, String, Option<(f64, f64)>) {
  let (x1, y1_c) = from;
  let (x2, y2_c) = to;
  let y1 = y1_c + NODE_H / 2.0;
  let y2 = y2_c - NODE_H / 2.0;

  if edge.dashed && to_row > from_row + 1 {
    // 残差 Skip 长跳连：沿左侧外轨绕行并垂直汇入目标左侧，避免横穿内部算子框
    let start_x = x1 - NODE_W / 2.0;
    let end_x = x2 - NODE_W / 2.0;
    let d = format!(
      "M {start_x:.1} {y1_c:.1} L {skip_rail_x:.1} {y1_c:.1} L {skip_rail_x:.1} {y2_c:.1} L {:.1} {y2_c:.1}",
      end_x - 7.0
    );
    let arrow = format!(
      "{end_x:.1},{y2_c:.1} {:.1},{:.1} {:.1},{:.1}",
      end_x - 8.0,
      y2_c - 4.2,
      end_x - 8.0,
      y2_c + 4.2
    );
    let label_pos = Some((skip_rail_x + 6.0, (y1_c + y2_c) / 2.0));
    return (d, arrow, label_pos);
  }

  // 正常直连或扇出/扇入斜线
  let dx = x2 - x1;
  let dy = (y2 - y1).max(1.0);
  let len = (dx * dx + dy * dy).sqrt().max(1.0);
  let ux = dx / len;
  let uy = dy / len;
  let tip_x = x2;
  let tip_y = y2;
  let base_x = tip_x - ux * 8.5;
  let base_y = tip_y - uy * 8.5;
  let perp_x = -uy * 4.2;
  let perp_y = ux * 4.2;

  let d = format!("M {x1:.1} {y1:.1} L {base_x:.1} {base_y:.1}");
  let arrow = format!(
    "{tip_x:.1},{tip_y:.1} {:.1},{:.1} {:.1},{:.1}",
    base_x + perp_x,
    base_y + perp_y,
    base_x - perp_x,
    base_y - perp_y
  );
  let label_pos =
    if !edge.label.is_empty() { Some(((x1 + x2) / 2.0 + 8.0, (y1 + y2) / 2.0)) } else { None };
  (d, arrow, label_pos)
}

fn truncate_label(s: &str, max_chars: usize) -> String {
  let count = s.chars().count();
  if count <= max_chars {
    s.to_string()
  } else {
    let prefix: String = s.chars().take(max_chars.saturating_sub(1)).collect();
    format!("{prefix}…")
  }
}

fn xml_escape(s: &str) -> String {
  s.replace('&', "&amp;")
    .replace('<', "&lt;")
    .replace('>', "&gt;")
    .replace('"', "&quot;")
    .replace('\'', "&apos;")
}

fn percent_encode_svg(svg: &str) -> String {
  let mut out = String::with_capacity(svg.len() + svg.len() / 4);
  for b in svg.bytes() {
    match b {
      b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => out.push(b as char),
      _ => {
        out.push('%');
        let hi = b >> 4;
        let lo = b & 0x0f;
        let hex = |v: u8| if v < 10 { (b'0' + v) as char } else { (b'A' + v - 10) as char };
        out.push(hex(hi));
        out.push(hex(lo));
      }
    }
  }
  out
}

/// 生成独立可下载的高清 `.svg` 矢量文件源码（内联全部样式，无需外部 Tailwind 即可在浏览器/Figma/PPT 中直接打开）。
pub fn export_standalone_svg(graph: &TensorGraph) -> String {
  let layout = compute_layout(graph);
  let legend = graph.active_legend();
  let node_map: HashMap<&str, &TensorNode> =
    graph.nodes.iter().map(|n| (n.id.as_str(), n)).collect();
  let min_col = graph.nodes.iter().map(|n| n.col).fold(0.0_f64, f64::min);
  let skip_rail_x = (layout.graph_origin_x + min_col * COL_STEP - NODE_W / 2.0 - 26.0).max(20.0);

  let mut out = String::with_capacity(16_384);
  out.push_str(&format!(
    "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {:.0} {:.0}\" width=\"{:.0}\" height=\"{:.0}\" style=\"background:#ffffff;font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace\">\n",
    layout.canvas_w, layout.canvas_h, layout.canvas_w, layout.canvas_h
  ));

  // 背景浅网格线
  out.push_str("  <g stroke=\"#e2e8f0\" stroke-width=\"0.75\">\n");
  let mut gx = 32.0;
  while gx < layout.canvas_w {
    out.push_str(&format!(
      "    <line x1=\"{gx:.0}\" y1=\"0\" x2=\"{gx:.0}\" y2=\"{:.0}\"/>\n",
      layout.canvas_h
    ));
    gx += 32.0;
  }
  let mut gy = 32.0;
  while gy < layout.canvas_h {
    out.push_str(&format!(
      "    <line x1=\"0\" y1=\"{gy:.0}\" x2=\"{:.0}\" y2=\"{gy:.0}\"/>\n",
      layout.canvas_w
    ));
    gy += 32.0;
  }
  out.push_str("  </g>\n");

  // 顶部标题
  out.push_str(&format!(
    "  <text x=\"24\" y=\"28\" font-size=\"14\" font-weight=\"700\" fill=\"#0f172a\">{}</text>\n",
    xml_escape(&graph.title)
  ));

  // 重复层段外框
  for grp in &graph.groups {
    let x = layout.graph_origin_x + grp.min_col * COL_STEP - NODE_W / 2.0 - 16.0;
    let right_x = layout.graph_origin_x + grp.max_col * COL_STEP + NODE_W / 2.0 + 78.0;
    let y = TOP_PAD + (grp.row_start as f64) * ROW_STEP - 14.0;
    let h = ((grp.row_end - grp.row_start) as f64) * ROW_STEP + NODE_H + 24.0;
    let w = (right_x - x).max(NODE_W + 40.0);
    out.push_str(&format!(
      "  <rect x=\"{x:.1}\" y=\"{y:.1}\" width=\"{w:.1}\" height=\"{h:.1}\" rx=\"10\" fill=\"#f8fafc\" fill-opacity=\"0.68\" stroke=\"#94a3b8\" stroke-width=\"1.2\" stroke-dasharray=\"5 4\"/>\n"
    ));
    out.push_str(&format!(
      "  <text x=\"{:.1}\" y=\"{:.1}\" font-size=\"10.5\" font-weight=\"700\" fill=\"#334155\">{}</text>\n",
      x + 10.0,
      y - 4.0,
      xml_escape(&grp.badge)
    ));
  }

  // 有向连线与箭头
  for edge in &graph.edges {
    if let (Some(&from_pt), Some(&to_pt), Some(fn_node), Some(tn_node)) = (
      layout.coords.get(&edge.from_id),
      layout.coords.get(&edge.to_id),
      node_map.get(edge.from_id.as_str()),
      node_map.get(edge.to_id.as_str()),
    ) {
      let (d_path, arrow_pts, lbl_pos) =
        compute_edge_geometry(edge, from_pt, to_pt, fn_node.row, tn_node.row, skip_rail_x);
      let stroke = if edge.dashed { "#6366f1" } else { "#334155" };
      let dash = if edge.dashed { " stroke-dasharray=\"4 3\"" } else { "" };
      out.push_str(&format!(
        "  <path d=\"{d_path}\" fill=\"none\" stroke=\"{stroke}\" stroke-width=\"1.5\"{dash}/>\n"
      ));
      out.push_str(&format!("  <polygon points=\"{arrow_pts}\" fill=\"{stroke}\"/>\n"));
      if let Some((lx, ly)) = lbl_pos {
        out.push_str(&format!(
          "  <text x=\"{lx:.1}\" y=\"{ly:.1}\" font-size=\"9.5\" font-weight=\"700\" fill=\"{stroke}\">{}</text>\n",
          xml_escape(&edge.label)
        ));
      }
    }
  }

  // 色块节点与右侧注解
  for n in &graph.nodes {
    if let Some(&(cx, cy)) = layout.coords.get(&n.id) {
      let (fill, stroke, text_col) = n.kind.palette();
      let rx_x = cx - NODE_W / 2.0;
      let rx_y = cy - NODE_H / 2.0;
      out.push_str(&format!(
        "  <rect x=\"{rx_x:.1}\" y=\"{rx_y:.1}\" width=\"{NODE_W:.1}\" height=\"{NODE_H:.1}\" rx=\"5\" fill=\"{fill}\" stroke=\"{stroke}\" stroke-width=\"1.2\"/>\n"
      ));
      out.push_str(&format!(
        "  <text x=\"{cx:.1}\" y=\"{:.1}\" text-anchor=\"middle\" font-size=\"11.5\" font-weight=\"700\" fill=\"{text_col}\">{}</text>\n",
        cy + 4.0,
        xml_escape(&truncate_label(&n.shape, 20))
      ));
      if !n.side_note.is_empty() {
        out.push_str(&format!(
          "  <text x=\"{:.1}\" y=\"{:.1}\" font-size=\"10.5\" font-weight=\"600\" fill=\"#334155\">{}</text>\n",
          rx_x + NODE_W + 8.0,
          cy + 4.0,
          xml_escape(&truncate_label(&n.side_note, 13))
        ));
      }
    }
  }

  // 右上角算子色块图例表
  let lx = layout.legend_x;
  let ly = layout.legend_y;
  let lh = 30.0 + (legend.len() as f64) * 28.0;
  out.push_str(&format!(
    "  <rect x=\"{lx:.1}\" y=\"{ly:.1}\" width=\"{LEGEND_W:.1}\" height=\"{lh:.1}\" rx=\"6\" fill=\"#ffffff\" stroke=\"#cbd5e1\" stroke-width=\"1.2\"/>\n"
  ));
  out.push_str(&format!(
    "  <text x=\"{:.1}\" y=\"{:.1}\" font-size=\"11\" font-weight=\"700\" fill=\"#0f172a\">算子色块图例 (Legend)</text>\n",
    lx + 12.0,
    ly + 20.0
  ));
  for (idx, (kind, lbl)) in legend.iter().enumerate() {
    let row_y = ly + 30.0 + (idx as f64) * 28.0;
    let (fill, stroke, _) = kind.palette();
    out.push_str(&format!(
      "  <line x1=\"{lx:.1}\" y1=\"{row_y:.1}\" x2=\"{:.1}\" y2=\"{row_y:.1}\" stroke=\"#e2e8f0\" stroke-width=\"1\"/>\n",
      lx + LEGEND_W
    ));
    out.push_str(&format!(
      "  <rect x=\"{:.1}\" y=\"{:.1}\" width=\"48\" height=\"18\" rx=\"3\" fill=\"{fill}\" stroke=\"{stroke}\" stroke-width=\"1\"/>\n",
      lx + 10.0,
      row_y + 5.0
    ));
    out.push_str(&format!(
      "  <text x=\"{:.1}\" y=\"{:.1}\" font-size=\"11\" font-weight=\"600\" fill=\"#1e293b\">{}</text>\n",
      lx + 68.0,
      row_y + 18.0,
      xml_escape(lbl)
    ));
  }

  out.push_str("</svg>\n");
  out
}

/// 交互式「全网张量维流图（Tensor Flow DAG）」组件。
#[component]
pub fn TensorFlowGraphView(
  blueprint: ModelBlueprint,
  on_drill_to_workbench: EventHandler<(Option<u64>, InspectorFocus)>,
) -> Element {
  let mut source = use_signal(|| GraphSource::CurrentModel);
  let mut shape_preset = use_signal(|| ShapePreset::Symbolic);
  let mut compact = use_signal(|| false);
  let mut selected_node_id: Signal<Option<String>> = use_signal(|| None);

  let graph = match source() {
    GraphSource::CurrentModel => {
      build_blueprint_tensor_graph(&blueprint, shape_preset(), compact())
    }
    GraphSource::InceptionV3 => build_inception_v3_graph(shape_preset()),
    GraphSource::ResNet50 => build_resnet50_graph(shape_preset()),
  };

  let layout = compute_layout(&graph);
  let legend = graph.active_legend();
  let min_col = graph.nodes.iter().map(|n| n.col).fold(0.0_f64, f64::min);
  let skip_rail_x = (layout.graph_origin_x + min_col * COL_STEP - NODE_W / 2.0 - 26.0).max(20.0);

  let node_map: HashMap<String, TensorNode> =
    graph.nodes.iter().map(|n| (n.id.clone(), n.clone())).collect();

  // 当前选中节点（默认高亮首个合并/核心节点，确保底部信息条始终有实用信息）
  let active_node: Option<TensorNode> =
    selected_node_id().and_then(|id| node_map.get(&id).cloned()).or_else(|| {
      graph
        .nodes
        .iter()
        .find(|n| matches!(n.kind, OpKind::Attention | OpKind::Merge))
        .or_else(|| graph.nodes.first())
        .cloned()
    });
  let active_id_str = active_node.as_ref().map(|n| n.id.clone()).unwrap_or_default();

  // 零 JS 独立 SVG 下载链接
  let standalone_svg = export_standalone_svg(&graph);
  let svg_data_uri =
    format!("data:image/svg+xml;charset=utf-8,{}", percent_encode_svg(&standalone_svg));
  let download_filename = format!("{}-tensor-graph.svg", graph.file_slug);

  // 预生成网格线坐标
  let mut v_grid: Vec<f64> = Vec::new();
  let mut gx = 32.0;
  while gx < layout.canvas_w {
    v_grid.push(gx);
    gx += 32.0;
  }
  let mut h_grid: Vec<f64> = Vec::new();
  let mut gy = 32.0;
  while gy < layout.canvas_h {
    h_grid.push(gy);
    gy += 32.0;
  }

  let legend_h = 30.0 + (legend.len() as f64) * 28.0;
  let short_repo = truncate_label(&blueprint.repo_id, 26);

  rsx! {
    div { class: card_class("p-4 sm:p-5 bg-white dark:bg-slate-900/70 space-y-4"),
      // 1. 顶部控制栏：网络源选择 + 维度代入 + 粒度切换 + 导出高清 SVG
      div { class: "flex flex-wrap items-center justify-between gap-3 pb-3.5 border-b border-slate-200 dark:border-slate-800",
        div { class: "flex flex-wrap items-center gap-2",
          span { class: "text-xs font-bold text-slate-500 dark:text-slate-400", "网络拓扑:" }
          for (src_opt, src_lbl) in [
            (GraphSource::CurrentModel, format!("当前模型 ({short_repo})")),
            (GraphSource::InceptionV3, "Inception-v3 (经典 CNN)".to_string()),
            (GraphSource::ResNet50, "ResNet-50 (残差 CNN)".to_string()),
          ] {
            button {
              key: "{src_lbl}",
              r#type: "button",
              class: if source() == src_opt {
                "px-2.5 py-1 rounded-md text-xs font-bold bg-indigo-600 text-white shadow-2xs cursor-pointer"
              } else {
                "px-2.5 py-1 rounded-md text-xs font-medium bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700 cursor-pointer"
              },
              onclick: move |_| {
                source.set(src_opt);
                selected_node_id.set(None);
              },
              "{src_lbl}"
            }
          }
        }

        div { class: "flex flex-wrap items-center gap-2",
          // 维度模式切换
          div { class: "inline-flex items-center rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-950 p-0.5",
            for preset_opt in [
              ShapePreset::Symbolic,
              ShapePreset::Decode1,
              ShapePreset::Prefill512,
              ShapePreset::Batch4K,
            ] {
              button {
                key: "{preset_opt.label()}",
                r#type: "button",
                class: if shape_preset() == preset_opt {
                  "px-2 py-1 rounded-md text-[11px] font-mono font-bold bg-slate-900 text-white dark:bg-white dark:text-slate-900 cursor-pointer"
                } else {
                  "px-2 py-1 rounded-md text-[11px] font-mono font-medium text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white cursor-pointer"
                },
                onclick: move |_| shape_preset.set(preset_opt),
                "{preset_opt.label()}"
              }
            }
          }

          if source() == GraphSource::CurrentModel {
            button {
              r#type: "button",
              class: "px-2.5 py-1.5 rounded-lg text-xs font-semibold border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 text-slate-700 dark:text-slate-200 hover:border-indigo-400 cursor-pointer",
              onclick: move |_| compact.set(!compact()),
              if compact() { "展开 Norm 节点" } else { "紧凑主干流" }
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

      // 2. 标题与统计副标题
      div { class: "flex flex-wrap items-baseline justify-between gap-2",
        div {
          h3 { class: "text-sm sm:text-[15px] font-bold text-slate-900 dark:text-white",
            "{graph.title}"
          }
          p { class: "text-xs text-slate-500 dark:text-slate-400 mt-0.5 font-mono",
            "{graph.subtitle}"
          }
        }
        span { class: "text-xs text-slate-500 dark:text-slate-400",
          "共 {graph.nodes.len()} 个张量算子节点 · 点击任意色块查看精确变换公式"
        }
      }

      // 3. 网格蓝图 SVG 画布（内置右上角算子色块图例，支持横向滚动适配窄屏）
      div { class: "w-full overflow-x-auto rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950/90 p-2 sm:p-3",
        svg {
          view_box: "0 0 {layout.canvas_w:.0} {layout.canvas_h:.0}",
          class: "w-full min-w-[780px] h-auto font-mono select-none",
          role: "img",
          "aria-label": "{graph.title}",

          // 3A. 浅色/暗黑自适应工程坐标网格
          g { class: "stroke-slate-200/80 dark:stroke-slate-800/80", stroke_width: "0.75",
            for x in v_grid {
              line { x1: "{x:.0}", y1: "0", x2: "{x:.0}", y2: "{layout.canvas_h:.0}" }
            }
            for y in h_grid {
              line { x1: "0", y1: "{y:.0}", x2: "{layout.canvas_w:.0}", y2: "{y:.0}" }
            }
          }

          // 3B. 重复层段 / Block 外框
          for grp in &graph.groups {
            {
              let x = layout.graph_origin_x + grp.min_col * COL_STEP - NODE_W / 2.0 - 16.0;
              let right_x = layout.graph_origin_x + grp.max_col * COL_STEP + NODE_W / 2.0 + 78.0;
              let y = TOP_PAD + (grp.row_start as f64) * ROW_STEP - 14.0;
              let h = ((grp.row_end - grp.row_start) as f64) * ROW_STEP + NODE_H + 24.0;
              let w = (right_x - x).max(NODE_W + 40.0);
              let lbl_x = x + 10.0;
              let lbl_y = y - 5.0;
              rsx! {
                g { key: "{grp.badge}-{grp.row_start}",
                  rect {
                    x: "{x:.1}",
                    y: "{y:.1}",
                    width: "{w:.1}",
                    height: "{h:.1}",
                    rx: "10",
                    class: "fill-slate-50/70 dark:fill-slate-900/50 stroke-slate-400 dark:stroke-slate-600",
                    stroke_width: "1.2",
                    stroke_dasharray: "5 4",
                  }
                  text {
                    x: "{lbl_x:.1}",
                    y: "{lbl_y:.1}",
                    font_size: "10.5",
                    class: "fill-slate-700 dark:fill-slate-300 font-bold",
                    "{grp.badge}"
                  }
                }
              }
            }
          }

          // 3C. 有向张量流连线与实心箭头
          for (idx, edge) in graph.edges.iter().enumerate() {
            if let (Some(&from_pt), Some(&to_pt), Some(fn_node), Some(tn_node)) = (
              layout.coords.get(&edge.from_id),
              layout.coords.get(&edge.to_id),
              node_map.get(&edge.from_id),
              node_map.get(&edge.to_id),
            ) {
              {
                let (d_path, arrow_pts, lbl_pos) =
                  compute_edge_geometry(edge, from_pt, to_pt, fn_node.row, tn_node.row, skip_rail_x);
                let path_cls = if edge.dashed {
                  "stroke-indigo-500 dark:stroke-indigo-400"
                } else {
                  "stroke-slate-700 dark:stroke-slate-400"
                };
                let poly_cls = if edge.dashed {
                  "fill-indigo-500 dark:fill-indigo-400"
                } else {
                  "fill-slate-700 dark:fill-slate-400"
                };
                rsx! {
                  g { key: "e-{idx}-{edge.from_id}-{edge.to_id}",
                    path {
                      d: "{d_path}",
                      fill: "none",
                      class: path_cls,
                      stroke_width: "1.5",
                      stroke_dasharray: if edge.dashed { "4 3" } else { "" },
                    }
                    polygon {
                      points: "{arrow_pts}",
                      class: poly_cls,
                    }
                    if let Some((lx, ly)) = lbl_pos {
                      text {
                        x: "{lx:.1}",
                        y: "{ly:.1}",
                        font_size: "9.5",
                        class: "fill-indigo-600 dark:fill-indigo-400 font-bold",
                        "{edge.label}"
                      }
                    }
                  }
                }
              }
            }
          }

          // 3D. 算子实色块节点 + 框内张量维度 + 右侧短参数注解
          for n in &graph.nodes {
            if let Some(&(cx, cy)) = layout.coords.get(&n.id) {
              {
                let (fill, stroke, text_col) = n.kind.palette();
                let rx_x = cx - NODE_W / 2.0;
                let rx_y = cy - NODE_H / 2.0;
                let is_sel = n.id == active_id_str;
                let nid = n.id.clone();
                let shape_txt = truncate_label(&n.shape, 20);
                let note_txt = truncate_label(&n.side_note, 13);
                let note_x = rx_x + NODE_W + 8.0;
                let text_y = cy + 4.0;
                rsx! {
                  g {
                    key: "{n.id}",
                    class: "cursor-pointer",
                    onclick: move |_| selected_node_id.set(Some(nid.clone())),
                    if is_sel {
                      rect {
                        x: "{rx_x - 3.5:.1}",
                        y: "{rx_y - 3.5:.1}",
                        width: "{NODE_W + 7.0:.1}",
                        height: "{NODE_H + 7.0:.1}",
                        rx: "8",
                        fill: "none",
                        class: "stroke-indigo-600 dark:stroke-amber-400",
                        stroke_width: "2.2",
                      }
                    }
                    rect {
                      x: "{rx_x:.1}",
                      y: "{rx_y:.1}",
                      width: "{NODE_W:.1}",
                      height: "{NODE_H:.1}",
                      rx: "5",
                      fill: "{fill}",
                      stroke: "{stroke}",
                      stroke_width: "1.2",
                    }
                    text {
                      x: "{cx:.1}",
                      y: "{text_y:.1}",
                      text_anchor: "middle",
                      font_size: "11.5",
                      font_weight: "700",
                      fill: "{text_col}",
                      "{shape_txt}"
                    }
                    if !note_txt.is_empty() {
                      text {
                        x: "{note_x:.1}",
                        y: "{text_y:.1}",
                        font_size: "10.5",
                        class: "fill-slate-700 dark:fill-slate-300 font-semibold",
                        "{note_txt}"
                      }
                    }
                  }
                }
              }
            }
          }

          // 3E. 右上角算子色块图例表（严格复刻表格图例直观风格，且仅列当前模型实际用到的算子）
          g {
            rect {
              x: "{layout.legend_x:.1}",
              y: "{layout.legend_y:.1}",
              width: "{LEGEND_W:.1}",
              height: "{legend_h:.1}",
              rx: "6",
              class: "fill-white/95 dark:fill-slate-900/95 stroke-slate-300 dark:stroke-slate-700",
              stroke_width: "1.2",
            }
            text {
              x: "{layout.legend_x + 12.0:.1}",
              y: "{layout.legend_y + 20.0:.1}",
              font_size: "11",
              class: "fill-slate-900 dark:fill-slate-100 font-bold",
              "算子色块图例 (Legend)"
            }
            for (idx, (kind, lbl)) in legend.iter().enumerate() {
              {
                let row_y = layout.legend_y + 30.0 + (idx as f64) * 28.0;
                let (fill, stroke, _) = kind.palette();
                rsx! {
                  g { key: "leg-{idx}-{lbl}",
                    line {
                      x1: "{layout.legend_x:.1}",
                      y1: "{row_y:.1}",
                      x2: "{layout.legend_x + LEGEND_W:.1}",
                      y2: "{row_y:.1}",
                      class: "stroke-slate-200 dark:stroke-slate-800",
                      stroke_width: "1",
                    }
                    rect {
                      x: "{layout.legend_x + 10.0:.1}",
                      y: "{row_y + 5.0:.1}",
                      width: "48",
                      height: "18",
                      rx: "3",
                      fill: "{fill}",
                      stroke: "{stroke}",
                      stroke_width: "1",
                    }
                    text {
                      x: "{layout.legend_x + 68.0:.1}",
                      y: "{row_y + 18.0:.1}",
                      font_size: "11",
                      class: "fill-slate-800 dark:fill-slate-200 font-semibold",
                      "{lbl}"
                    }
                  }
                }
              }
            }
          }
        }
      }

      // 4. 选中节点工程详情栏（支持一键跳回分层工作台查看对应算子子图与权重张量）
      if let Some(node) = active_node {
        {
          let (fill, _, text_col) = node.kind.palette();
          let badge_style = format!("background-color:{fill};color:{text_col};");
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
                  span { class: "text-[13px] font-bold text-slate-900 dark:text-white",
                    "{node.op_title}"
                  }
                  span { class: "px-2 py-0.5 rounded bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 text-xs font-mono font-bold text-indigo-600 dark:text-indigo-400",
                    "{node.shape}"
                  }
                  if !node.side_note.is_empty() {
                    span { class: "text-xs font-mono text-slate-500 dark:text-slate-400",
                      "({node.side_note})"
                    }
                  }
                }
                div { class: "text-[13px] font-mono text-slate-700 dark:text-slate-300",
                  "{node.formula}"
                }
                if !node.params_hint.is_empty() {
                  div { class: "text-xs text-slate-500 dark:text-slate-400",
                    "{node.params_hint}"
                  }
                }
              }

              if let Some(target_focus) = drill_focus {
                button {
                  r#type: "button",
                  class: "shrink-0 px-3 py-1.5 rounded-lg text-xs font-semibold bg-indigo-600 text-white hover:bg-indigo-700 transition-colors cursor-pointer",
                  onclick: move |_| on_drill_to_workbench.call((drill_layer, target_focus)),
                  "切回分层工作台查看该算子 →"
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

        let ids: HashSet<&str> = graph.nodes.iter().map(|n| n.id.as_str()).collect();
        for e in &graph.edges {
          assert!(ids.contains(e.from_id.as_str()), "{slug}: missing edge from {}", e.from_id);
          assert!(ids.contains(e.to_id.as_str()), "{slug}: missing edge to {}", e.to_id);
        }

        let svg = export_standalone_svg(&graph);
        assert!(svg.starts_with("<svg "));
        assert!(svg.contains("算子色块图例 (Legend)"));
      }
    }

    // 验证 DeepSeek-V3 异构双段（Dense + MoE）与数值代入
    let ds = load_builtin_preset("deepseek-v3").expect("deepseek-v3");
    let ds_g = build_blueprint_tensor_graph(&ds, ShapePreset::Prefill512, false);
    assert_eq!(ds_g.groups.len(), 2);
    assert!(ds_g.nodes.iter().any(|n| n.shape.contains("1×512×7168")));
    assert!(ds_g.nodes.iter().any(|n| n.shape.contains("1×512×576")));
    assert!(ds_g.nodes.iter().any(|n| n.side_note.contains("Top-8/256")));

    // 验证经典 Inception-v3 包含截图里的全部 8 类图例与 4 分支拼接 35*35*256
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

    // 验证 ResNet-50 瓶颈残差拓扑
    let res = build_resnet50_graph(ShapePreset::Symbolic);
    assert!(res.nodes.iter().any(|n| n.shape == "224*224*3"));
    assert!(res.nodes.iter().any(|n| n.shape == "56*56*256"));
  }
}
