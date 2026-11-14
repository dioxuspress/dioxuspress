//! LLM 模型架构推导引擎（L1 `config.json` + L2 `SafetensorsSummary` 融合建模）。
//!
//! 核心职责：
//! 1. 解析 HuggingFace / ModelScope 的 `config.json`（兼容顶层配置及 VLM 的 `text_config` 内嵌结构）、
//!    可选 `generation_config.json` 与 `tokenizer_config.json`；
//! 2. 结合 L2 Safetensors 真实权重元数据进行交叉核验（如 QK-Norm、QKV Bias、真实存储 Dtype）；
//! 3. 精确推导 **Llama (Dense GQA)**、**Qwen2/3 (Dense & Sparse MoE + QK-Norm)**、
//!    **DeepSeek-V2/V3 (MLA 潜空间压缩注意力 + 异构 Dense/MoE 层栈 + MTP 多 Token 预测头)**
//!    的每一层 Tensor Shape、总参数量 vs 单 Token 激活参数量、以及推理显存与 KV Cache 压缩比。

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::safetensors_meta::{SafetensorsSummary, TensorPatternEntry};

/// 识别出的模型架构家族。
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ArchFamily {
  /// Llama / Mistral 等标准 Pre-Norm + RoPE + GQA + SwiGLU 架构
  LlamaDense,
  /// Llama-4 iRoPE（RoPE/NoPE 交替）+ 共享专家稀疏 MoE（含多模态视觉融合）
  Llama4Moe,
  /// Qwen2 / Qwen2.5 / Qwen3 稠密架构（含 QK-Norm 或 QKV Bias 细节）
  QwenDense,
  /// Qwen2-MoE / Qwen3-MoE 稀疏混合专家架构
  QwenMoe,
  /// Mixtral 经典 Top-K 稀疏混合专家架构（无共享专家）
  MixtralMoe,
  /// DeepSeek-V2 / V3 / R1：MLA 多头潜在注意力 + 混合 Dense/DeepSeekMoE (+ MTP)
  DeepSeekMla,
  /// Gemma-2 / Gemma-3：Local SWA + Global 全注意力交替层栈 + Pre/Post 四重 RMSNorm + GeGLU
  GemmaHybrid,
  /// GLM-4 系列：高压缩比 MQA/GQA + Partial RoPE + 融合 gate_up_proj FFN
  GlmGqa,
  /// 其它标准 Decoder-Only Transformer
  GenericTransformer,
  /// 未提供标准 Transformer 超参、仅由 Safetensors 权重树还原的自定义模型
  SafetensorsTree,
}

impl ArchFamily {
  pub fn badge_label(&self) -> &'static str {
    match self {
      Self::LlamaDense => "Llama / GQA Dense",
      Self::Llama4Moe => "Llama-4 iRoPE + MoE",
      Self::QwenDense => "Qwen Dense (GQA)",
      Self::QwenMoe => "Qwen Sparse MoE",
      Self::MixtralMoe => "Mixtral Sparse MoE",
      Self::DeepSeekMla => "DeepSeek MLA + MoE",
      Self::GemmaHybrid => "Gemma Hybrid SWA",
      Self::GlmGqa => "GLM Extreme GQA",
      Self::GenericTransformer => "Transformer Decoder",
      Self::SafetensorsTree => "Safetensors Weight Tree",
    }
  }
}

/// 多模态视觉编码器（Vision Tower + Projector）规格。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct VisionSpec {
  pub model_type: String,
  pub hidden_size: u64,
  pub num_hidden_layers: u64,
  pub num_attention_heads: u64,
  pub intermediate_size: u64,
  pub image_size: Option<u64>,
  pub patch_size: Option<u64>,
  pub estimated_params: u64,
}

/// 注意力子层精确规格。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AttentionSpec {
  /// 标准 MHA / MQA / GQA（含可选每头 QK-Norm、滑动窗口 SWA 及交替层调度）
  StandardGqa {
    num_q_heads: u64,
    num_kv_heads: u64,
    head_dim: u64,
    qkv_bias: bool,
    o_bias: bool,
    qk_norm: bool,
    sliding_window: Option<u64>,
    /// 如 Gemma-3 的 `6`：每 6 层中前 5 层用 `sliding_window` 局部注意力，第 6 层用全局全注意力
    #[serde(default)]
    sliding_window_pattern: Option<u64>,
    /// 如 Llama-4 的 `4`（iRoPE）：每 4 层中 3 层启用 RoPE，第 4 层为无位置编码 NoPE 全局层
    #[serde(default)]
    nope_layer_interval: Option<u64>,
  },
  /// DeepSeek Multi-head Latent Attention (MLA)：低秩联合压缩 KV + 解耦 RoPE
  DeepSeekMla {
    num_heads: u64,
    /// `None` 表示未对 Query 做低秩压缩（例如 DeepSeek-V2-Lite 直接使用 `q_proj`）
    q_lora_rank: Option<u64>,
    kv_lora_rank: u64,
    qk_nope_head_dim: u64,
    qk_rope_head_dim: u64,
    v_head_dim: u64,
  },
}

impl AttentionSpec {
  /// 简短机制名称（用于架构图节点副标题）。
  pub fn short_label(&self) -> String {
    match self {
      Self::StandardGqa {
        num_q_heads,
        num_kv_heads,
        head_dim,
        qk_norm,
        sliding_window,
        sliding_window_pattern,
        nope_layer_interval,
        ..
      } => {
        let kind = if *num_q_heads == *num_kv_heads {
          "MHA"
        } else if *num_kv_heads == 1 {
          "MQA"
        } else {
          "GQA"
        };
        let norm_tag = if *qk_norm { " + QK-Norm" } else { "" };
        let extra = if let (Some(w), Some(pat)) = (sliding_window, sliding_window_pattern) {
          format!(" · {}:1 SWA({w})", pat.saturating_sub(1))
        } else if let Some(iv) = nope_layer_interval {
          format!(" · iRoPE({}:1)", iv.saturating_sub(1))
        } else if let Some(w) = sliding_window {
          format!(" · SWA({w})")
        } else {
          String::new()
        };
        format!("{kind} ({num_q_heads}Q:{num_kv_heads}KV × {head_dim}d{norm_tag}{extra})")
      }
      Self::DeepSeekMla { num_heads, kv_lora_rank, qk_rope_head_dim, .. } => {
        format!("MLA ({num_heads}H, c_KV={kv_lora_rank}d + k_R={qk_rope_head_dim}d)")
      }
    }
  }

  /// 单层单个 Token 在推理 KV Cache 中实际需要缓存的标量元素个数。
  pub fn cached_elements_per_token(&self) -> u64 {
    match self {
      Self::StandardGqa { num_kv_heads, head_dim, .. } => 2 * num_kv_heads * head_dim,
      Self::DeepSeekMla { kv_lora_rank, qk_rope_head_dim, .. } => kv_lora_rank + qk_rope_head_dim,
    }
  }

  /// 等效同头数标准 MHA 单层单 Token 需要的 KV 元素数（用于计算 KV 压缩倍率）。
  pub fn mha_baseline_elements_per_token(&self) -> u64 {
    match self {
      Self::StandardGqa { num_q_heads, head_dim, .. } => 2 * num_q_heads * head_dim,
      Self::DeepSeekMla { num_heads, v_head_dim, .. } => 2 * num_heads * v_head_dim,
    }
  }

  /// 相比标准 MHA 的 KV Cache 压缩倍数（如 32Q:8KV 为 4.0×，DeepSeek-V3 MLA 为 56.89×）。
  pub fn kv_compression_ratio(&self) -> f64 {
    let actual = self.cached_elements_per_token();
    if actual == 0 {
      return 1.0;
    }
    (self.mha_baseline_elements_per_token() as f64) / (actual as f64)
  }

  /// 单层 Attention 子块的参数量（全部激活）。
  pub fn params_per_layer(&self, hidden_size: u64) -> u64 {
    match self {
      Self::StandardGqa {
        num_q_heads, num_kv_heads, head_dim, qkv_bias, o_bias, qk_norm, ..
      } => {
        let q_out = num_q_heads * head_dim;
        let kv_out = num_kv_heads * head_dim;
        let mut p =
          hidden_size * q_out + hidden_size * kv_out + hidden_size * kv_out + q_out * hidden_size;
        if *qkv_bias {
          p += q_out + 2 * kv_out;
        }
        if *o_bias {
          p += hidden_size;
        }
        if *qk_norm {
          p += 2 * head_dim;
        }
        p
      }
      Self::DeepSeekMla {
        num_heads,
        q_lora_rank,
        kv_lora_rank,
        qk_nope_head_dim,
        qk_rope_head_dim,
        v_head_dim,
      } => {
        let qk_head_dim = qk_nope_head_dim + qk_rope_head_dim;
        let q_params = match q_lora_rank {
          Some(q_rank) => hidden_size * q_rank + q_rank + q_rank * (num_heads * qk_head_dim),
          None => hidden_size * (num_heads * qk_head_dim),
        };
        let kv_a = hidden_size * (kv_lora_rank + qk_rope_head_dim) + kv_lora_rank;
        let kv_b = kv_lora_rank * (num_heads * (qk_nope_head_dim + v_head_dim));
        let o_proj = (num_heads * v_head_dim) * hidden_size;
        q_params + kv_a + kv_b + o_proj
      }
    }
  }
}

/// 前馈网络（Dense FFN 或 Sparse MoE）子层精确规格。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum FfnSpec {
  /// 稠密门控前馈网络（SwiGLU / GeGLU：支持标准三矩阵或 GLM-4 融合 `gate_up_proj`）
  DenseGated {
    intermediate_size: u64,
    act_fn: String,
    bias: bool,
    #[serde(default)]
    fused_gate_up: bool,
  },
  /// 稀疏混合专家（支持 Routed Top-K Experts + 可选常驻 Shared Experts + 分组路由）
  SparseMoe {
    num_routed_experts: u64,
    top_k: u64,
    moe_intermediate_size: u64,
    num_shared_experts: u64,
    shared_intermediate_size: u64,
    has_shared_gate: bool,
    /// `(n_group, topk_group)`，例如 DeepSeek-V3 的 `(8, 4)` 节点受限分组路由
    router_groups: Option<(u64, u64)>,
    scoring_func: String,
    act_fn: String,
  },
}

impl FfnSpec {
  pub fn short_label(&self) -> String {
    match self {
      Self::DenseGated { intermediate_size, act_fn, fused_gate_up, .. } => {
        if *fused_gate_up {
          format!("Fused GateUp FFN ({act_fn}, I={intermediate_size})")
        } else {
          format!("Dense FFN ({act_fn}, I={intermediate_size})")
        }
      }
      Self::SparseMoe {
        num_routed_experts,
        top_k,
        num_shared_experts,
        moe_intermediate_size,
        ..
      } => {
        if *num_shared_experts > 0 {
          format!(
            "MoE ({num_shared_experts} Shared + Top-{top_k}/{num_routed_experts} Routed, I={moe_intermediate_size})"
          )
        } else {
          format!("MoE (Top-{top_k}/{num_routed_experts} Routed, I={moe_intermediate_size})")
        }
      }
    }
  }

  /// 单层 FFN/MoE 的总参数量（含未激活专家）与单 Token 激活参数量。
  pub fn params_per_layer(&self, hidden_size: u64) -> (u64, u64) {
    match self {
      Self::DenseGated { intermediate_size, bias, .. } => {
        let mut p = 3 * hidden_size * intermediate_size;
        if *bias {
          p += 2 * intermediate_size + hidden_size;
        }
        (p, p)
      }
      Self::SparseMoe {
        num_routed_experts,
        top_k,
        moe_intermediate_size,
        num_shared_experts,
        shared_intermediate_size,
        has_shared_gate,
        ..
      } => {
        let router = hidden_size * num_routed_experts;
        let one_routed = 3 * hidden_size * moe_intermediate_size;
        let routed_total = num_routed_experts * one_routed;
        let routed_active = (*top_k).min(*num_routed_experts) * one_routed;
        let shared = if *num_shared_experts > 0 && *shared_intermediate_size > 0 {
          3 * hidden_size * shared_intermediate_size
            + if *has_shared_gate { hidden_size } else { 0 }
        } else {
          0
        };
        let total = router + routed_total + shared;
        let active = router + routed_active + shared;
        (total, active)
      }
    }
  }
}

fn default_norms_per_layer() -> u64 {
  2
}

/// 连续同构（或周期性交替）的 Transformer 层区间（精确表达如 DeepSeek-V3 的 `Layers 0..2 Dense` + `Layers 3..60 MoE`）。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LayerSegment {
  /// 起始层号（含，0-based）
  pub start_layer: u64,
  /// 结束层号（含，0-based）
  pub end_layer: u64,
  /// 该段标题，如 `"Dense Pre-fill Stack"` 或 `"Sparse MoE Trunk"`
  pub title: String,
  pub attention: AttentionSpec,
  pub ffn: FfnSpec,
  /// 每层 RMSNorm 数量：标准 Pre-Norm 为 `2`，Gemma-2/3 与 GLM-4（Pre + Post 双重归一化）为 `4`
  #[serde(default = "default_norms_per_layer")]
  pub norms_per_layer: u64,
  /// 周期性混合层调度说明（如 Gemma-3 的 `5:1 Local SWA / Global` 或 Llama-4 的 `3:1 iRoPE`）
  #[serde(default)]
  pub schedule_note: Option<String>,
}

impl LayerSegment {
  pub fn layer_count(&self) -> u64 {
    self.end_layer.saturating_sub(self.start_layer) + 1
  }

  pub fn range_label(&self) -> String {
    if self.start_layer == self.end_layer {
      format!("Layer {}", self.start_layer)
    } else {
      format!("Layers {}–{} (×{})", self.start_layer, self.end_layer, self.layer_count())
    }
  }
}

/// 参数量分项统计（单位：参数个数）。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct ParamBreakdown {
  #[serde(default)]
  pub vision_encoder: u64,
  pub embedding: u64,
  pub attention: u64,
  pub dense_ffn: u64,
  pub moe_routed: u64,
  pub moe_shared: u64,
  pub norms_and_routers: u64,
  pub lm_head: u64,
  pub mtp_head: u64,
  /// 模型总参数量（主干 + Embedding + LM Head + 多模态视觉编码器，不含训练辅助 MTP 时可对照 `mtp_head`）
  pub total_params: u64,
  /// 每个 Token 前向推理实际激活的参数量
  pub active_params: u64,
}

/// 架构图中可点击检视的模块焦点。
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum InspectorFocus {
  Vision,
  Embedding,
  #[default]
  Attention,
  FfnOrMoe,
  NormAndResidual,
  LmHead,
  MtpHead,
}

impl InspectorFocus {
  pub fn from_attr(raw: &str) -> Self {
    match raw.trim().to_ascii_lowercase().as_str() {
      "vision" | "vit" | "siglip" => Self::Vision,
      "embed" | "embedding" => Self::Embedding,
      "attn" | "attention" | "mla" | "gqa" => Self::Attention,
      "ffn" | "mlp" | "moe" | "experts" => Self::FfnOrMoe,
      "norm" | "rmsnorm" | "residual" => Self::NormAndResidual,
      "head" | "lm_head" | "logits" => Self::LmHead,
      "mtp" => Self::MtpHead,
      _ => Self::Attention,
    }
  }

  pub fn stage_filter(&self) -> &'static str {
    match self {
      Self::Vision => "vision",
      Self::Embedding => "embedding",
      Self::Attention => "attention",
      Self::FfnOrMoe => "ffn_or_moe",
      Self::NormAndResidual => "norm",
      Self::LmHead => "lm_head",
      Self::MtpHead => "mtp",
    }
  }
}

/// 完整的模型架构可视化数据包（SSR / Server Function / 客户端自定义解析共用）。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelBlueprint {
  pub repo_id: String,
  pub source_label: String,
  pub family: ArchFamily,
  pub architectures: Vec<String>,
  pub model_type: String,
  pub torch_dtype: String,
  pub quantization_summary: Option<String>,
  /// 若外层为多模态 VLM（`vision_config` 存在），记录视觉编码器摘要提示与精确规格
  pub multimodal_note: Option<String>,
  #[serde(default)]
  pub vision: Option<VisionSpec>,

  pub vocab_size: u64,
  pub hidden_size: u64,
  pub num_hidden_layers: u64,
  pub max_position_embeddings: u64,
  pub rms_norm_eps: Option<f64>,
  pub tie_word_embeddings: bool,
  pub rope_summary: String,
  pub hidden_act: String,
  /// DeepSeek-V3 等模型的 Multi-Token Prediction 预测层数
  pub num_nextn_predict_layers: u64,

  /// 分词器与生成超参补充信息（来自 `tokenizer_config.json` / `generation_config.json`）
  pub tokenizer_class: Option<String>,
  pub default_temperature: Option<f64>,
  pub default_top_p: Option<f64>,

  /// 纵向异构/同构层区间列表
  pub segments: Vec<LayerSegment>,
  /// 参数量分项核算
  pub params: ParamBreakdown,
  /// L2 Safetensors 真实权重元数据（若获取到）
  pub safetensors: Option<SafetensorsSummary>,
  /// 格式化后的原始 `config.json` 片段（供「原始配置」标签页对照审阅）
  pub raw_config_pretty: String,
}

impl ModelBlueprint {
  /// 获取指定层号 `layer_idx` 对应的层段（超出范围时回退首段或末段）。
  pub fn segment_for_layer(&self, layer_idx: u64) -> Option<&LayerSegment> {
    self
      .segments
      .iter()
      .find(|s| layer_idx >= s.start_layer && layer_idx <= s.end_layer)
      .or_else(|| self.segments.last())
  }

  /// 是否包含稀疏 MoE 层。
  pub fn has_moe(&self) -> bool {
    self.segments.iter().any(|s| matches!(s.ffn, FfnSpec::SparseMoe { .. }))
  }

  /// 默认推荐高亮聚焦的子模块：DeepSeek 聚焦 MLA，纯 MoE 聚焦 MoE，其余聚焦 Attention。
  pub fn default_focus(&self) -> InspectorFocus {
    match self.family {
      ArchFamily::QwenMoe | ArchFamily::MixtralMoe | ArchFamily::Llama4Moe => {
        InspectorFocus::FfnOrMoe
      }
      _ => InspectorFocus::Attention,
    }
  }

  /// 按当前选中的模块焦点筛选匹配的 L2 Safetensors 真实权重模式。
  pub fn matching_tensors(&self, focus: InspectorFocus) -> Vec<&TensorPatternEntry> {
    let Some(ref st) = self.safetensors else {
      return Vec::new();
    };
    let target = focus.stage_filter();
    st.patterns
      .iter()
      .filter(|p| {
        if target == "ffn_or_moe" {
          p.stage == "ffn" || p.stage == "moe"
        } else {
          p.stage == target
        }
      })
      .collect()
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 推理显存（VRAM）与 KV Cache 精算器
// ─────────────────────────────────────────────────────────────────────────────

/// 推理部署量化精度预设。
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum PrecisionPreset {
  /// BF16 / FP16（权重 16-bit，KV Cache 16-bit）
  Bf16,
  /// FP8 E4M3（权重 8-bit 块量化，KV Cache 8-bit FP8）
  Fp8,
  /// INT8 W8A16（权重 8-bit，KV Cache 16-bit）
  Int8,
  /// INT4 AWQ / GPTQ（权重 4-bit 分组量化，KV Cache 16-bit）
  Int4,
}

impl PrecisionPreset {
  pub const ALL: [PrecisionPreset; 4] = [Self::Bf16, Self::Fp8, Self::Int8, Self::Int4];

  pub fn label(&self) -> &'static str {
    match self {
      Self::Bf16 => "BF16 / FP16",
      Self::Fp8 => "FP8 (W8KV8)",
      Self::Int8 => "INT8 (W8A16)",
      Self::Int4 => "INT4 (AWQ/GPTQ)",
    }
  }

  /// 每个模型参数的平均有效字节数（计入量化 group=128 scale 开销与未量化的 Norm 层）。
  pub fn bytes_per_weight(&self) -> f64 {
    match self {
      Self::Bf16 => 2.0,
      Self::Fp8 => 1.03,
      Self::Int8 => 1.06,
      Self::Int4 => 0.56,
    }
  }

  /// 每个 KV Cache 标量元素的字节数。
  pub fn bytes_per_kv_elem(&self) -> f64 {
    match self {
      Self::Fp8 => 1.0,
      Self::Bf16 | Self::Int8 | Self::Int4 => 2.0,
    }
  }
}

/// 给定精度、上下文长度与并发 Batch 下的显存精算结果（单位：字节）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VramEstimate {
  pub weights_bytes: f64,
  pub kv_cache_bytes: f64,
  pub mha_baseline_kv_bytes: f64,
  pub activation_overhead_bytes: f64,
  pub total_bytes: f64,
}

impl ModelBlueprint {
  /// 计算指定部署配置下的显存占用。
  pub fn estimate_vram(
    &self,
    precision: PrecisionPreset,
    context_len: u64,
    batch_size: u64,
  ) -> VramEstimate {
    let b = batch_size.max(1) as f64;
    let weights_bytes = (self.params.total_params as f64) * precision.bytes_per_weight();
    let kv_elem_bytes = precision.bytes_per_kv_elem();

    let mut kv_elems_total: f64 = 0.0;
    let mut mha_elems_total: f64 = 0.0;

    for seg in &self.segments {
      let n_layers = seg.layer_count();
      let elems_per_tok = seg.attention.cached_elements_per_token() as f64;
      match seg.attention {
        AttentionSpec::StandardGqa {
          sliding_window: Some(w),
          sliding_window_pattern: Some(pat),
          ..
        } if w > 0 && pat > 1 => {
          let global_layers = (n_layers / pat) as f64;
          let local_layers = (n_layers.saturating_sub(n_layers / pat)) as f64;
          let local_ctx = context_len.min(w) as f64;
          kv_elems_total +=
            b * elems_per_tok * (global_layers * (context_len as f64) + local_layers * local_ctx);
        }
        AttentionSpec::StandardGqa { sliding_window: Some(w), .. } if w > 0 => {
          kv_elems_total += (n_layers as f64) * b * (context_len.min(w) as f64) * elems_per_tok;
        }
        _ => {
          kv_elems_total += (n_layers as f64) * b * (context_len as f64) * elems_per_tok;
        }
      }
      mha_elems_total += (n_layers as f64)
        * b
        * (context_len as f64)
        * (seg.attention.mha_baseline_elements_per_token() as f64);
    }

    let kv_cache_bytes = kv_elems_total * kv_elem_bytes;
    let mha_baseline_kv_bytes = mha_elems_total * kv_elem_bytes;

    // 运行时激活缓冲区与 CUDA Graph / Workspace 预留：与 hidden_size、batch 及单步序列工作区成正比
    let active_workspace = (self.hidden_size as f64) * b * (context_len.min(4096) as f64) * 4.0
      + 512.0 * 1024.0 * 1024.0;
    let activation_overhead_bytes = active_workspace.max(weights_bytes * 0.04);

    VramEstimate {
      weights_bytes,
      kv_cache_bytes,
      mha_baseline_kv_bytes,
      activation_overhead_bytes,
      total_bytes: weights_bytes + kv_cache_bytes + activation_overhead_bytes,
    }
  }
}

/// 推荐单机/多卡 GPU 规格文案。
pub fn recommend_gpu_tier(total_bytes: f64) -> &'static str {
  let gb = total_bytes / 1_073_741_824.0;
  if gb <= 7.0 {
    "8 GB 级（RTX 4060 / Apple M 系列 16GB）"
  } else if gb <= 14.5 {
    "16 GB 级（RTX 4080 / Mac 24GB 统一内存）"
  } else if gb <= 22.5 {
    "24 GB 级（1× RTX 3090 / 4090 / L4）"
  } else if gb <= 45.0 {
    "48 GB 级（1× L40S / RTX 6000 Ada 或 2× 4090）"
  } else if gb <= 76.0 {
    "80 GB 级（1× A100 / H100 80GB）"
  } else if gb <= 152.0 {
    "160 GB 级（2× H100 80GB / 1× H200 141GB）"
  } else if gb <= 310.0 {
    "320 GB 级（4× A100 / H100 80GB）"
  } else if gb <= 620.0 {
    "640 GB 级（8× H100 80GB 整机节点）"
  } else {
    "多机集群（2+ 节点 × 8× H100/H200 RDMA）"
  }
}

/// 人类可读参数量格式化（如 `671.03B`、`8.03B`、`396.4M`）。
pub fn format_count(n: u64) -> String {
  if n >= 1_000_000_000 {
    format!("{:.2}B", (n as f64) / 1_000_000_000.0)
  } else if n >= 1_000_000 {
    format!("{:.1}M", (n as f64) / 1_000_000.0)
  } else if n >= 1_000 {
    format!("{:.1}K", (n as f64) / 1_000.0)
  } else {
    n.to_string()
  }
}

/// 人类可读字节体积格式化（GiB / MiB / KiB）。
pub fn format_bytes_iec(bytes: f64) -> String {
  const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
  const MIB: f64 = 1024.0 * 1024.0;
  if bytes >= GIB {
    format!("{:.2} GiB", bytes / GIB)
  } else if bytes >= MIB {
    format!("{:.1} MiB", bytes / MIB)
  } else {
    format!("{:.0} KiB", (bytes / 1024.0).max(1.0))
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 核心配置解析逻辑
// ─────────────────────────────────────────────────────────────────────────────

/// 从原始 `config.json`（及可选附属配置、Safetensors 摘要）构建完整的 [`ModelBlueprint`]。
pub fn build_model_blueprint(
  repo_id: &str,
  source_label: &str,
  config_json: &str,
  generation_json: Option<&str>,
  tokenizer_json: Option<&str>,
  safetensors: Option<SafetensorsSummary>,
) -> Result<ModelBlueprint, String> {
  let root: Value =
    serde_json::from_str(config_json).map_err(|e| format!("解析 config.json 失败: {e}"))?;
  let Some(root_obj) = root.as_object() else {
    return Err("config.json 根节点必须是 JSON 对象".to_string());
  };

  let raw_config_pretty =
    serde_json::to_string_pretty(&root).unwrap_or_else(|_| config_json.to_string());

  // 兼容多模态 VLM 外壳：若顶层没有 `hidden_size`，自动下钻 `text_config` 或 `llm_config`
  let mut multimodal_note: Option<String> = None;
  let vision_cfg_obj = root_obj.get("vision_config").and_then(Value::as_object);
  let cfg_obj = if root_obj.contains_key("hidden_size") || root_obj.contains_key("n_embd") {
    if let Some(vc) = vision_cfg_obj {
      let vt = vc.get("model_type").and_then(Value::as_str).unwrap_or("vision_tower");
      let vd = vc.get("hidden_size").and_then(Value::as_u64).unwrap_or(0);
      multimodal_note =
        Some(format!("内含多模态视觉编码器 ({vt}, hidden={vd})，支持点击展开视觉支路"));
    }
    root_obj
  } else if let Some(sub) = root_obj
    .get("text_config")
    .or_else(|| root_obj.get("llm_config"))
    .or_else(|| root_obj.get("language_config"))
    .and_then(Value::as_object)
  {
    let vision_tag = vision_cfg_obj
      .and_then(|v| v.get("model_type"))
      .and_then(Value::as_str)
      .unwrap_or("Vision Encoder");
    multimodal_note = Some(format!(
      "多模态复合架构（含 {vision_tag} 视觉编码器 + 语言主干），可点击视觉支路展开细节"
    ));
    sub
  } else {
    root_obj
  };

  let architectures: Vec<String> = root_obj
    .get("architectures")
    .and_then(Value::as_array)
    .map(|arr| arr.iter().filter_map(Value::as_str).map(str::to_string).collect())
    .unwrap_or_default();

  let model_type = cfg_obj
    .get("model_type")
    .or_else(|| root_obj.get("model_type"))
    .and_then(Value::as_str)
    .unwrap_or("unknown")
    .to_string();

  let torch_dtype = cfg_obj
    .get("torch_dtype")
    .or_else(|| root_obj.get("torch_dtype"))
    .or_else(|| cfg_obj.get("dtype"))
    .and_then(Value::as_str)
    .unwrap_or("bfloat16")
    .to_string();

  let quantization_summary =
    extract_quantization_summary(root_obj).or_else(|| extract_quantization_summary(cfg_obj));

  let (tokenizer_class, tok_vocab) = parse_tokenizer_meta(tokenizer_json);
  let (default_temperature, default_top_p) = parse_generation_meta(generation_json);

  let hidden_opt = u64_any(cfg_obj, &["hidden_size", "n_embd", "d_model"]);
  let layers_opt = u64_any(cfg_obj, &["num_hidden_layers", "n_layer", "num_layers"]);

  // 若既无标准 Transformer 维度、但存在 Safetensors 真实权重树，则走纯 L2 兜底建模
  let (Some(hidden_size), Some(num_hidden_layers)) = (hidden_opt, layers_opt) else {
    if let Some(st) = safetensors {
      return Ok(build_safetensors_fallback_blueprint(
        repo_id,
        source_label,
        architectures,
        model_type,
        torch_dtype,
        quantization_summary,
        st,
        raw_config_pretty,
      ));
    }
    return Err(
      "config.json 中缺少 hidden_size / num_hidden_layers，且未获取到 Safetensors 权重索引"
        .to_string(),
    );
  };

  if hidden_size == 0 || num_hidden_layers == 0 {
    return Err("hidden_size 与 num_hidden_layers 必须大于 0".to_string());
  }

  let vision = vision_cfg_obj.and_then(|vc| parse_vision_spec(vc, hidden_size));

  let arch_joined = format!("{} {}", architectures.join(" "), model_type).to_ascii_lowercase();
  let is_llama4 = arch_joined.contains("llama4");
  let is_gemma_any = arch_joined.contains("gemma");
  let is_gemma2_or_3 = arch_joined.contains("gemma2") || arch_joined.contains("gemma3");
  let is_glm = arch_joined.contains("glm") || arch_joined.contains("chatglm");
  let is_mixtral = arch_joined.contains("mixtral");

  let vocab_size =
    u64_any(cfg_obj, &["padded_vocab_size", "vocab_size"]).or(tok_vocab).unwrap_or(32000);
  let max_position_embeddings =
    u64_any(cfg_obj, &["max_position_embeddings", "n_positions", "seq_length"]).unwrap_or(4096);
  let rms_norm_eps = f64_any(
    cfg_obj,
    &["rms_norm_eps", "layernorm_epsilon", "layer_norm_eps", "layer_norm_epsilon"],
  );
  let tie_word_embeddings = cfg_obj
    .get("tie_word_embeddings")
    .or_else(|| root_obj.get("tie_word_embeddings"))
    .and_then(Value::as_bool)
    .unwrap_or(is_gemma_any);
  let hidden_act = cfg_obj
    .get("hidden_act")
    .or_else(|| cfg_obj.get("hidden_activation"))
    .or_else(|| cfg_obj.get("activation_function"))
    .and_then(Value::as_str)
    .unwrap_or("silu")
    .to_string();
  let rope_summary = extract_rope_summary(cfg_obj);
  let num_nextn_predict_layers = u64_any(cfg_obj, &["num_nextn_predict_layers"]).unwrap_or(0);

  let num_q_heads = u64_any(cfg_obj, &["num_attention_heads", "n_head"]).unwrap_or(32).max(1);
  let num_kv_heads =
    u64_any(cfg_obj, &["num_key_value_heads", "num_kv_heads", "multi_query_group_num"])
      .unwrap_or(num_q_heads)
      .max(1);

  // 1. 识别 Attention 形态（MLA vs Standard GQA）
  let kv_lora_rank = u64_any(cfg_obj, &["kv_lora_rank"]);
  let is_mla = kv_lora_rank.is_some()
    || arch_joined.contains("deepseek_v2")
    || arch_joined.contains("deepseek_v3");

  let attention = if let Some(kv_rank) = kv_lora_rank {
    let q_lora_rank = u64_any(cfg_obj, &["q_lora_rank"]).filter(|&v| v > 0);
    let qk_nope_head_dim = u64_any(cfg_obj, &["qk_nope_head_dim"]).unwrap_or(128);
    let qk_rope_head_dim = u64_any(cfg_obj, &["qk_rope_head_dim"]).unwrap_or(64);
    let v_head_dim = u64_any(cfg_obj, &["v_head_dim"]).unwrap_or(128);
    AttentionSpec::DeepSeekMla {
      num_heads: num_q_heads,
      q_lora_rank,
      kv_lora_rank: kv_rank,
      qk_nope_head_dim,
      qk_rope_head_dim,
      v_head_dim,
    }
  } else {
    let head_dim =
      u64_any(cfg_obj, &["head_dim", "kv_channels"]).unwrap_or(hidden_size / num_q_heads).max(1);
    let default_qkv_bias = arch_joined.contains("qwen2") || arch_joined.contains("chatglm");
    let qkv_bias = cfg_obj
      .get("attention_bias")
      .or_else(|| cfg_obj.get("qkv_bias"))
      .or_else(|| cfg_obj.get("add_qkv_bias"))
      .and_then(Value::as_bool)
      .or_else(|| safetensors.as_ref().map(|s| s.has_attn_bias))
      .unwrap_or(default_qkv_bias);
    let o_bias = cfg_obj
      .get("attention_bias")
      .or_else(|| cfg_obj.get("add_bias_linear"))
      .and_then(Value::as_bool)
      .unwrap_or(false);
    let qk_norm = cfg_obj
      .get("qk_norm")
      .or_else(|| cfg_obj.get("use_qk_norm"))
      .and_then(Value::as_bool)
      .unwrap_or_else(|| {
        arch_joined.contains("qwen3")
          || arch_joined.contains("gemma3")
          || safetensors.as_ref().is_some_and(|s| s.has_qk_norm)
      });
    let sliding_window = cfg_obj.get("sliding_window").and_then(Value::as_u64).filter(|&w| w > 0);
    let sliding_window_pattern = u64_any(cfg_obj, &["sliding_window_pattern"])
      .or(if arch_joined.contains("gemma3") && sliding_window.is_some() { Some(6) } else { None })
      .filter(|&p| p > 1);
    let nope_layer_interval = u64_any(cfg_obj, &["nope_layer_interval"])
      .or_else(|| detect_nope_interval(cfg_obj))
      .filter(|&iv| iv > 1);

    AttentionSpec::StandardGqa {
      num_q_heads,
      num_kv_heads,
      head_dim,
      qkv_bias,
      o_bias,
      qk_norm,
      sliding_window,
      sliding_window_pattern,
      nope_layer_interval,
    }
  };

  // 2. 识别 FFN / MoE 与纵向分层区间
  let intermediate_size = u64_any(cfg_obj, &["intermediate_size", "ffn_hidden_size", "n_inner"])
    .unwrap_or(hidden_size * 4);
  let dense_mlp_inter = u64_any(cfg_obj, &["intermediate_size_mlp"]).unwrap_or(intermediate_size);
  let mlp_bias = cfg_obj
    .get("mlp_bias")
    .or_else(|| cfg_obj.get("add_bias_linear"))
    .and_then(Value::as_bool)
    .unwrap_or(false);
  let norms_per_layer = if is_gemma2_or_3 || arch_joined.contains("glm4") { 4 } else { 2 };

  let num_routed_experts =
    u64_any(cfg_obj, &["n_routed_experts", "num_experts", "num_local_experts"]).unwrap_or(0);
  let is_moe = num_routed_experts > 1;

  let family = if is_mla {
    ArchFamily::DeepSeekMla
  } else if is_llama4 && is_moe {
    ArchFamily::Llama4Moe
  } else if is_mixtral && is_moe {
    ArchFamily::MixtralMoe
  } else if is_moe {
    ArchFamily::QwenMoe
  } else if is_gemma2_or_3 {
    ArchFamily::GemmaHybrid
  } else if is_glm {
    ArchFamily::GlmGqa
  } else if arch_joined.contains("qwen") {
    ArchFamily::QwenDense
  } else if arch_joined.contains("llama")
    || arch_joined.contains("mistral")
    || arch_joined.contains("gemma")
  {
    ArchFamily::LlamaDense
  } else {
    ArchFamily::GenericTransformer
  };

  let schedule_note = build_schedule_note(&attention, cfg_obj);

  let dense_ffn = FfnSpec::DenseGated {
    intermediate_size: dense_mlp_inter,
    act_fn: hidden_act.clone(),
    bias: mlp_bias,
    fused_gate_up: is_glm,
  };

  let segments = if is_moe {
    let top_k = u64_any(cfg_obj, &["num_experts_per_tok", "num_selected_experts"]).unwrap_or(2);
    let moe_inter = u64_any(cfg_obj, &["moe_intermediate_size"]).unwrap_or(intermediate_size);
    let n_shared_cfg = u64_any(cfg_obj, &["n_shared_experts"]).unwrap_or(0);
    let shared_inter_cfg = u64_any(cfg_obj, &["shared_expert_intermediate_size"]).unwrap_or(0);
    let (num_shared_experts, shared_intermediate_size, has_shared_gate) = if n_shared_cfg > 0 {
      (n_shared_cfg, n_shared_cfg * moe_inter, false)
    } else if shared_inter_cfg > 0 {
      (1, shared_inter_cfg, arch_joined.contains("qwen2_moe") || arch_joined.contains("qwen2moe"))
    } else if is_llama4 {
      // Llama-4 (Scout / Maverick) 固定内置 1 个常驻 Shared Expert（维度与单专家 intermediate_size 相同）
      (1, moe_inter, false)
    } else {
      (0, 0, false)
    };
    let router_groups = match (u64_any(cfg_obj, &["n_group"]), u64_any(cfg_obj, &["topk_group"])) {
      (Some(ng), Some(tg)) if ng > 0 && tg > 0 => Some((ng, tg)),
      _ => None,
    };
    let scoring_func = cfg_obj
      .get("scoring_func")
      .and_then(Value::as_str)
      .unwrap_or(if is_llama4 { "sigmoid" } else { "softmax" })
      .to_string();

    let moe_ffn = FfnSpec::SparseMoe {
      num_routed_experts,
      top_k,
      moe_intermediate_size: moe_inter,
      num_shared_experts,
      shared_intermediate_size,
      has_shared_gate,
      router_groups,
      scoring_func,
      act_fn: hidden_act.clone(),
    };

    let moe_step = u64_any(cfg_obj, &["interleave_moe_layer_step"]).unwrap_or(1).max(1);
    let first_k_dense = u64_any(cfg_obj, &["first_k_dense_replace"])
      .or_else(|| safetensors.as_ref().and_then(|s| s.detected_first_moe_layer))
      .unwrap_or(0)
      .min(num_hidden_layers);

    let mut segs = Vec::new();
    if moe_step > 1 && first_k_dense == 0 {
      // 如 Llama-4 Maverick：Dense 层与 MoE 层按周期交替堆叠
      let moe_layers = num_hidden_layers / moe_step;
      let dense_layers = num_hidden_layers.saturating_sub(moe_layers);
      if dense_layers > 0 {
        segs.push(LayerSegment {
          start_layer: 0,
          end_layer: dense_layers - 1,
          title: format!("交替稠密层组 (Interleaved Dense ×{dense_layers})"),
          attention: attention.clone(),
          ffn: dense_ffn.clone(),
          norms_per_layer,
          schedule_note: Some(format!("与稀疏 MoE 按 step={moe_step} 逐层交替堆叠（偶/奇层分流）")),
        });
      }
      if moe_layers > 0 {
        segs.push(LayerSegment {
          start_layer: dense_layers,
          end_layer: num_hidden_layers - 1,
          title: format!("交替稀疏专家层组 (Interleaved MoE ×{moe_layers})"),
          attention: attention.clone(),
          ffn: moe_ffn,
          norms_per_layer,
          schedule_note: schedule_note.clone(),
        });
      }
    } else {
      if first_k_dense > 0 {
        segs.push(LayerSegment {
          start_layer: 0,
          end_layer: first_k_dense - 1,
          title: "前置稠密预热层 (Dense SwiGLU)".to_string(),
          attention: attention.clone(),
          ffn: dense_ffn.clone(),
          norms_per_layer,
          schedule_note: None,
        });
      }
      if first_k_dense < num_hidden_layers {
        segs.push(LayerSegment {
          start_layer: first_k_dense,
          end_layer: num_hidden_layers - 1,
          title: "稀疏混合专家主干 (Sparse MoE)".to_string(),
          attention: attention.clone(),
          ffn: moe_ffn,
          norms_per_layer,
          schedule_note: schedule_note.clone(),
        });
      }
    }
    segs
  } else {
    vec![LayerSegment {
      start_layer: 0,
      end_layer: num_hidden_layers - 1,
      title: "Transformer 解码器层栈 (Decoder Stack)".to_string(),
      attention,
      ffn: dense_ffn,
      norms_per_layer,
      schedule_note,
    }]
  };

  let vision_params = vision.as_ref().map(|v| v.estimated_params).unwrap_or(0);
  let params = compute_param_breakdown(
    vision_params,
    vocab_size,
    hidden_size,
    tie_word_embeddings,
    num_nextn_predict_layers,
    &segments,
  );

  Ok(ModelBlueprint {
    repo_id: repo_id.to_string(),
    source_label: source_label.to_string(),
    family,
    architectures,
    model_type,
    torch_dtype,
    quantization_summary,
    multimodal_note,
    vision,
    vocab_size,
    hidden_size,
    num_hidden_layers,
    max_position_embeddings,
    rms_norm_eps,
    tie_word_embeddings,
    rope_summary,
    hidden_act,
    num_nextn_predict_layers,
    tokenizer_class,
    default_temperature,
    default_top_p,
    segments,
    params,
    safetensors,
    raw_config_pretty,
  })
}

fn parse_vision_spec(vc: &serde_json::Map<String, Value>, text_hidden: u64) -> Option<VisionSpec> {
  let hidden_size = u64_any(vc, &["hidden_size", "embed_dim", "vision_width"])?;
  if hidden_size == 0 {
    return None;
  }
  let num_hidden_layers = u64_any(vc, &["num_hidden_layers", "depth"]).unwrap_or(24).max(1);
  let num_attention_heads = u64_any(vc, &["num_attention_heads", "num_heads"]).unwrap_or(16).max(1);
  let intermediate_size = u64_any(vc, &["intermediate_size", "mlp_dim"]).unwrap_or(hidden_size * 4);
  let image_size = u64_any(vc, &["image_size"]);
  let patch_size = u64_any(vc, &["patch_size"]);
  let model_type = vc.get("model_type").and_then(Value::as_str).unwrap_or("vision_encoder");

  let patch = patch_size.unwrap_or(14);
  let patch_embed = 3 * patch * patch * hidden_size;
  let per_layer = 4 * hidden_size * hidden_size + 2 * hidden_size * intermediate_size;
  let projector = hidden_size * text_hidden * 2;
  let estimated_params = patch_embed + num_hidden_layers * per_layer + projector;

  Some(VisionSpec {
    model_type: model_type.to_string(),
    hidden_size,
    num_hidden_layers,
    num_attention_heads,
    intermediate_size,
    image_size,
    patch_size,
    estimated_params,
  })
}

fn detect_nope_interval(cfg: &serde_json::Map<String, Value>) -> Option<u64> {
  let arr = cfg.get("no_rope_layers")?.as_array()?;
  if arr.len() < 4 {
    return None;
  }
  // HuggingFace Llama4TextConfig 中 `no_rope_layers` 取 0 表示禁用 RoPE (NoPE)，1 表示启用 RoPE
  let zeros = arr.iter().filter(|v| v.as_u64() == Some(0) || v.as_bool() == Some(false)).count();
  if zeros == 0 || zeros == arr.len() {
    return None;
  }
  Some((arr.len() / zeros).max(2) as u64)
}

fn build_schedule_note(
  attn: &AttentionSpec,
  cfg: &serde_json::Map<String, Value>,
) -> Option<String> {
  match attn {
    AttentionSpec::StandardGqa {
      sliding_window: Some(w),
      sliding_window_pattern: Some(pat),
      ..
    } => {
      let local_n = pat.saturating_sub(1);
      Some(format!(
        "每 {pat} 层含 {local_n} 层局部滑动窗口注意力 (SWA W={w}) + 1 层全局全注意力，大幅削减长文本 KV 显存"
      ))
    }
    AttentionSpec::StandardGqa { nope_layer_interval: Some(iv), .. } => {
      let rope_n = iv.saturating_sub(1);
      let temp_tuning = cfg.get("attn_temperature_tuning").and_then(Value::as_bool).unwrap_or(true);
      let extra = if temp_tuning { "（带注意力温度自适应缩放）" } else { "" };
      Some(format!(
        "iRoPE 混合位置架构：每 {iv} 层含 {rope_n} 层 RoPE 局部旋转层 + 1 层 NoPE 无位置编码全局层{extra}"
      ))
    }
    _ => None,
  }
}

fn compute_param_breakdown(
  vision_encoder: u64,
  vocab_size: u64,
  hidden_size: u64,
  tie_word_embeddings: bool,
  num_nextn_predict_layers: u64,
  segments: &[LayerSegment],
) -> ParamBreakdown {
  let embedding = vocab_size * hidden_size;
  let lm_head = if tie_word_embeddings { 0 } else { vocab_size * hidden_size };

  let mut attention = 0u64;
  let mut dense_ffn = 0u64;
  let mut moe_routed = 0u64;
  let mut moe_shared = 0u64;
  let mut norms_and_routers = 0u64;
  let mut active_trunk = 0u64;

  for seg in segments {
    let count = seg.layer_count();
    let attn_one = seg.attention.params_per_layer(hidden_size);
    let norms_one = seg.norms_per_layer.max(2) * hidden_size;
    attention += count * attn_one;
    norms_and_routers += count * norms_one;
    active_trunk += count * (attn_one + norms_one);

    match &seg.ffn {
      FfnSpec::DenseGated { .. } => {
        let (tot, act) = seg.ffn.params_per_layer(hidden_size);
        dense_ffn += count * tot;
        active_trunk += count * act;
      }
      FfnSpec::SparseMoe {
        num_routed_experts,
        top_k,
        moe_intermediate_size,
        num_shared_experts,
        shared_intermediate_size,
        has_shared_gate,
        ..
      } => {
        let router = hidden_size * num_routed_experts;
        let one_expert = 3 * hidden_size * moe_intermediate_size;
        let routed_tot = num_routed_experts * one_expert;
        let routed_act = (*top_k).min(*num_routed_experts) * one_expert;
        let shared = if *num_shared_experts > 0 && *shared_intermediate_size > 0 {
          3 * hidden_size * shared_intermediate_size
            + if *has_shared_gate { hidden_size } else { 0 }
        } else {
          0
        };
        norms_and_routers += count * router;
        moe_routed += count * routed_tot;
        moe_shared += count * shared;
        active_trunk += count * (router + routed_act + shared);
      }
    }
  }

  // Final RMSNorm
  norms_and_routers += hidden_size;
  active_trunk += hidden_size;

  // MTP 预测头参数估算（每个 MTP 层含 eh_proj [hidden, 2*hidden] + 1 个同构末端 Block）
  let mtp_head = if num_nextn_predict_layers > 0 {
    if let Some(last_seg) = segments.last() {
      let (ffn_tot, _) = last_seg.ffn.params_per_layer(hidden_size);
      let one_mtp = 2 * hidden_size * hidden_size
        + last_seg.attention.params_per_layer(hidden_size)
        + ffn_tot
        + 4 * hidden_size;
      num_nextn_predict_layers * one_mtp
    } else {
      0
    }
  } else {
    0
  };

  let total_params = vision_encoder
    + embedding
    + attention
    + dense_ffn
    + moe_routed
    + moe_shared
    + norms_and_routers
    + lm_head;
  let active_params = vision_encoder + embedding + active_trunk + lm_head;

  ParamBreakdown {
    vision_encoder,
    embedding,
    attention,
    dense_ffn,
    moe_routed,
    moe_shared,
    norms_and_routers,
    lm_head,
    mtp_head,
    total_params,
    active_params,
  }
}

#[allow(clippy::too_many_arguments)]
fn build_safetensors_fallback_blueprint(
  repo_id: &str,
  source_label: &str,
  architectures: Vec<String>,
  model_type: String,
  torch_dtype: String,
  quantization_summary: Option<String>,
  st: SafetensorsSummary,
  raw_config_pretty: String,
) -> ModelBlueprint {
  let summed_elems: u64 = st.patterns.iter().filter_map(TensorPatternEntry::total_elements).sum();
  let approx_params =
    if summed_elems > 0 { summed_elems } else { st.total_bytes.map(|b| b / 2).unwrap_or(0) };
  let layers = st.detected_num_layers.unwrap_or(1);

  ModelBlueprint {
    repo_id: repo_id.to_string(),
    source_label: source_label.to_string(),
    family: ArchFamily::SafetensorsTree,
    architectures,
    model_type,
    torch_dtype,
    quantization_summary,
    multimodal_note: None,
    vision: None,
    vocab_size: 0,
    hidden_size: 0,
    num_hidden_layers: layers,
    max_position_embeddings: 0,
    rms_norm_eps: None,
    tie_word_embeddings: false,
    rope_summary: "N/A (Safetensors Tree)".to_string(),
    hidden_act: "unknown".to_string(),
    num_nextn_predict_layers: 0,
    tokenizer_class: None,
    default_temperature: None,
    default_top_p: None,
    segments: Vec::new(),
    params: ParamBreakdown {
      total_params: approx_params,
      active_params: approx_params,
      ..ParamBreakdown::default()
    },
    safetensors: Some(st),
    raw_config_pretty,
  }
}

fn u64_any(obj: &serde_json::Map<String, Value>, keys: &[&str]) -> Option<u64> {
  keys.iter().find_map(|k| obj.get(*k).and_then(Value::as_u64))
}

fn f64_any(obj: &serde_json::Map<String, Value>, keys: &[&str]) -> Option<f64> {
  keys.iter().find_map(|k| obj.get(*k).and_then(Value::as_f64))
}

fn extract_rope_summary(cfg: &serde_json::Map<String, Value>) -> String {
  let theta = f64_any(cfg, &["rope_theta"]).unwrap_or(10_000.0);
  let theta_str = if theta >= 1_000_000.0 && (theta % 1_000_000.0).abs() < 1.0 {
    format!("{:.0}M", theta / 1_000_000.0)
  } else if theta >= 1_000.0 && (theta % 1_000.0).abs() < 1.0 {
    format!("{:.0}K", theta / 1_000.0)
  } else {
    format!("{theta:.0}")
  };

  let partial = f64_any(cfg, &["partial_rotary_factor", "rotary_percent"])
    .filter(|&p| p > 0.0 && p < 0.999)
    .map(|p| format!(" · {:.0}%d", p * 100.0))
    .unwrap_or_default();

  if let Some(scaling) = cfg.get("rope_scaling").and_then(Value::as_object) {
    let rtype = scaling
      .get("rope_type")
      .or_else(|| scaling.get("type"))
      .and_then(Value::as_str)
      .unwrap_or("scaled");
    let factor = scaling.get("factor").and_then(Value::as_f64);
    match factor {
      Some(f) => format!("RoPE θ={theta_str}{partial} ({rtype} ×{f:.1})"),
      None => format!("RoPE θ={theta_str}{partial} ({rtype})"),
    }
  } else {
    format!("RoPE θ={theta_str}{partial}")
  }
}

fn extract_quantization_summary(cfg: &serde_json::Map<String, Value>) -> Option<String> {
  let q = cfg.get("quantization_config")?.as_object()?;
  let method = q.get("quant_method").and_then(Value::as_str).unwrap_or("quantized");
  let bits = q.get("bits").and_then(Value::as_u64);
  let fmt = q.get("fmt").and_then(Value::as_str);
  let group = q.get("group_size").and_then(Value::as_u64);
  let block = q.get("weight_block_size").and_then(Value::as_array).map(|a| {
    a.iter().filter_map(Value::as_u64).map(|n| n.to_string()).collect::<Vec<_>>().join("×")
  });

  let mut parts = vec![method.to_ascii_uppercase()];
  if let Some(b) = bits {
    parts.push(format!("{b}-bit"));
  }
  if let Some(f) = fmt {
    parts.push(f.to_string());
  }
  if let Some(g) = group {
    parts.push(format!("g={g}"));
  }
  if let Some(bl) = block.filter(|s| !s.is_empty()) {
    parts.push(format!("block {bl}"));
  }
  Some(parts.join(" · "))
}

fn parse_tokenizer_meta(raw: Option<&str>) -> (Option<String>, Option<u64>) {
  let Some(s) = raw else {
    return (None, None);
  };
  let Ok(val) = serde_json::from_str::<Value>(s) else {
    return (None, None);
  };
  let cls = val.get("tokenizer_class").and_then(Value::as_str).map(str::to_string);
  let vocab = val.get("vocab_size").and_then(Value::as_u64);
  (cls, vocab)
}

fn parse_generation_meta(raw: Option<&str>) -> (Option<f64>, Option<f64>) {
  let Some(s) = raw else {
    return (None, None);
  };
  let Ok(val) = serde_json::from_str::<Value>(s) else {
    return (None, None);
  };
  let temp = val.get("temperature").and_then(Value::as_f64);
  let top_p = val.get("top_p").and_then(Value::as_f64);
  (temp, top_p)
}

// ─────────────────────────────────────────────────────────────────────────────
// 内置四大旗舰模型真实快照（零网络延迟，SSR 直出与离线对照基准）
// ─────────────────────────────────────────────────────────────────────────────

/// 内置预设条目描述。
#[derive(Debug, Clone, Copy)]
pub struct BuiltinPresetMeta {
  pub id: &'static str,
  pub repo_id: &'static str,
  pub title: &'static str,
  pub tag: &'static str,
}

pub const BUILTIN_PRESETS: &[BuiltinPresetMeta] = &[
  BuiltinPresetMeta {
    id: "deepseek-v3",
    repo_id: "deepseek-ai/DeepSeek-V3",
    title: "DeepSeek-V3 (671B)",
    tag: "MLA + 256E MoE + MTP",
  },
  BuiltinPresetMeta {
    id: "qwen3-30b-a3b",
    repo_id: "Qwen/Qwen3-30B-A3B",
    title: "Qwen3-30B-A3B",
    tag: "128E MoE + QK-Norm",
  },
  BuiltinPresetMeta {
    id: "qwen3-8b",
    repo_id: "Qwen/Qwen3-8B",
    title: "Qwen3-8B",
    tag: "GQA + QK-Norm",
  },
  BuiltinPresetMeta {
    id: "llama-3.1-8b",
    repo_id: "meta-llama/Llama-3.1-8B-Instruct",
    title: "Llama-3.1-8B",
    tag: "Classic GQA + 128K",
  },
];

/// 按预设 slug（如 `"deepseek-v3"`）或仓库名（如 `"Qwen/Qwen3-8B"`）加载内置真实架构快照。
pub fn load_builtin_preset(key: &str) -> Option<ModelBlueprint> {
  let norm = key.trim().to_ascii_lowercase();
  match norm.as_str() {
    "deepseek-v3" | "deepseek-ai/deepseek-v3" | "deepseek-r1" | "deepseek-ai/deepseek-r1" => {
      build_model_blueprint(
        "deepseek-ai/DeepSeek-V3",
        "Built-in Snapshot (HF/ModelScope Verified)",
        DEEPSEEK_V3_CONFIG,
        Some(r#"{"temperature": 0.6, "top_p": 0.95}"#),
        Some(r#"{"tokenizer_class": "LlamaTokenizerFast"}"#),
        Some(deepseek_v3_safetensors_snapshot()),
      )
      .ok()
    }
    "qwen3-30b-a3b" | "qwen/qwen3-30b-a3b" => build_model_blueprint(
      "Qwen/Qwen3-30B-A3B",
      "Built-in Snapshot (HF/ModelScope Verified)",
      QWEN3_30B_A3B_CONFIG,
      Some(r#"{"temperature": 0.6, "top_p": 0.95}"#),
      Some(r#"{"tokenizer_class": "Qwen2Tokenizer"}"#),
      Some(qwen3_30b_safetensors_snapshot()),
    )
    .ok(),
    "qwen3-8b" | "qwen/qwen3-8b" => build_model_blueprint(
      "Qwen/Qwen3-8B",
      "Built-in Snapshot (HF/ModelScope Verified)",
      QWEN3_8B_CONFIG,
      Some(r#"{"temperature": 0.6, "top_p": 0.95}"#),
      Some(r#"{"tokenizer_class": "Qwen2Tokenizer"}"#),
      Some(qwen3_8b_safetensors_snapshot()),
    )
    .ok(),
    "llama-3.1-8b" | "llama3-8b" | "meta-llama/llama-3.1-8b-instruct" => build_model_blueprint(
      "meta-llama/Llama-3.1-8B-Instruct",
      "Built-in Snapshot (HF/ModelScope Verified)",
      LLAMA_3_1_8B_CONFIG,
      Some(r#"{"temperature": 0.6, "top_p": 0.9}"#),
      Some(r#"{"tokenizer_class": "PreTrainedTokenizerFast"}"#),
      Some(llama3_8b_safetensors_snapshot()),
    )
    .ok(),
    _ => None,
  }
}

const DEEPSEEK_V3_CONFIG: &str = r#"{
  "architectures": ["DeepseekV3ForCausalLM"],
  "model_type": "deepseek_v3",
  "vocab_size": 129280,
  "hidden_size": 7168,
  "intermediate_size": 18432,
  "moe_intermediate_size": 2048,
  "num_hidden_layers": 61,
  "num_nextn_predict_layers": 1,
  "num_attention_heads": 128,
  "num_key_value_heads": 128,
  "n_shared_experts": 1,
  "n_routed_experts": 256,
  "num_experts_per_tok": 8,
  "first_k_dense_replace": 3,
  "n_group": 8,
  "topk_group": 4,
  "scoring_func": "sigmoid",
  "routed_scaling_factor": 2.5,
  "kv_lora_rank": 512,
  "q_lora_rank": 1536,
  "qk_rope_head_dim": 64,
  "v_head_dim": 128,
  "qk_nope_head_dim": 128,
  "hidden_act": "silu",
  "max_position_embeddings": 163840,
  "rms_norm_eps": 1e-6,
  "rope_theta": 10000.0,
  "rope_scaling": { "type": "yarn", "factor": 40.0, "mscale": 1.0, "mscale_all_dim": 1.0 },
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "quantization_config": {
    "quant_method": "fp8",
    "fmt": "e4m3",
    "activation_scheme": "dynamic",
    "weight_block_size": [128, 128]
  }
}"#;

const QWEN3_30B_A3B_CONFIG: &str = r#"{
  "architectures": ["Qwen3MoeForCausalLM"],
  "model_type": "qwen3_moe",
  "vocab_size": 151936,
  "hidden_size": 2048,
  "intermediate_size": 6144,
  "moe_intermediate_size": 768,
  "num_hidden_layers": 48,
  "num_attention_heads": 32,
  "num_key_value_heads": 4,
  "head_dim": 128,
  "num_experts": 128,
  "num_experts_per_tok": 8,
  "norm_topk_prob": true,
  "hidden_act": "silu",
  "max_position_embeddings": 40960,
  "rms_norm_eps": 1e-6,
  "rope_theta": 1000000.0,
  "attention_bias": false,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16"
}"#;

const QWEN3_8B_CONFIG: &str = r#"{
  "architectures": ["Qwen3ForCausalLM"],
  "model_type": "qwen3",
  "vocab_size": 151936,
  "hidden_size": 4096,
  "intermediate_size": 12288,
  "num_hidden_layers": 36,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "head_dim": 128,
  "hidden_act": "silu",
  "max_position_embeddings": 40960,
  "rms_norm_eps": 1e-6,
  "rope_theta": 1000000.0,
  "attention_bias": false,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16"
}"#;

const LLAMA_3_1_8B_CONFIG: &str = r#"{
  "architectures": ["LlamaForCausalLM"],
  "model_type": "llama",
  "vocab_size": 128256,
  "hidden_size": 4096,
  "intermediate_size": 14336,
  "num_hidden_layers": 32,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "hidden_act": "silu",
  "max_position_embeddings": 131072,
  "rms_norm_eps": 1e-5,
  "rope_theta": 500000.0,
  "rope_scaling": { "rope_type": "llama3", "factor": 8.0 },
  "attention_bias": false,
  "mlp_bias": false,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16"
}"#;

#[allow(clippy::too_many_arguments)]
fn pat(
  pattern: &str,
  example_name: &str,
  stage: &str,
  repeat_count: usize,
  shape: &[u64],
  dtype: &str,
  bytes_per_elem: u64,
  shard: &str,
) -> TensorPatternEntry {
  let elems: u64 = shape.iter().product();
  TensorPatternEntry {
    pattern: pattern.to_string(),
    example_name: example_name.to_string(),
    stage: stage.to_string(),
    repeat_count,
    shape: Some(shape.to_vec()),
    dtype: Some(dtype.to_string()),
    single_byte_len: Some(elems * bytes_per_elem),
    example_shard: Some(shard.to_string()),
  }
}

fn deepseek_v3_safetensors_snapshot() -> SafetensorsSummary {
  SafetensorsSummary {
    source_kind: "model.safetensors.index.json (163 shards, FP8 + BF16)".to_string(),
    total_bytes: Some(685_520_000_000),
    shard_count: 163,
    total_tensor_count: 90_945,
    header_inspected_count: 90_945,
    detected_num_layers: Some(61),
    detected_num_experts: Some(256),
    detected_first_moe_layer: Some(3),
    has_attn_bias: false,
    has_qk_norm: false,
    dtype_histogram: vec![
      ("F8_E4M3".to_string(), 45_280),
      ("F32".to_string(), 45_280),
      ("BF16".to_string(), 385),
    ],
    patterns: vec![
      pat(
        "model.embed_tokens.weight",
        "model.embed_tokens.weight",
        "embedding",
        1,
        &[129280, 7168],
        "BF16",
        2,
        "model-00001-of-00163.safetensors",
      ),
      pat(
        "model.layers.[0..60].self_attn.q_a_proj.weight",
        "model.layers.0.self_attn.q_a_proj.weight",
        "attention",
        61,
        &[1536, 7168],
        "F8_E4M3",
        1,
        "model-00001-of-00163.safetensors",
      ),
      pat(
        "model.layers.[0..60].self_attn.q_a_layernorm.weight",
        "model.layers.0.self_attn.q_a_layernorm.weight",
        "attention",
        61,
        &[1536],
        "BF16",
        2,
        "model-00001-of-00163.safetensors",
      ),
      pat(
        "model.layers.[0..60].self_attn.q_b_proj.weight",
        "model.layers.0.self_attn.q_b_proj.weight",
        "attention",
        61,
        &[24576, 1536],
        "F8_E4M3",
        1,
        "model-00001-of-00163.safetensors",
      ),
      pat(
        "model.layers.[0..60].self_attn.kv_a_proj_with_mqa.weight",
        "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
        "attention",
        61,
        &[576, 7168],
        "F8_E4M3",
        1,
        "model-00001-of-00163.safetensors",
      ),
      pat(
        "model.layers.[0..60].self_attn.kv_a_layernorm.weight",
        "model.layers.0.self_attn.kv_a_layernorm.weight",
        "attention",
        61,
        &[512],
        "BF16",
        2,
        "model-00001-of-00163.safetensors",
      ),
      pat(
        "model.layers.[0..60].self_attn.kv_b_proj.weight",
        "model.layers.0.self_attn.kv_b_proj.weight",
        "attention",
        61,
        &[32768, 512],
        "F8_E4M3",
        1,
        "model-00001-of-00163.safetensors",
      ),
      pat(
        "model.layers.[0..60].self_attn.o_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "attention",
        61,
        &[7168, 16384],
        "F8_E4M3",
        1,
        "model-00001-of-00163.safetensors",
      ),
      pat(
        "model.layers.[0..60].input_layernorm.weight",
        "model.layers.0.input_layernorm.weight",
        "norm",
        61,
        &[7168],
        "BF16",
        2,
        "model-00001-of-00163.safetensors",
      ),
      pat(
        "model.layers.[0..60].post_attention_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "norm",
        61,
        &[7168],
        "BF16",
        2,
        "model-00001-of-00163.safetensors",
      ),
      pat(
        "model.layers.[0..2].mlp.gate_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "ffn",
        3,
        &[18432, 7168],
        "F8_E4M3",
        1,
        "model-00001-of-00163.safetensors",
      ),
      pat(
        "model.layers.[0..2].mlp.up_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "ffn",
        3,
        &[18432, 7168],
        "F8_E4M3",
        1,
        "model-00001-of-00163.safetensors",
      ),
      pat(
        "model.layers.[0..2].mlp.down_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "ffn",
        3,
        &[7168, 18432],
        "F8_E4M3",
        1,
        "model-00001-of-00163.safetensors",
      ),
      pat(
        "model.layers.[3..60].mlp.gate.weight",
        "model.layers.3.mlp.gate.weight",
        "moe",
        58,
        &[256, 7168],
        "BF16",
        2,
        "model-00002-of-00163.safetensors",
      ),
      pat(
        "model.layers.[3..60].mlp.gate.e_score_correction_bias",
        "model.layers.3.mlp.gate.e_score_correction_bias",
        "moe",
        58,
        &[256],
        "BF16",
        2,
        "model-00002-of-00163.safetensors",
      ),
      pat(
        "model.layers.[3..60].mlp.shared_experts.gate_proj.weight",
        "model.layers.3.mlp.shared_experts.gate_proj.weight",
        "moe",
        58,
        &[2048, 7168],
        "F8_E4M3",
        1,
        "model-00002-of-00163.safetensors",
      ),
      pat(
        "model.layers.[3..60].mlp.shared_experts.up_proj.weight",
        "model.layers.3.mlp.shared_experts.up_proj.weight",
        "moe",
        58,
        &[2048, 7168],
        "F8_E4M3",
        1,
        "model-00002-of-00163.safetensors",
      ),
      pat(
        "model.layers.[3..60].mlp.shared_experts.down_proj.weight",
        "model.layers.3.mlp.shared_experts.down_proj.weight",
        "moe",
        58,
        &[7168, 2048],
        "F8_E4M3",
        1,
        "model-00002-of-00163.safetensors",
      ),
      pat(
        "model.layers.[3..60].mlp.experts.[0..255].gate_proj.weight",
        "model.layers.3.mlp.experts.0.gate_proj.weight",
        "moe",
        14_848,
        &[2048, 7168],
        "F8_E4M3",
        1,
        "model-00002-of-00163.safetensors",
      ),
      pat(
        "model.layers.[3..60].mlp.experts.[0..255].up_proj.weight",
        "model.layers.3.mlp.experts.0.up_proj.weight",
        "moe",
        14_848,
        &[2048, 7168],
        "F8_E4M3",
        1,
        "model-00002-of-00163.safetensors",
      ),
      pat(
        "model.layers.[3..60].mlp.experts.[0..255].down_proj.weight",
        "model.layers.3.mlp.experts.0.down_proj.weight",
        "moe",
        14_848,
        &[7168, 2048],
        "F8_E4M3",
        1,
        "model-00002-of-00163.safetensors",
      ),
      pat(
        "model.layers.61.eh_proj.weight",
        "model.layers.61.eh_proj.weight",
        "mtp",
        1,
        &[7168, 14336],
        "BF16",
        2,
        "model-00163-of-00163.safetensors",
      ),
      pat(
        "model.norm.weight",
        "model.norm.weight",
        "norm",
        1,
        &[7168],
        "BF16",
        2,
        "model-00162-of-00163.safetensors",
      ),
      pat(
        "lm_head.weight",
        "lm_head.weight",
        "lm_head",
        1,
        &[129280, 7168],
        "BF16",
        2,
        "model-00162-of-00163.safetensors",
      ),
    ],
  }
}

fn qwen3_30b_safetensors_snapshot() -> SafetensorsSummary {
  SafetensorsSummary {
    source_kind: "model.safetensors.index.json (16 shards, BF16)".to_string(),
    total_bytes: Some(61_060_000_000),
    shard_count: 16,
    total_tensor_count: 18_866,
    header_inspected_count: 18_866,
    detected_num_layers: Some(48),
    detected_num_experts: Some(128),
    detected_first_moe_layer: Some(0),
    has_attn_bias: false,
    has_qk_norm: true,
    dtype_histogram: vec![("BF16".to_string(), 18_866)],
    patterns: vec![
      pat(
        "model.embed_tokens.weight",
        "model.embed_tokens.weight",
        "embedding",
        1,
        &[151936, 2048],
        "BF16",
        2,
        "model-00001-of-00016.safetensors",
      ),
      pat(
        "model.layers.[0..47].self_attn.q_proj.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "attention",
        48,
        &[4096, 2048],
        "BF16",
        2,
        "model-00001-of-00016.safetensors",
      ),
      pat(
        "model.layers.[0..47].self_attn.k_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "attention",
        48,
        &[512, 2048],
        "BF16",
        2,
        "model-00001-of-00016.safetensors",
      ),
      pat(
        "model.layers.[0..47].self_attn.v_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "attention",
        48,
        &[512, 2048],
        "BF16",
        2,
        "model-00001-of-00016.safetensors",
      ),
      pat(
        "model.layers.[0..47].self_attn.o_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "attention",
        48,
        &[2048, 4096],
        "BF16",
        2,
        "model-00001-of-00016.safetensors",
      ),
      pat(
        "model.layers.[0..47].self_attn.q_norm.weight",
        "model.layers.0.self_attn.q_norm.weight",
        "attention",
        48,
        &[128],
        "BF16",
        2,
        "model-00001-of-00016.safetensors",
      ),
      pat(
        "model.layers.[0..47].self_attn.k_norm.weight",
        "model.layers.0.self_attn.k_norm.weight",
        "attention",
        48,
        &[128],
        "BF16",
        2,
        "model-00001-of-00016.safetensors",
      ),
      pat(
        "model.layers.[0..47].input_layernorm.weight",
        "model.layers.0.input_layernorm.weight",
        "norm",
        48,
        &[2048],
        "BF16",
        2,
        "model-00001-of-00016.safetensors",
      ),
      pat(
        "model.layers.[0..47].post_attention_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "norm",
        48,
        &[2048],
        "BF16",
        2,
        "model-00001-of-00016.safetensors",
      ),
      pat(
        "model.layers.[0..47].mlp.gate.weight",
        "model.layers.0.mlp.gate.weight",
        "moe",
        48,
        &[128, 2048],
        "BF16",
        2,
        "model-00001-of-00016.safetensors",
      ),
      pat(
        "model.layers.[0..47].mlp.experts.[0..127].gate_proj.weight",
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "moe",
        6144,
        &[768, 2048],
        "BF16",
        2,
        "model-00001-of-00016.safetensors",
      ),
      pat(
        "model.layers.[0..47].mlp.experts.[0..127].up_proj.weight",
        "model.layers.0.mlp.experts.0.up_proj.weight",
        "moe",
        6144,
        &[768, 2048],
        "BF16",
        2,
        "model-00001-of-00016.safetensors",
      ),
      pat(
        "model.layers.[0..47].mlp.experts.[0..127].down_proj.weight",
        "model.layers.0.mlp.experts.0.down_proj.weight",
        "moe",
        6144,
        &[2048, 768],
        "BF16",
        2,
        "model-00001-of-00016.safetensors",
      ),
      pat(
        "model.norm.weight",
        "model.norm.weight",
        "norm",
        1,
        &[2048],
        "BF16",
        2,
        "model-00016-of-00016.safetensors",
      ),
      pat(
        "lm_head.weight",
        "lm_head.weight",
        "lm_head",
        1,
        &[151936, 2048],
        "BF16",
        2,
        "model-00016-of-00016.safetensors",
      ),
    ],
  }
}

fn qwen3_8b_safetensors_snapshot() -> SafetensorsSummary {
  SafetensorsSummary {
    source_kind: "model.safetensors.index.json (5 shards, BF16)".to_string(),
    total_bytes: Some(16_380_000_000),
    shard_count: 5,
    total_tensor_count: 399,
    header_inspected_count: 399,
    detected_num_layers: Some(36),
    detected_num_experts: None,
    detected_first_moe_layer: None,
    has_attn_bias: false,
    has_qk_norm: true,
    dtype_histogram: vec![("BF16".to_string(), 399)],
    patterns: vec![
      pat(
        "model.embed_tokens.weight",
        "model.embed_tokens.weight",
        "embedding",
        1,
        &[151936, 4096],
        "BF16",
        2,
        "model-00001-of-00005.safetensors",
      ),
      pat(
        "model.layers.[0..35].self_attn.q_proj.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "attention",
        36,
        &[4096, 4096],
        "BF16",
        2,
        "model-00001-of-00005.safetensors",
      ),
      pat(
        "model.layers.[0..35].self_attn.k_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "attention",
        36,
        &[1024, 4096],
        "BF16",
        2,
        "model-00001-of-00005.safetensors",
      ),
      pat(
        "model.layers.[0..35].self_attn.v_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "attention",
        36,
        &[1024, 4096],
        "BF16",
        2,
        "model-00001-of-00005.safetensors",
      ),
      pat(
        "model.layers.[0..35].self_attn.o_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "attention",
        36,
        &[4096, 4096],
        "BF16",
        2,
        "model-00001-of-00005.safetensors",
      ),
      pat(
        "model.layers.[0..35].self_attn.q_norm.weight",
        "model.layers.0.self_attn.q_norm.weight",
        "attention",
        36,
        &[128],
        "BF16",
        2,
        "model-00001-of-00005.safetensors",
      ),
      pat(
        "model.layers.[0..35].self_attn.k_norm.weight",
        "model.layers.0.self_attn.k_norm.weight",
        "attention",
        36,
        &[128],
        "BF16",
        2,
        "model-00001-of-00005.safetensors",
      ),
      pat(
        "model.layers.[0..35].input_layernorm.weight",
        "model.layers.0.input_layernorm.weight",
        "norm",
        36,
        &[4096],
        "BF16",
        2,
        "model-00001-of-00005.safetensors",
      ),
      pat(
        "model.layers.[0..35].post_attention_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "norm",
        36,
        &[4096],
        "BF16",
        2,
        "model-00001-of-00005.safetensors",
      ),
      pat(
        "model.layers.[0..35].mlp.gate_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "ffn",
        36,
        &[12288, 4096],
        "BF16",
        2,
        "model-00001-of-00005.safetensors",
      ),
      pat(
        "model.layers.[0..35].mlp.up_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "ffn",
        36,
        &[12288, 4096],
        "BF16",
        2,
        "model-00001-of-00005.safetensors",
      ),
      pat(
        "model.layers.[0..35].mlp.down_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "ffn",
        36,
        &[4096, 12288],
        "BF16",
        2,
        "model-00001-of-00005.safetensors",
      ),
      pat(
        "model.norm.weight",
        "model.norm.weight",
        "norm",
        1,
        &[4096],
        "BF16",
        2,
        "model-00005-of-00005.safetensors",
      ),
      pat(
        "lm_head.weight",
        "lm_head.weight",
        "lm_head",
        1,
        &[151936, 4096],
        "BF16",
        2,
        "model-00005-of-00005.safetensors",
      ),
    ],
  }
}

fn llama3_8b_safetensors_snapshot() -> SafetensorsSummary {
  SafetensorsSummary {
    source_kind: "model.safetensors.index.json (4 shards, BF16)".to_string(),
    total_bytes: Some(16_060_522_496),
    shard_count: 4,
    total_tensor_count: 291,
    header_inspected_count: 291,
    detected_num_layers: Some(32),
    detected_num_experts: None,
    detected_first_moe_layer: None,
    has_attn_bias: false,
    has_qk_norm: false,
    dtype_histogram: vec![("BF16".to_string(), 291)],
    patterns: vec![
      pat(
        "model.embed_tokens.weight",
        "model.embed_tokens.weight",
        "embedding",
        1,
        &[128256, 4096],
        "BF16",
        2,
        "model-00001-of-00004.safetensors",
      ),
      pat(
        "model.layers.[0..31].self_attn.q_proj.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "attention",
        32,
        &[4096, 4096],
        "BF16",
        2,
        "model-00001-of-00004.safetensors",
      ),
      pat(
        "model.layers.[0..31].self_attn.k_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "attention",
        32,
        &[1024, 4096],
        "BF16",
        2,
        "model-00001-of-00004.safetensors",
      ),
      pat(
        "model.layers.[0..31].self_attn.v_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "attention",
        32,
        &[1024, 4096],
        "BF16",
        2,
        "model-00001-of-00004.safetensors",
      ),
      pat(
        "model.layers.[0..31].self_attn.o_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "attention",
        32,
        &[4096, 4096],
        "BF16",
        2,
        "model-00001-of-00004.safetensors",
      ),
      pat(
        "model.layers.[0..31].input_layernorm.weight",
        "model.layers.0.input_layernorm.weight",
        "norm",
        32,
        &[4096],
        "BF16",
        2,
        "model-00001-of-00004.safetensors",
      ),
      pat(
        "model.layers.[0..31].post_attention_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "norm",
        32,
        &[4096],
        "BF16",
        2,
        "model-00001-of-00004.safetensors",
      ),
      pat(
        "model.layers.[0..31].mlp.gate_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "ffn",
        32,
        &[14336, 4096],
        "BF16",
        2,
        "model-00001-of-00004.safetensors",
      ),
      pat(
        "model.layers.[0..31].mlp.up_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "ffn",
        32,
        &[14336, 4096],
        "BF16",
        2,
        "model-00001-of-00004.safetensors",
      ),
      pat(
        "model.layers.[0..31].mlp.down_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "ffn",
        32,
        &[4096, 14336],
        "BF16",
        2,
        "model-00001-of-00004.safetensors",
      ),
      pat(
        "model.norm.weight",
        "model.norm.weight",
        "norm",
        1,
        &[4096],
        "BF16",
        2,
        "model-00004-of-00004.safetensors",
      ),
      pat(
        "lm_head.weight",
        "lm_head.weight",
        "lm_head",
        1,
        &[128256, 4096],
        "BF16",
        2,
        "model-00004-of-00004.safetensors",
      ),
    ],
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn deepseek_v3_blueprint_matches_671b_total_and_37b_active() {
    let bp = load_builtin_preset("deepseek-v3").unwrap();
    assert_eq!(bp.family, ArchFamily::DeepSeekMla);
    assert_eq!(bp.segments.len(), 2);
    assert_eq!(bp.segments[0].range_label(), "Layers 0–2 (×3)");
    assert_eq!(bp.segments[1].range_label(), "Layers 3–60 (×58)");
    assert_eq!(bp.num_nextn_predict_layers, 1);

    // 671.0B total, ~37.5B active
    assert!(bp.params.total_params > 670_000_000_000 && bp.params.total_params < 673_000_000_000);
    assert!(bp.params.active_params > 37_000_000_000 && bp.params.active_params < 38_000_000_000);

    // MLA KV Cache compression vs MHA: 32768 / 576 = 56.888...x
    let ratio = bp.segments[1].attention.kv_compression_ratio();
    assert!((ratio - 56.888).abs() < 0.05);
  }

  #[test]
  fn qwen3_moe_respects_decoupled_head_dim_and_qk_norm() {
    let bp = load_builtin_preset("qwen3-30b-a3b").unwrap();
    assert_eq!(bp.family, ArchFamily::QwenMoe);
    assert!(bp.params.total_params > 30_000_000_000 && bp.params.total_params < 31_000_000_000);
    assert!(bp.params.active_params > 3_200_000_000 && bp.params.active_params < 3_500_000_000);

    match &bp.segments[0].attention {
      AttentionSpec::StandardGqa { num_q_heads, num_kv_heads, head_dim, qk_norm, .. } => {
        assert_eq!((*num_q_heads, *num_kv_heads, *head_dim), (32, 4, 128));
        assert!(*qk_norm);
      }
      other => panic!("unexpected attention spec: {other:?}"),
    }
  }

  #[test]
  fn llama_3_1_8b_matches_8_03b_and_4x_gqa() {
    let bp = load_builtin_preset("llama-3.1-8b").unwrap();
    assert_eq!(bp.family, ArchFamily::LlamaDense);
    assert_eq!(bp.params.total_params, bp.params.active_params);
    assert!(bp.params.total_params > 8_020_000_000 && bp.params.total_params < 8_040_000_000);
    assert!((bp.segments[0].attention.kv_compression_ratio() - 4.0).abs() < 1e-6);
  }

  #[test]
  fn unwraps_vlm_text_config_cleanly() {
    let vlm_cfg = r#"{
      "architectures": ["Qwen2_5_VLForConditionalGeneration"],
      "model_type": "qwen2_5_vl",
      "vision_config": { "model_type": "qwen2_5_vl_vision", "hidden_size": 1280 },
      "text_config": {
        "model_type": "qwen2",
        "vocab_size": 152064,
        "hidden_size": 3584,
        "intermediate_size": 18944,
        "num_hidden_layers": 28,
        "num_attention_heads": 28,
        "num_key_value_heads": 4
      }
    }"#;
    let bp = build_model_blueprint("Qwen/Qwen2.5-VL-7B", "HF", vlm_cfg, None, None, None).unwrap();
    assert_eq!(bp.hidden_size, 3584);
    assert_eq!(bp.num_hidden_layers, 28);
    assert!(bp.multimodal_note.is_some());
    assert!(bp.vision.is_some());
  }

  #[test]
  fn parses_llama4_scout_irope_vision_and_shared_expert() {
    let cfg = r#"{
      "architectures": ["Llama4ForConditionalGeneration"],
      "model_type": "llama4",
      "vision_config": {
        "model_type": "llama4_vision_model",
        "hidden_size": 1408,
        "intermediate_size": 5632,
        "num_hidden_layers": 34,
        "num_attention_heads": 16,
        "image_size": 336,
        "patch_size": 14
      },
      "text_config": {
        "model_type": "llama4_text",
        "vocab_size": 202048,
        "hidden_size": 5120,
        "intermediate_size": 8192,
        "intermediate_size_mlp": 16384,
        "num_hidden_layers": 48,
        "num_attention_heads": 40,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "num_local_experts": 16,
        "num_experts_per_tok": 1,
        "interleave_moe_layer_step": 1,
        "use_qk_norm": true,
        "no_rope_layers": [1, 1, 1, 0, 1, 1, 1, 0]
      }
    }"#;
    let bp = build_model_blueprint("meta-llama/Llama-4-Scout-17B-16E", "HF", cfg, None, None, None)
      .unwrap();
    assert_eq!(bp.family, ArchFamily::Llama4Moe);
    assert!(bp.vision.is_some());
    assert!(bp.params.total_params > 107_000_000_000 && bp.params.total_params < 110_000_000_000);
    assert!(bp.params.active_params > 17_000_000_000 && bp.params.active_params < 19_000_000_000);
    match &bp.segments[0].attention {
      AttentionSpec::StandardGqa { nope_layer_interval, qk_norm, .. } => {
        assert_eq!(*nope_layer_interval, Some(4));
        assert!(*qk_norm);
      }
      other => panic!("unexpected attention: {other:?}"),
    }
  }

  #[test]
  fn parses_gemma3_27b_hybrid_swa_and_glm4_9b_extreme_gqa() {
    let gemma_cfg = r#"{
      "architectures": ["Gemma3ForConditionalGeneration"],
      "model_type": "gemma3",
      "vision_config": {
        "model_type": "siglip_vision_model",
        "hidden_size": 1152,
        "intermediate_size": 4304,
        "num_hidden_layers": 27,
        "num_attention_heads": 16,
        "image_size": 896,
        "patch_size": 14
      },
      "text_config": {
        "model_type": "gemma3_text",
        "vocab_size": 262208,
        "hidden_size": 5376,
        "intermediate_size": 21504,
        "num_hidden_layers": 62,
        "num_attention_heads": 32,
        "num_key_value_heads": 16,
        "head_dim": 128,
        "sliding_window": 1024,
        "sliding_window_pattern": 6,
        "hidden_activation": "gelu_pytorch_tanh"
      }
    }"#;
    let g =
      build_model_blueprint("google/gemma-3-27b-it", "HF", gemma_cfg, None, None, None).unwrap();
    assert_eq!(g.family, ArchFamily::GemmaHybrid);
    assert!(g.tie_word_embeddings);
    assert_eq!(g.segments[0].norms_per_layer, 4);
    assert!(g.params.total_params > 27_000_000_000 && g.params.total_params < 28_000_000_000);
    // At 128K context, 5/6 layers cap KV cache at W=1024, saving huge VRAM vs non-windowed GQA!
    let est = g.estimate_vram(PrecisionPreset::Bf16, 131_072, 1);
    assert!(est.kv_cache_bytes < est.mha_baseline_kv_bytes * 0.15);

    let glm_cfg = r#"{
      "architectures": ["ChatGLMModel"],
      "model_type": "chatglm",
      "padded_vocab_size": 151552,
      "hidden_size": 4096,
      "ffn_hidden_size": 13696,
      "num_layers": 40,
      "num_attention_heads": 32,
      "multi_query_group_num": 2,
      "kv_channels": 128,
      "add_qkv_bias": true,
      "rotary_percent": 0.5,
      "tie_word_embeddings": false
    }"#;
    let glm =
      build_model_blueprint("THUDM/glm-4-9b-chat", "HF", glm_cfg, None, None, None).unwrap();
    assert_eq!(glm.family, ArchFamily::GlmGqa);
    assert!(glm.params.total_params > 9_300_000_000 && glm.params.total_params < 9_500_000_000);
    assert!((glm.segments[0].attention.kv_compression_ratio() - 16.0).abs() < 1e-6);
  }
}
