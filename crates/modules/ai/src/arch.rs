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
#[cfg(feature = "server")]
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
  /// ASR 语音识别架构（Whisper Encoder-Decoder / Qwen2-Audio / SenseVoice / CTC）
  AudioAsr,
  /// TTS / Omni 原生多阶段声学架构（Qwen2.5-Omni Thinker-Talker-Token2Wav / Parler-TTS / SpeechT5 / VITS）
  SpeechTtsOmni,
  /// 混合线性注意力 / 状态空间模型（Qwen3-Next Gated DeltaNet / Mamba-2 / Jamba / RWKV-7 / MiniMax-01）
  HybridLinearSsm,
  /// 双向编码器与向量表征 / 重排序模型（BGE-M3 / BERT / XLM-RoBERTa / ModernBERT / Qwen3-Embedding）
  EncoderEmbedding,
  /// 图像 / 视频扩散与流匹配多组件生成管线（FLUX.1 MMDiT / SD3.5 / Wan2.1 3D DiT / HunyuanVideo / SDXL UNet）
  DiffusionPipeline,
  /// 其它标准 Decoder-Only Transformer
  GenericTransformer,
  /// 未提供标准 Transformer 超参、仅由 Safetensors 权重树还原的自定义模型
  SafetensorsTree,
  /// 由外部声明式 `graph.json` 驱动的全谱系架构图（YOLO / U-Net / SAM / 经典网络等）
  DeclarativeGraph,
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
      Self::AudioAsr => "ASR Audio-Text",
      Self::SpeechTtsOmni => "TTS / Omni Speech",
      Self::HybridLinearSsm => "Hybrid Linear / SSM",
      Self::EncoderEmbedding => "Encoder / Embedding",
      Self::DiffusionPipeline => "Diffusion / DiT Pipeline",
      Self::GenericTransformer => "Transformer Decoder",
      Self::SafetensorsTree => "Safetensors Weight Tree",
      Self::DeclarativeGraph => "Architecture Graph",
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

/// 语音声学前端与音频编码塔（Audio Frontend + Acoustic Encoder）规格。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AudioSpec {
  pub model_type: String,
  pub num_mel_bins: u64,
  pub sampling_rate_hz: u64,
  pub hidden_size: u64,
  pub num_layers: u64,
  pub num_heads: u64,
  pub ffn_dim: u64,
  pub max_source_positions: u64,
  pub estimated_params: u64,
}

/// 模型末端输出阶段规格（区分自回归词表头、句向量池化头、重排序打分头与多阶段声学声码器）。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum OutputHeadSpec {
  /// 标准自回归词表投影头 `lm_head [D -> V]`
  #[default]
  CausalLmHead,
  /// 句向量池化与归一化头（`CLS` / `Mean` / `Last-Token` + L2 Normalize → `[B, output_dim]`）
  EmbeddingPool { pooling_mode: String, output_dim: u64, normalize: bool },
  /// 交叉编码器重排序/分类打分头（`Classifier / Score [D -> num_labels]`）
  SequenceScorer { num_labels: u64 },
  /// 多阶段语音合成 / Omni 声学声码管线（如 `Talker AR -> Flow-Matching DiT + BigVGAN` 或 `RVQ -> DAC`）
  SpeechSynthesizer {
    talker_summary: Option<String>,
    codec_summary: String,
    num_codebooks: u64,
    sampling_rate_hz: u64,
    estimated_params: u64,
  },
  /// 扩散生成管线潜空间反块化与空间/时空 VAE 解码器（`Unpatchify -> AutoencoderKL / 3D Causal VAE -> RGB`）
  LatentDecoder {
    vae_class: String,
    latent_channels: u64,
    spatial_compression: u64,
    temporal_compression: u64,
    is_video: bool,
    estimated_params: u64,
  },
}

/// 扩散/流匹配多组件管线中的单个子模块规格（Text Encoder / Denoiser DiT·UNet / Latent VAE / Scheduler）。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PipelineComponentSpec {
  pub slot: String,
  pub class_name: String,
  /// `"conditioner"` | `"denoiser"` | `"vae"` | `"scheduler"`
  pub role: String,
  pub spec_summary: String,
  pub estimated_params: u64,
}

/// 图像 / 视频扩散与流匹配（Flow-Matching / DDPM）多阶段生成管线规格。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DiffusionPipelineSpec {
  pub pipeline_class: String,
  pub is_video: bool,
  pub scheduler_summary: String,
  pub denoiser_kind: String,
  pub latent_channels: u64,
  pub spatial_compression: u64,
  pub temporal_compression: u64,
  pub patch_summary: String,
  pub default_latent_shape: String,
  pub components: Vec<PipelineComponentSpec>,
}

fn default_true() -> bool {
  true
}

/// 注意力或序列状态混合子层精确规格。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AttentionSpec {
  /// 标准 MHA / MQA / GQA（兼容双向全注意力、Seq2Seq Cross-Attention、QK-Norm、SWA 及 iRoPE）
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
    /// 是否使用因果掩码（BERT / RoBERTa 等双向编码器为 `false`）
    #[serde(default = "default_true")]
    is_causal: bool,
    /// 是否附带编码器-解码器交叉注意力 `encoder_attn`（如 Whisper / Parler-TTS / SpeechT5 解码层）
    #[serde(default)]
    has_cross_attn: bool,
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
  /// 线性注意力 / 选择性状态空间模型（Gated DeltaNet / Mamba-2 / Mamba / RWKV-7 / Lightning Linear）：
  /// 推理期仅维护 O(1) 恒定大小的循环状态与短卷积缓存，随序列长度 S 零增长。
  LinearRecurrent {
    kind: String,
    num_heads: u64,
    head_dim: u64,
    state_dim: u64,
    conv_kernel: u64,
    expand_factor: u64,
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
        is_causal,
        has_cross_attn,
        ..
      } => {
        let base_kind = if *num_q_heads == *num_kv_heads {
          "MHA"
        } else if *num_kv_heads == 1 {
          "MQA"
        } else {
          "GQA"
        };
        let kind = if *is_causal { base_kind.to_string() } else { format!("Bi-{base_kind}") };
        let norm_tag = if *qk_norm { " + QK-Norm" } else { "" };
        let cross_tag = if *has_cross_attn { " + CrossAttn" } else { "" };
        let extra = if let (Some(w), Some(pat)) = (sliding_window, sliding_window_pattern) {
          format!(" · {}:1 SWA({w})", pat.saturating_sub(1))
        } else if let Some(iv) = nope_layer_interval {
          format!(" · iRoPE({}:1)", iv.saturating_sub(1))
        } else if let Some(w) = sliding_window {
          format!(" · SWA({w})")
        } else {
          String::new()
        };
        format!(
          "{kind} ({num_q_heads}Q:{num_kv_heads}KV × {head_dim}d{norm_tag}{cross_tag}{extra})"
        )
      }
      Self::DeepSeekMla { num_heads, kv_lora_rank, qk_rope_head_dim, .. } => {
        format!("MLA ({num_heads}H, c_KV={kv_lora_rank}d + k_R={qk_rope_head_dim}d)")
      }
      Self::LinearRecurrent { kind, num_heads, head_dim, state_dim, .. } => {
        format!("{kind} ({num_heads}H×{head_dim}d, O(1) state={state_dim}d)")
      }
    }
  }

  /// 单层单个 Token 在推理 KV Cache 中随序列长度增长需要缓存的标量元素个数（`LinearRecurrent` 为 `0`）。
  pub fn cached_elements_per_token(&self) -> u64 {
    match self {
      Self::StandardGqa { num_kv_heads, head_dim, is_causal, .. } => {
        if *is_causal {
          2 * num_kv_heads * head_dim
        } else {
          0
        }
      }
      Self::DeepSeekMla { kv_lora_rank, qk_rope_head_dim, .. } => kv_lora_rank + qk_rope_head_dim,
      Self::LinearRecurrent { .. } => 0,
    }
  }

  /// 单层单条序列在推理期需常驻的固定循环状态元素个数（与上下文长度 S 无关的 O(1) State Cache）。
  pub fn fixed_state_elements_per_seq(&self, hidden_size: u64) -> u64 {
    match self {
      Self::LinearRecurrent {
        num_heads, head_dim, state_dim, conv_kernel, expand_factor, ..
      } => {
        let d_inner = (num_heads * head_dim).max((*expand_factor).max(1) * hidden_size);
        num_heads * head_dim * state_dim + d_inner * conv_kernel.saturating_sub(1)
      }
      _ => 0,
    }
  }

  /// 等效同头数标准 MHA 单层单 Token 需要的 KV 元素数（用于计算 KV 压缩倍率）。
  pub fn mha_baseline_elements_per_token(&self) -> u64 {
    match self {
      Self::StandardGqa { num_q_heads, head_dim, .. } => 2 * num_q_heads * head_dim,
      Self::DeepSeekMla { num_heads, v_head_dim, .. } => 2 * num_heads * v_head_dim,
      Self::LinearRecurrent { num_heads, head_dim, .. } => 2 * num_heads * head_dim,
    }
  }

  /// 相比标准 MHA 的 KV Cache 压缩倍数（如 32Q:8KV 为 4.0×，DeepSeek-V3 MLA 为 56.89×；
  /// 对于 `LinearRecurrent`，以标准 32K 上下文下的固定循环状态对比同维度 MHA 计算等效压缩比）。
  pub fn kv_compression_ratio(&self) -> f64 {
    match self {
      Self::LinearRecurrent { state_dim, .. } => 65_536.0 / ((*state_dim).max(1) as f64),
      _ => {
        let actual = self.cached_elements_per_token();
        if actual == 0 {
          return 1.0;
        }
        (self.mha_baseline_elements_per_token() as f64) / (actual as f64)
      }
    }
  }

  /// 单层 Attention / SSM 子块的参数量（全部激活）。
  pub fn params_per_layer(&self, hidden_size: u64) -> u64 {
    match self {
      Self::StandardGqa {
        num_q_heads,
        num_kv_heads,
        head_dim,
        qkv_bias,
        o_bias,
        qk_norm,
        has_cross_attn,
        ..
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
        if *has_cross_attn {
          p += 4 * hidden_size * hidden_size;
          if *qkv_bias {
            p += 3 * hidden_size;
          }
          if *o_bias {
            p += hidden_size;
          }
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
      Self::LinearRecurrent {
        num_heads, head_dim, state_dim, conv_kernel, expand_factor, ..
      } => {
        let d_inner = (num_heads * head_dim).max((*expand_factor).max(1) * hidden_size);
        if *expand_factor >= 2 {
          3 * hidden_size * d_inner + 2 * hidden_size * state_dim + d_inner * conv_kernel
        } else {
          4 * hidden_size * d_inner + d_inner * conv_kernel
        }
      }
    }
  }
}

/// 前馈网络（Dense FFN、非门控 2 矩阵 MLP 或 Sparse MoE）子层精确规格。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum FfnSpec {
  /// 稠密前馈网络（`is_gated=true` 为 3 矩阵 SwiGLU/GeGLU；`is_gated=false` 为 2 矩阵标准 MLP；`intermediate_size=0` 表示纯 Mamba 无独立 MLP）
  DenseGated {
    intermediate_size: u64,
    act_fn: String,
    bias: bool,
    #[serde(default)]
    fused_gate_up: bool,
    #[serde(default = "default_true")]
    is_gated: bool,
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
      Self::DenseGated { intermediate_size, act_fn, fused_gate_up, is_gated, .. } => {
        if *intermediate_size == 0 {
          "Integrated in SSM Mixer (No Separate MLP)".to_string()
        } else if *fused_gate_up {
          format!("Fused GateUp FFN ({act_fn}, I={intermediate_size})")
        } else if !*is_gated {
          format!("Standard 2-Layer MLP ({act_fn}, I={intermediate_size})")
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
      Self::DenseGated { intermediate_size, bias, is_gated, .. } => {
        if *intermediate_size == 0 {
          return (0, 0);
        }
        let matrices = if *is_gated { 3 } else { 2 };
        let mut p = matrices * hidden_size * intermediate_size;
        if *bias {
          p += (matrices - 1) * intermediate_size + hidden_size;
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
  #[serde(default)]
  pub audio_and_speech: u64,
  pub embedding: u64,
  pub attention: u64,
  pub dense_ffn: u64,
  pub moe_routed: u64,
  pub moe_shared: u64,
  pub norms_and_routers: u64,
  pub lm_head: u64,
  pub mtp_head: u64,
  /// 模型总参数量（主干 + Embedding + 输出头 + 多模态视觉/语音子系统，不含训练辅助 MTP 时可对照 `mtp_head`）
  pub total_params: u64,
  /// 每个 Token 前向推理实际激活的参数量
  pub active_params: u64,
}

/// 架构图中可点击检视的模块焦点。
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum InspectorFocus {
  #[default]
  Overview,
  Vision,
  Audio,
  Embedding,
  Attention,
  FfnOrMoe,
  NormAndResidual,
  LmHead,
  MtpHead,
}

impl InspectorFocus {
  pub fn from_attr(raw: &str) -> Self {
    match raw.trim().to_ascii_lowercase().as_str() {
      "overview" | "macro" | "summary" | "all" => Self::Overview,
      "vision" | "vit" | "siglip" => Self::Vision,
      "audio" | "whisper" | "speech" | "mel" => Self::Audio,
      "embed" | "embedding" => Self::Embedding,
      "attn" | "attention" | "mla" | "gqa" | "ssm" | "mamba" => Self::Attention,
      "ffn" | "mlp" | "moe" | "experts" => Self::FfnOrMoe,
      "norm" | "rmsnorm" | "residual" => Self::NormAndResidual,
      "head" | "lm_head" | "logits" | "pool" | "pooling" | "vocoder" => Self::LmHead,
      "mtp" => Self::MtpHead,
      _ => Self::Attention,
    }
  }

  /// 是否属于具体 Transformer 层块内部的子模块（仅此类模块在标题中显示 `Layer #N`）。
  pub fn is_layer_scoped(self) -> bool {
    matches!(self, Self::Attention | Self::FfnOrMoe | Self::NormAndResidual)
  }

  pub fn stage_filter(&self) -> &'static str {
    match self {
      Self::Overview => "all",
      Self::Vision => "vision",
      Self::Audio => "audio",
      Self::Embedding => "embedding",
      Self::Attention => "attention",
      Self::FfnOrMoe => "ffn_or_moe",
      Self::NormAndResidual => "norm",
      Self::LmHead => "lm_head",
      Self::MtpHead => "mtp",
    }
  }
}

/// 附属配置文件裁剪快照（如 `generation_config.json`、`tokenizer_config.json`、`preprocessor_config.json`、`1_Pooling/config.json`）。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AuxConfigSnapshot {
  pub filename: String,
  pub raw_json_pretty: String,
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
  /// 若外层为多模态（`vision_config` / `audio_config` / Omni 多阶段），记录摘要提示
  pub multimodal_note: Option<String>,
  #[serde(default)]
  pub vision: Option<VisionSpec>,
  #[serde(default)]
  pub audio: Option<AudioSpec>,
  #[serde(default)]
  pub diffusion: Option<DiffusionPipelineSpec>,
  #[serde(default)]
  pub output_head: OutputHeadSpec,

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
  /// 附属配置文件裁剪快照（`generation_config.json` / `tokenizer_config.json` / `preprocessor_config.json` / `1_Pooling/config.json`）
  #[serde(default)]
  pub aux_configs: Vec<AuxConfigSnapshot>,
  /// 可选声明式张量维流图紧凑线编码（由服务端读取 `models/<slug>/graph.json` 校验并编码，客户端零 serde 膨胀解码）
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub declarative_graph_wire: Option<String>,
}

impl ModelBlueprint {
  /// 是否为基于外部声明式 `graph.json` 的架构图模型（无标准 LLM `config.json`）。
  pub fn is_declarative_graph(&self) -> bool {
    self.family == ArchFamily::DeclarativeGraph || self.declarative_graph_wire.is_some()
  }
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

  /// 是否包含线性注意力 / SSM 恒定循环状态层。
  pub fn has_linear_recurrent(&self) -> bool {
    self.segments.iter().any(|s| matches!(s.attention, AttentionSpec::LinearRecurrent { .. }))
  }

  /// 是否为单次前向编码/打分模型（Embedding / Reranker，推理期无需驻留自回归 KV Cache）。
  pub fn is_single_pass_encoder(&self) -> bool {
    matches!(
      self.output_head,
      OutputHeadSpec::EmbeddingPool { .. } | OutputHeadSpec::SequenceScorer { .. }
    )
  }

  /// 是否为图像/视频扩散生成多组件管线（推理期执行迭代潜空间去噪而非自回归 KV 缓存增长）。
  pub fn is_diffusion_pipeline(&self) -> bool {
    self.family == ArchFamily::DiffusionPipeline || self.diffusion.is_some()
  }

  /// 工作台未指定初始焦点时的默认视图：先呈现全模型端到端总体架构，再按需钻取算子细节。
  pub fn default_focus(&self) -> InspectorFocus {
    InspectorFocus::Overview
  }

  /// 该模型最具代表性的算子级子图焦点（用于从总览钻取或默认实验室推荐）。
  pub fn primary_operator_focus(&self) -> InspectorFocus {
    match self.family {
      ArchFamily::AudioAsr if self.audio.is_some() => InspectorFocus::Audio,
      ArchFamily::SpeechTtsOmni => InspectorFocus::LmHead,
      ArchFamily::QwenMoe | ArchFamily::MixtralMoe | ArchFamily::Llama4Moe => {
        InspectorFocus::FfnOrMoe
      }
      _ => InspectorFocus::Attention,
    }
  }

  /// 是否存在多个层段且所有层段共用相同的注意力机制（例如 DeepSeek-V3 全部层段均使用 MLA，仅 FFN 异构）。
  pub fn has_uniform_attention(&self) -> bool {
    let Some(first) = self.segments.first() else {
      return false;
    };
    if self.segments.len() <= 1 {
      return false;
    }
    let first_lbl = first.attention.short_label();
    self.segments.iter().skip(1).all(|s| s.attention.short_label() == first_lbl)
  }

  /// 点击左侧某个层段时，自动选择最能体现该层段架构特征的算子焦点。
  pub fn recommended_focus_for_segment(&self, _seg_idx: usize) -> InspectorFocus {
    if self.segments.len() > 1 {
      let uniform_attn = self.has_uniform_attention();
      let uniform_ffn = self
        .segments
        .first()
        .map(|first| {
          let lbl = first.ffn.short_label();
          self.segments.iter().skip(1).all(|s| s.ffn.short_label() == lbl)
        })
        .unwrap_or(true);
      if uniform_attn && !uniform_ffn {
        return InspectorFocus::FfnOrMoe;
      }
      if !uniform_attn {
        return InspectorFocus::Attention;
      }
    }
    self.primary_operator_focus()
  }

  /// 按当前选中的模块焦点筛选匹配的 L2 Safetensors 真实权重模式。
  pub fn matching_tensors(&self, focus: InspectorFocus) -> Vec<&TensorPatternEntry> {
    if focus == InspectorFocus::Overview {
      return Vec::new();
    }
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

  /// 按当前选中的模块焦点与具体层号筛选 L2 Safetensors 权重，区分异构模型中的 Dense FFN 层段与 Sparse MoE 层段。
  pub fn matching_tensors_for_layer(
    &self,
    focus: InspectorFocus,
    layer_idx: u64,
  ) -> Vec<&TensorPatternEntry> {
    if focus == InspectorFocus::Overview {
      return Vec::new();
    }
    let Some(ref st) = self.safetensors else {
      return Vec::new();
    };
    if focus == InspectorFocus::FfnOrMoe {
      if let Some(seg) = self.segment_for_layer(layer_idx) {
        let preferred_stage = match seg.ffn {
          FfnSpec::DenseGated { .. } => "ffn",
          FfnSpec::SparseMoe { .. } => "moe",
        };
        let exact: Vec<&TensorPatternEntry> =
          st.patterns.iter().filter(|p| p.stage == preferred_stage).collect();
        if !exact.is_empty() {
          return exact;
        }
      }
    }
    self.matching_tensors(focus)
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 推理显存（VRAM）与 KV / Recurrent State Cache 精算器
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
  /// 线性注意力 / SSM 层常驻的 O(1) 循环状态缓存字节数（与上下文长度 `context_len` 无关）
  pub state_cache_bytes: f64,
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
    let mut state_elems_total: f64 = 0.0;
    let mut mha_elems_total: f64 = 0.0;

    if !self.is_single_pass_encoder() && !self.is_diffusion_pipeline() {
      let cross_frames = self.audio.as_ref().map(|a| a.max_source_positions).unwrap_or(1500) as f64;
      for seg in &self.segments {
        let n_layers = seg.layer_count();
        let elems_per_tok = seg.attention.cached_elements_per_token() as f64;
        match seg.attention {
          AttentionSpec::LinearRecurrent { .. } => {
            let fixed_state = seg.attention.fixed_state_elements_per_seq(self.hidden_size) as f64;
            state_elems_total += (n_layers as f64) * b * fixed_state;
          }
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
          AttentionSpec::StandardGqa { has_cross_attn: true, .. } => {
            kv_elems_total +=
              (n_layers as f64) * b * ((context_len as f64) + cross_frames) * elems_per_tok;
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
    }

    let kv_cache_bytes = kv_elems_total * kv_elem_bytes;
    let state_cache_bytes = state_elems_total * 2.0;
    let mha_baseline_kv_bytes = mha_elems_total * kv_elem_bytes;

    // 运行时激活缓冲区与 CUDA Graph / FlashAttention Workspace 预留：
    // 对扩散管线，`context_len` 代表时空潜空间 Token 序列长度（如图像 4096、视频 32K~64K），含双流/CFG 与 VAE 解码峰值缓冲
    let active_workspace = if self.is_diffusion_pipeline() {
      let latent_tokens = context_len.max(1024) as f64;
      let vae_mult = if self.diffusion.as_ref().is_some_and(|d| d.is_video) { 18.0 } else { 10.0 };
      (self.hidden_size.max(1024) as f64) * b * latent_tokens * vae_mult + 1024.0 * 1024.0 * 1024.0
    } else {
      (self.hidden_size as f64) * b * (context_len.min(4096) as f64) * 4.0 + 512.0 * 1024.0 * 1024.0
    };
    let activation_overhead_bytes = active_workspace.max(weights_bytes * 0.04);

    VramEstimate {
      weights_bytes,
      kv_cache_bytes,
      state_cache_bytes,
      mha_baseline_kv_bytes,
      activation_overhead_bytes,
      total_bytes: weights_bytes + kv_cache_bytes + state_cache_bytes + activation_overhead_bytes,
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

/// 跨端传输编解码器：将嵌套枚举较多的 [`ModelBlueprint`] 统一通过 `serde_json` 字节切片编解码，
/// 消除 `use_server_future` (`ciborium` CBOR) 与 `#[post]` (`serde_json`) 对同一棵类型树的重复单态化膨胀。
pub mod blueprint_wire {
  use serde::de::Error as _;
  use serde::{Deserialize, Deserializer, Serializer};

  use super::ModelBlueprint;

  pub fn serialize<S: Serializer>(bp: &ModelBlueprint, serializer: S) -> Result<S::Ok, S::Error> {
    let raw = serde_json::to_string(bp).map_err(serde::ser::Error::custom)?;
    serializer.serialize_str(&raw)
  }

  pub fn deserialize<'de, D: Deserializer<'de>>(
    deserializer: D,
  ) -> Result<ModelBlueprint, D::Error> {
    let raw = String::deserialize(deserializer)?;
    serde_json::from_slice(raw.as_bytes()).map_err(D::Error::custom)
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 核心配置解析逻辑
// ─────────────────────────────────────────────────────────────────────────────

/// 可选附属配置文件集合（与主 `config.json` 协同建模，仅服务端解析）。
#[cfg(feature = "server")]
#[derive(Debug, Clone, Copy, Default)]
pub struct AuxConfigFiles<'a> {
  pub generation_json: Option<&'a str>,
  pub tokenizer_json: Option<&'a str>,
  /// `preprocessor_config.json`（提供音频采样率、Mel 频带数或视觉分辨率/Patch 尺寸）
  pub preprocessor_json: Option<&'a str>,
  /// Sentence-Transformers `1_Pooling/config.json`（提供 CLS / Mean / Last-Token 池化策略）
  pub pooling_json: Option<&'a str>,
}

/// 从原始 `config.json`（及可选附属配置、Safetensors 摘要）构建完整的 [`ModelBlueprint`]。
#[cfg(feature = "server")]
pub fn build_model_blueprint(
  repo_id: &str,
  source_label: &str,
  config_json: &str,
  generation_json: Option<&str>,
  tokenizer_json: Option<&str>,
  safetensors: Option<SafetensorsSummary>,
) -> Result<ModelBlueprint, String> {
  build_model_blueprint_with_aux(
    repo_id,
    source_label,
    config_json,
    AuxConfigFiles { generation_json, tokenizer_json, preprocessor_json: None, pooling_json: None },
    safetensors,
  )
}

/// 支持完整附属配置（含 `preprocessor_config.json` 与 `1_Pooling/config.json`）的架构推导入口。
#[cfg(feature = "server")]
pub fn build_model_blueprint_with_aux(
  repo_id: &str,
  source_label: &str,
  config_json: &str,
  aux: AuxConfigFiles<'_>,
  safetensors: Option<SafetensorsSummary>,
) -> Result<ModelBlueprint, String> {
  let root: Value =
    serde_json::from_str(config_json).map_err(|e| format!("解析 config.json 失败: {e}"))?;
  let Some(root_obj) = root.as_object() else {
    return Err("config.json 根节点必须是 JSON 对象".to_string());
  };

  // 若输入本身是 Diffusers `model_index.json` 或独立 DiT / MMDiT / UNet 组件 `config.json`，自动转入扩散管线解析器（仅服务端编译以保持客户端 WASM 轻量）
  #[cfg(feature = "server")]
  if is_diffusion_root_json(root_obj) {
    return build_diffusion_from_single_json(
      repo_id,
      source_label,
      root_obj,
      config_json,
      safetensors,
    );
  }

  let raw_config_pretty =
    serde_json::to_string_pretty(&root).unwrap_or_else(|_| config_json.to_string());

  let preproc = parse_preprocessor_meta(aux.preprocessor_json);

  // 兼容多模态外壳：支持顶层 `thinker_config`（如 Qwen2.5-Omni）及 `vision_config` / `audio_config` / `text_config` / `decoder`
  let thinker_obj = root_obj.get("thinker_config").and_then(Value::as_object);
  let mm_container = thinker_obj.unwrap_or(root_obj);
  let vision_cfg_obj = mm_container
    .get("vision_config")
    .or_else(|| root_obj.get("vision_config"))
    .and_then(Value::as_object);
  let audio_cfg_obj = mm_container
    .get("audio_config")
    .or_else(|| root_obj.get("audio_config"))
    .and_then(Value::as_object);

  let mut multimodal_note: Option<String> = None;
  let cfg_obj = if mm_container.contains_key("hidden_size")
    || mm_container.contains_key("n_embd")
    || mm_container.contains_key("d_model")
  {
    if let Some(vc) = vision_cfg_obj {
      let vt = vc.get("model_type").and_then(Value::as_str).unwrap_or("vision_tower");
      let vd = vc.get("hidden_size").and_then(Value::as_u64).unwrap_or(0);
      multimodal_note =
        Some(format!("内含多模态视觉编码器 ({vt}, hidden={vd})，支持点击展开视觉支路"));
    }
    mm_container
  } else if let Some(sub) = mm_container
    .get("text_config")
    .or_else(|| mm_container.get("llm_config"))
    .or_else(|| mm_container.get("language_config"))
    .or_else(|| mm_container.get("decoder"))
    .and_then(Value::as_object)
  {
    if thinker_obj.is_some() {
      multimodal_note = Some(
        "原生全模态 Thinker-Talker 架构（含视觉塔 + 音频塔 + 语言主干 + 声码合成流）".to_string(),
      );
    } else if let Some(vc) = vision_cfg_obj {
      let vision_tag = vc.get("model_type").and_then(Value::as_str).unwrap_or("Vision Encoder");
      multimodal_note = Some(format!(
        "多模态复合架构（含 {vision_tag} 视觉编码器 + 语言主干），可点击视觉支路展开细节"
      ));
    } else if let Some(ac) = audio_cfg_obj {
      let audio_tag = ac.get("model_type").and_then(Value::as_str).unwrap_or("Audio Encoder");
      multimodal_note = Some(format!(
        "语音-文本复合架构（含 {audio_tag} 声学编码塔 + 语言主干），可点击音频支路展开细节"
      ));
    }
    sub
  } else {
    mm_container
  };

  let architectures: Vec<String> = root_obj
    .get("architectures")
    .and_then(Value::as_array)
    .map(|arr| arr.iter().filter_map(Value::as_str).map(str::to_string).collect())
    .unwrap_or_default();

  let root_model_type = root_obj.get("model_type").and_then(Value::as_str).unwrap_or("");
  let model_type = if !root_model_type.is_empty() && thinker_obj.is_some() {
    root_model_type.to_string()
  } else {
    cfg_obj
      .get("model_type")
      .or_else(|| root_obj.get("model_type"))
      .and_then(Value::as_str)
      .unwrap_or("unknown")
      .to_string()
  };

  let torch_dtype = cfg_obj
    .get("torch_dtype")
    .or_else(|| root_obj.get("torch_dtype"))
    .or_else(|| cfg_obj.get("dtype"))
    .and_then(Value::as_str)
    .unwrap_or("bfloat16")
    .to_string();

  let quantization_summary =
    extract_quantization_summary(root_obj).or_else(|| extract_quantization_summary(cfg_obj));

  let (tokenizer_class, tok_vocab) = parse_tokenizer_meta(aux.tokenizer_json);
  let (default_temperature, default_top_p) = parse_generation_meta(aux.generation_json);
  let aux_configs = collect_aux_config_snapshots(aux);

  let hidden_opt =
    u64_any(cfg_obj, &["hidden_size", "n_embd", "d_model", "dim", "hidden_channels"]);
  let layers_opt = u64_any(
    cfg_obj,
    &["num_hidden_layers", "n_layer", "num_layers", "decoder_layers", "encoder_layers", "depth"],
  );

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

  let vision = vision_cfg_obj.and_then(|vc| parse_vision_spec(vc, hidden_size, &preproc));

  let arch_joined =
    format!("{} {} {}", architectures.join(" "), model_type, root_model_type).to_ascii_lowercase();
  let repo_lc = repo_id.to_ascii_lowercase();

  let is_whisper_like = arch_joined.contains("whisper")
    || (cfg_obj.contains_key("encoder_layers")
      && cfg_obj.contains_key("decoder_layers")
      && (cfg_obj.contains_key("num_mel_bins") || arch_joined.contains("speecht5")));
  let is_ctc_asr = arch_joined.contains("wav2vec2")
    || arch_joined.contains("hubert")
    || arch_joined.contains("sensevoice")
    || arch_joined.contains("wavlm")
    || arch_joined.contains("paraformer");

  let audio = parse_audio_spec(
    audio_cfg_obj,
    cfg_obj,
    hidden_size,
    is_whisper_like,
    is_ctc_asr,
    !std::ptr::eq(cfg_obj, mm_container),
    &preproc,
  );
  if multimodal_note.is_none() {
    if let Some(ref aud) = audio {
      multimodal_note = Some(format!(
        "语音编码前端：{} ({} Mel bins @ {}kHz, {}L×{}d)，可点击音频支路查看声学下采样流",
        aud.model_type,
        aud.num_mel_bins,
        aud.sampling_rate_hz / 1000,
        aud.num_layers,
        aud.hidden_size
      ));
    }
  }

  // 识别 TTS / Omni 声码输出管线 vs Embedding / Reranker 编码头
  let speech_synth = parse_speech_synthesizer(root_obj, cfg_obj, &arch_joined, &preproc);
  let is_tts_or_omni = speech_synth.is_some();
  let is_asr = !is_tts_or_omni && (is_whisper_like || is_ctc_asr || audio.is_some());

  let is_bi_encoder_arch = matches!(
    model_type.as_str(),
    "bert"
      | "roberta"
      | "xlm-roberta"
      | "modernbert"
      | "deberta"
      | "deberta-v2"
      | "electra"
      | "mpnet"
      | "jina_bert"
      | "nomic_bert"
  ) || arch_joined.contains("xlmroberta")
    || arch_joined.contains("modernbert")
    || arch_joined.contains("bertmodel");
  let is_seq_cls = architectures.iter().any(|a| a.contains("ForSequenceClassification"));
  let is_embed_repo = !is_asr
    && !is_tts_or_omni
    && vision.is_none()
    && (repo_lc.contains("embedding")
      || repo_lc.contains("bge-")
      || repo_lc.contains("gte-")
      || repo_lc.contains("e5-"));
  let is_rerank = !is_asr && !is_tts_or_omni && (is_seq_cls || repo_lc.contains("rerank"));
  let pool_cfg = parse_pooling_meta(aux.pooling_json);
  let is_embedding_model = !is_asr
    && !is_tts_or_omni
    && (is_rerank || pool_cfg.is_some() || is_bi_encoder_arch || is_embed_repo);

  let is_llama4 = arch_joined.contains("llama4");
  let is_gemma_any = arch_joined.contains("gemma");
  let is_gemma2_or_3 = arch_joined.contains("gemma2") || arch_joined.contains("gemma3");
  let is_glm = arch_joined.contains("glm") || arch_joined.contains("chatglm");
  let is_mixtral = arch_joined.contains("mixtral");
  let is_pure_mamba = matches!(model_type.as_str(), "mamba" | "mamba2" | "falcon_mamba");

  let vocab_size =
    u64_any(cfg_obj, &["padded_vocab_size", "vocab_size"]).or(tok_vocab).unwrap_or(32000);
  let max_position_embeddings = u64_any(
    cfg_obj,
    &["max_position_embeddings", "n_positions", "seq_length", "max_target_positions"],
  )
  .unwrap_or(if is_whisper_like { 448 } else { 4096 });
  let rms_norm_eps = f64_any(
    cfg_obj,
    &["rms_norm_eps", "layernorm_epsilon", "layer_norm_eps", "layer_norm_epsilon"],
  );
  let tie_word_embeddings = cfg_obj
    .get("tie_word_embeddings")
    .or_else(|| root_obj.get("tie_word_embeddings"))
    .and_then(Value::as_bool)
    .unwrap_or(is_gemma_any || is_whisper_like || is_pure_mamba);
  let hidden_act = cfg_obj
    .get("hidden_act")
    .or_else(|| cfg_obj.get("hidden_activation"))
    .or_else(|| cfg_obj.get("activation_function"))
    .and_then(Value::as_str)
    .unwrap_or(if is_bi_encoder_arch || is_whisper_like { "gelu" } else { "silu" })
    .to_string();

  let has_explicit_rope = cfg_obj.contains_key("rope_theta")
    || cfg_obj.contains_key("rope_scaling")
    || cfg_obj.contains_key("rotary_percent")
    || cfg_obj.contains_key("rotary_emb_base");
  let rope_summary = if is_pure_mamba || arch_joined.contains("rwkv") {
    "Implicit Recurrent State (No Pos Embed)".to_string()
  } else if (is_bi_encoder_arch || is_whisper_like) && !has_explicit_rope {
    format!("Learned / Sinusoidal Pos ({max_position_embeddings})")
  } else {
    extract_rope_summary(cfg_obj)
  };
  let num_nextn_predict_layers = u64_any(cfg_obj, &["num_nextn_predict_layers"]).unwrap_or(0);

  let num_q_heads =
    u64_any(cfg_obj, &["num_attention_heads", "n_head", "decoder_attention_heads", "num_heads"])
      .unwrap_or(32)
      .max(1);
  let num_kv_heads =
    u64_any(cfg_obj, &["num_key_value_heads", "num_kv_heads", "multi_query_group_num"])
      .unwrap_or(num_q_heads)
      .max(1);

  let is_causal = !(is_bi_encoder_arch || is_ctc_asr);
  let has_cross_attn = is_whisper_like || root_obj.contains_key("decoder");

  // 1. 识别 Attention 形态（MLA vs Hybrid/Pure Linear SSM vs Standard GQA）
  let kv_lora_rank = u64_any(cfg_obj, &["kv_lora_rank"]);
  let is_mla = kv_lora_rank.is_some()
    || arch_joined.contains("deepseek_v2")
    || arch_joined.contains("deepseek_v3");

  let linear_ssm_plan = detect_linear_ssm(
    cfg_obj,
    &model_type,
    &arch_joined,
    hidden_size,
    num_hidden_layers,
    num_q_heads,
  );

  let standard_attention = if let Some(kv_rank) = kv_lora_rank {
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
    let default_qkv_bias = arch_joined.contains("qwen2")
      || arch_joined.contains("chatglm")
      || is_bi_encoder_arch
      || is_whisper_like;
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
      .unwrap_or(is_bi_encoder_arch || is_whisper_like);
    let qk_norm = cfg_obj
      .get("qk_norm")
      .or_else(|| cfg_obj.get("use_qk_norm"))
      .and_then(Value::as_bool)
      .unwrap_or_else(|| {
        (arch_joined.contains("qwen3") && !arch_joined.contains("qwen3_next_no_qknorm"))
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
      is_causal,
      has_cross_attn,
    }
  };

  // 2. 识别 FFN / MoE 与纵向分层区间
  let intermediate_size = u64_any(
    cfg_obj,
    &["intermediate_size", "ffn_hidden_size", "n_inner", "decoder_ffn_dim", "ffn_dim"],
  )
  .unwrap_or(hidden_size * 4);
  let dense_mlp_inter = if linear_ssm_plan.as_ref().is_some_and(|p| !p.has_separate_ffn) {
    0
  } else {
    u64_any(cfg_obj, &["intermediate_size_mlp"]).unwrap_or(intermediate_size)
  };
  let mlp_bias = cfg_obj
    .get("mlp_bias")
    .or_else(|| cfg_obj.get("add_bias_linear"))
    .and_then(Value::as_bool)
    .unwrap_or(is_bi_encoder_arch || is_whisper_like);
  let norms_per_layer = if is_gemma2_or_3 || arch_joined.contains("glm4") {
    4
  } else if has_cross_attn {
    3
  } else if dense_mlp_inter == 0 {
    1
  } else {
    2
  };

  let is_gated_ffn = if is_whisper_like || is_ctc_asr {
    false
  } else if is_bi_encoder_arch {
    let act_lc = hidden_act.to_ascii_lowercase();
    act_lc.contains("glu") || model_type == "modernbert"
  } else {
    true
  };

  let num_routed_experts =
    u64_any(cfg_obj, &["n_routed_experts", "num_experts", "num_local_experts"]).unwrap_or(0);
  let is_moe = num_routed_experts > 1;

  let family = if is_tts_or_omni {
    ArchFamily::SpeechTtsOmni
  } else if is_asr {
    ArchFamily::AudioAsr
  } else if is_embedding_model {
    ArchFamily::EncoderEmbedding
  } else if linear_ssm_plan.is_some() {
    ArchFamily::HybridLinearSsm
  } else if is_mla {
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

  let output_head = if let Some(synth) = speech_synth {
    synth
  } else if is_rerank {
    OutputHeadSpec::SequenceScorer {
      num_labels: u64_any(cfg_obj, &["num_labels"]).unwrap_or(1).max(1),
    }
  } else if is_embedding_model {
    let (mode, out_dim) = pool_cfg.unwrap_or_else(|| {
      let default_mode = if is_bi_encoder_arch { "cls / mean" } else { "last_token" };
      (default_mode.to_string(), hidden_size)
    });
    OutputHeadSpec::EmbeddingPool { pooling_mode: mode, output_dim: out_dim, normalize: true }
  } else {
    OutputHeadSpec::CausalLmHead
  };

  let schedule_note = build_schedule_note(&standard_attention, cfg_obj);

  let dense_ffn = FfnSpec::DenseGated {
    intermediate_size: dense_mlp_inter,
    act_fn: hidden_act.clone(),
    bias: mlp_bias,
    fused_gate_up: is_glm,
    is_gated: is_gated_ffn,
  };

  let active_ffn = if is_moe {
    let top_k = u64_any(cfg_obj, &["num_experts_per_tok", "num_selected_experts"]).unwrap_or(2);
    let moe_inter = u64_any(cfg_obj, &["moe_intermediate_size"]).unwrap_or(intermediate_size);
    let n_shared_cfg = u64_any(cfg_obj, &["n_shared_experts"]).unwrap_or(0);
    let shared_inter_cfg = u64_any(cfg_obj, &["shared_expert_intermediate_size"]).unwrap_or(0);
    let (num_shared_experts, shared_intermediate_size, has_shared_gate) = if n_shared_cfg > 0 {
      (n_shared_cfg, n_shared_cfg * moe_inter, false)
    } else if shared_inter_cfg > 0 {
      (
        1,
        shared_inter_cfg,
        arch_joined.contains("qwen2_moe")
          || arch_joined.contains("qwen2moe")
          || arch_joined.contains("qwen3_next"),
      )
    } else if is_llama4 {
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

    Some(FfnSpec::SparseMoe {
      num_routed_experts,
      top_k,
      moe_intermediate_size: moe_inter,
      num_shared_experts,
      shared_intermediate_size,
      has_shared_gate,
      router_groups,
      scoring_func,
      act_fn: hidden_act.clone(),
    })
  } else {
    None
  };

  let segments = if let Some(ssm) = linear_ssm_plan {
    let seg_ffn = active_ffn.unwrap_or_else(|| dense_ffn.clone());
    if ssm.full_attn_layers == 0 {
      vec![LayerSegment {
        start_layer: 0,
        end_layer: num_hidden_layers - 1,
        title: format!("纯循环状态层栈 ({} ×{num_hidden_layers})", ssm.kind_short),
        attention: ssm.spec,
        ffn: seg_ffn,
        norms_per_layer,
        schedule_note: Some(
          "全栈 O(1) 恒定循环状态推理（零增长 KV Cache），解码时间与显存复杂度随序列长度恒定"
            .to_string(),
        ),
      }]
    } else {
      let full_n = ssm.full_attn_layers.min(num_hidden_layers - 1).max(1);
      let lin_n = num_hidden_layers - full_n;
      let ratio_note = format!(
        "{}:1 混合序列架构：{lin_n} 层 {} (O(1) 恒定 State Cache) + {full_n} 层全局注意力 (O(S) KV Cache)",
        lin_n / full_n.max(1),
        ssm.kind_short
      );
      vec![
        LayerSegment {
          start_layer: 0,
          end_layer: lin_n - 1,
          title: format!("线性/状态空间层组 ({} ×{lin_n})", ssm.kind_short),
          attention: ssm.spec,
          ffn: seg_ffn.clone(),
          norms_per_layer,
          schedule_note: Some(ratio_note.clone()),
        },
        LayerSegment {
          start_layer: lin_n,
          end_layer: num_hidden_layers - 1,
          title: format!("交替全局注意力层组 (Full Attention ×{full_n})"),
          attention: standard_attention,
          ffn: seg_ffn,
          norms_per_layer,
          schedule_note: Some(ratio_note),
        },
      ]
    }
  } else if let Some(moe_ffn) = active_ffn {
    let moe_step =
      u64_any(cfg_obj, &["interleave_moe_layer_step", "expert_layer_period"]).unwrap_or(1).max(1);
    let first_k_dense = u64_any(cfg_obj, &["first_k_dense_replace", "expert_layer_offset"])
      .or_else(|| safetensors.as_ref().and_then(|s| s.detected_first_moe_layer))
      .unwrap_or(0)
      .min(num_hidden_layers);

    let mut segs = Vec::new();
    if moe_step > 1 && first_k_dense <= 1 {
      let moe_layers = num_hidden_layers / moe_step;
      let dense_layers = num_hidden_layers.saturating_sub(moe_layers);
      if dense_layers > 0 {
        segs.push(LayerSegment {
          start_layer: 0,
          end_layer: dense_layers - 1,
          title: format!("交替稠密层组 (Interleaved Dense ×{dense_layers})"),
          attention: standard_attention.clone(),
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
          attention: standard_attention.clone(),
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
          attention: standard_attention.clone(),
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
          attention: standard_attention,
          ffn: moe_ffn,
          norms_per_layer,
          schedule_note: schedule_note.clone(),
        });
      }
    }
    segs
  } else {
    let stack_title = if !is_causal {
      "双向编码器层栈 (Bidirectional Encoder Stack)".to_string()
    } else if has_cross_attn {
      "交叉注意力解码器层栈 (Seq2Seq Decoder Stack)".to_string()
    } else {
      "Transformer 解码器层栈 (Decoder Stack)".to_string()
    };
    vec![LayerSegment {
      start_layer: 0,
      end_layer: num_hidden_layers - 1,
      title: stack_title,
      attention: standard_attention,
      ffn: dense_ffn,
      norms_per_layer,
      schedule_note,
    }]
  };

  let vision_params = vision.as_ref().map(|v| v.estimated_params).unwrap_or(0);
  let audio_and_speech_params = audio.as_ref().map(|a| a.estimated_params).unwrap_or(0)
    + match &output_head {
      OutputHeadSpec::SpeechSynthesizer { estimated_params, .. } => *estimated_params,
      _ => 0,
    };

  let params = compute_param_breakdown(
    vision_params,
    audio_and_speech_params,
    vocab_size,
    hidden_size,
    tie_word_embeddings,
    &output_head,
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
    audio,
    diffusion: None,
    output_head,
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
    aux_configs,
    declarative_graph_wire: None,
  })
}

#[cfg(feature = "server")]
fn collect_aux_config_snapshots(aux: AuxConfigFiles<'_>) -> Vec<AuxConfigSnapshot> {
  let mut out = Vec::new();
  for (filename, raw_opt) in [
    ("generation_config.json", aux.generation_json),
    ("tokenizer_config.json", aux.tokenizer_json),
    ("preprocessor_config.json", aux.preprocessor_json),
    ("1_Pooling/config.json", aux.pooling_json),
  ] {
    if let Some(raw) = raw_opt {
      if let Some(compacted) = crate::config_guide::compact_aux_json_for_snapshot(raw) {
        out.push(AuxConfigSnapshot { filename: filename.to_string(), raw_json_pretty: compacted });
      }
    }
  }
  out
}

#[cfg(feature = "server")]
#[derive(Default)]
struct PreprocessorHints {
  mel_bins: Option<u64>,
  sampling_rate: Option<u64>,
  image_size: Option<u64>,
  patch_size: Option<u64>,
}

#[cfg(feature = "server")]
mod multimodal_parsers {
  use super::*;

  pub(super) fn parse_preprocessor_meta(raw: Option<&str>) -> PreprocessorHints {
    let Some(s) = raw else {
      return PreprocessorHints::default();
    };
    let Ok(val) = serde_json::from_str::<Value>(s) else {
      return PreprocessorHints::default();
    };
    let Some(obj) = val.as_object() else {
      return PreprocessorHints::default();
    };
    let mel_bins = u64_any(obj, &["feature_size", "num_mel_bins", "n_mels"]).filter(|&v| v >= 16);
    let sampling_rate = u64_any(obj, &["sampling_rate", "sample_rate"]).filter(|&v| v >= 8000);
    let image_size = u64_any(obj, &["image_size"]).or_else(|| {
      obj
        .get("size")
        .or_else(|| obj.get("crop_size"))
        .and_then(Value::as_object)
        .and_then(|sz| u64_any(sz, &["height", "shortest_edge", "longest_edge"]))
    });
    let patch_size = u64_any(obj, &["patch_size"]);
    PreprocessorHints { mel_bins, sampling_rate, image_size, patch_size }
  }

  pub(super) fn parse_pooling_meta(raw: Option<&str>) -> Option<(String, u64)> {
    let s = raw?;
    let val = serde_json::from_str::<Value>(s).ok()?;
    let obj = val.as_object()?;
    let dim = u64_any(obj, &["word_embedding_dimension"]).unwrap_or(0);
    let mode = if obj.get("pooling_mode_lasttoken").and_then(Value::as_bool) == Some(true) {
      "last_token"
    } else if obj.get("pooling_mode_cls_token").and_then(Value::as_bool) == Some(true) {
      "cls"
    } else {
      "mean"
    };
    Some((mode.to_string(), dim))
  }

  pub(super) fn parse_vision_spec(
    vc: &serde_json::Map<String, Value>,
    text_hidden: u64,
    preproc: &PreprocessorHints,
  ) -> Option<VisionSpec> {
    let hidden_size = u64_any(vc, &["hidden_size", "embed_dim", "vision_width"])?;
    if hidden_size == 0 {
      return None;
    }
    let num_hidden_layers = u64_any(vc, &["num_hidden_layers", "depth"]).unwrap_or(24).max(1);
    let num_attention_heads =
      u64_any(vc, &["num_attention_heads", "num_heads"]).unwrap_or(16).max(1);
    let intermediate_size =
      u64_any(vc, &["intermediate_size", "mlp_dim"]).unwrap_or(hidden_size * 4);
    let image_size = u64_any(vc, &["image_size"]).or(preproc.image_size);
    let patch_size = u64_any(vc, &["patch_size"]).or(preproc.patch_size);
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

  pub(super) fn parse_audio_spec(
    audio_cfg: Option<&serde_json::Map<String, Value>>,
    cfg_obj: &serde_json::Map<String, Value>,
    text_hidden: u64,
    is_whisper_like: bool,
    is_ctc_asr: bool,
    has_projector: bool,
    preproc: &PreprocessorHints,
  ) -> Option<AudioSpec> {
    if let Some(ac) = audio_cfg {
      let d = u64_any(ac, &["d_model", "hidden_size", "encoder_hidden_size"])?;
      let layers = u64_any(ac, &["encoder_layers", "num_hidden_layers", "num_layers"])?;
      let heads =
        u64_any(ac, &["encoder_attention_heads", "num_attention_heads"]).unwrap_or(20).max(1);
      let ffn = u64_any(ac, &["encoder_ffn_dim", "intermediate_size"]).unwrap_or(d * 4);
      let mels = u64_any(ac, &["num_mel_bins", "n_mels"]).or(preproc.mel_bins).unwrap_or(128);
      let sr = u64_any(ac, &["sampling_rate"]).or(preproc.sampling_rate).unwrap_or(16_000);
      let max_pos = u64_any(ac, &["max_source_positions"]).unwrap_or(1500);
      let mt = ac.get("model_type").and_then(Value::as_str).unwrap_or("audio_encoder");
      let stem = mels * 3 * d + d * 3 * d + max_pos * d;
      let trunk = layers * (4 * d * d + 2 * d * ffn + 4 * d);
      let proj = if has_projector { d * text_hidden * 2 } else { 0 };
      return Some(AudioSpec {
        model_type: mt.to_string(),
        num_mel_bins: mels,
        sampling_rate_hz: sr,
        hidden_size: d,
        num_layers: layers,
        num_heads: heads,
        ffn_dim: ffn,
        max_source_positions: max_pos,
        estimated_params: stem + trunk + proj,
      });
    }

    if is_whisper_like {
      let d = u64_any(cfg_obj, &["d_model", "hidden_size"])?;
      let layers = u64_any(cfg_obj, &["encoder_layers"])?;
      let heads =
        u64_any(cfg_obj, &["encoder_attention_heads", "num_attention_heads"]).unwrap_or(20).max(1);
      let ffn = u64_any(cfg_obj, &["encoder_ffn_dim", "intermediate_size"]).unwrap_or(d * 4);
      let mels = u64_any(cfg_obj, &["num_mel_bins"]).or(preproc.mel_bins).unwrap_or(128);
      let sr = preproc.sampling_rate.unwrap_or(16_000);
      let max_pos = u64_any(cfg_obj, &["max_source_positions"]).unwrap_or(1500);
      let stem = mels * 3 * d + d * 3 * d + max_pos * d;
      let trunk = layers * (4 * d * d + 2 * d * ffn + 4 * d);
      return Some(AudioSpec {
        model_type: "whisper_encoder".to_string(),
        num_mel_bins: mels,
        sampling_rate_hz: sr,
        hidden_size: d,
        num_layers: layers,
        num_heads: heads,
        ffn_dim: ffn,
        max_source_positions: max_pos,
        estimated_params: stem + trunk,
      });
    }

    if is_ctc_asr {
      let mels = preproc.mel_bins.unwrap_or(80);
      let sr = preproc.sampling_rate.unwrap_or(16_000);
      let conv_params = 4_200_000u64;
      return Some(AudioSpec {
        model_type: "conv_feature_extractor".to_string(),
        num_mel_bins: mels,
        sampling_rate_hz: sr,
        hidden_size: text_hidden,
        num_layers: 7,
        num_heads: 1,
        ffn_dim: text_hidden,
        max_source_positions: 1500,
        estimated_params: conv_params,
      });
    }

    None
  }

  pub(super) fn parse_speech_synthesizer(
    root_obj: &serde_json::Map<String, Value>,
    cfg_obj: &serde_json::Map<String, Value>,
    arch_joined: &str,
    preproc: &PreprocessorHints,
  ) -> Option<OutputHeadSpec> {
    // 1) Qwen2.5-Omni / Qwen3-Omni: `talker_config` + `token2wav_config`
    if root_obj.contains_key("talker_config") || root_obj.contains_key("token2wav_config") {
      let talker_raw = root_obj.get("talker_config").and_then(Value::as_object);
      let talker_inner =
        talker_raw.and_then(|t| t.get("text_config").and_then(Value::as_object)).or(talker_raw);
      let (talker_summary, talker_params) = if let Some(tc) = talker_inner {
        let td = u64_any(tc, &["hidden_size", "d_model"]).unwrap_or(896);
        let tl = u64_any(tc, &["num_hidden_layers", "num_layers"]).unwrap_or(24);
        let ti = u64_any(tc, &["intermediate_size"]).unwrap_or(td * 4);
        let tv = u64_any(tc, &["vocab_size"]).unwrap_or(8448);
        let p = tv * td + tl * (4 * td * td + 3 * td * ti);
        (Some(format!("Talker Dual-Track AR ({tl}L×{td}d, codec_vocab={tv})")), p)
      } else {
        (None, 0)
      };
      let t2w_params = if root_obj.contains_key("token2wav_config") { 180_000_000 } else { 0 };
      return Some(OutputHeadSpec::SpeechSynthesizer {
        talker_summary,
        codec_summary: "Token2Wav (Flow-Matching DiT + BigVGAN)".to_string(),
        num_codebooks: 1,
        sampling_rate_hz: preproc.sampling_rate.unwrap_or(24_000),
        estimated_params: talker_params + t2w_params,
      });
    }

    // 2) Parler-TTS / MusicGen: `text_encoder` + `decoder` + `audio_encoder` (DAC / EnCodec)
    if arch_joined.contains("parler_tts") || arch_joined.contains("musicgen") {
      let te_info = root_obj.get("text_encoder").and_then(Value::as_object).map(|te| {
        let d = u64_any(te, &["d_model", "hidden_size"]).unwrap_or(1024);
        let l = u64_any(te, &["num_layers", "num_hidden_layers"]).unwrap_or(24);
        (format!("Prompt Encoder ({l}L×{d}d)"), l * 12 * d * d)
      });
      let ae = root_obj.get("audio_encoder").and_then(Value::as_object);
      let codebooks = u64_any(cfg_obj, &["num_codebooks"])
        .or_else(|| ae.and_then(|a| u64_any(a, &["num_codebooks"])))
        .unwrap_or(9);
      let sr =
        ae.and_then(|a| u64_any(a, &["sampling_rate"])).or(preproc.sampling_rate).unwrap_or(44_100);
      let codec_type =
        ae.and_then(|a| a.get("model_type")).and_then(Value::as_str).unwrap_or("DAC");
      let (talker_summary, te_params) = match te_info {
        Some((s, p)) => (Some(s), p),
        None => (None, 0),
      };
      return Some(OutputHeadSpec::SpeechSynthesizer {
        talker_summary,
        codec_summary: format!(
          "{} Neural Audio Codec ({codebooks} RVQ Codebooks)",
          codec_type.to_uppercase()
        ),
        num_codebooks: codebooks,
        sampling_rate_hz: sr,
        estimated_params: te_params + 75_000_000,
      });
    }

    // 3) SpeechT5 / VITS / Bark / Kokoro / F5-TTS
    if arch_joined.contains("speecht5")
      || arch_joined.contains("vits")
      || arch_joined.contains("bark")
      || arch_joined.contains("kokoro")
      || arch_joined.contains("f5tts")
      || arch_joined.contains("cosyvoice")
    {
      let sr = u64_any(cfg_obj, &["sampling_rate"]).or(preproc.sampling_rate).unwrap_or(24_000);
      return Some(OutputHeadSpec::SpeechSynthesizer {
        talker_summary: None,
        codec_summary: "Acoustic Post-Net / Flow-Matching / HiFi-GAN Vocoder".to_string(),
        num_codebooks: u64_any(cfg_obj, &["num_codebooks"]).unwrap_or(1),
        sampling_rate_hz: sr,
        estimated_params: 28_000_000,
      });
    }

    None
  }

  pub(super) struct LinearSsmPlan {
    pub(super) kind_short: &'static str,
    pub(super) spec: AttentionSpec,
    pub(super) full_attn_layers: u64,
    pub(super) has_separate_ffn: bool,
  }

  pub(super) fn detect_linear_ssm(
    cfg: &serde_json::Map<String, Value>,
    model_type: &str,
    arch_joined: &str,
    hidden_size: u64,
    num_layers: u64,
    num_q_heads: u64,
  ) -> Option<LinearSsmPlan> {
    // 1) Qwen3-Next: Gated DeltaNet + Full GQA (default 3:1)
    if model_type == "qwen3_next" || cfg.contains_key("linear_num_value_heads") {
      let v_heads = u64_any(cfg, &["linear_num_value_heads"]).unwrap_or(num_q_heads).max(1);
      let v_dim = u64_any(cfg, &["linear_value_head_dim"]).unwrap_or(128);
      let k_dim = u64_any(cfg, &["linear_key_head_dim"]).unwrap_or(128);
      let conv = u64_any(cfg, &["linear_conv_kernel_dim"]).unwrap_or(4);
      let full_attn_layers = cfg
        .get("layer_types")
        .and_then(Value::as_array)
        .map(|arr| {
          arr
            .iter()
            .filter(|v| v.as_str().is_some_and(|s| s.contains("full") || s == "attention"))
            .count() as u64
        })
        .filter(|&c| c > 0)
        .unwrap_or_else(|| {
          let interval = u64_any(cfg, &["full_attention_interval"]).unwrap_or(4).max(2);
          (num_layers / interval).max(1)
        });
      return Some(LinearSsmPlan {
        kind_short: "Gated DeltaNet",
        spec: AttentionSpec::LinearRecurrent {
          kind: "Gated DeltaNet".to_string(),
          num_heads: v_heads,
          head_dim: v_dim,
          state_dim: k_dim,
          conv_kernel: conv,
          expand_factor: 1,
        },
        full_attn_layers,
        has_separate_ffn: true,
      });
    }

    // 2) Pure Mamba / Mamba-2 / Falcon-Mamba
    if matches!(model_type, "mamba" | "mamba2" | "falcon_mamba") {
      let expand = u64_any(cfg, &["expand"]).unwrap_or(2).max(1);
      let d_inner = expand * hidden_size;
      let is_m2 = model_type == "mamba2";
      let state_dim = u64_any(cfg, &["state_size", "ssm_state_size", "d_state"])
        .unwrap_or(if is_m2 { 128 } else { 16 });
      let conv = u64_any(cfg, &["conv_kernel", "d_conv"]).unwrap_or(4);
      let head_dim = u64_any(cfg, &["head_dim", "headdim"]).unwrap_or(64).max(1);
      let n_heads = u64_any(cfg, &["n_heads"]).unwrap_or((d_inner / head_dim).max(1));
      let kind_short = if is_m2 { "Mamba-2 SSD" } else { "Mamba Selective SSM" };
      return Some(LinearSsmPlan {
        kind_short,
        spec: AttentionSpec::LinearRecurrent {
          kind: kind_short.to_string(),
          num_heads: n_heads,
          head_dim,
          state_dim,
          conv_kernel: conv,
          expand_factor: expand,
        },
        full_attn_layers: 0,
        has_separate_ffn: false,
      });
    }

    // 3) Hybrid Mamba + Transformer (Jamba / Bamba / Zamba2 / Nemotron-H / Granite-4)
    if matches!(model_type, "jamba" | "bamba" | "zamba2" | "nemotron_h" | "granite_moe_hybrid")
      || cfg.contains_key("mamba_d_state")
      || cfg.contains_key("attn_layer_period")
    {
      let expand = u64_any(cfg, &["mamba_expand", "expand"]).unwrap_or(2).max(1);
      let state_dim =
        u64_any(cfg, &["mamba_d_state", "ssm_state_size", "state_size"]).unwrap_or(16);
      let conv = u64_any(cfg, &["mamba_d_conv", "conv_kernel", "d_conv"]).unwrap_or(4);
      let head_dim = u64_any(cfg, &["mamba_headdim", "head_dim"]).unwrap_or(64).max(1);
      let n_heads = ((expand * hidden_size) / head_dim).max(1);
      let period = u64_any(cfg, &["attn_layer_period"]).unwrap_or(8).max(2);
      let full_attn_layers = cfg
        .get("hybrid_override_pattern")
        .and_then(Value::as_str)
        .map(|pat| pat.chars().filter(|&c| c == '*' || c == 'A').count() as u64)
        .filter(|&c| c > 0)
        .unwrap_or_else(|| (num_layers / period).max(1));
      return Some(LinearSsmPlan {
        kind_short: "Mamba SSM",
        spec: AttentionSpec::LinearRecurrent {
          kind: "Mamba Selective SSM".to_string(),
          num_heads: n_heads,
          head_dim,
          state_dim,
          conv_kernel: conv,
          expand_factor: 1,
        },
        full_attn_layers,
        has_separate_ffn: true,
      });
    }

    // 4) RWKV-6 / RWKV-7 Recurrent WKV
    if arch_joined.contains("rwkv") {
      let head_dim = u64_any(cfg, &["head_size", "head_dim"]).unwrap_or(64).max(1);
      let n_heads = (hidden_size / head_dim).max(1);
      let kind_short = if arch_joined.contains("rwkv7") { "RWKV-7 WKV" } else { "RWKV WKV" };
      return Some(LinearSsmPlan {
        kind_short,
        spec: AttentionSpec::LinearRecurrent {
          kind: kind_short.to_string(),
          num_heads: n_heads,
          head_dim,
          state_dim: head_dim,
          conv_kernel: 2,
          expand_factor: 1,
        },
        full_attn_layers: 0,
        has_separate_ffn: true,
      });
    }

    // 5) MiniMax-01 Lightning Linear Attention + Softmax GQA
    if arch_joined.contains("minimax") && cfg.contains_key("attn_type_list") {
      let head_dim = u64_any(cfg, &["head_dim"]).unwrap_or(hidden_size / num_q_heads).max(1);
      let full_attn_layers = cfg
        .get("attn_type_list")
        .and_then(Value::as_array)
        .map(|arr| arr.iter().filter(|v| v.as_u64() == Some(1)).count() as u64)
        .filter(|&c| c > 0)
        .unwrap_or_else(|| (num_layers / 8).max(1));
      return Some(LinearSsmPlan {
        kind_short: "Lightning Linear",
        spec: AttentionSpec::LinearRecurrent {
          kind: "Lightning Linear Attn".to_string(),
          num_heads: num_q_heads,
          head_dim,
          state_dim: head_dim,
          conv_kernel: 0,
          expand_factor: 1,
        },
        full_attn_layers,
        has_separate_ffn: true,
      });
    }

    None
  }
}

#[cfg(feature = "server")]
use multimodal_parsers::{
  detect_linear_ssm, parse_audio_spec, parse_pooling_meta, parse_preprocessor_meta,
  parse_speech_synthesizer, parse_vision_spec,
};

#[cfg(feature = "server")]
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

#[cfg(feature = "server")]
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

#[allow(clippy::too_many_arguments)]
fn compute_param_breakdown(
  vision_encoder: u64,
  audio_and_speech: u64,
  vocab_size: u64,
  hidden_size: u64,
  tie_word_embeddings: bool,
  output_head: &OutputHeadSpec,
  num_nextn_predict_layers: u64,
  segments: &[LayerSegment],
) -> ParamBreakdown {
  let embedding = vocab_size * hidden_size;
  let lm_head = match output_head {
    OutputHeadSpec::CausalLmHead | OutputHeadSpec::SpeechSynthesizer { .. } => {
      if tie_word_embeddings {
        0
      } else {
        vocab_size * hidden_size
      }
    }
    OutputHeadSpec::EmbeddingPool { output_dim, .. } => {
      if *output_dim > 0 && *output_dim != hidden_size {
        hidden_size * (*output_dim)
      } else {
        0
      }
    }
    OutputHeadSpec::SequenceScorer { num_labels } => hidden_size * (*num_labels).max(1),
    OutputHeadSpec::LatentDecoder { .. } => 0,
  };

  let mut attention = 0u64;
  let mut dense_ffn = 0u64;
  let mut moe_routed = 0u64;
  let mut moe_shared = 0u64;
  let mut norms_and_routers = 0u64;
  let mut active_trunk = 0u64;

  for seg in segments {
    let count = seg.layer_count();
    let attn_one = seg.attention.params_per_layer(hidden_size);
    let norms_one = seg.norms_per_layer.max(1) * hidden_size;
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

  // Final RMSNorm / LayerNorm
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
    + audio_and_speech
    + embedding
    + attention
    + dense_ffn
    + moe_routed
    + moe_shared
    + norms_and_routers
    + lm_head;
  let active_params = vision_encoder + audio_and_speech + embedding + active_trunk + lm_head;

  ParamBreakdown {
    vision_encoder,
    audio_and_speech,
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
#[cfg(feature = "server")]
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
    audio: None,
    diffusion: None,
    output_head: OutputHeadSpec::CausalLmHead,
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
    aux_configs: Vec::new(),
    declarative_graph_wire: None,
  }
}

/// 从外置 `models/<slug>/graph.json` 构建声明式架构蓝图（供 YOLO / U-Net / SAM / 经典网络直接驱动）。
#[cfg(feature = "server")]
pub fn build_declarative_graph_blueprint(
  repo_id: &str,
  graph_json_raw: &str,
  total_params: u64,
  arch_class: &str,
) -> Result<ModelBlueprint, String> {
  let spec: crate::tensor_graph::DeclarativeGraphSpec =
    serde_json::from_str(graph_json_raw).map_err(|e| format!("解析 graph.json 失败: {e}"))?;
  spec.validate()?;
  let wire = spec.encode_wire();
  let node_count = spec.nodes.len() as u64;
  Ok(ModelBlueprint {
    repo_id: repo_id.to_string(),
    source_label: "Declarative Architecture Graph (graph.json)".to_string(),
    family: ArchFamily::DeclarativeGraph,
    architectures: if arch_class.trim().is_empty() {
      Vec::new()
    } else {
      vec![arch_class.trim().to_string()]
    },
    model_type: spec.shape_mode,
    torch_dtype: "fp32/fp16".to_string(),
    quantization_summary: None,
    multimodal_note: None,
    vision: None,
    audio: None,
    diffusion: None,
    output_head: OutputHeadSpec::default(),
    vocab_size: 0,
    hidden_size: 0,
    num_hidden_layers: node_count,
    max_position_embeddings: 0,
    rms_norm_eps: None,
    tie_word_embeddings: false,
    rope_summary: String::new(),
    hidden_act: String::new(),
    num_nextn_predict_layers: 0,
    tokenizer_class: None,
    default_temperature: None,
    default_top_p: None,
    segments: Vec::new(),
    params: ParamBreakdown {
      total_params,
      active_params: total_params,
      ..ParamBreakdown::default()
    },
    safetensors: None,
    raw_config_pretty: String::new(),
    aux_configs: Vec::new(),
    declarative_graph_wire: Some(wire),
  })
}

#[cfg(feature = "server")]
fn u64_any(obj: &serde_json::Map<String, Value>, keys: &[&str]) -> Option<u64> {
  keys.iter().find_map(|k| obj.get(*k).and_then(Value::as_u64))
}

#[cfg(feature = "server")]
fn f64_any(obj: &serde_json::Map<String, Value>, keys: &[&str]) -> Option<f64> {
  keys.iter().find_map(|k| obj.get(*k).and_then(Value::as_f64))
}

#[cfg(feature = "server")]
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

#[cfg(feature = "server")]
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

#[cfg(feature = "server")]
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

#[cfg(feature = "server")]
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

/// 构造内置静态预设蓝图（零 `serde_json` 运行时解析，使 `build_model_blueprint*` 得以完全收拢在服务端）。
#[allow(clippy::too_many_arguments)]
fn direct_preset_blueprint(
  repo_id: &str,
  family: ArchFamily,
  architecture: &str,
  model_type: &str,
  quantization_summary: Option<&str>,
  vocab_size: u64,
  hidden_size: u64,
  num_hidden_layers: u64,
  max_position_embeddings: u64,
  rms_norm_eps: f64,
  rope_summary: &str,
  num_nextn_predict_layers: u64,
  tokenizer_class: &str,
  default_temperature: f64,
  default_top_p: f64,
  segments: Vec<LayerSegment>,
  safetensors: SafetensorsSummary,
  raw_config_pretty: &str,
) -> ModelBlueprint {
  let output_head = OutputHeadSpec::CausalLmHead;
  let params = compute_param_breakdown(
    0,
    0,
    vocab_size,
    hidden_size,
    false,
    &output_head,
    num_nextn_predict_layers,
    &segments,
  );
  ModelBlueprint {
    repo_id: repo_id.to_string(),
    source_label: "Built-in Snapshot (HF/ModelScope Verified)".to_string(),
    family,
    architectures: vec![architecture.to_string()],
    model_type: model_type.to_string(),
    torch_dtype: "bfloat16".to_string(),
    quantization_summary: quantization_summary.map(str::to_string),
    multimodal_note: None,
    vision: None,
    audio: None,
    diffusion: None,
    output_head,
    vocab_size,
    hidden_size,
    num_hidden_layers,
    max_position_embeddings,
    rms_norm_eps: Some(rms_norm_eps),
    tie_word_embeddings: false,
    rope_summary: rope_summary.to_string(),
    hidden_act: "silu".to_string(),
    num_nextn_predict_layers,
    tokenizer_class: Some(tokenizer_class.to_string()),
    default_temperature: Some(default_temperature),
    default_top_p: Some(default_top_p),
    segments,
    params,
    safetensors: Some(safetensors),
    raw_config_pretty: raw_config_pretty.to_string(),
    aux_configs: Vec::new(),
    declarative_graph_wire: None,
  }
}

/// 按预设 slug（如 `"deepseek-v3"`）或仓库名（如 `"Qwen/Qwen3-8B"`）加载内置真实架构快照。
pub fn load_builtin_preset(key: &str) -> Option<ModelBlueprint> {
  let norm = key.trim().to_ascii_lowercase();
  match norm.as_str() {
    "deepseek-v3" | "deepseek-ai/deepseek-v3" | "deepseek-r1" | "deepseek-ai/deepseek-r1" => {
      let mla = AttentionSpec::DeepSeekMla {
        num_heads: 128,
        q_lora_rank: Some(1536),
        kv_lora_rank: 512,
        qk_nope_head_dim: 128,
        qk_rope_head_dim: 64,
        v_head_dim: 128,
      };
      let segs = vec![
        LayerSegment {
          start_layer: 0,
          end_layer: 2,
          title: "前置稠密预热层 (Dense SwiGLU)".to_string(),
          attention: mla.clone(),
          ffn: FfnSpec::DenseGated {
            intermediate_size: 18432,
            act_fn: "silu".to_string(),
            bias: false,
            fused_gate_up: false,
            is_gated: true,
          },
          norms_per_layer: 2,
          schedule_note: None,
        },
        LayerSegment {
          start_layer: 3,
          end_layer: 60,
          title: "稀疏混合专家主干 (Sparse MoE)".to_string(),
          attention: mla,
          ffn: FfnSpec::SparseMoe {
            num_routed_experts: 256,
            top_k: 8,
            moe_intermediate_size: 2048,
            num_shared_experts: 1,
            shared_intermediate_size: 2048,
            has_shared_gate: false,
            router_groups: Some((8, 4)),
            scoring_func: "sigmoid".to_string(),
            act_fn: "silu".to_string(),
          },
          norms_per_layer: 2,
          schedule_note: None,
        },
      ];
      Some(direct_preset_blueprint(
        "deepseek-ai/DeepSeek-V3",
        ArchFamily::DeepSeekMla,
        "DeepseekV3ForCausalLM",
        "deepseek_v3",
        Some("FP8 · e4m3 · block 128×128"),
        129280,
        7168,
        61,
        163840,
        1e-6,
        "RoPE θ=10K (yarn ×40.0)",
        1,
        "LlamaTokenizerFast",
        0.6,
        0.95,
        segs,
        deepseek_v3_safetensors_snapshot(),
        DEEPSEEK_V3_CONFIG,
      ))
    }
    "qwen3-30b-a3b" | "qwen/qwen3-30b-a3b" => {
      let segs = vec![LayerSegment {
        start_layer: 0,
        end_layer: 47,
        title: "稀疏混合专家主干 (Sparse MoE)".to_string(),
        attention: AttentionSpec::StandardGqa {
          num_q_heads: 32,
          num_kv_heads: 4,
          head_dim: 128,
          qkv_bias: false,
          o_bias: false,
          qk_norm: true,
          sliding_window: None,
          sliding_window_pattern: None,
          nope_layer_interval: None,
          is_causal: true,
          has_cross_attn: false,
        },
        ffn: FfnSpec::SparseMoe {
          num_routed_experts: 128,
          top_k: 8,
          moe_intermediate_size: 768,
          num_shared_experts: 0,
          shared_intermediate_size: 0,
          has_shared_gate: false,
          router_groups: None,
          scoring_func: "softmax".to_string(),
          act_fn: "silu".to_string(),
        },
        norms_per_layer: 2,
        schedule_note: None,
      }];
      Some(direct_preset_blueprint(
        "Qwen/Qwen3-30B-A3B",
        ArchFamily::QwenMoe,
        "Qwen3MoeForCausalLM",
        "qwen3_moe",
        None,
        151936,
        2048,
        48,
        40960,
        1e-6,
        "RoPE θ=1M",
        0,
        "Qwen2Tokenizer",
        0.6,
        0.95,
        segs,
        qwen3_30b_safetensors_snapshot(),
        QWEN3_30B_A3B_CONFIG,
      ))
    }
    "qwen3-8b" | "qwen/qwen3-8b" => {
      let segs = vec![LayerSegment {
        start_layer: 0,
        end_layer: 35,
        title: "Transformer 解码器层栈 (Decoder Stack)".to_string(),
        attention: AttentionSpec::StandardGqa {
          num_q_heads: 32,
          num_kv_heads: 8,
          head_dim: 128,
          qkv_bias: false,
          o_bias: false,
          qk_norm: true,
          sliding_window: None,
          sliding_window_pattern: None,
          nope_layer_interval: None,
          is_causal: true,
          has_cross_attn: false,
        },
        ffn: FfnSpec::DenseGated {
          intermediate_size: 12288,
          act_fn: "silu".to_string(),
          bias: false,
          fused_gate_up: false,
          is_gated: true,
        },
        norms_per_layer: 2,
        schedule_note: None,
      }];
      Some(direct_preset_blueprint(
        "Qwen/Qwen3-8B",
        ArchFamily::QwenDense,
        "Qwen3ForCausalLM",
        "qwen3",
        None,
        151936,
        4096,
        36,
        40960,
        1e-6,
        "RoPE θ=1M",
        0,
        "Qwen2Tokenizer",
        0.6,
        0.95,
        segs,
        qwen3_8b_safetensors_snapshot(),
        QWEN3_8B_CONFIG,
      ))
    }
    "llama-3.1-8b" | "llama3-8b" | "meta-llama/llama-3.1-8b-instruct" => {
      let segs = vec![LayerSegment {
        start_layer: 0,
        end_layer: 31,
        title: "Transformer 解码器层栈 (Decoder Stack)".to_string(),
        attention: AttentionSpec::StandardGqa {
          num_q_heads: 32,
          num_kv_heads: 8,
          head_dim: 128,
          qkv_bias: false,
          o_bias: false,
          qk_norm: false,
          sliding_window: None,
          sliding_window_pattern: None,
          nope_layer_interval: None,
          is_causal: true,
          has_cross_attn: false,
        },
        ffn: FfnSpec::DenseGated {
          intermediate_size: 14336,
          act_fn: "silu".to_string(),
          bias: false,
          fused_gate_up: false,
          is_gated: true,
        },
        norms_per_layer: 2,
        schedule_note: None,
      }];
      Some(direct_preset_blueprint(
        "meta-llama/Llama-3.1-8B-Instruct",
        ArchFamily::LlamaDense,
        "LlamaForCausalLM",
        "llama",
        None,
        128256,
        4096,
        32,
        131072,
        1e-5,
        "RoPE θ=500K (llama3 ×8.0)",
        0,
        "PreTrainedTokenizerFast",
        0.6,
        0.9,
        segs,
        llama3_8b_safetensors_snapshot(),
        LLAMA_3_1_8B_CONFIG,
      ))
    }
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

/// `(pattern, stage, repeat_count, dim0, dim1_or_0, is_fp8, shard_idx)`
type SnapRow = (&'static str, &'static str, u16, u32, u32, bool, u8);

fn pattern_to_example(pattern: &str) -> String {
  let mut out = String::with_capacity(pattern.len());
  let mut rest = pattern;
  while let Some(open) = rest.find('[') {
    out.push_str(&rest[..open]);
    let after = &rest[open + 1..];
    if let Some(close) = after.find(']') {
      let inner = &after[..close];
      if let Some((start, _)) = inner.split_once("..") {
        out.push_str(start);
      } else {
        out.push('[');
        out.push_str(inner);
        out.push(']');
      }
      rest = &after[close + 1..];
    } else {
      out.push('[');
      rest = after;
      break;
    }
  }
  out.push_str(rest);
  out
}

fn rows_to_patterns(total_shards: u16, rows: &[SnapRow]) -> Vec<TensorPatternEntry> {
  rows
    .iter()
    .map(|&(pattern, stage, repeat_count, d0, d1, is_fp8, shard_idx)| {
      let shape = if d1 == 0 { vec![d0 as u64] } else { vec![d0 as u64, d1 as u64] };
      let elems: u64 = if d1 == 0 { d0 as u64 } else { (d0 as u64) * (d1 as u64) };
      let (dtype, bpe) = if is_fp8 { ("F8_E4M3", 1u64) } else { ("BF16", 2u64) };
      TensorPatternEntry {
        pattern: pattern.to_string(),
        example_name: pattern_to_example(pattern),
        stage: stage.to_string(),
        repeat_count: repeat_count as usize,
        shape: Some(shape),
        dtype: Some(dtype.to_string()),
        single_byte_len: Some(elems * bpe),
        example_shard: Some(format!("model-{shard_idx:05}-of-{total_shards:05}.safetensors")),
      }
    })
    .collect()
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
    patterns: rows_to_patterns(
      163,
      &[
        ("model.embed_tokens.weight", "embedding", 1, 129280, 7168, false, 1),
        ("model.layers.[0..60].self_attn.q_a_proj.weight", "attention", 61, 1536, 7168, true, 1),
        ("model.layers.[0..60].self_attn.q_a_layernorm.weight", "attention", 61, 1536, 0, false, 1),
        ("model.layers.[0..60].self_attn.q_b_proj.weight", "attention", 61, 24576, 1536, true, 1),
        (
          "model.layers.[0..60].self_attn.kv_a_proj_with_mqa.weight",
          "attention",
          61,
          576,
          7168,
          true,
          1,
        ),
        ("model.layers.[0..60].self_attn.kv_a_layernorm.weight", "attention", 61, 512, 0, false, 1),
        ("model.layers.[0..60].self_attn.kv_b_proj.weight", "attention", 61, 32768, 512, true, 1),
        ("model.layers.[0..60].self_attn.o_proj.weight", "attention", 61, 7168, 16384, true, 1),
        ("model.layers.[0..60].input_layernorm.weight", "norm", 61, 7168, 0, false, 1),
        ("model.layers.[0..60].post_attention_layernorm.weight", "norm", 61, 7168, 0, false, 1),
        ("model.layers.[0..2].mlp.gate_proj.weight", "ffn", 3, 18432, 7168, true, 1),
        ("model.layers.[0..2].mlp.up_proj.weight", "ffn", 3, 18432, 7168, true, 1),
        ("model.layers.[0..2].mlp.down_proj.weight", "ffn", 3, 7168, 18432, true, 1),
        ("model.layers.[3..60].mlp.gate.weight", "moe", 58, 256, 7168, false, 2),
        ("model.layers.[3..60].mlp.gate.e_score_correction_bias", "moe", 58, 256, 0, false, 2),
        (
          "model.layers.[3..60].mlp.shared_experts.gate_proj.weight",
          "moe",
          58,
          2048,
          7168,
          true,
          2,
        ),
        ("model.layers.[3..60].mlp.shared_experts.up_proj.weight", "moe", 58, 2048, 7168, true, 2),
        (
          "model.layers.[3..60].mlp.shared_experts.down_proj.weight",
          "moe",
          58,
          7168,
          2048,
          true,
          2,
        ),
        (
          "model.layers.[3..60].mlp.experts.[0..255].gate_proj.weight",
          "moe",
          14_848,
          2048,
          7168,
          true,
          2,
        ),
        (
          "model.layers.[3..60].mlp.experts.[0..255].up_proj.weight",
          "moe",
          14_848,
          2048,
          7168,
          true,
          2,
        ),
        (
          "model.layers.[3..60].mlp.experts.[0..255].down_proj.weight",
          "moe",
          14_848,
          7168,
          2048,
          true,
          2,
        ),
        ("model.layers.61.eh_proj.weight", "mtp", 1, 7168, 14336, false, 163),
        ("model.norm.weight", "norm", 1, 7168, 0, false, 162),
        ("lm_head.weight", "lm_head", 1, 129280, 7168, false, 162),
      ],
    ),
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
    patterns: rows_to_patterns(
      16,
      &[
        ("model.embed_tokens.weight", "embedding", 1, 151936, 2048, false, 1),
        ("model.layers.[0..47].self_attn.q_proj.weight", "attention", 48, 4096, 2048, false, 1),
        ("model.layers.[0..47].self_attn.k_proj.weight", "attention", 48, 512, 2048, false, 1),
        ("model.layers.[0..47].self_attn.v_proj.weight", "attention", 48, 512, 2048, false, 1),
        ("model.layers.[0..47].self_attn.o_proj.weight", "attention", 48, 2048, 4096, false, 1),
        ("model.layers.[0..47].self_attn.q_norm.weight", "attention", 48, 128, 0, false, 1),
        ("model.layers.[0..47].self_attn.k_norm.weight", "attention", 48, 128, 0, false, 1),
        ("model.layers.[0..47].input_layernorm.weight", "norm", 48, 2048, 0, false, 1),
        ("model.layers.[0..47].post_attention_layernorm.weight", "norm", 48, 2048, 0, false, 1),
        ("model.layers.[0..47].mlp.gate.weight", "moe", 48, 128, 2048, false, 1),
        (
          "model.layers.[0..47].mlp.experts.[0..127].gate_proj.weight",
          "moe",
          6144,
          768,
          2048,
          false,
          1,
        ),
        (
          "model.layers.[0..47].mlp.experts.[0..127].up_proj.weight",
          "moe",
          6144,
          768,
          2048,
          false,
          1,
        ),
        (
          "model.layers.[0..47].mlp.experts.[0..127].down_proj.weight",
          "moe",
          6144,
          2048,
          768,
          false,
          1,
        ),
        ("model.norm.weight", "norm", 1, 2048, 0, false, 16),
        ("lm_head.weight", "lm_head", 1, 151936, 2048, false, 16),
      ],
    ),
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
    patterns: rows_to_patterns(
      5,
      &[
        ("model.embed_tokens.weight", "embedding", 1, 151936, 4096, false, 1),
        ("model.layers.[0..35].self_attn.q_proj.weight", "attention", 36, 4096, 4096, false, 1),
        ("model.layers.[0..35].self_attn.k_proj.weight", "attention", 36, 1024, 4096, false, 1),
        ("model.layers.[0..35].self_attn.v_proj.weight", "attention", 36, 1024, 4096, false, 1),
        ("model.layers.[0..35].self_attn.o_proj.weight", "attention", 36, 4096, 4096, false, 1),
        ("model.layers.[0..35].self_attn.q_norm.weight", "attention", 36, 128, 0, false, 1),
        ("model.layers.[0..35].self_attn.k_norm.weight", "attention", 36, 128, 0, false, 1),
        ("model.layers.[0..35].input_layernorm.weight", "norm", 36, 4096, 0, false, 1),
        ("model.layers.[0..35].post_attention_layernorm.weight", "norm", 36, 4096, 0, false, 1),
        ("model.layers.[0..35].mlp.gate_proj.weight", "ffn", 36, 12288, 4096, false, 1),
        ("model.layers.[0..35].mlp.up_proj.weight", "ffn", 36, 12288, 4096, false, 1),
        ("model.layers.[0..35].mlp.down_proj.weight", "ffn", 36, 4096, 12288, false, 1),
        ("model.norm.weight", "norm", 1, 4096, 0, false, 5),
        ("lm_head.weight", "lm_head", 1, 151936, 4096, false, 5),
      ],
    ),
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
    patterns: rows_to_patterns(
      4,
      &[
        ("model.embed_tokens.weight", "embedding", 1, 128256, 4096, false, 1),
        ("model.layers.[0..31].self_attn.q_proj.weight", "attention", 32, 4096, 4096, false, 1),
        ("model.layers.[0..31].self_attn.k_proj.weight", "attention", 32, 1024, 4096, false, 1),
        ("model.layers.[0..31].self_attn.v_proj.weight", "attention", 32, 1024, 4096, false, 1),
        ("model.layers.[0..31].self_attn.o_proj.weight", "attention", 32, 4096, 4096, false, 1),
        ("model.layers.[0..31].input_layernorm.weight", "norm", 32, 4096, 0, false, 1),
        ("model.layers.[0..31].post_attention_layernorm.weight", "norm", 32, 4096, 0, false, 1),
        ("model.layers.[0..31].mlp.gate_proj.weight", "ffn", 32, 14336, 4096, false, 1),
        ("model.layers.[0..31].mlp.up_proj.weight", "ffn", 32, 14336, 4096, false, 1),
        ("model.layers.[0..31].mlp.down_proj.weight", "ffn", 32, 4096, 14336, false, 1),
        ("model.norm.weight", "norm", 1, 4096, 0, false, 4),
        ("lm_head.weight", "lm_head", 1, 128256, 4096, false, 4),
      ],
    ),
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Diffusers 图像 / 视频扩散与流匹配多组件管线解析器（FLUX.1 / SD3.5 / Wan2.1 / HunyuanVideo / SDXL）
// 仅服务端编译，避免将 700+ 行 Diffusers 解析器打入客户端 WASM。
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "server")]
mod diffusion_parser {
  use super::*;

  pub(super) fn is_diffusion_root_json(obj: &serde_json::Map<String, Value>) -> bool {
    let Some(cls) = obj.get("_class_name").and_then(Value::as_str) else {
      return false;
    };
    cls.ends_with("Pipeline")
      || obj.contains_key("_diffusers_version")
        && (cls.contains("Transformer2DModel")
          || cls.contains("Transformer3DModel")
          || cls.contains("UNet2DConditionModel")
          || cls.contains("UNetSpatioTemporalConditionModel")
          || cls.contains("FluxTransformer")
          || cls.contains("WanTransformer")
          || cls.contains("HunyuanVideoTransformer")
          || cls.contains("SD3Transformer")
          || cls.contains("CogVideoXTransformer")
          || cls.contains("LTXVideoTransformer"))
  }

  fn diffusion_slot_role(slot: &str, class_name: &str) -> Option<&'static str> {
    let sl = slot.to_ascii_lowercase();
    let cl = class_name.to_ascii_lowercase();
    if sl.starts_with("tokenizer")
      || sl.contains("processor")
      || sl.contains("extractor")
      || sl.contains("safety_checker")
      || sl.contains("watermarker")
      || sl.starts_with("image_encoder")
    {
      return None;
    }
    if sl == "scheduler" || cl.contains("scheduler") {
      Some("scheduler")
    } else if sl.starts_with("text_encoder") {
      Some("conditioner")
    } else if sl == "vae" || cl.contains("autoencoder") {
      Some("vae")
    } else if sl.starts_with("transformer")
      || sl.starts_with("unet")
      || sl == "prior"
      || cl.contains("transformer2d")
      || cl.contains("transformer3d")
      || cl.contains("unet")
    {
      Some("denoiser")
    } else {
      None
    }
  }

  /// 解析 `model_index.json`，返回 `(pipeline_class, Vec<(slot, class_name)>)`（已过滤分词器/安全检查器并按管线顺序排列）。
  pub fn extract_diffusion_slots(
    model_index_json: &str,
  ) -> Result<(String, Vec<(String, String)>), String> {
    let val: Value = serde_json::from_str(model_index_json)
      .map_err(|e| format!("解析 model_index.json 失败: {e}"))?;
    let Some(obj) = val.as_object() else {
      return Err("model_index.json 根节点必须是 JSON 对象".to_string());
    };
    let pipeline_class =
      obj.get("_class_name").and_then(Value::as_str).unwrap_or("DiffusionPipeline").to_string();

    let mut items: Vec<(u8, String, String)> = Vec::new();
    for (k, v) in obj {
      if k.starts_with('_') {
        continue;
      }
      let class_opt = v
        .as_array()
        .and_then(|arr| arr.get(1).or_else(|| arr.first()))
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty());
      let Some(cls) = class_opt else {
        continue;
      };
      let Some(role) = diffusion_slot_role(k, cls) else {
        continue;
      };
      let order = match role {
        "conditioner" => 0,
        "scheduler" => 1,
        "denoiser" => 2,
        "vae" => 3,
        _ => 4,
      };
      items.push((order, k.clone(), cls.to_string()));
    }
    items.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    let slots = items.into_iter().map(|(_, s, c)| (s, c)).collect();
    Ok((pipeline_class, slots))
  }

  pub(super) fn build_diffusion_from_single_json(
    repo_id: &str,
    source_label: &str,
    root_obj: &serde_json::Map<String, Value>,
    raw_json: &str,
    safetensors: Option<SafetensorsSummary>,
  ) -> Result<ModelBlueprint, String> {
    let cls = root_obj.get("_class_name").and_then(Value::as_str).unwrap_or("DiffusionPipeline");
    if cls.ends_with("Pipeline") {
      return build_diffusion_pipeline_blueprint(repo_id, source_label, raw_json, &[], safetensors);
    }
    let slot = if cls.to_ascii_lowercase().contains("unet") { "unet" } else { "transformer" };
    let synth_index = serde_json::json!({
      "_class_name": format!("{cls}Pipeline"),
      slot: ["diffusers", cls],
    })
    .to_string();
    let comps = vec![(slot.to_string(), raw_json.to_string())];
    build_diffusion_pipeline_blueprint(repo_id, source_label, &synth_index, &comps, safetensors)
  }

  fn parse_conditioner_component(
    slot: &str,
    class_name: &str,
    cfg_opt: Option<&serde_json::Map<String, Value>>,
  ) -> PipelineComponentSpec {
    let cl = class_name.to_ascii_lowercase();
    if let Some(c) = cfg_opt {
      let sub = c.get("text_config").and_then(Value::as_object).unwrap_or(c);
      let d = u64_any(sub, &["hidden_size", "d_model"]).unwrap_or(768);
      let layers = u64_any(sub, &["num_hidden_layers", "num_layers"]).unwrap_or(12);
      let vocab = u64_any(sub, &["vocab_size"]).unwrap_or(49_408);
      let ffn = u64_any(sub, &["intermediate_size", "d_ff"]).unwrap_or(d * 4);
      let is_gated = sub.get("is_gated_act").and_then(Value::as_bool).unwrap_or(false)
        || sub
          .get("feed_forward_proj")
          .and_then(Value::as_str)
          .is_some_and(|s| s.contains("gated"))
        || cl.contains("llama")
        || cl.contains("qwen")
        || cl.contains("gemma")
        || cl.contains("mistral");
      let ffn_mats = if is_gated { 3 } else { 2 };
      let est = vocab * d + layers * (4 * d * d + ffn_mats * d * ffn);
      return PipelineComponentSpec {
        slot: slot.to_string(),
        class_name: class_name.to_string(),
        role: "conditioner".to_string(),
        spec_summary: format!("{layers}L × {d}d (FFN={ffn})"),
        estimated_params: est,
      };
    }

    let (summary, est) = if cl.contains("umt5") {
      ("24L × 4096d (Multilingual UMT5-XXL)".to_string(), 5_680_000_000)
    } else if cl.contains("t5") {
      ("24L × 4096d (T5-XXL Encoder)".to_string(), 4_762_000_000)
    } else if cl.contains("llama") || cl.contains("mistral") {
      ("32L × 4096d (LLM Conditioner)".to_string(), 7_500_000_000)
    } else if cl.contains("clip") && slot.ends_with('2') {
      ("32L × 1280d (OpenCLIP-bigG)".to_string(), 694_000_000)
    } else {
      ("12L × 768d (CLIP-ViT-L/14)".to_string(), 123_060_000)
    };
    PipelineComponentSpec {
      slot: slot.to_string(),
      class_name: class_name.to_string(),
      role: "conditioner".to_string(),
      spec_summary: summary,
      estimated_params: est,
    }
  }

  fn parse_scheduler_component(
    slot: &str,
    class_name: &str,
    cfg_opt: Option<&serde_json::Map<String, Value>>,
  ) -> (PipelineComponentSpec, String) {
    let mut parts = vec![class_name.to_string()];
    if let Some(c) = cfg_opt {
      if let Some(t) = u64_any(c, &["num_train_timesteps"]) {
        parts.push(format!("{t}T"));
      }
      if let Some(sh) = f64_any(c, &["flow_shift", "shift", "base_shift"]) {
        parts.push(format!("shift={sh:.2}"));
      } else if c.get("use_dynamic_shifting").and_then(Value::as_bool) == Some(true) {
        parts.push("dynamic-shift".to_string());
      }
      if let Some(pt) = c.get("prediction_type").and_then(Value::as_str) {
        parts.push(pt.to_string());
      }
    }
    let summary = parts.join(" · ");
    let comp = PipelineComponentSpec {
      slot: slot.to_string(),
      class_name: class_name.to_string(),
      role: "scheduler".to_string(),
      spec_summary: summary.clone(),
      estimated_params: 0,
    };
    (comp, summary)
  }

  fn parse_vae_component(
    slot: &str,
    class_name: &str,
    cfg_opt: Option<&serde_json::Map<String, Value>>,
    is_video_hint: bool,
  ) -> (PipelineComponentSpec, u64, u64, u64) {
    let cl = class_name.to_ascii_lowercase();
    let is_3d = is_video_hint
      || cl.contains("wan")
      || cl.contains("hunyuan")
      || cl.contains("cogvideo")
      || cl.contains("video")
      || cl.contains("3d");

    let latent_ch = cfg_opt
      .and_then(|c| u64_any(c, &["latent_channels", "z_dim"]))
      .unwrap_or(if cl.contains("sdxl") { 4 } else { 16 });
    let n_stages = cfg_opt
      .and_then(|c| c.get("block_out_channels").and_then(Value::as_array).map(Vec::len))
      .unwrap_or(4);
    let spatial_comp = 1u64 << (n_stages.saturating_sub(1).min(5) as u32);
    let temporal_comp = if is_3d {
      cfg_opt
        .and_then(|c| u64_any(c, &["temporal_compression_ratio"]))
        .unwrap_or(if cl.contains("ltx") { 8 } else { 4 })
    } else {
      1
    };
    let est = if is_3d { 127_000_000 } else { 83_800_000 };
    let summary = if is_3d {
      format!("{latent_ch}ch 3D Causal Latent ({temporal_comp}×{spatial_comp}×{spatial_comp})")
    } else {
      format!("{latent_ch}ch 2D Latent ({spatial_comp}×{spatial_comp} Spatial)")
    };
    (
      PipelineComponentSpec {
        slot: slot.to_string(),
        class_name: class_name.to_string(),
        role: "vae".to_string(),
        spec_summary: summary,
        estimated_params: est,
      },
      latent_ch,
      spatial_comp.max(8),
      temporal_comp,
    )
  }

  /// 从 `model_index.json` 与子目录 `<slot>/config.json` 集合构建完整的图像/视频扩散生成管线蓝图。
  pub fn build_diffusion_pipeline_blueprint(
    repo_id: &str,
    source_label: &str,
    model_index_json: &str,
    component_configs: &[(String, String)],
    safetensors: Option<SafetensorsSummary>,
  ) -> Result<ModelBlueprint, String> {
    let (pipeline_class, slots) = extract_diffusion_slots(model_index_json)?;
    let mut parsed_cfgs: std::collections::HashMap<String, Value> =
      std::collections::HashMap::new();
    for (slot, raw) in component_configs {
      if let Ok(v) = serde_json::from_str::<Value>(raw) {
        parsed_cfgs.insert(slot.clone(), v);
      }
    }

    let pipe_lc = format!("{} {}", pipeline_class, repo_id).to_ascii_lowercase();
    let mut is_video = pipe_lc.contains("video")
      || pipe_lc.contains("wan")
      || pipe_lc.contains("t2v")
      || pipe_lc.contains("i2v")
      || pipe_lc.contains("cogvideo")
      || pipe_lc.contains("ltx");

    let mut components: Vec<PipelineComponentSpec> = Vec::new();
    let mut conditioner_params = 0u64;
    let mut scheduler_summary = "FlowMatch / Diffusion Scheduler".to_string();
    let mut denoiser_slot_cls: Option<(String, String)> = None;
    let mut vae_slot_cls: Option<(String, String)> = None;

    for (slot, cls) in &slots {
      if cls.to_ascii_lowercase().contains("3d") || cls.to_ascii_lowercase().contains("video") {
        is_video = true;
      }
      match diffusion_slot_role(slot, cls) {
        Some("conditioner") => {
          let comp = parse_conditioner_component(
            slot,
            cls,
            parsed_cfgs.get(slot).and_then(Value::as_object),
          );
          conditioner_params += comp.estimated_params;
          components.push(comp);
        }
        Some("scheduler") => {
          let (comp, sched_s) =
            parse_scheduler_component(slot, cls, parsed_cfgs.get(slot).and_then(Value::as_object));
          scheduler_summary = sched_s;
          components.push(comp);
        }
        Some("denoiser") if denoiser_slot_cls.is_none() => {
          denoiser_slot_cls = Some((slot.clone(), cls.clone()));
        }
        Some("vae") if vae_slot_cls.is_none() => {
          vae_slot_cls = Some((slot.clone(), cls.clone()));
        }
        _ => {}
      }
    }

    let (den_slot, den_cls) = denoiser_slot_cls.unwrap_or_else(|| {
      let fallback_cls = if pipe_lc.contains("xl") {
        "UNet2DConditionModel"
      } else if is_video {
        "WanTransformer3DModel"
      } else {
        "FluxTransformer2DModel"
      };
      ("transformer".to_string(), fallback_cls.to_string())
    });
    let den_obj = parsed_cfgs.get(&den_slot).and_then(Value::as_object);
    let den_lc = den_cls.to_ascii_lowercase();
    if den_lc.contains("3d") || den_lc.contains("wan") || den_lc.contains("hunyuanvideo") {
      is_video = true;
    }

    let (vae_comp, mut latent_channels, spatial_comp, temporal_comp) =
      if let Some((vs, vc)) = vae_slot_cls {
        parse_vae_component(&vs, &vc, parsed_cfgs.get(&vs).and_then(Value::as_object), is_video)
      } else {
        parse_vae_component(
          "vae",
          if is_video { "AutoencoderKL3DCausal" } else { "AutoencoderKL" },
          None,
          is_video,
        )
      };
    let vae_class = vae_comp.class_name.clone();
    let vae_params = vae_comp.estimated_params;

    // 解析去噪主干（MMDiT 双+单流 / 纯联合 MMDiT / 3D Video DiT / 多尺度 CrossAttn UNet）
    let is_unet = den_slot.starts_with("unet") || den_lc.contains("unet");
    let mut patch_summary =
      if is_video { "[1, 2, 2] 3D Patch".to_string() } else { "2×2 Patch".to_string() };
    let mut rope_summary = if is_video {
      "3D RoPE (Temporal + H + W)".to_string()
    } else {
      "2D Axial / RoPE + AdaLN-Zero".to_string()
    };
    let mut torch_dtype = "bfloat16".to_string();
    let denoiser_kind: String;
    let hidden_size: u64;
    let num_hidden_layers: u64;
    let segments: Vec<LayerSegment>;
    let denoiser_attn_params: u64;
    let denoiser_ffn_params: u64;
    let denoiser_mod_params: u64;

    if is_unet {
      denoiser_kind = "Multi-Scale Cross-Attention UNet".to_string();
      patch_summary = "Conv2d 3×3 Stem (Latent Grid)".to_string();
      rope_summary = "Sinusoidal Timestep + Spatial CrossAttn".to_string();
      let chans: Vec<u64> = den_obj
        .and_then(|c| c.get("block_out_channels"))
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_u64).collect())
        .filter(|v: &Vec<u64>| !v.is_empty())
        .unwrap_or_else(|| vec![320, 640, 1280]);
      let tf_per_block: Vec<u64> = den_obj
        .and_then(|c| c.get("transformer_layers_per_block"))
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_u64).collect())
        .unwrap_or_else(|| vec![1, 2, 10]);
      if let Some(in_c) = den_obj.and_then(|c| u64_any(c, &["in_channels"])) {
        latent_channels = in_c.min(latent_channels.max(4));
      }
      torch_dtype = "float16".to_string();
      let top_ch = *chans.last().unwrap_or(&1280);
      hidden_size = top_ch;
      let down_n = chans.len() as u64;
      num_hidden_layers = down_n * 2 + 1;
      let head_d = 64u64;
      let heads = (top_ch / head_d).max(8);

      // 精确核算 ResNet2D + SpatialTransformer 多尺度卷积与交叉注意力参数
      let mut unet_attn = 0u64;
      let mut unet_ffn = 0u64;
      let cross_d = den_obj.and_then(|c| u64_any(c, &["cross_attention_dim"])).unwrap_or(2048);
      let layers_per_blk = den_obj.and_then(|c| u64_any(c, &["layers_per_block"])).unwrap_or(2);
      for (idx, &ch) in chans.iter().enumerate() {
        let n_tf = tf_per_block.get(idx).copied().unwrap_or(1);
        // Down (layers_per_blk) + Up (layers_per_blk + 1)
        let stage_blocks = 2 * layers_per_blk + 1;
        let res_conv = stage_blocks * 2 * 9 * ch * ch;
        let tf_a = stage_blocks * n_tf * (4 * ch * ch + 2 * ch * ch + 2 * ch * cross_d);
        let tf_f = stage_blocks * n_tf * (8 * ch * ch);
        unet_attn += tf_a;
        unet_ffn += res_conv + tf_f;
      }
      // Mid bottleneck block at top_ch
      let mid_tf = tf_per_block.last().copied().unwrap_or(10);
      unet_attn += mid_tf * (6 * top_ch * top_ch + 2 * top_ch * cross_d);
      unet_ffn += 4 * 9 * top_ch * top_ch + mid_tf * 8 * top_ch * top_ch;
      denoiser_attn_params = unet_attn;
      denoiser_ffn_params = unet_ffn;
      denoiser_mod_params = 64 * top_ch * top_ch / 16;

      let attn_spec = AttentionSpec::StandardGqa {
        num_q_heads: heads,
        num_kv_heads: heads,
        head_dim: head_d,
        qkv_bias: false,
        o_bias: true,
        qk_norm: false,
        sliding_window: None,
        sliding_window_pattern: None,
        nope_layer_interval: None,
        is_causal: false,
        has_cross_attn: true,
      };
      let ffn_spec = FfnSpec::DenseGated {
        intermediate_size: top_ch * 4,
        act_fn: "geglu + conv3x3".to_string(),
        bias: true,
        fused_gate_up: false,
        is_gated: true,
      };
      segments = vec![
        LayerSegment {
          start_layer: 0,
          end_layer: down_n - 1,
          title: format!("UNet 下采样编码塔 (Down Blocks {:?})", chans),
          attention: attn_spec.clone(),
          ffn: ffn_spec.clone(),
          norms_per_layer: 2,
          schedule_note: Some(
            "逐级 2× 空间下采样并将多尺度特征通过 Skip Connection 缓存至解码端".to_string(),
          ),
        },
        LayerSegment {
          start_layer: down_n,
          end_layer: down_n,
          title: format!("UNet 瓶颈交叉注意力层 (Mid Block {top_ch}ch)"),
          attention: attn_spec.clone(),
          ffn: ffn_spec.clone(),
          norms_per_layer: 2,
          schedule_note: None,
        },
        LayerSegment {
          start_layer: down_n + 1,
          end_layer: num_hidden_layers - 1,
          title: "UNet 跳跃连接上采样解码塔 (Up Blocks + Skip Concat)".to_string(),
          attention: attn_spec,
          ffn: ffn_spec,
          norms_per_layer: 2,
          schedule_note: None,
        },
      ];
    } else {
      // DiT / MMDiT / 3D Spatio-Temporal Video DiT
      let heads = den_obj
        .and_then(|c| u64_any(c, &["num_attention_heads", "num_heads"]))
        .unwrap_or(if is_video { 40 } else { 24 })
        .max(1);
      let head_dim =
        den_obj.and_then(|c| u64_any(c, &["attention_head_dim", "head_dim"])).unwrap_or(128).max(1);
      hidden_size = den_obj
        .and_then(|c| u64_any(c, &["dim", "hidden_size", "inner_dim"]))
        .unwrap_or(heads * head_dim)
        .max(64);
      let double_layers = den_obj
        .and_then(|c| u64_any(c, &["num_layers", "depth"]))
        .unwrap_or(if is_video { 40 } else { 19 })
        .max(1);
      let single_layers = den_obj.and_then(|c| u64_any(c, &["num_single_layers"])).unwrap_or(0);
      num_hidden_layers = double_layers + single_layers;

      if let Some(c) = den_obj {
        if let Some(arr) = c.get("patch_size").and_then(Value::as_array) {
          let dims: Vec<String> =
            arr.iter().filter_map(Value::as_u64).map(|n| n.to_string()).collect();
          if !dims.is_empty() {
            patch_summary = format!("[{}] 3D Patchify", dims.join("×"));
          }
        } else if let Some(p) = u64_any(c, &["patch_size"]) {
          if let Some(pt) = u64_any(c, &["patch_size_t"]) {
            patch_summary = format!("[{pt}×{p}×{p}] 3D Patchify");
          } else {
            patch_summary = format!("{p}×{p} Patchify");
          }
        }
        if let Some(axes) = c.get("axes_dims_rope").and_then(Value::as_array) {
          let dims: Vec<String> =
            axes.iter().filter_map(Value::as_u64).map(|n| n.to_string()).collect();
          if !dims.is_empty() {
            rope_summary = format!("Multi-Axis RoPE [{}] + AdaLN-Zero", dims.join("+"));
          }
        }
      }

      let is_joint_mmdit = single_layers > 0
        || den_lc.contains("flux")
        || den_lc.contains("sd3")
        || den_lc.contains("mmdit")
        || den_obj.is_some_and(|c| {
          c.contains_key("joint_attention_dim") || c.contains_key("caption_projection_dim")
        });

      if single_layers > 0 {
        denoiser_kind = if is_video {
          "Hybrid Double+Single Stream 3D MMDiT".to_string()
        } else {
          "Hybrid Double+Single Stream MMDiT".to_string()
        };
        // Double-stream: img+txt QKVO (8 D²) + img+txt 2-layer MLP (16 D²) + dual 6-way AdaLN-Zero (12 D²) = 36 D²
        // Single-stream: fused QKVO (4 D²) + 2-layer MLP (8 D²) + 3-way AdaLN-Zero (3 D²) = 15 D²
        let d2 = hidden_size * hidden_size;
        denoiser_attn_params = double_layers * 8 * d2 + single_layers * 4 * d2;
        denoiser_ffn_params = double_layers * 16 * d2 + single_layers * 8 * d2;
        denoiser_mod_params = double_layers * 12 * d2 + single_layers * 3 * d2 + 6 * d2;

        segments = vec![
        LayerSegment {
          start_layer: 0,
          end_layer: double_layers - 1,
          title: format!("双流联合模态块 (Double-Stream MMDiT ×{double_layers})"),
          attention: AttentionSpec::StandardGqa {
            num_q_heads: heads,
            num_kv_heads: heads,
            head_dim,
            qkv_bias: true,
            o_bias: true,
            qk_norm: true,
            sliding_window: None,
            sliding_window_pattern: None,
            nope_layer_interval: None,
            is_causal: false,
            has_cross_attn: true,
          },
          ffn: FfnSpec::DenseGated {
            intermediate_size: hidden_size * 4,
            act_fn: "gelu_tanh (Dual Img+Txt MLP)".to_string(),
            bias: true,
            fused_gate_up: false,
            is_gated: false,
          },
          norms_per_layer: 4,
          schedule_note: Some(
            "图像潜空间 Token 与文本 Prompt Token 各自拥有独立 QKV/MLP 权重，并在全双向联合注意力中交互"
              .to_string(),
          ),
        },
        LayerSegment {
          start_layer: double_layers,
          end_layer: num_hidden_layers - 1,
          title: format!("单流融合去噪块 (Single-Stream DiT ×{single_layers})"),
          attention: AttentionSpec::StandardGqa {
            num_q_heads: heads,
            num_kv_heads: heads,
            head_dim,
            qkv_bias: true,
            o_bias: true,
            qk_norm: true,
            sliding_window: None,
            sliding_window_pattern: None,
            nope_layer_interval: None,
            is_causal: false,
            has_cross_attn: false,
          },
          ffn: FfnSpec::DenseGated {
            intermediate_size: hidden_size * 4,
            act_fn: "gelu_tanh (Parallel Attn+MLP)".to_string(),
            bias: true,
            fused_gate_up: false,
            is_gated: false,
          },
          norms_per_layer: 2,
          schedule_note: Some(
            "拼接图像与文本流后执行并行 QKV+MLP 融合线性投影，兼顾高吞吐与全局空间一致性"
              .to_string(),
          ),
        },
      ];
      } else if is_joint_mmdit {
        denoiser_kind = "Joint Multimodal Diffusion Transformer (MMDiT)".to_string();
        let d2 = hidden_size * hidden_size;
        denoiser_attn_params = double_layers * 8 * d2;
        denoiser_ffn_params = double_layers * 16 * d2;
        denoiser_mod_params = double_layers * 12 * d2 + 4 * d2;
        segments = vec![LayerSegment {
          start_layer: 0,
          end_layer: double_layers - 1,
          title: format!("双流联合去噪主干 (Joint MMDiT ×{double_layers})"),
          attention: AttentionSpec::StandardGqa {
            num_q_heads: heads,
            num_kv_heads: heads,
            head_dim,
            qkv_bias: true,
            o_bias: true,
            qk_norm: true,
            sliding_window: None,
            sliding_window_pattern: None,
            nope_layer_interval: None,
            is_causal: false,
            has_cross_attn: true,
          },
          ffn: FfnSpec::DenseGated {
            intermediate_size: hidden_size * 4,
            act_fn: "gelu_tanh (Dual Stream)".to_string(),
            bias: true,
            fused_gate_up: false,
            is_gated: false,
          },
          norms_per_layer: 4,
          schedule_note: Some(
            "全程双向空间注意力（无自回归因果掩码）+ AdaLN-Zero 时间步调制".to_string(),
          ),
        }];
      } else {
        denoiser_kind = if is_video {
          "3D Spatio-Temporal Video DiT".to_string()
        } else {
          "Diffusion Transformer (DiT)".to_string()
        };
        let ffn_dim = den_obj
          .and_then(|c| u64_any(c, &["ffn_dim", "intermediate_size"]))
          .unwrap_or(hidden_size * 4);
        let has_cross = den_obj.is_none_or(|c| {
          c.contains_key("text_dim") || c.contains_key("cross_attention_dim") || is_video
        });
        let attn_mats = if has_cross { 8 } else { 4 };
        let d2 = hidden_size * hidden_size;
        denoiser_attn_params = double_layers * attn_mats * d2;
        denoiser_ffn_params = double_layers * 2 * hidden_size * ffn_dim;
        denoiser_mod_params = double_layers * 6 * hidden_size * 256 + 8 * d2;
        segments = vec![LayerSegment {
          start_layer: 0,
          end_layer: double_layers - 1,
          title: if is_video {
            format!("3D 时空视频去噪层栈 (3D Full-Attn + CrossAttn ×{double_layers})")
          } else {
            format!("DiT 空间去噪层栈 (DiT Blocks ×{double_layers})")
          },
          attention: AttentionSpec::StandardGqa {
            num_q_heads: heads,
            num_kv_heads: heads,
            head_dim,
            qkv_bias: true,
            o_bias: true,
            qk_norm: true,
            sliding_window: None,
            sliding_window_pattern: None,
            nope_layer_interval: None,
            is_causal: false,
            has_cross_attn: has_cross,
          },
          ffn: FfnSpec::DenseGated {
            intermediate_size: ffn_dim,
            act_fn: "gelu_tanh".to_string(),
            bias: true,
            fused_gate_up: false,
            is_gated: false,
          },
          norms_per_layer: 2,
          schedule_note: Some(if is_video {
            "对完整 3D 视频潜空间网格 [F', H', W'] 执行全时空自注意力，并经 Cross-Attention 注入文本语义"
            .to_string()
          } else {
            "对 2D 潜空间 Patch 网格执行全双向注意力与 AdaLN 时间步条件调制".to_string()
          }),
        }];
      }
    }

    let denoiser_total = denoiser_attn_params + denoiser_ffn_params + denoiser_mod_params;
    components.push(PipelineComponentSpec {
      slot: den_slot,
      class_name: den_cls.clone(),
      role: "denoiser".to_string(),
      spec_summary: format!("{num_hidden_layers}L × {hidden_size}d ({denoiser_kind})"),
      estimated_params: denoiser_total,
    });
    components.push(vae_comp);

    let default_latent_shape = if is_video {
      format!(
        "81f×832×480 → Latent [B, {latent_channels}, {}, {}, {}] (~32.8K tokens)",
        81_u64.div_ceil(temporal_comp),
        480 / spatial_comp,
        832 / spatial_comp
      )
    } else {
      format!(
        "1024×1024 px → Latent [B, {latent_channels}, {}, {}] (4096 tokens)",
        1024 / spatial_comp,
        1024 / spatial_comp
      )
    };
    let max_latent_tokens = if is_video { 32_768 } else { 4_096 };

    let total_params = conditioner_params + denoiser_total + vae_params;
    let params = ParamBreakdown {
      vision_encoder: vae_params,
      audio_and_speech: 0,
      embedding: conditioner_params,
      attention: denoiser_attn_params,
      dense_ffn: denoiser_ffn_params,
      moe_routed: 0,
      moe_shared: 0,
      norms_and_routers: denoiser_mod_params,
      lm_head: 0,
      mtp_head: 0,
      total_params,
      // 单步迭代去噪（t = T..1）仅激活去噪主干（Text Encoder 前置单次运行，VAE 末尾单次解码）
      active_params: denoiser_total,
    };

    let mut combined_json = serde_json::Map::new();
    if let Ok(idx_val) = serde_json::from_str::<Value>(model_index_json) {
      combined_json.insert("model_index.json".to_string(), idx_val);
    }
    for (slot, raw) in component_configs {
      if let Ok(v) = serde_json::from_str::<Value>(raw) {
        combined_json.insert(format!("{slot}/config.json"), v);
      }
    }
    let raw_config_pretty = serde_json::to_string_pretty(&Value::Object(combined_json))
      .unwrap_or_else(|_| model_index_json.to_string());

    let comp_count = components.len();
    let multimodal_note = Some(format!(
    "{} 多组件{}生成管线（共 {comp_count} 个协同子模块：Prompt 编码 → {scheduler_summary} → {den_cls} → {vae_class}）",
    pipeline_class,
    if is_video { "视频" } else { "图像" }
  ));

    Ok(ModelBlueprint {
      repo_id: repo_id.to_string(),
      source_label: source_label.to_string(),
      family: ArchFamily::DiffusionPipeline,
      architectures: vec![pipeline_class.clone(), den_cls],
      model_type: if is_video {
        "video_diffusion_pipeline".to_string()
      } else {
        "image_diffusion_pipeline".to_string()
      },
      torch_dtype,
      quantization_summary: None,
      multimodal_note,
      vision: None,
      audio: None,
      diffusion: Some(DiffusionPipelineSpec {
        pipeline_class,
        is_video,
        scheduler_summary,
        denoiser_kind,
        latent_channels,
        spatial_compression: spatial_comp,
        temporal_compression: temporal_comp,
        patch_summary,
        default_latent_shape,
        components,
      }),
      output_head: OutputHeadSpec::LatentDecoder {
        vae_class,
        latent_channels,
        spatial_compression: spatial_comp,
        temporal_compression: temporal_comp,
        is_video,
        estimated_params: vae_params,
      },
      vocab_size: 0,
      hidden_size,
      num_hidden_layers,
      max_position_embeddings: max_latent_tokens,
      rms_norm_eps: Some(1e-6),
      tie_word_embeddings: false,
      rope_summary,
      hidden_act: "gelu_tanh".to_string(),
      num_nextn_predict_layers: 0,
      tokenizer_class: None,
      default_temperature: None,
      default_top_p: None,
      segments,
      params,
      safetensors,
      raw_config_pretty,
      aux_configs: Vec::new(),
      declarative_graph_wire: None,
    })
  }
}

#[cfg(feature = "server")]
use diffusion_parser::{build_diffusion_from_single_json, is_diffusion_root_json};
#[cfg(feature = "server")]
pub use diffusion_parser::{build_diffusion_pipeline_blueprint, extract_diffusion_slots};

#[cfg(all(test, feature = "server"))]
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

  #[test]
  fn parses_whisper_large_v3_asr_dual_tower_and_cross_attention() {
    let whisper_cfg = r#"{
      "architectures": ["WhisperForConditionalGeneration"],
      "model_type": "whisper",
      "vocab_size": 51866,
      "d_model": 1280,
      "encoder_layers": 32,
      "encoder_attention_heads": 20,
      "encoder_ffn_dim": 5120,
      "decoder_layers": 32,
      "decoder_attention_heads": 20,
      "decoder_ffn_dim": 5120,
      "max_source_positions": 1500,
      "max_target_positions": 448,
      "num_mel_bins": 128,
      "activation_function": "gelu"
    }"#;
    let preproc = r#"{"feature_size": 128, "sampling_rate": 16000}"#;
    let bp = build_model_blueprint_with_aux(
      "openai/whisper-large-v3",
      "HF",
      whisper_cfg,
      AuxConfigFiles { preprocessor_json: Some(preproc), ..AuxConfigFiles::default() },
      None,
    )
    .unwrap();

    assert_eq!(bp.family, ArchFamily::AudioAsr);
    let aud = bp.audio.as_ref().expect("audio encoder parsed");
    assert_eq!((aud.num_mel_bins, aud.sampling_rate_hz, aud.num_layers), (128, 16000, 32));
    // Whisper-large-v3 is ~1.54B params (635M audio encoder + 905M cross-attn decoder)
    assert!(bp.params.total_params > 1_500_000_000 && bp.params.total_params < 1_580_000_000);
    match &bp.segments[0].attention {
      AttentionSpec::StandardGqa { has_cross_attn, .. } => assert!(*has_cross_attn),
      other => panic!("unexpected attention: {other:?}"),
    }
  }

  #[test]
  fn parses_qwen2_5_omni_thinker_talker_token2wav_pipeline() {
    let omni_cfg = r#"{
      "architectures": ["Qwen2_5OmniModel"],
      "model_type": "qwen2_5_omni",
      "thinker_config": {
        "vision_config": {
          "model_type": "qwen2_5_omni_vision",
          "hidden_size": 1280,
          "num_hidden_layers": 32,
          "num_attention_heads": 16,
          "intermediate_size": 3420
        },
        "audio_config": {
          "model_type": "qwen2_5_omni_audio",
          "d_model": 1280,
          "encoder_layers": 32,
          "encoder_attention_heads": 20,
          "encoder_ffn_dim": 5120,
          "num_mel_bins": 128
        },
        "text_config": {
          "model_type": "qwen2_5_omni_text",
          "vocab_size": 152064,
          "hidden_size": 3584,
          "intermediate_size": 18944,
          "num_hidden_layers": 28,
          "num_attention_heads": 28,
          "num_key_value_heads": 4
        }
      },
      "talker_config": {
        "vocab_size": 8448,
        "hidden_size": 896,
        "intermediate_size": 4864,
        "num_hidden_layers": 24,
        "num_attention_heads": 14,
        "num_key_value_heads": 2
      },
      "token2wav_config": {
        "dit_config": { "dim": 1024, "depth": 22 },
        "bigvgan_config": { "upsample_rates": [8, 5, 4, 2] }
      }
    }"#;
    let bp =
      build_model_blueprint("Qwen/Qwen2.5-Omni-7B", "HF", omni_cfg, None, None, None).unwrap();
    assert_eq!(bp.family, ArchFamily::SpeechTtsOmni);
    assert!(bp.vision.is_some());
    assert!(bp.audio.is_some());
    assert!(matches!(bp.output_head, OutputHeadSpec::SpeechSynthesizer { .. }));
    assert!(bp.params.audio_and_speech > 800_000_000);
    assert!(bp.params.total_params > 9_000_000_000 && bp.params.total_params < 12_500_000_000);
  }

  #[test]
  fn parses_mamba_constant_state_and_qwen3_next_hybrid_deltanet() {
    let mamba_cfg = r#"{
      "architectures": ["MambaForCausalLM"],
      "model_type": "mamba",
      "vocab_size": 50280,
      "hidden_size": 2560,
      "state_size": 16,
      "num_hidden_layers": 64,
      "conv_kernel": 4,
      "expand": 2
    }"#;
    let m = build_model_blueprint("state-spaces/mamba-2.8b-hf", "HF", mamba_cfg, None, None, None)
      .unwrap();
    assert_eq!(m.family, ArchFamily::HybridLinearSsm);
    assert!(m.params.total_params > 2_600_000_000 && m.params.total_params < 2_900_000_000);
    let est_4k = m.estimate_vram(PrecisionPreset::Bf16, 4096, 1);
    let est_128k = m.estimate_vram(PrecisionPreset::Bf16, 131_072, 1);
    assert_eq!(est_4k.kv_cache_bytes, 0.0);
    assert_eq!(est_128k.kv_cache_bytes, 0.0);
    assert!(est_4k.state_cache_bytes > 0.0);
    assert_eq!(est_4k.state_cache_bytes, est_128k.state_cache_bytes);

    let qwen_next = r#"{
      "architectures": ["Qwen3NextForCausalLM"],
      "model_type": "qwen3_next",
      "vocab_size": 151936,
      "hidden_size": 2048,
      "num_hidden_layers": 48,
      "num_attention_heads": 16,
      "num_key_value_heads": 2,
      "head_dim": 256,
      "linear_num_key_heads": 16,
      "linear_num_value_heads": 32,
      "linear_key_head_dim": 128,
      "linear_value_head_dim": 128,
      "linear_conv_kernel_dim": 4,
      "full_attention_interval": 4,
      "moe_intermediate_size": 512,
      "shared_expert_intermediate_size": 512,
      "num_experts": 512,
      "num_experts_per_tok": 10
    }"#;
    let qn =
      build_model_blueprint("Qwen/Qwen3-Next-80B-A3B-Instruct", "HF", qwen_next, None, None, None)
        .unwrap();
    assert_eq!(qn.family, ArchFamily::HybridLinearSsm);
    assert_eq!(qn.segments.len(), 2);
    assert_eq!(qn.segments[0].layer_count(), 36);
    assert_eq!(qn.segments[1].layer_count(), 12);
    assert!(qn.params.total_params > 78_000_000_000 && qn.params.total_params < 83_000_000_000);
    assert!(qn.params.active_params > 2_500_000_000 && qn.params.active_params < 4_200_000_000);
  }

  #[test]
  fn parses_bge_m3_bidirectional_encoder_with_zero_kv_cache() {
    let bge_cfg = r#"{
      "architectures": ["XLMRobertaModel"],
      "model_type": "xlm-roberta",
      "vocab_size": 250002,
      "hidden_size": 1024,
      "num_hidden_layers": 24,
      "num_attention_heads": 16,
      "intermediate_size": 4096,
      "hidden_act": "gelu",
      "max_position_embeddings": 8194
    }"#;
    let pool_json = r#"{"word_embedding_dimension": 1024, "pooling_mode_cls_token": true, "pooling_mode_mean_tokens": false}"#;
    let bp = build_model_blueprint_with_aux(
      "BAAI/bge-m3",
      "HF",
      bge_cfg,
      AuxConfigFiles { pooling_json: Some(pool_json), ..AuxConfigFiles::default() },
      None,
    )
    .unwrap();
    assert_eq!(bp.family, ArchFamily::EncoderEmbedding);
    assert!(bp.is_single_pass_encoder());
    assert!(bp.params.total_params > 550_000_000 && bp.params.total_params < 580_000_000);
    let vram = bp.estimate_vram(PrecisionPreset::Bf16, 8192, 16);
    assert_eq!(vram.kv_cache_bytes, 0.0);
  }

  #[test]
  fn parses_flux1_dev_mmdit_multi_component_pipeline() {
    let model_index = r#"{
      "_class_name": "FluxPipeline",
      "_diffusers_version": "0.31.0",
      "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
      "text_encoder": ["transformers", "CLIPTextModel"],
      "text_encoder_2": ["transformers", "T5EncoderModel"],
      "tokenizer": ["transformers", "CLIPTokenizer"],
      "tokenizer_2": ["transformers", "T5TokenizerFast"],
      "transformer": ["diffusers", "FluxTransformer2DModel"],
      "vae": ["diffusers", "AutoencoderKL"]
    }"#;
    let comps = vec![
      (
        "text_encoder".to_string(),
        r#"{"vocab_size":49408,"hidden_size":768,"intermediate_size":3072,"num_hidden_layers":12}"#
          .to_string(),
      ),
      (
        "text_encoder_2".to_string(),
        r#"{"vocab_size":32128,"d_model":4096,"d_ff":10240,"num_layers":24,"is_gated_act":true}"#
          .to_string(),
      ),
      (
        "transformer".to_string(),
        r#"{"in_channels":64,"num_layers":19,"num_single_layers":38,"attention_head_dim":128,"num_attention_heads":24,"joint_attention_dim":4096,"axes_dims_rope":[16,56,56]}"#
          .to_string(),
      ),
      (
        "vae".to_string(),
        r#"{"latent_channels":16,"block_out_channels":[128,256,512,512]}"#.to_string(),
      ),
      (
        "scheduler".to_string(),
        r#"{"num_train_timesteps":1000,"use_dynamic_shifting":true}"#.to_string(),
      ),
    ];

    let bp = build_diffusion_pipeline_blueprint(
      "black-forest-labs/FLUX.1-dev",
      "HF",
      model_index,
      &comps,
      None,
    )
    .unwrap();

    assert_eq!(bp.family, ArchFamily::DiffusionPipeline);
    assert!(bp.is_diffusion_pipeline());
    assert_eq!(bp.segments.len(), 2);
    assert_eq!(bp.segments[0].layer_count(), 19);
    assert_eq!(bp.segments[1].layer_count(), 38);
    // ~11.90B MMDiT denoiser active per step, ~16.87B full pipeline resident in VRAM
    assert!(bp.params.active_params > 11_700_000_000 && bp.params.active_params < 12_100_000_000);
    assert!(bp.params.total_params > 16_600_000_000 && bp.params.total_params < 17_100_000_000);
    let est = bp.estimate_vram(PrecisionPreset::Bf16, 4096, 1);
    assert_eq!(est.kv_cache_bytes, 0.0);
  }

  #[test]
  fn parses_wan2_1_14b_3d_video_dit_and_sdxl_unet_pipelines() {
    let wan_idx = r#"{
      "_class_name": "WanPipeline",
      "text_encoder": ["transformers", "UMT5EncoderModel"],
      "transformer": ["diffusers", "WanTransformer3DModel"],
      "vae": ["diffusers", "AutoencoderKLWan"],
      "scheduler": ["diffusers", "UniPCMultistepScheduler"]
    }"#;
    let wan_comps = vec![
      (
        "text_encoder".to_string(),
        r#"{"vocab_size":256384,"d_model":4096,"d_ff":10240,"num_layers":24,"is_gated_act":true}"#
          .to_string(),
      ),
      (
        "transformer".to_string(),
        r#"{"num_attention_heads":40,"attention_head_dim":128,"ffn_dim":13824,"num_layers":40,"patch_size":[1,2,2],"text_dim":4096}"#
          .to_string(),
      ),
      ("vae".to_string(), r#"{"z_dim":16,"temporal_compression_ratio":4}"#.to_string()),
      ("scheduler".to_string(), r#"{"num_train_timesteps":1000,"flow_shift":5.0}"#.to_string()),
    ];
    let wan = build_diffusion_pipeline_blueprint(
      "Wan-AI/Wan2.1-T2V-14B-Diffusers",
      "HF",
      wan_idx,
      &wan_comps,
      None,
    )
    .unwrap();
    let diff = wan.diffusion.as_ref().expect("diffusion spec");
    assert!(diff.is_video);
    assert_eq!(diff.temporal_compression, 4);
    // ~14.28B 3D Video DiT denoiser, ~20.0B total pipeline with UMT5-XXL
    assert!(wan.params.active_params > 14_000_000_000 && wan.params.active_params < 14_600_000_000);
    assert!(wan.params.total_params > 19_600_000_000 && wan.params.total_params < 20_500_000_000);

    // Also verify pasting raw `model_index.json` directly into `build_model_blueprint` works!
    let sdxl_idx = r#"{
      "_class_name": "StableDiffusionXLPipeline",
      "text_encoder": ["transformers", "CLIPTextModel"],
      "text_encoder_2": ["transformers", "CLIPTextModelWithProjection"],
      "unet": ["diffusers", "UNet2DConditionModel"],
      "vae": ["diffusers", "AutoencoderKL"],
      "scheduler": ["diffusers", "EulerDiscreteScheduler"]
    }"#;
    let sdxl = build_model_blueprint(
      "stabilityai/stable-diffusion-xl-base-1.0",
      "HF",
      sdxl_idx,
      None,
      None,
      None,
    )
    .unwrap();
    assert_eq!(sdxl.family, ArchFamily::DiffusionPipeline);
    assert_eq!(sdxl.segments.len(), 3);
  }

  #[test]
  fn direct_builtin_presets_match_json_parser_exactly() {
    let cases = [
      (
        "deepseek-v3",
        "deepseek-ai/DeepSeek-V3",
        DEEPSEEK_V3_CONFIG,
        r#"{"temperature": 0.6, "top_p": 0.95}"#,
        r#"{"tokenizer_class": "LlamaTokenizerFast"}"#,
        deepseek_v3_safetensors_snapshot(),
      ),
      (
        "qwen3-30b-a3b",
        "Qwen/Qwen3-30B-A3B",
        QWEN3_30B_A3B_CONFIG,
        r#"{"temperature": 0.6, "top_p": 0.95}"#,
        r#"{"tokenizer_class": "Qwen2Tokenizer"}"#,
        qwen3_30b_safetensors_snapshot(),
      ),
      (
        "qwen3-8b",
        "Qwen/Qwen3-8B",
        QWEN3_8B_CONFIG,
        r#"{"temperature": 0.6, "top_p": 0.95}"#,
        r#"{"tokenizer_class": "Qwen2Tokenizer"}"#,
        qwen3_8b_safetensors_snapshot(),
      ),
      (
        "llama-3.1-8b",
        "meta-llama/Llama-3.1-8B-Instruct",
        LLAMA_3_1_8B_CONFIG,
        r#"{"temperature": 0.6, "top_p": 0.9}"#,
        r#"{"tokenizer_class": "PreTrainedTokenizerFast"}"#,
        llama3_8b_safetensors_snapshot(),
      ),
    ];

    for (key, repo, cfg, gen, tok, st) in cases {
      let direct = load_builtin_preset(key).expect("direct preset");
      let parsed = build_model_blueprint(
        repo,
        "Built-in Snapshot (HF/ModelScope Verified)",
        cfg,
        Some(gen),
        Some(tok),
        Some(st),
      )
      .expect("parsed preset");

      assert_eq!(direct.repo_id, parsed.repo_id);
      assert_eq!(direct.family, parsed.family);
      assert_eq!(direct.architectures, parsed.architectures);
      assert_eq!(direct.model_type, parsed.model_type);
      assert_eq!(direct.quantization_summary, parsed.quantization_summary);
      assert_eq!(direct.vocab_size, parsed.vocab_size);
      assert_eq!(direct.hidden_size, parsed.hidden_size);
      assert_eq!(direct.num_hidden_layers, parsed.num_hidden_layers);
      assert_eq!(direct.max_position_embeddings, parsed.max_position_embeddings);
      assert_eq!(direct.rope_summary, parsed.rope_summary);
      assert_eq!(direct.num_nextn_predict_layers, parsed.num_nextn_predict_layers);
      assert_eq!(direct.tokenizer_class, parsed.tokenizer_class);
      assert_eq!(direct.default_temperature, parsed.default_temperature);
      assert_eq!(direct.default_top_p, parsed.default_top_p);
      assert_eq!(direct.segments, parsed.segments);
      assert_eq!(direct.params, parsed.params);
      assert_eq!(direct.safetensors, parsed.safetensors);
    }
  }
}
