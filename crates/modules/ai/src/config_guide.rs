//! 大模型配置文件全景词典与 `config.json` 逐字段解析引擎（单一真相源）。
//!
//! 同时服务于：
//! 1. `/models/configs` 全景配置文件解剖指南页（8 类核心文件蓝图 + 5 种模态仓库文件树 + 字段速查词典 + 工程踩坑）；
//! 2. `/models/:slug` 与 `/ai/visualizer` 的「逐字段架构解剖」联动面板（将真实模型的 `config.json` 与附属配置文件解析为语义分组与拓扑图联动焦点）。

use std::collections::BTreeMap;

use app_core::i18n::Language;
use dioxus::prelude::*;
use dioxus_shadcn::{
  badge_class, card_class, BadgeVariant, Button, ButtonSize, ButtonVariant, Input, Spinner,
  SpinnerSize,
};
use serde_json::Value;
use widgets::PageMeta;

use crate::arch::InspectorFocus;

/// 大模型权重仓库中的 8 类核心配置文件。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConfigFileKind {
  /// `config.json`：模型架构超参蓝图
  ModelConfig,
  /// `model.safetensors.index.json` & `.safetensors`：权重分片映射与零拷贝头
  SafetensorsIndex,
  /// `tokenizer_config.json` / `tokenizer.json` / `chat_template.jinja`：分词与对话模板
  TokenizerConfig,
  /// `generation_config.json`：默认解码采样与停止符
  GenerationConfig,
  /// `preprocessor_config.json` / `video_preprocessor_config.json`：视觉与音频前处理
  PreprocessorConfig,
  /// `model_index.json` & `scheduler/scheduler_config.json`：Diffusers 多组件管线与调度器
  DiffusionPipeline,
  /// `modules.json` & `1_Pooling/config.json`：Sentence-Transformers 向量池化与归一化管线
  EmbeddingModules,
  /// `adapter_config.json`：PEFT LoRA / DoRA / QLoRA 适配器配置
  AdapterConfig,
}

impl ConfigFileKind {
  pub const ALL: [Self; 8] = [
    Self::ModelConfig,
    Self::SafetensorsIndex,
    Self::TokenizerConfig,
    Self::GenerationConfig,
    Self::PreprocessorConfig,
    Self::DiffusionPipeline,
    Self::EmbeddingModules,
    Self::AdapterConfig,
  ];

  pub fn id(self) -> &'static str {
    match self {
      Self::ModelConfig => "config-json",
      Self::SafetensorsIndex => "safetensors-index",
      Self::TokenizerConfig => "tokenizer-config",
      Self::GenerationConfig => "generation-config",
      Self::PreprocessorConfig => "preprocessor-config",
      Self::DiffusionPipeline => "diffusion-pipeline",
      Self::EmbeddingModules => "embedding-modules",
      Self::AdapterConfig => "adapter-config",
    }
  }

  pub fn short_name(self) -> &'static str {
    match self {
      Self::ModelConfig => "config.json",
      Self::SafetensorsIndex => "safetensors.index.json",
      Self::TokenizerConfig => "tokenizer_config.json",
      Self::GenerationConfig => "generation_config.json",
      Self::PreprocessorConfig => "preprocessor_config.json",
      Self::DiffusionPipeline => "model_index.json",
      Self::EmbeddingModules => "modules.json / 1_Pooling",
      Self::AdapterConfig => "adapter_config.json",
    }
  }

  pub fn from_id(id: &str) -> Self {
    match id {
      "safetensors-index" => Self::SafetensorsIndex,
      "tokenizer-config" => Self::TokenizerConfig,
      "generation-config" => Self::GenerationConfig,
      "preprocessor-config" => Self::PreprocessorConfig,
      "diffusion-pipeline" => Self::DiffusionPipeline,
      "embedding-modules" => Self::EmbeddingModules,
      "adapter-config" => Self::AdapterConfig,
      _ => Self::ModelConfig,
    }
  }

  pub fn from_filename(filename: &str) -> Self {
    let lower = filename.trim().to_ascii_lowercase();
    if lower.contains("generation_config") {
      Self::GenerationConfig
    } else if lower.contains("tokenizer") || lower.contains("chat_template") {
      Self::TokenizerConfig
    } else if lower.contains("preprocessor") || lower.contains("feature_extractor") {
      Self::PreprocessorConfig
    } else if lower.contains("pooling") || lower.contains("modules.json") {
      Self::EmbeddingModules
    } else if lower.contains("adapter_config") {
      Self::AdapterConfig
    } else if lower.contains("safetensors") {
      Self::SafetensorsIndex
    } else if lower.contains("model_index") || lower.contains("scheduler") {
      Self::DiffusionPipeline
    } else {
      Self::ModelConfig
    }
  }
}

/// 单个配置文件类型的全景蓝图说明（服务端静态表）。
#[cfg(feature = "server")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConfigFileSpec {
  pub kind: ConfigFileKind,
  pub filenames: &'static str,
  pub modalities_zh: &'static str,
  pub modalities_en: &'static str,
  pub role_zh: &'static str,
  pub role_en: &'static str,
  pub summary_zh: &'static str,
  pub summary_en: &'static str,
  pub loader_zh: &'static str,
  pub loader_en: &'static str,
  pub pitfall_zh: &'static str,
  pub pitfall_en: &'static str,
  pub snippet: &'static str,
}

#[cfg(feature = "server")]
impl ConfigFileSpec {
  pub fn modalities(&self, lang: Language) -> &'static str {
    lang.pick(self.modalities_zh, self.modalities_en)
  }

  pub fn role(&self, lang: Language) -> &'static str {
    lang.pick(self.role_zh, self.role_en)
  }

  pub fn summary(&self, lang: Language) -> &'static str {
    lang.pick(self.summary_zh, self.summary_en)
  }

  pub fn loader(&self, lang: Language) -> &'static str {
    lang.pick(self.loader_zh, self.loader_en)
  }

  pub fn pitfall(&self, lang: Language) -> &'static str {
    lang.pick(self.pitfall_zh, self.pitfall_en)
  }
}

/// 跨端传输的配置文件蓝图 DTO（通过共享 `serde_json::Value` 解析，避免 `ciborium`/`serde` 结构体派生膨胀 WASM）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigFileSpecDto {
  pub kind: ConfigFileKind,
  pub filenames: String,
  pub modalities_zh: String,
  pub modalities_en: String,
  pub role_zh: String,
  pub role_en: String,
  pub summary_zh: String,
  pub summary_en: String,
  pub loader_zh: String,
  pub loader_en: String,
  pub pitfall_zh: String,
  pub pitfall_en: String,
  pub snippet: String,
}

impl ConfigFileSpecDto {
  pub fn modalities(&self, lang: Language) -> &str {
    lang.pick(&self.modalities_zh, &self.modalities_en)
  }

  pub fn role(&self, lang: Language) -> &str {
    lang.pick(&self.role_zh, &self.role_en)
  }

  pub fn summary(&self, lang: Language) -> &str {
    lang.pick(&self.summary_zh, &self.summary_en)
  }

  pub fn loader(&self, lang: Language) -> &str {
    lang.pick(&self.loader_zh, &self.loader_en)
  }

  pub fn pitfall(&self, lang: Language) -> &str {
    lang.pick(&self.pitfall_zh, &self.pitfall_en)
  }
}

/// 配置文件字段的 7 大工程语义分组。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FieldGroup {
  Backbone,
  AttentionKv,
  FfnMoe,
  PositionContext,
  QuantPrecision,
  MultimodalPipeline,
  TokenizerGeneration,
}

impl FieldGroup {
  pub const ALL: [Self; 7] = [
    Self::Backbone,
    Self::AttentionKv,
    Self::FfnMoe,
    Self::PositionContext,
    Self::QuantPrecision,
    Self::MultimodalPipeline,
    Self::TokenizerGeneration,
  ];

  pub fn id(self) -> &'static str {
    match self {
      Self::Backbone => "backbone",
      Self::AttentionKv => "attention-kv",
      Self::FfnMoe => "ffn-moe",
      Self::PositionContext => "position-context",
      Self::QuantPrecision => "quant-precision",
      Self::MultimodalPipeline => "multimodal-pipeline",
      Self::TokenizerGeneration => "tokenizer-generation",
    }
  }

  pub fn from_id(id: &str) -> Self {
    match id {
      "attention-kv" => Self::AttentionKv,
      "ffn-moe" => Self::FfnMoe,
      "position-context" => Self::PositionContext,
      "quant-precision" => Self::QuantPrecision,
      "multimodal-pipeline" => Self::MultimodalPipeline,
      "tokenizer-generation" => Self::TokenizerGeneration,
      _ => Self::Backbone,
    }
  }

  pub fn label(self, lang: Language) -> &'static str {
    match self {
      Self::Backbone => lang.pick("主干维度与身份", "Backbone & Identity"),
      Self::AttentionKv => lang.pick("注意力与 KV / 状态缓存", "Attention & KV/State Cache"),
      Self::FfnMoe => lang.pick("FFN 与稀疏 MoE", "FFN & Sparse MoE"),
      Self::PositionContext => lang.pick("位置编码与长上下文", "Position & Long Context"),
      Self::QuantPrecision => lang.pick("量化精度与权重存储", "Quantization & Precision"),
      Self::MultimodalPipeline => {
        lang.pick("多模态、扩散与池化", "Multimodal, Diffusion & Pooling")
      }
      Self::TokenizerGeneration => {
        lang.pick("词表、采样与适配器", "Tokenizer, Sampling & Adapters")
      }
    }
  }
}

/// 单个配置字段的中英双语工程字典条目（服务端静态表）。
#[cfg(feature = "server")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConfigFieldDoc {
  /// 规范键名（支持顶层键或点分嵌套路径，如 `rope_scaling.type`、`vision_config.patch_size`）
  pub key: &'static str,
  /// 常见别名（用于兼容不同模型家族的命名习惯）
  pub aliases: &'static [&'static str],
  pub file: ConfigFileKind,
  pub group: FieldGroup,
  pub value_type: &'static str,
  pub summary_zh: &'static str,
  pub summary_en: &'static str,
  /// 对参数量、张量形状或显存/KV Cache 的定量公式影响
  pub formula_impact: Option<&'static str>,
  /// 与架构图检视器的联动焦点（点击字段可直接高亮左/右架构图）
  pub focus: Option<InspectorFocus>,
  /// 站内图鉴代表模型 slug（`/models/:slug`）
  pub example_model_slug: Option<&'static str>,
  /// 站内算子/数学实验室直通链接 `(href, label)`
  pub lab_link: Option<(&'static str, &'static str)>,
}

#[cfg(feature = "server")]
impl ConfigFieldDoc {
  pub fn summary(&self, lang: Language) -> &'static str {
    lang.pick(self.summary_zh, self.summary_en)
  }
}

/// 跨端传输的配置字段词典条目 DTO。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigFieldDocDto {
  pub key: String,
  pub aliases: Vec<String>,
  pub file: ConfigFileKind,
  pub group: FieldGroup,
  pub value_type: String,
  pub summary_zh: String,
  pub summary_en: String,
  pub formula_impact: Option<String>,
  pub focus: Option<InspectorFocus>,
  pub example_model_slug: Option<String>,
  pub lab_link: Option<(String, String)>,
}

impl ConfigFieldDocDto {
  pub fn summary(&self, lang: Language) -> &str {
    lang.pick(&self.summary_zh, &self.summary_en)
  }
}

/// 典型模型仓库按模态划分的文件树条目（服务端静态表）。
#[cfg(feature = "server")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RepoTreeItem {
  pub path: &'static str,
  pub required: bool,
  pub file_kind: ConfigFileKind,
  pub note_zh: &'static str,
  pub note_en: &'static str,
}

#[cfg(feature = "server")]
impl RepoTreeItem {
  pub fn note(&self, lang: Language) -> &'static str {
    lang.pick(self.note_zh, self.note_en)
  }
}

/// 跨端传输的仓库文件树条目 DTO。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepoTreeItemDto {
  pub path: String,
  pub required: bool,
  pub file_kind: ConfigFileKind,
  pub note_zh: String,
  pub note_en: String,
}

impl RepoTreeItemDto {
  pub fn note(&self, lang: Language) -> &str {
    lang.pick(&self.note_zh, &self.note_en)
  }
}

/// 按模态组织的标准仓库文件树蓝图（服务端静态表）。
#[cfg(feature = "server")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RepoModalityBlueprint {
  pub id: &'static str,
  pub label_zh: &'static str,
  pub label_en: &'static str,
  pub example_repo: &'static str,
  pub example_slug: Option<&'static str>,
  pub flow_zh: &'static str,
  pub flow_en: &'static str,
  pub items: &'static [RepoTreeItem],
}

#[cfg(feature = "server")]
impl RepoModalityBlueprint {
  pub fn label(&self, lang: Language) -> &'static str {
    lang.pick(self.label_zh, self.label_en)
  }

  pub fn flow(&self, lang: Language) -> &'static str {
    lang.pick(self.flow_zh, self.flow_en)
  }
}

/// 跨端传输的模态仓库文件树蓝图 DTO。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepoModalityBlueprintDto {
  pub id: String,
  pub label_zh: String,
  pub label_en: String,
  pub example_repo: String,
  pub example_slug: Option<String>,
  pub flow_zh: String,
  pub flow_en: String,
  pub items: Vec<RepoTreeItemDto>,
}

impl RepoModalityBlueprintDto {
  pub fn label(&self, lang: Language) -> &str {
    lang.pick(&self.label_zh, &self.label_en)
  }

  pub fn flow(&self, lang: Language) -> &str {
    lang.pick(&self.flow_zh, &self.flow_en)
  }
}

/// 常见大模型配置文件工程踩坑速查条目（服务端静态表）。
#[cfg(feature = "server")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConfigGotcha {
  pub id: &'static str,
  pub badge: &'static str,
  pub title_zh: &'static str,
  pub title_en: &'static str,
  pub symptom_zh: &'static str,
  pub symptom_en: &'static str,
  pub root_cause_zh: &'static str,
  pub root_cause_en: &'static str,
  pub fix_zh: &'static str,
  pub fix_en: &'static str,
  pub related_keys: &'static [&'static str],
}

#[cfg(feature = "server")]
impl ConfigGotcha {
  pub fn title(&self, lang: Language) -> &'static str {
    lang.pick(self.title_zh, self.title_en)
  }

  pub fn symptom(&self, lang: Language) -> &'static str {
    lang.pick(self.symptom_zh, self.symptom_en)
  }

  pub fn root_cause(&self, lang: Language) -> &'static str {
    lang.pick(self.root_cause_zh, self.root_cause_en)
  }

  pub fn fix(&self, lang: Language) -> &'static str {
    lang.pick(self.fix_zh, self.fix_en)
  }
}

/// 跨端传输的工程踩坑条目 DTO。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigGotchaDto {
  pub id: String,
  pub badge: String,
  pub title_zh: String,
  pub title_en: String,
  pub symptom_zh: String,
  pub symptom_en: String,
  pub root_cause_zh: String,
  pub root_cause_en: String,
  pub fix_zh: String,
  pub fix_en: String,
  pub related_keys: Vec<String>,
}

impl ConfigGotchaDto {
  pub fn title(&self, lang: Language) -> &str {
    lang.pick(&self.title_zh, &self.title_en)
  }

  pub fn symptom(&self, lang: Language) -> &str {
    lang.pick(&self.symptom_zh, &self.symptom_en)
  }

  pub fn root_cause(&self, lang: Language) -> &str {
    lang.pick(&self.root_cause_zh, &self.root_cause_en)
  }

  pub fn fix(&self, lang: Language) -> &str {
    lang.pick(&self.fix_zh, &self.fix_en)
  }
}

/// `/models/configs` 全景指南页的完整服务端下发载荷。
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ConfigGuidePayload {
  pub file_specs: Vec<ConfigFileSpecDto>,
  pub repo_trees: Vec<RepoModalityBlueprintDto>,
  pub field_docs: Vec<ConfigFieldDocDto>,
  pub gotchas: Vec<ConfigGotchaDto>,
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. 8 类主流大模型配置文件蓝图（仅服务端编译，避免静态长文撑大客户端 WASM）
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "server")]
pub const CONFIG_FILE_SPECS: &[ConfigFileSpec] = &[
  ConfigFileSpec {
    kind: ConfigFileKind::ModelConfig,
    filenames: "config.json",
    modalities_zh: "LLM / VLM / Audio / Embedding",
    modalities_en: "LLM / VLM / Audio / Embedding",
    role_zh: "模型架构超参总蓝图",
    role_en: "Master Architecture Blueprint",
    summary_zh: "决定神经网络计算图的拓扑结构、张量维度、注意力机制（GQA/MLA/SWA/SSM）、MoE 专家数、位置编码与量化元数据。推理引擎先读此文件分配权重形状与 KV Cache，再加载权重。",
    summary_en: "Defines network topology, tensor shapes, attention/SSM mechanisms (GQA/MLA/SWA), MoE routing, positional encoding, and quantization metadata. Engines read this first to allocate layers and KV cache.",
    loader_zh: "Transformers (`AutoConfig.from_pretrained`)、vLLM、SGLang、Candle 与 Burn 均以 `architectures` / `model_type` 分发对应模型实现。",
    loader_en: "Consumed by Transformers (`AutoConfig`), vLLM, SGLang, Candle, and Burn to dispatch the exact model graph via `architectures` and `model_type`.",
    pitfall_zh: "多模态模型常把语言主干嵌套在 `text_config` 或 `thinker_config` 下；且 `vocab_size` 可能大于实际词表条数（为 Tensor Core 128 对齐填充）。",
    pitfall_en: "VLMs often nest language params under `text_config` or `thinker_config`; `vocab_size` is frequently padded to multiples of 64/128 beyond tokenizer size.",
    snippet: r#"{
  "architectures": ["DeepseekV3ForCausalLM"],
  "model_type": "deepseek_v3",
  "hidden_size": 7168,
  "num_hidden_layers": 61,
  "kv_lora_rank": 512,
  "n_routed_experts": 256,
  "num_experts_per_tok": 8,
  "torch_dtype": "bfloat16"
}"#,
  },
  ConfigFileSpec {
    kind: ConfigFileKind::SafetensorsIndex,
    filenames: "model.safetensors.index.json & *.safetensors",
    modalities_zh: "全模态权重存储",
    modalities_en: "All Modalities (Weights)",
    role_zh: "分片张量索引与零拷贝元数据",
    role_en: "Sharded Tensor Index & Zero-Copy Header",
    summary_zh: "`index.json` 记录 `metadata.total_size` 与每个参数张量名到分片文件的 `weight_map`；每个 `.safetensors` 文件前 8 字节是头部长度，紧随其后的 JSON 头存储全部张量的精确 `dtype`、`shape` 与 `data_offsets`，可仅用 HTTP Range 请求读完整个权重树。",
    summary_en: "`index.json` maps every tensor name to its shard via `weight_map` plus `metadata.total_size`. Each `.safetensors` file starts with an 8-byte header length + JSON metadata (`dtype`, `shape`, `data_offsets`), enabling zero-weight-download inspection via HTTP Range.",
    loader_zh: "Candle / `safetensors` crate 直接 `mmap` 映射磁盘分片，按 `weight_map` 逐层绑定张量，彻底规避 Pickle 任意代码执行风险。",
    loader_en: "Candle and the Rust `safetensors` crate `mmap` shards directly via `weight_map`, avoiding Python Pickle execution risks.",
    pitfall_zh: "单文件小模型可能没有 `index.json`，只有单个 `model.safetensors`；FP8 模型会伴生 `*_scale_inv` 缩放张量，计算磁盘体积时需注意区分真实 dtype。",
    pitfall_en: "Small single-shard repos omit `index.json` and ship only `model.safetensors`. FP8 checkpoints include extra `*_scale_inv` tensors alongside quantized weights.",
    snippet: r#"{
  "metadata": { "total_size": 688953540608 },
  "weight_map": {
    "model.embed_tokens.weight": "model-00001-of-000163.safetensors",
    "model.layers.0.self_attn.kv_a_proj_with_mqa.weight": "model-00001-of-000163.safetensors",
    "model.layers.0.self_attn.kv_a_proj_with_mqa.weight_scale_inv": "model-00001-of-000163.safetensors"
  }
}"#,
  },
  ConfigFileSpec {
    kind: ConfigFileKind::TokenizerConfig,
    filenames: "tokenizer_config.json / tokenizer.json / chat_template.jinja",
    modalities_zh: "LLM / VLM / Audio / Embedding",
    modalities_en: "LLM / VLM / Audio / Embedding",
    role_zh: "文本切分规则与对话提示词模板",
    role_en: "Tokenization Rules & Chat Prompt Template",
    summary_zh: "`tokenizer.json` 包含 Rust `tokenizers` 可直接加载的完整词表、BPE 合并表或 Unigram 得分；`tokenizer_config.json` 与独立的 `chat_template.jinja` 定义特殊控制符（BOS/EOS/PAD）以及多轮对话、思考链 `<think>`、工具调用的 Jinja 渲染模板。",
    summary_en: "`tokenizer.json` holds the full self-contained vocab and BPE/Unigram model for Rust `tokenizers`; `tokenizer_config.json` & `chat_template.jinja` define special tokens (BOS/EOS/PAD) and Jinja templates for chat, reasoning `<think>`, and tool calls.",
    loader_zh: "Rust `tokenizers::Tokenizer::from_file(\"tokenizer.json\")` 零 Python 依赖即可运行；对话模板通常由 `minijinja` 渲染。",
    loader_en: "Loaded directly in Rust via `tokenizers::Tokenizer::from_file(\"tokenizer.json\")`, while chat templates are rendered with `minijinja`.",
    pitfall_zh: "`tokenizer_config.json` 里的 `model_max_length` 经常被写成 `1e30` 或保留历史默认值，不可当作架构上下文上限；真实上限应以 `config.json` 的 `max_position_embeddings` 为准。",
    pitfall_en: "`model_max_length` in `tokenizer_config.json` is often a sentinel (`1e30`) or stale value; always trust `config.json`'s `max_position_embeddings` for model limits.",
    snippet: r#"{
  "tokenizer_class": "Qwen2Tokenizer",
  "bos_token": null,
  "eos_token": "<|im_end|>",
  "pad_token": "<|endoftext|>",
  "add_bos_token": false,
  "clean_up_tokenization_spaces": false
}"#,
  },
  ConfigFileSpec {
    kind: ConfigFileKind::GenerationConfig,
    filenames: "generation_config.json",
    modalities_zh: "LLM / VLM / Audio",
    modalities_en: "LLM / VLM / Audio",
    role_zh: "推理默认解码采样与停止条件",
    role_en: "Default Decoding & Stop-Token Policy",
    summary_zh: "控制自回归解码器的默认行为：`do_sample`、`temperature`、`top_p`、`top_k`、`repetition_penalty` 以及**一个或多个** `eos_token_id`。推理模型（如 R1 / Qwen3 Thinking）对温度和停止符尤为敏感。",
    summary_en: "Configures autoregressive decoding defaults: `do_sample`, `temperature`, `top_p`, `top_k`, `repetition_penalty`, and single or array `eos_token_id`. Reasoning models are highly sensitive to these defaults.",
    loader_zh: "vLLM / SGLang 启动服务时会读取其中的 `eos_token_id` 列表与默认采样配置；自定义 Rust 推理循环若忽略此文件极易漏配停止符。",
    loader_en: "vLLM and SGLang load default sampling parameters and stop token IDs from here; custom Rust inference loops must mirror multi-EOS lists.",
    pitfall_zh: "Llama-3 与 Qwen-Chat 的 `eos_token_id` 往往是数组（如 `[151645, 151643]` 分别对应 `<|im_end|>` 与 `<|endoftext|>`），只检查 `config.json` 里单个标量 `eos_token_id` 会导致生成停不下来。",
    pitfall_en: "Chat models frequently declare an array `eos_token_id` (e.g. `[151645, 151643]`); checking only the single scalar in `config.json` causes runaway generation.",
    snippet: r#"{
  "bos_token_id": 151643,
  "eos_token_id": [151645, 151643],
  "do_sample": true,
  "temperature": 0.6,
  "top_p": 0.95,
  "top_k": 20
}"#,
  },
  ConfigFileSpec {
    kind: ConfigFileKind::PreprocessorConfig,
    filenames: "preprocessor_config.json / video_preprocessor_config.json",
    modalities_zh: "VLM / Audio ASR / Omni",
    modalities_en: "VLM / Audio ASR / Omni",
    role_zh: "图像、视频与音频信号前端预处理",
    role_en: "Image, Video & Audio Frontend Spec",
    summary_zh: "规定非文本输入如何变成模型可接收的张量：视觉侧定义动态分辨率上下限（`min_pixels`/`max_pixels`）、`patch_size`、`merge_size`、归一化均值方差；语音侧定义 `sampling_rate`、`feature_size`（Mel 频带数）、`n_fft` 与 `hop_length`。",
    summary_en: "Specifies how raw pixels or waveforms become tensors: dynamic resolution bounds (`min_pixels`/`max_pixels`), `patch_size`, `merge_size`, and RGB normalization for VLMs; `sampling_rate`, `feature_size` (Mel bins), `n_fft`, and `hop_length` for audio.",
    loader_zh: "视觉处理器（`Qwen2VLImageProcessor`）与声学特征提取器（`WhisperFeatureExtractor`）严格依赖这些参数，否则视觉 Token 数量或梅尔语谱频率轴会完全错位。",
    loader_en: "Vision processors and audio feature extractors rely on exact pixel/STFT parameters; mismatches silently corrupt visual token counts or Mel spectrograms.",
    pitfall_zh: "`max_pixels` 调得过大时，单张高分辨率图会被切成数千个视觉 Token，直接撑爆 Prefill 显存；音频若未重采样至 `sampling_rate`（如 16000Hz）会导致识别乱码。",
    pitfall_en: "Setting `max_pixels` too high explodes visual token count during prefill; feeding audio at 44.1kHz into a 16kHz model silently degrades transcription.",
    snippet: r#"{
  "image_processor_type": "Qwen2VLImageProcessor",
  "patch_size": 14,
  "merge_size": 2,
  "min_pixels": 3136,
  "max_pixels": 12845056,
  "image_mean": [0.48145466, 0.4578275, 0.40821073]
}"#,
  },
  ConfigFileSpec {
    kind: ConfigFileKind::DiffusionPipeline,
    filenames: "model_index.json & <slot>/config.json & scheduler/scheduler_config.json",
    modalities_zh: "Diffusion (图像 / 视频生成)",
    modalities_en: "Diffusion (Image / Video)",
    role_zh: "多组件扩散管线编排与噪声调度",
    role_en: "Multi-Component Pipeline & Noise Scheduler",
    summary_zh: "Diffusers 仓库根目录不是单一 `config.json`，而是 `model_index.json` 清单，指向各子目录槽位：`text_encoder*`（Prompt 编码）、`transformer`/`unet`（去噪主干）、`vae`（潜空间编解码）与 `scheduler/scheduler_config.json`（FlowMatch / Euler 时间步调度）。",
    summary_en: "Diffusers repos use a root `model_index.json` manifest wiring subdirectories: `text_encoder*`, `transformer`/`unet` denoiser, `vae` latent codec, and `scheduler/scheduler_config.json` (FlowMatch/Euler timestep schedule).",
    loader_zh: "`DiffusionPipeline.from_pretrained` 先读 `model_index.json` 获取每个槽位的 `[library, class_name]`，再分别加载对应子目录的 `config.json` 与权重。",
    loader_en: "`DiffusionPipeline.from_pretrained` parses `[library, class_name]` tuples from `model_index.json` and loads each slot's subdirectory independently.",
    pitfall_zh: "高分辨率图像/视频生成（FLUX / SD3.5 / Wan2.1）强依赖 `scheduler_config.json` 中的 `shift` / `use_dynamic_shifting`（Time-Shift），以及 `vae/config.json` 的 `scaling_factor` 与 `shift_factor`，漏乘会导致画面灰暗或色偏。",
    pitfall_en: "Flow-matching models (FLUX/SD3.5/Wan2.1) depend heavily on `shift`/`use_dynamic_shifting` in scheduler config and `scaling_factor`/`shift_factor` in `vae/config.json`.",
    snippet: r#"{
  "_class_name": "FluxPipeline",
  "text_encoder": ["transformers", "CLIPTextModel"],
  "text_encoder_2": ["transformers", "T5EncoderModel"],
  "transformer": ["diffusers", "FluxTransformer2DModel"],
  "vae": ["diffusers", "AutoencoderKL"],
  "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"]
}"#,
  },
  ConfigFileSpec {
    kind: ConfigFileKind::EmbeddingModules,
    filenames: "modules.json & 1_Pooling/config.json",
    modalities_zh: "Embedding / Reranker",
    modalities_en: "Embedding / Reranker",
    role_zh: "句向量流水线、池化与归一化定义",
    role_en: "Sentence Embedding Pipeline & Pooling Spec",
    summary_zh: "Sentence-Transformers 格式用 `modules.json` 串联阶段：`0_Transformer`（编码主干）→ `1_Pooling/config.json`（决定取 CLS、Mean 还是 Last-Token 池化）→ 可选 `2_Normalize`（L2 归一化以便直接点积算余弦相似度）。",
    summary_en: "Sentence-Transformers chains stages via `modules.json`: `0_Transformer` backbone -> `1_Pooling/config.json` (CLS, Mean, or Last-Token pooling) -> optional `2_Normalize` (L2 unit norm for cosine similarity).",
    loader_zh: "TEI（Text Embeddings Inference）、Candle 与 `sentence-transformers` 必须按 `1_Pooling/config.json` 聚合 token 隐藏状态，不同池化策略互不兼容。",
    loader_en: "TEI, Candle, and `sentence-transformers` pool token hidden states strictly according to `1_Pooling/config.json`.",
    pitfall_zh: "把本该用 `last_token` 池化的解码器 Embedding 模型（如 Qwen3-Embedding）错用成 `mean` 或 `cls` 池化，余弦相似度虽能算出数值但检索召回率会断崖式下跌。",
    pitfall_en: "Applying `mean` or `cls` pooling to a decoder-based embedding model trained with `last_token` pooling silently destroys retrieval recall.",
    snippet: r#"{
  "word_embedding_dimension": 1024,
  "pooling_mode_cls_token": true,
  "pooling_mode_mean_tokens": false,
  "pooling_mode_lasttoken": false
}"#,
  },
  ConfigFileSpec {
    kind: ConfigFileKind::AdapterConfig,
    filenames: "adapter_config.json & adapter_model.safetensors",
    modalities_zh: "微调适配器 (LoRA / DoRA / QLoRA)",
    modalities_en: "Fine-Tuning Adapters (LoRA / DoRA)",
    role_zh: "低秩增量权重注入规范",
    role_en: "Low-Rank Delta Injection Spec",
    summary_zh: "PEFT 微调产物并不保存完整基座权重，而是用 `adapter_config.json` 声明 `base_model_name_or_path`、低秩维度 `r`、缩放系数 `lora_alpha`、注入模块名 `target_modules` 及全量保存层 `modules_to_save`。",
    summary_en: "PEFT checkpoints store only deltas: `adapter_config.json` specifies `base_model_name_or_path`, rank `r`, scaling `lora_alpha`, injected `target_modules`, and full-weight `modules_to_save`.",
    loader_zh: "推理时按 `W_eff = W_0 + (lora_alpha / r) · B @ A`（若启用 `use_rslora` 则为 `lora_alpha / sqrt(r)`）热挂载或离线合并进基座。",
    loader_en: "Loaded at runtime or merged via `W_eff = W_0 + (lora_alpha / r) · B @ A` (or `lora_alpha / sqrt(r)` when `use_rslora` is true).",
    pitfall_zh: "若训练时扩充了词表或新加了 special token，必须检查 `modules_to_save` 是否包含 `embed_tokens` 和 `lm_head`，否则推理加载后新 token 权重会丢失。",
    pitfall_en: "When adding new special tokens during fine-tuning, `modules_to_save` must include `embed_tokens`/`lm_head`, or merged weights will revert new token rows.",
    snippet: r#"{
  "peft_type": "LORA",
  "base_model_name_or_path": "Qwen/Qwen3-8B",
  "r": 32,
  "lora_alpha": 64,
  "lora_dropout": 0.05,
  "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
  "use_rslora": false
}"#,
  },
];

// ─────────────────────────────────────────────────────────────────────────────
// 2. 五大模态仓库文件树全景蓝图
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "server")]
pub const REPO_MODALITY_BLUEPRINTS: &[RepoModalityBlueprint] = &[
  RepoModalityBlueprint {
    id: "llm",
    label_zh: "大语言模型 (LLM)",
    label_en: "Large Language Model (LLM)",
    example_repo: "deepseek-ai/DeepSeek-V3 / Qwen/Qwen3-30B-A3B",
    example_slug: Some("deepseek-v3"),
    flow_zh: "tokenizer_config/tokenizer.json 切分文本 → config.json 构建层栈与 KV Cache → safetensors.index.json 映射权重分片 → generation_config.json 控制采样与多 EOS 停止",
    flow_en: "tokenizer.json encodes prompt -> config.json allocates layers & KV cache -> safetensors.index.json maps weight shards -> generation_config.json drives sampling & EOS stopping",
    items: &[
      RepoTreeItem {
        path: "config.json",
        required: true,
        file_kind: ConfigFileKind::ModelConfig,
        note_zh: "主干维度、注意力(GQA/MLA/SSM)、MoE、RoPE 与量化元数据",
        note_en: "Backbone dims, Attention (GQA/MLA/SSM), MoE, RoPE & quantization",
      },
      RepoTreeItem {
        path: "model.safetensors.index.json",
        required: false,
        file_kind: ConfigFileKind::SafetensorsIndex,
        note_zh: "多分片权重张量名 → 分片文件映射与总字节数（单分片小模型可省略）",
        note_en: "Multi-shard tensor-to-file map & total byte size (optional for single shard)",
      },
      RepoTreeItem {
        path: "model-00001-of-000xx.safetensors",
        required: true,
        file_kind: ConfigFileKind::SafetensorsIndex,
        note_zh: "实际权重分片；文件头内嵌每个张量的精确 shape / dtype / offset",
        note_en: "Weight shards with embedded JSON header of exact shapes, dtypes & offsets",
      },
      RepoTreeItem {
        path: "tokenizer.json",
        required: true,
        file_kind: ConfigFileKind::TokenizerConfig,
        note_zh: "自包含 Fast Tokenizer（词表 + BPE/Unigram 规则），Rust tokenizers 直接加载",
        note_en: "Self-contained Fast Tokenizer (vocab + merges) loaded directly by Rust tokenizers",
      },
      RepoTreeItem {
        path: "tokenizer_config.json",
        required: true,
        file_kind: ConfigFileKind::TokenizerConfig,
        note_zh: "BOS/EOS/PAD 特殊符号、ChatML / Thinking 对话模板定义",
        note_en: "Special tokens (BOS/EOS/PAD) and ChatML / reasoning chat template",
      },
      RepoTreeItem {
        path: "generation_config.json",
        required: false,
        file_kind: ConfigFileKind::GenerationConfig,
        note_zh: "官方推荐 temperature / top_p / top_k 与多停止符 eos_token_id 数组",
        note_en: "Recommended temperature / top_p / top_k and multi-EOS stop token array",
      },
    ],
  },
  RepoModalityBlueprint {
    id: "vlm",
    label_zh: "多模态视觉语言 (VLM)",
    label_en: "Vision-Language Model (VLM)",
    example_repo: "Qwen/Qwen2.5-VL-72B-Instruct / meta-llama/Llama-4-Scout",
    example_slug: Some("qwen2-5-vl-72b"),
    flow_zh: "preprocessor_config.json 将图像/视频切 Patch 归一化 → config.json(vision_config) 编码视觉特征并投影 → config.json(text_config) 与文本 Token 拼接自回归解码",
    flow_en: "preprocessor_config.json patches & normalizes frames -> vision_config encodes visual tokens -> text_config fuses vision + text tokens for autoregressive decoding",
    items: &[
      RepoTreeItem {
        path: "config.json",
        required: true,
        file_kind: ConfigFileKind::ModelConfig,
        note_zh: "顶层统筹 vision_config（ViT/SigLIP 视觉塔）+ text_config（语言主干）+ 视觉投影维度",
        note_en: "Combines nested vision_config (ViT/SigLIP tower), text_config & projector dims",
      },
      RepoTreeItem {
        path: "preprocessor_config.json",
        required: true,
        file_kind: ConfigFileKind::PreprocessorConfig,
        note_zh: "动态分辨率 min_pixels/max_pixels、patch_size、merge_size 与 RGB 均值方差",
        note_en: "Dynamic resolution bounds, patch_size, merge_size & RGB normalization",
      },
      RepoTreeItem {
        path: "video_preprocessor_config.json",
        required: false,
        file_kind: ConfigFileKind::PreprocessorConfig,
        note_zh: "视频抽帧率 fps、temporal_patch_size 与单视频最大帧数限制",
        note_en: "Video frame sampling fps, temporal_patch_size & max frame budget",
      },
      RepoTreeItem {
        path: "chat_template.jinja / tokenizer_config.json",
        required: true,
        file_kind: ConfigFileKind::TokenizerConfig,
        note_zh: "视觉占位符（如 <|vision_start|><|image_pad|><|vision_end|>）拼接规范",
        note_en: "Visual placeholder injection template (<|vision_start|>, <|image_pad|>)",
      },
      RepoTreeItem {
        path: "model.safetensors.index.json",
        required: true,
        file_kind: ConfigFileKind::SafetensorsIndex,
        note_zh: "同时索引 visual.* 视觉塔权重、merger/projector 与 model.layers.* 语言层权重",
        note_en: "Indexes visual.* tower weights, projector/merger, and language model.layers.*",
      },
    ],
  },
  RepoModalityBlueprint {
    id: "diffusion",
    label_zh: "扩散与流匹配生成 (Diffusion)",
    label_en: "Diffusion & Flow-Matching",
    example_repo: "black-forest-labs/FLUX.1-dev / Wan-AI/Wan2.1-T2V-14B",
    example_slug: Some("flux-1-dev"),
    flow_zh: "model_index.json 装配管线 → text_encoder* 编码提示词 → scheduler_config.json 生成时间步 → transformer(DiT) 迭代去噪 → vae 解码潜空间为图像/视频",
    flow_en: "model_index.json wires slots -> text_encoder* encodes prompt -> scheduler_config.json builds timesteps -> transformer denoises latents -> vae decodes pixels/video",
    items: &[
      RepoTreeItem {
        path: "model_index.json",
        required: true,
        file_kind: ConfigFileKind::DiffusionPipeline,
        note_zh: "多组件编排清单：声明每个子目录对应的库与类名（无顶层单一 config.json）",
        note_en: "Multi-slot manifest mapping each subdirectory to [library, class_name]",
      },
      RepoTreeItem {
        path: "transformer/config.json (或 unet/config.json)",
        required: true,
        file_kind: ConfigFileKind::DiffusionPipeline,
        note_zh: "去噪主干架构：MMDiT 双流/单流层数、patch_size、joint_attention_dim、in_channels",
        note_en: "Denoiser architecture: MMDiT/DiT layers, patch_size, joint_attention_dim, in_channels",
      },
      RepoTreeItem {
        path: "scheduler/scheduler_config.json",
        required: true,
        file_kind: ConfigFileKind::DiffusionPipeline,
        note_zh: "扩散/流匹配噪声调度：num_train_timesteps、shift、use_dynamic_shifting",
        note_en: "Noise schedule & FlowMatch parameters: num_train_timesteps, shift, dynamic shifting",
      },
      RepoTreeItem {
        path: "vae/config.json",
        required: true,
        file_kind: ConfigFileKind::DiffusionPipeline,
        note_zh: "潜空间编解码器：latent_channels、scaling_factor、shift_factor、空间/时间压缩倍率",
        note_en: "Latent codec: latent_channels, scaling_factor, shift_factor & compression ratio",
      },
      RepoTreeItem {
        path: "text_encoder*/config.json & tokenizer*/",
        required: true,
        file_kind: ConfigFileKind::TokenizerConfig,
        note_zh: "前置条件编码器（如 CLIP + T5-XXL 或 UMT5），仅在去噪前单次执行",
        note_en: "Prompt encoders (e.g. CLIP + T5-XXL / UMT5), executed once before denoising",
      },
    ],
  },
  RepoModalityBlueprint {
    id: "embedding",
    label_zh: "向量表征与重排序 (Embedding)",
    label_en: "Embedding & Reranker",
    example_repo: "BAAI/bge-m3 / Qwen/Qwen3-Embedding-8B",
    example_slug: Some("bge-m3"),
    flow_zh: "tokenizer.json 编码文本 → config.json 单次前向输出序列隐藏态 → modules.json 按序经 1_Pooling/config.json 池化与 2_Normalize L2 归一输出稠密向量",
    flow_en: "tokenizer.json encodes text -> config.json runs single forward pass -> modules.json applies 1_Pooling/config.json & 2_Normalize to emit unit-norm vectors",
    items: &[
      RepoTreeItem {
        path: "config.json",
        required: true,
        file_kind: ConfigFileKind::ModelConfig,
        note_zh: "双向编码器（XLM-RoBERTa/BERT）或无 LM Head 的解码主干维度与最大输入长度",
        note_en: "Bidirectional encoder or causal backbone dimensions & max sequence length",
      },
      RepoTreeItem {
        path: "modules.json",
        required: true,
        file_kind: ConfigFileKind::EmbeddingModules,
        note_zh: "声明 Sentence-Transformers 阶段顺序（0_Transformer → 1_Pooling → 2_Normalize）",
        note_en: "Declares ordered Sentence-Transformers stages (Transformer -> Pooling -> Normalize)",
      },
      RepoTreeItem {
        path: "1_Pooling/config.json",
        required: true,
        file_kind: ConfigFileKind::EmbeddingModules,
        note_zh: "决定句向量提取方式：CLS、Mean 均值或 Last-Token 池化及输出维度",
        note_en: "Specifies pooling strategy: CLS, Mean, or Last-Token & output dimension",
      },
      RepoTreeItem {
        path: "tokenizer.json & tokenizer_config.json",
        required: true,
        file_kind: ConfigFileKind::TokenizerConfig,
        note_zh: "编码分词与截断/填充规则（注意检索模型常对 Query 加指令前缀）",
        note_en: "Tokenization & truncation rules (retrieval models often require query instructions)",
      },
    ],
  },
  RepoModalityBlueprint {
    id: "lora",
    label_zh: "参数高效微调 (LoRA / DoRA)",
    label_en: "PEFT Adapter (LoRA / DoRA)",
    example_repo: "user/qwen3-8b-rust-lora",
    example_slug: Some("qwen3-8b"),
    flow_zh: "adapter_config.json 定位基座模型与注入目标层 → adapter_model.safetensors 仅存储低秩矩阵 A/B → 推理时热插拔或按 alpha/r 合并进基座权重",
    flow_en: "adapter_config.json points to base model & target modules -> adapter_model.safetensors stores low-rank A/B -> runtime attach or merge via alpha/r",
    items: &[
      RepoTreeItem {
        path: "adapter_config.json",
        required: true,
        file_kind: ConfigFileKind::AdapterConfig,
        note_zh: "基座仓库名、秩 r、lora_alpha、target_modules、use_rslora/use_dora 与 modules_to_save",
        note_en: "Base repo ID, rank r, lora_alpha, target_modules, rsLoRA/DoRA & modules_to_save",
      },
      RepoTreeItem {
        path: "adapter_model.safetensors",
        required: true,
        file_kind: ConfigFileKind::SafetensorsIndex,
        note_zh: "仅含 *.lora_A.weight / *.lora_B.weight（以及 modules_to_save 全量层），体积仅数十至数百 MB",
        note_en: "Contains only *.lora_A / *.lora_B (and modules_to_save), typically tens to hundreds of MB",
      },
      RepoTreeItem {
        path: "tokenizer_config.json (可选)",
        required: false,
        file_kind: ConfigFileKind::TokenizerConfig,
        note_zh: "若微调阶段修改了 chat_template 或新增特殊 token，必须与适配器一同分发",
        note_en: "Required whenever fine-tuning modified chat_template or added special tokens",
      },
    ],
  },
];

// ─────────────────────────────────────────────────────────────────────────────
// 3. ~88 个高频配置字段双语工程词典（Single Source of Truth）
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "server")]
pub const CONFIG_FIELD_DOCS: &[ConfigFieldDoc] = &[
  // ── A. Backbone & Identity（主干维度与身份）──
  ConfigFieldDoc {
    key: "architectures",
    aliases: &[],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::Backbone,
    value_type: "string[]",
    summary_zh: "模型入口类列表（如 `DeepseekV3ForCausalLM`、`Qwen3MoeForCausalLM`）。推理框架据此决定加载哪套前向计算图。",
    summary_en: "Model class list (e.g. `DeepseekV3ForCausalLM`). Inference engines dispatch the exact forward graph from this field.",
    formula_impact: None,
    focus: None,
    example_model_slug: Some("deepseek-v3"),
    lab_link: None,
  },
  ConfigFieldDoc {
    key: "model_type",
    aliases: &[],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::Backbone,
    value_type: "string",
    summary_zh: "架构家族标识符（如 `llama`、`qwen3_moe`、`deepseek_v3`、`gemma3`），决定默认权重命名映射与配置解析分支。",
    summary_en: "Architecture family identifier (e.g. `llama`, `qwen3_moe`, `deepseek_v3`) driving config parsing and tensor naming conventions.",
    formula_impact: None,
    focus: None,
    example_model_slug: Some("qwen3-30b-a3b"),
    lab_link: None,
  },
  ConfigFieldDoc {
    key: "hidden_size",
    aliases: &["n_embd", "d_model", "dim"],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::Backbone,
    value_type: "int",
    summary_zh: "残差流（Residual Stream）主干隐藏维度 D。几乎所有线性投影、归一化与词嵌入矩阵的输入/输出宽度都由 D 锚定。",
    summary_en: "Residual stream hidden width D. Anchors embeddings, layer norms, and almost every projection matrix shape.",
    formula_impact: Some("Dense 单层参数量大致按 O(D²) 缩放；Embed = V × D"),
    focus: Some(InspectorFocus::Embedding),
    example_model_slug: Some("deepseek-v3"),
    lab_link: None,
  },
  ConfigFieldDoc {
    key: "num_hidden_layers",
    aliases: &["n_layer", "num_layers", "decoder_layers"],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::Backbone,
    value_type: "int",
    summary_zh: "主干 Transformer / 混合状态层堆叠深度 L。直接线性倍增总参数量、单 Token 串行延迟与 KV Cache 总显存。",
    summary_en: "Total stacked decoder/encoder depth L. Scales total parameters, per-token latency, and KV cache memory linearly.",
    formula_impact: Some("Total KV Cache ∝ L × B × S × kv_elems_per_layer"),
    focus: Some(InspectorFocus::Attention),
    example_model_slug: Some("llama-3-3-70b"),
    lab_link: None,
  },
  ConfigFieldDoc {
    key: "vocab_size",
    aliases: &[],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::Backbone,
    value_type: "int",
    summary_zh: "Embedding 查表行数与 LM Head 输出类别数 V。常被向上填充至 64/128 的倍数以匹配 GPU Tensor Core 内存对齐。",
    summary_en: "Embedding table rows and LM Head logit width V. Frequently padded to multiples of 64/128 for Tensor Core alignment.",
    formula_impact: Some("Embed = V × D；若 tie_word_embeddings=false，总词表参数 = 2 × V × D"),
    focus: Some(InspectorFocus::Embedding),
    example_model_slug: Some("qwen3-8b"),
    lab_link: None,
  },
  ConfigFieldDoc {
    key: "tie_word_embeddings",
    aliases: &[],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::Backbone,
    value_type: "bool",
    summary_zh: "是否令输入 `embed_tokens` 与输出 `lm_head` 共享同一份权重矩阵。小模型常设为 `true` 省参数，大模型多设为 `false` 提升表征自由度。",
    summary_en: "Whether input `embed_tokens` and output `lm_head` share weights. Often `true` for small models to save params, `false` for large models.",
    formula_impact: Some("true 时省去 V × D 参数（152K×4096 ≈ 0.62B 参数 / 1.2GB BF16）"),
    focus: Some(InspectorFocus::LmHead),
    example_model_slug: Some("qwen3-8b"),
    lab_link: None,
  },
  ConfigFieldDoc {
    key: "rms_norm_eps",
    aliases: &["layer_norm_eps", "norm_eps"],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::Backbone,
    value_type: "float",
    summary_zh: "RMSNorm / LayerNorm 分母中防止除零的极小常数值 ε（常见 `1e-5` 或 `1e-6`），实现 Rust/CUDA 核函数时必须与官方严格一致。",
    summary_en: "Numerical stability epsilon ε inside RMSNorm/LayerNorm denominator (typically `1e-5` or `1e-6`). Must match exactly for logit parity.",
    formula_impact: Some("y = x / sqrt(mean(x²) + ε) · γ（须在 FP32 累加）"),
    focus: Some(InspectorFocus::NormAndResidual),
    example_model_slug: Some("llama-3-1-8b"),
    lab_link: Some(("/ai/components/rmsnorm-and-adaln-zero", "RMSNorm 精度实验台")),
  },
  ConfigFieldDoc {
    key: "num_nextn_predict_layers",
    aliases: &[],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::Backbone,
    value_type: "int",
    summary_zh: "Multi-Token Prediction（MTP）辅助预测头层数（如 DeepSeek-V3 为 1）。训练时提升表征密度，推理时可直接充当原生投机解码草稿头。",
    summary_en: "Number of Multi-Token Prediction (MTP) auxiliary heads (e.g. 1 in DeepSeek-V3). Boosts training signal and serves as native speculative draft head.",
    formula_impact: Some("额外增加 N × (投影 + 单层 Transformer) 参数；部署可按需裁剪"),
    focus: Some(InspectorFocus::MtpHead),
    example_model_slug: Some("deepseek-v3"),
    lab_link: Some(("/ai/components/moe-routing-quant-and-speculative-decoding", "投机解码实验台")),
  },
  ConfigFieldDoc {
    key: "bos_token_id",
    aliases: &[],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::Backbone,
    value_type: "int | null",
    summary_zh: "序列起始符（Beginning of Sequence）的词表 ID。部分模型（如 Qwen2/3）对话模式下不自动加 BOS，强行拼接会影响首词概率。",
    summary_en: "Beginning-of-Sequence token ID. Models like Qwen2/3 omit BOS in chat mode; prepending it unexpectedly shifts output logits.",
    formula_impact: None,
    focus: Some(InspectorFocus::Embedding),
    example_model_slug: Some("llama-3-1-8b"),
    lab_link: None,
  },
  ConfigFieldDoc {
    key: "eos_token_id",
    aliases: &[],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::Backbone,
    value_type: "int | int[]",
    summary_zh: "序列结束停止符 ID。注意 `config.json` 常只存单个基础 EOS，对话模型的完整多停止符列表应同时核对 `generation_config.json`。",
    summary_en: "End-of-Sequence stop token ID(s). `config.json` often lists only one base EOS; always check `generation_config.json` for chat stop-token arrays.",
    formula_impact: None,
    focus: Some(InspectorFocus::LmHead),
    example_model_slug: Some("llama-3-1-8b"),
    lab_link: None,
  },
  ConfigFieldDoc {
    key: "pad_token_id",
    aliases: &[],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::Backbone,
    value_type: "int | null",
    summary_zh: "批量对齐填充符 ID，配合 Attention Mask 屏蔽无效位置。许多 Base LLM 缺省为 `null`，批量推理或 SFT 时需显式指定。",
    summary_en: "Padding token ID masked out during batched attention. Often `null` in base LLMs and must be assigned explicitly for batched inference or SFT.",
    formula_impact: None,
    focus: Some(InspectorFocus::Embedding),
    example_model_slug: Some("qwen3-8b"),
    lab_link: None,
  },
  ConfigFieldDoc {
    key: "transformers_version",
    aliases: &[],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::Backbone,
    value_type: "string",
    summary_zh: "发布权重时使用的 HuggingFace Transformers 最低参考版本，用于判断当前推理栈是否支持该模型的定制算子。",
    summary_en: "Minimum HuggingFace Transformers version used at checkpoint export, indicating whether older runtimes support its custom layers.",
    formula_impact: None,
    focus: None,
    example_model_slug: Some("qwen3-30b-a3b"),
    lab_link: None,
  },
  ConfigFieldDoc {
    key: "auto_map",
    aliases: &[],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::Backbone,
    value_type: "object",
    summary_zh: "自定义 Python 模型代码映射表（指向仓库内 `modeling_*.py`）。存在此字段说明上游标准库尚未内置，Python 加载需 `trust_remote_code=True`。",
    summary_en: "Maps Auto classes to custom repo Python files (`modeling_*.py`), signaling custom architecture code requiring `trust_remote_code=True` in Python.",
    formula_impact: None,
    focus: None,
    example_model_slug: Some("deepseek-v3"),
    lab_link: None,
  },
  ConfigFieldDoc {
    key: "use_cache",
    aliases: &[],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::Backbone,
    value_type: "bool",
    summary_zh: "前向推理默认是否返回并复用 KV Cache。训练开启梯度检查点（Gradient Checkpointing）时必须关闭 `use_cache=false`。",
    summary_en: "Whether forward passes return and reuse KV cache by default. Must be set `false` during training with gradient checkpointing.",
    formula_impact: None,
    focus: Some(InspectorFocus::Attention),
    example_model_slug: Some("llama-3-1-8b"),
    lab_link: None,
  },
  ConfigFieldDoc {
    key: "initializer_range",
    aliases: &[],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::Backbone,
    value_type: "float",
    summary_zh: "预训练权重正态初始化标准差（通常 `0.02`），仅在从零初始化模型时生效，推理期加载已训练权重时不使用。",
    summary_en: "Standard deviation for random weight initialization during pretraining (typically `0.02`); unused when loading pretrained checkpoints.",
    formula_impact: None,
    focus: None,
    example_model_slug: Some("llama-3-1-8b"),
    lab_link: None,
  },

  // ── B. Attention & KV/State Cache（注意力与 KV / 状态缓存）──
  ConfigFieldDoc {
    key: "num_attention_heads",
    aliases: &["n_head", "decoder_attention_heads"],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::AttentionKv,
    value_type: "int",
    summary_zh: "Query 注意力头数 H_q。决定并行子空间切分粒度以及张量并行（TP）可整除的最大卡数上限。",
    summary_en: "Number of Query attention heads H_q. Sets subspace granularity and constrains Tensor Parallelism (TP) divisibility.",
    formula_impact: Some("Q 投影输出宽度 = H_q × head_dim（未必等于 hidden_size）"),
    focus: Some(InspectorFocus::Attention),
    example_model_slug: Some("qwen3-8b"),
    lab_link: Some(("/ai/components/attention-mla-and-flash-online-softmax", "GQA & FlashAttn 实验台")),
  },
  ConfigFieldDoc {
    key: "num_key_value_heads",
    aliases: &["multi_query_group_num"],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::AttentionKv,
    value_type: "int",
    summary_zh: "Key/Value 共享头数 H_kv（GQA 分组注意力核心旋钮）。当 `H_kv < H_q` 时，每 `H_q / H_kv` 个 Q 头共享 1 组 KV，成倍压缩 KV Cache。",
    summary_en: "Number of Key/Value heads H_kv for GQA. Every `H_q / H_kv` query heads share one KV head, shrinking KV cache proportionally.",
    formula_impact: Some("单层单 Token KV 元素 = 2 × H_kv × head_dim；压缩比 = H_q / H_kv"),
    focus: Some(InspectorFocus::Attention),
    example_model_slug: Some("llama-3-1-8b"),
    lab_link: Some(("/ai/components/attention-mla-and-flash-online-softmax", "GQA & FlashAttn 实验台")),
  },
  ConfigFieldDoc {
    key: "head_dim",
    aliases: &["kv_channels"],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::AttentionKv,
    value_type: "int",
    summary_zh: "单个注意力头的向量维度 d_h。现代模型（如 Qwen3、Gemma-3）常解耦 `H_q × head_dim != hidden_size`，不可盲目用 `hidden_size / num_attention_heads` 推断！",
    summary_en: "Per-head dimension d_h. Modern models (Qwen3, Gemma-3) decouple `H_q × head_dim` from `hidden_size`, so never assume `D / H_q` when `head_dim` is explicit.",
    formula_impact: Some("显式指定时，Q=[D, H_q·d_h], K/V=[D, H_kv·d_h], O=[H_q·d_h, D]"),
    focus: Some(InspectorFocus::Attention),
    example_model_slug: Some("qwen3-30b-a3b"),
    lab_link: Some(("/ai/components/attention-mla-and-flash-online-softmax", "GQA & FlashAttn 实验台")),
  },
  ConfigFieldDoc {
    key: "kv_lora_rank",
    aliases: &[],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::AttentionKv,
    value_type: "int",
    summary_zh: "DeepSeek MLA（多头潜在注意力）的 KV 联合低秩压缩维度 d_c（如 512）。推理期仅缓存压缩潜在向量 c_KV 与解耦 RoPE 键，KV Cache 压缩比可达 50× 以上。",
    summary_en: "DeepSeek MLA joint KV latent compression rank d_c (e.g. 512). Inference caches only compressed c_KV plus decoupled RoPE key, yielding >50× KV savings.",
    formula_impact: Some("MLA 单层单 Token 缓存元素 = kv_lora_rank + qk_rope_head_dim（如 512+64=576）"),
    focus: Some(InspectorFocus::Attention),
    example_model_slug: Some("deepseek-v3"),
    lab_link: Some(("/ai/math/svd-eckart-young-pseudoinverse", "SVD 低秩分解数学证明")),
  },
  ConfigFieldDoc {
    key: "q_lora_rank",
    aliases: &[],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::AttentionKv,
    value_type: "int | null",
    summary_zh: "DeepSeek MLA 的 Query 低秩投影维度（如 V3 为 1536；V2-Lite 为 `null` 不压缩 Q）。降低训练激活显存并保持矩阵吸收等价性。",
    summary_en: "DeepSeek MLA Query low-rank bottleneck dimension (1536 in V3; `null` in V2-Lite). Reduces training activation memory while preserving weight absorption.",
    formula_impact: Some("Q 拆分为 q_a_proj [D→r_q] + RMSNorm + q_b_proj [r_q→H·(d_nope+d_rope)]"),
    focus: Some(InspectorFocus::Attention),
    example_model_slug: Some("deepseek-v3"),
    lab_link: Some(("/ai/components/attention-mla-and-flash-online-softmax", "MLA 矩阵吸收实验台")),
  },
  ConfigFieldDoc {
    key: "qk_nope_head_dim",
    aliases: &[],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::AttentionKv,
    value_type: "int",
    summary_zh: "MLA 注意力中**不施加 RoPE** 的 Q/K 内容子维度（如 128）。正因这部分无位置旋转，推理期才能将 W_UK 数学吸收到 W_Q 中实现免解压计算。",
    summary_en: "Non-RoPE content head dimension in MLA (e.g. 128). Keeping this slice position-free allows absorbing W_UK into W_Q during decoding.",
    formula_impact: Some("单头总 Q/K 维度 = qk_nope_head_dim + qk_rope_head_dim"),
    focus: Some(InspectorFocus::Attention),
    example_model_slug: Some("deepseek-v3"),
    lab_link: Some(("/ai/components/attention-mla-and-flash-online-softmax", "MLA 矩阵吸收实验台")),
  },
  ConfigFieldDoc {
    key: "qk_rope_head_dim",
    aliases: &[],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::AttentionKv,
    value_type: "int",
    summary_zh: "MLA 解耦 RoPE（Decoupled RoPE）专用的位置子维度（如 64）。所有注意力头共享同一份 k_R，仅额外占用 64 标量/Token 的 KV Cache。",
    summary_en: "Decoupled RoPE position head dimension in MLA (e.g. 64). Shared across heads, adding only 64 scalars per token per layer to KV cache.",
    formula_impact: Some("k_R 缓存开销 = qk_rope_head_dim 标量/Token/层"),
    focus: Some(InspectorFocus::Attention),
    example_model_slug: Some("deepseek-v3"),
    lab_link: Some(("/ai/components/rope-and-yarn-extrapolation", "RoPE 旋转位置实验台")),
  },
  ConfigFieldDoc {
    key: "v_head_dim",
    aliases: &[],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::AttentionKv,
    value_type: "int",
    summary_zh: "MLA 中每个 Value 头的还原维度（如 128），决定输出投影 `o_proj` 输入宽度 `H × v_head_dim`。",
    summary_en: "Per-head Value dimension in MLA (e.g. 128), setting `o_proj` input width to `H × v_head_dim`.",
    formula_impact: Some("o_proj = [H × v_head_dim → D]"),
    focus: Some(InspectorFocus::Attention),
    example_model_slug: Some("deepseek-v3"),
    lab_link: Some(("/ai/components/attention-mla-and-flash-online-softmax", "MLA 矩阵吸收实验台")),
  },
  ConfigFieldDoc {
    key: "attention_bias",
    aliases: &["add_qkv_bias"],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::AttentionKv,
    value_type: "bool",
    summary_zh: "Q/K/V/O 线性投影是否附带偏置项（Bias）。Llama/Qwen3 多为 `false`，Qwen2/2.5 为 QKV 开启 Bias 以缓解长文本注意力离群值。",
    summary_en: "Whether Q/K/V/O projections include additive bias. `false` in Llama/Qwen3; Qwen2/2.5 uses QKV bias to stabilize long-context attention.",
    formula_impact: None,
    focus: Some(InspectorFocus::Attention),
    example_model_slug: Some("qwen2-5-72b"),
    lab_link: None,
  },
  ConfigFieldDoc {
    key: "qk_layernorm",
    aliases: &["qk_norm"],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::AttentionKv,
    value_type: "bool",
    summary_zh: "是否在 Q/K 点积前对每个头独立执行 RMSNorm/LayerNorm（如 Qwen3、Gemma-3、FLUX）。显著抑制深层大模型训练期注意力 Logits 爆炸。",
    summary_en: "Applies per-head RMSNorm/LayerNorm to Q and K before dot-product (Qwen3, Gemma-3, FLUX), preventing attention logit blow-up at scale.",
    formula_impact: Some("每层新增 2 × head_dim 归一化缩放参数，零显存开销但显著提升数值稳定性"),
    focus: Some(InspectorFocus::Attention),
    example_model_slug: Some("qwen3-30b-a3b"),
    lab_link: Some(("/ai/components/rmsnorm-and-adaln-zero", "RMSNorm 精度实验台")),
  },
  ConfigFieldDoc {
    key: "sliding_window",
    aliases: &[],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::AttentionKv,
    value_type: "int | null",
    summary_zh: "滑动窗口局部注意力（SWA）半径 W（如 1024 或 4096）。局部层的 KV Cache 封顶在 W 个 Token，不再随上下文长度 S 线性增长。",
    summary_en: "Sliding Window Attention (SWA) span W. Caps local-layer KV cache at W tokens regardless of total sequence length S.",
    formula_impact: Some("局部层 KV 显存封顶于 min(S, W) × 2 × H_kv × d_h"),
    focus: Some(InspectorFocus::Attention),
    example_model_slug: Some("gemma-3-27b"),
    lab_link: Some(("/ai/components/attention-mla-and-flash-online-softmax", "Attention 实验台")),
  },
  ConfigFieldDoc {
    key: "sliding_window_pattern",
    aliases: &[],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::AttentionKv,
    value_type: "int",
    summary_zh: "局部 SWA 与全局全注意力的交替周期（如 Gemma-3 为 6，即 5 层局部 SWA + 1 层全局 Full Attention），兼顾长文召回与极低 KV 显存。",
    summary_en: "Interleaving ratio of local SWA to global full attention (e.g. 6 in Gemma-3 = 5 local + 1 global), balancing recall with bounded KV memory.",
    formula_impact: Some("仅 1/P 的层随上下文 S 线性增长，其余 (P-1)/P 层 KV 封顶于 W"),
    focus: Some(InspectorFocus::Attention),
    example_model_slug: Some("gemma-3-27b"),
    lab_link: None,
  },
  ConfigFieldDoc {
    key: "layer_types",
    aliases: &[],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::AttentionKv,
    value_type: "string[]",
    summary_zh: "逐层异构序列混合机制声明数组（如 `linear_attention` / `full_attention` / `sliding_attention`），用于 Qwen3-Next、Gemma 等混合层栈。",
    summary_en: "Per-layer sequence mixer schedule array (e.g. `linear_attention`, `full_attention`, `sliding_attention`) used in hybrid stacks like Qwen3-Next.",
    formula_impact: Some("线性注意力层推理期为 O(1) 固定状态，不随上下文长度增长"),
    focus: Some(InspectorFocus::Attention),
    example_model_slug: Some("qwen3-next-80b-a3b"),
    lab_link: Some(("/ai/components/lstm-deltanet-and-mamba-ssm", "DeltaNet & Mamba SSM 实验台")),
  },
  ConfigFieldDoc {
    key: "full_attention_interval",
    aliases: &[],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::AttentionKv,
    value_type: "int",
    summary_zh: "混合线性/SSM 模型中每隔多少层插入 1 层全局 Softmax Attention（如 Qwen3-Next 为 4，即 3 层 Gated DeltaNet + 1 层 Full GQA）。",
    summary_en: "Stride at which a global Softmax Attention layer is interleaved among linear/SSM layers (e.g. 4 = 3 DeltaNet + 1 Full GQA in Qwen3-Next).",
    formula_impact: Some("KV Cache 仅占传统纯 Transformer 的 1 / interval（如 25%）"),
    focus: Some(InspectorFocus::Attention),
    example_model_slug: Some("qwen3-next-80b-a3b"),
    lab_link: Some(("/ai/components/lstm-deltanet-and-mamba-ssm", "DeltaNet & Mamba SSM 实验台")),
  },
  ConfigFieldDoc {
    key: "linear_num_key_heads",
    aliases: &["linear_num_value_heads"],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::AttentionKv,
    value_type: "int",
    summary_zh: "Gated DeltaNet / 线性注意力层的并行状态头数，决定恒定 O(1) 矩阵循环状态的总槽位数。",
    summary_en: "Number of recurrent heads in Gated DeltaNet / linear attention layers, determining total O(1) recurrent state slots.",
    formula_impact: Some("每层固定状态显存 = H_lin × d_k × d_v × sizeof(dtype)（与长度 S 无关）"),
    focus: Some(InspectorFocus::Attention),
    example_model_slug: Some("qwen3-next-80b-a3b"),
    lab_link: Some(("/ai/components/lstm-deltanet-and-mamba-ssm", "DeltaNet & Mamba SSM 实验台")),
  },
  ConfigFieldDoc {
    key: "linear_key_head_dim",
    aliases: &["linear_value_head_dim"],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::AttentionKv,
    value_type: "int",
    summary_zh: "线性注意力 / Gated DeltaNet 单头状态矩阵维度（d_k / d_v），控制每个头能压缩记忆的历史键值容量。",
    summary_en: "Per-head state matrix dimension (d_k / d_v) in linear attention / Gated DeltaNet, governing associative recall capacity per head.",
    formula_impact: Some("状态矩阵按 d_k × d_v 平方增长"),
    focus: Some(InspectorFocus::Attention),
    example_model_slug: Some("qwen3-next-80b-a3b"),
    lab_link: Some(("/ai/components/lstm-deltanet-and-mamba-ssm", "DeltaNet & Mamba SSM 实验台")),
  },
  ConfigFieldDoc {
    key: "linear_conv_kernel_dim",
    aliases: &["d_conv"],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::AttentionKv,
    value_type: "int",
    summary_zh: "SSM / DeltaNet 前端局部因果短卷积核宽度（常见 4），负责捕获相邻 Token 的局部 N-gram 特征。",
    summary_en: "Causal short depthwise convolution kernel width (typically 4) preceding SSM / DeltaNet recurrence for local n-gram mixing.",
    formula_impact: Some("推理每层需常驻 (k-1) × d_inner 卷积滑动缓存"),
    focus: Some(InspectorFocus::Attention),
    example_model_slug: Some("qwen3-next-80b-a3b"),
    lab_link: Some(("/ai/components/lstm-deltanet-and-mamba-ssm", "DeltaNet & Mamba SSM 实验台")),
  },
  ConfigFieldDoc {
    key: "no_rope_layers",
    aliases: &["no_rope_layer_interval"],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::AttentionKv,
    value_type: "int[] | int",
    summary_zh: "Llama-4 iRoPE 架构中关闭旋转位置编码（NoPE）的全局注意力层位置标记，使超长上下文外推时减少高频相位衰减。",
    summary_en: "Marks layers disabling RoPE (NoPE) in Llama-4 iRoPE, improving ultra-long context generalization without positional decay.",
    formula_impact: None,
    focus: Some(InspectorFocus::Attention),
    example_model_slug: Some("llama-4-scout"),
    lab_link: Some(("/ai/components/rope-and-yarn-extrapolation", "RoPE & YaRN 外推实验台")),
  },
  ConfigFieldDoc {
    key: "attn_logit_softcapping",
    aliases: &["final_logit_softcapping"],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::AttentionKv,
    value_type: "float | null",
    summary_zh: "Gemma-2/3 专用的 Attention/Final Logits 软截断上限 c：执行 `c · tanh(logits / c)`，防止极端大值导致 Softmax 梯度饱和。",
    summary_en: "Gemma-2/3 soft-capping ceiling c (`c * tanh(logits / c)`), preventing extreme attention or output logits from saturating Softmax.",
    formula_impact: Some("需在 FlashAttention 核函数内部融合 tanh，不能用普通标准核替代"),
    focus: Some(InspectorFocus::Attention),
    example_model_slug: Some("gemma-3-27b"),
    lab_link: None,
  },
  ConfigFieldDoc {
    key: "attention_dropout",
    aliases: &["hidden_dropout_prob"],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::AttentionKv,
    value_type: "float",
    summary_zh: "训练期注意力权重随机丢弃率（现代大规模预训练 LLM 通常设为 `0.0`，推理期恒不生效）。",
    summary_en: "Dropout probability on attention weights during training (typically `0.0` in modern LLM pretraining; always disabled at inference).",
    formula_impact: None,
    focus: Some(InspectorFocus::Attention),
    example_model_slug: Some("llama-3-1-8b"),
    lab_link: None,
  },

  // ── C. FFN & Sparse MoE（前馈网络与稀疏混合专家）──
  ConfigFieldDoc {
    key: "intermediate_size",
    aliases: &["ffn_hidden_size", "n_inner", "decoder_ffn_dim"],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::FfnMoe,
    value_type: "int",
    summary_zh: "稠密 FFN（SwiGLU / GeGLU / MLP）的中间扩展维度 D_ff。在 Dense 模型中，FFN 通常占整网参数量的 65%~75%。",
    summary_en: "Dense FFN intermediate expansion width D_ff. In dense Transformers, FFN weights account for ~65–75% of total parameters.",
    formula_impact: Some("SwiGLU 单层 FFN 参数 = 3 × D × D_ff（gate + up + down 三矩阵）"),
    focus: Some(InspectorFocus::FfnOrMoe),
    example_model_slug: Some("llama-3-3-70b"),
    lab_link: Some(("/ai/components/activations-gelu-silu-swiglu", "SwiGLU 激活实验台")),
  },
  ConfigFieldDoc {
    key: "hidden_act",
    aliases: &["activation_function"],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::FfnMoe,
    value_type: "string",
    summary_zh: "FFN 门控/激活函数类型（`silu` 对应 SwiGLU，`gelu_pytorch_tanh` 对应 Gemma GeGLU，`gelu` 常见于 BERT/DiT）。",
    summary_en: "FFN activation function (`silu` for SwiGLU, `gelu_pytorch_tanh` for Gemma GeGLU, `gelu` for BERT/DiT).",
    formula_impact: None,
    focus: Some(InspectorFocus::FfnOrMoe),
    example_model_slug: Some("qwen3-8b"),
    lab_link: Some(("/ai/components/activations-gelu-silu-swiglu", "GELU / SiLU / SwiGLU 实验台")),
  },
  ConfigFieldDoc {
    key: "n_routed_experts",
    aliases: &["num_experts", "num_local_experts"],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::FfnMoe,
    value_type: "int",
    summary_zh: "每个 MoE 层内可被路由器按需选中的稀疏专家总数 E（如 DeepSeek-V3 为 256，Qwen3-MoE 为 128，Mixtral 为 8）。决定模型总容量与权重显存。",
    summary_en: "Total routed sparse experts E per MoE layer (e.g. 256 in DeepSeek-V3, 128 in Qwen3-MoE, 8 in Mixtral). Drives total parameter capacity & weight VRAM.",
    formula_impact: Some("MoE 单层路由专家总参 = E × 3 × D × D_moe_ff"),
    focus: Some(InspectorFocus::FfnOrMoe),
    example_model_slug: Some("deepseek-v3"),
    lab_link: Some(("/ai/components/moe-routing-quant-and-speculative-decoding", "稀疏 MoE 路由实验台")),
  },
  ConfigFieldDoc {
    key: "num_experts_per_tok",
    aliases: &["num_selected_experts"],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::FfnMoe,
    value_type: "int",
    summary_zh: "每个 Token 每次前向实际激活的 Top-K 路由专家数 K（如 8）。决定单 Token 激活参数量与推理算力（FLOPs）。",
    summary_en: "Top-K routed experts activated per token (e.g. 8). Directly determines active parameter count and per-token FLOPs.",
    formula_impact: Some("单 Token 激活 FFN 参数 = (K + n_shared) × 3 × D × D_moe_ff"),
    focus: Some(InspectorFocus::FfnOrMoe),
    example_model_slug: Some("qwen3-30b-a3b"),
    lab_link: Some(("/ai/components/moe-routing-quant-and-speculative-decoding", "稀疏 MoE 路由实验台")),
  },
  ConfigFieldDoc {
    key: "moe_intermediate_size",
    aliases: &["intermediate_size_mlp"],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::FfnMoe,
    value_type: "int",
    summary_zh: "单个细粒度 MoE 专家内部的 FFN 中间维度（如 DeepSeek-V3 为 2048，Qwen3-30B-A3B 为 768），通常远小于 Dense `intermediate_size`。",
    summary_en: "Per-expert FFN intermediate width in fine-grained MoE (e.g. 2048 in DeepSeek-V3, 768 in Qwen3-30B-A3B), much narrower than dense `intermediate_size`.",
    formula_impact: Some("单专家参数 = 3 × D × moe_intermediate_size"),
    focus: Some(InspectorFocus::FfnOrMoe),
    example_model_slug: Some("qwen3-30b-a3b"),
    lab_link: Some(("/ai/components/moe-routing-quant-and-speculative-decoding", "稀疏 MoE 路由实验台")),
  },
  ConfigFieldDoc {
    key: "n_shared_experts",
    aliases: &["shared_expert_intermediate_size"],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::FfnMoe,
    value_type: "int",
    summary_zh: "常驻激活的共享专家数量或维度（DeepSeek-V3 为 1，Llama-4 为 1）。负责承载通用语言知识，减少路由专家间的冗余重复。",
    summary_en: "Always-active shared expert count or width (1 in DeepSeek-V3 and Llama-4), capturing common knowledge to reduce redundancy across routed experts.",
    formula_impact: Some("每个 Token 恒定激活共享专家，不经过 Top-K 门控稀疏化"),
    focus: Some(InspectorFocus::FfnOrMoe),
    example_model_slug: Some("deepseek-v3"),
    lab_link: Some(("/ai/components/moe-routing-quant-and-speculative-decoding", "稀疏 MoE 路由实验台")),
  },
  ConfigFieldDoc {
    key: "first_k_dense_replace",
    aliases: &[],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::FfnMoe,
    value_type: "int",
    summary_zh: "前 K 层保持稠密 Dense FFN 而不替换为 MoE（如 DeepSeek-V3 前 3 层为 Dense，第 3~60 层为 MoE），稳定早期词法特征提取与负载均衡。",
    summary_en: "Number of initial layers kept as dense FFN before switching to MoE (e.g. first 3 layers in DeepSeek-V3) to stabilize early representation & routing.",
    formula_impact: Some("MoE 层数 = num_hidden_layers - first_k_dense_replace"),
    focus: Some(InspectorFocus::FfnOrMoe),
    example_model_slug: Some("deepseek-v3"),
    lab_link: None,
  },
  ConfigFieldDoc {
    key: "decoder_sparse_step",
    aliases: &["interleave_moe_layer_step"],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::FfnMoe,
    value_type: "int",
    summary_zh: "每隔多少层部署一次 MoE 层（`1` 表示全层均为 MoE，`2` 表示 Dense 与 MoE 交替出现）。",
    summary_en: "Stride of MoE layers in the decoder (`1` = every layer is MoE; `2` = alternating Dense and MoE layers).",
    formula_impact: None,
    focus: Some(InspectorFocus::FfnOrMoe),
    example_model_slug: Some("qwen3-30b-a3b"),
    lab_link: None,
  },
  ConfigFieldDoc {
    key: "n_group",
    aliases: &[],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::FfnMoe,
    value_type: "int",
    summary_zh: "DeepSeek-V3 分组受限路由的专家组总数（如 8 组 × 32 专家 = 256），配合 `topk_group` 限制单个 Token 最多跨多少节点通信。",
    summary_en: "Expert group count for node-limited routing in DeepSeek-V3 (e.g. 8 groups × 32 experts = 256), bounding All-to-All communication.",
    formula_impact: Some("单 Token 最多路由到 topk_group 个 EP 节点，大幅降低跨机通信尾延迟"),
    focus: Some(InspectorFocus::FfnOrMoe),
    example_model_slug: Some("deepseek-v3"),
    lab_link: Some(("/ai/components/moe-routing-quant-and-speculative-decoding", "MoE 负载均衡实验台")),
  },
  ConfigFieldDoc {
    key: "topk_group",
    aliases: &[],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::FfnMoe,
    value_type: "int",
    summary_zh: "分组路由中每个 Token 最多允许选中的专家组上限（DeepSeek-V3 为 4），确保 EP 专家并行跨节点流量可控。",
    summary_en: "Max expert groups selectable per token (4 in DeepSeek-V3), capping cross-node Expert Parallel traffic.",
    formula_impact: None,
    focus: Some(InspectorFocus::FfnOrMoe),
    example_model_slug: Some("deepseek-v3"),
    lab_link: None,
  },
  ConfigFieldDoc {
    key: "routed_scaling_factor",
    aliases: &[],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::FfnMoe,
    value_type: "float",
    summary_zh: "MoE 路由专家加权求和后的输出缩放因子（如 DeepSeek-V3 为 `2.5`），用于平衡路由分支与共享专家分支的幅值比例。",
    summary_en: "Output multiplier applied to summed routed expert outputs (e.g. `2.5` in DeepSeek-V3) to balance routed vs. shared branches.",
    formula_impact: None,
    focus: Some(InspectorFocus::FfnOrMoe),
    example_model_slug: Some("deepseek-v3"),
    lab_link: None,
  },
  ConfigFieldDoc {
    key: "scoring_func",
    aliases: &[],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::FfnMoe,
    value_type: "string",
    summary_zh: "MoE 路由器打分激活函数（`sigmoid` 用于 DeepSeek-V3 无辅助损失路由，`softmax` 用于经典 Mixtral/Qwen-MoE）。",
    summary_en: "MoE router scoring function (`sigmoid` for DeepSeek-V3 aux-loss-free routing; `softmax` for Mixtral/Qwen-MoE).",
    formula_impact: None,
    focus: Some(InspectorFocus::FfnOrMoe),
    example_model_slug: Some("deepseek-v3"),
    lab_link: Some(("/ai/components/moe-routing-quant-and-speculative-decoding", "MoE 路由实验台")),
  },
  ConfigFieldDoc {
    key: "norm_topk_prob",
    aliases: &[],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::FfnMoe,
    value_type: "bool",
    summary_zh: "选出 Top-K 专家后是否对其路由概率重新归一化使之和为 1。实现推理核时若漏掉此步会导致 FFN 激活幅度整体偏小。",
    summary_en: "Whether to renormalize the top-K router weights to sum to 1. Omitting this step silently under-scales MoE FFN activations.",
    formula_impact: None,
    focus: Some(InspectorFocus::FfnOrMoe),
    example_model_slug: Some("qwen3-30b-a3b"),
    lab_link: None,
  },
  ConfigFieldDoc {
    key: "mlp_only_layers",
    aliases: &[],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::FfnMoe,
    value_type: "int[]",
    summary_zh: "显式指定保持普通稠密 MLP 而非 MoE 的层索引列表（Qwen3-MoE 常见字段）。",
    summary_en: "Explicit list of layer indices forced to stay dense MLP instead of sparse MoE (used in Qwen3-MoE).",
    formula_impact: None,
    focus: Some(InspectorFocus::FfnOrMoe),
    example_model_slug: Some("qwen3-30b-a3b"),
    lab_link: None,
  },

  // ── D. Position Encoding & Long Context（位置编码与长上下文外推）──
  ConfigFieldDoc {
    key: "max_position_embeddings",
    aliases: &["n_positions", "seq_length"],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::PositionContext,
    value_type: "int",
    summary_zh: "架构官方支持的最大上下文窗口长度（Token 数，如 32768、131072、163840）。决定 RoPE 预计算频率表长度与默认推理上限。",
    summary_en: "Maximum supported context window in tokens (e.g. 32K, 128K, 160K). Governs RoPE frequency tables and engine context limits.",
    formula_impact: Some("全注意力层 KV Cache 随实际上下文长度线性增长直至此上限"),
    focus: Some(InspectorFocus::Attention),
    example_model_slug: Some("deepseek-v3"),
    lab_link: Some(("/ai/components/rope-and-yarn-extrapolation", "RoPE & YaRN 外推实验台")),
  },
  ConfigFieldDoc {
    key: "rope_theta",
    aliases: &["rope_local_base_freq"],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::PositionContext,
    value_type: "float",
    summary_zh: "RoPE 旋转位置编码基频 θ（如早期 10000，长上下文模型大幅调高至 500000、1000000 甚至 10000000），决定低频维度的波长跨度。",
    summary_en: "RoPE base frequency θ (raised from 10k to 500k–10M in long-context models) controlling low-frequency wavelength span.",
    formula_impact: Some("θ_i = rope_theta^(-2i / d_rope)；基频越高长距离相对衰减越缓"),
    focus: Some(InspectorFocus::Attention),
    example_model_slug: Some("llama-3-1-8b"),
    lab_link: Some(("/ai/components/rope-and-yarn-extrapolation", "RoPE & YaRN 外推实验台")),
  },
  ConfigFieldDoc {
    key: "rope_scaling.type",
    aliases: &["rope_scaling.rope_type"],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::PositionContext,
    value_type: "string",
    summary_zh: "长上下文外推算法类型（如 `yarn`、`llama3`、`linear`、`dynamic`、`longrope`、`mrope`），决定如何在超出原始预训练长度后重缩放旋转频率。",
    summary_en: "Context-extension scaling algorithm (`yarn`, `llama3`, `linear`, `dynamic`, `mrope`) governing frequency rescaling beyond base pretraining length.",
    formula_impact: None,
    focus: Some(InspectorFocus::Attention),
    example_model_slug: Some("deepseek-v3"),
    lab_link: Some(("/ai/components/rope-and-yarn-extrapolation", "RoPE & YaRN 外推实验台")),
  },
  ConfigFieldDoc {
    key: "rope_scaling.factor",
    aliases: &[],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::PositionContext,
    value_type: "float",
    summary_zh: "上下文长度扩展倍率 s（如从 4K 扩至 160K 时 `factor = 40.0`），与 `original_max_position_embeddings` 共同决定外推范围。",
    summary_en: "Context extension multiplier s (e.g. `40.0` expanding 4K to 160K), paired with `original_max_position_embeddings`.",
    formula_impact: Some("扩展后有效窗口 ≈ original_max_position_embeddings × factor"),
    focus: Some(InspectorFocus::Attention),
    example_model_slug: Some("deepseek-v3"),
    lab_link: Some(("/ai/components/rope-and-yarn-extrapolation", "RoPE & YaRN 外推实验台")),
  },
  ConfigFieldDoc {
    key: "rope_scaling.original_max_position_embeddings",
    aliases: &[],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::PositionContext,
    value_type: "int",
    summary_zh: "做 YaRN / Llama3 外推前的原始预训练上下文长度（如 4096 或 8192），用于计算高频/低频过渡边界。",
    summary_en: "Original pretraining context length before YaRN/Llama3 scaling (e.g. 4096 or 8192), anchoring frequency ramp boundaries.",
    formula_impact: None,
    focus: Some(InspectorFocus::Attention),
    example_model_slug: Some("deepseek-v3"),
    lab_link: Some(("/ai/components/rope-and-yarn-extrapolation", "RoPE & YaRN 外推实验台")),
  },
  ConfigFieldDoc {
    key: "rope_scaling.beta_fast",
    aliases: &["rope_scaling.beta_slow"],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::PositionContext,
    value_type: "float",
    summary_zh: "YaRN 频段分段插值的快/慢波长阈值（典型值 `beta_fast=32, beta_slow=1`）：高频保留原样保局部精度，低频线性插值保长程外推。",
    summary_en: "YaRN wavelength ramp thresholds (`beta_fast=32, beta_slow=1`): preserves high-frequency local resolution while interpolating low frequencies.",
    formula_impact: None,
    focus: Some(InspectorFocus::Attention),
    example_model_slug: Some("deepseek-v3"),
    lab_link: Some(("/ai/components/rope-and-yarn-extrapolation", "RoPE & YaRN 外推实验台")),
  },
  ConfigFieldDoc {
    key: "rope_scaling.mscale",
    aliases: &["rope_scaling.mscale_all_dim"],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::PositionContext,
    value_type: "float",
    summary_zh: "YaRN 注意力温度补偿因子，补偿超长上下文下 QK 点积熵增，直接乘入 Attention Softmax 缩放系数。",
    summary_en: "YaRN attention temperature compensation factor offsetting QK entropy growth at long context lengths.",
    formula_impact: Some("Softmax 缩放从 1/√d 修正为 mscale² / √d"),
    focus: Some(InspectorFocus::Attention),
    example_model_slug: Some("deepseek-v3"),
    lab_link: Some(("/ai/components/rope-and-yarn-extrapolation", "RoPE & YaRN 外推实验台")),
  },
  ConfigFieldDoc {
    key: "rope_scaling.mrope_section",
    aliases: &[],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::PositionContext,
    value_type: "int[]",
    summary_zh: "Qwen2.5-VL 多模态 M-RoPE 的 3D 维度配额（时间 T / 高度 H / 宽度 W，如 `[16, 24, 24]`），让文本、图像与视频共享统一位置编码空间。",
    summary_en: "Qwen2.5-VL Multimodal M-RoPE dimension split across `[T, H, W]` (e.g. `[16, 24, 24]`), unifying text, image, and video coordinates.",
    formula_impact: Some("sum(mrope_section) × 2 == head_dim"),
    focus: Some(InspectorFocus::Vision),
    example_model_slug: Some("qwen2-5-vl-72b"),
    lab_link: Some(("/ai/components/rope-and-yarn-extrapolation", "多模态 RoPE 实验台")),
  },
  ConfigFieldDoc {
    key: "partial_rotary_factor",
    aliases: &[],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::PositionContext,
    value_type: "float",
    summary_zh: "仅对每个注意力头前 `partial_rotary_factor` 比例的维度施加 RoPE（如 GLM-4 为 `0.5`，Qwen3-Next 为 `0.25`），其余维度保持内容直通。",
    summary_en: "Fraction of `head_dim` rotated by RoPE (e.g. `0.5` in GLM-4, `0.25` in Qwen3-Next); remaining channels pass through unrotated.",
    formula_impact: Some("旋转维度 d_rope = floor(head_dim × partial_rotary_factor)"),
    focus: Some(InspectorFocus::Attention),
    example_model_slug: Some("glm-4-9b"),
    lab_link: Some(("/ai/components/rope-and-yarn-extrapolation", "RoPE 实验台")),
  },

  // ── E. Quantization & Storage Precision（量化精度与权重存储）──
  ConfigFieldDoc {
    key: "torch_dtype",
    aliases: &["dtype"],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::QuantPrecision,
    value_type: "string",
    summary_zh: "权重默认浮点存储与计算精度（`bfloat16`、`float16`、`float32`）。BF16 指数位与 FP32 相同不易溢出，是现代大模型首选。",
    summary_en: "Default weight storage/compute dtype (`bfloat16`, `float16`, `float32`). BF16 preserves FP32 dynamic range and is standard for modern LLMs.",
    formula_impact: Some("BF16/FP16 权重占用 ≈ 2 Bytes/参数；FP32 ≈ 4 Bytes/参数"),
    focus: None,
    example_model_slug: Some("llama-3-3-70b"),
    lab_link: Some(("/ai/math/ieee754-bf16-fp8-numerical-math", "IEEE-754 & BF16/FP8 数值证明")),
  },
  ConfigFieldDoc {
    key: "quantization_config.quant_method",
    aliases: &[],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::QuantPrecision,
    value_type: "string",
    summary_zh: "权重在线量化方案标识（`fp8`、`awq`、`gptq`、`bitsandbytes`、`compressed-tensors`）。推理引擎据此切换反量化或原生低精度 GEMM 核。",
    summary_en: "Quantization scheme (`fp8`, `awq`, `gptq`, `bitsandbytes`, `compressed-tensors`) selecting the matching dequant/GEMM kernels.",
    formula_impact: Some("FP8 ≈ 1.03 B/参（省 50% 显存）；INT4 ≈ 0.56 B/参（省 72% 显存）"),
    focus: Some(InspectorFocus::FfnOrMoe),
    example_model_slug: Some("deepseek-v3"),
    lab_link: Some(("/ai/math/ieee754-bf16-fp8-numerical-math", "BF16/FP8 数值精度证明")),
  },
  ConfigFieldDoc {
    key: "quantization_config.fmt",
    aliases: &["quantization_config.activation_scheme"],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::QuantPrecision,
    value_type: "string",
    summary_zh: "FP8 格式（如 `e4m3`）与激活缩放方式（`dynamic` 在线逐块缩放 vs `static` 离线校准），DeepSeek-V3 原生采用 `e4m3` + `dynamic`。",
    summary_en: "FP8 format (`e4m3`) and activation scaling mode (`dynamic` vs `static`). DeepSeek-V3 ships native `e4m3` with dynamic scaling.",
    formula_impact: Some("E4M3 动态范围 ±448，依赖细粒度块缩放因子避免离群值截断"),
    focus: Some(InspectorFocus::FfnOrMoe),
    example_model_slug: Some("deepseek-v3"),
    lab_link: Some(("/ai/math/ieee754-bf16-fp8-numerical-math", "FP8 E4M3 数值解剖")),
  },
  ConfigFieldDoc {
    key: "quantization_config.weight_block_size",
    aliases: &["quantization_config.group_size"],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::QuantPrecision,
    value_type: "int[] | int",
    summary_zh: "分块/分组量化粒度（如 DeepSeek-V3 的 `[128, 128]` 二维 tile 块，或 AWQ/GPTQ 的 `group_size=128`），粒度越细精度越高但 scale 元数据开销略增。",
    summary_en: "Block/group quantization granularity (e.g. `[128, 128]` 2D tiles in DeepSeek-V3 or `group_size=128` in AWQ/GPTQ).",
    formula_impact: Some("128×128 FP8 仅增加 4/(128×128) ≈ 0.02% 额外 FP32 scale 存储"),
    focus: Some(InspectorFocus::FfnOrMoe),
    example_model_slug: Some("deepseek-v3"),
    lab_link: Some(("/ai/math/ieee754-bf16-fp8-numerical-math", "FP8 块量化数值分析")),
  },
  ConfigFieldDoc {
    key: "quantization_config.bits",
    aliases: &[],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::QuantPrecision,
    value_type: "int",
    summary_zh: "整数量化位宽（典型值 `4` 或 `8`），直接决定权重的理论压缩倍率。",
    summary_en: "Integer quantization bit-width (typically `4` or `8`), setting the theoretical weight compression factor.",
    formula_impact: Some("理论权重体积 ≈ total_params × (bits / 8) + scale/zero_point 开销"),
    focus: Some(InspectorFocus::FfnOrMoe),
    example_model_slug: Some("qwen3-8b"),
    lab_link: None,
  },
  ConfigFieldDoc {
    key: "metadata.total_size",
    aliases: &[],
    file: ConfigFileKind::SafetensorsIndex,
    group: FieldGroup::QuantPrecision,
    value_type: "int",
    summary_zh: "`model.safetensors.index.json` 中记录的全部权重分片精确磁盘总字节数（Bytes），比纯理论参数量更能反映真实显存下限。",
    summary_en: "Exact aggregate weight byte size recorded in `model.safetensors.index.json`, reflecting true disk & weight VRAM floor.",
    formula_impact: Some("最小权重显存 ≥ total_size / 1024³ GiB"),
    focus: None,
    example_model_slug: Some("deepseek-v3"),
    lab_link: None,
  },
  ConfigFieldDoc {
    key: "weight_map",
    aliases: &[],
    file: ConfigFileKind::SafetensorsIndex,
    group: FieldGroup::QuantPrecision,
    value_type: "object",
    summary_zh: "`model.safetensors.index.json` 中从每个张量全名到所属 `.safetensors` 分片文件名的完整映射表，用于按需加载与权重树核验。",
    summary_en: "Complete mapping from every tensor name to its `.safetensors` shard filename inside `model.safetensors.index.json`.",
    formula_impact: None,
    focus: None,
    example_model_slug: Some("deepseek-v3"),
    lab_link: None,
  },

  // ── F. Multimodal, Diffusion & Pooling（多模态、扩散与池化）──
  ConfigFieldDoc {
    key: "vision_config.hidden_size",
    aliases: &["vision_config.embed_dim"],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::MultimodalPipeline,
    value_type: "int",
    summary_zh: "VLM 视觉编码塔（ViT / SigLIP）内部隐藏维度，经由视觉投影器（Projector/Merger）映射后再与语言主干 `hidden_size` 对齐。",
    summary_en: "Vision tower (ViT/SigLIP) hidden width before being projected to match the language backbone's `hidden_size`.",
    formula_impact: None,
    focus: Some(InspectorFocus::Vision),
    example_model_slug: Some("qwen2-5-vl-72b"),
    lab_link: Some(("/ai/components/conv-and-depthwise-separable", "视觉 Patch 与卷积实验台")),
  },
  ConfigFieldDoc {
    key: "vision_config.depth",
    aliases: &["vision_config.num_hidden_layers"],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::MultimodalPipeline,
    value_type: "int",
    summary_zh: "视觉编码器 Transformer 层数（如 SigLIP 为 27 层，Qwen2.5-VL 视觉塔为 32 层）。",
    summary_en: "Number of Transformer blocks inside the vision encoder tower (e.g. 27 in SigLIP, 32 in Qwen2.5-VL).",
    formula_impact: None,
    focus: Some(InspectorFocus::Vision),
    example_model_slug: Some("qwen2-5-vl-72b"),
    lab_link: None,
  },
  ConfigFieldDoc {
    key: "vision_config.patch_size",
    aliases: &["patch_size"],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::MultimodalPipeline,
    value_type: "int",
    summary_zh: "图像/潜空间分块边长 P（如 ViT 常见 `14`/`16`，DiT 潜空间常见 `2`）。Patch 越小空间细节越丰富，但序列长度按 1/P² 平方暴涨。",
    summary_en: "Spatial patch edge length P (e.g. `14`/`16` in ViT; `2` in DiT latents). Smaller P increases token count quadratically as 1/P².",
    formula_impact: Some("视觉 Token 数 = (H / P) × (W / P) / merge_size²"),
    focus: Some(InspectorFocus::Vision),
    example_model_slug: Some("qwen2-5-vl-72b"),
    lab_link: Some(("/ai/components/conv-and-depthwise-separable", "Patch 卷积与下采样实验台")),
  },
  ConfigFieldDoc {
    key: "vision_config.spatial_merge_size",
    aliases: &["merge_size"],
    file: ConfigFileKind::PreprocessorConfig,
    group: FieldGroup::MultimodalPipeline,
    value_type: "int",
    summary_zh: "相邻 `m × m` 个视觉 Patch 合并为 1 个语言视觉 Token 的压缩步长（常见 `2`，可将视觉序列长度压缩 4×）。",
    summary_en: "Merges adjacent `m × m` visual patches into 1 LLM visual token (typically `2`, shrinking visual sequence length by 4×).",
    formula_impact: Some("送入 LLM 的视觉 Token 长度缩短 m² 倍，Prefill 算力大幅下降"),
    focus: Some(InspectorFocus::Vision),
    example_model_slug: Some("qwen2-5-vl-72b"),
    lab_link: None,
  },
  ConfigFieldDoc {
    key: "vision_config.image_size",
    aliases: &["image_size"],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::MultimodalPipeline,
    value_type: "int",
    summary_zh: "固定分辨率视觉塔的标准输入边长（如 384、448、896）；支持原生动态分辨率的 VLM 则改由 `min_pixels`/`max_pixels` 控制。",
    summary_en: "Fixed input image resolution (e.g. 384, 448, 896) for static-grid vision towers; dynamic VLMs rely on `min_pixels`/`max_pixels` instead.",
    formula_impact: Some("单图固定 Patch 数 = (image_size / patch_size)²"),
    focus: Some(InspectorFocus::Vision),
    example_model_slug: Some("gemma-3-27b"),
    lab_link: None,
  },
  ConfigFieldDoc {
    key: "min_pixels",
    aliases: &["max_pixels"],
    file: ConfigFileKind::PreprocessorConfig,
    group: FieldGroup::MultimodalPipeline,
    value_type: "int",
    summary_zh: "动态分辨率 VLM 允许的最小/最大单图像素面积，直接决定每张输入图片会被切成多少个视觉 Token。",
    summary_en: "Min/max pixel area bounds for dynamic-resolution VLMs, directly controlling visual token budget per image.",
    formula_impact: Some("最大视觉 Token ≈ max_pixels / (patch_size × merge_size)²"),
    focus: Some(InspectorFocus::Vision),
    example_model_slug: Some("qwen2-5-vl-72b"),
    lab_link: None,
  },
  ConfigFieldDoc {
    key: "num_mel_bins",
    aliases: &["audio_config.num_mel_bins", "feature_size"],
    file: ConfigFileKind::PreprocessorConfig,
    group: FieldGroup::MultimodalPipeline,
    value_type: "int",
    summary_zh: "音频前端短时傅里叶变换（STFT）后的梅尔滤波器组频带数（如 Whisper-v3 与 Qwen-Audio 为 128，旧版 Whisper 为 80）。",
    summary_en: "Number of Mel filterbank frequency bins after STFT (128 in Whisper-v3 & Qwen-Audio; 80 in earlier Whisper).",
    formula_impact: Some("声学输入特征形状 = [Batch, num_mel_bins, num_frames]"),
    focus: Some(InspectorFocus::Audio),
    example_model_slug: Some("whisper-large-v3"),
    lab_link: Some(("/ai/components/audio-stft-conformer-ctc-and-rvq", "STFT & 梅尔语谱实验台")),
  },
  ConfigFieldDoc {
    key: "sampling_rate",
    aliases: &["sample_rate"],
    file: ConfigFileKind::PreprocessorConfig,
    group: FieldGroup::MultimodalPipeline,
    value_type: "int",
    summary_zh: "语音模型要求的原始波形采样率（Hz，ASR 常见 `16000`，高保真 TTS/声码器常见 `24000` 或 `44100`）。",
    summary_en: "Required input/output waveform sampling rate in Hz (`16000` for ASR; `24000`/`44100` for high-fidelity TTS vocoders).",
    formula_impact: Some("奈奎斯特上限频率 = sampling_rate / 2"),
    focus: Some(InspectorFocus::Audio),
    example_model_slug: Some("whisper-large-v3"),
    lab_link: Some(("/ai/math/fourier-convolution-aliasing", "傅里叶与奈奎斯特混叠证明")),
  },
  ConfigFieldDoc {
    key: "max_source_positions",
    aliases: &["audio_config.max_source_positions"],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::MultimodalPipeline,
    value_type: "int",
    summary_zh: "音频编码器单次窗口最大声学帧序列长度（如 Whisper 30 秒音频经 2× 下采样后恒为 `1500` 帧）。",
    summary_en: "Max acoustic frame length per encoder window (e.g. `1500` frames for Whisper's 30-second window after 2× conv stride).",
    formula_impact: Some("Cross-Attention 缓存固定承载 max_source_positions 帧编码特征"),
    focus: Some(InspectorFocus::Audio),
    example_model_slug: Some("whisper-large-v3"),
    lab_link: Some(("/ai/components/audio-stft-conformer-ctc-and-rvq", "Audio & Conformer 实验台")),
  },
  ConfigFieldDoc {
    key: "encoder_layers",
    aliases: &["audio_config.encoder_layers"],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::MultimodalPipeline,
    value_type: "int",
    summary_zh: "Seq2Seq 架构（如 Whisper / T5）中编码器侧的 Transformer 层数，与 `decoder_layers` 共同组成编码-解码双塔。",
    summary_en: "Number of Transformer layers in the encoder stack of Seq2Seq models (e.g. Whisper, T5).",
    formula_impact: None,
    focus: Some(InspectorFocus::Audio),
    example_model_slug: Some("whisper-large-v3"),
    lab_link: None,
  },
  ConfigFieldDoc {
    key: "_class_name",
    aliases: &[],
    file: ConfigFileKind::DiffusionPipeline,
    group: FieldGroup::MultimodalPipeline,
    value_type: "string",
    summary_zh: "Diffusers 管线或子组件的精确类名（如 `FluxPipeline`、`FluxTransformer2DModel`、`AutoencoderKL`），决定加载与调用契约。",
    summary_en: "Exact Diffusers pipeline or submodule class name (e.g. `FluxPipeline`, `FluxTransformer2DModel`, `AutoencoderKL`).",
    formula_impact: None,
    focus: None,
    example_model_slug: Some("flux-1-dev"),
    lab_link: None,
  },
  ConfigFieldDoc {
    key: "num_single_layers",
    aliases: &[],
    file: ConfigFileKind::DiffusionPipeline,
    group: FieldGroup::MultimodalPipeline,
    value_type: "int",
    summary_zh: "FLUX MMDiT 后端单流融合 Transformer 块数（如 38 层），与前端 `num_layers`（19 层双流）拼接构成完整去噪主干。",
    summary_en: "Number of single-stream joint Transformer blocks in FLUX MMDiT (e.g. 38), following `num_layers` (19 dual-stream blocks).",
    formula_impact: Some("FLUX 去噪总深度 = num_layers (双流) + num_single_layers (单流)"),
    focus: Some(InspectorFocus::Attention),
    example_model_slug: Some("flux-1-dev"),
    lab_link: Some(("/ai/math/continuity-equation-cfm-theorem", "条件流匹配 CFM 核心定理")),
  },
  ConfigFieldDoc {
    key: "joint_attention_dim",
    aliases: &["caption_projection_dim", "cross_attention_dim"],
    file: ConfigFileKind::DiffusionPipeline,
    group: FieldGroup::MultimodalPipeline,
    value_type: "int",
    summary_zh: "扩散去噪主干接收文本条件编码（如 T5-XXL 4096 维）的投影输入维度，决定文本-视觉联合注意力接口宽度。",
    summary_en: "Input dimension of text conditioning embeddings (e.g. 4096 from T5-XXL) projected into MMDiT/UNet attention.",
    formula_impact: None,
    focus: Some(InspectorFocus::Embedding),
    example_model_slug: Some("flux-1-dev"),
    lab_link: None,
  },
  ConfigFieldDoc {
    key: "in_channels",
    aliases: &["out_channels"],
    file: ConfigFileKind::DiffusionPipeline,
    group: FieldGroup::MultimodalPipeline,
    value_type: "int",
    summary_zh: "扩散主干输入/输出潜空间张量通道数（如 FLUX pack 2×2 后 `16×4=64`，SD3.5/Wan 为 `16`，SDXL 为 `4`）。",
    summary_en: "Input/output latent channel count for the denoiser (e.g. 64 packed in FLUX, 16 in SD3.5/Wan, 4 in SDXL).",
    formula_impact: Some("潜空间张量维度直接决定 DiT 输入投影 [in_channels → hidden_size]"),
    focus: Some(InspectorFocus::LmHead),
    example_model_slug: Some("flux-1-dev"),
    lab_link: None,
  },
  ConfigFieldDoc {
    key: "attention_head_dim",
    aliases: &[],
    file: ConfigFileKind::DiffusionPipeline,
    group: FieldGroup::MultimodalPipeline,
    value_type: "int",
    summary_zh: "Diffusers DiT / MMDiT 中单个注意力头的维度，配合 `num_attention_heads` 共同算出主干隐藏宽度 `D = H × d_h`。",
    summary_en: "Per-head dimension in Diffusers DiT/MMDiT; backbone hidden size is computed as `num_attention_heads × attention_head_dim`.",
    formula_impact: Some("DiT hidden_size = num_attention_heads × attention_head_dim"),
    focus: Some(InspectorFocus::Attention),
    example_model_slug: Some("flux-1-dev"),
    lab_link: None,
  },
  ConfigFieldDoc {
    key: "guidance_embeds",
    aliases: &[],
    file: ConfigFileKind::DiffusionPipeline,
    group: FieldGroup::MultimodalPipeline,
    value_type: "bool",
    summary_zh: "去噪器是否内置蒸馏 CFG 引导强度嵌入（FLUX.1-dev 为 `true`，无需跑 2× batch 的无条件分支；FLUX.1-schnell 为 `false`）。",
    summary_en: "Whether classifier-free guidance scale is distilled as an input embedding (`true` in FLUX.1-dev, eliminating 2× CFG batch overhead).",
    formula_impact: Some("true 时推理单步去噪算力减半（Batch 1 替代无条件+有条件 Batch 2）"),
    focus: Some(InspectorFocus::NormAndResidual),
    example_model_slug: Some("flux-1-dev"),
    lab_link: Some(("/ai/components/rmsnorm-and-adaln-zero", "AdaLN-Zero 调制实验台")),
  },
  ConfigFieldDoc {
    key: "latent_channels",
    aliases: &["z_dim"],
    file: ConfigFileKind::DiffusionPipeline,
    group: FieldGroup::MultimodalPipeline,
    value_type: "int",
    summary_zh: "VAE 潜空间通道数（如 FLUX/SD3.5/Wan2.1 为 `16`，大幅高于早期 SD1.5/SDXL 的 `4`，显著提升文字与细节保真度）。",
    summary_en: "VAE latent channel depth (`16` in FLUX/SD3.5/Wan2.1 vs `4` in SDXL), dramatically improving text rendering and fine texture fidelity.",
    formula_impact: Some("潜空间体积 = [B, C_lat, T/f_t, H/f_s, W/f_s]"),
    focus: Some(InspectorFocus::LmHead),
    example_model_slug: Some("wan2-1-t2v-14b"),
    lab_link: None,
  },
  ConfigFieldDoc {
    key: "scaling_factor",
    aliases: &["shift_factor"],
    file: ConfigFileKind::DiffusionPipeline,
    group: FieldGroup::MultimodalPipeline,
    value_type: "float",
    summary_zh: "VAE 潜空间与扩散模型之间的方差缩放与偏移常数。解码前必须执行 `z = z / scaling_factor + shift_factor`，否则重建图像严重失真。",
    summary_en: "VAE latent normalization constants. Decoding must apply `z = z / scaling_factor + shift_factor` before passing latents into the VAE decoder.",
    formula_impact: None,
    focus: Some(InspectorFocus::LmHead),
    example_model_slug: Some("flux-1-dev"),
    lab_link: None,
  },
  ConfigFieldDoc {
    key: "num_train_timesteps",
    aliases: &[],
    file: ConfigFileKind::DiffusionPipeline,
    group: FieldGroup::MultimodalPipeline,
    value_type: "int",
    summary_zh: "扩散/流匹配调度器的训练离散时间步网格数（通常 `1000`），推理期按采样步数（如 28~50 步）在其上插值取样。",
    summary_en: "Training timestep grid resolution (typically `1000`); inference samplers draw e.g. 28–50 steps across this horizon.",
    formula_impact: None,
    focus: None,
    example_model_slug: Some("flux-1-dev"),
    lab_link: Some(("/ai/math/continuity-equation-cfm-theorem", "条件流匹配 CFM 定理")),
  },
  ConfigFieldDoc {
    key: "shift",
    aliases: &["use_dynamic_shifting", "flow_shift"],
    file: ConfigFileKind::DiffusionPipeline,
    group: FieldGroup::MultimodalPipeline,
    value_type: "float | bool",
    summary_zh: "高分辨率 Flow Matching 的时间步偏移（Time-Shift）参数。分辨率越高越需要把更多去噪步数分配在高噪声阶段以稳住全局构图。",
    summary_en: "Flow-matching timestep shifting (`shift` / `use_dynamic_shifting`). Allocates more steps to high-noise regimes for high-res images/videos.",
    formula_impact: Some("t' = shift · t / (1 + (shift - 1) · t)"),
    focus: None,
    example_model_slug: Some("sd-3-5-large"),
    lab_link: Some(("/ai/math/continuity-equation-cfm-theorem", "条件流匹配 CFM 定理")),
  },
  ConfigFieldDoc {
    key: "word_embedding_dimension",
    aliases: &[],
    file: ConfigFileKind::EmbeddingModules,
    group: FieldGroup::MultimodalPipeline,
    value_type: "int",
    summary_zh: "`1_Pooling/config.json` 声明的输入隐藏态与默认句向量输出维度（如 BGE-M3 为 `1024`，Qwen3-Embedding-8B 为 `4096`）。",
    summary_en: "Token embedding width and default sentence vector dimension declared in `1_Pooling/config.json` (e.g. `1024` in BGE-M3).",
    formula_impact: Some("向量索引单条 FP32 存储 = dim × 4 Bytes（1024d = 4 KiB/条）"),
    focus: Some(InspectorFocus::LmHead),
    example_model_slug: Some("bge-m3"),
    lab_link: None,
  },
  ConfigFieldDoc {
    key: "pooling_mode_cls_token",
    aliases: &["pooling_mode_mean_tokens", "pooling_mode_lasttoken"],
    file: ConfigFileKind::EmbeddingModules,
    group: FieldGroup::MultimodalPipeline,
    value_type: "bool",
    summary_zh: "句向量池化策略开关：双向编码器多取首位 `[CLS]` 或 `mean` 均值；因果解码器 Embedding（如 Qwen3-Embedding）必须启用 `lasttoken`。",
    summary_en: "Pooling mode switch: bidirectional encoders use `[CLS]` or `mean`; causal decoder embeddings (Qwen3-Embedding) require `lasttoken`.",
    formula_impact: None,
    focus: Some(InspectorFocus::LmHead),
    example_model_slug: Some("bge-m3"),
    lab_link: None,
  },
  ConfigFieldDoc {
    key: "num_labels",
    aliases: &[],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::MultimodalPipeline,
    value_type: "int",
    summary_zh: "分类或交叉编码器重排序（Reranker）输出头的类别/打分数（Reranker 通常为 `1`，直接输出相关性标量打分）。",
    summary_en: "Output head width for sequence classification or cross-encoder rerankers (`1` for scalar relevance scoring).",
    formula_impact: Some("输出头参数 = hidden_size × num_labels"),
    focus: Some(InspectorFocus::LmHead),
    example_model_slug: Some("bge-m3"),
    lab_link: None,
  },

  // ── G. Tokenizer, Generation & Adapters（词表、采样与适配器）──
  ConfigFieldDoc {
    key: "tokenizer_class",
    aliases: &[],
    file: ConfigFileKind::TokenizerConfig,
    group: FieldGroup::TokenizerGeneration,
    value_type: "string",
    summary_zh: "分词器实现类名（如 `LlamaTokenizerFast`、`Qwen2Tokenizer`、`PreTrainedTokenizerFast`），提示底层采用字节级 BPE 还是 SentencePiece Unigram。",
    summary_en: "Tokenizer class name (e.g. `LlamaTokenizerFast`, `Qwen2Tokenizer`) indicating byte-level BPE vs. SentencePiece Unigram.",
    formula_impact: None,
    focus: Some(InspectorFocus::Embedding),
    example_model_slug: Some("qwen3-8b"),
    lab_link: None,
  },
  ConfigFieldDoc {
    key: "model_max_length",
    aliases: &[],
    file: ConfigFileKind::TokenizerConfig,
    group: FieldGroup::TokenizerGeneration,
    value_type: "int",
    summary_zh: "分词器侧默认截断长度上限。工程上经常出现 `1000000000000000019884624838656` 占位大数，判断真实上下文请以 `config.json` 为准。",
    summary_en: "Tokenizer-side truncation default. Frequently set to a huge sentinel value; always rely on `config.json`'s `max_position_embeddings`.",
    formula_impact: None,
    focus: None,
    example_model_slug: Some("qwen3-8b"),
    lab_link: None,
  },
  ConfigFieldDoc {
    key: "chat_template",
    aliases: &[],
    file: ConfigFileKind::TokenizerConfig,
    group: FieldGroup::TokenizerGeneration,
    value_type: "string (Jinja2)",
    summary_zh: "将多轮 `[{role, content}]` 消息与工具定义渲染为模型原始提示词的 Jinja 模板。错用模板会导致模型拒绝遵循指令或工具调用失效。",
    summary_en: "Jinja2 template rendering multi-turn `[{role, content}]` messages and tool schemas into raw prompt text. Wrong templates break instruction following.",
    formula_impact: None,
    focus: None,
    example_model_slug: Some("qwen3-30b-a3b"),
    lab_link: None,
  },
  ConfigFieldDoc {
    key: "add_bos_token",
    aliases: &["add_eos_token"],
    file: ConfigFileKind::TokenizerConfig,
    group: FieldGroup::TokenizerGeneration,
    value_type: "bool",
    summary_zh: "编码文本时是否自动在句首追加 BOS（或在句尾追加 EOS）。应用 `chat_template` 时若模板本身已包含 BOS，双重 BOS 会劣化首词生成质量。",
    summary_en: "Whether encoding automatically prepends BOS (or appends EOS). Beware double-BOS bugs when `chat_template` already emits BOS.",
    formula_impact: None,
    focus: Some(InspectorFocus::Embedding),
    example_model_slug: Some("llama-3-1-8b"),
    lab_link: None,
  },
  ConfigFieldDoc {
    key: "bos_token",
    aliases: &["eos_token", "pad_token", "unk_token"],
    file: ConfigFileKind::TokenizerConfig,
    group: FieldGroup::TokenizerGeneration,
    value_type: "string | object",
    summary_zh: "特殊控制符的文本字面量（如 `<|im_end|>`、`<|eot_id|>`、`</s>`），与 `added_tokens_decoder` 中的数字 ID 对应。",
    summary_en: "String literal of special control tokens (`<|im_end|>`, `<|eot_id|>`, `</s>`) paired with numeric IDs in `added_tokens_decoder`.",
    formula_impact: None,
    focus: Some(InspectorFocus::Embedding),
    example_model_slug: Some("qwen3-8b"),
    lab_link: None,
  },
  ConfigFieldDoc {
    key: "do_sample",
    aliases: &[],
    file: ConfigFileKind::GenerationConfig,
    group: FieldGroup::TokenizerGeneration,
    value_type: "bool",
    summary_zh: "是否启用随机采样解码；设为 `false` 时退化为贪婪解码（Greedy Decoding，此时 `temperature`/`top_p` 不再生效）。",
    summary_en: "Enables stochastic sampling; when `false`, decoding becomes greedy and ignores `temperature`/`top_p`.",
    formula_impact: None,
    focus: Some(InspectorFocus::LmHead),
    example_model_slug: Some("deepseek-r1"),
    lab_link: Some(("/ai/math/forward-reverse-kl-and-k3", "概率分布与 KL 散度推导")),
  },
  ConfigFieldDoc {
    key: "temperature",
    aliases: &[],
    file: ConfigFileKind::GenerationConfig,
    group: FieldGroup::TokenizerGeneration,
    value_type: "float",
    summary_zh: "采样温度 T：在 Softmax 前将 Logits 除以 T。T 越低越保守确定，推理模型（如 R1 / Qwen3 Thinking）通常推荐 `0.6` 以防死循环复读。",
    summary_en: "Sampling temperature T dividing logits before Softmax. Reasoning models (R1, Qwen3) typically recommend `0.6` to prevent greedy repetition loops.",
    formula_impact: Some("p_i = exp(z_i / T) / Σ exp(z_j / T)"),
    focus: Some(InspectorFocus::LmHead),
    example_model_slug: Some("deepseek-r1"),
    lab_link: None,
  },
  ConfigFieldDoc {
    key: "top_p",
    aliases: &[],
    file: ConfigFileKind::GenerationConfig,
    group: FieldGroup::TokenizerGeneration,
    value_type: "float",
    summary_zh: "核采样（Nucleus Sampling）累计概率截断阈值（如 `0.95`），动态保留累计概率达到 p 的最小候选词集合。",
    summary_en: "Nucleus sampling cumulative probability cutoff (e.g. `0.95`), retaining the smallest token set whose probability mass reaches p.",
    formula_impact: None,
    focus: Some(InspectorFocus::LmHead),
    example_model_slug: Some("qwen3-30b-a3b"),
    lab_link: None,
  },
  ConfigFieldDoc {
    key: "top_k",
    aliases: &[],
    file: ConfigFileKind::GenerationConfig,
    group: FieldGroup::TokenizerGeneration,
    value_type: "int",
    summary_zh: "每步解码仅从概率最高的前 K 个候选 Token 中采样（如 Qwen3 默认 `20`），过滤长尾低概率噪声词。",
    summary_en: "Restricts sampling at each decoding step to the top K most probable tokens (e.g. `20` in Qwen3), pruning low-probability tail noise.",
    formula_impact: None,
    focus: Some(InspectorFocus::LmHead),
    example_model_slug: Some("qwen3-30b-a3b"),
    lab_link: None,
  },
  ConfigFieldDoc {
    key: "repetition_penalty",
    aliases: &["presence_penalty"],
    file: ConfigFileKind::GenerationConfig,
    group: FieldGroup::TokenizerGeneration,
    value_type: "float",
    summary_zh: "重复惩罚系数（`>1.0` 抑制已出现 Token 的 Logits），过大（如 >1.2）会破坏代码缩进、JSON 语法和数学公式符号。",
    summary_en: "Penalty discounting logits of already-generated tokens (`>1.0`). Excessive values (>1.2) corrupt code syntax, JSON keys, and math notation.",
    formula_impact: None,
    focus: Some(InspectorFocus::LmHead),
    example_model_slug: Some("qwen2-5-72b"),
    lab_link: None,
  },
  ConfigFieldDoc {
    key: "max_new_tokens",
    aliases: &["max_length"],
    file: ConfigFileKind::GenerationConfig,
    group: FieldGroup::TokenizerGeneration,
    value_type: "int",
    summary_zh: "单次请求默认允许生成的最大新 Token 数。注意区分 `max_new_tokens`（仅计输出）与 `max_length`（输入 + 输出总长）。",
    summary_en: "Default maximum output tokens per generation. Distinguish `max_new_tokens` (output only) from `max_length` (prompt + output).",
    formula_impact: None,
    focus: None,
    example_model_slug: Some("deepseek-r1"),
    lab_link: None,
  },
  ConfigFieldDoc {
    key: "r",
    aliases: &[],
    file: ConfigFileKind::AdapterConfig,
    group: FieldGroup::TokenizerGeneration,
    value_type: "int",
    summary_zh: "LoRA 低秩分解矩阵的内维度 r（常见 8 / 16 / 32 / 64）。每个目标线性层新增 `r × (d_in + d_out)` 个可训练参数。",
    summary_en: "LoRA low-rank bottleneck dimension r (typically 8–64), adding `r × (d_in + d_out)` trainable parameters per targeted projection.",
    formula_impact: Some("单层 LoRA 增量参数 = r × (d_in + d_out)；秩越小越省显存"),
    focus: Some(InspectorFocus::Attention),
    example_model_slug: Some("qwen3-8b"),
    lab_link: Some(("/ai/math/svd-eckart-young-pseudoinverse", "SVD 低秩最优逼近证明")),
  },
  ConfigFieldDoc {
    key: "lora_alpha",
    aliases: &[],
    file: ConfigFileKind::AdapterConfig,
    group: FieldGroup::TokenizerGeneration,
    value_type: "int | float",
    summary_zh: "LoRA 缩放分子系数 α。标准 LoRA 增量缩放比为 `lora_alpha / r`（常设为 `r` 或 `2r`），决定适配增量相对基座权重的步幅。",
    summary_en: "LoRA scaling numerator α. Standard LoRA scales deltas by `lora_alpha / r` (often `r` or `2r`), governing adapter strength relative to base weights.",
    formula_impact: Some("W_eff = W_0 + (lora_alpha / r) · B @ A"),
    focus: Some(InspectorFocus::Attention),
    example_model_slug: Some("qwen3-8b"),
    lab_link: Some(("/ai/math/svd-eckart-young-pseudoinverse", "SVD 低秩分解证明")),
  },
  ConfigFieldDoc {
    key: "target_modules",
    aliases: &[],
    file: ConfigFileKind::AdapterConfig,
    group: FieldGroup::TokenizerGeneration,
    value_type: "string[] | string",
    summary_zh: "注入 LoRA 适配器的基座线性层模块名列表（如 `q_proj`/`v_proj` 或全线性层 `all-linear`），覆盖 FFN 层通常比仅改 Q/V 拟合能力更强。",
    summary_en: "Base linear module names injected with LoRA (e.g. `q_proj`, `v_proj`, or `all-linear`). Including FFN projections generally improves adaptation.",
    formula_impact: None,
    focus: Some(InspectorFocus::FfnOrMoe),
    example_model_slug: Some("qwen3-8b"),
    lab_link: None,
  },
  ConfigFieldDoc {
    key: "modules_to_save",
    aliases: &[],
    file: ConfigFileKind::AdapterConfig,
    group: FieldGroup::TokenizerGeneration,
    value_type: "string[] | null",
    summary_zh: "微调时跳过低秩分解、直接全量训练并保存的模块（如扩词表时的 `embed_tokens`、`lm_head` 或分类头）。",
    summary_en: "Modules trained and saved at full rank instead of LoRA (essential for `embed_tokens`/`lm_head` when new special tokens are added).",
    formula_impact: None,
    focus: Some(InspectorFocus::LmHead),
    example_model_slug: Some("qwen3-8b"),
    lab_link: None,
  },
  ConfigFieldDoc {
    key: "use_rslora",
    aliases: &["use_dora"],
    file: ConfigFileKind::AdapterConfig,
    group: FieldGroup::TokenizerGeneration,
    value_type: "bool",
    summary_zh: "高秩稳定 LoRA（`use_rslora` 将缩放因子改为 `α / √r` 防大秩发散）或权重分解 DoRA（分离幅值与方向更新）。合并权重时必须识别该标志。",
    summary_en: "Rank-stabilized LoRA (`α / √r` scaling for large ranks) or DoRA (magnitude/direction decomposition). Merge scripts must honor these flags.",
    formula_impact: Some("use_rslora=true 时缩放因子由 α/r 变为 α/√r"),
    focus: Some(InspectorFocus::Attention),
    example_model_slug: Some("qwen3-8b"),
    lab_link: None,
  },
  ConfigFieldDoc {
    key: "base_model_name_or_path",
    aliases: &[],
    file: ConfigFileKind::AdapterConfig,
    group: FieldGroup::TokenizerGeneration,
    value_type: "string",
    summary_zh: "该适配器所绑定的精确基座模型仓库 ID（如 `Qwen/Qwen3-8B`），推理与合并必须加载同一版本的基座权重。",
    summary_en: "Exact base model repo ID this adapter attaches to (e.g. `Qwen/Qwen3-8B`); merging onto a mismatched checkpoint produces gibberish.",
    formula_impact: None,
    focus: None,
    example_model_slug: Some("qwen3-8b"),
    lab_link: None,
  },
  ConfigFieldDoc {
    key: "peft_type",
    aliases: &["task_type"],
    file: ConfigFileKind::AdapterConfig,
    group: FieldGroup::TokenizerGeneration,
    value_type: "string",
    summary_zh: "PEFT 适配器算法类型（`LORA`、`IA3`、`PREFIX_TUNING`）及任务类型（`CAUSAL_LM`、`SEQ_CLS`）。",
    summary_en: "PEFT adapter method (`LORA`, `IA3`, `PREFIX_TUNING`) and task wrapper (`CAUSAL_LM`, `SEQ_CLS`).",
    formula_impact: None,
    focus: None,
    example_model_slug: Some("qwen3-8b"),
    lab_link: None,
  },

  // ── H. 补充高频实用字段（覆盖现代旗舰模型完整细节）──
  ConfigFieldDoc {
    key: "pretraining_tp",
    aliases: &[],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::Backbone,
    value_type: "int",
    summary_zh: "预训练阶段使用的张量并行切分数（Llama 系常见；`>1` 时为复现原始数值精度会拆分线性计算顺序，推理期通常保持 `1`）。",
    summary_en: "Tensor parallelism rank used during pretraining (Llama family); `>1` emulates split matmul order for parity, normally `1` at inference.",
    formula_impact: None,
    focus: None,
    example_model_slug: Some("llama-3-1-8b"),
    lab_link: None,
  },
  ConfigFieldDoc {
    key: "mlp_bias",
    aliases: &[],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::FfnMoe,
    value_type: "bool",
    summary_zh: "FFN 线性投影层是否包含偏置项（Llama/Qwen/DeepSeek 普遍为 `false` 以减少访存并提升融合算子效率）。",
    summary_en: "Whether FFN projections include additive bias (`false` across Llama/Qwen/DeepSeek to reduce memory traffic in fused kernels).",
    formula_impact: None,
    focus: Some(InspectorFocus::FfnOrMoe),
    example_model_slug: Some("llama-3-1-8b"),
    lab_link: None,
  },
  ConfigFieldDoc {
    key: "query_pre_attn_scalar",
    aliases: &[],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::AttentionKv,
    value_type: "int | float",
    summary_zh: "Gemma-2/3 显式指定的注意力缩放分母基底 s（计算 `Q / sqrt(s)`，常与 `head_dim` 不同，实现自定义核时切勿硬编码 `1/sqrt(head_dim)`）。",
    summary_en: "Explicit Gemma-2/3 attention scale denominator s (`Q / sqrt(s)`), which can differ from `head_dim`; never hardcode `1/sqrt(head_dim)`.",
    formula_impact: Some("Attention 缩放系数 = 1 / sqrt(query_pre_attn_scalar)"),
    focus: Some(InspectorFocus::Attention),
    example_model_slug: Some("gemma-3-27b"),
    lab_link: None,
  },
  ConfigFieldDoc {
    key: "max_window_layers",
    aliases: &["use_sliding_window"],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::AttentionKv,
    value_type: "int | bool",
    summary_zh: "Qwen2/3 中控制前多少层使用全局注意力、其余层是否开启滑动窗口（SWA）的开关与分界层号。",
    summary_en: "Controls whether SWA is enabled and how many initial layers stay full global attention in Qwen2/3 configs.",
    formula_impact: None,
    focus: Some(InspectorFocus::Attention),
    example_model_slug: Some("qwen2-5-72b"),
    lab_link: None,
  },
  ConfigFieldDoc {
    key: "moe_layer_freq",
    aliases: &[],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::FfnMoe,
    value_type: "int",
    summary_zh: "DeepSeek-V2/V3 的 MoE 层出现频率（`1` 表示从 `first_k_dense_replace` 起每层都是 MoE）。",
    summary_en: "MoE layer frequency in DeepSeek-V2/V3 (`1` means every layer after `first_k_dense_replace` is MoE).",
    formula_impact: None,
    focus: Some(InspectorFocus::FfnOrMoe),
    example_model_slug: Some("deepseek-v3"),
    lab_link: None,
  },
  ConfigFieldDoc {
    key: "topk_method",
    aliases: &[],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::FfnMoe,
    value_type: "string",
    summary_zh: "DeepSeek-V3 MoE 专家选择策略（`noaux_tc` 表示无辅助损失动态偏置 + 组限贪心路由，`greedy` 为标准 Top-K）。",
    summary_en: "DeepSeek-V3 expert selection strategy (`noaux_tc` = aux-loss-free dynamic bias + group-limited greedy routing).",
    formula_impact: None,
    focus: Some(InspectorFocus::FfnOrMoe),
    example_model_slug: Some("deepseek-v3"),
    lab_link: Some(("/ai/components/moe-routing-quant-and-speculative-decoding", "无辅助损失 MoE 实验台")),
  },
  ConfigFieldDoc {
    key: "router_aux_loss_coef",
    aliases: &["output_router_logits"],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::FfnMoe,
    value_type: "float | bool",
    summary_zh: "训练期 MoE 负载均衡辅助损失权重系数（及是否返回路由器 Logits），推理部署期不参与前向输出。",
    summary_en: "Training-time MoE load-balancing auxiliary loss coefficient (and whether to return router logits); unused during inference.",
    formula_impact: None,
    focus: Some(InspectorFocus::FfnOrMoe),
    example_model_slug: Some("mixtral-8x7b"),
    lab_link: Some(("/ai/components/moe-routing-quant-and-speculative-decoding", "MoE 负载均衡实验台")),
  },
  ConfigFieldDoc {
    key: "rope_scaling.low_freq_factor",
    aliases: &["rope_scaling.high_freq_factor"],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::PositionContext,
    value_type: "float",
    summary_zh: "Llama-3.1/3.3 平滑分段 RoPE 外推的高低频分界系数（典型值 `low=1.0, high=4.0`），控制不同频段波长如何向 `factor=8.0` 过渡。",
    summary_en: "Llama-3.1/3.3 smooth RoPE scaling frequency ramp factors (`low=1.0, high=4.0`), blending wavelengths toward `factor=8.0`.",
    formula_impact: None,
    focus: Some(InspectorFocus::Attention),
    example_model_slug: Some("llama-3-1-8b"),
    lab_link: Some(("/ai/components/rope-and-yarn-extrapolation", "RoPE & YaRN 外推实验台")),
  },
  ConfigFieldDoc {
    key: "vision_config.num_heads",
    aliases: &["vision_config.num_attention_heads"],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::MultimodalPipeline,
    value_type: "int",
    summary_zh: "多模态视觉编码塔（ViT / SigLIP）内部的自注意力头数。",
    summary_en: "Number of self-attention heads inside the multimodal vision encoder tower.",
    formula_impact: None,
    focus: Some(InspectorFocus::Vision),
    example_model_slug: Some("qwen2-5-vl-72b"),
    lab_link: None,
  },
  ConfigFieldDoc {
    key: "vision_config.intermediate_size",
    aliases: &[],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::MultimodalPipeline,
    value_type: "int",
    summary_zh: "视觉编码塔内部每层 FFN 的中间维度，用于精确核算视觉编码器的参数占比。",
    summary_en: "Intermediate FFN width inside the vision encoder tower, governing vision encoder parameter share.",
    formula_impact: None,
    focus: Some(InspectorFocus::Vision),
    example_model_slug: Some("qwen2-5-vl-72b"),
    lab_link: None,
  },
  ConfigFieldDoc {
    key: "vision_config.out_hidden_size",
    aliases: &["mm_hidden_size"],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::MultimodalPipeline,
    value_type: "int",
    summary_zh: "视觉编码器经 Projector/Merger 映射后的输出维度，必须严格等于语言主干的 `hidden_size` 才能直接拼接输入。",
    summary_en: "Output width after the visual projector/merger; must equal the language backbone's `hidden_size` for token concatenation.",
    formula_impact: None,
    focus: Some(InspectorFocus::Vision),
    example_model_slug: Some("qwen2-5-vl-72b"),
    lab_link: None,
  },
  ConfigFieldDoc {
    key: "vision_config.temporal_patch_size",
    aliases: &["patch_size_t"],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::MultimodalPipeline,
    value_type: "int",
    summary_zh: "视频 VLM / 3D DiT 的时间轴分块步长（如 2 帧合为 1 个时空 Patch），将视频序列长度沿时间维度再压缩。",
    summary_en: "Temporal patch stride for video VLMs and 3D DiTs (e.g. grouping 2 frames per spatiotemporal patch), reducing sequence length.",
    formula_impact: Some("时空 Token 数 = (T / P_t) × (H / P_h) × (W / P_w)"),
    focus: Some(InspectorFocus::Vision),
    example_model_slug: Some("qwen2-5-vl-72b"),
    lab_link: None,
  },
  ConfigFieldDoc {
    key: "pooled_projection_dim",
    aliases: &[],
    file: ConfigFileKind::DiffusionPipeline,
    group: FieldGroup::MultimodalPipeline,
    value_type: "int",
    summary_zh: "FLUX / SD3.5 中来自 CLIP 池化句向量的条件输入维度（如 768），与时间步嵌入相加驱动 AdaLN-Zero 调制。",
    summary_en: "Pooled CLIP text embedding width (e.g. 768) in FLUX/SD3.5, added to timestep embeddings to drive AdaLN-Zero modulation.",
    formula_impact: None,
    focus: Some(InspectorFocus::NormAndResidual),
    example_model_slug: Some("flux-1-dev"),
    lab_link: Some(("/ai/components/rmsnorm-and-adaln-zero", "AdaLN-Zero 调制实验台")),
  },
  ConfigFieldDoc {
    key: "axes_dims_rope",
    aliases: &[],
    file: ConfigFileKind::DiffusionPipeline,
    group: FieldGroup::PositionContext,
    value_type: "int[]",
    summary_zh: "FLUX 等多维轴向 RoPE 的各轴维度分配（如 `[16, 56, 56]` 对应 `[时间/文本, 高度, 宽度]`），总和等于 `attention_head_dim`。",
    summary_en: "Per-axis RoPE dimension allocation in FLUX (`[16, 56, 56]` for `[axis0, H, W]`), summing to `attention_head_dim`.",
    formula_impact: Some("sum(axes_dims_rope) == attention_head_dim"),
    focus: Some(InspectorFocus::Attention),
    example_model_slug: Some("flux-1-dev"),
    lab_link: Some(("/ai/components/rope-and-yarn-extrapolation", "2D/3D 轴向 RoPE 实验台")),
  },
  ConfigFieldDoc {
    key: "block_out_channels",
    aliases: &["down_block_types", "up_block_types"],
    file: ConfigFileKind::DiffusionPipeline,
    group: FieldGroup::MultimodalPipeline,
    value_type: "int[] | string[]",
    summary_zh: "VAE 或 UNet 各下采样/上采样阶段的特征通道数（如 `[128, 256, 512, 512]`），数组长度减 1 即决定空间压缩倍率 `2^(N-1)`（典型 8×）。",
    summary_en: "Per-stage channel widths in VAE/UNet (`[128, 256, 512, 512]`); `2^(len - 1)` determines spatial downsampling factor (typically 8×).",
    formula_impact: Some("空间压缩倍率 = 2^(len(block_out_channels) - 1)"),
    focus: Some(InspectorFocus::LmHead),
    example_model_slug: Some("flux-1-dev"),
    lab_link: Some(("/ai/components/conv-and-depthwise-separable", "VAE 卷积与下采样实验台")),
  },
  ConfigFieldDoc {
    key: "type_vocab_size",
    aliases: &["position_embedding_type"],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::Backbone,
    value_type: "int | string",
    summary_zh: "BERT / RoBERTa 双向编码器的段落类型嵌入数与位置编码类型（`absolute` 查表位置嵌入 vs 现代 `rotary`）。",
    summary_en: "Segment type embedding count and position embedding type (`absolute` learned table vs modern `rotary`) in BERT/RoBERTa encoders.",
    formula_impact: None,
    focus: Some(InspectorFocus::Embedding),
    example_model_slug: Some("bge-m3"),
    lab_link: None,
  },
  ConfigFieldDoc {
    key: "d_state",
    aliases: &["state_size"],
    file: ConfigFileKind::ModelConfig,
    group: FieldGroup::AttentionKv,
    value_type: "int",
    summary_zh: "Mamba / Mamba-2 状态空间模型（SSM）的隐状态维度 N（如 16 或 128），控制 O(1) 递推状态的记忆容量。",
    summary_en: "State Space Model (Mamba/Mamba-2) latent state dimension N (e.g. 16 or 128), governing O(1) recurrent memory capacity.",
    formula_impact: Some("单层固定状态显存 ∝ d_inner × d_state（与上下文长度 S 无关）"),
    focus: Some(InspectorFocus::Attention),
    example_model_slug: Some("qwen3-next-80b-a3b"),
    lab_link: Some(("/ai/components/lstm-deltanet-and-mamba-ssm", "Mamba SSM 实验台")),
  },
  ConfigFieldDoc {
    key: "image_mean",
    aliases: &["image_std", "rescale_factor"],
    file: ConfigFileKind::PreprocessorConfig,
    group: FieldGroup::MultimodalPipeline,
    value_type: "float[] | float",
    summary_zh: "视觉预处理 RGB 像素缩放与标准化均值/标准差（常见 CLIP 参数或 `[0.5, 0.5, 0.5]`），必须与预训练视觉塔完全一致。",
    summary_en: "RGB pixel rescaling factor and channel mean/std normalization (CLIP stats or `0.5`); must match vision tower pretraining exactly.",
    formula_impact: Some("x_norm = (x × rescale_factor - mean) / std"),
    focus: Some(InspectorFocus::Vision),
    example_model_slug: Some("qwen2-5-vl-72b"),
    lab_link: None,
  },
  ConfigFieldDoc {
    key: "n_fft",
    aliases: &["hop_length", "chunk_length"],
    file: ConfigFileKind::PreprocessorConfig,
    group: FieldGroup::MultimodalPipeline,
    value_type: "int",
    summary_zh: "语音 STFT 窗长（如 400）、帧移（如 160 = 10ms @ 16kHz）与切片时长（如 30s），决定梅尔频谱的时间分辨率。",
    summary_en: "Audio STFT FFT window (`400`), hop stride (`160` = 10ms @ 16kHz), and chunk duration (`30s`), defining spectrogram frame rate.",
    formula_impact: Some("声学帧数 = sampling_rate × chunk_length / hop_length"),
    focus: Some(InspectorFocus::Audio),
    example_model_slug: Some("whisper-large-v3"),
    lab_link: Some(("/ai/components/audio-stft-conformer-ctc-and-rvq", "STFT & 梅尔语谱实验台")),
  },
  ConfigFieldDoc {
    key: "lora_dropout",
    aliases: &["bias"],
    file: ConfigFileKind::AdapterConfig,
    group: FieldGroup::TokenizerGeneration,
    value_type: "float | string",
    summary_zh: "LoRA 训练期适配器支路的 Dropout 比例与偏置训练策略（`none` / `all` / `lora_only`），推理期 Dropout 恒关闭。",
    summary_en: "Adapter branch dropout during LoRA training and bias training policy (`none`, `all`, `lora_only`); dropout is disabled at inference.",
    formula_impact: None,
    focus: None,
    example_model_slug: Some("qwen3-8b"),
    lab_link: None,
  },
];

// ─────────────────────────────────────────────────────────────────────────────
// 4. 常见大模型配置文件工程踩坑速查（Gotchas）
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "server")]
pub const CONFIG_GOTCHAS: &[ConfigGotcha] = &[
  ConfigGotcha {
    id: "tie-word-embeddings",
    badge: "参数核算",
    title_zh: "`tie_word_embeddings` 导致小模型参数量算错近 10 亿",
    title_en: "`tie_word_embeddings` Shifts Small-Model Param Counts by Nearly 1B",
    symptom_zh: "手算大词表模型（如 `vocab_size=151936, hidden_size=4096`）总参数时，结果与官方标注或 `.safetensors` 体积相差约 `0.62B`（约 1.2 GiB BF16）。",
    symptom_en: "Hand-computed parameter counts on large-vocab models (`151936 × 4096`) disagree with official numbers or shard sizes by ~0.62B (~1.2 GiB BF16).",
    root_cause_zh: "当 `tie_word_embeddings: true` 时，`lm_head.weight` 与 `model.embed_tokens.weight` 共享同一张张量，磁盘上甚至不保存 `lm_head.weight`。",
    root_cause_en: "When `tie_word_embeddings: true`, `lm_head.weight` aliases `embed_tokens.weight` and is omitted from `.safetensors` files.",
    fix_zh: "核算参数量与权重显存时必须检查 `tie_word_embeddings`：仅在 `false` 时额外计入一份 `V × D` 的 `lm_head` 参数。",
    fix_en: "Always branch on `tie_word_embeddings` when counting parameters or loading weights; only add `lm_head` (`V × D`) when `false`.",
    related_keys: &["tie_word_embeddings", "vocab_size", "hidden_size"],
  },
  ConfigGotcha {
    id: "vocab-size-padding",
    badge: "词表对齐",
    title_zh: "`config.json` 的 `vocab_size` 大于 `tokenizer.json` 实际词条数",
    title_en: "`config.json` `vocab_size` Exceeds `tokenizer.json` Token Count",
    symptom_zh: "用 `tokenizer.get_vocab_size()` 初始化 Embedding/LM Head 矩阵，加载 Safetensors 时立刻报错形状不匹配（如 151665 vs 151936）。",
    symptom_en: "Initializing Embedding/LM Head with `tokenizer.get_vocab_size()` fails at weight load with shape mismatch (e.g. 151665 vs 151936).",
    root_cause_zh: "训练团队会将权重矩阵的词表维度向上填充至 64/128 的整数倍以获得最佳 GPU Tensor Core GEMM 吞吐，尾部是未使用的预留槽位。",
    root_cause_en: "Training teams pad the weight matrix row count to multiples of 64/128 for Tensor Core GEMM efficiency, leaving trailing unused rows.",
    fix_zh: "模型张量分配永远以 `config.json` 的 `vocab_size`（或 Safetensors 真实 shape）为准，切勿用分词器词表大小替代。",
    fix_en: "Always allocate model weight tensors from `config.json`'s `vocab_size` (or actual safetensors shape), never from tokenizer length.",
    related_keys: &["vocab_size", "tokenizer_class"],
  },
  ConfigGotcha {
    id: "multi-eos-token-id",
    badge: "推理停不下来",
    title_zh: "只读 `config.json` 单一 `eos_token_id` 导致对话模型无限复读",
    title_en: "Reading Only Scalar `eos_token_id` in `config.json` Causes Runaway Generation",
    symptom_zh: "自研 Rust/Candle 推理引擎跑 Llama-3 / Qwen-Chat 时，模型回答完后不停机，继续自己扮演 User 往下编对话直到撞上 `max_new_tokens`。",
    symptom_en: "Custom Rust/Candle inference loops on Llama-3 / Qwen-Chat keep hallucinating subsequent user turns until hitting `max_new_tokens`.",
    root_cause_zh: "`config.json` 往往只记录预训练文档结束符（如 `<|endoftext|>`），而对话轮次结束符（`<|im_end|>` / `<|eot_id|>`）写在 `generation_config.json` 的 `eos_token_id` 数组里。",
    root_cause_en: "`config.json` often only lists the base document EOS, while chat turn stop tokens (`<|im_end|>`, `<|eot_id|>`) live as an array in `generation_config.json`.",
    fix_zh: "解析停止符时同时合并 `generation_config.json` 与 `config.json` 的标量或数组 `eos_token_id`，任一命中即停止生成。",
    fix_en: "Union both scalar and array `eos_token_id` values from `generation_config.json` and `config.json` into your stop-token set.",
    related_keys: &["eos_token_id", "do_sample", "temperature"],
  },
  ConfigGotcha {
    id: "explicit-head-dim",
    badge: "张量形状",
    title_zh: "硬算 `hidden_size / num_attention_heads` 在 Qwen3 / Gemma-3 上直接崩溃",
    title_en: "Assuming `head_dim = hidden_size / num_attention_heads` Breaks on Qwen3 & Gemma-3",
    symptom_zh: "照传统 Llama 公式推断 `head_dim = D / H_q` 后，加载 Qwen3-8B/30B-A3B 或 Gemma-3 权重时报 `q_proj` 维度不匹配。",
    symptom_en: "Computing `head_dim = D / H_q` causes `q_proj` shape mismatch when loading Qwen3 or Gemma-3 checkpoints.",
    root_cause_zh: "现代架构解耦了注意力总宽度与残差流宽度（例如 Qwen3-30B-A3B：`hidden_size=2048`、`num_attention_heads=32`、但显式 `head_dim=128`，于是 `Q` 宽度为 `4096 != 2048`）。",
    root_cause_en: "Modern architectures decouple attention width from `hidden_size` (e.g. Qwen3-30B-A3B has `D=2048`, `H=32`, explicit `head_dim=128`, so `Q=4096`).",
    fix_zh: "始终优先读取显式 `head_dim` 字段，仅在缺失时才回退到 `hidden_size / num_attention_heads`。",
    fix_en: "Always prefer explicit `head_dim` over `hidden_size / num_attention_heads`, falling back only when `head_dim` is absent.",
    related_keys: &["head_dim", "num_attention_heads", "num_key_value_heads"],
  },
  ConfigGotcha {
    id: "rope-scaling-context",
    badge: "长上下文乱码",
    title_zh: "只改 `max_position_embeddings` 却忽略 `rope_scaling` / `mscale` 导致长文坍塌",
    title_en: "Extending Context Without Honoring `rope_scaling` & `mscale` Causes Gibberish",
    symptom_zh: "输入超过 8K/32K 后，模型困惑度骤升、输出乱码或忘记前文，尽管 `max_position_embeddings` 写着 128K。",
    symptom_en: "Beyond 8K–32K tokens, perplexity spikes and outputs degrade into gibberish even though `max_position_embeddings` is 128K.",
    root_cause_zh: "YaRN / Llama-3 长上下文依赖分段频率缩放与 `mscale` 注意力温度修正，仅用原始 `rope_theta` 生成 cos/sin 表无法复现训练期相位分布。",
    root_cause_en: "YaRN and Llama-3 scaling require per-band frequency ramps and `mscale` attention temperature correction rather than raw `rope_theta`.",
    fix_zh: "完整实现 `rope_scaling` 子对象解析（区分 `yarn` / `llama3` / `linear` / `mrope`），并在 YaRN 下同步修正 Softmax 缩放。",
    fix_en: "Parse the full `rope_scaling` object (`yarn`, `llama3`, `linear`, `mrope`) and apply `mscale` to attention scaling when using YaRN.",
    related_keys: &["rope_theta", "rope_scaling.type", "rope_scaling.factor", "rope_scaling.mscale"],
  },
  ConfigGotcha {
    id: "diffusers-multi-slot",
    badge: "扩散管线",
    title_zh: "在图像/视频扩散仓库根目录找不到 `config.json` 或漏乘 VAE 缩放系数",
    title_en: "Missing Root `config.json` & Unscaled VAE Latents in Diffusers Repos",
    symptom_zh: "直接按 LLM 方式拉取 FLUX / Wan 根目录 `config.json` 报 404；或者手写推理管线跑通后，出图严重发灰、噪点满屏。",
    symptom_en: "Fetching root `config.json` on FLUX/Wan returns 404; or custom pipelines render washed-out, noisy images.",
    root_cause_zh: "Diffusers 仓库以 `model_index.json` 为入口，把真实超参拆在 `transformer/config.json`、`vae/config.json` 与 `scheduler/scheduler_config.json` 中；且 VAE 解码前必须反做 `scaling_factor` 与 `shift_factor`。",
    root_cause_en: "Diffusers splits configs across subdirectories referenced by `model_index.json`, and VAE decoding requires inverting `scaling_factor` and `shift_factor`.",
    fix_zh: "检测到 `model_index.json` 后逐槽位加载各子目录配置，严格读取 `vae/config.json` 的缩放偏移与 `scheduler_config.json` 的 Time-Shift 参数。",
    fix_en: "Detect `model_index.json`, load each slot's subdirectory config, and apply both VAE scaling/shift and scheduler Time-Shift.",
    related_keys: &["_class_name", "scaling_factor", "shift", "latent_channels"],
  },
];

// ─────────────────────────────────────────────────────────────────────────────
// 5. 服务端载荷构造、Zero-DB Server Functions 与 JSON 逐字段注解引擎
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "server")]
pub fn build_config_field_docs() -> Vec<ConfigFieldDocDto> {
  CONFIG_FIELD_DOCS
    .iter()
    .map(|d| ConfigFieldDocDto {
      key: d.key.to_string(),
      aliases: d.aliases.iter().map(|s| (*s).to_string()).collect(),
      file: d.file,
      group: d.group,
      value_type: d.value_type.to_string(),
      summary_zh: d.summary_zh.to_string(),
      summary_en: d.summary_en.to_string(),
      formula_impact: d.formula_impact.map(str::to_string),
      focus: d.focus,
      example_model_slug: d.example_model_slug.map(str::to_string),
      lab_link: d.lab_link.map(|(h, l)| (h.to_string(), l.to_string())),
    })
    .collect()
}

#[cfg(feature = "server")]
pub fn build_config_guide_payload() -> ConfigGuidePayload {
  ConfigGuidePayload {
    file_specs: CONFIG_FILE_SPECS
      .iter()
      .map(|s| ConfigFileSpecDto {
        kind: s.kind,
        filenames: s.filenames.to_string(),
        modalities_zh: s.modalities_zh.to_string(),
        modalities_en: s.modalities_en.to_string(),
        role_zh: s.role_zh.to_string(),
        role_en: s.role_en.to_string(),
        summary_zh: s.summary_zh.to_string(),
        summary_en: s.summary_en.to_string(),
        loader_zh: s.loader_zh.to_string(),
        loader_en: s.loader_en.to_string(),
        pitfall_zh: s.pitfall_zh.to_string(),
        pitfall_en: s.pitfall_en.to_string(),
        snippet: s.snippet.to_string(),
      })
      .collect(),
    repo_trees: REPO_MODALITY_BLUEPRINTS
      .iter()
      .map(|b| RepoModalityBlueprintDto {
        id: b.id.to_string(),
        label_zh: b.label_zh.to_string(),
        label_en: b.label_en.to_string(),
        example_repo: b.example_repo.to_string(),
        example_slug: b.example_slug.map(str::to_string),
        flow_zh: b.flow_zh.to_string(),
        flow_en: b.flow_en.to_string(),
        items: b
          .items
          .iter()
          .map(|it| RepoTreeItemDto {
            path: it.path.to_string(),
            required: it.required,
            file_kind: it.file_kind,
            note_zh: it.note_zh.to_string(),
            note_en: it.note_en.to_string(),
          })
          .collect(),
      })
      .collect(),
    field_docs: build_config_field_docs(),
    gotchas: CONFIG_GOTCHAS
      .iter()
      .map(|g| ConfigGotchaDto {
        id: g.id.to_string(),
        badge: g.badge.to_string(),
        title_zh: g.title_zh.to_string(),
        title_en: g.title_en.to_string(),
        symptom_zh: g.symptom_zh.to_string(),
        symptom_en: g.symptom_en.to_string(),
        root_cause_zh: g.root_cause_zh.to_string(),
        root_cause_en: g.root_cause_en.to_string(),
        fix_zh: g.fix_zh.to_string(),
        fix_en: g.fix_en.to_string(),
        related_keys: g.related_keys.iter().map(|s| (*s).to_string()).collect(),
      })
      .collect(),
  }
}

#[cfg(feature = "server")]
fn focus_code(f: InspectorFocus) -> &'static str {
  match f {
    InspectorFocus::Vision => "v",
    InspectorFocus::Audio => "a",
    InspectorFocus::Embedding => "e",
    InspectorFocus::Attention => "t",
    InspectorFocus::FfnOrMoe => "f",
    InspectorFocus::NormAndResidual => "n",
    InspectorFocus::LmHead => "l",
    InspectorFocus::MtpHead => "m",
  }
}

fn focus_from_code(s: &str) -> Option<InspectorFocus> {
  match s {
    "v" => Some(InspectorFocus::Vision),
    "a" => Some(InspectorFocus::Audio),
    "e" => Some(InspectorFocus::Embedding),
    "t" => Some(InspectorFocus::Attention),
    "f" => Some(InspectorFocus::FfnOrMoe),
    "n" => Some(InspectorFocus::NormAndResidual),
    "l" => Some(InspectorFocus::LmHead),
    "m" => Some(InspectorFocus::MtpHead),
    _ => None,
  }
}

#[cfg(feature = "server")]
static SERVER_GUIDE_JSON_CACHE: std::sync::OnceLock<String> = std::sync::OnceLock::new();

/// 将服务端静态词典编码为紧凑 JSON 字符串（复用 `Result<String, ServerFnError>` 零 `ciborium`/`serde` 结构体派生开销）。
#[cfg(feature = "server")]
pub fn build_config_guide_json() -> String {
  SERVER_GUIDE_JSON_CACHE
    .get_or_init(|| {
      let s: Vec<Value> = CONFIG_FILE_SPECS
        .iter()
        .map(|x| {
          Value::Array(vec![
            Value::String(x.kind.id().to_string()),
            Value::String(x.filenames.to_string()),
            Value::String(x.modalities_zh.to_string()),
            Value::String(x.modalities_en.to_string()),
            Value::String(x.role_zh.to_string()),
            Value::String(x.role_en.to_string()),
            Value::String(x.summary_zh.to_string()),
            Value::String(x.summary_en.to_string()),
            Value::String(x.loader_zh.to_string()),
            Value::String(x.loader_en.to_string()),
            Value::String(x.pitfall_zh.to_string()),
            Value::String(x.pitfall_en.to_string()),
            Value::String(x.snippet.to_string()),
          ])
        })
        .collect();
      let r: Vec<Value> = REPO_MODALITY_BLUEPRINTS
        .iter()
        .map(|b| {
          let items = b
            .items
            .iter()
            .map(|it| {
              Value::Array(vec![
                Value::String(it.path.to_string()),
                Value::Bool(it.required),
                Value::String(it.file_kind.id().to_string()),
                Value::String(it.note_zh.to_string()),
                Value::String(it.note_en.to_string()),
              ])
            })
            .collect();
          Value::Array(vec![
            Value::String(b.id.to_string()),
            Value::String(b.label_zh.to_string()),
            Value::String(b.label_en.to_string()),
            Value::String(b.example_repo.to_string()),
            b.example_slug.map(|v| Value::String(v.to_string())).unwrap_or(Value::Null),
            Value::String(b.flow_zh.to_string()),
            Value::String(b.flow_en.to_string()),
            Value::Array(items),
          ])
        })
        .collect();
      let d: Vec<Value> = CONFIG_FIELD_DOCS
        .iter()
        .map(|f| {
          let aliases = f.aliases.iter().map(|a| Value::String((*a).to_string())).collect();
          let lab = match f.lab_link {
            Some((h, l)) => {
              Value::Array(vec![Value::String(h.to_string()), Value::String(l.to_string())])
            }
            None => Value::Null,
          };
          Value::Array(vec![
            Value::String(f.key.to_string()),
            Value::Array(aliases),
            Value::String(f.file.id().to_string()),
            Value::String(f.group.id().to_string()),
            Value::String(f.value_type.to_string()),
            Value::String(f.summary_zh.to_string()),
            Value::String(f.summary_en.to_string()),
            f.formula_impact.map(|v| Value::String(v.to_string())).unwrap_or(Value::Null),
            f.focus.map(|fc| Value::String(focus_code(fc).to_string())).unwrap_or(Value::Null),
            f.example_model_slug.map(|v| Value::String(v.to_string())).unwrap_or(Value::Null),
            lab,
          ])
        })
        .collect();
      let g: Vec<Value> = CONFIG_GOTCHAS
        .iter()
        .map(|x| {
          let rks = x.related_keys.iter().map(|k| Value::String((*k).to_string())).collect();
          Value::Array(vec![
            Value::String(x.id.to_string()),
            Value::String(x.badge.to_string()),
            Value::String(x.title_zh.to_string()),
            Value::String(x.title_en.to_string()),
            Value::String(x.symptom_zh.to_string()),
            Value::String(x.symptom_en.to_string()),
            Value::String(x.root_cause_zh.to_string()),
            Value::String(x.root_cause_en.to_string()),
            Value::String(x.fix_zh.to_string()),
            Value::String(x.fix_en.to_string()),
            Value::Array(rks),
          ])
        })
        .collect();
      serde_json::to_string(&Value::Array(vec![
        Value::Array(s),
        Value::Array(r),
        Value::Array(d),
        Value::Array(g),
      ]))
      .unwrap_or_default()
    })
    .clone()
}

fn arr_str(a: &[Value], idx: usize) -> String {
  a.get(idx).and_then(Value::as_str).unwrap_or_default().to_string()
}

fn arr_opt_str(a: &[Value], idx: usize) -> Option<String> {
  a.get(idx).and_then(Value::as_str).map(str::to_string)
}

fn arr_str_list(a: &[Value], idx: usize) -> Vec<String> {
  a.get(idx)
    .and_then(Value::as_array)
    .map(|items| items.iter().filter_map(Value::as_str).map(str::to_string).collect())
    .unwrap_or_default()
}

/// 解析紧凑 JSON 载荷并缓存为静态引用（客户端与服务端各只解析一次，避免 `use_memo` 泛型单态化膨胀）。
pub fn get_or_parse_config_guide(raw: &str) -> &'static ConfigGuidePayload {
  static PARSED_GUIDE_CACHE: std::sync::OnceLock<ConfigGuidePayload> = std::sync::OnceLock::new();
  PARSED_GUIDE_CACHE.get_or_init(|| parse_config_guide_json(raw))
}

fn parse_field_docs_slice(rows: &[Value]) -> Vec<ConfigFieldDocDto> {
  rows
    .iter()
    .filter_map(Value::as_array)
    .map(|a| ConfigFieldDocDto {
      key: arr_str(a, 0),
      aliases: arr_str_list(a, 1),
      file: ConfigFileKind::from_id(&arr_str(a, 2)),
      group: FieldGroup::from_id(&arr_str(a, 3)),
      value_type: arr_str(a, 4),
      summary_zh: arr_str(a, 5),
      summary_en: arr_str(a, 6),
      formula_impact: arr_opt_str(a, 7),
      focus: a.get(8).and_then(Value::as_str).and_then(focus_from_code),
      example_model_slug: arr_opt_str(a, 9),
      lab_link: a
        .get(10)
        .and_then(Value::as_array)
        .and_then(|p| Some((p.first()?.as_str()?.to_string(), p.get(1)?.as_str()?.to_string()))),
    })
    .collect()
}

/// 解析 `/api/ai/config-guide` 紧凑 JSON 载荷为完整的 `ConfigGuidePayload`。
pub fn parse_config_guide_json(raw: &str) -> ConfigGuidePayload {
  let Ok(Value::Array(top)) = serde_json::from_str::<Value>(raw) else {
    return ConfigGuidePayload::default();
  };
  let file_specs = top
    .first()
    .and_then(Value::as_array)
    .map(|rows| {
      rows
        .iter()
        .filter_map(Value::as_array)
        .map(|a| ConfigFileSpecDto {
          kind: ConfigFileKind::from_id(&arr_str(a, 0)),
          filenames: arr_str(a, 1),
          modalities_zh: arr_str(a, 2),
          modalities_en: arr_str(a, 3),
          role_zh: arr_str(a, 4),
          role_en: arr_str(a, 5),
          summary_zh: arr_str(a, 6),
          summary_en: arr_str(a, 7),
          loader_zh: arr_str(a, 8),
          loader_en: arr_str(a, 9),
          pitfall_zh: arr_str(a, 10),
          pitfall_en: arr_str(a, 11),
          snippet: arr_str(a, 12),
        })
        .collect()
    })
    .unwrap_or_default();
  let repo_trees = top
    .get(1)
    .and_then(Value::as_array)
    .map(|rows| {
      rows
        .iter()
        .filter_map(Value::as_array)
        .map(|a| {
          let items = a
            .get(7)
            .and_then(Value::as_array)
            .map(|it_rows| {
              it_rows
                .iter()
                .filter_map(Value::as_array)
                .map(|it| RepoTreeItemDto {
                  path: arr_str(it, 0),
                  required: it.get(1).and_then(Value::as_bool).unwrap_or(false),
                  file_kind: ConfigFileKind::from_id(&arr_str(it, 2)),
                  note_zh: arr_str(it, 3),
                  note_en: arr_str(it, 4),
                })
                .collect()
            })
            .unwrap_or_default();
          RepoModalityBlueprintDto {
            id: arr_str(a, 0),
            label_zh: arr_str(a, 1),
            label_en: arr_str(a, 2),
            example_repo: arr_str(a, 3),
            example_slug: arr_opt_str(a, 4),
            flow_zh: arr_str(a, 5),
            flow_en: arr_str(a, 6),
            items,
          }
        })
        .collect()
    })
    .unwrap_or_default();
  let field_docs =
    parse_field_docs_slice(top.get(2).and_then(Value::as_array).map(Vec::as_slice).unwrap_or(&[]));
  let gotchas = top
    .get(3)
    .and_then(Value::as_array)
    .map(|rows| {
      rows
        .iter()
        .filter_map(Value::as_array)
        .map(|a| ConfigGotchaDto {
          id: arr_str(a, 0),
          badge: arr_str(a, 1),
          title_zh: arr_str(a, 2),
          title_en: arr_str(a, 3),
          symptom_zh: arr_str(a, 4),
          symptom_en: arr_str(a, 5),
          root_cause_zh: arr_str(a, 6),
          root_cause_en: arr_str(a, 7),
          fix_zh: arr_str(a, 8),
          fix_en: arr_str(a, 9),
          related_keys: arr_str_list(a, 10),
        })
        .collect()
    })
    .unwrap_or_default();
  ConfigGuidePayload { file_specs, repo_trees, field_docs, gotchas }
}

/// 下发 `/models/configs` 与 `/models/:slug` 共用的紧凑 JSON 词典载荷（零数据库依赖，复用 `String` 传输避免客户端 WASM 膨胀）。
#[get("/api/ai/config-guide")]
pub async fn get_config_guide_json() -> Result<String, ServerFnError> {
  Ok(build_config_guide_json())
}

#[cfg(not(feature = "server"))]
static CLIENT_GUIDE_JSON_CACHE: std::sync::OnceLock<String> = std::sync::OnceLock::new();

/// 带客户端内存缓存的词典 JSON 获取入口，切换页面或标签页时避免重复发起 HTTP 请求。
pub async fn get_config_guide_json_cached() -> Result<String, ServerFnError> {
  #[cfg(feature = "server")]
  {
    Ok(build_config_guide_json())
  }
  #[cfg(not(feature = "server"))]
  {
    if let Some(cached) = CLIENT_GUIDE_JSON_CACHE.get() {
      return Ok(cached.clone());
    }
    let raw = get_config_guide_json().await?;
    let _ = CLIENT_GUIDE_JSON_CACHE.set(raw.clone());
    Ok(raw)
  }
}

/// 单个真实模型配置中解析出的一条注解字段。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnnotatedConfigField {
  /// 展示用键路径（如 `hidden_size`、`rope_scaling.type`、`vision_config.patch_size`）
  pub path: String,
  /// 来源槽位或附属文件名（Diffusers 如 `transformer/config.json`，单文件 `config.json` 为 `None`）
  pub source_slot: Option<String>,
  /// 格式化后的真实值（标量完整展示，大对象/长数组安全折叠）
  pub display_value: String,
  /// 语义分组
  pub group: FieldGroup,
  /// 命中的词典文档（未知扩展字段为 `None`）
  pub doc: Option<ConfigFieldDocDto>,
  /// 推荐的架构图联动焦点
  pub focus: Option<InspectorFocus>,
}

/// 按规范键、别名与多模态/Diffusers 槽位前缀剥离规则查询服务端静态字段词典。
#[cfg(feature = "server")]
pub fn lookup_field_doc(path: &str) -> Option<&'static ConfigFieldDoc> {
  lookup_field_doc_with_hint(path, None)
}

/// 带文件类型提示的服务端静态字段词典查询。
#[cfg(feature = "server")]
pub fn lookup_field_doc_with_hint(
  path: &str,
  file_hint: Option<ConfigFileKind>,
) -> Option<&'static ConfigFieldDoc> {
  let clean = path.trim();
  if clean.is_empty() {
    return None;
  }

  let find_static = |candidate: &str| -> Option<&'static ConfigFieldDoc> {
    if let Some(hint) = file_hint {
      if let Some(doc) = CONFIG_FIELD_DOCS
        .iter()
        .find(|d| d.file == hint && (d.key == candidate || d.aliases.contains(&candidate)))
      {
        return Some(doc);
      }
    }
    CONFIG_FIELD_DOCS.iter().find(|d| d.key == candidate || d.aliases.contains(&candidate))
  };

  if let Some(doc) = find_static(clean) {
    return Some(doc);
  }
  let stripped = strip_known_container_prefix(clean);
  if stripped != clean {
    if let Some(doc) = find_static(stripped) {
      return Some(doc);
    }
  }
  if let Some(rest) =
    clean.strip_prefix("thinker_config.").or_else(|| clean.strip_prefix("talker_config."))
  {
    if let Some(doc) = find_static(rest) {
      return Some(doc);
    }
  }
  if let Some((prefix, leaf)) = stripped.rsplit_once('.') {
    let is_isolated_submodel = prefix.contains("vision_config")
      || prefix.contains("audio_config")
      || prefix.contains("text_encoder")
      || prefix.contains("tokenizer")
      || prefix.contains("vae");
    if !is_isolated_submodel {
      if let Some(doc) = find_static(leaf) {
        return Some(doc);
      }
    } else if prefix.contains("vae")
      || prefix.contains("vision_config")
      || prefix.contains("audio_config")
    {
      if let Some(doc) = find_static(leaf)
        .filter(|d| d.group == FieldGroup::MultimodalPipeline || d.key == "torch_dtype")
      {
        return Some(doc);
      }
    }
  }
  None
}

/// 在已加载的字段词典切片中按规范键、别名与多模态/Diffusers 槽位前缀剥离规则查询条目（服务端/WASM 通用）。
pub fn lookup_field_doc_in<'a>(
  docs: &'a [ConfigFieldDocDto],
  path: &str,
  file_hint: Option<ConfigFileKind>,
) -> Option<&'a ConfigFieldDocDto> {
  let clean = path.trim();
  if clean.is_empty() {
    return None;
  }

  // 1. 直接精确匹配（优先命中指定文件类型）
  if let Some(doc) = find_exact_in(docs, clean, file_hint) {
    return Some(doc);
  }

  // 2. 剥离已知顶层多模态外壳或 Diffusers 槽位前缀后精确匹配
  let stripped = strip_known_container_prefix(clean);
  if stripped != clean {
    if let Some(doc) = find_exact_in(docs, stripped, file_hint) {
      return Some(doc);
    }
  }

  // 3. 保留单层语义命名空间（如 `thinker_config.vision_config.patch_size` -> `vision_config.patch_size`）
  if let Some(rest) =
    clean.strip_prefix("thinker_config.").or_else(|| clean.strip_prefix("talker_config."))
  {
    if let Some(doc) = find_exact_in(docs, rest, file_hint) {
      return Some(doc);
    }
  }

  // 4. 对于非 vision/audio/vae/text_encoder 的普通前缀（如 `transformer/config.json.num_layers`），
  //    仅在末段 key 属于目标文件（或无冲突）时做末段匹配
  if let Some((prefix, leaf)) = stripped.rsplit_once('.') {
    let is_isolated_submodel = prefix.contains("vision_config")
      || prefix.contains("audio_config")
      || prefix.contains("text_encoder")
      || prefix.contains("tokenizer")
      || prefix.contains("vae");
    if !is_isolated_submodel {
      if let Some(doc) = find_exact_in(docs, leaf, file_hint) {
        return Some(doc);
      }
    } else if prefix.contains("vae")
      || prefix.contains("vision_config")
      || prefix.contains("audio_config")
    {
      // 仅匹配明确归属多模态/扩散/池化组的词条，防止把 vision/vae 的 `hidden_size` 误认成 LLM 主干
      if let Some(doc) = find_exact_in(docs, leaf, file_hint)
        .filter(|d| d.group == FieldGroup::MultimodalPipeline || d.key == "torch_dtype")
      {
        return Some(doc);
      }
    }
  }

  None
}

fn find_exact_in<'a>(
  docs: &'a [ConfigFieldDocDto],
  candidate: &str,
  file_hint: Option<ConfigFileKind>,
) -> Option<&'a ConfigFieldDocDto> {
  if let Some(hint) = file_hint {
    if let Some(doc) = docs
      .iter()
      .find(|d| d.file == hint && (d.key == candidate || d.aliases.iter().any(|a| a == candidate)))
    {
      return Some(doc);
    }
  }
  docs.iter().find(|d| d.key == candidate || d.aliases.iter().any(|a| a == candidate))
}

fn strip_known_container_prefix(path: &str) -> &str {
  for prefix in [
    "thinker_config.text_config.",
    "text_config.",
    "llm_config.",
    "language_config.",
    "decoder.",
    "model_index.json.",
    "transformer/config.json.",
    "unet/config.json.",
    "vae/config.json.",
    "scheduler/config.json.",
    "scheduler/scheduler_config.json.",
    "text_encoder/config.json.",
    "text_encoder_2/config.json.",
    "text_encoder_3/config.json.",
  ] {
    if let Some(rest) = path.strip_prefix(prefix) {
      return rest;
    }
  }
  path
}

/// 将任意模型的 `raw_config_pretty`（含标准 `config.json` 与 Diffusers 多槽位组合 JSON）解析为带词典释义的扁平字段列表（服务端便捷入口）。
#[cfg(feature = "server")]
pub fn extract_annotated_fields(raw_json: &str) -> Vec<AnnotatedConfigField> {
  extract_annotated_fields_for_file(raw_json, ConfigFileKind::ModelConfig)
}

/// 面向指定配置文件类型的 JSON 逐字段解析器（服务端便捷入口）。
#[cfg(feature = "server")]
pub fn extract_annotated_fields_for_file(
  raw_json: &str,
  default_file: ConfigFileKind,
) -> Vec<AnnotatedConfigField> {
  let docs = build_config_field_docs();
  extract_annotated_fields_with_docs(raw_json, default_file, &docs)
}

/// 面向指定配置文件类型与已下发词典的 JSON 逐字段解析器（服务端/WASM 通用）。
pub fn extract_annotated_fields_with_docs(
  raw_json: &str,
  default_file: ConfigFileKind,
  docs: &[ConfigFieldDocDto],
) -> Vec<AnnotatedConfigField> {
  let Ok(root) = serde_json::from_str::<Value>(raw_json) else {
    return Vec::new();
  };
  let Some(obj) = root.as_object() else {
    return Vec::new();
  };

  let is_combined_diffusers = obj.contains_key("model_index.json")
    || obj.keys().any(|k| k.ends_with("/config.json") || k.ends_with("/scheduler_config.json"));

  let mut out = Vec::new();
  if is_combined_diffusers {
    for (slot_key, slot_val) in obj {
      let slot_hint = if slot_key.contains("scheduler")
        || slot_key == "model_index.json"
        || slot_key.contains("transformer")
        || slot_key.contains("unet")
        || slot_key.contains("vae")
      {
        ConfigFileKind::DiffusionPipeline
      } else {
        ConfigFileKind::ModelConfig
      };
      if let Some(slot_obj) = slot_val.as_object() {
        flatten_object(slot_obj, "", Some(slot_key.as_str()), slot_hint, 0, docs, &mut out);
      } else {
        push_leaf(slot_key, slot_val, None, slot_hint, docs, &mut out);
      }
    }
  } else {
    flatten_object(obj, "", None, default_file, 0, docs, &mut out);
  }

  // 排序：按语义分组有序聚合，组内优先展示命中词典的重点字段，再按原始键名字母序确定性排列
  out.sort_unstable_by(|a, b| {
    a.group
      .cmp(&b.group)
      .then_with(|| b.doc.is_some().cmp(&a.doc.is_some()))
      .then_with(|| a.source_slot.cmp(&b.source_slot))
      .then_with(|| a.path.cmp(&b.path))
  });
  out
}

/// 按语义分组聚合注解字段，便于 UI 直接渲染分组折叠/卡片。
pub fn group_annotated_fields(
  fields: &[AnnotatedConfigField],
) -> Vec<(FieldGroup, Vec<AnnotatedConfigField>)> {
  let mut buckets: BTreeMap<FieldGroup, Vec<AnnotatedConfigField>> = BTreeMap::new();
  for f in fields {
    buckets.entry(f.group).or_default().push(f.clone());
  }
  buckets.into_iter().collect()
}

/// 对附属配置文件（如 `tokenizer_config.json`）做体积安全的摘要裁剪，避免 `added_tokens_decoder` 等超大字典撑爆 SSR/数据库序列化体积。
#[cfg(feature = "server")]
pub fn compact_aux_json_for_snapshot(raw_json: &str) -> Option<String> {
  let val: Value = serde_json::from_str(raw_json).ok()?;
  let obj = val.as_object()?;
  let mut compacted = serde_json::Map::new();
  for (k, v) in obj {
    match v {
      Value::Object(m) if m.len() > 16 || k == "added_tokens_decoder" => {
        compacted.insert(k.clone(), Value::String(format!("{{{} entries}}", m.len())));
      }
      Value::Array(arr) if arr.len() > 24 => {
        compacted.insert(k.clone(), Value::String(format!("[{} items]", arr.len())));
      }
      Value::String(s) if s.len() > 220 => {
        let preview: String = s.chars().take(180).collect();
        compacted
          .insert(k.clone(), Value::String(format!("{preview}… ({} chars)", s.chars().count())));
      }
      _ => {
        compacted.insert(k.clone(), v.clone());
      }
    }
  }
  serde_json::to_string_pretty(&Value::Object(compacted)).ok()
}

fn flatten_object(
  obj: &serde_json::Map<String, Value>,
  prefix: &str,
  source_slot: Option<&str>,
  file_hint: ConfigFileKind,
  depth: usize,
  docs: &[ConfigFieldDocDto],
  out: &mut Vec<AnnotatedConfigField>,
) {
  for (k, v) in obj {
    let full_key = if prefix.is_empty() { k.clone() } else { format!("{prefix}.{k}") };
    match v {
      Value::Object(sub) => {
        // 避免展开超大字典（如 id2label / added_tokens_decoder / weight_map）造成噪音
        let is_noisy_map = sub.len() > 18
          || matches!(
            k.as_str(),
            "id2label" | "label2id" | "added_tokens_decoder" | "special_tokens_map" | "weight_map"
          );
        if is_noisy_map || depth >= 3 || sub.is_empty() {
          push_leaf(&full_key, v, source_slot, file_hint, docs, out);
        } else {
          flatten_object(sub, &full_key, source_slot, file_hint, depth + 1, docs, out);
        }
      }
      _ => push_leaf(&full_key, v, source_slot, file_hint, docs, out),
    }
  }
}

fn push_leaf(
  full_key: &str,
  val: &Value,
  source_slot: Option<&str>,
  file_hint: ConfigFileKind,
  docs: &[ConfigFieldDocDto],
  out: &mut Vec<AnnotatedConfigField>,
) {
  let lookup_path = match source_slot {
    Some(slot) => format!("{slot}.{full_key}"),
    None => full_key.to_string(),
  };
  let doc = lookup_field_doc_in(docs, &lookup_path, Some(file_hint))
    .or_else(|| lookup_field_doc_in(docs, full_key, Some(file_hint)));

  let group = infer_group(&lookup_path, doc, file_hint);
  let focus = infer_focus(&lookup_path, doc);
  let display_value = format_json_value(val);

  out.push(AnnotatedConfigField {
    path: full_key.to_string(),
    source_slot: source_slot.map(str::to_string),
    display_value,
    group,
    doc: doc.cloned(),
    focus,
  });
}

fn infer_group(
  path: &str,
  doc: Option<&ConfigFieldDocDto>,
  file_hint: ConfigFileKind,
) -> FieldGroup {
  if let Some(d) = doc {
    return d.group;
  }
  let lower = path.to_ascii_lowercase();
  if matches!(
    file_hint,
    ConfigFileKind::GenerationConfig
      | ConfigFileKind::TokenizerConfig
      | ConfigFileKind::AdapterConfig
  ) {
    return FieldGroup::TokenizerGeneration;
  }
  if matches!(
    file_hint,
    ConfigFileKind::PreprocessorConfig
      | ConfigFileKind::DiffusionPipeline
      | ConfigFileKind::EmbeddingModules
  ) {
    return FieldGroup::MultimodalPipeline;
  }
  if lower.contains("quant") || lower.contains("dtype") || lower.contains("fp8") {
    FieldGroup::QuantPrecision
  } else if lower.contains("rope") || lower.contains("position") {
    FieldGroup::PositionContext
  } else if lower.contains("moe")
    || lower.contains("expert")
    || lower.contains("mlp")
    || lower.contains("ffn")
  {
    FieldGroup::FfnMoe
  } else if lower.contains("attn")
    || lower.contains("attention")
    || lower.contains("head")
    || lower.contains("kv_")
    || lower.contains("sliding_window")
    || lower.contains("linear_")
    || lower.contains("ssm")
  {
    FieldGroup::AttentionKv
  } else if lower.contains("vision")
    || lower.contains("image")
    || lower.contains("video")
    || lower.contains("audio")
    || lower.contains("speech")
    || lower.contains("vae")
    || lower.contains("pool")
    || lower.contains("patch")
  {
    FieldGroup::MultimodalPipeline
  } else if lower.contains("token") || lower.contains("chat") || lower.contains("lora") {
    FieldGroup::TokenizerGeneration
  } else {
    FieldGroup::Backbone
  }
}

fn infer_focus(path: &str, doc: Option<&ConfigFieldDocDto>) -> Option<InspectorFocus> {
  if let Some(d) = doc {
    if let Some(f) = d.focus {
      return Some(f);
    }
  }
  let lower = path.to_ascii_lowercase();
  if lower.contains("vision_config") || lower.contains("image_") || lower.contains("video_") {
    Some(InspectorFocus::Vision)
  } else if lower.contains("audio_config") || lower.contains("mel") || lower.contains("speech") {
    Some(InspectorFocus::Audio)
  } else if lower.contains("vae/") {
    Some(InspectorFocus::LmHead)
  } else if lower.contains("nextn") || lower.contains("mtp") {
    Some(InspectorFocus::MtpHead)
  } else if lower.contains("expert") || lower.contains("moe") || lower.contains("mlp") {
    Some(InspectorFocus::FfnOrMoe)
  } else if lower.contains("attn") || lower.contains("kv_") || lower.contains("rope") {
    Some(InspectorFocus::Attention)
  } else {
    None
  }
}

fn format_json_value(val: &Value) -> String {
  match val {
    Value::Null => "null".to_string(),
    Value::Bool(b) => b.to_string(),
    Value::Number(n) => n.to_string(),
    Value::String(s) => {
      let char_count = s.chars().count();
      if char_count > 72 {
        let head: String = s.chars().take(64).collect();
        format!("\"{head}…\" ({char_count} chars)")
      } else {
        format!("\"{s}\"")
      }
    }
    Value::Array(arr) => {
      if arr.is_empty() {
        "[]".to_string()
      } else if arr.len() <= 6
        && arr.iter().all(|v| v.is_number() || v.is_boolean() || v.is_string())
      {
        let items: Vec<String> = arr
          .iter()
          .map(|v| match v {
            Value::String(s) => format!("\"{s}\""),
            _ => v.to_string(),
          })
          .collect();
        let joined = format!("[{}]", items.join(", "));
        if joined.chars().count() <= 80 {
          joined
        } else {
          format!("[{} items]", arr.len())
        }
      } else {
        format!("[{} items]", arr.len())
      }
    }
    Value::Object(map) => {
      if map.is_empty() {
        "{}".to_string()
      } else {
        format!("{{{} entries}}", map.len())
      }
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. `/models/configs` 全景配置文件解剖指南页 UI
// ─────────────────────────────────────────────────────────────────────────────

fn current_lang() -> Language {
  try_consume_context::<Signal<Language>>().map(|s| s()).unwrap_or_default()
}

/// `/models/configs` 独立全景指南页：拆解 HuggingFace / ModelScope 模型仓库中 8 类配置文件的协作关系与高频字段。
#[component]
pub fn ModelConfigsGuidePage() -> Element {
  let lang = current_lang();
  let title = lang
    .pick(
      "大模型配置文件全景解剖指南（config.json / safetensors / tokenizer / model_index）",
      "LLM & Multimodal Model Config Files Guide (config.json / safetensors / tokenizer / model_index)",
    )
    .to_string();
  let desc = lang
    .pick(
      "面向模型架构学习者与推理引擎开发者的实战手册：一图看懂 LLM、VLM、Diffusion、Embedding 与 LoRA 仓库里每个 JSON 文件的职责、80+ 高频字段对显存与张量形状的定量影响及常见工程踩坑。",
      "A practical reference for learning LLM/VLM/Diffusion/Embedding/LoRA architectures from repository config files: 8 file blueprints, 80+ searchable fields with memory/shape formulas, and real engineering gotchas.",
    )
    .to_string();

  rsx! {
    PageMeta {
      title: title.clone(),
      description: desc.clone(),
      path: "/models/configs".to_string(),
    }

    section { class: "py-10 bg-white dark:bg-slate-950 min-h-screen",
      div { class: "max-w-7xl mx-auto px-4 sm:px-6",
        SuspenseBoundary {
          fallback: |_| rsx! {
            div { class: "flex items-center justify-center py-24",
              Spinner { size: SpinnerSize::Lg, class: "border-t-primary" }
            }
          },
          ModelConfigsGuideLoaded { desc }
        }
      }
    }
  }
}

#[component]
fn ModelConfigsGuideLoaded(desc: String) -> Element {
  let lang = current_lang();
  let res = use_server_future(get_config_guide_json_cached)?;
  let active_modality = use_signal(|| "llm".to_string());
  let active_file_spec = use_signal(|| ConfigFileKind::ModelConfig);
  let filter_file: Signal<Option<ConfigFileKind>> = use_signal(|| None);
  let filter_group: Signal<Option<FieldGroup>> = use_signal(|| None);
  let search_q = use_signal(String::new);

  match res() {
    Some(Err(_)) => rsx! {
      widgets::LoadFailed {
        message: lang
          .pick("配置文件全景指南加载失败，请稍后重试", "Failed to load Model Config Files Guide")
          .to_string(),
      }
    },
    None => rsx! {
      div { class: "flex items-center justify-center py-24",
        Spinner { size: SpinnerSize::Lg, class: "border-t-primary" }
      }
    },
    Some(Ok(json)) => {
      let payload = get_or_parse_config_guide(&json);
      render_config_guide_content(
        payload,
        &desc,
        active_modality,
        active_file_spec,
        filter_file,
        filter_group,
        search_q,
      )
    }
  }
}

pub(crate) fn render_config_guide_content(
  payload: &ConfigGuidePayload,
  desc: &str,
  mut active_modality: Signal<String>,
  mut active_file_spec: Signal<ConfigFileKind>,
  mut filter_file: Signal<Option<ConfigFileKind>>,
  mut filter_group: Signal<Option<FieldGroup>>,
  mut search_q: Signal<String>,
) -> Element {
  let lang = current_lang();

  let cur_tree = payload
    .repo_trees
    .iter()
    .find(|b| b.id == active_modality())
    .or_else(|| payload.repo_trees.first());
  let cur_spec = payload
    .file_specs
    .iter()
    .find(|s| s.kind == active_file_spec())
    .or_else(|| payload.file_specs.first());

  let q_norm = search_q().trim().to_ascii_lowercase();
  let sel_file = filter_file();
  let sel_group = filter_group();

  let filtered_docs: Vec<&ConfigFieldDocDto> = payload
    .field_docs
    .iter()
    .filter(|d| sel_file.is_none_or(|f| d.file == f))
    .filter(|d| sel_group.is_none_or(|g| d.group == g))
    .filter(|d| {
      if q_norm.is_empty() {
        return true;
      }
      d.key.to_ascii_lowercase().contains(&q_norm)
        || d.aliases.iter().any(|a| a.to_ascii_lowercase().contains(&q_norm))
        || d.summary_zh.to_ascii_lowercase().contains(&q_norm)
        || d.summary_en.to_ascii_lowercase().contains(&q_norm)
        || d.file.short_name().to_ascii_lowercase().contains(&q_norm)
        || d.formula_impact.as_deref().is_some_and(|f| f.to_ascii_lowercase().contains(&q_norm))
    })
    .collect();

  let has_active_filter = sel_file.is_some() || sel_group.is_some() || !q_norm.is_empty();

  rsx! {
    div { class: "space-y-10",
      // 1. 顶部 Hero 区域
      div { class: "rounded-3xl border border-slate-200 dark:border-slate-800 bg-linear-to-br from-slate-50 via-white to-indigo-50/70 dark:from-slate-900 dark:via-slate-950 dark:to-indigo-950/30 p-6 sm:p-8 shadow-xs",
        div { class: "flex flex-wrap items-center gap-2 text-xs font-semibold text-indigo-600 dark:text-indigo-400 mb-3",
          Link { to: "/models", class: "hover:underline",
            {lang.pick("← 返回主流大模型架构图鉴", "← Back to Model Atlas")}
          }
          span { class: "text-slate-300 dark:text-slate-700", "·" }
          Link { to: "/ai/visualizer", class: "hover:underline text-slate-600 dark:text-slate-400",
            {lang.pick("打开任意仓库实时解析器 →", "Open Live Repo Visualizer →")}
          }
        }

        div { class: "flex flex-col lg:flex-row lg:items-end justify-between gap-6",
          div { class: "max-w-3xl",
            div { class: "inline-flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-indigo-600 dark:text-indigo-400 mb-2",
              span { class: "px-2.5 py-0.5 rounded-full bg-indigo-600 text-white text-[10px]",
                "Config Anatomy"
              }
              span {
                {lang.pick(
                  "读懂仓库里每一个 JSON · 与架构拓扑图双向联动",
                  "Master Every Repo JSON · Linked to Live Architecture Graphs",
                )}
              }
            }
            h1 { class: "text-2xl sm:text-4xl font-extrabold tracking-tight text-slate-900 dark:text-white",
              {lang.pick("大模型配置文件全景解剖指南", "Model Repository Config Files Guide")}
            }
            p { class: "mt-2.5 text-sm sm:text-base text-slate-600 dark:text-slate-400 leading-relaxed",
              "{desc}"
            }
          }

          div { class: "grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-2 gap-2.5 shrink-0",
            div { class: "rounded-xl border border-slate-200 dark:border-slate-800 bg-white/80 dark:bg-slate-900/80 px-3.5 py-2.5",
              div { class: "text-lg font-extrabold font-mono text-indigo-600 dark:text-indigo-400",
                "{payload.file_specs.len()}"
              }
              div { class: "text-[11px] text-slate-500 dark:text-slate-400",
                {lang.pick("类核心配置文件", "Core Config Files")}
              }
            }
            div { class: "rounded-xl border border-slate-200 dark:border-slate-800 bg-white/80 dark:bg-slate-900/80 px-3.5 py-2.5",
              div { class: "text-lg font-extrabold font-mono text-emerald-600 dark:text-emerald-400",
                "{payload.field_docs.len()}+"
              }
              div { class: "text-[11px] text-slate-500 dark:text-slate-400",
                {lang.pick("高频字段双语词典", "Annotated Fields")}
              }
            }
            div { class: "rounded-xl border border-slate-200 dark:border-slate-800 bg-white/80 dark:bg-slate-900/80 px-3.5 py-2.5",
              div { class: "text-lg font-extrabold font-mono text-violet-600 dark:text-violet-400",
                "{payload.repo_trees.len()}"
              }
              div { class: "text-[11px] text-slate-500 dark:text-slate-400",
                {lang.pick("模态仓库文件树", "Modality Repo Trees")}
              }
            }
            div { class: "rounded-xl border border-slate-200 dark:border-slate-800 bg-white/80 dark:bg-slate-900/80 px-3.5 py-2.5",
              div { class: "text-lg font-extrabold font-mono text-amber-600 dark:text-amber-400",
                "{payload.gotchas.len()}"
              }
              div { class: "text-[11px] text-slate-500 dark:text-slate-400",
                {lang.pick("工程高频踩坑", "Production Gotchas")}
              }
            }
          }
        }
      }

      // 2. 按模态切换仓库文件树全景图
      div { class: card_class("p-5 sm:p-6 bg-white dark:bg-slate-900/60 space-y-5"),
        div { class: "flex flex-col md:flex-row md:items-center justify-between gap-3",
          div {
            h2 { class: "text-lg sm:text-xl font-extrabold text-slate-900 dark:text-white",
              {lang.pick("1. 模型仓库文件树与加载协作关系", "1. Repository File Tree by Modality")}
            }
            p { class: "mt-1 text-xs sm:text-sm text-slate-500 dark:text-slate-400",
              {lang.pick(
                "切换不同模态，快速看清打开一个 HuggingFace / ModelScope 仓库时，各个文件各自管什么、按什么顺序被推理引擎消费。",
                "Switch modalities to see what each JSON/weight file controls and how inference engines load them together.",
              )}
            }
          }

          div { class: "flex flex-wrap gap-1.5",
            for bp in payload.repo_trees.iter() {
              {
                let bid = bp.id.clone();
                let active = active_modality() == bid;
                let cls = if active {
                  "px-3 py-1.5 rounded-xl text-xs font-bold bg-indigo-600 text-white shadow-xs cursor-pointer transition-all"
                } else {
                  "px-3 py-1.5 rounded-xl text-xs font-semibold bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700 cursor-pointer transition-all"
                };
                rsx! {
                  button {
                    key: "tree-{bid}",
                    r#type: "button",
                    class: cls,
                    onclick: move |_| active_modality.set(bid.clone()),
                    "{bp.label(lang)}"
                  }
                }
              }
            }
          }
        }

        if let Some(tree) = cur_tree {
          div { class: "rounded-2xl border border-indigo-200/80 dark:border-indigo-900/60 bg-indigo-50/50 dark:bg-indigo-950/25 px-4 py-3 flex flex-col sm:flex-row sm:items-center justify-between gap-2 text-xs",
            div { class: "space-y-1",
              div { class: "font-bold text-indigo-700 dark:text-indigo-300",
                {lang.pick("推理加载数据流：", "Loader Pipeline Flow: ")}
                span { class: "font-normal text-slate-700 dark:text-slate-300", "{tree.flow(lang)}" }
              }
              div { class: "font-mono text-[11px] text-slate-500 dark:text-slate-400",
                {lang.pick("典型仓库参考：", "Reference Repos: ")}
                "{tree.example_repo}"
              }
            }
            if let Some(ref slug) = tree.example_slug {
              Link {
                to: format!("/models/{slug}"),
                class: "shrink-0 inline-flex items-center gap-1 px-3 py-1.5 rounded-lg bg-indigo-600 text-white font-semibold hover:bg-indigo-700 transition-colors",
                {lang.pick("查看真实代表模型 →", "Inspect Example Model →")}
              }
            }
          }

          div { class: "grid grid-cols-1 md:grid-cols-2 gap-3",
            for item in tree.items.iter() {
              {
                let fk = item.file_kind;
                rsx! {
                  div {
                    key: "{tree.id}-{item.path}",
                    class: "rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/60 dark:bg-slate-950/50 p-3.5 flex flex-col justify-between gap-2",
                    div { class: "flex items-start justify-between gap-2",
                      code { class: "text-xs sm:text-sm font-mono font-bold text-slate-900 dark:text-white break-all",
                        "{item.path}"
                      }
                      if item.required {
                        span { class: badge_class(BadgeVariant::Default, "text-[10px] shrink-0"),
                          {lang.pick("必需", "Required")}
                        }
                      } else {
                        span { class: badge_class(BadgeVariant::Secondary, "text-[10px] shrink-0"),
                          {lang.pick("可选", "Optional")}
                        }
                      }
                    }
                    p { class: "text-xs text-slate-600 dark:text-slate-400 leading-relaxed",
                      "{item.note(lang)}"
                    }
                    div { class: "flex items-center justify-between pt-1",
                      span { class: "text-[11px] font-mono text-slate-400", "{fk.short_name()}" }
                      button {
                        r#type: "button",
                        class: "text-[11px] font-semibold text-indigo-600 dark:text-indigo-400 hover:underline cursor-pointer",
                        onclick: move |_| {
                          active_file_spec.set(fk);
                          filter_file.set(Some(fk));
                        },
                        {lang.pick("聚焦此文件蓝图与字段 ↓", "Inspect File & Fields ↓")}
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }

      // 3. 8 类主流配置文件深度蓝图
      div { class: card_class("p-5 sm:p-6 bg-white dark:bg-slate-900/60 space-y-5"),
        div {
          h2 { class: "text-lg sm:text-xl font-extrabold text-slate-900 dark:text-white",
            {lang.pick("2. 八大主流配置文件深度拆解", "2. Deep Dive: The 8 Standard Config Files")}
          }
          p { class: "mt-1 text-xs sm:text-sm text-slate-500 dark:text-slate-400",
            {lang.pick(
              "点击任意配置文件类型，查看它的核心职责、推理框架消费机制、典型避坑点与真实精简 JSON 示例。",
              "Select any config file type to view its responsibilities, engine loading mechanics, pitfalls, and real JSON anatomy.",
            )}
          }
        }

        div { class: "flex flex-wrap gap-1.5",
          for spec in payload.file_specs.iter() {
            {
              let kind = spec.kind;
              let active = active_file_spec() == kind;
              let cls = if active {
                "px-3 py-1.5 rounded-xl text-xs font-mono font-bold bg-slate-900 text-white dark:bg-white dark:text-slate-900 cursor-pointer transition-all"
              } else {
                "px-3 py-1.5 rounded-xl text-xs font-mono font-medium bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700 cursor-pointer transition-all"
              };
              rsx! {
                button {
                  key: "spec-{kind.id()}",
                  r#type: "button",
                  class: cls,
                  onclick: move |_| active_file_spec.set(kind),
                  "{kind.short_name()}"
                }
              }
            }
          }
        }

        if let Some(spec) = cur_spec {
          {
            let spec_kind = spec.kind;
            rsx! {
              div { class: "grid grid-cols-1 lg:grid-cols-12 gap-5 items-start",
                div { class: "lg:col-span-7 space-y-3.5",
                  div { class: "flex flex-wrap items-center gap-2",
                    h3 { class: "text-base sm:text-lg font-extrabold font-mono text-slate-900 dark:text-white",
                      "{spec.filenames}"
                    }
                    span { class: badge_class(BadgeVariant::Secondary, "text-[11px]"),
                      "{spec.modalities(lang)}"
                    }
                  }
                  div { class: "text-sm font-bold text-indigo-600 dark:text-indigo-400",
                    "{spec.role(lang)}"
                  }
                  p { class: "text-sm text-slate-700 dark:text-slate-300 leading-relaxed",
                    "{spec.summary(lang)}"
                  }
                  div { class: "rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/70 dark:bg-slate-950/60 p-3.5 text-xs space-y-1.5",
                    div { class: "font-bold text-slate-800 dark:text-slate-200",
                      {lang.pick("⚙ 框架与推理引擎加载方式", "⚙ How Frameworks Consume It")}
                    }
                    p { class: "text-slate-600 dark:text-slate-400 leading-relaxed",
                      "{spec.loader(lang)}"
                    }
                  }
                  div { class: "rounded-xl border border-amber-200 dark:border-amber-900/60 bg-amber-50/70 dark:bg-amber-950/30 p-3.5 text-xs space-y-1.5",
                    div { class: "font-bold text-amber-800 dark:text-amber-300",
                      {lang.pick("⚠ 实战最容易踩的坑", "⚠ Common Pitfall")}
                    }
                    p { class: "text-amber-900/90 dark:text-amber-200/90 leading-relaxed",
                      "{spec.pitfall(lang)}"
                    }
                  }
                  div { class: "pt-1",
                    Button {
                      variant: ButtonVariant::Outline,
                      size: ButtonSize::Sm,
                      onclick: move |_| {
                        filter_file.set(Some(spec_kind));
                        filter_group.set(None);
                      },
                      {lang.pick("在下方词典中只看该文件的字段 ↓", "Filter Dictionary to This File ↓")}
                    }
                  }
                }

                div { class: "lg:col-span-5",
                  div { class: "rounded-2xl border border-slate-800 bg-slate-950 overflow-hidden",
                    div { class: "px-3.5 py-2 border-b border-slate-800 flex items-center justify-between text-[11px] font-mono text-slate-400",
                      span { "{spec_kind.short_name()}" }
                      span { "Annotated Example" }
                    }
                    pre { class: "p-3.5 text-xs font-mono text-slate-100 overflow-x-auto leading-relaxed",
                      "{spec.snippet}"
                    }
                  }
                }
              }
            }
          }
        }
      }

      // 4. 交互式高频配置字段速查词典
      div { class: card_class("p-5 sm:p-6 bg-white dark:bg-slate-900/60 space-y-5"),
        div { class: "flex flex-col md:flex-row md:items-end justify-between gap-3",
          div {
            h2 { class: "text-lg sm:text-xl font-extrabold text-slate-900 dark:text-white",
              {lang.pick("3. 交互式配置字段速查词典", "3. Interactive Config Field Dictionary")}
            }
            p { class: "mt-1 text-xs sm:text-sm text-slate-500 dark:text-slate-400",
              {lang.pick(
                "按文件、语义分组或关键词（如 kv / rope / fp8 / eos / patch / lora）即时检索，附显存与维度影响公式、代表真实模型及实验室直达。",
                "Filter by file, semantic group, or keyword (e.g. kv, rope, fp8, eos, patch, lora) with memory/shape formulas and live model links.",
              )}
            }
          }
          div { class: "text-xs font-mono text-slate-500 dark:text-slate-400 shrink-0",
            {lang.pick("当前命中 ", "Showing ")}
            span { class: "font-bold text-indigo-600 dark:text-indigo-400", "{filtered_docs.len()}" }
            " / {payload.field_docs.len()}"
          }
        }

        // 搜索与筛选控件
        div { class: "space-y-3",
          div { class: "flex flex-col sm:flex-row items-stretch sm:items-center gap-2",
            div { class: "flex-1",
              Input {
                class: "font-mono text-xs sm:text-sm bg-slate-50 dark:bg-slate-950",
                placeholder: lang.pick(
                  "搜索字段名、别名或含义（例如 kv_lora_rank、rope、fp8、eos、lora_alpha、patch_size）...",
                  "Search field key, alias, or meaning (e.g. kv_lora_rank, rope, fp8, eos, lora_alpha)...",
                ),
                "aria-label": "Search model config fields",
                value: search_q(),
                on_value_change: move |v: String| search_q.set(v),
              }
            }
            if has_active_filter {
              Button {
                variant: ButtonVariant::Outline,
                size: ButtonSize::Sm,
                onclick: move |_| {
                  filter_file.set(None);
                  filter_group.set(None);
                  search_q.set(String::new());
                },
                {lang.pick("重置筛选", "Reset Filters")}
              }
            }
          }

          // 文件筛选行
          div { class: "flex flex-wrap items-center gap-1.5",
            span { class: "text-[11px] font-semibold text-slate-500 dark:text-slate-400 mr-1",
              {lang.pick("按文件：", "File:")}
            }
            button {
              r#type: "button",
              class: if sel_file.is_none() {
                "px-2.5 py-1 rounded-lg text-[11px] font-bold bg-indigo-600 text-white cursor-pointer"
              } else {
                "px-2.5 py-1 rounded-lg text-[11px] font-medium bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 cursor-pointer"
              },
              onclick: move |_| filter_file.set(None),
              {lang.pick("全部文件", "All Files")}
            }
            for kind in ConfigFileKind::ALL {
              {
                let active = sel_file == Some(kind);
                let cls = if active {
                  "px-2.5 py-1 rounded-lg text-[11px] font-mono font-bold bg-indigo-600 text-white cursor-pointer"
                } else {
                  "px-2.5 py-1 rounded-lg text-[11px] font-mono bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700 cursor-pointer"
                };
                rsx! {
                  button {
                    key: "ff-{kind.id()}",
                    r#type: "button",
                    class: cls,
                    onclick: move |_| filter_file.set(Some(kind)),
                    "{kind.short_name()}"
                  }
                }
              }
            }
          }

          // 语义分组筛选行
          div { class: "flex flex-wrap items-center gap-1.5",
            span { class: "text-[11px] font-semibold text-slate-500 dark:text-slate-400 mr-1",
              {lang.pick("按维度：", "Group:")}
            }
            button {
              r#type: "button",
              class: if sel_group.is_none() {
                "px-2.5 py-1 rounded-lg text-[11px] font-bold bg-slate-900 text-white dark:bg-white dark:text-slate-900 cursor-pointer"
              } else {
                "px-2.5 py-1 rounded-lg text-[11px] font-medium bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 cursor-pointer"
              },
              onclick: move |_| filter_group.set(None),
              {lang.pick("全部语义分组", "All Groups")}
            }
            for grp in FieldGroup::ALL {
              {
                let active = sel_group == Some(grp);
                let cls = if active {
                  "px-2.5 py-1 rounded-lg text-[11px] font-bold bg-slate-900 text-white dark:bg-white dark:text-slate-900 cursor-pointer"
                } else {
                  "px-2.5 py-1 rounded-lg text-[11px] font-medium bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700 cursor-pointer"
                };
                rsx! {
                  button {
                    key: "fg-{grp.id()}",
                    r#type: "button",
                    class: cls,
                    onclick: move |_| filter_group.set(Some(grp)),
                    "{grp.label(lang)}"
                  }
                }
              }
            }
          }
        }

        // 字段结果列表
        if filtered_docs.is_empty() {
          div { class: "py-10 text-center text-sm text-slate-500 dark:text-slate-400",
            {lang.pick(
              "未找到匹配的配置字段，请尝试更简短的关键词或点击「重置筛选」。",
              "No matching config fields found. Try a shorter keyword or reset filters.",
            )}
          }
        } else {
          div { class: "grid grid-cols-1 md:grid-cols-2 gap-3.5",
            for doc in filtered_docs {
              div {
                key: "doc-{doc.key}",
                class: "rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-950/50 p-4 flex flex-col justify-between gap-2.5 hover:border-indigo-300 dark:hover:border-indigo-800 transition-colors",
                div { class: "space-y-2",
                  div { class: "flex flex-wrap items-center justify-between gap-2",
                    code { class: "text-xs sm:text-sm font-mono font-bold text-indigo-700 dark:text-indigo-300 break-all",
                      "{doc.key}"
                    }
                    div { class: "flex flex-wrap items-center gap-1.5",
                      span { class: "px-1.5 py-0.5 rounded bg-slate-200/80 dark:bg-slate-800 text-[10px] font-mono text-slate-700 dark:text-slate-300",
                        "{doc.value_type}"
                      }
                      span { class: badge_class(BadgeVariant::Secondary, "text-[10px] font-mono"),
                        "{doc.file.short_name()}"
                      }
                    }
                  }

                  if !doc.aliases.is_empty() {
                    div { class: "text-[11px] font-mono text-slate-400",
                      {lang.pick("别名：", "Aliases: ")}
                      "{doc.aliases.join(\", \")}"
                    }
                  }

                  p { class: "text-xs sm:text-sm text-slate-700 dark:text-slate-300 leading-relaxed",
                    "{doc.summary(lang)}"
                  }

                  if let Some(ref formula) = doc.formula_impact {
                    div { class: "rounded-lg bg-indigo-50/80 dark:bg-indigo-950/40 border border-indigo-200/70 dark:border-indigo-900/60 px-2.5 py-1.5 text-[11px] font-mono text-indigo-800 dark:text-indigo-300",
                      "ƒ {formula}"
                    }
                  }
                }

                div { class: "flex flex-wrap items-center justify-between gap-2 pt-2 border-t border-slate-200/70 dark:border-slate-800/70 text-[11px]",
                  span { class: "text-slate-500 dark:text-slate-400", "{doc.group.label(lang)}" }
                  div { class: "flex flex-wrap items-center gap-2",
                    if let Some(ref slug) = doc.example_model_slug {
                      Link {
                        to: format!("/models/{slug}"),
                        class: "font-mono font-semibold text-indigo-600 dark:text-indigo-400 hover:underline",
                        "Model: {slug} →"
                      }
                    }
                    if let Some((ref href, ref label)) = doc.lab_link {
                      Link {
                        to: href.clone(),
                        class: "font-semibold text-emerald-600 dark:text-emerald-400 hover:underline",
                        "{label} ↗"
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }

      // 5. 常见工程踩坑速查（Gotchas）
      div { class: card_class("p-5 sm:p-6 bg-white dark:bg-slate-900/60 space-y-5"),
        div {
          h2 { class: "text-lg sm:text-xl font-extrabold text-slate-900 dark:text-white",
            {lang.pick("4. 真实工程高频踩坑速查（Gotchas）", "4. Production Engineering Gotchas")}
          }
          p { class: "mt-1 text-xs sm:text-sm text-slate-500 dark:text-slate-400",
            {lang.pick(
              "写推理引擎、算显存预算或对齐微调权重时最容易因为误读配置文件而翻车的 6 类经典陷阱。",
              "Six classic failure modes when implementing inference engines, sizing VRAM, or merging fine-tuned checkpoints.",
            )}
          }
        }

        div { class: "grid grid-cols-1 lg:grid-cols-2 gap-4",
          for g in payload.gotchas.iter() {
            div {
              key: "gotcha-{g.id}",
              class: "rounded-2xl border border-slate-200 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-950/50 p-4 space-y-2.5",
              div { class: "flex items-center gap-2",
                span { class: "px-2 py-0.5 rounded-md bg-amber-500/15 text-amber-700 dark:text-amber-300 text-[10px] font-bold shrink-0",
                  "{g.badge}"
                }
                h3 { class: "text-sm font-extrabold text-slate-900 dark:text-white",
                  "{g.title(lang)}"
                }
              }
              div { class: "text-xs space-y-1.5 leading-relaxed",
                p { class: "text-slate-600 dark:text-slate-400",
                  span { class: "font-bold text-rose-600 dark:text-rose-400 mr-1",
                    {lang.pick("现象：", "Symptom: ")}
                  }
                  "{g.symptom(lang)}"
                }
                p { class: "text-slate-600 dark:text-slate-400",
                  span { class: "font-bold text-slate-800 dark:text-slate-200 mr-1",
                    {lang.pick("根因：", "Cause: ")}
                  }
                  "{g.root_cause(lang)}"
                }
                p { class: "text-slate-700 dark:text-slate-300",
                  span { class: "font-bold text-emerald-600 dark:text-emerald-400 mr-1",
                    {lang.pick("解法：", "Fix: ")}
                  }
                  "{g.fix(lang)}"
                }
              }
              div { class: "flex flex-wrap items-center gap-1.5 pt-1",
                for rk in g.related_keys.iter() {
                  {
                    let rk_owned = rk.clone();
                    rsx! {
                      button {
                        key: "{g.id}-{rk}",
                        r#type: "button",
                        class: "px-2 py-0.5 rounded bg-slate-200/80 dark:bg-slate-800 text-[11px] font-mono text-indigo-700 dark:text-indigo-300 hover:bg-indigo-100 dark:hover:bg-indigo-950/60 cursor-pointer",
                        onclick: move |_| {
                          filter_file.set(None);
                          filter_group.set(None);
                          search_q.set(rk_owned.clone());
                        },
                        "{rk}"
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
}

#[cfg(all(test, feature = "server"))]
mod tests {
  use std::collections::HashSet;

  use super::*;
  use crate::arch::{build_diffusion_pipeline_blueprint, load_builtin_preset};

  const VALID_MODEL_SLUGS: &[&str] = &[
    "bge-m3",
    "deepseek-r1",
    "deepseek-v2-lite",
    "deepseek-v3",
    "flux-1-dev",
    "gemma-3-27b",
    "glm-4-9b",
    "llama-3-1-8b",
    "llama-3-3-70b",
    "llama-4-scout",
    "mixtral-8x7b",
    "parler-tts-large-v1",
    "qwen2-5-72b",
    "qwen2-5-omni-7b",
    "qwen2-5-vl-72b",
    "qwen3-235b-a22b",
    "qwen3-30b-a3b",
    "qwen3-8b",
    "qwen3-embedding-8b",
    "qwen3-next-80b-a3b",
    "sd-3-5-large",
    "wan2-1-t2v-14b",
    "whisper-large-v3",
  ];

  const VALID_LAB_LINKS: &[&str] = &[
    "/ai/components/attention-mla-and-flash-online-softmax",
    "/ai/components/rope-and-yarn-extrapolation",
    "/ai/components/moe-routing-quant-and-speculative-decoding",
    "/ai/components/activations-gelu-silu-swiglu",
    "/ai/components/rmsnorm-and-adaln-zero",
    "/ai/components/lstm-deltanet-and-mamba-ssm",
    "/ai/components/audio-stft-conformer-ctc-and-rvq",
    "/ai/components/conv-and-depthwise-separable",
    "/ai/math/svd-eckart-young-pseudoinverse",
    "/ai/math/continuity-equation-cfm-theorem",
    "/ai/math/fourier-convolution-aliasing",
    "/ai/math/ieee754-bf16-fp8-numerical-math",
    "/ai/math/computational-graph-vjp-vs-jvp",
    "/ai/math/forward-reverse-kl-and-k3",
  ];

  #[test]
  fn config_guide_dictionary_is_consistent_and_valid() {
    assert_eq!(CONFIG_FILE_SPECS.len(), ConfigFileKind::ALL.len());
    for kind in ConfigFileKind::ALL {
      assert!(
        CONFIG_FILE_SPECS.iter().any(|s| s.kind == kind),
        "missing ConfigFileSpec for {kind:?}"
      );
    }

    assert!(
      CONFIG_FIELD_DOCS.len() >= 80,
      "expected >= 80 field docs, got {}",
      CONFIG_FIELD_DOCS.len()
    );

    let mut seen_keys = HashSet::new();
    for doc in CONFIG_FIELD_DOCS {
      assert!(seen_keys.insert(doc.key), "duplicate config field key: {}", doc.key);
      assert!(!doc.summary_zh.trim().is_empty(), "empty zh summary for {}", doc.key);
      assert!(!doc.summary_en.trim().is_empty(), "empty en summary for {}", doc.key);
      if let Some(slug) = doc.example_model_slug {
        assert!(
          VALID_MODEL_SLUGS.contains(&slug),
          "invalid example_model_slug `{slug}` on field `{}`",
          doc.key
        );
      }
      if let Some((href, label)) = doc.lab_link {
        assert!(
          VALID_LAB_LINKS.contains(&href),
          "invalid lab_link `{href}` on field `{}`",
          doc.key
        );
        assert!(!label.trim().is_empty());
      }
    }

    for tree in REPO_MODALITY_BLUEPRINTS {
      assert!(!tree.items.is_empty());
      if let Some(slug) = tree.example_slug {
        assert!(VALID_MODEL_SLUGS.contains(&slug), "invalid repo tree slug `{slug}`");
      }
    }

    for gotcha in CONFIG_GOTCHAS {
      for k in gotcha.related_keys {
        assert!(
          lookup_field_doc(k).is_some(),
          "gotcha `{}` references unknown key `{k}`",
          gotcha.id
        );
      }
    }

    // 校验紧凑 JSON 序列化与反序列化严格无损
    let encoded = build_config_guide_json();
    let decoded = parse_config_guide_json(&encoded);
    assert_eq!(decoded, build_config_guide_payload());
    assert_eq!(get_or_parse_config_guide(&encoded).field_docs, build_config_field_docs());
  }

  #[test]
  fn annotates_deepseek_v3_mla_moe_fp8_mtp() {
    let bp = load_builtin_preset("deepseek-v3").expect("deepseek-v3 preset");
    let fields = extract_annotated_fields(&bp.raw_config_pretty);
    assert!(!fields.is_empty());

    let find = |p: &str| fields.iter().find(|f| f.path == p);

    let kv_lora = find("kv_lora_rank").expect("kv_lora_rank present");
    assert_eq!(kv_lora.display_value, "512");
    assert_eq!(kv_lora.focus, Some(InspectorFocus::Attention));
    assert!(kv_lora.doc.is_some());

    let experts = find("n_routed_experts").expect("n_routed_experts present");
    assert_eq!(experts.display_value, "256");
    assert_eq!(experts.focus, Some(InspectorFocus::FfnOrMoe));
    assert!(experts.doc.is_some());

    let mtp = find("num_nextn_predict_layers").expect("num_nextn_predict_layers present");
    assert_eq!(mtp.focus, Some(InspectorFocus::MtpHead));
    assert!(mtp.doc.is_some());

    let quant =
      find("quantization_config.quant_method").expect("quantization_config.quant_method present");
    assert_eq!(quant.group, FieldGroup::QuantPrecision);
    assert!(quant.doc.is_some());

    let rope_type = find("rope_scaling.type").expect("rope_scaling.type present");
    assert_eq!(rope_type.group, FieldGroup::PositionContext);
    assert!(rope_type.doc.is_some());
  }

  #[test]
  fn annotates_qwen3_next_hybrid_ssm_config() {
    let raw = r#"{
      "architectures": ["Qwen3NextForCausalLM"],
      "model_type": "qwen3_next",
      "hidden_size": 2048,
      "num_hidden_layers": 48,
      "num_attention_heads": 16,
      "num_key_value_heads": 2,
      "head_dim": 256,
      "partial_rotary_factor": 0.25,
      "full_attention_interval": 4,
      "linear_num_key_heads": 16,
      "linear_key_head_dim": 128,
      "linear_conv_kernel_dim": 4,
      "num_experts": 512,
      "num_experts_per_tok": 10,
      "moe_intermediate_size": 512,
      "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
      "custom_vendor_flag": true
    }"#;
    let fields = extract_annotated_fields(raw);

    for expected in [
      "full_attention_interval",
      "linear_num_key_heads",
      "linear_key_head_dim",
      "linear_conv_kernel_dim",
      "layer_types",
      "partial_rotary_factor",
      "num_experts",
      "num_experts_per_tok",
    ] {
      let item = fields.iter().find(|f| f.path == expected).expect(expected);
      assert!(item.doc.is_some(), "expected dictionary hit for {expected}");
    }

    // 未知扩展字段必须被保留而非丢弃
    let custom = fields.iter().find(|f| f.path == "custom_vendor_flag").expect("custom field kept");
    assert!(custom.doc.is_none());
    assert_eq!(custom.display_value, "true");
  }

  #[test]
  fn annotates_flux_diffusers_combined_json() {
    let model_index = r#"{
      "_class_name": "FluxPipeline",
      "transformer": ["diffusers", "FluxTransformer2DModel"],
      "vae": ["diffusers", "AutoencoderKL"],
      "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"]
    }"#;
    let transformer = r#"{
      "_class_name": "FluxTransformer2DModel",
      "num_layers": 19,
      "num_single_layers": 38,
      "num_attention_heads": 24,
      "attention_head_dim": 128,
      "joint_attention_dim": 4096,
      "pooled_projection_dim": 768,
      "in_channels": 64,
      "guidance_embeds": true,
      "axes_dims_rope": [16, 56, 56]
    }"#;
    let vae = r#"{
      "_class_name": "AutoencoderKL",
      "latent_channels": 16,
      "scaling_factor": 0.3611,
      "shift_factor": 0.1159,
      "block_out_channels": [128, 256, 512, 512]
    }"#;
    let scheduler = r#"{
      "_class_name": "FlowMatchEulerDiscreteScheduler",
      "num_train_timesteps": 1000,
      "shift": 3.0,
      "use_dynamic_shifting": true
    }"#;

    let bp = build_diffusion_pipeline_blueprint(
      "black-forest-labs/FLUX.1-dev",
      "test",
      model_index,
      &[
        ("transformer".to_string(), transformer.to_string()),
        ("vae".to_string(), vae.to_string()),
        ("scheduler".to_string(), scheduler.to_string()),
      ],
      None,
    )
    .expect("flux blueprint");

    let fields = extract_annotated_fields(&bp.raw_config_pretty);
    let single_layers = fields
      .iter()
      .find(|f| {
        f.path == "num_single_layers" && f.source_slot.as_deref() == Some("transformer/config.json")
      })
      .expect("num_single_layers in transformer slot");
    assert!(single_layers.doc.is_some());
    assert_eq!(single_layers.display_value, "38");

    let shift = fields
      .iter()
      .find(|f| f.path == "shift" && f.source_slot.as_deref() == Some("scheduler/config.json"))
      .expect("shift in scheduler slot");
    assert!(shift.doc.is_some());

    let scale = fields
      .iter()
      .find(|f| f.path == "scaling_factor" && f.source_slot.as_deref() == Some("vae/config.json"))
      .expect("scaling_factor in vae slot");
    assert!(scale.doc.is_some());
    assert_eq!(scale.focus, Some(InspectorFocus::LmHead));
  }

  #[test]
  fn annotates_bge_m3_embedding_pooling_and_compacts_large_aux() {
    let bge_cfg = r#"{
      "architectures": ["XLMRobertaModel"],
      "model_type": "xlm-roberta",
      "hidden_size": 1024,
      "num_hidden_layers": 24,
      "num_attention_heads": 16,
      "intermediate_size": 4096,
      "vocab_size": 250002,
      "type_vocab_size": 1,
      "max_position_embeddings": 8194
    }"#;
    let pooling_cfg = r#"{
      "word_embedding_dimension": 1024,
      "pooling_mode_cls_token": true,
      "pooling_mode_mean_tokens": false
    }"#;
    let gen_cfg = r#"{
      "do_sample": true,
      "temperature": 0.6,
      "top_p": 0.95,
      "eos_token_id": [151645, 151643]
    }"#;

    let main_fields = extract_annotated_fields(bge_cfg);
    assert!(main_fields.iter().any(|f| f.path == "type_vocab_size" && f.doc.is_some()));

    let pool_fields =
      extract_annotated_fields_for_file(pooling_cfg, ConfigFileKind::EmbeddingModules);
    assert!(pool_fields.iter().any(|f| f.path == "word_embedding_dimension" && f.doc.is_some()));
    assert!(pool_fields.iter().any(|f| f.path == "pooling_mode_cls_token" && f.doc.is_some()));

    let gen_fields = extract_annotated_fields_for_file(gen_cfg, ConfigFileKind::GenerationConfig);
    let temp = gen_fields.iter().find(|f| f.path == "temperature").expect("temperature");
    assert_eq!(temp.group, FieldGroup::TokenizerGeneration);
    assert!(temp.doc.is_some());

    // 测试超大 tokenizer_config 裁剪不会膨胀
    let mut big_decoder = serde_json::Map::new();
    for i in 0..30 {
      big_decoder.insert(i.to_string(), Value::String(format!("<tok_{i}>")));
    }
    let big_tok = serde_json::json!({
      "tokenizer_class": "Qwen2Tokenizer",
      "added_tokens_decoder": big_decoder,
      "chat_template": "x".repeat(500)
    })
    .to_string();
    let compacted = compact_aux_json_for_snapshot(&big_tok).expect("compacted");
    assert!(compacted.contains("{30 entries}"));
    assert!(compacted.contains("500 chars"));
  }
}
