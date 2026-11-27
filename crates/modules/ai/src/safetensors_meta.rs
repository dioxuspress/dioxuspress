//! L2 Safetensors 元数据解析器（`model.safetensors.index.json` + `.safetensors` 二进制文件头）。
//!
//! 设计目标：
//! - **零权重下载**：只需拉取几 KB ~ 几百 KB 的 `index.json` 或单分片前 `8 + header_len` 字节，
//!   即可提取真实张量名称、精确 Shape、真实 Dtype（`BF16` / `F8_E4M3` / `I4` / `U8`）与磁盘字节数。
//! - **MoE 大模型压缩折叠**：像 DeepSeek-V3 的 `index.json` 含约 9 万条张量记录（61 层 × 256 专家），
//!   解析器在统计完精确总数后，会将同构重复的 `layers.<L>` 与 `experts.<E>` 折叠为带 `repeat_count`
//!   的模式条目，使跨端传输体积保持在数 KB，同时完整保留真实 Shape、Dtype 与分片定位。

#[cfg(feature = "server")]
use std::collections::{BTreeMap, BTreeSet, HashMap};

use serde::{Deserialize, Serialize};
#[cfg(feature = "server")]
use serde_json::Value;

/// 单分片 Safetensors JSON Header 允许的最大字节数（8 MiB，防止畸形文件触发大内存分配）。
pub const MAX_SAFETENSORS_HEADER_BYTES: usize = 8 * 1024 * 1024;

/// 折叠后的单类（或单个）真实权重张量描述。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TensorPatternEntry {
  /// 规范化张量路径模式，例如 `model.layers.[3..60].mlp.experts.[0..255].gate_proj.weight`
  pub pattern: String,
  /// 代表性真实张量名（便于复制对照源码），例如 `model.layers.3.mlp.experts.0.gate_proj.weight`
  pub example_name: String,
  /// 所属架构阶段分类：`embedding` | `attention` | `ffn` | `moe` | `norm` | `lm_head` | `mtp` | `other`
  pub stage: String,
  /// 该模式匹配的真实张量个数（普通单张量为 1，MoE 跨层专家可达 `58 * 256 = 14848`）
  pub repeat_count: usize,
  /// 从 Safetensors Header 采样解析出的单个张量精确维度（仅读 `index.json` 未采样命中时为 `None`）
  pub shape: Option<Vec<u64>>,
  /// 真实存储精度，例如 `"BF16"`、`"F8_E4M3"`、`"F16"`、`"I32"`
  pub dtype: Option<String>,
  /// 单个张量在文件中的真实字节长度（`data_offsets[1] - data_offsets[0]`）
  pub single_byte_len: Option<u64>,
  /// 代表性分片文件名（多文件时）
  pub example_shard: Option<String>,
}

impl TensorPatternEntry {
  /// 单个张量的元素个数（若已知真实 Shape）。
  pub fn single_elements(&self) -> Option<u64> {
    let shape = self.shape.as_ref()?;
    shape.iter().try_fold(1u64, |acc, &d| acc.checked_mul(d))
  }

  /// 该模式覆盖的总元素个数（`single_elements * repeat_count`）。
  pub fn total_elements(&self) -> Option<u64> {
    self.single_elements()?.checked_mul(self.repeat_count as u64)
  }
}

/// 聚合后的 L2 Safetensors 真实权重摘要。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct SafetensorsSummary {
  /// 来源描述，如 `"model.safetensors.index.json + 3 shard headers"` 或 `"model.safetensors header"`
  pub source_kind: String,
  /// 模型总字节数（来自 `index.json` 的 `metadata.total_size` 或单文件偏移汇总）
  pub total_bytes: Option<u64>,
  /// 分片总数（单文件为 1）
  pub shard_count: usize,
  /// 原始张量总条数（未折叠前）
  pub total_tensor_count: usize,
  /// 成功解析出真实 Shape/Dtype 的张量条数
  pub header_inspected_count: usize,
  /// 检测到的 Transformer 主干层数（从 `layers.<idx>` 最大下标推导）
  pub detected_num_layers: Option<u64>,
  /// 检测到的单层路由专家数（从 `experts.<idx>` 最大下标推导）
  pub detected_num_experts: Option<u64>,
  /// 检测到的首个 MoE 层下标（区分前几层 Dense 与后续 MoE）
  pub detected_first_moe_layer: Option<u64>,
  /// 真实权重中是否存在 Attention 投影偏置（`*_proj.bias`）
  pub has_attn_bias: bool,
  /// 真实权重中是否存在 QK-Norm（`q_norm` / `k_norm`）
  pub has_qk_norm: bool,
  /// 各种 Dtype 的张量数量分布（按数量降序）
  pub dtype_histogram: Vec<(String, usize)>,
  /// 基于 L2 采样张量真实 Shape（含常见 4-bit 打包解压、排除 scale/zero-point）估算的全模型总参数量（仅服务端交叉核验）
  #[cfg(feature = "server")]
  #[serde(skip)]
  pub l2_estimated_params: Option<u64>,
  /// L2 从真实张量名检测到的线性注意力 / SSM 层数量（仅服务端交叉核验）
  #[cfg(feature = "server")]
  #[serde(skip)]
  pub detected_linear_attn_layers: Option<u64>,
  /// L2 从真实张量名检测到的标准/MLA 注意力层数量（仅服务端交叉核验）
  #[cfg(feature = "server")]
  #[serde(skip)]
  pub detected_full_attn_layers: Option<u64>,
  /// L2 真实权重中是否存在视觉编码器或跨模态投影层（仅服务端交叉核验）
  #[cfg(feature = "server")]
  #[serde(skip)]
  pub has_vision_tower: bool,
  /// L2 真实权重中是否存在潜空间 MoE 上下投影（仅服务端交叉核验）
  #[cfg(feature = "server")]
  #[serde(skip)]
  pub has_latent_moe_proj: bool,
  /// 折叠后的结构化权重模式列表（可直接用于架构节点联动与权重树浏览）
  pub patterns: Vec<TensorPatternEntry>,
}

#[cfg(feature = "server")]
mod parser {
  use super::*;

  /// 原始单个张量的头信息（内部解析态）。
  #[derive(Debug, Clone, PartialEq, Eq)]
  pub struct RawTensorMeta {
    pub name: String,
    pub shape: Option<Vec<u64>>,
    pub dtype: Option<String>,
    pub byte_len: Option<u64>,
    pub shard: Option<String>,
  }

  #[derive(Debug, Deserialize)]
  struct RawSafetensorsIndex {
    #[serde(default)]
    metadata: HashMap<String, Value>,
    weight_map: HashMap<String, String>,
  }

  /// 解析 `.safetensors` 文件前 8 个字节（小端 `u64`）得到 JSON Header 字节长度。
  pub fn parse_safetensors_header_len(first_8_bytes: &[u8]) -> Result<usize, String> {
    if first_8_bytes.len() < 8 {
      return Err(format!("Safetensors 文件头不足 8 字节（实际 {} 字节）", first_8_bytes.len()));
    }
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&first_8_bytes[..8]);
    let raw_len = u64::from_le_bytes(buf);
    if raw_len == 0 {
      return Err("Safetensors JSON Header 长度为 0".to_string());
    }
    let len = usize::try_from(raw_len).map_err(|_| "Safetensors Header 长度溢出".to_string())?;
    if len > MAX_SAFETENSORS_HEADER_BYTES {
      return Err(format!(
        "Safetensors Header 长度 ({len} B) 超过安全上限 ({MAX_SAFETENSORS_HEADER_BYTES} B)"
      ));
    }
    Ok(len)
  }

  /// 解析完整的（或包含前 8 字节 + JSON Header 的）Safetensors 头部切片。
  pub fn parse_safetensors_header_bytes(
    bytes: &[u8],
    shard_name: Option<&str>,
  ) -> Result<Vec<RawTensorMeta>, String> {
    let header_len = parse_safetensors_header_len(bytes)?;
    let end = 8usize.saturating_add(header_len);
    if bytes.len() < end {
      return Err(format!(
        "Safetensors Header 数据不完整：需要 {end} 字节，实际 {} 字节",
        bytes.len()
      ));
    }
    parse_safetensors_header_json(&bytes[8..end], shard_name)
  }

  /// 解析 Safetensors 的 JSON Header 主体（不含前 8 字节长度前缀）。
  pub fn parse_safetensors_header_json(
    json_bytes: &[u8],
    shard_name: Option<&str>,
  ) -> Result<Vec<RawTensorMeta>, String> {
    let obj: BTreeMap<String, Value> = serde_json::from_slice(json_bytes)
      .map_err(|e| format!("解析 Safetensors Header JSON 失败: {e}"))?;

    let mut out = Vec::with_capacity(obj.len());
    for (name, val) in obj {
      if name == "__metadata__" {
        continue;
      }
      let Some(map) = val.as_object() else {
        continue;
      };
      let dtype = map.get("dtype").and_then(Value::as_str).map(str::to_string);
      let shape = map
        .get("shape")
        .and_then(Value::as_array)
        .map(|arr| arr.iter().filter_map(Value::as_u64).collect::<Vec<u64>>());
      let byte_len = map.get("data_offsets").and_then(Value::as_array).and_then(|offs| {
        if offs.len() == 2 {
          let start = offs[0].as_u64()?;
          let finish = offs[1].as_u64()?;
          finish.checked_sub(start)
        } else {
          None
        }
      });
      out.push(RawTensorMeta {
        name,
        shape,
        dtype,
        byte_len,
        shard: shard_name.map(str::to_string),
      });
    }
    Ok(out)
  }

  /// 解析 `model.safetensors.index.json`，返回 `(total_size, weight_map)`。
  pub fn parse_safetensors_index_json(
    raw_json: &str,
  ) -> Result<(Option<u64>, HashMap<String, String>), String> {
    let parsed: RawSafetensorsIndex = serde_json::from_str(raw_json)
      .map_err(|e| format!("解析 model.safetensors.index.json 失败: {e}"))?;
    let total_size = parsed
      .metadata
      .get("total_size")
      .and_then(|v| v.as_u64().or_else(|| v.as_str().and_then(|s| s.parse::<u64>().ok())));
    Ok((total_size, parsed.weight_map))
  }

  /// 从 `weight_map` 中挑选最多 3 个最具代表性的分片文件名用于 HTTP Range 采样：
  /// 1. 含 `embed_tokens` 或 `layers.0.` 的首分片（覆盖词表嵌入、第 0 层 Attention 与 Dense FFN）；
  /// 2. 含首个 MoE 专家（`.experts.0.` 或 `.experts.`）的分片（覆盖路由专家真实精度与维度）；
  /// 3. 含 `lm_head` 或末尾 `norm` / MTP 层的尾分片。
  pub fn select_representative_shards(weight_map: &HashMap<String, String>) -> Vec<String> {
    fn push_unique(picked: &mut Vec<String>, shard: &str) {
      if is_safe_shard_filename(shard) && !picked.iter().any(|s| s == shard) {
        picked.push(shard.to_string());
      }
    }

    let mut picked: Vec<String> = Vec::new();

    // 稳定排序键，确保测试与线上挑选结果确定。
    let mut entries: Vec<(&str, &str)> =
      weight_map.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
    entries.sort_by_key(|(k, _)| *k);

    if let Some((_, s)) = entries.iter().find(|(k, _)| {
      k.contains("embed_tokens")
        || k.contains("layers.0.self_attn")
        || k.contains("transformer_blocks.0.")
        || k.contains("blocks.0.")
        || k.contains("down_blocks.0.")
    }) {
      push_unique(&mut picked, s);
    }
    if let Some((_, s)) = entries.iter().find(|(k, _)| {
      k.contains(".experts.0.")
        || k.contains(".experts.")
        || k.contains("single_transformer_blocks.0.")
        || k.contains("mid_block.")
    }) {
      push_unique(&mut picked, s);
    }
    if let Some((_, s)) = entries
      .iter()
      .rfind(|(k, _)| k.contains("lm_head") || k.contains("proj_out") || k.contains("norm"))
    {
      push_unique(&mut picked, s);
    }
    if picked.is_empty() {
      if let Some((_, s)) = entries.first() {
        push_unique(&mut picked, s);
      }
    }
    picked.truncate(3);
    picked
  }

  /// 校验分片文件名仅由安全字符组成（禁止路径穿越 `..` 或 `/`）。
  pub fn is_safe_shard_filename(name: &str) -> bool {
    !name.is_empty()
      && name.len() <= 160
      && !name.contains("..")
      && !name.contains('/')
      && !name.contains('\\')
      && name.ends_with(".safetensors")
      && name.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
  }

  /// 由单文件 Safetensors Header 列表直接构建 [`SafetensorsSummary`]。
  pub fn summarize_single_header(tensors: Vec<RawTensorMeta>) -> SafetensorsSummary {
    let total_bytes: Option<u64> =
      tensors.iter().try_fold(0u64, |acc, t| t.byte_len.and_then(|b| acc.checked_add(b)));
    build_summary("model.safetensors header".to_string(), total_bytes, 1, tensors)
  }

  /// 合并 `model.safetensors.index.json` 的全量 `weight_map` 与若干采样分片 Header，构建 [`SafetensorsSummary`]。
  pub fn summarize_index_and_headers(
    total_bytes_from_index: Option<u64>,
    weight_map: &HashMap<String, String>,
    sampled_headers: &[RawTensorMeta],
  ) -> SafetensorsSummary {
    let shards: BTreeSet<&str> = weight_map.values().map(String::as_str).collect();
    let shard_count = shards.len().max(1);

    let mut by_name: HashMap<&str, &RawTensorMeta> = HashMap::with_capacity(sampled_headers.len());
    for item in sampled_headers {
      by_name.insert(item.name.as_str(), item);
    }

    // 同时按「去掉具体层号与专家号后的骨架键」建立后备表：
    // 即使我们只采样了第 3 层的 MoE 分片，第 4..60 层的同构权重也能继承完全一致的 Shape 与 Dtype！
    let mut by_skeleton: HashMap<String, &RawTensorMeta> = HashMap::new();
    for item in sampled_headers {
      let skel = canonical_skeleton(&item.name).skeleton;
      by_skeleton.entry(skel).or_insert(item);
    }

    let mut merged: Vec<RawTensorMeta> = Vec::with_capacity(weight_map.len());
    let mut keys: Vec<(&String, &String)> = weight_map.iter().collect();
    keys.sort_by_key(|(k, _)| k.as_str());

    for (name, shard) in keys {
      if let Some(exact) = by_name.get(name.as_str()) {
        merged.push(RawTensorMeta {
          name: name.clone(),
          shape: exact.shape.clone(),
          dtype: exact.dtype.clone(),
          byte_len: exact.byte_len,
          shard: Some(shard.clone()),
        });
      } else {
        let skel = canonical_skeleton(name).skeleton;
        let inferred = by_skeleton.get(&skel).copied();
        merged.push(RawTensorMeta {
          name: name.clone(),
          shape: inferred.and_then(|m| m.shape.clone()),
          dtype: inferred.and_then(|m| m.dtype.clone()),
          byte_len: inferred.and_then(|m| m.byte_len),
          shard: Some(shard.clone()),
        });
      }
    }

    let source_kind = if sampled_headers.is_empty() {
      format!("model.safetensors.index.json ({shard_count} shards)")
    } else {
      format!("model.safetensors.index.json + sampled headers ({shard_count} shards)")
    };

    build_summary(source_kind, total_bytes_from_index, shard_count, merged)
  }

  struct SkeletonParts {
    skeleton: String,
    layer_idx: Option<u64>,
    expert_idx: Option<u64>,
  }

  /// 提取张量名中的 `layers.<L>` 与 `experts.<E>` 索引并生成骨架键。
  fn canonical_skeleton(name: &str) -> SkeletonParts {
    let parts: Vec<&str> = name.split('.').collect();
    let mut out: Vec<&str> = Vec::with_capacity(parts.len());
    let mut layer_idx = None;
    let mut expert_idx = None;

    let mut i = 0;
    while i < parts.len() {
      let seg = parts[i];
      out.push(seg);
      if (matches!(seg, "layers" | "layer" | "h" | "block" | "blocks") || seg.ends_with("_blocks"))
        && i + 1 < parts.len()
      {
        if let Ok(idx) = parts[i + 1].parse::<u64>() {
          layer_idx = Some(idx);
          out.push("<L>");
          i += 2;
          continue;
        }
      }
      if seg == "experts" && i + 1 < parts.len() {
        if let Ok(idx) = parts[i + 1].parse::<u64>() {
          expert_idx = Some(idx);
          out.push("<E>");
          i += 2;
          continue;
        }
      }
      i += 1;
    }

    SkeletonParts { skeleton: out.join("."), layer_idx, expert_idx }
  }

  /// 根据张量路径判断其所属模块阶段，供架构图点击联动过滤使用。
  pub fn classify_tensor_stage(
    name: &str,
    max_main_layer: Option<u64>,
    layer_idx: Option<u64>,
  ) -> &'static str {
    if let (Some(max_l), Some(l)) = (max_main_layer, layer_idx) {
      if l >= max_l
        && (name.contains("eh_proj")
          || name.contains("enorm")
          || name.contains("hnorm")
          || name.contains("shared_head"))
      {
        return "mtp";
      }
    }
    if name.contains("eh_proj")
      || name.contains("enorm")
      || name.contains("hnorm")
      || name.contains("shared_head")
    {
      return "mtp";
    }
    if name.starts_with("visual.")
      || name.contains("vision_tower")
      || name.contains("vision_model")
      || name.contains("mm_projector")
      || name.contains("multi_modal_projector")
    {
      return "vision";
    }
    if name.starts_with("audio_tower")
      || name.contains("audio_encoder")
      || name.starts_with("talker.")
      || name.starts_with("token2wav")
    {
      return "audio";
    }
    if name.contains("embed_tokens")
      || name.contains("wte")
      || name.contains("word_embeddings")
      || name.contains("embed_positions")
      || name.starts_with("time_text_embed.")
      || name.starts_with("condition_embedder.")
      || name.starts_with("context_embedder.")
      || name.starts_with("x_embedder.")
      || name.starts_with("patch_embedding.")
      || name.starts_with("pos_embed.")
    {
      "embedding"
    } else if name.contains("lm_head")
      || name.contains("output.weight")
      || name.contains("proj_out")
      || name.contains("conv_out.")
      || name.contains("pooler.")
      || name.contains("classifier.")
      || name.ends_with("score.weight")
    {
      "lm_head"
    } else if name.contains("self_attn")
      || name.contains("encoder_attn")
      || name.contains(".attn.")
      || name.contains(".attn1.")
      || name.contains(".attn2.")
      || name.contains("attention")
      || name.contains("linear_attn")
      || name.contains(".mamba.")
      || name.contains("mixer.")
      || name.contains("time_mix")
    {
      "attention"
    } else if name.contains("experts")
      || name.contains("shared_expert")
      || name.contains("shared_experts")
      || name.contains("routed_expert_")
      || name.contains("block_sparse_moe.gate")
      || name.contains("mlp.gate.weight")
      || name.contains("e_score_correction_bias")
    {
      "moe"
    } else if name.contains(".mlp.")
      || name.contains("proj_mlp")
      || name.contains(".ff.")
      || name.contains(".ff_context.")
      || name.contains("resnets.")
      || name.contains("feed_forward")
      || name.contains("ffn")
      || name.contains("fc1.")
      || name.contains("fc2.")
      || name.contains("intermediate.dense")
      || name.contains("channel_mix")
    {
      "ffn"
    } else if name.contains("layernorm")
      || name.contains("layer_norm")
      || name.contains("LayerNorm")
      || name.contains(".norm.")
      || name.contains(".norm1.")
      || name.contains(".norm2.")
      || name.contains("norm_q.")
      || name.contains("norm_k.")
      || name.contains("norm_out.")
      || name.ends_with("norm.weight")
    {
      "norm"
    } else {
      "other"
    }
  }

  fn format_index_set(indices: &BTreeSet<u64>) -> String {
    let Some(&first) = indices.first() else {
      return "*".to_string();
    };
    let Some(&last) = indices.last() else {
      return first.to_string();
    };
    if indices.len() == 1 {
      first.to_string()
    } else if (last - first + 1) as usize == indices.len() {
      format!("[{first}..{last}]")
    } else {
      format!("[{first}..{last} ({}x)]", indices.len())
    }
  }

  /// 判断张量是否为量化辅助元数据（scale / zero-point / g_idx），不应重复计入有效模型参数量。
  fn is_quant_aux_tensor(name: &str) -> bool {
    name.ends_with(".weight_scale")
      || name.ends_with(".weight_scale_inv")
      || name.ends_with(".input_scale")
      || name.ends_with(".scales")
      || name.ends_with(".qzeros")
      || name.ends_with(".g_idx")
      || name.ends_with(".absmax")
      || name.ends_with(".quant_map")
      || name.ends_with(".zero_point")
  }

  /// 计算单个模式覆盖的逻辑参数个数（自动识别 4-bit 打包张量，跳过量化 scale/zero 辅助张量）。
  fn effective_pattern_params(p: &TensorPatternEntry) -> Option<u64> {
    if is_quant_aux_tensor(&p.example_name) {
      return Some(0);
    }
    let raw = p.total_elements()?;
    let is_packed =
      p.example_name.ends_with(".weight_packed") || p.example_name.ends_with(".qweight");
    let mul = if is_packed {
      match p.dtype.as_deref() {
        Some("U8" | "I8") => 2,
        Some("U16" | "I16") => 4,
        Some("U32" | "I32") => 8,
        _ => 1,
      }
    } else {
      1
    };
    raw.checked_mul(mul)
  }

  /// 当采样 Shape 覆盖了绝大部分真实张量时，计算基于 L2 真实维度的全模型参数总量估算。
  fn estimate_l2_params(patterns: &[TensorPatternEntry], total_tensor_count: usize) -> Option<u64> {
    if total_tensor_count == 0 || patterns.is_empty() {
      return None;
    }
    let mut covered_tensors = 0usize;
    let mut total_params = 0u64;
    for p in patterns {
      if is_quant_aux_tensor(&p.example_name) {
        covered_tensors += p.repeat_count;
        continue;
      }
      if let Some(params) = effective_pattern_params(p) {
        covered_tensors += p.repeat_count;
        total_params = total_params.saturating_add(params);
      }
    }
    // 只有当至少 85% 的张量条数被真实 Shape 覆盖（例如代表性专家与主干层都命中采样）时才给出 L2 总参估算，避免漏采分片导致误报。
    if total_params > 0 && covered_tensors * 100 >= total_tensor_count * 85 {
      Some(total_params)
    } else {
      None
    }
  }

  fn is_linear_attn_signal(name: &str) -> bool {
    name.contains("conv1d")
      || name.contains("A_log")
      || name.contains("dt_bias")
      || name.contains("linear_attn")
      || name.contains(".mamba.")
      || name.contains("time_mix")
      || name.contains("f_a_proj")
      || name.contains("in_proj_qkvz")
  }

  fn is_full_attn_signal(name: &str) -> bool {
    (name.contains("self_attn") || name.contains(".attn."))
      && (name.contains("kv_a_proj")
        || name.contains("q_proj")
        || name.contains("k_proj")
        || name.contains("qkv_proj")
        || name.contains("wqkv")
        || name.contains(".indexer."))
  }

  fn is_vision_signal(name: &str) -> bool {
    name.starts_with("visual.")
      || name.contains("vision_tower")
      || name.contains("vision_model")
      || name.contains("mm_projector")
      || name.contains("multi_modal_projector")
  }

  /// 流式扫描 `model.safetensors.index.json` 后的轻量骨架汇总（避免为 50 万条张量构建全量 HashMap）。
  pub struct StreamingIndexScan {
    total_bytes: Option<u64>,
    shard_count: usize,
    total_tensor_count: usize,
    max_layer: Option<u64>,
    max_expert: Option<u64>,
    first_moe_layer: Option<u64>,
    has_attn_bias: bool,
    has_qk_norm: bool,
    has_vision_tower: bool,
    has_latent_moe_proj: bool,
    linear_attn_layers: BTreeSet<u64>,
    full_attn_layers: BTreeSet<u64>,
    mtp_layers: BTreeSet<u64>,
    representative_shards: Vec<String>,
    groups: BTreeMap<String, StreamingGroupAcc>,
  }

  struct StreamingGroupAcc {
    skeleton: String,
    example_name: String,
    count: usize,
    layers: BTreeSet<u64>,
    experts: BTreeSet<u64>,
    shard: String,
    is_mtp_signal: bool,
  }

  impl StreamingIndexScan {
    pub fn sample_shards(&self) -> &[String] {
      &self.representative_shards
    }
  }

  /// 使用自定义 `serde::de::Visitor` 流式扫描 `model.safetensors.index.json`：
  /// 边解析 `weight_map` 边按骨架折叠，即使面对 50 万+ 条张量的 3T 模型也仅占用 < 1 MiB 结构体内存。
  pub fn parse_safetensors_index_streaming(raw_json: &str) -> Result<StreamingIndexScan, String> {
    use serde::de::{DeserializeSeed, Deserializer, IgnoredAny, MapAccess, Visitor};
    use std::fmt;

    struct ScanState {
      total_bytes: Option<u64>,
      shards: BTreeSet<String>,
      total_tensor_count: usize,
      max_layer_non_mtp: Option<u64>,
      max_expert: Option<u64>,
      first_moe_layer: Option<u64>,
      has_attn_bias: bool,
      has_qk_norm: bool,
      has_vision_tower: bool,
      has_latent_moe_proj: bool,
      linear_attn_layers: BTreeSet<u64>,
      full_attn_layers: BTreeSet<u64>,
      mtp_layers: BTreeSet<u64>,
      first_trunk_shard: Option<String>,
      hetero_attn_shard: Option<String>,
      first_moe_shard: Option<String>,
      vision_shard: Option<String>,
      tail_shard: Option<String>,
      fallback_shard: Option<String>,
      groups: BTreeMap<String, StreamingGroupAcc>,
    }

    impl ScanState {
      fn record(&mut self, name: String, shard: String) {
        self.total_tensor_count += 1;
        if !self.shards.contains(&shard) {
          self.shards.insert(shard.clone());
        }
        if self.fallback_shard.is_none() && is_safe_shard_filename(&shard) {
          self.fallback_shard = Some(shard.clone());
        }
        if is_safe_shard_filename(&shard) {
          if self.first_trunk_shard.is_none()
            && !is_vision_signal(&name)
            && (name.contains("embed_tokens")
              || name.contains("layers.0.self_attn")
              || name.contains("transformer_blocks.0.")
              || name.contains("blocks.0."))
          {
            self.first_trunk_shard = Some(shard.clone());
          }
          if self.hetero_attn_shard.is_none()
            && (name.contains("kv_a_proj") || name.contains(".indexer."))
            && self.first_trunk_shard.as_deref() != Some(shard.as_str())
          {
            self.hetero_attn_shard = Some(shard.clone());
          }
          if self.first_moe_shard.is_none()
            && (name.contains(".experts.0.") || name.contains(".experts."))
          {
            self.first_moe_shard = Some(shard.clone());
          }
          if self.vision_shard.is_none() && is_vision_signal(&name) {
            self.vision_shard = Some(shard.clone());
          }
          if name.contains("lm_head") || name.contains("proj_out") || name.ends_with(".norm.weight")
          {
            self.tail_shard = Some(shard.clone());
          }
        }

        if (name.contains("self_attn") || name.contains(".attn.")) && name.ends_with(".bias") {
          self.has_attn_bias = true;
        }
        if name.contains("q_norm.") || name.contains("k_norm.") {
          self.has_qk_norm = true;
        }
        if is_vision_signal(&name) {
          self.has_vision_tower = true;
        }
        if name.contains("routed_expert_down_proj") || name.contains("routed_expert_up_proj") {
          self.has_latent_moe_proj = true;
        }

        let sp = canonical_skeleton(&name);
        let is_mtp_sig = name.contains("eh_proj")
          || name.contains("enorm")
          || name.contains("hnorm")
          || name.contains("shared_head");
        if is_mtp_sig {
          if let Some(l) = sp.layer_idx {
            self.mtp_layers.insert(l);
          }
        }
        if let Some(l) = sp.layer_idx {
          if !is_mtp_sig && !is_vision_signal(&name) {
            self.max_layer_non_mtp = Some(self.max_layer_non_mtp.map_or(l, |cur| cur.max(l)));
            if name.contains(".experts.")
              || name.contains("mlp.gate.weight")
              || name.contains("block_sparse_moe.gate.weight")
            {
              self.first_moe_layer = Some(self.first_moe_layer.map_or(l, |cur| cur.min(l)));
            }
            if is_linear_attn_signal(&name) {
              self.linear_attn_layers.insert(l);
            } else if is_full_attn_signal(&name) {
              self.full_attn_layers.insert(l);
            }
          }
        }
        if let Some(e) = sp.expert_idx {
          self.max_expert = Some(self.max_expert.map_or(e, |cur| cur.max(e)));
        }

        let entry = self.groups.entry(sp.skeleton.clone()).or_insert_with(|| StreamingGroupAcc {
          skeleton: sp.skeleton,
          example_name: name,
          count: 0,
          layers: BTreeSet::new(),
          experts: BTreeSet::new(),
          shard,
          is_mtp_signal: is_mtp_sig,
        });
        entry.count += 1;
        if let Some(l) = sp.layer_idx {
          entry.layers.insert(l);
        }
        if let Some(e) = sp.expert_idx {
          entry.experts.insert(e);
        }
      }
    }

    struct WeightMapSeed<'a>(&'a mut ScanState);
    impl<'de> DeserializeSeed<'de> for WeightMapSeed<'_> {
      type Value = ();
      fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<(), D::Error> {
        struct MapVis<'a>(&'a mut ScanState);
        impl<'de> Visitor<'de> for MapVis<'_> {
          type Value = ();
          fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("a weight_map JSON object")
          }
          fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<(), A::Error> {
            while let Some((k, v)) = map.next_entry::<String, String>()? {
              self.0.record(k, v);
            }
            Ok(())
          }
        }
        deserializer.deserialize_map(MapVis(self.0))
      }
    }

    struct MetaSeed<'a>(&'a mut Option<u64>);
    impl<'de> DeserializeSeed<'de> for MetaSeed<'_> {
      type Value = ();
      fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<(), D::Error> {
        struct MetaVis<'a>(&'a mut Option<u64>);
        impl<'de> Visitor<'de> for MetaVis<'_> {
          type Value = ();
          fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("a metadata JSON object")
          }
          fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<(), A::Error> {
            while let Some(k) = map.next_key::<String>()? {
              if k == "total_size" {
                let val: Value = map.next_value()?;
                *self.0 = val.as_u64().or_else(|| val.as_str().and_then(|s| s.parse::<u64>().ok()));
              } else {
                let _ = map.next_value::<IgnoredAny>()?;
              }
            }
            Ok(())
          }
        }
        deserializer.deserialize_map(MetaVis(self.0))
      }
    }

    struct RootVis(ScanState);
    impl<'de> Visitor<'de> for RootVis {
      type Value = ScanState;
      fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a safetensors index JSON object")
      }
      fn visit_map<A: MapAccess<'de>>(mut self, mut map: A) -> Result<ScanState, A::Error> {
        while let Some(k) = map.next_key::<String>()? {
          if k == "metadata" {
            map.next_value_seed(MetaSeed(&mut self.0.total_bytes))?;
          } else if k == "weight_map" {
            map.next_value_seed(WeightMapSeed(&mut self.0))?;
          } else {
            let _ = map.next_value::<IgnoredAny>()?;
          }
        }
        Ok(self.0)
      }
    }

    let state = ScanState {
      total_bytes: None,
      shards: BTreeSet::new(),
      total_tensor_count: 0,
      max_layer_non_mtp: None,
      max_expert: None,
      first_moe_layer: None,
      has_attn_bias: false,
      has_qk_norm: false,
      has_vision_tower: false,
      has_latent_moe_proj: false,
      linear_attn_layers: BTreeSet::new(),
      full_attn_layers: BTreeSet::new(),
      mtp_layers: BTreeSet::new(),
      first_trunk_shard: None,
      hetero_attn_shard: None,
      first_moe_shard: None,
      vision_shard: None,
      tail_shard: None,
      fallback_shard: None,
      groups: BTreeMap::new(),
    };

    let mut de = serde_json::Deserializer::from_str(raw_json);
    let mut state = de
      .deserialize_map(RootVis(state))
      .map_err(|e| format!("流式解析 model.safetensors.index.json 失败: {e}"))?;

    // 排除纯 MTP 层对最大层号和注意力层的干扰
    for l in &state.mtp_layers {
      state.linear_attn_layers.remove(l);
      state.full_attn_layers.remove(l);
    }
    // 若某层同时出现线性注意力信号（如 KDA 既有 q_proj 又有 q_conv1d/A_log），归入线性注意力层
    for l in &state.linear_attn_layers {
      state.full_attn_layers.remove(l);
    }

    let max_layer = state
      .groups
      .values()
      .filter(|g| !g.is_mtp_signal && !is_vision_signal(&g.example_name))
      .flat_map(|g| g.layers.iter().copied())
      .filter(|l| !state.mtp_layers.contains(l))
      .max()
      .or(state.max_layer_non_mtp);

    let mut representative_shards: Vec<String> = Vec::new();
    for cand in [
      state.first_trunk_shard,
      state.hetero_attn_shard,
      state.first_moe_shard,
      state.vision_shard,
      state.tail_shard,
      state.fallback_shard,
    ]
    .into_iter()
    .flatten()
    {
      if !representative_shards.iter().any(|s| s == &cand) {
        representative_shards.push(cand);
      }
      if representative_shards.len() >= 5 {
        break;
      }
    }

    Ok(StreamingIndexScan {
      total_bytes: state.total_bytes,
      shard_count: state.shards.len().max(1),
      total_tensor_count: state.total_tensor_count,
      max_layer,
      max_expert: state.max_expert,
      first_moe_layer: state.first_moe_layer,
      has_attn_bias: state.has_attn_bias,
      has_qk_norm: state.has_qk_norm,
      has_vision_tower: state.has_vision_tower,
      has_latent_moe_proj: state.has_latent_moe_proj,
      linear_attn_layers: state.linear_attn_layers,
      full_attn_layers: state.full_attn_layers,
      mtp_layers: state.mtp_layers,
      representative_shards,
      groups: state.groups,
    })
  }

  /// 将流式索引扫描结果与最多 5 个采样分片 Header 合并为 [`SafetensorsSummary`]。
  pub fn finalize_streaming_index(
    scan: StreamingIndexScan,
    sampled_headers: &[RawTensorMeta],
  ) -> SafetensorsSummary {
    let mut by_skeleton: HashMap<String, &RawTensorMeta> = HashMap::new();
    for item in sampled_headers {
      let skel = canonical_skeleton(&item.name).skeleton;
      by_skeleton.entry(skel).or_insert(item);
    }

    let mut header_inspected_count = 0usize;
    let mut dtype_map: HashMap<String, usize> = HashMap::new();
    let stage_rank = |s: &str| match s {
      "embedding" => 0,
      "vision" => 1,
      "audio" => 1,
      "attention" => 2,
      "norm" => 3,
      "ffn" => 4,
      "moe" => 5,
      "mtp" => 6,
      "lm_head" => 7,
      _ => 8,
    };

    let mut patterns: Vec<TensorPatternEntry> = scan
      .groups
      .into_values()
      .map(|g| {
        let inferred = by_skeleton.get(&g.skeleton).copied();
        let shape = inferred.and_then(|m| m.shape.clone());
        let dtype = inferred.and_then(|m| m.dtype.clone());
        let byte_len = inferred.and_then(|m| m.byte_len);
        if shape.is_some() || dtype.is_some() {
          header_inspected_count += g.count;
        }
        if let Some(ref dt) = dtype {
          *dtype_map.entry(dt.clone()).or_insert(0) += g.count;
        }

        let is_mtp_group = g.is_mtp_signal
          || (!g.layers.is_empty() && g.layers.iter().all(|l| scan.mtp_layers.contains(l)));
        let stage = if is_mtp_group {
          "mtp"
        } else {
          classify_tensor_stage(&g.example_name, None, g.layers.first().copied())
        };

        let mut pat = g.skeleton;
        if !g.layers.is_empty() {
          pat = pat.replace("<L>", &format_index_set(&g.layers));
        }
        if !g.experts.is_empty() {
          pat = pat.replace("<E>", &format_index_set(&g.experts));
        }
        TensorPatternEntry {
          pattern: pat,
          example_name: g.example_name,
          stage: stage.to_string(),
          repeat_count: g.count,
          shape,
          dtype,
          single_byte_len: byte_len,
          example_shard: Some(g.shard),
        }
      })
      .collect();

    patterns.sort_by(|a, b| {
      stage_rank(&a.stage).cmp(&stage_rank(&b.stage)).then_with(|| a.pattern.cmp(&b.pattern))
    });

    let mut dtype_histogram: Vec<(String, usize)> = dtype_map.into_iter().collect();
    dtype_histogram.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    let l2_estimated_params = estimate_l2_params(&patterns, scan.total_tensor_count);
    let source_kind = if sampled_headers.is_empty() {
      format!("model.safetensors.index.json ({} shards)", scan.shard_count)
    } else {
      format!("model.safetensors.index.json + sampled headers ({} shards)", scan.shard_count)
    };

    SafetensorsSummary {
      source_kind,
      total_bytes: scan.total_bytes,
      shard_count: scan.shard_count,
      total_tensor_count: scan.total_tensor_count,
      header_inspected_count,
      detected_num_layers: scan.max_layer.map(|l| l + 1),
      detected_num_experts: scan.max_expert.map(|e| e + 1),
      detected_first_moe_layer: scan.first_moe_layer,
      has_attn_bias: scan.has_attn_bias,
      has_qk_norm: scan.has_qk_norm,
      dtype_histogram,
      l2_estimated_params,
      detected_linear_attn_layers: (!scan.linear_attn_layers.is_empty())
        .then_some(scan.linear_attn_layers.len() as u64),
      detected_full_attn_layers: (!scan.full_attn_layers.is_empty())
        .then_some(scan.full_attn_layers.len() as u64),
      has_vision_tower: scan.has_vision_tower,
      has_latent_moe_proj: scan.has_latent_moe_proj,
      patterns,
    }
  }

  fn build_summary(
    source_kind: String,
    total_bytes: Option<u64>,
    shard_count: usize,
    tensors: Vec<RawTensorMeta>,
  ) -> SafetensorsSummary {
    let total_tensor_count = tensors.len();
    let mut header_inspected_count = 0usize;
    let mut max_layer: Option<u64> = None;
    let mut max_expert: Option<u64> = None;
    let mut first_moe_layer: Option<u64> = None;
    let mut has_attn_bias = false;
    let mut has_qk_norm = false;
    let mut has_vision_tower = false;
    let mut has_latent_moe_proj = false;
    let mut linear_attn_layers: BTreeSet<u64> = BTreeSet::new();
    let mut full_attn_layers: BTreeSet<u64> = BTreeSet::new();
    let mut dtype_map: HashMap<String, usize> = HashMap::new();

    struct GroupAcc {
      skeleton: String,
      example_name: String,
      stage: &'static str,
      count: usize,
      layers: BTreeSet<u64>,
      experts: BTreeSet<u64>,
      shape: Option<Vec<u64>>,
      dtype: Option<String>,
      byte_len: Option<u64>,
      shard: Option<String>,
    }

    // 先过一遍识别 MTP 层（含 `eh_proj` / `shared_head` 的层不计入常规主干层上限）。
    let mut mtp_layers: BTreeSet<u64> = BTreeSet::new();
    for t in &tensors {
      let sp = canonical_skeleton(&t.name);
      if let Some(l) = sp.layer_idx {
        if t.name.contains("eh_proj") || t.name.contains("enorm") || t.name.contains("shared_head")
        {
          mtp_layers.insert(l);
        }
      }
    }

    // 为了区分 Dense 层与 MoE 层的同构骨架，把“是否属于 MTP 层”编码进分组键。
    let mut groups: BTreeMap<(bool, String), GroupAcc> = BTreeMap::new();

    for t in tensors {
      if t.shape.is_some() || t.dtype.is_some() {
        header_inspected_count += 1;
      }
      if let Some(ref dt) = t.dtype {
        *dtype_map.entry(dt.clone()).or_insert(0) += 1;
      }
      if (t.name.contains("self_attn") || t.name.contains(".attn.")) && t.name.ends_with(".bias") {
        has_attn_bias = true;
      }
      if t.name.contains("q_norm.") || t.name.contains("k_norm.") {
        has_qk_norm = true;
      }
      if is_vision_signal(&t.name) {
        has_vision_tower = true;
      }
      if t.name.contains("routed_expert_down_proj") || t.name.contains("routed_expert_up_proj") {
        has_latent_moe_proj = true;
      }

      let sp = canonical_skeleton(&t.name);
      let is_mtp_layer = sp.layer_idx.is_some_and(|l| mtp_layers.contains(&l));
      if let Some(l) = sp.layer_idx {
        if !is_mtp_layer && !is_vision_signal(&t.name) {
          max_layer = Some(max_layer.map_or(l, |cur| cur.max(l)));
          if is_linear_attn_signal(&t.name) {
            linear_attn_layers.insert(l);
          } else if is_full_attn_signal(&t.name) {
            full_attn_layers.insert(l);
          }
        }
        if (t.name.contains(".experts.")
          || t.name.contains("mlp.gate.weight")
          || t.name.contains("block_sparse_moe.gate.weight"))
          && !is_mtp_layer
        {
          first_moe_layer = Some(first_moe_layer.map_or(l, |cur| cur.min(l)));
        }
      }
      if let Some(e) = sp.expert_idx {
        max_expert = Some(max_expert.map_or(e, |cur| cur.max(e)));
      }

      let stage =
        if is_mtp_layer { "mtp" } else { classify_tensor_stage(&t.name, None, sp.layer_idx) };

      let entry = groups.entry((is_mtp_layer, sp.skeleton.clone())).or_insert_with(|| GroupAcc {
        skeleton: sp.skeleton,
        example_name: t.name.clone(),
        stage,
        count: 0,
        layers: BTreeSet::new(),
        experts: BTreeSet::new(),
        shape: t.shape.clone(),
        dtype: t.dtype.clone(),
        byte_len: t.byte_len,
        shard: t.shard.clone(),
      });
      entry.count += 1;
      if let Some(l) = sp.layer_idx {
        entry.layers.insert(l);
      }
      if let Some(e) = sp.expert_idx {
        entry.experts.insert(e);
      }
      if entry.shape.is_none() && t.shape.is_some() {
        entry.shape = t.shape;
      }
      if entry.dtype.is_none() && t.dtype.is_some() {
        entry.dtype = t.dtype;
      }
      if entry.byte_len.is_none() && t.byte_len.is_some() {
        entry.byte_len = t.byte_len;
      }
    }

    let mut dtype_histogram: Vec<(String, usize)> = dtype_map.into_iter().collect();
    dtype_histogram.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    let stage_rank = |s: &str| match s {
      "embedding" => 0,
      "attention" => 1,
      "norm" => 2,
      "ffn" => 3,
      "moe" => 4,
      "mtp" => 5,
      "lm_head" => 6,
      _ => 7,
    };

    let mut patterns: Vec<TensorPatternEntry> = groups
      .into_values()
      .map(|g| {
        let mut pat = g.skeleton;
        if !g.layers.is_empty() {
          pat = pat.replace("<L>", &format_index_set(&g.layers));
        }
        if !g.experts.is_empty() {
          pat = pat.replace("<E>", &format_index_set(&g.experts));
        }
        TensorPatternEntry {
          pattern: pat,
          example_name: g.example_name,
          stage: g.stage.to_string(),
          repeat_count: g.count,
          shape: g.shape,
          dtype: g.dtype,
          single_byte_len: g.byte_len,
          example_shard: g.shard,
        }
      })
      .collect();

    patterns.sort_by(|a, b| {
      stage_rank(&a.stage).cmp(&stage_rank(&b.stage)).then_with(|| a.pattern.cmp(&b.pattern))
    });

    for l in &linear_attn_layers {
      full_attn_layers.remove(l);
    }
    let l2_estimated_params = estimate_l2_params(&patterns, total_tensor_count);

    SafetensorsSummary {
      source_kind,
      total_bytes,
      shard_count,
      total_tensor_count,
      header_inspected_count,
      detected_num_layers: max_layer.map(|l| l + 1),
      detected_num_experts: max_expert.map(|e| e + 1),
      detected_first_moe_layer: first_moe_layer,
      has_attn_bias,
      has_qk_norm,
      dtype_histogram,
      l2_estimated_params,
      detected_linear_attn_layers: (!linear_attn_layers.is_empty())
        .then_some(linear_attn_layers.len() as u64),
      detected_full_attn_layers: (!full_attn_layers.is_empty())
        .then_some(full_attn_layers.len() as u64),
      has_vision_tower,
      has_latent_moe_proj,
      patterns,
    }
  }
}

#[cfg(feature = "server")]
pub use parser::{
  finalize_streaming_index, parse_safetensors_header_bytes, parse_safetensors_header_json,
  parse_safetensors_header_len, parse_safetensors_index_json, parse_safetensors_index_streaming,
  select_representative_shards, summarize_index_and_headers, summarize_single_header,
  RawTensorMeta, StreamingIndexScan,
};

#[cfg(all(test, feature = "server"))]
mod tests {
  use super::*;

  #[test]
  fn parses_binary_safetensors_header_frame() {
    let header_json = br#"{"__metadata__":{"format":"pt"},"model.embed_tokens.weight":{"dtype":"BF16","shape":[151936,4096],"data_offsets":[0,1244659712]},"model.layers.0.self_attn.q_norm.weight":{"dtype":"BF16","shape":[128],"data_offsets":[1244659712,1244659968]}}"#;
    let len = header_json.len() as u64;
    let mut framed = Vec::from(len.to_le_bytes());
    framed.extend_from_slice(header_json);
    framed.extend_from_slice(&[0u8; 16]); // trailing tensor payload bytes ignored cleanly

    let parsed =
      parse_safetensors_header_bytes(&framed, Some("model-00001-of-00004.safetensors")).unwrap();
    assert_eq!(parsed.len(), 2);
    assert_eq!(parsed[0].name, "model.embed_tokens.weight");
    assert_eq!(parsed[0].shape, Some(vec![151936, 4096]));
    assert_eq!(parsed[0].dtype.as_deref(), Some("BF16"));
    assert_eq!(parsed[0].byte_len, Some(1_244_659_712));

    let summary = summarize_single_header(parsed);
    assert!(summary.has_qk_norm);
    assert_eq!(summary.detected_num_layers, Some(1));
    assert_eq!(summary.total_bytes, Some(1_244_659_968));
  }

  #[test]
  fn rejects_oversized_or_truncated_safetensors_header() {
    assert!(parse_safetensors_header_len(&[1, 2, 3]).is_err());
    let huge = ((MAX_SAFETENSORS_HEADER_BYTES as u64) + 1).to_le_bytes();
    assert!(parse_safetensors_header_len(&huge).is_err());

    let truncated = 64u64.to_le_bytes();
    assert!(parse_safetensors_header_bytes(&truncated, None).is_err());
  }

  #[test]
  fn collapses_moe_layers_and_propagates_sampled_header_shapes() {
    let mut weight_map = HashMap::new();
    weight_map
      .insert("model.embed_tokens.weight".to_string(), "model-00001.safetensors".to_string());
    // Layer 0 dense
    weight_map.insert(
      "model.layers.0.mlp.gate_proj.weight".to_string(),
      "model-00001.safetensors".to_string(),
    );
    // Layers 1..=2 with 4 experts each
    for l in 1..=2 {
      for e in 0..4 {
        weight_map.insert(
          format!("model.layers.{l}.mlp.experts.{e}.gate_proj.weight"),
          format!("model-0000{}.safetensors", l + 1),
        );
      }
    }
    // MTP layer 3
    weight_map
      .insert("model.layers.3.eh_proj.weight".to_string(), "model-00004.safetensors".to_string());
    weight_map.insert("lm_head.weight".to_string(), "model-00004.safetensors".to_string());

    let shards = select_representative_shards(&weight_map);
    assert_eq!(shards.len(), 3);
    assert!(shards.contains(&"model-00001.safetensors".to_string()));
    assert!(shards.contains(&"model-00002.safetensors".to_string()));
    assert!(shards.contains(&"model-00004.safetensors".to_string()));

    // Only sample expert 0 of layer 1 — propagation should fill all 8 expert tensors!
    let sampled = vec![RawTensorMeta {
      name: "model.layers.1.mlp.experts.0.gate_proj.weight".to_string(),
      shape: Some(vec![2048, 7168]),
      dtype: Some("F8_E4M3".to_string()),
      byte_len: Some(14_680_064),
      shard: Some("model-00002.safetensors".to_string()),
    }];

    let summary = summarize_index_and_headers(Some(685_000_000_000), &weight_map, &sampled);
    assert_eq!(summary.total_tensor_count, 12);
    assert_eq!(summary.detected_num_layers, Some(3)); // Layers 0, 1, 2 (MTP layer 3 excluded!)
    assert_eq!(summary.detected_num_experts, Some(4));
    assert_eq!(summary.detected_first_moe_layer, Some(1));

    let moe_pat =
      summary.patterns.iter().find(|p| p.stage == "moe").expect("moe pattern should exist");
    assert_eq!(moe_pat.pattern, "model.layers.[1..2].mlp.experts.[0..3].gate_proj.weight");
    assert_eq!(moe_pat.repeat_count, 8);
    assert_eq!(moe_pat.shape, Some(vec![2048, 7168]));
    assert_eq!(moe_pat.dtype.as_deref(), Some("F8_E4M3"));
  }

  #[test]
  fn streaming_index_handles_hetero_kda_dsa_latent_mxfp4_and_estimates_params() {
    let raw_idx = r#"{
      "metadata": {"total_size": 1561217377280},
      "weight_map": {
        "vision_tower.encoder.blocks.0.wqkv.weight": "model-00095.safetensors",
        "mm_projector.proj.0.weight": "model-00096.safetensors",
        "language_model.model.embed_tokens.weight": "model-00001.safetensors",
        "language_model.model.layers.0.self_attn.q_conv1d.weight": "model-00001.safetensors",
        "language_model.model.layers.0.self_attn.q_proj.weight": "model-00001.safetensors",
        "language_model.model.layers.1.self_attn.q_conv1d.weight": "model-00002.safetensors",
        "language_model.model.layers.1.block_sparse_moe.routed_expert_down_proj.weight": "model-00002.safetensors",
        "language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight_packed": "model-00002.safetensors",
        "language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight_scale": "model-00002.safetensors",
        "language_model.model.layers.1.block_sparse_moe.experts.1.w1.weight_packed": "model-00002.safetensors",
        "language_model.model.layers.1.block_sparse_moe.experts.1.w1.weight_scale": "model-00002.safetensors",
        "language_model.model.layers.2.self_attn.kv_a_proj_with_mqa.weight": "model-00004.safetensors",
        "language_model.model.layers.2.self_attn.indexer.wk.weight": "model-00004.safetensors",
        "language_model.lm_head.weight": "model-00096.safetensors"
      }
    }"#;

    let scan = parse_safetensors_index_streaming(raw_idx).expect("streaming index scan");
    let shards = scan.sample_shards();
    assert!(shards.contains(&"model-00001.safetensors".to_string()));
    assert!(shards.contains(&"model-00002.safetensors".to_string()));
    assert!(shards.contains(&"model-00004.safetensors".to_string()));
    assert!(shards.contains(&"model-00095.safetensors".to_string()));

    let sampled = vec![
      RawTensorMeta {
        name: "vision_tower.encoder.blocks.0.wqkv.weight".to_string(),
        shape: Some(vec![4608, 1024]),
        dtype: Some("BF16".to_string()),
        byte_len: Some(4608 * 1024 * 2),
        shard: Some("model-00095.safetensors".to_string()),
      },
      RawTensorMeta {
        name: "mm_projector.proj.0.weight".to_string(),
        shape: Some(vec![4096, 4096]),
        dtype: Some("BF16".to_string()),
        byte_len: Some(4096 * 4096 * 2),
        shard: Some("model-00096.safetensors".to_string()),
      },
      RawTensorMeta {
        name: "language_model.model.embed_tokens.weight".to_string(),
        shape: Some(vec![163840, 7168]),
        dtype: Some("BF16".to_string()),
        byte_len: Some(163840 * 7168 * 2),
        shard: Some("model-00001.safetensors".to_string()),
      },
      RawTensorMeta {
        name: "language_model.model.layers.0.self_attn.q_conv1d.weight".to_string(),
        shape: Some(vec![12288, 1, 4]),
        dtype: Some("F32".to_string()),
        byte_len: Some(12288 * 4 * 4),
        shard: Some("model-00001.safetensors".to_string()),
      },
      RawTensorMeta {
        name: "language_model.model.layers.0.self_attn.q_proj.weight".to_string(),
        shape: Some(vec![12288, 7168]),
        dtype: Some("BF16".to_string()),
        byte_len: Some(12288 * 7168 * 2),
        shard: Some("model-00001.safetensors".to_string()),
      },
      RawTensorMeta {
        name: "language_model.model.layers.1.block_sparse_moe.routed_expert_down_proj.weight"
          .to_string(),
        shape: Some(vec![3584, 7168]),
        dtype: Some("BF16".to_string()),
        byte_len: Some(3584 * 7168 * 2),
        shard: Some("model-00002.safetensors".to_string()),
      },
      RawTensorMeta {
        name: "language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight_packed"
          .to_string(),
        shape: Some(vec![3072, 1792]),
        dtype: Some("U8".to_string()),
        byte_len: Some(3072 * 1792),
        shard: Some("model-00002.safetensors".to_string()),
      },
      RawTensorMeta {
        name: "language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight_scale"
          .to_string(),
        shape: Some(vec![3072, 112]),
        dtype: Some("U8".to_string()),
        byte_len: Some(3072 * 112),
        shard: Some("model-00002.safetensors".to_string()),
      },
      RawTensorMeta {
        name: "language_model.model.layers.2.self_attn.kv_a_proj_with_mqa.weight".to_string(),
        shape: Some(vec![576, 7168]),
        dtype: Some("BF16".to_string()),
        byte_len: Some(576 * 7168 * 2),
        shard: Some("model-00004.safetensors".to_string()),
      },
      RawTensorMeta {
        name: "language_model.model.layers.2.self_attn.indexer.wk.weight".to_string(),
        shape: Some(vec![128, 7168]),
        dtype: Some("BF16".to_string()),
        byte_len: Some(128 * 7168 * 2),
        shard: Some("model-00004.safetensors".to_string()),
      },
      RawTensorMeta {
        name: "language_model.lm_head.weight".to_string(),
        shape: Some(vec![163840, 7168]),
        dtype: Some("BF16".to_string()),
        byte_len: Some(163840 * 7168 * 2),
        shard: Some("model-00096.safetensors".to_string()),
      },
    ];

    let summary = finalize_streaming_index(scan, &sampled);
    assert_eq!(summary.detected_num_layers, Some(3));
    assert_eq!(summary.detected_num_experts, Some(2));
    assert_eq!(summary.detected_first_moe_layer, Some(1));
    assert_eq!(summary.detected_linear_attn_layers, Some(2)); // layers 0, 1
    assert_eq!(summary.detected_full_attn_layers, Some(1)); // layer 2
    assert!(summary.has_vision_tower);
    assert!(summary.has_latent_moe_proj);
    // 2 个 MXFP4 打包专家各 3072 * 1792 * 2 = 11,010,048，且 weight_scale 不重复计入
    let expected_experts = 2 * 3072 * 1792 * 2;
    let l2_est = summary.l2_estimated_params.expect("l2_estimated_params should be computed");
    assert!(l2_est > expected_experts);
    assert!(summary
      .patterns
      .iter()
      .any(|p| p.stage == "vision" && p.pattern.contains("mm_projector")));
  }
}
