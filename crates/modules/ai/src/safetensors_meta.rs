//! L2 Safetensors 元数据解析器（`model.safetensors.index.json` + `.safetensors` 二进制文件头）。
//!
//! 设计目标：
//! - **零权重下载**：只需拉取几 KB ~ 几百 KB 的 `index.json` 或单分片前 `8 + header_len` 字节，
//!   即可提取真实张量名称、精确 Shape、真实 Dtype（`BF16` / `F8_E4M3` / `I4` / `U8`）与磁盘字节数。
//! - **MoE 大模型压缩折叠**：像 DeepSeek-V3 的 `index.json` 含约 9 万条张量记录（61 层 × 256 专家），
//!   解析器在统计完精确总数后，会将同构重复的 `layers.<L>` 与 `experts.<E>` 折叠为带 `repeat_count`
//!   的模式条目，使跨端传输体积保持在数 KB，同时完整保留真实 Shape、Dtype 与分片定位。

use std::collections::{BTreeMap, BTreeSet, HashMap};

use serde::{Deserialize, Serialize};
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
  /// 折叠后的结构化权重模式列表（可直接用于架构节点联动与权重树浏览）
  pub patterns: Vec<TensorPatternEntry>,
}

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
    out.push(RawTensorMeta { name, shape, dtype, byte_len, shard: shard_name.map(str::to_string) });
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

  if let Some((_, s)) =
    entries.iter().find(|(k, _)| k.contains("embed_tokens") || k.contains("layers.0.self_attn"))
  {
    push_unique(&mut picked, s);
  }
  if let Some((_, s)) =
    entries.iter().find(|(k, _)| k.contains(".experts.0.") || k.contains(".experts."))
  {
    push_unique(&mut picked, s);
  }
  if let Some((_, s)) = entries.iter().rfind(|(k, _)| k.contains("lm_head") || k.contains("norm")) {
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
    if matches!(seg, "layers" | "h" | "block" | "blocks") && i + 1 < parts.len() {
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
  if name.contains("embed_tokens") || name.contains("wte") || name.contains("word_embeddings") {
    "embedding"
  } else if name.contains("lm_head") || name.contains("output.weight") {
    "lm_head"
  } else if name.contains("self_attn") || name.contains(".attn.") || name.contains("attention") {
    "attention"
  } else if name.contains("experts")
    || name.contains("shared_expert")
    || name.contains("shared_experts")
    || name.contains("mlp.gate.weight")
    || name.contains("e_score_correction_bias")
  {
    "moe"
  } else if name.contains(".mlp.") || name.contains("feed_forward") || name.contains("ffn") {
    "ffn"
  } else if name.contains("layernorm") || name.contains(".norm.") || name.ends_with("norm.weight") {
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
      if t.name.contains("eh_proj") || t.name.contains("enorm") || t.name.contains("shared_head") {
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

    let sp = canonical_skeleton(&t.name);
    let is_mtp_layer = sp.layer_idx.is_some_and(|l| mtp_layers.contains(&l));
    if let Some(l) = sp.layer_idx {
      if !is_mtp_layer {
        max_layer = Some(max_layer.map_or(l, |cur| cur.max(l)));
      }
      if (t.name.contains(".experts.") || t.name.contains("mlp.gate.weight")) && !is_mtp_layer {
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
    patterns,
  }
}

#[cfg(test)]
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
}
