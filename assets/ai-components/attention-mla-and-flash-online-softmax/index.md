---
slug: "attention-mla-and-flash-online-softmax"
title: "注意力与硬件内核全景：SDPA、GQA、DeepSeek MLA 矩阵吸收与 FlashAttention 在线重缩放"
title_en: "Attention & Hardware Kernels: SDPA, GQA, DeepSeek MLA Absorption & FlashAttention"
family: "attention"
modalities: ["nlp", "vision", "multimodal", "video"]
order: 48
lab_kind: "attention_matrix"
lab_mode: "flash"
formula_short: "Õ^(j) = diag(e^{m^(j-1)-m^(j)}) Õ^(j-1) + e^{S_j - m^(j)} V_j"
complexity_badge: "O(N²·d) FLOPs · O(N) HBM Memory"
summary: "从 √d_k 方差定标与 Softmax 雅可比零化，到 GQA/MLA 56.9× 潜空间矩阵吸收代数变形、FlashAttention SRAM 在线分块恒等式与 PagedAttention 物理页表。"
summary_en: "Covers sqrt(d_k) variance scaling, GQA vs DeepSeek MLA matrix absorption, exact FlashAttention online softmax tiling, and PagedAttention KV virtualization."
tags: ["Attention", "GQA", "MLA", "FlashAttention", "Online Softmax", "PagedAttention"]
model_keywords: ["deepseek", "qwen", "llama", "gemma", "flux", "wan", "mistral"]
---

## 1. 注意力机制的三道工程生死线：梯度饱和、HBM 中间矩阵爆炸与解码带宽墙

标准缩放点积注意力（SDPA）写作 $\operatorname{Attention}(Q, K, V) = \operatorname{Softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V$。表面只有两次矩阵乘与一次归一化，落地百亿大模型时却直面三道物理墙：
1. **反向传播死锁墙**：漏除 $\sqrt{d_k}$ 会令 Softmax 掉入极端饱和区，雅可比矩阵逼近全零；
2. **训练/Prefill 显存墙**：显式落盘 $N \times N$ 注意力分数矩阵在 $N = 128\text{K}$ 时单头即占 `32 GiB` HBM；
3. **自回归 Decode 带宽墙**：每生成一个新词都要从慢速 HBM 搬运全部历史 KV Cache，算术强度跌至 `< 2 FLOPs/Byte`。

---

## 2. 核心数学原理与等价恒等变形

### 2.1 为什么必须除以 $\sqrt{d_k}$？—— 方差膨胀与雅可比坍缩定理

设 $q, k \in \mathbb{R}^{d_k}$ 各分量独立同分布，均值 $0$、方差 $1$，则点积 $z = \sum_{i=1}^{d_k} q_i k_i$ 满足 $\mathbb{E}[z] = 0,\; \operatorname{Var}(z) = d_k$。当 $d_k = 128$ 时，标准差放大 $\sqrt{128} \approx 11.31$ 倍，最大值比次大值高出十余个单位，导致 $p = \operatorname{Softmax}(z) \to [0, \dots, 1, \dots, 0]$。考察 Softmax 的精确雅可比矩阵：

$$\frac{\partial p}{\partial z} = \operatorname{diag}(p) - p p^\top$$

当某个分量 $p_k \to 1$、其余 $p_{j \neq k} \to 0$ 时，对角元 $p_k(1 - p_k) \to 0$，非对角元 $-p_i p_j \to 0$，**整张雅可比矩阵退化为零矩阵**，上游 $W_Q, W_K$ 梯度彻底断流。除以 $\sqrt{d_k}$ 将点积方差精确拉回 $1$。

### 2.2 DeepSeek MLA：为什么低秩压缩必须配合「矩阵吸收」与「解耦 RoPE」？

`GQA` 通过将键值头数从 $H$ 砍到 $G$ 压缩缓存，但损失了各头独立子空间。`DeepSeek-V2/V3` 的 **MLA（Multi-head Latent Attention）** 将键值联合下投影为单个低维潜向量 $c_t^{KV} = h_t W^{DKV} \in \mathbb{R}^{d_c}$（$d_c = 512 \ll H d_h = 16384$）。若推理期先恢复出显式 $k_{t,i}^C = c_t^{KV} W_i^{UK}$，中间张量会再次撑爆显存；利用矩阵乘结合律，可将解压矩阵**吸收**进查询与输出侧：

$$q_{t,i}^C (k_{j,i}^C)^\top = q_{t,i}^C \bigl(c_j^{KV} W_i^{UK}\bigr)^\top = \underbrace{\bigl(q_{t,i}^C (W_i^{UK})^\top\bigr)}_{\tilde{q}_{t,i} \in \mathbb{R}^{d_c}} (c_j^{KV})^\top, \qquad \sum_{j=1}^t \alpha_{t,j,i}\bigl(c_j^{KV} W_i^{UV}\bigr) W_i^O = \left(\sum_{j=1}^t \alpha_{t,j,i} c_j^{KV}\right) \bigl(W_i^{UV} W_i^O\bigr)$$

唯一障碍是**旋转位置编码矩阵 $R_{t-j}$ 依赖相对位置，夹在 $W_Q$ 与 $W^{UK}$ 之间无法静态合并**！MLA 因此采用**解耦 RoPE（Decoupled RoPE）**：单独切出极窄的共享旋转键 $k_t^R = \operatorname{RoPE}(h_t W^{KR}) \in \mathbb{R}^{64}$，单词元仅需缓存 $[c_t^{KV};\, k_t^R]$ 共 $512 + 64 = 576$ 个 `BF16` 元素（`1,152 Bytes`），较同规模 MHA 压缩 **56.89 倍**。

### 2.3 FlashAttention 与 Online Softmax：不落盘 $N \times N$ 矩阵的精确递推

将键值序列切分为片上 SRAM 可容纳的小块 $S^{(j)} = Q_i K_j^\top / \sqrt{d_k}$。每读入新块 $j$，更新全局最大值 $m^{(j)} = \max\bigl(m^{(j-1)}, \operatorname{rowmax}(S^{(j)})\bigr)$，并用指数因子 $e^{m^{(j-1)} - m^{(j)}}$ 实时重缩放旧分子与旧分母：

$$\ell^{(j)} = e^{m^{(j-1)} - m^{(j)}} \ell^{(j-1)} + \sum_{\tau} e^{S^{(j)}_\tau - m^{(j)}}, \qquad \tilde{O}^{(j)} = \operatorname{diag}\!\bigl(e^{m^{(j-1)} - m^{(j)}}\bigr)\tilde{O}^{(j-1)} + e^{S^{(j)} - m^{(j)}} V_j$$

遍历完所有块后做一次终态除法 $O_i = \operatorname{diag}(\ell^{(T_c)})^{-1} \tilde{O}^{(T_c)}$。代数展开后所有中间缩放项两两相消，结果与全矩阵 Softmax **在实数域严格恒等（浮点误差 `< 1e-15`）**，而 HBM 额外显存从 $O(N^2)$ 降至 $O(N)$。

---

## 3. PyTorch 与原生 Rust（Online Softmax Tiling）对照实现

```python
import torch

def flash_online_attention_1d(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, block_size: int = 64) -> torch.Tensor:
  scale = q.shape[-1] ** -0.5
  m_i = torch.full((q.shape[0], 1), float("-inf"), device=q.device, dtype=torch.float32)
  l_i = torch.zeros((q.shape[0], 1), device=q.device, dtype=torch.float32)
  acc = torch.zeros((q.shape[0], v.shape[-1]), device=q.device, dtype=torch.float32)

  for start in range(0, k.shape[0], block_size):
    k_blk, v_blk = k[start : start + block_size], v[start : start + block_size]
    s_ij = (q.float() @ k_blk.float().T) * scale
    m_new = torch.maximum(m_i, s_ij.max(dim=-1, keepdim=True).values)
    alpha = torch.exp(m_i - m_new)
    p_ij = torch.exp(s_ij - m_new)
    acc = acc * alpha + p_ij @ v_blk.float()
    l_i = l_i * alpha + p_ij.sum(dim=-1, keepdim=True)
    m_i = m_new
  return (acc / l_i).to(q.dtype)
```

```rust
/// 纯 Rust 单查询行分块 Online Softmax 注意力核（零 N×N 矩阵分配）
pub fn online_softmax_row(scores: &[f64], values: &[[f64; 4]], block_size: usize) -> [f64; 4] {
  let mut m_run = f64::NEG_INFINITY;
  let mut l_run = 0.0_f64;
  let mut acc = [0.0_f64; 4];

  for (s_chunk, v_chunk) in scores.chunks(block_size).zip(values.chunks(block_size)) {
    let blk_max = s_chunk.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let m_new = m_run.max(blk_max);
    let alpha = if m_run.is_finite() { (m_run - m_new).exp() } else { 0.0 };
    for d in &mut acc {
      *d *= alpha;
    }
    l_run *= alpha;
    for (&s, v_row) in s_chunk.iter().zip(v_chunk.iter()) {
      let w = (s - m_new).exp();
      l_run += w;
      for (d, &vd) in acc.iter_mut().zip(v_row.iter()) {
        *d += w * vd;
      }
    }
    m_run = m_new;
  }
  acc.map(|v| v / l_run.max(1e-15))
}
```
