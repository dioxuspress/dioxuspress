---
slug: "lstm-deltanet-and-mamba-ssm"
title: "序列记忆状态机：从 RNN/LSTM 梯度公路到 Gated DeltaNet 与 Mamba 选择性状态空间"
title_en: "Sequence State Machines: RNN, LSTM, Gated DeltaNet & Selective Mamba SSM"
family: "recurrent_ssm"
modalities: ["nlp", "audio", "general"]
order: 40
lab_kind: "sequence_state"
lab_mode: "deltanet"
formula_short: "S_t = α_t S_{t-1}(I - β_t k_t k_t^T) + β_t v_t k_t^T"
complexity_badge: "O(T·d²) Train · O(1) State Decode"
summary: "贯通朴素 RNN 谱半径相变、LSTM 加法细胞公路、Gated DeltaNet 在线 Householder 键值擦写与 Mamba 输入选择性 ZOH 离散化。"
summary_en: "Unifies RNN BPTT spectral phase transitions, LSTM additive cell highways, Gated DeltaNet Householder erase-and-write, and Mamba selective ZOH discretization."
tags: ["RNN", "LSTM", "Gated DeltaNet", "Mamba", "SSM", "Linear Attention"]
model_keywords: ["qwen3-next", "mamba", "rwkv", "jamba", "kimi"]
---

## 1. 为什么序列建模要在「无限增长 KV Cache」与「固定维度状态压缩」之间博弈？

自回归 Transformer 解码第 $t$ 个词元时必须回溯全部历史键值缓存 $K_{1:t}, V_{1:t}$，单步访存量随上下文长度 $T$ 呈 $O(T)$ 线性增长；而循环状态机将任意长度历史压缩进**固定尺寸隐状态**（向量 $h_t \in \mathbb{R}^d$ 或矩阵记忆 $S_t \in \mathbb{R}^{d_v \times d_k}$），解码第 $100$ 万个词元时的显存占用与第 $1$ 个词元严格相等（**`0 Bytes/token` 增量**）。核心难点在于：**有限容量状态如何在不发生梯度消失/爆炸的前提下，精准擦除过期记忆并写入新事实？**

---

## 2. 四代序列状态转移方程与梯度推导

### 2.1 朴素循环网络（Elman RNN）的谱半径相变与全局范数裁剪

$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b), \qquad \frac{\partial \mathcal{L}_T}{\partial h_t} = \frac{\partial \mathcal{L}_T}{\partial h_T} \prod_{k=t+1}^{T} \operatorname{diag}\!\bigl(1 - h_k^2\bigr) W_{hh}^\top$$

设循环雅可比谱半径为 $\rho$：当 $\rho < 1$ 时长距梯度按 $\rho^{T-t}$ 指数归零；当 $\rho > 1$ 时梯度指数爆炸冲毁权重。工程上必须使用**全局 L2 范数等比例裁剪**而非逐元素硬截断（`clamp` 会扭曲梯度方向）：

$$g \leftarrow g \cdot \min\!\left(1,\; \frac{c}{\|g\|_2 + \epsilon}\right)$$

### 2.2 长短期记忆网络（LSTM）：用逐元素加法公路打破连乘衰减

$$f_t = \sigma(W_f [h_{t-1}, x_t] + b_f), \quad i_t = \sigma(W_i [h_{t-1}, x_t] + b_i), \quad c_t = f_t \odot c_{t-1} + i_t \odot \tanh(W_c [h_{t-1}, x_t] + b_c)$$

对细胞状态做单步求导可得主干项 $\frac{\partial c_t}{\partial c_{t-1}} = \operatorname{diag}(f_t)$：只要遗忘门偏置初始化为正值（如 $b_f \in [1, 2]$ 使初始 $f_t \approx 0.85\text{--}0.95$），关键句法特征即可跨越数十个从句干扰词无损透传，并在句末标点处令 $f_t \to 0$ 一键清零。

### 2.3 核化线性注意力 vs Gated DeltaNet：Householder 在线擦写准则

朴素线性注意力将键值外积直接累加进矩阵记忆 $S_t = S_{t-1} + v_t k_t^\top$。一旦同一归一化键 $k$（$\|k\|_2 = 1$）在后文被赋予新值 $v_{\text{new}}$，检索结果 $S_t k = v_{\text{old}} + v_{\text{new}} + \text{crosstalk}$ 会永久混叠旧值。

**`Gated DeltaNet`**（`Qwen3-Next` 混合骨干核心层）引入在线梯度下降（Delta Rule）与标量遗忘门 $\alpha_t \in (0, 1]$：先用 $\alpha_t S_{t-1} k_t$ 读出当前键上的旧记忆并擦除，再按步长 $\beta_t \in (0, 1]$ 写入新值：

$$S_t = \alpha_t S_{t-1}\bigl(I - \beta_t k_t k_t^\top\bigr) + \beta_t v_t k_t^\top = \alpha_t S_{t-1} + \beta_t \bigl(v_t - \alpha_t S_{t-1} k_t\bigr) k_t^\top$$

当 $\alpha_t = 1, \beta_t = 1$ 且 $\|k_t\|_2 = 1$ 时，右乘 $k_t$ 立即验证旧记忆被 **100% 精确覆写**：

$$S_t k_t = S_{t-1}(k_t - k_t(k_t^\top k_t)) + v_t(k_t^\top k_t) = v_t$$

### 2.4 选择性状态空间（Mamba / Mamba-2）：输入依赖步长 $\Delta_t(x_t)$ 与 ZOH 离散化

连续线性常微分系统 $h'(t) = A h(t) + B x(t)$ 在零阶保持（Zero-Order Hold）假设下离散化为：

$$\bar{A}_t = \exp(\Delta_t A), \qquad \bar{B}_t = (\Delta_t A)^{-1}\bigl(\exp(\Delta_t A) - I\bigr) \cdot \Delta_t B, \qquad h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t$$

传统 `S4` 固定 $(\Delta, A, B)$ 满足线性时不变（LTI），虽可用 FFT 卷积并行，却无法根据词元语义选择性记忆；`Mamba` 令 $\Delta_t = \operatorname{softplus}(W_\Delta x_t + b_\Delta)$ 随输入动态变化：
- 遇到语气词或无关背景时 $\Delta_t \to 0 \implies \bar{A}_t \to I,\; \bar{B}_t \to 0$，状态机自动闭锁防污染；
- 遇到关键实体词时 $\Delta_t$ 放大 $\implies \bar{A}_t$ 衰减历史、$\bar{B}_t$ 打开大门录入新特征。

---

## 3. PyTorch 与原生 Rust（Candle）Gated DeltaNet 递推核对照

```python
import torch
import torch.nn.functional as F

def gated_delta_step(
  state: torch.Tensor,  # [B, H, d_v, d_k]
  k: torch.Tensor,      # [B, H, d_k]
  v: torch.Tensor,      # [B, H, d_v]
  alpha: torch.Tensor,  # [B, H, 1, 1] in (0, 1]
  beta: torch.Tensor,   # [B, H, 1] in (0, 1]
) -> tuple[torch.Tensor, torch.Tensor]:
  k_norm = F.normalize(k, p=2, dim=-1)
  decayed = alpha * state
  retrieved = torch.einsum("bhvk,bhk->bhv", decayed, k_norm)
  delta_v = beta * (v - retrieved)
  new_state = decayed + torch.einsum("bhv,bhk->bhvk", delta_v, k_norm)
  return new_state, retrieved
```

```rust
use candle_core::{Result, Tensor};

/// 单步 Gated DeltaNet 在线擦写状态转移：state [B, H, Dv, Dk], k [B, H, Dk], v [B, H, Dv]
pub fn gated_delta_step(
  state: &Tensor,
  k_unit: &Tensor,
  v: &Tensor,
  alpha: &Tensor,
  beta: &Tensor,
) -> Result<Tensor> {
  let decayed = state.broadcast_mul(alpha)?;
  let k_col = k_unit.unsqueeze(3)?; // [B, H, Dk, 1]
  let retrieved = decayed.matmul(&k_col)?.squeeze(3)?; // [B, H, Dv]
  let diff = (v - &retrieved)?.broadcast_mul(beta)?;
  let outer = diff.unsqueeze(3)?.matmul(&k_unit.unsqueeze(2)?)?; // [B, H, Dv, Dk]
  decayed + outer
}
```
