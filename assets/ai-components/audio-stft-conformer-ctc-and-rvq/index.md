---
slug: "audio-stft-conformer-ctc-and-rvq"
title: "音频与语音核心原语：加窗 STFT、Mel 滤波器组、CTC 空白符网格与 RVQ 多级神经码本"
title_en: "Speech & Audio Primitives: STFT, Mel Filterbank, CTC Lattice & RVQ Codebooks"
family: "audio_codec"
modalities: ["audio", "multimodal"]
order: 62
lab_kind: "audio_spectral"
lab_mode: "stft"
formula_short: "m = 2595·log10(1 + f/700) · ẑ = ∑_{q=1}^Q e_{k_q}^(q)"
complexity_badge: "O(T·N log N) STFT · O(T·|Y|) CTC"
summary: "贯穿连续波形短时傅里叶时频不确定性权衡、梅尔非线性感知刻度、CTC 端到端无对齐动态规划与语音大模型 RVQ 残差向量量化。"
summary_en: "Spans STFT time-frequency uncertainty, Mel perceptual filterbanks, monotonic CTC forward dynamic programming, and RVQ multi-stage residual codebooks."
tags: ["STFT", "Log-Mel", "CTC", "Conformer", "RVQ", "Audio Codec"]
model_keywords: ["whisper", "qwen2.5-omni", "qwen2-audio", "sensevoice", "cosyvoice"]
---

## 1. 为什么语音大模型不能像文本一样直接按采样点做自注意力？

1 秒钟 `24 kHz` 单声道音频包含 `24,000` 个一维浮点采样点：每个单点仅代表鼓膜瞬时位移，毫无独立语义，且 30 秒语音长度高达 `720,000`，直接送入 $O(N^2)$ 自注意力在算力与归纳偏置上双双崩溃。现代语音系统依赖三条关键桥梁：
1. **声学感知前端（`STFT → Log-Mel`）**：把高频波形浓缩为 `50~100 Hz` 帧率、符合人耳耳蜗对数频率分辨特性的二维时频语谱图；
2. **单调序列对齐头（`CTC`）**：在不知道每个音素起止毫秒标注的前提下，对不等长声学帧与文本序列精确求边际似然；
3. **离散神经编解码（`RVQ`）**：将连续声学潜向量逐级拆解为离散码本 Token，使原生多模态大模型（如 `Qwen2.5-Omni`、`Moshi`）能够直接自回归“说出”高保真语音。

---

## 2. 核心信号与序列公式推导

### 2.1 短时傅里叶变换（STFT）Gabor 时频权衡与梅尔（Mel）滤波器组

对帧移为 $H$、窗长为 $N_w$ 的 Hann 窗 $w[n] = \frac{1}{2}\bigl(1 - \cos\frac{2\pi n}{N_w - 1}\bigr)$ 信号做离散傅里叶变换：

$$X(t, k) = \sum_{n=0}^{N_w - 1} x[t H + n]\, w[n]\, e^{-i 2\pi k n / N_{\text{FFT}}}, \qquad \Delta t \cdot \Delta f \ge \frac{1}{4\pi}$$

窄窗（如 `10 ms`）时间分辨率极高，可锁定塞音 `/p/, /t/, /k/` 的爆破瞬间，但主瓣过宽导致元音谐波糊成一片；宽窗（如 `40 ms`）能清晰分离基频 $F_0$ 与泛音，却会把急促辅音在时间轴上抹平（工业 ASR 普遍折中取 `25 ms` 窗长、`10 ms` 帧移）。随后经梅尔非线性映射叠加三角滤波器组 $M \in \mathbb{R}^{N_{\text{mel}} \times (N_{\text{FFT}}/2 + 1)}$ 并取对数压缩动态范围：

$$m(f) = 2595 \log_{10}\!\left(1 + \frac{f}{700}\right), \qquad S_{\text{log-mel}}(t) = \log\!\bigl(\max(M \cdot |X(t)|^2,\; \epsilon)\bigr)$$

### 2.2 CTC（Connectionist Temporal Classification）空白符折叠与前向递推

给定 $T$ 帧声学输出分布与长度 $U \le T$ 的目标文本 $Y = [y_1, \dots, y_U]$，CTC 在每个字符两侧插入空白符 $\epsilon$ 构造长度 $2U + 1$ 的扩展序列 $\tilde{Y} = [\epsilon, y_1, \epsilon, y_2, \dots, y_U, \epsilon]$，定义多对一折叠映射 $\mathcal{B}$（先合并连续相同符号，再删去所有 $\epsilon$）。

> **为什么重复双字母（如 `"apple"` 中的 `"pp"`）严禁跨过中间 $\epsilon$ 直接跳转？**
> 若帧路径为 `[..., p, p, ...]`，合并连续重复符号后会坍缩为单个 `'p'`（变成 `"aple"`）！唯有经过至少一帧空白符 `[..., p, ϵ, p, ...]`，折叠后才能保留 `"pp"`。

因此前向变量 $\alpha_{t, s}$（前 $t$ 帧落在扩展位置 $s$ 的总概率）严格遵循三分支单调状态转移：

$$\alpha_{t, s} = P_t(\tilde{y}_s) \cdot \begin{cases} \alpha_{t-1, s} + \alpha_{t-1, s-1}, & \text{若 } \tilde{y}_s = \epsilon \text{ 或 } \tilde{y}_s = \tilde{y}_{s-2} \\ \alpha_{t-1, s} + \alpha_{t-1, s-1} + \alpha_{t-1, s-2}, & \text{其它情形} \end{cases}$$

### 2.3 残差向量量化（RVQ）与直通梯度估计器（STE）

单层码本若想达到高保真音频所需的 $2^{72}$ 精细度，码本尺寸将天文级爆炸。`RVQ` 串联 $Q$ 级小型码本（如每级 $K = 1024 = 2^{10}$），每级仅量化上一级剩下的残差向量：

$$r_0 = z, \qquad k_q = \arg\min_{k \in \{1..K\}} \bigl\|r_{q-1} - e_k^{(q)}\bigr\|_2^2, \qquad r_q = r_{q-1} - e_{k_q}^{(q)}, \qquad \hat{z} = \sum_{q=1}^{Q} e_{k_q}^{(q)}$$

第 $1$ 级码本自动捕获低频语义与韵律轮廓，第 $2 \dots Q$ 级逐层补齐高频音色与呼吸细节；反向传播时因 $\arg\min$ 不可导，采用直通估计器 $\hat{z} = z + \operatorname{sg}(\hat{z} - z)$ 将解码器梯度恒等透传回编码器。

---

## 3. PyTorch 与原生 Rust（RVQ 多级残差量化）对照实现

```python
import torch
import torch.nn as nn

class ResidualVectorQuantizer(nn.Module):
  def __init__(self, num_quantizers: int, codebook_size: int, dim: int) -> None:
    super().__init__()
    self.codebooks = nn.ParameterList([
      nn.Parameter(torch.randn(codebook_size, dim) * 0.02) for _ in range(num_quantizers)
    ])

  def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    residual = z
    z_hat = torch.zeros_like(z)
    indices = []
    for cb in self.codebooks:
      dists = torch.cdist(residual, cb)
      idx = dists.argmin(dim=-1)
      quant = cb[idx]
      residual = residual - quant
      z_hat = z_hat + quant
      indices.append(idx)
    # Straight-Through Estimator (STE)
    z_ste = z + (z_hat - z).detach()
    return z_ste, torch.stack(indices, dim=-1)
```

```rust
use candle_core::{Result, Tensor, D};

/// 原生 Rust Candle 多级残差向量量化前向查表与残差递推
pub fn rvq_quantize(z: &Tensor, codebooks: &[Tensor]) -> Result<(Tensor, Vec<Tensor>)> {
  let mut residual = z.clone();
  let mut z_hat = z.zeros_like()?;
  let mut all_indices = Vec::with_capacity(codebooks.len());

  for cb in codebooks {
    // ||r - e||^2 = ||r||^2 + ||e||^2 - 2 r·e^T
    let r_sq = residual.sqr()?.sum_keepdim(D::Minus1)?;
    let e_sq = cb.sqr()?.sum(D::Minus1)?.unsqueeze(0)?;
    let dot = residual.matmul(&cb.t()?)?;
    let dists = (r_sq.broadcast_add(&e_sq)? - (dot * 2.0)?)?;
    let idx = dists.argmin(D::Minus1)?;
    let quant = cb.index_select(&idx.flatten_all()?, 0)?.reshape(z.shape())?;
    residual = (&residual - &quant)?;
    z_hat = (&z_hat + &quant)?;
    all_indices.push(idx);
  }
  Ok((z_hat, all_indices))
}
```
