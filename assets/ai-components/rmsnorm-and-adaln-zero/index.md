---
slug: "rmsnorm-and-adaln-zero"
title: "归一化与条件调制：Batch/Layer/Group/RMSNorm、FP16 溢出防线与 AdaLN-Zero"
title_en: "Normalization & Conditioning: BN, LN, GN, RMSNorm, FP16 Guard & AdaLN-Zero"
family: "norm_conditioning"
era: "frontier"
modalities: ["general", "nlp", "vision", "video"]
order: 29
featured: true
lab_kind: "norm_precision"
lab_mode: "rmsnorm"
formula_short: "RMSNorm(x) = x / √(mean(x²) + ε) ⊙ γ · AdaLN-Zero: h' = x + α(t)⊙Sub(y)"
complexity_badge: "Memory-Bandwidth Bound Reduction"
summary: "通过三维张量切片对比五大归一化统计轴，实测半精度 x² > 65504 上溢陷阱，并拆解扩散 Transformer 核心条件门控 AdaLN-Zero 零初始化恒等原理。"
summary_en: "Compares normalization axes across 3D slices, demonstrates FP16 square overflow, and derives DiT AdaLN-Zero identity conditioning."
tags: ["RMSNorm", "LayerNorm", "GroupNorm", "QK-Norm", "AdaLN-Zero", "FP16"]
model_keywords: ["deepseek", "qwen", "llama", "flux", "wan", "sd3"]
---

## 1. 四维张量上的统计归约轴全景对比

给定激活张量 $X \in \mathbb{R}^{N \times C \times S}$（其中 $S = H \times W$ 或序列长度），各类归一化的本质区别仅在于**沿哪些维度求均值/均方根**：

| 算子 | 统计归约集合 | 典型场景与核心工程权衡 |
| :--- | :--- | :--- |
| **`BatchNorm`** | 对固定通道 $c$，跨批次 $N$ 与空间 $S$ 归约 | 推理依赖训练期移动平均统计量；当微调 `BatchSize ≤ 2` 时统计方差剧烈抖动导致崩盘 |
| **`LayerNorm`** | 对单样本单位置 $(n, s)$，跨全通道 $C$ 归约 | 完全不依赖 Batch 大小，训练与推理公式 100% 一致（经典 Transformer 标配） |
| **`InstanceNorm`** | 对单样本单通道 $(n, c)$，仅跨空间 $S$ 归约 | 抹除单张图像对比度与声学音量均值，常用于风格迁移与 `AdaIN` 音色控制 |
| **`GroupNorm`** | 将通道分为 $G$ 组（通常 $G=32$），组内跨 $(C/G, S)$ 归约 | 在 `Batch=1` 高清图像/视频 VAE 与 UNet 中兼顾空间统计稳定性与通道独立性 |
| **`RMSNorm`** | 对 $(n, s)$ 跨通道 $C$ 仅求均方根，**省略均值中心化与偏置 $\beta$** | 减少一次全通道归约同步与约 $30\%$ 显存带宽读写，现代 LLM/VLM 事实标准 |

---

## 2. RMSNorm、QK-Norm 与半精度 `x²` 上溢事故

### 2.1 RMSNorm 与尺度不变性

$$\operatorname{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \varepsilon}, \qquad y = \frac{x}{\operatorname{RMS}(x)} \odot \gamma$$

对任意正常数 $\alpha > 0$，有 $\operatorname{RMS}(\alpha x) = \alpha \operatorname{RMS}(x)$，因此输出对输入权重缩放具有严格尺度不变性，且反向雅可比矩阵正交于输入向量方向，有效稳定深层残差梯度。

### 2.2 为什么绝对不能在纯 `FP16` 下直接计算归一化？

IEEE-754 `FP16`（1 位符号 + 5 位指数 + 10 位尾数）能表示的最大有限正数为 **$65504$**。当深层大模型或 VAE 潜特征中出现幅度超过 $\sqrt{65504} = 255.94$ 的离群激活通道（例如 `x = 260`）时，若未提前升格精度：
1. 平方项 `x * x = 67600 > 65504` 在 `FP16` 寄存器中直接变为 `+Inf`；
2. 随后 `x / sqrt(+Inf)` 或 `+Inf - +Inf` 产生整行 `0` 或传染性 **`NaN`**（可在上方实验台将离群值拉至 `265` 并切换 `纯 FP16` 亲手复现！）。

**工业铁律**：无论外层权重与矩阵乘使用 `FP16`、`BF16` 还是 `FP8`，所有 `RMSNorm / LayerNorm / GroupNorm` 的平方累加必须在片上寄存器先转为 **`FP32`** 计算完毕后再转回目标精度。

### 2.3 QK-Norm：多模态与百亿 DiT 防梯度爆炸安全阀

在百亿参数多模态模型与扩散 Transformer（如 `FLUX.1`、`SD3.5`、`Qwen3`）中，高学习率下查询与键向量模长容易随训练步数失控增长，使 $q^\top k / \sqrt{d_k}$ 冲上数百，Softmax 瞬间退化为不可导阶跃函数。在施加 `RoPE` 旋转前对每个注意力头的 $q, k$ 独立执行 `RMSNorm`（即 **`QK-Norm`**），可将内积严格锚定在受控动态范围内。

---

## 3. 扩散 Transformer 核心：AdaLN-Zero 零门控恒等机制

扩散模型需要将连续时间步 $t \in [0, 1]$ 与全局条件向量 $c$ 注入每一层残差块。`DiT` 由条件嵌入经 `SiLU + Linear` 一次性预测出 6 组通道向量 $(\gamma_1, \beta_1, \alpha_1, \gamma_2, \beta_2, \alpha_2)$：

$$\tilde{h} = \bigl(1 + \gamma_1(t, c)\bigr) \odot \operatorname{LN}(h) + \beta_1(t, c), \qquad h' = h + \alpha_1(t, c) \odot \operatorname{Attn}(\tilde{h})$$

**为什么把条件投影线性层的权重和偏置全部初始化为 `0`（AdaLN-Zero）至关重要？**
在第 0 步时，$\gamma_1 = 0, \beta_1 = 0, \alpha_1 = 0$，于是整座 60 层 DiT 块的初始前向传播严格退化为恒等直通 $h' = h + 0 = h$，消除了深层随机子层叠加引起的初始信号方差爆炸，同时上游残差梯度无损直达底层。

---

## 4. PyTorch 与原生 Rust（Candle）安全升格实现

```python
import torch
import torch.nn as nn

class SafeRMSNorm(nn.Module):
  def __init__(self, dim: int, eps: float = 1e-6) -> None:
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(dim))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    orig_dtype = x.dtype
    x_fp32 = x.float()
    rms = torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
    return (x_fp32 * rms).to(orig_dtype) * self.weight
```

```rust
use candle_core::{DType, Result, Tensor};

pub fn safe_rms_norm(xs: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
  let orig_dtype = xs.dtype();
  let x_f32 = xs.to_dtype(DType::F32)?;
  let hidden = x_f32.dim(candle_core::D::Minus1)?;
  let mean_sq = (x_f32.sqr()?.sum_keepdim(candle_core::D::Minus1)? / (hidden as f64))?;
  let normed = x_f32.broadcast_div(&(mean_sq + eps)?.sqrt()?)?;
  normed.to_dtype(orig_dtype)?.broadcast_mul(weight)
}
```
