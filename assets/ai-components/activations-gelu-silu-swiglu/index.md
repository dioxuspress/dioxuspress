---
slug: "activations-gelu-silu-swiglu"
title: "激活函数演进：从 Sigmoid/ReLU 到 GELU、SiLU、SwiGLU 与 Snake"
title_en: "Activation Functions: Sigmoid, ReLU, GELU, SiLU, SwiGLU & Snake"
family: "activation"
era: "frontier"
modalities: ["general", "nlp", "vision", "audio"]
order: 10
featured: true
lab_kind: "activation"
lab_mode: "swiglu"
formula_short: "SwiGLU(x) = (SiLU(x·W_g) ⊙ (x·W_u)) · W_d"
complexity_badge: "O(B·S·d_ff) Pointwise / Bilinear"
summary: "系统拆解饱和激活梯度消失、ReLU 死亡神经元、GELU 精确与近似残差、SwiGLU 双线性门控 8/3d 等参数推导及音频声码器周期激活 Snake。"
summary_en: "Dissects vanishing gradients, Dead ReLU, GELU approximation error, SwiGLU 8/3d parameter parity, and periodic audio Snake activations."
tags: ["SwiGLU", "SiLU", "GELU", "ReLU", "Snake", "FFN"]
model_keywords: ["deepseek", "qwen", "llama", "gemma", "flux", "whisper"]
---

## 1. 为什么神经网络不能没有非线性激活？

若去掉所有非线性激活函数，任意 $L$ 层线性投影的串联 $h_L = (\dots(x W_1) W_2 \dots) W_L = x (W_1 W_2 \dots W_L)$ 在代数上都会坍缩为**单个矩阵乘法** $x W_{\text{eff}}$，无论堆叠多少层都只能切出线性超平面。激活函数在逐元素或双线性通道上折弯特征空间，决定了**前向表征能力**与**反向雅可比连乘梯度存活率**。

---

## 2. 四代激活函数的核心公式与梯度边界

### 2.1 早期饱和族（Sigmoid / Tanh）与梯度连乘消失

$$\sigma(x) = \frac{1}{1 + e^{-x}}, \qquad \sigma'(x) = \sigma(x)\bigl(1 - \sigma(x)\bigr) \le \frac{1}{4}$$

即使每一层激活恰好落在梯度最大的 $x=0$ 处，经过 $10$ 层反向传播连乘后，上层传下的梯度也会至少衰减 $(0.25)^{10} \approx 9.54 \times 10^{-7}$ 倍（可在上方交互实验台将主函数切换为 `Sigmoid` 验证）。此外，$\sigma(x) > 0$ 恒正导致下一层权重梯度同正或同负，优化轨迹呈现低效锯齿震荡。

### 2.2 整流线性族（ReLU / LeakyReLU / ReLU6 / ReLU²）

$$\operatorname{ReLU}(x) = \max(0, x), \qquad \operatorname{LeakyReLU}_\alpha(x) = \begin{cases} x, & x \ge 0 \\ \alpha x, & x < 0 \end{cases}$$

- **正半轴恒等导数**：当 $x > 0$ 时导数恒为 $1$，解决了深层连乘梯度衰减。
- **Dead ReLU 不可逆死区**：一旦某次大步长更新使神经元偏置大幅变负，对所有训练样本均有 $x < 0 \implies f'(x) = 0$，该神经元权重梯度永远归零。`LeakyReLU`（固定 $\alpha$）与 `PReLU`（可学习 $\alpha$）通过负半轴微斜率兜底；`ReLU6 = min(max(0, x), 6)` 则将上界锁在 $6$ 以提升端侧 `INT8` 定点量化分辨率。

### 2.3 平滑自门控族（GELU 精确解 vs Tanh 近似 / SiLU）

`GELU` 按输入值在标准正态分布中的累积概率 $\Phi(x)$ 对自身做软门控；工业实现常混用精确误差函数版与 `tanh` 多项式近似版：

$$\operatorname{GELU}_{\text{exact}}(x) = \frac{1}{2}x\left[1 + \operatorname{erf}\!\left(\frac{x}{\sqrt{2}}\right)\right], \qquad \operatorname{GELU}_{\text{tanh}}(x) = \frac{1}{2}x\left[1 + \tanh\!\left(\sqrt{\frac{2}{\pi}}(x + 0.044715 x^3)\right)\right]$$

两者在 $|x| \approx 2.3$ 附近存在最大约 $4.73 \times 10^{-4}$ 的确定性残差——在将 PyTorch 权重跨语言迁移至 Rust 时，必须核对 `config.json` 中的 `hidden_act`（`"gelu"` vs `"gelu_pytorch_tanh"`），否则深层累积误差会突破 `1e-2`。

`SiLU`（即 $\beta=1$ 的 `Swish`）定义为 $\operatorname{SiLU}(x) = x \cdot \sigma(x)$，其在 $x \approx -1.28$ 处拥有一个微弱的负波谷（约 $-0.278$），既提供隐式上界正则，又允许负边缘特征回传非零负梯度。

### 2.4 现代大模型标配：SwiGLU 与 $\frac{8}{3}d$ 参数对齐推导

现代旗舰大模型（Llama、Qwen、DeepSeek）普遍将传统两矩阵 FFN 替换为三矩阵门控线性单元 `SwiGLU`：

$$\operatorname{SwiGLU}(x) = \bigl(\operatorname{SiLU}(x W_{\text{gate}}) \odot (x W_{\text{up}})\bigr) W_{\text{down}}$$

- **双通路乘积梯度流**：设 $g = \operatorname{SiLU}(x W_{\text{gate}}), u = x W_{\text{up}}$，对输入求导有 $\nabla (g \odot u) = (\nabla g)\odot u + g \odot (\nabla u)$，两条支路互为乘法调制器，表达局部二阶交互。
- **为什么中间维取 $\frac{8}{3}d$？** 传统两矩阵 FFN（$d \to 4d \to d$）参数量为 $2 \times d \times 4d = 8d^2$；`SwiGLU` 包含三块形状为 $[d, d_{\text{ff}}]$ 的矩阵，参数量为 $3 d \cdot d_{\text{ff}}$。令 $3 d \cdot d_{\text{ff}} = 8d^2$，即得等参数与等 FLOPs 黄金比例：

$$d_{\text{ff}} = \frac{8}{3} d$$

工程实践中通常再向上对齐到 `128` 或 `256` 的整数倍以打满 GPU Tensor Core 分块对齐。

### 2.5 音频声码器专属：周期谐波激活 Snake

标准 `ReLU` / `SiLU` 在训练分布外仅呈单调线性外推，缺乏周期性归纳偏置，直接合成音频波形容易在高频泛音处塌陷发闷。`BigVGAN` 与 `DAC` 采用的 `Snake` 激活叠加了平方正弦项：

$$\operatorname{Snake}_a(x) = x + \frac{1}{a}\sin^2(a x), \qquad \operatorname{Snake}'_a(x) = 1 + \sin(2 a x) \in [0, 2]$$

导数恒非负（保持单调趋势易优化），同时内建可控角频率 $a$ 的周期振荡。

---

## 3. PyTorch 与原生 Rust（Candle / Burn）对照实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLUFFN(nn.Module):
  def __init__(self, dim: int, hidden_dim: int) -> None:
    super().__init__()
    self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
    self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
    self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
```

```rust
use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module};

pub struct SwiGluFfn {
  gate_proj: Linear,
  up_proj: Linear,
  down_proj: Linear,
}

impl Module for SwiGluFfn {
  fn forward(&self, xs: &Tensor) -> Result<Tensor> {
    let gate = candle_nn::ops::silu(&self.gate_proj.forward(xs)?)?;
    let up = self.up_proj.forward(xs)?;
    self.down_proj.forward(&(gate * up)?)
  }
}
```
