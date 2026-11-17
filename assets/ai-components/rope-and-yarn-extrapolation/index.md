---
slug: "rope-and-yarn-extrapolation"
title: "旋转位置编码与长窗口外推：1D RoPE、PI、NTK-Aware、YaRN 与多维时空 M-RoPE"
title_en: "Rotary Position Embeddings & Extrapolation: RoPE, NTK, YaRN & Multimodal M-RoPE"
family: "token_positional"
era: "frontier"
modalities: ["nlp", "vision", "video", "multimodal"]
order: 59
featured: true
lab_kind: "positional"
lab_mode: "yarn"
formula_short: "⟨R_m q, R_n k⟩ = Re(Σ q_j k_j* · e^{i(m-n)θ_j}) · θ_j = b^{-2j/d}"
complexity_badge: "O(B·S·H·d_h) Orthogonal Rotation"
summary: "从复平面正交旋转证明绝对位置操作为何精确等价于相对距离内积，剖析线性内插、NTK 与 YaRN 分频段外推机理，并拓展至图像 2D Axial RoPE 与视频 3D M-RoPE。"
summary_en: "Proves RoPE relative inner-product invariance, compares PI, NTK-Aware, and YaRN wavelength-banded extrapolation, and extends to 2D/3D M-RoPE."
tags: ["RoPE", "YaRN", "NTK", "LongContext", "2D-RoPE", "M-RoPE"]
model_keywords: ["deepseek", "qwen", "llama", "gemma", "flux", "wan"]
---

## 1. 核心难题：如何用零缓存代价的「绝对位置操作」实现「纯相对距离注意力」？

如果直接在注意力矩阵上叠加相对位置表 $B_{m-n}$（如 `T5`），推理缓存与跨块分片会受二维索引牵制；如果只在词嵌入层加绝对正余弦向量 $x_m + p_m$，多层线性变换后绝对坐标与相对距离项会混杂在一起。

**旋转位置编码（RoPE）**给出了优雅的解析几何解法：将注意力头维度 $d$ 两两配对为 $d/2$ 个二维复平面子空间，对位置 $m$ 的向量第 $j$ 对分量乘上复指数旋转因子 $e^{i m \theta_j}$（其中几何级数角频率 $\theta_j = \theta_{\text{base}}^{-2j/d}$，常见基底为 $10^4$ 或 $10^6$）：

$$R_{\Theta, m}^{(j)} \begin{pmatrix} q_{2j} \\ q_{2j+1} \end{pmatrix} = \begin{pmatrix} \cos m\theta_j & -\sin m\theta_j \\ \sin m\theta_j & \cos m\theta_j \end{pmatrix} \begin{pmatrix} q_{2j} \\ q_{2j+1} \end{pmatrix}$$

### 相对内积平移不变性证明

利用正交旋转矩阵转置性质 $(R_m)^\top R_n = R_{-m} R_n = R_{n - m}$，查询与键的内积自动消去绝对起点，严格只依赖相对位移 $\Delta = m - n$：

$$\langle R_m q,\, R_n k \rangle = \sum_{j=0}^{d/2 - 1} \operatorname{Re}\!\left[ (q_{2j} + i q_{2j+1})(k_{2j} - i k_{2j+1}) e^{i(m - n)\theta_j} \right]$$

> 在上方交互实验台点击 **「整体平移 +4 (验不变性)」**，即使 $m, n$ 同时增大、左侧四个复平面指针全部转到新角度，右侧注意力内积在小数点后全部位上**纹丝不动**！

---

## 2. 长上下文外推演进：为什么高频绝对不能被压缩？

设预训练窗口长度为 $L$（如 `4096`），希望外推至 $s \cdot L$（倍率 $s > 1$）。观察第 $j$ 个子空间的波长 $\lambda_j = \frac{2\pi}{\theta_j} = 2\pi b^{2j/d}$：
- **高频分量（$j$ 小，$\lambda_j \ll L$）**：训练期已经转过成百上千整圈，专门编码**相邻几个词的精细语序**（如 `"A 击败 B"` vs `"B 击败 A"`）；
- **低频分量（$j$ 大，$\lambda_j \ge L$）**：训练期连一圈都没转完，若直接外推遇到从未见过的相位区间，注意力分布立即震荡崩塌。

| 方案 | 频率调制公式 | 优势与致命短板 |
| :--- | :--- | :--- |
| **线性内插（`PI`）** | 所有频率统一除以 $s$：$\theta'_j = \theta_j / s$ | 消除越界相位，但把高频旋转角也压窄了 $s$ 倍，局部相邻 Token 夹角过小难以区分，需较多微调步数恢复 |
| **基底扩展（`NTK-Aware`）** | 放大基底 $b' = b \cdot s^{d/(d-2)}$ | 高频缩放极小、低频缩放充分，免训练外推显著优于线性内插，但最低频仍略超原相位边界 |
| **分频斜坡（`YaRN`）** | 按圈数比 $r_j = L / \lambda_j$ 定义斜坡函数 $\gamma(r_j) \in [0, 1]$：<br>$\theta^{\text{YaRN}}_j = (1-\gamma)\frac{\theta_j}{s} + \gamma \theta_j$ | **高频（$\gamma=1$）原封不动保局部精度，低频（$\gamma=0$）严格内插防越界，中频平滑线性过渡**；配合注意力对数温度补偿 $\sqrt{t} = 0.1 \ln s + 1$ 抵消长序列熵增（`DeepSeek-V3` 等旗舰长窗口核心方案） |

---

## 3. 从一维文本走向图像与视频：2D Axial RoPE 与 3D M-RoPE

- **图像扩散 `2D Axial RoPE`（`FLUX.1` / `PixArt`）**：将头维度平分为两半，前 $d/2$ 维编码行坐标 $y$，后 $d/2$ 维编码列坐标 $x$。无需像传统 `ViT` 那样对固定二维位置表做双三次插值，天然支持竖版海报、超宽横幅等任意宽高比无损生成。
- **多模态交错 `M-RoPE`（`Qwen2.5-VL`）**：将头维度切分为 `[时间 t, 高度 h, 宽度 w]` 三段——处理纯文本时令 $t=h=w=p$，在数学上与标准 `1D RoPE` 逐位完全等价；处理图像与视频帧时各自填入真实二维/三维网格坐标，实现多模态统一位置空间。

---

## 4. PyTorch 与原生 Rust（Candle）对照实现

```python
import torch

def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
  # x: [B, H, S, D], cos/sin: [1, 1, S, D // 2]
  x_even = x[..., 0::2]
  x_odd = x[..., 1::2]
  out_even = x_even * cos - x_odd * sin
  out_odd = x_even * sin + x_odd * cos
  return torch.stack((out_even, out_odd), dim=-1).flatten(-2)
```

```rust
use candle_core::{Result, Tensor};

pub fn apply_rotary_half(xs: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
  let d = xs.dim(candle_core::D::Minus1)?;
  let half = d / 2;
  let x1 = xs.narrow(candle_core::D::Minus1, 0, half)?;
  let x2 = xs.narrow(candle_core::D::Minus1, half, half)?;
  let rot1 = (x1.broadcast_mul(cos)? - x2.broadcast_mul(sin)?)?;
  let rot2 = (x1.broadcast_mul(sin)? + x2.broadcast_mul(cos)?)?;
  Tensor::cat(&[&rot1, &rot2], candle_core::D::Minus1)
}
```
