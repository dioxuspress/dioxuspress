---
slug: "conv-and-depthwise-separable"
title: "卷积与池化全谱系：标准卷积、深度可分离、空洞盲区、转置棋盘伪影与 RoIAlign"
title_en: "Convolutions & Pooling: Standard, Depthwise Separable, Dilated, Transposed & RoIAlign"
family: "convolution"
era: "modern"
modalities: ["vision", "audio", "video"]
order: 15
featured: true
lab_kind: "grid_kernel"
lab_mode: "depthwise_separable"
formula_short: "L_out = ⌊(L_in + 2p - d·(k-1) - 1)/s + 1⌋ · Speedup ≈ (1/C_out + 1/k²)⁻¹"
complexity_badge: "Spatial & Temporal Local Operator"
summary: "一站式打通标准卷积、分组与深度可分离卷积算力拆解、空洞卷积棋盘盲区、转置卷积重叠伪影、Max/Avg/SoftPool 反向梯度路由及 RoIAlign 双线性插值。"
summary_en: "Interactive guide to standard, grouped, depthwise separable, dilated, and transposed convolutions plus pooling gradient routing and RoIAlign."
tags: ["Conv2D", "DepthwiseSeparable", "DilatedConv", "TransposedConv", "MaxPool", "RoIAlign"]
model_keywords: ["whisper", "flux", "wan", "qwen2-vl", "siglip", "sensevoice"]
---

## 1. 标准卷积（Standard Conv）局部性与平移等变性

全连接层把输入展平后参数量高达 $O(H_{\text{in}} W_{\text{in}} C_{\text{in}} \cdot H_{\text{out}} W_{\text{out}} C_{\text{out}})$，且丢失空间拓扑。卷积算子引入**局部连接（Local Connectivity）**与**空间权重共享（Weight Sharing）**，赋予网络天然的平移等变性（输入平移 $(\Delta y, \Delta x)$，特征响应同步平移）：

$$Y_{b, c_o, i, j} = \sum_{c_i=0}^{C_{\text{in}}-1} \sum_{u=0}^{k_H-1} \sum_{v=0}^{k_W-1} X_{b, c_i,\, i\cdot s + u\cdot d - p,\, j\cdot s + v\cdot d - p} \cdot W_{c_o, c_i, u, v}$$

任意维度（`1D` 音频波形、`2D` 图像特征、`3D` 视频时空体）的精确输出空间尺度遵循统一整除公式：

$$L_{\text{out}} = \left\lfloor \frac{L_{\text{in}} + 2p - d(k - 1) - 1}{s} + 1 \right\rfloor$$

---

## 2. 四大关键演进变体的物理本质

### 2.1 分组卷积与深度可分离卷积（Depthwise Separable Conv）

标准卷积在同一算子中同时完成**空间邻域聚合**与**跨通道特征重组**，单点乘加开销正比于 $C_{\text{in}} \cdot C_{\text{out}} \cdot k^2$。`MobileNet`、`ConvNeXt` 与语音骨干 `Conformer` 将其拆解为串联两步：

1. **Depthwise Conv（`groups = C_in`）**：每个输入通道只配 1 个独立的 $k\times k$ 滤波器，参数量为 $C_{\text{in}} k^2$；
2. **Pointwise `1×1` Conv**：对每个空间坐标执行跨通道线性混合 $[C_{\text{in}} \to C_{\text{out}}]$，参数量为 $C_{\text{in}} C_{\text{out}}$。

相对标准卷积的计算量与参数比值为：

$$\frac{\text{MAC}_{\text{sep}}}{\text{MAC}_{\text{std}}} = \frac{C_{\text{in}} k^2 + C_{\text{in}} C_{\text{out}}}{C_{\text{in}} C_{\text{out}} k^2} = \frac{1}{C_{\text{out}}} + \frac{1}{k^2}$$

当 $k = 3, C_{\text{out}} \gg 9$ 时，开销直接降至原先的约 $\frac{1}{9}$（节省 **88%+** 算力）。

### 2.2 空洞卷积（Dilated / Atrous Conv）与棋盘盲区效应

通过在核元素间插入 $d - 1$ 个零孔洞，有效感受野扩展为 $k_{\text{eff}} = d(k - 1) + 1$，无需池化损失空间分辨率即可获得大视野（如语义分割 `ASPP` 与音频 `WaveNet`）。但若连续堆叠相同偶数膨胀率（例如三层 `k=3, d=2`），输入网格会出现大量永远无法被中心点采样到的孔洞（**Gridding Effect**）；工程上应采用互质锯齿膨胀率（如 `d = [1, 2, 5]`）铺满全部像素。

### 2.3 转置卷积（Transposed Conv）为什么会产生棋盘格伪影？

在上方实验台切换至「转置卷积棋盘伪影」模式即可直观看到：当卷积核尺寸 $k$ **不能被上采样步长 $s$ 整除**（典型反例 `k=3, s=2`）时，输出网格相邻像素被不同数量的卷积核窗口覆盖（交替出现 `1×`、`2×`、`4×` 叠加），导致生成图像或合成音频自带周期性高频网格噪声。根治方案有二：
- 确保核大小是步长的整数倍（如 `k=4, s=2, p=1`）；
- 改用**最近邻/双线性插值上采样 + 标准 `3×3` 卷积**或无损通道重排 **`PixelShuffle`**（VAE 解码器主流方案）。

### 2.4 池化梯度路由与检测框亚像素对齐（RoIAlign）

- **池化反向传播**：`MaxPool` 仅把上游梯度传给窗口内唯一最大值坐标（其余置 `0`）；`AvgPool` 将梯度均分为 $1/k^2$；`SoftPool` 按局部指数权重平滑分配。
- **`RoI Pooling` vs `RoIAlign`**：早期两阶段检测器对浮点候选框坐标连续两次向下取整 `floor`，在 `stride=32` 特征图上 `0.8` 格截断即对应原图 **`25.6` 像素**的严重偏位；`RoIAlign` 彻底取消坐标量化，直接用四邻域双线性插值保留连续亚像素梯度。

---

## 3. PyTorch 与原生 Rust（Candle）对照实现

```python
import torch
import torch.nn as nn

class DepthwiseSeparableConv2d(nn.Module):
  def __init__(self, c_in: int, c_out: int, kernel_size: int = 3) -> None:
    super().__init__()
    pad = kernel_size // 2
    self.depthwise = nn.Conv2d(
      c_in, c_in, kernel_size, padding=pad, groups=c_in, bias=False
    )
    self.pointwise = nn.Conv2d(c_in, c_out, kernel_size=1, bias=False)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.pointwise(self.depthwise(x))
```

```rust
use candle_core::{Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, Module};

pub struct DepthwiseSeparableConv2d {
  depthwise: Conv2d,
  pointwise: Conv2d,
}

impl DepthwiseSeparableConv2d {
  pub fn new(dw_w: Tensor, pw_w: Tensor, c_in: usize, pad: usize) -> Self {
    let dw_cfg = Conv2dConfig { padding: pad, groups: c_in, ..Default::default() };
    let pw_cfg = Conv2dConfig::default();
    Self {
      depthwise: Conv2d::new(dw_w, None, dw_cfg),
      pointwise: Conv2d::new(pw_w, None, pw_cfg),
    }
  }

  pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
    let h = self.depthwise.forward(xs)?;
    self.pointwise.forward(&h)
  }
}
```
