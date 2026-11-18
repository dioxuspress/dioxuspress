---
slug: "fourier-convolution-aliasing"
title: "离散傅里叶复平面缠绕正交性、循环卷积定理与奈奎斯特混叠证明"
title_en: "Complex DFT Winding Orthogonality, Convolution Theorem & Nyquist Aliasing Proof"
pillar: "signal_spectral_graph"
domains: ["cnn_audio", "llm", "diffusion"]
order: 230
lab_kind: "spectral_and_graph"
lab_mode: "dft_winding"
theorem_short: "ℱ{x ⊛ h}[k] = X[k]·H[k],  f_alias = |f - f_s·round(f / f_s)|"
proof_steps_badge: "4-Step Zero-Skip Proof · FFT & Aliasing"
summary: "从单位根等比级数正交消去引理出发，零跳步证明离散傅里叶变换复平面缠绕质心原理、O(N log N) 快速循环卷积对角化定理以及欠采样高频折叠伪装（Aliasing）的周期延拓根源。"
summary_en: "Proves roots-of-unity orthogonality, circular convolution diagonalization via DFT, and Nyquist-Shannon aliasing folding from Poisson periodic spectrum replication."
tags: ["DFT", "FFT", "Convolution-Theorem", "Nyquist", "Aliasing", "Roots-of-Unity"]
component_slugs: ["audio-stft-conformer-ctc-and-rvq", "conv-and-depthwise-separable", "lstm-deltanet-and-mamba-ssm"]
model_keywords: ["qwen", "deepseek"]
---

## 1. 几何直觉：为什么把波形“缠绕”在单位圆上就能挑出隐藏频率？

给定长度为 $N$ 的离散序列 $x[n]$，离散傅里叶变换（DFT）表面上看是一串枯燥的复指数求和，但在几何上它只做了一件极其直观的事（对应上方交互画布左半区的缠绕曲线）：**以探测角速度 $\omega_k = \frac{2\pi k}{N}$ 将信号幅度作为极径顺时针缠绕在复平面单位圆上，并计算整条缠绕曲线的质心** $\textcolor{#4f46e5}{\mu_k = \frac{1}{N} X[k]}$：

$$\textcolor{#4f46e5}{X[k]} = \sum_{n=0}^{N-1} x[n] \textcolor{#d97706}{e^{-i \frac{2\pi}{N} k n}}, \qquad k = 0, 1, \dots, N-1$$

- **非共振相消**：当探测频率与信号真实频率不一致时，波峰均匀散落在圆周各个象限，复向量对称抵消，质心死死钉在原点附近；
- **共振偏心弹出**：一旦探测频率精确命中信号内含谐波，所有波峰每一圈都转回同一相位角，质心模长 $\textcolor{#4f46e5}{|X[k]|/N}$ 猛然偏离原点弹出峰值！

---

## 2. 零跳步色标锚定证明链：单位根正交性、卷积定理与混叠折叠

### Step 1：$N$ 次单位根等比级数正交消去引理

记本原单位根 $W_N = e^{-i 2\pi / N}$。考察任意两个离散频率索引 $k, m \in \{0, \dots, N-1\}$ 的内积和 $S = \sum_{n=0}^{N-1} e^{i \frac{2\pi}{N}(k - m)n}$：

1. 若 $k \equiv m \pmod N$，每一项均为 $e^0 = 1$，故 $S = N$；
2. 若 $k \not\equiv m \pmod N$，公比 $q = e^{i \frac{2\pi}{N}(k - m)} \neq 1$，但满足 $q^N = e^{i 2\pi (k - m)} = 1$。直接套用有限等比数列求和公式（零近似）：

$$\sum_{n=0}^{N-1} \textcolor{#d97706}{e^{i \frac{2\pi}{N}(k - m)n}} = \frac{1 - q^N}{1 - q} = \frac{1 - 1}{1 - q} = 0 \implies \frac{1}{N}\sum_{n=0}^{N-1} e^{-i\frac{2\pi}{N}kn} e^{i\frac{2\pi}{N}mn} = \delta_{k,m}$$

这严格证明了 $\{ \frac{1}{\sqrt{N}} W_N^{kn} \}_{k=0}^{N-1}$ 构成 $\mathbb{C}^N$ 上的**标准酉正交基**，逆变换 $x[n] = \frac{1}{N}\sum_{k=0}^{N-1} \textcolor{#4f46e5}{X[k]} e^{i\frac{2\pi}{N}kn}$ 恒精确重构原信号。

### Step 2：循环卷积对角化定理 $\mathcal{F}\{x \circledast h\} = X \odot H$

定义模 $N$ 循环卷积 $y[n] = (x \circledast h)[n] = \sum_{m=0}^{N-1} x[m]\, h[(n - m) \bmod N]$。对其取第 $k$ 个 DFT 分量并交换有限求和次序：

$$\begin{aligned}
Y[k] &= \sum_{n=0}^{N-1} \left(\sum_{m=0}^{N-1} x[m]\, h[(n - m) \bmod N]\right) e^{-i\frac{2\pi}{N}kn} \\
&= \sum_{m=0}^{N-1} x[m] \textcolor{#d97706}{e^{-i\frac{2\pi}{N}km}} \sum_{n=0}^{N-1} h[(n - m) \bmod N]\, \textcolor{#d97706}{e^{-i\frac{2\pi}{N}k(n - m)}}
\end{aligned}$$

令内层指标作双射平移 $r = (n - m) \bmod N$。因复指数满足周期性 $e^{-i\frac{2\pi}{N}k(r + qN)} = e^{-i\frac{2\pi}{N}kr}$，当 $n$ 跑遍 $\{0, \dots, N-1\}$ 时，$r$ 恰好不重不漏跑遍 $\{0, \dots, N-1\}$！内外两重求和彻底解耦为两个独立标量之积：

$$Y[k] = \left(\sum_{m=0}^{N-1} x[m] e^{-i\frac{2\pi}{N}km}\right) \left(\sum_{r=0}^{N-1} h[r] e^{-i\frac{2\pi}{N}kr}\right) = \textcolor{#4f46e5}{X[k]} \cdot \textcolor{#059669}{H[k]}$$

由此将时域 $O(N^2)$ 长序列滑动卷积化归为 `FFT → 逐点复数乘 → IFFT` 的 $O(N \log N)$ 快速算法。

### Step 3：离散采样为什么在网格点上无法区分 $f$ 与 $f + m f_s$？（混叠恒等式）

设连续余弦信号真实频率为 $f_1$，以采样率 $f_s = 1 / \Delta t$（奈奎斯特截止频率 $\textcolor{#059669}{f_{\text{Nyq}} = f_s / 2}$）在 $t_n = n / f_s$ 处等距采样。对任意整数倍偏移 $m \in \mathbb{Z}$，考察折叠频率 $\textcolor{#e11d48}{f_{\text{alias}} = |f_1 - m f_s|}$ 在完全相同的采样时刻 $t_n$ 处的取值：

$$\begin{aligned}
\cos(2\pi (f_1 - m f_s) t_n) &= \cos\!\left(2\pi f_1 \frac{n}{f_s} - 2\pi m n\right) \\
&= \cos\!\left(2\pi f_1 \frac{n}{f_s}\right)\underbrace{\cos(2\pi m n)}_{=1} + \sin\!\left(2\pi f_1 \frac{n}{f_s}\right)\underbrace{\sin(2\pi m n)}_{=0} = \cos(2\pi f_1 t_n)
\end{aligned}$$

### Step 4：唯一主带折叠公式与抗混叠滤波器设计准则

取整数 $m^* = \operatorname{round}(f_1 / f_s)$，则 $\textcolor{#e11d48}{f_{\text{alias}} = |f_1 - m^* f_s|} \le \textcolor{#059669}{\frac{f_s}{2}}$ 必落入主基带 $[0, f_s/2]$：
- 当 $f_1 \le \textcolor{#059669}{f_s / 2}$ 时，$m^* = 0$，$\textcolor{#e11d48}{f_{\text{alias}}} = f_1$，信号唯一无损恢复；
- 一旦欠采样 $f_1 > \textcolor{#059669}{f_s / 2}$，高频能量在所有采样点上与低频波 $\textcolor{#e11d48}{f_{\text{alias}}}$ **逐点 100% 相等**，事后任何算法都无法剥离，必须在下采样（Stride-2 Pool / 音频降采样）前先施加低通抗混叠滤波！ $\blacksquare$

---

## 3. 深度学习算子落点与双语数值验证

1. **长序列状态空间与快速卷积**：在长卷积与结构化状态空间算子（参见 [Mamba 选择性状态空间与线性注意力](/ai/components/lstm-deltanet-and-mamba-ssm) 及 [标准卷积与深度可分离卷积](/ai/components/conv-and-depthwise-separable)）中，固定核长序列卷积依赖频域对角化加速。
2. **语音前端与下采样抗混叠**：声学特征重采样与加窗谱提取（参见 [STFT 语谱图、Conformer 与 RVQ 残差量化](/ai/components/audio-stft-conformer-ctc-and-rvq)）严格受奈奎斯特截止界约束。

```python
import torch

N = 64
x = torch.randn(N, dtype=torch.float64)
h = torch.randn(N, dtype=torch.float64)
# 时域循环卷积 vs FFT 频域逐点相乘
y_circ = torch.stack([sum(x[m] * h[(n - m) % N] for m in range(N)) for n in range(N)])
y_fft = torch.fft.ifft(torch.fft.fft(x) * torch.fft.fft(h)).real
assert torch.allclose(y_circ, y_fft, atol=1e-12)
```

```rust
pub fn aliased_baseband_freq(f_true: f64, f_sample: f64) -> f64 {
  let fs = f_sample.max(1e-9);
  (f_true - fs * (f_true / fs).round()).abs()
}
```
