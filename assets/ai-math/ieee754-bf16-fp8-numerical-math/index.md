---
slug: "ieee754-bf16-fp8-numerical-math"
title: "计算机浮点算术：IEEE-754 位域解剖、BF16 梯度吞没与 Log-Sum-Exp 防溢出证明"
title_en: "IEEE-754 Floating-Point Anatomy, BF16 Swamping & Log-Sum-Exp Stability Proof"
pillar: "rl_alignment_precision"
domains: ["quant_opt", "llm", "rl_align"]
order: 480
lab_kind: "rl_and_bit_precision"
lab_mode: "ieee754_bits"
theorem_short: "|Δw| < ½·ulp(w) = 2^{e-p} ⇒ fl(w + Δw) = w,  LSE(z) = m + ln ∑ exp(z_i - m)"
proof_steps_badge: "4-Step Zero-Skip Proof · Mixed Precision"
summary: "从 IEEE-754 符号—偏置指数—隐含前导 1 尾数代数表达式出发，严格推导 FP16 溢出/下溢边界、BF16 权重累加微小梯度 100% 停滞（Swamping）的末位单位定理，以及 Log-Sum-Exp 与 Welford 在线方差无损递推恒等式。"
summary_en: "Derives IEEE-754 representation bounds, proves why BF16 gradient updates below 1/2 ulp(w) stagnate without FP32 Master Weights, and proves Log-Sum-Exp and Welford stability."
tags: ["IEEE-754", "BF16", "FP16", "FP8", "Master-Weights", "Log-Sum-Exp", "Welford"]
component_slugs: ["attention-mla-and-flash-online-softmax", "moe-routing-quant-and-speculative-decoding", "rmsnorm-and-adaln-zero"]
model_keywords: ["deepseek", "qwen"]
---

## 1. 硬件真相：为什么同样占 16 比特，`FP16` 动不动爆 `NaN`，而 `BF16` 必须配 `FP32` 主权重？

实数在现代 GPU / TPU 寄存器中并非连续稠密集合，而是按 **IEEE-754 二进制科学计数法**离散排布的有限格点（可在上方 32 位翻转实验台中亲手点击任意比特验证）。对含 $1$ 位符号 $s$、$E$ 位无符号指数 $e$（偏置常数 $B = 2^{E-1} - 1$）与 $M$ 位尾数 $m = b_1 b_2 \dots b_M$ 的正规化浮点数（$1 \le e \le 2^E - 2$），其精确有理数值为：

$$x = (-1)^s \cdot \textcolor{#4f46e5}{2^{e - B}} \cdot \left(1 + \textcolor{#059669}{\sum_{k=1}^{M} b_k 2^{-k}}\right)$$

| 格式 | 位域 `(S-E-M)` | 偏置 $B$ | 最大正规数 $x_{\max}$ | 最小正正规数 | 机器相对精度 $\epsilon_{\text{mach}} = 2^{-M}$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`FP32`** | `1 - 8 - 23` | $127$ | $\approx 3.40 \times 10^{38}$ | $\approx 1.18 \times 10^{-38}$ | $2^{-23} \approx 1.19 \times 10^{-7}$ |
| **`BF16`** | `1 - 8 - 7` | $127$ | $\approx 3.39 \times 10^{38}$ | $\approx 1.18 \times 10^{-38}$ | $2^{-7} \approx 7.81 \times 10^{-3}$ |
| **`FP16`** | `1 - 5 - 10` | $15$ | $\textcolor{#d97706}{65504}$ | $2^{-14} \approx 6.10 \times 10^{-5}$ | $2^{-10} \approx 9.77 \times 10^{-4}$ |
| **`FP8 E4M3`** | `1 - 4 - 3` | $7$ | $\textcolor{#e11d48}{448}$ | $2^{-6} = 0.015625$ | $2^{-3} = 0.125$ |

---

## 2. 零跳步色标锚定证明链：梯度吞没定理、Log-Sum-Exp 与 Welford 恒等式

### Step 1：精确计算 `FP16` 上溢死线 `65504` 与下溢盲区

在 `FP16 (1-5-10)` 中，$E = 5 \implies B = 2^4 - 1 = 15$。全 `1` 指数 $e = 31$ 被保留用于编码 `±Inf` 与 `NaN`，故最大可用正规指数为 $e_{\max} = 30$（即无偏指数 $30 - 15 = 15$），此时 10 位尾数全取 `1`：

$$x_{\max}^{\text{FP16}} = 2^{30 - 15} \cdot \left(1 + \sum_{k=1}^{10} 2^{-k}\right) = 2^{15} \cdot \left(2 - 2^{-10}\right) = 32768 \times \frac{2047}{1024} = \textcolor{#d97706}{65504}$$

一旦注意力未缩放点积或激活平方和超过 `65504`（例如上方预设按钮 `70000`），`FP16` 立即溢出为 `Inf`，紧接着 `Inf - Inf` 或 `0 * Inf` 传染为全网 `NaN`；而 `BF16` 保留完整 8 位指数（$e_{\max} = 254, x_{\max} \approx 3.39 \times 10^{38}$），从根源上免疫激活值上溢与微小梯度下溢！

### Step 2：末位单位（ULP）与大数吃小数（Swamping）严格判据——为什么必须保留 `FP32 Master Weights`？

设当前权重落在双曲区间 $w \in [\textcolor{#4f46e5}{2^k}, 2^{k+1})$ 内。由于 `BF16` 仅有 $M = 7$ 位显式尾数，该区间内相邻两个可表示浮点数之间的栅格间距（Unit in the Last Place）为：

$$\operatorname{ulp}_{\text{BF16}}(w) = \textcolor{#4f46e5}{2^k} \cdot \textcolor{#059669}{2^{-7}} = 2^{k - 7}$$

在标准最近偶数舍入（Round-to-Nearest-Even）规则下，若优化器单步更新量 $\Delta w = -\eta g$ 的绝对值不及半个栅格宽度，舍入算子 $\operatorname{fl}_{\text{BF16}}(\cdot)$ 必将精确退回原值：

$$|\Delta w| < \frac{1}{2}\operatorname{ulp}_{\text{BF16}}(w) = \textcolor{#e11d48}{2^{k - 8}} \implies \operatorname{fl}_{\text{BF16}}(w + \Delta w) \equiv w$$

代入典型权重尺度 $w = 1.0$（$k = 0$）与大模型后期学习率 $\eta = 10^{-4}$、梯度 $g \approx 1.0$：此时 $|\Delta w| = 10^{-4} \ll \textcolor{#e11d48}{2^{-8} \approx 3.906 \times 10^{-3}}$！如果在纯 `BF16` 参数上直接做 `w -= lr * g`，**每一步梯度更新都会被 100% 抹平丢弃，模型彻底停止学习**！唯有在优化器状态中维护 `FP32 Master Weights`（$\frac{1}{2}\operatorname{ulp}_{\text{FP32}}(1.0) = 2^{-24} \approx 5.96 \times 10^{-8} \ll 10^{-4}$），微小的每步更新才能在低 16 位尾数中逐轮累积并进位到高 16 位前向权重中。

### Step 3：Log-Sum-Exp 与 Online Softmax 最大值平移零溢出证明

直接计算 $\ln \sum_{i=1}^V e^{z_i}$ 当 logit $z_i > 89$ 时连 `FP32` 也会溢出（$e^{89} > 3.4 \times 10^{38}$）。令全局最大值 $m = \max_j z_j$，提取公因子 $e^m > 0$：

$$\operatorname{LSE}(z) = \ln \left(\sum_{i=1}^V e^{m} \cdot e^{z_i - m}\right) = \ln(e^m) + \ln \sum_{i=1}^V e^{z_i - m} = m + \ln \underbrace{\sum_{i=1}^V e^{z_i - m}}_{\in [1, V]}$$

因为对所有 $i$ 恒有 $z_i - m \le 0 \implies e^{z_i - m} \in (0, 1]$，**上溢风险被 100% 根除**；又因至少存在一个下标 $j^*$ 满足 $z_{j^*} = m \implies e^{0} = 1$，括号内总和严格落在闭区间 $[1, V]$ 内，**对数真数下溢归零风险同样被 100% 根除**！

### Step 4：Welford 单遍无灾难性相消方差递推恒等式

朴素公式 $\sum x_i^2 - \frac{1}{n}(\sum x_i)^2$ 在均值远大于标准差时两个相近巨数相减会损失全部有效尾数。记前 $n$ 项均值 $\mu_n = \mu_{n-1} + \frac{x_n - \mu_{n-1}}{n}$ 及离差平方和 $M_{2,n} = \sum_{i=1}^n (x_i - \mu_n)^2$，利用 $(n-1)(\mu_n - \mu_{n-1}) = x_n - \mu_n$ 展开差值：

$$\begin{aligned}
M_{2,n} - M_{2,n-1} &= (x_n - \mu_n)^2 + \sum_{i=1}^{n-1} \bigl[(x_i - \mu_n)^2 - (x_i - \mu_{n-1})^2\bigr] \\
&= (x_n - \mu_n)^2 + (\mu_{n-1} - \mu_n)\sum_{i=1}^{n-1}(2x_i - \mu_n - \mu_{n-1}) \\
&= (x_n - \mu_n)^2 + (n-1)(\mu_n - \mu_{n-1})^2 = (x_n - \mu_n)\bigl[(x_n - \mu_n) + (\mu_n - \mu_{n-1})\bigr] \\
&= \textcolor{#059669}{(x_n - \mu_{n-1})(x_n - \mu_n)} \qquad \blacksquare
\end{aligned}$$

---

## 3. 大模型算子与量化训练落点验证

- **FlashAttention 在线归约与 `DeepSeek-V3` FP8 块级缩放**：分块注意力复用 Step 3 在线最大值重缩放（参见 [MLA 与 FlashAttention 在线 Softmax](/ai/components/attention-mla-and-flash-online-softmax)），而 [DeepSeek-V3](/models/deepseek-v3) 的 `FP8 E4M3` 训练通过 `1x128` 激活块缩放因子把动态范围实时拉回 $[-448, 448]$ 甜点区（参见 [MoE 路由、FP8/AWQ 量化与投机解码](/ai/components/moe-routing-quant-and-speculative-decoding)）。

```python
import torch

# 验证纯 BF16 权重 1.0 累加 1000 次 -1e-4 更新纹丝不动，而 FP32 Master Weight 精确前进 -0.1
w_bf16 = torch.tensor(1.0, dtype=torch.bfloat16)
w_master = torch.tensor(1.0, dtype=torch.float32)
for _ in range(1000):
  w_bf16 = w_bf16 - torch.tensor(1e-4, dtype=torch.bfloat16)
  w_master = w_master - 1e-4
assert float(w_bf16) == 1.0
assert abs(float(w_master.to(torch.bfloat16)) - 0.8984375) < 1e-6
```

```rust
pub fn welford_variance(xs: &[f64]) -> (f64, f64) {
  let (mut mean, mut m2) = (0.0, 0.0);
  for (idx, &x) in xs.iter().enumerate() {
    let n = (idx + 1) as f64;
    let delta = x - mean;
    mean += delta / n;
    m2 += delta * (x - mean);
  }
  let var = if xs.is_empty() { 0.0 } else { m2 / (xs.len() as f64) };
  (mean, var)
}
```
