---
slug: "forward-reverse-kl-and-k3"
title: "Forward KL（均值覆盖）vs Reverse KL（寻峰坍缩）与 Schulman k₃ 无偏低方差估计器"
title_en: "Forward KL vs Reverse KL Mode-Seeking & Schulman's Unbiased Low-Variance k3 Estimator"
pillar: "information_variational"
domains: ["llm", "rl_align", "diffusion"]
order: 30
lab_kind: "distribution_divergence"
lab_mode: "kl_modes_k3"
theorem_short: "D_KL(p‖q) vs D_KL(q‖p),  k₃(r) = (r - 1) - ln r ≥ 0,  E_q[k₃] = D_KL(q‖p)"
proof_steps_badge: "4-Step Zero-Skip Proof · SFT vs GRPO/RLHF"
summary: "彻底证透为什么监督微调（SFT）最小化 Forward KL 会撑宽分布造成多峰平均模糊，而强化对齐（RLHF/GRPO）最小化 Reverse KL 会引发寻峰坍缩；并零跳步证明 GRPO 核心正则项 k₃ = (r-1)-ln r 为何恒非负、严格无偏且在近策略区一阶方差相消。"
summary_en: "Proves why Forward KL causes mass-covering blurriness while Reverse KL drives mode-seeking collapse, and proves non-negativity, unbiasedness, and Taylor variance reduction of Schulman's k3 KL estimator."
tags: ["KL-Divergence", "Forward-KL", "Reverse-KL", "GRPO", "RLHF", "k3-Estimator"]
component_slugs: ["moe-routing-quant-and-speculative-decoding", "attention-mla-and-flash-online-softmax"]
model_keywords: ["deepseek", "qwen"]
---

## 1. 核心悖论：仅仅调换 $p$ 和 $q$ 的位置，优化行为为何截然相反？

给定多模态真实目标分布 $\textcolor{#4f46e5}{p(x)}$（例如一道数学题存在多种正确解法路径）与参数化模型分布 $q_\theta(x)$，Kullback-Leibler 散度定义为对数密度比的期望，但它**不满足对称性**：

$$\textcolor{#d97706}{D_{\mathrm{KL}}(p \parallel q_\theta)} = \int \textcolor{#4f46e5}{p(x)} \ln \frac{\textcolor{#4f46e5}{p(x)}}{q_\theta(x)} \,\mathrm{d}x \;\neq\; \textcolor{#e11d48}{D_{\mathrm{KL}}(q_\theta \parallel p)} = \int q_\theta(x) \ln \frac{q_\theta(x)}{\textcolor{#4f46e5}{p(x)}} \,\mathrm{d}x$$

在上方交互画布中点击切换 **「最小化 Forward KL」** 与 **「最小化 Reverse KL」**，可以直接观察到两大生成范式的根本分水岭：
- **Forward KL $\textcolor{#d97706}{D_{\mathrm{KL}}(p \parallel q_\theta)}$（SFT 最大似然预训练）——零避免（Zero-Avoiding / 盖峰）**：在任何真实数据出现的位置 $\textcolor{#4f46e5}{p(x) > 0}$，一旦模型赋予 $q_\theta(x) \to 0$，则 $\ln(p/q_\theta) \to +\infty$ 触发无穷大惩罚！单峰模型被迫撑大方差盖住所有峰，甚至把最高概率中心放在两峰之间 $\textcolor{#4f46e5}{p(x) \approx 0}$ 的无人区（表现为均值模糊或安全套话）。
- **Reverse KL $\textcolor{#e11d48}{D_{\mathrm{KL}}(q_\theta \parallel p)}$（RLHF / 策略蒸馏）——零强制（Zero-Forcing / 寻峰）**：积分权重换成了模型自身的 $q_\theta(x)$。只要在真实分布低谷 $\textcolor{#4f46e5}{p(x) \approx 0}$ 处果断令 $q_\theta(x) = 0$，由极限 $\lim_{q\to 0^+} q \ln q = 0$ 可知惩罚瞬间归零；模型宁可彻底抛弃其余模式，也要死死锁住其中一个高密度单峰（表现为输出锐利但多样性坍缩 / 熵崩塌）。

---

## 2. 零跳步色标锚定证明链：矩匹配性质与 Schulman $k_3$ 估计器定理

### Step 1：证明最小化 Forward KL 对高斯族严格等价于矩匹配（均值覆盖）

设模型为高斯分布 $q_{\mu,\sigma^2}(x) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$，展开 $\textcolor{#d97706}{D_{\mathrm{KL}}(p \parallel q_{\mu,\sigma^2})}$ 并省去与 $(\mu, \sigma^2)$ 无关的负熵项 $\int p \ln p\,\mathrm{d}x$：

$$J(\mu, \sigma^2) = -\int \textcolor{#4f46e5}{p(x)} \ln q_{\mu,\sigma^2}(x)\,\mathrm{d}x = \frac{1}{2}\ln(2\pi\sigma^2) + \frac{1}{2\sigma^2}\int \textcolor{#4f46e5}{p(x)}(x - \mu)^2\,\mathrm{d}x$$

分别对 $\mu$ 与 $\sigma^2$ 求偏导令其为零：

$$\frac{\partial J}{\partial \mu} = -\frac{1}{\sigma^2}\int \textcolor{#4f46e5}{p(x)}(x - \mu)\,\mathrm{d}x = 0 \implies \textcolor{#d97706}{\mu^* = \mathbb{E}_p[x]}$$

$$\frac{\partial J}{\partial (\sigma^2)} = \frac{1}{2\sigma^2} - \frac{1}{2\sigma^4}\mathbb{E}_p\bigl[(x - \mu^*)^2\bigr] = 0 \implies \textcolor{#d97706}{\sigma^{*2} = \operatorname{Var}_p(x)}$$

当 $\textcolor{#4f46e5}{p(x)}$ 是对称双峰 $\frac{1}{2}\mathcal{N}(-\Delta, \sigma_0^2) + \frac{1}{2}\mathcal{N}(+\Delta, \sigma_0^2)$ 时，$\textcolor{#d97706}{\mu^* = 0}$（正好落在两峰中间的空白低谷！），且方差被拉宽至 $\textcolor{#d97706}{\sigma^{*2} = \sigma_0^2 + \Delta^2}$。

### Step 2：为什么强化学习里朴素采样估计器 $k_1 = -\ln r$ 方差过大？

在大模型后训练（如 [Qwen3](/models/qwen3-30b-a3b) 与 `DeepSeek-R1` 采用的 `GRPO`，配合 [采样与稀疏解码算子](/ai/components/moe-routing-quant-and-speculative-decoding)）中，我们需要从当前策略 $x \sim \pi_\theta$（记作 $q$）仅采样少量回答来估计与参考策略 $\pi_{\mathrm{ref}}$（记作 $p$）的逆向散度 $\textcolor{#e11d48}{D_{\mathrm{KL}}(q \parallel p)}$。记单样本重要性比率为 $r(x) = \frac{p(x)}{q(x)} > 0$。

朴素定义给出 $k_1(r) = \ln\frac{q(x)}{p(x)} = -\ln r$。虽然 $\mathbb{E}_{x\sim q}[k_1(r)] = D_{\mathrm{KL}}(q \parallel p)$ 无偏，但当 $r > 1$（样本在参考模型下概率更高）时 **$k_1(r) < 0$ 为负值**！在 $q \approx p$（即 $r = 1 + \delta, \; |\delta| \ll 1$）的近策略训练主航道上，泰勒展开显示：

$$k_1(1 + \delta) = -\ln(1 + \delta) = -\delta + \frac{1}{2}\delta^2 - O(\delta^3)$$

一阶线性项 $-\delta$ 虽然期望为 $0$，但单样本剧烈正负震荡，贡献了高达 $\operatorname{Var}(k_1) = \mathbb{E}[\delta^2] + O(\delta^3)$ 的一阶噪声方差！

### Step 3：构造控制变量（Control Variate）零跳步导出 $k_3$ 无偏性与恒非负性

注意到密度比 $r(x) = \frac{p(x)}{q(x)}$ 在分布 $q$ 下的期望恒等于 $1$：

$$\mathbb{E}_{x \sim q}\bigl[r(x) - 1\bigr] = \int q(x)\left(\frac{p(x)}{q(x)} - 1\right)\mathrm{d}x = \int p(x)\,\mathrm{d}x - \int q(x)\,\mathrm{d}x = 1 - 1 = 0$$

将这个**期望严格为 $0$、一阶展开恰好为 $+\delta$ 的零均值控制变量 $(r - 1)$** 直接加到 $k_1 = -\ln r$ 上，即得 John Schulman 著名的 **$k_3$ 估计器**：

$$\textcolor{#059669}{k_3(r) = (r - 1) - \ln r}$$

1. **严格无偏性**：由期望线性性，$\mathbb{E}_{x\sim q}[\textcolor{#059669}{k_3(r)}] = 0 + \mathbb{E}_{x\sim q}[-\ln r] = \textcolor{#e11d48}{D_{\mathrm{KL}}(q \parallel p)}$；
2. **逐样本恒非负性**：利用基本凸不等式 $\ln r \le r - 1$（函数 $g(r) = r - 1 - \ln r$ 满足 $g'(r) = 1 - \frac{1}{r}$，$g''(r) = \frac{1}{r^2} > 0$，唯一极小值在 $r = 1$ 处取 $g(1) = 0$），对一切 $r > 0$ 恒有 **$\textcolor{#059669}{k_3(r) \ge 0}$**，彻底杜绝了奖励黑客通过刷负 KL 项骗取高分的漏洞！

### Step 4：二阶泰勒展开证明近策略区方差高阶衰减

将 $r = 1 + \delta$ 代入 $\textcolor{#059669}{k_3(1 + \delta)}$ 做精确泰勒展开：

$$\textcolor{#059669}{k_3(1 + \delta)} = \delta - \left(\delta - \frac{1}{2}\delta^2 + \frac{1}{3}\delta^3 - \cdots\right) = \textcolor{#059669}{\frac{1}{2}\delta^2} - O(\delta^3)$$

一阶噪声项 $\pm\delta$ **正负 100% 精确抵消**！方差从 $k_1$ 的 $O(\mathbb{E}[\delta^2])$ 骤降至高阶无穷小 $O(\mathbb{E}[\delta^4])$（例如当 $|\delta| \approx 0.1$ 时，$k_3$ 的采样方差比 $k_1$ 缩小近 **100 倍**）$\qquad \blacksquare$

---

## 3. GRPO / RLHF 工业级防溢出 Log-Space 实现对照

在工程代码中，切勿先对对数概率差做 `exp` 再减 1，而应在日志域计算 $\Delta \ell = \ln \pi_{\mathrm{ref}}(y|x) - \ln \pi_\theta(y|x)$ 并使用高精度 `expm1`：

```python
import torch

def grpo_k3_kl_penalty(logp_policy: torch.Tensor, logp_ref: torch.Tensor) -> torch.Tensor:
  # log_r = ln(pi_ref / pi_theta)
  log_r = logp_ref - logp_policy
  # expm1(log_r) = r - 1，在 log_r ≈ 0 附近避免浮点有效数字相减抵消
  return torch.expm1(log_r) - log_r
```

```rust
/// 数值稳定计算 Schulman k3 = (exp(log_r) - 1) - log_r >= 0
pub fn k3_kl_from_log_ratio(logp_policy: f64, logp_ref: f64) -> f64 {
  let log_r = (logp_ref - logp_policy).clamp(-20.0, 20.0);
  (log_r.exp_m1() - log_r).max(0.0)
}
```
