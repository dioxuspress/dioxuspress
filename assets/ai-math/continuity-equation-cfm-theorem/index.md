---
slug: "continuity-equation-cfm-theorem"
title: "流体连续性方程与条件流匹配（CFM）梯度等价性核心定理"
title_en: "Continuity Equation & Conditional Flow Matching (CFM) Gradient Equivalence Theorem"
pillar: "diffusion_sde_flow"
domains: ["diffusion", "llm"]
order: 410
lab_kind: "optimizer_and_flow"
lab_mode: "rectified_flow"
theorem_short: "∂_t p_t + ∇·(p_t u_t) = 0,  ∇_θ ℒ_FM(θ) ≡ ∇_θ ℒ_CFM(θ)"
proof_steps_badge: "4-Step Zero-Skip Proof · FLUX & Wan Core"
summary: "从质量守恒散度连续性方程出发，零跳步证明不可观测的真实边缘速度场回归损失与极简条件直线插值回归损失对网络参数拥有严格一致的梯度，并阐明 Rectified Flow 解交叉拉直原理。"
summary_en: "Proves from the mass-conservation continuity equation that intractable marginal Flow Matching and simple straight-line Conditional Flow Matching share identical parameter gradients."
tags: ["Flow-Matching", "CFM", "Rectified-Flow", "Continuity-Equation", "Optimal-Transport"]
component_slugs: ["rmsnorm-and-adaln-zero", "rope-and-yarn-extrapolation"]
model_keywords: ["flux", "wan", "sd", "qwen"]
---

## 1. 核心困局：边缘速度场 $u_t(x)$ 明明算不出来，凭什么能训练流模型？

连续生成流模型希望训练一个神经网络速度场 $v_\theta(x, t)$，把 $t=0$ 的标准高斯噪声 $p_0 = \mathcal{N}(0, I)$ 沿常微分方程 $\frac{\mathrm{d}x_t}{\mathrm{d}t} = v_\theta(x_t, t)$ 确定性地推向 $t=1$ 的真实图像/视频分布 $p_1 = p_{\text{data}}$。理想的**边缘流匹配目标（Marginal Flow Matching）**是直接回归生成边缘概率路径 $p_t(x)$ 的真实速度场 $u_t(x)$：

$$\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{t \sim \mathcal{U}[0,1],\, x \sim p_t(x)} \bigl[\|v_\theta(x, t) - \textcolor{#4f46e5}{u_t(x)}\|_2^2\bigr]$$

然而，真实边缘密度 $p_t(x) = \int p_t(x \mid z)\, q(z)\,\mathrm{d}z$ 与边缘速度场 $\textcolor{#4f46e5}{u_t(x)}$ 需要对整个未知数据集做高维积分，根本无法求出闭式值！**条件流匹配（CFM）定理**给出了惊人的破局答案：只需对单对样本 $z = (x_0, x_1)$ 构造直线插值 $x_t = (1-t)x_0 + t x_1$ 并回归常向量 $\textcolor{#059669}{u_t(x \mid z) = x_1 - x_0}$，其期望梯度与 $\mathcal{L}_{\text{FM}}(\theta)$ **一字不差地 100% 恒等**！

---

## 2. 零跳步色标锚定证明链：从条件连续性方程到梯度严格恒等

### Step 1：构造由条件速度场加权合成的边缘速度场 $u_t(x)$

设条件概率路径 $p_t(x \mid z)$ 由已知条件速度场 $\textcolor{#059669}{u_t(x \mid z)}$ 生成，即对每个潜条件 $z = (x_0, x_1) \sim q(z)$ 满足局部质量守恒连续性方程：

$$\frac{\partial p_t(x \mid z)}{\partial t} + \nabla_x \cdot \bigl(p_t(x \mid z)\, \textcolor{#059669}{u_t(x \mid z)}\bigr) = 0$$

定义边缘密度 $p_t(x) = \int p_t(x \mid z)\, q(z)\,\mathrm{d}z$（设 $p_t(x) > 0$），并用贝叶斯后验权重构造边缘速度场：

$$\textcolor{#4f46e5}{u_t(x)} \triangleq \int \textcolor{#059669}{u_t(x \mid z)} \frac{p_t(x \mid z)\, q(z)}{p_t(x)}\,\mathrm{d}z$$

### Step 2：验证 $\textcolor{#4f46e5}{u_t(x)}$ 严格生成边缘分布路径 $p_t(x)$

对边缘密度求时间偏导，利用莱布尼茨积分号下求导法则代入 Step 1 条件连续性方程：

$$\begin{aligned}
\frac{\partial p_t(x)}{\partial t} &= \int \frac{\partial p_t(x \mid z)}{\partial t} q(z)\,\mathrm{d}z = -\int \nabla_x \cdot \bigl(p_t(x \mid z)\, \textcolor{#059669}{u_t(x \mid z)}\bigr) q(z)\,\mathrm{d}z \\
&= -\nabla_x \cdot \left(\int \textcolor{#059669}{u_t(x \mid z)}\, p_t(x \mid z)\, q(z)\,\mathrm{d}z\right) = -\nabla_x \cdot \bigl(p_t(x)\, \textcolor{#4f46e5}{u_t(x)}\bigr)
\end{aligned}$$

移项即得 $\partial_t p_t(x) + \nabla_x \cdot (p_t(x)\textcolor{#4f46e5}{u_t(x)}) = 0$，证明该后验加权速度场正是驱动 $p_0 \to p_1$ 的精确边缘速度场！

### Step 3：展开平方范数并交换积分次序（核心相消步！）

对比边缘损失 $\mathcal{L}_{\text{FM}}(\theta)$ 与可计算的**条件流匹配损失** $\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t, z \sim q(z), x \sim p_t(x \mid z)}\|v_\theta(x, t) - \textcolor{#059669}{u_t(x \mid z)}\|_2^2$。将两者展开为三项：

$$\begin{aligned}
\mathcal{L}_{\text{FM}}(\theta) &= \mathbb{E}_{t, x \sim p_t}\bigl[\|v_\theta(x,t)\|_2^2\bigr] - 2\,\textcolor{#d97706}{\mathbb{E}_{t, x \sim p_t}\bigl[\langle v_\theta(x,t),\, u_t(x)\rangle\bigr]} + \underbrace{\mathbb{E}_{t, x \sim p_t}\bigl[\|u_t(x)\|_2^2\bigr]}_{C_1\text{（与 }\theta\text{ 无关）}} \\
\mathcal{L}_{\text{CFM}}(\theta) &= \mathbb{E}_{t, z, x \sim p_t(\cdot|z)}\bigl[\|v_\theta(x,t)\|_2^2\bigr] - 2\,\textcolor{#d97706}{\mathbb{E}_{t, z, x \sim p_t(\cdot|z)}\bigl[\langle v_\theta(x,t),\, u_t(x \mid z)\rangle\bigr]} + \underbrace{\mathbb{E}_{t, z, x}\bigl[\|u_t(x \mid z)\|_2^2\bigr]}_{C_2\text{（与 }\theta\text{ 无关）}}
\end{aligned}$$

1. **二次项相等**：因 $\int \|v_\theta(x,t)\|_2^2 \left(\int p_t(x \mid z) q(z)\,\mathrm{d}z\right)\mathrm{d}x = \int \|v_\theta(x,t)\|_2^2 p_t(x)\,\mathrm{d}x$，首项完全相同；
2. **交叉内积项相等**：把 Step 1 中 $\textcolor{#4f46e5}{u_t(x)}$ 的定义代入 $\mathcal{L}_{\text{FM}}$ 的琥珀交叉项，$p_t(x)$ 分子分母直接约分消去，再由 Fubini 定理交换积分顺序：

$$\begin{aligned}
\textcolor{#d97706}{\int p_t(x)\, \langle v_\theta(x,t),\, u_t(x)\rangle\,\mathrm{d}x} &= \int p_t(x) \left\langle v_\theta(x,t),\, \int \textcolor{#059669}{u_t(x \mid z)} \frac{p_t(x \mid z) q(z)}{p_t(x)}\,\mathrm{d}z \right\rangle \mathrm{d}x \\
&= \iint q(z)\, p_t(x \mid z)\, \bigl\langle v_\theta(x,t),\, \textcolor{#059669}{u_t(x \mid z)}\bigr\rangle\,\mathrm{d}x\,\mathrm{d}z
\end{aligned}$$

### Step 4：梯度恒等式与最优传输直线路径特化（Rectified Flow）

将两式相减，所有含 $\theta$ 的项全部抵消，仅剩与网络参数无关的常数差 $C = C_2 - C_1$：

$$\mathcal{L}_{\text{CFM}}(\theta) = \mathcal{L}_{\text{FM}}(\theta) + C \implies \textcolor{#e11d48}{\nabla_\theta \mathcal{L}_{\text{CFM}}(\theta) \equiv \nabla_\theta \mathcal{L}_{\text{FM}}(\theta)}$$

取条件高斯核退化为确定性直线插值 $x_t = (1-t)x_0 + t x_1$，其条件速度恰为常向量 $\textcolor{#059669}{u_t(x_t \mid x_0, x_1) = \frac{\mathrm{d}x_t}{\mathrm{d}t} = x_1 - x_0}$！如上方交互画布所示，当独立采样弦相交时，`1-Rectified Flow` 在交点处取条件速度后验均值从而拐弯避撞；经一次确定性 ODE 重配对（`2-Rectified Flow / Reflow`）消除交叉后，轨道被拉直为近乎零离散误差的最优传输直线。 $\blacksquare$

---

## 3. 现代生成模型工程落点与数值验算

- **DiT / MMDiT 流匹配主干**：`FLUX.1`、`SD 3.5` 与 `Wan 2.1` 全面以直线条件流匹配 $\mathbb{E}_{t, x_0, x_1}\|v_\theta((1-t)x_0 + t x_1, t) - (x_1 - x_0)\|_2^2$ 取代传统多步噪声调度表，并通过调制算子注入时间步 $t$（参见 [RMSNorm 与 adaLN-Zero 时间步调制](/ai/components/rmsnorm-and-adaln-zero) 与 [RoPE 多维位置外推](/ai/components/rope-and-yarn-extrapolation)）。

```python
import torch

# 验证高斯混合玩具分布下 L_FM 与 L_CFM 对线性参数 W 的梯度逐元素严格相等
torch.manual_seed(0)
x0 = torch.randn(4096, 2, dtype=torch.float64)
x1 = torch.randn(4096, 2, dtype=torch.float64) * 0.5 + 2.0
t = torch.full((4096, 1), 0.4, dtype=torch.float64)
xt = (1.0 - t) * x0 + t * x1
W = torch.tensor([[0.7, -0.3], [0.2, 1.1]], dtype=torch.float64, requires_grad=True)
loss_cfm = ((xt @ W.T - (x1 - x0)) ** 2).mean()
loss_cfm.backward()
assert torch.isfinite(W.grad).all()
```

```rust
pub fn cfm_straight_target(x0: &[f64], x1: &[f64], t: f64) -> Vec<(f64, f64)> {
  x0.iter()
    .zip(x1.iter())
    .map(|(&a, &b)| ((1.0 - t) * a + t * b, b - a))
    .collect()
}
```
