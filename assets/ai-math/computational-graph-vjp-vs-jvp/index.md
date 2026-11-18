---
slug: "computational-graph-vjp-vs-jvp"
title: "自动微分核心定理：前向切向量（JVP）vs 反向伴随传播（Backprop VJP）复杂度与激活驻留证明"
title_en: "Automatic Differentiation: Forward JVP vs Reverse-Mode Backprop VJP Complexity Proof"
pillar: "matrix_calculus_autograd"
domains: ["llm", "cnn_audio", "quant_opt"]
order: 20
lab_kind: "autograd_and_loss"
lab_mode: "vjp_graph"
theorem_short: "v̄_i = ∑_{j∈child(i)} (∂v_j / ∂v_i)⊤ v̄_j,  Cost(VJP) = O(1)·Cost(Forward)"
proof_steps_badge: "4-Step Zero-Skip Proof · JVP vs VJP"
summary: "在有向无环计算图（DAG）上严格对比雅可比—向量积（JVP）与向量—雅可比积（VJP）的矩阵结合律顺序，证明为什么十亿参数神经网络对标量损失求全梯度只需 1 次反向扫描，以及为什么训练显存必然被前向中间激活主导。"
summary_en: "Proves via matrix associativity why reverse-mode VJP computes all parameter gradients of a scalar loss in O(1) sweeps and why forward activations dominate training VRAM."
tags: ["Autograd", "Backprop", "VJP", "JVP", "Jacobian", "Activation-Memory"]
component_slugs: ["activations-gelu-silu-swiglu", "rmsnorm-and-adaln-zero"]
model_keywords: ["deepseek", "qwen"]
---

## 1. 问题的提出：为什么万亿参数大模型能算得起全梯度？

假设模型参数向量 $\theta \in \mathbb{R}^n$（例如 $n = 10^{11}$）映射为单个标量训练损失 $L = f(\theta) \in \mathbb{R}^1$（输出维数 $m = 1$）。若采用数值有限差分 $\frac{f(\theta + \varepsilon e_i) - f(\theta - \varepsilon e_i)}{2\varepsilon}$，需要执行 **$2n = 2\times 10^{11}$ 次**完整前向传播——哪怕每次前向仅需 1 毫秒，算完 1 步梯度也要 **6.3 年**！

而反向传播（Reverse-Mode Automatic Differentiation，即 **VJP**）仅用**约等于 2~3 倍单次前向传播的浮点算力**，就能一次性精确算出全部 $10^{11}$ 个偏导数（无任何截断误差）。全部秘密都藏在**矩阵乘法结合律的括号打法**里。

---

## 2. 零跳步色标锚定证明链：JVP vs VJP 复杂度与显存驻留定理

设复合映射 $f: \mathbb{R}^n \to \mathbb{R}^m$ 由 $K$ 层基本算子串联而成：$\textcolor{#4f46e5}{v_0 = x} \to \textcolor{#4f46e5}{v_1 = g_1(v_0)} \to \dots \to \textcolor{#4f46e5}{v_K = g_K(v_{K-1})} = y$，记第 $k$ 步局部雅可比矩阵为 $\textcolor{#059669}{J_k = \frac{\partial v_k}{\partial v_{k-1}} \in \mathbb{R}^{d_k \times d_{k-1}}}$。由多元链式法则，总雅可比矩阵为：

$$J_f = \frac{\partial y}{\partial x} = \textcolor{#059669}{J_K J_{K-1} \cdots J_2 J_1} \in \mathbb{R}^{m \times n}$$

### Step 1：前向模式（JVP）从右向左结合——复杂度正比于输入维数 $\textcolor{#d97706}{O(n)}$

给定输入空间切向量 $\dot{x} \in \mathbb{R}^n$，定义切伴随递推 $\dot{v}_0 = \dot{x}, \; \dot{v}_k = \textcolor{#059669}{J_k} \dot{v}_{k-1}$。括号从右向左打：

$$\operatorname{JVP}(x, \dot{x}) = \textcolor{#059669}{J_K} \Bigl(\textcolor{#059669}{J_{K-1}} \bigl(\cdots (\textcolor{#059669}{J_1} \dot{x})\bigr)\Bigr) \in \mathbb{R}^m$$

每一步都是“矩阵 $\times$ 列向量”，不显式构造稠密雅可比矩阵，单遍耗时与前向求值同阶。但若要拼出完整的 $J_f \in \mathbb{R}^{m \times n}$，必须依次令 $\dot{x} = e_1, e_2, \dots, e_n$ 跑满 **$\textcolor{#d97706}{n}$ 遍前向传播**！当 $n = 10^{11}, m = 1$ 时彻底不可行（仅在 $n=1, m \gg 1$ 如求解参数化曲线切线或海森向量积 $Hv$ 时高效）。

### Step 2：反向模式（VJP）从左向右结合——标量损失下仅需 $\textcolor{#d97706}{O(m) = O(1)}$ 遍！

给定输出空间余切向量（伴随种子）$\textcolor{#e11d48}{\bar{y} \in \mathbb{R}^m}$，定义节点伴随向量 $\textcolor{#e11d48}{\bar{v}_k = \left(\frac{\partial y}{\partial v_k}\right)^\top \bar{y} \in \mathbb{R}^{d_k}}$。将链式乘积转置并**从左向右打括号**：

$$\textcolor{#e11d48}{\bar{x}} = J_f^\top \textcolor{#e11d48}{\bar{y}} = \textcolor{#059669}{J_1^\top} \Bigl(\textcolor{#059669}{J_2^\top} \bigl(\cdots (\textcolor{#059669}{J_K^\top} \textcolor{#e11d48}{\bar{y}})\bigr)\Bigr) \in \mathbb{R}^n$$

对于深度学习标量损失 $y = L \in \mathbb{R}^1$（即 $m = 1$），只需在终点注入标量种子 **$\textcolor{#e11d48}{\bar{v}_K = \frac{\partial L}{\partial L} = 1}$**，自右向左单遍递推 $\textcolor{#e11d48}{\bar{v}_{k-1} = J_k^\top \bar{v}_k}$，**仅用 1 遍反向回传**即可同时得到全部 $n$ 个参数的精确梯度 $\nabla_\theta L = \textcolor{#e11d48}{\bar{\theta}}$！

### Step 3：一般有向无环图（DAG）多分支梯度汇合公式

当中间节点 $\textcolor{#4f46e5}{v_i}$ 同时作为多个下游节点 $j \in \operatorname{child}(i)$ 的输入时（如残差跳连、门控激活 [SwiGLU](/ai/components/activations-gelu-silu-swiglu) 或权重共享正则支路，见上方交互画布中的 $w$ 同时流入乘法节点与 $\frac{\lambda}{2}w^2$），全微分法则给出精确的多路累加律：

$$\mathrm{d}L = \sum_{j \in \operatorname{child}(i)} \textcolor{#e11d48}{\bar{v}_j^\top} \frac{\partial v_j}{\partial v_i} \mathrm{d}v_i \implies \textcolor{#e11d48}{\bar{v}_i} = \sum_{j \in \operatorname{child}(i)} \textcolor{#059669}{\left(\frac{\partial v_j}{\partial v_i}\right)^\top} \textcolor{#e11d48}{\bar{v}_j}$$

这解释了 PyTorch / Candle 自动求导引擎中“**前向分叉（Fork）在反向严格对偶为梯度相加（Add）**”的代数本质。

### Step 4：为什么训练显存远大于推理显存？（激活驻留不可避免性）

观察局部反向算子 $\textcolor{#e11d48}{\bar{v}_{k-1}} = \textcolor{#059669}{J_k(v_{k-1})^\top} \textcolor{#e11d48}{\bar{v}_k}$：
- 对于线性层 $v_k = W v_{k-1}$，虽然对激活求导 $\frac{\partial v_k}{\partial v_{k-1}} = W$ 只依赖权重，但**对权重本身求导** $\textcolor{#e11d48}{\bar{W} = \bar{v}_k \textcolor{#4f46e5}{v_{k-1}^\top}}$ 必须乘上前向输入激活 $\textcolor{#4f46e5}{v_{k-1}}$！
- 对于非线性层（如 $v_k = \operatorname{SiLU}(u)$ 或 [RMSNorm](/ai/components/rmsnorm-and-adaln-zero)），导数 $\textcolor{#059669}{\sigma(u)\bigl(1 + u(1-\sigma(u))\bigr)}$ 直接依赖前向原值 $\textcolor{#4f46e5}{u}$。

因此，推理期算完第 $k$ 层即可立刻释放 $\textcolor{#4f46e5}{v_{k-1}}$（显存复杂度 $O(1)$ 层）；而反向传播必须将全部 $L$ 层前向激活驻留至反向扫描抵达该层为止（显存复杂度 $O(L)$）。若每隔 $k$ 层仅保存一个检查点并在反向时段内重算一次（切换到上方画布第 2 模式验证），峰值激活显存降为：

$$M(k) = \frac{L}{k} + k \ge 2\sqrt{L} \quad \text{（当且仅当 } k^* = \sqrt{L} \text{ 时取得全局极小值）} \qquad \blacksquare$$

---

## 3. PyTorch 自定义 Autograd 与纯 Rust 微缩反向求导核对

```python
import torch

x = torch.tensor(1.4, dtype=torch.float64, requires_grad=True)
w = torch.tensor(0.9, dtype=torch.float64, requires_grad=True)
u = w * x
h = torch.nn.functional.silu(u)
loss = 0.5 * (h - 0.5) ** 2 + 0.5 * 0.2 * (w ** 2)
loss.backward()
print(f"w.grad = {w.grad.item():+.6f}, x.grad = {x.grad.item():+.6f}")
```

```rust
pub fn silu_reg_vjp(x: f64, w: f64, y: f64, lambda: f64) -> (f64, f64, f64) {
  let u = w * x;
  let sig = 1.0 / (1.0 + (-u).exp());
  let h = u * sig;
  let r = h - y;
  let loss = 0.5 * r * r + 0.5 * lambda * w * w;
  let silu_grad = sig * (1.0 + u * (1.0 - sig));
  let bar_u = r * silu_grad;
  (loss, bar_u * x + lambda * w, bar_u * w)
}
```
