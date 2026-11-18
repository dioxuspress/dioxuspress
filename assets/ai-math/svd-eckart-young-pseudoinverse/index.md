---
slug: "svd-eckart-young-pseudoinverse"
title: "奇异值分解（SVD）几何三步曲与 Eckart-Young 最优低秩近似定理"
title_en: "Singular Value Decomposition (SVD) & Eckart-Young Optimal Low-Rank Theorem"
pillar: "linear_geometry"
domains: ["llm", "diffusion", "quant_opt"]
order: 10
lab_kind: "linear_transform_2d"
lab_mode: "svd"
theorem_short: "A = U·Σ·V⊤,  min_{rank(B)≤r} ‖A - B‖_F = √(∑_{i>r} σ_i²)"
proof_steps_badge: "4-Step Zero-Skip Proof · SVD & LoRA"
summary: "从对称半正定格拉姆矩阵谱分解出发，零跳步构造任意实矩阵的正交旋转—轴向缩放—正交旋转三步几何分解，并利用正交不变范数与外尔交错不等式严格证明截断前 r 项奇异值即为全局最优低秩逼近。"
summary_en: "Constructs the three-step rotation-scaling-rotation SVD for arbitrary matrices and proves Eckart-Young-Mirsky low-rank optimality powering LoRA and DeepSeek MLA."
tags: ["SVD", "Eckart-Young", "Low-Rank", "LoRA", "MLA", "Condition-Number"]
component_slugs: ["attention-mla-and-flash-online-softmax", "moe-routing-quant-and-speculative-decoding"]
model_keywords: ["deepseek", "qwen"]
---

## 1. 物理直觉：为什么任何线性层本质上都只是“旋转 + 拉伸 + 旋转”？

神经网络中的权重矩阵 $A \in \mathbb{R}^{m \times n}$ 通常既非方阵也非对称阵，无法直接做特征值分解。但奇异值分解（SVD）揭示了一个极其纯粹的几何事实：**世界上不存在复杂的线性扭曲——任何矩阵 $A$ 对单位球面的作用，都可以严格拆解为三步正交与轴向动作**（对应上方交互画布阶段滑块 $s: 0 \to 3$）：

$$A = U \textcolor{#059669}{\Sigma} \textcolor{#4f46e5}{V^\top} = \sum_{i=1}^{\min(m,n)} \textcolor{#059669}{\sigma_i} u_i \textcolor{#4f46e5}{v_i^\top}$$

1. **第 1 步（靛蓝 $\textcolor{#4f46e5}{V^\top}$ 输入基对齐旋转）**：正交矩阵 $V^\top$ 将输入向量 $x$ 投影到右奇异正交基 $\{v_i\}$ 上，由于 $\|V^\top x\|_2 = \|x\|_2$，单位圆只旋转不形变；
2. **第 2 步（翠绿 $\textcolor{#059669}{\Sigma}$ 坐标轴独立拉伸）**：对角非负矩阵 $\Sigma = \operatorname{diag}(\sigma_1, \dots, \sigma_p)$ 沿各正交轴按奇异值 $\sigma_1 \ge \sigma_2 \ge \dots \ge 0$ 独立伸缩，将圆拉成轴对齐超椭球，长短轴之比即为**条件数** $\textcolor{#d97706}{\kappa(A) = \sigma_1 / \sigma_{\min}}$；
3. **第 3 步（琥珀 $U$ 输出空间定向旋转）**：左奇异正交矩阵 $U$ 将轴对齐椭球刚体旋转嵌入 $\mathbb{R}^m$ 输出流形。

---

## 2. 零跳步色标锚定证明链：Eckart-Young-Mirsky 最优低秩定理

**定理陈述**：设 $A \in \mathbb{R}^{m \times n}$ 的奇异值为 $\sigma_1 \ge \dots \ge \sigma_p \ge 0$，记秩-$r$ 截断和为 $A_r = \sum_{i=1}^r \sigma_i u_i v_i^\top$。则对任意满足 $\operatorname{rank}(B) \le r$ 的矩阵 $B \in \mathbb{R}^{m \times n}$，恒有：

$$\min_{\operatorname{rank}(B) \le r} \|A - B\|_F = \|A - A_r\|_F = \textcolor{#e11d48}{\sqrt{\sum_{i=r+1}^p \sigma_i^2}}, \qquad \min_{\operatorname{rank}(B) \le r} \|A - B\|_2 = \textcolor{#e11d48}{\sigma_{r+1}}$$

### Step 1：Frobenius 范数的正交不变性对角化

利用迹恒等式 $\|M\|_F^2 = \operatorname{tr}(M^\top M)$ 及正交矩阵 $U^\top U = I_m, V^\top V = I_n$，对任意正交变换 $U^\top M V$ 计算平方范数：

$$\|U^\top M V\|_F^2 = \operatorname{tr}\!\bigl((U^\top M V)^\top (U^\top M V)\bigr) = \operatorname{tr}(V^\top M^\top U U^\top M V) = \operatorname{tr}(M^\top M V V^\top) = \|M\|_F^2$$

代入 $M = A - A_r = U(\textcolor{#059669}{\Sigma} - \textcolor{#059669}{\Sigma_r})\textcolor{#4f46e5}{V^\top}$，立得截断矩阵 $A_r$ 达到了上界误差：

$$\|A - A_r\|_F = \|\Sigma - \Sigma_r\|_F = \textcolor{#e11d48}{\sqrt{\sum_{i=r+1}^p \sigma_i^2}}$$

### Step 2：构造零空间与主奇异子空间的维数交集（谱范数下界）

现在证**不存在任何秩 $\le r$ 的矩阵 $B$ 能做得比 $A_r$ 更好**。任取 $\operatorname{rank}(B) \le r$，由秩—零化度定理：

$$\dim \ker(B) = n - \operatorname{rank}(B) \ge n - r$$

与此同时，考察前 $r+1$ 个右奇异向量张成的输入子空间 $\mathcal{W}_{r+1} = \operatorname{span}\{\textcolor{#4f46e5}{v_1, v_2, \dots, v_{r+1}}\}$，其维数为 $\dim \mathcal{W}_{r+1} = r + 1$。由子空间维数公式：

$$\dim\bigl(\ker(B) \cap \mathcal{W}_{r+1}\bigr) = \dim\ker(B) + \dim\mathcal{W}_{r+1} - \dim\bigl(\ker(B) + \mathcal{W}_{r+1}\bigr) \ge (n - r) + (r + 1) - n = 1$$

因此交集中**必存在至少一个单位向量** $w \in \ker(B) \cap \mathcal{W}_{r+1}$（即 $\|w\|_2 = 1$ 且 $Bw = 0$），并可写为正交展开 $w = \sum_{i=1}^{r+1} c_i \textcolor{#4f46e5}{v_i}$，满足 $\sum_{i=1}^{r+1} c_i^2 = 1$。

### Step 3：瑞利商放缩锁定谱范数下界 $\sigma_{r+1}$

用该单位测试向量 $w$ 作用在残差矩阵 $(A - B)$ 上，因 $Bw = 0$ 且左奇异向量 $\{u_i\}$ 两两标准正交：

$$\begin{aligned}
\|A - B\|_2^2 &\ge \|(A - B)w\|_2^2 = \|Aw\|_2^2 = \left\|\sum_{i=1}^{r+1} c_i \textcolor{#059669}{\sigma_i} u_i\right\|_2^2 \\
&= \sum_{i=1}^{r+1} c_i^2 \textcolor{#059669}{\sigma_i^2} \ge \textcolor{#e11d48}{\sigma_{r+1}^2} \sum_{i=1}^{r+1} c_i^2 = \textcolor{#e11d48}{\sigma_{r+1}^2}
\end{aligned}$$

这证明了任何秩 $\le r$ 矩阵的谱范数逼近误差不可能小于 $\textcolor{#e11d48}{\sigma_{r+1}}$，而 $A_r$ 恰好取到等号！

### Step 4：Weyl 奇异值交错不等式推广至全阶 Frobenius 范数

对任意 $k \ge 1$，将 Step 3 应用于 $A - B$ 的第 $k$ 奇异值：取 $X = A - B$ 与秩 $\le k-1$ 的截断 $X_{k-1}$，则 $B + X_{k-1}$ 的秩至多为 $r + k - 1$。再次套用 Step 3 结论可得外尔（Weyl）奇异值不等式：

$$\sigma_k(A - B) = \|(A - B) - X_{k-1}\|_2 = \|A - (B + X_{k-1})\|_2 \ge \textcolor{#e11d48}{\sigma_{r+k}(A)}$$

将所有 $k = 1, \dots, p - r$ 两端平方求和，即证完 Frobenius 全局最优性：

$$\|A - B\|_F^2 = \sum_{k=1}^{p} \sigma_k^2(A - B) \ge \sum_{k=1}^{p-r} \sigma_k^2(A - B) \ge \textcolor{#e11d48}{\sum_{i=r+1}^p \sigma_i^2(A)} = \|A - A_r\|_F^2 \qquad \blacksquare$$

---

## 3. 大模型工程落点与数值验算对照

1. **`DeepSeek-V2/V3` 多头潜在注意力（MLA）**：将原本形状为 $[B, S, n_h d_h]$ 的巨型 KV 缓存压缩进低维潜向量 $c_t^{KV} = h_t W^{DKV} \in \mathbb{R}^{d_c}$（$d_c \ll n_h d_h$），正是利用权重激活谱快速衰减的 Eckart-Young 结构（参见算子实验台 [MLA 矩阵吸收与 FlashAttention](/ai/components/attention-mla-and-flash-online-softmax) 与整机拆解 [DeepSeek-V3](/models/deepseek-v3)）。
2. **`LoRA` 参数高效微调**：微调增量 $\Delta W = B A$（$B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}$）将优化空间约束在秩-$r$ 流形上，用极小参数捕获任务梯度的前 $r$ 个主奇异方向。

```python
import torch

A = torch.tensor([[1.75, 0.65], [-0.35, 0.85]], dtype=torch.float64)
U, S, Vh = torch.linalg.svd(A)
A_rank1 = S[0] * torch.outer(U[:, 0], Vh[0, :])
err_frob = torch.linalg.norm(A - A_rank1, ord="fro")
assert torch.allclose(err_frob, S[1], atol=1e-12)
```

```rust
pub fn rank1_eckart_young_residual_2x2(a: [[f64; 2]; 2]) -> (f64, f64) {
  let e = 0.5 * (a[0][0] + a[1][1]);
  let f = 0.5 * (a[0][0] - a[1][1]);
  let g = 0.5 * (a[1][0] + a[0][1]);
  let h = 0.5 * (a[1][0] - a[0][1]);
  let (q, r) = ((e * e + h * h).sqrt(), (f * f + g * g).sqrt());
  let (sigma1, sigma2) = (q + r, (q - r).abs());
  (sigma1, sigma2) // ||A - A_1||_F == sigma2 恒精确成立
}
```
