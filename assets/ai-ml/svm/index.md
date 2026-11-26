---
slug: svm
title: "支持向量机 (SVM · 最大间隔与核技巧)"
title_en: "Support Vector Machine (Max-Margin & Kernel Trick)"
family: linear_kernel
lab_kind: boundary_2d
lab_mode: svm_rbf
time_complexity: "O(N²·D ~ N³)"
space_complexity: "O(N_sv·D) 稀疏支持向量"
needs_scaling: true
interpretability: low
rust_crates: "linfa-svm · smartcore"
formula_short: "min ½‖w‖² + C Σ ξ_i,  K_RBF(x, z) = exp(-γ‖x - z‖²)"
summary: "在特征空间中寻找几何间隔最大化的分离超平面，并借由拉格朗日对偶与核技巧（RBF/多项式核）隐式映射至无穷维希尔伯特空间求解非线性边界，解完全由少数支持向量决定。"
summary_en: "Finds the maximum-margin separating hyperplane and leverages dual formulations with kernel tricks (RBF/Polynomial) to solve nonlinear boundaries using only sparse support vectors."
tags:
  - svm
  - kernel-trick
  - max-margin
  - hinge-loss
related_models:
  - perceptron
  - mlp
related_math:
  - lagrange-kkt-bregman-mirror
order: 7
---

## 1. 从感知机到最大间隔超平面

单层 [Perceptron (1958)](/models/perceptron) 只要找到任意一条分开两类的直线就会停止，而支持向量机（Vapnik & Cortes, 1995）进一步要求超平面距两侧最近样本的**几何间隔 $\frac{2}{\|\mathbf{w}\|_2}$ 最大化**：

$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2}\|\mathbf{w}\|_2^2 + C\sum_{i=1}^{N}\xi_i \quad \text{s.t.} \quad y_i(\mathbf{w}^\top \mathbf{x}_i + b) \ge 1 - \xi_i,\; \xi_i \ge 0$$

利用拉格朗日对偶（见 [Lagrange / KKT 与 Bregman 对偶](/ai/math/lagrange-kkt-bregman-mirror)），决策函数只依赖样本间内积，只需将内积替换为正定核函数 $K(\mathbf{x}_i, \mathbf{x}) = \exp(-\gamma\|\mathbf{x}_i - \mathbf{x}\|_2^2)$ 即可高效切分双月牙与同心圆等非线性流形，且最终只有落在间隔边界上或内部的支持向量拥有 $\alpha_i > 0$。

## 2. Rust 生态代码对照（`linfa-svm`）

```rust
use linfa::prelude::*;
use linfa_svm::Svm;
use ndarray::{Array1, Array2};

fn train_rbf_svm(x: Array2<f64>, y: Array1<bool>) {
  let ds = Dataset::new(x, y);
  let _model = Svm::<_, bool>::params()
    .pos_neg_weights(3.0, 3.0)
    .gaussian_kernel(2.5)
    .fit(&ds)
    .expect("svm convergence");
}
```
