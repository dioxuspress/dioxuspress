---
slug: decision-tree
title: "决策树 (CART / ID3)"
title_en: "Decision Tree (CART / ID3)"
family: tree_ensemble
lab_kind: tree_boost
lab_mode: decision_tree
time_complexity: "O(N·D·log N)"
space_complexity: "O(Nodes) / 推理 O(Depth)"
needs_scaling: false
interpretability: high
rust_crates: "linfa-trees · smartcore"
formula_short: "Gini(D) = 1 - Σ p_k² · ΔGini = Gini(D) - Σ (|D_v|/|D|) Gini(D_v)"
summary: "通过递归轴对齐特征切分将特征空间划分为纯度递增的矩形区域，天然无需特征归一化且完全白盒可解释，是一切随机森林、GBDT 与 XGBoost 集成树的原子基学习器。"
summary_en: "Recursively partitions feature space via axis-aligned splits maximizing Gini or Entropy impurity reduction; scale-invariant, white-box interpretable, and the building block of Random Forests and GBDT."
tags:
  - decision-tree
  - cart
  - gini
  - tabular-ml
related_models:
  - perceptron
  - mlp
related_math:
  - shannon-entropy-ce-perplexity
order: 1
---

## 1. 算法核心思想与分裂准则

决策树（CART: Classification and Regression Trees）通过贪心递归寻找最优特征 $j$ 与切分阈值 $t$，将父节点样本集 $D$ 一分为二（$D_L = \{x_j \le t\}$、$D_R = \{x_j > t\}$），使子节点的不纯度加权和最小化：

$$\operatorname{Gini}(D) = 1 - \sum_{k=1}^{K} p_k^2, \qquad \Delta \operatorname{Gini}(j, t) = \operatorname{Gini}(D) - \frac{|D_L|}{|D|}\operatorname{Gini}(D_L) - \frac{|D_R|}{|D|}\operatorname{Gini}(D_R)$$

- **为什么无需特征归一化**：因为切分只依赖单个特征的**序关系（$x_j \le t$）**，任何严格单调变换（标准化、对数缩放）都不会改变分裂点排序。
- **过拟合与剪枝**：不加深度限制时，单棵树可把每个训练样本切到独立叶节点（低偏差、极高方差）。工程中通过限制 `max_depth`、`min_samples_leaf`、代价复杂度剪枝（CCP）或升级为随机森林/梯度提升树控制方差。

## 2. Rust 生态代码对照（`linfa-trees`）

```rust
use linfa::prelude::*;
use linfa_trees::{DecisionTree, SplitQuality};
use ndarray::{array, Array1, Array2};

fn train_cart(x: Array2<f64>, y: Array1<usize>) -> DecisionTree<f64, usize> {
  let dataset = Dataset::new(x, y);
  DecisionTree::params()
    .split_quality(SplitQuality::Gini)
    .max_depth(Some(4))
    .min_weight_split(4.0)
    .fit(&dataset)
    .expect("valid tabular dataset")
}
```
