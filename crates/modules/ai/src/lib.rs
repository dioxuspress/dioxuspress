//! Rust AI 内容板块与 LLM 模型架构可视化器。
//!
//! - 文章扫描与列表由共享的 [`board`] crate 提供（`assets/topics/ai/`）；
//! - [`visualizer::AiVisualizerPage`] 提供 `/ai/visualizer` 独立交互式模型架构解析页；
//! - [`register_components`] 向 `widgets` 全局注册 `<ModelArch ... />` MDX 嵌入组件。

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

pub mod arch;
pub mod atlas;
pub mod atlas_store;
#[cfg(feature = "server")]
pub mod audit;
pub mod component_labs;
pub mod component_labs_m2;
pub mod components_atlas;
pub mod config_guide;
pub mod diagram;
pub mod math_atlas;
pub mod math_labs;
pub mod math_labs_m2;
pub mod ml_atlas;
pub mod ml_labs;
pub mod safetensors_meta;
pub mod server;
pub mod tensor_graph;
pub mod visualizer;

use app_core::i18n::Language;
use board::{Board, BoardArticle, BoardIndex, FeaturedCrate};
use dioxus::prelude::*;

pub use atlas::{ModelDetailPage, ModelsAtlasPage};
pub use components_atlas::{AiComponentDetailPage, AiComponentsAtlasPage};
pub use config_guide::ModelConfigsGuidePage;
pub use math_atlas::{AiMathAtlasPage, AiMathDetailPage};
pub use ml_atlas::{AiMlAtlasPage, AiMlDetailPage};
pub use visualizer::AiVisualizerPage;

pub const BOARD: Board = Board {
  id: "ai",
  route: "/ai",
  subtopics: &["tensors", "inference", "llm", "tokenizers", "training", "embeddings"],
  featured_crates: &[
    FeaturedCrate { name: "candle", url: "https://github.com/huggingface/candle" },
    FeaturedCrate { name: "burn", url: "https://burn.dev" },
    FeaturedCrate { name: "tch", url: "https://github.com/LaurentMazare/tch-rs" },
    FeaturedCrate { name: "tokenizers", url: "https://github.com/huggingface/tokenizers" },
    FeaturedCrate { name: "ort", url: "https://ort.pyke.io" },
    FeaturedCrate { name: "safetensors", url: "https://github.com/huggingface/safetensors" },
  ],
  doc_slugs: &[
    "candle",
    "burn",
    "inference",
    "pytorch-to-candle",
    "pytorch-to-burn",
    "tenferro",
    "jax-to-tenferro",
  ],
  match_tags: &[
    "ai",
    "llm",
    "candle",
    "burn",
    "cubecl",
    "rag",
    "ai-agent",
    "ai agent",
    "yolo",
    "深度学习",
    "机器学习",
    "quantization",
    "fine-tuning",
    "diffusion",
    "huggingface",
    "onnx",
    "safetensors",
    "tokenizers",
    "openai",
    "vlm",
    "gpu 编程",
  ],
};

/// 启动期向 `widgets` 全局注册表注入 `<ModelArch ... />`、`<ComponentLab ... />`、`<MathLab ... />` 与 `<MlLab ... />` MDX 组件（服务端与客户端双端幂等调用）。
pub fn register_components() {
  widgets::register(Box::new(visualizer::ModelArchComponent));
  widgets::register(Box::new(component_labs::ComponentLabMdx));
  widgets::register(Box::new(math_labs::MathLabMdx));
  widgets::register(Box::new(ml_labs::MlLabMdx));
}

#[component]
pub fn AiIndexPage() -> Element {
  let lang = try_consume_context::<Signal<Language>>().map(|s| s()).unwrap_or_default();
  let cards = [
    (
      "/models",
      "border-indigo-200 dark:border-indigo-900/70 bg-linear-to-r from-indigo-50/90 via-white to-violet-50/80 dark:from-indigo-950/50 dark:via-slate-900 dark:to-slate-900 hover:border-indigo-400 dark:hover:border-indigo-600",
      "text-indigo-600 dark:text-indigo-400",
      "bg-indigo-600",
      "group-hover:text-indigo-600 dark:group-hover:text-indigo-400",
      "Model Atlas",
      "DeepSeek · YOLO · U-Net · SAM",
      lang.pick("全谱系 AI 模型架构图鉴", "Full-Spectrum AI Model Atlas"),
      lang.pick(
        "覆盖旗舰 LLM/VLM、YOLO 13 代、U-Net/SAM 与经典网络的多尺度可折叠张量流拓扑图。",
        "Multi-scale collapsible tensor flow graphs across LLMs, YOLO v1-26, U-Net, SAM, and classics.",
      ),
      lang.pick("浏览模型图鉴 →", "Browse Models →"),
    ),
    (
      "/ai/components",
      "border-emerald-200 dark:border-emerald-900/70 bg-linear-to-r from-emerald-50/90 via-white to-teal-50/80 dark:from-emerald-950/40 dark:via-slate-900 dark:to-slate-900 hover:border-emerald-400 dark:hover:border-emerald-600",
      "text-emerald-600 dark:text-emerald-400",
      "bg-emerald-600",
      "group-hover:text-emerald-600 dark:group-hover:text-emerald-400",
      "Operators Lab",
      "Conv · MLA · FlashAttn · MoE",
      lang.pick("组件与算子动态实验室", "Interactive Operators Lab"),
      lang.pick(
        "从卷积、池化、激活到 RoPE、MLA、Mamba 与 MoE 的纯 Rust WASM 动态交互与公式对照。",
        "Pure Rust WASM interactive visualizations and derivations across DL & LLM primitives.",
      ),
      lang.pick("进入算子实验室 →", "Explore Operators →"),
    ),
    (
      "/ai/math",
      "border-violet-200 dark:border-violet-900/70 bg-linear-to-r from-violet-50/90 via-white to-amber-50/70 dark:from-violet-950/40 dark:via-slate-900 dark:to-slate-900 hover:border-violet-400 dark:hover:border-violet-600",
      "text-violet-600 dark:text-violet-400",
      "bg-violet-600",
      "group-hover:text-violet-600 dark:group-hover:text-violet-400",
      "Math Proof Lab",
      "SVD · VJP · KL · SDE · PPO",
      lang.pick("AI 数学基石与色标推导", "AI Math Proof Lab"),
      lang.pick(
        "零跳步色标同步证明链配纯 Rust WASM 几何、散度、扩散流与强化对齐动态画布。",
        "Zero-skip color-synced proofs paired with live WASM geometry, divergence, flow, and RL labs.",
      ),
      lang.pick("进入数学实验室 →", "Open Math Lab →"),
    ),
    (
      "/ai/ml",
      "border-teal-200 dark:border-teal-900/70 bg-linear-to-r from-teal-50/90 via-white to-cyan-50/70 dark:from-teal-950/40 dark:via-slate-900 dark:to-slate-900 hover:border-teal-400 dark:hover:border-teal-600",
      "text-teal-600 dark:text-teal-400",
      "bg-teal-600",
      "group-hover:text-teal-600 dark:group-hover:text-teal-400",
      "Classic ML",
      "XGBoost · SVM · K-Means · PCA",
      lang.pick("传统机器学习交互图鉴", "Classic ML Interactive Atlas"),
      lang.pick(
        "决策树、随机森林、GBDT/XGBoost、SVM、回归、聚类与 PCA 的 2D 实时边界画布与 Rust 对照。",
        "Live 2D Rust WASM decision boundary, boosting, regression, K-Means, and PCA labs.",
      ),
      lang.pick("探索传统 ML 图鉴 →", "Explore Classic ML →"),
    ),
    (
      "/ai/visualizer",
      "border-slate-200 dark:border-slate-800 bg-linear-to-r from-slate-50/90 via-white to-sky-50/80 dark:from-slate-900 dark:via-slate-900 dark:to-sky-950/30 hover:border-indigo-400 dark:hover:border-indigo-600",
      "text-sky-600 dark:text-sky-400",
      "bg-sky-600",
      "group-hover:text-indigo-600 dark:group-hover:text-indigo-400",
      "Live Sandbox",
      "HF · ModelScope · Safetensors",
      lang.pick("任意仓库架构与显存解析器", "Live Repo & VRAM Inspector"),
      lang.pick(
        "输入任意 HuggingFace / ModelScope 模型或粘贴 config.json，零权重下载解析张量结构。",
        "Paste any HuggingFace/ModelScope repo or config.json to inspect tensor shapes live.",
      ),
      lang.pick("打开实时解析沙盒 →", "Open Visualizer →"),
    ),
  ];

  rsx! {
    div { class: "bg-white dark:bg-slate-950 pt-8",
      div { class: "max-w-7xl mx-auto px-4 sm:px-6 grid grid-cols-1 md:grid-cols-2 xl:grid-cols-5 gap-3.5",
        for (href, box_cls, accent_cls, pill_cls, hover_cls, pill, sub, h2_txt, p_txt, cta) in cards {
          Link {
            key: "{href}",
            to: href,
            class: "group flex flex-col justify-between gap-4 rounded-2xl border p-5 transition-all shadow-xs {box_cls}",
            div {
              div { class: "inline-flex items-center gap-2 text-xs font-bold uppercase tracking-wider {accent_cls}",
                span { class: "px-2 py-0.5 rounded-full text-white text-[10px] shrink-0 {pill_cls}", "{pill}" }
                span { class: "truncate", "{sub}" }
              }
              h2 { class: "mt-1.5 text-base font-extrabold text-slate-900 dark:text-white transition-colors {hover_cls}",
                "{h2_txt}"
              }
              p { class: "mt-1 text-xs sm:text-sm text-slate-600 dark:text-slate-400 leading-relaxed",
                "{p_txt}"
              }
            }
            span { class: "inline-flex items-center gap-1 text-xs font-bold {accent_cls}", "{cta}" }
          }
        }
      }
    }
    BoardIndex { board: BOARD }
  }
}

#[component]
pub fn AiArticlePage(slug: String) -> Element {
  rsx! { BoardArticle { board: BOARD, slug } }
}

#[cfg(test)]
mod tests {
  #[test]
  fn board_config_is_valid() {
    assert_eq!(super::BOARD.problems(), Vec::<String>::new());
  }
}
