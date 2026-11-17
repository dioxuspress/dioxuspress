//! Rust AI 内容板块与 LLM 模型架构可视化器。
//!
//! - 文章扫描与列表由共享的 [`board`] crate 提供（`assets/topics/ai/`）；
//! - [`visualizer::AiVisualizerPage`] 提供 `/ai/visualizer` 独立交互式模型架构解析页；
//! - [`register_components`] 向 `widgets` 全局注册 `<ModelArch ... />` MDX 嵌入组件。

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

pub mod arch;
pub mod atlas;
pub mod atlas_store;
pub mod component_labs;
pub mod components_atlas;
pub mod diagram;
pub mod safetensors_meta;
pub mod server;
pub mod visualizer;

use app_core::i18n::Language;
use board::{Board, BoardArticle, BoardIndex, FeaturedCrate};
use dioxus::prelude::*;

pub use atlas::{ModelDetailPage, ModelsAtlasPage};
pub use components_atlas::{AiComponentDetailPage, AiComponentsAtlasPage};
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
};

/// 启动期向 `widgets` 全局注册表注入 `<ModelArch ... />` 与 `<ComponentLab ... />` MDX 组件（服务端与客户端双端幂等调用）。
pub fn register_components() {
  widgets::register(Box::new(visualizer::ModelArchComponent));
  widgets::register(Box::new(component_labs::ComponentLabMdx));
}

#[component]
pub fn AiIndexPage() -> Element {
  let lang = try_consume_context::<Signal<Language>>().map(|s| s()).unwrap_or_default();
  rsx! {
    div { class: "bg-white dark:bg-slate-950 pt-8",
      div { class: "max-w-6xl mx-auto px-4 sm:px-6 grid grid-cols-1 md:grid-cols-3 gap-4",
        Link {
          to: "/models",
          class: "group flex flex-col justify-between gap-4 rounded-2xl border border-indigo-200 dark:border-indigo-900/70 bg-linear-to-r from-indigo-50/90 via-white to-violet-50/80 dark:from-indigo-950/50 dark:via-slate-900 dark:to-slate-900 p-5 hover:border-indigo-400 dark:hover:border-indigo-600 transition-all shadow-xs",
          div {
            div { class: "inline-flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-indigo-600 dark:text-indigo-400",
              span { class: "px-2 py-0.5 rounded-full bg-indigo-600 text-white text-[10px]", "Model Atlas" }
              span { "DeepSeek · Qwen3 · FLUX · Wan" }
            }
            h2 { class: "mt-1.5 text-lg font-extrabold text-slate-900 dark:text-white group-hover:text-indigo-600 dark:group-hover:text-indigo-400 transition-colors",
              {lang.pick("主流大模型架构图鉴", "Flagship Model Architecture Atlas")}
            }
            p { class: "mt-1 text-sm text-slate-600 dark:text-slate-400",
              {lang.pick(
                "精选主流开源旗舰模型官方配置，支持多尺度收起与展开层段、视觉塔、稀疏专家簇与内部算子矩阵。",
                "Curated flagship open-weight models with multi-scale collapsible layer stacks, vision towers, and MoE subgraphs."
              )}
            }
          }
          span { class: "inline-flex items-center gap-1 text-xs font-bold text-indigo-600 dark:text-indigo-400",
            {lang.pick("浏览模型图鉴 →", "Browse Model Atlas →")}
          }
        }

        Link {
          to: "/ai/components",
          class: "group flex flex-col justify-between gap-4 rounded-2xl border border-emerald-200 dark:border-emerald-900/70 bg-linear-to-r from-emerald-50/90 via-white to-teal-50/80 dark:from-emerald-950/40 dark:via-slate-900 dark:to-slate-900 p-5 hover:border-emerald-400 dark:hover:border-emerald-600 transition-all shadow-xs",
          div {
            div { class: "inline-flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-emerald-600 dark:text-emerald-400",
              span { class: "px-2 py-0.5 rounded-full bg-emerald-600 text-white text-[10px]", "Operators Lab" }
              span { "Conv · Pool · Norm · RoPE · SwiGLU" }
            }
            h2 { class: "mt-1.5 text-lg font-extrabold text-slate-900 dark:text-white group-hover:text-emerald-600 dark:group-hover:text-emerald-400 transition-colors",
              {lang.pick("组件与算子动态实验室", "Interactive Operators & Primitives Lab")}
            }
            p { class: "mt-1 text-sm text-slate-600 dark:text-slate-400",
              {lang.pick(
                "从卷积、池化、激活函数到归一化、位置外推与大模型原语的纯 Rust WASM 动态交互、数学推导与代码对照。",
                "Pure Rust WASM interactive visualizations, derivations, and PyTorch/Rust code across classic DL & LLM primitives."
              )}
            }
          }
          span { class: "inline-flex items-center gap-1 text-xs font-bold text-emerald-600 dark:text-emerald-400",
            {lang.pick("进入算子实验室 →", "Explore Operators Lab →")}
          }
        }

        Link {
          to: "/ai/visualizer",
          class: "group flex flex-col justify-between gap-4 rounded-2xl border border-slate-200 dark:border-slate-800 bg-linear-to-r from-slate-50/90 via-white to-sky-50/80 dark:from-slate-900 dark:via-slate-900 dark:to-sky-950/30 p-5 hover:border-indigo-400 dark:hover:border-indigo-600 transition-all shadow-xs",
          div {
            div { class: "inline-flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-sky-600 dark:text-sky-400",
              span { class: "px-2 py-0.5 rounded-full bg-sky-600 text-white text-[10px]", "Live Sandbox" }
              span { "HuggingFace · ModelScope · Safetensors" }
            }
            h2 { class: "mt-1.5 text-lg font-extrabold text-slate-900 dark:text-white group-hover:text-indigo-600 dark:group-hover:text-indigo-400 transition-colors",
              {lang.pick("任意仓库实时架构与显存解析器", "Live Repo Architecture & VRAM Inspector")}
            }
            p { class: "mt-1 text-sm text-slate-600 dark:text-slate-400",
              {lang.pick(
                "输入任意 HuggingFace / ModelScope 模型仓库或粘贴本地 config.json，零权重下载解析张量结构。",
                "Enter any HuggingFace / ModelScope repo ID or paste local config.json to inspect tensor shapes live."
              )}
            }
          }
          span { class: "inline-flex items-center gap-1 text-xs font-bold text-sky-600 dark:text-sky-400",
            {lang.pick("打开实时解析沙盒 →", "Open Live Visualizer →")}
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
