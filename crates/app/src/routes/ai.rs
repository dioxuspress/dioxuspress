//! AI 板块与大模型架构图鉴页面（`ai` feature）。未编译时由 `absent_pages!` 生成同名 404 组件。

use dioxus::prelude::*;

use crate::components::module_gate::ModuleGate;
use module_ai::{
  AiArticlePage, AiComponentDetailPage, AiComponentsAtlasPage, AiIndexPage, AiMathAtlasPage,
  AiMathDetailPage, AiVisualizerPage, ModelConfigsGuidePage, ModelDetailPage, ModelsAtlasPage,
};

#[component]
pub fn Ai() -> Element {
  rsx! { ModuleGate { id: "ai".to_string(), AiIndexPage {} } }
}

#[component]
pub fn AiVisualizer() -> Element {
  rsx! { ModuleGate { id: "ai".to_string(), AiVisualizerPage {} } }
}

#[component]
pub fn AiComponents() -> Element {
  rsx! { ModuleGate { id: "ai".to_string(), AiComponentsAtlasPage {} } }
}

#[component]
pub fn AiComponentDetail(slug: String) -> Element {
  rsx! { ModuleGate { id: "ai".to_string(), AiComponentDetailPage { slug } } }
}

#[component]
pub fn AiMath() -> Element {
  rsx! { ModuleGate { id: "ai".to_string(), AiMathAtlasPage {} } }
}

#[component]
pub fn AiMathDetail(slug: String) -> Element {
  rsx! { ModuleGate { id: "ai".to_string(), AiMathDetailPage { slug } } }
}

#[component]
pub fn AiArticle(slug: String) -> Element {
  rsx! { ModuleGate { id: "ai".to_string(), AiArticlePage { slug } } }
}

#[component]
pub fn Models() -> Element {
  rsx! { ModuleGate { id: "ai".to_string(), ModelsAtlasPage {} } }
}

#[component]
pub fn ModelConfigsGuide() -> Element {
  rsx! { ModuleGate { id: "ai".to_string(), ModelConfigsGuidePage {} } }
}

#[component]
pub fn ModelDetail(slug: String) -> Element {
  rsx! { ModuleGate { id: "ai".to_string(), ModelDetailPage { slug } } }
}
