//! Phase 3.3：Routable layout 入口。
//!
//! `Navbar` 是 `routes/mod.rs::Route` `#[layout(Navbar)]` 的入口组件；
//! Dioxus Router 编译期固定该入口，但内部我们通过 server fn `get_active_layout`
//! 动态选择 [`crate::components::layouts::ClassicShell`] 或
//! [`crate::components::layouts::MinimalShell`]。
//!
//! 历史 Navbar 的实现已下沉到 `layouts/classic.rs`；本文件仅负责分发。

use dioxus::prelude::*;

use crate::components::layouts::{ClassicShell, DocsShell, MinimalShell};
use crate::server::get_active_layout;

/// Routable layout 入口：内部根据 `get_active_layout` 选择 shell。
///
/// SSR 与水合首帧优先使用编译期内嵌的 `SiteConfig::embedded().active_layout_or_default()`，
/// 彻底消除 `docs` / `minimal` 布局在首屏水合时闪回 `classic` 的问题。
#[component]
pub fn Navbar() -> Element {
  let default_layout = app_core::settings::SiteConfig::embedded().active_layout_or_default();
  let layout_res = use_resource(move || async move {
    get_active_layout().await.unwrap_or_else(|_| default_layout.to_string())
  });
  let layout: String =
    layout_res.read().as_ref().cloned().unwrap_or_else(|| default_layout.to_string());

  rsx! {
    match layout.as_str() {
      "docs" => rsx! { DocsShell {} },
      "minimal" => rsx! { MinimalShell {} },
      _ => rsx! { ClassicShell {} },
    }
  }
}
