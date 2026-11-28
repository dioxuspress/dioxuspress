//! Phase 3.3：Routable layout 入口。
//!
//! `Navbar` 是 `routes/mod.rs::Route` `#[layout(Navbar)]` 的入口组件；
//! Dioxus Router 编译期固定该入口，但内部我们通过 server fn `get_active_layout`
//! 动态选择 [`crate::components::layouts::ClassicShell`] 或
//! [`crate::components::layouts::MinimalShell`]。
//!
//! 历史 Navbar 的实现已下沉到 `layouts/classic.rs`；本文件仅负责分发。

use dioxus::prelude::*;
use dioxus::router::Outlet;

use crate::components::layouts::{
  ClassicShell, DocsShell, FooterBoardLink, MinimalShell, NavEcosystem,
};
use crate::routes::Route;

/// 与具体 `Route` 枚举解耦的通用站点布局壳，供宿主站点（含私有扩展仓）复用。
/// 直接同步读取已水合的 `SiteConfig::active().active_layout_or_default()`，避免首屏额外发起 POST RPC。
#[component]
pub fn SiteNavbar(
  current_path: String,
  #[props(default)] ecosystems: Vec<NavEcosystem>,
  #[props(default)] extra_nav_links: Vec<FooterBoardLink>,
  #[props(default)] footer_boards: Vec<FooterBoardLink>,
  children: Element,
) -> Element {
  let cfg = app_core::settings::SiteConfig::active();
  let layout = cfg.active_layout_or_default();

  rsx! {
    match layout {
      "docs" => rsx! { DocsShell { current_path, {children} } },
      "minimal" => rsx! { MinimalShell { {children} } },
      _ => rsx! { ClassicShell { current_path, ecosystems, extra_nav_links, footer_boards, {children} } },
    }
  }
}

/// DioxusPress 默认 `Route` 的 `#[layout(Navbar)]` 入口组件。
#[component]
pub fn Navbar() -> Element {
  let current_path = use_route::<Route>().to_string();
  rsx! {
    SiteNavbar {
      current_path,
      Outlet::<Route> {}
    }
  }
}
