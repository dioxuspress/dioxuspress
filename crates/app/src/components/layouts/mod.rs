//! Phase 3.3：Layout 包。
//!
//! `Navbar`（在 `crates/app/src/components/nav.rs`）作为 Dioxus Routable
//! `#[layout(...)]` 入口，内部根据 server fn `get_active_layout` 的结果
//! 选择渲染 [`ClassicShell`] 或 [`MinimalShell`]。
//!
//! - **ClassicShell**：原始 Navbar+Footer，最完整体验。
//! - **MinimalShell**：紧凑顶部条 + 无 Footer + 仅 Logo/搜索/主题/暗色/语言；
//!   适合写作 / 阅读优先场景。
//!
//! 两个 shell 都嵌入 `Outlet::<Route>` 以渲染当前路由的内容。

use dioxus::prelude::*;

pub mod classic;
pub mod docs;
pub mod minimal;

pub use classic::ClassicShell;
pub use docs::DocsShell;
pub use minimal::MinimalShell;

/// 生态下拉菜单中的单个领域链接（与具体 `Route` 枚举解耦，供外部站点壳注入）。
#[derive(Clone, PartialEq)]
pub struct NavDomain {
  pub id: &'static str,
  pub label_key: &'static str,
  pub path: &'static str,
  pub module_id: &'static str,
}

/// 顶部导航生态分组（与具体 `Route` 枚举解耦）。
#[derive(Clone, PartialEq)]
pub struct NavEcosystem {
  pub id: &'static str,
  pub label_key: &'static str,
  pub blurb_key: &'static str,
  pub domains: Vec<NavDomain>,
}

/// 页脚专题列条目（与具体 `Route` 枚举解耦）。
#[derive(Clone, PartialEq)]
pub struct FooterBoardLink {
  pub module_id: &'static str,
  pub path: &'static str,
  pub label_key: &'static str,
}

/// `current` 是否等于 `section` 或位于其子路径下（按路径段，`/papers-x` 不匹配 `/papers`）。
pub fn path_within(current: &str, section: &str) -> bool {
  let current = current.split(['?', '#']).next().unwrap_or(current);
  current == section || current.strip_prefix(section).is_some_and(|rest| rest.starts_with('/'))
}

/// 站点品牌标识：根据 `site.json` 渲染横版明暗双态 Logo、独立紧凑图标或纯文字站名。
///
/// 采用纯 CSS `dark:hidden` / `hidden dark:block` 切换明暗图，保证 SSR 与 WASM 首屏
/// DOM 完全一致，配合 `DARK_MODE_BOOT_SCRIPT` 零闪烁。
#[component]
pub(super) fn BrandMark(#[props(default = false)] compact: bool) -> Element {
  let site_cfg = app_core::settings::SiteConfig::active();
  let site_title = widgets::site_name();

  if compact {
    if let Some(compact_src) = site_cfg.logo_mobile_url() {
      return rsx! {
        img { src: "{compact_src}", alt: "", class: "h-6 w-6 object-contain" }
        "{site_title}"
      };
    }
  }

  let wordmark_cls =
    if compact { "h-8 w-auto object-contain" } else { "h-9 sm:h-10 w-auto object-contain" };
  let icon_cls = if compact { "h-5 w-5 object-contain" } else { "h-7 w-7 object-contain" };

  match (site_cfg.logo_url(), site_cfg.logo_dark_url(), site_cfg.logo_wordmark) {
    (Some(light), Some(dark), true) => rsx! {
      img { src: "{light}", alt: "{site_title}", class: "{wordmark_cls} dark:hidden" }
      img { src: "{dark}", alt: "{site_title}", class: "{wordmark_cls} hidden dark:block" }
    },
    (Some(light), None, true) => rsx! {
      img { src: "{light}", alt: "{site_title}", class: "{wordmark_cls}" }
    },
    (Some(light), Some(dark), false) => rsx! {
      img { src: "{light}", alt: "", class: "{icon_cls} dark:hidden" }
      img { src: "{dark}", alt: "", class: "{icon_cls} hidden dark:block" }
      "{site_title}"
    },
    (Some(light), None, false) => rsx! {
      img { src: "{light}", alt: "", class: "{icon_cls}" }
      "{site_title}"
    },
    (None, _, _) => rsx! {
      "{site_title}"
    },
  }
}
