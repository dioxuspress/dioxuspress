//! Phase 3.1：ThemePicker — Navbar 中的主题切换下拉。
//!
//! 通过 `list_available_themes` server fn 拉取内置主题，
//! 点击后调用 `set_user_theme` 写入 cookie 并 bump `ThemeVersion` Signal，
//! 触发上层 `theme_css` `use_resource` 重新请求合并 CSS。

use dioxus::prelude::*;
use dioxus_shadcn::{
  Dropdown, DropdownContent, DropdownItem, DropdownLabel, DropdownRadioGroup, DropdownRadioItem,
  DropdownSeparator, DropdownTrigger,
};

use crate::i18n::{t, use_i18n};
use crate::server::{list_available_themes, set_user_theme, ThemeInfo};

/// 主题 cookie 名（与后端 `THEME_COOKIE_NAME` 保持一致）。
const THEME_COOKIE_NAME: &str = "site_theme";

/// 全局主题版本号：每次切换主题 +1，App 组件中的 `use_resource` 依赖该值，
/// 使聚合 CSS 在用户切换主题后立即重取。
#[derive(Clone, Copy)]
pub struct ThemeVersion(pub Signal<u32>);

/// 在 App 根上挂载 `ThemeVersion` context，返回 Signal 以便上层订阅。
pub fn use_theme_version_provider() -> Signal<u32> {
  let sig = use_signal(|| 0u32);
  use_context_provider(|| ThemeVersion(sig));
  sig
}

/// 子组件读取当前 ThemeVersion Signal。
pub fn use_theme_version() -> Signal<u32> {
  use_context::<ThemeVersion>().0
}

/// 切换访客主题（服务端 Set-Cookie + 浏览器 `document.cookie` 双写 + 触发全局 `ThemeVersion` 刷新）。
/// 传入空字符串表示清除偏好、回退到 `site.json` 默认主题。
pub fn apply_user_theme(id: String, mut version: Signal<u32>) {
  spawn(async move {
    if let Err(e) = set_user_theme(id.clone()).await {
      tracing::error!(error = %e, "theme picker: set_user_theme failed");
    }

    let trimmed = id.trim();
    let max_age = if trimmed.is_empty() { 0 } else { 31_536_000 };
    widgets::browser::set_cookie(THEME_COOKIE_NAME, trimmed, max_age);

    version.with_mut(|v| *v += 1);
  });
}

/// 主题下拉。展示当前激活主题，点击展开列表，点击列表项写 cookie + bump version。
#[component]
pub fn ThemePicker() -> Element {
  // 列表随 SSR 下发：以前 hydration 后才拉，按钮晚出现、把右侧导航项往左挤一下。
  // 单独一层 boundary，挂起只影响这个按钮。
  rsx! {
      SuspenseBoundary { fallback: |_| VNode::empty(), ThemePickerInner {} }
  }
}

#[component]
fn ThemePickerInner() -> Element {
  let version = use_theme_version();
  let lang = use_i18n();

  let first =
    use_server_future(|| async move { list_available_themes().await.unwrap_or_default() })?;
  // 切换后重取（激活态变了）；旧列表保留到新数据到达
  let mut loaded = use_signal::<Option<Vec<ThemeInfo>>>(|| None);
  use_effect(move || {
    if version() == 0 {
      return;
    }
    spawn(async move {
      if let Ok(list) = list_available_themes().await {
        loaded.set(Some(list));
      }
    });
  });
  let themes: Vec<ThemeInfo> = loaded().unwrap_or_else(|| first().unwrap_or_default());

  let active = themes.iter().find(|t| t.is_active);
  let active_id = active.map(|t| t.id.clone()).unwrap_or_default();
  let active_label = active.map(|t| t.label.clone()).unwrap_or_else(|| t(lang(), "theme.heading"));

  // 不显示 picker 当只有 ≤ 1 个主题（无切换意义）
  if themes.len() <= 1 {
    return rsx! {};
  }

  let switch = use_callback(move |id: String| {
    apply_user_theme(id, version);
  });

  rsx! {
    Dropdown {
      DropdownTrigger {
        class: "flex items-center gap-1.5 px-2 py-1 rounded-md hover:bg-accent text-muted-foreground transition-colors text-xs font-semibold",
        title: "{t(lang(), \"theme.toggle\")}",
        span {
          class: "inline-block w-2.5 h-2.5 rounded-full bg-primary ring-1 ring-border shrink-0",
          "aria-hidden": "true",
        }
        span { class: "hidden sm:inline whitespace-nowrap", "{active_label}" }
      }
      DropdownContent { class: "w-52",
        DropdownLabel { class: "text-[10px] uppercase tracking-wider", "{t(lang(), \"theme.heading\")}" }
        div { class: "max-h-72 overflow-y-auto",
          DropdownRadioGroup { value: active_id, on_value_change: move |id: String| switch.call(id),
            for item in themes.iter() {
              DropdownRadioItem {
                key: "{item.id}",
                value: item.id.clone(),
                class: "flex items-center justify-between gap-2 w-full",
                span { class: "truncate", "{item.label}" }
                span {
                  class: "flex items-center gap-1 shrink-0",
                  "aria-hidden": "true",
                  span {
                    class: "inline-block w-3 h-3 rounded-full border border-border/80",
                    style: "background-color: {item.light_primary};",
                    title: "Light {item.light_primary}",
                  }
                  span {
                    class: "inline-block w-3 h-3 rounded-full border border-border/80",
                    style: "background-color: {item.dark_primary};",
                    title: "Dark {item.dark_primary}",
                  }
                }
              }
            }
          }
        }
        DropdownSeparator {}
        DropdownItem { class: "text-xs text-muted-foreground", onclick: move |_| switch.call(String::new()),
          "{t(lang(), \"theme.reset\")}"
        }
      }
    }
  }
}
