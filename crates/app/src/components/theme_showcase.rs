//! Phase E2.3：`ThemeShowcase` — 嵌入文档页的交互式双模主题色卡矩阵。
//!
//! 在 Markdown 中书写 `<ThemeShowcase />` 即可渲染：
//! - 展示当前站点开放的全部内置双模主题（亮色 + 暗色色板同屏对比）；
//! - 点击卡片「实时试穿」立即全站切换配色并与顶栏 `ThemePicker` 联动；
//! - 点击「复制配置」一键复制 `"theme": "<id>"` JSON 片段。

use app_core::i18n::Language;
use dioxus::prelude::*;

use crate::components::theme_picker::{apply_user_theme, use_theme_version};
use crate::i18n::use_i18n;
use crate::server::{list_available_themes, ThemeInfo};

/// 将 `<ThemeShowcase />` 注册到 `widgets` 全局 MDX 组件注册表（服务端与客户端双端幂等调用）。
pub fn register_components() {
  widgets::register(Box::new(ThemeShowcaseMdxComponent));
}

struct ThemeShowcaseMdxComponent;

impl widgets::MdxComponent for ThemeShowcaseMdxComponent {
  fn name(&self) -> &'static str {
    "ThemeShowcase"
  }

  fn render(&self, _attrs: &std::collections::HashMap<String, String>) -> Element {
    rsx! { ThemeShowcase {} }
  }
}

/// 交互式主题矩阵外壳（独立 Suspense 边界，SSR 直出零偏移）。
#[component]
pub fn ThemeShowcase() -> Element {
  rsx! {
    SuspenseBoundary { fallback: |_| VNode::empty(), ThemeShowcaseInner {} }
  }
}

#[component]
fn ThemeShowcaseInner() -> Element {
  let version = use_theme_version();
  let lang = use_i18n();
  let mut copied_id = use_signal::<Option<String>>(|| None);

  let first =
    use_server_future(|| async move { list_available_themes().await.unwrap_or_default() })?;
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
  if themes.is_empty() {
    return rsx! {};
  }

  let is_en = matches!(lang(), Language::En);
  let active_badge_text = if is_en { "Active" } else { "当前生效" };
  let apply_btn_text = if is_en { "Preview Live" } else { "实时试穿" };
  let applied_btn_text = if is_en { "Applied" } else { "使用中" };
  let copy_btn_text = if is_en { "Copy JSON" } else { "复制配置" };
  let copied_btn_text = if is_en { "✓ Copied" } else { "✓ 已复制" };
  let reset_btn_text = if is_en { "Reset to site default" } else { "恢复站点默认主题" };
  let ui_link_text = if is_en {
    "Explore all 84 components across themes on ui.dioxuspress.dev ↗"
  } else {
    "前往 ui.dioxuspress.dev 预览 84 个组件的主题表现 ↗"
  };

  rsx! {
    div { class: "not-prose my-8 space-y-4",
      div { class: "grid grid-cols-1 sm:grid-cols-2 gap-4",
        for item in themes.iter() {
          {
            let theme_id = item.id.clone();
            let theme_id_for_copy = item.id.clone();
            let desc = if is_en { item.description_en.as_str() } else { item.description.as_str() };
            let is_copied = copied_id().as_deref() == Some(item.id.as_str());
            let card_border = if item.is_active {
              "border-primary ring-2 ring-primary/25 shadow-sm"
            } else {
              "border-border hover:border-primary/50"
            };
            rsx! {
              div {
                key: "{item.id}",
                class: "flex flex-col justify-between rounded-xl border bg-card text-card-foreground overflow-hidden transition-all {card_border}",
                // 顶部亮色 / 暗色双极性实景迷你色带
                div { class: "grid grid-cols-2 h-14 border-b border-border/70 text-[10px] font-mono select-none",
                  div {
                    class: "flex items-center justify-between px-3",
                    style: "background-color: {item.light_bg};",
                    span {
                      class: "px-1.5 py-0.5 rounded font-semibold",
                      style: "background-color: {item.light_primary}; color: #ffffff;",
                      "Light"
                    }
                    span {
                      class: "inline-block w-3.5 h-3.5 rounded-full border border-black/15",
                      style: "background-color: {item.light_primary};",
                    }
                  }
                  div {
                    class: "flex items-center justify-between px-3 border-l border-border/50",
                    style: "background-color: {item.dark_bg};",
                    span {
                      class: "px-1.5 py-0.5 rounded font-semibold",
                      style: "background-color: {item.dark_primary}; color: {item.dark_bg};",
                      "Dark"
                    }
                    span {
                      class: "inline-block w-3.5 h-3.5 rounded-full border border-white/20",
                      style: "background-color: {item.dark_primary};",
                    }
                  }
                }

                // 主题标题、ID 与定位说明
                div { class: "p-4 flex-1 flex flex-col justify-between gap-3",
                  div {
                    div { class: "flex items-center justify-between gap-2",
                      div { class: "flex items-center gap-2 min-w-0",
                        span { class: "font-bold text-sm text-foreground truncate", "{item.label}" }
                        code { class: "px-1.5 py-0.5 rounded bg-muted text-muted-foreground font-mono text-[11px]",
                          "\"{item.id}\""
                        }
                      }
                      if item.is_active {
                        span { class: "shrink-0 inline-flex items-center rounded-full bg-primary/15 px-2 py-0.5 text-[11px] font-semibold text-primary",
                          "{active_badge_text}"
                        }
                      }
                    }
                    p { class: "mt-1.5 text-xs text-muted-foreground leading-relaxed",
                      "{desc}"
                    }
                  }

                  // 底部操作按钮：实时试穿 + 一键复制 JSON 配置
                  div { class: "pt-1 flex items-center gap-2",
                    button {
                      r#type: "button",
                      disabled: item.is_active,
                      class: if item.is_active {
                        "flex-1 inline-flex items-center justify-center rounded-lg bg-primary/15 text-primary px-3 py-1.5 text-xs font-semibold cursor-default"
                      } else {
                        "flex-1 inline-flex items-center justify-center rounded-lg bg-primary text-primary-foreground hover:opacity-90 px-3 py-1.5 text-xs font-semibold transition-opacity cursor-pointer"
                      },
                      onclick: move |_| {
                        apply_user_theme(theme_id.clone(), version);
                      },
                      if item.is_active { "{applied_btn_text}" } else { "{apply_btn_text}" }
                    }
                    button {
                      r#type: "button",
                      class: "inline-flex items-center justify-center rounded-lg border border-border bg-background hover:bg-accent hover:text-accent-foreground px-3 py-1.5 text-xs font-medium transition-colors cursor-pointer",
                      onclick: move |_| {
                        let id_str = theme_id_for_copy.clone();
                        let snippet = format!("\"theme\": \"{id_str}\"");
                        spawn(async move {
                          if widgets::browser::copy_text(&snippet).await {
                            copied_id.set(Some(id_str));
                          }
                        });
                      },
                      if is_copied { "{copied_btn_text}" } else { "{copy_btn_text}" }
                    }
                  }
                }
              }
            }
          }
        }
      }

      div { class: "flex flex-col sm:flex-row sm:items-center justify-between gap-2 pt-1 text-xs text-muted-foreground",
        button {
          r#type: "button",
          class: "self-start underline underline-offset-4 hover:text-foreground transition-colors cursor-pointer",
          onclick: move |_| apply_user_theme(String::new(), version),
          "{reset_btn_text}"
        }
        a {
          href: "https://ui.dioxuspress.dev",
          target: "_blank",
          rel: "noopener noreferrer",
          class: "font-medium text-primary hover:underline",
          "{ui_link_text}"
        }
      }
    }
  }
}
