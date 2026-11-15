//! 双主题可缩放架构图查看器（[`ZoomableDiagram`]）。
//!
//! 针对文档卷首与正文中的 `.light.<ext>` / `.dark.<ext>` 双主题配图：
//! - 首屏内联卡片保留双主题自动切换（`dark:hidden` / `hidden dark:block`），SSR 与客户端首帧 DOM 完全一致。
//! - 位于文档前部（`eager = true`）的架构图启用 `loading="eager"` 与 `fetchpriority="high"`，并为标准
//!   `arch.light.*` 预置 `1600×1000` 固有宽高比以消除 CLS 并提升 LCP。
//! - 点击（或回车/空格）弹出全屏检视层：支持滚轮无级缩放（`50%–400%`）、指针拖拽平移、双击切换放大、
//!   浅色/深色主题预览切换、原图新标签打开以及 `Esc` / `+` / `-` / `0` 快捷键。

use dioxus::prelude::*;

use crate::browser::{on_document_keydown, set_class, KeyListener};

const MIN_SCALE: f64 = 0.5;
const MAX_SCALE: f64 = 4.0;
const ZOOM_STEP: f64 = 1.25;
const DOUBLE_CLICK_SCALE: f64 = 1.8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ThemeOverride {
  FollowSite,
  Light,
  Dark,
}

impl ThemeOverride {
  fn next(self) -> Self {
    match self {
      Self::FollowSite => Self::Light,
      Self::Light => Self::Dark,
      Self::Dark => Self::FollowSite,
    }
  }

  fn label(self) -> &'static str {
    match self {
      Self::FollowSite => "跟随主题",
      Self::Light => "浅色图",
      Self::Dark => "深色图",
    }
  }
}

pub(crate) fn clamp_scale(scale: f64) -> f64 {
  scale.clamp(MIN_SCALE, MAX_SCALE)
}

pub(crate) fn step_zoom(current: f64, zoom_in: bool) -> f64 {
  clamp_scale(if zoom_in { current * ZOOM_STEP } else { current / ZOOM_STEP })
}

/// 标准卷首架构图（`arch.light.svg` / `arch.light.webp`）统一采用 `1600×1000` 画布；
/// 其余双主题图（如基准测试条形图 `*-bench.light.svg`）保留自身固有尺寸，不强行指定比例。
pub(crate) fn arch_intrinsic_size(src: &str) -> Option<(u32, u32)> {
  let clean = src.split(['?', '#']).next().unwrap_or(src);
  let file = clean.rsplit('/').next().unwrap_or(clean);
  matches!(
    file.to_ascii_lowercase().as_str(),
    "arch.light.svg" | "arch.dark.svg" | "arch.light.webp" | "arch.dark.webp"
  )
  .then_some((1600, 1000))
}

#[component]
pub fn ZoomableDiagram(
  light_src: String,
  dark_src: String,
  alt: String,
  title: String,
  #[props(default)] eager: bool,
) -> Element {
  let mut open = use_signal(|| false);
  let mut scale = use_signal(|| 1.0_f64);
  let mut tx = use_signal(|| 0.0_f64);
  let mut ty = use_signal(|| 0.0_f64);
  // (pointer_start_x, pointer_start_y, origin_tx, origin_ty)
  let mut drag_origin = use_signal(|| None::<(f64, f64, f64, f64)>);
  let mut drag_moved = use_signal(|| false);
  let mut theme_mode = use_signal(|| ThemeOverride::FollowSite);
  let mut key_listener = use_signal(|| None::<KeyListener>);

  use_drop(move || {
    set_class(false, "overflow-hidden", false);
  });

  use_effect(move || {
    let is_open = open();
    set_class(false, "overflow-hidden", is_open);
    if is_open {
      key_listener.set(Some(on_document_keydown(move |key, ctrl_or_meta| match key {
        "Escape" => {
          open.set(false);
          drag_origin.set(None);
          true
        }
        "+" | "=" if !ctrl_or_meta => {
          let next = step_zoom(*scale.peek(), true);
          scale.set(next);
          true
        }
        "-" | "_" if !ctrl_or_meta => {
          let next = step_zoom(*scale.peek(), false);
          scale.set(next);
          true
        }
        "0" if !ctrl_or_meta => {
          scale.set(1.0);
          tx.set(0.0);
          ty.set(0.0);
          true
        }
        _ => false,
      })));
    } else {
      key_listener.set(None);
    }
  });

  let open_viewer = move || {
    let mut scale = scale;
    let mut tx = tx;
    let mut ty = ty;
    let mut drag_origin = drag_origin;
    let mut drag_moved = drag_moved;
    let mut theme_mode = theme_mode;
    let mut open = open;
    scale.set(1.0);
    tx.set(0.0);
    ty.set(0.0);
    drag_origin.set(None);
    drag_moved.set(false);
    theme_mode.set(ThemeOverride::FollowSite);
    open.set(true);
  };

  let size = arch_intrinsic_size(&light_src);
  let loading_attr = if eager { "eager" } else { "lazy" };
  let aria_open = if alt.is_empty() {
    "放大查看架构图".to_string()
  } else {
    format!("放大查看架构图：{alt}")
  };
  let caption = if !title.is_empty() { title.clone() } else { alt.clone() };
  let pct = (scale() * 100.0).round() as i32;

  rsx! {
    figure { class: "not-prose my-8",
      div {
        role: "button",
        tabindex: "0",
        "aria-label": "{aria_open}",
        class: "group relative rounded-2xl overflow-hidden shadow-xl border border-slate-200 dark:border-slate-800 bg-slate-50/40 dark:bg-slate-900/40 cursor-zoom-in focus:outline-none focus-visible:ring-2 focus-visible:ring-primary",
        onclick: move |_| open_viewer(),
        onkeydown: move |e: KeyboardEvent| {
          let k = e.key();
          if k == Key::Enter || k == Key::Character(" ".to_string()) {
            e.prevent_default();
            open_viewer();
          }
        },
        img {
          src: "{light_src}",
          alt: "{alt}",
          width: if let Some((w, _)) = size { "{w}" },
          height: if let Some((_, h)) = size { "{h}" },
          loading: "{loading_attr}",
          fetchpriority: if eager { "high" },
          decoding: "async",
          class: "dark:hidden w-full h-auto block mx-auto select-none",
        }
        img {
          src: "{dark_src}",
          alt: "{alt}",
          width: if let Some((w, _)) = size { "{w}" },
          height: if let Some((_, h)) = size { "{h}" },
          loading: "{loading_attr}",
          fetchpriority: if eager { "high" },
          decoding: "async",
          class: "hidden dark:block w-full h-auto block mx-auto select-none",
        }
        span {
          "aria-hidden": "true",
          class: "pointer-events-none absolute top-3 right-3 inline-flex items-center gap-1.5 rounded-full bg-slate-900/75 dark:bg-slate-800/85 px-3 py-1 text-xs font-medium text-white shadow-md backdrop-blur-sm opacity-0 group-hover:opacity-100 group-focus-visible:opacity-100 transition-opacity",
          svg {
            class: "w-3.5 h-3.5",
            fill: "none",
            stroke: "currentColor",
            view_box: "0 0 24 24",
            path {
              stroke_linecap: "round",
              stroke_linejoin: "round",
              stroke_width: "2",
              d: "M21 21l-4.35-4.35M11 18a7 7 0 100-14 7 7 0 000 14zm0-10v6m-3-3h6",
            }
          }
          "放大检视"
        }
      }
      if !title.is_empty() {
        figcaption { class: "text-center text-sm text-slate-500 dark:text-slate-400 mt-3 italic",
          "{title}"
        }
      }

      if open() {
        div {
          role: "dialog",
          "aria-modal": "true",
          "aria-label": "{aria_open}",
          class: "fixed inset-0 z-50 flex flex-col bg-slate-950/88 backdrop-blur-md select-none",
          onwheel: move |e: WheelEvent| {
            e.prevent_default();
            let dy = e.delta().strip_units().y;
            if dy.abs() > f64::EPSILON {
              let factor = if dy < 0.0 { 1.12 } else { 1.0 / 1.12 };
              scale.set(clamp_scale(scale() * factor));
            }
          },
          // 顶部工具栏
          div {
            class: "flex flex-wrap items-center justify-between gap-3 px-4 sm:px-6 py-3 border-b border-white/10 bg-slate-950/70 text-slate-200",
            onclick: move |e| e.stop_propagation(),
            p { class: "min-w-0 flex-1 truncate text-xs sm:text-sm font-medium text-slate-200",
              "{caption}"
            }
            div { class: "flex items-center gap-1.5 sm:gap-2 text-xs",
              button {
                r#type: "button",
                class: "rounded-lg border border-white/15 bg-white/5 hover:bg-white/15 px-2.5 py-1.5 font-medium text-slate-200 transition-colors",
                onclick: move |_| theme_mode.set(theme_mode().next()),
                "{theme_mode().label()}"
              }
              div { class: "inline-flex items-center rounded-lg border border-white/15 bg-white/5 overflow-hidden",
                button {
                  r#type: "button",
                  "aria-label": "缩小",
                  class: "px-2.5 py-1.5 hover:bg-white/15 text-slate-200 font-semibold transition-colors",
                  onclick: move |_| scale.set(step_zoom(scale(), false)),
                  "−"
                }
                button {
                  r#type: "button",
                  "aria-label": "重置缩放比例",
                  class: "min-w-[3.5rem] px-2 py-1.5 hover:bg-white/15 text-center font-mono text-slate-100 transition-colors",
                  onclick: move |_| {
                    scale.set(1.0);
                    tx.set(0.0);
                    ty.set(0.0);
                  },
                  "{pct}%"
                }
                button {
                  r#type: "button",
                  "aria-label": "放大",
                  class: "px-2.5 py-1.5 hover:bg-white/15 text-slate-200 font-semibold transition-colors",
                  onclick: move |_| scale.set(step_zoom(scale(), true)),
                  "+"
                }
              }
              button {
                r#type: "button",
                class: "hidden sm:inline-flex rounded-lg border border-white/15 bg-white/5 hover:bg-white/15 px-2.5 py-1.5 font-medium text-slate-200 transition-colors",
                onclick: move |_| {
                  scale.set(1.0);
                  tx.set(0.0);
                  ty.set(0.0);
                },
                "重置"
              }
              match theme_mode() {
                ThemeOverride::FollowSite => rsx! {
                  a {
                    href: "{light_src}",
                    target: "_blank",
                    rel: "noopener",
                    class: "dark:hidden rounded-lg border border-white/15 bg-white/5 hover:bg-white/15 px-2.5 py-1.5 font-medium text-slate-200 no-underline transition-colors",
                    "原图 ↗"
                  }
                  a {
                    href: "{dark_src}",
                    target: "_blank",
                    rel: "noopener",
                    class: "hidden dark:inline-flex rounded-lg border border-white/15 bg-white/5 hover:bg-white/15 px-2.5 py-1.5 font-medium text-slate-200 no-underline transition-colors",
                    "原图 ↗"
                  }
                },
                ThemeOverride::Light => rsx! {
                  a {
                    href: "{light_src}",
                    target: "_blank",
                    rel: "noopener",
                    class: "rounded-lg border border-white/15 bg-white/5 hover:bg-white/15 px-2.5 py-1.5 font-medium text-slate-200 no-underline transition-colors",
                    "原图 ↗"
                  }
                },
                ThemeOverride::Dark => rsx! {
                  a {
                    href: "{dark_src}",
                    target: "_blank",
                    rel: "noopener",
                    class: "rounded-lg border border-white/15 bg-white/5 hover:bg-white/15 px-2.5 py-1.5 font-medium text-slate-200 no-underline transition-colors",
                    "原图 ↗"
                  }
                },
              }
              button {
                r#type: "button",
                "aria-label": "关闭预览",
                class: "rounded-lg border border-white/15 bg-white/10 hover:bg-white/20 px-2.5 py-1.5 font-semibold text-white transition-colors",
                onclick: move |_| {
                  open.set(false);
                  drag_origin.set(None);
                },
                "✕"
              }
            }
          }
          // 画布舞台：拖拽平移、双击切换放大、点击空白处关闭
          div {
            class: if drag_origin().is_some() {
              "relative flex-1 overflow-hidden flex items-center justify-center p-4 sm:p-8 cursor-grabbing"
            } else {
              "relative flex-1 overflow-hidden flex items-center justify-center p-4 sm:p-8 cursor-grab"
            },
            style: "touch-action: none;",
            onpointerdown: move |e: PointerEvent| {
              let coords = e.client_coordinates();
              drag_origin.set(Some((coords.x, coords.y, tx(), ty())));
              drag_moved.set(false);
            },
            onpointermove: move |e: PointerEvent| {
              if let Some((sx, sy, ox, oy)) = drag_origin() {
                let coords = e.client_coordinates();
                let dx = coords.x - sx;
                let dy = coords.y - sy;
                if dx.hypot(dy) > 4.0 {
                  drag_moved.set(true);
                }
                tx.set(ox + dx);
                ty.set(oy + dy);
              }
            },
            onpointerup: move |_| {
              drag_origin.set(None);
            },
            onpointercancel: move |_| {
              drag_origin.set(None);
            },
            onclick: move |_| {
              if !drag_moved() {
                open.set(false);
              }
            },
            div {
              style: "transform: translate3d({tx():.1}px, {ty():.1}px, 0) scale({scale():.3}); transform-origin: center center;",
              class: if drag_origin().is_some() {
                ""
              } else {
                "transition-transform duration-75 ease-out"
              },
              onclick: move |e| e.stop_propagation(),
              ondoubleclick: move |e| {
                e.stop_propagation();
                if (scale() - 1.0).abs() < 0.05 {
                  scale.set(DOUBLE_CLICK_SCALE);
                } else {
                  scale.set(1.0);
                  tx.set(0.0);
                  ty.set(0.0);
                }
              },
              match theme_mode() {
                ThemeOverride::FollowSite => rsx! {
                  img {
                    src: "{light_src}",
                    alt: "{alt}",
                    draggable: "false",
                    class: "dark:hidden max-w-[92vw] max-h-[82vh] w-auto h-auto rounded-xl shadow-2xl pointer-events-none select-none",
                  }
                  img {
                    src: "{dark_src}",
                    alt: "{alt}",
                    draggable: "false",
                    class: "hidden dark:block max-w-[92vw] max-h-[82vh] w-auto h-auto rounded-xl shadow-2xl pointer-events-none select-none",
                  }
                },
                ThemeOverride::Light => rsx! {
                  img {
                    src: "{light_src}",
                    alt: "{alt}",
                    draggable: "false",
                    class: "max-w-[92vw] max-h-[82vh] w-auto h-auto rounded-xl shadow-2xl pointer-events-none select-none",
                  }
                },
                ThemeOverride::Dark => rsx! {
                  img {
                    src: "{dark_src}",
                    alt: "{alt}",
                    draggable: "false",
                    class: "max-w-[92vw] max-h-[82vh] w-auto h-auto rounded-xl shadow-2xl pointer-events-none select-none",
                  }
                },
              }
            }
          }
          // 底部操作提示
          div {
            class: "pointer-events-none pb-3 text-center text-[11px] text-slate-400",
            "滚轮缩放 · 拖拽平移 · 双击切换 180% · 按 Esc 关闭"
          }
        }
      }
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn arch_intrinsic_size_matches_only_standard_arch_diagrams() {
    assert_eq!(arch_intrinsic_size("/docs-files/tokio/arch.light.svg"), Some((1600, 1000)));
    assert_eq!(arch_intrinsic_size("/docs-files/burn/arch.light.webp?v=1"), Some((1600, 1000)));
    assert_eq!(
      arch_intrinsic_size("/docs-files/patterns/05-behavioural/strategy/strategy-bench.light.svg"),
      None
    );
  }

  #[test]
  fn step_zoom_clamps_within_bounds() {
    assert_eq!(clamp_scale(0.1), MIN_SCALE);
    assert_eq!(clamp_scale(9.0), MAX_SCALE);
    let z_in = step_zoom(1.0, true);
    assert!((z_in - 1.25).abs() < 1e-6);
    let z_out = step_zoom(z_in, false);
    assert!((z_out - 1.0).abs() < 1e-6);
  }
}
