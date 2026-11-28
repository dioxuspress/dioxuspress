//! Phase 3.4：`ModuleGate` — 把页面包成 module-switch 感知组件。
//!
//! 同步读取 [`app_core::settings::SiteConfig::active`] 与已注册的编译期模块清单：
//! - 服务端（SSR）直接读取磁盘缓存的 `site.json`，停用模块在 SSR 首帧即拦截，绝不泄露正文；
//! - 客户端（WASM）从 `<meta name="dp-site-config">` 水合后的同一份配置同步判定，零网络 RPC、零闪烁。

use dioxus::prelude::*;
use dioxus::router::Link;

use crate::components::view::Container;
use crate::i18n::use_i18n;

/// 已启用模块 id：编译进来（FEATURE_TIERS DL2）且 `site.json` 未关。
/// 同步计算，SSR 与 hydration 首帧完全一致，不额外发起 POST RPC。
pub fn use_enabled_modules() -> Vec<String> {
  let cfg = app_core::settings::SiteConfig::active();
  app_core::engines::module::enabled_module_ids_in_config(&cfg)
}

#[component]
pub fn ModuleGate(id: String, children: Element) -> Element {
  let cfg = app_core::settings::SiteConfig::active();
  let is_enabled = app_core::engines::module::is_module_enabled_in_config(&cfg, &id);

  if !is_enabled {
    let lang = use_i18n()();
    let id_label = id.clone();
    let title = lang.pick("模块已停用", "Module Disabled");
    let desc = match lang {
      app_core::i18n::Language::En => {
        format!("The \"{id_label}\" module is disabled in site.json and is currently unavailable.")
      }
      app_core::i18n::Language::Zh => {
        format!("「{id_label}」模块在 site.json 中被设置为 enabled: false，目前不可访问。")
      }
    };
    let back_home = lang.pick("回到首页", "Back to Home");
    rsx! {
      section { class: "py-20 bg-white dark:bg-slate-950",
        Container {
          div { class: "max-w-xl mx-auto text-center",
            div { class: "text-5xl mb-4", "⚙️" }
            h1 { class: "text-2xl font-bold text-slate-900 dark:text-white mb-3",
              "{title}"
            }
            p { class: "text-slate-500 dark:text-slate-400 mb-8",
              "{desc}"
            }
            Link {
              to: "/",
              class: "inline-flex items-center rounded-md btn-flow px-4 py-2 text-sm font-semibold",
              "{back_home}"
            }
          }
        }
      }
    }
  } else {
    rsx! { {children} }
  }
}
