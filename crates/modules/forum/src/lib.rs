use std::sync::{OnceLock, RwLock};

pub mod forum;
pub mod server;

/// 宿主站点扩展的论坛话题引用类型（如 `"paper"` → `/papers/<path>`），
/// 避免公开框架硬编码下游垂直业务模块。
#[derive(Clone)]
pub struct ForumRefKindSpec {
  pub kind: String,
  pub badge_label: String,
  /// 跳转 URL 前缀（如 `"/papers/"`）。
  pub url_prefix: String,
  /// 可选标题格式化回调（如 `|path| format!("arXiv:{path}")`）。
  pub title_formatter: Option<fn(&str) -> String>,
}

fn extra_ref_kinds_slot() -> &'static RwLock<Vec<ForumRefKindSpec>> {
  static EXTRA: OnceLock<RwLock<Vec<ForumRefKindSpec>>> = OnceLock::new();
  EXTRA.get_or_init(|| RwLock::new(Vec::new()))
}

/// 供宿主站点在启动期（双端）注册额外的论坛关联引用类型（按 `kind` 去重，支持叠加）。
pub fn register_extra_ref_kinds(specs: Vec<ForumRefKindSpec>) {
  if let Ok(mut guard) = extra_ref_kinds_slot().write() {
    for spec in specs {
      if !guard.iter().any(|existing| existing.kind == spec.kind) {
        guard.push(spec);
      }
    }
  }
}

/// 按 `kind` 查找宿主注册的扩展引用类型。
pub fn find_extra_ref_kind(kind: &str) -> Option<ForumRefKindSpec> {
  extra_ref_kinds_slot().read().ok().and_then(|g| g.iter().find(|s| s.kind == kind).cloned())
}
