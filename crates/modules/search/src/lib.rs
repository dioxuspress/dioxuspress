use std::sync::{OnceLock, RwLock};

#[cfg(feature = "server")]
pub mod engine;
#[cfg(feature = "server")]
pub mod indexer;
pub mod search;
pub mod server;
pub mod text;

/// 宿主站点注册的自定义搜索类型（如垂直内容板块或自定义频道），
/// 使公开框架不再硬编码任何私有站点的垂直板块名。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchKindSpec {
  /// 索引文档的 `kind` 字段（如 `"ai"` / `"paper"` / `"embedded"`）。
  pub kind: String,
  /// 所属模块 id（供 `site.json::modules` 开关与编译分层过滤）。
  pub module_id: String,
  /// 搜索结果卡片徽章文字（如 `"AI"` / `"PAPER"`）。
  pub badge_label: String,
  /// 徽章 Tailwind 类名。
  pub badge_class: String,
  /// 搜索过滤栏显示的 i18n key（如 `"ai.label"`）；为 `None` 时回退到 `badge_label`。
  pub label_i18n_key: Option<String>,
  /// 是否展示在搜索弹窗顶部的 kind 过滤栏。默认 `true`。
  pub show_in_filter: bool,
  /// 为 `true` 时，服务端索引器自动扫描 `$CONTENT_DIR/topics/<kind>/`（回退 `assets/topics/<kind>/`）。
  pub scan_topics_dir: bool,
}

impl SearchKindSpec {
  pub fn new(
    kind: impl Into<String>,
    module_id: impl Into<String>,
    badge_label: impl Into<String>,
  ) -> Self {
    Self {
      kind: kind.into(),
      module_id: module_id.into(),
      badge_label: badge_label.into(),
      badge_class: "bg-indigo-100 dark:bg-indigo-900/40 text-indigo-700 dark:text-indigo-300"
        .to_string(),
      label_i18n_key: None,
      show_in_filter: true,
      scan_topics_dir: false,
    }
  }

  pub fn with_badge_class(mut self, class: impl Into<String>) -> Self {
    self.badge_class = class.into();
    self
  }

  pub fn with_label_i18n_key(mut self, key: impl Into<String>) -> Self {
    self.label_i18n_key = Some(key.into());
    self
  }

  pub fn with_filter_bar(mut self, show: bool) -> Self {
    self.show_in_filter = show;
    self
  }

  pub fn with_topics_dir(mut self) -> Self {
    self.scan_topics_dir = true;
    self
  }
}

fn extra_search_kinds_slot() -> &'static RwLock<Vec<SearchKindSpec>> {
  static EXTRA: OnceLock<RwLock<Vec<SearchKindSpec>>> = OnceLock::new();
  EXTRA.get_or_init(|| RwLock::new(Vec::new()))
}

/// 供宿主站点在启动期（双端）注册额外的搜索内容类型（按 `kind` 去重，支持多次叠加）。
pub fn register_extra_search_kinds(specs: Vec<SearchKindSpec>) {
  if let Ok(mut guard) = extra_search_kinds_slot().write() {
    for spec in specs {
      if !guard.iter().any(|existing| existing.kind == spec.kind) {
        guard.push(spec);
      }
    }
  }
}

/// 列出所有由宿主站点注册的额外搜索类型快照。
pub fn extra_search_kinds() -> Vec<SearchKindSpec> {
  extra_search_kinds_slot().read().map(|g| g.clone()).unwrap_or_default()
}

/// 查找已注册的扩展搜索类型。
pub fn find_extra_search_kind(kind: &str) -> Option<SearchKindSpec> {
  extra_search_kinds_slot().read().ok().and_then(|g| g.iter().find(|s| s.kind == kind).cloned())
}

/// 索引文档的 kind 属于哪个模块（模块开关 / 编译分层据此过滤）。`None` = 不属于可关的模块。
pub fn module_of_kind(kind: &str) -> Option<String> {
  match kind {
    "blog" => Some("blog".to_string()),
    "doc" => Some("docs".to_string()),
    "topic" => Some("forum".to_string()),
    "case" => Some("cases".to_string()),
    "course" => Some("course".to_string()),
    other => find_extra_search_kind(other).map(|spec| spec.module_id),
  }
}
