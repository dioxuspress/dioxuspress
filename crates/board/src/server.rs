//! 板块文章读取与领域聚合 Hub 服务端逻辑。
//!
//! - 文章扫描优先读取 `CONTENT_DIR/topics/<board>/<slug>/index.md(x)` 覆盖，
//!   再回退合并内置 `assets/topics/<board>/<slug>/index.md(x)`；
//! - 跨模块领域资源聚合（文档卷 / 课程 / 案例 / 博客）通过 [`register_hub_provider`]
//!   由组合根 `app` 在启动期注入，保持 `board` crate 零业务反向依赖。

use dioxus::prelude::*;
#[allow(unused_imports)]
use std::fs;
#[allow(unused_imports)]
use std::path::{Path, PathBuf};

#[allow(unused_imports)]
use crate::sort_by_date_desc;
use crate::{ArticleSummary, BoardHubResources};

#[cfg(feature = "server")]
pub type HubProvider = fn(&str, app_core::i18n::Language) -> BoardHubResources;

#[cfg(feature = "server")]
static HUB_PROVIDER: std::sync::OnceLock<HubProvider> = std::sync::OnceLock::new();

/// 注册领域 Hub 资源聚合器（由组合根 `crates/app/src/main.rs` 启动期调用一次）。
#[cfg(feature = "server")]
pub fn register_hub_provider(provider: HubProvider) {
  let _ = HUB_PROVIDER.set(provider);
}

#[cfg(feature = "server")]
fn parse_lang(raw: Option<&str>) -> app_core::i18n::Language {
  match raw.map(str::trim) {
    Some(s)
      if s.eq_ignore_ascii_case("en")
        || s.get(..3).is_some_and(|p| p.eq_ignore_ascii_case("en-")) =>
    {
      app_core::i18n::Language::En
    }
    Some(s)
      if s.eq_ignore_ascii_case("zh")
        || s.get(..3).is_some_and(|p| p.eq_ignore_ascii_case("zh-")) =>
    {
      app_core::i18n::Language::Zh
    }
    _ => app_core::settings::SiteConfig::active().default_language_enum(),
  }
}

/// 某领域板块的跨模块聚合资源（文档卷 / 课程 / 实战案例 / 相关博客）。
#[server]
pub async fn get_board_hub(
  board: String,
  lang: Option<String>,
) -> Result<BoardHubResources, ServerFnError> {
  if !app_core::utils::is_safe_slug(&board) {
    return Err(ServerFnError::new("无效的板块标识".to_string()));
  }
  let target_lang = parse_lang(lang.as_deref());
  Ok(match HUB_PROVIDER.get() {
    Some(provider) => provider(&board, target_lang),
    None => BoardHubResources::default(),
  })
}

#[allow(dead_code)]
#[derive(Debug, serde::Deserialize, Default)]
struct FrontMatter {
  #[serde(default)]
  title: String,
  #[serde(default)]
  description: String,
  #[serde(default)]
  date: String,
  #[serde(default)]
  subtopic: String,
  #[serde(default)]
  tags: Vec<String>,
}

#[allow(dead_code)]
fn parse_frontmatter(content: &str) -> FrontMatter {
  widgets::split_frontmatter(content)
    .and_then(|(yaml, _)| serde_yaml::from_str(yaml).ok())
    .unwrap_or_default()
}

/// 文章目录下的正文文件，`index.mdx` 优先。
#[allow(dead_code)]
fn index_file(dir: &Path) -> Option<PathBuf> {
  ["index.mdx", "index.md"].iter().map(|f| dir.join(f)).find(|p| p.exists())
}

/// Phase 8.5：列表 mtime cache。整个 `topics/` 一份指纹，按板块分组缓存。
#[cfg(feature = "server")]
static LIST_CACHE: app_core::utils::DirListingCache<
  std::collections::BTreeMap<String, Vec<ArticleSummary>>,
> = app_core::utils::DirListingCache::new();

/// E5.3：清空全部领域板块文章目录缓存（供 `SIGHUP` 或内容热刷新调用）。
#[cfg(feature = "server")]
pub fn invalidate_board_cache() {
  LIST_CACHE.invalidate();
}

#[cfg(feature = "server")]
fn build_article_list(dir: &Path) -> Vec<ArticleSummary> {
  let Ok(entries) = fs::read_dir(dir) else { return Vec::new() };
  let mut items = Vec::new();
  for entry in entries.flatten() {
    let path = entry.path();
    if !path.is_dir() {
      continue;
    }
    let slug = path.file_name().and_then(|n| n.to_str()).unwrap_or_default().to_string();
    let Some(file) = index_file(&path) else { continue };
    let meta = parse_frontmatter(&fs::read_to_string(&file).unwrap_or_default());
    items.push(ArticleSummary {
      slug,
      title: meta.title,
      description: meta.description,
      date: meta.date,
      subtopic: meta.subtopic,
      tags: meta.tags,
    });
  }
  sort_by_date_desc(&mut items);
  items
}

#[cfg(feature = "server")]
fn build_all(root: &Path) -> std::collections::BTreeMap<String, Vec<ArticleSummary>> {
  let Ok(entries) = fs::read_dir(root) else { return Default::default() };
  entries
    .flatten()
    .filter(|e| e.path().is_dir())
    .filter_map(|e| {
      let id = e.file_name().to_str()?.to_string();
      Some((id, build_article_list(&e.path())))
    })
    .collect()
}

#[cfg(feature = "server")]
fn build_all_with_overlay(
  base_root: &Path,
  overlay_root: Option<&Path>,
) -> std::collections::BTreeMap<String, Vec<ArticleSummary>> {
  let mut merged = build_all(base_root);
  if let Some(overlay) = overlay_root.filter(|p| *p != base_root && p.is_dir()) {
    for (board_id, overlay_items) in build_all(overlay) {
      let entry = merged.entry(board_id).or_default();
      for item in overlay_items {
        if let Some(pos) = entry.iter().position(|existing| existing.slug == item.slug) {
          entry[pos] = item;
        } else {
          entry.push(item);
        }
      }
      sort_by_date_desc(entry);
    }
  }
  merged
}

#[cfg(feature = "server")]
fn topic_roots() -> (PathBuf, Option<PathBuf>) {
  let base = app_core::utils::get_asset_root().join("topics");
  let overlay = app_core::utils::content_root()
    .map(|root| root.join("topics"))
    .filter(|p| p != &base && p.is_dir());
  (base, overlay)
}

/// 某板块的文章摘要，按日期降序。支持 `CONTENT_DIR/topics` 覆盖同名文章或追加新文章。
#[server]
pub async fn list_board_articles(board: String) -> Result<Vec<ArticleSummary>, ServerFnError> {
  if !app_core::utils::is_safe_slug(&board) {
    return Err(ServerFnError::new("无效的板块标识".to_string()));
  }
  let (base_root, overlay_root) = topic_roots();
  let mut fp = app_core::utils::fingerprint_for_dir(&base_root, |p| {
    matches!(p.file_name().and_then(|n| n.to_str()), Some("index.md" | "index.mdx"))
  });
  if let Some(overlay) = overlay_root.as_deref() {
    fp.extend(app_core::utils::fingerprint_for_dir(overlay, |p| {
      matches!(p.file_name().and_then(|n| n.to_str()), Some("index.md" | "index.mdx"))
    }));
    fp.sort();
  }
  let all =
    LIST_CACHE.get_or_rebuild(fp, || build_all_with_overlay(&base_root, overlay_root.as_deref()));
  Ok(all.get(&board).cloned().unwrap_or_default())
}

/// 读取单篇文章的原始 markdown（含 frontmatter）。不存在（含非法标识）返回 `None`。
#[cfg(feature = "server")]
fn read_article(root: &Path, board: &str, slug: &str) -> std::io::Result<Option<String>> {
  if !app_core::utils::is_safe_slug(board) || !app_core::utils::is_safe_slug(slug) {
    return Ok(None);
  }
  match index_file(&root.join(board).join(slug)) {
    Some(file) => fs::read_to_string(file).map(Some),
    None => Ok(None),
  }
}

#[cfg(feature = "server")]
fn read_article_with_overlay(
  base_root: &Path,
  overlay_root: Option<&Path>,
  board: &str,
  slug: &str,
) -> std::io::Result<Option<String>> {
  if let Some(overlay) = overlay_root.filter(|p| *p != base_root) {
    if let Some(content) = read_article(overlay, board, slug)? {
      return Ok(Some(content));
    }
  }
  read_article(base_root, board, slug)
}

/// 单篇文章原文。`Ok(None)` = 不存在（页面回 404），`Err` = 读取出错（回 500）。
#[server]
pub async fn get_board_article(
  board: String,
  slug: String,
) -> Result<Option<String>, ServerFnError> {
  let (base_root, overlay_root) = topic_roots();
  read_article_with_overlay(&base_root, overlay_root.as_deref(), &board, &slug).map_err(|e| {
    tracing::error!(board = %board, slug = %slug, error = %e, "board: read article failed");
    ServerFnError::new("读取失败".to_string())
  })
}

#[cfg(all(test, feature = "server"))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
  use super::*;

  #[test]
  fn frontmatter_parses_fields() {
    let md = "---\ntitle: Candle 推理\ndescription: 本地大模型\ndate: 2026-05-01\nsubtopic: llm\ntags: [candle, llm]\n---\n# body";
    let fm = parse_frontmatter(md);
    assert_eq!(fm.title, "Candle 推理");
    assert_eq!(fm.subtopic, "llm");
    assert_eq!(fm.tags, vec!["candle".to_string(), "llm".to_string()]);
  }

  #[test]
  fn frontmatter_missing_is_default() {
    let fm = parse_frontmatter("no frontmatter here");
    assert!(fm.title.is_empty());
    assert!(fm.tags.is_empty());
  }

  #[test]
  fn read_article_distinguishes_missing_from_present() {
    let tmp = std::env::temp_dir().join(format!("board-read-{}", std::process::id()));
    fs::create_dir_all(tmp.join("ai/real")).unwrap();
    fs::write(tmp.join("ai/real/index.md"), "# hi").unwrap();
    assert_eq!(read_article(&tmp, "ai", "real").unwrap().as_deref(), Some("# hi"));
    assert_eq!(read_article(&tmp, "ai", "missing").unwrap(), None);
    assert_eq!(read_article(&tmp, "ai", "../ai").unwrap(), None);
    assert_eq!(read_article(&tmp, "nope", "real").unwrap(), None);
    fs::remove_dir_all(&tmp).unwrap();
  }

  /// 内容校验：frontmatter 解析失败会静默退化为空标题（曾因描述里未转义的 `"` 发生）。
  #[test]
  fn every_real_article_has_a_title() {
    let all = build_all(&app_core::utils::get_asset_root().join("topics"));
    assert!(!all.is_empty(), "找不到 assets/topics");
    let untitled: Vec<String> = all
      .iter()
      .flat_map(|(b, items)| items.iter().map(move |a| (b, a)))
      .filter(|(_, a)| a.title.is_empty() || a.date.is_empty())
      .map(|(b, a)| format!("{b}/{}", a.slug))
      .collect();
    assert!(untitled.is_empty(), "frontmatter 解析失败或缺 title/date: {untitled:?}");
  }

  #[test]
  fn build_all_groups_articles_by_board() {
    let tmp = std::env::temp_dir().join(format!("board-test-{}", std::process::id()));
    let write = |rel: &str, body: &str| {
      let p = tmp.join(rel);
      fs::create_dir_all(p.parent().unwrap()).unwrap();
      fs::write(p, body).unwrap();
    };
    write("ai/old/index.md", "---\ntitle: Old\ndate: 2026-01-01\n---\n");
    write("ai/new/index.mdx", "---\ntitle: New\ndate: 2026-02-01\n---\n");
    write("ai/no-index/notes.md", "ignored");
    write("cli/one/index.md", "---\ntitle: One\n---\n");
    let all = build_all(&tmp);
    let ai: Vec<&str> = all["ai"].iter().map(|a| a.slug.as_str()).collect();
    assert_eq!(ai, vec!["new", "old"]);
    assert_eq!(all["cli"].len(), 1);
    fs::remove_dir_all(&tmp).unwrap();
  }

  #[test]
  fn overlay_overrides_same_slug_and_merges_new_articles() {
    let base = std::env::temp_dir().join(format!("board-base-{}", std::process::id()));
    let overlay = std::env::temp_dir().join(format!("board-overlay-{}", std::process::id()));
    let write = |root: &Path, rel: &str, body: &str| {
      let p = root.join(rel);
      fs::create_dir_all(p.parent().unwrap()).unwrap();
      fs::write(p, body).unwrap();
    };
    write(&base, "ai/shared/index.md", "---\ntitle: Base\ndate: 2026-01-01\n---\nbase body");
    write(&base, "ai/only-base/index.md", "---\ntitle: OnlyBase\ndate: 2026-01-02\n---\n");
    write(
      &overlay,
      "ai/shared/index.md",
      "---\ntitle: Override\ndate: 2026-03-01\n---\noverride body",
    );
    write(&overlay, "ai/only-overlay/index.md", "---\ntitle: OnlyOverlay\ndate: 2026-02-01\n---\n");

    let all = build_all_with_overlay(&base, Some(&overlay));
    let titles: Vec<&str> = all["ai"].iter().map(|a| a.title.as_str()).collect();
    assert_eq!(titles, vec!["Override", "OnlyOverlay", "OnlyBase"]);
    assert_eq!(
      read_article_with_overlay(&base, Some(&overlay), "ai", "shared").unwrap().as_deref(),
      Some("---\ntitle: Override\ndate: 2026-03-01\n---\noverride body")
    );
    assert!(read_article_with_overlay(&base, Some(&overlay), "ai", "only-base").unwrap().is_some());

    fs::remove_dir_all(&base).unwrap();
    fs::remove_dir_all(&overlay).unwrap();
  }
}
