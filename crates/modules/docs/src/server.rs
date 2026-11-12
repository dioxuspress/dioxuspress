use dioxus::fullstack::{post, ServerFnError};
#[allow(unused_imports)]
use dioxus::prelude::*;
use serde::{Deserialize, Serialize};
#[allow(unused_imports)]
use std::fs;
#[allow(unused_imports)]
use std::path::PathBuf;

/// 文档 frontmatter 元数据（类似 Docusaurus）
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct DocMeta {
  #[serde(default)]
  pub title: String,
  #[serde(default)]
  pub description: String,
  #[serde(default)]
  pub keywords: Vec<String>,
  #[serde(default)]
  pub sidebar_label: Option<String>, // 侧栏显示名称（覆盖 title）
  #[serde(default)]
  pub sidebar_position: Option<i32>, // 侧栏排序（越小越前）
  #[serde(default)]
  pub image: Option<String>, // OG 图片
  /// 子项排序方向："asc"（默认，升序）或 "desc"（降序）
  /// 在父目录的 index.md 中设置，控制该目录下子项的排序方向
  /// 适合：周报/日报等以递增编号但需要最新期优先的场景
  #[serde(default)]
  pub sort_children: Option<String>,
  /// 这篇文档对着哪个库的哪个版本写的（S3），页面上显示为「基于 tokio 1.54.0」
  #[serde(default)]
  pub source: Option<DocSource>,
}

/// frontmatter `source: { crate: tokio, version: "1.54.0" }`
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DocSource {
  #[serde(rename = "crate")]
  pub krate: String,
  /// 不加引号也行：serde_yaml 读进 String 时保留原文，`1.10` 不会变成 `1.1`
  pub version: String,
}

/// 文档树节点（最多三级）
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DocTreeNode {
  pub slug: String,
  pub title: String,
  pub path: String,
  pub has_content: bool,
  /// 该节点是否同时存在中文（`index.{md,mdx}`）与英文伴生版（`index.en.{md,mdx}`）。
  #[serde(default)]
  pub has_bilingual: bool,
  pub description: String,
  pub children: Vec<DocTreeNode>,
}

fn default_resolved_lang() -> String {
  "zh-CN".to_string()
}

/// 文档内容响应（内容 + 元数据）
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DocContentResponse {
  pub content: String,
  pub meta: DocMeta,
  /// 同一个一级分类里按侧栏顺序的上一篇 / 下一篇（S2）
  pub prev: Option<DocLink>,
  pub next: Option<DocLink>,
  /// 从一级分类到当前页的面包屑链路（无正文分组节点的 `path` 为空字符串，仅作文本展示）。
  #[serde(default)]
  pub breadcrumbs: Vec<DocLink>,
  /// 相对于 `docs/` 根目录的源文件路径（如 `"guide/intro/index.md"`），用于拼接「编辑此页」外链。
  #[serde(default)]
  pub source_file: String,
  /// 实际命中的正文语言标签（`"en"` 或 `"zh-CN"`）：英文请求若回退到中文源文件则仍标 `"zh-CN"`。
  #[serde(default = "default_resolved_lang")]
  pub resolved_lang: String,
  /// 当前路径是否同时拥有中英双语源文件（用于注入 `hreflang` 与按语言区分的 `canonical`）。
  #[serde(default)]
  pub has_bilingual: bool,
}

/// 指向另一篇文档的链接
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DocLink {
  pub path: String,
  pub title: String,
}

/// 数据结构：_meta.json 中的条目
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
struct MetaEntry {
  slug: String,
  title: String,
}

/// 解析 frontmatter（YAML between --- delimiters）
#[cfg(feature = "server")]
pub(crate) fn parse_doc_frontmatter(content: &str, file: &std::path::Path) -> (DocMeta, String) {
  let Some((yaml, body)) = widgets::split_frontmatter(content) else {
    return (DocMeta::default(), content.to_string());
  };
  let meta: DocMeta = serde_yaml::from_str(yaml).unwrap_or_else(|e| {
    tracing::warn!(file = %file.display(), error = %e, "doc frontmatter: invalid YAML, metadata ignored");
    DocMeta::default()
  });
  (meta, body.to_string())
}

/// 正文里第一个 `# ` 标题。
#[cfg(feature = "server")]
fn first_heading(body: &str) -> Option<String> {
  body.lines().find_map(|l| l.trim().strip_prefix("# ")).map(|t| t.trim().to_string())
}

/// 正文第一段纯文本（跳过标题、代码块、列表、引用、表格、HTML），去掉行内代码与加粗标记，
/// 截到 120 字。没有 frontmatter 的文档页用它当 meta description（Y8）。
#[cfg(feature = "server")]
fn first_paragraph(body: &str) -> Option<String> {
  let mut in_fence = false;
  for line in body.lines() {
    let t = line.trim();
    if t.starts_with("```") {
      in_fence = !in_fence;
      continue;
    }
    if in_fence || t.is_empty() || t.starts_with(['#', '-', '*', '>', '|', '<', '!']) {
      continue;
    }
    let text = t.replace("**", "").replace('`', "");
    return Some(text.chars().take(120).collect());
  }
  None
}

/// frontmatter 没写 title / description 时从正文补上，页面标题和摘要不再是空的（Y8）。
#[cfg(feature = "server")]
fn fill_missing_meta(meta: &mut DocMeta, body: &str) {
  if meta.title.is_empty() {
    meta.title = first_heading(body).unwrap_or_default();
  }
  if meta.description.is_empty() {
    meta.description = first_paragraph(body).unwrap_or_default();
  }
}

#[cfg(feature = "server")]
fn parse_requested_lang(lang: Option<&str>) -> app_core::i18n::Language {
  match lang.map(str::trim) {
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

#[cfg(feature = "server")]
fn find_zh_doc_in_dir(dir: &std::path::Path) -> Option<PathBuf> {
  ["index.md", "index.mdx"].into_iter().map(|n| dir.join(n)).find(|p| p.is_file())
}

#[cfg(feature = "server")]
fn find_en_doc_in_dir(dir: &std::path::Path) -> Option<PathBuf> {
  ["index.en.md", "index.en.mdx"].into_iter().map(|n| dir.join(n)).find(|p| p.is_file())
}

/// 从指定语言的伴生文档（回退默认文档）提取元数据（标题、描述、排序、存在性与双语状态）
#[cfg(feature = "server")]
fn extract_doc_info_for_lang(
  dir: &std::path::Path,
  lang: app_core::i18n::Language,
) -> (Option<String>, String, Option<i32>, bool, bool) {
  let (title, desc, pos, _, has_content, has_bilingual) = extract_doc_info_full_for_lang(dir, lang);
  (title, desc, pos, has_content, has_bilingual)
}

/// 完整提取（额外返回 `sort_children`、`has_content` 与 `has_bilingual`）
#[cfg(feature = "server")]
fn extract_doc_info_full_for_lang(
  dir: &std::path::Path,
  lang: app_core::i18n::Language,
) -> (Option<String>, String, Option<i32>, Option<String>, bool, bool) {
  let zh_path = find_zh_doc_in_dir(dir);
  let en_path = find_en_doc_in_dir(dir);
  let has_bilingual = zh_path.is_some() && en_path.is_some();
  let (primary, fallback) = match lang {
    app_core::i18n::Language::En => (en_path, zh_path),
    app_core::i18n::Language::Zh => (zh_path, en_path),
  };
  let Some(chosen) = primary.or_else(|| fallback.clone()) else {
    return (None, String::new(), None, None, false, false);
  };

  let content = fs::read_to_string(&chosen).unwrap_or_default();
  let (meta, body) = parse_doc_frontmatter(&content, &chosen);

  // 若英文伴生文件漏写了 sidebar_position / sort_children，自动继承同目录主文件的排序设置
  let (fallback_pos, fallback_sort) = fallback
    .filter(|fb| fb != &chosen)
    .and_then(|fb| fs::read_to_string(&fb).ok().map(|raw| parse_doc_frontmatter(&raw, &fb).0))
    .map(|m| (m.sidebar_position, m.sort_children))
    .unwrap_or((None, None));

  // 标题优先级：sidebar_label > frontmatter title > # heading > 目录名
  let title = meta
    .sidebar_label
    .clone()
    .or_else(|| if !meta.title.is_empty() { Some(meta.title.clone()) } else { None })
    .or_else(|| first_heading(&body));

  (
    title,
    meta.description,
    meta.sidebar_position.or(fallback_pos),
    meta.sort_children.or(fallback_sort),
    true,
    has_bilingual,
  )
}

/// 扫描目录生成默认中文视图文档树（递归，最多 3 级）
#[cfg(all(test, feature = "server"))]
pub(crate) fn scan_doc_dir(
  dir: &std::path::Path,
  rel_prefix: &str,
  depth: u32,
) -> Vec<DocTreeNode> {
  scan_doc_dir_for_lang(dir, rel_prefix, depth, app_core::i18n::Language::Zh)
}

/// 按指定语言扫描目录生成文档树（递归，最多 3 级）。
/// 优先级：`_meta.en.json`（英文）/ `_meta.json` > 自动扫描目录（英文优先读 `index.en.{md,mdx}` 回退 `index.{md,mdx}`）。
#[cfg(feature = "server")]
pub(crate) fn scan_doc_dir_for_lang(
  dir: &std::path::Path,
  rel_prefix: &str,
  depth: u32,
  lang: app_core::i18n::Language,
) -> Vec<DocTreeNode> {
  if depth > 3 {
    return vec![];
  }

  // 优先读取 _meta.en.json（英文模式）或 _meta.json
  let meta_en_path = dir.join("_meta.en.json");
  let meta_zh_path = dir.join("_meta.json");
  let chosen_meta = if lang == app_core::i18n::Language::En && meta_en_path.exists() {
    Some(meta_en_path)
  } else if meta_zh_path.exists() {
    Some(meta_zh_path)
  } else {
    None
  };

  let entries: Vec<MetaEntry> = if let Some(meta_path) = chosen_meta {
    fs::read_to_string(&meta_path)
      .ok()
      .and_then(|s| serde_json::from_str(&s).ok())
      .unwrap_or_default()
  } else {
    // 读取当前目录的 sort_children 设置
    let (_, _, _, sort_dir, _, _) = extract_doc_info_full_for_lang(dir, lang);
    let descending = sort_dir.as_deref().map(|s| s.eq_ignore_ascii_case("desc")).unwrap_or(false);

    // 自动扫描子目录，按当前语言提取标题和排序
    let mut dirs: Vec<(String, String, String, Option<i32>)> = fs::read_dir(dir)
      .into_iter()
      .flatten()
      .flatten()
      .filter(|e| e.path().is_dir())
      .filter_map(|e| {
        let name = e.file_name().to_str()?.to_string();
        if name.starts_with('_') || name.starts_with('.') {
          return None;
        }
        let (title, desc, pos, _, _) = extract_doc_info_for_lang(&e.path(), lang);
        let title = title.unwrap_or_else(|| name.clone());
        Some((name, title, desc, pos))
      })
      .collect();
    // 按 sidebar_position 排序，无 position 的按字母顺序排在后面
    dirs.sort_by(|a, b| {
      let ord = match (a.3, b.3) {
        (Some(pa), Some(pb)) => pa.cmp(&pb),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => a.0.cmp(&b.0),
      };
      if descending {
        ord.reverse()
      } else {
        ord
      }
    });
    dirs.into_iter().map(|(slug, title, _, _)| MetaEntry { slug, title }).collect()
  };

  entries
    .into_iter()
    .map(|entry| {
      let child_dir = dir.join(&entry.slug);
      let rel_path = if rel_prefix.is_empty() {
        entry.slug.clone()
      } else {
        format!("{}/{}", rel_prefix, entry.slug)
      };
      let (_, desc, _, has_content, has_bilingual) = extract_doc_info_for_lang(&child_dir, lang);
      let children = if child_dir.is_dir() {
        scan_doc_dir_for_lang(&child_dir, &rel_path, depth + 1, lang)
      } else {
        vec![]
      };
      DocTreeNode {
        slug: entry.slug,
        title: entry.title,
        path: rel_path,
        has_content,
        has_bilingual,
        description: desc,
        children,
      }
    })
    .collect()
}

/// 整棵文档树（约 2000 个节点）。页面不用它：侧栏用 [`doc_nav`]，首页用 [`list_doc_volumes`]；
/// sitemap 在服务端直接调用。
#[post("/api/docs/tree")]
pub async fn list_doc_tree() -> Result<Vec<DocTreeNode>, ServerFnError> {
  #[cfg(feature = "server")]
  {
    Ok((*cached_tree_for_lang(app_core::i18n::Language::Zh)).clone())
  }
  #[cfg(not(feature = "server"))]
  {
    Ok(vec![])
  }
}

/// 侧栏数据：各卷的一级节点，只有 `volume` 这一卷带完整子树（见 [`nav_tree`]）。
#[post("/api/docs/nav")]
pub async fn doc_nav(
  volume: String,
  lang: Option<String>,
) -> Result<Vec<DocTreeNode>, ServerFnError> {
  #[cfg(feature = "server")]
  {
    let target_lang = parse_requested_lang(lang.as_deref());
    Ok(nav_tree(&cached_tree_for_lang(target_lang), &volume))
  }
  #[cfg(not(feature = "server"))]
  {
    let _ = (volume, lang);
    Ok(vec![])
  }
}

/// 文档首页的卷卡片（见 [`volume_cards`]）。
#[post("/api/docs/volumes")]
pub async fn list_doc_volumes(lang: Option<String>) -> Result<Vec<DocVolume>, ServerFnError> {
  #[cfg(feature = "server")]
  {
    let target_lang = parse_requested_lang(lang.as_deref());
    Ok(volume_cards(&cached_tree_for_lang(target_lang)))
  }
  #[cfg(not(feature = "server"))]
  {
    let _ = lang;
    Ok(vec![])
  }
}

/// 文档首页一张卷卡片：卷节点（只留前 [`CARD_CHAPTERS`] 个章、不含更深层）与全卷页数。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DocVolume {
  pub node: DocTreeNode,
  pub pages: usize,
}

/// 卡片列出的章数；多留一个，页面据此判断要不要显示「更多」。
pub const CARD_CHAPTERS: usize = 5;

/// 侧栏用的树：每卷只留一级节点，`volume` 那一卷保留完整子树；description 全部清空
/// （侧栏不显示）。整棵树连 description 序列化约 1.9 MB，曾让每个文档页 HTML 达 4 MB。
pub fn nav_tree(full: &[DocTreeNode], volume: &str) -> Vec<DocTreeNode> {
  fn strip(n: &DocTreeNode, deep: bool) -> DocTreeNode {
    DocTreeNode {
      slug: n.slug.clone(),
      title: n.title.clone(),
      path: n.path.clone(),
      has_content: n.has_content,
      has_bilingual: n.has_bilingual,
      description: String::new(),
      children: if deep { n.children.iter().map(|c| strip(c, true)).collect() } else { Vec::new() },
    }
  }
  full.iter().map(|n| strip(n, n.path == volume)).collect()
}

/// 有正文的节点数（含自身）
pub fn count_pages(node: &DocTreeNode) -> usize {
  node.has_content as usize + node.children.iter().map(count_pages).sum::<usize>()
}

/// 首页卡片：每卷保留 description 与前 `CARD_CHAPTERS + 1` 个章（章下不再展开）。
pub fn volume_cards(full: &[DocTreeNode]) -> Vec<DocVolume> {
  full
    .iter()
    .map(|n| DocVolume {
      pages: count_pages(n),
      node: DocTreeNode {
        children: n
          .children
          .iter()
          .take(CARD_CHAPTERS + 1)
          .map(|c| DocTreeNode { children: Vec::new(), description: String::new(), ..c.clone() })
          .collect(),
        ..n.clone()
      },
    })
    .collect()
}

/// 扫描结果按 (index.{md,mdx} / index.en.{md,mdx} / _meta*.json, mtime) 分中英文缓存：
/// 扫一遍要读约 2000 个文件的 frontmatter，缓存命中时只剩一次目录遍历的 stat。
#[cfg(feature = "server")]
static TREE_CACHE_ZH: app_core::utils::DirListingCache<Vec<DocTreeNode>> =
  app_core::utils::DirListingCache::new();
#[cfg(feature = "server")]
static TREE_CACHE_EN: app_core::utils::DirListingCache<Vec<DocTreeNode>> =
  app_core::utils::DirListingCache::new();

#[cfg(feature = "server")]
fn cached_tree_for_lang(lang: app_core::i18n::Language) -> std::sync::Arc<Vec<DocTreeNode>> {
  let docs_dir = app_core::utils::content_dir("docs");
  if !docs_dir.exists() {
    return std::sync::Arc::new(Vec::new());
  }
  let fp = app_core::utils::fingerprint_for_dir(&docs_dir, |p| {
    matches!(
      p.file_name().and_then(|n| n.to_str()),
      Some(
        "index.md" | "index.mdx" | "index.en.md" | "index.en.mdx" | "_meta.json" | "_meta.en.json"
      )
    )
  });
  let cache = match lang {
    app_core::i18n::Language::En => &TREE_CACHE_EN,
    app_core::i18n::Language::Zh => &TREE_CACHE_ZH,
  };
  cache.get_or_rebuild(fp, || scan_doc_dir_for_lang(&docs_dir, "", 1, lang))
}

/// 文档路径 `path`（相对 docs 根目录，来自公开接口的请求体）对应的源文件解析结果：
/// 返回 `(安全文件路径, 实际命中的语言, 是否具备中英双语伴生文件)`。
/// 整条文件路径都过 `safe_join_under`，目录或文件本身是指向根外的符号链接也会被拒绝（SEC-03）。
#[cfg(feature = "server")]
fn resolve_doc_file_for_lang(
  docs_root: &std::path::Path,
  path: &str,
  lang: app_core::i18n::Language,
) -> Option<(PathBuf, app_core::i18n::Language, bool)> {
  let dir = path.trim_end_matches('/');
  let find_in_names = |names: &[&str]| {
    names.iter().find_map(|name| {
      let rel = if dir.is_empty() { (*name).to_string() } else { format!("{dir}/{name}") };
      app_core::utils::safe_join_under(docs_root, &rel).filter(|p| p.is_file())
    })
  };
  let zh_file = find_in_names(&["index.md", "index.mdx"]);
  let en_file = find_in_names(&["index.en.md", "index.en.mdx"]);
  let has_bilingual = zh_file.is_some() && en_file.is_some();
  match lang {
    app_core::i18n::Language::En => en_file
      .map(|f| (f, app_core::i18n::Language::En, has_bilingual))
      .or_else(|| zh_file.map(|f| (f, app_core::i18n::Language::Zh, false))),
    app_core::i18n::Language::Zh => zh_file
      .map(|f| (f, app_core::i18n::Language::Zh, has_bilingual))
      .or_else(|| en_file.map(|f| (f, app_core::i18n::Language::En, false))),
  }
}

#[cfg(all(test, feature = "server"))]
fn resolve_doc_file(docs_root: &std::path::Path, path: &str) -> Option<PathBuf> {
  resolve_doc_file_for_lang(docs_root, path, app_core::i18n::Language::Zh).map(|(p, _, _)| p)
}

/// 文档页的最后修改日期（`index.md` 的 mtime，UTC `YYYY-MM-DD`），给 sitemap 的 lastmod。
/// 内容随 git / rsync 部署，只有改过的文件 mtime 会变。
#[cfg(feature = "server")]
pub fn doc_lastmod(path: &str) -> Option<String> {
  doc_lastmod_for_lang(path, app_core::i18n::Language::Zh)
}

/// 指定语言版本的文档最后修改日期（给双语 `/docs/...?lang=en` sitemap 条目用）。
#[cfg(feature = "server")]
pub fn doc_lastmod_for_lang(path: &str, lang: app_core::i18n::Language) -> Option<String> {
  let (file, _, _) = resolve_doc_file_for_lang(&app_core::utils::content_dir("docs"), path, lang)?;
  let modified = std::fs::metadata(file).ok()?.modified().ok()?;
  Some(chrono::DateTime::<chrono::Utc>::from(modified).format("%Y-%m-%d").to_string())
}

/// `path` 在它所属的一级分类里按侧栏顺序（先序：分类首页、章、章下的节）的前后两篇，
/// 只取有正文的节点，不跨分类（S2）。`tree` 是整棵文档树（`scan_doc_dir(root, "", 1)`）。
#[cfg(feature = "server")]
fn doc_neighbors(tree: &[DocTreeNode], path: &str) -> (Option<DocLink>, Option<DocLink>) {
  let path = path.trim_end_matches('/');
  let Some(top) = path.split('/').next().filter(|s| !s.is_empty()) else {
    return (None, None);
  };
  let Some(root) = tree.iter().find(|n| n.path == top) else {
    return (None, None);
  };
  let mut pages = Vec::new();
  collect_pages(root, &mut pages);
  let Some(i) = pages.iter().position(|n| n.path == path) else {
    return (None, None);
  };
  let link = |n: &DocTreeNode| DocLink { path: n.path.clone(), title: n.title.clone() };
  (i.checked_sub(1).map(|j| link(pages[j])), pages.get(i + 1).map(|n| link(n)))
}

#[cfg(feature = "server")]
fn collect_pages<'a>(node: &'a DocTreeNode, out: &mut Vec<&'a DocTreeNode>) {
  if node.has_content {
    out.push(node);
  }
  for child in &node.children {
    collect_pages(child, out);
  }
}

/// 在目录树中查找从顶层卷到 `target_path` 的面包屑节点链。
/// 无独立正文（`!has_content`）的纯分组节点将 `DocLink::path` 置空，避免渲染死链。
#[cfg(feature = "server")]
pub(crate) fn doc_breadcrumbs(tree: &[DocTreeNode], target_path: &str) -> Vec<DocLink> {
  fn find_chain(nodes: &[DocTreeNode], target: &str, out: &mut Vec<DocLink>) -> bool {
    for n in nodes {
      if target == n.path || target.starts_with(&format!("{}/", n.path)) {
        out.push(DocLink {
          path: if n.has_content { n.path.clone() } else { String::new() },
          title: n.title.clone(),
        });
        if target == n.path || find_chain(&n.children, target, out) {
          return true;
        }
        out.pop();
      }
    }
    false
  }

  let trimmed = target_path.trim_matches('/');
  let mut chain = Vec::new();
  find_chain(tree, trimmed, &mut chain);
  chain
}

/// 为文档子章节补齐所属卷名后缀（SEO `<title>` / `og:title` / `TechArticle.headline`）。
///
/// 章节 frontmatter 常写简短目录名（如 `常见坑`、`路由`、`设计取舍`），直接用作页面 `<title>`
/// 会导致全站大量重复且丢失库名相关性。仅在 `path` 为子路径（含 `/`）且 `title` 未包含
/// `volume_slug` 或 `volume_label`（忽略大小写）时追加 ` · {volume_label}`。
#[cfg(feature = "server")]
pub(crate) fn qualify_doc_title(title: &str, path: &str, volume_label: Option<&str>) -> String {
  let trimmed = path.trim_matches('/');
  let Some((vol_slug, _)) = trimmed.split_once('/') else {
    return title.to_string();
  };
  let label = volume_label.filter(|s| !s.is_empty()).unwrap_or(vol_slug);
  if title.is_empty() || label.is_empty() {
    return title.to_string();
  }
  let low = title.to_lowercase();
  if low.contains(&label.to_lowercase()) || low.contains(&vol_slug.to_lowercase()) {
    return title.to_string();
  }
  format!("{title} · {label}")
}

/// 文档正文与 frontmatter；`Ok(None)` 表示没有这篇文档（页面据此回 404）。
#[post("/api/docs/content")]
pub async fn get_doc_content(
  path: String,
  lang: Option<String>,
) -> Result<Option<DocContentResponse>, ServerFnError> {
  #[cfg(feature = "server")]
  {
    let target_lang = parse_requested_lang(lang.as_deref());
    let docs_root = app_core::utils::content_dir("docs");
    let Some((filepath, actual_lang, has_bilingual)) =
      resolve_doc_file_for_lang(&docs_root, &path, target_lang)
    else {
      return Ok(None);
    };
    let raw =
      fs::read_to_string(&filepath).map_err(|e| ServerFnError::new(format!("读取失败: {}", e)))?;
    let (mut meta, content) = parse_doc_frontmatter(&raw, &filepath);
    fill_missing_meta(&mut meta, &content);
    let tree = cached_tree_for_lang(target_lang);
    let vol_slug = path.trim_matches('/').split('/').next().unwrap_or_default();
    let vol_label = tree.iter().find(|n| n.path == vol_slug).map(|n| n.title.as_str());
    meta.title = qualify_doc_title(&meta.title, &path, vol_label);
    let (prev, next) = doc_neighbors(&tree, &path);
    let breadcrumbs = doc_breadcrumbs(&tree, &path);
    let source_file = docs_root
      .canonicalize()
      .ok()
      .and_then(|canon_root| filepath.strip_prefix(canon_root).ok())
      .or_else(|| filepath.strip_prefix(&docs_root).ok())
      .map(|rel| {
        rel.components().filter_map(|c| c.as_os_str().to_str()).collect::<Vec<_>>().join("/")
      })
      .unwrap_or_default();
    let resolved_lang = match actual_lang {
      app_core::i18n::Language::En => "en".to_string(),
      app_core::i18n::Language::Zh => "zh-CN".to_string(),
    };
    Ok(Some(DocContentResponse {
      content,
      meta,
      prev,
      next,
      breadcrumbs,
      source_file,
      resolved_lang,
      has_bilingual,
    }))
  }
  #[cfg(not(feature = "server"))]
  {
    let _ = (path, lang);
    Ok(None)
  }
}

// ========== Tests ==========

// clippy 的 allow-unwrap-in-tests 只认 `#[cfg(test)]`，组合 cfg 要手动放开
#[cfg(all(test, feature = "server"))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
  use super::*;
  use std::path::Path;
  use tempfile::TempDir;

  /// 辅助：在指定目录下创建 index.md（可选带 frontmatter）
  fn write_index(dir: &Path, frontmatter: Option<&str>, body: &str) {
    std::fs::create_dir_all(dir).unwrap();
    let content = match frontmatter {
      Some(fm) => format!("---\n{}\n---\n\n{}", fm, body),
      None => body.to_string(),
    };
    std::fs::write(dir.join("index.md"), content).unwrap();
  }

  /// Y8：缺 frontmatter 时 title 取首个 `#` 标题，description 取首段纯文本；已写的不覆盖。
  #[test]
  fn missing_meta_filled_from_body() {
    let body = "# Handler 函数\n\n```rust\nfn a() {}\n```\n\n- 列表\n\nHandler 是处理 **HTTP** 请求的 `async` 函数：\n";
    let mut meta = DocMeta::default();
    fill_missing_meta(&mut meta, body);
    assert_eq!(meta.title, "Handler 函数");
    assert_eq!(meta.description, "Handler 是处理 HTTP 请求的 async 函数：");
    let mut meta = DocMeta { title: "T".into(), description: "D".into(), ..Default::default() };
    fill_missing_meta(&mut meta, body);
    assert_eq!((meta.title.as_str(), meta.description.as_str()), ("T", "D"));
  }

  #[test]
  fn qualify_doc_title_appends_volume_only_for_unqualified_subpages() {
    // 卷首页不追加后缀
    assert_eq!(qualify_doc_title("axum 源码解析", "axum", Some("axum")), "axum 源码解析");
    // 子章节未包含卷名时自动补齐 ` · <volume_label>`
    assert_eq!(qualify_doc_title("常见坑", "axum/10-pitfalls", Some("axum")), "常见坑 · axum");
    assert_eq!(
      qualify_doc_title("张量", "pytorch-to-burn/02-tensor", Some("从 PyTorch 到 burn")),
      "张量 · 从 PyTorch 到 burn"
    );
    // 已包含卷 label 或目录 slug（忽略大小写）时不重复追加
    assert_eq!(
      qualify_doc_title("一个请求穿过 axum", "axum/01-overview/req", Some("axum")),
      "一个请求穿过 axum"
    );
    assert_eq!(
      qualify_doc_title("Typst 脚本基础", "typst/03-scripting", Some("Typst 教程")),
      "Typst 脚本基础"
    );
  }

  /// 辅助：提取一棵子树的 slug 列表（保留顺序）
  fn slugs(nodes: &[DocTreeNode]) -> Vec<String> {
    nodes.iter().map(|n| n.slug.clone()).collect()
  }

  fn find<'a>(nodes: &'a [DocTreeNode], slug: &str) -> &'a DocTreeNode {
    nodes.iter().find(|n| n.slug == slug).unwrap_or_else(|| panic!("未找到节点: {}", slug))
  }

  #[test]
  fn nav_tree_keeps_only_the_active_volume_deep() {
    let tmp = TempDir::new().unwrap();
    let docs = tmp.path();
    write_index(&docs.join("tokio"), Some("title: tokio\ndescription: d"), "");
    write_index(&docs.join("tokio/01-a"), Some("title: A\ndescription: da"), "");
    write_index(&docs.join("tokio/01-a/x"), Some("title: X"), "");
    write_index(&docs.join("axum"), Some("title: axum"), "");
    write_index(&docs.join("axum/01-b"), Some("title: B"), "");
    let full = scan_doc_dir(docs, "", 1);
    let nav = nav_tree(&full, "tokio");
    let tokio = nav.iter().find(|n| n.path == "tokio").unwrap();
    let axum = nav.iter().find(|n| n.path == "axum").unwrap();
    assert_eq!(tokio.children[0].children[0].path, "tokio/01-a/x");
    assert!(axum.children.is_empty(), "其它卷只留一级节点");
    assert!(tokio.description.is_empty() && tokio.children[0].description.is_empty());
    assert!(nav_tree(&full, "").iter().all(|n| n.children.is_empty()));

    let cards = volume_cards(&full);
    let t = cards.iter().find(|c| c.node.path == "tokio").unwrap();
    assert_eq!(t.pages, 3);
    assert_eq!(t.node.description, "d");
    assert!(t.node.children[0].children.is_empty());
  }

  #[test]
  fn neighbors_follow_sidebar_order_within_one_category() {
    let tmp = TempDir::new().unwrap();
    let docs = tmp.path();
    write_index(&docs.join("tokio"), Some("title: tokio"), "");
    write_index(&docs.join("tokio/02-b/y"), Some("title: Y\nsidebar_position: 1"), "");
    write_index(&docs.join("tokio/01-a"), Some("title: A\nsidebar_position: 1"), "");
    write_index(&docs.join("tokio/01-a/x"), Some("title: X"), "");
    write_index(&docs.join("axum"), Some("title: axum"), "");
    // 02-b 没有 index.md：不是一页，跳过
    let tree = scan_doc_dir(docs, "", 1);
    let order = |p: &str| {
      let (prev, next) = doc_neighbors(&tree, p);
      (prev.map(|l| l.path), next.map(|l| l.path))
    };
    assert_eq!(order("tokio"), (None, Some("tokio/01-a".into())));
    assert_eq!(order("tokio/01-a"), (Some("tokio".into()), Some("tokio/01-a/x".into())));
    assert_eq!(order("tokio/01-a/x"), (Some("tokio/01-a".into()), Some("tokio/02-b/y".into())));
    assert_eq!(order("tokio/02-b/y/"), (Some("tokio/01-a/x".into()), None), "不跨到 axum");
    assert_eq!(order("axum"), (None, None));
    assert_eq!(order(""), (None, None));
    assert_eq!(order("tokio/nope"), (None, None));
    let (_, next) = doc_neighbors(&tree, "tokio");
    assert_eq!(next.map(|l| l.title), Some("A".to_string()));
  }

  #[test]
  fn test_frontmatter_parsing() {
    let raw = "---\ntitle: Hello\nkeywords: [a, b]\nsidebar_position: 5\n---\n\n# body";
    let (meta, body) = parse_doc_frontmatter(raw, Path::new("t.md"));
    assert_eq!(meta.title, "Hello");
    assert_eq!(meta.keywords, vec!["a", "b"]);
    assert_eq!(meta.sidebar_position, Some(5));
    assert!(body.trim_start().starts_with("# body"));
  }

  #[test]
  fn source_frontmatter_parsed() {
    let (meta, _) = parse_doc_frontmatter(
      "---\ntitle: T\nsource: { crate: tokio, version: 1.54.0 }\n---\nbody",
      Path::new("t.md"),
    );
    assert_eq!(meta.source, Some(DocSource { krate: "tokio".into(), version: "1.54.0".into() }));
    assert_eq!(meta.title, "T");
    let (meta, _) = parse_doc_frontmatter(
      "---\ntitle: T\nsource: { crate: axum, version: 1.10 }\n---\n",
      Path::new("t.md"),
    );
    assert_eq!(meta.source.map(|s| s.version), Some("1.10".to_string()));
  }

  #[test]
  fn test_no_frontmatter_returns_default() {
    let (meta, body) = parse_doc_frontmatter("# Just heading\n\nsome body", Path::new("t.md"));
    assert_eq!(meta, DocMeta::default());
    assert!(body.contains("# Just heading"));
  }

  #[test]
  fn test_default_ascending_by_position() {
    let tmp = TempDir::new().unwrap();
    let docs = tmp.path();

    write_index(&docs.join("a"), Some("title: A\nsidebar_position: 3"), "# A");
    write_index(&docs.join("b"), Some("title: B\nsidebar_position: 1"), "# B");
    write_index(&docs.join("c"), Some("title: C\nsidebar_position: 2"), "# C");

    let tree = scan_doc_dir(docs, "", 1);
    assert_eq!(slugs(&tree), vec!["b", "c", "a"]);
  }

  #[test]
  fn test_descending_via_sort_children() {
    let tmp = TempDir::new().unwrap();
    let docs = tmp.path();

    // 父目录创建 index.md，并设置 sort_children: desc
    write_index(docs, Some("sort_children: desc"), "# root");

    write_index(&docs.join("issue-001"), Some("sidebar_position: 1"), "# 1");
    write_index(&docs.join("issue-002"), Some("sidebar_position: 2"), "# 2");
    write_index(&docs.join("issue-003"), Some("sidebar_position: 3"), "# 3");
    write_index(&docs.join("issue-005"), Some("sidebar_position: 5"), "# 5");
    write_index(&docs.join("issue-004"), Some("sidebar_position: 4"), "# 4");

    let tree = scan_doc_dir(docs, "", 1);
    assert_eq!(slugs(&tree), vec!["issue-005", "issue-004", "issue-003", "issue-002", "issue-001"]);
  }

  #[test]
  fn test_sort_children_case_insensitive() {
    let tmp = TempDir::new().unwrap();
    let docs = tmp.path();

    // 大写 DESC 也应被识别
    write_index(docs, Some("sort_children: DESC"), "# root");
    write_index(&docs.join("v1"), Some("sidebar_position: 1"), "# 1");
    write_index(&docs.join("v2"), Some("sidebar_position: 2"), "# 2");

    let tree = scan_doc_dir(docs, "", 1);
    assert_eq!(slugs(&tree), vec!["v2", "v1"]);
  }

  #[test]
  fn test_no_position_falls_back_to_alphabetical() {
    let tmp = TempDir::new().unwrap();
    let docs = tmp.path();

    // 都没有 sidebar_position
    write_index(&docs.join("zebra"), None, "# Zebra");
    write_index(&docs.join("apple"), None, "# Apple");
    write_index(&docs.join("mango"), None, "# Mango");

    let tree = scan_doc_dir(docs, "", 1);
    assert_eq!(slugs(&tree), vec!["apple", "mango", "zebra"]);
  }

  #[test]
  fn test_mixed_position_and_no_position() {
    let tmp = TempDir::new().unwrap();
    let docs = tmp.path();

    write_index(&docs.join("first"), Some("sidebar_position: 1"), "# 1");
    write_index(&docs.join("middle"), Some("sidebar_position: 5"), "# 5");
    write_index(&docs.join("zzz"), None, "# zzz");
    write_index(&docs.join("aaa"), None, "# aaa");

    let tree = scan_doc_dir(docs, "", 1);
    // 有 position 的在前面（按 position 排），无 position 的在后面（按字母排）
    assert_eq!(slugs(&tree), vec!["first", "middle", "aaa", "zzz"]);
  }

  #[test]
  fn test_three_level_nesting_with_independent_sort() {
    let tmp = TempDir::new().unwrap();
    let docs = tmp.path();

    // 一级：axum（默认 asc）与 weekly（desc）
    write_index(&docs.join("axum"), Some("sidebar_position: 1"), "# Axum");
    write_index(&docs.join("weekly"), Some("sidebar_position: 2\nsort_children: desc"), "# Weekly");

    // 二级：axum/basic（默认）与 axum/advance
    write_index(&docs.join("axum/basic"), Some("sidebar_position: 1"), "# basic");
    write_index(&docs.join("axum/advance"), Some("sidebar_position: 2"), "# advance");

    // 三级：axum/basic/router 与 handler
    write_index(&docs.join("axum/basic/router"), Some("sidebar_position: 1"), "# router");
    write_index(&docs.join("axum/basic/handler"), Some("sidebar_position: 2"), "# handler");

    // weekly 子项：递增编号
    write_index(&docs.join("weekly/issue-001"), Some("sidebar_position: 1"), "# 1");
    write_index(&docs.join("weekly/issue-002"), Some("sidebar_position: 2"), "# 2");
    write_index(&docs.join("weekly/issue-003"), Some("sidebar_position: 3"), "# 3");

    let tree = scan_doc_dir(docs, "", 1);

    // 顶层：axum 排在 weekly 前（默认 asc，不受子项的 sort_children 影响）
    assert_eq!(slugs(&tree), vec!["axum", "weekly"]);

    // axum 下面 —— 默认升序
    let axum = find(&tree, "axum");
    assert_eq!(slugs(&axum.children), vec!["basic", "advance"]);

    // axum/basic 下面 —— 默认升序
    let basic = find(&axum.children, "basic");
    assert_eq!(slugs(&basic.children), vec!["router", "handler"]);

    // weekly 下面 —— desc 降序
    let weekly = find(&tree, "weekly");
    assert_eq!(slugs(&weekly.children), vec!["issue-003", "issue-002", "issue-001"]);
  }

  #[test]
  fn test_sort_children_only_affects_direct_children() {
    let tmp = TempDir::new().unwrap();
    let docs = tmp.path();

    // 父目录 sort_children: desc
    write_index(docs, Some("sort_children: desc"), "# root");

    // 一级：子项 a/b 会被逆序
    write_index(&docs.join("a"), Some("sidebar_position: 1"), "# a");
    write_index(&docs.join("b"), Some("sidebar_position: 2"), "# b");

    // 二级：a 下面的 a-1/a-2 仍然应该是升序（父的 sort_children 不会传递）
    write_index(&docs.join("a/a-1"), Some("sidebar_position: 1"), "# a1");
    write_index(&docs.join("a/a-2"), Some("sidebar_position: 2"), "# a2");

    let tree = scan_doc_dir(docs, "", 1);
    // 一级逆序
    assert_eq!(slugs(&tree), vec!["b", "a"]);
    // 二级仍然升序
    let a = find(&tree, "a");
    assert_eq!(slugs(&a.children), vec!["a-1", "a-2"]);
  }

  #[test]
  fn test_path_propagation_in_nested_tree() {
    let tmp = TempDir::new().unwrap();
    let docs = tmp.path();

    write_index(&docs.join("axum"), Some("sidebar_position: 1"), "# Axum");
    write_index(&docs.join("axum/basic"), Some("sidebar_position: 1"), "# basic");
    write_index(&docs.join("axum/basic/router"), Some("sidebar_position: 1"), "# router");

    let tree = scan_doc_dir(docs, "", 1);
    let axum = find(&tree, "axum");
    assert_eq!(axum.path, "axum");
    let basic = find(&axum.children, "basic");
    assert_eq!(basic.path, "axum/basic");
    let router = find(&basic.children, "router");
    assert_eq!(router.path, "axum/basic/router");
  }

  #[test]
  fn test_max_depth_three_levels() {
    let tmp = TempDir::new().unwrap();
    let docs = tmp.path();

    // 创建 4 级嵌套，验证只扫描前 3 级
    write_index(&docs.join("l1/l2/l3/l4"), None, "# deep");
    write_index(&docs.join("l1/l2/l3"), None, "# l3");
    write_index(&docs.join("l1/l2"), None, "# l2");
    write_index(&docs.join("l1"), None, "# l1");

    let tree = scan_doc_dir(docs, "", 1);
    let l1 = find(&tree, "l1");
    let l2 = find(&l1.children, "l2");
    let l3 = find(&l2.children, "l3");
    // l3 是第 3 级，其 children 应为空（第 4 级被截断）
    assert!(l3.children.is_empty(), "超过 3 级的节点应不被扫描");
  }

  #[test]
  fn test_sidebar_label_overrides_title() {
    let tmp = TempDir::new().unwrap();
    let docs = tmp.path();

    write_index(
      &docs.join("x"),
      Some("title: Long Title For SEO\nsidebar_label: Short"),
      "# heading",
    );

    let tree = scan_doc_dir(docs, "", 1);
    let x = find(&tree, "x");
    // 侧栏显示使用 sidebar_label
    assert_eq!(x.title, "Short");
  }

  #[test]
  fn test_title_falls_back_to_h1_then_slug() {
    let tmp = TempDir::new().unwrap();
    let docs = tmp.path();

    // 无 frontmatter，只有 # 标题
    write_index(&docs.join("with-h1"), None, "# From Heading");
    // 有 frontmatter 但不含标题相关字段，也没 h1
    write_index(&docs.join("no-title"), Some("description: just desc"), "some body");

    let tree = scan_doc_dir(docs, "", 1);
    assert_eq!(find(&tree, "with-h1").title, "From Heading");
    // 都拿不到时退化到目录名
    assert_eq!(find(&tree, "no-title").title, "no-title");
  }

  #[test]
  fn test_underscore_and_hidden_dirs_skipped() {
    let tmp = TempDir::new().unwrap();
    let docs = tmp.path();

    write_index(&docs.join("visible"), None, "# visible");
    write_index(&docs.join("_private"), None, "# private");
    write_index(&docs.join(".hidden"), None, "# hidden");

    let tree = scan_doc_dir(docs, "", 1);
    assert_eq!(slugs(&tree), vec!["visible"]);
  }

  #[test]
  fn test_empty_dir_returns_empty_tree() {
    let tmp = TempDir::new().unwrap();
    let tree = scan_doc_dir(tmp.path(), "", 1);
    assert!(tree.is_empty());
  }

  #[test]
  fn resolve_doc_file_stays_inside_docs_root() {
    // SEC-03：公开接口，path 来自请求体
    let tmp = TempDir::new().unwrap();
    let docs = tmp.path().join("docs");
    write_index(&docs.join("guide/intro"), None, "intro");
    write_index(&docs.join("mdx-only"), None, "x");
    std::fs::rename(docs.join("mdx-only/index.md"), docs.join("mdx-only/index.mdx")).unwrap();
    write_index(&tmp.path().join("courses/paid/lesson"), None, "secret");
    write_index(&tmp.path().join("outside"), None, "secret");
    #[cfg(unix)]
    std::os::unix::fs::symlink(tmp.path().join("outside"), docs.join("link")).unwrap();

    assert!(resolve_doc_file(&docs, "guide/intro").is_some());
    assert!(resolve_doc_file(&docs, "mdx-only").is_some_and(|p| p.ends_with("index.mdx")));
    let outside = tmp.path().join("outside");
    for bad in [
      "../courses/paid/lesson",
      "guide/../../courses/paid/lesson",
      outside.to_str().unwrap(),
      "..\\courses\\paid\\lesson",
      "link",
      "%2e%2e/courses/paid/lesson",
      "guide/missing",
    ] {
      assert_eq!(resolve_doc_file(&docs, bad), None, "{bad}");
    }
  }

  #[test]
  fn doc_breadcrumbs_handles_group_and_content_nodes() {
    let tmp = TempDir::new().unwrap();
    let docs = tmp.path();

    // guide 有 index.md；guide/advanced 仅作分组目录无 index.md；guide/advanced/ssr 有 index.md
    write_index(&docs.join("guide"), Some("title: 快速上手"), "# Guide");
    write_index(&docs.join("guide/advanced/ssr"), Some("title: SSR 水合"), "# SSR");

    let tree = scan_doc_dir(docs, "", 1);
    let crumbs = doc_breadcrumbs(&tree, "guide/advanced/ssr");
    assert_eq!(
      crumbs,
      vec![
        DocLink { path: "guide".to_string(), title: "快速上手".to_string() },
        DocLink { path: String::new(), title: "advanced".to_string() },
        DocLink { path: "guide/advanced/ssr".to_string(), title: "SSR 水合".to_string() },
      ]
    );
  }

  #[test]
  fn bilingual_docs_prefer_en_companion_and_smoothly_fallback_to_zh() {
    use app_core::i18n::Language;

    let tmp = TempDir::new().unwrap();
    let docs = tmp.path();

    // guide 卷与 quickstart 均提供 index.md + index.en.md；zh-only 仅有 index.md
    write_index(
      &docs.join("guide"),
      Some("sidebar_label: 快速上手\nsidebar_position: 1"),
      "# 导览",
    );
    std::fs::write(
      docs.join("guide/index.en.md"),
      "---\nsidebar_label: Getting Started\n---\n\n# Guide Overview\n",
    )
    .unwrap();

    write_index(
      &docs.join("guide/quickstart"),
      Some("sidebar_label: 5 分钟快速开始\nsidebar_position: 1"),
      "# 中文快速开始",
    );
    std::fs::write(
      docs.join("guide/quickstart/index.en.md"),
      "---\nsidebar_label: 5-Minute Quickstart\ndescription: Start in 50s\n---\n\n# Quickstart\n",
    )
    .unwrap();

    write_index(
      &docs.join("guide/zh-only"),
      Some("sidebar_label: 仅中文专题\nsidebar_position: 2"),
      "# 仅中文正文",
    );

    let tree_zh = scan_doc_dir_for_lang(docs, "", 1, Language::Zh);
    let tree_en = scan_doc_dir_for_lang(docs, "", 1, Language::En);

    assert_eq!(tree_zh[0].title, "快速上手");
    assert!(tree_zh[0].has_bilingual);
    assert_eq!(tree_zh[0].children[0].title, "5 分钟快速开始");
    assert!(tree_zh[0].children[0].has_bilingual);
    assert_eq!(tree_zh[0].children[1].title, "仅中文专题");
    assert!(!tree_zh[0].children[1].has_bilingual);

    // 英文树：有 .en.md 的用英文标题（且继承了主文件的 sidebar_position = 1），缺失的平滑回退中文
    assert_eq!(tree_en[0].title, "Getting Started");
    assert_eq!(slugs(&tree_en[0].children), vec!["quickstart", "zh-only"]);
    assert_eq!(tree_en[0].children[0].title, "5-Minute Quickstart");
    assert_eq!(tree_en[0].children[1].title, "仅中文专题");

    // 文件解析：英文请求命中 index.en.md 时返回 Language::En + has_bilingual = true
    let (en_file, actual_en, bi_en) =
      resolve_doc_file_for_lang(docs, "guide/quickstart", Language::En).unwrap();
    assert!(en_file.ends_with("index.en.md"));
    assert_eq!(actual_en, Language::En);
    assert!(bi_en);

    // 英文请求访问无伴生文件的中文目录时，平滑回退 index.md 且标明 Language::Zh + has_bilingual = false
    let (fb_file, actual_fb, bi_fb) =
      resolve_doc_file_for_lang(docs, "guide/zh-only", Language::En).unwrap();
    assert!(fb_file.ends_with("index.md"));
    assert_eq!(actual_fb, Language::Zh);
    assert!(!bi_fb);
  }
}
