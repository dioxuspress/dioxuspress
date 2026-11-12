//! 公共工具函数，跨 app / module 重用。

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

/// 自动探测资产根目录。
///
/// 启动方式不同（cargo run / dx serve / cargo test），运行目录可能是
/// 项目根目录（`./assets` 存在）或 crate 子目录（需上溯 `../../assets`）。
/// 调用方无需关心：本函数会先查相对路径 `assets`，否则退回到 `../../assets`。
///
/// 返回的 `PathBuf` 不保证存在；调用方需根据上下文做存在性检查。
pub fn get_asset_root() -> PathBuf {
  let mut path = PathBuf::from("assets");
  if !path.exists() {
    path = PathBuf::from("../../assets");
  }
  path
}

/// 容器默认内容挂载路径：用户执行 `docker run -v ./my-site:/app/content:ro ...`
/// 时，即使未显式传 `-e CONTENT_DIR=/app/content`，只要 `/app/content` 非空即自动启用。
pub const DEFAULT_DOCKER_CONTENT_DIR: &str = "/app/content";

/// 返回当前生效的外部内容根目录（`CONTENT_DIR` 环境变量优先；未设时若 `/app/content`
/// 存在且非空则自动命中，否则返回 `None` 回退到框架内置 `assets/`）。
pub fn content_root() -> Option<PathBuf> {
  content_root_from(std::env::var_os("CONTENT_DIR"), Path::new(DEFAULT_DOCKER_CONTENT_DIR))
}

fn content_root_from(
  content_env: Option<std::ffi::OsString>,
  default_mount: &Path,
) -> Option<PathBuf> {
  if let Some(root) = content_env.filter(|v| !v.is_empty()) {
    return Some(PathBuf::from(root));
  }
  if default_mount.is_dir()
    && std::fs::read_dir(default_mount).ok().and_then(|mut it| it.next()).is_some()
  {
    return Some(default_mount.to_path_buf());
  }
  None
}

/// 站点内容目录：`kind` 为 `posts`（文章）、`docs`（文档卷）、`courses`（课程）或 `images`（品牌与配图）。
///
/// 设了 `CONTENT_DIR`（或容器挂载了非空 `/app/content`）就读 `<content_root>/<kind>`——站点自己的内容
/// 放在框架仓库之外；否则用资产根下随框架发布的示例内容。
pub fn content_dir(kind: &str) -> PathBuf {
  content_dir_in(std::env::var_os("CONTENT_DIR"), kind)
}

fn content_dir_in(content_env: Option<std::ffi::OsString>, kind: &str) -> PathBuf {
  match content_root_from(content_env, Path::new(DEFAULT_DOCKER_CONTENT_DIR)) {
    Some(root) => root.join(kind),
    None => get_asset_root().join(kind),
  }
}

/// 当前生效的 `site.json` 磁盘路径：优先取 `<content_root>/site.json`（若该文件存在），
/// 否则回退到 `<asset_root>/site.json`。
pub fn site_json_path() -> PathBuf {
  site_json_path_in(&get_asset_root(), content_root().as_deref())
}

/// 在给定资产根与可选内容根下解析生效的 `site.json` 路径。
pub fn site_json_path_in(asset_root: &Path, content_root: Option<&Path>) -> PathBuf {
  if let Some(root) = content_root {
    let candidate = root.join("site.json");
    if candidate.is_file() {
      return candidate;
    }
  }
  asset_root.join("site.json")
}

/// Phase 8.5：目录扫描结果的 mtime 缓存。
///
/// 用于 `list_blog_posts` / `list_*_articles` / sitemap entries 等 hot path：
/// 这些函数 walk 资产目录、parse frontmatter / YAML，每页请求都跑一次成本
/// 不低（5–20 ms 的 IO + serde_yaml）。引入本缓存后：
/// - 命中：一次 fingerprint 计算（stat 每个文件），返回上次的 Arc<T>
/// - 未命中：调 builder 重新扫，写回缓存
///
/// 「fingerprint」由调用方负责生成 —— 通常是 sorted `(path, mtime)` 列表的
/// 包含哈希。文件添加 / 删除 / 修改任意一种都会让 fingerprint 变。
///
/// 类型参数 `T` 是缓存值；要求 `Send + Sync` 以便跨 tokio worker 共享。
pub struct DirListingCache<T: Send + Sync> {
  inner: Mutex<Option<CacheEntry<T>>>,
}

struct CacheEntry<T> {
  fingerprint: Vec<(PathBuf, SystemTime)>,
  value: Arc<T>,
}

impl<T: Send + Sync> Default for DirListingCache<T> {
  fn default() -> Self {
    Self::new()
  }
}

impl<T: Send + Sync> DirListingCache<T> {
  pub const fn new() -> Self {
    Self { inner: Mutex::new(None) }
  }

  /// 取出缓存值，或在 fingerprint 不一致时调 `builder` 重新构建。
  ///
  /// `fingerprint` 通常通过 [`fingerprint_for_dir`] 之类的工具生成；
  /// 调用方只要保证「同 fingerprint == 同 builder 输出」，缓存就一致。
  pub fn get_or_rebuild<F>(&self, fingerprint: Vec<(PathBuf, SystemTime)>, builder: F) -> Arc<T>
  where
    F: FnOnce() -> T,
  {
    if let Ok(guard) = self.inner.lock() {
      if let Some(entry) = guard.as_ref() {
        if entry.fingerprint == fingerprint {
          return entry.value.clone();
        }
      }
    }
    // 锁外 build：避免持锁跑 IO / parse
    let value = Arc::new(builder());
    if let Ok(mut guard) = self.inner.lock() {
      *guard = Some(CacheEntry { fingerprint, value: value.clone() });
    }
    value
  }

  /// 显式失效。一般不需要（fingerprint 失配自动重建），但 admin 上传 / 删除
  /// 资产后想强制刷新时可调用。
  pub fn invalidate(&self) {
    if let Ok(mut guard) = self.inner.lock() {
      *guard = None;
    }
  }
}

/// 给定目录 + 匹配条件，收集所有匹配文件的 `(PathBuf, mtime)` 作为 fingerprint。
///
/// 设计上 fingerprint 走 *排序后* 的列表以保证稳定性；任何文件添加 / 删除 / 修改
/// 都会让 fingerprint 变。
///
/// `match_fn` 用于过滤要纳入指纹的文件路径（例如 `path.ends_with("index.mdx")`）。
/// 错误路径（permission denied / 文件被删除）静默跳过，避免 cache 因偶发 IO 错
/// 整体 reject。
pub fn fingerprint_for_dir<P, F>(root: P, mut match_fn: F) -> Vec<(PathBuf, SystemTime)>
where
  P: AsRef<Path>,
  F: FnMut(&Path) -> bool,
{
  let mut fp: Vec<(PathBuf, SystemTime)> = Vec::new();
  let mut stack: Vec<PathBuf> = vec![root.as_ref().to_path_buf()];
  while let Some(dir) = stack.pop() {
    let Ok(entries) = std::fs::read_dir(&dir) else { continue };
    for entry in entries.flatten() {
      let path = entry.path();
      let Ok(meta) = entry.metadata() else { continue };
      if meta.is_dir() {
        stack.push(path);
      } else if match_fn(&path) {
        if let Ok(mtime) = meta.modified() {
          fp.push((path, mtime));
        }
      }
    }
  }
  fp.sort();
  fp
}

/// 把请求里的相对路径 `raw` 拼到 `sub_root` 下，确认没有逃出 `sub_root`。
///
/// 防御 `../../etc/passwd` 与符号链接逃逸（forum Phase 8.2 引用接口、docs
/// SEC-03 文档接口共用）：
/// 1. 含 `..`、以 `/` 或 `\\` 开头的输入直接拒绝（canonicalize 之前就排除歧义）。
/// 2. 目标存在时 canonicalize，必须仍以 `sub_root.canonicalize()` 为前缀。
/// 3. `sub_root` 或目标不存在时只能做字符级检查；第 1 步已拒绝 `..`，join 不会逃出。
pub fn safe_join_under(sub_root: &Path, raw: &str) -> Option<PathBuf> {
  if raw.contains("..") || raw.starts_with('/') || raw.starts_with('\\') {
    tracing::warn!(input = %raw, "safe_join_under: path traversal pattern, rejected");
    return None;
  }
  let joined = sub_root.join(raw);
  let Ok(canonical_root) = sub_root.canonicalize() else {
    return Some(joined);
  };
  match joined.canonicalize() {
    Ok(canonical) if canonical.starts_with(&canonical_root) => Some(canonical),
    Ok(canonical) => {
      tracing::warn!(input = %raw, resolved = %canonical.display(),
        "safe_join_under: resolved outside root, rejected");
      None
    }
    Err(_) => Some(joined),
  }
}

/// 单个路径段形式的 slug：非空，只含 ASCII 字母数字、`-`、`_`。
/// 用户给的 slug 拼进文件路径前必须过这一关，`..`、`/`、`\\` 都进不来。
pub fn is_safe_slug(s: &str) -> bool {
  !s.is_empty() && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn content_dir_prefers_content_root() {
    assert_eq!(
      content_dir_in(Some("/srv/content".into()), "posts"),
      PathBuf::from("/srv/content/posts")
    );
    assert_eq!(content_dir_in(None, "courses"), get_asset_root().join("courses"));
    assert_eq!(
      content_dir_in(Some("".into()), "posts"),
      get_asset_root().join("posts"),
      "空值视为未设"
    );
  }

  #[test]
  fn safe_slug_accepts_only_single_plain_segments() {
    for ok in ["welcome", "rust-basics", "01-fundamentals", "a_b"] {
      assert!(is_safe_slug(ok), "{ok}");
    }
    for bad in ["", "..", ".", "a/b", "../docs", "a\\b", "rust-basics/..", "a.b", "a b", "é"] {
      assert!(!is_safe_slug(bad), "{bad}");
    }
  }

  /// Phase 8.2 path-traversal 防御：用户输入的 `ref_path` 不能逃出 sub_root。
  #[test]
  fn safe_join_under_rejects_dotdot_segments() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let root = tmp.path();
    std::fs::create_dir_all(root.join("posts/article")).expect("setup posts");

    // 合法子路径
    assert!(safe_join_under(&root.join("posts"), "article").is_some());

    // `..` 段 → 立即拒绝（字符级别）
    assert!(safe_join_under(&root.join("posts"), "../etc").is_none());
    assert!(safe_join_under(&root.join("posts"), "article/../../etc").is_none());
    assert!(safe_join_under(&root.join("posts"), "../../etc/passwd").is_none());

    // 绝对路径 → 拒绝
    assert!(safe_join_under(&root.join("posts"), "/etc/passwd").is_none());

    // 不存在的子路径仍允许（lexical 检查通过），由上层 read_index_title 兜底
    assert!(safe_join_under(&root.join("posts"), "no-such-article").is_some());
  }

  #[test]
  fn returns_a_path_buf() {
    // 不能假设具体内容（依赖运行目录），只验证返回 PathBuf 即可
    let p = get_asset_root();
    // 一定是 "assets" 或 "../../assets" 之一
    let s = p.to_string_lossy();
    assert!(s == "assets" || s == "../../assets");
  }

  /// 基本命中 / 未命中：同 fingerprint 复用，变了才重建。
  #[test]
  fn dir_listing_cache_basic() {
    let cache: DirListingCache<u32> = DirListingCache::new();
    let fp1 = vec![(PathBuf::from("a"), SystemTime::UNIX_EPOCH)];
    let v1 = cache.get_or_rebuild(fp1.clone(), || 7);
    assert_eq!(*v1, 7);
    // 同 fingerprint：builder 不应再调；返回旧 Arc
    let v2 =
      cache.get_or_rebuild(fp1.clone(), || panic!("builder should not be invoked on cache hit"));
    assert_eq!(*v2, 7);
    assert!(Arc::ptr_eq(&v1, &v2));

    // fingerprint 变 → 重建
    let fp2 = vec![(PathBuf::from("a"), SystemTime::now())];
    let v3 = cache.get_or_rebuild(fp2, || 42);
    assert_eq!(*v3, 42);
  }

  #[test]
  fn dir_listing_cache_invalidate_forces_rebuild() {
    let cache: DirListingCache<u32> = DirListingCache::new();
    let fp = vec![(PathBuf::from("a"), SystemTime::UNIX_EPOCH)];
    let _ = cache.get_or_rebuild(fp.clone(), || 1);
    cache.invalidate();
    let v = cache.get_or_rebuild(fp, || 2);
    assert_eq!(*v, 2, "invalidate 后应当重建");
  }

  /// fingerprint_for_dir 静默跳过权限错 / 不存在路径；空目录返回空。
  #[test]
  fn fingerprint_for_missing_dir_is_empty() {
    let fp = fingerprint_for_dir("/tmp/__no_such_dir_for_app_core_tests__", |_| true);
    assert!(fp.is_empty());
  }

  #[test]
  fn fingerprint_collects_matched_files() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let root = tmp.path();
    std::fs::create_dir_all(root.join("a/x")).unwrap();
    std::fs::write(root.join("a/x/index.md"), "hello").unwrap();
    std::fs::write(root.join("a/x/extra.txt"), "ignore me").unwrap();
    std::fs::write(root.join("a/x/index.mdx"), "world").unwrap();

    let fp = fingerprint_for_dir(root, |p| {
      p.file_name().is_some_and(|n| n == "index.md" || n == "index.mdx")
    });
    // 只收集 2 个 index 文件，按路径排序
    assert_eq!(fp.len(), 2);
    let names: Vec<_> =
      fp.iter().map(|(p, _)| p.file_name().unwrap().to_string_lossy().into_owned()).collect();
    assert!(names.contains(&"index.md".to_string()));
    assert!(names.contains(&"index.mdx".to_string()));
  }

  #[test]
  fn content_root_auto_detects_non_empty_docker_mount() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let empty_mount = tmp.path().join("empty-mount");
    std::fs::create_dir_all(&empty_mount).unwrap();

    // 空挂载点不触发覆盖，避免未挂卷的占位目录遮蔽内置 assets
    assert_eq!(content_root_from(None, &empty_mount), None);
    assert_eq!(content_root_from(Some("".into()), &empty_mount), None);

    // 放入任意内容文件后自动识别为生效内容根
    std::fs::write(empty_mount.join("site.json"), "{}").unwrap();
    assert_eq!(content_root_from(None, &empty_mount), Some(empty_mount.clone()));

    // 显式 CONTENT_DIR 优先级最高
    let explicit = tmp.path().join("explicit");
    assert_eq!(content_root_from(Some(explicit.clone().into()), &empty_mount), Some(explicit));
  }

  #[test]
  fn site_json_path_prefers_content_root_with_asset_fallback() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let assets = tmp.path().join("assets");
    let content = tmp.path().join("content");
    std::fs::create_dir_all(&assets).unwrap();
    std::fs::create_dir_all(&content).unwrap();
    std::fs::write(assets.join("site.json"), r#"{"site_name":"Default"}"#).unwrap();

    // content 目录尚未放 site.json → 回退 assets/site.json
    assert_eq!(site_json_path_in(&assets, Some(&content)), assets.join("site.json"));

    // content 目录提供 site.json → 优先使用 content/site.json
    std::fs::write(content.join("site.json"), r#"{"site_name":"Custom"}"#).unwrap();
    assert_eq!(site_json_path_in(&assets, Some(&content)), content.join("site.json"));
  }
}
