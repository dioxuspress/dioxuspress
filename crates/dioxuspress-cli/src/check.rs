use std::collections::{BTreeSet, HashMap};
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use pulldown_cmark::{Event, Options, Parser, Tag};

const EMBEDDED_SITE_SCHEMA: &str = include_str!("../../../assets/schemas/site.schema.json");

const BUILTIN_THEMES: &[&str] = &[
  "ocean",
  "emerald",
  "nord",
  "corporate",
  "winter",
  "catppuccin",
  "dracula",
  "sunset",
  "coffee",
  "silk",
];

const BUILTIN_MODULES: &[&str] = &[
  "blog", "podcast", "cases", "docs", "embedded", "ai", "web3", "wasm", "cli", "about", "forum",
  "course", "papers",
];

const BUILTIN_AUTH_PROVIDERS: &[&str] = &["github", "google", "discord", "twitter"];

const BUILTIN_MDX_COMPONENTS: &[&str] = &[
  "YouTube",
  "Bilibili",
  "Yellow",
  "Green",
  "Blue",
  "Pink",
  "Purple",
  "Underline",
  "Strikethrough",
  "AudioPlayer",
  "PodcastCard",
  "ThemeShowcase",
  "ModelArch",
];

/// 框架镜像自带的兜底默认图：外部内容仓库在 `site.json` 未覆盖或显式引用这两个默认路径时不误报缺失。
const BUILTIN_FALLBACK_IMAGES: &[&str] = &["/images/favicon.ico", "/images/og-default.jpg"];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diagnostic {
  pub file: PathBuf,
  pub line: usize,
  pub message: String,
}

impl fmt::Display for Diagnostic {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "{}:{}: {}", self.file.display(), self.line, self.message)
  }
}

#[derive(Debug)]
pub struct CheckSummary {
  pub root: PathBuf,
  pub files_checked: usize,
  pub internal_links_checked: usize,
  pub diagnostics: Vec<Diagnostic>,
}

impl CheckSummary {
  pub fn is_ok(&self) -> bool {
    self.diagnostics.is_empty()
  }
}

struct SchemaRules {
  root_keys: BTreeSet<String>,
  defs: HashMap<String, BTreeSet<String>>,
}

impl SchemaRules {
  fn load() -> Self {
    let mut root_keys = BTreeSet::new();
    let mut defs = HashMap::new();
    if let Ok(val) = serde_json::from_str::<serde_json::Value>(EMBEDDED_SITE_SCHEMA) {
      if let Some(props) = val.get("properties").and_then(|v| v.as_object()) {
        for k in props.keys() {
          root_keys.insert(k.clone());
        }
      }
      for def_container in ["$defs", "definitions"] {
        if let Some(map) = val.get(def_container).and_then(|v| v.as_object()) {
          for (name, def_val) in map {
            if let Some(props) = def_val.get("properties").and_then(|v| v.as_object()) {
              defs.insert(name.clone(), props.keys().cloned().collect());
            }
          }
        }
      }
    }
    Self { root_keys, defs }
  }
}

/// 解析待校验的内容根目录：
/// - 若 `input` 下存在 `site.json` / `docs` / `posts` / `courses` / `cases`，直接取 `input`；
/// - 若在引擎源码根运行（`input/Cargo.toml` 且 `input/assets/site.json` 存在），自动取 `input/assets`。
pub fn resolve_target_root(input: &Path) -> Result<PathBuf, String> {
  if !input.exists() {
    return Err(format!("directory does not exist: {}", input.display()));
  }
  let assets = input.join("assets");
  if !input.join("site.json").is_file()
    && input.join("Cargo.toml").is_file()
    && assets.join("site.json").is_file()
  {
    return Ok(assets);
  }
  let has_direct_content = ["site.json", "docs", "posts", "courses", "cases", "models"]
    .iter()
    .any(|name| input.join(name).exists());
  if has_direct_content {
    return Ok(input.to_path_buf());
  }
  Err(format!(
    "no DioxusPress content found in {} (expected site.json, docs/, or posts/)",
    input.display()
  ))
}

pub fn check_site(input: &Path) -> Result<CheckSummary, String> {
  let root = resolve_target_root(input)?;
  let mut diagnostics = Vec::new();
  let mut files_checked = 0usize;
  let mut internal_links_checked = 0usize;

  let site_json = root.join("site.json");
  if site_json.is_file() {
    files_checked += 1;
    check_site_json(&site_json, &root, &mut diagnostics);
  }

  let docs_dir = root.join("docs");
  if docs_dir.is_dir() {
    let mut doc_files = Vec::new();
    collect_markdown_files(&docs_dir, &mut doc_files);
    doc_files.sort();
    for file in doc_files {
      files_checked += 1;
      check_markdown_file(
        &file,
        &root,
        ContentKind::Doc,
        &mut internal_links_checked,
        &mut diagnostics,
      );
    }
  }

  let posts_dir = root.join("posts");
  if posts_dir.is_dir() {
    let authors_yml = posts_dir.join("authors.yml");
    if authors_yml.is_file() {
      files_checked += 1;
      check_yaml_file(&authors_yml, &mut diagnostics);
    }
    if let Ok(entries) = fs::read_dir(&posts_dir) {
      let mut dirs: Vec<PathBuf> =
        entries.flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect();
      dirs.sort();
      for dir in dirs {
        let slug = dir.file_name().and_then(|s| s.to_str()).unwrap_or_default();
        if !is_safe_slug(slug) {
          diagnostics.push(Diagnostic {
            file: dir.clone(),
            line: 1,
            message: format!(
              "blog post directory '{slug}' is not a valid URL slug (only ASCII [A-Za-z0-9_-] allowed)"
            ),
          });
        }
        for name in ["index.mdx", "index.md"] {
          let candidate = dir.join(name);
          if candidate.is_file() {
            files_checked += 1;
            check_markdown_file(
              &candidate,
              &root,
              ContentKind::BlogPost,
              &mut internal_links_checked,
              &mut diagnostics,
            );
            break;
          }
        }
      }
    }
  }

  let courses_dir = root.join("courses");
  if courses_dir.is_dir() {
    let mut yaml_files = Vec::new();
    collect_named_files(&courses_dir, &["course.yaml", "_chapter.yaml"], &mut yaml_files);
    yaml_files.sort();
    for yf in yaml_files {
      files_checked += 1;
      check_yaml_file(&yf, &mut diagnostics);
    }

    let mut lesson_files = Vec::new();
    collect_markdown_files(&courses_dir, &mut lesson_files);
    lesson_files.sort();
    for file in lesson_files {
      files_checked += 1;
      check_markdown_file(
        &file,
        &root,
        ContentKind::CourseLesson,
        &mut internal_links_checked,
        &mut diagnostics,
      );
    }
  }

  let cases_dir = root.join("cases");
  if cases_dir.is_dir() {
    let mut yaml_files = Vec::new();
    collect_named_files(&cases_dir, &["case.yaml"], &mut yaml_files);
    yaml_files.sort();
    for yf in yaml_files {
      files_checked += 1;
      check_yaml_file(&yf, &mut diagnostics);
    }

    let mut readme_files = Vec::new();
    collect_named_files(&cases_dir, &["README.md"], &mut readme_files);
    readme_files.sort();
    for rf in readme_files {
      files_checked += 1;
      check_markdown_file(
        &rf,
        &root,
        ContentKind::FreePage,
        &mut internal_links_checked,
        &mut diagnostics,
      );
    }
  }

  let models_dir = root.join("models");
  if models_dir.is_dir() {
    if let Ok(entries) = fs::read_dir(&models_dir) {
      let mut dirs: Vec<PathBuf> =
        entries.flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect();
      dirs.sort();
      for dir in dirs {
        let slug = dir.file_name().and_then(|s| s.to_str()).unwrap_or_default();
        if slug.starts_with('.') || slug.starts_with('_') {
          continue;
        }
        if !is_safe_slug(slug) {
          diagnostics.push(Diagnostic {
            file: dir.clone(),
            line: 1,
            message: format!(
              "model directory '{slug}' is not a valid URL slug (only ASCII [A-Za-z0-9_-] allowed)"
            ),
          });
        }
        let yaml_path = if dir.join("model.yaml").is_file() {
          Some(dir.join("model.yaml"))
        } else if dir.join("model.yml").is_file() {
          Some(dir.join("model.yml"))
        } else {
          None
        };
        match yaml_path {
          Some(yf) => {
            files_checked += 1;
            check_yaml_file(&yf, &mut diagnostics);
          }
          None => diagnostics.push(Diagnostic {
            file: dir.clone(),
            line: 1,
            message: "curated model directory is missing model.yaml".to_string(),
          }),
        }
        let cfg_path = dir.join("config.json");
        if cfg_path.is_file() {
          files_checked += 1;
          match fs::read_to_string(&cfg_path)
            .ok()
            .and_then(|raw| serde_json::from_str::<serde_json::Value>(&raw).ok())
          {
            Some(v) if v.is_object() => {}
            _ => diagnostics.push(Diagnostic {
              file: cfg_path,
              line: 1,
              message: "invalid model config.json (must be a valid JSON object)".to_string(),
            }),
          }
        } else {
          diagnostics.push(Diagnostic {
            file: dir.clone(),
            line: 1,
            message: "curated model directory is missing config.json".to_string(),
          });
        }
        let readme = dir.join("README.md");
        if readme.is_file() {
          files_checked += 1;
          check_markdown_file(
            &readme,
            &root,
            ContentKind::FreePage,
            &mut internal_links_checked,
            &mut diagnostics,
          );
        }
      }
    }
  }

  let podcasts_dir = root.join("podcasts");
  if podcasts_dir.is_dir() {
    let mut yaml_files = Vec::new();
    collect_named_files(&podcasts_dir, &["episode.yaml"], &mut yaml_files);
    yaml_files.sort();
    for yf in yaml_files {
      files_checked += 1;
      check_yaml_file(&yf, &mut diagnostics);
    }
  }

  let pages_dir = root.join("pages");
  if pages_dir.is_dir() {
    if let Ok(entries) = fs::read_dir(&pages_dir) {
      let mut md_files: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
          p.is_file() && matches!(p.extension().and_then(|s| s.to_str()), Some("md" | "mdx"))
        })
        .collect();
      md_files.sort();
      for file in md_files {
        files_checked += 1;
        check_markdown_file(
          &file,
          &root,
          ContentKind::FreePage,
          &mut internal_links_checked,
          &mut diagnostics,
        );
      }
    }
  }

  Ok(CheckSummary { root, files_checked, internal_links_checked, diagnostics })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ContentKind {
  Doc,
  BlogPost,
  CourseLesson,
  FreePage,
}

fn check_site_json(path: &Path, root: &Path, diagnostics: &mut Vec<Diagnostic>) {
  let raw = match fs::read_to_string(path) {
    Ok(s) => s,
    Err(e) => {
      diagnostics.push(Diagnostic {
        file: path.to_path_buf(),
        line: 1,
        message: format!("failed to read site.json: {e}"),
      });
      return;
    }
  };

  let val: serde_json::Value = match serde_json::from_str(&raw) {
    Ok(v) => v,
    Err(e) => {
      diagnostics.push(Diagnostic {
        file: path.to_path_buf(),
        line: e.line().max(1),
        message: format!("invalid JSON syntax (column {}): {}", e.column(), e),
      });
      return;
    }
  };

  let Some(obj) = val.as_object() else {
    diagnostics.push(Diagnostic {
      file: path.to_path_buf(),
      line: 1,
      message: "site.json root value must be a JSON object".to_string(),
    });
    return;
  };

  let rules = SchemaRules::load();
  for key in obj.keys() {
    if matches!(key.as_str(), "$schema" | "paths" | "plugins_lock") {
      continue;
    }
    if !rules.root_keys.contains(key) {
      let hint = suggest_key(key, &rules.root_keys);
      diagnostics.push(Diagnostic {
        file: path.to_path_buf(),
        line: find_json_key_line(&raw, key),
        message: format!("unknown field '{key}' in site.json{hint}"),
      });
    }
  }

  let nested_defs = [
    ("announcement", "AnnouncementSettings"),
    ("hero", "HeroSettings"),
    ("footer", "FooterSettings"),
    ("annotations", "AnnotationsSettings"),
    ("auth", "AuthSettings"),
    ("moderation", "ModerationSettings"),
  ];
  for (section, def_name) in nested_defs {
    if let (Some(sub_obj), Some(allowed)) =
      (obj.get(section).and_then(|v| v.as_object()), rules.defs.get(def_name))
    {
      for key in sub_obj.keys() {
        if !allowed.contains(key) {
          let hint = suggest_key(key, allowed);
          diagnostics.push(Diagnostic {
            file: path.to_path_buf(),
            line: find_json_key_line_in_section(&raw, section, key),
            message: format!("unknown field '{section}.{key}' in site.json{hint}"),
          });
        }
      }
    }
  }

  if let (Some(arr), Some(allowed)) =
    (obj.get("features").and_then(|v| v.as_array()), rules.defs.get("FeatureItem"))
  {
    for (idx, item) in arr.iter().enumerate() {
      if let Some(item_obj) = item.as_object() {
        for key in item_obj.keys() {
          if !allowed.contains(key) {
            let hint = suggest_key(key, allowed);
            diagnostics.push(Diagnostic {
              file: path.to_path_buf(),
              line: find_json_key_line_in_section(&raw, "features", key),
              message: format!("unknown field 'features[{idx}].{key}' in site.json{hint}"),
            });
          }
        }
      }
    }
  }

  if let Some(layout) = obj.get("active_layout").and_then(|v| v.as_str()) {
    let trimmed = layout.trim();
    if !trimmed.is_empty() && !matches!(trimmed, "classic" | "docs" | "minimal") {
      diagnostics.push(Diagnostic {
        file: path.to_path_buf(),
        line: find_json_key_line(&raw, "active_layout"),
        message: format!(
          "invalid active_layout '{trimmed}' (expected \"classic\", \"docs\", or \"minimal\")"
        ),
      });
    }
  }

  if let Some(lang) = obj.get("default_language").and_then(|v| v.as_str()) {
    let trimmed = lang.trim();
    if !trimmed.is_empty() && !matches!(trimmed.to_ascii_lowercase().as_str(), "zh" | "en") {
      diagnostics.push(Diagnostic {
        file: path.to_path_buf(),
        line: find_json_key_line(&raw, "default_language"),
        message: format!("invalid default_language '{trimmed}' (expected \"zh\" or \"en\")"),
      });
    }
  }

  if let Some(theme) = obj.get("theme").and_then(|v| v.as_str()) {
    let trimmed = theme.trim();
    if !trimmed.is_empty() && !BUILTIN_THEMES.contains(&trimmed) {
      diagnostics.push(Diagnostic {
        file: path.to_path_buf(),
        line: find_json_key_line(&raw, "theme"),
        message: format!(
          "unknown theme '{trimmed}' (expected one of: {})",
          BUILTIN_THEMES.join(", ")
        ),
      });
    }
  }

  if let Some(arr) = obj.get("available_themes").and_then(|v| v.as_array()) {
    for item in arr {
      if let Some(id) = item.as_str().map(str::trim) {
        if !id.is_empty() && !BUILTIN_THEMES.contains(&id) {
          diagnostics.push(Diagnostic {
            file: path.to_path_buf(),
            line: find_json_key_line(&raw, "available_themes"),
            message: format!(
              "unknown theme '{id}' in available_themes (expected one of: {})",
              BUILTIN_THEMES.join(", ")
            ),
          });
        }
      }
    }
  }

  if let Some(mods) = obj.get("modules").and_then(|v| v.as_object()) {
    for mod_id in mods.keys() {
      if !BUILTIN_MODULES.contains(&mod_id.as_str()) {
        diagnostics.push(Diagnostic {
          file: path.to_path_buf(),
          line: find_json_key_line(&raw, mod_id),
          message: format!(
            "unknown module id '{mod_id}' in modules (expected one of: {})",
            BUILTIN_MODULES.join(", ")
          ),
        });
      }
    }
  }

  if let Some(providers) =
    obj.get("auth").and_then(|v| v.get("providers")).and_then(|v| v.as_array())
  {
    for p in providers {
      if let Some(id) = p.as_str().map(str::trim) {
        if !BUILTIN_AUTH_PROVIDERS.contains(&id) {
          diagnostics.push(Diagnostic {
            file: path.to_path_buf(),
            line: find_json_key_line(&raw, "providers"),
            message: format!(
              "unknown OAuth provider '{id}' in auth.providers (expected one of: {})",
              BUILTIN_AUTH_PROVIDERS.join(", ")
            ),
          });
        }
      }
    }
  }

  let image_fields = [
    "favicon",
    "favicon_16",
    "favicon_32",
    "apple_touch_icon",
    "manifest",
    "logo",
    "logo_dark",
    "logo_mobile",
    "og_image",
  ];
  for field in image_fields {
    if let Some(url) = obj.get(field).and_then(|v| v.as_str()).map(str::trim) {
      if let Some(rel) = url.strip_prefix("/images/") {
        let rel_clean = rel.split(['?', '#']).next().unwrap_or(rel);
        if !rel_clean.is_empty()
          && !root.join("images").join(rel_clean).is_file()
          && !BUILTIN_FALLBACK_IMAGES.contains(&url)
        {
          diagnostics.push(Diagnostic {
            file: path.to_path_buf(),
            line: find_json_key_line(&raw, field),
            message: format!(
              "field '{field}' references '{url}', which was not found at {}",
              root.join("images").join(rel_clean).display()
            ),
          });
        }
      }
    }
  }
}

fn check_yaml_file(path: &Path, diagnostics: &mut Vec<Diagnostic>) {
  let raw = match fs::read_to_string(path) {
    Ok(s) => s,
    Err(e) => {
      diagnostics.push(Diagnostic {
        file: path.to_path_buf(),
        line: 1,
        message: format!("failed to read YAML file: {e}"),
      });
      return;
    }
  };
  if let Err(e) = serde_yaml::from_str::<serde_yaml::Value>(&raw) {
    let line = e.location().map(|l| l.line()).unwrap_or(1).max(1);
    diagnostics.push(Diagnostic {
      file: path.to_path_buf(),
      line,
      message: format!("invalid YAML syntax: {e}"),
    });
  }
}

fn check_markdown_file(
  file: &Path,
  root: &Path,
  kind: ContentKind,
  internal_links_checked: &mut usize,
  diagnostics: &mut Vec<Diagnostic>,
) {
  let raw = match fs::read_to_string(file) {
    Ok(s) => s,
    Err(e) => {
      diagnostics.push(Diagnostic {
        file: file.to_path_buf(),
        line: 1,
        message: format!("failed to read Markdown file: {e}"),
      });
      return;
    }
  };

  let line_starts = compute_line_starts(&raw);
  let (frontmatter_opt, body, body_line_offset) = match split_frontmatter(&raw) {
    FrontmatterSplit::None(b) => (None, b, 1usize),
    FrontmatterSplit::Unclosed => {
      diagnostics.push(Diagnostic {
        file: file.to_path_buf(),
        line: 1,
        message: "unclosed YAML frontmatter (missing closing '---' delimiter)".to_string(),
      });
      return;
    }
    FrontmatterSplit::Found { yaml, body, body_start_line } => (Some(yaml), body, body_start_line),
  };

  let mut fm_map = None;
  if let Some(yaml) = frontmatter_opt {
    match serde_yaml::from_str::<serde_yaml::Value>(yaml) {
      Ok(val) => {
        fm_map = val.as_mapping().cloned();
      }
      Err(e) => {
        // 首行是 `---`（第 1 行），yaml 从第 2 行起算
        let yaml_line = e.location().map(|l| l.line() + 1).unwrap_or(2);
        diagnostics.push(Diagnostic {
          file: file.to_path_buf(),
          line: yaml_line,
          message: format!("invalid YAML frontmatter: {e}"),
        });
      }
    }
  }

  let is_draft = fm_map
    .as_ref()
    .and_then(|m| m.get(serde_yaml::Value::String("draft".to_string())))
    .and_then(|v| v.as_bool())
    .unwrap_or(false);

  let fm_title = fm_scalar_non_empty(fm_map.as_ref(), "title");
  let fm_date = fm_scalar_non_empty(fm_map.as_ref(), "date");

  match kind {
    ContentKind::BlogPost if !is_draft => {
      if !fm_title {
        diagnostics.push(Diagnostic {
          file: file.to_path_buf(),
          line: 1,
          message: "blog post frontmatter is missing non-empty 'title'".to_string(),
        });
      }
      if !fm_date {
        diagnostics.push(Diagnostic {
          file: file.to_path_buf(),
          line: 1,
          message: "blog post frontmatter is missing non-empty 'date'".to_string(),
        });
      }
    }
    ContentKind::Doc if !fm_title && !has_h1_heading(body) => {
      diagnostics.push(Diagnostic {
        file: file.to_path_buf(),
        line: 1,
        message: "doc page has neither frontmatter 'title' nor a '# Heading' in body".to_string(),
      });
    }
    _ => {}
  }

  // 逐行扫描围栏代码块与 ::: 提示框 / code-tabs 平衡性，同时生成保留换行符的散文掩码供链接与 MDX 检查
  check_fences_and_admonitions(file, body, body_line_offset, diagnostics);

  let body_byte_offset = raw.len().saturating_sub(body.len());
  check_links_and_mdx_tags(
    file,
    root,
    &raw,
    body,
    body_byte_offset,
    &line_starts,
    internal_links_checked,
    diagnostics,
  );
}

enum FrontmatterSplit<'a> {
  None(&'a str),
  Unclosed,
  Found { yaml: &'a str, body: &'a str, body_start_line: usize },
}

fn split_frontmatter(raw: &str) -> FrontmatterSplit<'_> {
  let stripped = raw.strip_prefix('\u{feff}').unwrap_or(raw);
  let after_open = if let Some(r) = stripped.strip_prefix("---\n") {
    r
  } else if let Some(r) = stripped.strip_prefix("---\r\n") {
    r
  } else {
    return FrontmatterSplit::None(raw);
  };

  let mut cursor = 0usize;
  let mut line_no = 2usize;
  for line in after_open.split_inclusive('\n') {
    let trimmed = line.trim_end_matches(['\r', '\n']);
    if trimmed == "---" {
      let yaml = &after_open[..cursor];
      let body = &after_open[cursor + line.len()..];
      return FrontmatterSplit::Found { yaml, body, body_start_line: line_no + 1 };
    }
    cursor += line.len();
    line_no += 1;
  }
  if after_open.trim_end_matches(['\r', '\n']) == "---" {
    return FrontmatterSplit::Found { yaml: "", body: "", body_start_line: line_no + 1 };
  }
  FrontmatterSplit::Unclosed
}

fn fm_scalar_non_empty(map: Option<&serde_yaml::Mapping>, key: &str) -> bool {
  let Some(m) = map else { return false };
  let Some(v) = m.get(serde_yaml::Value::String(key.to_string())) else {
    return false;
  };
  match v {
    serde_yaml::Value::String(s) => !s.trim().is_empty(),
    serde_yaml::Value::Number(_) => true,
    _ => false,
  }
}

fn has_h1_heading(body: &str) -> bool {
  let mut fence: Option<(char, usize)> = None;
  for line in body.lines() {
    let t = line.trim();
    if let Some((ch, len)) = fence {
      if is_closing_fence(t, ch, len) {
        fence = None;
      }
      continue;
    }
    if let Some(open) = parse_opening_fence(t) {
      fence = Some(open);
      continue;
    }
    if let Some(rest) = t.strip_prefix("# ") {
      if !rest.trim().is_empty() {
        return true;
      }
    }
  }
  false
}

fn parse_opening_fence(trimmed: &str) -> Option<(char, usize)> {
  let ch = trimmed.chars().next().filter(|c| *c == '`' || *c == '~')?;
  let len = trimmed.chars().take_while(|c| *c == ch).count();
  if len < 3 {
    return None;
  }
  let rest = &trimmed[len * ch.len_utf8()..];
  // CommonMark 规范：反引号围栏的 info string 不能再包含反引号（否则是同行行内代码，如 ```foo```）
  if ch == '`' && rest.contains('`') {
    return None;
  }
  Some((ch, len))
}

fn is_closing_fence(trimmed: &str, ch: char, min_len: usize) -> bool {
  let count = trimmed.chars().take_while(|c| *c == ch).count();
  count >= min_len && trimmed.chars().all(|c| c == ch)
}

fn split_admonition_directive(rest: &str) -> (&str, &str) {
  let word_len = rest.bytes().take_while(|b| b.is_ascii_alphabetic() || *b == b'-').count();
  (&rest[..word_len], &rest[word_len..])
}

fn check_fences_and_admonitions(
  file: &Path,
  body: &str,
  body_start_line: usize,
  diagnostics: &mut Vec<Diagnostic>,
) {
  let mut open_fence: Option<(char, usize, usize)> = None;
  let mut open_admonition: Option<(String, usize)> = None;
  let mut open_code_tabs: Option<(String, usize)> = None;

  for (idx, line) in body.lines().enumerate() {
    let line_no = body_start_line + idx;
    let trimmed = line.trim();

    if let Some((ch, len, _)) = open_fence {
      if is_closing_fence(trimmed, ch, len) {
        open_fence = None;
      }
      continue;
    }

    if let Some((ch, len)) = parse_opening_fence(trimmed) {
      open_fence = Some((ch, len, line_no));
      continue;
    }

    if let Some(rest) = trimmed.strip_prefix(":::") {
      if trimmed == ":::" {
        if open_code_tabs.is_some() {
          open_code_tabs = None;
        } else if open_admonition.is_some() {
          open_admonition = None;
        } else {
          diagnostics.push(Diagnostic {
            file: file.to_path_buf(),
            line: line_no,
            message: "unmatched closing ':::' admonition delimiter".to_string(),
          });
        }
        continue;
      }

      let (raw_word, tail) = split_admonition_directive(rest);
      let word = raw_word.to_ascii_lowercase();
      if word.is_empty() {
        continue;
      }

      if let Some((prev_tabs, prev_line)) = &open_code_tabs {
        diagnostics.push(Diagnostic {
          file: file.to_path_buf(),
          line: line_no,
          message: format!(
            "cannot open ':::{word}' inside unclosed ':::{prev_tabs}' (opened at line {prev_line})"
          ),
        });
        continue;
      }

      if matches!(word.as_str(), "code-tabs" | "codetabs" | "tabs") {
        // 允许在顶层或外层提示框（如 `:::important`）内部开启 `:::code-tabs`
        open_code_tabs = Some((word, line_no));
      } else if let Some((prev_kind, prev_line)) = &open_admonition {
        diagnostics.push(Diagnostic {
          file: file.to_path_buf(),
          line: line_no,
          message: format!(
            "cannot open ':::{word}' inside unclosed ':::{prev_kind}' (opened at line {prev_line})"
          ),
        });
      } else if tail.ends_with(":::") {
        // 单行提示框（如 `:::note命令名称必须唯一。:::`）当场闭合
      } else {
        open_admonition = Some((word, line_no));
      }
    }
  }

  if let Some((ch, len, open_line)) = open_fence {
    let marker: String = std::iter::repeat_n(ch, len).collect();
    diagnostics.push(Diagnostic {
      file: file.to_path_buf(),
      line: open_line,
      message: format!("unclosed code fence '{marker}'"),
    });
  }

  if let Some((tabs_kind, open_line)) = open_code_tabs {
    diagnostics.push(Diagnostic {
      file: file.to_path_buf(),
      line: open_line,
      message: format!("unclosed ':::{tabs_kind}' block (missing closing ':::')"),
    });
  }

  if let Some((kind, open_line)) = open_admonition {
    diagnostics.push(Diagnostic {
      file: file.to_path_buf(),
      line: open_line,
      message: format!("unclosed ':::{kind}' admonition block (missing closing ':::')"),
    });
  }
}

#[allow(clippy::too_many_arguments)]
fn check_links_and_mdx_tags(
  file: &Path,
  root: &Path,
  full_raw: &str,
  body: &str,
  body_byte_offset: usize,
  line_starts: &[usize],
  internal_links_checked: &mut usize,
  diagnostics: &mut Vec<Diagnostic>,
) {
  let parser = Parser::new_ext(body, Options::all());
  for (event, range) in parser.into_offset_iter() {
    let abs_offset = body_byte_offset + range.start;
    let line_no = byte_offset_to_line(line_starts, abs_offset);

    match event {
      Event::Start(Tag::Link { dest_url, .. } | Tag::Image { dest_url, .. }) => {
        if let Some(problem) = verify_target_url(&dest_url, file, root, internal_links_checked) {
          diagnostics.push(Diagnostic {
            file: file.to_path_buf(),
            line: line_no,
            message: problem,
          });
        }
      }
      Event::Html(html) | Event::InlineHtml(html) => {
        if let Some(problem) = check_mdx_component_snippet(&html, full_raw, abs_offset) {
          diagnostics.push(Diagnostic {
            file: file.to_path_buf(),
            line: line_no,
            message: problem,
          });
        }
      }
      _ => {}
    }
  }
}

fn check_mdx_component_snippet(snippet: &str, full_raw: &str, abs_offset: usize) -> Option<String> {
  // 若 `<` 紧跟在标识符或 `::` 后（如正文中未加反引号的 `Vec<String>`、`Option<T>`），属于泛型文本而非 MDX 组件
  if abs_offset > 0 {
    let prev = full_raw.as_bytes()[abs_offset - 1];
    if prev.is_ascii_alphanumeric() || prev == b'_' || prev == b':' {
      return None;
    }
  }

  let trimmed = snippet.trim();
  let after_lt = trimmed.strip_prefix('<')?;
  if after_lt.starts_with('/') || after_lt.starts_with('!') || after_lt.starts_with('?') {
    return None;
  }
  let end =
    after_lt.find(|c: char| c.is_whitespace() || c == '/' || c == '>').unwrap_or(after_lt.len());
  let name = &after_lt[..end];
  if name.is_empty()
    || !name.chars().next().is_some_and(|c| c.is_ascii_uppercase())
    || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
  {
    return None;
  }

  let rest = &after_lt[end..];
  let is_known_component = BUILTIN_MDX_COMPONENTS.contains(&name);
  let looks_like_jsx =
    is_known_component || rest.contains('=') || rest.trim_start().starts_with('/');
  if !looks_like_jsx {
    return None;
  }

  if !has_balanced_quotes(trimmed) {
    return Some(format!("MDX component <{name} ... /> has an unclosed attribute quote"));
  }
  if !trimmed.ends_with("/>") {
    return Some(format!(
      "MDX component <{name}> must be self-closing (expected '<{name} ... />')"
    ));
  }
  None
}

fn has_balanced_quotes(tag: &str) -> bool {
  let mut active_quote: Option<u8> = None;
  for &b in tag.as_bytes() {
    match active_quote {
      Some(q) if b == q => active_quote = None,
      None if b == b'"' || b == b'\'' => active_quote = Some(b),
      _ => {}
    }
  }
  active_quote.is_none()
}

fn verify_target_url(
  raw_url: &str,
  source_file: &Path,
  root: &Path,
  internal_links_checked: &mut usize,
) -> Option<String> {
  let trimmed = raw_url.trim();
  if trimmed.is_empty()
    || trimmed.starts_with('#')
    || trimmed.starts_with("http://")
    || trimmed.starts_with("https://")
    || trimmed.starts_with("mailto:")
    || trimmed.starts_with("tel:")
    || trimmed.starts_with("data:")
  {
    return None;
  }

  let without_frag = trimmed.split(['#', '?']).next().unwrap_or("").trim_end_matches('/');
  if without_frag.is_empty() {
    return None;
  }

  if without_frag == "/docs" {
    *internal_links_checked += 1;
    if root.join("docs").is_dir() {
      return None;
    }
    return Some(format!("broken link '{raw_url}': docs/ directory does not exist"));
  }

  if let Some(sub) = without_frag.strip_prefix("/docs/") {
    *internal_links_checked += 1;
    let target_base = root.join("docs").join(sub);
    if target_base.join("index.md").is_file()
      || target_base.join("index.mdx").is_file()
      || target_base.join("index.en.md").is_file()
      || target_base.join("index.en.mdx").is_file()
      || target_base.is_file()
    {
      return None;
    }
    return Some(format!(
      "broken doc link '{raw_url}': expected {}/index.md (or .mdx)",
      target_base.display()
    ));
  }

  if without_frag == "/blog" {
    *internal_links_checked += 1;
    return None;
  }

  if let Some(slug) = without_frag.strip_prefix("/blog/") {
    if slug.starts_with("page/") || slug.starts_with("tag/") {
      return None;
    }
    *internal_links_checked += 1;
    let post_dir = root.join("posts").join(slug);
    if post_dir.join("index.md").is_file() || post_dir.join("index.mdx").is_file() {
      return None;
    }
    return Some(format!(
      "broken blog link '{raw_url}': expected {}/index.md (or .mdx)",
      post_dir.display()
    ));
  }

  if let Some(rel) = without_frag.strip_prefix("/posts/") {
    *internal_links_checked += 1;
    let target = root.join("posts").join(rel);
    if target.is_file() {
      return None;
    }
    return Some(format!("missing post asset '{raw_url}': {} not found", target.display()));
  }

  if let Some(rel) = without_frag.strip_prefix("/docs-files/") {
    *internal_links_checked += 1;
    let target = root.join("docs").join(rel);
    if target.is_file() {
      return None;
    }
    return Some(format!("missing docs asset '{raw_url}': {} not found", target.display()));
  }

  if let Some(rel) = without_frag.strip_prefix("/images/") {
    *internal_links_checked += 1;
    let target = root.join("images").join(rel);
    if target.is_file() || BUILTIN_FALLBACK_IMAGES.contains(&without_frag) {
      return None;
    }
    return Some(format!("missing image '{raw_url}': {} not found", target.display()));
  }

  if let Some(sub) = without_frag.strip_prefix("/course/") {
    if root.join("courses").is_dir() {
      *internal_links_checked += 1;
      let mut parts = sub.split('/');
      let course_slug = parts.next().unwrap_or("");
      let chapter = parts.next();
      let lesson = parts.next();
      let course_dir = root.join("courses").join(course_slug);
      if !course_dir.join("course.yaml").is_file() {
        return Some(format!(
          "broken course link '{raw_url}': expected {}/course.yaml",
          course_dir.display()
        ));
      }
      if let (Some(ch), Some(ls)) = (chapter, lesson) {
        let lesson_dir = course_dir.join(ch).join(ls);
        if !lesson_dir.join("index.md").is_file() && !lesson_dir.join("index.mdx").is_file() {
          return Some(format!(
            "broken lesson link '{raw_url}': expected {}/index.md (or .mdx)",
            lesson_dir.display()
          ));
        }
      }
    }
    return None;
  }

  if let Some(slug) = without_frag.strip_prefix("/case/") {
    if root.join("cases").is_dir() {
      *internal_links_checked += 1;
      let case_dir = root.join("cases").join(slug);
      if !case_dir.join("case.yaml").is_file() {
        return Some(format!(
          "broken case link '{raw_url}': expected {}/case.yaml",
          case_dir.display()
        ));
      }
    }
    return None;
  }

  if let Some(slug) = without_frag.strip_prefix("/models/") {
    if root.join("models").is_dir() {
      *internal_links_checked += 1;
      let model_dir = root.join("models").join(slug);
      if !model_dir.join("model.yaml").is_file() && !model_dir.join("model.yml").is_file() {
        return Some(format!(
          "broken model link '{raw_url}': expected {}/model.yaml",
          model_dir.display()
        ));
      }
    }
    return None;
  }

  if without_frag.starts_with('/') {
    // 其它应用动态路由（如 /topics、/course、/case、/podcast、/about、/feed.xml、/schemas/*）不对应静态 Markdown
    return None;
  }

  // 相对路径：相对于当前 Markdown 所在目录解析
  *internal_links_checked += 1;
  let parent = source_file.parent().unwrap_or(root);
  let candidate = parent.join(without_frag);
  if candidate.is_file()
    || candidate.join("index.md").is_file()
    || candidate.join("index.mdx").is_file()
    || candidate.join("index.en.md").is_file()
    || candidate.join("index.en.mdx").is_file()
  {
    return None;
  }
  Some(format!("broken relative path '{raw_url}': {} not found", candidate.display()))
}

fn collect_markdown_files(dir: &Path, out: &mut Vec<PathBuf>) {
  let Ok(entries) = fs::read_dir(dir) else { return };
  for entry in entries.flatten() {
    let path = entry.path();
    if path.is_dir() {
      collect_markdown_files(&path, out);
    } else if matches!(
      path.file_name().and_then(|s| s.to_str()),
      Some("index.md" | "index.mdx" | "index.en.md" | "index.en.mdx")
    ) {
      out.push(path);
    }
  }
}

fn collect_named_files(dir: &Path, names: &[&str], out: &mut Vec<PathBuf>) {
  let Ok(entries) = fs::read_dir(dir) else { return };
  for entry in entries.flatten() {
    let path = entry.path();
    if path.is_dir() {
      collect_named_files(&path, names, out);
    } else if path.file_name().and_then(|s| s.to_str()).is_some_and(|n| names.contains(&n)) {
      out.push(path);
    }
  }
}

fn is_safe_slug(s: &str) -> bool {
  !s.is_empty() && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

fn compute_line_starts(text: &str) -> Vec<usize> {
  let mut starts = vec![0usize];
  for (idx, b) in text.bytes().enumerate() {
    if b == b'\n' {
      starts.push(idx + 1);
    }
  }
  starts
}

fn byte_offset_to_line(line_starts: &[usize], offset: usize) -> usize {
  match line_starts.binary_search(&offset) {
    Ok(idx) => idx + 1,
    Err(idx) => idx.max(1),
  }
}

fn find_json_key_line(raw: &str, key: &str) -> usize {
  find_json_key_line_after(raw, 0, key)
}

fn find_json_key_line_in_section(raw: &str, section: &str, key: &str) -> usize {
  let start_line = find_json_key_line(raw, section).saturating_sub(1);
  find_json_key_line_after(raw, start_line, key)
}

fn find_json_key_line_after(raw: &str, start_line_idx: usize, key: &str) -> usize {
  let needle = format!("\"{key}\"");
  for (idx, line) in raw.lines().enumerate().skip(start_line_idx) {
    if let Some(pos) = line.find(&needle) {
      let rest = line[pos + needle.len()..].trim_start();
      if rest.starts_with(':') {
        return idx + 1;
      }
    }
  }
  1
}

fn suggest_key(input: &str, candidates: &BTreeSet<String>) -> String {
  let mut best: Option<(&str, usize)> = None;
  for cand in candidates {
    let dist = edit_distance(input, cand);
    if dist <= 3 && best.is_none_or(|(_, d)| dist < d) {
      best = Some((cand.as_str(), dist));
    }
  }
  match best {
    Some((cand, _)) => format!(" (did you mean '{cand}'?)"),
    None => String::new(),
  }
}

fn edit_distance(a: &str, b: &str) -> usize {
  let a_chars: Vec<char> = a.chars().collect();
  let b_chars: Vec<char> = b.chars().collect();
  let m = a_chars.len();
  let n = b_chars.len();
  let mut prev: Vec<usize> = (0..=n).collect();
  let mut curr = vec![0usize; n + 1];
  for i in 1..=m {
    curr[0] = i;
    for j in 1..=n {
      let cost = usize::from(a_chars[i - 1] != b_chars[j - 1]);
      curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
    }
    std::mem::swap(&mut prev, &mut curr);
  }
  prev[n]
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::init::{init_site, Template};

  #[test]
  fn all_init_templates_pass_check_cleanly() {
    for tpl in [Template::Docs, Template::Blog, Template::Community, Template::Portal] {
      let tmp = tempfile::tempdir().unwrap();
      let dir = tmp.path().join(format!("sample-{}", tpl.id()));
      init_site(&dir, tpl, false).unwrap();
      let summary = check_site(&dir).unwrap();
      assert!(
        summary.is_ok(),
        "template '{}' produced check errors: {:?}",
        tpl.id(),
        summary.diagnostics
      );
      assert!(summary.files_checked >= 5);
      assert!(summary.internal_links_checked >= 2);
    }
  }

  #[test]
  fn official_assets_pass_check_with_zero_errors() {
    let repo_assets = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../assets");
    let summary = check_site(&repo_assets).unwrap();
    assert!(
      summary.is_ok(),
      "official assets/ failed dioxuspress check:\n{}",
      summary.diagnostics.iter().map(ToString::to_string).collect::<Vec<_>>().join("\n")
    );
  }

  #[test]
  fn official_starter_templates_pass_check_with_zero_errors() {
    let templates_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../templates");
    for tpl in Template::ALL {
      let dir = templates_root.join(tpl.id());
      let summary = check_site(&dir).unwrap();
      assert!(
        summary.is_ok(),
        "templates/{} failed dioxuspress check:\n{}",
        tpl.id(),
        summary.diagnostics.iter().map(ToString::to_string).collect::<Vec<_>>().join("\n")
      );
    }
  }

  #[test]
  fn catches_site_json_typo_with_exact_line_and_suggestion() {
    let tmp = tempfile::tempdir().unwrap();
    fs::write(
      tmp.path().join("site.json"),
      "{\n  \"site_name\": \"Demo\",\n  \"actve_layout\": \"docs\",\n  \"theme\": \"unknown-theme\"\n}\n",
    )
    .unwrap();

    let summary = check_site(tmp.path()).unwrap();
    assert_eq!(summary.diagnostics.len(), 2, "{:?}", summary.diagnostics);
    assert_eq!(summary.diagnostics[0].line, 3);
    assert!(summary.diagnostics[0].message.contains("actve_layout"));
    assert!(summary.diagnostics[0].message.contains("active_layout"));
    assert_eq!(summary.diagnostics[1].line, 4);
    assert!(summary.diagnostics[1].message.contains("unknown-theme"));
  }

  #[test]
  fn catches_broken_links_unclosed_fences_admonitions_and_mdx_tags() {
    let tmp = tempfile::tempdir().unwrap();
    let doc_dir = tmp.path().join("docs/guide/intro");
    fs::create_dir_all(&doc_dir).unwrap();
    let md_path = doc_dir.join("index.mdx");
    fs::write(
      &md_path,
      "---\ntitle: Intro\n---\n\n# Intro\n\nSee [missing](/docs/guide/does-not-exist) and `Vec<String>`.\n\n<Yellow text=\"unclosed\">\n\n:::tip\nForgot closing\n\n```rust\nfn x() {}\n",
    )
    .unwrap();

    let summary = check_site(tmp.path()).unwrap();
    let msgs: Vec<String> = summary.diagnostics.iter().map(ToString::to_string).collect();
    assert_eq!(summary.diagnostics.len(), 4, "got:\n{}", msgs.join("\n"));

    assert!(summary
      .diagnostics
      .iter()
      .any(|d| d.line == 7 && d.message.contains("does-not-exist")));
    assert!(summary.diagnostics.iter().any(|d| d.line == 9 && d.message.contains("self-closing")));
    assert!(summary.diagnostics.iter().any(|d| d.line == 11 && d.message.contains(":::tip")));
    assert!(summary
      .diagnostics
      .iter()
      .any(|d| d.line == 14 && d.message.contains("unclosed code fence")));
  }
}
