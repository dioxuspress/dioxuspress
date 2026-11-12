use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Template {
  Docs,
  Blog,
  Community,
  Portal,
}

pub struct EmbeddedFile {
  pub rel_path: &'static str,
  pub content: &'static str,
  pub replace_site_name: bool,
}

const DOCS_FILES: &[EmbeddedFile] = &[
  EmbeddedFile {
    rel_path: ".github/workflows/check.yml",
    content: include_str!("../../../templates/docs/.github/workflows/check.yml"),
    replace_site_name: false,
  },
  EmbeddedFile {
    rel_path: ".gitignore",
    content: include_str!("../../../templates/docs/.gitignore"),
    replace_site_name: false,
  },
  EmbeddedFile {
    rel_path: "README.md",
    content: include_str!("../../../templates/docs/README.md"),
    replace_site_name: true,
  },
  EmbeddedFile {
    rel_path: "docker-compose.yml",
    content: include_str!("../../../templates/docs/docker-compose.yml"),
    replace_site_name: false,
  },
  EmbeddedFile {
    rel_path: "docs/api/cli-reference/index.md",
    content: include_str!("../../../templates/docs/docs/api/cli-reference/index.md"),
    replace_site_name: true,
  },
  EmbeddedFile {
    rel_path: "docs/api/index.md",
    content: include_str!("../../../templates/docs/docs/api/index.md"),
    replace_site_name: true,
  },
  EmbeddedFile {
    rel_path: "docs/guide/advanced/deployment/index.md",
    content: include_str!("../../../templates/docs/docs/guide/advanced/deployment/index.md"),
    replace_site_name: true,
  },
  EmbeddedFile {
    rel_path: "docs/guide/advanced/index.md",
    content: include_str!("../../../templates/docs/docs/guide/advanced/index.md"),
    replace_site_name: true,
  },
  EmbeddedFile {
    rel_path: "docs/guide/configuration/index.md",
    content: include_str!("../../../templates/docs/docs/guide/configuration/index.md"),
    replace_site_name: true,
  },
  EmbeddedFile {
    rel_path: "docs/guide/getting-started/index.md",
    content: include_str!("../../../templates/docs/docs/guide/getting-started/index.md"),
    replace_site_name: true,
  },
  EmbeddedFile {
    rel_path: "docs/guide/index.md",
    content: include_str!("../../../templates/docs/docs/guide/index.md"),
    replace_site_name: true,
  },
  EmbeddedFile {
    rel_path: "docs/index.md",
    content: include_str!("../../../templates/docs/docs/index.md"),
    replace_site_name: true,
  },
  EmbeddedFile {
    rel_path: "images/.gitkeep",
    content: include_str!("../../../templates/docs/images/.gitkeep"),
    replace_site_name: false,
  },
  EmbeddedFile {
    rel_path: "posts/authors.yml",
    content: include_str!("../../../templates/docs/posts/authors.yml"),
    replace_site_name: false,
  },
  EmbeddedFile {
    rel_path: "posts/v0-1-release/index.md",
    content: include_str!("../../../templates/docs/posts/v0-1-release/index.md"),
    replace_site_name: true,
  },
  EmbeddedFile {
    rel_path: "site.json",
    content: include_str!("../../../templates/docs/site.json"),
    replace_site_name: true,
  },
];

const BLOG_FILES: &[EmbeddedFile] = &[
  EmbeddedFile {
    rel_path: ".gitignore",
    content: include_str!("../../../templates/blog/.gitignore"),
    replace_site_name: false,
  },
  EmbeddedFile {
    rel_path: "README.md",
    content: include_str!("../../../templates/blog/README.md"),
    replace_site_name: true,
  },
  EmbeddedFile {
    rel_path: "docker-compose.yml",
    content: include_str!("../../../templates/blog/docker-compose.yml"),
    replace_site_name: false,
  },
  EmbeddedFile {
    rel_path: "docs/index.md",
    content: include_str!("../../../templates/blog/docs/index.md"),
    replace_site_name: true,
  },
  EmbeddedFile {
    rel_path: "docs/notes/index.md",
    content: include_str!("../../../templates/blog/docs/notes/index.md"),
    replace_site_name: true,
  },
  EmbeddedFile {
    rel_path: "docs/notes/reading-list/index.md",
    content: include_str!("../../../templates/blog/docs/notes/reading-list/index.md"),
    replace_site_name: true,
  },
  EmbeddedFile {
    rel_path: "images/.gitkeep",
    content: include_str!("../../../templates/blog/images/.gitkeep"),
    replace_site_name: false,
  },
  EmbeddedFile {
    rel_path: "pages/about.md",
    content: include_str!("../../../templates/blog/pages/about.md"),
    replace_site_name: true,
  },
  EmbeddedFile {
    rel_path: "posts/authors.yml",
    content: include_str!("../../../templates/blog/posts/authors.yml"),
    replace_site_name: false,
  },
  EmbeddedFile {
    rel_path: "posts/systems-notes-01-zero-copy/index.md",
    content: include_str!("../../../templates/blog/posts/systems-notes-01-zero-copy/index.md"),
    replace_site_name: true,
  },
  EmbeddedFile {
    rel_path: "posts/systems-notes-02-latency-math/index.md",
    content: include_str!("../../../templates/blog/posts/systems-notes-02-latency-math/index.md"),
    replace_site_name: true,
  },
  EmbeddedFile {
    rel_path: "site.json",
    content: include_str!("../../../templates/blog/site.json"),
    replace_site_name: true,
  },
];

const COMMUNITY_FILES: &[EmbeddedFile] = &[
  EmbeddedFile {
    rel_path: ".env.example",
    content: include_str!("../../../templates/community/.env.example"),
    replace_site_name: false,
  },
  EmbeddedFile {
    rel_path: ".gitignore",
    content: include_str!("../../../templates/community/.gitignore"),
    replace_site_name: false,
  },
  EmbeddedFile {
    rel_path: "README.md",
    content: include_str!("../../../templates/community/README.md"),
    replace_site_name: true,
  },
  EmbeddedFile {
    rel_path: "docker-compose.yml",
    content: include_str!("../../../templates/community/docker-compose.yml"),
    replace_site_name: false,
  },
  EmbeddedFile {
    rel_path: "docs/handbook/index.md",
    content: include_str!("../../../templates/community/docs/handbook/index.md"),
    replace_site_name: true,
  },
  EmbeddedFile {
    rel_path: "docs/handbook/oauth-and-forum/index.md",
    content: include_str!("../../../templates/community/docs/handbook/oauth-and-forum/index.md"),
    replace_site_name: true,
  },
  EmbeddedFile {
    rel_path: "docs/index.md",
    content: include_str!("../../../templates/community/docs/index.md"),
    replace_site_name: true,
  },
  EmbeddedFile {
    rel_path: "images/.gitkeep",
    content: include_str!("../../../templates/community/images/.gitkeep"),
    replace_site_name: false,
  },
  EmbeddedFile {
    rel_path: "pages/about.md",
    content: include_str!("../../../templates/community/pages/about.md"),
    replace_site_name: true,
  },
  EmbeddedFile {
    rel_path: "posts/authors.yml",
    content: include_str!("../../../templates/community/posts/authors.yml"),
    replace_site_name: false,
  },
  EmbeddedFile {
    rel_path: "posts/welcome-to-community/index.md",
    content: include_str!("../../../templates/community/posts/welcome-to-community/index.md"),
    replace_site_name: true,
  },
  EmbeddedFile {
    rel_path: "site.json",
    content: include_str!("../../../templates/community/site.json"),
    replace_site_name: true,
  },
];

const PORTAL_FILES: &[EmbeddedFile] = &[
  EmbeddedFile {
    rel_path: ".env.example",
    content: include_str!("../../../templates/portal/.env.example"),
    replace_site_name: false,
  },
  EmbeddedFile {
    rel_path: ".gitignore",
    content: include_str!("../../../templates/portal/.gitignore"),
    replace_site_name: false,
  },
  EmbeddedFile {
    rel_path: "README.md",
    content: include_str!("../../../templates/portal/README.md"),
    replace_site_name: true,
  },
  EmbeddedFile {
    rel_path: "cases/dioxuspress-starter/README.md",
    content: include_str!("../../../templates/portal/cases/dioxuspress-starter/README.md"),
    replace_site_name: true,
  },
  EmbeddedFile {
    rel_path: "cases/dioxuspress-starter/case.yaml",
    content: include_str!("../../../templates/portal/cases/dioxuspress-starter/case.yaml"),
    replace_site_name: false,
  },
  EmbeddedFile {
    rel_path: "courses/fullstack-rust/01-foundations/01-ssr-hydration/index.md",
    content: include_str!(
      "../../../templates/portal/courses/fullstack-rust/01-foundations/01-ssr-hydration/index.md"
    ),
    replace_site_name: false,
  },
  EmbeddedFile {
    rel_path: "courses/fullstack-rust/01-foundations/02-production-ops/index.md",
    content: include_str!(
      "../../../templates/portal/courses/fullstack-rust/01-foundations/02-production-ops/index.md"
    ),
    replace_site_name: false,
  },
  EmbeddedFile {
    rel_path: "courses/fullstack-rust/01-foundations/_chapter.yaml",
    content: include_str!(
      "../../../templates/portal/courses/fullstack-rust/01-foundations/_chapter.yaml"
    ),
    replace_site_name: false,
  },
  EmbeddedFile {
    rel_path: "courses/fullstack-rust/course.yaml",
    content: include_str!("../../../templates/portal/courses/fullstack-rust/course.yaml"),
    replace_site_name: false,
  },
  EmbeddedFile {
    rel_path: "docker-compose.yml",
    content: include_str!("../../../templates/portal/docker-compose.yml"),
    replace_site_name: false,
  },
  EmbeddedFile {
    rel_path: "docs/architecture/content-pipeline/index.md",
    content: include_str!("../../../templates/portal/docs/architecture/content-pipeline/index.md"),
    replace_site_name: true,
  },
  EmbeddedFile {
    rel_path: "docs/architecture/index.md",
    content: include_str!("../../../templates/portal/docs/architecture/index.md"),
    replace_site_name: true,
  },
  EmbeddedFile {
    rel_path: "docs/index.md",
    content: include_str!("../../../templates/portal/docs/index.md"),
    replace_site_name: true,
  },
  EmbeddedFile {
    rel_path: "images/.gitkeep",
    content: include_str!("../../../templates/portal/images/.gitkeep"),
    replace_site_name: false,
  },
  EmbeddedFile {
    rel_path: "pages/about.md",
    content: include_str!("../../../templates/portal/pages/about.md"),
    replace_site_name: true,
  },
  EmbeddedFile {
    rel_path: "podcasts/ep01-architecture/episode.yaml",
    content: include_str!("../../../templates/portal/podcasts/ep01-architecture/episode.yaml"),
    replace_site_name: false,
  },
  EmbeddedFile {
    rel_path: "posts/authors.yml",
    content: include_str!("../../../templates/portal/posts/authors.yml"),
    replace_site_name: false,
  },
  EmbeddedFile {
    rel_path: "posts/launching-knowledge-portal/index.mdx",
    content: include_str!("../../../templates/portal/posts/launching-knowledge-portal/index.mdx"),
    replace_site_name: true,
  },
  EmbeddedFile {
    rel_path: "site.json",
    content: include_str!("../../../templates/portal/site.json"),
    replace_site_name: true,
  },
];

impl Template {
  pub const ALL: &[Self] = &[Self::Docs, Self::Blog, Self::Community, Self::Portal];

  pub fn parse(raw: &str) -> Result<Self, String> {
    match raw.trim().to_ascii_lowercase().as_str() {
      "docs" => Ok(Self::Docs),
      "blog" | "minimal" => Ok(Self::Blog),
      "community" => Ok(Self::Community),
      "portal" => Ok(Self::Portal),
      other => {
        Err(format!("unknown template '{other}' (expected one of: docs, blog, community, portal)"))
      }
    }
  }

  pub const fn id(self) -> &'static str {
    match self {
      Self::Docs => "docs",
      Self::Blog => "blog",
      Self::Community => "community",
      Self::Portal => "portal",
    }
  }

  pub const fn uses_database(self) -> bool {
    matches!(self, Self::Community | Self::Portal)
  }

  pub const fn default_site_name(self) -> &'static str {
    match self {
      Self::Docs => "DioxusPress Docs Starter",
      Self::Blog => "DioxusPress Blog Starter",
      Self::Community => "DioxusPress Community Starter",
      Self::Portal => "DioxusPress Portal Starter",
    }
  }

  pub const fn files(self) -> &'static [EmbeddedFile] {
    match self {
      Self::Docs => DOCS_FILES,
      Self::Blog => BLOG_FILES,
      Self::Community => COMMUNITY_FILES,
      Self::Portal => PORTAL_FILES,
    }
  }
}

pub fn init_site(target_dir: &Path, template: Template, force: bool) -> Result<(), String> {
  let site_json_file = target_dir.join("site.json");
  if site_json_file.exists() && !force {
    return Err(format!("{} already exists (pass --force to overwrite)", site_json_file.display()));
  }

  let raw_name = target_dir
    .file_name()
    .and_then(|s| s.to_str())
    .filter(|s| !s.is_empty() && *s != ".")
    .unwrap_or("My DioxusPress Site");
  let site_name = humanize_site_name(raw_name);
  let json_escaped_name = escape_json_str(&site_name);
  let placeholder = template.default_site_name();

  for entry in template.files() {
    let dest = target_dir.join(entry.rel_path);
    if let Some(parent) = dest.parent() {
      fs::create_dir_all(parent)
        .map_err(|e| format!("failed to create directory {}: {e}", parent.display()))?;
    }
    let rendered = if !entry.replace_site_name {
      entry.content.to_string()
    } else if entry.rel_path == "site.json" {
      entry.content.replace(placeholder, &json_escaped_name)
    } else {
      entry.content.replace(placeholder, &site_name)
    };
    fs::write(&dest, rendered).map_err(|e| format!("failed to write {}: {e}", dest.display()))?;
  }

  Ok(())
}

fn humanize_site_name(slug: &str) -> String {
  let words: Vec<String> = slug
    .split(['-', '_'])
    .filter(|w| !w.is_empty())
    .map(|w| {
      let cleaned: String = w.chars().filter(|c| !c.is_control()).collect();
      let mut chars = cleaned.chars();
      match chars.next() {
        Some(first) => format!("{}{}", first.to_uppercase(), chars.as_str()),
        None => String::new(),
      }
    })
    .filter(|w| !w.is_empty())
    .collect();
  if words.is_empty() {
    "My DioxusPress Site".to_string()
  } else {
    words.join(" ")
  }
}

fn escape_json_str(raw: &str) -> String {
  match serde_json::to_string(raw) {
    Ok(quoted) if quoted.len() >= 2 => quoted[1..quoted.len() - 1].to_string(),
    _ => raw.replace('\\', "\\\\").replace('"', "\\\""),
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use std::collections::BTreeSet;
  use std::path::PathBuf;

  fn collect_relative_files(root: &Path, dir: &Path, out: &mut BTreeSet<String>) {
    let entries = fs::read_dir(dir).unwrap();
    for entry in entries.flatten() {
      let path = entry.path();
      if path.is_dir() {
        collect_relative_files(root, &path, out);
      } else if let Ok(rel) = path.strip_prefix(root) {
        let normalized =
          rel.components().map(|c| c.as_os_str().to_string_lossy()).collect::<Vec<_>>().join("/");
        out.insert(normalized);
      }
    }
  }

  #[test]
  fn embedded_file_list_matches_templates_directory_on_disk() {
    let templates_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../templates");
    for tpl in Template::ALL {
      let preset_dir = templates_root.join(tpl.id());
      assert!(preset_dir.is_dir(), "missing template directory: {}", preset_dir.display());
      let mut disk_files = BTreeSet::new();
      collect_relative_files(&preset_dir, &preset_dir, &mut disk_files);

      let embedded_files: BTreeSet<String> =
        tpl.files().iter().map(|f| f.rel_path.to_string()).collect();
      assert_eq!(
        embedded_files,
        disk_files,
        "Template::{:?} embedded files drifted from templates/{}/ on disk",
        tpl,
        tpl.id()
      );
    }
  }
}
