use std::env;
use std::path::PathBuf;
use std::process::ExitCode;

use dioxuspress_cli::check::check_site;
use dioxuspress_cli::init::{init_site, Template};

const VERSION: &str = env!("CARGO_PKG_VERSION");

fn main() -> ExitCode {
  let args: Vec<String> = env::args().skip(1).collect();
  match run(&args) {
    Ok(code) => code,
    Err(msg) => {
      eprintln!("error: {msg}");
      ExitCode::from(2)
    }
  }
}

fn run(args: &[String]) -> Result<ExitCode, String> {
  let Some(first) = args.first().map(String::as_str) else {
    print_help();
    return Ok(ExitCode::SUCCESS);
  };

  match first {
    "-h" | "--help" | "help" => {
      print_help();
      Ok(ExitCode::SUCCESS)
    }
    "-V" | "--version" | "version" => {
      println!("dioxuspress {VERSION}");
      Ok(ExitCode::SUCCESS)
    }
    "init" => run_init(&args[1..]),
    "check" => run_check(&args[1..]),
    other => Err(format!("unknown command '{other}' (run `dioxuspress --help` for usage)")),
  }
}

fn run_init(args: &[String]) -> Result<ExitCode, String> {
  let mut dir: Option<PathBuf> = None;
  let mut template = Template::Docs;
  let mut force = false;

  let mut i = 0usize;
  while i < args.len() {
    match args[i].as_str() {
      "-h" | "--help" => {
        print_init_help();
        return Ok(ExitCode::SUCCESS);
      }
      "-f" | "--force" => {
        force = true;
      }
      "-t" | "--template" => {
        i += 1;
        let val = args.get(i).ok_or_else(|| "missing value for --template".to_string())?;
        template = Template::parse(val)?;
      }
      arg if arg.starts_with("--template=") => {
        let val = arg.trim_start_matches("--template=");
        template = Template::parse(val)?;
      }
      arg if arg.starts_with('-') => {
        return Err(format!("unknown flag '{arg}' for `dioxuspress init`"));
      }
      positional => {
        if dir.is_some() {
          return Err(format!("unexpected extra argument '{positional}'"));
        }
        dir = Some(PathBuf::from(positional));
      }
    }
    i += 1;
  }

  let target = dir.ok_or_else(|| {
    "missing target directory (usage: `dioxuspress init <dir> [--template docs|blog|community|portal]`)"
      .to_string()
  })?;

  init_site(&target, template, force).map_err(|e| format!("init failed: {e}"))?;

  println!("Initialized DioxusPress '{}' site at {}", template.id(), target.display());
  println!("\nNext steps:");
  println!("  cd {}", target.display());
  if template.uses_database() {
    println!("  cp .env.example .env   # fill in JWT_SECRET and POSTGRES_PASSWORD");
  }
  println!("  dioxuspress check");
  println!("  docker compose up -d");
  println!("  # Open http://localhost:8080");
  Ok(ExitCode::SUCCESS)
}

fn run_check(args: &[String]) -> Result<ExitCode, String> {
  let mut dir: Option<PathBuf> = None;
  for arg in args {
    match arg.as_str() {
      "-h" | "--help" => {
        print_check_help();
        return Ok(ExitCode::SUCCESS);
      }
      flag if flag.starts_with('-') => {
        return Err(format!("unknown flag '{flag}' for `dioxuspress check`"));
      }
      positional => {
        if dir.is_some() {
          return Err(format!("unexpected extra argument '{positional}'"));
        }
        dir = Some(PathBuf::from(positional));
      }
    }
  }

  let target = dir.unwrap_or_else(|| PathBuf::from("."));
  let summary = check_site(&target)?;

  if summary.is_ok() {
    println!(
      "ok: checked {} files and {} internal links in {} (0 issues)",
      summary.files_checked,
      summary.internal_links_checked,
      summary.root.display()
    );
    Ok(ExitCode::SUCCESS)
  } else {
    for diag in &summary.diagnostics {
      eprintln!("{diag}");
    }
    eprintln!(
      "\ncheck failed: {} issue(s) across {} files in {}",
      summary.diagnostics.len(),
      summary.files_checked,
      summary.root.display()
    );
    Ok(ExitCode::from(1))
  }
}

fn print_help() {
  println!(
    "dioxuspress {VERSION} — Official CLI for DioxusPress content sites

USAGE:
  dioxuspress <COMMAND> [OPTIONS]

COMMANDS:
  init <dir> [--template <name>] [--force]   Scaffold a new site (docs | blog | community | portal)
  check [dir]                                Statically validate site.json, Markdown/MDX, and links

OPTIONS:
  -h, --help      Print help
  -V, --version   Print version"
  );
}

fn print_init_help() {
  println!(
    "Usage: dioxuspress init <dir> [--template docs|blog|community|portal] [--force]

Templates:
  docs        Zero-DB documentation site (DocsShell + emerald theme) [default]
  blog        Zero-DB minimalist blog / digital garden (MinimalShell + coffee theme)
  community   Fullstack community with OAuth login, forum & annotations (Postgres)
  portal      Fullstack knowledge portal with courses, cases & moderation (Postgres)"
  );
}

fn print_check_help() {
  println!(
    "Usage: dioxuspress check [dir]

Validates:
  - site.json syntax, unknown/misspelled fields, enum values, and /images/* existence
  - Markdown/MDX YAML frontmatter syntax and required fields (title, date)
  - Fenced code block (``` / ~~~) and admonition (:::) balance
  - Self-closing custom MDX component syntax (<YouTube ... />, <Yellow ... />, etc.)
  - Internal /docs/*, /blog/*, /images/*, /posts/*, and relative file links"
  );
}
