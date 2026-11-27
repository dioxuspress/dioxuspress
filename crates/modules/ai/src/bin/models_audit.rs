//! 开源模型架构核验与每日巡检 CLI（`models-audit`）。
//!
//! 典型用法：
//! ```sh
//! # 1. 单模型即时取证（无需数据库，输出完整 L1×L2 诊断证据包 JSON，供人工或 model-arch-reviewer Skill 消费）
//! models-audit --check deepseek-ai/DeepSeek-V4-Pro --source modelscope
//!
//! # 2. 每日巡检数据库已收录模型（需 DATABASE_URL；退化模型绝不覆盖线上快照，仅入队告警）
//! models-audit --audit-curated
//!
//! # 3. 每日发现并预审 HF + ModelScope 热门新开源模型（绿灯进候选，黄红灯拦截入待复核队列）
//! models-audit --discover-trending --limit 12
//! models-audit --discover-trending --trending-source modelscope --limit 8 --dry-run
//!
//! # 4. 试运行（不写库，仅打印巡检结果）
//! models-audit --discover-trending --limit 5 --dry-run
//! ```

use std::process::ExitCode;

use module_ai::audit::{audit_curated_models, check_single_repo, discover_and_audit_trending};

#[derive(Debug, Default)]
struct Args {
  check_repo: Option<String>,
  source: String,
  revision: String,
  audit_curated: bool,
  discover_trending: bool,
  trending_source: String,
  limit: usize,
  dry_run: bool,
}

fn parse_args() -> Result<Args, String> {
  let mut args = Args {
    source: "auto".to_string(),
    trending_source: "all".to_string(),
    limit: 12,
    ..Args::default()
  };
  let mut it = std::env::args().skip(1);
  while let Some(flag) = it.next() {
    match flag.as_str() {
      "--check" => {
        args.check_repo = Some(it.next().ok_or("--check 缺少仓库名 (owner/model)")?);
      }
      "--source" => {
        args.source = it.next().ok_or("--source 缺少值 (auto|hf|modelscope)")?;
      }
      "--revision" => {
        args.revision = it.next().ok_or("--revision 缺少分支/版本号")?;
      }
      "--audit-curated" => args.audit_curated = true,
      "--discover-trending" => args.discover_trending = true,
      "--trending-source" => {
        args.trending_source = it.next().ok_or("--trending-source 缺少值 (all|hf|modelscope)")?;
      }
      "--limit" => {
        let raw = it.next().ok_or("--limit 缺少正整数")?;
        let n: usize = raw.parse().map_err(|_| format!("--limit 非法整数: {raw}"))?;
        if n == 0 {
          return Err("--limit 必须大于 0".to_string());
        }
        args.limit = n;
      }
      "--dry-run" => args.dry_run = true,
      "-h" | "--help" => {
        println!(
          "models-audit — DioxusPress 模型架构 L1×L2 硬校验与每日巡检工具\n\
           \n\
           用法:\n\
             models-audit --check <owner/repo> [--source auto|hf|modelscope] [--revision <rev>]\n\
             models-audit --audit-curated [--dry-run]\n\
             models-audit --discover-trending [--trending-source all|hf|modelscope] [--limit 12] [--dry-run]"
        );
        std::process::exit(0);
      }
      other => return Err(format!("未知参数: {other}")),
    }
  }
  if args.check_repo.is_none() && !args.audit_curated && !args.discover_trending {
    return Err(
      "请至少指定 --check <owner/repo>、--audit-curated 或 --discover-trending 之一".to_string(),
    );
  }
  Ok(args)
}

#[tokio::main]
async fn main() -> ExitCode {
  let _ = dotenvy::dotenv();
  tracing_subscriber::fmt()
    .with_env_filter(
      tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
    )
    .with_writer(std::io::stderr)
    .init();

  let args = match parse_args() {
    Ok(a) => a,
    Err(e) => {
      eprintln!("参数错误: {e}");
      return ExitCode::from(2);
    }
  };

  if let Some(ref repo) = args.check_repo {
    match check_single_repo(repo, &args.source, &args.revision).await {
      Ok((_bp, evidence)) => match serde_json::to_string_pretty(&evidence) {
        Ok(json) => {
          println!("{json}");
          return if evidence.verification.circuit_broken {
            ExitCode::from(3)
          } else {
            ExitCode::SUCCESS
          };
        }
        Err(e) => {
          eprintln!("序列化证据包失败: {e}");
          return ExitCode::FAILURE;
        }
      },
      Err(e) => {
        eprintln!("解析失败 ({repo}): {e}");
        return ExitCode::FAILURE;
      }
    }
  }

  let db_opt = if args.audit_curated || !args.dry_run {
    match app_core::db::get_or_init_pool().await {
      Ok(pool) => Some(pool),
      Err(e) => {
        eprintln!("连接数据库失败 (请检查 DATABASE_URL): {e}");
        return ExitCode::FAILURE;
      }
    }
  } else {
    None
  };

  if args.audit_curated {
    let Some(ref db) = db_opt else {
      eprintln!("--audit-curated 需要可用的 DATABASE_URL");
      return ExitCode::FAILURE;
    };
    match audit_curated_models(db, args.dry_run).await {
      Ok(summary) => {
        if let Ok(js) = serde_json::to_string_pretty(&summary) {
          println!("curated_audit: {js}");
        }
      }
      Err(e) => {
        eprintln!("存量巡检失败: {e}");
        return ExitCode::FAILURE;
      }
    }
  }

  if args.discover_trending {
    match discover_and_audit_trending(
      db_opt.as_ref(),
      &args.trending_source,
      args.limit,
      args.dry_run,
    )
    .await
    {
      Ok((summary, evidences)) => {
        let out = serde_json::json!({
          "summary": summary,
          "results": evidences,
        });
        if let Ok(js) = serde_json::to_string_pretty(&out) {
          println!("{js}");
        }
      }
      Err(e) => {
        eprintln!("热门模型预审失败: {e}");
        return ExitCode::FAILURE;
      }
    }
  }

  ExitCode::SUCCESS
}
