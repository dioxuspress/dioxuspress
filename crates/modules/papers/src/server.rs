//! 页面用的 server fn，以及 app 组装 sitemap / 搜索时调用的服务端函数。
//! 只读已经生成好解读（`status = done`）的论文。

use dioxus::prelude::*;
use serde::{Deserialize, Serialize};

use crate::digest::PaperDigest;

/// 列表页默认与最大每页篇数。
pub const DEFAULT_PAGE_SIZE: u64 = 30;
pub const MAX_PAGE_SIZE: u64 = 100;

/// 兼容旧接口 `list_papers` 的最大返回条数。
#[cfg(feature = "server")]
const LIST_LIMIT: u64 = 200;

/// 列表里的一篇。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PaperCard {
  pub arxiv_id: String,
  /// 界面语言的标题：中文界面有中文标题就用中文
  pub title: String,
  /// HF 日期，`YYYY-MM-DD`
  pub date: String,
  pub hf_upvotes: i32,
  /// 界面语言的一句话总结，没有就用英文
  pub tldr: String,
  /// 主题 slug，见 `topics.rs`
  pub topics: Vec<String>,
  /// 是否关联有效的 GitHub 开源仓库链接
  #[serde(default)]
  pub has_code: bool,
  /// 是否基于 arXiv HTML 全文生成（false 表示仅基于摘要）
  #[serde(default)]
  pub has_html: bool,
}

/// 分页列表结果（含各主题计数，避免 N+1 查询）。
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct PaperPage {
  pub items: Vec<PaperCard>,
  /// 当前筛选条件下的总篇数
  pub total: u64,
  /// 全站全部已发布论文总数（用于「全部」主题 chip）
  pub all_total: u64,
  /// 当前页码（1-based）
  pub page: u64,
  pub page_size: u64,
  pub has_more: bool,
  /// 全站已发布论文在 12 个主题下的篇数，按 `TOPICS` 顺序排列
  pub topic_counts: Vec<(String, u64)>,
}

/// 单篇解读页的数据。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PaperDetail {
  pub arxiv_id: String,
  /// 原文标题（英文）
  pub title: String,
  pub title_zh: Option<String>,
  pub topics: Vec<String>,
  /// arXiv 首次发布时间，`YYYY-MM-DD`
  pub published: String,
  pub authors: Vec<String>,
  pub date: String,
  pub hf_upvotes: i32,
  pub github_repo: Option<String>,
  /// false：没有 HTML 全文，只依据摘要生成
  pub has_html: bool,
  pub arxiv_version: Option<i32>,
  /// 实际显示的语言（要的语言没有时退回 `en`）
  pub lang: String,
  pub model: String,
  pub digest: PaperDigest,
  /// 原文里没找到的数字
  pub unverified_numbers: Vec<String>,
  /// 同主题近期高热度相关论文（最多 4 篇）
  #[serde(default)]
  pub related: Vec<PaperCard>,
}

/// 规范化每页条数：0 使用默认值，其余限制在 `[1, MAX_PAGE_SIZE]`。
pub fn normalize_page_size(raw: u64) -> u64 {
  if raw == 0 {
    DEFAULT_PAGE_SIZE
  } else {
    raw.clamp(1, MAX_PAGE_SIZE)
  }
}

/// 规范化排序参数：只接受 `latest` / `top_week` / `top_all`，未知值退回 `latest`。
pub fn normalize_sort(raw: &str) -> &'static str {
  match raw {
    "top_week" => "top_week",
    "top_all" => "top_all",
    _ => "latest",
  }
}

/// 计算总页数。
pub fn total_pages(total: u64, page_size: u64) -> u64 {
  let ps = normalize_page_size(page_size);
  total.div_ceil(ps)
}

/// 校验是否为有效的开源仓库链接（与详情页展示逻辑保持一致）。
pub fn has_valid_code_repo(repo: Option<&str>) -> bool {
  repo.is_some_and(|r| r.trim().starts_with("https://"))
}

/// 统计各主题下的已发布论文数，顺序与 `TOPICS` 一致。
pub fn count_topics(raw_topics: &[Option<String>]) -> Vec<(String, u64)> {
  let mut counts = std::collections::HashMap::<String, u64>::new();
  for entry in raw_topics {
    if let Some(json) = entry.as_deref() {
      if let Ok(slugs) = serde_json::from_str::<Vec<String>>(json) {
        for slug in slugs {
          *counts.entry(slug).or_insert(0) += 1;
        }
      }
    }
  }
  crate::topics::TOPICS
    .iter()
    .map(|t| (t.slug.to_string(), counts.get(t.slug).copied().unwrap_or(0)))
    .collect()
}

/// 分页查询已发布论文，支持主题、排序（`latest` / `top_week` / `top_all`）与仅看有代码筛选。
#[post("/api/papers/page")]
pub async fn list_papers_page(
  lang: String,
  topic: String,
  sort: String,
  code_only: bool,
  page: u64,
  page_size: u64,
) -> Result<PaperPage, ServerFnError> {
  #[cfg(feature = "server")]
  {
    use crate::entities::paper;
    use sea_orm::{ColumnTrait, EntityTrait, PaginatorTrait, QueryFilter, QueryOrder, QuerySelect};

    let ps = normalize_page_size(page_size);
    let sort_mode = normalize_sort(&sort);
    let db = open_db().await?;

    // 单次轻量查询统计全站已发布论文数及 12 主题篇数
    let all_topic_rows: Vec<Option<String>> = paper::Entity::find()
      .select_only()
      .column(paper::Column::Topics)
      .filter(paper::Column::Status.eq("done"))
      .into_tuple()
      .all(&db)
      .await
      .map_err(|e| ServerFnError::new(e.to_string()))?;
    let all_total = all_topic_rows.len() as u64;
    let topic_counts = count_topics(&all_topic_rows);

    let mut query = paper::Entity::find().filter(paper::Column::Status.eq("done"));
    if !topic.is_empty() {
      // 只接受列表里的 slug：它们只含小写字母、数字和 `-`，拼进 LIKE 不用转义
      if crate::topics::find(&topic).is_none() {
        return Ok(PaperPage {
          items: vec![],
          total: 0,
          all_total,
          page: 1,
          page_size: ps,
          has_more: false,
          topic_counts,
        });
      }
      query = query.filter(paper::Column::Topics.contains(format!("\"{topic}\"")));
    }
    if code_only {
      query = query
        .filter(paper::Column::GithubRepo.is_not_null())
        .filter(paper::Column::GithubRepo.starts_with("https://"));
    }
    if sort_mode == "top_week" {
      let cutoff = chrono::Utc::now().date_naive() - chrono::Duration::days(7);
      query = query.filter(paper::Column::FirstSeenOn.gte(cutoff));
    }

    let total = query.clone().count(&db).await.map_err(|e| ServerFnError::new(e.to_string()))?;
    let pages = total_pages(total, ps);
    let safe_page = if pages == 0 { 1 } else { page.max(1).min(pages) };
    let offset = (safe_page - 1) * ps;

    let ordered = match sort_mode {
      "top_week" | "top_all" => query
        .order_by_desc(paper::Column::HfUpvotes)
        .order_by_desc(paper::Column::FirstSeenOn)
        .order_by_desc(paper::Column::ArxivId),
      _ => query
        .order_by_desc(paper::Column::FirstSeenOn)
        .order_by_desc(paper::Column::HfUpvotes)
        .order_by_desc(paper::Column::ArxivId),
    };

    let rows = ordered
      .offset(offset)
      .limit(ps)
      .all(&db)
      .await
      .map_err(|e| ServerFnError::new(e.to_string()))?;

    let items = hydrate_cards(&db, rows, &lang).await?;
    Ok(PaperPage {
      items,
      total,
      all_total,
      page: safe_page,
      page_size: ps,
      has_more: safe_page < pages,
      topic_counts,
    })
  }
  #[cfg(not(feature = "server"))]
  {
    let _ = (lang, topic, sort, code_only, page, page_size);
    Ok(PaperPage::default())
  }
}

/// 最近的论文，按 HF 日期倒序、同一天按 upvotes 倒序。`topic` 为空时不按主题筛。
/// 保留原签名兼容外部调用；列表页已迁移至 [`list_papers_page`]。
#[post("/api/papers/list")]
pub async fn list_papers(lang: String, topic: String) -> Result<Vec<PaperCard>, ServerFnError> {
  #[cfg(feature = "server")]
  {
    use crate::entities::paper;
    use sea_orm::{ColumnTrait, EntityTrait, QueryFilter, QueryOrder, QuerySelect};
    let db = open_db().await?;
    let mut query = paper::Entity::find().filter(paper::Column::Status.eq("done"));
    if !topic.is_empty() {
      if crate::topics::find(&topic).is_none() {
        return Ok(vec![]);
      }
      query = query.filter(paper::Column::Topics.contains(format!("\"{topic}\"")));
    }
    let rows = query
      .order_by_desc(paper::Column::FirstSeenOn)
      .order_by_desc(paper::Column::HfUpvotes)
      .limit(LIST_LIMIT)
      .all(&db)
      .await
      .map_err(|e| ServerFnError::new(e.to_string()))?;
    hydrate_cards(&db, rows, &lang).await
  }
  #[cfg(not(feature = "server"))]
  {
    let _ = (lang, topic);
    Ok(vec![])
  }
}

/// 单篇解读。论文不存在或还没生成时返回 `None`。
#[post("/api/papers/get")]
pub async fn get_paper(
  arxiv_id: String,
  lang: String,
) -> Result<Option<PaperDetail>, ServerFnError> {
  #[cfg(feature = "server")]
  {
    use crate::entities::{paper, paper_summary};
    use sea_orm::{ColumnTrait, EntityTrait, QueryFilter};
    let db = open_db().await?;
    let Some(p) = paper::Entity::find_by_id(arxiv_id.clone())
      .filter(paper::Column::Status.eq("done"))
      .one(&db)
      .await
      .map_err(|e| ServerFnError::new(e.to_string()))?
    else {
      return Ok(None);
    };
    let summaries = paper_summary::Entity::find()
      .filter(paper_summary::Column::ArxivId.eq(&arxiv_id))
      .all(&db)
      .await
      .map_err(|e| ServerFnError::new(e.to_string()))?;
    let Some(s) = pick_summary(&summaries, &p.arxiv_id, &lang) else {
      return Ok(None);
    };
    let digest: PaperDigest =
      serde_json::from_str(&s.digest).map_err(|e| ServerFnError::new(e.to_string()))?;
    let topics = parse_topics(p.topics.as_deref());
    let related = load_related_papers(&db, &p.arxiv_id, &topics, &lang).await;
    Ok(Some(PaperDetail {
      arxiv_id: p.arxiv_id,
      title: p.title,
      title_zh: p.title_zh.filter(|t| !t.is_empty()),
      topics,
      published: p.published_at.date_naive().to_string(),
      authors: serde_json::from_str(&p.authors).unwrap_or_default(),
      date: p.first_seen_on.to_string(),
      hf_upvotes: p.hf_upvotes,
      github_repo: p.github_repo,
      has_html: p.has_html.unwrap_or(false),
      arxiv_version: p.arxiv_version,
      lang: s.lang.clone(),
      model: s.model.clone(),
      digest,
      unverified_numbers: serde_json::from_str(&s.unverified_numbers).unwrap_or_default(),
      related,
    }))
  }
  #[cfg(not(feature = "server"))]
  {
    let _ = (arxiv_id, lang);
    Ok(None)
  }
}

/// 批量把 `papers` 行与对应语言的 `paper_summaries` 拼成 [`PaperCard`]。
#[cfg(feature = "server")]
async fn hydrate_cards(
  db: &sea_orm::DatabaseConnection,
  rows: Vec<crate::entities::paper::Model>,
  lang: &str,
) -> Result<Vec<PaperCard>, ServerFnError> {
  use crate::entities::paper_summary;
  use sea_orm::{ColumnTrait, EntityTrait, QueryFilter};

  if rows.is_empty() {
    return Ok(vec![]);
  }
  let ids: Vec<String> = rows.iter().map(|p| p.arxiv_id.clone()).collect();
  let summaries = paper_summary::Entity::find()
    .filter(paper_summary::Column::ArxivId.is_in(ids))
    .filter(paper_summary::Column::Lang.is_in([lang, "en"]))
    .all(db)
    .await
    .map_err(|e| ServerFnError::new(e.to_string()))?;

  Ok(
    rows
      .into_iter()
      .map(|p| {
        let tldr = pick_summary(&summaries, &p.arxiv_id, lang)
          .and_then(|s| serde_json::from_str::<PaperDigest>(&s.digest).ok())
          .map(|d| d.tldr)
          .unwrap_or_default();
        let title = match p.title_zh {
          Some(zh) if lang == "zh" && !zh.is_empty() => zh,
          _ => p.title,
        };
        let has_code = has_valid_code_repo(p.github_repo.as_deref());
        PaperCard {
          arxiv_id: p.arxiv_id,
          title,
          date: p.first_seen_on.to_string(),
          hf_upvotes: p.hf_upvotes,
          tldr,
          topics: parse_topics(p.topics.as_deref()),
          has_code,
          has_html: p.has_html.unwrap_or(false),
        }
      })
      .collect(),
  )
}

/// 相关论文（最多 4 篇）：优先匹配主主题近 30 天高热度论文，不足则按主主题历史高热度补足。
#[cfg(feature = "server")]
async fn load_related_papers(
  db: &sea_orm::DatabaseConnection,
  current_id: &str,
  topics: &[String],
  lang: &str,
) -> Vec<PaperCard> {
  use crate::entities::paper;
  use sea_orm::{ColumnTrait, EntityTrait, QueryFilter, QueryOrder, QuerySelect};

  const MAX_RELATED: usize = 4;
  let Some(primary) = topics.first().filter(|t| crate::topics::find(t).is_some()) else {
    return Vec::new();
  };
  let needle = format!("\"{primary}\"");
  let cutoff = chrono::Utc::now().date_naive() - chrono::Duration::days(30);

  let mut picked = paper::Entity::find()
    .filter(paper::Column::Status.eq("done"))
    .filter(paper::Column::ArxivId.ne(current_id))
    .filter(paper::Column::Topics.contains(&needle))
    .filter(paper::Column::FirstSeenOn.gte(cutoff))
    .order_by_desc(paper::Column::HfUpvotes)
    .order_by_desc(paper::Column::FirstSeenOn)
    .limit(MAX_RELATED as u64)
    .all(db)
    .await
    .unwrap_or_default();

  if picked.len() < MAX_RELATED {
    let mut exclude: Vec<String> = picked.iter().map(|p| p.arxiv_id.clone()).collect();
    exclude.push(current_id.to_string());
    let need = (MAX_RELATED - picked.len()) as u64;
    let more = paper::Entity::find()
      .filter(paper::Column::Status.eq("done"))
      .filter(paper::Column::ArxivId.is_not_in(exclude))
      .filter(paper::Column::Topics.contains(&needle))
      .order_by_desc(paper::Column::HfUpvotes)
      .order_by_desc(paper::Column::FirstSeenOn)
      .limit(need)
      .all(db)
      .await
      .unwrap_or_default();
    picked.extend(more);
  }

  hydrate_cards(db, picked, lang).await.unwrap_or_default()
}

/// 独立论文 Atom 订阅源（`/papers/feed.xml`）用的最近 `limit` 篇论文条目。
#[cfg(feature = "server")]
pub async fn feed_entries(limit: u64) -> Vec<widgets::ContentEntry> {
  use crate::entities::paper;
  use sea_orm::{ColumnTrait, EntityTrait, QueryFilter, QueryOrder, QuerySelect};

  let Ok(db) = app_core::db::get_or_init_pool().await else {
    return Vec::new();
  };
  let rows = paper::Entity::find()
    .filter(paper::Column::Status.eq("done"))
    .order_by_desc(paper::Column::FirstSeenOn)
    .order_by_desc(paper::Column::HfUpvotes)
    .limit(limit.clamp(1, 100))
    .all(&db)
    .await
    .unwrap_or_default();
  let cards = hydrate_cards(&db, rows, "zh").await.unwrap_or_default();
  cards
    .into_iter()
    .map(|c| widgets::ContentEntry {
      url_path: format!("/papers/{}", c.arxiv_id),
      title: c.title,
      description: c.tldr,
      date: c.date,
      tags: c.topics,
    })
    .collect()
}

/// `papers.topics` 的 JSON 文本 → slug 列表；没分类或解析不了时为空。
#[cfg(feature = "server")]
fn parse_topics(json: Option<&str>) -> Vec<String> {
  json.and_then(|j| serde_json::from_str(j).ok()).unwrap_or_default()
}

/// 要的语言，没有就英文。
#[cfg(feature = "server")]
fn pick_summary<'a>(
  summaries: &'a [crate::entities::paper_summary::Model],
  arxiv_id: &str,
  lang: &str,
) -> Option<&'a crate::entities::paper_summary::Model> {
  let find = |l: &str| summaries.iter().find(|s| s.arxiv_id == arxiv_id && s.lang == l);
  find(lang).or_else(|| find("en"))
}

#[cfg(feature = "server")]
async fn open_db() -> Result<sea_orm::DatabaseConnection, ServerFnError> {
  app_core::db::get_or_init_pool().await.map_err(|e| ServerFnError::new(e.to_string()))
}

/// 全部已生成论文的 `(/papers/<id>, 更新日期)` 与各主题页（无日期），以及它们和 `/papers` 的
/// 英文版（`?lang=en`），给 sitemap。
/// 数据库不可用时为空。
#[cfg(feature = "server")]
pub async fn sitemap_paths() -> Vec<(String, Option<String>)> {
  use crate::entities::paper;
  use sea_orm::{ColumnTrait, EntityTrait, QueryFilter, QuerySelect};
  let Ok(db) = app_core::db::get_or_init_pool().await else {
    return Vec::new();
  };
  let rows: Vec<(String, chrono::DateTime<chrono::FixedOffset>)> = paper::Entity::find()
    .select_only()
    .column(paper::Column::ArxivId)
    .column(paper::Column::UpdatedAt)
    .filter(paper::Column::Status.eq("done"))
    .into_tuple()
    .all(&db)
    .await
    .unwrap_or_default();
  let topic_pages =
    crate::topics::TOPICS.iter().map(|t| (format!("/papers/topic/{}", t.slug), None));
  let zh: Vec<(String, Option<String>)> = topic_pages
    .chain(
      rows
        .into_iter()
        .map(|(id, at)| (format!("/papers/{id}"), Some(at.format("%Y-%m-%d").to_string()))),
    )
    .collect();
  // 论文页有英文版（`?lang=en`，docs/I18N_URLS.md），一并列出；中文 /papers 在模块静态路径里
  let en: Vec<(String, Option<String>)> = std::iter::once(("/papers".to_string(), None))
    .chain(zh.iter().cloned())
    .map(|(p, m)| (widgets::lang_path(&p, true), m))
    .collect();
  zh.into_iter().chain(en).collect()
}

/// 搜索索引用的文档（kind = `paper`）：标题、摘要、各语言的一句话总结与要点。
#[cfg(feature = "server")]
pub async fn search_docs() -> Vec<app_core::engines::doc_source::ExternalIndexedDoc> {
  use crate::entities::{paper, paper_summary};
  use sea_orm::{ColumnTrait, EntityTrait, QueryFilter};
  let Ok(db) = app_core::db::get_or_init_pool().await else {
    return Vec::new();
  };
  let papers = match paper::Entity::find()
    .filter(paper::Column::Status.eq("done"))
    .find_with_related(paper_summary::Entity)
    .all(&db)
    .await
  {
    Ok(rows) => rows,
    Err(e) => {
      tracing::warn!(error = %e, "papers: search source unavailable");
      return Vec::new();
    }
  };
  papers
    .into_iter()
    .map(|(p, summaries)| {
      let mut body = p.title_zh.clone().unwrap_or_default();
      body.push(' ');
      body.push_str(&p.r#abstract);
      for s in summaries {
        if let Ok(d) = serde_json::from_str::<PaperDigest>(&s.digest) {
          body.push(' ');
          body.push_str(&d.tldr);
          for k in d.key_points {
            body.push(' ');
            body.push_str(&k.text);
          }
        }
      }
      app_core::engines::doc_source::ExternalIndexedDoc {
        kind: "paper".to_string(),
        url: format!("/papers/{}", p.arxiv_id),
        ref_id: p.arxiv_id,
        title: p.title,
        body,
        created_at: p.first_seen_on.to_string(),
      }
    })
    .collect()
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn page_size_and_total_pages_bounds() {
    assert_eq!(normalize_page_size(0), DEFAULT_PAGE_SIZE);
    assert_eq!(normalize_page_size(15), 15);
    assert_eq!(normalize_page_size(999), MAX_PAGE_SIZE);

    assert_eq!(total_pages(0, 30), 0);
    assert_eq!(total_pages(1, 30), 1);
    assert_eq!(total_pages(30, 30), 1);
    assert_eq!(total_pages(31, 30), 2);
    assert_eq!(total_pages(347, 30), 12);
  }

  #[test]
  fn sort_mode_normalization() {
    assert_eq!(normalize_sort("latest"), "latest");
    assert_eq!(normalize_sort("top_week"), "top_week");
    assert_eq!(normalize_sort("top_all"), "top_all");
    assert_eq!(normalize_sort("unknown"), "latest");
    assert_eq!(normalize_sort(""), "latest");
  }

  #[test]
  fn valid_code_repo_requires_https() {
    assert!(has_valid_code_repo(Some("https://github.com/org/repo")));
    assert!(has_valid_code_repo(Some(" https://github.com/org/repo ")));
    assert!(!has_valid_code_repo(Some("http://github.com/org/repo")));
    assert!(!has_valid_code_repo(Some("")));
    assert!(!has_valid_code_repo(None));
  }

  #[test]
  fn count_topics_follows_canonical_order() {
    let rows = vec![
      Some(r#"["agents","llm"]"#.to_string()),
      Some(r#"["agents"]"#.to_string()),
      Some("not-json".to_string()),
      None,
    ];
    let counts = count_topics(&rows);
    assert_eq!(counts.len(), crate::topics::TOPICS.len());
    let find = |slug: &str| counts.iter().find(|(s, _)| s == slug).map(|(_, n)| *n);
    assert_eq!(find("agents"), Some(2));
    assert_eq!(find("llm"), Some(1));
    assert_eq!(find("code"), Some(0));
  }
}
