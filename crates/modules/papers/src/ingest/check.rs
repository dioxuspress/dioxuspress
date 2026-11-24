//! 生成结果的校验：出处节号必须存在、导图不超限、数字能在原文找到，翻译不改结构。

use crate::digest::{clean_section, MindNode, PaperDigest};

/// 导图在根以下最多几层。
pub const MINDMAP_MAX_DEPTH: usize = 3;
/// 导图任一节点最多几个子节点。
pub const MINDMAP_MAX_CHILDREN: usize = 6;

/// 通过校验后的附带信息。
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Checked {
  /// 原文里没找到的数字（不删，页面上标「未核实」）
  pub unverified_numbers: Vec<String>,
}

/// 把模型写出的节号归并到原文已知节号（精确匹配优先；未命中时允许剥离
/// `Section` / `Appendix` 前缀与节号后面的标题，前提是能归并到已知节号，
/// 且节号后紧跟空白或分隔符，不会把 `"4.1"` 误判成 `"4"`）。
pub fn resolve_section<'a>(raw: &str, sections: &'a [String]) -> Option<&'a str> {
  let cleaned = clean_section(raw);
  if cleaned.is_empty() {
    return None;
  }
  if cleaned == "Abstract" {
    return Some("Abstract");
  }
  if let Some(exact) = sections.iter().find(|x| x.as_str() == cleaned) {
    return Some(exact.as_str());
  }

  let candidate = strip_section_prefix(cleaned);
  if let Some(exact) = sections.iter().find(|x| x.as_str() == candidate) {
    return Some(exact.as_str());
  }

  // 允许 `"4.6 Intra-Model ..."` / `"H Appendix H ..."` / `"4.6: Title"` 匹配已知节号，
  // 但禁止 `"4.1"` 匹配 `"4"`（`.` 不是边界分隔符）。取最长已知节号避免歧义。
  sections
    .iter()
    .filter(|known| {
      let Some(rest) = candidate.strip_prefix(known.as_str()) else {
        return false;
      };
      if rest.is_empty() {
        return true;
      }
      // 末尾仅多了一个点（如 `"4.6."`）或后接空白/冒号/破折号
      if rest == "." {
        return true;
      }
      let mut chars = rest.chars();
      match chars.next() {
        Some(c) if c.is_whitespace() || matches!(c, ':' | '—' | '–' | '-') => true,
        Some('.') => chars.next().is_some_and(char::is_whitespace),
        _ => false,
      }
    })
    .max_by_key(|known| known.len())
    .map(String::as_str)
}

fn strip_section_prefix(s: &str) -> &str {
  for prefix in
    ["Section ", "section ", "Sec. ", "sec. ", "Appendix ", "appendix ", "App. ", "app. "]
  {
    if let Some(rest) = s.strip_prefix(prefix) {
      return rest.trim_start();
    }
  }
  s
}

/// 将解读里能归并的节号规范化回原文节号（例如 `"4.6 Intra-Model ..."` → `"4.6"`）。
pub fn normalize_sections(d: &mut PaperDigest, sections: &[String]) {
  for r in &mut d.results {
    if let Some(canonical) = resolve_section(&r.section, sections) {
      r.section = canonical.to_string();
    }
  }
  for k in &mut d.key_points {
    if let Some(canonical) = resolve_section(&k.section, sections) {
      k.section = canonical.to_string();
    }
  }
}

/// 校验英文解读。`sections` 是原文里存在的节号（含无编号节的标题），
/// `source` 是喂给模型的全文。返回 `Err` 时列出全部问题，用来让模型重写一次。
pub fn check_digest(
  d: &PaperDigest,
  sections: &[String],
  source: &str,
) -> Result<Checked, Vec<String>> {
  let mut problems = Vec::new();
  if d.tldr.trim().is_empty() {
    problems.push("tldr is empty".to_string());
  }
  if d.results.is_empty() || d.key_points.is_empty() {
    problems.push("results and key_points must not be empty".to_string());
  }
  let cited = d
    .results
    .iter()
    .map(|r| ("results", &r.section))
    .chain(d.key_points.iter().map(|k| ("key_points", &k.section)));
  for (kind, section) in cited {
    let s = clean_section(section);
    if resolve_section(s, sections).is_none() {
      problems.push(format!("{kind}: section \"{s}\" does not exist in the paper"));
    }
  }
  if d.mindmap.depth() > MINDMAP_MAX_DEPTH {
    problems.push(format!(
      "mindmap has {} levels below the root, at most {MINDMAP_MAX_DEPTH} allowed",
      d.mindmap.depth()
    ));
  }
  if d.mindmap.widest() > MINDMAP_MAX_CHILDREN {
    problems.push(format!(
      "a mindmap node has {} children, at most {MINDMAP_MAX_CHILDREN} allowed",
      d.mindmap.widest()
    ));
  }
  if !problems.is_empty() {
    return Err(problems);
  }

  let src = normalize_number(source);
  let unverified_numbers = d
    .results
    .iter()
    .flat_map(|r| r.numbers.iter())
    .filter(|n| n.chars().any(|c| c.is_ascii_digit()) && !src.contains(&normalize_number(n)))
    .cloned()
    .collect();
  Ok(Checked { unverified_numbers })
}

/// 翻译只能改文字：条目数、节号、数字、导图形状都要和英文版一样。
/// 发现差异时指明具体索引与期望值，让二次重试精准修正。
pub fn check_translation(en: &PaperDigest, tr: &PaperDigest) -> Result<(), Vec<String>> {
  let mut problems = Vec::new();
  if en.results.len() != tr.results.len() {
    problems.push(format!(
      "results: expected {} items, got {}",
      en.results.len(),
      tr.results.len()
    ));
  } else {
    for (i, (a, b)) in en.results.iter().zip(&tr.results).enumerate() {
      if clean_section(&a.section) != clean_section(&b.section) {
        problems.push(format!(
          "results[{i}].section changed: expected \"{}\", got \"{}\"",
          clean_section(&a.section),
          clean_section(&b.section)
        ));
      }
      if a.numbers != b.numbers {
        problems.push(format!(
          "results[{i}].numbers changed: expected {:?}, got {:?} (copy numbers verbatim without translating)",
          a.numbers, b.numbers
        ));
      }
    }
  }
  if en.key_points.len() != tr.key_points.len() {
    problems.push(format!(
      "key_points: expected {} items, got {}",
      en.key_points.len(),
      tr.key_points.len()
    ));
  } else {
    for (i, (a, b)) in en.key_points.iter().zip(&tr.key_points).enumerate() {
      if clean_section(&a.section) != clean_section(&b.section) {
        problems.push(format!(
          "key_points[{i}].section changed: expected \"{}\", got \"{}\"",
          clean_section(&a.section),
          clean_section(&b.section)
        ));
      }
    }
  }
  if en.limitations.len() != tr.limitations.len() {
    problems.push(format!(
      "limitations: expected {} items, got {}",
      en.limitations.len(),
      tr.limitations.len()
    ));
  }
  if !same_shape(&en.mindmap, &tr.mindmap) {
    problems.push("mindmap: shape changed".to_string());
  }
  if problems.is_empty() {
    Ok(())
  } else {
    Err(problems)
  }
}

fn same_shape(a: &MindNode, b: &MindNode) -> bool {
  a.children.len() == b.children.len()
    && a.children.iter().zip(&b.children).all(|(x, y)| same_shape(x, y))
}

/// 数字比对前的规范化：原文里的数字常在 LaTeX 公式里（`$49\times$`、`$84.6\%$`），
/// 模型抄出来是 `49×`、`84.6%`；区间的连字符也有好几种写法。
pub fn normalize_number(s: &str) -> String {
  let mut t = s.to_string();
  for (from, to) in [
    ("\\times", "×"),
    ("\\%", "%"),
    ("\\pm", "±"),
    ("\\sim", "~"),
    ("\\approx", "≈"),
    ("\\,", ""),
    ("\\!", ""),
    ("−", "-"),
    ("–", "-"),
    ("—", "-"),
  ] {
    t = t.replace(from, to);
  }
  t.chars().filter(|c| !c.is_whitespace() && !matches!(c, '$' | ',' | '{' | '}')).collect()
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::digest::{KeyPoint, ResultItem};

  fn leaf(l: &str) -> MindNode {
    MindNode { label: l.into(), children: vec![] }
  }

  fn digest() -> PaperDigest {
    PaperDigest {
      tldr: "t".into(),
      problem: "p".into(),
      method: "m".into(),
      results: vec![ResultItem {
        claim: "c".into(),
        numbers: vec!["49×".into(), "82.8–84.6%".into(), "0.6439".into()],
        section: "§4.2".into(),
      }],
      limitations: vec!["l".into()],
      key_points: vec![
        KeyPoint { text: "k".into(), section: "3".into() },
        KeyPoint { text: "k".into(), section: "Limitations".into() },
        KeyPoint { text: "k".into(), section: "Abstract".into() },
      ],
      mindmap: MindNode { label: "r".into(), children: vec![leaf("a"), leaf("b")] },
    }
  }

  fn sections() -> Vec<String> {
    vec!["3".into(), "4.2".into(), "Limitations".into()]
  }

  const SOURCE: &str =
    "[§4.2] Results\nIt is $49\\times$ faster, scoring $82.8$ - $84.6\\%$, from 0.6439 to 0.6354.";

  #[test]
  fn good_digest_passes_and_latex_numbers_verify() {
    assert_eq!(check_digest(&digest(), &sections(), SOURCE), Ok(Checked::default()));
  }

  #[test]
  fn invented_number_is_marked_unverified_not_rejected() {
    let mut d = digest();
    d.results[0].numbers.push("97.7".into());
    let checked = check_digest(&d, &sections(), SOURCE).unwrap();
    assert_eq!(checked.unverified_numbers, ["97.7"]);
  }

  #[test]
  fn missing_section_is_rejected() {
    let mut d = digest();
    d.key_points[0].section = "9.9".into();
    let err = check_digest(&d, &sections(), SOURCE).unwrap_err();
    assert!(err[0].contains("\"9.9\""), "{err:?}");

    let mut d2 = digest();
    d2.key_points[0].section = "9.9 Fake Title".into();
    let err2 = check_digest(&d2, &sections(), SOURCE).unwrap_err();
    assert!(err2[0].contains("\"9.9 Fake Title\""), "{err2:?}");

    // 存在 "3" 时绝不能把不存在的子节 "3.1" 误匹配到 "3"
    let mut d3 = digest();
    d3.key_points[0].section = "3.1".into();
    assert!(check_digest(&d3, &sections(), SOURCE).is_err());
    d3.key_points[0].section = "3.1 Sub".into();
    assert!(check_digest(&d3, &sections(), SOURCE).is_err());
  }

  #[test]
  fn overqualified_section_citations_resolve_and_normalize() {
    let secs: Vec<String> = vec!["4".into(), "4.6".into(), "H".into(), "Limitations".into()];
    // 线上实证失败案例：模型把节号连同标题一起写出
    assert_eq!(resolve_section("4.6 Intra-Model Recurrence Depth Scaling", &secs), Some("4.6"));
    assert_eq!(resolve_section("H Appendix H FFT Hyperparameter Sweeps", &secs), Some("H"));
    assert_eq!(resolve_section("Section 4.6: Scaling", &secs), Some("4.6"));
    assert_eq!(resolve_section("Appendix H Extra", &secs), Some("H"));
    assert_eq!(resolve_section("§4.6.", &secs), Some("4.6"));
    // 不存在的节号依然拒绝
    assert_eq!(resolve_section("4.1", &secs), None);
    assert_eq!(resolve_section("4.1 Setup", &secs), None);
    assert_eq!(resolve_section("9.9 Fake Title", &secs), None);

    let mut d = digest();
    d.results[0].section = "4.6 Intra-Model Recurrence Depth Scaling".into();
    d.key_points[0].section = "H Appendix H FFT Hyperparameter Sweeps".into();
    assert!(check_digest(&d, &secs, SOURCE).is_ok());
    normalize_sections(&mut d, &secs);
    assert_eq!(d.results[0].section, "4.6");
    assert_eq!(d.key_points[0].section, "H");
  }

  #[test]
  fn deep_or_wide_mindmap_is_rejected() {
    let mut d = digest();
    let mut deep = leaf("5");
    for l in ["4", "3", "2", "1"] {
      deep = MindNode { label: l.into(), children: vec![deep] };
    }
    d.mindmap.children.push(deep);
    assert!(check_digest(&d, &sections(), SOURCE).unwrap_err()[0].contains("levels"));

    let mut d = digest();
    d.mindmap.children = (0..7).map(|i| leaf(&i.to_string())).collect();
    assert!(check_digest(&d, &sections(), SOURCE).unwrap_err()[0].contains("children"));
  }

  #[test]
  fn translation_must_keep_structure() {
    let en = digest();
    let mut zh = en.clone();
    zh.tldr = "一句话".into();
    zh.results[0].claim = "结论".into();
    assert_eq!(check_translation(&en, &zh), Ok(()));

    let mut bad = zh.clone();
    bad.results[0].numbers[0] = "49 倍".into();
    bad.mindmap.children.pop();
    let errs = check_translation(&en, &bad).unwrap_err();
    assert_eq!(errs.len(), 2);
    assert!(errs[0].contains("results[0].numbers"), "{errs:?}");
    assert!(errs[0].contains("49×"), "{errs:?}");
  }
}
