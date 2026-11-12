use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::engines::module::ModuleSettings;
use crate::i18n::Language;

/// 默认站点名（未配置或为空时的回退）。
pub const DEFAULT_SITE_NAME: &str = "DioxusPress";
/// 默认布局名（Phase 3.3）。
pub const DEFAULT_LAYOUT: &str = "classic";
/// 默认主题 id。
pub const DEFAULT_THEME: &str = "ocean";
/// 默认站点图标路径（对应 `assets/images/favicon.ico`）。
pub const DEFAULT_FAVICON: &str = "/images/favicon.ico";
/// 默认社交分享图路径（对应 `assets/images/og-default.jpg`）。
pub const DEFAULT_OG_IMAGE: &str = "/images/og-default.jpg";
/// 默认工信部备案查询链接。
pub const DEFAULT_ICP_URL: &str = "https://beian.miit.gov.cn/";

fn default_site_name() -> String {
  DEFAULT_SITE_NAME.to_string()
}

fn default_layout() -> String {
  DEFAULT_LAYOUT.to_string()
}

fn default_theme() -> String {
  DEFAULT_THEME.to_string()
}

fn default_language() -> String {
  "zh".to_string()
}

fn default_favicon() -> String {
  DEFAULT_FAVICON.to_string()
}

fn default_og_image() -> String {
  DEFAULT_OG_IMAGE.to_string()
}

fn default_true() -> bool {
  true
}

fn non_empty_str(opt: Option<&str>) -> Option<&str> {
  opt.map(str::trim).filter(|s| !s.is_empty())
}

/// 官方 `site.json` JSON Schema 在线地址（用于 IDE 自动补全与字段校验）。
pub const SITE_SCHEMA_URL: &str = "https://dioxuspress.dev/schemas/site.schema.json";

/// DioxusPress 站点级非敏感配置文件（`assets/site.json`）根结构。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(test, derive(schemars::JsonSchema))]
pub struct SiteConfig {
  /// 站点名称（用于浏览器标题后缀、顶栏纯文字站名与页脚默认版权主体；留空回退 `"DioxusPress"`）。
  #[serde(default = "default_site_name")]
  pub site_name: String,
  /// 站点一句话描述（注入首页默认 `<meta name="description">` 与 RSS `<description>`）。
  #[serde(default)]
  pub site_description: String,
  /// 默认主题 id（见 [`crate::engines::theme::THEMES`]）；访客可用 cookie 覆盖。
  #[serde(default = "default_theme")]
  pub theme: String,
  /// 当前生效布局：`"classic"`（默认门户）、`"docs"`（类 Docusaurus 文档站）或 `"minimal"`（极简阅读条）。
  #[serde(default = "default_layout")]
  pub active_layout: String,
  /// 默认界面语言：`"zh"`（默认）或 `"en"`。无 `?lang=` 与 `site_lang` cookie 时生效。
  #[serde(default = "default_language")]
  pub default_language: String,
  /// 站点默认作者署名（如 `"yuxuetr"`）。
  #[serde(default)]
  pub author: String,
  /// 站点图标路径（如 `"/images/favicon.ico"` 或 `"/images/favicon.svg"`）。
  #[serde(default = "default_favicon")]
  pub favicon: String,
  /// 16×16 PNG 浏览器图标路径（可选，如 `"/images/favicon-16.png"`）。
  #[serde(default)]
  pub favicon_16: Option<String>,
  /// 32×32 PNG 浏览器图标路径（可选，如 `"/images/favicon-32.png"`）。
  #[serde(default)]
  pub favicon_32: Option<String>,
  /// iOS 主屏图标路径（可选，如 `"/images/apple-touch-icon.png"`）。
  #[serde(default)]
  pub apple_touch_icon: Option<String>,
  /// Web App Manifest 路径（可选，如 `"/images/site.webmanifest"`）。
  #[serde(default)]
  pub manifest: Option<String>,
  /// 顶栏浅色/默认 Logo 图片路径（可选，如 `"/images/logo-light.png"`；留空则仅显示文字站名）。
  #[serde(default)]
  pub logo: Option<String>,
  /// 顶栏深色模式 Logo 图片路径（可选，如 `"/images/logo-dark.png"`；未配置时复用 `logo`）。
  #[serde(default)]
  pub logo_dark: Option<String>,
  /// 紧凑顶栏 / 窄屏独立品牌图标路径（可选，如 `"/images/logo-mobile.png"`）。
  #[serde(default)]
  pub logo_mobile: Option<String>,
  /// `logo` / `logo_dark` 是否已内含站名文字（为 `true` 时横版 Logo 旁不再重复渲染文字站名）。
  #[serde(default)]
  pub logo_wordmark: bool,
  /// 默认 Open Graph / Twitter 分享图路径（默认 `"/images/og-default.jpg"`）。
  #[serde(default = "default_og_image")]
  pub og_image: String,
  /// 可选 X / Twitter 账号（如 `"@yuxuetr"`），非空时注入 `<meta name="twitter:site">`。
  #[serde(default)]
  pub twitter_handle: Option<String>,
  /// 案例页「提交案例」按钮外链（留空则隐藏提交按钮）。
  #[serde(default)]
  pub submit_case_url: Option<String>,
  /// 文档页底部「在 GitHub 上编辑此页」链接前缀（如 `"https://github.com/org/repo/edit/main/assets/docs"`）。
  #[serde(default)]
  pub docs_edit_url_base: Option<String>,
  /// 顶部可关闭公告条配置（留空时不展示）。
  #[serde(default)]
  pub announcement: Option<AnnouncementSettings>,
  /// 首页 Hero 区域自定义文案（留空时回退到 `assets/i18n/{zh,en}.ftl` 的 `hero.*`）。
  #[serde(default)]
  pub hero: HeroSettings,
  /// 开源项目首页（`active_layout = "docs"`）特性展示卡片列表。
  #[serde(default)]
  pub features: Vec<FeatureItem>,
  /// 页脚口号、版权、ICP 备案号与外部社交链接。
  #[serde(default)]
  pub footer: FooterSettings,
  /// 段落标注（Annotation）按资源类型的启用开关。
  #[serde(default)]
  pub annotations: AnnotationsSettings,
  /// 传统导航项配置（兼容旧版布局；当前主导航主要由 `modules` 控制）。
  #[serde(default)]
  pub navigation: Vec<NavItem>,
  /// OAuth 第三方授权登录配置（需 `accounts` 特性档位与对应环境变量凭据）。
  #[serde(default)]
  pub auth: AuthSettings,
  /// 模块开关：key = ModuleSpec.id，value = `{ enabled: true|false }`。
  /// `ModuleEngine::init` 读取该字段覆盖默认 enabled 状态。
  #[serde(default)]
  pub modules: HashMap<String, ModuleSettings>,
  /// Phase 4.3：审核配置（独立于通用 `modules` 开关，因为还有 LLM 开关、
  /// 阈值与链接黑名单）。
  /// 默认 disabled → 整条流水线为空，提交直接通过，不调 LLM。
  /// 由 `crates/modules/moderation::ModerationPipeline::from_site_config`
  /// 读取并装载。
  #[serde(default)]
  pub moderation: ModerationSettings,
}

/// 顶部可关闭公告条配置（常用于版本发布或活动提示）。
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
#[cfg_attr(test, derive(schemars::JsonSchema))]
pub struct AnnouncementSettings {
  /// 公告唯一标识（访客关闭后以 `dp_ann_{id}` 写入 `localStorage`；变更 `id` 会重新展示）。
  #[serde(default)]
  pub id: String,
  /// 公告正文（中文或单语默认文本；为空时不渲染公告条）。
  #[serde(default)]
  pub text: String,
  /// 公告正文英文覆盖（可选；未配置时回退到 `text`）。
  #[serde(default)]
  pub text_en: Option<String>,
  /// 点击公告条跳转链接（支持站内路由如 `"/docs/guide"` 或外部 URL）。
  #[serde(default)]
  pub href: Option<String>,
}

impl AnnouncementSettings {
  /// 按语言取非空公告文案（英文未配置时回退中文）。
  pub fn text_for(&self, lang: Language) -> Option<&str> {
    match lang {
      Language::En => {
        non_empty_str(self.text_en.as_deref()).or_else(|| non_empty_str(Some(&self.text)))
      }
      Language::Zh => non_empty_str(Some(&self.text)),
    }
  }

  /// 可选跳转链接。
  pub fn href_or_none(&self) -> Option<&str> {
    non_empty_str(self.href.as_deref())
  }

  /// `localStorage` 持久化键名。
  pub fn storage_key(&self) -> String {
    let suffix = non_empty_str(Some(&self.id)).unwrap_or("default");
    format!("dp_ann_{suffix}")
  }
}

/// 首页 Hero 区域自定义文案（留空时回退到 `i18n` 的 `hero.title` / `hero.subtitle`）。
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
#[cfg_attr(test, derive(schemars::JsonSchema))]
pub struct HeroSettings {
  /// 首页主标题上方的胶囊徽章文案（中文）。
  #[serde(default)]
  pub badge: Option<String>,
  /// 首页主标题上方的胶囊徽章文案（英文覆盖）。
  #[serde(default)]
  pub badge_en: Option<String>,
  /// 首页 Hero 主标题（中文）。
  #[serde(default)]
  pub title: Option<String>,
  /// 首页 Hero 主标题（英文覆盖）。
  #[serde(default)]
  pub title_en: Option<String>,
  /// 首页 Hero 副标题描述（中文）。
  #[serde(default)]
  pub subtitle: Option<String>,
  /// 首页 Hero 副标题描述（英文覆盖）。
  #[serde(default)]
  pub subtitle_en: Option<String>,
  /// 开源项目首页一键复制终端命令（如 `"git clone ..."` 或 `"cargo add ..."`）。
  #[serde(default)]
  pub install_command: Option<String>,
  /// 主行动按钮文案（中文）。
  #[serde(default)]
  pub primary_cta_text: Option<String>,
  /// 主行动按钮文案（英文覆盖）。
  #[serde(default)]
  pub primary_cta_text_en: Option<String>,
  /// 主行动按钮跳转路径或 URL（留空时不展示主按钮）。
  #[serde(default)]
  pub primary_cta_href: Option<String>,
  /// 次行动按钮文案（中文）。
  #[serde(default)]
  pub secondary_cta_text: Option<String>,
  /// 次行动按钮文案（英文覆盖）。
  #[serde(default)]
  pub secondary_cta_text_en: Option<String>,
  /// 次行动按钮跳转路径或 URL（留空时不展示次按钮）。
  #[serde(default)]
  pub secondary_cta_href: Option<String>,
}

/// 开源项目首页特性卡片（`active_layout = "docs"` 时渲染为 3 列栅格）。
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
#[cfg_attr(test, derive(schemars::JsonSchema))]
pub struct FeatureItem {
  /// 图标标识（支持内置 `"bolt"` / `"book"` / `"layers"` / `"search"` / `"shield"` / `"palette"` 或短字符）。
  #[serde(default)]
  pub icon: String,
  /// 特性标题（中文或默认语言）。
  pub title: String,
  /// 特性标题（英文覆盖）。
  #[serde(default)]
  pub title_en: Option<String>,
  /// 特性详细说明（中文或默认语言）。
  #[serde(default)]
  pub description: String,
  /// 特性详细说明（英文覆盖）。
  #[serde(default)]
  pub description_en: Option<String>,
  /// 点击卡片跳转的目标路由或外链（可选）。
  #[serde(default)]
  pub href: Option<String>,
}

impl FeatureItem {
  pub fn title_for(&self, lang: Language) -> &str {
    match lang {
      Language::En => non_empty_str(self.title_en.as_deref()).unwrap_or(&self.title),
      Language::Zh => &self.title,
    }
  }

  pub fn description_for(&self, lang: Language) -> &str {
    match lang {
      Language::En => non_empty_str(self.description_en.as_deref()).unwrap_or(&self.description),
      Language::Zh => &self.description,
    }
  }

  pub fn href_or_none(&self) -> Option<&str> {
    non_empty_str(self.href.as_deref())
  }
}

/// 页脚自定义信息（口号、版权、备案号、社交链接、多列分组）。
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
#[cfg_attr(test, derive(schemars::JsonSchema))]
pub struct FooterSettings {
  /// 页脚口号（中文）；未配置时回退到 `i18n` 的 `footer.tagline`。
  #[serde(default)]
  pub tagline: Option<String>,
  /// 页脚口号（英文）；未配置时回退到 `i18n` 的 `footer.tagline`。
  #[serde(default)]
  pub tagline_en: Option<String>,
  /// 版权声明文本（如 `"2024–2026 Rust in Everything"`）；未配置时回退到 `site_name`。
  #[serde(default)]
  pub copyright: Option<String>,
  /// ICP 备案号（如 `"京ICP备xxxxxxxx号"`）；为空时不渲染备案链接。
  #[serde(default)]
  pub icp: Option<String>,
  /// 备案跳转链接（默认 `"https://beian.miit.gov.cn/"`）。
  #[serde(default)]
  pub icp_url: Option<String>,
  /// 页脚「社区」栏展示的外部社交链接（GitHub / X / Discord 等）。
  #[serde(default)]
  pub social: Vec<SocialLink>,
  /// 类 Docusaurus 多列页脚链接分组（非空时在 `DocsShell` 中优先渲染）。
  #[serde(default)]
  pub columns: Vec<FooterColumn>,
}

/// 多列结构化页脚单列配置。
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
#[cfg_attr(test, derive(schemars::JsonSchema))]
pub struct FooterColumn {
  /// 列分组标题（中文或默认语言）。
  pub title: String,
  /// 列分组标题（英文覆盖）。
  #[serde(default)]
  pub title_en: Option<String>,
  /// 该列包含的导航或外部链接列表。
  #[serde(default)]
  pub links: Vec<FooterLink>,
}

impl FooterColumn {
  pub fn title_for(&self, lang: Language) -> &str {
    match lang {
      Language::En => non_empty_str(self.title_en.as_deref()).unwrap_or(&self.title),
      Language::Zh => &self.title,
    }
  }
}

/// 多列页脚中的单条内外链。
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
#[cfg_attr(test, derive(schemars::JsonSchema))]
pub struct FooterLink {
  /// 链接显示名称（中文或默认语言）。
  pub label: String,
  /// 链接显示名称（英文覆盖）。
  #[serde(default)]
  pub label_en: Option<String>,
  /// 跳转目标（以 `"/"` 开头走 SPA 站内路由，以 `"http"` 开头作为新标签外链打开）。
  pub href: String,
}

impl FooterLink {
  pub fn label_for(&self, lang: Language) -> &str {
    match lang {
      Language::En => non_empty_str(self.label_en.as_deref()).unwrap_or(&self.label),
      Language::Zh => &self.label,
    }
  }
}

/// 页脚外部社交或社区链接。
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
#[cfg_attr(test, derive(schemars::JsonSchema))]
pub struct SocialLink {
  /// 平台名称（如 `"GitHub"`、`"Discord"`；包含 `"GitHub"` 时会自动提取为文档顶栏仓库图标链接）。
  pub label: String,
  /// 完整外部链接地址。
  pub url: String,
}

/// 段落标注（Annotation）按资源类型的启用开关（需 `accounts` 特性档位）。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(test, derive(schemars::JsonSchema))]
pub struct AnnotationsSettings {
  /// 是否在课程章节页开启段落划线标注（默认 `true`）。
  #[serde(default = "default_true")]
  pub course: bool,
  /// 是否在文档详情页开启段落划线标注（默认 `true`）。
  #[serde(default = "default_true")]
  pub doc: bool,
  /// 是否在博客文章页开启段落划线标注（默认 `false`）。
  #[serde(default)]
  pub blog: bool,
}

impl Default for AnnotationsSettings {
  fn default() -> Self {
    Self { course: true, doc: true, blog: false }
  }
}

impl AnnotationsSettings {
  /// 判断指定资源类型是否启用段落标注。
  pub fn is_enabled(&self, kind: &str) -> bool {
    match kind {
      "course" => self.course,
      "doc" => self.doc,
      "blog" => self.blog,
      _ => false,
    }
  }
}

/// 审核功能在 site.json 中的配置块。**默认 disabled**，意味着评论 / 话题 /
/// 标注的提交路径不会调用任何 LLM。
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[cfg_attr(test, derive(schemars::JsonSchema))]
pub struct ModerationSettings {
  /// 总开关。默认 `false`。
  #[serde(default)]
  pub enabled: bool,
  /// 是否让 LLM 审核（内置 stage，见 `module_moderation::LlmModerationStage`）。
  /// 还需配置 LLM 环境变量（`OPENAI_LLM_*` / `ANTHROPIC_LLM_*`），未配置时跳过。
  #[serde(default)]
  pub llm_review: bool,
  /// LLM 调用失败 / 回复无法解析时怎么处理这条提交。默认送人工复核。
  #[serde(default)]
  pub on_llm_failure: LlmFailureAction,
  /// 可选：覆盖默认阈值（block_above = 0.9 / flag_above = 0.5）。
  /// 留空表示用默认。
  #[serde(default)]
  pub thresholds: Option<ModerationThresholdsConfig>,
  /// URL 域名黑名单。命中即 Block（score=1.0），不走 LLM。
  /// 匹配规则（不区分大小写）：
  /// - 精确：`"scam.com"` 只匹配 host = `scam.com`
  /// - 通配：`"*.phishing.example"` 匹配 `sub.phishing.example` 与
  ///   `phishing.example` 本身
  ///
  /// 默认空 → 该 stage 不注册，零开销。
  #[serde(default)]
  pub url_blocklist: Vec<String>,
}

/// LLM 审核失败时的处理（SEC-12：不再放行）。没有「放行」选项：
/// 开了 LLM 审核却在审核失败时放行，等于攻击者把审核打挂就能绕过。
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq)]
#[cfg_attr(test, derive(schemars::JsonSchema))]
#[serde(rename_all = "lowercase")]
pub enum LlmFailureAction {
  /// 记为 Flag：先不公开，进审核队列等管理员通过后显示。
  #[default]
  Review,
  /// 记为 Block：拒绝提交，提示用户稍后再试。
  Reject,
}

/// `ModerationSettings::thresholds` 的可选覆盖。字段缺失时不影响其它字段。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(test, derive(schemars::JsonSchema))]
pub struct ModerationThresholdsConfig {
  /// 风险分高于该阈值直接拦截（范围 `0.0..=1.0`，默认 `0.9`）。
  pub block_above: Option<f32>,
  /// 风险分高于该阈值进入人工审核队列（范围 `0.0..=1.0`，默认 `0.5`）。
  pub flag_above: Option<f32>,
}

/// 自定义导航栏条目。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(test, derive(schemars::JsonSchema))]
pub struct NavItem {
  /// 国际化词条 key 或字面显示文本。
  pub key: String,
  /// 点击跳转路由路径（如 `"/docs"`）。
  pub route: String,
}

/// OAuth 第三方授权登录配置（需 `accounts` 特性档位与对应 `*_CLIENT_ID` / `*_CLIENT_SECRET` 环境变量）。
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[cfg_attr(test, derive(schemars::JsonSchema))]
pub struct AuthSettings {
  /// 是否开启用户登录入口。
  #[serde(default)]
  pub enabled: bool,
  /// 启用的内置 provider id（`"github"` / `"google"` / `"discord"` / `"twitter"`），按登录弹窗显示顺序排列。
  #[serde(default)]
  pub providers: Vec<String>,
}

impl SiteConfig {
  /// 从路径读取并解析 site.json（无缓存，每次磁盘 IO）。
  ///
  /// 适用于写后立即重读等必须绕过缓存的场景（admin 保存配置）；
  /// 读多写少的热路径请用 [`SiteConfig::load_cached`]。
  ///
  /// 返回统一的 [`crate::error::AppResult`]，底层 IO 失败会变为
  /// `AppError::Io`，不合法 JSON 会变为 `AppError::Validation`。
  pub fn from_file(path: &str) -> crate::error::AppResult<Self> {
    let content = std::fs::read_to_string(path)?;
    let config: SiteConfig = serde_json::from_str(&content)?;
    Ok(config)
  }

  /// S10（风险 R11）：按 (path, mtime) 缓存的统一读取入口。
  ///
  /// 此前多处 server fn（主题 CSS / 布局 / feed / auth）每次请求都
  /// `from_file` 直读磁盘 + serde 解析。改用本入口后：
  /// - mtime 未变 → 直接 clone Arc（一次 metadata 系统调用，无读盘/解析）
  /// - mtime 变化（admin 保存 / 手改文件）→ 自动重读，无需显式失效
  ///
  /// 缓存 key 为路径字符串；实际部署只有一个 site.json，表容量恒为 1。
  pub fn load_cached(path: &str) -> crate::error::AppResult<std::sync::Arc<Self>> {
    use std::collections::HashMap as Map;
    use std::sync::{Arc, Mutex, OnceLock};
    use std::time::SystemTime;

    struct Entry {
      mtime: SystemTime,
      cfg: Arc<SiteConfig>,
    }
    static CACHE: OnceLock<Mutex<Map<String, Entry>>> = OnceLock::new();

    let mtime = std::fs::metadata(path)?.modified()?;
    let cache = CACHE.get_or_init(|| Mutex::new(Map::new()));

    if let Ok(guard) = cache.lock() {
      if let Some(entry) = guard.get(path) {
        if entry.mtime == mtime {
          return Ok(entry.cfg.clone());
        }
      }
    }

    // 未命中 / mtime 变化 → 锁外重读，写回缓存（锁失败仅跳过本次缓存）。
    let cfg = Arc::new(Self::from_file(path)?);
    if let Ok(mut guard) = cache.lock() {
      guard.insert(path.to_string(), Entry { mtime, cfg: cfg.clone() });
    }
    Ok(cfg)
  }

  /// 编译期内嵌的 `assets/site.json` 配置（SSR 与 WASM 双端一致、零 IO）。
  /// 解析失败时回退到 [`SiteConfig::default`]。
  pub fn embedded() -> &'static Self {
    use std::sync::OnceLock;
    static EMBEDDED: OnceLock<SiteConfig> = OnceLock::new();
    EMBEDDED.get_or_init(|| {
      serde_json::from_str::<SiteConfig>(include_str!("../../../assets/site.json"))
        .unwrap_or_default()
    })
  }

  /// 站点名称（为空或全空白时回退到 [`DEFAULT_SITE_NAME`]）。
  pub fn site_name_or_default(&self) -> &str {
    non_empty_str(Some(self.site_name.as_str())).unwrap_or(DEFAULT_SITE_NAME)
  }

  /// 默认界面语言枚举（仅 `"en"` 识别为英文，其余回退中文）。
  pub fn default_language_enum(&self) -> Language {
    if self.default_language.trim().eq_ignore_ascii_case("en") {
      Language::En
    } else {
      Language::Zh
    }
  }

  /// 站点图标路径（为空时回退到 [`DEFAULT_FAVICON`]）。
  pub fn favicon_or_default(&self) -> &str {
    non_empty_str(Some(self.favicon.as_str())).unwrap_or(DEFAULT_FAVICON)
  }

  /// 16×16 PNG 图标路径（未配置或为空字符串时返回 `None`）。
  pub fn favicon_16_url(&self) -> Option<&str> {
    non_empty_str(self.favicon_16.as_deref())
  }

  /// 32×32 PNG 图标路径（未配置或为空字符串时返回 `None`）。
  pub fn favicon_32_url(&self) -> Option<&str> {
    non_empty_str(self.favicon_32.as_deref())
  }

  /// Apple Touch Icon 路径（未配置或为空字符串时返回 `None`）。
  pub fn apple_touch_icon_url(&self) -> Option<&str> {
    non_empty_str(self.apple_touch_icon.as_deref())
  }

  /// Web App Manifest 路径（未配置或为空字符串时返回 `None`）。
  pub fn manifest_url(&self) -> Option<&str> {
    non_empty_str(self.manifest.as_deref())
  }

  /// 顶栏浅色/默认 Logo 路径（未配置或为空字符串时返回 `None`）。
  pub fn logo_url(&self) -> Option<&str> {
    non_empty_str(self.logo.as_deref())
  }

  /// 顶栏深色模式 Logo 路径（未配置或为空字符串时返回 `None`）。
  pub fn logo_dark_url(&self) -> Option<&str> {
    non_empty_str(self.logo_dark.as_deref())
  }

  /// 紧凑顶栏 / 窄屏独立品牌图标路径（未配置或为空字符串时返回 `None`）。
  pub fn logo_mobile_url(&self) -> Option<&str> {
    non_empty_str(self.logo_mobile.as_deref())
  }

  /// 默认社交分享图路径（为空时回退到 [`DEFAULT_OG_IMAGE`]）。
  pub fn og_image_or_default(&self) -> &str {
    non_empty_str(Some(self.og_image.as_str())).unwrap_or(DEFAULT_OG_IMAGE)
  }

  /// 可选 Twitter / X 账号（未配置或为空字符串时返回 `None`）。
  pub fn twitter_handle_or_none(&self) -> Option<&str> {
    non_empty_str(self.twitter_handle.as_deref())
  }

  /// 案例页「提交案例」按钮链接（未配置或为空字符串时返回 `None`）。
  pub fn submit_case_url_or_none(&self) -> Option<&str> {
    non_empty_str(self.submit_case_url.as_deref())
  }

  /// 首页 Hero 主标题：优先取 `site.json::hero` 对应语言字段，留空回退 `i18n` 的 `hero.title`。
  pub fn hero_title(&self, lang: Language) -> String {
    let custom = match lang {
      Language::En => non_empty_str(self.hero.title_en.as_deref()),
      Language::Zh => non_empty_str(self.hero.title.as_deref()),
    };
    custom.map(str::to_string).unwrap_or_else(|| crate::i18n::t(lang, "hero.title"))
  }

  /// 首页 Hero 副标题：优先取 `site.json::hero` 对应语言字段，留空回退 `i18n` 的 `hero.subtitle`。
  pub fn hero_subtitle(&self, lang: Language) -> String {
    let custom = match lang {
      Language::En => non_empty_str(self.hero.subtitle_en.as_deref()),
      Language::Zh => non_empty_str(self.hero.subtitle.as_deref()),
    };
    custom.map(str::to_string).unwrap_or_else(|| crate::i18n::t(lang, "hero.subtitle"))
  }

  /// 页脚口号：优先取 `site.json::footer` 对应语言字段，留空回退 `i18n` 的 `footer.tagline`。
  pub fn footer_tagline(&self, lang: Language) -> String {
    let custom = match lang {
      Language::En => non_empty_str(self.footer.tagline_en.as_deref()),
      Language::Zh => non_empty_str(self.footer.tagline.as_deref()),
    };
    custom.map(str::to_string).unwrap_or_else(|| crate::i18n::t(lang, "footer.tagline"))
  }

  /// 页脚版权主体（未配置 `footer.copyright` 时回退到站名）。
  pub fn footer_copyright(&self) -> &str {
    non_empty_str(self.footer.copyright.as_deref()).unwrap_or_else(|| self.site_name_or_default())
  }

  /// 页脚 ICP 备案号及跳转链接（`footer.icp` 非空时返回 `Some((icp, url))`）。
  pub fn footer_icp(&self) -> Option<(&str, &str)> {
    let icp = non_empty_str(self.footer.icp.as_deref())?;
    let url = non_empty_str(self.footer.icp_url.as_deref()).unwrap_or(DEFAULT_ICP_URL);
    Some((icp, url))
  }

  /// 首页 Hero 顶部胶囊标：优先取对应语言，英文未配时回退中文。
  pub fn hero_badge(&self, lang: Language) -> Option<&str> {
    match lang {
      Language::En => non_empty_str(self.hero.badge_en.as_deref())
        .or_else(|| non_empty_str(self.hero.badge.as_deref())),
      Language::Zh => non_empty_str(self.hero.badge.as_deref()),
    }
  }

  /// 首页一键复制终端命令。
  pub fn hero_install_command(&self) -> Option<&str> {
    non_empty_str(self.hero.install_command.as_deref())
  }

  /// 首页主行动按钮 `(文案, 链接)`；链接为空时不展示。
  pub fn hero_primary_cta(&self, lang: Language) -> Option<(&str, &str)> {
    let href = non_empty_str(self.hero.primary_cta_href.as_deref())?;
    let text = match lang {
      Language::En => non_empty_str(self.hero.primary_cta_text_en.as_deref())
        .or_else(|| non_empty_str(self.hero.primary_cta_text.as_deref()))?,
      Language::Zh => non_empty_str(self.hero.primary_cta_text.as_deref())?,
    };
    Some((text, href))
  }

  /// 首页次行动按钮 `(文案, 链接)`；链接为空时不展示。
  pub fn hero_secondary_cta(&self, lang: Language) -> Option<(&str, &str)> {
    let href = non_empty_str(self.hero.secondary_cta_href.as_deref())?;
    let text = match lang {
      Language::En => non_empty_str(self.hero.secondary_cta_text_en.as_deref())
        .or_else(|| non_empty_str(self.hero.secondary_cta_text.as_deref()))?,
      Language::Zh => non_empty_str(self.hero.secondary_cta_text.as_deref())?,
    };
    Some((text, href))
  }

  /// 根据文档相对源文件路径（如 `"guide/intro/index.md"`）拼出完整编辑链接。
  pub fn docs_edit_url(&self, source_file: &str) -> Option<String> {
    let base = non_empty_str(self.docs_edit_url_base.as_deref())?.trim_end_matches('/');
    let rel = non_empty_str(Some(source_file))?.trim_start_matches('/');
    Some(format!("{base}/{rel}"))
  }

  /// 从 `footer.social` 中提取首个 GitHub 仓库/组织链接（供文档顶栏右上角图标使用）。
  pub fn github_repo_url(&self) -> Option<&str> {
    self
      .footer
      .social
      .iter()
      .find(|s| {
        let u = s.url.trim();
        !u.is_empty()
          && (s.label.trim().eq_ignore_ascii_case("github") || u.starts_with("https://github.com/"))
      })
      .map(|s| s.url.trim())
  }

  /// 当前布局名（空字符串回退到 `"classic"`）。
  pub fn active_layout_or_default(&self) -> &str {
    if self.active_layout.is_empty() {
      DEFAULT_LAYOUT
    } else {
      self.active_layout.as_str()
    }
  }
}

impl Default for SiteConfig {
  fn default() -> Self {
    Self {
      site_name: DEFAULT_SITE_NAME.to_string(),
      site_description: String::new(),
      theme: default_theme(),
      active_layout: DEFAULT_LAYOUT.to_string(),
      default_language: default_language(),
      author: String::new(),
      favicon: default_favicon(),
      favicon_16: None,
      favicon_32: None,
      apple_touch_icon: None,
      manifest: None,
      logo: None,
      logo_dark: None,
      logo_mobile: None,
      logo_wordmark: false,
      og_image: default_og_image(),
      twitter_handle: None,
      submit_case_url: None,
      docs_edit_url_base: None,
      announcement: None,
      hero: HeroSettings::default(),
      features: vec![],
      footer: FooterSettings::default(),
      annotations: AnnotationsSettings::default(),
      navigation: vec![],
      auth: AuthSettings::default(),
      modules: HashMap::new(),
      moderation: ModerationSettings::default(),
    }
  }
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)] // 测试 setup：Default + 逐字段赋值更易读
mod tests {
  use super::*;

  // ─── S10 load_cached ──────────────────────────────────

  fn minimal_site_json(name: &str) -> String {
    format!(
      r#"{{"site_name":"{}","site_description":"","default_language":"zh","author":"","navigation":[]}}"#,
      name
    )
  }

  /// 同 mtime 下重复读取应命中缓存（返回同一个 Arc）。
  #[test]
  fn load_cached_hits_on_same_mtime() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("site.json");
    std::fs::write(&path, minimal_site_json("A")).expect("write");
    let p = path.to_str().expect("utf8 path");

    let first = SiteConfig::load_cached(p).expect("first load");
    let second = SiteConfig::load_cached(p).expect("second load");
    assert!(std::sync::Arc::ptr_eq(&first, &second), "同 mtime 应返回同一 Arc");
    assert_eq!(first.site_name, "A");
  }

  /// mtime 变化后应自动重读新内容。
  #[test]
  fn load_cached_reloads_on_mtime_change() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("site.json");
    std::fs::write(&path, minimal_site_json("OLD")).expect("write old");
    let p = path.to_str().expect("utf8 path");
    let first = SiteConfig::load_cached(p).expect("first load");
    assert_eq!(first.site_name, "OLD");

    // 写新内容并显式把 mtime 向后拨 2 秒（避免文件系统 mtime 粒度引发 flaky）
    std::fs::write(&path, minimal_site_json("NEW")).expect("write new");
    let f = std::fs::OpenOptions::new().append(true).open(&path).expect("open");
    let later = std::time::SystemTime::now() + std::time::Duration::from_secs(2);
    f.set_modified(later).expect("set mtime");

    let second = SiteConfig::load_cached(p).expect("reload");
    assert_eq!(second.site_name, "NEW", "mtime 变化应重读");
  }

  #[test]
  fn load_cached_missing_file_errors() {
    assert!(SiteConfig::load_cached("/nonexistent/__site__.json").is_err());
  }

  #[test]
  fn active_layout_falls_back_to_classic() {
    let mut cfg = SiteConfig::default();
    cfg.active_layout = String::new();
    assert_eq!(cfg.active_layout_or_default(), "classic");
    cfg.active_layout = "minimal".to_string();
    assert_eq!(cfg.active_layout_or_default(), "minimal");
  }

  #[test]
  fn deserialize_missing_theme_and_layout_uses_defaults() {
    let json = r#"{
            "site_name": "X",
            "site_description": "",
            "default_language": "zh",
            "author": "",
            "navigation": []
        }"#;
    let cfg: SiteConfig = serde_json::from_str(json).expect("parse");
    assert_eq!(cfg.theme, "ocean");
    assert_eq!(cfg.active_layout, "classic");
  }

  // ─── Phase 4.3 moderation settings ───────────────────────

  #[test]
  fn moderation_defaults_to_disabled_empty() {
    let cfg = SiteConfig::default();
    assert!(!cfg.moderation.enabled);
    assert!(!cfg.moderation.llm_review);
    assert!(cfg.moderation.thresholds.is_none());
    assert!(cfg.moderation.url_blocklist.is_empty());
  }

  #[test]
  fn moderation_back_compat_when_field_missing() {
    // 老 site.json 完全没有 moderation block → 默认 disabled
    let json = r#"{
            "site_name": "X",
            "site_description": "",
            "default_language": "zh",
            "author": "",
            "navigation": []
        }"#;
    let cfg: SiteConfig = serde_json::from_str(json).expect("parse");
    assert!(!cfg.moderation.enabled);
  }

  /// 插件运行时移除前的 site.json 带 `paths` / `plugins_lock`，仍应能读。
  #[test]
  fn legacy_plugin_fields_are_ignored() {
    let json = r#"{
            "site_name": "X",
            "site_description": "",
            "default_language": "zh",
            "author": "",
            "paths": {"plugins": "assets/plugins"},
            "navigation": [],
            "plugins_lock": {"theme_ocean_plugin.wasm": "deadbeef"}
        }"#;
    let cfg: SiteConfig = serde_json::from_str(json).expect("parse");
    assert_eq!(cfg.site_name, "X");
  }

  #[test]
  fn moderation_parses_full_block() {
    let json = r#"{
            "site_name": "X",
            "site_description": "",
            "default_language": "zh",
            "author": "",
            "navigation": [],
            "moderation": {
                "enabled": true,
                "llm_review": true,
                "thresholds": { "block_above": 0.95, "flag_above": 0.6 }
            }
        }"#;
    let cfg: SiteConfig = serde_json::from_str(json).expect("parse");
    assert!(cfg.moderation.enabled);
    assert!(cfg.moderation.llm_review);
    let t = cfg.moderation.thresholds.unwrap();
    assert_eq!(t.block_above, Some(0.95));
    assert_eq!(t.flag_above, Some(0.6));
  }

  /// R4 之前的配置用 `plugins` 列 wasm 文件；插件已内置，旧字段被忽略，
  /// LLM 审核需显式 `llm_review: true` 才开启。
  #[test]
  fn legacy_moderation_plugins_list_does_not_enable_llm_review() {
    let json = r#"{"enabled": true, "plugins": ["plugin_moderation_deepseek.wasm"]}"#;
    let m: ModerationSettings = serde_json::from_str(json).expect("parse");
    assert!(m.enabled);
    assert!(!m.llm_review);
  }

  #[test]
  fn llm_failure_defaults_to_review() {
    let m: ModerationSettings = serde_json::from_str(r#"{"enabled": true}"#).expect("parse");
    assert_eq!(m.on_llm_failure, LlmFailureAction::Review);
    let m: ModerationSettings =
      serde_json::from_str(r#"{"on_llm_failure": "reject"}"#).expect("parse");
    assert_eq!(m.on_llm_failure, LlmFailureAction::Reject);
    assert!(serde_json::from_str::<ModerationSettings>(r#"{"on_llm_failure": "allow"}"#).is_err());
  }

  #[test]
  fn branding_hero_footer_and_annotations_helpers() {
    let json = r#"{
      "site_name": "MySite",
      "default_language": "en",
      "favicon": "/images/custom.svg",
      "favicon_16": " /images/favicon-16.png ",
      "favicon_32": "/images/favicon-32.png",
      "apple_touch_icon": "/images/apple-touch-icon.png",
      "manifest": "/images/site.webmanifest",
      "logo": " /images/logo.svg ",
      "logo_dark": " /images/logo-dark.svg ",
      "logo_mobile": "/images/logo-mobile.png",
      "logo_wordmark": true,
      "og_image": "/images/custom-og.png",
      "twitter_handle": "@mysite",
      "submit_case_url": "https://github.com/example/repo/issues/new",
      "hero": {
        "title": "自定义标题",
        "title_en": "Custom Title",
        "subtitle": "自定义副标题",
        "subtitle_en": "Custom Subtitle"
      },
      "footer": {
        "tagline": "自定义页脚口号",
        "tagline_en": "Custom Footer Tagline",
        "copyright": "2026 MySite",
        "icp": "京ICP备12345678号",
        "social": [{ "label": "GitHub", "url": "https://github.com/example" }]
      },
      "annotations": {
        "course": true,
        "doc": false,
        "blog": true
      }
    }"#;
    let cfg: SiteConfig = serde_json::from_str(json).expect("parse full branding");
    assert_eq!(cfg.site_name_or_default(), "MySite");
    assert_eq!(cfg.default_language_enum(), Language::En);
    assert_eq!(cfg.favicon_or_default(), "/images/custom.svg");
    assert_eq!(cfg.favicon_16_url(), Some("/images/favicon-16.png"));
    assert_eq!(cfg.favicon_32_url(), Some("/images/favicon-32.png"));
    assert_eq!(cfg.apple_touch_icon_url(), Some("/images/apple-touch-icon.png"));
    assert_eq!(cfg.manifest_url(), Some("/images/site.webmanifest"));
    assert_eq!(cfg.logo_url(), Some("/images/logo.svg"));
    assert_eq!(cfg.logo_dark_url(), Some("/images/logo-dark.svg"));
    assert_eq!(cfg.logo_mobile_url(), Some("/images/logo-mobile.png"));
    assert!(cfg.logo_wordmark);
    assert_eq!(cfg.og_image_or_default(), "/images/custom-og.png");
    assert_eq!(cfg.twitter_handle_or_none(), Some("@mysite"));
    assert_eq!(cfg.submit_case_url_or_none(), Some("https://github.com/example/repo/issues/new"));
    assert_eq!(cfg.hero_title(Language::Zh), "自定义标题");
    assert_eq!(cfg.hero_title(Language::En), "Custom Title");
    assert_eq!(cfg.hero_subtitle(Language::Zh), "自定义副标题");
    assert_eq!(cfg.hero_subtitle(Language::En), "Custom Subtitle");
    assert_eq!(cfg.footer_tagline(Language::Zh), "自定义页脚口号");
    assert_eq!(cfg.footer_tagline(Language::En), "Custom Footer Tagline");
    assert_eq!(cfg.footer_copyright(), "2026 MySite");
    assert_eq!(cfg.footer_icp(), Some(("京ICP备12345678号", DEFAULT_ICP_URL)));
    assert_eq!(cfg.footer.social.len(), 1);
    assert!(cfg.annotations.is_enabled("course"));
    assert!(!cfg.annotations.is_enabled("doc"));
    assert!(cfg.annotations.is_enabled("blog"));
    assert!(!cfg.annotations.is_enabled("other"));
  }

  #[test]
  fn empty_optional_branding_falls_back_cleanly() {
    let cfg: SiteConfig = serde_json::from_str("{}").expect("empty object uses defaults");
    assert_eq!(cfg.site_name_or_default(), DEFAULT_SITE_NAME);
    assert_eq!(cfg.default_language_enum(), Language::Zh);
    assert_eq!(cfg.favicon_or_default(), DEFAULT_FAVICON);
    assert_eq!(cfg.favicon_16_url(), None);
    assert_eq!(cfg.favicon_32_url(), None);
    assert_eq!(cfg.apple_touch_icon_url(), None);
    assert_eq!(cfg.manifest_url(), None);
    assert_eq!(cfg.logo_url(), None);
    assert_eq!(cfg.logo_dark_url(), None);
    assert_eq!(cfg.logo_mobile_url(), None);
    assert!(!cfg.logo_wordmark);
    assert_eq!(cfg.og_image_or_default(), DEFAULT_OG_IMAGE);
    assert_eq!(cfg.twitter_handle_or_none(), None);
    assert_eq!(cfg.submit_case_url_or_none(), None);
    assert_eq!(cfg.footer_copyright(), DEFAULT_SITE_NAME);
    assert_eq!(cfg.footer_icp(), None);
    assert!(cfg.annotations.is_enabled("course"));
    assert!(cfg.annotations.is_enabled("doc"));
    assert!(!cfg.annotations.is_enabled("blog"));
    assert_eq!(cfg.docs_edit_url("guide/index.md"), None);
    assert_eq!(cfg.github_repo_url(), None);
    assert!(!SiteConfig::embedded().site_name_or_default().is_empty());
  }

  #[test]
  fn docs_layout_settings_resolve_bilingual_and_urls() {
    let json = r#"{
      "active_layout": "docs",
      "docs_edit_url_base": "https://github.com/dioxuspress/dioxuspress/edit/main/assets/docs/",
      "announcement": {
        "id": "v07",
        "text": "DioxusPress 全新文档模板上线",
        "text_en": "New Docusaurus-style template available",
        "href": "/docs/guide"
      },
      "hero": {
        "badge": "Dioxus 0.7 · 纯 Rust",
        "badge_en": "Dioxus 0.7 · Pure Rust",
        "install_command": "git clone https://github.com/dioxuspress/dioxuspress.git",
        "primary_cta_text": "快速开始",
        "primary_cta_text_en": "Get Started",
        "primary_cta_href": "/docs/guide",
        "secondary_cta_text": "GitHub",
        "secondary_cta_href": "https://github.com/dioxuspress/dioxuspress"
      },
      "features": [
        {
          "icon": "bolt",
          "title": "零数据库静态档位",
          "title_en": "Zero-DB Basic Tier",
          "description": "一条命令启动",
          "description_en": "Run with a single binary"
        }
      ],
      "footer": {
        "social": [{ "label": "GitHub", "url": "https://github.com/dioxuspress/dioxuspress" }],
        "columns": [
          {
            "title": "文档",
            "title_en": "Docs",
            "links": [{ "label": "入门", "label_en": "Intro", "href": "/docs/guide" }]
          }
        ]
      }
    }"#;
    let cfg: SiteConfig = serde_json::from_str(json).expect("valid docs site.json");
    assert_eq!(cfg.active_layout_or_default(), "docs");
    assert_eq!(
      cfg.docs_edit_url("/guide/index.md").as_deref(),
      Some("https://github.com/dioxuspress/dioxuspress/edit/main/assets/docs/guide/index.md")
    );
    assert_eq!(cfg.github_repo_url(), Some("https://github.com/dioxuspress/dioxuspress"));
    assert_eq!(cfg.hero_badge(Language::Zh), Some("Dioxus 0.7 · 纯 Rust"));
    assert_eq!(cfg.hero_badge(Language::En), Some("Dioxus 0.7 · Pure Rust"));
    assert_eq!(cfg.hero_primary_cta(Language::En), Some(("Get Started", "/docs/guide")));
    assert_eq!(
      cfg.hero_secondary_cta(Language::En),
      Some(("GitHub", "https://github.com/dioxuspress/dioxuspress"))
    );
    let ann = cfg.announcement.as_ref().expect("announcement");
    assert_eq!(ann.storage_key(), "dp_ann_v07");
    assert_eq!(ann.text_for(Language::En), Some("New Docusaurus-style template available"));
    assert_eq!(cfg.features[0].title_for(Language::En), "Zero-DB Basic Tier");
    assert_eq!(cfg.footer.columns[0].title_for(Language::En), "Docs");
    assert_eq!(cfg.footer.columns[0].links[0].label_for(Language::En), "Intro");
  }

  /// 生成官方 `assets/schemas/site.schema.json`，补充 `$id`、`$schema` 自引用字段及动态枚举约束。
  fn build_site_schema_json() -> String {
    let mut root = serde_json::to_value(schemars::schema_for!(SiteConfig)).expect("schema to json");
    let obj = root.as_object_mut().expect("schema root object");
    obj.insert("$id".to_string(), serde_json::json!(SITE_SCHEMA_URL));

    if let Some(props) = obj.get_mut("properties").and_then(|v| v.as_object_mut()) {
      props.insert(
        "$schema".to_string(),
        serde_json::json!({
          "type": "string",
          "description": "JSON Schema URI（用于 VS Code / Zed 等编辑器的字段悬停说明与自动补全）。"
        }),
      );
      if let Some(layout_prop) = props.get_mut("active_layout").and_then(|v| v.as_object_mut()) {
        layout_prop.insert("enum".to_string(), serde_json::json!(["classic", "docs", "minimal"]));
      }
      if let Some(lang_prop) = props.get_mut("default_language").and_then(|v| v.as_object_mut()) {
        lang_prop.insert("enum".to_string(), serde_json::json!(["zh", "en"]));
      }
      if let Some(theme_prop) = props.get_mut("theme").and_then(|v| v.as_object_mut()) {
        let theme_ids: Vec<&str> = crate::engines::theme::THEMES.iter().map(|t| t.id).collect();
        theme_prop.insert("enum".to_string(), serde_json::json!(theme_ids));
      }
    }

    // 为 AuthSettings.providers 数组元素注入内置 OAuth Provider 枚举
    let provider_ids: Vec<&str> = crate::auth::Provider::ALL.into_iter().map(|p| p.id()).collect();
    for defs_key in ["$defs", "definitions"] {
      if let Some(items) = obj
        .get_mut(defs_key)
        .and_then(|d| d.get_mut("AuthSettings"))
        .and_then(|a| a.get_mut("properties"))
        .and_then(|p| p.get_mut("providers"))
        .and_then(|pr| pr.get_mut("items"))
        .and_then(|i| i.as_object_mut())
      {
        items.insert("enum".to_string(), serde_json::json!(provider_ids));
      }
    }

    let mut out = serde_json::to_string_pretty(&root).expect("pretty schema");
    out.push('\n');
    out
  }

  /// 守卫测试：保证 `assets/schemas/site.schema.json` 与 `SiteConfig` Rust 定义严格一致。
  /// 修改 `SiteConfig` 字段或注释后，运行 `UPDATE_SCHEMA=1 cargo test -p app-core` 自动更新。
  #[test]
  fn site_schema_json_is_up_to_date() {
    let schema_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
      .join("../../assets/schemas/site.schema.json");
    let generated = build_site_schema_json();

    if std::env::var("UPDATE_SCHEMA").as_deref() == Ok("1") {
      if let Some(parent) = schema_path.parent() {
        std::fs::create_dir_all(parent).expect("create schemas dir");
      }
      std::fs::write(&schema_path, &generated).expect("write site.schema.json");
    }

    let on_disk = std::fs::read_to_string(&schema_path).unwrap_or_else(|e| {
      panic!(
        "无法读取 {} ({e})；请运行 `UPDATE_SCHEMA=1 cargo test -p app-core` 生成",
        schema_path.display()
      )
    });
    assert_eq!(
      on_disk, generated,
      "assets/schemas/site.schema.json 与 SiteConfig 定义不一致，请运行 `UPDATE_SCHEMA=1 cargo test -p app-core` 同步"
    );

    // 确认带 "$schema" 的官方 assets/site.json 可正常解析
    let embedded = SiteConfig::embedded();
    assert_eq!(embedded.site_name_or_default(), "DioxusPress");
  }
}
