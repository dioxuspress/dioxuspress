use sea_orm::entity::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, DeriveEntityModel, Serialize, Deserialize)]
#[sea_orm(table_name = "curated_models")]
pub struct Model {
  #[sea_orm(primary_key, auto_increment = false)]
  pub slug: String,
  #[sea_orm(column_type = "Text")]
  pub name: String,
  pub vendor: String,
  pub modality: String,
  pub series: String,
  pub release_date: String,
  pub hf_repo: String,
  pub ms_repo: Option<String>,
  pub featured: bool,
  pub sort_order: i32,
  #[sea_orm(column_type = "Text")]
  pub summary: String,
  #[sea_orm(column_type = "Text")]
  pub summary_en: String,
  #[sea_orm(column_type = "Text")]
  pub tags_json: String,
  #[sea_orm(column_type = "Text")]
  pub highlights_json: String,
  #[sea_orm(column_type = "Text")]
  pub highlights_en_json: String,
  #[sea_orm(column_type = "Text", nullable)]
  pub readme_md: Option<String>,
  #[sea_orm(column_type = "Text")]
  pub blueprint_json: String,
  pub created_by: Option<i32>,
  pub created_at: DateTimeWithTimeZone,
  pub updated_at: DateTimeWithTimeZone,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {}

impl ActiveModelBehavior for ActiveModel {}
