use sea_orm::entity::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, DeriveEntityModel, Serialize, Deserialize)]
#[sea_orm(table_name = "model_audit_queue")]
pub struct Model {
  #[sea_orm(primary_key)]
  pub id: i32,
  #[sea_orm(unique)]
  pub repo_id: String,
  pub source: String,
  pub queue_status: String,
  pub verification_status: String,
  #[sea_orm(column_type = "Text")]
  pub headline: String,
  #[sea_orm(column_type = "Text")]
  pub evidence_json: String,
  #[sea_orm(column_type = "Text")]
  pub modeling_files_json: String,
  pub discovered_at: DateTimeWithTimeZone,
  pub updated_at: DateTimeWithTimeZone,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {}

impl ActiveModelBehavior for ActiveModel {}
