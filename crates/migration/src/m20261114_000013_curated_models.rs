//! 模型架构图鉴（`module-ai`）在线收录表 `curated_models`。
//!
//! 用于持久化管理员通过 `owner/model` 在线解析并收录的模型元数据与 `ModelBlueprint` JSON 快照，
//! 运行时与只读挂载的 `<CONTENT_DIR>/models/<slug>/` 静态目录自动合并（同 `slug` 时数据库记录优先覆盖）。

use sea_orm_migration::prelude::*;

pub struct Migration;

impl MigrationName for Migration {
  fn name(&self) -> &str {
    "m20261114_000013_curated_models"
  }
}

#[derive(DeriveIden)]
enum CuratedModels {
  Table,
  Slug,
  Name,
  Vendor,
  Modality,
  Series,
  ReleaseDate,
  HfRepo,
  MsRepo,
  Featured,
  SortOrder,
  Summary,
  SummaryEn,
  TagsJson,
  HighlightsJson,
  HighlightsEnJson,
  ReadmeMd,
  BlueprintJson,
  CreatedBy,
  CreatedAt,
  UpdatedAt,
}

#[async_trait::async_trait]
impl MigrationTrait for Migration {
  async fn up(&self, manager: &SchemaManager) -> Result<(), DbErr> {
    manager
      .create_table(
        Table::create()
          .table(CuratedModels::Table)
          .if_not_exists()
          .col(ColumnDef::new(CuratedModels::Slug).string_len(96).not_null().primary_key())
          .col(ColumnDef::new(CuratedModels::Name).text().not_null())
          .col(ColumnDef::new(CuratedModels::Vendor).string_len(128).not_null())
          .col(ColumnDef::new(CuratedModels::Modality).string_len(32).not_null().default("llm"))
          .col(ColumnDef::new(CuratedModels::Series).string_len(64).not_null().default("other"))
          .col(ColumnDef::new(CuratedModels::ReleaseDate).string_len(32).not_null().default(""))
          .col(ColumnDef::new(CuratedModels::HfRepo).string_len(192).not_null())
          .col(ColumnDef::new(CuratedModels::MsRepo).string_len(192))
          .col(ColumnDef::new(CuratedModels::Featured).boolean().not_null().default(false))
          .col(ColumnDef::new(CuratedModels::SortOrder).integer().not_null().default(100))
          .col(ColumnDef::new(CuratedModels::Summary).text().not_null())
          .col(ColumnDef::new(CuratedModels::SummaryEn).text().not_null().default(""))
          .col(ColumnDef::new(CuratedModels::TagsJson).text().not_null().default("[]"))
          .col(ColumnDef::new(CuratedModels::HighlightsJson).text().not_null().default("[]"))
          .col(ColumnDef::new(CuratedModels::HighlightsEnJson).text().not_null().default("[]"))
          .col(ColumnDef::new(CuratedModels::ReadmeMd).text())
          .col(ColumnDef::new(CuratedModels::BlueprintJson).text().not_null())
          .col(ColumnDef::new(CuratedModels::CreatedBy).integer())
          .col(ColumnDef::new(CuratedModels::CreatedAt).timestamp_with_time_zone().not_null())
          .col(ColumnDef::new(CuratedModels::UpdatedAt).timestamp_with_time_zone().not_null())
          .to_owned(),
      )
      .await?;

    manager
      .create_index(
        Index::create()
          .if_not_exists()
          .name("idx_curated_models_sort_order")
          .table(CuratedModels::Table)
          .col(CuratedModels::SortOrder)
          .col(CuratedModels::Slug)
          .to_owned(),
      )
      .await
  }

  async fn down(&self, manager: &SchemaManager) -> Result<(), DbErr> {
    manager.drop_table(Table::drop().table(CuratedModels::Table).to_owned()).await
  }
}
