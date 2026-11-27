//! 模型架构核验状态与每日巡检待复核队列迁移：
//! 1. 为 `curated_models` 追加 `verification_status` / `verification_report_json` / `verified_at` / `upstream_fingerprint`；
//! 2. 新增 `model_audit_queue` 表，记录每日定时任务发现的新模型预审结果与存量回归告警证据包。

use sea_orm_migration::prelude::*;

pub struct Migration;

impl MigrationName for Migration {
  fn name(&self) -> &str {
    "m20261127_000014_model_verification_audit"
  }
}

#[derive(DeriveIden)]
enum CuratedModels {
  Table,
  VerificationStatus,
  VerificationReportJson,
  VerifiedAt,
  UpstreamFingerprint,
}

#[derive(DeriveIden)]
enum ModelAuditQueue {
  Table,
  Id,
  RepoId,
  Source,
  QueueStatus,
  VerificationStatus,
  Headline,
  EvidenceJson,
  ModelingFilesJson,
  DiscoveredAt,
  UpdatedAt,
}

#[async_trait::async_trait]
impl MigrationTrait for Migration {
  async fn up(&self, manager: &SchemaManager) -> Result<(), DbErr> {
    manager
      .alter_table(
        Table::alter()
          .table(CuratedModels::Table)
          .add_column_if_not_exists(
            ColumnDef::new(CuratedModels::VerificationStatus)
              .string_len(24)
              .not_null()
              .default("partial"),
          )
          .add_column_if_not_exists(ColumnDef::new(CuratedModels::VerificationReportJson).text())
          .add_column_if_not_exists(
            ColumnDef::new(CuratedModels::VerifiedAt).timestamp_with_time_zone(),
          )
          .add_column_if_not_exists(
            ColumnDef::new(CuratedModels::UpstreamFingerprint).string_len(64),
          )
          .to_owned(),
      )
      .await?;

    manager
      .create_table(
        Table::create()
          .table(ModelAuditQueue::Table)
          .if_not_exists()
          .col(
            ColumnDef::new(ModelAuditQueue::Id).integer().not_null().auto_increment().primary_key(),
          )
          .col(ColumnDef::new(ModelAuditQueue::RepoId).string_len(192).not_null().unique_key())
          .col(ColumnDef::new(ModelAuditQueue::Source).string_len(32).not_null().default("auto"))
          .col(
            ColumnDef::new(ModelAuditQueue::QueueStatus)
              .string_len(32)
              .not_null()
              .default("pending_review"),
          )
          .col(
            ColumnDef::new(ModelAuditQueue::VerificationStatus)
              .string_len(24)
              .not_null()
              .default("partial"),
          )
          .col(ColumnDef::new(ModelAuditQueue::Headline).text().not_null().default(""))
          .col(ColumnDef::new(ModelAuditQueue::EvidenceJson).text().not_null())
          .col(ColumnDef::new(ModelAuditQueue::ModelingFilesJson).text().not_null().default("[]"))
          .col(ColumnDef::new(ModelAuditQueue::DiscoveredAt).timestamp_with_time_zone().not_null())
          .col(ColumnDef::new(ModelAuditQueue::UpdatedAt).timestamp_with_time_zone().not_null())
          .to_owned(),
      )
      .await?;

    manager
      .create_index(
        Index::create()
          .if_not_exists()
          .name("idx_model_audit_queue_status_updated")
          .table(ModelAuditQueue::Table)
          .col(ModelAuditQueue::QueueStatus)
          .col(ModelAuditQueue::UpdatedAt)
          .to_owned(),
      )
      .await
  }

  async fn down(&self, manager: &SchemaManager) -> Result<(), DbErr> {
    manager.drop_table(Table::drop().table(ModelAuditQueue::Table).if_exists().to_owned()).await?;
    manager
      .alter_table(
        Table::alter()
          .table(CuratedModels::Table)
          .drop_column(CuratedModels::VerificationStatus)
          .drop_column(CuratedModels::VerificationReportJson)
          .drop_column(CuratedModels::VerifiedAt)
          .drop_column(CuratedModels::UpstreamFingerprint)
          .to_owned(),
      )
      .await
  }
}
