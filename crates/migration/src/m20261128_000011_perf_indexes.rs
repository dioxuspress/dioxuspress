//! Phase 11.4：高频查询性能索引迁移。
//!
//! 1. `orders(provider, provider_txn)`：Stripe 退款/拒付回调（`stripe_reversal`）与对账查单
//!    直接走索引扫描，避免随订单量增长退化为全表扫描。
//! 2. `moderation_queue(user_id, status)`：加速按用户与状态筛选审核队列。

use sea_orm_migration::prelude::*;

pub struct Migration;

impl MigrationName for Migration {
  fn name(&self) -> &str {
    "m20261128_000011_perf_indexes"
  }
}

#[async_trait::async_trait]
impl MigrationTrait for Migration {
  async fn up(&self, manager: &SchemaManager) -> Result<(), DbErr> {
    manager
      .create_index(
        Index::create()
          .name("idx_orders_provider_txn")
          .table(Orders::Table)
          .col(Orders::Provider)
          .col(Orders::ProviderTxn)
          .if_not_exists()
          .to_owned(),
      )
      .await?;

    manager
      .create_index(
        Index::create()
          .name("idx_moderation_queue_user_status")
          .table(ModerationQueue::Table)
          .col(ModerationQueue::UserId)
          .col(ModerationQueue::Status)
          .if_not_exists()
          .to_owned(),
      )
      .await?;

    Ok(())
  }

  async fn down(&self, manager: &SchemaManager) -> Result<(), DbErr> {
    manager
      .drop_index(
        Index::drop()
          .name("idx_moderation_queue_user_status")
          .table(ModerationQueue::Table)
          .to_owned(),
      )
      .await?;
    manager
      .drop_index(Index::drop().name("idx_orders_provider_txn").table(Orders::Table).to_owned())
      .await?;
    Ok(())
  }
}

#[derive(DeriveIden)]
enum Orders {
  Table,
  Provider,
  ProviderTxn,
}

#[derive(DeriveIden)]
enum ModerationQueue {
  Table,
  UserId,
  Status,
}
