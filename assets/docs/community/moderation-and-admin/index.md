---
title: 内容安全审核流水线（URL 黑名单 + LLM）与 /admin 管理后台
sidebar_label: 内容审核与后台
sidebar_position: 4
description: 配置域名黑名单秒级拦截、大模型语义风控以及通过 /admin 在线热调优审核策略与人工复核队列。
keywords: [内容审核, URL 黑名单, LLM 风控, 管理后台, 垃圾链接拦截]
---

# 内容审核流水线与管理后台

公开开放评论与发帖的站点极易遭遇黑产垃圾外链投毒（甚至触发搜索引擎安全警告）。DioxusPress 内置了可分层编排的 `ModerationPipeline` 与可视化 `/admin` 管理后台。

## 两级审核流水线

1. **第一级：URL 域名黑名单（零延迟确定性拦截）**
   支持精确域名（`"scam.example"`）与通配子域名（`"*.phishing.example"`）。一旦用户提交的标题或正文包含命中黑名单的链接，直接返回拦截提示（`Block`），不消耗任何大模型 Token。
2. **第二级：LLM 语义风控（可选开启）**
   兼容 OpenAI 与 Anthropic 接口协议（通过 `.env` 配置 `OPENAI_LLM_API_KEY` / `OPENAI_LLM_BASE_URL` / `OPENAI_LLM_MODEL`）。根据风险评分自动分流：
   - `< flag_above`（默认 `0.5`）：直接公开展示；
   - `flag_above .. block_above`（`0.5 .. 0.9`）：转入 **人工复核队列**，待管理员批准后对外可见且自动补写搜索索引；
   - `>= block_above`（默认 `0.9`）：当场拒绝提交。

:::important
**故障安全（Fail-Safe）设计**：当开启了 LLM 审核但外部 API 超时或欠费时，`on_llm_failure` 仅允许配置为 `"review"`（进人工待审队列，默认）或 `"reject"`（直接拒收），从类型上消除了「把审核接口打挂即可绕过风控」的漏洞。
:::

## `/admin` 管理后台功能一览

管理员访问 `/admin` 可完成全部日常站点治理：
- **概览仪表盘**：查看用户总量、待审内容数、近期话题与评论统计；
- **审核队列（`/admin/moderation`）**：一键批准（Approve）或驳回（Reject）可疑提交；
- **审核策略设置（`/admin/moderation/settings`）**：网页端直接开关审核、调整阈值与域名黑名单，保存后立即写回 `site.json` 并热刷新内存缓存；
- **用户、评论、话题与课程权益管理**：角色升降级、违规内容清理与会员权益发放。
