---
title: 1.1 同构 SSR 与客户端水合原理（免费试读）
description: 深入理解首帧一致性契约，消除水合闪烁并兼顾搜索引擎抓取。
sidebar_position: 1
duration: "14:20"
preview: true
---

# 1.1 同构 SSR 与客户端水合原理

本节 Frontmatter 设置了 `preview: true`，因此即使所属课程为付费专栏（`access_tier: paid`），所有访客均可直接免费试读。

## 首帧一致性黄金法则

在同构组件中，服务端与客户端首次渲染产出必须完全一致；任何依赖浏览器 `window` / `localStorage` 的读取都应放入 `use_effect` 钩子在水合完成后执行。

下一节（`1.2 生产容器安全加固与零停机运维`）标记为 `preview: false`，用于演示付费课节的权限拦截与解锁引导界面。
