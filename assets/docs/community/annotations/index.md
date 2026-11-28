---
title: 段落划线高亮与笔记标注系统（Annotations）
sidebar_label: 段落划线标注
sidebar_position: 3
description: 为课程、文档与博客开启电子书级的多色文字划线高亮、私密学习笔记与个人标注中心。
keywords: [段落标注, 划线笔记, 电子书高亮, AnnotationLayer]
---

# 段落划线与笔记标注

在深度阅读长篇源码解析文档或学习实战课程时，登录用户可以直接选中正文中的任意文字片段，添加多色高亮划线或记录个人思考笔记。

## 按资源类型精细启停

可在 `site.json::annotations` 中分别控制课程、文档与博客是否开启划线标注层：

```json
{
  "annotations": {
    "course": true,
    "doc": true,
    "blog": false
  }
}
```

## 工作机制与个人中心

- **跨刷新精确定位**：前端脚本通过稳定的资源前缀（如 `doc:guide/quickstart`）与段落内文本偏移锚定选区，水合后自动恢复用户在该页留下的全部高亮标记。
- **集中回顾管理**：用户可通过右上角头像菜单进入 **「我的标注（`/me/annotations`）」**，按时间倒序检索自己在全站各课程与文档中积累的重点摘录，并支持一键跳回原文上下文。
