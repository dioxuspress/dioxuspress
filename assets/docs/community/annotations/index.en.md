---
title: "Inline Paragraph Highlights & Personal Study Notes (Annotations)"
sidebar_label: Paragraph Annotations
sidebar_position: 3
description: Enable e-reader-grade multi-color text highlights, private study notes, and a personal annotations hub across courses, docs, and blogs.
keywords: [paragraph annotations, text highlights, study notes, AnnotationLayer]
---

# Inline Paragraph Highlights & Notes

While studying long-form source-code deep dives or structured courses, signed-in readers can select any passage of text to add multi-color highlights or private study notes.

## Per-Module Switches

Toggle the annotation layer independently for courses, docs, and blog posts in `site.json::annotations`:

```json
{
  "annotations": {
    "course": true,
    "doc": true,
    "blog": false
  }
}
```

## Anchoring & Personal Hub

- **Deterministic Re-Anchoring**: Selections bind to stable resource identifiers (such as `doc:guide/quickstart`) and paragraph character offsets, restoring highlights seamlessly after WASM hydration.
- **Centralized Review (`/me/annotations`)**: Readers can browse all their highlights and notes in reverse-chronological order from the avatar menu and jump back to the exact paragraph in one click.
