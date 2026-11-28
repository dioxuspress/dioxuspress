---
title: "Structured Courses Module: 3-Level Syllabi, Free Previews & Server-Side Entitlement Gates"
sidebar_label: Courses Module
sidebar_position: 4
description: Build multi-chapter courses with free preview lessons and strict server-enforced single-course or Pro membership entitlements.
keywords: [online courses, paywall security, Pro membership, course.yaml, DioxusPress]
---

# Structured Courses Module

When compiled with the `course` or default full feature set, DioxusPress provides a three-level **Course → Chapter → Lesson** learning experience backed by server-enforced access control.

## Directory Structure

```text
assets/courses/
└── rust-fullstack-in-action/
    ├── course.yaml             # Course metadata: title, price (cents/fen), level, access_tier
    ├── 01-foundations/
    │   ├── _chapter.yaml       # Chapter title, description, and order
    │   ├── 01-setup/
    │   │   └── index.mdx       # Free preview lesson (preview: true in Frontmatter)
    │   └── 02-ssr-hydration/
    │       └── index.mdx       # Protected lesson (preview: false — gated on the server)
    └── 02-production/
        └── ...
```

## Server-Side Entitlement Gate (Never Client-Only CSS)

Unlike static sites that hide paid text behind client-side CSS overlays, DioxusPress verifies user sessions and database entitlements (`user_course_entitlements` and active Pro memberships) inside Axum before reading lesson content:

- **Free Preview Lessons (`preview: true`)**: Readable by anonymous visitors and indexed in `/sitemap.xml` for organic search discovery;
- **Protected Lessons (`preview: false` or omitted)**: Unentitled requests receive only syllabus metadata and unlock prompts — **the raw Markdown and lesson attachments never leave the server**.

See the [Frontmatter Reference](/docs/content/frontmatter-reference) for all `course.yaml`, `_chapter.yaml`, and lesson fields. Administrators can grant access manually at `/admin/entitlements` (`course` tier) or automate fulfillment via WeChat Pay, Alipay, and Stripe (`payments` tier).
