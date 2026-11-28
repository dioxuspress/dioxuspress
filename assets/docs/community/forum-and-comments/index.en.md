---
title: "Forum Topics & Post Comments: Community Plaza and Contextual Page Q&A"
sidebar_label: Forum & Comments
sidebar_position: 2
description: Combine lightweight blog comments (module-comments) with structured, searchable forum discussions (module-forum) anchored directly to documentation pages.
keywords: [forum module, post comments, DiscussionPanel, untrusted Markdown sanitization, Tantivy search]
---

# Forum Topics & Post Comments

Enabling the `accounts` tier unlocks both lightweight **Post Comments (`module-comments`)** and structured **Community Forum Topics (`module-forum`)**.

## Choosing Between Comments and Forum Topics

| Dimension | Post Comments (`CommentBox`) | Community Forum (`Topics` + `DiscussionPanel`) |
| :--- | :--- | :--- |
| **Placement** | Footer of blog posts | Dedicated `/topics` plaza + contextual footer on docs and showcase pages |
| **Structure** | Flat chronological replies on a single post | Dedicated title, category tag, multi-reply threads, permanent `/topics/:id` URL |
| **Search Indexing** | Scoped to the host post | Approved public topics are incrementally indexed into `⌘K` Tantivy search |

## Starting Contextual Discussions from Docs

Clicking **Start Discussion** at the bottom of any doc chapter opens `/topics/new?kind=doc&path=...`. Once published, the thread appears both in `/topics` and directly beneath that documentation page so future readers find answers in context.

## Strict Untrusted UGC Isolation

Every comment and forum post renders through `widgets::parse_untrusted_markdown`:
- Supports syntax-highlighted code blocks, lists, emphasis, and safe links;
- Escapes all raw HTML tags and restricts images strictly to `/uploads/`;
- Demotes `#` headings to `##` to preserve host page heading hierarchy.
