---
title: "Showcase Gallery & Podcast Channel Configuration Guide"
sidebar_label: Showcase & Podcast
sidebar_position: 5
description: Configure the curated project showcase (/case) and built-in audio podcast channel (/podcast) in zero-database mode.
keywords: [project showcase, tech podcast, web audio player, PodcastCard]
---

# Showcase & Podcast Modules

Even in the zero-database `basic` tier, DioxusPress includes two rich media modules out of the box: **Project Showcase (`cases`)** and **Podcast (`podcast`)**.

## Project Showcase (`assets/cases/`)

Ideal for "Built with..." galleries, ecosystem directories, or customer case studies:

```text
assets/cases/
└── dioxus-fullstack-template/
    ├── case.yaml               # Name, summary, category, repo, website, stars, favorite flag
    └── README.md               # Full Markdown write-up
```

- Setting `"submit_case_url": "https://github.com/<org>/<repo>/issues/new?template=add-case.yml"` in `site.json` displays a **Submit Showcase** button on `/case`; leaving it `null` hides the button.

## Podcast Channel (`assets/podcasts/`)

Maintain per-episode metadata in `assets/podcasts/<slug>/episode.yaml` (`id`, `title`, `duration`, `date`, `audio_url`, `guest`, `tags`) paired with `assets/audio/` or external object-storage URLs to generate `/podcast` with an interactive audio scrubber — and embed any episode inline inside docs or blog posts via `<PodcastCard id="1" />`. See [Frontmatter Reference](/docs/content/frontmatter-reference) for full field schemas.
