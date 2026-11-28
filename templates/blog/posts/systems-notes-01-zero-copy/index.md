---
title: "系统工程笔记（一）：零拷贝切片在网络协议解析中的边界设计"
description: "系列专栏第一篇：结合 Rust 借用检查与多语言代码对比标签组，剖析如何安全消除多余堆内存拷贝。"
date: "2026-11-10"
authors: [author]
tags: [系统工程专栏, Rust, 零拷贝]
---

# 系统工程笔记（一）：零拷贝切片在网络协议解析中的边界设计

本篇是 **系统工程笔记** 系列专栏的第一篇（下一篇见 [排队论与尾延迟数学推导](/blog/systems-notes-02-latency-math)）。在 `minimal` 布局与 `coffee` 羊皮纸主题下，长篇技术文章能够保持舒适的阅读行宽与柔和对比度。

## 裸指针切片 vs 生命周期借用切片

通过 `:::code-tabs` 同屏对照 C 与 Rust 的报文头切分方式：

:::code-tabs
```rust [Rust (&[u8])]
pub struct FrameHeader<'a> {
  pub topic: &'a str,
  pub payload: &'a [u8],
}

pub fn parse_frame(buf: &[u8]) -> Option<FrameHeader<'_>> {
  let sep = buf.iter().position(|&b| b == b'\n')?;
  let topic = std::str::from_utf8(&buf[..sep]).ok()?;
  Some(FrameHeader { topic, payload: &buf[sep + 1..] })
}
```
```c [C (const uint8_t *)]
struct frame_header {
  const char *topic;
  size_t topic_len;
  const uint8_t *payload;
  size_t payload_len;
};
```
:::

:::tip 订阅更新
你可以使用任何 RSS 阅读器订阅本站的 [Atom 订阅源（`/feed.xml`）](/feed.xml)，或翻阅我的 [数字花园长期笔记](/docs/notes/reading-list)。
:::
