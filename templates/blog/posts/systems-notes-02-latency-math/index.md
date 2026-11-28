---
title: "系统工程笔记（二）：用利特尔法则与排队论理解 P99 尾延迟"
description: "系列专栏第二篇：演示服务端零 JS 负担的原生 MathML 数学公式渲染与 Mermaid 延迟分解图。"
date: "2026-11-12"
authors: [author]
tags: [系统工程专栏, 性能优化, 数学公式]
---

# 系统工程笔记（二）：用利特尔法则与排队论理解 P99 尾延迟

承接上一篇 [零拷贝切片边界设计](/blog/systems-notes-01-zero-copy)，本文演示 DioxusPress 的**服务端原生 LaTeX → MathML 公式渲染**（无需在浏览器加载体积庞大的 KaTeX / MathJax 脚本）。

## 利特尔法则（Little's Law）与利用率拐点

在稳态排队系统中，系统内的平均在途请求数 $L$、平均到达率 $\lambda$ 与平均驻留时间 $W$ 满足经典的线性恒等式：$L = \lambda W$。

而在经典的 $M/M/1$ 队列模型下，当服务台利用率 $\rho = \frac{\lambda}{\mu}$ 趋近于 $1$ 时，平均响应时间呈非线性陡增：

$$
E[T] = \frac{1}{\mu - \lambda} = \frac{1}{\mu (1 - \rho)}
$$

## 请求全链路耗时分解

```mermaid
flowchart LR
  Client["客户端请求"] --> Queue["连接池排队: W_q"]
  Queue --> Exec["SSR 渲染服务: 1/μ"]
  Exec --> Resp["首字节直出 (TTFB)"]
```

## 持续追踪本专栏

- 通过 RSS 阅读器添加订阅地址：`http://localhost:8080/feed.xml`（线上部署后自动替换为你的正式域名 `/feed.xml`）
- 浏览 [关于作者](/about) 与 [数字花园书单](/docs/notes/reading-list)
