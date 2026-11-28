---
title: "Online Payments & TipJar Integration: WeChat Pay v3, Alipay, Stripe & Reconciliation"
sidebar_label: Payments & TipJar
sidebar_position: 5
description: Configure WeChat Pay v3, Alipay, and Stripe webhooks in the default payments tier for automated course fulfillment and reader tips.
keywords: [WeChat Pay, Alipay, Stripe checkout, course payments, webhook verification, TipJar]
---

# Online Payments & TipJar Integration

The default full build (`payments` Cargo feature) includes pure-Rust cryptographic payment gateways for WeChat Pay v3, Alipay, and Stripe — with no external billing microservice required.

## End-to-End Fulfillment Guarantees

1. **Idempotent Order State Machine**: Orders transition strictly forward (`Pending -> Paid -> Fulfilled`);
2. **Asymmetric Webhook Verification**: Every `/api/pay/*/notify` callback verifies provider signatures and exact order amounts on the server;
3. **Transactional Entitlement Granting**: Verified callbacks mark the order paid and insert the user's course or Pro membership entitlement inside a single PostgreSQL transaction;
4. **Reader TipJar (`TipJar`)**: Blog posts and podcast episodes display a lightweight support card when compiled with `payments`, and strip all payment routes and UI cleanly in `basic`, `accounts`, and `course` builds.
