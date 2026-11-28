// S9（风险 R12）：生产代码禁 unwrap/expect（workspace lints）；测试代码豁免。
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

fn main() {
  dioxuspress_app::run_default_app();
}
