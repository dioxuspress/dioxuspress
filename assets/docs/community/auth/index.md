---
title: OAuth2 + PKCE 登录配置：接入 GitHub、Google、Discord 与 X (Twitter)
sidebar_label: OAuth2 登录配置
sidebar_position: 1
description: 配置环境变量与 site.json::auth 开启第三方一键登录、JWT 会话续期与首个管理员账号提权。
keywords: [OAuth2 PKCE, GitHub 登录, Google 登录, JWT 会话, 管理员提权]
---

# OAuth2 登录与会话配置

DioxusPress 原生支持 **GitHub**、**Google**、**Discord** 与 **X (Twitter)** 四种主流第三方 OAuth2（含 PKCE 扩展）登录方式，无需维护明文密码库或发信验证码服务。

## 1. 在 `site.json` 声明显示顺序

```json
{
  "auth": {
    "enabled": true,
    "providers": ["github", "google", "discord", "twitter"]
  }
}
```

如果某个 Provider 尚未申请密钥，无需修改代码，只要不在 `.env` 中填写它的 Client ID，或从 `providers` 数组中移除即可。

## 2. 在 `.env` 注入敏感凭据

以 GitHub OAuth App 为例，回调地址填写 `<BASE_URL>/api/auth/callback/github`：

```dotenv
BASE_URL=https://your-domain.com
JWT_SECRET=<使用 openssl rand -hex 32 生成的高强度随机密钥>

GITHUB_CLIENT_ID=Iv1.xxxxxxxxxxxxxxxx
GITHUB_CLIENT_SECRET=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Google、Discord、Twitter 同理分别对应 `GOOGLE_CLIENT_*`、`DISCORD_CLIENT_*` 与 `TWITTER_CLIENT_*`（详见仓库根目录 `.env.example`）。

## 3. 提权首个管理员（Admin）

所有首次通过 OAuth 登录的新用户默认角色为普通成员（`member`）。登录一次后，在服务器或本地终端执行内置提权脚本：

```bash
set -a; source .env; set +a
./scripts/promote_admin.sh <你的用户昵称>
```

随后在网页端点击「退出登录」并重新登录一次，新签发的会话令牌即携带 `admin` 权限，顶栏用户菜单将出现 **🛡️ 管理后台（`/admin`）** 入口。
