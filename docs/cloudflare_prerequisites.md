# Cloudflare Tunnel Prerequisites

Complete these steps **before** running `scripts/setup_https.sh` on your Vast.ai server.

## Step 1: Buy a domain ($10–12/year)

- Register at [Namecheap](https://www.namecheap.com/) or [Cloudflare Registrar](https://www.cloudflare.com/products/registrar/)
- Example: `yourdomain.com`
- Cheaper alternative: `.xyz` domains (~$1/year)

## Step 2: Add domain to Cloudflare (free)

1. Sign up at [cloudflare.com](https://cloudflare.com)
2. **Add site** → enter your domain
3. Cloudflare provides two nameservers, for example:
   - `ns1.cloudflare.com`
   - `ns2.cloudflare.com`
4. At your domain registrar, replace the existing nameservers with Cloudflare's
5. Wait 5–30 minutes for DNS propagation

## Step 3: Get a Cloudflare API token

1. Cloudflare dashboard → **My Profile** → **API Tokens**
2. **Create Token** → use template **Edit zone DNS**
3. **Zone Resources**: include your domain
4. Copy the token and save it in `.env`:

```bash
CF_API_TOKEN=your-cloudflare-token
DOMAIN=yourdomain.com
```

The setup script uses interactive `cloudflared tunnel login` for tunnel credentials. The API token is optional but useful for automation and DNS management outside the tunnel CLI.

## Step 4: Verify domain is active on Cloudflare

1. Open the Cloudflare dashboard
2. Your domain should show status **Active**
3. Confirm SSL/TLS mode is **Full** or **Full (strict)** (tunnel terminates TLS at Cloudflare edge)

## Step 5: Set root `.env` variables

Add to your workspace `.env` before running setup:

```bash
DOMAIN=yourdomain.com
TUNNEL_NAME=llm-api-tunnel
CF_API_TOKEN=your-cloudflare-token
```

`TUNNEL_ID` is written automatically by `scripts/setup_https.sh` after the tunnel is created.

## What you get

After setup, each client API is reachable over HTTPS:

| Client           | URL                                      | Local port |
|------------------|------------------------------------------|------------|
| clinical-notes   | `https://clinical-notes.yourdomain.com`  | 8001       |
| medical-coding   | `https://medical-coding.yourdomain.com`  | 8002       |
| patient-support  | `https://patient-support.yourdomain.com` | 8003       |

Traffic path: **Client (HTTPS) → Cloudflare Edge → cloudflared tunnel → localhost HTTP**

No inbound ports need to be open on Vast.ai. The tunnel works even when the instance IP changes.
