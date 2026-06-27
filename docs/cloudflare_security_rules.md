# Cloudflare Security Rules

Configure these in the Cloudflare dashboard for defense in depth on top of the tunnel and application-layer controls (API keys, rate limiting, sanitization).

## Rule 1: Block non-API traffic

**Security → WAF → Custom Rules → Create rule**

| Field   | Value |
|---------|-------|
| Name    | Block non-API paths |
| If      | URI Path does not contain `/generate` **AND** URI Path does not contain `/health` **AND** URI Path does not contain `/result` **AND** URI Path does not contain `/metrics` **AND** URI Path does not contain `/admin` |
| Then    | Block |

This reduces scanner noise and limits exposed surface to known API routes.

## Rule 2: Rate limit at Cloudflare level

**Security → WAF → Rate Limiting Rules → Create rule**

| Field   | Value |
|---------|-------|
| Name    | API rate limit |
| If      | URI Path contains `/generate` |
| Rate    | 100 requests per minute per IP |
| Action  | Block for 1 minute |

This complements Redis-backed rate limiting in the FastAPI layer — clients get double protection.

## Rule 3: Block bad bots

**Security → Bots**

- **Bot Fight Mode**: ON (free tier)
- Blocks many automated scanners and credential-stuffing bots

## Rule 4: Enforce HTTPS only

**SSL/TLS → Edge Certificates**

| Setting                    | Value |
|----------------------------|-------|
| Always Use HTTPS           | ON    |
| Minimum TLS Version        | TLS 1.2 |
| Opportunistic Encryption   | ON    |

Cloudflare terminates TLS; traffic between Cloudflare and your origin travels inside the encrypted tunnel.

## Rule 5: Enable HSTS

**SSL/TLS → Edge Certificates → HTTP Strict Transport Security**

| Setting   | Value     |
|-----------|-----------|
| Enable HSTS | ON      |
| Max Age   | 6 months  |

HSTS tells browsers to always use HTTPS for your domain and subdomains.

## HIPAA note

Cloudflare Tunnel encrypts traffic in transit and hides your origin IP. For HIPAA workloads you still need:

- Signed BAAs with Cloudflare (Enterprise) if required by your compliance review
- Application-level controls already in this repo (API keys, PII sanitization, audit logs)
- No PHI in URLs or logs

Consult your compliance officer before production PHI use.
