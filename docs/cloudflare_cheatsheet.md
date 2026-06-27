# Cloudflare Tunnel Cheatsheet

Quick reference for managing HTTPS on your Vast.ai LLM serving stack.

## Service status

```bash
sudo systemctl status cloudflared
```

## Live logs

```bash
sudo journalctl -u cloudflared -f
```

## Restart / stop

```bash
sudo systemctl restart cloudflared
sudo systemctl stop cloudflared
```

## Tunnel inventory

```bash
cloudflared tunnel list
cloudflared tunnel info llm-api-tunnel
cloudflared tunnel route ip show
```

## Validate config before restart

```bash
cloudflared tunnel --config ~/.cloudflared/config.yml ingress validate
```

Always validate after editing `~/.cloudflared/config.yml` manually.

## Test HTTPS endpoints

```bash
curl -I https://clinical-notes.yourdomain.com/health
```

Expected: `HTTP/2 200` (or `HTTP/1.1 200`) and a `cf-ray:` response header proving Cloudflare is in the path.

## Verify SSL certificate

```bash
openssl s_client -connect clinical-notes.yourdomain.com:443 \
  -servername clinical-notes.yourdomain.com </dev/null
```

Look for a valid certificate chain issued for your subdomain.

## Add a new client subdomain

```bash
./scripts/add_client_https.sh new-client 8004 yourdomain.com
```

Then start the new PM2 app on that port and share the client API guide.

## One-time setup

```bash
# Set DOMAIN in .env first
./scripts/setup_https.sh
```

Safe to re-run — skips steps that are already complete.

## Vast.ai IP changed?

**Nothing to do.** The tunnel connects outbound from your server. Cloudflare DNS points to the tunnel, not your Vast.ai IP.

## Cost estimate

| Item              | Cost              |
|-------------------|-------------------|
| Cloudflare Tunnel | Free              |
| Cloudflare WAF (basic rules) | Free on Free plan |
| Domain            | ~$10/year         |
| **Total HTTPS**   | **~$1/month**     |

## Troubleshooting

| Symptom | Check |
|---------|-------|
| 502 Bad Gateway | `pm2 status` — is the local API running on the mapped port? |
| 404 from Cloudflare | Ingress hostname mismatch in `~/.cloudflared/config.yml` |
| Tunnel not connecting | `sudo journalctl -u cloudflared -n 50` |
| DNS not resolving | Cloudflare dashboard → DNS records; wait for propagation |
| Cert errors | SSL/TLS mode should be **Full**; tunnel handles origin |
