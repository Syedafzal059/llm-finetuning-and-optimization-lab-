# Cloudflare verification checklist

Post-setup checks after running `scripts/setup_https.sh` on Ubuntu 24.

Replace `YOURDOMAIN.com` with the value of `DOMAIN` in your root `.env`.

## 1. Tunnel is running

```bash
systemctl status cloudflared
```

Should show: `active (running)`.

## 2. Logs look clean

```bash
journalctl -u cloudflared -n 50
```

Should show: `registered tunnel connection` (or similar successful connection messages).

## 3. HTTPS works

```bash
curl -I "https://clinical-notes.YOURDOMAIN.com/health"
```

Should show:

- `HTTP/2 200`
- `cf-ray: ...` — proves Cloudflare is routing the request

Repeat for `medical-coding` and `patient-support` subdomains.

## 4. SSL cert is valid

```bash
openssl s_client -connect clinical-notes.YOURDOMAIN.com:443 \
  -servername clinical-notes.YOURDOMAIN.com </dev/null 2>/dev/null \
  | openssl x509 -noout -subject -dates
```

Should show a valid certificate for your subdomain with future `notAfter` date.

Full verify (expect `Verify return code: 0 (ok)`):

```bash
echo | openssl s_client -connect clinical-notes.YOURDOMAIN.com:443 \
  -servername clinical-notes.YOURDOMAIN.com 2>&1 \
  | grep "Verify return code"
```

## 5. HTTP redirects to HTTPS

```bash
curl -I "http://clinical-notes.YOURDOMAIN.com/health"
```

Should show a `301` or `302` redirect to `https://`.

## 6. Local HTTP ports still work on the server

```bash
curl http://localhost:8001/health
curl http://localhost:8002/health
curl http://localhost:8003/health
```

Each should return `200` when the corresponding PM2 app is running.

## 7. Raw HTTP ports are not reachable externally

```bash
curl --max-time 5 "http://YOUR_SERVER_IP:8001/health"
```

Should timeout or refuse connection when the Vast.ai (or host) firewall blocks inbound ports 8001–8003. Only Cloudflare tunnel HTTPS should be public.

## 8. PM2 apps are up before testing HTTPS

```bash
pm2 status
```

All three clients should be `online` before expecting `/health` to return 200 through the tunnel.

## 9. Tunnel config matches ports

```bash
grep -A2 hostname ~/.cloudflared/config.yml
```

Expected mapping:

| Subdomain | Local port |
|-----------|------------|
| clinical-notes | 8001 |
| medical-coding | 8002 |
| patient-support | 8003 |

## 10. Adding a new client later

```bash
./scripts/add_https_client.sh new-client 8004
curl "https://new-client.YOURDOMAIN.com/health"
```

See `scripts/add_https_client.sh` for onboarding additional projects.
