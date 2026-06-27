# UptimeRobot Setup (Layer 1 — External Monitoring)

External monitoring catches server-down and port-unreachable failures before clients notice.

## 1. Create a free account

1. Go to [https://uptimerobot.com](https://uptimerobot.com) and sign up (free tier supports up to 50 monitors).
2. Confirm your email address.

## 2. Add monitors for each client API

Repeat **Add New Monitor** for all three clients:

| Field | clinical-notes | medical-coding | patient-support |
|-------|----------------|----------------|-----------------|
| Monitor type | HTTP(s) | HTTP(s) | HTTP(s) |
| Friendly name | clinical-notes API | medical-coding API | patient-support API |
| URL | `http://YOUR_SERVER_IP:8001/health` | `http://YOUR_SERVER_IP:8002/health` | `http://YOUR_SERVER_IP:8003/health` |
| Monitoring interval | 1 minute (paid) or 5 minutes (free) | same | same |

Replace `YOUR_SERVER_IP` with your Vast.ai public IP.

## 3. Configure alert contacts

1. Open **My Settings → Alert Contacts**.
2. Add:
   - **Email** — your primary inbox
   - **SMS** — your mobile number (if available on your plan)
   - **Telegram** (optional) — instant mobile push

Assign all contacts to each monitor.

## 4. Enable keyword monitoring

For each HTTP monitor:

1. Open monitor **Advanced Settings**.
2. Enable **Keyword monitoring**.
3. Set keyword to: `ok`
4. Alert when keyword is **not** found.

This catches cases where the server returns HTTP 200 but the model is degraded or the JSON body is wrong.

## 5. Optional status page for clients

1. Go to **Public Status Pages**.
2. Create a page listing all three APIs.
3. Share the URL with clients for transparency.

## 6. Verify

From your laptop (outside the server):

```bash
curl http://YOUR_SERVER_IP:8001/health
```

You should see JSON with `"status": "ok"`. UptimeRobot should show **Up** within one check interval.

## Free plan limits

- 50 monitors (more than enough for 3 APIs)
- 5-minute check interval on free tier (upgrade for 1-minute checks)
- Email alerts included
- Public status page included

## Complement internal monitoring

| Layer | Tool | Catches |
|-------|------|---------|
| External | UptimeRobot | Server down, firewall, port blocked |
| Internal | `scripts/run_monitor.py` | GPU OOM, queue backup, latency trends |
| System | Netdata | CPU, RAM, disk, GPU charts |

Use all three together for complete visibility.
