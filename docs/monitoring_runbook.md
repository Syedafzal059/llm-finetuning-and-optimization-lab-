# Monitoring Runbook

What to do when an alert fires. Keep this open on your phone bookmarks.

## Morning routine (30 seconds)

```bash
pm2 status
curl -H "X-API-Key: $ADMIN_KEY" http://localhost:8001/admin/health | python -m json.tool
tail -5 logs/backup.log
```

## Test alerts without waiting

```bash
python scripts/run_monitor.py --test
```

## Silence an alert temporarily (30-minute cooldown reset)

```bash
redis-cli DEL alert_sent:gpu_memory:critical
redis-cli DEL alert_sent:clinical-notes_api:critical
```

Replace the key with `alert_sent:{check_name}:{status}` from the alert email.

## See current system status

```bash
curl -H "X-API-Key: your-admin-key" http://localhost:8001/admin/health | python -m json.tool
```

---

## Alert: API unreachable / HTTP error

**Source:** UptimeRobot or `{project}_api` critical

**Symptoms:** Connection refused, timeout, or non-200 from `/health`

**Actions:**

1. `pm2 status` — is the process online?
2. `pm2 logs {project} --lines 50` — startup or CUDA OOM?
3. `pm2 restart {project}` if crashed
4. If restart loop: `pm2 stop {project}` and investigate GPU memory

**Escalate if:** More than 5 restarts or model won't load after 2 minutes.

---

## Alert: API degraded

**Source:** `{project}_api` warning — health returns `degraded`

**Symptoms:** Model not loaded, worker stopped

**Actions:**

1. `curl http://localhost:{port}/health` — check `worker_status`
2. `pm2 logs {project}` — model load errors
3. Verify adapter path exists under `projects/{project}/models/`
4. Restart: `pm2 restart {project}`

---

## Alert: Slow response / latency degrading

**Source:** `{project}_api` warning or `{project}_latency` warning

**Symptoms:** `/health` > 5s or recent latency 2× hourly average

**Actions:**

1. Check queue depth: `redis-cli LLEN queue:{project}`
2. Check GPU: `nvidia-smi`
3. If queue > 10: client burst — normal if temporary
4. If GPU at 95%+: restart least-critical client or reduce concurrent load

---

## Alert: Queue backed up

**Source:** `{project}_queue` warning/critical

**Thresholds:** warning ≥ 10, critical ≥ 20 (configurable via `.env`)

**Actions:**

1. `redis-cli LLEN queue:{project}` — confirm depth
2. `pm2 logs {project}` — worker processing or stuck?
3. If worker running but queue growing: GPU saturated — notify client of delay
4. If worker stopped: `pm2 restart {project}`

---

## Alert: GPU memory critical

**Source:** `gpu_memory` critical (internal monitor or Netdata)

**Thresholds:** warning ≥ 85%, critical ≥ 95%

**Actions:**

1. `nvidia-smi` — confirm usage
2. Identify which PM2 process holds VRAM: `pm2 status`
3. Short term: stop one non-critical API temporarily
4. Long term: reduce `gpu_memory_utilization` in project config or upgrade GPU

---

## Alert: Redis down

**Source:** `redis` critical

**Actions:**

1. `systemctl status redis` or `redis-cli ping`
2. `sudo systemctl restart redis`
3. Restart all APIs: `pm2 restart all`
4. Queued jobs during outage are lost — notify clients if outage > 1 minute

---

## Alert: Disk almost full

**Source:** `disk_space` warning/critical

**Actions:**

1. `df -h /`
2. Clear old logs: `find projects/*/logs -name '*.log' -mtime +30 -delete`
3. Prune old backups per retention policy
4. Remove unused model checkpoints

---

## Alert: PM2 process not online

**Source:** `pm2_{name}` critical

**Actions:**

1. `pm2 status`
2. `pm2 logs {name} --err --lines 100`
3. `pm2 restart {name}`
4. If crash loop: check `projects/{name}/logs/pm2_err.log`

---

## Alert: PM2 restart storm

**Source:** `pm2_{name}` warning — restarts > 5

**Actions:**

1. Same as crash alert from `scripts/crash_alert.py`
2. Do not ignore — indicates unstable GPU or config issue
3. SSH in and fix root cause before clients hit rate limits on retries

---

## Alert: Backup overdue

**Source:** `backup` warning/critical

**Thresholds:** warning > 26h, critical > 48h since last "Backup complete"

**Actions:**

1. `tail -50 logs/backup.log`
2. Run manually: `bash scripts/backup.sh`
3. Verify cron: `crontab -l | grep backup`
4. Check rclone remotes if cloud upload failed

---

## Alert: Daily health summary

**Source:** Email at 8:00 (configurable via `MONITOR_DAILY_SUMMARY_HOUR`)

**Actions:**

1. Scan critical and warning counts
2. If all OK: no action needed
3. If warnings persist for 24h+: investigate during business hours

---

## Useful commands

```bash
# Full check output (no email)
python scripts/run_monitor.py --test

# Monitor log
tail -f logs/monitor.log

# All queue depths
redis-cli LLEN queue:clinical-notes
redis-cli LLEN queue:medical-coding
redis-cli LLEN queue:patient-support

# Netdata dashboard
# http://YOUR_SERVER_IP:19999

# Re-install cron (backup + monitor)
bash scripts/setup_cron.sh
```
