# PM2 Cheatsheet

Quick reference for managing multi-client LLM serving on Vast.ai.

## One-time setup

```bash
chmod +x scripts/setup_pm2.sh scripts/manage.sh
./scripts/setup_pm2.sh
# Run the sudo pm2 startup command printed at the end
```

## See all clients

```bash
pm2 status
# or
./scripts/manage.sh status
```

## Live dashboard

```bash
pm2 monit
# or
./scripts/manage.sh monitor
```

## Restart one client

```bash
pm2 restart clinical-notes
# or
./scripts/manage.sh restart clinical-notes
```

## Restart all

```bash
pm2 restart all
# or
./scripts/manage.sh restart-all
```

## See logs live

```bash
pm2 logs clinical-notes
```

## See last 100 lines

```bash
pm2 logs clinical-notes --lines 100
# or
./scripts/manage.sh logs clinical-notes
```

## Stop one client

```bash
pm2 stop clinical-notes
# or
./scripts/manage.sh stop clinical-notes
```

## Add new client

1. Add an app block to `ecosystem.config.cjs` (copy an existing entry, change name/port).
2. Start only that client:

```bash
pm2 start ecosystem.config.js --only new-client-name
pm2 save
```

Or use the helper:

```bash
./scripts/manage.sh add new-client 8004
```

## Check all APIs responding

```bash
./scripts/manage.sh health
```

## Clear stuck queue

```bash
redis-cli DEL queue:clinical-notes
```

## Crash logs

Per-client PM2 logs are kept under:

```
projects/<client>/logs/pm2_out.log
projects/<client>/logs/pm2_err.log
```

Log rotation is handled by `pm2-logrotate` (installed by `setup_pm2.sh`).

## Environment

Set paths and alert credentials in the repo root `.env` (see `.env.example`):

- `WORKSPACE` — absolute path to this repo on the server
- `ALERT_FROM_EMAIL`, `ALERT_TO_EMAIL`, `ALERT_EMAIL_PASSWORD` — crash alert SMTP

## Graceful shutdown

When PM2 sends SIGTERM (restart/stop), each API:

1. Returns `503` with `server_restarting` for new requests
2. Finishes in-flight `/generate` requests (up to 60s)
3. Drains the Redis inference worker queue when enabled
4. Shuts down cleanly before PM2 restarts the process
