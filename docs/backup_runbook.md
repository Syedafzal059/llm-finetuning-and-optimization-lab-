# Backup Disaster Recovery Runbook

This runbook covers setup, testing, restore, and troubleshooting for the automated
3-2-1 backup system (local archive + Google Drive + Backblaze B2).

## What Gets Backed Up

| Priority | Paths |
|----------|-------|
| Critical | `projects/*/model/lora/`, `projects/*/model/qlora/`, `projects/*/data/`, `usage.db`, `pii_audit.log`, `security.log`, `config.yaml` |
| Important | All serving logs, `prompt_template.py`, project `.env` files |
| Excluded | Base models, `venv/`, `__pycache__/`, `node_modules/` |

## One-Time Setup (Vast.ai Linux)

```bash
cd /workspace/llm-finetuning-lab

# 1. Configure backup env vars
cp .env.example .env
# Set ALERT_FROM_EMAIL, ALERT_TO_EMAIL, ALERT_EMAIL_PASSWORD
# Set B2_BUCKET if using a custom bucket name

# 2. Install rclone and configure remotes
bash scripts/setup_backup.sh

# 3. Run a manual test backup
bash scripts/backup.sh

# 4. Install cron (daily 3am backup, 4am verification)
bash scripts/setup_cron.sh
```

## Test Backup Without Waiting for Cron

```bash
bash scripts/backup.sh
tail -f logs/backup.log
```

Verify outputs:

```bash
ls -lh /backups/daily/
rclone size gdrive:llm-finetuning-backups/
rclone ls gdrive:llm-finetuning-backups/ | head
bash scripts/verify_backup.sh
```

## Check Backup Sizes

```bash
# Google Drive total
rclone size gdrive:llm-finetuning-backups/

# List all remote files
rclone ls gdrive:llm-finetuning-backups/

# Backblaze adapters only
rclone size b2:llm-finetuning-backups/
```

## Restore Procedures

### Scenario A: Small issue — restore from local archive (fastest)

```bash
bash scripts/restore.sh local all latest
```

Restore a specific date:

```bash
bash scripts/restore.sh local all 2026-06-27
```

### Scenario B: Disk corruption — restore from Google Drive

Full restore on a **new Vast.ai instance**:

```bash
cd /workspace/llm-finetuning-lab
bash scripts/setup_backup.sh   # reconfigure rclone remotes only
bash scripts/restore.sh gdrive all latest
bash scripts/setup_pm2.sh
```

Single client:

```bash
bash scripts/restore.sh gdrive clinical-notes latest
```

### Scenario C: Lost adapters only — restore from Backblaze B2

```bash
bash scripts/restore.sh b2 clinical-notes latest
bash scripts/restore.sh b2 all latest
```

### Scenario D: Complete disaster — new instance from scratch

1. Rent new Vast.ai instance with sufficient disk
2. Clone repo: `git clone <repo> /workspace/llm-finetuning-lab`
3. Copy `.env` from secure password manager (or decrypt stored backup)
4. Run `bash scripts/setup_backup.sh` to configure rclone
5. Run `bash scripts/restore.sh gdrive all latest`
6. Run `bash scripts/setup_pm2.sh`
7. Run `bash scripts/verify_backup.sh`
8. Smoke-test each client API

## Estimated Monthly Cost

| Storage | Typical size | Cost |
|---------|--------------|------|
| Google Drive | ~2–5 GB | Free (15 GB tier) |
| Backblaze B2 (adapters) | ~450 MB (3 × 150 MB LoRA) | ~$0.003/month |
| Local `/backups/` | 7 daily + 4 weekly archives | Disk space on instance |

B2 pricing: $0.006/GB/month. Three Mistral-7B LoRA adapters (~150 MB each) ≈ 450 MB → **~$0.003/month**.

## Monitoring

- Daily success email: `Backup OK — YYYY-MM-DD`
- Failure email: check `logs/backup.log`
- Cron schedule:
  - 03:00 — `scripts/backup.sh`
  - 04:00 — `scripts/verify_backup.sh`
  - Sunday 02:00 — prune project logs older than 30 days

## Troubleshooting

### Backup failed — rclone auth expired

```bash
rclone config reconnect gdrive:
bash scripts/backup.sh
```

### Local archive corrupt

```bash
df -h
dmesg | grep -i error
bash scripts/backup.sh
```

### GDrive quota exceeded

Prioritize adapters to B2, reduce weekly retention, or upgrade Google storage:

```bash
rclone size gdrive:llm-finetuning-backups/
find /backups/daily -name "*.tar.gz" -mtime +3 -delete
```

### Missing usage.db on GDrive

Ensure the file exists locally and re-sync:

```bash
ls projects/*/usage.db
bash scripts/backup.sh
bash scripts/verify_backup.sh
```

## Security Notes

- Project `.env` files are synced to Google Drive — restrict Drive access to operators only
- Use Gmail app passwords (not account password) for `ALERT_EMAIL_PASSWORD`
- B2 application keys should be read/write scoped to the backup bucket only
- Rotate B2 and Gmail credentials if an instance is compromised

## File Reference

```
scripts/
├── setup_backup.sh      # One-time rclone + folder setup
├── backup.sh            # Daily backup (cron)
├── verify_backup.sh     # Post-backup integrity check
├── restore.sh           # Disaster recovery
├── setup_cron.sh        # Install cron jobs
└── backup_alert.py      # Email alerts

/backups/
├── daily/               # Last 7 days of .tar.gz archives
└── weekly/              # Last 4 weekly full archives

logs/backup.log          # Append-only backup log
```
