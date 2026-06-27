#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env"
  set +a
fi

WORKSPACE="${WORKSPACE:-$ROOT_DIR}"

BACKUP_CRON="0 3 * * * bash $WORKSPACE/scripts/backup.sh >> $WORKSPACE/logs/backup.log 2>&1"
VERIFY_CRON="0 4 * * * bash $WORKSPACE/scripts/verify_backup.sh >> $WORKSPACE/logs/backup.log 2>&1"
LOG_CLEAN_CRON="0 2 * * 0 find $WORKSPACE/projects/*/logs -name '*.log' -mtime +30 -delete"
MONITOR_PYTHON="${PM2_PYTHON:-python}"
MONITOR_CRON="* * * * * cd $WORKSPACE && $MONITOR_PYTHON $WORKSPACE/scripts/run_monitor.py >> $WORKSPACE/logs/monitor.log 2>&1"

EXISTING_CRON="$(crontab -l 2>/dev/null || true)"

append_if_missing() {
  local line="$1"
  if ! echo "$EXISTING_CRON" | grep -Fq "$line"; then
    EXISTING_CRON="${EXISTING_CRON}
$line"
  fi
}

append_if_missing "$BACKUP_CRON"
append_if_missing "$VERIFY_CRON"
append_if_missing "$LOG_CLEAN_CRON"
append_if_missing "$MONITOR_CRON"

printf '%s\n' "$EXISTING_CRON" | crontab -

echo "Cron jobs installed:"
crontab -l
