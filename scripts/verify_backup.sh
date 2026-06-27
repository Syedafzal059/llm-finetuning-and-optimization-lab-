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
BACKUP_DIR="${BACKUP_DIR:-/backups}"
GDRIVE_REMOTE="${GDRIVE_REMOTE:-gdrive}"
GDRIVE_FOLDER="${GDRIVE_FOLDER:-llm-finetuning-backups}"
LOG="$WORKSPACE/logs/backup.log"
ERRORS=0

mkdir -p "$WORKSPACE/logs"

log() {
  echo "$(date '+%Y-%m-%d %H:%M:%S') | $1" | tee -a "$LOG"
}

log "=== Backup verification started ==="

if ! command -v rclone >/dev/null 2>&1; then
  log "rclone not installed — verification skipped"
  exit 1
fi

TODAY_ARCHIVE="$BACKUP_DIR/daily/backup-$(date +%Y-%m-%d).tar.gz"
if [[ -f "$TODAY_ARCHIVE" ]]; then
  if tar -tzf "$TODAY_ARCHIVE" >/dev/null 2>&1; then
    log "Today's local archive integrity OK"
  else
    log "Today's local archive is CORRUPT"
    ERRORS=$((ERRORS + 1))
  fi
else
  log "Today's local archive missing: $TODAY_ARCHIVE"
  ERRORS=$((ERRORS + 1))
fi

for PROJECT_DIR in "$WORKSPACE"/projects/*/; do
  [[ -d "$PROJECT_DIR" ]] || continue
  PROJECT="$(basename "$PROJECT_DIR")"

  if [[ -d "$PROJECT_DIR/model/lora" ]]; then
    COUNT="$(rclone ls \
      "${GDRIVE_REMOTE}:${GDRIVE_FOLDER}/projects/${PROJECT}/model/lora/" \
      2>/dev/null | wc -l | tr -d ' ')"

    if [[ "$COUNT" -gt 0 ]]; then
      log "$PROJECT lora adapter on GDrive: OK ($COUNT files)"
    else
      log "$PROJECT lora adapter on GDrive: MISSING"
      ERRORS=$((ERRORS + 1))
    fi
  fi

  if [[ -f "$PROJECT_DIR/usage.db" ]]; then
    EXISTS="$(rclone ls \
      "${GDRIVE_REMOTE}:${GDRIVE_FOLDER}/projects/${PROJECT}/usage.db" \
      2>/dev/null | wc -l | tr -d ' ')"

    if [[ "$EXISTS" -gt 0 ]]; then
      log "$PROJECT usage.db on GDrive: OK"
    else
      log "$PROJECT usage.db on GDrive: MISSING"
      ERRORS=$((ERRORS + 1))
    fi
  fi

  for audit_log in pii_audit.log security.log; do
    if [[ -f "$PROJECT_DIR/logs/$audit_log" ]]; then
      REMOTE_EXISTS="$(rclone ls \
        "${GDRIVE_REMOTE}:${GDRIVE_FOLDER}/projects/${PROJECT}/logs/${audit_log}" \
        2>/dev/null | wc -l | tr -d ' ')"

      if [[ "$REMOTE_EXISTS" -gt 0 ]]; then
        log "$PROJECT $audit_log on GDrive: OK"
      else
        log "$PROJECT $audit_log on GDrive: MISSING"
        ERRORS=$((ERRORS + 1))
      fi
    fi
  done
done

if [[ "$ERRORS" -gt 0 ]]; then
  log "Verification failed: $ERRORS errors"
  python "$ROOT_DIR/scripts/backup_alert.py" \
    "FAILED" "$(date +%Y-%m-%d)" "$ERRORS verification errors"
  exit 1
fi

log "All backups verified successfully"
