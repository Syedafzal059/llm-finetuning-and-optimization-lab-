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
B2_REMOTE="${B2_REMOTE:-b2}"
B2_BUCKET="${B2_BUCKET:-llm-finetuning-backups}"
GDRIVE_FOLDER="${GDRIVE_FOLDER:-llm-finetuning-backups}"
KEEP_DAILY_DAYS="${KEEP_DAILY_DAYS:-7}"
KEEP_WEEKLY_WEEKS="${KEEP_WEEKLY_WEEKS:-4}"

DATE="$(date +%Y-%m-%d)"
TIME="$(date +%H-%M-%S)"
LOG="$WORKSPACE/logs/backup.log"

mkdir -p "$BACKUP_DIR/daily" "$BACKUP_DIR/weekly" "$WORKSPACE/logs"

log() {
  echo "$(date '+%Y-%m-%d %H:%M:%S') | $1" | tee -a "$LOG"
}

send_failure_alert() {
  local message="$1"
  if command -v python >/dev/null 2>&1; then
    python "$ROOT_DIR/scripts/backup_alert.py" "FAILED" "$DATE" "$message" || true
  fi
}

on_error() {
  local exit_code=$?
  log "Backup failed with exit code $exit_code at line ${BASH_LINENO[0]}"
  send_failure_alert "Backup script failed at line ${BASH_LINENO[0]} (exit $exit_code)"
  exit "$exit_code"
}

trap on_error ERR

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    log "Required command not found: $1"
    send_failure_alert "Required command not found: $1"
    exit 1
  fi
}

require_command tar
require_command rclone
require_command python

collect_backup_paths() {
  local -n _paths_ref=$1
  local candidate

  for candidate in \
    "$WORKSPACE"/projects/*/model/lora \
    "$WORKSPACE"/projects/*/model/qlora \
    "$WORKSPACE"/projects/*/data \
    "$WORKSPACE"/projects/*/usage.db \
    "$WORKSPACE"/projects/*/config.yaml \
    "$WORKSPACE"/projects/*/prompt_template.py \
    "$WORKSPACE"/projects/*/logs \
    "$WORKSPACE"/projects/*/.env \
    "$WORKSPACE"/ecosystem.config.cjs \
    "$WORKSPACE"/base_config.yaml; do
    if [[ -e "$candidate" ]]; then
      _paths_ref+=("$candidate")
    fi
  done
}

log "=== Backup started: $DATE $TIME ==="

# ─────────────────────────────────────────
# STEP 1: LOCAL COMPRESSED ARCHIVE
# ─────────────────────────────────────────

log "Step 1: Creating local archive..."

ARCHIVE="$BACKUP_DIR/daily/backup-$DATE.tar.gz"
BACKUP_PATHS=()
collect_backup_paths BACKUP_PATHS

if [[ ${#BACKUP_PATHS[@]} -eq 0 ]]; then
  log "No backup paths found under $WORKSPACE/projects/"
  send_failure_alert "No backup paths found"
  exit 1
fi

RELATIVE_PATHS=()
for path in "${BACKUP_PATHS[@]}"; do
  RELATIVE_PATHS+=("${path#"$WORKSPACE"/}")
done

tar -czf "$ARCHIVE" \
  --exclude="*/venv/*" \
  --exclude="*/__pycache__/*" \
  --exclude="*/node_modules/*" \
  --exclude="*/model/sft/*" \
  --exclude="*/model/base/*" \
  -C "$WORKSPACE" \
  "${RELATIVE_PATHS[@]}"

ARCHIVE_SIZE="$(du -sh "$ARCHIVE" | cut -f1)"
log "Local archive: $ARCHIVE ($ARCHIVE_SIZE)"

find "$BACKUP_DIR/daily" -name "*.tar.gz" -mtime +"$KEEP_DAILY_DAYS" -delete
log "Old local archives cleaned up (keep ${KEEP_DAILY_DAYS} days)"

# ─────────────────────────────────────────
# STEP 2: GOOGLE DRIVE SYNC
# ─────────────────────────────────────────

log "Step 2: Syncing to Google Drive..."

rclone sync \
  "$WORKSPACE/projects/" \
  "${GDRIVE_REMOTE}:${GDRIVE_FOLDER}/projects/" \
  --exclude "*/venv/**" \
  --exclude "**/__pycache__/**" \
  --exclude "**/node_modules/**" \
  --exclude "*/model/sft/**" \
  --exclude "*/model/base/**" \
  --transfers 4 \
  --checkers 8 \
  --log-level INFO \
  --log-file "$LOG"

log "Google Drive sync complete"

rclone copy \
  "$ARCHIVE" \
  "${GDRIVE_REMOTE}:${GDRIVE_FOLDER}/archives/"

log "Archive uploaded to Google Drive"

# ─────────────────────────────────────────
# STEP 3: BACKBLAZE B2 (MODEL ADAPTERS)
# ─────────────────────────────────────────

log "Step 3: Syncing adapters to Backblaze..."

for PROJECT_DIR in "$WORKSPACE"/projects/*/; do
  [[ -d "$PROJECT_DIR" ]] || continue
  PROJECT="$(basename "$PROJECT_DIR")"

  if [[ -d "$PROJECT_DIR/model/lora" ]]; then
    rclone sync \
      "$PROJECT_DIR/model/lora/" \
      "${B2_REMOTE}:${B2_BUCKET}/${PROJECT}/lora/" \
      --log-level INFO \
      --log-file "$LOG"
    log "$PROJECT lora adapter backed up to B2"
  fi

  if [[ -d "$PROJECT_DIR/model/qlora" ]]; then
    rclone sync \
      "$PROJECT_DIR/model/qlora/" \
      "${B2_REMOTE}:${B2_BUCKET}/${PROJECT}/qlora/" \
      --log-level INFO \
      --log-file "$LOG"
    log "$PROJECT qlora adapter backed up to B2"
  fi
done

# ─────────────────────────────────────────
# STEP 4: VERIFY BACKUP INTEGRITY
# ─────────────────────────────────────────

log "Step 4: Verifying backup integrity..."

if tar -tzf "$ARCHIVE" >/dev/null 2>&1; then
  log "Local archive integrity OK"
else
  log "Local archive CORRUPT"
  python "$ROOT_DIR/scripts/backup_alert.py" "CORRUPT" "$DATE" "$ARCHIVE"
  exit 1
fi

GDRIVE_COUNT="$(rclone ls "${GDRIVE_REMOTE}:${GDRIVE_FOLDER}/projects/" 2>/dev/null | wc -l | tr -d ' ')"
log "GDrive file count: $GDRIVE_COUNT"

# ─────────────────────────────────────────
# STEP 5: WEEKLY FULL BACKUP
# ─────────────────────────────────────────

if [[ "$(date +%u)" -eq 7 ]]; then
  log "Step 5: Weekly full backup..."

  WEEKLY="$BACKUP_DIR/weekly/full-$DATE.tar.gz"

  tar -czf "$WEEKLY" \
    --exclude="*/venv/*" \
    --exclude="*/__pycache__/*" \
    --exclude="*/node_modules/*" \
    --exclude="*/model/sft/*" \
    --exclude="*/model/base/*" \
    -C "$WORKSPACE" projects/

  rclone copy "$WEEKLY" "${GDRIVE_REMOTE}:${GDRIVE_FOLDER}/weekly/"
  rclone copy "$WEEKLY" "${B2_REMOTE}:${B2_BUCKET}/weekly/"

  find "$BACKUP_DIR/weekly" -name "*.tar.gz" \
    -mtime +$((KEEP_WEEKLY_WEEKS * 7)) -delete

  log "Weekly full backup complete"
fi

# ─────────────────────────────────────────
# STEP 6: SEND SUMMARY EMAIL
# ─────────────────────────────────────────

log "Step 6: Sending backup summary..."

python "$ROOT_DIR/scripts/backup_alert.py" \
  "SUCCESS" \
  "$DATE" \
  "$ARCHIVE_SIZE" \
  "$GDRIVE_COUNT"

log "=== Backup complete: $DATE ==="
log "================================"
