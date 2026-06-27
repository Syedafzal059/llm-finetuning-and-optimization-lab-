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

SOURCE="${1:-}"
PROJECT="${2:-}"
DATE="${3:-latest}"

usage() {
  cat <<EOF
Usage: bash scripts/restore.sh <local|gdrive|b2> <project|all> [date|latest]

Examples:
  bash scripts/restore.sh local all latest
  bash scripts/restore.sh gdrive clinical-notes latest
  bash scripts/restore.sh b2 medical-coding latest
EOF
}

if [[ -z "$SOURCE" || -z "$PROJECT" ]]; then
  usage
  exit 1
fi

if ! command -v rclone >/dev/null 2>&1; then
  echo "rclone is required for gdrive/b2 restores"
  exit 1
fi

case "$SOURCE" in
  local)
    echo "=== Restoring from local archive ==="

    if [[ "$DATE" == "latest" ]]; then
      ARCHIVE="$(ls -t "$BACKUP_DIR/daily/"*.tar.gz 2>/dev/null | head -1 || true)"
      if [[ -z "$ARCHIVE" ]]; then
        echo "No local archives found in $BACKUP_DIR/daily/"
        exit 1
      fi
    else
      ARCHIVE="$BACKUP_DIR/daily/backup-$DATE.tar.gz"
    fi

    if [[ ! -f "$ARCHIVE" ]]; then
      echo "Archive not found: $ARCHIVE"
      exit 1
    fi

    echo "Restoring from: $ARCHIVE"
    tar -xzf "$ARCHIVE" -C "$WORKSPACE"
    echo "Local restore complete"
    ;;

  gdrive)
    echo "=== Restoring from Google Drive ==="

    if [[ "$PROJECT" == "all" ]]; then
      rclone sync \
        "${GDRIVE_REMOTE}:${GDRIVE_FOLDER}/projects/" \
        "$WORKSPACE/projects/"
    else
      mkdir -p "$WORKSPACE/projects/$PROJECT"
      rclone sync \
        "${GDRIVE_REMOTE}:${GDRIVE_FOLDER}/projects/$PROJECT/" \
        "$WORKSPACE/projects/$PROJECT/"
    fi
    echo "GDrive restore complete"
    ;;

  b2)
    echo "=== Restoring adapters from B2 ==="

    if [[ "$PROJECT" == "all" ]]; then
      for PROJECT_DIR in "$WORKSPACE"/projects/*/; do
        [[ -d "$PROJECT_DIR" ]] || continue
        CLIENT="$(basename "$PROJECT_DIR")"
        mkdir -p "$WORKSPACE/projects/$CLIENT/model"
        rclone sync \
          "${B2_REMOTE}:${B2_BUCKET}/${CLIENT}/" \
          "$WORKSPACE/projects/$CLIENT/model/" \
          2>/dev/null || true
      done
    else
      mkdir -p "$WORKSPACE/projects/$PROJECT/model"
      rclone sync \
        "${B2_REMOTE}:${B2_BUCKET}/${PROJECT}/" \
        "$WORKSPACE/projects/$PROJECT/model/"
    fi
    echo "B2 restore complete"
    ;;

  *)
    usage
    exit 1
    ;;
esac

echo "=== Restarting APIs ==="
if command -v pm2 >/dev/null 2>&1; then
  pm2 restart all || true
  pm2 status
else
  echo "pm2 not installed — restart services manually"
fi
