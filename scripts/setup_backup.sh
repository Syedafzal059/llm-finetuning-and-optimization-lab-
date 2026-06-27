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

echo "=== Installing rclone ==="
if ! command -v rclone >/dev/null 2>&1; then
  curl https://rclone.org/install.sh | sudo bash
else
  echo "rclone already installed: $(rclone version | head -n 1)"
fi

echo ""
echo "=== Configure Google Drive ==="
echo "Run: rclone config"
echo "Then follow these steps:"
echo "  n (new remote)"
echo "  name: ${GDRIVE_REMOTE}"
echo "  storage: drive (Google Drive)"
echo "  Follow OAuth flow in browser"
echo ""
read -r -p "Press enter when ${GDRIVE_REMOTE} is configured..."

echo ""
echo "=== Configure Backblaze B2 ==="
echo "Run: rclone config"
echo "Then:"
echo "  n (new remote)"
echo "  name: ${B2_REMOTE}"
echo "  storage: b2 (Backblaze B2)"
echo "  Enter your B2 key ID and application key"
echo ""
read -r -p "Press enter when ${B2_REMOTE} is configured..."

echo ""
echo "=== Create backup folders ==="
mkdir -p "$BACKUP_DIR/daily"
mkdir -p "$BACKUP_DIR/weekly"
mkdir -p "$WORKSPACE/logs"
echo "Local backup dir: $BACKUP_DIR"

echo ""
echo "=== Test Google Drive connection ==="
rclone lsd "${GDRIVE_REMOTE}:"
echo "Google Drive connected"

echo ""
echo "=== Test Backblaze connection ==="
rclone lsd "${B2_REMOTE}:"
echo "Backblaze B2 connected"

echo ""
echo "=== Create B2 bucket (if missing) ==="
rclone mkdir "${B2_REMOTE}:${B2_BUCKET}" 2>/dev/null || true
echo "B2 bucket ready: ${B2_BUCKET}"

echo ""
echo "=== Create GDrive folder (if missing) ==="
rclone mkdir "${GDRIVE_REMOTE}:${GDRIVE_FOLDER}" 2>/dev/null || true
echo "GDrive folder ready: ${GDRIVE_FOLDER}"

echo ""
echo "=== Setup complete ==="
echo "Next steps:"
echo "  1. Fill backup settings in .env (see .env.example)"
echo "  2. Run a test backup: bash scripts/backup.sh"
echo "  3. Install cron: bash scripts/setup_cron.sh"
