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
export WORKSPACE

ensure_log_dirs() {
  local project
  for project in clinical-notes medical-coding patient-support; do
    mkdir -p "$WORKSPACE/projects/$project/logs"
  done
}

echo "=== Installing pm2 ==="
if ! command -v pm2 >/dev/null 2>&1; then
  npm install -g pm2
else
  echo "pm2 already installed: $(pm2 -v)"
fi

echo "=== Installing pm2-logrotate (keep crash logs) ==="
pm2 install pm2-logrotate || true
pm2 set pm2-logrotate:max_size 100M
pm2 set pm2-logrotate:retain 365
pm2 set pm2-logrotate:compress true

ensure_log_dirs

echo "=== Starting all clients ==="
pm2 start "$ROOT_DIR/ecosystem.config.js"

echo "=== Starting crash alert listener ==="
pm2 start "$ROOT_DIR/scripts/pm2_event_listener.js" \
  --name pm2-crash-listener \
  --cwd "$ROOT_DIR"

echo "=== Saving pm2 process list ==="
pm2 save

echo "=== Setting up auto start on reboot ==="
STARTUP_CMD="$(pm2 startup | tail -n 1 || true)"
if [[ -n "$STARTUP_CMD" && "$STARTUP_CMD" == sudo* ]]; then
  echo ""
  echo "Run this command as root to enable auto-start on reboot:"
  echo "$STARTUP_CMD"
else
  pm2 startup || true
fi

echo ""
echo "=== Current status ==="
pm2 status

echo ""
echo "IMPORTANT: Copy and run the startup command printed above"
echo "to enable auto start on server reboot."
