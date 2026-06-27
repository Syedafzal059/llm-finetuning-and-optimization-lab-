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
CLIENTS=(clinical-notes medical-coding patient-support)
PORTS=(8001 8002 8003)

ACTION="${1:-}"
CLIENT="${2:-}"
PORT="${3:-}"

usage() {
  cat <<'EOF'
Usage: ./scripts/manage.sh <command> [client] [port]

Commands:
  status                 Show pm2 status and Redis queue depths
  logs <client>          Tail last 50 log lines for one client
  restart <client>       Restart one client
  restart-all            Restart all clients
  stop <client>          Stop one client
  stop-all               Stop all clients
  add <client> <port>    Start a new client via pm2
  health                 Hit /health on all configured ports
  monitor                Open pm2 live dashboard
EOF
}

require_pm2() {
  if ! command -v pm2 >/dev/null 2>&1; then
    echo "pm2 is not installed. Run ./scripts/setup_pm2.sh first." >&2
    exit 1
  fi
}

queue_depth() {
  local project="$1"
  if command -v redis-cli >/dev/null 2>&1; then
    redis-cli LLEN "queue:${project}" 2>/dev/null || echo "unavailable"
  else
    echo "redis-cli not installed"
  fi
}

case "$ACTION" in
  status)
    require_pm2
    echo "=== All client status ==="
    pm2 status
    echo ""
    echo "=== Queue depths ==="
    for project in "${CLIENTS[@]}"; do
      echo "${project}: $(queue_depth "$project")"
    done
    ;;

  logs)
    require_pm2
    if [[ -z "$CLIENT" ]]; then
      echo "Usage: ./scripts/manage.sh logs <client>" >&2
      exit 1
    fi
    pm2 logs "$CLIENT" --lines 50
    ;;

  restart)
    require_pm2
    if [[ -z "$CLIENT" ]]; then
      echo "Usage: ./scripts/manage.sh restart <client>" >&2
      exit 1
    fi
    echo "Restarting ${CLIENT}..."
    pm2 restart "$CLIENT"
    echo "Done. New status:"
    pm2 status "$CLIENT"
    ;;

  restart-all)
    require_pm2
    echo "Restarting all clients..."
    pm2 restart all
    pm2 status
    ;;

  stop)
    require_pm2
    if [[ -z "$CLIENT" ]]; then
      echo "Usage: ./scripts/manage.sh stop <client>" >&2
      exit 1
    fi
    pm2 stop "$CLIENT"
    ;;

  stop-all)
    require_pm2
    pm2 stop all
    ;;

  add)
    require_pm2
    if [[ -z "$CLIENT" || -z "$PORT" ]]; then
      echo "Usage: ./scripts/manage.sh add <client> <port>" >&2
      exit 1
    fi
    mkdir -p "$WORKSPACE/projects/${CLIENT}/logs"
    pm2 start run.py \
      --name "$CLIENT" \
      --interpreter "${PM2_PYTHON:-python}" \
      --cwd "$WORKSPACE" \
      -- \
      --project "$CLIENT" \
      --mode serve \
      --port "$PORT"
    pm2 save
    echo "${CLIENT} started on port ${PORT}"
    ;;

  health)
    echo "=== Checking all endpoints ==="
    for index in "${!PORTS[@]}"; do
      port="${PORTS[$index]}"
      client="${CLIENTS[$index]}"
      echo "--- ${client} (port ${port}) ---"
      if curl -sf "http://localhost:${port}/health" | python -m json.tool; then
        continue
      fi
      echo "Health check failed for ${client} on port ${port}" >&2
    done
    ;;

  monitor)
    require_pm2
    pm2 monit
    ;;

  *)
    usage
    exit 1
    ;;
esac
