#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env"
  set +a
else
  echo "❌ .env not found at $ROOT_DIR/.env"
  exit 1
fi

CLIENT="${1:-}"
PORT="${2:-}"
TUNNEL_NAME="${TUNNEL_NAME:-llm-api-tunnel}"
CONFIG_FILE="${HOME}/.cloudflared/config.yml"

if [[ -z "$CLIENT" || -z "$PORT" ]]; then
  echo "Usage: ./scripts/add_https_client.sh <client-name> <port>"
  echo "Example: ./scripts/add_https_client.sh new-client 8004"
  exit 1
fi

if [[ -z "${DOMAIN:-}" ]]; then
  echo "❌ DOMAIN not set in .env"
  exit 1
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "❌ Cloudflared config not found at $CONFIG_FILE"
  echo "   Run ./scripts/setup_https.sh first."
  exit 1
fi

if ! command -v cloudflared >/dev/null 2>&1; then
  echo "❌ cloudflared is not installed"
  exit 1
fi

echo "=== Adding HTTPS for $CLIENT ==="

cloudflared tunnel route dns "$TUNNEL_NAME" "$CLIENT.$DOMAIN"
echo "✅ DNS: $CLIENT.$DOMAIN"

if grep -q "hostname: $CLIENT.$DOMAIN" "$CONFIG_FILE"; then
  echo "⚠️  Hostname $CLIENT.$DOMAIN already present in config; skipping insert."
else
  sed -i "/- service: http_status:404/i\\
  - hostname: $CLIENT.$DOMAIN\\
    service: http://localhost:$PORT\\
    originRequest:\\
      connectTimeout: 30s\\
      keepAliveTimeout: 90s\\
" "$CONFIG_FILE"
  echo "✅ Config updated"
fi

cloudflared tunnel --config "$CONFIG_FILE" ingress validate
echo "✅ Config valid"

if command -v systemctl >/dev/null 2>&1; then
  systemctl restart cloudflared
  sleep 3
  echo "✅ Tunnel restarted"
else
  echo "⚠️  systemctl not available; restart cloudflared manually."
fi

echo ""
echo "New endpoint live:"
echo "https://$CLIENT.$DOMAIN"
echo ""
echo "Test:"
echo "curl https://$CLIENT.$DOMAIN/health"
