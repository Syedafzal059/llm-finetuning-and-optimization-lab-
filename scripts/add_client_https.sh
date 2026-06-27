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

CLIENT="${1:-}"
PORT="${2:-}"
DOMAIN="${3:-${DOMAIN:-}}"
TUNNEL_NAME="${TUNNEL_NAME:-llm-api-tunnel}"
CLOUDFLARED_DIR="${HOME}/.cloudflared"
CONFIG_FILE="${CLOUDFLARED_DIR}/config.yml"

usage() {
  echo "Usage: $0 <client-subdomain> <local-port> [domain.com]"
  echo ""
  echo "Example:"
  echo "  $0 new-client 8004 yourdomain.com"
  echo ""
  echo "Requires DOMAIN (or pass as 3rd arg) and an existing tunnel (${TUNNEL_NAME})."
  exit 1
}

require_args() {
  if [[ -z "$CLIENT" || -z "$PORT" ]]; then
    usage
  fi
  if [[ -z "$DOMAIN" ]]; then
    echo "ERROR: DOMAIN is not set. Pass it as the 3rd argument or set DOMAIN in .env"
    exit 1
  fi
  if ! command -v cloudflared >/dev/null 2>&1; then
    echo "ERROR: cloudflared is not installed. Run scripts/setup_https.sh first."
    exit 1
  fi
  if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "ERROR: Tunnel config not found at ${CONFIG_FILE}. Run scripts/setup_https.sh first."
    exit 1
  fi
}

hostname_exists() {
  local host="$1"
  grep -q "hostname: ${host}" "$CONFIG_FILE"
}

add_dns_route() {
  local subdomain="${CLIENT}.${DOMAIN}"
  echo "=== Adding DNS route: ${subdomain} ==="
  if cloudflared tunnel route dns "$TUNNEL_NAME" "$subdomain" 2>&1 | grep -qi "already exists"; then
    echo "DNS route already exists: ${subdomain}"
  else
    cloudflared tunnel route dns "$TUNNEL_NAME" "$subdomain"
    echo "DNS route created: ${subdomain}"
  fi
}

add_ingress_rule() {
  local subdomain="${CLIENT}.${DOMAIN}"

  if hostname_exists "$subdomain"; then
    echo "Ingress rule already exists for ${subdomain} — updating port to ${PORT}"
    sed -i "/hostname: ${subdomain}/,/^  - hostname:/ s|service: http://localhost:[0-9]*|service: http://localhost:${PORT}|" "$CONFIG_FILE"
    return
  fi

  echo "=== Adding ingress rule for ${subdomain} → localhost:${PORT} ==="
  local tmp_file
  tmp_file="$(mktemp)"

  awk -v client="$CLIENT" -v domain="$DOMAIN" -v port="$PORT" '
    /# Catch-all/ {
      print "  - hostname: " client "." domain
      print "    service: http://localhost:" port
      print "    originRequest:"
      print "      connectTimeout: 30s"
      print "      keepAliveTimeout: 90s"
      print "      keepAliveConnections: 100"
      print "      noTLSVerify: false"
      print ""
    }
    { print }
  ' "$CONFIG_FILE" > "$tmp_file"

  mv "$tmp_file" "$CONFIG_FILE"
  echo "Ingress rule added"
}

restart_tunnel() {
  echo ""
  echo "=== Validating config ==="
  cloudflared tunnel --config "$CONFIG_FILE" ingress validate
  echo "Config validated"

  echo ""
  echo "=== Restarting cloudflared ==="
  sudo systemctl restart cloudflared
  echo "Tunnel restarted"
}

print_summary() {
  echo ""
  echo "New endpoint live:"
  echo "  https://${CLIENT}.${DOMAIN}"
  echo ""
  echo "Test it:"
  echo "  curl -I https://${CLIENT}.${DOMAIN}/health"
}

main() {
  require_args
  add_dns_route
  add_ingress_rule
  restart_tunnel
  print_summary
}

main "$@"
