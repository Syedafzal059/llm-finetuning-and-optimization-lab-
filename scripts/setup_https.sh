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
  echo "   Copy .env.example to .env and fill in Cloudflare settings."
  exit 1
fi

TUNNEL_NAME="${TUNNEL_NAME:-llm-api-tunnel}"

require_root() {
  if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
    echo "❌ This script must be run as root (sudo ./scripts/setup_https.sh)"
    exit 1
  fi
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "❌ Required command not found: $1"
    exit 1
  fi
}

echo "=== Checking prerequisites ==="

require_root
require_command curl
require_command python3
require_command lsb_release
require_command systemctl

if [[ -z "${DOMAIN:-}" ]]; then
  echo "❌ DOMAIN not set in .env"
  echo "   Add: DOMAIN=yourdomain.com"
  exit 1
fi

if [[ -z "${CF_API_TOKEN:-}" ]]; then
  echo "❌ CF_API_TOKEN not set in .env"
  echo "   Get from: cloudflare.com → Profile → API Tokens"
  exit 1
fi

echo "✅ Prerequisites OK"
echo "   Domain: $DOMAIN"
echo "   Tunnel name: $TUNNEL_NAME"

echo ""
echo "=== Installing cloudflared ==="

mkdir -p --mode=0755 /usr/share/keyrings

curl -fsSL \
  https://pkg.cloudflare.com/cloudflare-main.gpg \
  | tee /usr/share/keyrings/cloudflare-main.gpg \
  >/dev/null

echo "deb [signed-by=/usr/share/keyrings/cloudflare-main.gpg] https://pkg.cloudflare.com/cloudflared $(lsb_release -cs) main" \
  | tee /etc/apt/sources.list.d/cloudflared.list

apt-get update -qq
apt-get install -y cloudflared

echo "✅ cloudflared $(cloudflared --version)"

echo ""
echo "=== Login to Cloudflare ==="
echo "A browser window will open."
echo "Login and click Authorize."
echo ""
cloudflared tunnel login

echo ""
echo "=== Creating tunnel ==="
if cloudflared tunnel list --output json | python3 -c "
import json, sys
name = sys.argv[1]
tunnels = json.load(sys.stdin)
sys.exit(0 if any(t.get('name') == name for t in tunnels) else 1)
" "$TUNNEL_NAME" 2>/dev/null; then
  echo "Tunnel '$TUNNEL_NAME' already exists; reusing it."
else
  cloudflared tunnel create "$TUNNEL_NAME"
fi

TUNNEL_ID="$(cloudflared tunnel list --output json | python3 -c "
import json, sys
name = sys.argv[1]
for tunnel in json.load(sys.stdin):
    if tunnel.get('name') == name:
        print(tunnel['id'])
        break
" "$TUNNEL_NAME")"

if [[ -z "$TUNNEL_ID" ]]; then
  echo "❌ Could not get tunnel ID"
  exit 1
fi

echo "✅ Tunnel ID: $TUNNEL_ID"

if grep -q "^TUNNEL_ID=" "$ROOT_DIR/.env"; then
  sed -i "s/^TUNNEL_ID=.*/TUNNEL_ID=$TUNNEL_ID/" "$ROOT_DIR/.env"
else
  echo "TUNNEL_ID=$TUNNEL_ID" >> "$ROOT_DIR/.env"
fi

echo ""
echo "=== Creating DNS routes ==="

cloudflared tunnel route dns "$TUNNEL_NAME" "clinical-notes.$DOMAIN"
echo "✅ clinical-notes.$DOMAIN"

cloudflared tunnel route dns "$TUNNEL_NAME" "medical-coding.$DOMAIN"
echo "✅ medical-coding.$DOMAIN"

cloudflared tunnel route dns "$TUNNEL_NAME" "patient-support.$DOMAIN"
echo "✅ patient-support.$DOMAIN"

echo ""
echo "=== Writing tunnel config ==="

mkdir -p ~/.cloudflared

cat > ~/.cloudflared/config.yml << EOF
tunnel: $TUNNEL_ID
credentials-file: /root/.cloudflared/$TUNNEL_ID.json

ingress:
  - hostname: clinical-notes.$DOMAIN
    service: http://localhost:8001
    originRequest:
      connectTimeout: 30s
      keepAliveTimeout: 90s

  - hostname: medical-coding.$DOMAIN
    service: http://localhost:8002
    originRequest:
      connectTimeout: 30s
      keepAliveTimeout: 90s

  - hostname: patient-support.$DOMAIN
    service: http://localhost:8003
    originRequest:
      connectTimeout: 30s
      keepAliveTimeout: 90s

  - service: http_status:404
EOF

echo "✅ Config written"

echo ""
echo "=== Validating config ==="
cloudflared tunnel --config ~/.cloudflared/config.yml ingress validate
echo "✅ Config valid"

echo ""
echo "=== Installing as system service ==="
cloudflared --config ~/.cloudflared/config.yml service install

systemctl enable cloudflared
systemctl restart cloudflared

sleep 3

STATUS="$(systemctl is-active cloudflared)"
if [[ "$STATUS" == "active" ]]; then
  echo "✅ Tunnel service running"
else
  echo "❌ Tunnel service failed"
  echo "   Check: journalctl -u cloudflared"
  exit 1
fi

echo ""
echo "=== Waiting 10s for DNS to propagate ==="
sleep 10

echo ""
echo "=== Testing endpoints ==="

for SUBDOMAIN in clinical-notes medical-coding patient-support; do
  URL="https://$SUBDOMAIN.$DOMAIN/health"
  STATUS="$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "$URL" || echo "000")"

  if [[ "$STATUS" == "200" ]]; then
    echo "✅ $URL → $STATUS"
  else
    echo "⚠️  $URL → $STATUS (may need more time)"
  fi
done

echo ""
echo "=== Setup complete ==="
echo ""
echo "Your HTTPS endpoints:"
echo "  https://clinical-notes.$DOMAIN"
echo "  https://medical-coding.$DOMAIN"
echo "  https://patient-support.$DOMAIN"
echo ""
echo "Note: DNS can take up to 5 minutes to fully propagate."
echo "If health checks returned non-200, wait and retry:"
echo "  curl https://clinical-notes.$DOMAIN/health"
echo ""
echo "See docs/cloudflare_verify.md for post-setup checks."
