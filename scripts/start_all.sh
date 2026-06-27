#!/usr/bin/env bash
set -e

WORKSPACE="${WORKSPACE:-/workspace/llm-finetuning-lab}"
ADAPTER_DIR="$WORKSPACE/projects/clinical-notes/model/qlora"
PORTS=(8001 8002 8003)

cd "$WORKSPACE"

if [[ ! -d "$WORKSPACE/venv" ]]; then
  echo "❌ Virtualenv not found. Run scripts/deploy.sh first."
  exit 1
fi

# shellcheck disable=SC1091
source "$WORKSPACE/venv/bin/activate"

if ! command -v pm2 >/dev/null 2>&1; then
  echo "❌ pm2 not installed. Run scripts/vastai_first_login.sh first."
  exit 1
fi

if ! redis-cli ping | grep -q PONG; then
  echo "❌ Redis is not running. Run: sudo systemctl start redis-server"
  exit 1
fi

if [[ ! -f "$ADAPTER_DIR/adapter_config.json" ]]; then
  echo "❌ clinical-notes adapter missing at $ADAPTER_DIR"
  echo "   Run training first: bash scripts/train.sh"
  echo "   Expected file: $ADAPTER_DIR/adapter_config.json"
  exit 1
fi

echo "✅ clinical-notes adapter found"

echo ""
echo "=== Starting pm2 processes ==="
pm2 start "$WORKSPACE/ecosystem.config.js"
pm2 save

echo ""
echo "=== Waiting 30s for models to load ==="
sleep 30

echo ""
echo "=== Checking health ==="
for PORT in "${PORTS[@]}"; do
  STATUS="$(curl -s -o /dev/null -w "%{http_code}" --max-time 15 "http://localhost:$PORT/health" || echo "000")"
  if [[ "$STATUS" == "200" ]]; then
    echo "✅ Port $PORT: OK"
  else
    echo "⚠️  Port $PORT: $STATUS (may still be loading — check: pm2 logs)"
  fi
done

echo ""
echo "=== pm2 status ==="
pm2 status

echo ""
echo "=== APIs started ==="
echo "Local endpoints:"
echo "  http://localhost:8001/health  (clinical-notes)"
echo "  http://localhost:8002/health  (medical-coding)"
echo "  http://localhost:8003/health  (patient-support)"
echo ""
echo "Next steps:"
echo "  sudo bash scripts/setup_https.sh     # public HTTPS"
echo "  bash scripts/setup_pm2.sh            # crash alerts + reboot auto-start"
