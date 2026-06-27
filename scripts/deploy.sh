#!/usr/bin/env bash
set -e

REPO_URL="${1:-}"
WORKSPACE="${WORKSPACE:-/workspace/llm-finetuning-lab}"
MODEL_DIR="/workspace/models/mistral-7b"
PROJECTS=(clinical-notes medical-coding patient-support)

usage() {
  echo "Usage: bash scripts/deploy.sh <github-repo-url>"
  echo "Example: bash scripts/deploy.sh https://github.com/you/llm-finetuning-lab"
  exit 1
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "❌ Required command not found: $1"
    echo "   Run scripts/vastai_first_login.sh first."
    exit 1
  fi
}

if [[ -z "$REPO_URL" ]]; then
  usage
fi

echo "=== Deploying LLM Fine-tuning Lab ==="
echo "Workspace: $WORKSPACE"
echo ""

echo "=== Checking prerequisites ==="
require_command git
require_command python3
require_command redis-cli
require_command nvidia-smi
redis-cli ping | grep -q PONG || {
  echo "❌ Redis is not running. Run: sudo systemctl start redis-server"
  exit 1
}
python3 -c "import torch; assert torch.cuda.is_available()" || {
  echo "❌ CUDA GPU not available"
  exit 1
}
echo "✅ Prerequisites OK"

# ─────────────────────────────────────────
# CLONE REPO
# ─────────────────────────────────────────
echo ""
echo "=== Cloning repo ==="
mkdir -p /workspace
if [[ -d "$WORKSPACE/.git" ]]; then
  echo "Repo already exists at $WORKSPACE — pulling latest"
  cd "$WORKSPACE"
  git pull --ff-only
else
  git clone "$REPO_URL" "$WORKSPACE"
  cd "$WORKSPACE"
fi
echo "✅ Repo ready at $WORKSPACE"

# ─────────────────────────────────────────
# PYTHON ENVIRONMENT
# ─────────────────────────────────────────
echo ""
echo "=== Setting up Python environment ==="
if [[ ! -d "$WORKSPACE/venv" ]]; then
  python3 -m venv "$WORKSPACE/venv"
fi
# shellcheck disable=SC1091
source "$WORKSPACE/venv/bin/activate"

pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "✅ Python environment ready"

# ─────────────────────────────────────────
# VERIFY KEY PACKAGES
# ─────────────────────────────────────────
echo ""
echo "=== Verifying key packages ==="
python3 -c "
import torch
import transformers
import peft
import fastapi
import vllm
import redis
print('✅ torch:', torch.__version__)
print('✅ transformers:', transformers.__version__)
print('✅ peft:', peft.__version__)
print('✅ fastapi:', fastapi.__version__)
print('✅ vllm: OK')
print('✅ redis: OK')
"

# ─────────────────────────────────────────
# SETUP ENV FILES
# ─────────────────────────────────────────
echo ""
echo "=== Setting up env files ==="

if [[ ! -f "$WORKSPACE/.env" ]]; then
  cp "$WORKSPACE/.env.example" "$WORKSPACE/.env"
  echo "⚠️  Created .env from template"
fi

for PROJECT in "${PROJECTS[@]}"; do
  ENV_FILE="$WORKSPACE/projects/$PROJECT/.env"
  EXAMPLE="$WORKSPACE/projects/$PROJECT/.env.example"

  if [[ -f "$EXAMPLE" && ! -f "$ENV_FILE" ]]; then
    cp "$EXAMPLE" "$ENV_FILE"
    echo "⚠️  Created projects/$PROJECT/.env"
  fi
done

# Ensure pm2 uses the venv interpreter
if grep -q "^PM2_PYTHON=" "$WORKSPACE/.env"; then
  sed -i "s|^PM2_PYTHON=.*|PM2_PYTHON=$WORKSPACE/venv/bin/python|" "$WORKSPACE/.env"
else
  echo "PM2_PYTHON=$WORKSPACE/venv/bin/python" >> "$WORKSPACE/.env"
fi

if grep -q "^WORKSPACE=" "$WORKSPACE/.env"; then
  sed -i "s|^WORKSPACE=.*|WORKSPACE=$WORKSPACE|" "$WORKSPACE/.env"
else
  echo "WORKSPACE=$WORKSPACE" >> "$WORKSPACE/.env"
fi

echo ""
echo "IMPORTANT: Fill in all .env files before continuing."
echo ""
echo "Root .env needs:"
echo "  DOMAIN=yourdomain.com"
echo "  TUNNEL_NAME=llm-api-tunnel"
echo "  CF_API_TOKEN=xxxx"
echo "  HF_TOKEN=hf_xxxx"
echo "  ALERT_FROM_EMAIL=you@gmail.com"
echo "  ALERT_TO_EMAIL=you@gmail.com"
echo "  ALERT_EMAIL_PASSWORD=xxxx"
echo ""
echo "Each project .env needs:"
echo "  CLIENT_API_KEY=xxxx"
echo "  ADMIN_API_KEY=xxxx"
echo ""
read -r -p "Press enter when .env files are ready..."

# ─────────────────────────────────────────
# DOWNLOAD BASE MODEL
# ─────────────────────────────────────────
echo ""
echo "=== Downloading Mistral 7B ==="
echo "This takes 10-15 minutes (~14GB)"
echo ""

set -a
# shellcheck disable=SC1091
source "$WORKSPACE/.env"
if [[ -f "$WORKSPACE/projects/clinical-notes/.env" ]]; then
  # shellcheck disable=SC1091
  source "$WORKSPACE/projects/clinical-notes/.env"
fi
set +a

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "❌ HF_TOKEN not set in .env or projects/clinical-notes/.env"
  echo "   Mistral 7B is gated — add your Hugging Face token."
  exit 1
fi

if [[ -d "$MODEL_DIR" && -f "$MODEL_DIR/config.json" ]]; then
  echo "✅ Mistral 7B already present at $MODEL_DIR"
else
  mkdir -p /workspace/models
  python3 -c "
from huggingface_hub import snapshot_download
import os

token = os.getenv('HF_TOKEN')
print('Downloading Mistral 7B...')
snapshot_download(
    repo_id='mistralai/Mistral-7B-v0.1',
    local_dir='$MODEL_DIR',
    token=token,
    ignore_patterns=['*.msgpack', '*.h5'],
)
print('✅ Mistral 7B downloaded')
"
fi

# ─────────────────────────────────────────
# UPDATE CONFIGS
# ─────────────────────────────────────────
echo ""
echo "=== Updating model paths ==="

for PROJECT in "${PROJECTS[@]}"; do
  CONFIG="$WORKSPACE/projects/$PROJECT/config.yaml"
  if [[ ! -f "$CONFIG" ]]; then
    echo "❌ Missing config: $CONFIG"
    exit 1
  fi
  sed -i "s|^base_model:.*|base_model: $MODEL_DIR|" "$CONFIG"
  echo "✅ Updated $PROJECT base_model"
done

# clinical-notes config ships with adapter_path pointing at lora/; training uses qlora/
sed -i "s|^adapter_path:.*|adapter_path: projects/clinical-notes/model/qlora/|" \
  "$WORKSPACE/projects/clinical-notes/config.yaml"
echo "✅ Updated clinical-notes adapter_path → model/qlora/"

# ─────────────────────────────────────────
# VERIFY SETUP
# ─────────────────────────────────────────
echo ""
echo "=== Verifying full setup ==="

redis-cli ping | grep -q PONG && echo "✅ Redis: OK" || echo "❌ Redis: FAILED"

python3 -c "
import torch
assert torch.cuda.is_available(), 'No GPU'
print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
mem = torch.cuda.get_device_properties(0).total_memory
print(f'✅ VRAM: {mem/1e9:.1f}GB')
"

[[ -d "$MODEL_DIR" && -f "$MODEL_DIR/config.json" ]] && \
  echo "✅ Mistral 7B: Downloaded" || \
  echo "❌ Mistral 7B: Missing"

for PROJECT in "${PROJECTS[@]}"; do
  [[ -d "$WORKSPACE/projects/$PROJECT" ]] && \
    echo "✅ Project $PROJECT: OK" || \
    echo "❌ Project $PROJECT: Missing"
done

echo ""
echo "=== Deployment ready ==="
echo "Next steps:"
echo "  1. Fine-tune:  bash scripts/train.sh"
echo "  2. Start APIs:   bash scripts/start_all.sh"
echo "  3. Setup HTTPS:  sudo bash scripts/setup_https.sh"
echo "  4. Production:   bash scripts/setup_pm2.sh (crash alerts + reboot persistence)"
echo "  5. Backups:      bash scripts/setup_backup.sh && bash scripts/setup_cron.sh"
