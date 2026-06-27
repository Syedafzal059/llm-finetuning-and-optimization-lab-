#!/usr/bin/env bash
set -e

echo "=== Vast.ai First Login Setup ==="
echo "This takes about 10-15 minutes"
echo ""

if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
  echo "❌ Run as root: sudo bash scripts/vastai_first_login.sh"
  exit 1
fi

# ─────────────────────────────────────────
# SYSTEM UPDATE
# ─────────────────────────────────────────
echo "=== Updating system ==="
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y \
  git \
  curl \
  wget \
  vim \
  htop \
  tmux \
  screen \
  unzip \
  lsof \
  net-tools \
  python3-pip \
  python3-venv \
  redis-server
echo "✅ System updated"

# ─────────────────────────────────────────
# VERIFY GPU
# ─────────────────────────────────────────
echo ""
echo "=== Verifying GPU ==="
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "❌ nvidia-smi not found. Rent a GPU template (PyTorch) on Vast.ai."
  exit 1
fi
nvidia-smi
echo "✅ GPU verified"

# ─────────────────────────────────────────
# INSTALL NODE + NPM (for pm2)
# ─────────────────────────────────────────
echo ""
echo "=== Installing Node.js ==="
if ! command -v node >/dev/null 2>&1; then
  curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
  apt-get install -y nodejs
else
  echo "Node already installed: $(node --version)"
fi
echo "✅ Node $(node --version)"
echo "✅ npm $(npm --version)"

# ─────────────────────────────────────────
# INSTALL PM2
# ─────────────────────────────────────────
echo ""
echo "=== Installing pm2 ==="
if ! command -v pm2 >/dev/null 2>&1; then
  npm install -g pm2
else
  echo "pm2 already installed"
fi
echo "✅ pm2 $(pm2 --version)"

# ─────────────────────────────────────────
# REDIS
# ─────────────────────────────────────────
echo ""
echo "=== Starting Redis ==="
systemctl enable redis-server
systemctl start redis-server
if ! redis-cli ping | grep -q PONG; then
  echo "❌ Redis failed to start"
  exit 1
fi
echo "✅ Redis running"

# ─────────────────────────────────────────
# INSTALL RCLONE (for backups)
# ─────────────────────────────────────────
echo ""
echo "=== Installing rclone ==="
if ! command -v rclone >/dev/null 2>&1; then
  curl https://rclone.org/install.sh | bash
else
  echo "rclone already installed"
fi
echo "✅ rclone $(rclone --version | head -1)"

# ─────────────────────────────────────────
# VERIFY CUDA / PyTorch
# ─────────────────────────────────────────
echo ""
echo "=== Verifying CUDA ==="
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
else:
    raise SystemExit('CUDA not available — check GPU template')
"
echo "✅ CUDA verified"

# ─────────────────────────────────────────
# SET WORKSPACE
# ─────────────────────────────────────────
echo ""
echo "=== Setting workspace ==="
mkdir -p /workspace
WORKSPACE="/workspace/llm-finetuning-lab"
if ! grep -q "WORKSPACE=/workspace/llm-finetuning-lab" ~/.bashrc 2>/dev/null; then
  echo "export WORKSPACE=$WORKSPACE" >> ~/.bashrc
fi
export WORKSPACE
echo "✅ Workspace set to $WORKSPACE"

echo ""
echo "=== First login setup complete ==="
echo "Next: clone repo and run deploy.sh"
echo "  cd /workspace"
echo "  git clone <your-repo-url> llm-finetuning-lab"
echo "  cd llm-finetuning-lab"
echo "  bash scripts/deploy.sh"
