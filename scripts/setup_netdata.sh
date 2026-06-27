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
GPU_WARN="${MONITOR_GPU_WARNING_PCT:-85}"
GPU_CRIT="${MONITOR_GPU_CRITICAL_PCT:-95}"

echo "=== Installing Netdata ==="
bash <(curl -Ss https://my-netdata.io/kickstart.sh) --non-interactive

echo ""
echo "=== Netdata running at ==="
echo "http://YOUR_SERVER_IP:19999"
echo ""
echo "=== Enable GPU monitoring ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi not found — install NVIDIA drivers for GPU charts."
fi

echo ""
echo "=== Set up Netdata alerts ==="
mkdir -p /etc/netdata/health.d
cat > /etc/netdata/health.d/gpu.conf <<EOF
alarm: gpu_memory_high
on: nvidia_smi.gpu_mem_utilization
lookup: average -5m
every: 1m
warn: \$this > ${GPU_WARN}
crit: \$this > ${GPU_CRIT}
info: GPU memory utilization is high
to: sysadmin
EOF

if systemctl is-active --quiet netdata; then
  systemctl restart netdata
  echo "Netdata service restarted."
else
  echo "Netdata installed. Start it with: systemctl start netdata"
fi

echo ""
echo "Netdata configured."
echo "Workspace: $WORKSPACE"
echo "GPU warn/crit thresholds: ${GPU_WARN}% / ${GPU_CRIT}%"
