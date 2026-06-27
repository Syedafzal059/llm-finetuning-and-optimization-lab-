#!/usr/bin/env bash
set -e

WORKSPACE="${WORKSPACE:-/workspace/llm-finetuning-lab}"
PROJECT="clinical-notes"
SESSION="training"

cd "$WORKSPACE"

if [[ ! -d "$WORKSPACE/venv" ]]; then
  echo "❌ Virtualenv not found. Run scripts/deploy.sh first."
  exit 1
fi

if [[ ! -f "$WORKSPACE/projects/$PROJECT/config.yaml" ]]; then
  echo "❌ Project config missing: projects/$PROJECT/config.yaml"
  exit 1
fi

python3 -c "import torch; assert torch.cuda.is_available()" || {
  echo "❌ No CUDA GPU available"
  exit 1
}

if ! command -v tmux >/dev/null 2>&1; then
  echo "❌ tmux not installed. Run scripts/vastai_first_login.sh first."
  exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "⚠️  tmux session '$SESSION' already exists."
  echo "   Attach:  tmux attach -t $SESSION"
  echo "   Or kill: tmux kill-session -t $SESSION"
  exit 1
fi

LOG_FILE="projects/$PROJECT/logs/training_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "projects/$PROJECT/logs"

echo "=== Starting fine-tuning ==="
echo "Project: $PROJECT"
echo "Mode: QLoRA"
echo "Model: Mistral 7B (from project config)"
echo ""
echo "Estimated time: 2-4 hours"
echo "Monitor GPU: watch -n1 nvidia-smi"
echo "Log file: $LOG_FILE"
echo ""

tmux new-session -d -s "$SESSION" -x 220 -y 50
tmux send-keys -t "$SESSION" \
  "cd $WORKSPACE && source venv/bin/activate && python run.py --project $PROJECT --mode qlora 2>&1 | tee $LOG_FILE" \
  Enter

echo "✅ Training started in tmux session '$SESSION'"
echo ""
echo "Watch progress:"
echo "  tmux attach -t $SESSION"
echo ""
echo "Watch GPU:"
echo "  watch -n1 nvidia-smi"
echo ""
echo "View logs:"
echo "  tail -f $LOG_FILE"
echo ""
echo "Check loss is decreasing:"
echo "  grep -E 'loss|train_loss' $LOG_FILE | tail -20"
