# Vast.ai Account Setup

One-time account and instance setup. Allow ~30 minutes before running repo scripts.

## Step 1: Create account

1. Go to [vast.ai](https://vast.ai)
2. Sign up with email
3. Add payment method (credit card or crypto)
4. Add **$50 credit** to start (~3–4 days of RTX 3090 at $0.30/hr)

> **Warning:** When credit hits $0, the instance **terminates** and local disk is **lost**. Enable auto-top-up or spending alerts in Vast.ai settings.

## Step 2: Add SSH key

1. Vast.ai → **Settings** → **SSH Keys**
2. Paste your public key:

```bash
cat ~/.ssh/id_rsa.pub
```

If you have no key:

```bash
ssh-keygen -t rsa -b 4096 -C "you@email.com"
cat ~/.ssh/id_rsa.pub
```

Copy the full line (`ssh-rsa AAAA...`) into Vast.ai.

## Step 3: Rent an instance

1. Open **Search** and apply filters from [vastai_instance_guide.md](vastai_instance_guide.md)
2. Sort by **DLPerf/Cost**
3. Click **Rent** on a reliable RTX 3090 offer
4. **Template:** PyTorch
5. **Disk:** 150 GB
6. Confirm **Rent Instance**

## Step 4: SSH into the instance

1. **Instances** → expand your instance (`>`)
2. Copy the SSH command, e.g.:

```bash
ssh -p 12345 root@ssh.vast.ai
```

3. On first login, run one-time host setup:

```bash
# Option A: after cloning the repo
bash scripts/vastai_first_login.sh

# Option B: curl script before clone (if repo is private, clone first)
```

Expected output: `nvidia-smi` shows GPU, Redis returns `PONG`, Node + pm2 installed.

## Step 5: Deploy the repo

```bash
cd /workspace
git clone https://github.com/YOU/llm-finetuning-lab.git llm-finetuning-lab
cd llm-finetuning-lab
bash scripts/deploy.sh https://github.com/YOU/llm-finetuning-lab.git
```

Follow prompts to fill `.env` files, then continue through [deployment_checklist.md](deployment_checklist.md).

---

## Troubleshooting

### SSH connection refused

- Instance may still be starting — wait 1–2 minutes
- Confirm port and host from the Vast.ai dashboard (port changes per instance)
- Verify your public key is saved in Vast.ai settings

### `nvidia-smi` not found

- You rented a CPU template — destroy and re-rent with **PyTorch + GPU**

### Out of disk during model download

- Rent with ≥ 150 GB disk, or expand volume if the host allows
- Base model alone is ~14 GB

---

## If training crashes midway

1. **Check the log** (timestamped file from `train.sh`):

```bash
tail -100 projects/clinical-notes/logs/training_*.log
```

2. **Common causes:**

| Error | Fix |
|-------|-----|
| CUDA OOM | Lower `batch_size` in `projects/clinical-notes/config.yaml`, restart training |
| SSH disconnect killed process | Always use `train.sh` (tmux) — re-run if you ran `python run.py` directly |
| HF token / 401 | Set `HF_TOKEN` in root `.env` |
| Disk full | `df -h` — free space or expand disk |

3. **Checkpoints:** Training saves checkpoints every 10 steps (default) under:

```text
projects/clinical-notes/model/qlora/checkpoint-*/
```

4. **Resume from latest checkpoint** (Hugging Face saves every 10 steps by default):

```bash
cd /workspace/llm-finetuning-lab
source venv/bin/activate

LATEST=$(ls -d projects/clinical-notes/model/qlora/checkpoint-* 2>/dev/null | sort -V | tail -1)
echo "Resuming from: $LATEST"

python3 <<PY
from pathlib import Path
from peft import get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from core.training.dataset_loader import prepare_datasets
from core.training.train_lora import _build_lora_config
from core.training.train_qlora import _build_bnb_config
from core.utils.config_loader import get_model_output_dir, load_project_config, load_prompt_template

project = "clinical-notes"
checkpoint = Path("$LATEST")
config = load_project_config(project)
tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
model = AutoModelForCausalLM.from_pretrained(
    config["base_model"],
    quantization_config=_build_bnb_config(config),
    device_map="auto",
)
model = get_peft_model(model, _build_lora_config(config))
train_ds, val_ds = prepare_datasets(config, load_prompt_template(project), tokenizer)
output_dir = get_model_output_dir(config, "qlora")
args = TrainingArguments(
    output_dir=str(output_dir),
    per_device_train_batch_size=config["training"]["batch_size"],
    num_train_epochs=config["training"]["epochs"],
    learning_rate=config["training"]["learning_rate"],
    logging_steps=1,
    save_steps=10,
    fp16=True,
)
trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds)
trainer.train(resume_from_checkpoint=str(checkpoint))
model.save_pretrained(str(output_dir))
print("Done. Adapters in", output_dir)
PY
```

Or restart from scratch (2–4 hours on RTX 3090):

```bash
bash scripts/train.sh
```

---

## If the instance is terminated unexpectedly

Vast.ai deletes the local disk when credit runs out or you stop the instance.

**What you lose:** anything not backed up (adapters, `.env`, usage DBs, local logs).

**What you keep:** backups on Google Drive / Backblaze B2 (if `setup_backup.sh` was configured).

### Recovery steps

1. Rent a **new** instance (same spec guide)
2. Run `vastai_first_login.sh` → `deploy.sh`
3. Restore from backup:

```bash
bash scripts/restore.sh gdrive all latest
# or
bash scripts/restore.sh b2 clinical-notes latest
```

4. Re-fill secrets in `.env` if not in backup (API keys, `CF_API_TOKEN`)
5. Re-run `setup_https.sh`, `setup_cron.sh`, UptimeRobot monitors

---

## Migrate to a new instance (planned move)

Use this when upgrading GPU or moving regions without losing data.

### Before shutdown (old instance)

```bash
cd /workspace/llm-finetuning-lab
bash scripts/backup.sh
bash scripts/verify_backup.sh
pm2 save
```

Confirm archives exist on GDrive/B2:

```bash
rclone ls gdrive:llm-finetuning-backups/daily/ | tail -5
```

### On new instance

```bash
bash scripts/vastai_first_login.sh
bash scripts/deploy.sh https://github.com/YOU/llm-finetuning-lab.git
bash scripts/restore.sh gdrive all latest
bash scripts/start_all.sh
sudo bash scripts/setup_https.sh
bash scripts/setup_cron.sh
```

Copy `.env` values from your password manager — never commit them to git.

---

## How to tell training is actually learning

Healthy QLoRA runs show **decreasing loss** over time.

```bash
# Live tail
tail -f projects/clinical-notes/logs/training_*.log

# Extract loss lines
grep -E "'loss'|train_loss" projects/clinical-notes/logs/training_*.log | tail -30
```

**Good signs:**

- `loss` drops over the first 50–100 steps (e.g. 2.5 → 1.2 → 0.8)
- GPU util 80–100% in `nvidia-smi`
- Checkpoints appearing under `model/qlora/checkpoint-*`

**Bad signs:**

- Loss flat or increasing after many steps → check learning rate, dataset path, label formatting
- GPU util near 0% → process crashed; attach tmux or read log
- No checkpoints after 15+ minutes → training not running

After training completes, verify adapter files:

```bash
ls -la projects/clinical-notes/model/qlora/
# Expect: adapter_config.json, adapter_model.safetensors (or .bin)
```
