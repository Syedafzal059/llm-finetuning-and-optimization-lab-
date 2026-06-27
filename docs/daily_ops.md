# Daily Management on Vast.ai

~2 minutes each morning. Prevents surprise downtime and credit exhaustion.

## Morning check

```bash
ssh -p PORT root@ssh.vast.ai
cd /workspace/llm-finetuning-lab
pm2 status
curl -s localhost:8001/health | python3 -m json.tool
```

## GPU

```bash
nvidia-smi
```

Watch for:

- Persistent 100% util + growing queue → consider upgrade ([vastai_instance_guide.md](vastai_instance_guide.md))
- OOM or processes missing from `pm2 status`

## Logs

```bash
pm2 logs --lines 20
tail -5 logs/backup.log
```

## Restart one client

```bash
pm2 restart clinical-notes
pm2 logs clinical-notes --lines 30
```

## Restart all APIs

```bash
pm2 restart all
```

## View live requests

```bash
tmux attach -t main   # optional persistent shell session
pm2 logs clinical-notes
```

## SSH disconnect safely

Long-running jobs must use tmux so they survive disconnect:

```bash
tmux new-session -d -s main
tmux attach -t main
# Detach: Ctrl+B then D
```

Training uses session `training` (created by `scripts/train.sh`).

## Vast.ai instance management

| Rule | Why |
|------|-----|
| Never let credit hit $0 | Instance **terminates**; local disk **lost** |
| Enable auto-top-up or $40 alert | Avoid Friday-night outages |
| Run daily backups before experiments | `bash scripts/backup.sh` |
| Verify off-site copy weekly | `bash scripts/verify_backup.sh` |

## Training health (during fine-tune)

```bash
tmux attach -t training
grep -E "'loss'|train_loss" projects/clinical-notes/logs/training_*.log | tail -15
```

Loss should trend down over the first hour. Flat loss → see [vastai_setup.md](vastai_setup.md).

## Quick recovery commands

| Problem | Command |
|---------|---------|
| Redis down | `sudo systemctl restart redis-server` |
| Tunnel down | `sudo systemctl restart cloudflared` |
| One API stuck | `pm2 restart clinical-notes` |
| Full redeploy from backup | `bash scripts/restore.sh gdrive all latest` |

## Related docs

- [deployment_checklist.md](deployment_checklist.md)
- [vastai_setup.md](vastai_setup.md) — crash, resume, migration
- [cost_vs_revenue.md](cost_vs_revenue.md)
