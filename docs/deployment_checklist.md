# Vast.ai Deployment Checklist

Complete deployment in one day. Check each box in order.

## Phase 1: Instance setup (~30 min)

- [ ] Read [vastai_instance_guide.md](vastai_instance_guide.md) and pick RTX 3090
- [ ] Created Vast.ai account with ≥ $50 credit
- [ ] Added SSH public key ([vastai_setup.md](vastai_setup.md))
- [ ] Rented PyTorch template, Ubuntu 22.04+, 150 GB disk
- [ ] SSH login works: `ssh -p PORT root@ssh.vast.ai`
- [ ] Ran `bash scripts/vastai_first_login.sh`
- [ ] `nvidia-smi` shows GPU
- [ ] `redis-cli ping` → `PONG`
- [ ] `node --version` and `pm2 --version` OK

## Phase 2: Deploy repo (~20 min)

- [ ] Cloned repo to `/workspace/llm-finetuning-lab`
- [ ] Ran `bash scripts/deploy.sh <github-url>`
- [ ] Python venv created; `requirements.txt` installed
- [ ] Root `.env` filled: `DOMAIN`, `TUNNEL_NAME`, `CF_API_TOKEN`, `HF_TOKEN`, alert emails
- [ ] `PM2_PYTHON=/workspace/llm-finetuning-lab/venv/bin/python` set in root `.env`
- [ ] Project `.env` files filled: `CLIENT_API_KEY`, `ADMIN_API_KEY` (each project)
- [ ] Mistral 7B downloaded to `/workspace/models/mistral-7b`
- [ ] All three `projects/*/config.yaml` point `base_model` at local Mistral path
- [ ] `clinical-notes` `adapter_path` → `projects/clinical-notes/model/qlora/`

## Phase 3: Fine-tune clinical-notes (2–4 hours)

- [ ] Ran `bash scripts/train.sh`
- [ ] Training running in tmux: `tmux attach -t training`
- [ ] GPU high utilization: `watch -n1 nvidia-smi`
- [ ] Loss decreasing in log: `grep loss projects/clinical-notes/logs/training_*.log | tail -20`
- [ ] Adapter saved: `projects/clinical-notes/model/qlora/adapter_config.json` exists
- [ ] If crash: see recovery section in [vastai_setup.md](vastai_setup.md)

## Phase 4: Start APIs (~10 min)

- [ ] Ran `bash scripts/start_all.sh`
- [ ] `pm2 status` — 3 apps online (clinical-notes, medical-coding, patient-support)
- [ ] `curl -s localhost:8001/health` → HTTP 200
- [ ] `curl -s localhost:8002/health` → HTTP 200
- [ ] `curl -s localhost:8003/health` → HTTP 200
- [ ] Ran `bash scripts/setup_pm2.sh` for crash listener + reboot instructions

## Phase 5: HTTPS (~20 min)

- [ ] Domain purchased and DNS on Cloudflare
- [ ] Root `.env`: `DOMAIN`, `CF_API_TOKEN` set
- [ ] Ran `sudo bash scripts/setup_https.sh`
- [ ] `systemctl is-active cloudflared` → `active`
- [ ] `https://clinical-notes.DOMAIN/health` → 200
- [ ] `https://medical-coding.DOMAIN/health` → 200
- [ ] `https://patient-support.DOMAIN/health` → 200

## Phase 6: Production ops (~20 min)

- [ ] Ran `bash scripts/setup_backup.sh` (GDrive + B2 configured)
- [ ] Ran `bash scripts/setup_cron.sh` (daily backup ~3 AM)
- [ ] Ran test backup: `bash scripts/backup.sh`
- [ ] UptimeRobot monitors for all 3 HTTPS `/health` URLs
- [ ] Test alert email received (crash or monitor script)
- [ ] Executed `pm2 startup` command printed by `setup_pm2.sh` (survives reboot)
- [ ] Vast.ai auto-top-up or low-balance alert enabled

## Phase 7: First client test (~10 min)

- [ ] POST test to `/generate` with `CLIENT_API_KEY`
- [ ] Poll `/result/{job_id}` until complete
- [ ] Valid clinical summary in response
- [ ] Record in `projects/clinical-notes/usage.db` (or project usage store)
- [ ] Entry in `projects/clinical-notes/logs/` serving logs

## Done

- [ ] System fully deployed
- [ ] HTTPS endpoints documented for client
- [ ] Backups verified: `bash scripts/verify_backup.sh`

---

## Quick reference — full day timeline

| Time | Step | Command |
|------|------|---------|
| 0:00 | First login setup | `bash scripts/vastai_first_login.sh` |
| 0:15 | Deploy | `bash scripts/deploy.sh <url>` |
| 0:35 | Train (background) | `bash scripts/train.sh` |
| 3:00 | Start APIs | `bash scripts/start_all.sh` |
| 3:15 | HTTPS | `sudo bash scripts/setup_https.sh` |
| 3:35 | Backups + cron | `setup_backup.sh`, `setup_cron.sh` |
| 4:00 | Client smoke test | curl `/generate` |

## Related docs

- [vastai_setup.md](vastai_setup.md) — crashes, resume, migration
- [daily_ops.md](daily_ops.md) — morning routine
- [cost_vs_revenue.md](cost_vs_revenue.md) — unit economics
