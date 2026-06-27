# Vast.ai Instance Guide

Choose the right GPU instance before you deploy. Wrong specs waste money or block training entirely.

## Minimum specs for this setup

| Resource | Minimum | Why |
|----------|---------|-----|
| GPU | RTX 3090 (24 GB VRAM) or better | Mistral 7B 4-bit ~5 GB + 3 LoRA adapters ~18 GB + queue buffer |
| System RAM | 64 GB | Data loading, vLLM host memory, Redis, OS headroom |
| Disk | 150 GB+ | Base model, adapters, datasets, logs, backups buffer |
| CUDA | 11.8+ | PyTorch + bitsandbytes + vLLM compatibility |
| OS | Ubuntu 22.04 or 24.04 | Matches repo scripts and Cloudflare packages |

### Disk budget (~150 GB)

| Item | Size |
|------|------|
| Mistral 7B base model | ~14 GB |
| 3 QLoRA adapters | ~500 MB |
| Datasets | ~1 GB |
| Logs + usage DBs | ~5 GB |
| OS + Python + libs | ~30 GB |
| Free buffer | ~100 GB |

## Recommended instances (cheapest first)

### Option 1: RTX 3090 × 1 (start here)

- **Cost:** ~$0.20–0.35/hr (~$150–250/month always-on)
- **VRAM:** 24 GB
- **Best for:** 1–3 clients, first production deployment
- **Verdict:** Cheapest path that fits this repo today

### Option 2: RTX 4090 × 1

- **Cost:** ~$0.40–0.60/hr (~$290–430/month)
- **VRAM:** 24 GB (faster training + inference)
- **Best for:** Faster QLoRA runs, lower latency per request

### Option 3: A100 × 1

- **Cost:** ~$1.50–2.50/hr (~$1,080–1,800/month)
- **VRAM:** 40–80 GB
- **Best for:** 5+ clients at high concurrent load

**Recommendation:** Start with RTX 3090. Upgrade when revenue covers the delta.

## Filters on vast.ai

Use these on the **Search** page:

1. **GPU:** RTX 3090 (or RTX 4090 if budget allows)
2. **CUDA:** ≥ 11.8
3. **Disk:** ≥ 150 GB (set at rent time if template allows)
4. **RAM:** ≥ 64 GB
5. **Sort by:** DLPerf / $ (performance per dollar)
6. **Reliability:** ≥ 99%
7. **Internet:** ≥ 500 Mbps up/down (HTTPS tunnel + model download)
8. **Region:** US (lower latency for US clients)

## Template selection

When renting:

- **Template:** PyTorch (CUDA pre-installed)
- **Disk:** 150 GB minimum
- **Image:** Ubuntu 22.04 or 24.04

Avoid CPU-only templates — `nvidia-smi` must show your GPU on first login.

## When to upgrade

| Signal | Action |
|--------|--------|
| GPU util consistently > 90% with queue backlog | Move to RTX 4090 or A100 |
| OOM errors in pm2 logs during serving | Reduce `gpu_memory_utilization` in project config, or upgrade GPU |
| Fine-tune runs > 6 hours | RTX 4090 or larger batch-friendly GPU |
| 5+ paying clients | A100 or multi-GPU instance |

## Related docs

- Account and SSH: [vastai_setup.md](vastai_setup.md)
- Full checklist: [deployment_checklist.md](deployment_checklist.md)
- Daily ops: [daily_ops.md](daily_ops.md)
