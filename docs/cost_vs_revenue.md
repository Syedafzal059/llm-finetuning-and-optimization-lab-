# Cost vs Revenue Tracker

Simple unit economics for a single RTX 3090 deployment serving three clients.

## Monthly costs

| Item | Cost |
|------|------|
| RTX 3090 Vast.ai (always on, ~$0.28/hr) | ~$200 |
| Domain (annual ÷ 12) | ~$1 |
| Cloudflare tunnel | Free |
| Backblaze B2 (adapters + archives) | ~$0.01 |
| Google Drive (backup copy) | Free tier |
| UptimeRobot | Free tier |
| Redis (on same server) | Free |
| **Total** | **~$201/month** |

> Adjust GPU row if you upgrade to RTX 4090 (~$290–430/mo) or A100 (~$1,080+/mo).

## Break-even

| Scenario | Revenue | Cost | Profit |
|----------|---------|------|--------|
| 1 client @ $500/mo | $500 | $201 | **$299** |

First paying client covers infrastructure with margin.

## Revenue at scale (same $201 infra)

| Clients | Revenue | Cost | Profit |
|---------|---------|------|--------|
| 1 | $500 | $201 | $299 |
| 3 | $1,500 | $201 | $1,299 |
| 5 | $7,500 | $201 | $7,299 |
| 11 | $27,500 | $201 | $27,299 |

Higher tiers assume expanded per-client pricing or add-on services. At 3+ clients running simultaneously at high load, upgrade GPU when revenue covers it ([vastai_instance_guide.md](vastai_instance_guide.md)).

## Cost control on Vast.ai

1. **Auto-top-up** or alert at $40 remaining — instance dies at $0
2. **Stop instance** when not demoing (only if backups are current)
3. **Spot vs on-demand** — cheaper but higher interrupt risk; not recommended for production APIs without failover
4. Track hourly burn: Vast.ai dashboard → instance → $/hr × 730 ≈ monthly

## When to upgrade GPU

| Trigger | Upgrade |
|---------|---------|
| Queue depth often > 10 | RTX 4090 or tune `gpu_memory_utilization` |
| Training blocks serving for hours | Separate train instance, or faster GPU |
| 5+ clients, SLA requirements | A100 or dedicated provider |

## Related docs

- [vastai_instance_guide.md](vastai_instance_guide.md)
- [daily_ops.md](daily_ops.md)
