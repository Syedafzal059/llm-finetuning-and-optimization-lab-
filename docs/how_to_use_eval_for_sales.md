# How to Use Evaluation Reports for Client Sales

After running evaluation, you have a client-ready report at:

```
projects/{project-name}/logs/eval_report.md
```

Run evaluation after training:

```bash
python run.py --project clinical-notes --mode eval
```

Compare a specific adapter version:

```bash
python run.py --project clinical-notes --mode eval --adapter v2
```

Read the report:

```bash
cat projects/clinical-notes/logs/eval_report.md
```

---

## 1. During a Sales Call

Share your screen and open `eval_report.md`.

Example pitch:

> "Here is our base model vs fine-tuned on your type of data. ROUGE-1 improved from 0.18 to 0.67 — that's 272% better at summarization on held-out clinical notes."

Point to the **Summary** table and **Example Outputs** sections. Real side-by-side outputs are more convincing than abstract benchmark numbers.

---

## 2. In Cold Outreach Email

Use concrete metrics from the report:

> "We fine-tuned Mistral 7B on clinical notes and achieved ROUGE-1 of 0.67 vs 0.18 for the base model — 3.7× better at medical summarization on our validation set."

Attach or link `eval_report.md` as proof.

---

## 3. On GitHub README

Add a metrics table to your project README:

| Metric | Base Model | Fine-tuned | Improvement |
|--------|-----------|------------|-------------|
| ROUGE-1 | 0.18 | 0.67 | +272% |
| BLEU | 0.07 | 0.48 | +586% |
| Perplexity | 45.2 | 8.3 | -82% |

This signals technical credibility to prospects browsing your repository.

---

## 4. After Client Onboarding

Re-run evaluation on **their** data:

1. Replace or augment `projects/{client}/data/raw/` with client samples.
2. Fine-tune: `python run.py --project {client} --mode qlora`
3. Evaluate: `python run.py --project {client} --mode eval`

Deliver:

> "Here is your model's performance on your own clinical notes."

Client-specific numbers are far more persuasive than generic benchmarks.

---

## 5. Via the Serving API

If the serving layer is running, admins can fetch reports programmatically:

```bash
# JSON summary
curl -H "X-API-Key: $ADMIN_KEY" https://your-api.example.com/eval-report

# Full markdown report
curl -H "X-API-Key: $ADMIN_KEY" https://your-api.example.com/eval-report/markdown
```

Use the JSON endpoint in dashboards; share the markdown endpoint with stakeholders.

---

## Verdict Labels

The API returns a `verdict` field:

| Verdict | Meaning |
|---------|---------|
| `production_ready` | Strong improvement on ROUGE and perplexity |
| `promising` | Meaningful gains; may need more validation |
| `needs_improvement` | Fine-tuned model did not clearly beat base |
| `adapter_missing` | No adapter found; re-train before sharing |

Only share reports with `production_ready` or `promising` verdicts externally unless you're setting expectations for iterative improvement.
