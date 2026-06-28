# LLM Fine-Tuning & Optimization Lab

**Reusable multi-project template: one core engine, many client projects—config-driven SFT, LoRA, QLoRA training, local inference, and FastAPI + vLLM serving.**

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-transformers-orange.svg)](https://pytorch.org/)

---

## Overview

This repository is an **end-to-end Hugging Face Transformers pipeline** restructured as a **multi-project template**. The **core engine** (`core/`) never changes. Each **client project** is a folder under `projects/` that only defines:

- `config.yaml` — model, training, data paths, serving settings
- `prompt_template.py` — task-specific `format_prompt(sample)` function
- `data/raw/` — JSON training dataset

**Problem it solves:** Teams need a repeatable path from “model on Hub” to **domain-tuned behavior** across many use cases without forking the pipeline each time.

**Why it matters:** In production, full fine-tunes rarely scale. **PEFT LoRA/QLoRA**, rigorous data formatting, and a config-first layout are how most systems ship updates. This template encodes that workflow once and makes each new domain a thin configuration layer.

---

## Key Features

| Area | What you get |
|------|----------------|
| **Multi-project layout** | `core/` engine + `projects/{name}/` per client |
| **Universal CLI** | `python run.py --project X --mode {sft,lora,qlora,inference,serve,eval}` |
| **Config merge** | `base_config.yaml` + project `config.yaml` with deep override |
| **Prompt templates** | Each project defines `format_prompt()`; core loader calls it automatically |
| **SFT / LoRA / QLoRA** | Hugging Face Trainer + PEFT + bitsandbytes, project-aware paths |
| **Local inference** | Load base model + adapters from `projects/{name}/model/` |
| **Serving layer** | FastAPI + vLLM, API key auth, rate limiting, health + metrics |
| **Centralized logging** | `projects/{name}/logs/training.log` and `serving.log` |
| **Post-training evaluation** | ROUGE, BLEU, perplexity — base vs fine-tuned reports |
| **No hardcoded paths** | All paths resolved from config relative to repo root |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        run.py                               │
│   --project clinical-notes  --mode qlora | inference | serve│
└──────────────────────────┬──────────────────────────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                 ▼
  core/training/    core/inference/    core/serving/
  SFT, LoRA, QLoRA  generate.py        FastAPI + vLLM
         │                 │                 │
         └────────┬────────┴────────┬────────┘
                  ▼                 ▼
         projects/clinical-notes/
         ├── config.yaml
         ├── prompt_template.py
         ├── download_data.py         # Fetch medical training data from Hugging Face
         ├── data/raw/sample.json
         ├── model/{sft,lora,qlora}/
         └── logs/{training,serving}.log
```

**End-to-end flow:**

1. **Configure** — Set global defaults in `base_config.yaml`, override per project in `projects/{name}/config.yaml`.
2. **Format data** — `core/training/dataset_loader.py` loads JSON and calls the project's `format_prompt()`.
3. **Train** — SFT (full weights), LoRA (adapters), or QLoRA (4-bit base + adapters) via `run.py`.
4. **Infer** — Local generation with optional adapter weights from `projects/{name}/model/`.
5. **Serve** — FastAPI + vLLM loads the project model and adapter; logs to `projects/{name}/logs/serving.log`.

---

## Project Structure

```
llm-finetuning-and-optimization-lab/
├── core/                              # Reusable engine (never changes per client)
│   ├── training/
│   │   ├── dataset_loader.py          # Generic JSON loader, prompt-template aware
│   │   ├── train_sft.py               # Accepts --project
│   │   ├── train_lora.py              # Accepts --project
│   │   └── train_qlora.py             # Accepts --project
│   ├── serving/
│   │   ├── api.py                     # create_app(project_name)
│   │   ├── model.py                   # vLLM loader with LoRA adapter
│   │   ├── schemas.py                 # Pydantic request/response models
│   │   └── config.yaml                # Serving defaults
│   ├── inference/
│   │   ├── model_loader.py            # Base + PeftModel loading
│   │   └── generate.py                # Tokenize, generate, decode
│   └── utils/
│       ├── config_loader.py           # base + project config merge
│       └── logger.py                  # Console + file logging
├── projects/
│   └── clinical-notes/                # Example: medical note summarization
│       ├── config.yaml
│       ├── prompt_template.py
│       ├── download_data.py           # Download medical flashcard dataset
│       └── data/raw/sample.json
├── base_config.yaml                   # Global defaults
├── run.py                             # Universal entry point
├── MIGRATION_NOTES.txt                # Old layout → new layout mapping
├── .env.example                       # API keys and HF token template
└── requirements.txt
```

Each project also creates these directories at runtime (gitignored):

```
projects/clinical-notes/
├── model/
│   ├── sft/         # Full fine-tune checkpoints
│   ├── lora/        # LoRA adapter weights
│   └── qlora/       # QLoRA adapter weights
└── logs/
    ├── training.log
    └── serving.log
```

---

## Setup

**Prerequisites:** Python 3.10+, [PyTorch](https://pytorch.org/) matching your OS/CUDA, and a Hugging Face account/token for gated models (Mistral, Llama, etc.). **QLoRA** requires [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) with a **CUDA** GPU. **vLLM serving** also requires CUDA.

```bash
cd llm-finetuning-and-optimization-lab
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

Copy environment template and fill in secrets (never commit `.env`):

```bash
cp .env.example .env   # Linux/macOS
copy .env.example .env # Windows
```

---

## Google Colab (GPU Training)

Use Colab for **QLoRA on a free GPU** without a local CUDA setup. Full workflow for the `clinical-notes` project:

### 1. Setup runtime and clone

```python
# Optional: persist adapters to Drive
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/SyedAfzal059/llm-finetuning-and-optimization-lab-.git
%cd /content/llm-finetuning-and-optimization-lab-
```

### 2. Install dependencies

```python
!pip install -q -r requirements.txt
```

`requirements.txt` includes `bitsandbytes` and `torchao>=0.16.0` for QLoRA. Colab may show pip conflict warnings for preinstalled packages (`cuml`, `gradio`, etc.) — usually safe to ignore.

Optional `torchao` messages like `Failed to load .../_C_mxfp8...` mean optional CUDA extensions are missing; **training still works**.

### 3. (Optional) Hugging Face token

```python
import os
os.environ["HF_TOKEN"] = "hf_your_token_here"
```

### 4. Download training data

**Run before training** to replace the bundled sample with 500 real medical Q&A examples:

```python
!python projects/clinical-notes/download_data.py --limit 500
```

Writes to `projects/clinical-notes/data/raw/sample.json` (path from `config.yaml`). Verify:

```python
import json
with open("projects/clinical-notes/data/raw/sample.json") as f:
    print(len(json.load(f)), "examples")  # expect 500
```

Training should log `Map: 450/450` (train) and `Map: 50/50` (val). If you see `7/7`, the data step did not run.

### 5. Colab config overrides (recommended)

Patch config for GPU + QLoRA so eval finds the right adapter path:

```python
import yaml
from pathlib import Path

cfg_path = Path("projects/clinical-notes/config.yaml")
cfg = yaml.safe_load(cfg_path.read_text())
cfg["model"]["device"] = "cuda"
cfg["training"]["mode"] = "qlora"
cfg_path.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
```

### 6. Train, evaluate, and save

```python
!python run.py --project clinical-notes --mode qlora
!python run.py --project clinical-notes --mode eval
# If training.mode is still "lora": --adapter qlora
```

```python
# Optional: copy adapters to Drive before session ends
import shutil
from pathlib import Path
shutil.copytree(
    "projects/clinical-notes/model/qlora",
    "/content/drive/MyDrive/clinical-notes-qlora",
    dirs_exist_ok=True,
)
```

### Colab troubleshooting

| Symptom | Fix |
|---------|-----|
| `Map: 7/7` train samples | Re-run `download_data.py`; confirm 500 rows in `sample.json` |
| Loss stuck ~12 | Same — old 8-row sample still in place |
| Eval loads wrong adapter | Set `training.mode: qlora` or use `--adapter qlora` |
| QLoRA falls back to CPU LoRA | Runtime → Change runtime type → T4 GPU |

---

## Usage — clinical-notes Example

The **clinical-notes** project fine-tunes a model on medical text. Download real training data before your first run (see [Download training data](#download-training-data) below).

### Download training data

Fetches **medical flashcards** from Hugging Face (`medalpaca/medical_meadow_medical_flashcards`), converts them to `{clinical_note, summary}` JSON, and saves to the path in `config.yaml`:

```bash
python projects/clinical-notes/download_data.py
```

| Flag | Default | Description |
|------|---------|-------------|
| `--limit` | `500` | Max examples after filtering (`0` = all) |
| `--output` | from config | Override output JSON path |
| `--min-note-len` | `50` | Minimum input length (chars) |
| `--min-summary-len` | `20` | Minimum output length (chars) |

Examples:

```bash
python projects/clinical-notes/download_data.py --limit 500
python projects/clinical-notes/download_data.py --limit 0 --output projects/clinical-notes/data/raw/full.json
```

The default dataset is medical **Q&A**, while `prompt_template.py` asks for SOAP summarization. Training works, but aligning the instruction text with your data improves results.

### Train with QLoRA

```bash
python run.py --project clinical-notes --mode qlora
```

Saves adapters to `projects/clinical-notes/model/qlora/`. Logs to `projects/clinical-notes/logs/training.log`.

### Train SFT or LoRA

```bash
python run.py --project clinical-notes --mode sft
python run.py --project clinical-notes --mode lora
```

You can also invoke training scripts directly:

```bash
python core/training/train_qlora.py --project clinical-notes
```

### Local inference

```bash
python run.py --project clinical-notes --mode inference --prompt "Patient presented with chest pain and shortness of breath."
```

Loads adapters from `projects/clinical-notes/model/qlora/` when `adapter_config.json` exists; otherwise uses the base model only.

### Evaluate (base vs fine-tuned)

```bash
python run.py --project clinical-notes --mode eval
```

Writes `projects/clinical-notes/logs/eval_report.json` and `eval_report.md` with ROUGE, BLEU, and perplexity comparisons. Compare a specific adapter version with `--adapter v2`.

### Start API server

```bash
python run.py --project clinical-notes --mode serve
```

Server binds to the port in project config (default `8000`). Logs to `projects/clinical-notes/logs/serving.log`.

| Endpoint | Auth | Description |
|----------|------|-------------|
| `GET /health` | No | Health check, uptime, GPU info |
| `GET /metrics` | `X-API-Key` | Total requests, latency, GPU memory, RPM |
| `POST /generate` | `X-API-Key` | Text generation (optional streaming) |
| `POST /fine-tune-status` | `X-API-Key` | Model/adapter readiness |

Example request:

```bash
curl -X POST http://localhost:8000/generate \
  -H "X-API-Key: client-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Summarize: Patient with fever and productive cough...", "max_tokens": 256}'
```

---

## Configuration

Configuration is **two-layer**: `base_config.yaml` provides global defaults; `projects/{name}/config.yaml` overrides them. `core/utils/config_loader.py` deep-merges both into a single object passed everywhere.

### Full project config example

```yaml
project_name: clinical-notes
base_model: mistralai/Mistral-7B-v0.1

model:
  device: cuda
  max_new_tokens: 512

training:
  mode: qlora              # sft | lora | qlora
  epochs: 3
  batch_size: 4
  learning_rate: 2.0e-4
  max_length: 512
  lora:
    rank: 8
    alpha: 16
    target_modules: [q_proj, v_proj]
  qlora:
    bits: 4
    double_quant: true
    quant_type: nf4

data:
  train_path: projects/clinical-notes/data/raw/sample.json
  val_split: 0.1

serving:
  port: 8000
  max_tokens: 512
  temperature: 0.7
  rate_limit: 10/minute
  timeout_seconds: 60

inference:
  temperature: 0.7
  top_p: 0.9
```

Global defaults (TinyLlama, LoRA rank/alpha, vLLM settings, auth env var name) live in `base_config.yaml` and apply when a project field is omitted.

---

## Prompt Templates

Each project must define `format_prompt(sample: dict) -> str` in `prompt_template.py`. The core dataset loader imports and calls it for every training row—no changes to `core/` needed for new task formats.

**clinical-notes** example:

```python
def format_prompt(sample: dict) -> str:
    return f"""### Instruction:
Summarize the following clinical note into a structured SOAP format.

### Input:
{sample['clinical_note']}

### Response:
{sample['summary']}"""
```

Training data is a JSON array. Field names must match what your `format_prompt()` expects (`clinical_note` / `summary` for this project).

---

## Adding a New Project

1. Create the project folder:

```
projects/my-project/
├── config.yaml
├── prompt_template.py
└── data/raw/my_data.json
```

2. Set `project_name`, `base_model`, `data.train_path`, and training settings in `config.yaml`.

3. Implement `format_prompt()` in `prompt_template.py`.

4. Train and serve:

```bash
python run.py --project my-project --mode lora
python run.py --project my-project --mode inference --prompt "Your test prompt"
python run.py --project my-project --mode serve
```

No changes to `core/` are required.

---

## Command Reference

| Task | Command |
|------|---------|
| Download training data | `python projects/clinical-notes/download_data.py --limit 500` |
| QLoRA training | `python run.py --project clinical-notes --mode qlora` |
| LoRA training | `python run.py --project clinical-notes --mode lora` |
| SFT training | `python run.py --project clinical-notes --mode sft` |
| Local inference | `python run.py --project clinical-notes --mode inference --prompt "..."` |
| Evaluate model | `python run.py --project clinical-notes --mode eval` |
| Start API | `python run.py --project clinical-notes --mode serve` |

### Migration from the old single-project layout

| Old | New |
|-----|-----|
| `python run.py` | `python run.py --project clinical-notes --mode inference` |
| `python src/training/train_lora.py` | `python run.py --project clinical-notes --mode lora` |
| `python run_lora.py` | `python run.py --project clinical-notes --mode inference` |
| `configs.py/base.yaml` | `base_config.yaml` + `projects/{name}/config.yaml` |
| `model/lora/` | `projects/{name}/model/lora/` |

See **MIGRATION_NOTES.txt** for the complete file mapping.

---

## Results & Insights

| Metric | Insight |
|--------|---------|
| **Trainable params (LoRA)** | ~**0.1%** of base model (exact figure from `print_trainable_parameters()` at run time) |
| **Memory / iteration cost** | LoRA trains small matrices; **QLoRA** cuts base-model memory via 4-bit weights; full SFT updates all weights |
| **Quality** | SFT and LoRA both align the model to your **prompt format**; LoRA trades some flexibility for efficiency |
| **Artifacts** | LoRA/QLoRA exports are **MB-scale adapters** vs multi-GB full checkpoints |
| **Multi-project gain** | New domains require only config + data + prompt template—core engine stays stable |

---

## Engineering Learnings

- **Separate engine from project:** Stable `core/` + thin `projects/` folders scales better than one monolithic config.
- **Data formatting is half the product:** Instruction templates and label alignment (`labels = input_ids`) make or break causal LM fine-tunes.
- **Config merge beats duplication:** Global defaults in `base_config.yaml` with per-project overrides keeps shared settings DRY.
- **PEFT is an operational win:** Adapter training and storage scale better than full-model runs for most iteration paths.
- **QLoRA extends reach:** When VRAM is the bottleneck, 4-bit loading plus LoRA keeps the same adapter workflow on smaller GPUs.
- **Serving follows training:** vLLM loads the same base model + adapter path derived from project config—no manual path wiring.

---

## Resume Value

This project demonstrates:

- **Hugging Face ecosystem:** `transformers`, `datasets`, `Trainer`, `TrainingArguments`, PEFT `LoraConfig` / `get_peft_model`
- **LLM training mechanics:** causal LM labels, instruction formatting, checkpointing, CPU/GPU dtype selection
- **Efficient fine-tuning:** LoRA target modules, rank/alpha, QLoRA with `BitsAndBytesConfig`
- **Production serving:** FastAPI + vLLM, API key auth, rate limiting, health/metrics endpoints
- **Software engineering:** multi-project template design, YAML config merge, modular packages, reproducible CLI entrypoints

---

## License

Add a `LICENSE` file when you publish (e.g. MIT).

---

## Acknowledgments

Built with [Hugging Face Transformers](https://huggingface.co/docs/transformers), [PEFT](https://huggingface.co/docs/peft), [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes), [vLLM](https://docs.vllm.ai/), [FastAPI](https://fastapi.tiangolo.com/), and [PyTorch](https://pytorch.org/).
