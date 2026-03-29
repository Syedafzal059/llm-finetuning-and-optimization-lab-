# LLM Fine-Tuning & Optimization Lab

**End-to-end Hugging Face Transformers pipeline: baseline inference → supervised fine-tuning (SFT) → LoRA / QLoRA adapters—config-driven, modular, and production-minded.**

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-transformers-orange.svg)](https://pytorch.org/)

---

## Overview

This repository is a **compact but complete** LLM engineering lab: load a causal LM, generate text with controlled sampling, fine-tune on instruction–response data with **Hugging Face Trainer**, then specialize further with **PEFT LoRA** or **QLoRA** (4-bit base + LoRA)—all without a monolithic notebook stack.

**Problem it solves:** Teams need a repeatable path from “model on disk / Hub” to **domain-tuned behavior** without full-model retraining on every iteration. This project encodes that path with clear separation between **inference**, **full SFT**, and **adapter-only** training (including memory-efficient QLoRA when GPU memory is tight).

**Why it matters:** In production, the cost of full fine-tunes (GPU time, storage, regression risk) rarely scales. **Parameter-efficient fine-tuning (PEFT)** and **rigorous data formatting** are how most real systems ship updates. This repo demonstrates that workflow with tooling that maps directly to ML platform jobs (Trainer, checkpoints, artifacts).

---

## Key Features

| Area | What you get |
|------|----------------|
| **Baseline inference** | `AutoModelForCausalLM` + tokenizer, YAML-driven `max_new_tokens`, temperature, `top_p`, `do_sample`, CPU/CUDA |
| **SFT** | JSON instruction–response data → formatted strings → tokenized `datasets` with labels for causal LM |
| **LoRA** | PEFT `LoraConfig` on attention projections (`q_proj`, `v_proj`), trainable fraction ~**0.1%** of base params (typical for this setup) |
| **QLoRA** | `BitsAndBytesConfig` (4-bit NF4, double quant, fp16 compute) + same LoRA targets; lower VRAM than full-precision LoRA; outputs under `model/qlora/` |
| **Config-first design** | Single `configs.py/base.yaml` for model, device, and generation knobs |
| **Modular layout** | `src/inference`, `src/training`, `src/utils`; scripts runnable from repo root |
| **Efficient artifacts** | LoRA / QLoRA save **adapter weights only** under `model/lora` or `model/qlora` (full model checkpoints gitignored) |

---

## Architecture

End-to-end flow:

1. **Baseline** — Load a Hub model (default: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`), tokenize, generate with explicit sampling settings.
2. **SFT** — Curate `instruction` / `response` pairs → map to supervised causal language modeling → `Trainer` optimizes **full** weights → checkpoints under `model/sft/`.
3. **LoRA** — Freeze base weights, inject low-rank adapters (rank `8`, `alpha` `16`) → train **small** matrices → save adapters only under `model/lora/`.
4. **QLoRA** — Load the base in **4-bit** (NF4), attach the same LoRA setup → train adapters with less GPU memory → save under `model/qlora/`.
5. **Future** — FastAPI serving layer, eval harness, and extra quantization paths for deployment.

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────────┐
│  YAML config │ ──▶ │ Load model   │ ──▶ │  Inference  │     │  (Roadmap) API   │
└─────────────┘     └──────────────┘     └─────────────┘     └──────────────────┘
                            │
                            ▼
                   ┌──────────────┐     ┌─────────────┐
                   │  SFT Trainer │ ──▶ │ Full weights │
                   └──────────────┘     └─────────────┘
                            │
                            ▼
                   ┌──────────────┐     ┌─────────────┐
                   │ PEFT + LoRA  │ ──▶ │ Adapters only│
                   └──────────────┘     └─────────────┘
                            │
                            ▼
                   ┌──────────────┐     ┌─────────────┐
                   │ PEFT + QLoRA │ ──▶ │ 4-bit + adapters│
                   └──────────────┘     └─────────────┘
```

---

## Project Structure

```
llm-finetuning-and-optimization-lab/
├── configs.py/
│   └── base.yaml              # Model name, device, max_new_tokens, sampling
├── data/
│   └── raw/
│       └── sample_dataset.json   # Instruction–response JSON for SFT / LoRA
├── src/
│   ├── inference/
│   │   ├── model_loader.py    # Base load + LoRA via PeftModel.from_pretrained
│   │   └── generate.py        # Tokenize, generate, decode
│   ├── training/
│   │   ├── dataset_loader.py  # Load JSON, format, tokenize, labels
│   │   ├── train_sft.py       # Full fine-tune (Trainer)
│   │   ├── train_lora.py      # LoRA fine-tune (PEFT)
│   │   └── train_qlora.py     # QLoRA: 4-bit base + LoRA (bitsandbytes)
│   └── utils/
│       └── config_loader.py   # YAML loader
├── model/                     # Training outputs (gitignored)
├── run.py                     # Baseline inference demo
├── run_lora.py                # Inference with trained LoRA adapters
├── requirements.txt
└── README.md
```

> **Note:** The config directory is named `configs.py/` (historical layout). Paths in code point to `configs.py/base.yaml`.

---

## Setup

**Prerequisites:** Python 3.10+, [PyTorch](https://pytorch.org/) matching your OS/CUDA, and a Hugging Face account/token optional for higher Hub rate limits). **QLoRA** requires [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) with a **CUDA** GPU (4-bit loading is not used on CPU in this script).

```bash
cd llm-finetuning-and-optimization-lab
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

Copy `.env.example` to `.env` if you add API keys or `HF_TOKEN` later (never commit secrets).

---

## Usage

### Inference (baseline)

```bash
python run.py
```

Reads `configs.py/base.yaml`, loads the model, runs one generation (prompt is set in `run.py`).

### Train SFT (full supervised fine-tuning)

```bash
python src/training/train_sft.py
```

Writes checkpoints under `model/sft/` (ignored by git).

### Train LoRA (PEFT adapters)

```bash
python src/training/train_lora.py
```

Saves adapters under `model/lora/`. `print_trainable_parameters()` reports trainable vs total parameters—expect on the order of **~0.1%** trainable for this TinyLlama + `q_proj`/`v_proj` setup.

### Train QLoRA (4-bit base + LoRA)

```bash
python src/training/train_qlora.py
```

Loads the base model in **4-bit** (NF4), applies the same LoRA configuration as `train_lora.py`, and writes adapter checkpoints under `model/qlora/`. Use this when **full-precision LoRA** does not fit comfortably in GPU memory.

### Inference with LoRA adapters

After training, load base weights + adapters and generate:

```bash
python run_lora.py
```

Requires `model/lora/adapter_config.json` (and adapter weights) from `train_lora.py`.

**QLoRA-trained adapters:** `run_lora.py` points at `model/lora/` by default. After `train_qlora.py`, either change the `adapter_dir` in `run_lora.py` to `model/qlora`, or load `model/qlora` with the same `PeftModel.from_pretrained` pattern used in `src/inference/model_loader.py`—the adapter format matches standard PEFT checkpoints.

---

## Results & Insights

| Metric | Insight |
|--------|---------|
| **Trainable params (LoRA)** | ~**0.1%** of base model (adapter-only training; exact figure from `print_trainable_parameters()` at run time) |
| **Memory / iteration cost** | LoRA trains small matrices; **QLoRA** further cuts base-model memory via 4-bit weights; full SFT updates all weights—use SFT for max capacity, LoRA/QLoRA for fast iteration |
| **Quality** | SFT and LoRA both align the model to your **instruction–response** format; LoRA trades some flexibility for efficiency |
| **Artifacts** | LoRA exports are **MB-scale adapters** vs multi-GB full checkpoints |

---

## Example Outputs (Illustrative)

| Stage | Prompt | Representative behavior |
|-------|--------|-------------------------|
| **Base** | “Explain what is LoRA in simple terms” | General chat-style answer; may not match your task format |
| **SFT** | Same | Closer adherence to instruction-following style after training on formatted examples |
| **LoRA** | Same | Task-specific adaptation with **minimal** trainable weights; combine with base or merged workflow for inference |
| **QLoRA** | Same | Same adapter idea as LoRA; training uses a **quantized** base so you can iterate on smaller GPUs |

> Outputs vary with **sampling**, **seed**, and **epochs**. Treat the table as a **behavioral** comparison, not a fixed benchmark.

---

## Engineering Learnings

- **Data formatting is half the product:** Instruction templates and label alignment (`labels = input_ids`) make or break causal LM fine-tunes.
- **Sampling is not cosmetic:** `temperature`, `top_p`, and `do_sample` change both quality and variance; defaults belong in config, not scattered code.
- **PEFT is an operational win:** Adapter training and storage scale better than full-model runs for most iteration and deployment paths.
- **QLoRA extends reach:** When VRAM is the bottleneck, 4-bit loading plus LoRA keeps the same adapter workflow with a smaller memory footprint during training.
- **Import paths and cwd:** Training scripts prepend the project root to `sys.path` so `python src/training/...` runs reliably from the repo root.

---

## Roadmap

| Priority | Item |
|----------|------|
| Done | **QLoRA** — `train_qlora.py`: 4-bit NF4 base + LoRA (`bitsandbytes`) |
| Next | **Quantization** (INT8/INT4) for edge and batch inference beyond the training path |
| Soon | **FastAPI** service: load config, model, optional adapters |
| Soon | **Benchmarking** — held-out prompts, BLEU/ROUGE or LLM-as-judge hooks |
| Later | **Merge adapters** into base for single-file deployment |

---

## Resume Value

This project demonstrates:

- **Hugging Face ecosystem:** `transformers`, `datasets`, `Trainer`, `TrainingArguments`, PEFT `LoraConfig` / `get_peft_model`
- **LLM training mechanics:** causal LM labels, instruction formatting, checkpointing, CPU/GPU dtype selection
- **Efficient fine-tuning:** LoRA target modules, rank/alpha, trainable-parameter reporting; QLoRA with `BitsAndBytesConfig` and 4-bit loading
- **Software engineering:** YAML configuration, modular packages, reproducible entrypoints, `.gitignore` for large artifacts

---

## License

Add a `LICENSE` file when you publish (e.g. MIT).

---

## Acknowledgments

Built with [Hugging Face Transformers](https://huggingface.co/docs/transformers), [PEFT](https://huggingface.co/docs/peft), [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) (QLoRA), and [PyTorch](https://pytorch.org/).
