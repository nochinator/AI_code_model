# Codex-Style Multimodal Coding Assistant

This repository contains:

1. `codex_style_assistant/`: product-like runtime scaffold.
2. `train_system/`: a configurable LLM training stack with improved architecture and data mixing support.

## What changed in this revision

- Added a **175M-class model config** (`configs/train_175m.yaml`) targeting roughly ~170M parameters.
- Upgraded architecture to a more modern decoder block:
  - RMSNorm
  - SwiGLU MLP
  - scaled dot-product causal attention
  - tied input/output embeddings
- Reworked data pipeline for **mixed corpora**:
  - synthetic conversational+coding samples
  - local JSONL sources
  - optional Hugging Face dataset ingestion
- Added tokenizer upgrades:
  - SentencePiece BPE tokenizer (default)
  - simple whitespace fallback
- Added parameter-budget checks so model settings stay close to target size.

## Important expectation setting

A 175M model can become semi-coherent and code-capable with good data/training, but it is **not expected to match ChatGPT-3 quality** broadly. The goal here is a strong small-model baseline for experimentation and constrained tasks.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training

### Baseline (smaller/easier)

```bash
python run.py --config configs/train_base.yaml
```

### 175M-class run

```bash
python run.py --config configs/train_175m.yaml
```

Artifacts are written to the config `output_dir`:

- tokenizer metadata (`tokenizer.json`)
- checkpoints (`checkpoints/best.pt`, `checkpoints/final.pt`)
- metrics (`metrics.json`)

## Data configuration

The training corpus is built from:

- `train_sources` / `val_sources` (real data, recommended)
- `train_synthetic_examples` / `val_synthetic_examples` (fallback augmentation)

Source types:

1. `jsonl`
   - `path`, `text_key`, optional `max_rows`
2. `huggingface`
   - `dataset`, optional `subset`, `split`, `text_field`, optional `max_samples`

Example source entry:

```yaml
- type: jsonl
  path: data/real/mixed_train.jsonl
  text_key: text
  max_rows: 1000000
```

## Inference

```bash
python -m train_system.run_infer \
  --checkpoint outputs/run_175m/checkpoints/best.pt \
  --tokenizer outputs/run_175m/tokenizer.json \
  --prompt "<|system|> You are a helpful coding assistant <|user|> implement retry middleware <|assistant|>" \
  --max-new-tokens 160
```

## Notes on dataset quality

To move toward the requested behavior (semi-coherent code + broad conversation), prioritize:

- large, deduplicated, license-clean text+code corpora
- strong instruction and dialogue data
- balanced code/explanation mixtures
- robust validation/evaluation sets

The synthetic generator should be treated as a bootstrap source, not the primary training corpus.
