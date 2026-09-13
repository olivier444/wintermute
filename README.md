# Wintermute

**530M parameters · 24.3B base-pretraining tokens · single RTX 5070 Ti (16 GB) · English/French/code (trained from scratch)**

Wintermute is a simple end-to-end research codebase for building and training small language models under constrained hardware budget. The goal is to own the full stack - data ingestion and deduplication, tokenizer training, Transformer implementation, pretraining/SFT, evaluation, checkpointing, and inference - and explore how far compact models can be pushed on a single consumer GPU.

## First Results At A Glance

The reference 530M base model was trained from scratch on a single NVIDIA RTX 5070 Ti with 16 GB of VRAM. The published base-pretraining checkpoint below had seen `24B` tokens; continued-pretraining stages are tracked separately.

All models in this comparison were evaluated zero-shot with the same `lm-eval` harness, task suite, and metric definitions. `Mean` is the unweighted mean of nine accuracy columns: ARC-C, ARC-E, COPA, HellaSwag, OpenBookQA, PIQA, WinoGrande, TruthfulQA MC2, and LAMBADA accuracy.

| Model | ARC-C | ARC-E | COPA | HellaSwag | OBQA | PIQA | WinoGrande | TruthfulQA MC2 | LAMBADA acc | Mean (9 tasks) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Wintermute 530M (base)** | **27.05%** | **49.62%** | **64.00%** | **39.88%** | **32.00%** | **65.29%** | **51.54%** | **41.21%** | **34.74%** | **45.04%** |
| GPT-2 | 22.70% | 39.48% | 62.00% | 31.14% | 27.20% | 62.51% | 51.62% | 40.69% | 32.56% | 41.10% |
| SmolLM2-135M | 30.03% | 58.46% | 69.00% | 43.08% | 32.40% | 68.17% | 52.80% | 38.73% | 42.69% | 48.37% |
| Qwen2.5-0.5B | 31.74% | 58.42% | 74.00% | 52.22% | 35.00% | 70.24% | 56.20% | 40.03% | 51.95% | 52.20% |
| SmolLM2-360M | 38.05% | 68.35% | 79.00% | 56.28% | 37.60% | 71.76% | 58.88% | 33.50% | 53.43% | 55.21% |

The Wintermute row corresponds to checkpoint `step_0_38700` from the base-pretraining run. See the [full pretraining report](configs/training/run530M/notes/00-Pretrain-Base.md) for details.

## Installation

Create a virtual environment and install it in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m wintermute --help
pytest
```

## Current Experiment Family

The newest configuration family is [`configs/training/run530M/`](configs/training/run530M/). Its base model has `530M` parameters and uses the following architecture:

| Component | Configuration |
| --- | --- |
| Model | `DecoderV2`, decoder-only Transformer |
| Context | 1,024 tokens |
| Vocabulary | 48,000-token byte-level BPE |
| Width / depth | `d_model=1024`, 24 layers |
| Attention | 16 query heads, 8 key/value heads, QK normalization |
| Position encoding | RoPE |
| Feed-forward block | SwiGLU, `d_ff=5504` |
| Normalization | RMSNorm |
| Embeddings | tied input/output embeddings |
| Precision | bfloat16 on CUDA |

## Pipeline

```text
Hugging Face datasets
    -> raw JSONL shards and stable record ids
    -> MinHash signatures and parquet indexes
    -> duplicate edges and clusters
    -> budgeted snapshot membership
    -> task-shaped materialized JSONL
    -> tokenizer and runtime training views
    -> training, evaluation, checkpoints, and probes
    -> local or Slack-backed inference
```

## Tokenization And Chat Formatting

Local tokenizers use byte-level BPE with configurable NFC or NFKC normalization. Runtime and training input share a normalization boundary that standardizes newlines and selected invisible, spacing, and bidirectional-control characters. Operator boundaries are isolated and digit runs are split into individual digits before BPE merges.

Locally trained tokenizers reserve EOS/PAD/UNK plus the Wintermute conversation controls, including `[:system:]`, `[:user:]`, `[:assistant:]`, `[:scratchpad:]`, `[:task:]`, `[:format:]`, `[:final:]`, and `[:eot:]`.

Chat protocols are separate from tokenizer wrappers:

- `WintermuteChatFormat` renders the compact local control-token grammar.
- `HuggingFaceChatFormat` delegates rendering to an external tokenizer's official chat template.


## Models And Training

The active model implementation is `DecoderV2`, built directly in PyTorch. It supports grouped-query causal attention, RoPE, LayerNorm or RMSNorm, GELU or SwiGLU feed-forward blocks, factorized embeddings, tied or untied output heads, QK normalization, and activation checkpointing.

Training setup:

- token-based gradient accumulation + gradient checkpointing;
- AdamW / weight-decay;
- configurable learning-rate schedules;
- bfloat16;
- periodic evaluation and external/local baselines;
- rolling, best-validation, named, resumable, and forkable checkpoints;
- TensorBoard, JSONL, text, and optional Slack telemetry;
- behavioural generation probes
- various runtime diagnostics such as gradient-noise or linear approximations checks.

Prompt/completion views can supervise only the assistant continuation or include the prompt in causal loss. Plain text and SFT records use the same batching, windowing, packing, evaluation, and checkpoint stack.

## CLI

The main commands are:

- Raw/index pipeline: `get_raw`
- Materialization and analysis: `create-snapshot`, `analyze`
- Tokenizers: `train-tokenizer`
- Runs: `run-new`, `resume`, `fork`
- Slack inference: `start-inference-endpoint`

Example commands:

```bash

python -m wintermute run-new \
  --output-root /data/slm \
  --training-config configs/training/run530M/00-pretrain.yaml

python -m wintermute analyze \
  --output-root /data/slm \
  --training-config configs/training/run530M/01-continued-pretrain.yaml

python -m wintermute start-inference-endpoint \
  --output-root /data/slm \
  --endpoint-config configs/inference/default.yaml
```

## Artifacts

For an output root such as `/data/slm`, Wintermute writes:

```text
raw/                  source-like JSONL shards and ingestion metadata (mostly from HuggingFace)
index/                parquet source, record, band, duplicate, and snapshot tables
materialized/         task-facing snapshot splits and audits
tokenizer/            trained tokenizer JSON files
run_store/            manifests, logs, models, checkpoints, and fork provenance
```

Run manifests use the canonical `configuration` wrapper and are persisted at `run_store/<run_id>/description.yaml`.

## Monitoring And Inference

Training can attach a Slack thread to a run for status updates, metrics history, pause/resume/stop controls, runtime parameter changes, and named checkpoints.

`generate` opens an interactive local generation loop. `start-inference-endpoint` runs a Slack Socket Mode endpoint with one conversation per thread, model switching, regeneration, system-prompt controls, sampling controls, and optional external Hugging Face models.


## License

The code is available under the [MIT License](LICENSE). Dataset-specific use remains subject to each source's license and terms.
