# 00-Pretrain-Base

Experiments were conducted on a local Ubuntu Server 24.04 LTS machine / AMD Ryzen 9 7900 CPU (12 cores, 24 threads), 64 GB of RAM, and an NVIDIA GeForce RTX 5070 Ti GPU with 16 GB of VRAM.

## Stage: `qlike3-run.000-pretrain`

Stage role: **base pretraining**.

| Field | Value |
| --- | --- |
| Config file | `00-pretrain.yaml` |
| Reference run id | `` |
| Primary eval | `mix40b-txt` |
| Include prompt in loss | `True` |
| Batch size | `12` |
| Grad accum units | `500000` initially; raised live to `750000`, then `1000000` |
| Planned training units | `41000000000` |
| Terminal units seen | `24368459165` (`59%`) |
| LR schedule | `max=0.0003`, `end=5e-05`, `warmup=0.006`, `decay_start=0.9`; subsequently adjusted live |
| Optimizer | AdamW, `beta1=0.9`, `beta2=0.95`, `eps=1e-08`, initial `weight_decay=0.05` |
| Config note | Newer Qwen-like architecture with fixed initialization and tokenizer; QK normalization enabled. |

### Timing

| Field | Value |
| --- | --- |
| Start | 2026-07-07 |
| Wall-clock | 20d 14h 47m 21s |
| Tokens used | `24,368,459,165` |

### Curriculum Inputs

Training view: `mix40b-pretrain`.

| Eval name | Primary | Snapshot ref | Evaluation domain |
| --- | --- | --- | --- |
| `mix40b-txt` | `yes` | `mix40b-txt/eval` | Mixed pretraining distribution: mostly English, with French, mathematics, reasoning, code, and a small instruction component. |
| `def4` | `no` | `default4-txt/eval3` | English general-domain text. |
| `code` | `no` | `code-txt/eval` | Source code, predominantly Python (`75.4%` of estimated tokens; `24.6%` other languages). |
| `french` | `no` | `french-txt/eval` | French text: FineWeb2-HQ web pages (`80.6%`) and French Wikipedia (`19.4%`). |

### Dataset Snapshot Summary

Materialized audits estimate tokens at four characters per token; trainer-side tokenizer counts are reported separately below.

| Role | Snapshot | Split | Examples | Unique examples | Approx tokens | Chars | Avg chars/ex | Sources | Duplicate rate | Oversampling extra ex | Oversampling extra chars | Main Hugging Face sources |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `train` | `mix40b-txt` | `train` | 31,110,539 | 29,911,107 | 39,999,869,385 | 159,999,477,538 | 5142.9 | 28 | 0.00% | 1,199,432 | 4,462,674,660 | HuggingFaceFW/fineweb-edu (28.4%); epfml/FineWeb2-HQ (11.8%); open-web-math/open-web-math (11.8%) |
| `eval` | `mix40b-txt` | `eval` | 2,064 | 1,993 | 2,515,091 | 10,060,364 | 4874.2 | 23 | 0.00% | 71 | 270,852 | HuggingFaceFW/fineweb-edu (28.1%); open-web-math/open-web-math (12.8%); epfml/FineWeb2-HQ (11.6%) |
| `eval` | `default4-txt` | `eval3` | 2,443 | 2,319 | 2,453,112 | 9,812,448 | 4016.6 | 9 | 0.12% | 124 | 521,905 | HuggingFaceFW/fineweb-edu (30.6%); HuggingFaceFW/fineweb (20.4%); HuggingFaceTB/cosmopedia [stories] (15.3%) |
| `eval` | `code-txt` | `eval` | 1,855 | 1,855 | 2,487,695 | 9,950,778 | 5364.3 | 2 | 0.00% | 0 | 0 | bigcode/the-stack-dedup [Python] (75.4%); bigcode/the-stack-dedup [other] (24.6%) |
| `eval` | `french-txt` | `eval` | 2,977 | 2,977 | 2,495,420 | 9,981,679 | 3352.9 | 2 | 0.00% | 0 | 0 | epfml/FineWeb2-HQ (80.6%); wikimedia/wikipedia [French] (19.4%) |

### Declared Training Recipe

Declarative `mix40b-txt` manifest

| Source id | Hugging Face source | Weight | Oversampling | Record transform |
| --- | --- | ---: | ---: | --- |
| `fwe-2` | HuggingFaceFW/fineweb-edu | 12.0 | 1 | `text` |
| `fwe` | HuggingFaceFW/fineweb | 2.4 | 1 | `text` |
| `wk` | wikimedia/wikipedia [20231101.en] | 2.2 | 1 | `text` |
| `pg` | emozilla/pg19 | 2.2 | 1 | `text` |
| `cst` | HuggingFaceTB/cosmopedia [stories] | 1.0 | 1 | `text` |
| `cox` | HuggingFaceTB/cosmopedia [openstax] | 0.7 | 6 | `text` |
| `cwh` | HuggingFaceTB/cosmopedia [wikihow] | 0.5 | 3 | `text` |
| `owm` | open-web-math/open-web-math | 5.0 | 1 | `text` |
| `oth` | open-thoughts/OpenThoughts3-1.2M | 4.0 | 1 | `chat_to_text` |
| `cmt` | HuggingFaceTB/cosmopedia [auto_math_text] | 1.4 | 1 | `text` |
| `ckh` | HuggingFaceTB/cosmopedia [khanacademy] | 0.25 | 12 | `text` |
| `stp2` | bigcode/the-stack-dedup [Python] | 3.0 | 1 | `text` |
| `sto2` | bigcode/the-stack-dedup [other] | 1.0 | 1 | `text` |
| `fw2-fr` | epfml/FineWeb2-HQ [fra_Latn] | 5.0 | 1 | `text` |
| `wk-fr` | wikimedia/wikipedia [20231101.fr] | 1.2 | 1 | `text` |
| `smt` | HuggingFaceTB/smol-smoltalk | 0.15 | 1 | `chat_to_text` |
| `uch` | HuggingFaceH4/ultrachat_200k | 0.1 | 1 | `chat_to_text` |
| `mlp3` | Magpie-Align/Magpie-Llama-3.1-Pro-MT-300K-Filtered | 0.05 | 1 | `chat_to_text` |
| `mpm` | Magpie-Align/Magpie-Pro-300K-Filtered | 0.03 | 1 | `chat_to_text` |
| `sor` | Open-Orca/SlimOrca | 0.04 | 1 | `chat_to_text` |
| `oa1` | OpenAssistant/oasst1 | 0.007 | 1 | `chat_to_text` |
| `opy` | garage-bAInd/Open-Platypus | 0.015 | 2 | `field_mapping` |
| `d15` | databricks/databricks-dolly-15k | 0.01 | 4 | `field_mapping` |
| `nor` | HuggingFaceH4/no_robots | 0.015 | 5 | `chat_to_text` |
| `edc` | HuggingFaceTB/everyday-conversations-llama3.1-2k | 0.003 | 6 | `chat_to_text` |
| `xp3-fr` | bigscience/xP3 [French] | 0.05 | 1 | `field_mapping` |
| `oa2-fr` | OpenAssistant/oasst2 [French] | 0.004 | 7 | `chat_to_text` |
| `aya-fr` | CohereLabs/aya_dataset [French] | 0.002 | 10 | `field_mapping` |

### Trainer-Side Effective Mixture

| Source id | Hugging Face source | Segments | Total tokens | Loss tokens | Loss-token share | Loss-token ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `fwe-2` | HuggingFaceFW/fineweb-edu | 15,400,424 | 11,840,796,680 | 10,131,745,073 | 23.15% | 85.57% |
| `owm` | open-web-math/open-web-math | 7,521,897 | 6,850,425,009 | 5,873,684,213 | 13.42% | 85.74% |
| `oth` | open-thoughts/OpenThoughts3-1.2M | 6,172,373 | 6,251,879,111 | 5,434,352,236 | 12.42% | 86.92% |
| `fw2-fr` | epfml/FineWeb2-HQ | 8,095,833 | 5,520,368,154 | 4,768,516,136 | 10.89% | 86.38% |
| `stp2` | bigcode/the-stack-dedup [Python] | 5,630,141 | 4,835,152,175 | 4,146,913,252 | 9.47% | 85.77% |
| `pg` | emozilla/pg19 | 2,499,776 | 2,554,968,769 | 2,233,140,276 | 5.10% | 87.40% |
| `wk` | wikimedia/wikipedia [English] | 3,918,553 | 2,565,646,650 | 2,219,863,687 | 5.07% | 86.52% |
| `fwe` | HuggingFaceFW/fineweb | 3,088,578 | 2,369,391,383 | 2,027,495,177 | 4.63% | 85.57% |
| `sto2` | bigcode/the-stack-dedup [other] | 2,286,440 | 2,064,723,199 | 1,790,449,498 | 4.09% | 86.72% |
| `cmt` | HuggingFaceTB/cosmopedia [auto_math_text] | 2,241,945 | 1,604,340,105 | 1,416,920,481 | 3.24% | 88.32% |
| `wk-fr` | wikimedia/wikipedia [French] | 2,226,551 | 1,498,152,244 | 1,296,630,343 | 2.96% | 86.55% |
| `cst` | HuggingFaceTB/cosmopedia [stories] | 1,395,128 | 783,800,284 | 776,148,764 | 1.77% | 99.02% |
| `cox` | HuggingFaceTB/cosmopedia [openstax] | 869,040 | 666,442,674 | 573,551,346 | 1.31% | 86.06% |
| `cwh` | HuggingFaceTB/cosmopedia [wikihow] | 562,716 | 504,079,467 | 407,342,451 | 0.93% | 80.81% |
| `ckh` | HuggingFaceTB/cosmopedia [khanacademy] | 339,504 | 288,912,372 | 240,786,144 | 0.55% | 83.34% |
| `smt` | HuggingFaceTB/smol-smoltalk | 328,788 | 138,185,266 | 134,642,304 | 0.31% | 97.44% |
| `uch` | HuggingFaceH4/ultrachat_200k | 209,341 | 87,691,487 | 81,906,495 | 0.19% | 93.40% |
| `mlp3` | Magpie-Align/Magpie-Llama-3.1-Pro-MT-300K-Filtered | 87,688 | 56,068,400 | 50,959,239 | 0.12% | 90.89% |
| `xp3-fr` | bigscience/xP3 [French] | 301,977 | 48,824,563 | 48,514,870 | 0.11% | 99.37% |
| `sor` | Open-Orca/SlimOrca | 89,710 | 38,089,673 | 35,919,646 | 0.08% | 94.30% |
| `mpm` | Magpie-Align/Magpie-Pro-300K-Filtered | 34,659 | 26,886,525 | 25,536,880 | 0.06% | 94.98% |
| `opy` | garage-bAInd/Open-Platypus | 50,244 | 21,205,586 | 19,538,020 | 0.04% | 92.14% |
| `nor` | HuggingFaceH4/no_robots | 51,670 | 14,396,135 | 13,842,335 | 0.03% | 96.15% |
| `d15` | databricks/databricks-dolly-15k | 51,104 | 9,669,524 | 9,167,320 | 0.02% | 94.81% |
| `oa1` | OpenAssistant/oasst1 | 25,976 | 6,162,548 | 6,014,220 | 0.01% | 97.59% |
| `oa2-fr` | OpenAssistant/oasst2 [French] | 17,661 | 3,640,770 | 3,542,854 | 0.01% | 97.31% |
| `edc` | HuggingFaceTB/everyday-conversations-llama3.1-2k | 44,526 | 2,499,942 | 2,455,416 | 0.01% | 98.22% |
| `aya-fr` | CohereLabs/aya_dataset [French] | 13,240 | 2,041,700 | 1,985,140 | 0.00% | 97.23% |

### Observed Runtime Adjustments

The persisted run description records the values active at run creation; the current YAML incorporates some later choices. The logs show these material execution changes:

| Date (UTC) | Change |
| --- | --- |
| 2026-07-07 to 2026-07-08 | Gradient clipping reduced from `5` to `3`, then `2`, then `1`. |
| 2026-07-13 | Checkpoint cadence changed from `100` to `200` optimizer batches. |
| 2026-07-15 | Training resumed from checkpoint `19000` after a short interruption. |
| 2026-07-16 to 2026-07-17 | Maximum LR was temporarily raised as high as `0.0004`, then returned to `0.0003`; gradient-noise estimation was enabled and varied between 4 and 8 update batches. |
| 2026-07-19 | Gradient accumulation rose to `750000` units; gradient-noise estimation was disabled later that day. |
| 2026-07-20 | Weight decay changed from `0.05` to `0.01`. |
| 2026-07-22 | Gradient accumulation rose to `1000000`, checkpoint cadence changed to `150`, and gradient clipping fell to `0.5`. |
| 2026-07-23 to 2026-07-27 | Maximum LR was manually stepped down through several values, ending at `0.0001`. |

### Last-Checkpoint Metrics

| Eval | First step | First loss | First top10 | First bpb | Last step | Last loss | Last top10 | Last bpb |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mix40b-txt` | 100 | 6.528862 | 0.408215 | 2.492634 | 38700 | 2.042538 | 0.837065 | 0.779815 |
| `def4` | 100 | 6.932056 | 0.358921 | 2.281033 | 38700 | 2.267161 | 0.817303 | 0.746022 |
| `code` | 100 | 6.122239 | 0.466919 | 3.290305 | 38700 | 1.298458 | 0.892272 | 0.697837 |
| `french` | 100 | 7.100813 | 0.360860 | 2.546442 | 38700 | 2.348080 | 0.804750 | 0.842051 |

### Runtime Performance

| Metric | Value |
| --- | ---: |
| Average allocated VRAM | 6166.13 MB |
| Average reserved VRAM | 14198.87 MB |
| Peak allocated VRAM | 12896.49 MB |
| Peak reserved VRAM | 14942.00 MB |
| Average throughput | 13703.52 units/sec |
| Median throughput | 13574.44 units/sec |
| Average sec/step | 44.23 s |
| Median sec/step | 37.33 s |
| Average eval/checkpoint time | 246.94 s |
| Peak eval/checkpoint time | 301.56 s |

### Inline Baseline Comparisons

All baseline models were evaluated on the same four suites at stage initialization. `current_run_last` is the terminal checkpoint.
`def4` is English, `code` is source code dominated by Python, and `french` is French.

#### `mix40b-txt` - mixed pretraining distribution

| Model | Loss | BPB | Top10 acc |
| --- | ---: | ---: | ---: |
| `current_run_last` | 2.042538 | 0.779815 | 0.837065 |
| `smollm2_360m` | 2.148625 | 0.814745 | 0.825880 |
| `smollm2_360m_instruct` | 2.267878 | 0.859965 | 0.814214 |
| `smollm2_135m` | 2.415187 | 0.915824 | 0.795335 |
| `smollm2_135m_instruct` | 2.543583 | 0.964511 | 0.781735 |
| `bigger2` | 2.543647 | 0.972338 | 0.780246 |
| `pythia_160m` | 3.297286 | 1.200586 | 0.662174 |
| `gpt2` | 3.008447 | 1.211201 | 0.725097 |

#### `def4` - English general-domain text

| Model | Loss | BPB | Top10 acc |
| --- | ---: | ---: | ---: |
| `current_run_last` | 2.267161 | 0.746022 | 0.817303 |
| `smollm2_360m` | 2.180161 | 0.690018 | 0.831844 |
| `smollm2_360m_instruct` | 2.334173 | 0.738763 | 0.815507 |
| `smollm2_135m` | 2.445716 | 0.774066 | 0.799423 |
| `bigger2` | 2.518161 | 0.787740 | 0.790766 |
| `smollm2_135m_instruct` | 2.608674 | 0.825642 | 0.781518 |
| `gpt2` | 3.113803 | 0.965462 | 0.712826 |
| `pythia_160m` | 3.536506 | 1.098540 | 0.629214 |

#### `code` - predominantly Python source code

| Model | Loss | BPB | Top10 acc |
| --- | ---: | ---: | ---: |
| `current_run_last` | 1.298458 | 0.697837 | 0.892272 |
| `smollm2_360m` | 1.378211 | 0.636282 | 0.881308 |
| `smollm2_360m_instruct` | 1.456678 | 0.672508 | 0.875156 |
| `smollm2_135m` | 1.573261 | 0.726331 | 0.863359 |
| `smollm2_135m_instruct` | 1.659413 | 0.766105 | 0.855692 |
| `bigger2` | 1.488975 | 0.771200 | 0.868674 |
| `pythia_160m` | 2.211498 | 1.026733 | 0.792653 |
| `gpt2` | 2.077507 | 1.390086 | 0.838687 |

#### `french` - French web and encyclopedic text

| Model | Loss | BPB | Top10 acc |
| --- | ---: | ---: | ---: |
| `current_run_last` | 2.348080 | 0.842051 | 0.804750 |
| `smollm2_360m` | 2.513601 | 1.216970 | 0.771165 |
| `smollm2_360m_instruct` | 2.693059 | 1.303855 | 0.753927 |
| `smollm2_135m` | 2.862830 | 1.386051 | 0.730905 |
| `smollm2_135m_instruct` | 3.046597 | 1.475022 | 0.712620 |
| `pythia_160m` | 3.475925 | 1.505467 | 0.623545 |
| `bigger2` | 3.273298 | 1.626424 | 0.692676 |
| `gpt2` | 3.968527 | 1.911575 | 0.598816 |

### Offline Zero-Shot Benchmarks

The standard zero-shot suite was run later against checkpoint `step_0_38700` (`24,324,220,243` units seen at checkpoint time). Accuracy-normalized values are used for ARC, HellaSwag, OpenBookQA, and PIQA.

| ARC-C | ARC-E | COPA | HellaSwag | OBQA | PIQA | WinoGrande | TruthfulQA MC2 | LAMBADA acc | Mean | LAMBADA PPL |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 27.05% | 49.62% | 64.00% | 39.88% | 32.00% | 65.29% | 51.54% | 41.21% | 34.74% | 45.04% | 30.74 |


## Selected Late-Step Probe Examples

Selected prompt/completion pairs from `log/info/sample_random/*/*_38406.txt`. These are illustrative spot checks, not representative averages.

- Prompt: `What is the capital of France?`
  Completion: `The capital of France is Paris.`

- Prompt: `Explain in one sentence why the sky appears blue.`
  Completion: `The sky appears blue because of a phenomenon called Rayleigh scattering...` The answer continued beyond one sentence.

- Prompt: `What is 2 + 7? Answer with just a number.`
  Completion: `2 + 7 = 9. Here's a simple visual representation...`

- Prompt: `Complete this Python function: def add(a, b): return`
  Completion: `Here is a Python function that adds two numbers: def add(a, b): return a + b.`

- Prompt: `To make a cup of tea, first`
  Completion: `you need to make tea. This is usually done by adding tea leaves to tea bags...`

- Prompt: `Monday, Tuesday, Wednesday,`
  Completion: `Thursday, Friday, Sunday, Monday, Tuesday, Wednesday...`

- Prompt: `The cup was on the table. It fell down. Is it more likely on the floor or still on the table? Answer with one word.`
  Completion: `The cup fell down on the table... On the floor, the cup was on the floor, which means it was on the table.`

- Prompt: `The Eiffel Tower is a iron lattice tower located in`
  Completion: `the city of Paris, France. The tower was built by Gustave Eiffel in 1889...`

- Prompt: `The difference between correlation and causation is`
  Completion: `100%. In the case of correlation, it is the degree to which two variables move in relation to each other...`

- Prompt: `“I didn't mean to lie,” she said. “Then why did you?” She looked away and`
  Completion: `saw him, staring down at the floor, with a cold, hard look on his face...`
