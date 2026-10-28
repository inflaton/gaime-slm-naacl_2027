# Aggregate results

All numbers below are retained study aggregates. Synthetic demonstration outputs are separate. Quality comparisons use 1,213 paired records. F1 and exact-match rates are percentages; differences are percentage points.

## Main serving comparison

This is the manuscript table subset. Values are medians from three runs with server restarts, warm-up and matched requests; model order was rotated. Ratios use the same-prompt **bf16** proxy measured in the same block. Displayed bf16 Llama rows come from the 12B blocks; fp8/W4A16 Llama ratios use bf16 baselines from the separate `llama-quant` block. Bold marks column bests within each manuscript table, including proxies and ties at the displayed precision. Prior 4B adapters and Llama proxies are test-selected; the displayed Qwen adapters use the Think training template. Loaded weights exclude server and KV-cache memory. Throughput values, ratios and rankings in this table use concurrency 64; latency and speed-up use concurrency 1. E4B GPTQ uses basic full-model calibration with its per_layer_model_projection retained in bf16, whereas 12B and Llama use sequential calibration; quantization-loss comparisons reflect both model and pipeline.

### Gemma 4: validation-F1 checkpoints

| Model / precision | F1 (%) ↑ | Weights (GiB) ↓ | Model/proxy ↓ | p50 c=1 (s) ↓ | Speed-up ↑ | c=64 rows/h ↑ | Model/proxy ↑ |
|:--|--:|--:|--:|--:|--:|--:|--:|
| 12B JO bf16 | **97.00** | 22.8 | 1.52× | 9.33 | 0.53× | 17,355 | 0.53× |
| &emsp;fp8 | 96.86 | 12.9 | 0.86× | 4.49 | 1.11× | 26,427 | 0.81× |
| &emsp;W4A16 | 96.88 | 7.8 | 0.52× | 2.78 | 1.79× | 27,902 | 0.85× |
| 12B FT bf16 | 96.85 | 22.8 | 1.52× | 9.28 | 0.54× | 16,805 | 0.53× |
| &emsp;fp8 | 96.72 | 12.9 | 0.86× | 4.51 | 1.10× | 24,691 | 0.79× |
| &emsp;W4A16 | 96.66 | 7.8 | 0.52× | 2.74 | 1.81× | 26,399 | 0.84× |
| E4B JO bf16 | 95.91 | 15.2 | 1.01× | 3.76 | 1.40× | 45,592 | 1.43× |
| &emsp;fp8 | 95.48 | 11.1 | 0.74× | 1.97 | 2.66× | 67,934 | 2.14× |
| &emsp;W4A16 | 95.15 | 9.7 | 0.64× | **1.35** | **3.89×** | **74,143** | **2.33×** |
| E4B FT bf16 | 95.82 | 15.2 | 1.01× | 3.75 | 1.33× | 45,823 | 1.45× |
| &emsp;fp8 | 95.70 | 11.1 | 0.74× | 1.97 | 2.54× | 67,825 | 2.14× |
| &emsp;W4A16 | 94.80 | 9.7 | 0.64× | 1.36 | 3.68× | 73,376 | 2.32× |

### Llama 8B proxies: production architecture at three precisions

| Model / precision | F1 (%) ↑ | Weights (GiB) ↓ | Model/proxy ↓ | p50 c=1 (s) ↓ | Speed-up ↑ | c=64 rows/h ↑ | Model/proxy ↑ |
|:--|--:|--:|--:|--:|--:|--:|--:|
| Llama 8B JO bf16 | 96.63 | 15.0 | 1.00× | 4.99 | 1.00× | 32,706 | 1.00× |
| &emsp;fp8 | 96.59 | 8.5 | 0.57× | 2.72 | 1.85× | 51,132 | 1.58× |
| &emsp;W4A16 | 95.84 | **5.4** | **0.36×** | 1.57 | 3.19× | 61,547 | 1.90× |
| Llama 8B FT bf16 | 96.70 | 15.0 | 1.00× | 4.97 | 1.00× | 31,426 | 1.00× |
| &emsp;fp8 | 96.31 | 8.5 | 0.57× | 2.73 | 1.82× | 48,969 | 1.56× |
| &emsp;W4A16 | 96.09 | **5.4** | **0.36×** | 1.57 | 3.17× | 58,300 | 1.85× |

## Prior 4B adapters (Appendix Table 6)

This separate table ranks only its twelve displayed prior-adapter configurations.

### Prior 4B adapters: retained test-selected checkpoints

| Model / precision | F1 (%) ↑ | Weights (GiB) ↓ | Model/proxy ↓ | p50 c=1 (s) ↓ | Speed-up ↑ | c=64 rows/h ↑ | Model/proxy ↑ |
|:--|--:|--:|--:|--:|--:|--:|--:|
| Qwen 3.5 4B JO bf16 | **96.62** | 8.6 | 0.57× | 3.53 | 1.42× | 21,486 | 0.66× |
| &emsp;fp8 | 96.35 | 5.3 | 0.36× | 1.85 | 2.70× | 27,593 | 0.85× |
| &emsp;W4A16 | 96.17 | **3.7** | **0.25×** | 1.29 | 3.88× | 23,625 | 0.73× |
| Qwen 3.5 4B FT bf16 | 96.16 | 8.6 | 0.57× | 3.63 | 1.38× | 21,166 | 0.67× |
| &emsp;fp8 | 96.19 | 5.3 | 0.36× | 1.90 | 2.64× | 26,291 | 0.83× |
| &emsp;W4A16 | 96.00 | **3.7** | **0.25×** | 1.34 | 3.74× | 23,511 | 0.74× |
| Gemma 3 4B JO bf16 | 95.29 | 8.6 | 0.57× | 3.02 | 1.65× | 50,990 | 1.57× |
| &emsp;fp8 | 95.26 | 5.5 | 0.37× | 1.65 | 3.04× | 73,905 | 2.28× |
| &emsp;W4A16 | 94.66 | 4.2 | 0.28× | **1.15** | **4.35×** | **81,313** | **2.51×** |
| Gemma 3 4B FT bf16 | 96.11 | 8.6 | 0.57× | 3.03 | 1.66× | 49,773 | 1.57× |
| &emsp;fp8 | 95.71 | 5.5 | 0.37× | 1.65 | 3.05× | 70,794 | 2.24× |
| &emsp;W4A16 | 95.07 | 4.2 | 0.28× | **1.15** | **4.35×** | 77,589 | 2.46× |

### Supplementary configurations and block-matched proxies

These rows retain the Qwen Nothink variants and the remaining measured proxy references. They are excluded from the main table ranking.

| Model / precision | F1 (%) ↑ | Weights (GiB) ↓ | Model/proxy ↓ | p50 c=1 (s) ↓ | Speed-up ↑ | c=64 rows/h ↑ | Model/proxy ↑ |
|:--|--:|--:|--:|--:|--:|--:|--:|
| Llama 3.1 8B proxy JO bf16 — gemma4-e4b-jo | 96.63 | 15.0 | 1.00× | 5.25 | 1.00× | 31,795 | 1.00× |
| Llama 3.1 8B proxy FT bf16 — gemma4-e4b-ft | 96.70 | 15.0 | 1.00× | 4.99 | 1.00× | 31,622 | 1.00× |
| Llama 3.1 8B proxy JO bf16 — prior-4b | 96.63 | 15.0 | 1.00× | 4.99 | 1.00× | 32,385 | 1.00× |
| Qwen 3.5 4B JO Nothink bf16 — prior-4b | 96.33 | 8.6 | 0.57× | 3.59 | 1.39× | 21,488 | 0.66× |
| Qwen 3.5 4B JO Nothink fp8 — prior-4b | 96.39 | 5.3 | 0.36× | 1.86 | 2.68× | 27,404 | 0.85× |
| Qwen 3.5 4B JO Nothink W4A16 — prior-4b | 96.16 | 3.7 | 0.25× | 1.30 | 3.85× | 23,406 | 0.72× |
| Llama 3.1 8B proxy FT bf16 — prior-4b | 96.70 | 15.0 | 1.00× | 5.02 | 1.00× | 31,603 | 1.00× |
| Qwen 3.5 4B FT Nothink bf16 — prior-4b | 96.42 | 8.6 | 0.57× | 3.63 | 1.38× | 21,137 | 0.67× |
| Qwen 3.5 4B FT Nothink fp8 — prior-4b | 95.72 | 5.3 | 0.36× | 1.90 | 2.64× | 26,194 | 0.83× |
| Qwen 3.5 4B FT Nothink W4A16 — prior-4b | 96.09 | 3.7 | 0.25× | 1.31 | 3.84× | 23,352 | 0.74× |
| Llama 3.1 8B proxy JO bf16 — llama-quant | 96.63 | 15.0 | 1.00× | 5.02 | 1.00× | 32,430 | 1.00× |
| Llama 3.1 8B proxy FT bf16 — llama-quant | 96.70 | 15.0 | 1.00× | 4.99 | 1.00× | 31,482 | 1.00× |

## Comparisons at matched precision

### Paired quality comparisons

Each Gemma 4 checkpoint is compared with the same-prompt Llama proxy at the same precision on 1,213 records. Differences and two-sided 95% intervals use 50,000 paired-record bootstrap resamples, seed 42 and linear percentiles. These intervals are exploratory and unadjusted; they condition on the retained checkpoints, including test-selected proxies, and do not extend the primary production-reference families.

| Model | Reference | Precision | Delta F1 (pp) | Unadjusted paired 95% CI (pp) |
|:--|:--|:--|--:|:--|
| Gemma 4 12B JO | Llama 3.1 8B proxy JO | bf16 | 0.37 | [-0.29, 1.04] |
| Gemma 4 12B JO | Llama 3.1 8B proxy JO | fp8 | 0.27 | [-0.37, 0.92] |
| Gemma 4 12B JO | Llama 3.1 8B proxy JO | W4A16 | 1.03 | [0.34, 1.73] |
| Gemma 4 12B FT | Llama 3.1 8B proxy FT | bf16 | 0.15 | [-0.51, 0.81] |
| Gemma 4 12B FT | Llama 3.1 8B proxy FT | fp8 | 0.41 | [-0.28, 1.11] |
| Gemma 4 12B FT | Llama 3.1 8B proxy FT | W4A16 | 0.57 | [-0.05, 1.20] |
| Gemma 4 E4B JO | Llama 3.1 8B proxy JO | bf16 | -0.72 | [-1.47, 0.03] |
| Gemma 4 E4B JO | Llama 3.1 8B proxy JO | fp8 | -1.11 | [-1.90, -0.32] |
| Gemma 4 E4B JO | Llama 3.1 8B proxy JO | W4A16 | -0.70 | [-1.51, 0.10] |
| Gemma 4 E4B FT | Llama 3.1 8B proxy FT | bf16 | -0.88 | [-1.61, -0.16] |
| Gemma 4 E4B FT | Llama 3.1 8B proxy FT | fp8 | -0.61 | [-1.37, 0.14] |
| Gemma 4 E4B FT | Llama 3.1 8B proxy FT | W4A16 | -1.29 | [-2.06, -0.52] |

### Resource ratios across benchmark blocks

These resource ratios compare medians from the Gemma 4 benchmark block with the same-prompt, same-precision Llama setting in the separate `llama-quant` block. They are descriptive comparisons across blocks, not paired within-block serving measurements. Repeated bf16 proxies provide a consistency check; the paired quality intervals above do not quantify uncertainty in these resource ratios. Latency speed-up uses concurrency 1; throughput ratios use concurrency 64.

| Model | Reference | Precision | Weights model/reference ↓ | Latency speed-up c=1 reference/model ↑ | Throughput c=64 model/reference ↑ |
|:--|:--|:--|--:|--:|--:|
| Gemma 4 12B JO | Llama 3.1 8B proxy JO | bf16 | 1.52× | 0.54× | 0.54× |
| Gemma 4 12B JO | Llama 3.1 8B proxy JO | fp8 | 1.51× | 0.61× | 0.52× |
| Gemma 4 12B JO | Llama 3.1 8B proxy JO | W4A16 | 1.46× | 0.56× | 0.45× |
| Gemma 4 12B FT | Llama 3.1 8B proxy FT | bf16 | 1.52× | 0.54× | 0.53× |
| Gemma 4 12B FT | Llama 3.1 8B proxy FT | fp8 | 1.51× | 0.61× | 0.50× |
| Gemma 4 12B FT | Llama 3.1 8B proxy FT | W4A16 | 1.46× | 0.58× | 0.45× |
| Gemma 4 E4B JO | Llama 3.1 8B proxy JO | bf16 | 1.01× | 1.33× | 1.41× |
| Gemma 4 E4B JO | Llama 3.1 8B proxy JO | fp8 | 1.30× | 1.38× | 1.33× |
| Gemma 4 E4B JO | Llama 3.1 8B proxy JO | W4A16 | 1.80× | 1.17× | 1.20× |
| Gemma 4 E4B FT | Llama 3.1 8B proxy FT | bf16 | 1.01× | 1.33× | 1.46× |
| Gemma 4 E4B FT | Llama 3.1 8B proxy FT | fp8 | 1.30× | 1.39× | 1.39× |
| Gemma 4 E4B FT | Llama 3.1 8B proxy FT | W4A16 | 1.80× | 1.16× | 1.26× |

## Checkpoint selection (Table 1)

| Model | Rule | Step | Validation F1 (%) | Sweep test F1 (%) | Delta vs production (pp) | Gap to test oracle (pp) |
|:--|:--|--:|--:|--:|--:|--:|
| Gemma 4 12B JO | Validation F1 | 1400 | 97.40 | 97.04 | 0.09 | 0.00 |
| Gemma 4 12B JO | Validation loss | 1200 | 97.21 | 96.22 | -0.73 | 0.82 |
| Gemma 4 12B JO | Final checkpoint | 2298 | 97.17 | 96.61 | -0.34 | 0.43 |
| Gemma 4 12B JO | Test oracle (descriptive) | 1400 | 97.40 | 97.04 | 0.09 | 0.00 |
| Gemma 4 12B FT | Validation F1 | 2298 | 97.32 | 96.76 | -0.19 | 0.03 |
| Gemma 4 12B FT | Validation loss | 1200 | 97.30 | 96.34 | -0.61 | 0.45 |
| Gemma 4 12B FT | Final checkpoint | 2298 | 97.32 | 96.76 | -0.19 | 0.03 |
| Gemma 4 12B FT | Test oracle (descriptive) | 2200 | 97.26 | 96.79 | -0.16 | 0.00 |
| Gemma 4 E4B JO | Validation F1 | 2298 | 96.68 | 95.87 | -1.08 | 0.12 |
| Gemma 4 E4B JO | Validation loss | 800 | 95.40 | 93.99 | -2.96 | 2.00 |
| Gemma 4 E4B JO | Final checkpoint | 2298 | 96.68 | 95.87 | -1.08 | 0.12 |
| Gemma 4 E4B JO | Test oracle (descriptive) | 1400 | 96.28 | 95.99 | -0.95 | 0.00 |
| Gemma 4 E4B FT | Validation F1 | 1400 | 96.83 | 95.90 | -1.05 | 0.25 |
| Gemma 4 E4B FT | Validation loss | 1000 | 96.59 | 95.21 | -1.74 | 0.94 |
| Gemma 4 E4B FT | Final checkpoint | 2298 | 96.75 | 95.77 | -1.18 | 0.38 |
| Gemma 4 E4B FT | Test oracle (descriptive) | 1800 | 96.79 | 96.15 | -0.80 | 0.00 |

Validation-F1 selection maximizes the fractional score rounded to four decimal places (0.01 F1 percentage point), then validation strict record exact-match count, then prefers the earlier step. All 48 primary tie-break counts and 12 replicate counts are retained in checkpoints.csv. Validation-loss ties prefer the earlier step. The final checkpoint is a no-selection baseline; the test oracle is descriptive. Retained selection contrasts need not equal subtraction of rounded F1 scores; final-checkpoint contrasts follow the manuscript’s rounded-score calculation. Validation-F1 selection was adopted after inspection of test sweeps. Sweep and serving-evaluation scores are distinct observations; headline results consistently use the complete serving evaluation.

## Hosted reference

Evaluation date: 2026-10-19.

| Setting | F1 (%) | Raw exact (%) | Delta F1 (pp) | 95% CI (pp) | Supplementary lower bound (pp) |
|:--|--:|--:|--:|:--|--:|
| JO, 0-shot | 95.68 | 84.34 | -1.27 | [-2.13, -0.42] | -2.389 |
| FT, 0-shot | 95.96 | 84.67 | -0.98 | [-1.84, -0.13] | -2.089 |
| JO, 10-shot | 96.91 | 87.88 | -0.04 | [-0.84, 0.77] | -1.076 |
| FT, 10-shot | 96.65 | 87.63 | -0.29 | [-1.09, 0.51] | -1.319 |

Hosted bounds use the same alpha/9 cutoff outside the primary families. No setting clears −1 pp; this does not prove a deficit greater than one point.

## Second training seed (Table 5)

The E4B JO replicate uses training/data seed 43 with the same prepared split, recipe and six-epoch schedule. It is reported separately from the four primary seed-42 trajectories. The validation-resampling summaries below cover only those primary runs; no resampling or serving benchmark is reported for this replicate.

| Model | Rule | Step | Validation F1 (%) | Sweep test F1 (%) | Delta vs production (pp) | Gap to test oracle (pp) |
|:--|:--|--:|--:|--:|--:|--:|
| Gemma 4 E4B JO (seed 42) | Validation F1 | 2298 | 96.68 | 95.87 | -1.08 | 0.12 |
| Gemma 4 E4B JO (seed 42) | Validation loss | 800 | 95.40 | 93.99 | -2.96 | 2.00 |
| Gemma 4 E4B JO (seed 42) | Final checkpoint | 2298 | 96.68 | 95.87 | -1.08 | 0.12 |
| Gemma 4 E4B JO (seed 42) | Test oracle (descriptive) | 1400 | 96.28 | 95.99 | -0.95 | 0.00 |
| Gemma 4 E4B JO (seed 43) | Validation F1 | 1200 | 96.87 | 95.97 | -0.98 | 0.00 |
| Gemma 4 E4B JO (seed 43) | Validation loss | 800 | 96.39 | 94.96 | -1.99 | 1.01 |
| Gemma 4 E4B JO (seed 43) | Final checkpoint | 2298 | 96.78 | 95.01 | -1.94 | 0.96 |
| Gemma 4 E4B JO (seed 43) | Test oracle (descriptive) | 1200 | 96.87 | 95.97 | -0.98 | 0.00 |

The seed-43 run took 16.4 hours. Validation-F1 selection and the observed test oracle both choose step 1200, with 95.97% test F1: -0.98 points against production, exploratory unadjusted paired-record 95% CI [-1.68, -0.29]. The interval uses 50,000 paired-record resamples; it measures record uncertainty, not variation across training seeds, and makes no confirmatory non-inferiority claim.

For seed 43, final minus loss-selected test F1 is 0.05 points, and validation-selected minus final is 0.96 points. The ordering repeats across these two seeds, but one replicate of one run does not establish the ranking or size of the gaps.

## Primary quality comparisons (Table 2)

Production canonical micro-F1 is 96.95% and raw record exact match is 92.58%. Each JO/FT family retains the divisor of nine planned cells. A one-sided lower bound strictly above −1 percentage point clears the planning criterion; this does not establish superiority or operational equivalence. Two-sided 95% intervals are exploratory. Hosted references above are supplementary; local Llama proxies remain descriptive.

| Configuration | F1 (%) | Raw exact (%) | Delta F1 (pp) | 95% CI (pp) | Adjusted lower bound (pp) | Clears −1 pp |
|:--|--:|--:|--:|:--|--:|:--|
| 12B JO bf16 | 97.00 | 92.75 | 0.05 | [-0.60, 0.71] | -0.785 | Yes |
| 12B JO fp8 | 96.86 | 92.50 | -0.09 | [-0.73, 0.57] | -0.923 | Yes |
| 12B JO W4A16 | 96.88 | 91.92 | -0.07 | [-0.72, 0.58] | -0.917 | Yes |
| 12B FT bf16 | 96.85 | 92.09 | -0.10 | [-0.74, 0.54] | -0.934 | Yes |
| 12B FT fp8 | 96.72 | 91.67 | -0.22 | [-0.88, 0.42] | -1.071 | No |
| 12B FT W4A16 | 96.66 | 91.84 | -0.29 | [-0.94, 0.35] | -1.119 | No |
| E4B JO bf16 | 95.91 | 90.19 | -1.04 | [-1.76, -0.32] | -1.983 | No |
| E4B JO fp8 | 95.48 | 89.12 | -1.47 | [-2.22, -0.72] | -2.449 | No |
| E4B JO W4A16 | 95.15 | 88.71 | -1.80 | [-2.55, -1.07] | -2.782 | No |
| E4B FT bf16 | 95.82 | 89.69 | -1.13 | [-1.90, -0.37] | -2.127 | No |
| E4B FT fp8 | 95.70 | 89.45 | -1.25 | [-2.02, -0.49] | -2.259 | No |
| E4B FT W4A16 | 94.80 | 87.96 | -2.15 | [-2.99, -1.34] | -3.235 | No |

## Field-level quality (Table 7)

Canonical field F1 (%) on 1,213 records. Support counts nonempty human-labelled values. Gemma columns use validation-selected JO checkpoints. These field comparisons are descriptive, especially for COUNTRY with eight nonempty values.

| Field | Support | Production | 12B bf16 | 12B W4A16 | E4B bf16 | E4B W4A16 |
|:--|--:|--:|--:|--:|--:|--:|
| INTERMEDIARY | 112 | 94.32 | 96.86 | 95.96 | 96.92 | 95.15 |
| MERCHANT_NAME | 1213 | 95.38 | 94.81 | 94.89 | 94.31 | 93.16 |
| DYNAMIC_DESCRIPTOR | 229 | 94.81 | 95.01 | 95.22 | 92.87 | 92.41 |
| STREET | 372 | 94.21 | 93.71 | 93.55 | 91.03 | 90.40 |
| ZIP | 382 | 98.81 | 99.87 | 99.87 | 99.34 | 99.08 |
| CITY | 643 | 97.89 | 98.12 | 97.58 | 96.48 | 94.97 |
| STATE | 811 | 99.63 | 99.69 | 99.63 | 99.32 | 99.38 |
| COUNTRY | 8 | 93.33 | 94.12 | 100.00 | 93.33 | 93.33 |
| PHONE | 75 | 97.96 | 99.34 | 98.68 | 98.68 | 98.01 |
| WEBSITE | 92 | 98.90 | 99.45 | 98.36 | 92.74 | 92.90 |

## Decision guide (Table 4)

These labels apply only to the six configurations in this guide. Local Llama proxies are separately trained and test-selected; their precision variants have descriptive quality comparisons. Device-hours and GPU energy per million records use the full 192-request workload at concurrency 64, including ramp-down. For each repetition, H = 1,000,000 × wall seconds / (3,600 × 192), and GPU kWh/M = H × mean GPU watts / 1,000. Reported quantities are medians of these repetition-level estimates, not inverse headline throughput or products of medians. GPU telemetry excludes CPU and other system power. These short-workload extrapolations are not production costs or sustained capacity.

| Priority in this guide | Configuration | F1 (%) ↑ | Delta F1 (pp) ↑ | Weights (GiB) ↓ | p50 c=1 (s) ↓ | c=64 rows/h ↑ | Device-hours/M ↓ | GPU kWh/M ↓ |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| Llama architecture (proxy) | Llama 8B JO bf16 | 96.63 | -0.32 | 15.0 | 4.99 | 32,706 | 28.2 | 1.7 |
| Highest F1 in this guide | 12B JO bf16 | **97.00** | **+0.05** | 22.8 | 9.33 | 17,355 | 54.3 | 3.1 |
| Within criterion, half proxy weight memory | 12B JO W4A16 | 96.88 | -0.07 | 7.8 | 2.78 | 27,902 | 31.1 | 2.0 |
| Same proxy architecture, fp8 | Llama 8B JO fp8 | 96.59 | -0.36 | 8.5 | 2.72 | 51,132 | 17.7 | 1.1 |
| Lowest memory in this guide | Llama 8B JO W4A16 | 95.84 | -1.11 | **5.4** | 1.57 | 61,547 | 14.0 | 1.1 |
| Fastest in this guide | E4B JO W4A16 | 95.15 | -1.80 | 9.7 | **1.35** | **74,143** | **12.0** | **0.7** |

## Validation-selection stability

Each scheme re-applies checkpoint selection 2,000 times, then maps each selected step to its retained sweep test F1. Bootstrap samples contain 668 validation records drawn with replacement; random halves contain 334 without replacement. These summaries quantify sensitivity on fixed data and trajectories, not uncertainty on new data or repeated training runs. They differ from the paired-record quality intervals.

| Model | Scheme | Same full-validation pick (%) | Beats loss pick (%) | Matches or beats final (%) | Expected test F1 (%) | Expected delta vs final (pp) |
|:--|:--|--:|--:|--:|--:|--:|
| Gemma 4 12B JO | bootstrap | 41.8 | 82.0 | 80.0 | 96.769 | 0.159 |
| Gemma 4 12B JO | random_half | 42.9 | 81.8 | 79.6 | 96.772 | 0.162 |
| Gemma 4 12B FT | bootstrap | 21.8 | 66.0 | 50.0 | 96.594 | -0.166 |
| Gemma 4 12B FT | random_half | 18.6 | 64.1 | 46.0 | 96.583 | -0.177 |
| Gemma 4 E4B JO | bootstrap | 27.5 | 100.0 | 39.6 | 95.584 | -0.286 |
| Gemma 4 E4B JO | random_half | 25.1 | 100.0 | 42.3 | 95.579 | -0.291 |
| Gemma 4 E4B FT | bootstrap | 28.3 | 85.9 | 76.8 | 95.836 | 0.066 |
| Gemma 4 E4B FT | random_half | 25.4 | 83.2 | 72.0 | 95.813 | 0.043 |

## Paired field errors

A loss means production is correct and the candidate is wrong on a nonempty gold field; a win reverses those roles. The first matching string relation supplies the loss type. Counts are descriptive, exclude shared errors from the loss taxonomy, and do not diagnose causes or establish label correctness. For W4A16, shared/new/resolved losses compare with the same model at bf16. All nine loss-type counts and production-loss counts are retained in error-analysis.csv.

| Field | Model / precision | Losses | Wins | Shared with bf16 | Identical loss outputs | New W4A16 losses | Resolved bf16 losses |
|:--|:--|--:|--:|--:|--:|--:|--:|
| MERCHANT_NAME | Gemma 4 12B JO bf16 | 33 | 26 | — | — | — | — |
| MERCHANT_NAME | Gemma 4 12B JO W4A16 | 30 | 24 | 26 | 26 | 4 | 7 |
| MERCHANT_NAME | Gemma 4 E4B JO bf16 | 33 | 20 | — | — | — | — |
| MERCHANT_NAME | Gemma 4 E4B JO W4A16 | 44 | 17 | 26 | 26 | 18 | 7 |
| MERCHANT_NAME | Llama 3.1 8B proxy JO bf16 | 25 | 15 | — | — | — | — |
| MERCHANT_NAME | Llama 3.1 8B proxy JO W4A16 | 35 | 11 | 21 | 21 | 14 | 4 |
| STREET | Gemma 4 12B JO bf16 | 6 | 6 | — | — | — | — |
| STREET | Gemma 4 12B JO W4A16 | 8 | 6 | 5 | 4 | 3 | 1 |
| STREET | Gemma 4 E4B JO bf16 | 16 | 6 | — | — | — | — |
| STREET | Gemma 4 E4B JO W4A16 | 16 | 5 | 14 | 14 | 2 | 2 |
| STREET | Llama 3.1 8B proxy JO bf16 | 7 | 3 | — | — | — | — |
| STREET | Llama 3.1 8B proxy JO W4A16 | 14 | 2 | 7 | 7 | 7 | 0 |
| WEBSITE | Gemma 4 12B JO bf16 | 1 | 2 | — | — | — | — |
| WEBSITE | Gemma 4 12B JO W4A16 | 2 | 2 | 1 | 1 | 1 | 0 |
| WEBSITE | Gemma 4 E4B JO bf16 | 7 | 0 | — | — | — | — |
| WEBSITE | Gemma 4 E4B JO W4A16 | 7 | 2 | 4 | 4 | 3 | 3 |
| WEBSITE | Llama 3.1 8B proxy JO bf16 | 3 | 2 | — | — | — | — |
| WEBSITE | Llama 3.1 8B proxy JO W4A16 | 4 | 0 | 2 | 2 | 2 | 1 |

## Production training context

Operator-reported training time was under one hour for full fine-tuning of Llama 3.1 8B through Databricks' managed Mosaic service. The service did not expose GPU type or count to the team; exact runtime and detailed training settings remain unavailable.
