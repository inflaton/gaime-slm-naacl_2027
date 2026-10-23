# Aggregate results

All numbers below are retained study aggregates. Synthetic demonstration outputs are separate. Quality comparisons use 1,213 paired records. F1 and exact-match rates are percentages; differences are percentage points.

## Main serving comparison

This is the manuscript table subset. Values are medians from three runs with server restarts, warm-up and matched requests; model order was rotated. Ratios use the same-prompt **bf16** proxy measured in the same block. Displayed bf16 Llama rows come from the 12B blocks; fp8/W4A16 Llama ratios use bf16 baselines from the separate `llama-quant` block. Bold marks best displayed candidate values, including ties at the displayed precision. Proxy rows are excluded from ranking. Prior 4B adapters and Llama proxies are test-selected; the displayed Qwen adapters use the Think training template. Loaded weights exclude server and KV-cache memory.

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
| &emsp;W4A16 | 95.15 | 9.7 | 0.64× | 1.35 | 3.89× | 74,143 | 2.33× |
| E4B FT bf16 | 95.82 | 15.2 | 1.01× | 3.75 | 1.33× | 45,823 | 1.45× |
| &emsp;fp8 | 95.70 | 11.1 | 0.74× | 1.97 | 2.54× | 67,825 | 2.14× |
| &emsp;W4A16 | 94.80 | 9.7 | 0.64× | 1.36 | 3.68× | 73,376 | 2.32× |

### Prior 4B adapters: retained test-selected checkpoints

| Model / precision | F1 (%) ↑ | Weights (GiB) ↓ | Model/proxy ↓ | p50 c=1 (s) ↓ | Speed-up ↑ | c=64 rows/h ↑ | Model/proxy ↑ |
|:--|--:|--:|--:|--:|--:|--:|--:|
| Qwen 3.5 4B JO bf16 | 96.62 | 8.6 | 0.57× | 3.53 | 1.42× | 21,486 | 0.66× |
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

### Llama 8B proxies: production architecture at three precisions

| Model / precision | F1 (%) ↑ | Weights (GiB) ↓ | Model/proxy ↓ | p50 c=1 (s) ↓ | Speed-up ↑ | c=64 rows/h ↑ | Model/proxy ↑ |
|:--|--:|--:|--:|--:|--:|--:|--:|
| Llama 8B JO bf16 | 96.63 | 15.0 | 1.00× | 4.99 | 1.00× | 32,706 | 1.00× |
| &emsp;fp8 | 96.59 | 8.5 | 0.57× | 2.72 | 1.85× | 51,132 | 1.58× |
| &emsp;W4A16 | 95.84 | 5.4 | 0.36× | 1.57 | 3.19× | 61,547 | 1.90× |
| Llama 8B FT bf16 | 96.70 | 15.0 | 1.00× | 4.97 | 1.00× | 31,426 | 1.00× |
| &emsp;fp8 | 96.31 | 8.5 | 0.57× | 2.73 | 1.82× | 48,969 | 1.56× |
| &emsp;W4A16 | 96.09 | 5.4 | 0.36× | 1.57 | 3.17× | 58,300 | 1.85× |

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

These resource ratios compare medians from the Gemma 4 benchmark block with the same-prompt, same-precision Llama setting in the separate `llama-quant` block. They are descriptive comparisons across blocks, not paired within-block serving measurements. Repeated bf16 proxies provide a consistency check; the paired quality intervals above do not quantify uncertainty in these resource ratios.

| Model | Reference | Precision | Weights model/reference ↓ | Latency speed-up reference/model ↑ | Throughput model/reference ↑ |
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

## Checkpoint selection

| Model | Validation-loss step | Validation-F1 step | Sweep test-F1 gain (pp) | Serving-evaluation F1 (%) |
|:--|--:|--:|--:|--:|
| Gemma 4 12B JO | 1200 | 1400 | 0.82 | 97.00 |
| Gemma 4 12B FT | 1200 | 2298 | 0.42 | 96.85 |
| Gemma 4 E4B JO | 800 | 2298 | 1.88 | 95.91 |
| Gemma 4 E4B FT | 1000 | 1400 | 0.69 | 95.82 |

The test oracle is descriptive. Validation-F1 selection was adopted after inspection of test sweeps. Sweep and serving-evaluation scores are distinct observations; headline results consistently use the complete serving evaluation.

## Hosted reference

| Setting | F1 (%) | Raw exact (%) | Delta F1 (pp) | 95% CI (pp) | Supplementary lower bound (pp) |
|:--|--:|--:|--:|:--|--:|
| JO, 0-shot | 95.68 | 84.34 | -1.27 | [-2.13, -0.42] | -2.389 |
| FT, 0-shot | 95.96 | 84.67 | -0.98 | [-1.84, -0.13] | -2.089 |
| JO, 10-shot | 96.91 | 87.88 | -0.04 | [-0.84, 0.77] | -1.076 |
| FT, 10-shot | 96.65 | 87.63 | -0.29 | [-1.09, 0.51] | -1.319 |

Hosted bounds use the same alpha/9 cutoff outside the primary families. No setting clears −1 pp; this does not prove a deficit greater than one point.
