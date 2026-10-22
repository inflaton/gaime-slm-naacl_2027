# Aggregate results

All numbers below are retained study aggregates. Synthetic demonstration outputs are separate. Quality comparisons use 1,213 paired records. F1 and exact-match rates are percentages; differences are percentage points.

## Serving

Medians of three independently restarted and warmed server repetitions. Model order was rotated each repetition. Ratios use the same-prompt proxy measured in the same block. Bold denotes the best candidate in each column; proxy reference rows are excluded from ranking. Prior 4B checkpoints were retained test-selected references, not validation-selected competitors.

| Model / precision | F1 (%) ↑ | Weights (GiB) ↓ | Model/proxy ↓ | p50 c=1 (s) ↓ | Speed-up ↑ | c=64 rows/h ↑ | Model/proxy ↑ |
|:--|--:|--:|--:|--:|--:|--:|--:|
| Gemma 4 12B JO bf16 | **97.00** | 22.8 | 1.52 | 9.33 | 0.53 | 17355 | 0.53 |
| Gemma 4 12B JO fp8 | 96.86 | 12.9 | 0.86 | 4.49 | 1.11 | 26427 | 0.81 |
| Gemma 4 12B JO W4A16 | 96.88 | 7.8 | 0.52 | 2.78 | 1.79 | 27902 | 0.85 |
| Gemma 4 12B FT bf16 | 96.85 | 22.8 | 1.52 | 9.28 | 0.54 | 16805 | 0.53 |
| Gemma 4 12B FT fp8 | 96.72 | 12.9 | 0.86 | 4.51 | 1.10 | 24691 | 0.79 |
| Gemma 4 12B FT W4A16 | 96.66 | 7.8 | 0.52 | 2.74 | 1.81 | 26399 | 0.84 |
| Gemma 4 E4B JO bf16 | 95.91 | 15.2 | 1.01 | 3.76 | 1.40 | 45592 | 1.43 |
| Gemma 4 E4B JO fp8 | 95.48 | 11.1 | 0.74 | 1.97 | 2.66 | 67935 | 2.14 |
| Gemma 4 E4B JO W4A16 | 95.15 | 9.7 | 0.64 | 1.35 | 3.89 | 74143 | 2.33 |
| Gemma 4 E4B FT bf16 | 95.82 | 15.2 | 1.01 | 3.75 | 1.33 | 45823 | 1.45 |
| Gemma 4 E4B FT fp8 | 95.70 | 11.1 | 0.74 | 1.97 | 2.54 | 67825 | 2.14 |
| Gemma 4 E4B FT W4A16 | 94.80 | 9.7 | 0.64 | 1.36 | 3.68 | 73376 | 2.32 |
| Qwen 3.5 4B JO, think template bf16 | 96.62 | 8.6 | 0.57 | 3.53 | 1.42 | 21487 | 0.66 |
| Qwen 3.5 4B JO, think template fp8 | 96.35 | 5.3 | 0.36 | 1.85 | 2.70 | 27593 | 0.85 |
| Qwen 3.5 4B JO, think template W4A16 | 96.17 | **3.7** | **0.25** | 1.29 | 3.88 | 23625 | 0.73 |
| Qwen 3.5 4B JO bf16 | 96.33 | 8.6 | 0.57 | 3.59 | 1.39 | 21488 | 0.66 |
| Qwen 3.5 4B JO fp8 | 96.39 | 5.3 | 0.36 | 1.86 | 2.68 | 27404 | 0.85 |
| Qwen 3.5 4B JO W4A16 | 96.16 | **3.7** | **0.25** | 1.30 | 3.85 | 23406 | 0.72 |
| Qwen 3.5 4B FT, think template bf16 | 96.16 | 8.6 | 0.57 | 3.63 | 1.38 | 21166 | 0.67 |
| Qwen 3.5 4B FT, think template fp8 | 96.19 | 5.3 | 0.36 | 1.90 | 2.64 | 26291 | 0.83 |
| Qwen 3.5 4B FT, think template W4A16 | 96.00 | **3.7** | **0.25** | 1.34 | 3.74 | 23511 | 0.74 |
| Qwen 3.5 4B FT bf16 | 96.42 | 8.6 | 0.57 | 3.63 | 1.38 | 21137 | 0.67 |
| Qwen 3.5 4B FT fp8 | 95.72 | 5.3 | 0.36 | 1.90 | 2.64 | 26194 | 0.83 |
| Qwen 3.5 4B FT W4A16 | 96.09 | **3.7** | **0.25** | 1.31 | 3.84 | 23352 | 0.74 |
| Gemma 3 4B FT bf16 | 96.11 | 8.6 | 0.57 | 3.03 | 1.66 | 49773 | 1.57 |
| Gemma 3 4B FT fp8 | 95.71 | 5.5 | 0.37 | 1.65 | 3.05 | 70794 | 2.24 |
| Gemma 3 4B FT W4A16 | 95.07 | 4.2 | 0.28 | **1.15** | **4.35** | **77589** | **2.46** |

### Proxy references (each matched block)

| Model / precision | F1 (%) ↑ | Weights (GiB) ↓ | Model/proxy ↓ | p50 c=1 (s) ↓ | Speed-up ↑ | c=64 rows/h ↑ | Model/proxy ↑ |
|:--|--:|--:|--:|--:|--:|--:|--:|
| Llama 3.1 8B proxy JO — gemma4-12b-jo | 96.63 | 15.0 | 1.00 | 4.99 | 1.00 | 32706 | 1.00 |
| Llama 3.1 8B proxy FT — gemma4-12b-ft | 96.70 | 15.0 | 1.00 | 4.97 | 1.00 | 31426 | 1.00 |
| Llama 3.1 8B proxy JO — gemma4-e4b-jo | 96.63 | 15.0 | 1.00 | 5.25 | 1.00 | 31795 | 1.00 |
| Llama 3.1 8B proxy FT — gemma4-e4b-ft | 96.70 | 15.0 | 1.00 | 4.99 | 1.00 | 31622 | 1.00 |
| Llama 3.1 8B proxy JO — prior-4b | 96.63 | 15.0 | 1.00 | 4.99 | 1.00 | 32385 | 1.00 |
| Llama 3.1 8B proxy FT — prior-4b | 96.70 | 15.0 | 1.00 | 5.02 | 1.00 | 31603 | 1.00 |

## Checkpoint selection

| Model | Validation-loss step | Validation-F1 step | Sweep test-F1 gain (pp) | Primary rerun F1 (%) |
|:--|--:|--:|--:|--:|
| Gemma 4 12B JO | 1200 | 1400 | 0.82 | 97.00 |
| Gemma 4 12B FT | 1200 | 2298 | 0.42 | 96.85 |
| Gemma 4 E4B JO | 800 | 2298 | 1.88 | 95.91 |
| Gemma 4 E4B FT | 1000 | 1400 | 0.69 | 95.82 |

The test oracle is descriptive. The choice to report validation-F1 selection followed inspection of test sweeps. Sweep and rerun scores are distinct observations.

## Hosted reference

| Setting | F1 (%) | Raw exact (%) | Delta F1 (pp) | 95% CI (pp) | Supplementary lower bound (pp) |
|:--|--:|--:|--:|:--|--:|
| JO, 0-shot | 95.68 | 84.34 | -1.27 | [-2.13, -0.42] | -2.389 |
| FT, 0-shot | 95.96 | 84.67 | -0.98 | [-1.84, -0.13] | -2.089 |
| JO, 10-shot | 96.91 | 87.88 | -0.04 | [-0.84, 0.77] | -1.076 |
| FT, 10-shot | 96.65 | 87.63 | -0.29 | [-1.09, 0.51] | -1.319 |

Hosted bounds use the same alpha/9 cutoff outside the primary families. No setting clears −1 pp; this does not prove a deficit greater than one point.
