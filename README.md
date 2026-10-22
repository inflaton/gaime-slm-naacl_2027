# NAACL 2027: Local fine-tuning and evaluation artifact

This standalone artifact accompanies **“Selecting and Serving Gemma 4 Merchant Extractors: Quality and Resource Trade-offs against an 8B Reference.”** It contains the retained aggregate results, portable evaluation code, documented experiment settings, and clearly synthetic examples for ten-field transaction entity extraction. The aggregate reports and figures can be regenerated offline with Node.js; no GPU, model download, account, or private repository is required.

The private transaction dataset, record-level labels and predictions, production model, and trained adapters are not distributed. Aggregate scores alone are insufficient to reconstruct paired bootstrap intervals or rerun the original training and inference. The synthetic fixture demonstrates the evaluation contract; it does not reproduce the study's measured performance.

## Quick start

Requires Node.js 20 or later. Run these commands from this directory:

```sh
node scripts/validate.mjs
node --test
node scripts/regenerate.mjs
node scripts/evaluate.mjs score --gold fixtures/gold.jsonl --pred fixtures/model-a.jsonl
node scripts/evaluate.mjs compare --gold fixtures/gold.jsonl --a fixtures/model-a.jsonl --b fixtures/model-b.jsonl --resamples 50000 --family-size 9
```

The validation command checks aggregate consistency and exercises the scorer and bootstrap with synthetic records. The test suite checks command-line validation and parser behavior, including malformed outputs that must not earn matches. Regeneration rebuilds all six CSV files, `generated/tables.md`, and three standalone SVG figures. The scoring and comparison commands print synthetic results; they perform no inference. Add `--out result.json` to save a demonstration result. Outputs are deterministic and contain no timestamps or local source paths.

## Contents

| Path | Contents |
| --- | --- |
| `aggregates/quality.json` | Production reference and 41 model/precision/selection cells, including 12 primary cells, 12 loss-selection sensitivity cells, 15 prior 4B cells, and two local proxies; canonical field F1, raw exact match, differences, intervals, lower bounds, and per-field aggregates. |
| `aggregates/checkpoints.json` | Four complete 12-checkpoint sweeps with validation loss, generated validation F1, test F1, selected steps, separate primary rerun scores, observed output-disagreement counts, and training durations. |
| `aggregates/hosted.json` | Four Astra prompt/demonstration settings and supplementary bounds. |
| `aggregates/serving.json` | 27 candidate precision configurations and six same-prompt, same-block proxy measurements; 165 concurrency points with three-repetition medians and ranges. |
| `configs/` | Sanitized training, serving, quantization, evaluation and software settings, plus an optional LoRA YAML starting template and dataset schema. Supply your own model and data paths; the descriptive JSON settings alone are not launch-ready jobs. |
| `prompts/` | Full JO/FT instruction structures with concrete examples replaced by explicitly fictional ones. These are illustrative, example-sanitized templates, **not byte-identical study prompts**. |
| `fixtures/` | Six wholly synthetic gold records, two sets of synthetic output strings, and one synthetic training-format example. The `.example` domain and synthetic names are fictional. Only these synthetic record-level inputs are included. |
| `lib/evaluation.mjs` | Dependency-free Node scorer and paired canonical-F1 bootstrap for the ten-string-field contract. |
| `tests/` | Offline command-line and parser regression tests using synthetic inputs. |
| `reference-python/` | Portable reference implementations of the original scorer and NumPy paired bootstrap, plus an optional CLI. |
| `generated/` | Rebuilt CSVs, readable result tables, interval plot, checkpoint plot and concurrency plot. |

## Reading the results

Quality rates are percentages. F1 differences, confidence intervals and lower bounds are percentage points, measured against the retained production outputs on 1,213 common records. Per-field support is the number of records with a nonempty gold value. All published numeric aggregates are retained measurements; none comes from the synthetic examples.

The main configurations use checkpoints selected by generated validation F1. Validation loss is a sensitivity analysis. Test-F1 maxima are descriptive oracles within the observed sweep. The reporting choice was retrospective, after inspecting test sweeps; the aggregate artifact does not turn it into a prospective evaluation. Primary rerun scores and original sweep scores are separate observations and must not be substituted for one another. The prior 4B adapters have retained test-selected checkpoints only.

Each prompt's primary comparison retains the family divisor of nine despite six completed model/precision cells. A bound strictly greater than −1 F1 point meets the reported criterion. The ordinary 95% intervals are exploratory. Astra's supplementary bounds use the same per-comparison cutoff outside those primary families, without a new family-wide guarantee. Failure to establish non-inferiority does not prove that a model's true deficit exceeds one point.

Serving measurements use three independent repetitions. For each repetition, each model server was restarted, warmed with two requests, and measured at concurrency 1, 8, 16, 32, and 64 with 16, 24, 48, 96, and 192 requests respectively. Model order rotated between repetitions. Ranges summarize those repeated measurements; they are not confidence intervals. At concurrency above one, throughput uses completions before the final request starts, excluding ramp-down; at concurrency one it uses wall-clock completion rate. Each comparison uses the proxy measured in the same block and with the same prompt. Loaded weight memory excludes the KV cache. The local Llama proxy is separately trained and is not the production model or endpoint.

The generated serving table bolds the best candidate in each column; proxies are reference rows. That ranking does not make retained test-selected 4B checkpoints equivalent to validation-selected checkpoints. Full block-specific proxy values are provided because latency and throughput vary between blocks. Ratios in generated CSVs are recomputed from the released medians rather than copied from rounded paper cells.

## Scoring and statistical methods

Gold JSONL rows contain `id` and `target`, where `target` is an object or a JSON-encoded object with all ten string-valued fields. Prediction rows contain `id` and `output`; `output` is the raw generated text or null. IDs align paired records and must be unique. A missing/null output stays in the full gold denominator and fails record exact match. Duplicate or unknown prediction IDs are errors. Prediction data supplied by a user stay local.

Canonicalization uppercases values and removes `*`, ASCII spaces, commas and hyphens. Field micro-F1 counts nonempty gold and predicted values, with a true positive when both are nonempty and canonically equal. Raw record exact match requires all ten values to match without canonicalization. Parsing tries a JSON object, a fenced JSON block, an outermost object, and a safe dictionary-literal fallback; text preceding the final `</think>` is ignored.

The Node implementation supports string/null field values and safe string/null Python dictionary literals. Missing and extra keys are counted as schema invalid while the recognized fields are scored. **Non-string numeric, boolean or container values raise an error in Node** instead of approximating Python's legacy string conversion. Use the reference Python scorer for that historical coercion behavior. Node reports unrounded percentages; the reference scorer reports rounded fractional rates. The Node comparator implements canonical micro-F1 intervals; the Python comparator additionally reports precision, recall, raw exact-match intervals and an exploratory exact-match sign test.

The Node bootstrap is fixed to seed 42 using PCG64 XSL-RR and paired record resampling, with linear percentile interpolation. The same sampled records are used for both models, and F1 is recomputed from summed field counts in every resample. It supports up to 2,097,151 records. Passing `--family-size` enables the one-sided Bonferroni cutoff; omitting it produces only a descriptive interval. Statistical interpretation and any family definition remain the analyst's responsibility.

For the optional Python reference scorer, install NumPy:

```sh
python -m pip install -r reference-python/requirements.txt
python reference-python/evaluate.py score --gold fixtures/gold.jsonl --pred fixtures/model-a.jsonl
python reference-python/evaluate.py compare --gold fixtures/gold.jsonl --a fixtures/model-a.jsonl --b fixtures/model-b.jsonl --family-size 9
```

The Node workflow is covered by the bundled offline checks. An independent check of one 50,000-resample primary comparison agreed with its retained lower bound within 1.2e-14 percentage points and reproduced its published 95% interval; the underlying private rows are not included. The Python modules preserve the reference algorithms and were not executed during this artifact's preparation. No training, generation or new benchmark was run to build this package.

## Optional training and serving starting point

`configs/lora-training-template.yaml` preserves the LoRA settings and the `empty` template. `configs/dataset_info.example.json` documents the prepared Alpaca schema; `fixtures/training-input.example.jsonl` shows a wholly synthetic raw input row. The single example is a format illustration, not a training or validation set.

In a separately configured environment with compatible Transformers and LLaMA-Factory, use distinct user-supplied training and validation files. Their rows contain `id`, `input` (the descriptor) and `target` (the ten-string-field object). The optional preparation script renders each model-native chat template, appends the assistant end-of-turn marker, and checks token parity through LLaMA-Factory before writing data. This preserves assistant-only supervision rather than applying a second chat template. It performs no training.

```sh
python reference-python/prepare_training_data.py --model MODEL_PATH --prompt prompts/jo-example.txt --train private/train.jsonl --val private/val.jsonl --out private/prepared
llamafactory-cli train configs/lora-training-template.yaml model_name_or_path=MODEL_PATH
```

Replace `MODEL_PATH` with your licensed model location. Use `prompts/ft-example.txt` for the FT variant and a separate output directory for each experiment. Do not train on the held-out test set. Evaluate generated validation outputs for each saved checkpoint with the same prompt and scorer, then select the best validation-F1 checkpoint. The supplied YAML does not automate that generation sweep.

After merging the selected adapter into its base model and exporting a compatible bf16 checkpoint with your model tooling, a serving starting command is:

```sh
vllm serve MERGED_MODEL_DIRECTORY --dtype bfloat16 --max-model-len 8192 --max-num-seqs 64 --gpu-memory-utilization 0.85 --enable-prefix-caching
```

Requests must use the same user prompt, temperature 0, seed 42, max_tokens 1024, and the applicable model-native `enable_thinking=false` chat-template setting. The merge/export step and hardware-specific installation are not automated by this artifact. Online fp8 serving and W4A16 calibration need the separate precision settings in `configs/serving.json`; do not reinterpret a LoRA adapter directory as a merged model. These optional commands have not been run during packaging and require your own model/data and GPU environment.

## Reusing the methods

Supply your own appropriately authorized data and model weights. Resolve model-native chat templates, prepare assistant-only training targets, and use the settings in `configs/` with your chosen installation of LLaMA-Factory and vLLM. The sample prompts retain the JO/FT distinction but changing their examples can change model outputs. A production environment must support the selected model, precision and prompt. Moving a trained checkpoint avoids repeating fine-tuning; model conversion, validation and serving can still require compute.

The released files exclude private paths, study record identifiers, source-file digests, trained weights, logs, organization identifiers and author metadata. Public model/package names and upstream software references identify the method. Keep your own data and model outputs under the ignored `private/` directory. The four synthetic JSONL fixtures are tracked and are required by the quick start.

This is a version-controlled standalone package. Git history and hosting-account metadata are separate from the released file contents. For double-blind review, provide an anonymized artifact link or an archive without Git metadata; the existing public ICDM repository and its attribution can remain intact. No license file is included in this standalone package; release terms remain to be selected by the repository owner.
