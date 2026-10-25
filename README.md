# NAACL 2027: Local fine-tuning and evaluation artifact

This standalone artifact accompanies **“Quantize the Incumbent or Switch Models? Checkpoint Selection and 4-Bit Serving for Merchant Field Extraction.”** It contains the retained aggregate results, portable evaluation code, documented experiment settings, and clearly synthetic examples for ten-field transaction entity extraction. The aggregate reports and figures can be regenerated offline with Node.js; no GPU, model download, account, or private repository is required.

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

The validation command checks aggregate consistency and exercises the scorer and bootstrap with synthetic records. The test suite checks command-line validation and parser behavior, including malformed outputs that must not earn matches. Regeneration rebuilds all seven CSV files, including `generated/paired-comparisons.csv`, plus `generated/tables.md` and three standalone SVG figures. The scoring and comparison commands print synthetic results; they perform no inference. Add `--out result.json` to save a demonstration result. Outputs are deterministic and contain no timestamps or local source paths.

## Contents

| Path | Contents |
| --- | --- |
| `aggregates/quality.json` | Production reference and 48 model/precision/selection cells: 12 primary, 12 loss-selection sensitivity, 18 prior 4B, and six Llama proxy variants; canonical field F1, raw exact match, differences, intervals, lower bounds, and per-field aggregates. |
| `aggregates/checkpoints.json` | Four complete 12-checkpoint sweeps with validation loss, generated validation F1, strict validation record-match counts for all 48 checkpoints, test F1, selected steps, separate serving-evaluation scores (`serving_evaluation_f1_pct`), observed output-disagreement counts, and training durations. |
| `aggregates/hosted.json` | Four Astra prompt/demonstration settings and supplementary bounds. |
| `aggregates/serving.json` | 42 serving rows: 30 candidate configurations and 12 proxy measurements across benchmark blocks; 210 concurrency points with three-repetition medians and ranges. |
| `aggregates/paired-comparisons.json` | Twelve same-precision paired quality comparisons between validation-selected Gemma 4 and same-prompt Llama proxies, with exploratory unadjusted 95% intervals, plus four bf16 proxy consistency comparisons across benchmark blocks. |
| `configs/` | Sanitized training, serving, quantization, evaluation and software settings, plus an optional LoRA YAML starting template and dataset schema. Supply your own model and data paths; the descriptive JSON settings alone are not launch-ready jobs. |
| `prompts/` | Full JO/FT instruction structures with concrete examples replaced by explicitly fictional ones. These are illustrative, example-sanitized templates, **not byte-identical study prompts**. |
| `fixtures/` | Six wholly synthetic gold records, two sets of synthetic output strings, and one synthetic training-format example. The `.example` domain and synthetic names are fictional. Only these synthetic record-level inputs are included. |
| `lib/evaluation.mjs` | Dependency-free Node scorer and paired canonical-F1 bootstrap for the ten-string-field contract. |
| `tests/` | Offline command-line and parser regression tests using synthetic inputs. |
| `reference-python/` | Portable reference implementations of the original scorer and NumPy paired bootstrap, plus an optional CLI. |
| `generated/` | Rebuilt CSVs, readable result tables, interval plot, checkpoint plot and concurrency plot. |

## Reading the results

Quality rates are percentages. F1 differences, confidence intervals and lower bounds are percentage points. The quality and hosted aggregates compare with retained production outputs on 1,213 common records; the paired-comparison aggregates instead compare each Gemma 4 configuration with the same-prompt Llama proxy at the same precision. Per-field support is the number of records with a nonempty gold value. All published numeric aggregates are retained measurements; none comes from the synthetic examples.

The main configurations maximize generated validation canonical micro-F1 after rounding the fractional rate to four decimal places (0.01 F1 percentage point), then break ties by higher validation strict record exact-match count and earlier step. The original scorer uses Python round (nearest, ties to even). The checkpoint aggregates retain all 48 tie-break counts, so these selections can be reconstructed. Validation loss is a sensitivity analysis, with ties resolved by earlier step. Test-F1 maxima are descriptive oracles within the observed sweep. The reporting choice was retrospective, after inspecting test sweeps; the aggregate artifact does not turn it into a prospective evaluation. Selected-model serving-evaluation scores and original sweep scores are separate observations and must not be substituted for one another. The prior 4B adapters and local Llama proxies have retained test-selected checkpoints only.

Each prompt's primary comparison retains the family divisor of nine despite six completed model/precision cells. A bound strictly greater than −1 F1 point meets the reported criterion. The ordinary 95% intervals are exploratory. Astra's supplementary bounds use the same per-comparison cutoff outside those primary families, without a new family-wide guarantee. Failure to establish non-inferiority does not prove that a model's true deficit exceeds one point.

The twelve same-precision Gemma 4–Llama comparisons use 50,000 paired record resamples and unadjusted two-sided 95% intervals. They are exploratory, have no adjusted lower bounds or non-inferiority decisions, and do not extend the primary families. Their intervals also do not correct for the proxies' test-based checkpoint selection. Reported contrasts come from the retained paired comparisons rather than subtraction of rounded F1 scores.

Serving measurements use three benchmark repetitions. For each repetition, each model server was restarted, warmed with two requests, and measured at concurrency 1, 8, 16, 32, and 64 with 16, 24, 48, 96, and 192 requests respectively. Model order rotated between repetitions. Ranges summarize those repeated measurements; they are not confidence intervals. At concurrency above one, throughput uses completions before the final request starts, excluding ramp-down; at concurrency one it uses wall-clock completion rate. Table 3 ratios use the bf16 proxy measured in the same block and with the same prompt. Both Llama proxies were additionally measured at bf16, fp8 and W4A16 in a separate block; its bf16 values differ from the four primary Gemma 4 blocks' corresponding proxies by at most 4.4% in latency at concurrency 1 and 2% in throughput at concurrency 64. Loaded weight memory excludes the KV cache. The local Llama proxy is separately trained and is not the production model or endpoint.

The main serving table matches the paper's 24 candidates and six Llama variants. Its prior 4B subset is Gemma 3 JO/FT and Qwen 3.5 JO/FT with Think training templates; Qwen is served with thinking disabled. Qwen Nothink results and all block-specific proxy measurements remain available as supplementary aggregates. Bold values mark the best displayed candidate in each column, including displayed ties; proxies are reference rows. This ranking does not equate retained test-selected 4B checkpoints with validation-selected checkpoints. The displayed bf16 Llama rows use the 12B blocks, while fp8 and W4A16 rows use the separate quantized-proxy block. Ratios in generated CSVs are recomputed from released medians rather than copied from rounded paper cells. Main-table throughput, throughput ratios and prose rankings refer to concurrency 64; the concurrency curves report the other measured levels.

GPTQ uses 64 training calibration rows and group size 128. Gemma 4 E4B uses basic full-model calibration to support KV sharing, without layer-to-layer quantization-error propagation; its `per_layer_model_projection` remains bf16. Gemma 4 12B and the Llama proxies use sequential calibration. Cross-model quantization losses therefore reflect both the model and its supported calibration pipeline.

At W4A16, the Llama JO proxy loads 5.4 GiB of weights, compared with 7.8 GiB for Gemma 4 12B JO and 9.7 GiB for E4B JO. Relative to that same-precision proxy, 12B JO has higher F1 (+1.03 points; paired 95% interval [0.34, 1.73]) and 0.45× throughput at concurrency 64. E4B JO delivers 1.20× throughput at concurrency 64, with an F1 point difference of −0.70 [−1.51, 0.10]. These measured trade-offs support comparing model choice with quantization of the existing architecture; they do not establish production equivalence or monetary cost savings. Gemma 3 4B JO W4A16 has the highest measured throughput at concurrency 64, at 81,313 rows/hour, with 94.66 F1 and a test-selected checkpoint.

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

`configs/lora-training-template.yaml` is a Gemma 4 starting point that preserves the LoRA settings and the `empty` template. Prior Gemma 3 adapters used the `gemma3` template and automatic `all` targets for the same seven projection types. Their recorded LLaMA-Factory/Transformers versions were 0.9.5.dev0/5.2.0, compared with 0.9.6.dev0/5.10.4 for Gemma 4; the common main hyperparameters do not make the implementations identical. `configs/dataset_info.example.json` documents the prepared Alpaca schema; `fixtures/training-input.example.jsonl` shows a wholly synthetic raw input row. The single example is a format illustration, not a training or validation set.

In a separately configured environment with compatible Transformers and LLaMA-Factory, use distinct user-supplied training and validation files. Their rows contain `id`, `input` (the descriptor) and `target` (the ten-string-field object). The optional preparation script renders each model-native chat template, appends the assistant end-of-turn marker, and checks token parity through LLaMA-Factory before writing data. This preserves assistant-only supervision rather than applying a second chat template. It performs no training.

```sh
python reference-python/prepare_training_data.py --model MODEL_PATH --prompt prompts/jo-example.txt --train private/train.jsonl --val private/val.jsonl --out private/prepared
llamafactory-cli train configs/lora-training-template.yaml model_name_or_path=MODEL_PATH
```

Replace `MODEL_PATH` with your licensed model location. Use `prompts/ft-example.txt` for the FT variant and a separate output directory for each experiment. Do not train on the held-out test set. Evaluate generated validation outputs for each saved checkpoint with the same prompt and scorer, then apply the documented rounded-F1 rule and its strict-match/earlier-step tie breaks. Node scoring keeps unrounded percentages: divide F1 by 100 and apply the original scorer’s four-decimal fractional rounding before ranking, rather than maximizing the unrounded percentage. The supplied YAML does not automate that generation sweep or checkpoint selection.

After merging the selected adapter into its base model and exporting a compatible bf16 checkpoint with your model tooling, a serving starting command is:

```sh
vllm serve MERGED_MODEL_DIRECTORY --dtype bfloat16 --max-model-len 8192 --max-num-seqs 64 --gpu-memory-utilization 0.85 --enable-prefix-caching
```

Requests must use the same user prompt, temperature 0, seed 42, max_tokens 1024, and the applicable model-native `enable_thinking=false` chat-template setting. The merge/export step and hardware-specific installation are not automated by this artifact. Online fp8 serving and W4A16 calibration need the separate precision settings in `configs/serving.json`; do not reinterpret a LoRA adapter directory as a merged model. These optional commands have not been run during packaging and require your own model/data and GPU environment.

## Reusing the methods

Supply your own appropriately authorized data and model weights. Resolve model-native chat templates, prepare assistant-only training targets, and use the settings in `configs/` with your chosen installation of LLaMA-Factory and vLLM. The sample prompts retain the JO/FT distinction but changing their examples can change model outputs. A production environment must support the selected model, precision and prompt. Moving a trained checkpoint avoids repeating fine-tuning; model conversion, validation and serving can still require compute.

The released files exclude private paths, study record identifiers, source-file digests, trained weights, logs, organization identifiers and author metadata. Public model/package names and upstream software references identify the method. Keep your own data and model outputs under the ignored `private/` directory. The four synthetic JSONL fixtures are tracked and are required by the quick start.

This is a version-controlled standalone package. Git history and hosting-account metadata are separate from the released file contents. For double-blind review, provide an anonymized artifact link or an archive without Git metadata; the existing public ICDM repository and its attribution can remain intact. No license file is included in this standalone package; release terms remain to be selected by the repository owner.
