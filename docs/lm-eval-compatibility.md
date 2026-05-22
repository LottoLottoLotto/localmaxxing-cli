# LM-Eval Compatibility

LocalMaxxing supports lm-eval-harness suites by treating lm-eval as the local runner and LocalMaxxing as the result registry/leaderboard.

## Compatibility Summary

Supported today:

- lm-eval task groups and task names, including `leaderboard`, `mmlu`, `mmlu_pro`, `arc_easy`, `arc_challenge`, `hellaswag`, `winogrande`, `piqa`, `openbookqa`, `truthfulqa`, `gsm8k`, `gpqa`, `bbh`, `ifeval`, `humaneval`, `mbpp`, multilingual MMLU variants, and most accuracy/F1/pass@k tasks.
- lm-eval result JSON containing `results` task entries.
- lm-eval result JSON containing `groups` aggregate entries.
- Common metric names like `acc`, `acc_norm`, `exact_match`, `f1`, `macro_f1`, `pass@1`, `pass_at_1`, `inst_level_strict_acc`, and `prompt_level_strict_acc`, with or without `,none` suffixes.
- `rouge1`/`rougeL` style metrics when represented as 0-1 or 0-100 values; 0-100 values are normalized to 0-1 by the CLI.

Not fully supported yet:

- Lower-is-better suites.
- Perplexity/language-modeling suites such as `wikitext`, `c4`, `pile`, or `paloma` where the primary metric is perplexity or bits-per-byte.
- Multimodal evals that require image/audio inputs.
- Code execution artifacts from lm-eval are not imported automatically; aggregate `pass@1` upload is supported.

## Create an LM-Eval Suite

```bash
lmx eval suite init \
  --slug local-open-llm-core \
  --name "Local Open LLM Core" \
  --category leaderboard \
  --runner lm-eval-harness \
  --tasks arc_challenge,hellaswag,mmlu,truthfulqa_mc2,winogrande,gsm8k \
  --out local-open-llm-core.eval-suite.json
```

Validate and submit:

```bash
lmx eval suite validate local-open-llm-core.eval-suite.json
lmx eval suite submit local-open-llm-core.eval-suite.json --api-key bhk_...
```

## Run LM-Eval

```bash
lm-eval run \
  --model hf \
  --model_args pretrained=Qwen/Qwen3-8B \
  --tasks arc_challenge,hellaswag,mmlu,truthfulqa_mc2,winogrande,gsm8k \
  --output_path localmaxxing-lm-eval-results.json
```

Then upload:

```bash
lmx eval run local-open-llm-core \
  --model Qwen/Qwen3-8B \
  --results localmaxxing-lm-eval-results.json \
  --hardware hardware.json \
  --api-key bhk_... \
  --submit
```

## Metric Mapping

LocalMaxxing stores per-task scores in `[0, 1]`. The CLI maps common lm-eval metrics as follows:

| Suite scoringMethod | lm-eval metrics searched |
|---|---|
| `exact_match` | `acc_norm`, `acc`, `exact_match`, `exact`, `em`, `pass@1`, `pass_at_1`, `inst_level_strict_acc`, `prompt_level_strict_acc` |
| `f1` | `f1`, `macro_f1`, `rouge1`, `rougeL` |
| `pass_at_k` | `pass@1`, `pass_at_1`, `pass@k`, `pass_at_k` |
| `llm_judge` | `score`, `acc` |

If parsing fails, the CLI emits `[localmaxxing:error] lm_eval_metric_not_found` with the metrics it found for that task.
