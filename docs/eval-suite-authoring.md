# Eval Suite Authoring

This guide is intended for AI agents and humans creating new LocalMaxxing eval suites.

## Commands

Create a starter suite:

```bash
lmx eval suite init \
  --slug my-topic-eval \
  --name "My Topic Eval" \
  --category reasoning \
  --kind multiple_choice \
  --out my-topic-eval.json
```

Validate it:

```bash
lmx eval suite validate my-topic-eval.json
```

Submit it:

```bash
lmx eval suite submit my-topic-eval.json --api-key bhk_...
```

Submitted suites start as `PENDING` and appear publicly after admin approval.

## Suite Types

Use `--kind multiple_choice` for objective questions with choices and a gold answer.

Use `--kind qa` for short exact-answer questions.

Use `--kind math` for numeric word-problems (GSM8K-style **one-off / custom** datasets). The model is prompted to reason step by step, and the runner extracts the final answer from the chain-of-thought before scoring — so you do **not** have to force answer-only output. This is the recommended way to build private math evals when public benchmarks like GSM8K have leaked into training data.

Use `--kind judge` for open-ended LM-Judge/rubric-scored questions.

Use `--kind loglikelihood` for multiple-choice questions scored by forced-continuation log-probability (the way lm-eval scores MMLU/HellaSwag/ARC) instead of by parsing generated text. Requires a `/v1/completions` endpoint exposing `echo`+`logprobs` (vLLM, SGLang, llama.cpp server); chat-only servers cannot run it. Tune `runConfig.loglikelihoodTarget` (`choice_text` ranks the full answer text, `letter` ranks " A"/" B"… after an `Answer:` prompt) and `runConfig.loglikelihoodNorm` (`byte` = acc_norm, `none` = acc). Scores are directionally comparable to lm-eval but not byte-identical (the context/continuation split is by character offset, not exact tokenization).

## Inline Multiple-Choice Item

```json
{
  "input": "In TCP congestion control, what does slow start primarily increase?",
  "choices": ["MTU", "Congestion window", "TTL", "Checksum size"],
  "gold": "B"
}
```

## Inline QA Item

```json
{
  "input": "What is the time complexity of binary search on a sorted array?",
  "gold": "O(log n)"
}
```

## Answer Extraction (chain-of-thought scoring)

For `exact_match` tasks where the model reasons before answering (math word-problems, multi-step QA), set `answerExtraction` on the task so the final answer is pulled out of the response before comparison:

- `"last_number"` — take the last number-like token (handles `$`, commas, decimals, `%`, sign). This is how `--kind math` is configured and mirrors lm-eval's GSM8K flexible-extract.
- `"regex"` — take the last match of `answerRegex` (capture group 1 if present, else the whole match), e.g. `answerRegex: "answer:\\s*\\(([A-D])\\)"`.
- omitted / `"none"` — compare the full response (current default).

Matching is numeric-aware: `72`, `72.0`, `1,000` vs `1000`, and `50%` vs `50` all compare equal. The extracted value is recorded on each artifact as `extractedAnswer` for auditability.

Example math task:

```json
{
  "key": "math",
  "taskType": "qa",
  "promptTemplate": "Solve the problem. Show your reasoning, then state the final answer.\n\n{{input}}",
  "answerExtraction": "last_number",
  "maxNewTokens": 512,
  "dataset": { "source": "inline", "items": [ { "input": "Natalia sold 48 clips in April and half as many in May. How many altogether?", "gold": "72" } ] }
}
```

## Inline LM-Judge Item

```json
{
  "input": "Explain the tradeoffs between B-tree and LSM-tree storage engines.",
  "referenceAnswer": "A strong answer discusses read/write amplification, compaction, range scans, write throughput, and workload fit.",
  "rubric": "Score 0 to 1. Reward accuracy, mention of core tradeoffs, and concrete examples. Penalize hallucinated claims."
}
```

## Agent Checklist

1. Research the field and define the capability being measured.
2. Create original questions; do not copy private or unlicensed benchmark items.
3. Prefer 20-100 items for a useful public suite.
4. For multiple-choice items, randomize answer positions and store the correct choice as `gold`.
5. For exact-answer QA, keep expected answers short and unambiguous.
6. For judge evals, provide a precise `rubric` and optional `referenceAnswer`.
7. Ensure prompts never include gold answers.
8. Put source notes, methodology, and license context in `description` or `sourceUrl`.
9. Run `lmx eval suite validate <file>` before submitting.
10. Submit with `lmx eval suite submit <file> --api-key bhk_...`.

## Label Redaction

Inline gold answers and judging references are stored in the suite document for scoring, but public API responses redact these fields:

`gold`, `tests`, `expectedOutput`, `referenceAnswer`, and `rubric`.

## Running Approved Suites

```bash
lmx eval run my-topic-eval \
  --model Qwen/Qwen3-8B \
  --base-url http://localhost:8000 \
  --hardware hardware.json \
  --api-key bhk_... \
  --submit
```

For LM-Judge suites:

```bash
lmx eval run my-topic-judge-eval \
  --model Qwen/Qwen3-8B \
  --base-url http://localhost:8000 \
  --judge-base-url https://api.openai.com \
  --judge-model gpt-4.1-mini \
  --judge-api-key "$OPENAI_API_KEY" \
  --hardware hardware.json \
  --api-key bhk_... \
  --submit
```

## Offline Pull, Run, and Deferred Submit

To inspect a suite's dataset, run without a live site connection, or avoid losing a completed run if the site is temporarily unavailable, pull the suite once and run against the local copy.

```bash
# 1. Pull the suite + datasets (gold labels included) to a local directory.
lmx eval pull my-topic-eval --api-key bhk_... --out localmaxxing-eval-my-topic-eval

# 2. Run fully offline against the pulled copy — no site connection or API key needed.
lmx eval run my-topic-eval \
  --suite-file localmaxxing-eval-my-topic-eval/suite.json \
  --model Qwen/Qwen3-8B --base-url http://localhost:8000 \
  --hardware hardware.json --out run.json

# 3. Submit the saved run later, once the site is reachable (no re-run needed).
lmx eval submit run.json --model Qwen/Qwen3-8B --hardware hardware.json --api-key bhk_...
```

`eval pull` writes `suite.json` (datasets resolved to inline items so `eval run --suite-file` works offline), one `<taskKey>.jsonl` per task for inspection, and `manifest.json`. **The pulled files contain gold labels — do not publish them.**

`eval submit` re-sends a run payload that `eval run --out` already wrote to disk, so a failed submit (e.g. site outage) never forces a GPU re-run. Pass `--model`/`--hardware` to fill fields an offline run omitted, and `--dry-run` to validate first.

## Agent Self-Correction

When a CLI command fails, use the error code and fix hints to correct the next attempt:

- `suite_validation_failed`: edit the suite JSON fields listed in `Details`.
- `lm_eval_metric_not_found`: inspect the available metrics in `Details`, then adjust suite `scoringMethod` or rerun lm-eval with the expected tasks.
- `model_server_error` or `network_error`: check `--base-url`, `--model`, and whether the local OpenAI-compatible server is running.
- `judge_config_missing` or `judge_score_parse_failed`: pass judge endpoint/model/API key or make the judge return strict JSON with a 0-1 score.
- `api_error`: fix API validation details, auth, duplicate slug, or rate-limit issues based on the status code.
