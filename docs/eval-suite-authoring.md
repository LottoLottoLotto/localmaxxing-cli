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

Use `--kind judge` for open-ended LM-Judge/rubric-scored questions.

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

## Agent Self-Correction

When a CLI command fails, use the error code and fix hints to correct the next attempt:

- `suite_validation_failed`: edit the suite JSON fields listed in `Details`.
- `lm_eval_metric_not_found`: inspect the available metrics in `Details`, then adjust suite `scoringMethod` or rerun lm-eval with the expected tasks.
- `model_server_error` or `network_error`: check `--base-url`, `--model`, and whether the local OpenAI-compatible server is running.
- `judge_config_missing` or `judge_score_parse_failed`: pass judge endpoint/model/API key or make the judge return strict JSON with a 0-1 score.
- `api_error`: fix API validation details, auth, duplicate slug, or rate-limit issues based on the status code.
