# Evals

Use eval commands to measure quality / accuracy rather than speed.

## Suites

```bash
lmx eval suite list --out suites.json
lmx eval suite search reasoning --out reasoning-suites.json
lmx eval suite show hellaswag --out hellaswag-suite.json
lmx eval suite init --slug my-eval --name "My Eval" --category reasoning --out my-eval.json
lmx eval suite validate my-eval.json
lmx eval suite submit my-eval.json --api-key bhk_...
```

User-submitted suites start `PENDING` and appear publicly after admin approval.

## Run a LocalMaxxing suite

```bash
lmx eval run <suite> \
  --model Qwen/Qwen3-8B \
  --base-url http://localhost:8000 \
  --hardware hardware.json \
  --submit
```

`lmx eval execute <suiteSlug> --model <hfId> --base-url http://localhost:8000 --hardware hardware.json --submit` is also available for approved suites.

Eval runs start `PENDING` and appear after admin approval. Eval submissions are rate-limited to 30 per rolling hour, or 300 for Pro users.

## lm-eval harness

Run lm-eval through the CLI:

```bash
lmx eval lm-eval hellaswag --model Qwen/Qwen3-8B --backend hf --hardware hardware.json --dry-run
```

Upload existing lm-eval results with `--results <path>`. Optional dependencies include `lm-eval` and `transformers`.

Useful lm-eval flags:

- `--backend hf`
- `--model-args <args>`
- `--num-fewshot <n>`
- `--lm-eval-bin <path>`
- `--results <path>`

## LM-judge

Use judge flags when a suite or run needs a judge model:

- `--judge-base-url <url>`
- `--judge-model <name>`
- `--judge-api-key <key>`

## Eval shards

Eval shards run a blob-backed dataset against an endpoint and submit pass/fail traces.

```bash
lmx eval shard gsm8k --base-url http://localhost:8000 --questions 200 --dry-run
lmx eval shard hellaswag --base-url http://localhost:8000 --questions 200 --dry-run
lmx eval shard gsm8k --base-url http://localhost:8000 --model Qwen/Qwen3-8B --hardware hardware.json --submit
lmx eval shard status hellaswag --model Qwen/Qwen3-8B
lmx eval shard hellaswag --base-url http://localhost:8000 --model Qwen/Qwen3-8B --hardware hardware.json --missing-only --submit
```

Useful shard flags include `--questions`, `--shard`, `--missing-only`, `--all-missing`, `--rerun`/`--force`, `--answer-extraction`, `--answer-regex`, `--prompt-template`, `--concurrency`, `--artifact-limit`, `--temperature`, `--top-p`, `--quant-format`, and `--model-revision`. The printed `aggregateCoverage` is APPROVED aggregate coverage for the dataset/model/quantization/quantFormat/harness key, not a recent-window sample.

## Terminal-Bench

```bash
lmx eval terminal import ./terminal-bench-tasks --out ./tb-bundles --version 2.1
lmx eval terminal verify ./tb-bundles/smoke --oracle
lmx eval terminal run terminal-bench-2-1 --base-url http://localhost:8000 --model Qwen/Qwen3-8B --hardware hardware.json --submit
lmx eval terminal submit ./completed-terminal-run --dataset terminal-bench-2-1 --hf-id Qwen/Qwen3-8B --hardware hardware.json --quantization Q4_K_M --quant-format gguf --dry-run --out terminal-submit-payload.json
```

The approved Terminal-Bench 2.1 dataset partitions 89 tasks into 10 disjoint
shards. A full deferred checkpoint is validated against the exact canonical task
set and written/submitted as 10 ordered shard payloads; `--shard-index <n>` is
required for an already-isolated shard or any other dataset. Dry-run performs no
network calls. Saved shard-local and full-checkpoint token totals remain in
`runConfig`.

Useful Terminal-Bench flags include `--task-dir`, `--dataset`, `--hf-id`, `--shard-index`, `--max-turns`, `--agent-timeout`, `--agent`, `--agent-cmd`, `--agent-execution`, `--agent-name`, `--container-base-url`, `--command-timeout`, `--endpoint-timeout-seconds`, `--trace-dir`, `--cleanup-images`, `--shell-mode`, and `--oracle`. `--agent terminus-2` uses the release-binary-embedded Harbor adapter.



