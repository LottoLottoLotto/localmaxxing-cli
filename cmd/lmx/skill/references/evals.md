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
lmx endpoint discover --out endpoint.json --include-server-metadata
lmx eval terminal inspect terminal-bench-2-1 --api-url https://www.localmaxxing.com --verify-bundles --json
lmx eval terminal run terminal-bench-2-1 --api-url https://www.localmaxxing.com --endpoint-file endpoint.json --model Qwen/Qwen3-8B --hardware hardware.json --out completed-terminal-run.json
lmx eval terminal submit completed-terminal-run.json --api-url https://www.localmaxxing.com --dry-run --out terminal-submit-batch.json
lmx eval terminal submit completed-terminal-run.json --api-url https://www.localmaxxing.com --api-key "$LMX_API_KEY"
```

`run` executes tasks; omitting `--submit` keeps the run local, while its `--out`
JSON is a complete monolithic artifact accepted directly by deferred `submit`.
Do not execute the tasks again merely to upload them. Deferred `submit --dry-run`
validates and packages the saved work entirely offline, so it cannot prove
`--api-url` routing; use `inspect` for that. `submit` also accepts legacy
completed checkpoint directories. An isolated/noncanonical directory requires
`--shard-index <n>`; a full canonical 89-task Terminal-Bench 2.1 directory is
partitioned into 10 payloads automatically.

For the built-in agent, `--base-url` and `--endpoint-file` are mutually exclusive
endpoint selectors. The file supplies only its one `ok:true` URL; model, path,
and quantization are re-probed from live endpoint metadata. With neither
selector, uncredentialed localhost probing must find exactly one unambiguous
match. `--model-api-key` requires an explicit URL or trusted endpoint file.
`--served-model` selects an exact live ID; `--model-path`, `--quantization`, and
canonical `--model` are reconciled with live evidence and conflicts are rejected.
Loaded-filename HF resolution succeeds only when the exact filename is verified
in one unambiguous source repo; it never guesses. The run artifact saves
dataset/shard, model and resolution evidence, quantization, hardware, harness,
timing, and token metadata; deferred submit reuses matching saved values.
`--api-url` selects LocalMaxxing; `--base-url` selects model inference.

Failures print a task/outcome/turn/reason table with an `--out` result pointer or
`--trace-dir` path; `--json-status` emits the same categorized
`terminal_failure_summary`. Useful flags include `--task-dir`, `--dataset`,
`--endpoint-file`, `--base-url`, `--served-model`, `--model-path`, `--hf-id`,
`--shard-index`, `--verify-bundles`, `--max-turns`, `--agent-timeout`, `--agent`,
`--agent-cmd`, `--agent-execution`, `--agent-name`, `--container-base-url`,
`--command-timeout`, `--endpoint-timeout-seconds`, `--trace-dir`,
`--cleanup-images`, `--shell-mode`, and `--oracle`. `--agent terminus-2` uses the
release-binary-embedded Harbor adapter.



