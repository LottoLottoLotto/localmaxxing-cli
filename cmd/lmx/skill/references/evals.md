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
lmx eval terminal run terminal-bench-2-1 --api-url https://www.localmaxxing.com --endpoint-file endpoint.json --model Qwen/Qwen3-8B --hardware hardware.json --out completed-terminal-run.json --resume auto --repeat-batch-limit 3
lmx eval terminal recover completed-terminal-run.json.checkpoint --task-id caffe-cifar-10 --container lmx-tb-caffe-recovered --bundle ./tb-bundles/caffe-cifar-10 --result recovered-task.json
lmx eval terminal submit completed-terminal-run.json.checkpoint --api-url https://www.localmaxxing.com --dry-run --out terminal-submit-batch.json
lmx eval terminal submit completed-terminal-run.json.checkpoint --api-url https://www.localmaxxing.com --api-key "$LMX_API_KEY"
```

`run` executes non-resumed tasks. Without `--submit` it emits
`local_execution_results_not_submitted`; deferred `submit --dry-run` emits
`offline_submit_validation_no_execution` and performs no task, Docker, model,
verifier, or network execution. A private v3 checkpoint uses same-directory
file transactions: task wrappers are synced and renamed first, checkpoint
metadata is atomic, and ordered `summary.json` is the final task-set commit.
The directory/parent are synced, a process lock rejects concurrent writers, and
`--resume none` safely initializes a clean checkpoint before the first task.
Legacy checkpoint directories remain supported for submit, never strict resume.

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
`--resume none|auto|<dir>` accepts a task only when complete manifest/task order,
bundle digests/versions, declared/canonical/served model identity, quantization,
hardware hash, runner/harness fingerprint, and run configuration match exactly.
Only complete parsed canonical verifier rewards resume; incomplete, unscored,
or verifier-incomplete wrappers rerun. Partial task events contain the exact
`resumeCommand` and no submit command; the submit command appears only when the
checkpoint is complete.

`recover <checkpoint> --task-id <id> --container <name> --bundle <dir>
[--result <wrapper>]` validates exact v3 provenance, task membership, and the
bundle's deterministic hash/version/task/verifier identity before invoking
Docker. It reruns the canonical verifier against the existing container and
persists only the score derived from the fresh canonical reward. Optional
`--result` input is incomplete-task telemetry only; self-authored pass, scored,
canonical, evidence, and digest claims never establish a score. Completed tasks
cannot be overwritten.

`--api-url` selects LocalMaxxing; `--base-url` selects model inference. Failures
emit `terminal_failure_summary` rows with task ID, verifier summary, turns/max,
artifact path, and last-progress timestamp. `--command-timeout-seconds` (legacy
alias `--command-timeout`) bounds each shell command. Built-in timeouts kill only
that command and return a recovery observation on the next agent turn;
`--endpoint-timeout-seconds` independently bounds model HTTP. Repeated
no-progress batches receive a nudge and then end as `agent_protocol_exhausted`
before canonical verification. Use `--trace-dir`, `--cleanup-images`,
`--shell-mode`, and `--oracle` as needed.
