# Evals

Use eval commands to measure quality / accuracy rather than speed.

## Publish a suite

Prefer the guarded one-command workflow:

```bash
lmx eval publish questions.jsonl \
  --description "Original networking questions written to test protocol reasoning." \
  --source-url https://github.com/org/repo
```

It accepts CSV, JSONL, JSON arrays, suite manifests, and terminal task
directories. It detects safe column mappings, validates and audits locally,
runs the authenticated server preflight before uploads, uploads inline data,
and submits as `PENDING`. Use `--dry-run` before ordinary suite writes and
`--strict` to block audit warnings. Missing or ambiguous mappings must be fixed
with `--input-column`, `--gold-column`, `--choices-column`, or
`--rubric-column`; never guess after the CLI rejects a mapping.

Use `lmx eval suite submissions` for status and admin feedback, then
`lmx eval suite resubmit <id> --file .localmaxxing/<slug>.eval-suite.json` to
correct a rejected submission.

Lower-level discovery and authoring commands remain available:

```bash
lmx eval suite list --out suites.json
lmx eval suite search reasoning --out reasoning-suites.json
lmx eval suite show hellaswag --out hellaswag-suite.json
lmx eval suite init --slug my-eval --name "My Eval" --category reasoning --out my-eval.json
lmx eval suite import questions.jsonl --slug my-eval --name "My Eval" --kind qa --out my-eval.json
lmx eval suite validate my-eval.json
lmx eval suite audit my-eval.json
lmx eval suite check my-eval.json --model Qwen/Qwen3-8B --base-url http://localhost:8000 --samples 5
lmx eval suite submit my-eval.json --upload-datasets
```

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

Publish raw Harbor/Terminal-Bench tasks with the same guarded command:

```bash
lmx eval publish ./terminal-bench-tasks \
  --description "Repository-level repair tasks with deterministic verifiers." \
  --source-url https://github.com/org/repo
```

Terminal publication requires Pro, Docker, at least two tasks, and a public
provenance URL. The CLI checks access and quota before oracle work, imports,
oracle-verifies, packages, uploads, preflights, and submits as `PENDING`.
`.localmaxxing/` keeps canonical bundles and resumable upload state. Terminal
`--dry-run` still uploads and verifies archives but does not create the dataset.
Use `--skip-oracle` only for a collection already verified locally.

Harbor Dockerfile and multi-service Docker Compose environments are both
supported. Compose dependencies and GPU requests are retained. For isolated
verifiers, the agent environment is stopped and only task-declared artifacts
(after exclusion rules) are materialized into the fresh verifier environment.

Lower-level import/publish and execution commands remain available:

```bash
lmx eval terminal import ./terminal-bench-tasks --out ./tb-bundles --version=2.1
lmx eval terminal publish ./tb-bundles --slug my-terminal-bench --name "My Terminal Benchmark" --source-url https://github.com/org/repo --shard-count 5
lmx eval terminal verify ./tb-bundles/smoke --oracle
lmx eval terminal run terminal-bench-2-1 --base-url http://localhost:8000 --model Qwen/Qwen3-8B --hardware hardware.json --run-dir ./runs/tb21-qwen --resume auto --json-status --json --submit
lmx eval terminal submit ./completed-terminal-run --dataset terminal-bench-2-1 --hf-id Qwen/Qwen3-8B --hardware hardware.json --quantization Q4_K_M --quant-format gguf --dry-run --out terminal-submit-payload.json
```

For an unattended run, use this sequence:

```bash
# 1. Validate the final selection and identity without Docker, model calls, or submission.
lmx eval terminal run terminal-bench-2-1 --base-url http://localhost:8000 --model Qwen/Qwen3-8B --hardware hardware.json --submit --dry-run --out terminal-preflight.json

# 2. Launch a durable worker. --detach requires --run-dir and rejects --dry-run.
lmx eval terminal run terminal-bench-2-1 --base-url http://localhost:8000 --model Qwen/Qwen3-8B --hardware hardware.json --run-dir ./runs/tb21-qwen --resume auto --json-status --submit --detach

# 3. Poll one JSON snapshot and consume canonical JSONL events.
lmx eval terminal status ./runs/tb21-qwen --json
lmx eval terminal logs ./runs/tb21-qwen --follow

# 4. Stop cooperatively when necessary.
lmx eval terminal cancel ./runs/tb21-qwen --json
```

The detached launcher returns after writing `process.json` and starting the worker. `events.jsonl` is the canonical progress/error stream; raw diagnostics go to `worker.stderr`, and final worker stdout is isolated in `worker.stdout`. `status --json` emits exactly one JSON document with lifecycle state, verified PID/liveness, dataset/shard, task totals, scored progress, current tasks, checkpoint timestamps, `lastActivityAt` from the newest complete canonical event, paths, and any final error. `logs` replays only complete newline-terminated JSONL records; `--follow` waits for a persisted terminal state and EOF. Do not infer success from launcher exit, PID disappearance, log EOF, or a heartbeat gap.

Cancellation first writes `terminal_cancel_requested`, then signals the detached worker. The worker stops scheduling, cancels active model/command work, removes its task container, persists `terminal_eval_interrupted`, and leaves scored task files reusable. SIGINT or SIGTERM follows the same resumable path; a second signal may hard-stop cleanup. On the next exact `--resume auto`, stale containers bearing the run label are reconciled before work starts, scored tasks emit `terminal_task_resumed`, and interrupted or unscored tasks run again. Use `cancel --force` only when cooperative shutdown cannot finish.

Completion is authoritative only when `status.state` is `completed` or `submitted` and `result.json` decodes with every task scored. Publication additionally requires `submitted`, a durable `submission` receipt, and `terminal_eval_submitted`. A parsed reward below 1 is a scored benchmark failure; an unscored infrastructure error blocks shard publication. For an intentional same-shard rerun, keep the old directory intact and choose a new unique `--run-dir` with `--resume none`.

The approved Terminal-Bench 2.1 dataset partitions 89 tasks into 10 disjoint
shards. A full deferred checkpoint is validated against the exact canonical task
set and written/submitted as 10 ordered shard payloads; `--shard-index <n>` is
required for an already-isolated shard or any other dataset. Dry-run performs no
network calls. Saved shard-local and full-checkpoint token totals remain in
`runConfig`.

For long agent-driven runs, always pass a unique `--run-dir`; `--resume auto` is
then the default. Each task result and `run.json` are atomically persisted before
`terminal_task_done`. Restart the exact command to reuse `scored: true` tasks;
unscored tasks rerun, while changed manifests, model/endpoint identity, hardware,
or execution options fail with `checkpoint_mismatch`. A checkpoint already marked
`submitted` fails with `checkpoint_already_submitted`; use a new run directory
with `--resume none` for an intentional fresh rerun of the same shard.

With `--json-status`, parse stderr only as JSONL. Human prose is suppressed and
stdout stays empty unless `--json` requests one final JSON document. Treat
`terminal_eval_completed` as local completion and `terminal_eval_submitted` as
publication completion; use their `resultPath`, `checkpointResultPath`, or
`runDir` rather than scraping prose. A long in-flight built-in model request emits
`terminal_model_call_heartbeat`; a long external command emits
`terminal_external_agent_heartbeat`. Both report elapsed and remaining agent
time. An agent deadline can still proceed to the verifier and produce a scored
result.

The built-in agent loop and bundled Terminus-2 adapter enforce the resolved
`--max-turns`/manifest/fallback cap. An arbitrary `--agent-cmd` only receives
that requested cap in `LMX_TERMINAL_MAX_TURNS`; its receipt records
`maxTurnsEnforcement: "not-enforced"`, and an unreported turn count is `null`.

Useful Terminal-Bench flags include `--task-dir`, `--run-dir`, `--resume`,
`--dataset`, `--hf-id`, `--shard-index`, `--max-turns`, `--agent-timeout`,
`--agent`, `--agent-cmd`, `--agent-execution`, `--agent-name`,
`--container-base-url`, `--command-timeout`, `--endpoint-timeout-seconds`,
`--trace-dir`, `--cleanup-images`, `--shell-mode`, `--native-tools`,
`--json-status`, `--json`, and `--oracle`. `--native-tools` sends the built-in
shell function to compatible chat endpoints. `--agent terminus-2` uses the
release-binary-embedded Harbor adapter.


## Eval-derived training

```bash
lmx eval train prepare ./completed-terminal-run --out ./training-data --base-model Qwen/Qwen3-Coder-30B-A3B-Instruct --allow-benchmark-training
lmx eval train run ./training-data/manifest.json --trainer-cmd "python3 python/localmaxxing_helpers/train_eval_sft.py --backend unsloth --dataset {dataset} --model {base_model} --output {output} --lora-dropout 0"
```

Preparation exports scored passing OMP trajectories to conversational
`sft.jsonl` and failed-task verifier diagnostics to `failures.jsonl`. It omits
streaming deltas and hidden thinking and never fabricates preference pairs.
`eval train run` is non-executing unless `--execute` is supplied. The required
benchmark acknowledgement records that the trained model must be measured on a
separate holdout, not the same eval tasks.
The bundled trainer supports `--backend unsloth` and `--backend trl`. Unsloth
uses the checkpoint chat template and explicit assistant-token labels, including
tool-call conversations. Current MoE guidance favors a supported Unsloth
checkpoint rather than generic bitsandbytes 4-bit loading; `--load-in-16bit`
requires enough aggregate VRAM.

### Online GRPO

```bash
lmx eval train rl prepare ./tb-bundles --out ./rl-training --base-model Qwen/Qwen3-Coder-30B-A3B-Instruct --environment-factory my_package.environments:make_environment --allow-benchmark-training
lmx eval train rl run ./rl-training/manifest.json --resume auto
lmx eval train rl run ./rl-training/manifest.json --output-dir ./checkpoints/grpo --resume none --python-bin python3 --execute
```

RL preparation accepts imported terminal bundles, not completed results, and
writes prompt-only `prompts.jsonl` plus a typed `manifest.json`; it includes no
historical completion, pass/fail, or reward labels. Optional preparation flags
are `--environment-config <json-object-file>` and
`--grpo-config <json-object-file>`. The trusted `module:callable` plugin is
required—training cannot run until it is importable. Its factory accepts the
`bundle_root` and `config` keyword arguments, returns a TRL 1.8 environment, exposes
typed/documented tool methods, resets isolated task state, and computes each
fresh policy rollout's reward with `get_reward()`. It must sandbox tools and
keep verifier data and reference solutions outside the policy's reach.

`rl run` validates and prints a direct-argv plan by default; only `--execute`
starts training, and `--trainer-cmd` is unsupported. Output defaults to the
manifest's `grpo-output`; `--output-dir` overrides it. `--resume auto` starts
fresh for empty output or chooses the highest valid `checkpoint-N`, `none`
requires empty output, and an explicit path must contain `trainer_state.json`.
Install a hardware-specific PyTorch build first, then
`python -m pip install 'trl==1.8.0' 'transformers>=5.2.0,<6'`. Training on these
prompts contaminates same-task scores; use a separate unseen holdout.

## Eval storage

```bash
lmx eval storage upload traces.jsonl --kind artifact --format jsonl --out artifact-bundle.json
lmx eval storage download <storageKey> --out traces.jsonl
```

Storage supports eval artifacts/datasets for deferred or offline workflows.
