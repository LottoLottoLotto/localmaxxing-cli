# Build And Run An Eval On LocalMaxxing

This guide shows the practical flow for using the LocalMaxxing CLI to discover evals, run them locally, validate the result, and submit it to localmaxxing.com.

## 1. Install The CLI

Download a pre-built archive from the [latest release](https://github.com/LottoLottoLotto/localmaxxing-cli/releases/latest) (see the README for checksum verification):

```bash
curl -fsSLO https://github.com/LottoLottoLotto/localmaxxing-cli/releases/latest/download/lmx-linux-amd64.tar.gz
tar -xzf lmx-linux-amd64.tar.gz
sudo mv lmx /usr/local/bin/
```

Or build from source (requires Go 1.22+):

```bash
git clone https://github.com/LottoLottoLotto/localmaxxing-cli.git
cd localmaxxing-cli
go build -o lmx ./cmd/lmx
```

Verify it works:

```bash
lmx --help
```

## 2. Authenticate

Create an API key in LocalMaxxing, then either save it:

```bash
lmx auth --key bhk_...
```

Or use an environment variable.

macOS/Linux:

```bash
export LMX_API_KEY=bhk_...
```

Windows PowerShell:

```powershell
$env:LMX_API_KEY = "bhk_..."
```

Check auth status:

```bash
lmx auth
```

## 3. Generate Hardware Metadata

Submissions need a hardware object. Generate a best-effort file:

```bash
lmx hardware --out hardware.json
```

Open `hardware.json` and correct anything the machine detector cannot know, such as exact GPU name, VRAM, power limits, or multi-GPU layout.

## 4. Discover Evals And Models

Fetch the current LocalMaxxing agent context:

```bash
lmx context --out localmaxxing-agent-context.json
```

List approved eval suites:

```bash
lmx eval suite list --out localmaxxing-suites.json
```

Search suites:

```bash
lmx eval suite search reasoning --limit 10 --out reasoning-suites.json
```

Inspect one suite:

```bash
lmx eval suite show hellaswag --out hellaswag-suite.json
```

Search for the canonical model ID:

```bash
lmx model search qwen3-8b --out models.json
```

The suite file includes `suiteDoc`, task keys, scoring method, aggregation, and `agentInstructions`. Discovery output is redacted defensively so structured gold/reference fields are not written to agent-facing files.

## 5. Run An LM-Eval Harness Suite

For `LM_EVAL_HARNESS` suites such as `hellaswag`, `mmlu`, or `gsm8k`, use the wrapper:

```bash
lmx eval lm-eval hellaswag \
  --model Qwen/Qwen3-8B \
  --backend hf \
  --hardware hardware.json \
  --dry-run
```

For `--backend hf`, the CLI defaults `--model-args` to `pretrained=<model>`.

For another lm-eval backend, pass explicit model args:

```bash
lmx eval lm-eval hellaswag \
  --model Qwen/Qwen3-8B \
  --backend vllm \
  --model-args pretrained=Qwen/Qwen3-8B,tensor_parallel_size=1 \
  --hardware hardware.json \
  --dry-run
```

The wrapper:

- Fetches the suite from LocalMaxxing.
- Verifies it is an `LM_EVAL_HARNESS` suite.
- Builds and runs `lm_eval`.
- Parses the output JSON.
- Writes a LocalMaxxing run payload.
- Calls the LocalMaxxing dry-run or submit endpoint.

If `lm_eval` is not installed:

```bash
pip install lm-eval
```

If your executable has a custom name or path:

```bash
lmx eval lm-eval hellaswag \
  --model Qwen/Qwen3-8B \
  --lm-eval-bin /path/to/lm_eval \
  --hardware hardware.json \
  --dry-run
```

## 6. Upload Existing LM-Eval Results

If you already ran lm-eval yourself, upload the result JSON:

```bash
lmx eval run hellaswag \
  --model Qwen/Qwen3-8B \
  --results localmaxxing-lm-eval-results.json \
  --hardware hardware.json \
  --dry-run
```

When the dry-run passes, submit it:

```bash
lmx eval run hellaswag \
  --model Qwen/Qwen3-8B \
  --results localmaxxing-lm-eval-results.json \
  --hardware hardware.json \
  --submit
```

## 7. Run A Custom LocalMaxxing Suite

For `CUSTOM` suites, start a local OpenAI-compatible server first. Then run:

```bash
lmx eval run local-reasoning-mini \
  --model Qwen/Qwen3-8B \
  --base-url http://localhost:8000 \
  --hardware hardware.json \
  --dry-run
```

Use `--submit` after the dry-run passes.

For public OpenAI-compatible endpoints that LocalMaxxing can reach directly, use server-side execution:

```bash
lmx eval execute local-reasoning-mini \
  --model Qwen/Qwen3-8B \
  --base-url https://your-public-endpoint.example \
  --hardware hardware.json \
  --submit
```

Use `eval run` for localhost/private endpoints. Use `eval execute` when LocalMaxxing should call a public endpoint and score server-side.

Multiple-choice suites can also be scored by forced-continuation log-probability (lm-eval style) with `scoringMethod: loglikelihood`. This ranks each choice by the model's logprob of emitting it, instead of parsing generated text, and needs a `/v1/completions` endpoint exposing `echo`+`logprobs` (vLLM, SGLang, llama.cpp server).

For numeric word-problem suites (GSM8K-style **one-offs** you author to avoid leaked public benchmarks), scaffold with `lmx eval suite init --kind math`. The model reasons step by step and the runner extracts the final answer from the chain-of-thought (`answerExtraction: last_number`) before scoring `exact_match`, with numeric-aware matching (`72` == `72.0`, `1,000` == `1000`). No answer-only prompting required.

### Offline pull, run, and deferred submit

Pull a suite (including datasets and gold labels) once, then run offline and submit later — useful for inspecting data, air-gapped runs, or so a site outage never costs you a completed run:

```bash
lmx eval pull local-reasoning-mini --api-key bhk_... --out localmaxxing-eval-local-reasoning-mini
lmx eval run local-reasoning-mini \
  --suite-file localmaxxing-eval-local-reasoning-mini/suite.json \
  --model Qwen/Qwen3-8B --base-url http://localhost:8000 \
  --hardware hardware.json --out run.json
lmx eval submit run.json --model Qwen/Qwen3-8B --hardware hardware.json --api-key bhk_...
```

`eval run --suite-file` needs only the local model endpoint — no LocalMaxxing connection. The pulled files contain gold labels; do not publish them.

### Run A Blob-Backed Eval Shard (GSM8K, MMLU, …)

Official datasets like `gsm8k`, `mmlu`, and `hellaswag` are stored as
deterministic JSONL shards in object storage and scored by pooled pass/fail with
a Wilson confidence interval (no suite registration needed). The `eval shard` command
fetches a shard, runs the first N questions against your local endpoint, scores
them locally (gold answers travel inside the shard), and submits the
`question_id`/`pass` pairs plus a per-question trace for every scored question.
The server pools the pass/fail counts and stores the traces as one whole-shard
JSONL bundle (downloadable from the run), keeping only a small preview sample in
its database. Use `--artifact-limit <n>` to submit only a balanced pass/fail
sample instead of the full set.

Scoring mode is dataset-aware: HellaSwag uses canonical continuation
loglikelihood. With a logprob-capable OpenAI endpoint, the CLI uses
`--scoring loglikelihood` (`/v1/completions` with echoed prompt token logprobs).
With llama.cpp GGUF files, pass `--model-path model.gguf`; the CLI switches to
`llama_cpp_loglikelihood` and calls the bundled local scorer
(`lmx-llama-score-hellaswag`) so no server echo-logprobs are required.
GSM8K-style short-answer datasets use the server-published canonical evaluation
config when available: `exact_match`, the official prompt template, and
`answerExtraction: final_answer`. Use `--scoring exact_match` only for debugging
chat-letter prompts on multiple-choice shards.

Release archives bundle this scorer next to `lmx`. To build it from a source
checkout (CPU-only, statically linked against a pinned llama.cpp):

```bash
scripts/build-scorer.sh                                # fetches + builds llama.cpp
LLAMA_SRC=/path/to/llama.cpp scripts/build-scorer.sh   # reuse a local checkout
```

The CLI auto-discovers `lmx-llama-score-hellaswag` next to `lmx` (or in `dist/`);
override the path with `--llama-scorer <path>`.

HumanEval, MBPP, and CRUXEval are **execution-based** code evals. HumanEval/MBPP
use `scoring: code_execution`; CRUXEval uses `scoring: cruxeval_execution`, where
input-prediction passes if `f(generated_input) == observed_output` and output-
prediction passes if `generated_output == f(function_input)`. The CRUXEval prompt
uses a `CRUX_ANSWER:` safeword plus `<END_CRUX>` stop sequence so models can reason
before the final extractable answer. The CLI runs checks inside a hardened sandbox
and records pass/fail.
Build the sandbox image once:

```bash
docker build -t lmx-sandbox sandbox
```

The default container command is:

```bash
docker run --rm -i --network none --memory 2g --cpus 2 --pids-limit 128 \
  --cap-drop ALL --security-opt no-new-privileges --read-only \
  --tmpfs /tmp:exec,size=128m lmx-sandbox
```

That uses a read-only root filesystem, dropped capabilities,
`no-new-privileges`, no network, and CPU/memory/PID/time limits; the harness adds
per-task CPU/RAM/wall limits as defense in depth.

If your user cannot access `/var/run/docker.sock`, either add the user to the
Docker group and re-login, or run the default runtime through sudo:

```bash
lmx eval shard humaneval-plus --sandbox-use-sudo ...
```

`--sandbox-runtime "sudo docker"` is equivalent for hosts where you prefer to
spell out the runtime. If the container starts but fails with
`exec /usr/local/bin/python3: operation not permitted`, retry without the strict
cap/no-new-privileges/read-only profile:

```bash
lmx eval shard humaneval-plus --sandbox-use-sudo --sandbox-relaxed-security ...
```

This keeps `--network none`, memory/CPU/PID caps, and the executable `/tmp`
tmpfs, but omits `--cap-drop ALL`, `--security-opt no-new-privileges`, and
`--read-only` for Docker/rootless/security-profile combinations that reject that
hardening profile.

Other knobs: `--sandbox-runtime podman`, `--sandbox-image <name>`,
`--sandbox-memory`, `--sandbox-cpus`, or replace the launcher entirely with
`--sandbox-cmd` (e.g. `--sandbox-cmd "python3 sandbox/run_sandbox.py"` for hosts
without Docker — note that bypasses container isolation and should only be used
on disposable boxes).

Scoring follows canonical pass@1: HumanEval programs keep the prompt stub's
imports (so a dropped `import` is not a false fail), and a generation that errors
(after retries) or yields no runnable code is counted as a fail rather than
dropped, so the denominator stays at the number of questions run. If most
generations fail, the run aborts instead of submitting an all-fail result.

EvalPlus variants `humaneval-plus` and `mbpp-plus` use the extended EvalPlus test
suites (the sandbox image bundles numpy for them). MBPP-family datasets default to
canonical 3-shot prompting (`--few-shot N` to change). For pass@k, pass
`--n-samples N` (sampling temperature defaults to 0.8 when N>1) and `--k K`; the
CLI reports `passAtK`/`passAt1` and submits the greedy/first-sample pass@1 per
question (matching the modern EvalPlus greedy-pass@1 leaderboard).

### Run Terminal-Bench (our runner)

Terminal-Bench bundles are task directories containing `task.json`, `environment/`,
`tests/`, and optionally `solution/`. The CLI imports harbor-format tasks into
that bundle format, then runs each task in Docker. By default it uses the
localmaxxing fenced-bash agent harness; with `--agent-cmd` it can hand the running
container to your preferred host-side agent harness.

Import an operator-supplied harbor task tree:

```bash
lmx eval terminal import ./terminal-bench-2.1-tasks \
  --out ./tb-bundles \
  --version 2.1
```

Verify a bundle with its oracle solution before scoring a model:

```bash
lmx eval terminal verify ./tb-bundles/adaptive-rejection-sampler --oracle --out oracle.json
```

Preflight a local bundle selection before execution:

```bash
lmx eval terminal run --task-dir ./tb-bundles \
  --base-url http://localhost:8000 \
  --model Qwen/Qwen3-8B \
  --out terminal-preflight.json \
  --dry-run
```

`--dry-run` is an explicit preflight: it validates the selected local bundles
or fetches and resolves an approved dataset manifest. It writes the resolved
plan to `--out` when provided, but does not start Docker, call the model
endpoint, run an external agent, or invoke a verifier.

Run the same bundles locally without publishing:

```bash
lmx eval terminal run --task-dir ./tb-bundles \
  --base-url http://localhost:8000 \
  --model Qwen/Qwen3-8B \
  --out terminal-run.json \
  --trace-dir terminal-traces
```

With neither `--dry-run` nor `--submit`, the CLI executes the tasks, agent, and
verifier locally, saves the results, and prints `Local run complete — nothing
submitted.` No LocalMaxxing API key or hardware file is required for that local
execution.

By default the built-in harness runs every turn in one persistent shell session
(`docker exec -i <container> bash -l`), so `cd`, `export`, and `source` carry
across turns (`protocol: "react-shell"`). Pass `--shell-mode stateless` to run
each command in a fresh `docker exec` with no shared state (`protocol:
"react-bash"`).

Persistent shell mode changes command state, not task budgets. The resolved turn
cap uses the first positive value in this order: explicit `--max-turns`, the
task manifest's `agent.maxTurns`, then 50 turns. The built-in agent loop and the
bundled Terminus-2 adapter enforce that cap. An arbitrary `--agent-cmd` only
receives it through `LMX_TERMINAL_MAX_TURNS`; the CLI cannot verify that the
wrapper enforces it. The whole-agent deadline uses explicit `--agent-timeout`,
manifest `agent.timeoutSec`, then 900 seconds, and is enforced for every agent
mode. Persistent mode does not inflate either limit. Built-in model requests default
to a 10-minute per-request timeout, and the first attempt reserves part of the
remaining agent budget for one retry. Terminal model completions are capped at
16,384 tokens by default and 8,192 tokens on retry; pass `--max-tokens` to set
an explicit cap for both attempts.

While a model request is in flight, JSON status emits a
`terminal_model_call_heartbeat` every minute with the task ID, turn, initial or
retry attempt, elapsed seconds, and remaining whole-agent time. This makes slow
local inference distinguishable from a stalled CLI without changing the model
request or agent deadlines.

While an external command is in flight, JSON status similarly emits
`terminal_external_agent_heartbeat` every minute with the task ID, execution
mode, elapsed seconds, and remaining whole-agent time. Detached `status --json`
reports the newest complete canonical event as `lastActivityAt`; checkpoint
`updatedAt` remains the last task-state transition.

Each shell command has a 300-second timeout by default, capped by the remaining
whole-agent time; override it with `--command-timeout`. A timed-out command is
visible as exit 124 in the transcript and as a `terminal_turn` status with
`timedOut: true`; the next model observation includes a recovery hint to inspect
partial state or try a smaller, bounded, non-interactive command. In persistent
mode the timed-out shell is restarted and its shell state is reset, while the
agent can continue within its remaining turn and time budgets.

Use `--trace-dir <dir>` when you want a local, browsable copy of what happened
inside each task. The CLI writes one subdirectory per `question_id` containing:

- `transcript.md` — model replies, extracted shell commands, stdout/stderr, exit
  codes, restarts, and timeouts.
- `verifier.txt` — verifier stdout/stderr and reward parsing output.
- `prompt.txt` and `instruction.txt` — the exact harness prompt/task text.
- `result.json` — pass/scored/error/turn metadata, explicit `wallTimeMs`, and
  `tokenUsage` (`inputTokens`, `outputTokens`, cache tokens, total tokens, model
  call count).
- `agent/` — external-agent logs written via `LMX_TERMINAL_TRACE_DIR`, plus
  `usage.json` when the agent trace exposes model usage, when `--agent-cmd` is
  used.

Scoring follows harbor canonical verifier semantics: after the agent finishes,
`tests/` is copied to `/tests` in the task container and the verifier command
(default `bash /tests/test.sh`) runs in a non-login shell. The reward file is
the sole pass signal — `/logs/verifier/reward.json` (`{"reward": <num>}`) takes
precedence over `/logs/verifier/reward.txt` (bare float), and a task passes
when `reward >= 1.0`. The verifier's exit code is ignored once a reward was
written. A verifier timeout or missing/unparseable reward records a failed
verifier result locally, but cannot satisfy complete-shard publication. If the
built-in agent reaches its whole-agent deadline, the CLI still runs the verifier
against whatever container state the agent left behind, so partial work
completed before the deadline can score. Oracle runs (`verify --oracle`) mirror
harbor's OracleAgent — `solution/` is copied to `/solution` and `solve.sh`
runs as root with `DEBIAN_FRONTEND=noninteractive` plus the task's
`[solution].env`, bounded by the agent timeout, and verification proceeds even
when `solve.sh` exits non-zero. Task env values support harbor `${VAR}` /
`${VAR:-default}` templates resolved from the host environment, and
`[agent].user` / `[verifier].user` are honored on every `docker exec`.
Harbor tasks whose environment is a `docker-compose.yaml` are skipped at
import with a `terminal_import_skipped` event; this runner only supports
single-container Dockerfile or prebuilt-image environments (all of Terminal-
Bench 2.1 qualifies).

Run and submit an approved public dataset:

```bash
lmx eval terminal run terminal-bench-2-1 \
  --base-url http://localhost:8000 \
  --model Qwen/Qwen3-8B \
  --hardware hardware.json \
  --run-dir ./runs/tb21-qwen \
  --resume auto \
  --out terminal-run.json \
  --json-status \
  --json \
  --submit
```

The approved `terminal-bench-2-1` dataset deterministically partitions its 89
tasks into 10 disjoint shards. For dataset preflight and publication flows,
omitting `--shard` queries aggregate coverage for the resolved model,
quantization, quantization format, and harness key, then selects the first
missing shard. Pass `--shard <index>` to override that assignment explicitly.
A preflight reports the selected shard but does not reserve it.

### Durable runs and machine output

For long-running or agent-driven execution, pass a unique `--run-dir`. The CLI
creates the directory and maintains:

- `run.json` — checkpoint version, state, immutable run fingerprint, ordered
  task IDs, and task IDs whose latest attempt was durably persisted.
- `results/<task>-<hash>.json` — one atomic result per attempted task, including
  the full response, verifier output, usage, timing, and explicit `scored` value.
- `traces/<task>/` — the normal browsable trace tree when `--trace-dir` was not
  supplied separately.
- `result.json` — the assembled final result and, after publication, submission
  receipt.
- `process.json` — detached-worker PID, lifecycle, timestamps, event/output paths,
  exit status, and final error.
- `events.jsonl` — canonical append-only JSONL progress and error records used by
  `eval terminal logs`; raw diagnostics go to `worker.stderr`, while final worker
  stdout remains isolated in `worker.stdout`.

Preflight with the final dataset, endpoint/model, hardware, and selection first.
Then launch the same run with a unique directory:

```bash
lmx eval terminal run terminal-bench-2-1 \
  --base-url http://localhost:8000 \
  --model Qwen/Qwen3-8B \
  --hardware hardware.json \
  --run-dir ./runs/tb21-qwen \
  --resume auto \
  --json-status \
  --submit \
  --detach

lmx eval terminal status ./runs/tb21-qwen --json
lmx eval terminal logs ./runs/tb21-qwen --follow
# When a cooperative stop is required:
lmx eval terminal cancel ./runs/tb21-qwen --json
```

`--detach` requires `--run-dir` and cannot be combined with `--dry-run`. The
launcher returns only after writing the process record and starting the worker;
that acknowledgement is not run success. `status --json` emits one snapshot with
worker liveness, scored/pending/current task counts, timestamps, durable paths,
and any final error. `logs` emits only complete stored JSONL records; `--follow`
waits for a persisted terminal state and EOF. Do not use PID disappearance, log
EOF, or a heartbeat gap as a completion test.

`cancel` is idempotent and cooperative: it records the request, signals the
worker, stops scheduling new tasks, cancels active work, and lets bounded Docker
cleanup finish. SIGINT and SIGTERM take the same resumable path. A second signal
or `cancel --force` may hard-stop cleanup; on the next exact `--resume auto`, the
CLI reconciles containers labeled for that run before starting pending work.
Previously scored task files remain reusable; the interrupted active task runs
again.

`--resume auto` is the default when `--run-dir` is present. A restarted command
validates the dataset, shard, ordered task manifests, endpoint/model identity,
quantization, hardware, harness, and execution options against the saved
fingerprint. It reuses only tasks with `scored: true`; interrupted or unscored
tasks run again. Each scored task emits `terminal_task_resumed` instead of
`terminal_task_started`. A mismatch fails with `checkpoint_mismatch` rather
than mixing results from different runs. A checkpoint already marked `submitted`
fails with `checkpoint_already_submitted` instead of posting a duplicate receipt.
`--resume none` requires a directory without an existing `run.json`; use a new
directory to rerun and submit the same shard again intentionally.

Per-task result files and `run.json` are written through a synced temporary file
and atomic rename before `terminal_task_done` is emitted. An abrupt kill can lose
the active task, but not a previously completed task. A graceful signal records
`terminal_eval_interrupted` after active workers unwind. A run with all scored
tasks can be restarted to assemble its final result and retry publication without
starting Docker tasks again.

`--json-status` is strict machine mode for terminal runs. Stderr contains one
JSON object per line for progress, completion, and errors; human status blocks
and prose are suppressed. Stdout is empty unless `--json`, `--print`, or
`--verbose` requests one final JSON result document. Successful local execution
ends with `terminal_eval_completed`; successful publication ends with
`terminal_eval_submitted`. Both carry final summary or receipt fields and durable
result paths. Keep stdout and stderr separate: parse stderr incrementally as
JSONL and parse stdout once after process exit.

Publication rejects an incomplete shard unless every fetched task has a scored
result. A parsed failing reward is still a scored result; an infrastructure
error or otherwise unscored task prevents the entire shard from being
submitted. After a successful `eval terminal run --submit`, the local `summary`
and `results` are preserved and a durable `submission` receipt is merged into
the same JSON document named by `--out`. The receipt contains the shard index,
submitted count, run, and aggregate returned by LocalMaxxing.

The submitted `runConfig` records resolved limits, their sources, and whether
the turn cap was actually enforced: `maxTurnsPolicy`, `maxTurnsEnforcement`,
`agentTimeoutPolicy`, `commandTimeoutSec`, and one `taskLimits` entry per task.
`maxTurnsEnforcement` is `cli-agent-loop` for the built-in agent,
`bundled-adapter` for Terminus-2, `not-enforced` for arbitrary external
commands, and `not-applicable` for oracle runs. Each task entry repeats that
value alongside the resolved/requested turn cap, deadline, command timeout,
and CLI/task-manifest/fallback source. When every task has the same resolved
turn cap, `maxTurns` carries that nonzero value. External agents that do not
report a turn count serialize `turns: null`, never a misleading zero.

Repeat the command to publish each missing shard. A complete 89-task checkpoint
produced by Harbor or another offline runner can instead be validated and
partitioned into the same 10 shard submissions without rerunning tasks or
contacting a model endpoint:

```bash
lmx eval terminal submit ./completed-terminal-run \
  --dataset terminal-bench-2-1 \
  --hf-id Qwen/Qwen3-8B \
  --hardware hardware.json \
  --quantization Q4_K_M \
  --quant-format gguf \
  --dry-run \
  --out terminal-submit-batch.json
```

The dry-run output is `{ "dataset": "terminal-bench-2-1", "shards": [...] }`
with 10 ordered payloads carrying explicit `shardIndex` values. The CLI verifies
the exact canonical 89-task ID set before partitioning. Inspect the batch, then
remove `--dry-run` and provide `--api-key` (or `LMX_API_KEY`) to submit all 10
payloads sequentially. For a checkpoint that already contains one isolated
shard, pass `--shard-index <n>`; other dataset slugs always require an explicit
shard index. Deferred submission preserves shard-local and full-checkpoint token
usage in `runConfig` and never starts Docker, reruns a verifier, or calls the
model endpoint.

The built-in `--agent terminus-2` adapter is embedded in release binaries. It
requires Harbor's Python environment at runtime but does not require a source
checkout or a loose adapter script.

Use a preferred external agent harness:

```bash
lmx eval terminal run --task-dir ./tb-bundles \
  --agent-cmd './my-agent-wrapper.sh' \
  --agent-name my-agent \
  --agent-execution routed-shell \
  --model Qwen/Qwen3-8B \
  --out terminal-run.json
```

This no-submit command actually launches the external agent and verifier
locally. Add `--dry-run` only when you want preflight validation without
launching the wrapper, Docker, the model, or the verifier.

`--agent-cmd` runs on the host after the task container starts and before the
verifier runs. The command receives:

- `LMX_TERMINAL_CONTAINER` — Docker container name.
- `LMX_TERMINAL_TASK_ID` — task id.
- `LMX_TERMINAL_BUNDLE_DIR` / `LMX_TERMINAL_TASK_DIR` — imported bundle path.
- `LMX_TERMINAL_TASK_JSON` — manifest path.
- `LMX_TERMINAL_INSTRUCTION_FILE` — temporary file containing the task prompt.
- `LMX_TERMINAL_BASE_URL` and `LMX_TERMINAL_MODEL` — values passed to the CLI,
  when your external agent wants to call the model itself.
- `LMX_TERMINAL_MODEL_API_KEY` — optional model API key passed to an external
  agent that owns model requests.
- `LMX_TERMINAL_MAX_TURNS` — resolved requested turn cap. The bundled Terminus-2
  adapter enforces it; arbitrary wrappers must explicitly consume it, and their
  receipts remain `maxTurnsEnforcement: "not-enforced"`.
- `LMX_TERMINAL_AGENT_TIMEOUT_SEC` — enforced whole-agent deadline in seconds.
- `LMX_TERMINAL_CONTAINER_BASE_URL` — model/API URL as seen from Docker
  containers (default: `http://172.17.0.1:8080`, override with
  `--container-base-url`).
- `LMX_TERMINAL_WORKDIR` — canonical task workdir (`/app`).
- `LMX_TERMINAL_TRACE_DIR` — host directory for agent traces/logs; text files
  written here are appended to the submitted artifact.
- `LMX_TERMINAL_EXECUTION_MODE` — `host`, `container`, or `routed-shell`.
- `LMX_TERMINAL_AGENT_USER` — the task's `[agent].user` (empty for the image
  default); routed-shell helpers already apply it.
- `LMX_TERMINAL_SHELL_COMMAND` — helper script for routed-shell wrappers. Call
  `"$LMX_TERMINAL_SHELL_COMMAND" 'ls /app'`; it executes in the task container,
  not on the host.

If the wrapper is still running when the agent timeout elapses, it is killed
and the task proceeds to verification anyway (harbor canonical: agent timeouts
are scored, not errored). A wrapper that exits non-zero before the timeout is
treated as an infrastructure failure and left unscored.

Choose an execution mode explicitly for comparability:

- `--agent-execution container`: the wrapper copies/runs the agent inside the
  task container. The model/agent sees `/app` directly, closest to Harbor's
  environment-routed model.
- `--agent-execution routed-shell`: the agent stays on the host, but its shell
  tool must call `LMX_TERMINAL_SHELL_COMMAND` so commands execute in `/app`
  inside the task container. Use this for agents with complex host installs or
  host-only auth.
- `--agent-execution host`: legacy mode. The agent has host shell access and is
  responsible for routing into Docker. This is marked separately in
  `runConfig.protocol` and should not be mixed with container-shell scores.

Example wrappers are in `examples/agents/`. External-agent submissions are
labeled with `runConfig.protocol="external-command/<mode>"`,
`runConfig.agent=<--agent-name>`, `runConfig.agentExecution=<mode>`, and
`runConfig.toolRouting`.

For Oh My Pi / OMP container parity, use
`examples/agents/omp-container-shell.sh` with `--agent-execution container`.
The wrapper mirrors Harbor's installed-agent shape: it runs OMP inside `/app`,
passes the task as direct prompt text, emits JSON event logs, and copies live
SQLite model-cache state for local llama.cpp models. It uses
`--tools bash --no-extensions`: the routed container shell is the only required
model tool, and restricting the schema avoids llama.cpp rejecting unrelated OMP
tools with open boolean JSON subschemas. A terminal model/provider error before
the first completed tool execution makes the wrapper exit nonzero and leaves
the task unscored. A provider error after useful tool execution, or an outer
agent timeout, preserves Harbor semantics by proceeding to verification.

### Prepare eval-derived adapter training

`eval train prepare` converts a completed terminal run into two different
datasets:

- `sft.jsonl` contains only scored, passing OMP trajectories reconstructed from
  finalized `message_end` events. Cumulative streaming updates and hidden
  thinking are excluded.
- `failures.jsonl` contains instructions, verifier output, and error metadata
  for curriculum analysis. Failed trajectories are not correct SFT labels.

```bash
lmx eval train prepare ./completed-terminal-run \
  --out ./training-data \
  --base-model Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --allow-benchmark-training
```

The command also writes `manifest.json` with source provenance, counts,
truncation information, contamination status, and adapter output location. A
single pass/fail attempt is outcome data, not a DPO preference pair, so the
preparer does not fabricate preference examples.

Use the bundled local QLoRA trainer through the explicit command boundary:

```bash
lmx eval train run ./training-data/manifest.json \
  --trainer-cmd "python3 python/localmaxxing_helpers/train_eval_sft.py --backend unsloth --dataset {dataset} --model {base_model} --output {output} --lora-dropout 0"
```

This prints the fully expanded command without running it. Install a current
Unsloth build for the host's CUDA and PyTorch versions, then add `--execute`.
The Unsloth backend loads dense checkpoints in 4-bit by default, applies its
gradient-checkpointing path, renders the checkpoint's own chat template, and
constructs explicit labels so user, system, and tool-result tokens are excluded
from assistant-only loss. Use `--full-sequence-loss` only intentionally.

For an MoE checkpoint, current Unsloth guidance does not recommend bitsandbytes
4-bit loading. The trainer warns when the model ID or loaded config identifies
MoE; pass `--load-in-16bit` only if the checkpoint fits in aggregate VRAM, or
select a supported prequantized checkpoint according to current Unsloth model
documentation. `--target-modules` overrides the inferred dense or fused-MoE
LoRA module suffixes.

The plain alternative is `--backend trl`, after installing `trl[peft]`,
`bitsandbytes`, and `datasets`. The base model must be a loadable HuggingFace
checkpoint, not the evaluated GGUF file.

The acknowledgement flag is deliberately required: once benchmark tasks become
training examples, that model's score on the same tasks is contaminated. Use a
separate, previously unseen holdout to measure whether the adapter generalizes.

### Prepare online GRPO training

`eval train rl` is a separate, online workflow. Its input is one imported
terminal task bundle or a parent directory containing only imported bundles,
as written by `eval terminal import`. Do not point it at a completed run,
`summary.json`, an SFT export, or a checkpoint directory. Online GRPO generates
new tool-using completions from the current policy and scores those rollouts in
an environment; it does not learn from historical terminal transcripts,
pass/fail fields, reward labels, or fabricated preference pairs.

The complete preparation syntax is:

```bash
lmx eval train rl prepare <imported-bundle-or-parent> \
  --out <rl-dir> \
  --base-model <huggingface-id-or-path> \
  --environment-factory <module:callable> \
  [--environment-config <json-object-file>] \
  [--grpo-config <json-object-file>] \
  --allow-benchmark-training
```

For example:

```bash
lmx eval terminal import ./terminal-bench-tasks --out ./tb-bundles --version 2.1
lmx eval train rl prepare ./tb-bundles \
  --out ./rl-training \
  --base-model Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --environment-factory my_package.environments:make_environment \
  --environment-config ./environment.json \
  --grpo-config ./grpo.json \
  --allow-benchmark-training
```

Both optional files must contain exactly one JSON object; omitted environment
configuration is `{}`. `--grpo-config` can override only
`num_generations`, `max_steps`, `learning_rate`,
`per_device_train_batch_size`, `gradient_accumulation_steps`,
`max_completion_length`, `max_tool_calling_iterations`,
`gradient_checkpointing`, `logging_steps`,
`save_steps`, `save_total_limit`, and `seed`. Unknown keys are rejected. The
`--out` directory must be missing or empty and must not overlap the source tree.

Preparation writes:

- `prompts.jsonl`: sorted conversational prompt rows with exactly `prompt`,
  `task_id`, and `bundle_ref`. Each prompt is the imported task instruction.
  There are deliberately no completion, pass/fail, verifier-result, solution,
  or reward columns.
- `manifest.json`: a typed `online_grpo` manifest containing absolute source and
  dataset paths, task count, base model, environment contract/configuration,
  validated GRPO settings, default `<rl-dir>/grpo-output`, and the required
  contamination acknowledgement.

#### Required environment plugin

There is no bundled task environment and no static-reward fallback. A plugin is
required, and training is not runnable until `--environment-factory` names
trusted, importable Python code. The callable has this runner-facing signature:

```python
def make_environment(*, bundle_root: pathlib.Path, config: dict) -> object:
    ...
```

The runner binds those two keyword arguments, then gives TRL a zero-argument
factory. `bundle_root` is the canonical imported-bundle root and `config` is a
defensive copy of the JSON object. The returned TRL 1.8 environment must:

- implement `reset(**row) -> str | None`; row metadata includes `task_id` and
  `bundle_ref`, allowing the plugin to select and cleanly reset the task. A
  returned string is appended to the final user message before generation;
- expose each policy tool as a public method with typed parameters and return
  type plus a Google-style docstring, as required by TRL tool schema generation;
- implement synchronous or asynchronous `get_reward() -> float`, called after
  the completed rollout, and derive that reward from the resulting environment
  state and verifier rather than from a prepared label.

TRL keeps separate environment instances for active rollouts and resets pooled
instances before reuse. Its loop repeatedly generates with the current policy,
parses a tool call, executes the corresponding environment method, appends the
tool result, and generates again. Therefore the invariant is: **every training
reward belongs to a fresh policy rollout**. Historical pass/fail or reward data
must never be substituted for `get_reward()`.

The plugin is loaded as trusted local code and has the same authority as the
training process. It is responsible for real isolation: create a clean sandbox
and tool state per rollout, constrain tool execution to that sandbox, run the
verifier outside the policy's tool surface, and keep tests/verifier artifacts,
expected outputs, reward files, and any `solution/` content unavailable to the
policy. Do not expose `bundle_root` itself as a policy filesystem. A reset must
remove state left by the previous rollout. Audit the plugin and its environment
configuration before `--execute`.

#### Plan, execute, output, and resume

The complete run syntax is:

```bash
lmx eval train rl run <rl-manifest.json> \
  [--output-dir <dir>] \
  [--resume auto|none|<checkpoint-dir>] \
  [--python-bin <path>] \
  [--execute]
```

For example, inspect the default plan first, then execute it:

```bash
lmx eval train rl run ./rl-training/manifest.json --resume auto
lmx eval train rl run ./rl-training/manifest.json \
  --output-dir ./checkpoints/grpo \
  --resume auto \
  --python-bin python3 \
  --execute
```

Without `--execute`, the CLI validates the manifest and prints JSON containing
a direct-argv preview with an embedded-helper placeholder, plus the dataset,
model, output directory, and resume selector; no plugin is imported and no
trainer is started. Execution materializes the embedded helper and launches it
with direct argv. It does not use a shell, does not accept `--trainer-cmd`, and
offers no arbitrary trainer pass-through.

The manifest's `<rl-dir>/grpo-output` is the default output. `--output-dir`
overrides it, but may not be the imported bundle root or a directory beneath
that root. Resume selectors behave as follows:

- `auto` (default): start fresh when output is missing or empty. For nonempty
  output, resume the numerically highest `checkpoint-N` containing
  `trainer_state.json`; fail rather than overwrite if no valid checkpoint exists.
- `none`: start fresh and require output to be missing or empty.
- an explicit path: require an existing checkpoint directory containing
  `trainer_state.json` and resume exactly that checkpoint.

On execution, the helper writes `resolved-run.json` before training, calls
`trainer.train(resume_from_checkpoint=...)`, and saves the trained model to the
chosen output directory.

#### Training prerequisites

Install PyTorch first using the official selector for this host's GPU or other
accelerator, driver, operating system, and desired CUDA/ROCm/CPU build. PyTorch
is intentionally not pinned by the CLI because that choice is hardware-specific.
Then install the exact trainer requirements:

```bash
python -m pip install 'trl==1.8.0' 'transformers>=5.2.0,<6'
```

TRL environment tools also need `jmespath` in that Python environment. The
base model and chat template must support tool calling and preserve the
assistant prefix across tool turns. The runner rejects every TRL version except
`1.8.0` and Transformers versions outside `>=5.2.0,<6`; this environment API is
version-sensitive, so do not silently upgrade either dependency.

`--allow-benchmark-training` is deliberately required. Prompt-only preparation
still exposes benchmark tasks to the model during online training, so scores on
those tasks are contaminated even though historical outcomes are absent. Keep
a separate, previously unseen holdout and use only that holdout for reported
post-training quality.

Security and footprint warning: terminal tasks execute arbitrary task Docker
images, copy verifier assets into those containers, and give a model/agent control
over shell commands in the task container. Run them on a disposable host with
Docker installed. Honor task CPU/memory settings (`--concurrency` defaults to 1
because containers are heavy). `no-network` is enforced with Docker `--network none`; `allowlist`
currently degrades to normal Docker egress and emits a `terminal_network_degraded`
warning because per-host firewalling is outside v1.

Start a local OpenAI-compatible server, then dry-run first:

```bash
lmx eval shard gsm8k \
  --base-url http://localhost:8000 \
  --questions 200 \
  --concurrency 4 \
  --dry-run
```

The command auto-detects the served model id, prints accuracy, and (with `--out`)
writes per-question predictions for inspection. With no `--questions`, it defaults
to the dataset's recommended sample size for a 95% / ±5% confidence interval.

When the dry-run looks right, submit with a real model id, an API key, and
hardware metadata:

```bash
lmx eval shard gsm8k \
  --base-url http://localhost:8000 \
  --model Qwen/Qwen3-8B \
  --hardware hardware.json \
  --questions 200 \
  --submit
```

Before continuing a sharded run, inspect aggregate coverage for the same model and
quantization:

```bash
lmx eval shard status hellaswag \
  --model Qwen/Qwen3-8B \
  --quantization Q4_K_M \
  --quant-format gguf
```

`coveredShards` and `missingShards` come from APPROVED aggregate coverage for the
dataset/model/quantization/quantFormat/harness key. A normal `--submit` refuses a
covered shard unless you pass `--rerun` or `--force`. To avoid duplicates, use
`--missing-only --submit` for the next missing shard, or `--all-missing --submit`
to walk every currently missing shard in ascending order.

Scoring is automatic per row: rows with `choices` are scored as multiple choice
(letter match); otherwise the final answer is extracted (server default for
GSM8K: `answerExtraction: final_answer`) and compared numerically, so
chain-of-thought output scores correctly when it ends with the canonical final
answer line. Override only for experiments with `--prompt-template`,
`--answer-extraction none|final_answer|last_number|regex`, and
`--answer-regex`. Submitting repeatedly with different shards or question counts
grows unique-question coverage; the pooled score dedupes by `question_id`.

## 8. Upload Large Artifacts

Small artifacts can be included inline by eval runs. Large traces should be uploaded as an artifact bundle:

```bash
lmx eval artifacts upload traces.jsonl \
  --format jsonl \
  --item-count 1000 \
  --out artifact-bundle.json
```

For lower-level storage use:

```bash
lmx eval storage upload traces.jsonl \
  --kind artifact \
  --format jsonl \
  --out artifact-bundle.json
```

Download a storage object when you have a storage key:

```bash
lmx eval storage download <storageKey> --out traces.jsonl
```

## 9. Build And Submit A New Eval Suite

Create a starter suite:

```bash
lmx eval suite init \
  --slug my-reasoning-eval \
  --name "My Reasoning Eval" \
  --category reasoning \
  --kind multiple_choice \
  --out my-reasoning-eval.json
```

Edit the generated JSON and replace sample items with your dataset. Then validate:

```bash
lmx eval suite validate my-reasoning-eval.json
```

Submit the suite for LocalMaxxing approval:

```bash
lmx eval suite submit my-reasoning-eval.json
```

Submitted suites may require approval before public runs can target them.

## 10. Safety Notes

- Always run `--dry-run` before `--submit`.
- Do not put gold answers inside `input`, `choices`, prompt text, or artifact text.
- Structured fields such as `gold`, `answer`, and `referenceAnswer` are redacted from suite discovery output and eval run artifacts, but arbitrary prose cannot be reliably redacted.
- Never commit API keys, `.env` files, or raw private eval datasets.
- For public comparison, record exact model ID, quantization, backend, command flags, hardware, and relevant runner versions.

## Common Errors

Missing API key:

```text
[localmaxxing:error] missing_api_key
```

Fix: run `lmx auth --key bhk_...` or set `LMX_API_KEY`.

Missing hardware:

```text
[localmaxxing:error] missing_hardware
```

Fix: run `lmx hardware --out hardware.json` and pass `--hardware hardware.json`.

Wrong suite type for lm-eval wrapper:

```text
[localmaxxing:error] suite_runner_mismatch
```

Fix: use `lmx eval run` for `CUSTOM` suites, or choose an `LM_EVAL_HARNESS` suite.

Missing lm-eval executable:

```text
[localmaxxing:error] command_failed
```

Fix: install with `pip install lm-eval` or pass `--lm-eval-bin <path>`.

Terminal run-directory and job errors:

- `checkpoint_exists`: use the exact invocation with `--resume auto`, or choose a
  new directory for an intentional rerun.
- `checkpoint_mismatch`: restore the original dataset/shard, task manifests,
  endpoint/model, hardware, and execution options; never mix runs in one directory.
- `checkpoint_already_submitted`: read the durable receipt, or rerun intentionally
  in a new directory.
- `terminal_job_running`: inspect the existing owner with `eval terminal status`;
  do not start a competing worker in the same directory.
- `terminal_cancelled`: a resumable operator stop, not a benchmark failure.
- `incomplete_shard`: inspect `status`, `logs`, and unscored task results, then
  resume; publication requires a scored result for every fetched task.
