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

Run local bundles against an OpenAI-compatible endpoint:

```bash
lmx eval terminal run --task-dir ./tb-bundles \
  --api-url https://www.localmaxxing.com \
  --base-url http://localhost:8000 \
  --model Qwen/Qwen3-8B \
  --hardware hardware.json \
  --out terminal-run.json \
  --trace-dir terminal-traces \
  --dry-run
```

By default the built-in harness runs every turn in one persistent shell session
(`docker exec -i <container> bash -l`), so `cd`, `export`, and `source` carry
across turns (`protocol: "react-shell"`). Pass `--shell-mode stateless` to run
each command in a fresh `docker exec` with no shared state (`protocol:
"react-bash"`).

Terminal runs use Harbor-like long task budgets by default. If `--agent-timeout`
is omitted, each task gets `max(task.json agent.timeoutSec, 14400)` seconds
(four hours), so imported Terminal-Bench manifests with 15–60 minute limits do
not prematurely stop local long-running models. Built-in model requests default
to a 10-minute per-request timeout, and the first attempt reserves part of the
remaining task budget for one retry. Terminal model completions are capped at
16,384 tokens by default and 8,192 tokens on retry; pass `--max-tokens` to set
an explicit cap for both attempts. Per-shell-command execution remains bounded
by the remaining task budget; override with `--endpoint-timeout-seconds` or
`--command-timeout` when needed.

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
written; a verifier timeout or a missing/unparseable reward file scores the
task as failed (it is still submitted, matching harbor's fail-on-error
accounting). Agent timeouts do not abort the trial either: the task is
verified with whatever state the agent left behind, so partial work completed
before the deadline still counts. Oracle runs (`verify --oracle`) mirror
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

Inspect every declared public shard before starting a live run. By default this
downloads manifests only; add `--verify-bundles` to download and validate every
referenced archive without starting Docker, contacting a model or verifier, or
submitting results:

```bash
lmx eval terminal inspect terminal-bench-2-1 \
  --api-url https://www.localmaxxing.com \
  --verify-bundles \
  --json
```

Run and submit the inspected public dataset against the separate model origin:

```bash
lmx eval terminal run terminal-bench-2-1 \
  --api-url https://www.localmaxxing.com \
  --base-url http://localhost:8000 \
  --model Qwen/Qwen3-8B \
  --hardware hardware.json \
  --submit
```


The approved `terminal-bench-2-1` dataset deterministically partitions its 89
tasks into 10 disjoint shards. Each `eval terminal run` invocation acquires and
submits one shard; repeat it to build full benchmark coverage. A complete
89-task checkpoint produced by Harbor or another offline runner can be validated
and partitioned into the same 10 shard submissions without rerunning tasks or
contacting a model endpoint:

```bash
lmx eval terminal submit ./completed-terminal-run \
  --dataset terminal-bench-2-1 \
  --api-url https://www.localmaxxing.com \
  --hf-id Qwen/Qwen3-8B \
  --hardware hardware.json \
  --quantization Q4_K_M \
  --quant-format gguf \
  --dry-run \
  --out terminal-submit-batch.json
```

The dry-run output is `{ "dataset": "terminal-bench-2-1", "shards": [...] }`
with 10 ordered payloads carrying explicit `shardIndex` values. The CLI verifies
the exact canonical 89-task ID set before partitioning. Deferred `--dry-run` is
entirely offline: it proves the saved checkpoint and payload partition but cannot
prove `--api-url` routing or production readiness. Inspect the batch, run
`eval terminal inspect` against that LocalMaxxing origin, then remove `--dry-run`
and provide `--api-key` (or `LMX_API_KEY`) to submit all 10 payloads sequentially.
For a checkpoint that already contains one isolated shard, pass `--shard-index
<n>`; other dataset slugs always require an explicit shard index. Deferred
submission preserves shard-local and full-checkpoint token usage in `runConfig`
and never starts Docker, reruns a verifier, or calls the model endpoint.

`--api-url` always selects the LocalMaxxing dataset/submission origin;
`--base-url` selects the OpenAI-compatible model inference origin. Keep both
explicit in live-run commands rather than substituting one for the other.


The built-in `--agent terminus-2` adapter is embedded in release binaries. It
requires Harbor's Python environment at runtime but does not require a source
checkout or a loose adapter script.

Use a preferred external agent harness:

```bash
lmx eval terminal run --task-dir ./tb-bundles \
  --api-url https://www.localmaxxing.com \
  --agent-cmd './my-agent-wrapper.sh' \
  --agent-name my-agent \
  --agent-execution routed-shell \
  --model Qwen/Qwen3-8B \
  --hardware hardware.json \
  --out terminal-run.json \
  --dry-run
```

`--agent-cmd` runs on the host after the task container starts and before the
verifier runs. The command receives:

- `LMX_TERMINAL_CONTAINER` — Docker container name.
- `LMX_TERMINAL_TASK_ID` — task id.
- `LMX_TERMINAL_BUNDLE_DIR` / `LMX_TERMINAL_TASK_DIR` — imported bundle path.
- `LMX_TERMINAL_TASK_JSON` — manifest path.
- `LMX_TERMINAL_INSTRUCTION_FILE` — temporary file containing the task prompt.
- `LMX_TERMINAL_BASE_URL` and `LMX_TERMINAL_MODEL` — values passed to the CLI,
  when your external agent wants to call the model itself.
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
passes the task as direct prompt text, emits JSON event logs, copies live SQLite
model-cache state for local llama.cpp models, and exits non-zero on model/API
errors or no-tool-call runs so setup failures are not submitted as scored task
failures.

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
