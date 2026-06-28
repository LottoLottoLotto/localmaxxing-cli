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

HumanEval and MBPP are **execution-based** code evals (`scoring: code_execution`,
the default for those datasets). The CLI generates a solution per problem, then
runs `solution + hidden tests` inside a hardened sandbox and records pass/fail.
Build the sandbox image once:

```bash
docker build -t lmx-sandbox sandbox
```

The container runs with `--network none`, a read-only root filesystem, dropped
capabilities, `no-new-privileges`, and CPU/memory/PID/time limits; the harness
adds per-task CPU/RAM/wall limits as defense in depth. Override the launcher with
`--sandbox-runtime podman`, `--sandbox-image <name>`, `--sandbox-memory`,
`--sandbox-cpus`, or replace it entirely with `--sandbox-cmd` (e.g.
`--sandbox-cmd "python3 sandbox/run_sandbox.py"` for hosts without Docker — note
that bypasses container isolation and should only be used on disposable boxes).

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
