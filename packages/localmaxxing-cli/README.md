# LocalMaxxing CLI

Public CLI for authoring, running, validating, and submitting LocalMaxxing eval suites.

This package is standalone. It does not include the private LocalMaxxing web app, database, Prisma schema, deployment configuration, or server internals.

## Implementation Status

The Go CLI is the primary executable. Optional Python helpers are used for ML-specific behavior such as Hugging Face token counting. The previous TypeScript CLI remains in `src/index.ts` temporarily as a legacy migration reference and should not be treated as the source of truth.

Build the Go CLI from this repository:

```bash
go build -o dist/lmx ./cmd/lmx
```

Optional tokenizer fallback requires Python plus `transformers`:

```bash
python -m pip install transformers
```

## Install

From npm after publish:

```bash
npm install -g localmaxxing
```

Or run directly:

```bash
npx localmaxxing --help
```

From this repository checkout:

```bash
git clone https://github.com/LottoLottoLotto/localmaxxing-cli.git
cd localmaxxing-cli
npm install
npm run build:cli
npm link -w localmaxxing
lmx --help
```

## Authentication

Create a LocalMaxxing API key from your dashboard, then pass it with `--api-key` or set:

```bash
export LMX_API_KEY=bhk_...
```

On Windows PowerShell:

```powershell
$env:LMX_API_KEY = "bhk_..."
```

Or save it for future CLI commands:

```bash
lmx auth --key bhk_...
lmx auth
lmx auth --logout
```

## Hardware File

Submissions require a hardware JSON object for the machine that actually ran the benchmark. Example `hardware.json`:

```json
{
  "hwClass": "DISCRETE_GPU",
  "gpuName": "RTX 4090",
  "vramGb": 24
}
```

You can generate a best-effort hardware file from the current machine:

```bash
lmx hardware --out hardware.json
```

For remote endpoint benchmarks, generate or review this file on the server running the endpoint. The CLI will not use client auto-detected hardware for remote submissions.

## Submit An Inference Benchmark

The Go CLI supports two benchmark modes:

Remote endpoint mode measures TPS from the client against an OpenAI-compatible server. Use this when you do not have shell access to the host running the model:

```bash
lmx benchmark run vllm \
  --mode remote \
  --base-url http://server:8000 \
  --hf-id Qwen/Qwen3-8B \
  --served-model Qwen/Qwen3-8B \
  --quantization fp16 \
  --hardware hardware.json \
  --max-tokens 256 \
  --dry-run
```

If the endpoint is on another machine, `hardware.json` must describe that server. Create it on the server with `lmx hardware --out hardware.json`, or provide an equivalent reviewed file, before API dry-run or submit.

If `--served-model` is omitted, the CLI tries `GET /v1/models` and uses the matching or first returned model ID before falling back to `--hf-id`.

For Ollama, use remote mode against the native Ollama endpoint. The CLI calls `/api/generate` and maps Ollama's `prompt_eval_*`, `eval_*`, and `total_duration` fields into benchmark metrics:

```bash
lmx benchmark run ollama \
  --mode remote \
  --base-url http://localhost:11434 \
  --hf-id Qwen/Qwen3-8B \
  --served-model qwen3:8b \
  --quantization Q4_K_M \
  --hardware hardware.json \
  --max-tokens 256
```

Local host mode runs a benchmark command on the machine where the model/runtime is installed. Use this when you are on the server and can run `llama-bench` or another benchmark executable:

```bash
lmx benchmark run llama.cpp \
  --mode local \
  --hf-id Qwen/Qwen3-8B \
  --quantization Q4_K_M \
  --command "llama-bench -m model.gguf -p 512 -n 128" \
  --dry-run
```

For llama.cpp, the CLI can generate the `llama-bench` command:

```bash
lmx benchmark run llama.cpp \
  --mode local \
  --hf-id Qwen/Qwen3-8B \
  --quantization Q4_K_M \
  --model-path model.gguf \
  --prompt-tokens 512 \
  --output-tokens 128 \
  --dry-run
```

If `--mode` is omitted, the CLI infers remote mode from `--base-url` and local mode from `--command` or `--results`.

Local and remote benchmark payloads use the same conceptual metric fields:

```text
tokSPrefill = promptTokens / prefillSeconds
tokSOut     = outputTokens / decodeSeconds
tokSTotal   = (promptTokens + outputTokens) / totalSeconds
ttftMs      = request-to-first-token time, or local estimate from prefill
```

For local `llama-bench`, `pp<N>` provides prefill throughput and `tg<N>` provides decode throughput. The CLI derives comparable values when missing:

```text
ttftMs    ~= promptTokens / tokSPrefill * 1000
tokSTotal ~= (promptTokens + outputTokens) / ((promptTokens / tokSPrefill) + (outputTokens / tokSOut))
```

Payloads include `metricSource`, `timingSource`, and `ttftSource` so dashboard comparisons can distinguish local runtime measurements from client-observed remote endpoint measurements.

For agent workflows, pass `--json-status` to emit machine-readable progress events on stderr, or `--quiet` to suppress progress events:

```bash
lmx benchmark run vllm \
  --mode remote \
  --base-url http://server:8000 \
  --hf-id Qwen/Qwen3-8B \
  --quantization fp16 \
  --json-status \
  --out benchmark.json
```

Save repeated options in a profile:

```bash
lmx profile save my-4090 \
  --mode local \
  --hf-id Qwen/Qwen3-8B \
  --quantization Q4_K_M \
  --hardware hardware.json

lmx benchmark run llama.cpp --profile my-4090 --model-path model.gguf --dry-run
```

Create a hardware file with:

```bash
lmx hardware init --out hardware.json
```

Agent-first path: run any benchmark tool, then route the observed values into LocalMaxxing fields explicitly. The CLI writes a payload and uses the API dry-run as the source of truth for schema validation:

```bash
lmx benchmark run vllm \
  --hf-id Qwen/Qwen3-8B \
  --quantization fp16 \
  --hardware hardware.json \
  --tok-s-out 120.4 \
  --tok-s-prefill 1840.2 \
  --tok-s-total 101.9 \
  --ttft-ms 86 \
  --peak-vram-gb 18.7 \
  --context-length 8192 \
  --batch-size 1 \
  --dry-run
```

You can also give the CLI a local benchmark command and let it pre-fill fields from common output labels. Treat this as convenience only; explicit flags and editable JSON are the robust path:

```bash
lmx benchmark run llama.cpp \
  --hf-id Qwen/Qwen3-8B \
  --quantization Q4_K_M \
  --hardware hardware.json \
  --command "llama-bench -m Qwen3-8B-Q4_K_M.gguf -p 512 -n 128" \
  --dry-run
```

The same wrapper accepts vLLM and SGLang benchmark output. Prefer their JSON output when available:

```bash
lmx benchmark run vllm \
  --hf-id Qwen/Qwen3-8B \
  --quantization fp16 \
  --hardware hardware.json \
  --bench-kind throughput \
  --input-len 512 \
  --output-len 128 \
  --num-prompts 100 \
  --benchmark-output vllm-throughput.json \
  --dry-run

lmx benchmark run sglang \
  --hf-id meta-llama/Llama-3.1-8B-Instruct \
  --quantization fp8 \
  --hardware hardware.json \
  --bench-kind serving \
  --base-url http://127.0.0.1:30000 \
  --input-len 512 \
  --output-len 128 \
  --num-prompts 100 \
  --dry-run
```

Built-in vLLM presets generate `vllm bench serve`, `vllm bench throughput`, or `vllm bench latency`. Built-in SGLang presets generate `python -m sglang.bench_serving`, `sglang.bench_offline_throughput`, `sglang.bench_one_batch`, or `sglang.bench_one_batch_server`. Use `--extra-bench-args "..."` for engine-specific flags or `--command "..."` to bypass the preset completely.

For an already running OpenAI-compatible endpoint, the CLI measures streaming TTFT, output TPS, total TPS, and prefill TPS when token counts are available from the endpoint or Hugging Face tokenizer:

```bash
lmx benchmark run vllm \
  --hf-id Qwen/Qwen3-8B \
  --served-model Qwen/Qwen3-8B \
  --quantization fp16 \
  --hardware hardware.json \
  --base-url http://localhost:8000 \
  --max-tokens 256 \
  --dry-run
```

Create a benchmark payload JSON matching `POST /api/benchmarks`, then validate it without writing:

```bash
lmx benchmark dry-run benchmark.json --api-key bhk_...
```

Submit after the dry-run passes:

```bash
lmx benchmark submit benchmark.json --api-key bhk_...
```

`bench` is accepted as a shorter alias for `benchmark`.

Saved `lmx-bench` result files are also accepted. If the JSON has a top-level `payload` field, the CLI submits that payload automatically.

If engine token counts are missing or marked as estimated and the JSON includes generated text (`outputText`, `generatedText`, `completion`, `response`, or `text`), the CLI uses the model's Hugging Face tokenizer to fill `outputTokens`. If the submitted model has no tokenizer, it tries `tokenizer_name` or `base_model_name_or_path` from `config.json`.

`benchmark run` writes a saved run under `runs/<model>/<timestamp>.json` by default. Use `--out <path>` to save elsewhere, `--submit` after the dry-run passes, or explicit metric overrides such as `--tok-s-out`, `--tok-s-prefill`, `--tok-s-total`, `--ttft-ms`, and `--peak-vram-gb` when a benchmark tool prints an unsupported label.

Saved benchmark run files can be listed, viewed, edited, rerun, submitted, dry-run validated, or deleted:

```bash
lmx benchmark runs list
lmx benchmark runs show runs/Qwen-Qwen3-8B/run.json
lmx benchmark runs edit runs/Qwen-Qwen3-8B/run.json --set-json '{"tokSOut":120}'
lmx benchmark runs rerun runs/Qwen-Qwen3-8B/run.json --dry-run
lmx benchmark runs submit runs/Qwen-Qwen3-8B/run.json
lmx benchmark runs delete runs/Qwen-Qwen3-8B/run.json --yes
```

Saved runs can also be analyzed or extracted locally:

```bash
lmx benchmark runs stats --group-by quantization --metric tokSOut
lmx benchmark runs stats --group-by hardware --model Qwen/Qwen3-8B
lmx benchmark runs compare --by quantization --metric tokSOut
lmx benchmark runs compare runs/base.json runs/candidate.json --metrics tokSOut,ttftMs
lmx benchmark runs export --format csv --out runs.csv
```

`stats`, `compare`, and `export` accept `--runs-dir`, plus filters such as `--model`, `--engine`, `--mode`, `--quantization`, `--kind`, and `--hardware-name`. `export` defaults to JSON and supports `--fields path,hfId,hardware,tokSOut` for custom extraction.

Agents can skip `benchmark run` entirely: create any JSON object matching `POST /api/benchmarks`, then call `lmx benchmark dry-run benchmark.json` and `lmx benchmark submit benchmark.json`. Fetch `lmx context --out localmaxxing-agent-context.json` first for the current schema, accepted engines, and methodology tips.

## Measure KV-Cache Slowdown Across Context Lengths

Use `kvcache run` to load progressively larger contexts and record how prefill, TTFT, decode TPS, and total TPS change. It writes `localmaxxing-kvcache.json` by default.

Local mode runs one engine benchmark per requested context level. For `llama.cpp`, `--levels` maps to `llama-bench -d <n>` so the KV cache is prefilled to that depth; `--prompt-tokens` controls the measured prompt-processing chunk and defaults to `512`:

```bash
lmx kvcache run llama.cpp \
  --mode local \
  --hf-id Qwen/Qwen3-8B \
  --quantization Q4_K_M \
  --model-path model.gguf \
  --levels 10000,20000,30000,40000 \
  --prompt-tokens 512 \
  --output-tokens 128
```

For local custom tools, provide a template. `{input}` is replaced with the current context level and `{output}` with the decode length:

```bash
lmx kvcache run custom \
  --mode local \
  --hf-id Qwen/Qwen3-8B \
  --command-template "my-bench --input {input} --output {output}" \
  --levels 10000,20000,30000,40000
```

For local vLLM, `--levels` maps to `vllm bench latency --input-len <n>`. vLLM does not expose a separate llama-bench-style `-d` depth flag, so this measures latency/throughput with a long input context. The CLI sets `--max-model-len` to at least `input + output` when `--context-length` is not provided and writes one `--output-json` file per level:

```bash
lmx kvcache run vllm \
  --mode local \
  --hf-id Qwen/Qwen3-8B \
  --levels 10000,20000,30000,40000 \
  --output-tokens 128 \
  --batch-size 1
```

Remote mode pre-warms the target context prefix, inspects llama.cpp `/slots` for `n_prompt_tokens_cache`, then times a streaming OpenAI-compatible chat completion with the same prefix plus a probe:

```bash
lmx kvcache run vllm \
  --mode remote \
  --base-url http://server:8000 \
  --hf-id Qwen/Qwen3-8B \
  --served-model Qwen/Qwen3-8B \
  --levels 10000,20000,30000,40000 \
  --output-tokens 128
```

If `/slots` reports no retained prompt cache, the CLI records a warning and labels the point as a cold inline prefill measurement instead of cached-context speed. Remote context sizing combines the requested context level with endpoint `usage.prompt_tokens` when present, because llama.cpp may report only new non-cached prompt tokens. The output includes `methodology`, `cacheReuse`, `timingSource`, `ttftMs`, `tokSPrefill`, `tokSOut`, and `tokSTotal` per point so results can be compared across context depths.

## Discover Eval Requirements

Agents can pull the current LocalMaxxing schemas, endpoints, benchmark requirements, methodology tips, approved suite list, and per-suite run instructions directly from the API:

```bash
lmx context --out localmaxxing-agent-context.json
lmx eval suite list --out localmaxxing-suites.json
lmx eval suite search reasoning --out reasoning-suites.json
lmx eval suite show hellaswag --out hellaswag-suite.json
lmx model search qwen3-8b --out models.json
```

`eval suite show` returns the suite document, task keys, scoring method, aggregation mode, and `agentInstructions` such as lm-eval command templates or server-side custom eval payloads.

Suite discovery output and eval run artifacts are redacted defensively so fields such as `gold`, `answer`, and `referenceAnswer` are not written to agent-facing files or uploaded as run artifacts.

## Artifact Bundles And Storage

Large eval traces should be uploaded as bucket-backed artifacts instead of inline run artifacts:

```bash
lmx eval artifacts upload traces.jsonl \
  --format jsonl \
  --item-count 1000 \
  --out artifact-bundle.json
```

The lower-level storage commands support both artifact bundles and bucket-backed datasets:

```bash
lmx eval storage upload dataset.jsonl \
  --kind dataset \
  --format jsonl \
  --out dataset-storage.json

lmx eval storage download <storageKey> --out traces.jsonl
```

## Run A Custom Eval

Start a local OpenAI-compatible server, then run:

```bash
lmx eval run my-custom-suite \
  --model Qwen/Qwen3-8B \
  --base-url http://localhost:8000 \
  --hardware hardware.json \
  --submit
```

For a public OpenAI-compatible endpoint that LocalMaxxing can reach directly, use server-side execution:

```bash
lmx eval execute my-custom-suite \
  --model Qwen/Qwen3-8B \
  --base-url https://my-public-endpoint.example \
  --hardware hardware.json \
  --submit
```

Use `eval run` for localhost/private endpoints; use `eval execute` for public endpoints you want LocalMaxxing to call and score server-side.

## Run An LM-Judge Eval

```bash
lmx eval run my-judge-suite \
  --model Qwen/Qwen3-8B \
  --base-url http://localhost:8000 \
  --judge-base-url https://api.openai.com \
  --judge-model gpt-4.1-mini \
  --judge-api-key "$OPENAI_API_KEY" \
  --hardware hardware.json \
  --submit
```

## Run An LM-Eval Suite

Run lm-eval-harness through the CLI wrapper:

```bash
lmx eval lm-eval hellaswag \
  --model Qwen/Qwen3-8B \
  --backend hf \
  --hardware hardware.json \
  --dry-run
```

For `--backend hf`, the CLI defaults `--model-args` to `pretrained=<model>`. Override it for other backends:

```bash
lmx eval lm-eval hellaswag \
  --model Qwen/Qwen3-8B \
  --backend vllm \
  --model-args pretrained=Qwen/Qwen3-8B,tensor_parallel_size=1 \
  --hardware hardware.json \
  --submit
```

The wrapper fetches the suite, runs `lm_eval`, parses the output JSON, writes a LocalMaxxing run payload, then dry-runs or submits it.

You can still run lm-eval-harness yourself:

```bash
lm_eval \
  --model hf \
  --model_args pretrained=Qwen/Qwen3-8B \
  --tasks arc_challenge,hellaswag,mmlu \
  --output_path localmaxxing-lm-eval-results.json
```

Then upload:

```bash
lmx eval run local-open-llm-core \
  --model Qwen/Qwen3-8B \
  --results localmaxxing-lm-eval-results.json \
  --hardware hardware.json \
  --submit
```

## Create A New Eval Suite

```bash
lmx eval suite init \
  --slug my-reasoning-eval \
  --name "My Reasoning Eval" \
  --category reasoning \
  --kind multiple_choice \
  --out my-reasoning-eval.json
```

Edit the generated JSON, then validate and submit:

```bash
lmx eval suite validate my-reasoning-eval.json
lmx eval suite submit my-reasoning-eval.json
```

## Agent-Friendly Errors

The CLI emits structured diagnostics such as:

```text
[localmaxxing:error] suite_validation_failed
Suite validation failed.
Fix:
- Edit the suite JSON file and rerun eval suite validate.
Details:
[
  "slug must be 3-64 lowercase alphanumeric characters with hyphens"
]
```

Agents should read the error code, apply the `Fix:` hints, and rerun the command.

## Docs

- `docs/running-evals.md`
- `docs/eval-suite-authoring.md`
- `docs/lm-eval-compatibility.md`

## Publish

```bash
npm version patch
npm publish --access public
```
