# LocalMaxxing CLI

Command-line tools for LocalMaxxing benchmark and eval workflows. The CLI helps you discover LocalMaxxing schemas and suites, collect hardware metadata, run inference benchmarks, measure KV-cache/context slowdown, validate payloads, and submit benchmark or eval results to LocalMaxxing.

This repository is the public CLI workspace. The current primary executable is the Go CLI in `cmd/lmx`; the npm workspace package is `packages/localmaxxing-cli` and publishes as `localmaxxing` with `localmaxxing` and `lmx` binaries.

## Features

- Authenticate with LocalMaxxing using a saved API key or `LMX_API_KEY`.
- Generate a best-effort hardware metadata file for submissions.
- Discover models, eval suites, schemas, methodology tips, and agent context from the LocalMaxxing API.
- Run or package local and remote inference benchmarks for vLLM, llama.cpp, SGLang, Ollama, and custom commands.
- Measure KV-cache and long-context slowdown across configurable context lengths.
- Run LM-Eval Harness suites and upload existing lm-eval result JSON.
- Run custom LocalMaxxing eval suites against local or public OpenAI-compatible endpoints.
- Upload large eval traces as bucket-backed artifacts.
- Save benchmark profiles and edit, rerun, or submit saved benchmark runs.

## Requirements

- Go 1.22 or newer for the current CLI build.
- Node.js 20.9 or newer and npm for workspace scripts and package publishing checks.
- Optional: Python plus `transformers` for Hugging Face tokenizer fallbacks.
- Optional: `lm-eval` for LM-Eval Harness suites.

## Install And Build

From a checkout:

```bash
git clone https://github.com/LottoLottoLotto/localmaxxing-cli.git
cd localmaxxing-cli
npm install
npm run build:cli
./dist/lmx --help
```

You can also build the Go executable directly:

```bash
go build -o dist/lmx ./cmd/lmx
```

After the npm package is published, install or run it with:

```bash
npm install -g localmaxxing
lmx --help
```

or:

```bash
npx localmaxxing --help
```

Optional tokenizer fallback:

```bash
python -m pip install transformers
```

Optional lm-eval support:

```bash
python -m pip install lm-eval
```

## Authentication

Create a LocalMaxxing API key from your dashboard, then either pass it per command:

```bash
lmx benchmark dry-run benchmark.json --api-key bhk_...
```

set it in the environment:

```bash
export LMX_API_KEY=bhk_...
```

or save it locally:

```bash
lmx auth --key bhk_...
lmx auth
lmx auth --logout
```

Saved CLI config lives under `~/.config/localmaxxing`.

## Hardware Metadata

Submissions require a hardware JSON object. Generate one on the machine that actually ran the benchmark and review it before submitting results:

```bash
lmx hardware --out hardware.json
```

Example:

```json
{
  "hwClass": "DISCRETE_GPU",
  "gpuName": "RTX 4090",
  "vramGb": 24
}
```

## Discover Context, Suites, And Models

Fetch current API context, approved suite metadata, and canonical model IDs:

```bash
lmx context --out localmaxxing-agent-context.json
lmx eval suite list --out localmaxxing-suites.json
lmx eval suite search reasoning --out reasoning-suites.json
lmx eval suite show hellaswag --out hellaswag-suite.json
lmx model search qwen3-8b --out models.json
```

Discovery output is redacted defensively so fields such as `gold`, `answer`, and `referenceAnswer` are not written to agent-facing files.

## Benchmarks

Remote endpoint mode measures throughput and latency from an OpenAI-compatible server:

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

For remote endpoint submissions, `--hardware` must describe the server running the endpoint, not the client machine running `lmx`. Run `lmx hardware --out hardware.json` on that server, or provide an equivalent reviewed hardware JSON for that server, before `benchmark dry-run` or `benchmark submit`.

Local host mode runs a benchmark command on the machine where the runtime is installed:

```bash
lmx benchmark run llama.cpp \
  --mode local \
  --hf-id Qwen/Qwen3-8B \
  --quantization Q4_K_M \
  --hardware hardware.json \
  --command "llama-bench -m model.gguf -p 512 -n 128" \
  --dry-run
```

For llama.cpp, the CLI can generate the benchmark command from a model path:

```bash
lmx benchmark run llama.cpp \
  --mode local \
  --hf-id Qwen/Qwen3-8B \
  --quantization Q4_K_M \
  --hardware hardware.json \
  --model-path model.gguf \
  --prompt-tokens 512 \
  --output-tokens 128 \
  --dry-run
```

Validate and submit a saved benchmark payload:

```bash
lmx benchmark dry-run benchmark.json
lmx benchmark submit benchmark.json
```

`bench` is accepted as a shorter alias for `benchmark`. Use `--out <path>` to choose the output file, `--json-status` for machine-readable progress events, and `--quiet` to suppress progress events.

## Saved Profiles And Runs

Save repeated benchmark or server options in a profile:

```bash
lmx profile save my-4090 \
  --mode local \
  --hf-id Qwen/Qwen3-8B \
  --quantization Q4_K_M \
  --hardware hardware.json

lmx benchmark run llama.cpp --profile my-4090 --model-path model.gguf --dry-run
```

Saved benchmark run files can be listed, viewed, edited, rerun, submitted, or deleted:

```bash
lmx benchmark runs list
lmx benchmark runs show runs/Qwen-Qwen3-8B/run.json
lmx benchmark runs edit runs/Qwen-Qwen3-8B/run.json --set-json '{"tokSOut":120}'
lmx benchmark runs rerun runs/Qwen-Qwen3-8B/run.json --dry-run
lmx benchmark runs submit runs/Qwen-Qwen3-8B/run.json
lmx benchmark runs delete runs/Qwen-Qwen3-8B/run.json --yes
```

Saved runs can also be analyzed or extracted without submitting anything:

```bash
lmx benchmark runs stats --group-by quantization --metric tokSOut
lmx benchmark runs stats --group-by hardware --model Qwen/Qwen3-8B
lmx benchmark runs compare --by quantization --metric tokSOut
lmx benchmark runs compare runs/base.json runs/candidate.json --metrics tokSOut,ttftMs
lmx benchmark runs export --format csv --out runs.csv
```

`stats`, `compare`, and `export` accept `--runs-dir`, plus filters such as `--model`, `--engine`, `--mode`, `--quantization`, `--kind`, and `--hardware-name`. `export` defaults to JSON and supports `--fields path,hfId,hardware,tokSOut` for custom extraction.

## KV-Cache Context Sweeps

Use `kvcache run` to record how prefill, TTFT, decode TPS, and total TPS change as context length grows. It writes `localmaxxing-kvcache.json` by default.

Local llama.cpp example:

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

Remote OpenAI-compatible endpoint example:

```bash
lmx kvcache run vllm \
  --mode remote \
  --base-url http://server:8000 \
  --hf-id Qwen/Qwen3-8B \
  --served-model Qwen/Qwen3-8B \
  --levels 10000,20000,30000,40000 \
  --output-tokens 128
```

Remote sweeps pre-warm the target context prefix, inspect llama.cpp `/slots` for `n_prompt_tokens_cache`, then time a streaming probe with the same prefix. If `/slots` reports no retained prompt cache, the CLI records a warning and labels the point as a cold inline prefill measurement instead of cached-context speed.

## Eval Workflows

Run an LM-Eval Harness suite through the CLI wrapper:

```bash
lmx eval lm-eval hellaswag \
  --model Qwen/Qwen3-8B \
  --backend hf \
  --hardware hardware.json \
  --dry-run
```

Upload existing lm-eval output:

```bash
lmx eval run local-open-llm-core \
  --model Qwen/Qwen3-8B \
  --results localmaxxing-lm-eval-results.json \
  --hardware hardware.json \
  --submit
```

Run a custom suite against a local OpenAI-compatible endpoint:

```bash
lmx eval run my-custom-suite \
  --model Qwen/Qwen3-8B \
  --base-url http://localhost:8000 \
  --hardware hardware.json \
  --submit
```

For public endpoints that LocalMaxxing can reach directly, use server-side execution:

```bash
lmx eval execute my-custom-suite \
  --model Qwen/Qwen3-8B \
  --base-url https://my-public-endpoint.example \
  --hardware hardware.json \
  --submit
```

Run an LM-Judge suite:

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

## Eval Suite Authoring

Create a starter suite:

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

Submitted suites start as `PENDING` and appear publicly after admin approval.

## Artifact Storage

Large eval traces should be uploaded as artifact bundles instead of inline run artifacts:

```bash
lmx eval artifacts upload traces.jsonl \
  --format jsonl \
  --item-count 1000 \
  --out artifact-bundle.json
```

Lower-level storage commands support artifact bundles and bucket-backed datasets:

```bash
lmx eval storage upload dataset.jsonl \
  --kind dataset \
  --format jsonl \
  --out dataset-storage.json

lmx eval storage download <storageKey> --out traces.jsonl
```

## Development

Install dependencies:

```bash
npm install
```

Run the Go test suite:

```bash
go test ./...
```

Build the CLI:

```bash
npm run build:cli
```

Build the legacy TypeScript workspace package, if needed for migration comparison:

```bash
npm run build:ts-cli
```

Check npm package contents:

```bash
npm pack -w localmaxxing --dry-run
```

## Documentation

- `docs/running-evals.md`
- `docs/eval-suite-authoring.md`
- `docs/lm-eval-compatibility.md`
- `packages/localmaxxing-cli/README.md`

## License

MIT. See `LICENSE`.
