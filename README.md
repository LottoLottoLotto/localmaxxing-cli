# LocalMaxxing CLI

The official CLI for [localmaxxing.com](https://localmaxxing.com) — benchmark and evaluate local LLM inference and submit results from your terminal.

## Install

### Download Binary (Recommended)

Download a pre-built binary for your platform from the [latest release](https://github.com/LottoLottoLotto/localmaxxing-cli/releases/latest):

| Platform | Binary |
|----------|--------|
| Linux (amd64) | `lmx-linux-amd64` |
| Linux (arm64) | `lmx-linux-arm64` |
| macOS (Intel) | `lmx-darwin-amd64` |
| macOS (Apple Silicon) | `lmx-darwin-arm64` |
| Windows (amd64) | `lmx-windows-amd64.exe` |

```bash
# Linux / macOS example
chmod +x lmx-linux-amd64
sudo mv lmx-linux-amd64 /usr/local/bin/lmx
lmx --help
```

### Build From Source

Requires Go 1.22 or newer.

```bash
git clone https://github.com/LottoLottoLotto/localmaxxing-cli.git
cd localmaxxing-cli
go build -o lmx ./cmd/lmx
./lmx --help
```

## Authentication

Create an API key from your [LocalMaxxing dashboard](https://localmaxxing.com), then either set it in the environment:

```bash
export LMX_API_KEY=bhk_...
```

or save it locally:

```bash
lmx auth --key bhk_...
lmx auth
lmx auth --logout
```

Saved config lives under `~/.config/localmaxxing`. You can also pass `--api-key bhk_...` to any command directly.

## Hardware Metadata

Submissions require a hardware JSON object. Generate one on the machine running the benchmark:

```bash
lmx hardware --out hardware.json
```

Example output:

```json
{
  "hwClass": "DISCRETE_GPU",
  "gpuName": "RTX 4090",
  "vramGb": 24
}
```

> For remote endpoint benchmarks, run `lmx hardware` on the **server** hosting the model, not your local machine.

## Benchmarks

### Remote Endpoint (vLLM, SGLang, Ollama, custom OpenAI-compatible)

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

<<<<<<< HEAD
For remote endpoint submissions, `--hardware` must describe the server running the endpoint, not the client machine running `lmx`. Run `lmx hardware --out hardware.json` on that server, or provide an equivalent reviewed hardware JSON for that server, before `benchmark dry-run` or `benchmark submit`.

Remote endpoint runs issue one untimed warmup request and three timed iterations by default, reporting the median of each metric plus per-iteration `samples` and `sampleStats` (min/p50/mean/max/stddev). Tune with `--warmup <n>` and `--iterations <n>`; `--warmup 0 --iterations 1` restores single-shot measurement. Decode throughput is measured over the inter-token window (first to last streamed token) when more than one token arrives.

For Ollama, use remote mode against the native Ollama endpoint. The CLI calls `/api/generate` and maps Ollama's `prompt_eval_*`, `eval_*`, and `total_duration` fields into benchmark metrics:
=======
Ollama uses the native `/api/generate` endpoint:
>>>>>>> 61878f85bfd22dfab260cae8eb6ab51454cd9c9c

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

### Local llama.cpp

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

### Validate and Submit

```bash
lmx benchmark validate-local benchmark.json
lmx benchmark dry-run benchmark.json
lmx benchmark submit benchmark.json
```

`bench` is a shorter alias for `benchmark`. Use `--out <path>` to set the output file, `--json-status` for machine-readable progress, and `--quiet` to suppress output.

<<<<<<< HEAD
Local benchmark commands run without a time limit by default; pass `--command-timeout-seconds <n>` to abort hung runs.

## Saved Profiles And Runs
=======
## Saved Profiles and Runs
>>>>>>> 61878f85bfd22dfab260cae8eb6ab51454cd9c9c

Save repeated options as a profile:

```bash
lmx profile save my-4090 \
  --mode local \
  --hf-id Qwen/Qwen3-8B \
  --quantization Q4_K_M \
  --hardware hardware.json

lmx benchmark run llama.cpp --profile my-4090 --model-path model.gguf --dry-run
```

Manage saved run files:

```bash
lmx benchmark runs list
lmx benchmark runs show runs/Qwen-Qwen3-8B/run.json
lmx benchmark runs edit runs/Qwen-Qwen3-8B/run.json --set-json '{"tokSOut":120}'
lmx benchmark runs rerun runs/Qwen-Qwen3-8B/run.json --dry-run
lmx benchmark runs submit runs/Qwen-Qwen3-8B/run.json
lmx benchmark runs delete runs/Qwen-Qwen3-8B/run.json --yes
```

Analyze and export runs:

```bash
lmx benchmark runs stats --group-by quantization --metric tokSOut
lmx benchmark runs compare --by quantization --metric tokSOut
lmx benchmark runs export --format csv --out runs.csv
```

<<<<<<< HEAD
`stats`, `compare`, and `export` accept `--runs-dir`, plus filters such as `--model`, `--engine`, `--mode`, `--quantization`, `--kind`, and `--hardware-name`. Group stats report min/p50/mean/max/p95/stddev and the best single run; group comparisons rank by the median (`p50`) so a single outlier run cannot win. `export` defaults to JSON and supports `--fields path,hfId,hardware,tokSOut` for custom extraction.

=======
>>>>>>> 61878f85bfd22dfab260cae8eb6ab51454cd9c9c
## KV-Cache Context Sweeps

Measure how prefill, TTFT, and decode TPS change as context length grows:

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

Remote endpoint:

```bash
lmx kvcache run vllm \
  --mode remote \
  --base-url http://server:8000 \
  --hf-id Qwen/Qwen3-8B \
  --served-model Qwen/Qwen3-8B \
  --levels 10000,20000,30000,40000 \
  --output-tokens 128
```

<<<<<<< HEAD
Remote sweeps pre-warm the target context prefix, inspect llama.cpp `/slots` for `n_prompt_tokens_cache`, then time a streaming probe with the same prefix. The default filler is a deterministic varied-word sequence (a single repeated word is unrealistically friendly to prefix caching); pass `--filler-token <word>` to override. Reported `promptTokens` come from the endpoint's `usage.prompt_tokens` when available, and cached points estimate prefill speed from the non-cached suffix only. If `/slots` reports no retained prompt cache, the CLI records a warning and labels the point as a cold inline prefill measurement instead of cached-context speed.
=======
## Evals
>>>>>>> 61878f85bfd22dfab260cae8eb6ab51454cd9c9c

### Run a custom suite against a local endpoint

```bash
lmx eval run my-custom-suite \
  --model Qwen/Qwen3-8B \
  --base-url http://localhost:8000 \
  --hardware hardware.json \
  --submit
```

### LM-Eval Harness

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

### LM-Judge

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

## Discover Models and Suites

```bash
lmx context --out localmaxxing-agent-context.json
lmx eval suite list --out localmaxxing-suites.json
lmx eval suite search reasoning
lmx model search qwen3-8b
```

## Eval Suite Authoring

```bash
lmx eval suite init \
  --slug my-reasoning-eval \
  --name "My Reasoning Eval" \
  --category reasoning \
  --kind multiple_choice \
  --out my-reasoning-eval.json

lmx eval suite validate my-reasoning-eval.json
lmx eval suite submit my-reasoning-eval.json
```

Submitted suites start as `PENDING` and appear publicly after admin approval.

## Development

```bash
# Run tests
go test ./...

# Build
go build -o lmx ./cmd/lmx
```

## Optional Dependencies

- `lm-eval` — required for LM-Eval Harness suites: `pip install lm-eval`
- `transformers` — Hugging Face tokenizer fallback: `pip install transformers`

## License

MIT. See `LICENSE`.
