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

Ollama uses the native `/api/generate` endpoint:

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

## Saved Profiles and Runs

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

## Evals

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
