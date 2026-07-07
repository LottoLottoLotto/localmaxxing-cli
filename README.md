# LocalMaxxing CLI

The official CLI for [localmaxxing.com](https://localmaxxing.com) — benchmark and evaluate local LLM inference and submit results from your terminal.

## Install

### Download Binary (Recommended)

Download a pre-built archive for your platform from the [latest release](https://github.com/LottoLottoLotto/localmaxxing-cli/releases/latest):

| Platform | Asset |
|----------|-------|
| Linux (amd64) | `lmx-linux-amd64.tar.gz` |
| Linux (arm64) | `lmx-linux-arm64.tar.gz` |
| macOS (Apple Silicon) | `lmx-darwin-arm64.tar.gz` |
| Windows (amd64) | `lmx-windows-amd64.zip` |

Linux and macOS binaries ship inside `.tar.gz` archives so executable bits survive the download. Windows binaries ship inside a `.zip`. Each archive includes `lmx`/`lmx.exe` and the bundled `lmx-llama-score-hellaswag` helper.

```bash
# Linux / macOS (adjust the asset name for your platform)
base=https://github.com/LottoLottoLotto/localmaxxing-cli/releases/latest/download
curl -fsSLO "$base/lmx-linux-amd64.tar.gz"
curl -fsSLO "$base/checksums.txt"
sha256sum --check --ignore-missing checksums.txt   # macOS: shasum -a 256 --check --ignore-missing checksums.txt
tar -xzf lmx-linux-amd64.tar.gz
sudo mv lmx /usr/local/bin/
lmx --help
```

Every release includes `checksums.txt` with SHA-256 hashes keyed by asset basename, so `--check --ignore-missing` succeeds even when you downloaded a single asset.

### Update

Once `lmx` is installed from a release archive, update it in place:

```bash
lmx update
```

`lmx update` downloads the newest GitHub release asset for the current OS/architecture, verifies it against `checksums.txt`, replaces the running `lmx` binary, and updates the bundled scorer next to it when present. Use `lmx update --dry-run` to print the asset URL and target path without changing files.

Windows cannot safely replace a running `.exe`; download the latest zip and replace `lmx.exe` after the process exits.

On Windows, download the `.zip`, extract `lmx.exe`, and compare hashes in PowerShell:

```powershell
(Get-FileHash lmx-windows-amd64.zip -Algorithm SHA256).Hash
Select-String -Path checksums.txt -Pattern lmx-windows-amd64.zip
```

> macOS: release binaries are not signed or notarized. If Gatekeeper blocks the first run, clear the quarantine attribute: `xattr -d com.apple.quarantine /usr/local/bin/lmx`.

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

## Agent Skill

The binary embeds an Agent Skill documenting the CLI so coding agents can discover and use `lmx` without external docs.

```bash
lmx skill print                       # print SKILL.md to stdout
lmx skill print --out SKILL.md        # or write it to a file
lmx skill install                     # install into ./.claude/skills/localmaxxing-cli
lmx skill install --dir ~/.claude/skills
```

`lmx skill install` writes the skill tree to `<dir>/localmaxxing-cli/`. It works with any harness that discovers `skills/<name>/SKILL.md`, including Claude `.claude/skills` and GitHub `.github/skills`.

## Hardware Metadata

Submissions require a hardware JSON object. Generate one on the machine running the benchmark:

```bash
lmx hardware --out hardware.json
```

If you cannot run detection on the endpoint host yet, generate a reviewed template from known specs:

```bash
lmx hardware template --gpu-name "RTX 3090" --gpu-count 2 --vram-gb 24 --cpu "Ryzen 9 9950X" --ram-gb 96 --os Linux > hardware.json
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

### Pull a saved setup

If you have saved hardware/engine setups in your LocalMaxxing account, pull one into a `hardware.json` instead of detecting locally — handy when the agent host is not the inference rig. Both commands require an API key (`--api-key`, `LMX_API_KEY`, or saved config) and return only your own setups.

```bash
lmx setups list
lmx setups pull --default --out hardware.json
lmx setups pull --name "2x RTX 3090" --out hardware.json
lmx setups pull --id <setupId> --out hardware.json
```

`setups pull` selects by `--id`, case-insensitive `--name`, or `--default`; with no selector it uses your default setup, or the only setup when you have exactly one. Without `--out` it prints the hardware JSON to stdout.

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

For remote endpoint submissions, `--hardware` must describe the server running the endpoint, not the client machine running `lmx`. Run `lmx hardware --out hardware.json` on that server, or provide an equivalent reviewed hardware JSON for that server. If a remote run already has metrics but lacks hardware, attach it without rerunning: `lmx benchmark add-hardware runs/Model/run.json --hardware hardware.json`.

Remote endpoint runs issue one untimed warmup request and three timed iterations by default, reporting the median of each metric plus per-iteration `samples` and `sampleStats` (min/p50/mean/max/stddev). Tune with `--warmup <n>` and `--iterations <n>`; `--warmup 0 --iterations 1` restores single-shot measurement. Decode throughput is measured over the inter-token window (first to last streamed token) when more than one token arrives.

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

Local benchmark commands run without a time limit by default; pass `--command-timeout-seconds <n>` to abort hung runs.

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

Inspect a saved run for post-run fixes before submission:

```bash
lmx benchmark fixup runs/Qwen-Qwen3-8B/run.json
lmx benchmark add-hardware runs/Qwen-Qwen3-8B/run.json --hardware hardware.json
```


Analyze and export runs:

```bash
lmx benchmark runs stats --group-by quantization --metric tokSOut
lmx benchmark runs compare --by quantization --metric tokSOut
lmx benchmark runs export --format csv --out runs.csv
```

`stats`, `compare`, and `export` accept `--runs-dir`, plus filters such as `--model`, `--engine`, `--mode`, `--quantization`, `--kind`, and `--hardware-name`. Group stats report min/p50/mean/max/p95/stddev and the best single run; group comparisons rank by the median (`p50`) so a single outlier run cannot win. `export` defaults to JSON and supports `--fields path,hfId,hardware,tokSOut` for custom extraction.
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

Remote sweeps pre-warm the target context prefix, inspect llama.cpp `/slots` for `n_prompt_tokens_cache`, then time a streaming probe with the same prefix. The default filler is a deterministic varied-word sequence (a single repeated word is unrealistically friendly to prefix caching); pass `--filler-token <word>` to override. Reported `promptTokens` come from the endpoint's `usage.prompt_tokens` when available, and cached points estimate prefill speed from the non-cached suffix only. If `/slots` reports no retained prompt cache, the CLI records a warning and labels the point as a cold inline prefill measurement instead of cached-context speed.

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
lmx model search Qwen3-8B-Q4_K_M.gguf   # GGUF filenames/paths are normalized to the model name
```

Resolve a remote endpoint's served-model alias to likely HuggingFace IDs:

```bash
lmx model resolve-remote --base-url http://server:8080
lmx endpoint discover --base-url http://server:8080 --include-server-metadata
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
