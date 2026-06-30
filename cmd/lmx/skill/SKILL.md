---
name: localmaxxing-cli
description: Use the LocalMaxxing CLI (the `lmx` binary) to benchmark local and remote LLM inference (tokens/sec, TTFT, peak VRAM), run quality evals (custom suites, lm-eval harness, LM-judge, eval shards, Terminal-Bench), generate hardware metadata, run KV-cache context sweeps, and submit results to localmaxxing.com. Covers authentication, choosing local vs remote benchmark mode, controlling the benchmark prompt for spec-decode-representative numbers, dry-run validation, saved runs and profiles, and reading live enums via `lmx context`.
---

# LocalMaxxing CLI

Use `lmx` when an agent needs to measure and submit local LLM performance or quality results to the LocalMaxxing public leaderboard. It covers benchmark throughput, TTFT, VRAM, eval scores, hardware metadata, saved runs, and API validation.

## Quickstart

Install `lmx` from a release tarball, or build from source:

```bash
go build -o lmx ./cmd/lmx
```

Authenticate with an API key (prefix `bhk_`) or environment variable:

```bash
lmx auth --key bhk_...
export LMX_API_KEY=bhk_...
```

Saved config lives under `~/.config/localmaxxing`. Generate hardware metadata before submitting:

```bash
lmx hardware --out hardware.json
```

## Decision tree

- Already-running OpenAI-compatible / vLLM / SGLang / Ollama endpoint: use a **remote** benchmark with `--mode remote --base-url ...`. This is the path where you control the prompt.
- Raw llama.cpp throughput on the host: use a **local** benchmark with `--mode local --model-path model.gguf`; it runs `llama-bench` with synthetic token counts.
- Quality / accuracy instead of speed: use evals; see `skill://localmaxxing-cli/references/evals.md`.
- Speed vs context depth: use a KV-cache sweep with `lmx kvcache run`.

## Canonical benchmark workflow

1. Run with `--dry-run` to write a measurement plan.
2. Run again without `--dry-run` to measure.
3. Validate with `lmx benchmark dry-run <file>` (authenticated API validation, no write).
4. Submit with `lmx benchmark submit <file>`.

Submissions require `hfId`, `hardware`, `engineName`, `quantization`, and `tokSOut` plus at least one secondary metric: `tokSPrefill`, `tokSTotal`, `ttftMs`, or `peakVramGb`. For remote endpoints, the `hardware` object must describe the server running the model, not the client running `lmx`. Benchmark submissions are rate-limited to 30 per rolling hour, or 300 for Pro users.

## Prompt control for spec decoding

Remote benchmarks send a real prompt. Set it with:

```bash
lmx benchmark run vllm --mode remote --base-url http://server:8000 --prompt "<text>"
```

The default prompt is:

> Explain why local inference benchmarks should report prompt prefill throughput, decode throughput, and time to first token.

Control output length with `--max-tokens` (default 256) and sampling with `--temperature` (default 0). Speculative-decoding acceptance depends on prompt and output distribution, so use a representative workload prompt and realistic `--max-tokens` for comparable spec-decode numbers. The local `llama-bench` path only accepts synthetic `--prompt-tokens` / `--output-tokens` counts, not semantic text, so it does not reflect spec-decode behavior on real content.

## Agent-friendly flags

- `--json-status`: emit JSON progress events on stderr.
- `--quiet`: suppress progress events and human status text.
- `--out <path>`: write output payloads/files.
- `--dry-run`: write plans or validate without submitting, depending on command.
- `lmx context --out ...`: fetch live enum values and schemas from the site.
- `lmx <command> --help`: show examples and relevant flags for a command.

## Reference docs

- `skill://localmaxxing-cli/references/benchmarks.md`: benchmark modes, flags, saved runs, profiles, and KV-cache sweeps.
- `skill://localmaxxing-cli/references/evals.md`: suite management, eval runs, lm-eval, LM-judge, shards, Terminal-Bench, and storage.
- `skill://localmaxxing-cli/references/hardware-and-setups.md`: hardware detection/templates and saved setup pulls.
- `skill://localmaxxing-cli/references/reference.md`: command list, flag glossary, and troubleshooting.
