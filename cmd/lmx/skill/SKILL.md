---
name: localmaxxing-cli
description: Use the LocalMaxxing CLI (the `lmx` binary) to calculate model memory fit and decode upper bounds, run local and remote LLM inference speed tests, run quality benchmarks, create standardized GFM model reports, generate hardware metadata, and submit results to localmaxxing.com. Covers agent-readable output, authentication, report authoring/editing/images/publication, local vs remote speed-test modes, dry-run validation, saved runs, profiles, and live schemas via `lmx context`.
---

# LocalMaxxing CLI

Use `lmx` when an agent needs to estimate model fit, measure and submit local LLM performance, run quality evaluations, or author a model report that a user can continue editing in the web studio. Reports use the same standardized GitHub Flavored Markdown document in CLI, API, and web workflows.

## Quickstart

Install `lmx` from a release tarball, or build from source:

```bash
go build -o lmx ./cmd/lmx
```

Authenticate with an API key (prefix `bhk_`). Prefer the environment so the
secret does not enter shell history or process arguments:

```bash
export LMX_API_KEY=bhk_...
printf '%s\n' "$LMX_API_KEY" | lmx auth --key-stdin  # persist without argv exposure
```

Only use `--api-key` or `lmx auth --key ...` when command-line secret exposure
is acceptable. Never include an API key in agent traces, generated plans, or logs.

Saved config lives under `~/.config/localmaxxing`. Generate hardware metadata before submitting:

```bash
lmx hardware --out hardware.json
```

## Decision tree

- Already-running OpenAI-compatible / vLLM / SGLang / Ollama endpoint: use a **remote speed test** with `--mode remote --base-url ...`. This is the path where you control the prompt.
- Raw llama.cpp throughput on the host: use a **local speed test** with `--mode local --model-path model.gguf`; it runs `llama-bench` with synthetic token counts.
- Quality / accuracy instead of speed: use evals; see `skill://localmaxxing-cli/references/evals.md`.
- Long-form findings or analysis that a user should edit in the web report studio: use `lmx report`; see `skill://localmaxxing-cli/references/reports.md`.
- Speed vs context depth: use a KV-cache sweep with `lmx kvcache run`.
- Planning a model/hardware configuration without running inference: use `lmx calculate decode --json`.

## Agent-run decode calculator

Use the local deterministic calculator before downloading a model or launching a server. It requires no authentication or network request. Always request `--json` when another agent will consume the result:

```bash
lmx calculate decode \
  --capacity-gb 24 \
  --bandwidth-gbps 936.2 \
  --total-params-b 32 \
  --weight-bits 4.5 \
  --allocated-context 32768 \
  --read-context 16000 \
  --min-tok-s 20 \
  --json
```

For speculative decoding, include the draft acceptance and cost instead of treating draft work as free:

```bash
lmx calculate decode \
  --decoding speculative \
  --draft-tokens 4 \
  --acceptance-percent 80 \
  --draft-resident-gb 1.8 \
  --draft-traffic-gb 1.8 \
  --json
```

Read `result.state` first. `usable` includes `result.optimum` with aggregate and per-session throughput. `resident-fail`, `session-fail`, and `floor-fail` include a structured `constraint` with the blocking resource, measured shortfall, and suggested flag changes. The estimate is a memory-side roofline, not a measured benchmark; use `speed-test` to verify real performance.

## Canonical speed-test workflow

1. Run with `--dry-run --out plan.json` to write a measurement plan without adding managed run history.
2. Run again without `--dry-run` to measure.
3. Validate with `lmx speed-test dry-run <file>` (authenticated API validation, no write).
4. Submit with `lmx speed-test submit <file>`.

Submissions require `hfId`, `hardware`, `engineName`, `quantization`, and `tokSOut` plus at least one secondary metric: `tokSPrefill`, `tokSTotal`, `ttftMs`, or `peakVramGb`. Optionally include `gpuPowerWatts`: an array of measured watts, one entry per physical GPU (heterogeneous rigs list each card, e.g. `[285.5,310.2]`; 1–64 entries, each positive and ≤10000). Pass it to `speed-test run` as `--gpu-power-watts 285.5,310.2`; the server computes and returns `totalPowerWatts`. For remote endpoints, the hardware must describe the server running the endpoint, not the machine running `lmx`; run `lmx hardware --out hardware.json` on the server or provide an equivalent reviewed file, then pass `--hardware hardware.json`. Remote mode never auto-detects client hardware. If a completed remote run lacks hardware, repair the saved JSON without rerunning: `lmx speed-test add-hardware <run.json> --hardware hardware.json`.

`quantization` is free-form; prefer labels from `lmx context` → `commonSchemas.benchmarkFields.quantization.commonValues` when possible. The common list includes GGUF (`Q4_K_M`, `IQ4_XS`), NVIDIA (`NVFP4`), Unsloth dynamic (`Unsloth-Dynamic-Q4_K_M`), bitsandbytes (`bnb-nf4`), AWQ/GPTQ/EXL2/FP8 variants, and engine-specific strings.

Hardware purchase metadata is optional and personal to the submitting user/setup. Use `--hardware-cost` to record one purchase record per canonical hardware component. Component names must come from `lmx context` → `hardwareOptions.hardwareCostComponentNames` so costs can be filtered and linked.

```bash
lmx context --out localmaxxing-agent-context.json

lmx speed-test run llama.cpp \
  --hardware-cost "NVIDIA GeForce RTX 3090|used|2021|700|USD;NVIDIA GeForce RTX 4090|new|2024|1599|USD"

lmx speed-test run llama.cpp \
  --hardware-cost '[{"component":"NVIDIA GeForce RTX 3090","condition":"USED","yearPurchased":2021,"price":700,"currency":"USD"}]'
```

Compact format is `component|condition|year|price|currency` separated by semicolons; `year` may be empty. JSON format is an array of `{ component, condition: "NEW"|"USED", yearPurchased?, price, currency }`. Currencies are not converted or summed across codes. The server validates and canonicalizes component names during dry-run/submit.

## Prompt control for spec decoding

Remote speed tests send a real prompt. Set it with:

```bash
lmx speed-test run vllm --mode remote --base-url http://server:8000 --prompt "<text>"
```

The default prompt is:

> Explain why local inference speed tests should report prompt prefill throughput, decode throughput, and time to first token.

Control output length with `--max-tokens` (default 256) and sampling with `--temperature` (default 0). Speculative-decoding acceptance depends on prompt and output distribution, so use a representative workload prompt and realistic `--max-tokens` for comparable spec-decode numbers. The local `llama-bench` path only accepts synthetic `--prompt-tokens` / `--output-tokens` counts, not semantic text, so it does not reflect spec-decode behavior on real content.

Describe the speculative configuration using engine-neutral submission fields:

```bash
lmx speed-test run sglang \
  --spec-method dflash2 \
  --spec-draft-model z-lab/Qwen3.8-27B-DFlash2 \
  --spec-num-tokens 8 \
  --spec-draft-window-size 16
```

Available fields are `--spec-decoding`, `--spec-method`, `--spec-draft-model`, `--spec-num-tokens`, `--spec-draft-tp`, and `--spec-draft-window-size`. `dflash2`/`draft-dflash2`, `dflash`/`draft-dflash`, and `mtp`/`draft-mtp` are canonicalized to `DFlash2`, `DFlash`, and `MTP`. A DFlash method paired with a draft-model name containing `DFlash2` is recorded as `DFlash2`.

## Agent-friendly flags

- `--json`: request a final JSON result on stdout.
- `--json-status`: emit JSONL progress and errors on stderr.
- `--quiet`: suppress progress events and human status text, but not a requested final JSON result.
- `--out <path>`: write output payloads/files.
- `--dry-run`: plan or validate without submitting. Speed-test dry-runs do not add managed run history unless `--save-run` is passed.
- `lmx commands --json`: inspect the versioned machine-readable command schema.
- `lmx context list`: list live context sections.
- `lmx context get <dotted.path> --compact`: fetch only the needed live enum or schema.
- `lmx <command> --help`: show examples and relevant flags for a command.
- Long Terminal-Bench jobs: preflight first, then launch with a unique `--run-dir ... --detach`; poll with `lmx eval terminal status <run-dir> --json`, consume `logs <run-dir> [--follow]` as JSONL, and cancel with `cancel <run-dir>`. Resume by repeating the exact run identity with the same directory and `--resume auto`; see `references/evals.md`.

## Reference docs

- `skill://localmaxxing-cli/references/reports.md`: standardized GFM authoring, API-compatible editing, inline images, and publication lifecycle.
- `skill://localmaxxing-cli/references/speed-tests.md`: speed-test modes, flags, saved runs, profiles, and KV-cache sweeps.
- `skill://localmaxxing-cli/references/evals.md`: suite management, eval runs, lm-eval, LM-judge, shards, Terminal-Bench, and storage.
- `skill://localmaxxing-cli/references/hardware-and-setups.md`: hardware detection/templates and saved setup pulls.
- `skill://localmaxxing-cli/references/reference.md`: command list, flag glossary, and troubleshooting.
