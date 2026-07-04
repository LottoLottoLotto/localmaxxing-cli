# Benchmarks

## Remote endpoint benchmarks

Use remote mode for an already-running endpoint. vLLM and SGLang use an OpenAI-compatible API (`POST /v1/chat/completions`) and stream by default. Ollama uses its native `/api/generate` API.

```bash
lmx benchmark run vllm \
  --mode remote \
  --base-url http://server:8000 \
  --hf-id Qwen/Qwen3-8B \
  --quantization fp16 \
  --hardware hardware.json

lmx benchmark run sglang --mode remote --base-url http://server:30000 --hf-id Qwen/Qwen3-8B --quantization fp16
lmx benchmark run ollama --mode remote --base-url http://server:11434 --served-model qwen3:8b --hf-id Qwen/Qwen3-8B --quantization Q4_K_M
```

Defaults: 1 untimed warmup and 3 timed iterations. The submitted summary reports the median, and per-iteration details appear in `samples` / `sampleStats`. Tune with `--warmup` and `--iterations`; `--warmup 0 --iterations 1` is a single shot.

Remote decode throughput is measured over the inter-token window from first streamed token to last streamed token. `tokSPrefill` is estimated from prompt tokens divided by TTFT.

Important remote flags:

- `--base-url <url>`: model server base URL; accepts host or host plus `/v1`.
- `--served-model <name>`: model name served by the endpoint.
- `--model-api-key <key>`: bearer token for the model endpoint.
- `--prompt <text>`: semantic prompt sent to the endpoint.
- `--max-tokens <n>`: max generated tokens; remote benchmark default is 256.
- `--temperature <f>`: sampling temperature; default is 0.
- `--no-stream`: disable streaming.
- `--endpoint-timeout-seconds <n>`: remote endpoint timeout; default is 600.

Optional cost/power metadata for submissions:

- `--gpu-power-watts 285.5,310.2`: measured watts per physical GPU; heterogeneous rigs list each card separately.
- `--hardware-cost "NVIDIA GeForce RTX 3090|used|2021|700|USD;NVIDIA GeForce RTX 4090|new|2024|1599|USD"`: one purchase record per canonical hardware component, compact form `component|condition|year|price|currency`. `year` may be empty. Component must come from `lmx context` → `hardwareOptions.hardwareCostComponentNames`.
- `--hardware-cost '[{"component":"NVIDIA GeForce RTX 3090","condition":"USED","yearPurchased":2021,"price":700,"currency":"USD"}]'`: JSON array form. `condition` is `NEW` or `USED`; currency is a 3-letter ISO code. The server stores this as run-level personal purchase metadata and does not sum across currencies.

## Local llama.cpp benchmarks

Use local mode when running `llama-bench` on the host that owns the model/hardware. With `--model-path`, `lmx` generates:

```bash
llama-bench -m <model-path> -p <prompt-tokens|512> -n <output-tokens|128>
```

Useful flags:

- `--threads <n>`
- `--gpu-layers <n>`
- `--depth <n>`
- `--batch-size <n>`
- `--micro-batch-size <n>`
- `--repetitions <n>`
- `--cache-type-k <type>` / `--cache-type-v <type>`
- `--flash-attn` / `--no-flash-attn`
- `--benchmark-format <fmt>`

Use `--command "..."` for custom local engines or exact commands.

## Local vLLM and SGLang generated benches

For local vLLM/SGLang benchmarks, `lmx` can generate `bench` commands. Control request shapes with `--input-len`, `--output-len`, and `--num-prompts`. Set `--bench-kind serve`, `--bench-kind throughput`, or `--bench-kind latency` depending on the engine and benchmark path.

## Saved runs and profiles

Saved run commands:

```bash
lmx benchmark runs list
lmx benchmark runs show runs/Model/run.json
lmx benchmark runs edit runs/Model/run.json --set-json '{"tokSOut":120}'
lmx benchmark runs rerun runs/Model/run.json --dry-run
lmx benchmark runs submit runs/Model/run.json --api-key bhk_...
lmx benchmark runs delete runs/Model/run.json --yes
lmx benchmark runs stats --group-by quantization --metric tokSOut
lmx benchmark runs compare --by hardware --model Qwen/Qwen3-8B
lmx benchmark runs compare runs/base.json runs/candidate.json --metrics tokSOut,ttftMs
lmx benchmark runs export --format csv --out runs.csv
```

Profiles and repair helpers:

```bash
lmx profile save my-4090 --mode local --hf-id Qwen/Qwen3-8B --quantization Q4_K_M --hardware hardware.json
lmx benchmark add-hardware runs/Model/run.json --hardware hardware.json
lmx benchmark fixup runs/Model/run.json
```

## KV-cache / context sweeps

Use context sweeps to measure speed across context depth:

```bash
lmx kvcache run llama.cpp --hf-id Qwen/Qwen3-8B --model-path model.gguf --levels 10000,20000,30000,40000
lmx kvcache run vllm --mode remote --base-url http://server:8000 --hf-id Qwen/Qwen3-8B --levels 10000,20000,30000,40000
```

Remote sweeps pre-warm the prefix, inspect `/slots` when available, use a deterministic varied-word filler by default, and finish with `--probe-prompt`. Use `--filler-token` to override the filler text.
