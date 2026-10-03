# Speed tests

## Remote endpoint speed tests

Use remote mode for an already-running endpoint. FreeToken, NInfer, vLLM, and SGLang use an OpenAI-compatible API (`POST /v1/chat/completions`) and stream by default. Ollama uses its native `/api/generate` API.

```bash
lmx speed-test run vllm \
  --mode remote \
  --base-url http://server:8000 \
  --hf-id Qwen/Qwen3-8B \
  --quantization fp16 \
  --hardware hardware.json

lmx speed-test run sglang --mode remote --base-url http://server:30000 --hf-id Qwen/Qwen3-8B --quantization fp16
lmx speed-test run ollama --mode remote --base-url http://server:11434 --served-model qwen3:8b --hf-id Qwen/Qwen3-8B --quantization Q4_K_M
lmx speed-test run freetoken --mode remote --base-url http://server:8000 --hf-id Qwen/Qwen3-8B --quantization fp16 --engine-version 0.3.1 --engine-commit abcdef123456
lmx speed-test run ninfer --mode remote --base-url http://server:8000 --hf-id Qwen/Qwen3-8B --quantization fp16
```

Defaults: 1 untimed warmup and 3 timed iterations. The submitted summary reports the median, and per-iteration details appear in `samples` / `sampleStats`. Tune with `--warmup` and `--iterations`; `--warmup 0 --iterations 1` is a single shot.

Remote decode throughput is measured over the inter-token window from first streamed token to last streamed token. For OpenAI-compatible endpoints, each warmup and timed request receives a unique leading nonce so prefix caches cannot turn a cold-prefill measurement into a cache-hit measurement. `tokSPrefill` is estimated independently for each request from the endpoint's `usage.prompt_tokens` divided by TTFT. When usage is unavailable, the CLI falls back to the declared or locally estimated prompt count and marks the source accordingly.

Important remote flags:

- `--base-url <url>`: model server base URL; accepts host or host plus `/v1`.
- `--served-model <name>`: model name served by the endpoint.
- `--model-api-key <key>`: bearer token for the model endpoint.
- `--backend <name>`: accelerator backend recorded for the run, such as `cuda`, `rocm`, or `tt-metal`. Remote runs do not assume CUDA when omitted.
- `--prompt <text>` / `--prompt-file <path>`: semantic prompt sent to the endpoint; use `--prompt-file -` for stdin. These inputs are mutually exclusive.
- `--prompt-tokens <n>`: target prompt size when prompt text is omitted; the CLI synthesizes deterministic text of approximately that many tokens. Endpoint usage remains authoritative and a mismatch is recorded in `warnings`.
- `--kv-cache-tokens <n>`: tokens already held in KV cache before the test. Omit for a fresh request; the CLI submits `0`. `--prefill-tokens` and `--depth` remain aliases.
- `--max-tokens <n>`: `max_tokens` sent to the endpoint; remote speed-test default is 256. Larger values give a steadier `tokSOut` at the cost of a longer run; `--output-tokens` is not used on the remote path.
- `--temperature <f>`: sampling temperature; default is 0.
- `--no-stream`: disable streaming.
- `--endpoint-timeout-seconds <n>`: remote endpoint timeout; default is 600.
- `--engine-version <value>`: engine release/version used for the run.
- `--engine-repository <url>`: source repository used for the run. FreeToken and NInfer default to their canonical GitHub repositories; override this for a fork or mirror.
- `--engine-build <value>`: build or artifact identifier used for the run.
- `--engine-commit <hash>`: 7–64 character hexadecimal Git commit used for the run.

Optional cost/power metadata for submissions:

- `--gpu-power-watts 285.5,310.2`: measured watts per physical GPU; heterogeneous rigs list each card separately.
- `--hardware-cost "NVIDIA GeForce RTX 3090|used|2021|700|USD;NVIDIA GeForce RTX 4090|new|2024|1599|USD"`: one purchase record per canonical hardware component, compact form `component|condition|year|price|currency`. `year` may be empty. Component must come from `lmx context` → `hardwareOptions.hardwareCostComponentNames`.
- `--hardware-cost '[{"component":"NVIDIA GeForce RTX 3090","condition":"USED","yearPurchased":2021,"price":700,"currency":"USD"}]'`: JSON array form. `condition` is `NEW` or `USED`; currency is a 3-letter ISO code. The server stores this as run-level personal purchase metadata and does not sum across currencies.

## Local llama.cpp speed tests

Use local mode when running `llama-bench` on the host that owns the model/hardware. With `--model-path`, `lmx` generates:

```bash
llama-bench -m <model-path> -p <prompt-tokens|512> -n <output-tokens|128> [-d <kv-cache-tokens>]
```

Useful flags:

- `--threads <n>`
- `--gpu-layers <n>`
- `--prompt-tokens <n>`: llama-bench `-p`; default 512
- `--output-tokens <n>` (alias `--output-len`): llama-bench `-n`, the number of generated tokens the decode rate is measured over; default 128. Raise it (e.g. 512) when short runs are noisy or dominated by warm-up.
- `--kv-cache-tokens <n>` / `--prefill-tokens <n>` / `--depth <n>`: llama-bench `-d`; submitted as `prefillTokens`
- `--batch-size <n>`
- `--micro-batch-size <n>`
- `--repetitions <n>`
- `--cache-type-k <type>` / `--cache-type-v <type>`
- `--flash-attn` / `--no-flash-attn`
- `--benchmark-format <fmt>`

`promptTokens` is the fresh prompt size (`-p`/`n_prompt`). `prefillTokens` is
the cached depth present before generation (`-d`/`n_depth`), supplied most
clearly as `--kv-cache-tokens`. The CLI submits `0` when a fresh run omits it.

Use `--command "..."` for custom local engines or exact commands.

## Local vLLM and SGLang generated benches

For local vLLM/SGLang speed tests, `lmx` can generate `bench` commands. Control request shapes with `--input-len` (prompt tokens), `--output-len` / `--output-tokens` (generated tokens per request; default 128, passed as `--random-output-len`), and `--num-prompts`. Set `--bench-kind serve`, `--bench-kind throughput`, or `--bench-kind latency` depending on the engine and speed-test path.

## Output-token flag summary

| Path | Flag | Default | Effect |
|---|---|---|---|
| Remote endpoint (`--base-url`) | `--max-tokens` | 256 | `max_tokens` in the chat-completions request |
| Local llama.cpp (`--model-path`) | `--output-tokens` / `--output-len` | 128 | `llama-bench -n` |
| Local vLLM / SGLang bench | `--output-len` / `--output-tokens` | 128 | `--random-output-len` per request |
| KV-cache / context sweeps (`--levels`) | `--output-tokens` / `--output-len` / `--max-tokens` (first set wins) | 128 | Completion cap at every sweep level |
| Manual submit (`--tok-s-out …`) | `--output-len` / `--output-tokens` | — | Recorded as `outputLen` / `outputTokens` metadata only |

## Submit an existing completed run

`speed-test submit` accepts a LocalMaxxing speed-test JSON document; it does not require the measurement to have been launched by `lmx`. A minimal discrete-GPU payload is:

```json
{
  "hfId": "Qwen/Qwen3-8B",
  "modelRevision": "main",
  "engineName": "vllm",
  "quantization": "fp16",
  "backend": "cuda",
  "tokSOut": 123.4,
  "ttftMs": 86.2,
  "promptTokens": 512,
  "outputTokens": 128,
  "hardware": {
    "hwClass": "DISCRETE_GPU",
    "gpuName": "Intel Arc Pro B65",
    "gpuCount": 1,
    "vramGb": 32,
    "cpu": "AMD Ryzen 9 9950X",
    "ramGb": 64,
    "os": "Linux"
  },
  "engineFlags": {
    "commandSnippet": "vllm serve Qwen/Qwen3-8B",
    "tensorParallel": 1,
    "concurrency": 1,
    "temperature": 0
  },
  "notes": "Imported from an existing benchmark run."
}
```

Required fields are `hfId`, `hardware`, `engineName`, `quantization`, and `tokSOut`, plus at least one of `tokSPrefill`, `tokSTotal`, `ttftMs`, or `peakVramGb`. Use canonical hardware names from the live context; never copy the example GPU when it does not match the measured host.

Validate in increasing order of authority:

```bash
lmx speed-test validate-local run.json
lmx speed-test dry-run run.json
lmx speed-test submit run.json
```

`validate-local` checks the CLI-known shape without network writes. Authenticated `dry-run` applies the current server schema and hardware allowlist without creating a run. Only `submit` persists it.

For an older LocalMaxxing run file:

```bash
lmx speed-test fixup run.json
lmx speed-test add-hardware run.json --hardware hardware.json
```

`fixup` normalizes known LocalMaxxing fields; it is not a converter for arbitrary vLLM, SGLang, or llama.cpp output. Map external results into the upload shape explicitly and preserve the original command in `engineFlags.commandSnippet`.

### Verified-run evidence

Ordinary submissions may be accepted without a verified-run badge. For verification, preserve the evidence produced by the measurement instead of inventing or reconstructing it:

- Top level: `promptSha256`, `promptSample`, `outputSha256`, `outputSample`, `engineTimingsRaw`, `engineVersion`, `engineRepository`, `engineBuild`, `engineCommit`, `backend`, and `peakVramGb`.
- `engineFlags`: `specDraftTokens`, `specAcceptedTokens`, `specAcceptanceRate`, `specMeanAcceptedLength`, and `temperature` when applicable.

Explicit engine-flag zero and false values survive `dry-run` and `submit`, including zero accepted speculative tokens, sampling settings, and disabled caching/batching options. Absent or null flags remain omitted; unknown/internal keys are not forwarded. Sampling belongs in `engineFlags`: older top-level `temperature` and `topP` values are moved there only when that nested key is absent, so explicit nested settings take precedence.

The CLI preserves supplied evidence; it does not create missing measurements or guarantee a badge. Read `verification.verified` and `verification.issues` in the dry-run response, and `verifiedRun` and `verificationIssues` in the submission response. A successful submission is not necessarily a verified run.

The live contract is authoritative:

```bash
lmx context list
lmx context --out localmaxxing-agent-context.json
curl -fsS https://www.localmaxxing.com/api/openapi.json -o localmaxxing-openapi.json
```

## Saved runs and profiles

Saved run commands:

```bash
lmx speed-test runs list
lmx speed-test runs show runs/Model/run.json --format table
lmx speed-test runs edit runs/Model/run.json --set-json '{"tokSOut":120}'
lmx speed-test runs rerun runs/Model/run.json --dry-run
lmx speed-test runs submit runs/Model/run.json --api-key bhk_...
lmx speed-test runs delete runs/Model/run.json --yes
lmx speed-test runs stats --group-by quantization --metric tokSOut
lmx speed-test runs compare --by hardware --model Qwen/Qwen3-8B --format table
lmx speed-test runs compare runs/base.json runs/candidate.json --metrics tokSOut,ttftMs --format table
lmx speed-test runs export --format csv --out runs.csv
```

Authenticated remote submission commands:

```bash
lmx speed-test submissions list --limit 20 --offset 0
lmx speed-test submissions edit <runId> --set-json '{"prefillTokens":4096,"notes":"corrected"}'
lmx speed-test submissions edit <runId> --set tensorParallel=2 --set gpuMemUtil=0.94 --set kvCacheDtype=bfloat16 --set prefixCaching=true
lmx speed-test submissions edit <runId> --set-json '{"engineFlags":{"tensorParallel":2,"prefixCaching":false,"commandSnippet":"vllm serve org/model --tensor-parallel-size 2"}}'
```

The list includes pending, approved, and rejected runs owned by the API key.
Remote edits are limited by the API's owner edit window and cooldown.
Choose one edit request; do not run every example against the same run in a row.
Repeated `--set` applies every assignment; the last assignment to a field wins.
Precedence is `--patch` < `--set-json` < ordered `--set`, independent of option
position. Remote JSON/file edits accept flat fields or a nested `engineFlags`
object and send flat API fields. Explicit false, zero and null survive; value
and nullability validation remains server-side. Unknown fields, malformed
assignments, empty/non-object `engineFlags` and flat/nested collisions in a
single input fail locally without consuming an edit. Use flat assignment names
(`--set tensorParallel=2`, not `--set engineFlags.tensorParallel=2`).
Supply the real command snippet explicitly; endpoint metadata is not proof of
the server's launch flags. Saved local run edits also retain repeated `--set`.


Profiles and repair helpers:

```bash
lmx profile save my-4090 --mode local --hf-id Qwen/Qwen3-8B --quantization Q4_K_M --hardware hardware.json
lmx speed-test add-hardware runs/Model/run.json --hardware hardware.json
lmx speed-test fixup runs/Model/run.json
```

## KV-cache / context sweeps

Use context sweeps to measure speed across context depth:

```bash
lmx kvcache run llama.cpp --hf-id Qwen/Qwen3-8B --model-path model.gguf --levels 10000,20000,30000,40000
lmx kvcache run vllm --mode remote --base-url http://server:8000 --hf-id Qwen/Qwen3-8B --levels 10000,20000,30000,40000
```

Remote sweeps pre-warm the prefix, use a deterministic varied-word filler by default, and finish with `--probe-prompt`. Use `--filler-token` to override the filler text.

Cache evidence comes only from the timed response: `usage.prompt_tokens_details.cached_tokens` and llama.cpp `timings.cache_n` / `timings.prompt_n`, checked against `usage.prompt_tokens`. `/slots` and warm-up counters do not establish reuse by the timed request.

`cacheReuse.status` is `retained` when the entire timed prompt is cached, `partial` for positive reuse below the full prompt length, `not_retained` for an explicit zero, and `unknown` for missing/invalid/conflicting counters. Inspect `cachedTokens`, `uncachedTokens`, and the raw usage/timing evidence; nominal context depth does not establish cache coverage.

Prefill throughput uses verified uncached tokens divided by client-observed TTFT, with `tokSPrefillSource` persisted as `estimated_from_ttft_uncached`. It remains an estimate including HTTP and first-token overhead. Unknown reuse carries a warning and no derived prefill rate; fully cached prompts also have no prefill rate.
