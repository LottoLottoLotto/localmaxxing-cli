# CLI reference

Raw HTTP API docs are available from the site through `GET /api/agent-context` and `GET /api/openapi.json`.

## Top-level commands

- `lmx context` / `lmx agent-context`: fetch live enum values and schemas.
- `lmx auth`: manage LocalMaxxing API authentication.
- `lmx hardware`: detect, validate, or template hardware metadata.
- `lmx setups`: list and pull saved account setups.
- `lmx model`: search or resolve HuggingFace model IDs.
- `lmx profile`: save and manage reusable CLI defaults.
- `lmx engines` / `lmx engine`: list engine helpers.
- `lmx server`: build or run local model server commands.
- `lmx endpoint`: discover OpenAI-compatible endpoints.
- `lmx kvcache` / `lmx kv-cache` / `lmx context-sweep`: run KV-cache/context sweeps.
- `lmx benchmark` / `lmx bench`: create, manage, validate, and submit benchmarks.
- `lmx eval`: discover, run, and submit evaluation suites.
- `lmx skill`: print or install the bundled agent skill.

## Flag glossary

Global/auth/output:

- `--api-url <url>`: LocalMaxxing origin; default `https://www.localmaxxing.com`.
- `--api-key <key>`: API key; defaults to `LMX_API_KEY`, then saved config.
- `--no-browser`: do not open the device-login browser automatically.
- `--profile <name>`: load saved defaults from `lmx profile save`.
- `--json-status`: emit progress events as JSON lines on stderr.
- `--quiet`: suppress progress events.
- `--out <path>`: write computed payload/result JSON, or write `SKILL.md` for `lmx skill print`. On `eval terminal run`, this is a monolithic completed-run artifact accepted directly by deferred `eval terminal submit`.
- `--dir <dir>`: target skills directory for `lmx skill install`; default `.claude/skills`.
- `--submit`: upload run to LocalMaxxing. `eval terminal run` still executes when this is omitted; it only keeps the result local.
- `--dry-run`: for benchmark run, write a measurement plan; for deferred `eval terminal submit`, validate/package the completed artifact entirely offline. It is not a no-execution mode for `eval terminal run`.

Model/eval:

- `--model <hfId>`: HuggingFace model ID. Terminal run rejects it when it conflicts with a source repo unambiguously verified from the live loaded filename.
- `--backend <name>`: lm-eval backend name; default `hf`.
- `--model-args <args>`: lm-eval `--model_args` value.
- `--num-fewshot <n>`: lm-eval `--num_fewshot` override.
- `--lm-eval-bin <path>`: lm-eval executable; default `lm_eval`.
- `--results <path>`: existing lm-eval output JSON for run upload.
- `--kind <kind>`: storage upload kind, usually `artifact` or `dataset`.
- `--format <format>`: storage file format, e.g. `json`, `jsonl`, `parquet`, `zip`.
- `--item-count <n>`: optional record/sample count for storage metadata.
- `--limit <n>`: optional search/list result limit.

Eval shards and Terminal-Bench:

- `--questions <n>`: eval-shard questions to run.
- `--shard <index>`: pin a specific eval-shard index.
- `--missing-only`: with `eval shard --submit`, run the first missing aggregate shard instead of duplicating a covered default shard.
- `--all-missing`: with `eval shard --submit`, submit every currently missing aggregate shard.
- `--rerun` / `--force`: allow submitting a shard that already has APPROVED aggregate coverage.
- `--answer-extraction <m>`: `none`, `final_answer`, `last_number`, or `regex`.
- `--answer-regex <re>`: regex used with `--answer-extraction regex`.
- `--prompt-template <t>`: eval-shard prompt template using `{{input}}` and `{{choices}}`.
- `--concurrency <n>`: eval-shard parallel requests; default 1.
- `--artifact-limit <n>`: shard traces to submit; default 0 means all.
- `--scoring <mode>`: `exact_match`, `loglikelihood`, `llama_cpp_loglikelihood`, `code_execution`, or `cruxeval_execution`.
- `--temperature <f>`: sampling temperature; eval-shard default 0.
- `--top-p <f>`: sampling top_p; default 1.
- `--quant-format <label>`: quantization container format for eval-shard submit.
- `--model-revision <rev>`: model revision for eval-shard submit; default `main`.
- `--task-dir <dir>`: Terminal eval bundle directory.
- `--dataset <slug>`: Terminal eval dataset slug.
- `--hf-id <hfId>`: canonical HuggingFace model ID for deferred terminal submit.
- `--endpoint-file <path>`: JSON written by `lmx endpoint discover --out endpoint.json`; Terminal run requires exactly one `ok:true` endpoint, uses the file only to select its URL, and re-probes live identity metadata. Mutually exclusive with `--base-url`.
- `--shard-index <n>`: explicit shard for an isolated deferred artifact or legacy checkpoint directory; required for noncanonical datasets. Omit only for a complete canonical Terminal-Bench 2.1 checkpoint, which is validated and partitioned into 10 submissions.
- `--max-turns <n>`: Terminal eval agent turn cap.
- `--max-tokens <n>`: Terminal model completion cap; default 16,384 and 8,192 on retry. An explicit value applies to both attempts.
- `--agent-timeout <sec>`: Terminal eval whole-agent timeout.
- `--agent <name>`: built-in terminal agent backend; `terminus-2` uses the embedded Harbor adapter.
- `--agent-cmd <cmd>`: external agent command with `LMX_TERMINAL_*` env vars.
- `--agent-execution <m>`: `host`, `container`, or `routed-shell`; default `host`.
- `--agent-name <name>`: external terminal agent label; default `external-agent`.
- `--container-base-url <url>`: base URL visible from task containers.
- `--command-timeout <sec>`: terminal per-shell-command timeout; default is the remaining task budget, up to 4h.
- `--endpoint-timeout-seconds <n>`: Terminal model request timeout; default 600 seconds. The first attempt reserves part of the remaining task budget for retry.
- `--trace-dir <dir>`: save per-task `transcript.md`, `verifier.txt`, `prompt.txt`, `instruction.txt`, `result.json` with `wallTimeMs`/`tokenUsage`, and external agent logs.
- `--cleanup-images`: remove locally built terminal task images after each task.
- `--shell-mode <mode>`: `persistent` or `stateless`; default `persistent`.
- `--oracle`: run terminal task `solution/solve.sh` instead of the model agent.
- Terminal `run --out` persists dataset/shard, canonical `hfId` and model-resolution evidence, model revision, quantization, hardware, harness configuration, timing, and token usage. Deferred submit reuses these authoritative values and rejects conflicting flags rather than rerunning work.
- `--llama-scorer <path>`: local helper for `llama_cpp_loglikelihood` scoring.
- `--sandbox-image <name>`: code sandbox image; default `lmx-sandbox`.
- `--sandbox-runtime <bin>`: container runtime for code execution; default `docker`.
- `--sandbox-cmd <cmd>`: override sandbox launcher entirely.
- `--sandbox-memory <size>`: sandbox memory cap; default `2g`.
- `--sandbox-cpus <n>`: sandbox CPU cap; default 2.
- `--n-samples <n>`: samples per question for code evals.
- `--k <n>`: pass@k value over `--n-samples`; default 1.
- `--few-shot <n>`: few-shot examples for MBPP-family code evals.

Remote benchmark / endpoint:

- `--base-url <url>`: explicit OpenAI-compatible endpoint; accepts host or host plus `/v1`. Mutually exclusive with Terminal `--endpoint-file`. With neither selector, uncredentialed Terminal runs auto-probe supported localhost endpoints and require one unambiguous match.
- `--mode <mode>`: `remote` endpoint or `local` host command.
- `--served-model <name>`: exact live model selector for Terminal run; it must match `/v1/models`.
- `--model-api-key <key>`: optional endpoint bearer token. For Terminal run it requires explicit `--base-url` or trusted `--endpoint-file`; credentials are never sent during broad auto-probing.
- `--prompt <text>`: prompt for remote endpoint benchmark.
- `--max-tokens <n>`: max generated tokens; remote benchmark default 256.
- `--endpoint-timeout-seconds <n>`: remote endpoint timeout; default 600.
- `--warmup <n>`: untimed warmup requests; default 1.
- `--iterations <n>`: timed remote iterations; median is reported; default 3.
- `--no-stream`: disable streaming for remote endpoint benchmark.
- `--include-server-metadata`: probe optional endpoint `/props` and `/hardware` during discover.

Local benchmark / server:

- `--command <cmd>`: local benchmark command, e.g. `llama-bench`.
- `--command-timeout-seconds <n>`: local benchmark command timeout; default unlimited.
- `--host <addr>`: local model server host.
- `--port <n>`: local model server port.
- `--model-path <path>`: llama.cpp model artifact path. Terminal run reconciles an explicit value with the live loaded path and rejects conflicts. Canonical `hfId` auto-resolution is accepted only when the exact loaded filename is verified in one unambiguous HuggingFace source repo.
- `--depth <n>`: llama-bench `-d` depth; KV sweeps use `--levels`.
- `--batch-size <n>`: llama-bench batch size.
- `--micro-batch-size <n>`: llama-bench micro-batch size.
- `--repetitions <n>`: llama-bench repetitions.
- `--benchmark-format <fmt>`: llama-bench output format.
- `--flash-attn`: llama-bench `-fa 1`; use `--no-flash-attn` for `-fa 0`.
- `--cache-type-k <type>`: llama-bench KV cache K type.
- `--cache-type-v <type>`: llama-bench KV cache V type.
- `--server-bin <path>`: server executable override.
- `--bench-kind <kind>`: vLLM benchmark kind: `serve`, `throughput`, or `latency`.
- `--benchmark-output <p>`: engine benchmark JSON output path.
- `--benchmark-bin <path>`: benchmark executable; default `vllm` for vLLM.
- `--python-bin <path>`: Python executable for SGLang commands.
- `--input-len <n>`: prompt/input tokens for built-in benchmark commands.
- `--output-len <n>`: generated/output tokens for built-in benchmark commands.
- `--num-prompts <n>`: number of prompts for vLLM serve/throughput benchmarks.

KV-cache and saved runs:

- `--levels <list>`: KV-cache/context sweep levels, e.g. `10000,20000,30000`.
- `--command-template <cmd>`: local sweep command template using `{input}` and `{output}`.
- `--probe-prompt <text>`: final remote prompt after loading retained context.
- `--filler-token <text>`: repeated token used for remote context filler.
- `--kv-cache-dtype <dtype>`: vLLM KV cache dtype for local latency sweeps.
- `--enable-prefix-caching`: enable vLLM prefix caching.
- `--runs-dir <dir>`: saved benchmark runs directory; default `runs`.
- `--group-by <field>`: group saved-run stats by field.
- `--by <field>`: group saved-run comparisons by field.
- `--metric <field>`: saved-run metric for stats/compare; default `tokSOut`.
- `--metrics <fields>`: comma-separated metrics for comparing two run files.
- `--fields <fields>`: comma-separated saved-run export fields.
- `--hardware-name <text>`: filter saved runs by hardware label substring.
- `--set field=value`: edit one field in a saved benchmark run.
- `--set-json <json>`: merge JSON object into a saved benchmark run.
- `--patch <path>`: merge JSON object file into a saved benchmark run.
- `--unset <fields>`: remove fields from a saved benchmark run.
- `--yes`: confirm saved-run deletion.

Hardware and submission metadata:

- `--hardware <path>`: JSON hardware object required when submitting.
- `--quantization <label>`: explicit quantization assertion; Terminal run reconciles it with the live loaded filename/endpoint metadata and rejects conflicts. Common values are exposed by `lmx context` (`commonSchemas.benchmarkFields.quantization.commonValues`) and include GGUF (`Q4_K_M`, `IQ4_XS`), NVIDIA (`NVFP4`), Unsloth (`Unsloth-Dynamic-Q4_K_M`), bitsandbytes (`bnb-nf4`), AWQ/GPTQ/EXL2/FP8 variants.
- `--gpu-name <name>`: hardware template GPU name.
- `--gpu-count <n>`: hardware template GPU count.
- `--vram-gb <gb>`: hardware template VRAM in GB.
- `--cpu <name>`: hardware template CPU name.
- `--ram-gb <gb>`: hardware template system RAM in GB.
- `--os <name>`: hardware template OS name; default runtime OS.
- `--power-watts <n>`: hardware template power draw in watts.
- `--gpu-power-watts <list>`: benchmark submission per-GPU measured watts, e.g. `285.5,310.2`.
- `--hardware-cost <entries|json>`: benchmark submission purchase records. Component names must come from `lmx context` → `hardwareOptions.hardwareCostComponentNames`. Compact entries use `component|condition|year|price|currency` separated by semicolons; JSON uses an array of objects with `component`, `condition`, optional `yearPurchased`, `price`, and `currency`.
- `--hw-class <class>`: hardware template class, e.g. `DISCRETE_GPU` or `CPU_ONLY`.
- `--name <name>`: saved setup name to pull, case-insensitive.
- `--id <id>`: saved setup id to pull.
- `--default`: pull the default saved setup.

## Troubleshooting

- `benchmark_metric_missing`: pass `--tok-s-out` and at least one secondary metric, or run a benchmark path that measures them.
- `missing_remote_hardware`: run `lmx hardware --out hardware.json` on the server and pass that file when submitting.
- `lm-eval` not found: install it with `pip install lm-eval`, or pass `--lm-eval-bin <path>`.
- Missing API key: run `lmx auth --key bhk_...`, set `LMX_API_KEY`, or pass `--api-key bhk_...`.
- Terminal failures: the human summary lists task, outcome, turns, reason, and an `--out` result pointer or `--trace-dir` path; `--json-status` emits categorized `terminal_failure_summary` data.
- Unknown enum or schema mismatch: refresh live metadata with `lmx context --out localmaxxing-agent-context.json`.
