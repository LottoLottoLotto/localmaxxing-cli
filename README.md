# LocalMaxxing CLI

The official CLI for [localmaxxing.com](https://localmaxxing.com) — run local LLM inference speed tests and quality benchmarks, then submit results from your terminal.

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
lmx --version
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

## Command-name cutover (v0.1.47)

v0.1.47 removes the aliases below. Update scripts and saved command
templates to the canonical spelling when upgrading. Removed spellings are
unknown commands; they do not forward to the replacement or perform work.
Arguments after the renamed command stay unchanged.

| Removed spelling | Canonical spelling |
|---|---|
| `lmx eval lmeval` | `lmx eval lm-eval` |
| `lmx calculator` | `lmx calculate` |
| `lmx decode-calculator` | `lmx calculate decode` |
| `lmx agent-context` | `lmx context` |
| `lmx reports` | `lmx report` |
| `lmx upgrade` | `lmx update` |
| `lmx kv-cache` | `lmx kvcache` |
| `lmx speed-test kv-cache <engine>` | `lmx kvcache run <engine>` |
| `lmx context-sweep` | `lmx kvcache` |
| `lmx speed-test context-sweep <engine>` | `lmx kvcache run <engine>` |

The command-name cutover does not change API endpoints, JSON field names, or
benchmark formats. For example, `/api/agent-context` remains the API route.
The lm-eval importer ignores nonnumeric metadata when selecting scores; absent
preferred metrics no longer mask a valid fallback, and an explicit zero remains
valid. The wrapper collects timestamped harness result files into `--results`
(default `localmaxxing-lm-eval-results.json`) from an isolated current-run directory.
With `--json`, harness logs and tables go to stderr so stdout remains JSON.

Remote KV-cache sweeps now classify reuse from the timed response's counters,
not warm-up slot snapshots. Partial reuse is reported explicitly; missing or
inconsistent evidence no longer produces a cold-prefill claim or prefill rate.

## Authentication

Create an API key from your [LocalMaxxing dashboard](https://localmaxxing.com), then either set it in the environment:

```bash
export LMX_API_KEY=bhk_...
```

or save it locally without putting the secret in process arguments:

```bash
printf '%s\n' "$LMX_API_KEY" | lmx auth --key-stdin
lmx auth
lmx auth --logout
```

Saved config lives under `~/.config/localmaxxing`. `LMX_API_KEY` is preferred
for agents and automation. Passing `--api-key` or `lmx auth --key` can expose
the secret through shell history, process inspection, and agent traces.

## Agent Skill

The binary embeds an Agent Skill documenting the CLI so coding agents can discover and use `lmx` without external docs.

```bash
lmx skill print                       # print SKILL.md to stdout
lmx skill print --out SKILL.md        # or write it to a file
lmx skill install                     # install into ./.claude/skills/localmaxxing-cli
lmx skill install --dir ~/.claude/skills
```

`lmx skill install` writes the skill tree to `<dir>/localmaxxing-cli/`. It works with any harness that discovers `skills/<name>/SKILL.md`, including Claude `.claude/skills` and GitHub `.github/skills`.

Agent-oriented discovery and machine output:

```bash
lmx version --json
lmx commands --json
lmx context list
lmx context get hardwareOptions.hardwareCostComponentNames --compact
```

### Decode upper-bound calculator

Agents can check whether a model fits and estimate its memory-bandwidth decode ceiling without authentication, downloads, or a running inference server:

```bash
lmx calculate decode \
  --capacity-gb 24 \
  --bandwidth-gbps 936.2 \
  --total-params-b 32 \
  --weight-bits 4.5 \
  --allocated-context 32768 \
  --read-context 16000 \
  --json
```

Add `--decoding speculative --draft-tokens <n> --acceptance-percent <p>` plus the draft model’s `--draft-resident-gb` and `--draft-traffic-gb` to model speculative decoding. Machine output reports `result.state`, the best usable batch and throughput when viable, or a structured `constraint` with the exact blocking resource and suggested flag changes.

## Hardware Metadata

Submissions require a hardware JSON object. Generate one on the machine running the speed test:

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

> For remote endpoint speed tests, run `lmx hardware` on the **server** hosting the model, not your local machine.

### Pull a saved setup

If you have saved hardware/engine setups in your LocalMaxxing account, pull one into a `hardware.json` instead of detecting locally — handy when the agent host is not the inference rig. Both commands require an API key (`--api-key`, `LMX_API_KEY`, or saved config) and return only your own setups.

```bash
lmx setups list
lmx setups pull --default --out hardware.json
lmx setups pull --name "2x RTX 3090" --out hardware.json
lmx setups pull --id <setupId> --out hardware.json
```

`setups pull` selects by `--id`, case-insensitive `--name`, or `--default`; with no selector it uses your default setup, or the only setup when you have exactly one. Without `--out` it prints the hardware JSON to stdout.

## Local models

Register existing files without copying them, or download a specific HuggingFace
artifact at an immutable revision. These commands do not publish anything to
LocalMaxxing and require no LocalMaxxing account.

```bash
lmx model list                       # every local model, with details
lmx model list --scan-dir ~/models   # also plain model folders (or set LMX_MODEL_DIRS)
lmx model show Qwen/Qwen3-8B         # all details and files for one model
lmx model import /models/model.gguf
lmx model list --json
lmx model verify <artifact-id>

# Resolve the revision and preview the exact file, bytes, destination and hash.
lmx model pull org/model --revision main --file model.gguf --dry-run
# Use the resolved commit from the plan to keep the reviewed download immutable.
lmx model pull org/model --revision <commit> --file model.gguf --yes
lmx model remove <artifact-id> --yes
```

`model list` shows one table of every local model: files registered with lmx,
lmx snapshots, revisions in the Hugging Face cache (`HF_HUB_CACHE`/`HF_HOME`,
skip with `--no-hf-cache`), and model folders under `--scan-dir` or
`LMX_MODEL_DIRS` (comma- or colon-separated; each folder and its immediate
subfolders). Columns: name, format, quantization, parameters, size, context
length, status, source and lmx ID. Details are read from the files only — no
download, no code: `config.json` (architecture, context, layers, experts,
dtype or quantization scheme such as FP8, GPTQ, W4A16, MXFP4), safetensors
headers (exact parameter count for unquantized weights) and GGUF metadata
(architecture, context, layers, quant type, parameter count). Each GGUF quant
in a folder is its own entry; split GGUFs are grouped. Status flags `partial`
when safetensors shards or GGUF split parts are missing, `missing` when a
registered file is gone, and `loaded <url>` while `lmx model load` serves it. Folders without weights are hidden unless `--all`; `--kind
file|snapshot|hf-cache|directory` filters. `model show <name, ID, ID prefix or
path>` adds revision, path, hashes and the file list. `--json` returns the same
fields (`models[]` with `id`, `kind`, `path`, `status`, …).

Without `--yes`, pull only prints a plan. `--dry-run` takes precedence over
`--yes`; `--out plan.json` may explicitly save the plan without changing model
storage. Completed files are checked against upstream LFS SHA256 or Git blob
hashes when available; otherwise the record explicitly reports only local
SHA256 integrity.

Downloads use up to `--download-connections` parallel HTTP streams (default 8,
max 32). Files of 64 MiB or more are split into byte ranges fetched
concurrently; snapshots also download several files at once, sharing the same
stream limit. Progress (`model_download_progress`, MiB received and MiB/s) is
printed every 10 seconds. An interrupted download keeps `<file>.part` plus a
`<file>.part.chunks` progress record and resumes only the missing ranges; each
range is retried up to 3 times before the run stops. `--download-connections 1`
uses one stream per file.

Storage precedence is `--models-dir`, `LMX_MODELS_DIR`,
`$XDG_DATA_HOME/localmaxxing/models`, then `~/.local/share/localmaxxing/models`.
Existing exact-revision HuggingFace cache files are verified and reused without
copying (`HF_HUB_CACHE`, `HF_HOME`, or the default HF cache).
Removing an imported/cache-backed registration preserves its source; removing a
managed download deletes only that artifact and its registration.

A killed model download may leave a `<destination>.lock`: remove it only after
confirming no operation is active on that artifact. Verification reads the
entire file and can take time for large models.

### Loading and unloading models

`lmx model load` starts any model from `lmx model list` on the engine that fits
it and returns once it answers on an OpenAI-compatible URL; `unload` stops it.

```bash
lmx engines set vllm ~/vllm/bin/vllm --server-env-file vllm-env.json   # once per engine
lmx engines set sglang ~/sglang/bin/python
lmx model load Qwen3-0.6B                                   # safetensors → vLLM (SGLang if no vLLM)
lmx model load LFM2.5-2.6B-GGUF/LFM2.5-2.6B-Q4_K_M --gpus 0 # GGUF → llama.cpp, all layers on GPU
lmx model load Qwen3-0.6B --engine sglang --context-length 8192
lmx model loaded                                            # name, engine, URL, ready/loading/exited
lmx model logs qwen3-0.6b --follow
lmx model unload qwen3-0.6b                                 # or --all
```

- **Engine choice:** GGUF goes to llama.cpp; safetensors/PyTorch weights go to
  vLLM, or SGLang when only SGLang is set up. `--engine` overrides (llama.cpp
  only serves GGUF, vLLM/SGLang not GGUF here). Engines come from
  `lmx engines set <engine> <path>` (vLLM binary, the Python that has SGLang,
  or llama-server; optional `--server-env-file` JSON of environment values
  for that engine), else `vllm` / `llama-server` / a `python3` with SGLang on
  `PATH`. Settings live in `~/.config/localmaxxing/engines.json` (0600);
  `lmx engines` shows them and `lmx engines unset <engine>` removes one.
- **Server:** binds `127.0.0.1` (`--host` to change) on the engine's default
  port (vLLM 8000, SGLang 30000, llama.cpp 8080) or the next free one
  (`--port` to pin). Clients use the repository name, the folder name, or for
  GGUF the file stem as the model name (`--served-model` to change).
  `--context-length`, `--gpus 0,1` (sets `CUDA_VISIBLE_DEVICES` and tensor
  parallelism), `--server-args '["--gpu-memory-utilization","0.5"]'` and
  `--server-env-file` pass through. The saved Hugging Face token is given to
  the server's environment.
- **Lifecycle:** the server runs in its own session, so it keeps running after
  lmx exits; its log is `~/.local/share/localmaxxing/loaded/<name>.log`.
  `load` waits up to `--ready-timeout-seconds` (900) for `/v1/models`; if the
  engine exits, the wait times out or you press Ctrl-C, the server is stopped
  and the log tail is printed. `--no-wait` returns at once. Loading a model
  that is already loaded reports the running server; `--name` loads a second
  copy. `unload` sends SIGTERM to the server's process group and SIGKILL after
  `--stop-timeout-seconds` (30). `lmx model list` marks loaded models with
  their URL.
- vLLM reserves 90% of GPU memory by default; to run several models on one GPU
  lower it with `--server-args '["--gpu-memory-utilization","0.4"]'` (SGLang:
  `--mem-fraction-static`).

### Hugging Face token

A token is needed for gated/private repositories and raises Hugging Face's rate
limits. lmx uses the same token as the `hf` CLI, vLLM, SGLang and transformers:
`HF_TOKEN`, else the file at `HF_TOKEN_PATH`, else `$HF_HOME/token`
(default `~/.cache/huggingface/token`). If you already ran `hf auth login`, lmx
uses that login.

```bash
lmx auth hf --token-stdin   # paste a token from huggingface.co/settings/tokens
lmx auth hf status          # which token is active, and its account
lmx auth hf logout          # delete the saved token file
```

`--token-stdin` checks the token against the Hub before saving and refuses a
rejected one. It is written to the token file readable only by you (0600),
never to lmx's own config. Servers started by `lmx model load` receive the saved
token as `HF_TOKEN` in their environment only; it never appears in argv, plans
or reports. The token is separate from LocalMaxxing auth.

### Complete model snapshots

```bash
lmx model snapshot org/model --revision main --exclude "*.bin,*.h5" --dry-run
lmx model snapshot org/model --revision <resolved-commit> --exclude "*.bin,*.h5" --yes
lmx model snapshots list
lmx model snapshots verify <snapshot-id>
lmx model snapshots remove <snapshot-id> --yes
```

Without filters, the plan includes **all repository files**, potentially
including several weight variants. Review filenames and total bytes before
authorizing. Include/exclude patterns are comma-separated; `*` matches within a
path component and `**` spans directories. Filters do not validate model
semantics: retain the configuration, tokenizer, weight index and shards your
engine needs. Each selected file must have an upstream hash.

Snapshots preserve relative filenames using links to shared verified artifacts.
An interrupted snapshot remains incomplete; rerun the same pinned revision and
selection to resume. Only completed snapshots can be selected for serving.
Removal preserves shared artifacts/cache files, and referenced artifacts cannot
be removed until their snapshots are removed.

## Model reports

Agents and users edit the same standardized GitHub Flavored Markdown document. Fetch the live machine-readable contract, generate a template, and create a private draft for browser review:

```bash
lmx report format --json
lmx report init --out report.md
lmx report create \
  --model Qwen/Qwen3.8-27B \
  --title "Qwen local inference report" \
  --summary "Reproducible setup, measurements, and observations." \
  --content-file report.md \
  --engine FreeToken \
  --quantization BF16 \
  --tags moe,bf16,throughput \
  --draft \
  --json
```

Continue through either the CLI or the web report studio:

```bash
lmx report list --model Qwen/Qwen3.8-27B
lmx report show <reportId>
lmx report edit <reportId> --content-file revised-report.md
lmx report image upload <reportId> --file evidence.png --caption "Measured output"
lmx report publish <reportId>
lmx report unpublish <reportId>
lmx report delete <reportId> --yes
```

The API, CLI, and web editor all use `contentFormat: "gfm"` and format version `1`; no rich-text conversion is required. Use `lmx skill print` for the embedded agent workflow and image-token syntax.

## Speed tests

### Remote Endpoint (FreeToken, NInfer, Strata, vLLM, SGLang, Ollama, custom OpenAI-compatible)

```bash
lmx speed-test run vllm \
  --mode remote \
  --base-url http://server:8000 \
  --hf-id Qwen/Qwen3-8B \
  --served-model Qwen/Qwen3-8B \
  --quantization fp16 \
  --hardware hardware.json \
  --max-tokens 256 \
  --dry-run
```

FreeToken and NInfer are first-class OpenAI-compatible engines:

```bash
lmx speed-test run freetoken \
  --mode remote \
  --base-url http://server:8000 \
  --hf-id Qwen/Qwen3-8B \
  --quantization fp16 \
  --hardware hardware.json \
  --engine-version 0.3.1 \
  --engine-build cuda-12.8-sm120 \
  --engine-commit abcdef123456
```

[Strata](https://github.com/Niko1221/Strata) (Qwen3.8-Flash-Next on consumer GPUs) serves the same API on port 8080:

```bash
lmx speed-test run strata --mode remote --base-url http://127.0.0.1:8080/v1 \
  --hf-id Qwen/Qwen3.8-Flash-Next --quantization IQ2_XS --hardware hardware.json
```

`freetoken`, `ninfer` and `strata` default `--engine-repository` to their canonical GitHub repositories. For Strata, lmx also reads the engine version from the server's `/props` (`build_info: "Strata 0.1.44"`) when `--engine-version` is not given. Override it for forks or mirrors. `--engine-version`, `--engine-repository`, `--engine-build`, and `--engine-commit` are saved with the run and forwarded by later dry-run/submission commands.

For remote endpoint submissions, `--hardware` must describe the server running the endpoint, not the client machine running `lmx`. Run `lmx hardware --out hardware.json` on that server, or provide an equivalent reviewed hardware JSON for that server. If a remote run already has metrics but lacks hardware, attach it without rerunning: `lmx speed-test add-hardware runs/Model/run.json --hardware hardware.json`.

Use `--backend tt-metal` for Tenstorrent endpoints. Remote runs no longer assume CUDA when the backend is omitted.

Remote endpoint runs issue one untimed warmup request and three timed iterations by default, reporting the median of each metric plus per-iteration `samples` and `sampleStats` (min/p50/mean/max/stddev). Tune with `--warmup <n>` and `--iterations <n>`; `--warmup 0 --iterations 1` restores single-shot measurement. Decode throughput is measured over the inter-token window (first to last streamed token) when more than one token arrives.

Record speculative decoding with normalized fields so runs remain comparable across engines:

```bash
lmx speed-test run sglang \
  --mode remote \
  --base-url http://server:30000 \
  --hf-id Qwen/Qwen3.8-27B \
  --quantization bf16 \
  --hardware hardware.json \
  --spec-method dflash2 \
  --spec-draft-model z-lab/Qwen3.8-27B-DFlash2 \
  --spec-num-tokens 8 \
  --spec-draft-window-size 16
```

Use `--spec-method`, `--spec-draft-model`, `--spec-num-tokens`, `--spec-draft-tp`, and `--spec-draft-window-size` for DFlash2, DFlash, MTP, EAGLE, n-gram, or other speculative methods. `dflash2`/`draft-dflash2`, `dflash`/`draft-dflash`, and `mtp`/`draft-mtp` are canonicalized to `DFlash2`, `DFlash`, and `MTP`. A DFlash method paired with a draft-model name containing `DFlash2` is recorded as `DFlash2`. Engine commands from llama.cpp, vLLM `--speculative-config` JSON, and SGLang are normalized when present; explicit CLI fields take precedence and survive saved-run reruns.

Ollama uses the native `/api/generate` endpoint:

```bash
lmx speed-test run ollama \
  --mode remote \
  --base-url http://localhost:11434 \
  --hf-id Qwen/Qwen3-8B \
  --served-model qwen3:8b \
  --quantization Q4_K_M \
  --hardware hardware.json \
  --max-tokens 256
```

Use `--kv-cache-tokens <n>` to record how many tokens were already held in KV cache before the measured request. Omit it for a fresh run; the CLI records `0`. `--prefill-tokens` and `--depth` remain aliases for local `llama-bench -d` compatibility.

### Local llama.cpp

```bash
lmx speed-test run llama.cpp \
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
lmx speed-test validate-local speed-test.json
lmx speed-test dry-run speed-test.json
lmx speed-test submit speed-test.json
```

Use `--out <path>` to set the output file, `--json-status` for machine-readable progress, and `--quiet` to suppress output.

Local speed-test commands run without a time limit by default; pass `--command-timeout-seconds <n>` to abort hung runs.

## Saved Profiles and Runs

Save repeated options as a profile:

```bash
lmx profile save my-4090 \
  --mode local \
  --hf-id Qwen/Qwen3-8B \
  --quantization Q4_K_M \
  --hardware hardware.json

lmx speed-test run llama.cpp --profile my-4090 --model-path model.gguf --dry-run
```

Manage saved run files:

```bash
lmx speed-test runs list
lmx speed-test runs show runs/Qwen-Qwen3-8B/run.json
lmx speed-test runs edit runs/Qwen-Qwen3-8B/run.json --set-json '{"tokSOut":120}'
lmx speed-test runs rerun runs/Qwen-Qwen3-8B/run.json --dry-run
lmx speed-test runs submit runs/Qwen-Qwen3-8B/run.json
lmx speed-test runs delete runs/Qwen-Qwen3-8B/run.json --yes
```

List every submission owned by the authenticated account, including pending and
rejected runs, or patch a recent submission by its server run ID:

```bash
lmx speed-test submissions list --limit 20 --offset 0
lmx speed-test submissions edit <runId> --set-json '{"prefillTokens":4096,"notes":"corrected"}'

# Multiple fields in one edit request; every --set is retained.
lmx speed-test submissions edit <runId> \
  --set tensorParallel=2 \
  --set gpuMemUtil=0.94 \
  --set kvCacheDtype=bfloat16 \
  --set prefixCaching=true \
  --set chunkedPrefill=true \
  --set prefillChunkSize=4096 \
  --set maxRunningSeqs=6

# Alternatively, use the nested engineFlags shape from a submission.
lmx speed-test submissions edit <runId> --set-json \
  '{"engineFlags":{"tensorParallel":2,"prefixCaching":false,"commandSnippet":"vllm serve org/model --tensor-parallel-size 2"}}'
```

Remote edits use `PATCH /api/runs/{id}`. The API limits non-admin owners to the
first 24 hours after submission and enforces a five-minute edit cooldown.
Choose one request form above; each successful edit is subject to that cooldown.
Remote `--patch` and `--set-json` accept flat API fields or a nested `engineFlags`
object, which the CLI flattens before sending. Supply the actual launch command
as `commandSnippet`; the CLI does not invent one from a remote endpoint.

Precedence is `--patch` file, then `--set-json`, then all `--set` assignments in
argument order. Repeating the same field in `--set` uses its last value.
Explicit `false`, zero and `null` are preserved; the API decides which fields
can be cleared and validates value ranges. Unknown remote fields, malformed
assignments, empty/non-object `engineFlags`, or the same field both flat and
nested within one input fail before any API request. Use `--set tensorParallel=2`,
not dotted `--set engineFlags.tensorParallel=2`. Repeated `--set` also works for
local saved-run edits; their arbitrary-field editing behavior is unchanged.


Inspect a saved run for post-run fixes before submission:

```bash
lmx speed-test fixup runs/Qwen-Qwen3-8B/run.json
lmx speed-test add-hardware runs/Qwen-Qwen3-8B/run.json --hardware hardware.json
```


Analyze and export runs:

```bash
lmx speed-test runs show runs/Qwen-Qwen3-8B/run.json --format table
lmx speed-test runs stats --group-by quantization --metric tokSOut
lmx speed-test runs compare --by quantization --metric tokSOut --format table
lmx speed-test runs compare runs/base.json runs/candidate.json --metrics tokSOut,ttftMs --format table
lmx speed-test runs export --format csv --out runs.csv
```

`stats`, `show`, `compare`, and `export` accept `--runs-dir`, plus filters such as `--model`, `--engine`, `--mode`, `--quantization`, `--kind`, and `--hardware-name`. Group stats report min/p50/mean/max/p95/stddev and the best single run; group comparisons rank by the median (`p50`) so a single outlier run cannot win. `show --format table` prints every field from a single run as a flattened ASCII field/value table. `compare` defaults to an ASCII table with baseline, candidate/value, deltas, percent deltas, and a better/worse column; pass `--format json` for machine-readable output. `export` defaults to JSON and supports `--fields path,hfId,hardware,tokSOut` for custom extraction.
## Inference Matrices (llm-inference-bench)

`lmx speed-test matrix` integrates [local-inference-lab/llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench) (MIT, by Local Inference Lab). lmx runs the upstream `llm_decode_bench.py` unmodified against a vLLM or SGLang server and uploads its prefill and context × concurrency decode matrix as one run to `POST /api/inference-benchmarks`. It does not reimplement the measurement.

```bash
# Measure a local server (hardware and engine version are detected) and validate the upload
lmx speed-test matrix run --base-url http://127.0.0.1:8000 --hf-id Qwen/Qwen3-8B \
  --concurrency 1,4,8 --contexts 0,16384 --prefill-contexts 8k --dry-run
lmx speed-test matrix submit inference-matrices/Qwen-Qwen3-8B/matrix-<time>.json

# Let lmx start vLLM for the run, size it for the matrix, and stop it afterwards
lmx speed-test matrix run --model-path ~/models/Qwen3-0.6B --hf-id Qwen/Qwen3-0.6B \
  --concurrency 1,4,16 --contexts 0,4k,16k,32k --dry-run

# Import a report produced elsewhere: llm_decode_bench.py --output JSON or a lil-bench .json.gz
lmx speed-test matrix import benchmark_results.json --hf-id Qwen/Qwen3-8B --quantization FP8 \
  --hardware hardware.json --engine-version 0.11.0 --submit
```

- The default script is upstream release `v0.7.7`, downloaded once to the user cache and checked against a pinned SHA-256; `--bench-script <path>` runs any other copy (e.g. a checkout of a newer release). Python 3.10+ with `httpx rich psutil` is required (`--python-bin` selects the interpreter).
- `run` is a thin wrapper around `import`: it detects the server's hardware and version, runs the bench, then imports the report through the same code as `lmx speed-test matrix import`. Pass `--server-command "<launch command>"` to record how the server was started. For containers built from lil-bench, running lil-bench and importing its `.json.gz` is equivalent.
- You choose the grid. lmx checks every option before the bench (or a managed server) starts, prints the plan (`decodeCells`, a lower bound on the timed decode minutes) and records it with the matrix:

  | Option | What it sets | Default |
  |---|---|---|
  | `--contexts 0,4k,16k,32k` | decode context lengths in tokens (`k` = ×1024) | `0,16k,32k,64k,128k` |
  | `--concurrency 1,2,4,8` | simultaneous requests per decode cell | `1,2,4,8,16,32,64,128` |
  | `--max-tokens N` | output tokens per decode request | 8192 |
  | `--duration-seconds N` | seconds per decode cell | 30 |
  | `--warmup-seconds N` | one-request decode warmup before measuring (0 skips) | 3 |
  | `--prefill-contexts 8k,64k` | extra prefill-only context lengths | none |
  | `--standalone-prefill`, `--prefill-duration-seconds N` | measure prefill cold before decode, N s per context | from decode warmup; 10 s |
  | `--skip-prefill` | decode only | off |
  | `--run-burst`, `--burst-requests N` | add a short burst grid after sustained decode, N requests per concurrency level | off; 5 |
  | `--request-count N` | burst only: exactly N requests per cell instead of timed decode | off |

  Contradictory combinations (`--skip-prefill` with prefill options, `--request-count` with timed or burst options) are refused. Other bench options go through `--extra-bench-args "..."`; the grid options above, and the ones lmx sets itself (`--host`, `--model`, `--output`, ...), are refused there so the recorded plan is the one that ran. `--max-tokens` defaults to 8192, the bench's own decode-matrix default.
- Before the bench starts, `run` reads the server's context window (`max_model_len` from `/v1/models`, or SGLang's `context_length`) and checks every decode context: context + `--max-tokens` must fit, and contexts below 8k reserve 256 more tokens because the bench's prompt wrapper (chat template, document framing, generation prompt) is calibrated on an 8k probe and can push small prompts slightly over their target. Oversize contexts fail fast with `context_exceeds_server_limit`, naming the `--max-model-len` to restart with, the largest `--max-tokens` that fits, and the contexts that fit; `--skip-oversize` drops them with a warning instead. Prefill contexts are not checked; the bench caps them itself. Servers that do not report a limit skip the check.
- Managed mode: `--model-path <dir>` (instead of `--base-url`) starts `vllm serve <dir>` on a free `127.0.0.1` port with `--served-model-name <hf-id>`, `--max-model-len` = largest context + `--max-tokens` (+ the margin above) rounded up to 1024, and `--gpu-memory-utilization 0.9`. Contexts beyond the model's `max_position_embeddings` (config.json) are dropped with a warning. The executable is `--engine-bin`, else `vllm` on `PATH`, else `~/vllm/bin/vllm`, `~/.venv/bin/vllm`, `~/venvs/vllm/bin/vllm` or `~/.local/share/vllm/bin/vllm`. Add server flags with `--engine-args "--enforce-eager --max-num-seqs 64"` (a `--max-model-len` there replaces lmx's choice). The engine binary's directory goes first on the server's `PATH` (as an activated venv would), so FlashInfer's startup JIT finds the venv's `ninja`. Without `nvcc` (PATH, `$CUDA_HOME/bin`, `/usr/local/cuda/bin`) lmx sets `VLLM_USE_FLASHINFER_SAMPLER=0` up front; if the server still dies in FlashInfer's JIT build, lmx restarts it once with that setting (first log kept as `.server.first-attempt.log`) and records the command that actually ran. lmx waits for `/health` and the model in `/v1/models` (`--start-timeout`, default 900 s), logs the server to `matrix-<time>.server.log` next to the matrix, records the exact launch command as the server command, and stops the server (SIGTERM, then SIGKILL after 30 s, or 5 s if it never became ready) when the run ends, fails, or is interrupted. If vLLM dies during startup, the error names its root-cause exception (e.g. `max_num_seqs (256) exceeds available Mamba cache blocks`). `--model-path` must contain `config.json`; for GGUF weights start the server yourself and use `--base-url`. Only vLLM can be managed; start SGLang yourself and use `--base-url`.
- `run` saves `inference-matrices/<model>/matrix-<time>.json` plus the untouched bench report beside it (`.bench.json`). `--submit` imports and records the run ID in the file.
- `--quantization` is optional. lmx asks the endpoint first; vLLM and SGLang report nothing for unquantized models, so it then uses the launch command's `--quantization`, the model's `config.json` `quantization_config` (FP8, AWQ, GPTQ, MXFP4, NVFP4), the launch `--dtype`, and finally the config dtype (BF16/FP16/FP32). `config.json` is read from the local model directory in the launch command, else from Hugging Face (the launch repo, else `--hf-id`). Formats that cover several layouts (compressed-tensors, bitsandbytes) still need `--quantization`. `run` resolves this before the benchmark starts; the launch command comes from `--server-command` or the lil-bench serving command.
- A matrix whose decode throughput is implausible (any cell above 10× the memory-bandwidth decode ceiling at its concurrency, the same check as speed tests) is stored but held for admin review and hidden until approved; `--dry-run` and `--submit` say so with the reason. The saved receipt links the run page, which you can open while it is held.
- `--dry-run` (or `lmx speed-test matrix dry-run <file>`) runs the site's complete import — quota, duplicate check, HuggingFace lookup, hardware/engine resolution and every database write — in a transaction that is rolled back, and prints exactly what would be stored. A passing dry-run means the same file submitted now is accepted; a duplicate or exhausted quota fails the dry-run with the same 409/429 the submit would get.
- Manage what you uploaded with `lmx speed-test matrix submissions list`, `submissions edit <id> --set quantization=FP8 --set engineVersion=0.11.0 [--hardware hardware.json] [--dry-run]`, and `submissions delete <id> --yes`. Only metadata is editable (model ID, revision, quantization, hardware, engine version/repository/build/commit, backend, launch command), within 24 hours and once per 5 minutes; `--set field=null` clears optional fields. Wrong numbers mean deleting and re-importing. The run ID is in `submission.id` of the saved matrix file. Owners also get Edit and Delete buttons on their matrix cards on the model page.
- On localmaxxing.com each upload appears on the model page (with a throughput-vs-concurrency chart), on its own run page (`/inference-benchmarks/<id>`: full hardware/engine specs, launch command, every cell's metrics), and under Inference matrices in your dashboard. lmx records which command produced it (`lmx <version> run`, `import`, or `import (lil-bench …)`).
- `run` detects hardware only when `--base-url` names this machine (localhost, loopback, or one of its interface addresses); remote endpoints fail with `missing_remote_hardware` before the benchmark starts, so pass `--hardware hardware.json` generated on the server. SSH/kubectl tunnels and proxies make remote servers look local, so pass `--hardware` for them too. lil-bench imports use the container's own GPU/CPU/RAM inventory, serving command and vLLM version. Secret-looking arguments (`--api-key`, tokens) are redacted from recorded commands.
- Only `vllm` and `sglang` reports are accepted; hosted APIs (`openai_proxy`) are rejected. Per-request samples, event logs and server URLs are not uploaded. Matrix runs are separate from single speed tests and are not eligible for the Verified-run badge.

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

Remote sweeps pre-warm the target context prefix, then time a streaming probe
with the same prefix. The default filler is a deterministic varied-word
sequence; pass `--filler-token <word>` to override.

`cacheReuse` uses only the timed response's `usage.prompt_tokens_details.cached_tokens`
or llama.cpp `timings.cache_n` / `timings.prompt_n`, checked against
`usage.prompt_tokens`. Warm-up results and `/slots` snapshots are not evidence
of reuse by the timed request.

- `retained`: the entire timed prompt was cached.
- `partial`: some, but not all, timed prompt tokens were cached.
- `not_retained`: the timed response explicitly reports zero cached tokens.
- `unknown`: counters are missing, invalid, or inconsistent; a warning explains
  the uncertainty, and no prefill rate is derived.

Verified points include `cachedTokens` and `uncachedTokens` inside `cacheReuse`,
along with raw usage/timing evidence. These counts describe the full timed
prompt, including chat-template and probe tokens—not guaranteed retention of
the nominal `contextTokens` depth. Prefill throughput is estimated from only
`uncachedTokens / TTFT`, with `tokSPrefillSource` persisted as
`estimated_from_ttft_uncached`; HTTP overhead and first-token generation remain
included in TTFT. No prefill rate is reported for a fully cached prompt.

## Evals

Discover approved shard datasets first:

```bash
lmx eval dataset list
lmx eval dataset show hellaswag
lmx eval shard hellaswag --base-url http://localhost:8000 --questions 3 --dry-run
```

Shard datasets and registered suites are separate catalogs. HellaSwag and GSM8K
use `eval shard`, not `eval run` or `eval lm-eval`. An empty `eval suite list`
does not mean evals are unavailable. Terminal datasets use `eval terminal`.
The suite examples below require an approved suite on the selected API instance.

### Run a custom suite against a local endpoint

```bash
lmx eval run my-custom-suite \
  --model Qwen/Qwen3-8B \
  --base-url http://localhost:8000 \
  --hardware hardware.json \
  --submit
```

### LM-Eval Harness

Set `SUITE_SLUG` to an approved `LM_EVAL_HARNESS` suite from `lmx eval suite list`;
do not substitute a shard dataset slug.

```bash
lmx eval lm-eval "$SUITE_SLUG" \
  --model Qwen/Qwen3-8B \
  --backend hf \
  --hardware hardware.json \
  --dry-run
```

The wrapper saves the native aggregate result at `--results` (default
`localmaxxing-lm-eval-results.json`) and the imported run payload at `--out`.
It collects only this invocation's output; missing or multiple aggregate files
are errors. `--dry-run` performs inference and validates the payload, `--submit`
submits it, and neither flag means local files only. With `--json`, harness
output goes to stderr.

Upload existing lm-eval output to the approved suite whose task keys match:

```bash
lmx eval run "$SUITE_SLUG" \
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

### Publish a Harbor / Terminal-Bench dataset

The recommended path is one command against raw Harbor/Terminal-Bench tasks:

```bash
lmx eval publish ./harbor-tasks \
  --description "Repository-level API repair tasks with deterministic verifiers." \
  --source-url https://github.com/org/repo
```

The CLI derives the slug and name from the directory; override them with
`--slug` and `--name`. It imports raw tasks into a managed `.localmaxxing/`
workspace, checks Pro access and storage quota **before** expensive work,
oracle-verifies every task, rejects unsupported or unsafe bundles, creates
deterministic `.tar.gz` archives, uploads them with signed SHA-256 metadata,
validates the full manifest, and submits it as `PENDING` for admin review.
Interrupted publishes are resumable and reuse verified uploads.

Executable terminal publication requires LocalMaxxing Pro, Docker, at least two
tasks, a description of at least 20 characters, and a public HTTPS
`--source-url` for provenance and review. `--dry-run` stops before dataset
creation but still uploads and verifies the task archives. `--skip-oracle` is an
advanced escape hatch for a collection already verified locally; it never
bypasses server policy or admin approval.

The lower-level `lmx eval terminal import` and
`lmx eval terminal publish` commands remain available when the canonical bundle
directory must be inspected or edited between steps.

Imported Harbor tasks may use either one Dockerfile or a multi-service
`docker-compose.yaml`. Compose service dependencies, GPU requests, artifact
collection/exclusions, and `verifier.environment_mode = "separate"` are
preserved through execution; separate verifiers receive only declared artifacts.

### Train from verified eval trajectories

Prepare conversational SFT data from scored passing OMP trajectories. Failed
tasks are written separately as diagnostics and are never treated as correct
training targets:

```bash
lmx eval train prepare ./completed-terminal-run \
  --out ./training-data \
  --base-model Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --allow-benchmark-training

lmx eval train run ./training-data/manifest.json \
  --trainer-cmd "python3 python/localmaxxing_helpers/train_eval_sft.py --backend unsloth --dataset {dataset} --model {base_model} --output {output} --lora-dropout 0"
```

`eval train run` only prints the expanded local command by default. Add
`--execute` to start it. The bundled trainer supports Unsloth's patched QLoRA
stack and a plain TRL/PEFT backend. Install Unsloth using its CUDA/PyTorch-specific
installation instructions; use `--backend trl` with
`python3 -m pip install 'trl[peft]' bitsandbytes datasets` for the plain backend.

Training on benchmark tasks contaminates future scores on those tasks, so
preparation requires an explicit acknowledgement and the resulting adapter must
be evaluated on a separate holdout.

### Train with online GRPO

Online GRPO starts from the task bundles produced by `eval terminal import`, not
from a completed terminal result. Preparation writes prompt-only data and a
typed manifest; it does not copy historical pass/fail or reward labels:

```bash
lmx eval train rl prepare ./tb-bundles \
  --out ./rl-training \
  --base-model Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --environment-factory my_package.environments:make_environment \
  --environment-config ./environment.json \
  --grpo-config ./grpo.json \
  --allow-benchmark-training
```

`./rl-training/prompts.jsonl` contains each user prompt, task ID, and bundle
reference. `./rl-training/manifest.json` records that dataset, the absolute
bundle root, base model, validated GRPO settings, output path, contamination
acknowledgement, and trusted environment plugin. The plugin is required: the
CLI does not provide a task environment, so training is not runnable until the
named `module:callable` is importable by the selected Python environment.

The factory must accept keyword arguments `bundle_root` (a `pathlib.Path`) and
`config` (the JSON object from `--environment-config`) and return a TRL 1.8
environment. Its `reset(**row)` selects and resets an isolated task from
`task_id`/`bundle_ref`; typed, documented public methods become policy tools;
and `get_reward()` returns the verifier reward after the rollout. Every reward
must come from a fresh rollout of the current policy. The plugin must sandbox
tools, run verification out of band, and keep verifier assets and any reference
solution inaccessible to the policy. Treat the plugin as trusted local code.

```bash
lmx eval train rl run ./rl-training/manifest.json --resume auto
lmx eval train rl run ./rl-training/manifest.json \
  --output-dir ./checkpoints/grpo \
  --resume none \
  --python-bin python3 \
  --execute
```

`rl run` prints a direct-argv plan by default; it never invokes a shell or
accepts `--trainer-cmd`. `--execute` starts the embedded trainer. `--resume auto`
(the default) starts fresh for a missing/empty output directory or selects the
highest valid `checkpoint-N` in a nonempty one; `none` requires an empty output
directory; a path must name a checkpoint containing `trainer_state.json`.

Install a hardware-appropriate PyTorch build first, using PyTorch's selector for
the host GPU/accelerator and driver. Then install the runner's pinned versions:

```bash
python -m pip install 'trl==1.8.0' 'transformers>=5.2.0,<6'
```

Environment tools also require `jmespath`. Use a model and chat template that
support tool calling and preserve the generated prefix. As with SFT, these eval
tasks contaminate same-task measurements: train only after acknowledging that
fact and report quality on a separate, previously unseen holdout.

## Discover Models, Datasets, and Suites

```bash
lmx context --out localmaxxing-agent-context.json
lmx eval dataset list --out localmaxxing-datasets.json
lmx eval dataset show hellaswag
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

Start with the guarded one-command workflow:

```bash
lmx auth login

lmx eval publish questions.jsonl \
  --description "Original networking questions written to test protocol reasoning." \
  --source-url https://github.com/org/repo
```

Accepted inputs are CSV, JSONL, JSON arrays, LocalMaxxing suite manifests,
imported terminal bundles, and raw Harbor/Terminal-Bench directories. For
tabular data, the CLI safely detects common columns:

- Prompt: `input`, `question`, `prompt`, or `instruction`
- Gold answer: `gold`, `answer`, `label`, `target`, or `expected_answer`
- Choices: `choices`, `options`, or `answers`
- Judge rubric: `rubric`, `grading_rubric`, or `criteria`

Ambiguous or missing mappings stop before network writes and explain the exact
`--input-column`, `--gold-column`, `--choices-column`, or `--rubric-column`
override to use. The CLI infers `multiple_choice`, `judge`, or `qa`, prints the
detected mapping, and derives a name and slug from the filename. Override any
inference with `--kind`, `--name`, or `--slug`.

Before submission it validates the manifest, requires a useful description,
audits duplicates, invalid choices, likely label leakage, answer imbalance, and
dataset size, then runs the authenticated server preflight. Only after that
passes does it upload inline datasets and create a `PENDING` submission. Use
`--dry-run` to stop before uploads and submission, and `--strict` when audit
warnings should also block publication.

Generated manifests live under `.localmaxxing/` so rejected submissions can be
edited and resubmitted:

```bash
lmx eval suite submissions
lmx eval suite resubmit <submission-id> \
  --file .localmaxxing/my-benchmark.eval-suite.json
```

The lower-level `eval suite init`, `import`, `validate`, `audit`, `check`, and
`submit` commands remain available for authors who need full control.

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
