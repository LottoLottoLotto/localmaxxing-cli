# LocalMaxxing CLI

Public CLI for authoring, running, validating, and submitting LocalMaxxing eval suites.

This package is standalone. It does not include the private LocalMaxxing web app, database, Prisma schema, deployment configuration, or server internals.

## Install

From npm after publish:

```bash
npm install -g @localmaxxing/cli
```

Or run directly:

```bash
npx @localmaxxing/cli --help
```

From this repository checkout:

```bash
git clone https://github.com/LottoLottoLotto/localmaxxing-cli.git
cd localmaxxing-cli
npm install
npm run build
npm link
lmx --help
```

## Authentication

Create a LocalMaxxing API key from your dashboard, then pass it with `--api-key` or set:

```bash
export LMX_API_KEY=bhk_...
```

On Windows PowerShell:

```powershell
$env:LMX_API_KEY = "bhk_..."
```

Or save it for future CLI commands:

```bash
lmx auth --key bhk_...
lmx auth
lmx auth --logout
```

## Hardware File

Submissions require a hardware JSON object. Example `hardware.json`:

```json
{
  "hwClass": "DISCRETE_GPU",
  "gpuName": "RTX 4090",
  "vramGb": 24
}
```

You can generate a best-effort hardware file from the current machine:

```bash
lmx hardware --out hardware.json
```

## Submit An Inference Benchmark

Create a benchmark payload JSON matching `POST /api/benchmarks`, then validate it without writing:

```bash
lmx benchmark dry-run benchmark.json --api-key bhk_...
```

Submit after the dry-run passes:

```bash
lmx benchmark submit benchmark.json --api-key bhk_...
```

`bench` is accepted as a shorter alias for `benchmark`.

Saved `lmx-bench` result files are also accepted. If the JSON has a top-level `payload` field, the CLI submits that payload automatically.

If engine token counts are missing or marked as estimated and the JSON includes generated text (`outputText`, `generatedText`, `completion`, `response`, or `text`), the CLI uses the model's Hugging Face tokenizer to fill `outputTokens`. If the submitted model has no tokenizer, it tries `tokenizer_name` or `base_model_name_or_path` from `config.json`.

## Discover Eval Requirements

Agents can pull the current LocalMaxxing schemas, endpoints, benchmark requirements, methodology tips, approved suite list, and per-suite run instructions directly from the API:

```bash
lmx context --out localmaxxing-agent-context.json
lmx eval suite list --out localmaxxing-suites.json
lmx eval suite search reasoning --out reasoning-suites.json
lmx eval suite show hellaswag --out hellaswag-suite.json
lmx model search qwen3-8b --out models.json
```

`eval suite show` returns the suite document, task keys, scoring method, aggregation mode, and `agentInstructions` such as lm-eval command templates or server-side custom eval payloads.

Suite discovery output and eval run artifacts are redacted defensively so fields such as `gold`, `answer`, and `referenceAnswer` are not written to agent-facing files or uploaded as run artifacts.

## Artifact Bundles And Storage

Large eval traces should be uploaded as bucket-backed artifacts instead of inline run artifacts:

```bash
lmx eval artifacts upload traces.jsonl \
  --format jsonl \
  --item-count 1000 \
  --out artifact-bundle.json
```

The lower-level storage commands support both artifact bundles and bucket-backed datasets:

```bash
lmx eval storage upload dataset.jsonl \
  --kind dataset \
  --format jsonl \
  --out dataset-storage.json

lmx eval storage download <storageKey> --out traces.jsonl
```

## Run A Custom Eval

Start a local OpenAI-compatible server, then run:

```bash
lmx eval run my-custom-suite \
  --model Qwen/Qwen3-8B \
  --base-url http://localhost:8000 \
  --hardware hardware.json \
  --submit
```

For a public OpenAI-compatible endpoint that LocalMaxxing can reach directly, use server-side execution:

```bash
lmx eval execute my-custom-suite \
  --model Qwen/Qwen3-8B \
  --base-url https://my-public-endpoint.example \
  --hardware hardware.json \
  --submit
```

Use `eval run` for localhost/private endpoints; use `eval execute` for public endpoints you want LocalMaxxing to call and score server-side.

## Run An LM-Judge Eval

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

## Run An LM-Eval Suite

Run lm-eval-harness through the CLI wrapper:

```bash
lmx eval lm-eval hellaswag \
  --model Qwen/Qwen3-8B \
  --backend hf \
  --hardware hardware.json \
  --dry-run
```

For `--backend hf`, the CLI defaults `--model-args` to `pretrained=<model>`. Override it for other backends:

```bash
lmx eval lm-eval hellaswag \
  --model Qwen/Qwen3-8B \
  --backend vllm \
  --model-args pretrained=Qwen/Qwen3-8B,tensor_parallel_size=1 \
  --hardware hardware.json \
  --submit
```

The wrapper fetches the suite, runs `lm_eval`, parses the output JSON, writes a LocalMaxxing run payload, then dry-runs or submits it.

You can still run lm-eval-harness yourself:

```bash
lm_eval \
  --model hf \
  --model_args pretrained=Qwen/Qwen3-8B \
  --tasks arc_challenge,hellaswag,mmlu \
  --output_path localmaxxing-lm-eval-results.json
```

Then upload:

```bash
lmx eval run local-open-llm-core \
  --model Qwen/Qwen3-8B \
  --results localmaxxing-lm-eval-results.json \
  --hardware hardware.json \
  --submit
```

## Create A New Eval Suite

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

## Agent-Friendly Errors

The CLI emits structured diagnostics such as:

```text
[localmaxxing:error] suite_validation_failed
Suite validation failed.
Fix:
- Edit the suite JSON file and rerun eval suite validate.
Details:
[
  "slug must be 3-64 lowercase alphanumeric characters with hyphens"
]
```

Agents should read the error code, apply the `Fix:` hints, and rerun the command.

## Docs

- `docs/running-evals.md`
- `docs/eval-suite-authoring.md`
- `docs/lm-eval-compatibility.md`

## Publish

```bash
npm version patch
npm publish --access public
```
