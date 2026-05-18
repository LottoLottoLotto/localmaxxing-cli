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

## Hardware File

Submissions require a hardware JSON object. Example `hardware.json`:

```json
{
  "hwClass": "DISCRETE_GPU",
  "gpuName": "RTX 4090",
  "vramGb": 24
}
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

## Run A Custom Eval

Start a local OpenAI-compatible server, then run:

```bash
lmx eval run my-custom-suite \
  --model Qwen/Qwen3-8B \
  --base-url http://localhost:8000 \
  --hardware hardware.json \
  --submit
```

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

Run lm-eval-harness yourself:

```bash
lm-eval run \
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

- `docs/eval-suite-authoring.md`
- `docs/lm-eval-compatibility.md`

## Publish

```bash
npm version patch
npm publish --access public
```
