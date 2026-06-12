# Build And Run An Eval On LocalMaxxing

This guide shows the practical flow for using the LocalMaxxing CLI to discover evals, run them locally, validate the result, and submit it to localmaxxing.com.

## 1. Install The CLI

Download a pre-built archive from the [latest release](https://github.com/LottoLottoLotto/localmaxxing-cli/releases/latest) (see the README for checksum verification):

```bash
curl -fsSLO https://github.com/LottoLottoLotto/localmaxxing-cli/releases/latest/download/lmx-linux-amd64.tar.gz
tar -xzf lmx-linux-amd64.tar.gz
sudo mv lmx /usr/local/bin/
```

Or build from source (requires Go 1.22+):

```bash
git clone https://github.com/LottoLottoLotto/localmaxxing-cli.git
cd localmaxxing-cli
go build -o lmx ./cmd/lmx
```

Verify it works:

```bash
lmx --help
```

## 2. Authenticate

Create an API key in LocalMaxxing, then either save it:

```bash
lmx auth --key bhk_...
```

Or use an environment variable.

macOS/Linux:

```bash
export LMX_API_KEY=bhk_...
```

Windows PowerShell:

```powershell
$env:LMX_API_KEY = "bhk_..."
```

Check auth status:

```bash
lmx auth
```

## 3. Generate Hardware Metadata

Submissions need a hardware object. Generate a best-effort file:

```bash
lmx hardware --out hardware.json
```

Open `hardware.json` and correct anything the machine detector cannot know, such as exact GPU name, VRAM, power limits, or multi-GPU layout.

## 4. Discover Evals And Models

Fetch the current LocalMaxxing agent context:

```bash
lmx context --out localmaxxing-agent-context.json
```

List approved eval suites:

```bash
lmx eval suite list --out localmaxxing-suites.json
```

Search suites:

```bash
lmx eval suite search reasoning --limit 10 --out reasoning-suites.json
```

Inspect one suite:

```bash
lmx eval suite show hellaswag --out hellaswag-suite.json
```

Search for the canonical model ID:

```bash
lmx model search qwen3-8b --out models.json
```

The suite file includes `suiteDoc`, task keys, scoring method, aggregation, and `agentInstructions`. Discovery output is redacted defensively so structured gold/reference fields are not written to agent-facing files.

## 5. Run An LM-Eval Harness Suite

For `LM_EVAL_HARNESS` suites such as `hellaswag`, `mmlu`, or `gsm8k`, use the wrapper:

```bash
lmx eval lm-eval hellaswag \
  --model Qwen/Qwen3-8B \
  --backend hf \
  --hardware hardware.json \
  --dry-run
```

For `--backend hf`, the CLI defaults `--model-args` to `pretrained=<model>`.

For another lm-eval backend, pass explicit model args:

```bash
lmx eval lm-eval hellaswag \
  --model Qwen/Qwen3-8B \
  --backend vllm \
  --model-args pretrained=Qwen/Qwen3-8B,tensor_parallel_size=1 \
  --hardware hardware.json \
  --dry-run
```

The wrapper:

- Fetches the suite from LocalMaxxing.
- Verifies it is an `LM_EVAL_HARNESS` suite.
- Builds and runs `lm_eval`.
- Parses the output JSON.
- Writes a LocalMaxxing run payload.
- Calls the LocalMaxxing dry-run or submit endpoint.

If `lm_eval` is not installed:

```bash
pip install lm-eval
```

If your executable has a custom name or path:

```bash
lmx eval lm-eval hellaswag \
  --model Qwen/Qwen3-8B \
  --lm-eval-bin /path/to/lm_eval \
  --hardware hardware.json \
  --dry-run
```

## 6. Upload Existing LM-Eval Results

If you already ran lm-eval yourself, upload the result JSON:

```bash
lmx eval run hellaswag \
  --model Qwen/Qwen3-8B \
  --results localmaxxing-lm-eval-results.json \
  --hardware hardware.json \
  --dry-run
```

When the dry-run passes, submit it:

```bash
lmx eval run hellaswag \
  --model Qwen/Qwen3-8B \
  --results localmaxxing-lm-eval-results.json \
  --hardware hardware.json \
  --submit
```

## 7. Run A Custom LocalMaxxing Suite

For `CUSTOM` suites, start a local OpenAI-compatible server first. Then run:

```bash
lmx eval run local-reasoning-mini \
  --model Qwen/Qwen3-8B \
  --base-url http://localhost:8000 \
  --hardware hardware.json \
  --dry-run
```

Use `--submit` after the dry-run passes.

For public OpenAI-compatible endpoints that LocalMaxxing can reach directly, use server-side execution:

```bash
lmx eval execute local-reasoning-mini \
  --model Qwen/Qwen3-8B \
  --base-url https://your-public-endpoint.example \
  --hardware hardware.json \
  --submit
```

Use `eval run` for localhost/private endpoints. Use `eval execute` when LocalMaxxing should call a public endpoint and score server-side.

## 8. Upload Large Artifacts

Small artifacts can be included inline by eval runs. Large traces should be uploaded as an artifact bundle:

```bash
lmx eval artifacts upload traces.jsonl \
  --format jsonl \
  --item-count 1000 \
  --out artifact-bundle.json
```

For lower-level storage use:

```bash
lmx eval storage upload traces.jsonl \
  --kind artifact \
  --format jsonl \
  --out artifact-bundle.json
```

Download a storage object when you have a storage key:

```bash
lmx eval storage download <storageKey> --out traces.jsonl
```

## 9. Build And Submit A New Eval Suite

Create a starter suite:

```bash
lmx eval suite init \
  --slug my-reasoning-eval \
  --name "My Reasoning Eval" \
  --category reasoning \
  --kind multiple_choice \
  --out my-reasoning-eval.json
```

Edit the generated JSON and replace sample items with your dataset. Then validate:

```bash
lmx eval suite validate my-reasoning-eval.json
```

Submit the suite for LocalMaxxing approval:

```bash
lmx eval suite submit my-reasoning-eval.json
```

Submitted suites may require approval before public runs can target them.

## 10. Safety Notes

- Always run `--dry-run` before `--submit`.
- Do not put gold answers inside `input`, `choices`, prompt text, or artifact text.
- Structured fields such as `gold`, `answer`, and `referenceAnswer` are redacted from suite discovery output and eval run artifacts, but arbitrary prose cannot be reliably redacted.
- Never commit API keys, `.env` files, or raw private eval datasets.
- For public comparison, record exact model ID, quantization, backend, command flags, hardware, and relevant runner versions.

## Common Errors

Missing API key:

```text
[localmaxxing:error] missing_api_key
```

Fix: run `lmx auth --key bhk_...` or set `LMX_API_KEY`.

Missing hardware:

```text
[localmaxxing:error] missing_hardware
```

Fix: run `lmx hardware --out hardware.json` and pass `--hardware hardware.json`.

Wrong suite type for lm-eval wrapper:

```text
[localmaxxing:error] suite_runner_mismatch
```

Fix: use `lmx eval run` for `CUSTOM` suites, or choose an `LM_EVAL_HARNESS` suite.

Missing lm-eval executable:

```text
[localmaxxing:error] command_failed
```

Fix: install with `pip install lm-eval` or pass `--lm-eval-bin <path>`.
