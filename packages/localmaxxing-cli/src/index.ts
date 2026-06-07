#!/usr/bin/env node
import { execFileSync, execSync, spawnSync } from 'node:child_process'
import { createHash } from 'node:crypto'
import { homedir, platform, release, totalmem, cpus } from 'node:os'
import { basename, dirname, join } from 'node:path'
import { mkdir, readFile, readdir, rm, stat, writeFile } from 'node:fs/promises'

type Runner = 'LM_EVAL_HARNESS' | 'CUSTOM'
type ScoringMethod = 'exact_match' | 'f1' | 'pass_at_k' | 'perplexity' | 'llm_judge'

type DatasetItem = {
  input: string
  gold?: string | number
  choices?: string[]
  referenceAnswer?: string
  rubric?: string
}

type EvalTask = {
  key: string
  displayName?: string
  taskType?: string
  weight?: number
  nShots?: number
  higherIsBetter?: boolean
  dataset?: { source: string; url?: string; downloadUrl?: string; downloadUrls?: unknown; format?: string; hfPath?: string; hfName?: string; split?: string; items?: DatasetItem[] }
  promptTemplate?: string
  maxNewTokens?: number
  stopSequences?: string[]
}

type SuiteDoc = {
  version?: string
  runner: 'lm-eval-harness' | 'custom'
  scoringMethod: ScoringMethod
  higherIsBetter?: boolean
  aggregation?: 'mean' | 'weighted_mean' | 'min' | 'max'
  tasks: EvalTask[]
  runConfig?: { temperature?: number; topP?: number; fewShot?: number }
  judge?: { provider?: string; model: string; baseUrl: string }
}

type SuiteResponse = {
  slug: string
  name: string
  runner: Runner
  suiteDoc: SuiteDoc
}

type RunBundleResponse = SuiteResponse & {
  suite?: Partial<SuiteResponse>
  downloadUrls?: unknown
  datasetDownloadUrls?: unknown
  datasets?: unknown
}

type SuiteCreatePayload = {
  slug: string
  name: string
  description?: string
  category: string
  runner: Runner
  version?: string
  sourceUrl?: string
  suiteDoc: SuiteDoc
}

const LMEVAL_METRIC_CANDIDATES_BY_SCORING: Record<ScoringMethod, string[]> = {
  exact_match: [
    'acc_norm,none',
    'acc,none',
    'exact_match,none',
    'exact,none',
    'em,none',
    'pass_at_1,none',
    'pass@1,none',
    'inst_level_strict_acc,none',
    'prompt_level_strict_acc,none',
    'acc_norm',
    'acc',
    'exact_match',
    'exact',
    'em',
    'pass_at_1',
    'pass@1',
    'inst_level_strict_acc',
    'prompt_level_strict_acc',
  ],
  f1: ['f1,none', 'f1', 'macro_f1,none', 'macro_f1', 'rouge1,none', 'rouge1', 'rougeL,none', 'rougeL'],
  pass_at_k: ['pass_at_1,none', 'pass@1,none', 'pass_at_1', 'pass@1', 'pass_at_k,none', 'pass@k,none', 'pass_at_k', 'pass@k'],
  llm_judge: ['score,none', 'score', 'acc,none', 'acc'],
  perplexity: ['word_perplexity,none', 'perplexity,none', 'ppl,none', 'word_perplexity', 'perplexity', 'ppl'],
}

type Artifact = {
  taskKey: string
  itemIndex: number
  attempt?: number
  promptHash: string
  question?: string
  prompt: string
  response?: string
  score?: number
  judgeModel?: string
  judgeScore?: number
  judgeRationale?: string
  latencyMs?: number
  error?: string
}

type TaskScore = { score: number; nSamples?: number; nShots?: number }

type CliConfig = { apiKey?: string; authProvider?: string; authSavedAt?: string }

const CONFIG_DIR = join(homedir(), '.config', 'localmaxxing')
const CONFIG_FILE = join(CONFIG_DIR, 'config.json')
const GOLD_FIELD_NAMES = new Set(['gold', 'answer', 'referenceAnswer', 'expectedAnswer', 'correctAnswer', 'label', 'target'])

class CliError extends Error {
  code: string
  hints: string[]
  details?: unknown

  constructor(code: string, message: string, hints: string[] = [], details?: unknown) {
    super(message)
    this.name = 'CliError'
    this.code = code
    this.hints = hints
    this.details = details
    Object.setPrototypeOf(this, CliError.prototype)
  }
}

function asMessage(error: unknown) {
  return error instanceof Error ? error.message : String(error)
}

function printInfo(title: string, fields: Record<string, unknown>) {
  console.log(`[localmaxxing] ${title}`)
  for (const [key, value] of Object.entries(fields)) {
    if (value === undefined || value === null || value === '') continue
    console.log(`  ${key}: ${Array.isArray(value) ? value.join(', ') : String(value)}`)
  }
}

function printError(error: unknown) {
  if (error && typeof error === 'object' && 'code' in error && 'hints' in error && 'message' in error) {
    const cliError = error as { code: string; message: string; hints: string[]; details?: unknown }
    console.error(`[localmaxxing:error] ${cliError.code}`)
    console.error(cliError.message)
    if (cliError.hints.length) {
      console.error('Fix:')
      for (const hint of cliError.hints) console.error(`- ${hint}`)
    }
    if (cliError.details !== undefined) {
      console.error('Details:')
      console.error(typeof cliError.details === 'string' ? cliError.details : JSON.stringify(cliError.details, null, 2))
    }
    return
  }
  console.error('[localmaxxing:error] unexpected_error')
  console.error(asMessage(error))
}

function redactGoldFields(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(redactGoldFields)
  if (!value || typeof value !== 'object') return value
  const redacted: Record<string, unknown> = {}
  for (const [key, child] of Object.entries(value as Record<string, unknown>)) {
    if (GOLD_FIELD_NAMES.has(key)) continue
    redacted[key] = redactGoldFields(child)
  }
  return redacted
}

function usage() {
  console.log(`LocalMaxxing CLI

Usage:
  lmx context --out localmaxxing-agent-context.json
  lmx auth --key bhk_...
  lmx hardware --out hardware.json
  lmx benchmark run llama.cpp --hf-id Qwen/Qwen3-8B --quantization Q4_K_M --command "llama-bench -m model.gguf" --dry-run
  lmx benchmark run vllm --hf-id Qwen/Qwen3-8B --quantization fp16 --bench-kind throughput --input-len 512 --output-len 128 --dry-run
  lmx benchmark run sglang --hf-id Qwen/Qwen3-8B --quantization fp16 --bench-kind serving --base-url http://127.0.0.1:30000 --dry-run
  lmx benchmark runs list
  lmx benchmark runs show runs/Qwen-Qwen3-8B/run.json
  lmx benchmark runs edit runs/Qwen-Qwen3-8B/run.json --set-json '{"tokSOut":120}'
  lmx benchmark runs rerun runs/Qwen-Qwen3-8B/run.json --dry-run
  lmx benchmark submit benchmark.json --api-key bhk_...
  lmx benchmark dry-run benchmark.json --api-key bhk_...
  lmx eval suite list --out suites.json
  lmx eval suite search reasoning --out reasoning-suites.json
  lmx eval suite show hellaswag --out hellaswag-suite.json
  lmx model search qwen3-8b --out models.json
  lmx eval storage upload traces.jsonl --kind artifact --format jsonl --out artifact-bundle.json
  lmx eval storage download <storageKey> --out traces.jsonl
  lmx eval lm-eval hellaswag --model Qwen/Qwen3-8B --backend hf --hardware hardware.json --dry-run
  lmx eval suite init --slug my-eval --name "My Eval" --category reasoning --kind multiple_choice --out my-eval.json
  lmx eval suite init --slug hellaswag --name "HellaSwag" --category commonsense --runner lm-eval-harness --tasks hellaswag --out hellaswag.json
  lmx eval suite validate my-eval.json
  lmx eval suite submit my-eval.json --api-key bhk_...
  lmx eval run <suiteSlug> --model <hfId> --base-url http://localhost:8000 --hardware hardware.json --submit
  lmx eval run <suiteSlug> --model <hfId> --results lm-eval-results.json --hardware hardware.json --submit
  lmx eval execute <suiteSlug> --model <hfId> --base-url https://public-endpoint.example --hardware hardware.json --submit

Options:
  --api-url <url>          LocalMaxxing origin (default: https://www.localmaxxing.com)
  --api-key <key>          API key, defaults to LMX_API_KEY, then saved config
  --model <hfId>           HuggingFace model ID
  --backend <name>         lm-eval backend name for eval lm-eval (default: hf)
  --model-args <args>      lm-eval --model_args value (default for hf: pretrained=<model>)
  --num-fewshot <n>        lm-eval --num_fewshot override
  --lm-eval-bin <path>     lm-eval executable (default: lm_eval)
  --base-url <url>         OpenAI-compatible model endpoint for custom evals
  --model-api-key <key>    Optional bearer token for the local model endpoint
  --command <cmd>          Benchmark command to execute for benchmark run
  --bench-kind <kind>      Built-in benchmark: vLLM serve/throughput/latency; SGLang serving/offline/one-batch
  --benchmark-bin <path>   Benchmark executable (default: vllm for vLLM, python for SGLang)
  --benchmark-output <path> Engine benchmark JSON output path
  --extra-bench-args <args> Raw extra args appended to built-in benchmark commands
  --hf-id <hfId>           Alias for --model in benchmark commands
  --hardware <path>        JSON hardware object required when submitting
  --quantization <label>   Quantization label, required for benchmark run
  --model-revision <rev>   Optional model revision/branch/commit (default: main)
  --results <path>         Existing lm-eval output JSON for LM_EVAL_HARNESS suites
  --suite-file <path>      Local suite JSON for offline run parsing/testing
  --judge-base-url <url>   OpenAI-compatible judge endpoint for llm_judge suites
  --judge-model <model>    Judge model override
  --judge-api-key <key>    Judge bearer token, defaults to EVAL_JUDGE_API_KEY
  --notes <text>           Optional submission notes
  --kind <kind>            Storage upload kind, usually artifact or dataset
  --format <format>        Storage file format, e.g. json, jsonl, parquet, zip
  --content-type <type>    Storage upload content type override
  --item-count <n>         Optional record/sample count for storage metadata
  --limit <n>              Optional search/list result limit
  --runs-dir <dir>         Saved benchmark runs directory (default: runs)
  --submit                 Upload run to LocalMaxxing
  --dry-run                Validate upload without creating a run
  --out <path>             Write computed payload/result JSON (default: localmaxxing-eval-run.json)

Benchmark submissions:
  benchmark run builds a payload from explicit metric fields, built-in vLLM/SGLang presets, --command, --results, or --base-url measurement.
  benchmark run writes saved runs to runs/<model>/<timestamp>.json by default; use --out to write elsewhere.
  benchmark runs list/show/edit/rerun/submit/dry-run/delete manages saved run JSON files.
  Pass a JSON object matching POST /api/benchmarks, or a lmx-bench result with a payload field.
  Use dry-run first to validate without writing.

Discovery for agents:
  lmx context fetches /api/agent-context with benchmark schemas, endpoints, and methodology tips.
  lmx eval suite list fetches approved eval suites.
  lmx eval suite search <query> filters approved suites by slug, name, category, runner, or task key.
  lmx eval suite show <slug> fetches suiteDoc, task keys, scoring rules, and agentInstructions.
  lmx model search <query> resolves fuzzy model names to canonical HuggingFace IDs.
  lmx eval storage upload/download manages bucket-backed datasets and artifact bundles.
  lmx eval lm-eval <suiteSlug> runs lm_eval, parses its JSON output, and dry-runs or submits the run.

Server-side custom evals:
  eval execute calls POST /api/evals/execute for approved CUSTOM suites and public OpenAI-compatible endpoints.
  Use eval run for localhost/private endpoints because LocalMaxxing cannot reach them directly.

Auth and hardware:
  lmx auth --key bhk_... saves your key to ~/.config/localmaxxing/config.json
  lmx auth --logout clears the saved key
  lmx hardware prints detected hardware JSON; --out writes it to a file

Suite authoring:
  --kind <kind>            qa, multiple_choice, or judge (default: multiple_choice)
  --slug <slug>            Suite slug for init
  --name <name>            Suite display name for init
  --category <category>    Suite category for init
  --runner <runner>        custom or lm-eval-harness for suite init (default: custom)
  --tasks <tasks>          Comma-separated lm-eval task keys for --runner lm-eval-harness
  --scoring-method <name>  exact_match, f1, pass_at_k, or llm_judge (default: exact_match)

Authoring flow for agents:
  1. Read docs/eval-suite-authoring.md for custom suites or docs/lm-eval-compatibility.md for lm-eval suites
  2. Run eval suite init with the closest --kind
  3. Replace generated sample items with researched original questions
  4. Run eval suite validate before submit
  5. Run eval suite submit after validation passes
`)
}

function parseArgs(argv: string[]) {
  const positional: string[] = []
  const opts: Record<string, string | boolean> = {}
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i]
    if (!arg.startsWith('--')) {
      positional.push(arg)
      continue
    }
    const key = arg.slice(2)
    const next = argv[i + 1]
    if (!next || next.startsWith('--')) {
      opts[key] = true
    } else {
      opts[key] = next
      i++
    }
  }
  return { positional, opts }
}

function optString(opts: Record<string, string | boolean>, key: string) {
  const value = opts[key]
  return typeof value === 'string' ? value : undefined
}

async function loadConfig(): Promise<CliConfig> {
  try {
    return JSON.parse(await readFile(CONFIG_FILE, 'utf8')) as CliConfig
  } catch {
    return {}
  }
}

async function saveConfig(config: CliConfig) {
  await mkdir(CONFIG_DIR, { recursive: true })
  await writeFile(CONFIG_FILE, JSON.stringify(config, null, 2) + '\n')
}

async function getApiKey(opts: Record<string, string | boolean>) {
  return optString(opts, 'api-key') ?? process.env.LMX_API_KEY ?? (await loadConfig()).apiKey
}

function requireOpt(opts: Record<string, string | boolean>, key: string) {
  const value = optString(opts, key)
  if (!value) throw new CliError('missing_option', `--${key} is required`, [`Pass --${key} <value>. Run lmx --help for examples.`])
  return value
}

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(url, init)
  } catch (error) {
    throw new CliError('network_error', `Could not reach ${url}: ${asMessage(error)}`, [
      'Check --api-url or endpoint URL.',
      'If this is a local model server, make sure it is running and reachable.',
    ])
  }
  const text = await res.text()
  let body: unknown = null
  try {
    body = text ? JSON.parse(text) as unknown : null
  } catch {
    body = text
  }
  if (!res.ok) {
    const message = body && typeof body === 'object' && 'error' in body ? String((body as { error: unknown }).error) : text
    const hints = [
      res.status === 401 ? 'Check --api-key or LMX_API_KEY.' : '',
      res.status === 400 ? 'Run the relevant validate/dry-run command and fix the reported field.' : '',
      res.status === 404 ? 'Check the suite slug or API URL.' : '',
      res.status === 409 ? 'Choose a different suite slug; this one already exists.' : '',
      res.status === 422 ? 'The suite or run shape is valid JSON but incompatible with the API rules.' : '',
      res.status === 429 ? 'Wait for the rate-limit window before submitting again.' : '',
    ].filter(Boolean)
    throw new CliError('api_error', `${res.status} ${res.statusText}: ${message}`, hints, body)
  }
  return body as T
}

async function readJson<T>(path: string): Promise<T> {
  let text: string
  try {
    text = await readFile(path, 'utf8')
  } catch (error) {
    throw new CliError('file_read_error', `Could not read ${path}: ${asMessage(error)}`, [
      'Check that the path exists and is readable.',
      'Use an absolute path if the file is outside the current directory.',
    ])
  }
  try {
    return JSON.parse(text) as T
  } catch (error) {
    throw new CliError('json_parse_error', `Invalid JSON in ${path}: ${asMessage(error)}`, [
      'Fix the JSON syntax, then rerun the command.',
      'Common issues: trailing commas, comments, or unescaped newlines in strings.',
    ])
  }
}

function hashPrompt(prompt: string) {
  return createHash('sha256').update(prompt).digest('hex')
}

function tryExec(cmd: string, args: string[] = []) {
  try {
    const output = args.length
      ? execFileSync(cmd, args, { stdio: ['pipe', 'pipe', 'pipe'], timeout: 5000 })
      : execSync(cmd, { stdio: ['pipe', 'pipe', 'pipe'], timeout: 5000 })
    return output.toString().trim()
  } catch {
    return null
  }
}

function runStreamingCommand(cmd: string, args: string[]) {
  try {
    execFileSync(cmd, args, { stdio: 'inherit' })
  } catch (error) {
    const status = error && typeof error === 'object' && 'status' in error ? (error as { status?: unknown }).status : undefined
    throw new CliError('command_failed', `${cmd} failed${typeof status === 'number' ? ` with exit code ${status}` : ''}`, [
      'Check that the executable is installed and available on PATH.',
      'For lm-eval, install it with: pip install lm-eval',
      'Rerun with --lm-eval-bin <path> if the executable has a custom location.',
    ])
  }
}

function round1(n: number) {
  return Math.round(n * 10) / 10
}

function detectCpu() {
  if (platform() === 'linux') {
    const out = tryExec("grep -m1 'model name' /proc/cpuinfo")
    return out?.split(':')[1]?.trim()
  }
  if (platform() === 'darwin') return tryExec('sysctl', ['-n', 'machdep.cpu.brand_string']) ?? undefined
  return cpus()[0]?.model
}

function detectNvidiaHardware() {
  const out = tryExec('nvidia-smi', ['--query-gpu=name,memory.total,driver_version', '--format=csv,noheader,nounits'])
  if (!out) return null
  const lines = out.split('\n').filter(Boolean)
  const gpuNames: string[] = []
  let totalVramMb = 0
  for (const line of lines) {
    const [name, vram] = line.split(',').map(s => s.trim())
    if (name) gpuNames.push(name)
    totalVramMb += parseInt(vram ?? '0', 10) || 0
  }
  if (!gpuNames.length) return null
  return {
    hwClass: 'DISCRETE_GPU',
    gpuName: gpuNames[0],
    gpuCount: gpuNames.length,
    vramGb: round1(totalVramMb / 1024),
    cpu: detectCpu(),
    ramGb: round1(totalmem() / 1024 ** 3),
    os: `${platform()} ${release()}`,
  }
}

function detectRocmHardware() {
  const out = tryExec('rocm-smi', ['--showproductname', '--showmeminfo', 'vram', '--csv'])
  if (!out) return null
  const nameMatch = out.match(/Card series:\s*(.+)/i)
  const vramMatch = out.match(/(\d+)\s*MB.*?VRAM/i)
  if (!nameMatch || !vramMatch) return null
  return {
    hwClass: 'DISCRETE_GPU',
    gpuName: nameMatch[1].trim(),
    vramGb: round1(parseInt(vramMatch[1], 10) / 1024),
    cpu: detectCpu(),
    ramGb: round1(totalmem() / 1024 ** 3),
    os: `${platform()} ${release()}`,
  }
}

function detectAppleHardware() {
  const out = tryExec('system_profiler', ['SPHardwareDataType'])
  if (!out) return null
  const chipMatch = out.match(/Chip:\s*(.+)/i)
  const memMatch = out.match(/Memory:\s*(\d+)\s*GB/i)
  if (!chipMatch) return null
  const chip = chipMatch[1].trim()
  const parts = chip.replace('Apple ', '').split(/\s+/)
  return {
    hwClass: 'UNIFIED',
    chipVendor: 'Apple',
    chipFamily: parts[0] ?? chip,
    chipVariant: parts.slice(1).join(' ') || 'base',
    unifiedMemoryGb: memMatch ? parseInt(memMatch[1], 10) : round1(totalmem() / 1024 ** 3),
    cpu: chip,
    os: `darwin ${release()}`,
  }
}

function detectHardware() {
  if (platform() === 'darwin') return detectAppleHardware() ?? detectCpuOnlyHardware()
  return detectNvidiaHardware() ?? detectRocmHardware() ?? detectCpuOnlyHardware()
}

function detectCpuOnlyHardware() {
  return {
    hwClass: 'CPU_ONLY',
    cpu: detectCpu() ?? cpus()[0]?.model ?? 'Unknown CPU',
    ramGb: round1(totalmem() / 1024 ** 3),
    os: `${platform()} ${release()}`,
  }
}

function textField(value: unknown) {
  return typeof value === 'string' && value.trim() ? value : undefined
}

function firstTextField(source: Record<string, unknown>, fields: string[]) {
  for (const field of fields) {
    const value = textField(source[field])
    if (value) return value
  }
  return undefined
}

function benchmarkText(raw: unknown, payload: Record<string, unknown>, kind: 'prompt' | 'output') {
  const fields = kind === 'prompt'
    ? ['prompt', 'promptText', 'input', 'inputText']
    : ['output', 'outputText', 'generatedText', 'completion', 'response', 'text', 'reasoning_content']
  const payloadText = firstTextField(payload, fields)
  if (payloadText) return payloadText
  if (raw && typeof raw === 'object' && !Array.isArray(raw)) {
    const rawText = firstTextField(raw as Record<string, unknown>, fields)
    if (rawText) return rawText
  }
  return undefined
}

function hasEstimatedOutputTokens(raw: unknown, payload: Record<string, unknown>) {
  if (payload.outputTokensEstimated === true) return true
  if (raw && typeof raw === 'object' && !Array.isArray(raw) && (raw as Record<string, unknown>).outputTokensEstimated === true) return true
  const engineFlags = payload.engineFlags
  if (engineFlags && typeof engineFlags === 'object' && !Array.isArray(engineFlags)) {
    const extraFlags = (engineFlags as Record<string, unknown>).extraFlags
    return typeof extraFlags === 'string' && /outputTokensEstimated=true/.test(extraFlags)
  }
  return false
}

function needsTokenFallback(value: unknown, estimated = false) {
  return estimated || typeof value !== 'number' || !Number.isFinite(value) || value <= 0
}

function appendTokenSource(payload: Record<string, unknown>, source: string) {
  const engineFlags = payload.engineFlags
  if (!engineFlags || typeof engineFlags !== 'object' || Array.isArray(engineFlags)) return
  const flags = engineFlags as Record<string, unknown>
  const existing = typeof flags.extraFlags === 'string' ? flags.extraFlags : ''
  flags.extraFlags = existing ? `${existing};tokenizerSource=${source}` : `tokenizerSource=${source}`
}

async function importAutoTokenizer() {
  try {
    const dynamicImport = new Function('specifier', 'return import(specifier)') as (specifier: string) => Promise<unknown>
    const mod = await dynamicImport('@huggingface/transformers') as { AutoTokenizer?: { from_pretrained: (...args: unknown[]) => Promise<unknown> } }
    if (!mod.AutoTokenizer) throw new Error('AutoTokenizer export not found')
    return mod.AutoTokenizer
  } catch (error) {
    throw new CliError('tokenizer_dependency_missing', `Could not load @huggingface/transformers: ${asMessage(error)}`, [
      'Run npm install in localmaxxing-cli before using HF tokenizer fallback.',
      'Or provide engine-reported outputTokens in the benchmark payload.',
    ])
  }
}

async function hfJson(path: string, revision?: string) {
  const ref = encodeURIComponent(revision ?? 'main')
  const res = await fetch(`https://huggingface.co/${path}/raw/${ref}/config.json`)
  if (!res.ok) return null
  return await res.json() as Record<string, unknown>
}

function parentTokenizerId(config: Record<string, unknown> | null) {
  const value = config?.tokenizer_name ?? config?.base_model_name_or_path
  return typeof value === 'string' && /^[\w][\w.-]*\/[\w][\w.@-]*$/.test(value) ? value : undefined
}

async function countWithTokenizer(tokenizer: unknown, text: string) {
  const t = tokenizer as { encode?: (input: string, options?: unknown) => unknown; call?: unknown }
  if (typeof t.encode === 'function') {
    const encoded = await t.encode(text, { add_special_tokens: false })
    if (Array.isArray(encoded)) return encoded.length
  }
  if (typeof tokenizer === 'function') {
    const encoded = await (tokenizer as (input: string, options?: unknown) => Promise<unknown>)(text, { add_special_tokens: false })
    const ids = encoded && typeof encoded === 'object' ? (encoded as Record<string, unknown>).input_ids : undefined
    if (Array.isArray(ids)) return ids.length
    if (ids && typeof ids === 'object' && 'size' in ids && Array.isArray((ids as { size?: unknown }).size)) {
      const size = (ids as { size: number[] }).size
      return size[size.length - 1]
    }
  }
  throw new Error('Loaded tokenizer did not return token IDs')
}

async function hfTokenCount(hfId: string, revision: string | undefined, text: string) {
  const AutoTokenizer = await importAutoTokenizer()
  try {
    const tokenizer = await AutoTokenizer.from_pretrained(hfId, revision ? { revision } : undefined)
    return { count: await countWithTokenizer(tokenizer, text), source: 'hf-tokenizer' }
  } catch {
    const parent = parentTokenizerId(await hfJson(hfId, revision))
    if (!parent) throw new Error(`No tokenizer found for ${hfId}`)
    const tokenizer = await AutoTokenizer.from_pretrained(parent)
    return { count: await countWithTokenizer(tokenizer, text), source: 'parent-hf-tokenizer' }
  }
}

async function normalizeBenchmarkPayload(raw: unknown, payload: Record<string, unknown>) {
  const hfId = typeof payload.hfId === 'string' ? payload.hfId : undefined
  if (!hfId) return payload
  const revision = typeof payload.modelRevision === 'string' ? payload.modelRevision : undefined
  const normalized = { ...payload }
  const outputText = benchmarkText(raw, normalized, 'output')
  const promptText = benchmarkText(raw, normalized, 'prompt')
  let tokenizerSource: string | undefined

  if (outputText && needsTokenFallback(normalized.outputTokens, hasEstimatedOutputTokens(raw, normalized))) {
    const result = await hfTokenCount(hfId, revision, outputText)
    normalized.outputTokens = result.count
    tokenizerSource = result.source
  }

  if (promptText && needsTokenFallback(normalized.promptTokens)) {
    const result = await hfTokenCount(hfId, revision, promptText)
    normalized.promptTokens = result.count
    tokenizerSource = tokenizerSource ?? result.source
  }

  if (tokenizerSource) appendTokenSource(normalized, tokenizerSource)
  return normalized
}

function optNumber(opts: Record<string, string | boolean>, key: string) {
  const value = optString(opts, key)
  if (value === undefined) return undefined
  const parsed = Number(value)
  if (!Number.isFinite(parsed) || parsed < 0) throw new CliError('invalid_option', `--${key} must be a non-negative number`, [`Pass --${key} <number>.`])
  return parsed
}

function requireBenchmarkModel(opts: Record<string, string | boolean>) {
  return optString(opts, 'hf-id') ?? optString(opts, 'model') ?? (() => { throw new CliError('missing_model', 'benchmark run requires --hf-id or --model', ['Pass the canonical HuggingFace model ID, e.g. --hf-id Qwen/Qwen3-8B.']) })()
}

function normalizeEngineName(value: string | undefined) {
  const raw = value?.trim().toLowerCase()
  if (!raw || raw === 'endpoint') return undefined
  if (['llama', 'llama.cpp', 'llamacpp', 'llama-bench', 'llama.cpp-bench'].includes(raw)) return 'llama.cpp'
  if (['vllm', 'vllm-bench', 'vllm benchmark'].includes(raw)) return 'vllm'
  if (['sglang', 'sgl', 'sglang-bench', 'sglang benchmark'].includes(raw)) return 'sglang'
  return value
}

function normalizedMetricKey(key: string) {
  return key.toLowerCase().replace(/[^a-z0-9]/g, '')
}

function walkNumbers(value: unknown, visit: (key: string, n: number) => void, path = '') {
  if (!value || typeof value !== 'object') return
  if (Array.isArray(value)) {
    for (const child of value) walkNumbers(child, visit, path)
    return
  }
  for (const [key, child] of Object.entries(value as Record<string, unknown>)) {
    const nextPath = path ? `${path}.${key}` : key
    if (typeof child === 'number' && Number.isFinite(child)) visit(nextPath, child)
    else walkNumbers(child, visit, nextPath)
  }
}

function numberFromJsonAliases(value: unknown, aliases: string[]) {
  const normalizedAliases = aliases.map(normalizedMetricKey)
  let found: number | undefined
  walkNumbers(value, (key, n) => {
    if (found !== undefined) return
    const normalized = normalizedMetricKey(key)
    if (normalizedAliases.some(alias => normalized.endsWith(alias) || normalized.includes(alias))) found = n
  })
  return found
}

function parseJsonFromOutput(text: string) {
  const trimmed = text.trim()
  if (!trimmed) return undefined
  try {
    return JSON.parse(trimmed) as unknown
  } catch {}
  for (const [startChar, endChar] of [['{', '}'], ['[', ']']] as const) {
    const start = trimmed.indexOf(startChar)
    const end = trimmed.lastIndexOf(endChar)
    if (start >= 0 && end > start) {
      try {
        return JSON.parse(trimmed.slice(start, end + 1)) as unknown
      } catch {}
    }
  }
  return undefined
}

function firstRegexNumber(text: string, patterns: RegExp[]) {
  for (const pattern of patterns) {
    const match = text.match(pattern)
    if (!match) continue
    const parsed = Number(match[1])
    if (Number.isFinite(parsed)) return parsed
  }
  return undefined
}

function shellArg(value: string | number) {
  const text = String(value)
  if (/^[\w@%+=:,./-]+$/.test(text)) return text
  return `"${text.replace(/(["\\])/g, '\\$1')}"`
}

function appendArg(args: string[], flag: string, value: string | number | undefined) {
  if (value === undefined || value === '') return
  args.push(flag, String(value))
}

function appendRawArgs(args: string[], raw: string | undefined) {
  if (raw?.trim()) args.push(`__RAW__${raw.trim()}`)
}

function commandFromArgs(command: string, args: string[]) {
  return [shellArg(command), ...args.map(arg => arg.startsWith('__RAW__') ? arg.slice('__RAW__'.length) : arg.startsWith('--') ? arg : shellArg(arg))].join(' ')
}

function shellQuote(value: string) {
  if (/^[\w@%+=:,./-]+$/.test(value)) return value
  return `'${value.replace(/'/g, `'"'"'`)}'`
}

function safePathPart(value: string) {
  return value.replace(/[^A-Za-z0-9_.-]+/g, '-').replace(/^-+|-+$/g, '') || 'unknown-model'
}

function timestampForPath(date = new Date()) {
  return date.toISOString().replace(/[-:]/g, '').replace(/\.\d{3}Z$/, 'Z')
}

function benchmarkRunPath(payload: Record<string, unknown>, opts: Record<string, string | boolean>) {
  const runsDir = optString(opts, 'runs-dir') ?? 'runs'
  const model = typeof payload.hfId === 'string' ? payload.hfId : 'unknown-model'
  const prefix = typeof payload.contextLength === 'number' ? `kvcache-${payload.contextLength}-` : ''
  return join(runsDir, safePathPart(model), `${prefix}${timestampForPath()}.json`)
}

function benchmarkPayloadObject(value: unknown) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return undefined
  const obj = value as Record<string, unknown>
  const payload = obj.payload
  if (payload && typeof payload === 'object' && !Array.isArray(payload)) return payload as Record<string, unknown>
  return obj
}

function runNumberField(payload: Record<string, unknown>, key: string) {
  const value = payload[key]
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined
}

async function walkJsonFiles(root: string) {
  const files: string[] = []
  async function walk(dir: string) {
    for (const entry of await readdir(dir, { withFileTypes: true })) {
      const path = join(dir, entry.name)
      if (entry.isDirectory()) await walk(path)
      else if (entry.isFile() && entry.name.toLowerCase().endsWith('.json')) files.push(path)
    }
  }
  try {
    const info = await stat(root)
    if (!info.isDirectory()) throw new CliError('invalid_runs_dir', `${root} is not a directory`, ['Pass --runs-dir <dir> if saved runs are elsewhere.'])
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') return []
    throw error
  }
  await walk(root)
  return files
}

async function benchmarkRunSummaries(opts: Record<string, string | boolean>) {
  const root = optString(opts, 'runs-dir') ?? 'runs'
  const paths = await walkJsonFiles(root)
  const withStats = await Promise.all(paths.map(async path => ({ path, info: await stat(path) })))
  withStats.sort((a, b) => b.info.mtimeMs - a.info.mtimeMs)
  const runs: Record<string, unknown>[] = []
  for (const { path, info } of withStats) {
    let payload: Record<string, unknown> | undefined
    try {
      payload = benchmarkPayloadObject(await readJson<unknown>(path))
    } catch {
      continue
    }
    if (!payload?.hfId && !payload?.engineName) continue
    runs.push({
      path,
      model: payload.hfId,
      engine: payload.engineName,
      mode: payload.benchmarkMode,
      quantization: payload.quantization,
      tokSOut: payload.tokSOut,
      canSubmit: (runNumberField(payload, 'tokSOut') ?? 0) > 0,
      updatedAt: info.mtime.toISOString(),
      showCommand: `lmx benchmark runs show ${shellQuote(path)}`,
      editCommand: `lmx benchmark runs edit ${shellQuote(path)} --set-json '{"notes":"..."}'`,
      rerunCommand: `lmx benchmark runs rerun ${shellQuote(path)}`,
      submitCommand: `lmx benchmark runs submit ${shellQuote(path)}`,
    })
  }
  return { runsDir: root, runs }
}

function resolveBenchmarkRunPath(target: string | undefined, opts: Record<string, string | boolean>) {
  const path = target ?? optString(opts, 'path') ?? optString(opts, 'run')
  if (!path) throw new CliError('missing_run_path', 'Saved benchmark run path is required.', ['Use lmx benchmark runs list, then pass a path to show/edit/rerun/submit.'])
  return path
}

function mergeObject(target: Record<string, unknown>, patch: Record<string, unknown>) {
  for (const [key, value] of Object.entries(patch)) {
    const current = target[key]
    if (value && typeof value === 'object' && !Array.isArray(value) && current && typeof current === 'object' && !Array.isArray(current)) {
      mergeObject(current as Record<string, unknown>, value as Record<string, unknown>)
    } else {
      target[key] = value
    }
  }
}

function benchmarkKind(opts: Record<string, string | boolean>, fallback: string) {
  return (optString(opts, 'bench-kind') ?? optString(opts, 'benchmark') ?? fallback).toLowerCase()
}

function benchmarkOutputPath(opts: Record<string, string | boolean>) {
  return optString(opts, 'benchmark-output') ?? optString(opts, 'bench-output')
}

function buildVllmBenchmarkCommand(opts: Record<string, string | boolean>, hfId: string) {
  const kind = benchmarkKind(opts, optString(opts, 'base-url') ? 'serve' : 'throughput')
  const command = optString(opts, 'benchmark-bin') ?? 'vllm'
  const outputPath = benchmarkOutputPath(opts)
  const servedModel = optString(opts, 'served-model') ?? optString(opts, 'model-name') ?? hfId
  const inputLen = optInt(opts, 'input-len') ?? 512
  const outputLen = optInt(opts, 'output-len') ?? 128
  const numPrompts = optInt(opts, 'num-prompts') ?? 100
  const args = ['bench', kind]

  if (kind === 'serve') {
    appendArg(args, '--backend', optString(opts, 'benchmark-backend') ?? 'openai')
    appendArg(args, '--model', servedModel)
    appendArg(args, '--base-url', optString(opts, 'base-url'))
    appendArg(args, '--host', optString(opts, 'host'))
    appendArg(args, '--port', optInt(opts, 'port'))
    appendArg(args, '--endpoint', optString(opts, 'endpoint'))
    appendArg(args, '--dataset-name', optString(opts, 'dataset-name') ?? 'random')
    appendArg(args, '--dataset-path', optString(opts, 'dataset-path'))
    appendArg(args, '--input-len', inputLen)
    appendArg(args, '--output-len', outputLen)
    appendArg(args, '--num-prompts', numPrompts)
    appendArg(args, '--request-rate', optString(opts, 'request-rate'))
    appendArg(args, '--max-concurrency', optInt(opts, 'max-concurrency'))
    if (outputPath) args.push('--save-result', '--result-filename', outputPath)
  } else if (kind === 'throughput') {
    appendArg(args, '--backend', optString(opts, 'benchmark-backend') ?? 'vllm')
    appendArg(args, '--model', hfId)
    appendArg(args, '--dataset-name', optString(opts, 'dataset-name') ?? 'random')
    appendArg(args, '--dataset-path', optString(opts, 'dataset-path'))
    appendArg(args, '--input-len', inputLen)
    appendArg(args, '--output-len', outputLen)
    appendArg(args, '--num-prompts', numPrompts)
    appendArg(args, '--num-warmups', optInt(opts, 'num-warmups'))
    appendArg(args, '--tensor-parallel-size', optInt(opts, 'tensor-parallel'))
    appendArg(args, '--max-model-len', optInt(opts, 'context-length'))
    if (outputPath) args.push('--output-json', outputPath)
  } else if (kind === 'latency') {
    appendArg(args, '--model', hfId)
    appendArg(args, '--input-len', inputLen)
    appendArg(args, '--output-len', outputLen)
    appendArg(args, '--batch-size', optInt(opts, 'batch-size') ?? 1)
    appendArg(args, '--num-iters-warmup', optInt(opts, 'num-warmups'))
    appendArg(args, '--num-iters', optInt(opts, 'num-iters'))
    appendArg(args, '--tensor-parallel-size', optInt(opts, 'tensor-parallel'))
    appendArg(args, '--max-model-len', optInt(opts, 'context-length'))
    if (outputPath) args.push('--output-json', outputPath)
  } else {
    throw new CliError('unsupported_benchmark_kind', `Unsupported vLLM benchmark kind "${kind}"`, ['Use --bench-kind serve, throughput, or latency.', 'Or pass the exact command with --command.'])
  }

  appendRawArgs(args, optString(opts, 'extra-bench-args'))
  return commandFromArgs(command, args)
}

function buildSglangBenchmarkCommand(opts: Record<string, string | boolean>, hfId: string) {
  const kind = benchmarkKind(opts, optString(opts, 'base-url') ? 'serving' : 'offline')
  const command = optString(opts, 'benchmark-bin') ?? optString(opts, 'python-bin') ?? 'python'
  const outputPath = benchmarkOutputPath(opts)
  const servedModel = optString(opts, 'served-model') ?? optString(opts, 'model-name') ?? hfId
  const inputLen = optInt(opts, 'input-len') ?? 512
  const outputLen = optInt(opts, 'output-len') ?? 128
  const numPrompts = optInt(opts, 'num-prompts') ?? 100
  const args = ['-m']

  if (kind === 'serving' || kind === 'serve') {
    args.push('sglang.bench_serving')
    appendArg(args, '--backend', optString(opts, 'benchmark-backend') ?? 'sglang')
    appendArg(args, '--model', servedModel)
    appendArg(args, '--base-url', optString(opts, 'base-url'))
    appendArg(args, '--host', optString(opts, 'host'))
    appendArg(args, '--port', optInt(opts, 'port'))
    appendArg(args, '--dataset-name', optString(opts, 'dataset-name') ?? 'random')
    appendArg(args, '--random-input-len', inputLen)
    appendArg(args, '--random-output-len', outputLen)
    appendArg(args, '--num-prompts', numPrompts)
    appendArg(args, '--request-rate', optString(opts, 'request-rate'))
    appendArg(args, '--max-concurrency', optInt(opts, 'max-concurrency'))
    if (outputPath) appendArg(args, '--output-file', outputPath)
  } else if (kind === 'offline' || kind === 'offline-throughput') {
    args.push('sglang.bench_offline_throughput')
    appendArg(args, '--model-path', hfId)
    appendArg(args, '--dataset-name', optString(opts, 'dataset-name') ?? 'random')
    appendArg(args, '--num-prompts', numPrompts)
    appendArg(args, '--random-input-len', inputLen)
    appendArg(args, '--random-output-len', outputLen)
  } else if (kind === 'one-batch') {
    args.push('sglang.bench_one_batch')
    appendArg(args, '--model-path', hfId)
    appendArg(args, '--batch-size', optInt(opts, 'batch-size') ?? 1)
    appendArg(args, '--input-len', inputLen)
    appendArg(args, '--output-len', outputLen)
  } else if (kind === 'one-batch-server') {
    args.push('sglang.bench_one_batch_server')
    appendArg(args, '--base-url', optString(opts, 'base-url'))
    appendArg(args, '--model-path', hfId)
    appendArg(args, '--batch-size', optInt(opts, 'batch-size') ?? 1)
    appendArg(args, '--input-len', inputLen)
    appendArg(args, '--output-len', outputLen)
  } else {
    throw new CliError('unsupported_benchmark_kind', `Unsupported SGLang benchmark kind "${kind}"`, ['Use --bench-kind serving, offline, one-batch, or one-batch-server.', 'Or pass the exact command with --command.'])
  }

  appendRawArgs(args, optString(opts, 'extra-bench-args'))
  return commandFromArgs(command, args)
}

function builtInBenchmarkCommand(engineName: string, opts: Record<string, string | boolean>, hfId: string) {
  if (engineName === 'vllm') return buildVllmBenchmarkCommand(opts, hfId)
  if (engineName === 'sglang') return buildSglangBenchmarkCommand(opts, hfId)
  return undefined
}

function parseLlamaBenchTable(text: string) {
  const metrics: Record<string, number> = {}
  for (const line of text.split(/\r?\n/)) {
    const cells = line.split('|').map(cell => cell.trim()).filter(Boolean)
    const testIndex = cells.findIndex(cell => /^(pp|tg)\d+\b/i.test(cell))
    if (testIndex < 0) continue
    const test = cells[testIndex]
    const valueCell = cells.slice(testIndex + 1).find(cell => /^\d+(?:\.\d+)?(?:\s*[+-]|\s*±|\s*$)/.test(cell))
    const value = valueCell ? Number(valueCell.match(/^(\d+(?:\.\d+)?)/)?.[1]) : NaN
    if (!Number.isFinite(value)) continue
    const tokens = Number(test.match(/\d+/)?.[0])
    if (/^pp/i.test(test)) {
      metrics.tokSPrefill = metrics.tokSPrefill ?? value
      if (Number.isFinite(tokens)) metrics.promptTokens = metrics.promptTokens ?? tokens
    }
    if (/^tg/i.test(test)) {
      metrics.tokSOut = metrics.tokSOut ?? value
      if (Number.isFinite(tokens)) metrics.outputTokens = metrics.outputTokens ?? tokens
    }
  }
  return metrics
}

function parseBenchmarkOutput(text: string) {
  const json = parseJsonFromOutput(text)
  const metrics: Record<string, number> = {
    ...parseLlamaBenchTable(text),
  }

  if (json !== undefined) {
    metrics.tokSOut = metrics.tokSOut ?? numberFromJsonAliases(json, ['tokSOut', 'outputTokensPerSecond', 'outputTokenThroughput', 'outputThroughput', 'generationTokensPerSecond', 'decodeThroughput', 'genThroughput'])
    metrics.tokSPrefill = metrics.tokSPrefill ?? numberFromJsonAliases(json, ['tokSPrefill', 'prefillTokensPerSecond', 'promptTokensPerSecond', 'inputTokensPerSecond', 'prefillThroughput', 'promptThroughput'])
    metrics.tokSTotal = metrics.tokSTotal ?? numberFromJsonAliases(json, ['tokSTotal', 'totalTokensPerSecond', 'totalTokenThroughput', 'totalThroughput', 'tokensPerSecond'])
    metrics.ttftMs = metrics.ttftMs ?? numberFromJsonAliases(json, ['ttftMs', 'timeToFirstTokenMs', 'meanTtftMs', 'medianTtftMs', 'meanTtft', 'medianTtft'])
    metrics.peakVramGb = metrics.peakVramGb ?? numberFromJsonAliases(json, ['peakVramGb', 'peakGpuMemoryGb', 'maxVramGb'])
    metrics.promptTokens = metrics.promptTokens ?? numberFromJsonAliases(json, ['promptTokens', 'inputTokens', 'totalInputTokens', 'totalPromptTokens', 'numPromptTokens'])
    metrics.outputTokens = metrics.outputTokens ?? numberFromJsonAliases(json, ['outputTokens', 'completionTokens', 'generatedTokens', 'totalOutputTokens', 'totalGeneratedTokens', 'numOutputTokens'])
  }

  metrics.tokSOut = metrics.tokSOut ?? firstRegexNumber(text, [
    /output\s+token\s+throughput[^\d]*(\d+(?:\.\d+)?)/i,
    /output\s+throughput[^\d]*(\d+(?:\.\d+)?)[^\n]*(?:tok|token)\/s/i,
    /decode\s+throughput[^\d]*(\d+(?:\.\d+)?)[^\n]*(?:tok|token)\/s/i,
    /generation\s+throughput[^\d]*(\d+(?:\.\d+)?)[^\n]*(?:tok|token)\/s/i,
    /generated\s+tokens\s+per\s+second[^\d]*(\d+(?:\.\d+)?)/i,
    /(\d+(?:\.\d+)?)\s+output\s+tokens\/s/i,
  ])
  metrics.tokSPrefill = metrics.tokSPrefill ?? firstRegexNumber(text, [
    /prefill\s+throughput[^\d]*(\d+(?:\.\d+)?)[^\n]*(?:tok|token)\/s/i,
    /prompt\s+throughput[^\d]*(\d+(?:\.\d+)?)[^\n]*(?:tok|token)\/s/i,
    /input\s+token\s+throughput[^\d]*(\d+(?:\.\d+)?)/i,
  ])
  metrics.tokSTotal = metrics.tokSTotal ?? firstRegexNumber(text, [
    /total\s+token\s+throughput[^\d]*(\d+(?:\.\d+)?)/i,
    /total\s+throughput[^\d]*(\d+(?:\.\d+)?)[^\n]*(?:tok|token)\/s/i,
    /(\d+(?:\.\d+)?)\s+total\s+tokens\/s/i,
  ])
  metrics.ttftMs = metrics.ttftMs ?? firstRegexNumber(text, [
    /(?:mean|median)?\s*ttft[^\d]*(\d+(?:\.\d+)?)[^\n]*ms/i,
    /time\s+to\s+first\s+token[^\d]*(\d+(?:\.\d+)?)[^\n]*ms/i,
  ])
  metrics.peakVramGb = metrics.peakVramGb ?? firstRegexNumber(text, [
    /peak\s+(?:gpu\s+)?(?:vram|memory)[^\d]*(\d+(?:\.\d+)?)[^\n]*gb/i,
  ])
  metrics.promptTokens = metrics.promptTokens ?? firstRegexNumber(text, [
    /total\s+(?:input|prompt)\s+tokens[^\d]*(\d+)/i,
    /total\s+num\s+prompt\s+tokens[^\d]*(\d+)/i,
  ])
  metrics.outputTokens = metrics.outputTokens ?? firstRegexNumber(text, [
    /total\s+(?:generated|output)\s+tokens[^\d]*(\d+)/i,
    /total\s+num\s+output\s+tokens[^\d]*(\d+)/i,
  ])
  return metrics
}

function runBenchmarkCommand(commandSnippet: string) {
  const result = spawnSync(commandSnippet, { encoding: 'utf8', shell: true, stdio: ['ignore', 'pipe', 'pipe'], maxBuffer: 50 * 1024 * 1024 })
  const stdout = result.stdout ?? ''
  const stderr = result.stderr ?? ''
  const output = [stdout, stderr].filter(Boolean).join('\n').trim()
  if (result.error || (typeof result.status === 'number' && result.status !== 0)) {
    throw new CliError('benchmark_command_failed', 'Benchmark command failed.', [
      'Check that the benchmark executable is installed and available on PATH.',
      'For llama.cpp, pass a complete llama-bench command with --command.',
      'For vLLM/SGLang, prefer their JSON output if available, then pass --results <path>.',
    ], output || result.error?.message)
  }
  return output
}

function openAiUsageTokens(usage: unknown) {
  if (!usage || typeof usage !== 'object' || Array.isArray(usage)) return {}
  const obj = usage as Record<string, unknown>
  const promptTokens = typeof obj.prompt_tokens === 'number' ? obj.prompt_tokens : undefined
  const outputTokens = typeof obj.completion_tokens === 'number' ? obj.completion_tokens : typeof obj.output_tokens === 'number' ? obj.output_tokens : undefined
  return { promptTokens, outputTokens }
}

async function ensureTokenCount(hfId: string, revision: string | undefined, text: string, known: number | undefined, kind: string) {
  if (known && known > 0) return known
  try {
    return (await hfTokenCount(hfId, revision, text)).count
  } catch (error) {
    throw new CliError('token_count_missing', `Could not determine ${kind} token count: ${asMessage(error)}`, [
      `Pass --${kind === 'prompt' ? 'prompt-tokens' : 'output-tokens'} <n> from the engine benchmark output.`,
      'Or use a model/tokenizer that can be loaded by @huggingface/transformers.',
    ])
  }
}

async function measureOpenAiEndpoint(opts: Record<string, string | boolean>, hfId: string) {
  const baseUrl = requireOpt(opts, 'base-url').replace(/\/$/, '')
  const model = optString(opts, 'served-model') ?? optString(opts, 'model-name') ?? hfId
  const prompt = optString(opts, 'prompt') ?? 'Explain why local inference benchmarks should report prompt prefill throughput, decode throughput, and time to first token.'
  const maxTokens = optInt(opts, 'max-tokens') ?? 256
  const started = Date.now()
  let firstTokenAt: number | undefined
  let completedAt = started
  let outputText = ''
  let usage: unknown

  const res = await fetch(`${baseUrl}/v1/chat/completions`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(optString(opts, 'model-api-key') ? { Authorization: `Bearer ${optString(opts, 'model-api-key')}` } : {}),
    },
    body: JSON.stringify({
      model,
      messages: [{ role: 'user', content: prompt }],
      max_tokens: maxTokens,
      temperature: optNumber(opts, 'temperature') ?? 0,
      stream: !opts['no-stream'],
      ...(!opts['no-stream'] ? { stream_options: { include_usage: true } } : {}),
    }),
  })
  if (!res.ok) throw new CliError('endpoint_benchmark_failed', `OpenAI-compatible endpoint returned ${res.status}: ${await res.text()}`, [
    'Check --base-url, --served-model, and --model-api-key.',
    'Confirm the endpoint supports POST /v1/chat/completions.',
  ])

  if (opts['no-stream']) {
    const json = await res.json() as Record<string, unknown>
    const choices = Array.isArray(json.choices) ? json.choices : []
    const first = choices[0] && typeof choices[0] === 'object' ? choices[0] as Record<string, unknown> : {}
    const message = first.message && typeof first.message === 'object' ? first.message as Record<string, unknown> : {}
    outputText = typeof message.content === 'string' && message.content ? message.content
      : typeof message.reasoning_content === 'string' ? message.reasoning_content : ''
    usage = json.usage
    completedAt = Date.now()
  } else {
    if (!res.body) throw new CliError('endpoint_stream_missing', 'Endpoint response did not include a stream body.', ['Retry with --no-stream, or use an OpenAI-compatible endpoint that supports streaming.'])
    const reader = res.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split(/\r?\n/)
      buffer = lines.pop() ?? ''
      for (const line of lines) {
        if (!line.startsWith('data:')) continue
        const data = line.slice(5).trim()
        if (!data || data === '[DONE]') continue
        const chunk = JSON.parse(data) as Record<string, unknown>
        if (chunk.usage) usage = chunk.usage
        const choices = Array.isArray(chunk.choices) ? chunk.choices : []
        const first = choices[0] && typeof choices[0] === 'object' ? choices[0] as Record<string, unknown> : {}
        const delta = first.delta && typeof first.delta === 'object' ? first.delta as Record<string, unknown> : {}
        const content = typeof delta.content === 'string' && delta.content ? delta.content
          : typeof delta.reasoning_content === 'string' ? delta.reasoning_content : ''
        if (content) {
          firstTokenAt = firstTokenAt ?? Date.now()
          outputText += content
        }
      }
    }
    completedAt = Date.now()
  }

  const revision = optString(opts, 'model-revision') ?? 'main'
  const usageTokens = openAiUsageTokens(usage)
  const promptTokens = optInt(opts, 'prompt-tokens') ?? await ensureTokenCount(hfId, revision, prompt, usageTokens.promptTokens, 'prompt')
  const outputTokens = optInt(opts, 'output-tokens') ?? await ensureTokenCount(hfId, revision, outputText, usageTokens.outputTokens, 'output')
  const ttftMs = firstTokenAt ? firstTokenAt - started : undefined
  const generationMs = firstTokenAt ? Math.max(1, completedAt - firstTokenAt) : Math.max(1, completedAt - started)
  const totalMs = Math.max(1, completedAt - started)

  return {
    prompt,
    outputText,
    promptTokens,
    outputTokens,
    ttftMs,
    tokSOut: round1(outputTokens / (generationMs / 1000)),
    tokSTotal: round1((promptTokens + outputTokens) / (totalMs / 1000)),
    tokSPrefill: ttftMs && ttftMs > 0 ? round1(promptTokens / (ttftMs / 1000)) : undefined,
  }
}

async function benchmarkPayloadFromRun(engineArg: string | undefined, opts: Record<string, string | boolean>) {
  const hfId = requireBenchmarkModel(opts)
  const engineName = normalizeEngineName(optString(opts, 'engine') ?? engineArg)
  if (!engineName) throw new CliError('missing_engine', 'benchmark run requires an engine name', ['Pass it positionally, e.g. lmx benchmark run llama.cpp, or with --engine vllm.'])
  const quantization = requireOpt(opts, 'quantization')
  const hardwarePath = optString(opts, 'hardware')
  const hardware = hardwarePath ? await readJson<unknown>(hardwarePath) : detectHardware()
  const resultsPath = optString(opts, 'results')
  const explicitMetric = optString(opts, 'tok-s-out') !== undefined || optString(opts, 'tok-s-prefill') !== undefined || optString(opts, 'tok-s-total') !== undefined || optString(opts, 'ttft-ms') !== undefined
  const hasBuiltInSignal = optString(opts, 'bench-kind') !== undefined || optString(opts, 'benchmark') !== undefined || benchmarkOutputPath(opts) !== undefined
  const wantsBuiltInCommand = !resultsPath && (hasBuiltInSignal || (!explicitMetric && !optString(opts, 'base-url')))
  const commandSnippet = optString(opts, 'command') ?? (wantsBuiltInCommand ? builtInBenchmarkCommand(engineName, opts, hfId) : undefined)
  const endpointMetrics = !commandSnippet && !resultsPath && optString(opts, 'base-url') ? await measureOpenAiEndpoint(opts, hfId) : undefined
  const commandStdout = resultsPath ? undefined : commandSnippet ? runBenchmarkCommand(commandSnippet) : undefined
  const generatedOutputPath = commandStdout ? benchmarkOutputPath(opts) : undefined
  let generatedOutput: string | undefined
  if (generatedOutputPath) {
    try {
      generatedOutput = await readFile(generatedOutputPath, 'utf8')
    } catch {}
  }
  const commandOutput = resultsPath ? await readFile(resultsPath, 'utf8') : [commandStdout, generatedOutput].filter(Boolean).join('\n') || undefined
  const parsedMetrics = commandOutput ? parseBenchmarkOutput(commandOutput) : {}
  const metrics = { ...parsedMetrics, ...endpointMetrics } as Record<string, number | string | undefined>

  const payload = await normalizeBenchmarkPayload({ ...metrics, payload: metrics }, {
    hfId,
    modelRevision: optString(opts, 'model-revision') ?? 'main',
    hardware,
    engineName,
    engineVersion: optString(opts, 'engine-version'),
    quantization,
    backend: optString(opts, 'backend'),
    promptTokens: optInt(opts, 'prompt-tokens') ?? metrics.promptTokens,
    outputTokens: optInt(opts, 'output-tokens') ?? metrics.outputTokens,
    contextLength: optInt(opts, 'context-length'),
    batchSize: optInt(opts, 'batch-size'),
    prefillTokens: optInt(opts, 'prefill-tokens'),
    tokSOut: optNumber(opts, 'tok-s-out') ?? metrics.tokSOut,
    tokSPrefill: optNumber(opts, 'tok-s-prefill') ?? metrics.tokSPrefill,
    tokSTotal: optNumber(opts, 'tok-s-total') ?? metrics.tokSTotal,
    ttftMs: optNumber(opts, 'ttft-ms') ?? metrics.ttftMs,
    peakVramGb: optNumber(opts, 'peak-vram-gb') ?? metrics.peakVramGb,
    notes: optString(opts, 'notes'),
    ...(endpointMetrics ? { prompt: endpointMetrics.prompt, outputText: endpointMetrics.outputText } : {}),
    engineFlags: {
      ...(commandSnippet ? { commandSnippet } : {}),
      ...(optString(opts, 'base-url') ? { commandSnippet: commandSnippet ?? `OpenAI-compatible endpoint ${optString(opts, 'base-url')}` } : {}),
      concurrency: optInt(opts, 'concurrency'),
      numParallel: optInt(opts, 'num-parallel'),
      tensorParallel: optInt(opts, 'tensor-parallel'),
      gpuLayers: optInt(opts, 'gpu-layers'),
      temperature: optNumber(opts, 'temperature'),
      topP: optNumber(opts, 'top-p'),
      extraFlags: optString(opts, 'extra-flags'),
    },
  })

  if (typeof payload.tokSOut !== 'number' || !Number.isFinite(payload.tokSOut) || payload.tokSOut <= 0) {
    throw new CliError('benchmark_metric_missing', 'Could not determine tokSOut from the benchmark output.', [
      'Pass --tok-s-out <tokens_per_second> explicitly.',
      'For llama-bench, include text table output with a tg<N> row.',
      'For vLLM/SGLang benchmark scripts, prefer JSON output or include "Output token throughput" text.',
    ], commandOutput?.slice(0, 4000))
  }

  return payload
}

function normalizeText(value: string) {
  return value.toLowerCase().replace(/[^\w\s]/g, '').trim()
}

function tokenF1(pred: string, gold: string) {
  const predToks = normalizeText(pred).split(/\s+/).filter(Boolean)
  const goldToks = normalizeText(gold).split(/\s+/).filter(Boolean)
  if (!predToks.length || !goldToks.length) return 0
  const predMap = new Map<string, number>()
  const goldMap = new Map<string, number>()
  for (const token of predToks) predMap.set(token, (predMap.get(token) ?? 0) + 1)
  for (const token of goldToks) goldMap.set(token, (goldMap.get(token) ?? 0) + 1)
  let common = 0
  for (const [token, count] of predMap) common += Math.min(count, goldMap.get(token) ?? 0)
  if (!common) return 0
  const precision = common / predToks.length
  const recall = common / goldToks.length
  return (2 * precision * recall) / (precision + recall)
}

function choiceLabel(index: number) {
  return String.fromCharCode(65 + index)
}

function normalizeChoice(value: string | number, choices?: string[]) {
  if (typeof value === 'number') return choiceLabel(value)
  const normalized = normalizeText(value)
  if (/^[a-z]$/.test(normalized)) return normalized.toUpperCase()
  if (/^\d+$/.test(normalized)) return choiceLabel(Math.max(0, Number(normalized) - 1))
  const choiceIndex = choices?.findIndex(choice => normalizeText(choice) === normalized) ?? -1
  if (choiceIndex >= 0) return choiceLabel(choiceIndex)
  const tokens = normalized.split(/\s+/).filter(Boolean)
  const firstChoiceToken = tokens.find(token => /^[a-z]$/.test(token))
  return firstChoiceToken ? firstChoiceToken.toUpperCase() : normalized
}

function renderPrompt(template: string, item: DatasetItem) {
  let prompt = template.replace(/\{\{input\}\}/g, String(item.input)).replace(/\{\{gold\}\}/g, '')
  if (item.choices) {
    const choices = item.choices.map((choice, i) => `${choiceLabel(i)}. ${choice}`).join('\n')
    prompt = prompt.replace(/\{\{choices\}\}/g, choices)
  }
  return prompt.trim()
}

function renderQuestion(item: DatasetItem) {
  if (!item.choices?.length) return String(item.input).trim()
  return `${item.input}\n\n${item.choices.map((choice, i) => `${choiceLabel(i)}. ${choice}`).join('\n')}`.trim()
}

function firstDatasetDownloadUrl(dataset: NonNullable<EvalTask['dataset']>) {
  if (dataset.downloadUrl) return dataset.downloadUrl
  return downloadUrlFromValue(dataset.downloadUrls)
}

function parseDatasetItems(text: string, url: string, format?: string) {
  if (format === 'jsonl' || url.toLowerCase().endsWith('.jsonl')) {
    return text.split(/\r?\n/).filter(Boolean).map(line => JSON.parse(line) as DatasetItem)
  }
  try {
    return JSON.parse(text) as DatasetItem[]
  } catch (jsonError) {
    try {
      return text.split(/\r?\n/).filter(Boolean).map(line => JSON.parse(line) as DatasetItem)
    } catch {
      throw jsonError
    }
  }
}

async function fetchDatasetItems(url: string, format?: string) {
  let res: Response
  try {
    res = await fetch(url)
  } catch (error) {
    throw new CliError('network_error', `Could not reach dataset URL: ${asMessage(error)}`, ['Check that the dataset download URL is reachable and retry.'])
  }
  const text = await res.text()
  if (!res.ok) throw new CliError('dataset_download_failed', `Dataset download failed: ${res.status} ${res.statusText}`, ['Signed dataset download URLs can expire; retry eval run to fetch a fresh run bundle.'], text)
  try {
    return parseDatasetItems(text, url, format)
  } catch (error) {
    throw new CliError('dataset_parse_failed', `Could not parse dataset downloaded from ${url}: ${asMessage(error)}`, ['Use JSON array or JSONL dataset files.'])
  }
}

async function loadDataset(dataset: NonNullable<EvalTask['dataset']>) {
  if (dataset.source === 'inline') return dataset.items ?? []
  const downloadUrl = firstDatasetDownloadUrl(dataset)
  if (downloadUrl) return fetchDatasetItems(downloadUrl, dataset.format)
  if (dataset.source === 'url') {
    if (!dataset.url) throw new CliError('dataset_missing_url', 'url dataset missing url', ['Add dataset.url to the suite task.'])
    return fetchDatasetItems(dataset.url, dataset.format)
  }
  if (dataset.source === 'huggingface') {
    if (!dataset.hfPath) throw new CliError('dataset_missing_hf_path', 'huggingface dataset missing hfPath', ['Add dataset.hfPath to the suite task.'])
    const name = dataset.hfName ?? 'default'
    const split = dataset.split ?? 'test'
    const url = `https://datasets-server.huggingface.co/rows?dataset=${encodeURIComponent(dataset.hfPath)}&config=${encodeURIComponent(name)}&split=${encodeURIComponent(split)}&offset=0&limit=500`
    const json = await fetchJson<{ rows: Array<{ row: Record<string, unknown> }> }>(url)
    return json.rows.map(row => ({
      input: String(row.row.question ?? row.row.input ?? row.row.prompt ?? ''),
      gold: String(row.row.answer ?? row.row.gold ?? row.row.label ?? ''),
      choices: Array.isArray(row.row.choices) ? row.row.choices.map(String) : undefined,
    }))
  }
  throw new CliError('dataset_source_unknown', `Unknown dataset source "${dataset.source}"`, ['Use one of: inline, url, huggingface.'])
}

async function callOpenAIChat(baseUrl: string, model: string, prompt: string, options: { apiKey?: string; maxTokens?: number; temperature?: number; topP?: number; stop?: string[] }) {
  const res = await fetch(baseUrl.replace(/\/$/, '') + '/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(options.apiKey ? { Authorization: `Bearer ${options.apiKey}` } : {}),
    },
    body: JSON.stringify({
      model,
      messages: [{ role: 'user', content: prompt }],
      max_tokens: options.maxTokens ?? 256,
      temperature: options.temperature ?? 0,
      top_p: options.topP ?? 1,
      ...(options.stop?.length ? { stop: options.stop } : {}),
    }),
  })
  if (!res.ok) throw new CliError('model_server_error', `OpenAI-compatible server returned ${res.status}: ${await res.text()}`, [
    'Check --base-url and --model.',
    'Confirm the server supports POST /v1/chat/completions.',
    'If the server requires auth, pass --model-api-key.',
  ])
  const json = await res.json() as { choices?: Array<{ message?: { content?: string; reasoning_content?: string } }> }
  return (json.choices?.[0]?.message?.content ?? json.choices?.[0]?.message?.reasoning_content)?.trim() ?? ''
}

function parseJudgeResponse(raw: string) {
  const match = raw.match(/\{[\s\S]*\}/)
  if (match) {
    try {
      const parsed = JSON.parse(match[0]) as { score?: unknown; rationale?: unknown }
      const score = typeof parsed.score === 'number' ? parsed.score : Number(parsed.score)
      if (Number.isFinite(score)) return { score: Math.max(0, Math.min(1, score)), rationale: String(parsed.rationale ?? raw) }
    } catch {}
  }
  const numeric = raw.match(/(?:score\D+)?([01](?:\.\d+)?)/i)
  const score = numeric ? Number(numeric[1]) : NaN
  if (!Number.isFinite(score)) throw new CliError('judge_score_parse_failed', `Judge did not return a parseable score: ${raw}`, [
    'Adjust the judge prompt/rubric to request strict JSON with a numeric score from 0 to 1.',
    'Check --judge-base-url, --judge-model, and --judge-api-key.',
  ])
  return { score: Math.max(0, Math.min(1, score)), rationale: raw }
}

async function judgeResponse(args: { baseUrl: string; model: string; apiKey?: string; task: EvalTask; item: DatasetItem; prompt: string; response: string }) {
  const rubric = args.item.rubric ?? 'Score the response from 0 to 1 for correctness and quality.'
  const judgePrompt = `You are grading a model response. Return strict JSON: {"score": number, "rationale": string}.\n\nRubric:\n${rubric}\n\nQuestion:\n${renderQuestion(args.item)}\n\nPrompt sent to model:\n${args.prompt}\n\nModel response:\n${args.response}\n\nReference answer, if any:\n${args.item.referenceAnswer ?? ''}`
  const raw = await callOpenAIChat(args.baseUrl, args.model, judgePrompt, { apiKey: args.apiKey, maxTokens: 512, temperature: 0 })
  return parseJudgeResponse(raw)
}

function computeAggregate(doc: SuiteDoc, scores: Record<string, number>) {
  const present = doc.tasks.filter(task => scores[task.key] !== undefined)
  if (!present.length) throw new Error('No matching task scores produced')
  if (doc.aggregation === 'min') return Math.min(...present.map(task => scores[task.key]))
  if (doc.aggregation === 'max') return Math.max(...present.map(task => scores[task.key]))
  if (doc.aggregation === 'mean') return present.reduce((sum, task) => sum + scores[task.key], 0) / present.length
  const totalWeight = present.reduce((sum, task) => sum + (task.weight ?? 1), 0)
  return present.reduce((sum, task) => sum + scores[task.key] * (task.weight ?? 1), 0) / totalWeight
}

function validateSuitePayload(payload: SuiteCreatePayload) {
  const errors: string[] = []
  if (!/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(payload.slug) || payload.slug.length < 3 || payload.slug.length > 64) {
    errors.push('slug must be 3-64 lowercase alphanumeric characters with hyphens')
  }
  if (!payload.name || payload.name.length > 256) errors.push('name is required and must be <= 256 chars')
  if (!payload.category || payload.category.length > 64) errors.push('category is required and must be <= 64 chars')
  if (payload.runner !== 'CUSTOM' && payload.runner !== 'LM_EVAL_HARNESS') errors.push('runner must be CUSTOM or LM_EVAL_HARNESS')
  const expectedDocRunner = payload.runner === 'CUSTOM' ? 'custom' : 'lm-eval-harness'
  if (payload.suiteDoc.runner !== expectedDocRunner) errors.push(`suiteDoc.runner must be ${expectedDocRunner}`)
  if (!['exact_match', 'f1', 'pass_at_k', 'perplexity', 'llm_judge'].includes(payload.suiteDoc.scoringMethod)) errors.push('suiteDoc.scoringMethod is invalid')
  if (payload.suiteDoc.scoringMethod === 'perplexity') errors.push('perplexity scoring is not supported yet')
  if (!Array.isArray(payload.suiteDoc.tasks) || payload.suiteDoc.tasks.length === 0) errors.push('suiteDoc.tasks must contain at least one task')
  if (payload.suiteDoc.tasks.length > 100) errors.push('suiteDoc.tasks cannot exceed 100 tasks')

  const keys = new Set<string>()
  payload.suiteDoc.tasks.forEach((task, taskIndex) => {
    const prefix = `suiteDoc.tasks[${taskIndex}]`
    if (!task.key) errors.push(`${prefix}.key is required`)
    if (keys.has(task.key)) errors.push(`${prefix}.key duplicates "${task.key}"`)
    keys.add(task.key)
    if (!task.displayName) errors.push(`${prefix}.displayName is required`)
    if (payload.runner === 'CUSTOM') {
      if (!task.promptTemplate) errors.push(`${prefix}.promptTemplate is required for CUSTOM suites`)
      if (!task.dataset) errors.push(`${prefix}.dataset is required for CUSTOM suites`)
    }
    if (task.taskType === 'multiple_choice' && task.dataset?.source === 'inline') {
      task.dataset.items?.forEach((item, itemIndex) => {
        if (!item.choices?.length) errors.push(`${prefix}.dataset.items[${itemIndex}].choices is required for multiple_choice tasks`)
      })
    }
    if (task.dataset?.source === 'inline') {
      if (!task.dataset.items?.length) errors.push(`${prefix}.dataset.items is required for inline datasets`)
      task.dataset.items?.forEach((item, itemIndex) => {
        if (!item.input) errors.push(`${prefix}.dataset.items[${itemIndex}].input is required`)
        if (item.gold === undefined && !item.rubric) errors.push(`${prefix}.dataset.items[${itemIndex}] needs gold or rubric`)
      })
    }
  })

  if (errors.length) throw new CliError('suite_validation_failed', 'Suite validation failed.', [
    'Edit the suite JSON file and rerun eval suite validate.',
    'For custom suites, every task needs promptTemplate and dataset.',
    'For inline multiple-choice datasets, every item needs choices and gold.',
  ], errors)
}

function buildSuiteTemplate(opts: Record<string, string | boolean>): SuiteCreatePayload {
  const kind = optString(opts, 'kind') ?? 'multiple_choice'
  const slug = requireOpt(opts, 'slug')
  const name = requireOpt(opts, 'name')
  const category = requireOpt(opts, 'category')
  const runner = optString(opts, 'runner') ?? 'custom'
  const scoringMethod = (optString(opts, 'scoring-method') ?? 'exact_match') as ScoringMethod

  if (runner === 'lm-eval-harness') {
    const tasks = requireOpt(opts, 'tasks').split(',').map(task => task.trim()).filter(Boolean)
    if (!tasks.length) throw new CliError('missing_lm_eval_tasks', '--tasks must include at least one lm-eval task key', ['Pass --tasks task_a,task_b for lm-eval-harness suite init.'])
    return {
      slug,
      name,
      description: 'LM-Eval Harness suite. Run with lm-eval-harness, then upload the output JSON with the LocalMaxxing CLI.',
      category,
      runner: 'LM_EVAL_HARNESS',
      version: '1.0',
      suiteDoc: {
        version: '1.0',
        runner: 'lm-eval-harness',
        scoringMethod,
        higherIsBetter: true,
        aggregation: 'weighted_mean',
        tasks: tasks.map(task => ({
          key: task,
          displayName: task.replace(/_/g, ' '),
          taskType: 'multiple_choice',
          weight: 1,
          higherIsBetter: true,
        })),
      },
    }
  }

  if (runner !== 'custom') throw new Error('--runner must be custom or lm-eval-harness')

  const base = {
    slug,
    name,
    description: `Custom ${kind.replace(/_/g, ' ')} eval suite. Replace the sample items before submitting.`,
    category,
    runner: 'CUSTOM' as const,
    version: '1.0',
  }

  if (kind === 'qa') {
    return {
      ...base,
      suiteDoc: {
        version: '1.0',
        runner: 'custom',
        scoringMethod: 'exact_match',
        higherIsBetter: true,
        aggregation: 'weighted_mean',
        runConfig: { temperature: 0 },
        tasks: [{
          key: 'qa',
          displayName: 'Short-answer QA',
          taskType: 'qa',
          weight: 1,
          promptTemplate: 'Answer the question with only the final answer.\n\nQuestion: {{input}}',
          maxNewTokens: 64,
          dataset: { source: 'inline', items: [{ input: 'What is 2 + 2?', gold: '4' }] },
        }],
      },
    }
  }

  if (kind === 'judge') {
    return {
      ...base,
      suiteDoc: {
        version: '1.0',
        runner: 'custom',
        scoringMethod: 'llm_judge',
        higherIsBetter: true,
        aggregation: 'weighted_mean',
        runConfig: { temperature: 0.7 },
        tasks: [{
          key: 'judge_quality',
          displayName: 'Judge-scored response quality',
          taskType: 'judge',
          weight: 1,
          promptTemplate: 'Write a concise answer to the following prompt.\n\n{{input}}',
          maxNewTokens: 512,
          dataset: {
            source: 'inline',
            items: [{
              input: 'Explain why local inference benchmarks should include both speed and quality metrics.',
              referenceAnswer: 'A strong answer mentions that speed alone can hide regressions in reasoning, instruction following, or output quality.',
              rubric: 'Score 0 to 1. Reward clear explanation, mention of speed/quality tradeoffs, and relevance to local inference benchmarking.',
            }],
          },
        }],
      },
    }
  }

  if (kind !== 'multiple_choice') throw new Error('--kind must be qa, multiple_choice, or judge')
  return {
    ...base,
    suiteDoc: {
      version: '1.0',
      runner: 'custom',
      scoringMethod: 'exact_match',
      higherIsBetter: true,
      aggregation: 'weighted_mean',
      runConfig: { temperature: 0 },
      tasks: [{
        key: 'multiple_choice',
        displayName: 'Multiple choice questions',
        taskType: 'multiple_choice',
        weight: 1,
        promptTemplate: 'Choose the correct answer. Reply with only A, B, C, or D.\n\n{{input}}\n\n{{choices}}',
        maxNewTokens: 8,
        dataset: {
          source: 'inline',
          items: [{
            input: 'Which number is even?',
            choices: ['3', '5', '8', '9'],
            gold: 'C',
          }],
        },
      }],
    },
  }
}

async function runCustomLocal(suite: SuiteResponse, opts: Record<string, string | boolean>) {
  const model = optString(opts, 'model')
  const baseUrl = optString(opts, 'base-url') ?? 'http://localhost:8000'
  if (!model) throw new CliError('missing_model', '--model is required', ['Pass --model <HuggingFace model id>.'])

  const doc = suite.suiteDoc
  const scores: Record<string, TaskScore> = {}
  const artifacts: Artifact[] = []
  const judgeBaseUrl = optString(opts, 'judge-base-url') ?? doc.judge?.baseUrl
  const judgeModel = optString(opts, 'judge-model') ?? doc.judge?.model
  const judgeApiKey = optString(opts, 'judge-api-key') ?? process.env.EVAL_JUDGE_API_KEY

  if (doc.scoringMethod === 'llm_judge' && (!judgeBaseUrl || !judgeModel)) {
    throw new CliError('judge_config_missing', 'llm_judge suites require --judge-base-url and --judge-model, or suiteDoc.judge defaults', [
      'Pass --judge-base-url and --judge-model.',
      'If the judge requires auth, pass --judge-api-key or set EVAL_JUDGE_API_KEY.',
    ])
  }

  printInfo('custom_eval_start', {
    suite: suite.slug,
    tasks: doc.tasks.length,
    model,
    baseUrl,
    scoringMethod: doc.scoringMethod,
  })

  for (const task of doc.tasks) {
    if (!task.promptTemplate || !task.dataset) throw new CliError('task_not_runnable', `Task "${task.key}" requires promptTemplate and dataset`, ['Fix the suite JSON or use an LM_EVAL_HARNESS suite for external lm-eval tasks.'])
    const items = await loadDataset(task.dataset).catch((error) => {
      throw new CliError('dataset_load_failed', `Failed to load dataset for task "${task.key}": ${asMessage(error)}`, ['Check dataset source fields and network access.'], error instanceof CliError ? error.details : undefined)
    })
    let totalScore = 0
    let counted = 0
    for (let itemIndex = 0; itemIndex < items.length; itemIndex++) {
      const item = items[itemIndex]
      const prompt = renderPrompt(task.promptTemplate, item)
      const question = renderQuestion(item)
      const started = Date.now()
      try {
        const response = await callOpenAIChat(baseUrl, model, prompt, {
          apiKey: optString(opts, 'model-api-key'),
          maxTokens: task.maxNewTokens ?? 256,
          temperature: doc.runConfig?.temperature ?? 0,
          topP: doc.runConfig?.topP ?? 1,
          stop: task.stopSequences,
        })
        let score = 0
        let judgeRationale: string | undefined
        if (doc.scoringMethod === 'exact_match') {
          if (item.gold === undefined) throw new CliError('item_missing_gold', `Task "${task.key}" item ${itemIndex} is missing gold answer for exact_match scoring`, ['Add gold to the dataset item or use llm_judge with a rubric.'])
          const multipleChoice = task.taskType === 'multiple_choice' || !!item.choices?.length
          score = multipleChoice
            ? normalizeChoice(response, item.choices) === normalizeChoice(item.gold, item.choices) ? 1 : 0
            : normalizeText(response) === normalizeText(String(item.gold)) ? 1 : 0
        } else if (doc.scoringMethod === 'f1') {
          if (item.gold === undefined) throw new CliError('item_missing_gold', `Task "${task.key}" item ${itemIndex} is missing gold answer for f1 scoring`, ['Add gold to the dataset item or use llm_judge with a rubric.'])
          score = tokenF1(response, String(item.gold))
        } else if (doc.scoringMethod === 'llm_judge') {
          const judged = await judgeResponse({ baseUrl: judgeBaseUrl!, model: judgeModel!, apiKey: judgeApiKey, task, item, prompt, response })
          score = judged.score
          judgeRationale = judged.rationale
        } else {
          throw new CliError('scoring_method_unsupported', `CLI custom evals do not support scoringMethod "${doc.scoringMethod}" yet`, ['Use exact_match, f1, or llm_judge for custom CLI execution.'])
        }
        totalScore += score
        counted++
        artifacts.push({ taskKey: task.key, itemIndex, promptHash: hashPrompt(prompt), question, prompt, response, score, judgeModel, judgeScore: doc.scoringMethod === 'llm_judge' ? score : undefined, judgeRationale, latencyMs: Date.now() - started })
      } catch (error) {
        counted++
        artifacts.push({ taskKey: task.key, itemIndex, promptHash: hashPrompt(prompt), question, prompt, latencyMs: Date.now() - started, error: error instanceof Error ? error.message : 'Eval item failed' })
      }
    }
    if (counted > 0) scores[task.key] = { score: totalScore / counted, nSamples: counted, nShots: task.nShots }
    const failures = artifacts.filter(artifact => artifact.taskKey === task.key && artifact.error).length
    printInfo('task_complete', {
      task: task.key,
      samples: counted,
      failures,
      score: scores[task.key]?.score,
    })
  }

  return { scores, artifacts, aggregate: computeAggregate(doc, Object.fromEntries(Object.entries(scores).map(([key, result]) => [key, result.score]))) }
}

function normalizeMetricScore(metricName: string, value: number) {
  if (!Number.isFinite(value)) return undefined
  if (value >= 0 && value <= 1) return value
  if ((metricName.startsWith('rouge') || metricName.includes('bleu') || metricName.includes('chrf')) && value >= 0 && value <= 100) {
    return value / 100
  }
  return undefined
}

function scoreFromLmEvalTask(value: unknown, scoringMethod: ScoringMethod) {
  if (!value || typeof value !== 'object') return undefined
  const obj = value as Record<string, unknown>
  for (const key of LMEVAL_METRIC_CANDIDATES_BY_SCORING[scoringMethod] ?? []) {
    if (typeof obj[key] === 'number') return normalizeMetricScore(key, obj[key])
  }
  for (const [key, raw] of Object.entries(obj)) {
    if (typeof raw !== 'number') continue
    if (key.endsWith('_stderr') || key.includes('stderr')) continue
    const normalized = normalizeMetricScore(key, raw)
    if (normalized !== undefined) return normalized
  }
  return undefined
}

function availableMetricNames(value: unknown) {
  if (!value || typeof value !== 'object') return []
  return Object.entries(value as Record<string, unknown>)
    .filter(([, raw]) => typeof raw === 'number')
    .map(([key]) => key)
}

function lmEvalResultForTask(raw: Record<string, unknown>, taskKey: string) {
  const results = raw.results && typeof raw.results === 'object' ? raw.results as Record<string, unknown> : raw
  const groups = raw.groups && typeof raw.groups === 'object' ? raw.groups as Record<string, unknown> : {}
  return results[taskKey] ?? groups[taskKey] ?? results[taskKey.replace(/-/g, '_')] ?? groups[taskKey.replace(/-/g, '_')]
}

async function loadLmEvalResults(path: string, suite: SuiteResponse) {
  const raw = await readJson<Record<string, unknown>>(path)
  const scores: Record<string, TaskScore> = {}
  printInfo('lm_eval_parse_start', {
    suite: suite.slug,
    tasks: suite.suiteDoc.tasks.length,
    scoringMethod: suite.suiteDoc.scoringMethod,
    results: path,
  })
  for (const task of suite.suiteDoc.tasks) {
    const taskResult = lmEvalResultForTask(raw, task.key)
    const score = scoreFromLmEvalTask(taskResult, suite.suiteDoc.scoringMethod)
    if (score === undefined) throw new CliError('lm_eval_metric_not_found', `Could not find lm-eval score for task "${task.key}"`, [
      `Ensure the lm-eval output contains results.${task.key} or groups.${task.key}.`,
      `For scoringMethod ${suite.suiteDoc.scoringMethod}, expected one of: ${(LMEVAL_METRIC_CANDIDATES_BY_SCORING[suite.suiteDoc.scoringMethod] ?? []).join(', ')}.`,
      'If the task uses a different metric, edit the suite scoringMethod or extend the CLI metric mapping.',
    ], {
      taskKey: task.key,
      availableMetrics: availableMetricNames(taskResult),
    })
    scores[task.key] = { score, nShots: task.nShots ?? suite.suiteDoc.runConfig?.fewShot }
    printInfo('lm_eval_task_score', { task: task.key, score })
  }
  return { scores, artifacts: [] as Artifact[], aggregate: computeAggregate(suite.suiteDoc, Object.fromEntries(Object.entries(scores).map(([key, result]) => [key, result.score]))) }
}

async function submitJson(apiUrl: string, apiKey: string, endpoint: string, payload: unknown) {
  return fetchJson<unknown>(`${apiUrl}${endpoint}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${apiKey}` },
    body: JSON.stringify(payload),
  })
}

async function fetchAuthedJson<T>(apiUrl: string, apiKey: string | undefined, endpoint: string): Promise<T> {
  return fetchJson<T>(`${apiUrl}${endpoint}`, apiKey ? { headers: { Authorization: `Bearer ${apiKey}` } } : undefined)
}

async function writeOrPrintJson(title: string, opts: Record<string, string | boolean>, payload: unknown) {
  const outPath = optString(opts, 'out')
  if (!outPath) {
    console.log(JSON.stringify(payload, null, 2))
    return
  }
  await writeFile(outPath, JSON.stringify(payload, null, 2) + '\n')
  printInfo(`${title}_written`, { path: outPath })
}

async function handleContextCommand(opts: Record<string, string | boolean>) {
  const apiUrl = (optString(opts, 'api-url') ?? 'https://www.localmaxxing.com').replace(/\/$/, '')
  const context = await fetchJson<unknown>(`${apiUrl}/api/agent-context`)
  await writeOrPrintJson('context', opts, context)
}

function optInt(opts: Record<string, string | boolean>, key: string) {
  const value = optString(opts, key)
  if (!value) return undefined
  const parsed = Number.parseInt(value, 10)
  if (!Number.isFinite(parsed) || parsed < 0) throw new CliError('invalid_option', `--${key} must be a non-negative integer`, [`Pass --${key} <number>.`])
  return parsed
}

function defaultContentType(format: string | undefined) {
  if (format === 'json') return 'application/json'
  if (format === 'jsonl') return 'application/jsonl'
  if (format === 'txt') return 'text/plain'
  if (format === 'zip') return 'application/zip'
  return 'application/octet-stream'
}

function suitesFromResponse(value: unknown) {
  if (value && typeof value === 'object' && !Array.isArray(value)) {
    const suites = (value as Record<string, unknown>).suites
    if (Array.isArray(suites)) return suites as Record<string, unknown>[]
  }
  return Array.isArray(value) ? value as Record<string, unknown>[] : []
}

function suiteMatches(suite: Record<string, unknown>, query: string) {
  const haystack = JSON.stringify({
    slug: suite.slug,
    name: suite.name,
    description: suite.description,
    category: suite.category,
    runner: suite.runner,
    tasks: suite.suiteDoc && typeof suite.suiteDoc === 'object' ? (suite.suiteDoc as Record<string, unknown>).tasks : undefined,
  }).toLowerCase()
  return haystack.includes(query.toLowerCase())
}

async function handleModelCommand(action: string | undefined, target: string | undefined, opts: Record<string, string | boolean>) {
  if (action !== 'search') throw new Error('Unknown model command. Use search.')
  const query = target ?? optString(opts, 'q') ?? optString(opts, 'query')
  if (!query) throw new CliError('missing_query', 'model search requires a query', ['Run lmx model search qwen3-8b.'])
  const apiUrl = (optString(opts, 'api-url') ?? 'https://www.localmaxxing.com').replace(/\/$/, '')
  const limit = optString(opts, 'limit') ?? '10'
  const models = await fetchJson<unknown>(`${apiUrl}/api/models/search?q=${encodeURIComponent(query)}&limit=${encodeURIComponent(limit)}`)
  await writeOrPrintJson('models', opts, models)
}

function stringField(value: unknown, key: string) {
  return value && typeof value === 'object' && !Array.isArray(value) && typeof (value as Record<string, unknown>)[key] === 'string'
    ? (value as Record<string, string>)[key]
    : undefined
}

function recordField(value: unknown, key: string) {
  const field = value && typeof value === 'object' && !Array.isArray(value) ? (value as Record<string, unknown>)[key] : undefined
  return field && typeof field === 'object' && !Array.isArray(field) ? field as Record<string, unknown> : undefined
}

function stringHeaders(value: unknown) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return {}
  return Object.fromEntries(Object.entries(value as Record<string, unknown>).map(([key, headerValue]) => [key, String(headerValue)]))
}

async function handleStorageCommand(action: string | undefined, target: string | undefined, opts: Record<string, string | boolean>, forcedKind?: string) {
  const apiUrl = (optString(opts, 'api-url') ?? 'https://www.localmaxxing.com').replace(/\/$/, '')

  if (action === 'upload') {
    if (!target) throw new Error('eval storage upload requires a file path')
    const apiKey = await getApiKey(opts)
    if (!apiKey) throw new CliError('missing_api_key', '--api-key or LMX_API_KEY is required for storage upload', [
      'Create an API key in the LocalMaxxing dashboard.',
      'Pass it with --api-key bhk_... or set LMX_API_KEY.',
    ])
    const data = await readFile(target)
    const info = await stat(target)
    const format = optString(opts, 'format') ?? basename(target).split('.').pop() ?? 'other'
    const contentType = optString(opts, 'content-type') ?? defaultContentType(format)
    const metadata = {
      kind: forcedKind ?? optString(opts, 'kind') ?? 'artifact',
      filename: optString(opts, 'filename') ?? basename(target),
      contentType,
      format,
      byteSize: info.size,
      sha256: createHash('sha256').update(data).digest('hex'),
      itemCount: optInt(opts, 'item-count'),
    }
    const upload = await submitJson(apiUrl, apiKey, '/api/evals/storage/upload-url', metadata)
    const uploadUrl = stringField(upload, 'uploadUrl') ?? stringField(upload, 'url')
    if (!uploadUrl) throw new CliError('storage_upload_url_missing', 'Storage upload-url response did not include uploadUrl', [
      'Check that the LocalMaxxing API supports /api/evals/storage/upload-url.',
      'Try again or inspect the response details.',
    ], upload)
    const headers = stringHeaders(recordField(upload, 'headers'))
    if (!Object.keys(headers).some(key => key.toLowerCase() === 'content-type')) headers['Content-Type'] = contentType
    const put = await fetch(uploadUrl, { method: 'PUT', headers, body: data })
    if (!put.ok) throw new CliError('storage_put_failed', `Storage PUT failed: ${put.status} ${put.statusText}`, [
      'Retry the upload; signed upload URLs can expire.',
      'Check --content-type and file size.',
    ], await put.text())
    const storageRef = recordField(upload, 'storageRef') ?? stringField(upload, 'storageRef') ?? stringField(upload, 'key')
    if (!storageRef) throw new CliError('storage_ref_missing', 'Storage upload-url response did not include storageRef or key', [
      'Check that the LocalMaxxing API returned a storage reference for completion.',
    ], upload)
    const completed = await submitJson(apiUrl, apiKey, '/api/evals/storage/complete', { storageRef })
    await writeOrPrintJson('storage_upload', opts, { metadata, storageRef, completed })
    return
  }

  if (action === 'download') {
    if (!target) throw new Error('eval storage download requires a storage key')
    const outPath = optString(opts, 'out')
    if (!outPath) throw new CliError('missing_option', '--out is required for storage download', ['Pass --out <path> to write the downloaded object.'])
    const apiKey = await getApiKey(opts)
    const signed = await fetchAuthedJson<unknown>(apiUrl, apiKey, `/api/evals/storage/download-url?key=${encodeURIComponent(target)}`)
    const downloadUrl = stringField(signed, 'downloadUrl') ?? stringField(signed, 'url')
    if (!downloadUrl) throw new CliError('storage_download_url_missing', 'Storage download-url response did not include downloadUrl', [
      'Check the storage key and API URL.',
    ], signed)
    const res = await fetch(downloadUrl)
    if (!res.ok) throw new CliError('storage_download_failed', `Storage download failed: ${res.status} ${res.statusText}`, [
      'Check the storage key and retry; signed download URLs can expire.',
    ], await res.text())
    await writeFile(outPath, new Uint8Array(await res.arrayBuffer()))
    printInfo('storage_downloaded', { path: outPath, key: target })
    return
  }

  throw new Error('Unknown storage command. Use upload or download.')
}

async function handleSuiteCommand(action: string | undefined, target: string | undefined, opts: Record<string, string | boolean>) {
  const apiUrl = (optString(opts, 'api-url') ?? 'https://www.localmaxxing.com').replace(/\/$/, '')

  if (action === 'init') {
    const payload = buildSuiteTemplate(opts)
    validateSuitePayload(payload)
    const outPath = optString(opts, 'out') ?? `${payload.slug}.eval-suite.json`
    await writeFile(outPath, JSON.stringify(payload, null, 2) + '\n')
    printInfo('suite_template_written', {
      path: outPath,
      slug: payload.slug,
      runner: payload.runner,
      scoringMethod: payload.suiteDoc.scoringMethod,
      tasks: payload.suiteDoc.tasks.length,
    })
    console.log('Edit the inline dataset items, then run:')
    console.log(`  lmx eval suite validate ${outPath}`)
    console.log(`  lmx eval suite submit ${outPath} --api-key bhk_...`)
    return
  }

  if (action === 'list') {
    const suites = await fetchJson<unknown>(`${apiUrl}/api/evals/suites`)
    await writeOrPrintJson('suites', opts, redactGoldFields(suites))
    return
  }

  if (action === 'search') {
    const query = target ?? optString(opts, 'q') ?? optString(opts, 'query')
    if (!query) throw new CliError('missing_query', 'eval suite search requires a query', ['Run lmx eval suite search reasoning.'])
    const raw = await fetchJson<unknown>(`${apiUrl}/api/evals/suites`)
    const limit = optInt(opts, 'limit') ?? 20
    const runner = optString(opts, 'runner')?.toUpperCase()
    const category = optString(opts, 'category')?.toLowerCase()
    const suites = suitesFromResponse(raw)
      .filter(suite => suiteMatches(suite, query))
      .filter(suite => !runner || String(suite.runner).toUpperCase() === runner)
      .filter(suite => !category || String(suite.category).toLowerCase() === category)
      .slice(0, limit)
    await writeOrPrintJson('suites', opts, redactGoldFields({ suites, total: suites.length, query }))
    return
  }

  if (action === 'show' || action === 'get') {
    if (!target) throw new Error('eval suite show requires a suite slug')
    const suite = await fetchJson<unknown>(`${apiUrl}/api/evals/suites/${encodeURIComponent(target)}`)
    await writeOrPrintJson('suite', opts, redactGoldFields(suite))
    return
  }

  if (!target) throw new Error(`eval suite ${action ?? '<action>'} requires a suite JSON path`)
  const payload = await readJson<SuiteCreatePayload>(target)
  validateSuitePayload(payload)

  if (action === 'validate') {
    printInfo('suite_valid', {
      slug: payload.slug,
      runner: payload.runner,
      scoringMethod: payload.suiteDoc.scoringMethod,
      tasks: payload.suiteDoc.tasks.length,
      submit: `lmx eval suite submit ${target} --api-key bhk_...`,
    })
    if (payload.suiteDoc.runner === 'custom') {
      console.log('Inline gold answers/rubrics are accepted in the suite payload; public suite responses redact gold fields.')
    }
    return
  }

  if (action === 'submit') {
    const apiKey = await getApiKey(opts)
    if (!apiKey) throw new CliError('missing_api_key', '--api-key or LMX_API_KEY is required', ['Create an API key in the LocalMaxxing dashboard.', 'Pass it with --api-key bhk_... or set LMX_API_KEY.'])
    const response = await submitJson(apiUrl, apiKey, '/api/evals/suites', payload)
    console.log(JSON.stringify(response, null, 2))
    printInfo('suite_submitted', { slug: payload.slug, status: 'PENDING', next: 'Wait for admin approval before running public submissions.' })
    return
  }

  throw new Error('Unknown suite command. Use init, validate, or submit.')
}

async function editBenchmarkRun(path: string, opts: Record<string, string | boolean>) {
  const raw = await readJson<unknown>(path)
  const payload = benchmarkPayloadObject(raw)
  if (!payload) throw new CliError('invalid_benchmark_run', 'Saved benchmark run must be a JSON object or { payload: object }.', [], raw)

  let changed = false
  const patchPath = optString(opts, 'patch')
  if (patchPath) {
    const patch = await readJson<unknown>(patchPath)
    if (!patch || typeof patch !== 'object' || Array.isArray(patch)) throw new CliError('invalid_patch', '--patch must point to a JSON object.', [], patch)
    mergeObject(payload, patch as Record<string, unknown>)
    changed = true
  }

  const patchText = optString(opts, 'set-json')
  if (patchText) {
    let patch: unknown
    try {
      patch = JSON.parse(patchText) as unknown
    } catch (error) {
      throw new CliError('json_parse_error', '--set-json must be a JSON object.', [], asMessage(error))
    }
    if (!patch || typeof patch !== 'object' || Array.isArray(patch)) throw new CliError('invalid_patch', '--set-json must be a JSON object.', [], patch)
    mergeObject(payload, patch as Record<string, unknown>)
    changed = true
  }

  for (const [flag, key] of [
    ['notes', 'notes'],
    ['tok-s-out', 'tokSOut'],
    ['tok-s-prefill', 'tokSPrefill'],
    ['tok-s-total', 'tokSTotal'],
    ['ttft-ms', 'ttftMs'],
    ['peak-vram-gb', 'peakVramGb'],
  ] as const) {
    if (optString(opts, flag) === undefined) continue
    payload[key] = flag === 'notes' ? optString(opts, flag) : optNumber(opts, flag)
    changed = true
  }

  if (!changed) throw new CliError('no_edit_requested', 'No edit was requested.', ['Pass --set-json, --patch, --notes, or a metric override such as --tok-s-out.'])
  const normalized = await normalizeBenchmarkPayload(raw, payload)
  if (raw && typeof raw === 'object' && !Array.isArray(raw) && 'payload' in raw) {
    ;(raw as Record<string, unknown>).payload = normalized
    await writeFile(path, JSON.stringify(raw, null, 2) + '\n')
  } else {
    await writeFile(path, JSON.stringify(normalized, null, 2) + '\n')
  }
  printInfo('benchmark_run_updated', { path, canSubmit: (runNumberField(normalized, 'tokSOut') ?? 0) > 0 })
}

async function rerunBenchmarkRun(path: string, opts: Record<string, string | boolean>) {
  const raw = await readJson<unknown>(path)
  const saved = benchmarkPayloadObject(raw)
  if (!saved) throw new CliError('invalid_benchmark_run', 'Saved benchmark run must be a JSON object or { payload: object }.', [], raw)
  const payload = { ...saved }
  if (opts['dry-run']) payload.dryRun = true
  if (optString(opts, 'notes')) payload.notes = optString(opts, 'notes')
  const outPath = optString(opts, 'out') ?? benchmarkRunPath(payload, opts)
  await mkdir(dirname(outPath), { recursive: true })
  await writeFile(outPath, JSON.stringify(payload, null, 2) + '\n')
  printInfo('benchmark_run_rerun_written', { source: path, path: outPath, dryRun: Boolean(opts['dry-run']) })
}

async function handleBenchmarkRunsCommand(action: string | undefined, target: string | undefined, opts: Record<string, string | boolean>) {
  const normalizedAction = action === 'validate' ? 'dry-run' : action ?? 'list'
  if (normalizedAction === 'list' || normalizedAction === 'ls') {
    const limit = optInt(opts, 'limit') ?? 50
    const { runsDir, runs } = await benchmarkRunSummaries(opts)
    await writeOrPrintJson('benchmark_runs', opts, { runsDir, count: Math.min(runs.length, limit), runs: runs.slice(0, limit) })
    return
  }

  const path = resolveBenchmarkRunPath(target, opts)
  if (normalizedAction === 'show' || normalizedAction === 'cat') {
    await writeOrPrintJson('benchmark_run', opts, await readJson<unknown>(path))
    return
  }
  if (normalizedAction === 'edit' || normalizedAction === 'patch') {
    await editBenchmarkRun(path, opts)
    return
  }
  if (normalizedAction === 'rerun' || normalizedAction === 'replay') {
    await rerunBenchmarkRun(path, opts)
    return
  }
  if (normalizedAction === 'submit' || normalizedAction === 'dry-run') {
    await handleBenchmarkCommand(normalizedAction, path, opts)
    return
  }
  if (normalizedAction === 'delete' || normalizedAction === 'rm' || normalizedAction === 'remove') {
    if (!opts.yes && !opts.force) throw new CliError('confirmation_required', 'Deleting a saved run requires --yes.', [`Rerun: lmx benchmark runs delete ${shellQuote(path)} --yes`])
    await rm(path)
    printInfo('benchmark_run_deleted', { path })
    return
  }

  throw new Error('Unknown benchmark runs command. Use list, show, edit, rerun, submit, dry-run, or delete.')
}

async function handleBenchmarkCommand(action: string | undefined, target: string | undefined, opts: Record<string, string | boolean>, extraTarget?: string) {
  const apiUrl = (optString(opts, 'api-url') ?? 'https://www.localmaxxing.com').replace(/\/$/, '')
  const normalizedAction = action === 'validate' ? 'dry-run' : action

  if (normalizedAction === 'runs' || normalizedAction === 'run-file' || normalizedAction === 'run-files') {
    await handleBenchmarkRunsCommand(target, extraTarget ?? optString(opts, 'path') ?? optString(opts, 'run'), opts)
    return
  }

  if (['list', 'ls', 'show', 'cat', 'edit', 'patch', 'rerun', 'replay'].includes(normalizedAction ?? '')) {
    await handleBenchmarkRunsCommand(normalizedAction, target, opts)
    return
  }

  if (normalizedAction === 'run' || normalizedAction === 'measure') {
    const payload = await benchmarkPayloadFromRun(target, opts)
    const outPath = optString(opts, 'out') ?? benchmarkRunPath(payload, opts)
    await mkdir(dirname(outPath), { recursive: true })
    await writeFile(outPath, JSON.stringify(payload, null, 2) + '\n')
    printInfo('benchmark_payload_written', {
      path: outPath,
      engine: payload.engineName,
      tokSOut: payload.tokSOut,
      tokSPrefill: payload.tokSPrefill,
      tokSTotal: payload.tokSTotal,
      ttftMs: payload.ttftMs,
    })

    if (opts.submit || opts['dry-run']) {
      const apiKey = await getApiKey(opts)
      if (!apiKey) throw new CliError('missing_api_key', '--api-key or LMX_API_KEY is required for benchmark submit/dry-run', [
        'Create an API key in the LocalMaxxing dashboard.',
        'Pass it with --api-key bhk_... or set LMX_API_KEY.',
      ])
      const endpoint = opts['dry-run'] ? '/api/benchmarks/dry-run' : '/api/benchmarks'
      const response = await submitJson(apiUrl, apiKey, endpoint, payload)
      console.log(JSON.stringify(response, null, 2))
      printInfo(opts['dry-run'] ? 'benchmark_dry_run_valid' : 'benchmark_submitted', {
        endpoint,
        status: opts['dry-run'] ? 'valid' : 'submitted',
      })
    }
    return
  }

  if (normalizedAction !== 'submit' && normalizedAction !== 'dry-run') {
    throw new Error('Unknown benchmark command. Use run, runs, list, show, edit, rerun, submit, or dry-run.')
  }
  if (!target) throw new Error(`benchmark ${normalizedAction} requires a benchmark JSON path`)

  const apiKey = await getApiKey(opts)
  if (!apiKey) throw new CliError('missing_api_key', '--api-key or LMX_API_KEY is required for benchmark submit/dry-run', [
    'Create an API key in the LocalMaxxing dashboard.',
    'Pass it with --api-key bhk_... or set LMX_API_KEY.',
  ])

  const raw = await readJson<unknown>(target)
  const payloadRaw = raw && typeof raw === 'object' && !Array.isArray(raw) && 'payload' in raw
    ? (raw as { payload: unknown }).payload
    : raw
  if (!payloadRaw || typeof payloadRaw !== 'object' || Array.isArray(payloadRaw)) {
    throw new CliError('invalid_benchmark_payload', 'Benchmark payload must be a JSON object.', [
      'Pass a JSON object matching POST /api/benchmarks.',
      'Use docs/agent-skill.md or GET /api/agent-context for the current schema.',
    ])
  }
  const payload = await normalizeBenchmarkPayload(raw, payloadRaw as Record<string, unknown>)

  const endpoint = normalizedAction === 'dry-run' ? '/api/benchmarks/dry-run' : '/api/benchmarks'
  const response = await submitJson(apiUrl, apiKey, endpoint, payload)
  console.log(JSON.stringify(response, null, 2))
  printInfo(normalizedAction === 'dry-run' ? 'benchmark_dry_run_valid' : 'benchmark_submitted', {
    endpoint,
    status: normalizedAction === 'dry-run' ? 'valid' : 'submitted',
  })
}

async function handleAuthCommand(opts: Record<string, string | boolean>) {
  const key = optString(opts, 'key') ?? optString(opts, 'api-key')
  if (opts.logout) {
    await saveConfig({})
    printInfo('auth_cleared', { path: CONFIG_FILE })
    return
  }
  if (key) {
    await saveConfig({ apiKey: key, authProvider: 'manual', authSavedAt: new Date().toISOString() })
    printInfo('auth_saved', { path: CONFIG_FILE, key: `${key.slice(0, 8)}...` })
    return
  }
  const config = await loadConfig()
  const apiKey = process.env.LMX_API_KEY ?? config.apiKey
  if (!apiKey) {
    printInfo('auth_missing', { next: 'Run lmx auth --key bhk_... or set LMX_API_KEY.' })
    return
  }
  printInfo('auth_status', {
    source: process.env.LMX_API_KEY ? 'LMX_API_KEY' : CONFIG_FILE,
    key: `${apiKey.slice(0, 8)}...`,
    provider: config.authProvider,
  })
}

async function handleHardwareCommand(opts: Record<string, string | boolean>) {
  const hardware = detectHardware()
  if (opts.out) {
    const outPath = optString(opts, 'out') ?? 'hardware.json'
    await writeFile(outPath, JSON.stringify(hardware, null, 2) + '\n')
    printInfo('hardware_written', { path: outPath })
    return
  }
  console.log(JSON.stringify(hardware, null, 2))
}

async function loadSuite(apiUrl: string, suiteSlug: string, suiteFile?: string) {
  return suiteFile
    ? await readJson<SuiteResponse | SuiteCreatePayload>(suiteFile).then((payload) => ({
        slug: payload.slug,
        name: payload.name,
        runner: payload.runner,
        suiteDoc: payload.suiteDoc,
      }))
    : await fetchJson<SuiteResponse>(`${apiUrl}/api/evals/suites/${encodeURIComponent(suiteSlug)}`)
}

function downloadUrlFromValue(value: unknown): string | undefined {
  if (typeof value === 'string') return value
  if (Array.isArray(value)) {
    for (const child of value) {
      const url = downloadUrlFromValue(child)
      if (url) return url
    }
    return undefined
  }
  if (!value || typeof value !== 'object') return undefined
  const record = value as Record<string, unknown>
  return stringField(record, 'downloadUrl')
    ?? stringField(record, 'url')
    ?? stringField(record, 'signedUrl')
    ?? downloadUrlFromValue(record.downloadUrls)
}

function downloadUrlForTask(value: unknown, task: EvalTask, taskIndex: number) {
  if (Array.isArray(value)) {
    const indexed = downloadUrlFromValue(value[taskIndex])
    if (indexed) return indexed
    for (const child of value) {
      if (!child || typeof child !== 'object') continue
      const record = child as Record<string, unknown>
      const key = stringField(record, 'taskKey') ?? stringField(record, 'task') ?? stringField(record, 'key')
      if (key === task.key) return downloadUrlFromValue(record)
    }
    return undefined
  }

  if (!value || typeof value !== 'object') return undefined
  const record = value as Record<string, unknown>
  const dataset = task.dataset && typeof task.dataset === 'object' ? task.dataset as Record<string, unknown> : {}
  const datasetKeys = ['storageKey', 'storageRef', 'datasetKey', 'id']
    .map(key => typeof dataset[key] === 'string' ? dataset[key] as string : undefined)
    .filter((key): key is string => !!key)
  for (const key of [task.key, String(taskIndex), ...datasetKeys]) {
    const url = downloadUrlFromValue(record[key])
    if (url) return url
  }
  return undefined
}

function applyRunBundleDownloadUrls(suite: SuiteResponse, bundle: RunBundleResponse) {
  const candidates = [bundle.downloadUrls, bundle.datasetDownloadUrls, bundle.datasets]
  for (let taskIndex = 0; taskIndex < suite.suiteDoc.tasks.length; taskIndex++) {
    const task = suite.suiteDoc.tasks[taskIndex]
    const existing = task.dataset ? firstDatasetDownloadUrl(task.dataset) : undefined
    const url = existing ?? candidates.map(candidate => downloadUrlForTask(candidate, task, taskIndex)).find(Boolean)
      ?? (suite.suiteDoc.tasks.length === 1 ? candidates.map(downloadUrlFromValue).find(Boolean) : undefined)
    if (!url) continue
    task.dataset = { ...(task.dataset ?? { source: 'url' }), source: 'url', url, downloadUrl: url }
  }
}

function suiteFromRunBundle(bundle: RunBundleResponse, suiteSlug: string): SuiteResponse {
  const suiteRecord = recordField(bundle, 'suite') ?? bundle as Record<string, unknown>
  const suiteDoc = (recordField(suiteRecord, 'suiteDoc') ?? recordField(bundle, 'suiteDoc')) as SuiteDoc | undefined
  const runner = stringField(suiteRecord, 'runner') ?? stringField(bundle, 'runner')
  if (!suiteDoc || (runner !== 'CUSTOM' && runner !== 'LM_EVAL_HARNESS')) {
    throw new CliError('run_bundle_invalid', 'Run bundle response did not include a runnable suite document.', [
      'Check that the LocalMaxxing API supports /run-bundle for this suite.',
      'Retry with a valid API key or inspect the suite with eval suite show.',
    ], bundle)
  }
  const suite = {
    slug: stringField(suiteRecord, 'slug') ?? stringField(bundle, 'slug') ?? suiteSlug,
    name: stringField(suiteRecord, 'name') ?? stringField(bundle, 'name') ?? suiteSlug,
    runner,
    suiteDoc,
  } as SuiteResponse
  applyRunBundleDownloadUrls(suite, bundle)
  return suite
}

async function loadSuiteForRun(apiUrl: string, suiteSlug: string, opts: Record<string, string | boolean>) {
  const suiteFile = optString(opts, 'suite-file')
  if (suiteFile) return loadSuite(apiUrl, suiteSlug, suiteFile)
  const apiKey = await getApiKey(opts)
  if (!apiKey) return loadSuite(apiUrl, suiteSlug)
  const bundle = await fetchAuthedJson<RunBundleResponse>(apiUrl, apiKey, `/api/evals/suites/${encodeURIComponent(suiteSlug)}/run-bundle`)
  return suiteFromRunBundle(bundle, suiteSlug)
}

function validateRemoteSuite(suite: SuiteResponse | SuiteCreatePayload) {
  validateSuitePayload({
    slug: suite.slug,
    name: suite.name,
    category: 'remote',
    runner: suite.runner,
    suiteDoc: suite.suiteDoc,
  })
}

function inferredFewShot(doc: SuiteDoc, opts: Record<string, string | boolean>) {
  const explicit = optString(opts, 'num-fewshot') ?? optString(opts, 'fewshot')
  if (explicit !== undefined) return explicit
  if (typeof doc.runConfig?.fewShot === 'number') return String(doc.runConfig.fewShot)
  const shots = [...new Set(doc.tasks.map(task => task.nShots).filter((value): value is number => typeof value === 'number'))]
  return shots.length === 1 ? String(shots[0]) : undefined
}

function lmEvalArgs(suite: SuiteResponse | SuiteCreatePayload, opts: Record<string, string | boolean>) {
  const backend = optString(opts, 'backend') ?? 'hf'
  const model = requireOpt(opts, 'model')
  const tasks = optString(opts, 'tasks') ?? suite.suiteDoc.tasks.map(task => task.key).join(',')
  const resultsPath = optString(opts, 'results') ?? 'localmaxxing-lm-eval-results.json'
  const modelArgs = optString(opts, 'model-args') ?? (backend === 'hf' ? `pretrained=${model}` : undefined)
  const fewShot = inferredFewShot(suite.suiteDoc, opts)
  const args = ['--model', backend]
  if (modelArgs) args.push('--model_args', modelArgs)
  args.push('--tasks', tasks)
  if (fewShot !== undefined) args.push('--num_fewshot', fewShot)
  args.push('--output_path', resultsPath)
  return { args, backend, tasks, resultsPath, modelArgs, fewShot }
}

async function handleLmEvalCommand(suiteSlug: string | undefined, opts: Record<string, string | boolean>) {
  if (!suiteSlug) throw new Error('eval lm-eval requires a suite slug')
  const apiUrl = (optString(opts, 'api-url') ?? 'https://www.localmaxxing.com').replace(/\/$/, '')
  const suite = await loadSuite(apiUrl, suiteSlug, optString(opts, 'suite-file'))
  if (suite.runner !== 'LM_EVAL_HARNESS') throw new CliError('suite_runner_mismatch', `Suite "${suiteSlug}" is ${suite.runner}, not LM_EVAL_HARNESS`, [
    'Use lmx eval run for CUSTOM suites.',
    'Use lmx eval suite list or show to find LM_EVAL_HARNESS suites.',
  ])
  validateRemoteSuite(suite)

  const command = optString(opts, 'lm-eval-bin') ?? 'lm_eval'
  const { args, backend, tasks, resultsPath, modelArgs, fewShot } = lmEvalArgs(suite, opts)
  printInfo('lm_eval_start', {
    suite: suiteSlug,
    command,
    backend,
    modelArgs,
    tasks,
    numFewshot: fewShot ?? 'suite/default',
    output: resultsPath,
  })
  runStreamingCommand(command, args)

  await handleRunCommand(suiteSlug, { ...opts, results: resultsPath })
}

async function handleRunCommand(suiteSlug: string | undefined, opts: Record<string, string | boolean>) {
  if (!suiteSlug) throw new Error('eval run requires a suite slug')
  const apiUrl = (optString(opts, 'api-url') ?? 'https://www.localmaxxing.com').replace(/\/$/, '')
  const suite = await loadSuiteForRun(apiUrl, suiteSlug, opts)
  validateRemoteSuite(suite)
  const result = suite.runner === 'LM_EVAL_HARNESS'
    ? await loadLmEvalResults(optString(opts, 'results') ?? (() => { throw new Error('LM-Eval suites require --results <lm-eval-output.json>') })(), suite)
    : await runCustomLocal(suite, opts)

  const hardwarePath = optString(opts, 'hardware')
  const payload = {
    suiteSlug,
    hfId: optString(opts, 'model') ?? '<required-before-submit>',
    ...(hardwarePath ? { hardware: await readJson<unknown>(hardwarePath) } : {}),
    quantization: optString(opts, 'quantization'),
    executionMode: suite.runner === 'CUSTOM' ? 'CUSTOM_LOCAL' : 'LM_EVAL_LOCAL',
    judgeMode: suite.suiteDoc.scoringMethod === 'llm_judge' ? 'LOCAL_REPORTED' : 'NONE',
    runnerVersion: suite.runner === 'CUSTOM' ? 'localmaxxing-cli custom-local' : 'localmaxxing-cli lm-eval-upload',
    results: result.scores,
    artifacts: redactGoldFields(result.artifacts),
    runConfig: { aggregatePreview: result.aggregate },
  }

  const outPath = optString(opts, 'out') ?? 'localmaxxing-eval-run.json'
  await writeFile(outPath, JSON.stringify(payload, null, 2) + '\n')
  printInfo('run_payload_written', {
    path: outPath,
    suite: suiteSlug,
    tasks: Object.keys(result.scores).length,
    aggregatePreview: result.aggregate,
    artifacts: result.artifacts.length,
  })

  if (opts.submit || opts['dry-run']) {
    const apiKey = await getApiKey(opts)
    if (!apiKey) throw new CliError('missing_api_key', '--api-key or LMX_API_KEY is required for submit/dry-run', ['Create an API key in the LocalMaxxing dashboard.', 'Pass it with --api-key bhk_... or set LMX_API_KEY.'])
    if (!hardwarePath) throw new CliError('missing_hardware', '--hardware is required for submit/dry-run', ['Create a hardware JSON file matching /api/agent-context hardwareSchemas.', 'Pass --hardware hardware.json.'])
    if (!optString(opts, 'model')) throw new CliError('missing_model', '--model is required for submit/dry-run', ['Pass --model <HuggingFace model id>.'])
    const endpoint = opts['dry-run'] ? '/api/evals/runs/dry-run' : '/api/evals/runs'
    const response = await submitJson(apiUrl, apiKey, endpoint, payload)
    console.log(JSON.stringify(response, null, 2))
    printInfo(opts['dry-run'] ? 'run_dry_run_valid' : 'run_submitted', {
      suite: suiteSlug,
      endpoint,
      status: opts['dry-run'] ? 'valid' : 'submitted',
    })
  }
}

async function handleExecuteCommand(suiteSlug: string | undefined, opts: Record<string, string | boolean>) {
  if (!suiteSlug) throw new Error('eval execute requires a suite slug')
  const apiUrl = (optString(opts, 'api-url') ?? 'https://www.localmaxxing.com').replace(/\/$/, '')
  const model = requireOpt(opts, 'model')
  const baseUrl = requireOpt(opts, 'base-url')
  const hardwarePath = optString(opts, 'hardware')
  if (opts.submit && !hardwarePath) throw new CliError('missing_hardware', '--hardware is required when using eval execute --submit', [
    'Create a hardware JSON file matching /api/agent-context hardwareSchemas.',
    'Pass --hardware hardware.json, or omit --submit to execute without auto-submitting a run.',
  ])
  const apiKey = await getApiKey(opts)
  if (!apiKey) throw new CliError('missing_api_key', '--api-key or LMX_API_KEY is required for eval execute', [
    'Create an API key in the LocalMaxxing dashboard.',
    'Pass it with --api-key bhk_... or set LMX_API_KEY.',
  ])

  const payload = {
    suiteSlug,
    model,
    baseUrl,
    autoSubmit: Boolean(opts.submit),
    ...(hardwarePath ? { hardware: await readJson<unknown>(hardwarePath) } : {}),
    quantization: optString(opts, 'quantization'),
    modelRevision: optString(opts, 'model-revision') ?? 'main',
    notes: optString(opts, 'notes'),
  }

  const response = await submitJson(apiUrl, apiKey, '/api/evals/execute', payload)
  console.log(JSON.stringify(response, null, 2))
  printInfo('execute_submitted', {
    suite: suiteSlug,
    endpoint: '/api/evals/execute',
    autoSubmit: payload.autoSubmit,
  })
}

async function main() {
  const { positional, opts } = parseArgs(process.argv.slice(2))
  if (!['eval', 'benchmark', 'bench', 'auth', 'hardware', 'context', 'agent-context', 'model'].includes(positional[0] ?? '') || opts.help) {
    usage()
    process.exit(positional[0] ? 1 : 0)
  }

  if (positional[0] === 'auth') {
    await handleAuthCommand(opts)
    return
  }

  if (positional[0] === 'hardware') {
    await handleHardwareCommand(opts)
    return
  }

  if (positional[0] === 'context' || positional[0] === 'agent-context') {
    await handleContextCommand(opts)
    return
  }

  if (positional[0] === 'model') {
    await handleModelCommand(positional[1], positional[2], opts)
    return
  }

  if (positional[0] === 'benchmark' || positional[0] === 'bench') {
    await handleBenchmarkCommand(positional[1], positional[2], opts, positional[3])
    return
  }

  if (positional[1] === 'storage') {
    await handleStorageCommand(positional[2], positional[3], opts)
    return
  }

  if (positional[1] === 'artifact' || positional[1] === 'artifacts') {
    await handleStorageCommand(positional[2], positional[3], opts, 'artifact')
    return
  }

  if (positional[1] === 'suite') {
    await handleSuiteCommand(positional[2], positional[3], opts)
    return
  }

  if (positional[1] === 'run') {
    await handleRunCommand(positional[2], opts)
    return
  }

  if (positional[1] === 'lm-eval' || positional[1] === 'lmeval') {
    await handleLmEvalCommand(positional[2], opts)
    return
  }

  if (positional[1] === 'execute') {
    await handleExecuteCommand(positional[2], opts)
    return
  }

  usage()
  process.exit(1)
}

main().catch(error => {
  printError(error)
  process.exit(1)
})
