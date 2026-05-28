#!/usr/bin/env tsx
import { execFileSync, execSync } from 'node:child_process'
import { createHash } from 'node:crypto'
import { homedir, platform, release, totalmem, cpus } from 'node:os'
import { join } from 'node:path'
import { mkdir, readFile, writeFile } from 'node:fs/promises'

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
  dataset?: { source: string; url?: string; hfPath?: string; hfName?: string; split?: string; items?: DatasetItem[] }
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

function usage() {
  console.log(`LocalMaxxing CLI

Usage:
  lmx auth --key bhk_...
  lmx hardware --out hardware.json
  lmx benchmark submit benchmark.json --api-key bhk_...
  lmx benchmark dry-run benchmark.json --api-key bhk_...
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
  --base-url <url>         OpenAI-compatible model endpoint for custom evals
  --model-api-key <key>    Optional bearer token for the local model endpoint
  --hardware <path>        JSON hardware object required when submitting
  --quantization <label>   Optional quantization label
  --model-revision <rev>   Optional model revision/branch/commit (default: main)
  --results <path>         Existing lm-eval output JSON for LM_EVAL_HARNESS suites
  --suite-file <path>      Local suite JSON for offline run parsing/testing
  --judge-base-url <url>   OpenAI-compatible judge endpoint for llm_judge suites
  --judge-model <model>    Judge model override
  --judge-api-key <key>    Judge bearer token, defaults to EVAL_JUDGE_API_KEY
  --notes <text>           Optional submission notes
  --submit                 Upload run to LocalMaxxing
  --dry-run                Validate upload without creating a run
  --out <path>             Write computed payload/result JSON (default: localmaxxing-eval-run.json)

Benchmark submissions:
  Pass a JSON object matching POST /api/benchmarks, or a lmx-bench result with a payload field.
  Use dry-run first to validate without writing.

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
    : ['output', 'outputText', 'generatedText', 'completion', 'response', 'text']
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

async function loadDataset(dataset: NonNullable<EvalTask['dataset']>) {
  if (dataset.source === 'inline') return dataset.items ?? []
  if (dataset.source === 'url') {
    if (!dataset.url) throw new CliError('dataset_missing_url', 'url dataset missing url', ['Add dataset.url to the suite task.'])
    return fetchJson<DatasetItem[]>(dataset.url)
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
  const json = await res.json() as { choices?: Array<{ message?: { content?: string } }> }
  return json.choices?.[0]?.message?.content?.trim() ?? ''
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

async function handleBenchmarkCommand(action: string | undefined, target: string | undefined, opts: Record<string, string | boolean>) {
  const apiUrl = (optString(opts, 'api-url') ?? 'https://www.localmaxxing.com').replace(/\/$/, '')
  const normalizedAction = action === 'validate' ? 'dry-run' : action

  if (normalizedAction !== 'submit' && normalizedAction !== 'dry-run') {
    throw new Error('Unknown benchmark command. Use submit or dry-run.')
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

async function handleRunCommand(suiteSlug: string | undefined, opts: Record<string, string | boolean>) {
  if (!suiteSlug) throw new Error('eval run requires a suite slug')
  const apiUrl = (optString(opts, 'api-url') ?? 'https://www.localmaxxing.com').replace(/\/$/, '')
  const suiteFile = optString(opts, 'suite-file')
  const suite = suiteFile
    ? await readJson<SuiteResponse | SuiteCreatePayload>(suiteFile).then((payload) => ({
        slug: payload.slug,
        name: payload.name,
        runner: payload.runner,
        suiteDoc: payload.suiteDoc,
      }))
    : await fetchJson<SuiteResponse>(`${apiUrl}/api/evals/suites/${encodeURIComponent(suiteSlug)}`)
  validateSuitePayload({
    slug: suite.slug,
    name: suite.name,
    category: 'remote',
    runner: suite.runner,
    suiteDoc: suite.suiteDoc,
  })
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
    artifacts: result.artifacts,
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
  if (!['eval', 'benchmark', 'bench', 'auth', 'hardware'].includes(positional[0] ?? '') || opts.help) {
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

  if (positional[0] === 'benchmark' || positional[0] === 'bench') {
    await handleBenchmarkCommand(positional[1], positional[2], opts)
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
