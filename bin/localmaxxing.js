#!/usr/bin/env node
import { existsSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

const root = dirname(dirname(fileURLToPath(import.meta.url)))
const built = join(root, 'dist', 'localmaxxing.js')
const source = join(root, 'src', 'localmaxxing.ts')

if (existsSync(built)) {
  await import(pathToFileURL(built).href)
} else if (existsSync(source)) {
  try {
    await import('tsx/esm')
    await import(pathToFileURL(source).href)
  } catch {
    console.error('[localmaxxing:error] cli_not_built')
    console.error('The CLI has not been built yet.')
    console.error('Fix:')
    console.error('- Run npm install')
    console.error('- Run npm run build')
    process.exit(1)
  }
} else {
  console.error('[localmaxxing:error] cli_entry_missing')
  console.error('Could not find dist/localmaxxing.js or src/localmaxxing.ts.')
  process.exit(1)
}
