#!/usr/bin/env bash
set -uo pipefail

: "${LMX_TERMINAL_CONTAINER:?}"
: "${LMX_TERMINAL_TASK_ID:?}"
: "${LMX_TERMINAL_INSTRUCTION_FILE:?}"
: "${LMX_TERMINAL_TRACE_DIR:?}"

TASK="$(cat "$LMX_TERMINAL_INSTRUCTION_FILE")"
SESSION_ROOT="$LMX_TERMINAL_TRACE_DIR"
SESSION_DIR="$SESSION_ROOT/omp-$LMX_TERMINAL_TASK_ID-$(date +%s%N)"
mkdir -p "$SESSION_DIR"
LOG="$SESSION_DIR/omp.jsonl"
OMP_BUDGET_SEC="${OMP_BUDGET_SEC:-${LMX_TERMINAL_AGENT_TIMEOUT_SEC:-900}}"

HOST_OMP="${OMP_BIN:-$(command -v omp)}"
HOST_MODELS_DB="${OMP_MODELS_DB:-$HOME/.omp/agent/models.db}"
HOST_OMP_CONFIG="${OMP_CONFIG:-$HOME/.omp/agent/config.yml}"
CONTAINER_OMP="/tmp/localmaxxing-omp"
CONTAINER_PROMPT_FILE="/tmp/localmaxxing-omp-prompt.txt"
CONTAINER_BASE_URL="${OMP_CONTAINER_BASE_URL:-${LMX_TERMINAL_CONTAINER_BASE_URL:-http://172.17.0.1:8080}}"
PATCHED_MODELS_DB="$SESSION_DIR/models.db"

python3 - "$HOST_MODELS_DB" "$PATCHED_MODELS_DB" "$CONTAINER_BASE_URL" <<'PY'
import json
import sqlite3
import sys

source, target, base_url = sys.argv[1:]
src = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
dst = sqlite3.connect(target)
src.backup(dst)
src.close()
row = dst.execute("select models from model_cache where provider_id='llama.cpp'").fetchone()
if row:
    models = json.loads(row[0])
    for model in models:
        model["baseUrl"] = base_url
    dst.execute(
        "update model_cache set models=? where provider_id='llama.cpp'",
        (json.dumps(models, separators=(",", ":")),),
    )
    dst.commit()
dst.close()
PY

docker cp "$HOST_OMP" "$LMX_TERMINAL_CONTAINER:$CONTAINER_OMP"
docker exec "$LMX_TERMINAL_CONTAINER" chmod +x "$CONTAINER_OMP"
docker exec "$LMX_TERMINAL_CONTAINER" rm -rf /root/.omp/agent
docker exec "$LMX_TERMINAL_CONTAINER" mkdir -p /root/.omp/agent
docker cp "$PATCHED_MODELS_DB" "$LMX_TERMINAL_CONTAINER:/root/.omp/agent/models.db"
if [ -f "$HOST_OMP_CONFIG" ]; then
  docker cp "$HOST_OMP_CONFIG" "$LMX_TERMINAL_CONTAINER:/root/.omp/agent/config.yml"
fi

PROMPT_FILE="$SESSION_DIR/prompt.txt"
cat > "$PROMPT_FILE" <<PROMPT
You are solving a Terminal-Bench task inside the task container.

Task:
$TASK

Execution contract:
- You are an autonomous coding agent, not a summarizer.
- Your first assistant action MUST be a bash tool call that inspects /app.
- Keep using bash tool calls until the task is complete.
- Do not reply with only a plan, restatement, or summary before running commands.

Benchmark rules:
- Treat the Task text above as authoritative. Do not use outside knowledge of the benchmark's hidden tests or solutions.
- Work in /app. Files you create or modify must be inside this container.
- Do NOT read, run, copy, or modify /tests, test.sh, hidden tests, verifier code, or solution files.
- Use only ordinary self-checks that a benchmark participant could perform from the task statement and visible container state.
- Stop only when you believe the task is complete as written; do not stop after inspection-only commands unless the task itself only asked for inspection.
PROMPT
docker cp "$PROMPT_FILE" "$LMX_TERMINAL_CONTAINER:$CONTAINER_PROMPT_FILE"

printf '[harness=omp-container-shell]\n[container=%s]\n[session=%s]\n' "$LMX_TERMINAL_CONTAINER" "$SESSION_DIR"

timeout -k 10 "$OMP_BUDGET_SEC" docker exec \
  -e LLAMA_CPP_BASE_URL="$CONTAINER_BASE_URL" \
  -e OMP_MODEL="${OMP_MODEL:-${LMX_TERMINAL_MODEL:-}}" \
  "$LMX_TERMINAL_CONTAINER" \
  bash -lc 'set -euo pipefail
prompt="$(cat /tmp/localmaxxing-omp-prompt.txt)"
exec /tmp/localmaxxing-omp -p \
  --mode json \
  --no-session \
  --model "${OMP_MODEL:?Set OMP_MODEL or LMX_TERMINAL_MODEL}" \
  --auto-approve \
  --approval-mode yolo \
  --cwd /app \
  "$prompt"' \
  > "$LOG" 2>&1
STATUS=$?
if [ "$STATUS" -eq 0 ] && grep -q '"stopReason":"error"\|"errorMessage"' "$LOG"; then
  STATUS=1
fi
if [ "$STATUS" -eq 0 ] && ! grep -q '"type":"function_call"' "$LOG"; then
  STATUS=1
fi

echo "[omp_exit=$STATUS]"
echo "----- omp output -----"
cat "$LOG" 2>/dev/null || true
exit "$STATUS"
