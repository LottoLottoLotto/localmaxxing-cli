#!/usr/bin/env bash
set -euo pipefail

: "${LMX_TERMINAL_CONTAINER:?}"
: "${LMX_TERMINAL_INSTRUCTION_FILE:?}"
: "${LMX_TERMINAL_TRACE_DIR:?}"

agent_bin="${AGENT_BIN:?Set AGENT_BIN to a portable agent binary/script}"
container_agent="/tmp/localmaxxing-agent"
container_prompt="/tmp/localmaxxing-agent-prompt.txt"
container_trace="/tmp/localmaxxing-agent-traces"

prompt="$LMX_TERMINAL_TRACE_DIR/prompt.txt"
cat > "$prompt" <<PROMPT
You are solving a Terminal-Bench task inside the task container.

Task:
$(cat "$LMX_TERMINAL_INSTRUCTION_FILE")

Rules:
- Work in /app.
- Do not read, run, copy, or modify /tests, test.sh, hidden tests, verifier code, or solution files.
- Save traces/logs under $container_trace.
PROMPT

docker cp "$agent_bin" "$LMX_TERMINAL_CONTAINER:$container_agent"
docker cp "$prompt" "$LMX_TERMINAL_CONTAINER:$container_prompt"
docker exec "$LMX_TERMINAL_CONTAINER" mkdir -p "$container_trace"
docker exec "$LMX_TERMINAL_CONTAINER" chmod +x "$container_agent"

docker exec \
  -e OPENAI_BASE_URL="${LMX_TERMINAL_CONTAINER_BASE_URL:-}" \
  -e OPENAI_API_KEY="${OPENAI_API_KEY:-}" \
  -e LMX_TERMINAL_MODEL="${LMX_TERMINAL_MODEL:-}" \
  -w "${LMX_TERMINAL_WORKDIR:-/app}" \
  "$LMX_TERMINAL_CONTAINER" \
  "$container_agent" "$container_prompt"

docker cp "$LMX_TERMINAL_CONTAINER:$container_trace/." "$LMX_TERMINAL_TRACE_DIR/" >/dev/null 2>&1 || true
