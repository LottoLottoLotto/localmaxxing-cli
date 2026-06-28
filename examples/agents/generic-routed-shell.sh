#!/usr/bin/env bash
set -euo pipefail

: "${LMX_TERMINAL_INSTRUCTION_FILE:?}"
: "${LMX_TERMINAL_SHELL_COMMAND:?}"
: "${LMX_TERMINAL_TRACE_DIR:?}"

prompt="$LMX_TERMINAL_TRACE_DIR/prompt.txt"
log="$LMX_TERMINAL_TRACE_DIR/agent.log"

cat > "$prompt" <<PROMPT
You are solving a Terminal-Bench task. Your shell is already routed into the task container at /app.

Task:
$(cat "$LMX_TERMINAL_INSTRUCTION_FILE")

Rules:
- Work in /app.
- Do not read, run, copy, or modify /tests, test.sh, hidden tests, verifier code, or solution files.
- Write all agent logs/traces to $LMX_TERMINAL_TRACE_DIR.
PROMPT

# Replace this with the user's host-installed agent. The key invariant is that
# the agent's shell/tool command must call "$LMX_TERMINAL_SHELL_COMMAND" '<cmd>'
# instead of using the host shell directly.
"$LMX_TERMINAL_SHELL_COMMAND" 'pwd && ls -la /app' > "$log" 2>&1
