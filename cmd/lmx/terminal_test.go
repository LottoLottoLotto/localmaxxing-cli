package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"
	"unicode/utf8"
)

func TestTerminalImportHarborTask(t *testing.T) {
	src := t.TempDir()
	taskDir := filepath.Join(src, "adaptive-rejection-sampler")
	mustMkdir(t, filepath.Join(taskDir, "environment"))
	mustMkdir(t, filepath.Join(taskDir, "tests"))
	mustWrite(t, filepath.Join(taskDir, "task.toml"), `[task]
name = "terminal-bench/adaptive-rejection-sampler"
description = "sample"

[metadata]
category = "scientific-computing"

[verifier]
timeout_sec = 900

[agent]
timeout_sec = 900

[environment]
docker_image = "alexgshaw/adaptive-rejection-sampler:20251031"
allow_internet = true
`)
	mustWrite(t, filepath.Join(taskDir, "instruction.md"), "Do the task.\n")
	mustWrite(t, filepath.Join(taskDir, "tests", "test.sh"), "#!/usr/bin/env bash\n")

	out := t.TempDir()
	args := parseArgs([]string{"eval", "terminal", "import", src, "--out", out, "--version", "2.1"})
	if err := runTerminalImport(args); err != nil {
		t.Fatalf("runTerminalImport failed: %v", err)
	}
	data, err := os.ReadFile(filepath.Join(out, "adaptive-rejection-sampler", "task.json"))
	if err != nil {
		t.Fatalf("task.json not written: %v", err)
	}
	var task terminalTask
	if err := json.Unmarshal(data, &task); err != nil {
		t.Fatalf("task.json invalid: %v", err)
	}
	if task.Image.Prebuilt != "alexgshaw/adaptive-rejection-sampler:20251031" {
		t.Fatalf("prebuilt image mismatch: %q", task.Image.Prebuilt)
	}
	if task.Verifier.TimeoutSec != 900 {
		t.Fatalf("verifier timeout mismatch: %d", task.Verifier.TimeoutSec)
	}
	if task.Environment.Network != "public" {
		t.Fatalf("network mismatch: %q", task.Environment.Network)
	}
	if _, err := os.Stat(filepath.Join(out, "adaptive-rejection-sampler", "tests", "test.sh")); err != nil {
		t.Fatalf("tests/ not copied: %v", err)
	}
}

func TestTerminus2AgentCommandExtractsEmbeddedAdapterOutsideReleaseWorkingDirectory(t *testing.T) {
	originalWorkingDir, err := os.Getwd()
	if err != nil {
		t.Fatalf("get original working directory: %v", err)
	}
	emptyWorkingDir := t.TempDir()
	if err := os.Chdir(emptyWorkingDir); err != nil {
		t.Fatalf("change to empty release working directory: %v", err)
	}
	defer func() {
		if err := os.Chdir(originalWorkingDir); err != nil {
			t.Errorf("restore working directory: %v", err)
		}
	}()

	command, cleanup, err := terminus2AgentCommand()
	if err != nil {
		t.Fatalf("extract bundled Terminus-2 adapter: %v", err)
	}
	if cleanup == nil {
		t.Fatal("Terminus-2 adapter extraction returned no cleanup function")
	}
	scriptPath := strings.Trim(command, "'\"")
	script, err := os.ReadFile(scriptPath)
	if err != nil {
		t.Fatalf("command %q does not reference an extracted adapter: %v", command, err)
	}
	if !bytes.Equal(script, []byte(terminus2RoutedShellScript)) {
		t.Fatal("extracted Terminus-2 adapter differs from the embedded adapter")
	}
	extractionDir := filepath.Dir(scriptPath)
	dirInfo, err := os.Stat(extractionDir)
	if err != nil {
		t.Fatalf("stat Terminus-2 extraction directory: %v", err)
	}
	if got := dirInfo.Mode().Perm(); got != 0o700 {
		t.Fatalf("Terminus-2 extraction directory permissions = %#o, want private 0700", got)
	}

	cleanup()
	if _, err := os.Stat(extractionDir); !os.IsNotExist(err) {
		t.Fatalf("Terminus-2 cleanup left extraction directory behind: %v", err)
	}
}

func TestExtractBashCommand(t *testing.T) {
	cases := []struct {
		name  string
		reply string
		want  string
		found bool
	}{
		{"bash", "thinking\n```bash\necho hi\n```", "echo hi", true},
		{"generic", "```\npwd\n```", "pwd", true},
		{"python fence", "```python\nprint('hi')\n```", "python3 <<'LMX_SCRIPT'\nprint('hi')\nLMX_SCRIPT", true},
		{"node fence", "```js\nconsole.log('hi')\n```", "node <<'LMX_SCRIPT'\nconsole.log('hi')\nLMX_SCRIPT", true},
		{"sentinel", "TASK_COMPLETE", "", false},
		{"garbage", "run ls", "", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, found := extractBashCommand(tc.reply)
			if found != tc.found || got != tc.want {
				t.Fatalf("extractBashCommand() = (%q, %v), want (%q, %v)", got, found, tc.want, tc.found)
			}
		})
	}
}

func TestRunTerminalEvalEmptyTaskDirIsBundleInvalid(t *testing.T) {
	empty := t.TempDir()
	args := parseArgs([]string{"eval", "terminal", "run", "--task-dir", empty, "--base-url", "http://127.0.0.1:9", "--model", "x", "--json-status", "--dry-run"})
	err := runTerminalEval(args, false)
	if err == nil {
		t.Fatal("expected bundle_invalid error")
	}
	var ce cliError
	if !strings.Contains(err.Error(), "No task.json") && !strings.Contains(err.Error(), "No terminal task bundles") {
		t.Fatalf("unexpected error: %v", err)
	}
	if errorsAsCli(err, &ce) && ce.Code != "bundle_invalid" {
		t.Fatalf("code = %q, want bundle_invalid", ce.Code)
	}
}

func TestParseReward(t *testing.T) {
	cases := []struct {
		json, txt string
		want      float64
		ok        bool
	}{
		{json: `{"reward": 1}`, want: 1, ok: true},
		{json: `{"reward": 0.85, "precision": 0.9}`, want: 0.85, ok: true},
		{json: `{"precision": 0.9}`, ok: false},
		{json: `not json`, ok: false},
		{txt: "1", want: 1, ok: true},
		{txt: "1.0\n", want: 1, ok: true},
		{txt: "0.5", want: 0.5, ok: true},
		{txt: "  0  ", want: 0, ok: true},
		{txt: "pass", ok: false},
		{txt: "", ok: false},
	}
	for _, c := range cases {
		var got float64
		var ok bool
		if c.json != "" {
			got, ok = parseRewardJSON(c.json)
		} else {
			got, ok = parseRewardText(c.txt)
		}
		if ok != c.ok || (ok && got != c.want) {
			t.Fatalf("parseReward(json=%q txt=%q) = (%v, %v), want (%v, %v)", c.json, c.txt, got, ok, c.want, c.ok)
		}
	}
}

func TestResolveEnvTemplates(t *testing.T) {
	t.Setenv("LMX_TEST_SET", "live")
	t.Setenv("LMX_TEST_EMPTY", "")
	got := resolveEnvTemplates(map[string]string{
		"plain":    "value",
		"set":      "${LMX_TEST_SET}",
		"fallback": "${LMX_TEST_UNSET_VAR:-def}",
		"empty":    "${LMX_TEST_EMPTY:-def}",
		"missing":  "${LMX_TEST_UNSET_VAR}",
	})
	want := map[string]string{"plain": "value", "set": "live", "fallback": "def", "empty": "def", "missing": ""}
	for k, v := range want {
		if got[k] != v {
			t.Fatalf("resolveEnvTemplates[%q] = %q, want %q", k, got[k], v)
		}
	}
}

func TestTerminalImportSkipsComposeTasks(t *testing.T) {
	src := t.TempDir()
	writeHarborTask := func(id string, compose bool) {
		taskDir := filepath.Join(src, id)
		mustMkdir(t, filepath.Join(taskDir, "environment"))
		mustMkdir(t, filepath.Join(taskDir, "tests"))
		mustWrite(t, filepath.Join(taskDir, "task.toml"), "[environment]\ndocker_image = \"ubuntu:24.04\"\n\n[solution]\nenv = { SOLVE_MODE = \"fast\" }\n")
		mustWrite(t, filepath.Join(taskDir, "instruction.md"), "Do it.\n")
		mustWrite(t, filepath.Join(taskDir, "tests", "test.sh"), "#!/usr/bin/env bash\n")
		if compose {
			mustWrite(t, filepath.Join(taskDir, "environment", "docker-compose.yaml"), "services: {}\n")
		}
	}
	writeHarborTask("plain-task", false)
	writeHarborTask("compose-task", true)

	out := t.TempDir()
	args := parseArgs([]string{"eval", "terminal", "import", src, "--out", out, "--version", "2.1"})
	if err := runTerminalImport(args); err != nil {
		t.Fatalf("runTerminalImport failed: %v", err)
	}
	if _, err := os.Stat(filepath.Join(out, "compose-task")); !os.IsNotExist(err) {
		t.Fatalf("compose task should be skipped, stat err = %v", err)
	}
	data, err := os.ReadFile(filepath.Join(out, "plain-task", "task.json"))
	if err != nil {
		t.Fatalf("plain task not imported: %v", err)
	}
	var task terminalTask
	if err := json.Unmarshal(data, &task); err != nil {
		t.Fatalf("task.json invalid: %v", err)
	}
	if task.Solution.Env["SOLVE_MODE"] != "fast" {
		t.Fatalf("[solution].env not imported: %#v", task.Solution.Env)
	}

	// A tree with only compose tasks must fail loudly rather than import nothing.
	onlyCompose := t.TempDir()
	src = onlyCompose
	writeHarborTask("compose-only", true)
	args = parseArgs([]string{"eval", "terminal", "import", onlyCompose, "--out", t.TempDir()})
	err = runTerminalImport(args)
	var ce cliError
	if err == nil || !errorsAsCli(err, &ce) || ce.Code != "task_import_failed" {
		t.Fatalf("expected task_import_failed for compose-only tree, got %v", err)
	}
}

// TestRunTerminalTaskHarborSemantics exercises the full task pipeline against a
// real container: the external agent must actually run (regression: branch
// wiring), reward.json must take precedence, and a non-zero verifier exit code
// must not override a written reward — harbor canonical behavior.
func TestRunTerminalTaskHarborSemantics(t *testing.T) {
	if dockerPreflight() != nil {
		t.Skip("docker unavailable")
	}
	bundle := t.TempDir()
	mustMkdir(t, filepath.Join(bundle, "tests"))
	mustWrite(t, filepath.Join(bundle, "tests", "test.sh"), "#!/bin/bash\nmkdir -p /logs/verifier\nif [ -f /agent-ran ]; then echo '{\"reward\": 1}' > /logs/verifier/reward.json; else echo 0 > /logs/verifier/reward.txt; fi\nexit 3\n")
	task := terminalTask{
		ID:          "harbor-semantics",
		Instruction: "synthetic",
		Image:       terminalImage{Prebuilt: "ubuntu:24.04"},
		Agent:       terminalAgentConfig{TimeoutSec: 60},
		Verifier:    terminalVerifierConfig{TimeoutSec: 60, Command: "bash /tests/test.sh", RewardFile: "/logs/verifier/reward.txt"},
		Environment: terminalEnvironmentConfig{CPUs: 1, MemoryMb: 512, Network: "public"},
	}
	cfg := terminalConfig{
		args:              parseArgs([]string{"eval", "terminal", "run"}),
		commandTimeoutSec: 60,
		agentTimeoutSec:   30,
		agentExecution:    "routed-shell",
		agentCommand:      `docker exec "$LMX_TERMINAL_CONTAINER" touch /agent-ran`,
	}
	res := runTerminalTask(context.Background(), task, bundle, "", "", cfg)
	if res.errCode != "" {
		t.Fatalf("unexpected error: %s %s", res.errCode, res.errText)
	}
	if !res.scored || !res.pass {
		t.Fatalf("expected scored pass (agent ran, reward.json=1 wins over exit 3): %+v", res)
	}
}

func TestWriteTerminalTaskTrace(t *testing.T) {
	traceRoot := t.TempDir()
	task := terminalTask{ID: "trace task", Instruction: "Do the traceable thing."}
	result := terminalTaskResult{
		pass:           true,
		scored:         true,
		turns:          2,
		transcript:     "# Turn 1\n## Command\n$ echo hi\nhi\n[exit=0]\n",
		verifierOutput: "reward: 1\n",
		wallTimeMs:     1234,
		usage:          terminalTokenUsage{inputTokens: 10, outputTokens: 4, cacheReadTokens: 20, totalTokens: 34, modelCalls: 1},
		prompt:         terminalSessionSystemPrompt,
	}
	cfg := terminalConfig{traceRoot: traceRoot, args: parseArgs([]string{"eval", "terminal", "run", "--quiet"})}
	if err := writeTerminalTaskTrace(task, result, cfg); err != nil {
		t.Fatalf("writeTerminalTaskTrace: %v", err)
	}
	dir := filepath.Join(traceRoot, "trace-task")
	for _, name := range []string{"instruction.txt", "prompt.txt", "transcript.md", "verifier.txt", "result.json"} {
		if _, err := os.Stat(filepath.Join(dir, name)); err != nil {
			t.Fatalf("%s not written: %v", name, err)
		}
	}
	data, err := os.ReadFile(filepath.Join(dir, "result.json"))
	if err != nil {
		t.Fatal(err)
	}
	var meta map[string]any
	if err := json.Unmarshal(data, &meta); err != nil {
		t.Fatalf("result.json invalid: %v", err)
	}
	if meta["question_id"] != "trace task" || meta["pass"] != true || meta["scored"] != true || meta["wallTimeMs"] != float64(1234) {
		t.Fatalf("unexpected result metadata: %#v", meta)
	}
	usage, _ := meta["tokenUsage"].(map[string]any)
	if usage["inputTokens"] != float64(10) || usage["outputTokens"] != float64(4) || usage["cacheReadTokens"] != float64(20) || usage["totalTokens"] != float64(34) || usage["modelCalls"] != float64(1) {
		t.Fatalf("unexpected token usage metadata: %#v", usage)
	}
}

func TestExternalAgentTokenUsageParsesOMPTrace(t *testing.T) {
	traceRoot := t.TempDir()
	traceFile := filepath.Join(traceRoot, "omp.jsonl")
	lines := []string{
		`{"type":"message_end","message":{"role":"user","usage":{"input":99,"output":99,"totalTokens":198}}}`,
		`{"type":"message_end","message":{"role":"assistant","usage":{"input":100,"output":25,"cacheRead":50,"totalTokens":175}}}`,
		`{"type":"message_end","message":{"role":"assistant","usage":{"prompt_tokens":7,"completion_tokens":3,"total_tokens":10}}}`,
	}
	if err := os.WriteFile(traceFile, []byte(strings.Join(lines, "\n")+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	usage := externalAgentTokenUsage(traceRoot)
	if usage.inputTokens != 107 || usage.outputTokens != 28 || usage.cacheReadTokens != 50 || usage.totalTokens != 185 || usage.modelCalls != 2 {
		t.Fatalf("unexpected usage: %+v", usage)
	}
}

func TestTrimTerminalMessagesCompactsOldTranscriptButKeepsInitialContext(t *testing.T) {
	messages := []map[string]any{
		{"role": "system", "content": "terminal system prompt"},
		{"role": "user", "content": "original task instructions"},
		{"role": "assistant", "content": "old command 1 " + strings.Repeat("x", 80)},
		{"role": "user", "content": "old observation 1 " + strings.Repeat("y", 80)},
		{"role": "assistant", "content": "old command 2 " + strings.Repeat("z", 80)},
		{"role": "user", "content": "old observation 2 " + strings.Repeat("q", 80)},
		{"role": "assistant", "content": "recent command 1"},
		{"role": "user", "content": "recent observation 1"},
		{"role": "assistant", "content": "recent command 2"},
		{"role": "user", "content": "recent observation 2"},
	}

	got := trimTerminalMessagesTo(messages, 120, 4)
	if len(got) != 7 {
		t.Fatalf("trimmed message count = %d, want 7: %#v", len(got), got)
	}
	if got[0]["content"] != "terminal system prompt" || got[1]["content"] != "original task instructions" {
		t.Fatalf("initial system/user context not preserved: %#v", got[:2])
	}
	if got[2]["role"] != "user" || !strings.Contains(got[2]["content"].(string), "4 old messages omitted") {
		t.Fatalf("compaction summary = %#v, want user summary for 4 omitted messages", got[2])
	}
	wantRecent := []string{"recent command 1", "recent observation 1", "recent command 2", "recent observation 2"}
	for i, want := range wantRecent {
		if got[i+3]["content"] != want {
			t.Fatalf("recent message %d = %q, want %q", i, got[i+3]["content"], want)
		}
	}
}

func TestTerminalObservationForModelIsShortAndTraceWriterKeepsTranscript(t *testing.T) {
	fullOutput := strings.Repeat("x", terminalModelObservationLimit) + "UNIQUE_TAIL_SHOULD_NOT_APPEAR"

	observation := terminalObservationForModel(fullOutput, 7, true)
	if !strings.HasPrefix(observation, fullOutput[:terminalModelObservationLimit]) {
		t.Fatal("model observation did not keep the leading command output")
	}
	if strings.Contains(observation, fullOutput[terminalModelObservationLimit:]) {
		t.Fatal("model observation included bytes past the observation limit")
	}
	if !strings.Contains(observation, "[timeout recovery hint:") || !strings.HasSuffix(observation, "\n[exit=7]") {
		t.Fatalf("model observation missing timeout hint or exit status: %q", observation)
	}

	traceRoot := t.TempDir()
	task := terminalTask{ID: "long trace"}
	result := terminalTaskResult{transcript: fullOutput + "\n[exit=7]\n"}
	cfg := terminalConfig{traceRoot: traceRoot, args: parseArgs([]string{"eval", "terminal", "run", "--quiet"})}
	if err := writeTerminalTaskTrace(task, result, cfg); err != nil {
		t.Fatalf("writeTerminalTaskTrace: %v", err)
	}
	data, err := os.ReadFile(filepath.Join(traceRoot, "long-trace", "transcript.md"))
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != result.transcript {
		t.Fatalf("trace transcript was not preserved verbatim: got %d bytes, want %d", len(data), len(result.transcript))
	}
}

func mustMkdir(t *testing.T, path string) {
	t.Helper()
	if err := os.MkdirAll(path, 0o755); err != nil {
		t.Fatal(err)
	}
}

func mustWrite(t *testing.T, path, content string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func errorsAsCli(err error, target *cliError) bool {
	if ce, ok := err.(cliError); ok {
		*target = ce
		return true
	}
	return false
}

func TestTerminalTimeoutDefaults(t *testing.T) {
	shortTask := terminalTask{Agent: terminalAgentConfig{TimeoutSec: 900}}
	longTask := terminalTask{Agent: terminalAgentConfig{TimeoutSec: defaultTerminalTaskTimeoutSec + 60}}

	if got := terminalAgentTimeoutSec(terminalConfig{}, shortTask); got != defaultTerminalTaskTimeoutSec {
		t.Fatalf("short manifest timeout = %d, want %d", got, defaultTerminalTaskTimeoutSec)
	}
	if got := terminalAgentTimeoutSec(terminalConfig{}, longTask); got != defaultTerminalTaskTimeoutSec+60 {
		t.Fatalf("long manifest timeout = %d, want %d", got, defaultTerminalTaskTimeoutSec+60)
	}
	if got := terminalAgentTimeoutSec(terminalConfig{agentTimeoutSec: 42}, shortTask); got != 42 {
		t.Fatalf("explicit agent timeout = %d, want 42", got)
	}
	if got := terminalCommandTimeoutSec(terminalConfig{}); got != defaultTerminalCommandTimeoutSec {
		t.Fatalf("default command timeout = %d, want %d", got, defaultTerminalCommandTimeoutSec)
	}
	if got := terminalCommandTimeoutSec(terminalConfig{commandTimeoutSec: 30}); got != 30 {
		t.Fatalf("explicit command timeout = %d, want 30", got)
	}
	if got := terminalEndpointTimeout(terminalConfig{}); got != defaultTerminalEndpointTimeout {
		t.Fatalf("default endpoint timeout = %v, want %v", got, defaultTerminalEndpointTimeout)
	}
	cmdDeadline := time.Now().Add(time.Hour)
	if got := terminalCommandExecutionTimeout("echo ok", 30*time.Minute, cmdDeadline); got > 30*time.Minute || got < 29*time.Minute {
		t.Fatalf("normal command timeout = %v, want about 30m", got)
	}
	if got := terminalCommandExecutionTimeout("apt-get update && apt-get install -y git", 30*time.Minute, cmdDeadline); got > 30*time.Minute || got < 29*time.Minute {
		t.Fatalf("setup command timeout = %v, want about 30m", got)
	}
	if got := terminalCommandExecutionTimeout("timeout 120 7z x /app/secrets.7z -o/app", 30*time.Minute, cmdDeadline); got > 30*time.Minute || got < 29*time.Minute {
		t.Fatalf("explicit timeout command timeout = %v, want about 30m", got)
	}
}

func TestTerminalShellPayloadProtectsCompletionMarker(t *testing.T) {
	payload := terminalShellPayload("cd /app\ncat > sample.txt <<'EOF'\nhello\nEOF", "__DONE__")
	if !strings.Contains(payload, "{\ncd /app\ncat > sample.txt <<'EOF'\nhello\nEOF\n} </dev/null\n") {
		t.Fatalf("payload should wrap command block with stdin from /dev/null, got:\n%s", payload)
	}
	if !strings.Contains(payload, "printf '\\n__DONE__%d__\\n' \"$__lmx_status\"") {
		t.Fatalf("payload should print completion marker after command block, got:\n%s", payload)
	}
}

func TestTerminalProtocolLabel(t *testing.T) {
	if got := terminalProtocolLabel(terminalConfig{shellMode: "persistent"}); got != "react-shell" {
		t.Fatalf("persistent: got %q want react-shell", got)
	}
	if got := terminalProtocolLabel(terminalConfig{shellMode: "stateless"}); got != "react-bash" {
		t.Fatalf("stateless: got %q want react-bash", got)
	}
	if got := terminalProtocolLabel(terminalConfig{shellMode: "persistent", agentCommand: "x"}); got != "external-command/host" {
		t.Fatalf("agentCommand should win: got %q want external-command/host", got)
	}
	if got := terminalProtocolLabel(terminalConfig{shellMode: "persistent", agentCommand: "x", agentExecution: "container"}); got != "external-command/container" {
		t.Fatalf("agentExecution should be included: got %q want external-command/container", got)
	}
}

func TestTerminalShellPersistsState(t *testing.T) {
	if dockerPreflight() != nil {
		t.Skip("docker unavailable")
	}
	name := "lmx-shelltest-" + randomHex(6)
	if _, code, _, err := runCommand(context.Background(), 60*time.Second, "docker", "run", "-d", "--rm", "--name", name, "ubuntu:24.04", "sleep", "infinity"); err != nil || code != 0 {
		t.Skipf("could not start test container (code=%d err=%v)", code, err)
	}
	defer runCommand(context.Background(), 30*time.Second, "docker", "rm", "-f", name)

	shell, err := startTerminalShell(name, "")
	if err != nil {
		t.Fatalf("startTerminalShell: %v", err)
	}
	defer shell.close()

	if _, code, _, restarted := shell.exec("cd /tmp && export FOO=bar", 30*time.Second); code != 0 || restarted {
		t.Fatalf("setup exec: code=%d restarted=%v", code, restarted)
	}
	out, code, _, restarted := shell.exec("pwd && echo $FOO", 30*time.Second)
	if code != 0 || restarted {
		t.Fatalf("state exec: code=%d restarted=%v out=%q", code, restarted, out)
	}
	if !strings.Contains(out, "/tmp") || !strings.Contains(out, "bar") {
		t.Fatalf("state not persisted: out=%q", out)
	}

	// Killing the shell (running `exit`) must trigger recovery. The EOF is
	// observed by whichever exec races the shell death, so accept a restart
	// flag on either call; the invariant is that the session keeps working.
	_, _, _, restartedExit := shell.exec("exit", 5*time.Second)
	out2, code2, _, restartedAlive := shell.exec("echo alive", 30*time.Second)
	if !restartedExit && !restartedAlive {
		t.Fatalf("expected restart after shell death, got neither; out=%q", out2)
	}
	if code2 != 0 || !strings.Contains(out2, "alive") {
		t.Fatalf("post-restart exec: code=%d out=%q", code2, out2)
	}
}

func TestRunTerminalAgentLoopSessionContinuesAfterCommandTimeout(t *testing.T) {
	if dockerPreflight() != nil {
		t.Skip("docker unavailable")
	}
	name := "lmx-timeout-recovery-" + randomHex(6)
	if _, code, _, err := runCommand(context.Background(), 60*time.Second, "docker", "run", "-d", "--rm", "--name", name, "ubuntu:24.04", "sleep", "infinity"); err != nil || code != 0 {
		t.Skipf("could not start test container (code=%d err=%v)", code, err)
	}
	defer runCommand(context.Background(), 30*time.Second, "docker", "rm", "-f", name)

	modelCalls := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		modelCalls++
		response := terminalJSONResponse{
			Analysis: "Exercise a command timeout.",
			Plan:     "Run a command that exceeds its bound.",
			Commands: []terminalJSONCommand{
				{Keystrokes: "sleep 2\n", Duration: 0.1},
				{Keystrokes: "printf should-not-run > /tmp/skipped\n", Duration: 0.1},
			},
		}
		if modelCalls > 1 {
			response = terminalJSONResponse{
				Analysis:     "Recover after the bounded command was stopped.",
				Plan:         "Write durable evidence and finish.",
				Commands:     []terminalJSONCommand{{Keystrokes: "printf recovered > /tmp/recovered\n", Duration: 0.1}},
				TaskComplete: true,
			}
		}
		content, err := json.Marshal(response)
		if err != nil {
			t.Errorf("encode model response: %v", err)
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []any{map[string]any{"message": map[string]any{"content": string(content)}}},
		})
	}))
	defer server.Close()

	turns, transcript, _, err := runTerminalAgentLoopSession(context.Background(), terminalTask{
		ID:          "timeout-recovery-test",
		Instruction: "Recover after a bounded command.",
		Agent:       terminalAgentConfig{MaxTurns: 3},
	}, name, server.URL, "fixture-model", terminalConfig{
		args:              parseArgs([]string{"eval", "terminal", "run", "--quiet"}),
		commandTimeoutSec: 1,
		agentTimeoutSec:   20,
		endpointTimeout:   5 * time.Second,
		repeatBatchLimit:  3,
	})
	if err != nil {
		t.Fatalf("runTerminalAgentLoopSession: %v\ntranscript:\n%s", err, transcript)
	}
	if turns != 2 || modelCalls != 2 {
		t.Fatalf("turns=%d modelCalls=%d, want two recovery turns", turns, modelCalls)
	}
	if !strings.Contains(transcript, "[command timed out]") || !strings.Contains(transcript, "printf recovered") {
		t.Fatalf("transcript does not show timeout followed by recovery:\n%s", transcript)
	}
	out, code, _, execErr := runCommand(context.Background(), 30*time.Second, "docker", "exec", name, "cat", "/tmp/recovered")
	if execErr != nil || code != 0 || strings.TrimSpace(out) != "recovered" {
		t.Fatalf("recovery marker: code=%d err=%v out=%q", code, execErr, out)
	}
	if out, code, _, execErr := runCommand(context.Background(), 30*time.Second, "docker", "exec", name, "sh", "-c", "test ! -e /tmp/skipped"); execErr != nil || code != 0 {
		t.Fatalf("post-timeout command batch continued unexpectedly: code=%d err=%v out=%q", code, execErr, out)
	}
}

func TestRunTerminalAgentLoopAppliesFiniteAndRetryTokenCaps(t *testing.T) {
	tests := []struct {
		name          string
		configuredCap int
		wantCaps      []int
	}{
		{name: "omitted cap is finite and retry is smaller", wantCaps: []int{16384, 8192}},
		{name: "explicit cap remains authoritative", configuredCap: 1234, wantCaps: []int{1234, 1234}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var gotCaps []int
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var request struct {
					MaxTokens          int            `json:"max_tokens"`
					ChatTemplateKwargs map[string]any `json:"chat_template_kwargs"`
				}
				if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
					t.Errorf("decode model request: %v", err)
					return
				}
				gotCaps = append(gotCaps, request.MaxTokens)
				if request.ChatTemplateKwargs["enable_thinking"] != false {
					t.Errorf("enable_thinking = %v, want false for Qwen terminal requests", request.ChatTemplateKwargs["enable_thinking"])
				}
				if len(gotCaps) == 1 {
					http.Error(w, "retry this request", http.StatusServiceUnavailable)
					return
				}
				_ = json.NewEncoder(w).Encode(map[string]any{
					"choices": []any{map[string]any{"message": map[string]any{"content": "TASK_COMPLETE"}}},
				})
			}))
			defer server.Close()

			_, _, _, err := runTerminalAgentLoop(context.Background(), terminalTask{
				ID:          "token-cap-test",
				Instruction: "Finish without executing a command.",
				Agent:       terminalAgentConfig{MaxTurns: 1},
			}, "unused-container", server.URL, "Qwen3.6-27b", terminalConfig{
				args:            parseArgs([]string{"eval", "terminal", "run", "--quiet"}),
				maxTokens:       tt.configuredCap,
				agentTimeoutSec: 30,
				endpointTimeout: 5 * time.Second,
			})
			if err != nil {
				t.Fatalf("runTerminalAgentLoop: %v", err)
			}
			if len(gotCaps) != len(tt.wantCaps) {
				t.Fatalf("model request caps = %v, want %v", gotCaps, tt.wantCaps)
			}
			for i, want := range tt.wantCaps {
				if gotCaps[i] != want {
					t.Fatalf("model request %d max_tokens = %d, want %d", i+1, gotCaps[i], want)
				}
			}
		})
	}
}

func TestTerminalModelRequestTimeoutReservesRetryBudget(t *testing.T) {
	deadline := time.Now().Add(12 * time.Minute)
	cfg := terminalConfig{endpointTimeout: 30 * time.Minute}

	firstAttempt := terminalModelRequestTimeout(cfg, deadline, true)
	retry := terminalModelRequestTimeout(cfg, deadline, false)
	if firstAttempt < 7*time.Minute+59*time.Second || firstAttempt > 8*time.Minute {
		t.Fatalf("first-attempt timeout = %v, want about 8m with retry budget reserved", firstAttempt)
	}
	if retry < 11*time.Minute+59*time.Second || retry > 12*time.Minute {
		t.Fatalf("retry timeout = %v, want the remaining task budget", retry)
	}
	if retry-firstAttempt < 3*time.Minute+59*time.Second {
		t.Fatalf("first attempt reserved only %v for retry, want about 4m", retry-firstAttempt)
	}

	endpointLimited := terminalModelRequestTimeout(terminalConfig{endpointTimeout: 90 * time.Second}, deadline, true)
	if endpointLimited < 89*time.Second || endpointLimited > 90*time.Second {
		t.Fatalf("endpoint-limited first-attempt timeout = %v, want about 90s", endpointLimited)
	}
}

func TestTerminalModelCallFailureDistinguishesAgentTimeout(t *testing.T) {
	firstErr := errors.New("first")
	retryErr := errors.New("retry")

	timedOut := terminalModelCallFailure("task", time.Now().Add(-time.Second), firstErr, retryErr)
	if timedOut.Code != "agent_timeout" {
		t.Fatalf("expired-deadline error code = %q, want agent_timeout", timedOut.Code)
	}

	failed := terminalModelCallFailure("task", time.Now().Add(time.Minute), firstErr, retryErr)
	if failed.Code != "model_call_failed" {
		t.Fatalf("live-deadline error code = %q, want model_call_failed", failed.Code)
	}
}

func TestRunTerminalTaskPersistsTraceForEarlyBuiltInAgentError(t *testing.T) {
	binDir := t.TempDir()
	dockerPath := filepath.Join(binDir, "docker")
	if err := os.WriteFile(dockerPath, []byte("#!/bin/sh\nexit 0\n"), 0o755); err != nil {
		t.Fatalf("write fake docker: %v", err)
	}
	t.Setenv("PATH", binDir+string(os.PathListSeparator)+os.Getenv("PATH"))

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "synthetic upstream failure", http.StatusServiceUnavailable)
	}))
	defer server.Close()

	for _, shellMode := range []string{"stateless", "persistent"} {
		t.Run(shellMode, func(t *testing.T) {
			traceRoot := t.TempDir()
			task := terminalTask{
				ID:          "early-error-" + shellMode,
				Instruction: "This instruction must survive the early model error.",
				Image:       terminalImage{Prebuilt: "fake:image"},
				Agent:       terminalAgentConfig{MaxTurns: 1},
			}
			result := runTerminalTask(context.Background(), task, t.TempDir(), server.URL, "test-model", terminalConfig{
				args:            parseArgs([]string{"eval", "terminal", "run", "--quiet"}),
				shellMode:       shellMode,
				traceRoot:       traceRoot,
				agentTimeoutSec: 10,
				endpointTimeout: time.Second,
			})
			if result.errCode != "model_call_failed" {
				t.Fatalf("runTerminalTask error code = %q, want model_call_failed; result=%+v", result.errCode, result)
			}

			traceDir := filepath.Join(traceRoot, sanitizeDockerName(task.ID))
			instruction, err := os.ReadFile(filepath.Join(traceDir, "instruction.txt"))
			if err != nil {
				t.Fatalf("read early-error instruction trace: %v", err)
			}
			if string(instruction) != task.Instruction {
				t.Fatalf("instruction trace = %q, want %q", instruction, task.Instruction)
			}
			data, err := os.ReadFile(filepath.Join(traceDir, "result.json"))
			if err != nil {
				t.Fatalf("read early-error result trace: %v", err)
			}
			var metadata map[string]any
			if err := json.Unmarshal(data, &metadata); err != nil {
				t.Fatalf("decode early-error result trace: %v", err)
			}
			if metadata["errorCode"] != "model_call_failed" {
				t.Fatalf("trace errorCode = %#v, want model_call_failed", metadata["errorCode"])
			}
		})
	}
}

func TestSubmitTerminalEvalDryRunBuildsCanonicalPayloadWithoutNetworkOrBenchmarkExecution(t *testing.T) {
	runDir, hardwarePath := writeDeferredTerminalSubmitFixture(t)
	payloadPath := filepath.Join(t.TempDir(), "payload.json")
	binDir := t.TempDir()
	benchmarkSentinel := filepath.Join(t.TempDir(), "benchmark-executed")
	mustWrite(t, filepath.Join(binDir, "docker"), "#!/bin/sh\nprintf invoked > \"$LMX_TEST_SENTINEL\"\nexit 99\n")
	if err := os.Chmod(filepath.Join(binDir, "docker"), 0o755); err != nil {
		t.Fatalf("make fake docker executable: %v", err)
	}
	t.Setenv("LMX_TEST_SENTINEL", benchmarkSentinel)
	t.Setenv("PATH", binDir)

	var networkCalls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		networkCalls.Add(1)
		http.Error(w, "dry-run must not reach the submission API", http.StatusInternalServerError)
	}))
	defer server.Close()

	args := parseArgs([]string{
		"eval", "terminal", "submit", runDir,
		"--dataset", "terminal-bench-fixture",
		"--shard-index", "7",
		"--hf-id", "fixture/model",
		"--model-revision", "fixture-revision",
		"--hardware", hardwarePath,
		"--quantization", "Q4_K_M",
		"--quant-format", "gguf",
		"--agent-name", "fixture-agent",
		"--runner-version", "fixture-runner",
		"--notes", "fixture-notes",
		"--api-url", server.URL,
		"--out", payloadPath,
		"--dry-run", "--quiet",
	})
	if err := submitTerminalEval(args); err != nil {
		t.Fatalf("submitTerminalEval dry-run: %v", err)
	}
	if got := networkCalls.Load(); got != 0 {
		t.Fatalf("dry-run made %d network requests, want none", got)
	}
	if _, err := os.Stat(benchmarkSentinel); !os.IsNotExist(err) {
		t.Fatalf("deferred dry-run executed benchmark tooling: %v", err)
	}

	data, err := os.ReadFile(payloadPath)
	if err != nil {
		t.Fatalf("read durable payload: %v", err)
	}
	if !utf8.Valid(data) {
		t.Fatal("durable payload is not valid UTF-8")
	}
	var batch map[string]any
	if err := json.Unmarshal(data, &batch); err != nil {
		t.Fatalf("decode durable payload batch: %v", err)
	}
	if batch["dataset"] != "terminal-bench-fixture" {
		t.Fatalf("batch dataset = %#v, want terminal-bench-fixture", batch["dataset"])
	}
	shards := anySlice(batch["shards"])
	if len(shards) != 1 {
		t.Fatalf("explicit checkpoint payload count = %d, want 1", len(shards))
	}
	payload := asObject(shards[0])
	if payload["shardIndex"] != float64(7) {
		t.Fatalf("explicit checkpoint shardIndex = %#v, want 7", payload["shardIndex"])
	}
	if payload["hfId"] != "fixture/model" || payload["modelRevision"] != "fixture-revision" {
		t.Fatalf("model identity = (%#v, %#v), want explicit fixture identity", payload["hfId"], payload["modelRevision"])
	}
	if payload["quantization"] != "Q4_K_M" || payload["quantFormat"] != "gguf" {
		t.Fatalf("quantization metadata = (%#v, %#v), want (Q4_K_M, gguf)", payload["quantization"], payload["quantFormat"])
	}
	if payload["runnerVersion"] != "fixture-runner" || payload["notes"] != "fixture-notes" {
		t.Fatalf("explicit run metadata missing: runnerVersion=%#v notes=%#v", payload["runnerVersion"], payload["notes"])
	}
	hardware := asObject(payload["hardware"])
	if hardware["hwClass"] != "CPU_ONLY" || hardware["cpu"] != "Fixture CPU" || hardware["ramGb"] != float64(32) {
		t.Fatalf("hardware payload = %#v, want fixture hardware", hardware)
	}
	runConfig := asObject(payload["runConfig"])
	if runConfig["accuracy"] != 0.5 || runConfig["tasksRun"] != float64(2) || runConfig["errors"] != float64(0) || runConfig["avgLatencyMs"] != float64(150) {
		t.Fatalf("runConfig aggregate metrics = %#v, want one pass, one fail, and 150ms average", runConfig)
	}
	if runConfig["protocol"] != "deferred-saved-terminal-run" || runConfig["agent"] != "fixture-agent" || runConfig["deferredSubmit"] != true {
		t.Fatalf("runConfig deferred identity = %#v", runConfig)
	}
	totalUsage := asObject(runConfig["tokenUsage"])
	wantTotalUsage := map[string]any{
		"inputTokens":      float64(16),
		"outputTokens":     float64(10),
		"cacheReadTokens":  float64(6),
		"cacheWriteTokens": float64(4),
		"totalTokens":      float64(36),
		"modelCalls":       float64(5),
	}
	if !reflect.DeepEqual(totalUsage, wantTotalUsage) {
		t.Fatalf("runConfig token usage = %#v, want exact aggregate %#v", totalUsage, wantTotalUsage)
	}

	results := anySlice(payload["results"])
	if len(results) != 2 {
		t.Fatalf("results count = %d, want 2", len(results))
	}
	passes := map[string]bool{}
	for _, raw := range results {
		result := asObject(raw)
		id := stringValue(result["question_id"])
		if _, duplicate := passes[id]; duplicate {
			t.Fatalf("duplicate result for task %q: %#v", id, results)
		}
		passes[id] = terminalBoolField(result, "pass")
	}
	if len(passes) != 2 || !passes["pass-task"] || passes["fail-task"] {
		t.Fatalf("pass/fail records = %#v, want unique pass-task=true and fail-task=false", passes)
	}

	artifacts := anySlice(payload["artifacts"])
	if len(artifacts) != 2 {
		t.Fatalf("artifacts count = %d, want 2", len(artifacts))
	}
	artifactsByTask := map[string]map[string]any{}
	for _, raw := range artifacts {
		artifact := asObject(raw)
		id := stringValue(artifact["question_id"])
		if artifactsByTask[id] != nil {
			t.Fatalf("duplicate artifact for task %q", id)
		}
		artifactsByTask[id] = artifact
	}
	passArtifact := artifactsByTask["pass-task"]
	if passArtifact["itemIndex"] != float64(1) || passArtifact["question"] != "Pass question with café" || passArtifact["prompt"] != "Pass prompt" || passArtifact["score"] != float64(1) || passArtifact["testPassed"] != true {
		t.Fatalf("passing artifact schema values = %#v", passArtifact)
	}
	if passArtifact["latencyMs"] != float64(100) || passArtifact["wallTimeMs"] != float64(100) {
		t.Fatalf("passing artifact latency = (%#v, %#v), want saved wall time", passArtifact["latencyMs"], passArtifact["wallTimeMs"])
	}
	usage := asObject(passArtifact["tokenUsage"])
	if usage["inputTokens"] != float64(11) || usage["outputTokens"] != float64(7) || usage["cacheReadTokens"] != float64(2) || usage["cacheWriteTokens"] != float64(3) || usage["totalTokens"] != float64(23) || usage["modelCalls"] != float64(2) {
		t.Fatalf("passing artifact token usage = %#v", usage)
	}

	response := stringValue(passArtifact["response"])
	if !utf8.ValidString(response) {
		t.Fatal("canonical trace response is not valid UTF-8")
	}
	if len(response) > terminalArtifactResponseBytes {
		t.Fatalf("canonical trace response is %d bytes, limit is %d", len(response), terminalArtifactResponseBytes)
	}
	for _, want := range []string{
		"# Agent trace",
		"raw JSONL is not embedded",
		"## Assistant",
		"First finalized answer — naïve café",
		"## Tool activity",
		"Tool: bash",
		"Intent: Inspect résumé safely",
		"command: printf 'tool ✓'",
		"Outcome: completed",
		"tool output λ",
		"## Final answer",
		"Final answer: completed safely — café 🚀",
		"Stored bytes: 321",
		"Dropped streaming events: message_update=7",
		"Overflow bytes: 44",
		"Streaming message deltas observed (not required): 0",
		"Unknown event types ignored: future_event=1",
		"## Verifier",
		"VERIFIER_HEAD_検証",
		"VERIFIER_TAIL_合格",
	} {
		if !strings.Contains(response, want) {
			t.Fatalf("canonical response missing %q:\n%s", want, response)
		}
	}
	if strings.Contains(response, "RAW_JSON_FIELD_MUST_NOT_APPEAR") || strings.Contains(response, `"privateRaw"`) {
		t.Fatalf("canonical response embedded raw JSON event data:\n%s", response)
	}

	failArtifact := artifactsByTask["fail-task"]
	if failArtifact["itemIndex"] != float64(0) || failArtifact["score"] != float64(0) || failArtifact["testPassed"] != false {
		t.Fatalf("failing artifact schema values = %#v", failArtifact)
	}
	if failArtifact["latencyMs"] != float64(200) || failArtifact["wallTimeMs"] != float64(200) {
		t.Fatalf("failing artifact latency = (%#v, %#v), want latency fallback", failArtifact["latencyMs"], failArtifact["wallTimeMs"])
	}
	failResponse := stringValue(failArtifact["response"])
	if !strings.Contains(failResponse, "SAVED_FAIL_RESPONSE") || !strings.Contains(failResponse, "FAIL_VERIFIER_RETAINED") {
		t.Fatalf("fallback artifact did not retain saved response and verifier output:\n%s", failResponse)
	}
}

func TestSubmitTerminalEvalRejectsInvalidCheckpointRecords(t *testing.T) {
	tests := []struct {
		name     string
		mutate   func(t *testing.T, runDir string)
		wantCode string
	}{
		{
			name: "duplicate summary task",
			mutate: func(t *testing.T, runDir string) {
				writeTerminalTestJSON(t, filepath.Join(runDir, "summary.json"), []any{
					map[string]any{"index": 1, "total": 2, "task": "pass-task", "out": "pass-task.json", "pass": true, "scored": true},
					map[string]any{"index": 2, "total": 2, "task": "pass-task", "out": "pass-task.json", "pass": true, "scored": true},
				})
			},
			wantCode: "duplicate_summary_task",
		},
		{
			name: "missing result file",
			mutate: func(t *testing.T, runDir string) {
				if err := os.Remove(filepath.Join(runDir, "fail-task.json")); err != nil {
					t.Fatalf("remove result fixture: %v", err)
				}
			},
			wantCode: "task_result_missing",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			runDir, hardwarePath := writeDeferredTerminalSubmitFixture(t)
			tc.mutate(t, runDir)
			args := parseArgs([]string{
				"eval", "terminal", "submit", runDir,
				"--dataset", "terminal-bench-fixture",
				"--shard-index", "7",
				"--hf-id", "fixture/model",
				"--hardware", hardwarePath,
				"--quantization", "Q4_K_M",
				"--quant-format", "gguf",
				"--dry-run", "--quiet",
			})
			err := submitTerminalEval(args)
			var cliErr cliError
			if !errors.As(err, &cliErr) || cliErr.Code != tc.wantCode {
				t.Fatalf("error = %#v, want cliError code %q", err, tc.wantCode)
			}
		})
	}
}

func writeTerminalTestJSON(t *testing.T, path string, value any) {
	t.Helper()
	if err := writeJSON(path, value); err != nil {
		t.Fatal(err)
	}
}

func writeDeferredTerminalSubmitFixture(t *testing.T) (runDir, hardwarePath string) {
	t.Helper()
	runDir = t.TempDir()
	hardwarePath = filepath.Join(t.TempDir(), "hardware.json")
	writeTerminalTestJSON(t, hardwarePath, map[string]any{
		"hwClass": "CPU_ONLY",
		"cpu":     "Fixture CPU",
		"ramGb":   32,
	})
	writeTerminalTestJSON(t, filepath.Join(runDir, "summary.json"), []any{
		map[string]any{
			"index": 1, "total": 2, "task": "pass-task", "out": "pass-task.json", "pass": true, "scored": true,
			"summary": map[string]any{"quantization": "Q4_K_M", "quantFormat": "gguf"},
		},
		map[string]any{
			"index": 2, "total": 2, "task": "fail-task", "out": "fail-task.json", "pass": false, "scored": true,
			"summary": map[string]any{"quantization": "Q4_K_M", "quantFormat": "gguf"},
		},
	})
	verifier := "VERIFIER_HEAD_検証\n" + strings.Repeat("界", 4000) + "\nVERIFIER_TAIL_合格"
	writeTerminalTestJSON(t, filepath.Join(runDir, "pass-task.json"), map[string]any{"results": []any{map[string]any{
		"question_id":    "pass-task",
		"pass":           true,
		"scored":         true,
		"wallTimeMs":     100,
		"tokenUsage":     map[string]any{"inputTokens": 11, "outputTokens": 7, "cacheReadTokens": 2, "cacheWriteTokens": 3, "totalTokens": 23, "modelCalls": 2},
		"turns":          2,
		"question":       "Pass question with café",
		"prompt":         "Pass prompt",
		"response":       "SAVED_PASS_RESPONSE_MUST_NOT_REPLACE_OMP",
		"verifierOutput": verifier,
	}}})
	writeTerminalTestJSON(t, filepath.Join(runDir, "fail-task.json"), map[string]any{"results": []any{map[string]any{
		"question_id":    "fail-task",
		"pass":           false,
		"scored":         true,
		"latencyMs":      200,
		"tokenUsage":     map[string]any{"input_tokens": 5, "output_tokens": 3, "cache_read_tokens": 4, "cache_write_tokens": 1, "total_tokens": 13, "model_calls": 3},
		"turns":          1,
		"question":       "Fail question",
		"prompt":         "Fail prompt",
		"response":       "SAVED_FAIL_RESPONSE",
		"verifierOutput": "FAIL_VERIFIER_RETAINED",
	}}})

	tracePath := filepath.Join(runDir, "traces", "pass-task", "pass-task", "agent", "fixture-session", "omp.jsonl")
	mustMkdir(t, filepath.Dir(tracePath))
	events := []any{
		map[string]any{
			"type": "message_end",
			"message": map[string]any{"role": "assistant", "content": []any{
				map[string]any{"type": "text", "text": "First finalized answer — naïve café"},
			}},
		},
		map[string]any{
			"type": "tool_execution_end", "toolName": "bash", "intent": "Inspect résumé safely",
			"args":   map[string]any{"command": "printf 'tool ✓'"},
			"result": map[string]any{"content": []any{map[string]any{"type": "text", "text": "tool output λ"}}},
		},
		map[string]any{"type": "future_event", "privateRaw": "RAW_JSON_FIELD_MUST_NOT_APPEAR"},
		map[string]any{
			"type": "trace_filter_summary", "storedBytes": 321, "overflowBytes": 44,
			"droppedEvents": map[string]any{"message_update": 7},
		},
		map[string]any{
			"type": "message_end",
			"message": map[string]any{"role": "assistant", "content": []any{
				map[string]any{"type": "text", "text": "Final answer: completed safely — café 🚀"},
			}},
		},
	}
	file, err := os.Create(tracePath)
	if err != nil {
		t.Fatalf("create OMP fixture: %v", err)
	}
	encoder := json.NewEncoder(file)
	for _, event := range events {
		if err := encoder.Encode(event); err != nil {
			file.Close()
			t.Fatalf("encode OMP fixture: %v", err)
		}
	}
	if err := file.Close(); err != nil {
		t.Fatalf("close OMP fixture: %v", err)
	}
	return runDir, hardwarePath
}

const terminalBench21CanonicalTestTaskIDsText = `adaptive-rejection-sampler
bn-fit-modify
break-filter-js-from-html
build-cython-ext
build-pmars
build-pov-ray
caffe-cifar-10
cancel-async-tasks
chess-best-move
circuit-fibsqrt
cobol-modernization
code-from-image
compile-compcert
configure-git-webserver
constraints-scheduling
count-dataset-tokens
crack-7z-hash
custom-memory-heap-crash
db-wal-recovery
distribution-search
dna-assembly
dna-insert
extract-elf
extract-moves-from-video
feal-differential-cryptanalysis
feal-linear-cryptanalysis
filter-js-from-html
financial-document-processor
fix-code-vulnerability
fix-git
fix-ocaml-gc
gcode-to-text
git-leak-recovery
git-multibranch
gpt2-codegolf
headless-terminal
hf-model-inference
install-windows-3.11
kv-store-grpc
large-scale-text-editing
largest-eigenval
llm-inference-batching-scheduler
log-summary-date-ranges
mailman
make-doom-for-mips
make-mips-interpreter
mcmc-sampling-stan
merge-diff-arc-agi-task
model-extraction-relu-logits
modernize-scientific-stack
mteb-leaderboard
mteb-retrieve
multi-source-data-merger
nginx-request-logging
openssl-selfsigned-cert
overfull-hbox
password-recovery
path-tracing
path-tracing-reverse
polyglot-c-py
polyglot-rust-c
portfolio-optimization
protein-assembly
prove-plus-comm
pypi-server
pytorch-model-cli
pytorch-model-recovery
qemu-alpine-ssh
qemu-startup
query-optimize
raman-fitting
regex-chess
regex-log
reshard-c4-data
rstan-to-pystan
sam-cell-seg
sanitize-git-repo
schemelike-metacircular-eval
sparql-university
sqlite-db-truncate
sqlite-with-gcov
torch-pipeline-parallelism
torch-tensor-parallelism
train-fasttext
tune-mjcf
video-processing
vulnerable-secret
winning-avg-corewars
write-compressor`

func TestInspectTerminalDatasetRequiresExplicitAPIURLBeforeHTTP(t *testing.T) {
	counter := &terminalInspectRequestCounter{}
	originalClient := apiHTTPClient
	apiHTTPClient = &http.Client{Transport: counter}
	t.Cleanup(func() { apiHTTPClient = originalClient })

	err := inspectTerminalDataset(terminalBench21Dataset, parseArgs([]string{
		"eval", "terminal", "inspect", terminalBench21Dataset,
		"--json",
	}))
	var cliErr cliError
	if !errors.As(err, &cliErr) || cliErr.Code != "missing_option" || cliErr.Message != "--api-url is required" {
		t.Fatalf("error = %#v, want missing_option for --api-url", err)
	}
	if got := counter.calls.Load(); got != 0 {
		t.Fatalf("inspect without --api-url made %d HTTP requests, want none", got)
	}
}

func TestInspectTerminalDatasetRoutesToConfiguredAPIAndReportsCanonicalReadiness(t *testing.T) {
	fixture := newTerminalInspectFixture(t, false)
	fixture.wantAuthorization = "Bearer fixture-key"
	server := fixture.start(t)
	defer server.Close()

	var modelCalls atomic.Int32
	modelServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		modelCalls.Add(1)
		http.Error(w, "inspect must not contact a model endpoint", http.StatusInternalServerError)
	}))
	defer modelServer.Close()
	dockerSentinel := filepath.Join(t.TempDir(), "docker-called")
	binDir := t.TempDir()
	dockerPath := filepath.Join(binDir, "docker")
	mustWrite(t, dockerPath, "#!/bin/sh\necho called > "+dockerSentinel+"\n")
	if err := os.Chmod(dockerPath, 0o755); err != nil {
		t.Fatalf("chmod fake docker: %v", err)
	}
	t.Setenv("PATH", binDir+string(os.PathListSeparator)+os.Getenv("PATH"))

	output, err := captureTerminalTestStdout(t, func() error {
		return inspectTerminalDataset(terminalBench21Dataset, parseArgs([]string{
			"eval", "terminal", "inspect", terminalBench21Dataset,
			"--api-url", server.URL,
			"--api-key", "fixture-key",
			"--base-url", modelServer.URL,
			"--json",
		}))
	})
	if err != nil {
		t.Fatalf("inspect canonical dataset: %v", err)
	}
	var summary map[string]any
	if err := json.Unmarshal([]byte(output), &summary); err != nil {
		t.Fatalf("decode inspect JSON %q: %v", output, err)
	}
	if summary["ready"] != true || summary["dataset"] != terminalBench21Dataset {
		t.Fatalf("inspect identity = %#v", summary)
	}
	for field, want := range map[string]float64{
		"itemCount": 89, "shardCount": 10, "manifestItems": 89, "uniqueTaskIds": 89,
	} {
		if summary[field] != want {
			t.Fatalf("inspect %s = %#v, want %.0f", field, summary[field], want)
		}
	}
	shards := anySlice(summary["shards"])
	wantSizes := []int{8, 9, 9, 9, 9, 9, 9, 9, 9, 9}
	if len(shards) != len(wantSizes) {
		t.Fatalf("inspect shard summaries = %d, want %d", len(shards), len(wantSizes))
	}
	for i, wantSize := range wantSizes {
		shard := asObject(shards[i])
		if shard["shardIndex"] != float64(i+1) || shard["itemCount"] != float64(wantSize) {
			t.Fatalf("inspect shard %d = %#v, want index %d size %d", i+1, shard, i+1, wantSize)
		}
	}
	wantIndexes := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	if !reflect.DeepEqual(fixture.shardRequests, wantIndexes) {
		t.Fatalf("shard requests = %v, want %v", fixture.shardRequests, wantIndexes)
	}
	if !reflect.DeepEqual(fixture.manifestRequests, wantIndexes) {
		t.Fatalf("manifest requests = %v, want %v", fixture.manifestRequests, wantIndexes)
	}
	if fixture.bundleRequests != 0 {
		t.Fatalf("default inspect downloaded %d bundles, want manifest-only inspection", fixture.bundleRequests)
	}
	if fixture.unexpectedRequests != 0 {
		t.Fatalf("inspect made %d unexpected API/submit requests", fixture.unexpectedRequests)
	}
	if got := modelCalls.Load(); got != 0 {
		t.Fatalf("inspect made %d model requests", got)
	}
	if _, err := os.Stat(dockerSentinel); !os.IsNotExist(err) {
		t.Fatalf("inspect executed Docker: %v", err)
	}
}

func TestInspectTerminalDatasetRejectsInvalidCanonicalManifests(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*terminalInspectFixture)
	}{
		{
			name: "duplicate id within shard",
			mutate: func(f *terminalInspectFixture) {
				f.shards[1][1]["question_id"] = f.shards[1][0]["question_id"]
			},
		},
		{
			name: "cross-shard duplicate id",
			mutate: func(f *terminalInspectFixture) {
				f.shards[1][0]["question_id"] = f.shards[0][0]["question_id"]
			},
		},
		{
			name: "missing id",
			mutate: func(f *terminalInspectFixture) {
				f.shards[9] = f.shards[9][:len(f.shards[9])-1]
			},
		},
		{
			name: "extra id",
			mutate: func(f *terminalInspectFixture) {
				f.shards[9] = append(f.shards[9], terminalInspectManifestRow("extra-noncanonical-task"))
			},
		},
		{
			name: "canonical ids in wrong shard sizes",
			mutate: func(f *terminalInspectFixture) {
				moved := f.shards[0][len(f.shards[0])-1]
				f.shards[0] = f.shards[0][:len(f.shards[0])-1]
				f.shards[1] = append(f.shards[1], moved)
			},
		},
		{
			name: "wrong declared shard count",
			mutate: func(f *terminalInspectFixture) {
				f.datasetShardCount = 9
			},
		},
		{
			name: "wrong response shard index",
			mutate: func(f *terminalInspectFixture) {
				f.reportedShardIndexes[3] = 4
			},
		},
		{
			name: "wrong response item count",
			mutate: func(f *terminalInspectFixture) {
				f.reportedItemCounts[4] = len(f.shards[3]) + 1
			},
		},
		{
			name: "malformed sha256",
			mutate: func(f *terminalInspectFixture) {
				f.shards[2][0]["sha256"] = "not-a-sha256"
			},
		},
		{
			name: "zero byte size",
			mutate: func(f *terminalInspectFixture) {
				f.shards[2][0]["byteSize"] = 0
			},
		},
		{
			name: "missing bundle key",
			mutate: func(f *terminalInspectFixture) {
				f.shards[2][0]["bundle_key"] = ""
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			fixture := newTerminalInspectFixture(t, false)
			tc.mutate(fixture)
			server := fixture.start(t)
			defer server.Close()
			err := inspectTerminalDataset(terminalBench21Dataset, parseArgs([]string{
				"eval", "terminal", "inspect", terminalBench21Dataset,
				"--api-url", server.URL,
				"--json",
			}))
			var cliErr cliError
			if !errors.As(err, &cliErr) || cliErr.Code != "terminal_inspect_failed" {
				t.Fatalf("error = %#v, want terminal_inspect_failed", err)
			}
			if fixture.bundleRequests != 0 {
				t.Fatalf("manifest rejection downloaded %d bundles", fixture.bundleRequests)
			}
		})
	}
}

func TestInspectTerminalDatasetRejectsCountPreservingCrossShardSwap(t *testing.T) {
	fixture := newTerminalInspectFixture(t, false)
	shardTwoID := stringValue(fixture.shards[1][0]["question_id"])
	fixture.shards[1][0], fixture.shards[2][0] = fixture.shards[2][0], fixture.shards[1][0]
	server := fixture.start(t)
	defer server.Close()

	err := inspectTerminalDataset(terminalBench21Dataset, parseArgs([]string{
		"eval", "terminal", "inspect", terminalBench21Dataset,
		"--api-url", server.URL,
		"--json",
	}))
	var cliErr cliError
	if !errors.As(err, &cliErr) || cliErr.Code != "terminal_inspect_failed" || cliErr.Message != "Terminal-Bench 2.1 task is assigned to a noncanonical shard." {
		t.Fatalf("error = %#v, want noncanonical shard assignment rejection", err)
	}
	details := asObject(cliErr.Details)
	if details["taskId"] != shardTwoID || details["expectedShardIndex"] != 2 || details["actualShardIndex"] != 3 {
		t.Fatalf("assignment evidence = %#v, want task %q moved from shard 2 to shard 3", details, shardTwoID)
	}
	if fixture.bundleRequests != 0 {
		t.Fatalf("shard assignment rejection downloaded %d bundles", fixture.bundleRequests)
	}
}

func TestInspectTerminalDatasetVerifyBundlesChecksHashAndExtraction(t *testing.T) {
	t.Run("success", func(t *testing.T) {
		fixture := newTerminalInspectFixture(t, true)
		server := fixture.start(t)
		defer server.Close()
		var modelCalls atomic.Int32
		modelServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			modelCalls.Add(1)
			http.Error(w, "bundle inspection must not contact a model endpoint", http.StatusInternalServerError)
		}))
		defer modelServer.Close()
		dockerSentinel := filepath.Join(t.TempDir(), "docker-called")
		binDir := t.TempDir()
		dockerPath := filepath.Join(binDir, "docker")
		mustWrite(t, dockerPath, "#!/bin/sh\necho called > "+dockerSentinel+"\n")
		if err := os.Chmod(dockerPath, 0o755); err != nil {
			t.Fatalf("chmod fake docker: %v", err)
		}
		t.Setenv("PATH", binDir+string(os.PathListSeparator)+os.Getenv("PATH"))
		output, err := captureTerminalTestStdout(t, func() error {
			return inspectTerminalDataset(terminalBench21Dataset, parseArgs([]string{
				"eval", "terminal", "inspect", terminalBench21Dataset,
				"--api-url", server.URL,
				"--base-url", modelServer.URL,
				"--verify-bundles", "--json",
			}))
		})
		if err != nil {
			t.Fatalf("verify canonical bundles: %v", err)
		}
		var summary map[string]any
		if err := json.Unmarshal([]byte(output), &summary); err != nil {
			t.Fatalf("decode verify summary %q: %v", output, err)
		}
		if summary["verifiedBundles"] != float64(89) {
			t.Fatalf("verifiedBundles = %#v, want 89", summary["verifiedBundles"])
		}
		if fixture.bundleRequests != 89 {
			t.Fatalf("bundle downloads = %d, want 89", fixture.bundleRequests)
		}
		if fixture.unexpectedRequests != 0 {
			t.Fatalf("bundle inspection made %d unexpected API/submit requests", fixture.unexpectedRequests)
		}
		if got := modelCalls.Load(); got != 0 {
			t.Fatalf("bundle inspection made %d model requests", got)
		}
		if _, err := os.Stat(dockerSentinel); !os.IsNotExist(err) {
			t.Fatalf("bundle inspection executed Docker: %v", err)
		}
	})

	t.Run("hash mismatch", func(t *testing.T) {
		fixture := newTerminalInspectFixture(t, true)
		fixture.shards[0][0]["sha256"] = strings.Repeat("0", 64)
		server := fixture.start(t)
		defer server.Close()
		err := inspectTerminalDataset(terminalBench21Dataset, parseArgs([]string{
			"eval", "terminal", "inspect", terminalBench21Dataset,
			"--api-url", server.URL,
			"--verify-bundles", "--json",
		}))
		var cliErr cliError
		if !errors.As(err, &cliErr) || cliErr.Code != "bundle_download_failed" {
			t.Fatalf("error = %#v, want bundle_download_failed", err)
		}
	})

	t.Run("byte size mismatch", func(t *testing.T) {
		fixture := newTerminalInspectFixture(t, true)
		fixture.shards[0][0]["byteSize"] = fixture.shards[0][0]["byteSize"].(int) + 1
		server := fixture.start(t)
		defer server.Close()
		err := inspectTerminalDataset(terminalBench21Dataset, parseArgs([]string{
			"eval", "terminal", "inspect", terminalBench21Dataset,
			"--api-url", server.URL,
			"--verify-bundles", "--json",
		}))
		var cliErr cliError
		if !errors.As(err, &cliErr) || cliErr.Code != "bundle_download_failed" {
			t.Fatalf("error = %#v, want bundle_download_failed", err)
		}
	})

	t.Run("traversal member", func(t *testing.T) {
		fixture := newTerminalInspectFixture(t, true)
		id := stringValue(fixture.shards[0][0]["question_id"])
		key := stringValue(fixture.shards[0][0]["bundle_key"])
		archive := terminalInspectBundleArchive(t, id, map[string]string{"../escaped": "unsafe"})
		fixture.setBundle(fixture.shards[0][0], key, archive)
		server := fixture.start(t)
		defer server.Close()
		err := inspectTerminalDataset(terminalBench21Dataset, parseArgs([]string{
			"eval", "terminal", "inspect", terminalBench21Dataset,
			"--api-url", server.URL,
			"--verify-bundles", "--json",
		}))
		var cliErr cliError
		if !errors.As(err, &cliErr) || cliErr.Code != "bundle_download_failed" {
			t.Fatalf("error = %#v, want bundle_download_failed", err)
		}
	})
}

func TestInspectTerminalDatasetRejectsBundleContainingSolution(t *testing.T) {
	fixture := newTerminalInspectFixture(t, true)
	id := stringValue(fixture.shards[0][0]["question_id"])
	key := stringValue(fixture.shards[0][0]["bundle_key"])
	archive := terminalInspectBundleArchive(t, id, map[string]string{id + "/solution/solve.sh": "#!/bin/sh\nexit 0\n"})
	fixture.setBundle(fixture.shards[0][0], key, archive)
	server := fixture.start(t)
	defer server.Close()
	err := inspectTerminalDataset(terminalBench21Dataset, parseArgs([]string{
		"eval", "terminal", "inspect", terminalBench21Dataset,
		"--api-url", server.URL,
		"--verify-bundles", "--json",
	}))
	var cliErr cliError
	if !errors.As(err, &cliErr) || cliErr.Code != "terminal_inspect_failed" {
		t.Fatalf("error = %#v, want terminal_inspect_failed for solution archive", err)
	}
}

func TestSubmitTerminalEvalCanonicalCheckpointDryRunPartitionsExactTaskSet(t *testing.T) {
	canonicalIDs := terminalBench21CanonicalTestTaskIDs(t)
	runDir, hardwarePath := writeTerminalCheckpointSetFixture(t, canonicalIDs, true)
	payloadPath := filepath.Join(t.TempDir(), "canonical-batch.json")

	var networkCalls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		networkCalls.Add(1)
		http.Error(w, "canonical dry-run must not submit", http.StatusInternalServerError)
	}))
	defer server.Close()

	err := submitTerminalEval(parseArgs([]string{
		"eval", "terminal", "submit", runDir,
		"--dataset", terminalBench21Dataset,
		"--hf-id", "fixture/model",
		"--hardware", hardwarePath,
		"--api-url", server.URL,
		"--out", payloadPath,
		"--dry-run", "--quiet",
	}))
	if err != nil {
		t.Fatalf("submit canonical dry-run: %v", err)
	}
	if got := networkCalls.Load(); got != 0 {
		t.Fatalf("canonical dry-run made %d network requests, want none", got)
	}

	batch := readTerminalSubmitBatch(t, payloadPath)
	if batch["dataset"] != terminalBench21Dataset {
		t.Fatalf("batch dataset = %#v, want %q", batch["dataset"], terminalBench21Dataset)
	}
	shards := anySlice(batch["shards"])
	if len(shards) != terminalBench21ShardCount {
		t.Fatalf("batch contains %d shards, want %d", len(shards), terminalBench21ShardCount)
	}
	wantSizes := []int{8, 9, 9, 9, 9, 9, 9, 9, 9, 9}
	flattened := make([]string, 0, len(canonicalIDs))
	offset := 0
	for i, rawShard := range shards {
		payload := asObject(rawShard)
		wantIDs := canonicalIDs[offset : offset+wantSizes[i]]
		assertTerminalShardPayload(t, payload, i+1, wantIDs)
		runConfig := asObject(payload["runConfig"])
		if runConfig["tasksRun"] != float64(wantSizes[i]) {
			t.Fatalf("shard %d tasksRun = %#v, want %d", i+1, runConfig["tasksRun"], wantSizes[i])
		}
		if runConfig["fullCheckpoint"] != true || runConfig["fullCheckpointTasksRun"] != float64(89) {
			t.Fatalf("shard %d full-checkpoint provenance = %#v, want full 89-task source", i+1, runConfig)
		}
		for _, result := range anySlice(payload["results"]) {
			flattened = append(flattened, stringValue(asObject(result)["question_id"]))
		}
		offset += wantSizes[i]
	}
	if offset != 89 || len(flattened) != 89 {
		t.Fatalf("partitioned task total = offset %d, flattened %d; want exactly 89", offset, len(flattened))
	}
	if !reflect.DeepEqual(flattened, canonicalIDs) {
		t.Fatalf("flattened shard task IDs do not equal the exact canonical sorted set\ngot:  %q\nwant: %q", flattened, canonicalIDs)
	}
}

func TestSubmitTerminalEvalRejectsNonCanonicalTerminalBench21TaskSets(t *testing.T) {
	canonicalIDs := terminalBench21CanonicalTestTaskIDs(t)
	tests := []struct {
		name        string
		taskIDs     []string
		wantMissing []string
		wantExtra   []string
	}{
		{
			name:        "missing canonical task",
			taskIDs:     append([]string(nil), canonicalIDs[:len(canonicalIDs)-1]...),
			wantMissing: []string{canonicalIDs[len(canonicalIDs)-1]},
			wantExtra:   []string{},
		},
		{
			name: "substituted task id",
			taskIDs: func() []string {
				ids := append([]string(nil), canonicalIDs...)
				ids[len(ids)/2] = "substituted-noncanonical-task"
				return ids
			}(),
			wantMissing: []string{canonicalIDs[len(canonicalIDs)/2]},
			wantExtra:   []string{"substituted-noncanonical-task"},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			runDir, hardwarePath := writeTerminalCheckpointSetFixture(t, tc.taskIDs, true)
			err := submitTerminalEval(parseArgs([]string{
				"eval", "terminal", "submit", runDir,
				"--dataset", terminalBench21Dataset,
				"--hf-id", "fixture/model",
				"--hardware", hardwarePath,
				"--dry-run", "--quiet",
			}))
			var cliErr cliError
			if !errors.As(err, &cliErr) || cliErr.Code != "checkpoint_task_set_mismatch" {
				t.Fatalf("error = %#v, want checkpoint_task_set_mismatch", err)
			}
			details := asObject(cliErr.Details)
			if !reflect.DeepEqual(details["missingTaskIds"], tc.wantMissing) {
				t.Fatalf("missing task IDs = %#v, want %#v", details["missingTaskIds"], tc.wantMissing)
			}
			if !reflect.DeepEqual(details["extraTaskIds"], tc.wantExtra) {
				t.Fatalf("extra task IDs = %#v, want %#v", details["extraTaskIds"], tc.wantExtra)
			}
		})
	}
}

func TestSubmitTerminalEvalExplicitCanonicalShardWritesIsolatedCheckpointPayload(t *testing.T) {
	canonicalIDs := terminalBench21CanonicalTestTaskIDs(t)
	firstShardEnd := len(canonicalIDs) / terminalBench21ShardCount
	if firstShardEnd != 8 {
		t.Fatalf("canonical first shard size = %d, want 8", firstShardEnd)
	}
	wantIDs := canonicalIDs[:firstShardEnd]
	runDir, hardwarePath := writeTerminalCheckpointSetFixture(t, wantIDs, true)
	payloadPath := filepath.Join(t.TempDir(), "isolated-shard.json")
	err := submitTerminalEval(parseArgs([]string{
		"eval", "terminal", "submit", runDir,
		"--dataset", terminalBench21Dataset,
		"--shard-index", "1",
		"--hf-id", "fixture/model",
		"--hardware", hardwarePath,
		"--out", payloadPath,
		"--dry-run", "--quiet",
	}))
	if err != nil {
		t.Fatalf("submit explicit canonical shard: %v", err)
	}

	batch := readTerminalSubmitBatch(t, payloadPath)
	shards := anySlice(batch["shards"])
	if len(shards) != 1 {
		t.Fatalf("explicit checkpoint payload count = %d, want one isolated shard", len(shards))
	}
	payload := asObject(shards[0])
	assertTerminalShardPayload(t, payload, 1, wantIDs)
	if got := asObject(payload["runConfig"])["fullCheckpoint"]; got != false {
		t.Fatalf("explicit shard fullCheckpoint = %#v, want false", got)
	}
}

func TestSubmitTerminalEvalCustomDatasetRequiresShardIndex(t *testing.T) {
	runDir := t.TempDir()
	err := submitTerminalEval(parseArgs([]string{
		"eval", "terminal", "submit", runDir,
		"--dataset", "custom-terminal-dataset",
		"--dry-run", "--quiet",
	}))
	var cliErr cliError
	if !errors.As(err, &cliErr) || cliErr.Code != "missing_shard_index" {
		t.Fatalf("error = %#v, want missing_shard_index", err)
	}
	if got := asObject(cliErr.Details)["dataset"]; got != "custom-terminal-dataset" {
		t.Fatalf("missing_shard_index dataset evidence = %#v, want custom-terminal-dataset", got)
	}
}

func TestSubmitTerminalEvalPostsAllCanonicalShards(t *testing.T) {
	canonicalIDs := terminalBench21CanonicalTestTaskIDs(t)
	runDir, hardwarePath := writeTerminalCheckpointSetFixture(t, canonicalIDs, true)
	wantSizes := []int{8, 9, 9, 9, 9, 9, 9, 9, 9, 9}
	seenIDs := make(map[string]int, len(canonicalIDs))
	requestCount := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requestCount++
		if r.Method != http.MethodPost {
			t.Errorf("request %d method = %s, want POST", requestCount, r.Method)
		}
		if r.URL.Path != "/api/evals/terminal-bench-2-1/submit" {
			t.Errorf("request %d path = %q, want canonical submit path", requestCount, r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer fixture-key" {
			t.Errorf("request %d authorization = %q, want bearer fixture key", requestCount, got)
		}
		if got := r.Header.Get("Content-Type"); got != "application/json" {
			t.Errorf("request %d content type = %q, want application/json", requestCount, got)
		}
		var payload map[string]any
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Errorf("decode shard request %d: %v", requestCount, err)
			http.Error(w, "invalid request", http.StatusBadRequest)
			return
		}
		if requestCount > terminalBench21ShardCount {
			t.Errorf("received unexpected request %d", requestCount)
			http.Error(w, "too many requests", http.StatusBadRequest)
			return
		}
		offset := 0
		for i := range requestCount - 1 {
			offset += wantSizes[i]
		}
		wantIDs := canonicalIDs[offset : offset+wantSizes[requestCount-1]]
		assertTerminalShardPayload(t, payload, requestCount, wantIDs)
		for _, id := range wantIDs {
			seenIDs[id]++
		}
		covered := make([]int, requestCount)
		for i := range covered {
			covered[i] = i + 1
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"run":       map[string]any{"id": fmt.Sprintf("run-%02d", requestCount), "status": "approved"},
			"aggregate": map[string]any{"pooledScore": float64(requestCount) / 100, "shardsCovered": covered},
		})
	}))
	defer server.Close()

	output, err := captureTerminalTestStdout(t, func() error {
		return submitTerminalEval(parseArgs([]string{
			"eval", "terminal", "submit", runDir,
			"--dataset", terminalBench21Dataset,
			"--hf-id", "fixture/model",
			"--hardware", hardwarePath,
			"--api-url", server.URL,
			"--api-key", "fixture-key",
			"--quiet",
		}))
	})
	if err != nil {
		t.Fatalf("submit canonical checkpoint: %v", err)
	}
	if requestCount != terminalBench21ShardCount {
		t.Fatalf("submission request count = %d, want %d", requestCount, terminalBench21ShardCount)
	}
	if len(seenIDs) != len(canonicalIDs) {
		t.Fatalf("submitted unique task IDs = %d, want %d", len(seenIDs), len(canonicalIDs))
	}
	for _, id := range canonicalIDs {
		if seenIDs[id] != 1 {
			t.Fatalf("task %q submitted %d times, want exactly once", id, seenIDs[id])
		}
	}
	for shard := range terminalBench21ShardCount {
		if want := fmt.Sprintf("run-%02d", shard+1); !strings.Contains(output, want) {
			t.Fatalf("completion output did not contain parsed receipt %q: %s", want, output)
		}
	}
	for _, want := range []string{
		"status:approved",
		"pooledScore:0.1",
		"coverage:[1 2 3 4 5 6 7 8 9 10]",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("completion output did not contain parsed receipt field %q: %s", want, output)
		}
	}
}

func TestSubmitTerminalEvalPostsCanonicalShardsSequentiallyAndStopsAtFailure(t *testing.T) {
	canonicalIDs := terminalBench21CanonicalTestTaskIDs(t)
	runDir, hardwarePath := writeTerminalCheckpointSetFixture(t, canonicalIDs, true)
	var requestCount atomic.Int32
	var received [terminalBench21ShardCount]atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		call := int(requestCount.Add(1)) - 1
		if r.Method != http.MethodPost {
			t.Errorf("request %d method = %s, want POST", call+1, r.Method)
		}
		if r.URL.Path != "/api/evals/terminal-bench-2-1/submit" {
			t.Errorf("request %d path = %q, want canonical submit path", call+1, r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer fixture-key" {
			t.Errorf("request %d authorization = %q, want bearer fixture key", call+1, got)
		}
		if got := r.Header.Get("Content-Type"); got != "application/json" {
			t.Errorf("request %d content type = %q, want application/json", call+1, got)
		}
		var payload struct {
			ShardIndex int `json:"shardIndex"`
		}
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Errorf("decode shard request %d: %v", call+1, err)
			http.Error(w, "invalid request", http.StatusBadRequest)
			return
		}
		if call < len(received) {
			received[call].Store(int32(payload.ShardIndex))
		}
		if payload.ShardIndex == 3 {
			http.Error(w, "synthetic shard failure", http.StatusServiceUnavailable)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"run":{"id":"accepted","status":"approved"}}`))
	}))
	defer server.Close()

	err := submitTerminalEval(parseArgs([]string{
		"eval", "terminal", "submit", runDir,
		"--dataset", terminalBench21Dataset,
		"--hf-id", "fixture/model",
		"--hardware", hardwarePath,
		"--api-url", server.URL,
		"--api-key", "fixture-key",
		"--quiet",
	}))
	var cliErr cliError
	if !errors.As(err, &cliErr) || cliErr.Code != "terminal_submit_shard_failed" {
		t.Fatalf("error = %#v, want terminal_submit_shard_failed", err)
	}
	if got := requestCount.Load(); got != 3 {
		t.Fatalf("submission request count = %d, want exactly 3 (stop after shard 3)", got)
	}
	for i, want := range []int32{1, 2, 3} {
		if got := received[i].Load(); got != want {
			t.Fatalf("submission request %d carried shardIndex %d, want %d", i+1, got, want)
		}
	}
	details := asObject(cliErr.Details)
	if details["failedShardIndex"] != 3 {
		t.Fatalf("failed shard evidence = %#v, want 3", details["failedShardIndex"])
	}
	if !reflect.DeepEqual(details["completedShardIndexes"], []int{1, 2}) {
		t.Fatalf("completed shard evidence = %#v, want [1 2]", details["completedShardIndexes"])
	}
}

type terminalInspectRequestCounter struct {
	calls atomic.Int32
}

func (c *terminalInspectRequestCounter) RoundTrip(*http.Request) (*http.Response, error) {
	c.calls.Add(1)
	return nil, errors.New("unexpected HTTP request")
}

type terminalInspectFixture struct {
	datasetItemCount     int
	datasetShardCount    int
	shards               [][]map[string]any
	bundles              map[string][]byte
	reportedShardIndexes map[int]int
	reportedItemCounts   map[int]int
	wantAuthorization    string
	shardRequests        []int
	manifestRequests     []int
	bundleRequests       int
	unexpectedRequests   int
}

func newTerminalInspectFixture(t *testing.T, includeBundles bool) *terminalInspectFixture {
	t.Helper()
	ids := terminalBench21CanonicalTestTaskIDs(t)
	fixture := &terminalInspectFixture{
		datasetItemCount:     len(ids),
		datasetShardCount:    terminalBench21ShardCount,
		shards:               make([][]map[string]any, terminalBench21ShardCount),
		bundles:              make(map[string][]byte, len(ids)),
		reportedShardIndexes: map[int]int{},
		reportedItemCounts:   map[int]int{},
	}
	for shard := range terminalBench21ShardCount {
		start := shard * len(ids) / terminalBench21ShardCount
		end := (shard + 1) * len(ids) / terminalBench21ShardCount
		fixture.shards[shard] = make([]map[string]any, 0, end-start)
		for _, id := range ids[start:end] {
			row := terminalInspectManifestRow(id)
			if includeBundles {
				key := stringValue(row["bundle_key"])
				archive := terminalInspectBundleArchive(t, id, nil)
				fixture.setBundle(row, key, archive)
			}
			fixture.shards[shard] = append(fixture.shards[shard], row)
		}
	}
	return fixture
}

func terminalInspectManifestRow(id string) map[string]any {
	return map[string]any{
		"question_id": id,
		"bundle_key":  "eval-datasets/terminal-bench-2-1/tasks/" + id + ".tar.gz",
		"sha256":      strings.Repeat("a", 64),
		"byteSize":    1,
	}
}

func terminalInspectBundleArchive(t *testing.T, id string, extraFiles map[string]string) []byte {
	t.Helper()
	taskJSON, err := json.Marshal(map[string]any{
		"id": id, "version": "2.1", "instruction": "Inspect fixture " + id,
		"source": "terminal-bench/" + id,
	})
	if err != nil {
		t.Fatalf("marshal task fixture: %v", err)
	}
	files := map[string]string{
		id + "/task.json":              string(taskJSON),
		id + "/environment/Dockerfile": "FROM scratch\n",
		id + "/tests/test.sh":          "#!/bin/sh\nexit 0\n",
	}
	for name, body := range extraFiles {
		files[name] = body
	}
	return testReleaseTarGz(t, files)
}

func (f *terminalInspectFixture) setBundle(row map[string]any, key string, archive []byte) {
	f.bundles[key] = archive
	sum := sha256.Sum256(archive)
	row["sha256"] = fmt.Sprintf("%x", sum)
	row["byteSize"] = len(archive)
}

func (f *terminalInspectFixture) start(t *testing.T) *httptest.Server {
	t.Helper()
	var server *httptest.Server
	server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		canonicalShardPath := "/api/evals/terminal-bench-2-1/shard"
		switch {
		case r.URL.Path == canonicalShardPath:
			if r.Method != http.MethodGet {
				t.Errorf("shard method = %s, want GET", r.Method)
			}
			f.assertAuthorization(t, r)
			shardIndex, err := strconv.Atoi(r.URL.Query().Get("shard"))
			if err != nil || shardIndex < 1 || shardIndex > len(f.shards) {
				http.Error(w, "invalid shard", http.StatusNotFound)
				return
			}
			f.shardRequests = append(f.shardRequests, shardIndex)
			reportedIndex := shardIndex
			if override := f.reportedShardIndexes[shardIndex]; override != 0 {
				reportedIndex = override
			}
			reportedCount := len(f.shards[shardIndex-1])
			if override, ok := f.reportedItemCounts[shardIndex]; ok {
				reportedCount = override
			}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{
				"dataset": map[string]any{
					"slug": terminalBench21Dataset, "itemCount": f.datasetItemCount, "shardCount": f.datasetShardCount,
				},
				"shard": map[string]any{
					"shardIndex": reportedIndex, "itemCount": reportedCount, "selectedQuestionCount": reportedCount,
				},
				"downloadUrl": server.URL + "/manifest/" + strconv.Itoa(shardIndex),
			})
		case strings.HasPrefix(r.URL.Path, "/manifest/"):
			shardIndex, err := strconv.Atoi(strings.TrimPrefix(r.URL.Path, "/manifest/"))
			if err != nil || shardIndex < 1 || shardIndex > len(f.shards) {
				http.NotFound(w, r)
				return
			}
			f.manifestRequests = append(f.manifestRequests, shardIndex)
			w.Header().Set("Content-Type", "application/x-ndjson")
			encoder := json.NewEncoder(w)
			for _, row := range f.shards[shardIndex-1] {
				if err := encoder.Encode(row); err != nil {
					t.Errorf("encode manifest shard %d: %v", shardIndex, err)
					return
				}
			}
		case r.URL.Path == "/api/evals/storage/download-url":
			if r.Method != http.MethodGet {
				t.Errorf("presign method = %s, want GET", r.Method)
			}
			f.assertAuthorization(t, r)
			key := r.URL.Query().Get("key")
			if _, ok := f.bundles[key]; !ok {
				http.Error(w, "unknown bundle", http.StatusNotFound)
				return
			}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{"downloadUrl": server.URL + "/bundle?key=" + url.QueryEscape(key)})
		case r.URL.Path == "/bundle":
			key := r.URL.Query().Get("key")
			archive, ok := f.bundles[key]
			if !ok {
				http.Error(w, "unknown bundle", http.StatusNotFound)
				return
			}
			f.bundleRequests++
			w.Header().Set("Content-Type", "application/gzip")
			_, _ = w.Write(archive)
		default:
			f.unexpectedRequests++
			http.Error(w, "unexpected inspect request", http.StatusInternalServerError)
		}
	}))
	return server
}

func (f *terminalInspectFixture) assertAuthorization(t *testing.T, r *http.Request) {
	t.Helper()
	if f.wantAuthorization != "" && r.Header.Get("Authorization") != f.wantAuthorization {
		t.Errorf("authorization = %q, want %q", r.Header.Get("Authorization"), f.wantAuthorization)
	}
}

func captureTerminalTestStdout(t *testing.T, fn func() error) (string, error) {
	t.Helper()
	reader, writer, err := os.Pipe()
	if err != nil {
		t.Fatalf("create stdout pipe: %v", err)
	}
	original := os.Stdout
	os.Stdout = writer
	defer func() { os.Stdout = original }()
	callErr := fn()
	if err := writer.Close(); err != nil {
		t.Fatalf("close stdout writer: %v", err)
	}
	output, readErr := io.ReadAll(reader)
	if err := reader.Close(); err != nil {
		t.Fatalf("close stdout reader: %v", err)
	}
	if readErr != nil {
		t.Fatalf("read captured stdout: %v", readErr)
	}
	return string(output), callErr
}

func terminalBench21CanonicalTestTaskIDs(t *testing.T) []string {
	t.Helper()
	ids := strings.Fields(terminalBench21CanonicalTestTaskIDsText)
	if len(ids) != 89 {
		t.Fatalf("canonical test fixture contains %d task IDs, want exactly 89", len(ids))
	}
	return ids
}

func writeTerminalCheckpointSetFixture(t *testing.T, taskIDs []string, reverseSummary bool) (runDir, hardwarePath string) {
	t.Helper()
	runDir = t.TempDir()
	hardwarePath = filepath.Join(t.TempDir(), "hardware.json")
	writeTerminalTestJSON(t, hardwarePath, map[string]any{"hwClass": "CPU_ONLY", "cpu": "Fixture CPU", "ramGb": 32})
	summary := make([]any, 0, len(taskIDs))
	for offset := range taskIDs {
		i := offset
		if reverseSummary {
			i = len(taskIDs) - 1 - offset
		}
		id := taskIDs[i]
		passed := i%3 != 0
		summary = append(summary, map[string]any{
			"index": i + 1, "total": len(taskIDs), "task": id, "out": id + ".json", "pass": passed, "scored": true,
		})
		writeTerminalTestJSON(t, filepath.Join(runDir, id+".json"), map[string]any{"results": []any{map[string]any{
			"question_id": id,
			"pass":        passed,
			"scored":      true,
			"wallTimeMs":  i + 1,
			"question":    "Question for " + id,
			"prompt":      "Prompt for " + id,
			"response":    "Response for " + id,
			"tokenUsage":  map[string]any{"inputTokens": 1, "outputTokens": 1, "totalTokens": 2, "modelCalls": 1},
		}}})
	}
	writeTerminalTestJSON(t, filepath.Join(runDir, "summary.json"), summary)
	return runDir, hardwarePath
}

func readTerminalSubmitBatch(t *testing.T, path string) map[string]any {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read terminal submit batch: %v", err)
	}
	var batch map[string]any
	if err := json.Unmarshal(data, &batch); err != nil {
		t.Fatalf("decode terminal submit batch: %v", err)
	}
	return batch
}

func assertTerminalShardPayload(t *testing.T, payload map[string]any, wantShardIndex int, wantIDs []string) {
	t.Helper()
	if payload["shardIndex"] != float64(wantShardIndex) {
		t.Fatalf("shardIndex = %#v, want %d", payload["shardIndex"], wantShardIndex)
	}
	results := anySlice(payload["results"])
	artifacts := anySlice(payload["artifacts"])
	if len(results) != len(wantIDs) || len(artifacts) != len(wantIDs) {
		t.Fatalf("shard %d sizes = results %d, artifacts %d; want %d each", wantShardIndex, len(results), len(artifacts), len(wantIDs))
	}
	for i, wantID := range wantIDs {
		result := asObject(results[i])
		artifact := asObject(artifacts[i])
		if got := stringValue(result["question_id"]); got != wantID {
			t.Fatalf("shard %d result %d question_id = %q, want %q", wantShardIndex, i, got, wantID)
		}
		if got := stringValue(artifact["question_id"]); got != wantID {
			t.Fatalf("shard %d artifact %d question_id = %q, want %q", wantShardIndex, i, got, wantID)
		}
		if artifact["itemIndex"] != float64(i) {
			t.Fatalf("shard %d artifact %q itemIndex = %#v, want local index %d", wantShardIndex, wantID, artifact["itemIndex"], i)
		}
	}
}

type terminalTestRoundTripFunc func(*http.Request) (*http.Response, error)

func (fn terminalTestRoundTripFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return fn(request)
}

func terminalTestHTTPResponse(request *http.Request, status int, body string) *http.Response {
	return &http.Response{
		StatusCode: status,
		Status:     fmt.Sprintf("%d %s", status, http.StatusText(status)),
		Header:     make(http.Header),
		Body:       io.NopCloser(strings.NewReader(body)),
		Request:    request,
	}
}

func TestTerminalUXEndpointFileDecodesSelectedEndpointMetadata(t *testing.T) {
	path := filepath.Join(t.TempDir(), "endpoint.json")
	writeTerminalTestJSON(t, path, map[string]any{"endpoints": []any{
		map[string]any{"ok": false, "baseUrl": "http://ignored.invalid", "servedModel": "ignored"},
		map[string]any{
			"ok":          true,
			"baseUrl":     "http://selected.test/v1",
			"servedModel": "selected-alias",
			"serverMetadata": map[string]any{
				"quantization": "Q5_K_M",
				"model_path":   "/models/selected-model-Q5_K_M.gguf",
			},
		},
	}})

	metadata, err := loadTerminalEndpointFile(path)
	if err != nil {
		t.Fatalf("loadTerminalEndpointFile: %v", err)
	}
	if metadata.baseURL != "http://selected.test/v1" || metadata.servedModel != "selected-alias" {
		t.Fatalf("selected endpoint identity = %#v", metadata)
	}
	if metadata.quantization != "Q5_K_M" || metadata.modelPath != "/models/selected-model-Q5_K_M.gguf" {
		t.Fatalf("selected endpoint model metadata = %#v", metadata)
	}
}

func TestTerminalUXEndpointFileRejectsZeroOrMultipleHealthyEndpoints(t *testing.T) {
	tests := []struct {
		name     string
		items    []any
		wantCode string
	}{
		{
			name: "zero healthy",
			items: []any{
				map[string]any{"ok": false, "baseUrl": "http://one.test"},
				map[string]any{"ok": false, "baseUrl": "http://two.test"},
			},
			wantCode: "endpoint_file_no_selection",
		},
		{
			name: "multiple healthy",
			items: []any{
				map[string]any{"ok": true, "baseUrl": "http://one.test"},
				map[string]any{"ok": true, "baseUrl": "http://two.test"},
			},
			wantCode: "endpoint_file_ambiguous",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "endpoint.json")
			writeTerminalTestJSON(t, path, map[string]any{"endpoints": tc.items})
			_, err := loadTerminalEndpointFile(path)
			var cliErr cliError
			if !errors.As(err, &cliErr) || cliErr.Code != tc.wantCode {
				t.Fatalf("error = %#v, want cliError code %q", err, tc.wantCode)
			}
		})
	}
}

func TestTerminalUXAutoDiscoverySelectsUniqueHealthyCandidateWithoutNetwork(t *testing.T) {
	originalClient := apiHTTPClient
	t.Cleanup(func() { apiHTTPClient = originalClient })
	requests := []string{}
	apiHTTPClient = &http.Client{Transport: terminalTestRoundTripFunc(func(request *http.Request) (*http.Response, error) {
		requests = append(requests, request.URL.String())
		if request.URL.Host == "localhost:8000" {
			return terminalTestHTTPResponse(request, http.StatusOK, `{"data":[{"id":"fixture-model","quantization":"Q6_K"}]}`), nil
		}
		return nil, errors.New("fixture endpoint unavailable")
	})}

	baseURL, servedModel, info, err := discoverTerminalEndpoint(parseArgs([]string{"--quiet"}), "")
	if err != nil {
		t.Fatalf("discoverTerminalEndpoint: %v", err)
	}
	if baseURL != "http://localhost:8000" || servedModel != "fixture-model" || stringValue(info["quantization"]) != "Q6_K" {
		t.Fatalf("discovery result = (%q, %q, %#v)", baseURL, servedModel, info)
	}
	wantRequests := []string{
		"http://localhost:8080/v1/models",
		"http://localhost:8000/v1/models",
		"http://localhost:11434/v1/models",
		"http://127.0.0.1:30000/v1/models",
	}
	if !reflect.DeepEqual(requests, wantRequests) {
		t.Fatalf("discovery requests = %#v, want %#v", requests, wantRequests)
	}
}

func TestTerminalUXAutoDiscoveryRejectsCredentialsWithoutExplicitTarget(t *testing.T) {
	originalClient := apiHTTPClient
	t.Cleanup(func() { apiHTTPClient = originalClient })
	var networkCalls atomic.Int32
	apiHTTPClient = &http.Client{Transport: terminalTestRoundTripFunc(func(request *http.Request) (*http.Response, error) {
		networkCalls.Add(1)
		return nil, fmt.Errorf("credentials must not be broadcast to %s", request.URL)
	})}

	_, _, _, err := discoverTerminalEndpoint(parseArgs([]string{"--model-api-key", "secret"}), "fixture-model")
	var cliErr cliError
	if !errors.As(err, &cliErr) || cliErr.Code != "endpoint_credentials_require_explicit_target" {
		t.Fatalf("error = %#v, want endpoint_credentials_require_explicit_target", err)
	}
	if networkCalls.Load() != 0 {
		t.Fatalf("credentialed auto-discovery made %d requests", networkCalls.Load())
	}
}

func TestTerminalUXEndpointSelectionRejectsFileAndBaseTogether(t *testing.T) {
	err := runTerminalEval(parseArgs([]string{
		"eval", "terminal", "run", "--task-dir", t.TempDir(),
		"--endpoint-file", "endpoint.json", "--base-url", "http://model.test",
	}), false)
	var cliErr cliError
	if !errors.As(err, &cliErr) || cliErr.Code != "endpoint_selection_conflict" {
		t.Fatalf("error = %#v, want endpoint_selection_conflict", err)
	}
}

func TestTerminalUXEndpointMetadataRejectsExplicitConflicts(t *testing.T) {
	for _, tc := range []struct {
		name            string
		field           string
		explicit, live  string
		compareFilename bool
	}{
		{name: "model path", field: "model-path", explicit: "/models/explicit.gguf", live: "/models/live.gguf", compareFilename: true},
		{name: "quantization", field: "quantization", explicit: "Q8_0", live: "Q4_K_M"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := reconcileTerminalEndpointField(tc.field, tc.explicit, tc.live, tc.compareFilename)
			var cliErr cliError
			if !errors.As(err, &cliErr) || cliErr.Code != "endpoint_metadata_conflict" {
				t.Fatalf("error = %#v, want endpoint_metadata_conflict", err)
			}
		})
	}
}

func TestTerminalUXEndpointMetadataAcceptsMatchingExplicitFallbacks(t *testing.T) {
	modelPath, err := reconcileTerminalEndpointField("model-path", "/client/path/model-Q5_K_M.gguf", "/server/path/model-Q5_K_M.gguf", true)
	if err != nil || modelPath != "/server/path/model-Q5_K_M.gguf" {
		t.Fatalf("matching model filename reconciliation = (%q, %v)", modelPath, err)
	}
	quantization, err := reconcileTerminalEndpointField("quantization", "q5_k_m", "Q5_K_M", false)
	if err != nil || quantization != "Q5_K_M" {
		t.Fatalf("matching quantization reconciliation = (%q, %v)", quantization, err)
	}
}

func TestTerminalUXEndpointFileReconcilesLiveMetadataAndCompletionIsNonSubmit(t *testing.T) {
	bundleDir := t.TempDir()
	mustMkdir(t, filepath.Join(bundleDir, "tests"))
	writeTerminalTestJSON(t, filepath.Join(bundleDir, "task.json"), map[string]any{
		"id": "endpoint-reconciliation", "version": "2.1", "instruction": "Exercise endpoint reconciliation.", "source": "terminal-bench/endpoint-reconciliation",
		"image":    map[string]any{"prebuilt": "fixture-image"},
		"agent":    map[string]any{"timeoutSec": 30, "maxTurns": 1},
		"verifier": map[string]any{"timeoutSec": 30, "command": "bash /tests/test.sh", "rewardFile": "/logs/verifier/reward.txt"},
	})
	endpointFile := filepath.Join(t.TempDir(), "endpoint.json")
	writeTerminalTestJSON(t, endpointFile, map[string]any{"endpoints": []any{map[string]any{
		"ok": true, "baseUrl": "http://saved-endpoint.test", "servedModel": "explicit-alias", "quantization": "Q8_0", "modelPath": "/server/models/explicit-model-Q8_0.gguf",
	}}})
	out := filepath.Join(t.TempDir(), "completed.json")

	binDir := t.TempDir()
	dockerLog := filepath.Join(t.TempDir(), "docker.log")
	dockerScript := "#!/bin/sh\nprintf '%s\\n' \"$*\" >> \"$LMX_DOCKER_LOG\"\ncase \"$*\" in\n  *\"cat /logs/verifier/reward.json\"*) printf '{\"reward\":1}\\n' ;;\nesac\nexit 0\n"
	mustWrite(t, filepath.Join(binDir, "docker"), dockerScript)
	if err := os.Chmod(filepath.Join(binDir, "docker"), 0o755); err != nil {
		t.Fatalf("make fake docker executable: %v", err)
	}
	t.Setenv("LMX_DOCKER_LOG", dockerLog)
	t.Setenv("PATH", binDir+string(os.PathListSeparator)+os.Getenv("PATH"))

	originalAPIClient := apiHTTPClient
	originalDefaultClient := http.DefaultClient
	t.Cleanup(func() {
		apiHTTPClient = originalAPIClient
		http.DefaultClient = originalDefaultClient
	})
	var endpointCalls atomic.Int32
	apiHTTPClient = &http.Client{Transport: terminalTestRoundTripFunc(func(request *http.Request) (*http.Response, error) {
		switch {
		case request.URL.Host == "saved-endpoint.test" && request.URL.Path == "/v1/models":
			endpointCalls.Add(1)
			return terminalTestHTTPResponse(request, http.StatusOK, `{"data":[{"id":"explicit-alias","quantization":"Q8_0","model_path":"/server/models/explicit-model-Q8_0.gguf"}]}`), nil
		case request.URL.Host == "saved-endpoint.test" && request.URL.Path == "/props":
			endpointCalls.Add(1)
			return terminalTestHTTPResponse(request, http.StatusOK, `{"model_path":"/server/models/explicit-model-Q8_0.gguf","quantization":"Q8_0"}`), nil
		case request.URL.Host == "localmaxxing.test" && request.URL.Path == "/api/models/search":
			return terminalTestHTTPResponse(request, http.StatusOK, `{"models":[]}`), nil
		default:
			return nil, fmt.Errorf("unexpected API request %s", request.URL)
		}
	})}
	http.DefaultClient = &http.Client{Transport: terminalTestRoundTripFunc(func(request *http.Request) (*http.Response, error) {
		if request.URL.Host == "saved-endpoint.test" && request.URL.Path == "/props" {
			return terminalTestHTTPResponse(request, http.StatusOK, `{"model_path":"/server/models/explicit-model-Q8_0.gguf","quantization":"Q8_0"}`), nil
		}
		return nil, fmt.Errorf("unexpected default-client request %s", request.URL)
	})}

	stdout, err := captureTerminalTestStdout(t, func() error {
		return runTerminalEval(parseArgs([]string{
			"eval", "terminal", "run", "--task-dir", bundleDir,
			"--endpoint-file", endpointFile,
			"--served-model", "explicit-alias",
			"--model", "explicit/model",
			"--model-path", "/client/models/explicit-model-Q8_0.gguf",
			"--quantization", "Q8_0", "--quant-format", "gguf",
			"--api-url", "http://localmaxxing.test",
			"--agent-cmd", "true", "--agent-name", "fixture-agent",
			"--out", out, "--quiet",
		}), false)
	})
	if err != nil {
		t.Fatalf("runTerminalEval: %v", err)
	}
	if endpointCalls.Load() < 2 {
		t.Fatalf("selected endpoint received %d metadata probes, want live models and props", endpointCalls.Load())
	}
	for _, want := range []string{"[localmaxxing] local_execution_results_not_submitted", "local_execution_results_not_submitted: terminal tasks executed and checkpointed; no result was submitted to LocalMaxxing.", "Submit later with: lmx eval terminal submit"} {
		if !strings.Contains(stdout, want) {
			t.Fatalf("completion output missing %q:\n%s", want, stdout)
		}
	}
	if strings.Contains(stdout, "submitted successfully") {
		t.Fatalf("non-submit completion claimed submission:\n%s", stdout)
	}

	var artifact map[string]any
	data, readErr := os.ReadFile(out)
	if readErr != nil {
		t.Fatalf("read completed artifact: %v", readErr)
	}
	if err := json.Unmarshal(data, &artifact); err != nil {
		t.Fatalf("decode completed artifact: %v", err)
	}
	summary := asObject(artifact["summary"])
	if summary["servedModel"] != "explicit-alias" || summary["quantization"] != "Q8_0" || summary["quantFormat"] != "gguf" {
		t.Fatalf("reconciled endpoint metadata = %#v", summary)
	}
	resolution := asObject(summary["modelResolution"])
	if resolution["loadedFilename"] != "explicit-model-Q8_0.gguf" || resolution["declaredBaseModel"] != "explicit/model" {
		t.Fatalf("reconciled model identity metadata = %#v", resolution)
	}
	runConfig := asObject(summary["runConfig"])
	if runConfig["modelEndpoint"] != "http://saved-endpoint.test" || runConfig["servedModelSource"] != "cli" {
		t.Fatalf("run endpoint identity = %#v", runConfig)
	}
}

func TestTerminalUXLoadedFilenameResolvesCanonicalHuggingFaceRepository(t *testing.T) {
	originalAPIClient := apiHTTPClient
	originalDefaultClient := http.DefaultClient
	t.Cleanup(func() {
		apiHTTPClient = originalAPIClient
		http.DefaultClient = originalDefaultClient
	})
	var searchQuery string
	apiHTTPClient = &http.Client{Transport: terminalTestRoundTripFunc(func(request *http.Request) (*http.Response, error) {
		if request.URL.Path != "/api/models/search" {
			return nil, fmt.Errorf("unexpected LocalMaxxing request %s", request.URL)
		}
		searchQuery = request.URL.Query().Get("q")
		return terminalTestHTTPResponse(request, http.StatusOK, `{"models":[{"hfId":"unsloth/Qwen3-8B-GGUF"}]}`), nil
	})}
	http.DefaultClient = &http.Client{Transport: terminalTestRoundTripFunc(func(request *http.Request) (*http.Response, error) {
		if request.URL.Path != "/api/models/unsloth/Qwen3-8B-GGUF" {
			return nil, fmt.Errorf("unexpected HuggingFace request %s", request.URL)
		}
		return terminalTestHTTPResponse(request, http.StatusOK, `{"siblings":[{"rfilename":"Qwen3-8B-UD-Q4_K_XL.gguf"}]}`), nil
	})}

	args := parseArgs([]string{"--api-url", "http://localmaxxing.test", "--hf-api-url", "http://huggingface.test", "--quiet"})
	resolved, metadata := resolveTerminalModelIdentity(args, "", "/models/Qwen3-8B-UD-Q4_K_XL.gguf", "v1_models", "/models/Qwen3-8B-UD-Q4_K_XL.gguf")
	if resolved != "unsloth/Qwen3-8B-GGUF" {
		t.Fatalf("resolved model = %q, want exact repository containing loaded filename", resolved)
	}
	if searchQuery != "Qwen3-8B" {
		t.Fatalf("model search query = %q, want filename-derived Qwen3-8B", searchQuery)
	}
	if metadata["loadedFilename"] != "Qwen3-8B-UD-Q4_K_XL.gguf" || metadata["sourceRepoMatch"] != "exact_filename" || metadata["status"] != "source_repo_verified" {
		t.Fatalf("model resolution metadata = %#v", metadata)
	}
}

func TestTerminalUXMonolithicArtifactUsesSavedMetadataForOfflineDeferredSubmit(t *testing.T) {
	artifactPath := writeTerminalMonolithicFixture(t)
	payloadPath := filepath.Join(t.TempDir(), "payload.json")

	originalAPIClient := apiHTTPClient
	originalDefaultClient := http.DefaultClient
	t.Cleanup(func() {
		apiHTTPClient = originalAPIClient
		http.DefaultClient = originalDefaultClient
	})
	var networkCalls atomic.Int32
	denyNetwork := terminalTestRoundTripFunc(func(request *http.Request) (*http.Response, error) {
		networkCalls.Add(1)
		return nil, fmt.Errorf("deferred submit must not access %s", request.URL)
	})
	apiHTTPClient = &http.Client{Transport: denyNetwork}
	http.DefaultClient = &http.Client{Transport: denyNetwork}

	binDir := t.TempDir()
	dockerSentinel := filepath.Join(t.TempDir(), "docker-called")
	mustWrite(t, filepath.Join(binDir, "docker"), "#!/bin/sh\nprintf called > \"$LMX_DOCKER_SENTINEL\"\nexit 99\n")
	if err := os.Chmod(filepath.Join(binDir, "docker"), 0o755); err != nil {
		t.Fatalf("make docker sentinel executable: %v", err)
	}
	t.Setenv("LMX_DOCKER_SENTINEL", dockerSentinel)
	t.Setenv("PATH", binDir+string(os.PathListSeparator)+os.Getenv("PATH"))

	if err := submitTerminalEval(parseArgs([]string{
		"eval", "terminal", "submit", artifactPath,
		"--dry-run", "--out", payloadPath, "--quiet",
	})); err != nil {
		t.Fatalf("submitTerminalEval monolithic dry-run: %v", err)
	}
	if networkCalls.Load() != 0 {
		t.Fatalf("deferred submit made %d model/API requests", networkCalls.Load())
	}
	if _, err := os.Stat(dockerSentinel); !os.IsNotExist(err) {
		t.Fatalf("deferred submit accessed Docker/verifier execution: %v", err)
	}

	batch := readTerminalSubmitBatch(t, payloadPath)
	if batch["dataset"] != "fixture-dataset" {
		t.Fatalf("dataset = %#v, want saved fixture-dataset", batch["dataset"])
	}
	shards := anySlice(batch["shards"])
	if len(shards) != 1 {
		t.Fatalf("shard payload count = %d, want 1", len(shards))
	}
	payload := asObject(shards[0])
	if payload["hfId"] != "fixture/saved-model" || payload["modelRevision"] != "saved-revision" || payload["shardIndex"] != float64(4) {
		t.Fatalf("saved model/shard defaults = %#v", payload)
	}
	if payload["quantization"] != "Q5_K_M" || payload["quantFormat"] != "gguf" || payload["runnerVersion"] != "saved-runner" {
		t.Fatalf("saved run metadata defaults = %#v", payload)
	}
	hardware := asObject(payload["hardware"])
	if hardware["cpu"] != "Saved CPU" || hardware["ramGb"] != float64(64) {
		t.Fatalf("saved hardware = %#v", hardware)
	}
	runConfig := asObject(payload["runConfig"])
	if runConfig["agent"] != "saved-agent" || runConfig["sourceArtifactVersion"] != float64(2) || runConfig["sourceServedModel"] != "saved-served-model" {
		t.Fatalf("saved/additive monolithic provenance = %#v", runConfig)
	}
	artifacts := anySlice(payload["artifacts"])
	response := stringValue(asObject(artifacts[0])["response"])
	if !strings.Contains(response, "SAVED_MONOLITHIC_RESPONSE") || !strings.Contains(response, "SAVED_MONOLITHIC_VERIFIER") {
		t.Fatalf("monolithic response/verifier were not packaged: %q", response)
	}
}

func TestTerminalUXMonolithicArtifactRejectsExplicitSavedMetadataConflicts(t *testing.T) {
	tests := []struct {
		name string
		args []string
	}{
		{name: "dataset", args: []string{"--dataset", "other-dataset"}},
		{name: "hf id", args: []string{"--hf-id", "other/model"}},
		{name: "model revision", args: []string{"--model-revision", "other-revision"}},
		{name: "shard index", args: []string{"--shard-index", "3"}},
		{name: "quantization", args: []string{"--quantization", "Q8_0"}},
		{name: "quant format", args: []string{"--quant-format", "safetensors"}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			artifactPath := writeTerminalMonolithicFixture(t)
			argv := []string{"eval", "terminal", "submit", artifactPath, "--dry-run", "--quiet"}
			argv = append(argv, tc.args...)
			err := submitTerminalEval(parseArgs(argv))
			var cliErr cliError
			if !errors.As(err, &cliErr) || cliErr.Code != "checkpoint_metadata_conflict" {
				t.Fatalf("error = %#v, want checkpoint_metadata_conflict", err)
			}
		})
	}

	t.Run("hardware", func(t *testing.T) {
		artifactPath := writeTerminalMonolithicFixture(t)
		hardwarePath := filepath.Join(t.TempDir(), "hardware.json")
		writeTerminalTestJSON(t, hardwarePath, map[string]any{"hwClass": "CPU_ONLY", "cpu": "Different CPU", "ramGb": 64})
		err := submitTerminalEval(parseArgs([]string{"eval", "terminal", "submit", artifactPath, "--hardware", hardwarePath, "--dry-run", "--quiet"}))
		var cliErr cliError
		if !errors.As(err, &cliErr) || cliErr.Code != "checkpoint_metadata_conflict" {
			t.Fatalf("error = %#v, want checkpoint_metadata_conflict", err)
		}
	})
}

func TestTerminalUXLegacyCheckpointDirectoryRemainsAccepted(t *testing.T) {
	runDir, _ := writeDeferredTerminalSubmitFixture(t)
	source, err := loadTerminalDeferredSource(runDir)
	if err != nil {
		t.Fatalf("loadTerminalDeferredSource legacy directory: %v", err)
	}
	if source.monolithic || source.root == "" || len(source.entries) != 2 || len(source.results) != 0 {
		t.Fatalf("legacy deferred source = %#v", source)
	}
	if source.entries[0].Task != "pass-task" || source.entries[1].Task != "fail-task" {
		t.Fatalf("legacy summary entries = %#v", source.entries)
	}
}

func TestTerminalUXFailureSummaryClassifiesOutcomesAndIncludesActionableFields(t *testing.T) {
	bundles := []terminalBundle{
		{Task: terminalTask{ID: "passed"}},
		{Task: terminalTask{ID: "model-error"}},
		{Task: terminalTask{ID: "unscored"}},
		{Task: terminalTask{ID: "timed-out"}},
		{Task: terminalTask{ID: "turn-limit", Agent: terminalAgentConfig{MaxTurns: 3}}},
		{Task: terminalTask{ID: "verifier-rejected"}},
	}
	results := []terminalTaskResult{
		{scored: true, pass: true},
		{errCode: "model_call_failed", errText: "endpoint returned 503", turns: 1, lastProgressAt: "2026-07-17T01:01:01Z"},
		{errText: "verifier never started", turns: 2, lastProgressAt: "2026-07-17T01:02:02Z"},
		{scored: true, agentOutcomeCode: "agent_timeout", agentOutcomeText: "external agent timed out before verification", turns: 1, lastProgressAt: "2026-07-17T01:03:03Z"},
		{scored: true, agentOutcomeCode: "max_turns_exhausted", agentOutcomeText: "agent exhausted its maximum turns before verifier success", turns: 3, lastProgressAt: "2026-07-17T01:04:04Z"},
		{scored: true, turns: 1, verifierOutput: "canonical verifier rejected the task", lastProgressAt: "2026-07-17T01:05:05Z"},
	}
	checkpoint := filepath.Join("saved", "atomic-checkpoint")
	stderr := captureTerminalTestStderr(t, func() {
		printTerminalFailureSummary(
			parseArgs([]string{"--json-status"}),
			bundles,
			results,
			terminalConfig{maxTurns: 3},
			checkpoint,
		)
	})
	line := strings.TrimSpace(stderr)
	var event map[string]any
	if err := json.Unmarshal([]byte(line), &event); err != nil {
		t.Fatalf("decode terminal_failure_summary event %q: %v", line, err)
	}
	if event["event"] != "terminal_failure_summary" || event["failedTasks"] != float64(5) {
		t.Fatalf("failure summary identity = %#v", event)
	}
	categories := asObject(event["categories"])
	for _, category := range []string{"model_call_failed", "infrastructure_error", "agent_timeout", "max_turns_exhausted", "verifier_failed"} {
		if categories[category] != float64(1) {
			t.Fatalf("category %q count = %#v; all categories = %#v", category, categories[category], categories)
		}
	}
	failures := anySlice(event["failures"])
	if len(failures) != 5 {
		t.Fatalf("failure rows = %d, want 5", len(failures))
	}
	byTask := map[string]map[string]any{}
	for _, raw := range failures {
		failure := asObject(raw)
		byTask[stringValue(failure["taskId"])] = failure
		for _, field := range []string{"taskId", "outcome", "verifierSummary", "turns", "maxTurns", "artifactPath", "lastProgressAt"} {
			if _, present := failure[field]; !present {
				t.Fatalf("failure row missing actionable field %q: %#v", field, failure)
			}
		}
	}
	if byTask["model-error"]["outcome"] != "model_call_failed" || byTask["model-error"]["verifierSummary"] != "endpoint returned 503" {
		t.Fatalf("model failure row = %#v", byTask["model-error"])
	}
	if byTask["turn-limit"]["outcome"] != "max_turns_exhausted" || byTask["turn-limit"]["turns"] != float64(3) || byTask["turn-limit"]["maxTurns"] != float64(3) {
		t.Fatalf("turn exhaustion row = %#v", byTask["turn-limit"])
	}
	wantArtifact := filepath.Join(checkpoint, terminalCheckpointWrapperName("verifier-rejected"))
	if byTask["verifier-rejected"]["artifactPath"] != wantArtifact || byTask["verifier-rejected"]["lastProgressAt"] != "2026-07-17T01:05:05Z" {
		t.Fatalf("verifier failure recovery fields = %#v, want artifact %q", byTask["verifier-rejected"], wantArtifact)
	}
	if _, included := byTask["passed"]; included {
		t.Fatalf("successful task appeared in failure summary: %#v", byTask["passed"])
	}
}

func writeTerminalMonolithicFixture(t *testing.T) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "completed-terminal-run.json")
	writeTerminalTestJSON(t, path, map[string]any{
		"summary": map[string]any{
			"artifactVersion": 2,
			"dataset":         "fixture-dataset",
			"hfId":            "fixture/saved-model",
			"modelRevision":   "saved-revision",
			"shardIndex":      4,
			"hardware":        map[string]any{"hwClass": "CPU_ONLY", "cpu": "Saved CPU", "ramGb": 64},
			"quantization":    "Q5_K_M",
			"quantFormat":     "gguf",
			"agent":           "saved-agent",
			"runnerVersion":   "saved-runner",
			"servedModel":     "saved-served-model",
			"declaredModel":   "fixture/saved-model",
			"modelResolution": map[string]any{"status": "matched"},
			"quantizationResolution": map[string]any{
				"trusted": "Q5_K_M", "trustedSource": "endpoint_file",
			},
			"runConfig": map[string]any{"protocol": "react-shell", "modelEndpoint": "http://saved-endpoint.test"},
		},
		"results": []any{map[string]any{
			"question_id": "saved-task", "pass": true, "scored": true,
			"wallTimeMs": 125,
			"tokenUsage": map[string]any{"inputTokens": 3, "outputTokens": 2, "totalTokens": 5, "modelCalls": 1},
			"turns":      1, "question": "Saved question", "prompt": "Saved prompt",
			"response": "SAVED_MONOLITHIC_RESPONSE", "verifierOutput": "SAVED_MONOLITHIC_VERIFIER",
		}},
	})
	return path
}

func captureTerminalTestStderr(t *testing.T, fn func()) string {
	t.Helper()
	reader, writer, err := os.Pipe()
	if err != nil {
		t.Fatalf("create stderr pipe: %v", err)
	}
	original := os.Stderr
	os.Stderr = writer
	fn()
	if err := writer.Close(); err != nil {
		t.Fatalf("close stderr writer: %v", err)
	}
	os.Stderr = original
	data, err := io.ReadAll(reader)
	if closeErr := reader.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		t.Fatalf("read captured stderr: %v", err)
	}
	return string(data)
}
