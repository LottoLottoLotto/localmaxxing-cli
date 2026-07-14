package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
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
	var payload map[string]any
	if err := json.Unmarshal(data, &payload); err != nil {
		t.Fatalf("decode durable payload: %v", err)
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
	if passArtifact["itemIndex"] != float64(0) || passArtifact["question"] != "Pass question with café" || passArtifact["prompt"] != "Pass prompt" || passArtifact["score"] != float64(1) || passArtifact["testPassed"] != true {
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
	if failArtifact["itemIndex"] != float64(1) || failArtifact["score"] != float64(0) || failArtifact["testPassed"] != false {
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
		"question_id": "pass-task",
		"pass":        true,
		"scored":      true,
		"wallTimeMs":  100,
		"tokenUsage":  map[string]any{"inputTokens": 11, "outputTokens": 7, "cacheReadTokens": 2, "cacheWriteTokens": 3, "totalTokens": 23, "modelCalls": 2},
		"turns":       2,
		"question":    "Pass question with café",
		"prompt":      "Pass prompt",
		"response":    "SAVED_PASS_RESPONSE_MUST_NOT_REPLACE_OMP",
		"verifierOutput": verifier,
	}}})
	writeTerminalTestJSON(t, filepath.Join(runDir, "fail-task.json"), map[string]any{"results": []any{map[string]any{
		"question_id":   "fail-task",
		"pass":          false,
		"scored":        true,
		"latencyMs":     200,
		"tokenUsage":    map[string]any{"input_tokens": 5, "output_tokens": 3, "cache_read_tokens": 4, "cache_write_tokens": 1, "total_tokens": 13, "model_calls": 3},
		"turns":         1,
		"question":      "Fail question",
		"prompt":        "Fail prompt",
		"response":      "SAVED_FAIL_RESPONSE",
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
			"args": map[string]any{"command": "printf 'tool ✓'"},
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
