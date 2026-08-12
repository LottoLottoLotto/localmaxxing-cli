package main

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
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

func TestRunTerminalEvalRejectsDetectedModelConflictBeforeLocalRun(t *testing.T) {
	const (
		declared = "Jackrong/Qwopus3.6-27B-v2-MTP-GGUF"
		detected = "Jackrong/Qwopus3.6-27B-Coder-MTP-GGUF"
		filename = "Qwopus3.6-27B-Coder-MTP-Q5_K_M.gguf"
	)
	var searchQueries []string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/models":
			io.WriteString(w, `{"data":[{"id":"qwopus-27b"}]}`)
		case "/props":
			io.WriteString(w, `{"model_path":"/models/`+filename+`"}`)
		case "/api/models/search":
			query := r.URL.Query().Get("q")
			searchQueries = append(searchQueries, query)
			if query == "Qwopus3.6-27B-Coder-MTP" {
				io.WriteString(w, `{"models":[{"hfId":"`+detected+`"}]}`)
			} else {
				io.WriteString(w, `{"models":[]}`)
			}
		case "/api/models/" + detected:
			io.WriteString(w, `{"siblings":[{"rfilename":"`+filename+`"}]}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	err := runTerminalEval(parseArgs([]string{
		"eval", "terminal", "run",
		"--task-dir", t.TempDir(),
		"--base-url", server.URL,
		"--api-url", server.URL,
		"--hf-api-url", server.URL,
		"--model", declared,
		"--dry-run",
	}), false)
	var cliErr cliError
	if !errors.As(err, &cliErr) || cliErr.Code != "model_identity_conflict" {
		t.Fatalf("terminal eval error = %#v, want model_identity_conflict", err)
	}
	if want := []string{"qwopus-27b", "Qwopus3.6-27B-Coder-MTP"}; !reflect.DeepEqual(searchQueries, want) {
		t.Fatalf("model search queries = %#v, want %#v", searchQueries, want)
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

func TestTerminalMaxTurnsEnforcementReportsHarnessResponsibility(t *testing.T) {
	tests := []struct {
		name string
		cfg  terminalConfig
		want string
	}{
		{name: "oracle", cfg: terminalConfig{oracle: true, agentCommand: "ignored"}, want: "not-applicable"},
		{name: "bundled Terminus adapter", cfg: terminalConfig{args: parseArgs([]string{"eval", "terminal", "run", "--agent", "terminus-2"}), agentCommand: "embedded-adapter"}, want: "bundled-adapter"},
		{name: "arbitrary external command", cfg: terminalConfig{agentCommand: "custom-agent"}, want: "not-enforced"},
		{name: "built-in loop", cfg: terminalConfig{}, want: "cli-agent-loop"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := terminalMaxTurnsEnforcement(test.cfg); got != test.want {
				t.Fatalf("terminalMaxTurnsEnforcement() = %q, want %q", got, test.want)
			}
		})
	}
}

func TestTerminalLimitSourceDistinguishesCLIManifestAndOmission(t *testing.T) {
	tests := []struct {
		name               string
		explicit, manifest int
		want               string
	}{
		{name: "CLI override", explicit: 7, manifest: 11, want: "cli"},
		{name: "task manifest", manifest: 11, want: "task-manifest"},
		{name: "omitted", want: "fallback"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := terminalLimitSource(test.explicit, test.manifest); got != test.want {
				t.Fatalf("terminalLimitSource(%d, %d) = %q, want %q", test.explicit, test.manifest, got, test.want)
			}
		})
	}
}

func TestTerminalCheckpointKeepsExternalTurnsNullAndLoadsLegacyNumericTurns(t *testing.T) {
	runDir := t.TempDir()
	store := &terminalLiveCheckpointStore{
		root: runDir,
		state: terminalLiveCheckpoint{
			Version: terminalLiveCheckpointVersion,
			State:   "running",
			TaskIDs: []string{"external-turns"},
		},
	}
	task := terminalTask{ID: "external-turns", Instruction: "exercise external turn provenance"}
	if err := store.persistTask(task, terminalTaskResult{scored: true, turns: 99, turnsUnreported: true}); err != nil {
		t.Fatalf("persist external turn checkpoint: %v", err)
	}
	data, err := os.ReadFile(terminalLiveResultPath(runDir, task.ID))
	if err != nil {
		t.Fatalf("read external turn checkpoint: %v", err)
	}
	var encoded map[string]any
	if err := json.Unmarshal(data, &encoded); err != nil {
		t.Fatalf("decode external turn checkpoint: %v", err)
	}
	turns, present := encoded["turns"]
	if !present || turns != nil {
		t.Fatalf("external checkpoint turns = %#v (present %v), want explicit null", turns, present)
	}

	scored := true
	legacyTurns := 4
	writeTerminalTestJSON(t, terminalLiveResultPath(runDir, task.ID), terminalSavedResult{
		QuestionID: task.ID,
		Scored:     &scored,
		Turns:      &legacyTurns,
		Response:   "legacy numeric checkpoint",
	})
	loaded, ok, err := loadTerminalLiveResult(runDir, task.ID)
	if err != nil {
		t.Fatalf("load legacy numeric turn checkpoint: %v", err)
	}
	if !ok || loaded.turns != legacyTurns || loaded.turnsUnreported || loaded.transcript != "legacy numeric checkpoint" {
		t.Fatalf("legacy numeric checkpoint = %+v (found %v), want reported %d turns", loaded, ok, legacyTurns)
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

func TestTerminalManifestDefaultsAreAuthoritative(t *testing.T) {
	manifestTask := terminalTask{Agent: terminalAgentConfig{TimeoutSec: 1200, MaxTurns: 37}}
	if got := terminalAgentTimeoutSec(terminalConfig{}, manifestTask); got != 1200 {
		t.Fatalf("manifest timeout = %d, want 1200", got)
	}
	if got := terminalAgentMaxTurns(terminalConfig{}, manifestTask); got != 37 {
		t.Fatalf("manifest max turns = %d, want 37", got)
	}

	omitted := terminalTask{}
	if got := terminalAgentTimeoutSec(terminalConfig{}, omitted); got != 4*60*60 {
		t.Fatalf("omitted manifest timeout = %d, want %d", got, 4*60*60)
	}
	if got := terminalAgentMaxTurns(terminalConfig{}, omitted); got != 200 {
		t.Fatalf("omitted manifest max turns = %d, want 200", got)
	}

	if got := terminalAgentTimeoutSec(terminalConfig{agentTimeoutSec: 42}, manifestTask); got != 42 {
		t.Fatalf("explicit agent timeout = %d, want 42", got)
	}
	if got := terminalAgentMaxTurns(terminalConfig{maxTurns: 7}, manifestTask); got != 7 {
		t.Fatalf("explicit max turns = %d, want 7", got)
	}
}

func TestTerminalCommandTimeoutDefaultsToThirtyMinutesAndRemainingBudget(t *testing.T) {
	if got := terminalCommandTimeoutSec(terminalConfig{}); got != 30*60 {
		t.Fatalf("default command timeout = %d, want %d", got, 30*60)
	}
	if got := terminalCommandTimeoutSec(terminalConfig{commandTimeoutSec: 30}); got != 30 {
		t.Fatalf("explicit command timeout = %d, want 30", got)
	}

	requested := time.Duration(terminalCommandTimeoutSec(terminalConfig{})) * time.Second
	longDeadline := time.Now().Add(time.Hour)
	if got := terminalCommandExecutionTimeout("sleep infinity", requested, longDeadline); got > 30*time.Minute || got < 29*time.Minute+59*time.Second {
		t.Fatalf("default command execution timeout = %v, want about 30m", got)
	}
	shortDeadline := time.Now().Add(2 * time.Minute)
	if got := terminalCommandExecutionTimeout("sleep infinity", requested, shortDeadline); got > 2*time.Minute || got < time.Minute+59*time.Second {
		t.Fatalf("remaining-budget command timeout = %v, want about 2m", got)
	}
}

func TestTerminalChangedFingerprintFields(t *testing.T) {
	got := terminalChangedFingerprintFields(
		map[string]string{"baseUrl": "same", "shellMode": "old"},
		map[string]string{"baseUrl": "same", "shellMode": "new", "maxTurns": "added"},
	)
	if !slicesEqual(got, []string{"maxTurns", "shellMode"}) {
		t.Fatalf("changed fields = %#v, want maxTurns and shellMode", got)
	}
}

func TestTerminalStagnationTrackerWarnsThenStopsAndResetsOnProgress(t *testing.T) {
	var tracker terminalStagnationTracker
	for repeat := 1; repeat <= 6; repeat++ {
		warn, stop := tracker.observe("printf same", "same\n[exit=0]")
		if warn != (repeat == 3) {
			t.Fatalf("repeat %d warning = %v, want %v", repeat, warn, repeat == 3)
		}
		if stop != (repeat >= 6) {
			t.Fatalf("repeat %d stop = %v, want %v", repeat, stop, repeat >= 6)
		}
	}
	if warn, stop := tracker.observe("printf changed", "changed\n[exit=0]"); warn || stop {
		t.Fatalf("changed interaction should reset stagnation: warn=%v stop=%v", warn, stop)
	}
	if tracker.repeats != 1 {
		t.Fatalf("changed interaction repeats = %d, want 1", tracker.repeats)
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

func TestCallTerminalModelWithHeartbeatEmitsInFlightStatusAndReturnsResult(t *testing.T) {
	stderr := os.Stderr
	reader, writer, err := os.Pipe()
	if err != nil {
		t.Fatalf("create stderr capture: %v", err)
	}
	os.Stderr = writer
	restoreStderr := func() {
		os.Stderr = stderr
		_ = writer.Close()
		_ = reader.Close()
	}
	defer restoreStderr()

	type callResult struct {
		content   string
		reasoning string
		usage     terminalTokenUsage
		err       error
	}
	releaseCall := make(chan struct{})
	resultCh := make(chan callResult, 1)
	wantUsage := terminalTokenUsage{inputTokens: 3, outputTokens: 2, totalTokens: 5, modelCalls: 1}
	go func() {
		content, reasoning, usage, err := callTerminalModelWithHeartbeat(
			terminalConfig{args: parseArgs([]string{"eval", "terminal", "run", "--json-status"}), modelHeartbeatInterval: 2 * time.Millisecond},
			"heartbeat-task", 4, "retry", time.Now().Add(5*time.Second),
			func() (string, string, terminalTokenUsage, error) {
				<-releaseCall
				return "call-content", "call-reasoning", wantUsage, nil
			},
		)
		resultCh <- callResult{content: content, reasoning: reasoning, usage: usage, err: err}
	}()

	type decodedStatus struct {
		value map[string]any
		err   error
	}
	statusCh := make(chan decodedStatus, 1)
	go func() {
		var status map[string]any
		err := json.NewDecoder(reader).Decode(&status)
		statusCh <- decodedStatus{value: status, err: err}
	}()

	var decoded decodedStatus
	select {
	case decoded = <-statusCh:
		close(releaseCall)
	case <-time.After(2 * time.Second):
		close(releaseCall)
		<-resultCh
		t.Fatal("timed out waiting for in-flight terminal model heartbeat")
	}
	result := <-resultCh
	restoreStderr()

	if decoded.err != nil {
		t.Fatalf("decode heartbeat status: %v", decoded.err)
	}
	status := decoded.value
	if status["event"] != "terminal_model_call_heartbeat" || status["taskId"] != "heartbeat-task" || status["turn"] != float64(4) || status["attempt"] != "retry" {
		t.Fatalf("heartbeat identity = %#v", status)
	}
	elapsed, elapsedOK := status["elapsedSec"].(float64)
	remaining, remainingOK := status["agentTimeRemainingSec"].(float64)
	if !elapsedOK || elapsed < 0 || elapsed > 1 || !remainingOK || remaining <= 0 || remaining > 5 {
		t.Fatalf("heartbeat timing = elapsed %#v remaining %#v, want live bounded agent timing", status["elapsedSec"], status["agentTimeRemainingSec"])
	}
	if result.err != nil || result.content != "call-content" || result.reasoning != "call-reasoning" || !reflect.DeepEqual(result.usage, wantUsage) {
		t.Fatalf("completed model call = %+v, want the delayed call result and usage", result)
	}
}

func TestRunExternalTerminalAgentHeartbeatOrdersSchemaAndStopsOnCancellation(t *testing.T) {
	originalStderr := os.Stderr
	reader, writer, err := os.Pipe()
	if err != nil {
		t.Fatalf("create external heartbeat capture: %v", err)
	}
	os.Stderr = writer
	restoreStderr := func() {
		os.Stderr = originalStderr
		_ = writer.Close()
		_ = reader.Close()
	}
	defer restoreStderr()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	type decodedEvents struct {
		values []map[string]any
		err    error
	}
	eventsCh := make(chan decodedEvents, 1)
	go func() {
		decoder := json.NewDecoder(reader)
		events := []map[string]any{}
		for {
			var event map[string]any
			if err := decoder.Decode(&event); err != nil {
				if errors.Is(err, io.EOF) {
					err = nil
				}
				eventsCh <- decodedEvents{values: events, err: err}
				return
			}
			events = append(events, event)
			if event["event"] == "terminal_external_agent_heartbeat" {
				cancel()
			}
		}
	}()

	transcript, _, runErr := runExternalTerminalAgent(
		ctx,
		terminalTask{ID: "external-heartbeat", Instruction: "wait for cancellation"},
		t.TempDir(),
		"unused-container",
		"",
		"fixture-model",
		terminalConfig{
			args:                           parseArgs([]string{"eval", "terminal", "run", "--json-status"}),
			agentCommand:                   "while :; do :; done",
			agentExecution:                 "routed-shell",
			agentTimeoutSec:                1,
			externalAgentHeartbeatInterval: time.Millisecond,
			traceRoot:                      t.TempDir(),
		},
	)
	os.Stderr = originalStderr
	if err := writer.Close(); err != nil {
		t.Fatalf("close external heartbeat writer: %v", err)
	}
	decoded := <-eventsCh
	_ = reader.Close()
	if decoded.err != nil {
		t.Fatalf("decode external heartbeat events: %v", decoded.err)
	}
	if ctx.Err() != context.Canceled || runErr == nil {
		t.Fatalf("external cancellation = context %v error %v, want cancelled command failure", ctx.Err(), runErr)
	}
	if !strings.Contains(transcript, "[exit=130]") {
		t.Fatalf("cancelled external transcript = %q, want exit 130", transcript)
	}
	events := decoded.values
	if len(events) < 3 {
		t.Fatalf("external lifecycle events = %#v, want start, heartbeat, and done", events)
	}
	if events[0]["event"] != "terminal_external_agent_started" {
		t.Fatalf("first external lifecycle event = %#v, want started", events[0])
	}
	if events[len(events)-1]["event"] != "terminal_external_agent_done" {
		t.Fatalf("last external lifecycle event = %#v, want done", events[len(events)-1])
	}
	heartbeats := 0
	for _, event := range events[1 : len(events)-1] {
		if event["event"] != "terminal_external_agent_heartbeat" {
			t.Fatalf("event between external start and done = %#v, want heartbeat only", event)
		}
		heartbeats++
		elapsed, elapsedOK := event["elapsedSec"].(float64)
		remaining, remainingOK := event["agentTimeRemainingSec"].(float64)
		if event["taskId"] != "external-heartbeat" || event["execution"] != "routed-shell" || !elapsedOK || elapsed < 0 || elapsed > 1 || !remainingOK || remaining < 0 || remaining > 1 {
			t.Fatalf("external heartbeat schema = %#v, want bounded timing and task/execution identity", event)
		}
	}
	if heartbeats == 0 {
		t.Fatal("external agent emitted no heartbeat before cancellation")
	}
	done := events[len(events)-1]
	if done["taskId"] != "external-heartbeat" || done["execution"] != "routed-shell" || done["exitCode"] != float64(130) || done["timedOut"] != false {
		t.Fatalf("external done event = %#v, want cancelled exit without timeout", done)
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

func TestRunTerminalTaskAgentTimeoutStillRunsVerifierAndScores(t *testing.T) {
	dockerLog := installFakeTerminalDocker(t, "")
	bundleDir := t.TempDir()
	mustMkdir(t, filepath.Join(bundleDir, "tests"))
	mustWrite(t, filepath.Join(bundleDir, "tests", "test.sh"), "fixture verifier")

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		time.Sleep(1500 * time.Millisecond)
		_ = json.NewEncoder(w).Encode(map[string]any{"choices": []any{map[string]any{"message": map[string]any{"content": "too late"}}}})
	}))
	defer server.Close()

	task := terminalTask{
		ID:          "agent-timeout-scored",
		Instruction: "Let the model request consume its agent budget.",
		Image:       terminalImage{Prebuilt: "fake:image"},
		Agent:       terminalAgentConfig{MaxTurns: 1},
		Verifier: terminalVerifierConfig{
			Command:    "fixture-verifier",
			RewardFile: "/logs/verifier/reward.txt",
		},
	}
	result := runTerminalTask(context.Background(), task, bundleDir, server.URL, "fixture-model", terminalConfig{
		args:            parseArgs([]string{"eval", "terminal", "run", "--quiet"}),
		shellMode:       "stateless",
		agentTimeoutSec: 1,
		endpointTimeout: 10 * time.Second,
	})
	if !result.scored || !result.pass {
		t.Fatalf("timed-out agent result = %+v, want verifier-scored pass", result)
	}
	commands, err := os.ReadFile(dockerLog)
	if err != nil {
		t.Fatalf("read fake Docker log: %v", err)
	}
	if !strings.Contains(string(commands), "fixture-verifier") {
		t.Fatalf("verifier was not invoked after agent timeout; Docker calls:\n%s", commands)
	}
}

func TestRunTerminalEvalRejectsIncompleteShardBeforeSubmit(t *testing.T) {
	installFakeTerminalDocker(t, "bad-task")
	bundleRoot := t.TempDir()
	writeTerminalRuntimeBundle(t, bundleRoot, "good-task")
	writeTerminalRuntimeBundle(t, bundleRoot, "bad-task")
	hardwarePath := filepath.Join(t.TempDir(), "hardware.json")
	writeTerminalTestJSON(t, hardwarePath, map[string]any{"hwClass": "CPU_ONLY", "cpu": "Fixture CPU", "ramGb": 8})

	var submitCalls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/models":
			_ = json.NewEncoder(w).Encode(map[string]any{"data": []any{map[string]any{"id": "served-model"}}})
		case "/v1/chat/completions":
			_ = json.NewEncoder(w).Encode(map[string]any{"choices": []any{map[string]any{"message": map[string]any{"content": "TASK_COMPLETE"}}}})
		case "/api/benchmarks/terminal-fixture/submit":
			submitCalls.Add(1)
			t.Fatal("incomplete shard reached submission endpoint")
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	err := runTerminalEval(parseArgs([]string{
		"eval", "terminal", "run",
		"--task-dir", bundleRoot,
		"--dataset", "terminal-fixture",
		"--shard", "1",
		"--base-url", server.URL,
		"--served-model", "served-model",
		"--model", "fixture/model",
		"--quantization", "Q4_K_M",
		"--quant-format", "gguf",
		"--hardware", hardwarePath,
		"--api-url", server.URL,
		"--api-key", "fixture-key",
		"--submit", "--quiet",
	}), false)
	var cliErr cliError
	if !errors.As(err, &cliErr) || cliErr.Code != "incomplete_shard" {
		t.Fatalf("error = %#v, want incomplete_shard", err)
	}
	if got := submitCalls.Load(); got != 0 {
		t.Fatalf("submission calls = %d, want zero", got)
	}
	details := asObject(cliErr.Details)
	if details["tasks"] != 2 || details["scored"] != 1 {
		t.Fatalf("incomplete shard evidence = %#v, want tasks=2 scored=1", details)
	}
}

func TestRunTerminalEvalSelectsFirstModelCoverageMissingShard(t *testing.T) {
	archive := terminalTestBundleArchive(t, "coverage-task")
	hardwarePath := filepath.Join(t.TempDir(), "hardware.json")
	writeTerminalTestJSON(t, hardwarePath, map[string]any{"hwClass": "CPU_ONLY", "cpu": "Fixture CPU", "ramGb": 8})
	var coverageCalls atomic.Int32
	var requestedShards []string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/benchmarks/terminal-fixture/coverage":
			coverageCalls.Add(1)
			query := r.URL.Query()
			if query.Get("hfId") != "fixture/model" || query.Get("quantization") != "Q4_K_M" || query.Get("quantFormat") != "gguf" {
				t.Errorf("coverage identity query = %q, want model/quantization/format scoped", r.URL.RawQuery)
			}
			if query.Get("harnessKey") == "" {
				t.Errorf("coverage query omitted harnessKey: %q", r.URL.RawQuery)
			}
			_ = json.NewEncoder(w).Encode(map[string]any{
				"dataset":  map[string]any{"slug": "terminal-fixture", "shardCount": 3},
				"coverage": map[string]any{"shardsCovered": []any{1, 3}},
			})
		case "/api/benchmarks/terminal-fixture/shard":
			requestedShards = append(requestedShards, r.URL.Query().Get("shard"))
			_ = json.NewEncoder(w).Encode(map[string]any{
				"shard":       map[string]any{"shardIndex": numberFromString(r.URL.Query().Get("shard")), "itemCount": 1},
				"downloadUrl": terminalFixtureServerURL(r) + "/manifest",
			})
		case "/manifest":
			_ = json.NewEncoder(w).Encode(map[string]any{"question_id": "coverage-task", "bundle_key": "fixtures/coverage-task.tar.gz"})
		case "/api/benchmarks/storage/download-url":
			_ = json.NewEncoder(w).Encode(map[string]any{"downloadUrl": terminalFixtureServerURL(r) + "/bundle"})
		case "/bundle":
			_, _ = w.Write(archive)
		case "/v1/models":
			_ = json.NewEncoder(w).Encode(map[string]any{"data": []any{map[string]any{"id": "served-model"}}})
		case "/v1/chat/completions":
			http.Error(w, "model task execution forbidden during preflight", http.StatusInternalServerError)
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	baseArgs := []string{
		"eval", "terminal", "run", "terminal-fixture",
		"--api-url", server.URL,
		"--api-key", "fixture-key",
		"--base-url", server.URL,
		"--model", "fixture/model",
		"--quantization", "Q4_K_M",
		"--quant-format", "gguf",
		"--hardware", hardwarePath,
		"--submit", "--dry-run", "--quiet",
	}
	firstArgs := append(append([]string(nil), baseArgs...), "--out", filepath.Join(t.TempDir(), "missing-preflight.json"))
	if err := runTerminalEval(parseArgs(firstArgs), false); err != nil {
		t.Fatalf("preflight uncovered shard: %v", err)
	}

	explicitArgs := append(append([]string(nil), baseArgs...), "--shard", "3", "--out", filepath.Join(t.TempDir(), "explicit-preflight.json"))
	if err := runTerminalEval(parseArgs(explicitArgs), false); err != nil {
		t.Fatalf("preflight explicit shard: %v", err)
	}
	if got := coverageCalls.Load(); got != 1 {
		t.Fatalf("coverage calls = %d, want one (explicit shard must bypass coverage)", got)
	}
	if !reflect.DeepEqual(requestedShards, []string{"2", "3"}) {
		t.Fatalf("shard requests = %q, want [2 3]", requestedShards)
	}
}

func TestRunTerminalEvalOutIncludesSubmissionReceiptAndEffectiveLimits(t *testing.T) {
	installFakeTerminalDocker(t, "")
	archive := terminalTestBundleArchive(t, "receipt-task")
	hardwarePath := filepath.Join(t.TempDir(), "hardware.json")
	writeTerminalTestJSON(t, hardwarePath, map[string]any{"hwClass": "CPU_ONLY", "cpu": "Fixture CPU", "ramGb": 8})
	outPath := filepath.Join(t.TempDir(), "terminal-result.json")

	var submitted map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/benchmarks/terminal-fixture/shard":
			_ = json.NewEncoder(w).Encode(map[string]any{"shard": map[string]any{"shardIndex": 2, "itemCount": 1}, "downloadUrl": terminalFixtureServerURL(r) + "/manifest"})
		case "/manifest":
			_ = json.NewEncoder(w).Encode(map[string]any{"question_id": "receipt-task", "bundle_key": "fixtures/receipt-task.tar.gz"})
		case "/api/benchmarks/storage/download-url":
			_ = json.NewEncoder(w).Encode(map[string]any{"downloadUrl": terminalFixtureServerURL(r) + "/bundle"})
		case "/bundle":
			_, _ = w.Write(archive)
		case "/v1/models":
			_ = json.NewEncoder(w).Encode(map[string]any{"data": []any{map[string]any{"id": "served-model"}}})
		case "/v1/chat/completions":
			_ = json.NewEncoder(w).Encode(map[string]any{"choices": []any{map[string]any{"message": map[string]any{"content": "TASK_COMPLETE"}}}})
		case "/api/benchmarks/terminal-fixture/submit":
			if err := json.NewDecoder(r.Body).Decode(&submitted); err != nil {
				t.Errorf("decode terminal submission: %v", err)
				return
			}
			w.WriteHeader(http.StatusCreated)
			_ = json.NewEncoder(w).Encode(map[string]any{
				"run":       map[string]any{"id": "run-receipt", "status": "APPROVED"},
				"aggregate": map[string]any{"pooledScore": 1.0, "ciLower": 0.25, "ciUpper": 1.0, "shardsCovered": []any{2}},
			})
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	err := runTerminalEval(parseArgs([]string{
		"eval", "terminal", "run", "terminal-fixture",
		"--shard", "2",
		"--api-url", server.URL,
		"--api-key", "fixture-key",
		"--base-url", server.URL,
		"--served-model", "served-model",
		"--model", "fixture/model",
		"--quantization", "Q4_K_M",
		"--quant-format", "gguf",
		"--hardware", hardwarePath,
		"--agent-timeout", "7",
		"--out", outPath,
		"--submit", "--quiet",
	}), false)
	if err != nil {
		t.Fatalf("run terminal submit: %v", err)
	}
	saved := readTerminalSubmitBatch(t, outPath)
	summary := asObject(saved["summary"])
	if len(anySlice(saved["results"])) != 1 || summary["scored"] != float64(1) {
		t.Fatalf("saved run body was not preserved: %#v", saved)
	}
	if summary["shardIndex"] != float64(2) || summary["runId"] != "run-receipt" || summary["status"] != "APPROVED" {
		t.Fatalf("summary submission identity = %#v, want shard/run/status receipt", summary)
	}
	if summary["pooledScore"] != float64(1) || summary["ciLower"] != float64(0.25) || summary["ciUpper"] != float64(1) || !reflect.DeepEqual(anySlice(summary["coverage"]), []any{float64(2)}) {
		t.Fatalf("summary aggregate receipt = %#v, want pooled score/CI/coverage", summary)
	}
	receipt := asObject(saved["submission"])
	if receipt["shardIndex"] != float64(2) || receipt["submitted"] != float64(1) {
		t.Fatalf("submission receipt counts = %#v, want shard 2 submitted 1", receipt)
	}
	if !reflect.DeepEqual(asObject(receipt["run"]), map[string]any{"id": "run-receipt", "status": "APPROVED"}) {
		t.Fatalf("submission run receipt = %#v", receipt["run"])
	}
	aggregate := asObject(receipt["aggregate"])
	if aggregate["pooledScore"] != float64(1) || !reflect.DeepEqual(anySlice(aggregate["shardsCovered"]), []any{float64(2)}) {
		t.Fatalf("submission aggregate receipt = %#v", aggregate)
	}
	runConfig := asObject(submitted["runConfig"])
	if runConfig["maxTurns"] != float64(1) {
		t.Fatalf("effective maxTurns = %#v, want manifest limit 1", runConfig["maxTurns"])
	}
	if runConfig["maxTurnsPolicy"] != "per-task-manifest-or-fallback" || runConfig["agentTimeoutPolicy"] != "cli-override" {
		t.Fatalf("limit policies = (%#v, %#v), want manifest/fallback max-turn policy and CLI timeout policy", runConfig["maxTurnsPolicy"], runConfig["agentTimeoutPolicy"])
	}
	if runConfig["maxTurnsEnforcement"] != "cli-agent-loop" {
		t.Fatalf("run maxTurnsEnforcement = %#v, want cli-agent-loop", runConfig["maxTurnsEnforcement"])
	}
	commandTimeoutSec, ok := runConfig["commandTimeoutSec"].(float64)
	if !ok || commandTimeoutSec <= 0 {
		t.Fatalf("effective commandTimeoutSec = %#v, want a nonzero fallback limit", runConfig["commandTimeoutSec"])
	}
	taskLimits := anySlice(runConfig["taskLimits"])
	if len(taskLimits) != 1 {
		t.Fatalf("taskLimits = %#v, want one receipt-task entry", runConfig["taskLimits"])
	}
	limit := asObject(taskLimits[0])
	if limit["taskId"] != "receipt-task" || limit["maxTurns"] != float64(1) || limit["maxTurnsSource"] != "task-manifest" {
		t.Fatalf("manifest task limit metadata = %#v", limit)
	}
	if limit["maxTurnsEnforcement"] != "cli-agent-loop" {
		t.Fatalf("task maxTurnsEnforcement = %#v, want cli-agent-loop", limit["maxTurnsEnforcement"])
	}
	if limit["agentTimeoutSec"] != float64(7) || limit["agentTimeoutSource"] != "cli" {
		t.Fatalf("CLI task limit metadata = %#v", limit)
	}
	if limit["commandTimeoutSec"] != commandTimeoutSec || limit["commandTimeoutSource"] != "fallback" {
		t.Fatalf("fallback task limit metadata = %#v, want commandTimeoutSec=%v from fallback", limit, commandTimeoutSec)
	}
}

func TestRunTerminalEvalExplicitDryRunPreflightsWithoutDockerOrModel(t *testing.T) {
	dockerLog := installFakeTerminalDocker(t, "")
	bundleRoot := t.TempDir()
	writeTerminalRuntimeBundle(t, bundleRoot, "dry-run-task")
	outPath := filepath.Join(t.TempDir(), "preflight.json")
	var modelCalls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/models":
			_ = json.NewEncoder(w).Encode(map[string]any{"data": []any{map[string]any{"id": "served-model"}}})
		case "/v1/chat/completions":
			modelCalls.Add(1)
			_ = json.NewEncoder(w).Encode(map[string]any{"choices": []any{map[string]any{"message": map[string]any{"content": "TASK_COMPLETE"}}}})
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	err := runTerminalEval(parseArgs([]string{
		"eval", "terminal", "run",
		"--task-dir", bundleRoot,
		"--base-url", server.URL,
		"--model", "fixture/model",
		"--out", outPath,
		"--dry-run", "--quiet",
	}), false)
	if err != nil {
		t.Fatalf("terminal explicit dry-run: %v", err)
	}
	if got := modelCalls.Load(); got != 0 {
		t.Fatalf("model calls = %d, want zero", got)
	}
	if data, err := os.ReadFile(dockerLog); err != nil && !os.IsNotExist(err) {
		t.Fatalf("read fake Docker log: %v", err)
	} else if len(data) != 0 {
		t.Fatalf("explicit dry-run invoked Docker:\n%s", data)
	}
	plan := readTerminalSubmitBatch(t, outPath)
	if plan["dryRun"] != true || plan["tasks"] != float64(1) {
		t.Fatalf("preflight plan = %#v, want dryRun=true tasks=1", plan)
	}
}

func TestStartDetachedTerminalEvalCreatesProcessRecordAndSeparatesWorkerStreams(t *testing.T) {
	runDir := t.TempDir()
	t.Setenv(terminalJobTestHelperEnv, "detached-output")

	stdout, stderr, err := captureTerminalStreams(t, func() error {
		return startDetachedTerminalEval(parseArgs([]string{
			"eval", "terminal", "run",
			"--run-dir", runDir,
			"--detach", "--json-status", "--quiet",
		}))
	})
	if err != nil {
		t.Fatalf("start detached terminal worker: %v", err)
	}
	if stdout != "" || stderr != "" {
		t.Fatalf("quiet detached launcher output = stdout %q stderr %q, want both empty", stdout, stderr)
	}

	workerStdout := waitForTerminalFileContent(t, terminalStdoutPath(runDir), "detached worker stdout")
	workerStderr := waitForTerminalFileContent(t, terminalStderrPath(runDir), "detached worker event")
	workerEvents, err := os.ReadFile(terminalEventsPath(runDir))
	if err != nil {
		t.Fatalf("read detached event log: %v", err)
	}
	if bytes.Contains(workerStdout, []byte("detached worker event")) {
		t.Fatalf("worker stderr leaked into stdout file: %q", workerStdout)
	}
	if bytes.Contains(workerStderr, []byte("detached worker stdout")) {
		t.Fatalf("worker stdout leaked into stderr file: %q", workerStderr)
	}
	if bytes.Contains(workerEvents, []byte("detached worker event")) {
		t.Fatalf("raw worker stderr corrupted the event log: %q", workerEvents)
	}
	for index, line := range strings.Split(strings.TrimSpace(string(workerEvents)), "\n") {
		if line == "" {
			continue
		}
		var event map[string]any
		if err := json.Unmarshal([]byte(line), &event); err != nil {
			t.Fatalf("detached event log line %d is not JSON: %v\n%s", index+1, err, line)
		}
	}

	record, err := loadTerminalProcessRecord(runDir)
	if err != nil {
		t.Fatalf("load detached process record: %v", err)
	}
	root, err := filepath.Abs(runDir)
	if err != nil {
		t.Fatalf("resolve run directory: %v", err)
	}
	if record.PID <= 0 || record.State != "running" || record.RunDir != root {
		t.Fatalf("detached process record = %+v, want positive PID and running state for %s", record, root)
	}
	if record.EventsPath != terminalEventsPath(root) || record.StdoutPath != terminalStdoutPath(root) || record.StderrPath != terminalStderrPath(root) {
		t.Fatalf("detached output paths = events %q stdout %q stderr %q, want canonical files under %s", record.EventsPath, record.StdoutPath, record.StderrPath, root)
	}
}

func TestTerminalLogsWithholdsTrailingNonNewlineFragment(t *testing.T) {
	runDir := t.TempDir()
	mustWrite(t, terminalEventsPath(runDir), "{\"event\":\"complete\"}\n{\"event\":\"partial\"")

	stdout, stderr, err := captureTerminalStreams(t, func() error {
		return handleEvalTerminal("logs", parseArgs([]string{"eval", "terminal", "logs", runDir}))
	})
	if err != nil {
		t.Fatalf("read terminal logs: %v", err)
	}
	if stderr != "" {
		t.Fatalf("terminal logs wrote stderr: %q", stderr)
	}
	if stdout != "{\"event\":\"complete\"}\n" {
		t.Fatalf("terminal logs output = %q, want only the complete JSONL record", stdout)
	}
}

func TestTerminalStatusReportsCheckpointProgressAndActiveTask(t *testing.T) {
	runDir := t.TempDir()
	mustMkdir(t, filepath.Join(runDir, "results"))
	writeTerminalTestJSON(t, filepath.Join(runDir, "run.json"), terminalLiveCheckpoint{
		Version:        terminalLiveCheckpointVersion,
		State:          "running",
		Dataset:        "fixture-dataset",
		ShardIndex:     2,
		TaskIDs:        []string{"task-passed", "task-unscored", "task-running"},
		CompletedTasks: []string{"task-passed", "task-unscored"},
		ActiveTasks: []terminalLiveActiveTask{
			{TaskID: "task-running", Index: 2, StartedAt: "2026-07-19T12:00:00Z"},
		},
		CreatedAt: "2026-07-19T11:59:00Z",
		UpdatedAt: "2026-07-19T12:00:00Z",
	})
	scored, unscored := true, false
	writeTerminalTestJSON(t, terminalLiveResultPath(runDir, "task-passed"), terminalSavedResult{
		QuestionID: "task-passed", Pass: true, Scored: &scored,
	})
	writeTerminalTestJSON(t, terminalLiveResultPath(runDir, "task-unscored"), terminalSavedResult{
		QuestionID: "task-unscored", Scored: &unscored, ErrorCode: "verifier_failed",
	})

	stdout, stderr, err := captureTerminalStreams(t, func() error {
		return handleEvalTerminal("status", parseArgs([]string{"eval", "terminal", "status", runDir, "--json"}))
	})
	if err != nil {
		t.Fatalf("read terminal status: %v", err)
	}
	if stderr != "" {
		t.Fatalf("terminal status wrote stderr: %q", stderr)
	}
	var status map[string]any
	if err := json.Unmarshal([]byte(stdout), &status); err != nil {
		t.Fatalf("decode terminal status JSON: %v\n%s", err, stdout)
	}
	for field, want := range map[string]any{
		"state": "running", "dataset": "fixture-dataset", "shardIndex": float64(2),
		"tasksTotal": float64(3), "tasksPersisted": float64(2), "tasksScored": float64(1),
		"tasksPassed": float64(1), "tasksUnscored": float64(1),
	} {
		if got := status[field]; got != want {
			t.Errorf("status[%q] = %#v, want %#v", field, got, want)
		}
	}
	active, ok := status["activeTasks"].([]any)
	if !ok || len(active) != 1 || stringValue(asObject(active[0])["taskId"]) != "task-running" || numberField(asObject(active[0]), "index") != 2 {
		t.Fatalf("activeTasks = %#v, want task-running at index 2", status["activeTasks"])
	}
	root, err := filepath.Abs(runDir)
	if err != nil {
		t.Fatalf("resolve run directory: %v", err)
	}
	if status["eventsPath"] != terminalEventsPath(root) {
		t.Fatalf("eventsPath = %#v, want %q", status["eventsPath"], terminalEventsPath(root))
	}
}

func TestTerminalStatusSeparatesCanonicalActivityFromCheckpointUpdate(t *testing.T) {
	runDir := t.TempDir()
	checkpointUpdatedAt := "2026-07-19T12:00:00Z"
	lastCompleteActivityAt := "2026-07-19T12:03:04.567Z"
	writeTerminalTestJSON(t, filepath.Join(runDir, "run.json"), terminalLiveCheckpoint{
		Version:   terminalLiveCheckpointVersion,
		State:     "running",
		Dataset:   "fixture-dataset",
		CreatedAt: "2026-07-19T11:59:00Z",
		UpdatedAt: checkpointUpdatedAt,
	})
	mustWrite(t, terminalEventsPath(runDir),
		"{\"event\":\"terminal_task_started\",\"time\":\"2026-07-19T12:01:00Z\"}\n"+
			"{\"event\":\"terminal_external_agent_heartbeat\",\"time\":\""+lastCompleteActivityAt+"\"}\n"+
			"{\"event\":\"terminal_external_agent_heartbeat\",\"time\":\"2026-07-19T12:05:00Z\"")

	stdout, stderr, err := captureTerminalStreams(t, func() error {
		return handleEvalTerminal("status", parseArgs([]string{"eval", "terminal", "status", runDir, "--json"}))
	})
	if err != nil {
		t.Fatalf("read terminal activity status: %v", err)
	}
	if stderr != "" {
		t.Fatalf("terminal activity status wrote stderr: %q", stderr)
	}
	var status map[string]any
	if err := json.Unmarshal([]byte(stdout), &status); err != nil {
		t.Fatalf("decode terminal activity status: %v\n%s", err, stdout)
	}
	if status["updatedAt"] != checkpointUpdatedAt {
		t.Fatalf("status updatedAt = %#v, want checkpoint transition %q", status["updatedAt"], checkpointUpdatedAt)
	}
	if status["lastActivityAt"] != lastCompleteActivityAt {
		t.Fatalf("status lastActivityAt = %#v, want newest complete canonical event %q", status["lastActivityAt"], lastCompleteActivityAt)
	}
}

func TestTerminalCancelDurablyRecordsRequestBeforeSignallingRecordedWorker(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("cooperative signal receipt and robust process identity are exercised on Linux")
	}
	runDir := t.TempDir()
	root, err := filepath.Abs(runDir)
	if err != nil {
		t.Fatalf("resolve run directory: %v", err)
	}
	executable, err := os.Executable()
	if err != nil {
		t.Fatalf("locate test executable: %v", err)
	}
	helperDir := t.TempDir()
	readyPath := filepath.Join(helperDir, "ready")
	signalledPath := filepath.Join(helperDir, "signalled")
	observedPath := filepath.Join(helperDir, "observed-events")
	cmd := exec.Command(executable, "-test.run=^$")
	cmd.Env = append(os.Environ(),
		terminalJobTestHelperEnv+"=wait-for-termination",
		"LMX_TEST_TERMINAL_READY="+readyPath,
		"LMX_TEST_TERMINAL_SIGNALLED="+signalledPath,
		"LMX_TEST_TERMINAL_EVENTS="+terminalEventsPath(root),
		"LMX_TEST_TERMINAL_OBSERVED="+observedPath,
	)
	configureDetachedCommand(cmd)
	if err := cmd.Start(); err != nil {
		t.Fatalf("start controlled detached worker: %v", err)
	}
	identity, err := captureProcessIdentity(cmd.Process.Pid)
	if err != nil {
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
		t.Fatalf("capture controlled worker identity: %v", err)
	}
	waitDone := make(chan struct{})
	var waitErr error
	go func() {
		waitErr = cmd.Wait()
		close(waitDone)
	}()
	defer func() {
		if processMatchesIdentity(cmd.Process.Pid, identity) {
			_ = signalDetachedProcess(cmd.Process.Pid, true)
		}
		select {
		case <-waitDone:
		case <-time.After(5 * time.Second):
			t.Errorf("controlled detached worker did not exit during cleanup")
		}
	}()
	waitForTerminalFileContent(t, readyPath, "ready")

	now := time.Now().UTC().Format(time.RFC3339)
	if err := writeTerminalJSONAtomic(terminalProcessPath(root), terminalProcessRecord{
		Version: terminalProcessRecordVersion, PID: cmd.Process.Pid, Identity: identity, State: "running", RunDir: root,
		EventsPath: terminalEventsPath(root), StdoutPath: terminalStdoutPath(root), StderrPath: terminalStderrPath(root), StartedAt: now, UpdatedAt: now,
	}); err != nil {
		t.Fatalf("write controlled process record: %v", err)
	}

	stdout, stderr, err := captureTerminalStreams(t, func() error {
		return handleEvalTerminal("cancel", parseArgs([]string{"eval", "terminal", "cancel", runDir, "--json"}))
	})
	if err != nil {
		t.Fatalf("cancel controlled detached worker: %v", err)
	}
	if stderr != "" {
		t.Fatalf("terminal cancel wrote stderr: %q", stderr)
	}
	var response map[string]any
	if err := json.Unmarshal([]byte(stdout), &response); err != nil {
		t.Fatalf("decode terminal cancel response: %v\n%s", err, stdout)
	}
	if response["state"] != "cancel_requested" || response["force"] != false || response["pid"] != float64(cmd.Process.Pid) {
		t.Fatalf("terminal cancel response = %#v, want cooperative request for PID %d", response, cmd.Process.Pid)
	}
	observed := waitForTerminalFileContent(t, observedPath, "terminal_cancel_requested")
	events := decodeTerminalStatusLines(t, string(observed))
	cancelEvent := events[len(events)-1]
	if cancelEvent["event"] != "terminal_cancel_requested" || cancelEvent["pid"] != float64(cmd.Process.Pid) || cancelEvent["force"] != false {
		t.Fatalf("event visible to worker when signal arrived = %#v, want durable cooperative cancel request", cancelEvent)
	}
	waitForTerminalFileContent(t, signalledPath, "terminated")
	select {
	case <-waitDone:
		if waitErr != nil {
			t.Fatalf("controlled detached worker exit: %v", waitErr)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("controlled detached worker did not exit after cancellation")
	}
	record, err := loadTerminalProcessRecord(root)
	if err != nil {
		t.Fatalf("reload cancelled process record: %v", err)
	}
	if record.State != "cancel_requested" || record.PID != cmd.Process.Pid || record.Identity != identity {
		t.Fatalf("cancelled process record = %+v, want cancel_requested for controlled process identity", record)
	}
}

func TestTerminalCancelRejectsMismatchedProcessIdentityWithoutSignalling(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("robust process identity is exercised on Linux")
	}
	executable, err := os.Executable()
	if err != nil {
		t.Fatalf("locate test executable: %v", err)
	}
	helperDir := t.TempDir()
	readyPath := filepath.Join(helperDir, "ready")
	signalledPath := filepath.Join(helperDir, "signalled")
	cmd := exec.Command(executable, "-test.run=^$")
	cmd.Env = append(os.Environ(),
		terminalJobTestHelperEnv+"=wait-for-termination",
		"LMX_TEST_TERMINAL_READY="+readyPath,
		"LMX_TEST_TERMINAL_SIGNALLED="+signalledPath,
	)
	configureDetachedCommand(cmd)
	if err := cmd.Start(); err != nil {
		t.Fatalf("start controlled detached worker: %v", err)
	}
	identity, err := captureProcessIdentity(cmd.Process.Pid)
	if err != nil {
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
		t.Fatalf("capture controlled worker identity: %v", err)
	}
	waitDone := make(chan struct{})
	go func() {
		_ = cmd.Wait()
		close(waitDone)
	}()
	defer func() {
		if processMatchesIdentity(cmd.Process.Pid, identity) {
			_ = signalDetachedProcess(cmd.Process.Pid, true)
		}
		select {
		case <-waitDone:
		case <-time.After(5 * time.Second):
			t.Errorf("controlled detached worker did not exit during cleanup")
		}
	}()
	waitForTerminalFileContent(t, readyPath, "ready")
	mismatched := identity + ":reused"
	if err := signalTerminalProcess(terminalProcessRecord{PID: cmd.Process.Pid, Identity: mismatched}, false); !errors.Is(err, os.ErrProcessDone) {
		t.Fatalf("signal mismatched process identity error = %v, want os.ErrProcessDone", err)
	}

	runDir := t.TempDir()
	root, err := filepath.Abs(runDir)
	if err != nil {
		t.Fatalf("resolve run directory: %v", err)
	}
	now := time.Now().UTC().Format(time.RFC3339)
	before := terminalProcessRecord{
		Version: terminalProcessRecordVersion, PID: cmd.Process.Pid, Identity: mismatched, State: "running", RunDir: root,
		EventsPath: terminalEventsPath(root), StdoutPath: terminalStdoutPath(root), StderrPath: terminalStderrPath(root), StartedAt: now, UpdatedAt: now,
	}
	if err := writeTerminalJSONAtomic(terminalProcessPath(root), before); err != nil {
		t.Fatalf("write mismatched process record: %v", err)
	}
	stdout, stderr, err := captureTerminalStreams(t, func() error {
		return handleEvalTerminal("cancel", parseArgs([]string{"eval", "terminal", "cancel", runDir, "--json"}))
	})
	if err != nil {
		t.Fatalf("reject mismatched terminal process identity: %v", err)
	}
	if stderr != "" {
		t.Fatalf("mismatched terminal cancel wrote stderr: %q", stderr)
	}
	var response map[string]any
	if err := json.Unmarshal([]byte(stdout), &response); err != nil {
		t.Fatalf("decode mismatched cancel response: %v\n%s", err, stdout)
	}
	if response["alreadyStopped"] != true || response["processRunning"] != false {
		t.Fatalf("mismatched cancel response = %#v, want already-stopped without a live matching process", response)
	}
	after, err := loadTerminalProcessRecord(root)
	if err != nil {
		t.Fatalf("reload mismatched process record: %v", err)
	}
	if !reflect.DeepEqual(after, before) {
		t.Fatalf("mismatched cancel changed process record:\nafter  %+v\nbefore %+v", after, before)
	}
	if _, err := os.Stat(terminalEventsPath(root)); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("mismatched cancel created an event log: %v", err)
	}
	if !processMatchesIdentity(cmd.Process.Pid, identity) {
		t.Fatal("mismatched cancel stopped the live process with the different identity")
	}
	if _, err := os.Stat(signalledPath); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("mismatched cancel reached the helper signal handler: %v", err)
	}
}

func TestRunTerminalEvalRunDirPersistsStatusEventsWhileQuiet(t *testing.T) {
	installFakeTerminalDocker(t, "")
	bundleRoot := t.TempDir()
	writeTerminalRuntimeBundle(t, bundleRoot, "logged-task")
	runDir := t.TempDir()
	server, _ := newTerminalExecutionFixtureServer(t)

	stdout, stderr, err := captureTerminalStreams(t, func() error {
		return runTerminalEval(parseArgs([]string{
			"eval", "terminal", "run",
			"--task-dir", bundleRoot,
			"--base-url", server.URL,
			"--served-model", "fixture-model",
			"--model", "fixture/model",
			"--shell-mode", "stateless",
			"--run-dir", runDir,
			"--json-status", "--quiet",
		}), false)
	})
	if err != nil {
		t.Fatalf("terminal run with durable status log: %v", err)
	}
	if stdout != "" || stderr != "" {
		t.Fatalf("quiet machine run output = stdout %q stderr %q, want both empty", stdout, stderr)
	}
	data, err := os.ReadFile(terminalEventsPath(runDir))
	if err != nil {
		t.Fatalf("read durable terminal event log: %v", err)
	}
	events := decodeTerminalStatusLines(t, string(data))
	wantOrder := []string{"terminal_eval_start", "terminal_task_started", "terminal_task_done", "terminal_eval_completed"}
	next := 0
	for _, event := range events {
		if next < len(wantOrder) && stringValue(event["event"]) == wantOrder[next] {
			if wantOrder[next] == "terminal_task_started" && event["taskId"] != "logged-task" {
				t.Fatalf("persisted task-start event = %#v, want logged-task", event)
			}
			if wantOrder[next] == "terminal_task_done" && (event["taskId"] != "logged-task" || event["scored"] != true || event["pass"] != true) {
				t.Fatalf("persisted task completion = %#v, want scored passing logged-task", event)
			}
			next++
		}
	}
	if next != len(wantOrder) {
		t.Fatalf("durable event order = %#v, want ordered lifecycle %v", events, wantOrder)
	}
	completion := events[len(events)-1]
	if completion["event"] != "terminal_eval_completed" || completion["tasks"] != float64(1) || completion["scored"] != float64(1) {
		t.Fatalf("final durable event = %#v, want one-task scored completion", completion)
	}
}

func TestRunTerminalBundlesCheckpointFailureWithPendingTasksReturns(t *testing.T) {
	root := t.TempDir()
	if err := os.Mkdir(filepath.Join(root, "run.json"), 0o755); err != nil {
		t.Fatalf("create checkpoint write blocker: %v", err)
	}
	checkpoint := &terminalLiveCheckpointStore{
		root: root,
		state: terminalLiveCheckpoint{
			Version: terminalLiveCheckpointVersion,
			State:   "running",
			TaskIDs: []string{"pending-one", "pending-two", "pending-three"},
		},
	}
	bundles := []terminalBundle{
		{Task: terminalTask{ID: "pending-one"}},
		{Task: terminalTask{ID: "pending-two"}},
		{Task: terminalTask{ID: "pending-three"}},
	}
	args := parseArgs([]string{"eval", "terminal", "run", "--run-dir", root, "--quiet"})
	result := make(chan error, 1)
	go func() {
		_, err := runTerminalBundles(context.Background(), args, bundles, "", "", terminalConfig{args: args}, 1, nil, checkpoint)
		result <- err
	}()

	select {
	case err := <-result:
		var cliErr cliError
		if !errorsAsCli(err, &cliErr) || cliErr.Code != "checkpoint_write_failed" {
			t.Fatalf("checkpoint failure = %#v, want checkpoint_write_failed", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("runTerminalBundles deadlocked after the first checkpoint failure with tasks still pending")
	}
}

func TestTerminalManifestAndBundleDownloadsCancelBlockedRequests(t *testing.T) {
	tests := []struct {
		name       string
		blockedURL string
		run        func(context.Context, cliArgs, string) error
	}{
		{
			name:       "manifest download",
			blockedURL: "/blocked-manifest",
			run: func(ctx context.Context, args cliArgs, _ string) error {
				_, _, err := fetchTerminalManifestItems(ctx, args, "fixture-dataset")
				return err
			},
		},
		{
			name:       "bundle download",
			blockedURL: "/blocked-bundle",
			run: func(ctx context.Context, args cliArgs, tmp string) error {
				_, err := downloadTerminalBundle(ctx, args, tmp, "fixture-task", "bundles/fixture.tar.gz", "")
				return err
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			requestStarted := make(chan struct{}, 1)
			releaseHandler := make(chan struct{})
			serverURL := ""
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch {
				case strings.HasSuffix(r.URL.Path, "/shard"):
					_ = json.NewEncoder(w).Encode(map[string]any{"downloadUrl": serverURL + test.blockedURL})
				case r.URL.Path == "/api/benchmarks/storage/download-url":
					_ = json.NewEncoder(w).Encode(map[string]any{"downloadUrl": serverURL + test.blockedURL})
				case r.URL.Path == test.blockedURL:
					select {
					case requestStarted <- struct{}{}:
					default:
					}
					select {
					case <-r.Context().Done():
					case <-releaseHandler:
					}
				default:
					http.NotFound(w, r)
				}
			}))
			serverURL = server.URL
			t.Cleanup(func() {
				close(releaseHandler)
				server.Close()
			})

			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			result := make(chan error, 1)
			args := parseArgs([]string{"eval", "terminal", "run", "--api-url", server.URL})
			tmp := t.TempDir()
			go func() {
				result <- test.run(ctx, args, tmp)
			}()
			select {
			case <-requestStarted:
			case <-time.After(2 * time.Second):
				t.Fatalf("%s never reached the blocked HTTP request", test.name)
			}
			cancel()
			select {
			case err := <-result:
				var cliErr cliError
				if !errorsAsCli(err, &cliErr) || cliErr.Code != "terminal_cancelled" {
					t.Fatalf("%s cancellation error = %#v, want terminal_cancelled", test.name, err)
				}
			case <-time.After(2 * time.Second):
				t.Fatalf("%s did not return promptly after cancellation", test.name)
			}
		})
	}
}

func TestOpenTerminalLiveCheckpointRebuildsStaleProgressFromDurableResults(t *testing.T) {
	runDir := t.TempDir()
	args := parseArgs([]string{
		"eval", "terminal", "run",
		"--run-dir", runDir,
		"--resume", "auto",
		"--api-url", "http://fixture.invalid",
	})
	bundles := []terminalBundle{
		{Task: terminalTask{ID: "scored-task", Instruction: "scored fixture"}},
		{Task: terminalTask{ID: "unscored-task", Instruction: "unscored fixture"}},
	}
	cfg := terminalConfig{args: args, shellMode: "stateless", maxTurns: 1}
	hardware := map[string]any{"gpu": "fixture"}
	store, _, err := openTerminalLiveCheckpoint(args, "fixture-dataset", 1, bundles, "http://model.invalid", "fixture-model", "fixture/model", "", "", hardware, cfg)
	if err != nil {
		t.Fatalf("create terminal checkpoint: %v", err)
	}
	stale := store.state
	store.close()
	stale.CompletedTasks = []string{"ghost-task", "unscored-task"}
	stale.ActiveTasks = []terminalLiveActiveTask{{TaskID: "ghost-active", Index: 99, StartedAt: "2026-07-19T12:00:00Z"}}
	writeTerminalTestJSON(t, filepath.Join(runDir, "run.json"), stale)
	scored, unscored := true, false
	writeTerminalTestJSON(t, terminalLiveResultPath(runDir, "scored-task"), terminalSavedResult{
		QuestionID: "scored-task", Scored: &scored, Pass: true, Response: "durable scored response",
	})
	writeTerminalTestJSON(t, terminalLiveResultPath(runDir, "unscored-task"), terminalSavedResult{
		QuestionID: "unscored-task", Scored: &unscored, ErrorCode: "verifier_failed",
	})

	resumedStore, resumed, err := openTerminalLiveCheckpoint(args, "fixture-dataset", 1, bundles, "http://model.invalid", "fixture-model", "fixture/model", "", "", hardware, cfg)
	if err != nil {
		t.Fatalf("resume terminal checkpoint: %v", err)
	}
	defer resumedStore.close()
	if result, ok := resumed["scored-task"]; !ok || !result.scored || !result.pass || result.transcript != "durable scored response" {
		t.Fatalf("resumed scored result = %+v (found %v), want durable passing result", result, ok)
	}
	if _, ok := resumed["unscored-task"]; ok {
		t.Fatal("unscored durable result was incorrectly marked reusable")
	}
	runData, err := os.ReadFile(filepath.Join(runDir, "run.json"))
	if err != nil {
		t.Fatalf("read rebuilt run checkpoint: %v", err)
	}
	var rebuilt terminalLiveCheckpoint
	if err := json.Unmarshal(runData, &rebuilt); err != nil {
		t.Fatalf("decode rebuilt run checkpoint: %v", err)
	}
	if !reflect.DeepEqual(rebuilt.CompletedTasks, []string{"scored-task", "unscored-task"}) {
		t.Fatalf("rebuilt completed tasks = %v, want exactly the durable task result files", rebuilt.CompletedTasks)
	}
	if len(rebuilt.ActiveTasks) != 0 {
		t.Fatalf("rebuilt active tasks = %+v, want stale in-flight work cleared", rebuilt.ActiveTasks)
	}
}

func TestRunTerminalTaskRemovesContainerBeforeCleaningBuiltImage(t *testing.T) {
	dockerLog := installFakeTerminalDocker(t, "")
	bundleDir := t.TempDir()
	mustMkdir(t, filepath.Join(bundleDir, "tests"))
	mustWrite(t, filepath.Join(bundleDir, "Dockerfile"), "FROM scratch\n")
	mustWrite(t, filepath.Join(bundleDir, "tests", "test.sh"), "fixture verifier")
	server, _ := newTerminalExecutionFixtureServer(t)
	task := terminalTask{
		ID:          "cleanup-order",
		Instruction: "Complete the cleanup-order fixture",
		Image:       terminalImage{Dockerfile: "Dockerfile", Context: "."},
		Agent:       terminalAgentConfig{MaxTurns: 1},
		Verifier: terminalVerifierConfig{
			Command:    "fixture-verifier",
			RewardFile: "/logs/verifier/reward.txt",
		},
	}
	result := runTerminalTask(context.Background(), task, bundleDir, server.URL, "fixture-model", terminalConfig{
		args:            parseArgs([]string{"eval", "terminal", "run", "--quiet"}),
		shellMode:       "stateless",
		cleanupImages:   true,
		agentTimeoutSec: 10,
		endpointTimeout: time.Second,
	})
	if !result.scored || !result.pass {
		t.Fatalf("cleanup-order task result = %+v, want scored pass", result)
	}
	data, err := os.ReadFile(dockerLog)
	if err != nil {
		t.Fatalf("read fake Docker log: %v", err)
	}
	removeIndex, imageRemoveIndex := -1, -1
	commands := strings.Split(strings.TrimSpace(string(data)), "\n")
	for index, command := range commands {
		if strings.HasPrefix(command, "rm -f lmx-tb-cleanup-order-") {
			removeIndex = index
		}
		if strings.HasPrefix(command, "rmi lmx-tb-cleanup-order-") {
			imageRemoveIndex = index
		}
	}
	if removeIndex < 0 || imageRemoveIndex < 0 || removeIndex >= imageRemoveIndex {
		t.Fatalf("Docker cleanup order = %v, want container rm before built image rmi", commands)
	}
}

func TestRunTerminalEvalPersistsScoredTaskBeforeFinalizationAndResumesWithoutExecution(t *testing.T) {
	dockerLog := installFakeTerminalDocker(t, "")
	bundleRoot := t.TempDir()
	writeTerminalRuntimeBundle(t, bundleRoot, "durable-scored-task")
	runDir := t.TempDir()
	server, modelCalls := newTerminalExecutionFixtureServer(t)

	// Force aggregate finalization to fail after the per-task result has been
	// checkpointed, which models interruption between task completion and the
	// final result document becoming durable.
	blockingResultPath := filepath.Join(runDir, "result.json")
	if err := os.Mkdir(blockingResultPath, 0o755); err != nil {
		t.Fatalf("create finalization blocker: %v", err)
	}
	args := []string{
		"eval", "terminal", "run",
		"--task-dir", bundleRoot,
		"--base-url", server.URL,
		"--served-model", "fixture-model",
		"--model", "fixture/model",
		"--shell-mode", "stateless",
		"--run-dir", runDir,
		"--resume", "auto",
		"--json-status", "--quiet",
	}
	if err := runTerminalEval(parseArgs(args), false); err == nil {
		t.Fatal("first run unexpectedly finalized despite result.json blocker")
	}

	savedPath := terminalLiveResultPath(runDir, "durable-scored-task")
	savedData, err := os.ReadFile(savedPath)
	if err != nil {
		t.Fatalf("read task checkpoint after interrupted finalization: %v", err)
	}
	var saved terminalSavedResult
	if err := json.Unmarshal(savedData, &saved); err != nil {
		t.Fatalf("decode task checkpoint after interrupted finalization: %v", err)
	}
	if saved.QuestionID != "durable-scored-task" || saved.Scored == nil || !*saved.Scored || !saved.Pass || !strings.Contains(saved.Response, "TASK_COMPLETE") {
		t.Fatalf("durable task checkpoint = %+v, want complete scored task result", saved)
	}
	var interrupted terminalLiveCheckpoint
	runData, err := os.ReadFile(filepath.Join(runDir, "run.json"))
	if err != nil {
		t.Fatalf("read interrupted run checkpoint: %v", err)
	}
	if err := json.Unmarshal(runData, &interrupted); err != nil {
		t.Fatalf("decode interrupted run checkpoint: %v", err)
	}
	if interrupted.State != "running" || !reflect.DeepEqual(interrupted.CompletedTasks, []string{"durable-scored-task"}) {
		t.Fatalf("interrupted run state = %+v, want running with durable completed task", interrupted)
	}
	dockerBeforeResume, err := os.ReadFile(dockerLog)
	if err != nil {
		t.Fatalf("read Docker log before resume: %v", err)
	}
	modelCallsBeforeResume := modelCalls.Load()

	if err := os.Remove(blockingResultPath); err != nil {
		t.Fatalf("remove finalization blocker: %v", err)
	}
	if err := runTerminalEval(parseArgs(args), false); err != nil {
		t.Fatalf("resume interrupted terminal run: %v", err)
	}
	if got := modelCalls.Load(); got != modelCallsBeforeResume {
		t.Fatalf("model calls after scored resume = %d, want unchanged %d", got, modelCallsBeforeResume)
	}
	dockerAfterResume, err := os.ReadFile(dockerLog)
	if err != nil {
		t.Fatalf("read Docker log after resume: %v", err)
	}
	appendedDocker := strings.TrimSpace(strings.TrimPrefix(string(dockerAfterResume), string(dockerBeforeResume)))
	if !strings.HasPrefix(appendedDocker, "ps -aq --filter label=localmaxxing.run=") || strings.Contains(appendedDocker, "\n") {
		t.Fatalf("scored resume should only verify cleanup, appended Docker commands:\n%s", appendedDocker)
	}
	final := readTerminalSubmitBatch(t, blockingResultPath)
	if summary := asObject(final["summary"]); summary["scored"] != float64(1) || summary["correct"] != float64(1) {
		t.Fatalf("resumed final summary = %#v, want one scored passing task", summary)
	}
}

func TestRunTerminalEvalResumeRerunsUnscoredTaskResult(t *testing.T) {
	dockerLog := installFakeTerminalDocker(t, "retry-unscored-task")
	bundleRoot := t.TempDir()
	writeTerminalRuntimeBundle(t, bundleRoot, "retry-unscored-task")
	runDir := t.TempDir()
	server, modelCalls := newTerminalExecutionFixtureServer(t)
	args := []string{
		"eval", "terminal", "run",
		"--task-dir", bundleRoot,
		"--base-url", server.URL,
		"--served-model", "fixture-model",
		"--model", "fixture/model",
		"--shell-mode", "stateless",
		"--run-dir", runDir,
		"--resume", "auto",
		"--json-status", "--quiet",
	}
	if err := runTerminalEval(parseArgs(args), false); err == nil {
		t.Fatal("first run with verifier failure unexpectedly succeeded")
	}
	first, ok, err := loadTerminalLiveResult(runDir, "retry-unscored-task")
	if err != nil {
		t.Fatalf("load unscored task checkpoint: %v", err)
	}
	if !ok || first.scored {
		t.Fatalf("first checkpoint = %+v (found %v), want persisted unscored result", first, ok)
	}
	dockerBeforeRetry, err := os.ReadFile(dockerLog)
	if err != nil {
		t.Fatalf("read Docker log before retry: %v", err)
	}
	modelCallsBeforeRetry := modelCalls.Load()

	t.Setenv("LMX_TEST_FAIL_VERIFIER_TASK", "")
	if err := runTerminalEval(parseArgs(args), false); err != nil {
		t.Fatalf("resume with repaired verifier: %v", err)
	}
	if got := modelCalls.Load(); got != modelCallsBeforeRetry+1 {
		t.Fatalf("model calls after unscored resume = %d, want exactly one rerun after %d calls", got, modelCallsBeforeRetry)
	}
	dockerAfterRetry, err := os.ReadFile(dockerLog)
	if err != nil {
		t.Fatalf("read Docker log after retry: %v", err)
	}
	if !bytes.HasPrefix(dockerAfterRetry, dockerBeforeRetry) {
		t.Fatalf("resume rewrote the Docker log; before:\n%s\nafter:\n%s", dockerBeforeRetry, dockerAfterRetry)
	}
	root, err := filepath.Abs(runDir)
	if err != nil {
		t.Fatalf("resolve resumed run directory: %v", err)
	}
	sum := sha256.Sum256([]byte(root))
	wantCleanup := "ps -aq --filter label=localmaxxing.run=" + hex.EncodeToString(sum[:])[:12]
	cleanupCalls := 0
	for _, command := range strings.Split(strings.TrimSpace(string(dockerAfterRetry[len(dockerBeforeRetry):])), "\n") {
		if command == wantCleanup {
			cleanupCalls++
		}
	}
	if cleanupCalls != 2 {
		t.Fatalf("resume cleanup calls = %d, want start and completion checks %q in appended Docker commands:\n%s", cleanupCalls, wantCleanup, dockerAfterRetry[len(dockerBeforeRetry):])
	}
	if bytes.Equal(dockerAfterRetry, dockerBeforeRetry) {
		t.Fatal("unscored resume did not execute the task again")
	}
	retried, ok, err := loadTerminalLiveResult(runDir, "retry-unscored-task")
	if err != nil {
		t.Fatalf("load retried task checkpoint: %v", err)
	}
	if !ok || !retried.scored || !retried.pass {
		t.Fatalf("retried checkpoint = %+v (found %v), want scored passing replacement", retried, ok)
	}
}

func TestRunTerminalEvalResumeRerunsExternalFailureButReusesScoredVerifierFailure(t *testing.T) {
	installFakeTerminalDocker(t, "")
	t.Setenv("LMX_TEST_VERIFIER_REWARD", "0")
	bundleRoot := t.TempDir()
	writeTerminalRuntimeBundle(t, bundleRoot, "external-failure")
	writeTerminalRuntimeBundle(t, bundleRoot, "scored-verifier-failure")
	runDir := t.TempDir()
	agentLog := filepath.Join(t.TempDir(), "agent.log")
	failureMarker := filepath.Join(t.TempDir(), "external-failed-once")
	agentCommand := "printf '%s\\n' \"$LMX_TERMINAL_TASK_ID\" >> " + shellQuote(agentLog) + "; " +
		"if [ \"$LMX_TERMINAL_TASK_ID\" = external-failure ] && [ ! -e " + shellQuote(failureMarker) + " ]; then " +
		"touch " + shellQuote(failureMarker) + "; exit 17; fi"
	args := parseArgs([]string{
		"eval", "terminal", "run",
		"--task-dir", bundleRoot,
		"--agent-cmd", agentCommand,
		"--run-dir", runDir,
		"--resume", "auto",
		"--quiet",
	})

	if err := runTerminalEval(args, false); err != nil {
		t.Fatalf("initial mixed external run: %v", err)
	}
	external, ok, err := loadTerminalLiveResult(runDir, "external-failure")
	if err != nil {
		t.Fatalf("load external failure: %v", err)
	}
	if !ok || external.scored || external.errCode != "command_exec_failed" {
		t.Fatalf("external failure checkpoint = %+v (found %v), want unscored command_exec_failed", external, ok)
	}
	verifier, ok, err := loadTerminalLiveResult(runDir, "scored-verifier-failure")
	if err != nil {
		t.Fatalf("load scored verifier failure: %v", err)
	}
	if !ok || !verifier.scored || verifier.pass || verifier.errCode != "" {
		t.Fatalf("verifier failure checkpoint = %+v (found %v), want scored benchmark failure", verifier, ok)
	}
	firstExecutions := readTerminalAgentExecutions(t, agentLog)
	if firstExecutions["external-failure"] != 1 || firstExecutions["scored-verifier-failure"] != 1 {
		t.Fatalf("initial external executions = %#v, want each task once", firstExecutions)
	}

	if err := runTerminalEval(args, false); err != nil {
		t.Fatalf("resume mixed external run: %v", err)
	}
	resumedExecutions := readTerminalAgentExecutions(t, agentLog)
	if resumedExecutions["external-failure"] != 2 {
		t.Fatalf("external failure executions after resume = %#v, want failed task rerun once", resumedExecutions)
	}
	if resumedExecutions["scored-verifier-failure"] != 1 {
		t.Fatalf("scored verifier failure executions after resume = %#v, want scored task reused", resumedExecutions)
	}
	external, ok, err = loadTerminalLiveResult(runDir, "external-failure")
	if err != nil {
		t.Fatalf("load rerun external task: %v", err)
	}
	if !ok || !external.scored || external.pass || external.errCode != "" {
		t.Fatalf("rerun external checkpoint = %+v (found %v), want scored verifier failure replacement", external, ok)
	}
	final := readTerminalSubmitBatch(t, filepath.Join(runDir, "result.json"))
	results := anySlice(final["results"])
	if len(results) != 2 {
		t.Fatalf("resumed external results = %#v, want two task records", results)
	}
	for _, raw := range results {
		record := asObject(raw)
		if turns, present := record["turns"]; !present || turns != nil {
			t.Fatalf("external result turns = %#v (present %v) for %q, want explicit null", turns, present, record["question_id"])
		}
	}
}

func readTerminalAgentExecutions(t *testing.T, path string) map[string]int {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read external agent execution log: %v", err)
	}
	counts := map[string]int{}
	for _, taskID := range strings.Fields(string(data)) {
		counts[taskID]++
	}
	return counts
}

func TestRunTerminalEvalResumeRejectsChangedManifestOrExecutionIdentity(t *testing.T) {
	installFakeTerminalDocker(t, "")
	bundleRoot := t.TempDir()
	bundleDir := writeTerminalRuntimeBundle(t, bundleRoot, "identity-task")
	runDir := t.TempDir()
	server, modelCalls := newTerminalExecutionFixtureServer(t)
	baseArgs := []string{
		"eval", "terminal", "run",
		"--task-dir", bundleRoot,
		"--base-url", server.URL,
		"--served-model", "fixture-model",
		"--model", "fixture/model",
		"--shell-mode", "persistent",
		"--run-dir", runDir,
		"--resume", "auto",
		"--json-status", "--quiet",
	}
	if err := runTerminalEval(parseArgs(baseArgs), false); err != nil {
		t.Fatalf("create resumable terminal checkpoint: %v", err)
	}
	checkpointData, err := os.ReadFile(filepath.Join(runDir, "run.json"))
	if err != nil {
		t.Fatal(err)
	}
	var checkpoint terminalLiveCheckpoint
	if err := json.Unmarshal(checkpointData, &checkpoint); err != nil {
		t.Fatal(err)
	}
	if len(checkpoint.FingerprintFields) == 0 {
		t.Fatalf("checkpoint omitted fingerprint field hashes")
	}
	callsAfterInitialRun := modelCalls.Load()

	taskPath := filepath.Join(bundleDir, "task.json")
	taskData, err := os.ReadFile(taskPath)
	if err != nil {
		t.Fatalf("read original task manifest: %v", err)
	}
	var task terminalTask
	if err := json.Unmarshal(taskData, &task); err != nil {
		t.Fatalf("decode original task manifest: %v", err)
	}
	task.Instruction = "Changed instruction must invalidate the checkpoint"
	writeTerminalTestJSON(t, taskPath, task)
	assertTerminalCheckpointMismatch(t, parseArgs(baseArgs), "changed manifest")
	if got := modelCalls.Load(); got != callsAfterInitialRun {
		t.Fatalf("changed-manifest resume executed model: calls = %d, want %d", got, callsAfterInitialRun)
	}

	if err := os.WriteFile(taskPath, taskData, 0o644); err != nil {
		t.Fatalf("restore original task manifest: %v", err)
	}
	changedIdentityArgs := append(append([]string(nil), baseArgs...), "--shell-mode", "stateless")
	assertTerminalCheckpointMismatch(t, parseArgs(changedIdentityArgs), "changed execution identity")
	if got := modelCalls.Load(); got != callsAfterInitialRun {
		t.Fatalf("changed-identity resume executed model: calls = %d, want %d", got, callsAfterInitialRun)
	}
}

func TestRunTerminalEvalResumeSelectorValidation(t *testing.T) {
	bundleRoot := t.TempDir()
	writeTerminalRuntimeBundle(t, bundleRoot, "resume-selector-task")
	tests := []struct {
		name        string
		args        []string
		wantMessage string
	}{
		{
			name: "resume requires run directory",
			args: []string{
				"eval", "terminal", "run", "--task-dir", bundleRoot, "--oracle", "--resume", "auto",
			},
			wantMessage: "--resume requires --run-dir",
		},
		{
			name: "selector is limited to auto or none",
			args: []string{
				"eval", "terminal", "run", "--task-dir", bundleRoot, "--oracle", "--run-dir", t.TempDir(), "--resume", "latest",
			},
			wantMessage: "--resume must be auto or none",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := runTerminalEval(parseArgs(tc.args), false)
			var cliErr cliError
			if !errors.As(err, &cliErr) || cliErr.Code != "invalid_option" {
				t.Fatalf("error = %#v, want invalid_option", err)
			}
			if !strings.Contains(cliErr.Message, tc.wantMessage) {
				t.Fatalf("message = %q, want clear validation containing %q", cliErr.Message, tc.wantMessage)
			}
		})
	}
}

func TestRunTerminalEvalJSONStatusSeparatesStreamsAndEmitsStructuredCompletion(t *testing.T) {
	installFakeTerminalDocker(t, "")
	server, _ := newTerminalExecutionFixtureServer(t)
	tests := []struct {
		name            string
		submit          bool
		jsonOutput      bool
		wantFinalEvent  string
		wantRunIdentity bool
	}{
		{name: "local run leaves stdout empty", wantFinalEvent: "terminal_eval_completed"},
		{name: "local JSON is one document", jsonOutput: true, wantFinalEvent: "terminal_eval_completed"},
		{name: "submitted JSON is one document", submit: true, jsonOutput: true, wantFinalEvent: "terminal_eval_submitted", wantRunIdentity: true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			bundleRoot := t.TempDir()
			writeTerminalRuntimeBundle(t, bundleRoot, "status-task")
			args := []string{
				"eval", "terminal", "run",
				"--task-dir", bundleRoot,
				"--base-url", server.URL,
				"--served-model", "fixture-model",
				"--model", "fixture/model",
				"--shell-mode", "stateless",
				"--json-status",
				"--run-dir", filepath.Join(t.TempDir(), "run"),
			}
			if tc.jsonOutput {
				args = append(args, "--json")
			}
			if tc.submit {
				hardwarePath := filepath.Join(t.TempDir(), "hardware.json")
				writeTerminalTestJSON(t, hardwarePath, map[string]any{"hwClass": "CPU_ONLY", "cpu": "Fixture CPU", "ramGb": 8})
				args = append(args,
					"--dataset", "terminal-status-fixture",
					"--quantization", "Q4_K_M",
					"--quant-format", "gguf",
					"--hardware", hardwarePath,
					"--api-url", server.URL,
					"--api-key", "fixture-key",
					"--submit",
				)
			}

			stdout, stderr, err := captureTerminalStreams(t, func() error {
				return runTerminalEval(parseArgs(args), false)
			})
			if err != nil {
				t.Fatalf("terminal JSON-status run: %v", err)
			}
			statuses := decodeTerminalStatusLines(t, stderr)
			seen := map[string]bool{}
			for _, status := range statuses {
				seen[stringValue(status["event"])] = true
			}
			for _, event := range []string{"terminal_eval_start", "terminal_task_started", "terminal_task_done", "terminal_cleanup_verified"} {
				if !seen[event] {
					t.Fatalf("stderr events omitted %q: %#v", event, statuses)
				}
			}
			completion := statuses[len(statuses)-1]
			if completion["event"] != tc.wantFinalEvent {
				t.Fatalf("final stderr event = %#v, want %q", completion, tc.wantFinalEvent)
			}
			if tc.submit {
				if completion["submitted"] != float64(1) || completion["runId"] != "terminal-status-run" {
					t.Fatalf("submitted completion = %#v, want submitted count and run identity", completion)
				}
			} else if completion["submitted"] != false || completion["tasks"] != float64(1) || completion["scored"] != float64(1) {
				t.Fatalf("local completion = %#v, want structured one-task scored summary", completion)
			}

			if !tc.jsonOutput {
				if stdout != "" {
					t.Fatalf("stdout without --json = %q, want empty machine-output stream", stdout)
				}
				return
			}
			var document map[string]any
			if err := json.Unmarshal([]byte(stdout), &document); err != nil {
				t.Fatalf("stdout is not exactly one JSON document: %v\n%s", err, stdout)
			}
			summary := asObject(document["summary"])
			if summary["tasks"] != float64(1) || summary["scored"] != float64(1) {
				t.Fatalf("stdout result summary = %#v, want one scored task", summary)
			}
			if tc.wantRunIdentity {
				run := asObject(asObject(document["submission"])["run"])
				if run["id"] != "terminal-status-run" || run["status"] != "APPROVED" {
					t.Fatalf("stdout submission receipt = %#v, want approved terminal-status-run", document["submission"])
				}
			}
		})
	}
}

func assertTerminalCheckpointMismatch(t *testing.T, args cliArgs, changed string) {
	t.Helper()
	err := runTerminalEval(args, false)
	var cliErr cliError
	if !errors.As(err, &cliErr) || cliErr.Code != "checkpoint_mismatch" {
		t.Fatalf("%s error = %#v, want checkpoint_mismatch", changed, err)
	}
	if !strings.Contains(cliErr.Message, "does not match") {
		t.Fatalf("%s message = %q, want clear mismatch explanation", changed, cliErr.Message)
	}
	details := asObject(cliErr.Details)
	changedFields, ok := details["changedFields"].([]string)
	if !ok || len(changedFields) == 0 {
		t.Fatalf("%s mismatch details = %#v, want changedFields", changed, cliErr.Details)
	}
}

func TestTerminalDockerHintExplainsLinuxGroupAccess(t *testing.T) {
	hint := terminalHint("docker_unavailable")
	if !strings.Contains(hint, "usermod -aG docker") || !strings.Contains(hint, "new login session") {
		t.Fatalf("Docker hint = %q, want group repair and session refresh", hint)
	}
}

func newTerminalExecutionFixtureServer(t *testing.T) (*httptest.Server, *atomic.Int32) {
	t.Helper()
	var modelCalls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.URL.Path == "/v1/models":
			_ = json.NewEncoder(w).Encode(map[string]any{"data": []any{map[string]any{"id": "fixture-model"}}})
		case r.URL.Path == "/v1/chat/completions":
			modelCalls.Add(1)
			_ = json.NewEncoder(w).Encode(map[string]any{"choices": []any{map[string]any{"message": map[string]any{"content": "TASK_COMPLETE"}}}})
		case strings.HasSuffix(r.URL.Path, "/submit"):
			w.WriteHeader(http.StatusCreated)
			_ = json.NewEncoder(w).Encode(map[string]any{
				"run":       map[string]any{"id": "terminal-status-run", "status": "APPROVED"},
				"aggregate": map[string]any{"pooledScore": 1.0, "ciLower": 0.25, "ciUpper": 1.0, "shardsCovered": []any{1}},
			})
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(server.Close)
	return server, &modelCalls
}

func captureTerminalStreams(t *testing.T, run func() error) (string, string, error) {
	t.Helper()
	stdoutFile, err := os.Create(filepath.Join(t.TempDir(), "stdout"))
	if err != nil {
		t.Fatalf("create stdout capture: %v", err)
	}
	stderrFile, err := os.Create(filepath.Join(t.TempDir(), "stderr"))
	if err != nil {
		_ = stdoutFile.Close()
		t.Fatalf("create stderr capture: %v", err)
	}
	originalStdout, originalStderr := os.Stdout, os.Stderr
	os.Stdout, os.Stderr = stdoutFile, stderrFile
	defer func() {
		os.Stdout, os.Stderr = originalStdout, originalStderr
		_ = stdoutFile.Close()
		_ = stderrFile.Close()
	}()

	runErr := run()
	os.Stdout, os.Stderr = originalStdout, originalStderr
	if err := stdoutFile.Close(); err != nil {
		t.Fatalf("close stdout capture: %v", err)
	}
	if err := stderrFile.Close(); err != nil {
		t.Fatalf("close stderr capture: %v", err)
	}
	stdout, err := os.ReadFile(stdoutFile.Name())
	if err != nil {
		t.Fatalf("read stdout capture: %v", err)
	}
	stderr, err := os.ReadFile(stderrFile.Name())
	if err != nil {
		t.Fatalf("read stderr capture: %v", err)
	}
	return string(stdout), string(stderr), runErr
}

func decodeTerminalStatusLines(t *testing.T, stderr string) []map[string]any {
	t.Helper()
	trimmed := strings.TrimSpace(stderr)
	if trimmed == "" {
		t.Fatal("JSON-status stderr was empty")
	}
	lines := strings.Split(trimmed, "\n")
	statuses := make([]map[string]any, 0, len(lines))
	for index, line := range lines {
		var status map[string]any
		if err := json.Unmarshal([]byte(line), &status); err != nil {
			t.Fatalf("stderr line %d is not JSON: %v\n%s", index+1, err, line)
		}
		if stringValue(status["event"]) == "" {
			t.Fatalf("stderr line %d has no lifecycle event: %#v", index+1, status)
		}
		statuses = append(statuses, status)
	}
	return statuses
}

func waitForTerminalFileContent(t *testing.T, path, want string) []byte {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	ticker := time.NewTicker(5 * time.Millisecond)
	defer ticker.Stop()
	for {
		data, err := os.ReadFile(path)
		if err == nil && strings.Contains(string(data), want) {
			return data
		}
		if err != nil && !os.IsNotExist(err) {
			t.Fatalf("read polled file %s: %v", path, err)
		}
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %q in %s; last content %q", want, path, data)
		}
		<-ticker.C
	}
}

func writeTerminalRuntimeBundle(t *testing.T, root, id string) string {
	t.Helper()
	dir := filepath.Join(root, id)
	mustMkdir(t, filepath.Join(dir, "tests"))
	writeTerminalTestJSON(t, filepath.Join(dir, "task.json"), terminalTask{
		ID: id, Instruction: "Complete " + id, Image: terminalImage{Prebuilt: "fake:image"},
		Agent:    terminalAgentConfig{MaxTurns: 1},
		Verifier: terminalVerifierConfig{Command: "fixture-verifier", RewardFile: "/logs/verifier/reward.txt"},
	})
	mustWrite(t, filepath.Join(dir, "tests", "test.sh"), "fixture verifier")
	return dir
}

func installFakeTerminalDocker(t *testing.T, failVerifierTask string) string {
	t.Helper()
	binDir := t.TempDir()
	logPath := filepath.Join(t.TempDir(), "docker.log")
	script := `#!/bin/sh
printf '%s\n' "$*" >> "$LMX_TEST_DOCKER_LOG"
if [ -n "$LMX_TEST_FAIL_VERIFIER_TASK" ]; then
  case "$*" in
    *"$LMX_TEST_FAIL_VERIFIER_TASK/tests/."*) exit 1 ;;
  esac
fi
case "$*" in
  *"cat /logs/verifier/reward.json"*) exit 1 ;;
  *"cat /logs/verifier/reward.txt"*) printf '%s\n' "${LMX_TEST_VERIFIER_REWARD:-1}"; exit 0 ;;
esac
exit 0
`
	mustWrite(t, filepath.Join(binDir, "docker"), script)
	if err := os.Chmod(filepath.Join(binDir, "docker"), 0o755); err != nil {
		t.Fatalf("make fake Docker executable: %v", err)
	}
	t.Setenv("LMX_TEST_DOCKER_LOG", logPath)
	t.Setenv("LMX_TEST_FAIL_VERIFIER_TASK", failVerifierTask)
	t.Setenv("PATH", binDir+string(os.PathListSeparator)+os.Getenv("PATH"))
	return logPath
}

func terminalTestBundleArchive(t *testing.T, id string) []byte {
	t.Helper()
	var compressed bytes.Buffer
	gz := gzip.NewWriter(&compressed)
	tw := tar.NewWriter(gz)
	taskData, err := json.Marshal(terminalTask{
		ID: id, Instruction: "Complete " + id, Image: terminalImage{Prebuilt: "fake:image"},
		Agent:    terminalAgentConfig{MaxTurns: 1},
		Verifier: terminalVerifierConfig{Command: "fixture-verifier", RewardFile: "/logs/verifier/reward.txt"},
	})
	if err != nil {
		t.Fatalf("encode archive task: %v", err)
	}
	entries := []struct {
		name string
		mode int64
		body []byte
	}{
		{name: "task.json", mode: 0o644, body: taskData},
		{name: "tests/", mode: 0o755},
		{name: "tests/test.sh", mode: 0o755, body: []byte("#!/bin/sh\nexit 0\n")},
	}
	for _, entry := range entries {
		typeFlag := byte(tar.TypeReg)
		if strings.HasSuffix(entry.name, "/") {
			typeFlag = tar.TypeDir
		}
		if err := tw.WriteHeader(&tar.Header{Name: entry.name, Mode: entry.mode, Size: int64(len(entry.body)), Typeflag: typeFlag}); err != nil {
			t.Fatalf("write archive header %q: %v", entry.name, err)
		}
		if len(entry.body) > 0 {
			if _, err := tw.Write(entry.body); err != nil {
				t.Fatalf("write archive entry %q: %v", entry.name, err)
			}
		}
	}
	if err := tw.Close(); err != nil {
		t.Fatalf("close tar fixture: %v", err)
	}
	if err := gz.Close(); err != nil {
		t.Fatalf("close gzip fixture: %v", err)
	}
	return compressed.Bytes()
}

func terminalFixtureServerURL(r *http.Request) string {
	return "http://" + r.Host
}

func numberFromString(value string) int {
	if value == "2" {
		return 2
	}
	if value == "3" {
		return 3
	}
	return 1
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

func TestSubmitCRUDbenchCanonicalCheckpointDryRunPartitionsExactTaskSet(t *testing.T) {
	canonicalIDs := append([]string(nil), crudBenchCanonicalTaskIDs...)
	runDir, hardwarePath := writeTerminalCheckpointSetFixture(t, canonicalIDs, true)
	payloadPath := filepath.Join(t.TempDir(), "crud-bench-batch.json")

	err := submitTerminalEval(parseArgs([]string{
		"eval", "terminal", "submit", runDir,
		"--dataset", crudBenchDataset,
		"--hf-id", "fixture/model",
		"--hardware", hardwarePath,
		"--out", payloadPath,
		"--dry-run", "--quiet",
	}))
	if err != nil {
		t.Fatalf("submit CRUD-Bench dry-run: %v", err)
	}

	batch := readTerminalSubmitBatch(t, payloadPath)
	if batch["dataset"] != crudBenchDataset {
		t.Fatalf("batch dataset = %#v, want %q", batch["dataset"], crudBenchDataset)
	}
	shards := anySlice(batch["shards"])
	if len(shards) != crudBenchShardCount {
		t.Fatalf("batch contains %d shards, want %d", len(shards), crudBenchShardCount)
	}
	offset := 0
	for i, rawShard := range shards {
		wantIDs := canonicalIDs[offset : offset+8]
		assertTerminalShardPayload(t, asObject(rawShard), i+1, wantIDs)
		offset += 8
	}
	if offset != len(canonicalIDs) {
		t.Fatalf("partitioned task total = %d, want %d", offset, len(canonicalIDs))
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

func TestSubmitCRUDbenchExplicitCanonicalShardWritesIsolatedCheckpointPayload(t *testing.T) {
	wantIDs := crudBenchCanonicalTaskIDs[:8]
	runDir, hardwarePath := writeTerminalCheckpointSetFixture(t, wantIDs, true)
	payloadPath := filepath.Join(t.TempDir(), "crud-bench-shard.json")
	err := submitTerminalEval(parseArgs([]string{
		"eval", "terminal", "submit", runDir,
		"--dataset", crudBenchDataset,
		"--shard-index", "1",
		"--hf-id", "fixture/model",
		"--hardware", hardwarePath,
		"--out", payloadPath,
		"--dry-run", "--quiet",
	}))
	if err != nil {
		t.Fatalf("submit CRUD-Bench shard: %v", err)
	}
	shards := anySlice(readTerminalSubmitBatch(t, payloadPath)["shards"])
	if len(shards) != 1 {
		t.Fatalf("explicit checkpoint payload count = %d, want one", len(shards))
	}
	assertTerminalShardPayload(t, asObject(shards[0]), 1, wantIDs)
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

func TestSubmitTerminalEvalPostsCanonicalShardsSequentiallyAndStopsAtFailure(t *testing.T) {
	canonicalIDs := terminalBench21CanonicalTestTaskIDs(t)
	runDir, hardwarePath := writeTerminalCheckpointSetFixture(t, canonicalIDs, true)
	var requestCount atomic.Int32
	var received [terminalBench21ShardCount]atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		call := int(requestCount.Add(1)) - 1
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
		if r.Method != http.MethodPost {
			t.Errorf("request %d method = %s, want POST", call+1, r.Method)
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
