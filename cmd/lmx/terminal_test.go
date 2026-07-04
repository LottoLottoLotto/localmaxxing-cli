package main

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
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

func TestExtractBashCommand(t *testing.T) {
	cases := []struct {
		name  string
		reply string
		want  string
		found bool
	}{
		{"bash", "thinking\n```bash\necho hi\n```", "echo hi", true},
		{"generic", "```\npwd\n```", "pwd", true},
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
