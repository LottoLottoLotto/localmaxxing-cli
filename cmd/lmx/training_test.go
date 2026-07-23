package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestPrepareEvalTrainingDataExportsPassingFinalizedTrajectory(t *testing.T) {
	root, out := writeTrainingFixture(t)
	args := cliArgs{
		positional: []string{"eval", "train", "prepare", root},
		opts: map[string]string{
			"out":               out,
			"base-model":        "org/base-model",
			"max-message-bytes": "1024",
		},
		flags: map[string]bool{"allow-benchmark-training": true},
	}
	if err := prepareEvalTrainingData(args); err != nil {
		t.Fatalf("prepareEvalTrainingData returned error: %v", err)
	}

	sft := readJSONLRows(t, filepath.Join(out, "sft.jsonl"))
	if len(sft) != 1 {
		t.Fatalf("SFT rows = %d, want one passing trajectory", len(sft))
	}
	if got := stringValue(sft[0]["id"]); got != "pass-task" {
		t.Fatalf("SFT id = %q, want pass-task", got)
	}
	encoded, err := json.Marshal(sft[0])
	if err != nil {
		t.Fatal(err)
	}
	text := string(encoded)
	if strings.Contains(text, "STREAMING_DELTA_MUST_NOT_APPEAR") {
		t.Fatal("SFT row included a cumulative message_update")
	}
	if strings.Contains(text, "HIDDEN_REASONING_MUST_NOT_APPEAR") {
		t.Fatal("SFT row included assistant thinking")
	}
	if !strings.Contains(text, "LMX_TRAINING_DATA_TRUNCATED") {
		t.Fatal("oversized tool output was not explicitly truncated")
	}
	if !strings.Contains(text, `"tool_calls"`) || !strings.Contains(text, `"role":"tool"`) {
		t.Fatal("SFT row did not preserve tool call and tool result messages")
	}

	failures := readJSONLRows(t, filepath.Join(out, "failures.jsonl"))
	if len(failures) != 1 || stringValue(failures[0]["id"]) != "fail-task" {
		t.Fatalf("failure diagnostics = %#v, want only fail-task", failures)
	}
	if stringValue(failures[0]["trainingUse"]) != "diagnostic_only" {
		t.Fatalf("failed trajectory trainingUse = %q", stringValue(failures[0]["trainingUse"]))
	}

	manifestValue, err := readJSON(filepath.Join(out, "manifest.json"))
	if err != nil {
		t.Fatal(err)
	}
	manifest := asObject(manifestValue)
	results := asObject(manifest["results"])
	if numberField(results, "passed") != 1 || numberField(results, "failed") != 1 {
		t.Fatalf("manifest pass/fail = %#v", results)
	}
	if numberField(results, "preferencePairs") != 0 {
		t.Fatalf("preferencePairs = %v, want zero", results["preferencePairs"])
	}
	if numberField(results, "truncatedFields") < 1 {
		t.Fatalf("truncatedFields = %v, want at least one", results["truncatedFields"])
	}
}

func TestPrepareEvalTrainingDataRequiresContaminationAcknowledgement(t *testing.T) {
	root, out := writeTrainingFixture(t)
	err := prepareEvalTrainingData(cliArgs{
		positional: []string{"eval", "train", "prepare", root},
		opts:       map[string]string{"out": out, "base-model": "org/base-model"},
		flags:      map[string]bool{},
	})
	var cliErr cliError
	if !errors.As(err, &cliErr) || cliErr.Code != "benchmark_training_not_acknowledged" {
		t.Fatalf("error = %#v, want benchmark_training_not_acknowledged", err)
	}
	if _, statErr := os.Stat(filepath.Join(out, "sft.jsonl")); !os.IsNotExist(statErr) {
		t.Fatalf("guarded preparation wrote SFT data: %v", statErr)
	}
}

func TestRunEvalTrainerDoesNotExecuteByDefault(t *testing.T) {
	root, out := writeTrainingFixture(t)
	prepareArgs := cliArgs{
		positional: []string{"eval", "train", "prepare", root},
		opts:       map[string]string{"out": out, "base-model": "org/base-model"},
		flags:      map[string]bool{"allow-benchmark-training": true, "quiet": true},
	}
	if err := prepareEvalTrainingData(prepareArgs); err != nil {
		t.Fatal(err)
	}
	sentinel := filepath.Join(t.TempDir(), "trainer-executed")
	command := "touch " + sentinel + " {dataset} {manifest} {output} {base_model}"
	err := runEvalTrainer(cliArgs{
		positional: []string{"eval", "train", "run", filepath.Join(out, "manifest.json")},
		opts:       map[string]string{"trainer-cmd": command},
		flags:      map[string]bool{},
	})
	if err != nil {
		t.Fatalf("runEvalTrainer returned error: %v", err)
	}
	if _, statErr := os.Stat(sentinel); !os.IsNotExist(statErr) {
		t.Fatalf("trainer command executed without --execute: %v", statErr)
	}
}

func writeTrainingFixture(t *testing.T) (string, string) {
	t.Helper()
	root := t.TempDir()
	out := filepath.Join(t.TempDir(), "training")
	passResult := filepath.Join(root, "pass-task.json")
	failResult := filepath.Join(root, "fail-task.json")
	writeTestJSON(t, passResult, map[string]any{"results": []any{map[string]any{
		"question_id": "pass-task", "pass": true, "scored": true,
		"question": "solve the passing task", "verifierOutput": "reward 1", "wallTimeMs": 10,
	}}})
	writeTestJSON(t, failResult, map[string]any{"results": []any{map[string]any{
		"question_id": "fail-task", "pass": false, "scored": true,
		"question": "solve the failing task", "verifierOutput": "assertion failed", "wallTimeMs": 20,
	}}})
	writeTestJSON(t, filepath.Join(root, "summary.json"), []any{
		map[string]any{"task": "pass-task", "out": passResult},
		map[string]any{"task": "fail-task", "out": failResult},
	})

	trace := filepath.Join(root, "traces", "pass-task", "pass-task", "agent", "session", "omp.jsonl")
	if err := os.MkdirAll(filepath.Dir(trace), 0o755); err != nil {
		t.Fatal(err)
	}
	events := []any{
		map[string]any{"type": "message_update", "message": map[string]any{"role": "assistant", "content": []any{map[string]any{"type": "text", "text": "STREAMING_DELTA_MUST_NOT_APPEAR"}}}},
		map[string]any{"type": "message_end", "message": map[string]any{"role": "user", "content": []any{map[string]any{"type": "text", "text": "solve the passing task"}}}},
		map[string]any{"type": "message_end", "message": map[string]any{"role": "assistant", "content": []any{
			map[string]any{"type": "thinking", "thinking": "HIDDEN_REASONING_MUST_NOT_APPEAR"},
			map[string]any{"type": "toolCall", "id": "call-1", "name": "bash", "arguments": map[string]any{"command": "printf ok"}},
		}}},
		map[string]any{"type": "message_end", "message": map[string]any{"role": "toolResult", "toolCallId": "call-1", "toolName": "bash", "content": []any{map[string]any{"type": "text", "text": strings.Repeat("x", 2048)}}}},
		map[string]any{"type": "message_end", "message": map[string]any{"role": "assistant", "content": []any{map[string]any{"type": "text", "text": "task complete"}}}},
	}
	file, err := os.Create(trace)
	if err != nil {
		t.Fatal(err)
	}
	encoder := json.NewEncoder(file)
	for _, event := range events {
		if err := encoder.Encode(event); err != nil {
			file.Close()
			t.Fatal(err)
		}
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	return root, out
}

func writeTestJSON(t *testing.T, path string, value any) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := writeJSON(path, value); err != nil {
		t.Fatal(err)
	}
}

func readJSONLRows(t *testing.T, path string) []map[string]any {
	t.Helper()
	file, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	rows := []map[string]any{}
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		var row map[string]any
		if err := json.Unmarshal(scanner.Bytes(), &row); err != nil {
			t.Fatal(err)
		}
		rows = append(rows, row)
	}
	if err := scanner.Err(); err != nil {
		t.Fatal(err)
	}
	return rows
}

func TestPrepareEvalRLUsesPositionFourAndProjectsDeterministicPrompts(t *testing.T) {
	root := t.TempDir()
	zBundle := writeEvalRLTestBundle(t, root, "bundle-z", "task-z", "solve z exactly")
	writeTestJSON(t, filepath.Join(zBundle, "task.json"), map[string]any{
		"id": "task-z", "instruction": "solve z exactly",
		"pass": true, "reward": 1, "referenceAnswer": "SECRET_LABEL_MUST_NOT_APPEAR",
	})
	if err := os.WriteFile(filepath.Join(zBundle, "tests", "secret.txt"), []byte("SECRET_TEST_MUST_NOT_APPEAR"), 0o600); err != nil {
		t.Fatal(err)
	}
	writeEvalRLTestBundle(t, root, "bundle-a", "task-a", "solve a exactly")

	outOne := filepath.Join(t.TempDir(), "rl one")
	outTwo := filepath.Join(t.TempDir(), "rl two")
	prepareEvalRLForTest(t, root, outOne, nil)
	prepareEvalRLForTest(t, root, outTwo, nil)
	one, err := os.ReadFile(filepath.Join(outOne, "prompts.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	two, err := os.ReadFile(filepath.Join(outTwo, "prompts.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(one, two) {
		t.Fatal("preparing the same bundle parent produced different prompt rows")
	}
	if bytes.Contains(one, []byte("SECRET_")) || bytes.Contains(one, []byte(`"pass"`)) || bytes.Contains(one, []byte(`"reward"`)) {
		t.Fatalf("prompt projection leaked a secret or historical label: %s", one)
	}
	rows := readJSONLRows(t, filepath.Join(outOne, "prompts.jsonl"))
	assertEvalRLPromptRow(t, rows, 0, "task-a", "bundle-a", "solve a exactly")
	assertEvalRLPromptRow(t, rows, 1, "task-z", "bundle-z", "solve z exactly")
	manifestValue, err := readJSON(filepath.Join(outOne, "manifest.json"))
	if err != nil {
		t.Fatal(err)
	}
	grpoConfig := asObject(asObject(asObject(manifestValue)["trainer"])["grpoConfig"])
	wantGRPOKeys := []string{
		"num_generations", "max_steps", "learning_rate", "per_device_train_batch_size",
		"gradient_accumulation_steps", "max_completion_length", "max_tool_calling_iterations",
		"gradient_checkpointing", "logging_steps", "save_steps", "save_total_limit", "seed",
	}
	if len(grpoConfig) != len(wantGRPOKeys) {
		t.Fatalf("manifest grpoConfig has %d keys, want exactly %d: %#v", len(grpoConfig), len(wantGRPOKeys), grpoConfig)
	}
	for _, key := range wantGRPOKeys {
		if _, ok := grpoConfig[key]; !ok {
			t.Fatalf("manifest grpoConfig is missing supported key %q: %#v", key, grpoConfig)
		}
	}

	singleOut := filepath.Join(t.TempDir(), "single")
	prepareEvalRLForTest(t, zBundle, singleOut, nil)
	single := readJSONLRows(t, filepath.Join(singleOut, "prompts.jsonl"))
	assertEvalRLPromptRow(t, single, 0, "task-z", ".", "solve z exactly")
}

func TestPrepareEvalRLRequiresAcknowledgementBeforeWrites(t *testing.T) {
	out := filepath.Join(t.TempDir(), "must-not-exist")
	err := prepareEvalRL(cliArgs{
		positional: []string{"eval", "train", "rl", "prepare", filepath.Join(t.TempDir(), "missing-source")},
		opts: map[string]string{
			"out": out, "base-model": "org/model", "environment-factory": "pkg.environments:make",
		},
		flags: map[string]bool{},
	})
	requireCLIErrorCode(t, err, "benchmark_training_not_acknowledged")
	if _, statErr := os.Stat(out); !os.IsNotExist(statErr) {
		t.Fatalf("unacknowledged preparation touched output %q: %v", out, statErr)
	}
}

func TestPrepareEvalRLRejectsCompletedRunInput(t *testing.T) {
	root := t.TempDir()
	writeTestJSON(t, filepath.Join(root, "summary.json"), []any{})
	err := prepareEvalRL(evalRLPrepareArgs(root, filepath.Join(t.TempDir(), "out"), nil))
	requireCLIErrorCode(t, err, "rl_source_completed_run")
}

func TestPrepareEvalRLRejectsUnsafeDuplicateAndSymlinkBundles(t *testing.T) {
	t.Run("unsafe bundle reference", func(t *testing.T) {
		root := t.TempDir()
		writeEvalRLTestBundle(t, root, `unsafe\ref`, "task", "solve")
		err := prepareEvalRL(evalRLPrepareArgs(root, filepath.Join(t.TempDir(), "out"), nil))
		requireCLIErrorCode(t, err, "rl_bundle_ref_invalid")
	})

	t.Run("duplicate task id", func(t *testing.T) {
		root := t.TempDir()
		writeEvalRLTestBundle(t, root, "one", "duplicate", "solve one")
		writeEvalRLTestBundle(t, root, "two", "duplicate", "solve two")
		err := prepareEvalRL(evalRLPrepareArgs(root, filepath.Join(t.TempDir(), "out"), nil))
		requireCLIErrorCode(t, err, "rl_task_id_duplicate")
	})

	if runtime.GOOS == "windows" {
		return
	}
	t.Run("parent symlink escapes source", func(t *testing.T) {
		root := t.TempDir()
		outside := t.TempDir()
		writeEvalRLTestBundle(t, outside, ".", "outside", "solve outside")
		if err := os.Symlink(outside, filepath.Join(root, "escape")); err != nil {
			t.Fatal(err)
		}
		err := prepareEvalRL(evalRLPrepareArgs(root, filepath.Join(t.TempDir(), "out"), nil))
		requireCLIErrorCode(t, err, "rl_bundle_unsafe")
	})
	t.Run("nested symlink escapes bundle", func(t *testing.T) {
		root := t.TempDir()
		bundle := writeEvalRLTestBundle(t, root, "bundle", "task", "solve")
		outside := filepath.Join(t.TempDir(), "secret")
		if err := os.WriteFile(outside, []byte("secret"), 0o600); err != nil {
			t.Fatal(err)
		}
		if err := os.Symlink(outside, filepath.Join(bundle, "tests", "escape")); err != nil {
			t.Fatal(err)
		}
		err := prepareEvalRL(evalRLPrepareArgs(root, filepath.Join(t.TempDir(), "out"), nil))
		requireCLIErrorCode(t, err, "rl_bundle_unsafe")
	})
}

func TestPrepareEvalRLRejectsInvalidFactoryConfigAndGRPOKeys(t *testing.T) {
	root := t.TempDir()
	writeEvalRLTestBundle(t, root, "bundle", "task", "solve")
	configArray := filepath.Join(t.TempDir(), "environment.json")
	writeTestJSON(t, configArray, []any{"not", "an", "object"})
	unknownGRPO := filepath.Join(t.TempDir(), "grpo.json")
	writeTestJSON(t, unknownGRPO, map[string]any{"offline_reward": true})
	removedGRPO := filepath.Join(t.TempDir(), "removed-grpo.json")
	writeTestJSON(t, removedGRPO, map[string]any{"max_prompt_length": 1024})
	invalidGRPO := filepath.Join(t.TempDir(), "invalid-grpo.json")
	writeTestJSON(t, invalidGRPO, map[string]any{"max_steps": "many"})

	tests := []struct {
		name string
		opts map[string]string
		code string
	}{
		{name: "factory", opts: map[string]string{"environment-factory": "module:make"}, code: "invalid_environment_factory"},
		{name: "environment config type", opts: map[string]string{"environment-config": configArray}, code: "invalid_option"},
		{name: "unknown GRPO key", opts: map[string]string{"grpo-config": unknownGRPO}, code: "invalid_grpo_config"},
		{name: "removed max_prompt_length GRPO key", opts: map[string]string{"grpo-config": removedGRPO}, code: "invalid_grpo_config"},
		{name: "invalid GRPO value", opts: map[string]string{"grpo-config": invalidGRPO}, code: "invalid_grpo_config"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := prepareEvalRL(evalRLPrepareArgs(root, filepath.Join(t.TempDir(), "out"), test.opts))
			requireCLIErrorCode(t, err, test.code)
		})
	}
}

func TestRunEvalRLStrictlyValidatesManifest(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(map[string]any)
	}{
		{name: "type", mutate: func(manifest map[string]any) { manifest["kind"] = "localmaxxing.eval_sft" }},
		{name: "schema version", mutate: func(manifest map[string]any) { manifest["schemaVersion"] = 2 }},
		{name: "count", mutate: func(manifest map[string]any) { asObject(manifest["dataset"])["examples"] = 2 }},
		{name: "columns", mutate: func(manifest map[string]any) { asObject(manifest["dataset"])["columns"] = []any{"task_id", "prompt", "bundle_ref"} }},
		{name: "contamination", mutate: func(manifest map[string]any) { asObject(manifest["contamination"])["acknowledged"] = false }},
		{name: "missing GRPO key", mutate: func(manifest map[string]any) { delete(asObject(asObject(manifest["trainer"])["grpoConfig"]), "seed") }},
		{name: "unknown top-level field", mutate: func(manifest map[string]any) { manifest["labels"] = []any{1} }},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, out := prepareEvalRLFixture(t, false)
			manifestPath := filepath.Join(out, "manifest.json")
			value, err := readJSON(manifestPath)
			if err != nil {
				t.Fatal(err)
			}
			manifest := asObject(value)
			test.mutate(manifest)
			writeTestJSON(t, manifestPath, manifest)
			err = runEvalRLTrainer(evalRLRunArgs(manifestPath, nil, false))
			requireCLIErrorCode(t, err, "rl_manifest_invalid")
		})
	}
}

func TestRunEvalRLRejectsUnsafeAndDuplicatePromptRefs(t *testing.T) {
	t.Run("escaping reference", func(t *testing.T) {
		_, out := prepareEvalRLFixture(t, false)
		rows := readJSONLRows(t, filepath.Join(out, "prompts.jsonl"))
		rows[0]["bundle_ref"] = "../outside"
		writeJSONLTestRows(t, filepath.Join(out, "prompts.jsonl"), rows)
		err := runEvalRLTrainer(evalRLRunArgs(filepath.Join(out, "manifest.json"), nil, false))
		requireCLIErrorCode(t, err, "rl_dataset_invalid")
	})

	t.Run("duplicate reference", func(t *testing.T) {
		_, out := prepareEvalRLFixture(t, true)
		rows := readJSONLRows(t, filepath.Join(out, "prompts.jsonl"))
		rows[1]["bundle_ref"] = rows[0]["bundle_ref"]
		writeJSONLTestRows(t, filepath.Join(out, "prompts.jsonl"), rows)
		err := runEvalRLTrainer(evalRLRunArgs(filepath.Join(out, "manifest.json"), nil, false))
		requireCLIErrorCode(t, err, "rl_dataset_invalid")
	})
}

func TestRunEvalRLRejectsManifestOutputContainedBySource(t *testing.T) {
	root, out := prepareEvalRLFixture(t, false)
	preparedManifest := filepath.Join(out, "manifest.json")
	value, err := readJSON(preparedManifest)
	if err != nil {
		t.Fatal(err)
	}
	manifest := asObject(value)
	promptsPath := filepath.Join(root, "prompts.jsonl")
	prompts, err := os.ReadFile(filepath.Join(out, "prompts.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(promptsPath, prompts, 0o600); err != nil {
		t.Fatal(err)
	}
	asObject(manifest["dataset"])["path"] = promptsPath
	asObject(manifest["trainer"])["outputDir"] = filepath.Join(root, "grpo-output")
	containedManifest := filepath.Join(root, "manifest.json")
	writeTestJSON(t, containedManifest, manifest)
	err = runEvalRLTrainer(evalRLRunArgs(containedManifest, nil, false))
	requireCLIErrorCode(t, err, "rl_manifest_invalid")
}

func TestRunEvalRLCrossRejectsSFTAndRLManifests(t *testing.T) {
	t.Run("RL rejects SFT", func(t *testing.T) {
		root, out := writeTrainingFixture(t)
		if err := prepareEvalTrainingData(cliArgs{
			positional: []string{"eval", "train", "prepare", root},
			opts: map[string]string{"out": out, "base-model": "org/model"},
			flags: map[string]bool{"allow-benchmark-training": true, "quiet": true},
		}); err != nil {
			t.Fatal(err)
		}
		err := runEvalRLTrainer(evalRLRunArgs(filepath.Join(out, "manifest.json"), nil, false))
		requireCLIErrorCode(t, err, "rl_manifest_invalid")
	})

	t.Run("SFT rejects RL", func(t *testing.T) {
		_, out := prepareEvalRLFixture(t, false)
		err := runEvalTrainer(cliArgs{
			positional: []string{"eval", "train", "run", filepath.Join(out, "manifest.json")},
			opts:       map[string]string{"trainer-cmd": "must-not-run"},
			flags:      map[string]bool{},
		})
		requireCLIErrorCode(t, err, "training_manifest_invalid")
	})
}

func TestRunEvalRLPlanOnlyAllowsMissingPythonAndHasNoSideEffects(t *testing.T) {
	_, out := prepareEvalRLFixture(t, false)
	manifestPath := filepath.Join(out, "manifest.json")
	plannedOutput := filepath.Join(t.TempDir(), "not created ; $still-a-path")
	missingPython := filepath.Join(t.TempDir(), "python does not exist")
	err := runEvalRLTrainer(evalRLRunArgs(manifestPath, map[string]string{
		"python-bin": missingPython, "output-dir": plannedOutput, "resume": "none",
	}, false))
	if err != nil {
		t.Fatalf("plan-only run returned error: %v", err)
	}
	if _, statErr := os.Stat(plannedOutput); !os.IsNotExist(statErr) {
		t.Fatalf("plan-only run created output %q: %v", plannedOutput, statErr)
	}
	if _, statErr := os.Stat(filepath.Join(out, "grpo-output")); !os.IsNotExist(statErr) {
		t.Fatalf("plan-only run created manifest output: %v", statErr)
	}
}

func TestRunEvalRLExecuteUsesDirectArgvWithoutShellInjection(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("fake Python fixture uses a POSIX script")
	}
	root := filepath.Join(t.TempDir(), "source bundles ; $literal")
	if err := os.MkdirAll(root, 0o755); err != nil {
		t.Fatal(err)
	}
	writeEvalRLTestBundle(t, root, "bundle one", "task-one", "solve one")
	out := filepath.Join(t.TempDir(), "rl data ; $literal")
	prepareEvalRLForTest(t, root, out, nil)

	fakeDir := t.TempDir()
	fakePython := filepath.Join(fakeDir, "fake python;touch $LMX_INJECTION_SENTINEL")
	script := "#!/bin/sh\nset -eu\n: > \"$LMX_ARGV_LOG\"\nfor arg in \"$@\"; do printf '%s\\n' \"$arg\" >> \"$LMX_ARGV_LOG\"; done\ncp \"$1\" \"$LMX_HELPER_COPY\"\n"
	if err := os.WriteFile(fakePython, []byte(script), 0o700); err != nil {
		t.Fatal(err)
	}
	argvLog := filepath.Join(t.TempDir(), "argv.log")
	helperCopy := filepath.Join(t.TempDir(), "embedded helper.py")
	sentinel := filepath.Join(t.TempDir(), "SHELL_WAS_USED")
	t.Setenv("LMX_ARGV_LOG", argvLog)
	t.Setenv("LMX_HELPER_COPY", helperCopy)
	t.Setenv("LMX_INJECTION_SENTINEL", sentinel)
	outputOverride := filepath.Join(t.TempDir(), "trainer output ; $(not-executed) $literal")
	err := runEvalRLTrainer(evalRLRunArgs(filepath.Join(out, "manifest.json"), map[string]string{
		"python-bin": fakePython, "output-dir": outputOverride, "resume": "none",
	}, true))
	if err != nil {
		t.Fatalf("executed RL runner returned error: %v", err)
	}
	data, err := os.ReadFile(argvLog)
	if err != nil {
		t.Fatal(err)
	}
	argv := strings.Split(strings.TrimSuffix(string(data), "\n"), "\n")
	if len(argv) != 7 {
		t.Fatalf("fake Python argv = %#v, want seven arguments", argv)
	}
	if !strings.HasPrefix(filepath.Base(argv[0]), "lmx-train-eval-grpo-") || filepath.Ext(argv[0]) != ".py" {
		t.Fatalf("helper argv[0] = %q, want materialized embedded helper", argv[0])
	}
	want := []string{"--manifest", filepath.Join(out, "manifest.json"), "--output-dir", outputOverride, "--resume", "none"}
	for i := range want {
		if argv[i+1] != want[i] {
			t.Fatalf("argv[%d] = %q, want %q (full argv %#v)", i+1, argv[i+1], want[i], argv)
		}
	}
	if _, statErr := os.Stat(sentinel); !os.IsNotExist(statErr) {
		t.Fatalf("metacharacter executable path triggered shell injection: %v", statErr)
	}
	if _, statErr := os.Stat(argv[0]); !os.IsNotExist(statErr) {
		t.Fatalf("temporary helper was not removed after execution: %v", statErr)
	}
	materialized, err := os.ReadFile(helperCopy)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(materialized, trainEvalGRPOScript) {
		t.Fatal("executed helper bytes differ from embedded helper bytes")
	}
}

func TestRunEvalRLExecuteReportsPythonNotFound(t *testing.T) {
	_, out := prepareEvalRLFixture(t, false)
	missing := filepath.Join(t.TempDir(), "missing python")
	err := runEvalRLTrainer(evalRLRunArgs(filepath.Join(out, "manifest.json"), map[string]string{"python-bin": missing}, true))
	requireCLIErrorCode(t, err, "python_not_found")
}

func TestEmbeddedGRPOScriptMatchesCanonicalFile(t *testing.T) {
	canonical, err := os.ReadFile("train_eval_grpo.py")
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(trainEvalGRPOScript, canonical) {
		t.Fatal("embedded GRPO helper differs byte-for-byte from cmd/lmx/train_eval_grpo.py")
	}
}

func evalRLPrepareArgs(source, out string, overrides map[string]string) cliArgs {
	opts := map[string]string{
		"out": out, "base-model": "org/base-model", "environment-factory": "example.environments:make_environment",
	}
	for key, value := range overrides {
		opts[key] = value
	}
	return cliArgs{
		positional: []string{"eval", "train", "rl", "prepare", source},
		opts:       opts,
		flags:      map[string]bool{"allow-benchmark-training": true, "quiet": true},
	}
}

func evalRLRunArgs(manifest string, overrides map[string]string, execute bool) cliArgs {
	opts := map[string]string{}
	for key, value := range overrides {
		opts[key] = value
	}
	return cliArgs{
		positional: []string{"eval", "train", "rl", "run", manifest},
		opts:       opts,
		flags:      map[string]bool{"execute": execute, "quiet": true},
	}
}

func prepareEvalRLForTest(t *testing.T, source, out string, overrides map[string]string) {
	t.Helper()
	if err := prepareEvalRL(evalRLPrepareArgs(source, out, overrides)); err != nil {
		t.Fatalf("prepareEvalRL returned error: %v", err)
	}
}

func prepareEvalRLFixture(t *testing.T, twoBundles bool) (string, string) {
	t.Helper()
	root := t.TempDir()
	writeEvalRLTestBundle(t, root, "bundle-b", "task-b", "solve task b")
	if twoBundles {
		writeEvalRLTestBundle(t, root, "bundle-a", "task-a", "solve task a")
	}
	out := filepath.Join(t.TempDir(), "rl output")
	prepareEvalRLForTest(t, root, out, nil)
	return root, out
}

func writeEvalRLTestBundle(t *testing.T, root, ref, id, instruction string) string {
	t.Helper()
	bundle := root
	if ref != "." {
		bundle = filepath.Join(root, ref)
	}
	if err := os.MkdirAll(filepath.Join(bundle, "tests"), 0o755); err != nil {
		t.Fatal(err)
	}
	writeTestJSON(t, filepath.Join(bundle, "task.json"), map[string]any{"id": id, "instruction": instruction})
	return bundle
}

func assertEvalRLPromptRow(t *testing.T, rows []map[string]any, index int, taskID, bundleRef, instruction string) {
	t.Helper()
	if len(rows) <= index {
		t.Fatalf("prompt rows = %#v, missing index %d", rows, index)
	}
	row := rows[index]
	if len(row) != 3 || stringValue(row["task_id"]) != taskID || stringValue(row["bundle_ref"]) != bundleRef {
		t.Fatalf("prompt row %d = %#v, want exact task_id/bundle_ref projection", index, row)
	}
	prompt := anySlice(row["prompt"])
	if len(prompt) != 1 {
		t.Fatalf("prompt row %d messages = %#v, want one", index, prompt)
	}
	message := asObject(prompt[0])
	if len(message) != 2 || stringValue(message["role"]) != "user" || stringValue(message["content"]) != instruction {
		t.Fatalf("prompt row %d message = %#v, want exact user instruction", index, message)
	}
}

func requireCLIErrorCode(t *testing.T, err error, code string) {
	t.Helper()
	var cliErr cliError
	if !errors.As(err, &cliErr) || cliErr.Code != code {
		t.Fatalf("error = %#v, want cli error code %q", err, code)
	}
}

func writeJSONLTestRows(t *testing.T, path string, rows []map[string]any) {
	t.Helper()
	file, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	encoder := json.NewEncoder(file)
	for _, row := range rows {
		if err := encoder.Encode(row); err != nil {
			file.Close()
			t.Fatal(err)
		}
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
}
