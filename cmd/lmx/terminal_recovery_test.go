package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

func TestTerminalRecoveryCheckpointCommitsCompleteSnapshotsAtomically(t *testing.T) {
	fixture := newTerminalRecoveryTestFixture(t)
	manager := fixture.manager()
	first := terminalRecoveryCompletedResult(true)
	first.lastProgressAt = "2026-07-17T01:02:03Z"
	if err := manager.persist(0, fixture.bundles[0], first); err != nil {
		t.Fatalf("persist first verifier result: %v", err)
	}
	assertTerminalRecoveryCheckpointSnapshot(t, fixture.checkpoint, fixture.provenance, []string{"task-one"})

	before := terminalRecoveryReadFiles(t, fixture.checkpoint, "checkpoint.json", "summary.json", "task-one.json")
	manager.entries["task-two"] = terminalCheckpointEntry{Index: 2, Total: 2, Task: "task-two", Out: "task-two.json", Summary: managerSafeMap(fixture.provenance)}
	bad := terminalSavedResultFromRun(fixture.bundles[1], terminalRecoveryCompletedResult(false), fixture.provenance)
	bad.TokenUsage = map[string]any{"cannotEncode": make(chan int)}
	manager.results["task-two"] = bad
	if err := manager.persistLocked(); err == nil {
		t.Fatal("persistLocked accepted an unencodable staged task wrapper")
	}
	after := terminalRecoveryReadFiles(t, fixture.checkpoint, "checkpoint.json", "summary.json", "task-one.json")
	if !reflect.DeepEqual(after, before) {
		t.Fatalf("failed staged commit changed the last complete checkpoint snapshot\nbefore=%q\nafter=%q", before, after)
	}
	for _, pattern := range []string{"." + filepath.Base(fixture.checkpoint) + ".stage-*", filepath.Base(fixture.checkpoint) + ".previous-*"} {
		matches, err := filepath.Glob(filepath.Join(filepath.Dir(fixture.checkpoint), pattern))
		if err != nil {
			t.Fatalf("glob checkpoint transaction artifacts: %v", err)
		}
		if len(matches) != 0 {
			t.Fatalf("failed checkpoint commit left transaction artifacts: %v", matches)
		}
	}

	second := terminalRecoveryCompletedResult(false)
	second.lastProgressAt = "2026-07-17T01:03:04Z"
	if err := manager.persist(1, fixture.bundles[1], second); err != nil {
		t.Fatalf("persist second verifier result after failed staging attempt: %v", err)
	}
	assertTerminalRecoveryCheckpointSnapshot(t, fixture.checkpoint, fixture.provenance, []string{"task-one", "task-two"})
}

func TestTerminalRecoveryTaskStatusNeverAdvertisesPartialSubmit(t *testing.T) {
	fixture := newTerminalRecoveryTestFixture(t)
	manager := fixture.manager()
	if err := manager.persist(0, fixture.bundles[0], terminalRecoveryCompletedResult(true)); err != nil {
		t.Fatalf("persist partial checkpoint: %v", err)
	}
	stderr := captureTerminalTestStderr(t, func() {
		printTerminalTaskRecovery(
			parseArgs([]string{"--json-status", "--api-url", "http://localmaxxing.invalid"}),
			"fixture-dataset", 7, 0, len(fixture.bundles), fixture.bundles[0], terminalRecoveryCompletedResult(true), manager,
		)
	})
	var event map[string]any
	if err := json.Unmarshal([]byte(strings.TrimSpace(stderr)), &event); err != nil {
		t.Fatalf("decode terminal_task_recovery event %q: %v", stderr, err)
	}
	if event["event"] != "terminal_task_recovery" || event["taskId"] != "task-one" || event["checkpoint"] != fixture.checkpoint {
		t.Fatalf("partial task recovery event = %#v", event)
	}
	resumeCommand := stringValue(event["resumeCommand"])
	if !strings.HasPrefix(resumeCommand, "lmx eval terminal run ") || !strings.Contains(resumeCommand, "--resume "+shellQuote(fixture.checkpoint)) || strings.Contains(resumeCommand, "--submit") || strings.Contains(resumeCommand, "<") {
		t.Fatalf("partial checkpoint resumeCommand is not copy-safe and exact: %q; event=%#v", resumeCommand, event)
	}
	if _, advertised := event["deferredSubmitCommand"]; advertised {
		t.Fatalf("partial checkpoint advertised a submit command: %#v", event)
	}
	if err := manager.persist(1, fixture.bundles[1], terminalRecoveryCompletedResult(false)); err != nil {
		t.Fatalf("persist completed checkpoint: %v", err)
	}
	stderr = captureTerminalTestStderr(t, func() {
		printTerminalTaskRecovery(
			parseArgs([]string{"--json-status", "--api-url", "http://localmaxxing.invalid"}),
			"fixture-dataset", 7, 1, len(fixture.bundles), fixture.bundles[1], terminalRecoveryCompletedResult(false), manager,
		)
	})
	event = map[string]any{}
	if err := json.Unmarshal([]byte(strings.TrimSpace(stderr)), &event); err != nil {
		t.Fatalf("decode complete terminal_task_recovery event %q: %v", stderr, err)
	}
	if command := stringValue(event["deferredSubmitCommand"]); !strings.HasPrefix(command, "lmx eval terminal submit ") || strings.Contains(command, "<") {
		t.Fatalf("complete checkpoint submit command is not copy-safe: %q; event=%#v", command, event)
	}
	if _, advertised := event["resumeCommand"]; advertised {
		t.Fatalf("complete checkpoint advertised resume instead of submit: %#v", event)
	}
}

func TestTerminalRecoveryDeferredSubmitCommandRequiresCompleteScoring(t *testing.T) {
	args := parseArgs([]string{"--api-url", "http://localmaxxing.invalid"})
	partial := map[string]any{"dataset": "fixture-dataset", "tasks": 2, "scored": 1, "hfId": "fixture/model", "hardware": map[string]any{"hwClass": "CPU_ONLY"}}
	if command := terminalDeferredSubmitCommand(args, "atomic-checkpoint", partial); command != "" {
		t.Fatalf("partial scoring advertised deferred submit command %q", command)
	}
	complete := managerSafeMap(partial)
	complete["scored"] = 2
	command := terminalDeferredSubmitCommand(args, "atomic-checkpoint", complete)
	if !strings.Contains(command, "lmx eval terminal submit") || !strings.Contains(command, "--api-url http://localmaxxing.invalid") {
		t.Fatalf("complete checkpoint deferred submit command = %q", command)
	}
}

func TestTerminalRecoveryDeferredSubmitCommandRequiresCanonicalTB21TaskSet(t *testing.T) {
	args := parseArgs([]string{"--api-url", "http://localmaxxing.invalid"})
	summary := map[string]any{
		"artifactVersion": terminalCheckpointArtifactVersion,
		"dataset":         terminalBench21Dataset,
		"shardIndex":      1,
		"tasks":           1,
		"scored":          1,
		"taskOrder":       []string{"cancel-async-tasks"},
		"hfId":            "fixture/model",
		"hardware":        map[string]any{"hwClass": "CPU_ONLY"},
	}
	if command := terminalDeferredSubmitCommand(args, "atomic-checkpoint", summary); command != "" {
		t.Fatalf("filtered Terminal-Bench checkpoint advertised unusable submit command %q", command)
	}

	firstShardEnd := len(terminalBench21CanonicalTaskIDs) / terminalBench21ShardCount
	summary["tasks"] = firstShardEnd
	summary["scored"] = firstShardEnd
	summary["taskOrder"] = append([]string(nil), terminalBench21CanonicalTaskIDs[:firstShardEnd]...)
	if command := terminalDeferredSubmitCommand(args, "atomic-checkpoint", summary); !strings.Contains(command, "lmx eval terminal submit") {
		t.Fatalf("canonical Terminal-Bench shard did not advertise deferred submit command: %q", command)
	}
}

func TestTerminalRecoveryResumeRequiresExactImmutableProvenance(t *testing.T) {
	fixture := newTerminalRecoveryTestFixture(t)
	manager := fixture.manager()
	for index := range fixture.bundles {
		if err := manager.persist(index, fixture.bundles[index], terminalRecoveryCompletedResult(index == 0)); err != nil {
			t.Fatalf("persist task %d: %v", index, err)
		}
	}
	matching, err := newTerminalCheckpointManager(
		parseArgs([]string{"--resume", fixture.checkpoint}),
		"fixture-dataset", 7, fixture.bundles, managerSafeMap(fixture.provenance),
	)
	if err != nil {
		t.Fatalf("matching checkpoint resume: %v", err)
	}
	for _, id := range []string{"task-one", "task-two"} {
		if _, ok := matching.resumedResult(id); !ok {
			t.Fatalf("matching checkpoint did not resume completed task %q", id)
		}
	}

	tests := []struct {
		name  string
		field string
		value any
	}{
		{name: "dataset", field: "dataset", value: "other-dataset"},
		{name: "shard", field: "shardIndex", value: 8},
		{name: "canonical task set", field: "canonicalTaskIds", value: []string{"task-two", "task-one"}},
		{name: "selected task set", field: "selectedTaskIds", value: []string{"task-one"}},
		{name: "manifest identity", field: "manifestIdentity", value: "fixture-dataset/shard/8"},
		{name: "manifest hash", field: "manifestSha256", value: strings.Repeat("0", 64)},
		{name: "selected manifest hash", field: "selectedManifestSha256", value: strings.Repeat("1", 64)},
		{name: "model identity", field: "hfId", value: "other/model"},
		{name: "served model", field: "servedModel", value: "other-served-model"},
		{name: "quantization", field: "quantization", value: "Q8_0"},
		{name: "hardware", field: "hardwareSha256", value: strings.Repeat("2", 64)},
		{name: "runner", field: "runnerVersion", value: "different-runner"},
		{name: "model resolution", field: "modelResolution", value: map[string]any{"status": "different"}},
		{name: "run configuration", field: "runConfig", value: map[string]any{"protocol": "different"}},
		{name: "provenance digest", field: "provenanceSha256", value: strings.Repeat("3", 64)},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			current := managerSafeMap(fixture.provenance)
			current[tc.field] = tc.value
			_, err := newTerminalCheckpointManager(parseArgs([]string{"--resume", fixture.checkpoint}), "fixture-dataset", 7, fixture.bundles, current)
			var cliErr cliError
			if !errors.As(err, &cliErr) || cliErr.Code != "checkpoint_provenance_mismatch" {
				t.Fatalf("error = %#v, want checkpoint_provenance_mismatch for %s", err, tc.field)
			}
			details := asObject(cliErr.Details)
			if details["field"] != tc.field {
				t.Fatalf("mismatch details = %#v, want field %q", details, tc.field)
			}
		})
	}

	t.Run("metadata task order", func(t *testing.T) {
		metadata := terminalRecoveryReadObject(t, filepath.Join(fixture.checkpoint, "checkpoint.json"))
		metadata["taskOrder"] = []any{"task-two", "task-one"}
		writeTerminalTestJSON(t, filepath.Join(fixture.checkpoint, "checkpoint.json"), metadata)
		_, err := newTerminalCheckpointManager(parseArgs([]string{"--resume", fixture.checkpoint}), "fixture-dataset", 7, fixture.bundles, managerSafeMap(fixture.provenance))
		var cliErr cliError
		if !errors.As(err, &cliErr) || cliErr.Code != "checkpoint_task_order_mismatch" {
			t.Fatalf("error = %#v, want checkpoint_task_order_mismatch", err)
		}
	})
}

func TestTerminalRecoveryResumeRerunsIncompleteOrUnscoredEntries(t *testing.T) {
	tests := []struct {
		name   string
		result terminalTaskResult
		resume bool
	}{
		{name: "complete canonical score", result: terminalRecoveryCompletedResult(true), resume: true},
		{name: "unscored", result: terminalTaskResult{verifierAttempted: true, verifierCompleted: true, rewardParsed: true}},
		{name: "verifier not attempted", result: terminalTaskResult{scored: true, verifierCompleted: true, rewardParsed: true}},
		{name: "verifier not completed", result: terminalTaskResult{scored: true, verifierAttempted: true, rewardParsed: true}},
		{name: "reward not parsed", result: terminalTaskResult{scored: true, verifierAttempted: true, verifierCompleted: true}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			fixture := newTerminalRecoveryTestFixture(t)
			fixture.bundles = fixture.bundles[:1]
			fixture.provenance = terminalRecoveryProvenance(t, fixture.bundles)
			manager := fixture.manager()
			if err := manager.persist(0, fixture.bundles[0], tc.result); err != nil {
				t.Fatalf("persist fixture result: %v", err)
			}
			resumed, err := newTerminalCheckpointManager(parseArgs([]string{"--resume", fixture.checkpoint}), "fixture-dataset", 7, fixture.bundles, managerSafeMap(fixture.provenance))
			if err != nil {
				t.Fatalf("load resume checkpoint: %v", err)
			}
			_, ok := resumed.resumedResult("task-one")
			if ok != tc.resume {
				t.Fatalf("resumedResult present = %v, want %v; incomplete or unscored verifier evidence must rerun", ok, tc.resume)
			}
		})
	}
}

func TestTerminalRecoveryTimeoutDomainsStayIndependentForRoutedExternalAgent(t *testing.T) {
	capturePath := filepath.Join(t.TempDir(), "timeouts.txt")
	t.Setenv("LMX_TIMEOUT_CAPTURE", capturePath)
	task := terminalTask{ID: "timeout-task", Instruction: "finish quickly", Agent: terminalAgentConfig{TimeoutSec: 99}, Verifier: terminalVerifierConfig{TimeoutSec: 11}}
	cfg := terminalConfig{
		args:              parseArgs([]string{"--quiet"}),
		agentCommand:      `printf '%s|%s' "$LMX_TERMINAL_COMMAND_TIMEOUT_SECONDS" "$LMX_TERMINAL_AGENT_TIMEOUT_SEC" > "$LMX_TIMEOUT_CAPTURE"`,
		agentExecution:    "routed-shell",
		commandTimeoutSec: 2,
		agentTimeoutSec:   5,
		endpointTimeout:   9 * time.Second,
	}
	if terminalCommandTimeoutSec(cfg) != 2 || terminalAgentTimeoutSec(cfg, task) != 5 || terminalEndpointTimeout(cfg) != 9*time.Second || task.Verifier.TimeoutSec != 11 {
		t.Fatalf("timeout domains collapsed: command=%d agent=%d endpoint=%s verifier=%d", terminalCommandTimeoutSec(cfg), terminalAgentTimeoutSec(cfg, task), terminalEndpointTimeout(cfg), task.Verifier.TimeoutSec)
	}
	_, _, err := runExternalTerminalAgent(context.Background(), task, t.TempDir(), "unused-container", "http://model.invalid", "fixture-model", cfg)
	if err != nil {
		t.Fatalf("fast routed external agent: %v", err)
	}
	data, readErr := os.ReadFile(capturePath)
	if readErr != nil {
		t.Fatalf("read external-agent timeout environment: %v", readErr)
	}
	if string(data) != "2|5" {
		t.Fatalf("routed external-agent timeouts = %q, want command|agent = 2|5", data)
	}

	traceRoot := t.TempDir()
	cfg.traceRoot = traceRoot
	cfg.agentCommand = `printf '%s\n' '{"return_code":124,"stderr_tail":"command_timeout"}' > "$LMX_TERMINAL_TRACE_DIR/environment-exec.jsonl"`
	transcript, _, err := runExternalTerminalAgent(context.Background(), task, t.TempDir(), "unused-container", "http://model.invalid", "fixture-model", cfg)
	var outcome terminalAgentOutcomeError
	if !errors.As(err, &outcome) || outcome.code != "command_timeout" {
		t.Fatalf("routed command-timeout evidence error = %#v, want command_timeout", err)
	}
	if !strings.Contains(transcript, "command_timeout: routed shell command exceeded --command-timeout-seconds") {
		t.Fatalf("routed command-timeout transcript missing bounded-command diagnosis: %q", transcript)
	}
}

func TestTerminalRecoveryRepeatGuardResetsOnProgressBeforeExhaustion(t *testing.T) {
	guard := terminalRepeatGuard{}
	if nudge, exhausted := guard.observe([]string{"printf 12345"}, "same output", 3); nudge || exhausted {
		t.Fatalf("first command observation = (%v,%v), want no warning", nudge, exhausted)
	}
	if nudge, exhausted := guard.observe([]string{"  PRINTF   67890 "}, "same   output", 3); !nudge || exhausted {
		t.Fatalf("near-identical repeat = (%v,%v), want protocol nudge only", nudge, exhausted)
	}
	if nudge, exhausted := guard.observe([]string{"printf 67890"}, "progress happened", 3); nudge || exhausted {
		t.Fatalf("changed observation did not reset repeat guard: (%v,%v)", nudge, exhausted)
	}
	if nudge, exhausted := guard.observe([]string{"printf 99999"}, "progress happened", 3); !nudge || exhausted {
		t.Fatalf("second repeat after reset = (%v,%v), want protocol nudge only", nudge, exhausted)
	}
	if nudge, exhausted := guard.observe([]string{"printf 11111"}, "progress happened", 3); nudge || !exhausted {
		t.Fatalf("third repeat after reset = (%v,%v), want exhaustion", nudge, exhausted)
	}
}

func TestTerminalRecoveryProvenanceIsCompleteDigestBoundAndSecretFree(t *testing.T) {
	fixture := newTerminalRecoveryTestFixture(t)
	provenance := fixture.provenance
	for _, field := range []string{
		"artifactVersion", "dataset", "shardIndex", "canonicalTaskIds", "selectedTaskIds", "taskOrder", "taskOrderSha256",
		"manifestIdentity", "manifestSha256", "manifestVersion", "manifestTaskVersions", "selectedManifestSha256", "manifestItems",
		"declaredModel", "hfId", "servedModel", "quantization", "quantFormat", "hardware", "hardwareSha256", "runnerVersion",
		"modelResolution", "quantizationResolution", "runConfig", "provenanceSha256",
	} {
		if value, present := provenance[field]; !present || value == nil || value == "" {
			t.Fatalf("immutable provenance missing %q: %#v", field, provenance)
		}
	}
	items := anySlice(provenance["manifestItems"])
	if len(items) != len(fixture.bundles) {
		t.Fatalf("manifestItems = %d, want %d", len(items), len(fixture.bundles))
	}
	for _, raw := range items {
		item := asObject(raw)
		for _, field := range []string{"questionId", "version", "source", "bundleKey", "sha256", "byteSize", "verifierCommand", "rewardFile", "verifierTimeoutSeconds", "agentTimeoutSeconds", "agentMaxTurns"} {
			if _, present := item[field]; !present {
				t.Fatalf("manifest provenance item missing %q: %#v", field, item)
			}
		}
	}
	withoutHash := managerSafeMap(provenance)
	claimed := stringValue(withoutHash["provenanceSha256"])
	delete(withoutHash, "provenanceSha256")
	computed, err := terminalJSONHash(withoutHash)
	if err != nil || claimed != computed {
		t.Fatalf("provenance digest = (%q,%v), computed %q", claimed, err, computed)
	}
	encoded, err := json.Marshal(provenance)
	if err != nil {
		t.Fatalf("marshal provenance: %v", err)
	}
	for _, secret := range []string{"TOP_SECRET_TASK_INSTRUCTION", "TOP_SECRET_TASK_ENV", "model-password", "model-token", "secret-query"} {
		if strings.Contains(string(encoded), secret) {
			t.Fatalf("provenance leaked secret %q: %s", secret, encoded)
		}
	}
	if got := stringValue(asObject(provenance["runConfig"])["modelEndpoint"]); got != "https://model.example:8443" {
		t.Fatalf("sanitized model endpoint = %q, want origin only", got)
	}
}

func TestTerminalRecoveryLegacySubmitCompatibilityAndExactOfflineEvent(t *testing.T) {
	runDir, hardwarePath := writeDeferredTerminalSubmitFixture(t)
	payloadPath := filepath.Join(t.TempDir(), "legacy-payload.json")
	stdout, err := captureTerminalTestStdout(t, func() error {
		return submitTerminalEval(parseArgs([]string{
			"eval", "terminal", "submit", runDir,
			"--dataset", "terminal-bench-fixture", "--shard-index", "7",
			"--hf-id", "fixture/model", "--hardware", hardwarePath,
			"--out", payloadPath, "--dry-run", "--quiet",
		}))
	})
	if err != nil {
		t.Fatalf("legacy checkpoint dry-run submit: %v", err)
	}
	const exact = "offline_submit_validation_no_execution: saved results were validated and packaged; no Docker, model, verifier, or network submission was executed."
	lines := strings.Split(strings.TrimSpace(stdout), "\n")
	if len(lines) == 0 || lines[len(lines)-1] != exact {
		t.Fatalf("offline submit wording = %q, want final line exactly %q", stdout, exact)
	}
	batch := readTerminalSubmitBatch(t, payloadPath)
	shards := anySlice(batch["shards"])
	if batch["dataset"] != "terminal-bench-fixture" || len(shards) != 1 || len(anySlice(asObject(shards[0])["results"])) != 2 {
		t.Fatalf("legacy checkpoint compatibility payload = %#v", batch)
	}
}

func TestTerminalRecoveryResumeRejectsUnknownTraversalSymlinkAndTrailingJSON(t *testing.T) {
	t.Run("unknown summary task", func(t *testing.T) {
		fixture := newTerminalRecoveryTestFixture(t)
		manager := fixture.manager()
		if err := manager.persist(0, fixture.bundles[0], terminalRecoveryCompletedResult(true)); err != nil {
			t.Fatalf("persist checkpoint: %v", err)
		}
		entries := terminalRecoveryReadSummary(t, filepath.Join(fixture.checkpoint, "summary.json"))
		entries[0].Task = "unknown-task"
		writeTerminalTestJSON(t, filepath.Join(fixture.checkpoint, "summary.json"), entries)
		terminalRecoveryRequireResumeError(t, fixture, "checkpoint_task_order_mismatch")
	})

	t.Run("traversal result reference", func(t *testing.T) {
		fixture := newTerminalRecoveryTestFixture(t)
		manager := fixture.manager()
		if err := manager.persist(0, fixture.bundles[0], terminalRecoveryCompletedResult(true)); err != nil {
			t.Fatalf("persist checkpoint: %v", err)
		}
		entries := terminalRecoveryReadSummary(t, filepath.Join(fixture.checkpoint, "summary.json"))
		outside := filepath.Join(filepath.Dir(fixture.checkpoint), "outside-task-one.json")
		if err := os.Rename(filepath.Join(fixture.checkpoint, "task-one.json"), outside); err != nil {
			t.Fatalf("move wrapper outside checkpoint: %v", err)
		}
		entries[0].Out = "../outside-task-one.json"
		writeTerminalTestJSON(t, filepath.Join(fixture.checkpoint, "summary.json"), entries)
		terminalRecoveryRequireResumeError(t, fixture, "task_result_missing")
	})

	t.Run("symlinked result wrapper", func(t *testing.T) {
		fixture := newTerminalRecoveryTestFixture(t)
		manager := fixture.manager()
		if err := manager.persist(0, fixture.bundles[0], terminalRecoveryCompletedResult(true)); err != nil {
			t.Fatalf("persist checkpoint: %v", err)
		}
		direct := filepath.Join(fixture.checkpoint, "task-one.json")
		outside := filepath.Join(filepath.Dir(fixture.checkpoint), "outside-task-one.json")
		data, err := os.ReadFile(direct)
		if err != nil {
			t.Fatalf("read wrapper: %v", err)
		}
		if err := os.WriteFile(outside, data, 0o600); err != nil {
			t.Fatalf("write outside wrapper: %v", err)
		}
		if err := os.Remove(direct); err != nil {
			t.Fatalf("remove checkpoint wrapper: %v", err)
		}
		if err := os.Symlink(outside, direct); err != nil {
			t.Fatalf("symlink checkpoint wrapper: %v", err)
		}
		terminalRecoveryRequireResumeError(t, fixture, "task_result_invalid")
	})

	for _, tc := range []struct {
		name string
		file string
		code string
	}{
		{name: "trailing checkpoint JSON", file: "checkpoint.json", code: "checkpoint_metadata_invalid"},
		{name: "trailing summary JSON", file: "summary.json", code: "checkpoint_summary_invalid"},
		{name: "trailing result JSON", file: "task-one.json", code: "task_result_invalid"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fixture := newTerminalRecoveryTestFixture(t)
			manager := fixture.manager()
			if err := manager.persist(0, fixture.bundles[0], terminalRecoveryCompletedResult(true)); err != nil {
				t.Fatalf("persist checkpoint: %v", err)
			}
			path := filepath.Join(fixture.checkpoint, tc.file)
			file, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0)
			if err != nil {
				t.Fatalf("open %s: %v", tc.file, err)
			}
			if _, err := file.WriteString("{}\n"); err != nil {
				_ = file.Close()
				t.Fatalf("append trailing JSON to %s: %v", tc.file, err)
			}
			if err := file.Close(); err != nil {
				t.Fatalf("close %s: %v", tc.file, err)
			}
			terminalRecoveryRequireResumeError(t, fixture, tc.code)
		})
	}
}

func TestTerminalRecoveryCommandRerunsCanonicalVerifierAndImportsOnlyMissingTask(t *testing.T) {
	fixture := newTerminalRecoveryTestFixture(t)
	manager := fixture.manager()
	if err := manager.persist(0, fixture.bundles[0], terminalRecoveryCompletedResult(true)); err != nil {
		t.Fatalf("persist existing task: %v", err)
	}
	resultPath := terminalRecoveryWriteCandidateResult(t, fixture, 1, false)
	dockerLog := terminalRecoveryInstallDockerVerifier(t)

	stdout, err := captureTerminalTestStdout(t, func() error {
		return recoverTerminalCheckpoint(parseArgs([]string{
			"eval", "terminal", "recover", fixture.checkpoint,
			"--task-id", "task-two", "--container", "fixture-container", "--bundle", fixture.bundles[1].Dir,
			"--result", resultPath, "--quiet",
		}))
	})
	if err != nil {
		t.Fatalf("recover missing task with canonical verifier execution: %v", err)
	}
	logData, readErr := os.ReadFile(dockerLog)
	if readErr != nil {
		t.Fatalf("read verifier execution log: %v", readErr)
	}
	logText := string(logData)
	for _, want := range []string{"fixture-container", "bash -c bash /tests/test.sh", "cat /logs/verifier/reward.json"} {
		if !strings.Contains(logText, want) {
			t.Fatalf("canonical recovery verifier log missing %q:\n%s", want, logText)
		}
	}
	if strings.Contains(stdout, "no Docker") || strings.Contains(stdout, "verifier, or") {
		t.Fatalf("trusted recovery falsely claimed no Docker/verifier execution: %q", stdout)
	}
	assertTerminalRecoveryCheckpointSnapshot(t, fixture.checkpoint, fixture.provenance, []string{"task-one", "task-two"})
	result := terminalRecoveryReadSavedResult(t, filepath.Join(fixture.checkpoint, "task-two.json"))
	if result.Scored == nil || !*result.Scored || !result.VerifierAttempted || !result.VerifierCompleted || !result.RewardParsed || !result.Pass {
		t.Fatalf("recovered task was not derived from a complete canonical verifier score: %#v", result)
	}
	if !terminalJSONEqual(result.Provenance, fixture.provenance) {
		t.Fatalf("recovered result provenance differs from checkpoint provenance")
	}
}

func TestTerminalRecoveryCommandRejectsSelfAuthoredEvidenceWithoutVerifierExecution(t *testing.T) {
	for _, scored := range []bool{true, false} {
		name := map[bool]string{true: "self-asserted scored evidence", false: "self-asserted unscored evidence"}[scored]
		t.Run(name, func(t *testing.T) {
			fixture := newTerminalRecoveryTestFixture(t)
			manager := fixture.manager()
			if err := manager.persist(0, fixture.bundles[0], terminalRecoveryCompletedResult(true)); err != nil {
				t.Fatalf("persist existing task: %v", err)
			}
			before := terminalRecoveryReadFiles(t, fixture.checkpoint, "checkpoint.json", "summary.json", "task-one.json")
			result := terminalRecoveryCandidateSavedResult(fixture, 1, scored)
			resultPath := filepath.Join(t.TempDir(), "self-authored-result.json")
			evidencePath := filepath.Join(t.TempDir(), "self-authored-evidence.json")
			writeTerminalTestJSON(t, resultPath, terminalSavedTaskFile{Results: []terminalSavedResult{result}})
			evidence := terminalRecoveryEvidenceForResult(t, fixture.provenance, fixture.bundles[1], result)
			evidence["scored"] = scored
			writeTerminalTestJSON(t, evidencePath, evidence)
			dockerLog := terminalRecoveryInstallDockerDeny(t)

			err := recoverTerminalCheckpoint(parseArgs([]string{
				"eval", "terminal", "recover", fixture.checkpoint,
				"--result", resultPath, "--evidence", evidencePath, "--quiet",
			}))
			if err == nil {
				t.Fatalf("recover accepted %s without executing a trusted verifier", name)
			}
			if data, readErr := os.ReadFile(dockerLog); readErr == nil && len(data) != 0 {
				t.Fatalf("rejected %s unexpectedly invoked Docker: %q", name, data)
			}
			after := terminalRecoveryReadFiles(t, fixture.checkpoint, "checkpoint.json", "summary.json", "task-one.json")
			if !reflect.DeepEqual(after, before) {
				t.Fatalf("rejected %s mutated the checkpoint\nbefore=%q\nafter=%q", name, before, after)
			}
		})
	}
}

func TestTerminalRecoveryCommandRejectsUnknownCompletedAndMismatchedInputsBeforeVerifier(t *testing.T) {
	tests := []struct {
		name   string
		taskID string
		bundle int
		mutate func(t *testing.T, fixture *terminalRecoveryTestFixture) string
	}{
		{name: "unknown task", taskID: "unknown-task", bundle: 1},
		{name: "completed overwrite", taskID: "task-one", bundle: 0},
		{name: "bundle content hash mismatch", taskID: "task-two", bundle: 1, mutate: func(t *testing.T, fixture *terminalRecoveryTestFixture) string {
			if err := os.WriteFile(filepath.Join(fixture.bundles[1].Dir, "tests", "unexpected.txt"), []byte("changed after checkpoint"), 0o600); err != nil {
				t.Fatalf("mutate bundle: %v", err)
			}
			return ""
		}},
		{name: "candidate result provenance mismatch", taskID: "task-two", bundle: 1, mutate: func(t *testing.T, fixture *terminalRecoveryTestFixture) string {
			path := terminalRecoveryWriteCandidateResult(t, fixture, 1, false)
			wrapper := terminalRecoveryReadObject(t, path)
			asObject(anySlice(wrapper["results"])[0])["provenance"] = map[string]any{"dataset": "forged"}
			writeTerminalTestJSON(t, path, wrapper)
			return path
		}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			fixture := newTerminalRecoveryTestFixture(t)
			manager := fixture.manager()
			if err := manager.persist(0, fixture.bundles[0], terminalRecoveryCompletedResult(true)); err != nil {
				t.Fatalf("persist checkpoint: %v", err)
			}
			before := terminalRecoveryReadFiles(t, fixture.checkpoint, "checkpoint.json", "summary.json", "task-one.json")
			resultPath := ""
			if tc.mutate != nil {
				resultPath = tc.mutate(t, fixture)
			}
			args := []string{"eval", "terminal", "recover", fixture.checkpoint, "--task-id", tc.taskID, "--container", "fixture-container", "--bundle", fixture.bundles[tc.bundle].Dir, "--quiet"}
			if resultPath != "" {
				args = append(args, "--result", resultPath)
			}
			dockerLog := terminalRecoveryInstallDockerDeny(t)
			err := recoverTerminalCheckpoint(parseArgs(args))
			if err == nil {
				t.Fatalf("recover accepted invalid %s input", tc.name)
			}
			if data, readErr := os.ReadFile(dockerLog); readErr == nil && len(data) != 0 {
				t.Fatalf("invalid recovery reached Docker/verifier: %q", data)
			}
			after := terminalRecoveryReadFiles(t, fixture.checkpoint, "checkpoint.json", "summary.json", "task-one.json")
			if !reflect.DeepEqual(after, before) {
				t.Fatalf("rejected recovery mutated checkpoint\nbefore=%q\nafter=%q", before, after)
			}
		})
	}
	t.Run("verifier command mismatch", func(t *testing.T) {
		fixture := newTerminalRecoveryTestFixture(t)
		items := anySlice(fixture.provenance["manifestItems"])
		asObject(items[1])["verifierCommand"] = "bash /tests/different-verifier.sh"
		fixture.provenance["manifestItems"] = items
		delete(fixture.provenance, "provenanceSha256")
		hash, err := terminalJSONHash(fixture.provenance)
		if err != nil {
			t.Fatalf("rehash mismatched provenance fixture: %v", err)
		}
		fixture.provenance["provenanceSha256"] = hash
		manager := fixture.manager()
		if err := manager.persist(0, fixture.bundles[0], terminalRecoveryCompletedResult(true)); err != nil {
			t.Fatalf("persist checkpoint: %v", err)
		}
		before := terminalRecoveryReadFiles(t, fixture.checkpoint, "checkpoint.json", "summary.json", "task-one.json")
		dockerLog := terminalRecoveryInstallDockerDeny(t)
		err = recoverTerminalCheckpoint(parseArgs([]string{
			"eval", "terminal", "recover", fixture.checkpoint,
			"--task-id", "task-two", "--container", "fixture-container", "--bundle", fixture.bundles[1].Dir, "--quiet",
		}))
		if err == nil {
			t.Fatal("recover accepted a bundle whose canonical verifier command differs from immutable manifest provenance")
		}
		if data, readErr := os.ReadFile(dockerLog); readErr == nil && len(data) != 0 {
			t.Fatalf("verifier-command mismatch reached Docker: %q", data)
		}
		after := terminalRecoveryReadFiles(t, fixture.checkpoint, "checkpoint.json", "summary.json", "task-one.json")
		if !reflect.DeepEqual(after, before) {
			t.Fatalf("verifier-command mismatch mutated checkpoint\nbefore=%q\nafter=%q", before, after)
		}
	})
}

func TestTerminalRecoveryCommandRejectsSymlinkAndTrailingCandidateResult(t *testing.T) {
	for _, tc := range []struct {
		name   string
		mutate func(t *testing.T, path string) string
	}{
		{name: "symlink", mutate: func(t *testing.T, path string) string {
			link := filepath.Join(t.TempDir(), "candidate-link.json")
			if err := os.Symlink(path, link); err != nil {
				t.Fatalf("create candidate symlink: %v", err)
			}
			return link
		}},
		{name: "trailing JSON", mutate: func(t *testing.T, path string) string {
			file, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0)
			if err != nil {
				t.Fatalf("open candidate result: %v", err)
			}
			if _, err := file.WriteString("{}\n"); err != nil {
				_ = file.Close()
				t.Fatalf("append candidate result: %v", err)
			}
			if err := file.Close(); err != nil {
				t.Fatalf("close candidate result: %v", err)
			}
			return path
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fixture := newTerminalRecoveryTestFixture(t)
			manager := fixture.manager()
			if err := manager.persist(0, fixture.bundles[0], terminalRecoveryCompletedResult(true)); err != nil {
				t.Fatalf("persist checkpoint: %v", err)
			}
			before := terminalRecoveryReadFiles(t, fixture.checkpoint, "checkpoint.json", "summary.json", "task-one.json")
			resultPath := tc.mutate(t, terminalRecoveryWriteCandidateResult(t, fixture, 1, false))
			dockerLog := terminalRecoveryInstallDockerDeny(t)
			err := recoverTerminalCheckpoint(parseArgs([]string{
				"eval", "terminal", "recover", fixture.checkpoint,
				"--task-id", "task-two", "--container", "fixture-container", "--bundle", fixture.bundles[1].Dir,
				"--result", resultPath, "--quiet",
			}))
			if err == nil {
				t.Fatalf("recover accepted unsafe %s candidate result", tc.name)
			}
			if data, readErr := os.ReadFile(dockerLog); readErr == nil && len(data) != 0 {
				t.Fatalf("unsafe candidate result reached Docker/verifier: %q", data)
			}
			after := terminalRecoveryReadFiles(t, fixture.checkpoint, "checkpoint.json", "summary.json", "task-one.json")
			if !reflect.DeepEqual(after, before) {
				t.Fatalf("unsafe candidate result mutated checkpoint\nbefore=%q\nafter=%q", before, after)
			}
		})
	}
}

type terminalRecoveryTestFixture struct {
	checkpoint string
	bundles    []terminalBundle
	provenance map[string]any
}

func newTerminalRecoveryTestFixture(t *testing.T) *terminalRecoveryTestFixture {
	t.Helper()
	bundles := []terminalBundle{
		terminalRecoveryBundle(t, "task-one", "TOP_SECRET_TASK_INSTRUCTION one"),
		terminalRecoveryBundle(t, "task-two", "TOP_SECRET_TASK_INSTRUCTION two"),
	}
	return &terminalRecoveryTestFixture{
		checkpoint: filepath.Join(t.TempDir(), "atomic-checkpoint"),
		bundles:    bundles,
		provenance: terminalRecoveryProvenance(t, bundles),
	}
}

func (fixture *terminalRecoveryTestFixture) manager() *terminalCheckpointManager {
	order := make([]string, len(fixture.bundles))
	for i := range fixture.bundles {
		order[i] = fixture.bundles[i].Task.ID
	}
	return &terminalCheckpointManager{
		path: fixture.checkpoint, provenance: managerSafeMap(fixture.provenance), taskOrder: order,
		entries: map[string]terminalCheckpointEntry{}, results: map[string]terminalSavedResult{},
	}
}

func terminalRecoveryBundle(t *testing.T, id, instruction string) terminalBundle {
	t.Helper()
	dir := filepath.Join(t.TempDir(), id)
	if err := os.MkdirAll(filepath.Join(dir, "tests"), 0o755); err != nil {
		t.Fatalf("create recovery bundle tests: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "tests", "test.sh"), []byte("#!/bin/sh\nexit 0\n"), 0o700); err != nil {
		t.Fatalf("write recovery verifier: %v", err)
	}
	writeTerminalTestJSON(t, filepath.Join(dir, "task.json"), terminalTask{
		ID: id, Version: "2.1", Instruction: instruction, Source: "terminal-bench/" + id,
		Agent:       terminalAgentConfig{TimeoutSec: 101, MaxTurns: 17},
		Verifier:    terminalVerifierConfig{TimeoutSec: 202, Command: "bash /tests/test.sh", RewardFile: "/logs/verifier/reward.txt"},
		Environment: terminalEnvironmentConfig{Env: map[string]string{"TOKEN": "TOP_SECRET_TASK_ENV"}},
	})
	bundle, err := loadSingleTerminalBundle(dir)
	if err != nil {
		t.Fatalf("load recovery bundle: %v", err)
	}
	bundle.BundleKey = "eval-datasets/fixture/tasks/" + id + ".tar.gz"
	bundle.ManifestIdentity = "fixture-dataset/shard/7"
	bundle.ManifestSHA256 = strings.Repeat("c", 64)
	bundle.ManifestVersion = "terminal-manifest-jsonl/v1"
	bundle.ManifestTaskIDs = []string{"task-one", "task-two"}
	return bundle
}

func terminalRecoveryProvenance(t *testing.T, bundles []terminalBundle) map[string]any {
	t.Helper()
	rawEndpoint := "https://user:model-password@model.example:8443/v1?token=model-token&x=secret-query"
	provenance, err := terminalRunProvenance(
		"fixture-dataset", 7, bundles,
		"fixture/declared", "fixture/model", "fixture-served-model", "Q4_K_M", "gguf", "fixture-runner",
		map[string]any{"hwClass": "CPU_ONLY", "cpu": "Fixture CPU", "ramGb": 32},
		map[string]any{"status": "matched", "declaredHfId": "fixture/declared", "verifiedSourceRepo": "fixture/model"},
		map[string]any{"status": "matched", "trusted": "Q4_K_M", "trustedSource": "live_endpoint"},
		map[string]any{
			"protocol": "react-shell", "agent": "fixture-agent", "commandTimeoutSeconds": 2, "endpointTimeoutSeconds": 9,
			"agentTimeoutSec": 5, "repeatBatchLimit": 3, "modelEndpoint": terminalSanitizedEndpointOrigin(rawEndpoint),
		},
	)
	if err != nil {
		t.Fatalf("terminalRunProvenance: %v", err)
	}
	return provenance
}

func terminalRecoveryCompletedResult(pass bool) terminalTaskResult {
	return terminalTaskResult{
		pass: pass, scored: true, verifierAttempted: true, verifierCompleted: true, rewardParsed: true,
		turns: 2, transcript: "bounded transcript", verifierOutput: "canonical verifier output", wallTimeMs: 123,
		lastProgressAt: "2026-07-17T01:02:03Z",
	}
}

func assertTerminalRecoveryCheckpointSnapshot(t *testing.T, root string, provenance map[string]any, wantTasks []string) {
	t.Helper()
	metadata := terminalRecoveryReadObject(t, filepath.Join(root, "checkpoint.json"))
	if int(numberField(metadata, "artifactVersion")) != terminalCheckpointArtifactVersion || int(numberField(metadata, "completedTasks")) != len(wantTasks) {
		t.Fatalf("checkpoint metadata = %#v, want v%d with %d completed tasks", metadata, terminalCheckpointArtifactVersion, len(wantTasks))
	}
	if !terminalJSONEqual(metadata["provenance"], provenance) {
		t.Fatalf("checkpoint metadata provenance differs from immutable run provenance")
	}
	entries := terminalRecoveryReadSummary(t, filepath.Join(root, "summary.json"))
	if len(entries) != len(wantTasks) {
		t.Fatalf("summary entries = %d, want %d", len(entries), len(wantTasks))
	}
	for i, taskID := range wantTasks {
		entry := entries[i]
		if entry.Task != taskID || entry.Index != i+1 || entry.Total != 2 || entry.Out != terminalCheckpointWrapperName(taskID) || entry.Scored == nil || !*entry.Scored {
			t.Fatalf("summary[%d] = %#v, want complete reference for %q", i, entry, taskID)
		}
		if filepath.IsAbs(entry.Out) || filepath.Clean(entry.Out) != entry.Out || filepath.Dir(entry.Out) != "." {
			t.Fatalf("summary wrapper reference is not a safe checkpoint-local filename: %q", entry.Out)
		}
		info, err := os.Lstat(filepath.Join(root, entry.Out))
		if err != nil || !info.Mode().IsRegular() {
			t.Fatalf("summary wrapper %q is missing or non-regular: info=%v err=%v", entry.Out, info, err)
		}
		wrapper := terminalRecoveryReadObject(t, filepath.Join(root, entry.Out))
		results := anySlice(wrapper["results"])
		if len(results) != 1 || stringValue(asObject(results[0])["question_id"]) != taskID || !terminalJSONEqual(asObject(results[0])["provenance"], provenance) {
			t.Fatalf("summary wrapper %q does not contain exactly the bound task/provenance: %#v", entry.Out, wrapper)
		}
	}
}

func terminalRecoveryReadSummary(t *testing.T, path string) []terminalCheckpointEntry {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	var entries []terminalCheckpointEntry
	if err := json.Unmarshal(data, &entries); err != nil {
		t.Fatalf("decode %s: %v", path, err)
	}
	return entries
}

func terminalRecoveryReadObject(t *testing.T, path string) map[string]any {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	var value map[string]any
	if err := json.Unmarshal(data, &value); err != nil {
		t.Fatalf("decode %s: %v", path, err)
	}
	return value
}

func terminalRecoveryReadFiles(t *testing.T, root string, names ...string) map[string]string {
	t.Helper()
	out := make(map[string]string, len(names))
	for _, name := range names {
		data, err := os.ReadFile(filepath.Join(root, name))
		if err != nil {
			t.Fatalf("read checkpoint file %s: %v", name, err)
		}
		out[name] = string(data)
	}
	return out
}

func terminalRecoveryRequireResumeError(t *testing.T, fixture *terminalRecoveryTestFixture, wantCode string) {
	t.Helper()
	_, err := newTerminalCheckpointManager(
		parseArgs([]string{"--resume", fixture.checkpoint}),
		"fixture-dataset", 7, fixture.bundles, managerSafeMap(fixture.provenance),
	)
	var cliErr cliError
	if !errors.As(err, &cliErr) || cliErr.Code != wantCode {
		t.Fatalf("resume error = %#v, want %s", err, wantCode)
	}
}

func terminalRecoveryCandidateSavedResult(fixture *terminalRecoveryTestFixture, index int, scored bool) terminalSavedResult {
	result := terminalRecoveryCompletedResult(index == 1)
	result.scored = scored
	result.verifierAttempted = scored
	result.verifierCompleted = scored
	result.rewardParsed = scored
	if !scored {
		result.verifierOutput = ""
	}
	return terminalSavedResultFromRun(fixture.bundles[index], result, fixture.provenance)
}

func terminalRecoveryWriteCandidateResult(t *testing.T, fixture *terminalRecoveryTestFixture, index int, scored bool) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), fixture.bundles[index].Task.ID+"-candidate.json")
	writeTerminalTestJSON(t, path, terminalSavedTaskFile{Results: []terminalSavedResult{terminalRecoveryCandidateSavedResult(fixture, index, scored)}})
	return path
}

func terminalRecoveryReadSavedResult(t *testing.T, path string) terminalSavedResult {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read saved result %s: %v", path, err)
	}
	var wrapper terminalSavedTaskFile
	if err := json.Unmarshal(data, &wrapper); err != nil || len(wrapper.Results) != 1 {
		t.Fatalf("decode saved result %s: results=%d err=%v", path, len(wrapper.Results), err)
	}
	return wrapper.Results[0]
}

func terminalRecoveryInstallDockerVerifier(t *testing.T) string {
	t.Helper()
	binDir := t.TempDir()
	logPath := filepath.Join(t.TempDir(), "docker.log")
	script := "#!/bin/sh\nprintf '%s\\n' \"$*\" >> \"$LMX_RECOVERY_DOCKER_LOG\"\ncase \"$*\" in\n  *'cat /logs/verifier/reward.json'*) printf '{\"reward\":1}\\n' ;;\nesac\nexit 0\n"
	if err := os.WriteFile(filepath.Join(binDir, "docker"), []byte(script), 0o700); err != nil {
		t.Fatalf("write fake recovery docker: %v", err)
	}
	t.Setenv("LMX_RECOVERY_DOCKER_LOG", logPath)
	t.Setenv("PATH", binDir+string(os.PathListSeparator)+os.Getenv("PATH"))
	return logPath
}

func terminalRecoveryInstallDockerDeny(t *testing.T) string {
	t.Helper()
	binDir := t.TempDir()
	logPath := filepath.Join(t.TempDir(), "docker-deny.log")
	script := "#!/bin/sh\nprintf '%s\\n' \"$*\" >> \"$LMX_RECOVERY_DOCKER_LOG\"\nexit 97\n"
	if err := os.WriteFile(filepath.Join(binDir, "docker"), []byte(script), 0o700); err != nil {
		t.Fatalf("write deny recovery docker: %v", err)
	}
	t.Setenv("LMX_RECOVERY_DOCKER_LOG", logPath)
	t.Setenv("PATH", binDir+string(os.PathListSeparator)+os.Getenv("PATH"))
	return logPath
}

func terminalRecoveryEvidenceForResult(t *testing.T, provenance map[string]any, bundle terminalBundle, result terminalSavedResult) map[string]any {
	t.Helper()
	resultHash, err := terminalJSONHash(result)
	if err != nil {
		t.Fatalf("hash recovered result: %v", err)
	}
	verifierHash := sha256.Sum256([]byte(result.VerifierOutput))
	return map[string]any{
		"question_id": result.QuestionID, "pass": result.Pass, "scored": true, "canonical": true,
		"verifierOutput": result.VerifierOutput, "verifierSha256": hex.EncodeToString(verifierHash[:]), "sourceResultSha256": resultHash,
		"provenanceSha256": stringValue(provenance["provenanceSha256"]), "bundleSha256": bundle.BundleSHA256,
		"verifierCommand": bundle.Task.Verifier.Command, "rewardSource": filepath.Base(bundle.Task.Verifier.RewardFile),
		"rewardValue": map[bool]float64{true: 1, false: 0}[result.Pass], "provenance": managerSafeMap(provenance),
	}
}
