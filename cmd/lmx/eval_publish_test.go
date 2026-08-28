package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

func TestDetectEvalPublishInputInfersSafeColumnMapping(t *testing.T) {
	path := filepath.Join(t.TempDir(), "network-reasoning.jsonl")
	content := "{\"question\":\"Which protocol is connection-oriented?\",\"options\":[\"TCP\",\"UDP\"],\"answer\":\"A\"}\n"
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
	input, err := detectEvalPublishInput(path, cliArgs{opts: map[string]string{}, flags: map[string]bool{}})
	if err != nil {
		t.Fatal(err)
	}
	if input.Kind != "dataset" || input.InputCol != "question" || input.ChoicesCol != "options" || input.GoldCol != "answer" {
		t.Fatalf("unexpected detection: %#v", input)
	}
}

func TestDetectEvalPublishInputRejectsMissingGoldMapping(t *testing.T) {
	path := filepath.Join(t.TempDir(), "broken.jsonl")
	if err := os.WriteFile(path, []byte("{\"prompt\":\"What is 2+2?\"}\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	_, err := detectEvalPublishInput(path, cliArgs{opts: map[string]string{}, flags: map[string]bool{}})
	var got cliError
	if !asCLIError(err, &got) || got.Code != "publish_mapping_ambiguous" {
		t.Fatalf("error = %#v, want publish_mapping_ambiguous", err)
	}
}

func TestPublishSuiteManifestDryRunAuditsAndUsesServerPreflight(t *testing.T) {
	preflightCalls := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/benchmarks/suites/dry-run" || r.Method != http.MethodPost {
			http.NotFound(w, r)
			return
		}
		preflightCalls++
		_ = json.NewEncoder(w).Encode(map[string]any{"valid": true})
	}))
	defer server.Close()

	args := parseArgs([]string{"eval", "publish", "suite.json", "--api-url", server.URL, "--api-key", "bhk_test", "--dry-run", "--kind", "qa"})
	payload := buildSuiteTemplate("safe-suite", "Safe Suite", "reasoning", "CUSTOM", "exact_match", args)
	payload["description"] = "Original arithmetic questions for deterministic reasoning checks."
	payload["sourceUrl"] = "https://example.com/methodology"
	task := evalTasks(suiteDoc(payload))[0]
	asObject(task["dataset"])["items"] = []any{map[string]any{"input": "What is 2+2?", "gold": "4"}}

	if err := publishSuiteManifest("suite.json", payload, args); err != nil {
		t.Fatal(err)
	}
	if preflightCalls != 1 {
		t.Fatalf("preflight calls = %d, want 1", preflightCalls)
	}
}

func TestValidateSuiteRemoteFallsBackToSlugGuardOnLegacyServer(t *testing.T) {
	for _, test := range []struct {
		name      string
		lookup    int
		wantError string
	}{
		{name: "available", lookup: http.StatusNotFound},
		{name: "duplicate", lookup: http.StatusOK, wantError: "suite_slug_exists"},
	} {
		t.Run(test.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Method == http.MethodPost {
					w.WriteHeader(http.StatusMethodNotAllowed)
					return
				}
				w.WriteHeader(test.lookup)
				if test.lookup == http.StatusOK {
					_ = json.NewEncoder(w).Encode(map[string]any{"slug": "legacy-suite"})
				}
			}))
			defer server.Close()
			args := parseArgs([]string{"eval", "publish", "suite.json", "--api-url", server.URL, "--api-key", "bhk_test"})
			err := validateSuiteRemote(map[string]any{"slug": "legacy-suite"}, args)
			if test.wantError == "" && err != nil {
				t.Fatal(err)
			}
			if test.wantError != "" {
				var got cliError
				if !asCLIError(err, &got) || got.Code != test.wantError {
					t.Fatalf("error = %#v, want %s", err, test.wantError)
				}
			}
		})
	}
}

func TestPublishTerminalInputRequiresPublicProvenance(t *testing.T) {
	args := parseArgs([]string{"eval", "publish", "tasks", "--api-key", "bhk_test", "--description", "Original repository repair tasks for API correctness."})
	err := publishTerminalInput(evalPublishInput{Kind: "terminal-imported", Path: "tasks"}, args)
	var got cliError
	if !asCLIError(err, &got) || got.Code != "source_url_required" {
		t.Fatalf("error = %#v, want source_url_required", err)
	}
}

func TestTerminalPublisherRejectsSingleTaskBeforeUpload(t *testing.T) {
	dir := t.TempDir()
	if err := os.Mkdir(filepath.Join(dir, "tests"), 0o755); err != nil {
		t.Fatal(err)
	}
	task := map[string]any{"id": "only-task", "instruction": "Repair the application.", "image": map[string]any{"prebuilt": "example/image:latest"}}
	data, _ := json.Marshal(task)
	if err := os.WriteFile(filepath.Join(dir, "task.json"), data, 0o600); err != nil {
		t.Fatal(err)
	}
	args := parseArgs([]string{"eval", "terminal", "publish", dir, "--slug", "single-task", "--name", "Single Task", "--api-key", "bhk_test", "--skip-oracle"})
	err := publishTerminalDataset(args)
	var got cliError
	if !asCLIError(err, &got) || got.Code != "terminal_dataset_too_small" {
		t.Fatalf("error = %#v, want terminal_dataset_too_small", err)
	}
}

func TestTerminalPublisherChecksProAccessBeforeOracle(t *testing.T) {
	parent := t.TempDir()
	for _, id := range []string{"task-one", "task-two"} {
		dir := filepath.Join(parent, id)
		if err := os.MkdirAll(filepath.Join(dir, "tests"), 0o755); err != nil {
			t.Fatal(err)
		}
		data, _ := json.Marshal(map[string]any{"id": id, "instruction": "Repair the application.", "image": map[string]any{"prebuilt": "example/image:latest"}})
		if err := os.WriteFile(filepath.Join(dir, "task.json"), data, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, `{"error":"Pro subscription required"}`, http.StatusForbidden)
	}))
	defer server.Close()
	args := parseArgs([]string{"eval", "terminal", "publish", parent, "--slug", "access-check", "--name", "Access Check", "--api-key", "bhk_test", "--api-url", server.URL})
	err := publishTerminalDataset(args)
	var got cliError
	if !asCLIError(err, &got) || got.Code != "terminal_publish_access_failed" {
		t.Fatalf("error = %#v, want terminal_publish_access_failed", err)
	}
}

func TestEvalPublishReportsBadPathBeforeAuthentication(t *testing.T) {
	args := parseArgs([]string{"eval", "publish", "/definitely/missing/benchmark.jsonl"})
	err := handleEvalPublish("/definitely/missing/benchmark.jsonl", args)
	var got cliError
	if !asCLIError(err, &got) || got.Code != "publish_input_unreadable" {
		t.Fatalf("error = %#v, want publish_input_unreadable", err)
	}
}

func TestEvalPublishBooleanGuardrailsDoNotConsumeInput(t *testing.T) {
	args := parseArgs([]string{"eval", "publish", "--strict", "--no-upload-datasets", "questions.jsonl"})
	if got := positional(args, 2); got != "questions.jsonl" {
		t.Fatalf("publish input = %q, want questions.jsonl", got)
	}
	if !hasFlag(args, "strict") || !hasFlag(args, "no-upload-datasets") {
		t.Fatal("publish guardrail flags were not parsed as booleans")
	}
}

func asCLIError(err error, target *cliError) bool {
	if err == nil {
		return false
	}
	value, ok := err.(cliError)
	if ok {
		*target = value
	}
	return ok
}
