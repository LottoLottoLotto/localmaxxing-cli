package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLmEvalMetricSelection(t *testing.T) {
	cases := []struct {
		name   string
		result map[string]any
		score  float64
		found  bool
	}{
		{"fallback accuracy", map[string]any{"acc,none": 0.75}, 0.75, true},
		{"preferred zero is a real score", map[string]any{"acc_norm,none": 0.0, "acc,none": 0.75}, 0, true},
		{"standard error is not a score", map[string]any{"acc_stderr,none": 0.1}, 0, false},
		{"metadata is not a score", map[string]any{"alias": "gpu_arithmetic"}, 0, false},
		{"filtered metric ignores metadata", map[string]any{"alias": "gpu_arithmetic", "exact_match,numeric": 1.0, "exact_match_stderr,numeric": 0.0}, 1, true},
		{"invalid preferred metric falls back", map[string]any{"acc_norm,none": "unavailable", "acc,none": 0.75}, 0.75, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			score, found := scoreFromLmEvalTask(tc.result, "exact_match")
			if score != tc.score || found != tc.found {
				t.Fatalf("score = %v, found = %v; want %v, %v", score, found, tc.score, tc.found)
			}
		})
	}
}

func TestLmEvalResultDiscovery(t *testing.T) {
	cases := []struct {
		name      string
		files     []string
		wantFile  string
		errorCode string
	}{
		{"timestamped aggregate", []string{"results_2026-10-03T12-00-00.json", "samples_task.jsonl", "config.json"}, "results_2026-10-03T12-00-00.json", ""},
		{"exact aggregate", []string{"results.json"}, "results.json", ""},
		{"no aggregate", []string{"samples_task.jsonl", "config.json"}, "", "lm_eval_results_missing"},
		{"multiple aggregates", []string{"results.json", "results_2026-10-03T12-00-00.json"}, "", "lm_eval_results_ambiguous"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			parent := t.TempDir()
			// A previous destination must never satisfy this invocation's output.
			if err := os.WriteFile(filepath.Join(parent, "results.json"), []byte(`{"results":{"task":{"acc,none":0.25}}}`), 0o600); err != nil {
				t.Fatal(err)
			}
			outputDir := filepath.Join(parent, "current")
			if err := os.Mkdir(outputDir, 0o700); err != nil {
				t.Fatal(err)
			}
			for _, name := range tc.files {
				if err := os.WriteFile(filepath.Join(outputDir, name), []byte(`{"results":{"task":{"acc,none":0.75}}}`), 0o600); err != nil {
					t.Fatal(err)
				}
			}
			resultFile, err := lmEvalResultFile(outputDir)
			if tc.errorCode != "" {
				if failure, ok := err.(cliError); !ok || failure.Code != tc.errorCode || resultFile != "" {
					t.Fatalf("result = %q, error = %v; want %s", resultFile, err, tc.errorCode)
				}
				return
			}
			if err != nil || resultFile != filepath.Join(outputDir, tc.wantFile) {
				t.Fatalf("result = %q, error = %v; want current %s", resultFile, err, tc.wantFile)
			}
		})
	}
}
