package main

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"sync/atomic"
	"testing"
)

func TestBenchmarkSubmissionPatchRepeatedSetIssue21(t *testing.T) {
	assignments := []string{
		"tensorParallel=2",
		"gpuMemUtil=0.94",
		"kvCacheDtype=bfloat16",
		"prefixCaching=true",
		"chunkedPrefill=true",
		"prefillChunkSize=4096",
		"maxRunningSeqs=6",
	}
	want := map[string]any{
		"tensorParallel":   float64(2),
		"gpuMemUtil":       0.94,
		"kvCacheDtype":     "bfloat16",
		"prefixCaching":    true,
		"chunkedPrefill":   true,
		"prefillChunkSize": float64(4096),
		"maxRunningSeqs":   float64(6),
	}
	for _, syntax := range []string{"separate", "equals", "mixed"} {
		t.Run(syntax, func(t *testing.T) {
			argv := []string{"speed-test", "submissions", "edit", "run_123"}
			for i, assignment := range assignments {
				if syntax == "equals" || syntax == "mixed" && i%2 == 0 {
					argv = append(argv, "--set="+assignment)
				} else {
					argv = append(argv, "--set", assignment)
				}
			}
			argv = append(argv, "--json")
			patch, err := benchmarkSubmissionPatch(parseArgs(argv))
			if err != nil {
				t.Fatalf("benchmarkSubmissionPatch: %v", err)
			}
			if !reflect.DeepEqual(patch, want) {
				t.Fatalf("patch = %#v, want %#v", patch, want)
			}
		})
	}
}

func TestBenchmarkSubmissionPatchNestedEngineFlags(t *testing.T) {
	const body = `{"contextLength":262144,"engineQuant":"modelopt_mixed","engineFlags":{"tensorParallel":2,"gpuMemUtil":0.94,"kvCacheDtype":"bfloat16","prefixCaching":true,"commandSnippet":"vllm serve org/model --tensor-parallel-size=2 --gpu-memory-utilization=0.94"}}`
	want := map[string]any{
		"contextLength":  float64(262144),
		"engineQuant":    "modelopt_mixed",
		"tensorParallel": float64(2),
		"gpuMemUtil":     0.94,
		"kvCacheDtype":   "bfloat16",
		"prefixCaching":  true,
		"commandSnippet": "vllm serve org/model --tensor-parallel-size=2 --gpu-memory-utilization=0.94",
	}
	path := filepath.Join(t.TempDir(), "patch.json")
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name string
		argv []string
	}{
		{"set-json", []string{"--set-json", body}},
		{"patch-file", []string{"--patch", path}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			patch, err := benchmarkSubmissionPatch(parseArgs(tc.argv))
			if err != nil {
				t.Fatalf("benchmarkSubmissionPatch: %v", err)
			}
			if !reflect.DeepEqual(patch, want) {
				t.Fatalf("patch = %#v, want flattened flags %#v", patch, want)
			}
		})
	}
}

func TestBenchmarkSubmissionPatchSourcePrecedence(t *testing.T) {
	path := filepath.Join(t.TempDir(), "patch.json")
	if err := os.WriteFile(path, []byte(`{"notes":"patch","contextLength":4096,"engineFlags":{"tensorParallel":1,"gpuMemUtil":0.5,"kvCacheDtype":"bfloat16","prefixCaching":true}}`), 0o600); err != nil {
		t.Fatal(err)
	}
	patch, err := benchmarkSubmissionPatch(parseArgs([]string{
		"--set", "tensorParallel=3",
		"--set", "notes=first",
		"--set-json", `{"contextLength":262144,"tensorParallel":2,"engineFlags":{"gpuMemUtil":0.94,"prefixCaching":false}}`,
		"--set=notes=corrected=after profiling",
		"--set=tensorParallel=4",
		"--patch", path,
	}))
	if err != nil {
		t.Fatalf("benchmarkSubmissionPatch: %v", err)
	}
	want := map[string]any{
		"notes":          "corrected=after profiling",
		"contextLength":  float64(262144),
		"tensorParallel": float64(4),
		"gpuMemUtil":     0.94,
		"kvCacheDtype":   "bfloat16",
		"prefixCaching":  false,
	}
	if !reflect.DeepEqual(patch, want) {
		t.Fatalf("patch = %#v, want patch < set-json < ordered set: %#v", patch, want)
	}
}

func TestBenchmarkSubmissionPatchPreservesExplicitClearsAndZero(t *testing.T) {
	patch, err := benchmarkSubmissionPatch(parseArgs([]string{
		"--set-json", `{"gpuPowerWatts":null,"prefillTokens":0,"engineFlags":{"specDecoding":false,"specAcceptedTokens":0,"specAcceptanceRate":0,"cpuOffloadGb":0,"kvCacheDtype":null}}`,
		"--set", "prefixCaching=false",
		"--set=temperature=0",
		"--set", "notes=null",
	}))
	if err != nil {
		t.Fatalf("benchmarkSubmissionPatch: %v", err)
	}
	want := map[string]any{
		"gpuPowerWatts":      nil,
		"prefillTokens":      float64(0),
		"specDecoding":       false,
		"specAcceptedTokens": float64(0),
		"specAcceptanceRate": float64(0),
		"cpuOffloadGb":       float64(0),
		"kvCacheDtype":       nil,
		"prefixCaching":      false,
		"temperature":        float64(0),
		"notes":              nil,
	}
	if !reflect.DeepEqual(patch, want) {
		t.Fatalf("patch = %#v, want explicit values %#v", patch, want)
	}
}

func TestBenchmarkSubmissionPatchRejectsInvalidEdits(t *testing.T) {
	for _, tc := range []struct {
		name string
		argv []string
	}{
		{"missing-edit", nil},
		{"missing-assignment", []string{"--set"}},
		{"empty-assignment", []string{"--set="}},
		{"missing-before-option", []string{"--set", "--json"}},
		{"malformed-before-valid", []string{"--set", "tensorParallel", "--set=tensorParallel=2"}},
		{"malformed-trailing", []string{"--set", "tensorParallel=2", "--set", "gpuMemUtil"}},
		{"missing-trailing", []string{"--set=tensorParallel=2", "--set"}},
		{"empty-field", []string{"--set", " =2"}},
		{"unknown-set-field", []string{"--set", "tensorParalell=2"}},
		{"dotted-field", []string{"--set", "engineFlags.tensorParallel=2"}},
		{"unknown-root", []string{"--set-json", `{"tensorParalell":2}`}},
		{"unknown-flag", []string{"--set-json", `{"engineFlags":{"tensorParalell":2}}`}},
		{"root-field-as-flag", []string{"--set-json", `{"engineFlags":{"notes":"wrong level"}}`}},
		{"null-flags", []string{"--set-json", `{"notes":"valid","engineFlags":null}`}},
		{"string-flags", []string{"--set-json", `{"engineFlags":"tensorParallel=2"}`}},
		{"array-flags", []string{"--set-json", `{"engineFlags":[{"tensorParallel":2}]}`}},
		{"boolean-flags", []string{"--set-json", `{"engineFlags":false}`}},
		{"number-flags", []string{"--set-json", `{"engineFlags":2}`}},
		{"empty-flags", []string{"--set-json", `{"notes":"valid","engineFlags":{}}`}},
		{"flat-nested-conflict", []string{"--set-json", `{"tensorParallel":1,"engineFlags":{"tensorParallel":2}}`}},
		{"flat-nested-identical", []string{"--set-json", `{"prefixCaching":false,"engineFlags":{"prefixCaching":false}}`}},
		{"nonobject-json", []string{"--set-json", `[]`}},
		{"null-json", []string{"--set-json", `null`}},
		{"malformed-json", []string{"--set-json", `{"tensorParallel":`}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if patch, err := benchmarkSubmissionPatch(parseArgs(tc.argv)); err == nil {
				t.Fatalf("invalid edit accepted: %#v", patch)
			}
		})
	}
}

func TestBenchmarkSubmissionInvalidPatchFileDoesNotSendRequest(t *testing.T) {
	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests.Add(1)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"run_123"}`))
	}))
	defer server.Close()
	path := filepath.Join(t.TempDir(), "patch.json")
	if err := os.WriteFile(path, []byte(`{"tensorParallel":1,"engineFlags":{"tensorParallel":2}}`), 0o600); err != nil {
		t.Fatal(err)
	}
	args := parseArgs([]string{
		"--api-url", server.URL, "--api-key", "bhk_test", "--quiet",
		"--patch", path, "--set", "tensorParallel=4",
	})
	if err := handleBenchmarkSubmissions("edit", "run_123", args); err == nil {
		t.Fatal("ambiguous patch file accepted despite a later override")
	}
	if got := requests.Load(); got != 0 {
		t.Fatalf("invalid edit sent %d requests, want zero", got)
	}
}

func TestEditBenchmarkRunPersistsRepeatedAssignments(t *testing.T) {
	path := filepath.Join(t.TempDir(), "run.json")
	if err := os.WriteFile(path, []byte(`{"id":"saved-envelope","payload":{"hfId":"org/model","engineName":"vllm","tokSOut":1,"notes":"original"}}`), 0o600); err != nil {
		t.Fatal(err)
	}
	args := parseArgs([]string{
		"--quiet", "--api-key", "bhk_test",
		"--set", "tokSOut=93.5", "--set=notes=first",
		"--set", "prefillTokens=0", "--set=notes=corrected=metadata",
		"--set", "engineFlags={\"prefixCaching\":false,\"kvCacheDtype\":null}",
	})
	if err := editBenchmarkRun(path, args); err != nil {
		t.Fatalf("editBenchmarkRun: %v", err)
	}
	updated, err := readJSON(path)
	if err != nil {
		t.Fatal(err)
	}
	envelope := asObject(updated)
	payload := asObject(envelope["payload"])
	want := map[string]any{
		"hfId":          "org/model",
		"engineName":    "vllm",
		"tokSOut":       93.5,
		"notes":         "corrected=metadata",
		"prefillTokens": float64(0),
		"engineFlags":   map[string]any{"prefixCaching": false, "kvCacheDtype": nil},
	}
	for field, value := range want {
		if got, ok := payload[field]; !ok || !reflect.DeepEqual(got, value) {
			t.Errorf("saved %s = %#v (present %v), want %#v", field, got, ok, value)
		}
	}
	if envelope["id"] != "saved-envelope" {
		t.Fatalf("edit lost envelope metadata: %#v", envelope)
	}
}

func TestEditBenchmarkRunRejectsTrailingAssignmentWithoutRewriting(t *testing.T) {
	for _, trailing := range [][]string{{"--set", "notes"}, {"--set"}, {"--set="}} {
		t.Run(trailing[len(trailing)-1], func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "run.json")
			original := []byte("{ \"hfId\": \"org/model\", \"tokSOut\": 42, \"notes\": \"unchanged\" }\n")
			if err := os.WriteFile(path, original, 0o600); err != nil {
				t.Fatal(err)
			}
			argv := []string{"--quiet", "--api-key", "bhk_test", "--set-json", `{"notes":"must not persist"}`, "--set", "tokSOut=93.5"}
			argv = append(argv, trailing...)
			if err := editBenchmarkRun(path, parseArgs(argv)); err == nil {
				t.Fatal("invalid trailing assignment accepted")
			}
			got, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(got, original) {
				t.Fatalf("invalid edit rewrote saved run: %s", got)
			}
		})
	}
}
