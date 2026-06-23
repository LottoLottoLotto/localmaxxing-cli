package main

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// echoLogprobServer returns an OpenAI-compatible /v1/completions stub that echoes
// per-token logprobs, scoring the final continuation token by whether the prompt
// resolves the embedded arithmetic correctly.
func echoLogprobServer(t *testing.T) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/completions" {
			http.NotFound(w, r)
			return
		}
		var req struct {
			Prompt    []string `json:"prompt"`
			Echo      bool     `json:"echo"`
			MaxTokens int      `json:"max_tokens"`
			Logprobs  int      `json:"logprobs"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode completions request: %v", err)
		}
		if !req.Echo {
			t.Errorf("expected echo:true, got false")
		}
		if req.MaxTokens != 0 {
			t.Errorf("expected max_tokens 0, got %d", req.MaxTokens)
		}
		choices := make([]map[string]any, len(req.Prompt))
		for i, p := range req.Prompt {
			lp := -5.0
			if strings.HasSuffix(p, " 4") {
				lp = -0.1
			}
			choices[i] = map[string]any{
				"index": i,
				"text":  p,
				"logprobs": map[string]any{
					"token_logprobs": []any{nil, lp},
					"text_offset":    []any{0, len(p) - 2},
				},
			}
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"choices": choices})
	}))
}

func TestScoreContinuationsLogprobSumsOnlyContinuation(t *testing.T) {
	srv := echoLogprobServer(t)
	defer srv.Close()
	ctx := "Q: 2+2=?\nAnswer:"
	sums, byteLens, err := scoreContinuationsLogprob(srv.URL, "m", "", ctx, []string{" 3", " 4", " 5"})
	if err != nil {
		t.Fatalf("scoreContinuationsLogprob: %v", err)
	}
	if len(sums) != 3 {
		t.Fatalf("len(sums) = %d, want 3", len(sums))
	}
	if sums[1] != -0.1 {
		t.Errorf("sums[1] = %v, want -0.1 (continuation token only)", sums[1])
	}
	if sums[0] != -5 {
		t.Errorf("sums[0] = %v, want -5", sums[0])
	}
	if byteLens[1] != 2 {
		t.Errorf("byteLens[1] = %d, want 2", byteLens[1])
	}
}

func TestScoreLoglikelihoodItemPicksGold(t *testing.T) {
	srv := echoLogprobServer(t)
	defer srv.Close()
	doc := map[string]any{"runConfig": map[string]any{"loglikelihoodTarget": "choice_text", "loglikelihoodNorm": "byte"}}
	item := map[string]any{"input": "2+2=?", "choices": []any{"3", "4", "5"}, "gold": "B"}
	ctx := renderEvalPrompt("Q: {{input}}\nAnswer:", item)
	score, predicted, gold, err := scoreLoglikelihoodItem(srv.URL, "m", "", doc, item, ctx)
	if err != nil {
		t.Fatalf("scoreLoglikelihoodItem: %v", err)
	}
	if score != 1 || predicted != "B" || gold != "B" {
		t.Fatalf("score=%v predicted=%q gold=%q, want 1/B/B", score, predicted, gold)
	}
}

func TestScoreLoglikelihoodItemRequiresChoicesAndGold(t *testing.T) {
	srv := echoLogprobServer(t)
	defer srv.Close()
	doc := map[string]any{}
	if _, _, _, err := scoreLoglikelihoodItem(srv.URL, "m", "", doc, map[string]any{"input": "x", "gold": "A"}, "ctx"); err == nil {
		t.Fatal("expected error for missing choices")
	}
	if _, _, _, err := scoreLoglikelihoodItem(srv.URL, "m", "", doc, map[string]any{"input": "x", "choices": []any{"a", "b"}}, "ctx"); err == nil {
		t.Fatal("expected error for missing gold")
	}
}

func TestRunCustomLocalEvalLoglikelihood(t *testing.T) {
	srv := echoLogprobServer(t)
	defer srv.Close()
	suite := map[string]any{
		"slug":   "arith-ll",
		"runner": "CUSTOM",
		"suiteDoc": map[string]any{
			"version": "1.0", "runner": "custom", "scoringMethod": "loglikelihood",
			"higherIsBetter": true, "aggregation": "weighted_mean",
			"runConfig": map[string]any{"loglikelihoodTarget": "choice_text", "loglikelihoodNorm": "byte"},
			"tasks": []any{map[string]any{
				"key": "arith", "displayName": "Arithmetic", "taskType": "multiple_choice", "weight": 1,
				"promptTemplate": "Q: {{input}}\nAnswer:",
				"dataset": map[string]any{"source": "inline", "items": []any{
					map[string]any{"input": "2+2=?", "choices": []any{"3", "4", "5"}, "gold": "B"},
					map[string]any{"input": "1+3=?", "choices": []any{"2", "4", "6"}, "gold": "B"},
				}},
			}},
		},
	}
	args := cliArgs{opts: map[string]string{"model": "m", "base-url": srv.URL}, flags: map[string]bool{"quiet": true}}
	result, err := runCustomLocalEval(suite, args)
	if err != nil {
		t.Fatalf("runCustomLocalEval: %v", err)
	}
	scores := asObject(result["scores"])
	arith := asObject(scores["arith"])
	if numberField(arith, "score") != 1 {
		t.Fatalf("arith score = %v, want 1 (both gold are choice 'B' = '4')", arith["score"])
	}
}

func TestHandleEvalPullWritesOfflineCopyWithGold(t *testing.T) {
	dataset := `{"input":"2+2=?","choices":["3","4","5"],"gold":"B"}` + "\n" +
		`{"input":"1+3=?","choices":["2","4","6"],"gold":"B"}` + "\n"
	var srv *httptest.Server
	srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.HasSuffix(r.URL.Path, "/run-bundle"):
			_ = json.NewEncoder(w).Encode(map[string]any{
				"suite": map[string]any{"slug": "arith-ll", "name": "Arith", "runner": "CUSTOM"},
				"suiteDoc": map[string]any{
					"version": "1.0", "runner": "custom", "scoringMethod": "loglikelihood",
					"tasks": []any{map[string]any{
						"key": "arith", "displayName": "Arith", "taskType": "multiple_choice",
						"promptTemplate": "Q: {{input}}\nAnswer:",
						"dataset":        map[string]any{"source": "bucket", "storageRef": map[string]any{"storageKey": "k", "format": "jsonl"}},
					}},
				},
				"tasks": []any{map[string]any{
					"key":     "arith",
					"dataset": map[string]any{"source": "bucket", "downloadUrl": srv.URL + "/dataset.jsonl", "storageRef": map[string]any{"storageKey": "k", "format": "jsonl"}},
				}},
			})
		case r.URL.Path == "/dataset.jsonl":
			_, _ = io.WriteString(w, dataset)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	outDir := filepath.Join(t.TempDir(), "pulled")
	args := cliArgs{opts: map[string]string{"api-url": srv.URL, "api-key": "bhk_test", "out": outDir}, flags: map[string]bool{"quiet": true}}
	if err := handleEvalPull("arith-ll", args); err != nil {
		t.Fatalf("handleEvalPull: %v", err)
	}

	// Per-task JSONL with gold is written for inspection.
	jsonlBytes, err := os.ReadFile(filepath.Join(outDir, "arith.jsonl"))
	if err != nil {
		t.Fatalf("read arith.jsonl: %v", err)
	}
	if !strings.Contains(string(jsonlBytes), `"gold":"B"`) {
		t.Fatalf("arith.jsonl missing gold labels: %s", jsonlBytes)
	}

	// suite.json embeds the dataset inline so it runs offline.
	suiteValue, err := readJSON(filepath.Join(outDir, "suite.json"))
	if err != nil {
		t.Fatalf("read suite.json: %v", err)
	}
	suite := asObject(suiteValue)
	doc := suiteDoc(suite)
	task := evalTasks(doc)[0]
	ds := asObject(task["dataset"])
	if stringValue(ds["source"]) != "inline" {
		t.Fatalf("pulled dataset source = %q, want inline", ds["source"])
	}
	if got := len(anySlice(ds["items"])); got != 2 {
		t.Fatalf("pulled inline items = %d, want 2", got)
	}
}

func TestHandleEvalSubmitPostsSavedRun(t *testing.T) {
	var posted map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/evals/runs/dry-run" {
			http.NotFound(w, r)
			return
		}
		_ = json.NewDecoder(r.Body).Decode(&posted)
		_ = json.NewEncoder(w).Encode(map[string]any{"ok": true, "status": "valid"})
	}))
	defer srv.Close()

	runFile := filepath.Join(t.TempDir(), "run.json")
	if err := writeJSON(runFile, map[string]any{
		"suiteSlug":     "arith-ll",
		"hfId":          "<required-before-submit>",
		"executionMode": "CUSTOM_LOCAL",
		"results":       map[string]any{"arith": map[string]any{"score": 1.0}},
	}); err != nil {
		t.Fatalf("writeJSON: %v", err)
	}

	hwFile := filepath.Join(t.TempDir(), "hw.json")
	if err := writeJSON(hwFile, map[string]any{"gpus": []any{map[string]any{"name": "RTX 3090"}}}); err != nil {
		t.Fatalf("writeJSON hw: %v", err)
	}

	// Missing model (placeholder hfId) must fail before any request.
	noModel := cliArgs{opts: map[string]string{"api-url": srv.URL, "api-key": "bhk_test", "hardware": hwFile}, flags: map[string]bool{"dry-run": true, "quiet": true}}
	requireCliErrorCode(t, handleEvalSubmit(runFile, noModel), "missing_model")

	// Filling model + hardware via flags submits the saved payload.
	args := cliArgs{opts: map[string]string{"api-url": srv.URL, "api-key": "bhk_test", "model": "Qwen/Qwen3-8B", "hardware": hwFile}, flags: map[string]bool{"dry-run": true, "quiet": true}}
	if err := handleEvalSubmit(runFile, args); err != nil {
		t.Fatalf("handleEvalSubmit: %v", err)
	}
	if stringValue(posted["hfId"]) != "Qwen/Qwen3-8B" {
		t.Fatalf("posted hfId = %v, want filled from --model", posted["hfId"])
	}
	if posted["hardware"] == nil {
		t.Fatalf("posted payload missing hardware")
	}
	if asObject(posted["results"]) == nil {
		t.Fatalf("posted payload missing results")
	}
}
func TestExtractAnswerLastNumber(t *testing.T) {
	cases := map[string]string{
		"Natalia sold 48 clips in April and 24 in May, so 48 + 24 = 72 clips.": "72",
		"The total cost is $1,234.50 after tax.":                               "$1,234.50",
		"Temperature dropped to -5 degrees.":                                   "-5",
		"That is a 50% increase.":                                              "50%",
		"no digits here":                                                       "no digits here",
	}
	for response, want := range cases {
		if got := extractAnswer(response, "last_number", ""); got != want {
			t.Errorf("extractAnswer(%q) = %q, want %q", response, got, want)
		}
	}
}

func TestExtractAnswerRegex(t *testing.T) {
	got := extractAnswer("Reasoning... Final answer: (C)", "regex", `answer:\s*\(([A-D])\)`)
	if got != "C" {
		t.Errorf("regex extraction = %q, want C", got)
	}
	// Bad pattern falls back to the raw response.
	if got := extractAnswer("x", "regex", "("); got != "x" {
		t.Errorf("bad regex should fall back to response, got %q", got)
	}
}

func TestAnswersMatchNumericAndFormatting(t *testing.T) {
	matching := [][2]string{{"72", "72"}, {"18.0", "18"}, {"$1,000", "1000"}, {"50%", "50"}, {"  72 ", "72"}}
	for _, pair := range matching {
		if !answersMatch(pair[0], pair[1]) {
			t.Errorf("answersMatch(%q,%q) = false, want true", pair[0], pair[1])
		}
	}
	if answersMatch("73", "72") {
		t.Error("answersMatch(73,72) = true, want false")
	}
}

func TestRunCustomLocalEvalExactMatchExtractsFinalNumber(t *testing.T) {
	// Stub returns a chain-of-thought ending in the correct sum (no answer-only prompting).
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			http.NotFound(w, r)
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []any{map[string]any{"message": map[string]any{"content": "Let me think. 48 + 24 = 72. The answer is 72."}}},
		})
	}))
	defer srv.Close()
	suite := map[string]any{
		"slug":   "math-oneoff",
		"runner": "CUSTOM",
		"suiteDoc": map[string]any{
			"version": "1.0", "runner": "custom", "scoringMethod": "exact_match",
			"higherIsBetter": true, "aggregation": "weighted_mean",
			"tasks": []any{map[string]any{
				"key": "math", "displayName": "Grade-school math", "taskType": "qa", "weight": 1,
				"promptTemplate":   "Solve step by step, then state the answer.\n\n{{input}}",
				"answerExtraction": "last_number",
				"maxNewTokens":     256,
				"dataset": map[string]any{"source": "inline", "items": []any{
					map[string]any{"input": "Natalia sold 48 clips in April and half as many in May. How many total?", "gold": "72"},
				}},
			}},
		},
	}
	args := cliArgs{opts: map[string]string{"model": "m", "base-url": srv.URL}, flags: map[string]bool{"quiet": true}}
	result, err := runCustomLocalEval(suite, args)
	if err != nil {
		t.Fatalf("runCustomLocalEval: %v", err)
	}
	math := asObject(asObject(result["scores"])["math"])
	if numberField(math, "score") != 1 {
		t.Fatalf("math score = %v, want 1 (extracted 72 from chain-of-thought)", math["score"])
	}
	// The extracted answer is recorded on the artifact for auditability.
	arts := anySlice(result["artifacts"])
	if got := stringValue(asObject(arts[0])["extractedAnswer"]); got != "72" {
		t.Fatalf("artifact extractedAnswer = %q, want 72", got)
	}
}

func TestHandleEvalRunSubmitsScoresAndArtifactsForExtractedMath(t *testing.T) {
	var posted map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/chat/completions":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"choices": []any{map[string]any{"message": map[string]any{"content": "We compute 48 + 24 = 72, so the final answer is 72."}}},
			})
		case "/api/evals/runs/dry-run":
			_ = json.NewDecoder(r.Body).Decode(&posted)
			_ = json.NewEncoder(w).Encode(map[string]any{"ok": true, "status": "valid"})
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	tmp := t.TempDir()
	suiteFile := filepath.Join(tmp, "suite.json")
	if err := writeJSON(suiteFile, map[string]any{
		"slug":   "math-oneoff",
		"name":   "Math Oneoff",
		"runner": "CUSTOM",
		"suiteDoc": map[string]any{
			"version": "1.0", "runner": "custom", "scoringMethod": "exact_match",
			"higherIsBetter": true, "aggregation": "mean",
			"tasks": []any{map[string]any{
				"key": "math", "displayName": "Math", "taskType": "qa", "weight": 1,
				"promptTemplate":   "Solve step by step.\n\n{{input}}",
				"answerExtraction": "last_number",
				"maxNewTokens":     256,
				"dataset": map[string]any{"source": "inline", "items": []any{
					map[string]any{"input": "Natalia sold 48 clips in April and 24 in May. How many total?", "gold": "72"},
				}},
			}},
		},
	}); err != nil {
		t.Fatalf("write suite: %v", err)
	}
	hwFile := filepath.Join(tmp, "hardware.json")
	if err := writeJSON(hwFile, map[string]any{"gpus": []any{map[string]any{"name": "RTX 3090"}}}); err != nil {
		t.Fatalf("write hardware: %v", err)
	}
	outFile := filepath.Join(tmp, "run.json")
	args := cliArgs{
		opts: map[string]string{
			"suite-file": suiteFile,
			"model":      "Qwen/Qwen3-8B",
			"base-url":   srv.URL,
			"api-url":    srv.URL,
			"api-key":    "bhk_test",
			"hardware":   hwFile,
			"out":        outFile,
		},
		flags: map[string]bool{"dry-run": true, "quiet": true},
	}
	if err := handleEvalRun("math-oneoff", args); err != nil {
		t.Fatalf("handleEvalRun: %v", err)
	}
	if posted == nil {
		t.Fatal("dry-run endpoint was not called")
	}
	results := asObject(posted["results"])
	mathScore := asObject(results["math"])
	if numberField(mathScore, "score") != 1 {
		t.Fatalf("posted math score = %v, want 1", mathScore["score"])
	}
	artifacts := anySlice(posted["artifacts"])
	if len(artifacts) != 1 {
		t.Fatalf("posted artifacts len = %d, want 1", len(artifacts))
	}
	artifact := asObject(artifacts[0])
	if stringValue(artifact["extractedAnswer"]) != "72" {
		t.Fatalf("posted extractedAnswer = %q, want 72", artifact["extractedAnswer"])
	}
	if stringValue(artifact["response"]) == "" || stringValue(artifact["prompt"]) == "" || numberField(artifact, "score") != 1 {
		t.Fatalf("posted artifact missing response/prompt/score: %#v", artifact)
	}
	saved, err := readJSON(outFile)
	if err != nil {
		t.Fatalf("read saved run: %v", err)
	}
	savedArtifacts := anySlice(asObject(saved)["artifacts"])
	if len(savedArtifacts) != 1 || stringValue(asObject(savedArtifacts[0])["extractedAnswer"]) != "72" {
		t.Fatalf("saved run artifacts missing extracted answer: %#v", savedArtifacts)
	}
}

func TestEvalSubmitResolvesServedModelAliasToHFID(t *testing.T) {
	var posted map[string]any
	var searchQuery string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/models":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"data": []any{map[string]any{"id": "gemma-4-12b-it"}},
			})
		case "/api/models/search":
			searchQuery = r.URL.Query().Get("q")
			_ = json.NewEncoder(w).Encode(map[string]any{
				"models": []any{map[string]any{"hfId": "google/gemma-3-12b-it"}},
			})
		case "/api/evals/runs/dry-run":
			_ = json.NewDecoder(r.Body).Decode(&posted)
			_ = json.NewEncoder(w).Encode(map[string]any{"ok": true, "status": "valid"})
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	tmp := t.TempDir()
	runFile := filepath.Join(tmp, "run.json")
	if err := writeJSON(runFile, map[string]any{
		"suiteSlug":     "gsm8k-sample",
		"hfId":          "gemma-4-12b-it",
		"executionMode": "CUSTOM_LOCAL",
		"results":       map[string]any{"math": map[string]any{"score": 1.0, "nSamples": 12}},
		"runConfig":     map[string]any{"aggregatePreview": 1.0},
	}); err != nil {
		t.Fatalf("write run: %v", err)
	}
	hwFile := filepath.Join(tmp, "hardware.json")
	if err := writeJSON(hwFile, map[string]any{"hwClass": "DISCRETE_GPU", "gpuName": "NVIDIA GeForce RTX 3090", "gpuCount": 1, "vramGb": 24}); err != nil {
		t.Fatalf("write hardware: %v", err)
	}
	args := cliArgs{
		opts: map[string]string{
			"api-url":  srv.URL,
			"api-key":  "bhk_test",
			"base-url": srv.URL,
			"hardware": hwFile,
		},
		flags: map[string]bool{"dry-run": true, "quiet": true},
	}
	if err := handleEvalSubmit(runFile, args); err != nil {
		t.Fatalf("handleEvalSubmit: %v", err)
	}
	if searchQuery != "gemma-4-12b-it" {
		t.Fatalf("search query = %q, want served alias", searchQuery)
	}
	if stringValue(posted["hfId"]) != "google/gemma-3-12b-it" {
		t.Fatalf("posted hfId = %v, want resolved public HF id", posted["hfId"])
	}
	if mr := asObject(asObject(posted["runConfig"])["modelResolution"]); mr == nil || stringValue(mr["servedModel"]) != "gemma-4-12b-it" {
		t.Fatalf("posted runConfig missing modelResolution: %#v", posted["runConfig"])
	}
}

func TestLoadEvalDatasetBucketRequiresRunBundle(t *testing.T) {
	_, err := loadEvalDataset(map[string]any{"source": "bucket", "storageRef": map[string]any{"storageKey": "eval-datasets/x/items.jsonl", "format": "jsonl"}})
	requireCliErrorCode(t, err, "bucket_dataset_requires_run_bundle")
}

func TestNormalizeHardwarePayloadAcceptsLegacyGpuNameField(t *testing.T) {
	hw := normalizeHardwarePayload(map[string]any{
		"hwClass":  "DISCRETE_GPU",
		"gpuCount": 2,
		"gpus": []any{
			map[string]any{"name": "NVIDIA GeForce RTX 3090", "vramGb": 24},
			map[string]any{"gpuModel": "NVIDIA GeForce RTX 4090", "vramGb": 24},
		},
	})
	obj := asObject(hw)
	slots := anySlice(obj["gpus"])
	if stringValue(asObject(slots[0])["gpuName"]) != "NVIDIA GeForce RTX 3090" {
		t.Fatalf("slot 0 gpuName = %#v", asObject(slots[0]))
	}
	if asObject(slots[0])["name"] != nil {
		t.Fatalf("legacy name field was not removed: %#v", asObject(slots[0]))
	}
	if stringValue(asObject(slots[1])["gpuName"]) != "NVIDIA GeForce RTX 4090" {
		t.Fatalf("slot 1 gpuName = %#v", asObject(slots[1]))
	}
}
