package main

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

func TestSuiteImportJSONLMapsColumns(t *testing.T) {
	tmp := t.TempDir()
	source := filepath.Join(tmp, "questions.jsonl")
	out := filepath.Join(tmp, "suite.json")
	if err := os.WriteFile(source, []byte("{\"question\":\"2+2?\",\"answer\":\"4\"}\n{\"question\":\"3+3?\",\"answer\":\"6\"}\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	args := cliArgs{
		opts:  map[string]string{"slug": "math-private", "name": "Private Math", "kind": "qa", "input-column": "question", "gold-column": "answer", "out": out},
		flags: map[string]bool{"quiet": true},
	}
	if err := handleSuiteImport(source, args); err != nil {
		t.Fatalf("handleSuiteImport: %v", err)
	}
	payload, err := readJSON(out)
	if err != nil {
		t.Fatal(err)
	}
	items := anySlice(asObject(evalTasks(suiteDoc(asObject(payload)))[0]["dataset"])["items"])
	if len(items) != 2 || stringValue(asObject(items[1])["gold"]) != "6" {
		t.Fatalf("imported items = %#v", items)
	}
}

func TestSuiteAuditFindsDuplicateInputsAndInvalidChoiceGold(t *testing.T) {
	payload := buildSuiteTemplate("audit-suite", "Audit Suite", "reasoning", "CUSTOM", "exact_match", cliArgs{opts: map[string]string{"kind": "multiple_choice"}})
	task := evalTasks(suiteDoc(payload))[0]
	asObject(task["dataset"])["items"] = []any{
		map[string]any{"input": "Same prompt", "choices": []any{"One", "Two"}, "gold": "A"},
		map[string]any{"input": " same   prompt ", "choices": []any{"One", "One"}, "gold": "Z"},
	}
	result := auditSuite(payload)
	if numberField(result, "errorCount") != 3 {
		t.Fatalf("audit result = %#v", result)
	}
}

func TestSuiteValidatorMatchesServerScoringAndExtraction(t *testing.T) {
	userRated := buildSuiteTemplate("rated-suite", "Rated Suite", "writing", "CUSTOM", "exact_match", cliArgs{opts: map[string]string{"kind": "judge"}})
	suiteDoc(userRated)["scoringMethod"] = "user_rating"
	if err := validateSuite(userRated); err != nil {
		t.Fatalf("user_rating rejected: %v", err)
	}
	task := evalTasks(suiteDoc(userRated))[0]
	task["answerExtraction"] = "final_answer"
	if err := validateSuite(userRated); err == nil {
		t.Fatal("final_answer unexpectedly accepted for a suite task")
	}
}

func TestUploadSuiteInlineDatasetsRewritesVerifiedStorageRef(t *testing.T) {
	var uploaded []byte
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/benchmarks/storage/upload-url":
			var metadata map[string]any
			_ = json.NewDecoder(r.Body).Decode(&metadata)
			_ = json.NewEncoder(w).Encode(map[string]any{
				"uploadUrl":  "http://" + r.Host + "/upload",
				"headers":    map[string]string{"Content-Type": "application/x-ndjson"},
				"storageRef": map[string]any{"storageKey": "eval-uploads/user/suite-dataset/items.jsonl", "format": "jsonl", "itemCount": metadata["itemCount"]},
			})
		case "/upload":
			uploaded, _ = io.ReadAll(r.Body)
			w.WriteHeader(http.StatusOK)
		case "/api/benchmarks/storage/complete":
			var body map[string]any
			_ = json.NewDecoder(r.Body).Decode(&body)
			_ = json.NewEncoder(w).Encode(map[string]any{"storageRef": body["storageRef"]})
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	payload := buildSuiteTemplate("upload-suite", "Upload Suite", "reasoning", "CUSTOM", "exact_match", cliArgs{opts: map[string]string{"kind": "qa"}})
	converted, err := uploadSuiteInlineDatasets(payload, cliArgs{opts: map[string]string{"api-url": server.URL, "api-key": "bhk_test"}, flags: map[string]bool{"quiet": true}})
	if err != nil {
		t.Fatalf("uploadSuiteInlineDatasets: %v", err)
	}
	dataset := asObject(evalTasks(suiteDoc(asObject(converted)))[0]["dataset"])
	if stringValue(dataset["source"]) != "bucket" || stringValue(asObject(dataset["storageRef"])["storageKey"]) == "" {
		t.Fatalf("converted dataset = %#v", dataset)
	}
	if len(uploaded) == 0 {
		t.Fatal("dataset bytes were not uploaded")
	}
}
