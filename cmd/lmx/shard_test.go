package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

// chatContentServer returns an OpenAI-compatible /v1/chat/completions stub whose
// reply is chosen by replyFor(prompt), letting a test drive per-question output.
func chatContentServer(t *testing.T, replyFor func(prompt string) string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			http.NotFound(w, r)
			return
		}
		var body struct {
			Messages []struct {
				Content string `json:"content"`
			} `json:"messages"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		prompt := ""
		if len(body.Messages) > 0 {
			prompt = body.Messages[len(body.Messages)-1].Content
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []any{map[string]any{"message": map[string]any{"content": replyFor(prompt)}}},
		})
	}))
}

func TestScoreShardItemNumericExtraction(t *testing.T) {
	// Chain-of-thought ending in the correct number scores via last_number + numeric match.
	srv := chatContentServer(t, func(string) string { return "We add 48 + 24 = 72. Final answer: 72." })
	defer srv.Close()
	cfg := runShardConfig{maxTokens: 64, topP: 1}
	res := scoreShardItem(0, map[string]any{"question_id": "gsm8k:1", "input": "48 + 24?", "gold": "72"}, cfg, srv.URL, "m")
	if !res.scored || !res.pass {
		t.Fatalf("expected scored pass, got scored=%v pass=%v err=%q", res.scored, res.pass, res.errText)
	}
	if res.predicted != "72" {
		t.Fatalf("predicted = %q, want 72", res.predicted)
	}
}

func TestScoreShardItemNumericFormattingMatch(t *testing.T) {
	// "$81" must match gold "81" via formatting-insensitive numeric comparison.
	srv := chatContentServer(t, func(string) string { return "It costs **$81** in total." })
	defer srv.Close()
	res := scoreShardItem(0, map[string]any{"question_id": "q", "input": "x", "gold": "81"}, runShardConfig{maxTokens: 64, topP: 1}, srv.URL, "m")
	if !res.pass {
		t.Fatalf("expected $81 to match gold 81, got pass=%v predicted=%q", res.pass, res.predicted)
	}
}

func TestScoreShardItemWrongAnswerFails(t *testing.T) {
	srv := chatContentServer(t, func(string) string { return "The answer is 5." })
	defer srv.Close()
	res := scoreShardItem(0, map[string]any{"question_id": "q", "input": "x", "gold": "72"}, runShardConfig{maxTokens: 64, topP: 1}, srv.URL, "m")
	if !res.scored {
		t.Fatalf("expected scored, got err=%q", res.errText)
	}
	if res.pass {
		t.Fatal("expected wrong answer to fail")
	}
}

func TestScoreShardItemMultipleChoiceByLetter(t *testing.T) {
	srv := chatContentServer(t, func(string) string { return "The correct option is C." })
	defer srv.Close()
	item := map[string]any{"question_id": "q", "input": "Which is even?", "choices": []any{"3", "5", "8", "9"}, "gold": "C"}
	res := scoreShardItem(0, item, runShardConfig{maxTokens: 8, topP: 1}, srv.URL, "m")
	if !res.pass || res.predicted != "C" {
		t.Fatalf("MC scoring: pass=%v predicted=%q gold=%q", res.pass, res.predicted, res.gold)
	}
}

func TestScoreShardItemMultipleChoiceLoglikelihood(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/completions" {
			http.NotFound(w, r)
			return
		}
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatalf("decode completions body: %v", err)
		}
		prompts := anySlice(body["prompt"])
		choices := make([]any, len(prompts))
		for i := range prompts {
			score := -10.0
			if i == 2 {
				score = -1.0 // C wins.
			}
			choices[i] = map[string]any{
				"index": i,
				"logprobs": map[string]any{
					"text_offset":    []int{999},
					"token_logprobs": []float64{score},
				},
			}
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"choices": choices})
	}))
	defer srv.Close()
	item := map[string]any{"question_id": "q", "input": "Which is even?", "choices": []any{"3", "5", "8", "9"}, "gold": "C"}
	res := scoreShardItem(0, item, runShardConfig{scoring: "loglikelihood"}, srv.URL, "m")
	if !res.pass || res.predicted != "C" || res.gold != "C" || res.response != "C" {
		t.Fatalf("loglikelihood MC scoring: pass=%v predicted=%q gold=%q response=%q err=%q", res.pass, res.predicted, res.gold, res.response, res.errText)
	}
	if strings.Contains(res.prompt, "Reply with only the letter") {
		t.Fatalf("loglikelihood prompt should be continuation context, got chat prompt: %q", res.prompt)
	}
}

func TestDefaultShardScoring(t *testing.T) {
	if got := defaultShardScoring("hellaswag"); got != "loglikelihood" {
		t.Fatalf("hellaswag default scoring = %q, want loglikelihood", got)
	}
	if got := defaultShardScoring("cruxeval"); got != "cruxeval_execution" {
		t.Fatalf("cruxeval default scoring = %q, want cruxeval_execution", got)
	}
	if got := defaultShardScoring("gsm8k"); got != "exact_match" {
		t.Fatalf("gsm8k default scoring = %q, want exact_match", got)
	}
}

func TestHellaSwagDefaultLoglikelihoodWarnsForInstructModel(t *testing.T) {
	hint, ok := shouldWarnHellaSwagDefaultLoglikelihood("hellaswag", "loglikelihood", "", "Qwen/Qwen3-8B-Instruct", nil)
	if !ok || hint != "Qwen/Qwen3-8B-Instruct" {
		t.Fatalf("warning = (%q, %v), want instruct model warning", hint, ok)
	}
}

func TestHellaSwagDefaultLoglikelihoodWarningRespectsExplicitScoring(t *testing.T) {
	if hint, ok := shouldWarnHellaSwagDefaultLoglikelihood("hellaswag", "loglikelihood", "loglikelihood", "Qwen/Qwen3-8B-Instruct", nil); ok {
		t.Fatalf("explicit scoring should not warn, got hint %q", hint)
	}
	if hint, ok := shouldWarnHellaSwagDefaultLoglikelihood("hellaswag", "exact_match", "", "Qwen/Qwen3-8B-Instruct", nil); ok {
		t.Fatalf("exact_match should not warn, got hint %q", hint)
	}
}

func TestLlamaScorerDefaultsToCPUWhenServerConfigured(t *testing.T) {
	got := llamaScorerGPULayers(cliArgs{opts: map[string]string{"base-url": "http://127.0.0.1:8080"}})
	if got != "0" {
		t.Fatalf("llamaScorerGPULayers with base-url = %q, want 0", got)
	}
}

func TestLlamaScorerRespectsExplicitGPULayers(t *testing.T) {
	got := llamaScorerGPULayers(cliArgs{opts: map[string]string{"base-url": "http://127.0.0.1:8080", "gpu-layers": "99"}})
	if got != "99" {
		t.Fatalf("llamaScorerGPULayers explicit gpu-layers = %q, want 99", got)
	}
}

func TestScoreShardItemRequiresGoldAndID(t *testing.T) {
	srv := chatContentServer(t, func(string) string { return "anything" })
	defer srv.Close()
	cfg := runShardConfig{maxTokens: 8, topP: 1}
	if r := scoreShardItem(0, map[string]any{"input": "x", "gold": "1"}, cfg, srv.URL, "m"); r.errText == "" || r.scored {
		t.Fatalf("missing question_id should error, got %#v", r)
	}
	if r := scoreShardItem(0, map[string]any{"question_id": "q", "input": "x"}, cfg, srv.URL, "m"); r.errText == "" || r.scored {
		t.Fatalf("missing gold should error, got %#v", r)
	}
}

func TestScoreShardItemIgnoresTrailingQuantity(t *testing.T) {
	// Final-answer extraction must not pick the "2" from "for the 2 shirts".
	reply := "Total is 2 x $30 = $60. With 40% off, discount is $24.\n\n**Final Answer:**\nDavos paid **$36** for the 2 shirts."
	srv := chatContentServer(t, func(string) string { return reply })
	defer srv.Close()
	res := scoreShardItem(0, map[string]any{"question_id": "q", "input": "x", "gold": "36"}, runShardConfig{maxTokens: 256, topP: 1}, srv.URL, "m")
	if res.predicted != "$36" {
		t.Fatalf("predicted = %q, want $36", res.predicted)
	}
	if !res.pass {
		t.Fatalf("expected pass for gold 36, got pass=%v", res.pass)
	}
}

func TestExtractFinalAnswerModes(t *testing.T) {
	cases := []struct{ name, in, want string }{
		{"marker", "The answer is 42 because reasons.", "42"},
		{"hash", "work...\n#### 1,024", "1,024"},
		{"bold", "so the result is **$36** for the 2 shirts.", "$36"},
		{"equation_over_heading", "**Step 4: total**\nGrand Total = 84 + 88 = 172\nThen 20 + 22 + 64", "172"},
		{"ignores_bold_heading", "**Step 2: setup**\nso 2 + 2 gives us... 88 total chairs.", "88"},
		{"fallback", "no markers, just 7 then 19", "19"},
		{"generic_answer_word_not_marker", "The question asks for the answer. 1 SS = 1 SB.\nTotal buttons = 20 + 87 = 107.\nCheck: 1 LB = 3 SS.", "107"},
	}
	for _, tc := range cases {
		if got := extractAnswer(tc.in, "final_answer", ""); got != tc.want {
			t.Errorf("%s: extractAnswer(%q) = %q, want %q", tc.name, tc.in, got, tc.want)
		}
	}
}

func TestThinkingModeLogging(t *testing.T) {
	if got := promptThinkingDirective("/no_think\n\nQuestion"); got != "disabled" {
		t.Fatalf("promptThinkingDirective(/no_think) = %q, want disabled", got)
	}
	if got := promptThinkingDirective("/think\n\nQuestion"); got != "enabled" {
		t.Fatalf("promptThinkingDirective(/think) = %q, want enabled", got)
	}
	if got := observedThinkingMode("disabled", ""); got != "disabled" {
		t.Fatalf("observedThinkingMode(disabled, empty) = %q, want disabled", got)
	}
	if got := observedThinkingMode("disabled", "hidden reasoning"); got != "enabled" {
		t.Fatalf("observedThinkingMode(disabled, reasoning) = %q, want enabled", got)
	}
}

func TestCallOpenAIChatOmitsMaxTokensWhenUncapped(t *testing.T) {
	var sawMaxTokens bool
	var present bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		_, present = body["max_tokens"]
		sawMaxTokens = present
		_ = json.NewEncoder(w).Encode(map[string]any{"choices": []any{map[string]any{"message": map[string]any{"content": "ok"}}}})
	}))
	defer srv.Close()
	if _, err := callOpenAIChat(srv.URL, "m", "hi", "", 0, 0, 1, nil); err != nil {
		t.Fatalf("callOpenAIChat: %v", err)
	}
	if sawMaxTokens {
		t.Fatal("max_tokens should be omitted when maxTokens <= 0")
	}
	if _, err := callOpenAIChat(srv.URL, "m", "hi", "", 256, 0, 1, nil); err != nil {
		t.Fatalf("callOpenAIChat: %v", err)
	}
	if !present {
		t.Fatal("max_tokens should be sent when maxTokens > 0")
	}
}

func TestScoreShardItemSeparatesReasoning(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []any{map[string]any{"message": map[string]any{
				"content":           "Final answer: 42",
				"reasoning_content": "long chain of thought... 6*7=42",
			}}},
		})
	}))
	defer srv.Close()
	res := scoreShardItem(0, map[string]any{"question_id": "q", "input": "6*7?", "gold": "42"}, runShardConfig{topP: 1}, srv.URL, "m")
	if res.response != "Final answer: 42" {
		t.Fatalf("answer(response) = %q, want 'Final answer: 42'", res.response)
	}
	if res.reasoning != "long chain of thought... 6*7=42" {
		t.Fatalf("reasoning = %q", res.reasoning)
	}
	if !res.pass {
		t.Fatalf("expected pass, predicted %q", res.predicted)
	}
}

func TestRunEvalShardPreservesOrderAndStats(t *testing.T) {
	// Reply depends on the embedded gold so concurrent workers must still map
	// each result back to its own item (order preserved by index).
	srv := chatContentServer(t, func(prompt string) string {
		if strings.Contains(prompt, "even") {
			return "Answer: B" // wrong for a gold of C
		}
		return "The total is 10."
	})
	defer srv.Close()
	items := []map[string]any{
		{"question_id": "a", "input": "5 + 5?", "gold": "10"},                                    // pass
		{"question_id": "b", "input": "Which is even?", "choices": []any{"1", "2"}, "gold": "A"}, // B != A -> fail
		{"question_id": "c", "input": "5 + 5?", "gold": "10"},                                    // pass
	}
	results, stats := runEvalShard(cliArgs{opts: map[string]string{}, flags: map[string]bool{"quiet": true}}, srv.URL, "m", items, runShardConfig{maxTokens: 32, topP: 1, concurrency: 3})
	if len(results) != 3 || results[0].questionID != "a" || results[2].questionID != "c" {
		t.Fatalf("result order not preserved: %#v", results)
	}
	if stats.scored != 3 || stats.correct != 2 || stats.errors != 0 {
		t.Fatalf("stats = %+v, want scored 3 correct 2 errors 0", stats)
	}
}

// shardTestServer stubs the LocalMaxxing shard fetch + blob + submit and the
// model chat endpoint in one server so handleEvalShard can be exercised offline.
func shardTestServer(t *testing.T, rows []map[string]any, reply string, posted *map[string]any) *httptest.Server {
	t.Helper()
	var blob strings.Builder
	for _, row := range rows {
		line, _ := json.Marshal(row)
		blob.Write(line)
		blob.WriteByte('\n')
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.URL.Path == "/api/evals/gsm8k/shard":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"shard":       map[string]any{"shardIndex": 1, "itemCount": len(rows), "selectedQuestionCount": len(rows)},
				"sampling":    map[string]any{"recommendations": map[string]any{"margin05": len(rows)}},
				"evaluation":  map[string]any{"scoring": "exact_match", "promptTemplate": "/no_think\n\nCANONICAL GSM8K\n\n{{input}}", "answerExtraction": "final_answer", "maxNewTokens": 512},
				"downloadUrl": "http://" + r.Host + "/blob",
			})
		case r.URL.Path == "/blob":
			w.Header().Set("Content-Type", "application/x-ndjson")
			_, _ = w.Write([]byte(blob.String()))
		case r.URL.Path == "/v1/chat/completions":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"choices": []any{map[string]any{"message": map[string]any{"content": reply}}},
			})
		case r.URL.Path == "/api/evals/gsm8k/coverage":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"dataset":  map[string]any{"slug": "gsm8k", "shardCount": 3},
				"coverage": map[string]any{"uniqueQuestionCount": 0, "questionsNeeded": len(rows), "shardsCovered": []any{}},
			})
		case r.URL.Path == "/api/evals/gsm8k/submit":
			m := map[string]any{}
			_ = json.NewDecoder(r.Body).Decode(&m)
			*posted = m
			w.WriteHeader(http.StatusCreated)
			_ = json.NewEncoder(w).Encode(map[string]any{
				"run":       map[string]any{"id": "run_1", "status": "APPROVED"},
				"aggregate": map[string]any{"pooledScore": 1.0, "ciLower": 0.4, "ciUpper": 1.0, "shardsCovered": []any{1}},
			})
		default:
			http.NotFound(w, r)
		}
	}))
	return srv
}

func TestHandleEvalShardDryRunComputesAccuracy(t *testing.T) {
	rows := []map[string]any{
		{"question_id": "gsm8k:1", "input": "5 + 5?", "gold": "10"},
		{"question_id": "gsm8k:2", "input": "2 + 2?", "gold": "4"},
	}
	var posted map[string]any
	srv := shardTestServer(t, rows, "The final answer is 10.", &posted)
	defer srv.Close()
	tmp := t.TempDir()
	out := filepath.Join(tmp, "results.json")
	args := cliArgs{
		opts:  map[string]string{"api-url": srv.URL, "base-url": srv.URL, "model": "Qwen/Qwen3-8B", "questions": "2", "out": out},
		flags: map[string]bool{"quiet": true},
	}
	if err := handleEvalShard("gsm8k", args); err != nil {
		t.Fatalf("handleEvalShard dry-run: %v", err)
	}
	if posted != nil {
		t.Fatal("dry-run must not POST to the submit endpoint")
	}
	saved, err := readJSON(out)
	if err != nil {
		t.Fatalf("read results: %v", err)
	}
	summary := asObject(asObject(saved)["summary"])
	// Both questions answered "10": first passes, second fails -> 50%.
	if numberField(summary, "correct") != 1 || numberField(summary, "scored") != 2 {
		t.Fatalf("summary = %#v, want correct 1 scored 2", summary)
	}
	if numberField(summary, "accuracyPct") != 50 {
		t.Fatalf("accuracyPct = %v, want 50", summary["accuracyPct"])
	}
}

func TestHandleEvalShardSubmitsPassFail(t *testing.T) {
	rows := []map[string]any{
		{"question_id": "gsm8k:1", "input": "5 + 5?", "gold": "10"},
		{"question_id": "gsm8k:2", "input": "2 + 2?", "gold": "4"},
	}
	var posted map[string]any
	srv := shardTestServer(t, rows, "The final answer is 10.", &posted)
	defer srv.Close()
	tmp := t.TempDir()
	hwFile := filepath.Join(tmp, "hardware.json")
	if err := writeJSON(hwFile, map[string]any{"hwClass": "DISCRETE_GPU", "gpuName": "RTX 3090", "gpuCount": 1, "vramGb": 24}); err != nil {
		t.Fatalf("write hardware: %v", err)
	}
	args := cliArgs{
		opts:  map[string]string{"api-url": srv.URL, "base-url": srv.URL, "model": "Qwen/Qwen3-8B", "questions": "2", "hardware": hwFile, "api-key": "bhk_test"},
		flags: map[string]bool{"quiet": true, "submit": true},
	}
	if err := handleEvalShard("gsm8k", args); err != nil {
		t.Fatalf("handleEvalShard submit: %v", err)
	}
	if posted == nil {
		t.Fatal("submit endpoint was not called")
	}
	if numberField(posted, "shardIndex") != 1 {
		t.Fatalf("posted shardIndex = %v, want 1", posted["shardIndex"])
	}
	results := anySlice(posted["results"])
	if len(results) != 2 {
		t.Fatalf("posted results len = %d, want 2", len(results))
	}
	first := asObject(results[0])
	second := asObject(results[1])
	if stringValue(first["question_id"]) != "gsm8k:1" || first["pass"] != true {
		t.Fatalf("first result = %#v, want gsm8k:1 pass", first)
	}
	if second["pass"] != false {
		t.Fatalf("second result = %#v, want fail (gold 4 vs answer 10)", second)
	}
	artifacts := anySlice(posted["artifacts"])
	if len(artifacts) != 2 {
		t.Fatalf("posted artifacts len = %d, want 2", len(artifacts))
	}
	artifact := asObject(artifacts[0])
	if stringValue(artifact["question_id"]) != "gsm8k:1" || stringValue(artifact["prompt"]) == "" || stringValue(artifact["response"]) == "" {
		t.Fatalf("artifact missing question_id/prompt/response: %#v", artifact)
	}
	if stringValue(artifact["extractedAnswer"]) != "10" || stringValue(artifact["gold"]) != "10" || artifact["testPassed"] != true {
		t.Fatalf("artifact scoring fields wrong: %#v", artifact)
	}
	if stringValue(artifact["promptHash"]) == "" || numberField(artifact, "latencyMs") < 0 {
		t.Fatalf("artifact missing promptHash/latency: %#v", artifact)
	}
	if prompt := stringValue(artifact["prompt"]); !strings.HasPrefix(prompt, "/no_think\n\nCANONICAL GSM8K\n\n5 + 5?") {
		t.Fatalf("artifact prompt = %q, want server-canonical GSM8K prompt", prompt)
	}
	if stringValue(artifact["thinkingRequested"]) != "disabled" || stringValue(artifact["thinkingObserved"]) != "disabled" {
		t.Fatalf("artifact thinking fields wrong: %#v", artifact)
	}
	if stringValue(posted["hfId"]) != "Qwen/Qwen3-8B" {
		t.Fatalf("posted hfId = %v, want Qwen/Qwen3-8B", posted["hfId"])
	}
}

func TestHandleEvalShardRejectsCoveredShardWithoutRerun(t *testing.T) {
	rows := []map[string]any{{"question_id": "gsm8k:1", "input": "5 + 5?", "gold": "10"}}
	var blob strings.Builder
	for _, row := range rows {
		line, _ := json.Marshal(row)
		blob.Write(line)
		blob.WriteByte('\n')
	}
	var posted map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/evals/gsm8k/shard":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"shard":       map[string]any{"shardIndex": 1, "itemCount": len(rows)},
				"sampling":    map[string]any{"recommendations": map[string]any{"margin05": len(rows)}},
				"downloadUrl": "http://" + r.Host + "/blob",
			})
		case "/blob":
			_, _ = w.Write([]byte(blob.String()))
		case "/api/evals/gsm8k/coverage":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"dataset":  map[string]any{"slug": "gsm8k", "shardCount": 1},
				"coverage": map[string]any{"uniqueQuestionCount": 1, "questionsNeeded": 0, "shardsCovered": []any{1}},
			})
		case "/v1/chat/completions":
			t.Fatal("duplicate guard should run before model calls")
		case "/api/evals/gsm8k/submit":
			_ = json.NewDecoder(r.Body).Decode(&posted)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	args := shardSubmitArgs(t, srv.URL, map[string]string{"questions": "1"})
	if err := handleEvalShard("gsm8k", args); err == nil || !strings.Contains(err.Error(), "Shard 1 already") {
		t.Fatalf("handleEvalShard duplicate err = %v, want shard already submitted", err)
	}
	if posted != nil {
		t.Fatal("duplicate guard must not submit")
	}
}

func TestHandleEvalShardMissingOnlySelectsUncoveredShard(t *testing.T) {
	rows := []map[string]any{{"question_id": "gsm8k:2", "input": "5 + 5?", "gold": "10"}}
	var blob strings.Builder
	for _, row := range rows {
		line, _ := json.Marshal(row)
		blob.Write(line)
		blob.WriteByte('\n')
	}
	var posted map[string]any
	var shardRequests []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/evals/gsm8k/shard":
			shard := r.URL.Query().Get("shard")
			shardRequests = append(shardRequests, shard)
			if shard == "" {
				shard = "1"
			}
			shardIndex := 1
			if shard == "2" {
				shardIndex = 2
			}
			_ = json.NewEncoder(w).Encode(map[string]any{
				"shard":       map[string]any{"shardIndex": shardIndex, "itemCount": len(rows)},
				"sampling":    map[string]any{"recommendations": map[string]any{"margin05": len(rows)}},
				"downloadUrl": "http://" + r.Host + "/blob",
			})
		case "/blob":
			_, _ = w.Write([]byte(blob.String()))
		case "/api/evals/gsm8k/coverage":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"dataset":  map[string]any{"slug": "gsm8k", "shardCount": 2},
				"coverage": map[string]any{"uniqueQuestionCount": 1, "questionsNeeded": 1, "shardsCovered": []any{1}},
			})
		case "/v1/chat/completions":
			_ = json.NewEncoder(w).Encode(map[string]any{"choices": []any{map[string]any{"message": map[string]any{"content": "The final answer is 10."}}}})
		case "/api/evals/gsm8k/submit":
			_ = json.NewDecoder(r.Body).Decode(&posted)
			w.WriteHeader(http.StatusCreated)
			_ = json.NewEncoder(w).Encode(map[string]any{"run": map[string]any{"id": "run_2", "status": "APPROVED"}, "aggregate": map[string]any{"shardsCovered": []any{1, 2}}})
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	args := shardSubmitArgs(t, srv.URL, map[string]string{"questions": "1"})
	args.flags["missing-only"] = true
	if err := handleEvalShard("gsm8k", args); err != nil {
		t.Fatalf("handleEvalShard missing-only: %v", err)
	}
	if numberField(posted, "shardIndex") != 2 {
		t.Fatalf("posted shardIndex = %v, want 2; shard requests=%v", posted["shardIndex"], shardRequests)
	}
}

func TestHandleEvalShardStatusReadsCoverage(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/evals/gsm8k/coverage" {
			http.NotFound(w, r)
			return
		}
		if r.URL.Query().Get("hfId") != "Qwen/Qwen3-8B" {
			t.Fatalf("hfId query = %q", r.URL.Query().Get("hfId"))
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"dataset":  map[string]any{"slug": "gsm8k", "shardCount": 3},
			"coverage": map[string]any{"uniqueQuestionCount": 20, "questionsNeeded": 10, "shardsCovered": []any{1, 3}},
		})
	}))
	defer srv.Close()
	out := filepath.Join(t.TempDir(), "status.json")
	args := cliArgs{opts: map[string]string{"api-url": srv.URL, "model": "Qwen/Qwen3-8B", "out": out}, flags: map[string]bool{}}
	if err := handleEvalShardStatus("gsm8k", args); err != nil {
		t.Fatalf("handleEvalShardStatus: %v", err)
	}
	saved, err := readJSON(out)
	if err != nil {
		t.Fatalf("read status: %v", err)
	}
	status := asObject(asObject(saved)["status"])
	if got := anySlice(status["missingShards"]); len(got) != 1 || int(numberField(map[string]any{"v": got[0]}, "v")) != 2 {
		t.Fatalf("missingShards = %#v, want [2]", got)
	}
}

// shardArtifactRows returns 4 passing (gold 10) + 2 failing rows for the fixed
// reply "The final answer is 10."
func shardArtifactRows() []map[string]any {
	return []map[string]any{
		{"question_id": "gsm8k:1", "input": "a", "gold": "10"},
		{"question_id": "gsm8k:2", "input": "b", "gold": "10"},
		{"question_id": "gsm8k:3", "input": "c", "gold": "10"},
		{"question_id": "gsm8k:4", "input": "d", "gold": "10"},
		{"question_id": "gsm8k:5", "input": "e", "gold": "4"},
		{"question_id": "gsm8k:6", "input": "f", "gold": "7"},
	}
}

func shardSubmitArgs(t *testing.T, srvURL string, extra map[string]string) cliArgs {
	t.Helper()
	hwFile := filepath.Join(t.TempDir(), "hardware.json")
	if err := writeJSON(hwFile, map[string]any{"hwClass": "DISCRETE_GPU", "gpuName": "RTX 3090", "gpuCount": 1, "vramGb": 24}); err != nil {
		t.Fatalf("write hardware: %v", err)
	}
	opts := map[string]string{"api-url": srvURL, "base-url": srvURL, "model": "Qwen/Qwen3-8B", "questions": "6", "hardware": hwFile, "api-key": "bhk_test"}
	for k, v := range extra {
		opts[k] = v
	}
	return cliArgs{opts: opts, flags: map[string]bool{"quiet": true, "submit": true}}
}

func TestHandleEvalShardSubmitsAllArtifactsByDefault(t *testing.T) {
	// With no --artifact-limit, every scored question must be submitted so the
	// server can persist a complete whole-shard trace bundle.
	var posted map[string]any
	srv := shardTestServer(t, shardArtifactRows(), "The final answer is 10.", &posted)
	defer srv.Close()
	if err := handleEvalShard("gsm8k", shardSubmitArgs(t, srv.URL, nil)); err != nil {
		t.Fatalf("handleEvalShard submit: %v", err)
	}
	if posted == nil {
		t.Fatal("submit endpoint was not called")
	}
	if got := len(anySlice(posted["artifacts"])); got != 6 {
		t.Fatalf("posted artifacts len = %d, want 6 (all scored questions)", got)
	}
	if got := len(anySlice(posted["results"])); got != 6 {
		t.Fatalf("posted results len = %d, want 6", got)
	}
}

func TestHandleEvalShardArtifactLimitCapsBalancedSample(t *testing.T) {
	// A positive --artifact-limit keeps only a balanced pass/fail sample, while
	// results still cover every question.
	var posted map[string]any
	srv := shardTestServer(t, shardArtifactRows(), "The final answer is 10.", &posted)
	defer srv.Close()
	if err := handleEvalShard("gsm8k", shardSubmitArgs(t, srv.URL, map[string]string{"artifact-limit": "4"})); err != nil {
		t.Fatalf("handleEvalShard submit: %v", err)
	}
	artifacts := anySlice(posted["artifacts"])
	if len(artifacts) != 4 {
		t.Fatalf("posted artifacts len = %d, want 4 (capped)", len(artifacts))
	}
	pass, fail := 0, 0
	for _, a := range artifacts {
		if asObject(a)["testPassed"] == true {
			pass++
		} else {
			fail++
		}
	}
	if pass != 2 || fail != 2 {
		t.Fatalf("capped sample = %d pass / %d fail, want 2/2", pass, fail)
	}
	if got := len(anySlice(posted["results"])); got != 6 {
		t.Fatalf("posted results len = %d, want 6", got)
	}
}

func TestHandleEvalShardPullsQuantFromEndpoint(t *testing.T) {
	// With no --quantization, the runner pulls the quant from the endpoint like
	// `benchmark run`: the llama.cpp /props model_path filename yields Q4_K_M, and
	// the .gguf extension yields the container format.
	rows := []map[string]any{
		{"question_id": "gsm8k:1", "input": "5 + 5?", "gold": "10"},
		{"question_id": "gsm8k:2", "input": "2 + 2?", "gold": "4"},
	}
	var blob strings.Builder
	for _, row := range rows {
		line, _ := json.Marshal(row)
		blob.Write(line)
		blob.WriteByte('\n')
	}
	var posted map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/evals/gsm8k/shard":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"shard":       map[string]any{"shardIndex": 1, "itemCount": len(rows)},
				"sampling":    map[string]any{"recommendations": map[string]any{"margin05": len(rows)}},
				"downloadUrl": "http://" + r.Host + "/blob",
			})
		case "/blob":
			_, _ = w.Write([]byte(blob.String()))
		case "/props":
			_ = json.NewEncoder(w).Encode(map[string]any{"model_path": "/models/gemma-4-12b-it-Q4_K_M.gguf"})
		case "/v1/chat/completions":
			_ = json.NewEncoder(w).Encode(map[string]any{"choices": []any{map[string]any{"message": map[string]any{"content": "The final answer is 10."}}}})
		case "/api/evals/gsm8k/coverage":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"dataset":  map[string]any{"slug": "gsm8k", "shardCount": 3},
				"coverage": map[string]any{"uniqueQuestionCount": 0, "questionsNeeded": len(rows), "shardsCovered": []any{}},
			})
		case "/api/evals/gsm8k/submit":
			m := map[string]any{}
			_ = json.NewDecoder(r.Body).Decode(&m)
			posted = m
			w.WriteHeader(http.StatusCreated)
			_ = json.NewEncoder(w).Encode(map[string]any{"run": map[string]any{"id": "run_1", "status": "APPROVED"}, "aggregate": map[string]any{"pooledScore": 1.0}})
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	if err := handleEvalShard("gsm8k", shardSubmitArgs(t, srv.URL, map[string]string{"questions": "2"})); err != nil {
		t.Fatalf("handleEvalShard submit: %v", err)
	}
	if posted == nil {
		t.Fatal("submit endpoint was not called")
	}
	if got := stringValue(posted["quantization"]); got != "Q4_K_M" {
		t.Fatalf("posted quantization = %q, want Q4_K_M (pulled from /props model_path)", got)
	}
	if got := stringValue(posted["quantFormat"]); got != "gguf" {
		t.Fatalf("posted quantFormat = %q, want gguf (derived from .gguf path)", got)
	}
}

func TestExtractGeneratedCodeFencedBlock(t *testing.T) {
	resp := "Sure!\n```python\ndef add(a, b):\n    return a + b\n```\nDone."
	code := extractGeneratedCode(resp, "def add(a, b):\n", "add")
	if code != "def add(a, b):\n    return a + b" {
		t.Fatalf("unexpected extracted code: %q", code)
	}
}

func TestExtractGeneratedCodeBodyContinuation(t *testing.T) {
	// Model returned only the body (no def for entry point) -> prepend prompt stub.
	resp := "```python\n    return a + b\n```"
	prompt := "def add(a, b):\n"
	code := extractGeneratedCode(resp, prompt, "add")
	if code != "def add(a, b):\n    return a + b" {
		t.Fatalf("expected prompt-prefixed body, got: %q", code)
	}
}

func TestBuildCodeProgramAppendsCheck(t *testing.T) {
	item := map[string]any{
		"entry_point": "add",
		"test":        "def check(candidate):\n    assert candidate(1, 2) == 3\n",
	}
	program := buildCodeProgram(item, "def add(a, b):\n    return a + b")
	if !strings.Contains(program, "def add(a, b):") || !strings.HasSuffix(strings.TrimSpace(program), "check(add)") {
		t.Fatalf("program missing solution or check call: %q", program)
	}
}

func TestSandboxCommandUseSudoAndRelaxedSecurity(t *testing.T) {
	args := cliArgs{
		opts:  map[string]string{"sandbox-memory": "1g", "sandbox-cpus": "1"},
		flags: map[string]bool{"sandbox-use-sudo": true, "sandbox-relaxed-security": true},
	}
	cmd, snippet := sandboxCommand(args)
	if cmd.Args[0] != "sudo" {
		t.Fatalf("cmd.Args[0] = %q, want sudo", cmd.Args[0])
	}
	if !strings.HasPrefix(snippet, "sudo docker run --rm -i") {
		t.Fatalf("snippet missing sudo docker prefix: %q", snippet)
	}
	if strings.Contains(snippet, "--cap-drop") || strings.Contains(snippet, "--security-opt") || strings.Contains(snippet, "--read-only") {
		t.Fatalf("relaxed security should omit hardening flags: %q", snippet)
	}
	if !strings.Contains(snippet, "--tmpfs /tmp:exec,size=128m") || !strings.Contains(snippet, "--memory 1g") || !strings.Contains(snippet, "--cpus 1") {
		t.Fatalf("snippet missing resource/tmpfs flags: %q", snippet)
	}
}

func TestSandboxCommandQuotedRuntime(t *testing.T) {
	cmd, snippet := sandboxCommand(cliArgs{
		opts:  map[string]string{"sandbox-runtime": "sudo docker", "sandbox-image": "custom-sandbox"},
		flags: map[string]bool{},
	})
	if cmd.Args[0] != "sudo" {
		t.Fatalf("cmd.Args[0] = %q, want sudo", cmd.Args[0])
	}
	if !strings.HasPrefix(snippet, "sudo docker run --rm -i") || !strings.HasSuffix(snippet, " custom-sandbox") {
		t.Fatalf("snippet did not preserve runtime/image: %q", snippet)
	}
	if !strings.Contains(snippet, "--cap-drop ALL") || !strings.Contains(snippet, "--security-opt no-new-privileges") || !strings.Contains(snippet, "--read-only") {
		t.Fatalf("default command should keep hardening flags: %q", snippet)
	}
}

func TestSandboxFailureHintsDockerPermission(t *testing.T) {
	hints := strings.Join(sandboxFailureHints("permission denied while trying to connect to the docker API at unix:///var/run/docker.sock"), "\n")
	if !strings.Contains(hints, "Docker socket permission denied") || !strings.Contains(hints, "--sandbox-use-sudo") || !strings.Contains(hints, "docker build -t lmx-sandbox sandbox") {
		t.Fatalf("missing docker permission hints:\n%s", hints)
	}
}

func TestSandboxFailureHintsPythonOperationNotPermitted(t *testing.T) {
	hints := strings.Join(sandboxFailureHints("exec /usr/local/bin/python3: operation not permitted"), "\n")
	if !strings.Contains(hints, "--sandbox-relaxed-security") {
		t.Fatalf("missing relaxed-security hint:\n%s", hints)
	}
}

func TestRunEvalShardCodeExecGradesViaSandbox(t *testing.T) {
	// Model server returns a correct solution; sandbox runs locally via python3.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			http.NotFound(w, r)
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"choices": []any{map[string]any{"message": map[string]any{"content": "```python\ndef add(a, b):\n    return a + b\n```"}}}})
	}))
	defer srv.Close()
	items := []map[string]any{
		{"question_id": "humaneval:T1", "input": "def add(a, b):\n    \"\"\"add\"\"\"\n", "entry_point": "add", "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n"},
		{"question_id": "humaneval:T2", "input": "def add(a, b):\n    \"\"\"add\"\"\"\n", "entry_point": "add", "test": "def check(candidate):\n    assert candidate(1, 2) == 999\n"},
	}
	cfg := runShardConfig{scoring: "code_execution", concurrency: 1}
	args := cliArgs{opts: map[string]string{"sandbox-cmd": "python3 ../../sandbox/run_sandbox.py"}, flags: map[string]bool{"quiet": true}}
	results, stats, _, err := runEvalShardCodeExec(args, srv.URL, "m", items, cfg)
	if err != nil {
		t.Skipf("sandbox unavailable: %v", err)
	}
	if stats.scored != 2 {
		t.Fatalf("scored = %d, want 2", stats.scored)
	}
	if !results[0].pass {
		t.Fatalf("T1 should pass: %+v", results[0])
	}
	if results[1].pass {
		t.Fatalf("T2 should fail (wrong assertion)")
	}
}

func TestRunEvalShardCruxExecAcceptsEquivalentInput(t *testing.T) {
	srv := chatContentServer(t, func(string) string { return "Final answer: \"abc\"" })
	defer srv.Close()
	items := []map[string]any{
		{
			"question_id":     "cruxeval-i:sample_641",
			"task_type":       "input_prediction",
			"input":           "Given this Python function and its output, predict the input expression(s) that produced it. Return only the Python input expression(s).\n\ndef f(number):\n    return True if number.isdecimal() else False\n\nOutput:\nFalse",
			"code":            "def f(number):\n    return True if number.isdecimal() else False",
			"observed_output": "False",
			"gold":            "'dummy33;d'",
		},
	}
	cfg := runShardConfig{scoring: "cruxeval_execution", concurrency: 1}
	args := cliArgs{opts: map[string]string{"sandbox-cmd": "python3 ../../sandbox/run_sandbox.py"}, flags: map[string]bool{"quiet": true}}
	results, stats, _, err := runEvalShardCruxExec(args, srv.URL, "m", items, cfg)
	if err != nil {
		t.Skipf("sandbox unavailable: %v", err)
	}
	if stats.scored != 1 || stats.correct != 1 || !results[0].pass {
		t.Fatalf("equivalent input should pass: stats=%+v result=%+v", stats, results[0])
	}
	if results[0].predicted != "\"abc\"" {
		t.Fatalf("predicted = %q, want quoted candidate", results[0].predicted)
	}
}

func TestRunEvalShardCruxExecScoresOutputPrediction(t *testing.T) {
	srv := chatContentServer(t, func(string) string { return "Final answer: False" })
	defer srv.Close()
	items := []map[string]any{
		{
			"question_id":    "cruxeval-o:sample_739",
			"task_type":      "output_prediction",
			"input":          "Given this Python function and input expression(s), predict the exact Python repr output. Return only the output value.\n\ndef f(st, pattern):\n    for p in pattern:\n        if not st.startswith(p): return False\n        st = st[len(p):]\n    return True\n\nInput:\n'qwbnjrxs', ['jr', 'b', 'r', 'qw']",
			"code":           "def f(st, pattern):\n    for p in pattern:\n        if not st.startswith(p): return False\n        st = st[len(p):]\n    return True",
			"function_input": "'qwbnjrxs', ['jr', 'b', 'r', 'qw']",
			"gold":           "False",
		},
	}
	cfg := runShardConfig{scoring: "cruxeval_execution", concurrency: 1}
	args := cliArgs{opts: map[string]string{"sandbox-cmd": "python3 ../../sandbox/run_sandbox.py"}, flags: map[string]bool{"quiet": true}}
	results, stats, _, err := runEvalShardCruxExec(args, srv.URL, "m", items, cfg)
	if err != nil {
		t.Skipf("sandbox unavailable: %v", err)
	}
	if stats.scored != 1 || stats.correct != 1 || !results[0].pass {
		t.Fatalf("output prediction should pass: stats=%+v result=%+v", stats, results[0])
	}
	if results[0].predicted != "False" {
		t.Fatalf("predicted = %q, want False", results[0].predicted)
	}
}

func TestBuildCodeProgramKeepsPromptImports(t *testing.T) {
	// HumanEval-style item: prompt declares an import the model's code omitted.
	item := map[string]any{
		"entry_point": "f",
		"input":       "from typing import List\n\n\ndef f(xs: List[int]) -> int:\n    \"\"\"sum\"\"\"\n",
		"test":        "def check(candidate):\n    assert candidate([1, 2]) == 3\n",
	}
	solution := "def f(xs):\n    return sum(xs)"
	program := buildCodeProgram(item, solution)
	if !strings.Contains(program, "from typing import List") {
		t.Fatalf("program dropped prompt import:\n%s", program)
	}
	if !strings.HasSuffix(strings.TrimSpace(program), "check(f)") {
		t.Fatalf("program missing check call:\n%s", program)
	}
}

func TestRunEvalShardCodeExecEmptyGenerationCountsAsFail(t *testing.T) {
	// Model returns prose with no code -> unrunnable -> scored fail, not dropped.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{"choices": []any{map[string]any{"message": map[string]any{"content": "I cannot help with that."}}}})
	}))
	defer srv.Close()
	items := []map[string]any{
		{"question_id": "humaneval:E1", "input": "def f():\n    \"\"\"x\"\"\"\n", "entry_point": "f", "test": "def check(candidate):\n    assert candidate() == 1\n"},
	}
	cfg := runShardConfig{scoring: "code_execution", concurrency: 1}
	args := cliArgs{opts: map[string]string{"sandbox-cmd": "python3 ../../sandbox/run_sandbox.py"}, flags: map[string]bool{"quiet": true}}
	results, stats, _, err := runEvalShardCodeExec(args, srv.URL, "m", items, cfg)
	if err != nil {
		t.Skipf("sandbox unavailable: %v", err)
	}
	if stats.scored != 1 || stats.correct != 0 {
		t.Fatalf("empty generation should be scored fail: scored=%d correct=%d", stats.scored, stats.correct)
	}
	if results[0].pass || !results[0].scored {
		t.Fatalf("expected scored fail, got %+v", results[0])
	}
}

func TestPassAtKEstimator(t *testing.T) {
	cases := []struct {
		n, c, k int
		want    float64
	}{
		{1, 1, 1, 1.0},
		{1, 0, 1, 0.0},
		{5, 0, 1, 0.0},
		{5, 5, 1, 1.0},
		{2, 1, 1, 0.5},
		{10, 5, 1, 0.5},
		{5, 2, 5, 1.0}, // k clamped to n; n-c=3 < 5 -> 1.0
	}
	for _, tc := range cases {
		got := passAtK(tc.n, tc.c, tc.k)
		if got < tc.want-1e-9 || got > tc.want+1e-9 {
			t.Fatalf("passAtK(%d,%d,%d)=%v want %v", tc.n, tc.c, tc.k, got, tc.want)
		}
	}
}

func TestDefaultFewShotAndPreamble(t *testing.T) {
	if defaultFewShot("mbpp") != 3 || defaultFewShot("mbpp-plus") != 3 {
		t.Fatalf("mbpp family should default to 3-shot")
	}
	if defaultFewShot("humaneval") != 0 || defaultFewShot("gsm8k") != 0 {
		t.Fatalf("non-mbpp should default to 0-shot")
	}
	pre := mbppFewShotPreamble(3)
	if !strings.Contains(pre, "```python") || !strings.Contains(pre, "Your code should pass these tests") {
		t.Fatalf("few-shot preamble missing expected structure:\n%s", pre[:min(200, len(pre))])
	}
	if mbppFewShotPreamble(0) != "" {
		t.Fatalf("0-shot should be empty")
	}
}

func TestRunEvalShardCodeExecPassAtK(t *testing.T) {
	// Deterministic alternating server: sample 0 correct, sample 1 wrong, ...
	var calls int
	var mu sync.Mutex
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		i := calls
		calls++
		mu.Unlock()
		body := "```python\ndef add(a, b):\n    return a + b\n```"
		if i%2 == 1 {
			body = "```python\ndef add(a, b):\n    return 0\n```"
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"choices": []any{map[string]any{"message": map[string]any{"content": body}}}})
	}))
	defer srv.Close()
	items := []map[string]any{
		{"question_id": "humaneval:P1", "input": "def add(a, b):\n    \"\"\"add\"\"\"\n", "entry_point": "add", "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n"},
	}
	cfg := runShardConfig{scoring: "code_execution", concurrency: 1, nSamples: 2, passK: 2}
	args := cliArgs{opts: map[string]string{"sandbox-cmd": "python3 ../../sandbox/run_sandbox.py"}, flags: map[string]bool{"quiet": true}}
	_, _, metrics, err := runEvalShardCodeExec(args, srv.URL, "m", items, cfg)
	if err != nil {
		t.Skipf("sandbox unavailable: %v", err)
	}
	// 1 of 2 samples pass -> pass@2 = 1.0 (n-c=1 < k=2), pass@1 mean = 0.5.
	if pk, _ := metrics["passAtK"].(float64); pk < 0.999 {
		t.Fatalf("passAtK=%v want 1.0", metrics["passAtK"])
	}
	if p1, _ := metrics["passAt1"].(float64); p1 < 0.49 || p1 > 0.51 {
		t.Fatalf("passAt1=%v want 0.5", metrics["passAt1"])
	}
}
