package main

import (
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestApplyNvidiaSMIHardwareParsesMultipleGPUs(t *testing.T) {
	hardware := map[string]any{"hwClass": "CPU_ONLY"}
	applyNvidiaSMIHardware(hardware, "NVIDIA GeForce RTX 4090, 24564\nNVIDIA GeForce RTX 3090, 24576\n")

	if hardware["hwClass"] != "DISCRETE_GPU" {
		t.Fatalf("hwClass = %v, want DISCRETE_GPU", hardware["hwClass"])
	}
	if hardware["gpuCount"] != 2 {
		t.Fatalf("gpuCount = %v, want 2", hardware["gpuCount"])
	}
	if hardware["gpuName"] != "NVIDIA GeForce RTX 4090" {
		t.Fatalf("gpuName = %v", hardware["gpuName"])
	}
	if hardware["vramGb"] != 24.0 {
		t.Fatalf("vramGb = %v, want 24.0", hardware["vramGb"])
	}
	if hardware["totalVramGb"] != 48.0 {
		t.Fatalf("totalVramGb = %v, want 48.0", hardware["totalVramGb"])
	}
	gpus, ok := hardware["gpus"].([]map[string]any)
	if !ok || len(gpus) != 2 {
		t.Fatalf("gpus = %#v, want two parsed GPUs", hardware["gpus"])
	}
}

func TestEndpointTimeout(t *testing.T) {
	defaultTimeout, err := endpointTimeout(cliArgs{opts: map[string]string{}, flags: map[string]bool{}})
	if err != nil {
		t.Fatalf("default endpointTimeout returned error: %v", err)
	}
	if defaultTimeout != defaultEndpointTimeout {
		t.Fatalf("default timeout = %v, want %v", defaultTimeout, defaultEndpointTimeout)
	}

	customTimeout, err := endpointTimeout(cliArgs{opts: map[string]string{"endpoint-timeout-seconds": "42"}, flags: map[string]bool{}})
	if err != nil {
		t.Fatalf("custom endpointTimeout returned error: %v", err)
	}
	if customTimeout != 42*time.Second {
		t.Fatalf("custom timeout = %v, want 42s", customTimeout)
	}

	if _, err := endpointTimeout(cliArgs{opts: map[string]string{"endpoint-timeout-seconds": "0"}, flags: map[string]bool{}}); err == nil {
		t.Fatal("endpointTimeout accepted zero seconds")
	}
}

func TestReadOpenAIStreamParsesContentAndUsage(t *testing.T) {
	stream := strings.Join([]string{
		`data: {"choices":[{"delta":{"content":"hello"}}]}`,
		`data: {"choices":[{"delta":{"reasoning_content":" world"}}]}`,
		`data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2}}`,
		`data: [DONE]`,
		``,
	}, "\n")
	result, err := readOpenAIStream(cliArgs{opts: map[string]string{}, flags: map[string]bool{"quiet": true}}, strings.NewReader(stream), time.Now())
	if err != nil {
		t.Fatalf("readOpenAIStream returned error: %v", err)
	}
	if result.outputText != "hello world" {
		t.Fatalf("outputText = %q, want hello world", result.outputText)
	}
	if result.firstTokenAt.IsZero() {
		t.Fatal("firstTokenAt was not set")
	}
	if usageToken(result.usage, "prompt_tokens") != 3 || usageToken(result.usage, "completion_tokens") != 2 {
		t.Fatalf("usage = %#v", result.usage)
	}
}

func TestMeasureOpenAIEndpointMarksEstimatedPrefillAndStringEngineFlags(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/models":
			fmt.Fprint(w, `{"data":[{"id":"served-model","quantization":"fp16"}]}`)
		case "/props":
			fmt.Fprint(w, `{"model_path":"served-model.fp16.gguf"}`)
		case "/v1/chat/completions":
			w.Header().Set("Content-Type", "text/event-stream")
			time.Sleep(2 * time.Millisecond)
			fmt.Fprintln(w, `data: {"choices":[{"delta":{"content":"hello"}}]}`)
			fmt.Fprintln(w, `data: {"choices":[],"usage":{"prompt_tokens":8,"completion_tokens":2}}`)
			fmt.Fprintln(w, `data: [DONE]`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	metrics, err := measureOpenAIEndpoint(cliArgs{opts: map[string]string{"base-url": server.URL, "served-model": "served-model", "max-tokens": "16"}, flags: map[string]bool{"quiet": true}}, "org/model")
	if err != nil {
		t.Fatalf("measureOpenAIEndpoint returned error: %v", err)
	}
	if metrics["tokSPrefillSource"] != "estimated_from_ttft" {
		t.Fatalf("tokSPrefillSource = %v, want estimated_from_ttft", metrics["tokSPrefillSource"])
	}
	engineFlags, ok := metrics["engineFlags"].(map[string]any)
	if !ok {
		t.Fatalf("engineFlags type = %T, want map[string]any", metrics["engineFlags"])
	}
	if engineFlags["stream"] != true || engineFlags["maxTokens"] != 16 || engineFlags["servedModel"] != "served-model" {
		t.Fatalf("engineFlags = %#v", engineFlags)
	}
	if metrics["tokSPrefill"] == nil || metrics["ttftMs"] == nil {
		t.Fatalf("expected tokSPrefill and ttftMs, got %#v", metrics)
	}
}

func TestOpenAIBaseURLAcceptsV1Suffix(t *testing.T) {
	if got := openAIBaseURL("http://localhost:8080/v1/"); got != "http://localhost:8080" {
		t.Fatalf("openAIBaseURL = %q, want http://localhost:8080", got)
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/models":
			fmt.Fprint(w, `{"data":[{"id":"served-model"}]}`)
		case "/v1/chat/completions":
			w.Header().Set("Content-Type", "text/event-stream")
			fmt.Fprintln(w, `data: {"choices":[{"delta":{"content":"hello"}}]}`)
			fmt.Fprintln(w, `data: {"choices":[],"usage":{"prompt_tokens":8,"completion_tokens":2}}`)
			fmt.Fprintln(w, `data: [DONE]`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	metrics, err := measureOpenAIEndpoint(cliArgs{opts: map[string]string{"base-url": server.URL + "/v1", "served-model": "served-model", "max-tokens": "16"}, flags: map[string]bool{"quiet": true}}, "org/model")
	if err != nil {
		t.Fatalf("measureOpenAIEndpoint with /v1 base-url returned error: %v", err)
	}
	engineFlags := metrics["engineFlags"].(map[string]any)
	if engineFlags["baseUrl"] != server.URL {
		t.Fatalf("baseUrl = %v, want %v", engineFlags["baseUrl"], server.URL)
	}
}

func TestEndpointDiscoveryUsesProvidedBaseURL(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/models" {
			http.NotFound(w, r)
			return
		}
		fmt.Fprint(w, `{"data":[{"id":"served-model","quantization":"fp16"}]}`)
	}))
	defer server.Close()

	result := endpointDiscoveryCandidates(cliArgs{opts: map[string]string{"base-url": server.URL + "/v1"}, flags: map[string]bool{}})
	if len(result) != 1 || result[0] != server.URL {
		t.Fatalf("endpointDiscoveryCandidates = %#v, want normalized server URL", result)
	}
	args := cliArgs{opts: map[string]string{"base-url": server.URL + "/v1", "hf-id": "org/model", "quantization": "fp16"}, flags: map[string]bool{"quiet": true}}
	normalizeEndpointArgs(&args)
	if err := handleEndpoint("discover", args); err != nil {
		t.Fatalf("handleEndpoint returned error: %v", err)
	}
}

func TestRemoteModelResolutionTreatsServedModelAsAlias(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/models/search":
			fmt.Fprint(w, `{"models":[{"hfId":"unsloth/gemma-4-31B-it-GGUF"},{"hfId":"google/gemma-4-31B-it"}]}`)
		case "/api/models/unsloth/gemma-4-31B-it-GGUF":
			fmt.Fprint(w, `{"siblings":[{"rfilename":"gemma-4-31B-it-UD-Q4_K_XL.gguf"}]}`)
		case "/api/models/google/gemma-4-31B-it":
			fmt.Fprint(w, `{"siblings":[{"rfilename":"README.md"}]}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	resolution := remoteModelResolution(
		cliArgs{opts: map[string]string{"api-url": server.URL, "hf-api-url": server.URL}, flags: map[string]bool{"quiet": true}},
		"gemma-4-31b-it",
		"explicit",
		"google/gemma-4-31B-it",
		"/models/gemma-4-31B-it-UD-Q4_K_XL.gguf",
	)

	if resolution["status"] != "source_repo_detected" {
		t.Fatalf("status = %v, want source_repo_detected", resolution["status"])
	}
	if len(resolution["candidates"].([]any)) != 2 {
		t.Fatalf("candidates = %#v, want two model candidates", resolution["candidates"])
	}
	if resolution["sourceRepo"] != "unsloth/gemma-4-31B-it-GGUF" || resolution["sourceRepoMatch"] != "exact_filename" {
		t.Fatalf("source repo resolution = %#v", resolution)
	}
	if resolution["declaredBaseModel"] != "google/gemma-4-31B-it" || resolution["loadedFilename"] != "gemma-4-31B-it-UD-Q4_K_XL.gguf" {
		t.Fatalf("model metadata = %#v", resolution)
	}
}

func TestBenchmarkDryRunDoesNotExecuteGeneratedLocalCommand(t *testing.T) {
	payload, err := benchmarkPayloadFromFlags("llama.cpp", cliArgs{
		opts: map[string]string{
			"mode":         "local",
			"hf-id":        "Qwen/Qwen3-8B",
			"quantization": "Q4_K_M",
			"model-path":   "/definitely/missing/model.gguf",
		},
		flags: map[string]bool{"dry-run": true, "quiet": true},
	})
	if err != nil {
		t.Fatalf("benchmarkPayloadFromFlags returned error: %v", err)
	}
	if payload["dryRun"] != true {
		t.Fatalf("dryRun = %v, want true", payload["dryRun"])
	}
	if payload["tokSOut"] != nil {
		t.Fatalf("tokSOut = %v, want nil for dry-run plan without metrics", payload["tokSOut"])
	}
	engineFlags, ok := payload["engineFlags"].(map[string]any)
	if !ok {
		t.Fatalf("engineFlags type = %T, want map[string]any", payload["engineFlags"])
	}
	if !strings.Contains(stringValue(engineFlags["commandSnippet"]), "/definitely/missing/model.gguf") {
		t.Fatalf("commandSnippet = %q, want generated command", engineFlags["commandSnippet"])
	}
	if payload["promptTokens"] != 512.0 || payload["outputTokens"] != 128.0 {
		t.Fatalf("token fields = prompt %v output %v, want generated defaults", payload["promptTokens"], payload["outputTokens"])
	}
}

func TestBenchmarkDryRunInfersTokensFromExplicitCommand(t *testing.T) {
	payload, err := benchmarkPayloadFromFlags("llama.cpp", cliArgs{
		opts: map[string]string{
			"mode":         "local",
			"hf-id":        "Qwen/Qwen3-8B",
			"quantization": "Q4_K_M",
			"command":      "llama-bench -m qwen.gguf -p 64 -n 16",
		},
		flags: map[string]bool{"dry-run": true, "quiet": true},
	})
	if err != nil {
		t.Fatalf("benchmarkPayloadFromFlags returned error: %v", err)
	}
	if payload["promptTokens"] != 64.0 || payload["outputTokens"] != 16.0 {
		t.Fatalf("token fields = prompt %v output %v", payload["promptTokens"], payload["outputTokens"])
	}
	detected := payload["detectedEngines"].([]detectedEngine)
	for _, engine := range detected {
		if engine.Name == "llama.cpp" && engine.ServerCommand == "llama-bench -m qwen.gguf -p 64 -n 16" {
			t.Fatalf("serverCommand reused benchmark command: %#v", engine)
		}
	}
}

func TestBenchmarkManualMetricsDeriveTotalsFromCommandTokens(t *testing.T) {
	payload, err := benchmarkPayloadFromFlags("llama.cpp", cliArgs{
		opts: map[string]string{
			"mode":          "local",
			"hf-id":         "Qwen/Qwen3-8B",
			"quantization":  "Q4_K_M",
			"command":       "llama-bench -m qwen.gguf -p 64 -n 16",
			"tok-s-out":     "120",
			"tok-s-prefill": "1800",
		},
		flags: map[string]bool{"dry-run": true, "quiet": true},
	})
	if err != nil {
		t.Fatalf("benchmarkPayloadFromFlags returned error: %v", err)
	}
	if payload["promptTokens"] != 64.0 || payload["outputTokens"] != 16.0 {
		t.Fatalf("token fields = prompt %v output %v", payload["promptTokens"], payload["outputTokens"])
	}
	if payload["ttftMs"] == nil || payload["tokSTotal"] == nil {
		t.Fatalf("expected derived comparable metrics, got %#v", payload)
	}
}

func TestBenchmarkDryRunAutoDetectsVLLMEngine(t *testing.T) {
	binDir := t.TempDir()
	writeExecutable(t, filepath.Join(binDir, "vllm"))
	t.Setenv("PATH", binDir)

	payload, err := benchmarkPayloadFromFlags("", cliArgs{
		opts: map[string]string{
			"mode":         "local",
			"hf-id":        "Qwen/Qwen3-8B",
			"quantization": "fp16",
		},
		flags: map[string]bool{"dry-run": true, "quiet": true},
	})
	if err != nil {
		t.Fatalf("benchmarkPayloadFromFlags returned error: %v", err)
	}
	if payload["engineName"] != "vllm" {
		t.Fatalf("engineName = %v, want vllm", payload["engineName"])
	}
	engineFlags, ok := payload["engineFlags"].(map[string]any)
	if !ok || !strings.Contains(stringValue(engineFlags["commandSnippet"]), "vllm bench throughput") {
		t.Fatalf("engineFlags = %#v, want generated vLLM command", payload["engineFlags"])
	}
}

func TestLocalServerCommandBuildsLlamaServerCommand(t *testing.T) {
	command := localServerCommand("llama.cpp", cliArgs{
		opts: map[string]string{
			"model-path":     "/models/qwen.gguf",
			"host":           "127.0.0.1",
			"port":           "8080",
			"context-length": "8192",
			"gpu-layers":     "99",
		},
		flags: map[string]bool{},
	})
	for _, want := range []string{"llama-server", "-m /models/qwen.gguf", "--host 127.0.0.1", "--port 8080", "-c 8192", "-ngl 99"} {
		if !strings.Contains(command, want) {
			t.Fatalf("server command = %q, missing %q", command, want)
		}
	}
}

func TestLlamaBenchmarkCommandSupportsCommonBenchFlags(t *testing.T) {
	command := localBenchmarkCommand("llama.cpp", cliArgs{
		opts: map[string]string{
			"model-path":       "/models/qwen.gguf",
			"prompt-tokens":    "256",
			"output-tokens":    "64",
			"threads":          "8",
			"gpu-layers":       "99",
			"depth":            "1024",
			"batch-size":       "512",
			"micro-batch-size": "128",
			"repetitions":      "5",
			"benchmark-format": "json",
			"cache-type-k":     "q8_0",
			"cache-type-v":     "f16",
		},
		flags: map[string]bool{"flash-attn": true},
	})
	for _, want := range []string{"llama-bench", "-m /models/qwen.gguf", "-p 256", "-n 64", "-t 8", "-ngl 99", "-d 1024", "-b 512", "-ub 128", "-r 5", "-o json", "-fa 1", "-ctk q8_0", "-ctv f16"} {
		if !strings.Contains(command, want) {
			t.Fatalf("benchmark command = %q, missing %q", command, want)
		}
	}
}

func TestLlamaBenchmarkCommandSupportsNoFlashAttention(t *testing.T) {
	command := localBenchmarkCommand("llama.cpp", cliArgs{
		opts:  map[string]string{"model-path": "/models/qwen.gguf"},
		flags: map[string]bool{"no-flash-attn": true},
	})
	if !strings.Contains(command, "-fa 0") {
		t.Fatalf("benchmark command = %q, want -fa 0", command)
	}
}

func TestLocalServerCommandBuildsSGLangCommandWithDefaultPort(t *testing.T) {
	command := localServerCommand("sglang", cliArgs{
		opts: map[string]string{
			"hf-id":           "meta-llama/Llama-3.1-8B-Instruct",
			"host":            "127.0.0.1",
			"tensor-parallel": "2",
		},
		flags: map[string]bool{},
	})
	for _, want := range []string{"python3 -m sglang.launch_server", "--model-path meta-llama/Llama-3.1-8B-Instruct", "--host 127.0.0.1", "--port 30000", "--tp 2"} {
		if !strings.Contains(command, want) {
			t.Fatalf("server command = %q, missing %q", command, want)
		}
	}
}

func TestSGLangBenchmarkCommandUsesBenchServingFlags(t *testing.T) {
	command := sglangBenchmarkCommand(cliArgs{
		opts: map[string]string{
			"hf-id":            "meta-llama/Llama-3.1-8B-Instruct",
			"input-len":        "1024",
			"output-len":       "256",
			"num-prompts":      "2000",
			"request-rate":     "100",
			"max-concurrency":  "512",
			"benchmark-output": "sglang_random.jsonl",
		},
		flags: map[string]bool{},
	})
	for _, want := range []string{"python3 -m sglang.bench_serving", "--backend sglang", "--base-url http://localhost:30000", "--dataset-name random", "--random-input-len 1024", "--random-output-len 256", "--num-prompts 2000", "--request-rate 100", "--max-concurrency 512", "--output-file sglang_random.jsonl"} {
		if !strings.Contains(command, want) {
			t.Fatalf("benchmark command = %q, missing %q", command, want)
		}
	}
}

func writeExecutable(t *testing.T, path string) {
	t.Helper()
	if err := os.WriteFile(path, []byte("#!/bin/sh\nexit 0\n"), 0o755); err != nil {
		t.Fatalf("write executable: %v", err)
	}
}

func TestBenchmarkExplicitTokenFlagsDeriveComparableMetrics(t *testing.T) {
	payload, err := benchmarkPayloadFromFlags("llama.cpp", cliArgs{
		opts: map[string]string{
			"mode":          "local",
			"hf-id":         "Qwen/Qwen3-8B",
			"quantization":  "Q4_K_M",
			"tok-s-out":     "120",
			"tok-s-prefill": "1800",
			"prompt-tokens": "512",
			"output-tokens": "128",
		},
		flags: map[string]bool{"quiet": true},
	})
	if err != nil {
		t.Fatalf("benchmarkPayloadFromFlags returned error: %v", err)
	}
	if payload["promptTokens"] != 512.0 || payload["outputTokens"] != 128.0 {
		t.Fatalf("token fields = prompt %v output %v", payload["promptTokens"], payload["outputTokens"])
	}
	if payload["ttftMs"] != 284.4 {
		t.Fatalf("ttftMs = %v, want 284.4", payload["ttftMs"])
	}
	if payload["tokSTotal"] != 473.7 {
		t.Fatalf("tokSTotal = %v, want 473.7", payload["tokSTotal"])
	}
	provenance, ok := payload["provenance"].(map[string]any)
	if !ok {
		t.Fatalf("provenance type = %T, want map[string]any", payload["provenance"])
	}
	if _, exists := provenance["ttftSource"]; !exists || provenance["ttftSource"] != "estimated_from_prefill" {
		t.Fatalf("provenance ttftSource = %v", provenance["ttftSource"])
	}
}

func TestBenchmarkResultsFileAddsEngineFlags(t *testing.T) {
	results := filepath.Join(t.TempDir(), "results.json")
	if err := os.WriteFile(results, []byte(`{"output_token_throughput":222.2,"input_len":256,"output_len":64}`), 0o644); err != nil {
		t.Fatalf("write results: %v", err)
	}
	payload, err := benchmarkPayloadFromFlags("vllm", cliArgs{
		opts: map[string]string{
			"mode":         "local",
			"hf-id":        "Qwen/Qwen3-8B",
			"quantization": "fp16",
			"results":      results,
		},
		flags: map[string]bool{"quiet": true},
	})
	if err != nil {
		t.Fatalf("benchmarkPayloadFromFlags returned error: %v", err)
	}
	engineFlags, ok := payload["engineFlags"].(map[string]any)
	if !ok {
		t.Fatalf("engineFlags type = %T", payload["engineFlags"])
	}
	if engineFlags["resultsPath"] != results || !strings.Contains(stringValue(engineFlags["commandSnippet"]), results) {
		t.Fatalf("engineFlags = %#v", engineFlags)
	}
}

func TestBenchmarkRunPathUsesRunsDirectoryByModelID(t *testing.T) {
	path := benchmarkRunPath(map[string]any{"hfId": "Qwen/Qwen3-8B Instruct"})
	if !strings.HasPrefix(path, filepath.Join("runs", "Qwen-Qwen3-8B-Instruct")+string(os.PathSeparator)) {
		t.Fatalf("path = %q, want runs directory organized by sanitized model id", path)
	}
	if !strings.HasSuffix(path, ".json") {
		t.Fatalf("path = %q, want json file", path)
	}
}

func TestBenchmarkRunWritesDefaultRunFile(t *testing.T) {
	tmp := t.TempDir()
	wd, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	if err := os.Chdir(tmp); err != nil {
		t.Fatalf("chdir temp: %v", err)
	}
	defer func() {
		if err := os.Chdir(wd); err != nil {
			t.Fatalf("restore cwd: %v", err)
		}
	}()

	err = handleBenchmark("run", "llama.cpp", cliArgs{
		opts: map[string]string{
			"mode":         "local",
			"hf-id":        "Qwen/Qwen3-8B",
			"quantization": "Q4_K_M",
			"tok-s-out":    "120",
		},
		flags: map[string]bool{"quiet": true},
	})
	if err != nil {
		t.Fatalf("handleBenchmark returned error: %v", err)
	}
	entries, err := os.ReadDir(filepath.Join(tmp, "runs", "Qwen-Qwen3-8B"))
	if err != nil {
		t.Fatalf("read run dir: %v", err)
	}
	if len(entries) != 1 {
		t.Fatalf("run files = %d, want 1", len(entries))
	}
	value, err := readJSON(filepath.Join(tmp, "runs", "Qwen-Qwen3-8B", entries[0].Name()))
	if err != nil {
		t.Fatalf("read run json: %v", err)
	}
	payload := value.(map[string]any)
	if payload["hfId"] != "Qwen/Qwen3-8B" || payload["tokSOut"] != 120.0 {
		t.Fatalf("payload = %#v", payload)
	}
}

func TestBenchmarkRunEditAppliesAgentPatch(t *testing.T) {
	path := filepath.Join(t.TempDir(), "run.json")
	value := map[string]any{"hfId": "Qwen/Qwen3-8B", "engineName": "llama.cpp", "benchmarkMode": "local", "quantization": "Q4_K_M"}
	if err := writeJSON(path, value); err != nil {
		t.Fatalf("write run: %v", err)
	}
	if err := editBenchmarkRun(path, cliArgs{opts: map[string]string{"set-json": `{"tokSOut":120,"notes":"agent fixed metrics"}`}, flags: map[string]bool{}}); err != nil {
		t.Fatalf("editBenchmarkRun returned error: %v", err)
	}
	updated, err := readJSON(path)
	if err != nil {
		t.Fatalf("read run: %v", err)
	}
	payload := updated.(map[string]any)
	if payload["tokSOut"] != 120.0 || payload["notes"] != "agent fixed metrics" {
		t.Fatalf("payload = %#v", payload)
	}
	feedback := payload["agentFeedback"].(map[string]any)
	if feedback["canSubmit"] != true || feedback["submitCommand"] != "lmx benchmark submit "+path {
		t.Fatalf("feedback = %#v", feedback)
	}
}

func TestBenchmarkRunSummariesListsSavedRuns(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "Qwen-Qwen3-8B", "run.json")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if err := writeJSON(path, map[string]any{"hfId": "Qwen/Qwen3-8B", "engineName": "vllm", "benchmarkMode": "remote", "quantization": "fp16", "tokSOut": 42.0}); err != nil {
		t.Fatalf("write run: %v", err)
	}
	runs, err := benchmarkRunSummaries(root)
	if err != nil {
		t.Fatalf("benchmarkRunSummaries returned error: %v", err)
	}
	if len(runs) != 1 {
		t.Fatalf("runs len = %d, want 1", len(runs))
	}
	summary := runs[0].(map[string]any)
	if summary["path"] != path || summary["canSubmit"] != true || !strings.Contains(summary["rerunCommand"].(string), path) {
		t.Fatalf("summary = %#v", summary)
	}
	if !strings.Contains(summary["showCommand"].(string), path) || !strings.Contains(summary["deleteCommand"].(string), "--yes") {
		t.Fatalf("summary commands = %#v", summary)
	}
}

func TestBenchmarkRunStatsGroupsByQuantization(t *testing.T) {
	root := t.TempDir()
	writeBenchmarkRunForTest(t, filepath.Join(root, "run-fp16.json"), map[string]any{"hfId": "Qwen/Qwen3-8B", "engineName": "vllm", "benchmarkMode": "local", "quantization": "fp16", "tokSOut": 100.0})
	writeBenchmarkRunForTest(t, filepath.Join(root, "run-q4-a.json"), map[string]any{"hfId": "Qwen/Qwen3-8B", "engineName": "llama.cpp", "benchmarkMode": "local", "quantization": "Q4_K_M", "tokSOut": 120.0})
	writeBenchmarkRunForTest(t, filepath.Join(root, "run-q4-b.json"), map[string]any{"hfId": "Qwen/Qwen3-8B", "engineName": "llama.cpp", "benchmarkMode": "local", "quantization": "Q4_K_M", "tokSOut": 80.0})

	records, err := benchmarkRunRecords(root, cliArgs{opts: map[string]string{}, flags: map[string]bool{}})
	if err != nil {
		t.Fatalf("benchmarkRunRecords returned error: %v", err)
	}
	stats := benchmarkRunStatsResult(records, cliArgs{opts: map[string]string{"group-by": "quantization"}, flags: map[string]bool{}})
	groups := stats["groups"].([]any)
	if len(groups) != 2 {
		t.Fatalf("groups len = %d, want 2", len(groups))
	}
	first := groups[0].(map[string]any)
	if first["key"] != "Q4_K_M" || first["best"] != 120.0 || first["mean"] != 100.0 {
		t.Fatalf("first group = %#v", first)
	}
}

func TestBenchmarkRunExportCSVIncludesHardwareLabel(t *testing.T) {
	record := benchmarkRunRecord{
		Path:      "run.json",
		UpdatedAt: time.Date(2026, 1, 2, 3, 4, 5, 0, time.UTC),
		Payload: map[string]any{
			"hfId":         "Qwen/Qwen3-8B",
			"engineName":   "vllm",
			"quantization": "fp16",
			"tokSOut":      42.5,
			"hardware":     map[string]any{"gpuName": "NVIDIA RTX 4090", "vramGb": 24.0},
		},
	}
	fields := []string{"path", "model", "hardware", "tokSOut"}
	rows := benchmarkRunExportRows([]benchmarkRunRecord{record}, fields)
	text, err := benchmarkRunRowsCSV(fields, rows)
	if err != nil {
		t.Fatalf("benchmarkRunRowsCSV returned error: %v", err)
	}
	if !strings.Contains(text, "Qwen/Qwen3-8B") || !strings.Contains(text, "NVIDIA RTX 4090 24GB") || !strings.Contains(text, "42.5") {
		t.Fatalf("csv = %q", text)
	}
}

func TestCompareTwoBenchmarkRunsReportsPercent(t *testing.T) {
	left := benchmarkRunRecord{Path: "base.json", UpdatedAt: time.Now(), Payload: map[string]any{"hfId": "Qwen/Qwen3-8B", "engineName": "vllm", "quantization": "fp16", "tokSOut": 100.0, "ttftMs": 50.0}}
	right := benchmarkRunRecord{Path: "candidate.json", UpdatedAt: time.Now(), Payload: map[string]any{"hfId": "Qwen/Qwen3-8B", "engineName": "vllm", "quantization": "fp16", "tokSOut": 125.0, "ttftMs": 25.0}}
	result := compareTwoBenchmarkRuns(left, right, cliArgs{opts: map[string]string{"metrics": "tokSOut,ttftMs"}, flags: map[string]bool{}})
	comparisons := result["comparisons"].([]any)
	if len(comparisons) != 2 {
		t.Fatalf("comparisons len = %d, want 2", len(comparisons))
	}
	throughput := comparisons[0].(map[string]any)
	latency := comparisons[1].(map[string]any)
	if throughput["percent"] != 25.0 || throughput["better"] != true {
		t.Fatalf("throughput comparison = %#v", throughput)
	}
	if latency["percent"] != 50.0 || latency["better"] != true {
		t.Fatalf("latency comparison = %#v", latency)
	}
}

func writeBenchmarkRunForTest(t *testing.T, path string, payload map[string]any) {
	t.Helper()
	if err := writeJSON(path, payload); err != nil {
		t.Fatalf("write benchmark run: %v", err)
	}
}

func TestBenchmarkRunDeleteRequiresConfirmation(t *testing.T) {
	path := filepath.Join(t.TempDir(), "run.json")
	if err := writeJSON(path, map[string]any{"hfId": "Qwen/Qwen3-8B", "engineName": "vllm", "tokSOut": 42.0}); err != nil {
		t.Fatalf("write run: %v", err)
	}
	if err := deleteBenchmarkRun(path, cliArgs{opts: map[string]string{}, flags: map[string]bool{}}); err == nil {
		t.Fatal("deleteBenchmarkRun succeeded without confirmation")
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("run file was removed without confirmation: %v", err)
	}
}

func TestBenchmarkRunDeleteRemovesFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "run.json")
	if err := writeJSON(path, map[string]any{"hfId": "Qwen/Qwen3-8B", "engineName": "vllm", "tokSOut": 42.0}); err != nil {
		t.Fatalf("write run: %v", err)
	}
	if err := deleteBenchmarkRun(path, cliArgs{opts: map[string]string{}, flags: map[string]bool{"yes": true}}); err != nil {
		t.Fatalf("deleteBenchmarkRun returned error: %v", err)
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("run file still exists after delete: %v", err)
	}
}

func TestBenchmarkRunRerunUsesSavedCommand(t *testing.T) {
	tmp := t.TempDir()
	wd, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	if err := os.Chdir(tmp); err != nil {
		t.Fatalf("chdir temp: %v", err)
	}
	defer func() {
		if err := os.Chdir(wd); err != nil {
			t.Fatalf("restore cwd: %v", err)
		}
	}()

	saved := filepath.Join(tmp, "saved.json")
	if err := writeJSON(saved, map[string]any{
		"hfId":          "Qwen/Qwen3-8B",
		"engineName":    "llama.cpp",
		"benchmarkMode": "local",
		"quantization":  "Q4_K_M",
		"engineFlags":   map[string]any{"commandSnippet": "llama-bench -m model.gguf"},
	}); err != nil {
		t.Fatalf("write saved run: %v", err)
	}
	if err := rerunBenchmarkRun(saved, cliArgs{opts: map[string]string{}, flags: map[string]bool{"dry-run": true, "quiet": true}}); err != nil {
		t.Fatalf("rerunBenchmarkRun returned error: %v", err)
	}
	entries, err := os.ReadDir(filepath.Join(tmp, "runs", "Qwen-Qwen3-8B"))
	if err != nil {
		t.Fatalf("read run dir: %v", err)
	}
	if len(entries) != 1 {
		t.Fatalf("run files = %d, want 1", len(entries))
	}
	value, err := readJSON(filepath.Join(tmp, "runs", "Qwen-Qwen3-8B", entries[0].Name()))
	if err != nil {
		t.Fatalf("read rerun json: %v", err)
	}
	payload := value.(map[string]any)
	engineFlags := payload["engineFlags"].(map[string]any)
	if engineFlags["commandSnippet"] != "llama-bench -m model.gguf" || payload["dryRun"] != true {
		t.Fatalf("payload = %#v", payload)
	}
}

func TestBenchmarkRunRerunHonorsRunsDir(t *testing.T) {
	tmp := t.TempDir()
	wd, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	if err := os.Chdir(tmp); err != nil {
		t.Fatalf("chdir temp: %v", err)
	}
	defer func() {
		if err := os.Chdir(wd); err != nil {
			t.Fatalf("restore cwd: %v", err)
		}
	}()

	saved := filepath.Join(tmp, "saved.json")
	customRuns := filepath.Join(tmp, "custom-runs")
	if err := writeJSON(saved, map[string]any{
		"hfId":          "Qwen/Qwen3-8B",
		"engineName":    "llama.cpp",
		"benchmarkMode": "local",
		"quantization":  "Q4_K_M",
		"engineFlags":   map[string]any{"commandSnippet": "llama-bench -m model.gguf"},
	}); err != nil {
		t.Fatalf("write saved run: %v", err)
	}
	if err := rerunBenchmarkRun(saved, cliArgs{opts: map[string]string{"runs-dir": customRuns}, flags: map[string]bool{"dry-run": true, "quiet": true}}); err != nil {
		t.Fatalf("rerunBenchmarkRun returned error: %v", err)
	}
	entries, err := os.ReadDir(filepath.Join(customRuns, "Qwen-Qwen3-8B"))
	if err != nil {
		t.Fatalf("read custom run dir: %v", err)
	}
	if len(entries) != 1 {
		t.Fatalf("custom run files = %d, want 1", len(entries))
	}
	if _, err := os.Stat(filepath.Join(tmp, "runs")); !os.IsNotExist(err) {
		t.Fatalf("default runs dir exists after --runs-dir rerun: %v", err)
	}
}

func TestKVCacheDryRunBuildsLlamaCommandsPerLevel(t *testing.T) {
	payload, err := kvCachePayloadFromFlags("llama.cpp", cliArgs{
		opts: map[string]string{
			"mode":          "local",
			"hf-id":         "Qwen/Qwen3-8B",
			"quantization":  "Q4_K_M",
			"model-path":    "/models/qwen.gguf",
			"levels":        "10000,20000",
			"prompt-tokens": "512",
			"output-tokens": "64",
			"batch-size":    "256",
			"ubatch-size":   "64",
			"repetitions":   "3",
			"cache-type-k":  "q8_0",
			"cache-type-v":  "f16",
		},
		flags: map[string]bool{"dry-run": true, "quiet": true, "flash-attn": true},
	})
	if err != nil {
		t.Fatalf("kvCachePayloadFromFlags returned error: %v", err)
	}
	points := payload["points"].([]any)
	if len(points) != 2 {
		t.Fatalf("points len = %d, want 2", len(points))
	}
	first := points[0].(map[string]any)
	command := first["commandSnippet"].(string)
	for _, want := range []string{"llama-bench", "-p 512", "-n 64", "-d 10000", "-o json", "-b 256", "-ub 64", "-r 3", "-fa 1", "-ctk q8_0", "-ctv f16"} {
		if !strings.Contains(command, want) {
			t.Fatalf("command = %q, missing %q", command, want)
		}
	}
	second := points[1].(map[string]any)
	if !strings.Contains(second["commandSnippet"].(string), "-d 20000") {
		t.Fatalf("second command = %q, want -d 20000", second["commandSnippet"])
	}
}

func TestHandleKVCacheHonorsOutAndRunsDir(t *testing.T) {
	tmp := t.TempDir()
	out := filepath.Join(tmp, "sweep.json")
	runsDir := filepath.Join(tmp, "runs-out")
	args := cliArgs{
		opts: map[string]string{
			"mode":          "local",
			"hf-id":         "Qwen/Qwen3-8B",
			"quantization":  "Q4_K_M",
			"model-path":    "/models/qwen.gguf",
			"levels":        "512,1024",
			"prompt-tokens": "128",
			"output-tokens": "32",
			"out":           out,
			"runs-dir":      runsDir,
		},
		flags: map[string]bool{"dry-run": true, "quiet": true},
	}
	if err := handleKVCache("run", "llama.cpp", args); err != nil {
		t.Fatalf("handleKVCache returned error: %v", err)
	}
	aggregate, err := readJSON(out)
	if err != nil {
		t.Fatalf("read aggregate: %v", err)
	}
	obj := aggregate.(map[string]any)
	if len(obj["points"].([]any)) != 2 || len(obj["savedRuns"].([]any)) != 2 {
		t.Fatalf("aggregate = %#v", obj)
	}
	entries, err := os.ReadDir(filepath.Join(runsDir, "Qwen-Qwen3-8B"))
	if err != nil {
		t.Fatalf("read custom kvcache run dir: %v", err)
	}
	if len(entries) != 2 {
		t.Fatalf("custom kvcache run files = %d, want 2", len(entries))
	}
	value, err := readJSON(filepath.Join(runsDir, "Qwen-Qwen3-8B", entries[0].Name()))
	if err != nil {
		t.Fatalf("read saved kvcache run: %v", err)
	}
	payload := value.(map[string]any)["payload"].(map[string]any)
	engineFlags := payload["engineFlags"].(map[string]any)
	if !strings.Contains(stringValue(engineFlags["commandSnippet"]), "llama-bench") {
		t.Fatalf("engineFlags = %#v", engineFlags)
	}
}

func TestParseLlamaBenchJSONDepthMetrics(t *testing.T) {
	metrics := parseBenchmarkOutput(`[
  {"n_prompt":512,"n_gen":0,"n_depth":10000,"avg_ts":6425.91},
  {"n_prompt":0,"n_gen":128,"n_depth":10000,"avg_ts":116.71}
]`)
	if metrics["contextTokens"] != 10000 || metrics["promptTokens"] != 512 || metrics["outputTokens"] != 128 {
		t.Fatalf("token metrics = %#v", metrics)
	}
	if metrics["tokSPrefill"] != 6425.91 || metrics["tokSOut"] != 116.71 {
		t.Fatalf("throughput metrics = %#v", metrics)
	}
}

func TestKVCacheDryRunBuildsVLLMLatencyCommandsPerLevel(t *testing.T) {
	payload, err := kvCachePayloadFromFlags("vllm", cliArgs{
		opts: map[string]string{
			"mode":          "local",
			"hf-id":         "Qwen/Qwen3-8B",
			"levels":        "10000",
			"output-tokens": "64",
			"batch-size":    "1",
		},
		flags: map[string]bool{"dry-run": true, "quiet": true},
	})
	if err != nil {
		t.Fatalf("kvCachePayloadFromFlags returned error: %v", err)
	}
	points := payload["points"].([]any)
	command := points[0].(map[string]any)["commandSnippet"].(string)
	for _, want := range []string{"vllm bench latency", "--input-len 10000", "--output-len 64", "--batch-size 1", "--max-model-len 10064", "--output-json"} {
		if !strings.Contains(command, want) {
			t.Fatalf("command = %q, missing %q", command, want)
		}
	}
}

func TestParseVLLMLatencyJSONMetrics(t *testing.T) {
	metrics := parseBenchmarkOutput(`{"input_len":10000,"output_len":64,"batch_size":1,"avg_latency":2.0}`)
	if metrics["contextTokens"] != 10000 || metrics["promptTokens"] != 10000 || metrics["outputTokens"] != 64 {
		t.Fatalf("token metrics = %#v", metrics)
	}
	if metrics["latencyMs"] != 2000 || metrics["tokSTotal"] != 5032 || metrics["tokSOut"] != 32 {
		t.Fatalf("derived latency metrics = %#v", metrics)
	}
}

func TestRemoteKVCachePointMeasuresStreamingChatHistory(t *testing.T) {
	var requestBody string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			http.NotFound(w, r)
			return
		}
		data, _ := io.ReadAll(r.Body)
		requestBody = string(data)
		w.Header().Set("Content-Type", "text/event-stream")
		time.Sleep(2 * time.Millisecond)
		fmt.Fprintln(w, `data: {"choices":[{"delta":{"content":"hello"}}]}`)
		fmt.Fprintln(w, `data: {"choices":[],"usage":{"prompt_tokens":10000,"completion_tokens":2}}`)
		fmt.Fprintln(w, `data: [DONE]`)
	}))
	defer server.Close()

	point, err := measureRemoteKVCachePoint(cliArgs{opts: map[string]string{"base-url": server.URL, "served-model": "served-model", "max-tokens": "16"}, flags: map[string]bool{"quiet": true}}, "org/model", 10000)
	if err != nil {
		t.Fatalf("measureRemoteKVCachePoint returned error: %v", err)
	}
	if point["contextTokens"] != 10000.0 || point["promptTokens"] != 10000.0 || point["outputTokens"] != 2.0 {
		t.Fatalf("point token fields = %#v", point)
	}
	if point["tokSOut"] == nil || point["ttftMs"] == nil || point["tokSPrefill"] == nil {
		t.Fatalf("expected throughput fields, got %#v", point)
	}
	if !strings.Contains(requestBody, "Context received.") || !strings.Contains(requestBody, "stream_options") {
		t.Fatalf("requestBody missing retained chat history or usage options: %s", requestBody)
	}
}

func TestBenchmarkAgentFeedbackDistinguishesPlanFromReadyPayload(t *testing.T) {
	plan := benchmarkAgentFeedback(map[string]any{"engineName": "llama.cpp", "benchmarkMode": "local", "dryRun": true}, "plan.json", cliArgs{opts: map[string]string{}, flags: map[string]bool{}}, true, false)
	if plan["status"] != "plan_needs_metrics" {
		t.Fatalf("plan status = %v, want plan_needs_metrics", plan["status"])
	}
	if plan["canApiValidate"] != false || plan["requiresMetrics"] != true {
		t.Fatalf("plan feedback = %#v", plan)
	}
	if plan["nextCommand"] == "" {
		t.Fatalf("plan nextCommand is empty")
	}

	ready := benchmarkAgentFeedback(map[string]any{"engineName": "llama.cpp", "benchmarkMode": "local", "tokSOut": 120.0}, "ready.json", cliArgs{opts: map[string]string{"api-key": "bhk_test"}, flags: map[string]bool{}}, false, false)
	if ready["status"] != "ready_for_api_validation" {
		t.Fatalf("ready status = %v, want ready_for_api_validation", ready["status"])
	}
	if ready["canApiValidate"] != true || ready["submitCommand"] != "lmx benchmark submit ready.json" {
		t.Fatalf("ready feedback = %#v", ready)
	}

	t.Setenv("HOME", t.TempDir())
	t.Setenv("LMX_API_KEY", "")
	unauthenticated := benchmarkAgentFeedback(map[string]any{"engineName": "llama.cpp", "benchmarkMode": "local", "tokSOut": 120.0}, "ready.json", cliArgs{opts: map[string]string{}, flags: map[string]bool{}}, false, false)
	if unauthenticated["authRequired"] != true || unauthenticated["nextCommand"] != "lmx auth --key bhk_..." {
		t.Fatalf("unauthenticated feedback = %#v", unauthenticated)
	}
	if unauthenticated["validationCommand"] != "lmx benchmark dry-run ready.json" {
		t.Fatalf("validationCommand = %v", unauthenticated["validationCommand"])
	}
}
