package main

import (
	"bytes"
	"compress/gzip"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func matrixTestReport(engine string) map[string]any {
	return map[string]any{
		"metadata": map[string]any{"version": "0.7.5", "engine": engine, "model": "served", "timestamp": "2026-10-04T09:40:00", "decode_mode": "duration", "primary_decode_layer": "sustained_decode"},
		"prefill":  map[string]any{"4096": map[string]any{"tok_per_sec": 11385.0, "ttft_seconds": 0.36}},
		"results": []any{map[string]any{
			"concurrency": 1.0, "context_tokens": 0.0, "aggregate_tps": 129.0,
			"request_samples": []any{map[string]any{"ttft": 0.02}}, "loop_diagnostics": []any{"x"},
		}},
		"event_log":           []any{"warmup"},
		"startup_diagnostics": map[string]any{"env": "secret"},
	}
}

func matrixTestInput(engine string) inferenceMatrixInput {
	return inferenceMatrixInput{
		HfID: "LiquidAI/LFM2.5-2.6B", Quantization: "BF16",
		Hardware:       map[string]any{"gpuName": "NVIDIA GeForce RTX 3090", "vramGb": 24.0, "gpuCount": 1},
		Report:         matrixTestReport(engine),
		CommandSnippet: `vllm serve /m --api-key sk-1 --hf-token="hf x" --api-key --port 8000 --max-tokens 5`,
	}
}

func TestBuildInferenceMatrixEnvelopeKeepsCellsAndDropsBulk(t *testing.T) {
	envelope, err := buildInferenceMatrixEnvelope(matrixTestInput("vllm"))
	if err != nil {
		t.Fatal(err)
	}
	benchmark := asObject(envelope["benchmark"])
	for _, dropped := range []string{"event_log", "startup_diagnostics"} {
		if _, ok := benchmark[dropped]; ok {
			t.Fatalf("%s must not be uploaded", dropped)
		}
	}
	cell := asObject(anySlice(benchmark["results"])[0])
	if cell["aggregate_tps"] != 129.0 || cell["request_samples"] != nil || cell["loop_diagnostics"] != nil {
		t.Fatalf("decode cell not trimmed correctly: %#v", cell)
	}
	if len(asObject(benchmark["prefill"])) != 1 {
		t.Fatal("prefill cells lost")
	}
	if envelope["modelRevision"] != "main" || asObject(envelope["hardware"])["hwClass"] != "DISCRETE_GPU" {
		t.Fatalf("envelope defaults wrong: %#v", envelope)
	}
	if _, ok := envelope["engineVersion"]; ok {
		t.Fatal("empty engineVersion must be omitted, not sent blank")
	}
	if got := envelope["commandSnippet"]; got != `vllm serve /m --api-key <redacted> --hf-token=<redacted> --api-key --port 8000 --max-tokens 5` {
		t.Fatalf("commandSnippet not redacted correctly: %v", got)
	}
	// The source report must stay intact for the saved raw file.
	if anySlice(matrixTestInput("vllm").Report["results"])[0].(map[string]any)["request_samples"] == nil {
		t.Fatal("source report mutated")
	}
}

func TestBuildInferenceMatrixEnvelopeRejectsProxyAndMissingFields(t *testing.T) {
	_, err := buildInferenceMatrixEnvelope(matrixTestInput("openai_proxy"))
	if cliErr, ok := err.(cliError); !ok || cliErr.Code != "unsupported_matrix_engine" {
		t.Fatalf("openai_proxy must be rejected, got %v", err)
	}
	input := matrixTestInput("sglang")
	input.Hardware = nil
	input.Quantization = ""
	input.Report["results"] = []any{}
	input.Report["prefill"] = map[string]any{}
	_, err = buildInferenceMatrixEnvelope(input)
	cliErr, ok := err.(cliError)
	if !ok || cliErr.Code != "invalid_matrix_payload" {
		t.Fatalf("expected validation error, got %v", err)
	}
	fields := asObject(cliErr.Details)["fieldErrors"].(map[string][]string)
	for _, field := range []string{"hardware", "quantization", "benchmark"} {
		if fields[field] == nil {
			t.Fatalf("missing %s error in %#v", field, fields)
		}
	}
}

func TestLilBenchHardwareMapsInventory(t *testing.T) {
	hardware := lilBenchHardware(map[string]any{
		"gpus":   []any{map[string]any{"name": "NVIDIA RTX PRO 6000", "memory_total_mib": 97887.0}, map[string]any{"name": "NVIDIA RTX PRO 6000", "memory_total_mib": 97887.0}},
		"cpu":    map[string]any{"lscpu": map[string]any{"Model name": "AMD EPYC 9575F"}},
		"memory": map[string]any{"mem_total_mib": 773120.0},
	})
	submit := normalizeHardwareForSubmit(hardware)
	if submit["hwClass"] != "DISCRETE_GPU" || len(anySliceOfMaps(submit["gpus"])) != 2 || submit["cpu"] != "AMD EPYC 9575F" || submit["ramGb"] != 755.0 {
		t.Fatalf("unexpected hardware: %#v", submit)
	}
	if lilBenchHardware(map[string]any{"gpus": []any{}}) != nil {
		t.Fatal("inventory without GPUs must not invent hardware")
	}
}

func anySliceOfMaps(value any) []map[string]any {
	if slots, ok := value.([]map[string]any); ok {
		return slots
	}
	return nil
}

func TestServingPortAndRedaction(t *testing.T) {
	cases := []struct {
		argv []string
		want string
	}{
		{[]string{"/venv/bin/python3", "/venv/bin/vllm", "serve", "/m", "--port", "5199"}, "5199"},
		{[]string{"vllm", "serve", "/m"}, "8000"},
		{[]string{"python", "-m", "sglang.launch_server", "--model-path", "/m", "--port=31000"}, "31000"},
		{[]string{"python", "-m", "sglang.launch_server", "--model-path", "/m"}, "30000"},
		{[]string{"llama-server", "--port", "8000"}, ""},
	}
	for _, tc := range cases {
		if got := servingPort(tc.argv); got != tc.want {
			t.Errorf("servingPort(%v) = %q, want %q", tc.argv, got, tc.want)
		}
	}
	got := redactSecretArgs([]string{"vllm", "serve", "/m", "--api-key", "sk-1", "--hf-token=hf_x", "--max-tokens", "256", "--max-num-batched-tokens", "8192", "--enable-prefix-caching"})
	want := []string{"vllm", "serve", "/m", "--api-key", "<redacted>", "--hf-token=<redacted>", "--max-tokens", "256", "--max-num-batched-tokens", "8192", "--enable-prefix-caching"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("redactSecretArgs = %v", got)
	}
}

func TestReadMatrixSourceReadsGzip(t *testing.T) {
	var buffer bytes.Buffer
	writer := gzip.NewWriter(&buffer)
	_ = json.NewEncoder(writer).Encode(map[string]any{"schema": "lil-bench-result/1"})
	_ = writer.Close()
	path := filepath.Join(t.TempDir(), "run.json.gz")
	if err := os.WriteFile(path, buffer.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	value, err := readMatrixSource(path)
	if err != nil || !strings.HasPrefix(stringValue(asObject(value)["schema"]), "lil-bench-result/") {
		t.Fatalf("gzip report not decoded: %v %v", value, err)
	}
}

func TestInferenceMatrixEditPatch(t *testing.T) {
	hardwarePath := filepath.Join(t.TempDir(), "hw.json")
	if err := os.WriteFile(hardwarePath, []byte(`{"gpuName":"NVIDIA GeForce RTX 3090","vramGb":24,"gpuCount":1}`), 0o644); err != nil {
		t.Fatal(err)
	}
	patch, err := inferenceMatrixEditPatch(parseArgs([]string{"speed-test", "matrix", "submissions", "edit", "id",
		"--set", "engineVersion=0.26", "--set", "backend=null",
		"--set", "commandSnippet=vllm serve m --api-key sk-1 --port 8000",
		"--hardware", hardwarePath}))
	if err != nil {
		t.Fatal(err)
	}
	if patch["engineVersion"] != "0.26" {
		t.Fatalf("version must stay a string, got %#v", patch["engineVersion"])
	}
	if value, present := patch["backend"]; !present || value != nil {
		t.Fatalf("null must clear backend, got %#v", patch["backend"])
	}
	if patch["commandSnippet"] != "vllm serve m --api-key <redacted> --port 8000" {
		t.Fatalf("command not redacted: %v", patch["commandSnippet"])
	}
	if asObject(patch["hardware"])["hwClass"] != "DISCRETE_GPU" {
		t.Fatalf("hardware not normalized: %#v", patch["hardware"])
	}

	for _, argv := range [][]string{
		{"--set", "tokS=999"},
		{"--set", "benchmark={}"},
		{"--set", "hardware={}"},
		{"--set-json", `{"quantization":"FP8"}`},
		{},
	} {
		_, err := inferenceMatrixEditPatch(parseArgs(append([]string{"speed-test", "matrix", "submissions", "edit", "id"}, argv...)))
		if cliErr, ok := err.(cliError); !ok || cliErr.Code != "invalid_patch" {
			t.Fatalf("%v must be rejected as invalid_patch, got %v", argv, err)
		}
	}
}

// vLLM and SGLang report no quantization for unquantized models, so run/import
// fall back to launch flags and config.json; precedence decides the stored format.
func TestMatrixQuantizationFromModel(t *testing.T) {
	writeConfig := func(config map[string]any) string {
		dir := t.TempDir()
		data, _ := json.Marshal(config)
		if err := os.WriteFile(filepath.Join(dir, "config.json"), data, 0o644); err != nil {
			t.Fatal(err)
		}
		return dir
	}
	bf16 := writeConfig(map[string]any{"torch_dtype": "bfloat16"})
	awq := writeConfig(map[string]any{"torch_dtype": "float16", "quantization_config": map[string]any{"quant_method": "awq"}})
	compressed := writeConfig(map[string]any{"torch_dtype": "bfloat16", "quantization_config": map[string]any{"quant_method": "compressed-tensors"}})
	hub := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/org/fp8-model/resolve/main/config.json" {
			http.NotFound(w, r)
			return
		}
		json.NewEncoder(w).Encode(map[string]any{"dtype": "bfloat16", "quantization_config": map[string]any{"quant_method": "fp8"}})
	}))
	defer hub.Close()
	args := cliArgs{opts: map[string]string{"hf-api-url": hub.URL}}
	for _, tc := range []struct {
		name, want, source string
		argv               []string
		hfID               string
	}{
		{"vllm serve dir dtype", "BF16", "config_dtype", []string{"python", "vllm", "serve", bf16, "--port", "8000"}, "x/y"},
		{"sglang model-path quantized config", "AWQ", "config_quantization", []string{"python", "-m", "sglang.launch_server", "--model-path", awq}, "x/y"},
		{"launch quantization beats config", "FP8", "launch_flag", []string{"vllm", "serve", bf16, "--quantization=fp8"}, "x/y"},
		{"launch dtype beats config dtype", "FP16", "launch_dtype", []string{"vllm", "serve", bf16, "--dtype", "half"}, "x/y"},
		{"auto dtype falls back to config", "BF16", "config_dtype", []string{"vllm", "serve", bf16, "--dtype", "auto"}, "x/y"},
		{"ambiguous method never falls back to dtype", "", "", []string{"vllm", "serve", compressed}, "x/y"},
		{"hub repo from launch command", "FP8", "config_quantization", []string{"vllm", "serve", "org/fp8-model"}, "x/y"},
		{"hub repo from hf-id when argv unknown", "FP8", "config_quantization", nil, "org/fp8-model"},
		{"unreachable config", "", "", nil, "org/missing"},
	} {
		got, resolution := matrixQuantizationFromModel(tc.argv, tc.hfID, args)
		if got != tc.want || stringValue(resolution["trustedSource"]) != tc.source {
			t.Errorf("%s: got %q via %v, want %q via %q (%v)", tc.name, got, resolution["trustedSource"], tc.want, tc.source, resolution)
		}
	}
}
