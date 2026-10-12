package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"os/signal"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"
)

// The test binary doubles as a fake `vllm serve` (see TestMain): the value
// selects its behaviour, LMX_TEST_FAKE_VLLM_STATE receives its pid, argv and
// sampler env.
const matrixFakeEngineEnv = "LMX_TEST_FAKE_VLLM"

func runFakeMatrixEngine() {
	mode := os.Getenv(matrixFakeEngineEnv)
	argv := os.Args[1:]
	flags := launchFlags(argv)
	state, _ := json.Marshal(map[string]any{"pid": os.Getpid(), "argv": argv, "sampler": os.Getenv(matrixSamplerEnv), "path": os.Getenv("PATH")})
	_ = os.WriteFile(os.Getenv("LMX_TEST_FAKE_VLLM_STATE"), state, 0o644)
	fmt.Println("INFO fake vllm starting")
	if mode == "flashinfer-jit" && os.Getenv(matrixSamplerEnv) == "" {
		// The shape of vLLM's crash when FlashInfer's sampler cannot build.
		fmt.Println(`ERROR [core.py:1330]   File "/venv/lib/python3.12/site-packages/flashinfer/jit/core.py", line 316, in build_and_load`)
		fmt.Println(`ERROR [core.py:1330]   File "/venv/lib/python3.12/site-packages/flashinfer/jit/cpp_ext.py", line 368, in run_ninja`)
		fmt.Println(`ERROR [core.py:1330] FileNotFoundError: [Errno 2] No such file or directory: 'ninja'`)
		os.Exit(1)
	}
	if mode == "exit" {
		fmt.Println("ERROR boom: engine core failed")
		os.Exit(3)
	}
	if mode == "ignore-term" {
		signal.Ignore(syscall.SIGTERM)
	}
	ready := time.Now().Add(150 * time.Millisecond)
	maxModelLen, _ := strconv.Atoi(flags["max-model-len"])
	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		if time.Now().Before(ready) {
			w.WriteHeader(http.StatusServiceUnavailable)
		}
	})
	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{"data": []any{map[string]any{"id": flags["served-model-name"], "max_model_len": maxModelLen}}})
	})
	mux.HandleFunc("/version", func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{"version": "0.0.0-fake"})
	})
	_ = http.ListenAndServe(flags["host"]+":"+flags["port"], mux)
}

func matrixContexts(t *testing.T, list string) []matrixContextValue {
	t.Helper()
	values, err := parseMatrixContexts(list)
	if err != nil {
		t.Fatalf("parseMatrixContexts(%q): %v", list, err)
	}
	return values
}

func matrixErrorCode(err error) string {
	var cliErr cliError
	if errors.As(err, &cliErr) {
		return cliErr.Code
	}
	return ""
}

func TestParseMatrixContextsAndDefaults(t *testing.T) {
	values := matrixContexts(t, "0, 4k,16K,1.5k,3000")
	tokens := []int{}
	for _, value := range values {
		tokens = append(tokens, value.Tokens)
	}
	if !reflect.DeepEqual(tokens, []int{0, 4096, 16384, 1536, 3000}) || joinMatrixContexts(values) != "0,4k,16K,1.5k,3000" {
		t.Fatalf("contexts = %v (%s)", tokens, joinMatrixContexts(values))
	}
	for _, bad := range []string{"abc", "4,,8", "-1", "1.5"} {
		if _, err := parseMatrixContexts(bad); matrixErrorCode(err) != "invalid_option" {
			t.Fatalf("parseMatrixContexts(%q) error = %v", bad, err)
		}
	}
	plan, err := resolveMatrixBenchPlan(cliArgs{opts: map[string]string{}})
	if err != nil || plan.MaxTokens != 8192 || joinMatrixContexts(plan.Contexts) != "0,16k,32k,64k,128k" {
		t.Fatalf("default plan = %+v, %v", plan, err)
	}
	if _, err := resolveMatrixBenchPlan(cliArgs{opts: map[string]string{"extra-bench-args": "--max-tokens 512"}}); matrixErrorCode(err) != "invalid_option" {
		t.Fatalf("--max-tokens inside --extra-bench-args error = %v", err)
	}
	if _, err := resolveMatrixBenchPlan(cliArgs{opts: map[string]string{"max-tokens": "0"}}); matrixErrorCode(err) != "invalid_option" {
		t.Fatalf("--max-tokens 0 error = %v", err)
	}
}

func TestMatrixPreflightMath(t *testing.T) {
	plan := matrixBenchPlan{Contexts: matrixContexts(t, "0,4k,16k,32k"), MaxTokens: 8192}

	// 32k + 8192 = 40960 fits exactly: contexts at or above the 8k
	// calibration probe need no prompt margin.
	if checked, dropped, err := matrixPreflight(plan, 40960, false); err != nil || len(dropped) != 0 || len(checked.Contexts) != 4 {
		t.Fatalf("40960 fit: %+v %v %v", checked, dropped, err)
	}

	// The real RTX 3090 failure: --max-model-len 32768 with 32k contexts.
	_, _, err := matrixPreflight(plan, 32768, false)
	var cliErr cliError
	if !errors.As(err, &cliErr) || cliErr.Code != "context_exceeds_server_limit" {
		t.Fatalf("32768 error = %v", err)
	}
	if !strings.Contains(cliErr.Message, "Contexts 32k ") || !strings.Contains(cliErr.Message, "max_model_len 32768") || !strings.Contains(cliErr.Message, "8192 output tokens") {
		t.Fatalf("message = %q", cliErr.Message)
	}
	hints := strings.Join(cliErr.Hints, "\n")
	if !strings.Contains(hints, "--max-model-len 40960 or more") || !strings.Contains(hints, "--contexts 0,4k,16k, or pass --skip-oversize") || strings.Contains(hints, "--max-tokens") {
		t.Fatalf("hints = %q", hints)
	}

	// A --max-tokens hint appears when some output length still fits every context.
	_, _, err = matrixPreflight(matrixBenchPlan{Contexts: matrixContexts(t, "0,16k,32k"), MaxTokens: 8192}, 40000, false)
	if !errors.As(err, &cliErr) || !strings.Contains(strings.Join(cliErr.Hints, "\n"), "--max-tokens 7232,") {
		t.Fatalf("max-tokens hint: %#v", err)
	}

	// Small contexts reserve the prompt margin: 4k + 8192 + 256 > 12500.
	_, dropped, err := matrixPreflight(matrixBenchPlan{Contexts: matrixContexts(t, "0,4k"), MaxTokens: 8192}, 12500, true)
	if err != nil || joinMatrixContexts(dropped) != "4k" {
		t.Fatalf("margin: dropped %v err %v", dropped, err)
	}

	checked, dropped, err := matrixPreflight(plan, 32768, true)
	if err != nil || joinMatrixContexts(checked.Contexts) != "0,4k,16k" || joinMatrixContexts(dropped) != "32k" {
		t.Fatalf("--skip-oversize: %+v %v %v", checked, dropped, err)
	}
	if _, _, err := matrixPreflight(matrixBenchPlan{Contexts: matrixContexts(t, "32k"), MaxTokens: 8192}, 32768, true); matrixErrorCode(err) != "context_exceeds_server_limit" {
		t.Fatalf("--skip-oversize with nothing left: %v", err)
	}
}

func TestApplyMatrixPreflightReadsServerLimit(t *testing.T) {
	newServer := func(models map[string]any, serverInfo map[string]any) *httptest.Server {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			switch r.URL.Path {
			case "/v1/models":
				_ = json.NewEncoder(w).Encode(map[string]any{"data": []any{models}})
			case "/get_server_info":
				if serverInfo == nil {
					http.NotFound(w, r)
					return
				}
				_ = json.NewEncoder(w).Encode(serverInfo)
			default:
				http.NotFound(w, r)
			}
		}))
		t.Cleanup(server.Close)
		return server
	}
	run := func(server *httptest.Server, flags map[string]bool) (matrixBenchPlan, error) {
		_, info, err := detectServedModel(server.URL, "", "org/model")
		if err != nil {
			t.Fatal(err)
		}
		flags["quiet"] = true
		return applyMatrixPreflight(cliArgs{opts: map[string]string{}, flags: flags}, server.URL, "", info, matrixBenchPlan{Contexts: matrixContexts(t, "0,4k,16k,32k"), MaxTokens: 8192})
	}

	vllm := newServer(map[string]any{"id": "org/model", "max_model_len": 32768}, nil)
	if _, err := run(vllm, map[string]bool{}); matrixErrorCode(err) != "context_exceeds_server_limit" {
		t.Fatalf("vLLM 32768 error = %v", err)
	}
	if plan, err := run(vllm, map[string]bool{"skip-oversize": true}); err != nil || joinMatrixContexts(plan.Contexts) != "0,4k,16k" || joinMatrixContexts(plan.Skipped) != "32k" {
		t.Fatalf("vLLM --skip-oversize = %+v %v", plan, err)
	}
	if plan, err := run(newServer(map[string]any{"id": "org/model", "max_model_len": 40960}, nil), map[string]bool{}); err != nil || len(plan.Contexts) != 4 {
		t.Fatalf("vLLM 40960 = %+v %v", plan, err)
	}
	if _, err := run(newServer(map[string]any{"id": "org/model"}, map[string]any{"context_length": 20000}), map[string]bool{}); matrixErrorCode(err) != "context_exceeds_server_limit" {
		t.Fatalf("SGLang get_server_info context_length error = %v", err)
	}
	if plan, err := run(newServer(map[string]any{"id": "org/model"}, nil), map[string]bool{}); err != nil || len(plan.Contexts) != 4 {
		t.Fatalf("unknown limit must skip the check: %+v %v", plan, err)
	}
}

func TestChooseManagedMaxModelLen(t *testing.T) {
	maxLen, kept, dropped, err := chooseManagedMaxModelLen(matrixContexts(t, "0,4k,16k,32k,64k"), 8192, 40960)
	if err != nil || maxLen != 40960 || joinMatrixContexts(kept) != "0,4k,16k,32k" || joinMatrixContexts(dropped) != "64k" {
		t.Fatalf("Qwen3 40960 cap: %d %v %v %v", maxLen, kept, dropped, err)
	}
	if maxLen, _, _, _ := chooseManagedMaxModelLen(matrixContexts(t, "0,64k"), 8192, 0); maxLen != 73728 {
		t.Fatalf("unknown model limit: %d", maxLen)
	}
	if maxLen, _, _, _ := chooseManagedMaxModelLen(matrixContexts(t, "0,4k"), 8192, 0); maxLen != 13312 {
		t.Fatalf("4k + 8192 + margin rounds to 13312, got %d", maxLen)
	}
	if maxLen, _, _, _ := chooseManagedMaxModelLen(matrixContexts(t, "0,32000"), 8192, 40500); maxLen != 40500 {
		t.Fatalf("40192 rounds to 40960, which must be capped at the model's 40500; got %d", maxLen)
	}
	if _, _, _, err := chooseManagedMaxModelLen(matrixContexts(t, "32k"), 8192, 32768); matrixErrorCode(err) != "context_exceeds_model_limit" {
		t.Fatalf("nothing fits: %v", err)
	}

	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"text_config":{"max_position_embeddings":131072}}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if limit, source := matrixModelContextLimit(dir); limit != 131072 || source == "" {
		t.Fatalf("text_config limit = %d %q", limit, source)
	}
	if limit, _ := matrixModelContextLimit(t.TempDir()); limit != 0 {
		t.Fatalf("missing config.json limit = %d", limit)
	}
}

func writeMatrixScript(t *testing.T, path, content string) string {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0o755); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestFindMatrixEngineBinaryOrder(t *testing.T) {
	home, pathDir := t.TempDir(), t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("PATH", pathDir)

	if _, err := findMatrixEngineBinary(""); matrixErrorCode(err) != "engine_not_found" {
		t.Fatalf("nothing installed: %v", err)
	}
	dotVenv := writeMatrixScript(t, filepath.Join(home, ".venv", "bin", "vllm"), "#!/bin/sh\n")
	if got, _ := findMatrixEngineBinary(""); got != dotVenv {
		t.Fatalf("~/.venv: %q", got)
	}
	homeVLLM := writeMatrixScript(t, filepath.Join(home, "vllm", "bin", "vllm"), "#!/bin/sh\n")
	if got, _ := findMatrixEngineBinary(""); got != homeVLLM {
		t.Fatalf("~/vllm must win over ~/.venv: %q", got)
	}
	onPath := writeMatrixScript(t, filepath.Join(pathDir, "vllm"), "#!/bin/sh\n")
	if got, _ := findMatrixEngineBinary(""); got != onPath {
		t.Fatalf("PATH must win over venvs: %q", got)
	}
	explicit := writeMatrixScript(t, filepath.Join(t.TempDir(), "custom-vllm"), "#!/bin/sh\n")
	if got, _ := findMatrixEngineBinary(explicit); got != explicit {
		t.Fatalf("--engine-bin must win: %q", got)
	}
	if _, err := findMatrixEngineBinary(filepath.Join(home, "missing", "vllm")); matrixErrorCode(err) != "engine_not_found" {
		t.Fatalf("missing --engine-bin: %v", err)
	}
}

func TestMatrixSamplerFallback(t *testing.T) {
	if env := matrixSamplerFallbackEnv("", []string{"PATH=/bin"}); !reflect.DeepEqual(env, []string{"VLLM_USE_FLASHINFER_SAMPLER=0"}) {
		t.Fatalf("no nvcc: %v", env)
	}
	if env := matrixSamplerFallbackEnv("/usr/local/cuda/bin/nvcc", nil); env != nil {
		t.Fatalf("nvcc present: %v", env)
	}
	if env := matrixSamplerFallbackEnv("", []string{"VLLM_USE_FLASHINFER_SAMPLER=1"}); env != nil {
		t.Fatalf("explicit user env must win: %v", env)
	}

	saved := matrixNvccFallbacks
	t.Cleanup(func() { matrixNvccFallbacks = saved })
	matrixNvccFallbacks = nil
	t.Setenv("PATH", t.TempDir())
	t.Setenv("CUDA_HOME", "")
	if got := matrixNvccPath(); got != "" {
		t.Fatalf("nvcc found without any install: %q", got)
	}
	cudaHome := t.TempDir()
	nvcc := writeMatrixScript(t, filepath.Join(cudaHome, "bin", "nvcc"), "#!/bin/sh\n")
	t.Setenv("CUDA_HOME", cudaHome)
	if got := matrixNvccPath(); got != nvcc {
		t.Fatalf("CUDA_HOME nvcc: %q", got)
	}
}

func TestParseArgsEngineArgsTakeFlagValues(t *testing.T) {
	args := parseArgs([]string{"speed-test", "matrix", "run", "--model-path", "/m", "--engine-args", "--enforce-eager --max-num-seqs 8", "--skip-oversize", "--hf-id", "org/model", "--start-timeout", "60", "--gpu-memory-utilization", "0.8", "--engine-bin", "/v/bin/vllm", "--duration-seconds", "10"})
	if opt(args, "engine-args") != "--enforce-eager --max-num-seqs 8" || !hasFlag(args, "skip-oversize") || opt(args, "hf-id") != "org/model" {
		t.Fatalf("parsed = %+v", args)
	}
	if err := validateOptionNames(args); err != nil {
		t.Fatalf("matrix options must be known: %v", err)
	}
}

func TestManagedInferenceMatrixRejectsUnsupportedSetups(t *testing.T) {
	modelDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(modelDir, "config.json"), []byte(`{}`), 0o644); err != nil {
		t.Fatal(err)
	}
	ggufDir := t.TempDir()
	for name, tc := range map[string]struct {
		opts map[string]string
		code string
	}{
		"sglang":    {map[string]string{"engine": "sglang"}, "unsupported_managed_engine"},
		"base-url":  {map[string]string{"base-url": "http://127.0.0.1:8000"}, "conflicting_options"},
		"port":      {map[string]string{"engine-args": "--port 9000"}, "invalid_option"},
		"gpu-util":  {map[string]string{"gpu-memory-utilization": "1.5"}, "invalid_option"},
		"not a dir": {map[string]string{"model-path": filepath.Join(modelDir, "missing")}, "invalid_model_path"},
		"no config": {map[string]string{"model-path": ggufDir}, "invalid_model_path"},
	} {
		t.Run(name, func(t *testing.T) {
			opts := map[string]string{"model-path": modelDir, "hf-id": "org/model"}
			for key, value := range tc.opts {
				opts[key] = value
			}
			if err := runInferenceMatrix(cliArgs{opts: opts, flags: map[string]bool{"quiet": true}}); matrixErrorCode(err) != tc.code {
				t.Fatalf("error = %v, want %s", err, tc.code)
			}
		})
	}
}

type managedMatrixFixture struct {
	args      cliArgs
	out       string
	modelDir  string
	statePath string
	benchArgv string
}

// newManagedMatrixFixture wires the fake engine (this test binary), a fake
// python that writes a bench report, and a host without nvcc.
func newManagedMatrixFixture(t *testing.T, engineMode string) managedMatrixFixture {
	t.Helper()
	dir := t.TempDir()
	modelDir := filepath.Join(dir, "Qwen3-0.6B")
	if err := os.MkdirAll(modelDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(modelDir, "config.json"), []byte(`{"max_position_embeddings":40960,"torch_dtype":"bfloat16"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	python := writeMatrixScript(t, filepath.Join(dir, "python3"), `#!/bin/sh
PATH=/usr/bin:/bin; export PATH
if [ "$1" = "-c" ]; then exit 0; fi
printf '%s\n' "$@" > "$LMX_TEST_BENCH_ARGV"
if [ -n "$LMX_TEST_BENCH_FAIL" ]; then exit 1; fi
out=""; prev=""
for a in "$@"; do if [ "$prev" = "--output" ]; then out="$a"; fi; prev="$a"; done
cat > "$out" <<'EOF'
{"metadata":{"engine":"vllm","version":"0.7.7","max_tokens":8192},"results":[{"context_tokens":0,"concurrency":1,"aggregate_tps":100}]}
EOF
`)
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	fixture := managedMatrixFixture{
		out:       filepath.Join(dir, "runs", "matrix.json"),
		modelDir:  modelDir,
		statePath: filepath.Join(dir, "engine-state.json"),
		benchArgv: filepath.Join(dir, "bench-argv.txt"),
	}
	t.Setenv(matrixFakeEngineEnv, engineMode)
	t.Setenv("LMX_TEST_FAKE_VLLM_STATE", fixture.statePath)
	t.Setenv("LMX_TEST_BENCH_ARGV", fixture.benchArgv)
	t.Setenv("LMX_TEST_BENCH_FAIL", "")
	t.Setenv("PATH", t.TempDir())
	t.Setenv("CUDA_HOME", "")
	t.Setenv(matrixSamplerEnv, "")
	os.Unsetenv(matrixSamplerEnv)
	savedNvcc, savedPoll, savedGrace, savedStartupGrace := matrixNvccFallbacks, managedEnginePollInterval, managedEngineStopGrace, managedEngineStartupStopGrace
	t.Cleanup(func() {
		matrixNvccFallbacks, managedEnginePollInterval, managedEngineStopGrace, managedEngineStartupStopGrace = savedNvcc, savedPoll, savedGrace, savedStartupGrace
	})
	matrixNvccFallbacks, managedEnginePollInterval, managedEngineStopGrace, managedEngineStartupStopGrace = nil, 20*time.Millisecond, 300*time.Millisecond, 300*time.Millisecond

	fixture.args = cliArgs{opts: map[string]string{
		"model-path":    modelDir,
		"hf-id":         "org/model",
		"engine-bin":    exe,
		"python-bin":    python,
		"bench-script":  python,
		"hardware":      writeBenchmarkHardwareForTest(t),
		"quantization":  "BF16",
		"out":           fixture.out,
		"contexts":      "0,4k,16k,32k,64k",
		"concurrency":   "1",
		"start-timeout": "30",
	}, flags: map[string]bool{"quiet": true}}
	return fixture
}

func (f managedMatrixFixture) engineState(t *testing.T) (int, []string, string) {
	t.Helper()
	data, err := os.ReadFile(f.statePath)
	if err != nil {
		t.Fatalf("fake engine never started: %v", err)
	}
	var state struct {
		Pid     int      `json:"pid"`
		Argv    []string `json:"argv"`
		Sampler string   `json:"sampler"`
	}
	if err := json.Unmarshal(data, &state); err != nil {
		t.Fatal(err)
	}
	return state.Pid, state.Argv, state.Sampler
}

func TestManagedInferenceMatrixLifecycle(t *testing.T) {
	fixture := newManagedMatrixFixture(t, "serve")
	if err := runInferenceMatrix(fixture.args); err != nil {
		t.Fatalf("managed run: %v", err)
	}
	pid, argv, sampler := fixture.engineState(t)
	if processIsRunning(pid) {
		t.Fatalf("server pid %d still running after a successful run", pid)
	}
	joined := strings.Join(argv, " ")
	// 64k cannot fit the model's 40960 positions with 8192 output tokens; 32k fits exactly.
	want := "serve " + fixture.modelDir + " --served-model-name org/model --host 127.0.0.1 --port "
	if !strings.HasPrefix(joined, want) || !strings.Contains(joined, "--max-model-len 40960 --gpu-memory-utilization 0.9") {
		t.Fatalf("engine argv = %q", joined)
	}
	if sampler != "0" {
		t.Fatalf("sampler fallback env = %q, want 0 without nvcc", sampler)
	}
	benchArgv, err := os.ReadFile(fixture.benchArgv)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(benchArgv), "--contexts\n0,4k,16k,32k\n--max-tokens\n8192\n") {
		t.Fatalf("bench argv = %q", benchArgv)
	}
	saved, err := readJSON(fixture.out)
	if err != nil {
		t.Fatal(err)
	}
	payload := asObject(asObject(saved)["payload"])
	snippet := stringValue(payload["commandSnippet"])
	if !strings.HasPrefix(snippet, "VLLM_USE_FLASHINFER_SAMPLER=0 ") || !strings.Contains(snippet, " "+want) || !strings.Contains(snippet, "--max-model-len 40960") {
		t.Fatalf("recorded command = %q", snippet)
	}
	if payload["engineVersion"] != "0.0.0-fake" {
		t.Fatalf("engineVersion = %v", payload["engineVersion"])
	}
	logData, err := os.ReadFile(strings.TrimSuffix(fixture.out, ".json") + ".server.log")
	if err != nil || !strings.Contains(string(logData), "fake vllm starting") {
		t.Fatalf("server log = %q, %v", logData, err)
	}
}

func TestManagedInferenceMatrixStopsServerAfterBenchFailure(t *testing.T) {
	// The engine ignores SIGTERM, so stopping also exercises the SIGKILL escalation.
	fixture := newManagedMatrixFixture(t, "ignore-term")
	t.Setenv("LMX_TEST_BENCH_FAIL", "1")
	if err := runInferenceMatrix(fixture.args); matrixErrorCode(err) != "matrix_benchmark_failed" {
		t.Fatalf("error = %v, want matrix_benchmark_failed", err)
	}
	pid, _, _ := fixture.engineState(t)
	if processIsRunning(pid) {
		t.Fatalf("server pid %d still running after a failed bench", pid)
	}
}

func TestManagedInferenceMatrixReportsEngineExit(t *testing.T) {
	fixture := newManagedMatrixFixture(t, "exit")
	err := runInferenceMatrix(fixture.args)
	var cliErr cliError
	if !errors.As(err, &cliErr) || cliErr.Code != "engine_exited" {
		t.Fatalf("error = %v, want engine_exited", err)
	}
	tail := strings.Join(asObject(cliErr.Details)["logTail"].([]string), "\n")
	if !strings.Contains(tail, "ERROR boom: engine core failed") {
		t.Fatalf("log tail = %q", tail)
	}
	if _, statErr := os.Stat(fixture.benchArgv); statErr == nil {
		t.Fatal("the bench must not run when the server never became ready")
	}
}

func TestMatrixEngineEnvironPutsEngineDirFirstOnPath(t *testing.T) {
	sep := string(os.PathListSeparator)
	got := matrixEngineEnviron("/home/u/vllm/bin/vllm", []string{"HOME=/home/u", "PATH=/usr/bin" + sep + "/bin"})
	if !reflect.DeepEqual(got, []string{"HOME=/home/u", "PATH=/home/u/vllm/bin" + sep + "/usr/bin" + sep + "/bin"}) {
		t.Fatalf("environ = %v", got)
	}
	if again := matrixEngineEnviron("/home/u/vllm/bin/vllm", got); !reflect.DeepEqual(again, got) {
		t.Fatalf("already first on PATH must not be prepended twice: %v", again)
	}
	if missing := matrixEngineEnviron("/opt/v/bin/vllm", []string{"HOME=/x"}); !reflect.DeepEqual(missing, []string{"HOME=/x", "PATH=/opt/v/bin"}) {
		t.Fatalf("no PATH = %v", missing)
	}
}

func TestFlashInferJITFailedDetection(t *testing.T) {
	crash := `  File "/v/site-packages/flashinfer/jit/cpp_ext.py", line 368, in run_ninja
FileNotFoundError: [Errno 2] No such file or directory: 'ninja'`
	if !flashInferJITFailed(crash) {
		t.Fatal("ninja crash inside flashinfer/jit not recognised")
	}
	if flashInferJITFailed("torch.OutOfMemoryError: CUDA out of memory") {
		t.Fatal("unrelated startup failures must not trigger the sampler retry")
	}
}

func TestManagedInferenceMatrixRetriesWithoutFlashInferSampler(t *testing.T) {
	fixture := newManagedMatrixFixture(t, "flashinfer-jit")
	// nvcc is present, so no fallback is applied up front; the JIT build still fails.
	nvcc := writeMatrixScript(t, filepath.Join(t.TempDir(), "nvcc"), "#!/bin/sh\nexit 0\n")
	matrixNvccFallbacks = []string{nvcc}
	if err := runInferenceMatrix(fixture.args); err != nil {
		t.Fatalf("managed run with sampler retry: %v", err)
	}
	pid, _, sampler := fixture.engineState(t)
	if sampler != "0" || processIsRunning(pid) {
		t.Fatalf("retried engine sampler=%q running=%v", sampler, processIsRunning(pid))
	}
	saved, err := readJSON(fixture.out)
	if err != nil {
		t.Fatal(err)
	}
	if snippet := stringValue(asObject(asObject(saved)["payload"])["commandSnippet"]); !strings.HasPrefix(snippet, "VLLM_USE_FLASHINFER_SAMPLER=0 ") {
		t.Fatalf("recorded command must carry the fallback that actually ran: %q", snippet)
	}
	first, err := os.ReadFile(strings.TrimSuffix(fixture.out, ".json") + ".server.first-attempt.log")
	if err != nil || !strings.Contains(string(first), "run_ninja") {
		t.Fatalf("first attempt log = %q, %v", first, err)
	}
	var state struct {
		Path string `json:"path"`
	}
	data, _ := os.ReadFile(fixture.statePath)
	_ = json.Unmarshal(data, &state)
	exe, _ := os.Executable()
	if !strings.HasPrefix(state.Path, filepath.Dir(exe)+string(os.PathListSeparator)) {
		t.Fatalf("engine PATH = %q, want the engine binary's directory first", state.Path)
	}
}

func TestEngineFailureReasonFindsRootCause(t *testing.T) {
	log := `(EngineCore pid=1) ERROR 10-12 10:09:01 [core.py:1330]     raise ValueError(
(EngineCore pid=1) ValueError: max_num_seqs (256) exceeds available Mamba cache blocks (79). Please lower max_num_seqs to at most 79.
(APIServer pid=2)     raise RuntimeError(
(APIServer pid=2) RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}`
	if got := engineFailureReason(log); got != "ValueError: max_num_seqs (256) exceeds available Mamba cache blocks (79). Please lower max_num_seqs to at most 79." {
		t.Fatalf("reason = %q", got)
	}
	if got := engineFailureReason("INFO starting\nINFO loading weights"); got != "" {
		t.Fatalf("no exception must give no reason, got %q", got)
	}
}
