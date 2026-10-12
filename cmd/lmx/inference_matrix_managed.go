package main

// Managed mode for lmx speed-test matrix run: --model-path starts vLLM on a
// free localhost port sized for the requested matrix, runs the bench against
// it exactly like --base-url mode, and always stops the server afterwards.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

const matrixSamplerEnv = "VLLM_USE_FLASHINFER_SAMPLER"

// Overridden by tests.
var (
	managedEnginePollInterval = 2 * time.Second
	managedEngineStopGrace    = 30 * time.Second
	// vLLM defers SIGTERM while it loads and compiles; a server that never
	// became ready has nothing to drain, so it is killed sooner.
	managedEngineStartupStopGrace = 5 * time.Second
	// nvcc locations checked after PATH and $CUDA_HOME/bin.
	matrixNvccFallbacks = []string{"/usr/local/cuda/bin/nvcc"}
)

// vLLM installs commonly kept in a virtualenv that is not on PATH, relative to $HOME.
var matrixVLLMVenvCandidates = []string{"vllm/bin/vllm", ".venv/bin/vllm", "venvs/vllm/bin/vllm", ".local/share/vllm/bin/vllm"}

// Flags managed mode sets itself; engine args may not override them.
var matrixManagedReservedFlags = []string{"host", "port", "served-model-name"}

func runManagedInferenceMatrix(args cliArgs) error {
	if opt(args, "base-url") != "" {
		return cliError{"conflicting_options", "--model-path starts its own vLLM server and cannot be combined with --base-url.", []string{"Drop --base-url to let lmx start and stop vLLM, or drop --model-path to measure a server you started."}, nil}
	}
	hfID := firstNonEmpty(opt(args, "hf-id"), opt(args, "model"))
	if hfID == "" {
		return cliError{"missing_model", "speed-test matrix run --model-path requires --hf-id", []string{"Pass the HuggingFace model ID of the weights in --model-path; it is also the served model name."}, nil}
	}
	switch engine := strings.ToLower(firstNonEmpty(opt(args, "engine"), "vllm")); engine {
	case "vllm":
	case "sglang":
		return cliError{"unsupported_managed_engine", "lmx can start vLLM only; start SGLang yourself and pass --base-url.", []string{"python -m sglang.launch_server --model-path <dir> --port 30000, then lmx speed-test matrix run --base-url http://127.0.0.1:30000 --hf-id " + hfID + " --server-command \"<that command>\"."}, nil}
	default:
		return cliError{"unsupported_managed_engine", "Unknown --engine " + engine + " for speed-test matrix run.", []string{"Use --engine vllm (the default) with --model-path, or --base-url for an SGLang server you started."}, nil}
	}
	modelPath := opt(args, "model-path")
	if info, err := os.Stat(modelPath); err != nil || !info.IsDir() {
		return cliError{"invalid_model_path", "--model-path " + modelPath + " is not a local model directory.", []string{"Pass the directory that contains config.json and the weights, e.g. ~/models/Qwen3-0.6B."}, nil}
	}
	if _, err := os.Stat(filepath.Join(modelPath, "config.json")); err != nil {
		return cliError{"invalid_model_path", "--model-path " + modelPath + " has no config.json; lmx starts vLLM on Hugging Face model directories (safetensors).", []string{"Pass a directory with config.json and safetensors weights.", "For GGUF weights, start your server yourself and pass --base-url and --server-command."}, nil}
	}
	if abs, err := filepath.Abs(modelPath); err == nil {
		modelPath = abs
	}
	plan, err := resolveMatrixBenchPlan(args)
	if err != nil {
		return err
	}
	engineArgs := strings.Fields(opt(args, "engine-args"))
	engineFlags := launchFlags(engineArgs)
	for _, name := range matrixManagedReservedFlags {
		if _, ok := engineFlags[name]; ok {
			return cliError{"invalid_option", "--engine-args may not set --" + name + "; managed mode chooses it.", []string{"Remove --" + name + " from --engine-args."}, nil}
		}
	}
	gpuUtilization := firstNonEmpty(opt(args, "gpu-memory-utilization"), "0.9")
	if value, err := strconv.ParseFloat(gpuUtilization, 64); err != nil || value <= 0 || value > 1 {
		return cliError{"invalid_option", "--gpu-memory-utilization must be a number in (0, 1]", []string{"Pass e.g. --gpu-memory-utilization 0.9."}, nil}
	}
	startTimeout, err := intOption(args, 900, 1, "start-timeout")
	if err != nil {
		return err
	}

	maxModelLen := 0
	if explicit := engineFlags["max-model-len"]; explicit == "" {
		modelLimit, limitSource := matrixModelContextLimit(modelPath)
		var dropped []matrixContextValue
		maxModelLen, plan.Contexts, dropped, err = chooseManagedMaxModelLen(plan.Contexts, plan.MaxTokens, modelLimit)
		if err != nil {
			return err
		}
		if len(dropped) > 0 {
			plan.Skipped = append(plan.Skipped, dropped...)
			printStatus(args, "inference_matrix_contexts_skipped", map[string]any{"skipped": joinMatrixContexts(dropped), "contexts": joinMatrixContexts(plan.Contexts), "modelLimit": modelLimit, "source": limitSource, "maxTokens": plan.MaxTokens, "reason": "context + max tokens exceeds the model's max_position_embeddings"})
		}
	}

	bin, err := findMatrixEngineBinary(opt(args, "engine-bin"))
	if err != nil {
		return err
	}
	env := matrixSamplerFallbackEnv(matrixNvccPath(), os.Environ())
	port, err := freeLocalPort()
	if err != nil {
		return err
	}
	argv := []string{bin, "serve", modelPath, "--served-model-name", hfID, "--host", "127.0.0.1", "--port", strconv.Itoa(port)}
	if maxModelLen > 0 {
		argv = append(argv, "--max-model-len", strconv.Itoa(maxModelLen))
	}
	if _, ok := engineFlags["gpu-memory-utilization"]; !ok {
		argv = append(argv, "--gpu-memory-utilization", gpuUtilization)
	}
	argv = append(argv, engineArgs...)
	command := strings.Join(append(append([]string{}, env...), shellJoin(redactSecretArgs(argv))), " ")
	// Fail before the (minutes-long) server start on problems the run would
	// only report after it: an undetectable weight format, a missing bench Python.
	if opt(args, "quantization") == "" {
		if quant, resolution := matrixQuantizationFromModel(argv, hfID, args); quant == "" {
			reason := firstNonEmpty(stringValue(resolution["unresolved"]), "config.json names no quantization or dtype")
			return cliError{"missing_quantization", "Could not detect the weight format of " + modelPath + ": " + reason + ".", []string{"Pass --quantization (e.g. BF16, FP8, NVFP4, AWQ); lmx did not start the server."}, resolution}
		}
	}
	if _, err := matrixBenchPython(args); err != nil {
		return err
	}

	outPath := matrixOutPath(args, hfID)
	if err := os.MkdirAll(filepath.Dir(outPath), 0o755); err != nil {
		return err
	}
	logPath := strings.TrimSuffix(outPath, filepath.Ext(outPath)) + ".server.log"

	runCtx, stopSignals := terminalSignalContext()
	defer stopSignals()
	status := map[string]any{"command": command, "log": logPath, "maxModelLen": maxModelLen, "contexts": joinMatrixContexts(plan.Contexts), "maxTokens": plan.MaxTokens}
	if len(env) > 0 {
		status["samplerFallback"] = matrixSamplerEnv + "=0 (no nvcc found; FlashInfer's sampler would JIT-compile at startup)"
	}
	printStatus(args, "inference_matrix_server_starting", status)
	baseURL := "http://127.0.0.1:" + strconv.Itoa(port)
	started := time.Now()
	var server *managedEngine
	defer func() {
		if server != nil {
			printStatus(args, "inference_matrix_server_stopping", map[string]any{"log": logPath})
			server.stop()
			printStatus(args, "inference_matrix_server_stopped", map[string]any{"log": logPath})
		}
	}()
	for attempt := 1; ; attempt++ {
		if server, err = startManagedEngine(argv, env, logPath); err != nil {
			return err
		}
		go func(s *managedEngine) {
			select {
			case <-runCtx.Done():
				s.stop()
			case <-s.done:
			}
		}(server)
		err = server.waitReady(runCtx, baseURL, hfID, time.Duration(startTimeout)*time.Second)
		if err == nil {
			break
		}
		// FlashInfer's sampler JIT-compiles at startup; when that build fails
		// (missing compiler, CUDA mismatch) retry once with vLLM's own sampler.
		if attempt == 1 && isCLIErrorCode(err, "engine_exited") && !matrixSamplerOverridden(os.Environ(), env) && flashInferJITFailed(logTail(logPath, 400)) {
			server.stop()
			_ = os.Rename(logPath, strings.TrimSuffix(logPath, ".log")+".first-attempt.log")
			env = append(env, matrixSamplerEnv+"=0")
			command = strings.Join(append(append([]string{}, env...), shellJoin(redactSecretArgs(argv))), " ")
			printStatus(args, "inference_matrix_server_retry", map[string]any{"reason": "FlashInfer's sampler failed to compile at startup; retrying with " + matrixSamplerEnv + "=0", "log": logPath, "command": command})
			continue
		}
		if tail := logTail(logPath, 40); tail != "" {
			printStatus(args, "inference_matrix_server_log_tail", map[string]any{"log": logPath, "lines": "\n" + tail})
		}
		return err
	}
	printStatus(args, "inference_matrix_server_ready", map[string]any{"baseUrl": baseURL, "seconds": int(time.Since(started).Seconds())})

	args.opts["base-url"] = baseURL
	args.opts["served-model"] = hfID
	args.opts["out"] = outPath
	if opt(args, "server-command") == "" {
		args.opts["server-command"] = command
	}
	return runInferenceMatrixAt(runCtx, args, plan)
}

// matrixModelContextLimit reads max_position_embeddings from the model's
// config.json (multimodal configs nest it under text_config). 0 means unknown.
func matrixModelContextLimit(modelPath string) (int, string) {
	path := filepath.Join(modelPath, "config.json")
	data, err := os.ReadFile(path)
	if err != nil {
		return 0, ""
	}
	var config map[string]any
	if json.Unmarshal(data, &config) != nil {
		return 0, ""
	}
	for _, scope := range []map[string]any{config, asObject(config["text_config"])} {
		if value := int(numberField(scope, "max_position_embeddings")); value > 0 {
			return value, path
		}
	}
	return 0, ""
}

// chooseManagedMaxModelLen sizes --max-model-len for the largest context plus
// the output length (and prompt margin), rounded up to a multiple of 1024 and
// capped at the model's own limit. Contexts that cannot fit the model are dropped.
func chooseManagedMaxModelLen(contexts []matrixContextValue, maxTokens, modelLimit int) (int, []matrixContextValue, []matrixContextValue, error) {
	kept, dropped := []matrixContextValue{}, []matrixContextValue{}
	need := 0
	for _, value := range contexts {
		contextNeed := matrixContextNeed(value.Tokens, maxTokens)
		if modelLimit > 0 && contextNeed > modelLimit {
			dropped = append(dropped, value)
			continue
		}
		kept = append(kept, value)
		need = max(need, contextNeed)
	}
	if len(kept) == 0 {
		return 0, nil, dropped, cliError{"context_exceeds_model_limit", fmt.Sprintf("No requested context fits the model's max_position_embeddings %d with %d output tokens per request.", modelLimit, maxTokens), []string{"Pass smaller --contexts or a smaller --max-tokens.", "If the model supports a longer context (e.g. RoPE scaling), pass --engine-args=\"--max-model-len N\" yourself."}, map[string]any{"modelLimit": modelLimit, "maxTokens": maxTokens}}
	}
	maxModelLen := roundUpTo(need, 1024)
	if modelLimit > 0 && maxModelLen > modelLimit {
		maxModelLen = modelLimit
	}
	return maxModelLen, kept, dropped, nil
}

// findMatrixEngineBinary resolves the vLLM executable: --engine-bin, then vllm
// on PATH, then common virtualenv locations under $HOME.
func findMatrixEngineBinary(engineBin string) (string, error) {
	if engineBin != "" {
		if strings.ContainsRune(engineBin, os.PathSeparator) {
			if isExecutableFile(engineBin) {
				return engineBin, nil
			}
		} else if path, err := exec.LookPath(engineBin); err == nil {
			return path, nil
		}
		return "", cliError{"engine_not_found", "--engine-bin " + engineBin + " is not an executable file.", []string{"Pass the vllm executable inside its virtualenv, e.g. --engine-bin ~/vllm/bin/vllm."}, nil}
	}
	if path, err := exec.LookPath("vllm"); err == nil {
		return path, nil
	}
	searched := []string{"vllm on PATH"}
	if home, err := os.UserHomeDir(); err == nil {
		for _, candidate := range matrixVLLMVenvCandidates {
			path := filepath.Join(home, candidate)
			if isExecutableFile(path) {
				return path, nil
			}
			searched = append(searched, path)
		}
	}
	return "", cliError{"engine_not_found", "Could not find the vllm executable.", []string{"Pass --engine-bin /path/to/venv/bin/vllm.", "Or install vLLM: python3 -m venv ~/vllm && ~/vllm/bin/pip install vllm."}, map[string]any{"searched": searched}}
}

func isExecutableFile(path string) bool {
	info, err := os.Stat(path)
	return err == nil && info.Mode().IsRegular() && info.Mode().Perm()&0o111 != 0
}

// matrixNvccPath finds nvcc on PATH, in $CUDA_HOME/bin, or a standard CUDA install.
func matrixNvccPath() string {
	if path, err := exec.LookPath("nvcc"); err == nil {
		return path
	}
	candidates := append([]string{}, matrixNvccFallbacks...)
	if cudaHome := os.Getenv("CUDA_HOME"); cudaHome != "" {
		candidates = append([]string{filepath.Join(cudaHome, "bin", "nvcc")}, candidates...)
	}
	for _, path := range candidates {
		if isExecutableFile(path) {
			return path
		}
	}
	return ""
}

// matrixSamplerFallbackEnv disables vLLM's FlashInfer sampler when nvcc is
// missing: it JIT-compiles at startup and crashes the server without a CUDA
// compiler. An explicit VLLM_USE_FLASHINFER_SAMPLER in the environment wins.
func matrixSamplerFallbackEnv(nvccPath string, environ []string) []string {
	if nvccPath != "" || matrixSamplerOverridden(environ, nil) {
		return nil
	}
	return []string{matrixSamplerEnv + "=0"}
}

// matrixSamplerOverridden reports whether VLLM_USE_FLASHINFER_SAMPLER is
// already set by the user or by an earlier fallback.
func matrixSamplerOverridden(environ, extra []string) bool {
	for _, entry := range append(append([]string{}, environ...), extra...) {
		if strings.HasPrefix(entry, matrixSamplerEnv+"=") {
			return true
		}
	}
	return false
}

// flashInferJITFailed recognises a vLLM startup crash inside FlashInfer's
// JIT build (missing ninja/nvcc, compiler errors).
func flashInferJITFailed(log string) bool {
	return strings.Contains(log, "flashinfer/jit") && (strings.Contains(log, "build_and_load") || strings.Contains(log, "run_ninja") || strings.Contains(log, "nvcc"))
}

// matrixEngineEnviron returns environ with the engine binary's directory
// first on PATH, as an activated virtualenv would have it. vLLM's JIT
// builds (FlashInfer, Triton) look up ninja and friends on PATH, and those
// live next to the vllm entry point in its venv.
func matrixEngineEnviron(bin string, environ []string) []string {
	dir := filepath.Dir(bin)
	out := make([]string, 0, len(environ)+1)
	found := false
	for _, entry := range environ {
		if strings.HasPrefix(entry, "PATH=") {
			found = true
			current := strings.TrimPrefix(entry, "PATH=")
			if current != dir && !strings.HasPrefix(current, dir+string(os.PathListSeparator)) {
				entry = "PATH=" + dir + string(os.PathListSeparator) + current
			}
		}
		out = append(out, entry)
	}
	if !found {
		out = append(out, "PATH="+dir)
	}
	return out
}

func freeLocalPort() (int, error) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 0, err
	}
	defer listener.Close()
	return listener.Addr().(*net.TCPAddr).Port, nil
}

type managedEngine struct {
	cmd     *exec.Cmd
	logPath string
	done    chan struct{}
	waitErr error
	once    sync.Once
	ready   atomic.Bool
}

// startManagedEngine starts argv in its own process group with stdout and
// stderr appended to logPath.
func startManagedEngine(argv, env []string, logPath string) (*managedEngine, error) {
	logFile, err := os.Create(logPath)
	if err != nil {
		return nil, err
	}
	cmd := exec.Command(argv[0], argv[1:]...)
	cmd.Env = matrixEngineEnviron(argv[0], append(os.Environ(), env...))
	cmd.Stdout = logFile
	cmd.Stderr = logFile
	configureDetachedCommand(cmd)
	if err := cmd.Start(); err != nil {
		logFile.Close()
		return nil, cliError{"engine_start_failed", "Could not start " + argv[0] + ": " + err.Error(), []string{"Check --engine-bin."}, nil}
	}
	server := &managedEngine{cmd: cmd, logPath: logPath, done: make(chan struct{})}
	go func() {
		server.waitErr = cmd.Wait()
		logFile.Close()
		close(server.done)
	}()
	return server, nil
}

// waitReady polls until GET /health is 200 and /v1/models lists model.
func (s *managedEngine) waitReady(ctx context.Context, baseURL, model string, timeout time.Duration) error {
	deadline := time.NewTimer(timeout)
	defer deadline.Stop()
	ticker := time.NewTicker(managedEnginePollInterval)
	defer ticker.Stop()
	client := &http.Client{Timeout: 5 * time.Second}
	for {
		if managedEngineReady(client, baseURL, model) {
			s.ready.Store(true)
			return nil
		}
		select {
		case <-ctx.Done():
			return cliError{"interrupted", "Interrupted while waiting for vLLM to start; the server was stopped.", nil, map[string]any{"log": s.logPath}}
		case <-s.done:
			message := fmt.Sprintf("vLLM exited before it was ready (%v).", s.waitErr)
			if reason := engineFailureReason(logTail(s.logPath, 400)); reason != "" {
				message = "vLLM exited before it was ready: " + reason
			}
			return cliError{"engine_exited", message, []string{"Read the server log: " + s.logPath, "Pass extra server flags with --engine-args=\"...\" (e.g. --max-num-seqs 64, --enforce-eager), or change --gpu-memory-utilization."}, map[string]any{"log": s.logPath, "logTail": strings.Split(logTail(s.logPath, 40), "\n")}}
		case <-deadline.C:
			return cliError{"engine_start_timeout", fmt.Sprintf("vLLM was not ready within %s; the server was stopped.", timeout), []string{"Read the server log: " + s.logPath, "Raise --start-timeout for large models or slow first-time compilation."}, map[string]any{"log": s.logPath, "logTail": strings.Split(logTail(s.logPath, 40), "\n")}}
		case <-ticker.C:
		}
	}
}

func managedEngineReady(client *http.Client, baseURL, model string) bool {
	res, err := client.Get(baseURL + "/health")
	if err != nil {
		return false
	}
	res.Body.Close()
	if res.StatusCode != http.StatusOK {
		return false
	}
	res, err = client.Get(baseURL + "/v1/models")
	if err != nil {
		return false
	}
	defer res.Body.Close()
	var body map[string]any
	if res.StatusCode != http.StatusOK || json.NewDecoder(res.Body).Decode(&body) != nil {
		return false
	}
	for _, item := range modelInfoItems(body) {
		if stringValue(asObject(item)["id"]) == model {
			return true
		}
	}
	return false
}

// stop sends SIGTERM to the server's process group, SIGKILL after the grace
// period, waits for the server to exit, then kills any worker left in its
// group. Safe to call repeatedly and concurrently.
func (s *managedEngine) stop() {
	s.once.Do(func() {
		pid := s.cmd.Process.Pid
		grace := managedEngineStopGrace
		if !s.ready.Load() {
			grace = managedEngineStartupStopGrace
		}
		_ = signalDetachedProcess(pid, false)
		select {
		case <-s.done:
		case <-time.After(grace):
		}
		_ = signalDetachedProcess(pid, true)
		<-s.done
	})
}

// logTail returns the last n lines of a file.
func logTail(path string, n int) string {
	data, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	lines := strings.Split(strings.TrimRight(string(data), "\n"), "\n")
	if len(lines) > n {
		lines = lines[len(lines)-n:]
	}
	return strings.Join(lines, "\n")
}

func isCLIErrorCode(err error, code string) bool {
	var cliErr cliError
	return errors.As(err, &cliErr) && cliErr.Code == code
}

var (
	engineErrorLine   = regexp.MustCompile(`\b[A-Z][A-Za-z]*(?:Error|Exception): .+`)
	engineLogPrefixRE = regexp.MustCompile(`^\((?:APIServer|EngineCore)[^)]*\)\s*(?:ERROR \S+ \S+ \[[^\]]+\]\s*)?`)
)

// engineFailureReason returns the root-cause exception line from a vLLM
// startup log: the last "XxxError: ..." line that is not vLLM's generic
// "Engine core initialization failed" wrapper.
func engineFailureReason(log string) string {
	reason := ""
	for _, line := range strings.Split(log, "\n") {
		line = strings.TrimSpace(engineLogPrefixRE.ReplaceAllString(line, ""))
		match := engineErrorLine.FindString(line)
		if match == "" || strings.Contains(match, "Engine core initialization failed") {
			continue
		}
		reason = match
	}
	if len(reason) > 400 {
		reason = reason[:400] + "…"
	}
	return reason
}
