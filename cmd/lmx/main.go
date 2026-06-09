package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/csv"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"
)

const defaultAPIURL = "https://www.localmaxxing.com"
const defaultHFAPIURL = "https://huggingface.co"
const defaultEndpointTimeout = 10 * time.Minute
const remoteKVCacheColdMethodology = "Single streaming request with inline filler padded to target context size; measures cold prefill + decode at that context depth."
const remoteKVCacheReuseMethodology = "Two-step remote cache-reuse probe: pre-warm target context, then time a streaming request with the same prefix plus probe; measures cached-prefix decode at that context depth."
const remoteKVCacheFallbackWarning = "Remote OpenAI-compatible endpoints do not provide a portable persistent KV-cache session API; this sweep resends the full prefix at each depth and can only verify cache reuse when backend-specific cache metrics are exposed. Results may fall back to cold depth TPS instead of retained KV-cache TPS."

var goldFieldNames = map[string]bool{
	"gold": true, "answer": true, "referenceAnswer": true, "expectedAnswer": true,
	"correctAnswer": true, "label": true, "target": true,
}

type cliArgs struct {
	positional []string
	opts       map[string]string
	flags      map[string]bool
}

type cliError struct {
	Code    string
	Message string
	Hints   []string
	Details any
}

type tokenCountResult struct {
	Count  int
	Source string
}

type detectedEngine struct {
	Name             string            `json:"name"`
	Installed        bool              `json:"installed"`
	Binaries         map[string]string `json:"binaries,omitempty"`
	ServerCommand    string            `json:"serverCommand,omitempty"`
	BenchmarkCommand string            `json:"benchmarkCommand,omitempty"`
	Notes            []string          `json:"notes,omitempty"`
}

func (e cliError) Error() string { return e.Message }

func main() {
	args := parseArgs(os.Args[1:])
	if err := runWithArgs(args); err != nil {
		printError(args, err)
		os.Exit(1)
	}
}

func run(argv []string) error {
	return runWithArgs(parseArgs(argv))
}

func runWithArgs(args cliArgs) error {
	if err := applyProfile(&args); err != nil {
		return err
	}
	normalizeEndpointArgs(&args)
	cmd := positional(args, 0)
	if cmd == "" || hasFlag(args, "help") || !knownTopLevel(cmd) {
		usage()
		if cmd == "" || hasFlag(args, "help") {
			return nil
		}
		return errors.New("unknown command")
	}

	switch cmd {
	case "auth":
		return handleAuth(args)
	case "hardware":
		return handleHardware(positional(args, 1), args)
	case "context", "agent-context":
		return handleContext(args)
	case "model":
		return handleModel(positional(args, 1), positional(args, 2), args)
	case "profile":
		return handleProfile(positional(args, 1), positional(args, 2), args)
	case "engines", "engine":
		return handleEngines(args)
	case "server":
		return handleServer(positional(args, 1), positional(args, 2), args)
	case "endpoint":
		return handleEndpoint(positional(args, 1), args)
	case "kvcache", "kv-cache", "context-sweep":
		return handleKVCache(positional(args, 1), positional(args, 2), args)
	case "benchmark", "bench":
		return handleBenchmark(positional(args, 1), positional(args, 2), args)
	case "eval":
		sub := positional(args, 1)
		switch sub {
		case "storage":
			return handleStorage(positional(args, 2), positional(args, 3), args, "")
		case "artifact", "artifacts":
			return handleStorage(positional(args, 2), positional(args, 3), args, "artifact")
		case "suite":
			return handleSuite(positional(args, 2), positional(args, 3), args)
		case "execute":
			return handleExecute(positional(args, 2), args)
		case "lm-eval", "lmeval":
			return handleLmEval(positional(args, 2), args)
		case "run":
			return handleEvalRun(positional(args, 2), args)
		}
	}

	usage()
	return errors.New("unknown command")
}

func knownTopLevel(cmd string) bool {
	switch cmd {
	case "eval", "benchmark", "bench", "auth", "hardware", "context", "agent-context", "model", "profile", "engines", "engine", "server", "endpoint", "kvcache", "kv-cache", "context-sweep":
		return true
	default:
		return false
	}
}

func parseArgs(argv []string) cliArgs {
	args := cliArgs{opts: map[string]string{}, flags: map[string]bool{}}
	for i := 0; i < len(argv); i++ {
		arg := argv[i]
		if !strings.HasPrefix(arg, "--") {
			args.positional = append(args.positional, arg)
			continue
		}
		key := strings.TrimPrefix(arg, "--")
		if eq := strings.Index(key, "="); eq >= 0 {
			args.opts[key[:eq]] = key[eq+1:]
			continue
		}
		if i+1 >= len(argv) || strings.HasPrefix(argv[i+1], "--") {
			args.flags[key] = true
			continue
		}
		args.opts[key] = argv[i+1]
		i++
	}
	return args
}

func positional(args cliArgs, index int) string {
	if index < 0 || index >= len(args.positional) {
		return ""
	}
	return args.positional[index]
}

func opt(args cliArgs, key string) string   { return args.opts[key] }
func hasFlag(args cliArgs, key string) bool { return args.flags[key] }

func requireOpt(args cliArgs, key string) (string, error) {
	value := opt(args, key)
	if value == "" {
		return "", cliError{"missing_option", fmt.Sprintf("--%s is required", key), []string{fmt.Sprintf("Pass --%s <value>. Run lmx --help for examples.", key)}, nil}
	}
	return value, nil
}

func apiURL(args cliArgs) string {
	value := opt(args, "api-url")
	if value == "" {
		value = defaultAPIURL
	}
	return strings.TrimRight(value, "/")
}

func openAIBaseURL(raw string) string {
	value := strings.TrimRight(strings.TrimSpace(raw), "/")
	if strings.HasSuffix(value, "/v1") {
		return strings.TrimSuffix(value, "/v1")
	}
	return value
}

func normalizeEndpointArgs(args *cliArgs) {
	if value := opt(*args, "base-url"); value != "" {
		args.opts["base-url"] = openAIBaseURL(value)
	}
}

func configFile() string {
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".config", "localmaxxing", "config.json")
}

func loadConfig() map[string]any {
	data, err := os.ReadFile(configFile())
	if err != nil {
		return map[string]any{}
	}
	var cfg map[string]any
	if json.Unmarshal(data, &cfg) != nil {
		return map[string]any{}
	}
	return cfg
}

func profilesDir() string {
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".config", "localmaxxing", "profiles")
}

func profileFile(name string) string {
	clean := strings.NewReplacer("/", "-", "\\", "-", "..", "-").Replace(name)
	return filepath.Join(profilesDir(), clean+".json")
}

func applyProfile(args *cliArgs) error {
	name := opt(*args, "profile")
	if name == "" || positional(*args, 0) == "profile" {
		return nil
	}
	profile, err := readJSON(profileFile(name))
	if err != nil {
		return cliError{"profile_read_error", "Could not load profile \"" + name + "\".", []string{"Run lmx profile list to see saved profiles.", "Run lmx profile save " + name + " ... to create it."}, err.Error()}
	}
	obj := asObject(profile)
	if obj == nil {
		return cliError{"profile_invalid", "Profile must be a JSON object.", nil, profile}
	}
	profileOpts := asObject(obj["opts"])
	if profileOpts == nil {
		profileOpts = obj
	}
	for key, value := range profileOpts {
		if _, exists := args.opts[key]; exists {
			continue
		}
		if args.flags[key] {
			continue
		}
		switch typed := value.(type) {
		case string:
			args.opts[key] = typed
		case bool:
			if typed {
				args.flags[key] = true
			}
		case float64:
			args.opts[key] = strconv.FormatFloat(typed, 'f', -1, 64)
		}
	}
	printStatus(*args, "profile_loaded", map[string]any{"profile": name})
	return nil
}

func saveConfig(cfg map[string]any) error {
	path := configFile()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return writeJSON(path, cfg)
}

func apiKey(args cliArgs) string {
	if key := opt(args, "api-key"); key != "" {
		return key
	}
	if key := os.Getenv("LMX_API_KEY"); key != "" {
		return key
	}
	if key, ok := loadConfig()["apiKey"].(string); ok {
		return key
	}
	return ""
}

func fetchJSON(method, rawURL, key string, body any) (any, error) {
	var reader io.Reader
	if body != nil {
		data, err := json.Marshal(body)
		if err != nil {
			return nil, err
		}
		reader = bytes.NewReader(data)
	}
	req, err := http.NewRequest(method, rawURL, reader)
	if err != nil {
		return nil, err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	if key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, cliError{"network_error", fmt.Sprintf("Could not reach %s: %v", rawURL, err), []string{"Check --api-url or endpoint URL.", "If this is a local model server, make sure it is running and reachable."}, nil}
	}
	defer res.Body.Close()
	text, _ := io.ReadAll(res.Body)
	var parsed any
	if len(text) > 0 && json.Unmarshal(text, &parsed) != nil {
		parsed = string(text)
	}
	if res.StatusCode < 200 || res.StatusCode >= 300 {
		hints := []string{}
		switch res.StatusCode {
		case 401:
			hints = append(hints, "Check --api-key or LMX_API_KEY.")
		case 400:
			hints = append(hints, "Run the relevant validate/dry-run command and fix the reported field.")
		case 404:
			hints = append(hints, "Check the suite slug or API URL.")
		case 409:
			hints = append(hints, "Choose a different suite slug; this one already exists.")
		case 422:
			hints = append(hints, "The suite or run shape is valid JSON but incompatible with the API rules.")
		case 429:
			hints = append(hints, "Wait for the rate-limit window before submitting again.")
		}
		return nil, cliError{"api_error", fmt.Sprintf("%d %s: %s", res.StatusCode, res.Status, apiMessage(parsed, string(text))), hints, parsed}
	}
	return parsed, nil
}

func apiMessage(parsed any, fallback string) string {
	if obj, ok := parsed.(map[string]any); ok {
		if msg, ok := obj["error"].(string); ok {
			return msg
		}
	}
	return fallback
}

func readJSON(path string) (any, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, cliError{"file_read_error", fmt.Sprintf("Could not read %s: %v", path, err), []string{"Check that the path exists and is readable.", "Use an absolute path if the file is outside the current directory."}, nil}
	}
	var value any
	if err := json.Unmarshal(data, &value); err != nil {
		return nil, cliError{"json_parse_error", fmt.Sprintf("Could not parse %s as JSON: %v", path, err), []string{"Fix the JSON syntax and retry."}, nil}
	}
	return value, nil
}

func writeJSON(path string, value any) error {
	data, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	return os.WriteFile(path, data, 0o644)
}

func writeOrPrintJSON(title string, args cliArgs, value any) error {
	out := opt(args, "out")
	if out == "" {
		data, _ := json.MarshalIndent(value, "", "  ")
		fmt.Println(string(data))
		return nil
	}
	if err := writeJSON(out, value); err != nil {
		return err
	}
	printInfo(title+"_written", map[string]any{"path": out})
	return nil
}

func handleAuth(args cliArgs) error {
	key := opt(args, "key")
	if key == "" {
		key = opt(args, "api-key")
	}
	if hasFlag(args, "logout") {
		if err := saveConfig(map[string]any{}); err != nil {
			return err
		}
		printInfo("auth_cleared", map[string]any{"path": configFile()})
		return nil
	}
	if key != "" {
		if err := saveConfig(map[string]any{"apiKey": key, "authProvider": "manual", "authSavedAt": time.Now().UTC().Format(time.RFC3339)}); err != nil {
			return err
		}
		printInfo("auth_saved", map[string]any{"path": configFile(), "key": redactKey(key)})
		return nil
	}
	cfg := loadConfig()
	source := configFile()
	key = os.Getenv("LMX_API_KEY")
	if key != "" {
		source = "LMX_API_KEY"
	} else if stored, ok := cfg["apiKey"].(string); ok {
		key = stored
	}
	if key == "" {
		printInfo("auth_missing", map[string]any{"next": "Run lmx auth --key bhk_... or set LMX_API_KEY."})
		return nil
	}
	printInfo("auth_status", map[string]any{"source": source, "key": redactKey(key), "provider": cfg["authProvider"]})
	return nil
}

func handleProfile(action, name string, args cliArgs) error {
	switch action {
	case "save":
		if name == "" {
			return errors.New("profile save requires a profile name")
		}
		profileOpts := map[string]any{}
		for _, key := range []string{"mode", "api-url", "base-url", "model", "hf-id", "served-model", "model-name", "quantization", "hardware", "model-path", "command", "max-tokens", "prompt-tokens", "output-tokens", "engine", "backend", "bench-kind", "benchmark-output", "benchmark-bin", "server-bin", "python-bin", "host", "port", "input-len", "output-len", "num-prompts", "tensor-parallel", "context-length", "gpu-layers", "depth", "context-depth", "batch-size", "micro-batch-size", "ubatch-size", "repetitions", "runs", "benchmark-format", "bench-format", "output-format", "cache-type-k", "cache-type-v", "extra-server-args", "extra-bench-args"} {
			if value := opt(args, key); value != "" {
				profileOpts[key] = value
			}
		}
		for _, key := range []string{"no-stream", "flash-attn", "no-flash-attn"} {
			if hasFlag(args, key) {
				profileOpts[key] = true
			}
		}
		if len(profileOpts) == 0 {
			return cliError{"profile_empty", "No profile options were provided.", []string{"Example: lmx profile save my-4090 --mode local --hf-id Qwen/Qwen3-8B --quantization Q4_K_M --hardware hardware.json"}, nil}
		}
		if err := os.MkdirAll(profilesDir(), 0o755); err != nil {
			return err
		}
		payload := map[string]any{"name": name, "createdAt": time.Now().UTC().Format(time.RFC3339), "opts": profileOpts}
		if err := writeJSON(profileFile(name), payload); err != nil {
			return err
		}
		printInfo("profile_saved", map[string]any{"profile": name, "path": profileFile(name)})
		fmt.Println("Use it with:")
		fmt.Println("  lmx benchmark run <engine> --profile " + name)
		fmt.Println("  lmx server run <engine> --profile " + name)
		return nil
	case "list":
		entries, err := os.ReadDir(profilesDir())
		if err != nil && !os.IsNotExist(err) {
			return err
		}
		profiles := []string{}
		for _, entry := range entries {
			if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".json") {
				continue
			}
			profiles = append(profiles, strings.TrimSuffix(entry.Name(), ".json"))
		}
		return writeOrPrintJSON("profiles", args, map[string]any{"profiles": profiles})
	case "show":
		if name == "" {
			return errors.New("profile show requires a profile name")
		}
		profile, err := readJSON(profileFile(name))
		if err != nil {
			return err
		}
		return writeOrPrintJSON("profile", args, profile)
	case "delete", "rm":
		if name == "" {
			return errors.New("profile delete requires a profile name")
		}
		if err := os.Remove(profileFile(name)); err != nil {
			return err
		}
		printInfo("profile_deleted", map[string]any{"profile": name})
		return nil
	default:
		return errors.New("Unknown profile command. Use save, list, show, or delete.")
	}
}

func redactKey(key string) string {
	if len(key) <= 8 {
		return key + "..."
	}
	return key[:8] + "..."
}

func handleHardware(action string, args cliArgs) error {
	hardware := detectHardware()
	if action == "init" && opt(args, "out") == "" {
		args.opts["out"] = "hardware.json"
	}
	if hasFlag(args, "out") || opt(args, "out") != "" || action == "init" {
		out := opt(args, "out")
		if out == "" {
			out = "hardware.json"
		}
		if err := writeJSON(out, hardware); err != nil {
			return err
		}
		fields := map[string]any{"path": out, "hwClass": hardware["hwClass"], "gpuName": hardware["gpuName"], "vramGb": hardware["vramGb"]}
		if gpuCount := numberField(hardware, "gpuCount"); gpuCount > 1 {
			fields["gpuCount"] = gpuCount
			fields["note"] = "Multiple GPUs detected; verify the benchmark runtime used the intended GPU count before submitting."
		}
		printInfo("hardware_written", fields)
		fmt.Println("Use it with:")
		fmt.Println("  lmx benchmark run <engine> --hardware " + out)
		return nil
	}
	data, _ := json.MarshalIndent(hardware, "", "  ")
	fmt.Println(string(data))
	return nil
}

func handleEngines(args cliArgs) error {
	return writeOrPrintJSON("engines", args, map[string]any{"engines": detectInferenceEngines(args)})
}

func handleServer(action, target string, args cliArgs) error {
	if action != "run" && action != "start" && action != "dry-run" {
		return errors.New("Unknown server command. Use run or dry-run.")
	}
	if action == "dry-run" {
		args.flags["dry-run"] = true
	}
	engineName, err := resolveEngineName(target, args, false)
	if err != nil {
		return err
	}
	commandSnippet := localServerCommand(engineName, args)
	if commandSnippet == "" {
		return cliError{"server_command_unavailable", "Could not build a local server command for " + engineName + ".", []string{"Pass --model-path for llama.cpp or SGLang.", "Pass --hf-id or --model for vLLM/SGLang.", "Pass --command <server command> to run a custom engine."}, map[string]any{"detectedEngines": detectInferenceEngines(args)}}
	}
	payload := map[string]any{"engineName": engineName, "command": commandSnippet, "detectedEngines": detectInferenceEngines(args)}
	if hasFlag(args, "dry-run") || opt(args, "out") != "" {
		return writeOrPrintJSON("server_plan", args, payload)
	}
	printInfo("server_command_start", map[string]any{"engine": engineName, "command": commandSnippet})
	return runLongCommand(commandSnippet)
}

func handleEndpoint(action string, args cliArgs) error {
	if action == "" {
		action = "discover"
	}
	if action != "discover" && action != "scan" {
		return errors.New("Unknown endpoint command. Use discover.")
	}
	candidates := endpointDiscoveryCandidates(args)
	results := []any{}
	for _, baseURL := range candidates {
		model, info, err := detectServedModel(baseURL, opt(args, "model-api-key"), firstNonEmpty(opt(args, "served-model"), opt(args, "model-name")))
		result := map[string]any{"baseUrl": baseURL, "ok": err == nil}
		if err != nil {
			result["error"] = err.Error()
			results = append(results, result)
			continue
		}
		result["servedModel"] = model
		if quant := quantizationFromModelInfo(info); quant != "" {
			result["quantization"] = quant
		}
		if hfID := firstNonEmpty(opt(args, "hf-id"), opt(args, "model")); hfID != "" {
			cmd := []string{"lmx", "benchmark", "run", firstNonEmpty(opt(args, "engine"), "llama.cpp"), "--mode", "remote"}
			appendShellArg(&cmd, "--base-url", baseURL)
			appendShellArg(&cmd, "--served-model", model)
			appendShellArg(&cmd, "--hf-id", hfID)
			if quant := firstNonEmpty(opt(args, "quantization"), stringValue(result["quantization"])); quant != "" {
				appendShellArg(&cmd, "--quantization", quant)
			}
			if hardware := opt(args, "hardware"); hardware != "" {
				appendShellArg(&cmd, "--hardware", hardware)
			}
			result["benchmarkCommand"] = strings.Join(cmd, " ")
		}
		results = append(results, result)
	}
	return writeOrPrintJSON("endpoint_discovery", args, map[string]any{"endpoints": results})
}

func endpointDiscoveryCandidates(args cliArgs) []string {
	if baseURL := opt(args, "base-url"); baseURL != "" {
		return []string{openAIBaseURL(baseURL)}
	}
	return []string{"http://localhost:8080", "http://localhost:8000", "http://localhost:11434", "http://127.0.0.1:30000"}
}

func detectInferenceEngines(args cliArgs) []detectedEngine {
	engines := []detectedEngine{
		detectVLLM(args),
		detectLlamaCPP(args),
		detectSGLang(args),
		detectOllama(args),
	}
	installed := []detectedEngine{}
	for _, engine := range engines {
		if engine.Installed {
			installed = append(installed, engine)
		}
	}
	return installed
}

func detectVLLM(args cliArgs) detectedEngine {
	binaries := map[string]string{}
	if path, ok := lookupExecutable("vllm"); ok {
		binaries["vllm"] = path
	}
	engine := detectedEngine{Name: "vllm", Installed: len(binaries) > 0, Binaries: binaries}
	if engine.Installed {
		engine.ServerCommand = localServerCommand("vllm", args)
		engine.BenchmarkCommand = localBenchmarkCommand("vllm", args)
	}
	return engine
}

func detectLlamaCPP(args cliArgs) detectedEngine {
	binaries := map[string]string{}
	for _, name := range []string{"llama-server", "llama-bench", "llama-cli"} {
		if path, ok := lookupExecutable(name); ok {
			binaries[name] = path
		}
	}
	engine := detectedEngine{Name: "llama.cpp", Installed: len(binaries) > 0, Binaries: binaries}
	if engine.Installed {
		engine.ServerCommand = localServerCommand("llama.cpp", args)
		engine.BenchmarkCommand = localBenchmarkCommand("llama.cpp", args)
		if _, ok := binaries["llama-bench"]; !ok {
			engine.Notes = append(engine.Notes, "llama-bench was not found; benchmark run needs --command or explicit metrics.")
		}
	}
	return engine
}

func detectSGLang(args cliArgs) detectedEngine {
	binaries := map[string]string{}
	if path, ok := lookupExecutable("sglang"); ok {
		binaries["sglang"] = path
	}
	pythonPath := ""
	if path, ok := lookupExecutable(firstNonEmpty(opt(args, "python-bin"), "python3")); ok {
		pythonPath = path
		binaries["python"] = path
	}
	engine := detectedEngine{Name: "sglang", Installed: binaries["sglang"] != "" || hasPythonModule(pythonPath, "sglang"), Binaries: binaries}
	if engine.Installed {
		engine.ServerCommand = localServerCommand("sglang", args)
		engine.BenchmarkCommand = localBenchmarkCommand("sglang", args)
		engine.Notes = append(engine.Notes, "SGLang detection confirms Python is available; command execution will fail if the sglang module is not installed.")
	}
	return engine
}

func detectOllama(args cliArgs) detectedEngine {
	binaries := map[string]string{}
	if path, ok := lookupExecutable("ollama"); ok {
		binaries["ollama"] = path
	}
	engine := detectedEngine{Name: "ollama", Installed: len(binaries) > 0, Binaries: binaries}
	if engine.Installed {
		engine.ServerCommand = localServerCommand("ollama", args)
		engine.Notes = append(engine.Notes, "Ollama is detected for serving; benchmark with --mode remote --base-url http://localhost:11434.")
	}
	return engine
}

func lookupExecutable(name string) (string, bool) {
	if name == "" {
		return "", false
	}
	path, err := exec.LookPath(name)
	return path, err == nil
}

func hasPythonModule(pythonPath, module string) bool {
	if pythonPath == "" || module == "" {
		return false
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	return exec.CommandContext(ctx, pythonPath, "-c", "import "+module).Run() == nil
}

func resolveEngineName(target string, args cliArgs, requireBenchmark bool) (string, error) {
	engineName := normalizeEngineName(firstNonEmpty(opt(args, "engine"), target))
	if engineName != "" {
		return engineName, nil
	}
	for _, engine := range detectInferenceEngines(args) {
		if requireBenchmark && engine.BenchmarkCommand == "" {
			continue
		}
		return engine.Name, nil
	}
	hints := []string{"Install vLLM, llama.cpp, SGLang, or Ollama, then retry.", "Pass an engine positionally, e.g. lmx benchmark run llama.cpp, or with --engine vllm."}
	if requireBenchmark {
		hints = append(hints, "For local benchmark generation, install vllm or llama-bench, or pass --command <benchmark command>.")
	}
	return "", cliError{"missing_engine", "Could not detect an installed local inference engine.", hints, nil}
}

func localServerCommand(engineName string, args cliArgs) string {
	model := firstNonEmpty(opt(args, "hf-id"), opt(args, "model"))
	host := firstNonEmpty(opt(args, "host"), "0.0.0.0")
	port := firstNonEmpty(opt(args, "port"), "8000")
	switch engineName {
	case "vllm":
		if model == "" {
			return ""
		}
		cmd := []string{shellQuote(firstNonEmpty(opt(args, "server-bin"), "vllm")), "serve", shellQuote(model), "--host", shellQuote(host), "--port", shellQuote(port)}
		appendShellArg(&cmd, "--served-model-name", firstNonEmpty(opt(args, "served-model"), opt(args, "model-name")))
		appendShellArg(&cmd, "--tensor-parallel-size", opt(args, "tensor-parallel"))
		appendShellArg(&cmd, "--max-model-len", opt(args, "context-length"))
		appendExtraArgs(&cmd, opt(args, "extra-server-args"))
		return strings.Join(cmd, " ")
	case "llama.cpp":
		if opt(args, "model-path") == "" {
			return ""
		}
		cmd := []string{shellQuote(resolvedExecutable(firstNonEmpty(opt(args, "server-bin"), "llama-server"))), "-m", shellQuote(opt(args, "model-path")), "--host", shellQuote(host), "--port", shellQuote(port)}
		appendShellArg(&cmd, "-c", opt(args, "context-length"))
		appendShellArg(&cmd, "-ngl", opt(args, "gpu-layers"))
		appendExtraArgs(&cmd, opt(args, "extra-server-args"))
		return strings.Join(cmd, " ")
	case "sglang":
		modelPath := firstNonEmpty(opt(args, "model-path"), model)
		if modelPath == "" {
			return ""
		}
		port := firstNonEmpty(opt(args, "port"), "30000")
		cmd := []string{shellQuote(firstNonEmpty(opt(args, "python-bin"), "python3")), "-m", "sglang.launch_server", "--model-path", shellQuote(modelPath), "--host", shellQuote(host), "--port", shellQuote(port)}
		appendShellArg(&cmd, "--served-model-name", firstNonEmpty(opt(args, "served-model"), opt(args, "model-name")))
		appendShellArg(&cmd, "--tp", opt(args, "tensor-parallel"))
		appendShellArg(&cmd, "--context-length", opt(args, "context-length"))
		appendExtraArgs(&cmd, opt(args, "extra-server-args"))
		return strings.Join(cmd, " ")
	case "ollama":
		return strings.Join([]string{shellQuote(firstNonEmpty(opt(args, "server-bin"), "ollama")), "serve"}, " ")
	default:
		return ""
	}
}

func appendExtraArgs(cmd *[]string, value string) {
	if value != "" {
		*cmd = append(*cmd, value)
	}
}

func runLongCommand(commandSnippet string) error {
	var cmd *exec.Cmd
	if runtime.GOOS == "windows" {
		cmd = exec.Command("cmd", "/C", commandSnippet)
	} else {
		cmd = exec.Command("sh", "-c", commandSnippet)
	}
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return cliError{"server_command_failed", "Server command failed.", []string{"Check that the server executable is installed and available on PATH.", "Run lmx server dry-run <engine> with the same options to inspect the command."}, err.Error()}
	}
	return nil
}

func detectHardware() map[string]any {
	base := map[string]any{"hwClass": "CPU_ONLY", "cpu": detectCPU(), "os": runtime.GOOS, "cpuThreads": runtime.NumCPU()}
	if runtime.GOOS == "darwin" && runtime.GOARCH == "arm64" {
		base["hwClass"] = "UNIFIED"
		base["chipVendor"] = "Apple"
		applyAppleHardware(base)
	}
	if out, err := exec.Command("nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits").Output(); err == nil {
		applyNvidiaSMIHardware(base, string(out))
	} else if out, err := exec.Command("rocm-smi", "--showproductname", "--showmeminfo", "vram", "--csv").Output(); err == nil {
		applyRocmSMIHardware(base, string(out))
	}
	return base
}

func detectCPU() string {
	if runtime.GOOS == "linux" {
		if data, err := os.ReadFile("/proc/cpuinfo"); err == nil {
			for _, line := range strings.Split(string(data), "\n") {
				if strings.HasPrefix(line, "model name") {
					parts := strings.SplitN(line, ":", 2)
					if len(parts) == 2 {
						return strings.TrimSpace(parts[1])
					}
				}
			}
		}
	}
	if runtime.GOOS == "darwin" {
		if out, err := exec.Command("sysctl", "-n", "machdep.cpu.brand_string").Output(); err == nil {
			return strings.TrimSpace(string(out))
		}
	}
	return runtime.GOARCH
}

func applyAppleHardware(base map[string]any) {
	out, err := exec.Command("system_profiler", "SPHardwareDataType").Output()
	if err != nil {
		return
	}
	text := string(out)
	if match := regexp.MustCompile(`(?m)Chip:\s*(.+)$`).FindStringSubmatch(text); len(match) > 1 {
		chip := strings.TrimSpace(match[1])
		base["cpu"] = chip
		parts := strings.Fields(strings.TrimPrefix(chip, "Apple "))
		if len(parts) > 0 {
			base["chipFamily"] = parts[0]
			if len(parts) > 1 {
				base["chipVariant"] = strings.Join(parts[1:], " ")
			} else {
				base["chipVariant"] = "base"
			}
		}
	}
	if match := regexp.MustCompile(`(?m)Memory:\s*(\d+)\s*GB`).FindStringSubmatch(text); len(match) > 1 {
		if gb, err := strconv.Atoi(match[1]); err == nil {
			base["unifiedMemoryGb"] = gb
		}
	}
	base["os"] = "darwin"
}

func applyNvidiaSMIHardware(base map[string]any, output string) {
	gpus := []map[string]any{}
	totalVramMb := 0.0
	for _, line := range strings.Split(strings.TrimSpace(output), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		parts := strings.Split(line, ",")
		if len(parts) < 2 {
			continue
		}
		gpu := map[string]any{"name": strings.TrimSpace(parts[0])}
		if mb, err := strconv.ParseFloat(strings.TrimSpace(parts[1]), 64); err == nil {
			gpu["vramGb"] = round1(mb / 1024)
			totalVramMb += mb
		}
		gpus = append(gpus, gpu)
	}
	if len(gpus) == 0 {
		return
	}
	base["hwClass"] = "DISCRETE_GPU"
	base["gpuCount"] = len(gpus)
	base["gpus"] = gpus
	base["gpuName"] = gpus[0]["name"]
	if vramGb, ok := gpus[0]["vramGb"]; ok {
		base["vramGb"] = vramGb
	}
	if totalVramMb > 0 {
		base["totalVramGb"] = round1(totalVramMb / 1024)
	}
}

func applyRocmSMIHardware(base map[string]any, output string) {
	name := ""
	if match := regexp.MustCompile(`(?i)Card series:\s*(.+)`).FindStringSubmatch(output); len(match) > 1 {
		name = strings.TrimSpace(match[1])
	}
	if name == "" {
		for _, line := range strings.Split(output, "\n") {
			if strings.Contains(strings.ToLower(line), "card") && strings.Contains(line, ",") {
				parts := strings.Split(line, ",")
				name = strings.Trim(strings.TrimSpace(parts[len(parts)-1]), `"`)
				break
			}
		}
	}
	vramGb := 0.0
	for _, match := range regexp.MustCompile(`(?i)(\d+(?:\.\d+)?)\s*MB.*VRAM`).FindAllStringSubmatch(output, -1) {
		if len(match) > 1 {
			mb, _ := strconv.ParseFloat(match[1], 64)
			if mb > vramGb*1024 {
				vramGb = mb / 1024
			}
		}
	}
	if name == "" && vramGb == 0 {
		return
	}
	base["hwClass"] = "DISCRETE_GPU"
	if name != "" {
		base["gpuName"] = name
	}
	if vramGb > 0 {
		base["vramGb"] = round1(vramGb)
	}
	base["gpuCount"] = 1
}

func round1(n float64) float64 { return float64(int(n*10+0.5)) / 10 }

func handleContext(args cliArgs) error {
	value, err := fetchJSON("GET", apiURL(args)+"/api/agent-context", "", nil)
	if err != nil {
		return err
	}
	return writeOrPrintJSON("context", args, value)
}

func handleModel(action, target string, args cliArgs) error {
	if action != "search" {
		return errors.New("Unknown model command. Use search.")
	}
	query := target
	if query == "" {
		query = firstNonEmpty(opt(args, "q"), opt(args, "query"))
	}
	if query == "" {
		return cliError{"missing_query", "model search requires a query", []string{"Run lmx model search qwen3-8b."}, nil}
	}
	limit, err := strconv.Atoi(firstNonEmpty(opt(args, "limit"), "10"))
	if err != nil || limit <= 0 {
		return cliError{"invalid_option", "--limit must be a positive integer", []string{"Pass --limit <number>."}, nil}
	}
	value, err := searchModels(args, query, limit)
	if err != nil {
		return err
	}
	return writeOrPrintJSON("models", args, value)
}

func handleSuite(action, target string, args cliArgs) error {
	switch action {
	case "list":
		value, err := fetchJSON("GET", apiURL(args)+"/api/evals/suites", "", nil)
		if err != nil {
			return err
		}
		return writeOrPrintJSON("suites", args, redactGold(value))
	case "show", "get":
		if target == "" {
			return errors.New("eval suite show requires a suite slug")
		}
		value, err := fetchJSON("GET", apiURL(args)+"/api/evals/suites/"+url.PathEscape(target), "", nil)
		if err != nil {
			return err
		}
		return writeOrPrintJSON("suite", args, redactGold(value))
	case "search":
		return handleSuiteSearch(target, args)
	case "validate":
		if target == "" {
			return errors.New("eval suite validate requires a suite JSON path")
		}
		payload, err := readJSON(target)
		if err != nil {
			return err
		}
		if err := validateSuite(payload); err != nil {
			return err
		}
		obj := asObject(payload)
		suiteDoc := asObject(obj["suiteDoc"])
		printInfo("suite_valid", map[string]any{"slug": obj["slug"], "runner": obj["runner"], "scoringMethod": suiteDoc["scoringMethod"], "submit": "lmx eval suite submit " + target + " --api-key bhk_..."})
		return nil
	case "submit":
		if target == "" {
			return errors.New("eval suite submit requires a suite JSON path")
		}
		key := apiKey(args)
		if key == "" {
			return missingAPIKey("--api-key or LMX_API_KEY is required")
		}
		payload, err := readJSON(target)
		if err != nil {
			return err
		}
		if err := validateSuite(payload); err != nil {
			return err
		}
		value, err := fetchJSON("POST", apiURL(args)+"/api/evals/suites", key, payload)
		if err != nil {
			return err
		}
		printJSON(value)
		printInfo("suite_submitted", map[string]any{"slug": asObject(payload)["slug"], "status": "PENDING", "next": "Wait for admin approval before running public submissions."})
		return nil
	case "init":
		return handleSuiteInit(args)
	default:
		return errors.New("Unknown suite command. Use init, list, search, show, validate, or submit.")
	}
}

func handleSuiteSearch(target string, args cliArgs) error {
	query := firstNonEmpty(target, opt(args, "q"), opt(args, "query"))
	if query == "" {
		return cliError{"missing_query", "eval suite search requires a query", []string{"Run lmx eval suite search reasoning."}, nil}
	}
	raw, err := fetchJSON("GET", apiURL(args)+"/api/evals/suites", "", nil)
	if err != nil {
		return err
	}
	limit := 20
	if n, err := strconv.Atoi(firstNonEmpty(opt(args, "limit"), "20")); err == nil {
		limit = n
	}
	var suites []any
	if obj := asObject(raw); obj != nil {
		if arr, ok := obj["suites"].([]any); ok {
			suites = arr
		}
	} else if arr, ok := raw.([]any); ok {
		suites = arr
	}
	queryLower := strings.ToLower(query)
	runner := strings.ToUpper(opt(args, "runner"))
	category := strings.ToLower(opt(args, "category"))
	matched := []any{}
	for _, suite := range suites {
		obj := asObject(suite)
		if obj == nil || !strings.Contains(strings.ToLower(toJSON(obj)), queryLower) {
			continue
		}
		if runner != "" && strings.ToUpper(fmt.Sprint(obj["runner"])) != runner {
			continue
		}
		if category != "" && strings.ToLower(fmt.Sprint(obj["category"])) != category {
			continue
		}
		matched = append(matched, suite)
		if len(matched) >= limit {
			break
		}
	}
	return writeOrPrintJSON("suites", args, redactGold(map[string]any{"suites": matched, "total": len(matched), "query": query}))
}

func handleSuiteInit(args cliArgs) error {
	slug, err := requireOpt(args, "slug")
	if err != nil {
		return err
	}
	name := firstNonEmpty(opt(args, "name"), slug)
	category := firstNonEmpty(opt(args, "category"), "general")
	runner := strings.ToUpper(firstNonEmpty(opt(args, "runner"), "CUSTOM"))
	if strings.EqualFold(runner, "lm-eval-harness") {
		runner = "LM_EVAL_HARNESS"
	}
	scoring := firstNonEmpty(opt(args, "scoring-method"), "exact_match")
	payload := buildSuiteTemplate(slug, name, category, runner, scoring, args)
	if err := validateSuite(payload); err != nil {
		return err
	}
	out := firstNonEmpty(opt(args, "out"), slug+".eval-suite.json")
	if err := writeJSON(out, payload); err != nil {
		return err
	}
	printInfo("suite_template_written", map[string]any{"path": out, "slug": slug, "runner": runner, "scoringMethod": scoring, "tasks": 1})
	fmt.Println("Edit the suite, then run:")
	fmt.Println("  lmx eval suite validate " + out)
	fmt.Println("  lmx eval suite submit " + out + " --api-key bhk_...")
	return nil
}

func buildSuiteTemplate(slug, name, category, runner, scoring string, args cliArgs) map[string]any {
	base := map[string]any{"slug": slug, "name": name, "description": opt(args, "description"), "category": category, "runner": runner, "version": "1.0"}
	if runner == "LM_EVAL_HARNESS" {
		tasks := parseCSVList(firstNonEmpty(opt(args, "tasks"), opt(args, "task"), slug))
		items := []any{}
		for _, task := range tasks {
			items = append(items, map[string]any{"key": task, "displayName": strings.ReplaceAll(task, "_", " "), "taskType": "multiple_choice", "weight": 1, "higherIsBetter": true})
		}
		base["description"] = firstNonEmpty(opt(args, "description"), "LM-Eval Harness suite. Run with lm-eval-harness, then upload the output JSON with the LocalMaxxing CLI.")
		base["suiteDoc"] = map[string]any{"version": "1.0", "runner": "lm-eval-harness", "scoringMethod": scoring, "higherIsBetter": true, "aggregation": "weighted_mean", "tasks": items}
		return base
	}
	kind := firstNonEmpty(opt(args, "kind"), "multiple_choice")
	base["description"] = firstNonEmpty(opt(args, "description"), "Custom "+strings.ReplaceAll(kind, "_", " ")+" eval suite. Replace the sample items before submitting.")
	switch kind {
	case "qa":
		base["suiteDoc"] = map[string]any{"version": "1.0", "runner": "custom", "scoringMethod": "exact_match", "higherIsBetter": true, "aggregation": "weighted_mean", "runConfig": map[string]any{"temperature": 0}, "tasks": []any{map[string]any{"key": "qa", "displayName": "Short-answer QA", "taskType": "qa", "weight": 1, "promptTemplate": "Answer the question with only the final answer.\n\nQuestion: {{input}}", "maxNewTokens": 64, "dataset": map[string]any{"source": "inline", "items": []any{map[string]any{"input": "What is 2 + 2?", "gold": "4"}}}}}}
	case "judge":
		base["suiteDoc"] = map[string]any{"version": "1.0", "runner": "custom", "scoringMethod": "llm_judge", "higherIsBetter": true, "aggregation": "weighted_mean", "runConfig": map[string]any{"temperature": 0.7}, "tasks": []any{map[string]any{"key": "judge_quality", "displayName": "Judge-scored response quality", "taskType": "judge", "weight": 1, "promptTemplate": "Write a concise answer to the following prompt.\n\n{{input}}", "maxNewTokens": 512, "dataset": map[string]any{"source": "inline", "items": []any{map[string]any{"input": "Explain why local inference benchmarks should include both speed and quality metrics.", "referenceAnswer": "A strong answer mentions that speed alone can hide regressions in reasoning, instruction following, or output quality.", "rubric": "Score 0 to 1. Reward clear explanation, mention of speed/quality tradeoffs, and relevance to local inference benchmarking."}}}}}}
	default:
		base["suiteDoc"] = map[string]any{"version": "1.0", "runner": "custom", "scoringMethod": "exact_match", "higherIsBetter": true, "aggregation": "weighted_mean", "runConfig": map[string]any{"temperature": 0}, "tasks": []any{map[string]any{"key": "multiple_choice", "displayName": "Multiple choice questions", "taskType": "multiple_choice", "weight": 1, "promptTemplate": "Choose the correct answer. Reply with only A, B, C, or D.\n\n{{input}}\n\n{{choices}}", "maxNewTokens": 8, "dataset": map[string]any{"source": "inline", "items": []any{map[string]any{"input": "Which number is even?", "choices": []any{"3", "5", "8", "9"}, "gold": "C"}}}}}}
	}
	return base
}

func mapRunnerDoc(runner string) string {
	if runner == "LM_EVAL_HARNESS" {
		return "lm-eval-harness"
	}
	return "custom"
}

func validateSuite(value any) error {
	obj := asObject(value)
	if obj == nil {
		return cliError{"invalid_suite", "Suite payload must be a JSON object.", nil, nil}
	}
	errs := []string{}
	for _, key := range []string{"slug", "name", "runner", "suiteDoc"} {
		if obj[key] == nil || fmt.Sprint(obj[key]) == "" {
			errs = append(errs, key+" is required")
		}
	}
	slug := stringValue(obj["slug"])
	if !regexp.MustCompile(`^[a-z0-9]+(?:-[a-z0-9]+)*$`).MatchString(slug) || len(slug) < 3 || len(slug) > 64 {
		errs = append(errs, "slug must be 3-64 lowercase alphanumeric characters with hyphens")
	}
	if name := stringValue(obj["name"]); name == "" || len(name) > 256 {
		errs = append(errs, "name is required and must be <= 256 chars")
	}
	if category := stringValue(obj["category"]); category == "" || len(category) > 64 {
		errs = append(errs, "category is required and must be <= 64 chars")
	}
	runner := stringValue(obj["runner"])
	if runner != "CUSTOM" && runner != "LM_EVAL_HARNESS" {
		errs = append(errs, "runner must be CUSTOM or LM_EVAL_HARNESS")
	}
	doc := asObject(obj["suiteDoc"])
	if doc == nil {
		errs = append(errs, "suiteDoc must be an object")
	} else {
		expectedRunner := map[bool]string{true: "lm-eval-harness", false: "custom"}[runner == "LM_EVAL_HARNESS"]
		if stringValue(doc["runner"]) != expectedRunner {
			errs = append(errs, "suiteDoc.runner must be "+expectedRunner)
		}
		scoring := stringValue(doc["scoringMethod"])
		if !containsString([]string{"exact_match", "f1", "pass_at_k", "perplexity", "llm_judge"}, scoring) {
			errs = append(errs, "suiteDoc.scoringMethod is invalid")
		}
		if scoring == "perplexity" {
			errs = append(errs, "perplexity scoring is not supported yet")
		}
		tasks := evalTasks(doc)
		if len(tasks) == 0 {
			errs = append(errs, "suiteDoc.tasks must contain at least one task")
		}
		if len(tasks) > 100 {
			errs = append(errs, "suiteDoc.tasks cannot exceed 100 tasks")
		}
		keys := map[string]bool{}
		for i, task := range tasks {
			prefix := fmt.Sprintf("suiteDoc.tasks[%d]", i)
			key := stringValue(task["key"])
			if key == "" {
				errs = append(errs, prefix+".key is required")
			}
			if keys[key] {
				errs = append(errs, prefix+".key duplicates \""+key+"\"")
			}
			keys[key] = true
			if stringValue(task["displayName"]) == "" {
				errs = append(errs, prefix+".displayName is required")
			}
			if runner == "CUSTOM" {
				if stringValue(task["promptTemplate"]) == "" {
					errs = append(errs, prefix+".promptTemplate is required for CUSTOM suites")
				}
				dataset := asObject(task["dataset"])
				if dataset == nil {
					errs = append(errs, prefix+".dataset is required for CUSTOM suites")
				} else if stringValue(dataset["source"]) == "inline" {
					items := anySlice(dataset["items"])
					if len(items) == 0 {
						errs = append(errs, prefix+".dataset.items is required for inline datasets")
					}
					for itemIndex, rawItem := range items {
						item := asObject(rawItem)
						if item == nil {
							continue
						}
						itemPrefix := fmt.Sprintf("%s.dataset.items[%d]", prefix, itemIndex)
						if fmt.Sprint(item["input"]) == "" {
							errs = append(errs, itemPrefix+".input is required")
						}
						if item["gold"] == nil && item["rubric"] == nil {
							errs = append(errs, itemPrefix+" needs gold or rubric")
						}
						if stringValue(task["taskType"]) == "multiple_choice" && len(anySlice(item["choices"])) == 0 {
							errs = append(errs, itemPrefix+".choices is required for multiple_choice tasks")
						}
					}
				}
			}
		}
	}
	if len(errs) > 0 {
		return cliError{"suite_validation_failed", "Suite validation failed.", []string{"Edit the suite JSON file and rerun eval suite validate.", "For custom suites, every task needs promptTemplate and dataset.", "For inline multiple-choice datasets, every item needs choices and gold."}, errs}
	}
	return nil
}

func handleStorage(action, target string, args cliArgs, forcedKind string) error {
	switch action {
	case "upload":
		if target == "" {
			return errors.New("eval storage upload requires a file path")
		}
		key := apiKey(args)
		if key == "" {
			return missingAPIKey("--api-key or LMX_API_KEY is required for storage upload")
		}
		data, err := os.ReadFile(target)
		if err != nil {
			return err
		}
		info, err := os.Stat(target)
		if err != nil {
			return err
		}
		format := firstNonEmpty(opt(args, "format"), strings.TrimPrefix(filepath.Ext(target), "."), "other")
		kind := firstNonEmpty(forcedKind, opt(args, "kind"), "artifact")
		hash := sha256.Sum256(data)
		metadata := map[string]any{"kind": kind, "filename": firstNonEmpty(opt(args, "filename"), filepath.Base(target)), "contentType": firstNonEmpty(opt(args, "content-type"), defaultContentType(format)), "format": format, "byteSize": info.Size(), "sha256": hex.EncodeToString(hash[:])}
		if count := opt(args, "item-count"); count != "" {
			if n, err := strconv.Atoi(count); err == nil {
				metadata["itemCount"] = n
			}
		}
		upload, err := fetchJSON("POST", apiURL(args)+"/api/evals/storage/upload-url", key, metadata)
		if err != nil {
			return err
		}
		uploadObj := asObject(upload)
		uploadURL := firstString(uploadObj, "uploadUrl", "url")
		if uploadURL == "" {
			return cliError{"storage_upload_url_missing", "Storage upload-url response did not include uploadUrl", []string{"Check that the LocalMaxxing API supports /api/evals/storage/upload-url."}, upload}
		}
		putReq, _ := http.NewRequest("PUT", uploadURL, bytes.NewReader(data))
		hasContentType := false
		for key, value := range stringMap(uploadObj["headers"]) {
			putReq.Header.Set(key, value)
			if strings.EqualFold(key, "Content-Type") {
				hasContentType = true
			}
		}
		if !hasContentType {
			putReq.Header.Set("Content-Type", fmt.Sprint(metadata["contentType"]))
		}
		putRes, err := http.DefaultClient.Do(putReq)
		if err != nil {
			return err
		}
		putBody, _ := io.ReadAll(putRes.Body)
		putRes.Body.Close()
		if putRes.StatusCode < 200 || putRes.StatusCode >= 300 {
			return cliError{"storage_put_failed", fmt.Sprintf("Storage PUT failed: %s", putRes.Status), []string{"Retry the upload; signed upload URLs can expire.", "Check --content-type and file size."}, string(putBody)}
		}
		storageRef := firstString(uploadObj, "storageRef", "key")
		if storageRef == "" {
			return cliError{"storage_ref_missing", "Storage upload-url response did not include storageRef or key", nil, upload}
		}
		completed, err := fetchJSON("POST", apiURL(args)+"/api/evals/storage/complete", key, map[string]any{"storageRef": storageRef})
		if err != nil {
			return err
		}
		return writeOrPrintJSON("storage_upload", args, map[string]any{"metadata": metadata, "storageRef": storageRef, "completed": completed})
	case "download":
		if target == "" {
			return errors.New("eval storage download requires a storage key")
		}
		out := opt(args, "out")
		if out == "" {
			return cliError{"missing_option", "--out is required for storage download", []string{"Pass --out <path> to write the downloaded object."}, nil}
		}
		signed, err := fetchJSON("GET", apiURL(args)+"/api/evals/storage/download-url?key="+url.QueryEscape(target), apiKey(args), nil)
		if err != nil {
			return err
		}
		downloadURL := firstString(asObject(signed), "downloadUrl", "url")
		if downloadURL == "" {
			return cliError{"storage_download_url_missing", "Storage download-url response did not include downloadUrl", nil, signed}
		}
		res, err := http.Get(downloadURL)
		if err != nil {
			return err
		}
		defer res.Body.Close()
		data, _ := io.ReadAll(res.Body)
		if res.StatusCode < 200 || res.StatusCode >= 300 {
			return cliError{"storage_download_failed", fmt.Sprintf("Storage download failed: %s", res.Status), []string{"Check the storage key and retry; signed download URLs can expire."}, string(data)}
		}
		if err := os.WriteFile(out, data, 0o644); err != nil {
			return err
		}
		printInfo("storage_downloaded", map[string]any{"path": out, "key": target})
		return nil
	default:
		return errors.New("Unknown storage command. Use upload or download.")
	}
}

func defaultContentType(format string) string {
	switch format {
	case "json":
		return "application/json"
	case "jsonl":
		return "application/jsonl"
	case "txt":
		return "text/plain"
	case "zip":
		return "application/zip"
	default:
		return "application/octet-stream"
	}
}

func handleBenchmark(action, target string, args cliArgs) error {
	if action == "validate-local" || action == "local-validate" {
		return validateBenchmarkFileLocally(target)
	}
	if action == "validate" {
		action = "dry-run"
	}
	if action == "runs" || action == "run-file" || action == "run-files" {
		return handleBenchmarkRuns(target, positional(args, 3), args)
	}
	if action == "list" || action == "show" || action == "edit" || action == "rerun" || action == "delete" || action == "rm" || action == "remove" {
		return handleBenchmarkRuns(action, target, args)
	}
	if action == "kvcache" || action == "kv-cache" || action == "context-sweep" {
		return handleKVCache("run", target, args)
	}
	if action == "run" || action == "measure" {
		payload, err := benchmarkPayloadFromFlags(target, args)
		if err != nil {
			return err
		}
		out := firstNonEmpty(opt(args, "out"), benchmarkRunPathInDir(payload, firstNonEmpty(opt(args, "runs-dir"), "runs")))
		feedback := benchmarkAgentFeedback(payload, out, args, hasFlag(args, "dry-run"), hasFlag(args, "submit"))
		payload["agentFeedback"] = feedback
		if err := os.MkdirAll(filepath.Dir(out), 0o755); err != nil {
			return err
		}
		if err := writeJSON(out, payload); err != nil {
			return err
		}
		printInfo("benchmark_payload_file_written", map[string]any{"path": out, "engine": payload["engineName"]})
		printStatus(args, "benchmark_payload_status", feedback)
		if hasFlag(args, "dry-run") && !hasFlag(args, "submit") {
			fmt.Println(stringValue(feedback["message"]))
			return nil
		}
		if hasFlag(args, "submit") {
			endpoint := "/api/benchmarks"
			if hasFlag(args, "dry-run") {
				endpoint = "/api/benchmarks/dry-run"
			}
			if err := validateBenchmarkSubmitPayload(payload); err != nil {
				return err
			}
			apiPayload := toBenchmarkSubmit(payload)
			return submitPayload(endpoint, hasFlag(args, "dry-run"), "benchmark", args, apiPayload)
		}
		printBenchmarkNextSteps(feedback, out)
		return nil
	}
	if action != "submit" && action != "dry-run" {
		return errors.New("Unknown benchmark command. Use run, runs, list, show, edit, rerun, submit, dry-run, validate-local, delete, stats, export, or compare.")
	}
	if target == "" {
		return fmt.Errorf("benchmark %s requires a benchmark JSON path", action)
	}
	value, err := readJSON(target)
	if err != nil {
		return err
	}
	if obj := asObject(value); obj != nil {
		if payload, ok := obj["payload"]; ok {
			value = payload
		}
	}
	payload := asObject(value)
	if err := validateBenchmarkSubmitPayload(payload); err != nil {
		return err
	}
	endpoint := "/api/benchmarks"
	if action == "dry-run" {
		endpoint = "/api/benchmarks/dry-run"
	}
	apiPayload := toBenchmarkSubmit(payload)
	return submitPayload(endpoint, action == "dry-run", "benchmark", args, apiPayload)
}

func validateBenchmarkFileLocally(path string) error {
	if path == "" {
		return errors.New("benchmark validate-local requires a benchmark JSON path")
	}
	value, err := readJSON(path)
	if err != nil {
		return err
	}
	if obj := asObject(value); obj != nil {
		if payload, ok := obj["payload"]; ok {
			value = payload
		}
	}
	if err := validateBenchmarkSubmitPayload(asObject(value)); err != nil {
		return err
	}
	printInfo("benchmark_local_valid", map[string]any{"path": path, "status": "valid", "note": "Local validation only; API dry-run still requires --api-key or LMX_API_KEY."})
	return nil
}

func handleBenchmarkRuns(action, target string, args cliArgs) error {
	if action == "" {
		action = "list"
	}
	switch action {
	case "list", "ls":
		return listBenchmarkRuns(args)
	case "show", "cat":
		path, err := resolveBenchmarkRunPath(target, args)
		if err != nil {
			return err
		}
		value, err := readJSON(path)
		if err != nil {
			return err
		}
		printJSON(value)
		return nil
	case "edit", "patch":
		path, err := resolveBenchmarkRunPath(target, args)
		if err != nil {
			return err
		}
		return editBenchmarkRun(path, args)
	case "rerun", "replay":
		path, err := resolveBenchmarkRunPath(target, args)
		if err != nil {
			return err
		}
		return rerunBenchmarkRun(path, args)
	case "submit", "dry-run", "validate":
		if action == "validate" {
			action = "dry-run"
		}
		path, err := resolveBenchmarkRunPath(target, args)
		if err != nil {
			return err
		}
		return handleBenchmark(action, path, args)
	case "delete", "rm", "remove":
		path, err := resolveBenchmarkRunPath(target, args)
		if err != nil {
			return err
		}
		return deleteBenchmarkRun(path, args)
	case "stats", "stat", "summary", "summarize":
		return statsBenchmarkRuns(target, args)
	case "export", "extract":
		return exportBenchmarkRuns(target, args)
	case "compare", "diff":
		return compareBenchmarkRuns(target, positional(args, 4), args)
	default:
		return errors.New("Unknown benchmark runs command. Use list, show, edit, rerun, submit, dry-run, delete, stats, export, or compare.")
	}
}

func listBenchmarkRuns(args cliArgs) error {
	root := firstNonEmpty(opt(args, "runs-dir"), "runs")
	limit := 50
	if value := opt(args, "limit"); value != "" {
		parsed, err := strconv.Atoi(value)
		if err != nil || parsed <= 0 {
			return cliError{"invalid_option", "--limit must be a positive integer", []string{"Pass --limit <n>."}, nil}
		}
		limit = parsed
	}
	runs, err := benchmarkRunSummaries(root)
	if err != nil {
		return err
	}
	if len(runs) > limit {
		runs = runs[:limit]
	}
	printJSON(map[string]any{"runsDir": root, "count": len(runs), "runs": runs})
	return nil
}

type benchmarkRunRecord struct {
	Path      string
	UpdatedAt time.Time
	Payload   map[string]any
}

func statsBenchmarkRuns(target string, args cliArgs) error {
	records, err := benchmarkRunRecords(target, args)
	if err != nil {
		return err
	}
	result := benchmarkRunStatsResult(records, args)
	return writeOrPrintJSON("benchmark_run_stats", args, result)
}

func exportBenchmarkRuns(target string, args cliArgs) error {
	records, err := benchmarkRunRecords(target, args)
	if err != nil {
		return err
	}
	fields := parseCSVList(firstNonEmpty(opt(args, "fields"), "path,updatedAt,kind,hfId,modelRevision,engineName,benchmarkMode,quantization,hardware,tokSOut,tokSPrefill,tokSTotal,ttftMs,peakVramGb,promptTokens,outputTokens,contextLength,contextTokens,batchSize,metricSource,timingSource,notes"))
	rows := benchmarkRunExportRows(records, fields)
	format := strings.ToLower(firstNonEmpty(opt(args, "format"), "json"))
	if format == "csv" {
		text, err := benchmarkRunRowsCSV(fields, rows)
		if err != nil {
			return err
		}
		if out := opt(args, "out"); out != "" {
			if err := os.WriteFile(out, []byte(text), 0o644); err != nil {
				return err
			}
			printInfo("benchmark_run_export_written", map[string]any{"path": out, "format": "csv", "rows": len(rows)})
			return nil
		}
		fmt.Print(text)
		return nil
	}
	if format != "json" {
		return cliError{"invalid_option", "--format must be json or csv", []string{"Use --format json or --format csv."}, nil}
	}
	return writeOrPrintJSON("benchmark_run_export", args, map[string]any{"count": len(rows), "fields": fields, "runs": rows})
}

func compareBenchmarkRuns(target, other string, args cliArgs) error {
	if target != "" && other != "" {
		left, err := benchmarkRunRecordFromPath(target)
		if err != nil {
			return err
		}
		right, err := benchmarkRunRecordFromPath(other)
		if err != nil {
			return err
		}
		result := compareTwoBenchmarkRuns(left, right, args)
		return writeOrPrintJSON("benchmark_run_compare", args, result)
	}
	records, err := benchmarkRunRecords(target, args)
	if err != nil {
		return err
	}
	result := compareBenchmarkRunGroups(records, args)
	return writeOrPrintJSON("benchmark_run_compare", args, result)
}

func benchmarkRunRecords(target string, args cliArgs) ([]benchmarkRunRecord, error) {
	paths, strict, err := benchmarkRunRecordPaths(target, firstNonEmpty(opt(args, "runs-dir"), "runs"))
	if err != nil {
		return nil, err
	}
	records := []benchmarkRunRecord{}
	for _, path := range paths {
		record, err := benchmarkRunRecordFromPath(path)
		if err != nil {
			if strict {
				return nil, err
			}
			continue
		}
		if !benchmarkRunRecordMatches(record, args) {
			continue
		}
		records = append(records, record)
	}
	sort.Slice(records, func(i, j int) bool { return records[i].UpdatedAt.After(records[j].UpdatedAt) })
	return records, nil
}

func benchmarkRunRecordPaths(target, runsDir string) ([]string, bool, error) {
	root := firstNonEmpty(target, runsDir)
	info, err := os.Stat(root)
	if err != nil {
		if os.IsNotExist(err) && target == "" {
			return []string{}, false, nil
		}
		return nil, target != "", err
	}
	if !info.IsDir() {
		return []string{root}, true, nil
	}
	paths := []string{}
	if err := filepath.WalkDir(root, func(path string, entry os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if entry.IsDir() || strings.ToLower(filepath.Ext(path)) != ".json" {
			return nil
		}
		paths = append(paths, path)
		return nil
	}); err != nil {
		return nil, false, err
	}
	return paths, false, nil
}

func benchmarkRunRecordFromPath(path string) (benchmarkRunRecord, error) {
	value, err := readJSON(path)
	if err != nil {
		return benchmarkRunRecord{}, err
	}
	payload := benchmarkPayloadObject(value)
	if payload == nil {
		return benchmarkRunRecord{}, cliError{"invalid_benchmark_run", "Saved benchmark run must be a JSON object or { payload: object }.", nil, value}
	}
	info, err := os.Stat(path)
	if err != nil {
		return benchmarkRunRecord{}, err
	}
	return benchmarkRunRecord{Path: path, UpdatedAt: info.ModTime(), Payload: payload}, nil
}

func benchmarkRunRecordMatches(record benchmarkRunRecord, args cliArgs) bool {
	payload := record.Payload
	if model := firstNonEmpty(opt(args, "hf-id"), opt(args, "model")); model != "" && !strings.EqualFold(stringValue(payload["hfId"]), model) {
		return false
	}
	if engine := opt(args, "engine"); engine != "" && !strings.EqualFold(stringValue(payload["engineName"]), normalizeEngineName(engine)) {
		return false
	}
	if mode := opt(args, "mode"); mode != "" && !strings.EqualFold(stringValue(payload["benchmarkMode"]), mode) {
		return false
	}
	if quantization := opt(args, "quantization"); quantization != "" && !strings.EqualFold(stringValue(payload["quantization"]), quantization) {
		return false
	}
	if kind := opt(args, "kind"); kind != "" && !strings.EqualFold(stringValue(payload["kind"]), kind) {
		return false
	}
	if hardware := opt(args, "hardware-name"); hardware != "" && !strings.Contains(strings.ToLower(hardwareLabel(asObject(payload["hardware"]))), strings.ToLower(hardware)) {
		return false
	}
	return true
}

func benchmarkRunStatsResult(records []benchmarkRunRecord, args cliArgs) map[string]any {
	metric := firstNonEmpty(opt(args, "metric"), "tokSOut")
	groupBy := firstNonEmpty(opt(args, "group-by"), opt(args, "by"), "all")
	groups := map[string][]benchmarkRunRecord{}
	for _, record := range records {
		key := benchmarkRunGroupKey(record, groupBy)
		groups[key] = append(groups[key], record)
	}
	stats := []any{}
	for key, group := range groups {
		stats = append(stats, benchmarkRunGroupStats(key, group, metric))
	}
	sort.Slice(stats, func(i, j int) bool {
		left := stats[i].(map[string]any)
		right := stats[j].(map[string]any)
		leftBest := numberField(left, "best")
		rightBest := numberField(right, "best")
		if lowerIsBetterMetric(metric) {
			return leftBest < rightBest
		}
		return leftBest > rightBest
	})
	return map[string]any{"count": len(records), "metric": metric, "groupBy": groupBy, "groups": stats}
}

func benchmarkRunGroupStats(key string, records []benchmarkRunRecord, metric string) map[string]any {
	values := []float64{}
	var best benchmarkRunRecord
	bestSet := false
	for _, record := range records {
		value := numberField(record.Payload, metric)
		if value <= 0 {
			continue
		}
		values = append(values, value)
		if !bestSet || metricBetter(value, numberField(best.Payload, metric), metric) {
			best = record
			bestSet = true
		}
	}
	sort.Float64s(values)
	result := map[string]any{"key": key, "runs": len(records), "metricRuns": len(values)}
	if len(values) == 0 {
		return result
	}
	sum := 0.0
	for _, value := range values {
		sum += value
	}
	median := values[len(values)/2]
	if len(values)%2 == 0 {
		median = (values[len(values)/2-1] + values[len(values)/2]) / 2
	}
	result["min"] = roundMetric(values[0])
	result["p50"] = roundMetric(median)
	result["mean"] = roundMetric(sum / float64(len(values)))
	result["max"] = roundMetric(values[len(values)-1])
	if bestSet {
		bestValue := numberField(best.Payload, metric)
		result["best"] = roundMetric(bestValue)
		result["bestRun"] = benchmarkRunIdentity(best)
	}
	return result
}

func compareBenchmarkRunGroups(records []benchmarkRunRecord, args cliArgs) map[string]any {
	if args.opts == nil {
		args.opts = map[string]string{}
	}
	if _, ok := args.opts["by"]; !ok {
		if _, ok := args.opts["group-by"]; !ok {
			args.opts["by"] = "quantization"
		}
	}
	stats := benchmarkRunStatsResult(records, args)
	groups := anySlice(stats["groups"])
	if len(groups) == 0 {
		stats["comparisons"] = []any{}
		return stats
	}
	baseline := asObject(groups[0])
	if baselineName := opt(args, "baseline"); baselineName != "" {
		for _, item := range groups {
			group := asObject(item)
			if strings.EqualFold(stringValue(group["key"]), baselineName) {
				baseline = group
				break
			}
		}
	}
	metric := stringValue(stats["metric"])
	baseValue := numberField(baseline, "best")
	comparisons := []any{}
	for _, item := range groups {
		group := asObject(item)
		value := numberField(group, "best")
		comparisons = append(comparisons, metricComparison(stringValue(group["key"]), value, stringValue(baseline["key"]), baseValue, metric))
	}
	stats["baseline"] = baseline["key"]
	stats["comparisons"] = comparisons
	return stats
}

func compareTwoBenchmarkRuns(left, right benchmarkRunRecord, args cliArgs) map[string]any {
	metrics := parseCSVList(firstNonEmpty(opt(args, "metrics"), opt(args, "metric"), "tokSOut,tokSPrefill,tokSTotal,ttftMs,peakVramGb"))
	comparisons := []any{}
	for _, metric := range metrics {
		leftValue := numberField(left.Payload, metric)
		rightValue := numberField(right.Payload, metric)
		if leftValue == 0 && rightValue == 0 {
			continue
		}
		comparisons = append(comparisons, metricComparison(metric, rightValue, "baseline", leftValue, metric))
	}
	return map[string]any{"baseline": benchmarkRunIdentity(left), "candidate": benchmarkRunIdentity(right), "comparisons": comparisons}
}

func metricComparison(name string, value float64, baselineName string, baselineValue float64, metric string) map[string]any {
	comparison := map[string]any{"name": name, "metric": metric, "value": roundMetric(value), "baseline": baselineName, "baselineValue": roundMetric(baselineValue)}
	if baselineValue > 0 && value > 0 {
		delta := value - baselineValue
		if lowerIsBetterMetric(metric) {
			delta = baselineValue - value
		}
		comparison["delta"] = roundMetric(delta)
		comparison["ratio"] = roundMetric(value / baselineValue)
		comparison["percent"] = roundMetric((delta / baselineValue) * 100)
		comparison["better"] = metricBetter(value, baselineValue, metric)
	}
	return comparison
}

func benchmarkRunExportRows(records []benchmarkRunRecord, fields []string) []map[string]any {
	rows := []map[string]any{}
	for _, record := range records {
		row := map[string]any{}
		for _, field := range fields {
			row[field] = benchmarkRunField(record, field)
		}
		rows = append(rows, row)
	}
	return rows
}

func benchmarkRunRowsCSV(fields []string, rows []map[string]any) (string, error) {
	var buf bytes.Buffer
	writer := csv.NewWriter(&buf)
	if err := writer.Write(fields); err != nil {
		return "", err
	}
	for _, row := range rows {
		record := make([]string, len(fields))
		for i, field := range fields {
			record[i] = csvValue(row[field])
		}
		if err := writer.Write(record); err != nil {
			return "", err
		}
	}
	writer.Flush()
	if err := writer.Error(); err != nil {
		return "", err
	}
	return buf.String(), nil
}

func benchmarkRunField(record benchmarkRunRecord, field string) any {
	field = strings.TrimSpace(field)
	switch field {
	case "path":
		return record.Path
	case "updatedAt":
		return record.UpdatedAt.UTC().Format(time.RFC3339)
	case "model":
		return record.Payload["hfId"]
	case "hardware":
		return hardwareLabel(asObject(record.Payload["hardware"]))
	default:
		return dottedValue(record.Payload, field)
	}
}

func benchmarkRunGroupKey(record benchmarkRunRecord, groupBy string) string {
	groupBy = strings.TrimSpace(groupBy)
	if groupBy == "" || groupBy == "all" || groupBy == "none" {
		return "all"
	}
	if groupBy == "quant" {
		groupBy = "quantization"
	}
	value := benchmarkRunField(record, groupBy)
	text := csvValue(value)
	if strings.TrimSpace(text) == "" {
		return "unknown"
	}
	return text
}

func benchmarkRunIdentity(record benchmarkRunRecord) map[string]any {
	return map[string]any{
		"path":         record.Path,
		"model":        record.Payload["hfId"],
		"engine":       record.Payload["engineName"],
		"mode":         record.Payload["benchmarkMode"],
		"quantization": record.Payload["quantization"],
		"hardware":     hardwareLabel(asObject(record.Payload["hardware"])),
		"tokSOut":      record.Payload["tokSOut"],
		"updatedAt":    record.UpdatedAt.UTC().Format(time.RFC3339),
	}
}

func hardwareLabel(hw map[string]any) string {
	if hw == nil {
		return "unknown"
	}
	if gpus := anySlice(hw["gpus"]); len(gpus) > 0 {
		parts := []string{}
		for _, item := range gpus {
			gpu := asObject(item)
			name := firstNonEmpty(stringValue(gpu["name"]), stringValue(gpu["gpuName"]), "GPU")
			if vram := numberField(gpu, "vramGb"); vram > 0 {
				name += " " + strconv.FormatFloat(vram, 'f', -1, 64) + "GB"
			}
			parts = append(parts, name)
		}
		return strings.Join(parts, " + ")
	}
	if gpu := stringValue(hw["gpuName"]); gpu != "" {
		if count := numberField(hw, "gpuCount"); count > 1 {
			gpu = strconv.FormatFloat(count, 'f', -1, 64) + "x " + gpu
		}
		if vram := numberField(hw, "vramGb"); vram > 0 {
			gpu += " " + strconv.FormatFloat(vram, 'f', -1, 64) + "GB"
		}
		return gpu
	}
	unified := strings.TrimSpace(strings.Join([]string{stringValue(hw["chipVendor"]), stringValue(hw["chipFamily"]), stringValue(hw["chipVariant"])}, " "))
	if unified != "" {
		if mem := numberField(hw, "unifiedMemoryGb"); mem > 0 {
			unified += " " + strconv.FormatFloat(mem, 'f', -1, 64) + "GB"
		}
		return unified
	}
	return firstNonEmpty(stringValue(hw["cpu"]), stringValue(hw["hwClass"]), "unknown")
}

func dottedValue(obj map[string]any, path string) any {
	if obj == nil || path == "" {
		return nil
	}
	parts := strings.Split(path, ".")
	var value any = obj
	for _, part := range parts {
		child := asObject(value)
		if child == nil {
			return nil
		}
		value = child[part]
	}
	return value
}

func parseCSVList(value string) []string {
	items := []string{}
	for _, item := range strings.FieldsFunc(value, func(r rune) bool { return r == ',' || r == ';' }) {
		item = strings.TrimSpace(item)
		if item != "" {
			items = append(items, item)
		}
	}
	return items
}

func csvValue(value any) string {
	switch v := value.(type) {
	case nil:
		return ""
	case string:
		return v
	case float64:
		return strconv.FormatFloat(v, 'f', -1, 64)
	case float32:
		return strconv.FormatFloat(float64(v), 'f', -1, 64)
	case int:
		return strconv.Itoa(v)
	case int64:
		return strconv.FormatInt(v, 10)
	case bool:
		return strconv.FormatBool(v)
	default:
		data, _ := json.Marshal(v)
		return string(data)
	}
}

func roundMetric(value float64) float64 {
	return float64(int(value*100+0.5)) / 100
}

func lowerIsBetterMetric(metric string) bool {
	metric = strings.ToLower(metric)
	return strings.Contains(metric, "latency") || strings.Contains(metric, "ttft") || strings.Contains(metric, "ms")
}

func metricBetter(value, baseline float64, metric string) bool {
	if baseline == 0 {
		return value > 0
	}
	if lowerIsBetterMetric(metric) {
		return value > 0 && value < baseline
	}
	return value > baseline
}

func benchmarkRunSummaries(root string) ([]any, error) {
	info, err := os.Stat(root)
	if err != nil {
		if os.IsNotExist(err) {
			return []any{}, nil
		}
		return nil, err
	}
	if !info.IsDir() {
		return nil, cliError{"invalid_runs_dir", root + " is not a directory", []string{"Pass --runs-dir <dir> if saved runs are elsewhere."}, nil}
	}
	paths := []string{}
	if err := filepath.WalkDir(root, func(path string, entry os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if entry.IsDir() || strings.ToLower(filepath.Ext(path)) != ".json" {
			return nil
		}
		paths = append(paths, path)
		return nil
	}); err != nil {
		return nil, err
	}
	sort.Slice(paths, func(i, j int) bool {
		left, leftErr := os.Stat(paths[i])
		right, rightErr := os.Stat(paths[j])
		if leftErr != nil || rightErr != nil {
			return paths[i] > paths[j]
		}
		return left.ModTime().After(right.ModTime())
	})
	runs := []any{}
	for _, path := range paths {
		value, err := readJSON(path)
		if err != nil {
			continue
		}
		payload := benchmarkPayloadObject(value)
		if payload == nil {
			continue
		}
		info, _ := os.Stat(path)
		runs = append(runs, map[string]any{
			"path":          path,
			"model":         payload["hfId"],
			"engine":        payload["engineName"],
			"mode":          payload["benchmarkMode"],
			"quantization":  payload["quantization"],
			"tokSOut":       payload["tokSOut"],
			"canSubmit":     numberField(payload, "tokSOut") > 0,
			"updatedAt":     info.ModTime().UTC().Format(time.RFC3339),
			"showCommand":   "lmx benchmark runs show " + shellQuote(path),
			"submitCommand": "lmx benchmark runs submit " + shellQuote(path),
			"rerunCommand":  "lmx benchmark runs rerun " + shellQuote(path),
			"editCommand":   "lmx benchmark runs edit " + shellQuote(path) + " --set-json '{\"notes\":\"...\"}'",
			"deleteCommand": "lmx benchmark runs delete " + shellQuote(path) + " --yes",
		})
	}
	return runs, nil
}

func resolveBenchmarkRunPath(target string, args cliArgs) (string, error) {
	path := firstNonEmpty(target, opt(args, "path"), opt(args, "run"))
	if path == "" {
		return "", cliError{"missing_run_path", "Saved benchmark run path is required.", []string{"Use lmx benchmark runs list, then pass a path to show/edit/rerun/submit."}, nil}
	}
	return path, nil
}

func editBenchmarkRun(path string, args cliArgs) error {
	value, err := readJSON(path)
	if err != nil {
		return err
	}
	payload := benchmarkPayloadObject(value)
	if payload == nil {
		return cliError{"invalid_benchmark_run", "Saved benchmark run must be a JSON object or { payload: object }.", nil, value}
	}
	changed := false
	if patchPath := opt(args, "patch"); patchPath != "" {
		patch, err := readJSON(patchPath)
		if err != nil {
			return err
		}
		patchObj := asObject(patch)
		if patchObj == nil {
			return cliError{"invalid_patch", "--patch must point to a JSON object.", nil, patch}
		}
		mergeObject(payload, patchObj)
		changed = true
	}
	if patchText := opt(args, "set-json"); patchText != "" {
		var patch any
		if err := json.Unmarshal([]byte(patchText), &patch); err != nil {
			return cliError{"json_parse_error", "--set-json must be a JSON object.", nil, err.Error()}
		}
		patchObj := asObject(patch)
		if patchObj == nil {
			return cliError{"invalid_patch", "--set-json must be a JSON object.", nil, patch}
		}
		mergeObject(payload, patchObj)
		changed = true
	}
	if set := opt(args, "set"); set != "" {
		field, raw, ok := strings.Cut(set, "=")
		if !ok || strings.TrimSpace(field) == "" {
			return cliError{"invalid_option", "--set must be field=value", []string{"Example: --set tokSOut=120.5", "For multiple fields, use --set-json '{\"tokSOut\":120.5,\"notes\":\"fixed\"}'."}, nil}
		}
		payload[strings.TrimSpace(field)] = parseEditValue(raw)
		changed = true
	}
	if unset := opt(args, "unset"); unset != "" {
		for _, field := range strings.Split(unset, ",") {
			field = strings.TrimSpace(field)
			if field != "" {
				delete(payload, field)
				changed = true
			}
		}
	}
	if !changed {
		return cliError{"missing_edit", "No edit was provided.", []string{"Use --set field=value, --set-json '{...}', --patch patch.json, or --unset field1,field2."}, nil}
	}
	payload["agentFeedback"] = benchmarkAgentFeedback(payload, path, args, hasFlag(args, "dry-run"), hasFlag(args, "submit"))
	if err := writeJSON(path, value); err != nil {
		return err
	}
	printInfo("benchmark_run_edited", map[string]any{"path": path})
	if hasFlag(args, "print") || hasFlag(args, "json") {
		printJSON(value)
	}
	return nil
}

func deleteBenchmarkRun(path string, args cliArgs) error {
	if !hasFlag(args, "yes") && !hasFlag(args, "force") {
		return cliError{"confirmation_required", "Deleting a saved benchmark run requires --yes.", []string{"Run lmx benchmark runs delete " + shellQuote(path) + " --yes to remove this file."}, nil}
	}
	info, err := os.Stat(path)
	if err != nil {
		return err
	}
	if info.IsDir() {
		return cliError{"invalid_run_path", "Saved benchmark run path must be a file, not a directory.", []string{"Use lmx benchmark runs list to choose a saved run JSON file."}, nil}
	}
	if err := os.Remove(path); err != nil {
		return err
	}
	printInfo("benchmark_run_deleted", map[string]any{"path": path})
	return nil
}

func rerunBenchmarkRun(path string, args cliArgs) error {
	value, err := readJSON(path)
	if err != nil {
		return err
	}
	payload := benchmarkPayloadObject(value)
	if payload == nil {
		return cliError{"invalid_benchmark_run", "Saved benchmark run must be a JSON object or { payload: object }.", nil, value}
	}
	runArgs := benchmarkArgsFromPayload(payload, args)
	return handleBenchmark("run", stringValue(payload["engineName"]), runArgs)
}

func benchmarkArgsFromPayload(payload map[string]any, overrides cliArgs) cliArgs {
	args := cliArgs{opts: map[string]string{}, flags: map[string]bool{}}
	for key, value := range map[string]any{
		"mode":           payload["benchmarkMode"],
		"hf-id":          payload["hfId"],
		"model-revision": payload["modelRevision"],
		"quantization":   payload["quantization"],
		"hardware":       overrides.opts["hardware"],
	} {
		if text := stringValue(value); text != "" {
			args.opts[key] = text
		}
	}
	if engineFlags := asObject(payload["engineFlags"]); engineFlags != nil {
		for key, field := range map[string]string{"baseUrl": "base-url", "servedModel": "served-model", "maxTokens": "max-tokens", "commandSnippet": "command"} {
			if text := stringValue(engineFlags[key]); text != "" {
				args.opts[field] = text
			}
		}
	}
	for key, value := range overrides.opts {
		if key == "path" || key == "run" || key == "set" || key == "set-json" || key == "patch" || key == "unset" {
			continue
		}
		args.opts[key] = value
	}
	for key, value := range overrides.flags {
		args.flags[key] = value
	}
	args.flags["quiet"] = hasFlag(overrides, "quiet")
	return args
}

func benchmarkPayloadObject(value any) map[string]any {
	obj := asObject(value)
	if obj == nil {
		return nil
	}
	if payload := asObject(obj["payload"]); payload != nil {
		return payload
	}
	return obj
}

func mergeObject(dst, src map[string]any) {
	for key, value := range src {
		if srcObj := asObject(value); srcObj != nil {
			if dstObj := asObject(dst[key]); dstObj != nil {
				mergeObject(dstObj, srcObj)
				continue
			}
		}
		dst[key] = value
	}
}

func parseEditValue(value string) any {
	value = strings.TrimSpace(value)
	if value == "true" {
		return true
	}
	if value == "false" {
		return false
	}
	if value == "null" {
		return nil
	}
	if n, err := strconv.ParseFloat(value, 64); err == nil {
		return n
	}
	var parsed any
	if strings.HasPrefix(value, "{") || strings.HasPrefix(value, "[") {
		if json.Unmarshal([]byte(value), &parsed) == nil {
			return parsed
		}
	}
	return value
}

func benchmarkRunPath(payload map[string]any) string {
	return benchmarkRunPathInDir(payload, "runs")
}

func benchmarkRunPathInDir(payload map[string]any, runsDir string) string {
	modelID := safePathSegment(stringValue(payload["hfId"]))
	if modelID == "" {
		modelID = "unknown-model"
	}
	timestamp := time.Now().UTC().Format("20060102T150405Z")
	return filepath.Join(runsDir, modelID, timestamp+".json")
}

func safePathSegment(value string) string {
	value = strings.TrimSpace(value)
	var b strings.Builder
	lastDash := false
	for _, r := range value {
		valid := r >= 'A' && r <= 'Z' || r >= 'a' && r <= 'z' || r >= '0' && r <= '9' || r == '.' || r == '_' || r == '-'
		if valid {
			b.WriteRune(r)
			lastDash = false
			continue
		}
		if !lastDash {
			b.WriteByte('-')
			lastDash = true
		}
	}
	return strings.Trim(b.String(), "-.")
}

func handleKVCache(action, target string, args cliArgs) error {
	if action == "" {
		action = "run"
	}
	if action != "run" && action != "measure" {
		return errors.New("kvcache requires run")
	}
	model := firstNonEmpty(opt(args, "hf-id"), opt(args, "model"))
	if model == "" {
		return cliError{"missing_model", "kvcache run requires --hf-id or --model", []string{"Pass --hf-id <HuggingFace model id>."}, nil}
	}
	levels, err := parseIntList(firstNonEmpty(opt(args, "levels"), opt(args, "context-levels"), opt(args, "contexts")))
	if err != nil {
		return err
	}
	if len(levels) == 0 {
		levels = []int{10000, 20000, 30000, 40000}
	}
	mode := benchmarkMode(args)
	engineName := normalizeEngineName(firstNonEmpty(opt(args, "engine"), target))
	if engineName == "" {
		resolved, err := resolveEngineName(target, args, mode == "local")
		if err != nil {
			return err
		}
		engineName = resolved
	}
	quantization := opt(args, "quantization")
	remoteWarnings := []string{}
	if mode == "remote" {
		remoteWarnings = append(remoteWarnings, remoteKVCacheFallbackWarning)
		printStatus(args, "kvcache_remote_depth_fallback", map[string]any{"warning": remoteKVCacheFallbackWarning, "fallback": "remote_depth_tps"})
	}

	var hardware any
	hardwareSource := ""
	if !hasFlag(args, "dry-run") || opt(args, "hardware") != "" {
		loaded, source, err := benchmarkHardware(mode, args)
		if err != nil {
			return err
		}
		hardware = loaded
		hardwareSource = source
	}

	runsDir := firstNonEmpty(opt(args, "runs-dir"), "runs")
	modelID := safePathSegment(model)
	baseDir := filepath.Join(runsDir, modelID)

	savedPaths := []string{}
	points := []any{}
	for _, level := range levels {
		printStatus(args, "kvcache_point_start", map[string]any{"level": level, "mode": mode})
		var point map[string]any
		if mode == "remote" {
			point, err = measureRemoteKVCachePoint(args, model, level)
		} else {
			point, err = measureLocalKVCachePoint(args, engineName, level)
		}
		if err != nil {
			return err
		}

		runPayload := map[string]any{
			"kind":          "kvcache_context_sweep",
			"benchmarkMode": mode,
			"engineName":    engineName,
			"hfId":          model,
			"modelRevision": firstNonEmpty(opt(args, "model-revision"), "main"),
			"quantization":  quantization,
			"contextTokens": float64(level),
			"outputTokens":  point["outputTokens"],
			"promptTokens":  point["promptTokens"],
			"tokSOut":       point["tokSOut"],
			"tokSPrefill":   point["tokSPrefill"],
			"tokSTotal":     point["tokSTotal"],
			"ttftMs":        point["ttftMs"],
			"peakVramGb":    point["peakVramGb"],
			"metricSource":  point["metricSource"],
			"timingSource":  point["timingSource"],
			"provenance": map[string]any{
				"benchmarkMode": mode,
				"cli":           "localmaxxing-go",
				"createdAt":     time.Now().UTC().Format(time.RFC3339),
				"metricSource":  point["metricSource"],
				"timingSource":  point["timingSource"],
				"ttftSource":    "estimated_from_prefill",
			},
		}
		if hardware != nil {
			runPayload["hardware"] = hardware
		}
		if hardwareSource != "" {
			runPayload["hardwareSource"] = hardwareSource
		}
		if commandSnippet := stringValue(point["commandSnippet"]); commandSnippet != "" {
			runPayload["engineFlags"] = map[string]any{"mode": mode, "commandSnippet": commandSnippet}
		}
		for _, key := range []string{"methodology", "warnings", "cacheReuse", "usagePromptTokens", "modelResolution", "quantizationResolution"} {
			if value, ok := point[key]; ok {
				runPayload[key] = value
			}
		}
		if len(remoteWarnings) > 0 {
			runPayload["warnings"] = mergeWarnings(remoteWarnings, runPayload["warnings"])
		}

		if hasFlag(args, "dry-run") {
			runPayload["dryRun"] = true
		}
		points = append(points, point)

		timestamp := time.Now().UTC().Format("20060102T150405Z")
		runPath := filepath.Join(baseDir, fmt.Sprintf("kvcache-%d-%s.json", level, timestamp))
		if err := os.MkdirAll(filepath.Dir(runPath), 0o755); err != nil {
			return err
		}
		if err := writeJSON(runPath, map[string]any{"payload": runPayload}); err != nil {
			return err
		}
		savedPaths = append(savedPaths, runPath)
		printStatus(args, "kvcache_point_complete", map[string]any{"level": level, "tokSOut": point["tokSOut"], "ttftMs": point["ttftMs"], "path": runPath})
	}

	aggregatePath := firstNonEmpty(opt(args, "out"), "localmaxxing-kvcache.json")
	aggregate := map[string]any{
		"kind":          "kvcache_context_sweep",
		"mode":          mode,
		"engineName":    engineName,
		"hfId":          model,
		"modelRevision": firstNonEmpty(opt(args, "model-revision"), "main"),
		"quantization":  quantization,
		"levels":        levels,
		"points":        points,
		"savedRuns":     savedPaths,
		"createdAt":     time.Now().UTC().Format(time.RFC3339),
	}
	if hardware != nil {
		aggregate["hardware"] = hardware
	}
	if hardwareSource != "" {
		aggregate["hardwareSource"] = hardwareSource
	}
	if len(remoteWarnings) > 0 {
		aggregate["warnings"] = remoteWarnings
	}
	if hasFlag(args, "dry-run") {
		aggregate["dryRun"] = true
	}
	if err := os.MkdirAll(filepath.Dir(aggregatePath), 0o755); err != nil {
		return err
	}
	if err := writeJSON(aggregatePath, aggregate); err != nil {
		return err
	}
	printStatus(args, "kvcache_sweep_written", map[string]any{"path": aggregatePath, "points": len(points)})

	if !hasFlag(args, "quiet") {
		fmt.Println("KV-cache sweep written:")
		fmt.Println("  " + aggregatePath)
		fmt.Println()
		fmt.Println("Saved runs:")
		for _, p := range savedPaths {
			fmt.Println("  " + p)
		}
		fmt.Println("\nSubmit individually with:")
		for _, p := range savedPaths {
			fmt.Println("  lmx benchmark submit " + shellQuote(p) + " --api-key <key>")
		}
		fmt.Println("\nOr dry-run validate with:")
		for _, p := range savedPaths {
			fmt.Println("  lmx benchmark dry-run " + shellQuote(p))
		}
	}
	return nil
}

func kvCachePayloadFromFlags(target string, args cliArgs) (map[string]any, error) {
	model := firstNonEmpty(opt(args, "hf-id"), opt(args, "model"))
	if model == "" {
		return nil, cliError{"missing_model", "kvcache run requires --hf-id or --model", []string{"Pass --hf-id <HuggingFace model id>."}, nil}
	}
	levels, err := parseIntList(firstNonEmpty(opt(args, "levels"), opt(args, "context-levels"), opt(args, "contexts")))
	if err != nil {
		return nil, err
	}
	if len(levels) == 0 {
		levels = []int{10000, 20000, 30000, 40000}
	}
	mode := benchmarkMode(args)
	engineName := normalizeEngineName(firstNonEmpty(opt(args, "engine"), target))
	if engineName == "" {
		resolved, err := resolveEngineName(target, args, mode == "local")
		if err != nil {
			return nil, err
		}
		engineName = resolved
	}
	payload := map[string]any{
		"kind":          "kvcache_context_sweep",
		"mode":          mode,
		"engineName":    engineName,
		"hfId":          model,
		"modelRevision": firstNonEmpty(opt(args, "model-revision"), "main"),
		"levels":        levels,
		"outputTokens":  kvOutputTokens(args),
		"createdAt":     time.Now().UTC().Format(time.RFC3339),
		"points":        []any{},
	}
	if mode == "remote" {
		payload["warnings"] = []string{remoteKVCacheFallbackWarning}
		printStatus(args, "kvcache_remote_depth_fallback", map[string]any{"warning": remoteKVCacheFallbackWarning, "fallback": "remote_depth_tps"})
	}
	if q := opt(args, "quantization"); q != "" {
		payload["quantization"] = q
	}
	if !hasFlag(args, "dry-run") || opt(args, "hardware") != "" {
		hardware, source, err := benchmarkHardware(mode, args)
		if err != nil {
			return nil, err
		}
		if hardware != nil {
			payload["hardware"] = hardware
		}
		if source != "" {
			payload["hardwareSource"] = source
		}
	}

	points := []any{}
	for _, level := range levels {
		printStatus(args, "kvcache_point_start", map[string]any{"level": level, "mode": mode})
		var point map[string]any
		if mode == "remote" {
			point, err = measureRemoteKVCachePoint(args, model, level)
		} else {
			point, err = measureLocalKVCachePoint(args, engineName, level)
		}
		if err != nil {
			return nil, err
		}
		points = append(points, point)
		printStatus(args, "kvcache_point_complete", map[string]any{"level": level, "tokSOut": point["tokSOut"], "ttftMs": point["ttftMs"]})
	}
	payload["points"] = points
	return payload, nil
}

func parseIntList(value string) ([]int, error) {
	if strings.TrimSpace(value) == "" {
		return nil, nil
	}
	parts := strings.FieldsFunc(value, func(r rune) bool { return r == ',' || r == ' ' || r == ';' })
	levels := []int{}
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		parsed, err := strconv.Atoi(part)
		if err != nil || parsed <= 0 {
			return nil, cliError{"invalid_option", "--levels must be a comma-separated list of positive integers", []string{"Example: --levels 10000,20000,30000,40000"}, nil}
		}
		levels = append(levels, parsed)
	}
	return levels, nil
}

func mergeWarnings(prefix []string, existing any) []string {
	warnings := []string{}
	seen := map[string]bool{}
	appendWarning := func(value string) {
		value = strings.TrimSpace(value)
		if value == "" || seen[value] {
			return
		}
		seen[value] = true
		warnings = append(warnings, value)
	}
	for _, warning := range prefix {
		appendWarning(warning)
	}
	switch typed := existing.(type) {
	case []string:
		for _, warning := range typed {
			appendWarning(warning)
		}
	case []any:
		for _, warning := range typed {
			appendWarning(stringValue(warning))
		}
	case string:
		appendWarning(typed)
	}
	return warnings
}

func kvOutputTokens(args cliArgs) int {
	value := firstNonEmpty(opt(args, "output-tokens"), opt(args, "output-len"), opt(args, "max-tokens"), "128")
	parsed, err := strconv.Atoi(value)
	if err != nil || parsed <= 0 {
		return 128
	}
	return parsed
}

func kvPromptTokens(args cliArgs) int {
	value := firstNonEmpty(opt(args, "prompt-tokens"), opt(args, "input-len"), "512")
	parsed, err := strconv.Atoi(value)
	if err != nil || parsed < 0 {
		return 512
	}
	return parsed
}

func measureLocalKVCachePoint(args cliArgs, engineName string, level int) (map[string]any, error) {
	commandSnippet := localKVCacheCommand(engineName, args, level)
	if commandSnippet == "" {
		return nil, cliError{"kvcache_command_unavailable", "Could not build a local KV-cache benchmark command.", []string{"Use llama.cpp with --model-path, vLLM/SGLang with --hf-id, or pass --command-template containing {input} and optionally {output}."}, nil}
	}
	point := map[string]any{"contextTokens": float64(level), "commandSnippet": commandSnippet, "mode": "local", "engineName": engineName}
	if hasFlag(args, "dry-run") {
		point["dryRun"] = true
		return point, nil
	}
	output, err := runBenchmarkCommand(commandSnippet)
	if err != nil {
		return nil, err
	}
	if outputPath := localKVCacheOutputPath(engineName, args, level); outputPath != "" {
		if data, err := os.ReadFile(outputPath); err == nil {
			output = strings.TrimSpace(output + "\n" + string(data))
			point["benchmarkOutput"] = outputPath
		}
	}
	parsed := parseBenchmarkOutput(output)
	for key, value := range parsed {
		point[key] = value
	}
	if point["promptTokens"] == nil {
		point["promptTokens"] = float64(kvPromptTokens(args))
	}
	if point["outputTokens"] == nil {
		point["outputTokens"] = float64(kvOutputTokens(args))
	}
	applyComparableBenchmarkMetrics(point, "local", engineName)
	return point, nil
}

func localKVCacheCommand(engineName string, args cliArgs, inputTokens int) string {
	outputTokens := strconv.Itoa(kvOutputTokens(args))
	input := strconv.Itoa(inputTokens)
	if template := opt(args, "command-template"); template != "" {
		replacer := strings.NewReplacer("{input}", input, "{prompt}", input, "{context}", input, "{output}", outputTokens, "{max_tokens}", outputTokens)
		return replacer.Replace(template)
	}
	if command := opt(args, "command"); command != "" {
		return command
	}
	switch engineName {
	case "llama.cpp":
		if opt(args, "model-path") == "" {
			return ""
		}
		cmd := []string{"llama-bench", "-m", shellQuote(opt(args, "model-path")), "-p", strconv.Itoa(kvPromptTokens(args)), "-n", outputTokens, "-d", input, "-o", firstNonEmpty(opt(args, "benchmark-format"), opt(args, "bench-format"), opt(args, "output-format"), "json")}
		if value := opt(args, "threads"); value != "" {
			cmd = append(cmd, "-t", shellQuote(value))
		}
		if value := opt(args, "gpu-layers"); value != "" {
			cmd = append(cmd, "-ngl", shellQuote(value))
		}
		appendLlamaBenchArgs(&cmd, args, false)
		appendExtraArgs(&cmd, opt(args, "extra-bench-args"))
		return strings.Join(cmd, " ")
	case "vllm":
		return vllmKVCacheCommand(args, input, outputTokens, inputTokens)
	case "sglang":
		return sglangKVCacheCommand(args, input, outputTokens)
	default:
		return ""
	}
}

func vllmKVCacheCommand(args cliArgs, input, output string, inputTokens int) string {
	model := firstNonEmpty(opt(args, "hf-id"), opt(args, "model"))
	if model == "" {
		return ""
	}
	bin := firstNonEmpty(opt(args, "benchmark-bin"), "vllm")
	kind := firstNonEmpty(opt(args, "bench-kind"), "latency")
	cmd := []string{shellQuote(bin), "bench", kind}
	if kind == "serve" || kind == "serving" {
		appendShellArg(&cmd, "--backend", firstNonEmpty(opt(args, "benchmark-backend"), "openai"))
		appendShellArg(&cmd, "--model", firstNonEmpty(opt(args, "served-model"), opt(args, "model-name"), model))
		appendShellArg(&cmd, "--base-url", opt(args, "base-url"))
		appendShellArg(&cmd, "--dataset-name", firstNonEmpty(opt(args, "dataset-name"), "random"))
		appendShellArg(&cmd, "--input-len", input)
		appendShellArg(&cmd, "--output-len", output)
		appendShellArg(&cmd, "--num-prompts", firstNonEmpty(opt(args, "num-prompts"), "1"))
	} else {
		appendShellArg(&cmd, "--model", model)
		appendShellArg(&cmd, "--input-len", input)
		appendShellArg(&cmd, "--output-len", output)
		appendShellArg(&cmd, "--batch-size", firstNonEmpty(opt(args, "batch-size"), "1"))
		appendShellArg(&cmd, "--num-iters-warmup", opt(args, "num-warmups"))
		appendShellArg(&cmd, "--num-iters", opt(args, "num-iters"))
		appendShellArg(&cmd, "--tensor-parallel-size", opt(args, "tensor-parallel"))
		appendShellArg(&cmd, "--dtype", opt(args, "dtype"))
		appendShellArg(&cmd, "--quantization", opt(args, "quantization"))
		appendShellArg(&cmd, "--kv-cache-dtype", opt(args, "kv-cache-dtype"))
		appendShellArg(&cmd, "--max-model-len", firstNonEmpty(opt(args, "context-length"), strconv.Itoa(inputTokens+kvOutputTokens(args))))
		if hasFlag(args, "enable-prefix-caching") {
			cmd = append(cmd, "--enable-prefix-caching")
		}
		if outputPath := localKVCacheOutputPath("vllm", args, inputTokens); outputPath != "" {
			appendShellArg(&cmd, "--output-json", outputPath)
		}
	}
	appendExtraArgs(&cmd, opt(args, "extra-bench-args"))
	return strings.Join(cmd, " ")
}

func localKVCacheOutputPath(engineName string, args cliArgs, level int) string {
	if engineName != "vllm" {
		return ""
	}
	base := firstNonEmpty(opt(args, "benchmark-output"), opt(args, "bench-output"))
	if base == "" {
		return filepath.Join(os.TempDir(), fmt.Sprintf("localmaxxing-vllm-kvcache-%d.json", level))
	}
	ext := filepath.Ext(base)
	if ext == "" {
		return fmt.Sprintf("%s-%d", base, level)
	}
	return strings.TrimSuffix(base, ext) + fmt.Sprintf("-%d%s", level, ext)
}

func sglangKVCacheCommand(args cliArgs, input, output string) string {
	modelPath := firstNonEmpty(opt(args, "model-path"), opt(args, "hf-id"), opt(args, "model"))
	if modelPath == "" {
		return ""
	}
	baseURL := opt(args, "base-url")
	if baseURL == "" {
		baseURL = "http://localhost:" + firstNonEmpty(opt(args, "port"), "30000")
	}
	cmd := []string{shellQuote(firstNonEmpty(opt(args, "python-bin"), "python3")), "-m", "sglang.bench_serving"}
	appendShellArg(&cmd, "--backend", firstNonEmpty(opt(args, "benchmark-backend"), "sglang"))
	appendShellArg(&cmd, "--model", firstNonEmpty(opt(args, "served-model"), opt(args, "model-name"), modelPath))
	appendShellArg(&cmd, "--base-url", baseURL)
	appendShellArg(&cmd, "--dataset-name", firstNonEmpty(opt(args, "dataset-name"), "random"))
	appendShellArg(&cmd, "--random-input-len", input)
	appendShellArg(&cmd, "--random-output-len", output)
	appendShellArg(&cmd, "--num-prompts", firstNonEmpty(opt(args, "num-prompts"), "1"))
	appendShellArg(&cmd, "--max-concurrency", opt(args, "max-concurrency"))
	appendExtraArgs(&cmd, opt(args, "extra-bench-args"))
	return strings.Join(cmd, " ")
}

func measureRemoteKVCachePoint(args cliArgs, hfID string, level int) (map[string]any, error) {
	baseURL, err := requireOpt(args, "base-url")
	if err != nil {
		return nil, err
	}
	baseURL = openAIBaseURL(baseURL)
	servedModel := firstNonEmpty(opt(args, "served-model"), opt(args, "model-name"))
	servedModelSource := "explicit"
	var servedModelInfo map[string]any
	if servedModel == "" {
		if detected, info, err := detectServedModel(baseURL, opt(args, "model-api-key"), hfID); err == nil && detected != "" {
			servedModel = detected
			servedModelInfo = info
			servedModelSource = "v1_models"
			printStatus(args, "served_model_detected", map[string]any{"servedModel": servedModel, "source": servedModelSource})
		} else {
			servedModel = hfID
			servedModelSource = "hf_id_fallback"
			printStatus(args, "served_model_fallback", map[string]any{"servedModel": servedModel, "source": servedModelSource, "reason": errString(err)})
		}
	} else if _, info, err := detectServedModel(baseURL, opt(args, "model-api-key"), servedModel); err == nil {
		servedModelInfo = info
	}
	quantizationResolution := remoteQuantizationResolution(args, baseURL, opt(args, "model-api-key"), opt(args, "quantization"), servedModelInfo)
	var modelResolution map[string]any
	if modelPath := stringValue(quantizationResolution["modelPath"]); modelPath != "" {
		modelResolution = remoteModelResolution(args, servedModel, servedModelSource, hfID, modelPath)
	}
	maxTokens := kvOutputTokens(args)
	if hasFlag(args, "dry-run") {
		point := map[string]any{"contextTokens": float64(level), "mode": "remote", "baseUrl": baseURL, "servedModel": servedModel, "servedModelSource": servedModelSource, "maxTokens": float64(maxTokens), "dryRun": true, "methodology": remoteKVCacheColdMethodology}
		if modelResolution != nil {
			point["modelResolution"] = modelResolution
		}
		if quantizationResolution != nil {
			point["quantizationResolution"] = quantizationResolution
		}
		return point, nil
	}
	prompt := kvCachePrompt(args, level)
	prefixMessages := []any{
		map[string]any{"role": "system", "content": "You are measuring decode speed after a long retained context. Answer the final question concisely."},
		map[string]any{"role": "user", "content": prompt},
		map[string]any{"role": "assistant", "content": "Context received."},
	}
	messages := append(append([]any{}, prefixMessages...),
		map[string]any{"role": "user", "content": firstNonEmpty(opt(args, "probe-prompt"), "Continue with a concise benchmark response.")},
	)
	body := map[string]any{"model": servedModel, "messages": messages, "max_tokens": maxTokens, "temperature": 0, "stream": true, "stream_options": map[string]any{"include_usage": true}}
	timeout, err := endpointTimeout(args)
	if err != nil {
		return nil, err
	}
	cacheStatus := map[string]any{"status": "unknown"}
	methodology := remoteKVCacheColdMethodology
	warnings := []string{}
	if err := warmRemoteKVCachePrefix(args, baseURL, servedModel, prefixMessages, timeout); err != nil {
		return nil, err
	}
	cacheTokens, slots, err := remoteKVCacheSlotPromptTokens(args, baseURL, timeout)
	if err != nil {
		warning := "Could not verify llama.cpp /slots cache retention; results may reflect cold prefill rather than cached-context speed."
		warnings = append(warnings, warning)
		cacheStatus["status"] = "unverified"
		cacheStatus["warning"] = warning
		cacheStatus["error"] = err.Error()
		printStatus(args, "kvcache_cache_reuse_unverified", map[string]any{"level": level, "warning": warning, "error": err.Error()})
	} else {
		cacheStatus["nPromptTokensCacheMax"] = cacheTokens
		cacheStatus["slots"] = slots
		if cacheTokens > 0 {
			cacheStatus["status"] = "retained"
			methodology = remoteKVCacheReuseMethodology
			printStatus(args, "kvcache_cache_reuse_detected", map[string]any{"level": level, "nPromptTokensCacheMax": cacheTokens})
		} else {
			warning := "Server does not appear to retain KV cache between requests; results reflect cold prefill, not cached-context speed."
			warnings = append(warnings, warning)
			cacheStatus["status"] = "not_retained"
			cacheStatus["warning"] = warning
			printStatus(args, "kvcache_cache_reuse_missing", map[string]any{"level": level, "warning": warning})
		}
	}
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	bodyData, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	started := time.Now()
	req, err := http.NewRequestWithContext(ctx, "POST", baseURL+"/v1/chat/completions", bytes.NewReader(bodyData))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if key := opt(args, "model-api-key"); key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, cliError{"kvcache_remote_failed", fmt.Sprintf("Could not reach OpenAI-compatible endpoint: %v", err), []string{"Check --base-url and confirm the endpoint is reachable from this machine."}, nil}
	}
	defer res.Body.Close()
	if res.StatusCode < 200 || res.StatusCode >= 300 {
		text, _ := io.ReadAll(res.Body)
		return nil, cliError{"kvcache_remote_failed", fmt.Sprintf("OpenAI-compatible endpoint returned %s", res.Status), []string{"Check --base-url, --served-model, --model-api-key, and the target context level."}, string(text)}
	}
	streamResult, err := readOpenAIStream(args, res.Body, started)
	if err != nil {
		return nil, err
	}
	completedAt := streamResult.completedAt
	if completedAt.IsZero() {
		completedAt = time.Now()
	}
	generationStart := streamResult.firstTokenAt
	if generationStart.IsZero() {
		generationStart = started
	}
	usagePromptTokens := usageToken(streamResult.usage, "prompt_tokens")
	usageOutputTokens := firstNonZero(usageToken(streamResult.usage, "completion_tokens"), usageToken(streamResult.usage, "output_tokens"))
	outputTokens := usageOutputTokens
	if outputTokens == 0 {
		count, err := tokenCount(args, hfID, firstNonEmpty(opt(args, "model-revision"), "main"), streamResult.outputText, 0, "output")
		if err != nil {
			return nil, err
		}
		outputTokens = count.Count
	}
	promptTokens := level
	if usagePromptTokens > 0 {
		if stringValue(cacheStatus["status"]) == "retained" {
			promptTokens = level + usagePromptTokens
		} else {
			promptTokens = usagePromptTokens
		}
	}
	totalMs := maxDurationMS(completedAt.Sub(started))
	generationMs := maxDurationMS(completedAt.Sub(generationStart))
	point := map[string]any{
		"contextTokens":     float64(level),
		"promptTokens":      float64(promptTokens),
		"outputTokens":      float64(outputTokens),
		"tokSOut":           round1(float64(outputTokens) / (generationMs / 1000)),
		"tokSTotal":         round1(float64(promptTokens+outputTokens) / (totalMs / 1000)),
		"outputText":        streamResult.outputText,
		"mode":              "remote",
		"baseUrl":           baseURL,
		"servedModel":       servedModel,
		"servedModelSource": servedModelSource,
		"methodology":       methodology,
		"cacheReuse":        cacheStatus,
		"usagePromptTokens": float64(usagePromptTokens),
		"metricSource":      "remote_endpoint",
		"timingSource":      "client_observed_http",
		"tokSPrefillSource": "estimated_from_ttft",
	}
	if modelResolution != nil {
		point["modelResolution"] = modelResolution
	}
	if quantizationResolution != nil {
		point["quantizationResolution"] = quantizationResolution
	}
	if len(warnings) > 0 {
		point["warnings"] = warnings
	}
	if !streamResult.firstTokenAt.IsZero() {
		ttftMs := float64(streamResult.firstTokenAt.Sub(started).Milliseconds())
		point["ttftMs"] = ttftMs
		if ttftMs > 0 {
			point["tokSPrefill"] = round1(float64(promptTokens) / (ttftMs / 1000))
		}
	}
	return point, nil
}

func warmRemoteKVCachePrefix(args cliArgs, baseURL, servedModel string, messages []any, timeout time.Duration) error {
	body := map[string]any{"model": servedModel, "messages": messages, "max_tokens": 1, "temperature": 0, "stream": false}
	bodyData, err := json.Marshal(body)
	if err != nil {
		return err
	}
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, "POST", baseURL+"/v1/chat/completions", bytes.NewReader(bodyData))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	if key := opt(args, "model-api-key"); key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		return cliError{"kvcache_remote_failed", fmt.Sprintf("Could not pre-warm remote KV-cache prefix: %v", err), []string{"Check --base-url and confirm the endpoint is reachable from this machine."}, nil}
	}
	defer res.Body.Close()
	data, _ := io.ReadAll(res.Body)
	if res.StatusCode < 200 || res.StatusCode >= 300 {
		return cliError{"kvcache_remote_failed", fmt.Sprintf("Remote KV-cache pre-warm returned %s", res.Status), []string{"Check --base-url, --served-model, and the target context level."}, string(data)}
	}
	return nil
}

func remoteKVCacheSlotPromptTokens(args cliArgs, baseURL string, timeout time.Duration) (int, any, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, "GET", baseURL+"/slots", nil)
	if err != nil {
		return 0, nil, err
	}
	if key := opt(args, "model-api-key"); key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		return 0, nil, err
	}
	defer res.Body.Close()
	data, _ := io.ReadAll(res.Body)
	if res.StatusCode < 200 || res.StatusCode >= 300 {
		return 0, nil, fmt.Errorf("GET /slots returned %s", res.Status)
	}
	var slots any
	if err := json.Unmarshal(data, &slots); err != nil {
		return 0, nil, err
	}
	return maxSlotPromptCacheTokens(slots), slots, nil
}

func maxSlotPromptCacheTokens(value any) int {
	maxValue := 0
	var visit func(any)
	visit = func(current any) {
		switch typed := current.(type) {
		case []any:
			for _, item := range typed {
				visit(item)
			}
		case map[string]any:
			if n := int(numberField(typed, "n_prompt_tokens_cache")); n > maxValue {
				maxValue = n
			}
			for _, item := range typed {
				visit(item)
			}
		}
	}
	visit(value)
	return maxValue
}

func kvCachePrompt(args cliArgs, targetTokens int) string {
	if prompt := opt(args, "prompt"); prompt != "" {
		return prompt
	}
	word := firstNonEmpty(opt(args, "filler-token"), "context")
	if targetTokens < 1 {
		targetTokens = 1
	}
	return strings.TrimSpace(strings.Repeat(word+" ", targetTokens))
}

func benchmarkPayloadFromFlags(engine string, args cliArgs) (map[string]any, error) {
	model := firstNonEmpty(opt(args, "hf-id"), opt(args, "model"))
	if model == "" {
		return nil, cliError{"missing_model", "benchmark run requires --hf-id or --model", []string{"Pass --hf-id <HuggingFace model id>."}, nil}
	}
	engineName, err := resolveEngineName(engine, args, true)
	if err != nil {
		return nil, err
	}
	backend := opt(args, "backend")
	if backend == "" {
		backend = engineBackendDefault(engineName)
	}
	quantization := opt(args, "quantization")
	if quantization == "" {
		return nil, cliError{"missing_option", "--quantization is required", []string{"Pass --quantization <label>, e.g. fp16, Q4_K_M, or int8."}, nil}
	}

	metrics := map[string]any{}
	mode := benchmarkMode(args)
	printStatus(args, "benchmark_mode_selected", map[string]any{"mode": mode, "engine": engineName, "model": model})
	if mode == "remote" && opt(args, "command") != "" {
		return nil, cliError{"invalid_benchmark_mode", "Remote benchmark mode cannot run local commands.", []string{"Use --mode local when running llama-bench on the host server.", "Use --mode remote --base-url <url> for OpenAI-compatible endpoint TPS measurement."}, nil}
	}
	if mode == "local" && opt(args, "base-url") != "" && opt(args, "command") == "" && opt(args, "results") == "" && localBenchmarkCommand(engineName, args) == "" {
		return nil, cliError{"invalid_benchmark_mode", "Local benchmark mode needs --command, --results, or explicit metric flags.", []string{"Use --mode remote with --base-url when benchmarking an endpoint from another machine.", "Use --mode local --command \"llama-bench ...\" when running on the host server."}, nil}
	}
	var commandOutput string
	if hasFlag(args, "dry-run") {
		applyBenchmarkPlanMetrics(metrics, mode, engineName, args, model)
	} else if mode == "remote" {
		printStatus(args, "benchmark_remote_start", map[string]any{"baseUrl": opt(args, "base-url"), "servedModel": firstNonEmpty(opt(args, "served-model"), opt(args, "model-name"), model)})
		var endpointMetrics map[string]any
		if engineName == "ollama" {
			endpointMetrics, err = measureOllamaEndpoint(args, model)
		} else {
			endpointMetrics, err = measureOpenAIEndpoint(args, model)
		}
		if err != nil {
			return nil, err
		}
		for key, value := range endpointMetrics {
			metrics[key] = value
		}
	} else if resultsPath := opt(args, "results"); resultsPath != "" {
		printStatus(args, "benchmark_results_read_start", map[string]any{"path": resultsPath})
		data, err := os.ReadFile(resultsPath)
		if err != nil {
			return nil, err
		}
		commandOutput = string(data)
		metrics["engineFlags"] = map[string]any{"mode": mode, "commandSnippet": "# Metrics imported from " + resultsPath, "resultsPath": resultsPath}
		printStatus(args, "benchmark_results_read_complete", map[string]any{"path": resultsPath, "bytes": len(data)})
	} else if commandSnippet := localBenchmarkCommand(engineName, args); commandSnippet != "" {
		printStatus(args, "benchmark_local_command_start", map[string]any{"command": commandSnippet})
		output, err := runBenchmarkCommand(commandSnippet)
		if err != nil {
			return nil, err
		}
		commandOutput = output
		if outputPath := benchmarkOutputPath(args); outputPath != "" {
			if data, err := os.ReadFile(outputPath); err == nil {
				commandOutput = strings.TrimSpace(commandOutput + "\n" + string(data))
				printStatus(args, "benchmark_results_read_complete", map[string]any{"path": outputPath, "bytes": len(data)})
			}
		}
		metrics["engineFlags"] = localBenchmarkEngineFlags(engineName, commandSnippet)
		printStatus(args, "benchmark_local_command_complete", map[string]any{"outputBytes": len(output)})
	}
	if commandOutput != "" {
		parsed := parseBenchmarkOutput(commandOutput)
		for key, value := range parsed {
			metrics[key] = value
		}
		printStatus(args, "benchmark_metrics_detected", metricStatusFields(parsed))
	}

	hardware, hardwareSource, err := benchmarkHardware(mode, args)
	if err != nil {
		return nil, err
	}

	payload := map[string]any{"engineName": engineName, "hfId": model, "modelRevision": firstNonEmpty(opt(args, "model-revision"), "main"), "quantization": quantization, "detectedEngines": detectInferenceEngines(args)}
	if hardware != nil {
		payload["hardware"] = hardware
	}
	if hardwareSource != "" {
		payload["hardwareSource"] = hardwareSource
	}
	if backend != "" {
		payload["backend"] = backend
	}
	payload["benchmarkMode"] = mode
	for key, value := range metrics {
		payload[key] = value
	}
	for flag, field := range map[string]string{"tok-s-out": "tokSOut", "tok-s-prefill": "tokSPrefill", "tok-s-total": "tokSTotal", "ttft-ms": "ttftMs", "peak-vram-gb": "peakVramGb", "context-length": "contextLength", "batch-size": "batchSize", "input-len": "inputLen", "output-len": "outputLen", "prompt-tokens": "promptTokens", "prefill-tokens": "promptTokens", "output-tokens": "outputTokens", "num-prompts": "numPrompts"} {
		if value := opt(args, flag); value != "" {
			if n, err := strconv.ParseFloat(value, 64); err == nil {
				payload[field] = n
			} else {
				payload[field] = value
			}
		}
	}
	applyCommandTokenHints(payload)
	if notes := opt(args, "notes"); notes != "" {
		payload["notes"] = notes
	}
	applyComparableBenchmarkMetrics(payload, mode, engineName)
	payload["provenance"] = benchmarkProvenance(payload, mode)
	if hasFlag(args, "dry-run") && numberField(payload, "tokSOut") == 0 {
		printStatus(args, "benchmark_plan_ready", map[string]any{"mode": mode, "engine": engineName, "dryRunType": "measurement_plan", "next": "Run without --dry-run to measure, or pass explicit --tok-s-out metrics. API validation is lmx benchmark dry-run <payload.json>."})
		return payload, nil
	}
	if tokSOut, ok := payload["tokSOut"].(float64); !ok || tokSOut <= 0 {
		details := commandOutput
		if len(details) > 4000 {
			details = details[:4000]
		}
		return nil, cliError{"benchmark_metric_missing", "Could not determine tokSOut from the benchmark output.", []string{"Pass --tok-s-out <tokens_per_second> explicitly.", "For llama-bench, include text table output with a tg<N> row.", "For vLLM/SGLang benchmark scripts, prefer JSON output or include \"Output token throughput\" text."}, details}
	}
	printStatus(args, "benchmark_payload_ready", map[string]any{"mode": mode, "tokSOut": payload["tokSOut"], "tokSPrefill": payload["tokSPrefill"], "tokSTotal": payload["tokSTotal"], "ttftMs": payload["ttftMs"]})
	return payload, nil
}

func benchmarkHardware(mode string, args cliArgs) (any, string, error) {
	if hardwarePath := opt(args, "hardware"); hardwarePath != "" {
		loaded, err := readJSON(hardwarePath)
		if err != nil {
			return nil, "", err
		}
		if asObject(loaded) == nil {
			return nil, "", cliError{"invalid_hardware", "--hardware must point to a JSON object.", []string{"Generate one with lmx hardware --out hardware.json on the benchmark host."}, loaded}
		}
		return loaded, "file", nil
	}
	if mode == "remote" {
		return nil, "missing_remote", nil
	}
	return detectHardware(), "detected_local", nil
}

func validateBenchmarkSubmitPayload(payload map[string]any) error {
	if payload == nil {
		return cliError{"invalid_benchmark_payload", "Benchmark payload must be a JSON object.", nil, nil}
	}
	if stringValue(payload["benchmarkMode"]) == "remote" && asObject(payload["hardware"]) == nil {
		return cliError{"missing_remote_hardware", "Remote benchmark submission requires explicit server hardware metadata.", []string{"Run lmx hardware --out hardware.json on the machine running the endpoint, or create an equivalent hardware JSON for that server.", "Rerun the remote benchmark with --hardware hardware.json before dry-run or submit.", "Do not rely on client auto-detected hardware for endpoint benchmarks."}, nil}
	}
	return nil
}

// toBenchmarkSubmit strips internal-only fields and remaps hardware/engineFlags
// to match the localmaxxing.com POST /api/benchmarks schema.
func toBenchmarkSubmit(payload map[string]any) map[string]any {
	out := map[string]any{}
	if payload == nil {
		return out
	}

	for _, key := range []string{
		"hfId", "modelRevision", "engineName", "engineVersion",
		"quantization", "backend", "promptTokens", "outputTokens",
		"contextLength", "batchSize", "temperature", "topP",
		"ttftMs", "tokSOut", "tokSPrefill", "tokSTotal",
		"peakVramGb", "prefillTokens", "notes",
	} {
		if v, ok := payload[key]; ok && submitValuePresent(v) {
			out[key] = v
		}
	}

	if hw := asObject(payload["hardware"]); hw != nil {
		out["hardware"] = remapHardware(hw)
	}
	if ef := asObject(payload["engineFlags"]); ef != nil {
		out["engineFlags"] = remapEngineFlags(ef, payload)
	}

	return out
}

func remapHardware(hw map[string]any) map[string]any {
	hwClass := "CPU_ONLY"
	if firstNonEmpty(stringValue(hw["gpuName"]), stringValue(hw["gpus"])) != "" || len(anySlice(hw["gpus"])) > 0 {
		hwClass = "DISCRETE_GPU"
	} else if firstNonEmpty(stringValue(hw["chipVendor"]), stringValue(hw["chipFamily"])) != "" || stringValue(hw["hwClass"]) == "UNIFIED" || stringValue(hw["hwClass"]) == "APPLE_SILICON" {
		hwClass = "UNIFIED"
	}

	remapped := map[string]any{"hwClass": hwClass}
	if hwClass == "DISCRETE_GPU" {
		remapped["gpuName"] = firstNonEmpty(stringValue(hw["gpuName"]), "Unknown GPU")
		remapped["vramGb"] = numberField(hw, "vramGb")
		if remapped["vramGb"] == 0 {
			remapped["vramGb"] = numberField(hw, "gpuVramGb")
		}
		if c := numberField(hw, "gpuCount"); c > 0 {
			remapped["gpuCount"] = c
		}
		if gpus := anySlice(hw["gpus"]); len(gpus) > 0 {
			remapped["gpus"] = gpus
			delete(remapped, "gpuName")
			delete(remapped, "vramGb")
		}
	}
	if hwClass == "UNIFIED" {
		remapped["chipVendor"] = stringValue(hw["chipVendor"])
		remapped["chipFamily"] = stringValue(hw["chipFamily"])
		remapped["chipVariant"] = stringValue(hw["chipVariant"])
		remapped["unifiedMemoryGb"] = numberField(hw, "unifiedMemoryGb")
	}
	if cpu := firstNonEmpty(stringValue(hw["cpu"]), stringValue(hw["cpuName"])); cpu != "" {
		remapped["cpu"] = cpu
	}
	if ram := numberField(hw, "ramGb"); ram > 0 {
		remapped["ramGb"] = ram
	} else if ram = numberField(hw, "systemMemoryGb"); ram > 0 {
		remapped["ramGb"] = ram
	}
	if osName := firstNonEmpty(stringValue(hw["os"]), stringValue(hw["systemOs"])); osName != "" {
		remapped["os"] = osName
	}
	if power := numberField(hw, "powerWatts"); power > 0 {
		remapped["powerWatts"] = power
	}

	return remapped
}

func remapEngineFlags(ef map[string]any, payload map[string]any) map[string]any {
	remapped := map[string]any{}
	cmd := stringValue(ef["commandSnippet"])
	if cmd == "" {
		mode := stringValue(ef["mode"])
		engine := stringValue(payload["engineName"])
		if mode == "remote" {
			baseURL := stringValue(ef["baseUrl"])
			servedModel := stringValue(ef["servedModel"])
			cmd = fmt.Sprintf("# Remote endpoint: %s  servedModel: %s", baseURL, servedModel)
			if engine == "llama.cpp" {
				if mp := stringValue(payload["modelPath"]); mp != "" {
					cmd = fmt.Sprintf("llama-server -m %s --host %s", shellQuote(mp), shellQuote(baseURL))
				}
			}
		} else {
			cmd = localBenchmarkCommand(engine, cliArgs{})
		}
	}
	if cmd == "" {
		cmd = "# No command snippet available"
	}
	remapped["commandSnippet"] = cmd

	for _, key := range []string{
		"gpuLayers", "cpuLayers", "flashAttn", "kvCacheDtype", "prefixCaching",
		"contBatching", "chunkedPrefill", "tensorParallel", "pipelineParallel",
		"concurrency", "numParallel", "maxRunningSeqs", "gpuMemUtil",
		"specDecoding", "specModel", "specDraftModel", "specNgramSize",
		"specNumTokens", "specDraftTp", "specMethod", "mtpEnabled", "mtpDraftLayers",
		"temperature", "topP", "topK", "minP", "repeatPenalty", "mirostat",
		"ropeScale", "ropeScaling", "yarnExtFactor", "schedulerDelayFactor",
		"attentionBackend", "sglangQuant", "engineQuant", "splitMode", "warmup",
		"prefillChunkSize", "kvCacheSizeMb", "cpuOffloadGb", "extraFlags",
	} {
		if v, ok := ef[key]; ok && submitValuePresent(v) {
			remapped[key] = v
		}
	}

	return remapped
}

func engineBackendDefault(engine string) string {
	switch engine {
	case "llama.cpp", "vllm", "sglang", "tensorrt-llm", "exllamav2", "lmdeploy", "tgi":
		return "cuda"
	case "mlx":
		return "metal"
	default:
		return ""
	}
}

func submitValuePresent(value any) bool {
	switch v := value.(type) {
	case nil:
		return false
	case string:
		return v != ""
	case bool:
		return v
	case int:
		return v != 0
	case int64:
		return v != 0
	case float64:
		return v != 0
	case float32:
		return v != 0
	default:
		return true
	}
}

func anySlice(value any) []any {
	switch v := value.(type) {
	case []any:
		return v
	case []map[string]any:
		out := make([]any, len(v))
		for i, item := range v {
			out[i] = item
		}
		return out
	default:
		return nil
	}
}

func benchmarkAgentFeedback(payload map[string]any, out string, args cliArgs, dryRun, submit bool) map[string]any {
	ready := numberField(payload, "tokSOut") > 0
	requiresRemoteHardware := stringValue(payload["benchmarkMode"]) == "remote" && asObject(payload["hardware"]) == nil
	missingAuth := ready && !requiresRemoteHardware && !submit && apiKey(args) == ""
	status := "ready_for_api_validation"
	message := "Benchmark payload is ready for API validation."
	nextCommand := "lmx benchmark dry-run " + out
	if submit {
		status = "api_submission_requested"
		message = "Benchmark payload is being sent to the API."
		nextCommand = ""
	} else if ready && requiresRemoteHardware {
		status = "needs_remote_hardware"
		message = "Remote benchmark has metrics but no server hardware file. Generate hardware metadata on the endpoint host and rerun with --hardware before validating or submitting."
		nextCommand = "lmx hardware --out hardware.json"
	} else if dryRun && !ready {
		status = "plan_needs_metrics"
		message = "Measurement plan written. No benchmark command or API request ran. Run without --dry-run to measure, or pass explicit --tok-s-out metrics before API validation."
		nextCommand = "lmx benchmark run <engine> <same options without --dry-run>"
		if requiresRemoteHardware {
			message += " Remote endpoint submissions also require --hardware with metadata from the endpoint host."
			nextCommand += " --hardware hardware.json"
		}
	} else if dryRun {
		status = "plan_ready_for_api_validation"
		message = "Dry-run measurement plan written with metrics. No API request ran. Validate locally with lmx benchmark validate-local, or run authenticated API validation with lmx benchmark dry-run."
	}
	if missingAuth {
		message += " API validation and submission require an API key."
		nextCommand = "lmx auth --key bhk_..."
	}
	canUseAPI := ready && !requiresRemoteHardware && !missingAuth
	feedback := map[string]any{
		"status":          status,
		"message":         message,
		"outputPath":      out,
		"canApiValidate":  canUseAPI,
		"canSubmit":       canUseAPI,
		"requiresMetrics": !ready,
	}
	if requiresRemoteHardware {
		feedback["requiresHardware"] = true
		feedback["hardwareCommand"] = "lmx hardware --out hardware.json"
	}
	if nextCommand != "" {
		feedback["nextCommand"] = nextCommand
	}
	if ready && !requiresRemoteHardware && !submit && !missingAuth {
		feedback["submitCommand"] = "lmx benchmark submit " + out
	}
	if missingAuth {
		feedback["authRequired"] = true
		feedback["blockedByAuth"] = true
		feedback["authCommand"] = "lmx auth --key bhk_..."
		feedback["validationCommand"] = "lmx benchmark validate-local " + out
	}
	if engine := stringValue(payload["engineName"]); engine != "" {
		feedback["engine"] = engine
	}
	if mode := stringValue(payload["benchmarkMode"]); mode != "" {
		feedback["mode"] = mode
	}
	if cmd := detectedMetadataRerunCommand(payload, out); cmd != "" {
		feedback["rerunWithDetectedMetadataCommand"] = cmd
	}
	return feedback
}

func detectedMetadataRerunCommand(payload map[string]any, out string) string {
	if stringValue(payload["benchmarkMode"]) != "remote" {
		return ""
	}
	hfID := stringValue(payload["hfId"])
	quantization := stringValue(payload["quantization"])
	changed := false
	if modelResolution := asObject(payload["modelResolution"]); modelResolution != nil {
		if detected := stringValue(modelResolution["sourceRepo"]); detected != "" && detected != hfID {
			hfID = detected
			changed = true
		}
	}
	if quantizationResolution := asObject(payload["quantizationResolution"]); quantizationResolution != nil {
		if trusted := stringValue(quantizationResolution["trusted"]); trusted != "" && !quantizationEqual(trusted, quantization) {
			quantization = trusted
			changed = true
		}
	}
	if !changed {
		return ""
	}
	engine := firstNonEmpty(stringValue(payload["engineName"]), "llama.cpp")
	engineFlags := asObject(payload["engineFlags"])
	cmd := []string{"lmx", "benchmark", "run", engine, "--mode", "remote"}
	appendShellArg(&cmd, "--base-url", stringValue(engineFlags["baseUrl"]))
	appendShellArg(&cmd, "--served-model", stringValue(engineFlags["servedModel"]))
	appendShellArg(&cmd, "--hf-id", hfID)
	appendShellArg(&cmd, "--quantization", quantization)
	appendShellArg(&cmd, "--hardware", "hardware.json")
	appendShellArg(&cmd, "--out", out)
	return strings.Join(cmd, " ")
}

func applyBenchmarkPlanMetrics(metrics map[string]any, mode, engineName string, args cliArgs, model string) {
	metrics["dryRun"] = true
	if mode == "remote" {
		baseURL := opt(args, "base-url")
		servedModel := firstNonEmpty(opt(args, "served-model"), opt(args, "model-name"), model)
		cmdSnippet := fmt.Sprintf("# Remote endpoint: %s  servedModel: %s", baseURL, servedModel)
		if engineName == "llama.cpp" && opt(args, "model-path") != "" {
			cmdSnippet = fmt.Sprintf("llama-server -m %s", shellQuote(opt(args, "model-path")))
			if host := opt(args, "host"); host != "" {
				cmdSnippet += fmt.Sprintf(" --host %s", shellQuote(host))
			}
			if port := opt(args, "port"); port != "" {
				cmdSnippet += fmt.Sprintf(" --port %s", shellQuote(port))
			}
		}
		if sb := opt(args, "server-bin"); sb != "" {
			cmdSnippet = sb + " " + strings.TrimPrefix(cmdSnippet, "llama-server ")
		}
		metrics["engineFlags"] = map[string]any{"commandSnippet": cmdSnippet, "mode": "remote", "baseUrl": baseURL, "servedModel": servedModel}
		return
	}
	if commandSnippet := localBenchmarkCommand(engineName, args); commandSnippet != "" {
		metrics["engineFlags"] = localBenchmarkEngineFlags(engineName, commandSnippet)
	}
}

func benchmarkProvenance(payload map[string]any, mode string) map[string]any {
	provenance := map[string]any{"cli": "localmaxxing-go", "benchmarkMode": mode, "createdAt": time.Now().UTC().Format(time.RFC3339)}
	for _, key := range []string{"metricSource", "timingSource", "ttftSource"} {
		if value, ok := payload[key]; ok && value != nil && value != "" {
			provenance[key] = value
		}
	}
	return provenance
}

func applyComparableBenchmarkMetrics(payload map[string]any, mode, engineName string) {
	promptTokens := numberField(payload, "promptTokens")
	outputTokens := numberField(payload, "outputTokens")
	tokSPrefill := numberField(payload, "tokSPrefill")
	tokSOut := numberField(payload, "tokSOut")
	if mode == "local" {
		payload["metricSource"] = "local_runtime"
		payload["timingSource"] = strings.ReplaceAll(engineName, ".", "_") + "_runtime"
		if numberField(payload, "ttftMs") == 0 && promptTokens > 0 && tokSPrefill > 0 {
			payload["ttftMs"] = round1((promptTokens / tokSPrefill) * 1000)
			payload["ttftSource"] = "estimated_from_prefill"
		}
		if numberField(payload, "tokSTotal") == 0 && promptTokens > 0 && outputTokens > 0 && tokSPrefill > 0 && tokSOut > 0 {
			prefillSeconds := promptTokens / tokSPrefill
			decodeSeconds := outputTokens / tokSOut
			payload["tokSTotal"] = round1((promptTokens + outputTokens) / (prefillSeconds + decodeSeconds))
			payload["tokSTotalSource"] = "derived_from_prefill_and_decode"
		}
		if _, ok := payload["ttftSource"]; !ok && numberField(payload, "ttftMs") > 0 {
			payload["ttftSource"] = "reported"
		}
		return
	}
	if mode == "remote" {
		payload["metricSource"] = "remote_endpoint"
		if _, ok := payload["timingSource"]; !ok {
			payload["timingSource"] = "client_observed_http"
		}
		if _, ok := payload["ttftSource"]; !ok && numberField(payload, "ttftMs") > 0 {
			payload["ttftSource"] = "stream_first_token"
		}
	}
}

func numberField(obj map[string]any, key string) float64 {
	switch value := obj[key].(type) {
	case float64:
		return value
	case float32:
		return float64(value)
	case int:
		return float64(value)
	case int64:
		return float64(value)
	case string:
		parsed, _ := strconv.ParseFloat(value, 64)
		return parsed
	default:
		return 0
	}
}

func localBenchmarkCommand(engineName string, args cliArgs) string {
	if command := opt(args, "command"); command != "" {
		return command
	}
	if engineName == "vllm" {
		return vllmBenchmarkCommand(args)
	}
	if engineName == "sglang" {
		return sglangBenchmarkCommand(args)
	}
	if engineName != "llama.cpp" || opt(args, "model-path") == "" {
		return ""
	}
	cmd := []string{shellQuote(resolvedExecutable(firstNonEmpty(opt(args, "benchmark-bin"), "llama-bench"))), "-m", shellQuote(opt(args, "model-path"))}
	if value := firstNonEmpty(opt(args, "prompt-tokens"), opt(args, "prefill-tokens"), "512"); value != "" {
		cmd = append(cmd, "-p", shellQuote(value))
	}
	if value := firstNonEmpty(opt(args, "output-tokens"), "128"); value != "" {
		cmd = append(cmd, "-n", shellQuote(value))
	}
	if value := opt(args, "threads"); value != "" {
		cmd = append(cmd, "-t", shellQuote(value))
	}
	if value := opt(args, "gpu-layers"); value != "" {
		cmd = append(cmd, "-ngl", shellQuote(value))
	}
	appendLlamaBenchArgs(&cmd, args, true)
	if value := opt(args, "extra-bench-args"); value != "" {
		cmd = append(cmd, value)
	}
	return strings.Join(cmd, " ")
}

func resolvedExecutable(name string) string {
	if path, ok := lookupExecutable(name); ok {
		return path
	}
	return name
}

func localBenchmarkEngineFlags(engineName, commandSnippet string) map[string]any {
	flags := map[string]any{"mode": "local", "commandSnippet": commandSnippet}
	if engineName == "llama.cpp" {
		if strings.Contains(commandSnippet, "--no-warmup") {
			flags["warmup"] = "disabled"
		} else {
			flags["warmup"] = "llama-bench-default"
		}
	}
	return flags
}

func appendLlamaBenchArgs(cmd *[]string, args cliArgs, includeDepth bool) {
	if includeDepth {
		appendShellArg(cmd, "-d", firstNonEmpty(opt(args, "depth"), opt(args, "context-depth")))
	}
	appendShellArg(cmd, "-b", opt(args, "batch-size"))
	appendShellArg(cmd, "-ub", firstNonEmpty(opt(args, "micro-batch-size"), opt(args, "ubatch-size")))
	appendShellArg(cmd, "-r", firstNonEmpty(opt(args, "repetitions"), opt(args, "runs")))
	appendShellArg(cmd, "-ctk", opt(args, "cache-type-k"))
	appendShellArg(cmd, "-ctv", opt(args, "cache-type-v"))
	if includeDepth {
		appendShellArg(cmd, "-o", firstNonEmpty(opt(args, "benchmark-format"), opt(args, "bench-format"), opt(args, "output-format")))
	}
	if hasFlag(args, "flash-attn") {
		*cmd = append(*cmd, "-fa", "1")
	} else if hasFlag(args, "no-flash-attn") {
		*cmd = append(*cmd, "-fa", "0")
	} else if value := opt(args, "flash-attn"); value != "" {
		*cmd = append(*cmd, "-fa", shellQuote(value))
	}
}

func applyCommandTokenHints(payload map[string]any) {
	engineFlags := asObject(payload["engineFlags"])
	if engineFlags == nil {
		return
	}
	command := stringValue(engineFlags["commandSnippet"])
	if command == "" {
		return
	}
	if numberField(payload, "promptTokens") == 0 {
		if value := commandFlagNumber(command, "-p", "--prompt-tokens", "--prefill-tokens", "--input-len", "--random-input-len"); value > 0 {
			payload["promptTokens"] = value
		}
	}
	if numberField(payload, "outputTokens") == 0 {
		if value := commandFlagNumber(command, "-n", "--output-tokens", "--output-len", "--random-output-len", "--max-tokens"); value > 0 {
			payload["outputTokens"] = value
		}
	}
	if numberField(payload, "contextLength") == 0 {
		if value := commandFlagNumber(command, "-c", "-d", "--context-length", "--context-depth", "--max-model-len"); value > 0 {
			payload["contextLength"] = value
		}
	}
	if numberField(payload, "batchSize") == 0 {
		if value := commandFlagNumber(command, "-b", "--batch-size"); value > 0 {
			payload["batchSize"] = value
		}
	}
}

func commandFlagNumber(command string, flags ...string) float64 {
	for _, flag := range flags {
		patterns := []string{
			regexp.QuoteMeta(flag) + `(?:=|\s+)([0-9]+(?:\.[0-9]+)?)`,
		}
		if len(flag) == 2 && strings.HasPrefix(flag, "-") && !strings.HasPrefix(flag, "--") {
			patterns = append(patterns, regexp.QuoteMeta(flag)+`([0-9]+(?:\.[0-9]+)?)`)
		}
		for _, pattern := range patterns {
			re := regexp.MustCompile(`(?:^|\s)` + pattern + `(?:\s|$)`)
			match := re.FindStringSubmatch(command)
			if len(match) < 2 {
				continue
			}
			value, err := strconv.ParseFloat(match[1], 64)
			if err == nil && value > 0 {
				return value
			}
		}
	}
	return 0
}

func benchmarkKind(args cliArgs, fallback string) string {
	return strings.ToLower(firstNonEmpty(opt(args, "bench-kind"), opt(args, "benchmark"), fallback))
}

func benchmarkOutputPath(args cliArgs) string {
	return firstNonEmpty(opt(args, "benchmark-output"), opt(args, "bench-output"))
}

func appendShellArg(cmd *[]string, flag, value string) {
	if value == "" {
		return
	}
	*cmd = append(*cmd, flag, shellQuote(value))
}

func vllmBenchmarkCommand(args cliArgs) string {
	model := firstNonEmpty(opt(args, "hf-id"), opt(args, "model"))
	if model == "" {
		return ""
	}
	kind := benchmarkKind(args, func() string {
		if opt(args, "base-url") != "" {
			return "serve"
		}
		return "throughput"
	}())
	bin := firstNonEmpty(opt(args, "benchmark-bin"), "vllm")
	inputLen := firstNonEmpty(opt(args, "input-len"), opt(args, "prompt-tokens"), "512")
	outputLen := firstNonEmpty(opt(args, "output-len"), opt(args, "output-tokens"), "128")
	numPrompts := firstNonEmpty(opt(args, "num-prompts"), "100")
	outputPath := benchmarkOutputPath(args)
	cmd := []string{shellQuote(bin), "bench", kind}

	switch kind {
	case "serve", "serving":
		appendShellArg(&cmd, "--backend", firstNonEmpty(opt(args, "benchmark-backend"), "openai"))
		appendShellArg(&cmd, "--model", firstNonEmpty(opt(args, "served-model"), opt(args, "model-name"), model))
		appendShellArg(&cmd, "--base-url", opt(args, "base-url"))
		appendShellArg(&cmd, "--host", opt(args, "host"))
		appendShellArg(&cmd, "--port", opt(args, "port"))
		appendShellArg(&cmd, "--endpoint", opt(args, "endpoint"))
		appendShellArg(&cmd, "--dataset-name", firstNonEmpty(opt(args, "dataset-name"), "random"))
		appendShellArg(&cmd, "--dataset-path", opt(args, "dataset-path"))
		appendShellArg(&cmd, "--input-len", inputLen)
		appendShellArg(&cmd, "--output-len", outputLen)
		appendShellArg(&cmd, "--num-prompts", numPrompts)
		appendShellArg(&cmd, "--request-rate", opt(args, "request-rate"))
		appendShellArg(&cmd, "--max-concurrency", opt(args, "max-concurrency"))
		if outputPath != "" {
			cmd = append(cmd, "--save-result", "--result-filename", shellQuote(outputPath))
		}
	case "throughput":
		appendShellArg(&cmd, "--backend", firstNonEmpty(opt(args, "benchmark-backend"), "vllm"))
		appendShellArg(&cmd, "--model", model)
		appendShellArg(&cmd, "--dataset-name", firstNonEmpty(opt(args, "dataset-name"), "random"))
		appendShellArg(&cmd, "--dataset-path", opt(args, "dataset-path"))
		appendShellArg(&cmd, "--input-len", inputLen)
		appendShellArg(&cmd, "--output-len", outputLen)
		appendShellArg(&cmd, "--num-prompts", numPrompts)
		appendShellArg(&cmd, "--num-warmups", opt(args, "num-warmups"))
		appendShellArg(&cmd, "--tensor-parallel-size", opt(args, "tensor-parallel"))
		appendShellArg(&cmd, "--max-model-len", opt(args, "context-length"))
		if outputPath != "" {
			cmd = append(cmd, "--output-json", shellQuote(outputPath))
		}
	case "latency":
		appendShellArg(&cmd, "--model", model)
		appendShellArg(&cmd, "--input-len", inputLen)
		appendShellArg(&cmd, "--output-len", outputLen)
		appendShellArg(&cmd, "--batch-size", firstNonEmpty(opt(args, "batch-size"), "1"))
		appendShellArg(&cmd, "--num-iters-warmup", opt(args, "num-warmups"))
		appendShellArg(&cmd, "--num-iters", opt(args, "num-iters"))
		appendShellArg(&cmd, "--tensor-parallel-size", opt(args, "tensor-parallel"))
		appendShellArg(&cmd, "--max-model-len", opt(args, "context-length"))
		if outputPath != "" {
			cmd = append(cmd, "--output-json", shellQuote(outputPath))
		}
	default:
		return ""
	}
	if extra := opt(args, "extra-bench-args"); extra != "" {
		cmd = append(cmd, extra)
	}
	return strings.Join(cmd, " ")
}

func sglangBenchmarkCommand(args cliArgs) string {
	modelPath := firstNonEmpty(opt(args, "model-path"), opt(args, "hf-id"), opt(args, "model"))
	if modelPath == "" {
		return ""
	}
	kind := benchmarkKind(args, "serve")
	if kind != "serve" && kind != "serving" {
		return ""
	}
	baseURL := opt(args, "base-url")
	if baseURL == "" {
		baseURL = "http://localhost:" + firstNonEmpty(opt(args, "port"), "30000")
	}
	cmd := []string{shellQuote(firstNonEmpty(opt(args, "python-bin"), "python3")), "-m", "sglang.bench_serving"}
	appendShellArg(&cmd, "--backend", firstNonEmpty(opt(args, "benchmark-backend"), "sglang"))
	appendShellArg(&cmd, "--model", firstNonEmpty(opt(args, "served-model"), opt(args, "model-name"), modelPath))
	appendShellArg(&cmd, "--base-url", baseURL)
	appendShellArg(&cmd, "--dataset-name", firstNonEmpty(opt(args, "dataset-name"), "random"))
	appendShellArg(&cmd, "--dataset-path", opt(args, "dataset-path"))
	appendShellArg(&cmd, "--random-input-len", firstNonEmpty(opt(args, "input-len"), opt(args, "prompt-tokens"), "512"))
	appendShellArg(&cmd, "--random-output-len", firstNonEmpty(opt(args, "output-len"), opt(args, "output-tokens"), "128"))
	appendShellArg(&cmd, "--num-prompts", firstNonEmpty(opt(args, "num-prompts"), "100"))
	appendShellArg(&cmd, "--request-rate", opt(args, "request-rate"))
	appendShellArg(&cmd, "--max-concurrency", opt(args, "max-concurrency"))
	if outputPath := benchmarkOutputPath(args); outputPath != "" {
		cmd = append(cmd, "--output-file", shellQuote(outputPath))
	}
	appendExtraArgs(&cmd, opt(args, "extra-bench-args"))
	return strings.Join(cmd, " ")
}

func shellQuote(value string) string {
	if value == "" {
		return `""`
	}
	if !strings.ContainsAny(value, " \t\"'") {
		return value
	}
	if runtime.GOOS == "windows" {
		return "\"" + strings.ReplaceAll(value, "\"", "\\\"") + "\""
	}
	return "'" + strings.ReplaceAll(value, "'", "'\\''") + "'"
}

func benchmarkMode(args cliArgs) string {
	mode := strings.ToLower(firstNonEmpty(opt(args, "mode"), opt(args, "benchmark-mode")))
	switch mode {
	case "remote", "endpoint", "api", "openai":
		return "remote"
	case "local", "host", "command", "llama-bench":
		return "local"
	}
	if opt(args, "base-url") != "" && opt(args, "command") == "" && opt(args, "results") == "" {
		return "remote"
	}
	return "local"
}

func measureOpenAIEndpoint(args cliArgs, hfID string) (map[string]any, error) {
	baseURL, err := requireOpt(args, "base-url")
	if err != nil {
		return nil, err
	}
	baseURL = openAIBaseURL(baseURL)
	servedModel := firstNonEmpty(opt(args, "served-model"), opt(args, "model-name"))
	servedModelSource := "explicit"
	var servedModelInfo map[string]any
	if servedModel == "" {
		detected, info, err := detectServedModel(baseURL, opt(args, "model-api-key"), hfID)
		if err == nil && detected != "" {
			servedModel = detected
			servedModelInfo = info
			servedModelSource = "v1_models"
			printStatus(args, "served_model_detected", map[string]any{"servedModel": servedModel, "source": servedModelSource})
		} else {
			servedModel = hfID
			servedModelSource = "hf_id_fallback"
			printStatus(args, "served_model_fallback", map[string]any{"servedModel": servedModel, "source": servedModelSource, "reason": errString(err)})
		}
	} else if _, info, err := detectServedModel(baseURL, opt(args, "model-api-key"), servedModel); err == nil {
		servedModelInfo = info
	}
	prompt := firstNonEmpty(opt(args, "prompt"), "Explain why local inference benchmarks should report prompt prefill throughput, decode throughput, and time to first token.")
	maxTokens := 256
	if value := opt(args, "max-tokens"); value != "" {
		parsed, err := strconv.Atoi(value)
		if err != nil || parsed <= 0 {
			return nil, cliError{"invalid_option", "--max-tokens must be a positive integer", []string{"Pass --max-tokens <number>."}, nil}
		}
		maxTokens = parsed
	}
	temperature := 0.0
	if value := opt(args, "temperature"); value != "" {
		parsed, err := strconv.ParseFloat(value, 64)
		if err != nil {
			return nil, cliError{"invalid_option", "--temperature must be a number", []string{"Pass --temperature <number>."}, nil}
		}
		temperature = parsed
	}
	stream := !hasFlag(args, "no-stream")
	body := map[string]any{
		"model":       servedModel,
		"messages":    []any{map[string]any{"role": "user", "content": prompt}},
		"max_tokens":  maxTokens,
		"temperature": temperature,
		"stream":      stream,
	}
	if stream {
		body["stream_options"] = map[string]any{"include_usage": true}
	}
	quantizationResolution := remoteQuantizationResolution(args, baseURL, opt(args, "model-api-key"), opt(args, "quantization"), servedModelInfo)
	modelResolution := remoteModelResolution(args, servedModel, servedModelSource, hfID, stringValue(quantizationResolution["modelPath"]))
	timeout, err := endpointTimeout(args)
	if err != nil {
		return nil, err
	}
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	bodyData, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	started := time.Now()
	req, err := http.NewRequestWithContext(ctx, "POST", baseURL+"/v1/chat/completions", bytes.NewReader(bodyData))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if key := opt(args, "model-api-key"); key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, cliError{"endpoint_benchmark_failed", fmt.Sprintf("Could not reach OpenAI-compatible endpoint: %v", err), []string{"Check --base-url and confirm the endpoint is reachable from this machine."}, nil}
	}
	defer res.Body.Close()
	printStatus(args, "endpoint_response_received", map[string]any{"status": res.StatusCode, "stream": stream})
	if res.StatusCode < 200 || res.StatusCode >= 300 {
		text, _ := io.ReadAll(res.Body)
		return nil, cliError{"endpoint_benchmark_failed", fmt.Sprintf("OpenAI-compatible endpoint returned %s", res.Status), []string{"Check --base-url, --served-model, and --model-api-key.", "Confirm the endpoint supports POST /v1/chat/completions."}, string(text)}
	}

	var firstTokenAt time.Time
	completedAt := started
	outputText := ""
	var usage map[string]any
	if stream {
		printStatus(args, "endpoint_stream_started", map[string]any{"baseUrl": baseURL})
		streamResult, err := readOpenAIStream(args, res.Body, started)
		if err != nil {
			return nil, err
		}
		firstTokenAt = streamResult.firstTokenAt
		completedAt = streamResult.completedAt
		outputText = streamResult.outputText
		usage = streamResult.usage
		printStatus(args, "endpoint_stream_complete", map[string]any{"outputChars": len(outputText), "usageReturned": usage != nil})
	} else {
		var response map[string]any
		if err := json.NewDecoder(res.Body).Decode(&response); err != nil {
			return nil, err
		}
		outputText = nonStreamingContent(response)
		if obj := asObject(response["usage"]); obj != nil {
			usage = obj
		}
		completedAt = time.Now()
		printStatus(args, "endpoint_completion_received", map[string]any{"outputChars": len(outputText), "usageReturned": usage != nil})
	}

	revision := firstNonEmpty(opt(args, "model-revision"), "main")
	promptTokenResult, err := tokenCount(args, hfID, revision, prompt, usageToken(usage, "prompt_tokens"), "prompt")
	if err != nil {
		return nil, err
	}
	outputTokenResult, err := tokenCount(args, hfID, revision, outputText, firstNonZero(usageToken(usage, "completion_tokens"), usageToken(usage, "output_tokens")), "output")
	if err != nil {
		return nil, err
	}
	promptTokens := promptTokenResult.Count
	outputTokens := outputTokenResult.Count
	printStatus(args, "token_count_source", map[string]any{"prompt": promptTokenResult.Source, "output": outputTokenResult.Source, "promptTokens": promptTokens, "outputTokens": outputTokens})
	totalMs := maxDurationMS(completedAt.Sub(started))
	generationStart := started
	if !firstTokenAt.IsZero() {
		generationStart = firstTokenAt
	}
	generationMs := maxDurationMS(completedAt.Sub(generationStart))
	metrics := map[string]any{
		"prompt":       prompt,
		"outputText":   outputText,
		"promptTokens": float64(promptTokens),
		"outputTokens": float64(outputTokens),
		"tokSOut":      round1(float64(outputTokens) / (generationMs / 1000)),
		"tokSTotal":    round1(float64(promptTokens+outputTokens) / (totalMs / 1000)),
		"engineFlags":  map[string]any{"mode": "remote", "baseUrl": baseURL, "servedModel": servedModel, "servedModelSource": servedModelSource, "stream": stream, "maxTokens": maxTokens, "timeoutSeconds": int(timeout.Seconds())},
		"tokenSources": map[string]any{"prompt": promptTokenResult.Source, "output": outputTokenResult.Source},
		"timingSource": "client_observed_http",
		"metricSource": "remote_endpoint",
		"ttftSource":   map[bool]string{true: "stream_first_token", false: "unavailable_no_stream"}[!firstTokenAt.IsZero()],
	}
	if modelResolution != nil {
		metrics["modelResolution"] = modelResolution
	}
	if quantizationResolution != nil {
		metrics["quantizationResolution"] = quantizationResolution
	}
	if !firstTokenAt.IsZero() {
		ttftMs := float64(firstTokenAt.Sub(started).Milliseconds())
		metrics["ttftMs"] = ttftMs
		if ttftMs > 0 {
			metrics["tokSPrefill"] = round1(float64(promptTokens) / (ttftMs / 1000))
			metrics["tokSPrefillSource"] = "estimated_from_ttft"
		}
	}
	return metrics, nil
}

func measureOllamaEndpoint(args cliArgs, hfID string) (map[string]any, error) {
	baseURL := ollamaBaseURL(args)
	servedModel := firstNonEmpty(opt(args, "served-model"), opt(args, "model-name"), hfID)
	prompt := firstNonEmpty(opt(args, "prompt"), "Explain why local inference benchmarks should report prompt prefill throughput, decode throughput, and time to first token.")
	maxTokens := 256
	if value := opt(args, "max-tokens"); value != "" {
		parsed, err := strconv.Atoi(value)
		if err != nil || parsed <= 0 {
			return nil, cliError{"invalid_option", "--max-tokens must be a positive integer", []string{"Pass --max-tokens <number>."}, nil}
		}
		maxTokens = parsed
	}
	temperature := 0.0
	if value := opt(args, "temperature"); value != "" {
		parsed, err := strconv.ParseFloat(value, 64)
		if err != nil {
			return nil, cliError{"invalid_option", "--temperature must be a number", []string{"Pass --temperature <number>."}, nil}
		}
		temperature = parsed
	}
	timeout, err := endpointTimeout(args)
	if err != nil {
		return nil, err
	}
	body := map[string]any{
		"model":  servedModel,
		"prompt": prompt,
		"stream": false,
		"options": map[string]any{
			"num_predict": maxTokens,
			"temperature": temperature,
		},
	}
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	bodyData, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	started := time.Now()
	req, err := http.NewRequestWithContext(ctx, "POST", baseURL+"/api/generate", bytes.NewReader(bodyData))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if key := opt(args, "model-api-key"); key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, cliError{"endpoint_benchmark_failed", fmt.Sprintf("Could not reach Ollama endpoint: %v", err), []string{"Check --base-url and confirm Ollama is serving from this machine."}, nil}
	}
	defer res.Body.Close()
	printStatus(args, "ollama_response_received", map[string]any{"status": res.StatusCode})
	if res.StatusCode < 200 || res.StatusCode >= 300 {
		text, _ := io.ReadAll(res.Body)
		return nil, cliError{"endpoint_benchmark_failed", fmt.Sprintf("Ollama endpoint returned %s", res.Status), []string{"Check --base-url, --served-model, and --model-api-key.", "Confirm the endpoint supports POST /api/generate."}, string(text)}
	}
	var response map[string]any
	if err := json.NewDecoder(res.Body).Decode(&response); err != nil {
		return nil, err
	}
	completedAt := time.Now()
	promptTokens := firstPositiveNumber(response, "prompt_eval_count", "promptEvalCount")
	outputTokens := firstPositiveNumber(response, "eval_count", "evalCount")
	promptSeconds := ollamaDurationSeconds(firstPositiveNumber(response, "prompt_eval_duration", "promptEvalDuration"))
	decodeSeconds := ollamaDurationSeconds(firstPositiveNumber(response, "eval_duration", "evalDuration"))
	totalSeconds := ollamaDurationSeconds(firstPositiveNumber(response, "total_duration", "totalDuration"))
	if totalSeconds == 0 {
		totalSeconds = completedAt.Sub(started).Seconds()
	}
	metrics := map[string]any{
		"prompt":       prompt,
		"outputText":   stringValue(response["response"]),
		"engineFlags":  map[string]any{"mode": "remote", "baseUrl": baseURL, "servedModel": servedModel, "nativeApi": "ollama_generate", "maxTokens": maxTokens, "timeoutSeconds": int(timeout.Seconds())},
		"tokenSources": map[string]any{"prompt": "ollama_prompt_eval_count", "output": "ollama_eval_count"},
		"timingSource": "ollama_native_api",
		"metricSource": "remote_endpoint",
		"ttftSource":   "unavailable_ollama_nonstreaming",
	}
	if promptTokens > 0 {
		metrics["promptTokens"] = promptTokens
	}
	if outputTokens > 0 {
		metrics["outputTokens"] = outputTokens
	}
	if promptTokens > 0 && promptSeconds > 0 {
		metrics["tokSPrefill"] = round1(promptTokens / promptSeconds)
	}
	if outputTokens > 0 && decodeSeconds > 0 {
		metrics["tokSOut"] = round1(outputTokens / decodeSeconds)
	}
	if promptTokens+outputTokens > 0 && totalSeconds > 0 {
		metrics["tokSTotal"] = round1((promptTokens + outputTokens) / totalSeconds)
	}
	if promptSeconds > 0 {
		metrics["ttftMs"] = round1(promptSeconds * 1000)
		metrics["ttftSource"] = "ollama_prompt_eval_duration"
	}
	return metrics, nil
}

func ollamaBaseURL(args cliArgs) string {
	return openAIBaseURL(firstNonEmpty(opt(args, "base-url"), "http://localhost:11434"))
}

func firstPositiveNumber(obj map[string]any, keys ...string) float64 {
	for _, key := range keys {
		if value := numberField(obj, key); value > 0 {
			return value
		}
	}
	return 0
}

func ollamaDurationSeconds(value float64) float64 {
	if value <= 0 {
		return 0
	}
	return value / 1e9
}

func endpointTimeout(args cliArgs) (time.Duration, error) {
	value := firstNonEmpty(opt(args, "endpoint-timeout-seconds"), opt(args, "timeout-seconds"))
	if value == "" {
		return defaultEndpointTimeout, nil
	}
	seconds, err := strconv.Atoi(value)
	if err != nil || seconds <= 0 {
		return 0, cliError{"invalid_option", "--endpoint-timeout-seconds must be a positive integer", []string{"Pass --endpoint-timeout-seconds <seconds>."}, nil}
	}
	return time.Duration(seconds) * time.Second, nil
}

func remoteModelResolution(args cliArgs, servedModel, servedModelSource, hfID, modelPath string) map[string]any {
	if servedModel == "" || hfID == "" || (modelPath == "" && strings.EqualFold(servedModel, hfID)) {
		return nil
	}
	status := "alias"
	if strings.EqualFold(servedModel, hfID) {
		status = "matched"
	}
	resolution := map[string]any{"hfId": hfID, "servedModel": servedModel, "servedModelSource": servedModelSource, "status": status}
	if modelPath != "" {
		resolution["declaredBaseModel"] = hfID
		resolution["loadedFilename"] = filepath.Base(modelPath)
	}
	if status == "alias" {
		printStatus(args, "remote_model_alias", map[string]any{"hfId": hfID, "servedModel": servedModel, "hint": "Endpoint model names are often server aliases; verify --hf-id only if the underlying model is different."})
	}
	queryCommand := "lmx model search " + shellQuote(servedModel)
	value, err := searchModels(args, servedModel, 5)
	if err != nil {
		resolution["searchError"] = err.Error()
		resolution["searchCommand"] = queryCommand
		printStatus(args, "hf_id_search_unavailable", map[string]any{"query": servedModel, "next": queryCommand})
		return resolution
	}
	candidates := modelCandidates(value, 5)
	resolution["candidates"] = candidates
	resolution["searchCommand"] = queryCommand
	if len(candidates) == 0 {
		printStatus(args, "hf_id_candidates_empty", map[string]any{"query": servedModel, "next": queryCommand})
		return resolution
	}
	fields := map[string]any{"query": servedModel, "count": len(candidates), "next": "If the exact GGUF repo matters, rerun with that --hf-id."}
	for i, candidate := range candidates {
		if obj := asObject(candidate); obj != nil {
			fields[fmt.Sprintf("candidate%d", i+1)] = firstNonEmpty(stringValue(obj["hfId"]), stringValue(obj["id"]), stringValue(obj["modelId"]))
		}
	}
	if filename := stringValue(resolution["loadedFilename"]); filename != "" {
		match, err := sourceRepoFromFilename(args, candidates, filename)
		if err != nil {
			resolution["sourceRepoSearchError"] = err.Error()
		} else if match != "" {
			resolution["sourceRepo"] = match
			resolution["sourceRepoMatch"] = "exact_filename"
			resolution["status"] = "source_repo_detected"
			fields["sourceRepo"] = match
			fields["sourceRepoMatch"] = "exact_filename"
		}
	}
	printStatus(args, "hf_id_candidates_found", fields)
	return resolution
}

func sourceRepoFromFilename(args cliArgs, candidates []any, filename string) (string, error) {
	var firstErr error
	for _, repo := range filenameDerivedSourceRepos(filename) {
		matched, err := hfRepoContainsFilename(args, repo, filename)
		if err != nil {
			if firstErr == nil {
				firstErr = err
			}
			continue
		}
		if matched {
			return repo, nil
		}
	}
	for _, candidate := range candidates {
		repo := candidateRepoID(candidate)
		if repo == "" {
			continue
		}
		matched, err := hfRepoContainsFilename(args, repo, filename)
		if err != nil {
			if firstErr == nil {
				firstErr = err
			}
			continue
		}
		if matched {
			return repo, nil
		}
	}
	return "", firstErr
}

func filenameDerivedSourceRepos(filename string) []string {
	name := strings.TrimSuffix(filepath.Base(filename), filepath.Ext(filename))
	lower := strings.ToLower(name)
	if !strings.Contains(lower, "-qat-") {
		return nil
	}
	if idx := strings.Index(lower, "-ud-"); idx > 0 {
		return []string{"unsloth/" + name[:idx] + "-GGUF"}
	}
	return nil
}

func candidateRepoID(candidate any) string {
	if obj := asObject(candidate); obj != nil {
		return firstNonEmpty(stringValue(obj["hfId"]), stringValue(obj["id"]), stringValue(obj["modelId"]))
	}
	return stringValue(candidate)
}

func hfRepoContainsFilename(args cliArgs, repo, filename string) (bool, error) {
	body, err := fetchEndpointJSON(strings.TrimRight(hfAPIURL(args), "/")+"/api/models/"+hfRepoPath(repo), "")
	if err != nil {
		return false, err
	}
	obj := asObject(body)
	if obj == nil {
		return false, nil
	}
	normalizedFilename := normalizedModelFilename(filename)
	for _, sibling := range modelFileItems(obj) {
		file := firstNonEmpty(stringValue(sibling["rfilename"]), stringValue(sibling["filename"]), stringValue(sibling["path"]))
		if file == filename || strings.EqualFold(file, filename) || normalizedModelFilename(file) == normalizedFilename {
			return true, nil
		}
	}
	return false, nil
}

func normalizedModelFilename(filename string) string {
	name := strings.ToLower(filepath.Base(filename))
	name = strings.TrimSuffix(name, filepath.Ext(name))
	parts := strings.FieldsFunc(name, func(r rune) bool { return r == '-' || r == '_' || r == '.' })
	kept := parts[:0]
	for _, part := range parts {
		if part != "qat" {
			kept = append(kept, part)
		}
	}
	return strings.Join(kept, "-")
}

func modelFileItems(obj map[string]any) []map[string]any {
	items := []map[string]any{}
	for _, key := range []string{"siblings", "files"} {
		arr, _ := obj[key].([]any)
		for _, item := range arr {
			if file := asObject(item); file != nil {
				items = append(items, file)
			}
		}
	}
	return items
}

func hfRepoPath(repo string) string {
	parts := strings.Split(repo, "/")
	for i, part := range parts {
		parts[i] = url.PathEscape(part)
	}
	return strings.Join(parts, "/")
}

func hfAPIURL(args cliArgs) string {
	if value := opt(args, "hf-api-url"); value != "" {
		return value
	}
	return defaultHFAPIURL
}

func searchModels(args cliArgs, query string, limit int) (any, error) {
	endpoint := apiURL(args) + "/api/models/search?q=" + url.QueryEscape(query) + "&limit=" + url.QueryEscape(strconv.Itoa(limit))
	return fetchJSON("GET", endpoint, "", nil)
}

func modelCandidates(value any, limit int) []any {
	var items []any
	if arr, ok := value.([]any); ok {
		items = arr
	} else if obj := asObject(value); obj != nil {
		for _, key := range []string{"models", "results", "data"} {
			if arr, ok := obj[key].([]any); ok {
				items = arr
				break
			}
		}
	}
	if len(items) > limit {
		return items[:limit]
	}
	return items
}

func remoteQuantizationResolution(args cliArgs, baseURL, apiKey, cliQuantization string, servedModelInfo map[string]any) map[string]any {
	resolution := map[string]any{"cli": cliQuantization}
	if quant := quantizationFromModelInfo(servedModelInfo); quant != "" {
		resolution["v1Models"] = quant
	}
	if props, err := fetchEndpointJSON(baseURL+"/props", apiKey); err == nil {
		if obj := asObject(props); obj != nil {
			modelPath := stringValue(obj["model_path"])
			if modelPath != "" {
				resolution["modelPath"] = modelPath
				if quant := quantizationFromFilename(modelPath); quant != "" {
					resolution["filename"] = quant
				}
			}
		}
	}
	trusted := firstNonEmpty(stringValue(resolution["filename"]), stringValue(resolution["v1Models"]), cliQuantization)
	if trusted == "" {
		return nil
	}
	resolution["trusted"] = trusted
	resolution["trustedSource"] = map[bool]string{true: "filename", false: "v1_models"}[stringValue(resolution["filename"]) != ""]
	if stringValue(resolution["filename"]) == "" && stringValue(resolution["v1Models"]) == "" {
		resolution["trustedSource"] = "cli"
	}
	mismatches := quantizationMismatches(resolution, cliQuantization)
	if len(mismatches) == 0 {
		resolution["status"] = "matched"
		if stringValue(resolution["filename"]) != "" || stringValue(resolution["v1Models"]) != "" {
			printStatus(args, "remote_quantization_detected", map[string]any{"quantization": trusted, "source": resolution["trustedSource"]})
		}
		return resolution
	}
	resolution["status"] = "mismatch"
	resolution["mismatches"] = mismatches
	printStatus(args, "remote_quantization_mismatch", map[string]any{"cli": cliQuantization, "filename": resolution["filename"], "v1Models": resolution["v1Models"], "trusted": trusted, "hint": "Filename-derived quantization is trusted for llama.cpp endpoints; rerun with matching --quantization before submitting."})
	return resolution
}

func quantizationMismatches(resolution map[string]any, cliQuantization string) []any {
	sources := map[string]string{"cli": cliQuantization, "filename": stringValue(resolution["filename"]), "v1Models": stringValue(resolution["v1Models"])}
	mismatches := []any{}
	keys := []string{"cli", "filename", "v1Models"}
	for i := 0; i < len(keys); i++ {
		for j := i + 1; j < len(keys); j++ {
			left, right := sources[keys[i]], sources[keys[j]]
			if left == "" || right == "" || quantizationEqual(left, right) {
				continue
			}
			mismatches = append(mismatches, map[string]any{"leftSource": keys[i], "left": left, "rightSource": keys[j], "right": right})
		}
	}
	return mismatches
}

func quantizationEqual(left, right string) bool {
	return strings.EqualFold(strings.TrimSpace(left), strings.TrimSpace(right))
}

func quantizationFromModelInfo(info map[string]any) string {
	if info == nil {
		return ""
	}
	if value := firstNonEmpty(stringValue(info["quantization"]), stringValue(info["quantization_level"])); value != "" {
		return value
	}
	if details := asObject(info["details"]); details != nil {
		if value := firstNonEmpty(stringValue(details["quantization_level"]), stringValue(details["quantization"])); value != "" {
			return value
		}
	}
	if meta := asObject(info["meta"]); meta != nil {
		return firstNonEmpty(stringValue(meta["quantization"]), stringValue(meta["quantization_level"]))
	}
	return ""
}

func quantizationFromFilename(path string) string {
	name := filepath.Base(path)
	name = strings.TrimSuffix(name, filepath.Ext(name))
	pattern := regexp.MustCompile(`(?i)(?:^|[-_.])((?:IQ|Q)[0-9][A-Z0-9_]*|(?:BF|F|FP)[0-9]+)(?:[-_.]|$)`)
	matches := pattern.FindAllStringSubmatch(name, -1)
	if len(matches) == 0 {
		return ""
	}
	return strings.ToUpper(matches[len(matches)-1][1])
}

func fetchEndpointJSON(rawURL, apiKey string) (any, error) {
	req, err := http.NewRequest("GET", rawURL, nil)
	if err != nil {
		return nil, err
	}
	if apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if res.StatusCode < 200 || res.StatusCode >= 300 {
		return nil, fmt.Errorf("%s returned %s", rawURL, res.Status)
	}
	var body any
	if err := json.NewDecoder(res.Body).Decode(&body); err != nil {
		return nil, err
	}
	return body, nil
}

func detectServedModel(baseURL, apiKey, preferred string) (string, map[string]any, error) {
	req, err := http.NewRequest("GET", openAIBaseURL(baseURL)+"/v1/models", nil)
	if err != nil {
		return "", nil, err
	}
	if apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", nil, err
	}
	defer res.Body.Close()
	if res.StatusCode < 200 || res.StatusCode >= 300 {
		return "", nil, fmt.Errorf("/v1/models returned %s", res.Status)
	}
	var body map[string]any
	if err := json.NewDecoder(res.Body).Decode(&body); err != nil {
		return "", nil, err
	}
	data := modelInfoItems(body)
	first := ""
	var firstInfo map[string]any
	for _, item := range data {
		obj := asObject(item)
		if obj == nil {
			continue
		}
		id := firstNonEmpty(stringValue(obj["id"]), stringValue(obj["name"]), stringValue(obj["model"]))
		if id == "" {
			continue
		}
		if first == "" {
			first = id
			firstInfo = obj
		}
		if id == preferred || strings.EqualFold(id, preferred) {
			return id, obj, nil
		}
	}
	if first != "" {
		return first, firstInfo, nil
	}
	return "", nil, errors.New("/v1/models did not return any model ids")
}

func modelInfoItems(body map[string]any) []any {
	items := []any{}
	for _, key := range []string{"data", "models"} {
		if arr, ok := body[key].([]any); ok {
			items = append(items, arr...)
		}
	}
	return items
}

type openAIStreamResult struct {
	firstTokenAt time.Time
	completedAt  time.Time
	outputText   string
	usage        map[string]any
}

func readOpenAIStream(args cliArgs, body io.Reader, started time.Time) (openAIStreamResult, error) {
	result := openAIStreamResult{}
	buffer := ""
	chunk := make([]byte, 8192)
	for {
		n, err := body.Read(chunk)
		if n > 0 {
			buffer += string(chunk[:n])
			lines := strings.Split(buffer, "\n")
			buffer = lines[len(lines)-1]
			for _, line := range lines[:len(lines)-1] {
				consumeOpenAIStreamLine(args, strings.TrimSpace(line), started, &result)
			}
		}
		if err == io.EOF {
			break
		}
		if err != nil {
			return openAIStreamResult{}, err
		}
	}
	if strings.TrimSpace(buffer) != "" {
		consumeOpenAIStreamLine(args, strings.TrimSpace(buffer), started, &result)
	}
	result.completedAt = time.Now()
	return result, nil
}

func consumeOpenAIStreamLine(args cliArgs, line string, started time.Time, result *openAIStreamResult) {
	if !strings.HasPrefix(line, "data:") {
		return
	}
	payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
	if payload == "" || payload == "[DONE]" {
		return
	}
	var chunk map[string]any
	if json.Unmarshal([]byte(payload), &chunk) != nil {
		return
	}
	if obj := asObject(chunk["usage"]); obj != nil {
		result.usage = obj
	}
	content := streamingContent(chunk)
	if content == "" {
		return
	}
	if result.firstTokenAt.IsZero() {
		result.firstTokenAt = time.Now()
		printStatus(args, "first_token_received", map[string]any{"ttftMs": result.firstTokenAt.Sub(started).Milliseconds()})
	}
	result.outputText += content
}

func streamingContent(chunk map[string]any) string {
	choices, _ := chunk["choices"].([]any)
	if len(choices) == 0 {
		return ""
	}
	first := asObject(choices[0])
	if first == nil {
		return ""
	}
	delta := asObject(first["delta"])
	if delta == nil {
		return ""
	}
	return firstNonEmpty(stringValue(delta["content"]), stringValue(delta["reasoning_content"]))
}

func nonStreamingContent(response map[string]any) string {
	choices, _ := response["choices"].([]any)
	if len(choices) == 0 {
		return ""
	}
	first := asObject(choices[0])
	if first == nil {
		return ""
	}
	message := asObject(first["message"])
	if message == nil {
		return ""
	}
	return firstNonEmpty(stringValue(message["content"]), stringValue(message["reasoning_content"]))
}

func tokenCount(args cliArgs, hfID, revision, text string, known int, kind string) (tokenCountResult, error) {
	flag := kind + "-tokens"
	if kind == "prompt" {
		flag = "prompt-tokens"
	}
	if value := opt(args, flag); value != "" {
		parsed, err := strconv.Atoi(value)
		if err != nil || parsed <= 0 {
			return tokenCountResult{}, cliError{"invalid_option", "--" + flag + " must be a positive integer", []string{"Pass --" + flag + " <number>."}, nil}
		}
		return tokenCountResult{Count: parsed, Source: "explicit_flag"}, nil
	}
	if known > 0 {
		return tokenCountResult{Count: known, Source: "endpoint_usage"}, nil
	}
	count, err := pythonTokenCount(hfID, revision, text)
	if err == nil && count > 0 {
		return tokenCountResult{Count: count, Source: "python_transformers"}, nil
	}
	return tokenCountResult{}, cliError{"token_count_missing", fmt.Sprintf("Could not determine %s token count.", kind), []string{fmt.Sprintf("Pass --%s <n> from the endpoint usage or benchmark output.", flag), "Or install Python transformers so the optional tokenizer helper can count tokens."}, errString(err)}
}

func pythonTokenCount(model, revision, text string) (int, error) {
	script := filepath.Join("python", "localmaxxing_helpers", "token_count.py")
	request := map[string]any{"model": model, "revision": revision, "text": text}
	data, _ := json.Marshal(request)
	cmd := exec.Command("python", script)
	cmd.Stdin = bytes.NewReader(data)
	var stdout bytes.Buffer
	var stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return 0, fmt.Errorf("%v: %s", err, strings.TrimSpace(stderr.String()))
	}
	var response map[string]any
	if err := json.Unmarshal(stdout.Bytes(), &response); err != nil {
		return 0, err
	}
	if tokens, ok := response["tokens"].(float64); ok {
		return int(tokens), nil
	}
	return 0, errors.New("token helper did not return tokens")
}

func usageToken(usage map[string]any, key string) int {
	if usage == nil {
		return 0
	}
	switch value := usage[key].(type) {
	case float64:
		return int(value)
	case int:
		return value
	default:
		return 0
	}
}

func firstNonZero(values ...int) int {
	for _, value := range values {
		if value > 0 {
			return value
		}
	}
	return 0
}

func maxDurationMS(duration time.Duration) float64 {
	ms := float64(duration.Milliseconds())
	if ms < 1 {
		return 1
	}
	return ms
}

func stringValue(value any) string {
	if text, ok := value.(string); ok {
		return text
	}
	return ""
}

func errString(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}

func normalizeEngineName(value string) string {
	raw := strings.ToLower(strings.TrimSpace(value))
	switch raw {
	case "llama", "llama.cpp", "llamacpp", "llama-bench", "llama.cpp-bench":
		return "llama.cpp"
	case "vllm", "vllm serve", "vllm-bench":
		return "vllm"
	case "sglang", "sgl", "sglang-bench":
		return "sglang"
	case "":
		return ""
	default:
		return value
	}
}

func runBenchmarkCommand(commandSnippet string) (string, error) {
	var cmd *exec.Cmd
	if runtime.GOOS == "windows" {
		cmd = exec.Command("cmd", "/C", commandSnippet)
	} else {
		cmd = exec.Command("sh", "-c", commandSnippet)
	}
	var stdout bytes.Buffer
	var stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		output := strings.TrimSpace(strings.Join([]string{stdout.String(), stderr.String()}, "\n"))
		return "", cliError{"benchmark_command_failed", "Benchmark command failed.", []string{"Check that the benchmark executable is installed and available on PATH.", "For llama.cpp, pass a complete llama-bench command with --command.", "For vLLM/SGLang, prefer their JSON output if available, then pass --results <path>."}, firstNonEmpty(output, err.Error())}
	}
	return strings.TrimSpace(strings.Join([]string{stdout.String(), stderr.String()}, "\n")), nil
}

func parseBenchmarkOutput(text string) map[string]float64 {
	metrics := parseLlamaBenchTable(text)
	for key, value := range parseBenchmarkJSONMetrics(text) {
		if metrics[key] == 0 {
			metrics[key] = value
		}
	}
	if value, ok := firstRegexNumber(text, []string{
		`(?i)output\s+token\s+throughput[^\d]*(\d+(?:\.\d+)?)`,
		`(?i)output\s+throughput[^\d]*(\d+(?:\.\d+)?)[^\n]*(?:tok|token)/s`,
		`(?i)decode\s+throughput[^\d]*(\d+(?:\.\d+)?)[^\n]*(?:tok|token)/s`,
		`(?i)generation\s+throughput[^\d]*(\d+(?:\.\d+)?)[^\n]*(?:tok|token)/s`,
		`(?i)generated\s+tokens\s+per\s+second[^\d]*(\d+(?:\.\d+)?)`,
		`(?i)(\d+(?:\.\d+)?)\s+output\s+tokens/s`,
	}); ok && metrics["tokSOut"] == 0 {
		metrics["tokSOut"] = value
	}
	if value, ok := firstRegexNumber(text, []string{
		`(?i)prefill\s+throughput[^\d]*(\d+(?:\.\d+)?)[^\n]*(?:tok|token)/s`,
		`(?i)prompt\s+throughput[^\d]*(\d+(?:\.\d+)?)[^\n]*(?:tok|token)/s`,
		`(?i)input\s+token\s+throughput[^\d]*(\d+(?:\.\d+)?)`,
	}); ok && metrics["tokSPrefill"] == 0 {
		metrics["tokSPrefill"] = value
	}
	if value, ok := firstRegexNumber(text, []string{
		`(?i)total\s+token\s+throughput[^\d]*(\d+(?:\.\d+)?)`,
		`(?i)total\s+throughput[^\d]*(\d+(?:\.\d+)?)[^\n]*(?:tok|token)/s`,
		`(?i)(\d+(?:\.\d+)?)\s+total\s+tokens/s`,
	}); ok && metrics["tokSTotal"] == 0 {
		metrics["tokSTotal"] = value
	}
	if value, ok := firstRegexNumber(text, []string{`(?i)(?:mean|median)?\s*ttft[^\d]*(\d+(?:\.\d+)?)[^\n]*ms`, `(?i)time\s+to\s+first\s+token[^\d]*(\d+(?:\.\d+)?)[^\n]*ms`}); ok && metrics["ttftMs"] == 0 {
		metrics["ttftMs"] = value
	}
	if value, ok := firstRegexNumber(text, []string{`(?i)peak\s+(?:gpu\s+)?(?:vram|memory)[^\d]*(\d+(?:\.\d+)?)[^\n]*gb`}); ok && metrics["peakVramGb"] == 0 {
		metrics["peakVramGb"] = value
	}
	if value, ok := firstRegexNumber(text, []string{`(?i)total\s+(?:input|prompt)\s+tokens[^\d]*(\d+)`, `(?i)total\s+num\s+prompt\s+tokens[^\d]*(\d+)`}); ok && metrics["promptTokens"] == 0 {
		metrics["promptTokens"] = value
	}
	if value, ok := firstRegexNumber(text, []string{`(?i)total\s+(?:generated|output)\s+tokens[^\d]*(\d+)`, `(?i)total\s+num\s+output\s+tokens[^\d]*(\d+)`}); ok && metrics["outputTokens"] == 0 {
		metrics["outputTokens"] = value
	}
	return compactMetrics(metrics)
}

func parseBenchmarkJSONMetrics(text string) map[string]float64 {
	var value any
	trimmed := strings.TrimSpace(text)
	if trimmed == "" {
		return map[string]float64{}
	}
	if err := json.Unmarshal([]byte(trimmed), &value); err != nil {
		startArray := strings.Index(trimmed, "[")
		endArray := strings.LastIndex(trimmed, "]")
		startObj := strings.Index(trimmed, "{")
		endObj := strings.LastIndex(trimmed, "}")
		snippet := ""
		if startArray >= 0 && endArray > startArray {
			snippet = trimmed[startArray : endArray+1]
		} else if startObj >= 0 && endObj > startObj {
			snippet = trimmed[startObj : endObj+1]
		}
		if snippet == "" || json.Unmarshal([]byte(snippet), &value) != nil {
			return map[string]float64{}
		}
	}

	aliases := map[string][]string{
		"tokSOut":      []string{"tokSOut", "outputTokensPerSecond", "outputTokenThroughput", "outputThroughput", "generationTokensPerSecond", "decodeThroughput", "genThroughput"},
		"tokSPrefill":  []string{"tokSPrefill", "prefillTokensPerSecond", "promptTokensPerSecond", "inputTokensPerSecond", "prefillThroughput", "promptThroughput"},
		"tokSTotal":    []string{"tokSTotal", "totalTokensPerSecond", "totalTokenThroughput", "totalThroughput", "tokensPerSecond", "requestThroughput"},
		"ttftMs":       []string{"ttftMs", "timeToFirstTokenMs", "meanTtftMs", "medianTtftMs", "meanTTFTMs", "medianTTFTMs", "meanTtft", "medianTtft", "meanTTFT", "medianTTFT"},
		"peakVramGb":   []string{"peakVramGb", "peakGpuMemoryGb", "maxVramGb"},
		"promptTokens": []string{"promptTokens", "inputTokens", "totalInputTokens", "totalPromptTokens", "numPromptTokens"},
		"outputTokens": []string{"outputTokens", "completionTokens", "generatedTokens", "totalOutputTokens", "totalGeneratedTokens", "numOutputTokens"},
	}
	metrics := parseLlamaBenchJSONMetrics(value)
	for key, value := range parseVLLMBenchmarkJSONMetrics(value) {
		if metrics[key] == 0 {
			metrics[key] = value
		}
	}
	for field, names := range aliases {
		if metrics[field] != 0 {
			continue
		}
		if found, ok := jsonNumberByAliases(value, names); ok {
			metrics[field] = found
		}
	}
	return compactMetrics(metrics)
}

func parseLlamaBenchJSONMetrics(value any) map[string]float64 {
	metrics := map[string]float64{}
	for _, row := range jsonObjectRows(value) {
		avgTS, ok := anyNumber(row["avg_ts"])
		if !ok {
			continue
		}
		nPrompt, _ := anyNumber(row["n_prompt"])
		nGen, _ := anyNumber(row["n_gen"])
		nDepth, _ := anyNumber(row["n_depth"])
		if nDepth > 0 && metrics["contextTokens"] == 0 {
			metrics["contextTokens"] = nDepth
		}
		if nPrompt > 0 && nGen == 0 {
			if metrics["tokSPrefill"] == 0 {
				metrics["tokSPrefill"] = avgTS
			}
			if metrics["promptTokens"] == 0 {
				metrics["promptTokens"] = nPrompt
			}
		}
		if nGen > 0 && nPrompt == 0 {
			if metrics["tokSOut"] == 0 {
				metrics["tokSOut"] = avgTS
			}
			if metrics["outputTokens"] == 0 {
				metrics["outputTokens"] = nGen
			}
		}
	}
	return compactMetrics(metrics)
}

func jsonObjectRows(value any) []map[string]any {
	rows := []map[string]any{}
	switch typed := value.(type) {
	case []any:
		for _, item := range typed {
			if obj := asObject(item); obj != nil {
				rows = append(rows, obj)
			}
		}
	case map[string]any:
		rows = append(rows, typed)
	}
	return rows
}

func parseVLLMBenchmarkJSONMetrics(value any) map[string]float64 {
	metrics := map[string]float64{}
	for _, row := range jsonObjectRows(value) {
		inputTokens := firstJSONNumber(row, "input_len", "inputLen", "prompt_len", "promptLen", "num_input_tokens", "numPromptTokens")
		outputTokens := firstJSONNumber(row, "output_len", "outputLen", "generation_len", "generationLen", "num_output_tokens", "numOutputTokens")
		batchSize := firstJSONNumber(row, "batch_size", "batchSize")
		if batchSize == 0 {
			batchSize = 1
		}
		if inputTokens > 0 && metrics["promptTokens"] == 0 {
			metrics["promptTokens"] = inputTokens
		}
		if inputTokens > 0 && metrics["contextTokens"] == 0 {
			metrics["contextTokens"] = inputTokens
		}
		if outputTokens > 0 && metrics["outputTokens"] == 0 {
			metrics["outputTokens"] = outputTokens
		}
		if throughput := firstJSONNumber(row, "output_token_throughput", "outputTokenThroughput", "output_throughput", "generation_throughput", "tokens_per_second"); throughput > 0 && metrics["tokSOut"] == 0 {
			metrics["tokSOut"] = throughput
		}
		if throughput := firstJSONNumber(row, "total_token_throughput", "totalTokenThroughput", "total_throughput"); throughput > 0 && metrics["tokSTotal"] == 0 {
			metrics["tokSTotal"] = throughput
		}
		latencyMs := firstJSONNumber(row, "avg_latency_ms", "mean_latency_ms", "median_latency_ms", "latency_ms")
		if latencyMs == 0 {
			latencyMs = secondsToMs(firstJSONNumber(row, "avg_latency", "mean_latency", "median_latency", "latency", "avg_latency_s", "mean_latency_s", "median_latency_s"))
		}
		if latencyMs > 0 {
			if metrics["latencyMs"] == 0 {
				metrics["latencyMs"] = latencyMs
			}
			if inputTokens+outputTokens > 0 && metrics["tokSTotal"] == 0 {
				metrics["tokSTotal"] = round1(((inputTokens + outputTokens) * batchSize) / (latencyMs / 1000))
			}
			if outputTokens > 0 && metrics["tokSOut"] == 0 {
				metrics["tokSOut"] = round1((outputTokens * batchSize) / (latencyMs / 1000))
			}
		}
	}
	return compactMetrics(metrics)
}

func firstJSONNumber(obj map[string]any, names ...string) float64 {
	for _, name := range names {
		if value, ok := jsonNumberByAliases(obj, []string{name}); ok {
			return value
		}
	}
	return 0
}

func secondsToMs(value float64) float64 {
	if value == 0 {
		return 0
	}
	return round1(value * 1000)
}

func jsonNumberByAliases(value any, aliases []string) (float64, bool) {
	aliasSet := map[string]bool{}
	for _, alias := range aliases {
		aliasSet[normalizeMetricKey(alias)] = true
	}
	var walk func(any) (float64, bool)
	walk = func(current any) (float64, bool) {
		switch typed := current.(type) {
		case map[string]any:
			for key, child := range typed {
				if aliasSet[normalizeMetricKey(key)] {
					if number, ok := anyNumber(child); ok {
						return number, true
					}
				}
			}
			for _, child := range typed {
				if number, ok := walk(child); ok {
					return number, true
				}
			}
		case []any:
			for _, child := range typed {
				if number, ok := walk(child); ok {
					return number, true
				}
			}
		}
		return 0, false
	}
	return walk(value)
}

func normalizeMetricKey(value string) string {
	return strings.ToLower(strings.NewReplacer("_", "", "-", "", " ", "").Replace(value))
}

func anyNumber(value any) (float64, bool) {
	switch typed := value.(type) {
	case float64:
		return typed, typed > 0
	case float32:
		return float64(typed), typed > 0
	case int:
		return float64(typed), typed > 0
	case int64:
		return float64(typed), typed > 0
	case string:
		parsed, err := strconv.ParseFloat(strings.TrimSpace(typed), 64)
		return parsed, err == nil && parsed > 0
	default:
		return 0, false
	}
}

func parseLlamaBenchTable(text string) map[string]float64 {
	metrics := map[string]float64{}
	testPattern := regexp.MustCompile(`(?i)^(pp|tg)\s*(\d+)\b`)
	depthPattern := regexp.MustCompile(`(?i)@\s*d\s*(\d+)`)
	valuePattern := regexp.MustCompile(`^(\d+(?:\.\d+)?)(?:\s*[+-]|\s*±|\s*$)`)
	for _, line := range strings.FieldsFunc(text, func(r rune) bool { return r == '\n' || r == '\r' }) {
		cells := []string{}
		for _, cell := range strings.Split(line, "|") {
			cell = strings.TrimSpace(cell)
			if cell != "" {
				cells = append(cells, cell)
			}
		}
		for index, cell := range cells {
			match := testPattern.FindStringSubmatch(cell)
			if match == nil {
				continue
			}
			if depthMatch := depthPattern.FindStringSubmatch(cell); len(depthMatch) >= 2 && metrics["contextTokens"] == 0 {
				if depth, err := strconv.ParseFloat(depthMatch[1], 64); err == nil {
					metrics["contextTokens"] = depth
				}
			}
			var value float64
			found := false
			for _, valueCell := range cells[index+1:] {
				valueMatch := valuePattern.FindStringSubmatch(valueCell)
				if valueMatch == nil {
					continue
				}
				parsed, err := strconv.ParseFloat(valueMatch[1], 64)
				if err == nil {
					value = parsed
					found = true
					break
				}
			}
			if !found {
				continue
			}
			tokens, _ := strconv.ParseFloat(match[2], 64)
			if strings.EqualFold(match[1], "pp") {
				if metrics["tokSPrefill"] == 0 {
					metrics["tokSPrefill"] = value
				}
				if tokens > 0 && metrics["promptTokens"] == 0 {
					metrics["promptTokens"] = tokens
				}
			}
			if strings.EqualFold(match[1], "tg") {
				if metrics["tokSOut"] == 0 {
					metrics["tokSOut"] = value
				}
				if tokens > 0 && metrics["outputTokens"] == 0 {
					metrics["outputTokens"] = tokens
				}
			}
		}
	}
	return metrics
}

func firstRegexNumber(text string, patterns []string) (float64, bool) {
	for _, pattern := range patterns {
		re := regexp.MustCompile(pattern)
		match := re.FindStringSubmatch(text)
		if len(match) < 2 {
			continue
		}
		value, err := strconv.ParseFloat(match[1], 64)
		if err == nil {
			return value, true
		}
	}
	return 0, false
}

func compactMetrics(metrics map[string]float64) map[string]float64 {
	out := map[string]float64{}
	for key, value := range metrics {
		if value > 0 {
			out[key] = value
		}
	}
	return out
}

func handleExecute(suiteSlug string, args cliArgs) error {
	if suiteSlug == "" {
		return errors.New("eval execute requires a suite slug")
	}
	model, err := requireOpt(args, "model")
	if err != nil {
		return err
	}
	baseURL, err := requireOpt(args, "base-url")
	if err != nil {
		return err
	}
	baseURL = openAIBaseURL(baseURL)
	key := apiKey(args)
	if key == "" {
		return missingAPIKey("--api-key or LMX_API_KEY is required for eval execute")
	}
	payload := map[string]any{"suiteSlug": suiteSlug, "model": model, "baseUrl": baseURL, "autoSubmit": hasFlag(args, "submit"), "quantization": opt(args, "quantization"), "modelRevision": firstNonEmpty(opt(args, "model-revision"), "main"), "notes": opt(args, "notes")}
	if hardwarePath := opt(args, "hardware"); hardwarePath != "" {
		hardware, err := readJSON(hardwarePath)
		if err != nil {
			return err
		}
		payload["hardware"] = hardware
	}
	value, err := fetchJSON("POST", apiURL(args)+"/api/evals/execute", key, payload)
	if err != nil {
		return err
	}
	printJSON(value)
	printInfo("execute_submitted", map[string]any{"suite": suiteSlug, "endpoint": "/api/evals/execute", "autoSubmit": payload["autoSubmit"]})
	return nil
}

func handleLmEval(suiteSlug string, args cliArgs) error {
	if suiteSlug == "" {
		return errors.New("eval lm-eval requires a suite slug")
	}
	model, err := requireOpt(args, "model")
	if err != nil {
		return err
	}
	suite, err := loadSuiteForEvalRun(suiteSlug, args)
	if err != nil {
		return err
	}
	if !strings.EqualFold(stringValue(suite["runner"]), "LM_EVAL_HARNESS") {
		return cliError{"suite_runner_mismatch", fmt.Sprintf("Suite %q is %s, not LM_EVAL_HARNESS", suiteSlug, stringValue(suite["runner"])), []string{"Use lmx eval run for CUSTOM suites.", "Use lmx eval suite list or show to find LM_EVAL_HARNESS suites."}, nil}
	}
	backend := firstNonEmpty(opt(args, "backend"), "hf")
	command := firstNonEmpty(opt(args, "lm-eval-bin"), "lm_eval")
	resultsPath := firstNonEmpty(opt(args, "results"), "localmaxxing-lm-eval-results.json")
	tasks := firstNonEmpty(opt(args, "tasks"), strings.Join(evalTaskKeys(suiteDoc(suite)), ","), suiteSlug)
	modelArgs := opt(args, "model-args")
	if modelArgs == "" && backend == "hf" {
		modelArgs = "pretrained=" + model
	}
	cmdArgs := []string{"--model", backend}
	if modelArgs != "" {
		cmdArgs = append(cmdArgs, "--model_args", modelArgs)
	}
	cmdArgs = append(cmdArgs, "--tasks", tasks)
	if fewshot := firstNonEmpty(opt(args, "num-fewshot"), opt(args, "fewshot"), inferredEvalFewShot(suiteDoc(suite))); fewshot != "" {
		cmdArgs = append(cmdArgs, "--num_fewshot", fewshot)
	}
	cmdArgs = append(cmdArgs, "--output_path", resultsPath)
	printInfo("lm_eval_start", map[string]any{"suite": suiteSlug, "command": command, "backend": backend, "modelArgs": modelArgs, "tasks": tasks, "output": resultsPath})
	cmd := exec.Command(command, cmdArgs...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Stdin = os.Stdin
	if err := cmd.Run(); err != nil {
		return err
	}
	return handleEvalRun(suiteSlug, args)
}

func handleEvalRun(suiteSlug string, args cliArgs) error {
	if suiteSlug == "" {
		return errors.New("eval run requires a suite slug")
	}
	suite, err := loadSuiteForEvalRun(suiteSlug, args)
	if err != nil {
		return err
	}
	doc := suiteDoc(suite)
	runner := stringValue(suite["runner"])
	var result map[string]any
	if strings.EqualFold(runner, "LM_EVAL_HARNESS") {
		resultsPath := opt(args, "results")
		if resultsPath == "" {
			return cliError{"missing_results", "LM-Eval suites require --results <lm-eval-output.json>.", []string{"Run lmx eval lm-eval <suiteSlug> to produce results, or pass an existing lm-eval output JSON with --results."}, nil}
		}
		result, err = loadLmEvalRunResult(resultsPath, suite)
	} else if strings.EqualFold(runner, "CUSTOM") {
		result, err = runCustomLocalEval(suite, args)
	} else {
		err = cliError{"suite_runner_mismatch", "Suite runner must be CUSTOM or LM_EVAL_HARNESS.", nil, runner}
	}
	if err != nil {
		return err
	}
	payload := map[string]any{
		"suiteSlug":     suiteSlug,
		"hfId":          firstNonEmpty(opt(args, "model"), "<required-before-submit>"),
		"quantization":  opt(args, "quantization"),
		"executionMode": map[bool]string{true: "CUSTOM_LOCAL", false: "LM_EVAL_LOCAL"}[strings.EqualFold(runner, "CUSTOM")],
		"judgeMode":     map[bool]string{true: "LOCAL_REPORTED", false: "NONE"}[strings.EqualFold(stringValue(doc["scoringMethod"]), "llm_judge")],
		"runnerVersion": map[bool]string{true: "localmaxxing-go custom-local", false: "localmaxxing-go lm-eval-upload"}[strings.EqualFold(runner, "CUSTOM")],
		"results":       result["scores"],
		"artifacts":     redactGold(result["artifacts"]),
		"runConfig":     map[string]any{"aggregatePreview": result["aggregate"]},
	}
	if hardwarePath := opt(args, "hardware"); hardwarePath != "" {
		hardware, err := readJSON(hardwarePath)
		if err != nil {
			return err
		}
		payload["hardware"] = hardware
	}
	out := firstNonEmpty(opt(args, "out"), "localmaxxing-eval-run.json")
	if err := writeJSON(out, payload); err != nil {
		return err
	}
	printInfo("run_payload_written", map[string]any{"path": out, "suite": suiteSlug, "tasks": len(asObject(payload["results"])), "aggregatePreview": result["aggregate"]})
	if hasFlag(args, "submit") || hasFlag(args, "dry-run") {
		if opt(args, "hardware") == "" {
			return cliError{"missing_hardware", "--hardware is required for submit/dry-run", []string{"Create a hardware JSON file matching /api/agent-context hardwareSchemas.", "Pass --hardware hardware.json."}, nil}
		}
		if opt(args, "model") == "" {
			return cliError{"missing_model", "--model is required for submit/dry-run", []string{"Pass --model <HuggingFace model id>."}, nil}
		}
		endpoint := "/api/evals/runs"
		if hasFlag(args, "dry-run") {
			endpoint = "/api/evals/runs/dry-run"
		}
		return submitPayload(endpoint, hasFlag(args, "dry-run"), "run", args, payload)
	}
	return nil
}

func loadSuiteForEvalRun(suiteSlug string, args cliArgs) (map[string]any, error) {
	if path := opt(args, "suite-file"); path != "" {
		value, err := readJSON(path)
		if err != nil {
			return nil, err
		}
		obj := asObject(value)
		if obj == nil {
			return nil, cliError{"invalid_suite", "Suite file must contain a JSON object.", nil, value}
		}
		return obj, nil
	}
	key := apiKey(args)
	if key != "" {
		bundle, err := fetchJSON("GET", apiURL(args)+"/api/evals/suites/"+url.PathEscape(suiteSlug)+"/run-bundle", key, nil)
		if err == nil {
			return suiteFromRunBundle(bundle, suiteSlug)
		}
		printStatus(args, "run_bundle_unavailable", map[string]any{"suite": suiteSlug, "reason": err.Error()})
	}
	value, err := fetchJSON("GET", apiURL(args)+"/api/evals/suites/"+url.PathEscape(suiteSlug), key, nil)
	if err != nil {
		return nil, err
	}
	obj := asObject(value)
	if obj == nil {
		return nil, cliError{"invalid_suite", "Suite response must be a JSON object.", nil, value}
	}
	return obj, nil
}

func suiteFromRunBundle(bundle any, suiteSlug string) (map[string]any, error) {
	obj := asObject(bundle)
	if obj == nil {
		return nil, cliError{"run_bundle_invalid", "Run bundle response did not include a runnable suite document.", nil, bundle}
	}
	suite := asObject(obj["suite"])
	if suite == nil {
		suite = obj
	}
	if suite["suiteDoc"] == nil {
		suite["suiteDoc"] = obj["suiteDoc"]
	}
	if suite["slug"] == nil {
		suite["slug"] = suiteSlug
	}
	if suite["name"] == nil {
		suite["name"] = suiteSlug
	}
	if suite["runner"] == nil {
		suite["runner"] = obj["runner"]
	}
	if suiteDoc(suite) == nil || stringValue(suite["runner"]) == "" {
		return nil, cliError{"run_bundle_invalid", "Run bundle response did not include a runnable suite document.", []string{"Check that the LocalMaxxing API supports /run-bundle for this suite.", "Retry with a valid API key or inspect the suite with eval suite show."}, bundle}
	}
	applyRunBundleDownloadURLs(suite, obj)
	return suite, nil
}

func suiteDoc(suite map[string]any) map[string]any {
	return asObject(suite["suiteDoc"])
}

func evalTasks(doc map[string]any) []map[string]any {
	tasks := []map[string]any{}
	for _, item := range anySlice(doc["tasks"]) {
		if task := asObject(item); task != nil {
			tasks = append(tasks, task)
		}
	}
	return tasks
}

func evalTaskKeys(doc map[string]any) []string {
	keys := []string{}
	for _, task := range evalTasks(doc) {
		if key := stringValue(task["key"]); key != "" {
			keys = append(keys, key)
		}
	}
	return keys
}

func inferredEvalFewShot(doc map[string]any) string {
	if runConfig := asObject(doc["runConfig"]); runConfig != nil {
		if n := numberField(runConfig, "fewShot"); n > 0 {
			return strconv.Itoa(int(n))
		}
	}
	seen := map[int]bool{}
	for _, task := range evalTasks(doc) {
		if n := numberField(task, "nShots"); n > 0 {
			seen[int(n)] = true
		}
	}
	if len(seen) != 1 {
		return ""
	}
	for n := range seen {
		return strconv.Itoa(n)
	}
	return ""
}

func applyRunBundleDownloadURLs(suite map[string]any, bundle map[string]any) {
	doc := suiteDoc(suite)
	if doc == nil {
		return
	}
	candidates := []any{bundle["downloadUrls"], bundle["datasetDownloadUrls"], bundle["datasets"]}
	tasks := anySlice(doc["tasks"])
	for i, item := range tasks {
		task := asObject(item)
		if task == nil {
			continue
		}
		dataset := asObject(task["dataset"])
		if dataset != nil && firstDatasetDownloadURL(dataset) != "" {
			continue
		}
		urlText := ""
		for _, candidate := range candidates {
			urlText = downloadURLForTask(candidate, task, i, len(tasks))
			if urlText != "" {
				break
			}
		}
		if urlText == "" {
			continue
		}
		if dataset == nil {
			dataset = map[string]any{}
		}
		dataset["source"] = "url"
		dataset["url"] = urlText
		dataset["downloadUrl"] = urlText
		task["dataset"] = dataset
	}
}

func firstDatasetDownloadURL(dataset map[string]any) string {
	return firstNonEmpty(stringValue(dataset["downloadUrl"]), downloadURLFromValue(dataset["downloadUrls"]))
}

func downloadURLForTask(value any, task map[string]any, taskIndex, totalTasks int) string {
	if arr := anySlice(value); len(arr) > 0 {
		if taskIndex < len(arr) {
			if urlText := downloadURLFromValue(arr[taskIndex]); urlText != "" {
				return urlText
			}
		}
		for _, child := range arr {
			obj := asObject(child)
			if obj == nil {
				continue
			}
			key := firstNonEmpty(stringValue(obj["taskKey"]), stringValue(obj["task"]), stringValue(obj["key"]))
			if key == stringValue(task["key"]) {
				return downloadURLFromValue(obj)
			}
		}
		return ""
	}
	obj := asObject(value)
	if obj == nil {
		return downloadURLFromValue(value)
	}
	dataset := asObject(task["dataset"])
	keys := []string{stringValue(task["key"]), strconv.Itoa(taskIndex)}
	if dataset != nil {
		for _, key := range []string{"storageKey", "storageRef", "datasetKey", "id"} {
			keys = append(keys, stringValue(dataset[key]))
		}
	}
	for _, key := range keys {
		if key == "" {
			continue
		}
		if urlText := downloadURLFromValue(obj[key]); urlText != "" {
			return urlText
		}
	}
	if totalTasks == 1 {
		return downloadURLFromValue(value)
	}
	return ""
}

func downloadURLFromValue(value any) string {
	if text := stringValue(value); text != "" {
		return text
	}
	if arr := anySlice(value); len(arr) > 0 {
		for _, child := range arr {
			if urlText := downloadURLFromValue(child); urlText != "" {
				return urlText
			}
		}
		return ""
	}
	obj := asObject(value)
	if obj == nil {
		return ""
	}
	return firstNonEmpty(stringValue(obj["downloadUrl"]), stringValue(obj["url"]), stringValue(obj["signedUrl"]), downloadURLFromValue(obj["downloadUrls"]))
}

func loadLmEvalRunResult(path string, suite map[string]any) (map[string]any, error) {
	raw, err := readJSON(path)
	if err != nil {
		return nil, err
	}
	doc := suiteDoc(suite)
	scoring := firstNonEmpty(stringValue(doc["scoringMethod"]), "exact_match")
	scores := map[string]any{}
	flatScores := map[string]float64{}
	printInfo("lm_eval_parse_start", map[string]any{"suite": suite["slug"], "tasks": len(evalTasks(doc)), "scoringMethod": scoring, "results": path})
	for _, task := range evalTasks(doc) {
		key := stringValue(task["key"])
		result := lmEvalResultForTask(raw, key)
		score, ok := scoreFromLmEvalTask(result, scoring)
		if !ok {
			return nil, cliError{"lm_eval_metric_not_found", fmt.Sprintf("Could not find lm-eval score for task %q", key), []string{fmt.Sprintf("Ensure the lm-eval output contains results.%s or groups.%s.", key, key), "If the task uses a different metric, edit the suite scoringMethod or extend the CLI metric mapping."}, map[string]any{"taskKey": key, "availableMetrics": availableMetricNames(result)}}
		}
		scores[key] = map[string]any{"score": score, "nShots": firstNonZero(int(numberField(task, "nShots")), atoiDefault(inferredEvalFewShot(doc), 0))}
		flatScores[key] = score
		printInfo("lm_eval_task_score", map[string]any{"task": key, "score": score})
	}
	return map[string]any{"scores": scores, "artifacts": []any{}, "aggregate": computeEvalAggregate(doc, flatScores)}, nil
}

func lmEvalResultForTask(raw any, taskKey string) any {
	obj := asObject(raw)
	if obj == nil {
		return nil
	}
	results := asObject(obj["results"])
	if results == nil {
		results = obj
	}
	groups := asObject(obj["groups"])
	underscore := strings.ReplaceAll(taskKey, "-", "_")
	if results != nil {
		if results[taskKey] != nil {
			return results[taskKey]
		}
		if results[underscore] != nil {
			return results[underscore]
		}
	}
	if groups != nil {
		if groups[taskKey] != nil {
			return groups[taskKey]
		}
		if groups[underscore] != nil {
			return groups[underscore]
		}
	}
	return nil
}

func scoreFromLmEvalTask(value any, scoring string) (float64, bool) {
	obj := asObject(value)
	if obj == nil {
		return 0, false
	}
	for _, key := range lmEvalMetricCandidates(scoring) {
		if normalized, ok := normalizeMetricScore(key, numberField(obj, key)); ok {
			return normalized, true
		}
	}
	for key, raw := range obj {
		if strings.Contains(strings.ToLower(key), "stderr") {
			continue
		}
		if normalized, ok := normalizeMetricScore(key, numericValue(raw)); ok {
			return normalized, true
		}
	}
	return 0, false
}

func lmEvalMetricCandidates(scoring string) []string {
	switch scoring {
	case "f1":
		return []string{"f1,none", "f1", "macro_f1,none", "macro_f1", "rouge1,none", "rouge1", "rougeL,none", "rougeL"}
	case "pass_at_k":
		return []string{"pass_at_1,none", "pass@1,none", "pass_at_1", "pass@1", "pass_at_k,none", "pass@k,none", "pass_at_k", "pass@k"}
	case "llm_judge":
		return []string{"score,none", "score", "acc,none", "acc"}
	case "perplexity":
		return []string{"word_perplexity,none", "perplexity,none", "ppl,none", "word_perplexity", "perplexity", "ppl"}
	default:
		return []string{"acc_norm,none", "acc,none", "exact_match,none", "exact,none", "em,none", "pass_at_1,none", "pass@1,none", "inst_level_strict_acc,none", "prompt_level_strict_acc,none", "acc_norm", "acc", "exact_match", "exact", "em", "pass_at_1", "pass@1", "inst_level_strict_acc", "prompt_level_strict_acc"}
	}
}

func normalizeMetricScore(metric string, value float64) (float64, bool) {
	if value <= 0 && value != 0 {
		return 0, false
	}
	if value >= 0 && value <= 1 {
		return value, true
	}
	metric = strings.ToLower(metric)
	if (strings.HasPrefix(metric, "rouge") || strings.Contains(metric, "bleu") || strings.Contains(metric, "chrf")) && value >= 0 && value <= 100 {
		return value / 100, true
	}
	return 0, false
}

func availableMetricNames(value any) []string {
	obj := asObject(value)
	if obj == nil {
		return []string{}
	}
	names := []string{}
	for key, raw := range obj {
		if numericValue(raw) != 0 || raw == float64(0) || raw == 0 {
			names = append(names, key)
		}
	}
	sort.Strings(names)
	return names
}

func runCustomLocalEval(suite map[string]any, args cliArgs) (map[string]any, error) {
	model, err := requireOpt(args, "model")
	if err != nil {
		return nil, err
	}
	baseURL := openAIBaseURL(firstNonEmpty(opt(args, "base-url"), "http://localhost:8000"))
	doc := suiteDoc(suite)
	scoring := stringValue(doc["scoringMethod"])
	judgeBaseURL := firstNonEmpty(opt(args, "judge-base-url"), stringValue(asObject(doc["judge"])["baseUrl"]))
	judgeModel := firstNonEmpty(opt(args, "judge-model"), stringValue(asObject(doc["judge"])["model"]))
	judgeAPIKey := firstNonEmpty(opt(args, "judge-api-key"), os.Getenv("EVAL_JUDGE_API_KEY"))
	if scoring == "llm_judge" && (judgeBaseURL == "" || judgeModel == "") {
		return nil, cliError{"judge_config_missing", "llm_judge suites require --judge-base-url and --judge-model, or suiteDoc.judge defaults", []string{"Pass --judge-base-url and --judge-model.", "If the judge requires auth, pass --judge-api-key or set EVAL_JUDGE_API_KEY."}, nil}
	}
	printInfo("custom_eval_start", map[string]any{"suite": suite["slug"], "tasks": len(evalTasks(doc)), "model": model, "baseUrl": baseURL, "scoringMethod": scoring})
	scores := map[string]any{}
	flatScores := map[string]float64{}
	artifacts := []any{}
	for _, task := range evalTasks(doc) {
		if stringValue(task["promptTemplate"]) == "" || asObject(task["dataset"]) == nil {
			return nil, cliError{"task_not_runnable", fmt.Sprintf("Task %q requires promptTemplate and dataset", stringValue(task["key"])), []string{"Fix the suite JSON or use an LM_EVAL_HARNESS suite for external lm-eval tasks."}, nil}
		}
		items, err := loadEvalDataset(asObject(task["dataset"]))
		if err != nil {
			return nil, cliError{"dataset_load_failed", fmt.Sprintf("Failed to load dataset for task %q: %v", stringValue(task["key"]), err), []string{"Check dataset source fields and network access."}, nil}
		}
		totalScore := 0.0
		counted := 0
		failures := 0
		for i, item := range items {
			prompt := renderEvalPrompt(stringValue(task["promptTemplate"]), item)
			started := time.Now()
			artifact := map[string]any{"taskKey": stringValue(task["key"]), "itemIndex": i, "promptHash": sha256Hex(prompt), "question": renderEvalQuestion(item), "prompt": prompt}
			response, err := callOpenAIChat(baseURL, model, prompt, opt(args, "model-api-key"), int(firstNonZero(int(numberField(task, "maxNewTokens")), 256)), evalTemperature(doc), evalTopP(doc), stringSlice(task["stopSequences"]))
			if err == nil {
				artifact["response"] = response
				var score float64
				score, artifact, err = scoreCustomEvalItem(scoring, task, item, response, prompt, artifact, judgeBaseURL, judgeModel, judgeAPIKey)
				if err == nil {
					totalScore += score
					artifact["score"] = score
				}
			}
			if err != nil {
				failures++
				artifact["error"] = err.Error()
			}
			counted++
			artifact["latencyMs"] = time.Since(started).Milliseconds()
			artifacts = append(artifacts, artifact)
		}
		if counted > 0 {
			score := totalScore / float64(counted)
			scores[stringValue(task["key"])] = map[string]any{"score": score, "nSamples": counted, "nShots": numberField(task, "nShots")}
			flatScores[stringValue(task["key"])] = score
		}
		printInfo("task_complete", map[string]any{"task": task["key"], "samples": counted, "failures": failures, "score": flatScores[stringValue(task["key"])]})
	}
	return map[string]any{"scores": scores, "artifacts": artifacts, "aggregate": computeEvalAggregate(doc, flatScores)}, nil
}

func loadEvalDataset(dataset map[string]any) ([]map[string]any, error) {
	if dataset == nil {
		return nil, errors.New("dataset missing")
	}
	if stringValue(dataset["source"]) == "inline" {
		items := []map[string]any{}
		for _, item := range anySlice(dataset["items"]) {
			if obj := asObject(item); obj != nil {
				items = append(items, obj)
			}
		}
		return items, nil
	}
	if urlText := firstNonEmpty(firstDatasetDownloadURL(dataset), stringValue(dataset["url"])); urlText != "" {
		return fetchDatasetItems(urlText, stringValue(dataset["format"]))
	}
	if stringValue(dataset["source"]) == "huggingface" {
		hfPath := stringValue(dataset["hfPath"])
		if hfPath == "" {
			return nil, errors.New("huggingface dataset missing hfPath")
		}
		name := firstNonEmpty(stringValue(dataset["hfName"]), "default")
		split := firstNonEmpty(stringValue(dataset["split"]), "test")
		urlText := "https://datasets-server.huggingface.co/rows?dataset=" + url.QueryEscape(hfPath) + "&config=" + url.QueryEscape(name) + "&split=" + url.QueryEscape(split) + "&offset=0&limit=500"
		rows, err := fetchEndpointJSON(urlText, "")
		if err != nil {
			return nil, err
		}
		items := []map[string]any{}
		for _, item := range anySlice(asObject(rows)["rows"]) {
			row := asObject(asObject(item)["row"])
			if row == nil {
				continue
			}
			items = append(items, map[string]any{"input": firstNonEmpty(stringValue(row["question"]), stringValue(row["input"]), stringValue(row["prompt"])), "gold": firstNonEmpty(stringValue(row["answer"]), stringValue(row["gold"]), stringValue(row["label"])), "choices": row["choices"]})
		}
		return items, nil
	}
	return nil, fmt.Errorf("unknown dataset source %q", stringValue(dataset["source"]))
}

func fetchDatasetItems(urlText, format string) ([]map[string]any, error) {
	res, err := http.Get(urlText)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	text, _ := io.ReadAll(res.Body)
	if res.StatusCode < 200 || res.StatusCode >= 300 {
		return nil, fmt.Errorf("dataset download failed: %s", res.Status)
	}
	return parseDatasetItems(string(text), urlText, format)
}

func parseDatasetItems(text, urlText, format string) ([]map[string]any, error) {
	if format == "jsonl" || strings.HasSuffix(strings.ToLower(urlText), ".jsonl") {
		return parseJSONLDataset(text)
	}
	var arr []map[string]any
	if err := json.Unmarshal([]byte(text), &arr); err == nil {
		return arr, nil
	}
	return parseJSONLDataset(text)
}

func parseJSONLDataset(text string) ([]map[string]any, error) {
	items := []map[string]any{}
	for _, line := range strings.Split(text, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		var obj map[string]any
		if err := json.Unmarshal([]byte(line), &obj); err != nil {
			return nil, err
		}
		items = append(items, obj)
	}
	return items, nil
}

func renderEvalPrompt(template string, item map[string]any) string {
	prompt := strings.ReplaceAll(template, "{{input}}", fmt.Sprint(item["input"]))
	prompt = strings.ReplaceAll(prompt, "{{gold}}", "")
	if choices := stringChoices(item["choices"]); len(choices) > 0 {
		parts := []string{}
		for i, choice := range choices {
			parts = append(parts, choiceLabel(i)+". "+choice)
		}
		prompt = strings.ReplaceAll(prompt, "{{choices}}", strings.Join(parts, "\n"))
	}
	return strings.TrimSpace(prompt)
}

func renderEvalQuestion(item map[string]any) string {
	input := strings.TrimSpace(fmt.Sprint(item["input"]))
	choices := stringChoices(item["choices"])
	if len(choices) == 0 {
		return input
	}
	parts := []string{}
	for i, choice := range choices {
		parts = append(parts, choiceLabel(i)+". "+choice)
	}
	return strings.TrimSpace(input + "\n\n" + strings.Join(parts, "\n"))
}

func callOpenAIChat(baseURL, model, prompt, apiKey string, maxTokens int, temperature, topP float64, stop []string) (string, error) {
	body := map[string]any{"model": model, "messages": []any{map[string]any{"role": "user", "content": prompt}}, "max_tokens": maxTokens, "temperature": temperature, "top_p": topP}
	if len(stop) > 0 {
		body["stop"] = stop
	}
	data, _ := json.Marshal(body)
	ctx, cancel := context.WithTimeout(context.Background(), defaultEndpointTimeout)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, "POST", openAIBaseURL(baseURL)+"/v1/chat/completions", bytes.NewReader(data))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	if apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", err
	}
	defer res.Body.Close()
	if res.StatusCode < 200 || res.StatusCode >= 300 {
		text, _ := io.ReadAll(res.Body)
		return "", fmt.Errorf("OpenAI-compatible server returned %s: %s", res.Status, strings.TrimSpace(string(text)))
	}
	var response map[string]any
	if err := json.NewDecoder(res.Body).Decode(&response); err != nil {
		return "", err
	}
	return strings.TrimSpace(nonStreamingContent(response)), nil
}

func scoreCustomEvalItem(scoring string, task, item map[string]any, response, prompt string, artifact map[string]any, judgeBaseURL, judgeModel, judgeAPIKey string) (float64, map[string]any, error) {
	switch scoring {
	case "exact_match":
		gold, ok := item["gold"]
		if !ok {
			return 0, artifact, errors.New("item is missing gold answer for exact_match scoring")
		}
		if stringValue(task["taskType"]) == "multiple_choice" || len(stringChoices(item["choices"])) > 0 {
			if normalizeChoice(response, stringChoices(item["choices"])) == normalizeChoice(fmt.Sprint(gold), stringChoices(item["choices"])) {
				return 1, artifact, nil
			}
			return 0, artifact, nil
		}
		if normalizeEvalText(response) == normalizeEvalText(fmt.Sprint(gold)) {
			return 1, artifact, nil
		}
		return 0, artifact, nil
	case "f1":
		if item["gold"] == nil {
			return 0, artifact, errors.New("item is missing gold answer for f1 scoring")
		}
		return tokenF1(response, fmt.Sprint(item["gold"])), artifact, nil
	case "llm_judge":
		score, rationale, err := judgeEvalResponse(judgeBaseURL, judgeModel, judgeAPIKey, task, item, prompt, response)
		artifact["judgeModel"] = judgeModel
		artifact["judgeScore"] = score
		artifact["judgeRationale"] = rationale
		return score, artifact, err
	default:
		return 0, artifact, fmt.Errorf("CLI custom evals do not support scoringMethod %q yet", scoring)
	}
}

func judgeEvalResponse(baseURL, model, apiKey string, task, item map[string]any, prompt, response string) (float64, string, error) {
	rubric := firstNonEmpty(stringValue(item["rubric"]), "Score the response from 0 to 1 for correctness and quality.")
	judgePrompt := "You are grading a model response. Return strict JSON: {\"score\": number, \"rationale\": string}.\n\nRubric:\n" + rubric + "\n\nQuestion:\n" + renderEvalQuestion(item) + "\n\nPrompt sent to model:\n" + prompt + "\n\nModel response:\n" + response + "\n\nReference answer, if any:\n" + stringValue(item["referenceAnswer"])
	raw, err := callOpenAIChat(baseURL, model, judgePrompt, apiKey, 512, 0, 1, nil)
	if err != nil {
		return 0, "", err
	}
	return parseJudgeResponse(raw)
}

func parseJudgeResponse(raw string) (float64, string, error) {
	match := regexp.MustCompile(`(?s)\{.*\}`).FindString(raw)
	if match != "" {
		var obj map[string]any
		if json.Unmarshal([]byte(match), &obj) == nil {
			score := numericValue(obj["score"])
			if score >= 0 && score <= 1 {
				return score, firstNonEmpty(stringValue(obj["rationale"]), raw), nil
			}
		}
	}
	parts := regexp.MustCompile(`(?i)(?:score\D+)?([01](?:\.\d+)?)`).FindStringSubmatch(raw)
	if len(parts) > 1 {
		score, _ := strconv.ParseFloat(parts[1], 64)
		if score >= 0 && score <= 1 {
			return score, raw, nil
		}
	}
	return 0, raw, fmt.Errorf("judge did not return a parseable score: %s", raw)
}

func computeEvalAggregate(doc map[string]any, scores map[string]float64) float64 {
	tasks := evalTasks(doc)
	values := []float64{}
	weights := []float64{}
	for _, task := range tasks {
		key := stringValue(task["key"])
		value, ok := scores[key]
		if !ok {
			continue
		}
		values = append(values, value)
		weight := numberField(task, "weight")
		if weight <= 0 {
			weight = 1
		}
		weights = append(weights, weight)
	}
	if len(values) == 0 {
		return 0
	}
	switch stringValue(doc["aggregation"]) {
	case "min":
		min := values[0]
		for _, value := range values[1:] {
			if value < min {
				min = value
			}
		}
		return min
	case "max":
		max := values[0]
		for _, value := range values[1:] {
			if value > max {
				max = value
			}
		}
		return max
	case "mean":
		sum := 0.0
		for _, value := range values {
			sum += value
		}
		return sum / float64(len(values))
	default:
		sum := 0.0
		weightSum := 0.0
		for i, value := range values {
			sum += value * weights[i]
			weightSum += weights[i]
		}
		return sum / weightSum
	}
}

func numericValue(value any) float64 {
	switch v := value.(type) {
	case float64:
		return v
	case float32:
		return float64(v)
	case int:
		return float64(v)
	case int64:
		return float64(v)
	case json.Number:
		n, _ := v.Float64()
		return n
	default:
		return 0
	}
}

func atoiDefault(value string, fallback int) int {
	n, err := strconv.Atoi(value)
	if err != nil {
		return fallback
	}
	return n
}

func sha256Hex(value string) string {
	sum := sha256.Sum256([]byte(value))
	return hex.EncodeToString(sum[:])
}

func stringSlice(value any) []string {
	out := []string{}
	for _, item := range anySlice(value) {
		out = append(out, fmt.Sprint(item))
	}
	return out
}

func stringChoices(value any) []string {
	choices := []string{}
	for _, item := range anySlice(value) {
		choices = append(choices, fmt.Sprint(item))
	}
	return choices
}

func choiceLabel(index int) string {
	return string(rune('A' + index))
}

func normalizeEvalText(value string) string {
	lower := strings.ToLower(value)
	re := regexp.MustCompile(`[^\w\s]+`)
	return strings.TrimSpace(re.ReplaceAllString(lower, ""))
}

func tokenF1(pred, gold string) float64 {
	predTokens := strings.Fields(normalizeEvalText(pred))
	goldTokens := strings.Fields(normalizeEvalText(gold))
	if len(predTokens) == 0 || len(goldTokens) == 0 {
		return 0
	}
	predCounts := map[string]int{}
	goldCounts := map[string]int{}
	for _, token := range predTokens {
		predCounts[token]++
	}
	for _, token := range goldTokens {
		goldCounts[token]++
	}
	common := 0
	for token, count := range predCounts {
		if goldCounts[token] < count {
			common += goldCounts[token]
		} else {
			common += count
		}
	}
	if common == 0 {
		return 0
	}
	precision := float64(common) / float64(len(predTokens))
	recall := float64(common) / float64(len(goldTokens))
	return (2 * precision * recall) / (precision + recall)
}

func normalizeChoice(value string, choices []string) string {
	normalized := normalizeEvalText(value)
	if regexp.MustCompile(`^[a-z]$`).MatchString(normalized) {
		return strings.ToUpper(normalized)
	}
	if regexp.MustCompile(`^\d+$`).MatchString(normalized) {
		n, _ := strconv.Atoi(normalized)
		if n < 1 {
			n = 1
		}
		return choiceLabel(n - 1)
	}
	for i, choice := range choices {
		if normalizeEvalText(choice) == normalized {
			return choiceLabel(i)
		}
	}
	for _, token := range strings.Fields(normalized) {
		if regexp.MustCompile(`^[a-z]$`).MatchString(token) {
			return strings.ToUpper(token)
		}
	}
	return normalized
}

func evalTemperature(doc map[string]any) float64 {
	if runConfig := asObject(doc["runConfig"]); runConfig != nil {
		if value := numberField(runConfig, "temperature"); value != 0 {
			return value
		}
	}
	return 0
}

func evalTopP(doc map[string]any) float64 {
	if runConfig := asObject(doc["runConfig"]); runConfig != nil {
		if value := numberField(runConfig, "topP"); value != 0 {
			return value
		}
	}
	return 1
}

func containsString(values []string, target string) bool {
	for _, value := range values {
		if value == target {
			return true
		}
	}
	return false
}

func stringMap(value any) map[string]string {
	out := map[string]string{}
	obj := asObject(value)
	if obj == nil {
		return out
	}
	for key, raw := range obj {
		if raw == nil {
			continue
		}
		out[key] = fmt.Sprint(raw)
	}
	return out
}

func submitPayload(endpoint string, dryRun bool, label string, args cliArgs, payload any) error {
	key := apiKey(args)
	if key == "" {
		return missingAPIKey("--api-key or LMX_API_KEY is required for " + label + " submit/dry-run")
	}
	value, err := fetchJSON("POST", apiURL(args)+endpoint, key, payload)
	if err != nil {
		return err
	}
	printJSON(value)
	status := label + "_submitted"
	if dryRun {
		status = label + "_dry_run_valid"
	}
	fields := map[string]any{"endpoint": endpoint, "status": map[bool]string{true: "valid", false: "submitted"}[dryRun]}
	if receipt := receiptURL(value); receipt != "" {
		fields["url"] = receipt
	}
	printInfo(status, fields)
	if dryRun {
		fmt.Println("Dry-run passed. Submit with:")
		if label == "benchmark" {
			fmt.Println("  lmx benchmark submit <payload.json>")
		} else if label == "run" {
			fmt.Println("  lmx eval run <suiteSlug> --results <results.json> --submit")
		}
	}
	return nil
}

func printNextSteps(kind, out string) {
	fmt.Println("Next:")
	if kind == "benchmark" {
		fmt.Println("  lmx benchmark dry-run " + out)
		fmt.Println("  lmx benchmark submit " + out)
		return
	}
	fmt.Println("  lmx " + kind + " dry-run " + out)
	fmt.Println("  lmx " + kind + " submit " + out)
}

func printBenchmarkNextSteps(feedback map[string]any, out string) {
	fmt.Println("Next:")
	printedValidation := false
	if next := stringValue(feedback["nextCommand"]); next != "" {
		fmt.Println("  " + next)
		printedValidation = next == "lmx benchmark dry-run "+out
	}
	if validation := stringValue(feedback["validationCommand"]); validation != "" {
		fmt.Println("  " + validation)
		printedValidation = true
	} else if !printedValidation {
		fmt.Println("  lmx benchmark dry-run " + out)
	}
	if submit := stringValue(feedback["submitCommand"]); submit != "" {
		fmt.Println("  " + submit)
	} else if feedback["canSubmit"] != false {
		fmt.Println("  lmx benchmark submit " + out)
	}
}

func receiptURL(value any) string {
	obj := asObject(value)
	if obj == nil {
		return ""
	}
	for _, key := range []string{"url", "href", "dashboardUrl", "benchmarkUrl", "runUrl"} {
		if text := stringValue(obj[key]); text != "" {
			return text
		}
	}
	if nested := asObject(obj["benchmark"]); nested != nil {
		if text := receiptURL(nested); text != "" {
			return text
		}
	}
	if nested := asObject(obj["run"]); nested != nil {
		if text := receiptURL(nested); text != "" {
			return text
		}
	}
	return ""
}

func missingAPIKey(message string) error {
	return cliError{"missing_api_key", message, []string{"Create an API key in the LocalMaxxing dashboard.", "Pass it with --api-key bhk_... or set LMX_API_KEY."}, nil}
}

func asObject(value any) map[string]any {
	obj, _ := value.(map[string]any)
	return obj
}

func firstString(obj map[string]any, keys ...string) string {
	for _, key := range keys {
		if value, ok := obj[key].(string); ok && value != "" {
			return value
		}
	}
	return ""
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}
	return ""
}

func redactGold(value any) any {
	switch v := value.(type) {
	case []any:
		out := make([]any, len(v))
		for i, child := range v {
			out[i] = redactGold(child)
		}
		return out
	case map[string]any:
		out := map[string]any{}
		for key, child := range v {
			if goldFieldNames[key] {
				continue
			}
			out[key] = redactGold(child)
		}
		return out
	default:
		return value
	}
}

func toJSON(value any) string {
	data, _ := json.Marshal(value)
	return string(data)
}

func printJSON(value any) {
	data, _ := json.MarshalIndent(value, "", "  ")
	fmt.Println(string(data))
}

func printInfo(title string, fields map[string]any) {
	fmt.Println("[localmaxxing] " + title)
	for key, value := range fields {
		if value == nil || fmt.Sprint(value) == "" {
			continue
		}
		fmt.Printf("  %s: %v\n", key, value)
	}
}

func printStatus(args cliArgs, event string, fields map[string]any) {
	if fields == nil {
		fields = map[string]any{}
	}
	if hasFlag(args, "quiet") {
		return
	}
	if hasFlag(args, "json-status") {
		payload := map[string]any{"event": event, "time": time.Now().UTC().Format(time.RFC3339)}
		for key, value := range fields {
			if value == nil || fmt.Sprint(value) == "" {
				continue
			}
			payload[key] = value
		}
		data, _ := json.Marshal(payload)
		fmt.Fprintln(os.Stderr, string(data))
		return
	}
	fmt.Fprintln(os.Stderr, "[localmaxxing] "+event)
	for key, value := range fields {
		if value == nil || fmt.Sprint(value) == "" {
			continue
		}
		fmt.Fprintf(os.Stderr, "  %s: %v\n", key, value)
	}
}

func metricStatusFields(metrics map[string]float64) map[string]any {
	fields := map[string]any{"count": len(metrics)}
	for _, key := range []string{"tokSOut", "tokSPrefill", "tokSTotal", "ttftMs", "peakVramGb", "promptTokens", "outputTokens"} {
		if value, ok := metrics[key]; ok && value > 0 {
			fields[key] = value
		}
	}
	if len(metrics) == 0 {
		fields["next"] = "No metrics detected yet; pass explicit --tok-s-out or provide llama-bench/output text."
	}
	return fields
}

func printError(args cliArgs, err error) {
	var ce cliError
	if errors.As(err, &ce) {
		if hasFlag(args, "json-status") {
			payload := map[string]any{"event": "error", "time": time.Now().UTC().Format(time.RFC3339), "code": ce.Code, "message": ce.Message}
			if len(ce.Hints) > 0 {
				payload["hints"] = ce.Hints
			}
			if ce.Details != nil {
				payload["details"] = ce.Details
			}
			data, _ := json.Marshal(payload)
			fmt.Fprintln(os.Stderr, string(data))
			return
		}
		fmt.Fprintln(os.Stderr, "[localmaxxing:error] "+ce.Code)
		fmt.Fprintln(os.Stderr, ce.Message)
		if len(ce.Hints) > 0 {
			fmt.Fprintln(os.Stderr, "Fix:")
			for _, hint := range ce.Hints {
				fmt.Fprintln(os.Stderr, "- "+hint)
			}
		}
		if ce.Details != nil {
			fmt.Fprintln(os.Stderr, "Details:")
			if text, ok := ce.Details.(string); ok {
				fmt.Fprintln(os.Stderr, text)
			} else {
				data, _ := json.MarshalIndent(ce.Details, "", "  ")
				fmt.Fprintln(os.Stderr, string(data))
			}
		}
		return
	}
	if hasFlag(args, "json-status") {
		data, _ := json.Marshal(map[string]any{"event": "error", "time": time.Now().UTC().Format(time.RFC3339), "code": "unexpected_error", "message": err.Error()})
		fmt.Fprintln(os.Stderr, string(data))
		return
	}
	fmt.Fprintln(os.Stderr, "[localmaxxing:error] unexpected_error")
	fmt.Fprintln(os.Stderr, err.Error())
}

func usage() {
	fmt.Println(`LocalMaxxing CLI

Usage:
  lmx context --out localmaxxing-agent-context.json
  lmx auth --key bhk_...
  lmx profile save my-4090 --mode local --hf-id Qwen/Qwen3-8B --quantization Q4_K_M --hardware hardware.json
  lmx hardware --out hardware.json
  lmx hardware init --out hardware.json
  lmx engines
  lmx endpoint discover --hf-id Qwen/Qwen3-8B --quantization fp16
  lmx server dry-run vllm --hf-id Qwen/Qwen3-8B --quantization fp16
  lmx server dry-run llama.cpp --model-path model.gguf
  lmx benchmark run vllm --mode remote --base-url http://server:8000 --hf-id Qwen/Qwen3-8B --quantization fp16 --dry-run
  lmx benchmark run vllm --mode local --hf-id Qwen/Qwen3-8B --quantization fp16 --bench-kind throughput --benchmark-output vllm.json --dry-run
  lmx benchmark run llama.cpp --mode local --hf-id Qwen/Qwen3-8B --quantization Q4_K_M --command "llama-bench -m model.gguf" --dry-run
  lmx benchmark run llama.cpp --mode local --hf-id Qwen/Qwen3-8B --quantization Q4_K_M --model-path model.gguf --dry-run
  lmx benchmark runs list
  lmx benchmark runs show runs/Qwen-Qwen3-8B/run.json
  lmx benchmark runs edit runs/Qwen-Qwen3-8B/run.json --set-json '{"tokSOut":120}'
  lmx benchmark runs rerun runs/Qwen-Qwen3-8B/run.json --dry-run
  lmx benchmark runs submit runs/Qwen-Qwen3-8B/run.json --api-key bhk_...
  lmx benchmark runs delete runs/Qwen-Qwen3-8B/run.json --yes
  lmx benchmark runs stats --group-by quantization --metric tokSOut
  lmx benchmark runs compare --by hardware --model Qwen/Qwen3-8B
  lmx benchmark runs compare runs/base.json runs/candidate.json --metrics tokSOut,ttftMs
  lmx benchmark runs export --format csv --out runs.csv
  lmx kvcache run llama.cpp --hf-id Qwen/Qwen3-8B --model-path model.gguf --levels 10000,20000,30000,40000
  lmx kvcache run vllm --mode remote --base-url http://server:8000 --hf-id Qwen/Qwen3-8B --levels 10000,20000,30000,40000
  lmx benchmark submit benchmark.json --api-key bhk_...
  lmx benchmark dry-run benchmark.json --api-key bhk_...
  lmx benchmark validate-local benchmark.json
  lmx eval suite list --out suites.json
  lmx eval suite search reasoning --out reasoning-suites.json
  lmx eval suite show hellaswag --out hellaswag-suite.json
  lmx model search qwen3-8b --out models.json
  lmx eval storage upload traces.jsonl --kind artifact --format jsonl --out artifact-bundle.json
  lmx eval storage download <storageKey> --out traces.jsonl
  lmx eval lm-eval hellaswag --model Qwen/Qwen3-8B --backend hf --hardware hardware.json --dry-run
  lmx eval suite init --slug my-eval --name "My Eval" --category reasoning --out my-eval.json
  lmx eval suite validate my-eval.json
  lmx eval suite submit my-eval.json --api-key bhk_...
  lmx eval execute <suiteSlug> --model <hfId> --base-url http://localhost:8000 --hardware hardware.json --submit

Options:
  --api-url <url>          LocalMaxxing origin (default: https://www.localmaxxing.com)
  --api-key <key>          API key, defaults to LMX_API_KEY, then saved config
  --profile <name>         Load saved defaults from lmx profile save
  --model <hfId>           HuggingFace model ID
  --backend <name>         lm-eval backend name for eval lm-eval (default: hf)
  --model-args <args>      lm-eval --model_args value
  --num-fewshot <n>        lm-eval --num_fewshot override
  --lm-eval-bin <path>     lm-eval executable (default: lm_eval)
  --base-url <url>         OpenAI-compatible model endpoint; accepts host or host/v1
  --mode <mode>            Benchmark mode: remote endpoint or local host command
  --served-model <name>    Model name served by the OpenAI-compatible endpoint
  --model-api-key <key>    Optional bearer token for remote endpoint benchmarking
  --prompt <text>          Prompt for remote endpoint benchmark
  --max-tokens <n>         Max generated tokens for remote endpoint benchmark
  --endpoint-timeout-seconds <n> Timeout for remote endpoint benchmark (default: 600)
  --no-stream              Disable streaming for remote endpoint benchmark
  --command <cmd>          Local benchmark command, e.g. llama-bench
  --host <addr>            Local model server host for generated server commands
  --port <n>               Local model server port for generated server/benchmark commands
  --model-path <path>      llama.cpp model path; generates llama-bench command
  --depth <n>              llama-bench -d depth for benchmark run; KV sweeps use --levels
  --batch-size <n>         llama-bench -b batch size
  --micro-batch-size <n>   llama-bench -ub micro-batch size
  --repetitions <n>        llama-bench -r repetitions
  --benchmark-format <fmt> llama-bench -o output format, e.g. json or md
  --flash-attn             llama-bench -fa 1; use --no-flash-attn for -fa 0
  --cache-type-k <type>    llama-bench -ctk KV cache K type
  --cache-type-v <type>    llama-bench -ctv KV cache V type
  --server-bin <path>      Server executable override, e.g. llama-server
  --bench-kind <kind>      Built-in vLLM benchmark: serve, throughput, or latency
  --benchmark-output <p>   Engine benchmark JSON output path
  --benchmark-bin <path>   Benchmark executable (default: vllm for vLLM)
  --python-bin <path>      Python executable for SGLang commands
  --input-len <n>          Prompt/input tokens for built-in benchmark commands
  --output-len <n>         Generated/output tokens for built-in benchmark commands
  --levels <list>          KV-cache/context sweep levels, e.g. 10000,20000,30000
  --command-template <cmd> Local sweep command template using {input} and {output}
  --probe-prompt <text>    Final remote prompt after loading retained context
  --filler-token <text>    Repeated token used for remote context filler
  --kv-cache-dtype <dtype> vLLM KV cache dtype for local latency sweeps
  --enable-prefix-caching  Enable vLLM prefix caching for local latency sweeps
  --num-prompts <n>        Number of prompts for vLLM serve/throughput benchmarks
  --runs-dir <dir>         Saved benchmark runs directory (default: runs)
  --group-by <field>       Group saved-run stats by field, e.g. quantization or hardware
  --by <field>             Group saved-run comparisons by field
  --metric <field>         Saved-run metric for stats/compare (default: tokSOut)
  --metrics <fields>       Comma-separated metrics for comparing two run files
  --fields <fields>        Comma-separated saved-run export fields
  --hardware-name <text>   Filter saved runs by hardware label substring
  --set field=value        Edit one field in a saved benchmark run
  --set-json <json>        Merge JSON object into a saved benchmark run
  --patch <path>           Merge JSON object file into a saved benchmark run
  --unset <fields>         Comma-separated saved-run fields to remove
  --yes                    Confirm saved-run deletion
  --json-status            Emit progress events as JSON lines on stderr
  --quiet                  Suppress progress events
  --hardware <path>        JSON hardware object required when submitting
  --quantization <label>   Quantization label
  --results <path>         Existing lm-eval output JSON for run upload
  --kind <kind>            Storage upload kind, usually artifact or dataset
  --format <format>        Storage file format, e.g. json, jsonl, parquet, zip
  --item-count <n>         Optional record/sample count for storage metadata
  --limit <n>              Optional search/list result limit
  --submit                 Upload run to LocalMaxxing
  --dry-run                For benchmark run: write a measurement plan; for submit commands: authenticated API validation without creating a run
  --out <path>             Write computed payload/result JSON`)
}
