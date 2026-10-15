package main

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"text/tabwriter"
	"time"
)

// `lmx model load` starts a local model on the engine that fits its format
// (GGUF → llama.cpp, safetensors → vLLM, else SGLang) as a background server,
// waits until it answers /v1/models, and records it; `unload` stops it and
// `loaded` lists what runs. Engines are the host's own (`lmx engines set` or
// PATH); managed installations keep using `lmx server`.

// ── Engine configuration (~/.config/localmaxxing/engines.json) ──────────────

type hostEngineConfig struct {
	Path string            `json:"path"`          // vllm binary, sglang python, or llama-server
	Env  map[string]string `json:"env,omitempty"` // extra environment for this engine's servers
}

func hostEnginesFile() string {
	return filepath.Join(filepath.Dir(configFile()), "engines.json")
}

func loadHostEngines() map[string]hostEngineConfig {
	engines := map[string]hostEngineConfig{}
	if data, err := os.ReadFile(hostEnginesFile()); err == nil {
		_ = json.Unmarshal(data, &engines)
	}
	return engines
}

func saveHostEngines(engines map[string]hostEngineConfig) error {
	if err := os.MkdirAll(filepath.Dir(hostEnginesFile()), 0o700); err != nil {
		return err
	}
	data, err := json.MarshalIndent(engines, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(hostEnginesFile(), append(data, '\n'), 0o600)
}

var loadableEngines = []string{"vllm", "sglang", "llama.cpp"}

// handleEngineConfig: `lmx engines set <engine> <path> [--server-env-file env.json]` / `unset <engine>`.
func handleEngineConfig(action, engine string, args cliArgs) error {
	engine = normalizeEngineName(engine)
	if !containsString(loadableEngines, engine) {
		return cliError{"invalid_engine", "Engine must be vllm, sglang or llama.cpp.", []string{"lmx engines set vllm ~/vllm/bin/vllm", "lmx engines set sglang ~/sglang/bin/python", "lmx engines set llama.cpp ~/llama.cpp/build/bin/llama-server"}, nil}
	}
	engines := loadHostEngines()
	if action == "unset" {
		delete(engines, engine)
		if err := saveHostEngines(engines); err != nil {
			return err
		}
		printInfo(args, "engine_unset", map[string]any{"engine": engine, "path": hostEnginesFile()})
		return nil
	}
	path := positional(args, 3)
	if path == "" {
		return cliError{"missing_path", "engines set requires the engine executable.", []string{"vllm: the vllm binary; sglang: the Python with sglang installed; llama.cpp: llama-server."}, nil}
	}
	absolute, err := filepath.Abs(expandHome(path))
	if err != nil {
		return err
	}
	if info, err := os.Stat(absolute); err != nil || info.IsDir() || info.Mode()&0o111 == 0 {
		return cliError{"engine_not_executable", absolute + " is not an executable file.", nil, nil}
	}
	config := hostEngineConfig{Path: absolute}
	if envFile := opt(args, "server-env-file"); envFile != "" {
		data, err := os.ReadFile(envFile)
		if err != nil {
			return err
		}
		if err := json.Unmarshal(data, &config.Env); err != nil {
			return errors.New("--server-env-file must be a JSON object of string values")
		}
	}
	engines[engine] = config
	if err := saveHostEngines(engines); err != nil {
		return err
	}
	envKeys := make([]string, 0, len(config.Env))
	for key := range config.Env {
		envKeys = append(envKeys, key)
	}
	sort.Strings(envKeys)
	printInfo(args, "engine_set", map[string]any{"engine": engine, "path": absolute, "env": strings.Join(envKeys, ","), "saved": hostEnginesFile()})
	return nil
}

// hostEngineExecutable resolves an engine's executable: configured, else PATH.
func hostEngineExecutable(engine string) (string, map[string]string, bool) {
	if config, ok := loadHostEngines()[engine]; ok && config.Path != "" {
		return config.Path, config.Env, true
	}
	switch engine {
	case "vllm":
		if path, ok := lookupExecutable("vllm"); ok {
			return path, nil, true
		}
	case "llama.cpp":
		if path, ok := lookupExecutable("llama-server"); ok {
			return path, nil, true
		}
	case "sglang":
		if path, ok := lookupExecutable("python3"); ok && hasPythonModule(path, "sglang") {
			return path, nil, true
		}
	}
	return "", nil, false
}

// chooseLoadEngine picks the engine for a model: --engine, else GGUF →
// llama.cpp and safetensors/PyTorch → vLLM, falling back to SGLang.
func chooseLoadEngine(entry modelInventoryEntry, requested string) (string, error) {
	if requested != "" {
		engine := normalizeEngineName(requested)
		if !containsString(loadableEngines, engine) {
			return "", fmt.Errorf("--engine must be vllm, sglang or llama.cpp")
		}
		if engine == "llama.cpp" && entry.Format != "gguf" {
			return "", fmt.Errorf("llama.cpp serves GGUF files; %s is %s", entry.Name, dash(entry.Format))
		}
		if engine != "llama.cpp" && entry.Format == "gguf" {
			return "", fmt.Errorf("%s does not serve GGUF here; load it with llama.cpp", engine)
		}
		return engine, nil
	}
	switch entry.Format {
	case "gguf":
		return "llama.cpp", nil
	case "safetensors", "pytorch", "mixed":
		if _, _, ok := hostEngineExecutable("vllm"); ok {
			return "vllm", nil
		}
		if _, _, ok := hostEngineExecutable("sglang"); ok {
			return "sglang", nil
		}
		return "vllm", nil // reported as missing below, with setup hints
	}
	return "", fmt.Errorf("%s has no loadable weights (format %s)", entry.Name, dash(entry.Format))
}

// jsonStringArrayOption parses an option holding a JSON array of strings, such
// as --server-args '["--gpu-memory-utilization","0.5"]'.
func jsonStringArrayOption(args cliArgs, key string) ([]string, error) {
	value := opt(args, key)
	if value == "" {
		return nil, nil
	}
	var result []string
	if err := json.Unmarshal([]byte(value), &result); err != nil || result == nil {
		return nil, fmt.Errorf("--%s must be a JSON string array", key)
	}
	for _, item := range result {
		if strings.ContainsRune(item, 0) {
			return nil, fmt.Errorf("--%s contains a NUL byte", key)
		}
	}
	return result, nil
}

var engineDefaultPorts = map[string]int{"vllm": 8000, "sglang": 30000, "llama.cpp": 8080}

// loadServerArgv builds the exact server argv; no shell is involved.
func loadServerArgv(engine, executable, modelPath, served, host string, port int, args cliArgs, gpuCount int) ([]string, error) {
	extra, err := jsonStringArrayOption(args, "server-args")
	if err != nil {
		return nil, err
	}
	portText := strconv.Itoa(port)
	var argv []string
	switch engine {
	case "vllm":
		argv = []string{executable, "serve", modelPath, "--host", host, "--port", portText, "--served-model-name", served}
		if value := opt(args, "context-length"); value != "" {
			argv = append(argv, "--max-model-len", value)
		}
		if gpuCount > 1 && !containsString(extra, "--tensor-parallel-size") && !containsString(extra, "-tp") {
			argv = append(argv, "--tensor-parallel-size", strconv.Itoa(gpuCount))
		}
	case "sglang":
		argv = []string{executable, "-m", "sglang.launch_server", "--model-path", modelPath, "--host", host, "--port", portText, "--served-model-name", served}
		if value := opt(args, "context-length"); value != "" {
			argv = append(argv, "--context-length", value)
		}
		if gpuCount > 1 && !containsString(extra, "--tp") && !containsString(extra, "--tp-size") {
			argv = append(argv, "--tp", strconv.Itoa(gpuCount))
		}
	case "llama.cpp":
		argv = []string{executable, "-m", modelPath, "--host", host, "--port", portText, "--alias", served}
		if value := opt(args, "context-length"); value != "" {
			argv = append(argv, "-c", value)
		}
		if !containsString(extra, "-ngl") && !containsString(extra, "--n-gpu-layers") && !containsString(extra, "--gpu-layers") {
			argv = append(argv, "-ngl", firstNonEmpty(opt(args, "gpu-layers"), "999"))
		}
	default:
		return nil, fmt.Errorf("unsupported engine %s", engine)
	}
	return append(argv, extra...), nil
}

// ── Loaded-model registry ($XDG_DATA_HOME/localmaxxing/loaded) ──────────────

type loadedModel struct {
	Name            string   `json:"name"`
	ModelID         string   `json:"modelId"`
	ModelName       string   `json:"modelName"`
	ModelPath       string   `json:"modelPath"`
	Engine          string   `json:"engine"`
	Argv            []string `json:"argv"`
	EnvKeys         []string `json:"envKeys,omitempty"`
	GPUs            string   `json:"gpus,omitempty"`
	Host            string   `json:"host"`
	Port            int      `json:"port"`
	BaseURL         string   `json:"baseUrl"`
	ServedModel     string   `json:"servedModel"`
	PID             int      `json:"pid"`
	ProcessIdentity string   `json:"processIdentity"`
	LogPath         string   `json:"logPath"`
	StartedAt       string   `json:"startedAt"`
	ReadyAt         string   `json:"readyAt,omitempty"`
}

func loadedRoot() (string, error) {
	base := os.Getenv("XDG_DATA_HOME")
	if base == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return "", err
		}
		base = filepath.Join(home, ".local", "share")
	}
	return filepath.Join(base, "localmaxxing", "loaded"), nil
}

var loadedNamePattern = regexp.MustCompile(`^[a-z0-9][a-z0-9._-]{0,63}$`)

func loadedName(value string) string {
	value = strings.ToLower(value)
	if slash := strings.LastIndex(value, "/"); slash >= 0 && slash < len(value)-1 {
		value = value[slash+1:]
	}
	value = regexp.MustCompile(`[^a-z0-9._-]+`).ReplaceAllString(value, "-")
	value = strings.Trim(value, "-._")
	if len(value) > 64 {
		value = value[:64]
	}
	if value == "" {
		value = "model"
	}
	return value
}

func readLoadedModels() ([]loadedModel, error) {
	root, err := loadedRoot()
	if err != nil {
		return nil, err
	}
	files, err := filepath.Glob(filepath.Join(root, "*.json"))
	if err != nil {
		return nil, err
	}
	models := []loadedModel{}
	for _, file := range files {
		var model loadedModel
		data, err := os.ReadFile(file)
		if err != nil || json.Unmarshal(data, &model) != nil || !loadedNamePattern.MatchString(model.Name) {
			continue
		}
		models = append(models, model)
	}
	sort.Slice(models, func(i, j int) bool { return models[i].Name < models[j].Name })
	return models, nil
}

func writeLoadedModel(model loadedModel) error {
	root, err := loadedRoot()
	if err != nil {
		return err
	}
	if err := os.MkdirAll(root, 0o700); err != nil {
		return err
	}
	data, err := json.MarshalIndent(model, "", "  ")
	if err != nil {
		return err
	}
	tmp := filepath.Join(root, "."+model.Name+".json.tmp")
	if err := os.WriteFile(tmp, append(data, '\n'), 0o600); err != nil {
		return err
	}
	return os.Rename(tmp, filepath.Join(root, model.Name+".json"))
}

func removeLoadedModel(name string) error {
	root, err := loadedRoot()
	if err != nil {
		return err
	}
	if err := os.Remove(filepath.Join(root, name+".json")); err != nil && !os.IsNotExist(err) {
		return err
	}
	return nil
}

func (m loadedModel) alive() bool {
	return m.PID > 0 && processMatchesIdentity(m.PID, m.ProcessIdentity)
}

// probeLoaded reports "ready" when the server lists models, "loading" while
// its process runs, and "exited" once it is gone.
func probeLoaded(m loadedModel, timeout time.Duration) string {
	if !m.alive() {
		return "exited"
	}
	client := &http.Client{Timeout: timeout}
	response, err := client.Get(m.BaseURL + "/v1/models")
	if err != nil {
		return "loading"
	}
	defer response.Body.Close()
	var body struct {
		Data []any `json:"data"`
	}
	if response.StatusCode == http.StatusOK && json.NewDecoder(io.LimitReader(response.Body, 1<<20)).Decode(&body) == nil && len(body.Data) > 0 {
		return "ready"
	}
	return "loading"
}

func portFree(host string, port int, taken map[int]bool) bool {
	if taken[port] {
		return false
	}
	listener, err := net.Listen("tcp", net.JoinHostPort(host, strconv.Itoa(port)))
	if err != nil {
		return false
	}
	listener.Close()
	return true
}

func tailFile(path string, lines int) []string {
	file, err := os.Open(path)
	if err != nil {
		return nil
	}
	defer file.Close()
	if info, err := file.Stat(); err == nil && info.Size() > 1<<20 {
		file.Seek(-1<<20, io.SeekEnd)
	}
	var all []string
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 64<<10), 1<<20)
	for scanner.Scan() {
		all = append(all, scanner.Text())
	}
	if len(all) > lines {
		all = all[len(all)-lines:]
	}
	return all
}

// ── Commands ────────────────────────────────────────────────────────────────

func resolveInventoryModel(target string, args cliArgs) (modelInventoryEntry, error) {
	if target == "" {
		return modelInventoryEntry{}, cliError{"missing_model", "A model name, ID or path is required.", []string{"Run lmx model list to see local models."}, nil}
	}
	_, entries, err := modelInventory(args)
	if err != nil {
		return modelInventoryEntry{}, err
	}
	matches := matchInventory(entries, target)
	if len(matches) == 0 {
		return modelInventoryEntry{}, cliError{"model_not_found", "No local model matches " + target + ".", []string{"Run lmx model list (add --scan-dir <dir> for plain model folders)."}, nil}
	}
	if len(matches) > 1 {
		ids := []string{}
		for _, match := range matches {
			ids = append(ids, match.ID)
		}
		return modelInventoryEntry{}, cliError{"ambiguous_model", target + " matches several local models; pass an ID.", ids, nil}
	}
	return matches[0], nil
}

func handleModelLoad(target string, args cliArgs) error {
	entry, err := resolveInventoryModel(target, args)
	if err != nil {
		return err
	}
	if entry.Status != "available" && entry.Status != "ready" {
		return cliError{"model_not_loadable", entry.Name + " is " + entry.Status + ".", []string{firstNonEmpty(entry.SourceError, "Run lmx model show "+target+".")}, nil}
	}
	engine, err := chooseLoadEngine(entry, opt(args, "engine"))
	if err != nil {
		return cliError{"engine_not_supported", err.Error(), nil, nil}
	}
	executable, engineEnv, ok := hostEngineExecutable(engine)
	if !ok {
		hint := map[string]string{"vllm": "lmx engines set vllm /path/to/venv/bin/vllm", "sglang": "lmx engines set sglang /path/to/venv/bin/python", "llama.cpp": "lmx engines set llama.cpp /path/to/llama-server"}[engine]
		return cliError{"engine_not_found", engine + " was not found for " + entry.Name + ".", []string{"Point lmx at it once: " + hint, "Or pick another engine with --engine."}, nil}
	}
	name := opt(args, "name")
	if name == "" {
		name = loadedName(entry.Name)
	}
	if !loadedNamePattern.MatchString(name) {
		return cliError{"invalid_name", "--name must be lowercase letters, digits, '.', '_' or '-' (max 64).", nil, nil}
	}
	loaded, err := readLoadedModels()
	if err != nil {
		return err
	}
	taken := map[int]bool{}
	for _, other := range loaded {
		running := other.alive()
		if other.Name == name {
			if running {
				if other.ModelPath == entry.Path {
					printInfo(args, "model_already_loaded", map[string]any{"name": other.Name, "baseUrl": other.BaseURL, "servedModel": other.ServedModel, "status": probeLoaded(other, 2*time.Second)})
					return nil
				}
				return cliError{"name_in_use", "A different model is loaded as " + name + ".", []string{"Pass --name, or lmx model unload " + name + " first."}, nil}
			}
			_ = removeLoadedModel(other.Name)
			continue
		}
		if running {
			taken[other.Port] = true
		}
	}
	host := firstNonEmpty(opt(args, "host"), "127.0.0.1")
	port := 0
	if value := opt(args, "port"); value != "" {
		port, err = strconv.Atoi(value)
		if err != nil || port < 1 || port > 65535 {
			return cliError{"invalid_option", "--port must be 1-65535.", nil, nil}
		}
		if !portFree(host, port, taken) {
			return cliError{"port_in_use", fmt.Sprintf("Port %d is already in use.", port), []string{"Omit --port to pick a free one."}, nil}
		}
	} else {
		for candidate := engineDefaultPorts[engine]; candidate < engineDefaultPorts[engine]+100; candidate++ {
			if portFree(host, candidate, taken) {
				port = candidate
				break
			}
		}
		if port == 0 {
			return cliError{"port_in_use", "No free port near the engine default.", []string{"Pass --port."}, nil}
		}
	}
	gpus := opt(args, "gpus")
	gpuCount := 0
	if gpus != "" {
		gpuCount = len(strings.Split(gpus, ","))
	}
	// Clients address the server by this name: the repo, or for a GGUF its file stem.
	served := firstNonEmpty(entry.Repo, entry.Name)
	if entry.Format == "gguf" {
		served = entry.Name[strings.LastIndex(entry.Name, "/")+1:]
	}
	served = firstNonEmpty(opt(args, "served-model"), served)
	modelPath := entry.Path
	argv, err := loadServerArgv(engine, executable, modelPath, served, host, port, args, gpuCount)
	if err != nil {
		return err
	}
	probeHost := host
	if host == "0.0.0.0" || host == "::" {
		probeHost = "127.0.0.1"
	}
	record := loadedModel{
		Name: name, ModelID: entry.ID, ModelName: entry.Name, ModelPath: modelPath, Engine: engine, Argv: argv, GPUs: gpus,
		Host: host, Port: port, BaseURL: "http://" + net.JoinHostPort(probeHost, strconv.Itoa(port)), ServedModel: served,
		StartedAt: time.Now().UTC().Format(time.RFC3339),
	}
	environment := os.Environ()
	extraEnv := map[string]string{}
	for key, value := range engineEnv {
		extraEnv[key] = value
	}
	if envFile := opt(args, "server-env-file"); envFile != "" {
		data, err := os.ReadFile(envFile)
		if err != nil {
			return err
		}
		var values map[string]string
		if err := json.Unmarshal(data, &values); err != nil {
			return errors.New("--server-env-file must be a JSON object of string values")
		}
		for key, value := range values {
			extraEnv[key] = value
		}
	}
	if gpus != "" {
		extraEnv["CUDA_VISIBLE_DEVICES"] = gpus
	}
	for key, value := range extraEnv {
		environment = append(environment, key+"="+value)
		record.EnvKeys = append(record.EnvKeys, key)
	}
	sort.Strings(record.EnvKeys)
	environment = withHuggingFaceToken(environment)

	if hasFlag(args, "dry-run") {
		return writeOrPrintJSON("model_load_plan", args, record)
	}
	root, err := loadedRoot()
	if err != nil {
		return err
	}
	if err := os.MkdirAll(root, 0o700); err != nil {
		return err
	}
	record.LogPath = filepath.Join(root, name+".log")
	logFile, err := os.OpenFile(record.LogPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o600)
	if err != nil {
		return err
	}
	cmd := exec.Command(argv[0], argv[1:]...)
	cmd.Env, cmd.Stdout, cmd.Stderr, cmd.Stdin = environment, logFile, logFile, nil
	configureDetachedCommand(cmd) // own session: outlives lmx, stopped as a group
	if err := cmd.Start(); err != nil {
		logFile.Close()
		return fmt.Errorf("start %s: %w", engine, err)
	}
	logFile.Close()
	exited := make(chan error, 1)
	go func() { exited <- cmd.Wait() }()
	record.PID = cmd.Process.Pid
	if record.ProcessIdentity, err = captureProcessIdentity(record.PID); err != nil {
		_ = signalDetachedProcess(record.PID, true)
		return fmt.Errorf("could not record the server process: %w", err)
	}
	if err := writeLoadedModel(record); err != nil {
		_ = signalDetachedProcess(record.PID, true)
		return err
	}
	printStatus(args, "model_loading", map[string]any{"name": name, "model": entry.Name, "engine": engine, "baseUrl": record.BaseURL, "log": record.LogPath})
	if hasFlag(args, "no-wait") {
		return finishModelLoad(args, record, "loading")
	}

	timeout, err := intOption(args, 900, 1, "ready-timeout-seconds")
	if err != nil {
		return err
	}
	ctx, stop := terminalSignalContext()
	defer stop()
	deadline := time.Now().Add(time.Duration(timeout) * time.Second)
	fail := func(reason string) error {
		_ = stopLoadedModel(record, 30*time.Second)
		_ = removeLoadedModel(record.Name)
		return cliError{"model_load_failed", reason, append([]string{"Server log " + record.LogPath + " (last lines):"}, tailFile(record.LogPath, 25)...), nil}
	}
	for {
		select {
		case err := <-exited:
			return fail(fmt.Sprintf("%s exited before it was ready (%v).", engine, err))
		case <-ctx.Done():
			return fail("Interrupted; the server was stopped.")
		case <-time.After(time.Second):
		}
		if probeLoaded(record, 2*time.Second) == "ready" {
			break
		}
		if time.Now().After(deadline) {
			return fail(fmt.Sprintf("%s was not ready after %d s; pass --ready-timeout-seconds for slow first starts.", engine, timeout))
		}
	}
	record.ReadyAt = time.Now().UTC().Format(time.RFC3339)
	if err := writeLoadedModel(record); err != nil {
		return err
	}
	return finishModelLoad(args, record, "ready")
}

func finishModelLoad(args cliArgs, record loadedModel, status string) error {
	if !humanOutput(args) || opt(args, "out") != "" {
		return writeOrPrintJSON("model_load", args, map[string]any{"status": status, "loaded": record})
	}
	fmt.Printf("%s %s on %s → %s (served as %q)\n", map[string]string{"ready": "Loaded", "loading": "Loading"}[status], record.ModelName, record.Engine, record.BaseURL+"/v1", record.ServedModel)
	fmt.Printf("  try:    curl %s/v1/chat/completions -H 'Content-Type: application/json' -d '{\"model\":%q,\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}]}'\n", record.BaseURL, record.ServedModel)
	fmt.Printf("  logs:   lmx model logs %s\n  unload: lmx model unload %s\n", record.Name, record.Name)
	return nil
}

// stopLoadedModel terminates the server's process group, escalating to SIGKILL.
func stopLoadedModel(m loadedModel, grace time.Duration) error {
	if !m.alive() {
		return nil
	}
	if err := signalDetachedProcess(m.PID, false); err != nil {
		return err
	}
	deadline := time.Now().Add(grace)
	for time.Now().Before(deadline) {
		if !m.alive() {
			return nil
		}
		time.Sleep(200 * time.Millisecond)
	}
	if err := signalDetachedProcess(m.PID, true); err != nil {
		return err
	}
	for i := 0; i < 50 && m.alive(); i++ {
		time.Sleep(100 * time.Millisecond)
	}
	if m.alive() {
		return fmt.Errorf("%s (pid %d) did not exit", m.Name, m.PID)
	}
	return nil
}

func selectLoaded(loaded []loadedModel, target string) []loadedModel {
	var matches []loadedModel
	for _, m := range loaded {
		if target == m.Name || strings.EqualFold(target, m.ModelName) || target == m.ModelID || target == m.ModelPath || target == strconv.Itoa(m.Port) || target == m.ServedModel {
			matches = append(matches, m)
		}
	}
	return matches
}

func handleModelUnload(target string, args cliArgs) error {
	loaded, err := readLoadedModels()
	if err != nil {
		return err
	}
	targets := loaded
	if !hasFlag(args, "all") {
		if target == "" {
			return cliError{"missing_model", "model unload requires a loaded name, model or port (or --all).", []string{"Run lmx model loaded."}, nil}
		}
		targets = selectLoaded(loaded, target)
		if len(targets) == 0 {
			return cliError{"model_not_loaded", target + " is not loaded.", []string{"Run lmx model loaded."}, nil}
		}
	}
	grace, err := intOption(args, 30, 1, "stop-timeout-seconds")
	if err != nil {
		return err
	}
	unloaded := []string{}
	for _, m := range targets {
		if err := stopLoadedModel(m, time.Duration(grace)*time.Second); err != nil {
			return err
		}
		if err := removeLoadedModel(m.Name); err != nil {
			return err
		}
		unloaded = append(unloaded, m.Name)
	}
	printInfo(args, "model_unloaded", map[string]any{"unloaded": strings.Join(unloaded, ", ")})
	return nil
}

type loadedStatus struct {
	loadedModel
	Status string `json:"status"`
}

func handleModelLoaded(args cliArgs) error {
	loaded, err := readLoadedModels()
	if err != nil {
		return err
	}
	rows := make([]loadedStatus, 0, len(loaded))
	for _, m := range loaded {
		rows = append(rows, loadedStatus{m, probeLoaded(m, 2*time.Second)})
	}
	if !humanOutput(args) || opt(args, "out") != "" {
		return writeOrPrintJSON("model_loaded", args, map[string]any{"loaded": rows})
	}
	if len(rows) == 0 {
		fmt.Println("No models loaded. Load one with lmx model load <name>.")
		return nil
	}
	writer := tabwriter.NewWriter(os.Stdout, 0, 4, 2, ' ', 0)
	fmt.Fprintln(writer, "NAME\tMODEL\tENGINE\tURL\tSTATUS\tPID\tUP")
	for _, row := range rows {
		up := "-"
		if started, err := time.Parse(time.RFC3339, row.StartedAt); err == nil && row.Status != "exited" {
			up = time.Since(started).Round(time.Second).String()
		}
		fmt.Fprintf(writer, "%s\t%s\t%s\t%s\t%s\t%d\t%s\n", row.Name, row.ModelName, row.Engine, row.BaseURL+"/v1", row.Status, row.PID, up)
	}
	writer.Flush()
	return nil
}

func handleModelLogs(target string, args cliArgs) error {
	loaded, err := readLoadedModels()
	if err != nil {
		return err
	}
	matches := selectLoaded(loaded, target)
	if len(matches) != 1 {
		return cliError{"model_not_loaded", "No single loaded model matches " + target + ".", []string{"Run lmx model loaded."}, nil}
	}
	lines, err := intOption(args, 50, 1, "lines")
	if err != nil {
		return err
	}
	for _, line := range tailFile(matches[0].LogPath, lines) {
		fmt.Println(line)
	}
	if !hasFlag(args, "follow") {
		return nil
	}
	file, err := os.Open(matches[0].LogPath)
	if err != nil {
		return err
	}
	defer file.Close()
	file.Seek(0, io.SeekEnd)
	ctx, stop := terminalSignalContext()
	defer stop()
	reader := bufio.NewReader(file)
	for ctx.Err() == nil {
		line, err := reader.ReadString('\n')
		if line != "" {
			fmt.Print(line)
		}
		if err != nil {
			if !matches[0].alive() {
				return nil
			}
			select {
			case <-ctx.Done():
			case <-time.After(500 * time.Millisecond):
			}
		}
	}
	return nil
}

// matchInventory selects entries by exact ID, then name/path/repo, then ID prefix.
func matchInventory(entries []modelInventoryEntry, target string) []modelInventoryEntry {
	var matches []modelInventoryEntry
	for _, pass := range []func(modelInventoryEntry) bool{
		func(e modelInventoryEntry) bool { return e.ID == target },
		func(e modelInventoryEntry) bool {
			return strings.EqualFold(e.Name, target) || e.Path == target || strings.EqualFold(e.Repo, target)
		},
		func(e modelInventoryEntry) bool { return len(target) >= 6 && strings.HasPrefix(e.ID, target) },
	} {
		for _, entry := range entries {
			if pass(entry) {
				matches = append(matches, entry)
			}
		}
		if len(matches) > 0 {
			return matches
		}
	}
	return nil
}
