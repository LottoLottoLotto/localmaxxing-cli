package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"text/tabwriter"
	"time"
)

// lmx serves models through emx (github.com/LottoLottoLotto/endpointmaxxing):
// one OpenAI/Anthropic-compatible gateway that routes every request by its
// `model` field to the right engine, starts registered models on demand and
// places them on GPUs and Tenstorrent chips. lmx drives the `emx`
// binary and the gateway's management API; it runs no engines itself.

const defaultGatewayURL = "http://127.0.0.1:7777"

// emxBin finds the emx executable: --emx-bin,
// LMX_EMX_BIN, then PATH.
func emxBin(args cliArgs) (string, error) {
	if path := firstNonEmpty(opt(args, "emx-bin"), os.Getenv("LMX_EMX_BIN")); path != "" {
		return expandHome(path), nil
	}
	if path, ok := lookupExecutable("emx"); ok {
		return path, nil
	}
	return "", cliError{"emx_not_found", "emx was not found.", []string{
		"Install it (github.com/LottoLottoLotto/endpointmaxxing): go build -o ~/.local/bin/emx ./cmd/emx",
		"Or point lmx at it with --emx-bin <path> or LMX_EMX_BIN.",
	}, nil}
}

// runEMX runs `emx <argv...> --json` and returns stdout. Failures
// become cliErrors carrying emx's stable code (not_found, conflict,
// capacity, …) as emx_<code>.
func runEMX(args cliArgs, argv ...string) ([]byte, error) {
	bin, err := emxBin(args)
	if err != nil {
		return nil, err
	}
	full := append([]string{}, argv...)
	if sep := indexOf(full, "--"); sep >= 0 {
		full = append(append(append([]string{}, full[:sep]...), "--json"), full[sep:]...)
	} else {
		full = append(full, "--json")
	}
	cmd := exec.Command(bin, full...)
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	if err := cmd.Run(); err != nil {
		var failure struct {
			Code  string `json:"code"`
			Error string `json:"error"`
		}
		for _, output := range [][]byte{stdout.Bytes(), stderr.Bytes()} {
			if json.Unmarshal(bytes.TrimSpace(output), &failure) == nil && failure.Error != "" {
				return nil, cliError{"emx_" + firstNonEmpty(failure.Code, "error"), failure.Error, nil, map[string]any{"command": append([]string{"emx"}, argv...)}}
			}
		}
		message := strings.TrimSpace(firstNonEmpty(stderr.String(), stdout.String(), err.Error()))
		return nil, cliError{"emx_error", "emx " + strings.Join(argv, " ") + ": " + message, nil, nil}
	}
	return stdout.Bytes(), nil
}

func indexOf(values []string, target string) int {
	for i, value := range values {
		if value == target {
			return i
		}
	}
	return -1
}

// ── Gateway address and API ─────────────────────────────────────────────────

// gatewayURL is the gateway base URL (no /v1): --gateway-url, LMX_GATEWAY_URL,
// else the gateway host/port in emx's config.yaml, else 127.0.0.1:7777.
func gatewayURL(args cliArgs) string {
	if value := firstNonEmpty(opt(args, "gateway-url"), os.Getenv("LMX_GATEWAY_URL")); value != "" {
		return openAIBaseURL(value)
	}
	host, port := emxGatewayConfig()
	if host == "" && port == "" {
		return defaultGatewayURL
	}
	if host == "" || host == "0.0.0.0" || host == "::" {
		host = "127.0.0.1"
	}
	if port == "" {
		port = "7777"
	}
	return "http://" + net.JoinHostPort(host, port)
}

// emxConfigPath mirrors emx's own lookup (EMX_HOME, then XDG).
func emxConfigPath() string {
	if root := os.Getenv("EMX_HOME"); root != "" {
		return filepath.Join(root, "config", "config.yaml")
	}
	base := os.Getenv("XDG_CONFIG_HOME")
	if base == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return ""
		}
		base = filepath.Join(home, ".config")
	}
	return filepath.Join(base, "emx", "config.yaml")
}

// emxGatewayConfig reads gateway.host and gateway.port from config.yaml.
// Only those two scalars of the top-level gateway block are needed, so a line
// scan replaces a YAML dependency.
func emxGatewayConfig() (host, port string) {
	file, err := os.Open(emxConfigPath())
	if err != nil {
		return "", ""
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)
	inGateway := false
	for scanner.Scan() {
		line := scanner.Text()
		if trimmed := strings.TrimSpace(line); trimmed == "" || strings.HasPrefix(trimmed, "#") {
			continue
		}
		if !strings.HasPrefix(line, " ") && !strings.HasPrefix(line, "\t") {
			inGateway = strings.HasPrefix(line, "gateway:")
			continue
		}
		if !inGateway {
			continue
		}
		key, value, ok := strings.Cut(strings.TrimSpace(line), ":")
		if !ok {
			continue
		}
		value = strings.Trim(strings.TrimSpace(strings.SplitN(value, " #", 2)[0]), `"'`)
		switch key {
		case "host":
			host = value
		case "port":
			port = value
		}
	}
	return host, port
}

func gatewayUp(base string) bool {
	client := &http.Client{Timeout: 2 * time.Second}
	response, err := client.Get(base + "/health")
	if err != nil {
		return false
	}
	response.Body.Close()
	return response.StatusCode == http.StatusOK
}

// gatewayCall sends a management API request. EMX_API_KEY authenticates
// when the gateway has keys (management needs an admin key).
func gatewayCall(base, method, path string, body any, out any, timeout time.Duration) error {
	var reader io.Reader
	if body != nil {
		data, err := json.Marshal(body)
		if err != nil {
			return err
		}
		reader = bytes.NewReader(data)
	}
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, method, base+path, reader)
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	if key := os.Getenv("EMX_API_KEY"); key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}
	response, err := (&http.Client{}).Do(req)
	if err != nil {
		return cliError{"gateway_unreachable", "The emx gateway at " + base + " did not answer: " + err.Error(), []string{"Start it with lmx gateway start."}, nil}
	}
	defer response.Body.Close()
	data, _ := io.ReadAll(io.LimitReader(response.Body, 8<<20))
	if response.StatusCode >= 300 {
		var failure struct {
			Error struct {
				Message string `json:"message"`
				Type    string `json:"type"`
				Code    string `json:"code"`
			} `json:"error"`
		}
		message := strings.TrimSpace(string(data))
		if json.Unmarshal(data, &failure) == nil && failure.Error.Message != "" {
			message = failure.Error.Message
		}
		hints := []string{}
		if response.StatusCode == http.StatusUnauthorized || response.StatusCode == http.StatusForbidden {
			hints = append(hints, "The gateway has API keys: export EMX_API_KEY=<admin key>.")
		}
		return cliError{"gateway_" + strings.ToLower(strings.ReplaceAll(http.StatusText(response.StatusCode), " ", "_")), fmt.Sprintf("gateway %s %s: %s", method, path, message), hints, map[string]any{"status": response.StatusCode}}
	}
	if out == nil {
		return nil
	}
	return json.Unmarshal(data, out)
}

type emxInstance struct {
	ID          string     `json:"id"`
	Name        string     `json:"name"`
	Engine      string     `json:"engine"`
	Model       string     `json:"model"`
	Format      string     `json:"format"`
	PID         int        `json:"pid"`
	BaseURL     string     `json:"base_url"`
	Owner       string     `json:"owner"`
	Status      string     `json:"status"`
	StartedAt   time.Time  `json:"started_at"`
	ReadyAt     *time.Time `json:"ready_at"`
	GPUs        []int      `json:"gpus"`
	Accelerator string     `json:"accelerator"`
	Devices     []int      `json:"devices"`
}

type emxModel struct {
	Name   string `json:"name"`
	Source string `json:"source"`
	Format string `json:"format"`
	Arch   string `json:"arch"`
	Quant  string `json:"quant"`
	Engine string `json:"engine"`
}

func emxInstances(args cliArgs) ([]emxInstance, error) {
	out, err := runEMX(args, "ps")
	if err != nil {
		return nil, err
	}
	var instances []emxInstance
	if err := json.Unmarshal(out, &instances); err != nil {
		return nil, fmt.Errorf("emx ps: %w", err)
	}
	return instances, nil
}

func emxModels(args cliArgs) ([]emxModel, error) {
	out, err := runEMX(args, "model", "list")
	if err != nil {
		return nil, err
	}
	var models []emxModel
	if err := json.Unmarshal(out, &models); err != nil {
		return nil, fmt.Errorf("emx model list: %w", err)
	}
	return models, nil
}

// gatewayEnsure makes the gateway start (or reuse) a ready instance of model.
// The gateway owns it, so placement may evict it later and idle unload applies.
func gatewayEnsure(base, model string, timeout time.Duration) (emxInstance, error) {
	var result struct {
		Instance emxInstance `json:"instance"`
	}
	body := map[string]any{"model": model, "timeout": timeout.String()}
	err := gatewayCall(base, http.MethodPost, "/emx/v1/ensure", body, &result, timeout+time.Minute)
	return result.Instance, err
}

// ── Gateway process ─────────────────────────────────────────────────────────

type gatewayRecord struct {
	PID             int    `json:"pid"`
	ProcessIdentity string `json:"processIdentity"`
	URL             string `json:"url"`
	LogPath         string `json:"logPath"`
	StartedAt       string `json:"startedAt"`
}

func gatewayStatePath() (string, error) {
	base := os.Getenv("XDG_DATA_HOME")
	if base == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return "", err
		}
		base = filepath.Join(home, ".local", "share")
	}
	return filepath.Join(base, "localmaxxing", "gateway.json"), nil
}

func readGatewayRecord() (gatewayRecord, bool) {
	var record gatewayRecord
	path, err := gatewayStatePath()
	if err != nil {
		return record, false
	}
	data, err := os.ReadFile(path)
	if err != nil || json.Unmarshal(data, &record) != nil {
		return record, false
	}
	return record, record.PID > 0 && processMatchesIdentity(record.PID, record.ProcessIdentity)
}

// gatewayService reports the emx systemd user service: whether its unit
// is installed and whether it is active.
func gatewayService(args cliArgs) (installed, active bool) {
	out, err := runEMX(args, "service", "status")
	if err != nil {
		return false, false
	}
	var status map[string]any
	if json.Unmarshal(out, &status) != nil {
		return false, false
	}
	return stringValue(status["LoadState"]) == "loaded", stringValue(status["ActiveState"]) == "active"
}

// startGateway runs `emx serve` detached (its own session, so it outlives
// lmx; engines it starts run in their own sessions too) and waits for /health.
func startGateway(args cliArgs) (string, error) {
	base := gatewayURL(args)
	if gatewayUp(base) {
		return base, nil
	}
	if installed, _ := gatewayService(args); installed {
		if out, err := exec.Command("systemctl", "--user", "start", "emx").CombinedOutput(); err != nil {
			return "", cliError{"gateway_start_failed", "systemctl --user start emx: " + strings.TrimSpace(string(out)), nil, nil}
		}
		return base, waitGateway(base, 60*time.Second, nil, "")
	}
	bin, err := emxBin(args)
	if err != nil {
		return "", err
	}
	parsed, err := url.Parse(base)
	if err != nil {
		return "", err
	}
	argv := []string{"serve", "--host", parsed.Hostname(), "--port", parsed.Port()}
	statePath, err := gatewayStatePath()
	if err != nil {
		return "", err
	}
	if err := os.MkdirAll(filepath.Dir(statePath), 0o700); err != nil {
		return "", err
	}
	logPath := filepath.Join(filepath.Dir(statePath), "gateway.log")
	logFile, err := os.OpenFile(logPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600)
	if err != nil {
		return "", err
	}
	cmd := exec.Command(bin, argv...)
	cmd.Stdout, cmd.Stderr, cmd.Stdin = logFile, logFile, nil
	cmd.Env = withHuggingFaceToken(os.Environ())
	configureDetachedCommand(cmd)
	if err := cmd.Start(); err != nil {
		logFile.Close()
		return "", fmt.Errorf("start emx serve: %w", err)
	}
	logFile.Close()
	exited := make(chan error, 1)
	go func() { exited <- cmd.Wait() }()
	identity, err := captureProcessIdentity(cmd.Process.Pid)
	if err != nil {
		_ = signalDetachedProcess(cmd.Process.Pid, true)
		return "", fmt.Errorf("could not record the gateway process: %w", err)
	}
	record := gatewayRecord{PID: cmd.Process.Pid, ProcessIdentity: identity, URL: base, LogPath: logPath, StartedAt: time.Now().UTC().Format(time.RFC3339)}
	data, _ := json.MarshalIndent(record, "", "  ")
	if err := os.WriteFile(statePath, append(data, '\n'), 0o600); err != nil {
		_ = signalDetachedProcess(record.PID, true)
		return "", err
	}
	printStatus(args, "gateway_starting", map[string]any{"url": base, "log": logPath})
	if err := waitGateway(base, 60*time.Second, exited, logPath); err != nil {
		_ = signalDetachedProcess(record.PID, true)
		os.Remove(statePath)
		return "", err
	}
	return base, nil
}

func waitGateway(base string, timeout time.Duration, exited chan error, logPath string) error {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if gatewayUp(base) {
			return nil
		}
		select {
		case err := <-exited:
			return cliError{"gateway_start_failed", fmt.Sprintf("emx serve exited (%v).", err), append([]string{"Gateway log " + logPath + ":"}, tailFile(logPath, 15)...), nil}
		case <-time.After(300 * time.Millisecond):
		}
	}
	return cliError{"gateway_start_failed", "The gateway did not answer " + base + "/health within " + timeout.String() + ".", []string{"Check lmx gateway logs."}, nil}
}

func handleGateway(action string, args cliArgs) error {
	base := gatewayURL(args)
	switch action {
	case "start":
		if hasFlag(args, "service") {
			argv := []string{"service", "install", "--enable"}
			if _, err := runEMX(args, argv...); err != nil {
				return err
			}
			if err := waitGateway(base, 60*time.Second, nil, ""); err != nil {
				return err
			}
			printInfo(args, "gateway_service_started", map[string]any{"url": base + "/v1", "service": "systemctl --user status emx", "boot": "run loginctl enable-linger $USER to keep it running after logout"})
			return nil
		}
		wasUp := gatewayUp(base)
		if _, err := startGateway(args); err != nil {
			return err
		}
		printInfo(args, map[bool]string{true: "gateway_already_running", false: "gateway_started"}[wasUp], map[string]any{"url": base + "/v1", "next": "lmx model load <name>"})
		return nil
	case "stop":
		if record, alive := readGatewayRecord(); alive {
			if err := signalDetachedProcess(record.PID, false); err != nil {
				return err
			}
			for i := 0; i < 400 && processMatchesIdentity(record.PID, record.ProcessIdentity); i++ {
				time.Sleep(100 * time.Millisecond)
			}
			if path, err := gatewayStatePath(); err == nil {
				os.Remove(path)
			}
			printInfo(args, "gateway_stopped", map[string]any{"url": base, "note": "Loaded models keep running; lmx model unload --all stops them."})
			return nil
		}
		if _, active := gatewayService(args); active {
			if out, err := exec.Command("systemctl", "--user", "stop", "emx").CombinedOutput(); err != nil {
				return cliError{"gateway_stop_failed", "systemctl --user stop emx: " + strings.TrimSpace(string(out)), nil, nil}
			}
			printInfo(args, "gateway_stopped", map[string]any{"url": base, "service": "emx", "note": "Loaded models keep running; lmx model unload --all stops them."})
			return nil
		}
		if gatewayUp(base) {
			return cliError{"gateway_not_owned", "A gateway answers at " + base + " but lmx did not start it.", []string{"Stop it where it was started (its terminal, or systemctl --user stop emx)."}, nil}
		}
		printInfo(args, "gateway_not_running", map[string]any{"url": base})
		return nil
	case "status", "":
		return gatewayStatus(args, base)
	case "logs":
		lines, err := intOption(args, 50, 1, "lines")
		if err != nil {
			return err
		}
		if record, ok := readGatewayRecord(); ok || record.LogPath != "" {
			for _, line := range tailFile(record.LogPath, lines) {
				fmt.Println(line)
			}
			return nil
		}
		if installed, _ := gatewayService(args); installed {
			argv := []string{"--user", "-u", "emx", "-n", strconv.Itoa(lines), "--no-pager"}
			if hasFlag(args, "follow") {
				argv = append(argv, "-f")
			}
			cmd := exec.Command("journalctl", argv...)
			cmd.Stdout, cmd.Stderr = os.Stdout, os.Stderr
			return cmd.Run()
		}
		return cliError{"gateway_log_unavailable", "No gateway log: lmx did not start a gateway and no emx service is installed.", nil, nil}
	}
	return cliError{"unknown_command", "Unknown gateway command " + action + ".", []string{"Use lmx gateway start [--service], stop, status or logs."}, nil}
}

func gatewayStatus(args cliArgs, base string) error {
	up := gatewayUp(base)
	managedBy := "not running"
	if record, alive := readGatewayRecord(); alive {
		managedBy = fmt.Sprintf("lmx (pid %d)", record.PID)
	} else if installed, active := gatewayService(args); active {
		managedBy = "systemd user service emx"
	} else if up {
		managedBy = "external"
	} else if installed {
		managedBy = "systemd service installed, inactive"
	}
	models, modelsErr := emxModels(args)
	instances, instancesErr := emxInstances(args)
	status := map[string]any{"url": base + "/v1", "up": up, "managedBy": managedBy, "models": models, "instances": instances}
	if modelsErr != nil {
		status["error"] = modelsErr.Error()
	} else if instancesErr != nil {
		status["error"] = instancesErr.Error()
	}
	if !humanOutput(args) || opt(args, "out") != "" {
		return writeOrPrintJSON("gateway_status", args, status)
	}
	state := map[bool]string{true: "up", false: "down"}[up]
	fmt.Printf("Gateway %s/v1: %s (%s)\n", base, state, managedBy)
	if modelsErr != nil {
		return modelsErr
	}
	running := map[string]emxInstance{}
	for _, instance := range instances {
		running[instance.Name] = instance
	}
	if len(models) == 0 {
		fmt.Println("No models registered. lmx model load <name> registers and starts one.")
		return nil
	}
	writer := tabwriter.NewWriter(os.Stdout, 0, 4, 2, ' ', 0)
	fmt.Fprintln(writer, "\nMODEL\tENGINE\tSTATE\tSOURCE")
	for _, model := range models {
		engine, state := firstNonEmpty(model.Engine, "auto"), "registered (starts on first request)"
		if instance, ok := running[model.Name]; ok {
			engine, state = instance.Engine, firstNonEmpty(instance.Status, "running")
		}
		fmt.Fprintf(writer, "%s\t%s\t%s\t%s\n", model.Name, engine, state, model.Source)
	}
	return writer.Flush()
}

// ── Routing lmx's own traffic through the gateway ───────────────────────────

// gatewayEngineNames maps emx engine ids to LocalMaxxing engine names.
var gatewayEngineNames = map[string]string{"llama.cpp": "llama.cpp", "vllm": "vllm", "tt-vllm": "vllm", "sglang": "sglang", "strata": "strata"}

// gatewayModelFor picks the gateway model that serves hfID: a registered
// model named like it, or one whose source is that repository (or its HF
// cache folder or a folder named after it).
func gatewayModelFor(args cliArgs, hfID string) (string, error) {
	models, err := emxModels(args)
	if err != nil {
		return "", err
	}
	want := strings.ToLower(hfID)
	base := strings.ToLower(hfID[strings.LastIndex(hfID, "/")+1:])
	cacheDir := "models--" + strings.ReplaceAll(want, "/", "--")
	var matches []string
	for _, model := range models {
		source := strings.ToLower(model.Source)
		if strings.ToLower(model.Name) == want || source == want || strings.Contains(source, "/"+cacheDir+"/") || strings.ToLower(filepath.Base(strings.TrimSuffix(source, "/"))) == base || strings.ToLower(model.Name) == base {
			matches = append(matches, model.Name)
		}
	}
	switch len(matches) {
	case 1:
		return matches[0], nil
	case 0:
		return "", cliError{"gateway_model_not_found", "No model registered with the gateway matches " + hfID + ".", []string{"Load it first: lmx model load <name from lmx model list>.", "Or pass --served-model <gateway model name> (lmx gateway status lists them)."}, nil}
	}
	return "", cliError{"gateway_model_ambiguous", hfID + " matches several gateway models: " + strings.Join(matches, ", ") + ".", []string{"Pass --served-model <name>."}, nil}
}

// applyGatewayDefault points commands that need an endpoint at the gateway when
// --base-url is absent and the gateway is up, so all traffic goes through it.
// The served model is resolved and made ready first (no cold start inside a
// measurement). Matrix runs measure the engine instance directly, so they get
// the instance's own URL; everything else gets the gateway's.
func applyGatewayDefault(args *cliArgs) error {
	if opt(*args, "base-url") != "" || hasFlag(*args, "no-gateway") || !gatewayCommand(*args) {
		return nil
	}
	base := gatewayURL(*args)
	if !gatewayUp(base) {
		return nil
	}
	model := firstNonEmpty(opt(*args, "served-model"), opt(*args, "model-name"))
	if model == "" {
		hfID := firstNonEmpty(opt(*args, "hf-id"), opt(*args, "model"))
		if hfID == "" {
			return nil
		}
		resolved, err := gatewayModelFor(*args, hfID)
		if err != nil {
			return err
		}
		model = resolved
	}
	timeout, err := intOption(*args, 900, 1, "ready-timeout-seconds")
	if err != nil {
		return err
	}
	printStatus(*args, "gateway_ensure", map[string]any{"gateway": base + "/v1", "model": model})
	instance, err := gatewayEnsure(base, model, time.Duration(timeout)*time.Second)
	if err != nil {
		return err
	}
	options := make(map[string]string, len(args.opts)+3)
	for key, value := range args.opts {
		options[key] = value
	}
	target := base
	if positional(*args, 0) == "speed-test" && positional(*args, 1) == "matrix" {
		target = strings.TrimSuffix(instance.BaseURL, "/v1")
	}
	options["base-url"], options["served-model"] = target, model
	if engine := gatewayEngineNames[instance.Engine]; engine != "" && opt(*args, "engine") == "" && endpointEngineMissing(*args) {
		options["engine"] = engine
	}
	args.opts = options
	printStatus(*args, "gateway_route", map[string]any{"baseUrl": target, "servedModel": model, "engine": instance.Engine, "instance": instance.ID})
	return nil
}

// gatewayCommand lists the commands that send traffic to an endpoint. Speed
// tests and KV-cache sweeps only in remote mode: local mode runs llama-bench.
func gatewayCommand(args cliArgs) bool {
	first, second, third := positional(args, 0), positional(args, 1), positional(args, 2)
	remote := strings.EqualFold(firstNonEmpty(opt(args, "mode"), opt(args, "benchmark-mode")), "remote")
	switch {
	case first == "speed-test" && second == "matrix":
		return third == "run" && opt(args, "model-path") == ""
	case first == "speed-test" && second == "run", first == "kvcache" && second == "run":
		return remote
	case first == "eval" && (second == "shard" && third != "status" || second == "execute"):
		return true
	case first == "eval" && second == "terminal" && third == "run":
		return true
	case first == "model" && (second == "resolve-remote" || second == "resolve"):
		return true
	}
	return false
}

// endpointEngineMissing is true when no engine was named positionally
// (`lmx speed-test run <engine>`), so the gateway instance's engine applies.
func endpointEngineMissing(args cliArgs) bool {
	switch positional(args, 0) {
	case "speed-test", "kvcache":
		return positional(args, 1) == "run" && positional(args, 2) == ""
	}
	return false
}
