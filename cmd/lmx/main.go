package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	_ "embed"
	"encoding/csv"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
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
	"sync"
	"time"
	"unicode/utf16"
)

const defaultAPIURL = "https://www.localmaxxing.com"
const defaultHFAPIURL = "https://huggingface.co"
const defaultEndpointTimeout = 10 * time.Minute
const remoteKVCacheColdMethodology = "Single streaming request with inline filler padded to target context size; measures cold prefill + decode at that context depth."
const remoteKVCacheReuseMethodology = "Two-step remote cache-reuse probe: pre-warm target context, then time a streaming request with the same prefix plus probe; measures cached-prefix decode at that context depth."
const remoteKVCacheFallbackWarning = "Remote OpenAI-compatible endpoints do not provide a portable persistent KV-cache session API; this sweep resends the full prefix at each depth and can only verify cache reuse when backend-specific cache metrics are exposed. Results may fall back to cold depth TPS instead of retained KV-cache TPS."

var apiHTTPClient = &http.Client{Timeout: 30 * time.Second}

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
	if cmd == "" {
		usage()
		return nil
	}
	if wantsHelp(args) {
		if knownTopLevel(cmd) {
			if text, ok := commandHelp(args); ok {
				fmt.Println(text)
				return nil
			}
		}
		usage()
		return nil
	}
	if !knownTopLevel(cmd) {
		usage()
		return errors.New("unknown command")
	}

	switch cmd {
	case "auth":
		return handleAuth(args)
	case "hardware":
		return handleHardware(positional(args, 1), args)
	case "setups":
		return handleSetups(positional(args, 1), args)
	case "skill":
		return handleSkill(positional(args, 1), args)
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
		case "pull":
			return handleEvalPull(positional(args, 2), args)
		case "submit":
			return handleEvalSubmit(positional(args, 2), args)
		case "shard":
			if positional(args, 2) == "status" {
				return handleEvalShardStatus(positional(args, 3), args)
			}
			return handleEvalShard(positional(args, 2), args)
		case "terminal":
			return handleEvalTerminal(positional(args, 2), args)
		}
	}

	usage()
	return errors.New("unknown command")
}

func knownTopLevel(cmd string) bool {
	switch cmd {
	case "eval", "benchmark", "bench", "auth", "hardware", "setups", "context", "agent-context", "model", "profile", "engines", "engine", "server", "endpoint", "kvcache", "kv-cache", "context-sweep", "skill":
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

func wantsHelp(args cliArgs) bool {
	return hasFlag(args, "help") || sliceContainsString(args.positional, "-h")
}

func sliceContainsString(values []string, target string) bool {
	for _, value := range values {
		if value == target {
			return true
		}
	}
	return false
}

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
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	data, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	return os.WriteFile(path, data, 0o600)
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
	res, err := apiHTTPClient.Do(req)
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
	data, err = decodeJSONBytes(data)
	if err != nil {
		return nil, cliError{"json_parse_error", fmt.Sprintf("Could not parse %s as JSON: %v", path, err), []string{"Save the file as UTF-8 JSON and retry.", "PowerShell 5.1 redirection may write UTF-16; use Set-Content -Encoding utf8 hardware.json."}, nil}
	}
	var value any
	if err := json.Unmarshal(data, &value); err != nil {
		return nil, cliError{"json_parse_error", fmt.Sprintf("Could not parse %s as JSON: %v", path, err), []string{"Fix the JSON syntax and retry."}, nil}
	}
	return value, nil
}

func decodeJSONBytes(data []byte) ([]byte, error) {
	if bytes.HasPrefix(data, []byte{0xEF, 0xBB, 0xBF}) {
		return data[3:], nil
	}
	if bytes.HasPrefix(data, []byte{0xFF, 0xFE}) {
		return decodeUTF16JSONBytes(data[2:], true)
	}
	if bytes.HasPrefix(data, []byte{0xFE, 0xFF}) {
		return decodeUTF16JSONBytes(data[2:], false)
	}
	return data, nil
}

func decodeUTF16JSONBytes(data []byte, littleEndian bool) ([]byte, error) {
	if len(data)%2 != 0 {
		return nil, errors.New("UTF-16 JSON has an odd byte length")
	}
	words := make([]uint16, len(data)/2)
	for i := range words {
		j := i * 2
		if littleEndian {
			words[i] = uint16(data[j]) | uint16(data[j+1])<<8
		} else {
			words[i] = uint16(data[j])<<8 | uint16(data[j+1])
		}
	}
	return []byte(string(utf16.Decode(words))), nil
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
	sub := positional(args, 1)
	switch sub {
	case "login":
		return handleAuthLogin(args)
	case "logout":
		if err := saveConfig(map[string]any{}); err != nil {
			return err
		}
		printInfo("auth_cleared", map[string]any{"path": configFile()})
		return nil
	case "keys":
		return handleAuthKeys(positional(args, 2), positional(args, 3), args)
	case "whoami":
		return handleAuthWhoami(args)
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
		printInfo("auth_missing", map[string]any{"next": "Run lmx auth login, lmx auth --key bhk_..., or set LMX_API_KEY."})
		return nil
	}
	printInfo("auth_status", map[string]any{"source": source, "key": redactKey(key), "provider": cfg["authProvider"]})
	return nil
}

func handleAuthLogin(args cliArgs) error {
	base := strings.TrimRight(apiURL(args), "/")
	parsed, err := fetchJSON("POST", base+"/api/auth/device/code", "", map[string]any{})
	if err != nil {
		return err
	}
	obj := asObject(parsed)
	deviceCode := stringValue(obj["deviceCode"])
	userCode := stringValue(obj["userCode"])
	verificationURI := stringValue(obj["verificationUri"])
	if verificationURI == "" {
		verificationURI = base + "/auth/device"
	}
	if deviceCode == "" || userCode == "" {
		return cliError{"auth_device_code_invalid", "Device authorization response did not include deviceCode and userCode.", nil, parsed}
	}

	fmt.Printf("Open: %s\n", verificationURI)
	fmt.Printf("Enter code: %s\n", userCode)
	if !hasFlag(args, "no-browser") {
		openBrowser(verificationURI)
	}

	pollInterval, err := authDurationOption(args, "auth-poll-interval", 5*time.Second)
	if err != nil {
		return err
	}
	if pollInterval < time.Second {
		pollInterval = time.Second
	}
	timeout, err := authDurationOption(args, "auth-timeout", 15*time.Minute)
	if err != nil {
		return err
	}
	deadline := time.Now().Add(timeout)
	for {
		token, err := fetchJSON("POST", base+"/api/auth/device/token", "", map[string]any{"deviceCode": deviceCode})
		if err != nil {
			if code := authResponseError(err); code == "authorization_pending" {
				fmt.Print(".")
			} else if code == "expired_token" {
				return cliError{"auth_device_expired", "Device authorization expired before approval.", []string{"Run lmx auth login to request a new code."}, nil}
			} else if code == "key_limit_exceeded" {
				return authKeyLimitError()
			} else {
				return err
			}
		} else {
			tokenObj := asObject(token)
			switch code := stringValue(tokenObj["error"]); code {
			case "":
				key := stringValue(tokenObj["key"])
				if key == "" {
					return cliError{"auth_device_token_invalid", "Device token response did not include an API key.", nil, token}
				}
				cfg := map[string]any{"apiKey": key, "authProvider": "device", "authSavedAt": time.Now().UTC().Format(time.RFC3339)}
				if err := saveConfig(cfg); err != nil {
					return err
				}
				fmt.Println()
				printInfo("auth_saved", map[string]any{"path": configFile(), "key": redactKey(key)})
				return nil
			case "authorization_pending":
				fmt.Print(".")
			case "expired_token":
				return cliError{"auth_device_expired", "Device authorization expired before approval.", []string{"Run lmx auth login to request a new code."}, nil}
			case "key_limit_exceeded":
				return authKeyLimitError()
			default:
				return cliError{"auth_device_error", "Device authorization failed: " + code, nil, token}
			}
		}
		if time.Now().After(deadline) {
			return cliError{"auth_device_timeout", "Timed out waiting for browser approval.", []string{"Run lmx auth login to request a new code."}, nil}
		}
		sleep := pollInterval
		if remaining := time.Until(deadline); remaining < sleep {
			sleep = remaining
		}
		if sleep > 0 {
			time.Sleep(sleep)
		}
	}
}

func authKeyLimitError() error {
	return cliError{"auth_key_limit", "Account already has the maximum of 10 API keys.", []string{"Run lmx auth keys list, revoke an old key with lmx auth keys revoke <id>, then retry lmx auth login."}, nil}
}

func openBrowser(rawURL string) {
	switch runtime.GOOS {
	case "darwin":
		_ = exec.Command("open", rawURL).Start()
	case "windows":
		_ = exec.Command("rundll32", "url.dll,FileProtocolHandler", rawURL).Start()
	default:
		_ = exec.Command("xdg-open", rawURL).Start()
	}
}

func authDurationOption(args cliArgs, key string, fallback time.Duration) (time.Duration, error) {
	value := opt(args, key)
	if value == "" {
		return fallback, nil
	}
	if duration, err := time.ParseDuration(value); err == nil {
		return duration, nil
	}
	seconds, err := strconv.ParseFloat(value, 64)
	if err != nil || seconds <= 0 {
		return 0, cliError{"invalid_option", "--" + key + " must be a positive duration like 5s or 300ms.", nil, value}
	}
	return time.Duration(seconds * float64(time.Second)), nil
}

func authResponseError(err error) string {
	var cli cliError
	if errors.As(err, &cli) {
		return stringValue(asObject(cli.Details)["error"])
	}
	return ""
}

func handleAuthKeys(action, target string, args cliArgs) error {
	key := apiKey(args)
	if key == "" {
		return missingAPIKey("API key management requires authentication.")
	}
	base := strings.TrimRight(apiURL(args), "/")
	switch action {
	case "list", "":
		parsed, err := fetchJSON("GET", base+"/api/keys", key, nil)
		if err != nil {
			return err
		}
		return writeOrPrintJSON("auth_keys", args, parsed)
	case "create":
		name := firstNonEmpty(opt(args, "name"), target)
		if name == "" {
			return cliError{"missing_name", "Key creation requires --name or a positional name.", []string{"Run lmx auth keys create --name \"my key\"."}, nil}
		}
		parsed, err := fetchJSON("POST", base+"/api/keys", key, map[string]any{"name": name})
		if err != nil {
			return err
		}
		return writeOrPrintJSON("auth_key_created", args, parsed)
	case "revoke", "delete":
		id := target
		if id == "" {
			return cliError{"missing_key_id", "Key revocation requires an API key id.", []string{"Run lmx auth keys revoke <id>."}, nil}
		}
		parsed, err := fetchJSON("DELETE", base+"/api/keys/"+url.PathEscape(id), key, nil)
		if err != nil {
			return err
		}
		return writeOrPrintJSON("auth_key_revoked", args, parsed)
	default:
		return cliError{"unknown_auth_keys_command", "Unknown auth keys command: " + action, []string{"Use list, create, or revoke."}, nil}
	}
}

func handleAuthWhoami(args cliArgs) error {
	key := apiKey(args)
	if key == "" {
		return missingAPIKey("whoami requires authentication.")
	}
	parsed, err := fetchJSON("GET", strings.TrimRight(apiURL(args), "/")+"/api/user", key, nil)
	if err != nil {
		return err
	}
	return writeOrPrintJSON("auth_user", args, parsed)
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
	if action == "template" {
		return handleHardwareTemplate(args)
	}
	if action == "validate" {
		return validateHardwareFile(positional(args, 2), args)
	}
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

func validateHardwareFile(path string, args cliArgs) error {
	if path == "" {
		path = opt(args, "hardware")
	}
	if path == "" {
		return cliError{"missing_hardware", "hardware validate requires a hardware JSON path.", []string{"Run lmx hardware validate hardware.json."}, nil}
	}
	value, err := readJSON(path)
	if err != nil {
		return err
	}
	hw := asObject(value)
	if hw == nil {
		return cliError{"invalid_hardware", "Hardware file must contain a JSON object.", []string{"Generate one with lmx hardware --out hardware.json."}, value}
	}
	normalized := normalizeHardwareForSubmit(hw)
	if err := validateNormalizedHardwareShape(normalized); err != nil {
		return err
	}
	contextValue, err := fetchJSON("GET", apiURL(args)+"/api/agent-context", "", nil)
	if err != nil {
		return err
	}
	if err := validateHardwareAgainstContext(normalized, contextValue); err != nil {
		return err
	}
	return writeOrPrintJSON("hardware_valid", args, map[string]any{"path": path, "status": "valid", "hardware": normalized})
}

func hardwareTemplateFromArgs(args cliArgs) (map[string]any, error) {
	hw := map[string]any{}
	for optName, field := range map[string]string{"gpu-name": "gpuName", "cpu": "cpu", "os": "os", "hw-class": "hwClass"} {
		if v := opt(args, optName); v != "" {
			hw[field] = v
		}
	}
	if _, ok := hw["os"]; !ok {
		hw["os"] = runtime.GOOS
	}
	for optName, field := range map[string]string{"vram-gb": "vramGb", "ram-gb": "ramGb", "power-watts": "powerWatts"} {
		if v := opt(args, optName); v != "" {
			parsed, err := strconv.ParseFloat(v, 64)
			if err != nil {
				return nil, cliError{"invalid_option", "--" + optName + " must be a number", []string{"Pass a numeric value for --" + optName + "."}, nil}
			}
			hw[field] = parsed
		}
	}
	if v := opt(args, "gpu-count"); v != "" {
		parsed, err := strconv.Atoi(v)
		if err != nil {
			return nil, cliError{"invalid_option", "--gpu-count must be a number", []string{"Pass a whole number for --gpu-count."}, nil}
		}
		hw["gpuCount"] = parsed
	}
	if _, ok := hw["hwClass"]; !ok {
		if stringValue(hw["gpuName"]) != "" {
			hw["hwClass"] = "DISCRETE_GPU"
		} else {
			hw["hwClass"] = "CPU_ONLY"
		}
	}
	return hw, nil
}

func handleHardwareTemplate(args cliArgs) error {
	hw, err := hardwareTemplateFromArgs(args)
	if err != nil {
		return err
	}
	if opt(args, "out") != "" || hasFlag(args, "out") {
		out := opt(args, "out")
		if out == "" {
			out = "hardware.json"
		}
		if err := writeJSON(out, hw); err != nil {
			return err
		}
		printInfo("hardware_template_written", map[string]any{"path": out, "pathAbsolute": absPathOr(out)})
		return nil
	}
	data, _ := json.MarshalIndent(hw, "", "  ")
	fmt.Println(string(data))
	return nil
}

func handleSetups(action string, args cliArgs) error {
	key := apiKey(args)
	if key == "" {
		return missingAPIKey("--api-key or LMX_API_KEY is required for setups")
	}
	parsed, err := fetchJSON("GET", apiURL(args)+"/api/setups", key, nil)
	if err != nil {
		return err
	}
	setups := setupRows(parsed)
	if action == "" {
		action = "list"
	}
	switch action {
	case "list":
		if len(setups) == 0 {
			printInfo("no_setups", map[string]any{"next": "Save a setup from a benchmark submission or the dashboard."})
			return nil
		}
		for _, setup := range setups {
			fmt.Println(formatSetupLine(setup))
		}
		return nil
	case "pull":
		selected, err := selectSetup(setups, args)
		if err != nil {
			return err
		}
		return writeOrPrintJSON("hardware", args, setupToHardware(selected))
	default:
		return cliError{"unknown_action", "Unknown setups action: " + action, []string{"Use: setups list, setups pull"}, nil}
	}
}

func setupRows(value any) []map[string]any {
	var items []any
	if arr, ok := value.([]any); ok {
		items = arr
	} else if obj := asObject(value); obj != nil {
		if arr, ok := obj["setups"].([]any); ok {
			items = arr
		}
	}
	rows := make([]map[string]any, 0, len(items))
	for _, item := range items {
		if obj := asObject(item); obj != nil {
			rows = append(rows, obj)
		}
	}
	return rows
}

func setupNames(setups []map[string]any) []string {
	names := make([]string, 0, len(setups))
	for _, setup := range setups {
		label := stringValue(setup["name"])
		if label == "" {
			label = "(unnamed)"
		}
		if isDefault, _ := setup["isDefault"].(bool); isDefault {
			label += " (default)"
		}
		names = append(names, label)
	}
	return names
}

func selectSetup(setups []map[string]any, args cliArgs) (map[string]any, error) {
	if len(setups) == 0 {
		return nil, cliError{"setup_not_found", "No saved setups found for this API key.", []string{"Save a setup from a benchmark submission or the dashboard."}, nil}
	}
	if id := opt(args, "id"); id != "" {
		for _, setup := range setups {
			if stringValue(setup["id"]) == id {
				return setup, nil
			}
		}
		return nil, cliError{"setup_not_found", "No setup found with id " + id, setupNames(setups), nil}
	}
	if name := opt(args, "name"); name != "" {
		for _, setup := range setups {
			if strings.EqualFold(stringValue(setup["name"]), name) {
				return setup, nil
			}
		}
		return nil, cliError{"setup_not_found", "No setup found named " + name, setupNames(setups), nil}
	}
	for _, setup := range setups {
		if isDefault, _ := setup["isDefault"].(bool); isDefault {
			return setup, nil
		}
	}
	if hasFlag(args, "default") {
		return nil, cliError{"setup_not_found", "No default setup is set; pass --name or --id.", setupNames(setups), nil}
	}
	if len(setups) == 1 {
		return setups[0], nil
	}
	return nil, cliError{"ambiguous_setup", "Multiple setups; pass --default, --name, or --id", setupNames(setups), nil}
}

func setupGpus(setup map[string]any) []any {
	raw, ok := setup["gpus"].([]any)
	if !ok {
		return nil
	}
	gpus := []any{}
	for _, item := range raw {
		obj := asObject(item)
		if obj == nil {
			continue
		}
		name := stringValue(obj["name"])
		if name == "" {
			continue
		}
		gpu := map[string]any{"gpuName": name}
		if n := numberField(obj, "count"); n != 0 {
			gpu["count"] = n
		}
		if n := numberField(obj, "vramGb"); n != 0 {
			gpu["vramGb"] = n
		}
		gpus = append(gpus, gpu)
	}
	if len(gpus) == 0 {
		return nil
	}
	return gpus
}

func setupToHardware(setup map[string]any) map[string]any {
	hwClass := stringValue(setup["hwClass"])
	if hwClass == "" {
		hwClass = "DISCRETE_GPU"
	}
	hw := map[string]any{"hwClass": hwClass}
	setStr := func(key string) {
		if v := stringValue(setup[key]); v != "" {
			hw[key] = v
		}
	}
	setNum := func(key string) {
		if n := numberField(setup, key); n != 0 {
			hw[key] = n
		}
	}
	switch hwClass {
	case "UNIFIED":
		setStr("chipVendor")
		setStr("chipFamily")
		setStr("chipVariant")
		setNum("unifiedMemoryGb")
		setNum("npuTops")
		setStr("cpu")
		setStr("os")
	case "CPU_ONLY":
		setStr("cpu")
		setNum("ramGb")
		setStr("os")
	default:
		if gpus := setupGpus(setup); len(gpus) > 0 {
			hw["gpus"] = gpus
		} else {
			setStr("gpuName")
			setNum("gpuCount")
			setNum("vramGb")
		}
		setStr("cpu")
		setNum("ramGb")
		setStr("os")
	}
	return hw
}

func setupHardwareSummary(setup map[string]any) string {
	switch stringValue(setup["hwClass"]) {
	case "DISCRETE_GPU":
		if gpus, ok := setup["gpus"].([]any); ok && len(gpus) > 0 {
			parts := []string{}
			for _, item := range gpus {
				obj := asObject(item)
				if obj == nil {
					continue
				}
				parts = append(parts, gpuSummary(numberField(obj, "count"), stringValue(obj["name"]), numberField(obj, "vramGb")))
			}
			return strings.Join(parts, " + ")
		}
		return gpuSummary(numberField(setup, "gpuCount"), stringValue(setup["gpuName"]), numberField(setup, "vramGb"))
	case "UNIFIED":
		return firstNonEmpty(stringValue(setup["chipVariant"]), stringValue(setup["chipFamily"]), stringValue(setup["chipVendor"]), "Unified memory system")
	default:
		return firstNonEmpty(stringValue(setup["cpu"]), "CPU")
	}
}

func gpuSummary(count float64, name string, vramGb float64) string {
	if count == 0 {
		count = 1
	}
	label := fmt.Sprintf("%gx %s", count, firstNonEmpty(name, "GPU"))
	if vramGb != 0 {
		label += fmt.Sprintf(" (%g GB)", vramGb)
	}
	return label
}

func formatSetupLine(setup map[string]any) string {
	name := stringValue(setup["name"])
	if name == "" {
		name = "(unnamed)"
	}
	if isDefault, _ := setup["isDefault"].(bool); isDefault {
		name += " ★"
	}
	segments := []string{name}
	if hw := stringValue(setup["hwClass"]); hw != "" {
		segments = append(segments, hw)
	}
	if summary := setupHardwareSummary(setup); summary != "" {
		segments = append(segments, summary)
	}
	engineParts := []string{}
	if e := stringValue(setup["engineName"]); e != "" {
		engineParts = append(engineParts, e)
	}
	if q := stringValue(setup["quantization"]); q != "" {
		engineParts = append(engineParts, q)
	}
	if len(engineParts) > 0 {
		segments = append(segments, strings.Join(engineParts, " · "))
	}
	return strings.Join(segments, " — ")
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
		if hasFlag(args, "include-server-metadata") {
			if meta := discoverServerMetadata(baseURL, opt(args, "model-api-key"), info); len(meta) > 0 {
				result["serverMetadata"] = meta
			} else {
				result["serverMetadataNote"] = "endpoint exposed no opt-in server metadata"
			}
		}
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

func round1(n float64) float64 { return math.Round(n*10) / 10 }

func handleContext(args cliArgs) error {
	value, err := fetchJSON("GET", apiURL(args)+"/api/agent-context", "", nil)
	if err != nil {
		return err
	}
	return writeOrPrintJSON("context", args, value)
}

func handleModel(action, target string, args cliArgs) error {
	switch action {
	case "search":
		query := target
		if query == "" {
			query = firstNonEmpty(opt(args, "q"), opt(args, "query"))
		}
		if query == "" {
			return cliError{"missing_query", "model search requires a query", []string{"Run lmx model search qwen3-8b."}, nil}
		}
		if normalized, changed := normalizedModelSearchQuery(query); changed {
			printStatus(args, "model_search_query_normalized", map[string]any{"query": query, "normalized": normalized, "hint": "GGUF filename detected; searching by the derived model name."})
			query = normalized
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
	case "resolve-remote", "resolve":
		return handleModelResolveRemote(args)
	default:
		return errors.New("Unknown model command. Use search or resolve-remote.")
	}
}

func handleModelResolveRemote(args cliArgs) error {
	baseURL, err := requireOpt(args, "base-url")
	if err != nil {
		return err
	}
	baseURL = openAIBaseURL(baseURL)
	apiKey := opt(args, "model-api-key")
	model, info, err := detectServedModel(baseURL, apiKey, firstNonEmpty(opt(args, "served-model"), opt(args, "model-name")))
	if err != nil {
		return cliError{"endpoint_unreachable", "Could not read /v1/models from " + baseURL + ".", []string{"Verify --base-url and that the endpoint is running.", err.Error()}, nil}
	}
	quantRes := remoteQuantizationResolution(args, baseURL, apiKey, opt(args, "quantization"), info)
	filename := ""
	if quantRes != nil {
		if mp := stringValue(quantRes["modelPath"]); mp != "" {
			filename = filepath.Base(mp)
		}
	}
	query, querySource := remoteModelSearchQuery(model, filename)
	result := map[string]any{"baseUrl": baseURL, "servedModel": model, "searchQuery": query, "searchQuerySource": querySource}
	if filename != "" {
		result["loadedFilename"] = filename
	}
	if quantRes != nil && quantRes["trusted"] != nil {
		result["quantization"] = quantRes["trusted"]
	}
	value, err := searchModels(args, query, 5)
	if err != nil {
		result["searchError"] = err.Error()
		return writeOrPrintJSON("model_resolution", args, result)
	}
	candidates := modelCandidates(value, 5)
	result["candidates"] = candidates
	if filename != "" {
		if match, err := sourceRepoFromFilename(args, candidates, filename); err == nil && match != "" {
			result["sourceRepo"] = match
			result["sourceRepoMatch"] = "exact_filename"
		}
	}
	rerunHfID := firstNonEmpty(stringValue(result["sourceRepo"]), candidateID(candidates))
	if rerunHfID != "" {
		cmd := "lmx benchmark run llama.cpp --mode remote --base-url " + baseURL + " --served-model " + shellQuote(model) + " --hf-id " + rerunHfID
		if quant := stringValue(result["quantization"]); quant != "" {
			cmd += " --quantization " + quant
		}
		result["rerunCommand"] = cmd
	}
	return writeOrPrintJSON("model_resolution", args, result)
}

func candidateID(candidates []any) string {
	if len(candidates) == 0 {
		return ""
	}
	return candidateRepoID(candidates[0])
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
	printInfo("suite_template_written", map[string]any{"path": out, "slug": slug, "runner": runner, "scoringMethod": stringValue(suiteDoc(payload)["scoringMethod"]), "tasks": len(evalTasks(suiteDoc(payload)))})
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
	case "loglikelihood":
		base["suiteDoc"] = map[string]any{"version": "1.0", "runner": "custom", "scoringMethod": "loglikelihood", "higherIsBetter": true, "aggregation": "weighted_mean", "runConfig": map[string]any{"temperature": 0, "loglikelihoodTarget": "choice_text", "loglikelihoodNorm": "byte"}, "tasks": []any{map[string]any{"key": "loglikelihood", "displayName": "Multiple choice (loglikelihood)", "taskType": "multiple_choice", "weight": 1, "promptTemplate": "Question: {{input}}\nAnswer:", "dataset": map[string]any{"source": "inline", "items": []any{map[string]any{"input": "Which number is even?", "choices": []any{"3", "5", "8", "9"}, "gold": "C"}}}}}}
	case "math":
		base["suiteDoc"] = map[string]any{"version": "1.0", "runner": "custom", "scoringMethod": "exact_match", "higherIsBetter": true, "aggregation": "weighted_mean", "runConfig": map[string]any{"temperature": 0}, "tasks": []any{map[string]any{"key": "math", "displayName": "Numeric reasoning", "taskType": "qa", "weight": 1, "promptTemplate": "Solve the problem. Show your reasoning, then state the final answer.\n\n{{input}}", "answerExtraction": "last_number", "maxNewTokens": 512, "dataset": map[string]any{"source": "inline", "items": []any{map[string]any{"input": "Natalia sold 48 clips in April and half as many in May. How many clips did she sell altogether?", "gold": "72"}}}}}}
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
		if !containsString([]string{"exact_match", "f1", "pass_at_k", "perplexity", "llm_judge", "loglikelihood"}, scoring) {
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
				if ext := stringValue(task["answerExtraction"]); ext != "" {
					if !containsString([]string{"none", "final_answer", "last_number", "regex"}, ext) {
						errs = append(errs, prefix+".answerExtraction must be none, final_answer, last_number, or regex")
					}
					if ext == "regex" {
						pattern := stringValue(task["answerRegex"])
						if pattern == "" {
							errs = append(errs, prefix+".answerRegex is required when answerExtraction is regex")
						} else if _, err := regexp.Compile(pattern); err != nil {
							errs = append(errs, prefix+".answerRegex is not a valid regular expression")
						}
					}
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
		runPath := benchmarkRunPathInDir(payload, firstNonEmpty(opt(args, "runs-dir"), "runs"))
		out := firstNonEmpty(opt(args, "out"), runPath)
		feedback := benchmarkAgentFeedback(payload, out, args, hasFlag(args, "dry-run"), hasFlag(args, "submit"))
		if absOut, err := filepath.Abs(out); err == nil {
			feedback["outputPathAbsolute"] = absOut
		}
		feedback["savedRunPathRelative"] = runPath
		if absRun, err := filepath.Abs(runPath); err == nil {
			feedback["savedRunPathAbsolute"] = absRun
		}
		if out != runPath {
			feedback["savedRunPath"] = runPath
		}
		payload["agentFeedback"] = feedback
		if err := writeBenchmarkPayloadFiles(payload, out, runPath); err != nil {
			return err
		}
		printInfo("benchmark_payload_file_written", map[string]any{"path": out, "pathAbsolute": absPathOr(out), "engine": payload["engineName"]})
		if out != runPath {
			printInfo("benchmark_run_saved", map[string]any{"path": runPath, "pathAbsolute": absPathOr(runPath), "engine": payload["engineName"]})
		}
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
		fmt.Println(stringValue(feedback["message"]))
		printBenchmarkNextSteps(feedback, out)
		return nil
	}
	if action == "add-hardware" || action == "attach-hardware" {
		return addHardwareToRun(target, args)
	}
	if action == "fixup" {
		return fixupBenchmarkRun(target, args)
	}
	if action != "submit" && action != "dry-run" {
		return errors.New("Unknown benchmark command. Use run, runs, list, show, edit, rerun, add-hardware, fixup, submit, dry-run, validate-local, delete, stats, export, or compare.")
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

func writeBenchmarkPayloadFiles(payload map[string]any, out, runPath string) error {
	if err := os.MkdirAll(filepath.Dir(out), 0o755); err != nil {
		return err
	}
	if err := writeJSON(out, payload); err != nil {
		return err
	}
	if out == runPath {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(runPath), 0o755); err != nil {
		return err
	}
	return writeJSON(runPath, payload)
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
	case "add-hardware", "attach-hardware":
		return handleBenchmark("add-hardware", target, args)
	case "fixup":
		return handleBenchmark("fixup", target, args)
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
		return errors.New("Unknown benchmark runs command. Use list, show, edit, rerun, add-hardware, fixup, submit, dry-run, delete, stats, export, or compare.")
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
	result["p95"] = roundMetric(percentileOf(values, 95))
	if len(values) > 1 {
		mean := sum / float64(len(values))
		variance := 0.0
		for _, value := range values {
			variance += (value - mean) * (value - mean)
		}
		result["stddev"] = roundMetric(math.Sqrt(variance / float64(len(values)-1)))
	}
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
	groupStat := func(group map[string]any) float64 {
		if value := numberField(group, "p50"); value > 0 {
			return value
		}
		return numberField(group, "best")
	}
	baseValue := groupStat(baseline)
	comparisons := []any{}
	for _, item := range groups {
		group := asObject(item)
		comparisons = append(comparisons, metricComparison(stringValue(group["key"]), groupStat(group), stringValue(baseline["key"]), baseValue, metric))
	}
	stats["baseline"] = baseline["key"]
	stats["comparisonStat"] = "p50"
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
	return math.Round(value*100) / 100
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

type benchmarkPayload struct {
	EngineName      string
	HFID            string
	ModelRevision   string
	Quantization    string
	Backend         string
	BenchmarkMode   string
	DetectedEngines []detectedEngine
	Hardware        any
	HardwareSource  string
	Extra           map[string]any
}

func (payload benchmarkPayload) ToMap() map[string]any {
	out := map[string]any{
		"engineName":      payload.EngineName,
		"hfId":            payload.HFID,
		"modelRevision":   payload.ModelRevision,
		"quantization":    payload.Quantization,
		"benchmarkMode":   payload.BenchmarkMode,
		"detectedEngines": payload.DetectedEngines,
	}
	if payload.Backend != "" {
		out["backend"] = payload.Backend
	}
	if payload.Hardware != nil {
		out["hardware"] = payload.Hardware
	}
	if payload.HardwareSource != "" {
		out["hardwareSource"] = payload.HardwareSource
	}
	for key, value := range payload.Extra {
		out[key] = value
	}
	return out
}

func setNumericFlagFields(payload map[string]any, args cliArgs, mapping map[string]string) {
	for flag, field := range mapping {
		if value := opt(args, flag); value != "" {
			if n, err := strconv.ParseFloat(value, 64); err == nil {
				payload[field] = n
			} else {
				payload[field] = value
			}
		}
	}
}

func medianOf(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	sorted := append([]float64(nil), values...)
	sort.Float64s(sorted)
	mid := len(sorted) / 2
	if len(sorted)%2 == 0 {
		return (sorted[mid-1] + sorted[mid]) / 2
	}
	return sorted[mid]
}

// percentileOf interpolates the pct percentile of an ascending-sorted slice.
func percentileOf(sorted []float64, pct float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	rank := pct / 100 * float64(len(sorted)-1)
	lower := int(math.Floor(rank))
	upper := int(math.Ceil(rank))
	if lower == upper {
		return sorted[lower]
	}
	weight := rank - float64(lower)
	return sorted[lower]*(1-weight) + sorted[upper]*weight
}

func metricSummary(values []float64) map[string]any {
	if len(values) == 0 {
		return map[string]any{"count": 0}
	}
	sorted := append([]float64(nil), values...)
	sort.Float64s(sorted)
	sum := 0.0
	for _, value := range sorted {
		sum += value
	}
	mean := sum / float64(len(sorted))
	summary := map[string]any{
		"count": len(sorted),
		"min":   roundMetric(sorted[0]),
		"p50":   roundMetric(medianOf(sorted)),
		"mean":  roundMetric(mean),
		"max":   roundMetric(sorted[len(sorted)-1]),
	}
	if len(sorted) > 1 {
		variance := 0.0
		for _, value := range sorted {
			variance += (value - mean) * (value - mean)
		}
		summary["stddev"] = roundMetric(math.Sqrt(variance / float64(len(sorted)-1)))
	}
	return summary
}

func intOption(args cliArgs, fallback, minimum int, keys ...string) (int, error) {
	value := ""
	for _, key := range keys {
		if value = opt(args, key); value != "" {
			break
		}
	}
	if value == "" {
		return fallback, nil
	}
	parsed, err := strconv.Atoi(value)
	if err != nil || parsed < minimum {
		return 0, cliError{"invalid_option", fmt.Sprintf("--%s must be an integer >= %d", keys[0], minimum), []string{fmt.Sprintf("Pass --%s <number>.", keys[0])}, nil}
	}
	return parsed, nil
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

func addHardwareToRun(target string, args cliArgs) error {
	path, err := resolveBenchmarkRunPath(target, args)
	if err != nil {
		return err
	}
	value, err := readJSON(path)
	if err != nil {
		return err
	}
	payload := benchmarkPayloadObject(value)
	if payload == nil {
		return cliError{"invalid_benchmark_run", "Saved benchmark run must be a JSON object or { payload: object }.", nil, value}
	}
	hardwarePath := opt(args, "hardware")
	if hardwarePath == "" {
		return cliError{"missing_option", "--hardware is required", []string{"Generate it on the endpoint host: lmx hardware --out hardware.json", "Or build one from known specs: lmx hardware template --gpu-name ... --vram-gb ... > hardware.json", "Then: lmx benchmark add-hardware " + path + " --hardware hardware.json"}, nil}
	}
	loaded, err := readJSON(hardwarePath)
	if err != nil {
		return err
	}
	hardware := asObject(loaded)
	if hardware == nil {
		return cliError{"invalid_hardware", "--hardware must point to a JSON object.", []string{"Generate one with lmx hardware --out hardware.json on the benchmark host."}, loaded}
	}
	payload["hardware"] = loaded
	payload["hardwareSource"] = "file"
	feedback := benchmarkAgentFeedback(payload, path, args, false, false)
	payload["agentFeedback"] = feedback
	if err := writeJSON(path, value); err != nil {
		return err
	}
	printInfo("benchmark_hardware_attached", map[string]any{"path": path, "pathAbsolute": absPathOr(path), "gpuName": hardware["gpuName"], "vramGb": hardware["vramGb"], "benchmarkStatus": feedback["benchmarkStatus"], "submissionStatus": feedback["submissionStatus"]})
	printBenchmarkNextSteps(feedback, path)
	return nil
}

func benchmarkFixupReport(payload map[string]any, path string, args cliArgs) map[string]any {
	issues := []map[string]any{}
	if stringValue(payload["benchmarkMode"]) == "remote" && asObject(payload["hardware"]) == nil {
		issues = append(issues, map[string]any{"code": "missing_remote_hardware", "message": "Remote run has no server hardware metadata.", "commands": []string{"lmx hardware --out hardware.json   (run on the endpoint host)", "lmx hardware template --gpu-name \"...\" --vram-gb N --cpu \"...\" --ram-gb N --os Linux > hardware.json", "lmx benchmark add-hardware " + path + " --hardware hardware.json"}})
	}
	if mr := asObject(payload["modelResolution"]); mr != nil {
		if sr := stringValue(mr["sourceRepo"]); sr != "" && sr != stringValue(payload["hfId"]) {
			issues = append(issues, map[string]any{"code": "hf_id_mismatch", "message": "Detected source repo " + sr + " differs from hfId " + stringValue(payload["hfId"]) + ".", "commands": []string{"lmx benchmark runs edit " + path + " --set hfId=" + sr}})
		}
	}
	if qr := asObject(payload["quantizationResolution"]); qr != nil && stringValue(qr["status"]) == "mismatch" {
		if tr := stringValue(qr["trusted"]); tr != "" {
			issues = append(issues, map[string]any{"code": "quantization_mismatch", "message": "Endpoint quantization " + tr + " differs from declared " + stringValue(payload["quantization"]) + ".", "commands": []string{"lmx benchmark runs edit " + path + " --set quantization=" + tr}})
		}
	}
	feedback := benchmarkAgentFeedback(payload, path, args, false, false)
	return map[string]any{"path": path, "pathAbsolute": absPathOr(path), "benchmarkStatus": feedback["benchmarkStatus"], "submissionStatus": feedback["submissionStatus"], "issues": issues, "ready": len(issues) == 0 && feedback["submissionStatus"] == "ready", "nextCommands": []string{"lmx benchmark dry-run " + path, "lmx benchmark submit " + path}}
}

func fixupBenchmarkRun(target string, args cliArgs) error {
	path, err := resolveBenchmarkRunPath(target, args)
	if err != nil {
		return err
	}
	value, err := readJSON(path)
	if err != nil {
		return err
	}
	payload := benchmarkPayloadObject(value)
	if payload == nil {
		return cliError{"invalid_benchmark_run", "Saved benchmark run must be a JSON object or { payload: object }.", nil, value}
	}
	return writeOrPrintJSON("benchmark_fixup", args, benchmarkFixupReport(payload, path, args))
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

func absPathOr(p string) string {
	if abs, err := filepath.Abs(p); err == nil {
		return abs
	}
	return p
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
			"contextTokens": firstNonNil(point["contextTokens"], float64(level)),
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
	stdout, stderr, err := runBenchmarkCommand(args, commandSnippet)
	if err != nil {
		return nil, err
	}
	fileOutput := ""
	if outputPath := localKVCacheOutputPath(engineName, args, level); outputPath != "" {
		if data, readErr := os.ReadFile(outputPath); readErr == nil {
			fileOutput = string(data)
			point["benchmarkOutput"] = outputPath
		}
	}
	for key, value := range parseBenchmarkLayers(fileOutput, stdout, stderr) {
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
		appendShellArg(&cmd, "--host", opt(args, "host"))
		appendShellArg(&cmd, "--port", opt(args, "port"))
		appendShellArg(&cmd, "--endpoint", opt(args, "endpoint"))
		appendShellArg(&cmd, "--dataset-name", firstNonEmpty(opt(args, "dataset-name"), "random"))
		appendShellArg(&cmd, "--dataset-path", opt(args, "dataset-path"))
		appendShellArg(&cmd, "--input-len", input)
		appendShellArg(&cmd, "--output-len", output)
		appendShellArg(&cmd, "--num-prompts", firstNonEmpty(opt(args, "num-prompts"), "1"))
		appendShellArg(&cmd, "--request-rate", opt(args, "request-rate"))
		appendShellArg(&cmd, "--max-concurrency", opt(args, "max-concurrency"))
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
	prompt, contextTokens, contextTokenSource, err := kvCachePromptForRemote(args, hfID, firstNonEmpty(opt(args, "model-revision"), "main"), level)
	if err != nil {
		return nil, err
	}
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
	probe, err := timedChatCompletion(args, baseURL, body, timeout, "kvcache_remote_failed")
	if err != nil {
		return nil, err
	}
	usagePromptTokens := usageToken(probe.usage, "prompt_tokens")
	usageOutputTokens := firstNonZero(usageToken(probe.usage, "completion_tokens"), usageToken(probe.usage, "output_tokens"))
	outputTokens := usageOutputTokens
	if outputTokens == 0 {
		count, err := tokenCount(args, hfID, firstNonEmpty(opt(args, "model-revision"), "main"), probe.outputText, 0, "output")
		if err != nil {
			return nil, err
		}
		outputTokens = count.Count
	}
	// usage.prompt_tokens reports the full prompt of the timed request,
	// including tokens served from the retained KV cache; it is the measured
	// counterpart of the nominal context level.
	promptTokens := level
	if usagePromptTokens > 0 {
		promptTokens = usagePromptTokens
	}
	point := map[string]any{
		"contextTokens":      float64(contextTokens),
		"promptTokens":       float64(promptTokens),
		"outputTokens":       float64(outputTokens),
		"outputText":         probe.outputText,
		"mode":               "remote",
		"baseUrl":            baseURL,
		"servedModel":        servedModel,
		"servedModelSource":  servedModelSource,
		"methodology":        methodology,
		"cacheReuse":         cacheStatus,
		"usagePromptTokens":  float64(usagePromptTokens),
		"contextTokenSource": contextTokenSource,
		"metricSource":       "remote_endpoint",
		"timingSource":       "client_observed_http",
	}
	if tokSOut, source, ok := decodeThroughput(probe, outputTokens); ok {
		point["tokSOut"] = tokSOut
		point["tokSOutSource"] = source
	}
	if totalMs := durationMS(probe.completedAt.Sub(probe.started)); totalMs > 0 {
		point["tokSTotal"] = round1(float64(promptTokens+outputTokens) / (totalMs / 1000))
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
	if !probe.firstTokenAt.IsZero() {
		ttftMs := durationMS(probe.firstTokenAt.Sub(probe.started))
		point["ttftMs"] = roundMetric(ttftMs)
		prefillTokens := promptTokens
		prefillSource := "estimated_from_ttft"
		if stringValue(cacheStatus["status"]) == "retained" {
			// Only the non-cached suffix is actually prefetched during TTFT.
			prefillTokens = promptTokens - cacheTokens
			prefillSource = "estimated_from_ttft_uncached"
		}
		if ttftMs > 0 && prefillTokens > 0 {
			point["tokSPrefill"] = round1(float64(prefillTokens) / (ttftMs / 1000))
			point["tokSPrefillSource"] = prefillSource
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

// kvCacheFillerVocab lists common single-token words used to synthesize a
// non-repetitive long-context filler prompt. A single repeated word is
// maximally friendly to prefix caching and compression shortcuts, which
// overstates long-context speed; varied filler is more representative.
var kvCacheFillerVocab = []string{
	"context", "window", "memory", "tokens", "stream", "model", "cache", "depth",
	"prompt", "decode", "layer", "tensor", "batch", "query", "value", "state",
	"sample", "weight", "logit", "vector", "buffer", "index", "block", "frame",
	"graph", "kernel", "thread", "shard", "slice", "table", "queue", "stack",
}

// kvCachePrompt builds a deterministic filler prompt of approximately
// targetTokens tokens. Determinism is required so the warm request and the
// timed probe share an identical prefix for KV-cache reuse.
func kvCachePrompt(args cliArgs, targetTokens int) string {
	if prompt := opt(args, "prompt"); prompt != "" {
		return prompt
	}
	if targetTokens < 1 {
		targetTokens = 1
	}
	return kvCachePromptWords(args, targetTokens)
}

func kvCachePromptWords(args cliArgs, words int) string {
	if words < 1 {
		words = 1
	}
	if word := opt(args, "filler-token"); word != "" {
		return strings.TrimSpace(strings.Repeat(word+" ", words))
	}
	var builder strings.Builder
	builder.Grow(words * 8)
	seed := uint64(0x9E3779B97F4A7C15)
	for i := 0; i < words; i++ {
		seed ^= seed << 13
		seed ^= seed >> 7
		seed ^= seed << 17
		if i > 0 {
			builder.WriteByte(' ')
		}
		builder.WriteString(kvCacheFillerVocab[seed%uint64(len(kvCacheFillerVocab))])
	}
	return builder.String()
}

func kvCachePromptForRemote(args cliArgs, hfID, revision string, targetTokens int) (string, int, string, error) {
	if value := firstNonEmpty(opt(args, "prompt-tokens"), opt(args, "prefill-tokens")); value != "" {
		count, err := strconv.Atoi(value)
		if err != nil || count <= 0 {
			return "", 0, "", cliError{"invalid_option", "--prompt-tokens must be a positive integer", []string{"Pass --prompt-tokens <number>."}, nil}
		}
		return kvCachePrompt(args, targetTokens), count, "explicit_flag", nil
	}
	if prompt := opt(args, "prompt"); prompt != "" {
		count, err := pythonTokenCount(hfID, revision, prompt)
		if err != nil {
			return "", 0, "", cliError{"token_count_missing", "Could not tokenize --prompt for remote KV-cache measurement.", []string{"Install Python transformers, or pass a model tokenizer available to AutoTokenizer.", "Remote KV-cache sweeps require token-accurate context lengths."}, errString(err)}
		}
		return prompt, count, "python_transformers_explicit_prompt", nil
	}
	prompt, count, err := pythonExactTokenText(hfID, revision, targetTokens, kvCachePromptWords(args, targetTokens*2))
	if err != nil {
		return "", 0, "", cliError{"token_count_missing", "Could not build a token-accurate remote KV-cache filler prompt.", []string{"Install Python transformers so the CLI can synthesize exactly --levels tokens.", "Pass --prompt with text that the tokenizer helper can count if you need custom context content."}, errString(err)}
	}
	return prompt, count, "python_transformers_exact_filler", nil
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
	var outputLayers []string
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
		outputLayers = append(outputLayers, string(data))
		metrics["engineFlags"] = map[string]any{"mode": mode, "commandSnippet": "# Metrics imported from " + resultsPath, "resultsPath": resultsPath}
		printStatus(args, "benchmark_results_read_complete", map[string]any{"path": resultsPath, "bytes": len(data)})
	} else if commandSnippet := localBenchmarkCommand(engineName, args); commandSnippet != "" {
		printStatus(args, "benchmark_local_command_start", map[string]any{"command": commandSnippet})
		stdout, stderr, err := runBenchmarkCommand(args, commandSnippet)
		if err != nil {
			return nil, err
		}
		if outputPath := benchmarkOutputPath(args); outputPath != "" {
			if data, readErr := os.ReadFile(outputPath); readErr == nil {
				outputLayers = append(outputLayers, string(data))
				printStatus(args, "benchmark_results_read_complete", map[string]any{"path": outputPath, "bytes": len(data)})
			}
		}
		outputLayers = append(outputLayers, stdout, stderr)
		metrics["engineFlags"] = localBenchmarkEngineFlags(engineName, commandSnippet)
		printStatus(args, "benchmark_local_command_complete", map[string]any{"outputBytes": len(stdout) + len(stderr)})
	}
	if len(outputLayers) > 0 {
		parsed := parseBenchmarkLayers(outputLayers...)
		for key, value := range parsed {
			metrics[key] = value
		}
		printStatus(args, "benchmark_metrics_detected", metricStatusFields(parsed))
	}

	hardware, hardwareSource, err := benchmarkHardware(mode, args)
	if err != nil {
		return nil, err
	}

	builder := benchmarkPayload{
		EngineName:      engineName,
		HFID:            model,
		ModelRevision:   firstNonEmpty(opt(args, "model-revision"), "main"),
		Quantization:    quantization,
		Backend:         backend,
		BenchmarkMode:   mode,
		DetectedEngines: detectInferenceEngines(args),
		Hardware:        hardware,
		HardwareSource:  hardwareSource,
		Extra:           metrics,
	}
	payload := builder.ToMap()
	setNumericFlagFields(payload, args, map[string]string{"tok-s-out": "tokSOut", "tok-s-prefill": "tokSPrefill", "tok-s-total": "tokSTotal", "ttft-ms": "ttftMs", "peak-vram-gb": "peakVramGb", "context-length": "contextLength", "batch-size": "batchSize", "input-len": "inputLen", "output-len": "outputLen", "prompt-tokens": "promptTokens", "prefill-tokens": "promptTokens", "output-tokens": "outputTokens", "num-prompts": "numPrompts"})
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
		details := strings.TrimSpace(strings.Join(outputLayers, "\n"))
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
		out["hardware"] = normalizeHardwareForSubmit(hw)
	}
	if ef := asObject(payload["engineFlags"]); ef != nil {
		out["engineFlags"] = remapEngineFlags(ef, payload)
	}

	return out
}

func normalizeHardwareForSubmit(hw map[string]any) map[string]any {
	hwClass := "CPU_ONLY"
	if firstNonEmpty(stringValue(hw["gpuName"]), stringValue(hw["name"])) != "" || len(anySlice(hw["gpus"])) > 0 {
		hwClass = "DISCRETE_GPU"
	} else if firstNonEmpty(stringValue(hw["chipVendor"]), stringValue(hw["chipFamily"])) != "" || stringValue(hw["hwClass"]) == "UNIFIED" || stringValue(hw["hwClass"]) == "APPLE_SILICON" {
		hwClass = "UNIFIED"
	}

	remapped := map[string]any{"hwClass": hwClass}
	if hwClass == "DISCRETE_GPU" {
		flatGPUName := firstNonEmpty(stringValue(hw["gpuName"]), stringValue(hw["name"]))
		flatVRAM := firstPositiveFloat(numberField(hw, "vramGb"), numberField(hw, "gpuVramGb"), numberField(hw, "totalVramGb"))
		gpus := normalizeGpuSlots(anySlice(hw["gpus"]))
		if len(gpus) > 1 {
			remapped["gpus"] = gpus
		} else {
			if flatGPUName == "" && len(gpus) == 1 {
				flatGPUName = stringValue(gpus[0]["gpuName"])
			}
			if flatVRAM == 0 && len(gpus) == 1 {
				flatVRAM = numberField(gpus[0], "vramGb")
			}
			remapped["gpuName"] = firstNonEmpty(flatGPUName, "Unknown GPU")
			remapped["vramGb"] = flatVRAM
			if c := numberField(hw, "gpuCount"); c > 0 {
				remapped["gpuCount"] = c
			}
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

func normalizeGpuSlots(slots []any) []map[string]any {
	out := []map[string]any{}
	for _, slotValue := range slots {
		slot := asObject(slotValue)
		if slot == nil {
			continue
		}
		gpuName := firstNonEmpty(stringValue(slot["gpuName"]), stringValue(slot["name"]))
		vramGb := firstPositiveFloat(numberField(slot, "vramGb"), numberField(slot, "gpuVramGb"))
		if gpuName == "" && vramGb == 0 {
			continue
		}
		normalized := map[string]any{}
		if gpuName != "" {
			normalized["gpuName"] = gpuName
		}
		if count := numberField(slot, "count"); count > 0 {
			normalized["count"] = count
		}
		if vramGb > 0 {
			normalized["vramGb"] = vramGb
		}
		out = append(out, normalized)
	}
	return out
}

func validateNormalizedHardwareShape(hw map[string]any) error {
	switch stringValue(hw["hwClass"]) {
	case "DISCRETE_GPU":
		if slots := normalizedGpuSlotSlice(hw["gpus"]); len(slots) > 0 {
			for i, slot := range slots {
				if stringValue(slot["gpuName"]) == "" || numberField(slot, "vramGb") <= 0 {
					return cliError{"invalid_hardware", fmt.Sprintf("hardware.gpus[%d] must include gpuName and vramGb.", i), []string{"Use gpus entries shaped like {\"gpuName\":\"NVIDIA GeForce RTX 4090\",\"count\":1,\"vramGb\":24}."}, hw}
				}
			}
			return nil
		}
		if stringValue(hw["gpuName"]) == "" || numberField(hw, "vramGb") <= 0 {
			return cliError{"invalid_hardware", "DISCRETE_GPU hardware requires gpuName and vramGb.", []string{"Use lmx hardware template --gpu-name \"NVIDIA GeForce RTX 4090\" --vram-gb 24."}, hw}
		}
	case "UNIFIED":
		if stringValue(hw["chipVendor"]) == "" || stringValue(hw["chipFamily"]) == "" || stringValue(hw["chipVariant"]) == "" || numberField(hw, "unifiedMemoryGb") <= 0 {
			return cliError{"invalid_hardware", "UNIFIED hardware requires chipVendor, chipFamily, chipVariant, and unifiedMemoryGb.", nil, hw}
		}
	case "CPU_ONLY":
		if stringValue(hw["cpu"]) == "" || numberField(hw, "ramGb") <= 0 {
			return cliError{"invalid_hardware", "CPU_ONLY hardware requires cpu and ramGb.", nil, hw}
		}
	default:
		return cliError{"invalid_hardware", "hardware.hwClass must be DISCRETE_GPU, UNIFIED, or CPU_ONLY.", nil, hw}
	}
	return nil
}

func validateHardwareAgainstContext(hw map[string]any, contextValue any) error {
	contextObj := asObject(contextValue)
	options := asObject(contextObj["hardwareOptions"])
	if options == nil {
		return nil
	}
	switch stringValue(hw["hwClass"]) {
	case "DISCRETE_GPU":
		allowed := stringSet(anySlice(options["discreteGpuNames"]))
		if len(allowed) == 0 {
			return nil
		}
		if slots := normalizedGpuSlotSlice(hw["gpus"]); len(slots) > 0 {
			for _, slot := range slots {
				if name := stringValue(slot["gpuName"]); name != "" && !allowed[name] {
					return unsupportedHardwareName("GPU", name, "hardwareOptions.discreteGpuNames")
				}
			}
			return nil
		}
		if name := stringValue(hw["gpuName"]); name != "" && !allowed[name] {
			return unsupportedHardwareName("GPU", name, "hardwareOptions.discreteGpuNames")
		}
	case "UNIFIED":
		if allowed := stringSet(anySlice(options["chipVendors"])); len(allowed) > 0 {
			if name := stringValue(hw["chipVendor"]); name != "" && !allowed[name] {
				return unsupportedHardwareName("chip vendor", name, "hardwareOptions.chipVendors")
			}
		}
	}
	return nil
}

func normalizedGpuSlotSlice(value any) []map[string]any {
	raw := anySlice(value)
	if len(raw) == 0 {
		return nil
	}
	out := make([]map[string]any, 0, len(raw))
	for _, item := range raw {
		if obj := asObject(item); obj != nil {
			out = append(out, obj)
		}
	}
	return out
}

func stringSet(values []any) map[string]bool {
	out := map[string]bool{}
	for _, value := range values {
		if text := stringValue(value); text != "" {
			out[text] = true
		}
	}
	return out
}

func unsupportedHardwareName(kind, name, source string) error {
	return cliError{"unsupported_hardware", fmt.Sprintf("%s %q is not in the current LocalMaxxing allowlist.", kind, name), []string{"Run lmx context --out context.json and choose a value from " + source + ".", "If this hardware should be accepted, request an allowlist update before submitting."}, nil}
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
	benchmarkStatus := "incomplete"
	if ready {
		benchmarkStatus = "completed"
	} else if dryRun {
		benchmarkStatus = "planned"
	}
	submissionStatus := "ready"
	if submit {
		submissionStatus = "submitting"
	} else if requiresRemoteHardware {
		submissionStatus = "needs_remote_hardware"
	} else if missingAuth {
		submissionStatus = "needs_auth"
	} else if !ready {
		submissionStatus = "needs_metrics"
	}
	status := "ready_for_api_validation"
	message := "Benchmark payload is ready for API validation."
	nextCommand := "lmx benchmark dry-run " + out
	if submit {
		status = "api_submission_requested"
		message = "Benchmark payload is being sent to the API."
		nextCommand = ""
	} else if ready && requiresRemoteHardware {
		status = "needs_remote_hardware"
		message = "Benchmark completed successfully. Submission is not yet ready: server hardware metadata is missing. Generate it on the endpoint host, then attach it without rerunning."
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
		"status":           status,
		"message":          message,
		"outputPath":       out,
		"canApiValidate":   canUseAPI,
		"canSubmit":        canUseAPI,
		"requiresMetrics":  !ready,
		"benchmarkStatus":  benchmarkStatus,
		"submissionStatus": submissionStatus,
	}
	if requiresRemoteHardware {
		feedback["requiresHardware"] = true
		feedback["hardwareCommand"] = "lmx hardware --out hardware.json"
		feedback["attachHardwareCommand"] = "lmx benchmark add-hardware " + out + " --hardware hardware.json"
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

type chatProbe struct {
	started      time.Time
	firstTokenAt time.Time
	lastTokenAt  time.Time
	completedAt  time.Time
	outputText   string
	usage        map[string]any
}

// timedChatCompletion posts one chat completion and returns client-observed
// timing. Streaming requests record first/last token arrival times.
func timedChatCompletion(args cliArgs, baseURL string, body map[string]any, timeout time.Duration, errorCode string) (chatProbe, error) {
	stream, _ := body["stream"].(bool)
	bodyData, err := json.Marshal(body)
	if err != nil {
		return chatProbe{}, err
	}
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	started := time.Now()
	req, err := http.NewRequestWithContext(ctx, "POST", baseURL+"/v1/chat/completions", bytes.NewReader(bodyData))
	if err != nil {
		return chatProbe{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	if key := opt(args, "model-api-key"); key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		return chatProbe{}, cliError{errorCode, fmt.Sprintf("Could not reach OpenAI-compatible endpoint: %v", err), []string{"Check --base-url and confirm the endpoint is reachable from this machine."}, nil}
	}
	defer res.Body.Close()
	if res.StatusCode < 200 || res.StatusCode >= 300 {
		text, _ := io.ReadAll(res.Body)
		return chatProbe{}, cliError{errorCode, fmt.Sprintf("OpenAI-compatible endpoint returned %s", res.Status), []string{"Check --base-url, --served-model, and --model-api-key.", "Confirm the endpoint supports POST /v1/chat/completions."}, string(text)}
	}
	probe := chatProbe{started: started}
	if stream {
		streamResult, err := readOpenAIStream(args, res.Body, started)
		if err != nil {
			return chatProbe{}, err
		}
		probe.firstTokenAt = streamResult.firstTokenAt
		probe.lastTokenAt = streamResult.lastTokenAt
		probe.completedAt = streamResult.completedAt
		probe.outputText = streamResult.outputText
		probe.usage = streamResult.usage
	} else {
		var response map[string]any
		if err := json.NewDecoder(res.Body).Decode(&response); err != nil {
			return chatProbe{}, err
		}
		probe.outputText = nonStreamingContent(response)
		probe.usage = asObject(response["usage"])
		probe.completedAt = time.Now()
	}
	if probe.completedAt.IsZero() {
		probe.completedAt = time.Now()
	}
	return probe, nil
}

// decodeThroughput computes decode tokens/s, preferring the inter-token
// window (first to last token, excluding the first token from the count) and
// falling back to the full generation window when only one token arrived.
func decodeThroughput(probe chatProbe, outputTokens int) (float64, string, bool) {
	if outputTokens > 1 && !probe.firstTokenAt.IsZero() && probe.lastTokenAt.After(probe.firstTokenAt) {
		decodeMs := durationMS(probe.lastTokenAt.Sub(probe.firstTokenAt))
		if decodeMs > 0 {
			return round1(float64(outputTokens-1) / (decodeMs / 1000)), "inter_token", true
		}
	}
	generationStart := probe.started
	if !probe.firstTokenAt.IsZero() {
		generationStart = probe.firstTokenAt
	}
	generationMs := durationMS(probe.completedAt.Sub(generationStart))
	if generationMs <= 0 || outputTokens <= 0 {
		return 0, "", false
	}
	return round1(float64(outputTokens) / (generationMs / 1000)), "request_window", true
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
	warmup, err := intOption(args, 1, 0, "warmup")
	if err != nil {
		return nil, err
	}
	iterations, err := intOption(args, 3, 1, "iterations", "iters")
	if err != nil {
		return nil, err
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
	revision := firstNonEmpty(opt(args, "model-revision"), "main")
	for i := 0; i < warmup; i++ {
		printStatus(args, "endpoint_warmup_request", map[string]any{"index": i + 1, "warmup": warmup})
		if _, err := timedChatCompletion(args, baseURL, body, timeout, "endpoint_benchmark_failed"); err != nil {
			return nil, err
		}
	}
	promptTokens := 0
	promptTokenSource := ""
	outputTokenSource := ""
	outputText := ""
	decodeSource := ""
	samples := make([]map[string]any, 0, iterations)
	ttftValues := []float64{}
	tokSOutValues := []float64{}
	tokSTotalValues := []float64{}
	tokSPrefillValues := []float64{}
	outputTokenValues := []float64{}
	for i := 0; i < iterations; i++ {
		probe, err := timedChatCompletion(args, baseURL, body, timeout, "endpoint_benchmark_failed")
		if err != nil {
			return nil, err
		}
		if promptTokens == 0 {
			result, err := tokenCount(args, hfID, revision, prompt, usageToken(probe.usage, "prompt_tokens"), "prompt")
			if err != nil {
				return nil, err
			}
			promptTokens = result.Count
			promptTokenSource = result.Source
		}
		outputResult, err := tokenCount(args, hfID, revision, probe.outputText, firstNonZero(usageToken(probe.usage, "completion_tokens"), usageToken(probe.usage, "output_tokens")), "output")
		if err != nil {
			return nil, err
		}
		outputTokens := outputResult.Count
		outputTokenSource = outputResult.Source
		outputText = probe.outputText
		sample := map[string]any{"iteration": i + 1, "outputTokens": float64(outputTokens)}
		outputTokenValues = append(outputTokenValues, float64(outputTokens))
		if totalMs := durationMS(probe.completedAt.Sub(probe.started)); totalMs > 0 {
			value := round1(float64(promptTokens+outputTokens) / (totalMs / 1000))
			sample["tokSTotal"] = value
			tokSTotalValues = append(tokSTotalValues, value)
		}
		if value, source, ok := decodeThroughput(probe, outputTokens); ok {
			sample["tokSOut"] = value
			tokSOutValues = append(tokSOutValues, value)
			decodeSource = source
		}
		if !probe.firstTokenAt.IsZero() {
			ttftMs := durationMS(probe.firstTokenAt.Sub(probe.started))
			sample["ttftMs"] = roundMetric(ttftMs)
			ttftValues = append(ttftValues, ttftMs)
			if ttftMs > 0 && promptTokens > 0 {
				value := round1(float64(promptTokens) / (ttftMs / 1000))
				sample["tokSPrefill"] = value
				tokSPrefillValues = append(tokSPrefillValues, value)
			}
		}
		samples = append(samples, sample)
		printStatus(args, "endpoint_iteration_complete", map[string]any{"iteration": i + 1, "iterations": iterations, "tokSOut": sample["tokSOut"], "ttftMs": sample["ttftMs"]})
	}
	printStatus(args, "token_count_source", map[string]any{"prompt": promptTokenSource, "output": outputTokenSource, "promptTokens": promptTokens})
	metrics := map[string]any{
		"prompt":       prompt,
		"outputText":   outputText,
		"promptTokens": float64(promptTokens),
		"engineFlags":  map[string]any{"mode": "remote", "baseUrl": baseURL, "servedModel": servedModel, "servedModelSource": servedModelSource, "stream": stream, "maxTokens": maxTokens, "timeoutSeconds": int(timeout.Seconds()), "warmup": warmup, "iterations": iterations},
		"tokenSources": map[string]any{"prompt": promptTokenSource, "output": outputTokenSource},
		"timingSource": "client_observed_http",
		"metricSource": "remote_endpoint",
		"ttftSource":   "unavailable_no_stream",
	}
	if len(outputTokenValues) > 0 {
		metrics["outputTokens"] = medianOf(outputTokenValues)
	}
	if len(tokSOutValues) > 0 {
		metrics["tokSOut"] = roundMetric(medianOf(tokSOutValues))
		metrics["tokSOutSource"] = decodeSource
	}
	if len(tokSTotalValues) > 0 {
		metrics["tokSTotal"] = roundMetric(medianOf(tokSTotalValues))
	}
	if len(ttftValues) > 0 {
		metrics["ttftMs"] = roundMetric(medianOf(ttftValues))
		metrics["ttftSource"] = "stream_first_token"
	}
	if len(tokSPrefillValues) > 0 {
		metrics["tokSPrefill"] = roundMetric(medianOf(tokSPrefillValues))
		metrics["tokSPrefillSource"] = "estimated_from_ttft"
	}
	if iterations > 1 {
		metrics["samples"] = samples
		metrics["sampleStats"] = map[string]any{
			"tokSOut":   metricSummary(tokSOutValues),
			"tokSTotal": metricSummary(tokSTotalValues),
			"ttftMs":    metricSummary(ttftValues),
		}
	}
	if modelResolution != nil {
		metrics["modelResolution"] = modelResolution
	}
	if quantizationResolution != nil {
		metrics["quantizationResolution"] = quantizationResolution
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
	warmup, err := intOption(args, 1, 0, "warmup")
	if err != nil {
		return nil, err
	}
	iterations, err := intOption(args, 3, 1, "iterations", "iters")
	if err != nil {
		return nil, err
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
	for i := 0; i < warmup; i++ {
		printStatus(args, "ollama_warmup_request", map[string]any{"index": i + 1, "warmup": warmup})
		if _, _, err := ollamaGenerate(args, baseURL, body, timeout); err != nil {
			return nil, err
		}
	}
	promptTokenValues := []float64{}
	outputTokenValues := []float64{}
	tokSPrefillValues := []float64{}
	tokSOutValues := []float64{}
	tokSTotalValues := []float64{}
	ttftValues := []float64{}
	samples := make([]map[string]any, 0, iterations)
	outputText := ""
	for i := 0; i < iterations; i++ {
		response, elapsed, err := ollamaGenerate(args, baseURL, body, timeout)
		if err != nil {
			return nil, err
		}
		outputText = stringValue(response["response"])
		promptTokens := firstPositiveNumber(response, "prompt_eval_count", "promptEvalCount")
		outputTokens := firstPositiveNumber(response, "eval_count", "evalCount")
		promptSeconds := ollamaDurationSeconds(firstPositiveNumber(response, "prompt_eval_duration", "promptEvalDuration"))
		decodeSeconds := ollamaDurationSeconds(firstPositiveNumber(response, "eval_duration", "evalDuration"))
		totalSeconds := ollamaDurationSeconds(firstPositiveNumber(response, "total_duration", "totalDuration"))
		if totalSeconds == 0 {
			totalSeconds = elapsed.Seconds()
		}
		sample := map[string]any{"iteration": i + 1}
		if promptTokens > 0 {
			sample["promptTokens"] = promptTokens
			promptTokenValues = append(promptTokenValues, promptTokens)
		}
		if outputTokens > 0 {
			sample["outputTokens"] = outputTokens
			outputTokenValues = append(outputTokenValues, outputTokens)
		}
		if promptTokens > 0 && promptSeconds > 0 {
			value := round1(promptTokens / promptSeconds)
			sample["tokSPrefill"] = value
			tokSPrefillValues = append(tokSPrefillValues, value)
		}
		if outputTokens > 0 && decodeSeconds > 0 {
			value := round1(outputTokens / decodeSeconds)
			sample["tokSOut"] = value
			tokSOutValues = append(tokSOutValues, value)
		}
		if promptTokens+outputTokens > 0 && totalSeconds > 0 {
			value := round1((promptTokens + outputTokens) / totalSeconds)
			sample["tokSTotal"] = value
			tokSTotalValues = append(tokSTotalValues, value)
		}
		if promptSeconds > 0 {
			value := round1(promptSeconds * 1000)
			sample["ttftMs"] = value
			ttftValues = append(ttftValues, value)
		}
		samples = append(samples, sample)
		printStatus(args, "ollama_iteration_complete", map[string]any{"iteration": i + 1, "iterations": iterations, "tokSOut": sample["tokSOut"], "ttftMs": sample["ttftMs"]})
	}
	metrics := map[string]any{
		"prompt":       prompt,
		"outputText":   outputText,
		"engineFlags":  map[string]any{"mode": "remote", "baseUrl": baseURL, "servedModel": servedModel, "nativeApi": "ollama_generate", "maxTokens": maxTokens, "timeoutSeconds": int(timeout.Seconds()), "warmup": warmup, "iterations": iterations},
		"tokenSources": map[string]any{"prompt": "ollama_prompt_eval_count", "output": "ollama_eval_count"},
		"timingSource": "ollama_native_api",
		"metricSource": "remote_endpoint",
		"ttftSource":   "unavailable_ollama_nonstreaming",
	}
	if len(promptTokenValues) > 0 {
		metrics["promptTokens"] = medianOf(promptTokenValues)
	}
	if len(outputTokenValues) > 0 {
		metrics["outputTokens"] = medianOf(outputTokenValues)
	}
	if len(tokSPrefillValues) > 0 {
		metrics["tokSPrefill"] = roundMetric(medianOf(tokSPrefillValues))
	}
	if len(tokSOutValues) > 0 {
		metrics["tokSOut"] = roundMetric(medianOf(tokSOutValues))
	}
	if len(tokSTotalValues) > 0 {
		metrics["tokSTotal"] = roundMetric(medianOf(tokSTotalValues))
	}
	if len(ttftValues) > 0 {
		metrics["ttftMs"] = roundMetric(medianOf(ttftValues))
		metrics["ttftSource"] = "ollama_prompt_eval_duration"
	}
	if iterations > 1 {
		metrics["samples"] = samples
		metrics["sampleStats"] = map[string]any{
			"tokSOut": metricSummary(tokSOutValues),
			"ttftMs":  metricSummary(ttftValues),
		}
	}
	return metrics, nil
}

func ollamaGenerate(args cliArgs, baseURL string, body map[string]any, timeout time.Duration) (map[string]any, time.Duration, error) {
	bodyData, err := json.Marshal(body)
	if err != nil {
		return nil, 0, err
	}
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	started := time.Now()
	req, err := http.NewRequestWithContext(ctx, "POST", baseURL+"/api/generate", bytes.NewReader(bodyData))
	if err != nil {
		return nil, 0, err
	}
	req.Header.Set("Content-Type", "application/json")
	if key := opt(args, "model-api-key"); key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, 0, cliError{"endpoint_benchmark_failed", fmt.Sprintf("Could not reach Ollama endpoint: %v", err), []string{"Check --base-url and confirm Ollama is serving from this machine."}, nil}
	}
	defer res.Body.Close()
	if res.StatusCode < 200 || res.StatusCode >= 300 {
		text, _ := io.ReadAll(res.Body)
		return nil, 0, cliError{"endpoint_benchmark_failed", fmt.Sprintf("Ollama endpoint returned %s", res.Status), []string{"Check --base-url, --served-model, and --model-api-key.", "Confirm the endpoint supports POST /api/generate."}, string(text)}
	}
	var response map[string]any
	if err := json.NewDecoder(res.Body).Decode(&response); err != nil {
		return nil, 0, err
	}
	return response, time.Since(started), nil
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
	query, querySource := remoteModelSearchQuery(servedModel, stringValue(resolution["loadedFilename"]))
	resolution["searchQuery"] = query
	resolution["searchQuerySource"] = querySource
	if querySource == "loaded_filename" {
		printStatus(args, "remote_model_query_from_filename", map[string]any{"servedModel": servedModel, "query": query, "hint": "Served model alias looks generic; searching by the loaded GGUF filename instead."})
	}
	queryCommand := "lmx model search " + shellQuote(query)
	value, err := searchModels(args, query, 5)
	if err != nil {
		resolution["searchError"] = err.Error()
		resolution["searchCommand"] = queryCommand
		printStatus(args, "hf_id_search_unavailable", map[string]any{"query": query, "next": queryCommand})
		return resolution
	}
	candidates := modelCandidates(value, 5)
	resolution["candidates"] = candidates
	resolution["searchCommand"] = queryCommand
	if len(candidates) == 0 {
		printStatus(args, "hf_id_candidates_empty", map[string]any{"query": query, "next": queryCommand})
		return resolution
	}
	fields := map[string]any{"query": query, "count": len(candidates), "next": "If the exact GGUF repo matters, rerun with that --hf-id."}
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

func resolveEvalModelID(args cliArgs, declared string) (string, map[string]any) {
	declared = strings.TrimSpace(declared)
	if declared == "" || declared == "<required-before-submit>" {
		return declared, nil
	}
	baseURL := opt(args, "base-url")
	if baseURL != "" {
		servedModel, info, err := detectServedModel(baseURL, opt(args, "model-api-key"), firstNonEmpty(opt(args, "served-model"), opt(args, "model-name"), declared))
		if err == nil {
			modelPath := ""
			normalizedBaseURL := openAIBaseURL(baseURL)
			if props, err := fetchEndpointJSON(normalizedBaseURL+"/props", opt(args, "model-api-key")); err == nil {
				modelPath = stringValue(asObject(props)["model_path"])
			}
			if modelPath == "" {
				modelPath = firstNonEmpty(stringValue(info["filename"]), stringValue(info["model_path"]), stringValue(info["path"]))
			}
			resolution := remoteModelResolution(args, servedModel, "v1_models", declared, modelPath)
			if resolved := resolvedHFIDFromModelResolution(resolution); resolved != "" {
				printStatus(args, "eval_hf_id_resolved", map[string]any{"declared": declared, "resolved": resolved, "servedModel": servedModel})
				return resolved, resolution
			}
			if modelPath != "" {
				if sourceRepo, err := sourceRepoFromFilename(args, nil, filepath.Base(modelPath)); err == nil && sourceRepo != "" {
					resolution := map[string]any{
						"hfId":              declared,
						"servedModel":       servedModel,
						"servedModelSource": "v1_models",
						"loadedFilename":    filepath.Base(modelPath),
						"sourceRepo":        sourceRepo,
						"sourceRepoMatch":   "filename_derived",
						"status":            "source_repo_detected",
					}
					printStatus(args, "eval_hf_id_resolved", map[string]any{"declared": declared, "resolved": sourceRepo, "servedModel": servedModel, "filename": filepath.Base(modelPath)})
					return sourceRepo, resolution
				}
			}
			if resolution != nil {
				return declared, resolution
			}
		} else {
			printStatus(args, "eval_model_detection_unavailable", map[string]any{"baseUrl": baseURL, "reason": err.Error()})
		}
	}
	// If the user passed an endpoint alias (no org/name), search LocalMaxxing's
	// model index and use the first candidate, matching benchmark UX.
	if !strings.Contains(declared, "/") || genericModelAliases[strings.ToLower(declared)] {
		query, _ := normalizedModelSearchQuery(declared)
		value, err := searchModels(args, query, 5)
		if err == nil {
			candidates := modelCandidates(value, 5)
			if len(candidates) > 0 {
				resolved := candidateRepoID(candidates[0])
				if resolved != "" {
					resolution := map[string]any{"hfId": declared, "servedModel": declared, "searchQuery": query, "status": "search_candidate", "candidates": candidates}
					printStatus(args, "eval_hf_id_resolved", map[string]any{"declared": declared, "resolved": resolved, "query": query})
					return resolved, resolution
				}
			}
		}
	}
	return declared, nil
}

func resolvedHFIDFromModelResolution(resolution map[string]any) string {
	if resolution == nil {
		return ""
	}
	if sourceRepo := stringValue(resolution["sourceRepo"]); sourceRepo != "" {
		return sourceRepo
	}
	for _, candidate := range anySlice(resolution["candidates"]) {
		if repo := candidateRepoID(candidate); repo != "" {
			return repo
		}
	}
	return ""
}

// remoteModelSearchQuery picks the HF search query for resolving the served
// model: the served alias when it looks like a real model name, otherwise a
// name derived from the GGUF filename the endpoint reports as loaded.
func remoteModelSearchQuery(servedModel, loadedFilename string) (string, string) {
	if !canonicalModelAlias(servedModel) {
		if derived := modelNameFromGGUFFilename(loadedFilename); derived != "" {
			return derived, "loaded_filename"
		}
	}
	return servedModel, "served_model"
}

var genericModelAliases = map[string]bool{"default": true, "model": true, "local": true, "local-model": true, "localmodel": true, "gguf": true, "unknown": true, "custom": true}

// canonicalModelAlias reports whether a served model id looks like a real
// model name rather than a generic alias or a filesystem path (llama.cpp
// serves the loaded model path as the model id when no alias is set).
func canonicalModelAlias(name string) bool {
	name = strings.TrimSpace(name)
	if name == "" || genericModelAliases[strings.ToLower(name)] {
		return false
	}
	if strings.EqualFold(filepath.Ext(name), ".gguf") || strings.Contains(name, `\`) {
		return false
	}
	// HF repo ids have exactly one slash; deeper or rooted means a path.
	if strings.Count(name, "/") > 1 || strings.HasPrefix(name, "/") || strings.HasPrefix(name, "./") || strings.HasPrefix(name, "../") {
		return false
	}
	return true
}

// quantCorePattern matches a single quantization token: GGUF/IQ levels
// (Q4_K_M, Q8_0, IQ4_NL) or float formats with an optional NVIDIA/MX vendor
// prefix (FP16, BF16, FP8, FP4, NVFP4, MXFP4, MXFP6, MXFP8).
const quantCorePattern = `(?:IQ|Q)[0-9][A-Z0-9_]*|(?:(?:NV|MX)FP|BF|F|FP)[0-9]+`

var (
	ggufShardSuffix    = regexp.MustCompile(`(?i)-\d{5}-of-\d{5}$`)
	ggufQuantToken     = regexp.MustCompile(`(?i)^(?:` + quantCorePattern + `)$`)
	filenameQuantToken = regexp.MustCompile(`(?i)(?:^|[-_.])(` + quantCorePattern + `)(?:[-_.]|$)`)
)

// modelNameFromGGUFFilename strips the extension, shard suffix, and trailing
// quantization/packaging tokens from a GGUF filename, leaving a model name
// usable as a search query.
func modelNameFromGGUFFilename(path string) string {
	if path == "" {
		return ""
	}
	if idx := strings.LastIndexByte(path, '\\'); idx >= 0 {
		path = path[idx+1:]
	}
	name := strings.TrimSuffix(filepath.Base(path), filepath.Ext(path))
	name = ggufShardSuffix.ReplaceAllString(name, "")
	tokens := strings.Split(name, "-")
	for len(tokens) > 1 && droppableGGUFNameToken(tokens[len(tokens)-1]) {
		tokens = tokens[:len(tokens)-1]
	}
	return strings.Join(tokens, "-")
}

func droppableGGUFNameToken(token string) bool {
	if ggufQuantToken.MatchString(token) {
		return true
	}
	switch strings.ToLower(token) {
	case "ud", "gguf", "imat", "imatrix":
		return true
	}
	return false
}

// normalizedModelSearchQuery rewrites a GGUF filename or filesystem path
// passed as a search query into the model name it embeds. Plain queries and
// HF repo ids (org/name) pass through untouched.
func normalizedModelSearchQuery(query string) (string, bool) {
	trimmed := strings.TrimSpace(query)
	if trimmed == "" {
		return query, false
	}
	if !strings.EqualFold(filepath.Ext(trimmed), ".gguf") && canonicalModelAlias(trimmed) {
		return query, false
	}
	if derived := modelNameFromGGUFFilename(trimmed); derived != "" && !strings.EqualFold(derived, trimmed) {
		return derived, true
	}
	return query, false
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
	if idx := strings.Index(lower, "-ud-"); idx > 0 {
		return []string{"unsloth/" + name[:idx] + "-GGUF"}
	}
	if idx := strings.Index(lower, "-qat-"); idx > 0 {
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
	matches := filenameQuantToken.FindAllStringSubmatch(name, -1)
	if len(matches) == 0 {
		return ""
	}
	return strings.ToUpper(matches[len(matches)-1][1])
}

func discoverServerMetadata(baseURL, apiKey string, info map[string]any) map[string]any {
	meta := map[string]any{}
	if q := quantizationFromModelInfo(info); q != "" {
		meta["quantization"] = q
	}
	if props, err := fetchEndpointJSON(baseURL+"/props", apiKey); err == nil {
		if obj := asObject(props); obj != nil {
			if mp := stringValue(obj["model_path"]); mp != "" {
				meta["modelPath"] = mp
			}
			copyKnownMetaFields(meta, obj)
		}
	}
	if hw, err := fetchEndpointJSON(baseURL+"/hardware", apiKey); err == nil {
		if obj := asObject(hw); obj != nil {
			copyKnownMetaFields(meta, obj)
		}
	}
	return meta
}

func copyKnownMetaFields(dst, src map[string]any) {
	for _, name := range []string{"gpuName", "cpu", "os", "modelPath", "quantization", "engineName", "engineVersion"} {
		if _, exists := dst[name]; exists {
			continue
		}
		if value := stringValue(src[name]); value != "" {
			dst[name] = value
		}
	}
	for _, name := range []string{"gpuCount", "vramGb", "ramGb"} {
		if _, exists := dst[name]; exists {
			continue
		}
		if value := numberField(src, name); value > 0 {
			dst[name] = value
		}
	}
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
	lastTokenAt  time.Time
	completedAt  time.Time
	outputText   string
	usage        map[string]any
}

func readOpenAIStream(args cliArgs, body io.Reader, started time.Time) (openAIStreamResult, error) {
	result := openAIStreamResult{}
	var output strings.Builder
	reader := bufio.NewReaderSize(body, 64*1024)
	for {
		line, err := reader.ReadString('\n')
		if line != "" {
			consumeOpenAIStreamLine(args, strings.TrimSpace(line), started, &result, &output)
		}
		if err == io.EOF {
			break
		}
		if err != nil {
			return openAIStreamResult{}, err
		}
	}
	result.outputText = output.String()
	result.completedAt = time.Now()
	return result, nil
}

func consumeOpenAIStreamLine(args cliArgs, line string, started time.Time, result *openAIStreamResult, output *strings.Builder) {
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
	now := time.Now()
	if result.firstTokenAt.IsZero() {
		result.firstTokenAt = now
		printStatus(args, "first_token_received", map[string]any{"ttftMs": roundMetric(durationMS(now.Sub(started)))})
	}
	result.lastTokenAt = now
	output.WriteString(content)
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

//go:embed token_count.py
var tokenCountScript string

func pythonTokenCount(model, revision, text string) (int, error) {
	response, err := runPythonTokenHelper(map[string]any{"model": model, "revision": revision, "text": text})
	if err != nil {
		return 0, err
	}
	if tokens, ok := response["tokens"].(float64); ok {
		return int(tokens), nil
	}
	return 0, errors.New("token helper did not return tokens")
}

func pythonExactTokenText(model, revision string, targetTokens int, seedText string) (string, int, error) {
	response, err := runPythonTokenHelper(map[string]any{"model": model, "revision": revision, "target_tokens": targetTokens, "seed_text": seedText})
	if err != nil {
		return "", 0, err
	}
	text := stringValue(response["text"])
	tokensFloat, ok := response["tokens"].(float64)
	if !ok || text == "" {
		return "", 0, errors.New("token helper did not return exact text")
	}
	return text, int(tokensFloat), nil
}

func runPythonTokenHelper(request map[string]any) (map[string]any, error) {
	script, err := tokenCountScriptPath()
	if err != nil {
		return nil, err
	}
	data, _ := json.Marshal(request)
	var lastErr error
	for _, python := range []string{"python3", "python"} {
		if _, ok := lookupExecutable(python); !ok {
			lastErr = fmt.Errorf("%s not found on PATH", python)
			continue
		}
		cmd := exec.Command(python, script)
		cmd.Stdin = bytes.NewReader(data)
		var stdout bytes.Buffer
		var stderr bytes.Buffer
		cmd.Stdout = &stdout
		cmd.Stderr = &stderr
		if err := cmd.Run(); err != nil {
			lastErr = fmt.Errorf("%v: %s", err, strings.TrimSpace(stderr.String()))
			continue
		}
		var response map[string]any
		if err := json.Unmarshal(stdout.Bytes(), &response); err != nil {
			return nil, err
		}
		return response, nil
	}
	if lastErr == nil {
		lastErr = errors.New("no python interpreter found on PATH")
	}
	return nil, lastErr
}

// tokenCountScriptPath materializes the embedded tokenizer helper to a stable
// temp path so the installed binary works outside a repository checkout.
func tokenCountScriptPath() (string, error) {
	path := filepath.Join(os.TempDir(), "localmaxxing-token-count.py")
	if data, err := os.ReadFile(path); err == nil && string(data) == tokenCountScript {
		return path, nil
	}
	if err := os.WriteFile(path, []byte(tokenCountScript), 0o644); err != nil {
		return "", err
	}
	return path, nil
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

// durationMS converts a duration to fractional milliseconds without the
// integer truncation (and 1ms floor) that previously biased sub-ms timings.
func durationMS(duration time.Duration) float64 {
	return float64(duration.Nanoseconds()) / 1e6
}

func firstNonNil(values ...any) any {
	for _, value := range values {
		if value != nil {
			return value
		}
	}
	return nil
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

func runBenchmarkCommand(args cliArgs, commandSnippet string) (string, string, error) {
	ctx := context.Background()
	cancel := func() {}
	if value := opt(args, "command-timeout-seconds"); value != "" {
		seconds, err := strconv.Atoi(value)
		if err != nil || seconds <= 0 {
			return "", "", cliError{"invalid_option", "--command-timeout-seconds must be a positive integer", []string{"Pass --command-timeout-seconds <seconds>."}, nil}
		}
		ctx, cancel = context.WithTimeout(ctx, time.Duration(seconds)*time.Second)
	}
	defer cancel()
	var cmd *exec.Cmd
	if runtime.GOOS == "windows" {
		cmd = exec.Command("cmd", "/C", commandSnippet)
	} else {
		cmd = exec.Command("sh", "-c", commandSnippet)
	}
	configureCommandProcessGroup(cmd)
	var stdout bytes.Buffer
	var stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Start(); err != nil {
		return "", "", cliError{"benchmark_command_failed", "Benchmark command failed to start.", []string{"Check that the benchmark executable is installed and available on PATH."}, err.Error()}
	}
	done := make(chan error, 1)
	go func() {
		done <- cmd.Wait()
	}()
	var err error
	select {
	case err = <-done:
	case <-ctx.Done():
		killCommandProcessGroup(cmd)
		err = <-done
		return "", "", cliError{"benchmark_command_timeout", "Benchmark command timed out.", []string{"Raise --command-timeout-seconds or drop it to wait indefinitely."}, commandSnippet}
	}
	if err != nil {
		output := strings.TrimSpace(strings.Join([]string{stdout.String(), stderr.String()}, "\n"))
		return "", "", cliError{"benchmark_command_failed", "Benchmark command failed.", []string{"Check that the benchmark executable is installed and available on PATH.", "For llama.cpp, pass a complete llama-bench command with --command.", "For vLLM/SGLang, prefer their JSON output if available, then pass --results <path>."}, firstNonEmpty(output, err.Error())}
	}
	return strings.TrimSpace(stdout.String()), strings.TrimSpace(stderr.String()), nil
}

// parseBenchmarkLayers parses each output layer in priority order; earlier
// layers win, so structured result files beat stdout scraping beats stderr.
func parseBenchmarkLayers(layers ...string) map[string]float64 {
	merged := map[string]float64{}
	for _, layer := range layers {
		if strings.TrimSpace(layer) == "" {
			continue
		}
		for key, value := range parseBenchmarkOutput(layer) {
			if _, ok := merged[key]; !ok {
				merged[key] = value
			}
		}
	}
	return merged
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

// jsonNumberByAliases finds the first numeric value whose key matches an
// alias, searching breadth-first with sorted keys so shallow matches win and
// results are deterministic regardless of Go map iteration order.
func jsonNumberByAliases(value any, aliases []string) (float64, bool) {
	aliasSet := map[string]bool{}
	for _, alias := range aliases {
		aliasSet[normalizeMetricKey(alias)] = true
	}
	queue := []any{value}
	for len(queue) > 0 {
		current := queue[0]
		queue = queue[1:]
		switch typed := current.(type) {
		case map[string]any:
			keys := make([]string, 0, len(typed))
			for key := range typed {
				keys = append(keys, key)
			}
			sort.Strings(keys)
			for _, key := range keys {
				if aliasSet[normalizeMetricKey(key)] {
					if number, ok := anyNumber(typed[key]); ok {
						return number, true
					}
				}
			}
			for _, key := range keys {
				queue = append(queue, typed[key])
			}
		case []any:
			queue = append(queue, typed...)
		}
	}
	return 0, false
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
	hfID, modelResolution := resolveEvalModelID(args, firstNonEmpty(opt(args, "model"), "<required-before-submit>"))
	runConfig := map[string]any{"aggregatePreview": result["aggregate"]}
	if modelResolution != nil {
		runConfig["modelResolution"] = modelResolution
	}
	payload := map[string]any{
		"suiteSlug":     suiteSlug,
		"hfId":          hfID,
		"quantization":  opt(args, "quantization"),
		"executionMode": map[bool]string{true: "CUSTOM_LOCAL", false: "LM_EVAL_LOCAL"}[strings.EqualFold(runner, "CUSTOM")],
		"judgeMode":     map[bool]string{true: "LOCAL_REPORTED", false: "NONE"}[strings.EqualFold(stringValue(doc["scoringMethod"]), "llm_judge")],
		"runnerVersion": map[bool]string{true: "localmaxxing-go custom-local", false: "localmaxxing-go lm-eval-upload"}[strings.EqualFold(runner, "CUSTOM")],
		"results":       result["scores"],
		"artifacts":     redactGold(result["artifacts"]),
		"runConfig":     runConfig,
	}
	if hardwarePath := opt(args, "hardware"); hardwarePath != "" {
		hardware, err := readJSON(hardwarePath)
		if err != nil {
			return err
		}
		payload["hardware"] = normalizeHardwarePayload(hardware)
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

func handleEvalPull(suiteSlug string, args cliArgs) error {
	if suiteSlug == "" {
		return errors.New("eval pull requires a suite slug")
	}
	suite, err := loadSuiteForEvalRun(suiteSlug, args)
	if err != nil {
		return err
	}
	doc := suiteDoc(suite)
	slug := firstNonEmpty(stringValue(suite["slug"]), suiteSlug)
	outDir := firstNonEmpty(opt(args, "out"), "localmaxxing-eval-"+slug)
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return err
	}
	resolvedTasks := []any{}
	manifestTasks := []any{}
	totalItems := 0
	for _, task := range evalTasks(doc) {
		dataset := asObject(task["dataset"])
		if dataset == nil {
			resolvedTasks = append(resolvedTasks, task)
			manifestTasks = append(manifestTasks, map[string]any{"key": task["key"], "items": 0})
			continue
		}
		items, err := loadEvalDataset(dataset)
		if err != nil {
			return cliError{"dataset_pull_failed", fmt.Sprintf("Failed to pull dataset for task %q: %v", stringValue(task["key"]), err), []string{
				"Pass --api-key bhk_... so authenticated bucket datasets and gold labels are downloadable.",
				"Bucket download URLs expire after 15 minutes; re-run eval pull if it times out.",
			}, nil}
		}
		if len(items) > 0 {
			var b strings.Builder
			for _, it := range items {
				line, _ := json.Marshal(it)
				b.Write(line)
				b.WriteByte('\n')
			}
			if err := os.WriteFile(filepath.Join(outDir, stringValue(task["key"])+".jsonl"), []byte(b.String()), 0o644); err != nil {
				return err
			}
		}
		itemsAny := make([]any, len(items))
		for i := range items {
			itemsAny[i] = items[i]
		}
		newTask := map[string]any{}
		for k, v := range task {
			newTask[k] = v
		}
		newTask["dataset"] = map[string]any{"source": "inline", "items": itemsAny}
		resolvedTasks = append(resolvedTasks, newTask)
		manifestTasks = append(manifestTasks, map[string]any{"key": task["key"], "items": len(items)})
		totalItems += len(items)
	}
	offlineDoc := map[string]any{}
	for k, v := range doc {
		offlineDoc[k] = v
	}
	offlineDoc["tasks"] = resolvedTasks
	offlineSuite := map[string]any{
		"slug":        slug,
		"name":        firstNonEmpty(stringValue(suite["name"]), slug),
		"description": "Offline copy pulled from LocalMaxxing. Contains gold labels — do not publish.",
		"category":    "offline",
		"runner":      stringValue(suite["runner"]),
		"version":     firstNonEmpty(stringValue(doc["version"]), "1.0"),
		"suiteDoc":    offlineDoc,
	}
	if err := writeJSON(filepath.Join(outDir, "suite.json"), offlineSuite); err != nil {
		return err
	}
	manifest := map[string]any{
		"apiVersion":    "localmaxxing.evalPull.v1",
		"slug":          slug,
		"runner":        stringValue(suite["runner"]),
		"scoringMethod": stringValue(doc["scoringMethod"]),
		"pulledAt":      time.Now().UTC().Format(time.RFC3339),
		"apiUrl":        apiURL(args),
		"tasks":         manifestTasks,
	}
	if err := writeJSON(filepath.Join(outDir, "manifest.json"), manifest); err != nil {
		return err
	}
	printInfo("eval_pulled", map[string]any{"suite": slug, "outDir": outDir, "tasks": len(resolvedTasks), "items": totalItems, "containsLabels": true})
	fmt.Println("Gold labels are included in the .jsonl files and suite.json — do not publish them.")
	fmt.Println("Run fully offline (no site connection or API key needed):")
	fmt.Println("  lmx eval run " + slug + " --suite-file " + filepath.Join(outDir, "suite.json") + " --model <hfId> --base-url http://localhost:8000 --hardware hardware.json --out run.json")
	fmt.Println("Submit the saved run later, once the site is reachable:")
	fmt.Println("  lmx eval submit run.json --model <hfId> --hardware hardware.json --api-key bhk_...")
	return nil
}

func handleEvalSubmit(runFile string, args cliArgs) error {
	if runFile == "" {
		return errors.New("eval submit requires a saved run payload JSON path (written by eval run --out)")
	}
	value, err := readJSON(runFile)
	if err != nil {
		return err
	}
	payload := asObject(value)
	if payload == nil {
		return cliError{"invalid_run_payload", "Run payload must be a JSON object.", []string{"Pass a run payload written by eval run --out."}, value}
	}
	if stringValue(payload["suiteSlug"]) == "" {
		return cliError{"invalid_run_payload", fmt.Sprintf("%q is missing suiteSlug", runFile), []string{"Pass a run payload written by eval run --out."}, nil}
	}
	if model := opt(args, "model"); model != "" {
		payload["hfId"] = model
	}
	if hfID := stringValue(payload["hfId"]); hfID == "" || hfID == "<required-before-submit>" {
		return cliError{"missing_model", "Run payload has no hfId; pass --model <HuggingFace model id>", []string{"Pass --model here, or re-run eval run with --model."}, nil}
	}
	resolvedHFID, modelResolution := resolveEvalModelID(args, stringValue(payload["hfId"]))
	payload["hfId"] = resolvedHFID
	if modelResolution != nil {
		runConfig := asObject(payload["runConfig"])
		if runConfig == nil {
			runConfig = map[string]any{}
		}
		runConfig["modelResolution"] = modelResolution
		payload["runConfig"] = runConfig
	}
	if hardwarePath := opt(args, "hardware"); hardwarePath != "" {
		hardware, err := readJSON(hardwarePath)
		if err != nil {
			return err
		}
		payload["hardware"] = normalizeHardwarePayload(hardware)
	}
	if payload["hardware"] == nil {
		return cliError{"missing_hardware", "Run payload has no hardware; pass --hardware hardware.json", []string{"Create a hardware JSON file matching /api/agent-context hardwareSchemas and pass --hardware."}, nil}
	}
	endpoint := "/api/evals/runs"
	if hasFlag(args, "dry-run") {
		endpoint = "/api/evals/runs/dry-run"
	}
	return submitPayload(endpoint, hasFlag(args, "dry-run"), "run", args, payload)
}

type runShardConfig struct {
	maxTokens      int
	temperature    float64
	topP           float64
	extraction     string
	answerRegex    string
	promptTemplate string
	concurrency    int
	apiKey         string
	scoring        string
	dataset        string
	nSamples       int
	passK          int
	fewShot        int
	tempExplicit   bool
}

type shardItemResult struct {
	questionID         string
	itemIndex          int
	pass               bool
	scored             bool
	errText            string
	question           string
	prompt             string
	promptHash         string
	response           string
	reasoning          string
	thinkingRequested  string
	thinkingObserved   string
	predicted          string
	gold               string
	choices            []string
	choiceScores       map[string]float64
	scoreNormalization string
	latencyMs          int64
}

type shardStats struct {
	correct        int
	scored         int
	errors         int
	totalLatencyMs int64
}

// defaultShardScoring selects the canonical local scorer for official shard
// datasets. HellaSwag is conventionally scored by continuation likelihood, not
// by asking chat models to emit A/B/C/D.
func defaultShardScoring(dataset string) string {
	d := strings.ToLower(dataset)
	switch {
	case d == "hellaswag":
		return "loglikelihood"
	case strings.HasPrefix(d, "humaneval") || strings.HasPrefix(d, "mbpp"):
		return "code_execution"
	case d == "cruxeval":
		return "cruxeval_execution"
	default:
		return "exact_match"
	}
}

func isHellaSwagDataset(dataset string) bool {
	return strings.EqualFold(dataset, "hellaswag")
}

func chatOrInstructModelHint(model string, info map[string]any) string {
	candidates := []string{model}
	if info != nil {
		for _, key := range []string{"id", "name", "model", "root", "parent"} {
			candidates = append(candidates, stringValue(info[key]))
		}
	}
	for _, candidate := range candidates {
		normalized := strings.ToLower(strings.NewReplacer("_", "-", " ", "-").Replace(candidate))
		if normalized == "" {
			continue
		}
		for _, marker := range []string{"-instruct", "instruct-", "-chat", "chat-", "-it", "-sft", "-dpo", "-rlhf"} {
			if strings.Contains(normalized, marker) {
				return candidate
			}
		}
		if strings.HasSuffix(normalized, "-it") || strings.HasSuffix(normalized, "-instruct") || strings.HasSuffix(normalized, "-chat") {
			return candidate
		}
	}
	return ""
}

func shouldWarnHellaSwagDefaultLoglikelihood(dataset, scoring, explicitScoring, model string, info map[string]any) (string, bool) {
	if !isHellaSwagDataset(dataset) || scoring != "loglikelihood" || explicitScoring != "" {
		return "", false
	}
	hint := chatOrInstructModelHint(model, info)
	return hint, hint != ""
}

func printHellaSwagDefaultLoglikelihoodWarning(args cliArgs, modelHint string) {
	printStatus(args, "hellaswag_loglikelihood_chat_warning", map[string]any{
		"warning":       "HellaSwag default scoring is loglikelihood. For chat/instruct endpoints this may under-score due to prompt/template mismatch. If you intend generated multiple-choice answers, pass --scoring exact_match explicitly.",
		"modelHint":     modelHint,
		"canonical":     "For canonical continuation scoring, use a /v1/completions endpoint with echo prompt logprobs, or pass --model-path <model.gguf> to use the bundled llama_cpp_loglikelihood helper.",
		"helper":        "lmx-llama-score-hellaswag",
		"overrideExact": "--scoring exact_match",
	})
}

func normalizeShardScoring(value string) (string, error) {
	if value == "" {
		return "", nil
	}
	switch value {
	case "exact_match", "loglikelihood", "llama_cpp_loglikelihood", "code_execution", "cruxeval_execution":
		return value, nil
	default:
		return "", cliError{"invalid_shard_scoring", fmt.Sprintf("Unsupported shard scoring mode %q.", value), []string{"Use exact_match for chat answer matching, loglikelihood for OpenAI echo-logprobs endpoints, llama_cpp_loglikelihood with --model-path for local GGUF scoring, code_execution for HumanEval/MBPP, or cruxeval_execution for canonical CRUXEval execution checks."}, nil}
	}
}

func shardEvaluationConfig(meta map[string]any) map[string]any {
	if cfg := asObject(meta["evaluation"]); cfg != nil {
		return cfg
	}
	return nil
}

func shardConfigString(cliArgs cliArgs, meta map[string]any, optName, fieldName string) string {
	if value := opt(cliArgs, optName); value != "" {
		return value
	}
	return stringValue(meta[fieldName])
}

func shardConfigInt(cliArgs cliArgs, meta map[string]any, optName, fieldName string, fallback int) (int, error) {
	value, err := intOption(cliArgs, fallback, 0, optName)
	if err != nil {
		return 0, err
	}
	if opt(cliArgs, optName) == "" && value == fallback {
		if configured := int(numberField(meta, fieldName)); configured > 0 {
			return configured, nil
		}
	}
	return value, nil
}

func evalShardRunnerVersion(args cliArgs) string {
	return firstNonEmpty(opt(args, "runner-version"), "localmaxxing-go eval-shard")
}

func evalShardHarnessKeyForCoverage(args cliArgs) string {
	return evalShardRunnerVersion(args) + "|unknown-protocol|unknown-agent"
}

func intSetFromAny(value any) map[int]bool {
	set := map[int]bool{}
	for _, item := range anySlice(value) {
		n := 0
		switch v := item.(type) {
		case float64:
			n = int(v)
		case float32:
			n = int(v)
		case int:
			n = v
		case int64:
			n = int(v)
		case string:
			parsed, _ := strconv.Atoi(v)
			n = parsed
		}
		if n > 0 {
			set[n] = true
		}
	}
	return set
}

func sortedIntsFromSet(set map[int]bool) []int {
	values := make([]int, 0, len(set))
	for n := range set {
		values = append(values, n)
	}
	sort.Ints(values)
	return values
}

func missingShardIndexes(shardCount int, covered map[int]bool) []int {
	if shardCount <= 0 {
		return nil
	}
	missing := []int{}
	for i := 1; i <= shardCount; i++ {
		if !covered[i] {
			missing = append(missing, i)
		}
	}
	return missing
}

func copyStringMap(in map[string]string) map[string]string {
	out := make(map[string]string, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

func copyBoolMap(in map[string]bool) map[string]bool {
	out := make(map[string]bool, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

func fetchEvalShardCoverage(dataset, hfID, quantization, quantFormat string, args cliArgs) (map[string]any, error) {
	if hfID == "" {
		return nil, cliError{"missing_model", "eval shard status requires --model <HuggingFace model id>", []string{"Pass the same --model used for shard submissions."}, nil}
	}
	query := url.Values{}
	query.Set("hfId", hfID)
	query.Set("harnessKey", evalShardHarnessKeyForCoverage(args))
	if quantization != "" {
		query.Set("quantization", quantization)
	}
	if quantFormat != "" {
		query.Set("quantFormat", quantFormat)
	}
	value, err := fetchJSON("GET", apiURL(args)+"/api/evals/"+url.PathEscape(dataset)+"/coverage?"+query.Encode(), apiKey(args), nil)
	if err != nil {
		return nil, err
	}
	obj := asObject(value)
	if obj == nil {
		return nil, cliError{"invalid_coverage_response", "Eval shard coverage response was not a JSON object.", nil, value}
	}
	return obj, nil
}

func shardCoverageDetails(value any) (shardCount int, covered map[int]bool, missing []int) {
	obj := asObject(value)
	dataset := asObject(obj["dataset"])
	coverage := asObject(obj["coverage"])
	shardCount = int(numberField(dataset, "shardCount"))
	covered = intSetFromAny(coverage["shardsCovered"])
	missing = missingShardIndexes(shardCount, covered)
	return shardCount, covered, missing
}

func handleEvalShardStatus(dataset string, args cliArgs) error {
	if dataset == "" {
		return errors.New("eval shard status requires a dataset slug")
	}
	hfID := opt(args, "model")
	value, err := fetchEvalShardCoverage(dataset, hfID, opt(args, "quantization"), opt(args, "quant-format"), args)
	if err != nil {
		return err
	}
	shardCount, covered, missing := shardCoverageDetails(value)
	obj := asObject(value)
	coverage := asObject(obj["coverage"])
	fields := map[string]any{
		"dataset":             dataset,
		"model":               hfID,
		"quantization":        firstNonEmpty(opt(args, "quantization"), "(none)"),
		"quantFormat":         firstNonEmpty(opt(args, "quant-format"), "(none)"),
		"harnessKey":          evalShardHarnessKeyForCoverage(args),
		"shardCount":          shardCount,
		"coveredShards":       sortedIntsFromSet(covered),
		"missingShards":       missing,
		"uniqueQuestionCount": coverage["uniqueQuestionCount"],
		"questionsNeeded":     coverage["questionsNeeded"],
		"coverageMeaning":     "APPROVED aggregate shards for this dataset/model/quantization/quantFormat/harness key",
		"duplicateGuardHint":  "lmx eval shard " + dataset + " --model " + hfID + " --missing-only --submit",
	}
	if hasFlag(args, "json") || hasFlag(args, "print") || opt(args, "out") != "" {
		return writeOrPrintJSON("eval_shard_status", args, map[string]any{"status": fields, "raw": value})
	}
	printInfo("eval_shard_status", fields)
	return nil
}

// handleEvalShard runs a blob-backed eval-shard dataset against a local
// OpenAI-compatible endpoint, then prints a dry-run summary or submits
// question_id/pass pairs to LocalMaxxing.
func handleEvalShard(dataset string, args cliArgs) error {
	if dataset == "" {
		return errors.New("eval shard requires a dataset slug")
	}
	rawBaseURL, err := requireOpt(args, "base-url")
	if err != nil {
		return err
	}
	baseURL := openAIBaseURL(rawBaseURL)
	submit := hasFlag(args, "submit")

	requestedQuestions, err := intOption(args, 0, 1, "questions")
	if err != nil {
		return err
	}
	metaURL := apiURL(args) + "/api/evals/" + url.PathEscape(dataset) + "/shard"
	query := url.Values{}
	if shard := opt(args, "shard"); shard != "" {
		query.Set("shard", shard)
	}
	if requestedQuestions > 0 {
		query.Set("questions", strconv.Itoa(requestedQuestions))
	}
	if encoded := query.Encode(); encoded != "" {
		metaURL += "?" + encoded
	}
	meta, err := fetchJSON("GET", metaURL, apiKey(args), nil)
	if err != nil {
		return err
	}
	metaObj := asObject(meta)
	shardInfo := asObject(metaObj["shard"])
	downloadURL := stringValue(metaObj["downloadUrl"])
	if downloadURL == "" {
		return cliError{"shard_unavailable", "Shard response did not include a download URL.", []string{"Confirm the dataset exists and is approved.", "Confirm eval storage is configured on the LocalMaxxing instance."}, meta}
	}
	shardIndex := int(numberField(shardInfo, "shardIndex"))
	shardItemCount := int(numberField(shardInfo, "itemCount"))

	count := requestedQuestions
	if count == 0 {
		sampling := asObject(metaObj["sampling"])
		rec := asObject(sampling["recommendations"])
		count = int(numberField(rec, "margin05"))
		if count <= 0 {
			count = shardItemCount
		}
	}
	if shardItemCount > 0 && count > shardItemCount {
		count = shardItemCount
	}

	items, err := fetchDatasetItems(downloadURL, "jsonl")
	if err != nil {
		return cliError{"shard_download_failed", fmt.Sprintf("Could not download shard JSONL: %v", err), []string{"Signed download URLs expire after 15 minutes; re-run the command.", "Check network access to the storage host."}, nil}
	}
	if count > len(items) {
		count = len(items)
	}
	if count == 0 {
		return cliError{"empty_shard", "The shard contained no questions to run.", nil, nil}
	}
	items = items[:count]

	declaredModel := opt(args, "model")
	servedModel := opt(args, "served-model")
	var servedModelInfo map[string]any
	if servedModel == "" {
		if detected, info, derr := detectServedModel(baseURL, opt(args, "model-api-key"), declaredModel); derr == nil {
			servedModel = detected
			servedModelInfo = info
		}
	} else if _, info, derr := detectServedModel(baseURL, opt(args, "model-api-key"), servedModel); derr == nil {
		servedModelInfo = info
	}
	callModel := firstNonEmpty(servedModel, declaredModel, "local")

	// Pull the quantization from the endpoint exactly like `benchmark run`:
	// filename-derived (llama.cpp /props model_path) > /v1/models metadata >
	// --quantization, with the trusted value winning. This records the real quant
	// without the user passing a flag; --quantization still acts as an override
	// of last resort. Derive the GGUF container format from the model path too.
	quantResolution := remoteQuantizationResolution(args, baseURL, opt(args, "model-api-key"), opt(args, "quantization"), servedModelInfo)
	resolvedQuant := firstNonEmpty(stringValue(quantResolution["trusted"]), opt(args, "quantization"))
	resolvedQuantFormat := opt(args, "quant-format")
	if resolvedQuantFormat == "" && strings.EqualFold(filepath.Ext(stringValue(quantResolution["modelPath"])), ".gguf") {
		resolvedQuantFormat = "gguf"
	}

	evalConfig := shardEvaluationConfig(metaObj)

	maxTokens, err := shardConfigInt(args, evalConfig, "max-tokens", "maxNewTokens", 0)
	if err != nil {
		return err
	}
	// Default 0 = submit every scored question so the server's whole-shard trace
	// bundle is complete; the server samples its own Postgres preview rows.
	artifactLimit, err := intOption(args, 0, 0, "artifact-limit")
	if err != nil {
		return err
	}
	concurrency, err := intOption(args, 1, 1, "concurrency")
	if err != nil {
		return err
	}
	defaultScoring := defaultShardScoring(dataset)
	if strings.EqualFold(dataset, "hellaswag") && opt(args, "model-path") != "" {
		defaultScoring = "llama_cpp_loglikelihood"
	}
	explicitScoring := firstNonEmpty(opt(args, "scoring"), opt(args, "scoring-method"))
	scoring, err := normalizeShardScoring(firstNonEmpty(explicitScoring, stringValue(evalConfig["scoring"]), defaultScoring))
	if err != nil {
		return err
	}
	if modelHint, ok := shouldWarnHellaSwagDefaultLoglikelihood(dataset, scoring, explicitScoring, callModel, servedModelInfo); ok {
		printHellaSwagDefaultLoglikelihoodWarning(args, modelHint)
	}

	nSamples, err := intOption(args, 1, 1, "n-samples")
	if err != nil {
		return err
	}
	passK, err := intOption(args, 1, 1, "k")
	if err != nil {
		return err
	}
	if passK > nSamples {
		passK = nSamples
	}
	tempExplicit := opt(args, "temperature") != ""
	// pass@k with sampling uses a non-zero temperature by convention; pass@1 stays
	// greedy (temp 0) unless the user overrides --temperature.
	temperature := floatOption(args, "temperature", 0)
	if !tempExplicit && nSamples > 1 {
		temperature = 0.8
	}
	fewShot, err := intOption(args, defaultFewShot(dataset), 0, "few-shot")
	if err != nil {
		return err
	}
	cfg := runShardConfig{
		maxTokens:      maxTokens,
		temperature:    temperature,
		topP:           floatOption(args, "top-p", 1),
		extraction:     shardConfigString(args, evalConfig, "answer-extraction", "answerExtraction"),
		answerRegex:    shardConfigString(args, evalConfig, "answer-regex", "answerRegex"),
		promptTemplate: shardConfigString(args, evalConfig, "prompt-template", "promptTemplate"),
		concurrency:    concurrency,
		apiKey:         opt(args, "model-api-key"),
		scoring:        scoring,
		dataset:        dataset,
		nSamples:       nSamples,
		passK:          passK,
		fewShot:        fewShot,
		tempExplicit:   tempExplicit,
	}

	if submit {
		hfIDForCoverage := opt(args, "model")
		if hfIDForCoverage != "" {
			coverageValue, covErr := fetchEvalShardCoverage(dataset, hfIDForCoverage, resolvedQuant, resolvedQuantFormat, args)
			if covErr != nil {
				if hasFlag(args, "missing-only") || hasFlag(args, "all-missing") {
					return covErr
				}
				printStatus(args, "eval_shard_coverage_unavailable", map[string]any{"warning": covErr.Error(), "duplicateGuard": "skipped"})
			} else {
				shardCount, covered, missing := shardCoverageDetails(coverageValue)
				if hasFlag(args, "all-missing") && opt(args, "shard") == "" {
					if len(missing) == 0 {
						printInfo("eval_shard_all_missing_complete", map[string]any{"dataset": dataset, "model": hfIDForCoverage, "shardCount": shardCount, "coveredShards": sortedIntsFromSet(covered)})
						return nil
					}
					for _, nextShard := range missing {
						nextArgs := args
						nextArgs.opts = copyStringMap(args.opts)
						nextArgs.flags = copyBoolMap(args.flags)
						nextArgs.opts["shard"] = strconv.Itoa(nextShard)
						delete(nextArgs.flags, "all-missing")
						delete(nextArgs.flags, "missing-only")
						if err := handleEvalShard(dataset, nextArgs); err != nil {
							return err
						}
					}
					return nil
				}
				if hasFlag(args, "missing-only") && opt(args, "shard") == "" && covered[shardIndex] && len(missing) > 0 {
					nextArgs := args
					nextArgs.opts = copyStringMap(args.opts)
					nextArgs.flags = copyBoolMap(args.flags)
					nextArgs.opts["shard"] = strconv.Itoa(missing[0])
					delete(nextArgs.flags, "missing-only")
					return handleEvalShard(dataset, nextArgs)
				}
				if covered[shardIndex] && !hasFlag(args, "rerun") && !hasFlag(args, "force") {
					return cliError{"shard_already_submitted", fmt.Sprintf("Shard %d already has APPROVED coverage for this model/dataset/quantization/quantFormat/harness key.", shardIndex), []string{"Use --missing-only to run the next missing shard.", "Use --all-missing to walk every missing shard.", "Use --rerun or --force to submit another run for this shard."}, map[string]any{"dataset": dataset, "model": hfIDForCoverage, "shardIndex": shardIndex, "coveredShards": sortedIntsFromSet(covered), "missingShards": missing, "coverageMeaning": "APPROVED aggregate shards for this dataset/model/quantization/quantFormat/harness key"}}
				}
			}
		}
	}

	printInfo("eval_shard_start", map[string]any{"dataset": dataset, "shard": shardIndex, "questions": count, "model": callModel, "baseUrl": baseURL, "concurrency": concurrency, "scoring": scoring})

	var results []shardItemResult
	var stats shardStats
	var codeMetrics map[string]any
	if scoring == "llama_cpp_loglikelihood" {
		results, stats, err = runEvalShardLlamaCpp(args, items)
		if err != nil {
			return err
		}
	} else if scoring == "code_execution" {
		results, stats, codeMetrics, err = runEvalShardCodeExec(args, baseURL, callModel, items, cfg)
		if err != nil {
			return err
		}
		if len(codeMetrics) > 0 {
			printInfo("eval_shard_passk", codeMetrics)
		}
	} else if scoring == "cruxeval_execution" {
		results, stats, codeMetrics, err = runEvalShardCruxExec(args, baseURL, callModel, items, cfg)
		if err != nil {
			return err
		}
		if len(codeMetrics) > 0 {
			printInfo("eval_shard_execution", codeMetrics)
		}
	} else {
		results, stats = runEvalShard(args, baseURL, callModel, items, cfg)
	}
	accuracy := 0.0
	if stats.scored > 0 {
		accuracy = float64(stats.correct) / float64(stats.scored)
	}
	avgLatency := int64(0)
	if len(results) > 0 {
		avgLatency = stats.totalLatencyMs / int64(len(results))
	}
	summary := map[string]any{"dataset": dataset, "shardIndex": shardIndex, "questions": count, "correct": stats.correct, "scored": stats.scored, "errors": stats.errors, "accuracyPct": roundMetric(accuracy * 100), "avgLatencyMs": avgLatency, "quantization": resolvedQuant, "quantFormat": resolvedQuantFormat, "scoring": scoring}
	for mk, mv := range codeMetrics {
		summary[mk] = mv
	}
	submitResults := make([]any, 0, len(results))
	for _, r := range results {
		if !r.scored {
			continue
		}
		submitResults = append(submitResults, map[string]any{"question_id": r.questionID, "pass": r.pass})
	}

	// Submit one artifact per scored question by default (artifactLimit 0) so the
	// server can persist a complete whole-shard trace bundle; it caps the Postgres
	// preview rows itself. A positive --artifact-limit keeps only a balanced
	// sample (half passing, half failing). Send the full answer and reasoning.
	toArtifact := func(r shardItemResult) map[string]any {
		artifact := map[string]any{
			"question_id":       r.questionID,
			"itemIndex":         r.itemIndex,
			"promptHash":        r.promptHash,
			"question":          r.question,
			"prompt":            r.prompt,
			"response":          r.response,
			"reasoning":         r.reasoning,
			"thinkingRequested": r.thinkingRequested,
			"thinkingObserved":  r.thinkingObserved,
			"extractedAnswer":   r.predicted,
			"gold":              r.gold,
			"score":             boolScore(r.pass),
			"testPassed":        r.pass,
			"latencyMs":         r.latencyMs,
		}
		if len(r.choices) == 4 {
			artifact["choices"] = r.choices
		}
		if len(r.choiceScores) > 0 {
			artifact["choiceScores"] = r.choiceScores
		}
		if r.scoreNormalization != "" {
			artifact["scoreNormalization"] = r.scoreNormalization
		}
		return artifact
	}
	passCap, failCap := -1, -1
	if artifactLimit > 0 {
		failCap = artifactLimit / 2
		passCap = artifactLimit - failCap
	}
	submitArtifacts := make([]any, 0, len(results))
	passCount, failCount := 0, 0
	for _, r := range results {
		if !r.scored {
			continue
		}
		if r.pass {
			if passCap >= 0 && passCount >= passCap {
				continue
			}
			passCount++
		} else {
			if failCap >= 0 && failCount >= failCap {
				continue
			}
			failCount++
		}
		submitArtifacts = append(submitArtifacts, toArtifact(r))
	}

	if out := opt(args, "out"); out != "" {
		records := make([]any, len(results))
		for i, r := range results {
			records[i] = map[string]any{"question_id": r.questionID, "pass": r.pass, "scored": r.scored, "predicted": r.predicted, "gold": r.gold, "latencyMs": r.latencyMs, "error": r.errText, "question": r.question, "promptHash": r.promptHash, "prompt": r.prompt, "response": r.response, "reasoning": r.reasoning, "thinkingRequested": r.thinkingRequested, "thinkingObserved": r.thinkingObserved, "choices": r.choices, "choiceScores": r.choiceScores, "scoreNormalization": r.scoreNormalization}
		}
		if err := writeJSON(out, map[string]any{"summary": summary, "results": records}); err != nil {
			return err
		}
		printStatus(args, "eval_shard_results_written", map[string]any{"path": out, "containsLabels": true})
	}

	if !submit {
		printInfo("eval_shard_dry_run", summary)
		fmt.Println("Dry run only — nothing submitted.")
		fmt.Println("Publish with: lmx eval shard " + dataset + " --base-url " + rawBaseURL + " --model <hfId> --hardware hardware.json --submit")
		return nil
	}

	key := apiKey(args)
	if key == "" {
		return missingAPIKey("--api-key or LMX_API_KEY is required for eval shard --submit")
	}
	if declaredModel == "" {
		return cliError{"missing_model", "eval shard --submit requires --model <HuggingFace model id>", []string{"Pass --model org/name so the submission records a real model.", "Use lmx model search <name> to find the canonical id."}, nil}
	}
	hardwarePath := opt(args, "hardware")
	if hardwarePath == "" {
		return cliError{"missing_hardware", "eval shard --submit requires --hardware hardware.json", []string{"Run lmx hardware --out hardware.json and pass --hardware hardware.json."}, nil}
	}
	hardware, err := readJSON(hardwarePath)
	if err != nil {
		return err
	}
	if len(submitResults) == 0 {
		return cliError{"no_scored_questions", "Every question failed to score, so there is nothing to submit.", []string{"Check that the endpoint is reachable and returns completions.", "Inspect failures with --out results.json."}, nil}
	}
	hfID, modelResolution := resolveEvalModelID(args, declaredModel)
	runConfig := map[string]any{"accuracy": accuracy, "questionsRun": count, "errors": stats.errors, "avgLatencyMs": avgLatency, "answerExtraction": firstNonEmpty(cfg.extraction, "auto"), "artifactCount": len(submitArtifacts), "scoring": scoring}
	if scoring == "loglikelihood" {
		runConfig["answerExtraction"] = "none"
		runConfig["loglikelihoodTarget"] = "choice_text"
		runConfig["loglikelihoodNorm"] = "byte"
	}
	if modelResolution != nil {
		runConfig["modelResolution"] = modelResolution
	}
	if quantResolution != nil {
		runConfig["quantizationResolution"] = quantResolution
	}
	for mk, mv := range codeMetrics {
		runConfig[mk] = mv
	}
	payload := map[string]any{
		"hfId":          hfID,
		"modelRevision": firstNonEmpty(opt(args, "model-revision"), "main"),
		"hardware":      normalizeHardwarePayload(hardware),
		"shardIndex":    shardIndex,
		"results":       submitResults,
		"artifacts":     submitArtifacts,
		"runnerVersion": evalShardRunnerVersion(args),
		"runConfig":     runConfig,
	}
	if resolvedQuant != "" {
		payload["quantization"] = resolvedQuant
	}
	if resolvedQuantFormat != "" {
		payload["quantFormat"] = resolvedQuantFormat
	}
	if notes := opt(args, "notes"); notes != "" {
		payload["notes"] = notes
	}
	value, err := fetchJSON("POST", apiURL(args)+"/api/evals/"+url.PathEscape(dataset)+"/submit", key, payload)
	if err != nil {
		return err
	}
	if hasFlag(args, "json") || hasFlag(args, "print") || hasFlag(args, "verbose") {
		printJSON(value)
	}
	fields := map[string]any{"dataset": dataset, "shardIndex": shardIndex, "submitted": len(submitResults), "accuracyPct": summary["accuracyPct"]}
	if obj := asObject(value); obj != nil {
		if agg := asObject(obj["aggregate"]); agg != nil {
			fields["pooledScore"] = agg["pooledScore"]
			fields["ciLower"] = agg["ciLower"]
			fields["ciUpper"] = agg["ciUpper"]
			fields["aggregateCoverage"] = agg["shardsCovered"]
			fields["coverageMeaning"] = "APPROVED aggregate shards for this dataset/model/quantization/quantFormat/harness key"
		}
		if run := asObject(obj["run"]); run != nil {
			fields["runId"] = run["id"]
			fields["status"] = run["status"]
		}
	}
	printInfo("eval_shard_submitted", fields)
	return nil
}

func runEvalShard(args cliArgs, baseURL, model string, items []map[string]any, cfg runShardConfig) ([]shardItemResult, shardStats) {
	results := make([]shardItemResult, len(items))
	if cfg.concurrency < 1 {
		cfg.concurrency = 1
	}
	jobs := make(chan int)
	var wg sync.WaitGroup
	var mu sync.Mutex
	completed := 0
	worker := func() {
		defer wg.Done()
		for idx := range jobs {
			results[idx] = scoreShardItem(idx, items[idx], cfg, baseURL, model)
			mu.Lock()
			completed++
			done := completed
			mu.Unlock()
			if done%25 == 0 || done == len(items) {
				printStatus(args, "eval_shard_progress", map[string]any{"done": done, "total": len(items)})
			}
		}
	}
	for i := 0; i < cfg.concurrency; i++ {
		wg.Add(1)
		go worker()
	}
	for i := range items {
		jobs <- i
	}
	close(jobs)
	wg.Wait()

	stats := shardStats{}
	for _, r := range results {
		stats.totalLatencyMs += r.latencyMs
		if r.scored {
			stats.scored++
			if r.pass {
				stats.correct++
			}
		} else {
			stats.errors++
		}
	}
	return results, stats
}

func llamaScorerGPULayers(args cliArgs) string {
	if v := opt(args, "gpu-layers"); v != "" {
		return v
	}
	// The local HellaSwag scorer loads the GGUF directly; it does not reuse the
	// llama.cpp server process behind --base-url. When a server is already running,
	// default the scorer to CPU so it does not try to allocate a second copy of the
	// model in VRAM. Users with spare VRAM can still opt in with --gpu-layers.
	if opt(args, "base-url") != "" {
		return "0"
	}
	return ""
}

func runEvalShardLlamaCpp(args cliArgs, items []map[string]any) ([]shardItemResult, shardStats, error) {
	modelPath := opt(args, "model-path")
	if modelPath == "" {
		return nil, shardStats{}, cliError{"missing_model_path", "llama_cpp_loglikelihood scoring requires --model-path <model.gguf>.", []string{"Pass the same GGUF used by your llama.cpp server.", "Or use --scoring loglikelihood with a /v1/completions endpoint that returns echoed prompt token logprobs."}, nil}
	}
	tmp, err := os.CreateTemp("", "lmx-hellaswag-*.jsonl")
	if err != nil {
		return nil, shardStats{}, err
	}
	tmpPath := tmp.Name()
	defer os.Remove(tmpPath)
	enc := json.NewEncoder(tmp)
	for _, item := range items {
		if err := enc.Encode(item); err != nil {
			tmp.Close()
			return nil, shardStats{}, err
		}
	}
	if err := tmp.Close(); err != nil {
		return nil, shardStats{}, err
	}

	scorer := firstNonEmpty(opt(args, "llama-scorer"), opt(args, "scorer-bin"), bundledExecutable("lmx-llama-score-hellaswag"), "lmx-llama-score-hellaswag")
	cmdArgs := []string{"--model", modelPath, "--input", tmpPath}
	if v := opt(args, "context-length"); v != "" {
		cmdArgs = append(cmdArgs, "--ctx-size", v)
	}
	if v := opt(args, "ctx-size"); v != "" {
		cmdArgs = append(cmdArgs, "--ctx-size", v)
	}
	if v := opt(args, "batch-size"); v != "" {
		cmdArgs = append(cmdArgs, "--batch-size", v)
	}
	if v := llamaScorerGPULayers(args); v != "" {
		cmdArgs = append(cmdArgs, "--gpu-layers", v)
	}
	cmd := exec.Command(scorer, cmdArgs...)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	started := time.Now()
	if err := cmd.Run(); err != nil {
		return nil, shardStats{}, cliError{"llama_scorer_failed", fmt.Sprintf("local llama.cpp scorer failed: %v", err), []string{strings.TrimSpace(stderr.String()), "Build dist/lmx-llama-score-hellaswag and pass --llama-scorer if it is not next to lmx."}, nil}
	}
	elapsed := time.Since(started).Milliseconds()
	results := make([]shardItemResult, len(items))
	scanner := bufio.NewScanner(bytes.NewReader(stdout.Bytes()))
	idx := 0
	for scanner.Scan() {
		if idx >= len(items) {
			break
		}
		var row struct {
			QuestionID         string             `json:"question_id"`
			ItemIndex          int                `json:"itemIndex"`
			Predicted          string             `json:"predicted"`
			Gold               string             `json:"gold"`
			Pass               bool               `json:"pass"`
			Choices            []string           `json:"choices"`
			Scores             map[string]float64 `json:"scores"`
			ScoreNormalization string             `json:"scoreNormalization"`
			Error              string             `json:"error"`
		}
		if err := json.Unmarshal(scanner.Bytes(), &row); err != nil {
			return nil, shardStats{}, err
		}
		item := items[idx]
		scoreText := ""
		if len(row.Scores) > 0 {
			scoreText = fmt.Sprintf("A %.6f | B %.6f | C %.6f | D %.6f", row.Scores["A"], row.Scores["B"], row.Scores["C"], row.Scores["D"])
		}
		results[idx] = shardItemResult{
			questionID:         firstNonEmpty(row.QuestionID, firstNonEmpty(stringValue(item["question_id"]), stringValue(item["questionId"]), stringValue(item["id"]))),
			itemIndex:          idx,
			pass:               row.Pass,
			scored:             row.Error == "",
			predicted:          row.Predicted,
			gold:               row.Gold,
			response:           row.Predicted,
			reasoning:          "llama.cpp continuation loglikelihood scores: " + scoreText,
			prompt:             strings.TrimSpace(fmt.Sprint(item["input"])),
			promptHash:         sha256Hex(strings.TrimSpace(fmt.Sprint(item["input"]))),
			question:           renderEvalQuestion(item),
			choices:            append([]string(nil), row.Choices...),
			choiceScores:       row.Scores,
			scoreNormalization: row.ScoreNormalization,
			latencyMs:          0,
			errText:            row.Error,
		}
		idx++
	}
	if err := scanner.Err(); err != nil {
		return nil, shardStats{}, err
	}
	if idx != len(items) {
		return nil, shardStats{}, cliError{"llama_scorer_incomplete", fmt.Sprintf("local scorer returned %d rows for %d items", idx, len(items)), nil, nil}
	}
	stats := shardStats{}
	if len(results) > 0 {
		per := elapsed / int64(len(results))
		for i := range results {
			results[i].latencyMs = per
		}
	}
	for _, r := range results {
		if r.scored {
			stats.scored++
			if r.pass {
				stats.correct++
			}
		} else {
			stats.errors++
		}
		stats.totalLatencyMs += r.latencyMs
	}
	printStatus(args, "eval_shard_progress", map[string]any{"done": len(items), "total": len(items)})
	return results, stats, nil
}

func bundledExecutable(name string) string {
	exe, err := os.Executable()
	if err != nil {
		return ""
	}
	candidates := []string{
		filepath.Join(filepath.Dir(exe), name),
		filepath.Join(filepath.Dir(exe), "dist", name),
	}
	for _, candidate := range candidates {
		if st, err := os.Stat(candidate); err == nil && !st.IsDir() {
			return candidate
		}
	}
	return ""
}

var codeBlockRe = regexp.MustCompile("(?s)```(?:python|py)?[ \\t]*\\r?\\n(.*?)```")

type codeGeneration struct {
	prompt    string
	response  string
	code      string
	program   string
	latencyMs int64
	errText   string
}

// extractGeneratedCode pulls runnable Python out of a chat completion: prefer the
// first fenced code block, fall back to the raw text, and if the entry point is
// not defined treat the output as a body continuation of the prompt stub.
func extractGeneratedCode(response, prompt, entryPoint string) string {
	code := response
	if m := codeBlockRe.FindStringSubmatch(response); m != nil {
		code = m[1]
	}
	code = strings.TrimRight(code, " \t\r\n")
	if entryPoint != "" {
		defRe := regexp.MustCompile("(?m)^[ \\t]*def[ \\t]+" + regexp.QuoteMeta(entryPoint) + "[ \\t]*\\(")
		if !defRe.MatchString(code) {
			return strings.TrimRight(prompt, "\n") + "\n" + code
		}
	}
	return code
}

// promptImportPreamble returns import lines present in the prompt stub but absent
// from the model's code, so canonical HumanEval programs do not false-fail when a
// model omits an import (e.g. `from typing import List`) that the stub provided.
func promptImportPreamble(prompt, solution string) string {
	var lines []string
	for _, raw := range strings.Split(prompt, "\n") {
		line := strings.TrimSpace(raw)
		if strings.HasPrefix(line, "import ") || strings.HasPrefix(line, "from ") {
			if !strings.Contains(solution, line) {
				lines = append(lines, line)
			}
		}
	}
	if len(lines) == 0 {
		return ""
	}
	return strings.Join(lines, "\n") + "\n"
}

// buildCodeProgram assembles a self-contained Python program: solution + hidden
// tests + an invocation that raises on failure (HumanEval `check(entry)`, or the
// raw test block otherwise).
func buildCodeProgram(item map[string]any, solution string) string {
	test := stringValue(item["test"])
	entry := stringValue(item["entry_point"])
	var b strings.Builder
	if entry != "" {
		// HumanEval-style: keep the stub's imports, matching the canonical
		// `prompt + completion + test` program assembly.
		b.WriteString(promptImportPreamble(stringValue(item["input"]), solution))
	}
	b.WriteString(solution)
	b.WriteString("\n\n")
	b.WriteString(test)
	if entry != "" && strings.Contains(test, "def check(") {
		b.WriteString("\n\ncheck(")
		b.WriteString(entry)
		b.WriteString(")\n")
	}
	return b.String()
}

// sandboxCommand builds the process that runs the code sandbox. By default it
// launches the hardened Docker image over stdin/stdout; --sandbox-cmd overrides
// it entirely (e.g. podman, or `python3 sandbox/run_sandbox.py` without Docker).
func sandboxCommand(args cliArgs) (*exec.Cmd, string) {
	if custom := opt(args, "sandbox-cmd"); custom != "" {
		return exec.Command("sh", "-c", custom), custom
	}
	runtime := strings.Fields(firstNonEmpty(opt(args, "sandbox-runtime"), "docker"))
	if len(runtime) == 0 {
		runtime = []string{"docker"}
	}
	if hasFlag(args, "sandbox-use-sudo") && runtime[0] != "sudo" {
		runtime = append([]string{"sudo"}, runtime...)
	}
	image := firstNonEmpty(opt(args, "sandbox-image"), "lmx-sandbox")
	argv := append([]string{}, runtime...)
	argv = append(argv,
		"run", "--rm", "-i",
		"--network", "none",
		"--memory", firstNonEmpty(opt(args, "sandbox-memory"), "2g"),
		"--cpus", firstNonEmpty(opt(args, "sandbox-cpus"), "2"),
		"--pids-limit", "128",
	)
	if !hasFlag(args, "sandbox-relaxed-security") {
		argv = append(argv,
			"--cap-drop", "ALL",
			"--security-opt", "no-new-privileges",
			"--read-only",
		)
	}
	argv = append(argv,
		"--tmpfs", "/tmp:exec,size=128m",
		image,
	)
	return exec.Command(argv[0], argv[1:]...), strings.Join(argv, " ")
}

func sandboxFailureHints(stderr string) []string {
	clean := strings.TrimSpace(stderr)
	hints := make([]string, 0, 5)
	if clean != "" {
		hints = append(hints, clean)
	}
	lower := strings.ToLower(clean)
	if strings.Contains(lower, "/var/run/docker.sock") && strings.Contains(lower, "permission denied") {
		hints = append(hints,
			"Docker socket permission denied: add your user to the Docker group and re-login, or run the container via sudo with --sandbox-use-sudo.",
			"Equivalent override: --sandbox-cmd \"sudo docker run --rm -i --network none --memory 2g --cpus 2 --pids-limit 128 --tmpfs /tmp:exec,size=128m lmx-sandbox\"",
		)
	}
	if strings.Contains(lower, "operation not permitted") && strings.Contains(lower, "python3") {
		hints = append(hints, "If the container starts but cannot exec Python, retry with --sandbox-relaxed-security; some Docker/rootless/security-profile setups reject the stricter cap/no-new-privileges/read-only combination.")
	}
	hints = append(hints, "Build the image with `docker build -t lmx-sandbox sandbox`, or override with --sandbox-cmd.")
	return hints
}

//go:embed mbpp_fewshot.json
var mbppFewShotJSON []byte

func defaultFewShot(dataset string) int {
	if isMbppFamily(dataset) {
		return 3
	}
	return 0
}

// mbppFewShotPreamble renders the canonical MBPP n-shot prefix (task text + tests
// + reference solution) used by standard MBPP harnesses.
func mbppFewShotPreamble(n int) string {
	if n <= 0 {
		return ""
	}
	var examples []struct {
		Text  string   `json:"text"`
		Tests []string `json:"tests"`
		Code  string   `json:"code"`
	}
	if err := json.Unmarshal(mbppFewShotJSON, &examples); err != nil || len(examples) == 0 {
		return ""
	}
	if n > len(examples) {
		n = len(examples)
	}
	var b strings.Builder
	b.WriteString("Here are examples of Python tasks and correct solutions.\n\n")
	for i := 0; i < n; i++ {
		ex := examples[i]
		b.WriteString(ex.Text)
		b.WriteString("\nYour code should pass these tests:\n")
		b.WriteString(strings.Join(ex.Tests, "\n"))
		b.WriteString("\n```python\n")
		b.WriteString(strings.TrimRight(ex.Code, "\n"))
		b.WriteString("\n```\n\n")
	}
	return b.String()
}

// passAtK is the unbiased pass@k estimator from the Codex paper: given n samples
// of which c pass, the probability that k random samples contain a passing one.
func passAtK(n, c, k int) float64 {
	if k > n {
		k = n
	}
	if n-c < k {
		return 1.0
	}
	prod := 1.0
	for i := n - c + 1; i <= n; i++ {
		prod *= 1.0 - float64(k)/float64(i)
	}
	return 1.0 - prod
}

func isMbppFamily(dataset string) bool {
	return strings.HasPrefix(strings.ToLower(dataset), "mbpp")
}

// codePrompt builds the generation prompt for a code item: MBPP-family uses the
// canonical few-shot prefix; HumanEval-family uses an instruct completion prompt.
func codePrompt(cfg runShardConfig, item map[string]any) string {
	if cfg.promptTemplate != "" {
		return renderEvalPrompt(cfg.promptTemplate, item)
	}
	input := stringValue(item["input"])
	if isMbppFamily(cfg.dataset) && cfg.fewShot > 0 {
		return mbppFewShotPreamble(cfg.fewShot) + "Now complete this task. Reply with only the implementation in a single ```python code block and no prose.\n\n" + input
	}
	return "Complete the following Python function. Reply with the complete implementation in a single ```python code block and no prose.\n\n" + input
}

// runEvalShardCodeExec runs an execution-based code eval shard (HumanEval/MBPP and
// their EvalPlus variants): generate n solution samples per item against the model
// endpoint, grade each by running solution+tests in the sandbox, then record the
// submitted greedy/first-sample pass@1 plus a pass@k estimate over all samples.
func runEvalShardCodeExec(args cliArgs, baseURL, model string, items []map[string]any, cfg runShardConfig) ([]shardItemResult, shardStats, map[string]any, error) {
	n := cfg.nSamples
	if n < 1 {
		n = 1
	}
	k := cfg.passK
	if k < 1 {
		k = 1
	}
	if k > n {
		k = n
	}
	if cfg.concurrency < 1 {
		cfg.concurrency = 1
	}

	type cell struct {
		prompt            string
		code              string
		program           string
		errText           string
		latencyMs         int64
		thinkingRequested string
		thinkingObserved  string
	}
	grid := make([][]cell, len(items))
	for i := range grid {
		grid[i] = make([]cell, n)
	}

	type job struct{ i, s int }
	jobs := make(chan job)
	var wg sync.WaitGroup
	var mu sync.Mutex
	completed := 0
	total := len(items) * n
	worker := func() {
		defer wg.Done()
		for jb := range jobs {
			item := items[jb.i]
			prompt := codePrompt(cfg, item)
			entry := stringValue(item["entry_point"])
			start := time.Now()
			var content string
			var modelReasoning string
			var err error
			for attempt := 0; attempt < 3; attempt++ {
				content, modelReasoning, err = callOpenAIChatDetailed(baseURL, model, prompt, cfg.apiKey, cfg.maxTokens, cfg.temperature, cfg.topP, nil)
				if err == nil {
					break
				}
			}
			c := cell{prompt: prompt, latencyMs: time.Since(start).Milliseconds(), thinkingRequested: promptThinkingDirective(prompt), thinkingObserved: observedThinkingMode(promptThinkingDirective(prompt), modelReasoning)}
			if err != nil {
				c.errText = err.Error()
			} else {
				c.code = extractGeneratedCode(content, stringValue(item["input"]), entry)
				c.program = buildCodeProgram(item, c.code)
			}
			grid[jb.i][jb.s] = c
			mu.Lock()
			completed++
			done := completed
			mu.Unlock()
			if done%25 == 0 || done == total {
				printStatus(args, "eval_shard_progress", map[string]any{"done": done, "total": total, "phase": "generate"})
			}
		}
	}
	for w := 0; w < cfg.concurrency; w++ {
		wg.Add(1)
		go worker()
	}
	for i := range items {
		for s := 0; s < n; s++ {
			jobs <- job{i, s}
		}
	}
	close(jobs)
	wg.Wait()

	// Safety valve: if most first-sample generations failed the endpoint is likely
	// down, so abort rather than submit an all-fail run.
	genFailures := 0
	for i := range items {
		if grid[i][0].errText != "" {
			genFailures++
		}
	}
	if len(items) > 0 && genFailures > len(items)/2 {
		return nil, shardStats{}, nil, cliError{"generation_failures", fmt.Sprintf("%d/%d generations failed; aborting before grading", genFailures, len(items)), []string{"Check the model endpoint (--base-url) is reachable and healthy."}, nil}
	}

	qids := make([]string, len(items))
	for i, item := range items {
		qids[i] = firstNonEmpty(stringValue(item["question_id"]), stringValue(item["questionId"]), stringValue(item["id"]))
	}
	key := func(i, s int) string { return fmt.Sprintf("%s#%d", qids[i], s) }

	// Build the sandbox batch: one task per (item, sample) with a runnable program.
	var batch strings.Builder
	enc := json.NewEncoder(&batch)
	programs := 0
	for i := range items {
		for s := 0; s < n; s++ {
			c := grid[i][s]
			if c.errText != "" || c.program == "" {
				continue
			}
			if err := enc.Encode(map[string]any{"question_id": key(i, s), "program": c.program}); err != nil {
				return nil, shardStats{}, nil, err
			}
			programs++
		}
	}

	type sandboxResult struct {
		QuestionID string `json:"question_id"`
		Passed     bool   `json:"passed"`
		ReturnCode int    `json:"returncode"`
		TimedOut   bool   `json:"timed_out"`
		Stderr     string `json:"stderr"`
		DurationMs int64  `json:"duration_ms"`
	}
	byKey := map[string]sandboxResult{}
	if programs > 0 {
		cmd, snippet := sandboxCommand(args)
		printStatus(args, "eval_shard_sandbox", map[string]any{"command": snippet, "programs": programs, "nSamples": n})
		cmd.Stdin = strings.NewReader(batch.String())
		var stdout, stderr bytes.Buffer
		cmd.Stdout = &stdout
		cmd.Stderr = &stderr
		if err := cmd.Run(); err != nil {
			return nil, shardStats{}, nil, cliError{"sandbox_failed", fmt.Sprintf("code sandbox failed: %v", err), sandboxFailureHints(stderr.String()), nil}
		}
		scanner := bufio.NewScanner(bytes.NewReader(stdout.Bytes()))
		scanner.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)
		for scanner.Scan() {
			line := strings.TrimSpace(scanner.Text())
			if line == "" {
				continue
			}
			var sr sandboxResult
			if err := json.Unmarshal([]byte(line), &sr); err != nil {
				return nil, shardStats{}, nil, fmt.Errorf("parse sandbox result: %w", err)
			}
			byKey[sr.QuestionID] = sr
		}
		if err := scanner.Err(); err != nil {
			return nil, shardStats{}, nil, err
		}
	}

	results := make([]shardItemResult, len(items))
	stats := shardStats{}
	passAtKSum, passAt1Sum := 0.0, 0.0
	for i, item := range items {
		passedSamples := 0
		for s := 0; s < n; s++ {
			c := grid[i][s]
			if c.errText != "" || c.program == "" {
				continue
			}
			if sr, ok := byKey[key(i, s)]; ok && sr.Passed {
				passedSamples++
			}
		}
		first := grid[i][0]
		firstPass := false
		summary := ""
		switch {
		case first.errText != "":
			summary = "generation failed: " + first.errText
		case first.program == "":
			summary = "no runnable code extracted from model response"
		default:
			if sr, ok := byKey[key(i, 0)]; ok {
				firstPass = sr.Passed
				summary = fmt.Sprintf("sandbox: passed=%v returncode=%d timed_out=%v", sr.Passed, sr.ReturnCode, sr.TimedOut)
				if sr.Stderr != "" {
					summary += "\n" + sr.Stderr
				}
			} else {
				summary = "sandbox returned no result"
			}
		}
		pk := passAtK(n, passedSamples, k)
		passAtKSum += pk
		passAt1Sum += float64(passedSamples) / float64(n)
		latency := first.latencyMs
		if sr, ok := byKey[key(i, 0)]; ok {
			latency += sr.DurationMs
		}
		reasoning := summary
		if n > 1 {
			reasoning = fmt.Sprintf("pass@%d=%.3f (%d/%d samples passed)\n%s", k, pk, passedSamples, n, summary)
		}
		results[i] = shardItemResult{
			questionID:        qids[i],
			itemIndex:         i,
			question:          renderEvalQuestion(item),
			prompt:            first.prompt,
			promptHash:        sha256Hex(first.prompt),
			response:          first.code,
			reasoning:         reasoning,
			thinkingRequested: first.thinkingRequested,
			thinkingObserved:  first.thinkingObserved,
			predicted:         boolPassLabel(firstPass),
			gold:              "pass",
			scored:            true,
			pass:              firstPass,
			latencyMs:         latency,
		}
		stats.scored++
		if firstPass {
			stats.correct++
		}
		stats.totalLatencyMs += latency
	}
	metrics := map[string]any{"nSamples": n, "k": k, "temperature": cfg.temperature}

	if len(items) > 0 {
		metrics["passAtK"] = passAtKSum / float64(len(items))
		metrics["passAt1"] = passAt1Sum / float64(len(items))
	}
	if cfg.fewShot > 0 && isMbppFamily(cfg.dataset) {
		metrics["fewShot"] = cfg.fewShot
	}
	return results, stats, metrics, nil
}

// runEvalShardCruxExec scores CRUXEval with its canonical execution metric:
// input-prediction passes if f(generated_input) == observed_output, and
// output-prediction passes if generated_output == f(function_input).
func runEvalShardCruxExec(args cliArgs, baseURL, model string, items []map[string]any, cfg runShardConfig) ([]shardItemResult, shardStats, map[string]any, error) {
	if cfg.concurrency < 1 {
		cfg.concurrency = 1
	}

	type cell struct {
		prompt            string
		response          string
		candidate         string
		program           string
		errText           string
		latencyMs         int64
		thinkingRequested string
		thinkingObserved  string
	}

	cells := make([]cell, len(items))
	jobs := make(chan int)
	var wg sync.WaitGroup
	var mu sync.Mutex
	completed := 0
	total := len(items)
	worker := func() {
		defer wg.Done()
		for i := range jobs {
			item := items[i]
			prompt := cruxPrompt(cfg, item)
			start := time.Now()
			content, reasoning, err := callOpenAIChatDetailed(baseURL, model, prompt, cfg.apiKey, cfg.maxTokens, cfg.temperature, cfg.topP, nil)
			c := cell{
				prompt:            prompt,
				latencyMs:         time.Since(start).Milliseconds(),
				thinkingRequested: promptThinkingDirective(prompt),
				thinkingObserved:  observedThinkingMode(promptThinkingDirective(prompt), reasoning),
			}
			if err != nil {
				c.errText = err.Error()
			} else {
				response := content
				if response == "" {
					response = reasoning
				}
				c.response = response
				c.candidate = extractCRUXCandidate(response)
				if c.candidate == "" {
					c.errText = "no CRUXEval candidate answer extracted"
				} else {
					c.program = buildCRUXEvalProgram(item, c.candidate)
				}
			}
			cells[i] = c
			mu.Lock()
			completed++
			done := completed
			mu.Unlock()
			if done%25 == 0 || done == total {
				printStatus(args, "eval_shard_progress", map[string]any{"done": done, "total": total, "phase": "generate"})
			}
		}
	}
	for w := 0; w < cfg.concurrency; w++ {
		wg.Add(1)
		go worker()
	}
	for i := range items {
		jobs <- i
	}
	close(jobs)
	wg.Wait()

	genFailures := 0
	for i := range items {
		if cells[i].errText != "" {
			genFailures++
		}
	}
	if len(items) > 0 && genFailures > len(items)/2 {
		return nil, shardStats{}, nil, cliError{"generation_failures", fmt.Sprintf("%d/%d generations failed; aborting before grading", genFailures, len(items)), []string{"Check the model endpoint (--base-url) is reachable and healthy."}, nil}
	}

	qids := make([]string, len(items))
	var batch strings.Builder
	enc := json.NewEncoder(&batch)
	programs := 0
	for i, item := range items {
		qids[i] = firstNonEmpty(stringValue(item["question_id"]), stringValue(item["questionId"]), stringValue(item["id"]))
		if cells[i].errText != "" || cells[i].program == "" {
			continue
		}
		if err := enc.Encode(map[string]any{"question_id": qids[i], "program": cells[i].program}); err != nil {
			return nil, shardStats{}, nil, err
		}
		programs++
	}

	type sandboxResult struct {
		QuestionID string `json:"question_id"`
		Passed     bool   `json:"passed"`
		ReturnCode int    `json:"returncode"`
		TimedOut   bool   `json:"timed_out"`
		Stderr     string `json:"stderr"`
		DurationMs int64  `json:"duration_ms"`
	}
	byQID := map[string]sandboxResult{}
	if programs > 0 {
		cmd, snippet := sandboxCommand(args)
		printStatus(args, "eval_shard_sandbox", map[string]any{"command": snippet, "programs": programs, "scoring": "cruxeval_execution"})
		cmd.Stdin = strings.NewReader(batch.String())
		var stdout, stderr bytes.Buffer
		cmd.Stdout = &stdout
		cmd.Stderr = &stderr
		if err := cmd.Run(); err != nil {
			return nil, shardStats{}, nil, cliError{"sandbox_failed", fmt.Sprintf("CRUXEval sandbox failed: %v", err), sandboxFailureHints(stderr.String()), nil}
		}
		scanner := bufio.NewScanner(bytes.NewReader(stdout.Bytes()))
		scanner.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)
		for scanner.Scan() {
			line := strings.TrimSpace(scanner.Text())
			if line == "" {
				continue
			}
			var sr sandboxResult
			if err := json.Unmarshal([]byte(line), &sr); err != nil {
				return nil, shardStats{}, nil, fmt.Errorf("parse sandbox result: %w", err)
			}
			byQID[sr.QuestionID] = sr
		}
		if err := scanner.Err(); err != nil {
			return nil, shardStats{}, nil, err
		}
	}

	results := make([]shardItemResult, len(items))
	stats := shardStats{}
	for i, item := range items {
		c := cells[i]
		pass := false
		summary := ""
		switch {
		case c.errText != "":
			summary = "generation failed: " + c.errText
		case c.program == "":
			summary = "no runnable CRUXEval checker generated"
		default:
			if sr, ok := byQID[qids[i]]; ok {
				pass = sr.Passed
				summary = fmt.Sprintf("sandbox: passed=%v returncode=%d timed_out=%v", sr.Passed, sr.ReturnCode, sr.TimedOut)
				if sr.Stderr != "" {
					summary += "\n" + sr.Stderr
				}
				c.latencyMs += sr.DurationMs
			} else {
				summary = "sandbox returned no result"
			}
		}
		results[i] = shardItemResult{
			questionID:        qids[i],
			itemIndex:         i,
			question:          renderEvalQuestion(item),
			prompt:            c.prompt,
			promptHash:        sha256Hex(c.prompt),
			response:          c.response,
			reasoning:         summary,
			thinkingRequested: c.thinkingRequested,
			thinkingObserved:  c.thinkingObserved,
			predicted:         c.candidate,
			gold:              cruxExpectedLabel(item),
			scored:            true,
			pass:              pass,
			latencyMs:         c.latencyMs,
		}
		stats.scored++
		if pass {
			stats.correct++
		}
		stats.totalLatencyMs += c.latencyMs
	}
	return results, stats, map[string]any{"executionMetric": "cruxeval", "answerExtraction": "cruxeval_candidate"}, nil
}

func cruxPrompt(cfg runShardConfig, item map[string]any) string {
	if cfg.promptTemplate != "" {
		return renderEvalPrompt(cfg.promptTemplate, item)
	}
	return renderEvalPrompt("/no_think\n\nReturn only the exact Python expression/value requested. Do not explain, do not include Markdown, and do not call f unless the requested answer itself is a call.\n\n{{input}}", item)
}

var cruxFinalAnswerPattern = regexp.MustCompile(`(?is)(?:final\s+answer|answer)\s*[:=]\s*`)

func extractCRUXCandidate(response string) string {
	text := strings.TrimSpace(response)
	if text == "" {
		return ""
	}
	if locs := cruxFinalAnswerPattern.FindAllStringIndex(text, -1); len(locs) > 0 {
		text = strings.TrimSpace(text[locs[len(locs)-1][1]:])
	}
	if fenced := lastMarkdownCodeFence(text); fenced != "" {
		text = fenced
	}
	lines := strings.Split(text, "\n")
	for _, line := range lines {
		line = cleanCRUXCandidateLine(line)
		if line != "" {
			return line
		}
	}
	return cleanCRUXCandidateLine(text)
}

func lastMarkdownCodeFence(text string) string {
	re := regexp.MustCompile("(?s)```(?:python|py)?\\s*(.*?)```")
	matches := re.FindAllStringSubmatch(text, -1)
	if len(matches) == 0 {
		return ""
	}
	return strings.TrimSpace(matches[len(matches)-1][1])
}

func cleanCRUXCandidateLine(line string) string {
	line = strings.TrimSpace(line)
	line = strings.TrimPrefix(line, "- ")
	line = strings.TrimPrefix(line, "* ")
	line = strings.TrimSpace(line)
	line = strings.Trim(line, "`")
	line = strings.TrimSpace(line)
	if strings.HasPrefix(strings.ToLower(line), "final answer") {
		if idx := strings.Index(line, ":"); idx >= 0 {
			line = strings.TrimSpace(line[idx+1:])
		}
	}
	return strings.TrimSpace(line)
}

func cruxExpectedLabel(item map[string]any) string {
	if observed := stringValue(item["observed_output"]); observed != "" {
		return observed
	}
	if gold := stringValue(item["gold"]); gold != "" {
		return gold
	}
	return stringValue(item["function_input"])
}

func buildCRUXEvalProgram(item map[string]any, candidate string) string {
	taskType := strings.ToLower(firstNonEmpty(stringValue(item["task_type"]), stringValue(item["taskType"])))
	if taskType == "" {
		qid := strings.ToLower(firstNonEmpty(stringValue(item["question_id"]), stringValue(item["questionId"]), stringValue(item["id"])))
		if strings.Contains(qid, "cruxeval-i:") {
			taskType = "input_prediction"
		} else {
			taskType = "output_prediction"
		}
	}
	code := stringValue(item["code"])
	if code == "" {
		code = extractPythonFunction(renderEvalQuestion(item))
	}
	observedOutput := firstNonEmpty(stringValue(item["observed_output"]), stringValue(item["gold"]))
	functionInput := stringValue(item["function_input"])
	return fmt.Sprintf(`import ast
import inspect

%s

TASK_TYPE = %s
CANDIDATE_SRC = %s
OBSERVED_OUTPUT_SRC = %s
FUNCTION_INPUT_SRC = %s

def _literal(src):
    return ast.literal_eval(src.strip())

def _strip_call(src):
    src = src.strip()
    if src.startswith("f(") and src.endswith(")"):
        return src[2:-1].strip()
    return src

def _required_positional_count(fn):
    count = 0
    variadic = False
    for p in inspect.signature(fn).parameters.values():
        if p.kind == p.VAR_POSITIONAL:
            variadic = True
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is p.empty:
            count += 1
    return count, variadic

def _parse_args(src):
    src = _strip_call(src)
    count, variadic = _required_positional_count(f)
    if not variadic and count == 1:
        return (_literal(src),)
    tuple_src = "(" + src
    if "," not in src:
        tuple_src += ","
    tuple_src += ")"
    value = _literal(tuple_src)
    if not isinstance(value, tuple):
        value = (value,)
    return value

if TASK_TYPE == "input_prediction":
    expected = _literal(OBSERVED_OUTPUT_SRC)
    args = _parse_args(CANDIDATE_SRC)
    actual = f(*args)
    assert actual == expected, f"expected f(*candidate) == {expected!r}, got {actual!r}; candidate={CANDIDATE_SRC!r}"
else:
    args = _parse_args(FUNCTION_INPUT_SRC)
    expected = f(*args)
    candidate = _literal(CANDIDATE_SRC)
    assert candidate == expected, f"expected output {expected!r}, got {candidate!r}; candidate={CANDIDATE_SRC!r}"
`, code, goPythonString(taskType), goPythonString(candidate), goPythonString(observedOutput), goPythonString(functionInput))
}

func goPythonString(value string) string {
	encoded, _ := json.Marshal(value)
	return string(encoded)
}

func extractPythonFunction(text string) string {
	start := strings.Index(text, "def f(")
	if start < 0 {
		return ""
	}
	tail := text[start:]
	end := strings.Index(tail, "\n\n")
	if end < 0 {
		return tail
	}
	return tail[:end]
}

func boolPassLabel(pass bool) string {
	if pass {
		return "pass"
	}
	return "fail"
}

func promptThinkingDirective(prompt string) string {
	trimmed := strings.TrimSpace(prompt)
	if strings.HasPrefix(trimmed, "/no_think") || strings.Contains(prompt, "\n/no_think") {
		return "disabled"
	}
	if strings.HasPrefix(trimmed, "/think") || strings.Contains(prompt, "\n/think") {
		return "enabled"
	}
	return "unspecified"
}

func observedThinkingMode(requested, reasoning string) string {
	if strings.TrimSpace(reasoning) != "" {
		return "enabled"
	}
	if requested == "disabled" {
		return "disabled"
	}
	return "unknown"
}

// scoreShardItem renders/scores one eval-shard row. HellaSwag-style multiple
// choice can use loglikelihood continuation ranking; otherwise choices are
// scored by generated letter matching and numeric QA by extracted final answer.
func scoreShardItem(itemIndex int, item map[string]any, cfg runShardConfig, baseURL, model string) shardItemResult {
	qid := firstNonEmpty(stringValue(item["question_id"]), stringValue(item["questionId"]), stringValue(item["id"]))
	res := shardItemResult{questionID: qid, itemIndex: itemIndex, question: renderEvalQuestion(item)}
	if qid == "" {
		res.errText = "row is missing question_id"
		return res
	}
	if item["gold"] == nil {
		res.errText = "row is missing gold answer for local scoring"
		return res
	}
	gold := strings.TrimSpace(fmt.Sprint(item["gold"]))
	choices := stringChoices(item["choices"])
	template := cfg.promptTemplate
	if template == "" {
		if len(choices) > 0 {
			template = "Answer the question. Reply with only the letter of the correct choice.\n\n{{input}}\n\n{{choices}}"
		} else {
			template = "Solve the problem. Show your reasoning step by step, then end your reply with a line in the form 'Final answer: <number>'.\n\n{{input}}"
		}
	}
	prompt := renderEvalPrompt(template, item)
	res.prompt = prompt
	res.promptHash = sha256Hex(prompt)
	res.thinkingRequested = promptThinkingDirective(prompt)
	if cfg.scoring == "loglikelihood" {
		contextText := strings.TrimSpace(fmt.Sprint(item["input"]))
		if cfg.promptTemplate != "" {
			contextText = renderEvalPrompt(cfg.promptTemplate, item)
		}
		res.prompt = contextText
		res.promptHash = sha256Hex(contextText)
		res.thinkingRequested = promptThinkingDirective(contextText)
		res.thinkingObserved = "not_applicable"
		started := time.Now()
		score, predicted, goldLabel, err := scoreLoglikelihoodItem(baseURL, model, cfg.apiKey, map[string]any{
			"runConfig": map[string]any{"loglikelihoodTarget": "choice_text", "loglikelihoodNorm": "byte"},
		}, item, contextText)
		res.latencyMs = time.Since(started).Milliseconds()
		if err != nil {
			res.errText = err.Error()
			return res
		}
		res.scored = true
		res.predicted = predicted
		res.gold = goldLabel
		res.pass = score >= 1
		res.response = predicted
		return res
	}
	started := time.Now()
	content, reasoning, err := callOpenAIChatDetailed(baseURL, model, prompt, cfg.apiKey, cfg.maxTokens, cfg.temperature, cfg.topP, nil)
	res.latencyMs = time.Since(started).Milliseconds()
	if err != nil {
		res.errText = err.Error()
		return res
	}
	// Score and display the answer (message.content); keep the reasoning trace
	// separately. If the server only returned reasoning, treat it as the answer.
	response := content
	if response == "" {
		response = reasoning
		reasoning = ""
	}
	res.response = response
	res.reasoning = reasoning
	res.thinkingObserved = observedThinkingMode(res.thinkingRequested, reasoning)
	res.scored = true
	if len(choices) > 0 {
		predicted := normalizeChoice(response, choices)
		goldLabel := normalizeChoice(gold, choices)
		res.predicted = predicted
		res.gold = goldLabel
		res.pass = predicted != "" && predicted == goldLabel
		return res
	}
	extraction := cfg.extraction
	if extraction == "" {
		extraction = "final_answer"
	}
	candidate := response
	if extraction != "none" {
		candidate = extractAnswer(response, extraction, cfg.answerRegex)
	}
	res.predicted = candidate
	res.gold = gold
	res.pass = answersMatch(candidate, gold)
	return res
}

func boolScore(ok bool) float64 {
	if ok {
		return 1
	}
	return 0
}

func floatOption(args cliArgs, key string, fallback float64) float64 {
	if v := opt(args, key); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			return f
		}
	}
	return fallback
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
		if strings.Contains(err.Error(), "401") || strings.Contains(err.Error(), "403") {
			return nil, cliError{"run_bundle_auth_failed", "Could not download the authenticated eval run bundle for " + suiteSlug + ".", []string{"Bucket-backed eval datasets include gold labels and require a valid --api-key.", "Check that the API key is valid for this LocalMaxxing instance.", "If you only want public metadata, use eval suite show instead of eval run/pull."}, err.Error()}
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
	// The run-bundle exposes per-task bucket/url download access under bundle["tasks"].
	bundleTaskURL := map[string]string{}
	for _, item := range anySlice(bundle["tasks"]) {
		bt := asObject(item)
		if bt == nil {
			continue
		}
		if ds := asObject(bt["dataset"]); ds != nil {
			if u := firstDatasetDownloadURL(ds); u != "" {
				bundleTaskURL[stringValue(bt["key"])] = u
			}
		}
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
		urlText := bundleTaskURL[stringValue(task["key"])]
		for _, candidate := range candidates {
			if urlText != "" {
				break
			}
			urlText = downloadURLForTask(candidate, task, i, len(tasks))
		}
		if urlText == "" {
			continue
		}
		if dataset == nil {
			dataset = map[string]any{}
		}
		dataset["downloadUrl"] = urlText
		if stringValue(dataset["url"]) == "" {
			dataset["url"] = urlText
		}
		if stringValue(dataset["source"]) == "" {
			dataset["source"] = "url"
		}
		// Preserve the bucket object's declared format so JSONL parses deterministically.
		if sr := asObject(dataset["storageRef"]); sr != nil {
			if f := stringValue(sr["format"]); f != "" && stringValue(dataset["format"]) == "" {
				dataset["format"] = f
			}
		}
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
	case "loglikelihood":
		return []string{"acc_norm,none", "acc,none", "acc_norm", "acc"}
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
			if scoring == "loglikelihood" {
				score, predicted, gold, err := scoreLoglikelihoodItem(baseURL, model, opt(args, "model-api-key"), doc, item, prompt)
				if err == nil {
					artifact["response"] = fmt.Sprintf("argmax=%s gold=%s", predicted, gold)
					artifact["score"] = score
					totalScore += score
				} else {
					failures++
					artifact["error"] = err.Error()
				}
				counted++
				artifact["latencyMs"] = time.Since(started).Milliseconds()
				artifacts = append(artifacts, artifact)
				continue
			}
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
	source := stringValue(dataset["source"])
	if source == "inline" {
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
	if source == "bucket" {
		return nil, cliError{"bucket_dataset_requires_run_bundle", "Bucket-backed eval dataset is missing a signed download URL.", []string{"Run with --api-key so the CLI can fetch the authenticated run-bundle.", "Use lmx eval pull <suiteSlug> --api-key ... to cache the dataset before running offline.", "Gold-label datasets are not available from the public suite metadata."}, nil}
	}
	if source == "huggingface" {
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
	return nil, fmt.Errorf("unknown dataset source %q", source)
}

func normalizeHardwarePayload(value any) any {
	hw := asObject(value)
	if hw == nil {
		return value
	}
	if gpuName := firstNonEmpty(stringValue(hw["gpuName"]), stringValue(hw["gpuModel"]), stringValue(hw["gpu"])); gpuName != "" {
		hw["gpuName"] = gpuName
	}
	if slots := anySlice(hw["gpus"]); len(slots) > 0 {
		for _, slotValue := range slots {
			slot := asObject(slotValue)
			if slot == nil {
				continue
			}
			if gpuName := firstNonEmpty(stringValue(slot["gpuName"]), stringValue(slot["name"]), stringValue(slot["gpuModel"]), stringValue(slot["gpu"])); gpuName != "" {
				slot["gpuName"] = gpuName
			}
			delete(slot, "name")
			delete(slot, "gpuModel")
			delete(slot, "gpu")
		}
	}
	return hw
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
	content, reasoning, err := callOpenAIChatDetailed(baseURL, model, prompt, apiKey, maxTokens, temperature, topP, stop)
	if err != nil {
		return "", err
	}
	return firstNonEmpty(content, reasoning), nil
}

// callOpenAIChatDetailed returns the answer (message.content) and any separate
// reasoning trace (message.reasoning_content) a thinking-model server provides,
// so callers can store/score the answer and the reasoning independently.
func callOpenAIChatDetailed(baseURL, model, prompt, apiKey string, maxTokens int, temperature, topP float64, stop []string) (content, reasoning string, err error) {
	body := map[string]any{"model": model, "messages": []any{map[string]any{"role": "user", "content": prompt}}, "temperature": temperature, "top_p": topP}
	// max_tokens <= 0 means "no cap": omit it so the model can finish its reasoning.
	if maxTokens > 0 {
		body["max_tokens"] = maxTokens
	}
	if len(stop) > 0 {
		body["stop"] = stop
	}
	data, _ := json.Marshal(body)
	ctx, cancel := context.WithTimeout(context.Background(), defaultEndpointTimeout)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, "POST", openAIBaseURL(baseURL)+"/v1/chat/completions", bytes.NewReader(data))
	if err != nil {
		return "", "", err
	}
	req.Header.Set("Content-Type", "application/json")
	if apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", "", err
	}
	defer res.Body.Close()
	if res.StatusCode < 200 || res.StatusCode >= 300 {
		text, _ := io.ReadAll(res.Body)
		return "", "", fmt.Errorf("OpenAI-compatible server returned %s: %s", res.Status, strings.TrimSpace(string(text)))
	}
	var response map[string]any
	if err := json.NewDecoder(res.Body).Decode(&response); err != nil {
		return "", "", err
	}
	if choices, _ := response["choices"].([]any); len(choices) > 0 {
		if message := asObject(asObject(choices[0])["message"]); message != nil {
			content = strings.TrimSpace(stringValue(message["content"]))
			reasoning = strings.TrimSpace(stringValue(message["reasoning_content"]))
		}
	}
	return content, reasoning, nil
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
		candidate := response
		if mode := stringValue(task["answerExtraction"]); mode != "" && mode != "none" {
			candidate = extractAnswer(response, mode, stringValue(task["answerRegex"]))
			artifact["extractedAnswer"] = candidate
		}
		if answersMatch(candidate, fmt.Sprint(gold)) {
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

// scoreLoglikelihoodItem ranks each multiple-choice answer by the model's
// forced-continuation log-probability (lm-eval style) instead of parsing
// generated text. Returns the 0/1 score plus the predicted and gold labels.
func scoreLoglikelihoodItem(baseURL, model, apiKey string, doc, item map[string]any, contextText string) (float64, string, string, error) {
	choices := stringChoices(item["choices"])
	if len(choices) == 0 {
		return 0, "", "", errors.New("loglikelihood items require choices")
	}
	if item["gold"] == nil {
		return 0, "", "", errors.New("item is missing gold answer for loglikelihood scoring")
	}
	goldLabel := normalizeChoice(fmt.Sprint(item["gold"]), choices)
	runConfig := asObject(doc["runConfig"])
	mode := firstNonEmpty(stringValue(runConfig["loglikelihoodTarget"]), "choice_text")
	norm := firstNonEmpty(stringValue(runConfig["loglikelihoodNorm"]), "byte")
	continuations := make([]string, len(choices))
	for i, choice := range choices {
		if mode == "letter" {
			continuations[i] = " " + choiceLabel(i)
		} else {
			// Leading space keeps the context/continuation split on a natural token boundary.
			continuations[i] = " " + strings.TrimSpace(choice)
		}
	}
	sums, byteLens, err := scoreContinuationsLogprob(baseURL, model, apiKey, contextText, continuations)
	if err != nil {
		return 0, "", "", err
	}
	bestIdx := 0
	bestScore := math.Inf(-1)
	for i := range sums {
		s := sums[i]
		if norm == "byte" {
			denom := float64(byteLens[i])
			if denom < 1 {
				denom = 1
			}
			s = s / denom
		}
		if s > bestScore {
			bestScore = s
			bestIdx = i
		}
	}
	predicted := choiceLabel(bestIdx)
	if predicted == goldLabel {
		return 1, predicted, goldLabel, nil
	}
	return 0, predicted, goldLabel, nil
}

// scoreContinuationsLogprob scores every continuation for one context in a single
// /v1/completions request using echo+logprobs (no text generated). Returns the
// summed continuation logprob and continuation byte length per choice.
func scoreContinuationsLogprob(baseURL, model, apiKey, contextText string, continuations []string) ([]float64, []int, error) {
	prompts := make([]string, len(continuations))
	for i, c := range continuations {
		prompts[i] = contextText + c
	}
	body := map[string]any{"model": model, "prompt": prompts, "max_tokens": 0, "echo": true, "logprobs": 1, "temperature": 0}
	data, _ := json.Marshal(body)
	ctx, cancel := context.WithTimeout(context.Background(), defaultEndpointTimeout)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, "POST", openAIBaseURL(baseURL)+"/v1/completions", bytes.NewReader(data))
	if err != nil {
		return nil, nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, nil, err
	}
	defer res.Body.Close()
	if res.StatusCode < 200 || res.StatusCode >= 300 {
		text, _ := io.ReadAll(res.Body)
		return nil, nil, cliError{"model_server_error", fmt.Sprintf("completions logprob request returned %s: %s", res.Status, strings.TrimSpace(string(text))), []string{
			"loglikelihood scoring needs POST /v1/completions with echo:true + prompt token logprobs (vLLM/SGLang-style OpenAI compatibility).",
			"Chat-only servers, and llama.cpp OpenAI-compatible completions that only return generated-token probabilities, cannot run canonical loglikelihood scoring; use a logprob-capable server or override with --scoring exact_match for debugging.",
		}, nil}
	}
	var response struct {
		Choices []struct {
			Index    int `json:"index"`
			Logprobs struct {
				TokenLogprobs []*float64 `json:"token_logprobs"`
				TextOffset    []int      `json:"text_offset"`
			} `json:"logprobs"`
		} `json:"choices"`
	}
	if err := json.NewDecoder(res.Body).Decode(&response); err != nil {
		return nil, nil, err
	}
	if len(response.Choices) != len(prompts) {
		return nil, nil, cliError{"logprobs_unavailable", fmt.Sprintf("expected %d logprob results, got %d", len(prompts), len(response.Choices)), []string{"Confirm the endpoint echoes logprobs for batched prompts."}, nil}
	}
	sums := make([]float64, len(prompts))
	byteLens := make([]int, len(prompts))
	contextLen := len(contextText)
	for _, choice := range response.Choices {
		idx := choice.Index
		if idx < 0 || idx >= len(prompts) {
			return nil, nil, cliError{"logprobs_unavailable", "completion choice index out of range", nil, nil}
		}
		offsets := choice.Logprobs.TextOffset
		logprobs := choice.Logprobs.TokenLogprobs
		if len(offsets) == 0 || len(offsets) != len(logprobs) {
			return nil, nil, cliError{"logprobs_unavailable", "server did not return aligned echo prompt logprobs", []string{"Confirm the endpoint returns token_logprobs and text_offset for echoed prompt tokens on /v1/completions.", "llama.cpp OpenAI compatibility may expose only generated-token probabilities; use vLLM/SGLang or another echo-logprobs endpoint for canonical HellaSwag."}, nil}
		}
		sum := 0.0
		for t := range offsets {
			if offsets[t] < contextLen {
				continue
			}
			if logprobs[t] == nil {
				continue
			}
			sum += *logprobs[t]
		}
		sums[idx] = sum
		byteLens[idx] = len(continuations[idx])
	}
	return sums, byteLens, nil
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

var numberAnswerPattern = regexp.MustCompile(`-?\$?\d[\d,]*(?:\.\d+)?%?`)
var finalAnswerMarkerPattern = regexp.MustCompile(`(?im)(?:####|final\s+answer|(?:the\s+)?answer\s+is|^\s*answer)\s*[:=]?\s*`)
var equationResultPattern = regexp.MustCompile(`=\s*(-?\$?\d[\d,]*(?:\.\d+)?%?)`)
var boldSegmentPattern = regexp.MustCompile(`\*\*([^*]*\d[^*]*)\*\*`)
var summaryAnswerLinePattern = regexp.MustCompile(`(?i)\b(?:total|result|therefore|so)\b`)

// extractFinalAnswer pulls the most likely final answer out of chain-of-thought
// math output, in priority order:
//  1. number after an explicit marker ("####", "Final answer:", "the answer is", ...)
//  2. last number on a summary line ("Total ... = 107", "therefore 107")
//  3. number after the last "= " (the result of the final computation)
//  4. the last bolded number that is not a heading (headings contain ":")
//  5. the last number anywhere (fallback)
//
// Avoid treating generic prose like "the question asks for the answer" as an
// answer marker; reasoning models often mention "answer" before correcting
// themselves, which can otherwise extract an early conversion factor.
func lastNumberFromSummaryLine(response string) string {
	lines := strings.Split(response, "\n")
	for i := len(lines) - 1; i >= 0; i-- {
		line := strings.TrimSpace(lines[i])
		if !summaryAnswerLinePattern.MatchString(line) {
			continue
		}
		if bolds := boldSegmentPattern.FindAllStringSubmatch(line, -1); len(bolds) > 0 {
			for j := len(bolds) - 1; j >= 0; j-- {
				if strings.Contains(bolds[j][1], ":") {
					continue
				}
				if num := numberAnswerPattern.FindString(bolds[j][1]); num != "" {
					return num
				}
			}
		}
		matches := numberAnswerPattern.FindAllString(line, -1)
		if len(matches) > 0 {
			return matches[len(matches)-1]
		}
	}
	return ""
}
func extractFinalAnswer(response string) string {
	if locs := finalAnswerMarkerPattern.FindAllStringIndex(response, -1); len(locs) > 0 {
		tail := response[locs[len(locs)-1][1]:]
		if num := numberAnswerPattern.FindString(tail); num != "" {
			return num
		}
	}
	if num := lastNumberFromSummaryLine(response); num != "" {
		return num
	}
	if eqs := equationResultPattern.FindAllStringSubmatch(response, -1); len(eqs) > 0 {
		return eqs[len(eqs)-1][1]
	}
	if bolds := boldSegmentPattern.FindAllStringSubmatch(response, -1); len(bolds) > 0 {
		for i := len(bolds) - 1; i >= 0; i-- {
			if strings.Contains(bolds[i][1], ":") {
				continue
			}
			if num := numberAnswerPattern.FindString(bolds[i][1]); num != "" {
				return num
			}
		}
	}
	matches := numberAnswerPattern.FindAllString(response, -1)
	if len(matches) == 0 {
		return response
	}
	return matches[len(matches)-1]
}

// extractAnswer pulls a final answer out of a (possibly chain-of-thought) response.
//   - "final_answer": number after an answer marker / last bolded number, else last number.
//   - "last_number": the last number-like token (handles $, commas, decimals, %, sign).
//   - "regex": the last match of answerRegex (capture group 1 if present, else whole match).
//
// On no match or a bad pattern it falls back to the raw response so scoring still runs.
func extractAnswer(response, mode, pattern string) string {
	switch mode {
	case "final_answer":
		return extractFinalAnswer(response)
	case "last_number":
		matches := numberAnswerPattern.FindAllString(response, -1)
		if len(matches) == 0 {
			return response
		}
		return matches[len(matches)-1]
	case "regex":
		if pattern == "" {
			return response
		}
		re, err := regexp.Compile(pattern)
		if err != nil {
			return response
		}
		matches := re.FindAllStringSubmatch(response, -1)
		if len(matches) == 0 {
			return response
		}
		last := matches[len(matches)-1]
		if len(last) > 1 {
			return last[1]
		}
		return last[0]
	default:
		return response
	}
}

// normalizeNumericAnswer strips currency/grouping/percent formatting so "$1,234.0"
// and "1234" compare equal.
func normalizeNumericAnswer(value string) string {
	value = strings.TrimSpace(value)
	value = strings.NewReplacer("$", "", ",", "", "%", "", " ", "").Replace(value)
	return strings.TrimSuffix(value, ".")
}

// answersMatch compares an extracted answer to gold: exact normalized-text equality,
// formatting-insensitive equality, or numeric equality (so 18 == 18.0).
func answersMatch(pred, gold string) bool {
	if normalizeEvalText(pred) == normalizeEvalText(gold) {
		return true
	}
	pn := normalizeNumericAnswer(pred)
	gn := normalizeNumericAnswer(gold)
	if pn != "" && pn == gn {
		return true
	}
	pf, perr := strconv.ParseFloat(pn, 64)
	gf, gerr := strconv.ParseFloat(gn, 64)
	if perr == nil && gerr == nil {
		return math.Abs(pf-gf) < 1e-9
	}
	return false
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
	printFullResponse := label != "run" || hasFlag(args, "json") || hasFlag(args, "print") || hasFlag(args, "verbose")
	if printFullResponse {
		printJSON(value)
	}
	status := label + "_submitted"
	if dryRun {
		status = label + "_dry_run_valid"
	}
	fields := map[string]any{"endpoint": endpoint, "status": map[bool]string{true: "valid", false: "submitted"}[dryRun]}
	if label == "run" {
		addEvalRunReceiptFields(fields, value)
	}
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

func addEvalRunReceiptFields(fields map[string]any, value any) {
	obj := asObject(value)
	if obj == nil {
		return
	}
	parsed := asObject(obj["parsed"])
	source := obj
	if parsed != nil {
		source = parsed
	}
	for _, key := range []string{"id", "scoreAggregate", "status"} {
		if source[key] != nil {
			fields[key] = source[key]
		}
	}
	if suite := asObject(source["suite"]); suite != nil {
		fields["suite"] = firstNonEmpty(stringValue(suite["slug"]), stringValue(suite["name"]))
	} else if slug := stringValue(source["suiteSlug"]); slug != "" {
		fields["suite"] = slug
	}
	if model := asObject(source["model"]); model != nil {
		fields["model"] = firstNonEmpty(stringValue(model["hfId"]), stringValue(model["displayName"]))
	} else if hfID := stringValue(source["hfId"]); hfID != "" {
		fields["model"] = hfID
	}
	count := asObject(source["_count"])
	if count == nil {
		count = asObject(obj["_count"])
	}
	if count != nil && count["artifacts"] != nil {
		fields["artifacts"] = count["artifacts"]
		return
	}
	if artifacts := anySlice(source["artifacts"]); len(artifacts) > 0 {
		fields["artifacts"] = len(artifacts)
	}
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
	if attach := stringValue(feedback["attachHardwareCommand"]); attach != "" {
		fmt.Println("  " + attach)
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

var usageExamples = []string{
	`lmx context --out localmaxxing-agent-context.json`,
	`lmx auth --key bhk_...`,
	`lmx auth login`,
	`lmx auth logout`,
	`lmx auth keys list`,
	`lmx auth keys create --name "my key"`,
	`lmx auth keys revoke <id>`,
	`lmx auth whoami`,
	`lmx profile save my-4090 --mode local --hf-id Qwen/Qwen3-8B --quantization Q4_K_M --hardware hardware.json`,
	`lmx hardware --out hardware.json`,
	`lmx hardware init --out hardware.json`,
	`lmx hardware validate hardware.json`,
	`lmx setups list`,
	`lmx setups pull --default --out hardware.json`,
	`lmx setups pull --name "2x RTX 3090" --out hardware.json`,
	`lmx engines`,
	`lmx skill print`,
	`lmx skill install --dir .claude/skills`,
	`lmx endpoint discover --hf-id Qwen/Qwen3-8B --quantization fp16`,
	`lmx server dry-run vllm --hf-id Qwen/Qwen3-8B --quantization fp16`,
	`lmx server dry-run llama.cpp --model-path model.gguf`,
	`lmx benchmark run vllm --mode remote --base-url http://server:8000 --hf-id Qwen/Qwen3-8B --quantization fp16 --dry-run`,
	`lmx benchmark run vllm --mode local --hf-id Qwen/Qwen3-8B --quantization fp16 --bench-kind throughput --benchmark-output vllm.json --dry-run`,
	`lmx benchmark run llama.cpp --mode local --hf-id Qwen/Qwen3-8B --quantization Q4_K_M --command "llama-bench -m model.gguf" --dry-run`,
	`lmx benchmark run llama.cpp --mode local --hf-id Qwen/Qwen3-8B --quantization Q4_K_M --model-path model.gguf --dry-run`,
	`lmx benchmark runs list`,
	`lmx benchmark runs show runs/Qwen-Qwen3-8B/run.json`,
	`lmx benchmark runs edit runs/Qwen-Qwen3-8B/run.json --set-json '{"tokSOut":120}'`,
	`lmx benchmark runs rerun runs/Qwen-Qwen3-8B/run.json --dry-run`,
	`lmx benchmark runs submit runs/Qwen-Qwen3-8B/run.json --api-key bhk_...`,
	`lmx benchmark runs delete runs/Qwen-Qwen3-8B/run.json --yes`,
	`lmx benchmark runs stats --group-by quantization --metric tokSOut`,
	`lmx benchmark runs compare --by hardware --model Qwen/Qwen3-8B`,
	`lmx benchmark runs compare runs/base.json runs/candidate.json --metrics tokSOut,ttftMs`,
	`lmx benchmark runs export --format csv --out runs.csv`,
	`lmx kvcache run llama.cpp --hf-id Qwen/Qwen3-8B --model-path model.gguf --levels 10000,20000,30000,40000`,
	`lmx kvcache run vllm --mode remote --base-url http://server:8000 --hf-id Qwen/Qwen3-8B --levels 10000,20000,30000,40000`,
	`lmx benchmark submit benchmark.json --api-key bhk_...`,
	`lmx benchmark dry-run benchmark.json --api-key bhk_...`,
	`lmx benchmark validate-local benchmark.json`,
	`lmx eval suite list --out suites.json`,
	`lmx eval suite search reasoning --out reasoning-suites.json`,
	`lmx eval suite show hellaswag --out hellaswag-suite.json`,
	`lmx model search qwen3-8b --out models.json`,
	`lmx eval storage upload traces.jsonl --kind artifact --format jsonl --out artifact-bundle.json`,
	`lmx eval storage download <storageKey> --out traces.jsonl`,
	`lmx eval lm-eval hellaswag --model Qwen/Qwen3-8B --backend hf --hardware hardware.json --dry-run`,
	`lmx eval suite init --slug my-eval --name "My Eval" --category reasoning --out my-eval.json`,
	`lmx eval suite validate my-eval.json`,
	`lmx eval suite submit my-eval.json --api-key bhk_...`,
	`lmx eval execute <suiteSlug> --model <hfId> --base-url http://localhost:8000 --hardware hardware.json --submit`,
	`lmx eval shard gsm8k --base-url http://localhost:8000 --questions 200 --dry-run`,
	`lmx eval shard hellaswag --base-url http://localhost:8000 --questions 200 --dry-run`,
	`lmx eval shard gsm8k --base-url http://localhost:8000 --model Qwen/Qwen3-8B --hardware hardware.json --submit`,
	`lmx eval shard status hellaswag --model Qwen/Qwen3-8B`,
	`lmx eval shard hellaswag --base-url http://localhost:8000 --model Qwen/Qwen3-8B --hardware hardware.json --missing-only --submit`,
	`lmx eval shard hellaswag --base-url http://localhost:8000 --model Qwen/Qwen3-8B --hardware hardware.json --all-missing --submit`,
	`lmx eval terminal import ./terminal-bench-tasks --out ./tb-bundles --version 2.1`,
	`lmx eval terminal verify ./tb-bundles/smoke --oracle`,
	`lmx eval terminal run terminal-bench-2-1 --base-url http://localhost:8000 --model Qwen/Qwen3-8B --hardware hardware.json --submit`,
	`lmx benchmark add-hardware runs/Model/run.json --hardware hardware.json`,
	`lmx benchmark fixup runs/Model/run.json`,
	`lmx hardware template --gpu-name "RTX 3090" --gpu-count 2 --vram-gb 24 --cpu "Ryzen 9 9950X" --ram-gb 96 --os Linux`,
	`lmx model resolve-remote --base-url http://server:8080`,
	`lmx endpoint discover --base-url http://server:8080 --include-server-metadata`,
}

const usageOptions = `  --api-url <url>          LocalMaxxing origin (default: https://www.localmaxxing.com)
  --api-key <key>          API key, defaults to LMX_API_KEY, then saved config
  --no-browser            Do not open the device-login browser automatically
  --profile <name>         Load saved defaults from lmx profile save
  --model <hfId>           HuggingFace model ID
  --backend <name>         lm-eval backend name for eval lm-eval (default: hf)
  --model-args <args>      lm-eval --model_args value
  --num-fewshot <n>        lm-eval --num_fewshot override
  --lm-eval-bin <path>     lm-eval executable (default: lm_eval)
  --questions <n>          Eval-shard questions to run (default: 95%/±5% CI recommendation)
  --shard <index>          Pin a specific eval-shard index instead of the least-run one
  --missing-only          With --submit, skip a covered default shard and run the first missing shard
  --all-missing           With --submit, submit every currently missing shard in ascending order
  --rerun, --force        Allow submitting a shard already covered for the current aggregate key
  --answer-extraction <m>  Eval-shard answer extraction: none, final_answer, last_number, or regex (default: final_answer)
  --answer-regex <re>      Regex used when --answer-extraction regex
  --prompt-template <t>    Eval-shard prompt template using {{input}} and {{choices}}
  --concurrency <n>        Eval-shard parallel requests (default: 1)
  --artifact-limit <n>     Shard traces to submit (default: 0 = all, for a complete whole-shard bundle; >0 keeps a balanced pass/fail sample)
  --task-dir <dir>        Terminal eval bundle directory (one bundle or parent of bundles)
  --dataset <slug>        Terminal eval dataset slug to submit local --task-dir runs against
  --max-turns <n>         Terminal eval agent turn cap (defaults to task manifest)
  --agent-timeout <sec>   Terminal eval whole-agent timeout (defaults to task manifest)
  --agent-cmd <cmd>       Terminal eval external agent command; gets LMX_TERMINAL_* env vars
  --agent-execution <m>   External agent execution: host (default), container, or routed-shell
  --agent-name <name>     Label for external terminal agent submissions (default: external-agent)
  --container-base-url <url> Base URL visible from task containers for container-native agents
  --command-timeout <sec> Terminal eval per-shell-command timeout (default: 120)
  --cleanup-images        Remove locally built terminal task images after each task
  --shell-mode <mode>     Terminal eval built-in harness shell: persistent (default, one shared shell) or stateless (fresh shell per command)
  --oracle                Run terminal task solution/solve.sh instead of the model agent
  --scoring <mode>          Eval-shard scoring: exact_match, loglikelihood, llama_cpp_loglikelihood, code_execution, cruxeval_execution (dataset defaults are canonical)
  --temperature <f>        Sampling temperature for eval-shard runs (default: 0)
  --top-p <f>              Sampling top_p for eval-shard runs (default: 1)
  --quant-format <label>   Quantization container format for eval-shard submit (auto-detected as "gguf" from the model path; override if needed)
  --model-revision <rev>   Model revision for eval-shard submit (default: main)
  --base-url <url>         OpenAI-compatible model endpoint; accepts host or host/v1
  --mode <mode>            Benchmark mode: remote endpoint or local host command
  --served-model <name>    Model name served by the OpenAI-compatible endpoint
  --model-api-key <key>    Optional bearer token for remote endpoint benchmarking
  --prompt <text>          Prompt for remote endpoint benchmark
  --max-tokens <n>         Max generated tokens (eval shard default: 0 = no cap, let the model finish)
  --endpoint-timeout-seconds <n> Timeout for remote endpoint benchmark (default: 600)
  --warmup <n>             Untimed warmup requests before remote endpoint measurement (default: 1)
  --iterations <n>         Timed remote endpoint measurement iterations; median is reported (default: 3)
  --command-timeout-seconds <n> Timeout for local benchmark commands (default: unlimited)
  --no-stream              Disable streaming for remote endpoint benchmark
  --command <cmd>          Local benchmark command, e.g. llama-bench
  --host <addr>            Local model server host for generated server commands
  --port <n>               Local model server port for generated server/benchmark commands
  --model-path <path>      llama.cpp model path; generates llama-bench command
  --llama-scorer <path>     Local helper binary for llama_cpp_loglikelihood scoring (default: lmx-llama-score-hellaswag next to lmx)
  --sandbox-image <name>    Docker image for code_execution scoring (default: lmx-sandbox; build with: docker build -t lmx-sandbox sandbox)
  --sandbox-runtime <bin>   Container runtime for code_execution (default: docker; accepts quoted commands like "sudo docker")
  --sandbox-use-sudo        Prefix the default container runtime with sudo
  --sandbox-relaxed-security
                           Omit cap-drop/no-new-privileges/read-only when the host rejects the hardened profile
  --sandbox-cmd <cmd>       Override the sandbox launcher entirely (e.g. podman, or "python3 sandbox/run_sandbox.py" without Docker)
  --sandbox-memory <size>   Memory cap for the code sandbox container (default: 2g)
  --sandbox-cpus <n>        CPU cap for the code sandbox container (default: 2)
  --n-samples <n>           Samples per question for code evals (default: 1 greedy; >1 enables pass@k sampling)
  --k <n>                   k for pass@k over --n-samples (default: 1)
  --few-shot <n>            Few-shot examples for MBPP-family code evals (default: 3 for mbpp*, 0 otherwise)
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
  --dir <dir>              Target skills directory for lmx skill install (default: .claude/skills)
  --hardware <path>        JSON hardware object required when submitting
  --quantization <label>   Quantization label (auto-detected from the endpoint for remote benchmark and eval-shard runs when omitted)
  --results <path>         Existing lm-eval output JSON for run upload
  --kind <kind>            Storage upload kind, usually artifact or dataset
  --format <format>        Storage file format, e.g. json, jsonl, parquet, zip
  --item-count <n>         Optional record/sample count for storage metadata
  --limit <n>              Optional search/list result limit
  --submit                 Upload run to LocalMaxxing
  --dry-run                For benchmark run: write a measurement plan; for submit commands: authenticated API validation without creating a run
  --include-server-metadata Probe optional endpoint /props and /hardware metadata during discover
  --gpu-name <name>       Hardware template GPU name
  --gpu-count <n>         Hardware template GPU count
  --vram-gb <gb>          Hardware template VRAM in GB
  --cpu <name>            Hardware template CPU name
  --ram-gb <gb>           Hardware template system RAM in GB
  --os <name>             Hardware template OS name (default: runtime OS)
  --power-watts <n>       Hardware template power draw in watts
  --hw-class <class>      Hardware template class, e.g. DISCRETE_GPU or CPU_ONLY
  --name <name>            Saved setup name to pull (case-insensitive) for setups pull
  --id <id>                Saved setup id to pull for setups pull
  --default                Pull the default saved setup for setups pull
  --out <path>             Write computed payload/result JSON`

var commandDescriptions = map[string]string{
	"endpoint discover":      "Discover an OpenAI-compatible endpoint and benchmark command hints.",
	"endpoint":               "Discover OpenAI-compatible model endpoints.",
	"benchmark":              "Create, manage, validate, and submit benchmark runs.",
	"benchmark run":          "Measure a model or write a benchmark measurement plan.",
	"benchmark runs":         "List and edit saved benchmark run files.",
	"benchmark add-hardware": "Attach server hardware metadata to a saved run.",
	"benchmark fixup":        "Inspect a saved run and print remediation commands.",
	"hardware":               "Detect, validate, or template hardware metadata.",
	"hardware validate":      "Validate hardware metadata against the live LocalMaxxing schema and allowlist.",
	"hardware template":      "Generate hardware metadata from explicit flags.",
	"setups":                 "List and pull your saved hardware/engine setups.",
	"setups list":            "List saved setups stored in your LocalMaxxing account.",
	"setups pull":            "Write a hardware.json from a saved setup.",
	"skill":                  "Print or install the bundled agent skill that documents the CLI.",
	"skill print":            "Print the bundled SKILL.md to stdout (or --out <path>).",
	"skill install":          "Write the bundled skill files into a skills directory (default .claude/skills).",
	"model":                  "Search or resolve HuggingFace model IDs.",
	"model search":           "Search LocalMaxxing model records.",
	"model resolve-remote":   "Resolve a remote endpoint alias to likely HF candidates.",
	"eval":                   "Discover, run, and submit evaluation suites.",
	"eval suite":             "List, inspect, initialize, and submit eval suites.",
	"eval run":               "Run an approved suite locally and write/submit a run payload.",
	"eval pull":              "Download a suite + datasets for offline runs and inspection.",
	"eval submit":            "Submit a previously saved run payload (deferred submit).",
	"eval shard":             "Run eval shards, inspect aggregate shard coverage, and guard duplicate submissions.",
	"eval shard status":      "Print aggregate shard coverage and missing shard indexes for a model.",
	"eval terminal":          "Run Terminal-Bench task bundles with the localmaxxing Docker agent harness.",
	"kvcache":                "Run KV-cache and context-length sweeps.",
	"profile":                "Save and manage reusable CLI defaults.",
	"auth":                   "Manage LocalMaxxing API authentication.",
	"server":                 "Build or run local model server commands.",
}

func commandHelp(args cliArgs) (string, bool) {
	for n := len(args.positional); n >= 1; n-- {
		prefix := strings.Join(args.positional[:n], " ")
		matches := []string{}
		for _, ex := range usageExamples {
			if ex == "lmx "+prefix || strings.HasPrefix(ex, "lmx "+prefix+" ") {
				matches = append(matches, ex)
			}
		}
		if len(matches) == 0 {
			continue
		}
		var b strings.Builder
		if desc := commandDescriptions[prefix]; desc != "" {
			b.WriteString(desc)
			b.WriteString("\n\n")
		}
		b.WriteString("Usage:\n")
		for _, match := range matches {
			b.WriteString("  ")
			b.WriteString(match)
			b.WriteByte('\n')
		}
		b.WriteString("\nRun `lmx --help` for all commands and the full option list.")
		return b.String(), true
	}
	return "", false
}

func usage() {
	fmt.Println("LocalMaxxing CLI\n\nUsage:")
	for _, ex := range usageExamples {
		fmt.Println("  " + ex)
	}
	fmt.Println("\nOptions:")
	fmt.Println(usageOptions)
}
