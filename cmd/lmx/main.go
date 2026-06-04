package main

import (
	"bytes"
	"crypto/sha256"
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
	"strconv"
	"strings"
	"time"
)

const defaultAPIURL = "https://www.localmaxxing.com"

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

func (e cliError) Error() string { return e.Message }

func main() {
	if err := run(os.Args[1:]); err != nil {
		printError(err)
		os.Exit(1)
	}
}

func run(argv []string) error {
	args := parseArgs(argv)
	if err := applyProfile(&args); err != nil {
		return err
	}
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
	case "eval", "benchmark", "bench", "auth", "hardware", "context", "agent-context", "model", "profile":
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

func opt(args cliArgs, key string) string { return args.opts[key] }
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
		for _, key := range []string{"mode", "api-url", "base-url", "model", "hf-id", "served-model", "model-name", "quantization", "hardware", "model-path", "command", "max-tokens", "prompt-tokens", "output-tokens", "engine", "backend"} {
			if value := opt(args, key); value != "" {
				profileOpts[key] = value
			}
		}
		for _, key := range []string{"no-stream"} {
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
		printInfo("hardware_written", map[string]any{"path": out, "hwClass": hardware["hwClass"], "gpuName": hardware["gpuName"], "vramGb": hardware["vramGb"]})
		fmt.Println("Use it with:")
		fmt.Println("  lmx benchmark run <engine> --hardware " + out)
		return nil
	}
	data, _ := json.MarshalIndent(hardware, "", "  ")
	fmt.Println(string(data))
	return nil
}

func detectHardware() map[string]any {
	base := map[string]any{"hwClass": "CPU_ONLY", "cpuName": runtime.GOARCH, "systemOs": runtime.GOOS, "systemArch": runtime.GOARCH, "cpuThreads": runtime.NumCPU()}
	if runtime.GOOS == "darwin" && runtime.GOARCH == "arm64" {
		base["hwClass"] = "APPLE_SILICON"
	}
	if out, err := exec.Command("nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits").Output(); err == nil {
		line := strings.TrimSpace(strings.Split(string(out), "\n")[0])
		parts := strings.Split(line, ",")
		if len(parts) >= 2 {
			base["hwClass"] = "DISCRETE_GPU"
			base["gpuName"] = strings.TrimSpace(parts[0])
			if mb, err := strconv.ParseFloat(strings.TrimSpace(parts[1]), 64); err == nil {
				base["vramGb"] = round1(mb / 1024)
			}
		}
	}
	return base
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
	limit := firstNonEmpty(opt(args, "limit"), "10")
	endpoint := apiURL(args) + "/api/models/search?q=" + url.QueryEscape(query) + "&limit=" + url.QueryEscape(limit)
	value, err := fetchJSON("GET", endpoint, "", nil)
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
	runner := strings.ToUpper(firstNonEmpty(opt(args, "runner"), "CUSTOM"))
	if strings.EqualFold(runner, "lm-eval-harness") {
		runner = "LM_EVAL_HARNESS"
	}
	taskKey := firstNonEmpty(opt(args, "tasks"), opt(args, "task"), slug)
	scoring := firstNonEmpty(opt(args, "scoring-method"), "exact_match")
	payload := map[string]any{
		"slug": slug, "name": name, "description": opt(args, "description"), "category": firstNonEmpty(opt(args, "category"), "general"), "runner": runner,
		"suiteDoc": map[string]any{"version": "1", "runner": mapRunnerDoc(runner), "scoringMethod": scoring, "higherIsBetter": true, "aggregation": "mean", "tasks": []any{map[string]any{"key": taskKey, "displayName": name, "weight": 1}}},
	}
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
	doc := asObject(obj["suiteDoc"])
	if doc == nil {
		errs = append(errs, "suiteDoc must be an object")
	} else if tasks, ok := doc["tasks"].([]any); !ok || len(tasks) == 0 {
		errs = append(errs, "suiteDoc.tasks must contain at least one task")
	}
	if len(errs) > 0 {
		return cliError{"invalid_suite", "Suite payload is invalid.", errs, nil}
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
		putReq.Header.Set("Content-Type", fmt.Sprint(metadata["contentType"]))
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
	if action == "validate" {
		action = "dry-run"
	}
	if action == "run" || action == "measure" {
		payload, err := benchmarkPayloadFromFlags(target, args)
		if err != nil {
			return err
		}
		out := firstNonEmpty(opt(args, "out"), "localmaxxing-benchmark.json")
		if err := writeJSON(out, payload); err != nil {
			return err
		}
		printInfo("benchmark_payload_written", map[string]any{"path": out, "engine": payload["engineName"]})
		if hasFlag(args, "submit") || hasFlag(args, "dry-run") {
			endpoint := "/api/benchmarks"
			if hasFlag(args, "dry-run") {
				endpoint = "/api/benchmarks/dry-run"
			}
			return submitPayload(endpoint, hasFlag(args, "dry-run"), "benchmark", args, payload)
		}
		printNextSteps("benchmark", out)
		return nil
	}
	if action != "submit" && action != "dry-run" {
		return errors.New("Unknown benchmark command. Use run, submit, or dry-run.")
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
	endpoint := "/api/benchmarks"
	if action == "dry-run" {
		endpoint = "/api/benchmarks/dry-run"
	}
	return submitPayload(endpoint, action == "dry-run", "benchmark", args, value)
}

func benchmarkPayloadFromFlags(engine string, args cliArgs) (map[string]any, error) {
	model := firstNonEmpty(opt(args, "hf-id"), opt(args, "model"))
	if model == "" {
		return nil, cliError{"missing_model", "benchmark run requires --hf-id or --model", []string{"Pass --hf-id <HuggingFace model id>."}, nil}
	}
	engineName := normalizeEngineName(firstNonEmpty(opt(args, "engine"), engine))
	if engineName == "" {
		return nil, cliError{"missing_engine", "benchmark run requires an engine name", []string{"Pass it positionally, e.g. lmx benchmark run llama.cpp, or with --engine vllm."}, nil}
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
	if mode == "local" && opt(args, "base-url") != "" && opt(args, "command") == "" && opt(args, "results") == "" {
		return nil, cliError{"invalid_benchmark_mode", "Local benchmark mode needs --command, --results, or explicit metric flags.", []string{"Use --mode remote with --base-url when benchmarking an endpoint from another machine.", "Use --mode local --command \"llama-bench ...\" when running on the host server."}, nil}
	}
	var commandOutput string
	if mode == "remote" {
		printStatus(args, "benchmark_remote_start", map[string]any{"baseUrl": opt(args, "base-url"), "servedModel": firstNonEmpty(opt(args, "served-model"), opt(args, "model-name"), model)})
		endpointMetrics, err := measureOpenAIEndpoint(args, model)
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
		printStatus(args, "benchmark_results_read_complete", map[string]any{"path": resultsPath, "bytes": len(data)})
	} else if commandSnippet := localBenchmarkCommand(engineName, args); commandSnippet != "" {
		printStatus(args, "benchmark_local_command_start", map[string]any{"command": commandSnippet})
		output, err := runBenchmarkCommand(commandSnippet)
		if err != nil {
			return nil, err
		}
		commandOutput = output
		metrics["engineFlags"] = map[string]any{"mode": "local", "commandSnippet": commandSnippet}
		printStatus(args, "benchmark_local_command_complete", map[string]any{"outputBytes": len(output)})
	}
	if commandOutput != "" {
		parsed := parseBenchmarkOutput(commandOutput)
		for key, value := range parsed {
			metrics[key] = value
		}
		printStatus(args, "benchmark_metrics_detected", metricStatusFields(parsed))
	}

	hardware := any(detectHardware())
	if hardwarePath := opt(args, "hardware"); hardwarePath != "" {
		loaded, err := readJSON(hardwarePath)
		if err != nil {
			return nil, err
		}
		hardware = loaded
	}

	payload := map[string]any{"engineName": engineName, "hfId": model, "modelRevision": firstNonEmpty(opt(args, "model-revision"), "main"), "hardware": hardware, "quantization": quantization}
	payload["benchmarkMode"] = mode
	for key, value := range metrics {
		payload[key] = value
	}
	for flag, field := range map[string]string{"tok-s-out": "tokSOut", "tok-s-prefill": "tokSPrefill", "tok-s-total": "tokSTotal", "ttft-ms": "ttftMs", "peak-vram-gb": "peakVramGb", "context-length": "contextLength", "batch-size": "batchSize", "input-len": "inputLen", "output-len": "outputLen", "num-prompts": "numPrompts"} {
		if value := opt(args, flag); value != "" {
			if n, err := strconv.ParseFloat(value, 64); err == nil {
				payload[field] = n
			} else {
				payload[field] = value
			}
		}
	}
	if notes := opt(args, "notes"); notes != "" {
		payload["notes"] = notes
	}
	applyComparableBenchmarkMetrics(payload, mode)
	payload["provenance"] = map[string]any{"cli": "localmaxxing-go", "benchmarkMode": mode, "metricSource": payload["metricSource"], "timingSource": payload["timingSource"], "ttftSource": payload["ttftSource"], "createdAt": time.Now().UTC().Format(time.RFC3339)}
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

func applyComparableBenchmarkMetrics(payload map[string]any, mode string) {
	promptTokens := numberField(payload, "promptTokens")
	outputTokens := numberField(payload, "outputTokens")
	tokSPrefill := numberField(payload, "tokSPrefill")
	tokSOut := numberField(payload, "tokSOut")
	if mode == "local" {
		payload["metricSource"] = "local_runtime"
		payload["timingSource"] = "llama_bench_runtime"
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
	if engineName != "llama.cpp" || opt(args, "model-path") == "" {
		return ""
	}
	cmd := []string{"llama-bench", "-m", shellQuote(opt(args, "model-path"))}
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
	if value := opt(args, "extra-bench-args"); value != "" {
		cmd = append(cmd, value)
	}
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
	baseURL = strings.TrimRight(baseURL, "/")
	servedModel := firstNonEmpty(opt(args, "served-model"), opt(args, "model-name"))
	servedModelSource := "explicit"
	if servedModel == "" {
		detected, err := detectServedModel(baseURL, opt(args, "model-api-key"), hfID)
		if err == nil && detected != "" {
			servedModel = detected
			servedModelSource = "v1_models"
			printStatus(args, "served_model_detected", map[string]any{"servedModel": servedModel, "source": servedModelSource})
		} else {
			servedModel = hfID
			servedModelSource = "hf_id_fallback"
			printStatus(args, "served_model_fallback", map[string]any{"servedModel": servedModel, "source": servedModelSource, "reason": errString(err)})
		}
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
		"model": servedModel,
		"messages": []any{map[string]any{"role": "user", "content": prompt}},
		"max_tokens": maxTokens,
		"temperature": temperature,
		"stream": stream,
	}
	if stream {
		body["stream_options"] = map[string]any{"include_usage": true}
	}
	bodyData, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	started := time.Now()
	req, err := http.NewRequest("POST", baseURL+"/v1/chat/completions", bytes.NewReader(bodyData))
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
		"prompt": prompt,
		"outputText": outputText,
		"promptTokens": float64(promptTokens),
		"outputTokens": float64(outputTokens),
		"tokSOut": round1(float64(outputTokens) / (generationMs / 1000)),
		"tokSTotal": round1(float64(promptTokens+outputTokens) / (totalMs / 1000)),
		"engineFlags": map[string]any{"mode": "remote", "baseUrl": baseURL, "servedModel": servedModel, "servedModelSource": servedModelSource, "stream": stream, "maxTokens": maxTokens},
		"tokenSources": map[string]any{"prompt": promptTokenResult.Source, "output": outputTokenResult.Source},
		"timingSource": "client_observed_http",
		"metricSource": "remote_endpoint",
		"ttftSource": map[bool]string{true: "stream_first_token", false: "unavailable_no_stream"}[!firstTokenAt.IsZero()],
	}
	if !firstTokenAt.IsZero() {
		ttftMs := float64(firstTokenAt.Sub(started).Milliseconds())
		metrics["ttftMs"] = ttftMs
		if ttftMs > 0 {
			metrics["tokSPrefill"] = round1(float64(promptTokens) / (ttftMs / 1000))
		}
	}
	return metrics, nil
}

func detectServedModel(baseURL, apiKey, preferred string) (string, error) {
	req, err := http.NewRequest("GET", strings.TrimRight(baseURL, "/")+"/v1/models", nil)
	if err != nil {
		return "", err
	}
	if apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", err
	}
	defer res.Body.Close()
	if res.StatusCode < 200 || res.StatusCode >= 300 {
		return "", fmt.Errorf("/v1/models returned %s", res.Status)
	}
	var body map[string]any
	if err := json.NewDecoder(res.Body).Decode(&body); err != nil {
		return "", err
	}
	data, _ := body["data"].([]any)
	first := ""
	for _, item := range data {
		obj := asObject(item)
		if obj == nil {
			continue
		}
		id := stringValue(obj["id"])
		if id == "" {
			continue
		}
		if first == "" {
			first = id
		}
		if id == preferred || strings.EqualFold(id, preferred) {
			return id, nil
		}
	}
	if first != "" {
		return first, nil
	}
	return "", errors.New("/v1/models did not return any model ids")
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

func parseLlamaBenchTable(text string) map[string]float64 {
	metrics := map[string]float64{}
	testPattern := regexp.MustCompile(`(?i)^(pp|tg)(\d+)\b`)
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
	backend := firstNonEmpty(opt(args, "backend"), "hf")
	command := firstNonEmpty(opt(args, "lm-eval-bin"), "lm_eval")
	resultsPath := firstNonEmpty(opt(args, "results"), "localmaxxing-lm-eval-results.json")
	tasks := firstNonEmpty(opt(args, "tasks"), suiteSlug)
	modelArgs := opt(args, "model-args")
	if modelArgs == "" && backend == "hf" {
		modelArgs = "pretrained=" + model
	}
	cmdArgs := []string{"--model", backend}
	if modelArgs != "" {
		cmdArgs = append(cmdArgs, "--model_args", modelArgs)
	}
	cmdArgs = append(cmdArgs, "--tasks", tasks)
	if fewshot := firstNonEmpty(opt(args, "num-fewshot"), opt(args, "fewshot")); fewshot != "" {
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
	resultsPath := opt(args, "results")
	if resultsPath == "" {
		return cliError{"go_eval_run_partial", "The Go rewrite currently supports eval run uploads from --results only.", []string{"For CUSTOM local eval execution, keep using the TypeScript CLI until the runner is ported.", "Pass --results <lm-eval-output.json> for LM-Eval upload payload generation."}, nil}
	}
	results, err := readJSON(resultsPath)
	if err != nil {
		return err
	}
	payload := map[string]any{"suiteSlug": suiteSlug, "hfId": firstNonEmpty(opt(args, "model"), "<required-before-submit>"), "quantization": opt(args, "quantization"), "executionMode": "LM_EVAL_LOCAL", "judgeMode": "NONE", "runnerVersion": "localmaxxing-go lm-eval-upload", "results": results, "artifacts": []any{}}
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
	printInfo("run_payload_written", map[string]any{"path": out, "suite": suiteSlug})
	if hasFlag(args, "submit") || hasFlag(args, "dry-run") {
		endpoint := "/api/evals/runs"
		if hasFlag(args, "dry-run") {
			endpoint = "/api/evals/runs/dry-run"
		}
		return submitPayload(endpoint, hasFlag(args, "dry-run"), "run", args, payload)
	}
	return nil
}

func submitPayload(endpoint string, dryRun bool, label string, args cliArgs, payload any) error {
	key := apiKey(args)
	if key == "" {
		return missingAPIKey("--api-key or LMX_API_KEY is required for "+label+" submit/dry-run")
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

func printError(err error) {
	var ce cliError
	if errors.As(err, &ce) {
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
  lmx benchmark run vllm --mode remote --base-url http://server:8000 --hf-id Qwen/Qwen3-8B --quantization fp16 --dry-run
  lmx benchmark run llama.cpp --mode local --hf-id Qwen/Qwen3-8B --quantization Q4_K_M --command "llama-bench -m model.gguf" --dry-run
  lmx benchmark run llama.cpp --mode local --hf-id Qwen/Qwen3-8B --quantization Q4_K_M --model-path model.gguf --dry-run
  lmx benchmark submit benchmark.json --api-key bhk_...
  lmx benchmark dry-run benchmark.json --api-key bhk_...
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
  --base-url <url>         OpenAI-compatible model endpoint for eval execute
  --mode <mode>            Benchmark mode: remote endpoint or local host command
  --served-model <name>    Model name served by the OpenAI-compatible endpoint
  --model-api-key <key>    Optional bearer token for remote endpoint benchmarking
  --prompt <text>          Prompt for remote endpoint benchmark
  --max-tokens <n>         Max generated tokens for remote endpoint benchmark
  --no-stream              Disable streaming for remote endpoint benchmark
  --command <cmd>          Local benchmark command, e.g. llama-bench
  --model-path <path>      llama.cpp model path; generates llama-bench command
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
  --dry-run                Validate upload without creating a run
  --out <path>             Write computed payload/result JSON`)
}
