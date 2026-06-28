package main

import (
	"archive/tar"
	"bufio"
	"bytes"
	"compress/gzip"
	"context"
	"crypto/rand"
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
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/BurntSushi/toml"
)

const terminalSystemPrompt = "You control a Linux terminal. Each reply MUST contain exactly one ```bash fenced block with one command, which will be executed; its stdout/stderr/exit code is returned. When the task is complete, reply with the single token TASK_COMPLETE and no code block. Non-interactive only."

const terminalSessionSystemPrompt = "You control a persistent Linux shell session inside a container. State persists across replies: your working directory, environment variables, and background jobs carry over from one command to the next. Each reply MUST contain exactly one ```bash fenced block with one command, which is executed in that same shell; its stdout/stderr and exit code are returned. When the task is complete, reply with the single token TASK_COMPLETE and no code block. Run only non-interactive commands (no programs that wait for a TTY or block on stdin)."

type terminalTask struct {
	ID          string                    `json:"id"`
	Version     string                    `json:"version"`
	Instruction string                    `json:"instruction"`
	Category    string                    `json:"category,omitempty"`
	Source      string                    `json:"source"`
	Image       terminalImage             `json:"image"`
	Agent       terminalAgentConfig       `json:"agent"`
	Verifier    terminalVerifierConfig    `json:"verifier"`
	Environment terminalEnvironmentConfig `json:"environment"`
}

type terminalImage struct {
	Prebuilt        string `json:"prebuilt,omitempty"`
	Dockerfile      string `json:"dockerfile,omitempty"`
	Context         string `json:"context,omitempty"`
	BuildTimeoutSec int    `json:"buildTimeoutSec,omitempty"`
}

type terminalAgentConfig struct {
	TimeoutSec int    `json:"timeoutSec"`
	MaxTurns   int    `json:"maxTurns"`
	User       string `json:"user"`
}

type terminalVerifierConfig struct {
	TimeoutSec int               `json:"timeoutSec"`
	Command    string            `json:"command"`
	RewardFile string            `json:"rewardFile"`
	User       string            `json:"user"`
	Env        map[string]string `json:"env,omitempty"`
}

type terminalEnvironmentConfig struct {
	CPUs         float64           `json:"cpus"`
	MemoryMb     int               `json:"memoryMb"`
	StorageMb    int               `json:"storageMb"`
	GPUs         int               `json:"gpus"`
	Network      string            `json:"network"`
	AllowedHosts []string          `json:"allowedHosts"`
	Env          map[string]string `json:"env"`
}

type terminalConfig struct {
	apiKey            string
	args              cliArgs
	maxTokens         int
	temperature       float64
	topP              float64
	commandTimeoutSec int
	agentTimeoutSec   int
	maxTurns          int
	cleanupImages     bool
	oracle            bool
	agentCommand      string
	agentExecution    string
	shellMode         string
}

type terminalTaskResult struct {
	pass           bool
	scored         bool
	turns          int
	transcript     string
	verifierOutput string
	latencyMs      int64
	errText        string
	errCode        string
	instruction    string
	prompt         string
}

type terminalBundle struct {
	Task terminalTask
	Dir  string
}

type harborTaskToml struct {
	Task struct {
		Name        string `toml:"name"`
		Description string `toml:"description"`
	} `toml:"task"`
	Metadata struct {
		Category string `toml:"category"`
	} `toml:"metadata"`
	Verifier struct {
		TimeoutSec tomlNumber        `toml:"timeout_sec"`
		Command    string            `toml:"command"`
		RewardFile string            `toml:"reward_file"`
		Env        map[string]string `toml:"env"`
		User       string            `toml:"user"`
	} `toml:"verifier"`
	Agent struct {
		TimeoutSec tomlNumber `toml:"timeout_sec"`
		MaxTurns   tomlNumber `toml:"max_turns"`
		User       string     `toml:"user"`
	} `toml:"agent"`
	Environment struct {
		DockerImage     string            `toml:"docker_image"`
		BuildTimeoutSec tomlNumber        `toml:"build_timeout_sec"`
		CPUs            tomlNumber        `toml:"cpus"`
		MemoryMb        tomlNumber        `toml:"memory_mb"`
		StorageMb       tomlNumber        `toml:"storage_mb"`
		GPUs            tomlNumber        `toml:"gpus"`
		NetworkMode     string            `toml:"network_mode"`
		AllowedHosts    []string          `toml:"allowed_hosts"`
		AllowInternet   *bool             `toml:"allow_internet"`
		Env             map[string]string `toml:"env"`
	} `toml:"environment"`
}

// tomlNumber accepts TOML integers or floats, since harbor task.toml writes
// timeouts as floats (900.0) but resource limits as integers (2048).
type tomlNumber struct {
	value float64
}

func (n *tomlNumber) UnmarshalTOML(data any) error {
	switch x := data.(type) {
	case int64:
		n.value = float64(x)
	case int:
		n.value = float64(x)
	case float64:
		n.value = x
	case nil:
		n.value = 0
	default:
		return fmt.Errorf("unexpected numeric type %T", data)
	}
	return nil
}

func (n tomlNumber) Int() int       { return int(n.value) }
func (n tomlNumber) Float() float64 { return n.value }

func handleEvalTerminal(sub string, args cliArgs) error {
	switch sub {
	case "import":
		return runTerminalImport(args)
	case "run":
		return runTerminalEval(args, false)
	case "verify":
		return runTerminalEval(args, true)
	default:
		return cliError{"unknown_subcommand", "Unknown eval terminal subcommand.", []string{"Use one of: import, run, verify."}, map[string]any{"subcommand": sub}}
	}
}

func runTerminalImport(args cliArgs) error {
	src := positional(args, 3)
	if src == "" {
		return cliError{"missing_option", "eval terminal import requires a source directory.", []string{"Run: lmx eval terminal import <harbor-task-or-parent-dir> --out <bundles-dir>."}, nil}
	}
	out, err := requireOpt(args, "out")
	if err != nil {
		return err
	}
	version := firstNonEmpty(opt(args, "version"), "2.1")
	taskDirs, err := findHarborTaskDirs(src)
	if err != nil {
		return err
	}
	if len(taskDirs) == 0 {
		return cliError{"task_import_failed", "No harbor task.toml files were found.", []string{"Pass a harbor task directory or a parent directory containing task subdirectories."}, map[string]any{"src": src}}
	}
	for _, taskDir := range taskDirs {
		if err := importHarborTask(taskDir, out, version); err != nil {
			return err
		}
	}
	printInfo("terminal_import_complete", map[string]any{"src": src, "out": out, "tasks": len(taskDirs), "version": version})
	return nil
}

func findHarborTaskDirs(src string) ([]string, error) {
	info, err := os.Stat(src)
	if err != nil {
		return nil, cliError{"task_import_failed", fmt.Sprintf("Could not inspect source directory: %v", err), []string{"Check that the source path exists and is readable."}, map[string]any{"src": src}}
	}
	if !info.IsDir() {
		return nil, cliError{"task_import_failed", "Source must be a directory.", []string{"Pass a harbor task directory or parent directory."}, map[string]any{"src": src}}
	}
	if _, err := os.Stat(filepath.Join(src, "task.toml")); err == nil {
		return []string{src}, nil
	}
	entries, err := os.ReadDir(src)
	if err != nil {
		return nil, cliError{"task_import_failed", fmt.Sprintf("Could not read source directory: %v", err), []string{"Check permissions on the source directory."}, map[string]any{"src": src}}
	}
	var dirs []string
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		candidate := filepath.Join(src, entry.Name())
		if _, err := os.Stat(filepath.Join(candidate, "task.toml")); err == nil {
			dirs = append(dirs, candidate)
		}
	}
	sort.Strings(dirs)
	return dirs, nil
}

func importHarborTask(taskDir, out, version string) error {
	id := filepath.Base(filepath.Clean(taskDir))
	var ht harborTaskToml
	if _, err := toml.DecodeFile(filepath.Join(taskDir, "task.toml"), &ht); err != nil {
		return cliError{"task_import_failed", fmt.Sprintf("%s: could not parse task.toml: %v", id, err), []string{"Fix the harbor task.toml syntax and retry."}, map[string]any{"taskId": id, "path": filepath.Join(taskDir, "task.toml")}}
	}
	instructionBytes, err := os.ReadFile(filepath.Join(taskDir, "instruction.md"))
	if err != nil {
		return cliError{"task_import_failed", fmt.Sprintf("%s: no instruction.md", id), []string{"Ensure each harbor task directory contains instruction.md."}, map[string]any{"taskId": id, "path": filepath.Join(taskDir, "instruction.md"), "error": err.Error()}}
	}
	image := terminalImage{}
	if strings.TrimSpace(ht.Environment.DockerImage) != "" {
		image.Prebuilt = strings.TrimSpace(ht.Environment.DockerImage)
	} else {
		dockerfile := filepath.Join(taskDir, "environment", "Dockerfile")
		if _, err := os.Stat(dockerfile); err != nil {
			return cliError{"task_import_failed", fmt.Sprintf("%s: neither environment.docker_image nor environment/Dockerfile is present", id), []string{"Add [environment].docker_image or include environment/Dockerfile in the harbor task."}, map[string]any{"taskId": id, "dockerfile": dockerfile}}
		}
		image.Dockerfile = "environment/Dockerfile"
		image.Context = "environment"
		image.BuildTimeoutSec = firstPositive(ht.Environment.BuildTimeoutSec.Int(), 600)
	}
	network := strings.TrimSpace(ht.Environment.NetworkMode)
	if network == "" && ht.Environment.AllowInternet != nil {
		if *ht.Environment.AllowInternet {
			network = "public"
		} else {
			network = "no-network"
		}
	}
	network = normalizeTerminalNetwork(firstNonEmpty(network, "public"))
	if network == "" {
		return cliError{"task_import_failed", fmt.Sprintf("%s: invalid environment network", id), []string{"Set network_mode to public, no-network, or allowlist."}, map[string]any{"taskId": id, "network_mode": ht.Environment.NetworkMode}}
	}
	task := terminalTask{
		ID:          id,
		Version:     version,
		Instruction: string(instructionBytes),
		Category:    ht.Metadata.Category,
		Source:      firstNonEmpty(ht.Task.Name, "terminal-bench/"+id),
		Image:       image,
		Agent:       terminalAgentConfig{TimeoutSec: firstPositive(ht.Agent.TimeoutSec.Int(), 900), MaxTurns: firstPositive(ht.Agent.MaxTurns.Int(), 50), User: ht.Agent.User},
		Verifier:    terminalVerifierConfig{TimeoutSec: firstPositive(ht.Verifier.TimeoutSec.Int(), 900), Command: firstNonEmpty(ht.Verifier.Command, "bash /tests/test.sh"), RewardFile: firstNonEmpty(ht.Verifier.RewardFile, "/logs/verifier/reward.txt"), User: ht.Verifier.User, Env: nonNilStringMap(ht.Verifier.Env)},
		Environment: terminalEnvironmentConfig{CPUs: firstPositiveFloat(ht.Environment.CPUs.Float(), 1), MemoryMb: firstPositive(ht.Environment.MemoryMb.Int(), 2048), StorageMb: firstPositive(ht.Environment.StorageMb.Int(), 10240), GPUs: ht.Environment.GPUs.Int(), Network: network, AllowedHosts: ht.Environment.AllowedHosts, Env: nonNilStringMap(ht.Environment.Env)},
	}
	dest := filepath.Join(out, id)
	if err := os.RemoveAll(dest); err != nil {
		return cliError{"task_import_failed", fmt.Sprintf("%s: could not replace output directory: %v", id, err), []string{"Check permissions on --out."}, map[string]any{"taskId": id, "out": dest}}
	}
	if err := os.MkdirAll(dest, 0o755); err != nil {
		return cliError{"task_import_failed", fmt.Sprintf("%s: could not create output directory: %v", id, err), []string{"Check permissions on --out."}, map[string]any{"taskId": id, "out": dest}}
	}
	for _, subdir := range []string{"environment", "tests", "solution"} {
		srcSub := filepath.Join(taskDir, subdir)
		if _, err := os.Stat(srcSub); err == nil {
			if err := copyDir(srcSub, filepath.Join(dest, subdir)); err != nil {
				return cliError{"task_import_failed", fmt.Sprintf("%s: could not copy %s: %v", id, subdir, err), []string{"Check that all task files are readable and retry."}, map[string]any{"taskId": id, "subdir": subdir}}
			}
		}
	}
	if _, err := os.Stat(filepath.Join(dest, "tests")); err != nil {
		return cliError{"task_import_failed", fmt.Sprintf("%s: tests/ is required", id), []string{"Ensure harbor verifier assets are present under tests/."}, map[string]any{"taskId": id}}
	}
	if err := writeJSON(filepath.Join(dest, "task.json"), task); err != nil {
		return cliError{"task_import_failed", fmt.Sprintf("%s: could not write task.json: %v", id, err), []string{"Check permissions on --out."}, map[string]any{"taskId": id, "out": dest}}
	}
	return nil
}

func runTerminalEval(args cliArgs, forceOracle bool) error {
	localDir := opt(args, "task-dir")
	if forceOracle && localDir == "" {
		localDir = positional(args, 3)
	}
	dataset := positional(args, 3)
	if localDir != "" && (forceOracle || hasFlag(args, "oracle")) {
		dataset = opt(args, "dataset")
	}
	if hasFlag(args, "submit") && localDir != "" && dataset == "" {
		dataset = opt(args, "dataset")
	}
	if !forceOracle && localDir == "" && dataset == "" {
		return cliError{"missing_option", "eval terminal run requires a dataset slug or --task-dir.", []string{"Run: lmx eval terminal run terminal-bench-2-1 --base-url <url> --model <hfId>.", "For local bundles, run: lmx eval terminal run --task-dir ./bundles --base-url <url> --model <hfId>."}, nil}
	}
	bundles, cleanup, shardIndex, err := acquireTerminalBundles(args, dataset, localDir)
	if cleanup != "" {
		defer os.RemoveAll(cleanup)
	}
	if err != nil {
		return err
	}
	if len(bundles) == 0 {
		return cliError{"bundle_invalid", "No terminal task bundles were found.", []string{"Point --task-dir at one bundle or a parent directory of bundles containing task.json and tests/."}, map[string]any{"taskDir": localDir}}
	}
	if err := dockerPreflight(); err != nil {
		return err
	}
	rawBaseURL := opt(args, "base-url")
	baseURL := ""
	callModel := ""
	declaredModel := opt(args, "model")
	var quantResolution map[string]any
	resolvedQuant, resolvedQuantFormat := "", opt(args, "quant-format")
	var modelResolution map[string]any
	agentCommand := opt(args, "agent-cmd")
	if !forceOracle && !hasFlag(args, "oracle") {
		if agentCommand == "" {
			var err error
			rawBaseURL, err = requireOpt(args, "base-url")
			if err != nil {
				return err
			}
		} else {
			rawBaseURL = firstNonEmpty(rawBaseURL, os.Getenv("LM_STUDIO_BASE_URL"), os.Getenv("LLAMA_CPP_BASE_URL"))
		}
		if rawBaseURL != "" {
			baseURL = openAIBaseURL(rawBaseURL)
			servedModel := opt(args, "served-model")
			var servedModelInfo map[string]any
			if servedModel == "" {
				if detected, info, derr := detectServedModel(baseURL, opt(args, "model-api-key"), declaredModel); derr == nil {
					servedModel = detected
					servedModelInfo = info
				} else {
					printStatus(args, "eval_model_detection_unavailable", map[string]any{"baseUrl": baseURL, "reason": derr.Error()})
				}
			} else if _, info, derr := detectServedModel(baseURL, opt(args, "model-api-key"), servedModel); derr == nil {
				servedModelInfo = info
			}
			callModel = firstNonEmpty(servedModel, declaredModel, "local")
			quantResolution = remoteQuantizationResolution(args, baseURL, opt(args, "model-api-key"), opt(args, "quantization"), servedModelInfo)
			resolvedQuant = firstNonEmpty(stringValue(quantResolution["trusted"]), opt(args, "quantization"))
			if resolvedQuantFormat == "" && strings.EqualFold(filepath.Ext(stringValue(quantResolution["modelPath"])), ".gguf") {
				resolvedQuantFormat = "gguf"
			}
		} else if agentCommand != "" {
			callModel = firstNonEmpty(opt(args, "served-model"), declaredModel, "external-agent")
			printStatus(args, "eval_model_detection_unavailable", map[string]any{"reason": "external agent did not provide --base-url, LM_STUDIO_BASE_URL, or LLAMA_CPP_BASE_URL"})
		}
	}
	maxTokens, err := intOption(args, 0, 0, "max-tokens")
	if err != nil {
		return err
	}
	commandTimeout, err := intOption(args, 120, 1, "command-timeout", "command-timeout-seconds")
	if err != nil {
		return err
	}
	concurrency, err := intOption(args, 1, 1, "concurrency")
	if err != nil {
		return err
	}
	cfg := terminalConfig{apiKey: opt(args, "model-api-key"), args: args, maxTokens: maxTokens, temperature: floatOption(args, "temperature", 0), topP: floatOption(args, "top-p", 1), commandTimeoutSec: commandTimeout, cleanupImages: hasFlag(args, "cleanup-images"), oracle: forceOracle || hasFlag(args, "oracle"), agentCommand: agentCommand, agentExecution: firstNonEmpty(opt(args, "agent-execution"), "host")}
	cfg.maxTurns, err = intOption(args, 0, 0, "max-turns")
	if err != nil {
		return err
	}
	cfg.agentTimeoutSec, err = intOption(args, 0, 0, "agent-timeout")
	if err != nil {
		return err
	}
	cfg.shellMode = firstNonEmpty(opt(args, "shell-mode"), "persistent")
	if cfg.shellMode != "persistent" && cfg.shellMode != "stateless" {
		return cliError{"invalid_option", "--shell-mode must be persistent or stateless", []string{"Pass --shell-mode persistent (default) or --shell-mode stateless."}, map[string]any{"shellMode": cfg.shellMode}}
	}
	if cfg.agentExecution != "host" && cfg.agentExecution != "container" && cfg.agentExecution != "routed-shell" {
		return cliError{"invalid_option", "--agent-execution must be host, container, or routed-shell", []string{"Use host for legacy external commands, container for agents launched inside the task container, or routed-shell for host agents whose shell is mechanically routed into the task container."}, map[string]any{"agentExecution": cfg.agentExecution}}
	}

	printInfo("terminal_eval_start", map[string]any{"dataset": firstNonEmpty(dataset, "local"), "tasks": len(bundles), "model": firstNonEmpty(callModel, "oracle"), "baseUrl": baseURL, "concurrency": concurrency, "oracle": cfg.oracle})
	results := runTerminalBundles(args, bundles, baseURL, callModel, cfg, concurrency)
	stats := shardStats{}
	errorCodes := map[string]string{}
	for _, result := range results {
		stats.totalLatencyMs += result.latencyMs
		if result.scored {
			stats.scored++
			if result.pass {
				stats.correct++
			}
		} else {
			stats.errors++
			if result.errCode != "" {
				errorCodes[result.instruction] = result.errCode
			}
		}
	}
	accuracy := 0.0
	if stats.scored > 0 {
		accuracy = float64(stats.correct) / float64(stats.scored)
	}
	avgLatency := int64(0)
	if len(results) > 0 {
		avgLatency = stats.totalLatencyMs / int64(len(results))
	}
	summary := map[string]any{"dataset": firstNonEmpty(dataset, "local"), "tasks": len(results), "correct": stats.correct, "scored": stats.scored, "errors": stats.errors, "accuracyPct": roundMetric(accuracy * 100), "avgLatencyMs": avgLatency, "quantization": resolvedQuant, "quantFormat": resolvedQuantFormat, "errorCodes": errorCodes}
	if out := opt(args, "out"); out != "" {
		records := make([]any, len(results))
		for i, r := range results {
			records[i] = map[string]any{"question_id": bundles[i].Task.ID, "pass": r.pass, "scored": r.scored, "error": r.errText, "errorCode": r.errCode, "latencyMs": r.latencyMs, "turns": r.turns, "question": bundles[i].Task.Instruction, "prompt": r.prompt, "response": r.transcript, "verifierOutput": r.verifierOutput}
		}
		if err := writeJSON(out, map[string]any{"summary": summary, "results": records}); err != nil {
			return err
		}
		printStatus(args, "terminal_results_written", map[string]any{"path": out})
	}
	if !hasFlag(args, "submit") {
		printInfo("terminal_eval_dry_run", summary)
		fmt.Println("Dry run only — nothing submitted.")
		if dataset != "" {
			if cfg.agentCommand != "" {
				fmt.Println("Publish with: lmx eval terminal run " + dataset + " --agent-cmd '<your-agent-command>' --model <hfId> --hardware hardware.json --submit")
			} else {
				fmt.Println("Publish with: lmx eval terminal run " + dataset + " --base-url " + rawBaseURL + " --model <hfId> --hardware hardware.json --submit")
			}
		}
		if stats.errors == len(results) {
			code := dominantTerminalErrorCode(errorCodes)
			return cliError{code, "Every terminal task errored before scoring.", []string{"Inspect --out results.json and the terminal_task_error events."}, map[string]any{"errors": errorCodes}}
		}
		return nil
	}
	key := apiKey(args)
	if key == "" {
		return missingAPIKey("--api-key or LMX_API_KEY is required for eval terminal --submit")
	}
	if declaredModel == "" {
		return cliError{"missing_model", "eval terminal --submit requires --model <HuggingFace model id>", []string{"Pass --model org/name so the submission records a real model.", "Use lmx model search <name> to find the canonical id."}, nil}
	}
	hardwarePath := opt(args, "hardware")
	if hardwarePath == "" {
		return cliError{"missing_hardware", "eval terminal --submit requires --hardware hardware.json", []string{"Run lmx hardware --out hardware.json and pass --hardware hardware.json."}, nil}
	}
	hardware, err := readJSON(hardwarePath)
	if err != nil {
		return err
	}
	submitResults := []any{}
	submitArtifacts := []any{}
	for i, r := range results {
		if !r.scored {
			continue
		}
		task := bundles[i].Task
		submitResults = append(submitResults, map[string]any{"question_id": task.ID, "pass": r.pass})
		submitArtifacts = append(submitArtifacts, map[string]any{"question_id": task.ID, "itemIndex": i, "promptHash": shortHash(task.ID + ":" + r.prompt), "question": task.Instruction, "prompt": r.prompt, "response": r.transcript + "\n\n# Verifier\n\n" + r.verifierOutput, "score": boolScore(r.pass), "testPassed": r.pass, "latencyMs": r.latencyMs})
	}
	if len(submitResults) == 0 {
		return cliError{"no_scored_questions", "Every terminal task failed to score, so there is nothing to submit.", []string{"Check Docker and the model endpoint.", "Inspect failures with --out results.json."}, map[string]any{"errors": errorCodes}}
	}
	submitArgs := argsWithTerminalBaseURL(args, rawBaseURL)
	hfID, modelResolution := resolveEvalModelID(submitArgs, declaredModel)
	protocol := terminalProtocolLabel(cfg)
	agentName := "lmx-terminus"
	if cfg.agentCommand != "" {
		agentName = firstNonEmpty(opt(args, "agent-name"), "external-agent")
	}
	runConfig := map[string]any{"accuracy": accuracy, "tasksRun": len(results), "errors": stats.errors, "avgLatencyMs": avgLatency, "protocol": protocol, "agent": agentName, "maxTurns": cfg.maxTurns, "concurrency": concurrency, "modelResolution": modelResolution, "quantizationResolution": quantResolution}
	if cfg.agentCommand != "" {
		runConfig["agentCommand"] = cfg.agentCommand
		runConfig["agentExecution"] = cfg.agentExecution
		runConfig["toolRouting"] = map[string]any{"shell": cfg.agentExecution, "workdir": "/app", "hostFilesystemVisible": cfg.agentExecution == "host"}
	}
	payload := map[string]any{"hfId": hfID, "modelRevision": firstNonEmpty(opt(args, "model-revision"), "main"), "hardware": normalizeHardwarePayload(hardware), "results": submitResults, "artifacts": submitArtifacts, "runnerVersion": firstNonEmpty(opt(args, "runner-version"), "localmaxxing-go terminal-agent"), "runConfig": runConfig}
	if shardIndex >= 1 {
		payload["shardIndex"] = shardIndex
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
	fields := map[string]any{"dataset": dataset, "submitted": len(submitResults), "accuracyPct": summary["accuracyPct"]}
	if obj := asObject(value); obj != nil {
		if agg := asObject(obj["aggregate"]); agg != nil {
			fields["pooledScore"] = agg["pooledScore"]
			fields["ciLower"] = agg["ciLower"]
			fields["ciUpper"] = agg["ciUpper"]
			fields["coverage"] = agg["shardsCovered"]
		}
		if run := asObject(obj["run"]); run != nil {
			fields["runId"] = run["id"]
			fields["status"] = run["status"]
		}
	}
	printInfo("terminal_eval_submitted", fields)
	return nil
}

func acquireTerminalBundles(args cliArgs, dataset, localDir string) ([]terminalBundle, string, int, error) {
	if localDir != "" {
		bundles, err := loadTerminalBundles(localDir)
		return bundles, "", -1, err
	}
	params := url.Values{}
	if s := opt(args, "shard"); s != "" {
		params.Set("shard", s)
	}
	if q := opt(args, "questions"); q != "" {
		params.Set("questions", q)
	}
	metaURL := apiURL(args) + "/api/evals/" + url.PathEscape(dataset) + "/shard"
	if enc := params.Encode(); enc != "" {
		metaURL += "?" + enc
	}
	meta, err := fetchJSON("GET", metaURL, apiKey(args), nil)
	if err != nil {
		return nil, "", -1, cliError{"manifest_fetch_failed", fmt.Sprintf("Could not fetch terminal dataset manifest: %v", err), []string{"Check the dataset slug and API URL.", "Confirm the dataset is approved and eval storage is configured."}, map[string]any{"url": metaURL, "error": err.Error()}}
	}
	shardIndex := -1
	if sh := asObject(asObject(meta)["shard"]); sh != nil {
		if f, ok := sh["shardIndex"].(float64); ok {
			shardIndex = int(f)
		}
	}
	downloadURL := stringValue(asObject(meta)["downloadUrl"])
	if downloadURL == "" {
		return nil, "", shardIndex, cliError{"manifest_fetch_failed", "Dataset shard response did not include downloadUrl.", []string{"Confirm the dataset is approved and eval storage is configured."}, meta}
	}
	items, err := fetchDatasetItems(downloadURL, "jsonl")
	if err != nil {
		return nil, "", shardIndex, cliError{"manifest_fetch_failed", fmt.Sprintf("Could not download terminal manifest JSONL: %v", err), []string{"Signed manifest URLs expire after 15 minutes; re-run the command.", "Check network access to the storage host."}, map[string]any{"downloadUrl": downloadURL}}
	}
	if q := opt(args, "questions"); q != "" {
		limit, err := strconv.Atoi(q)
		if err != nil || limit < 1 {
			return nil, "", shardIndex, cliError{"manifest_fetch_failed", "--questions must be a positive integer.", []string{"Pass --questions <n> to run the first n manifest rows, or use --task <id>[,<id>...] to filter task ids."}, map[string]any{"questions": q}}
		}
		if limit < len(items) {
			items = items[:limit]
		}
	}
	requested := parseStringSet(opt(args, "task"))
	tmp, err := os.MkdirTemp("", "lmx-terminal-bundles-*")
	if err != nil {
		return nil, "", shardIndex, err
	}
	bundles := []terminalBundle{}
	for _, row := range items {
		id := firstNonEmpty(stringValue(row["question_id"]), stringValue(row["questionId"]), stringValue(row["id"]))
		if len(requested) > 0 && !requested[id] {
			continue
		}
		key := stringValue(row["bundle_key"])
		if key == "" {
			key = stringValue(row["bundleKey"])
		}
		if key == "" {
			return nil, tmp, shardIndex, cliError{"manifest_fetch_failed", "Terminal manifest row is missing bundle_key.", []string{"Re-ingest the terminal dataset manifest."}, row}
		}
		bundleDir, err := downloadTerminalBundle(args, tmp, id, key, stringValue(row["sha256"]))
		if err != nil {
			return nil, tmp, shardIndex, err
		}
		loaded, err := loadSingleTerminalBundle(bundleDir)
		if err != nil {
			return nil, tmp, shardIndex, err
		}
		bundles = append(bundles, loaded)
	}
	return bundles, tmp, shardIndex, nil
}

func downloadTerminalBundle(args cliArgs, tmp, id, key, wantHash string) (string, error) {
	value, err := fetchJSON("GET", apiURL(args)+"/api/evals/storage/download-url?key="+url.QueryEscape(key), apiKey(args), nil)
	if err != nil {
		return "", cliError{"bundle_download_failed", fmt.Sprintf("Could not presign terminal bundle download: %v", err), []string{"Check that the dataset is approved and the bundle key exists."}, map[string]any{"taskId": id, "bundle_key": key, "error": err.Error()}}
	}
	downloadURL := stringValue(asObject(value)["downloadUrl"])
	if downloadURL == "" {
		return "", cliError{"bundle_download_failed", "Presign response did not include downloadUrl.", []string{"Check the LocalMaxxing API response."}, value}
	}
	res, err := http.Get(downloadURL)
	if err != nil {
		return "", cliError{"bundle_download_failed", fmt.Sprintf("Could not download terminal bundle: %v", err), []string{"Retry; signed URLs expire quickly."}, map[string]any{"taskId": id, "bundle_key": key}}
	}
	defer res.Body.Close()
	data, _ := io.ReadAll(res.Body)
	if res.StatusCode < 200 || res.StatusCode >= 300 {
		return "", cliError{"bundle_download_failed", fmt.Sprintf("Terminal bundle download returned %s", res.Status), []string{"Retry; signed URLs expire quickly."}, map[string]any{"taskId": id, "bundle_key": key, "body": truncateString(string(data), 4096)}}
	}
	if wantHash != "" {
		sum := sha256.Sum256(data)
		got := hex.EncodeToString(sum[:])
		if !strings.EqualFold(got, wantHash) {
			return "", cliError{"bundle_download_failed", "Terminal bundle sha256 did not match the manifest.", []string{"Re-ingest the dataset; the manifest and bundle object are inconsistent."}, map[string]any{"taskId": id, "bundle_key": key, "expected": wantHash, "actual": got}}
		}
	}
	bundleDir := filepath.Join(tmp, sanitizeDockerName(id))
	if err := os.MkdirAll(bundleDir, 0o755); err != nil {
		return "", err
	}
	if err := extractTarGz(bytes.NewReader(data), bundleDir); err != nil {
		return "", cliError{"bundle_download_failed", fmt.Sprintf("Could not extract terminal bundle tar.gz: %v", err), []string{"Re-ingest the dataset; the bundle should be a tar.gz of one task directory."}, map[string]any{"taskId": id, "bundle_key": key}}
	}
	return findExtractedBundleDir(bundleDir), nil
}

func loadTerminalBundles(dir string) ([]terminalBundle, error) {
	if _, err := os.Stat(filepath.Join(dir, "task.json")); err == nil {
		b, err := loadSingleTerminalBundle(dir)
		if err != nil {
			return nil, err
		}
		return []terminalBundle{b}, nil
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, cliError{"bundle_invalid", fmt.Sprintf("Could not read task bundle directory: %v", err), []string{"Check --task-dir exists and is readable."}, map[string]any{"taskDir": dir}}
	}
	var bundles []terminalBundle
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		candidate := filepath.Join(dir, entry.Name())
		if _, err := os.Stat(filepath.Join(candidate, "task.json")); err == nil {
			b, err := loadSingleTerminalBundle(candidate)
			if err != nil {
				return nil, err
			}
			bundles = append(bundles, b)
		}
	}
	sort.Slice(bundles, func(i, j int) bool { return bundles[i].Task.ID < bundles[j].Task.ID })
	if len(bundles) == 0 {
		return nil, cliError{"bundle_invalid", "No task.json files were found in --task-dir.", []string{"Point --task-dir at one bundle or a parent directory of bundles."}, map[string]any{"taskDir": dir}}
	}
	return bundles, nil
}

func loadSingleTerminalBundle(dir string) (terminalBundle, error) {
	data, err := os.ReadFile(filepath.Join(dir, "task.json"))
	if err != nil {
		return terminalBundle{}, cliError{"bundle_invalid", "Terminal bundle is missing task.json.", []string{"Run lmx eval terminal import first, or point --task-dir at an imported bundle."}, map[string]any{"bundleDir": dir, "error": err.Error()}}
	}
	var task terminalTask
	if err := json.Unmarshal(data, &task); err != nil {
		return terminalBundle{}, cliError{"bundle_invalid", fmt.Sprintf("Could not parse task.json: %v", err), []string{"Fix task.json or re-run the importer."}, map[string]any{"bundleDir": dir}}
	}
	if task.ID == "" || task.Instruction == "" {
		return terminalBundle{}, cliError{"bundle_invalid", "task.json must include id and instruction.", []string{"Re-run lmx eval terminal import."}, map[string]any{"bundleDir": dir}}
	}
	if _, err := os.Stat(filepath.Join(dir, "tests")); err != nil {
		return terminalBundle{}, cliError{"bundle_invalid", "Terminal bundle is missing tests/.", []string{"Copy harbor tests/ into the bundle or re-run the importer."}, map[string]any{"bundleDir": dir, "taskId": task.ID}}
	}
	return terminalBundle{Task: task, Dir: dir}, nil
}

func runTerminalBundles(args cliArgs, bundles []terminalBundle, baseURL, model string, cfg terminalConfig, concurrency int) []terminalTaskResult {
	results := make([]terminalTaskResult, len(bundles))
	jobs := make(chan int)
	var wg sync.WaitGroup
	for worker := 0; worker < concurrency; worker++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for idx := range jobs {
				b := bundles[idx]
				printStatus(args, "terminal_task_started", map[string]any{"taskId": b.Task.ID, "index": idx + 1, "total": len(bundles), "image": firstNonEmpty(b.Task.Image.Prebuilt, b.Task.Image.Dockerfile)})
				results[idx] = runTerminalTask(context.Background(), b.Task, b.Dir, baseURL, model, cfg)
				if results[idx].errCode != "" {
					printStatus(args, "terminal_task_error", map[string]any{"taskId": b.Task.ID, "code": results[idx].errCode, "detail": results[idx].errText})
				}
				printStatus(args, "terminal_task_done", map[string]any{"taskId": b.Task.ID, "pass": results[idx].pass, "scored": results[idx].scored, "turns": results[idx].turns, "latencyMs": results[idx].latencyMs})
			}
		}()
	}
	for i := range bundles {
		jobs <- i
	}
	close(jobs)
	wg.Wait()
	return results
}

func runTerminalTask(ctx context.Context, task terminalTask, bundleDir, baseURL, model string, cfg terminalConfig) terminalTaskResult {
	started := time.Now()
	result := terminalTaskResult{instruction: task.ID, prompt: terminalSystemPrompt}
	if err := dockerPreflight(); err != nil {
		result.errCode = "docker_unavailable"
		result.errText = err.Error()
		return result
	}
	imageStart := time.Now()
	imageRef, cleanupImage, err := resolveTerminalImage(ctx, task, bundleDir)
	if err != nil {
		result.errCode, result.errText = cliErrorCodeText(err)
		return result
	}
	printStatus(cfg.args, "terminal_image_resolved", map[string]any{"taskId": task.ID, "mode": imageMode(task.Image), "ms": time.Since(imageStart).Milliseconds()})
	containerName := "lmx-tb-" + sanitizeDockerName(task.ID) + "-" + randomHex(6)
	startArgs := []string{"run", "-d", "--rm", "--name", containerName}
	if task.Environment.CPUs > 0 {
		startArgs = append(startArgs, "--cpus", strconv.FormatFloat(task.Environment.CPUs, 'f', -1, 64))
	}
	if task.Environment.MemoryMb > 0 {
		startArgs = append(startArgs, "--memory", fmt.Sprintf("%dm", task.Environment.MemoryMb))
	}
	if task.Environment.GPUs > 0 {
		startArgs = append(startArgs, "--gpus", "all")
	}
	if task.Environment.Network == "no-network" {
		startArgs = append(startArgs, "--network", "none")
	}
	if task.Environment.Network == "allowlist" {
		printStatus(cfg.args, "terminal_network_degraded", map[string]any{"taskId": task.ID, "allowedHosts": strings.Join(task.Environment.AllowedHosts, ",")})
	}
	for k, v := range task.Environment.Env {
		startArgs = append(startArgs, "-e", k+"="+v)
	}
	startArgs = append(startArgs, imageRef, "sleep", "infinity")
	out, code, timedOut, runErr := runCommand(ctx, 60*time.Second, "docker", startArgs...)
	if runErr != nil || timedOut || code != 0 {
		result.errCode = "container_start_failed"
		result.errText = terminalCommandError("container_start_failed", "Could not start terminal task container.", "docker", startArgs, code, out, timedOut).Error()
		cleanupTerminalImage(cleanupImage, cfg.cleanupImages)
		return result
	}
	defer runCommand(context.Background(), 30*time.Second, "docker", "rm", "-f", containerName)
	defer cleanupTerminalImage(cleanupImage, cfg.cleanupImages)
	if cfg.oracle {
		if err := runOracleSolution(ctx, task, bundleDir, containerName, cfg); err != nil {
			result.errCode, result.errText = cliErrorCodeText(err)
			result.latencyMs = time.Since(started).Milliseconds()
			return result
		}
	} else if cfg.agentCommand != "" {
		transcript, err := runExternalTerminalAgent(ctx, task, bundleDir, containerName, baseURL, model, cfg)
		result.transcript = transcript
		result.prompt = "external agent command"
		if err != nil {
			result.errCode, result.errText = cliErrorCodeText(err)
		}
	} else if cfg.shellMode == "stateless" {
		turns, transcript, err := runTerminalAgentLoop(ctx, task, containerName, baseURL, model, cfg)
		result.turns = turns
		result.transcript = transcript
		if err != nil {
			result.errCode, result.errText = cliErrorCodeText(err)
			result.latencyMs = time.Since(started).Milliseconds()
			return result
		}
	} else {
		result.prompt = terminalSessionSystemPrompt
		turns, transcript, err := runTerminalAgentLoopSession(ctx, task, containerName, baseURL, model, cfg)
		result.turns = turns
		result.transcript = transcript
		if err != nil {
			result.errCode, result.errText = cliErrorCodeText(err)
			result.latencyMs = time.Since(started).Milliseconds()
			return result
		}
	}
	pass, verifierOutput, err := runTerminalVerifier(ctx, task, bundleDir, containerName, cfg)
	result.pass = pass
	result.scored = true
	result.verifierOutput = verifierOutput
	if err != nil {
		result.pass = false
	}
	result.latencyMs = time.Since(started).Milliseconds()
	return result
}

func resolveTerminalImage(ctx context.Context, task terminalTask, bundleDir string) (string, string, error) {
	if task.Image.Prebuilt != "" {
		out, code, timedOut, err := runCommand(ctx, 15*time.Minute, "docker", "pull", task.Image.Prebuilt)
		if err != nil || timedOut || code != 0 {
			return "", "", terminalCommandError("image_pull_failed", "Could not pull terminal task image.", "docker", []string{"pull", task.Image.Prebuilt}, code, out, timedOut)
		}
		return task.Image.Prebuilt, "", nil
	}
	ctxPath := filepath.Join(bundleDir, task.Image.Context)
	tag := "lmx-tb-" + sanitizeDockerName(task.ID) + "-" + shortHash(bundleDir)
	timeout := time.Duration(firstPositive(task.Image.BuildTimeoutSec, 600)) * time.Second
	out, code, timedOut, err := runCommand(ctx, timeout, "docker", "build", "-t", tag, ctxPath)
	if err != nil || timedOut || code != 0 {
		return "", "", terminalCommandError("image_build_failed", "Could not build terminal task image.", "docker", []string{"build", "-t", tag, ctxPath}, code, out, timedOut)
	}
	return tag, tag, nil
}

func runExternalTerminalAgent(ctx context.Context, task terminalTask, bundleDir, containerName, baseURL, model string, cfg terminalConfig) (string, error) {
	tmp, err := os.MkdirTemp("", "lmx-terminal-agent-*")
	if err != nil {
		return "", cliError{"command_exec_failed", "Could not create external agent workspace.", []string{"Check temporary directory permissions."}, map[string]any{"taskId": task.ID, "error": err.Error()}}
	}
	defer os.RemoveAll(tmp)

	instructionFile := filepath.Join(tmp, "instruction.txt")
	traceDir := filepath.Join(tmp, "traces")
	if err := os.MkdirAll(traceDir, 0o700); err != nil {
		return "", cliError{"command_exec_failed", "Could not create external agent trace directory.", []string{"Check temporary directory permissions."}, map[string]any{"taskId": task.ID, "error": err.Error()}}
	}
	workdir := "/app"
	shellCommand := filepath.Join(tmp, "container-shell")
	shellScript := "#!/usr/bin/env bash\nset -euo pipefail\nif [ \"$#\" -eq 0 ]; then\n  exec docker exec -i -w " + shellQuote(workdir) + " " + shellQuote(containerName) + " bash -l\nfi\nexec docker exec -i -w " + shellQuote(workdir) + " " + shellQuote(containerName) + " bash -lc \"$*\"\n"
	if err := os.WriteFile(shellCommand, []byte(shellScript), 0o700); err != nil {
		return "", cliError{"command_exec_failed", "Could not write routed shell helper.", []string{"Check temporary directory permissions."}, map[string]any{"taskId": task.ID, "error": err.Error()}}
	}

	if err := os.WriteFile(instructionFile, []byte(task.Instruction), 0o600); err != nil {
		return "", cliError{"command_exec_failed", "Could not write external agent instruction file.", []string{"Check temporary directory permissions."}, map[string]any{"taskId": task.ID, "error": err.Error()}}
	}

	env := []string{
		"LMX_TERMINAL_TASK_ID=" + task.ID,
		"LMX_TERMINAL_CONTAINER=" + containerName,
		"LMX_TERMINAL_BUNDLE_DIR=" + bundleDir,
		"LMX_TERMINAL_TASK_DIR=" + bundleDir,
		"LMX_TERMINAL_TASK_JSON=" + filepath.Join(bundleDir, "task.json"),
		"LMX_TERMINAL_INSTRUCTION_FILE=" + instructionFile,
		"LMX_TERMINAL_BASE_URL=" + baseURL,
		"LMX_TERMINAL_CONTAINER_BASE_URL=" + firstNonEmpty(opt(cfg.args, "container-base-url"), "http://172.17.0.1:8080"),
		"LMX_TERMINAL_MODEL=" + model,
		"LMX_TERMINAL_WORKDIR=" + workdir,
		"LMX_TERMINAL_TRACE_DIR=" + traceDir,
		"LMX_TERMINAL_EXECUTION_MODE=" + cfg.agentExecution,
		"LMX_TERMINAL_SHELL_COMMAND=" + shellCommand,
	}
	timeout := time.Duration(firstPositive(cfg.agentTimeoutSec, task.Agent.TimeoutSec, 900)) * time.Second
	printStatus(cfg.args, "terminal_external_agent_started", map[string]any{"taskId": task.ID, "command": truncateString(cfg.agentCommand, 240), "execution": cfg.agentExecution})
	out, code, timedOut, runErr := runHostCommandWithEnv(ctx, timeout, env, cfg.agentCommand)
	transcript := "$ " + cfg.agentCommand + "\n" + out + "\n[exit=" + strconv.Itoa(code) + "]\n"
	if traceText := externalAgentTraceText(traceDir); traceText != "" {
		transcript += "\n\n# External agent trace directory\n\n" + traceText
	}
	printStatus(cfg.args, "terminal_external_agent_done", map[string]any{"taskId": task.ID, "exitCode": code, "timedOut": timedOut, "execution": cfg.agentExecution})
	if runErr != nil || timedOut || code != 0 {
		return transcript, terminalCommandError("command_exec_failed", "External terminal agent command failed.", "bash", []string{"-lc", cfg.agentCommand}, code, out, timedOut)
	}
	return transcript, nil
}

func runHostCommandWithEnv(ctx context.Context, timeout time.Duration, env []string, command string) (string, int, bool, error) {
	cctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	cmd := exec.CommandContext(cctx, "bash", "-lc", command)
	configureCommandProcessGroup(cmd)
	cmd.Env = append(os.Environ(), env...)
	var output bytes.Buffer
	cmd.Stdout = &output
	cmd.Stderr = &output
	err := cmd.Run()
	if cctx.Err() == context.DeadlineExceeded {
		killCommandProcessGroup(cmd)
		return output.String(), 124, true, err
	}
	code := 0
	if err != nil {
		code = 1
		if exitErr, ok := err.(*exec.ExitError); ok {
			code = exitErr.ExitCode()
		}
	}
	return output.String(), code, false, err
}

// terminalTraceHeader records one model turn (reasoning + full reply) into the
// artifact transcript. Every turn is captured — including non-conforming and
// completion replies — so submitted localmaxxing artifacts hold the complete
// trace rather than only the turns that produced a command.
func terminalTraceHeader(b *strings.Builder, turn int, reasoning, content string) {
	b.WriteString("# Turn " + strconv.Itoa(turn) + "\n")
	if reasoning != "" {
		b.WriteString("## Reasoning\n" + reasoning + "\n")
	}
	b.WriteString("## Assistant\n" + content + "\n")
}

func runTerminalAgentLoop(ctx context.Context, task terminalTask, containerName, baseURL, model string, cfg terminalConfig) (int, string, error) {
	messages := []map[string]any{{"role": "system", "content": terminalSystemPrompt}, {"role": "user", "content": task.Instruction}}
	maxTurns := firstPositive(cfg.maxTurns, task.Agent.MaxTurns, 50)
	timeoutSec := firstPositive(cfg.agentTimeoutSec, task.Agent.TimeoutSec, 900)
	deadline := time.Now().Add(time.Duration(timeoutSec) * time.Second)
	var transcript strings.Builder
	nonConforming := 0
	for turn := 1; turn <= maxTurns && time.Now().Before(deadline); turn++ {
		content, reasoning, err := callOpenAIChatMessages(baseURL, model, messages, cfg.apiKey, cfg.maxTokens, cfg.temperature, cfg.topP, nil)
		if err != nil {
			return turn - 1, transcript.String(), cliError{"model_call_failed", "Model call failed during terminal agent loop.", []string{"Check --base-url, --model-api-key, and that the OpenAI-compatible server is running."}, map[string]any{"taskId": task.ID, "error": err.Error()}}
		}
		cmdText, found := extractBashCommand(content)
		if !found {
			if strings.Contains(content, "TASK_COMPLETE") {
				terminalTraceHeader(&transcript, turn, reasoning, content)
				transcript.WriteString("## Note\nModel signaled TASK_COMPLETE.\n")
				return turn - 1, transcript.String(), nil
			}
			nonConforming++
			terminalTraceHeader(&transcript, turn, reasoning, content)
			transcript.WriteString("## Note\nNo bash block found; asked the model to emit one command or TASK_COMPLETE.\n")
			messages = append(messages, map[string]any{"role": "assistant", "content": content}, map[string]any{"role": "user", "content": "Reply with one ```bash fenced block or TASK_COMPLETE."})
			if nonConforming >= 2 {
				return turn - 1, transcript.String(), nil
			}
			continue
		}
		nonConforming = 0
		messages = append(messages, map[string]any{"role": "assistant", "content": content})
		terminalTraceHeader(&transcript, turn, reasoning, content)
		transcript.WriteString("## Command\n$ " + cmdText + "\n")
		out, code, timedOut, _ := runCommand(ctx, time.Duration(firstPositive(cfg.commandTimeoutSec, 120))*time.Second, "docker", "exec", containerName, "bash", "-lc", cmdText)
		if timedOut {
			out += "\n[command timed out]"
			code = 124
		}
		shown := truncateString(out, 8192)
		transcript.WriteString(shown + "\n[exit=" + strconv.Itoa(code) + "]\n")
		messages = append(messages, map[string]any{"role": "user", "content": shown + "\n[exit=" + strconv.Itoa(code) + "]"})
		printStatus(cfg.args, "terminal_turn", map[string]any{"taskId": task.ID, "turn": turn, "exitCode": code, "cmdPreview": truncateString(strings.ReplaceAll(cmdText, "\n", " "), 160)})
	}
	return maxTurns, transcript.String(), nil
}

type terminalShell struct {
	containerName string
	nonce         string
	cmd           *exec.Cmd
	stdin         io.WriteCloser
	reader        *bufio.Reader
	pr            *os.File
}

func startTerminalShell(containerName string) (*terminalShell, error) {
	s := &terminalShell{containerName: containerName, nonce: randomHex(8)}
	if err := s.start(); err != nil {
		return nil, err
	}
	return s, nil
}

func (s *terminalShell) start() error {
	// No context timeout on the shell process itself; per-command timeouts are
	// enforced in exec(). The shell lives for the whole agent loop.
	cmd := exec.Command("docker", "exec", "-i", s.containerName, "bash", "-l")
	configureCommandProcessGroup(cmd)
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return err
	}
	pr, pw, err := os.Pipe()
	if err != nil {
		return err
	}
	cmd.Stdout = pw
	cmd.Stderr = pw
	if err := cmd.Start(); err != nil {
		pw.Close()
		pr.Close()
		return err
	}
	pw.Close() // parent keeps only the read end; child holds the write end
	s.cmd = cmd
	s.stdin = stdin
	s.pr = pr
	s.reader = bufio.NewReader(pr)
	return nil
}

func (s *terminalShell) restart() error {
	s.close()
	s.nonce = randomHex(8)
	return s.start()
}

func (s *terminalShell) close() {
	if s.stdin != nil {
		s.stdin.Close()
	}
	if s.cmd != nil {
		killCommandProcessGroup(s.cmd)
		_ = s.cmd.Wait()
	}
	if s.pr != nil {
		s.pr.Close()
	}
}

// exec runs one command in the persistent shell. timedOut means the per-command
// budget elapsed; restarted means the shell was rebuilt (state reset) due to
// timeout or the shell dying (e.g. the command ran `exit` or crashed bash).
func (s *terminalShell) exec(command string, timeout time.Duration) (output string, exitCode int, timedOut bool, restarted bool) {
	marker := "__LMX_END_" + s.nonce + "__"
	payload := command + "\nprintf '\\n" + marker + "%d__\\n' \"$?\"\n"
	if _, err := io.WriteString(s.stdin, payload); err != nil {
		_ = s.restart()
		return "[shell write failed; session restarted]", 1, false, true
	}
	type readResult struct {
		out  string
		code int
		eof  bool
	}
	done := make(chan readResult, 1)
	go func() {
		var buf strings.Builder
		for {
			line, err := s.reader.ReadString('\n')
			if idx := strings.Index(line, marker); idx >= 0 {
				rest := line[idx+len(marker):]
				rest = strings.TrimSuffix(strings.TrimSpace(rest), "__")
				code, _ := strconv.Atoi(strings.TrimSuffix(rest, "__"))
				done <- readResult{out: buf.String(), code: code}
				return
			}
			buf.WriteString(line)
			if err != nil {
				done <- readResult{out: buf.String(), code: 1, eof: true}
				return
			}
		}
	}()
	select {
	case r := <-done:
		if r.eof {
			_ = s.restart()
			return r.out + "\n[shell ended; session restarted, state reset]", r.code, false, true
		}
		return r.out, r.code, false, false
	case <-time.After(timeout):
		_ = s.restart()
		return "[command timed out after " + timeout.String() + "; session restarted, state reset]", 124, true, true
	}
}

func runTerminalAgentLoopSession(ctx context.Context, task terminalTask, containerName, baseURL, model string, cfg terminalConfig) (int, string, error) {
	shell, err := startTerminalShell(containerName)
	if err != nil {
		return 0, "", cliError{"command_exec_failed", "Could not open a persistent shell in the task container.", []string{"Check Docker and that the task image provides /bin/bash.", "Or rerun with --shell-mode stateless."}, map[string]any{"taskId": task.ID, "error": err.Error()}}
	}
	defer shell.close()

	messages := []map[string]any{{"role": "system", "content": terminalSessionSystemPrompt}, {"role": "user", "content": task.Instruction}}
	maxTurns := firstPositive(cfg.maxTurns, task.Agent.MaxTurns, 50)
	timeoutSec := firstPositive(cfg.agentTimeoutSec, task.Agent.TimeoutSec, 900)
	deadline := time.Now().Add(time.Duration(timeoutSec) * time.Second)
	cmdTimeout := time.Duration(firstPositive(cfg.commandTimeoutSec, 120)) * time.Second
	var transcript strings.Builder
	nonConforming := 0
	for turn := 1; turn <= maxTurns && time.Now().Before(deadline); turn++ {
		content, reasoning, err := callOpenAIChatMessages(baseURL, model, messages, cfg.apiKey, cfg.maxTokens, cfg.temperature, cfg.topP, nil)
		if err != nil {
			return turn - 1, transcript.String(), cliError{"model_call_failed", "Model call failed during terminal agent loop.", []string{"Check --base-url, --model-api-key, and that the OpenAI-compatible server is running."}, map[string]any{"taskId": task.ID, "error": err.Error()}}
		}
		cmdText, found := extractBashCommand(content)
		if !found {
			if strings.Contains(content, "TASK_COMPLETE") {
				terminalTraceHeader(&transcript, turn, reasoning, content)
				transcript.WriteString("## Note\nModel signaled TASK_COMPLETE.\n")
				return turn - 1, transcript.String(), nil
			}
			nonConforming++
			terminalTraceHeader(&transcript, turn, reasoning, content)
			transcript.WriteString("## Note\nNo bash block found; asked the model to emit one command or TASK_COMPLETE.\n")
			messages = append(messages, map[string]any{"role": "assistant", "content": content}, map[string]any{"role": "user", "content": "Reply with one ```bash fenced block or TASK_COMPLETE."})
			if nonConforming >= 2 {
				return turn - 1, transcript.String(), nil
			}
			continue
		}
		nonConforming = 0
		messages = append(messages, map[string]any{"role": "assistant", "content": content})
		terminalTraceHeader(&transcript, turn, reasoning, content)
		transcript.WriteString("## Command\n$ " + cmdText + "\n")
		out, code, timedOut, restarted := shell.exec(cmdText, cmdTimeout)
		if timedOut {
			out += "\n[command timed out]"
		}
		shown := truncateString(out, 8192)
		transcript.WriteString(shown + "\n[exit=" + strconv.Itoa(code) + "]\n")
		messages = append(messages, map[string]any{"role": "user", "content": shown + "\n[exit=" + strconv.Itoa(code) + "]"})
		fields := map[string]any{"taskId": task.ID, "turn": turn, "exitCode": code, "cmdPreview": truncateString(strings.ReplaceAll(cmdText, "\n", " "), 160)}
		if restarted {
			fields["shellRestarted"] = true
		}
		printStatus(cfg.args, "terminal_turn", fields)
	}
	return maxTurns, transcript.String(), nil
}

func runOracleSolution(ctx context.Context, task terminalTask, bundleDir, containerName string, cfg terminalConfig) error {
	if _, err := os.Stat(filepath.Join(bundleDir, "solution", "solve.sh")); err != nil {
		return cliError{"bundle_invalid", "Oracle mode requires solution/solve.sh.", []string{"Import a harbor task with solution/ or do not use --oracle."}, map[string]any{"taskId": task.ID, "bundleDir": bundleDir}}
	}
	out, code, timedOut, err := runCommand(ctx, 120*time.Second, "docker", "cp", filepath.Join(bundleDir, "solution")+"/.", containerName+":/solution")
	if err != nil || timedOut || code != 0 {
		return terminalCommandError("command_exec_failed", "Could not copy oracle solution into container.", "docker", []string{"cp", filepath.Join(bundleDir, "solution") + "/.", containerName + ":/solution"}, code, out, timedOut)
	}
	out, code, timedOut, err = runCommand(ctx, time.Duration(firstPositive(cfg.commandTimeoutSec, task.Agent.TimeoutSec, 900))*time.Second, "docker", "exec", containerName, "bash", "/solution/solve.sh")
	if err != nil || timedOut || code != 0 {
		return terminalCommandError("command_exec_failed", "Oracle solution failed in the task container.", "docker", []string{"exec", containerName, "bash", "/solution/solve.sh"}, code, out, timedOut)
	}
	return nil
}

func runTerminalVerifier(ctx context.Context, task terminalTask, bundleDir, containerName string, cfg terminalConfig) (bool, string, error) {
	_, _, _, _ = runCommand(ctx, 30*time.Second, "docker", "exec", containerName, "mkdir", "-p", "/logs/verifier")
	out, code, timedOut, err := runCommand(ctx, 120*time.Second, "docker", "cp", filepath.Join(bundleDir, "tests")+"/.", containerName+":/tests")
	if err != nil || timedOut || code != 0 {
		return false, out, terminalCommandError("verifier_failed", "Could not copy verifier tests into the task container.", "docker", []string{"cp", filepath.Join(bundleDir, "tests") + "/.", containerName + ":/tests"}, code, out, timedOut)
	}
	cmdArgs := []string{"exec"}
	if task.Verifier.User != "" {
		cmdArgs = append(cmdArgs, "--user", task.Verifier.User)
	}
	for k, v := range task.Verifier.Env {
		cmdArgs = append(cmdArgs, "-e", k+"="+v)
	}
	cmdArgs = append(cmdArgs, containerName, "bash", "-lc", task.Verifier.Command)
	out, code, timedOut, _ = runCommand(ctx, time.Duration(firstPositive(task.Verifier.TimeoutSec, 900))*time.Second, "docker", cmdArgs...)
	rewardPath := firstNonEmpty(task.Verifier.RewardFile, "/logs/verifier/reward.txt")
	reward, rewardCode, _, _ := runCommand(ctx, 30*time.Second, "docker", "exec", containerName, "cat", rewardPath)
	trimmed := strings.TrimSpace(reward)
	pass := false
	if rewardCode == 0 {
		pass = trimmed == "1"
	} else {
		pass = code == 0 && !timedOut
	}
	printStatus(cfg.args, "terminal_verifier", map[string]any{"taskId": task.ID, "reward": trimmed, "exitCode": code})
	if timedOut || code != 0 {
		return pass, out + "\n" + reward, cliError{"verifier_failed", "Verifier completed with a failing exit status or timed out.", []string{"Inspect the task transcript and verifier output in --out results.json."}, map[string]any{"taskId": task.ID, "exitCode": code, "timedOut": timedOut, "output": truncateString(out, 4096)}}
	}
	return pass, out + "\n" + reward, nil
}

func callOpenAIChatMessages(baseURL, model string, messages []map[string]any, apiKey string, maxTokens int, temperature, topP float64, stop []string) (content, reasoning string, err error) {
	body := map[string]any{"model": model, "messages": messages, "temperature": temperature, "top_p": topP}
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
		return "", "", cliError{"model_call_failed", fmt.Sprintf("OpenAI-compatible server returned %s", res.Status), []string{"Check --base-url and --model-api-key."}, map[string]any{"status": res.Status, "body": truncateString(strings.TrimSpace(string(text)), 4096)}}
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

func extractBashCommand(reply string) (cmd string, found bool) {
	patterns := []string{"(?s)```(?:bash|sh)\\s*\\n(.*?)\\n```", "(?s)```\\s*\\n(.*?)\\n```"}
	for _, pattern := range patterns {
		re := regexp.MustCompile(pattern)
		match := re.FindStringSubmatch(reply)
		if len(match) > 1 {
			return strings.TrimSpace(match[1]), true
		}
	}
	return "", false
}

func dockerPreflight() error {
	out, code, timedOut, err := runCommand(context.Background(), 15*time.Second, "docker", "version")
	if err != nil || timedOut || code != 0 {
		return terminalCommandError("docker_unavailable", "Docker is not available.", "docker", []string{"version"}, code, out, timedOut)
	}
	return nil
}

func runCommand(ctx context.Context, timeout time.Duration, name string, args ...string) (string, int, bool, error) {
	cctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	cmd := exec.CommandContext(cctx, name, args...)
	configureCommandProcessGroup(cmd)
	var output bytes.Buffer
	cmd.Stdout = &output
	cmd.Stderr = &output
	err := cmd.Run()
	if cctx.Err() == context.DeadlineExceeded {
		killCommandProcessGroup(cmd)
		return output.String(), 124, true, err
	}
	code := 0
	if err != nil {
		code = 1
		if exitErr, ok := err.(*exec.ExitError); ok {
			code = exitErr.ExitCode()
		}
	}
	return output.String(), code, false, err
}

func terminalCommandError(code, message, cmd string, args []string, exitCode int, output string, timedOut bool) error {
	return cliError{code, message, []string{terminalHint(code)}, map[string]any{"command": strings.TrimSpace(cmd + " " + strings.Join(args, " ")), "exitCode": exitCode, "timedOut": timedOut, "output": truncateString(output, 4096)}}
}

func terminalHint(code string) string {
	switch code {
	case "docker_unavailable":
		return "Install Docker, start the Docker daemon, and ensure this user can run docker commands."
	case "image_build_failed":
		return "Inspect the Dockerfile and build context in the task environment/."
	case "image_pull_failed":
		return "Check network access and that the task image reference exists."
	case "container_start_failed":
		return "Check Docker resource limits, image architecture, and runtime availability."
	case "command_exec_failed":
		return "Inspect the failing command output and task bundle contents."
	case "verifier_failed":
		return "Inspect tests/test.sh and the verifier output."
	}
	return "Inspect command details and retry."
}

func argsWithTerminalBaseURL(args cliArgs, rawBaseURL string) cliArgs {
	if rawBaseURL == "" || opt(args, "base-url") != "" {
		return args
	}
	copied := cliArgs{
		positional: append([]string(nil), args.positional...),
		opts:       map[string]string{},
		flags:      map[string]bool{},
	}
	for k, v := range args.opts {
		copied.opts[k] = v
	}
	for k, v := range args.flags {
		copied.flags[k] = v
	}
	copied.opts["base-url"] = rawBaseURL
	return copied
}

func cliErrorCodeText(err error) (string, string) {
	if err == nil {
		return "", ""
	}
	var ce cliError
	if errors.As(err, &ce) {
		if ce.Details != nil {
			data, _ := json.Marshal(ce.Details)
			return ce.Code, ce.Message + " Details: " + string(data)
		}
		return ce.Code, ce.Message
	}
	return "unexpected_error", err.Error()
}

func dominantTerminalErrorCode(codes map[string]string) string {
	if len(codes) == 0 {
		return "no_scored_questions"
	}
	first := ""
	for _, code := range codes {
		if first == "" {
			first = code
			continue
		}
		if code != first {
			return "no_scored_questions"
		}
	}
	return first
}

func copyDir(src, dst string) error {
	return filepath.WalkDir(src, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, err := filepath.Rel(src, path)
		if err != nil {
			return err
		}
		target := filepath.Join(dst, rel)
		if d.IsDir() {
			return os.MkdirAll(target, 0o755)
		}
		info, err := d.Info()
		if err != nil {
			return err
		}
		in, err := os.Open(path)
		if err != nil {
			return err
		}
		defer in.Close()
		out, err := os.OpenFile(target, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, info.Mode())
		if err != nil {
			return err
		}
		defer out.Close()
		_, err = io.Copy(out, in)
		return err
	})
}

func extractTarGz(r io.Reader, dest string) error {
	gz, err := gzip.NewReader(r)
	if err != nil {
		return err
	}
	defer gz.Close()
	tr := tar.NewReader(gz)
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
		clean := filepath.Clean(hdr.Name)
		if clean == "." || strings.HasPrefix(clean, "../") || filepath.IsAbs(clean) {
			return fmt.Errorf("unsafe path %q", hdr.Name)
		}
		target := filepath.Join(dest, clean)
		if hdr.FileInfo().IsDir() {
			if err := os.MkdirAll(target, 0o755); err != nil {
				return err
			}
			continue
		}
		if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
			return err
		}
		out, err := os.OpenFile(target, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, hdr.FileInfo().Mode())
		if err != nil {
			return err
		}
		_, copyErr := io.Copy(out, tr)
		closeErr := out.Close()
		if copyErr != nil {
			return copyErr
		}
		if closeErr != nil {
			return closeErr
		}
	}
	return nil
}

func findExtractedBundleDir(root string) string {
	if _, err := os.Stat(filepath.Join(root, "task.json")); err == nil {
		return root
	}
	entries, _ := os.ReadDir(root)
	for _, entry := range entries {
		if entry.IsDir() {
			candidate := filepath.Join(root, entry.Name())
			if _, err := os.Stat(filepath.Join(candidate, "task.json")); err == nil {
				return candidate
			}
		}
	}
	return root
}

func parseStringSet(value string) map[string]bool {
	out := map[string]bool{}
	for _, part := range strings.Split(value, ",") {
		part = strings.TrimSpace(part)
		if part != "" {
			out[part] = true
		}
	}
	return out
}

func normalizeTerminalNetwork(value string) string {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "public", "bridge", "internet":
		return "public"
	case "no-network", "none", "disabled":
		return "no-network"
	case "allowlist", "allow-list":
		return "allowlist"
	default:
		return ""
	}
}

func nonNilStringMap(in map[string]string) map[string]string {
	if in == nil {
		return map[string]string{}
	}
	return in
}

func externalAgentTraceText(traceDir string) string {
	paths := []string{}
	_ = filepath.WalkDir(traceDir, func(path string, d os.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil
		}
		paths = append(paths, path)
		return nil
	})
	sort.Strings(paths)
	var b strings.Builder
	total := 0
	const maxTotal = 4 * 1024 * 1024
	const maxFile = 1024 * 1024
	for _, path := range paths {
		if total >= maxTotal {
			b.WriteString("\n...[trace directory truncated]\n")
			break
		}
		data, err := os.ReadFile(path)
		if err != nil || len(data) == 0 || bytes.IndexByte(data, 0) >= 0 {
			continue
		}
		rel, _ := filepath.Rel(traceDir, path)
		chunk := data
		if len(chunk) > maxFile {
			chunk = chunk[:maxFile]
		}
		if total+len(chunk) > maxTotal {
			chunk = chunk[:maxTotal-total]
		}
		b.WriteString("## ")
		b.WriteString(rel)
		b.WriteString("\n\n```text\n")
		b.Write(chunk)
		if len(data) > len(chunk) {
			b.WriteString(fmt.Sprintf("\n...[truncated %d bytes]", len(data)-len(chunk)))
		}
		b.WriteString("\n```\n\n")
		total += len(chunk)
	}
	return b.String()
}
func firstPositive(values ...int) int {
	for _, v := range values {
		if v > 0 {
			return v
		}
	}
	return 0
}
func firstPositiveFloat(values ...float64) float64 {
	for _, v := range values {
		if v > 0 {
			return v
		}
	}
	return 0
}
func truncateString(value string, max int) string {
	if max <= 0 || len(value) <= max {
		return value
	}
	return value[:max] + fmt.Sprintf("\n...[truncated %d bytes]", len(value)-max)
}
func sanitizeDockerName(value string) string {
	re := regexp.MustCompile(`[^a-zA-Z0-9_.-]+`)
	value = strings.Trim(re.ReplaceAllString(value, "-"), "-.")
	if value == "" {
		return "task"
	}
	return value
}
func randomHex(n int) string { b := make([]byte, n); _, _ = rand.Read(b); return hex.EncodeToString(b) }
func shortHash(value string) string {
	sum := sha256.Sum256([]byte(value))
	return hex.EncodeToString(sum[:])[:12]
}
func imageMode(img terminalImage) string {
	if img.Prebuilt != "" {
		return "prebuilt"
	}
	return "dockerfile"
}
func terminalProtocolLabel(cfg terminalConfig) string {
	if cfg.agentCommand != "" {
		return "external-command/" + firstNonEmpty(cfg.agentExecution, "host")
	}
	if cfg.shellMode == "stateless" {
		return "react-bash"
	}
	return "react-shell"
}
func cleanupTerminalImage(image string, enabled bool) {
	if enabled && image != "" {
		_, _, _, _ = runCommand(context.Background(), 2*time.Minute, "docker", "rmi", image)
	}
}

func terminalResultFromBundle(i int, b terminalBundle, r terminalTaskResult) shardItemResult {
	return shardItemResult{questionID: b.Task.ID, itemIndex: i, pass: r.pass, scored: r.scored, errText: r.errText, question: b.Task.Instruction, prompt: r.prompt, response: r.transcript, reasoning: r.verifierOutput, latencyMs: r.latencyMs}
}

var _ = bufio.ErrFinalToken
