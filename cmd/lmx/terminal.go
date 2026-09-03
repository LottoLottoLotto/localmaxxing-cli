package main

import (
	"archive/tar"
	"bufio"
	"bytes"
	"compress/gzip"
	"context"
	"crypto/rand"
	"crypto/sha256"
	_ "embed"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"html"
	"io"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"github.com/BurntSushi/toml"
)

const terminalSystemPrompt = "You control a Linux terminal. Each reply MUST contain exactly one ```bash fenced block containing one or more non-interactive shell commands, which will be executed; stdout/stderr/exit code are returned. Prefer batching related inspection/edit/test commands instead of spending one model turn per tiny command. When the task is complete, reply with the single token TASK_COMPLETE and no code block. If you need Python/Ruby/Node/etc., run it from bash with a heredoc (for example: python3 <<'PY' ... PY). Avoid dumping huge files; inspect with head/tail/grep/scripts. Bound password crackers and deliberately long-running commands yourself with timeout, but do not prematurely cap package installs, builds, or tests unless they are clearly stuck."

const terminalSessionSystemPrompt = "You control a persistent Linux shell session inside a container. State persists across replies: your working directory, environment variables, and background jobs carry over from one command block to the next. Each reply MUST contain exactly one ```bash fenced block containing one or more non-interactive shell commands, which are executed in that same shell; stdout/stderr and exit code are returned. Prefer batching related inspection/edit/test commands instead of spending one model turn per tiny command. When the task is complete, reply with the single token TASK_COMPLETE and no code block. If you need Python/Ruby/Node/etc., run it from bash with a heredoc (for example: python3 <<'PY' ... PY). Avoid dumping huge files; inspect with head/tail/grep/scripts. Bound password crackers and deliberately long-running commands yourself with timeout, but do not prematurely cap package installs, builds, or tests unless they are clearly stuck. Never run foreground servers; start them in the background and verify them."

const terminalNativeSystemPrompt = "You control a Linux terminal through the available tools read_file, edit_file, write_file, and terminal. Use read_file with a path argument to inspect files. Prefer edit_file with path, old_text, and new_text arguments for a unique surgical replacement. Use write_file with path and content arguments only when replacing a complete file is appropriate. Use terminal with a command argument for tests, directory listings, and other non-interactive shell commands. Do not emit bash code blocks and do not call any unlisted tool. Keep explanations outside tool arguments. Tool results, including stdout, stderr, and exit code, are returned after each call. Prefer completing the next concrete action instead of repeatedly reading an unchanged file. When the task is complete, reply with the single token TASK_COMPLETE and do not call a tool. Never run foreground servers; start them in the background and verify them."

const terminalNativeSessionSystemPrompt = "You control a persistent Linux shell session inside a container through the available tools read_file, edit_file, write_file, and terminal. State persists across tool calls: your working directory, environment variables, and background jobs carry over. Use read_file with a path argument to inspect files. Prefer edit_file with path, old_text, and new_text arguments for a unique surgical replacement. Use write_file with path and content arguments only when replacing a complete file is appropriate. Use terminal with a command argument for tests, directory listings, and other non-interactive shell commands. Do not emit bash code blocks and do not call any unlisted tool. Keep explanations outside tool arguments. Tool results, including stdout, stderr, and exit code, are returned after each call. Prefer completing the next concrete action instead of repeatedly reading an unchanged file. When the task is complete, reply with the single token TASK_COMPLETE and do not call a tool. Never run foreground servers; start them in the background and verify them."

const terminalNativeContinuePrompt = "Your previous reply did not call an available tool. Continue with read_file(path=...), edit_file(path=..., old_text=..., new_text=...), write_file(path=..., content=...), or terminal(command=...). If the task is already complete, reply with only TASK_COMPLETE."

const defaultTerminalTaskTimeoutSec = 4 * 60 * 60
const defaultTerminalCommandTimeoutSec = 30 * 60
const defaultTerminalMaxTurns = 200
const terminalModelObservationLimit = 12_000
const terminalMessageBudgetBytes = 180_000
const terminalRecentMessageKeep = 24
const defaultTerminalModelMaxTokens = 16_384
const defaultTerminalRetryMaxTokens = 8_192
const defaultTerminalEndpointTimeout = 10 * time.Minute
const maxTerminalRetryReserve = 5 * time.Minute

const terminalJSONProtocolTemplate = `You are an AI assistant tasked with solving command-line tasks in a Linux environment. You will be given a task description and the output from previously executed commands. Your goal is to solve the task by providing batches of shell commands.

Format your response as JSON with the following structure:

{
  "analysis": "Analyze the current state based on the terminal output provided. What do you see? What has been accomplished? What still needs to be done?",
  "plan": "Describe your plan for the next steps. What commands will you run and why? Be specific about what you expect each command to accomplish.",
  "commands": [
    {
      "keystrokes": "ls -la\n",
      "duration": 0.1
    },
    {
      "keystrokes": "cd project\n",
      "duration": 0.1
    }
  ],
  "task_complete": true
}

Required fields:
- "analysis": Your analysis of the current situation
- "plan": Your plan for the next steps
- "commands": Array of command objects to execute

Optional fields:
- "task_complete": Boolean indicating if the task is complete (defaults to false if not present)

Command object structure:
- "keystrokes": String containing the exact keystrokes to send to the terminal (required)
- "duration": Number of seconds to wait before the next command when no command is running (defaults to 1.0 if not present)

IMPORTANT: The text inside "keystrokes" will be used completely verbatim as shell input. Write commands exactly as you want them sent to the terminal:
- You must end every command with a newline (\n) or it will not execute.
- For special key sequences, use tmux-style escape sequences:
  - C-c for Ctrl+C
  - C-d for Ctrl+D

Important notes:
- Each command's keystrokes are sent exactly as written to the terminal.
- Batch related commands in one response when they are part of the same inspection/edit/test step.
- Do not include extra whitespace before or after the keystrokes unless it is part of the intended command.
- Extra text before or after the JSON will generate warnings but be tolerated.
- The JSON must be valid; use proper escaping for quotes and special characters within strings.
- Commands array can be empty if you want to wait without taking action.
- Before setting "task_complete": true, run a concise self-check that covers every explicit acceptance criterion in the task description, especially any tests the task says should pass.
- If the task asks for compiled/native extensions, verify the native modules or binaries are actually built and importable/executable, not just that Python fallbacks or partial smoke tests work.

Task Description:
%s

Current terminal state:
%s`

type terminalTask struct {
	ID          string                    `json:"id"`
	Version     string                    `json:"version"`
	Instruction string                    `json:"instruction"`
	Category    string                    `json:"category,omitempty"`
	Source      string                    `json:"source"`
	Image       terminalImage             `json:"image"`
	Agent       terminalAgentConfig       `json:"agent"`
	Verifier    terminalVerifierConfig    `json:"verifier"`
	Solution    terminalSolutionConfig    `json:"solution"`
	Environment terminalEnvironmentConfig `json:"environment"`
	Artifacts   []terminalArtifactConfig  `json:"artifacts,omitempty"`
}

type terminalImage struct {
	Prebuilt        string `json:"prebuilt,omitempty"`
	Dockerfile      string `json:"dockerfile,omitempty"`
	Context         string `json:"context,omitempty"`
	BuildTimeoutSec int    `json:"buildTimeoutSec,omitempty"`
	ComposeFile     string `json:"composeFile,omitempty"`
}

type terminalAgentConfig struct {
	TimeoutSec int    `json:"timeoutSec"`
	MaxTurns   int    `json:"maxTurns"`
	User       string `json:"user"`
}

type terminalVerifierCollectConfig struct {
	Command    string            `json:"command"`
	Service    string            `json:"service,omitempty"`
	TimeoutSec int               `json:"timeoutSec,omitempty"`
	Env        map[string]string `json:"env,omitempty"`
}

type terminalVerifierConfig struct {
	TimeoutSec      int                             `json:"timeoutSec"`
	Command         string                          `json:"command"`
	RewardFile      string                          `json:"rewardFile"`
	User            string                          `json:"user"`
	Env             map[string]string               `json:"env,omitempty"`
	EnvironmentMode string                          `json:"environmentMode,omitempty"`
	BuildTimeoutSec int                             `json:"buildTimeoutSec,omitempty"`
	Environment     terminalEnvironmentConfig       `json:"environment,omitempty"`
	Collect         []terminalVerifierCollectConfig `json:"collect,omitempty"`
}

type terminalSolutionConfig struct {
	Env map[string]string `json:"env,omitempty"`
}

type terminalArtifactConfig struct {
	Source  string   `json:"source"`
	Service string   `json:"service,omitempty"`
	Exclude []string `json:"exclude,omitempty"`
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
	apiKey                         string
	args                           cliArgs
	maxTokens                      int
	temperature                    float64
	topP                           float64
	commandTimeoutSec              int
	agentTimeoutSec                int
	maxTurns                       int
	cleanupImages                  bool
	oracle                         bool
	agentCommand                   string
	agentExecution                 string
	nativeTools                    bool
	shellMode                      string
	traceRoot                      string
	endpointTimeout                time.Duration
	modelHeartbeatInterval         time.Duration
	externalAgentHeartbeatInterval time.Duration
	runLabel                       string
	thinkingLevel                  string
	thinkingSource                 string
}

var (
	terminalThinkingOffPattern    = regexp.MustCompile(`(?i)(^|[^a-z0-9])(no[-_ ]?think(?:ing)?|non[-_ ]?think(?:ing)?|think(?:ing)?[-_ ]?off)([^a-z0-9]|$)`)
	terminalThinkingLevelPattern  = regexp.MustCompile(`(?i)(^|[^a-z0-9])(xhigh|high|medium|low|auto)([^a-z0-9]|$)`)
	terminalThinkingDefaultEffort = regexp.MustCompile(`(?i)reasoning_effort\s*\|\s*default\(\s*['"](xhigh|high|medium|low|auto)['"]\s*\)`)
)

func normalizeTerminalThinkingLevel(value string) (string, bool) {
	normalized := strings.ToLower(strings.TrimSpace(value))
	normalized = strings.NewReplacer("_", "-", " ", "-").Replace(normalized)
	switch normalized {
	case "off", "disabled", "none", "no-think", "nothink", "non-thinking", "nonthinking":
		return "off", true
	case "on", "enabled", "thinking":
		return "on", true
	case "low", "medium", "high", "xhigh", "auto":
		return normalized, true
	default:
		return "", false
	}
}

func inferTerminalThinkingLevel(text string) string {
	if terminalThinkingOffPattern.MatchString(text) {
		return "off"
	}
	match := terminalThinkingLevelPattern.FindStringSubmatch(text)
	if len(match) >= 3 {
		return strings.ToLower(match[2])
	}
	return ""
}

func inferTerminalThinkingLevelFromProps(props map[string]any) string {
	for _, key := range []string{"model_alias", "model_path"} {
		if level := inferTerminalThinkingLevel(stringValue(props[key])); level != "" {
			return level
		}
	}
	template := stringValue(props["chat_template"])
	if match := terminalThinkingDefaultEffort.FindStringSubmatch(template); len(match) == 2 {
		return strings.ToLower(match[1])
	}
	if strings.Contains(template, "enable_thinking") && strings.Contains(template, "<think>") {
		return "on"
	}
	return ""
}

func promptTerminalThinkingLevel(reader io.Reader, writer io.Writer) (string, error) {
	fmt.Fprintln(writer, "The model's thinking level could not be inferred from its endpoint metadata.")
	fmt.Fprintln(writer, "Choose the level actually used: off, on, low, medium, high, xhigh, or auto.")
	scanner := bufio.NewScanner(reader)
	for {
		fmt.Fprint(writer, "Thinking level: ")
		if !scanner.Scan() {
			if err := scanner.Err(); err != nil {
				return "", err
			}
			return "", cliError{"missing_thinking_level", "No thinking level was provided.", []string{"Pass --thinking-level off|on|low|medium|high|xhigh|auto."}, nil}
		}
		if level, ok := normalizeTerminalThinkingLevel(scanner.Text()); ok {
			return level, nil
		}
		fmt.Fprintln(writer, "Invalid level. Use off, on, low, medium, high, xhigh, or auto.")
	}
}

func terminalThinkingPromptAllowed(args cliArgs) bool {
	if hasFlag(args, "detach") || hasFlag(args, "detached-child") || hasFlag(args, "dry-run") || hasFlag(args, "json") || hasFlag(args, "json-status") || hasFlag(args, "quiet") {
		return false
	}
	info, err := os.Stdin.Stat()
	return err == nil && info.Mode()&os.ModeCharDevice != 0
}

func resolveTerminalThinkingLevel(ctx context.Context, args cliArgs, baseURL, model string, cfg terminalConfig, reader io.Reader, writer io.Writer, allowPrompt bool) (string, string, error) {
	explicitLevel := ""
	if raw := opt(args, "thinking-level"); raw != "" {
		level, ok := normalizeTerminalThinkingLevel(raw)
		if !ok {
			return "", "", cliError{"invalid_thinking_level", "Unsupported terminal thinking level.", []string{"Pass --thinking-level off|on|low|medium|high|xhigh|auto."}, map[string]any{"thinkingLevel": raw}}
		}
		explicitLevel = level
	}
	if cfg.oracle {
		return "not-applicable", "oracle", nil
	}
	// The built-in terminal harness explicitly sends enable_thinking=false for
	// Qwen models. External agents own their request configuration, so their
	// served model or endpoint metadata must identify the effective level.
	if cfg.agentCommand == "" && terminalDisablesTemplateThinking(model) {
		if explicitLevel != "" && explicitLevel != "off" {
			return "", "", cliError{"thinking_level_conflict", "The built-in terminal harness disables Qwen template thinking.", []string{"Use --thinking-level off.", "Use an external agent configured for the desired thinking level."}, map[string]any{"model": model, "thinkingLevel": explicitLevel}}
		}
		return "off", "harness", nil
	}
	if explicitLevel != "" {
		return explicitLevel, "cli", nil
	}
	if level := inferTerminalThinkingLevel(model); level != "" {
		return level, "served-model", nil
	}
	if baseURL != "" {
		if raw, err := fetchEndpointJSONContext(ctx, baseURL+"/props", cfg.apiKey); err == nil {
			if level := inferTerminalThinkingLevelFromProps(asObject(raw)); level != "" {
				return level, "endpoint", nil
			}
		}
	}
	if allowPrompt {
		level, err := promptTerminalThinkingLevel(reader, writer)
		return level, "prompt", err
	}
	return "not-provided", "unresolved", nil
}

type terminalJSONCommand struct {
	Keystrokes string  `json:"keystrokes"`
	Duration   float64 `json:"duration"`
}

type terminalJSONResponse struct {
	Analysis     string                `json:"analysis"`
	Plan         string                `json:"plan"`
	Commands     []terminalJSONCommand `json:"commands"`
	TaskComplete bool                  `json:"task_complete"`
}

type terminalTokenUsage struct {
	inputTokens      int64
	outputTokens     int64
	cacheReadTokens  int64
	cacheWriteTokens int64
	totalTokens      int64
	modelCalls       int
}

func (u *terminalTokenUsage) add(v terminalTokenUsage) {
	u.inputTokens += v.inputTokens
	u.outputTokens += v.outputTokens
	u.cacheReadTokens += v.cacheReadTokens
	u.cacheWriteTokens += v.cacheWriteTokens
	u.totalTokens += v.totalTokens
	u.modelCalls += v.modelCalls
}

func (u terminalTokenUsage) toMap() map[string]any {
	return map[string]any{
		"inputTokens":      u.inputTokens,
		"outputTokens":     u.outputTokens,
		"cacheReadTokens":  u.cacheReadTokens,
		"cacheWriteTokens": u.cacheWriteTokens,
		"totalTokens":      u.totalTokens,
		"modelCalls":       u.modelCalls,
	}
}

type terminalTaskResult struct {
	pass            bool
	scored          bool
	turns           int
	turnsUnreported bool
	transcript      string
	verifierOutput  string
	wallTimeMs      int64
	usage           terminalTokenUsage
	errText         string
	errCode         string
	instruction     string
	prompt          string
}

type terminalBundle struct {
	Task terminalTask
	Dir  string
}

type harborArtifacts []terminalArtifactConfig

func (artifacts *harborArtifacts) UnmarshalTOML(data any) error {
	items, ok := data.([]map[string]any)
	if ok {
		for _, item := range items {
			artifact, err := harborArtifactFromTOML(item)
			if err != nil {
				return err
			}
			*artifacts = append(*artifacts, artifact)
		}
		return nil
	}
	values, ok := data.([]any)
	if !ok {
		return fmt.Errorf("artifacts must be an array")
	}
	for _, value := range values {
		switch item := value.(type) {
		case string:
			if strings.TrimSpace(item) == "" {
				return fmt.Errorf("artifact source must not be empty")
			}
			*artifacts = append(*artifacts, terminalArtifactConfig{Source: item})
		case map[string]any:
			artifact, err := harborArtifactFromTOML(item)
			if err != nil {
				return err
			}
			*artifacts = append(*artifacts, artifact)
		default:
			return fmt.Errorf("artifact must be a path or inline table, got %T", value)
		}
	}
	return nil
}

func harborArtifactFromTOML(item map[string]any) (terminalArtifactConfig, error) {
	source, _ := item["source"].(string)
	if strings.TrimSpace(source) == "" {
		return terminalArtifactConfig{}, fmt.Errorf("artifact inline table requires source")
	}
	service, _ := item["service"].(string)
	var exclude []string
	switch values := item["exclude"].(type) {
	case nil:
	case []string:
		exclude = append(exclude, values...)
	case []any:
		for _, value := range values {
			pattern, ok := value.(string)
			if !ok {
				return terminalArtifactConfig{}, fmt.Errorf("artifact exclude entries must be strings")
			}
			exclude = append(exclude, pattern)
		}
	default:
		return terminalArtifactConfig{}, fmt.Errorf("artifact exclude must be an array")
	}
	return terminalArtifactConfig{Source: source, Service: service, Exclude: exclude}, nil
}

type harborVerifierConfig struct {
	TimeoutSec      tomlNumber        `toml:"timeout_sec"`
	Command         string            `toml:"command"`
	RewardFile      string            `toml:"reward_file"`
	Env             map[string]string `toml:"env"`
	User            string            `toml:"user"`
	EnvironmentMode string            `toml:"environment_mode"`
	Collect         []struct {
		Command    string            `toml:"command"`
		Service    string            `toml:"service"`
		TimeoutSec tomlNumber        `toml:"timeout_sec"`
		Env        map[string]string `toml:"env"`
	} `toml:"collect"`
	Environment struct {
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

type harborTaskToml struct {
	Artifacts harborArtifacts `toml:"artifacts"`
	Task      struct {
		Name        string `toml:"name"`
		Description string `toml:"description"`
	} `toml:"task"`
	Metadata struct {
		Category string `toml:"category"`
	} `toml:"metadata"`
	Verifier harborVerifierConfig `toml:"verifier"`
	Agent    struct {
		TimeoutSec tomlNumber `toml:"timeout_sec"`
		MaxTurns   tomlNumber `toml:"max_turns"`
		User       string     `toml:"user"`
	} `toml:"agent"`
	Solution struct {
		Env map[string]string `toml:"env"`
	} `toml:"solution"`
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
	case "publish":
		return publishTerminalDataset(args)
	case "run":
		return runTerminalEval(args, false)
	case "submit":
		return submitTerminalEval(args)
	case "verify":
		return runTerminalEval(args, true)
	case "status":
		return terminalStatus(args)
	case "logs":
		return terminalLogs(args)
	case "cancel":
		return terminalCancel(args)
	default:
		return cliError{"unknown_subcommand", "Unknown eval terminal subcommand.", []string{"Use one of: import, publish, run, submit, verify, status, logs, cancel."}, map[string]any{"subcommand": sub}}
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
	printInfo(args, "terminal_import_complete", map[string]any{"src": src, "out": out, "tasks": len(taskDirs), "version": version})
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

// harborComposeFile reports the docker-compose file name inside a harbor task's
// environment/, or "" when the task is a plain Dockerfile/prebuilt-image task.
func harborComposeFile(taskDir string) string {
	for _, name := range []string{"docker-compose.yaml", "docker-compose.yml"} {
		if _, err := os.Stat(filepath.Join(taskDir, "environment", name)); err == nil {
			return name
		}
	}
	return ""
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
	image := terminalImage{ComposeFile: harborComposeFile(taskDir)}
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
	verifierMode := firstNonEmpty(strings.TrimSpace(ht.Verifier.EnvironmentMode), "shared")
	if verifierMode != "shared" && verifierMode != "separate" {
		return cliError{"task_import_failed", fmt.Sprintf("%s: invalid verifier environment mode", id), []string{"Set [verifier].environment_mode to shared or separate."}, map[string]any{"taskId": id, "environment_mode": ht.Verifier.EnvironmentMode}}
	}
	verifierNetwork := strings.TrimSpace(ht.Verifier.Environment.NetworkMode)
	if verifierNetwork == "" && ht.Verifier.Environment.AllowInternet != nil {
		if *ht.Verifier.Environment.AllowInternet {
			verifierNetwork = "public"
		} else {
			verifierNetwork = "no-network"
		}
	}
	verifierNetwork = normalizeTerminalNetwork(firstNonEmpty(verifierNetwork, network))
	if verifierNetwork == "" {
		return cliError{"task_import_failed", fmt.Sprintf("%s: invalid verifier environment network", id), []string{"Set [verifier.environment].network_mode to public, no-network, or allowlist."}, map[string]any{"taskId": id, "network_mode": ht.Verifier.Environment.NetworkMode}}
	}
	collectHooks := make([]terminalVerifierCollectConfig, 0, len(ht.Verifier.Collect))
	for _, hook := range ht.Verifier.Collect {
		if strings.TrimSpace(hook.Command) == "" {
			return cliError{"task_import_failed", fmt.Sprintf("%s: verifier collect hook has no command", id), []string{"Set [[verifier.collect]].command."}, map[string]any{"taskId": id}}
		}
		collectHooks = append(collectHooks, terminalVerifierCollectConfig{
			Command: strings.TrimSpace(hook.Command), Service: strings.TrimSpace(hook.Service),
			TimeoutSec: firstPositive(hook.TimeoutSec.Int(), 30), Env: nonNilStringMap(hook.Env),
		})
	}
	verifierEnvironmentConfigured := ht.Verifier.Environment.BuildTimeoutSec.Float() > 0 ||
		ht.Verifier.Environment.CPUs.Float() > 0 || ht.Verifier.Environment.MemoryMb.Int() > 0 ||
		ht.Verifier.Environment.StorageMb.Int() > 0 || ht.Verifier.Environment.GPUs.Int() > 0 ||
		ht.Verifier.Environment.AllowInternet != nil || strings.TrimSpace(ht.Verifier.Environment.NetworkMode) != "" ||
		len(ht.Verifier.Environment.AllowedHosts) > 0 || len(ht.Verifier.Environment.Env) > 0
	verifierGPUs := ht.Environment.GPUs.Int()
	if verifierEnvironmentConfigured {
		verifierGPUs = ht.Verifier.Environment.GPUs.Int()
	}
	verifierEnvironment := terminalEnvironmentConfig{
		CPUs:      firstPositiveFloat(ht.Verifier.Environment.CPUs.Float(), firstPositiveFloat(ht.Environment.CPUs.Float(), 1)),
		MemoryMb:  firstPositive(ht.Verifier.Environment.MemoryMb.Int(), firstPositive(ht.Environment.MemoryMb.Int(), 2048)),
		StorageMb: firstPositive(ht.Verifier.Environment.StorageMb.Int(), firstPositive(ht.Environment.StorageMb.Int(), 10240)),
		GPUs:      verifierGPUs, Network: verifierNetwork,
		AllowedHosts: ht.Verifier.Environment.AllowedHosts, Env: nonNilStringMap(ht.Verifier.Environment.Env),
	}
	task := terminalTask{
		ID:          id,
		Version:     version,
		Instruction: string(instructionBytes),
		Category:    ht.Metadata.Category,
		Source:      firstNonEmpty(ht.Task.Name, "terminal-bench/"+id),
		Image:       image,
		Agent:       terminalAgentConfig{TimeoutSec: ht.Agent.TimeoutSec.Int(), MaxTurns: ht.Agent.MaxTurns.Int(), User: ht.Agent.User},
		Verifier: terminalVerifierConfig{
			TimeoutSec: firstPositive(ht.Verifier.TimeoutSec.Int(), 900), Command: firstNonEmpty(ht.Verifier.Command, "bash /tests/test.sh"),
			RewardFile: firstNonEmpty(ht.Verifier.RewardFile, "/logs/verifier/reward.txt"), User: ht.Verifier.User,
			Env: nonNilStringMap(ht.Verifier.Env), EnvironmentMode: verifierMode,
			BuildTimeoutSec: firstPositive(ht.Verifier.Environment.BuildTimeoutSec.Int(), image.BuildTimeoutSec),
			Environment:     verifierEnvironment, Collect: collectHooks,
		},
		Solution:    terminalSolutionConfig{Env: nonNilStringMap(ht.Solution.Env)},
		Environment: terminalEnvironmentConfig{CPUs: firstPositiveFloat(ht.Environment.CPUs.Float(), 1), MemoryMb: firstPositive(ht.Environment.MemoryMb.Int(), 2048), StorageMb: firstPositive(ht.Environment.StorageMb.Int(), 10240), GPUs: ht.Environment.GPUs.Int(), Network: network, AllowedHosts: ht.Environment.AllowedHosts, Env: nonNilStringMap(ht.Environment.Env)},
		Artifacts:   append([]terminalArtifactConfig(nil), ht.Artifacts...),
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
	if hasFlag(args, "detach") && !hasFlag(args, "detached-child") {
		return startDetachedTerminalEval(args)
	}
	if hasFlag(args, "detached-child") {
		if err := awaitDetachedTerminalProcessRecord(args); err != nil {
			return err
		}
	}
	ctx, stopSignals := terminalSignalContext()
	defer stopSignals()
	go func() {
		<-ctx.Done()
		stopSignals()
	}()
	if runDir := opt(args, "run-dir"); runDir != "" && !hasFlag(args, "detached-child") {
		root, err := filepath.Abs(runDir)
		if err != nil {
			return err
		}
		args.opts["status-log"] = terminalEventsPath(root)
	}
	localDir := opt(args, "task-dir")
	if forceOracle && localDir == "" {
		localDir = positional(args, 3)
	}
	dataset := positional(args, 3)
	if localDir != "" && (forceOracle || hasFlag(args, "oracle")) {
		dataset = opt(args, "dataset")
	}
	submit := hasFlag(args, "submit")
	dryRun := hasFlag(args, "dry-run")
	if submit && localDir != "" && dataset == "" {
		dataset = opt(args, "dataset")
	}
	if !forceOracle && localDir == "" && dataset == "" {
		return cliError{"missing_option", "eval terminal run requires a dataset slug or --task-dir.", []string{"Run: lmx eval terminal run terminal-bench-2-1 --base-url <url> --model <hfId>.", "For local bundles, run: lmx eval terminal run --task-dir ./bundles --base-url <url> --model <hfId>."}, nil}
	}

	rawBaseURL := opt(args, "base-url")
	baseURL := ""
	callModel := ""
	declaredModel := opt(args, "model")
	var quantResolution map[string]any
	resolvedQuant, resolvedQuantFormat := "", opt(args, "quant-format")
	var modelResolution map[string]any
	agentBackend := opt(args, "agent")
	agentCommand := opt(args, "agent-cmd")
	if agentBackend == "terminus-2" {
		if agentCommand != "" {
			return cliError{"invalid_option", "--agent terminus-2 cannot be combined with --agent-cmd.", []string{"Use --agent terminus-2 for the bundled Harbor Terminus-2 adapter, or use --agent-cmd with --agent-name for a custom wrapper."}, nil}
		}
		extractedCommand, cleanupAdapter, err := terminus2AgentCommand()
		if err != nil {
			return err
		}
		agentCommand = extractedCommand
		defer cleanupAdapter()
	}
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
			if dryRun {
				callModel = firstNonEmpty(servedModel, declaredModel, "local")
				resolvedQuant = opt(args, "quantization")
			} else {
				var servedModelInfo map[string]any
				if servedModel == "" {
					if detected, info, derr := detectTerminalServedModel(ctx, baseURL, opt(args, "model-api-key"), declaredModel); derr == nil {
						servedModel = detected
						servedModelInfo = info
					} else {
						if ctx.Err() != nil {
							return terminalCancelledError(ctx)
						}
						printStatus(args, "eval_model_detection_unavailable", map[string]any{"baseUrl": baseURL, "reason": derr.Error()})
					}
				} else if _, info, derr := detectTerminalServedModel(ctx, baseURL, opt(args, "model-api-key"), servedModel); derr == nil {
					servedModelInfo = info
				} else if ctx.Err() != nil {
					return terminalCancelledError(ctx)
				}
				callModel = firstNonEmpty(servedModel, declaredModel, "local")
				quantResolution = remoteQuantizationResolutionContext(ctx, args, baseURL, opt(args, "model-api-key"), opt(args, "quantization"), servedModelInfo)
				if ctx.Err() != nil {
					return terminalCancelledError(ctx)
				}
				resolvedQuant = firstNonEmpty(stringValue(quantResolution["trusted"]), opt(args, "quantization"))
				if resolvedQuantFormat == "" && strings.EqualFold(filepath.Ext(stringValue(quantResolution["modelPath"])), ".gguf") {
					resolvedQuantFormat = "gguf"
				}
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
	commandTimeout, err := intOption(args, 0, 1, "command-timeout", "command-timeout-seconds")
	if err != nil {
		return err
	}
	concurrency, err := intOption(args, 1, 1, "concurrency")
	if err != nil {
		return err
	}
	cfg := terminalConfig{apiKey: opt(args, "model-api-key"), args: args, maxTokens: maxTokens, temperature: floatOption(args, "temperature", 0), topP: floatOption(args, "top-p", 1), commandTimeoutSec: commandTimeout, cleanupImages: hasFlag(args, "cleanup-images"), oracle: forceOracle || hasFlag(args, "oracle"), agentCommand: agentCommand, agentExecution: firstNonEmpty(opt(args, "agent-execution"), map[bool]string{true: "routed-shell"}[agentBackend == "terminus-2"], "host"), nativeTools: hasFlag(args, "native-tools"), traceRoot: opt(args, "trace-dir")}
	cfg.maxTurns, err = intOption(args, 0, 0, "max-turns")
	if err != nil {
		return err
	}
	cfg.agentTimeoutSec, err = intOption(args, 0, 0, "agent-timeout")
	if err != nil {
		return err
	}
	if firstNonEmpty(opt(args, "endpoint-timeout-seconds"), opt(args, "timeout-seconds")) != "" {
		cfg.endpointTimeout, err = endpointTimeout(args)
		if err != nil {
			return err
		}
	}
	cfg.shellMode = firstNonEmpty(opt(args, "shell-mode"), "persistent")
	if cfg.shellMode != "persistent" && cfg.shellMode != "stateless" {
		return cliError{"invalid_option", "--shell-mode must be persistent or stateless", []string{"Pass --shell-mode persistent (default) or --shell-mode stateless."}, map[string]any{"shellMode": cfg.shellMode}}
	}
	if cfg.agentExecution != "host" && cfg.agentExecution != "container" && cfg.agentExecution != "routed-shell" {
		return cliError{"invalid_option", "--agent-execution must be host, container, or routed-shell", []string{"Use host for legacy external commands, container for agents launched inside the task container, or routed-shell for host agents whose shell is mechanically routed into the task container."}, map[string]any{"agentExecution": cfg.agentExecution}}
	}
	if cfg.traceRoot == "" && opt(args, "run-dir") != "" {
		cfg.traceRoot = filepath.Join(opt(args, "run-dir"), "traces")
	}

	cfg.thinkingLevel, cfg.thinkingSource, err = resolveTerminalThinkingLevel(ctx, args, baseURL, callModel, cfg, os.Stdin, os.Stderr, terminalThinkingPromptAllowed(args))
	if err != nil {
		return err
	}
	if submit && !dryRun && cfg.thinkingLevel == "not-provided" {
		return cliError{"missing_thinking_level", "Terminal benchmark submission requires a known thinking level.", []string{"Pass --thinking-level off|on|low|medium|high|xhigh|auto.", "Interactive terminal runs prompt when endpoint metadata is ambiguous."}, map[string]any{"model": callModel}}
	}

	var hardware any
	if submit || (dryRun && dataset != "") {
		if declaredModel == "" {
			return cliError{"missing_model", "eval terminal publication requires --model <HuggingFace model id>", []string{"Pass --model org/name so the submission records a real model.", "Use lmx model search <name> to find the canonical id."}, nil}
		}
		hardwarePath := opt(args, "hardware")
		if hardwarePath == "" {
			return cliError{"missing_hardware", "eval terminal publication requires --hardware hardware.json", []string{"Run lmx hardware --out hardware.json and pass --hardware hardware.json."}, nil}
		}
		hardware, err = readJSON(hardwarePath)
		if err != nil {
			return err
		}
		if submit && apiKey(args) == "" {
			return missingAPIKey("--api-key or LMX_API_KEY is required for eval terminal --submit")
		}
		if rawBaseURL != "" && opt(args, "quantization") == "" && resolvedQuant == "" {
			return cliError{"model_detection_failed", "Could not verify the local endpoint model/quantization before terminal execution.", []string{"Keep the model endpoint running.", "Or pass --quantization and --quant-format explicitly if the endpoint cannot expose model metadata."}, map[string]any{"baseUrl": baseURL}}
		}
	}

	hfID := declaredModel
	if declaredModel != "" && (rawBaseURL != "" || submit || (dryRun && dataset != "")) {
		submitArgs := argsWithTerminalBaseURL(args, rawBaseURL)
		hfID, modelResolution, err = resolveEvalModelID(submitArgs, declaredModel)
		if err != nil {
			return err
		}
	}
	var coverageBefore map[string]any
	if dataset != "" && localDir == "" && (submit || dryRun) && opt(args, "shard") == "" {
		args, coverageBefore, err = selectMissingTerminalShard(ctx, args, dataset, hfID, resolvedQuant, resolvedQuantFormat, cfg)
		if err != nil {
			return err
		}
		cfg.args = args
	}

	if dryRun {
		taskCount, shardIndex := 0, -1
		if localDir != "" {
			bundles, loadErr := loadTerminalBundles(localDir)
			if loadErr != nil {
				return loadErr
			}
			taskCount = len(bundles)
		} else {
			items, selectedShard, manifestErr := fetchTerminalManifestItems(ctx, args, dataset)
			if manifestErr != nil {
				return manifestErr
			}
			taskCount, shardIndex = len(items), selectedShard
		}
		plan := map[string]any{"dryRun": true, "dataset": firstNonEmpty(dataset, "local"), "tasks": taskCount, "shardIndex": shardIndex, "apiUrl": apiURL(args), "baseUrl": baseURL, "model": hfID, "servedModel": callModel, "quantization": resolvedQuant, "quantFormat": resolvedQuantFormat, "thinkingLevel": cfg.thinkingLevel, "thinkingLevelSource": cfg.thinkingSource, "harnessKey": terminalHarnessKey(args, cfg), "agentTimeoutDefaultSec": defaultTerminalTaskTimeoutSec, "commandTimeoutDefaultSec": defaultTerminalCommandTimeoutSec, "coverageBefore": coverageBefore, "canSubmit": apiKey(args) != "" && hfID != "" && hardware != nil}
		return writeOrPrintJSON("terminal_eval_preflight", args, plan)
	}

	bundles, cleanup, shardIndex, err := acquireTerminalBundles(ctx, args, dataset, localDir)
	if cleanup != "" {
		defer os.RemoveAll(cleanup)
	}
	if err != nil {
		return err
	}
	if len(bundles) == 0 {
		return cliError{"bundle_invalid", "No terminal task bundles were found.", []string{"Point --task-dir at one bundle or a parent directory of bundles containing task.json and tests/."}, map[string]any{"taskDir": localDir}}
	}
	checkpoint, resumed, err := openTerminalLiveCheckpoint(args, dataset, shardIndex, bundles, baseURL, callModel, hfID, resolvedQuant, resolvedQuantFormat, hardware, cfg)
	defer checkpoint.close()
	if err != nil {
		return err
	}
	if len(resumed) < len(bundles) {
		if err := dockerPreflightContext(ctx); err != nil {
			return err
		}
	}
	if checkpoint != nil {
		cfg.runLabel = shortHash(checkpoint.root)
		if len(resumed) < len(bundles) {
			if err := cleanupTerminalRunContainers(ctx, cfg.runLabel); err != nil {
				return err
			}
		}
	}
	startFields := map[string]any{"dataset": firstNonEmpty(dataset, "local"), "tasks": len(bundles), "resumedTasks": len(resumed), "pendingTasks": len(bundles) - len(resumed), "model": firstNonEmpty(callModel, "oracle"), "baseUrl": baseURL, "concurrency": concurrency, "oracle": cfg.oracle, "thinkingLevel": cfg.thinkingLevel, "thinkingLevelSource": cfg.thinkingSource}
	if checkpoint != nil {
		startFields["runDir"] = checkpoint.root
	}
	printStatus(args, "terminal_eval_start", startFields)
	results, err := runTerminalBundles(ctx, args, bundles, baseURL, callModel, cfg, concurrency, resumed, checkpoint)
	if err != nil {
		if errors.Is(err, context.Canceled) || ctx.Err() != nil {
			_ = checkpoint.interrupt()
			printStatus(args, "terminal_eval_interrupted", map[string]any{"runDir": opt(args, "run-dir"), "resumable": opt(args, "run-dir") != ""})
			return terminalCancelledError(ctx)
		}
		return err
	}
	if cfg.runLabel != "" {
		if err := cleanupTerminalRunContainers(ctx, cfg.runLabel); err != nil {
			return err
		}
		printStatus(args, "terminal_cleanup_verified", map[string]any{"runLabel": cfg.runLabel, "runDir": opt(args, "run-dir")})
	}
	stats := shardStats{}
	errorCodes := map[string]string{}
	for _, result := range results {
		stats.totalLatencyMs += result.wallTimeMs
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
	totalUsage := terminalTokenUsage{}
	for _, result := range results {
		totalUsage.add(result.usage)
	}
	accuracy := 0.0
	if stats.scored > 0 {
		accuracy = float64(stats.correct) / float64(stats.scored)
	}
	avgLatency := int64(0)
	if len(results) > 0 {
		avgLatency = stats.totalLatencyMs / int64(len(results))
	}
	summary := map[string]any{"dataset": firstNonEmpty(dataset, "local"), "shardIndex": shardIndex, "tasks": len(results), "correct": stats.correct, "scored": stats.scored, "errors": stats.errors, "accuracyPct": roundMetric(accuracy * 100), "avgWallTimeMs": avgLatency, "wallTimeMs": stats.totalLatencyMs, "tokenUsage": totalUsage.toMap(), "quantization": resolvedQuant, "quantFormat": resolvedQuantFormat, "thinkingLevel": cfg.thinkingLevel, "thinkingLevelSource": cfg.thinkingSource, "errorCodes": errorCodes}
	records := make([]any, len(results))
	for i, r := range results {
		records[i] = map[string]any{"question_id": bundles[i].Task.ID, "pass": r.pass, "scored": r.scored, "error": r.errText, "errorCode": r.errCode, "latencyMs": r.wallTimeMs, "wallTimeMs": r.wallTimeMs, "tokenUsage": r.usage.toMap(), "turns": terminalTaskTurnsValue(r), "question": bundles[i].Task.Instruction, "prompt": r.prompt, "response": r.transcript, "verifierOutput": r.verifierOutput}
	}
	resultDocument := map[string]any{"summary": summary, "results": records}
	if err := checkpoint.finish("completed", resultDocument); err != nil {
		return err
	}
	outPath := opt(args, "out")
	if outPath != "" {
		if err := writeJSON(outPath, resultDocument); err != nil {
			return err
		}
		printStatus(args, "terminal_results_written", map[string]any{"path": outPath})
	}
	if !submit {
		completeFields := map[string]any{"submitted": false}
		for key, value := range summary {
			completeFields[key] = value
		}
		if outPath != "" {
			completeFields["resultPath"] = outPath
		}
		if checkpoint != nil {
			completeFields["runDir"] = checkpoint.root
			completeFields["checkpointResultPath"] = filepath.Join(checkpoint.root, "result.json")
		}
		if hasFlag(args, "json-status") {
			printStatus(args, "terminal_eval_completed", completeFields)
		} else {
			printInfo(args, "terminal_eval_complete", summary)
			if humanOutput(args) {
				fmt.Println("Local run complete — nothing submitted.")
				if dataset != "" {
					if cfg.agentCommand != "" {
						fmt.Println("Publish with: lmx eval terminal run " + dataset + " --agent-cmd '<your-agent-command>' --model <hfId> --hardware hardware.json --submit")
					} else {
						fmt.Println("Publish with: lmx eval terminal run " + dataset + " --base-url " + rawBaseURL + " --model <hfId> --hardware hardware.json --submit")
					}
				}
			}
		}
		if hasFlag(args, "json-status") && (hasFlag(args, "json") || hasFlag(args, "print") || hasFlag(args, "verbose")) {
			printJSON(args, resultDocument)
		}
		if stats.errors == len(results) {
			code := dominantTerminalErrorCode(errorCodes)
			return cliError{code, "Every terminal task errored before scoring.", []string{"Inspect --out results.json and the terminal_task_error events."}, map[string]any{"errors": errorCodes}}
		}
		return nil
	}
	if stats.scored != len(bundles) || stats.errors != 0 {
		unscored := make([]string, 0, stats.errors)
		for i, result := range results {
			if !result.scored {
				unscored = append(unscored, bundles[i].Task.ID)
			}
		}
		return cliError{"incomplete_shard", "Terminal shard submission requires every fetched task to be scored.", []string{"Inspect --out results.json and fix every infrastructure error before retrying.", "Do not publish a partial shard."}, map[string]any{"tasks": len(bundles), "scored": stats.scored, "errors": stats.errors, "unscoredTaskIds": unscored, "errorCodes": errorCodes, "shardIndex": shardIndex}}
	}

	submitResults := make([]any, 0, len(results))
	submitArtifacts := make([]any, 0, len(results))
	for i, r := range results {
		task := bundles[i].Task
		submitResults = append(submitResults, map[string]any{"question_id": task.ID, "pass": r.pass})
		artifactResponse := truncateString(r.transcript+"\n\n# Verifier\n\n"+r.verifierOutput, 4_900_000)
		submitArtifacts = append(submitArtifacts, map[string]any{"question_id": task.ID, "itemIndex": i, "promptHash": shortHash(task.ID + ":" + r.prompt), "question": task.Instruction, "prompt": r.prompt, "response": artifactResponse, "score": boolScore(r.pass), "testPassed": r.pass, "latencyMs": r.wallTimeMs, "wallTimeMs": r.wallTimeMs, "tokenUsage": r.usage.toMap()})
	}
	protocol := terminalProtocolLabel(cfg)
	agentName := "lmx-terminus"
	if cfg.agentCommand != "" {
		agentName = firstNonEmpty(opt(args, "agent-name"), agentBackend, "external-agent")
	}
	taskLimits := make([]any, 0, len(bundles))
	commonMaxTurns := -1
	maxTurnsEnforcement := terminalMaxTurnsEnforcement(cfg)
	for _, bundle := range bundles {
		resolvedMaxTurns := terminalAgentMaxTurns(cfg, bundle.Task)
		if commonMaxTurns == -1 {
			commonMaxTurns = resolvedMaxTurns
		} else if commonMaxTurns != resolvedMaxTurns {
			commonMaxTurns = 0
		}
		taskLimits = append(taskLimits, map[string]any{
			"taskId":               bundle.Task.ID,
			"maxTurns":             resolvedMaxTurns,
			"maxTurnsSource":       terminalLimitSource(cfg.maxTurns, bundle.Task.Agent.MaxTurns),
			"maxTurnsEnforcement":  maxTurnsEnforcement,
			"agentTimeoutSec":      terminalAgentTimeoutSec(cfg, bundle.Task),
			"agentTimeoutSource":   terminalLimitSource(cfg.agentTimeoutSec, bundle.Task.Agent.TimeoutSec),
			"commandTimeoutSec":    terminalCommandTimeoutSec(cfg),
			"commandTimeoutSource": terminalLimitSource(cfg.commandTimeoutSec, 0),
		})
	}
	runConfig := map[string]any{"accuracy": accuracy, "tasksRun": len(results), "errors": stats.errors, "avgLatencyMs": avgLatency, "protocol": protocol, "agent": agentName, "thinkingLevel": cfg.thinkingLevel, "thinkingLevelSource": cfg.thinkingSource, "maxTurnsPolicy": map[bool]string{true: "cli-override", false: "per-task-manifest-or-fallback"}[cfg.maxTurns > 0], "maxTurnsEnforcement": maxTurnsEnforcement, "agentTimeoutPolicy": map[bool]string{true: "cli-override", false: "per-task-manifest-or-fallback"}[cfg.agentTimeoutSec > 0], "commandTimeoutSec": terminalCommandTimeoutSec(cfg), "taskLimits": taskLimits, "concurrency": concurrency, "modelResolution": modelResolution, "quantizationResolution": quantResolution}
	if commonMaxTurns > 0 {
		runConfig["maxTurns"] = commonMaxTurns
	}
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
	value, err := fetchJSONContext(ctx, "POST", apiURL(args)+"/api/benchmarks/"+url.PathEscape(dataset)+"/submit", apiKey(args), payload)
	if err != nil {
		if ctx.Err() != nil {
			return terminalCancelledError(ctx)
		}
		return err
	}
	if !hasFlag(args, "json-status") && (hasFlag(args, "json") || hasFlag(args, "print") || hasFlag(args, "verbose")) {
		printJSON(args, value)
	}
	obj := asObject(value)
	submission := map[string]any{"shardIndex": shardIndex, "submitted": len(submitResults), "run": obj["run"], "aggregate": obj["aggregate"]}
	resultDocument["submission"] = submission
	fields := map[string]any{"dataset": dataset, "submitted": len(submitResults), "accuracyPct": summary["accuracyPct"]}
	if agg := asObject(obj["aggregate"]); agg != nil {
		fields["pooledScore"] = agg["pooledScore"]
		fields["ciLower"] = agg["ciLower"]
		fields["ciUpper"] = agg["ciUpper"]
		fields["coverage"] = agg["shardsCovered"]
		summary["pooledScore"] = agg["pooledScore"]
		summary["ciLower"] = agg["ciLower"]
		summary["ciUpper"] = agg["ciUpper"]
		summary["coverage"] = agg["shardsCovered"]
	}
	if run := asObject(obj["run"]); run != nil {
		fields["runId"] = run["id"]
		fields["status"] = run["status"]
		summary["runId"] = run["id"]
		summary["status"] = run["status"]
	}
	if outPath != "" {
		if err := writeJSON(outPath, resultDocument); err != nil {
			return err
		}
		printStatus(args, "terminal_submission_receipt_written", map[string]any{"path": outPath, "shardIndex": shardIndex})
	}
	if outPath != "" {
		fields["resultPath"] = outPath
	}
	if err := checkpoint.finish("submitted", resultDocument); err != nil {
		return err
	}
	if checkpoint != nil {
		fields["runDir"] = checkpoint.root
		fields["checkpointResultPath"] = filepath.Join(checkpoint.root, "result.json")
	}
	if hasFlag(args, "json-status") {
		printStatus(args, "terminal_eval_submitted", fields)
		if hasFlag(args, "json") || hasFlag(args, "print") || hasFlag(args, "verbose") {
			printJSON(args, resultDocument)
		}
	} else {
		printInfo(args, "terminal_eval_submitted", fields)
	}
	return nil
}

const (
	terminalTracePreviewBytes     = 24 * 1024
	terminalVerifierPreviewBytes  = 7 * 1024
	terminalArtifactResponseBytes = 32 * 1024
	terminalTraceLineBytes        = 4 * 1024 * 1024
)

type terminalCheckpointEntry struct {
	Index   int            `json:"index"`
	Total   int            `json:"total"`
	Task    string         `json:"task"`
	Out     string         `json:"out"`
	Pass    bool           `json:"pass"`
	Scored  *bool          `json:"scored"`
	Summary map[string]any `json:"summary"`
}

type terminalSavedResult struct {
	QuestionID     string         `json:"question_id"`
	Pass           bool           `json:"pass"`
	Scored         *bool          `json:"scored"`
	Error          string         `json:"error"`
	ErrorCode      string         `json:"errorCode"`
	LatencyMs      int64          `json:"latencyMs"`
	WallTimeMs     int64          `json:"wallTimeMs"`
	TokenUsage     map[string]any `json:"tokenUsage"`
	Turns          *int           `json:"turns"`
	Question       string         `json:"question"`
	Prompt         string         `json:"prompt"`
	Response       string         `json:"response"`
	VerifierOutput string         `json:"verifierOutput"`
}

type terminalSavedTaskFile struct {
	Results []terminalSavedResult `json:"results"`
}

const terminalLiveCheckpointVersion = 1

type terminalLiveActiveTask struct {
	TaskID    string `json:"taskId"`
	Index     int    `json:"index"`
	StartedAt string `json:"startedAt"`
}

type terminalLiveCheckpoint struct {
	Version           int                      `json:"version"`
	State             string                   `json:"state"`
	Fingerprint       string                   `json:"fingerprint"`
	FingerprintFields map[string]string        `json:"fingerprintFields,omitempty"`
	ThinkingLevel     string                   `json:"thinkingLevel"`
	ThinkingSource    string                   `json:"thinkingLevelSource"`
	Dataset           string                   `json:"dataset"`
	ShardIndex        int                      `json:"shardIndex"`
	TaskIDs           []string                 `json:"taskIds"`
	CompletedTasks    []string                 `json:"completedTasks"`
	ActiveTasks       []terminalLiveActiveTask `json:"activeTasks,omitempty"`
	CreatedAt         string                   `json:"createdAt"`
	UpdatedAt         string                   `json:"updatedAt"`
}

type terminalLiveCheckpointStore struct {
	root  string
	state terminalLiveCheckpoint
	lock  *os.File
	mu    sync.Mutex
}

func terminalLiveCheckpointIdentity(args cliArgs, dataset string, shardIndex int, bundles []terminalBundle, baseURL, callModel, hfID, quantization, quantFormat string, hardware any, cfg terminalConfig) map[string]any {
	tasks := make([]terminalTask, len(bundles))
	for i := range bundles {
		tasks[i] = bundles[i].Task
	}
	return map[string]any{
		"dataset": dataset, "shardIndex": shardIndex, "tasks": tasks,
		"apiUrl": apiURL(args), "baseUrl": baseURL, "model": callModel,
		"hfId": hfID, "quantization": quantization, "quantFormat": quantFormat,
		"hardware": hardware, "harnessKey": terminalHarnessKey(args, cfg),
		"maxTokens": cfg.maxTokens, "temperature": cfg.temperature, "topP": cfg.topP,
		"commandTimeoutSec": cfg.commandTimeoutSec, "agentTimeoutSec": cfg.agentTimeoutSec,
		"endpointTimeoutNs": cfg.endpointTimeout.Nanoseconds(), "maxTurns": cfg.maxTurns,
		"oracle": cfg.oracle, "agentCommand": cfg.agentCommand,
		"agentExecution": cfg.agentExecution, "shellMode": cfg.shellMode,
		"containerBaseUrl": opt(args, "container-base-url"),
		"thinkingLevel":    cfg.thinkingLevel,
	}
}

func terminalLiveCheckpointFingerprint(identity map[string]any) (string, map[string]string, error) {
	data, err := json.Marshal(identity)
	if err != nil {
		return "", nil, err
	}
	fields := make(map[string]string, len(identity))
	for key, value := range identity {
		fieldData, err := json.Marshal(value)
		if err != nil {
			return "", nil, err
		}
		fieldSum := sha256.Sum256(fieldData)
		fields[key] = hex.EncodeToString(fieldSum[:])
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:]), fields, nil
}

func terminalChangedFingerprintFields(expected, actual map[string]string) []string {
	if len(expected) == 0 {
		return nil
	}
	changed := make([]string, 0)
	for key, expectedHash := range expected {
		if actual[key] != expectedHash {
			changed = append(changed, key)
		}
	}
	for key := range actual {
		if _, ok := expected[key]; !ok {
			changed = append(changed, key)
		}
	}
	sort.Strings(changed)
	return changed
}

func openTerminalLiveCheckpoint(args cliArgs, dataset string, shardIndex int, bundles []terminalBundle, baseURL, callModel, hfID, quantization, quantFormat string, hardware any, cfg terminalConfig) (*terminalLiveCheckpointStore, map[string]terminalTaskResult, error) {
	runDir := opt(args, "run-dir")
	resume := opt(args, "resume")
	if hasFlag(args, "resume") {
		return nil, nil, cliError{"invalid_option", "--resume requires a value for terminal runs.", []string{"Pass --resume auto or --resume none."}, nil}
	}
	if runDir == "" {
		if resume != "" {
			return nil, nil, cliError{"invalid_option", "--resume requires --run-dir for terminal runs.", []string{"Pass a durable --run-dir, or omit --resume."}, nil}
		}
		return nil, nil, nil
	}
	if resume == "" {
		resume = "auto"
	}
	if resume != "auto" && resume != "none" {
		return nil, nil, cliError{"invalid_option", "Terminal --resume must be auto or none.", []string{"Use --resume auto to continue scored tasks, or --resume none with a new run directory."}, map[string]any{"resume": resume}}
	}
	root, err := filepath.Abs(runDir)
	if err != nil {
		return nil, nil, err
	}
	if err := os.MkdirAll(filepath.Join(root, "results"), 0o755); err != nil {
		return nil, nil, err
	}
	lock, err := os.OpenFile(filepath.Join(root, "run.lock"), os.O_CREATE|os.O_RDWR, 0o644)
	if err != nil {
		return nil, nil, err
	}
	if err := lockTerminalRunFile(lock); err != nil {
		_ = lock.Close()
		return nil, nil, cliError{"terminal_run_locked", "Another terminal worker owns this run directory.", []string{"Inspect it with eval terminal status, or choose a different --run-dir."}, map[string]any{"runDir": root}}
	}
	releaseLock := true
	defer func() {
		if releaseLock {
			_ = unlockTerminalRunFile(lock)
			_ = lock.Close()
		}
	}()
	identity := terminalLiveCheckpointIdentity(args, dataset, shardIndex, bundles, baseURL, callModel, hfID, quantization, quantFormat, hardware, cfg)
	fingerprint, fingerprintFields, err := terminalLiveCheckpointFingerprint(identity)
	if err != nil {
		return nil, nil, err
	}
	taskIDs := make([]string, len(bundles))
	for i := range bundles {
		taskIDs[i] = bundles[i].Task.ID
	}
	now := time.Now().UTC().Format(time.RFC3339)
	store := &terminalLiveCheckpointStore{root: root, lock: lock, state: terminalLiveCheckpoint{Version: terminalLiveCheckpointVersion, State: "running", Fingerprint: fingerprint, FingerprintFields: fingerprintFields, Dataset: firstNonEmpty(dataset, "local"), ShardIndex: shardIndex, TaskIDs: taskIDs, CreatedAt: now, UpdatedAt: now}}
	store.state.ThinkingLevel = cfg.thinkingLevel
	store.state.ThinkingSource = cfg.thinkingSource
	statePath := filepath.Join(root, "run.json")
	resuming := false
	if data, readErr := os.ReadFile(statePath); readErr == nil {
		if resume == "none" {
			return nil, nil, cliError{"checkpoint_exists", "The terminal run directory already contains a checkpoint.", []string{"Use --resume auto to continue it, or choose a new --run-dir for a fresh rerun."}, map[string]any{"runDir": root}}
		}
		var savedState terminalLiveCheckpoint
		if err := json.Unmarshal(data, &savedState); err != nil {
			return nil, nil, cliError{"checkpoint_invalid", "Could not decode the terminal run checkpoint.", []string{"Repair or replace run.json, or choose a new --run-dir."}, map[string]any{"path": statePath, "error": err.Error()}}
		}
		store.state = savedState
		if store.state.Version != terminalLiveCheckpointVersion || store.state.Fingerprint != fingerprint || !slicesEqual(store.state.TaskIDs, taskIDs) {
			details := map[string]any{"runDir": root, "expectedFingerprint": store.state.Fingerprint, "actualFingerprint": fingerprint}
			if changedFields := terminalChangedFingerprintFields(store.state.FingerprintFields, fingerprintFields); len(changedFields) > 0 {
				details["changedFields"] = changedFields
			}
			return nil, nil, cliError{"checkpoint_mismatch", "The terminal checkpoint does not match this run configuration and task manifest.", []string{"Resume with the exact original model, endpoint, options, hardware, dataset, and shard.", "Use a new --run-dir to rerun with different inputs."}, details}
		}
		if store.state.State == "submitted" {
			return nil, nil, cliError{"checkpoint_already_submitted", "This terminal checkpoint already has a successful submission receipt.", []string{"Read the saved result.json receipt.", "Use a new --run-dir to rerun and submit the same shard again intentionally."}, map[string]any{"runDir": root, "resultPath": filepath.Join(root, "result.json")}}
		}
		resuming = true
	} else if !errors.Is(readErr, os.ErrNotExist) {
		return nil, nil, readErr
	}
	resumed := make(map[string]terminalTaskResult)
	if resuming {
		var completed []string
		for _, bundle := range bundles {
			result, ok, err := loadTerminalLiveResult(store.root, bundle.Task.ID)
			if err != nil {
				return nil, nil, err
			}
			if !ok {
				continue
			}
			completed = append(completed, bundle.Task.ID)
			if result.scored {
				resumed[bundle.Task.ID] = result
			}
		}
		sort.Strings(completed)
		store.state.CompletedTasks = completed
		store.state.ActiveTasks = nil
	}
	store.state.State = "running"
	store.state.UpdatedAt = now
	if err := writeTerminalJSONAtomic(statePath, store.state); err != nil {
		return nil, nil, err
	}
	releaseLock = false
	return store, resumed, nil
}

func slicesEqual(left, right []string) bool {
	if len(left) != len(right) {
		return false
	}
	for i := range left {
		if left[i] != right[i] {
			return false
		}
	}
	return true
}

func terminalLiveResultPath(root, taskID string) string {
	sum := sha256.Sum256([]byte(taskID))
	name := sanitizeDockerName(taskID) + "-" + hex.EncodeToString(sum[:6]) + ".json"
	return filepath.Join(root, "results", name)
}

func terminalSavedResultFromTask(task terminalTask, result terminalTaskResult) terminalSavedResult {
	scored := result.scored
	var turns *int
	if !result.turnsUnreported {
		turns = &result.turns
	}
	return terminalSavedResult{QuestionID: task.ID, Pass: result.pass, Scored: &scored, Error: result.errText, ErrorCode: result.errCode, LatencyMs: result.wallTimeMs, WallTimeMs: result.wallTimeMs, TokenUsage: result.usage.toMap(), Turns: turns, Question: task.Instruction, Prompt: result.prompt, Response: result.transcript, VerifierOutput: result.verifierOutput}
}

func terminalTaskResultFromSaved(saved terminalSavedResult) terminalTaskResult {
	scored := saved.Scored != nil && *saved.Scored
	turns := 0
	if saved.Turns != nil {
		turns = *saved.Turns
	}
	return terminalTaskResult{pass: saved.Pass, scored: scored, turns: turns, turnsUnreported: saved.Turns == nil, transcript: saved.Response, verifierOutput: saved.VerifierOutput, wallTimeMs: firstPositiveInt64(saved.WallTimeMs, saved.LatencyMs), usage: tokenUsageFromObject(saved.TokenUsage), errText: saved.Error, errCode: saved.ErrorCode, instruction: saved.QuestionID, prompt: saved.Prompt}
}

func firstPositiveInt64(values ...int64) int64 {
	for _, value := range values {
		if value > 0 {
			return value
		}
	}
	return 0
}

func loadTerminalLiveResult(root, taskID string) (terminalTaskResult, bool, error) {
	resultPath := terminalLiveResultPath(root, taskID)
	data, err := os.ReadFile(resultPath)
	if errors.Is(err, os.ErrNotExist) {
		return terminalTaskResult{}, false, nil
	}
	if err != nil {
		return terminalTaskResult{}, false, err
	}
	var saved terminalSavedResult
	if err := json.Unmarshal(data, &saved); err != nil {
		return terminalTaskResult{}, false, cliError{"checkpoint_invalid", "Could not decode a terminal task checkpoint.", []string{"Repair the task result or use a new --run-dir."}, map[string]any{"path": resultPath, "error": err.Error()}}
	}
	if saved.QuestionID != taskID || saved.Scored == nil {
		return terminalTaskResult{}, false, cliError{"checkpoint_invalid", "A terminal task checkpoint has invalid identity or scoring metadata.", []string{"Restore the result written for this run, or use a new --run-dir."}, map[string]any{"path": resultPath, "expectedTaskId": taskID, "actualTaskId": saved.QuestionID}}
	}
	return terminalTaskResultFromSaved(saved), true, nil
}

func (store *terminalLiveCheckpointStore) close() {
	if store == nil || store.lock == nil {
		return
	}
	_ = unlockTerminalRunFile(store.lock)
	_ = store.lock.Close()
	store.lock = nil
}

func (store *terminalLiveCheckpointStore) startTask(taskID string, index int) error {
	if store == nil {
		return nil
	}
	store.mu.Lock()
	defer store.mu.Unlock()
	for _, active := range store.state.ActiveTasks {
		if active.TaskID == taskID {
			return nil
		}
	}
	store.state.ActiveTasks = append(store.state.ActiveTasks, terminalLiveActiveTask{TaskID: taskID, Index: index, StartedAt: time.Now().UTC().Format(time.RFC3339)})
	store.state.UpdatedAt = time.Now().UTC().Format(time.RFC3339)
	return writeTerminalJSONAtomic(filepath.Join(store.root, "run.json"), store.state)
}

func (store *terminalLiveCheckpointStore) removeActiveTask(taskID string) {
	active := store.state.ActiveTasks[:0]
	for _, item := range store.state.ActiveTasks {
		if item.TaskID != taskID {
			active = append(active, item)
		}
	}
	store.state.ActiveTasks = active
}

func (store *terminalLiveCheckpointStore) persistTask(task terminalTask, result terminalTaskResult) error {
	if store == nil {
		return nil
	}
	store.mu.Lock()
	defer store.mu.Unlock()
	store.removeActiveTask(task.ID)
	if err := writeTerminalJSONAtomic(terminalLiveResultPath(store.root, task.ID), terminalSavedResultFromTask(task, result)); err != nil {
		return err
	}
	found := false
	for _, taskID := range store.state.CompletedTasks {
		if taskID == task.ID {
			found = true
			break
		}
	}
	if !found {
		store.state.CompletedTasks = append(store.state.CompletedTasks, task.ID)
		sort.Strings(store.state.CompletedTasks)
	}
	store.state.UpdatedAt = time.Now().UTC().Format(time.RFC3339)
	return writeTerminalJSONAtomic(filepath.Join(store.root, "run.json"), store.state)
}

func (store *terminalLiveCheckpointStore) finish(state string, resultDocument map[string]any) error {
	if store == nil {
		return nil
	}
	store.mu.Lock()
	defer store.mu.Unlock()
	if resultDocument != nil {
		if err := writeTerminalJSONAtomic(filepath.Join(store.root, "result.json"), resultDocument); err != nil {
			return err
		}
	}
	store.state.State = state
	store.state.UpdatedAt = time.Now().UTC().Format(time.RFC3339)
	return writeTerminalJSONAtomic(filepath.Join(store.root, "run.json"), store.state)
}

func (store *terminalLiveCheckpointStore) interrupt() error {
	if store == nil {
		return nil
	}
	store.mu.Lock()
	defer store.mu.Unlock()
	store.state.State = "interrupted"
	store.state.ActiveTasks = nil
	store.state.UpdatedAt = time.Now().UTC().Format(time.RFC3339)
	return writeTerminalJSONAtomic(filepath.Join(store.root, "run.json"), store.state)
}

func writeTerminalJSONAtomic(path string, value any) error {
	data, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(dir, ".lmx-checkpoint-*")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()
	remove := true
	defer func() {
		if remove {
			_ = os.Remove(tmpPath)
		}
	}()
	if err := tmp.Chmod(0o644); err != nil {
		_ = tmp.Close()
		return err
	}
	if _, err := tmp.Write(data); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Sync(); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	if err := os.Rename(tmpPath, path); err != nil {
		return err
	}
	remove = false
	handle, err := os.Open(dir)
	if err != nil {
		return err
	}
	defer handle.Close()
	return handle.Sync()
}

type terminalTracePreviewStats struct {
	AssistantMessages int
	ToolExecutions    int
	UnknownEvents     int
	OversizedLines    int
	MalformedLines    int
	SectionsOmitted   int
}

const terminalBench21Dataset = "terminal-bench-2-1"
const terminalBench21ShardCount = 10

// terminalBench21CanonicalTaskIDsText is the exact sorted 89-task set from
// .terminal-smoke/omp-full-tb21-20260710-localmaxxing-cli/summary.json. Keeping
// the IDs compiled into the binary makes deferred validation independent of it.
const terminalBench21CanonicalTaskIDsText = `adaptive-rejection-sampler
bn-fit-modify
break-filter-js-from-html
build-cython-ext
build-pmars
build-pov-ray
caffe-cifar-10
cancel-async-tasks
chess-best-move
circuit-fibsqrt
cobol-modernization
code-from-image
compile-compcert
configure-git-webserver
constraints-scheduling
count-dataset-tokens
crack-7z-hash
custom-memory-heap-crash
db-wal-recovery
distribution-search
dna-assembly
dna-insert
extract-elf
extract-moves-from-video
feal-differential-cryptanalysis
feal-linear-cryptanalysis
filter-js-from-html
financial-document-processor
fix-code-vulnerability
fix-git
fix-ocaml-gc
gcode-to-text
git-leak-recovery
git-multibranch
gpt2-codegolf
headless-terminal
hf-model-inference
install-windows-3.11
kv-store-grpc
large-scale-text-editing
largest-eigenval
llm-inference-batching-scheduler
log-summary-date-ranges
mailman
make-doom-for-mips
make-mips-interpreter
mcmc-sampling-stan
merge-diff-arc-agi-task
model-extraction-relu-logits
modernize-scientific-stack
mteb-leaderboard
mteb-retrieve
multi-source-data-merger
nginx-request-logging
openssl-selfsigned-cert
overfull-hbox
password-recovery
path-tracing
path-tracing-reverse
polyglot-c-py
polyglot-rust-c
portfolio-optimization
protein-assembly
prove-plus-comm
pypi-server
pytorch-model-cli
pytorch-model-recovery
qemu-alpine-ssh
qemu-startup
query-optimize
raman-fitting
regex-chess
regex-log
reshard-c4-data
rstan-to-pystan
sam-cell-seg
sanitize-git-repo
schemelike-metacircular-eval
sparql-university
sqlite-db-truncate
sqlite-with-gcov
torch-pipeline-parallelism
torch-tensor-parallelism
train-fasttext
tune-mjcf
video-processing
vulnerable-secret
winning-avg-corewars
write-compressor`

var terminalBench21CanonicalTaskIDs = strings.Fields(terminalBench21CanonicalTaskIDsText)

const crudBenchDataset = "crud-bench"
const crudBenchShardCount = 5

const crudBenchCanonicalTaskIDsText = `create-bulk-atomic
create-default-fields
create-idempotency-key
create-parent-with-children
create-tenant-scoped-unique
customer-create-normalized-email
customer-patch-null-semantics
delete-cascade-children
delete-hard-unreferenced
delete-restore-conflict
delete-restrict-children
delete-soft-hide
delete-soft-idempotent
read-active-by-id
read-cursor-pagination
read-deterministic-sort
read-filter-status-tenant
read-offset-pagination
read-parent-child-aggregate
robust-concurrent-version
robust-reject-unknown-fields
robust-restart-persistence
robust-transaction-rollback
update-immutable-fields
update-optimistic-version
update-put-replacement
update-status-transition
update-unique-email-atomic
workflow-archive-tenant
workflow-audit-log
workflow-batch-status
workflow-cancel-restores-value
workflow-clone-with-children
workflow-customer-lifecycle
workflow-import-partial-report
workflow-legacy-migration
workflow-merge-duplicates
workflow-reserve-value
workflow-tenant-isolation
workflow-transfer-record`

var crudBenchCanonicalTaskIDs = strings.Fields(crudBenchCanonicalTaskIDsText)

type canonicalTerminalDataset struct {
	name       string
	shardCount int
	taskIDs    []string
}

func canonicalTerminalDatasetFor(slug string) (canonicalTerminalDataset, bool) {
	switch slug {
	case terminalBench21Dataset:
		return canonicalTerminalDataset{name: "Terminal-Bench 2.1", shardCount: terminalBench21ShardCount, taskIDs: terminalBench21CanonicalTaskIDs}, true
	case crudBenchDataset:
		return canonicalTerminalDataset{name: "CRUDbench", shardCount: crudBenchShardCount, taskIDs: crudBenchCanonicalTaskIDs}, true
	default:
		return canonicalTerminalDataset{}, false
	}
}

type terminalSubmissionRecord struct {
	questionID string
	pass       bool
	latencyMs  int64
	usage      terminalTokenUsage
	result     map[string]any
	artifact   map[string]any
}

// submitTerminalEval packages an already completed checkpoint. It deliberately
// does not share runTerminalEval's acquisition path: deferred submit must never
// contact a model endpoint, acquire tasks, start Docker, or rerun a verifier.
func submitTerminalEval(args cliArgs) error {
	runDir := positional(args, 3)
	if runDir == "" {
		return cliError{"missing_option", "eval terminal submit requires a completed run directory.", []string{"Run: lmx eval terminal submit <run-dir> --dataset <slug> --hf-id <org/model> --hardware hardware.json --dry-run."}, nil}
	}
	root, err := filepath.Abs(runDir)
	if err != nil {
		return err
	}
	info, err := os.Stat(root)
	if err != nil {
		return cliError{"checkpoint_missing", "Could not open the completed terminal run directory.", []string{"Pass the directory containing summary.json and one <task>.json file per task."}, map[string]any{"runDir": runDir, "error": err.Error()}}
	}
	if !info.IsDir() {
		return cliError{"checkpoint_invalid", "The terminal checkpoint path is not a directory.", []string{"Pass the directory containing summary.json and the per-task JSON files."}, map[string]any{"runDir": runDir}}
	}
	dataset := opt(args, "dataset")
	if dataset == "" {
		return cliError{"missing_dataset", "eval terminal submit requires --dataset <slug>.", []string{"Pass the terminal dataset slug used for the completed run."}, nil}
	}
	shardIndex, explicitShard, err := terminalSubmitShardIndex(args, dataset)
	if err != nil {
		return err
	}
	canonicalDataset, canonical := canonicalTerminalDatasetFor(dataset)
	if !canonical && !explicitShard {
		return cliError{"missing_shard_index", "Deferred submission for this dataset requires --shard-index <n>.", []string{"Pass the registered shard index for this already-isolated checkpoint.", "Automatic full-checkpoint partitioning is available for Terminal-Bench 2.1 and CRUDbench."}, map[string]any{"dataset": dataset}}
	}
	hfID := opt(args, "hf-id")
	if hfID == "" {
		return cliError{"missing_model", "eval terminal submit requires --hf-id <HuggingFace model id>.", []string{"Pass the canonical org/model identifier for the completed run."}, nil}
	}
	hardwarePath := opt(args, "hardware")
	if hardwarePath == "" {
		return cliError{"missing_hardware", "eval terminal submit requires --hardware hardware.json.", []string{"Run lmx hardware --out hardware.json and pass that saved hardware object."}, nil}
	}
	hardware, err := readJSON(hardwarePath)
	if err != nil {
		return err
	}

	entries, err := loadTerminalCheckpointSummary(filepath.Join(root, "summary.json"))
	if err != nil {
		return err
	}
	seenTasks := make(map[string]bool, len(entries))
	seenIndexes := make(map[int]bool, len(entries))
	for _, entry := range entries {
		if err := validateTerminalCheckpointEntry(entry, len(entries), seenTasks, seenIndexes); err != nil {
			return err
		}
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].Task < entries[j].Task })
	if canonical {
		if explicitShard {
			if terminalCheckpointHasCanonicalTaskSet(entries, canonicalDataset.taskIDs) {
				return cliError{"full_checkpoint_with_shard_index", fmt.Sprintf("A full canonical %s checkpoint cannot be labeled as one shard.", canonicalDataset.name), []string{"Remove --shard-index to partition the full checkpoint into canonical shards."}, map[string]any{"tasks": len(entries), "shardIndex": shardIndex}}
			}
			if err := validateCanonicalTerminalShardTaskSet(entries, canonicalDataset, shardIndex); err != nil {
				return err
			}
		} else if err := validateTerminalTaskSet(entries, canonicalDataset.taskIDs, "canonical "+canonicalDataset.name+" checkpoint", 0); err != nil {
			return err
		}
	}

	quantization, quantFormat, err := terminalCheckpointQuantization(entries)
	if err != nil {
		return err
	}
	if quantization != "" && opt(args, "quantization") == "" {
		return cliError{"missing_quantization", "The saved terminal run records a quantization; pass --quantization explicitly.", []string{"Pass --quantization " + quantization + " to confirm the saved run metadata."}, map[string]any{"savedQuantization": quantization}}
	}
	if quantFormat != "" && opt(args, "quant-format") == "" {
		return cliError{"missing_quant_format", "The saved terminal run records a quantization format; pass --quant-format explicitly.", []string{"Pass --quant-format " + quantFormat + " to confirm the saved run metadata."}, map[string]any{"savedQuantFormat": quantFormat}}
	}

	records := make([]terminalSubmissionRecord, 0, len(entries))
	seenResults := make(map[string]bool, len(entries))
	passed := 0
	var totalLatencyMs int64
	artifactBytes, maxArtifactBytes, traceCount, fallbackCount := 0, 0, 0, 0
	totalUsage := terminalTokenUsage{}
	previewTotals := terminalTracePreviewStats{}
	for _, entry := range entries {
		recordPath, err := terminalCheckpointResultPath(root, entry)
		if err != nil {
			return err
		}
		record, err := loadTerminalSavedResult(recordPath, entry.Task)
		if err != nil {
			return err
		}
		if seenResults[record.QuestionID] {
			return cliError{"duplicate_task_result", fmt.Sprintf("Task result %q appears more than once.", record.QuestionID), []string{"Ensure every task has exactly one unique per-task result file."}, map[string]any{"taskId": record.QuestionID, "file": recordPath}}
		}
		seenResults[record.QuestionID] = true
		if record.Pass != entry.Pass {
			return cliError{"checkpoint_result_mismatch", fmt.Sprintf("Task %q has conflicting pass values in summary.json and its result file.", entry.Task), []string{"Recover the matching summary.json and per-task files from the same completed run."}, map[string]any{"taskId": entry.Task, "summaryPass": entry.Pass, "resultPass": record.Pass}}
		}
		if record.Pass {
			passed++
		}
		latency := record.WallTimeMs
		if latency == 0 {
			latency = record.LatencyMs
		}
		usage := tokenUsageFromObject(record.TokenUsage)
		totalLatencyMs += latency
		totalUsage.add(usage)
		response, previewStats, usedTrace, err := terminalSavedArtifactResponse(root, recordPath, record)
		if err != nil {
			return cliError{"trace_read_failed", fmt.Sprintf("Could not package the OMP trace for task %q.", entry.Task), []string{"Check that the selected omp.jsonl is readable, or remove the broken trace to use the bounded saved response fallback."}, map[string]any{"taskId": entry.Task, "error": err.Error()}}
		}
		if usedTrace {
			traceCount++
		} else {
			fallbackCount++
		}
		previewTotals.AssistantMessages += previewStats.AssistantMessages
		previewTotals.ToolExecutions += previewStats.ToolExecutions
		previewTotals.UnknownEvents += previewStats.UnknownEvents
		previewTotals.OversizedLines += previewStats.OversizedLines
		previewTotals.MalformedLines += previewStats.MalformedLines
		previewTotals.SectionsOmitted += previewStats.SectionsOmitted
		artifactBytes += len(response)
		if len(response) > maxArtifactBytes {
			maxArtifactBytes = len(response)
		}
		records = append(records, terminalSubmissionRecord{
			questionID: record.QuestionID,
			pass:       record.Pass,
			latencyMs:  latency,
			usage:      usage,
			result:     map[string]any{"question_id": record.QuestionID, "pass": record.Pass},
			artifact: map[string]any{
				"question_id": record.QuestionID,
				"promptHash":  shortHash(record.QuestionID + ":" + record.Prompt),
				"question":    record.Question,
				"prompt":      record.Prompt,
				"response":    response,
				"score":       boolScore(record.Pass),
				"testPassed":  record.Pass,
				"latencyMs":   latency,
				"wallTimeMs":  latency,
				"tokenUsage":  record.TokenUsage,
			},
		})
	}
	if len(seenResults) != len(entries) {
		return cliError{"checkpoint_incomplete", "The terminal checkpoint did not produce one unique result for every summary record.", []string{"Restore the missing per-task JSON files and rerun deferred submit."}, map[string]any{"summaryRecords": len(entries), "uniqueResults": len(seenResults)}}
	}
	sort.Slice(records, func(i, j int) bool { return records[i].questionID < records[j].questionID })

	fullAccuracy := float64(passed) / float64(len(records))
	fullAvgLatencyMs := totalLatencyMs / int64(len(records))
	fullProvenance := map[string]any{"fullCheckpoint": false}
	if !explicitShard {
		fullTaskIDs := make([]string, len(records))
		for i := range records {
			fullTaskIDs[i] = records[i].questionID
		}
		fullTaskSetHash := sha256.Sum256([]byte(strings.Join(fullTaskIDs, "\n")))
		// These fullCheckpoint* keys preserve the source checkpoint aggregate on
		// every shard while the unprefixed runConfig metrics remain shard-local.
		fullProvenance = map[string]any{
			"fullCheckpoint":                    true,
			"fullCheckpointTasksRun":            len(records),
			"fullCheckpointAccuracy":            fullAccuracy,
			"fullCheckpointAvgLatencyMs":        fullAvgLatencyMs,
			"fullCheckpointTokenUsage":          totalUsage.toMap(),
			"fullCheckpointTaskSetSha256":       hex.EncodeToString(fullTaskSetHash[:]),
			"fullCheckpointCanonicalShardCount": canonicalDataset.shardCount,
		}
	}

	recordShards := [][]terminalSubmissionRecord{records}
	shardIndexes := []int{shardIndex}
	if !explicitShard {
		recordShards = partitionTerminalSubmissionRecords(records, canonicalDataset.shardCount)
		shardIndexes = make([]int, len(recordShards))
		for i := range shardIndexes {
			shardIndexes[i] = i + 1
		}
	}
	thinkingLevel, thinkingSource, err := resolveTerminalThinkingLevel(context.Background(), args, "", hfID, terminalConfig{agentCommand: "deferred-submit"}, os.Stdin, os.Stderr, terminalThinkingPromptAllowed(args))
	if err != nil {
		return err
	}
	if thinkingLevel == "not-provided" {
		return cliError{"missing_thinking_level", "Deferred terminal submission requires the thinking level used by the completed run.", []string{"Pass --thinking-level off|on|low|medium|high|xhigh|auto."}, map[string]any{"model": hfID}}
	}
	args.opts["thinking-level"] = thinkingLevel
	args.opts["thinking-level-source"] = thinkingSource
	payloads := make([]any, 0, len(recordShards))
	shardSizes := make([]int, 0, len(recordShards))
	for i, shardRecords := range recordShards {
		payload := terminalSubmissionPayload(args, hfID, hardware, shardIndexes[i], shardRecords, fullProvenance)
		payloads = append(payloads, payload)
		shardSizes = append(shardSizes, len(shardRecords))
	}
	batch := map[string]any{"dataset": dataset, "shards": payloads}
	batchJSON, err := json.Marshal(batch)
	if err != nil {
		return cliError{"payload_invalid", "Could not encode the deferred terminal submission batch.", []string{"Inspect the saved hardware and task result JSON values."}, err.Error()}
	}
	fields := map[string]any{
		"dataset":                dataset,
		"tasks":                  len(records),
		"uniqueTasks":            len(seenResults),
		"scored":                 len(records),
		"passing":                passed,
		"failing":                len(records) - passed,
		"accuracyPct":            roundMetric(fullAccuracy * 100),
		"payloadBytes":           len(batchJSON),
		"shardIndexes":           shardIndexes,
		"shardSizes":             shardSizes,
		"artifactBytes":          artifactBytes,
		"maxArtifactBytes":       maxArtifactBytes,
		"tracePreviews":          traceCount,
		"responseFallbacks":      fallbackCount,
		"unknownTraceEvents":     previewTotals.UnknownEvents,
		"oversizedTraceLines":    previewTotals.OversizedLines,
		"malformedTraceLines":    previewTotals.MalformedLines,
		"previewSectionsOmitted": previewTotals.SectionsOmitted,
	}
	if out := opt(args, "out"); out != "" {
		if err := writeJSON(out, batch); err != nil {
			return err
		}
		fields["payloadOut"] = out
	}
	if hasFlag(args, "dry-run") {
		printInfo(args, "terminal_submit_dry_run", fields)
		if humanOutput(args) {
			fmt.Println("Dry run only — payload batch validated locally; no network request was made.")
		}
		return nil
	}
	key := apiKey(args)
	if key == "" {
		return missingAPIKey("--api-key or LMX_API_KEY is required for eval terminal submit")
	}
	completed := make([]int, 0, len(payloads))
	runIDs := make([]any, 0, len(payloads))
	receipts := make([]any, 0, len(payloads))
	for i, rawPayload := range payloads {
		currentShard := shardIndexes[i]
		value, err := fetchJSON("POST", apiURL(args)+"/api/benchmarks/"+url.PathEscape(dataset)+"/submit", key, rawPayload)
		if err != nil {
			return cliError{"terminal_submit_shard_failed", fmt.Sprintf("Terminal submission stopped after shard %d failed.", currentShard), []string{"Fix the server error, then submit the remaining already-isolated shard checkpoints explicitly with --shard-index."}, map[string]any{"failedShardIndex": currentShard, "completedShardIndexes": completed, "error": err.Error()}}
		}
		completed = append(completed, currentShard)
		receipt := map[string]any{"shardIndex": currentShard}
		if obj := asObject(value); obj != nil {
			if run := asObject(obj["run"]); run != nil {
				receipt["runId"] = run["id"]
				receipt["status"] = run["status"]
				runIDs = append(runIDs, run["id"])
			}
			if aggregate := asObject(obj["aggregate"]); aggregate != nil {
				receipt["pooledScore"] = aggregate["pooledScore"]
				receipt["coverage"] = aggregate["shardsCovered"]
			}
		}
		receipts = append(receipts, receipt)
	}
	fields["submitted"] = len(records)
	fields["completedShardIndexes"] = completed
	fields["runIds"] = runIDs
	fields["shardReceipts"] = receipts
	printInfo(args, "terminal_submit_complete", fields)
	return nil
}

func terminalSubmitShardIndex(args cliArgs, dataset string) (int, bool, error) {
	raw, explicit := args.opts["shard-index"]
	if hasFlag(args, "shard-index") {
		return 0, true, cliError{"missing_option_value", "--shard-index requires a positive integer value.", []string{"Pass --shard-index 1, for example."}, nil}
	}
	if !explicit {
		return 0, false, nil
	}
	index, err := strconv.Atoi(raw)
	if err != nil || index < 1 {
		return 0, true, cliError{"invalid_shard_index", "--shard-index must be a positive integer.", nil, map[string]any{"value": raw}}
	}
	if canonical, ok := canonicalTerminalDatasetFor(dataset); ok && index > canonical.shardCount {
		return 0, true, cliError{"invalid_shard_index", fmt.Sprintf("--shard-index for %s must be between 1 and %d.", canonical.name, canonical.shardCount), nil, map[string]any{"value": index}}
	}
	return index, true, nil
}

func terminalCheckpointHasCanonicalTaskSet(entries []terminalCheckpointEntry, canonical []string) bool {
	if len(entries) != len(canonical) {
		return false
	}
	for i := range canonical {
		if entries[i].Task != canonical[i] {
			return false
		}
	}
	return true
}

func validateCanonicalTerminalShardTaskSet(entries []terminalCheckpointEntry, canonical canonicalTerminalDataset, shardIndex int) error {
	start := ((shardIndex - 1) * len(canonical.taskIDs)) / canonical.shardCount
	end := (shardIndex * len(canonical.taskIDs)) / canonical.shardCount
	return validateTerminalTaskSet(entries, canonical.taskIDs[start:end], fmt.Sprintf("canonical %s shard %d", canonical.name, shardIndex), shardIndex)
}

func validateTerminalTaskSet(entries []terminalCheckpointEntry, expected []string, label string, shardIndex int) error {
	actual := make(map[string]bool, len(entries))
	for _, entry := range entries {
		actual[entry.Task] = true
	}
	wanted := make(map[string]bool, len(expected))
	missing := make([]string, 0)
	for _, id := range expected {
		wanted[id] = true
		if !actual[id] {
			missing = append(missing, id)
		}
	}
	extra := make([]string, 0)
	for id := range actual {
		if !wanted[id] {
			extra = append(extra, id)
		}
	}
	sort.Strings(extra)
	if len(missing) == 0 && len(extra) == 0 && len(entries) == len(expected) {
		return nil
	}
	details := map[string]any{"expectedTasks": len(expected), "actualTasks": len(entries), "missingTaskIds": missing, "extraTaskIds": extra}
	if shardIndex > 0 {
		details["shardIndex"] = shardIndex
	}
	return cliError{"checkpoint_task_set_mismatch", fmt.Sprintf("The checkpoint task IDs do not exactly match the %s task set.", label), []string{"Use the checkpoint produced from the canonical task bundles; task counts alone are not sufficient."}, details}
}

func partitionTerminalSubmissionRecords(records []terminalSubmissionRecord, count int) [][]terminalSubmissionRecord {
	shards := make([][]terminalSubmissionRecord, count)
	for i := range count {
		start := (i * len(records)) / count
		end := ((i + 1) * len(records)) / count
		shards[i] = records[start:end]
	}
	return shards
}

func terminalSubmissionPayload(args cliArgs, hfID string, hardware any, shardIndex int, records []terminalSubmissionRecord, fullProvenance map[string]any) map[string]any {
	results := make([]any, len(records))
	artifacts := make([]any, len(records))
	passed := 0
	var latencyMs int64
	usage := terminalTokenUsage{}
	for i, record := range records {
		results[i] = record.result
		artifact := make(map[string]any, len(record.artifact)+1)
		for key, value := range record.artifact {
			artifact[key] = value
		}
		artifact["itemIndex"] = i
		artifacts[i] = artifact
		if record.pass {
			passed++
		}
		latencyMs += record.latencyMs
		usage.add(record.usage)
	}
	runConfig := map[string]any{
		"accuracy":            float64(passed) / float64(len(records)),
		"tasksRun":            len(records),
		"errors":              0,
		"avgLatencyMs":        latencyMs / int64(len(records)),
		"protocol":            "deferred-saved-terminal-run",
		"agent":               firstNonEmpty(opt(args, "agent-name"), "external-agent"),
		"deferredSubmit":      true,
		"tokenUsage":          usage.toMap(),
		"thinkingLevel":       opt(args, "thinking-level"),
		"thinkingLevelSource": firstNonEmpty(opt(args, "thinking-level-source"), "cli"),
	}
	for key, value := range fullProvenance {
		runConfig[key] = value
	}
	payload := map[string]any{
		"hfId":          hfID,
		"modelRevision": firstNonEmpty(opt(args, "model-revision"), "main"),
		"hardware":      normalizeHardwarePayload(hardware),
		"shardIndex":    shardIndex,
		"results":       results,
		"artifacts":     artifacts,
		"runnerVersion": firstNonEmpty(opt(args, "runner-version"), "localmaxxing-go terminal-agent"),
		"runConfig":     runConfig,
	}
	if value := opt(args, "quantization"); value != "" {
		payload["quantization"] = value
	}
	if value := opt(args, "quant-format"); value != "" {
		payload["quantFormat"] = value
	}
	if value := opt(args, "notes"); value != "" {
		payload["notes"] = value
	}
	return payload
}

func loadTerminalCheckpointSummary(path string) ([]terminalCheckpointEntry, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, cliError{"checkpoint_summary_missing", "The terminal checkpoint is missing summary.json.", []string{"Pass the completed run directory containing summary.json."}, map[string]any{"path": path, "error": err.Error()}}
	}
	defer file.Close()
	var entries []terminalCheckpointEntry
	if err := json.NewDecoder(file).Decode(&entries); err != nil {
		return nil, cliError{"checkpoint_summary_invalid", "Could not decode terminal checkpoint summary.json as an array.", []string{"Use the summary.json written by the completed checkpoint runner."}, map[string]any{"path": path, "error": err.Error()}}
	}
	if len(entries) == 0 {
		return nil, cliError{"checkpoint_summary_empty", "Terminal checkpoint summary.json contains no task records.", []string{"Pass a completed run directory with at least one scored task."}, map[string]any{"path": path}}
	}
	return entries, nil
}

func validateTerminalCheckpointEntry(entry terminalCheckpointEntry, count int, seenTasks map[string]bool, seenIndexes map[int]bool) error {
	if entry.Task == "" {
		return cliError{"checkpoint_task_missing", "A summary.json record is missing its task id.", []string{"Every summary record must contain a non-empty task field."}, map[string]any{"index": entry.Index}}
	}
	if filepath.Base(entry.Task) != entry.Task || entry.Task == "." || entry.Task == ".." {
		return cliError{"checkpoint_task_invalid", fmt.Sprintf("Summary task id %q is not a safe task filename.", entry.Task), []string{"Task ids must not contain path separators or traversal components."}, nil}
	}
	if seenTasks[entry.Task] {
		return cliError{"duplicate_summary_task", fmt.Sprintf("summary.json contains duplicate task %q.", entry.Task), []string{"Keep exactly one summary record per task."}, map[string]any{"taskId": entry.Task}}
	}
	seenTasks[entry.Task] = true
	if entry.Scored == nil {
		return cliError{"checkpoint_score_missing", fmt.Sprintf("Summary record %q is missing scored.", entry.Task), []string{"Every deferred terminal task must explicitly record scored: true."}, nil}
	}
	if !*entry.Scored {
		return cliError{"checkpoint_unscored", fmt.Sprintf("Summary record %q was not scored.", entry.Task), []string{"Deferred submit only accepts complete checkpoints where every task has verifier scoring."}, nil}
	}
	if entry.Total > 0 && entry.Total != count {
		return cliError{"checkpoint_total_mismatch", fmt.Sprintf("Summary record %q declares total %d, but summary.json contains %d records.", entry.Task, entry.Total, count), []string{"Recover all per-task records from the same completed run."}, nil}
	}
	if entry.Index > 0 {
		if seenIndexes[entry.Index] {
			return cliError{"duplicate_summary_index", fmt.Sprintf("summary.json contains duplicate index %d.", entry.Index), []string{"Every checkpoint index must identify one task."}, nil}
		}
		seenIndexes[entry.Index] = true
	}
	return nil
}

func terminalCheckpointQuantization(entries []terminalCheckpointEntry) (string, string, error) {
	quantization, quantFormat := "", ""
	for _, entry := range entries {
		q := stringValue(entry.Summary["quantization"])
		f := stringValue(entry.Summary["quantFormat"])
		if quantization != "" && q != "" && q != quantization {
			return "", "", cliError{"checkpoint_metadata_mismatch", "Checkpoint task summaries contain conflicting quantization values.", []string{"Recover summary.json from a single completed run."}, map[string]any{"first": quantization, "taskId": entry.Task, "value": q}}
		}
		if quantFormat != "" && f != "" && f != quantFormat {
			return "", "", cliError{"checkpoint_metadata_mismatch", "Checkpoint task summaries contain conflicting quantization formats.", []string{"Recover summary.json from a single completed run."}, map[string]any{"first": quantFormat, "taskId": entry.Task, "value": f}}
		}
		if quantization == "" {
			quantization = q
		}
		if quantFormat == "" {
			quantFormat = f
		}
	}
	return quantization, quantFormat, nil
}

func terminalCheckpointResultPath(root string, entry terminalCheckpointEntry) (string, error) {
	direct := filepath.Join(root, entry.Task+".json")
	if info, err := os.Stat(direct); err == nil && !info.IsDir() {
		return direct, nil
	}
	if entry.Out != "" {
		candidate := entry.Out
		if !filepath.IsAbs(candidate) {
			candidate = filepath.Join(root, candidate)
		}
		candidate = filepath.Clean(candidate)
		rel, relErr := filepath.Rel(root, candidate)
		if relErr == nil && rel != ".." && !strings.HasPrefix(rel, ".."+string(os.PathSeparator)) {
			if info, statErr := os.Stat(candidate); statErr == nil && !info.IsDir() {
				return candidate, nil
			}
		}
	}
	return "", cliError{"task_result_missing", fmt.Sprintf("Missing per-task result file for %q.", entry.Task), []string{"Restore " + entry.Task + ".json beneath the completed run directory."}, map[string]any{"taskId": entry.Task, "expected": direct, "summaryOut": entry.Out}}
}

func loadTerminalSavedResult(path, taskID string) (terminalSavedResult, error) {
	file, err := os.Open(path)
	if err != nil {
		return terminalSavedResult{}, err
	}
	defer file.Close()
	var saved terminalSavedTaskFile
	if err := json.NewDecoder(file).Decode(&saved); err != nil {
		return terminalSavedResult{}, cliError{"task_result_invalid", fmt.Sprintf("Could not decode result file for task %q.", taskID), []string{"Use the per-task JSON written by the completed terminal run."}, map[string]any{"path": path, "error": err.Error()}}
	}
	if len(saved.Results) != 1 {
		return terminalSavedResult{}, cliError{"task_result_count_invalid", fmt.Sprintf("Task file %q contains %d results; expected exactly one.", path, len(saved.Results)), []string{"Keep exactly one result record in each <task>.json file."}, map[string]any{"taskId": taskID, "results": len(saved.Results)}}
	}
	record := saved.Results[0]
	if record.QuestionID == "" {
		return terminalSavedResult{}, cliError{"task_result_id_missing", fmt.Sprintf("Task file %q has no question_id.", path), []string{"Every saved terminal result must identify its task."}, nil}
	}
	if record.QuestionID != taskID {
		return terminalSavedResult{}, cliError{"task_result_id_mismatch", fmt.Sprintf("Task file for %q contains question_id %q.", taskID, record.QuestionID), []string{"Restore the matching per-task JSON file from the completed run."}, map[string]any{"path": path}}
	}
	if record.Scored == nil {
		return terminalSavedResult{}, cliError{"task_result_score_missing", fmt.Sprintf("Task result %q is missing scored.", taskID), []string{"Every deferred task result must explicitly record scored: true."}, nil}
	}
	if !*record.Scored {
		return terminalSavedResult{}, cliError{"task_result_unscored", fmt.Sprintf("Task result %q was not scored.", taskID), []string{"Deferred submit only accepts results completed by the verifier."}, nil}
	}
	return record, nil
}

func loadTerminalSavedResponse(path string) (string, error) {
	file, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer file.Close()
	var saved struct {
		Results []struct {
			Response string `json:"response"`
		} `json:"results"`
	}
	if err := json.NewDecoder(file).Decode(&saved); err != nil {
		return "", err
	}
	if len(saved.Results) != 1 {
		return "", fmt.Errorf("saved result contains %d records; expected one", len(saved.Results))
	}
	return saved.Results[0].Response, nil
}

func terminalBoolField(obj map[string]any, key string) bool {
	value, _ := obj[key].(bool)
	return value
}

func findTerminalOMPTrace(root, taskID string) string {
	patterns := []string{
		filepath.Join(root, "traces", taskID, taskID, "agent", "*", "omp.jsonl"),
		filepath.Join(root, "traces", taskID, "agent", "*", "omp.jsonl"),
		filepath.Join(root, taskID, taskID, "agent", "*", "omp.jsonl"),
	}
	var matches []string
	for _, pattern := range patterns {
		found, _ := filepath.Glob(pattern)
		matches = append(matches, found...)
	}
	sort.Strings(matches)
	if len(matches) == 0 {
		return ""
	}
	return matches[len(matches)-1]
}

func terminalSavedArtifactResponse(root, recordPath string, record terminalSavedResult) (string, terminalTracePreviewStats, bool, error) {
	tracePath := findTerminalOMPTrace(root, record.QuestionID)
	stats := terminalTracePreviewStats{}
	preview := ""
	usedTrace := tracePath != ""
	if usedTrace {
		var err error
		preview, stats, err = buildTerminalOMPPreview(tracePath, root)
		if err != nil {
			return "", stats, true, err
		}
	} else {
		savedResponse, err := loadTerminalSavedResponse(recordPath)
		if err != nil {
			return "", stats, false, err
		}
		preview = "# Agent trace\n\nSource: saved task response (no omp.jsonl trace was found).\n\n## Final answer\n\n" + terminalMarkdownCode(boundedTerminalUTF8(savedResponse, terminalTracePreviewBytes-128, "saved response"))
	}
	if strings.TrimSpace(record.VerifierOutput) != "" {
		verifier := boundedTerminalUTF8(record.VerifierOutput, terminalVerifierPreviewBytes, "verifier output")
		preview += "\n\n## Verifier\n\nSource: saved verifierOutput.\n\n" + terminalMarkdownCode(verifier)
	}
	return boundedTerminalUTF8(preview, terminalArtifactResponseBytes, "artifact response"), stats, usedTrace, nil
}

type terminalPreviewAccumulator struct {
	head         strings.Builder
	tail         []string
	tailBytes    int
	total        int
	headSections int
}

func (a *terminalPreviewAccumulator) add(section string) {
	section = boundedTerminalUTF8(section, 1800, "trace section")
	a.total++
	if a.head.Len()+len(section) <= 11*1024 {
		a.head.WriteString(section)
		a.headSections++
		return
	}
	a.tail = append(a.tail, section)
	a.tailBytes += len(section)
	for a.tailBytes > 9*1024 && len(a.tail) > 1 {
		a.tailBytes -= len(a.tail[0])
		a.tail = a.tail[1:]
	}
}

func (a *terminalPreviewAccumulator) render(header, summary string) (string, int) {
	var b strings.Builder
	b.WriteString(header)
	b.WriteString(a.head.String())
	omitted := a.total - a.headSections - len(a.tail)
	if omitted > 0 {
		b.WriteString(fmt.Sprintf("\n## Preview truncation\n\n%d middle trace sections omitted by the bounded inline preview.\n\n", omitted))
	}
	for _, section := range a.tail {
		b.WriteString(section)
	}
	b.WriteString(summary)
	return boundedTerminalUTF8(b.String(), terminalTracePreviewBytes, "agent trace preview"), omitted
}

func buildTerminalOMPPreview(tracePath, root string) (string, terminalTracePreviewStats, error) {
	file, err := os.Open(tracePath)
	if err != nil {
		return "", terminalTracePreviewStats{}, err
	}
	defer file.Close()
	reader := bufio.NewReaderSize(file, 128*1024)
	acc := terminalPreviewAccumulator{}
	stats := terminalTracePreviewStats{}
	eventCounts := map[string]int{}
	unknownCounts := map[string]int{}
	lastAssistant := ""
	for {
		line, oversized, readErr := readBoundedTerminalTraceLine(reader, terminalTraceLineBytes)
		if oversized {
			stats.OversizedLines++
		}
		if len(line) > 0 && !oversized {
			var envelope struct {
				Type string `json:"type"`
			}
			if err := json.Unmarshal(line, &envelope); err != nil {
				stats.MalformedLines++
			} else {
				typeName := envelope.Type
				if typeName == "" {
					typeName = "(missing type)"
				}
				eventCounts[typeName]++
				switch typeName {
				case "message_end":
					var event map[string]any
					if json.Unmarshal(line, &event) == nil {
						message := asObject(event["message"])
						if stringValue(message["role"]) == "assistant" {
							if text := terminalMessageText(message); text != "" {
								if lastAssistant != "" {
									acc.add("## Assistant\n\n" + terminalMarkdownCode(lastAssistant) + "\n")
								}
								lastAssistant = text
								stats.AssistantMessages++
							}
						}
					}
				case "tool_execution_end":
					var event map[string]any
					if json.Unmarshal(line, &event) == nil {
						stats.ToolExecutions++
						acc.add(terminalToolExecutionSection(event))
					}
				case "notice", "auto_compaction_start", "auto_compaction_end":
					var event map[string]any
					if json.Unmarshal(line, &event) == nil {
						acc.add(terminalLifecycleSection(typeName, event))
					}
				case "trace_filter_summary":
					var event map[string]any
					if json.Unmarshal(line, &event) == nil {
						acc.add(terminalTraceFilterSection(event))
					}
				case "session", "agent_start", "agent_end", "turn_start", "turn_end", "message_start", "message_update", "tool_execution_start", "tool_execution_update":
					// Counted for the lifecycle summary; streaming deltas are optional.
				default:
					unknownCounts[typeName]++
					stats.UnknownEvents++
				}
			}
		}
		if readErr != nil {
			if errors.Is(readErr, io.EOF) {
				break
			}
			return "", stats, readErr
		}
	}
	if lastAssistant != "" {
		acc.add("## Final answer\n\n" + terminalMarkdownCode(lastAssistant) + "\n")
	}
	rel, relErr := filepath.Rel(root, tracePath)
	if relErr != nil {
		rel = filepath.Base(tracePath)
	}
	header := "# Agent trace\n\nSource: `" + strings.ReplaceAll(terminalSingleLine(rel, 500), "`", "'") + "` (stream-parsed; raw JSONL is not embedded).\n\n"
	unknownNote := terminalUnknownEventNote(unknownCounts)
	summary := fmt.Sprintf("\n## Trace integrity\n\nFinalized assistant messages: %d  \nCompleted tool executions: %d  \nTurns started: %d  \nStreaming message deltas observed (not required): %d  \nOversized lines skipped: %d  \nMalformed lines skipped: %d%s\n", stats.AssistantMessages, stats.ToolExecutions, eventCounts["turn_start"], eventCounts["message_update"], stats.OversizedLines, stats.MalformedLines, unknownNote)
	preview, omitted := acc.render(header, summary)
	stats.SectionsOmitted = omitted
	return preview, stats, nil
}

func readBoundedTerminalTraceLine(reader *bufio.Reader, maxBytes int) ([]byte, bool, error) {
	line := make([]byte, 0, 4096)
	oversized := false
	for {
		fragment, isPrefix, err := reader.ReadLine()
		if !oversized {
			if len(line)+len(fragment) > maxBytes {
				oversized = true
				line = line[:0]
			} else {
				line = append(line, fragment...)
			}
		}
		if !isPrefix {
			return line, oversized, err
		}
		if err != nil {
			return line, oversized, err
		}
	}
}

func terminalMessageText(message map[string]any) string {
	if text := stringValue(message["content"]); text != "" {
		return boundedTerminalUTF8(text, 1600, "assistant message")
	}
	parts := []string{}
	for _, raw := range anySlice(message["content"]) {
		part := asObject(raw)
		if part == nil || stringValue(part["type"]) != "text" {
			continue
		}
		if text := stringValue(part["text"]); text != "" {
			parts = append(parts, text)
		}
	}
	return boundedTerminalUTF8(strings.Join(parts, "\n"), 1600, "assistant message")
}

func terminalToolExecutionSection(event map[string]any) string {
	name := terminalSingleLine(firstNonEmpty(stringValue(event["toolName"]), "unknown"), 80)
	intent := terminalSingleLine(stringValue(event["intent"]), 200)
	var b strings.Builder
	b.WriteString("## Tool activity\n\nTool: ")
	b.WriteString(name)
	b.WriteString("\n\n")
	if intent != "" {
		b.WriteString("Intent: ")
		b.WriteString(intent)
		b.WriteString("\n\n")
	}
	if args := terminalToolArgsText(asObject(event["args"])); args != "" {
		b.WriteString("Arguments:\n\n")
		b.WriteString(terminalMarkdownCode(args))
		b.WriteString("\n")
	}
	result := asObject(event["result"])
	outcome := terminalContentText(result["content"])
	if outcome == "" {
		outcome = firstNonEmpty(stringValue(result["text"]), stringValue(event["error"]), "(completed with no text output)")
	}
	if terminalBoolField(event, "isError") {
		b.WriteString("Outcome: error\n\n")
	} else {
		b.WriteString("Outcome: completed\n\n")
	}
	b.WriteString(terminalMarkdownCode(boundedTerminalUTF8(outcome, 1100, "tool outcome")))
	b.WriteString("\n")
	return b.String()
}

func terminalToolArgsText(args map[string]any) string {
	if args == nil {
		return ""
	}
	keys := make([]string, 0, len(args))
	for key := range args {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	lines := make([]string, 0, len(keys))
	for _, key := range keys {
		value := ""
		switch typed := args[key].(type) {
		case string:
			value = typed
		case float64, bool, nil:
			value = fmt.Sprint(typed)
		default:
			value = "(" + fmt.Sprintf("%T", typed) + ")"
		}
		lines = append(lines, terminalSingleLine(key, 80)+": "+boundedTerminalUTF8(value, 500, "tool argument"))
	}
	return boundedTerminalUTF8(strings.Join(lines, "\n"), 700, "tool arguments")
}

func terminalContentText(value any) string {
	if text := stringValue(value); text != "" {
		return text
	}
	parts := []string{}
	for _, raw := range anySlice(value) {
		part := asObject(raw)
		if part == nil {
			continue
		}
		if text := firstNonEmpty(stringValue(part["text"]), stringValue(part["content"])); text != "" {
			parts = append(parts, text)
		}
	}
	return strings.Join(parts, "\n")
}

func terminalLifecycleSection(typeName string, event map[string]any) string {
	label := strings.ReplaceAll(typeName, "_", " ")
	text := firstNonEmpty(stringValue(event["message"]), stringValue(event["reason"]), stringValue(event["action"]))
	if result := asObject(event["result"]); result != nil {
		text = firstNonEmpty(stringValue(result["shortSummary"]), stringValue(result["summary"]), text)
	}
	return "## Lifecycle: " + terminalSingleLine(label, 80) + "\n\n" + terminalMarkdownCode(boundedTerminalUTF8(text, 1400, "lifecycle event")) + "\n"
}

func terminalTraceFilterSection(event map[string]any) string {
	stored := int64(numberField(event, "storedBytes"))
	overflow := int64(numberField(event, "overflowBytes"))
	dropped := asObject(event["droppedEvents"])
	keys := make([]string, 0, len(dropped))
	for key := range dropped {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	parts := make([]string, 0, len(keys))
	for _, key := range keys {
		parts = append(parts, fmt.Sprintf("%s=%d", terminalSingleLine(key, 80), int64(numberField(dropped, key))))
	}
	return fmt.Sprintf("## Trace integrity\n\nStored bytes: %d  \nDropped streaming events: %s  \nOverflow bytes: %d\n\n", stored, firstNonEmpty(strings.Join(parts, ", "), "none"), overflow)
}

func terminalUnknownEventNote(counts map[string]int) string {
	if len(counts) == 0 {
		return ""
	}
	keys := make([]string, 0, len(counts))
	for key := range counts {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	if len(keys) > 20 {
		keys = keys[:20]
	}
	parts := make([]string, 0, len(keys))
	for _, key := range keys {
		parts = append(parts, fmt.Sprintf("%s=%d", terminalSingleLine(key, 80), counts[key]))
	}
	return "  \nUnknown event types ignored: " + strings.Join(parts, ", ")
}

func terminalMarkdownCode(value string) string {
	value = strings.ReplaceAll(value, "\r\n", "\n")
	value = strings.ReplaceAll(value, "\r", "\n")
	return "    " + strings.ReplaceAll(value, "\n", "\n    ") + "\n"
}

func terminalSingleLine(value string, maxBytes int) string {
	value = strings.Join(strings.Fields(value), " ")
	return boundedTerminalUTF8(value, maxBytes, "label")
}

// boundedTerminalUTF8 returns at most maxBytes, including its truncation note,
// while keeping both useful beginning and ending context on rune boundaries.
func boundedTerminalUTF8(value string, maxBytes int, label string) string {
	if maxBytes <= 0 {
		return ""
	}
	if len(value) <= maxBytes {
		return strings.ToValidUTF8(value, "�")
	}
	marker := fmt.Sprintf("\n...[truncated %s; %d bytes omitted]...\n", label, len(value)-maxBytes)
	if len(marker) >= maxBytes {
		return terminalUTF8Prefix(marker, maxBytes)
	}
	available := maxBytes - len(marker)
	headBytes := available * 2 / 3
	tailBytes := available - headBytes
	head := terminalUTF8Prefix(value, headBytes)
	tail := terminalUTF8Suffix(value, tailBytes)
	omitted := len(value) - len(head) - len(tail)
	marker = fmt.Sprintf("\n...[truncated %s; %d bytes omitted]...\n", label, omitted)
	for len(head)+len(marker)+len(tail) > maxBytes && len(head) > 0 {
		head = terminalUTF8Prefix(head, len(head)-1)
	}
	return head + marker + tail
}

func terminalUTF8Prefix(value string, maxBytes int) string {
	if len(value) <= maxBytes {
		return strings.ToValidUTF8(value, "�")
	}
	end := maxBytes
	for end > 0 && !utf8.ValidString(value[:end]) {
		end--
	}
	return strings.ToValidUTF8(value[:end], "�")
}

func terminalUTF8Suffix(value string, maxBytes int) string {
	if len(value) <= maxBytes {
		return strings.ToValidUTF8(value, "�")
	}
	start := len(value) - maxBytes
	for start < len(value) && !utf8.RuneStart(value[start]) {
		start++
	}
	return strings.ToValidUTF8(value[start:], "�")
}

func detectTerminalServedModel(ctx context.Context, baseURL, apiKey, preferred string) (string, map[string]any, error) {
	value, err := fetchJSONContext(ctx, http.MethodGet, openAIBaseURL(baseURL)+"/v1/models", apiKey, nil)
	if err != nil {
		return "", nil, err
	}
	if ctx.Err() != nil {
		return "", nil, terminalCancelledError(ctx)
	}
	first := ""
	var firstInfo map[string]any
	for _, item := range modelInfoItems(asObject(value)) {
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

func fetchTerminalManifestItems(ctx context.Context, args cliArgs, dataset string) ([]map[string]any, int, error) {
	params := url.Values{}
	if s := opt(args, "shard"); s != "" {
		params.Set("shard", s)
	}
	if q := opt(args, "questions"); q != "" {
		params.Set("questions", q)
	}
	metaURL := apiURL(args) + "/api/benchmarks/" + url.PathEscape(dataset) + "/shard"
	if enc := params.Encode(); enc != "" {
		metaURL += "?" + enc
	}
	meta, err := fetchJSONContext(ctx, "GET", metaURL, apiKey(args), nil)
	if err != nil {
		if ctx.Err() != nil {
			return nil, -1, terminalCancelledError(ctx)
		}
		var apiErr cliError
		if errors.As(err, &apiErr) && apiErr.Code == "cli_upgrade_required" {
			return nil, -1, apiErr
		}
		return nil, -1, cliError{"manifest_fetch_failed", fmt.Sprintf("Could not fetch terminal dataset manifest: %v", err), []string{"Check the dataset slug and API URL.", "Confirm the dataset is approved and eval storage is configured."}, map[string]any{"url": metaURL, "error": err.Error()}}
	}
	if ctx.Err() != nil {
		return nil, -1, terminalCancelledError(ctx)
	}
	shardIndex := -1
	if sh := asObject(asObject(meta)["shard"]); sh != nil {
		if f, ok := sh["shardIndex"].(float64); ok {
			shardIndex = int(f)
		}
	}
	downloadURL := stringValue(asObject(meta)["downloadUrl"])
	if downloadURL == "" {
		return nil, shardIndex, cliError{"manifest_fetch_failed", "Dataset shard response did not include downloadUrl.", []string{"Confirm the dataset is approved and eval storage is configured."}, meta}
	}
	items, err := fetchTerminalDatasetItems(ctx, downloadURL)
	if err != nil {
		if ctx.Err() != nil {
			return nil, shardIndex, terminalCancelledError(ctx)
		}
		return nil, shardIndex, cliError{"manifest_fetch_failed", fmt.Sprintf("Could not download terminal manifest JSONL: %v", err), []string{"Signed manifest URLs expire after 15 minutes; re-run the command.", "Check network access to the storage host."}, map[string]any{"downloadUrl": downloadURL}}
	}
	if q := opt(args, "questions"); q != "" {
		limit, err := strconv.Atoi(q)
		if err != nil || limit < 1 {
			return nil, shardIndex, cliError{"manifest_fetch_failed", "--questions must be a positive integer.", []string{"Pass --questions <n> to run the first n manifest rows, or use --task <id>[,<id>...] to filter task ids."}, map[string]any{"questions": q}}
		}
		if limit < len(items) {
			items = items[:limit]
		}
	}
	requested := parseStringSet(opt(args, "task"))
	if len(requested) > 0 {
		filtered := make([]map[string]any, 0, len(requested))
		for _, row := range items {
			id := firstNonEmpty(stringValue(row["question_id"]), stringValue(row["questionId"]), stringValue(row["id"]))
			if requested[id] {
				filtered = append(filtered, row)
			}
		}
		items = filtered
	}
	return items, shardIndex, nil
}

func fetchTerminalDatasetItems(ctx context.Context, rawURL string) ([]map[string]any, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, rawURL, nil)
	if err != nil {
		return nil, err
	}
	res, err := apiHTTPClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	text, err := io.ReadAll(res.Body)
	if err != nil {
		return nil, err
	}
	if res.StatusCode < http.StatusOK || res.StatusCode >= http.StatusMultipleChoices {
		return nil, fmt.Errorf("dataset download failed: %s", res.Status)
	}
	return parseDatasetItems(string(text), rawURL, "jsonl")
}

func acquireTerminalBundles(ctx context.Context, args cliArgs, dataset, localDir string) ([]terminalBundle, string, int, error) {
	if localDir != "" {
		bundles, err := loadTerminalBundles(localDir)
		return bundles, "", -1, err
	}
	items, shardIndex, err := fetchTerminalManifestItems(ctx, args, dataset)
	if err != nil {
		return nil, "", shardIndex, err
	}
	tmp, err := os.MkdirTemp("", "lmx-terminal-bundles-*")
	if err != nil {
		return nil, "", shardIndex, err
	}
	bundles := []terminalBundle{}
	for _, row := range items {
		if ctx.Err() != nil {
			return nil, tmp, shardIndex, terminalCancelledError(ctx)
		}
		id := firstNonEmpty(stringValue(row["question_id"]), stringValue(row["questionId"]), stringValue(row["id"]))
		key := stringValue(row["bundle_key"])
		if key == "" {
			key = stringValue(row["bundleKey"])
		}
		if key == "" {
			return nil, tmp, shardIndex, cliError{"manifest_fetch_failed", "Terminal manifest row is missing bundle_key.", []string{"Re-ingest the terminal dataset manifest."}, row}
		}
		bundleDir, err := downloadTerminalBundle(ctx, args, tmp, id, key, stringValue(row["sha256"]))
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

func terminalHarnessKey(args cliArgs, cfg terminalConfig) string {
	agentName := "lmx-terminus"
	if cfg.agentCommand != "" {
		agentName = firstNonEmpty(opt(args, "agent-name"), opt(args, "agent"), "external-agent")
	}
	return firstNonEmpty(opt(args, "runner-version"), "localmaxxing-go terminal-agent") + "|" + terminalProtocolLabel(cfg) + "|" + agentName
}

func terminalArgsWithOption(args cliArgs, name, value string) cliArgs {
	copied := cliArgs{opts: map[string]string{}, flags: map[string]bool{}, positional: append([]string(nil), args.positional...)}
	for k, v := range args.opts {
		copied.opts[k] = v
	}
	for k, v := range args.flags {
		copied.flags[k] = v
	}
	copied.opts[name] = value
	return copied
}

func selectMissingTerminalShard(ctx context.Context, args cliArgs, dataset, hfID, quantization, quantFormat string, cfg terminalConfig) (cliArgs, map[string]any, error) {
	if opt(args, "shard") != "" {
		return args, nil, nil
	}
	query := url.Values{}
	query.Set("hfId", hfID)
	query.Set("harnessKey", terminalHarnessKey(args, cfg))
	if quantization != "" {
		query.Set("quantization", quantization)
	}
	if quantFormat != "" {
		query.Set("quantFormat", quantFormat)
	}
	value, err := fetchJSONContext(ctx, "GET", apiURL(args)+"/api/benchmarks/"+url.PathEscape(dataset)+"/coverage?"+query.Encode(), apiKey(args), nil)
	if err != nil {
		if ctx.Err() != nil {
			return args, nil, terminalCancelledError(ctx)
		}
		return args, nil, cliError{"coverage_fetch_failed", fmt.Sprintf("Could not select a missing terminal shard: %v", err), []string{"Pass --shard <index> to select one explicitly.", "Check the model, quantization, quant format, harness, and API URL."}, nil}
	}
	if ctx.Err() != nil {
		return args, nil, terminalCancelledError(ctx)
	}
	coverage := asObject(value)
	_, _, missing := shardCoverageDetails(coverage)
	if len(missing) == 0 {
		return args, coverage, cliError{"no_missing_shards", "This terminal aggregate already covers every registered shard.", []string{"Use --shard <index> --rerun to repeat a covered shard deliberately."}, map[string]any{"dataset": dataset, "model": hfID, "harnessKey": terminalHarnessKey(args, cfg)}}
	}
	selected := strconv.Itoa(missing[0])
	printStatus(args, "terminal_missing_shard_selected", map[string]any{"dataset": dataset, "model": hfID, "shardIndex": missing[0], "missingShards": missing, "harnessKey": terminalHarnessKey(args, cfg)})
	return terminalArgsWithOption(args, "shard", selected), coverage, nil
}

func downloadTerminalBundle(ctx context.Context, args cliArgs, tmp, id, key, wantHash string) (string, error) {
	value, err := fetchJSONContext(ctx, "GET", apiURL(args)+"/api/benchmarks/storage/download-url?key="+url.QueryEscape(key), apiKey(args), nil)
	if err != nil {
		if ctx.Err() != nil {
			return "", terminalCancelledError(ctx)
		}
		return "", cliError{"bundle_download_failed", fmt.Sprintf("Could not presign terminal bundle download: %v", err), []string{"Check that the dataset is approved and the bundle key exists."}, map[string]any{"taskId": id, "bundle_key": key, "error": err.Error()}}
	}
	if ctx.Err() != nil {
		return "", terminalCancelledError(ctx)
	}
	downloadURL := stringValue(asObject(value)["downloadUrl"])
	if downloadURL == "" {
		return "", cliError{"bundle_download_failed", "Presign response did not include downloadUrl.", []string{"Check the LocalMaxxing API response."}, value}
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, downloadURL, nil)
	if err != nil {
		return "", cliError{"bundle_download_failed", fmt.Sprintf("Could not prepare terminal bundle download: %v", err), []string{"Check the signed download URL returned by the LocalMaxxing API."}, map[string]any{"taskId": id, "bundle_key": key}}
	}
	res, err := apiHTTPClient.Do(req)
	if err != nil {
		if ctx.Err() != nil {
			return "", terminalCancelledError(ctx)
		}
		return "", cliError{"bundle_download_failed", fmt.Sprintf("Could not download terminal bundle: %v", err), []string{"Retry; signed URLs expire quickly."}, map[string]any{"taskId": id, "bundle_key": key}}
	}
	defer res.Body.Close()
	data, readErr := io.ReadAll(res.Body)
	if readErr != nil {
		if ctx.Err() != nil {
			return "", terminalCancelledError(ctx)
		}
		return "", cliError{"bundle_download_failed", fmt.Sprintf("Could not read terminal bundle download: %v", readErr), []string{"Retry; signed URLs expire quickly."}, map[string]any{"taskId": id, "bundle_key": key}}
	}
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
	if ctx.Err() != nil {
		return "", terminalCancelledError(ctx)
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

func runTerminalBundles(ctx context.Context, args cliArgs, bundles []terminalBundle, baseURL, model string, cfg terminalConfig, concurrency int, resumed map[string]terminalTaskResult, checkpoint *terminalLiveCheckpointStore) ([]terminalTaskResult, error) {
	results := make([]terminalTaskResult, len(bundles))
	jobs := make(chan int)
	dispatchCtx, cancelDispatch := context.WithCancel(ctx)
	defer cancelDispatch()
	var wg sync.WaitGroup
	var persistMu sync.Mutex
	var persistErr error
	recordPersistError := func(err error) {
		persistMu.Lock()
		defer persistMu.Unlock()
		if persistErr == nil {
			persistErr = err
			cancelDispatch()
		}
	}
	for i, bundle := range bundles {
		if result, ok := resumed[bundle.Task.ID]; ok {
			results[i] = result
			printStatus(args, "terminal_task_resumed", map[string]any{"taskId": bundle.Task.ID, "index": i + 1, "total": len(bundles), "pass": result.pass, "scored": result.scored})
		}
	}
	for range concurrency {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-dispatchCtx.Done():
					return
				case idx, ok := <-jobs:
					if !ok || dispatchCtx.Err() != nil {
						return
					}
					b := bundles[idx]
					if err := checkpoint.startTask(b.Task.ID, idx+1); err != nil {
						recordPersistError(err)
						return
					}
					printStatus(args, "terminal_task_started", map[string]any{"taskId": b.Task.ID, "index": idx + 1, "total": len(bundles), "image": firstNonEmpty(b.Task.Image.Prebuilt, b.Task.Image.Dockerfile)})
					results[idx] = runTerminalTask(dispatchCtx, b.Task, b.Dir, baseURL, model, cfg)
					if results[idx].errCode != "" {
						printStatus(args, "terminal_task_error", map[string]any{"taskId": b.Task.ID, "code": results[idx].errCode, "detail": results[idx].errText})
					}
					if err := checkpoint.persistTask(b.Task, results[idx]); err != nil {
						recordPersistError(err)
						return
					}
					printStatus(args, "terminal_task_done", map[string]any{"taskId": b.Task.ID, "pass": results[idx].pass, "scored": results[idx].scored, "turns": terminalTaskTurnsValue(results[idx]), "wallTimeMs": results[idx].wallTimeMs, "tokenUsage": results[idx].usage.toMap()})
				}
			}
		}()
	}
sendJobs:
	for i, bundle := range bundles {
		if _, ok := resumed[bundle.Task.ID]; ok {
			continue
		}
		select {
		case jobs <- i:
		case <-dispatchCtx.Done():
			break sendJobs
		}
	}
	close(jobs)
	wg.Wait()
	if persistErr != nil {
		return nil, cliError{"checkpoint_write_failed", "Could not persist terminal task state.", []string{"Check --run-dir permissions and available disk space, then rerun with --resume auto."}, map[string]any{"runDir": opt(args, "run-dir"), "error": persistErr.Error()}}
	}
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}
	return results, nil
}

type terminalTaskEnvironment struct {
	containerName   string
	composeArgs     []string
	cleanup         func()
	skipTestsUpload bool
}

func (environment *terminalTaskEnvironment) serviceContainer(ctx context.Context, service string) (string, error) {
	if service == "" || service == "main" {
		return environment.containerName, nil
	}
	if len(environment.composeArgs) == 0 {
		return "", fmt.Errorf("artifact service %q requires a Docker Compose task", service)
	}
	args := append(append([]string(nil), environment.composeArgs...), "ps", "--quiet", service)
	out, code, timedOut, err := runCommand(ctx, 30*time.Second, "docker", args...)
	containerName := strings.TrimSpace(out)
	if err != nil || timedOut || code != 0 || containerName == "" {
		return "", terminalCommandError("container_lookup_failed", "Could not resolve Docker Compose service container.", "docker", args, code, out, timedOut)
	}
	return containerName, nil
}

func startTerminalTaskEnvironment(ctx context.Context, task terminalTask, bundleDir string, cfg terminalConfig) (*terminalTaskEnvironment, error) {
	if task.Image.ComposeFile != "" {
		return startTerminalComposeEnvironment(ctx, task, bundleDir, cfg)
	}
	imageStart := time.Now()
	imageRef, cleanupImage, err := resolveTerminalImage(ctx, task, bundleDir)
	if err != nil {
		return nil, err
	}
	printStatus(cfg.args, "terminal_image_resolved", map[string]any{"taskId": task.ID, "mode": imageMode(task.Image), "ms": time.Since(imageStart).Milliseconds()})
	containerName := "lmx-tb-" + sanitizeDockerName(task.ID) + "-" + randomHex(6)
	startArgs := []string{"run", "-d", "--rm", "--name", containerName}
	if cfg.runLabel != "" {
		startArgs = append(startArgs, "--label", "localmaxxing.run="+cfg.runLabel, "--label", "localmaxxing.task="+task.ID)
	}
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
	for k, v := range resolveEnvTemplates(task.Environment.Env) {
		startArgs = append(startArgs, "-e", k+"="+v)
	}
	startArgs = append(startArgs, imageRef, "sleep", "infinity")
	out, code, timedOut, runErr := runCommand(ctx, 60*time.Second, "docker", startArgs...)
	if runErr != nil || timedOut || code != 0 {
		cleanupTerminalImage(cleanupImage, cfg.cleanupImages)
		return nil, terminalCommandError("container_start_failed", "Could not start terminal task container.", "docker", startArgs, code, out, timedOut)
	}
	var once sync.Once
	return &terminalTaskEnvironment{
		containerName: containerName,
		cleanup: func() {
			once.Do(func() {
				_, _, _, _ = runCommand(context.Background(), 30*time.Second, "docker", "rm", "-f", containerName)
				cleanupTerminalImage(cleanupImage, cfg.cleanupImages)
			})
		},
	}, nil
}

func startTerminalVerifierEnvironment(ctx context.Context, task terminalTask, bundleDir string, cfg terminalConfig) (*terminalTaskEnvironment, error) {
	dockerfile := filepath.Join(bundleDir, "tests", "Dockerfile")
	if _, err := os.Stat(dockerfile); err != nil {
		return startTerminalTaskEnvironment(ctx, task, bundleDir, cfg)
	}
	verifierTask := task
	verifierTask.ID += "-verifier"
	verifierTask.Image = terminalImage{
		Dockerfile: "tests/Dockerfile", Context: "tests",
		BuildTimeoutSec: firstPositive(task.Verifier.BuildTimeoutSec, task.Image.BuildTimeoutSec),
	}
	imageStart := time.Now()
	imageRef, cleanupImage, err := resolveTerminalImage(ctx, verifierTask, bundleDir)
	if err != nil {
		return nil, err
	}
	printStatus(cfg.args, "terminal_verifier_image_resolved", map[string]any{"taskId": task.ID, "mode": "dockerfile", "ms": time.Since(imageStart).Milliseconds()})
	containerName := "lmx-tb-" + sanitizeDockerName(task.ID) + "-verifier-" + randomHex(6)
	startArgs := []string{"run", "-d", "--rm", "--name", containerName}
	if cfg.runLabel != "" {
		startArgs = append(startArgs, "--label", "localmaxxing.run="+cfg.runLabel, "--label", "localmaxxing.task="+task.ID)
	}
	environment := task.Verifier.Environment
	if environment.CPUs > 0 {
		startArgs = append(startArgs, "--cpus", strconv.FormatFloat(environment.CPUs, 'f', -1, 64))
	}
	if environment.MemoryMb > 0 {
		startArgs = append(startArgs, "--memory", fmt.Sprintf("%dm", environment.MemoryMb))
	}
	if environment.GPUs > 0 {
		startArgs = append(startArgs, "--gpus", "all")
	}
	if environment.Network == "no-network" {
		startArgs = append(startArgs, "--network", "none")
	}
	for k, v := range resolveEnvTemplates(environment.Env) {
		startArgs = append(startArgs, "-e", k+"="+v)
	}
	startArgs = append(startArgs, imageRef, "sleep", "infinity")
	out, code, timedOut, runErr := runCommand(ctx, 60*time.Second, "docker", startArgs...)
	if runErr != nil || timedOut || code != 0 {
		cleanupTerminalImage(cleanupImage, cfg.cleanupImages)
		return nil, terminalCommandError("verifier_container_start_failed", "Could not start terminal verifier container.", "docker", startArgs, code, out, timedOut)
	}
	var once sync.Once
	return &terminalTaskEnvironment{
		containerName:   containerName,
		skipTestsUpload: true,
		cleanup: func() {
			once.Do(func() {
				_, _, _, _ = runCommand(context.Background(), 30*time.Second, "docker", "rm", "-f", containerName)
				cleanupTerminalImage(cleanupImage, cfg.cleanupImages)
			})
		},
	}, nil
}

func startTerminalComposeEnvironment(ctx context.Context, task terminalTask, bundleDir string, cfg terminalConfig) (*terminalTaskEnvironment, error) {
	environmentDir, err := filepath.Abs(filepath.Join(bundleDir, "environment"))
	if err != nil {
		return nil, cliError{"compose_start_failed", "Could not resolve the task environment directory.", []string{"Check the downloaded task bundle path."}, map[string]any{"taskId": task.ID, "error": err.Error()}}
	}
	composeFile := filepath.Join(environmentDir, task.Image.ComposeFile)
	if _, err := os.Stat(composeFile); err != nil {
		return nil, cliError{"compose_start_failed", "Task bundle is missing its Docker Compose file.", []string{"Re-import or re-download the task bundle."}, map[string]any{"taskId": task.ID, "composeFile": composeFile, "error": err.Error()}}
	}
	tempDir, err := os.MkdirTemp("", "lmx-terminal-compose-*")
	if err != nil {
		return nil, cliError{"compose_start_failed", "Could not create the Docker Compose override.", []string{"Check temporary directory permissions."}, map[string]any{"taskId": task.ID, "error": err.Error()}}
	}
	basePath := filepath.Join(tempDir, "base.json")
	mainService := map[string]any{
		"command": []string{"sh", "-c", "sleep infinity"},
	}
	if task.Image.Prebuilt != "" {
		mainService["image"] = task.Image.Prebuilt
	} else {
		mainService["build"] = map[string]any{"context": environmentDir}
		mainService["pull_policy"] = "build"
	}
	if task.Environment.CPUs > 0 {
		mainService["cpus"] = task.Environment.CPUs
	}
	if task.Environment.MemoryMb > 0 {
		mainService["mem_limit"] = fmt.Sprintf("%dm", task.Environment.MemoryMb)
	}
	if task.Environment.GPUs > 0 {
		mainService["gpus"] = "all"
	}
	if task.Environment.Network == "no-network" {
		mainService["network_mode"] = "none"
	}
	if task.Environment.Network == "allowlist" {
		printStatus(cfg.args, "terminal_network_degraded", map[string]any{"taskId": task.ID, "allowedHosts": strings.Join(task.Environment.AllowedHosts, ",")})
	}
	if resolvedEnv := resolveEnvTemplates(task.Environment.Env); len(resolvedEnv) > 0 {
		mainService["environment"] = resolvedEnv
	}
	if cfg.runLabel != "" {
		mainService["labels"] = map[string]string{"localmaxxing.run": cfg.runLabel, "localmaxxing.task": task.ID}
	}
	baseContent, err := json.Marshal(map[string]any{"services": map[string]any{"main": mainService}})
	if err != nil {
		_ = os.RemoveAll(tempDir)
		return nil, err
	}
	if err := os.WriteFile(basePath, baseContent, 0o600); err != nil {
		_ = os.RemoveAll(tempDir)
		return nil, cliError{"compose_start_failed", "Could not write the Docker Compose override.", []string{"Check temporary directory permissions."}, map[string]any{"taskId": task.ID, "error": err.Error()}}
	}
	projectName := "lmx-tb-" + sanitizeDockerName(task.ID) + "-" + randomHex(6)
	composeArgs := []string{"compose", "--project-name", projectName, "--project-directory", environmentDir, "-f", basePath, "-f", composeFile}
	upArgs := append(append([]string(nil), composeArgs...), "up", "--detach", "--build", "--wait")
	timeout := time.Duration(firstPositive(task.Image.BuildTimeoutSec, 600)+300) * time.Second
	imageStart := time.Now()
	out, code, timedOut, runErr := runCommand(ctx, timeout, "docker", upArgs...)
	if runErr != nil || timedOut || code != 0 {
		downArgs := append(append([]string(nil), composeArgs...), "down", "--volumes", "--remove-orphans")
		_, _, _, _ = runCommand(context.Background(), 2*time.Minute, "docker", downArgs...)
		_ = os.RemoveAll(tempDir)
		return nil, terminalCommandError("compose_start_failed", "Could not start Docker Compose task services.", "docker", upArgs, code, out, timedOut)
	}
	psArgs := append(append([]string(nil), composeArgs...), "ps", "--quiet", "main")
	mainOut, mainCode, mainTimedOut, mainErr := runCommand(ctx, 30*time.Second, "docker", psArgs...)
	containerName := strings.TrimSpace(mainOut)
	if mainErr != nil || mainTimedOut || mainCode != 0 || containerName == "" {
		downArgs := append(append([]string(nil), composeArgs...), "down", "--volumes", "--remove-orphans")
		_, _, _, _ = runCommand(context.Background(), 2*time.Minute, "docker", downArgs...)
		_ = os.RemoveAll(tempDir)
		return nil, terminalCommandError("container_lookup_failed", "Could not resolve the Docker Compose main service.", "docker", psArgs, mainCode, mainOut, mainTimedOut)
	}
	printStatus(cfg.args, "terminal_image_resolved", map[string]any{"taskId": task.ID, "mode": imageMode(task.Image), "ms": time.Since(imageStart).Milliseconds()})
	var once sync.Once
	return &terminalTaskEnvironment{
		containerName: containerName,
		composeArgs:   composeArgs,
		cleanup: func() {
			once.Do(func() {
				downArgs := append(append([]string(nil), composeArgs...), "down", "--volumes", "--remove-orphans")
				if cfg.cleanupImages {
					downArgs = append(downArgs, "--rmi", "local")
				}
				_, _, _, _ = runCommand(context.Background(), 2*time.Minute, "docker", downArgs...)
				_ = os.RemoveAll(tempDir)
			})
		},
	}, nil
}

func runTerminalVerifierCollectHooks(ctx context.Context, task terminalTask, environment *terminalTaskEnvironment) error {
	for _, hook := range task.Verifier.Collect {
		containerName, err := environment.serviceContainer(ctx, hook.Service)
		if err != nil {
			return cliError{"artifact_collection_failed", "Could not locate a verifier collect service.", []string{"Check [[verifier.collect]].service in the canonical Harbor task."}, map[string]any{"taskId": task.ID, "service": hook.Service, "error": err.Error()}}
		}
		execArgs := []string{"exec"}
		resolvedEnv := resolveEnvTemplates(hook.Env)
		keys := make([]string, 0, len(resolvedEnv))
		for key := range resolvedEnv {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		for _, key := range keys {
			execArgs = append(execArgs, "--env", key+"="+resolvedEnv[key])
		}
		execArgs = append(execArgs, containerName, "sh", "-lc", hook.Command)
		timeout := time.Duration(firstPositive(hook.TimeoutSec, 30)) * time.Second
		out, code, timedOut, runErr := runCommand(ctx, timeout, "docker", execArgs...)
		if runErr != nil || timedOut || code != 0 {
			return terminalCommandError("artifact_collection_failed", "A verifier collect hook failed.", "docker", execArgs, code, out, timedOut)
		}
	}
	return nil
}

func collectTerminalArtifacts(ctx context.Context, task terminalTask, environment *terminalTaskEnvironment, cfg terminalConfig) (string, error) {
	root, err := os.MkdirTemp("", "lmx-terminal-artifacts-*")
	if err != nil {
		return "", cliError{"artifact_collection_failed", "Could not create the artifact staging directory.", []string{"Check temporary directory permissions."}, map[string]any{"taskId": task.ID, "error": err.Error()}}
	}
	for _, artifact := range task.Artifacts {
		source := path.Clean(strings.TrimSpace(artifact.Source))
		if !strings.HasPrefix(source, "/") || source == "/" {
			_ = os.RemoveAll(root)
			return "", cliError{"artifact_collection_failed", "Artifact sources must be absolute container paths.", []string{"Re-import the canonical Harbor task."}, map[string]any{"taskId": task.ID, "source": artifact.Source}}
		}
		containerName, err := environment.serviceContainer(ctx, artifact.Service)
		if err != nil {
			_ = os.RemoveAll(root)
			return "", cliError{"artifact_collection_failed", "Could not locate an artifact source service.", []string{"Check that every artifact service exists in the task Compose file."}, map[string]any{"taskId": task.ID, "service": artifact.Service, "source": source, "error": err.Error()}}
		}
		hostPath := filepath.Join(root, filepath.FromSlash(strings.TrimPrefix(source, "/")))
		if err := os.MkdirAll(filepath.Dir(hostPath), 0o755); err != nil {
			_ = os.RemoveAll(root)
			return "", err
		}
		out, code, timedOut, copyErr := runCommand(ctx, 2*time.Minute, "docker", "cp", containerName+":"+source, hostPath)
		if copyErr != nil || timedOut || code != 0 {
			printStatus(cfg.args, "terminal_artifact_missing", map[string]any{"taskId": task.ID, "service": firstNonEmpty(artifact.Service, "main"), "source": source, "detail": truncateString(out, 1024)})
			continue
		}
		if err := pruneTerminalArtifactExcludes(hostPath, artifact.Exclude); err != nil {
			_ = os.RemoveAll(root)
			return "", cliError{"artifact_collection_failed", "Could not apply artifact exclusions.", []string{"Check artifact paths and permissions."}, map[string]any{"taskId": task.ID, "source": source, "error": err.Error()}}
		}
	}
	return root, nil
}

func pruneTerminalArtifactExcludes(root string, patterns []string) error {
	if len(patterns) == 0 {
		return nil
	}
	cleanPatterns := make([]string, 0, len(patterns))
	for _, pattern := range patterns {
		cleanPatterns = append(cleanPatterns, strings.TrimPrefix(filepath.ToSlash(filepath.Clean(pattern)), "./"))
	}
	return filepath.WalkDir(root, func(current string, entry os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if current == root {
			return nil
		}
		relative, err := filepath.Rel(root, current)
		if err != nil {
			return err
		}
		relative = filepath.ToSlash(relative)
		for _, pattern := range cleanPatterns {
			relativeMatch, _ := path.Match(pattern, relative)
			baseMatch, _ := path.Match(pattern, path.Base(relative))
			if !relativeMatch && !baseMatch {
				continue
			}
			if err := os.RemoveAll(current); err != nil {
				return err
			}
			if entry.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}
		return nil
	})
}

func uploadTerminalArtifacts(ctx context.Context, task terminalTask, environment *terminalTaskEnvironment, root string) error {
	for _, artifact := range task.Artifacts {
		source := path.Clean(strings.TrimSpace(artifact.Source))
		hostPath := filepath.Join(root, filepath.FromSlash(strings.TrimPrefix(source, "/")))
		info, err := os.Stat(hostPath)
		if err != nil {
			if errors.Is(err, os.ErrNotExist) {
				continue
			}
			return err
		}
		parent := path.Dir(source)
		if out, code, timedOut, err := runCommand(ctx, 30*time.Second, "docker", "exec", environment.containerName, "mkdir", "-p", parent); err != nil || timedOut || code != 0 {
			return terminalCommandError("artifact_upload_failed", "Could not create the verifier artifact directory.", "docker", []string{"exec", environment.containerName, "mkdir", "-p", parent}, code, out, timedOut)
		}
		if !info.IsDir() {
			_, _, _, _ = runCommand(ctx, 30*time.Second, "docker", "exec", environment.containerName, "rm", "-rf", source)
		}
		copySource := hostPath
		if info.IsDir() {
			if out, code, timedOut, err := runCommand(ctx, 30*time.Second, "docker", "exec", environment.containerName, "mkdir", "-p", source); err != nil || timedOut || code != 0 {
				return terminalCommandError("artifact_upload_failed", "Could not create the verifier artifact directory.", "docker", []string{"exec", environment.containerName, "mkdir", "-p", source}, code, out, timedOut)
			}
			copySource += string(os.PathSeparator) + "."
		}
		if out, code, timedOut, err := runCommand(ctx, 2*time.Minute, "docker", "cp", copySource, environment.containerName+":"+source); err != nil || timedOut || code != 0 {
			return terminalCommandError("artifact_upload_failed", "Could not upload an artifact to the separate verifier environment.", "docker", []string{"cp", copySource, environment.containerName + ":" + source}, code, out, timedOut)
		}
	}
	return nil
}

func runTerminalTask(ctx context.Context, task terminalTask, bundleDir, baseURL, model string, cfg terminalConfig) (result terminalTaskResult) {
	started := time.Now()
	result = terminalTaskResult{instruction: task.ID, prompt: terminalSystemPrompt, turnsUnreported: cfg.oracle || cfg.agentCommand != ""}
	defer func() {
		if ctx.Err() != nil {
			result.scored = false
			result.errCode = "terminal_cancelled"
			result.errText = "Terminal execution was cancelled."
			result.wallTimeMs = time.Since(started).Milliseconds()
		}
	}()
	if err := dockerPreflightContext(ctx); err != nil {
		result.errCode = "docker_unavailable"
		result.errText = err.Error()
		return result
	}
	environment, err := startTerminalTaskEnvironment(ctx, task, bundleDir, cfg)
	if err != nil {
		result.errCode, result.errText = cliErrorCodeText(err)
		return result
	}
	defer environment.cleanup()
	containerName := environment.containerName
	if cfg.oracle {
		transcript, err := runOracleSolution(ctx, task, bundleDir, containerName, cfg)
		result.transcript = transcript
		result.prompt = "oracle solution"
		if err != nil {
			result.errCode, result.errText = cliErrorCodeText(err)
			result.wallTimeMs = time.Since(started).Milliseconds()
			return persistTerminalTaskErrorTrace(task, result, cfg)
		}
	} else if cfg.agentCommand != "" {
		transcript, usage, err := runExternalTerminalAgent(ctx, task, bundleDir, containerName, baseURL, model, cfg)
		result.transcript = transcript
		result.usage = usage
		result.prompt = "external agent command"
		if err != nil {
			result.errCode, result.errText = cliErrorCodeText(err)
			result.wallTimeMs = time.Since(started).Milliseconds()
			return persistTerminalTaskErrorTrace(task, result, cfg)
		}
	} else if cfg.shellMode == "stateless" {
		turns, transcript, usage, err := runTerminalAgentLoop(ctx, task, containerName, baseURL, model, cfg)
		result.turns = turns
		result.transcript = transcript
		result.usage = usage
		if err != nil {
			code, text := cliErrorCodeText(err)
			if code != "agent_timeout" {
				result.errCode, result.errText = code, text
				result.wallTimeMs = time.Since(started).Milliseconds()
				return persistTerminalTaskErrorTrace(task, result, cfg)
			}
			result.transcript += "\n\n## Agent timeout\n" + text + "\nProceeding to verification with the container state left by the agent.\n"
			printStatus(cfg.args, "terminal_agent_timeout", map[string]any{"taskId": task.ID, "timeoutSec": terminalAgentTimeoutSec(cfg, task), "proceedingToVerifier": true})
		}
	} else {
		result.prompt = terminalSessionSystemPrompt
		turns, transcript, usage, err := runTerminalAgentLoopSession(ctx, task, containerName, baseURL, model, cfg)
		result.turns = turns
		result.transcript = transcript
		result.usage = usage
		if err != nil {
			code, text := cliErrorCodeText(err)
			if code != "agent_timeout" {
				result.errCode, result.errText = code, text
				result.wallTimeMs = time.Since(started).Milliseconds()
				return persistTerminalTaskErrorTrace(task, result, cfg)
			}
			result.transcript += "\n\n## Agent timeout\n" + text + "\nProceeding to verification with the container state left by the agent.\n"
			printStatus(cfg.args, "terminal_agent_timeout", map[string]any{"taskId": task.ID, "timeoutSec": terminalAgentTimeoutSec(cfg, task), "proceedingToVerifier": true})
		}
	}
	if task.Verifier.EnvironmentMode == "separate" {
		if collectHookErr := runTerminalVerifierCollectHooks(ctx, task, environment); collectHookErr != nil {
			result.errCode, result.errText = cliErrorCodeText(collectHookErr)
			result.wallTimeMs = time.Since(started).Milliseconds()
			return persistTerminalTaskErrorTrace(task, result, cfg)
		}
		artifactsRoot, collectErr := collectTerminalArtifacts(ctx, task, environment, cfg)
		if collectErr != nil {
			result.errCode, result.errText = cliErrorCodeText(collectErr)
			result.wallTimeMs = time.Since(started).Milliseconds()
			return persistTerminalTaskErrorTrace(task, result, cfg)
		}
		defer os.RemoveAll(artifactsRoot)
		environment.cleanup()
		verifierEnvironment, startErr := startTerminalVerifierEnvironment(ctx, task, bundleDir, cfg)
		if startErr != nil {
			result.errCode, result.errText = cliErrorCodeText(startErr)
			result.wallTimeMs = time.Since(started).Milliseconds()
			return persistTerminalTaskErrorTrace(task, result, cfg)
		}
		defer verifierEnvironment.cleanup()
		if uploadErr := uploadTerminalArtifacts(ctx, task, verifierEnvironment, artifactsRoot); uploadErr != nil {
			result.errCode, result.errText = cliErrorCodeText(uploadErr)
			result.wallTimeMs = time.Since(started).Milliseconds()
			return persistTerminalTaskErrorTrace(task, result, cfg)
		}
		environment = verifierEnvironment
		containerName = verifierEnvironment.containerName
	}
	pass, verifierOutput, err := runTerminalVerifierWithMode(ctx, task, bundleDir, containerName, environment.skipTestsUpload, cfg)
	result.pass = pass
	result.verifierOutput = verifierOutput
	result.scored = err == nil
	result.wallTimeMs = time.Since(started).Milliseconds()
	if err != nil {
		result.errCode, result.errText = cliErrorCodeText(err)
	}
	if traceErr := writeTerminalTaskTrace(task, result, cfg); traceErr != nil && err == nil {
		result.errCode, result.errText = cliErrorCodeText(traceErr)
		return result
	}
	if err != nil {
		return result
	}
	return result
}

func persistTerminalTaskErrorTrace(task terminalTask, result terminalTaskResult, cfg terminalConfig) terminalTaskResult {
	if traceErr := writeTerminalTaskTrace(task, result, cfg); traceErr != nil {
		if result.errText != "" {
			result.errText += " Trace persistence also failed: " + traceErr.Error()
		} else {
			result.errCode, result.errText = cliErrorCodeText(traceErr)
		}
	}
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

//go:embed terminus2-routed-shell.py
var terminus2RoutedShellScript string

func terminus2AgentCommand() (string, func(), error) {
	tmpDir, err := os.MkdirTemp("", "lmx-terminus2-adapter-*")
	if err != nil {
		return "", nil, cliError{"command_exec_failed", "Could not create a temporary directory for the bundled Terminus-2 adapter.", []string{"Check temporary directory permissions and available disk space."}, map[string]any{"error": err.Error()}}
	}
	cleanup := func() {
		_ = os.RemoveAll(tmpDir)
	}
	scriptPath := filepath.Join(tmpDir, "terminus2-routed-shell.py")
	if err := os.WriteFile(scriptPath, []byte(terminus2RoutedShellScript), 0o700); err != nil {
		cleanup()
		return "", nil, cliError{"command_exec_failed", "Could not extract the bundled Terminus-2 adapter.", []string{"Check temporary directory permissions and available disk space."}, map[string]any{"error": err.Error()}}
	}
	return shellQuote(scriptPath), cleanup, nil
}

func runExternalTerminalAgent(ctx context.Context, task terminalTask, bundleDir, containerName, baseURL, model string, cfg terminalConfig) (string, terminalTokenUsage, error) {
	tmp, err := os.MkdirTemp("", "lmx-terminal-agent-*")
	if err != nil {
		return "", terminalTokenUsage{}, cliError{"command_exec_failed", "Could not create external agent workspace.", []string{"Check temporary directory permissions."}, map[string]any{"taskId": task.ID, "error": err.Error()}}
	}
	defer os.RemoveAll(tmp)

	instructionFile := filepath.Join(tmp, "instruction.txt")
	traceDir := filepath.Join(tmp, "traces")
	if cfg.traceRoot != "" {
		traceDir = filepath.Join(cfg.traceRoot, sanitizeDockerName(task.ID), "agent")
	}
	if err := os.MkdirAll(traceDir, 0o700); err != nil {
		return "", terminalTokenUsage{}, cliError{"command_exec_failed", "Could not create external agent trace directory.", []string{"Check temporary directory permissions."}, map[string]any{"taskId": task.ID, "error": err.Error()}}
	}
	workdir := "/app"
	shellCommand := filepath.Join(tmp, "container-shell")
	userFlag := ""
	if task.Agent.User != "" {
		userFlag = " --user " + shellQuote(task.Agent.User)
	}
	shellScript := "#!/usr/bin/env bash\nset -euo pipefail\nif [ \"$#\" -eq 0 ]; then\n  exec docker exec -i" + userFlag + " -w " + shellQuote(workdir) + " " + shellQuote(containerName) + " bash -l\nfi\nexec docker exec -i" + userFlag + " -w " + shellQuote(workdir) + " " + shellQuote(containerName) + " bash -lc \"$*\"\n"
	if err := os.WriteFile(shellCommand, []byte(shellScript), 0o700); err != nil {
		return "", terminalTokenUsage{}, cliError{"command_exec_failed", "Could not write routed shell helper.", []string{"Check temporary directory permissions."}, map[string]any{"taskId": task.ID, "error": err.Error()}}
	}

	if err := os.WriteFile(instructionFile, []byte(task.Instruction), 0o600); err != nil {
		return "", terminalTokenUsage{}, cliError{"command_exec_failed", "Could not write external agent instruction file.", []string{"Check temporary directory permissions."}, map[string]any{"taskId": task.ID, "error": err.Error()}}
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
		"LMX_TERMINAL_AGENT_TIMEOUT_SEC=" + strconv.Itoa(terminalAgentTimeoutSec(cfg, task)),
		"LMX_TERMINAL_AGENT_USER=" + task.Agent.User,
		"LMX_TERMINAL_SHELL_COMMAND=" + shellCommand,
		"LMX_TERMINAL_MODEL_API_KEY=" + cfg.apiKey,
		"LMX_TERMINAL_MAX_TURNS=" + strconv.Itoa(terminalAgentMaxTurns(cfg, task)),
	}
	timeout := time.Duration(terminalAgentTimeoutSec(cfg, task)) * time.Second
	agentStarted := time.Now()
	agentDeadline := agentStarted.Add(timeout)
	printStatus(cfg.args, "terminal_external_agent_started", map[string]any{"taskId": task.ID, "command": truncateString(cfg.agentCommand, 240), "execution": cfg.agentExecution})
	heartbeatInterval := cfg.externalAgentHeartbeatInterval
	if heartbeatInterval <= 0 {
		heartbeatInterval = time.Minute
	}
	out, code, timedOut, runErr := runHostCommandWithEnvHeartbeat(ctx, timeout, env, cfg.agentCommand, heartbeatInterval, func(now time.Time) {
		remaining := int(agentDeadline.Sub(now).Seconds())
		if remaining < 0 {
			remaining = 0
		}
		printStatus(cfg.args, "terminal_external_agent_heartbeat", map[string]any{"taskId": task.ID, "execution": cfg.agentExecution, "elapsedSec": int(now.Sub(agentStarted).Seconds()), "agentTimeRemainingSec": remaining})
	})
	transcript := "$ " + cfg.agentCommand + "\n" + out + "\n[exit=" + strconv.Itoa(code) + "]\n"
	usage := externalAgentTokenUsage(traceDir)
	if usage.modelCalls > 0 {
		usageData, _ := json.MarshalIndent(usage.toMap(), "", "  ")
		_ = os.WriteFile(filepath.Join(traceDir, "usage.json"), append(usageData, '\n'), 0o644)
	}
	if traceText := externalAgentTraceText(traceDir); traceText != "" {
		transcript += "\n\n# External agent trace directory\n\n" + traceText
	}
	printStatus(cfg.args, "terminal_external_agent_done", map[string]any{"taskId": task.ID, "exitCode": code, "timedOut": timedOut, "execution": cfg.agentExecution})
	if timedOut {
		transcript += "\n[agent timed out after " + timeout.String() + "; proceeding to verification]\n"
		printStatus(cfg.args, "terminal_external_agent_timeout", map[string]any{"taskId": task.ID, "timeoutSec": int(timeout.Seconds())})
		return transcript, usage, nil
	}
	if runErr != nil || code != 0 {
		return transcript, usage, terminalCommandError("command_exec_failed", "External terminal agent command failed.", "bash", []string{"-lc", cfg.agentCommand}, code, out, timedOut)
	}
	return transcript, usage, nil
}

func runHostCommandWithEnv(ctx context.Context, timeout time.Duration, env []string, command string) (string, int, bool, error) {
	return runHostCommandWithEnvHeartbeat(ctx, timeout, env, command, 0, nil)
}

func runHostCommandWithEnvHeartbeat(ctx context.Context, timeout time.Duration, env []string, command string, heartbeatInterval time.Duration, heartbeat func(time.Time)) (string, int, bool, error) {
	cctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	cmd := exec.CommandContext(cctx, "bash", "-lc", command)
	configureCommandProcessGroup(cmd)
	cmd.Cancel = func() error {
		killCommandProcessGroup(cmd)
		return nil
	}
	cmd.WaitDelay = 2 * time.Second
	cmd.Env = append(os.Environ(), env...)
	var output bytes.Buffer
	cmd.Stdout = &output
	cmd.Stderr = &output
	if err := cmd.Start(); err != nil {
		if cctx.Err() == context.DeadlineExceeded {
			return output.String(), 124, true, err
		}
		if cctx.Err() != nil {
			return output.String(), 130, false, cctx.Err()
		}
		return output.String(), 1, false, err
	}
	waitResult := make(chan error, 1)
	go func() {
		waitResult <- cmd.Wait()
	}()
	var ticker *time.Ticker
	var ticks <-chan time.Time
	if heartbeat != nil && heartbeatInterval > 0 {
		ticker = time.NewTicker(heartbeatInterval)
		ticks = ticker.C
		defer ticker.Stop()
	}
	finish := func(err error) (string, int, bool, error) {
		if cctx.Err() == context.DeadlineExceeded {
			return output.String(), 124, true, err
		}
		if cctx.Err() != nil {
			return output.String(), 130, false, cctx.Err()
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
	for {
		select {
		case err := <-waitResult:
			return finish(err)
		case <-cctx.Done():
			killCommandProcessGroup(cmd)
			return finish(<-waitResult)
		case now := <-ticks:
			select {
			case err := <-waitResult:
				return finish(err)
			case <-cctx.Done():
				killCommandProcessGroup(cmd)
				return finish(<-waitResult)
			default:
				heartbeat(now)
			}
		}
	}
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
	b.WriteString("## Assistant\n" + stripTerminalNativeCallMetadata(content) + "\n")
}

func terminalJSONPrompt(instruction, terminalState string) string {
	return fmt.Sprintf(terminalJSONProtocolTemplate, instruction, terminalState)
}

func terminalJSONContinuePrompt(terminalState string) string {
	return "Current terminal state:\n" + terminalState + "\n\nContinue with the same JSON response format: analysis, plan, commands, and optional task_complete."
}

func parseTerminalAgentResponse(content string, nativeTools bool) (terminalJSONResponse, bool) {
	if nativeTools {
		return terminalJSONResponse{}, false
	}
	return parseTerminalJSONResponse(content)
}
func parseTerminalJSONResponse(content string) (terminalJSONResponse, bool) {
	var response terminalJSONResponse
	start := strings.Index(content, "{")
	end := strings.LastIndex(content, "}")
	if start < 0 || end < start {
		return response, false
	}
	if err := json.Unmarshal([]byte(content[start:end+1]), &response); err != nil {
		return response, false
	}
	return response, true
}

func terminalCommandFromKeystrokes(keystrokes string) (string, bool) {
	trimmed := strings.TrimSpace(keystrokes)
	switch trimmed {
	case "":
		return "", true
	case "C-c", "^C":
		return "", true
	case "C-d", "^D":
		return "exit", true
	}
	return strings.TrimRight(keystrokes, "\r\n"), true
}

type terminalStagnationTracker struct {
	fingerprint [sha256.Size]byte
	repeats     int
	initialized bool
}

func (tracker *terminalStagnationTracker) observe(command, observation string) (warn, stop bool) {
	hash := sha256.New()
	_, _ = io.WriteString(hash, command)
	_, _ = io.WriteString(hash, "\x00")
	_, _ = io.WriteString(hash, observation)
	var fingerprint [sha256.Size]byte
	copy(fingerprint[:], hash.Sum(nil))
	if tracker.initialized && fingerprint == tracker.fingerprint {
		tracker.repeats++
	} else {
		tracker.fingerprint = fingerprint
		tracker.repeats = 1
		tracker.initialized = true
	}
	return tracker.repeats == 3, tracker.repeats >= 6
}

func runTerminalAgentLoop(ctx context.Context, task terminalTask, containerName, baseURL, model string, cfg terminalConfig) (int, string, terminalTokenUsage, error) {
	systemPrompt := terminalSystemPrompt
	if cfg.nativeTools {
		systemPrompt = terminalNativeSystemPrompt
	}
	messages := []map[string]any{{"role": "system", "content": systemPrompt}, {"role": "user", "content": task.Instruction}}
	maxTurns := firstPositive(cfg.maxTurns, task.Agent.MaxTurns, 50)
	timeoutSec := terminalAgentTimeoutSec(cfg, task)
	deadline := time.Now().Add(time.Duration(timeoutSec) * time.Second)
	var transcript strings.Builder
	nonConforming := 0
	usage := terminalTokenUsage{}
	stagnation := terminalStagnationTracker{}
	for turn := 1; turn <= maxTurns && time.Now().Before(deadline); turn++ {
		if ctx.Err() != nil {
			return turn - 1, transcript.String(), usage, ctx.Err()
		}
		messages = trimTerminalMessages(messages)
		requestTimeout := terminalModelRequestTimeout(cfg, deadline, true)
		content, reasoning, callUsage, err := callTerminalModelWithHeartbeatContext(ctx, cfg, task.ID, turn, "initial", deadline, func() (string, string, terminalTokenUsage, error) {
			return callOpenAIChatMessagesContext(ctx, baseURL, model, messages, cfg.apiKey, terminalModelMaxTokens(cfg, false), cfg.temperature, cfg.topP, nil, cfg.nativeTools, requestTimeout)
		})
		usage.add(callUsage)
		if err != nil {
			firstErr := err
			messages = trimTerminalMessagesForRetry(messages)
			retryTimeout := terminalModelRequestTimeout(cfg, deadline, false)
			content, reasoning, callUsage, err = callTerminalModelWithHeartbeatContext(ctx, cfg, task.ID, turn, "retry", deadline, func() (string, string, terminalTokenUsage, error) {
				return callOpenAIChatMessagesContext(ctx, baseURL, model, messages, cfg.apiKey, terminalModelMaxTokens(cfg, true), cfg.temperature, cfg.topP, nil, cfg.nativeTools, retryTimeout)
			})
			usage.add(callUsage)
			if err != nil {
				return turn - 1, transcript.String(), usage, terminalModelCallFailure(task.ID, deadline, firstErr, err)
			}
		}
		cmdText, found := extractTerminalCommand(content)
		if cfg.nativeTools {
			cmdText, found = extractNativeTerminalCommand(content, reasoning)
		}
		if !found {
			if strings.Contains(content, "TASK_COMPLETE") {
				terminalTraceHeader(&transcript, turn, reasoning, content)
				transcript.WriteString("## Note\nModel signaled TASK_COMPLETE.\n")
				return turn - 1, transcript.String(), usage, nil
			}
			nonConforming++
			terminalTraceHeader(&transcript, turn, reasoning, content)
			if cfg.nativeTools {
				transcript.WriteString("## Note\nNo shell tool call found; asked the model to call shell or return TASK_COMPLETE.\n")
				messages = append(messages, terminalAssistantMessage(compactAssistantForModel(content, ""), reasoning), map[string]any{"role": "user", "content": terminalNativeContinuePrompt})
			} else {
				transcript.WriteString("## Note\nNo executable command found; asked the model to emit one command or TASK_COMPLETE.\n")
				messages = append(messages, terminalAssistantMessage(compactAssistantForModel(content, ""), reasoning), map[string]any{"role": "user", "content": "Your previous reply was not executable. Reply with exactly one ```bash fenced block. If you meant to run Python, wrap it as: python3 <<'PY'\n...\nPY"})
			}
			if nonConforming >= 3 {
				transcript.WriteString("## Note\nStopping after repeated non-executable replies.\n")
				return turn - 1, transcript.String(), usage, nil
			}
			continue
		}
		nonConforming = 0
		terminalTraceHeader(&transcript, turn, reasoning, content)
		if cfg.nativeTools {
			calls := terminalNativeCallsForTurn(content, cmdText, reasoning, turn)
			messages = append(messages, terminalNativeAssistantMessage(content, cmdText, reasoning, turn))
			var combinedObservation strings.Builder
			var combinedCommands []string
			for index, call := range calls {
				callCommand := call.Command
				if callCommand == "" {
					callCommand = cmdText
				}
				combinedCommands = append(combinedCommands, callCommand)
				transcript.WriteString("## Command\n$ " + callCommand + "\n")
				execArgs := []string{"exec"}
				if task.Agent.User != "" {
					execArgs = append(execArgs, "--user", task.Agent.User)
				}
				execArgs = append(execArgs, containerName, "bash", "-lc", callCommand)
				out, code, timedOut, _ := runCommand(ctx, terminalCommandExecutionTimeout(callCommand, time.Duration(terminalCommandTimeoutSec(cfg))*time.Second, deadline), "docker", execArgs...)
				if timedOut {
					out += "\n[command timed out]"
					code = 124
				}
				shown := truncateString(out, 8192)
				callObservation := terminalObservationForModel(out, code, timedOut)
				transcript.WriteString(shown + "\n[exit=" + strconv.Itoa(code) + "]\n")
				messages = append(messages, terminalNativeToolMessage(callObservation, call.ID))
				combinedObservation.WriteString("$ " + callCommand + "\n" + callObservation + "\n")
				fields := map[string]any{"taskId": task.ID, "turn": turn, "commandIndex": index + 1, "exitCode": code, "cmdPreview": truncateString(strings.ReplaceAll(callCommand, "\n", " "), 160)}
				if timedOut {
					fields["timedOut"] = true
					fields["timeoutSec"] = terminalCommandTimeoutSec(cfg)
				}
				printStatus(cfg.args, "terminal_turn", fields)
			}
			cmdText = strings.Join(combinedCommands, "\n")
			observation := truncateString(combinedObservation.String(), terminalModelObservationLimit)
			warn, stop := stagnation.observe(cmdText, observation)
			if warn {
				messages = append(messages, map[string]any{"role": "user", "content": "No progress: you have repeated the same command and received the same result three times. Do not repeat it again; change your approach or finish the task."})
				transcript.WriteString("## Note\nRepeated command and result detected; instructed the model to change approach.\n")
				printStatus(cfg.args, "terminal_agent_stagnation_warning", map[string]any{"taskId": task.ID, "turn": turn})
			}
			if stop {
				transcript.WriteString("## Note\nStopping after six identical command/result turns; proceeding to verification.\n")
				printStatus(cfg.args, "terminal_agent_stagnation_stopped", map[string]any{"taskId": task.ID, "turn": turn})
				return turn, transcript.String(), usage, nil
			}
		} else {
			messages = append(messages, terminalAssistantMessage(compactAssistantForModel(content, cmdText), reasoning))
			transcript.WriteString("## Command\n$ " + cmdText + "\n")
			execArgs := []string{"exec"}
			if task.Agent.User != "" {
				execArgs = append(execArgs, "--user", task.Agent.User)
			}
			execArgs = append(execArgs, containerName, "bash", "-lc", cmdText)
			out, code, timedOut, _ := runCommand(ctx, terminalCommandExecutionTimeout(cmdText, time.Duration(terminalCommandTimeoutSec(cfg))*time.Second, deadline), "docker", execArgs...)
			if timedOut {
				out += "\n[command timed out]"
				code = 124
			}
			shown := truncateString(out, 8192)
			observation := terminalObservationForModel(out, code, timedOut)
			transcript.WriteString(shown + "\n[exit=" + strconv.Itoa(code) + "]\n")
			messages = append(messages, map[string]any{"role": "user", "content": observation})
			warn, stop := stagnation.observe(cmdText, observation)
			if warn {
				messages = append(messages, map[string]any{"role": "user", "content": "No progress: you have repeated the same command and received the same result three times. Do not repeat it again; change your approach or finish the task."})
				transcript.WriteString("## Note\nRepeated command and result detected; instructed the model to change approach.\n")
				printStatus(cfg.args, "terminal_agent_stagnation_warning", map[string]any{"taskId": task.ID, "turn": turn})
			}
			if stop {
				transcript.WriteString("## Note\nStopping after six identical command/result turns; proceeding to verification.\n")
				printStatus(cfg.args, "terminal_agent_stagnation_stopped", map[string]any{"taskId": task.ID, "turn": turn})
				return turn, transcript.String(), usage, nil
			}
			fields := map[string]any{"taskId": task.ID, "turn": turn, "exitCode": code, "cmdPreview": truncateString(strings.ReplaceAll(cmdText, "\n", " "), 160)}
			if timedOut {
				fields["timedOut"] = true
				fields["timeoutSec"] = terminalCommandTimeoutSec(cfg)
			}
			printStatus(cfg.args, "terminal_turn", fields)
		}
	}
	return maxTurns, transcript.String(), usage, nil
}

type terminalShell struct {
	containerName string
	user          string
	nonce         string
	cmd           *exec.Cmd
	stdin         io.WriteCloser
	reader        *bufio.Reader
	pr            *os.File
}

func startTerminalShell(containerName, user string) (*terminalShell, error) {
	s := &terminalShell{containerName: containerName, user: user, nonce: randomHex(8)}
	if err := s.start(); err != nil {
		return nil, err
	}
	return s, nil
}

func (s *terminalShell) start() error {
	// No context timeout on the shell process itself; per-command timeouts are
	// enforced in exec(). The shell lives for the whole agent loop.
	execArgs := []string{"exec", "-i"}
	if s.user != "" {
		execArgs = append(execArgs, "--user", s.user)
	}
	execArgs = append(execArgs, s.containerName, "bash", "-l")
	cmd := exec.Command("docker", execArgs...)
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

func terminalShellPayload(command, marker string) string {
	return "{\n" + command + "\n} </dev/null\n" +
		"__lmx_status=$?\n" +
		"printf '\\n" + marker + "%d__\\n' \"$__lmx_status\"\n"
}

// exec runs one command in the persistent shell. timedOut means the per-command
// budget elapsed; restarted means the shell was rebuilt (state reset) due to
// timeout or the shell dying (e.g. the command ran `exit` or crashed bash).
func (s *terminalShell) exec(command string, timeout time.Duration) (output string, exitCode int, timedOut bool, restarted bool) {
	return s.execContext(context.Background(), command, timeout)
}

func (s *terminalShell) execContext(ctx context.Context, command string, timeout time.Duration) (output string, exitCode int, timedOut bool, restarted bool) {
	marker := "__LMX_END_" + s.nonce + "__"
	payload := terminalShellPayload(command, marker)
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
	timer := time.NewTimer(timeout)
	defer timer.Stop()
	select {
	case r := <-done:
		if r.eof {
			_ = s.restart()
			return r.out + "\n[shell ended; session restarted, state reset]", r.code, false, true
		}
		return r.out, r.code, false, false
	case <-ctx.Done():
		killCommandProcessGroup(s.cmd)
		return "[command cancelled]", 130, false, false
	case <-timer.C:
		_ = s.restart()
		return "[command timed out after " + timeout.String() + "; session restarted, state reset]", 124, true, true
	}
}

func terminalAssistantMessage(content, reasoning string) map[string]any {
	message := map[string]any{"role": "assistant", "content": content}
	if reasoning = strings.TrimSpace(reasoning); reasoning != "" {
		message["reasoning_content"] = reasoning
	}
	return message
}

func terminalNativeCallsForTurn(content, command, reasoning string, turn int) []terminalNativeCallMetadata {
	if calls, found := terminalNativeCallsFromContent(content); found {
		for index := range calls {
			if calls[index].ID == "" {
				calls[index].ID = fmt.Sprintf("call_%d_%d", turn, index+1)
			}
		}
		return calls
	}
	call, found := terminalNativeCallFromLFMText(content)
	if !found {
		call, found = terminalNativeCallFromLFMText(reasoning)
	}
	if !found {
		arguments, _ := json.Marshal(map[string]string{"command": command})
		call = terminalNativeCallMetadata{Name: "terminal", Arguments: string(arguments)}
	}
	call.ID = fmt.Sprintf("call_%d_1", turn)
	call.Command = command
	return []terminalNativeCallMetadata{call}
}

func terminalNativeAssistantMessage(content, command, reasoning string, turn int) map[string]any {
	calls := terminalNativeCallsForTurn(content, command, reasoning, turn)
	toolCalls := make([]any, 0, len(calls))
	for _, call := range calls {
		toolCalls = append(toolCalls, map[string]any{
			"id":   call.ID,
			"type": "function",
			"function": map[string]any{
				"name":      call.Name,
				"arguments": call.Arguments,
			},
		})
	}
	message := map[string]any{
		"role":       "assistant",
		"content":    "",
		"tool_calls": toolCalls,
	}
	if reasoning = strings.TrimSpace(reasoning); reasoning != "" {
		message["reasoning_content"] = reasoning
	}
	return message
}

func terminalNativeToolMessage(observation, callID string) map[string]any {
	return map[string]any{
		"role":         "tool",
		"tool_call_id": callID,
		"content":      observation,
	}
}

func runTerminalAgentLoopSession(ctx context.Context, task terminalTask, containerName, baseURL, model string, cfg terminalConfig) (int, string, terminalTokenUsage, error) {
	shell, err := startTerminalShell(containerName, task.Agent.User)
	if err != nil {
		return 0, "", terminalTokenUsage{}, cliError{"command_exec_failed", "Could not open a persistent shell in the task container.", []string{"Check Docker and that the task image provides /bin/bash.", "Or rerun with --shell-mode stateless."}, map[string]any{"taskId": task.ID, "error": err.Error()}}
	}
	defer shell.close()

	var messages []map[string]any
	if cfg.nativeTools {
		messages = []map[string]any{{"role": "system", "content": terminalNativeSessionSystemPrompt}, {"role": "user", "content": task.Instruction}}
	} else {
		messages = []map[string]any{{"role": "user", "content": terminalJSONPrompt(task.Instruction, "")}}
	}
	maxTurns := terminalAgentMaxTurns(cfg, task)
	timeoutSec := terminalAgentTimeoutSec(cfg, task)
	deadline := time.Now().Add(time.Duration(timeoutSec) * time.Second)
	cmdTimeout := time.Duration(terminalCommandTimeoutSec(cfg)) * time.Second
	var transcript strings.Builder
	nonConforming := 0
	usage := terminalTokenUsage{}
	stagnation := terminalStagnationTracker{}
	terminalState := ""
	for turn := 1; turn <= maxTurns && time.Now().Before(deadline); turn++ {
		if ctx.Err() != nil {
			return turn - 1, transcript.String(), usage, ctx.Err()
		}
		messages = trimTerminalMessages(messages)
		requestTimeout := terminalModelRequestTimeout(cfg, deadline, true)
		content, reasoning, callUsage, err := callTerminalModelWithHeartbeatContext(ctx, cfg, task.ID, turn, "initial", deadline, func() (string, string, terminalTokenUsage, error) {
			return callOpenAIChatMessagesContext(ctx, baseURL, model, messages, cfg.apiKey, terminalModelMaxTokens(cfg, false), cfg.temperature, cfg.topP, nil, cfg.nativeTools, requestTimeout)
		})
		usage.add(callUsage)
		if err != nil {
			firstErr := err
			messages = trimTerminalMessagesForRetry(messages)
			retryTimeout := terminalModelRequestTimeout(cfg, deadline, false)
			content, reasoning, callUsage, err = callTerminalModelWithHeartbeatContext(ctx, cfg, task.ID, turn, "retry", deadline, func() (string, string, terminalTokenUsage, error) {
				return callOpenAIChatMessagesContext(ctx, baseURL, model, messages, cfg.apiKey, terminalModelMaxTokens(cfg, true), cfg.temperature, cfg.topP, nil, cfg.nativeTools, retryTimeout)
			})
			usage.add(callUsage)
			if err != nil {
				return turn - 1, transcript.String(), usage, terminalModelCallFailure(task.ID, deadline, firstErr, err)
			}
		}

		terminalTraceHeader(&transcript, turn, reasoning, content)
		response, foundJSON := parseTerminalAgentResponse(content, cfg.nativeTools)
		if !foundJSON {
			cmdText, foundBash := extractTerminalCommand(content)
			if cfg.nativeTools {
				cmdText, foundBash = extractNativeTerminalCommand(content, reasoning)
			}
			if !foundBash {
				if strings.Contains(content, "TASK_COMPLETE") {
					transcript.WriteString("## Note\nModel signaled TASK_COMPLETE.\n")
					return turn - 1, transcript.String(), usage, nil
				}
				nonConforming++
				if cfg.nativeTools {
					transcript.WriteString("## Note\nNo shell tool call found; asked the model to call shell or return TASK_COMPLETE.\n")
					messages = append(messages, terminalAssistantMessage(compactAssistantForModel(content, ""), reasoning), map[string]any{"role": "user", "content": terminalNativeContinuePrompt})
				} else {
					transcript.WriteString("## Note\nNo JSON command response or bash block found; asked the model to emit the required JSON.\n")
					messages = append(messages, terminalAssistantMessage(compactAssistantForModel(content, ""), reasoning), map[string]any{"role": "user", "content": terminalJSONContinuePrompt(terminalState)})
				}
				if nonConforming >= 3 {
					transcript.WriteString("## Note\nStopping after repeated non-executable replies.\n")
					return turn - 1, transcript.String(), usage, nil
				}
				continue
			}
			response = terminalJSONResponse{
				Analysis: content,
				Plan:     "Execute fallback bash block.",
				Commands: []terminalJSONCommand{{Keystrokes: cmdText + "\n", Duration: 1}},
			}
		}

		nonConforming = 0
		if !cfg.nativeTools {
			messages = append(messages, terminalAssistantMessage(content, reasoning))
		}
		if response.TaskComplete && len(response.Commands) == 0 {
			transcript.WriteString("## Note\nModel marked task complete.\n")
			return turn - 1, transcript.String(), usage, nil
		}

		var nativeCalls []terminalNativeCallMetadata
		if cfg.nativeTools {
			var fallbackCommands []string
			for _, command := range response.Commands {
				if cmdText, _ := terminalCommandFromKeystrokes(command.Keystrokes); cmdText != "" {
					fallbackCommands = append(fallbackCommands, cmdText)
				}
			}
			fallbackCommand := strings.Join(fallbackCommands, "\n")
			nativeCalls = terminalNativeCallsForTurn(content, fallbackCommand, reasoning, turn)
			if structuredCalls, found := terminalNativeCallsFromContent(content); found {
				response.Commands = make([]terminalJSONCommand, 0, len(structuredCalls))
				for _, call := range structuredCalls {
					response.Commands = append(response.Commands, terminalJSONCommand{Keystrokes: call.Command + "\n", Duration: 1})
				}
			}
		}

		var observation strings.Builder
		var nativeCommands []string
		var nativeResults []string
		for i, command := range response.Commands {
			cmdText, _ := terminalCommandFromKeystrokes(command.Keystrokes)
			if cmdText == "" {
				wait := time.Duration(command.Duration * float64(time.Second))
				if wait <= 0 {
					wait = time.Second
				}
				if wait > time.Minute {
					wait = time.Minute
				}
				timer := time.NewTimer(wait)
				select {
				case <-ctx.Done():
					timer.Stop()
					return turn - 1, transcript.String(), usage, ctx.Err()
				case <-timer.C:
				}
				continue
			}
			transcript.WriteString("## Command\n$ " + cmdText + "\n")
			nativeCommands = append(nativeCommands, cmdText)
			out, code, timedOut, restarted := shell.execContext(ctx, cmdText, terminalCommandExecutionTimeout(cmdText, cmdTimeout, deadline))
			if timedOut {
				out += "\n[command timed out]"
			}
			shown := truncateString(out, 8192)
			observation.WriteString("$ " + cmdText + "\n")
			observation.WriteString(terminalObservationForModel(out, code, timedOut))
			observation.WriteString("\n")
			if cfg.nativeTools {
				nativeResults = append(nativeResults, terminalObservationForModel(out, code, timedOut))
			}
			transcript.WriteString(shown + "\n[exit=" + strconv.Itoa(code) + "]\n")
			fields := map[string]any{"taskId": task.ID, "turn": turn, "commandIndex": i + 1, "exitCode": code, "cmdPreview": truncateString(strings.ReplaceAll(cmdText, "\n", " "), 160)}
			if timedOut {
				fields["timedOut"] = true
				fields["timeoutSec"] = terminalCommandTimeoutSec(cfg)
			}
			if restarted {
				fields["shellRestarted"] = true
			}
			printStatus(cfg.args, "terminal_turn", fields)
		}
		terminalState = truncateString(observation.String(), terminalModelObservationLimit)
		warn, stop := stagnation.observe("", terminalState)
		nextPrompt := terminalJSONContinuePrompt(terminalState)
		if warn {
			nextPrompt += "\n\nNo progress: you have repeated the same command batch and received the same result three times. Do not repeat it again; change your approach or finish the task."
			transcript.WriteString("## Note\nRepeated command batch and result detected; instructed the model to change approach.\n")
			printStatus(cfg.args, "terminal_agent_stagnation_warning", map[string]any{"taskId": task.ID, "turn": turn})
		}
		if cfg.nativeTools {
			messages = append(messages, terminalNativeAssistantMessage(content, strings.Join(nativeCommands, "\n"), reasoning, turn))
			for index, result := range nativeResults {
				if index < len(nativeCalls) {
					messages = append(messages, terminalNativeToolMessage(result, nativeCalls[index].ID))
				}
			}
		} else {
			messages = append(messages, map[string]any{"role": "user", "content": nextPrompt})
		}
		if stop {
			transcript.WriteString("## Note\nStopping after six identical command/result turns; proceeding to verification.\n")
			printStatus(cfg.args, "terminal_agent_stagnation_stopped", map[string]any{"taskId": task.ID, "turn": turn})
			return turn, transcript.String(), usage, nil
		}
		if response.TaskComplete {
			transcript.WriteString("## Note\nModel marked task complete after command batch.\n")
			return turn, transcript.String(), usage, nil
		}
	}
	return maxTurns, transcript.String(), usage, nil
}

// runOracleSolution mirrors harbor's OracleAgent: solution/ is copied to
// /solution, solve.sh runs as root with DEBIAN_FRONTEND=noninteractive and the
// task's [solution].env, bounded by the agent timeout. A non-zero exit or
// timeout does not abort the trial — harbor records it and still verifies.
func runOracleSolution(ctx context.Context, task terminalTask, bundleDir, containerName string, cfg terminalConfig) (string, error) {
	if _, err := os.Stat(filepath.Join(bundleDir, "solution", "solve.sh")); err != nil {
		return "", cliError{"bundle_invalid", "Oracle mode requires solution/solve.sh.", []string{"Import a harbor task with solution/ or do not use --oracle."}, map[string]any{"taskId": task.ID, "bundleDir": bundleDir}}
	}
	out, code, timedOut, err := runCommand(ctx, 120*time.Second, "docker", "cp", filepath.Join(bundleDir, "solution")+"/.", containerName+":/solution")
	if err != nil || timedOut || code != 0 {
		return "", terminalCommandError("command_exec_failed", "Could not copy oracle solution into container.", "docker", []string{"cp", filepath.Join(bundleDir, "solution") + "/.", containerName + ":/solution"}, code, out, timedOut)
	}
	execArgs := []string{"exec", "--user", "root", "-e", "DEBIAN_FRONTEND=noninteractive"}
	for k, v := range resolveEnvTemplates(task.Solution.Env) {
		execArgs = append(execArgs, "-e", k+"="+v)
	}
	execArgs = append(execArgs, containerName, "bash", "/solution/solve.sh")
	timeout := time.Duration(terminalAgentTimeoutSec(cfg, task)) * time.Second
	out, code, timedOut, _ = runCommand(ctx, timeout, "docker", execArgs...)
	transcript := "$ bash /solution/solve.sh\n" + truncateString(out, 200_000) + "\n[exit=" + strconv.Itoa(code) + "]\n"
	if timedOut {
		transcript += "[oracle solution timed out after " + timeout.String() + "; proceeding to verification]\n"
	}
	if timedOut || code != 0 {
		printStatus(cfg.args, "terminal_oracle_solution_failed", map[string]any{"taskId": task.ID, "exitCode": code, "timedOut": timedOut})
	}
	return transcript, nil
}

const terminalVerifierInfrastructureRetries = 2

var terminalVerifierInfrastructureFailureMarkers = []string{
	"failed to download https://github.com/",
	"failed to download https://astral.sh/",
	"curl: (22) the requested url returned error: 503",
	"curl: (28) operation timed out",
	"curl: (35) openssl ssl_connect",
	"curl: (56) connection ",
	"temporary failure resolving",
	"could not resolve host",
}

func terminalVerifierInfrastructureFailure(output string) bool {
	lower := strings.ToLower(output)
	for _, marker := range terminalVerifierInfrastructureFailureMarkers {
		if strings.Contains(lower, marker) {
			return true
		}
	}
	return false
}

func clearTerminalVerifierReward(ctx context.Context, containerName, rewardFile string) {
	rewardJSONFile := path.Join(path.Dir(rewardFile), "reward.json")
	_, _, _, _ = runCommand(ctx, 30*time.Second, "docker", "exec", containerName, "rm", "-f", rewardFile, rewardJSONFile)
}

// runTerminalVerifier follows harbor canonical semantics: tests/ is copied to
// /tests, the verifier command runs in a non-login shell, and the reward file
// is the sole pass signal — reward.json ({"reward": <num>}) takes precedence
// over reward.txt (bare float), pass means reward >= 1.0, and the verifier's
// exit code is ignored once a reward was written. A verifier timeout or a
// missing/unparseable reward file scores the task as failed.
func runTerminalVerifier(ctx context.Context, task terminalTask, bundleDir, containerName string, cfg terminalConfig) (bool, string, error) {
	return runTerminalVerifierWithMode(ctx, task, bundleDir, containerName, false, cfg)
}

func runTerminalVerifierWithMode(ctx context.Context, task terminalTask, bundleDir, containerName string, skipTestsUpload bool, cfg terminalConfig) (bool, string, error) {
	_, _, _, _ = runCommand(ctx, 30*time.Second, "docker", "exec", containerName, "install", "-d", "-m", "1777", "/logs/verifier")
	var out string
	var code int
	var timedOut bool
	if !skipTestsUpload {
		var err error
		out, code, timedOut, err = runCommand(ctx, 120*time.Second, "docker", "cp", filepath.Join(bundleDir, "tests")+"/.", containerName+":/tests")
		if err != nil || timedOut || code != 0 {
			return false, out, terminalCommandError("verifier_failed", "Could not copy verifier tests into the task container.", "docker", []string{"cp", filepath.Join(bundleDir, "tests") + "/.", containerName + ":/tests"}, code, out, timedOut)
		}
	}
	cmdArgs := []string{"exec"}
	if task.Verifier.User != "" {
		cmdArgs = append(cmdArgs, "--user", task.Verifier.User)
	}
	for k, v := range resolveEnvTemplates(task.Verifier.Env) {
		cmdArgs = append(cmdArgs, "-e", k+"="+v)
	}
	cmdArgs = append(cmdArgs, containerName, "bash", "-c", task.Verifier.Command)
	rewardFile := firstNonEmpty(task.Verifier.RewardFile, "/logs/verifier/reward.txt")
	timeout := time.Duration(firstPositive(task.Verifier.TimeoutSec, 900)) * time.Second
	attemptOutputs := make([]string, 0, terminalVerifierInfrastructureRetries+1)
	for attempt := 1; ; attempt++ {
		clearTerminalVerifierReward(ctx, containerName, rewardFile)
		out, code, timedOut, _ = runCommand(ctx, timeout, "docker", cmdArgs...)
		attemptOutputs = append(attemptOutputs, out)
		reward, _, rewardOK := readTerminalVerifierReward(ctx, containerName, rewardFile)
		if timedOut || rewardOK && reward >= 1.0 || !terminalVerifierInfrastructureFailure(out) || attempt > terminalVerifierInfrastructureRetries {
			break
		}
		printStatus(cfg.args, "terminal_verifier_retry", map[string]any{"taskId": task.ID, "attempt": attempt, "nextAttempt": attempt + 1, "reason": "dependency_download_failed"})
		select {
		case <-ctx.Done():
			return false, strings.Join(attemptOutputs, "\n\n[verifier infrastructure retry]\n\n"), ctx.Err()
		case <-time.After(time.Duration(attempt) * time.Second):
		}
	}
	out = strings.Join(attemptOutputs, "\n\n[verifier infrastructure retry]\n\n")
	if timedOut {
		printStatus(cfg.args, "terminal_verifier", map[string]any{"taskId": task.ID, "reward": "", "exitCode": code, "timedOut": true})
		return false, out + "\n[verifier timed out]", cliError{"verifier_failed", "Verifier timed out; the task is scored as failed.", []string{"Raise [verifier].timeout_sec in the task if legitimate verifications need longer."}, map[string]any{"taskId": task.ID, "timeoutSec": firstPositive(task.Verifier.TimeoutSec, 900), "output": truncateString(out, 4096)}}
	}
	if reward, _, rewardOK := readTerminalVerifierReward(ctx, containerName, rewardFile); (!rewardOK || reward < 1.0) && terminalVerifierInfrastructureFailure(attemptOutputs[len(attemptOutputs)-1]) {
		output := out + "\n[verifier infrastructure failure after " + strconv.Itoa(len(attemptOutputs)) + " attempts]"
		printStatus(cfg.args, "terminal_verifier_infrastructure_failed", map[string]any{"taskId": task.ID, "attempts": len(attemptOutputs), "reason": "dependency_download_failed"})
		return false, output, cliError{"verifier_infrastructure_failed", "Verifier dependency bootstrap failed after retries; the task is unscored.", []string{"Retry verification when network access is stable.", "Preinstall verifier dependencies in the canonical task image to eliminate runtime downloads."}, map[string]any{"taskId": task.ID, "attempts": len(attemptOutputs), "output": truncateString(out, 4096)}}
	}
	reward, rewardRaw, rewardOK := readTerminalVerifierReward(ctx, containerName, rewardFile)
	output := out + "\n[verifier exit=" + strconv.Itoa(code) + "]\nreward: " + rewardRaw
	printStatus(cfg.args, "terminal_verifier", map[string]any{"taskId": task.ID, "reward": rewardRaw, "exitCode": code})
	if !rewardOK {
		return false, output, cliError{"verifier_failed", "Verifier did not produce a parseable reward file; the task is scored as failed.", []string{"Harbor verifiers must write /logs/verifier/reward.txt (float) or reward.json ({\"reward\": <num>})."}, map[string]any{"taskId": task.ID, "rewardFile": rewardFile, "exitCode": code, "output": truncateString(out, 4096)}}
	}
	return reward >= 1.0, output, nil
}

func readTerminalVerifierReward(ctx context.Context, containerName, rewardFile string) (float64, string, bool) {
	rewardJSONFile := path.Join(path.Dir(rewardFile), "reward.json")
	if jsonText, jsonCode, _, _ := runCommand(ctx, 30*time.Second, "docker", "exec", containerName, "cat", rewardJSONFile); jsonCode == 0 {
		if reward, ok := parseRewardJSON(jsonText); ok {
			return reward, strings.TrimSpace(jsonText), true
		}
	}
	if txtText, txtCode, _, _ := runCommand(ctx, 30*time.Second, "docker", "exec", containerName, "cat", rewardFile); txtCode == 0 {
		if reward, ok := parseRewardText(txtText); ok {
			return reward, strings.TrimSpace(txtText), true
		}
	}
	return 0, "", false
}

type terminalModelCallResult struct {
	content   string
	reasoning string
	usage     terminalTokenUsage
	err       error
}

func callTerminalModelWithHeartbeat(cfg terminalConfig, taskID string, turn int, attempt string, deadline time.Time, call func() (string, string, terminalTokenUsage, error)) (string, string, terminalTokenUsage, error) {
	return callTerminalModelWithHeartbeatContext(context.Background(), cfg, taskID, turn, attempt, deadline, call)
}

func callTerminalModelWithHeartbeatContext(ctx context.Context, cfg terminalConfig, taskID string, turn int, attempt string, deadline time.Time, call func() (string, string, terminalTokenUsage, error)) (string, string, terminalTokenUsage, error) {
	started := time.Now()
	result := make(chan terminalModelCallResult, 1)
	go func() {
		content, reasoning, usage, err := call()
		result <- terminalModelCallResult{content: content, reasoning: reasoning, usage: usage, err: err}
	}()
	heartbeatInterval := cfg.modelHeartbeatInterval
	if heartbeatInterval <= 0 {
		heartbeatInterval = time.Minute
	}
	ticker := time.NewTicker(heartbeatInterval)
	defer ticker.Stop()
	for {
		select {
		case completed := <-result:
			return completed.content, completed.reasoning, completed.usage, completed.err
		case <-ctx.Done():
			return "", "", terminalTokenUsage{}, ctx.Err()
		case now := <-ticker.C:
			remaining := int(time.Until(deadline).Seconds())
			if remaining < 0 {
				remaining = 0
			}
			printStatus(cfg.args, "terminal_model_call_heartbeat", map[string]any{"taskId": taskID, "turn": turn, "attempt": attempt, "elapsedSec": int(now.Sub(started).Seconds()), "agentTimeRemainingSec": remaining})
		}
	}
}

func callOpenAIChatMessages(baseURL, model string, messages []map[string]any, apiKey string, maxTokens int, temperature, topP float64, stop []string, nativeTools bool, timeout time.Duration) (content, reasoning string, usage terminalTokenUsage, err error) {
	return callOpenAIChatMessagesContext(context.Background(), baseURL, model, messages, apiKey, maxTokens, temperature, topP, stop, nativeTools, timeout)
}

func callOpenAIChatMessagesContext(parent context.Context, baseURL, model string, messages []map[string]any, apiKey string, maxTokens int, temperature, topP float64, stop []string, nativeTools bool, timeout time.Duration) (content, reasoning string, usage terminalTokenUsage, err error) {
	body := map[string]any{"model": model, "messages": messages, "temperature": temperature, "top_p": topP}
	if terminalDisablesTemplateThinking(model) {
		body["chat_template_kwargs"] = map[string]any{"enable_thinking": false}
	}
	if nativeTools {
		body["tools"] = terminalNativeTools()
		body["tool_choice"] = "auto"
	}
	if maxTokens > 0 {
		body["max_tokens"] = maxTokens
	}
	if len(stop) > 0 {
		body["stop"] = stop
	}
	data, _ := json.Marshal(body)
	ctx, cancel := context.WithTimeout(parent, timeout)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, "POST", openAIBaseURL(baseURL)+"/v1/chat/completions", bytes.NewReader(data))
	if err != nil {
		return "", "", terminalTokenUsage{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	if apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", "", terminalTokenUsage{}, err
	}
	defer res.Body.Close()
	if res.StatusCode < 200 || res.StatusCode >= 300 {
		text, _ := io.ReadAll(res.Body)
		return "", "", terminalTokenUsage{}, cliError{"model_call_failed", fmt.Sprintf("OpenAI-compatible server returned %s", res.Status), []string{"Check --base-url and --model-api-key."}, map[string]any{"status": res.Status, "body": truncateString(strings.TrimSpace(string(text)), 4096)}}
	}
	var response map[string]any
	if err := json.NewDecoder(res.Body).Decode(&response); err != nil {
		return "", "", terminalTokenUsage{}, err
	}
	usage = tokenUsageFromObject(asObject(response["usage"]))
	if choices, _ := response["choices"].([]any); len(choices) > 0 {
		if message := asObject(asObject(choices[0])["message"]); message != nil {
			content = strings.TrimSpace(stringValue(message["content"]))
			if nativeTools {
				content = appendNativeShellToolCalls(content, message)
			}
			reasoning = strings.TrimSpace(firstNonEmpty(stringValue(message["reasoning_content"]), stringValue(message["reasoning"])))
		}
	}
	return content, reasoning, usage, nil
}

func terminalNativeTools() []map[string]any {
	return []map[string]any{
		{
			"type": "function",
			"function": map[string]any{
				"name":        "read_file",
				"description": "Read a text file from the task container.",
				"parameters": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"path": map[string]any{"type": "string", "description": "Absolute or working-directory-relative file path."},
					},
					"required":             []string{"path"},
					"additionalProperties": false,
				},
			},
		},
		{
			"type": "function",
			"function": map[string]any{
				"name":        "write_file",
				"description": "Write content to a file, completely replacing its current contents.",
				"parameters": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"path":    map[string]any{"type": "string", "description": "Absolute or working-directory-relative file path."},
						"content": map[string]any{"type": "string", "description": "Complete replacement file content."},
					},
					"required":             []string{"path", "content"},
					"additionalProperties": false,
				},
			},
		},
		{
			"type": "function",
			"function": map[string]any{
				"name":        "edit_file",
				"description": "Replace one unique text fragment in an existing file without rewriting the rest of the file.",
				"parameters": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"path":     map[string]any{"type": "string", "description": "Absolute or working-directory-relative file path."},
						"old_text": map[string]any{"type": "string", "description": "Exact existing text to replace; it must occur exactly once."},
						"new_text": map[string]any{"type": "string", "description": "Replacement text."},
					},
					"required":             []string{"path", "old_text", "new_text"},
					"additionalProperties": false,
				},
			},
		},
		{
			"type": "function",
			"function": map[string]any{
				"name":        "terminal",
				"description": "Run one or more non-interactive shell commands in the persistent task container.",
				"parameters": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"command": map[string]any{"type": "string", "description": "Shell commands to execute."},
					},
					"required":             []string{"command"},
					"additionalProperties": false,
				},
			},
		},
	}
}

const terminalNativeCallMetadataPrefix = "\n<!-- lmx-native-tool-call:"
const terminalNativeCallMetadataSuffix = " -->"

type terminalNativeCallMetadata struct {
	ID        string `json:"id,omitempty"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
	Command   string `json:"command,omitempty"`
}

type terminalNativeCallEnvelope struct {
	Calls []terminalNativeCallMetadata `json:"calls"`
}

func terminalNativeCallsFromContent(content string) ([]terminalNativeCallMetadata, bool) {
	index := strings.LastIndex(content, terminalNativeCallMetadataPrefix)
	if index < 0 {
		return nil, false
	}
	encoded := strings.TrimSuffix(content[index+len(terminalNativeCallMetadataPrefix):], terminalNativeCallMetadataSuffix)
	payload, err := hex.DecodeString(encoded)
	if err != nil {
		return nil, false
	}
	var envelope terminalNativeCallEnvelope
	if err := json.Unmarshal(payload, &envelope); err != nil || len(envelope.Calls) == 0 {
		var legacy terminalNativeCallMetadata
		if legacyErr := json.Unmarshal(payload, &legacy); legacyErr != nil {
			return nil, false
		}
		envelope.Calls = []terminalNativeCallMetadata{legacy}
	}
	for _, call := range envelope.Calls {
		switch call.Name {
		case "read_file", "edit_file", "write_file", "terminal":
		default:
			return nil, false
		}
		if !json.Valid([]byte(call.Arguments)) || strings.TrimSpace(call.Command) == "" {
			return nil, false
		}
	}
	return envelope.Calls, true
}

func terminalNativeCallFromContent(content string) (terminalNativeCallMetadata, bool) {
	calls, found := terminalNativeCallsFromContent(content)
	if !found || len(calls) != 1 {
		return terminalNativeCallMetadata{}, false
	}
	return calls[0], true
}

func stripTerminalNativeCallMetadata(content string) string {
	if index := strings.LastIndex(content, terminalNativeCallMetadataPrefix); index >= 0 {
		return strings.TrimSpace(content[:index])
	}
	return content
}

func appendNativeShellToolCalls(content string, message map[string]any) string {
	var commands []string
	var metadata []terminalNativeCallMetadata
	toolCalls, _ := message["tool_calls"].([]any)
	for index, rawCall := range toolCalls {
		call := asObject(rawCall)
		function := asObject(call["function"])
		name := strings.ToLower(strings.TrimSpace(stringValue(function["name"])))
		rawArguments := stringValue(function["arguments"])
		var arguments map[string]any
		if err := json.Unmarshal([]byte(rawArguments), &arguments); err != nil {
			continue
		}
		var command string
		canonicalName := name
		switch name {
		case "shell", "terminal":
			canonicalName = "terminal"
			command = strings.TrimSpace(firstNonEmpty(stringValue(arguments["cmd"]), stringValue(arguments["command"]), stringValue(arguments["commands"]), stringValue(arguments["keystrokes"])))
		case "read", "read_file":
			canonicalName = "read_file"
			command = terminalReadFileCommand(stringValue(arguments["path"]))
		case "edit_file":
			command = terminalEditFileCommand(stringValue(arguments["path"]), stringValue(arguments["old_text"]), stringValue(arguments["new_text"]))
		case "write_file":
			command = terminalWriteFileCommand(stringValue(arguments["path"]), stringValue(arguments["content"]))
		}
		if command != "" {
			id := strings.TrimSpace(stringValue(call["id"]))
			if id == "" {
				id = fmt.Sprintf("call_response_%d", index+1)
			}
			commands = append(commands, command)
			metadata = append(metadata, terminalNativeCallMetadata{ID: id, Name: canonicalName, Arguments: rawArguments, Command: command})
		}
	}
	if len(commands) == 0 {
		return content
	}
	if content != "" {
		content += "\n"
	}
	content += "```bash\n" + strings.Join(commands, "\n") + "\n```"
	payload, _ := json.Marshal(terminalNativeCallEnvelope{Calls: metadata})
	content += terminalNativeCallMetadataPrefix + hex.EncodeToString(payload) + terminalNativeCallMetadataSuffix
	return content
}

func terminalReadFileCommand(path string) string {
	path = strings.TrimSpace(path)
	if path == "" {
		return ""
	}
	return "cat -- " + shellQuote(path)
}

func terminalEditFileCommand(path, oldText, newText string) string {
	path = strings.TrimSpace(path)
	if path == "" || oldText == "" {
		return ""
	}
	program := `import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
old, new = sys.argv[2], sys.argv[3]
text = p.read_text()
count = text.count(old)
if count != 1:
    print(json.dumps({"error": "old_text_not_unique", "matches": count, "resolved_path": str(p)}))
    raise SystemExit(2)
updated = text.replace(old, new, 1)
p.write_text(updated)
print(json.dumps({"bytes_written": len(updated.encode()), "files_modified": [str(p)], "resolved_path": str(p)}))`
	return "python3 -c " + shellQuote(program) + " " + shellQuote(path) + " " + shellQuote(oldText) + " " + shellQuote(newText)
}
func terminalWriteFileCommand(path, content string) string {
	path = strings.TrimSpace(path)
	if path == "" {
		return ""
	}
	confirmation, _ := json.Marshal(map[string]any{
		"bytes_written":  len([]byte(content)),
		"files_modified": []string{path},
		"resolved_path":  path,
	})
	return "printf %s " + shellQuote(content) + " > " + shellQuote(path) +
		" && printf '%s\\n' " + shellQuote(string(confirmation))
}

type terminalNativeCommandMatch struct {
	index   int
	command string
}

var (
	terminalXMLToolCallPattern       = regexp.MustCompile(`(?s)<tool_call>\s*([A-Za-z0-9_.-]+)(.*?)</tool_call>`)
	terminalXMLArgumentPattern       = regexp.MustCompile(`(?s)<arg_key>\s*([^<]+?)\s*</arg_key>\s*<arg_value>(.*?)</arg_value>`)
	terminalLFMSimpleToolCallPattern = regexp.MustCompile(`(?s)(shell|terminal|read|read_file)\((cmd|command|commands|keystrokes|path)='((?:\\.|[^'])*)'(?:,\s*[^)]*)?\)`)
	terminalLFMWriteFileCallPattern  = regexp.MustCompile(`(?s)write_file\(path='((?:\\.|[^'])*)',\s*content='(.*?)'\)(?:\s*\]|,\s*(?:shell|terminal|read|read_file|write_file)\()`)
	terminalLFMToolCallMarkerPattern = regexp.MustCompile(`(?:^|[\[,\s])(?:shell|terminal|read|read_file|write_file)\(`)
)

func terminalNativeCallFromLFMText(reply string) (terminalNativeCallMetadata, bool) {
	var calls []terminalNativeCallMetadata
	for _, match := range terminalLFMSimpleToolCallPattern.FindAllStringSubmatchIndex(reply, -1) {
		if len(match) != 8 {
			continue
		}
		toolName := reply[match[2]:match[3]]
		argumentName := reply[match[4]:match[5]]
		value := unescapeLFMToolString(reply[match[6]:match[7]])
		var name string
		var arguments map[string]string
		switch toolName {
		case "shell", "terminal":
			if argumentName == "path" || strings.TrimSpace(value) == "" {
				continue
			}
			name = "terminal"
			arguments = map[string]string{"command": value}
		case "read", "read_file":
			if argumentName != "path" || strings.TrimSpace(value) == "" {
				continue
			}
			name = "read_file"
			arguments = map[string]string{"path": value}
		}
		rawArguments, _ := json.Marshal(arguments)
		calls = append(calls, terminalNativeCallMetadata{Name: name, Arguments: string(rawArguments)})
	}
	for _, match := range terminalLFMWriteFileCallPattern.FindAllStringSubmatchIndex(reply, -1) {
		if len(match) != 6 {
			continue
		}
		path := unescapeLFMToolString(reply[match[2]:match[3]])
		content := unescapeLFMToolString(reply[match[4]:match[5]])
		if strings.TrimSpace(path) == "" {
			continue
		}
		rawArguments, _ := json.Marshal(map[string]string{"path": path, "content": content})
		calls = append(calls, terminalNativeCallMetadata{Name: "write_file", Arguments: string(rawArguments)})
	}
	if len(calls) != 1 {
		return terminalNativeCallMetadata{}, false
	}
	return calls[0], true
}

func extractTerminalCommand(reply string) (string, bool) {
	if command, found := extractLFMNativeShellCommand(reply); found {
		return command, true
	}
	if command, found := extractXMLShellCommands(reply); found {
		return command, true
	}
	if terminalLFMToolCallMarkerPattern.MatchString(reply) {
		return "", false
	}
	return extractBashCommand(reply)
}

func extractNativeTerminalCommand(content, reasoning string) (string, bool) {
	if command, found := extractLFMNativeShellCommand(content); found {
		return command, true
	}
	if command, found := extractXMLShellCommands(content); found {
		return command, true
	}
	if command, found := extractLFMNativeShellCommand(reasoning); found {
		return command, true
	}
	if command, found := extractXMLShellCommands(reasoning); found {
		return command, true
	}
	if terminalLFMToolCallMarkerPattern.MatchString(content) || terminalLFMToolCallMarkerPattern.MatchString(reasoning) {
		return "", false
	}
	return extractBashCommand(content)
}

func extractLFMNativeShellCommand(reply string) (string, bool) {
	var translated []terminalNativeCommandMatch
	for _, match := range terminalLFMSimpleToolCallPattern.FindAllStringSubmatchIndex(reply, -1) {
		if len(match) != 8 {
			continue
		}
		toolName := reply[match[2]:match[3]]
		argumentName := reply[match[4]:match[5]]
		value := unescapeLFMToolString(reply[match[6]:match[7]])
		var command string
		switch toolName {
		case "shell", "terminal":
			if argumentName != "path" {
				command = value
			}
		case "read", "read_file":
			if argumentName == "path" {
				command = terminalReadFileCommand(value)
			}
		}
		if strings.TrimSpace(command) != "" {
			translated = append(translated, terminalNativeCommandMatch{index: match[0], command: command})
		}
	}
	for _, match := range terminalLFMWriteFileCallPattern.FindAllStringSubmatchIndex(reply, -1) {
		if len(match) != 6 {
			continue
		}
		path := unescapeLFMToolString(reply[match[2]:match[3]])
		content := unescapeLFMToolString(reply[match[4]:match[5]])
		if command := terminalWriteFileCommand(path, content); command != "" {
			translated = append(translated, terminalNativeCommandMatch{index: match[0], command: command})
		}
	}
	if len(translated) == 0 {
		return "", false
	}
	sort.Slice(translated, func(i, j int) bool { return translated[i].index < translated[j].index })
	commands := make([]string, 0, len(translated))
	for _, match := range translated {
		commands = append(commands, match.command)
	}
	return strings.Join(commands, "\n"), true
}

func unescapeLFMToolString(value string) string {
	var result strings.Builder
	result.Grow(len(value))
	for i := 0; i < len(value); i++ {
		if value[i] != '\\' || i+1 >= len(value) {
			result.WriteByte(value[i])
			continue
		}
		i++
		switch value[i] {
		case 'n':
			result.WriteByte('\n')
		case 'r':
			result.WriteByte('\r')
		case '\\', '\'':
			result.WriteByte(value[i])
		default:
			result.WriteByte('\\')
			result.WriteByte(value[i])
		}
	}
	return result.String()
}

func extractXMLShellCommands(reply string) (string, bool) {
	var commands []string
	for _, toolCall := range terminalXMLToolCallPattern.FindAllStringSubmatch(reply, -1) {
		if !strings.EqualFold(strings.TrimSpace(toolCall[1]), "shell") {
			continue
		}
		for _, argument := range terminalXMLArgumentPattern.FindAllStringSubmatch(toolCall[2], -1) {
			key := strings.ToLower(strings.TrimSpace(argument[1]))
			if key != "cmd" && key != "commands" && key != "keystrokes" {
				continue
			}
			command := strings.TrimSpace(html.UnescapeString(argument[2]))
			if command != "" {
				commands = append(commands, command)
			}
		}
	}
	if len(commands) == 0 {
		return "", false
	}
	return strings.Join(commands, "\n"), true
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
	interpreters := map[string]string{
		"python":     "python3",
		"python3":    "python3",
		"py":         "python3",
		"javascript": "node",
		"js":         "node",
		"node":       "node",
		"ruby":       "ruby",
		"rb":         "ruby",
		"perl":       "perl",
	}
	re := regexp.MustCompile("(?s)```([A-Za-z0-9_+-]+)\\s*\\n(.*?)\\n```")
	for _, match := range re.FindAllStringSubmatch(reply, -1) {
		interpreter := interpreters[strings.ToLower(match[1])]
		if interpreter == "" {
			continue
		}
		body := strings.TrimSpace(match[2])
		if body == "" {
			continue
		}
		return interpreter + " <<'LMX_SCRIPT'\n" + body + "\nLMX_SCRIPT", true
	}
	return "", false
}

func cleanupTerminalRunContainers(ctx context.Context, runLabel string) error {
	if runLabel == "" {
		return nil
	}
	filter := "label=localmaxxing.run=" + runLabel
	out, code, timedOut, err := runCommand(ctx, 30*time.Second, "docker", "ps", "-aq", "--filter", filter)
	if err != nil || timedOut || code != 0 {
		return terminalCommandError("container_cleanup_failed", "Could not inspect containers left by this terminal run.", "docker", []string{"ps", "-aq", "--filter", filter}, code, out, timedOut)
	}
	ids := strings.Fields(out)
	if len(ids) == 0 {
		return nil
	}
	args := append([]string{"rm", "-f"}, ids...)
	out, code, timedOut, err = runCommand(context.Background(), 30*time.Second, "docker", args...)
	if err != nil || timedOut || code != 0 {
		return terminalCommandError("container_cleanup_failed", "Could not remove containers left by this terminal run.", "docker", args, code, out, timedOut)
	}
	return nil
}

func dockerPreflight() error {
	return dockerPreflightContext(context.Background())
}

func dockerPreflightContext(ctx context.Context) error {
	out, code, timedOut, err := runCommand(ctx, 15*time.Second, "docker", "version")
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
	cmd.Cancel = func() error {
		killCommandProcessGroup(cmd)
		return nil
	}
	cmd.WaitDelay = 2 * time.Second
	var output bytes.Buffer
	cmd.Stdout = &output
	cmd.Stderr = &output
	err := cmd.Run()
	if cctx.Err() == context.DeadlineExceeded {
		return output.String(), 124, true, err
	}
	if cctx.Err() != nil {
		return output.String(), 130, false, cctx.Err()
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
		return "Start Docker and ensure this user can access its socket. On Linux, run `sudo usermod -aG docker $USER`, then start a new login session."
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

// parseRewardJSON parses a harbor reward.json payload: a JSON object mapping
// metric names to numbers. The canonical pass signal is the "reward" key.
func parseRewardJSON(text string) (float64, bool) {
	var m map[string]any
	if err := json.Unmarshal([]byte(text), &m); err != nil {
		return 0, false
	}
	v, ok := m["reward"].(float64)
	return v, ok
}

func parseRewardText(text string) (float64, bool) {
	v, err := strconv.ParseFloat(strings.TrimSpace(text), 64)
	if err != nil {
		return 0, false
	}
	return v, true
}

// resolveEnvTemplates expands harbor ${VAR} / ${VAR:-default} templates from
// the host environment, matching harbor's resolve_env_vars behavior.
func resolveEnvTemplates(env map[string]string) map[string]string {
	if len(env) == 0 {
		return env
	}
	resolved := make(map[string]string, len(env))
	for k, v := range env {
		resolved[k] = os.Expand(v, func(name string) string {
			if key, def, ok := strings.Cut(name, ":-"); ok {
				if val, found := os.LookupEnv(key); found && val != "" {
					return val
				}
				return def
			}
			return os.Getenv(name)
		})
	}
	return resolved
}

func nonNilStringMap(in map[string]string) map[string]string {
	if in == nil {
		return map[string]string{}
	}
	return in
}

func writeTerminalTaskTrace(task terminalTask, result terminalTaskResult, cfg terminalConfig) error {
	if cfg.traceRoot == "" {
		return nil
	}
	dir := filepath.Join(cfg.traceRoot, sanitizeDockerName(task.ID))
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return cliError{"trace_write_failed", "Could not create terminal trace directory.", []string{"Check --trace-dir permissions."}, map[string]any{"taskId": task.ID, "dir": dir, "error": err.Error()}}
	}
	files := map[string]string{
		"instruction.txt": task.Instruction,
		"prompt.txt":      result.prompt,
		"transcript.md":   result.transcript,
		"verifier.txt":    result.verifierOutput,
	}
	for name, content := range files {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(content), 0o644); err != nil {
			return cliError{"trace_write_failed", "Could not write terminal trace file.", []string{"Check --trace-dir permissions and available disk space."}, map[string]any{"taskId": task.ID, "file": name, "error": err.Error()}}
		}
	}
	resultJSON := map[string]any{"question_id": task.ID, "pass": result.pass, "scored": result.scored, "error": result.errText, "errorCode": result.errCode, "latencyMs": result.wallTimeMs, "wallTimeMs": result.wallTimeMs, "tokenUsage": result.usage.toMap(), "turns": terminalTaskTurnsValue(result)}
	data, err := json.MarshalIndent(resultJSON, "", "  ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(dir, "result.json"), append(data, '\n'), 0o644); err != nil {
		return cliError{"trace_write_failed", "Could not write terminal trace metadata.", []string{"Check --trace-dir permissions and available disk space."}, map[string]any{"taskId": task.ID, "error": err.Error()}}
	}
	printStatus(cfg.args, "terminal_trace_written", map[string]any{"taskId": task.ID, "dir": dir})
	return nil
}

func tokenUsageFromObject(obj map[string]any) terminalTokenUsage {
	if obj == nil {
		return terminalTokenUsage{}
	}
	input := int64(firstNonZero(usageToken(obj, "prompt_tokens"), usageToken(obj, "input_tokens"), usageToken(obj, "inputTokens"), usageToken(obj, "input")))
	output := int64(firstNonZero(usageToken(obj, "completion_tokens"), usageToken(obj, "output_tokens"), usageToken(obj, "outputTokens"), usageToken(obj, "output")))
	cacheRead := int64(firstNonZero(usageToken(obj, "cache_read_tokens"), usageToken(obj, "cacheReadTokens"), usageToken(obj, "cacheRead")))
	cacheWrite := int64(firstNonZero(usageToken(obj, "cache_write_tokens"), usageToken(obj, "cacheWriteTokens"), usageToken(obj, "cacheWrite")))
	total := int64(firstNonZero(usageToken(obj, "total_tokens"), usageToken(obj, "totalTokens")))
	if total == 0 {
		total = input + output + cacheRead + cacheWrite
	}
	if input == 0 && output == 0 && cacheRead == 0 && cacheWrite == 0 && total == 0 {
		return terminalTokenUsage{}
	}
	modelCalls := firstNonZero(usageToken(obj, "model_calls"), usageToken(obj, "modelCalls"))
	if modelCalls == 0 {
		modelCalls = 1
	}
	return terminalTokenUsage{inputTokens: input, outputTokens: output, cacheReadTokens: cacheRead, cacheWriteTokens: cacheWrite, totalTokens: total, modelCalls: modelCalls}
}

func externalAgentTokenUsage(traceDir string) terminalTokenUsage {
	usage := terminalTokenUsage{}
	_ = filepath.WalkDir(traceDir, func(path string, d os.DirEntry, err error) error {
		if err != nil || d.IsDir() || filepath.Ext(path) != ".jsonl" {
			return nil
		}
		f, err := os.Open(path)
		if err != nil {
			return nil
		}
		defer f.Close()
		scanner := bufio.NewScanner(f)
		scanner.Buffer(make([]byte, 64*1024), 4*1024*1024)
		for scanner.Scan() {
			var event map[string]any
			if err := json.Unmarshal(scanner.Bytes(), &event); err != nil {
				continue
			}
			if stringValue(event["type"]) != "message_end" {
				continue
			}
			message := asObject(event["message"])
			if stringValue(message["role"]) != "assistant" {
				continue
			}
			usage.add(tokenUsageFromObject(asObject(message["usage"])))
		}
		return nil
	})
	return usage
}

func externalAgentTraceText(traceDir string) string {
	ompPaths, _ := filepath.Glob(filepath.Join(traceDir, "*", "omp.jsonl"))
	if direct, err := filepath.Glob(filepath.Join(traceDir, "omp.jsonl")); err == nil {
		ompPaths = append(ompPaths, direct...)
	}
	sort.Strings(ompPaths)
	if len(ompPaths) > 0 {
		preview, _, err := buildTerminalOMPPreview(ompPaths[len(ompPaths)-1], traceDir)
		if err == nil {
			return preview
		}
		return "# Agent trace\n\n## Trace integrity\n\nSource: selected omp.jsonl could not be parsed.\n\n" + terminalMarkdownCode(boundedTerminalUTF8(err.Error(), 2048, "trace error"))
	}
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
func terminalTaskTurnsValue(result terminalTaskResult) any {
	if result.turnsUnreported {
		return (*int)(nil)
	}
	return result.turns
}

func terminalMaxTurnsEnforcement(cfg terminalConfig) string {
	if cfg.oracle {
		return "not-applicable"
	}
	if opt(cfg.args, "agent") == "terminus-2" {
		return "bundled-adapter"
	}
	if cfg.agentCommand != "" {
		return "not-enforced"
	}
	return "cli-agent-loop"
}

func terminalAgentMaxTurns(cfg terminalConfig, task terminalTask) int {
	return firstPositive(cfg.maxTurns, task.Agent.MaxTurns, defaultTerminalMaxTurns)
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

func terminalAgentTimeoutSec(cfg terminalConfig, task terminalTask) int {
	return firstPositive(cfg.agentTimeoutSec, task.Agent.TimeoutSec, defaultTerminalTaskTimeoutSec)
}

func terminalLimitSource(explicit, manifest int) string {
	if explicit > 0 {
		return "cli"
	}
	if manifest > 0 {
		return "task-manifest"
	}
	return "fallback"
}

func terminalCommandTimeoutSec(cfg terminalConfig) int {
	if cfg.commandTimeoutSec > 0 {
		return cfg.commandTimeoutSec
	}
	return defaultTerminalCommandTimeoutSec
}

func terminalObservationForModel(output string, exitCode int, timedOut bool) string {
	shown := truncateString(output, terminalModelObservationLimit)
	if timedOut {
		shown += "\n[timeout recovery hint: do not repeat the same long command. Use a smaller bounded command, inspect partial state, or choose a non-interactive alternative.]"
	}
	return shown + "\n[exit=" + strconv.Itoa(exitCode) + "]"
}

func compactAssistantForModel(content, command string) string {
	if command != "" {
		return "```bash\n" + command + "\n```"
	}
	return truncateString(content, 1000)
}

func trimTerminalMessages(messages []map[string]any) []map[string]any {
	return trimTerminalMessagesTo(messages, terminalMessageBudgetBytes, terminalRecentMessageKeep)
}

func trimTerminalMessagesForRetry(messages []map[string]any) []map[string]any {
	return trimTerminalMessagesTo(messages, terminalMessageBudgetBytes/2, terminalRecentMessageKeep/2)
}

func trimTerminalMessagesTo(messages []map[string]any, budget, keep int) []map[string]any {
	if terminalMessagesBytes(messages) <= budget || len(messages) <= 3 {
		return messages
	}
	if keep < 2 {
		keep = 2
	}
	if keep > len(messages)-2 {
		keep = len(messages) - 2
	}
	out := make([]map[string]any, 0, keep+3)
	out = append(out, messages[0], messages[1])
	omitted := len(messages) - 2 - keep
	if omitted > 0 {
		out = append(out, map[string]any{"role": "user", "content": fmt.Sprintf("[Earlier terminal transcript compacted: %d old messages omitted. Continue from the recent state below; rerun concise inspection commands if exact details are needed.]", omitted)})
	}
	out = append(out, messages[len(messages)-keep:]...)
	return out
}

func terminalMessagesBytes(messages []map[string]any) int {
	total := 0
	for _, message := range messages {
		encoded, err := json.Marshal(message)
		if err == nil {
			total += len(encoded)
		}
	}
	return total
}

func terminalCommandExecutionTimeout(_ string, requested time.Duration, deadline time.Time) time.Duration {
	return terminalRemainingTimeout(requested, deadline)
}

func looksLikeSetupOrBlockingCommand(command string) bool {
	lower := strings.ToLower(command)
	if strings.Contains(lower, "timeout ") {
		return false
	}
	markers := []string{
		"apt-get", " apt ", "pip install", "uv pip install", "npm install", "pnpm install", "yarn install",
		"cargo install", "go install", "conda install", "mamba install", "7z x", "7za x", "john ", "hashcat",
		"python3 -m http.server", "python -m http.server",
	}
	for _, marker := range markers {
		if strings.Contains(lower, marker) {
			return true
		}
	}
	return false
}

func terminalModelMaxTokens(cfg terminalConfig, retry bool) int {
	if cfg.maxTokens > 0 {
		return cfg.maxTokens
	}
	if retry {
		return defaultTerminalRetryMaxTokens
	}
	return defaultTerminalModelMaxTokens
}

func terminalDisablesTemplateThinking(model string) bool {
	return strings.Contains(strings.ToLower(model), "qwen")
}

func terminalModelCallFailure(taskID string, deadline time.Time, firstErr, retryErr error) cliError {
	fields := map[string]any{"taskId": taskID, "firstError": firstErr.Error(), "retryError": retryErr.Error()}
	if !time.Now().Before(deadline) {
		return cliError{"agent_timeout", "Terminal agent timed out during model call.", []string{"Raise --agent-timeout, lower --max-tokens, or use a faster model/server."}, fields}
	}
	return cliError{"model_call_failed", "Model call failed during terminal agent loop.", []string{"Check --base-url, --model-api-key, and that the OpenAI-compatible server is running."}, fields}
}

func terminalModelRequestTimeout(cfg terminalConfig, deadline time.Time, reserveRetry bool) time.Duration {
	remaining := time.Until(deadline)
	if remaining <= 0 {
		return time.Second
	}
	if reserveRetry && remaining > 2*time.Second {
		reserve := remaining / 3
		if reserve > maxTerminalRetryReserve {
			reserve = maxTerminalRetryReserve
		}
		if reserve < time.Second {
			reserve = time.Second
		}
		remaining -= reserve
	}
	configured := terminalEndpointTimeout(cfg)
	if configured <= 0 || remaining < configured {
		return remaining
	}
	return configured
}

func terminalEndpointTimeout(cfg terminalConfig) time.Duration {
	if cfg.endpointTimeout > 0 {
		return cfg.endpointTimeout
	}
	return defaultTerminalEndpointTimeout
}

func terminalRemainingTimeout(defaultTimeout time.Duration, deadline time.Time) time.Duration {
	remaining := time.Until(deadline)
	if remaining <= 0 {
		return time.Second
	}
	if defaultTimeout <= 0 || remaining < defaultTimeout {
		return remaining
	}
	return defaultTimeout
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
	if img.ComposeFile != "" {
		return "docker-compose"
	}
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
	return shardItemResult{questionID: b.Task.ID, itemIndex: i, pass: r.pass, scored: r.scored, errText: r.errText, question: b.Task.Instruction, prompt: r.prompt, response: r.transcript, reasoning: r.verifierOutput, latencyMs: r.wallTimeMs}
}

var _ = bufio.ErrFinalToken
