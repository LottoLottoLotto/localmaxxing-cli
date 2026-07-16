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

const defaultTerminalTaskTimeoutSec = 4 * 60 * 60
const defaultTerminalCommandTimeoutSec = defaultTerminalTaskTimeoutSec
const terminalModelObservationLimit = 2500
const terminalMessageBudgetBytes = 60_000
const terminalRecentMessageKeep = 12
const defaultTerminalModelMaxTokens = 16_384
const defaultTerminalRetryMaxTokens = 8_192
const defaultTerminalEndpointTimeout = 10 * time.Minute
const maxTerminalRetryReserve = 5 * time.Minute
const terminalEndpointProbeTimeout = 2 * time.Second
const terminalIdentityLookupTimeout = 30 * time.Second
const terminalMonolithicArtifactMaxBytes int64 = 256 * 1024 * 1024
const terminalMonolithicResultLimit = 1000

var errTerminalMonolithicResultLimit = errors.New("monolithic terminal result limit exceeded")

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

type terminalSolutionConfig struct {
	Env map[string]string `json:"env,omitempty"`
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
	traceRoot         string
	endpointTimeout   time.Duration
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
	pass             bool
	scored           bool
	turns            int
	transcript       string
	verifierOutput   string
	wallTimeMs       int64
	usage            terminalTokenUsage
	errText          string
	errCode          string
	agentOutcomeCode string
	agentOutcomeText string
	instruction      string
	prompt           string
}

type terminalAgentOutcomeError struct {
	code string
	text string
}

func (e terminalAgentOutcomeError) Error() string { return e.text }

func captureTerminalAgentOutcome(result *terminalTaskResult, err error) bool {
	var outcome terminalAgentOutcomeError
	if !errors.As(err, &outcome) {
		return false
	}
	result.agentOutcomeCode = outcome.code
	result.agentOutcomeText = outcome.text
	return true
}

type terminalEndpointMetadata struct {
	baseURL      string
	servedModel  string
	quantization string
	modelPath    string
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
	case "inspect":
		return inspectTerminalDataset(positional(args, 3), args)
	case "run":
		return runTerminalEval(args, false)
	case "submit":
		return submitTerminalEval(args)
	case "verify":
		return runTerminalEval(args, true)
	default:
		return cliError{"unknown_subcommand", "Unknown eval terminal subcommand.", []string{"Use one of: import, inspect, run, submit, verify."}, map[string]any{"subcommand": sub}}
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
	imported, skipped := 0, 0
	for _, taskDir := range taskDirs {
		if composeFile := harborComposeFile(taskDir); composeFile != "" {
			printStatus(args, "terminal_import_skipped", map[string]any{"taskId": filepath.Base(filepath.Clean(taskDir)), "reason": "docker-compose task environments are not supported by this runner", "composeFile": composeFile})
			skipped++
			continue
		}
		if err := importHarborTask(taskDir, out, version); err != nil {
			return err
		}
		imported++
	}
	if imported == 0 {
		return cliError{"task_import_failed", "Every harbor task was skipped.", []string{"docker-compose task environments are not supported; import tasks that use environment/Dockerfile or [environment].docker_image."}, map[string]any{"src": src, "skipped": skipped}}
	}
	printInfo("terminal_import_complete", map[string]any{"src": src, "out": out, "tasks": imported, "skipped": skipped, "version": version})
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
		Solution:    terminalSolutionConfig{Env: nonNilStringMap(ht.Solution.Env)},
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
	if localDir != "" && dataset == "" {
		dataset = opt(args, "dataset")
	}
	if !forceOracle && localDir == "" && dataset == "" {
		return cliError{"missing_option", "eval terminal run requires a dataset slug or --task-dir.", []string{"Run: lmx eval terminal run terminal-bench-2-1 --api-url <localmaxxing-origin> --base-url <model-origin> --model <hfId>.", "For local bundles, run: lmx eval terminal run --task-dir ./bundles --api-url <localmaxxing-origin> --base-url <model-origin> --model <hfId>."}, nil}
	}
	if !forceOracle && !hasFlag(args, "oracle") {
		if opt(args, "endpoint-file") != "" && opt(args, "base-url") != "" {
			return cliError{"endpoint_selection_conflict", "--endpoint-file cannot be combined with --base-url.", []string{"Choose the trusted endpoint file or pass the model endpoint explicitly, not both."}, nil}
		}
		if opt(args, "model-api-key") != "" && opt(args, "base-url") == "" && opt(args, "endpoint-file") == "" {
			return cliError{"endpoint_credentials_require_explicit_target", "--model-api-key requires an explicit --base-url or trusted --endpoint-file.", []string{"Do not attach model credentials while probing broad localhost candidates."}, nil}
		}
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
	if len(bundles) > terminalMonolithicResultLimit && opt(args, "out") != "" {
		return cliError{"bundle_count_invalid", "The terminal run selects too many task bundles for one bounded artifact.", []string{"Split the evaluation into smaller shards."}, map[string]any{"tasks": len(bundles), "maxTasks": terminalMonolithicResultLimit}}
	}
	if err := dockerPreflight(); err != nil {
		return err
	}

	rawBaseURL := opt(args, "base-url")
	baseURL := ""
	callModel := ""
	declaredModel := opt(args, "model")
	servedModel := opt(args, "served-model")
	servedModelSource := "cli"
	var servedModelInfo map[string]any
	var quantResolution map[string]any
	resolvedQuant, resolvedQuantFormat := "", opt(args, "quant-format")
	modelPath := opt(args, "model-path")
	verifiedModelPath := ""
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
		endpointFile := opt(args, "endpoint-file")
		explicitServedModel := servedModel
		explicitModelPath := modelPath
		explicitQuantization := opt(args, "quantization")
		if endpointFile != "" {
			savedEndpoint, loadErr := loadTerminalEndpointFile(endpointFile)
			if loadErr != nil {
				return loadErr
			}
			rawBaseURL = savedEndpoint.baseURL
			printStatus(args, "terminal_endpoint_file_selected", map[string]any{"baseUrl": terminalSanitizedEndpointOrigin(rawBaseURL), "path": endpointFile})
		}

		if agentCommand == "" && rawBaseURL == "" {
			rawBaseURL, servedModel, servedModelInfo, err = discoverTerminalEndpoint(args, declaredModel)
			if err != nil {
				return err
			}
			servedModelSource = "endpoint_discovery"
		} else if agentCommand != "" && rawBaseURL == "" {
			rawBaseURL = firstNonEmpty(os.Getenv("LM_STUDIO_BASE_URL"), os.Getenv("LLAMA_CPP_BASE_URL"))
		}

		if rawBaseURL != "" {
			baseURL = openAIBaseURL(rawBaseURL)
			if servedModelInfo == nil {
				preferred := explicitServedModel
				detected, info, probeErr := probeTerminalEndpoint(baseURL, opt(args, "model-api-key"), preferred, explicitServedModel != "", terminalEndpointProbeTimeout)
				if probeErr != nil {
					return cliError{"model_detection_failed", "Could not reconcile the selected endpoint with its live /v1/models metadata.", []string{"Check the endpoint, or correct --served-model to an exact model id exposed by that endpoint."}, map[string]any{"baseUrl": terminalSanitizedEndpointOrigin(baseURL), "error": probeErr.Error()}}
				}
				servedModel = detected
				servedModelInfo = info
				if explicitServedModel == "" {
					servedModelSource = "v1_models"
				}
			}
			callModel = servedModel

			liveModelPath := terminalModelPathFromMetadata(servedModelInfo)
			liveQuantization := terminalQuantizationFromMetadata(servedModelInfo)
			if props, propsErr := fetchTerminalEndpointJSON(baseURL+"/props", opt(args, "model-api-key"), terminalEndpointProbeTimeout); propsErr == nil {
				if obj := asObject(props); obj != nil {
					liveModelPath = firstNonEmpty(terminalModelPathFromMetadata(obj), liveModelPath)
					liveQuantization = firstNonEmpty(terminalQuantizationFromMetadata(obj), liveQuantization)
				}
			}
			verifiedModelPath = liveModelPath
			modelPath, err = reconcileTerminalEndpointField("model-path", explicitModelPath, liveModelPath, true)
			if err != nil {
				return err
			}
			_, err = reconcileTerminalEndpointField("quantization", explicitQuantization, liveQuantization, false)
			if err != nil {
				return err
			}
			quantResolution, err = terminalEndpointQuantizationResolution(explicitQuantization, liveQuantization, liveModelPath)
			if err != nil {
				return err
			}
			if quantResolution == nil {
				quantResolution = map[string]any{}
			}
			resolvedQuant = stringValue(quantResolution["trusted"])
			if modelPath != "" {
				quantResolution["modelPath"] = modelPath
			}
			if resolvedQuantFormat == "" && strings.EqualFold(filepath.Ext(modelPath), ".gguf") {
				resolvedQuantFormat = "gguf"
			}
		} else if agentCommand != "" {
			callModel = firstNonEmpty(servedModel, declaredModel, "external-agent")
			printStatus(args, "eval_model_detection_unavailable", map[string]any{"reason": "external agent did not provide --base-url, --endpoint-file, LM_STUDIO_BASE_URL, or LLAMA_CPP_BASE_URL"})
		} else {
			return cliError{"endpoint_discovery_failed", "No model endpoint was selected.", []string{"Start a supported local endpoint, pass --base-url, or pass --endpoint-file from lmx endpoint discover --out."}, nil}
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
	cfg := terminalConfig{apiKey: opt(args, "model-api-key"), args: args, maxTokens: maxTokens, temperature: floatOption(args, "temperature", 0), topP: floatOption(args, "top-p", 1), commandTimeoutSec: commandTimeout, cleanupImages: hasFlag(args, "cleanup-images"), oracle: forceOracle || hasFlag(args, "oracle"), agentCommand: agentCommand, agentExecution: firstNonEmpty(opt(args, "agent-execution"), map[bool]string{true: "routed-shell"}[agentBackend == "terminus-2"], "host"), traceRoot: opt(args, "trace-dir")}
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

	var hardware any
	if hardwarePath := opt(args, "hardware"); hardwarePath != "" {
		hardware, err = readJSON(hardwarePath)
		if err != nil {
			return err
		}
		hardware = normalizeHardwarePayload(hardware)
	}

	resolvedHFID := ""
	if !cfg.oracle {
		resolvedHFID, modelResolution, err = resolveTerminalModelIdentityChecked(argsWithTerminalBaseURL(args, rawBaseURL), declaredModel, servedModel, servedModelSource, verifiedModelPath)
		if err != nil {
			return err
		}
	}
	protocol := terminalProtocolLabel(cfg)
	agentName := "lmx-terminus"
	if cfg.oracle {
		agentName = "oracle"
	} else if cfg.agentCommand != "" {
		agentName = firstNonEmpty(opt(args, "agent-name"), agentBackend, "external-agent")
	}
	runConfig := map[string]any{
		"protocol":              protocol,
		"agent":                 agentName,
		"maxTurns":              cfg.maxTurns,
		"maxTokens":             cfg.maxTokens,
		"temperature":           cfg.temperature,
		"topP":                  cfg.topP,
		"commandTimeoutSec":     cfg.commandTimeoutSec,
		"agentTimeoutSec":       cfg.agentTimeoutSec,
		"concurrency":           concurrency,
		"shellMode":             cfg.shellMode,
		"agentExecution":        cfg.agentExecution,
		"oracle":                cfg.oracle,
		"modelEndpoint":         terminalSanitizedEndpointOrigin(baseURL),
		"servedModelSource":     servedModelSource,
		"endpointTimeoutMillis": terminalEndpointTimeout(cfg).Milliseconds(),
	}
	if cfg.agentCommand != "" {
		runConfig["agentBackend"] = firstNonEmpty(agentBackend, "custom")
		runConfig["toolRouting"] = map[string]any{"shell": cfg.agentExecution, "workdir": "/app", "hostFilesystemVisible": cfg.agentExecution == "host"}
	}

	printInfo("terminal_eval_start", map[string]any{"dataset": firstNonEmpty(dataset, "local"), "tasks": len(bundles), "model": firstNonEmpty(callModel, "oracle"), "baseUrl": terminalSanitizedEndpointOrigin(baseURL), "concurrency": concurrency, "oracle": cfg.oracle})
	results := runTerminalBundles(args, bundles, baseURL, callModel, cfg, concurrency)
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
	runConfig["accuracy"] = accuracy
	runConfig["tasksRun"] = len(results)
	runConfig["errors"] = stats.errors
	runConfig["avgLatencyMs"] = avgLatency
	runConfig["modelResolution"] = modelResolution
	runConfig["quantizationResolution"] = quantResolution
	summary := map[string]any{
		"artifactVersion":        2,
		"dataset":                firstNonEmpty(dataset, "local"),
		"tasks":                  len(results),
		"correct":                stats.correct,
		"scored":                 stats.scored,
		"errors":                 stats.errors,
		"accuracyPct":            roundMetric(accuracy * 100),
		"avgWallTimeMs":          avgLatency,
		"wallTimeMs":             stats.totalLatencyMs,
		"tokenUsage":             totalUsage.toMap(),
		"servedModel":            servedModel,
		"declaredModel":          declaredModel,
		"modelRevision":          firstNonEmpty(opt(args, "model-revision"), "main"),
		"modelResolution":        modelResolution,
		"quantization":           resolvedQuant,
		"quantFormat":            resolvedQuantFormat,
		"quantizationResolution": quantResolution,
		"runConfig":              runConfig,
		"agent":                  agentName,
		"runnerVersion":          firstNonEmpty(opt(args, "runner-version"), "localmaxxing-go terminal-agent"),
		"errorCodes":             errorCodes,
	}
	if shardIndex >= 1 {
		summary["shardIndex"] = shardIndex
	}
	if resolvedHFID != "" {
		summary["hfId"] = resolvedHFID
	}
	if hardware != nil {
		summary["hardware"] = hardware
	}

	out := opt(args, "out")
	printTerminalFailureSummary(args, bundles, results, cfg, out)
	if out != "" {
		records := make([]any, len(results))
		for i, r := range results {
			records[i] = map[string]any{"question_id": bundles[i].Task.ID, "pass": r.pass, "scored": r.scored, "error": boundedTerminalUTF8(r.errText, 2*1024, "saved task error"), "errorCode": r.errCode, "agentOutcomeCode": r.agentOutcomeCode, "agentOutcome": boundedTerminalUTF8(r.agentOutcomeText, 1024, "saved agent outcome"), "latencyMs": r.wallTimeMs, "wallTimeMs": r.wallTimeMs, "tokenUsage": r.usage.toMap(), "turns": r.turns, "question": boundedTerminalUTF8(bundles[i].Task.Instruction, 4*1024, "saved task question"), "prompt": boundedTerminalUTF8(r.prompt, 4*1024, "saved task prompt"), "response": boundedTerminalUTF8(r.transcript, 16*1024, "saved task response"), "verifierOutput": boundedTerminalUTF8(r.verifierOutput, 4*1024, "saved verifier output")}
		}
		if err := writeJSON(out, map[string]any{"summary": summary, "results": records}); err != nil {
			return err
		}
		printStatus(args, "terminal_results_written", map[string]any{"path": out})
	}
	if !hasFlag(args, "submit") {
		if stats.errors == len(results) {
			code := dominantTerminalErrorCode(errorCodes)
			return cliError{code, "Every terminal task errored before scoring.", []string{"Inspect --out results.json and the terminal_task_error events."}, map[string]any{"errors": errorCodes}}
		}
		printInfo("terminal_eval_complete", summary)
		fmt.Println("Terminal execution completed; results were not submitted.")
		if out != "" {
			fmt.Println("Submit later with: " + terminalDeferredSubmitCommand(args, out, summary))
		}
		return nil
	}
	key := apiKey(args)
	if key == "" {
		return missingAPIKey("--api-key or LMX_API_KEY is required for eval terminal --submit")
	}
	if resolvedHFID == "" {
		return cliError{"missing_model", "eval terminal --submit requires a canonical HuggingFace model id.", []string{"Pass --model org/name, or use an endpoint that exposes the loaded model filename so it can be resolved."}, nil}
	}
	if hardware == nil {
		return cliError{"missing_hardware", "eval terminal --submit requires --hardware hardware.json", []string{"Run lmx hardware --out hardware.json and pass --hardware hardware.json."}, nil}
	}
	submitResults := []any{}
	submitArtifacts := []any{}
	for i, r := range results {
		if !r.scored {
			continue
		}
		task := bundles[i].Task
		submitResults = append(submitResults, map[string]any{"question_id": task.ID, "pass": r.pass})
		artifactResponse := truncateString(r.transcript+"\n\n# Verifier\n\n"+r.verifierOutput, 4_900_000)
		submitArtifacts = append(submitArtifacts, map[string]any{"question_id": task.ID, "itemIndex": i, "promptHash": shortHash(task.ID + ":" + r.prompt), "question": task.Instruction, "prompt": r.prompt, "response": artifactResponse, "score": boolScore(r.pass), "testPassed": r.pass, "latencyMs": r.wallTimeMs, "wallTimeMs": r.wallTimeMs, "tokenUsage": r.usage.toMap()})
	}
	if len(submitResults) == 0 {
		return cliError{"no_scored_questions", "Every terminal task failed to score, so there is nothing to submit.", []string{"Check Docker and the model endpoint.", "Inspect failures with --out results.json."}, map[string]any{"errors": errorCodes}}
	}
	if rawBaseURL != "" && opt(args, "quantization") == "" && resolvedQuant == "" {
		return cliError{"model_detection_failed", "Could not verify the local endpoint model/quantization before submission.", []string{"Keep the model endpoint running through submission.", "Or pass --quantization and --quant-format explicitly if the endpoint cannot expose model metadata."}, map[string]any{"baseUrl": terminalSanitizedEndpointOrigin(baseURL)}}
	}
	payload := map[string]any{"hfId": resolvedHFID, "modelRevision": summary["modelRevision"], "hardware": hardware, "results": submitResults, "artifacts": submitArtifacts, "runnerVersion": firstNonEmpty(opt(args, "runner-version"), "localmaxxing-go terminal-agent"), "runConfig": runConfig}
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

func loadTerminalEndpointFile(filename string) (terminalEndpointMetadata, error) {
	value, err := readJSON(filename)
	if err != nil {
		return terminalEndpointMetadata{}, err
	}
	root := asObject(value)
	if root == nil {
		return terminalEndpointMetadata{}, cliError{"endpoint_file_invalid", "Endpoint discovery file must contain a JSON object.", []string{"Create it with lmx endpoint discover --out endpoint.json."}, map[string]any{"path": filename}}
	}
	items, ok := root["endpoints"].([]any)
	if !ok {
		return terminalEndpointMetadata{}, cliError{"endpoint_file_invalid", "Endpoint discovery file is missing its endpoints array.", []string{"Create it with lmx endpoint discover --out endpoint.json."}, map[string]any{"path": filename}}
	}
	selected := make([]map[string]any, 0, 1)
	for _, item := range items {
		obj := asObject(item)
		if obj == nil {
			continue
		}
		entryOK, _ := obj["ok"].(bool)
		if entryOK {
			selected = append(selected, obj)
		}
	}
	if len(selected) == 0 {
		return terminalEndpointMetadata{}, cliError{"endpoint_file_no_selection", "Endpoint discovery file contains no ok:true endpoint.", []string{"Regenerate it while exactly one intended endpoint is available."}, map[string]any{"path": filename}}
	}
	if len(selected) > 1 {
		return terminalEndpointMetadata{}, cliError{"endpoint_file_ambiguous", "Endpoint discovery file contains more than one ok:true endpoint.", []string{"Remove unintended entries, leaving one selected endpoint."}, map[string]any{"path": filename, "selectedEndpoints": len(selected)}}
	}
	entry := selected[0]
	metadata := terminalEndpointMetadata{
		baseURL:      stringValue(entry["baseUrl"]),
		servedModel:  stringValue(entry["servedModel"]),
		quantization: stringValue(entry["quantization"]),
		modelPath:    firstNonEmpty(stringValue(entry["modelPath"]), stringValue(entry["model_path"])),
	}
	if serverMetadata := asObject(entry["serverMetadata"]); serverMetadata != nil {
		metadata.quantization = firstNonEmpty(metadata.quantization, stringValue(serverMetadata["quantization"]))
		metadata.modelPath = firstNonEmpty(metadata.modelPath, stringValue(serverMetadata["modelPath"]), stringValue(serverMetadata["model_path"]))
	}
	if metadata.baseURL == "" {
		return terminalEndpointMetadata{}, cliError{"endpoint_file_invalid", "The selected endpoint has no baseUrl.", []string{"Regenerate the file with lmx endpoint discover --out endpoint.json."}, map[string]any{"path": filename}}
	}
	return metadata, nil
}

func discoverTerminalEndpoint(args cliArgs, _ string) (string, string, map[string]any, error) {
	probeKey := opt(args, "model-api-key")
	if probeKey != "" && opt(args, "base-url") == "" {
		return "", "", nil, cliError{"endpoint_credentials_require_explicit_target", "--model-api-key requires an explicit --base-url or trusted --endpoint-file.", []string{"Do not attach model credentials while probing broad localhost candidates."}, nil}
	}
	attempts := make([]any, 0)
	preferred := strings.TrimSpace(opt(args, "served-model"))
	type candidateMatch struct {
		index       int
		baseURL     string
		servedModel string
		info        map[string]any
	}
	matches := make([]candidateMatch, 0, 1)
	ambiguousCandidates := make([]string, 0)
	for index, candidate := range endpointDiscoveryCandidates(args) {
		servedModel, info, err := probeTerminalEndpoint(candidate, probeKey, preferred, preferred != "", terminalEndpointProbeTimeout)
		if err != nil {
			attempts = append(attempts, map[string]any{"baseUrl": terminalSanitizedEndpointOrigin(candidate), "error": err.Error()})
			if strings.Contains(strings.ToLower(err.Error()), "ambiguous") {
				ambiguousCandidates = append(ambiguousCandidates, terminalSanitizedEndpointOrigin(candidate))
			}
			continue
		}
		matches = append(matches, candidateMatch{index: index, baseURL: candidate, servedModel: servedModel, info: info})
	}
	if len(ambiguousCandidates) > 0 {
		candidates := append([]string{}, ambiguousCandidates...)
		for _, match := range matches {
			candidates = append(candidates, terminalSanitizedEndpointOrigin(match.baseURL))
		}
		return "", "", nil, cliError{"endpoint_discovery_ambiguous", "Automatic discovery found a healthy endpoint with multiple models or more than one healthy endpoint.", []string{"Pass --served-model with an exact id, or select a trusted single-model endpoint explicitly."}, map[string]any{"servedModelSelector": preferred, "candidates": candidates, "attempts": attempts}}
	}
	if len(matches) == 0 {
		return "", "", nil, cliError{"endpoint_discovery_failed", "No supported local model endpoint exposed an unambiguous served model.", []string{"Start the intended model server, pass --served-model with its exact live id, or pass --base-url/--endpoint-file explicitly."}, map[string]any{"servedModelSelector": preferred, "attempts": attempts}}
	}
	if len(matches) > 1 {
		candidates := make([]string, len(matches))
		for i, match := range matches {
			candidates[i] = terminalSanitizedEndpointOrigin(match.baseURL)
		}
		return "", "", nil, cliError{"endpoint_discovery_ambiguous", "More than one local model endpoint is healthy or matches the exact --served-model selector.", []string{"Pass --base-url or a trusted --endpoint-file to select one endpoint atomically."}, map[string]any{"servedModelSelector": preferred, "candidates": candidates}}
	}
	match := matches[0]
	printStatus(args, "terminal_endpoint_discovered", map[string]any{"baseUrl": terminalSanitizedEndpointOrigin(match.baseURL), "servedModel": match.servedModel, "candidateIndex": match.index + 1})
	return match.baseURL, match.servedModel, match.info, nil
}

func probeTerminalEndpoint(baseURL, apiKey, preferred string, requirePreferred bool, timeout time.Duration) (string, map[string]any, error) {
	value, err := fetchTerminalEndpointJSON(openAIBaseURL(baseURL)+"/v1/models", apiKey, timeout)
	if err != nil {
		return "", nil, err
	}
	body := asObject(value)
	if body == nil {
		return "", nil, errors.New("/v1/models did not return a JSON object")
	}
	type modelMatch struct {
		id   string
		info map[string]any
	}
	models := make([]modelMatch, 0)
	seen := map[string]bool{}
	for _, item := range modelInfoItems(body) {
		obj := asObject(item)
		if obj == nil {
			continue
		}
		id := firstNonEmpty(stringValue(obj["id"]), stringValue(obj["name"]), stringValue(obj["model"]))
		key := strings.ToLower(strings.TrimSpace(id))
		if key == "" || seen[key] {
			continue
		}
		seen[key] = true
		models = append(models, modelMatch{id: id, info: obj})
		if preferred != "" && strings.EqualFold(id, preferred) {
			return id, obj, nil
		}
	}
	if preferred != "" && requirePreferred {
		return "", nil, fmt.Errorf("/v1/models does not expose requested model %q", preferred)
	}
	if len(models) == 1 {
		return models[0].id, models[0].info, nil
	}
	if len(models) == 0 {
		return "", nil, errors.New("/v1/models did not return any model ids")
	}
	ids := make([]string, len(models))
	for i, model := range models {
		ids[i] = model.id
	}
	return "", nil, fmt.Errorf("/v1/models is ambiguous; select one of %s with --served-model", strings.Join(ids, ", "))
}

func fetchTerminalEndpointJSON(rawURL, apiKey string, timeout time.Duration) (any, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, rawURL, nil)
	if err != nil {
		return nil, errors.New("invalid model endpoint URL")
	}
	if apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}
	res, err := apiHTTPClient.Do(req)
	if err != nil {
		if errors.Is(err, context.DeadlineExceeded) || errors.Is(ctx.Err(), context.DeadlineExceeded) {
			return nil, fmt.Errorf("model endpoint probe timed out after %s", timeout)
		}
		return nil, errors.New("model endpoint probe failed")
	}
	defer res.Body.Close()
	if res.StatusCode < 200 || res.StatusCode >= 300 {
		return nil, fmt.Errorf("endpoint returned %s", res.Status)
	}
	var body any
	if err := json.NewDecoder(io.LimitReader(res.Body, 4*1024*1024)).Decode(&body); err != nil {
		return nil, err
	}
	return body, nil
}

func terminalModelPathFromMetadata(metadata map[string]any) string {
	return terminalModelPathFromMetadataDepth(metadata, 0)
}

func terminalModelPathFromMetadataDepth(metadata map[string]any, depth int) string {
	if metadata == nil || depth >= 4 {
		return ""
	}
	if value := firstNonEmpty(stringValue(metadata["model_path"]), stringValue(metadata["modelPath"]), stringValue(metadata["filename"]), stringValue(metadata["path"])); value != "" {
		return value
	}
	for _, key := range []string{"metadata", "meta", "serverMetadata"} {
		if nested := asObject(metadata[key]); nested != nil {
			if value := terminalModelPathFromMetadataDepth(nested, depth+1); value != "" {
				return value
			}
		}
	}
	return ""
}

func terminalQuantizationFromMetadata(metadata map[string]any) string {
	return terminalQuantizationFromMetadataDepth(metadata, 0)
}

func terminalQuantizationFromMetadataDepth(metadata map[string]any, depth int) string {
	if metadata == nil || depth >= 4 {
		return ""
	}
	if value := firstNonEmpty(stringValue(metadata["quantization"]), stringValue(metadata["quant"])); value != "" {
		return value
	}
	for _, key := range []string{"metadata", "meta", "serverMetadata"} {
		if nested := asObject(metadata[key]); nested != nil {
			if value := terminalQuantizationFromMetadataDepth(nested, depth+1); value != "" {
				return value
			}
		}
	}
	return ""
}

func terminalEndpointQuantizationResolution(explicit, live, liveModelPath string) (map[string]any, error) {
	resolution := map[string]any{"cli": explicit}
	if live != "" {
		resolution["v1Models"] = live
	}
	if liveModelPath != "" {
		resolution["modelPath"] = liveModelPath
		if filenameQuantization := quantizationFromFilename(liveModelPath); filenameQuantization != "" {
			resolution["filename"] = filenameQuantization
		}
	}
	trusted := firstNonEmpty(stringValue(resolution["filename"]), live, explicit)
	if trusted == "" {
		return nil, nil
	}
	if live != "" && !quantizationEqual(live, trusted) {
		return nil, cliError{"endpoint_metadata_conflict", "Live endpoint quantization metadata is inconsistent.", []string{"Correct the endpoint metadata before running the evaluation."}, map[string]any{"live": live, "resolved": trusted}}
	}
	if explicit != "" && !quantizationEqual(explicit, trusted) {
		return nil, cliError{"endpoint_metadata_conflict", "--quantization conflicts with the live endpoint artifact metadata.", []string{"Correct --quantization or load the intended model artifact."}, map[string]any{"explicit": explicit, "live": trusted}}
	}
	resolution["trusted"] = trusted
	switch {
	case stringValue(resolution["filename"]) != "":
		resolution["trustedSource"] = "live_filename"
	case live != "":
		resolution["trustedSource"] = "live_endpoint"
	default:
		resolution["trustedSource"] = "cli"
	}
	resolution["status"] = "matched"
	return resolution, nil
}

func reconcileTerminalEndpointField(field, explicit, live string, compareFilename bool) (string, error) {
	if explicit == "" {
		return live, nil
	}
	if live == "" {
		return explicit, nil
	}
	left, right := explicit, live
	matches := strings.EqualFold(left, right)
	if compareFilename {
		left, right = filepath.Base(left), filepath.Base(right)
		matches = strings.EqualFold(left, right)
	} else if field == "quantization" {
		matches = quantizationEqual(left, right)
	}
	if !matches {
		return "", cliError{"endpoint_metadata_conflict", "--" + field + " conflicts with metadata reported by the live endpoint.", []string{"Correct the explicit value or load the intended endpoint."}, map[string]any{"field": field, "explicit": explicit, "live": live}}
	}
	return live, nil
}

func terminalSanitizedEndpointOrigin(raw string) string {
	parsed, err := url.Parse(openAIBaseURL(raw))
	if err != nil || parsed.Scheme == "" || parsed.Host == "" {
		return ""
	}
	parsed.User = nil
	parsed.Path = ""
	parsed.RawPath = ""
	parsed.RawQuery = ""
	parsed.Fragment = ""
	return strings.TrimRight(parsed.String(), "/")
}

func terminalSanitizedDownloadURL(raw string) string {
	parsed, err := url.Parse(strings.TrimSpace(raw))
	if err != nil || parsed.Scheme == "" || parsed.Host == "" {
		return ""
	}
	parsed.User = nil
	parsed.RawQuery = ""
	parsed.ForceQuery = false
	parsed.Fragment = ""
	parsed.RawFragment = ""
	return parsed.String()
}

func terminalSafeDownloadError(err error) string {
	var urlErr *url.Error
	if errors.As(err, &urlErr) {
		return "request failed"
	}
	return err.Error()
}

func resolveTerminalModelIdentity(args cliArgs, declared, servedModel, servedModelSource, modelPath string) (string, map[string]any) {
	resolved, resolution, err := resolveTerminalModelIdentityChecked(args, declared, servedModel, servedModelSource, modelPath)
	if err != nil {
		if resolution == nil {
			resolution = map[string]any{}
		}
		resolution["status"] = "identity_conflict"
		resolution["conflict"] = err.Error()
		return "", resolution
	}
	return resolved, resolution
}

func resolveTerminalModelIdentityChecked(args cliArgs, declared, servedModel, servedModelSource, modelPath string) (string, map[string]any, error) {
	declared = strings.TrimSpace(declared)
	if declared == "<required-before-submit>" {
		declared = ""
	}
	canonicalDeclared := declared != "" && canonicalModelAlias(declared) && strings.Count(declared, "/") == 1
	seed := declared
	if seed == "" {
		seed = firstNonEmpty(modelNameFromGGUFFilename(modelPath), servedModel)
	}
	var resolution map[string]any
	if seed != "" {
		resolution = remoteModelResolution(args, servedModel, servedModelSource, seed, modelPath)
	}
	exactSourceRepo := ""
	if modelPath != "" {
		filename := filepath.Base(modelPath)
		var candidates []any
		if resolution != nil {
			candidates = anySlice(resolution["candidates"])
			if hint := stringValue(resolution["sourceRepo"]); hint != "" {
				resolution["sourceRepoHint"] = hint
			}
			delete(resolution, "sourceRepo")
			delete(resolution, "sourceRepoMatch")
		}
		sourceRepo, lookupErr := terminalExactSourceRepoFromFilename(args, candidates, filename)
		if lookupErr != nil {
			if resolution == nil {
				resolution = map[string]any{"hfId": seed, "servedModel": servedModel, "servedModelSource": servedModelSource}
			}
			resolution["sourceRepoVerificationError"] = lookupErr.Error()
		} else if sourceRepo != "" {
			exactSourceRepo = sourceRepo
			if resolution == nil {
				resolution = map[string]any{"hfId": seed, "servedModel": servedModel, "servedModelSource": servedModelSource}
			}
			resolution["loadedFilename"] = filename
			resolution["sourceRepo"] = sourceRepo
			resolution["sourceRepoMatch"] = "exact_filename"
			resolution["status"] = "source_repo_verified"
		}
	}
	if canonicalDeclared {
		if exactSourceRepo != "" && !strings.EqualFold(exactSourceRepo, declared) {
			if resolution == nil {
				resolution = map[string]any{}
			}
			resolution["declaredHfId"] = declared
			resolution["verifiedSourceRepo"] = exactSourceRepo
			return "", resolution, cliError{"model_identity_conflict", "The explicit canonical --model conflicts with the repository verified from the live loaded filename.", []string{"Correct --model to the verified repository, or load the intended model artifact."}, map[string]any{"declared": declared, "verifiedSourceRepo": exactSourceRepo, "filename": filepath.Base(modelPath)}}
		}
		return declared, resolution, nil
	}
	if exactSourceRepo != "" {
		printStatus(args, "eval_hf_id_resolved", map[string]any{"resolved": exactSourceRepo, "servedModel": servedModel, "filename": filepath.Base(modelPath), "source": "exact_filename"})
		return exactSourceRepo, resolution, nil
	}
	return "", resolution, nil
}

func terminalExactSourceRepoFromFilename(args cliArgs, candidates []any, filename string) (string, error) {
	repositories := append([]string{}, filenameDerivedSourceRepos(filename)...)
	for _, candidate := range candidates {
		if repo := candidateRepoID(candidate); repo != "" {
			repositories = append(repositories, repo)
		}
	}
	seen := map[string]bool{}
	matches := make([]string, 0, 1)
	var firstErr error
	for _, repo := range repositories {
		key := strings.ToLower(repo)
		if repo == "" || seen[key] {
			continue
		}
		seen[key] = true
		matched, err := terminalHFRepoContainsExactFilename(args, repo, filename)
		if err != nil {
			if firstErr == nil {
				firstErr = err
			}
			continue
		}
		if matched {
			matches = append(matches, repo)
		}
	}
	if len(matches) > 1 {
		return "", fmt.Errorf("loaded filename %q exists in multiple candidate repositories: %s", filename, strings.Join(matches, ", "))
	}
	if len(matches) == 1 {
		return matches[0], nil
	}
	return "", firstErr
}

func terminalHFRepoContainsExactFilename(args cliArgs, repo, filename string) (bool, error) {
	body, err := fetchEndpointJSONWithTimeout(strings.TrimRight(hfAPIURL(args), "/")+"/api/models/"+hfRepoPath(repo), "", terminalIdentityLookupTimeout)
	if err != nil {
		return false, err
	}
	obj := asObject(body)
	if obj == nil {
		return false, nil
	}
	for _, sibling := range modelFileItems(obj) {
		file := firstNonEmpty(stringValue(sibling["rfilename"]), stringValue(sibling["filename"]), stringValue(sibling["path"]))
		file = filepath.Base(strings.ReplaceAll(file, `\`, "/"))
		if file != "" && strings.EqualFold(file, filename) {
			return true, nil
		}
	}
	return false, nil
}

func terminalDeferredSubmitCommand(args cliArgs, out string, summary map[string]any) string {
	command := []string{"lmx", "eval", "terminal", "submit", shellQuote(out), "--api-url", shellQuote(apiURL(args))}
	dataset := stringValue(summary["dataset"])
	if dataset == "" || dataset == "local" {
		command = append(command, "--dataset", shellQuote("<dataset-slug>"))
	}
	if stringValue(summary["hfId"]) == "" {
		command = append(command, "--hf-id", shellQuote("<org/model>"))
	}
	if summary["hardware"] == nil {
		command = append(command, "--hardware", "hardware.json")
	}
	if dataset != "" && dataset != "local" && dataset != terminalBench21Dataset && int(numberField(summary, "shardIndex")) < 1 {
		command = append(command, "--shard-index", shellQuote("<n>"))
	}
	return strings.Join(command, " ")
}

func printTerminalFailureSummary(args cliArgs, bundles []terminalBundle, results []terminalTaskResult, cfg terminalConfig, out string) {
	rows := make([]any, 0)
	humanRows := make([][5]string, 0)
	counts := map[string]int{}
	for index, result := range results {
		if result.scored && result.pass && result.errCode == "" {
			continue
		}
		outcome := "verifier_failed"
		reason := "verifier scored the task as failed"
		if result.errCode != "" {
			outcome = result.errCode
			reason = firstNonEmpty(result.errText, "task reported an execution error")
		} else if !result.scored {
			outcome = "infrastructure_error"
			reason = firstNonEmpty(result.errText, "task did not reach verifier scoring")
		} else if result.agentOutcomeCode != "" {
			outcome = result.agentOutcomeCode
			reason = firstNonEmpty(result.agentOutcomeText, "agent stopped before verifier success")
		}
		hint := "use --out or --trace-dir for an artifact"
		if out != "" {
			hint = out + "#results[" + strconv.Itoa(index) + "]"
		} else if cfg.traceRoot != "" {
			hint = filepath.Join(cfg.traceRoot, sanitizeDockerName(bundles[index].Task.ID))
		}
		counts[outcome]++
		rows = append(rows, map[string]any{"taskId": bundles[index].Task.ID, "outcome": outcome, "reason": reason, "turns": result.turns, "artifactHint": hint})
		humanRows = append(humanRows, [5]string{bundles[index].Task.ID, outcome, reason, strconv.Itoa(result.turns), hint})
	}
	if len(rows) == 0 {
		return
	}
	printStatus(args, "terminal_failure_summary", map[string]any{"failedTasks": len(rows), "categories": counts, "failures": rows})
	if hasFlag(args, "quiet") || hasFlag(args, "json-status") {
		return
	}
	fmt.Fprintln(os.Stderr, "TASK                             RESULT                    TURNS  REASON / ARTIFACT")
	for _, row := range humanRows {
		combined := terminalSingleLine(row[2], 72) + " | " + row[4]
		fmt.Fprintf(os.Stderr, "%-32s %-25s %5s  %s\n", terminalSingleLine(row[0], 32), row[1], row[3], combined)
	}
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
	QuestionID       string         `json:"question_id"`
	Pass             bool           `json:"pass"`
	Scored           *bool          `json:"scored"`
	Error            string         `json:"error"`
	ErrorCode        string         `json:"errorCode"`
	AgentOutcomeCode string         `json:"agentOutcomeCode"`
	AgentOutcome     string         `json:"agentOutcome"`
	LatencyMs        int64          `json:"latencyMs"`
	WallTimeMs       int64          `json:"wallTimeMs"`
	TokenUsage       map[string]any `json:"tokenUsage"`
	Turns            int            `json:"turns"`
	Question         string         `json:"question"`
	Prompt           string         `json:"prompt"`
	Response         string         `json:"response"`
	VerifierOutput   string         `json:"verifierOutput"`
}

type terminalSavedTaskFile struct {
	Results []terminalSavedResult `json:"results"`
}

type terminalMonolithicArtifact struct {
	Summary map[string]any        `json:"summary"`
	Results []terminalSavedResult `json:"results"`
}

type terminalDeferredSource struct {
	root       string
	monolithic bool
	summary    map[string]any
	entries    []terminalCheckpointEntry
	results    map[string]terminalSavedResult
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

// terminalBench21CanonicalTaskIDsText is the exact sorted canonical
// Terminal-Bench 2.1 task set. Keeping the IDs compiled into the binary makes
// dataset inspection and deferred validation independent of local fixtures.
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

type terminalInspectionItem struct {
	questionID string
	bundleKey  string
	sha256     string
	byteSize   int64
}

var terminalInspectionSHA256 = regexp.MustCompile(`^[0-9a-fA-F]{64}$`)

// inspectTerminalDataset validates that every declared shard and manifest is
// ready to acquire. Bundle verification is opt-in and never executes a task.
func inspectTerminalDataset(dataset string, args cliArgs) error {
	dataset = strings.TrimSpace(dataset)
	if dataset == "" {
		return cliError{"missing_dataset", "eval terminal inspect requires a dataset slug.", []string{"Run: lmx eval terminal inspect <dataset> --api-url <localmaxxing-origin> [--verify-bundles] [--json]."}, nil}
	}
	inspectionOrigin, err := requireOpt(args, "api-url")
	if err != nil {
		return err
	}
	inspectionOrigin = strings.TrimRight(inspectionOrigin, "/")

	itemCount := 0
	shardCount := 0
	seenTasks := make(map[string]int)
	items := make([]terminalInspectionItem, 0)
	shards := make([]map[string]any, 0)

	for requestedShard := 1; requestedShard == 1 || requestedShard <= shardCount; requestedShard++ {
		metaURL := inspectionOrigin + "/api/evals/" + url.PathEscape(dataset) + "/shard?shard=" + strconv.Itoa(requestedShard)
		value, err := fetchJSON("GET", metaURL, apiKey(args), nil)
		if err != nil {
			return terminalInspectionError("Could not fetch a declared terminal dataset shard.", map[string]any{"dataset": dataset, "shardIndex": requestedShard, "url": metaURL, "error": err.Error()})
		}
		response := asObject(value)
		datasetMeta := asObject(response["dataset"])
		shardMeta := asObject(response["shard"])
		if datasetMeta == nil || shardMeta == nil {
			return terminalInspectionError("Terminal shard response is missing dataset or shard metadata.", map[string]any{"dataset": dataset, "shardIndex": requestedShard})
		}
		if responseSlug := strings.TrimSpace(stringValue(datasetMeta["slug"])); responseSlug != dataset {
			return terminalInspectionError("Terminal shard response names a different dataset.", map[string]any{"dataset": dataset, "actual": responseSlug, "shardIndex": requestedShard})
		}

		responseItemCount, ok := terminalInspectionPositiveInt(datasetMeta["itemCount"])
		if !ok {
			return terminalInspectionError("Terminal dataset itemCount must be a positive integer.", map[string]any{"dataset": dataset, "itemCount": datasetMeta["itemCount"]})
		}
		responseShardCount, ok := terminalInspectionPositiveInt(datasetMeta["shardCount"])
		if !ok {
			return terminalInspectionError("Terminal dataset shardCount must be a positive integer.", map[string]any{"dataset": dataset, "shardCount": datasetMeta["shardCount"]})
		}
		if requestedShard == 1 {
			itemCount = responseItemCount
			shardCount = responseShardCount
			if dataset == terminalBench21Dataset && (itemCount != len(terminalBench21CanonicalTaskIDs) || shardCount != terminalBench21ShardCount) {
				return terminalInspectionError("Terminal-Bench 2.1 dataset metadata is not canonical.", map[string]any{"expectedItemCount": len(terminalBench21CanonicalTaskIDs), "actualItemCount": itemCount, "expectedShardCount": terminalBench21ShardCount, "actualShardCount": shardCount})
			}
		} else if responseItemCount != itemCount || responseShardCount != shardCount {
			return terminalInspectionError("Terminal dataset metadata changed between shard responses.", map[string]any{"dataset": dataset, "shardIndex": requestedShard, "expectedItemCount": itemCount, "actualItemCount": responseItemCount, "expectedShardCount": shardCount, "actualShardCount": responseShardCount})
		}

		responseShardIndex, ok := terminalInspectionPositiveInt(shardMeta["shardIndex"])
		if !ok || responseShardIndex != requestedShard {
			return terminalInspectionError("Terminal shard response index does not match the requested shard.", map[string]any{"dataset": dataset, "requestedShardIndex": requestedShard, "actualShardIndex": shardMeta["shardIndex"]})
		}
		responseShardItems, ok := terminalInspectionPositiveInt(shardMeta["itemCount"])
		if !ok {
			return terminalInspectionError("Terminal shard itemCount must be a positive integer.", map[string]any{"dataset": dataset, "shardIndex": requestedShard, "itemCount": shardMeta["itemCount"]})
		}
		selectedItems, ok := terminalInspectionPositiveInt(shardMeta["selectedQuestionCount"])
		if !ok || selectedItems != responseShardItems {
			return terminalInspectionError("Terminal shard selectedQuestionCount does not match itemCount.", map[string]any{"dataset": dataset, "shardIndex": requestedShard, "itemCount": responseShardItems, "selectedQuestionCount": shardMeta["selectedQuestionCount"]})
		}

		downloadURL := strings.TrimSpace(stringValue(response["downloadUrl"]))
		if downloadURL == "" {
			return terminalInspectionError("Terminal shard response did not include downloadUrl.", map[string]any{"dataset": dataset, "shardIndex": requestedShard})
		}
		manifestRows, err := fetchDatasetItems(downloadURL, "jsonl")
		if err != nil {
			details := map[string]any{"dataset": dataset, "shardIndex": requestedShard, "error": terminalSafeDownloadError(err)}
			if sanitized := terminalSanitizedDownloadURL(downloadURL); sanitized != "" {
				details["downloadUrl"] = sanitized
			}
			return terminalInspectionError("Could not download a terminal shard manifest.", details)
		}
		if len(manifestRows) != responseShardItems {
			return terminalInspectionError("Terminal shard manifest row count does not match shard itemCount.", map[string]any{"dataset": dataset, "shardIndex": requestedShard, "itemCount": responseShardItems, "manifestItems": len(manifestRows)})
		}

		for rowIndex, row := range manifestRows {
			questionID := strings.TrimSpace(stringValue(row["question_id"]))
			bundleKey := strings.TrimSpace(stringValue(row["bundle_key"]))
			hash := strings.TrimSpace(stringValue(row["sha256"]))
			byteSize, validByteSize := terminalInspectionPositiveInt64(row["byteSize"])
			if questionID == "" {
				return terminalInspectionError("Terminal manifest row is missing question_id.", map[string]any{"dataset": dataset, "shardIndex": requestedShard, "rowIndex": rowIndex})
			}
			if previousShard, exists := seenTasks[questionID]; exists {
				return terminalInspectionError("Terminal task appears more than once across shard manifests.", map[string]any{"dataset": dataset, "taskId": questionID, "firstShardIndex": previousShard, "duplicateShardIndex": requestedShard})
			}
			if bundleKey == "" {
				return terminalInspectionError("Terminal manifest row is missing bundle_key.", map[string]any{"dataset": dataset, "shardIndex": requestedShard, "taskId": questionID})
			}
			if !terminalInspectionSHA256.MatchString(hash) {
				return terminalInspectionError("Terminal manifest row sha256 must contain exactly 64 hexadecimal characters.", map[string]any{"dataset": dataset, "shardIndex": requestedShard, "taskId": questionID, "sha256": hash})
			}
			if !validByteSize {
				return terminalInspectionError("Terminal manifest row byteSize must be a positive integer.", map[string]any{"dataset": dataset, "shardIndex": requestedShard, "taskId": questionID, "byteSize": row["byteSize"]})
			}
			seenTasks[questionID] = requestedShard
			items = append(items, terminalInspectionItem{questionID: questionID, bundleKey: bundleKey, sha256: hash, byteSize: byteSize})
		}
		shards = append(shards, map[string]any{"shardIndex": requestedShard, "itemCount": responseShardItems})
	}

	if len(items) != itemCount || len(seenTasks) != itemCount {
		return terminalInspectionError("Terminal manifests do not contain the dataset's declared number of unique tasks.", map[string]any{"dataset": dataset, "itemCount": itemCount, "manifestItems": len(items), "uniqueTaskIds": len(seenTasks)})
	}
	if dataset == terminalBench21Dataset {
		if err := validateTerminalBench21Inspection(seenTasks, shards); err != nil {
			return err
		}
	}

	summary := map[string]any{
		"ready":         true,
		"dataset":       dataset,
		"itemCount":     itemCount,
		"shardCount":    shardCount,
		"manifestItems": len(items),
		"uniqueTaskIds": len(seenTasks),
		"shards":        shards,
	}
	if hasFlag(args, "verify-bundles") {
		verified, err := verifyTerminalInspectionBundles(args, items)
		if err != nil {
			return err
		}
		summary["verifiedBundles"] = verified
	}
	if hasFlag(args, "json") || hasFlag(args, "print") || opt(args, "out") != "" {
		return writeOrPrintJSON("terminal_dataset_inspection", args, summary)
	}
	printInfo("terminal_dataset_ready", summary)
	return nil
}

func terminalInspectionError(message string, details any) error {
	return cliError{"terminal_inspect_failed", message, []string{"Check the dataset ingestion and LocalMaxxing API response, then retry inspection."}, details}
}

func terminalInspectionPositiveInt(value any) (int, bool) {
	n, ok := terminalInspectionPositiveInt64(value)
	if !ok || int64(int(n)) != n {
		return 0, false
	}
	return int(n), true
}

func terminalInspectionPositiveInt64(value any) (int64, bool) {
	var n int64
	switch typed := value.(type) {
	case float64:
		n = int64(typed)
		if float64(n) != typed {
			return 0, false
		}
	case float32:
		n = int64(typed)
		if float32(n) != typed {
			return 0, false
		}
	case int:
		n = int64(typed)
	case int64:
		n = typed
	case json.Number:
		parsed, err := typed.Int64()
		if err != nil {
			return 0, false
		}
		n = parsed
	default:
		return 0, false
	}
	return n, n > 0
}

func validateTerminalBench21Inspection(seenTasks map[string]int, shards []map[string]any) error {
	missing := make([]string, 0)
	extra := make([]string, 0)
	expected := make(map[string]bool, len(terminalBench21CanonicalTaskIDs))
	for _, id := range terminalBench21CanonicalTaskIDs {
		expected[id] = true
		if _, ok := seenTasks[id]; !ok {
			missing = append(missing, id)
		}
	}
	for id := range seenTasks {
		if !expected[id] {
			extra = append(extra, id)
		}
	}
	sort.Strings(extra)
	if len(missing) > 0 || len(extra) > 0 {
		return terminalInspectionError("Terminal-Bench 2.1 manifests do not match the canonical task set.", map[string]any{"missingTaskIds": missing, "extraTaskIds": extra})
	}
	for index, id := range terminalBench21CanonicalTaskIDs {
		expectedShard := (((index+1)*terminalBench21ShardCount)-1)/len(terminalBench21CanonicalTaskIDs) + 1
		if actualShard := seenTasks[id]; actualShard != expectedShard {
			return terminalInspectionError("Terminal-Bench 2.1 task is assigned to a noncanonical shard.", map[string]any{"taskId": id, "expectedShardIndex": expectedShard, "actualShardIndex": actualShard})
		}
	}
	if len(shards) != terminalBench21ShardCount {
		return terminalInspectionError("Terminal-Bench 2.1 does not contain exactly ten shards.", map[string]any{"expectedShardCount": terminalBench21ShardCount, "actualShardCount": len(shards)})
	}
	for i, shard := range shards {
		expectedSize := ((i + 1) * len(terminalBench21CanonicalTaskIDs) / terminalBench21ShardCount) - (i * len(terminalBench21CanonicalTaskIDs) / terminalBench21ShardCount)
		if shard["itemCount"] != expectedSize {
			return terminalInspectionError("Terminal-Bench 2.1 shard size is not canonical.", map[string]any{"shardIndex": i + 1, "expectedItemCount": expectedSize, "actualItemCount": shard["itemCount"]})
		}
	}
	return nil
}

func verifyTerminalInspectionBundles(args cliArgs, items []terminalInspectionItem) (int, error) {
	tmp, err := os.MkdirTemp("", "lmx-terminal-inspect-*")
	if err != nil {
		return 0, err
	}
	defer os.RemoveAll(tmp)

	for index, item := range items {
		extractionRoot := filepath.Join(tmp, strconv.Itoa(index+1))
		bundleDir, err := downloadTerminalBundle(args, extractionRoot, item.questionID, item.bundleKey, item.sha256, item.byteSize)
		if err != nil {
			return index, err
		}
		if err := rejectTerminalInspectionSolution(extractionRoot, item.questionID); err != nil {
			return index, err
		}
		bundle, err := loadSingleTerminalBundle(bundleDir)
		if err != nil {
			return index, err
		}
		if bundle.Task.ID != item.questionID {
			return index, terminalInspectionError("Downloaded terminal bundle task id does not match its manifest row.", map[string]any{"questionId": item.questionID, "bundleTaskId": bundle.Task.ID, "bundleKey": item.bundleKey})
		}
	}
	return len(items), nil
}

func rejectTerminalInspectionSolution(root, questionID string) error {
	return filepath.WalkDir(root, func(current string, entry os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if entry.IsDir() && entry.Name() == "solution" {
			return terminalInspectionError("Public terminal bundle contains a forbidden solution directory.", map[string]any{"taskId": questionID, "path": current})
		}
		return nil
	})
}

type terminalSubmissionRecord struct {
	questionID string
	pass       bool
	latencyMs  int64
	usage      terminalTokenUsage
	result     map[string]any
	artifact   map[string]any
}

// submitTerminalEval packages an already completed legacy checkpoint directory
// or monolithic run artifact. It deliberately does not share runTerminalEval's
// acquisition path: deferred submit must never contact a model endpoint,
// acquire tasks, start Docker, or rerun a verifier.
func submitTerminalEval(args cliArgs) error {
	runPath := positional(args, 3)
	if runPath == "" {
		return cliError{"missing_option", "eval terminal submit requires a completed run artifact.", []string{"Run: lmx eval terminal submit <run-dir-or-results.json> --dataset <slug> --hf-id <org/model> --hardware hardware.json --api-url <localmaxxing-origin> --dry-run."}, nil}
	}
	source, err := loadTerminalDeferredSource(runPath)
	if err != nil {
		return err
	}

	savedDataset := stringValue(source.summary["dataset"])
	if savedDataset == "local" {
		savedDataset = ""
	}
	dataset, err := resolveTerminalSavedString("dataset", savedDataset, opt(args, "dataset"), true)
	if err != nil {
		return err
	}
	savedHFID := stringValue(source.summary["hfId"])
	hfID, err := resolveTerminalSavedString("hf-id", savedHFID, opt(args, "hf-id"), true)
	if err != nil {
		return err
	}
	savedRevision := stringValue(source.summary["modelRevision"])
	modelRevision, err := resolveTerminalSavedString("model-revision", savedRevision, opt(args, "model-revision"), false)
	if err != nil {
		return err
	}
	modelRevision = firstNonEmpty(modelRevision, "main")

	shardIndex, explicitShard, err := terminalSubmitShardIndex(args, dataset)
	if err != nil {
		return err
	}
	savedShardIndex := 0
	if rawSavedShard, exists := source.summary["shardIndex"]; exists && rawSavedShard != nil {
		var valid bool
		savedShardIndex, valid = terminalInspectionPositiveInt(rawSavedShard)
		if !valid {
			return cliError{"checkpoint_metadata_invalid", "Saved shardIndex must be a positive integer.", []string{"Use an unmodified completed terminal run artifact."}, map[string]any{"savedShardIndex": rawSavedShard}}
		}
	}
	if dataset == terminalBench21Dataset && savedShardIndex > terminalBench21ShardCount {
		return cliError{"checkpoint_metadata_invalid", fmt.Sprintf("Saved shardIndex for Terminal-Bench 2.1 must be between 1 and %d.", terminalBench21ShardCount), nil, map[string]any{"savedShardIndex": savedShardIndex}}
	}
	if savedShardIndex > 0 {
		if explicitShard && shardIndex != savedShardIndex {
			return cliError{"checkpoint_metadata_conflict", "--shard-index conflicts with the saved terminal artifact.", []string{"Submit the artifact using its recorded shard index."}, map[string]any{"saved": savedShardIndex, "explicit": shardIndex}}
		}
		shardIndex = savedShardIndex
		explicitShard = true
	}
	if dataset != terminalBench21Dataset && !explicitShard {
		return cliError{"missing_shard_index", "Deferred submission for this dataset requires --shard-index <n>.", []string{"Pass the registered shard index for this already-isolated checkpoint.", "The CLI only performs automatic batching for terminal-bench-2-1 because its exact canonical task set is built in."}, map[string]any{"dataset": dataset}}
	}

	var hardware any
	if hardwarePath := opt(args, "hardware"); hardwarePath != "" {
		hardware, err = readJSON(hardwarePath)
		if err != nil {
			return err
		}
		hardware = normalizeHardwarePayload(hardware)
	}
	if savedHardware := source.summary["hardware"]; savedHardware != nil {
		savedHardware = normalizeHardwarePayload(savedHardware)
		if hardware != nil && !terminalJSONEqual(hardware, savedHardware) {
			return cliError{"checkpoint_metadata_conflict", "--hardware conflicts with the hardware saved in the terminal artifact.", []string{"Remove --hardware to use the saved value, or select the matching run artifact."}, nil}
		}
		hardware = savedHardware
	}
	if hardware == nil {
		return cliError{"missing_hardware", "eval terminal submit requires --hardware hardware.json or saved hardware metadata.", []string{"Run lmx hardware --out hardware.json and pass that saved hardware object."}, nil}
	}

	entries := source.entries
	seenTasks := make(map[string]bool, len(entries))
	seenIndexes := make(map[int]bool, len(entries))
	for _, entry := range entries {
		if err := validateTerminalCheckpointEntry(entry, len(entries), seenTasks, seenIndexes); err != nil {
			return err
		}
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].Task < entries[j].Task })
	if dataset == terminalBench21Dataset {
		if explicitShard {
			if terminalCheckpointHasCanonicalTaskSet(entries) {
				return cliError{"full_checkpoint_with_shard_index", "A full canonical Terminal-Bench 2.1 checkpoint cannot be labeled as one shard.", []string{"Remove --shard-index to partition all 89 tasks into the 10 canonical shards."}, map[string]any{"tasks": len(entries), "shardIndex": shardIndex}}
			}
			if err := validateTerminalBench21ShardTaskSet(entries, shardIndex); err != nil {
				return err
			}
		} else if err := validateTerminalBench21FullTaskSet(entries); err != nil {
			return err
		}
	}

	quantization := stringValue(source.summary["quantization"])
	quantFormat := stringValue(source.summary["quantFormat"])
	quantization, err = resolveTerminalSavedString("quantization", quantization, opt(args, "quantization"), false)
	if err != nil {
		return err
	}
	quantFormat, err = resolveTerminalSavedString("quant-format", quantFormat, opt(args, "quant-format"), false)
	if err != nil {
		return err
	}
	agentName, err := resolveTerminalSavedString("agent-name", stringValue(source.summary["agent"]), opt(args, "agent-name"), false)
	if err != nil {
		return err
	}
	runnerVersion, err := resolveTerminalSavedString("runner-version", stringValue(source.summary["runnerVersion"]), opt(args, "runner-version"), false)
	if err != nil {
		return err
	}
	resolvedArgs := terminalArgsWithOptions(args, map[string]string{"model-revision": modelRevision, "quantization": quantization, "quant-format": quantFormat, "agent-name": agentName, "runner-version": runnerVersion})

	records := make([]terminalSubmissionRecord, 0, len(entries))
	seenResults := make(map[string]bool, len(entries))
	passed := 0
	var totalLatencyMs int64
	artifactBytes, maxArtifactBytes, traceCount, fallbackCount := 0, 0, 0, 0
	totalUsage := terminalTokenUsage{}
	previewTotals := terminalTracePreviewStats{}
	for _, entry := range entries {
		record, recordPath, err := terminalDeferredResult(source, entry)
		if err != nil {
			return err
		}
		if seenResults[record.QuestionID] {
			return cliError{"duplicate_task_result", fmt.Sprintf("Task result %q appears more than once.", record.QuestionID), []string{"Ensure every task has exactly one unique result record."}, map[string]any{"taskId": record.QuestionID, "file": recordPath}}
		}
		seenResults[record.QuestionID] = true
		if record.Pass != entry.Pass {
			return cliError{"checkpoint_result_mismatch", fmt.Sprintf("Task %q has conflicting pass values in the summary and result record.", entry.Task), []string{"Recover matching metadata and results from the same completed run."}, map[string]any{"taskId": entry.Task, "summaryPass": entry.Pass, "resultPass": record.Pass}}
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
		response := ""
		previewStats := terminalTracePreviewStats{}
		usedTrace := false
		if source.monolithic {
			response = truncateString(record.Response+"\n\n# Verifier\n\n"+record.VerifierOutput, terminalArtifactResponseBytes)
			fallbackCount++
		} else {
			response, previewStats, usedTrace, err = terminalSavedArtifactResponse(source.root, recordPath, record)
			if err != nil {
				return cliError{"trace_read_failed", fmt.Sprintf("Could not package the OMP trace for task %q.", entry.Task), []string{"Check that the selected omp.jsonl is readable, or remove the broken trace to use the bounded saved response fallback."}, map[string]any{"taskId": entry.Task, "error": err.Error()}}
			}
			if usedTrace {
				traceCount++
			} else {
				fallbackCount++
			}
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
		return cliError{"checkpoint_incomplete", "The terminal checkpoint did not produce one unique result for every summary record.", []string{"Restore the missing task results and rerun deferred submit."}, map[string]any{"summaryRecords": len(entries), "uniqueResults": len(seenResults)}}
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
		fullProvenance = map[string]any{
			"fullCheckpoint":                    true,
			"fullCheckpointTasksRun":            len(records),
			"fullCheckpointAccuracy":            fullAccuracy,
			"fullCheckpointAvgLatencyMs":        fullAvgLatencyMs,
			"fullCheckpointTokenUsage":          totalUsage.toMap(),
			"fullCheckpointTaskSetSha256":       hex.EncodeToString(fullTaskSetHash[:]),
			"fullCheckpointCanonicalShardCount": terminalBench21ShardCount,
		}
	}
	if len(source.summary) > 0 {
		fullProvenance["sourceArtifactVersion"] = source.summary["artifactVersion"]
		fullProvenance["sourceAgent"] = source.summary["agent"]
		fullProvenance["sourceServedModel"] = source.summary["servedModel"]
		fullProvenance["sourceDeclaredModel"] = source.summary["declaredModel"]
		fullProvenance["sourceModelResolution"] = terminalSanitizeNestedRunConfig(source.summary["modelResolution"], []string{"hfId", "servedModel", "servedModelSource", "status", "declaredBaseModel", "loadedFilename", "sourceRepo", "sourceRepoMatch", "declaredHfId", "verifiedSourceRepo", "searchQuery", "searchQuerySource"})
		fullProvenance["sourceQuantizationResolution"] = terminalSanitizeNestedRunConfig(source.summary["quantizationResolution"], []string{"cli", "v1Models", "filename", "trusted", "trustedSource", "status"})
		fullProvenance["sourceRunConfig"] = terminalSanitizeSourceRunConfig(source.summary["runConfig"])
	}

	recordShards := [][]terminalSubmissionRecord{records}
	shardIndexes := []int{shardIndex}
	if !explicitShard {
		recordShards = partitionTerminalSubmissionRecords(records, terminalBench21ShardCount)
		shardIndexes = make([]int, len(recordShards))
		for i := range shardIndexes {
			shardIndexes[i] = i + 1
		}
	}
	payloads := make([]any, 0, len(recordShards))
	shardSizes := make([]int, 0, len(recordShards))
	for i, shardRecords := range recordShards {
		payload := terminalSubmissionPayload(resolvedArgs, hfID, hardware, shardIndexes[i], shardRecords, fullProvenance)
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
		"source":                 map[bool]string{true: "monolithic", false: "checkpoint_directory"}[source.monolithic],
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
		printInfo("terminal_submit_dry_run", fields)
		fmt.Println("Dry run only — payload batch validated locally; no network request was made.")
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
		value, err := fetchJSON("POST", apiURL(args)+"/api/evals/"+url.PathEscape(dataset)+"/submit", key, rawPayload)
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
	printInfo("terminal_submit_complete", fields)
	return nil
}

func loadTerminalDeferredSource(runPath string) (terminalDeferredSource, error) {
	resolved, err := filepath.Abs(runPath)
	if err != nil {
		return terminalDeferredSource{}, err
	}
	info, err := os.Stat(resolved)
	if err != nil {
		return terminalDeferredSource{}, cliError{"checkpoint_missing", "Could not open the completed terminal run artifact.", []string{"Pass a legacy checkpoint directory or the monolithic JSON file written by eval terminal run --out."}, map[string]any{"path": runPath, "error": err.Error()}}
	}
	if info.IsDir() {
		entries, err := loadTerminalCheckpointSummary(filepath.Join(resolved, "summary.json"))
		if err != nil {
			return terminalDeferredSource{}, err
		}
		summary, err := aggregateTerminalCheckpointSummary(entries)
		if err != nil {
			return terminalDeferredSource{}, err
		}
		return terminalDeferredSource{root: resolved, summary: summary, entries: entries}, nil
	}
	if !info.Mode().IsRegular() {
		return terminalDeferredSource{}, cliError{"checkpoint_invalid", "The monolithic terminal run artifact must be a regular file.", nil, map[string]any{"path": runPath}}
	}
	if info.Size() > terminalMonolithicArtifactMaxBytes {
		return terminalDeferredSource{}, cliError{"checkpoint_too_large", "The monolithic terminal run artifact exceeds the safe input limit.", []string{"Use the bounded artifact written by eval terminal run --out, or submit a legacy checkpoint directory."}, map[string]any{"path": runPath, "bytes": info.Size(), "maxBytes": terminalMonolithicArtifactMaxBytes}}
	}
	file, err := os.Open(resolved)
	if err != nil {
		return terminalDeferredSource{}, cliError{"checkpoint_invalid", "Could not read the monolithic terminal run artifact.", nil, map[string]any{"path": runPath, "error": err.Error()}}
	}
	defer file.Close()
	decoder := json.NewDecoder(io.LimitReader(file, terminalMonolithicArtifactMaxBytes+1))
	artifact, decodeErr := decodeTerminalMonolithicArtifact(decoder)
	if decodeErr != nil {
		if errors.Is(decodeErr, errTerminalMonolithicResultLimit) {
			return terminalDeferredSource{}, cliError{"checkpoint_result_count_invalid", "Monolithic terminal run JSON contains too many results.", []string{"Split unrelated runs; one artifact must describe one bounded completed evaluation."}, map[string]any{"path": runPath, "maxResults": terminalMonolithicResultLimit}}
		}
		return terminalDeferredSource{}, cliError{"checkpoint_invalid", "Could not decode the monolithic terminal run artifact as {summary,results} JSON.", []string{"Pass the JSON file written by eval terminal run --out."}, map[string]any{"path": runPath, "error": decodeErr.Error()}}
	}
	var trailing any
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		return terminalDeferredSource{}, cliError{"checkpoint_invalid", "Monolithic terminal run JSON contains trailing data.", []string{"Pass exactly one JSON object written by eval terminal run --out."}, map[string]any{"path": runPath}}
	}
	if artifact.Summary == nil || len(artifact.Results) == 0 {
		return terminalDeferredSource{}, cliError{"checkpoint_invalid", "Monolithic terminal run JSON must contain a summary object and non-empty results array.", []string{"Pass the completed JSON file written by eval terminal run --out."}, map[string]any{"path": runPath}}
	}
	entries := make([]terminalCheckpointEntry, len(artifact.Results))
	results := make(map[string]terminalSavedResult, len(artifact.Results))
	for index, record := range artifact.Results {
		if _, duplicate := results[record.QuestionID]; duplicate {
			return terminalDeferredSource{}, cliError{"duplicate_task_result", fmt.Sprintf("Monolithic results contain duplicate task %q.", record.QuestionID), []string{"Keep exactly one result per task."}, map[string]any{"taskId": record.QuestionID}}
		}
		entries[index] = terminalCheckpointEntry{Index: index + 1, Total: len(artifact.Results), Task: record.QuestionID, Pass: record.Pass, Scored: record.Scored, Summary: artifact.Summary}
		results[record.QuestionID] = record
	}
	return terminalDeferredSource{root: filepath.Dir(resolved), monolithic: true, summary: artifact.Summary, entries: entries, results: results}, nil
}

func decodeTerminalMonolithicArtifact(decoder *json.Decoder) (terminalMonolithicArtifact, error) {
	artifact := terminalMonolithicArtifact{}
	opening, err := decoder.Token()
	if err != nil {
		return artifact, err
	}
	if delimiter, ok := opening.(json.Delim); !ok || delimiter != '{' {
		return artifact, errors.New("monolithic terminal artifact must be a JSON object")
	}
	seenSummary, seenResults := false, false
	for decoder.More() {
		rawKey, err := decoder.Token()
		if err != nil {
			return artifact, err
		}
		key, ok := rawKey.(string)
		if !ok {
			return artifact, errors.New("monolithic terminal artifact contains a non-string field name")
		}
		switch key {
		case "summary":
			if seenSummary {
				return artifact, errors.New("monolithic terminal artifact contains duplicate summary fields")
			}
			seenSummary = true
			if err := decoder.Decode(&artifact.Summary); err != nil {
				return artifact, err
			}
		case "results":
			if seenResults {
				return artifact, errors.New("monolithic terminal artifact contains duplicate results fields")
			}
			seenResults = true
			openingResults, err := decoder.Token()
			if err != nil {
				return artifact, err
			}
			if delimiter, ok := openingResults.(json.Delim); !ok || delimiter != '[' {
				return artifact, errors.New("monolithic terminal results must be a JSON array")
			}
			for decoder.More() {
				if len(artifact.Results) >= terminalMonolithicResultLimit {
					return artifact, errTerminalMonolithicResultLimit
				}
				var record terminalSavedResult
				if err := decoder.Decode(&record); err != nil {
					return artifact, err
				}
				artifact.Results = append(artifact.Results, record)
			}
			if _, err := decoder.Token(); err != nil {
				return artifact, err
			}
		default:
			return artifact, fmt.Errorf("monolithic terminal artifact contains unexpected top-level field %q", key)
		}
	}
	if _, err := decoder.Token(); err != nil {
		return artifact, err
	}
	if !seenSummary || !seenResults {
		return artifact, errors.New("monolithic terminal artifact must contain summary and results")
	}
	return artifact, nil
}

func aggregateTerminalCheckpointSummary(entries []terminalCheckpointEntry) (map[string]any, error) {
	authoritative := []string{"artifactVersion", "dataset", "hfId", "modelRevision", "shardIndex", "hardware", "quantization", "quantFormat", "agent", "runnerVersion", "servedModel", "declaredModel", "modelResolution", "quantizationResolution", "runConfig"}
	summary := map[string]any{}
	sources := map[string]string{}
	for _, entry := range entries {
		for _, field := range authoritative {
			value, exists := entry.Summary[field]
			if !exists || !terminalMetadataValuePresent(value) {
				continue
			}
			if saved, found := summary[field]; found && !terminalJSONEqual(saved, value) {
				return nil, cliError{"checkpoint_metadata_mismatch", fmt.Sprintf("Checkpoint task summaries contain conflicting %s values.", field), []string{"Recover summary.json from one completed run."}, map[string]any{"field": field, "firstTaskId": sources[field], "taskId": entry.Task}}
			}
			if _, found := summary[field]; !found {
				summary[field] = value
				sources[field] = entry.Task
			}
		}
	}
	return summary, nil
}

func terminalMetadataValuePresent(value any) bool {
	if value == nil {
		return false
	}
	if text, ok := value.(string); ok {
		return strings.TrimSpace(text) != ""
	}
	return true
}

func terminalDeferredResult(source terminalDeferredSource, entry terminalCheckpointEntry) (terminalSavedResult, string, error) {
	if source.monolithic {
		record, ok := source.results[entry.Task]
		if !ok {
			return terminalSavedResult{}, "", cliError{"task_result_missing", fmt.Sprintf("Missing monolithic result for %q.", entry.Task), nil, map[string]any{"taskId": entry.Task}}
		}
		if err := validateTerminalSavedResult(record, entry.Task, "monolithic results"); err != nil {
			return terminalSavedResult{}, "", err
		}
		return record, "monolithic results", nil
	}
	recordPath, err := terminalCheckpointResultPath(source.root, entry)
	if err != nil {
		return terminalSavedResult{}, "", err
	}
	record, err := loadTerminalSavedResult(recordPath, entry.Task)
	return record, recordPath, err
}

func resolveTerminalSavedString(flag, saved, explicit string, required bool) (string, error) {
	if saved != "" && explicit != "" && saved != explicit {
		return "", cliError{"checkpoint_metadata_conflict", "--" + flag + " conflicts with authoritative metadata saved in the terminal artifact.", []string{"Remove the explicit flag to use the saved value, or select the matching run artifact."}, map[string]any{"field": flag, "saved": saved, "explicit": explicit}}
	}
	value := firstNonEmpty(explicit, saved)
	if value != "" || !required {
		return value, nil
	}
	if flag == "dataset" {
		return "", cliError{"missing_dataset", "eval terminal submit requires --dataset <slug> when the artifact does not record one.", []string{"Pass the terminal dataset slug used for the completed run."}, nil}
	}
	return "", cliError{"missing_model", "eval terminal submit requires --hf-id <HuggingFace model id> when the artifact does not record one.", []string{"Pass the canonical org/model identifier for the completed run."}, nil}
}

func terminalJSONEqual(left, right any) bool {
	leftJSON, leftErr := json.Marshal(left)
	rightJSON, rightErr := json.Marshal(right)
	return leftErr == nil && rightErr == nil && bytes.Equal(leftJSON, rightJSON)
}

func terminalSanitizeSourceRunConfig(value any) map[string]any {
	source := asObject(value)
	safe := map[string]any{}
	if source == nil {
		return safe
	}
	for _, key := range []string{"protocol", "agent", "agentBackend", "maxTurns", "maxTokens", "temperature", "topP", "commandTimeoutSec", "agentTimeoutSec", "concurrency", "shellMode", "agentExecution", "oracle", "servedModelSource", "endpointTimeoutMillis", "accuracy", "tasksRun", "errors", "avgLatencyMs", "deferredSubmit"} {
		if scalar, ok := terminalSafeRunConfigScalar(source[key]); ok {
			safe[key] = scalar
		}
	}
	if endpoint := terminalSanitizedEndpointOrigin(stringValue(source["modelEndpoint"])); endpoint != "" {
		safe["modelEndpoint"] = endpoint
	}
	if routing := terminalSanitizeNestedRunConfig(source["toolRouting"], []string{"shell", "workdir", "hostFilesystemVisible"}); len(routing) > 0 {
		safe["toolRouting"] = routing
	}
	if usage := terminalSanitizeNestedRunConfig(source["tokenUsage"], []string{"inputTokens", "outputTokens", "cacheReadTokens", "cacheWriteTokens", "totalTokens", "modelCalls"}); len(usage) > 0 {
		safe["tokenUsage"] = usage
	}
	return safe
}

func terminalSanitizeNestedRunConfig(value any, allowed []string) map[string]any {
	source := asObject(value)
	safe := map[string]any{}
	for _, key := range allowed {
		if scalar, ok := terminalSafeRunConfigScalar(source[key]); ok {
			safe[key] = scalar
		}
	}
	return safe
}

func terminalSafeRunConfigScalar(value any) (any, bool) {
	switch value.(type) {
	case string, bool, float64, float32, int, int64, int32, uint, uint64, uint32, json.Number:
		return value, true
	default:
		return nil, false
	}
}

func terminalArgsWithOptions(args cliArgs, values map[string]string) cliArgs {
	copied := cliArgs{positional: append([]string(nil), args.positional...), opts: map[string]string{}, flags: map[string]bool{}}
	for key, value := range args.opts {
		copied.opts[key] = value
	}
	for key, value := range args.flags {
		copied.flags[key] = value
	}
	for key, value := range values {
		if value != "" && copied.opts[key] == "" {
			copied.opts[key] = value
		}
	}
	return copied
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
	if dataset == terminalBench21Dataset && index > terminalBench21ShardCount {
		return 0, true, cliError{"invalid_shard_index", fmt.Sprintf("--shard-index for Terminal-Bench 2.1 must be between 1 and %d.", terminalBench21ShardCount), nil, map[string]any{"value": index}}
	}
	return index, true, nil
}

func terminalCheckpointHasCanonicalTaskSet(entries []terminalCheckpointEntry) bool {
	canonical := terminalBench21CanonicalTaskIDs
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

func validateTerminalBench21FullTaskSet(entries []terminalCheckpointEntry) error {
	return validateTerminalTaskSet(entries, terminalBench21CanonicalTaskIDs, "canonical Terminal-Bench 2.1 checkpoint", 0)
}

func validateTerminalBench21ShardTaskSet(entries []terminalCheckpointEntry, shardIndex int) error {
	canonical := terminalBench21CanonicalTaskIDs
	start := ((shardIndex - 1) * len(canonical)) / terminalBench21ShardCount
	end := (shardIndex * len(canonical)) / terminalBench21ShardCount
	return validateTerminalTaskSet(entries, canonical[start:end], fmt.Sprintf("canonical Terminal-Bench 2.1 shard %d", shardIndex), shardIndex)
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
		"accuracy":       float64(passed) / float64(len(records)),
		"tasksRun":       len(records),
		"errors":         0,
		"avgLatencyMs":   latencyMs / int64(len(records)),
		"protocol":       "deferred-saved-terminal-run",
		"agent":          firstNonEmpty(opt(args, "agent-name"), "external-agent"),
		"deferredSubmit": true,
		"tokenUsage":     usage.toMap(),
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
	if err := validateTerminalSavedResult(record, taskID, path); err != nil {
		return terminalSavedResult{}, err
	}
	return record, nil
}

func validateTerminalSavedResult(record terminalSavedResult, taskID, source string) error {
	if record.QuestionID == "" {
		return cliError{"task_result_id_missing", fmt.Sprintf("Task result %q has no question_id.", source), []string{"Every saved terminal result must identify its task."}, nil}
	}
	if record.QuestionID != taskID {
		return cliError{"task_result_id_mismatch", fmt.Sprintf("Task result for %q contains question_id %q.", taskID, record.QuestionID), []string{"Restore the matching task result from the completed run."}, map[string]any{"source": source}}
	}
	if record.Scored == nil {
		return cliError{"task_result_score_missing", fmt.Sprintf("Task result %q is missing scored.", taskID), []string{"Every deferred task result must explicitly record scored: true."}, nil}
	}
	if !*record.Scored {
		return cliError{"task_result_unscored", fmt.Sprintf("Task result %q was not scored.", taskID), []string{"Deferred submit only accepts results completed by the verifier."}, nil}
	}
	return nil
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
		details := map[string]any{"error": terminalSafeDownloadError(err)}
		if sanitized := terminalSanitizedDownloadURL(downloadURL); sanitized != "" {
			details["downloadUrl"] = sanitized
		}
		return nil, "", shardIndex, cliError{"manifest_fetch_failed", "Could not download terminal manifest JSONL: " + terminalSafeDownloadError(err), []string{"Signed manifest URLs expire after 15 minutes; re-run the command.", "Check network access to the storage host."}, details}
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

func downloadTerminalBundle(args cliArgs, tmp, id, key, wantHash string, wantByteSize ...int64) (string, error) {
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
		return "", cliError{"bundle_download_failed", "Could not download terminal bundle: " + terminalSafeDownloadError(err), []string{"Retry; signed URLs expire quickly."}, map[string]any{"taskId": id, "bundle_key": key}}
	}
	defer res.Body.Close()
	data, _ := io.ReadAll(res.Body)
	if res.StatusCode < 200 || res.StatusCode >= 300 {
		return "", cliError{"bundle_download_failed", fmt.Sprintf("Terminal bundle download returned %s", res.Status), []string{"Retry; signed URLs expire quickly."}, map[string]any{"taskId": id, "bundle_key": key}}
	}
	if len(wantByteSize) > 0 && wantByteSize[0] != int64(len(data)) {
		return "", cliError{"bundle_download_failed", "Terminal bundle byte size did not match the manifest.", []string{"Re-ingest the dataset; the manifest and bundle object are inconsistent."}, map[string]any{"taskId": id, "bundle_key": key, "expected": wantByteSize[0], "actual": len(data)}}
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
				printStatus(args, "terminal_task_done", map[string]any{"taskId": b.Task.ID, "pass": results[idx].pass, "scored": results[idx].scored, "turns": results[idx].turns, "wallTimeMs": results[idx].wallTimeMs, "tokenUsage": results[idx].usage.toMap()})
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
	for k, v := range resolveEnvTemplates(task.Environment.Env) {
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
		if err != nil && !captureTerminalAgentOutcome(&result, err) {
			result.errCode, result.errText = cliErrorCodeText(err)
			result.wallTimeMs = time.Since(started).Milliseconds()
			return persistTerminalTaskErrorTrace(task, result, cfg)
		}
	} else if cfg.shellMode == "stateless" {
		turns, transcript, usage, err := runTerminalAgentLoop(ctx, task, containerName, baseURL, model, cfg)
		result.turns = turns
		result.transcript = transcript
		result.usage = usage
		if err != nil && !captureTerminalAgentOutcome(&result, err) {
			result.errCode, result.errText = cliErrorCodeText(err)
			result.wallTimeMs = time.Since(started).Milliseconds()
			return persistTerminalTaskErrorTrace(task, result, cfg)
		}
	} else {
		result.prompt = terminalSessionSystemPrompt
		turns, transcript, usage, err := runTerminalAgentLoopSession(ctx, task, containerName, baseURL, model, cfg)
		result.turns = turns
		result.transcript = transcript
		result.usage = usage
		if err != nil && !captureTerminalAgentOutcome(&result, err) {
			result.errCode, result.errText = cliErrorCodeText(err)
			result.wallTimeMs = time.Since(started).Milliseconds()
			return persistTerminalTaskErrorTrace(task, result, cfg)
		}
	}
	pass, verifierOutput, err := runTerminalVerifier(ctx, task, bundleDir, containerName, cfg)
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
	printStatus(cfg.args, "terminal_external_agent_started", map[string]any{"taskId": task.ID, "backend": firstNonEmpty(opt(cfg.args, "agent"), "custom"), "execution": cfg.agentExecution})
	out, code, timedOut, runErr := runHostCommandWithEnv(ctx, timeout, env, cfg.agentCommand)
	out = terminalRedactAgentText(out, cfg.agentCommand, cfg.apiKey, baseURL)
	transcript := "$ [external agent command omitted]\n" + out + "\n[exit=" + strconv.Itoa(code) + "]\n"
	usage := externalAgentTokenUsage(traceDir)
	if usage.modelCalls > 0 {
		usageData, _ := json.MarshalIndent(usage.toMap(), "", "  ")
		_ = os.WriteFile(filepath.Join(traceDir, "usage.json"), append(usageData, '\n'), 0o644)
	}
	if traceText := terminalRedactAgentText(externalAgentTraceText(traceDir), cfg.agentCommand, cfg.apiKey, baseURL); traceText != "" {
		transcript += "\n\n# External agent trace directory\n\n" + traceText
	}
	printStatus(cfg.args, "terminal_external_agent_done", map[string]any{"taskId": task.ID, "exitCode": code, "timedOut": timedOut, "execution": cfg.agentExecution})
	if timedOut {
		transcript += "\n[agent timed out after " + timeout.String() + "; proceeding to verification]\n"
		printStatus(cfg.args, "terminal_external_agent_timeout", map[string]any{"taskId": task.ID, "timeoutSec": int(timeout.Seconds())})
		return transcript, usage, terminalAgentOutcomeError{code: "agent_timeout", text: "external agent timed out before verification"}
	}
	if runErr != nil || code != 0 {
		return transcript, usage, terminalCommandError("command_exec_failed", "External terminal agent command failed.", "bash", []string{"-lc", "[redacted]"}, code, out, timedOut)
	}
	return transcript, usage, nil
}

func terminalRedactAgentText(text, agentCommand, apiKey, baseURL string) string {
	if agentCommand != "" {
		text = strings.ReplaceAll(text, agentCommand, "[external agent command omitted]")
	}
	if apiKey != "" {
		text = strings.ReplaceAll(text, apiKey, "[model credential omitted]")
	}
	if safeEndpoint := terminalSanitizedEndpointOrigin(baseURL); safeEndpoint != "" && safeEndpoint != baseURL {
		text = strings.ReplaceAll(text, baseURL, safeEndpoint)
	}
	return text
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

func terminalJSONPrompt(instruction, terminalState string) string {
	return fmt.Sprintf(terminalJSONProtocolTemplate, instruction, terminalState)
}

func terminalJSONContinuePrompt(terminalState string) string {
	return "Current terminal state:\n" + terminalState + "\n\nContinue with the same JSON response format: analysis, plan, commands, and optional task_complete."
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

func runTerminalAgentLoop(ctx context.Context, task terminalTask, containerName, baseURL, model string, cfg terminalConfig) (int, string, terminalTokenUsage, error) {
	messages := []map[string]any{{"role": "system", "content": terminalSystemPrompt}, {"role": "user", "content": task.Instruction}}
	maxTurns := firstPositive(cfg.maxTurns, task.Agent.MaxTurns, 50)
	timeoutSec := terminalAgentTimeoutSec(cfg, task)
	deadline := time.Now().Add(time.Duration(timeoutSec) * time.Second)
	var transcript strings.Builder
	nonConforming := 0
	usage := terminalTokenUsage{}
	turnsCompleted := 0
	for turn := 1; turn <= maxTurns && time.Now().Before(deadline); turn++ {
		turnsCompleted = turn
		messages = trimTerminalMessages(messages)
		content, reasoning, callUsage, err := callOpenAIChatMessages(baseURL, model, messages, cfg.apiKey, terminalModelMaxTokens(cfg, false), cfg.temperature, cfg.topP, nil, terminalModelRequestTimeout(cfg, deadline, true))
		usage.add(callUsage)
		if err != nil {
			firstErr := err
			messages = trimTerminalMessagesForRetry(messages)
			content, reasoning, callUsage, err = callOpenAIChatMessages(baseURL, model, messages, cfg.apiKey, terminalModelMaxTokens(cfg, true), cfg.temperature, cfg.topP, nil, terminalModelRequestTimeout(cfg, deadline, false))
			usage.add(callUsage)
			if err != nil {
				return turn, transcript.String(), usage, terminalModelCallFailure(task.ID, deadline, firstErr, err)
			}
		}
		cmdText, found := extractBashCommand(content)
		if !found {
			if strings.Contains(content, "TASK_COMPLETE") {
				terminalTraceHeader(&transcript, turn, reasoning, content)
				transcript.WriteString("## Note\nModel signaled TASK_COMPLETE.\n")
				return turn - 1, transcript.String(), usage, nil
			}
			nonConforming++
			terminalTraceHeader(&transcript, turn, reasoning, content)
			transcript.WriteString("## Note\nNo bash block found; asked the model to emit one command or TASK_COMPLETE.\n")
			messages = append(messages, map[string]any{"role": "assistant", "content": compactAssistantForModel(content, "")}, map[string]any{"role": "user", "content": "Your previous reply was not executable. Reply with exactly one ```bash fenced block. If you meant to run Python, wrap it as: python3 <<'PY'\n...\nPY"})
			if nonConforming >= 3 {
				transcript.WriteString("## Note\nStopping after repeated non-executable replies.\n")
				return turn, transcript.String(), usage, terminalAgentOutcomeError{code: "protocol_exhausted", text: "agent exhausted the terminal response protocol retry limit"}
			}
			continue
		}
		nonConforming = 0
		messages = append(messages, map[string]any{"role": "assistant", "content": compactAssistantForModel(content, cmdText)})
		terminalTraceHeader(&transcript, turn, reasoning, content)
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
		printStatus(cfg.args, "terminal_turn", map[string]any{"taskId": task.ID, "turn": turn, "exitCode": code, "cmdPreview": truncateString(strings.ReplaceAll(cmdText, "\n", " "), 160)})
	}
	return turnsCompleted, transcript.String(), usage, terminalAgentLoopExhaustion(deadline)
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

func runTerminalAgentLoopSession(ctx context.Context, task terminalTask, containerName, baseURL, model string, cfg terminalConfig) (int, string, terminalTokenUsage, error) {
	shell, err := startTerminalShell(containerName, task.Agent.User)
	if err != nil {
		return 0, "", terminalTokenUsage{}, cliError{"command_exec_failed", "Could not open a persistent shell in the task container.", []string{"Check Docker and that the task image provides /bin/bash.", "Or rerun with --shell-mode stateless."}, map[string]any{"taskId": task.ID, "error": err.Error()}}
	}
	defer shell.close()

	messages := []map[string]any{{"role": "user", "content": terminalJSONPrompt(task.Instruction, "")}}
	maxTurns := terminalAgentMaxTurns(cfg, task)
	timeoutSec := terminalAgentTimeoutSec(cfg, task)
	deadline := time.Now().Add(time.Duration(timeoutSec) * time.Second)
	cmdTimeout := time.Duration(terminalCommandTimeoutSec(cfg)) * time.Second
	var transcript strings.Builder
	nonConforming := 0
	usage := terminalTokenUsage{}
	terminalState := ""
	turnsCompleted := 0
	for turn := 1; turn <= maxTurns && time.Now().Before(deadline); turn++ {
		turnsCompleted = turn
		messages = trimTerminalMessages(messages)
		content, reasoning, callUsage, err := callOpenAIChatMessages(baseURL, model, messages, cfg.apiKey, terminalModelMaxTokens(cfg, false), cfg.temperature, cfg.topP, nil, terminalModelRequestTimeout(cfg, deadline, true))
		usage.add(callUsage)
		if err != nil {
			firstErr := err
			messages = trimTerminalMessagesForRetry(messages)
			content, reasoning, callUsage, err = callOpenAIChatMessages(baseURL, model, messages, cfg.apiKey, terminalModelMaxTokens(cfg, true), cfg.temperature, cfg.topP, nil, terminalModelRequestTimeout(cfg, deadline, false))
			usage.add(callUsage)
			if err != nil {
				return turn, transcript.String(), usage, terminalModelCallFailure(task.ID, deadline, firstErr, err)
			}
		}

		terminalTraceHeader(&transcript, turn, reasoning, content)
		response, foundJSON := parseTerminalJSONResponse(content)
		if !foundJSON {
			cmdText, foundBash := extractBashCommand(content)
			if !foundBash {
				if strings.Contains(content, "TASK_COMPLETE") {
					transcript.WriteString("## Note\nModel signaled TASK_COMPLETE.\n")
					return turn - 1, transcript.String(), usage, nil
				}
				nonConforming++
				transcript.WriteString("## Note\nNo JSON command response or bash block found; asked the model to emit the required JSON.\n")
				messages = append(messages, map[string]any{"role": "assistant", "content": compactAssistantForModel(content, "")}, map[string]any{"role": "user", "content": terminalJSONContinuePrompt(terminalState)})
				if nonConforming >= 3 {
					transcript.WriteString("## Note\nStopping after repeated non-executable replies.\n")
					return turn, transcript.String(), usage, terminalAgentOutcomeError{code: "protocol_exhausted", text: "agent exhausted the terminal response protocol retry limit"}
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
		messages = append(messages, map[string]any{"role": "assistant", "content": content})
		if response.TaskComplete && len(response.Commands) == 0 {
			transcript.WriteString("## Note\nModel marked task complete.\n")
			return turn - 1, transcript.String(), usage, nil
		}

		var observation strings.Builder
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
				time.Sleep(wait)
				continue
			}
			transcript.WriteString("## Command\n$ " + cmdText + "\n")
			out, code, timedOut, restarted := shell.exec(cmdText, terminalCommandExecutionTimeout(cmdText, cmdTimeout, deadline))
			if timedOut {
				out += "\n[command timed out]"
			}
			shown := truncateString(out, 8192)
			observation.WriteString("$ " + cmdText + "\n")
			observation.WriteString(terminalObservationForModel(out, code, timedOut))
			observation.WriteString("\n")
			transcript.WriteString(shown + "\n[exit=" + strconv.Itoa(code) + "]\n")
			fields := map[string]any{"taskId": task.ID, "turn": turn, "commandIndex": i + 1, "exitCode": code, "cmdPreview": truncateString(strings.ReplaceAll(cmdText, "\n", " "), 160)}
			if restarted {
				fields["shellRestarted"] = true
			}
			printStatus(cfg.args, "terminal_turn", fields)
		}
		terminalState = truncateString(observation.String(), terminalModelObservationLimit)
		messages = append(messages, map[string]any{"role": "user", "content": terminalJSONContinuePrompt(terminalState)})
		if response.TaskComplete {
			transcript.WriteString("## Note\nModel marked task complete after command batch.\n")
			return turn, transcript.String(), usage, nil
		}
	}
	return turnsCompleted, transcript.String(), usage, terminalAgentLoopExhaustion(deadline)
}

func terminalAgentLoopExhaustion(deadline time.Time) terminalAgentOutcomeError {
	if !time.Now().Before(deadline) {
		return terminalAgentOutcomeError{code: "agent_timeout", text: "agent exhausted its wall-clock timeout before verifier success"}
	}
	return terminalAgentOutcomeError{code: "max_turns_exhausted", text: "agent exhausted its maximum turns before verifier success"}
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

// runTerminalVerifier follows harbor canonical semantics: tests/ is copied to
// /tests, the verifier command runs in a non-login shell, and the reward file
// is the sole pass signal — reward.json ({"reward": <num>}) takes precedence
// over reward.txt (bare float), pass means reward >= 1.0, and the verifier's
// exit code is ignored once a reward was written. A verifier timeout or a
// missing/unparseable reward file scores the task as failed.
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
	for k, v := range resolveEnvTemplates(task.Verifier.Env) {
		cmdArgs = append(cmdArgs, "-e", k+"="+v)
	}
	cmdArgs = append(cmdArgs, containerName, "bash", "-c", task.Verifier.Command)
	out, code, timedOut, _ = runCommand(ctx, time.Duration(firstPositive(task.Verifier.TimeoutSec, 900))*time.Second, "docker", cmdArgs...)
	if timedOut {
		printStatus(cfg.args, "terminal_verifier", map[string]any{"taskId": task.ID, "reward": "", "exitCode": code, "timedOut": true})
		return false, out + "\n[verifier timed out]", cliError{"verifier_failed", "Verifier timed out; the task is scored as failed.", []string{"Raise [verifier].timeout_sec in the task if legitimate verifications need longer."}, map[string]any{"taskId": task.ID, "timeoutSec": firstPositive(task.Verifier.TimeoutSec, 900), "output": truncateString(out, 4096)}}
	}
	rewardFile := firstNonEmpty(task.Verifier.RewardFile, "/logs/verifier/reward.txt")
	rewardJSONFile := path.Join(path.Dir(rewardFile), "reward.json")
	reward, rewardRaw, rewardOK := 0.0, "", false
	if jsonText, jsonCode, _, _ := runCommand(ctx, 30*time.Second, "docker", "exec", containerName, "cat", rewardJSONFile); jsonCode == 0 {
		reward, rewardOK = parseRewardJSON(jsonText)
		rewardRaw = strings.TrimSpace(jsonText)
	}
	if !rewardOK {
		if txtText, txtCode, _, _ := runCommand(ctx, 30*time.Second, "docker", "exec", containerName, "cat", rewardFile); txtCode == 0 {
			reward, rewardOK = parseRewardText(txtText)
			rewardRaw = strings.TrimSpace(txtText)
		}
	}
	output := out + "\n[verifier exit=" + strconv.Itoa(code) + "]\nreward: " + rewardRaw
	printStatus(cfg.args, "terminal_verifier", map[string]any{"taskId": task.ID, "reward": rewardRaw, "exitCode": code})
	if !rewardOK {
		return false, output, cliError{"verifier_failed", "Verifier did not produce a parseable reward file; the task is scored as failed.", []string{"Harbor verifiers must write /logs/verifier/reward.txt (float) or reward.json ({\"reward\": <num>})."}, map[string]any{"taskId": task.ID, "rewardFile": rewardFile, "exitCode": code, "output": truncateString(out, 4096)}}
	}
	return reward >= 1.0, output, nil
}

func callOpenAIChatMessages(baseURL, model string, messages []map[string]any, apiKey string, maxTokens int, temperature, topP float64, stop []string, timeout time.Duration) (content, reasoning string, usage terminalTokenUsage, err error) {
	body := map[string]any{"model": model, "messages": messages, "temperature": temperature, "top_p": topP}
	if terminalDisablesTemplateThinking(model) {
		body["chat_template_kwargs"] = map[string]any{"enable_thinking": false}
	}
	if maxTokens > 0 {
		body["max_tokens"] = maxTokens
	}
	if len(stop) > 0 {
		body["stop"] = stop
	}
	data, _ := json.Marshal(body)
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
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
			reasoning = strings.TrimSpace(stringValue(message["reasoning_content"]))
		}
	}
	return content, reasoning, usage, nil
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
	resultJSON := map[string]any{"question_id": task.ID, "pass": result.pass, "scored": result.scored, "error": result.errText, "errorCode": result.errCode, "latencyMs": result.wallTimeMs, "wallTimeMs": result.wallTimeMs, "tokenUsage": result.usage.toMap(), "turns": result.turns}
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
func terminalAgentMaxTurns(cfg terminalConfig, task terminalTask) int {
	if cfg.maxTurns > 0 {
		return cfg.maxTurns
	}
	if task.Agent.MaxTurns > 100 {
		return task.Agent.MaxTurns
	}
	return 100
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
	if cfg.agentTimeoutSec > 0 {
		return cfg.agentTimeoutSec
	}
	if task.Agent.TimeoutSec > defaultTerminalTaskTimeoutSec {
		return task.Agent.TimeoutSec
	}
	return defaultTerminalTaskTimeoutSec
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
		total += len(stringValue(message["role"]))
		total += len(stringValue(message["content"]))
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
