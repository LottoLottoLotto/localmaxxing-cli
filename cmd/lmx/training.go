package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"
)

const (
	trainingSchemaVersion          = 1
	defaultTrainingMaxMessageBytes = 64 * 1024
	trainingTruncationMarker       = "\n[LMX_TRAINING_DATA_TRUNCATED]"
)

type evalTrainingRecord struct {
	ID             string
	Pass           bool
	Scored         bool
	Question       string
	VerifierOutput string
	Error          string
	ErrorCode      string
	WallTimeMs     int64
	TokenUsage     any
	TracePath      string
	ResultPath     string
}

type trainingExportStats struct {
	Total           int
	Scored          int
	Passed          int
	Failed          int
	SFTExamples     int
	FailureExamples int
	Skipped         map[string]int
	TruncatedFields int
	PreferencePairs int
}

func handleEvalTrain(action string, args cliArgs) error {
	switch action {
	case "prepare":
		return prepareEvalTrainingData(args)
	case "run":
		return runEvalTrainer(args)
	case "rl":
		switch positional(args, 3) {
		case "prepare":
			return prepareEvalRL(args)
		case "run":
			return runEvalRLTrainer(args)
		default:
			return cliError{"unknown_subcommand", "Unknown eval train rl subcommand.", []string{"Use one of: prepare, run."}, map[string]any{"subcommand": positional(args, 3)}}
		}
	default:
		return cliError{
			Code:    "unknown_subcommand",
			Message: "Unknown eval train subcommand.",
			Hints:   []string{"Use one of: prepare, run, rl prepare, rl run."},
			Details: map[string]any{"subcommand": action},
		}
	}
}

func prepareEvalTrainingData(args cliArgs) error {
	source := positional(args, 3)
	if source == "" {
		return cliError{"missing_source", "eval train prepare requires a completed eval run directory or result JSON.", []string{"Run: lmx eval train prepare <run-dir-or-result.json> --out <training-dir> --base-model <hfId> --allow-benchmark-training."}, nil}
	}
	outDir := opt(args, "out")
	if outDir == "" {
		return cliError{"missing_option", "eval train prepare requires --out <training-dir>.", nil, nil}
	}
	baseModel := opt(args, "base-model")
	if baseModel == "" {
		return cliError{"missing_option", "eval train prepare requires --base-model <HuggingFace model id>.", []string{"Use the original loadable HuggingFace checkpoint, not a GGUF file; QLoRA adapters cannot be trained directly from GGUF."}, nil}
	}
	if !hasFlag(args, "allow-benchmark-training") {
		return cliError{
			Code:    "benchmark_training_not_acknowledged",
			Message: "Training on eval tasks contaminates future scores on those tasks.",
			Hints: []string{
				"Keep the eval tasks as a holdout if you need an honest comparable score.",
				"If this is an intentional improvement dataset, rerun with --allow-benchmark-training and evaluate the trained model on a separate holdout.",
			},
			Details: map[string]any{"source": source},
		}
	}
	maxMessageBytes := defaultTrainingMaxMessageBytes
	if raw := opt(args, "max-message-bytes"); raw != "" {
		value, err := strconv.Atoi(raw)
		if err != nil || value < 1024 {
			return cliError{"invalid_option", "--max-message-bytes must be an integer of at least 1024.", nil, map[string]any{"value": raw}}
		}
		maxMessageBytes = value
	}

	records, sourceRoot, err := loadEvalTrainingRecords(source)
	if err != nil {
		return err
	}
	stats := trainingExportStats{Skipped: map[string]int{}}
	sftRows := make([]map[string]any, 0)
	failureRows := make([]map[string]any, 0)
	for i := range records {
		record := &records[i]
		stats.Total++
		if record.Scored {
			stats.Scored++
		}
		if record.Pass {
			stats.Passed++
		} else {
			stats.Failed++
		}
		if record.TracePath == "" && sourceRoot != "" {
			record.TracePath = findOMPTrace(sourceRoot, record.ID)
		}

		if record.Scored && record.Pass {
			if record.TracePath == "" {
				stats.Skipped["passing_trace_missing"]++
				continue
			}
			messages, truncated, parseErr := readOMPTrainingMessages(record.TracePath, maxMessageBytes)
			stats.TruncatedFields += truncated
			if parseErr != nil {
				stats.Skipped["passing_trace_invalid"]++
				continue
			}
			if !validTrainingConversation(messages) {
				stats.Skipped["passing_conversation_incomplete"]++
				continue
			}
			sftRows = append(sftRows, map[string]any{
				"id":       record.ID,
				"messages": messages,
				"metadata": map[string]any{
					"source":        "terminal_eval",
					"pass":          true,
					"scored":        true,
					"wallTimeMs":    record.WallTimeMs,
					"tokenUsage":    record.TokenUsage,
					"resultPath":    record.ResultPath,
					"tracePath":     record.TracePath,
					"benchmarkTask": record.ID,
				},
			})
			continue
		}

		question, qTruncated := truncateTrainingText(record.Question, maxMessageBytes)
		verifier, vTruncated := truncateTrainingText(record.VerifierOutput, maxMessageBytes)
		if qTruncated {
			stats.TruncatedFields++
		}
		if vTruncated {
			stats.TruncatedFields++
		}
		failureRows = append(failureRows, map[string]any{
			"id":             record.ID,
			"instruction":    question,
			"verifierOutput": verifier,
			"pass":           record.Pass,
			"scored":         record.Scored,
			"error":          record.Error,
			"errorCode":      record.ErrorCode,
			"wallTimeMs":     record.WallTimeMs,
			"resultPath":     record.ResultPath,
			"tracePath":      record.TracePath,
			"trainingUse":    "diagnostic_only",
		})
	}
	stats.SFTExamples = len(sftRows)
	stats.FailureExamples = len(failureRows)
	if len(sftRows) == 0 {
		return cliError{"no_training_examples", "No scored passing OMP trajectories could be exported.", []string{"Confirm the run has passing tasks and retained agent/*/omp.jsonl traces.", "Only finalized message_end events are eligible; transcript.md and failed trajectories are not used as SFT labels."}, map[string]any{"records": len(records), "skipped": stats.Skipped}}
	}

	absOut, err := filepath.Abs(outDir)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(absOut, 0o755); err != nil {
		return cliError{"directory_create_error", fmt.Sprintf("Could not create training output directory: %v", err), nil, map[string]any{"path": absOut}}
	}
	sftPath := filepath.Join(absOut, "sft.jsonl")
	failuresPath := filepath.Join(absOut, "failures.jsonl")
	manifestPath := filepath.Join(absOut, "manifest.json")
	if err := writeJSONLAtomic(sftPath, sftRows); err != nil {
		return err
	}
	if err := writeJSONLAtomic(failuresPath, failureRows); err != nil {
		return err
	}
	absSource, _ := filepath.Abs(source)
	manifest := map[string]any{
		"schemaVersion": trainingSchemaVersion,
		"createdAt":     time.Now().UTC().Format(time.RFC3339),
		"source": map[string]any{
			"path": absSource,
			"kind": "terminal_eval",
		},
		"baseModel": baseModel,
		"dataset": map[string]any{
			"format":          "trl_conversational_sft_jsonl",
			"sftPath":         sftPath,
			"failuresPath":    failuresPath,
			"sftExamples":     stats.SFTExamples,
			"failureExamples": stats.FailureExamples,
			"maxMessageBytes": maxMessageBytes,
		},
		"results": map[string]any{
			"total":           stats.Total,
			"scored":          stats.Scored,
			"passed":          stats.Passed,
			"failed":          stats.Failed,
			"skipped":         stats.Skipped,
			"truncatedFields": stats.TruncatedFields,
			"preferencePairs": 0,
		},
		"policy": map[string]any{
			"sftEligibility":      "scored passing trajectories only",
			"failedTrajectoryUse": "diagnostic_only",
			"thinkingIncluded":    false,
			"preferenceData":      "not generated: a single pass/fail attempt is not a preference pair",
		},
		"contamination": map[string]any{
			"benchmarkDerived": true,
			"acknowledged":     true,
			"warning":          "This data contains eval tasks. Do not report scores on the same tasks as uncontaminated model quality; use a separate holdout.",
		},
		"trainer": map[string]any{
			"state":     "not_started",
			"outputDir": filepath.Join(absOut, "adapter"),
			"interface": "Provide --trainer-cmd with optional {dataset}, {manifest}, {output}, and {base_model} placeholders; execution requires --execute.",
		},
	}
	if err := writeJSONAtomic(manifestPath, manifest); err != nil {
		return err
	}
	printInfo("eval_training_data_written", map[string]any{
		"manifest":        manifestPath,
		"sft":             sftPath,
		"failures":        failuresPath,
		"examples":        stats.SFTExamples,
		"diagnostics":     stats.FailureExamples,
		"preferencePairs": 0,
	})
	return nil
}

func runEvalTrainer(args cliArgs) error {
	manifestPath := positional(args, 3)
	if manifestPath == "" {
		return cliError{"missing_source", "eval train run requires a manifest.json produced by eval train prepare.", nil, nil}
	}
	value, err := readJSON(manifestPath)
	if err != nil {
		return err
	}
	manifest, ok := value.(map[string]any)
	if !ok {
		return cliError{"training_manifest_invalid", "Training manifest must be a JSON object.", nil, map[string]any{"path": manifestPath}}
	}
	if int(numberField(manifest, "schemaVersion")) != trainingSchemaVersion {
		return cliError{"training_manifest_invalid", "Unsupported training manifest schema version.", nil, map[string]any{"path": manifestPath, "schemaVersion": manifest["schemaVersion"]}}
	}
	dataset := asObject(manifest["dataset"])
	trainer := asObject(manifest["trainer"])
	baseModel := stringValue(manifest["baseModel"])
	sftPath := stringValue(dataset["sftPath"])
	outputDir := stringValue(trainer["outputDir"])
	if baseModel == "" || sftPath == "" || outputDir == "" {
		return cliError{"training_manifest_invalid", "Training manifest is missing baseModel, dataset.sftPath, or trainer.outputDir.", nil, map[string]any{"path": manifestPath}}
	}
	if _, err := os.Stat(sftPath); err != nil {
		return cliError{"training_dataset_missing", "The manifest SFT dataset does not exist.", nil, map[string]any{"path": sftPath, "error": err.Error()}}
	}
	command := opt(args, "trainer-cmd")
	if command == "" {
		return cliError{"missing_option", "eval train run requires --trainer-cmd.", []string{"The CLI does not invent a remote trainer. Pass a local TRL/Axolotl/Unsloth command and use placeholders {dataset}, {manifest}, {output}, and {base_model}."}, nil}
	}
	absManifest, _ := filepath.Abs(manifestPath)
	replacements := map[string]string{
		"{dataset}":    sftPath,
		"{manifest}":   absManifest,
		"{output}":     outputDir,
		"{base_model}": baseModel,
	}
	for placeholder, replacement := range replacements {
		command = strings.ReplaceAll(command, placeholder, shellQuote(replacement))
	}
	if !hasFlag(args, "execute") {
		return writeOrPrintJSON("eval_training_plan", args, map[string]any{
			"command":   command,
			"execute":   false,
			"manifest":  absManifest,
			"dataset":   sftPath,
			"baseModel": baseModel,
			"output":    outputDir,
			"note":      "No trainer was started. Add --execute to run this explicit local command.",
		})
	}
	if err := os.MkdirAll(outputDir, 0o755); err != nil {
		return cliError{"directory_create_error", fmt.Sprintf("Could not create trainer output directory: %v", err), nil, map[string]any{"path": outputDir}}
	}
	printStatus(args, "eval_trainer_start", map[string]any{"command": command, "dataset": sftPath, "output": outputDir})
	var cmd *exec.Cmd
	if runtime.GOOS == "windows" {
		cmd = exec.Command("cmd", "/C", command)
	} else {
		cmd = exec.Command("sh", "-c", command)
	}
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return cliError{"trainer_command_failed", "The explicit local trainer command failed.", []string{"Inspect the trainer output and verify its dependencies, model access, and GPU memory settings."}, map[string]any{"command": command, "error": err.Error()}}
	}
	printStatus(args, "eval_trainer_complete", map[string]any{"output": outputDir})
	return nil
}

const (
	evalRLManifestKind      = "localmaxxing.eval_rl_grpo"
	evalRLAlgorithm         = "online_grpo"
	evalRLDatasetFormat     = "trl_conversational_prompt_jsonl"
	evalRLEnvironmentSchema = 1
	evalRLTrainerImpl       = "localmaxxing_trl_grpo"
	evalRLTRLVersion        = "1.8.0"
	evalRLContaminationNote = "This data contains eval tasks. Do not report scores on the same tasks as uncontaminated model quality; use a separate holdout."
)

var evalRLFactoryPattern = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+:[A-Za-z_][A-Za-z0-9_]*$`)

type evalRLPromptMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type evalRLPromptRow struct {
	Prompt    []evalRLPromptMessage `json:"prompt"`
	TaskID    string                `json:"task_id"`
	BundleRef string                `json:"bundle_ref"`
}

type evalRLSourceManifest struct {
	BundleRoot string `json:"bundleRoot"`
	TaskCount  int    `json:"taskCount"`
}

type evalRLDatasetManifest struct {
	Format   string   `json:"format"`
	Path     string   `json:"path"`
	Examples int      `json:"examples"`
	Columns  []string `json:"columns"`
}

type evalRLEnvironmentManifest struct {
	ContractVersion int            `json:"contractVersion"`
	Factory         string         `json:"factory"`
	Config          map[string]any `json:"config"`
}

type evalRLGRPOConfig struct {
	NumGenerations            int     `json:"num_generations"`
	MaxSteps                  int     `json:"max_steps"`
	LearningRate              float64 `json:"learning_rate"`
	PerDeviceTrainBatchSize   int     `json:"per_device_train_batch_size"`
	GradientAccumulationSteps int     `json:"gradient_accumulation_steps"`
	MaxCompletionLength       int     `json:"max_completion_length"`
	MaxToolCallingIterations  int     `json:"max_tool_calling_iterations"`
	GradientCheckpointing     bool    `json:"gradient_checkpointing"`
	LoggingSteps              int     `json:"logging_steps"`
	SaveSteps                 int     `json:"save_steps"`
	SaveTotalLimit            int     `json:"save_total_limit"`
	Seed                      int64   `json:"seed"`
	present                   map[string]bool
}

type evalRLTrainerManifest struct {
	Implementation string           `json:"implementation"`
	TRLVersion     string           `json:"trlVersion"`
	OutputDir      string           `json:"outputDir"`
	GRPOConfig     evalRLGRPOConfig `json:"grpoConfig"`
}

type evalRLContaminationManifest struct {
	BenchmarkDerived bool   `json:"benchmarkDerived"`
	Acknowledged     bool   `json:"acknowledged"`
	Warning          string `json:"warning"`
}

type evalRLManifest struct {
	SchemaVersion int                            `json:"schemaVersion"`
	Kind          string                         `json:"kind"`
	Algorithm     string                         `json:"algorithm"`
	BaseModel     string                         `json:"baseModel"`
	Source        evalRLSourceManifest           `json:"source"`
	Dataset       evalRLDatasetManifest          `json:"dataset"`
	Environment   evalRLEnvironmentManifest      `json:"environment"`
	Trainer       evalRLTrainerManifest          `json:"trainer"`
	Contamination evalRLContaminationManifest    `json:"contamination"`
}

type evalRLBundle struct {
	TaskID      string
	Instruction string
	Ref         string
	Dir         string
}

func defaultEvalRLGRPOConfig() evalRLGRPOConfig {
	return evalRLGRPOConfig{
		NumGenerations:            4,
		MaxSteps:                  100,
		LearningRate:              1e-6,
		PerDeviceTrainBatchSize:   1,
		GradientAccumulationSteps: 4,
		MaxCompletionLength:       2048,
		MaxToolCallingIterations:  20,
		GradientCheckpointing:     true,
		LoggingSteps:              1,
		SaveSteps:                 25,
		SaveTotalLimit:            2,
		Seed:                      42,
	}
}

func prepareEvalRL(args cliArgs) error {
	sourceArg := positional(args, 4)
	if !hasFlag(args, "allow-benchmark-training") {
		return cliError{
			Code:    "benchmark_training_not_acknowledged",
			Message: "Training on eval tasks contaminates future scores on those tasks.",
			Hints: []string{
				"Keep the eval tasks as a holdout if you need an honest comparable score.",
				"Rerun with --allow-benchmark-training only when you accept this contamination and will evaluate on a separate holdout.",
			},
			Details: map[string]any{"source": sourceArg},
		}
	}
	if sourceArg == "" {
		return cliError{"missing_source", "eval train rl prepare requires an imported terminal task bundle or a parent directory of bundles.", nil, nil}
	}
	outArg := strings.TrimSpace(opt(args, "out"))
	if outArg == "" {
		return cliError{"missing_option", "eval train rl prepare requires --out <rl-dir>.", nil, nil}
	}
	baseModel := strings.TrimSpace(opt(args, "base-model"))
	if baseModel == "" {
		return cliError{"missing_option", "eval train rl prepare requires --base-model <id-or-path>.", nil, nil}
	}
	factory := strings.TrimSpace(opt(args, "environment-factory"))
	if factory == "" {
		return cliError{"missing_option", "eval train rl prepare requires --environment-factory <module:callable>.", nil, nil}
	}
	if !evalRLFactoryPattern.MatchString(factory) {
		return cliError{"invalid_environment_factory", "--environment-factory must be a dotted Python module followed by one callable identifier.", []string{"Use a value such as my_package.environments:make_environment."}, map[string]any{"factory": factory}}
	}
	environmentConfig, err := parseEvalRLJSONObjectOption(args, "environment-config")
	if err != nil {
		return err
	}
	grpoOverrides, err := parseEvalRLJSONObjectOption(args, "grpo-config")
	if err != nil {
		return err
	}
	grpoConfig, err := evalRLGRPOConfigWithOverrides(grpoOverrides)
	if err != nil {
		return err
	}
	sourceRoot, err := canonicalExistingDirectory(sourceArg, "rl_source_invalid")
	if err != nil {
		return err
	}
	bundles, err := loadEvalRLBundles(sourceRoot)
	if err != nil {
		return err
	}
	absOut, err := canonicalPathForWrite(outArg)
	if err != nil {
		return cliError{"rl_output_invalid", "Could not resolve the RL output directory.", nil, map[string]any{"path": outArg, "error": err.Error()}}
	}
	if pathWithin(sourceRoot, absOut) || pathWithin(absOut, sourceRoot) {
		return cliError{"rl_output_overlaps_source", "The RL output directory and source bundle root must not overlap.", []string{"Choose --out outside the imported terminal bundle tree."}, map[string]any{"source": sourceRoot, "out": absOut}}
	}
	if err := requireEmptyOrMissingDirectory(absOut); err != nil {
		return err
	}

	rows := make([]evalRLPromptRow, 0, len(bundles))
	for _, bundle := range bundles {
		rows = append(rows, evalRLPromptRow{
			Prompt:    []evalRLPromptMessage{{Role: "user", Content: bundle.Instruction}},
			TaskID:    bundle.TaskID,
			BundleRef: bundle.Ref,
		})
	}
	sort.Slice(rows, func(i, j int) bool { return rows[i].TaskID < rows[j].TaskID })
	if err := os.MkdirAll(absOut, 0o755); err != nil {
		return cliError{"directory_create_error", "Could not create the RL output directory.", nil, map[string]any{"path": absOut, "error": err.Error()}}
	}
	promptsPath := filepath.Join(absOut, "prompts.jsonl")
	manifestPath := filepath.Join(absOut, "manifest.json")
	if err := writeEvalRLPromptRowsAtomic(promptsPath, rows); err != nil {
		return cliError{"rl_dataset_write_failed", "Could not write the RL prompt dataset.", nil, map[string]any{"path": promptsPath, "error": err.Error()}}
	}
	manifest := evalRLManifest{
		SchemaVersion: trainingSchemaVersion,
		Kind:          evalRLManifestKind,
		Algorithm:     evalRLAlgorithm,
		BaseModel:     baseModel,
		Source:        evalRLSourceManifest{BundleRoot: sourceRoot, TaskCount: len(rows)},
		Dataset: evalRLDatasetManifest{
			Format:   evalRLDatasetFormat,
			Path:     promptsPath,
			Examples: len(rows),
			Columns:  []string{"prompt", "task_id", "bundle_ref"},
		},
		Environment: evalRLEnvironmentManifest{ContractVersion: evalRLEnvironmentSchema, Factory: factory, Config: environmentConfig},
		Trainer: evalRLTrainerManifest{
			Implementation: evalRLTrainerImpl,
			TRLVersion:     evalRLTRLVersion,
			OutputDir:      filepath.Join(absOut, "grpo-output"),
			GRPOConfig:     grpoConfig,
		},
		Contamination: evalRLContaminationManifest{BenchmarkDerived: true, Acknowledged: true, Warning: evalRLContaminationNote},
	}
	if err := writeJSONAtomic(manifestPath, manifest); err != nil {
		return cliError{"rl_manifest_write_failed", "Could not write the RL training manifest.", nil, map[string]any{"path": manifestPath, "error": err.Error()}}
	}
	printStatus(args, "eval_rl_training_data_written", map[string]any{"manifest": manifestPath, "dataset": promptsPath, "examples": len(rows)})
	return nil
}

func runEvalRLTrainer(args cliArgs) error {
	manifestArg := positional(args, 4)
	if manifestArg == "" {
		return cliError{"missing_source", "eval train rl run requires a manifest.json produced by eval train rl prepare.", nil, nil}
	}
	if hasFlag(args, "trainer-cmd") || strings.TrimSpace(opt(args, "trainer-cmd")) != "" {
		return cliError{"invalid_option", "eval train rl run does not accept --trainer-cmd or arbitrary trainer pass-through.", []string{"The RL runner uses its embedded validated Python helper and direct argv execution."}, nil}
	}
	for _, key := range []string{"output-dir", "resume", "python-bin"} {
		if hasFlag(args, key) {
			return cliError{"invalid_option", "--" + key + " requires a value.", nil, nil}
		}
	}
	if _, provided := args.opts["output-dir"]; provided && strings.TrimSpace(opt(args, "output-dir")) == "" {
		return cliError{"invalid_option", "--output-dir must not be empty.", nil, nil}
	}
	if _, provided := args.opts["resume"]; provided && strings.TrimSpace(opt(args, "resume")) == "" {
		return cliError{"invalid_option", "--resume must not be empty.", nil, nil}
	}
	if _, provided := args.opts["python-bin"]; provided && strings.TrimSpace(opt(args, "python-bin")) == "" {
		return cliError{"invalid_option", "--python-bin must not be empty.", nil, nil}
	}
	manifestPath, err := canonicalExistingFile(manifestArg, "rl_manifest_invalid")
	if err != nil {
		return err
	}
	manifest, err := readEvalRLManifest(manifestPath)
	if err != nil {
		return err
	}
	if err := validateEvalRLManifest(manifestPath, manifest); err != nil {
		return err
	}
	outputDir := manifest.Trainer.OutputDir
	if override := strings.TrimSpace(opt(args, "output-dir")); override != "" {
		outputDir, err = canonicalPathForWrite(override)
		if err != nil {
			return cliError{"rl_output_invalid", "Could not resolve --output-dir.", nil, map[string]any{"path": override, "error": err.Error()}}
		}
	}
	if info, statErr := os.Stat(outputDir); statErr == nil && !info.IsDir() {
		return cliError{"rl_output_invalid", "The trainer output path exists and is not a directory.", nil, map[string]any{"path": outputDir}}
	} else if statErr != nil && !os.IsNotExist(statErr) {
		return cliError{"rl_output_invalid", "Could not inspect the trainer output path.", nil, map[string]any{"path": outputDir, "error": statErr.Error()}}
	}
	if pathWithin(manifest.Source.BundleRoot, outputDir) {
		return cliError{"rl_output_overlaps_source", "The trainer output directory must not be the source bundle root or a directory beneath it.", []string{"Choose --output-dir outside the imported terminal bundle tree."}, map[string]any{"source": manifest.Source.BundleRoot, "output": outputDir}}
	}
	resume, err := resolveEvalRLResumeSelector(opt(args, "resume"))
	if err != nil {
		return err
	}
	pythonBin := strings.TrimSpace(opt(args, "python-bin"))
	if pythonBin == "" {
		pythonBin = "python3"
	}
	planArgv := []string{pythonBin, "<embedded:train_eval_grpo.py>", "--manifest", manifestPath, "--output-dir", outputDir, "--resume", resume}
	if !hasFlag(args, "execute") {
		printJSON(map[string]any{
			"argv":           planArgv,
			"execute":        false,
			"manifest":       manifestPath,
			"dataset":        manifest.Dataset.Path,
			"baseModel":      manifest.BaseModel,
			"outputDir":      outputDir,
			"resumeSelector": resume,
			"note":           "No trainer was started. Add --execute to run the embedded online GRPO helper.",
		})
		return nil
	}
	pythonPath, err := exec.LookPath(pythonBin)
	if err != nil {
		return cliError{"python_not_found", "Could not find the requested Python executable.", []string{"Install Python or pass --python-bin with an executable path."}, map[string]any{"pythonBin": pythonBin, "error": err.Error()}}
	}
	helper, err := os.CreateTemp("", "lmx-train-eval-grpo-*.py")
	if err != nil {
		return cliError{"rl_helper_materialize_failed", "Could not create a temporary file for the embedded RL helper.", nil, map[string]any{"error": err.Error()}}
	}
	helperPath := helper.Name()
	defer os.Remove(helperPath)
	if _, err := helper.Write(trainEvalGRPOScript); err != nil {
		helper.Close()
		return cliError{"rl_helper_materialize_failed", "Could not write the embedded RL helper to its temporary file.", nil, map[string]any{"path": helperPath, "error": err.Error()}}
	}
	if err := helper.Close(); err != nil {
		return cliError{"rl_helper_materialize_failed", "Could not finish writing the embedded RL helper.", nil, map[string]any{"path": helperPath, "error": err.Error()}}
	}
	childArgs := []string{helperPath, "--manifest", manifestPath, "--output-dir", outputDir, "--resume", resume}
	printStatus(args, "eval_rl_trainer_start", map[string]any{"argv": append([]string{pythonPath}, childArgs...), "output": outputDir, "resume": resume})
	cmd := exec.Command(pythonPath, childArgs...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		details := map[string]any{"python": pythonPath, "manifest": manifestPath, "output": outputDir, "resume": resume, "error": err.Error()}
		var exitErr *exec.ExitError
		if errors.As(err, &exitErr) {
			details["exitCode"] = exitErr.ExitCode()
			return cliError{"rl_trainer_failed", "The embedded online GRPO trainer exited unsuccessfully.", []string{"Inspect the Python error above and verify the manifest, plugin, dependencies, model access, and GPU memory settings."}, details}
		}
		return cliError{"rl_trainer_start_failed", "Could not start the embedded online GRPO trainer.", nil, details}
	}
	printStatus(args, "eval_rl_trainer_complete", map[string]any{"output": outputDir})
	return nil
}

func parseEvalRLJSONObjectOption(args cliArgs, key string) (map[string]any, error) {
	rawPath, provided := args.opts[key]
	if !provided {
		if hasFlag(args, key) {
			return nil, cliError{"invalid_option", "--" + key + " requires a JSON object file path.", nil, nil}
		}
		return map[string]any{}, nil
	}
	if strings.TrimSpace(rawPath) == "" {
		return nil, cliError{"invalid_option", "--" + key + " file path must not be empty.", nil, nil}
	}
	configPath, err := canonicalExistingFile(rawPath, "invalid_option")
	if err != nil {
		return nil, cliError{"invalid_option", "Could not read --" + key + " JSON object file.", nil, map[string]any{"path": rawPath, "error": err.Error()}}
	}
	data, err := os.ReadFile(configPath)
	if err != nil {
		return nil, cliError{"invalid_option", "Could not read --" + key + " JSON object file.", nil, map[string]any{"path": configPath, "error": err.Error()}}
	}
	if len(bytes.TrimSpace(data)) == 0 {
		return nil, cliError{"invalid_option", "--" + key + " JSON object file must not be empty.", nil, map[string]any{"path": configPath}}
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	var value any
	if err := decoder.Decode(&value); err != nil {
		return nil, cliError{"invalid_option", "--" + key + " file must contain a valid JSON object.", nil, map[string]any{"path": configPath, "error": err.Error()}}
	}
	if err := requireJSONEOF(decoder); err != nil {
		return nil, cliError{"invalid_option", "--" + key + " file must contain exactly one JSON object.", nil, map[string]any{"path": configPath, "error": err.Error()}}
	}
	object, ok := value.(map[string]any)
	if !ok || object == nil {
		return nil, cliError{"invalid_option", "--" + key + " file must contain a JSON object, not null or another JSON type.", nil, map[string]any{"path": configPath}}
	}
	return object, nil
}

func evalRLGRPOConfigWithOverrides(overrides map[string]any) (evalRLGRPOConfig, error) {
	config := defaultEvalRLGRPOConfig()
	keys := make([]string, 0, len(overrides))
	for key := range overrides {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		value := overrides[key]
		switch key {
		case "num_generations":
			v, err := evalRLConfigInt(key, value, 2)
			if err != nil {
				return config, err
			}
			config.NumGenerations = int(v)
		case "max_steps":
			v, err := evalRLConfigInt(key, value, 1)
			if err != nil {
				return config, err
			}
			config.MaxSteps = int(v)
		case "learning_rate":
			v, err := evalRLConfigNumber(key, value)
			if err != nil {
				return config, err
			}
			config.LearningRate = v
		case "per_device_train_batch_size":
			v, err := evalRLConfigInt(key, value, 1)
			if err != nil {
				return config, err
			}
			config.PerDeviceTrainBatchSize = int(v)
		case "gradient_accumulation_steps":
			v, err := evalRLConfigInt(key, value, 1)
			if err != nil {
				return config, err
			}
			config.GradientAccumulationSteps = int(v)
		case "max_completion_length":
			v, err := evalRLConfigInt(key, value, 1)
			if err != nil {
				return config, err
			}
			config.MaxCompletionLength = int(v)
		case "max_tool_calling_iterations":
			v, err := evalRLConfigInt(key, value, 1)
			if err != nil {
				return config, err
			}
			config.MaxToolCallingIterations = int(v)
		case "gradient_checkpointing":
			v, ok := value.(bool)
			if !ok {
				return config, evalRLConfigError(key, "a boolean", value)
			}
			config.GradientCheckpointing = v
		case "logging_steps":
			v, err := evalRLConfigInt(key, value, 1)
			if err != nil {
				return config, err
			}
			config.LoggingSteps = int(v)
		case "save_steps":
			v, err := evalRLConfigInt(key, value, 1)
			if err != nil {
				return config, err
			}
			config.SaveSteps = int(v)
		case "save_total_limit":
			v, err := evalRLConfigInt(key, value, 1)
			if err != nil {
				return config, err
			}
			config.SaveTotalLimit = int(v)
		case "seed":
			v, err := evalRLConfigInt(key, value, -1<<63)
			if err != nil {
				return config, err
			}
			config.Seed = v
		default:
			return config, cliError{"invalid_grpo_config", "Unknown GRPO configuration key.", []string{"Only the documented conservative GRPO settings may be overridden."}, map[string]any{"key": key}}
		}
	}
	if err := validateEvalRLGRPOConfig(config, false); err != nil {
		return config, err
	}
	return config, nil
}

func evalRLConfigInt(key string, value any, minimum int64) (int64, error) {
	number, ok := value.(json.Number)
	if !ok {
		return 0, evalRLConfigError(key, "an integer", value)
	}
	parsed, err := strconv.ParseInt(number.String(), 10, 64)
	if err != nil || parsed < minimum {
		expectation := "an integer"
		if minimum != -1<<63 {
			expectation = fmt.Sprintf("an integer >= %d", minimum)
		}
		return 0, evalRLConfigError(key, expectation, value)
	}
	if strconv.IntSize == 32 && key != "seed" && (parsed > 1<<31-1 || parsed < -1<<31) {
		return 0, evalRLConfigError(key, "an integer supported by this platform", value)
	}
	return parsed, nil
}

func evalRLConfigNumber(key string, value any) (float64, error) {
	number, ok := value.(json.Number)
	if !ok {
		return 0, evalRLConfigError(key, "a number > 0", value)
	}
	parsed, err := strconv.ParseFloat(number.String(), 64)
	if err != nil || parsed <= 0 || math.IsInf(parsed, 0) || math.IsNaN(parsed) {
		return 0, evalRLConfigError(key, "a finite number > 0", value)
	}
	return parsed, nil
}

func evalRLConfigError(key, expectation string, value any) error {
	return cliError{"invalid_grpo_config", fmt.Sprintf("GRPO configuration %q must be %s.", key, expectation), nil, map[string]any{"key": key, "value": value}}
}

func canonicalExistingDirectory(input, code string) (string, error) {
	abs, err := filepath.Abs(input)
	if err != nil {
		return "", cliError{code, "Could not resolve the directory path.", nil, map[string]any{"path": input, "error": err.Error()}}
	}
	resolved, err := filepath.EvalSymlinks(abs)
	if err != nil {
		return "", cliError{code, "Could not resolve the directory path.", nil, map[string]any{"path": input, "error": err.Error()}}
	}
	info, err := os.Stat(resolved)
	if err != nil || !info.IsDir() {
		return "", cliError{code, "The path must be an existing directory.", nil, map[string]any{"path": input}}
	}
	return filepath.Clean(resolved), nil
}

func canonicalExistingFile(input, code string) (string, error) {
	abs, err := filepath.Abs(input)
	if err != nil {
		return "", cliError{code, "Could not resolve the file path.", nil, map[string]any{"path": input, "error": err.Error()}}
	}
	resolved, err := filepath.EvalSymlinks(abs)
	if err != nil {
		return "", cliError{code, "Could not resolve the file path.", nil, map[string]any{"path": input, "error": err.Error()}}
	}
	info, err := os.Stat(resolved)
	if err != nil || !info.Mode().IsRegular() {
		return "", cliError{code, "The path must be an existing regular file.", nil, map[string]any{"path": input}}
	}
	return filepath.Clean(resolved), nil
}

func canonicalPathForWrite(input string) (string, error) {
	abs, err := filepath.Abs(input)
	if err != nil {
		return "", err
	}
	abs = filepath.Clean(abs)
	cursor := abs
	tail := []string{}
	for {
		_, statErr := os.Lstat(cursor)
		if statErr == nil {
			resolved, resolveErr := filepath.EvalSymlinks(cursor)
			if resolveErr != nil {
				return "", resolveErr
			}
			for i := len(tail) - 1; i >= 0; i-- {
				resolved = filepath.Join(resolved, tail[i])
			}
			return filepath.Clean(resolved), nil
		}
		if !os.IsNotExist(statErr) {
			return "", statErr
		}
		parent := filepath.Dir(cursor)
		if parent == cursor {
			return "", statErr
		}
		tail = append(tail, filepath.Base(cursor))
		cursor = parent
	}
}

func pathWithin(root, candidate string) bool {
	rel, err := filepath.Rel(filepath.Clean(root), filepath.Clean(candidate))
	if err != nil {
		return false
	}
	return rel == "." || (rel != ".." && !strings.HasPrefix(rel, ".."+string(os.PathSeparator)))
}

func requireEmptyOrMissingDirectory(path string) error {
	info, err := os.Stat(path)
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		return cliError{"rl_output_invalid", "Could not inspect the RL output path.", nil, map[string]any{"path": path, "error": err.Error()}}
	}
	if !info.IsDir() {
		return cliError{"rl_output_not_empty", "The RL output path exists and is not an empty directory.", nil, map[string]any{"path": path}}
	}
	entries, err := os.ReadDir(path)
	if err != nil {
		return cliError{"rl_output_invalid", "Could not inspect the RL output directory.", nil, map[string]any{"path": path, "error": err.Error()}}
	}
	if len(entries) != 0 {
		return cliError{"rl_output_not_empty", "The RL output directory must be empty.", []string{"Choose a new --out directory or remove the existing contents intentionally."}, map[string]any{"path": path}}
	}
	return nil
}

func loadEvalRLBundles(root string) ([]evalRLBundle, error) {
	taskPath := filepath.Join(root, "task.json")
	if info, err := os.Lstat(taskPath); err == nil {
		if info.Mode()&os.ModeSymlink != 0 || !info.Mode().IsRegular() {
			return nil, cliError{"rl_bundle_unsafe", "task.json must be a regular file, not a symlink or special file.", nil, map[string]any{"path": taskPath}}
		}
		bundle, err := loadEvalRLBundle(root, ".", root)
		if err != nil {
			return nil, err
		}
		return []evalRLBundle{bundle}, nil
	}
	if _, err := os.Lstat(filepath.Join(root, "summary.json")); err == nil {
		return nil, cliError{"rl_source_completed_run", "Online GRPO preparation requires imported terminal task bundles, not a completed result or checkpoint directory.", []string{"Point to the output of eval terminal import, before any tasks have been run."}, map[string]any{"source": root}}
	}
	entries, err := os.ReadDir(root)
	if err != nil {
		return nil, cliError{"rl_source_invalid", "Could not read the terminal bundle parent directory.", nil, map[string]any{"source": root, "error": err.Error()}}
	}
	bundles := make([]evalRLBundle, 0)
	seenIDs := map[string]string{}
	seenRefs := map[string]bool{}
	for _, entry := range entries {
		if entry.Type()&os.ModeSymlink != 0 {
			return nil, cliError{"rl_bundle_unsafe", "Symlink entries are not allowed in a terminal bundle parent.", nil, map[string]any{"path": filepath.Join(root, entry.Name())}}
		}
		if !entry.IsDir() {
			continue
		}
		ref := filepath.ToSlash(entry.Name())
		if !safeEvalRLPathSegment(ref) {
			return nil, cliError{"rl_bundle_ref_invalid", "A terminal bundle directory has an unsafe bundle reference.", nil, map[string]any{"bundleRef": ref}}
		}
		if seenRefs[ref] {
			return nil, cliError{"rl_bundle_ref_duplicate", "Duplicate terminal bundle references are not allowed.", nil, map[string]any{"bundleRef": ref}}
		}
		seenRefs[ref] = true
		bundleDir := filepath.Join(root, entry.Name())
		if _, err := os.Lstat(filepath.Join(bundleDir, "task.json")); err != nil {
			return nil, cliError{"rl_bundle_missing_task", "A terminal bundle directory is missing task.json.", []string{"Point to one imported bundle or a parent containing only imported bundles."}, map[string]any{"bundleRef": ref, "path": bundleDir}}
		}
		bundle, err := loadEvalRLBundle(bundleDir, ref, root)
		if err != nil {
			return nil, err
		}
		if previous, duplicate := seenIDs[bundle.TaskID]; duplicate {
			return nil, cliError{"rl_task_id_duplicate", "Duplicate terminal task ids are not allowed.", nil, map[string]any{"taskId": bundle.TaskID, "bundleRef": ref, "previousBundleRef": previous}}
		}
		seenIDs[bundle.TaskID] = ref
		bundles = append(bundles, bundle)
	}
	if len(bundles) == 0 {
		return nil, cliError{"rl_source_invalid", "No imported terminal task bundles were found.", []string{"Point to a bundle containing task.json and tests/, or a parent directory of such bundles."}, map[string]any{"source": root}}
	}
	sort.Slice(bundles, func(i, j int) bool {
		if bundles[i].TaskID == bundles[j].TaskID { return bundles[i].Ref < bundles[j].Ref }
		return bundles[i].TaskID < bundles[j].TaskID
	})
	return bundles, nil
}

func loadEvalRLBundle(bundleDir, ref, root string) (evalRLBundle, error) {
	if !pathWithin(root, bundleDir) {
		return evalRLBundle{}, cliError{"rl_bundle_unsafe", "A terminal bundle resolves outside the source root.", nil, map[string]any{"bundleRef": ref, "path": bundleDir}}
	}
	if err := rejectEvalRLSymlinks(bundleDir); err != nil {
		return evalRLBundle{}, err
	}
	taskPath := filepath.Join(bundleDir, "task.json")
	taskInfo, err := os.Stat(taskPath)
	if err != nil || !taskInfo.Mode().IsRegular() {
		return evalRLBundle{}, cliError{"rl_bundle_missing_task", "Terminal bundle is missing a regular task.json file.", nil, map[string]any{"bundleRef": ref, "path": taskPath}}
	}
	testsPath := filepath.Join(bundleDir, "tests")
	testsInfo, err := os.Stat(testsPath)
	if err != nil || !testsInfo.IsDir() {
		return evalRLBundle{}, cliError{"rl_bundle_missing_tests", "Terminal bundle is missing its tests directory.", nil, map[string]any{"bundleRef": ref, "path": testsPath}}
	}
	data, err := os.ReadFile(taskPath)
	if err != nil {
		return evalRLBundle{}, cliError{"rl_bundle_invalid", "Could not read terminal bundle task.json.", nil, map[string]any{"bundleRef": ref, "error": err.Error()}}
	}
	var task struct {
		ID          string `json:"id"`
		Instruction string `json:"instruction"`
	}
	if err := json.Unmarshal(data, &task); err != nil {
		return evalRLBundle{}, cliError{"rl_bundle_invalid", "Could not decode terminal bundle task.json.", nil, map[string]any{"bundleRef": ref, "error": err.Error()}}
	}
	if !safeEvalRLPathSegment(task.ID) {
		return evalRLBundle{}, cliError{"rl_task_id_invalid", "Terminal task id is empty or unsafe.", []string{"Task ids must be single safe path segments without separators or control characters."}, map[string]any{"bundleRef": ref, "taskId": task.ID}}
	}
	if strings.TrimSpace(task.Instruction) == "" {
		return evalRLBundle{}, cliError{"rl_bundle_invalid", "Terminal task instruction must not be empty.", nil, map[string]any{"bundleRef": ref, "taskId": task.ID}}
	}
	return evalRLBundle{TaskID: task.ID, Instruction: task.Instruction, Ref: ref, Dir: bundleDir}, nil
}

func rejectEvalRLSymlinks(root string) error {
	return filepath.WalkDir(root, func(path string, entry os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return cliError{"rl_bundle_invalid", "Could not inspect terminal bundle contents.", nil, map[string]any{"path": path, "error": walkErr.Error()}}
		}
		if entry.Type()&os.ModeSymlink != 0 {
			return cliError{"rl_bundle_unsafe", "Symlinks are not allowed inside terminal task bundles.", []string{"Re-import the bundle with real files and directories."}, map[string]any{"path": path}}
		}
		return nil
	})
}

func safeEvalRLPathSegment(value string) bool {
	if value == "" || value != strings.TrimSpace(value) || value == "." || value == ".." || filepath.Base(value) != value || strings.ContainsAny(value, `/\\`) {
		return false
	}
	for _, r := range value {
		if r == 0 || r < 0x20 || r == 0x7f {
			return false
		}
	}
	return true
}

func writeEvalRLPromptRowsAtomic(path string, rows []evalRLPromptRow) error {
	tmp, err := os.CreateTemp(filepath.Dir(path), ".lmx-rl-prompts-*.tmp")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()
	defer os.Remove(tmpPath)
	encoder := json.NewEncoder(tmp)
	encoder.SetEscapeHTML(false)
	for _, row := range rows {
		if err := encoder.Encode(row); err != nil {
			tmp.Close()
			return err
		}
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmpPath, path)
}

func resolveEvalRLResumeSelector(raw string) (string, error) {
	selector := strings.TrimSpace(raw)
	if selector == "" {
		return "auto", nil
	}
	if selector == "auto" || selector == "none" {
		return selector, nil
	}
	checkpoint, err := canonicalExistingDirectory(selector, "rl_resume_invalid")
	if err != nil {
		return "", err
	}
	state := filepath.Join(checkpoint, "trainer_state.json")
	info, err := os.Stat(state)
	if err != nil || !info.Mode().IsRegular() {
		return "", cliError{"rl_resume_invalid", "An explicit resume checkpoint must contain trainer_state.json.", nil, map[string]any{"checkpoint": checkpoint, "path": state}}
	}
	return checkpoint, nil
}

func requireJSONEOF(decoder *json.Decoder) error {
	var extra any
	err := decoder.Decode(&extra)
	if errors.Is(err, io.EOF) {
		return nil
	}
	if err == nil {
		return errors.New("unexpected trailing JSON value")
	}
	return err
}

func (config *evalRLGRPOConfig) UnmarshalJSON(data []byte) error {
	type plainConfig evalRLGRPOConfig
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	var decoded plainConfig
	if err := decoder.Decode(&decoded); err != nil {
		return err
	}
	if err := requireJSONEOF(decoder); err != nil {
		return err
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	*config = evalRLGRPOConfig(decoded)
	config.present = make(map[string]bool, len(raw))
	for key := range raw {
		config.present[key] = true
	}
	return nil
}

func readEvalRLManifest(path string) (evalRLManifest, error) {
	file, err := os.Open(path)
	if err != nil {
		return evalRLManifest{}, cliError{"rl_manifest_invalid", "Could not open the RL manifest.", nil, map[string]any{"path": path, "error": err.Error()}}
	}
	defer file.Close()
	decoder := json.NewDecoder(file)
	decoder.DisallowUnknownFields()
	var manifest evalRLManifest
	if err := decoder.Decode(&manifest); err != nil {
		return evalRLManifest{}, cliError{"rl_manifest_invalid", "Could not decode the strict RL manifest.", []string{"Use a manifest produced by eval train rl prepare, not an SFT training manifest."}, map[string]any{"path": path, "error": err.Error()}}
	}
	if err := requireJSONEOF(decoder); err != nil {
		return evalRLManifest{}, cliError{"rl_manifest_invalid", "The RL manifest contains trailing JSON content.", nil, map[string]any{"path": path, "error": err.Error()}}
	}
	return manifest, nil
}

func validateEvalRLManifest(manifestPath string, manifest evalRLManifest) error {
	invalid := func(message string, details any) error {
		return cliError{"rl_manifest_invalid", message, []string{"Use a manifest produced by eval train rl prepare."}, details}
	}
	if manifest.SchemaVersion != trainingSchemaVersion {
		return invalid("Unsupported RL manifest schemaVersion.", map[string]any{"schemaVersion": manifest.SchemaVersion})
	}
	if manifest.Kind != evalRLManifestKind {
		return invalid("The manifest is not an online GRPO RL manifest.", map[string]any{"kind": manifest.Kind})
	}
	if manifest.Algorithm != evalRLAlgorithm {
		return invalid("The RL manifest algorithm must be online_grpo.", map[string]any{"algorithm": manifest.Algorithm})
	}
	if strings.TrimSpace(manifest.BaseModel) == "" {
		return invalid("The RL manifest baseModel must not be empty.", nil)
	}
	if manifest.Source.TaskCount < 1 {
		return invalid("The RL manifest source.taskCount must be positive.", map[string]any{"taskCount": manifest.Source.TaskCount})
	}
	if !filepath.IsAbs(manifest.Source.BundleRoot) || filepath.Clean(manifest.Source.BundleRoot) != manifest.Source.BundleRoot {
		return invalid("The RL manifest source.bundleRoot must be a clean absolute path.", map[string]any{"bundleRoot": manifest.Source.BundleRoot})
	}
	canonicalRoot, err := canonicalExistingDirectory(manifest.Source.BundleRoot, "rl_manifest_invalid")
	if err != nil || canonicalRoot != manifest.Source.BundleRoot {
		details := map[string]any{"bundleRoot": manifest.Source.BundleRoot}
		if err != nil { details["error"] = err.Error() } else { details["canonical"] = canonicalRoot }
		return invalid("The RL manifest source.bundleRoot must resolve to its canonical existing directory.", details)
	}
	manifestDir := filepath.Dir(manifestPath)
	expectedDataset := filepath.Join(manifestDir, "prompts.jsonl")
	if manifest.Dataset.Format != evalRLDatasetFormat {
		return invalid("The RL manifest dataset format is unsupported.", map[string]any{"format": manifest.Dataset.Format})
	}
	if manifest.Dataset.Path != expectedDataset || !filepath.IsAbs(manifest.Dataset.Path) {
		return invalid("The RL dataset path must be the absolute prompts.jsonl beside its manifest.", map[string]any{"path": manifest.Dataset.Path, "expected": expectedDataset})
	}
	canonicalDataset, err := canonicalExistingFile(manifest.Dataset.Path, "rl_dataset_invalid")
	if err != nil {
		return err
	}
	if canonicalDataset != manifest.Dataset.Path {
		return invalid("The RL dataset path must be canonical and must not traverse symlinks.", map[string]any{"path": manifest.Dataset.Path, "canonical": canonicalDataset})
	}
	if manifest.Dataset.Examples < 1 {
		return invalid("The RL manifest dataset.examples must be positive.", map[string]any{"examples": manifest.Dataset.Examples})
	}
	if !equalStringSlice(manifest.Dataset.Columns, []string{"prompt", "task_id", "bundle_ref"}) {
		return invalid("The RL dataset columns must be exactly prompt, task_id, and bundle_ref in that order.", map[string]any{"columns": manifest.Dataset.Columns})
	}
	if manifest.Environment.ContractVersion != evalRLEnvironmentSchema {
		return invalid("Unsupported RL environment contractVersion.", map[string]any{"contractVersion": manifest.Environment.ContractVersion})
	}
	if !evalRLFactoryPattern.MatchString(manifest.Environment.Factory) {
		return invalid("The RL environment factory is not a valid dotted module:callable reference.", map[string]any{"factory": manifest.Environment.Factory})
	}
	if manifest.Environment.Config == nil {
		return invalid("The RL environment config must be a JSON object.", nil)
	}
	if manifest.Trainer.Implementation != evalRLTrainerImpl || manifest.Trainer.TRLVersion != evalRLTRLVersion {
		return invalid("The RL trainer implementation or pinned TRL version is unsupported.", map[string]any{"implementation": manifest.Trainer.Implementation, "trlVersion": manifest.Trainer.TRLVersion})
	}
	if err := validateEvalRLGRPOConfig(manifest.Trainer.GRPOConfig, true); err != nil {
		return err
	}
	expectedOutput := filepath.Join(manifestDir, "grpo-output")
	if !filepath.IsAbs(manifest.Trainer.OutputDir) || filepath.Clean(manifest.Trainer.OutputDir) != manifest.Trainer.OutputDir || manifest.Trainer.OutputDir != expectedOutput {
		return invalid("The prepared trainer.outputDir must be the absolute grpo-output directory beside the manifest.", map[string]any{"outputDir": manifest.Trainer.OutputDir, "expected": expectedOutput})
	}
	canonicalOutput, err := canonicalPathForWrite(manifest.Trainer.OutputDir)
	if err != nil || canonicalOutput != manifest.Trainer.OutputDir {
		details := map[string]any{"outputDir": manifest.Trainer.OutputDir}
		if err != nil { details["error"] = err.Error() } else { details["canonical"] = canonicalOutput }
		return invalid("The trainer output path must be canonical and free of symlink redirection.", details)
	}
	if pathWithin(manifest.Source.BundleRoot, manifest.Trainer.OutputDir) {
		return invalid("The trainer output directory must not be contained by source.bundleRoot.", map[string]any{"bundleRoot": manifest.Source.BundleRoot, "outputDir": manifest.Trainer.OutputDir})
	}
	if !manifest.Contamination.BenchmarkDerived || !manifest.Contamination.Acknowledged || manifest.Contamination.Warning != evalRLContaminationNote {
		return invalid("The RL manifest must preserve the benchmark contamination acknowledgement.", nil)
	}
	rows, err := readAndValidateEvalRLPromptRows(manifest.Dataset.Path, manifest.Source.BundleRoot)
	if err != nil {
		return err
	}
	if len(rows) != manifest.Dataset.Examples || len(rows) != manifest.Source.TaskCount {
		return invalid("The RL manifest counts do not match the prompt dataset.", map[string]any{"rows": len(rows), "examples": manifest.Dataset.Examples, "taskCount": manifest.Source.TaskCount})
	}
	return nil
}

func validateEvalRLGRPOConfig(config evalRLGRPOConfig, requireAll bool) error {
	if requireAll {
		required := []string{"num_generations", "max_steps", "learning_rate", "per_device_train_batch_size", "gradient_accumulation_steps", "max_completion_length", "max_tool_calling_iterations", "gradient_checkpointing", "logging_steps", "save_steps", "save_total_limit", "seed"}
		for _, key := range required {
			if !config.present[key] {
				return cliError{"rl_manifest_invalid", "The RL manifest grpoConfig is missing a required key.", nil, map[string]any{"key": key}}
			}
		}
		if len(config.present) != len(required) {
			return cliError{"rl_manifest_invalid", "The RL manifest grpoConfig must contain exactly the supported keys.", nil, map[string]any{"keys": len(config.present)}}
		}
	}
	checks := []struct { key string; value int; minimum int }{
		{"num_generations", config.NumGenerations, 2},
		{"max_steps", config.MaxSteps, 1},
		{"per_device_train_batch_size", config.PerDeviceTrainBatchSize, 1},
		{"gradient_accumulation_steps", config.GradientAccumulationSteps, 1},
		{"max_completion_length", config.MaxCompletionLength, 1},
		{"max_tool_calling_iterations", config.MaxToolCallingIterations, 1},
		{"logging_steps", config.LoggingSteps, 1},
		{"save_steps", config.SaveSteps, 1},
		{"save_total_limit", config.SaveTotalLimit, 1},
	}
	for _, check := range checks {
		if check.value < check.minimum {
			return cliError{"invalid_grpo_config", fmt.Sprintf("GRPO configuration %q must be an integer >= %d.", check.key, check.minimum), nil, map[string]any{"key": check.key, "value": check.value}}
		}
	}
	if config.LearningRate <= 0 || math.IsNaN(config.LearningRate) || math.IsInf(config.LearningRate, 0) {
		return cliError{"invalid_grpo_config", "GRPO configuration learning_rate must be a finite number > 0.", nil, map[string]any{"value": config.LearningRate}}
	}
	return nil
}

func readAndValidateEvalRLPromptRows(path, bundleRoot string) ([]evalRLPromptRow, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, cliError{"rl_dataset_invalid", "Could not open the RL prompt dataset.", nil, map[string]any{"path": path, "error": err.Error()}}
	}
	defer file.Close()
	reader := bufio.NewReader(file)
	rows := []evalRLPromptRow{}
	seenIDs := map[string]bool{}
	seenRefs := map[string]bool{}
	previousID := ""
	lineNumber := 0
	for {
		line, readErr := reader.ReadBytes('\n')
		if len(line) > 0 {
			lineNumber++
			if len(bytes.TrimSpace(line)) == 0 {
				return nil, cliError{"rl_dataset_invalid", "The RL prompt dataset contains an empty line.", nil, map[string]any{"path": path, "line": lineNumber}}
			}
			decoder := json.NewDecoder(bytes.NewReader(line))
			decoder.DisallowUnknownFields()
			var row evalRLPromptRow
			if err := decoder.Decode(&row); err != nil {
				return nil, cliError{"rl_dataset_invalid", "Could not decode a strict RL prompt row.", nil, map[string]any{"path": path, "line": lineNumber, "error": err.Error()}}
			}
			if err := requireJSONEOF(decoder); err != nil {
				return nil, cliError{"rl_dataset_invalid", "An RL prompt row contains trailing JSON content.", nil, map[string]any{"path": path, "line": lineNumber}}
			}
			if len(row.Prompt) != 1 || row.Prompt[0].Role != "user" || strings.TrimSpace(row.Prompt[0].Content) == "" {
				return nil, cliError{"rl_dataset_invalid", "Each RL row prompt must contain exactly one nonempty user message.", nil, map[string]any{"path": path, "line": lineNumber}}
			}
			if !safeEvalRLPathSegment(row.TaskID) || seenIDs[row.TaskID] {
				return nil, cliError{"rl_dataset_invalid", "RL prompt task_id values must be safe and unique.", nil, map[string]any{"path": path, "line": lineNumber, "taskId": row.TaskID}}
			}
			if row.BundleRef != "." && !safeEvalRLPathSegment(row.BundleRef) {
				return nil, cliError{"rl_dataset_invalid", "RL prompt bundle_ref values must be safe relative bundle references.", nil, map[string]any{"path": path, "line": lineNumber, "bundleRef": row.BundleRef}}
			}
			if seenRefs[row.BundleRef] {
				return nil, cliError{"rl_dataset_invalid", "RL prompt bundle_ref values must be unique.", nil, map[string]any{"path": path, "line": lineNumber, "bundleRef": row.BundleRef}}
			}
			if previousID != "" && row.TaskID <= previousID {
				return nil, cliError{"rl_dataset_invalid", "RL prompt rows must be sorted by task_id.", nil, map[string]any{"path": path, "line": lineNumber, "taskId": row.TaskID}}
			}
			bundleDir := bundleRoot
			if row.BundleRef != "." {
				bundleDir = filepath.Join(bundleRoot, filepath.FromSlash(row.BundleRef))
			}
			bundle, err := loadEvalRLBundle(bundleDir, row.BundleRef, bundleRoot)
			if err != nil {
				return nil, err
			}
			if bundle.TaskID != row.TaskID || bundle.Instruction != row.Prompt[0].Content {
				return nil, cliError{"rl_dataset_invalid", "An RL prompt row does not match its referenced imported terminal task.", nil, map[string]any{"path": path, "line": lineNumber, "taskId": row.TaskID, "bundleRef": row.BundleRef}}
			}
			seenIDs[row.TaskID] = true
			seenRefs[row.BundleRef] = true
			previousID = row.TaskID
			rows = append(rows, row)
		}
		if errors.Is(readErr, io.EOF) {
			break
		}
		if readErr != nil {
			return nil, cliError{"rl_dataset_invalid", "Could not read the RL prompt dataset.", nil, map[string]any{"path": path, "line": lineNumber, "error": readErr.Error()}}
		}
	}
	if len(rows) == 0 {
		return nil, cliError{"rl_dataset_invalid", "The RL prompt dataset must contain at least one row.", nil, map[string]any{"path": path}}
	}
	return rows, nil
}

func equalStringSlice(left, right []string) bool {
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

func loadEvalTrainingRecords(source string) ([]evalTrainingRecord, string, error) {
	info, err := os.Stat(source)
	if err != nil {
		return nil, "", cliError{"file_read_error", fmt.Sprintf("Could not inspect %s: %v", source, err), nil, nil}
	}
	root := ""
	input := source
	if info.IsDir() {
		root = source
		input = filepath.Join(source, "summary.json")
	}
	value, err := readJSON(input)
	if err != nil {
		return nil, "", err
	}
	if entries, ok := value.([]any); ok {
		if root == "" {
			root = filepath.Dir(input)
		}
		return loadCheckpointTrainingRecords(entries, root)
	}
	obj, ok := value.(map[string]any)
	if !ok {
		return nil, "", cliError{"training_source_invalid", "Training source must be a completed-run summary array or canonical terminal result object.", nil, map[string]any{"path": input}}
	}
	results := anySlice(obj["results"])
	if len(results) == 0 {
		return nil, "", cliError{"training_source_invalid", "Canonical terminal result object has no results.", nil, map[string]any{"path": input}}
	}
	if root == "" {
		root = filepath.Dir(input)
	}
	return normalizeCanonicalTrainingResults(results, input), root, nil
}

func loadCheckpointTrainingRecords(entries []any, root string) ([]evalTrainingRecord, string, error) {
	records := make([]evalTrainingRecord, 0, len(entries))
	seen := map[string]bool{}
	for _, raw := range entries {
		entry := asObject(raw)
		if entry == nil {
			continue
		}
		out := stringValue(entry["out"])
		if out == "" {
			continue
		}
		if !filepath.IsAbs(out) {
			out = filepath.Join(root, out)
		}
		value, err := readJSON(out)
		if err != nil {
			return nil, "", err
		}
		obj, ok := value.(map[string]any)
		if !ok {
			return nil, "", cliError{"training_source_invalid", "Checkpoint output is not a canonical terminal result object.", nil, map[string]any{"path": out}}
		}
		for _, record := range normalizeCanonicalTrainingResults(anySlice(obj["results"]), out) {
			if record.ID == "" || seen[record.ID] {
				continue
			}
			seen[record.ID] = true
			records = append(records, record)
		}
	}
	if len(records) == 0 {
		return nil, "", cliError{"training_source_invalid", "Completed-run checkpoint did not reference any canonical terminal results.", nil, map[string]any{"root": root}}
	}
	sort.Slice(records, func(i, j int) bool { return records[i].ID < records[j].ID })
	return records, root, nil
}

func normalizeCanonicalTrainingResults(results []any, resultPath string) []evalTrainingRecord {
	records := make([]evalTrainingRecord, 0, len(results))
	for _, raw := range results {
		row := asObject(raw)
		if row == nil {
			continue
		}
		records = append(records, evalTrainingRecord{
			ID:             firstNonEmpty(stringValue(row["question_id"]), stringValue(row["questionId"]), stringValue(row["id"])),
			Pass:           boolField(row, "pass"),
			Scored:         boolField(row, "scored"),
			Question:       stringValue(row["question"]),
			VerifierOutput: stringValue(row["verifierOutput"]),
			Error:          stringValue(row["error"]),
			ErrorCode:      stringValue(row["errorCode"]),
			WallTimeMs:     int64(numberField(row, "wallTimeMs")),
			TokenUsage:     row["tokenUsage"],
			ResultPath:     resultPath,
		})
	}
	return records
}

func boolField(obj map[string]any, key string) bool {
	value, _ := obj[key].(bool)
	return value
}

func findOMPTrace(root, taskID string) string {
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

func readOMPTrainingMessages(path string, maxMessageBytes int) ([]map[string]any, int, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, 0, err
	}
	defer file.Close()
	reader := bufio.NewReaderSize(file, 128*1024)
	messages := make([]map[string]any, 0)
	truncated := 0
	for {
		line, readErr := reader.ReadBytes('\n')
		if len(line) > 0 {
			var event map[string]any
			if json.Unmarshal(line, &event) == nil && stringValue(event["type"]) == "message_end" {
				message := asObject(event["message"])
				if normalized := normalizeOMPTrainingMessage(message, maxMessageBytes, &truncated); normalized != nil {
					messages = append(messages, normalized)
				}
			}
		}
		if errors.Is(readErr, io.EOF) {
			break
		}
		if readErr != nil {
			return nil, truncated, readErr
		}
	}
	if len(messages) == 0 {
		return nil, truncated, errors.New("no finalized OMP messages")
	}
	return messages, truncated, nil
}

func normalizeOMPTrainingMessage(message map[string]any, maxMessageBytes int, truncated *int) map[string]any {
	if message == nil {
		return nil
	}
	role := stringValue(message["role"])
	contentItems := anySlice(message["content"])
	switch role {
	case "user":
		text := ompTextContent(contentItems, false)
		text, wasTruncated := truncateTrainingText(text, maxMessageBytes)
		if wasTruncated {
			*truncated++
		}
		if strings.TrimSpace(text) == "" {
			return nil
		}
		return map[string]any{"role": "user", "content": text}
	case "assistant":
		text := ompTextContent(contentItems, false)
		text, wasTruncated := truncateTrainingText(text, maxMessageBytes)
		if wasTruncated {
			*truncated++
		}
		toolCalls := make([]map[string]any, 0)
		for _, raw := range contentItems {
			item := asObject(raw)
			if item == nil || stringValue(item["type"]) != "toolCall" {
				continue
			}
			arguments := item["arguments"]
			if arguments == nil {
				arguments = map[string]any{}
			}
			toolCalls = append(toolCalls, map[string]any{
				"id":   stringValue(item["id"]),
				"type": "function",
				"function": map[string]any{
					"name":      stringValue(item["name"]),
					"arguments": arguments,
				},
			})
		}
		if strings.TrimSpace(text) == "" && len(toolCalls) == 0 {
			return nil
		}
		result := map[string]any{"role": "assistant", "content": text}
		if len(toolCalls) > 0 {
			result["tool_calls"] = toolCalls
		}
		return result
	case "toolResult":
		text := ompTextContent(contentItems, false)
		text, wasTruncated := truncateTrainingText(text, maxMessageBytes)
		if wasTruncated {
			*truncated++
		}
		if text == "" {
			text = "(no output)"
		}
		return map[string]any{
			"role":         "tool",
			"content":      text,
			"tool_call_id": stringValue(message["toolCallId"]),
			"name":         stringValue(message["toolName"]),
		}
	default:
		return nil
	}
}

func ompTextContent(items []any, includeThinking bool) string {
	parts := make([]string, 0)
	for _, raw := range items {
		item := asObject(raw)
		if item == nil {
			continue
		}
		kind := stringValue(item["type"])
		if text := stringValue(item["text"]); text != "" {
			parts = append(parts, text)
		}
		if includeThinking && kind == "thinking" {
			if thinking := stringValue(item["thinking"]); thinking != "" {
				parts = append(parts, thinking)
			}
		}
	}
	return strings.Join(parts, "\n")
}

func validTrainingConversation(messages []map[string]any) bool {
	hasUser := false
	hasAssistant := false
	for _, message := range messages {
		switch stringValue(message["role"]) {
		case "user":
			hasUser = true
		case "assistant":
			hasAssistant = true
		}
	}
	return hasUser && hasAssistant
}

func truncateTrainingText(value string, maxBytes int) (string, bool) {
	if len(value) <= maxBytes {
		return value, false
	}
	keep := maxBytes - len(trainingTruncationMarker)
	if keep < 0 {
		keep = 0
	}
	for keep > 0 && !utf8.ValidString(value[:keep]) {
		keep--
	}
	return value[:keep] + trainingTruncationMarker, true
}

func writeJSONLAtomic(path string, rows []map[string]any) error {
	dir := filepath.Dir(path)
	tmp, err := os.CreateTemp(dir, ".lmx-training-*.tmp")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()
	defer os.Remove(tmpPath)
	encoder := json.NewEncoder(tmp)
	encoder.SetEscapeHTML(false)
	for _, row := range rows {
		if err := encoder.Encode(row); err != nil {
			tmp.Close()
			return err
		}
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmpPath, path)
}

func writeJSONAtomic(path string, value any) error {
	data, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	tmp, err := os.CreateTemp(filepath.Dir(path), ".lmx-manifest-*.tmp")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()
	defer os.Remove(tmpPath)
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmpPath, path)
}
