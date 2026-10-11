package main

// lmx speed-test matrix integrates local-inference-lab/llm-inference-bench
// (https://github.com/local-inference-lab/llm-inference-bench, MIT licensed,
// by Local Inference Lab). lmx runs the upstream llm_decode_bench.py unmodified
// against a vLLM or SGLang endpoint, or imports a report it (or lil-bench)
// already produced, and uploads the prefill and context × concurrency decode
// matrix to POST /api/inference-benchmarks.

import (
	"bytes"
	"compress/gzip"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
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

const (
	llmInferenceBenchRepository = "https://github.com/local-inference-lab/llm-inference-bench"
	// Pinned upstream release, checksum-verified; --bench-script runs any other copy.
	llmInferenceBenchDefaultRef    = "v0.7.7"
	llmInferenceBenchDefaultSHA256 = "12ff23289c8d8d4741a261ae2fb95aa3bb9b83e85732d8665011a0d0617f1dbe"
	inferenceMatrixKind            = "inference_matrix"
	inferenceMatrixEndpoint        = "/api/inference-benchmarks"
)

// Upstream engines the site accepts. The bench reports remote OpenAI-compatible
// APIs as "openai_proxy"; those are not local inference and are rejected.
var inferenceMatrixEngines = map[string]bool{"vllm": true, "sglang": true}

// Per-request payloads that can make a report megabytes large; the site does not store them.
var inferenceMatrixHeavyCellKeys = []string{"request_samples", "loop_diagnostics"}

var secretArgPattern = regexp.MustCompile(`(?i)^--(?:[\w-]*-)?(?:key|token|secret|password)$`)

// Same flag names inside a shell command string, e.g. a user-supplied --server-command.
var secretCommandPattern = regexp.MustCompile(`(?i)(--(?:[\w-]*-)?(?:key|token|secret|password))(=|\s+)('[^']*'|"[^"]*"|[^\s-]\S*)`)

func handleBenchmarkMatrix(action, target string, args cliArgs) error {
	switch action {
	case "run":
		return runInferenceMatrix(args)
	case "import":
		return importInferenceMatrix(target, args)
	case "dry-run", "validate":
		return sendInferenceMatrixFile(target, true, args)
	case "submit":
		return sendInferenceMatrixFile(target, false, args)
	case "submissions":
		return handleInferenceMatrixSubmissions(target, positional(args, 4), args)
	}
	return cliError{"unknown_subcommand", "Unknown speed-test matrix command: " + action, []string{"Use lmx speed-test matrix run, import, dry-run, submit, or submissions."}, map[string]any{"subcommand": action}}
}

// Metadata the site lets owners edit (lib/inferenceBenchmarkEdit.ts). Measurements
// are not editable; delete the submission and re-import instead.
var inferenceMatrixEditFields = map[string]bool{
	"hfId": true, "modelRevision": true, "quantization": true, "hardware": true,
	"engineVersion": true, "engineRepository": true, "engineBuild": true,
	"engineCommit": true, "backend": true, "commandSnippet": true,
}

func handleInferenceMatrixSubmissions(action, id string, args cliArgs) error {
	if action == "" {
		action = "list"
	}
	key := apiKey(args)
	if key == "" {
		return missingAPIKey("--api-key or LMX_API_KEY is required to manage your inference matrix submissions")
	}
	if action != "list" && action != "ls" && id == "" {
		return cliError{"missing_run_id", "Inference matrix submission ID is required.", []string{"Run lmx speed-test matrix submissions list, or read submission.id from the saved matrix file."}, nil}
	}
	endpoint := apiURL(args) + inferenceMatrixEndpoint
	switch action {
	case "list", "ls":
		limit, err := intOption(args, 20, 1, "limit")
		if err != nil {
			return err
		}
		offset, err := intOption(args, 0, 0, "offset")
		if err != nil {
			return err
		}
		query := url.Values{"mine": {"true"}, "limit": {strconv.Itoa(min(limit, 100))}, "offset": {strconv.Itoa(offset)}}
		value, err := fetchJSON("GET", endpoint+"?"+query.Encode(), key, nil)
		if err != nil {
			return err
		}
		return writeOrPrintJSON("inference_matrix_submissions", args, value)
	case "edit", "patch":
		patch, err := inferenceMatrixEditPatch(args)
		if err != nil {
			return err
		}
		target := endpoint + "/" + url.PathEscape(id)
		if hasFlag(args, "dry-run") {
			target += "?dryRun=true"
		}
		value, err := fetchJSON("PATCH", target, key, patch)
		if err != nil {
			return err
		}
		title := "inference_matrix_submission_edited"
		run := asObject(value)
		if hasFlag(args, "dry-run") {
			title = "inference_matrix_edit_dry_run_valid"
			run = asObject(run["run"])
		}
		if hasFlag(args, "json") || opt(args, "out") != "" {
			return writeOrPrintJSON(title, args, value)
		}
		engine := asObject(run["engine"])
		printInfo(args, title, map[string]any{
			"id":            run["id"],
			"hfId":          asObject(run["model"])["hfId"],
			"quantization":  engine["quantization"],
			"engineVersion": engine["engineVersion"],
			"hardware":      asObject(run["hardware"])["gpuName"],
			"lastEditedAt":  run["lastEditedAt"],
		})
		return nil
	case "delete", "rm", "remove":
		if !hasFlag(args, "yes") {
			return cliError{"confirmation_required", "Deleting an inference matrix submission is permanent and requires --yes.", []string{"lmx speed-test matrix submissions delete " + id + " --yes"}, nil}
		}
		if _, err := fetchJSON("DELETE", endpoint+"/"+url.PathEscape(id), key, nil); err != nil {
			return err
		}
		printInfo(args, "inference_matrix_submission_deleted", map[string]any{"id": id})
		return nil
	}
	return cliError{"unknown_subcommand", "Unknown speed-test matrix submissions command: " + action, []string{"Use list, edit, or delete."}, map[string]any{"subcommand": action}}
}

// inferenceMatrixEditPatch builds an edit from repeated --set field=value
// (values stay strings; "null" clears) and --hardware <file>.
func inferenceMatrixEditPatch(args cliArgs) (map[string]any, error) {
	if opt(args, "set-json") != "" || opt(args, "patch") != "" {
		return nil, cliError{"invalid_patch", "Matrix edits take --set field=value and --hardware hardware.json only.", []string{"lmx speed-test matrix submissions edit <id> --set quantization=FP8 --hardware hardware.json"}, nil}
	}
	patch := map[string]any{}
	for _, assignment := range args.setValues {
		field, raw, ok := strings.Cut(assignment, "=")
		field = strings.TrimSpace(field)
		if !ok || field == "" {
			return nil, cliError{"invalid_option", "--set requires field=value.", []string{"Repeat --set for multiple fields, e.g. --set quantization=FP8 --set engineVersion=0.11.0."}, nil}
		}
		if field == "hardware" {
			return nil, cliError{"invalid_patch", "Edit hardware with --hardware hardware.json.", nil, nil}
		}
		if !inferenceMatrixEditFields[field] {
			return nil, cliError{"invalid_patch", fmt.Sprintf("%q is not an editable inference matrix field.", field), []string{"Editable: hfId, modelRevision, quantization, engineVersion, engineRepository, engineBuild, engineCommit, backend, commandSnippet (--set), hardware (--hardware).", "Measurements cannot be edited; delete the submission and re-import the report."}, nil}
		}
		switch {
		case raw == "null":
			patch[field] = nil
		case field == "commandSnippet":
			patch[field] = secretCommandPattern.ReplaceAllString(raw, "$1$2<redacted>")
		default:
			patch[field] = raw
		}
	}
	if opt(args, "hardware") != "" {
		loaded, _, err := benchmarkHardware("remote", args)
		if err != nil {
			return nil, err
		}
		patch["hardware"] = normalizeHardwareForSubmit(asObject(loaded))
	}
	if len(patch) == 0 {
		return nil, cliError{"invalid_patch", "No edit given.", []string{"Pass --set field=value (repeatable) and/or --hardware hardware.json."}, nil}
	}
	return patch, nil
}

type inferenceMatrixInput struct {
	HfID             string
	ModelRevision    string
	Quantization     string
	EngineVersion    string
	EngineRepository string
	EngineBuild      string
	EngineCommit     string
	Backend          string
	CommandSnippet   string
	SubmittedWith    string
	Hardware         map[string]any
	Report           map[string]any
}

// buildInferenceMatrixEnvelope wraps an llm-inference-bench report in the
// POST /api/inference-benchmarks request body.
func buildInferenceMatrixEnvelope(in inferenceMatrixInput) (map[string]any, error) {
	fieldErrors := map[string][]string{}
	if strings.TrimSpace(in.HfID) == "" {
		fieldErrors["hfId"] = []string{"Model ID is required (--hf-id)"}
	}
	if strings.TrimSpace(in.Quantization) == "" {
		fieldErrors["quantization"] = []string{"Quantization is required (--quantization)"}
	}
	if in.Hardware == nil {
		fieldErrors["hardware"] = []string{"Server hardware is required (--hardware)"}
	}
	if len(in.CommandSnippet) > 4000 {
		fieldErrors["commandSnippet"] = []string{"Must be at most 4000 characters"}
	}
	metadata := asObject(in.Report["metadata"])
	if metadata == nil {
		return nil, cliError{"invalid_matrix_report", "The file is not an llm-inference-bench report: metadata is missing.", []string{"Pass the JSON written by llm_decode_bench.py --output, or a lil-bench result (.json.gz)."}, nil}
	}
	engine := stringValue(metadata["engine"])
	if !inferenceMatrixEngines[engine] {
		return nil, cliError{"unsupported_matrix_engine", fmt.Sprintf("llm-inference-bench measured engine %q; LocalMaxxing accepts vllm and sglang matrices.", engine), []string{"Run the benchmark directly against a local vLLM or SGLang server, not a hosted API or proxy."}, map[string]any{"engine": engine}}
	}

	report := map[string]any{"metadata": metadata}
	for _, key := range []string{"prefill", "methodology"} {
		if value, ok := in.Report[key]; ok && value != nil {
			report[key] = value
		}
	}
	cellCount := len(asObject(report["prefill"]))
	for _, key := range []string{"results", "burst_results"} {
		cells := []any{}
		for _, raw := range anySlice(in.Report[key]) {
			cell := asObject(raw)
			if cell == nil {
				continue
			}
			trimmed := make(map[string]any, len(cell))
			for field, value := range cell {
				trimmed[field] = value
			}
			for _, heavy := range inferenceMatrixHeavyCellKeys {
				delete(trimmed, heavy)
			}
			cells = append(cells, trimmed)
		}
		report[key] = cells
		cellCount += len(cells)
	}
	if cellCount == 0 {
		fieldErrors["benchmark"] = []string{"Report contains no prefill or decode cells"}
	}
	if len(fieldErrors) > 0 {
		return nil, cliError{"invalid_matrix_payload", "Inference matrix failed local validation.", []string{"Fix the reported fields and rerun."}, map[string]any{"fieldErrors": fieldErrors}}
	}

	envelope := map[string]any{
		"hfId":          in.HfID,
		"modelRevision": firstNonEmpty(in.ModelRevision, "main"),
		"hardware":      normalizeHardwareForSubmit(in.Hardware),
		"quantization":  in.Quantization,
		"benchmark":     report,
	}
	for key, value := range map[string]string{
		"engineVersion":    in.EngineVersion,
		"engineRepository": in.EngineRepository,
		"engineBuild":      in.EngineBuild,
		"engineCommit":     in.EngineCommit,
		"backend":          in.Backend,
		"commandSnippet":   secretCommandPattern.ReplaceAllString(in.CommandSnippet, "$1$2<redacted>"),
		"submittedWith":    in.SubmittedWith,
	} {
		if strings.TrimSpace(value) != "" {
			envelope[key] = value
		}
	}
	return envelope, nil
}

// matrixContext is what the measuring side knows besides the report itself.
// Explicit flags always override it.
type matrixContext struct {
	Hardware       map[string]any
	HardwareSource string
	Quantization   string
	EngineVersion  string
	CommandSnippet string
	ServedModel    string
	SubmittedWith  string
	Provenance     map[string]any
}

// runInferenceMatrix measures a local vLLM/SGLang server with llm-inference-bench
// and hands the report to the import path with what it detected.
func runInferenceMatrix(args cliArgs) error {
	baseURL := opt(args, "base-url")
	if baseURL == "" {
		return cliError{"missing_option", "speed-test matrix run requires --base-url", []string{"Pass the vLLM/SGLang server, e.g. --base-url http://127.0.0.1:8000."}, nil}
	}
	hfID := firstNonEmpty(opt(args, "hf-id"), opt(args, "model"))
	if hfID == "" {
		return cliError{"missing_model", "speed-test matrix run requires --hf-id", []string{"Pass the HuggingFace model ID that the server is serving."}, nil}
	}
	modelKey := opt(args, "model-api-key")
	servedModel, servedInfo, err := detectServedModel(baseURL, modelKey, opt(args, "served-model"))
	if err != nil {
		return cliError{"endpoint_unreachable", "Could not list models at " + baseURL + "/v1/models: " + err.Error(), []string{"Start the server first and check --base-url."}, nil}
	}
	ctx := matrixContext{ServedModel: servedModel, SubmittedWith: "lmx " + version + " run", Provenance: map[string]any{}}
	// Refuse unprovable hardware before spending time on the benchmark.
	hardware, hardwareSource, err := inferenceMatrixHardware(baseURL, args)
	if err != nil {
		return err
	}
	ctx.Hardware, ctx.HardwareSource = hardware, hardwareSource
	launchArgv := strings.Fields(opt(args, "server-command"))
	if opt(args, "quantization") == "" {
		resolution := remoteQuantizationResolution(args, baseURL, modelKey, "", servedInfo)
		ctx.Quantization = stringValue(resolution["trusted"])
		if ctx.Quantization == "" {
			ctx.Quantization, resolution = matrixQuantizationFromModel(launchArgv, hfID, args)
		}
		ctx.Provenance["quantizationResolution"] = resolution
		if ctx.Quantization == "" {
			return cliError{"missing_quantization", "Could not detect the served model's quantization.", []string{"Pass --quantization (e.g. BF16, FP8, AWQ) before the benchmark starts."}, resolution}
		}
	}

	script, scriptInfo, err := llmInferenceBenchScript(args)
	if err != nil {
		return err
	}
	python := firstNonEmpty(opt(args, "python-bin"), "python3")
	if out, err := exec.Command(python, "-c", "import httpx, rich, psutil").CombinedOutput(); err != nil {
		return cliError{"missing_python_dependencies", "llm-inference-bench needs Python 3.10+ with httpx, rich and psutil.", []string{python + " -m pip install httpx rich psutil", "Or pass --python-bin <interpreter> that already has them."}, strings.TrimSpace(string(out))}
	}

	outPath := firstNonEmpty(opt(args, "out"), filepath.Join("inference-matrices", safePathSegment(hfID), "matrix-"+time.Now().UTC().Format("20060102T150405Z")+".json"))
	if err := os.MkdirAll(filepath.Dir(outPath), 0o755); err != nil {
		return err
	}
	reportPath := strings.TrimSuffix(outPath, filepath.Ext(outPath)) + ".bench.json"
	benchArgs := []string{"--host", baseURL, "--model", servedModel, "--display-mode", "plain", "--output", reportPath, "--no-resume"}
	for flag, option := range map[string]string{
		"--concurrency":      "concurrency",
		"--contexts":         "contexts",
		"--prefill-contexts": "prefill-contexts",
		"--duration":         "duration-seconds",
		"--max-tokens":       "max-tokens",
	} {
		if value := opt(args, option); value != "" {
			benchArgs = append(benchArgs, flag, value)
		}
	}
	benchArgs = append(benchArgs, strings.Fields(opt(args, "extra-bench-args"))...)
	scriptInfo["argv"] = redactSecretArgs(benchArgs)
	if modelKey != "" {
		benchArgs = append(benchArgs, "--api-key", modelKey)
	}

	printStatus(args, "inference_matrix_start", map[string]any{"baseUrl": baseURL, "servedModel": servedModel, "report": reportPath})
	cmd := exec.Command(python, append([]string{script}, benchArgs...)...)
	// Keep stdout clean for --json; the bench's progress goes to stderr.
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr
	cmd.Env = append(os.Environ(), "LLM_BENCH_NO_UPDATE_CHECK=1")
	if err := cmd.Run(); err != nil {
		return cliError{"matrix_benchmark_failed", "llm_decode_bench.py failed: " + err.Error(), []string{"Read the benchmark output above.", "A partial report, if any, is at " + reportPath + "; import it with lmx speed-test matrix import."}, nil}
	}
	reportValue, err := readMatrixSource(reportPath)
	if err != nil {
		return err
	}
	report := asObject(reportValue)
	if opt(args, "engine-version") == "" {
		ctx.EngineVersion = probeInferenceEngineVersion(baseURL, modelKey, stringValue(asObject(report["metadata"])["engine"]))
	}
	ctx.Provenance["harness"] = scriptInfo
	ctx.Provenance["reportPath"] = reportPath
	return importMatrixReport(report, ctx, outPath, args)
}

// importInferenceMatrix turns an existing llm_decode_bench.py report or
// lil-bench result into a saved matrix upload.
func importInferenceMatrix(path string, args cliArgs) error {
	if path == "" {
		return cliError{"missing_path", "speed-test matrix import requires a report path", []string{"lmx speed-test matrix import benchmark_results.json --hf-id org/model --quantization FP8 --hardware hardware.json"}, nil}
	}
	value, err := readMatrixSource(path)
	if err != nil {
		return err
	}
	source := asObject(value)
	if source == nil {
		return cliError{"invalid_matrix_report", path + " is not a JSON object.", nil, nil}
	}
	ctx := matrixContext{SubmittedWith: "lmx " + version + " import", Provenance: map[string]any{"importedFrom": absPathOr(path)}}
	outPath := firstNonEmpty(opt(args, "out"), filepath.Join("inference-matrices", safePathSegment(firstNonEmpty(opt(args, "hf-id"), opt(args, "model"))), strings.TrimSuffix(strings.TrimSuffix(filepath.Base(path), ".gz"), ".json")+".matrix.json"))
	return importMatrixReport(source, ctx, outPath, args)
}

// importMatrixReport is the single path from a report to a saved upload, shared
// by run and import. A lil-bench result contributes its own inventory, serving
// command and engine version; flags override everything.
func importMatrixReport(source map[string]any, ctx matrixContext, outPath string, args cliArgs) error {
	report := source
	hfID := firstNonEmpty(opt(args, "hf-id"), opt(args, "model"))
	launchArgv := strings.Fields(opt(args, "server-command"))
	if schema := stringValue(source["schema"]); strings.HasPrefix(schema, "lil-bench-result/") {
		report = asObject(asObject(source["results"])["bench"])
		if report == nil {
			return cliError{"invalid_matrix_report", "lil-bench result has no benchmark report (status " + stringValue(source["status"]) + ").", nil, nil}
		}
		if hardware := lilBenchHardware(asObject(source["hardware"])); hardware != nil {
			ctx.Hardware, ctx.HardwareSource = hardware, "lil_bench_inventory"
		}
		server := asObject(source["server"])
		ctx.EngineVersion = firstNonEmpty(ctx.EngineVersion, stringValue(asObject(server["api"])["version"]))
		ctx.ServedModel = firstNonEmpty(ctx.ServedModel, stringValue(asObject(server["api"])["model_id"]))
		argv := []string{}
		for _, part := range anySlice(server["argv"]) {
			argv = append(argv, stringValue(part))
		}
		ctx.CommandSnippet = firstNonEmpty(ctx.CommandSnippet, shellJoin(redactSecretArgs(argv)))
		if len(launchArgv) == 0 {
			launchArgv = argv
		}
		ctx.SubmittedWith += " (lil-bench " + firstNonEmpty(stringValue(asObject(source["client"])["lil_bench"]), "unknown") + ")"
	}
	if opt(args, "hardware") != "" {
		loaded, _, err := benchmarkHardware("remote", args)
		if err != nil {
			return err
		}
		ctx.Hardware, ctx.HardwareSource = asObject(loaded), "file"
	}
	if ctx.Hardware == nil {
		return cliError{"missing_remote_hardware", "Importing a matrix requires the measured server's hardware.", []string{"Run lmx hardware --out hardware.json on the benchmark host, then pass --hardware hardware.json."}, nil}
	}
	flagVersion, repository, build, commit, err := engineProvenance("", args)
	if err != nil {
		return err
	}
	if opt(args, "quantization") == "" && ctx.Quantization == "" && hfID != "" {
		var resolution map[string]any
		ctx.Quantization, resolution = matrixQuantizationFromModel(launchArgv, hfID, args)
		ctx.Provenance["quantizationResolution"] = resolution
	}
	envelope, err := buildInferenceMatrixEnvelope(inferenceMatrixInput{
		HfID:             hfID,
		ModelRevision:    opt(args, "model-revision"),
		Quantization:     firstNonEmpty(opt(args, "quantization"), ctx.Quantization),
		EngineVersion:    firstNonEmpty(flagVersion, ctx.EngineVersion),
		EngineRepository: repository,
		EngineBuild:      build,
		EngineCommit:     commit,
		Backend:          firstNonEmpty(opt(args, "backend"), backendFromHardware(ctx.Hardware)),
		CommandSnippet:   firstNonEmpty(opt(args, "server-command"), ctx.CommandSnippet),
		SubmittedWith:    ctx.SubmittedWith,
		Hardware:         ctx.Hardware,
		Report:           report,
	})
	if err != nil {
		return err
	}
	if stringValue(envelope["engineVersion"]) == "" {
		printStatus(args, "inference_matrix_engine_version_missing", map[string]any{"hint": "The report does not record the engine version; pass --engine-version."})
	}
	if err := os.MkdirAll(filepath.Dir(outPath), 0o755); err != nil {
		return err
	}
	provenance := map[string]any{
		"cli":            "localmaxxing-go",
		"cliVersion":     version,
		"createdAt":      time.Now().UTC().Format(time.RFC3339),
		"harness":        map[string]any{"repository": llmInferenceBenchRepository, "version": stringValue(asObject(report["metadata"])["version"])},
		"servedModel":    ctx.ServedModel,
		"hardwareSource": ctx.HardwareSource,
	}
	for key, value := range ctx.Provenance {
		provenance[key] = value
	}
	return finishInferenceMatrix(outPath, map[string]any{"kind": inferenceMatrixKind, "payload": envelope, "provenance": provenance}, args)
}

func finishInferenceMatrix(outPath string, document map[string]any, args cliArgs) error {
	if err := writeJSON(outPath, document); err != nil {
		return err
	}
	payload := asObject(document["payload"])
	summary := inferenceMatrixSummary(payload)
	summary["path"] = outPath
	printInfo(args, "inference_matrix_saved", summary)
	if hasFlag(args, "dry-run") || hasFlag(args, "submit") {
		return sendInferenceMatrix(outPath, document, hasFlag(args, "dry-run"), args)
	}
	if humanOutput(args) {
		fmt.Println("Next:")
		fmt.Println("  lmx speed-test matrix dry-run " + shellQuote(outPath))
		fmt.Println("  lmx speed-test matrix submit " + shellQuote(outPath))
	}
	return nil
}

func inferenceMatrixSummary(payload map[string]any) map[string]any {
	benchmark := asObject(payload["benchmark"])
	metadata := asObject(benchmark["metadata"])
	return map[string]any{
		"hfId":          payload["hfId"],
		"quantization":  payload["quantization"],
		"engine":        metadata["engine"],
		"engineVersion": payload["engineVersion"],
		"benchVersion":  metadata["version"],
		"prefillCells":  len(asObject(benchmark["prefill"])),
		"decodeCells":   len(anySlice(benchmark["results"])),
		"burstCells":    len(anySlice(benchmark["burst_results"])),
	}
}

func sendInferenceMatrixFile(path string, dryRun bool, args cliArgs) error {
	if path == "" {
		return cliError{"missing_path", "Pass a saved matrix JSON written by lmx speed-test matrix run/import.", nil, nil}
	}
	value, err := readJSON(path)
	if err != nil {
		return err
	}
	document := asObject(value)
	if document == nil || stringValue(document["kind"]) != inferenceMatrixKind || asObject(document["payload"]) == nil {
		return cliError{"invalid_matrix_file", path + " is not an lmx inference matrix file.", []string{"Create one with lmx speed-test matrix run or lmx speed-test matrix import."}, nil}
	}
	return sendInferenceMatrix(path, document, dryRun, args)
}

func sendInferenceMatrix(path string, document map[string]any, dryRun bool, args cliArgs) error {
	key := apiKey(args)
	if key == "" {
		return missingAPIKey("--api-key or LMX_API_KEY is required for speed-test matrix dry-run/submit")
	}
	endpoint := inferenceMatrixEndpoint
	if dryRun {
		endpoint += "/dry-run"
	}
	payload := asObject(document["payload"])
	response, err := fetchJSON("POST", apiURL(args)+endpoint, key, payload)
	if err != nil {
		return err
	}
	if hasFlag(args, "json") || hasFlag(args, "print") || hasFlag(args, "verbose") {
		printJSON(args, response)
	}
	result := asObject(response)
	if dryRun {
		fields := inferenceMatrixSummary(payload)
		fields["status"] = "valid"
		run := asObject(result["run"])
		fields["cells"] = len(anySlice(run["cells"]))
		fields["storedHardware"] = asObject(run["hardware"])["gpuName"]
		fields["rateLimitRemaining"] = asObject(result["rateLimit"])["remaining"]
		if stringValue(run["status"]) == "SUSPICIOUS" {
			fields["review"] = "would be held for admin review: " + stringValue(run["reviewReason"])
		}
		printInfo(args, "inference_matrix_dry_run_valid", fields)
		return nil
	}
	submission := map[string]any{
		"id":          result["id"],
		"submittedAt": firstNonEmpty(stringValue(result["createdAt"]), time.Now().UTC().Format(time.RFC3339)),
		"url":         apiURL(args) + "/en/inference-benchmarks/" + stringValue(result["id"]),
		"apiUrl":      apiURL(args),
		"status":      result["status"],
	}
	document["submission"] = submission
	if path != "" {
		if err := writeJSON(path, document); err != nil {
			return cliError{"submission_receipt_write_failed", "Submission succeeded, but its receipt could not be saved.", []string{"The server run ID is " + stringValue(result["id"]) + "."}, err.Error()}
		}
	}
	fields := map[string]any{"id": result["id"], "cells": len(anySlice(result["cells"])), "url": submission["url"]}
	if stringValue(result["status"]) == "SUSPICIOUS" {
		fields["review"] = "held for admin review and hidden until approved: " + stringValue(result["reviewReason"])
	}
	printInfo(args, "inference_matrix_submitted", fields)
	return nil
}

// llmInferenceBenchScript returns a path to llm_decode_bench.py: --bench-script,
// or the pinned upstream release cached under the user cache directory.
func llmInferenceBenchScript(args cliArgs) (string, map[string]any, error) {
	info := map[string]any{"repository": llmInferenceBenchRepository}
	if local := opt(args, "bench-script"); local != "" {
		data, err := os.ReadFile(local)
		if err != nil {
			return "", nil, cliError{"bench_script_unreadable", "Cannot read --bench-script " + local + ": " + err.Error(), nil, nil}
		}
		info["scriptPath"] = absPathOr(local)
		info["sha256"] = sha256Hex(string(data))
		info["ref"] = "local"
		return local, info, nil
	}
	ref := llmInferenceBenchDefaultRef
	info["ref"] = ref
	cacheRoot, err := os.UserCacheDir()
	if err != nil {
		return "", nil, err
	}
	path := filepath.Join(cacheRoot, "localmaxxing", "llm-inference-bench", safePathSegment(ref), "llm_decode_bench.py")
	data, err := os.ReadFile(path)
	if err != nil {
		rawURL := "https://raw.githubusercontent.com/local-inference-lab/llm-inference-bench/" + url.PathEscape(ref) + "/llm_decode_bench.py"
		data, err = downloadBenchScript(rawURL)
		if err != nil {
			return "", nil, cliError{"bench_script_download_failed", "Could not download llm_decode_bench.py " + ref + ": " + err.Error(), []string{"Pass --bench-script <path to llm_decode_bench.py> from a checkout of " + llmInferenceBenchRepository + "."}, nil}
		}
	}
	sum := sha256Hex(string(data))
	if sum != llmInferenceBenchDefaultSHA256 {
		return "", nil, cliError{"bench_script_checksum_mismatch", "llm_decode_bench.py " + ref + " does not match the pinned checksum.", []string{"Delete " + path + " and retry, or pass --bench-script."}, map[string]any{"expected": llmInferenceBenchDefaultSHA256, "actual": sum}}
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return "", nil, err
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		return "", nil, err
	}
	info["scriptPath"] = path
	info["sha256"] = sum
	return path, info, nil
}

func downloadBenchScript(rawURL string) ([]byte, error) {
	client := &http.Client{Timeout: 2 * time.Minute}
	res, err := client.Get(rawURL)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if res.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("%s returned %s", rawURL, res.Status)
	}
	return io.ReadAll(res.Body)
}

// readMatrixSource reads plain or gzip-compressed JSON (lil-bench saves .json.gz).
func readMatrixSource(path string) (any, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if bytes.HasPrefix(data, []byte{0x1f, 0x8b}) {
		reader, err := gzip.NewReader(bytes.NewReader(data))
		if err != nil {
			return nil, err
		}
		defer reader.Close()
		if data, err = io.ReadAll(reader); err != nil {
			return nil, err
		}
	}
	var value any
	if err := json.Unmarshal(data, &value); err != nil {
		return nil, cliError{"invalid_json", path + ": " + err.Error(), nil, nil}
	}
	return value, nil
}

// inferenceMatrixHardware uses --hardware, or detects this host when the
// endpoint is on this machine. Remote endpoints need explicit server hardware.
func inferenceMatrixHardware(baseURL string, args cliArgs) (map[string]any, string, error) {
	if opt(args, "hardware") != "" {
		loaded, source, err := benchmarkHardware("remote", args)
		if err != nil {
			return nil, "", err
		}
		return asObject(loaded), source, nil
	}
	if !localURL(baseURL) {
		return nil, "", cliError{"missing_remote_hardware", "The endpoint is not on this machine, so its hardware cannot be detected.", []string{"Run lmx hardware --out hardware.json on the server, then pass --hardware hardware.json."}, nil}
	}
	return detectHardware(args), "detected_local", nil
}

// localURL reports whether rawURL names this machine: localhost, loopback, or
// an IP address assigned to one of its interfaces.
func localURL(rawURL string) bool {
	parsed, err := url.Parse(rawURL)
	if err != nil {
		return false
	}
	host := parsed.Hostname()
	if host == "localhost" {
		return true
	}
	ip := net.ParseIP(host)
	if ip == nil || ip.IsUnspecified() {
		return false
	}
	if ip.IsLoopback() {
		return true
	}
	addresses, _ := net.InterfaceAddrs()
	for _, address := range addresses {
		if network, ok := address.(*net.IPNet); ok && network.IP.Equal(ip) {
			return true
		}
	}
	return false
}

// Weight formats named by a launch --quantization flag or a config.json
// quantization_config.quant_method. compressed-tensors and bitsandbytes cover
// several formats, so they stay unresolved and need --quantization.
var matrixQuantMethods = map[string]string{
	"fp8": "FP8", "fbgemm_fp8": "FP8", "ptpc_fp8": "FP8",
	"awq": "AWQ", "awq_marlin": "AWQ",
	"gptq": "GPTQ", "gptq_marlin": "GPTQ",
	"mxfp4": "MXFP4", "modelopt_fp4": "NVFP4",
}

var matrixDtypes = map[string]string{
	"bfloat16": "BF16", "bf16": "BF16",
	"float16": "FP16", "fp16": "FP16", "half": "FP16",
	"float32": "FP32", "fp32": "FP32", "float": "FP32",
}

// matrixQuantizationFromModel resolves the served weight format when the
// endpoint does not report one (vLLM and SGLang never do for unquantized
// models). Precedence: a launch --quantization flag, the model's
// config.json quantization_config, a launch --dtype, then the config dtype.
// config.json comes from the model directory named in the launch command,
// else from Hugging Face (the repo in the launch command, else hfID).
func matrixQuantizationFromModel(argv []string, hfID string, args cliArgs) (string, map[string]any) {
	resolution := map[string]any{}
	flags := launchFlags(argv)
	if value := flags["quantization"]; value != "" {
		resolution["launchQuantization"] = value
		if quant := matrixQuantMethods[strings.ToLower(value)]; quant != "" {
			return matrixQuantResolved(resolution, quant, "launch_flag")
		}
		resolution["unresolved"] = "launch --quantization " + value + " does not name one weight format"
		return "", resolution
	}
	location := firstNonEmpty(flags["model-path"], flags["model"], servePositionalModel(argv))
	if strings.HasSuffix(strings.ToLower(location), ".gguf") {
		if quant := quantizationFromFilename(location); quant != "" {
			resolution["modelPath"] = location
			return matrixQuantResolved(resolution, quant, "gguf_filename")
		}
	}
	config, configSource, err := matrixModelConfig(location, hfID, args)
	if err != nil {
		resolution["configError"] = err.Error()
	}
	resolution["config"] = configSource
	if qc := asObject(config["quantization_config"]); qc != nil {
		method := strings.ToLower(stringValue(qc["quant_method"]))
		if method == "modelopt" {
			method = map[string]string{"FP8": "fp8", "NVFP4": "modelopt_fp4"}[strings.ToUpper(stringValue(qc["quant_algo"]))]
		}
		if quant := matrixQuantMethods[method]; quant != "" {
			return matrixQuantResolved(resolution, quant, "config_quantization")
		}
		resolution["unresolved"] = "config.json quantization_config " + firstNonEmpty(stringValue(qc["quant_method"]), "(no quant_method)") + " does not name one weight format"
		return "", resolution
	}
	if value := strings.ToLower(flags["dtype"]); value != "" && value != "auto" {
		resolution["launchDtype"] = value
		if quant := matrixDtypes[value]; quant != "" {
			return matrixQuantResolved(resolution, quant, "launch_dtype")
		}
	}
	dtype := strings.ToLower(firstNonEmpty(stringValue(config["torch_dtype"]), stringValue(config["dtype"])))
	if quant := matrixDtypes[dtype]; quant != "" {
		return matrixQuantResolved(resolution, quant, "config_dtype")
	}
	return "", resolution
}

func matrixQuantResolved(resolution map[string]any, quant, source string) (string, map[string]any) {
	resolution["trusted"], resolution["trustedSource"] = quant, source
	return quant, resolution
}

// launchFlags reads --name value and --name=value pairs from a server argv.
func launchFlags(argv []string) map[string]string {
	flags := map[string]string{}
	for i, arg := range argv {
		name, value, inline := strings.Cut(strings.TrimPrefix(arg, "--"), "=")
		if !strings.HasPrefix(arg, "--") || name == "" {
			continue
		}
		name = strings.ReplaceAll(name, "_", "-")
		if !inline && i+1 < len(argv) && !strings.HasPrefix(argv[i+1], "-") {
			value = argv[i+1]
		}
		if _, seen := flags[name]; !seen {
			flags[name] = value
		}
	}
	return flags
}

// servePositionalModel returns the model in `vllm serve <model>` / `sglang serve <model>`.
func servePositionalModel(argv []string) string {
	for i, arg := range argv {
		if arg == "serve" && i+1 < len(argv) && !strings.HasPrefix(argv[i+1], "-") {
			return argv[i+1]
		}
	}
	return ""
}

// matrixModelConfig reads config.json from a local model directory, else
// fetches it from Hugging Face for the launch repo or hfID.
func matrixModelConfig(location, hfID string, args cliArgs) (map[string]any, string, error) {
	if location != "" {
		path := filepath.Join(location, "config.json")
		if data, err := os.ReadFile(path); err == nil {
			var config map[string]any
			if err := json.Unmarshal(data, &config); err != nil {
				return nil, path, err
			}
			return config, path, nil
		}
	}
	repo := hfID
	if parts := strings.Split(location, "/"); len(parts) == 2 && parts[0] != "" && parts[1] != "" && !filepath.IsAbs(location) {
		repo = location
	}
	if repo == "" {
		return nil, "", errors.New("no local model directory or Hugging Face repo to read config.json from")
	}
	configURL := strings.TrimRight(hfAPIURL(args), "/") + "/" + hfRepoPath(repo) + "/resolve/main/config.json"
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	value, err := fetchJSONContext(ctx, "GET", configURL, "", nil)
	if err != nil {
		return nil, configURL, err
	}
	config := asObject(value)
	if config == nil {
		return nil, configURL, errors.New("config.json at " + configURL + " is not a JSON object")
	}
	return config, configURL, nil
}

// lilBenchHardware maps a lil-bench hardware inventory through the same
// builder as native nvidia-smi detection, so both describe one host
// identically. Returns nil without GPU data (pynvml missing in the container).
func lilBenchHardware(inventory map[string]any) map[string]any {
	var smi strings.Builder
	for _, raw := range anySlice(inventory["gpus"]) {
		gpu := asObject(raw)
		if name := strings.ReplaceAll(stringValue(gpu["name"]), ",", " "); name != "" {
			fmt.Fprintf(&smi, "%s, %g\n", name, numberField(gpu, "memory_total_mib"))
		}
	}
	if smi.Len() == 0 {
		return nil
	}
	hardware := map[string]any{"hwClass": "CPU_ONLY", "os": "linux"}
	if cpu := stringValue(asObject(asObject(inventory["cpu"])["lscpu"])["Model name"]); cpu != "" {
		hardware["cpu"] = cpu
	}
	if memMiB := numberField(asObject(inventory["memory"]), "mem_total_mib"); memMiB > 0 {
		hardware["ramGb"] = round1(memMiB / 1024)
	}
	applyNvidiaSMIHardware(hardware, smi.String())
	return hardware
}

// probeInferenceEngineVersion asks the server for its version: vLLM serves
// GET /version, SGLang GET /get_server_info.
func probeInferenceEngineVersion(baseURL, apiKey, engine string) string {
	paths := map[string][]string{"vllm": {"/version"}, "sglang": {"/get_server_info", "/version"}}[engine]
	for _, path := range paths {
		if value, err := fetchEndpointJSON(baseURL+path, apiKey); err == nil {
			if version := stringValue(asObject(value)["version"]); version != "" {
				return version
			}
		}
	}
	return ""
}

// findServingProcess returns the lowest-PID vLLM/SGLang process serving a
// local endpoint's port, or 0 when none is visible (e.g. inside Docker).
func findServingProcess(baseURL string) (int, []string) {
	if runtime.GOOS != "linux" || !localURL(baseURL) {
		return 0, nil
	}
	parsed, _ := url.Parse(baseURL)
	port := parsed.Port()
	entries, err := os.ReadDir("/proc")
	if err != nil {
		return 0, nil
	}
	pids := []int{}
	for _, entry := range entries {
		if pid, err := strconv.Atoi(entry.Name()); err == nil {
			pids = append(pids, pid)
		}
	}
	sort.Ints(pids)
	for _, pid := range pids {
		data, err := os.ReadFile(filepath.Join("/proc", strconv.Itoa(pid), "cmdline"))
		if err != nil || len(data) == 0 {
			continue
		}
		argv := strings.Split(strings.TrimRight(string(data), "\x00"), "\x00")
		if serverPort := servingPort(argv); serverPort != "" && serverPort == port {
			return pid, argv
		}
	}
	return 0, nil
}

// servingPort returns the port a vLLM or SGLang server argv listens on, or "".
func servingPort(argv []string) string {
	joined := " " + strings.Join(argv, " ") + " "
	defaultPort := ""
	switch {
	case strings.Contains(joined, "sglang.launch_server") || strings.Contains(joined, "sglang serve"):
		defaultPort = "30000"
	case strings.Contains(joined, "vllm serve") || strings.Contains(joined, "/vllm serve") || strings.Contains(joined, "vllm.entrypoints.openai.api_server"):
		defaultPort = "8000"
	default:
		return ""
	}
	for i, arg := range argv {
		if arg == "--port" && i+1 < len(argv) {
			return argv[i+1]
		}
		if strings.HasPrefix(arg, "--port=") {
			return strings.TrimPrefix(arg, "--port=")
		}
	}
	return defaultPort
}

func redactSecretArgs(argv []string) []string {
	out := make([]string, len(argv))
	copy(out, argv)
	for i := 0; i < len(out); i++ {
		name, _, hasValue := strings.Cut(out[i], "=")
		if !strings.HasPrefix(name, "--") || !secretArgPattern.MatchString(name) {
			continue
		}
		if hasValue {
			out[i] = name + "=<redacted>"
		} else if i+1 < len(out) && !strings.HasPrefix(out[i+1], "--") {
			out[i+1] = "<redacted>"
			i++
		}
	}
	return out
}

func shellJoin(argv []string) string {
	quoted := make([]string, len(argv))
	for i, arg := range argv {
		quoted[i] = shellQuote(arg)
	}
	return strings.Join(quoted, " ")
}
