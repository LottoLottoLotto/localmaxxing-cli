package main

import (
	"fmt"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

type evalPublishInput struct {
	Kind       string
	Path       string
	Rows       []map[string]any
	InputCol   string
	GoldCol    string
	ChoicesCol string
	RubricCol  string
	Suite      any
}

func handleEvalPublish(path string, args cliArgs) error {
	if path == "" {
		return cliError{"missing_input", "eval publish requires a benchmark file or Terminal-Bench directory.", []string{
			"Publish CSV, JSONL, JSON arrays, or suite manifests: lmx eval publish questions.jsonl --description \"What this measures and how it was created.\"",
			"Publish Harbor/Terminal-Bench tasks: lmx eval publish ./tasks --description \"What this measures.\" --source-url https://github.com/org/repo",
		}, nil}
	}

	input, err := detectEvalPublishInput(path, args)
	if err != nil {
		return err
	}
	if apiKey(args) == "" {
		return missingAPIKey("Sign in with `lmx auth login`, set LMX_API_KEY, or pass --api-key before publishing")
	}
	switch input.Kind {
	case "suite":
		printInfo(args, "eval_publish_plan", map[string]any{"detected": "suite manifest", "path": path, "steps": ordinaryPublishSteps(args, false)})
		return publishSuiteManifest(path, input.Suite, args)
	case "dataset":
		return publishImportedDataset(input, args)
	case "terminal", "terminal-imported":
		return publishTerminalInput(input, args)
	default:
		return cliError{"unsupported_publish_input", "Could not determine the benchmark format.", []string{"Use CSV, JSONL, a JSON array, a LocalMaxxing suite manifest, or a Harbor/Terminal-Bench task directory."}, map[string]any{"path": path}}
	}
}

func detectEvalPublishInput(path string, args cliArgs) (evalPublishInput, error) {
	info, err := os.Stat(path)
	if err != nil {
		return evalPublishInput{}, cliError{"publish_input_unreadable", fmt.Sprintf("Could not read publish input: %v", err), []string{"Check that the path exists and is readable."}, map[string]any{"path": path}}
	}
	if info.IsDir() {
		if bundles, bundleErr := loadTerminalBundles(path); bundleErr == nil && len(bundles) > 0 {
			return evalPublishInput{Kind: "terminal-imported", Path: path}, nil
		}
		tasks, taskErr := findHarborTaskDirs(path)
		if taskErr == nil && len(tasks) > 0 {
			return evalPublishInput{Kind: "terminal", Path: path}, nil
		}
		return evalPublishInput{}, cliError{"unsupported_publish_directory", "The directory is neither imported terminal bundles nor Harbor/Terminal-Bench tasks.", []string{
			"Harbor tasks need task.toml, instruction.md, tests/, and either environment/Dockerfile or environment.docker_image.",
			"Imported bundles need task.json and tests/.",
		}, map[string]any{"path": path}}
	}

	if strings.EqualFold(filepath.Ext(path), ".json") {
		value, readErr := readJSON(path)
		if readErr == nil {
			if obj := asObject(value); obj != nil && asObject(obj["suiteDoc"]) != nil {
				return evalPublishInput{Kind: "suite", Path: path, Suite: value}, nil
			}
		}
	}
	rows, err := readSuiteImportRows(path)
	if err != nil {
		return evalPublishInput{}, cliError{"publish_input_invalid", "Could not parse the benchmark input.", []string{"Use CSV with headers, JSONL objects, a JSON array, or a LocalMaxxing suite manifest."}, err.Error()}
	}
	if len(rows) == 0 {
		return evalPublishInput{}, cliError{"dataset_empty", "The benchmark dataset contains no rows.", []string{"Add at least one benchmark item before publishing."}, nil}
	}
	columns := publishDatasetColumns(rows[0])
	inputCol, err := resolvePublishColumn(args, "input-column", columns, []string{"input", "question", "prompt", "instruction"}, true)
	if err != nil {
		return evalPublishInput{}, err
	}
	rubricCol, _ := resolvePublishColumn(args, "rubric-column", columns, []string{"rubric", "grading_rubric", "criteria"}, false)
	choicesCol, _ := resolvePublishColumn(args, "choices-column", columns, []string{"choices", "options", "answers"}, false)
	goldCol, goldErr := resolvePublishColumn(args, "gold-column", columns, []string{"gold", "answer", "label", "target", "expected_answer"}, false)
	kind := strings.ToLower(opt(args, "kind"))
	if kind == "" {
		switch {
		case rubricCol != "":
			kind = "judge"
		case choicesCol != "":
			kind = "multiple_choice"
		default:
			kind = "qa"
		}
	}
	if kind != "judge" && (goldErr != nil || goldCol == "") {
		return evalPublishInput{}, cliError{"publish_mapping_ambiguous", "Could not find the expected-answer column.", []string{"Rename it to gold or answer, or pass --gold-column <column>."}, map[string]any{"columns": columns, "kind": kind}}
	}
	if (kind == "multiple_choice" || kind == "loglikelihood") && choicesCol == "" {
		return evalPublishInput{}, cliError{"publish_mapping_ambiguous", "Multiple-choice publication needs a choices column.", []string{"Rename it to choices or options, or pass --choices-column <column>."}, map[string]any{"columns": columns}}
	}
	if kind == "judge" && rubricCol == "" {
		return evalPublishInput{}, cliError{"publish_mapping_ambiguous", "Judge publication needs a rubric column.", []string{"Rename it to rubric, or pass --rubric-column <column>."}, map[string]any{"columns": columns}}
	}
	return evalPublishInput{Kind: "dataset", Path: path, Rows: rows, InputCol: inputCol, GoldCol: goldCol, ChoicesCol: choicesCol, RubricCol: rubricCol}, nil
}

func publishImportedDataset(input evalPublishInput, args cliArgs) error {
	slug := firstNonEmpty(opt(args, "slug"), publishSlug(input.Path))
	name := firstNonEmpty(opt(args, "name"), publishName(slug))
	description := strings.TrimSpace(opt(args, "description"))
	if len(description) < 20 {
		return cliError{"description_required", "Describe what the benchmark measures and how its items were created.", []string{"Pass --description with at least 20 characters. This is shown to reviewers and benchmark users."}, map[string]any{"slug": slug}}
	}
	kind := strings.ToLower(opt(args, "kind"))
	if kind == "" {
		if input.RubricCol != "" {
			kind = "judge"
		} else if input.ChoicesCol != "" {
			kind = "multiple_choice"
		} else {
			kind = "qa"
		}
	}
	out := filepath.Join(".localmaxxing", slug+".eval-suite.json")
	if err := os.MkdirAll(filepath.Dir(out), 0o755); err != nil {
		return err
	}
	importArgs := cloneEvalPublishArgs(args)
	importArgs.positional = []string{"eval", "suite", "import", input.Path}
	for key, value := range map[string]string{"slug": slug, "name": name, "description": description, "kind": kind, "input-column": input.InputCol, "gold-column": input.GoldCol, "choices-column": input.ChoicesCol, "rubric-column": input.RubricCol, "out": out} {
		if value != "" {
			importArgs.opts[key] = value
		}
	}
	printInfo(args, "eval_publish_plan", map[string]any{
		"detected": "tabular dataset", "path": input.Path, "items": len(input.Rows), "kind": kind, "slug": slug,
		"mapping":  map[string]any{"input": input.InputCol, "gold": input.GoldCol, "choices": input.ChoicesCol, "rubric": input.RubricCol},
		"manifest": out, "steps": ordinaryPublishSteps(args, true),
	})
	if err := handleSuiteImport(input.Path, importArgs); err != nil {
		return err
	}
	payload, err := readJSON(out)
	if err != nil {
		return err
	}
	return publishSuiteManifest(out, payload, args)
}

func publishSuiteManifest(path string, payload any, args cliArgs) error {
	if err := validateSuite(payload); err != nil {
		return err
	}
	obj := asObject(payload)
	if len(strings.TrimSpace(stringValue(obj["description"]))) < 20 {
		return cliError{"description_required", "Describe what the benchmark measures and how its items were created.", []string{"Add a description of at least 20 characters to the suite manifest."}, map[string]any{"path": path, "slug": obj["slug"]}}
	}
	audit := auditSuite(payload)
	printInfo(args, "eval_publish_audit", map[string]any{"items": audit["itemCount"], "errors": audit["errorCount"], "warnings": audit["warningCount"], "issues": audit["issues"]})
	if numberField(audit, "errorCount") > 0 || (hasFlag(args, "strict") && numberField(audit, "warningCount") > 0) {
		return cliError{"suite_audit_failed", "Publication stopped because the benchmark audit found problems.", []string{"Fix the listed issues and run the same lmx eval publish command again.", "Use --strict only when warnings must also block publication."}, audit["issues"]}
	}
	if err := validateSuiteRemote(payload, args); err != nil {
		return err
	}
	if hasFlag(args, "dry-run") {
		return writeOrPrintJSON("eval_publish_dry_run", args, map[string]any{"valid": true, "path": path, "slug": asObject(payload)["slug"], "audit": audit, "next": "Remove --dry-run to upload and submit for review."})
	}
	publishedPayload := payload
	if !hasFlag(args, "no-upload-datasets") {
		var err error
		publishedPayload, err = uploadSuiteInlineDatasets(payload, args)
		if err != nil {
			return err
		}
		if err := validateSuiteRemote(publishedPayload, args); err != nil {
			return err
		}
	}
	created, err := fetchJSON("POST", apiURL(args)+"/api/benchmarks/suites", apiKey(args), publishedPayload)
	if err != nil {
		return err
	}
	slug := stringValue(asObject(payload)["slug"])
	return writeOrPrintJSON("eval_published", args, map[string]any{
		"submission": created, "status": "PENDING", "slug": slug, "manifest": path,
		"next": []string{"Track review: lmx eval suite submissions", "If rejected: edit the manifest, then run lmx eval suite resubmit <submission-id> --file " + path, "After approval: " + strings.TrimRight(apiURL(args), "/") + "/benchmarks/" + slug},
	})
}

func publishTerminalInput(input evalPublishInput, args cliArgs) error {
	slug := firstNonEmpty(opt(args, "slug"), publishSlug(input.Path))
	name := firstNonEmpty(opt(args, "name"), publishName(slug))
	description := strings.TrimSpace(opt(args, "description"))
	if len(description) < 20 {
		return cliError{"description_required", "Describe what the terminal benchmark measures and how its tasks were created.", []string{"Pass --description with at least 20 characters."}, map[string]any{"slug": slug}}
	}
	sourceURL := strings.TrimSpace(opt(args, "source-url"))
	parsedURL, parseErr := url.ParseRequestURI(sourceURL)
	if parseErr != nil || parsedURL.Scheme != "https" || parsedURL.Host == "" {
		return cliError{"source_url_required", "Executable terminal benchmarks require a public HTTPS provenance URL.", []string{"Pass --source-url https://github.com/org/repo so reviewers can inspect source and licensing."}, map[string]any{"sourceUrl": sourceURL}}
	}
	publishPath := input.Path
	if input.Kind == "terminal" {
		publishPath = filepath.Join(".localmaxxing", slug+"-terminal-bundles")
		if err := os.RemoveAll(publishPath); err != nil {
			return err
		}
		importArgs := cloneEvalPublishArgs(args)
		importArgs.positional = []string{"eval", "terminal", "import", input.Path}
		importArgs.opts["out"] = publishPath
		printInfo(args, "eval_publish_plan", map[string]any{"detected": "Harbor/Terminal-Bench tasks", "path": input.Path, "slug": slug, "bundles": publishPath, "steps": terminalPublishSteps(args, true)})
		if err := runTerminalImport(importArgs); err != nil {
			return err
		}
	} else {
		printInfo(args, "eval_publish_plan", map[string]any{"detected": "imported terminal bundles", "path": input.Path, "slug": slug, "steps": terminalPublishSteps(args, false)})
	}
	publishArgs := cloneEvalPublishArgs(args)
	publishArgs.positional = []string{"eval", "terminal", "publish", publishPath}
	publishArgs.opts["slug"] = slug
	publishArgs.opts["name"] = name
	publishArgs.opts["description"] = description
	publishArgs.opts["source-url"] = sourceURL
	return publishTerminalDataset(publishArgs)
}

func resolvePublishColumn(args cliArgs, option string, columns []string, aliases []string, required bool) (string, error) {
	if explicit := strings.TrimSpace(opt(args, option)); explicit != "" {
		for _, column := range columns {
			if column == explicit {
				return column, nil
			}
		}
		return "", cliError{"publish_mapping_invalid", fmt.Sprintf("Column %q passed to --%s does not exist.", explicit, option), []string{"Choose one of the columns shown in Details."}, map[string]any{"columns": columns}}
	}
	for _, alias := range aliases {
		for _, column := range columns {
			if strings.EqualFold(strings.TrimSpace(column), alias) {
				return column, nil
			}
		}
	}
	if required {
		return "", cliError{"publish_mapping_ambiguous", "Could not find the benchmark input column.", []string{"Rename it to input, question, prompt, or instruction; or pass --input-column <column>."}, map[string]any{"columns": columns}}
	}
	return "", nil
}

func publishDatasetColumns(row map[string]any) []string {
	columns := make([]string, 0, len(row))
	for column := range row {
		columns = append(columns, column)
	}
	sort.Strings(columns)
	return columns
}

var publishSlugPattern = regexp.MustCompile(`[^a-z0-9]+`)

func publishSlug(path string) string {
	base := filepath.Base(filepath.Clean(path))
	for _, suffix := range []string{".eval-suite.json", ".jsonl", ".ndjson", ".json", ".csv"} {
		if strings.HasSuffix(strings.ToLower(base), suffix) {
			base = base[:len(base)-len(suffix)]
			break
		}
	}
	slug := strings.Trim(publishSlugPattern.ReplaceAllString(strings.ToLower(base), "-"), "-")
	if len(slug) > 64 {
		slug = strings.TrimRight(slug[:64], "-")
	}
	if len(slug) < 3 {
		slug = "my-benchmark"
	}
	return slug
}

func publishName(slug string) string {
	parts := strings.Fields(strings.NewReplacer("-", " ", "_", " ").Replace(slug))
	for index := range parts {
		if len(parts[index]) > 0 {
			parts[index] = strings.ToUpper(parts[index][:1]) + parts[index][1:]
		}
	}
	return strings.Join(parts, " ")
}

func ordinaryPublishSteps(args cliArgs, imports bool) []string {
	steps := []string{}
	if imports {
		steps = append(steps, "import")
	}
	steps = append(steps, "validate", "audit", "server preflight")
	if hasFlag(args, "dry-run") {
		return append(steps, "stop before upload and submission")
	}
	return append(steps, "upload dataset", "submit for review")
}

func terminalPublishSteps(args cliArgs, imports bool) []string {
	steps := []string{"verify Pro access and quota"}
	if imports {
		steps = append(steps, "import")
	}
	if hasFlag(args, "skip-oracle") {
		steps = append(steps, "skip oracle (explicit override)")
	} else {
		steps = append(steps, "oracle verify")
	}
	steps = append(steps, "package", "upload", "server preflight")
	if hasFlag(args, "dry-run") {
		return append(steps, "stop before dataset creation")
	}
	return append(steps, "submit for review")
}

func cloneEvalPublishArgs(args cliArgs) cliArgs {
	clone := args
	clone.positional = append([]string(nil), args.positional...)
	clone.opts = make(map[string]string, len(args.opts))
	for key, value := range args.opts {
		clone.opts[key] = value
	}
	clone.flags = make(map[string]bool, len(args.flags))
	for key, value := range args.flags {
		clone.flags[key] = value
	}
	clone.provided = make(map[string]bool, len(args.provided))
	for key, value := range args.provided {
		clone.provided[key] = value
	}
	return clone
}
