package main

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/csv"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

func handleSuiteImport(path string, args cliArgs) error {
	if path == "" {
		return cliError{"missing_input", "eval suite import requires a CSV, JSONL, or JSON path.", []string{"Pass the dataset path as the first argument."}, nil}
	}
	slug, err := requireOpt(args, "slug")
	if err != nil {
		return err
	}
	rows, err := readSuiteImportRows(path)
	if err != nil {
		return cliError{"dataset_import_failed", "Could not import the dataset.", []string{"Use CSV with a header row, newline-delimited JSON objects, or a JSON array."}, err.Error()}
	}
	if len(rows) == 0 {
		return cliError{"dataset_empty", "The imported dataset has no rows.", nil, nil}
	}

	kind := firstNonEmpty(opt(args, "kind"), "qa")
	templateKind := kind
	if kind == "exact_match" {
		templateKind = "qa"
	}
	name := firstNonEmpty(opt(args, "name"), slug)
	payload := buildSuiteTemplate(slug, name, firstNonEmpty(opt(args, "category"), "general"), "CUSTOM", "exact_match", withOption(args, "kind", templateKind))
	if sourceURL := opt(args, "source-url"); sourceURL != "" {
		payload["sourceUrl"] = sourceURL
	}

	inputColumn := firstNonEmpty(opt(args, "input-column"), "input")
	goldColumn := firstNonEmpty(opt(args, "gold-column"), "gold")
	choicesColumn := firstNonEmpty(opt(args, "choices-column"), "choices")
	referenceColumn := firstNonEmpty(opt(args, "reference-column"), "referenceAnswer")
	rubricColumn := firstNonEmpty(opt(args, "rubric-column"), "rubric")
	items := make([]any, 0, len(rows))
	for index, row := range rows {
		input := strings.TrimSpace(stringValue(row[inputColumn]))
		if input == "" {
			return cliError{"dataset_import_failed", fmt.Sprintf("Row %d has no %q value.", index+1, inputColumn), []string{"Pass --input-column <name> when the source uses another column."}, nil}
		}
		item := map[string]any{"input": input}
		switch kind {
		case "judge":
			rubric := strings.TrimSpace(stringValue(row[rubricColumn]))
			if rubric == "" {
				return cliError{"dataset_import_failed", fmt.Sprintf("Row %d has no %q value.", index+1, rubricColumn), []string{"Judge datasets require a rubric for every row."}, nil}
			}
			item["rubric"] = rubric
			if reference := strings.TrimSpace(stringValue(row[referenceColumn])); reference != "" {
				item["referenceAnswer"] = reference
			}
		default:
			gold, ok := row[goldColumn]
			if !ok {
				return cliError{"dataset_import_failed", fmt.Sprintf("Row %d has no %q column.", index+1, goldColumn), []string{"Pass --gold-column <name> when the source uses another column."}, nil}
			}
			item["gold"] = gold
		}
		if kind == "multiple_choice" || kind == "loglikelihood" {
			choices := parseImportedChoices(row[choicesColumn], firstNonEmpty(opt(args, "choices-separator"), "|"))
			if len(choices) < 2 {
				return cliError{"dataset_import_failed", fmt.Sprintf("Row %d needs at least two choices in %q.", index+1, choicesColumn), []string{"Store choices as a JSON array or separate them with --choices-separator (default: |)."}, nil}
			}
			item["choices"] = choices
		}
		items = append(items, item)
	}

	task := evalTasks(suiteDoc(payload))[0]
	dataset := asObject(task["dataset"])
	dataset["items"] = items
	if err := validateSuite(payload); err != nil {
		return err
	}
	out := firstNonEmpty(opt(args, "out"), slug+".eval-suite.json")
	if err := writeJSON(out, payload); err != nil {
		return err
	}
	printInfo(args, "suite_imported", map[string]any{"path": out, "slug": slug, "items": len(items), "kind": kind})
	return nil
}

func withOption(args cliArgs, name, value string) cliArgs {
	copyArgs := args
	copyArgs.opts = make(map[string]string, len(args.opts)+1)
	for key, existing := range args.opts {
		copyArgs.opts[key] = existing
	}
	copyArgs.opts[name] = value
	return copyArgs
}

func readSuiteImportRows(path string) ([]map[string]any, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	ext := strings.ToLower(filepath.Ext(path))
	if ext == ".csv" {
		reader := csv.NewReader(file)
		headers, err := reader.Read()
		if err != nil {
			return nil, err
		}
		rows := []map[string]any{}
		for {
			record, err := reader.Read()
			if err == io.EOF {
				break
			}
			if err != nil {
				return nil, err
			}
			row := map[string]any{}
			for index, header := range headers {
				if index < len(record) {
					row[strings.TrimSpace(header)] = record[index]
				}
			}
			rows = append(rows, row)
		}
		return rows, nil
	}
	if ext == ".json" {
		var rows []map[string]any
		if err := json.NewDecoder(file).Decode(&rows); err != nil {
			return nil, err
		}
		return rows, nil
	}
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 64*1024), 20*1024*1024)
	rows := []map[string]any{}
	line := 0
	for scanner.Scan() {
		line++
		if strings.TrimSpace(scanner.Text()) == "" {
			continue
		}
		row := map[string]any{}
		if err := json.Unmarshal(scanner.Bytes(), &row); err != nil {
			return nil, fmt.Errorf("line %d: %w", line, err)
		}
		rows = append(rows, row)
	}
	return rows, scanner.Err()
}

func parseImportedChoices(value any, separator string) []any {
	if values := anySlice(value); len(values) > 0 {
		return values
	}
	text := strings.TrimSpace(stringValue(value))
	if text == "" {
		return nil
	}
	var decoded []any
	if json.Unmarshal([]byte(text), &decoded) == nil && len(decoded) > 0 {
		return decoded
	}
	parts := strings.Split(text, separator)
	choices := make([]any, 0, len(parts))
	for _, part := range parts {
		if trimmed := strings.TrimSpace(part); trimmed != "" {
			choices = append(choices, trimmed)
		}
	}
	return choices
}

type suiteAuditIssue struct {
	Severity string `json:"severity"`
	Code     string `json:"code"`
	Path     string `json:"path,omitempty"`
	Message  string `json:"message"`
}

func handleSuiteAudit(path string, args cliArgs) error {
	if path == "" {
		return cliError{"missing_input", "eval suite audit requires a suite JSON path.", nil, nil}
	}
	payload, err := readJSON(path)
	if err != nil {
		return err
	}
	if err := validateSuite(payload); err != nil {
		return err
	}
	result := auditSuite(payload)
	if err := writeOrPrintJSON("suite_audit", args, result); err != nil {
		return err
	}
	if numberField(result, "errorCount") > 0 {
		return cliError{"suite_audit_failed", "Suite audit found blocking dataset problems.", []string{"Fix the reported items and rerun eval suite audit."}, result["issues"]}
	}
	return nil
}

func auditSuite(value any) map[string]any {
	obj := asObject(value)
	issues := []suiteAuditIssue{}
	if strings.TrimSpace(stringValue(obj["description"])) == "" {
		issues = append(issues, suiteAuditIssue{"warning", "description_missing", "description", "Describe what the suite measures and how the items were created."})
	}
	if strings.TrimSpace(stringValue(obj["sourceUrl"])) == "" {
		issues = append(issues, suiteAuditIssue{"warning", "source_missing", "sourceUrl", "Add methodology, provenance, or license information."})
	}
	totalItems := 0
	answerDistribution := map[string]int{}
	seenInputs := map[string]string{}
	for taskIndex, task := range evalTasks(suiteDoc(obj)) {
		dataset := asObject(task["dataset"])
		if stringValue(dataset["source"]) != "inline" {
			continue
		}
		for itemIndex, raw := range anySlice(dataset["items"]) {
			item := asObject(raw)
			path := fmt.Sprintf("suiteDoc.tasks[%d].dataset.items[%d]", taskIndex, itemIndex)
			input := strings.TrimSpace(stringValue(item["input"]))
			normalized := normalizeAuditText(input)
			if previous, exists := seenInputs[normalized]; exists {
				issues = append(issues, suiteAuditIssue{"error", "duplicate_input", path + ".input", "Duplicates " + previous + "."})
			} else {
				seenInputs[normalized] = path + ".input"
			}
			if len(input) > 12000 {
				issues = append(issues, suiteAuditIssue{"warning", "input_very_long", path + ".input", "Input exceeds 12,000 characters; verify context and cost expectations."})
			}
			gold := strings.TrimSpace(stringValue(item["gold"]))
			if gold != "" {
				answerDistribution[strings.ToUpper(gold)]++
				if len(gold) >= 4 && strings.Contains(strings.ToLower(input), strings.ToLower(gold)) {
					issues = append(issues, suiteAuditIssue{"warning", "possible_gold_leak", path, "The gold answer appears verbatim in the input."})
				}
			}
			choices := anySlice(item["choices"])
			if len(choices) > 0 {
				seenChoices := map[string]bool{}
				for _, rawChoice := range choices {
					choice := normalizeAuditText(stringValue(rawChoice))
					if seenChoices[choice] {
						issues = append(issues, suiteAuditIssue{"error", "duplicate_choice", path + ".choices", "Choices must be unique within an item."})
						break
					}
					seenChoices[choice] = true
				}
				if !validChoiceGold(gold, choices) {
					issues = append(issues, suiteAuditIssue{"error", "invalid_choice_gold", path + ".gold", "Gold must be a valid choice letter, zero-based index, or exact choice value."})
				}
			}
			totalItems++
		}
	}
	if totalItems > 0 && totalItems < 20 {
		issues = append(issues, suiteAuditIssue{"warning", "dataset_small", "suiteDoc.tasks", "Public suites should normally contain at least 20 items."})
	}
	if len(answerDistribution) > 1 && totalItems >= 20 {
		minCount, maxCount := totalItems, 0
		for _, count := range answerDistribution {
			if count < minCount {
				minCount = count
			}
			if count > maxCount {
				maxCount = count
			}
		}
		if minCount > 0 && maxCount > minCount*3 {
			issues = append(issues, suiteAuditIssue{"warning", "answer_imbalance", "suiteDoc.tasks", "Multiple-choice gold answers are strongly imbalanced."})
		}
	}
	sort.Slice(issues, func(i, j int) bool {
		if issues[i].Severity != issues[j].Severity {
			return issues[i].Severity < issues[j].Severity
		}
		if issues[i].Path != issues[j].Path {
			return issues[i].Path < issues[j].Path
		}
		return issues[i].Code < issues[j].Code
	})
	errors, warnings := 0, 0
	for _, issue := range issues {
		if issue.Severity == "error" {
			errors++
		} else {
			warnings++
		}
	}
	return map[string]any{"valid": errors == 0, "itemCount": totalItems, "errorCount": errors, "warningCount": warnings, "answerDistribution": answerDistribution, "issues": issues}
}

func normalizeAuditText(value string) string {
	return strings.Join(strings.Fields(strings.ToLower(value)), " ")
}

func validChoiceGold(gold string, choices []any) bool {
	if gold == "" {
		return false
	}
	upper := strings.ToUpper(strings.TrimSpace(gold))
	if len(upper) == 1 && upper[0] >= 'A' && int(upper[0]-'A') < len(choices) {
		return true
	}
	if index, err := strconv.Atoi(upper); err == nil && index >= 0 && index < len(choices) {
		return true
	}
	for _, choice := range choices {
		if strings.EqualFold(strings.TrimSpace(stringValue(choice)), strings.TrimSpace(gold)) {
			return true
		}
	}
	return false
}

func validateSuiteRemote(payload any, args cliArgs) error {
	key := apiKey(args)
	if key == "" {
		return missingAPIKey("--api-key or LMX_API_KEY is required for authoritative suite validation")
	}
	_, err := fetchJSON("POST", apiURL(args)+"/api/benchmarks/suites/dry-run", key, payload)
	if err == nil || !isAPIStatus(err, http.StatusMethodNotAllowed) {
		return err
	}

	slug := stringValue(asObject(payload)["slug"])
	_, lookupErr := fetchJSON("GET", apiURL(args)+"/api/benchmarks/suites/"+url.PathEscape(slug), "", nil)
	if lookupErr == nil {
		return cliError{"suite_slug_exists", fmt.Sprintf("A benchmark with slug %q already exists.", slug), []string{"Choose a different --slug before any dataset upload."}, nil}
	}
	if !isAPIStatus(lookupErr, http.StatusNotFound) {
		return lookupErr
	}
	printStatus(args, "suite_remote_preflight_legacy", map[string]any{"slug": slug, "warning": "Server does not expose suite dry-run yet; local validation and public slug availability passed. The create endpoint remains authoritative."})
	return nil
}

func isAPIStatus(err error, status int) bool {
	value, ok := err.(cliError)
	return ok && value.Code == "api_error" && strings.HasPrefix(value.Message, strconv.Itoa(status)+" ")
}

func uploadSuiteInlineDatasets(payload any, args cliArgs) (any, error) {
	key := apiKey(args)
	if key == "" {
		return nil, missingAPIKey("--api-key or LMX_API_KEY is required for dataset upload")
	}
	encoded, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	copyValue := map[string]any{}
	if err := json.Unmarshal(encoded, &copyValue); err != nil {
		return nil, err
	}
	for _, task := range evalTasks(suiteDoc(copyValue)) {
		dataset := asObject(task["dataset"])
		items := anySlice(dataset["items"])
		if stringValue(dataset["source"]) != "inline" || len(items) == 0 {
			continue
		}
		lines := bytes.Buffer{}
		for _, item := range items {
			line, err := json.Marshal(item)
			if err != nil {
				return nil, err
			}
			lines.Write(line)
			lines.WriteByte('\n')
		}
		filename := fmt.Sprintf("%s-%s.jsonl", stringValue(copyValue["slug"]), stringValue(task["key"]))
		storageRef, err := uploadSuiteDatasetBytes(lines.Bytes(), filename, len(items), key, args)
		if err != nil {
			return nil, err
		}
		task["dataset"] = map[string]any{"source": "bucket", "storageRef": storageRef}
	}
	return copyValue, nil
}

func uploadSuiteDatasetBytes(data []byte, filename string, itemCount int, key string, args cliArgs) (map[string]any, error) {
	hash := sha256.Sum256(data)
	metadata := map[string]any{"kind": "suite-dataset", "filename": filename, "contentType": "application/x-ndjson", "format": "jsonl", "byteSize": len(data), "sha256": hex.EncodeToString(hash[:]), "itemCount": itemCount}
	upload, err := fetchJSON("POST", apiURL(args)+"/api/benchmarks/storage/upload-url", key, metadata)
	if err != nil {
		return nil, err
	}
	uploadObj := asObject(upload)
	uploadURL := stringValue(uploadObj["uploadUrl"])
	storageRef := asObject(uploadObj["storageRef"])
	if uploadURL == "" || storageRef == nil {
		return nil, cliError{"storage_upload_response_invalid", "Storage upload response is missing uploadUrl or storageRef.", nil, upload}
	}
	req, err := http.NewRequest("PUT", uploadURL, bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	for header, value := range stringMap(uploadObj["headers"]) {
		req.Header.Set(header, value)
	}
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	body, _ := io.ReadAll(res.Body)
	res.Body.Close()
	if res.StatusCode < 200 || res.StatusCode >= 300 {
		return nil, cliError{"storage_put_failed", fmt.Sprintf("Storage PUT failed: %s", res.Status), []string{"Retry the upload; signed upload URLs can expire."}, string(body)}
	}
	completed, err := fetchJSON("POST", apiURL(args)+"/api/benchmarks/storage/complete", key, map[string]any{"storageRef": storageRef})
	if err != nil {
		return nil, err
	}
	verified := asObject(asObject(completed)["storageRef"])
	if verified == nil {
		return nil, cliError{"storage_complete_response_invalid", "Storage completion response is missing storageRef.", nil, completed}
	}
	return verified, nil
}

func handleSuiteCheck(path string, args cliArgs) error {
	if path == "" {
		return cliError{"missing_input", "eval suite check requires a suite JSON path.", nil, nil}
	}
	payload, err := readJSON(path)
	if err != nil {
		return err
	}
	if err := validateSuite(payload); err != nil {
		return err
	}
	suite := asObject(payload)
	if !strings.EqualFold(stringValue(suite["runner"]), "CUSTOM") {
		return cliError{"suite_check_unsupported", "Sample execution currently supports CUSTOM suites.", []string{"Use lmx eval lm-eval for LM_EVAL_HARNESS suites."}, nil}
	}
	limit := 5
	if raw := opt(args, "samples"); raw != "" {
		parsed, err := strconv.Atoi(raw)
		if err != nil || parsed < 1 || parsed > 100 {
			return cliError{"invalid_samples", "--samples must be between 1 and 100.", nil, raw}
		}
		limit = parsed
	}
	for _, task := range evalTasks(suiteDoc(suite)) {
		items, err := loadEvalDataset(asObject(task["dataset"]))
		if err != nil {
			return err
		}
		if len(items) > limit {
			items = items[:limit]
		}
		inline := make([]any, len(items))
		for index := range items {
			inline[index] = items[index]
		}
		task["dataset"] = map[string]any{"source": "inline", "items": inline}
	}
	result, err := runCustomLocalEval(suite, args)
	if err != nil {
		return err
	}
	failures := 0
	for _, raw := range anySlice(result["artifacts"]) {
		if stringValue(asObject(raw)["error"]) != "" {
			failures++
		}
	}
	summary := map[string]any{"valid": failures == 0, "samplesPerTask": limit, "failures": failures, "aggregate": result["aggregate"], "scores": result["scores"]}
	if err := writeOrPrintJSON("suite_check", args, summary); err != nil {
		return err
	}
	if failures > 0 {
		return cliError{"suite_check_failed", "One or more sampled benchmark items failed.", []string{"Inspect endpoint, prompt, scoring, and judge configuration."}, summary}
	}
	return nil
}
