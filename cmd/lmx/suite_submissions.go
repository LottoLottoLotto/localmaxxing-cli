package main

import (
	"fmt"
	"net/url"
)

func handleSuiteSubmissions(args cliArgs) error {
	key := apiKey(args)
	if key == "" {
		return missingAPIKey("--api-key or LMX_API_KEY is required to list your benchmark submissions")
	}
	value, err := fetchJSON("GET", apiURL(args)+"/api/benchmarks/submissions", key, nil)
	if err != nil {
		return err
	}
	return writeOrPrintJSON("suite_submissions", args, value)
}

func handleSuiteWithdraw(id string, args cliArgs) error {
	if id == "" {
		return cliError{"missing_submission_id", "eval suite withdraw requires a submission ID.", []string{"Run lmx eval suite submissions to find the ID."}, nil}
	}
	if !hasFlag(args, "yes") {
		return cliError{"confirmation_required", "Withdrawing a benchmark submission is permanent.", []string{"Rerun with --yes after checking the submission ID."}, map[string]any{"id": id}}
	}
	key := apiKey(args)
	if key == "" {
		return missingAPIKey("--api-key or LMX_API_KEY is required to withdraw a submission")
	}
	kind := firstNonEmpty(opt(args, "kind"), "suite")
	if kind != "suite" && kind != "dataset" {
		return cliError{"invalid_kind", "--kind must be suite or dataset.", nil, kind}
	}
	value, err := fetchJSON("DELETE", apiURL(args)+"/api/benchmarks/submissions/"+kind+"/"+url.PathEscape(id), key, nil)
	if err != nil {
		return err
	}
	return writeOrPrintJSON("suite_withdrawn", args, value)
}

func handleSuiteResubmit(id string, args cliArgs) error {
	if id == "" {
		return cliError{"missing_submission_id", "eval suite resubmit requires a suite submission ID.", []string{"Run lmx eval suite submissions to find the ID."}, nil}
	}
	path := opt(args, "file")
	if path == "" {
		return cliError{"missing_option", "--file <suite.json> is required.", nil, nil}
	}
	key := apiKey(args)
	if key == "" {
		return missingAPIKey("--api-key or LMX_API_KEY is required to resubmit a suite")
	}
	payload, err := readJSON(path)
	if err != nil {
		return err
	}
	if err := validateSuite(payload); err != nil {
		return err
	}
	if hasFlag(args, "upload-datasets") {
		payload, err = uploadSuiteInlineDatasets(payload, args)
		if err != nil {
			return err
		}
	}
	value, err := fetchJSON("PATCH", apiURL(args)+"/api/benchmarks/submissions/suite/"+url.PathEscape(id), key, map[string]any{"submission": payload})
	if err != nil {
		return err
	}
	printInfo(args, "suite_resubmitted", map[string]any{"id": id, "slug": asObject(payload)["slug"], "status": "PENDING"})
	printJSON(args, value)
	return nil
}

func handleSuiteClone(slug string, args cliArgs) error {
	if slug == "" {
		return cliError{"missing_suite", "eval suite clone requires an approved suite slug.", nil, nil}
	}
	newSlug, err := requireOpt(args, "slug")
	if err != nil {
		return err
	}
	suite, err := loadSuiteForEvalRun(slug, args)
	if err != nil {
		return err
	}
	if stringValue(suite["runner"]) == "CUSTOM" {
		for _, task := range evalTasks(suiteDoc(suite)) {
			items, err := loadEvalDataset(asObject(task["dataset"]))
			if err != nil {
				return cliError{"suite_clone_dataset_failed", fmt.Sprintf("Could not clone task %q: %v", stringValue(task["key"]), err), nil, nil}
			}
			inline := make([]any, len(items))
			for index := range items {
				inline[index] = items[index]
			}
			task["dataset"] = map[string]any{"source": "inline", "items": inline}
		}
	}
	delete(suite, "id")
	delete(suite, "createdBy")
	delete(suite, "isOfficial")
	delete(suite, "runCount")
	suite["slug"] = newSlug
	suite["name"] = firstNonEmpty(opt(args, "name"), stringValue(suite["name"])+" (fork)")
	suite["version"] = firstNonEmpty(opt(args, "version"), "1.0")
	out := firstNonEmpty(opt(args, "out"), newSlug+".eval-suite.json")
	if err := validateSuite(suite); err != nil {
		return err
	}
	if err := writeJSON(out, suite); err != nil {
		return err
	}
	printInfo(args, "suite_cloned", map[string]any{"source": slug, "slug": newSlug, "path": out})
	return nil
}
