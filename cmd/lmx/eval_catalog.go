package main

import (
	"fmt"
	"net/http"
)

func handleEvalDataset(action, target string, args cliArgs) error {
	if action != "list" && action != "show" {
		return fmt.Errorf("Unknown dataset command. Use eval dataset list or eval dataset show <slug>.")
	}
	if action == "show" && target == "" {
		return fmt.Errorf("eval dataset show requires a dataset slug")
	}
	value, err := fetchJSON("GET", apiURL(args)+"/api/benchmarks/datasets", "", nil)
	if err != nil {
		return err
	}
	if action == "list" {
		return writeOrPrintJSON("eval_datasets", args, value)
	}
	datasets, _ := asObject(value)["datasets"].([]any)
	for _, dataset := range datasets {
		if stringValue(asObject(dataset)["slug"]) == target {
			return writeOrPrintJSON("eval_dataset", args, dataset)
		}
	}
	return cliError{"dataset_not_found", fmt.Sprintf("No approved shard dataset named %q was found.", target), []string{"Run lmx eval dataset list to find approved shard datasets.", "Registered suites are separate: use lmx eval suite list."}, nil}
}

func suiteLookupError(slug string, err error) error {
	if !isAPIStatus(err, http.StatusNotFound) {
		return err
	}
	return cliError{"suite_not_found", fmt.Sprintf("No approved suite named %q was found.", slug), []string{
		"Run lmx eval dataset list: standard benchmarks such as HellaSwag and GSM8K are shard datasets, not suites.",
		"For a question dataset listed there, use lmx eval shard <dataset> --base-url <url> --questions 3 --dry-run; for terminal datasets, use lmx eval terminal --help.",
		"Use lmx eval suite list for registered suites; unpublished suites require approval before public runs.",
	}, err.Error()}
}
