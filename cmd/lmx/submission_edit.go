package main

import (
	"fmt"
	"strings"
)

// Both saved-run and remote edits consume every assignment. Other options keep
// their existing single-value parsing semantics.
func parseEditAssignments(args cliArgs) (map[string]any, error) {
	if hasFlag(args, "set") {
		return nil, cliError{Code: "invalid_option", Message: "--set requires field=value."}
	}
	if len(args.setValues) == 0 {
		return nil, nil
	}
	patch := make(map[string]any, len(args.setValues))
	for _, assignment := range args.setValues {
		field, raw, ok := strings.Cut(assignment, "=")
		field = strings.TrimSpace(field)
		if !ok || field == "" {
			return nil, cliError{Code: "invalid_option", Message: "--set requires field=value.", Hints: []string{"Repeat --set for multiple fields, for example --set tensorParallel=2 --set prefixCaching=false."}}
		}
		patch[field] = parseEditValue(raw)
	}
	return patch, nil
}

// PATCH /api/runs/{id} accepts flat fields, unlike the nested submission shape.
// Normalize each input before merging so source precedence is independent of
// whether an input uses the flat API shape or a nested engineFlags object.
func normalizeSubmissionEdit(patch map[string]any) error {
	for field := range patch {
		if field == "engineFlags" {
			continue
		}
		if known, _ := submissionEditField(field); !known {
			return cliError{Code: "invalid_patch", Message: fmt.Sprintf("Unknown remote edit field %q; refusing an edit the API could silently ignore.", field)}
		}
	}
	value, nested := patch["engineFlags"]
	if !nested {
		return nil
	}
	flags := asObject(value)
	if len(flags) == 0 {
		return cliError{Code: "invalid_patch", Message: "engineFlags must be a nonempty JSON object of editable engine fields."}
	}
	for field := range flags {
		if _, engineField := submissionEditField(field); !engineField {
			return cliError{Code: "invalid_patch", Message: fmt.Sprintf("Unknown engineFlags edit field %q.", field)}
		}
		if _, collision := patch[field]; collision {
			return cliError{Code: "invalid_patch", Message: fmt.Sprintf("Edit field %q appears both at the top level and inside engineFlags; use one representation per input.", field)}
		}
	}
	for field, value := range flags {
		patch[field] = value
	}
	delete(patch, "engineFlags")
	return nil
}

// The remote edit endpoint has a different field contract from POST submission
// and local file editing. Value/range validation remains the API's responsibility.
func submissionEditField(field string) (known, engineField bool) {
	switch field {
	case "ttftMs", "tokSOut", "tokSPrefill", "tokSTotal", "peakVramGb",
		"gpuPowerWatts", "hardwareCost", "promptTokens", "outputTokens",
		"contextLength", "batchSize", "prefillTokens", "notes", "quantization",
		"engineName", "engineVersion", "engineRepository", "engineBuild",
		"engineCommit", "backend":
		return true, false
	case "commandSnippet", "tensorParallel", "pipelineParallel", "gpuLayers",
		"splitMode", "kvCacheDtype", "gpuMemUtil", "kvCacheSizeMb",
		"prefixCaching", "attentionBackend", "flashAttn", "chunkedPrefill",
		"prefillChunkSize", "contBatching", "cpuOffloadGb", "cpuLayers",
		"ropeScaling", "ropeScale", "yarnExtFactor", "engineQuant", "sglangQuant",
		"maxRunningSeqs", "schedulerDelayFactor", "numParallel", "concurrency",
		"specDecoding", "specMethod", "specModel", "specDraftModel", "specNumTokens",
		"specNgramSize", "specDraftTp", "specDraftWindowSize", "mtpEnabled",
		"mtpDraftLayers", "specDraftTokens", "specAcceptedTokens", "specAcceptanceRate",
		"specMeanAcceptedLength", "temperature", "topP", "topK", "minP",
		"repeatPenalty", "mirostat", "extraFlags":
		return true, true
	default:
		return false, false
	}
}
