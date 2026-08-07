package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"runtime"
	"sort"
	"strings"
	"time"
)

// These values are overridden by release builds with -ldflags -X.
var (
	version   = "dev"
	commit    = "unknown"
	buildDate = "unknown"
)

var extraOptionNames = map[string]bool{
	"all": true, "auth-poll-interval": true, "auth-timeout": true, "baseline": true,
	"bench-format": true, "bench-output": true, "benchmark": true,
	"benchmark-backend": true, "benchmark-mode": true,
	"category": true, "compact": true, "content-type": true, "context-depth": true,
	"context-length": true, "context-levels": true, "contexts": true, "ctx-size": true,
	"dataset-name": true, "dataset-path": true, "description": true, "detached-child": true,
	"dtype": true, "endpoint": true, "engine": true, "extra-bench-args": true,
	"extra-server-args": true, "fewshot": true, "filename": true, "force": true,
	"gpu-layers": true, "gpu-power-watts": true, "hardware-cost": true, "help": true,
	"hf-api-url": true, "iters": true, "json": true, "judge-api-key": true,
	"judge-base-url": true, "judge-model": true, "key": true, "key-stdin": true,
	"logout": true, "max-concurrency": true, "max-num-seqs": true, "max-running-requests": true,
	"max-running-seqs": true, "model-args": true, "model-name": true, "no-flash-attn": true,
	"notes": true, "num-iters": true, "num-parallel": true, "num-warmups": true, "parallel": true,
	"output-format": true,
	"output-tokens": true, "path": true, "print": true, "q": true, "query": true,
	"request-rate": true, "run": true, "runner": true, "runner-version": true,
	"runs": true, "save-run": true, "scorer-bin": true, "scoring-method": true,
	"section": true, "slug": true, "status-log": true, "suite-file": true, "task": true,
	"tasks": true, "tensor-parallel": true, "threads": true, "timeout-seconds": true,
	"tok-s-out": true, "tok-s-prefill": true, "tok-s-total": true, "ttft-ms": true,
	"peak-vram-gb": true, "ubatch-size": true, "verbose": true, "version": true,
}

func knownOptionName(name string) bool {
	if extraOptionNames[name] {
		return true
	}
	needle := "--" + name
	for _, line := range strings.Split(usageOptions, "\n") {
		line = strings.TrimSpace(line)
		if line == needle || strings.HasPrefix(line, needle+" ") || strings.HasPrefix(line, needle+",") {
			return true
		}
	}
	return false
}

func validateOptionNames(args cliArgs) error {
	for name := range args.provided {
		if knownOptionName(name) {
			continue
		}
		hints := []string{"Run lmx <command> --help to list supported options."}
		if suggestion := nearestOption(name); suggestion != "" {
			hints = append([]string{"Did you mean --" + suggestion + "?"}, hints...)
		}
		return cliError{"unknown_option", "Unknown option --" + name + ".", hints, map[string]any{"option": "--" + name}}
	}
	return nil
}

func nearestOption(name string) string {
	candidates := map[string]bool{}
	for key := range extraOptionNames {
		candidates[key] = true
	}
	for _, line := range strings.Split(usageOptions, "\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "--") {
			continue
		}
		key := strings.Fields(strings.TrimPrefix(line, "--"))[0]
		key = strings.TrimSuffix(key, ",")
		candidates[key] = true
	}
	best, distance := "", 4
	for candidate := range candidates {
		if d := editDistance(name, candidate); d < distance {
			best, distance = candidate, d
		}
	}
	return best
}

func editDistance(a, b string) int {
	previous := make([]int, len(b)+1)
	for j := range previous {
		previous[j] = j
	}
	for i, ar := range []rune(a) {
		current := make([]int, len(previous))
		current[0] = i + 1
		for j, br := range []rune(b) {
			cost := 0
			if ar != br {
				cost = 1
			}
			current[j+1] = minInt(current[j]+1, previous[j+1]+1, previous[j]+cost)
		}
		previous = current
	}
	return previous[len(previous)-1]
}

func minInt(values ...int) int {
	result := values[0]
	for _, value := range values[1:] {
		if value < result {
			result = value
		}
	}
	return result
}

func handleVersion(args cliArgs) error {
	payload := map[string]any{
		"version": version, "commit": commit, "buildDate": buildDate,
		"goVersion": runtime.Version(), "platform": runtime.GOOS + "/" + runtime.GOARCH,
	}
	if hasFlag(args, "json") || hasFlag(args, "json-status") {
		printJSON(args, payload)
		return nil
	}
	fmt.Printf("lmx %s (%s, %s, %s)\n", version, commit, buildDate, payload["platform"])
	return nil
}

type commandSchemaEntry struct {
	Name               string   `json:"name"`
	Description        string   `json:"description,omitempty"`
	Examples           []string `json:"examples,omitempty"`
	ExampleOptionNames []string `json:"exampleOptionNames,omitempty"`
	Authentication     string   `json:"authentication"`
	SideEffects        []string `json:"sideEffects,omitempty"`
}

type optionSchemaEntry struct {
	Name        string `json:"name"`
	ValueHint   string `json:"valueHint,omitempty"`
	Description string `json:"description,omitempty"`
}

func commandSchema() map[string]any {
	names := make([]string, 0, len(commandDescriptions))
	for name := range commandDescriptions {
		names = append(names, name)
	}
	sort.Strings(names)
	commands := make([]commandSchemaEntry, 0, len(names))
	for _, name := range names {
		examples := []string{}
		for _, example := range usageExamples {
			if example == "lmx "+name || strings.HasPrefix(example, "lmx "+name+" ") {
				examples = append(examples, example)
			}
		}
		commands = append(commands, commandSchemaEntry{
			Name: name, Description: commandDescriptions[name], Examples: examples,
			ExampleOptionNames: optionNamesFromExamples(examples),
			Authentication:     commandAuthentication(name),
			SideEffects:        commandSideEffects(name),
		})
	}
	optionDetails := []optionSchemaEntry{}
	seen := map[string]bool{}
	for _, line := range strings.Split(usageOptions, "\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "--") {
			continue
		}
		fields := strings.Fields(line)
		name := strings.TrimSuffix(strings.TrimPrefix(fields[0], "--"), ",")
		if seen[name] {
			continue
		}
		seen[name] = true
		entry := optionSchemaEntry{Name: name}
		index := 1
		if len(fields) > 1 && strings.HasPrefix(fields[1], "<") {
			entry.ValueHint = fields[1]
			index = 2
		}
		if index < len(fields) {
			entry.Description = strings.Join(fields[index:], " ")
		}
		optionDetails = append(optionDetails, entry)
	}
	for name := range extraOptionNames {
		if !seen[name] {
			seen[name] = true
			optionDetails = append(optionDetails, optionSchemaEntry{Name: name})
		}
	}
	sort.Slice(optionDetails, func(i, j int) bool { return optionDetails[i].Name < optionDetails[j].Name })
	optionNames := make([]string, len(optionDetails))
	for i, option := range optionDetails {
		optionNames[i] = option.Name
	}
	return map[string]any{
		"schemaVersion":     1,
		"cliVersion":        version,
		"commands":          commands,
		"options":           optionDetails,
		"globalOptionNames": optionNames,
		"validation":        "Unknown option names are rejected before command execution.",
		"machineOutput": map[string]any{
			"json":       "Exactly one final JSON document on stdout; human progress is suppressed.",
			"jsonStatus": "JSONL progress and errors on stderr.",
			"quiet":      "Suppress progress and human status output without suppressing requested JSON results.",
		},
	}
}

func optionNamesFromExamples(examples []string) []string {
	names := map[string]bool{"help": true, "json": true, "json-status": true, "quiet": true}
	for _, example := range examples {
		for _, field := range strings.Fields(example) {
			if !strings.HasPrefix(field, "--") {
				continue
			}
			name := strings.SplitN(strings.TrimPrefix(field, "--"), "=", 2)[0]
			names[strings.TrimRight(name, `',"`)] = true
		}
	}
	out := make([]string, 0, len(names))
	for name := range names {
		out = append(out, name)
	}
	sort.Strings(out)
	return out
}

func commandAuthentication(name string) string {
	switch {
	case strings.HasPrefix(name, "auth keys"), strings.Contains(name, "submit"), name == "setups", strings.HasPrefix(name, "setups "):
		return "required"
	case name == "auth":
		return "optional"
	default:
		return "none"
	}
}

func commandSideEffects(name string) []string {
	switch {
	case name == "update" || name == "upgrade":
		return []string{"replaces installed executable"}
	case name == "auth":
		return []string{"may persist or remove credentials"}
	case name == "skill" || strings.HasPrefix(name, "skill "):
		return []string{"may write skill files"}
	case name == "speed-test" || strings.HasPrefix(name, "speed-test "):
		return []string{"may execute speed tests, write run files, or submit results; --dry-run does not persist unless --save-run"}
	case name == "eval" || strings.HasPrefix(name, "eval "):
		return []string{"may execute workloads, write checkpoints, or submit results"}
	case name == "hardware" || strings.HasPrefix(name, "hardware "):
		return []string{"may probe host hardware or write metadata"}
	case name == "context":
		return []string{"performs a read-only network request"}
	default:
		return nil
	}
}

func handleCommands(args cliArgs) error {
	return writeOrPrintJSON("commands", args, commandSchema())
}

func readKeyFromStdin(reader io.Reader) (string, error) {
	scanner := bufio.NewScanner(reader)
	if !scanner.Scan() {
		if err := scanner.Err(); err != nil {
			return "", err
		}
		return "", cliError{"missing_api_key", "No API key was provided on stdin.", []string{"Pipe an API key to lmx auth --key-stdin."}, nil}
	}
	key := strings.TrimSpace(scanner.Text())
	if key == "" {
		return "", cliError{"missing_api_key", "No API key was provided on stdin.", []string{"Pipe an API key to lmx auth --key-stdin."}, nil}
	}
	return key, nil
}

func projectContext(value any, args cliArgs) (any, error) {
	root := asObject(value)
	if root == nil {
		return nil, cliError{"context_invalid", "Agent context response was not a JSON object.", nil, nil}
	}
	root["_cli"] = map[string]any{
		"schemaVersion": 1,
		"fetchedAt":     time.Now().UTC().Format(time.RFC3339),
		"cliVersion":    version,
		"apiURL":        apiURL(args),
	}
	action := positional(args, 1)
	selector := firstNonEmpty(opt(args, "section"), positional(args, 2))
	if action == "list" {
		keys := make([]string, 0, len(root))
		for key := range root {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		return map[string]any{"schemaVersion": 1, "sections": keys}, nil
	}
	if action == "get" && selector == "" {
		return nil, cliError{"missing_context_section", "context get requires a dotted section path.", []string{"Run lmx context list, then lmx context get <path>."}, nil}
	}
	if selector == "" {
		return root, nil
	}
	var current any = root
	for _, part := range strings.Split(selector, ".") {
		object := asObject(current)
		if object == nil {
			return nil, cliError{"context_section_not_found", "Context section was not found: " + selector, []string{"Run lmx context list to inspect top-level sections."}, nil}
		}
		next, ok := object[part]
		if !ok {
			return nil, cliError{"context_section_not_found", "Context section was not found: " + selector, []string{"Run lmx context list to inspect top-level sections."}, nil}
		}
		current = next
	}
	return map[string]any{"schemaVersion": 1, "path": selector, "value": current}, nil
}

func writeJSONOutput(path string, value any, compact bool) error {
	var data []byte
	var err error
	if compact {
		data, err = json.Marshal(value)
	} else {
		data, err = json.MarshalIndent(value, "", "  ")
	}
	if err != nil {
		return err
	}
	data = append(data, '\n')
	return os.WriteFile(path, data, 0o644)
}
