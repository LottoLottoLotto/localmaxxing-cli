package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"text/tabwriter"
	"time"
)

// `lmx model load` registers a model from `lmx model list` with substrate and
// has the substrate gateway start it, so every model is served at one
// endpoint and addressed by name. `unload` stops it, `loaded` lists running
// instances, `logs` shows an instance's log. Engines, GPU placement,
// Tenstorrent chip swapping and restarts are substrate's (see substrate.go).

// substrateNamePattern is substrate's model-name rule.
var substrateNamePattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$`)

func loadedName(value string) string {
	value = strings.ToLower(value)
	if slash := strings.LastIndex(value, "/"); slash >= 0 && slash < len(value)-1 {
		value = value[slash+1:]
	}
	value = regexp.MustCompile(`[^a-z0-9._-]+`).ReplaceAllString(value, "-")
	value = strings.Trim(value, "-._")
	if len(value) > 64 {
		value = value[:64]
	}
	if value == "" {
		value = "model"
	}
	return value
}

// jsonStringArrayOption parses an option holding a JSON array of strings, such
// as --server-args '["--gpu-memory-utilization","0.5"]'.
func jsonStringArrayOption(args cliArgs, key string) ([]string, error) {
	value := opt(args, key)
	if value == "" {
		return nil, nil
	}
	var result []string
	if err := json.Unmarshal([]byte(value), &result); err != nil || result == nil {
		return nil, fmt.Errorf("--%s must be a JSON string array", key)
	}
	for _, item := range result {
		if strings.ContainsRune(item, 0) {
			return nil, fmt.Errorf("--%s contains a NUL byte", key)
		}
	}
	return result, nil
}

// modelRegistrationArgs maps lmx options to `substrate model add` flags. Engine
// arguments after -- are engine-specific, so substrate requires --engine with them.
func modelRegistrationArgs(args cliArgs) ([]string, error) {
	var flags []string
	for option, flag := range map[string]string{
		"engine": "--engine", "context-length": "--ctx", "gpus": "--gpus", "gpu-layers": "--gpu-layers",
		"tensor-parallel": "--tp", "mem-fraction": "--mem-fraction", "idle-unload": "--idle-unload",
	} {
		if value := opt(args, option); value != "" {
			flags = append(flags, flag, value)
		}
	}
	sortFlagPairs(flags)
	extra, err := jsonStringArrayOption(args, "server-args")
	if err != nil {
		return nil, err
	}
	if len(extra) > 0 {
		flags = append(append(flags, "--"), extra...)
	}
	return flags, nil
}

// sortFlagPairs orders --flag value pairs by flag so registrations are stable.
func sortFlagPairs(flags []string) {
	for i := 0; i+2 < len(flags); i += 2 {
		for j := i + 2; j+1 < len(flags); j += 2 {
			if flags[j] < flags[i] {
				flags[i], flags[j] = flags[j], flags[i]
				flags[i+1], flags[j+1] = flags[j+1], flags[i+1]
			}
		}
	}
}

func handleModelLoad(target string, args cliArgs) error {
	entry, err := resolveInventoryModel(target, args)
	if err != nil {
		return err
	}
	if entry.Status != "available" && entry.Status != "ready" {
		return cliError{"model_not_loadable", entry.Name + " is " + entry.Status + ".", []string{firstNonEmpty(entry.SourceError, "Run lmx model show "+target+".")}, nil}
	}
	name := firstNonEmpty(opt(args, "name"), loadedName(entry.Name))
	if !substrateNamePattern.MatchString(name) {
		return cliError{"invalid_name", "--name must start with a letter or digit and use letters, digits, '.', '_', ':' or '-' (max 128).", nil, nil}
	}
	registration, err := modelRegistrationArgs(args)
	if err != nil {
		return err
	}
	models, err := substrateModels(args)
	if err != nil {
		return err
	}
	var existing *substrateModel
	for i := range models {
		if models[i].Name == name {
			existing = &models[i]
		}
	}
	switch {
	case existing != nil && existing.Source != entry.Path:
		return cliError{"name_in_use", "The gateway already serves " + existing.Source + " as " + name + ".", []string{"Pass --name <other name>, or lmx model unload " + name + " --forget first."}, nil}
	case existing == nil || len(registration) > 0:
		argv := append([]string{"model", "add", name, entry.Path}, registration...)
		if existing != nil {
			argv = append(argv[:4], append([]string{"--force"}, argv[4:]...)...)
		}
		if _, err := runSubstrate(args, argv...); err != nil {
			return err
		}
		printStatus(args, "model_registered", map[string]any{"name": name, "source": entry.Path})
	}
	base, err := startGateway(args)
	if err != nil {
		return err
	}
	timeout, err := intOption(args, 900, 1, "ready-timeout-seconds")
	if err != nil {
		return err
	}
	printStatus(args, "model_loading", map[string]any{"name": name, "model": entry.Name, "gateway": base + "/v1"})
	instance, err := gatewayEnsure(base, name, time.Duration(timeout)*time.Second)
	if err != nil {
		if failure, ok := err.(cliError); ok {
			failure.Hints = append(failure.Hints, "Engine log: lmx model logs "+name, "The model stays registered; the gateway starts it on the next request (lmx model unload "+name+" --forget removes it).")
			return failure
		}
		return err
	}
	if !humanOutput(args) || opt(args, "out") != "" {
		return writeOrPrintJSON("model_load", args, map[string]any{"name": name, "gateway": base + "/v1", "instance": instance})
	}
	fmt.Printf("Loaded %s on %s → %s/v1 as %q\n", entry.Name, instance.Engine, base, name)
	fmt.Printf("  try:    curl %s/v1/chat/completions -H 'Content-Type: application/json' -d '{\"model\":%q,\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}]}'\n", base, name)
	fmt.Printf("  logs:   lmx model logs %s\n  unload: lmx model unload %s\n", name, name)
	return nil
}

func handleModelUnload(target string, args cliArgs) error {
	targets := args.positional[min(len(args.positional), 2):]
	if hasFlag(args, "all") {
		instances, err := substrateInstances(args)
		if err != nil {
			return err
		}
		targets = nil
		for _, instance := range instances {
			targets = append(targets, instance.ID)
		}
	} else if target == "" {
		return cliError{"missing_model", "model unload requires a loaded model name (or --all).", []string{"Run lmx model loaded."}, nil}
	}
	base := gatewayURL(args)
	up := gatewayUp(base)
	stopped := []string{}
	for _, id := range targets {
		if up {
			// The gateway lets in-flight requests finish before stopping.
			if err := gatewayCall(base, http.MethodPost, "/substrate/v1/stop", map[string]any{"id": id}, nil, 3*time.Minute); err != nil {
				return err
			}
		} else if _, err := runSubstrate(args, "stop", id); err != nil {
			return err
		}
		stopped = append(stopped, id)
	}
	if hasFlag(args, "forget") && !hasFlag(args, "all") {
		for _, name := range targets {
			if _, err := runSubstrate(args, "model", "rm", name); err != nil {
				return err
			}
		}
	}
	fields := map[string]any{"unloaded": strings.Join(stopped, ", ")}
	if len(stopped) == 0 {
		fields["unloaded"] = "nothing was running"
	}
	if !hasFlag(args, "forget") {
		fields["note"] = "Registered models start again on their next request; --forget unregisters."
	}
	printInfo(args, "model_unloaded", fields)
	return nil
}

func handleModelLoaded(args cliArgs) error {
	base := gatewayURL(args)
	var instances []substrateInstance
	if gatewayUp(base) {
		var result []struct {
			Instance substrateInstance `json:"instance"`
			Status   string            `json:"status"`
		}
		if err := gatewayCall(base, http.MethodGet, "/substrate/v1/instances", nil, &result, 10*time.Second); err != nil {
			return err
		}
		for _, entry := range result {
			entry.Instance.Status = entry.Status
			instances = append(instances, entry.Instance)
		}
	} else {
		var err error
		if instances, err = substrateInstances(args); err != nil {
			return err
		}
	}
	if !humanOutput(args) || opt(args, "out") != "" {
		return writeOrPrintJSON("model_loaded", args, map[string]any{"gateway": base + "/v1", "gatewayUp": gatewayUp(base), "instances": instances})
	}
	if len(instances) == 0 {
		fmt.Println("No models loaded. Load one with lmx model load <name>.")
		return nil
	}
	writer := tabwriter.NewWriter(os.Stdout, 0, 4, 2, ' ', 0)
	fmt.Fprintln(writer, "NAME\tENGINE\tDEVICE\tSTATUS\tOWNER\tUP\tINSTANCE URL")
	for _, instance := range instances {
		device := "-"
		if len(instance.GPUs) > 0 {
			device = "gpu:" + joinInts(instance.GPUs)
		} else if len(instance.Devices) > 0 {
			device = "tt:" + joinInts(instance.Devices)
		}
		up := "-"
		if !instance.StartedAt.IsZero() {
			up = time.Since(instance.StartedAt).Round(time.Second).String()
		}
		fmt.Fprintf(writer, "%s\t%s\t%s\t%s\t%s\t%s\t%s\n", instance.Name, instance.Engine, device, firstNonEmpty(instance.Status, "running"), instance.Owner, up, instance.BaseURL)
	}
	writer.Flush()
	fmt.Printf("\nAll models: %s/v1 (model = NAME)\n", base)
	return nil
}

func handleModelLogs(target string, args cliArgs) error {
	if target == "" {
		return cliError{"missing_model", "model logs requires a loaded model name.", []string{"Run lmx model loaded."}, nil}
	}
	bin, err := substrateBin(args)
	if err != nil {
		return err
	}
	lines, err := intOption(args, 50, 1, "lines")
	if err != nil {
		return err
	}
	argv := []string{"logs", target, "-n", strconv.Itoa(lines)}
	if hasFlag(args, "follow") {
		argv = append(argv, "-f")
	}
	cmd := exec.Command(bin, argv...)
	cmd.Stdout, cmd.Stderr = os.Stdout, os.Stderr
	return cmd.Run()
}

func tailFile(path string, lines int) []string {
	file, err := os.Open(path)
	if err != nil {
		return nil
	}
	defer file.Close()
	if info, err := file.Stat(); err == nil && info.Size() > 1<<20 {
		file.Seek(-1<<20, io.SeekEnd)
	}
	var all []string
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 64<<10), 1<<20)
	for scanner.Scan() {
		all = append(all, scanner.Text())
	}
	if len(all) > lines {
		all = all[len(all)-lines:]
	}
	return all
}

// ── Commands ────────────────────────────────────────────────────────────────

func resolveInventoryModel(target string, args cliArgs) (modelInventoryEntry, error) {
	if target == "" {
		return modelInventoryEntry{}, cliError{"missing_model", "A model name, ID or path is required.", []string{"Run lmx model list to see local models."}, nil}
	}
	_, entries, err := modelInventory(args)
	if err != nil {
		return modelInventoryEntry{}, err
	}
	matches := matchInventory(entries, target)
	if len(matches) == 0 {
		return modelInventoryEntry{}, cliError{"model_not_found", "No local model matches " + target + ".", []string{"Run lmx model list (add --scan-dir <dir> for plain model folders)."}, nil}
	}
	if len(matches) > 1 {
		ids := []string{}
		for _, match := range matches {
			ids = append(ids, match.ID)
		}
		return modelInventoryEntry{}, cliError{"ambiguous_model", target + " matches several local models; pass an ID.", ids, nil}
	}
	return matches[0], nil
}

// matchInventory selects entries by exact ID, then name/path/repo, then ID prefix.
func matchInventory(entries []modelInventoryEntry, target string) []modelInventoryEntry {
	var matches []modelInventoryEntry
	for _, pass := range []func(modelInventoryEntry) bool{
		func(e modelInventoryEntry) bool { return e.ID == target },
		func(e modelInventoryEntry) bool {
			return strings.EqualFold(e.Name, target) || e.Path == target || strings.EqualFold(e.Repo, target)
		},
		func(e modelInventoryEntry) bool { return len(target) >= 6 && strings.HasPrefix(e.ID, target) },
	} {
		for _, entry := range entries {
			if pass(entry) {
				matches = append(matches, entry)
			}
		}
		if len(matches) > 0 {
			return matches
		}
	}
	return nil
}
