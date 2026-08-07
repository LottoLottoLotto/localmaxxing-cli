package main

import (
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestUnknownOptionIsRejectedWithSuggestion(t *testing.T) {
	err := runWithArgs(parseArgs([]string{"hardware", "validate", "hardware.json", "--gpu-nam", "RTX 3090"}))
	var cliErr cliError
	if !errorsAs(err, &cliErr) {
		t.Fatalf("error = %T %v, want cliError", err, err)
	}
	if cliErr.Code != "unknown_option" {
		t.Fatalf("code = %q, want unknown_option", cliErr.Code)
	}
	if len(cliErr.Hints) == 0 || !strings.Contains(cliErr.Hints[0], "--gpu-name") {
		t.Fatalf("hints = %#v, want gpu-name suggestion", cliErr.Hints)
	}
}

func TestVersionCommandsAreRecognized(t *testing.T) {
	for _, argv := range [][]string{{"version", "--json"}, {"--version", "--json"}} {
		if err := runWithArgs(parseArgs(argv)); err != nil {
			t.Fatalf("runWithArgs(%v): %v", argv, err)
		}
	}
}

func TestSpeedTestCommandReplacesBenchmarkAliases(t *testing.T) {
	if !knownTopLevel("speed-test") {
		t.Fatal("speed-test command is not registered")
	}
	for _, old := range []string{"benchmark", "bench"} {
		if knownTopLevel(old) {
			t.Fatalf("legacy command %q is still registered", old)
		}
	}
	schema := commandSchema()
	commands := schema["commands"].([]commandSchemaEntry)
	found := false
	for _, command := range commands {
		if command.Name == "benchmark" || strings.HasPrefix(command.Name, "benchmark ") {
			t.Fatalf("machine-readable schema still exposes %q", command.Name)
		}
		if command.Name == "speed-test" {
			found = true
		}
	}
	if !found {
		t.Fatal("machine-readable schema does not expose speed-test")
	}
}

func TestContextProjectionListsAndGetsSections(t *testing.T) {
	value := map[string]any{"hardwareOptions": map[string]any{"gpuNames": []any{"RTX 3090"}}, "requiredFields": map[string]any{}}
	listed, err := projectContext(value, parseArgs([]string{"context", "list"}))
	if err != nil {
		t.Fatalf("projectContext list: %v", err)
	}
	sections, ok := asObject(listed)["sections"].([]string)
	if !ok || len(sections) != 3 || sections[0] != "_cli" {
		t.Fatalf("sections = %#v", sections)
	}
	selected, err := projectContext(value, parseArgs([]string{"context", "get", "hardwareOptions.gpuNames"}))
	if err != nil {
		t.Fatalf("projectContext get: %v", err)
	}
	if got := anySlice(asObject(selected)["value"]); len(got) != 1 || stringValue(got[0]) != "RTX 3090" {
		t.Fatalf("selected value = %#v", got)
	}
}

func TestCompactJSONOutputUsesOneLine(t *testing.T) {
	path := filepath.Join(t.TempDir(), "context.json")
	args := parseArgs([]string{"context", "--out", path, "--compact", "--quiet"})
	if err := writeOrPrintJSON("context", args, map[string]any{"a": map[string]any{"b": 1}}); err != nil {
		t.Fatalf("writeOrPrintJSON: %v", err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read output: %v", err)
	}
	if strings.Count(string(data), "\n") != 1 {
		t.Fatalf("compact output = %q, want one line", data)
	}
}

func TestBenchmarkDryRunDoesNotPersistManagedRun(t *testing.T) {
	tmp := t.TempDir()
	out := filepath.Join(tmp, "plan.json")
	runsDir := filepath.Join(tmp, "runs")
	args := cliArgs{
		opts: map[string]string{
			"mode": "local", "hf-id": "Qwen/Qwen3-8B", "quantization": "Q4_K_M",
			"model-path": "/models/qwen.gguf", "out": out, "runs-dir": runsDir,
		},
		flags: map[string]bool{"dry-run": true, "quiet": true},
	}
	if err := handleBenchmark("run", "llama.cpp", args); err != nil {
		t.Fatalf("handleBenchmark: %v", err)
	}
	if _, err := os.Stat(out); err != nil {
		t.Fatalf("plan not written: %v", err)
	}
	if _, err := os.Stat(runsDir); !os.IsNotExist(err) {
		t.Fatalf("dry-run created managed runs directory: %v", err)
	}
	value, err := readJSON(out)
	if err != nil {
		t.Fatalf("read plan: %v", err)
	}
	feedback := asObject(asObject(value)["agentFeedback"])
	if persisted, _ := feedback["runPersisted"].(bool); persisted {
		t.Fatalf("feedback = %#v, want runPersisted false", feedback)
	}
}

func TestReadKeyFromStdinTrimsSecret(t *testing.T) {
	key, err := readKeyFromStdin(strings.NewReader("bhk_secret\n"))
	if err != nil {
		t.Fatalf("readKeyFromStdin: %v", err)
	}
	if key != "bhk_secret" {
		t.Fatalf("key = %q", key)
	}
}

func TestCommandSchemaIsVersioned(t *testing.T) {
	schema := commandSchema()
	if numberField(schema, "schemaVersion") != 1 {
		t.Fatalf("schemaVersion = %v", schema["schemaVersion"])
	}
	if len(schema["commands"].([]commandSchemaEntry)) == 0 || len(schema["globalOptionNames"].([]string)) == 0 {
		t.Fatalf("incomplete schema: %#v", schema)
	}
}

func TestJSONModeEmitsOneDocumentWithoutGuidanceProse(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	original := os.Stdout
	reader, writer, err := os.Pipe()
	if err != nil {
		t.Fatalf("os.Pipe: %v", err)
	}
	os.Stdout = writer
	code := runCLI([]string{"profile", "save", "smoke", "--mode", "local", "--json"})
	_ = writer.Close()
	os.Stdout = original
	data, readErr := io.ReadAll(reader)
	_ = reader.Close()
	if readErr != nil {
		t.Fatalf("read stdout: %v", readErr)
	}
	if code != 0 {
		t.Fatalf("runCLI exit = %d, output = %s", code, data)
	}
	var payload map[string]any
	if err := json.Unmarshal(data, &payload); err != nil {
		t.Fatalf("stdout was not one JSON document: %v\n%s", err, data)
	}
	if payload["event"] != "profile_saved" {
		t.Fatalf("payload = %#v", payload)
	}
}

func errorsAs(err error, target any) bool {
	return errors.As(err, target)
}
