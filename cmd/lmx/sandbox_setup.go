package main

import (
	"bytes"
	"embed"
	"encoding/json"
	"fmt"
	"io"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// The sandbox image sources ship inside the binary so an installed CLI can
// build the code-execution image without a repository checkout.
//
//go:embed sandbox
var sandboxFS embed.FS

const defaultSandboxImage = "lmx-sandbox"

func sandboxImage(args cliArgs) string {
	return firstNonEmpty(opt(args, "sandbox-image"), defaultSandboxImage)
}

// sandboxSetupCommand is the exact command that rebuilds the configured image,
// so failure hints can be copied verbatim.
func sandboxSetupCommand(args cliArgs) string {
	parts := []string{"lmx eval sandbox setup"}
	if runtime := opt(args, "sandbox-runtime"); runtime != "" {
		parts = append(parts, "--sandbox-runtime "+shellQuote(runtime))
	}
	if hasFlag(args, "sandbox-use-sudo") {
		parts = append(parts, "--sandbox-use-sudo")
	}
	if image := opt(args, "sandbox-image"); image != "" {
		parts = append(parts, "--sandbox-image "+shellQuote(image))
	}
	return strings.Join(parts, " ")
}

func handleEvalSandbox(action string, args cliArgs) error {
	switch action {
	case "setup":
		return setupSandbox(args)
	case "check":
		return checkSandbox(args)
	default:
		return cliError{"unknown_subcommand", "Unknown sandbox command: " + action, []string{"Use lmx eval sandbox setup to build the code-execution image, or lmx eval sandbox check to verify it."}, nil}
	}
}

func setupSandbox(args cliArgs) error {
	if opt(args, "sandbox-cmd") != "" {
		return cliError{"invalid_option", "eval sandbox setup builds the container image; --sandbox-cmd launchers are not built.", []string{"Drop --sandbox-cmd, or verify a custom launcher with lmx eval sandbox check --sandbox-cmd <cmd>."}, nil}
	}
	runtimeArgs := sandboxRuntimeArgs(args)
	image := sandboxImage(args)
	plan := map[string]any{"runtime": strings.Join(runtimeArgs, " "), "image": image, "command": strings.Join(append(append([]string{}, runtimeArgs...), "build", "--tag", image, "EMBEDDED_CONTEXT_DIR"), " ")}
	if hasFlag(args, "dry-run") {
		return writeOrPrintJSON("eval_sandbox_setup_plan", args, plan)
	}
	if _, err := exec.LookPath(runtimeArgs[0]); err != nil {
		return cliError{"sandbox_unavailable", fmt.Sprintf("Sandbox runtime %q was not found.", runtimeArgs[0]), []string{"Install Docker (or Podman) and retry, or select a runtime with --sandbox-runtime."}, err.Error()}
	}
	contextDir, err := os.MkdirTemp("", "lmx-sandbox-context-")
	if err != nil {
		return err
	}
	defer os.RemoveAll(contextDir)
	if err := writeSandboxContext(contextDir); err != nil {
		return err
	}
	argv := append(append([]string{}, runtimeArgs...), "build", "--tag", image, contextDir)
	printStatus(args, "eval_sandbox_build_start", plan)
	cmd := exec.Command(argv[0], argv[1:]...)
	var output bytes.Buffer
	// Build progress goes to stderr so --json stdout stays machine-readable.
	var sink io.Writer = &output
	if !hasFlag(args, "quiet") {
		sink = io.MultiWriter(&output, os.Stderr)
	}
	cmd.Stdout = sink
	cmd.Stderr = sink
	if err := cmd.Run(); err != nil {
		return cliError{"sandbox_setup_failed", fmt.Sprintf("Building sandbox image %q failed.", image), sandboxFailureHints(tailText(output.String(), 4000), args), map[string]any{"runtime": strings.Join(runtimeArgs, " "), "image": image, "error": err.Error()}}
	}
	result, err := smokeTestSandbox(args)
	if err != nil {
		return err
	}
	result["built"] = true
	return writeOrPrintJSON("eval_sandbox_ready", args, result)
}

func checkSandbox(args cliArgs) error {
	if err := preflightSandbox(args); err != nil {
		return err
	}
	result, err := smokeTestSandbox(args)
	if err != nil {
		return err
	}
	return writeOrPrintJSON("eval_sandbox_ready", args, result)
}

func writeSandboxContext(dir string) error {
	return fs.WalkDir(sandboxFS, "sandbox", func(path string, entry fs.DirEntry, err error) error {
		if err != nil || entry.IsDir() {
			return err
		}
		data, err := sandboxFS.ReadFile(path)
		if err != nil {
			return err
		}
		return os.WriteFile(filepath.Join(dir, strings.TrimPrefix(path, "sandbox/")), data, 0o644)
	})
}

// smokeTestSandbox runs one passing and one failing program through the exact
// launcher used by code evals; a sandbox that cannot distinguish them would
// silently corrupt scores.
func smokeTestSandbox(args cliArgs) (map[string]any, error) {
	cmd, snippet := sandboxCommand(args)
	cmd.Stdin = strings.NewReader(`{"question_id":"pass","program":"assert sum([1, 2]) == 3\n"}` + "\n" +
		`{"question_id":"fail","program":"assert sum([1, 2]) == 4\n"}` + "\n")
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return nil, cliError{"sandbox_unavailable", "The sandbox could not execute a test program.", sandboxFailureHints(stderr.String(), args), map[string]any{"command": snippet, "error": err.Error()}}
	}
	passed := map[string]bool{}
	for _, line := range strings.Split(strings.TrimSpace(stdout.String()), "\n") {
		var result struct {
			QuestionID string `json:"question_id"`
			Passed     bool   `json:"passed"`
		}
		if json.Unmarshal([]byte(line), &result) == nil {
			passed[result.QuestionID] = result.Passed
		}
	}
	pass, passSeen := passed["pass"]
	fail, failSeen := passed["fail"]
	if !passSeen || !failSeen || !pass || fail {
		return nil, cliError{"sandbox_unavailable", "The sandbox did not grade test programs correctly.", sandboxFailureHints(stderr.String(), args), map[string]any{"command": snippet, "stdout": tailText(stdout.String(), 2000)}}
	}
	return map[string]any{"command": snippet, "image": sandboxImage(args), "verified": "passing and failing test programs graded correctly"}, nil
}

func tailText(text string, limit int) string {
	if len(text) <= limit {
		return text
	}
	return text[len(text)-limit:]
}
