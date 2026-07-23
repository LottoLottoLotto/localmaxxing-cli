package main

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"
)

type ompWrapperHarness struct {
	wrapperPath string
	bashPath    string
	fakeBin     string
	modelsDB    string
	ompBin      string
	home        string
}

func TestOMPContainerShellExitContract(t *testing.T) {
	harness := newOMPWrapperHarness(t)
	providerError := `{"type":"message_end","message":{"role":"assistant","stopReason":"error","errorMessage":"provider unavailable"}}` + "\n"

	cases := []struct {
		name       string
		trace      string
		ompStatus  int
		wantStatus int
		wantOutput string
	}{
		{
			name:       "provider error before tool execution is fatal",
			trace:      providerError,
			ompStatus:  0,
			wantStatus: 1,
			wantOutput: "omp failed before tool execution: provider unavailable",
		},
		{
			name:       "same provider error after tool execution remains verifier eligible",
			trace:      `{"type":"tool_execution_end","toolName":"bash"}` + "\n" + providerError,
			ompStatus:  17,
			wantStatus: 0,
		},
		{name: "timeout 124 remains verifier eligible", ompStatus: 124, wantStatus: 0},
		{name: "timeout 137 remains verifier eligible", ompStatus: 137, wantStatus: 0},
		{name: "ordinary OMP failure propagates", trace: "{}\n", ompStatus: 23, wantStatus: 23},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			status, dockerArgv, output := runOMPWrapper(t, harness, tc.trace, tc.ompStatus)
			if status != tc.wantStatus {
				t.Fatalf("wrapper exit status = %d, want %d; output:\n%s", status, tc.wantStatus, output)
			}
			if tc.wantOutput != "" && !strings.Contains(output, tc.wantOutput) {
				t.Fatalf("wrapper output does not contain %q:\n%s", tc.wantOutput, output)
			}
			assertOMPRestrictedInvocation(t, dockerArgv)
		})
	}
}

func newOMPWrapperHarness(t *testing.T) ompWrapperHarness {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("bundled OMP wrapper requires a POSIX shell")
	}
	bashPath, err := exec.LookPath("bash")
	if err != nil {
		t.Skipf("bash unavailable: %v", err)
	}
	pythonPath, err := exec.LookPath("python3")
	if err != nil {
		t.Skipf("python3 unavailable: %v", err)
	}
	_, testFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("locate wrapper contract test")
	}

	root := t.TempDir()
	fakeBin := filepath.Join(root, "bin")
	if err := os.Mkdir(fakeBin, 0o755); err != nil {
		t.Fatalf("create fake binary directory: %v", err)
	}
	dockerPath := filepath.Join(fakeBin, "docker")
	writeOMPWrapperExecutable(t, dockerPath, `#!/bin/sh
set -eu
printf '%s\0' "$@" >> "$FAKE_DOCKER_ARGV"
case " $* " in
  *" LLAMA_CPP_BASE_URL="*)
    cat "$FAKE_OMP_TRACE"
    exit "$FAKE_OMP_STATUS"
    ;;
esac
exit 0
`)
	writeOMPWrapperExecutable(t, filepath.Join(fakeBin, "timeout"), `#!/bin/sh
set -eu
if [ "$1" = "-k" ]; then
  shift 2
fi
shift
exec "$@"
`)
	writeOMPWrapperExecutable(t, filepath.Join(fakeBin, "trace-filter"), `#!/bin/sh
set -eu
cat > "$1"
exit "${FAKE_FILTER_STATUS:-0}"
`)
	ompBin := filepath.Join(root, "omp")
	writeOMPWrapperExecutable(t, ompBin, "#!/bin/sh\nexit 0\n")

	modelsDB := filepath.Join(root, "models.db")
	createDB := exec.Command(pythonPath, "-c", `import sqlite3, sys
db = sqlite3.connect(sys.argv[1])
db.execute("create table model_cache (provider_id text, models text)")
db.commit()
db.close()
`, modelsDB)
	if output, err := createDB.CombinedOutput(); err != nil {
		t.Fatalf("create minimal OMP models database: %v\n%s", err, output)
	}

	return ompWrapperHarness{
		wrapperPath: filepath.Join(filepath.Dir(testFile), "..", "..", "examples", "agents", "omp-container-shell.sh"),
		bashPath:    bashPath,
		fakeBin:     fakeBin,
		modelsDB:    modelsDB,
		ompBin:      ompBin,
		home:        filepath.Join(root, "home"),
	}
}

func runOMPWrapper(t *testing.T, harness ompWrapperHarness, trace string, ompStatus int) (int, []string, string) {
	t.Helper()
	runDir := t.TempDir()
	instructionPath := filepath.Join(runDir, "instruction.txt")
	if err := os.WriteFile(instructionPath, []byte("Solve the task.\n"), 0o600); err != nil {
		t.Fatalf("write instruction: %v", err)
	}
	tracePath := filepath.Join(runDir, "fake-omp-trace.jsonl")
	if err := os.WriteFile(tracePath, []byte(trace), 0o600); err != nil {
		t.Fatalf("write fake OMP trace: %v", err)
	}
	dockerArgvPath := filepath.Join(runDir, "docker.argv")

	command := exec.Command(harness.bashPath, harness.wrapperPath)
	command.Env = ompWrapperEnvironment(map[string]string{
		"PATH":                           harness.fakeBin + string(os.PathListSeparator) + os.Getenv("PATH"),
		"HOME":                           harness.home,
		"LMX_TERMINAL_CONTAINER":         "fake-container",
		"LMX_TERMINAL_TASK_ID":           "wrapper-contract",
		"LMX_TERMINAL_INSTRUCTION_FILE":  instructionPath,
		"LMX_TERMINAL_TRACE_DIR":         runDir,
		"LMX_TERMINAL_AGENT_TIMEOUT_SEC": "1",
		"LMX_TERMINAL_MODEL":             "fake-model",
		"OMP_BIN":                        harness.ompBin,
		"OMP_MODELS_DB":                  harness.modelsDB,
		"OMP_CONFIG":                     filepath.Join(runDir, "absent-config.yml"),
		"OMP_TRACE_FILTER":               filepath.Join(harness.fakeBin, "trace-filter"),
		"FAKE_DOCKER_ARGV":               dockerArgvPath,
		"FAKE_OMP_TRACE":                 tracePath,
		"FAKE_OMP_STATUS":                strconv.Itoa(ompStatus),
	})
	output, err := command.CombinedOutput()
	status := 0
	if err != nil {
		exitError, ok := err.(*exec.ExitError)
		if !ok {
			t.Fatalf("execute bundled OMP wrapper: %v\n%s", err, output)
		}
		status = exitError.ExitCode()
	}
	encodedArgv, err := os.ReadFile(dockerArgvPath)
	if err != nil {
		t.Fatalf("read captured docker argv: %v\n%s", err, output)
	}
	parts := bytes.Split(encodedArgv, []byte{0})
	argv := make([]string, 0, len(parts))
	for _, part := range parts {
		if len(part) != 0 {
			argv = append(argv, string(part))
		}
	}
	return status, argv, string(output)
}

func ompWrapperEnvironment(overrides map[string]string) []string {
	environment := make([]string, 0, len(os.Environ())+len(overrides))
	for _, entry := range os.Environ() {
		key, _, _ := strings.Cut(entry, "=")
		if _, replaced := overrides[key]; !replaced {
			environment = append(environment, entry)
		}
	}
	for key, value := range overrides {
		environment = append(environment, key+"="+value)
	}
	return environment
}

func assertOMPRestrictedInvocation(t *testing.T, dockerArgv []string) {
	t.Helper()
	var commandText string
	for _, argument := range dockerArgv {
		if strings.Contains(argument, "exec /tmp/localmaxxing-omp -p") {
			if commandText != "" {
				t.Fatal("docker received more than one OMP command")
			}
			commandText = argument
		}
	}
	if commandText == "" {
		t.Fatalf("docker did not receive the OMP command; argv = %q", dockerArgv)
	}

	tokens := strings.Fields(commandText)
	withoutContinuations := make([]string, 0, len(tokens))
	for _, token := range tokens {
		if token != `\` {
			withoutContinuations = append(withoutContinuations, token)
		}
	}
	toolsIndex, noExtensionsIndex := -1, -1
	toolsCount, noExtensionsCount := 0, 0
	for index, token := range withoutContinuations {
		switch token {
		case "--tools":
			toolsCount++
			toolsIndex = index
		case "--no-extensions":
			noExtensionsCount++
			noExtensionsIndex = index
		case "--extensions":
			t.Fatal("OMP invocation enables extensions")
		}
	}
	if toolsCount != 1 || noExtensionsCount != 1 {
		t.Fatalf("OMP invocation contains --tools %d times and --no-extensions %d times, want exactly once each", toolsCount, noExtensionsCount)
	}
	if toolsIndex+2 != noExtensionsIndex || withoutContinuations[toolsIndex+1] != "bash" {
		t.Fatalf("OMP invocation does not retain exact adjacent --tools bash --no-extensions restriction: %q", commandText)
	}
}

func writeOMPWrapperExecutable(t *testing.T, path, content string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(content), 0o755); err != nil {
		t.Fatalf("write executable %s: %v", filepath.Base(path), err)
	}
}
