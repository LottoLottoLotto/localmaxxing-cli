package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

const terminalProcessRecordVersion = 2

type terminalProcessRecord struct {
	Version    int    `json:"version"`
	PID        int    `json:"pid"`
	Identity   string `json:"identity"`
	State      string `json:"state"`
	RunDir     string `json:"runDir"`
	EventsPath string `json:"eventsPath"`
	StdoutPath string `json:"stdoutPath"`
	StderrPath string `json:"stderrPath"`
	StartedAt  string `json:"startedAt"`
	UpdatedAt  string `json:"updatedAt"`
	FinishedAt string `json:"finishedAt,omitempty"`
	ExitCode   *int   `json:"exitCode,omitempty"`
	ErrorCode  string `json:"errorCode,omitempty"`
	Error      string `json:"error,omitempty"`
}

func terminalProcessPath(root string) string { return filepath.Join(root, "process.json") }
func terminalEventsPath(root string) string  { return filepath.Join(root, "events.jsonl") }
func terminalStdoutPath(root string) string  { return filepath.Join(root, "worker.stdout") }
func terminalStderrPath(root string) string  { return filepath.Join(root, "worker.stderr") }

func terminalArgsWithoutFlags(args cliArgs, names ...string) cliArgs {
	removed := make(map[string]bool, len(names))
	for _, name := range names {
		removed[name] = true
	}
	copied := cliArgs{opts: map[string]string{}, flags: map[string]bool{}, positional: append([]string(nil), args.positional...)}
	for key, value := range args.opts {
		if !removed[key] {
			copied.opts[key] = value
		}
	}
	for key, value := range args.flags {
		if !removed[key] {
			copied.flags[key] = value
		}
	}
	return copied
}

func terminalArgsToArgv(args cliArgs) []string {
	argv := append([]string(nil), args.positional...)
	keys := make([]string, 0, len(args.opts))
	for key := range args.opts {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		argv = append(argv, "--"+key, args.opts[key])
	}
	keys = keys[:0]
	for key, enabled := range args.flags {
		if enabled {
			keys = append(keys, key)
		}
	}
	sort.Strings(keys)
	for _, key := range keys {
		argv = append(argv, "--"+key)
	}
	return argv
}

func startDetachedTerminalEval(args cliArgs) error {
	runDir := opt(args, "run-dir")
	if runDir == "" {
		return cliError{"missing_option", "Detached terminal execution requires --run-dir.", []string{"Pass a unique durable --run-dir so status, logs, cancel, and resume address one job."}, nil}
	}
	if hasFlag(args, "dry-run") {
		return cliError{"invalid_option", "--detach cannot be combined with --dry-run.", []string{"Run preflight synchronously, then launch the validated command with --detach."}, nil}
	}
	root, err := filepath.Abs(runDir)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(root, 0o755); err != nil {
		return err
	}
	launchLock, err := os.OpenFile(filepath.Join(root, "launch.lock"), os.O_CREATE|os.O_RDWR, 0o644)
	if err != nil {
		return err
	}
	defer launchLock.Close()
	if err := lockTerminalRunFile(launchLock); err != nil {
		return cliError{"terminal_job_starting", "Another process is starting a worker in this run directory.", []string{"Retry status shortly, or choose another --run-dir."}, map[string]any{"runDir": root}}
	}
	defer unlockTerminalRunFile(launchLock)
	if existing, err := loadTerminalProcessRecord(root); err == nil && processMatchesIdentity(existing.PID, existing.Identity) {
		return cliError{"terminal_job_running", "This terminal run directory already has a live worker.", []string{"Inspect it with: lmx eval terminal status " + root, "Choose a new --run-dir for a separate rerun."}, map[string]any{"runDir": root, "pid": existing.PID}}
	}
	events, err := os.OpenFile(terminalEventsPath(root), os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return err
	}
	if err := events.Close(); err != nil {
		return err
	}
	stdout, err := os.OpenFile(terminalStdoutPath(root), os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return err
	}
	defer stdout.Close()
	stderr, err := os.OpenFile(terminalStderrPath(root), os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return err
	}
	defer stderr.Close()
	stdin, err := os.Open(os.DevNull)
	if err != nil {
		return err
	}
	defer stdin.Close()
	executable, err := os.Executable()
	if err != nil {
		return err
	}
	childArgs := terminalArgsWithoutFlags(args, "detach", "detached-child", "json", "print", "verbose", "quiet", "status-log")
	childArgs.flags["detached-child"] = true
	childArgs.flags["json-status"] = true
	childArgs.opts["status-log"] = terminalEventsPath(root)
	cmd := exec.Command(executable, terminalArgsToArgv(childArgs)...)
	configureDetachedCommand(cmd)
	cmd.Stdin = stdin
	cmd.Stdout = stdout
	cmd.Stderr = stderr
	cmd.Env = os.Environ()
	if cwd, err := os.Getwd(); err == nil {
		cmd.Dir = cwd
	}
	now := time.Now().UTC().Format(time.RFC3339)
	record := terminalProcessRecord{Version: terminalProcessRecordVersion, State: "starting", RunDir: root, EventsPath: terminalEventsPath(root), StdoutPath: terminalStdoutPath(root), StderrPath: terminalStderrPath(root), StartedAt: now, UpdatedAt: now}
	if err := writeTerminalJSONAtomic(terminalProcessPath(root), record); err != nil {
		return err
	}
	if err := cmd.Start(); err != nil {
		exitCode := 1
		record.State = "failed"
		record.ExitCode = &exitCode
		record.ErrorCode = "terminal_detach_failed"
		record.Error = err.Error()
		record.FinishedAt = time.Now().UTC().Format(time.RFC3339)
		record.UpdatedAt = record.FinishedAt
		_ = writeTerminalJSONAtomic(terminalProcessPath(root), record)
		return cliError{"terminal_detach_failed", "Could not start the detached terminal worker.", []string{"Check executable and run-directory permissions."}, map[string]any{"runDir": root, "error": err.Error()}}
	}
	record.PID = cmd.Process.Pid
	record.Identity, err = captureProcessIdentity(record.PID)
	if err != nil {
		failedPID := record.PID
		record.PID = 0
		exitCode := 1
		record.State = "failed"
		record.ExitCode = &exitCode
		record.ErrorCode = "terminal_detach_failed"
		record.Error = err.Error()
		record.FinishedAt = time.Now().UTC().Format(time.RFC3339)
		record.UpdatedAt = record.FinishedAt
		_ = writeTerminalJSONAtomic(terminalProcessPath(root), record)
		_ = cmd.Process.Release()
		return cliError{"terminal_detach_failed", "Could not capture the detached terminal worker identity.", []string{"Check process inspection permissions and retry."}, map[string]any{"runDir": root, "pid": failedPID, "error": err.Error()}}
	}
	record.State = "running"
	record.UpdatedAt = time.Now().UTC().Format(time.RFC3339)
	if err := writeTerminalJSONAtomic(terminalProcessPath(root), record); err != nil {
		_ = signalTerminalProcess(record, true)
		return err
	}
	if err := cmd.Process.Release(); err != nil {
		_ = signalTerminalProcess(record, true)
		return err
	}
	quotedRoot := shellQuote(root)
	fields := map[string]any{"pid": record.PID, "state": record.State, "runDir": root, "eventsPath": record.EventsPath, "stdoutPath": record.StdoutPath, "stderrPath": record.StderrPath, "statusCommand": "lmx eval terminal status " + quotedRoot + " --json", "logsCommand": "lmx eval terminal logs " + quotedRoot + " --follow", "cancelCommand": "lmx eval terminal cancel " + quotedRoot + " --json"}
	if hasFlag(args, "json-status") {
		printStatus(args, "terminal_eval_detached", fields)
		if hasFlag(args, "json") || hasFlag(args, "print") || hasFlag(args, "verbose") {
			printJSON(args, fields)
		}
	} else if hasFlag(args, "json") || hasFlag(args, "print") || hasFlag(args, "verbose") {
		printJSON(args, fields)
	} else {
		printInfo(args, "terminal_eval_detached", fields)
	}
	return nil
}

func awaitDetachedTerminalProcessRecord(args cliArgs) error {
	root, err := filepath.Abs(opt(args, "run-dir"))
	if err != nil || root == "" {
		return cliError{"terminal_process_record_missing", "Detached worker has no valid --run-dir.", nil, nil}
	}
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		record, readErr := loadTerminalProcessRecord(root)
		if readErr == nil && record.PID == os.Getpid() && processMatchesIdentity(record.PID, record.Identity) && (record.State == "running" || record.State == "cancel_requested") {
			return nil
		}
		time.Sleep(10 * time.Millisecond)
	}
	return cliError{"terminal_process_record_missing", "Detached worker could not confirm its process record.", []string{"Inspect process.json and retry the detached launch."}, map[string]any{"runDir": root, "pid": os.Getpid()}}
}

func finalizeDetachedTerminalChild(args cliArgs, runErr error) {
	root, err := filepath.Abs(opt(args, "run-dir"))
	if err != nil || root == "" {
		return
	}
	record, err := loadTerminalProcessRecord(root)
	if err != nil {
		return
	}
	if record.PID != os.Getpid() || !processMatchesIdentity(record.PID, record.Identity) {
		return
	}
	now := time.Now().UTC().Format(time.RFC3339)
	exitCode := 0
	previousState := record.State
	record.State = "succeeded"
	if runErr != nil {
		exitCode = 1
		record.ErrorCode, record.Error = cliErrorCodeText(runErr)
		if previousState == "cancel_requested" || record.ErrorCode == "terminal_cancelled" {
			record.State = "cancelled"
		} else {
			record.State = "failed"
		}
	}
	record.ExitCode = &exitCode
	record.FinishedAt = now
	record.UpdatedAt = now
	_ = writeTerminalJSONAtomic(terminalProcessPath(root), record)
}

func loadTerminalProcessRecord(root string) (terminalProcessRecord, error) {
	data, err := os.ReadFile(terminalProcessPath(root))
	if err != nil {
		return terminalProcessRecord{}, err
	}
	var record terminalProcessRecord
	if err := json.Unmarshal(data, &record); err != nil {
		return terminalProcessRecord{}, err
	}
	if record.Version != terminalProcessRecordVersion || record.RunDir == "" || record.PID < 0 || (record.PID == 0) != (record.Identity == "") {
		return terminalProcessRecord{}, fmt.Errorf("invalid terminal process record")
	}
	return record, nil
}

func loadTerminalLiveCheckpointFile(root string) (terminalLiveCheckpoint, error) {
	data, err := os.ReadFile(filepath.Join(root, "run.json"))
	if err != nil {
		return terminalLiveCheckpoint{}, err
	}
	var checkpoint terminalLiveCheckpoint
	if err := json.Unmarshal(data, &checkpoint); err != nil {
		return terminalLiveCheckpoint{}, err
	}
	return checkpoint, nil
}

func terminalLastCanonicalEventTime(root string) string {
	file, err := os.Open(terminalEventsPath(root))
	if err != nil {
		return ""
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil || info.Size() == 0 {
		return ""
	}
	const (
		activityReadChunk = int64(32 * 1024)
		maxActivityTail   = int64(4 * 1024 * 1024)
	)
	offset := info.Size()
	tail := []byte{}
	for offset > 0 && int64(len(tail)) < maxActivityTail {
		readSize := activityReadChunk
		if readSize > offset {
			readSize = offset
		}
		if remaining := maxActivityTail - int64(len(tail)); readSize > remaining {
			readSize = remaining
		}
		offset -= readSize
		chunk := make([]byte, readSize)
		n, err := file.ReadAt(chunk, offset)
		if err != nil && !errors.Is(err, io.EOF) {
			return ""
		}
		combined := make([]byte, n+len(tail))
		copy(combined, chunk[:n])
		copy(combined[n:], tail)
		tail = combined
		lines := bytes.Split(tail, []byte{'\n'})
		minimum := 0
		if offset > 0 {
			minimum = 1
		}
		for i := len(lines) - 2; i >= minimum; i-- {
			var event map[string]any
			if json.Unmarshal(bytes.TrimSpace(lines[i]), &event) != nil || stringValue(event["event"]) == "" {
				continue
			}
			candidate := stringValue(event["time"])
			if _, err := time.Parse(time.RFC3339Nano, candidate); err == nil {
				return candidate
			}
		}
	}
	return ""
}

func terminalStatusSnapshot(root string) (map[string]any, error) {
	root, err := filepath.Abs(root)
	if err != nil {
		return nil, err
	}
	process, processErr := loadTerminalProcessRecord(root)
	checkpoint, checkpointErr := loadTerminalLiveCheckpointFile(root)
	if processErr != nil && checkpointErr != nil {
		return nil, cliError{"terminal_job_missing", "No terminal run or process record exists in this directory.", []string{"Pass the --run-dir used by eval terminal run."}, map[string]any{"runDir": root}}
	}
	alive := false
	if processErr == nil {
		alive = processMatchesIdentity(process.PID, process.Identity)
	}
	state := checkpoint.State
	if state == "" && processErr == nil {
		state = process.State
	}
	if processErr == nil {
		switch {
		case process.State == "cancel_requested" && alive:
			state = "cancelling"
		case alive:
			state = "running"
		case process.State == "running" || process.State == "starting" || process.State == "cancel_requested":
			if state != "completed" && state != "submitted" {
				state = "interrupted"
			}
		case process.State == "failed" || process.State == "cancelled":
			state = process.State
		}
	}
	scored, passed, unscored := 0, 0, 0
	if checkpointErr == nil {
		for _, taskID := range checkpoint.TaskIDs {
			result, ok, err := loadTerminalLiveResult(root, taskID)
			if err != nil {
				return nil, err
			}
			if !ok {
				continue
			}
			if result.scored {
				scored++
				if result.pass {
					passed++
				}
			} else {
				unscored++
			}
		}
	}
	snapshot := map[string]any{"runDir": root, "state": state, "processRunning": alive, "tasksScored": scored, "tasksPassed": passed, "tasksUnscored": unscored, "eventsPath": terminalEventsPath(root), "resultPath": filepath.Join(root, "result.json")}
	lastActivityAt := terminalLastCanonicalEventTime(root)
	if lastActivityAt == "" && checkpointErr == nil {
		lastActivityAt = checkpoint.UpdatedAt
	}
	if lastActivityAt == "" && processErr == nil {
		lastActivityAt = process.UpdatedAt
	}
	if lastActivityAt != "" {
		snapshot["lastActivityAt"] = lastActivityAt
	}
	if checkpointErr == nil {
		snapshot["dataset"] = checkpoint.Dataset
		snapshot["shardIndex"] = checkpoint.ShardIndex
		snapshot["tasksTotal"] = len(checkpoint.TaskIDs)
		snapshot["tasksPersisted"] = len(checkpoint.CompletedTasks)
		if checkpoint.ActiveTasks == nil {
			snapshot["activeTasks"] = []terminalLiveActiveTask{}
		} else {
			snapshot["activeTasks"] = checkpoint.ActiveTasks
		}
		snapshot["createdAt"] = checkpoint.CreatedAt
		snapshot["updatedAt"] = checkpoint.UpdatedAt
	}
	if processErr == nil {
		snapshot["pid"] = process.PID
		snapshot["processState"] = process.State
		snapshot["startedAt"] = process.StartedAt
		snapshot["finishedAt"] = process.FinishedAt
		snapshot["exitCode"] = process.ExitCode
		snapshot["errorCode"] = process.ErrorCode
		snapshot["error"] = process.Error
		snapshot["stdoutPath"] = process.StdoutPath
		snapshot["stderrPath"] = process.StderrPath
	}
	return snapshot, nil
}

func terminalRunDirArg(args cliArgs) (string, error) {
	root := positional(args, 3)
	if root == "" {
		return "", cliError{"missing_option", "Terminal job command requires <run-dir>.", []string{"Pass the directory used with eval terminal run --run-dir."}, nil}
	}
	return root, nil
}

func terminalStatus(args cliArgs) error {
	root, err := terminalRunDirArg(args)
	if err != nil {
		return err
	}
	snapshot, err := terminalStatusSnapshot(root)
	if err != nil {
		return err
	}
	return writeOrPrintJSON("terminal_status", args, snapshot)
}

func terminalLogs(args cliArgs) error {
	root, err := terminalRunDirArg(args)
	if err != nil {
		return err
	}
	root, err = filepath.Abs(root)
	if err != nil {
		return err
	}
	file, err := os.Open(terminalEventsPath(root))
	if err != nil {
		return cliError{"terminal_logs_missing", "The terminal event log is not available.", []string{"Confirm the detached worker started and the run directory is correct."}, map[string]any{"runDir": root, "error": err.Error()}}
	}
	defer file.Close()
	reader := bufio.NewReader(file)
	var pending strings.Builder
	ctx, cancel := terminalSignalContext()
	defer cancel()
	for {
		line, readErr := reader.ReadString('\n')
		pending.WriteString(line)
		if strings.HasSuffix(line, "\n") {
			if _, err := io.WriteString(os.Stdout, pending.String()); err != nil {
				return err
			}
			pending.Reset()
		}
		if readErr == nil {
			continue
		}
		if !errors.Is(readErr, io.EOF) {
			return readErr
		}
		if !hasFlag(args, "follow") {
			return nil
		}
		snapshot, statusErr := terminalStatusSnapshot(root)
		if statusErr == nil && !terminalJobStateActive(stringValue(snapshot["state"])) {
			return nil
		}
		select {
		case <-ctx.Done():
			return nil
		case <-time.After(250 * time.Millisecond):
		}
	}
}

func terminalJobStateActive(state string) bool {
	return state == "starting" || state == "running" || state == "cancelling"
}

func terminalCancel(args cliArgs) error {
	root, err := terminalRunDirArg(args)
	if err != nil {
		return err
	}
	root, err = filepath.Abs(root)
	if err != nil {
		return err
	}
	record, err := loadTerminalProcessRecord(root)
	if err != nil {
		return cliError{"terminal_job_missing", "No detached terminal process record exists in this directory.", []string{"Only runs started with --detach can be cancelled by run directory."}, map[string]any{"runDir": root}}
	}
	alive := processMatchesIdentity(record.PID, record.Identity)
	if !alive {
		snapshot, statusErr := terminalStatusSnapshot(root)
		if statusErr != nil {
			return statusErr
		}
		snapshot["alreadyStopped"] = true
		return writeOrPrintJSON("terminal_cancel", args, snapshot)
	}
	force := hasFlag(args, "force")
	record.State = "cancel_requested"
	record.UpdatedAt = time.Now().UTC().Format(time.RFC3339)
	if err := writeTerminalJSONAtomic(terminalProcessPath(root), record); err != nil {
		return err
	}
	cancelEvent := map[string]any{"event": "terminal_cancel_requested", "time": time.Now().UTC().Format(time.RFC3339), "runDir": root, "pid": record.PID, "force": force}
	if err := appendTerminalStatusLogDurable(terminalEventsPath(root), cancelEvent); err != nil {
		return err
	}
	if err := signalTerminalProcess(record, force); err != nil {
		return cliError{"terminal_cancel_failed", "Could not signal the detached terminal worker.", []string{"Inspect the process identity and permissions, then retry with --force if necessary."}, map[string]any{"pid": record.PID, "runDir": root, "error": err.Error()}}
	}
	result := map[string]any{"runDir": root, "pid": record.PID, "state": map[bool]string{true: "force_kill_requested", false: "cancel_requested"}[force], "force": force}
	return writeOrPrintJSON("terminal_cancel", args, result)
}

func appendTerminalStatusLog(path string, payload map[string]any) error {
	return appendTerminalStatusLogRecord(path, payload, false)
}

func appendTerminalStatusLogDurable(path string, payload map[string]any) error {
	return appendTerminalStatusLogRecord(path, payload, true)
}

func appendTerminalStatusLogRecord(path string, payload map[string]any, durable bool) error {
	if path == "" {
		return nil
	}
	data, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	file, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return err
	}
	defer file.Close()
	if _, err := file.Write(append(data, '\n')); err != nil {
		return err
	}
	if durable {
		return file.Sync()
	}
	return nil
}

func signalTerminalProcess(record terminalProcessRecord, force bool) error {
	if !processMatchesIdentity(record.PID, record.Identity) {
		return os.ErrProcessDone
	}
	return signalDetachedProcess(record.PID, force)
}

func terminalCancelledError(_ context.Context) error {
	return cliError{"terminal_cancelled", "Terminal execution was cancelled.", []string{"Restart the exact command with --resume auto to reuse every scored task."}, nil}
}

func terminalProcessSummary(record terminalProcessRecord) string {
	return strings.TrimSpace(fmt.Sprintf("pid=%d state=%s", record.PID, record.State))
}
