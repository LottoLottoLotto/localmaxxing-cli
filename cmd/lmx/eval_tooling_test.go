package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestTenstorrentBoardsNameAndMemory(t *testing.T) {
	hw := map[string]any{}
	if !applyTenstorrentBoards(hw, []string{"p150a", "P150B", "n300"}) {
		t.Fatal("no boards detected")
	}
	if hw["gpuName"] != "Tenstorrent P150" || hw["gpuCount"] != 2 || hw["vramGb"] != 32.0 || hw["totalVramGb"] != 64.0 {
		t.Fatalf("two P150 and an N300: %v (the most common board is described)", hw)
	}
	hw = map[string]any{}
	if !applyTenstorrentBoards(hw, []string{"n150"}) || hw["gpuName"] != "Tenstorrent N150" || hw["vramGb"] != 12.0 {
		t.Fatalf("n150: %v", hw)
	}
	if applyTenstorrentBoards(map[string]any{}, []string{"unknown-board"}) {
		t.Fatal("an unknown board type was described")
	}
}

func TestResolveEvalHardware(t *testing.T) {
	dir := t.TempDir()
	runDir := filepath.Join(dir, "run")
	explicit := filepath.Join(dir, "hw.json")
	os.WriteFile(explicit, []byte(`{"hwClass":"DISCRETE_GPU","gpuName":"Tenstorrent P150","gpuCount":1,"vramGb":32}`), 0o644)
	args := cliArgs{opts: map[string]string{}, flags: map[string]bool{"quiet": true}}

	if _, err := resolveEvalHardware(args, "http://gpu-box.example:8000/v1", ""); err == nil || !strings.Contains(err.Error(), "hardware") {
		t.Fatalf("a remote endpoint must not get this machine's hardware: %v", err)
	}
	if !isLoopbackURL("http://127.0.0.1:8000/v1") || !isLoopbackURL("http://localhost:1") || isLoopbackURL("http://10.0.0.2:8000") {
		t.Fatal("isLoopbackURL")
	}

	args.opts["hardware"] = explicit
	hw, err := resolveEvalHardware(args, "http://gpu-box.example:8000/v1", runDir)
	if err != nil || asObject(hw)["gpuName"] != "Tenstorrent P150" {
		t.Fatalf("--hardware: %v %v", hw, err)
	}
	if _, err := os.Stat(filepath.Join(runDir, "hardware.json")); err != nil {
		t.Fatal("the run directory must keep the hardware a later submit uses")
	}
	delete(args.opts, "hardware")
	hw, err = resolveEvalHardware(args, "http://gpu-box.example:8000/v1", runDir)
	if err != nil || asObject(hw)["gpuName"] != "Tenstorrent P150" {
		t.Fatalf("a resumed run reuses its saved hardware, even for a remote endpoint: %v %v", hw, err)
	}
}

func TestDatasetRunners(t *testing.T) {
	cases := map[evalDataset]string{
		{Slug: "terminal-bench-4-0", Category: "agentic"}:                     "terminal",
		{Slug: "crudbench-localmaxxing-v2", Category: "software-engineering"}: "terminal",
		{Slug: "crud-bench", Category: "agentic"}:                             "terminal",
		{Slug: "gsm8k", Category: "Reasoning"}:                                "shard",
		{Slug: "humaneval-plus", Category: "Coding"}:                          "shard",
	}
	for d, want := range cases {
		if got := d.runner(); got != want {
			t.Errorf("%s: runner %s, want %s", d.Slug, got, want)
		}
	}
}

func TestRecordedRunOptionsAndSubmitGuards(t *testing.T) {
	args := cliArgs{opts: map[string]string{"api-key": "bhk_secret", "model-api-key": "sk-x", "base-url": "http://127.0.0.1:8000",
		"model": "org/m", "max-turns": "30", "run-dir": "/x", "out": "o.json"},
		flags: map[string]bool{"submit": true, "json": true, "oracle": false, "no-network": true}}
	rec := replayableTerminalOptions(args)
	opts := rec["opts"].(map[string]string)
	if opts["api-key"] != "" || opts["model-api-key"] != "" || opts["run-dir"] != "" || opts["out"] != "" {
		t.Fatalf("secrets and per-invocation options must not be recorded: %v", opts)
	}
	if opts["max-turns"] != "30" || opts["base-url"] == "" {
		t.Fatalf("run-defining options must be recorded: %v", opts)
	}
	if flags := rec["flags"].([]string); len(flags) != 1 || flags[0] != "no-network" {
		t.Fatalf("flags %v: only run-defining flags that are on", flags)
	}

	root := t.TempDir()
	writeJSON(filepath.Join(root, "lmx-run.json"), map[string]any{"dataset": "terminal-bench-4-0", "model": "org/m", "options": rec})
	writeJSON(filepath.Join(root, "run.json"), map[string]any{"version": terminalLiveCheckpointVersion, "state": "interrupted", "dataset": "terminal-bench-4-0", "shardIndex": 2})
	err := submitRecordedTerminalRun(cliArgs{opts: map[string]string{}, flags: map[string]bool{"quiet": true}}, root)
	if err == nil || !strings.Contains(err.Error(), "not finished") {
		t.Fatalf("an unfinished run must not be submitted: %v", err)
	}
	writeJSON(filepath.Join(root, "run.json"), map[string]any{"version": terminalLiveCheckpointVersion, "state": "submitted", "dataset": "terminal-bench-4-0", "shardIndex": 2})
	if err := submitRecordedTerminalRun(cliArgs{opts: map[string]string{}, flags: map[string]bool{"quiet": true}}, root); err == nil || !strings.Contains(err.Error(), "already submitted") {
		t.Fatalf("a submitted run must not be submitted again: %v", err)
	}
}

func TestHardwareMayBeAddedToARecordedRunButNotChanged(t *testing.T) {
	_, none, _ := terminalLiveCheckpointFingerprint(map[string]any{"hardware": nil, "model": "m", "shardIndex": 2})
	_, p150, _ := terminalLiveCheckpointFingerprint(map[string]any{"hardware": map[string]any{"gpuName": "Tenstorrent P150"}, "model": "m", "shardIndex": 2})
	_, rtx, _ := terminalLiveCheckpointFingerprint(map[string]any{"hardware": map[string]any{"gpuName": "NVIDIA GeForce RTX 3090"}, "model": "m", "shardIndex": 2})
	_, otherModel, _ := terminalLiveCheckpointFingerprint(map[string]any{"hardware": map[string]any{"gpuName": "Tenstorrent P150"}, "model": "x", "shardIndex": 2})
	if !terminalHardwareAddedOnly(none, p150) {
		t.Fatal("a run recorded without hardware must accept a description at submit")
	}
	if terminalHardwareAddedOnly(p150, rtx) {
		t.Fatal("recorded hardware must not be replaced")
	}
	if terminalHardwareAddedOnly(none, otherModel) {
		t.Fatal("adding hardware must not excuse any other change")
	}
}

func TestForegroundRunStatusFollowsItsLock(t *testing.T) {
	runDir := t.TempDir()
	writeJSON(filepath.Join(runDir, "run.json"), terminalLiveCheckpoint{Version: terminalLiveCheckpointVersion, State: "running", Dataset: "d", TaskIDs: []string{"a"}})
	lock, err := os.OpenFile(filepath.Join(runDir, "run.lock"), os.O_CREATE|os.O_RDWR, 0o644)
	if err != nil {
		t.Fatal(err)
	}
	defer lock.Close()
	if err := lockTerminalRunFile(lock); err != nil {
		t.Fatal(err)
	}
	if s, _ := terminalStatusSnapshot(runDir); s["state"] != "running" {
		t.Fatalf("a worker holding run.lock is running, got %v", s["state"])
	}
	if err := cleanupStoppedForegroundTerminalRun(cliArgs{opts: map[string]string{}, flags: map[string]bool{"quiet": true}}, runDir); err == nil {
		t.Fatal("cancel must refuse a live foreground worker")
	}
	unlockTerminalRunFile(lock)
	if s, _ := terminalStatusSnapshot(runDir); s["state"] != "interrupted" {
		t.Fatalf("a killed foreground worker (lock released) is interrupted, got %v", s["state"])
	}
}
