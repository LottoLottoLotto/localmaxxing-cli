package main

import (
	"encoding/json"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// fakeEngineScript is a stand-in llama-server: it records argv and environment,
// then serves /v1/models after a short "load" (or exits at once when FAIL is set).
const fakeEngineScript = `#!/usr/bin/env python3
import json, os, sys, time, http.server
args = sys.argv[1:]
json.dump({"argv": args, "gpus": os.environ.get("CUDA_VISIBLE_DEVICES", ""), "extra": os.environ.get("FAKE_ENGINE_SETTING", "")}, open(os.environ["FAKE_ENGINE_RECORD"], "w"))
if os.environ.get("FAKE_ENGINE_FAIL"):
    print("fatal: cannot load model", flush=True)
    sys.exit(3)
port = int(args[args.index("--port") + 1])
alias = args[args.index("--alias") + 1]
time.sleep(0.5)
class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        body = json.dumps({"data": [{"id": alias}]}).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
http.server.HTTPServer(("127.0.0.1", port), H).serve_forever()
`

func modelLoadTestSetup(t *testing.T) (cliArgs, string, string) {
	t.Helper()
	if _, err := exec.LookPath("python3"); err != nil {
		t.Skip("python3 is required for the fake engine")
	}
	args := modelStoreTestArgs(t)
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("XDG_DATA_HOME", filepath.Join(home, "data"))
	t.Setenv("HF_TOKEN", "")
	t.Setenv("HF_HOME", filepath.Join(home, "hf"))
	engine := filepath.Join(home, "llama-server")
	if err := os.WriteFile(engine, []byte(fakeEngineScript), 0o755); err != nil {
		t.Fatal(err)
	}
	envFile := filepath.Join(home, "env.json")
	writeTestJSON(t, envFile, map[string]string{"FAKE_ENGINE_SETTING": "from-engine-config"})
	setArgs := cliArgs{positional: []string{"engines", "set", "llama.cpp", engine}, opts: map[string]string{"server-env-file": envFile}, flags: map[string]bool{"quiet": true}}
	if err := handleEngineConfig("set", "llama.cpp", setArgs); err != nil {
		t.Fatal(err)
	}
	scan := t.TempDir()
	os.MkdirAll(filepath.Join(scan, "tiny-gguf"), 0o755)
	writeTestGGUF(t, filepath.Join(scan, "tiny-gguf", "tiny-Q4_K_M.gguf"), [][]uint64{{4}})
	args.opts["scan-dir"] = scan
	record := filepath.Join(home, "engine-record.json")
	t.Setenv("FAKE_ENGINE_RECORD", record)
	return args, record, filepath.Join(scan, "tiny-gguf", "tiny-Q4_K_M.gguf")
}

// A GGUF loads on llama.cpp with the configured binary and environment, shows
// as loaded, is not started twice, and unload stops the process.
func TestModelLoadUnloadLifecycle(t *testing.T) {
	args, recordPath, modelPath := modelLoadTestSetup(t)
	args.opts["gpus"] = "0"
	args.opts["ready-timeout-seconds"] = "20"
	if err := handleModelLoad("tiny-gguf/tiny-Q4_K_M", args); err != nil {
		t.Fatal(err)
	}
	loaded, err := readLoadedModels()
	if err != nil || len(loaded) != 1 {
		t.Fatalf("loaded = %+v, %v", loaded, err)
	}
	m := loaded[0]
	t.Cleanup(func() { stopLoadedModel(m, time.Second) })
	if m.Engine != "llama.cpp" || m.Name != "tiny-q4_k_m" || m.ModelPath != modelPath || m.ReadyAt == "" || probeLoaded(m, time.Second) != "ready" {
		t.Fatalf("loaded record = %+v", m)
	}
	var started struct {
		Argv  []string `json:"argv"`
		GPUs  string   `json:"gpus"`
		Extra string   `json:"extra"`
	}
	data, _ := os.ReadFile(recordPath)
	json.Unmarshal(data, &started)
	joined := strings.Join(started.Argv, " ")
	if !strings.Contains(joined, "-m "+modelPath) || !strings.Contains(joined, "--alias tiny-Q4_K_M ") || !strings.Contains(joined, "-ngl 999") || started.GPUs != "0" || started.Extra != "from-engine-config" {
		t.Fatalf("engine started with %+v", started)
	}
	if e := inventoryByName(t, args)["tiny-gguf/tiny-Q4_K_M"]; len(e.LoadedAt) != 1 || e.LoadedAt[0] != m.BaseURL {
		t.Fatalf("model list does not show it loaded: %+v", e.LoadedAt)
	}
	// Loading the same model again reports the running server instead of a second copy.
	if err := handleModelLoad("tiny-gguf/tiny-Q4_K_M", args); err != nil {
		t.Fatal(err)
	}
	if again, _ := readLoadedModels(); len(again) != 1 || again[0].PID != m.PID {
		t.Fatalf("second load started another server: %+v", again)
	}
	if err := handleModelUnload(m.Name, args); err != nil {
		t.Fatal(err)
	}
	if m.alive() {
		t.Fatal("server still running after unload")
	}
	if left, _ := readLoadedModels(); len(left) != 0 {
		t.Fatalf("record survived unload: %+v", left)
	}
}

// An engine that dies while loading leaves nothing registered and reports its log.
func TestModelLoadReportsEngineFailure(t *testing.T) {
	args, _, _ := modelLoadTestSetup(t)
	t.Setenv("FAKE_ENGINE_FAIL", "1")
	err := handleModelLoad("tiny-gguf/tiny-Q4_K_M", args)
	var failure cliError
	if !errors.As(err, &failure) || failure.Code != "model_load_failed" || !strings.Contains(strings.Join(failure.Hints, "\n"), "fatal: cannot load model") {
		t.Fatalf("failure = %#v", err)
	}
	if left, _ := readLoadedModels(); len(left) != 0 {
		t.Fatalf("failed load left a record: %+v", left)
	}
}

func TestChooseLoadEngine(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	t.Setenv("PATH", t.TempDir()) // no host engines on PATH
	gguf := modelInventoryEntry{Name: "m", Format: "gguf"}
	weights := modelInventoryEntry{Name: "m", Format: "safetensors"}
	for _, tc := range []struct {
		entry     modelInventoryEntry
		requested string
		want      string
		fails     bool
	}{
		{gguf, "", "llama.cpp", false},
		{weights, "", "vllm", false},
		{weights, "sglang", "sglang", false},
		{gguf, "vllm", "", true},
		{weights, "llama.cpp", "", true},
		{modelInventoryEntry{Name: "m"}, "", "", true},
	} {
		got, err := chooseLoadEngine(tc.entry, tc.requested)
		if (err != nil) != tc.fails || got != tc.want {
			t.Errorf("%s/%q: got %q, %v", tc.entry.Format, tc.requested, got, err)
		}
	}
	// With only SGLang configured, safetensors go to SGLang.
	python := filepath.Join(t.TempDir(), "python")
	os.WriteFile(python, []byte("#!/bin/sh\n"), 0o755)
	if err := saveHostEngines(map[string]hostEngineConfig{"sglang": {Path: python}}); err != nil {
		t.Fatal(err)
	}
	if got, _ := chooseLoadEngine(weights, ""); got != "sglang" {
		t.Fatalf("safetensors with only SGLang available -> %q", got)
	}
}
