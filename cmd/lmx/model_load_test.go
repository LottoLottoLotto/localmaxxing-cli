package main

import (
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

// fakeSubstrate stands in for the substrate CLI: it keeps registered models in a
// JSON state file, records every invocation, and answers with --json output.
const fakeSubstrate = `#!/usr/bin/env python3
import json, os, sys
state_path = os.environ["FAKE_SUBSTRATE_STATE"]
state = json.load(open(state_path)) if os.path.exists(state_path) else {"models": [], "calls": [], "instances": []}
argv = [a for a in sys.argv[1:] if a != "--json"]
state["calls"].append(sys.argv[1:])
def save():
    json.dump(state, open(state_path, "w"))
def fail(code, message):
    save(); print(json.dumps({"code": code, "error": message})); sys.exit(3)
cmd = argv[:2]
if argv[:2] == ["model", "list"]:
    print(json.dumps(state["models"]))
elif argv[:2] == ["model", "add"]:
    name, source = argv[2], argv[3]
    force = "--force" in argv
    if any(m["name"] == name for m in state["models"]) and not force:
        fail("conflict", "model %s already registered" % name)
    state["models"] = [m for m in state["models"] if m["name"] != name] + [{"name": name, "source": source}]
    print(json.dumps({"name": name}))
elif argv[:2] == ["model", "rm"]:
    state["models"] = [m for m in state["models"] if m["name"] != argv[2]]
    print("{}")
elif argv[:1] == ["ps"]:
    print(json.dumps(state["instances"]))
elif argv[:1] == ["stop"]:
    print(json.dumps({"stopped": argv[1:]}))
elif argv[:2] == ["service", "status"]:
    print(json.dumps({"LoadState": "not-found", "ActiveState": "inactive"}))
else:
    fail("usage", "unexpected " + " ".join(argv))
save()
`

type fakeGateway struct {
	mu        sync.Mutex
	ensured   []string
	stopped   []string
	engine    string
	baseURL   string
	modelPath string
}

func (g *fakeGateway) handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) { w.Write([]byte("ok")) })
	mux.HandleFunc("POST /substrate/v1/ensure", func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Model string `json:"model"`
		}
		json.NewDecoder(r.Body).Decode(&body)
		g.mu.Lock()
		g.ensured = append(g.ensured, body.Model)
		g.mu.Unlock()
		json.NewEncoder(w).Encode(map[string]any{"instance": map[string]any{"id": body.Model + "." + g.engine, "name": body.Model, "engine": g.engine, "base_url": g.baseURL, "model": g.modelPath}, "status": "ready"})
	})
	mux.HandleFunc("GET /substrate/v1/instances", func(w http.ResponseWriter, r *http.Request) {
		g.mu.Lock()
		defer g.mu.Unlock()
		list := []any{}
		for _, name := range g.ensured {
			list = append(list, map[string]any{"status": "ready", "instance": map[string]any{"id": name + "." + g.engine, "name": name, "engine": g.engine}})
		}
		json.NewEncoder(w).Encode(list)
	})
	mux.HandleFunc("POST /substrate/v1/stop", func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			ID string `json:"id"`
		}
		json.NewDecoder(r.Body).Decode(&body)
		g.mu.Lock()
		g.stopped = append(g.stopped, body.ID)
		g.mu.Unlock()
		json.NewEncoder(w).Encode(map[string]any{"stopped": body.ID})
	})
	return mux
}

type substrateTestEnv struct {
	args    cliArgs
	state   string
	gateway *fakeGateway
	server  *httptest.Server
}

func newSubstrateTestEnv(t *testing.T) *substrateTestEnv {
	t.Helper()
	if _, err := exec.LookPath("python3"); err != nil {
		t.Skip("python3 is required for the fake substrate")
	}
	args := modelStoreTestArgs(t)
	dir := t.TempDir()
	bin := filepath.Join(dir, "substrate")
	if err := os.WriteFile(bin, []byte(fakeSubstrate), 0o755); err != nil {
		t.Fatal(err)
	}
	state := filepath.Join(dir, "state.json")
	t.Setenv("LMX_SUBSTRATE_BIN", bin)
	t.Setenv("FAKE_SUBSTRATE_STATE", state)
	t.Setenv("SUBSTRATE_API_KEY", "")
	gateway := &fakeGateway{engine: "llama.cpp", baseURL: "http://127.0.0.1:41000"}
	server := httptest.NewServer(gateway.handler())
	t.Cleanup(server.Close)
	t.Setenv("LMX_GATEWAY_URL", server.URL)
	args.flags["quiet"] = true
	return &substrateTestEnv{args: args, state: state, gateway: gateway, server: server}
}

func (e *substrateTestEnv) calls(t *testing.T) [][]string {
	t.Helper()
	var state struct {
		Calls [][]string `json:"calls"`
	}
	data, _ := os.ReadFile(e.state)
	json.Unmarshal(data, &state)
	return state.Calls
}

func (e *substrateTestEnv) addCalls(t *testing.T) [][]string {
	var adds [][]string
	for _, call := range e.calls(t) {
		if len(call) > 1 && call[0] == "model" && call[1] == "add" {
			adds = append(adds, call)
		}
	}
	return adds
}

// A local model is registered with substrate (lmx options mapped to its flags,
// engine arguments after --) and started by the gateway; loading it again
// reuses the registration, and a different model cannot take the same name.
func TestModelLoadRegistersAndEnsuresThroughGateway(t *testing.T) {
	env := newSubstrateTestEnv(t)
	scan := t.TempDir()
	os.MkdirAll(filepath.Join(scan, "tiny-gguf"), 0o755)
	model := filepath.Join(scan, "tiny-gguf", "tiny-Q4_K_M.gguf")
	writeTestGGUF(t, model, [][]uint64{{4}})
	env.args.opts["scan-dir"] = scan
	env.args.opts["engine"] = "llama.cpp"
	env.args.opts["context-length"] = "8192"
	env.args.opts["server-args"] = `["--seed","1"]`
	if err := handleModelLoad("tiny-gguf/tiny-Q4_K_M", env.args); err != nil {
		t.Fatal(err)
	}
	adds := env.addCalls(t)
	want := []string{"model", "add", "tiny-q4_k_m", model, "--ctx", "8192", "--engine", "llama.cpp", "--json", "--", "--seed", "1"}
	if len(adds) != 1 || strings.Join(adds[0], " ") != strings.Join(want, " ") {
		t.Fatalf("substrate model add = %v, want %v", adds, want)
	}
	if len(env.gateway.ensured) != 1 || env.gateway.ensured[0] != "tiny-q4_k_m" {
		t.Fatalf("gateway ensure = %v", env.gateway.ensured)
	}

	plain := cliArgs{opts: map[string]string{"scan-dir": scan, "models-dir": env.args.opts["models-dir"]}, flags: map[string]bool{"quiet": true}}
	if err := handleModelLoad("tiny-gguf/tiny-Q4_K_M", plain); err != nil {
		t.Fatal(err)
	}
	plain.flags["json"] = true
	if err := handleModelLoaded(plain); err != nil {
		t.Fatalf("model loaded through the gateway API: %v", err)
	}
	if len(env.addCalls(t)) != 1 || len(env.gateway.ensured) != 2 {
		t.Fatalf("reload re-registered: adds=%d ensures=%d", len(env.addCalls(t)), len(env.gateway.ensured))
	}

	other := filepath.Join(scan, "tiny-gguf", "tiny-Q8_0.gguf")
	writeTestGGUF(t, other, [][]uint64{{4}})
	clash := cliArgs{opts: map[string]string{"scan-dir": scan, "models-dir": env.args.opts["models-dir"], "name": "tiny-q4_k_m"}, flags: map[string]bool{"quiet": true}}
	var failure cliError
	if err := handleModelLoad("tiny-gguf/tiny-Q8_0", clash); !errors.As(err, &failure) || failure.Code != "name_in_use" {
		t.Fatalf("different model under a taken name: %v", err)
	}
}

// Unload stops through the gateway (in-flight requests finish first); --forget
// also unregisters so the gateway cannot start it again on demand.
func TestModelUnloadThroughGateway(t *testing.T) {
	env := newSubstrateTestEnv(t)
	env.args.positional = []string{"model", "unload", "qwen3-0.6b"}
	env.args.flags["forget"] = true
	if err := handleModelUnload("qwen3-0.6b", env.args); err != nil {
		t.Fatal(err)
	}
	if len(env.gateway.stopped) != 1 || env.gateway.stopped[0] != "qwen3-0.6b" {
		t.Fatalf("gateway stop = %v", env.gateway.stopped)
	}
	forgot := false
	for _, call := range env.calls(t) {
		forgot = forgot || strings.HasPrefix(strings.Join(call, " "), "model rm qwen3-0.6b")
	}
	if !forgot {
		t.Fatalf("--forget did not unregister: %v", env.calls(t))
	}
}

// Endpoint commands without --base-url go through the running gateway: the
// served model is the gateway model registered for --hf-id, made ready first,
// and the engine comes from its instance. Matrix runs get the instance's own
// URL; local-mode speed tests and explicit --base-url are left alone.
func TestGatewayDefaultRoutesEndpointCommands(t *testing.T) {
	env := newSubstrateTestEnv(t)
	env.gateway.engine, env.gateway.baseURL = "vllm", "http://127.0.0.1:41001"
	hub := filepath.Join(t.TempDir(), "hub", "models--Qwen--Qwen3-0.6B", "snapshots", strings.Repeat("a", 40))
	state := map[string]any{"models": []any{map[string]any{"name": "qwen3-0.6b", "source": hub}}, "calls": []any{}, "instances": []any{}}
	data, _ := json.Marshal(state)
	os.WriteFile(env.state, data, 0o600)

	run := func(positional []string, opts map[string]string) cliArgs {
		args := cliArgs{positional: positional, opts: opts, flags: map[string]bool{"quiet": true}}
		if err := applyGatewayDefault(&args); err != nil {
			t.Fatal(err)
		}
		return args
	}
	speed := run([]string{"speed-test", "run"}, map[string]string{"mode": "remote", "hf-id": "Qwen/Qwen3-0.6B"})
	if opt(speed, "base-url") != env.server.URL || opt(speed, "served-model") != "qwen3-0.6b" || opt(speed, "engine") != "vllm" {
		t.Fatalf("speed test routed to %q as %q on %q", opt(speed, "base-url"), opt(speed, "served-model"), opt(speed, "engine"))
	}
	matrix := run([]string{"speed-test", "matrix", "run"}, map[string]string{"hf-id": "Qwen/Qwen3-0.6B"})
	if opt(matrix, "base-url") != "http://127.0.0.1:41001" {
		t.Fatalf("matrix run measured %q, want the engine instance", opt(matrix, "base-url"))
	}
	if local := run([]string{"speed-test", "run", "llama.cpp"}, map[string]string{"hf-id": "Qwen/Qwen3-0.6B"}); opt(local, "base-url") != "" {
		t.Fatal("local-mode speed test was routed to the gateway")
	}
	if explicit := run([]string{"speed-test", "run"}, map[string]string{"mode": "remote", "base-url": "http://elsewhere:8000", "hf-id": "Qwen/Qwen3-0.6B"}); opt(explicit, "base-url") != "http://elsewhere:8000" {
		t.Fatal("explicit --base-url was overridden")
	}
	args := cliArgs{positional: []string{"speed-test", "run"}, opts: map[string]string{"mode": "remote", "hf-id": "Unknown/Model"}, flags: map[string]bool{"quiet": true}}
	var failure cliError
	if err := applyGatewayDefault(&args); !errors.As(err, &failure) || failure.Code != "gateway_model_not_found" {
		t.Fatalf("unregistered model: %v", err)
	}
	if len(env.gateway.ensured) != 2 {
		t.Fatalf("gateway ensured %v, want the model readied before each routed run", env.gateway.ensured)
	}
}

func TestGatewayURLFromSubstrateConfig(t *testing.T) {
	home := t.TempDir()
	t.Setenv("LMX_GATEWAY_URL", "")
	t.Setenv("SUBSTRATE_HOME", home)
	if got := gatewayURL(cliArgs{}); got != defaultGatewayURL {
		t.Fatalf("no config: %q", got)
	}
	writeTestJSON(t, filepath.Join(home, "config", "placeholder.json"), map[string]any{})
	config := "logging:\n    port: 1\ngateway:\n    host: 0.0.0.0   # all interfaces\n    port: 8800\n    placement: memory\nmodels: []\n"
	if err := os.WriteFile(filepath.Join(home, "config", "config.yaml"), []byte(config), 0o600); err != nil {
		t.Fatal(err)
	}
	if got := gatewayURL(cliArgs{}); got != "http://127.0.0.1:8800" {
		t.Fatalf("config gateway = %q", got)
	}
	t.Setenv("LMX_GATEWAY_URL", "http://gpu-box:7777/v1")
	if got := gatewayURL(cliArgs{}); got != "http://gpu-box:7777" {
		t.Fatalf("env override = %q", got)
	}
}
