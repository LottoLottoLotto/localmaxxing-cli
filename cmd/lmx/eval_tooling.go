package main

import (
	"bufio"
	"fmt"
	"math"
	"net"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"
)

// ---- host hardware ----

// hostOSName is the distribution name ("Ubuntu 26.04 LTS") on Linux, else runtime.GOOS.
func hostOSName() string {
	if runtime.GOOS != "linux" {
		return runtime.GOOS
	}
	data, err := os.ReadFile("/etc/os-release")
	if err != nil {
		return runtime.GOOS
	}
	for _, line := range strings.Split(string(data), "\n") {
		if v, ok := strings.CutPrefix(line, "PRETTY_NAME="); ok {
			if v = strings.Trim(strings.TrimSpace(v), `"`); v != "" {
				return v
			}
		}
	}
	return runtime.GOOS
}

// hostRAMGb is installed memory as sold: Linux MemTotal rounded up to a multiple of 8 GB (the kernel
// reserves some), or macOS hw.memsize; 0 when unknown.
func hostRAMGb() int {
	switch runtime.GOOS {
	case "linux":
		f, err := os.Open("/proc/meminfo")
		if err != nil {
			return 0
		}
		defer f.Close()
		sc := bufio.NewScanner(f)
		for sc.Scan() {
			var kb int64
			if _, err := fmt.Sscanf(sc.Text(), "MemTotal: %d kB", &kb); err == nil {
				return int(math.Ceil(float64(kb)/(1<<20)/8) * 8)
			}
		}
	case "darwin":
		if out, err := exec.Command("sysctl", "-n", "hw.memsize").Output(); err == nil {
			if b, err := strconv.ParseInt(strings.TrimSpace(string(out)), 10, 64); err == nil {
				return int(b >> 30)
			}
		}
	}
	return 0
}

// tenstorrentBoards maps board types (tt-smi board_type, sysfs tt_card_type) to LocalMaxxing names and memory.
var tenstorrentBoards = map[string]struct {
	name string
	gb   float64
}{
	"p150": {"Tenstorrent P150", 32}, "p150a": {"Tenstorrent P150", 32}, "p150b": {"Tenstorrent P150", 32}, "p150c": {"Tenstorrent P150", 32},
	"p100": {"Tenstorrent P100", 28}, "p100a": {"Tenstorrent P100", 28}, "p300": {"Tenstorrent P300", 64},
	"n150": {"Tenstorrent N150", 12}, "n300": {"Tenstorrent N300", 24},
}

// applyTenstorrentBoards describes the cards of the most common board type; false when none is known.
func applyTenstorrentBoards(base map[string]any, boardTypes []string) bool {
	counts := map[string]int{}
	gb := map[string]float64{}
	for _, t := range boardTypes {
		b, ok := tenstorrentBoards[strings.ToLower(strings.TrimSpace(t))]
		if !ok {
			continue
		}
		counts[b.name]++
		gb[b.name] = b.gb
	}
	best := ""
	for name, n := range counts {
		if best == "" || n > counts[best] || (n == counts[best] && name < best) {
			best = name
		}
	}
	if best == "" {
		return false
	}
	base["hwClass"] = "DISCRETE_GPU"
	base["gpuName"] = best
	base["gpuCount"] = counts[best]
	base["vramGb"] = gb[best]
	base["totalVramGb"] = gb[best] * float64(counts[best])
	return true
}

// sysfsTenstorrentBoards reads the tenstorrent driver's card types (no tt-smi needed).
func sysfsTenstorrentBoards() []string {
	paths, _ := filepath.Glob("/sys/class/tenstorrent/*/tt_card_type")
	sort.Strings(paths)
	var out []string
	for _, p := range paths {
		if data, err := os.ReadFile(p); err == nil {
			out = append(out, strings.TrimSpace(string(data)))
		}
	}
	return out
}

// hostAcceleratorBackends lists the accelerator stacks present: cuda, rocm, tt-metal.
func hostAcceleratorBackends() []string {
	var out []string
	if _, err := exec.LookPath("nvidia-smi"); err == nil {
		if o, err := exec.Command("nvidia-smi", "-L").Output(); err == nil && strings.TrimSpace(string(o)) != "" {
			out = append(out, "cuda")
		}
	}
	if _, err := exec.LookPath("rocm-smi"); err == nil {
		if o, err := exec.Command("rocm-smi", "--showproductname").Output(); err == nil && strings.Contains(strings.ToLower(string(o)), "card") {
			out = append(out, "rocm")
		}
	}
	if len(sysfsTenstorrentBoards()) > 0 {
		out = append(out, "tt-metal")
	}
	return out
}

// isLoopbackURL reports whether rawURL points at this machine.
func isLoopbackURL(rawURL string) bool {
	u, err := url.Parse(rawURL)
	if err != nil || u.Hostname() == "" {
		return false
	}
	host := u.Hostname()
	if host == "localhost" {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}

// resolveEvalHardware returns the hardware object an eval submission records: --hardware, else the
// hardware.json saved in runDir (resumed or deferred runs), else this host's hardware when the endpoint is
// local. Detection refuses to guess between accelerator stacks (pass --backend) and never describes a
// remote endpoint's machine. A detected or explicit description is saved to runDir for later submits.
func resolveEvalHardware(args cliArgs, baseURL, runDir string) (any, error) {
	saved := ""
	if runDir != "" {
		saved = filepath.Join(runDir, "hardware.json")
	}
	save := func(hw any) {
		if saved == "" {
			return
		}
		if _, err := os.Stat(saved); err == nil {
			return
		}
		if err := os.MkdirAll(runDir, 0o755); err == nil {
			_ = writeJSON(saved, hw)
		}
	}
	if path := opt(args, "hardware"); path != "" {
		hw, err := readJSON(path)
		if err == nil {
			save(hw)
		}
		return hw, err
	}
	if saved != "" {
		if hw, err := readJSON(saved); err == nil {
			printStatus(args, "hardware_reused", map[string]any{"path": saved})
			return hw, nil
		}
	}
	if baseURL == "" || !isLoopbackURL(baseURL) {
		hint := "This endpoint runs on another machine: run lmx hardware --out hardware.json there and pass --hardware hardware.json."
		if baseURL == "" {
			hint = "Pass the --hardware file of the machine that ran the model (lmx hardware --out hardware.json on it)."
		}
		return nil, cliError{"missing_hardware", "This eval needs the hardware the model ran on.", []string{hint}, map[string]any{"baseUrl": baseURL}}
	}
	backend := opt(args, "backend")
	if backends := hostAcceleratorBackends(); backend == "" && len(backends) > 1 {
		return nil, cliError{"ambiguous_hardware", "This machine has more than one accelerator stack (" + strings.Join(backends, ", ") + "); lmx cannot tell which one serves the endpoint.",
			[]string{"Pass --backend " + strings.Join(backends, " or --backend ") + ".", "Or pass --hardware hardware.json (lmx hardware --backend <b> --out hardware.json)."}, map[string]any{"backends": backends}}
	}
	hw := detectHardware(args)
	printStatus(args, "hardware_detected", map[string]any{"gpuName": hw["gpuName"], "gpuCount": hw["gpuCount"], "vramGb": hw["vramGb"],
		"cpu": hw["cpu"], "ramGb": hw["ramGb"], "os": hw["os"], "savedTo": saved})
	save(hw)
	return hw, nil
}

// ---- datasets ----

// evalDataset is one approved LocalMaxxing dataset.
type evalDataset struct {
	Slug       string
	Name       string
	Category   string
	ShardCount int
	ItemCount  int
}

// terminalDatasetCategories run through eval terminal (task containers and an agent); the rest are
// question shards for eval shard.
var terminalDatasetCategories = map[string]bool{"agentic": true, "software-engineering": true}

func (d evalDataset) terminal() bool {
	return terminalDatasetCategories[strings.ToLower(d.Category)] || strings.HasPrefix(d.Slug, "terminal-bench") || strings.Contains(d.Slug, "crud")
}

func (d evalDataset) runner() string {
	if d.terminal() {
		return "terminal"
	}
	return "shard"
}

func fetchEvalDatasets(args cliArgs) ([]evalDataset, error) {
	value, err := fetchJSON("GET", apiURL(args)+"/api/benchmarks/datasets", "", nil)
	if err != nil {
		return nil, err
	}
	var out []evalDataset
	for _, item := range anySlice(asObject(value)["datasets"]) {
		obj := asObject(item)
		out = append(out, evalDataset{Slug: stringValue(obj["slug"]), Name: stringValue(obj["name"]), Category: stringValue(obj["category"]),
			ShardCount: int(numberField(obj, "shardCount")), ItemCount: int(numberField(obj, "itemCount"))})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Slug < out[j].Slug })
	return out, nil
}

// datasetBySlug finds slug among the approved datasets (ok false when it is not one, e.g. a suite).
func datasetBySlug(args cliArgs, slug string) (evalDataset, bool, error) {
	ds, err := fetchEvalDatasets(args)
	if err != nil {
		return evalDataset{}, false, err
	}
	for _, d := range ds {
		if d.Slug == slug {
			return d, true, nil
		}
	}
	return evalDataset{}, false, nil
}

// withPositional returns args whose positionals are replaced (to reuse a handler under another command path).
func withPositional(args cliArgs, positional ...string) cliArgs {
	args.positional = positional
	return args
}

// runEvalDataset runs an approved dataset with the runner it needs: eval terminal run for agentic task
// containers, eval shard for question sets.
func runEvalDataset(d evalDataset, args cliArgs) error {
	printStatus(args, "eval_dataset_runner", map[string]any{"dataset": d.Slug, "runner": d.runner(), "category": d.Category})
	if d.terminal() {
		return runTerminalEval(withPositional(args, "eval", "terminal", "run", d.Slug), false)
	}
	return handleEvalShard(d.Slug, args)
}

// defaultTerminalRunDir is where a terminal run without --run-dir keeps its durable checkpoint, so every run
// can be resumed, inspected (eval runs) and submitted later.
func defaultTerminalRunDir(args cliArgs, dataset string) string {
	model := firstNonEmpty(opt(args, "model"), opt(args, "served-model"), "model")
	slug := strings.Trim(regexp.MustCompile(`[^A-Za-z0-9._-]+`).ReplaceAllString(model, "-"), "-")
	return filepath.Join(firstNonEmpty(opt(args, "runs-dir"), "runs"), "evals", dataset, slug+"-"+time.Now().Format("20060102-150405"))
}

// writeEvalRunMeta records what a terminal run measured, for eval runs and later submits.
func writeEvalRunMeta(runDir string, meta map[string]any) {
	if runDir == "" {
		return
	}
	path := filepath.Join(runDir, "lmx-run.json")
	if _, err := os.Stat(path); err == nil {
		return // a resumed run keeps its original record
	}
	meta["startedAt"] = time.Now().UTC().Format(time.RFC3339)
	if err := os.MkdirAll(runDir, 0o755); err == nil {
		_ = writeJSON(path, meta)
	}
}

// ---- eval coverage ----

// handleEvalCoverage shows, for one model and quantization, how much of every approved dataset LocalMaxxing
// already has and the command that runs the next missing shard.
func handleEvalCoverage(args cliArgs) error {
	model := opt(args, "model")
	if model == "" {
		return cliError{"missing_model", "eval coverage requires --model <HuggingFace model id>.", []string{"Pass the id you submit evals under, plus --quantization and --quant-format when the model is quantized."}, nil}
	}
	ds, err := fetchEvalDatasets(args)
	if err != nil {
		return err
	}
	var rows []map[string]any
	for _, d := range ds {
		harness := evalShardHarnessKeyForCoverage(args)
		if d.terminal() {
			harness = terminalHarnessKey(args, terminalConfig{shellMode: "persistent"})
		}
		row := map[string]any{"dataset": d.Slug, "name": d.Name, "runner": d.runner(), "shardCount": d.ShardCount, "items": d.ItemCount}
		q := url.Values{}
		q.Set("hfId", model)
		q.Set("harnessKey", harness)
		if v := opt(args, "quantization"); v != "" {
			q.Set("quantization", v)
		}
		if v := opt(args, "quant-format"); v != "" {
			q.Set("quantFormat", v)
		}
		value, err := fetchJSON("GET", apiURL(args)+"/api/benchmarks/"+url.PathEscape(d.Slug)+"/coverage?"+q.Encode(), apiKey(args), nil)
		if err != nil {
			row["error"] = err.Error()
			rows = append(rows, row)
			continue
		}
		shardCount, covered, missing := shardCoverageDetails(value)
		if shardCount > 0 {
			row["shardCount"] = shardCount
		}
		row["coveredShards"] = sortedIntsFromSet(covered)
		row["missingShards"] = missing
		if agg := asObject(asObject(value)["aggregate"]); agg != nil {
			row["score"] = agg["pooledScore"]
		}
		if len(missing) > 0 {
			cmd := "lmx eval run " + d.Slug + " --base-url <url> --model " + model
			if v := opt(args, "quantization"); v != "" {
				cmd += " --quantization " + v
			}
			if v := opt(args, "quant-format"); v != "" {
				cmd += " --quant-format " + v
			}
			row["next"] = cmd + " --shard " + strconv.Itoa(missing[0]) + " --submit"
		}
		rows = append(rows, row)
	}
	result := map[string]any{"model": model, "quantization": opt(args, "quantization"), "quantFormat": opt(args, "quant-format"), "datasets": rows}
	if !humanOutput(args) || opt(args, "out") != "" {
		return writeOrPrintJSON("eval_coverage", args, result)
	}
	fmt.Printf("Coverage for %s %s\n\n", model, firstNonEmpty(opt(args, "quantization"), "(no quantization)"))
	fmt.Printf("%-28s %-9s %-9s %-8s %s\n", "DATASET", "RUNNER", "COVERED", "SCORE", "NEXT")
	for _, r := range rows {
		covered := fmt.Sprintf("%d/%v", len(anyInts(r["coveredShards"])), r["shardCount"])
		score := "-"
		if s, ok := r["score"].(float64); ok {
			score = fmt.Sprintf("%.1f%%", s*100)
		}
		next := firstNonEmpty(stringValue(r["next"]), "complete")
		if e := stringValue(r["error"]); e != "" {
			next, covered = "error: "+e, "?"
		}
		fmt.Printf("%-28s %-9s %-9s %-8s %s\n", r["dataset"], r["runner"], covered, score, next)
	}
	return nil
}

func anyInts(v any) []int {
	switch x := v.(type) {
	case []int:
		return x
	case []any:
		out := make([]int, 0, len(x))
		for _, i := range x {
			if f, ok := i.(float64); ok {
				out = append(out, int(f))
			}
		}
		return out
	}
	return nil
}

// ---- eval runs ----

// handleEvalRuns lists terminal eval runs kept under <runs-dir>/evals (newest first).
func handleEvalRuns(args cliArgs) error {
	root := filepath.Join(firstNonEmpty(opt(args, "runs-dir"), "runs"), "evals")
	dirs, _ := filepath.Glob(filepath.Join(root, "*", "*"))
	var rows []map[string]any
	for _, dir := range dirs {
		snap, err := terminalStatusSnapshot(dir)
		if err != nil {
			continue
		}
		row := map[string]any{"runDir": dir, "dataset": snap["dataset"], "shardIndex": snap["shardIndex"], "state": snap["state"],
			"tasksTotal": snap["tasksTotal"], "tasksScored": snap["tasksScored"], "tasksPassed": snap["tasksPassed"], "updatedAt": snap["updatedAt"]}
		if meta, err := readJSON(filepath.Join(dir, "lmx-run.json")); err == nil {
			m := asObject(meta)
			for _, k := range []string{"model", "quantization", "quantFormat", "baseUrl", "startedAt"} {
				row[k] = m[k]
			}
		}
		if res, err := readJSON(filepath.Join(dir, "result.json")); err == nil {
			summary := asObject(asObject(res)["summary"])
			row["accuracyPct"] = summary["accuracyPct"]
			row["submissionId"] = summary["runId"]
			row["submissionStatus"] = summary["status"]
		}
		rows = append(rows, row)
	}
	sort.Slice(rows, func(i, j int) bool { return stringValue(rows[i]["updatedAt"]) > stringValue(rows[j]["updatedAt"]) })
	if n, err := intOption(args, 0, 0, "limit"); err == nil && n > 0 && len(rows) > n {
		rows = rows[:n]
	}
	if !humanOutput(args) || opt(args, "out") != "" {
		return writeOrPrintJSON("eval_runs", args, map[string]any{"runsDir": root, "runs": rows})
	}
	if len(rows) == 0 {
		fmt.Printf("No eval runs under %s (lmx eval run <dataset> keeps them there).\n", root)
		return nil
	}
	fmt.Printf("%-20s %-26s %-5s %-34s %-12s %-9s %-8s %s\n", "UPDATED", "DATASET", "SHARD", "MODEL", "STATE", "PASSED", "SUBMIT", "RUN DIR")
	for _, r := range rows {
		passed := fmt.Sprintf("%v/%v", r["tasksPassed"], r["tasksTotal"])
		sub := firstNonEmpty(stringValue(r["submissionStatus"]), "-")
		fmt.Printf("%-20s %-26v %-5v %-34s %-12v %-9s %-8s %s\n", strings.Replace(stringValue(r["updatedAt"]), "T", " ", 1),
			r["dataset"], r["shardIndex"], firstNonEmpty(stringValue(r["model"]), "?"), r["state"], passed, sub, r["runDir"])
	}
	return nil
}

// ---- deferred submit of a recorded terminal run ----

// replayExcludedOptions never go into a run record: secrets, and options that only steer this invocation.
var replayExcludedOptions = map[string]bool{"api-key": true, "model-api-key": true, "run-dir": true, "status-log": true, "out": true,
	"resume": true, "runs-dir": true}

// replayExcludedFlags only steer this invocation (output, submission, detaching).
var replayExcludedFlags = map[string]bool{"submit": true, "dry-run": true, "detach": true, "detached-child": true, "json": true,
	"json-status": true, "quiet": true, "print": true, "verbose": true, "help": true}

// replayableTerminalOptions are the options and flags that define a terminal run (its checkpoint fingerprint).
func replayableTerminalOptions(args cliArgs) map[string]any {
	opts := map[string]string{}
	for k, v := range args.opts {
		if !replayExcludedOptions[k] {
			opts[k] = v
		}
	}
	var flags []string
	for k, on := range args.flags {
		if on && !replayExcludedFlags[k] {
			flags = append(flags, k)
		}
	}
	sort.Strings(flags)
	return map[string]any{"opts": opts, "flags": flags}
}

// submitRecordedTerminalRun publishes a completed eval terminal run directory: it resumes the run with its
// recorded options plus --submit, so every scored task is reused (nothing reruns, the endpoint need not be
// up) and the checkpoint fingerprint matches.
func submitRecordedTerminalRun(args cliArgs, root string) error {
	recValue, err := readJSON(filepath.Join(root, "lmx-run.json"))
	if err != nil {
		return err
	}
	rec := asObject(recValue)
	state, err := loadTerminalLiveCheckpointFile(root)
	if err != nil {
		return cliError{"checkpoint_missing", "This run directory has no terminal checkpoint (run.json).", []string{"Pass a directory written by lmx eval run / eval terminal run."}, map[string]any{"runDir": root}}
	}
	switch state.State {
	case "completed":
	case "submitted":
		return cliError{"checkpoint_already_submitted", "This run was already submitted.", []string{"Read result.json in the run directory for the receipt."}, map[string]any{"runDir": root}}
	default:
		return cliError{"run_not_completed", "This run has not finished (" + state.State + ").", []string{"Resume it first: lmx eval terminal submit only publishes completed runs.",
			"lmx eval run " + state.Dataset + " --run-dir " + root + " (with the original options) continues where it stopped."}, map[string]any{"runDir": root, "state": state.State}}
	}
	options := asObject(rec["options"])
	replay := cliArgs{positional: []string{"eval", "terminal", "run", state.Dataset}, opts: map[string]string{}, flags: map[string]bool{},
		provided: map[string]bool{}, jsonPrinted: args.jsonPrinted, jsonFallback: args.jsonFallback}
	for k, v := range asObject(options["opts"]) {
		replay.opts[k] = stringValue(v)
	}
	for _, f := range anySlice(options["flags"]) {
		replay.flags[stringValue(f)] = true
	}
	set := func(k, v string) {
		if v != "" && replay.opts[k] == "" {
			replay.opts[k] = v
		}
	}
	set("model", stringValue(rec["model"]))
	set("served-model", stringValue(rec["servedModel"]))
	set("quantization", stringValue(rec["quantization"]))
	set("quant-format", stringValue(rec["quantFormat"]))
	set("thinking-level", stringValue(rec["thinkingLevel"]))
	replay.opts["shard"] = strconv.Itoa(state.ShardIndex)
	replay.opts["run-dir"] = root
	replay.opts["resume"] = "auto"
	// hardware/backend describe a run that could not record its hardware (remote or ambiguous host).
	for _, k := range []string{"api-key", "api-url", "model-api-key", "out", "hardware", "backend"} {
		if v := opt(args, k); v != "" {
			replay.opts[k] = v
		}
	}
	for _, f := range []string{"json", "json-status", "quiet", "dry-run"} {
		if hasFlag(args, f) {
			replay.flags[f] = true
		}
	}
	replay.flags["submit"] = true
	for k := range replay.opts {
		replay.provided[k] = true
	}
	for k := range replay.flags {
		replay.provided[k] = true
	}
	printStatus(args, "terminal_recorded_run_submit", map[string]any{"runDir": root, "dataset": state.Dataset, "shardIndex": state.ShardIndex, "model": rec["model"]})
	return runTerminalEval(replay, false)
}

// nullFingerprintField is the field hash of a JSON null (a value the run did not have).
var nullFingerprintField = func() string {
	_, fields, _ := terminalLiveCheckpointFingerprint(map[string]any{"v": nil})
	return fields["v"]
}()

// terminalHardwareAddedOnly reports whether actual differs from expected only by a hardware description
// the checkpoint recorded as null.
func terminalHardwareAddedOnly(expected, actual map[string]string) bool {
	if len(expected) == 0 || expected["hardware"] != nullFingerprintField || actual["hardware"] == nullFingerprintField {
		return false
	}
	changed := terminalChangedFingerprintFields(expected, actual)
	return len(changed) == 1 && changed[0] == "hardware"
}
